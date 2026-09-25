"""Re-fit the top-camera hand-eye transform offline from the frames calibrate_top.py saved.

Multi-start trimmed ICP: the initial rotation levels the camera's table normal onto base +z,
yaw is scanned over 360 deg, translation comes from the per-pose centroids; every start is run
through ICP and scored by the *trimmed mean* nearest-neighbour distance over all observed
points (not just inliers), which is what the first version got wrong.

    python scripts/box_packing/refit_top.py --side left
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tyro
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Kinematics  # noqa: E402
from calibrate_top import CALIB_DIR, ArmSurface, apply, arm_mask, kabsch, rot_a_to_b  # noqa: E402
from cameras import Frame  # noqa: E402


@dataclass
class Args:
    side: str = "left"
    debug_dir: str = str(CALIB_DIR / "debug")
    write: bool = True
    """Overwrite calib/top_<side>.json with the refit."""


def voxel_subsample(pts: np.ndarray, voxel: float, rng: np.random.Generator) -> np.ndarray:
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    out = pts[first]
    return out


def score(T: np.ndarray, obs: List[np.ndarray], trees: List[cKDTree], trim: float = 0.8) -> Tuple[float, float]:
    """(trimmed mean NN distance in m, fraction of observed points within 1.5 cm)."""
    dists = np.concatenate([tree.query(apply(T, o), workers=-1)[0] for o, tree in zip(obs, trees, strict=True)])
    k = int(len(dists) * trim)
    return float(np.sort(dists)[:k].mean()), float((dists < 0.015).mean())


def icp(obs: List[np.ndarray], model: List[np.ndarray], trees: List[cKDTree], T0: np.ndarray, iters: int = 80) -> np.ndarray:
    T = T0.copy()
    for it in range(iters):
        src, dst = [], []
        for o, mm, tree in zip(obs, model, trees, strict=True):
            p = apply(T, o)
            dist, idx = tree.query(p, workers=-1)
            # trimmed: keep the best 70 % of matches each iteration (robust to cable/shadow points)
            cut = np.percentile(dist, 70)
            keep = dist <= cut
            src.append(o[keep])
            dst.append(mm[idx[keep]])
        A, B = np.concatenate(src), np.concatenate(dst)
        T_new = kabsch(A, B)
        if np.linalg.norm(T_new[:3, 3] - T[:3, 3]) < 1e-5 and np.linalg.norm(T_new[:3, :3] - T[:3, :3]) < 1e-5:
            T = T_new
            break
        T = T_new
    return T


def main(args: Args) -> None:
    debug = Path(args.debug_dir)
    calib = json.loads((CALIB_DIR / f"top_{args.side}.json").read_text())
    n_cam, d_cam = np.array(calib["table_normal_cam"]), float(calib["table_d_cam"])
    qs = [np.array(q) for q in calib["poses_q"]]
    kin = Kinematics()
    surf = ArmSurface(kin)
    bg = Frame.load(debug, "background")
    rng = np.random.default_rng(0)

    # which saved pose frames correspond to the stored q's: those with a usable arm mask
    obs, model = [], []
    frames = sorted(debug.glob("pose_*_color.png"))
    used = []
    for f in frames:
        i = int(f.stem.split("_")[1])
        fr = Frame.load(debug, f"pose_{i:02d}")
        mask = arm_mask(fr, bg.depth)
        pts, _ = fr.point_cloud(mask)
        if len(pts) < 2000:
            continue
        used.append(i)
    assert len(used) == len(qs), f"{len(used)} usable frames vs {len(qs)} stored poses"
    for i, q in zip(used, qs, strict=True):
        fr = Frame.load(debug, f"pose_{i:02d}")
        mask = arm_mask(fr, bg.depth)
        pts, _ = fr.point_cloud(mask)
        obs.append(voxel_subsample(pts, 0.01, rng))
        model.append(voxel_subsample(surf.points(q, per_geom=6000, seed=i), 0.008, rng))
    trees = [cKDTree(mm) for mm in model]
    print(f"[refit] {len(obs)} poses, {sum(len(o) for o in obs)} observed / {sum(len(m) for m in model)} model points")

    T_old = np.array(calib["T_base_cam"])
    print(f"[refit] stored T: trimmed NN {score(T_old, obs, trees)[0] * 1e3:.1f} mm, within 1.5cm {score(T_old, obs, trees)[1]:.2f}")

    R_level = rot_a_to_b(n_cam, np.array([0.0, 0.0, 1.0]))
    co_all = np.concatenate(obs)
    cm_all = np.concatenate(model)
    best = None
    for yaw in np.radians(np.arange(0, 360, 30)):
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        T0 = np.eye(4)
        T0[:3, :3] = Rz @ R_level
        T0[:3, 3] = cm_all.mean(0) - T0[:3, :3] @ co_all.mean(0)
        T = icp(obs, model, trees, T0)
        s, cov = score(T, obs, trees)
        print(f"  start yaw {np.degrees(yaw):4.0f}: trimmed NN {s * 1e3:6.1f} mm, within 1.5 cm {cov:.2f}")
        if best is None or s < best[0]:
            best = (s, cov, T)
    s, cov, T = best
    print(f"[refit] best: trimmed NN {s * 1e3:.1f} mm, coverage {cov:.2f}")
    n_base = T[:3, :3] @ n_cam
    p_base = apply(T, -d_cam * n_cam)[0]
    d_base = float(-n_base @ p_base)
    print(f"[refit] table normal (base) {np.round(n_base, 3)}, z under origin {-d_base / n_base[2]:+.4f}; camera at {np.round(T[:3, 3], 3)}")
    print(f"[refit] optical axis in base {np.round(T[:3, 2], 3)}")

    # overlays for a visual check
    Tinv = np.linalg.inv(T)
    tiles = []
    for i, q in list(zip(used, qs, strict=True))[:4]:
        fr = Frame.load(debug, f"pose_{i:02d}")
        img = fr.color.copy()
        uv = fr.project(apply(Tinv, model[used.index(i)]))
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                img[int(v), int(u)] = (0, 255, 0)
        g = fr.project(apply(Tinv, kin.grasp(q)[0]))[0]
        cv2.circle(img, (int(g[0]), int(g[1])), 10, (0, 0, 255), 3)
        tiles.append(cv2.resize(img, (640, 360)))
    cv2.imwrite(str(debug / "refit_montage.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))

    if args.write:
        calib.update(
            T_base_cam=T.tolist(),
            table_normal_base=n_base.tolist(),
            table_d_base=d_base,
            refit={"trimmed_nn_m": s, "coverage_1p5cm": cov},
        )
        (CALIB_DIR / f"top_{args.side}.json").write_text(json.dumps(calib, indent=2))
        print("[refit] written")


if __name__ == "__main__":
    main(tyro.cli(Args))
