"""Offline re-solve of calibrate_wrist.py from its saved frames (+ per-pose q json files, or
the q values passed on the command line for runs that predate them)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Kinematics  # noqa: E402
from calibrate_top import CALIB_DIR, apply, fit_plane, kabsch, rot_a_to_b  # noqa: E402
from calibrate_wrist import T_mount, lift_landmarks, solve_hand_eye  # noqa: E402
from cameras import Frame  # noqa: E402
from perception import Detector  # noqa: E402


@dataclass
class Args:
    side: str = "left"
    landmarks: List[str] = field(default_factory=lambda: ["white cup", "lemon", "blue cube", "red cube", "carrot"])
    qs: List[str] = field(default_factory=list)
    """Per-pose joint vectors as comma-separated strings (6 or 7 values), in pose order."""
    write: bool = False


def main(args: Args) -> None:
    debug = CALIB_DIR / "debug"
    det = Detector()
    kin = Kinematics()
    top = Frame.load(debug, "wristcal_top")
    top_objs = lift_landmarks(top, det, args.landmarks)
    print("[top]", {n: np.round(o.center_base, 3).tolist() for n, o in top_objs.items()})
    mounts, obs, pose_ids = [], [], []
    for i in range(20):
        if not (debug / f"wristcal_{i:02d}_color.png").exists():
            break
        qfile = debug / f"wristcal_{i:02d}_q.json"
        if qfile.exists():
            q = np.array(json.loads(qfile.read_text()))
        elif i < len(args.qs):
            q = np.array([float(v) for v in args.qs[i].split(",")])
        else:
            print(f"pose {i}: no q, skipped")
            continue
        fr = Frame.load(debug, f"wristcal_{i:02d}")
        objs = {n: o for n, o in lift_landmarks(fr, det, args.landmarks).items() if n in top_objs}
        print(f"[pose {i}] sees {sorted(objs)}: " + ", ".join(f"{n} {np.round(o.center_base, 3).tolist()}" for n, o in objs.items()))
        if len(objs) < 3:
            continue
        mounts.append(kin.mount(q))
        obs.append({n: o.center_base for n, o in objs.items()})
        pose_ids.append(i)
    best = None
    for roll in np.radians(np.arange(0, 360, 45)):
        cr, sr = np.cos(roll), np.sin(roll)
        R0 = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.diag([1.0, -1.0, -1.0])
        X0 = np.eye(4)
        X0[:3, :3] = R0
        X0[:3, 3] = [0.0, 0.0, -0.02]
        X, L, rms = solve_hand_eye(mounts, obs, X0)
        print(f"   roll {np.degrees(roll):3.0f}: rms {rms * 1e3:.1f} mm")
        if best is None or rms < best[0]:
            best = (rms, X, L)
    rms, X, L = best
    print(f"[wrist] cam at {np.round(X[:3, 3], 3)} (mount frame); cam axes in mount: x {np.round(X[:3, 0], 2)} y {np.round(X[:3, 1], 2)} z {np.round(X[:3, 2], 2)}; rms {rms * 1e3:.1f} mm")
    for n in sorted(L):
        print(f"   {n:12s} base {np.round(L[n], 3)}")
    # table plane in the base frame from every wrist pose (FK * X * plane), averaged: the
    # landmarks are nearly coplanar and only 20 cm apart, so the top camera's tilt would be
    # poorly determined from them alone -- the plane normal pins it down instead.
    normals, points = [], []
    for m, i in zip(mounts, pose_ids, strict=True):
        fr = Frame.load(debug, f"wristcal_{i:02d}")
        pts_w, _ = fr.point_cloud((fr.depth > 0.15) & (fr.depth < 1.2))
        n_w, d_w = fit_plane(pts_w[:: max(1, len(pts_w) // 50000)])
        Tc = T_mount(m) @ X
        normals.append(Tc[:3, :3] @ n_w)
        points.append(apply(Tc, -d_w * n_w)[0])
    n_w_mean = np.median(np.stack(normals), axis=0)
    n_w_mean /= np.linalg.norm(n_w_mean)
    spread = np.degrees(np.arccos(np.clip([n @ n_w_mean for n in normals], -1, 1)))
    d_base_w = float(-n_w_mean @ np.median(np.stack(points), axis=0))
    print(f"[wrist] table normal (base) {np.round(n_w_mean, 3)} (per-pose spread {np.round(spread, 1)} deg), z under origin {-d_base_w / n_w_mean[2]:+.4f}")
    # Robust wrist-derived normal: drop poses whose plane disagrees with the median by > 6 deg
    # (oblique views sometimes fit the box or the wall), then average the rest. The data says
    # the table is tilted a few degrees from the base z axis, so no "table = z" prior here.
    good = [n for n, sp in zip(normals, spread, strict=True) if sp < 6.0]
    n_base = np.mean(good, axis=0)
    n_base /= np.linalg.norm(n_base)
    print(f"[wrist] using {len(good)}/{len(normals)} poses for the table normal: {np.round(n_base, 3)}")

    names = [n for n in sorted(L) if n in top_objs]
    A = np.stack([top_objs[n].center_base for n in names])
    B = np.stack([L[n] for n in names])
    pts, _ = top.point_cloud((top.depth > 0.3) & (top.depth < 1.5))
    n_cam, d_cam = fit_plane(pts[:: max(1, len(pts) // 60000)])
    # rotation: align the top camera's table normal with n_base, then yaw about n_base +
    # translation from the landmarks (a 4-DOF Kabsch in the plane's frame)
    R_align = rot_a_to_b(n_cam, n_base)
    R_up = rot_a_to_b(n_base, np.array([0.0, 0.0, 1.0]))  # plane frame: normal -> z
    A2 = (R_up @ (R_align @ A.T)).T
    B2 = (R_up @ B.T).T
    ca, cb = A2.mean(0), B2.mean(0)
    H = (A2 - ca)[:, :2].T @ (B2 - cb)[:, :2]
    U, _, Vt = np.linalg.svd(H)
    Dm = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R2 = Vt.T @ Dm @ U.T
    Rz = np.eye(3)
    Rz[:2, :2] = R2
    R = R_up.T @ Rz @ R_up @ R_align
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = B.mean(0) - R @ A.mean(0)
    res = np.linalg.norm(apply(T, A) - B, axis=1)
    print(f"[top] residuals {dict(zip(names, np.round(res * 1e3, 1).tolist()))} mm; camera at {np.round(T[:3, 3], 3)}, optical axis {np.round(T[:3, 2], 2)}")
    n_chk = T[:3, :3] @ n_cam
    d_base = float(-n_chk @ apply(T, -d_cam * n_cam)[0])
    print(f"[top] table normal (base) {np.round(n_chk, 3)}, z under origin {-d_base / n_chk[2]:+.4f} (wrist-derived {-d_base_w / n_base[2]:+.4f})")
    n_base = n_chk
    # overlay: landmarks projected into the top image
    import cv2

    img = top.color.copy()
    Tinv = np.linalg.inv(T)
    for n in names:
        uv = top.project(apply(Tinv, L[n]))[0]
        cv2.circle(img, (int(uv[0]), int(uv[1])), 8, (0, 0, 255), 2)
        cv2.putText(img, n, (int(uv[0]) + 8, int(uv[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(debug / "landmarks_top.png"), img)
    if args.write:
        (CALIB_DIR / f"wrist_{args.side}.json").write_text(
            json.dumps({"side": args.side, "T_mount_cam": X.tolist(), "rms_m": rms, "landmarks_base": {n: L[n].tolist() for n in L}}, indent=2)
        )
        (CALIB_DIR / f"top_{args.side}.json").write_text(
            json.dumps(
                {
                    "side": args.side,
                    "method": "landmarks",
                    "T_base_cam": T.tolist(),
                    "table_normal_base": n_base.tolist(),
                    "table_d_base": d_base,
                    "table_normal_cam": n_cam.tolist(),
                    "table_d_cam": d_cam,
                    "landmark_residuals_m": res.tolist(),
                    "K": top.K.tolist(),
                },
                indent=2,
            )
        )
        print("[calib] written")


if __name__ == "__main__":
    main(tyro.cli(Args))
