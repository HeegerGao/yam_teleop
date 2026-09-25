"""Hand-eye calibration of the third-person (top) camera against one arm's base frame.

Fully automatic and touch-free: the arm visits a handful of poses **high above the table**
(every link stays above ``--min-link-z``, higher than anything on the table) with the fingers
pointing forward-down, so no object can be hit no matter how the table is laid out. At each
pose the arm is segmented out of the top camera's depth by comparison with a background frame
taken at the start. The rigid transform camera->base is then fitted by ICP between the
observed arm point clouds (camera frame) and the arm's own MJCF mesh surface posed by FK
(base frame), jointly over all poses -- no marker, no fingertip heuristic.

Output: scripts/box_packing/calib/top_<side>.json with T_base_cam (4x4), the table plane in
the base frame, and the per-pose ICP residuals. Debug overlays land in calib/debug/.

    python scripts/box_packing/calibrate_top.py --side left --no-execute   # print the plan only
    python scripts/box_packing/calibrate_top.py --side left
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mujoco
import numpy as np
import tyro
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from arm import Arm, Kinematics, Workspace  # noqa: E402
from cameras import CameraRig, Frame  # noqa: E402

CALIB_DIR = _HERE / "calib"
PORTS = {"left": 1235, "right": 1234}


@dataclass
class Args:
    side: str = "left"
    execute: bool = True
    xs: Tuple[float, ...] = (0.30, 0.40, 0.50)
    ys: Tuple[float, ...] = (-0.14, 0.10)
    zs: Tuple[float, ...] = (0.27, 0.35)
    """Grasp-point grid (base frame). All well above the tallest object on the table."""
    tilt_deg: float = 70.0
    """Finger direction: this far from vertical, tilted away from the base (fingers forward)."""
    min_link_z: float = 0.17
    """Every link of every pose, and every joint-space transition between them, must stay
    above this base-frame height (the table is at ~0 and the tallest object ~0.12 m)."""
    max_joint_vel: float = 0.5
    out: Optional[str] = None
    debug_dir: str = str(CALIB_DIR / "debug")


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def fit_plane(pts: np.ndarray, thresh: float = 0.008, iters: int = 400, seed: int = 0) -> Tuple[np.ndarray, float]:
    """RANSAC + SVD plane through ``pts`` (Nx3). Returns (unit normal, d) with n.p + d = 0; the
    normal points towards the camera side of the plane (up, for a table seen from above)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ p[0]
        count = int((np.abs(pts @ n + d) < thresh).sum())
        if best is None or count > best[0]:
            best = (count, n, d)
    assert best is not None
    _, n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]
    cen = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
    n = vt[2]
    if n @ cen > 0:  # the camera (origin) must be on the positive side
        n = -n
    return n, float(-n @ cen)


def table_plane_cam(frame: Frame) -> Tuple[np.ndarray, float]:
    pts, _ = frame.point_cloud((frame.depth > 0.3) & (frame.depth < 1.5))
    return fit_plane(pts[:: max(1, len(pts) // 60000)])


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """4x4 T with B ~= T A (A, B: Nx3)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ np.atleast_2d(pts).T).T + T[:3, 3]


# ---------------------------------------------------------------------------
# arm model surface
# ---------------------------------------------------------------------------


class ArmSurface:
    """Mesh vertices of every arm/gripper geom, posed by FK -- the ICP target."""

    def __init__(self, kin: Kinematics, skip_bodies: Tuple[str, ...] = ("base", "link1")) -> None:
        self.kin = kin
        m = kin.model
        self.geoms: List[Tuple[int, np.ndarray]] = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if m.body(m.geom_bodyid[g]).name in skip_bodies:
                continue
            mesh = m.geom_dataid[g]
            adr, num = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.geoms.append((g, m.mesh_vert[adr : adr + num].copy()))

    def points(self, q: np.ndarray, per_geom: int = 1500, seed: int = 0) -> np.ndarray:
        """Base-frame surface points at joint config ``q`` (gripper joints at their command)."""
        m, d = self.kin.model, self.kin.data
        d.qpos[:] = 0.0
        d.qpos[:6] = q[:6]
        # linear_4310: slide joints 0 (open) .. 0.0475 (closed), command 1 = open
        if m.nq > 6:
            d.qpos[6:] = (1.0 - float(q[6])) * m.jnt_range[6:, 1] if len(q) > 6 else 0.0
        mujoco.mj_kinematics(m, d)
        rng = np.random.default_rng(seed)
        out = []
        for g, verts in self.geoms:
            v = verts if len(verts) <= per_geom else verts[rng.choice(len(verts), per_geom, replace=False)]
            out.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
        return np.concatenate(out)

    def min_link_z(self, q: np.ndarray, base_radius: float = 0.16) -> float:
        """Lowest mesh vertex of any link *outside the base column* at ``q`` (the shoulder and
        link1 sit low by construction and never hang over the table)."""
        pts = self.points(q, per_geom=4000)
        over = pts[np.hypot(pts[:, 0], pts[:, 1]) > base_radius]
        return float(over[:, 2].min()) if len(over) else np.inf


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------


def arm_mask(frame: Frame, background: np.ndarray, min_closer: float = 0.03) -> np.ndarray:
    """Pixels where the scene got closer to the camera than the background by ``min_closer``."""
    valid = (frame.depth > 0.2) & (background > 0.2)
    closer = (background - frame.depth) > min_closer
    mask = (valid & closer).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # drop depth-edge flying pixels
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    for i in range(1, n):  # the arm can split into a few blobs (black cables/fingers)
        if stats[i, cv2.CC_STAT_AREA] > 0.05 * biggest:
            keep |= labels == i
    return keep


def median_depth(rig: CameraRig, role: str, n: int = 8) -> Frame:
    frames = [rig.grab_fresh(role, min_frames=2) for _ in range(n)]
    stack = np.stack([f.depth for f in frames])
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    last = frames[-1]
    return Frame(last.role, last.color, depth, last.K, last.t)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def initial_guess(obs: List[np.ndarray], model: List[np.ndarray], plane: Tuple[np.ndarray, float]) -> np.ndarray:
    """Level the camera cloud with the table normal, then a 2D Kabsch on per-pose centroids
    (yaw + xy), z from the mean height difference. Good to a few cm -- enough to start ICP."""
    n_cam, _ = plane
    R_level = rot_a_to_b(n_cam, np.array([0.0, 0.0, 1.0]))
    co = np.stack([apply_R(R_level, o).mean(0) for o in obs])
    cm = np.stack([mm.mean(0) for mm in model])
    # 2D Kabsch on xy
    a, b = co[:, :2], cm[:, :2]
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    Dm = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R2 = Vt.T @ Dm @ U.T
    T = np.eye(4)
    Rz = np.eye(3)
    Rz[:2, :2] = R2
    T[:3, :3] = Rz @ R_level
    T[:3, 3] = cm.mean(0) - T[:3, :3] @ np.stack([o.mean(0) for o in obs]).mean(0)
    return T


def apply_R(R: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T


def icp(obs: List[np.ndarray], model: List[np.ndarray], T0: np.ndarray, iters: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Point-to-point ICP, observed (cam) -> model (base), one rigid T for all poses.
    Returns (T, per-pose mean residual of the inlier matches)."""
    trees = [cKDTree(mm) for mm in model]
    T = T0.copy()
    thresh = 0.12
    per_pose = np.zeros(len(obs))
    for it in range(iters):
        src, dst = [], []
        for i, o in enumerate(obs):
            p = apply(T, o)
            dist, idx = trees[i].query(p, workers=-1)
            keep = dist < thresh
            per_pose[i] = float(dist[keep].mean()) if keep.any() else np.nan
            src.append(o[keep])
            dst.append(model[i][idx[keep]])
        A, B = np.concatenate(src), np.concatenate(dst)
        if len(A) < 100:
            break
        T_new = kabsch(A, B)
        delta = np.linalg.norm(T_new[:3, 3] - T[:3, 3]) + np.linalg.norm(T_new[:3, :3] - T[:3, :3])
        T = T_new
        thresh = max(0.015, thresh * 0.85)
        if delta < 1e-6 and it > 10:
            break
    return T, per_pose


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def plan_poses(args: Args, kin: Kinematics, surface: ArmSurface, q_home: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(target, q7) for every reachable grid point that keeps the whole arm high; poses whose
    joint-space transition from the previous one dips are dropped."""
    t = np.radians(args.tilt_deg)
    plan: List[Tuple[np.ndarray, np.ndarray]] = []
    q_prev = q_home.copy()
    for z in args.zs:
        for x in args.xs:
            for y in args.ys:
                target = np.array([x, y, z])
                away = np.array([x, y, 0.0]) / np.linalg.norm([x, y])
                d = -np.array([0.0, 0.0, 1.0]) * np.cos(t) + away * np.sin(t)
                try:
                    q6, _ = kin.ik_grasp(target, d, q_prev)
                except ValueError as e:
                    print(f"[plan] {np.round(target, 2)}: unreachable ({e})")
                    continue
                q7 = np.concatenate([q6, [0.0]])  # gripper closed: a compact tip
                own = surface.min_link_z(q7)
                lows = [surface.min_link_z(q_prev + (q7 - q_prev) * s) for s in np.linspace(0, 1, 12)]
                # the folded home pose itself has its elbow low over the table (z ~ 0.09); the
                # first leg only has to rise from there, later legs must stay high throughout
                floor = args.min_link_z if plan else min(args.min_link_z, surface.min_link_z(q_home) - 0.01)
                if own < args.min_link_z or min(lows) < floor:
                    print(f"[plan] {np.round(target, 2)}: dropped (pose min z {own:.3f}, on the way {min(lows):.3f})")
                    continue
                plan.append((target, q7))
                print(f"[plan] {np.round(target, 2)}: q={np.round(q6, 2)}  pose min z {own:.3f}, on the way {min(lows):.3f}")
                q_prev = q7
    lows = [surface.min_link_z(q_prev + (q_home - q_prev) * s) for s in np.linspace(0, 1, 12)]
    print(f"[plan] return home: lowest link z {min(lows):.3f}")
    return plan


def main(args: Args) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    debug = Path(args.debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else CALIB_DIR / f"top_{args.side}.json"

    arm = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel, execute=args.execute, workspace=Workspace(z=(0.03, 0.6)))
    surface = ArmSurface(arm.kin)
    q_home = arm.q().copy()
    print(f"[calib] home q={np.round(q_home, 3)}, lowest link at home z={surface.min_link_z(q_home):.3f}")
    plan = plan_poses(args, arm.kin, surface, q_home)
    print(f"[calib] {len(plan)} poses")
    if not args.execute:
        arm.close()
        return
    if len(plan) < 4:
        raise SystemExit("[calib] fewer than 4 safe poses -- adjust the grid")

    rig = CameraRig.open(roles=("top",))
    try:
        bg = median_depth(rig, "top")
        plane = table_plane_cam(bg)
        print(f"[calib] table plane (cam): n={np.round(plane[0], 4)} d={plane[1]:.4f}")
        bg.save(debug, "background")

        obs: List[np.ndarray] = []
        model: List[np.ndarray] = []
        qs: List[np.ndarray] = []
        for i, (target, q7) in enumerate(plan):
            result = arm.move_joints(q7)
            if result.aborted:
                print(f"[calib] pose {i}: move aborted ({result.aborted}); stopping")
                break
            time.sleep(0.5)
            q_meas = arm.q()
            frame = median_depth(rig, "top")
            mask = arm_mask(frame, bg.depth)
            pts_cam, _ = frame.point_cloud(mask)
            overlay = frame.color.copy()
            overlay[mask] = (0.5 * overlay[mask] + np.array([0, 0, 127])).astype(np.uint8)
            cv2.putText(overlay, f"pose {i} target {np.round(target, 2)} mask {int(mask.sum())}px", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(debug / f"pose_{i:02d}.png"), overlay)
            frame.save(debug, f"pose_{i:02d}")
            if len(pts_cam) < 2000:
                print(f"[calib] pose {i}: arm not visible enough ({len(pts_cam)} pts)")
                continue
            sub = pts_cam[np.random.default_rng(i).choice(len(pts_cam), min(4000, len(pts_cam)), replace=False)]
            obs.append(sub)
            model.append(surface.points(q_meas, per_geom=3000))
            qs.append(q_meas)
            print(f"[calib] pose {i} q={np.round(q_meas[:6], 2)} arm pixels {int(mask.sum())}")

        if len(obs) < 4:
            raise RuntimeError(f"only {len(obs)} usable poses")
        T0 = initial_guess(obs, model, plane)
        T, res = icp(obs, model, T0)
        print(f"[calib] ICP residual per pose (mm): {np.round(res * 1e3, 1)}")
        n_base = T[:3, :3] @ plane[0]
        p_base = apply(T, -plane[1] * plane[0])[0]
        d_base = float(-n_base @ p_base)
        print(f"[calib] table in base frame: normal {np.round(n_base, 3)}, z under the base origin {-d_base / n_base[2]:+.4f} m")
        print(f"[calib] camera position in base frame: {np.round(T[:3, 3], 3)}")

        # overlay check: project the posed model into the last pose's image
        frame_last = Frame.load(debug, f"pose_{len(plan) - 1:02d}")
        check = frame_last.color.copy()
        uv = frame_last.project(apply(np.linalg.inv(T), model[-1]))
        for u, v in uv[::4]:
            if 0 <= u < check.shape[1] and 0 <= v < check.shape[0]:
                check[int(v), int(u)] = (0, 255, 0)
        cv2.imwrite(str(debug / "fit_check.png"), check)

        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "side": args.side,
                    "T_base_cam": T.tolist(),
                    "table_normal_base": n_base.tolist(),
                    "table_d_base": d_base,
                    "table_normal_cam": plane[0].tolist(),
                    "table_d_cam": plane[1],
                    "icp_residual_m": [None if np.isnan(r) else float(r) for r in res],
                    "poses_q": [q.tolist() for q in qs],
                    "K": rig.grab("top").K.tolist(),
                },
                indent=2,
            )
        )
        print(f"[calib] wrote {out}")
    finally:
        try:
            arm.move_joints(q_home)
            arm.set_gripper(1.0)
        finally:
            arm.close()
            rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
