"""Landmark hand-eye calibration: wrist camera -> gripper mount, and top camera -> arm base.

Why not the arm-in-depth method (calibrate_top.py): the D435 sits 0.3-0.45 m from the arm at
any useful pose, inside its minimum-range zone at 1280x720, so its depth on the arm is junk.
The table objects, however, are 0.55-0.8 m from the top camera and 0.3-0.5 m from the wrist
D405 -- both cameras measure *them* well. So the objects are the calibration target:

  1. top camera at rest: detect the landmark objects (GroundingDINO) and lift them to 3D in the
     top-camera frame (points above the table plane inside each box, centroid).
  2. the arm visits N high, touch-free survey poses that look down at the objects with the
     wrist camera; at each one the same objects are lifted to 3D in the wrist frame, and the
     mount pose comes from FK.
  3. solve X = T_mount_wristcam and the landmark positions L_j (base frame) by alternating
     least squares:  FK_k * X * p_kj  ~=  L_j   (multi-start over the camera roll).
  4. T_base_top from L_j vs. the top-camera centroids (Kabsch).

Outputs calib/wrist_<side>.json (X) and calib/top_<side>.json (T_base_cam + table plane, the
format perception.Scene reads). Every arm pose keeps all links above --min-link-z.

    python scripts/box_packing/calibrate_wrist.py --side left --no-execute   # plan only
    python scripts/box_packing/calibrate_wrist.py --side left
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Arm, Workspace  # noqa: E402
from calibrate_top import CALIB_DIR, ArmSurface, apply, fit_plane, kabsch, median_depth  # noqa: E402
from cameras import CameraRig, Frame  # noqa: E402
from perception import Detector, Scene, SceneObject  # noqa: E402

PORTS = {"left": 1235, "right": 1234}
WRIST = {"left": "left_wrist", "right": "right_wrist"}


@dataclass
class Args:
    side: str = "left"
    execute: bool = True
    landmarks: List[str] = field(default_factory=lambda: ["white cup", "lemon", "blue cube", "red cube", "carrot"])
    """Objects used as calibration targets; each must be unique on the table."""
    look_at: Tuple[float, float] = (0.40, 0.10)
    """Base-frame xy the survey poses aim the camera at (the landmark cluster)."""
    min_link_z: float = 0.17
    max_joint_vel: float = 0.5
    debug_dir: str = str(CALIB_DIR / "debug")


def survey_poses() -> List[Tuple[np.ndarray, float, float]]:
    """(grasp point, tilt deg, finger-axis yaw deg): high, varied in position and orientation."""
    return [
        (np.array([0.26, 0.10, 0.36]), 45.0, 0.0),
        (np.array([0.22, 0.02, 0.32]), 40.0, 45.0),
        (np.array([0.22, 0.18, 0.32]), 40.0, -45.0),
        (np.array([0.28, 0.06, 0.30]), 35.0, 90.0),
        (np.array([0.20, 0.10, 0.36]), 55.0, 0.0),
        (np.array([0.30, 0.14, 0.33]), 50.0, 45.0),
        (np.array([0.24, 0.00, 0.30]), 30.0, -45.0),
        (np.array([0.26, 0.14, 0.28]), 25.0, 90.0),
    ]


def finger_dir(p: np.ndarray, look_at: Tuple[float, float], tilt_deg: float) -> np.ndarray:
    """Fingers tilted from vertical towards ``look_at`` -- the wrist cam looks along them."""
    away = np.array([look_at[0] - p[0], look_at[1] - p[1], 0.0])
    away = away / max(np.linalg.norm(away), 1e-6)
    t = np.radians(tilt_deg)
    return -np.array([0.0, 0.0, 1.0]) * np.cos(t) + away * np.sin(t)


def lift_landmarks(frame: Frame, det: Detector, prompts: List[str], min_h: float = 0.012) -> Dict[str, SceneObject]:
    """Landmark -> object in the *camera* frame (Scene with identity T and the camera's own
    table plane). Only unambiguous, well-scored detections are kept."""
    pts, _ = frame.point_cloud((frame.depth > 0.15) & (frame.depth < 1.2))
    if len(pts) < 5000:
        return {}
    n, d = fit_plane(pts[:: max(1, len(pts) // 50000)])
    scene = Scene(np.eye(4), n, d)
    dets = det.detect(frame.color, prompts, box_threshold=0.35)
    out: Dict[str, SceneObject] = {}
    for prompt, hit in Detector.assign_verified(dets, prompts, frame.color).items():
        obj = scene.lift(frame, hit, min_height=min_h)
        if obj is None or obj.n_points < 80 or obj.size[0] > 0.15:
            continue
        out[prompt] = obj
    return out


def solve_hand_eye(
    mounts: List[Tuple[np.ndarray, np.ndarray]], obs: List[Dict[str, np.ndarray]], X0: np.ndarray, iters: int = 100
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """Alternating least squares for X (mount->cam) and landmarks L (base). Returns (X, L, rms)."""
    X = X0.copy()
    names = sorted({n for o in obs for n in o})
    for _ in range(iters):
        # landmarks given X
        L: Dict[str, np.ndarray] = {}
        for name in names:
            pts = [apply(T_mount(m), apply(X, o[name]))[0] for m, o in zip(mounts, obs, strict=True) if name in o]
            L[name] = np.mean(pts, axis=0)
        # X given landmarks: cam points -> mount-frame landmark positions
        A, B = [], []
        for m, o in zip(mounts, obs, strict=True):
            Tm_inv = np.linalg.inv(T_mount(m))
            for name, p in o.items():
                A.append(p)
                B.append(apply(Tm_inv, L[name])[0])
        X_new = kabsch(np.stack(A), np.stack(B))
        done = np.linalg.norm(X_new[:3, 3] - X[:3, 3]) < 1e-6 and np.linalg.norm(X_new[:3, :3] - X[:3, :3]) < 1e-6
        X = X_new
        if done:
            break
    res = [np.linalg.norm(apply(T_mount(m), apply(X, o[n]))[0] - L[n]) for m, o in zip(mounts, obs, strict=True) for n in o]
    return X, L, float(np.sqrt(np.mean(np.square(res))))


def T_mount(m: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = m[1]
    T[:3, 3] = m[0]
    return T


def main(args: Args) -> None:
    debug = Path(args.debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    wrist = WRIST[args.side]
    arm = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel, execute=args.execute, workspace=Workspace(z=(0.03, 0.6)))
    surf = ArmSurface(arm.kin)
    q_home = arm.q().copy()

    # ---- plan (and safety-check) the survey poses --------------------------------------
    plan: List[np.ndarray] = []
    q_prev = q_home.copy()
    for p, tilt, yaw in survey_poses():
        d = finger_dir(p, args.look_at, tilt)
        axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
        try:
            q6, _ = arm.kin.ik_grasp(p, d, q_prev, finger_axis=axis)
        except ValueError as e:
            print(f"[plan] {np.round(p, 2)} tilt {tilt:.0f}: unreachable ({e})")
            continue
        q7 = np.concatenate([q6, [1.0]])
        own = surf.min_link_z(q7)
        lows = [surf.min_link_z(q_prev + (q7 - q_prev) * s) for s in np.linspace(0, 1, 12)]
        floor = args.min_link_z if plan else min(args.min_link_z, surf.min_link_z(q_home) - 0.01)
        if own < args.min_link_z or min(lows) < floor:
            print(f"[plan] {np.round(p, 2)} tilt {tilt:.0f}: dropped (pose min z {own:.3f}, on the way {min(lows):.3f})")
            continue
        print(f"[plan] {np.round(p, 2)} tilt {tilt:.0f} yaw {yaw:.0f}: q {np.round(q6, 2)}  min link z {own:.3f} / {min(lows):.3f}")
        plan.append(q7)
        q_prev = q7
    print(f"[plan] {len(plan)} survey poses")
    if not args.execute:
        arm.close()
        return
    if len(plan) < 4:
        raise SystemExit("[calib] too few safe poses")

    det = Detector()
    rig = CameraRig.open(roles=("top", wrist))
    try:
        # ---- top camera landmarks -----------------------------------------------------------
        top = median_depth(rig, "top", n=6)
        top.save(debug, "wristcal_top")
        top_objs = lift_landmarks(top, det, args.landmarks)
        print(f"[top] landmarks: {sorted(top_objs)}")
        for n, o in top_objs.items():
            print(f"   {n:12s} cam {np.round(o.center_base, 3)} h {o.height * 100:.1f} cm pts {o.n_points}")
        if len(top_objs) < 4:
            raise SystemExit("[calib] need >= 4 landmarks in the top view")

        # ---- survey ---------------------------------------------------------------------------
        mounts: List[Tuple[np.ndarray, np.ndarray]] = []
        obs: List[Dict[str, np.ndarray]] = []
        for i, q7 in enumerate(plan):
            res = arm.move_joints(q7)
            if res.aborted:
                print(f"[calib] pose {i} aborted: {res.aborted}")
                break
            time.sleep(0.6)
            q = arm.q()
            frame = median_depth(rig, wrist, n=6)
            frame.save(debug, f"wristcal_{i:02d}")
            objs = lift_landmarks(frame, det, args.landmarks)
            objs = {n: o for n, o in objs.items() if n in top_objs}
            img = frame.color.copy()
            for n, o in objs.items():
                x0, y0, x1, y1 = [int(v) for v in o.box]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, n, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(debug / f"wristcal_{i:02d}_det.png"), img)
            print(f"[calib] pose {i}: q {np.round(q[:6], 2)} sees {sorted(objs)}")
            if len(objs) < 3:
                continue
            mounts.append(arm.kin.mount(q))
            obs.append({n: o.center_base for n, o in objs.items()})
            (debug / f"wristcal_{i:02d}_q.json").write_text(json.dumps(q.tolist()))
        if len(obs) < 4:
            raise SystemExit(f"[calib] only {len(obs)} poses with >= 3 landmarks")

        # ---- solve ------------------------------------------------------------------------
        best = None
        # camera roughly at the mount, looking along the fingers (mount -z); scan its roll
        for roll in np.radians(np.arange(0, 360, 45)):
            cr, sr = np.cos(roll), np.sin(roll)
            R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
            R0 = R_roll @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # cam z = mount -z
            X0 = np.eye(4)
            X0[:3, :3] = R0
            X0[:3, 3] = np.array([0.0, 0.0, -0.02])
            X, L, rms = solve_hand_eye(mounts, obs, X0)
            print(f"   roll start {np.degrees(roll):3.0f}: rms {rms * 1e3:.1f} mm")
            if best is None or rms < best[0]:
                best = (rms, X, L)
        rms, X, L = best
        print(f"[wrist] X = T_mount_cam: cam at {np.round(X[:3, 3], 3)} in the mount frame, cam z axis (mount) {np.round(X[:3, 2], 3)}; rms {rms * 1e3:.1f} mm")
        for n in sorted(L):
            print(f"   {n:12s} base {np.round(L[n], 3)}")

        # ---- top camera from the landmarks ----------------------------------------------
        names = [n for n in sorted(L) if n in top_objs]
        A = np.stack([top_objs[n].center_base for n in names])
        B = np.stack([L[n] for n in names])
        T = kabsch(A, B)
        res = np.linalg.norm(apply(T, A) - B, axis=1)
        print(f"[top] T_base_cam from {len(names)} landmarks: residuals {np.round(res * 1e3, 1)} mm; camera at {np.round(T[:3, 3], 3)}")
        pts, _ = top.point_cloud((top.depth > 0.3) & (top.depth < 1.5))
        n_cam, d_cam = fit_plane(pts[:: max(1, len(pts) // 60000)])
        n_base = T[:3, :3] @ n_cam
        d_base = float(-n_base @ apply(T, -d_cam * n_cam)[0])
        print(f"[top] table normal (base) {np.round(n_base, 3)}, z under origin {-d_base / n_base[2]:+.4f}")
        CALIB_DIR.mkdir(exist_ok=True)
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
        print("[calib] wrote calib/wrist_left.json and calib/top_left.json")
    finally:
        try:
            arm.move_joints(q_home)
        finally:
            arm.close()
            rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
