"""Pick one object off the table with one arm and put it in the box.

Pipeline for a single pick:
  1. fresh top frame -> GroundingDINO finds ``--object`` and ``--container`` -> depth lifts
     them to base-frame 3D (perception.Scene with the calibration from calibrate_top.py)
  2. plan: transit height above everything -> pre-grasp above the object -> straight descent
     to the grasp height -> close -> lift -> transit -> above the box centre -> lower to the
     drop height -> open -> retreat -> home
  3. execute with arm.Arm (slew-limited, workspace-boxed, tracking-error abort)

Every waypoint is printed first; with --no-execute nothing moves. Grasp height is measured
from the table plane, so the object's own height decides where the fingers close.

    python scripts/box_packing/pick_place.py --object "white cup" --no-execute
    python scripts/box_packing/pick_place.py --object "white cup"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Arm, MoveResult, Workspace  # noqa: E402
from calibrate_top import ArmSurface  # noqa: E402
from cameras import CameraRig, Frame  # noqa: E402
from perception import Detector, Scene, SceneObject, summarize  # noqa: E402
import json  # noqa: E402

WRIST = {"left": "left_wrist", "right": "right_wrist"}

PORTS = {"left": 1235, "right": 1234}


@dataclass
class Args:
    object: str = "white cup"
    container: str = "cardboard box"
    side: str = "left"
    execute: bool = False
    """Move the arm. Off by default: the plan is printed and drawn only."""
    grasp_below_top: float = 0.022
    """Fingertips this far below the object's measured top (rim / top face). Measured from the
    top, not the table: the object's top is what the camera sees best."""
    grasp_min_clear: float = 0.012
    """Never put the fingertips closer than this to the table."""
    approach: float = 0.10
    """Pre-grasp height above the object's top."""
    transit_clear: float = 0.16
    """Transit height above the table (must clear the box rim and every object; the carried
    object hangs below the fingertips by its grasp height)."""
    drop_clear: float = 0.06
    """Release height: object bottom this far above the box *floor* (the fingers go inside the
    box, below the rim), so the object is set down rather than dropped."""
    box_floor: float = 0.006
    """Cardboard thickness: the box floor sits this far above the table."""
    close_wait: float = 0.3
    """Pause after the close ramp before lifting, s."""
    open_wait: float = 0.3
    """Pause after the open ramp before retreating, s."""
    drop_offset_x: float = -0.02
    """Release point shifted from the box centre towards the base (x), which keeps the drop
    within a near-vertical reach while still well inside a 30 cm box."""
    close: float = 0.0
    """Gripper command when closed on the object (0 = fully closed)."""
    max_joint_vel: float = 0.6
    descend_speed: float = 0.06
    debug_dir: str = str(_HERE / "calib" / "debug")
    frames: int = 3
    """Top frames to median before detecting."""
    refine: bool = False
    """At the pre-grasp pose, re-detect the object with the wrist camera (hand-eye from
    calibrate_wrist.py) and correct the grasp point -- this measures the object relative to
    the gripper itself, so top-camera and FK errors largely cancel."""
    top_check: bool = False
    """At the pre-grasp pose, measure the gripper's real position in the top camera's depth and
    shift the grasp by the discrepancy (see gripper_offset_top)."""
    measure_only: bool = False
    """Go to the pre-grasp pose, run the top-camera gripper check, print it, come back."""
    refine_max_shift: float = 0.04
    """Ignore a wrist correction larger than this (m): it is a mis-detection, not a shift."""


def pick_object(objs: List[SceneObject], prompt: str) -> Optional[SceneObject]:
    words = set(prompt.lower().split())
    hits = [o for o in objs if words & set(o.label.lower().split())]
    if not hits:
        return None
    # highest score, but prefer exact phrase matches
    hits.sort(key=lambda o: (o.label.lower() == prompt.lower(), o.score), reverse=True)
    return hits[0]


def top_frame(rig: CameraRig, n: int) -> Frame:
    frames = [rig.grab_fresh("top", min_frames=2) for _ in range(n)]
    stack = np.stack([f.depth for f in frames])
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    f = frames[-1]
    return Frame(f.role, f.color, depth, f.K, f.t)


def finger_axis_for(obj: SceneObject) -> np.ndarray:
    """Close across the footprint's short side: fingers open perpendicular to the long axis."""
    return np.array([-np.sin(obj.yaw), np.cos(obj.yaw), 0.0])


def finger_axis_at_box(p: np.ndarray) -> np.ndarray:
    """Perpendicular to the base->box direction: a tilted wrist then keeps both fingertips at
    the same height, so the held object goes in level and neither tip leads into the floor."""
    a = np.arctan2(p[1], p[0])
    return np.array([-np.sin(a), np.cos(a), 0.0])


def plan(args: Args, scene: Scene, obj: SceneObject, box: SceneObject) -> List[Tuple[str, np.ndarray, str]]:
    """Waypoints as (name, xyz, how) with how in {'reach', 'line'}."""
    table_o = obj.bottom_z
    grasp_z = max(table_o + args.grasp_min_clear, obj.top_z - args.grasp_below_top)
    pre_z = obj.top_z + args.approach
    transit_z = table_o + args.transit_clear
    box_center = box.center_base.copy()
    box_center[2] = box.top_z  # rim
    # fingertips are (grasp_z - table_o) above the object's bottom; put that bottom drop_clear
    # above the box floor
    drop_z = box.bottom_z + args.box_floor + args.drop_clear + (grasp_z - table_o)
    xy_o = obj.center_top[:2]
    xy_b = box_center[:2] + np.array([args.drop_offset_x, 0.0])
    return [
        ("transit above object", np.array([*xy_o, transit_z]), "reach"),
        ("pre-grasp", np.array([*xy_o, pre_z]), "line"),
        ("grasp", np.array([*xy_o, grasp_z]), "line"),
        ("lift", np.array([*xy_o, transit_z]), "line"),
        ("transit above box", np.array([*xy_b, transit_z]), "reach"),
        # 'reach', not 'line': the placement wants the least-tilted wrist the spot allows (the
        # held object goes in as upright as possible), not the transit pose's orientation
        ("drop", np.array([*xy_b, drop_z]), "reach"),
        ("retreat", np.array([*xy_b, transit_z]), "reach"),
    ]


def refine_with_wrist(
    arm: Arm, rig: CameraRig, wrist_role: str, X: np.ndarray, scene: Scene, det: Detector, prompt: str, prior: SceneObject
) -> Optional[SceneObject]:
    """Re-locate the object from the wrist camera at the current pose: FK(mount) * X gives
    the camera pose in the base frame, the calibrated table plane the height reference."""
    time.sleep(0.4)
    frame = top_frame_like(rig, wrist_role, 4)
    p_m, R_m = arm.mount_pose()
    T_base_mount = np.eye(4)
    T_base_mount[:3, :3] = R_m
    T_base_mount[:3, 3] = p_m
    wrist_scene = Scene.from_pose(T_base_mount @ X, scene)
    objs = wrist_scene.locate(frame, det, [prompt], min_height=0.012, box_threshold=0.3)
    cv2.imwrite(str(Path(_HERE / "calib" / "debug" / "refine_wrist.png")), wrist_scene.draw(frame, objs))
    hit = pick_object(objs, prompt)
    if hit is None:
        print("       wrist refine: object not seen")
        return None
    return hit


def top_frame_like(rig: CameraRig, role: str, n: int) -> Frame:
    frames = [rig.grab_fresh(role, min_frames=2) for _ in range(n)]
    stack = np.stack([f.depth for f in frames])
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    f = frames[-1]
    return Frame(f.role, f.color, depth, f.K, f.t)


def gripper_offset_top(
    arm: Arm, rig: CameraRig, scene: Scene, background: np.ndarray, surface: "ArmSurface", debug: Path, tag: str
) -> Optional[np.ndarray]:
    """Where the gripper *really* is versus where FK + calibration say it is, measured in the
    top camera: the arm is segmented out of the depth image by comparison with the background
    frame, the points near the projected gripper are kept, and a translation-only trimmed ICP
    against the gripper's own mesh (posed by FK) gives the base-frame offset (measured - FK).
    Returns None if the gripper is not visible enough."""
    from scipy.spatial import cKDTree

    frame = top_frame(rig, 2)
    q = arm.q()
    model = surface.points(q, per_geom=4000)  # gripper + finger meshes only (see main)
    model_cam = scene.to_cam(model)
    uv = frame.project(model_cam)
    h, w = frame.depth.shape
    region = np.zeros((h, w), np.uint8)
    for u, v in uv:
        if 0 <= u < w and 0 <= v < h:
            region[int(v), int(u)] = 1
    region = cv2.dilate(region, np.ones((61, 61), np.uint8)).astype(bool)
    closer = (frame.depth > 0.25) & (background > 0.25) & ((background - frame.depth) > 0.03)
    mask = region & closer
    pts_cam, _ = frame.point_cloud(mask)
    overlay = frame.color.copy()
    overlay[mask] = (0.5 * overlay[mask] + np.array([0, 0, 127])).astype(np.uint8)
    for u, v in uv[::6]:
        if 0 <= u < w and 0 <= v < h:
            overlay[int(v), int(u)] = (0, 255, 0)
    cv2.imwrite(str(debug / f"gripper_offset_{tag}.png"), overlay)
    if len(pts_cam) < 15000:
        # a partial view (hand/arm in the way, depth dropout) gives a fit that is worse than
        # no correction at all -- the healthy runs see 50k-70k points here
        print(f"       gripper offset: only {len(pts_cam)} depth points on the gripper -- skipped")
        return None
    obs = scene.to_base(pts_cam)
    tree = cKDTree(model)
    t = np.zeros(3)
    for _ in range(30):
        dist, idx = tree.query(obs + t, workers=-1)
        keep = dist <= np.percentile(dist, 60)
        t_new = (model[idx[keep]] - obs[keep]).mean(0)
        if np.linalg.norm(t_new - t) < 1e-5:
            t = t_new
            break
        t = t_new
    dist, _ = tree.query(obs + t, workers=-1)
    print(f"       gripper offset (FK - measured) {np.round(t * 1e3, 1)} mm from {len(obs)} points, fit {np.median(dist) * 1e3:.1f} mm median")
    return -t  # measured - FK


def solve_approach(arm: Arm, p_grasp: np.ndarray, R_grasp: np.ndarray, q_seed: np.ndarray, approach: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pre-grasp pose straight above the grasp with the *grasp's own* orientation, so the
    descent is a pure translation (no wrist re-orientation on the way down). If that height
    is not reachable with that orientation, the approach shrinks until it is."""
    for h in np.arange(approach, 0.029, -0.01):
        p = p_grasp + np.array([0.0, 0.0, h])
        q6, perr, rerr = arm.kin.ik(p, R_grasp, q_seed)
        if perr < 5e-3 and rerr < 0.05:
            return q6, p, float(h)
    raise ValueError("no reachable pre-grasp above the grasp with its orientation")


def main(args: Args) -> None:
    debug = Path(args.debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    scene = Scene.from_calib(args.side)
    det = Detector()
    rig = CameraRig.open(roles=("top", WRIST[args.side]) if args.refine else ("top",))
    X_mount_cam = None
    if args.refine:
        X_mount_cam = np.array(json.loads((_HERE / "calib" / f"wrist_{args.side}.json").read_text())["T_mount_cam"])
    arm: Optional[Arm] = None
    try:
        frame = top_frame(rig, args.frames)
        background = frame.depth.copy()
        objs = scene.locate(frame, det, [args.object, args.container], keep_points=False)
        print("[scene]\n" + summarize(objs))
        cv2.imwrite(str(debug / "pick_scene.png"), scene.draw(frame, objs))
        obj = pick_object(objs, args.object)
        box = pick_object(objs, args.container)
        if obj is None or box is None:
            raise SystemExit(f"[error] not found: object={obj is not None} container={box is not None}")
        print(
            f"[target] {obj.label}: centre ({obj.center_base[0]:.3f}, {obj.center_base[1]:.3f}) top-band centre "
            f"({obj.center_top[0]:.3f}, {obj.center_top[1]:.3f}), table z {obj.bottom_z:.3f}, "
            f"height {obj.height * 100:.1f} cm, footprint {obj.size[0] * 100:.1f} x {obj.size[1] * 100:.1f} cm, yaw {np.degrees(obj.yaw):.0f}"
        )
        print(f"[box] centre ({box.center_base[0]:.3f}, {box.center_base[1]:.3f}), rim z {box.top_z:.3f} ({box.height * 100:.1f} cm tall)")
        if obj.size[1] > 0.09:
            raise SystemExit(f"[error] object footprint {obj.size[1] * 100:.1f} cm is wider than the 9.6 cm gripper opening")

        # remember where the object started, for reset_object.py
        (debug / "last_pick.json").write_text(
            json.dumps({"object": args.object, "start_xy": obj.center_top[:2].tolist(), "time": time.time()})
        )
        wps = plan(args, scene, obj, box)
        print("[plan]")
        for name, p, how in wps:
            print(f"  {name:22s} {np.round(p, 3)}  ({how})")

        table_z = min(obj.bottom_z, box.bottom_z)
        ws = Workspace(z=(table_z + 0.005, 0.6))
        arm = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel, workspace=ws, execute=args.execute)
        q_home = arm.q().copy()
        # gripper-only surface (the arm links are excluded so the check is local to the hand)
        surface = ArmSurface(arm.kin, skip_bodies=("base", "link1", "link2", "link3", "link4", "link5"))
        # Solve every waypoint's IK up front: the execution loop below then only streams
        # pre-computed joint targets (an IK scan over tilts/rolls costs a second or two and
        # used to show up as a pause at every waypoint).
        q_prev = q_home.copy()
        axis_obj = finger_axis_for(obj)
        print(f"[grasp] finger opening axis {np.round(axis_obj, 2)} (across the {obj.size[1] * 100:.1f} cm side)")
        solved: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        for name, p, how in wps:
            at_box = name in ("transit above box", "drop", "retreat")
            ax = finger_axis_at_box(p) if at_box else axis_obj
            q6, R, tilt = arm.kin.ik_reach(p, q_prev, finger_axis=ax)
            solved[name] = (q6, R, tilt)
            print(f"  ik ok: {name:22s} tilt {tilt:.0f} deg  q {np.round(q6, 2)}")
            q_prev = np.concatenate([q6, [q_prev[6]]])
        # the pre-grasp takes the grasp's orientation (the least tilt the grasp point allows),
        # so the descent is a pure translation and the grasp is as top-down as reachable
        p_g = next(p for n, p, _ in wps if n == "grasp")
        q6, p_pre, h = solve_approach(arm, p_g, solved["grasp"][1], solved["grasp"][0], args.approach)
        solved["pre-grasp"] = (q6, solved["grasp"][1], solved["grasp"][2])
        wps[[n for n, _, _ in wps].index("pre-grasp")] = ("pre-grasp", p_pre, "line")
        print(f"  pre-grasp re-solved {h * 100:.0f} cm above the grasp at the grasp tilt ({solved['grasp'][2]:.0f} deg)")
        if not args.execute:
            print("[dry run] pass --execute to move")
            return

        def q7(q6: np.ndarray) -> np.ndarray:
            return np.concatenate([q6, [arm.q_cmd[6]]])

        def report(name: str, p: np.ndarray, res: MoveResult) -> None:
            p_now = arm.grasp_pose()[0]
            print(f"[move] {name:22s} -> {np.round(p, 3)}  at {np.round(p_now, 3)}  err {np.linalg.norm(p_now - p) * 1e3:.1f} mm {res.aborted}")
            if res.aborted and res.aborted != "ik on correction":
                raise RuntimeError(f"aborted at {name}: {res.aborted}")

        targets = {name: p for name, p, _ in wps}
        corr = np.zeros(3)
        arm.set_gripper(1.0, wait=0.0)

        # --- to the object -------------------------------------------------------------
        name = "transit above object"
        res = arm.move_joints(q7(solved[name][0]), settle=0.0)
        report(name, targets[name], res)
        name = "pre-grasp"
        R_obj = solved[name][1]
        res = arm.move_joints(q7(solved[name][0]), settle=0.0)
        report(name, targets[name], res)
        if args.top_check:
            off = gripper_offset_top(arm, rig, scene, background, surface, debug, "pregrasp")
            if args.measure_only:
                print("[measure-only] done; returning home")
                arm.move_joints(q_home)
                return
            if off is not None and np.linalg.norm(off[:2]) <= args.refine_max_shift:
                corr = -off
                corr[2] = 0.0
                print(f"       applying xy correction {np.round(corr[:2] * 1e3, 1)} mm to the grasp")
            elif off is not None:
                print(f"       correction {np.linalg.norm(off[:2]) * 1e3:.0f} mm too large -- ignored")
        # --- grasp ------------------------------------------------------------------------
        name = "grasp"
        p_grasp = targets[name] + corr
        # descend with the pre-grasp orientation: re-orienting to the vertical grasp branch up here
        # is not reachable, and a 20 deg finger tilt grasps a cup just as well
        res = arm.move_linear(p_grasp, R_obj, speed=args.descend_speed, correct=1, track_abort_rad=0.12)
        report(name, p_grasp, res)
        arm.set_gripper(args.close, wait=args.close_wait)
        print(f"       gripper closed to {arm.gripper():.3f} (thin objects read close to 0 -- confirm with the camera)")
        # --- lift and carry ---------------------------------------------------------------
        name = "lift"
        p_lift = targets[name] + corr
        q6, perr, rerr = arm.kin.ik(p_lift, R_obj, arm.q_cmd)
        if perr > 5e-3 or rerr > 0.05:
            q6 = solved[name][0]
        res = arm.move_joints(q7(q6), settle=0.0)
        report(name, p_lift, res)
        name = "transit above box"
        res = arm.move_joints(q7(solved[name][0]), settle=0.0)
        report(name, targets[name], res)
        name = "drop"
        res = arm.move_joints(q7(solved[name][0]), settle=0.0)
        # one residual-compensation round: the far-reaching placement pose sags a few cm
        res = arm._correct(targets[name], solved[name][1], 0.1)
        report(name, targets[name], res)
        arm.set_gripper(1.0, wait=args.open_wait)
        name = "retreat"
        res = arm.move_joints(q7(solved[name][0]), settle=0.0)
        report(name, targets[name], res)
        print("[done] returning home")
        arm.move_joints(q_home)
    finally:
        if arm is not None:
            arm.close()
        rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
