"""Auto-reset: take an object out of the box and put it back on the table at a random free
spot near where it started, without touching anything else.

  1. top frame -> object (in the box), box, and an obstacle map: every depth point higher than
     1 cm above the table, in base-frame xy (other objects, the box walls, ...)
  2. sample a placement inside --reset-radius of --reset-center; accept it only if nothing
     stands inside the footprint the open gripper + object sweep there (an ellipse: half-axes
     --clear-along along the finger axis, --clear-across across it)
  3. pick from the box (fingers inside the box, both tips level), carry at transit height,
     set the object down --place-clear above the table, open, retreat, home.

Same execution style as pick_place.py: every waypoint solved before anything moves, joint
targets streamed without settling, one residual-compensation round at the grasp and the place.

    python scripts/box_packing/reset_object.py --object "white cup" --reset-center 0.405 0.018 --no-execute
    python scripts/box_packing/reset_object.py --object "white cup" --reset-center 0.405 0.018 --execute
"""

from __future__ import annotations

import json
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
from pick_place import PORTS, finger_axis_for, gripper_offset_top, pick_object, top_frame  # noqa: E402


@dataclass
class Args:
    object: str = "white cup"
    container: str = "cardboard box"
    side: str = "left"
    execute: bool = False
    reset_center: Optional[Tuple[float, float]] = None
    """Base-frame xy the object started at (where it should go back to, roughly). Default:
    the start position pick_place.py recorded in calib/debug/last_pick.json."""
    reset_radius: float = 0.05
    """Placement is sampled uniformly in this disc around --reset-center."""
    clear_along: float = 0.075
    """Half-axis of the required free ellipse along the finger-opening axis (open fingers
    reach +-4.8 cm, plus margin)."""
    clear_across: float = 0.055
    """Half-axis across the finger axis (object radius plus margin)."""
    obstacle_height: float = 0.015
    """Depth points higher than this above the table count as obstacles."""
    seed: Optional[int] = None
    grasp_below_top: float = 0.022
    box_floor: float = 0.006
    approach: float = 0.10
    transit_clear: float = 0.16
    place_clear: float = 0.01
    """Object bottom this far above the table when the gripper opens."""
    max_joint_vel: float = 0.6
    descend_speed: float = 0.06
    close: float = 0.0
    close_wait: float = 0.3
    open_wait: float = 0.3
    top_check: bool = False
    """Top-camera gripper check; off -- its correction has not been consistent run to run."""
    contact_abort_rad: float = 0.12
    """Tracking lag that aborts a descent (contact)."""
    refine_max_shift: float = 0.04
    debug_dir: str = str(_HERE / "calib" / "debug")


def obstacle_points(frame: Frame, scene: Scene, min_height: float, cell: float = 0.01, min_count: int = 40) -> np.ndarray:
    """Centres of 1 cm table cells that hold at least ``min_count`` depth points higher than
    ``min_height`` above the table. A real object surface puts ~200 points into such a cell
    at this range, the depth noise a handful, so the count separates them cleanly."""
    pts_cam, _ = frame.point_cloud((frame.depth > 0.3) & (frame.depth < 1.5))
    pts = scene.to_base(pts_cam)
    h = scene.height(pts)
    up = pts[(h > min_height) & (h < 0.5)]
    xe = np.arange(0.05, 0.85 + cell, cell)
    ye = np.arange(-0.55, 0.55 + cell, cell)
    counts, _, _ = np.histogram2d(up[:, 0], up[:, 1], bins=[xe, ye])
    ix, iy = np.nonzero(counts >= min_count)
    return np.stack([xe[ix] + cell / 2, ye[iy] + cell / 2, np.zeros(len(ix))], axis=1)


def spot_is_free(xy: np.ndarray, axis: np.ndarray, obstacles_xy: np.ndarray, along: float, across: float, max_points: int = 1) -> bool:
    """Free if no occupied cell (see obstacle_points) falls inside the ellipse."""
    d = obstacles_xy - xy
    u = d @ axis[:2]
    v = d @ np.array([-axis[1], axis[0]])
    return int(np.sum((u / along) ** 2 + (v / across) ** 2 < 1.0)) < max_points


def sample_spot(args: Args, axis: np.ndarray, obstacles_xy: np.ndarray, rng: np.random.Generator, tries: int = 400) -> Optional[np.ndarray]:
    c = np.asarray(args.reset_center)
    for _ in range(tries):
        r = args.reset_radius * np.sqrt(rng.random())
        a = rng.random() * 2 * np.pi
        xy = c + r * np.array([np.cos(a), np.sin(a)])
        if spot_is_free(xy, axis, obstacles_xy, args.clear_along, args.clear_across):
            return xy
    return None


def draw_map(frame: Frame, scene: Scene, obstacles: np.ndarray, spot: Optional[np.ndarray], center: np.ndarray, path: Path) -> None:
    img = frame.color.copy()
    cells = obstacles.copy()
    cells[:, 2] = [scene.table_z(x, y) for x, y in cells[:, :2]]
    uv = frame.project(scene.to_cam(cells))
    for u, v in uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (int(u), int(v)), 2, (0, 0, 255), -1)
    for p, color, label in ((center, (255, 0, 0), "reset centre"), (spot, (0, 255, 0), "target")):
        if p is None:
            continue
        uv = frame.project(scene.to_cam(np.array([p[0], p[1], scene.table_z(p[0], p[1])])))[0]
        cv2.circle(img, (int(uv[0]), int(uv[1])), 10, color, 2)
        cv2.putText(img, label, (int(uv[0]) + 12, int(uv[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(path), img)


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


def targets_of(wps: List[Tuple[str, np.ndarray, str]], name: str) -> np.ndarray:
    return next(p for n, p, _ in wps if n == name)


def main(args: Args) -> None:
    debug = Path(args.debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.reset_center is None:
        last = json.loads((Path(args.debug_dir) / "last_pick.json").read_text())
        args.reset_center = tuple(last["start_xy"])
        print(f"[reset] centre from last_pick.json ({last['object']}): {np.round(args.reset_center, 3)}")
    scene = Scene.from_calib(args.side)
    det = Detector()
    rig = CameraRig.open(roles=("top",))
    arm: Optional[Arm] = None
    try:
        frame = top_frame(rig, 3)
        background = frame.depth.copy()
        objs = scene.locate(frame, det, [args.object, args.container])
        print("[scene]\n" + summarize(objs))
        obj = pick_object(objs, args.object)
        box = pick_object(objs, args.container)
        if obj is None or box is None:
            raise SystemExit(f"[error] not found: object={obj is not None} container={box is not None}")
        # the object stands on the box floor, not the table: its true height and bottom
        true_height = obj.height - args.box_floor
        bottom_now = obj.top_z - true_height
        print(f"[object] {obj.label} in the box at ({obj.center_top[0]:.3f}, {obj.center_top[1]:.3f}), top z {obj.top_z:.3f}, height {true_height * 100:.1f} cm")

        # obstacle map, excluding the object itself
        obstacles = obstacle_points(frame, scene, args.obstacle_height)
        keep = np.linalg.norm(obstacles[:, :2] - obj.center_base[:2], axis=1) > 0.06
        obstacles = obstacles[keep]
        axis_place = np.array([1.0, 0.0, 0.0])  # fingers open along x at the table, like the pick
        spot = sample_spot(args, axis_place, obstacles[:, :2], rng)
        draw_map(frame, scene, obstacles, spot, np.asarray(args.reset_center), debug / "reset_map.png")
        if spot is None:
            raise SystemExit("[error] no free spot within the reset radius -- enlarge --reset-radius or clear the area")
        print(f"[place] random free spot ({spot[0]:.3f}, {spot[1]:.3f}), {np.linalg.norm(spot - np.asarray(args.reset_center)) * 100:.1f} cm from the reset centre")

        # ---- plan -------------------------------------------------------------------------
        table_b = box.bottom_z
        table_s = scene.table_z(spot[0], spot[1])
        grasp_z = obj.top_z - args.grasp_below_top
        grip_above_bottom = grasp_z - bottom_now
        transit_z = table_b + args.transit_clear
        xy_o = obj.center_top[:2]
        place_z = table_s + args.place_clear + grip_above_bottom
        wps: List[Tuple[str, np.ndarray, str]] = [
            ("transit above box", np.array([*xy_o, transit_z]), "reach"),
            ("pre-grasp", np.array([*xy_o, obj.top_z + args.approach]), "line"),
            ("grasp", np.array([*xy_o, grasp_z]), "line"),
            ("lift", np.array([*xy_o, transit_z]), "line"),
            ("transit above spot", np.array([*spot, transit_z]), "reach"),
            ("pre-place", np.array([*spot, place_z + 0.08]), "reach"),
            ("place", np.array([*spot, place_z]), "line"),
            ("retreat", np.array([*spot, transit_z]), "reach"),
        ]
        print("[plan]")
        for name, p, how in wps:
            print(f"  {name:20s} {np.round(p, 3)}  ({how})")

        arm = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel, workspace=Workspace(z=(min(table_b, table_s) + 0.005, 0.6)), execute=args.execute)
        surface = ArmSurface(arm.kin, skip_bodies=("base", "link1", "link2", "link3", "link4", "link5"))
        q_home = arm.q().copy()
        q_prev = q_home.copy()
        solved: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        axis_grasp = finger_axis_for(obj)  # close across the narrow side, clear of a handle
        print(f"[grasp] finger opening axis {np.round(axis_grasp, 2)} (object long axis yaw {np.degrees(obj.yaw):.0f} deg)")
        for name, p, how in wps:
            ax = axis_grasp if name in ("transit above box", "pre-grasp", "grasp", "lift") else axis_place
            q6, R, tilt = arm.kin.ik_reach(p, q_prev, finger_axis=ax)
            solved[name] = (q6, R, tilt)
            print(f"  ik ok: {name:20s} tilt {tilt:.0f} deg  q {np.round(q6, 2)}")
            q_prev = np.concatenate([q6, [q_prev[6]]])
        # the pre-grasp takes the grasp's orientation (the least tilt the grasp point allows)
        q6, p_pre, h = solve_approach(arm, targets_of(wps, "grasp"), solved["grasp"][1], solved["grasp"][0], args.approach)
        solved["pre-grasp"] = (q6, solved["grasp"][1], solved["grasp"][2])
        wps[[n for n, _, _ in wps].index("pre-grasp")] = ("pre-grasp", p_pre, "line")
        print(f"  pre-grasp re-solved {h * 100:.0f} cm above the grasp at the grasp tilt ({solved['grasp'][2]:.0f} deg)")
        # with a tilted wrist the two fingertips sit at different heights when the finger axis
        # has a component along the tilt; raise the grasp so the lower tip stays off the floor
        R_g = solved["grasp"][1]
        tip_dz = 0.048 * abs(float(R_g[2, 1]))  # site y = finger axis; its z component
        if tip_dz > 0.004:
            targets_shift = tip_dz / 2
            print(f"[grasp] tilt {solved['grasp'][2]:.0f} deg puts the tips {tip_dz * 100:.1f} cm apart in height -> grasp raised {targets_shift * 100:.1f} cm")
            for i, (n2, p2, h2) in enumerate(wps):
                if n2 == "grasp":
                    wps[i] = (n2, p2 + np.array([0, 0, targets_shift]), h2)
        if not args.execute:
            print("[dry run] pass --execute to move")
            return

        def q7(q6: np.ndarray) -> np.ndarray:
            return np.concatenate([q6, [arm.q_cmd[6]]])

        def report(name: str, p: np.ndarray, res: MoveResult) -> None:
            p_now = arm.grasp_pose()[0]
            print(f"[move] {name:20s} -> {np.round(p, 3)}  at {np.round(p_now, 3)}  err {np.linalg.norm(p_now - p) * 1e3:.1f} mm {res.aborted}")
            if res.aborted and res.aborted != "ik on correction":
                raise RuntimeError(f"aborted at {name}: {res.aborted}")

        targets = {name: p for name, p, _ in wps}
        arm.set_gripper(1.0, wait=0.0)
        corr = np.zeros(3)

        name = "transit above box"
        report(name, targets[name], arm.move_joints(q7(solved[name][0]), settle=0.0))
        name = "pre-grasp"
        R_box = solved[name][1]
        report(name, targets[name], arm.move_joints(q7(solved[name][0]), settle=0.0))
        if args.top_check:
            off = gripper_offset_top(arm, rig, scene, background, surface, debug, "reset_pregrasp")
            if off is not None and np.linalg.norm(off[:2]) <= args.refine_max_shift:
                corr = -off
                corr[2] = 0.0
                print(f"       applying xy correction {np.round(corr[:2] * 1e3, 1)} mm to the grasp")
        name = "grasp"
        p_grasp = targets[name] + corr
        res = arm.move_linear(p_grasp, R_box, speed=args.descend_speed, correct=1, track_abort_rad=args.contact_abort_rad)
        report(name, p_grasp, res)
        if np.linalg.norm(arm.grasp_pose()[0] - p_grasp) > 0.02:
            print("[abort] the descent did not reach the grasp point (contact?) -- backing out without closing")
            arm.move_joints(q7(solved["lift"][0]), settle=0.0)
            arm.move_joints(q_home)
            return
        arm.set_gripper(args.close, wait=args.close_wait)
        print(f"       gripper closed to {arm.gripper():.3f}")
        name = "lift"
        p_lift = targets[name] + corr
        q6, perr, rerr = arm.kin.ik(p_lift, R_box, arm.q_cmd)
        if perr > 5e-3 or rerr > 0.05:
            q6 = solved[name][0]
        report(name, p_lift, arm.move_joints(q7(q6), settle=0.0))
        for name in ("transit above spot", "pre-place"):
            report(name, targets[name], arm.move_joints(q7(solved[name][0]), settle=0.0))
        name = "place"
        report(name, targets[name], arm.move_linear(targets[name], solved["pre-place"][1], speed=args.descend_speed, correct=1, track_abort_rad=args.contact_abort_rad))
        arm.set_gripper(1.0, wait=args.open_wait)
        name = "retreat"
        report(name, targets[name], arm.move_joints(q7(solved[name][0]), settle=0.0))
        print("[done] returning home")
        arm.move_joints(q_home)
        (debug / "last_reset.json").write_text(json.dumps({"object": args.object, "spot": spot.tolist(), "time": time.time()}))
    finally:
        if arm is not None:
            arm.close()
        rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
