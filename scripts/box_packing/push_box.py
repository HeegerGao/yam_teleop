"""Pull the cardboard box closer to the arm, so its contents can be lifted straight out.

Why this exists: taking an object out of a box needs a *vertical* lift -- tilt the wrist while
the object hangs below the fingers and it sweeps into the wall. With the wrist vertical this
arm runs joint 4 into its limit a few centimetres up, and how many centimetres depends
entirely on how far out the grasp is: ~10 cm of headroom at x = 0.36 m, 3 cm at x = 0.44,
none past x = 0.47. So when the box sits too far out, move the box.

How: the gripper pinches the *near wall* of the box (one finger inside, one outside, exactly
the cup-rim grasp applied to cardboard), drags it towards the base along the table, lets go,
lifts clear and returns to the working pose. Pinching beats pushing -- a push needs a contact
point at the far wall, 25 cm further out, where the arm is at the end of its reach.

    python scripts/box_packing/push_box.py --object "white cup"            # plan only
    python scripts/box_packing/push_box.py --object "white cup" --execute
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import scan_table as st  # noqa: E402
from arm import Arm, Workspace  # noqa: E402
from cameras import Frame  # noqa: E402
from collide import scan_obstacles  # noqa: E402
from perception import Scene  # noqa: E402
from pick_object import PORTS, vertical_headroom  # noqa: E402
from policies import TableObject, load_objects  # noqa: E402
from poses import go_working, named  # noqa: E402

DEBUG = _HERE / "calib" / "debug"


@dataclass
class Args:
    object: str = "white cup"
    """The object whose extraction has to become possible. Empty: judge by the box centre."""
    side: str = "left"
    execute: bool = False
    rescan: bool = True
    lift_needed: float = 0.01
    """The object's bottom must clear the rim by this much on a purely vertical lift."""
    margin: float = 0.005
    """Extra headroom asked for beyond the strict requirement. Kept small on purpose: the
    distance below is already the minimum that works, and every extra centimetre drags the
    box further out of the camera's view."""
    trim: float = 0.0
    """Subtract this from the computed distance: the estimate is the *minimum* that buys the
    headroom, and overshooting drags the box further out of the camera's view than needed."""
    max_push: float = 0.16
    """Never drag further than this in one go."""
    grip_below_rim: float = 0.022
    """Fingertip centre this far below the wall's top edge while dragging."""
    drag_gripper: float = 0.0
    """How open the fingers are for the drag. Closed: a narrow tool bearing on the wall."""
    inset: float = 0.015
    """Contact this far inside the far wall."""
    contact_above_floor: float = 0.02
    """Height of the fingertips over the box floor while dragging."""
    approach: float = 0.09
    transit_clear: float = 0.16
    speed: float = 0.05
    """Drag speed, m/s."""
    max_joint_vel: float = 0.5
    contact_abort_rad: float = 0.2
    """Loose: dragging a box *is* a sustained contact, so the usual lag limit does not apply."""
    front_clear: float = 0.03
    """Refuse if something stands within this of the strip the box will sweep through."""
    start_at_working: bool = True


def push_needed(arm: Arm, obj: TableObject, box: TableObject, args: Args, u: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, float]:
    """(distance to drag along ``u``, that direction, headroom needed).

    The grasp height is taken 2.2 cm below the object's top, as the policies do; the distance
    is the smallest shift that buys enough vertical headroom there. ``u`` defaults to the
    radial direction from the base to the box.
    """
    if u is None:
        u = box.center[:2] / max(np.linalg.norm(box.center[:2]), 1e-9)
    p = obj.center[:2] if obj is not None else box.center[:2]
    z = (obj.top_z - 0.022) if obj is not None else (box.table_z + 0.03)
    bottom = obj.table_z + 0.006 if obj is not None else box.table_z
    need = box.top_z + args.lift_needed + (z - bottom) - z + args.margin
    for d in np.arange(0.0, args.max_push + 0.005, 0.01):
        target = np.array([p[0] - d * u[0], p[1] - d * u[1], z])
        best = 0.0
        for yaw in np.linspace(0, np.pi, 8, endpoint=False):
            ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            try:
                q6, R = arm.kin.ik_grasp(target, np.array([0.0, 0.0, -1.0]), arm.q(), finger_axis=ax)
            except ValueError:
                continue
            best = max(best, vertical_headroom(arm, target, R, q6))
            if best >= need:
                break
        if best >= need:
            return float(d), u, float(need)
    return float(args.max_push), u, float(need)


def far_wall_point(box: TableObject, u: np.ndarray, inset: float = 0.015) -> Tuple[np.ndarray, float]:
    """A contact point just inside the box's *far* wall, and that wall's distance along u.

    The far wall is the one the camera sees best; the near wall runs off the bottom of the
    frame once the box has been pulled in, and a contact point derived from a truncated mask
    lands in mid-air inside the box -- which is exactly what made a drag do nothing. Laterally
    the contact is at the box's own centre, so the box slides instead of spinning.
    """
    rim_pts = box.xyz[box.height >= box.height.max() - 0.012]
    v = np.array([-u[1], u[0]])
    along = rim_pts[:, :2] @ u if len(rim_pts) >= 100 else box.xyz[:, :2] @ u
    edge = float(np.percentile(along, 97))
    across = float(box.center[:2] @ v)
    return (edge - inset) * u + across * v, edge


def near_wall_point(box: TableObject, u: np.ndarray) -> Tuple[np.ndarray, float]:
    """Mid-point of the box's near wall (the one facing the base), measured rather than modelled.

    The footprint model over-states the box (its flaps belong to the mask, and the near edge
    can run off the bottom of the frame), so the wall is found in the point cloud instead: of
    the points at rim height, the ones nearest the base along ``u`` are the near rim, and
    their median across ``u`` is where to pinch.
    """
    rim_pts = box.xyz[box.height >= box.height.max() - 0.012]
    if len(rim_pts) < 100:
        ax_long, ax_short = box.long_axis()[:2], box.short_axis()[:2]
        extent = abs(box.size[0] / 2 * float(ax_long @ u)) + abs(box.size[1] / 2 * float(ax_short @ u))
        return box.center[:2] - extent * u, float(extent)
    v = np.array([-u[1], u[0]])
    along = rim_pts[:, :2] @ u
    edge = float(np.percentile(along, 3))
    # laterally, contact the box's *centre*: pulling off to one side spins the box instead of
    # sliding it (the rim points visible near the edge are a biased sample -- the near part of
    # the box runs off the bottom of the frame)
    across = float(box.center[:2] @ v)
    p = edge * u + across * v
    return p, float(np.linalg.norm(box.center[:2]) - edge)


def lateral_half_width(box: TableObject, u: np.ndarray) -> float:
    """Half the box's width across the pull direction (its own footprint, not the bigger side)."""
    v = np.array([-u[1], u[0]])
    return abs(box.size[0] / 2 * float(box.long_axis()[:2] @ v)) + abs(box.size[1] / 2 * float(box.short_axis()[:2] @ v))


def sweep_is_clear(objs, box: TableObject, u: np.ndarray, dist: float, args: Args, quiet: bool = False) -> bool:
    """Nothing (other than the box and what is inside it) may stand where the box will go."""
    half_w = lateral_half_width(box, u)
    ok = True
    for o in objs.values():
        if o is box:
            continue
        d = o.center[:2] - box.center[:2]
        along = float(d @ u)
        across = abs(float(d @ np.array([-u[1], u[0]])))
        if across > half_w + args.front_clear:
            continue
        # in the strip: it is a problem when it lies in front of the box (towards the base)
        extent_u = abs(box.size[0] / 2 * float(box.long_axis()[:2] @ u)) + abs(box.size[1] / 2 * float(box.short_axis()[:2] @ u))
        if -dist - extent_u - args.front_clear < along < -extent_u * 0.85:
            if not quiet:
                print(f"[sweep] {o.name} at {np.round(o.center[:2], 3)} stands in the box's way")
            ok = False
    return ok


def main(args: Args) -> None:
    scene = Scene.from_calib(args.side)
    if args.execute and args.start_at_working:
        a0 = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel)
        try:
            go_working(a0)
        finally:
            a0.close()
    if args.rescan:
        st.main(st.Args(raise_arms=False))
    objs = load_objects(DEBUG / "scan_result.json", DEBUG / "scan_masks.png", DEBUG, scene)
    box = next((o for o in objs.values() if o.name == "cardboard box"), None)
    if box is None:
        raise SystemExit("[error] no cardboard box in the scan")
    obj = None
    if args.object:
        hits = [o for o in objs.values() if args.object.lower() in o.name.lower() and o is not box]
        obj = max(hits, key=lambda o: o.rec.get("name_score", 0.0)) if hits else None
        if obj is None:
            print(f"[warn] '{args.object}' not found; judging by the box centre instead")

    arm = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel, workspace=Workspace(z=(box.table_z + 0.005, 0.6)), execute=args.execute)
    try:
        u_rad = box.center[:2] / max(np.linalg.norm(box.center[:2]), 1e-9)
        options = []
        for name, u_try in (("radial", u_rad), ("-x", np.array([1.0, 0.0])), ("mid", (u_rad + np.array([1.0, 0.0])) / np.linalg.norm(u_rad + np.array([1.0, 0.0])))):
            d, _, nd = push_needed(arm, obj, box, args, u_try)
            d = max(0.0, d - args.trim)
            clear = sweep_is_clear(objs, box, u_try, d, args, quiet=True)
            options.append((clear, d, name, u_try, nd))
            print(f"[option] pull along {name:6s} {np.round(u_try, 2)}: {d * 100:4.1f} cm, path {'clear' if clear else 'BLOCKED'}")
        viable = [o for o in options if o[0]]
        if not viable:
            sweep_is_clear(objs, box, u_rad, options[0][1], args)  # print what is in the way
            raise SystemExit("[error] every pull direction is blocked -- move what is in front of the box")
        _, dist, name, u, need = min(viable, key=lambda o: o[1])
        print(f"[pull] {name} it is")
        where = f"{obj.name} at {np.round(obj.center[:2], 3)}" if obj is not None else f"box centre {np.round(box.center[:2], 3)}"
        print(f"[box] centre {np.round(box.center, 3)}, size {np.round(box.size * 100, 1)} cm, rim z {box.top_z:.3f}")
        print(f"[need] {where} needs {need * 100:.1f} cm of vertical lift -> drag the box {dist * 100:.1f} cm towards the base along {np.round(u, 2)}")
        if dist <= 0.005:
            print("[done] the box is already close enough; nothing to do")
            return
        wall_xy, edge = far_wall_point(box, u, args.inset)
        floor = box.table_z + 0.006
        grip = np.array([wall_xy[0], wall_xy[1], floor + args.contact_above_floor])
        target = np.array([grip[0] - dist * u[0], grip[1] - dist * u[1], grip[2]])
        print(f"[contact] inside the far wall at {np.round(wall_xy, 3)} (wall at {edge * 100:.1f} cm along u), fingers at z {grip[2]:.3f} ({args.contact_above_floor * 100:.0f} cm over the floor)")
        print(f"[drag]  {np.round(grip[:2], 3)} -> {np.round(target[:2], 3)}")

        # closed fingers: a compact tool that bears on the wall's inner face and cannot catch
        # the rim. The wrist tilts as far as it must to reach the far side of the box.
        u3 = np.array([u[0], u[1], 0.0])
        q_grip, R, tilt = arm.kin.ik_reach(grip, arm.q(), finger_axis=u3)
        print(f"[contact] wrist tilt {tilt:.0f} deg")
        q_pre, p_pre, h = None, None, 0.0
        from pick_object import solve_approach

        q_pre, p_pre, h = solve_approach(arm, grip, R, q_grip, args.approach)
        q_end, perr, rerr = arm.kin.ik(target, R, q_grip)
        if perr > 5e-3 or rerr > 0.05:
            raise SystemExit(f"[error] the end of the drag is not reachable ({perr * 1e3:.0f} mm)")
        q_transit, R_t, tilt_t = arm.kin.ik_reach(np.array([grip[0], grip[1], box.table_z + args.transit_clear]), arm.q(), finger_axis=u3)
        print(f"[plan] transit -> pre-grip {h * 100:.0f} cm up -> pinch -> drag -> release -> up -> working")

        frame = Frame.load(DEBUG, "scan_frame", role="top")
        img = cv2.imread(str(DEBUG / "scan_overlay.png"))
        for p, color, label in ((grip, (0, 255, 255), "pinch"), (target, (255, 0, 255), "drag to")):
            uv = frame.project(scene.to_cam(p))[0]
            cv2.drawMarker(img, (int(uv[0]), int(uv[1])), color, cv2.MARKER_CROSS, 26, 2)
            cv2.putText(img, label, (int(uv[0]) + 10, int(uv[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(DEBUG / "push_plan.png"), img)
        if not args.execute:
            print("[dry run] pass --execute to move (plan drawn in calib/debug/push_plan.png)")
            return

        def q7(q6: np.ndarray) -> np.ndarray:
            return np.concatenate([q6, [arm.q_cmd[6]]])

        arm.set_gripper(args.drag_gripper, wait=0.0)
        arm.move_through([q7(q_transit)])
        arm.wait_settled()
        arm.move_joints(q7(q_pre), settle=0.0)
        arm.wait_settled()
        sag = arm.grasp_pose()[0] - arm.kin.grasp(arm.q_cmd)[0]
        print(f"       sag at the pre-grip {np.round(sag * 1e3, 0)} mm")
        arm.move_linear(grip - sag, R, speed=0.05, correct=0, track_abort_rad=args.contact_abort_rad)
        print(f"[move] fingers straddle the wall at {np.round(arm.grasp_pose()[0], 3)}, open {arm.gripper():.2f}")
        # No pinching: the fingers stay open, straddling the wall, and the inner one bears
        # against its inside face as the arm moves back. One less thing to get wrong, and
        # nothing to release afterwards.
        res = arm.move_linear(target - sag, R, speed=args.speed, correct=0, track_abort_rad=args.contact_abort_rad)
        print(f"[move] dragged to {np.round(arm.grasp_pose()[0], 3)} {res.aborted}")
        q_up, perr, rerr = arm.kin.ik(np.array([target[0], target[1], target[2] + 0.06]), R, arm.q_cmd)
        if perr < 5e-3:
            arm.move_joints(q7(q_up), settle=0.0)
        arm.move_through([named("working")])
        arm.wait_settled()
        print("[done] box moved; back at the working pose")
    finally:
        arm.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
