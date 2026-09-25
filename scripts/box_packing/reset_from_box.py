"""Reset policy: take an object out of the cardboard box and put it back where it started
(a random free spot near its recorded start position), without touching the box.

  1. fresh scan; the object must be inside the box's footprint
  2. its grasp policy runs with ``container=box``: for the cup's rim pinch that means only rim
     points where both open fingertips stay >= 1 cm from the inner walls are allowed (the
     gripper body rides above the rim -- only the fingers enter the box), the wrist vertical
  3. straight down into the box, close, straight up (Cartesian, no sideways motion inside the
     box) to above the rim, then over to the reset spot
  4. the reset spot is sampled within --reset-radius of the start position recorded by
     pick_object.py (calib/debug/last_pick.json) and must be free of other objects (1 cm
     occupancy grid from the scan); the object is set down (rim grasp) or released low

    python scripts/box_packing/reset_from_box.py --object "white cup" --style wall
    python scripts/box_packing/reset_from_box.py --object "white cup" --style wall --execute
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

import scan_table as st  # noqa: E402
from arm import Arm, MoveResult, Workspace  # noqa: E402
from cameras import CameraRig, Frame  # noqa: E402
from calibrate_top import ArmSurface  # noqa: E402
from collide import check_curve, check_path, scan_obstacles  # noqa: E402
from perception import Scene  # noqa: E402
from poses import named  # noqa: E402
from pick_object import PORTS, solve_approach, vertical_headroom  # noqa: E402
from policies import TableObject, inside_container, load_objects, plan_grasp, wall_clearance  # noqa: E402

DEBUG = _HERE / "calib" / "debug"


@dataclass
class Args:
    object: str = "white cup"
    style: Optional[str] = "topdown"
    """Grasp policy for the lift-out: topdown (pinch the body from two sides -- tolerant to a
    couple of cm of error, the right choice inside a box) | wall | None for the object's own.""" 
    reset_center: Optional[Tuple[float, float]] = None
    """Base-frame xy to return to. Default: the start position in calib/debug/last_pick.json."""
    reset_radius: float = 0.05
    seed: Optional[int] = None
    clear_along: float = 0.075
    clear_across: float = 0.055
    execute: bool = False
    start_at_working: bool = True
    """Move the arm to the working pose before scanning (it must not be over the table)."""
    rescan: bool = True
    side: str = "left"
    transit_clear: float = 0.16
    rim_clear: float = 0.05
    """Straight-up lift inside the box ends this far above the rim before any sideways move."""
    place_clear: float = 0.005
    """Object bottom above the table at release (set down)."""
    max_joint_vel: float = 0.6
    descend_speed: float = 0.06
    contact_abort_rad: float = 0.15
    close_wait: float = 0.3
    open_wait: float = 0.3
    final_pose: str = "working"
    """Where the arm goes when the task is done: working | home | hold (stay put)."""
    place_probe: float = 0.04
    grasp_toward_base: float = 0.03
    """Shift the grasp point this far towards the arm's base, along the table. The hand-eye
    fit was made from landmarks on the open table and extrapolates into the box, where it
    reads a couple of cm too far out; this is the measured correction for that region."""
    grasp_image_down: float = 0.03
    """Extra shift along the top image's 'down' direction (what you see as nearer the bottom
    of the picture). Measured by eye against the hover: the hand-eye fit extrapolates into the
    box region and reads short there."""
    hover_only: bool = False
    """Fly to the pre-grasp above the object and stop there -- for checking the aim by eye."""
    approach_gripper: float = 0.8
    """How open the gripper is while it flies to the box and descends (1 = fully open, 9.6 cm).
    Raised automatically if the object needs more room."""
    top_check: bool = False
    """Correct the grasp with a top-camera measurement of the gripper at the pre-grasp. Off:
    its correction has been inconsistent run to run and fights --grasp-toward-base."""
    refine_max_shift: float = 0.04


def occupancy(objs: Dict[int, TableObject], skip_id: int, cell: float = 0.01) -> np.ndarray:
    """Centres of occupied 1 cm cells from every object's points except ``skip_id``."""
    pts = np.concatenate([o.xyz[:, :2] for i, o in objs.items() if i != skip_id])
    keys = np.unique(np.floor(pts / cell).astype(np.int64), axis=0)
    return (keys + 0.5) * cell


def spot_is_free(xy: np.ndarray, axis: np.ndarray, occ: np.ndarray, along: float, across: float) -> bool:
    d = occ - xy
    u = d @ axis[:2]
    v = d @ np.array([-axis[1], axis[0]])
    return not np.any((u / along) ** 2 + (v / across) ** 2 < 1.0)


def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    scene = Scene.from_calib(args.side)
    if args.reset_center is None:
        last = json.loads((DEBUG / "last_pick.json").read_text())
        args.reset_center = tuple(last["start_xy"])
        print(f"[reset] centre from last_pick.json ({last['object']}): {np.round(args.reset_center, 3)}")
    # start from the working pose: an arm left over the table occludes the scan and leaks
    # into the object masks (a box then measures 18 cm tall)
    if args.execute and args.start_at_working:
        from arm import Arm as _Arm
        from poses import go_working

        _a = _Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel)
        try:
            go_working(_a)
        finally:
            _a.close()
    if args.rescan:
        st.main(st.Args(raise_arms=False))
    objs = load_objects(DEBUG / "scan_result.json", DEBUG / "scan_masks.png", DEBUG, scene)
    box = next((o for o in objs.values() if o.name == "cardboard box"), None)
    if box is None:
        raise SystemExit("[error] no cardboard box in the scan")
    hits = [o for o in objs.values() if args.object.lower() in o.name.lower() and inside_container(o, box)]
    if not hits:
        raise SystemExit(f"[error] no '{args.object}' inside the box (names: {sorted(o.name for o in objs.values())})")
    obj = max(hits, key=lambda o: o.rec.get("name_score", 0.0))
    print(f"[object] #{obj.rec['id']} {obj.name} in the box at {np.round(obj.center, 3)}, top z {obj.top_z:.3f}, {wall_clearance(box, obj.center[:2]) * 100:.1f} cm from the nearest wall")
    print(f"[box] centre {np.round(box.center, 3)}, rim z {box.top_z:.3f}, size {np.round(box.size * 100, 1)} cm, yaw {np.degrees(box.yaw):.0f}")

    grasp = plan_grasp(obj, scene, args.style, others=list(objs.values()), container=box)
    if args.grasp_toward_base:
        d = grasp.pos[:2] / max(np.linalg.norm(grasp.pos[:2]), 1e-9)
        grasp.pos[:2] -= args.grasp_toward_base * d
        print(f"[grasp] shifted {args.grasp_toward_base * 100:.1f} cm towards the base -> {np.round(grasp.pos, 3)}")
    if args.grasp_image_down:
        from policies import PlaceSpec

        _, down = PlaceSpec().image_axes(scene)
        grasp.pos[:2] += args.grasp_image_down * down[:2]
        print(f"[grasp] shifted {args.grasp_image_down * 100:.1f} cm down-image along {np.round(down[:2], 2)} -> {np.round(grasp.pos, 3)}")
    print(f"[grasp] {grasp.note}; fingertips {np.round(grasp.pos, 3)}, finger axis {np.round(grasp.finger_axis, 2)}")
    tips = [grasp.pos[:2] + s * 0.049 * grasp.finger_axis[:2] for s in (-1, 1)]
    clear_walls = min(wall_clearance(box, t) for t in tips)
    print(f"[grasp] open fingertips clear the walls by {clear_walls * 100:.1f} cm")
    if wall_clearance(box, grasp.pos[:2]) < 0.0:
        raise SystemExit(
            f"[error] the grasp point {np.round(grasp.pos[:2], 3)} is outside the box footprint "
            f"(the object is at {np.round(obj.center[:2], 3)}). The manual shifts "
            f"(--grasp-toward-base {args.grasp_toward_base * 100:.0f} cm, --grasp-image-down "
            f"{args.grasp_image_down * 100:.0f} cm) add up to more than the object is from the wall -- "
            "they are corrections for the same bias and should not both be applied."
        )

    # ---- reset spot ---------------------------------------------------------------------
    occ = occupancy(objs, obj.rec["id"])
    axis_place = grasp.finger_axis
    c = np.asarray(args.reset_center)
    body_r = float(max(obj.size[:2]) / 2) + 0.012  # the object's own footprint plus margin

    def free(body: np.ndarray) -> bool:
        if wall_clearance(box, body) > -0.06:  # not within 6 cm of the box
            return False
        if grasp.vertical_only:
            # rim pinch: the inner finger is inside the object, only the outer one sticks out
            # (r + 4.9 cm along the radial) -- a disc for the body and a spot for that finger
            if np.any(np.linalg.norm(occ - body, axis=1) < body_r):
                return False
            outer = body + (body_r - 0.012 + 0.049) * axis_place[:2]
            return not np.any(np.linalg.norm(occ - outer, axis=1) < 0.02)
        return spot_is_free(body, axis_place, occ, args.clear_along, args.clear_across)

    spot = None
    for _ in range(600):
        r = args.reset_radius * np.sqrt(rng.random())
        a = rng.random() * 2 * np.pi
        xy = c + r * np.array([np.cos(a), np.sin(a)])
        if free(xy):
            spot = xy
            break
    if spot is None:
        raise SystemExit("[error] no free spot near the reset centre")
    print(f"[place] reset spot ({spot[0]:.3f}, {spot[1]:.3f}), {np.linalg.norm(spot - c) * 100:.1f} cm from the start position")
    # fingertips at release: object body at the spot -> fingertips = spot - body_offset
    p_tips_xy = spot - grasp.body_offset
    table_s = scene.table_z(spot[0], spot[1])
    place_z = table_s + args.place_clear + grasp.grip_above_bottom

    # plan overlay
    frame = Frame.load(DEBUG, "scan_frame", role="top")
    background = frame.depth.copy()
    img = cv2.imread(str(DEBUG / "scan_overlay.png"))
    for p, color, label in ((grasp.pos, (0, 255, 255), "grasp"), (np.array([spot[0], spot[1], table_s]), (255, 0, 255), "reset")):
        uv = frame.project(scene.to_cam(p))[0]
        cv2.drawMarker(img, (int(uv[0]), int(uv[1])), color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, label, (int(uv[0]) + 10, int(uv[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(DEBUG / "reset_plan.png"), img)

    # ---- waypoints ------------------------------------------------------------------------
    transit_z = obj.table_z + args.transit_clear
    above_rim = np.array([grasp.pos[0], grasp.pos[1], box.top_z + args.rim_clear + max(0.0, grasp.grip_above_bottom)])
    wps: List[Tuple[str, np.ndarray]] = [
        ("transit above box", np.array([grasp.pos[0], grasp.pos[1], transit_z])),
        ("pre-grasp", grasp.pos + np.array([0, 0, grasp.approach])),
        ("grasp", grasp.pos.copy()),
        ("lift", above_rim),
        ("transit above spot", np.array([p_tips_xy[0], p_tips_xy[1], transit_z])),
        ("place", np.array([p_tips_xy[0], p_tips_xy[1], place_z])),
        ("retreat", np.array([p_tips_xy[0], p_tips_xy[1], transit_z])),
    ]
    print("[plan]")
    for n, p in wps:
        print(f"  {n:22s} {np.round(p, 3)}")

    arm = Arm(args.side, PORTS[args.side], max_joint_vel=args.max_joint_vel, workspace=Workspace(z=(obj.table_z + 0.005, 0.6)), execute=args.execute)
    # Lifting an object straight out of a box is what limits this task: at the far edge of the
    # workspace a vertical wrist runs joint 4 into its limit after 2-3 cm. For a round-ish
    # object the finger axis is free, so pick the one with the most vertical headroom.
    if float(obj.size[0] / max(obj.size[1], 1e-3)) < 1.6:
        need = box.top_z + 0.02 + max(0.0, grasp.grip_above_bottom) - grasp.pos[2]
        best = None
        for yaw in np.linspace(0, np.pi, 12, endpoint=False):
            ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            try:
                q6, R = arm.kin.ik_grasp(grasp.pos, np.array([0.0, 0.0, -1.0]), arm.q(), finger_axis=ax)
            except ValueError:
                continue
            head = vertical_headroom(arm, grasp.pos, R, q6)
            if best is None or head > best[0]:
                best = (head, ax, head >= need)
        if best is not None:
            print(f"[grasp] finger axis chosen for lift headroom: {np.round(best[1], 2)} -> {best[0] * 100:.1f} cm of vertical lift (needs {need * 100:.1f} cm)")
            grasp.finger_axis = best[1]
            if best[0] < need:
                print("[warn] the arm cannot lift this straight out of the box at any wrist angle -- "
                      "the box is at the edge of its reach; move the box ~5 cm towards the arm")
    q_final = named(args.final_pose) if args.final_pose in ("working", "home") else None
    q_prev = arm.q().copy()  # seed the IK chain from where the arm actually is
    solved: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    for n, p in wps:
        if n in ("pre-grasp", "lift"):
            continue  # solved below, straight above the grasp with the grasp's own orientation
        tilts = (0.0,) if (grasp.vertical_only and n == "grasp") else (0.0, 20.0, 35.0, 50.0, 65.0, 80.0)
        q6, R, tilt = arm.kin.ik_reach(p, q_prev, tilts=tilts, finger_axis=grasp.finger_axis)
        solved[n] = (q6, R, tilt)
        print(f"  ik ok: {n:22s} tilt {tilt:.0f} deg  q {np.round(q6, 2)}")
        q_prev = np.concatenate([q6, [q_prev[6]]])
    R_g, q_g = solved["grasp"][1], solved["grasp"][0]
    q6, p_pre, h = solve_approach(arm, grasp.pos, R_g, q_g, grasp.approach)
    solved["pre-grasp"] = (q6, R_g, solved["grasp"][2])
    wps[1] = ("pre-grasp", p_pre)
    print(f"  pre-grasp re-solved {h * 100:.0f} cm above the grasp at the grasp tilt")
    # the lift: as high as the vertical wrist reaches straight above the grasp, at least 2 cm
    # over the rim (the object hangs grip_above_bottom below the fingertips)
    # Two-stage lift: straight up with the vertical wrist as far as it reaches (the object's
    # bottom must at least clear the rim), then on up to transit height letting the wrist
    # tilt -- a joint move straight above the grasp, well inside the walls.
    # Stage 2 tilts the fingers away from the base, which swings the cup body (held on its
    # base-facing wall) upwards, so the lowest point from then on is the cup's bottom right
    # under the fingertips; the vertical stage only has to bring the fingertips near the rim
    lift_min = box.top_z - 0.02 + max(0.0, grasp.grip_above_bottom)
    q6, p_lift, h = solve_approach(arm, grasp.pos, R_g, q_g, float(above_rim[2] - grasp.pos[2]))
    if p_lift[2] < lift_min:
        raise SystemExit(f"[error] cannot lift straight up to clear the rim ({p_lift[2]:.3f} < {lift_min:.3f})")
    solved["lift"] = (q6, R_g, solved["grasp"][2])
    wps[3] = ("lift", p_lift)
    print(f"  lift re-solved {h * 100:.0f} cm above the grasp (object bottom rim + {(p_lift[2] - grasp.grip_above_bottom - box.top_z) * 100:.1f} cm)")
    p_lift2 = np.array([grasp.pos[0], grasp.pos[1], transit_z])
    q6, R2, tilt2 = arm.kin.ik_reach(p_lift2, np.concatenate([q6, [1.0]]), finger_axis=grasp.finger_axis)
    solved["lift2"] = (q6, R2, tilt2)
    wps.insert(4, ("lift2", p_lift2))
    print(f"  ik ok: {'lift2':22s} tilt {tilt2:.0f} deg  q {np.round(q6, 2)}")
    # the two big swings -- into the workspace and back out -- must clear everything the scan
    # found, except the object we are going to touch and the box we deliberately reach into
    obstacles = scan_obstacles(scene, skip_ids=(obj.rec["id"],))  # the box included: the
    # flight in must clear its walls; only the vertical entry below is allowed inside them
    q_now = arm.q()
    ok = check_curve(arm.kin, arm, [np.concatenate([solved["transit above box"][0], [1.0]])], obstacles, "approach curve")
    if q_final is not None:
        ok &= check_path(arm.kin, np.concatenate([solved["retreat"][0], [1.0]]), q_final, obstacles, "return")
    from collide import samples_clearance

    carry = arm.spline(
        [
            np.concatenate([solved["lift"][0], [0.0]]),
            np.concatenate([solved["lift2"][0], [0.0]]),
            np.concatenate([solved["transit above spot"][0], [0.0]]),
            np.concatenate([solved["place"][0], [0.0]]),
        ],
        samples=20,
    )
    # the carry curve starts inside the box on purpose -- the vertical lift above already
    # guarantees the rim is cleared, so the box is not an obstacle for this one
    obstacles_carry = scan_obstacles(scene, skip_ids=(obj.rec["id"], box.rec["id"]))
    if obstacles_carry is not None and len(obstacles_carry):
        gap, at = samples_clearance(arm.kin, carry, obstacles_carry)
        print(f"[clear] carry curve: {gap * 100:+.1f} cm over the nearest object at {at * 100:.0f}% of the curve")
        ok &= gap >= 0.01
    if not ok:
        raise SystemExit("[error] a transit sweeps into an object -- refusing")

    if not args.execute:
        print("[dry run] pass --execute to move (plan drawn in calib/debug/reset_plan.png)")
        arm.close()
        return

    def q7(q6: np.ndarray) -> np.ndarray:
        return np.concatenate([q6, [arm.q_cmd[6]]])

    def report(n: str, p: np.ndarray, res: MoveResult) -> None:
        p_now = arm.grasp_pose()[0]
        print(f"[move] {n:22s} -> {np.round(p, 3)}  at {np.round(p_now, 3)}  err {np.linalg.norm(p_now - p) * 1e3:.1f} mm {res.aborted}")
        if res.aborted and res.aborted != "ik on correction":
            raise RuntimeError(f"aborted at {n}: {res.aborted}")

    targets = dict(wps)
    print("[plan] final:", {n: np.round(p, 3).tolist() for n, p in wps})
    p_probe = targets["place"] + np.array([0.0, 0.0, args.place_probe])
    q_probe, perr, rerr = arm.kin.ik(p_probe, solved["place"][1], solved["place"][0])
    if perr > 5e-3 or rerr > 0.05:
        q_probe = solved["place"][0]
    # Go in already part-closed: fully open fingers span 9.6 cm and are the thing most likely
    # to catch a box wall on the way down. Never closer than the object needs, though.
    need = (float(obj.size[1]) + 0.02) / 0.096 if not grasp.vertical_only else 0.0
    approach_g = float(np.clip(max(args.approach_gripper, need), 0.2, 1.0))
    print(f"[grip] entering the box at {approach_g:.2f} open ({approach_g * 9.6:.1f} cm between the fingers)")
    wrist_role = "left_wrist" if args.side == "left" else "right_wrist"
    rig = CameraRig.open(roles=("top", wrist_role))
    try:
        def q7(q6: np.ndarray) -> np.ndarray:
            return np.concatenate([q6, [arm.q_cmd[6]]])

        def report(n: str, p: np.ndarray, res: MoveResult) -> None:
            p_now = arm.grasp_pose()[0]
            print(f"[move] {n:22s} -> {np.round(p, 3)}  at {np.round(p_now, 3)}  err {np.linalg.norm(p_now - p) * 1e3:.1f} mm {res.aborted}")
            if res.aborted and res.aborted != "ik on correction":
                raise RuntimeError(f"aborted at {n}: {res.aborted}")

        arm.set_gripper(approach_g, wait=0.0)
        # --- curve to *above the rim* only ---------------------------------------------
        # The corner over the box must not be rounded: a spline through [transit, pre-grasp]
        # cuts it, bringing the fingers down while they are still outside the box -- straight
        # into the near wall. So the curve ends above the rim and the entry is a separate
        # vertical move.
        report("transit above box", targets["transit above box"], arm.move_through([q7(solved["transit above box"][0])]))
        arm.wait_settled()
        report("pre-grasp", targets["pre-grasp"], arm.move_joints(q7(solved["pre-grasp"][0]), settle=0.0))
        arm.wait_settled()
        if args.hover_only:
            print("[hover] holding at the pre-grasp; the intended grasp point is "
                  f"{np.round(grasp.pos, 3)} ({(grasp.pos[2] - box.table_z) * 100:.1f} cm above the table, "
                  f"{(grasp.pos[2] - obj.table_z - 0.006) * 100:.1f} cm above the box floor)")
            print("[hover] nothing else will move; correct me and I will re-plan")
            return
        R_g = solved["grasp"][1]
        if args.top_check:
            from pick_place import gripper_offset_top

            surface = ArmSurface(arm.kin, skip_bodies=("base", "link1", "link2", "link3", "link4", "link5"))
            off = gripper_offset_top(arm, rig, scene, background, surface, DEBUG, "reset_pregrasp")
            if off is not None and np.linalg.norm(off[:2]) <= args.refine_max_shift:
                grasp.pos[:2] -= off[:2]
                print(f"       applying xy correction {np.round(-off[:2] * 1e3, 0)} mm to the grasp")
        sag_g = arm.grasp_pose()[0] - arm.kin.grasp(arm.q_cmd)[0]
        print(f"       sag at the pre-grasp {np.round(sag_g * 1e3, 0)} mm")
        res = arm.move_linear(grasp.pos - sag_g, R_g, speed=args.descend_speed, correct=1, track_abort_rad=args.contact_abort_rad)
        report("grasp", grasp.pos, res)
        if np.linalg.norm(arm.grasp_pose()[0] - grasp.pos) > 0.025:
            print("[abort] descent blocked -- straight back up without closing")
            arm.move_linear(targets["lift"], R_g, speed=args.descend_speed, correct=0)
            if q_final is not None:
                arm.move_through([q_final])
            return
        cv2.imwrite(str(DEBUG / "reset_wrist_before_close.png"), rig.grab_fresh(wrist_role).color)
        arm.set_gripper(grasp.close, wait=args.close_wait)
        print(f"       gripper closed to {arm.gripper():.3f}")
        # --- straight up out of the box (no sideways motion inside the walls) --------------
        res = arm.move_linear(targets["lift"], R_g, speed=args.descend_speed, correct=0)
        report("lift", targets["lift"], res)
        # --- then one curve across to above the reset spot ---------------------------------
        report("above the spot", p_probe, arm.move_through([q7(solved["lift2"][0]), q7(solved["transit above spot"][0]), q7(q_probe)]))
        arm.wait_settled()
        sag = arm.grasp_pose()[0] - arm.kin.grasp(arm.q_cmd)[0]
        print(f"       sag above the release point {np.round(sag * 1e3, 0)} mm")
        res = arm.move_linear(targets["place"] - sag, solved["place"][1], speed=args.descend_speed, correct=0, track_abort_rad=args.contact_abort_rad)
        report("place", targets["place"], res)
        arm.set_gripper(1.0, wait=args.open_wait)
        # a joint move, not a Cartesian one: it is the same near-vertical path (same xy) but
        # paced by the slew limit, and the wrist re-orientation on the way out is large enough
        # that a fixed Cartesian speed outruns what the arm can track
        report("retreat", targets["retreat"], arm.move_joints(q7(solved["retreat"][0]), settle=0.0))
        arm.wait_settled()
        if q_final is not None:
            print(f"[done] to the {args.final_pose} pose")
            arm.move_through([q_final])
        (DEBUG / "last_reset.json").write_text(json.dumps({"object": obj.name, "spot": spot.tolist(), "time": time.time()}))
    finally:
        arm.close()
        rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
