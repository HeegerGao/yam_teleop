"""Pick one scanned object with a per-object policy and place it inside the cardboard box.

  1. scan the table (scan_table.py machinery; the arms at home are outside the camera's view,
     --raise lifts them first if something was left over the table)
  2. pick the object by name (--object) -> policies.plan_grasp gives the grasp for that kind
     of object (a cup is pinched by its rim, a carrot across its short side, a hammer by its
     handle, ...); --style forces a policy
  3. --place says where in the box, as an offset from the box centre in *image* directions
     (right/down, metres) -- "5 cm below the red dot" is ``--place 0 0.05``
  4. the arm is chosen by which half of the table the object is on (--arm auto), and the
     whole motion is pre-solved and streamed without pauses (as pick_place.py)

    python scripts/box_packing/pick_object.py --object "white cup" --style wall --place 0 0.05
    python scripts/box_packing/pick_object.py --object carrot --place -0.05 0 --execute
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
from cameras import CameraRig  # noqa: E402
from collide import check_path, scan_obstacles  # noqa: E402
from perception import Scene  # noqa: E402
from poses import named  # noqa: E402
from policies import NOT_PICKABLE, Grasp, PlaceSpec, TableObject, load_objects, plan_grasp  # noqa: E402

PORTS = {"left": 1235, "right": 1234}
DEBUG = _HERE / "calib" / "debug"


@dataclass
class Args:
    object: str = "white cup"
    """Name as reported by scan_table.py (a substring works: 'cup')."""
    style: Optional[str] = None
    """Force a grasp policy: wall | topdown. Default: the object's own policy."""
    place: Tuple[float, float] = (0.0, 0.05)
    """Release point: (right, down) metres from the box centre in top-image directions."""
    place_clear: float = 0.06
    """Object bottom this far above the box floor at release."""
    arm: str = "auto"
    """left | right | auto (auto: by the object's side of the table, --midline-y)."""
    midline_y: float = -0.30
    """Left-base-frame y of the table's midline; objects with y below it go to the right arm."""
    execute: bool = False
    start_at_working: bool = True
    """Move the arm to the working pose before scanning (it must not be over the table)."""
    rescan: bool = True
    """Take a fresh scan (arms are at home, outside the view). --no-rescan reuses the last."""
    raise_arms: bool = False
    """Raise both arms before scanning (only needed if an arm is parked over the table)."""
    approach_override: Optional[float] = None
    transit_clear: float = 0.16
    max_joint_vel: float = 0.6
    descend_speed: float = 0.06
    contact_abort_rad: float = 0.15
    close_wait: float = 0.3
    open_wait: float = 0.3
    final_pose: str = "working"
    """Where the arm goes when the task is done: working | home | hold (stay put)."""
    place_probe: float = 0.04
    """Height above the release point where the arm's sag is measured before the final descent."""


def solve_approach(arm: Arm, p: np.ndarray, R: np.ndarray, q_seed: np.ndarray, approach: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Highest point straight above ``p`` (up to ``approach``) reachable with orientation ``R``.

    Near the far edge of the workspace a vertical wrist runs joint 4 into its limit after a
    couple of centimetres, so this returns how far it *can* go rather than failing.
    """
    for h in np.arange(approach, 0.0049, -0.005):
        q6, perr, rerr = arm.kin.ik(p + np.array([0, 0, h]), R, q_seed)
        if perr < 5e-3 and rerr < 0.05:
            return q6, p + np.array([0, 0, h]), float(h)
    raise ValueError("no reachable point above the grasp with its orientation")


def vertical_headroom(arm: Arm, p: np.ndarray, R: np.ndarray, q_seed: np.ndarray, limit: float = 0.14, step: float = 0.01) -> float:
    """How far straight up the arm can carry a grasp at ``p`` holding orientation ``R``."""
    best = 0.0
    for h in np.arange(step, limit, step):
        q6, perr, rerr = arm.kin.ik(p + np.array([0, 0, h]), R, q_seed)
        if perr < 5e-3 and rerr < 0.05:
            best = float(h)
        else:
            break
    return best


def choose_arm(args: Args, obj: TableObject) -> str:
    if args.arm in ("left", "right"):
        return args.arm
    side = "left" if obj.center[1] >= args.midline_y else "right"
    if side == "right" and not (_HERE / "calib" / "top_right.json").exists():
        print("[arm] object is on the right half but the right arm has no calibration yet (calib/top_right.json) -- using the left arm")
        side = "left"
    return side


def scene_for(side: str) -> Scene:
    return Scene.from_calib(side)


def main(args: Args) -> None:
    scene = scene_for("left")
    # ---- scan -------------------------------------------------------------------------
    # start from the working pose: an arm left over the table occludes the scan and leaks
    # into the object masks (a box then measures 18 cm tall)
    if args.execute and args.start_at_working:
        from arm import Arm as _Arm
        from poses import go_working

        _a = _Arm("left", PORTS["left"], max_joint_vel=args.max_joint_vel)
        try:
            go_working(_a)
        finally:
            _a.close()
    if args.rescan:
        sargs = st.Args(raise_arms=args.raise_arms, return_home=True)
        st.main(sargs)
    objs = load_objects(DEBUG / "scan_result.json", DEBUG / "scan_masks.png", DEBUG, scene)
    box = next((o for o in objs.values() if o.name == "cardboard box"), None)
    hits = [o for o in objs.values() if args.object.lower() in o.name.lower() and o.name not in NOT_PICKABLE]
    if box is None or not hits:
        raise SystemExit(f"[error] box found: {box is not None}; '{args.object}' found: {len(hits)} (names: {sorted(o.name for o in objs.values())})")
    obj = max(hits, key=lambda o: o.rec.get("name_score", 0.0))
    print(f"[object] #{obj.rec['id']} {obj.name}: centre {np.round(obj.center, 3)}, top z {obj.top_z:.3f}, size {np.round(obj.size * 100, 1)} cm, yaw {np.degrees(obj.yaw):.0f}")
    print(f"[box] #{box.rec['id']} centre {np.round(box.center, 3)}, rim z {box.top_z:.3f}, size {np.round(box.size * 100, 1)} cm, yaw {np.degrees(box.yaw):.0f}")

    # ---- grasp + place ----------------------------------------------------------------
    grasp = plan_grasp(obj, scene, args.style, others=list(objs.values()))
    if args.approach_override is not None:
        grasp.approach = args.approach_override
    clear = args.place_clear
    if grasp.vertical_only and args.place_clear > 0.005:
        # a rim-held cup hangs tilted from one wall: released with any height it lands on its
        # rim, tips and rolls -- it has to be set down with its bottom touching the floor
        clear = 0.005
        print(f"[place] rim grasp: release clearance lowered to {clear * 1000:.0f} mm (set down, not dropped)")
    spec = PlaceSpec(right=args.place[0], down=args.place[1], clear=clear)
    p_floor = spec.resolve(box, scene)
    r_ax, d_ax = spec.image_axes(scene)
    print(f"[grasp] {obj.name}: {grasp.note}; fingertips at {np.round(grasp.pos, 3)}, finger axis {np.round(grasp.finger_axis, 2)}, vertical only {grasp.vertical_only}")
    print(f"[place] image right axis {np.round(r_ax, 2)}, down axis {np.round(d_ax, 2)}; release over ({p_floor[0]:.3f}, {p_floor[1]:.3f}), {'inside' if spec.inside(box, p_floor) else 'NOT SAFELY INSIDE'} the box")
    if not spec.inside(box, p_floor):
        raise SystemExit("[error] the release point is too close to a box wall -- change --place")

    side = choose_arm(args, obj)
    print(f"[arm] {side}")
    # overlay of the plan on the scan image
    img = cv2.imread(str(DEBUG / "scan_overlay.png"))
    from cameras import Frame

    frame = Frame.load(DEBUG, "scan_frame", role="top")
    for p, color, label in ((grasp.pos, (0, 255, 255), "grasp"), (p_floor, (255, 0, 255), "release")):
        uv = frame.project(scene.to_cam(p))[0]
        cv2.drawMarker(img, (int(uv[0]), int(uv[1])), color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, label, (int(uv[0]) + 10, int(uv[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(DEBUG / "pick_plan.png"), img)

    # ---- waypoints ----------------------------------------------------------------------
    transit_z = obj.table_z + args.transit_clear
    place_z = p_floor[2] + spec.clear + grasp.grip_above_bottom
    # Wrist yaw over the box: the open fingertips (+-4.9 cm along the finger axis from the
    # tip centre) go below the rim at release, so they must stay clear of the walls -- the
    # scanned box points are the real walls (the footprint model includes the flaps). The
    # held object sits at tips + body_offset, and that offset turns with the wrist, so each
    # candidate yaw aims the tips at release_point - R(yaw) body_offset and is scored by the
    # nearer finger's distance to the box. The grasp yaw is kept when it is already fine.
    walls = box.xyz[:, :2]
    a0 = float(np.arctan2(grasp.finger_axis[1], grasp.finger_axis[0]))
    best = None
    for d_yaw in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        ang = a0 + d_yaw
        ax = np.array([np.cos(ang), np.sin(ang), 0.0])
        c, s_ = np.cos(d_yaw), np.sin(d_yaw)
        off = np.array([c * grasp.body_offset[0] - s_ * grasp.body_offset[1], s_ * grasp.body_offset[0] + c * grasp.body_offset[1]])
        tips = p_floor[:2] - off
        clear = min(float(np.min(np.linalg.norm(walls - (tips + sgn * 0.049 * ax[:2]), axis=1))) for sgn in (-1.0, 1.0))
        score = min(clear, 0.06) - 0.02 * abs(d_yaw) / np.pi  # prefer little wrist turning
        if best is None or score > best[0]:
            best = (score, ax, tips, clear, d_yaw)
    _, axis_box, tips_xy, clear, d_yaw = best
    if clear < 0.025:
        raise SystemExit(f"[error] no wrist yaw keeps the open fingers {clear * 100:.1f} cm clear of the box walls at that release point -- move --place inwards")
    print(f"[place] wrist turned {np.degrees(d_yaw):+.0f} deg over the box: nearer finger {clear * 100:.1f} cm from the box walls")
    p_floor = np.array([tips_xy[0], tips_xy[1], p_floor[2]])
    wps: List[Tuple[str, np.ndarray]] = [
        ("transit above object", np.array([grasp.pos[0], grasp.pos[1], transit_z])),
        ("pre-grasp", grasp.pos + np.array([0, 0, grasp.approach])),
        ("grasp", grasp.pos.copy()),
        ("lift", np.array([grasp.pos[0], grasp.pos[1], transit_z])),
        ("transit above box", np.array([p_floor[0], p_floor[1], transit_z])),
        ("place", np.array([p_floor[0], p_floor[1], place_z])),
        ("retreat", np.array([p_floor[0], p_floor[1], transit_z])),
    ]
    print("[plan]")
    for n, p in wps:
        print(f"  {n:22s} {np.round(p, 3)}")

    arm = Arm(side, PORTS[side], max_joint_vel=args.max_joint_vel, workspace=Workspace(z=(obj.table_z + 0.005, 0.6)), execute=args.execute)
    q_final = named(args.final_pose) if args.final_pose in ("working", "home") else None
    q_prev = arm.q().copy()  # seed the IK chain from where the arm actually is
    solved: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    for n, p in wps:
        ax = axis_box if n in ("transit above box", "place", "retreat") else grasp.finger_axis
        tilts = (0.0,) if (n == "grasp" and grasp.vertical_only) else (0.0, 20.0, 35.0, 50.0, 65.0, 80.0)
        q6, R, tilt = arm.kin.ik_reach(p, q_prev, tilts=tilts, finger_axis=ax)
        solved[n] = (q6, R, tilt)
        print(f"  ik ok: {n:22s} tilt {tilt:.0f} deg  q {np.round(q6, 2)}")
        q_prev = np.concatenate([q6, [q_prev[6]]])
    q6, p_pre, h = solve_approach(arm, grasp.pos, solved["grasp"][1], solved["grasp"][0], grasp.approach)
    solved["pre-grasp"] = (q6, solved["grasp"][1], solved["grasp"][2])
    wps[1] = ("pre-grasp", p_pre)
    print(f"  pre-grasp re-solved {h * 100:.0f} cm above the grasp at the grasp tilt ({solved['grasp'][2]:.0f} deg)")
    # the two big swings -- into the workspace and back out -- must clear everything the scan
    # found, except the object we are going to touch and the box we deliberately reach into
    obstacles = scan_obstacles(scene, skip_ids=(obj.rec["id"], box.rec["id"]))
    q_now = arm.q()
    ok = check_path(arm.kin, q_now, np.concatenate([solved["transit above object"][0], [1.0]]), obstacles, "approach")
    if q_final is not None:
        ok &= check_path(arm.kin, np.concatenate([solved["retreat"][0], [1.0]]), q_final, obstacles, "return")
    # the carry curve is checked where it actually flies, not along the straight line
    from collide import samples_clearance

    q1 = np.concatenate([solved["grasp"][0], [0.0]])
    carry = arm.spline([q1, np.concatenate([solved["lift"][0], [0.0]]), np.concatenate([solved["transit above box"][0], [0.0]]), np.concatenate([solved["place"][0], [0.0]])], samples=20)
    if obstacles is not None and len(obstacles):
        gap, at = samples_clearance(arm.kin, carry, obstacles)
        print(f"[clear] carry curve: {gap * 100:+.1f} cm over the nearest object at {at * 100:.0f}% of the curve")
        ok &= gap >= 0.01
    if not ok:
        raise SystemExit("[error] a transit sweeps into an object -- refusing")

    if not args.execute:
        print("[dry run] pass --execute to move (plan drawn in calib/debug/pick_plan.png)")
        arm.close()
        return

    # ---- execute (pre-solved joint targets, no settling) --------------------------------
    def q7(q6: np.ndarray) -> np.ndarray:
        return np.concatenate([q6, [arm.q_cmd[6]]])

    def report(n: str, p: np.ndarray, res: MoveResult) -> None:
        p_now = arm.grasp_pose()[0]
        print(f"[move] {n:22s} -> {np.round(p, 3)}  at {np.round(p_now, 3)}  err {np.linalg.norm(p_now - p) * 1e3:.1f} mm {res.aborted}")
        if res.aborted and res.aborted != "ik on correction":
            raise RuntimeError(f"aborted at {n}: {res.aborted}")

    targets = dict(wps)
    # the probe pose above the release point, solved now so the flight there is one curve
    p_probe = targets["place"] + np.array([0.0, 0.0, args.place_probe])
    q_probe, perr, rerr = arm.kin.ik(p_probe, solved["place"][1], solved["place"][0])
    if perr > 5e-3 or rerr > 0.05:
        q_probe = solved["place"][0]
    wrist_role = "left_wrist" if side == "left" else "right_wrist"
    rig = CameraRig.open(roles=("top", wrist_role))
    try:
        arm.set_gripper(1.0, wait=0.0)

        def q7(q6: np.ndarray) -> np.ndarray:
            return np.concatenate([q6, [arm.q_cmd[6]]])

        def report(n: str, p: np.ndarray, res: MoveResult) -> None:
            p_now = arm.grasp_pose()[0]
            print(f"[move] {n:22s} -> {np.round(p, 3)}  at {np.round(p_now, 3)}  err {np.linalg.norm(p_now - p) * 1e3:.1f} mm {res.aborted}")
            if res.aborted and res.aborted != "ik on correction":
                raise RuntimeError(f"aborted at {n}: {res.aborted}")

        # --- fly to the object: one curve through the transit height to the pre-grasp ------
        report("pre-grasp", targets["pre-grasp"], arm.move_through([q7(solved["transit above object"][0]), q7(solved["pre-grasp"][0])]))
        arm.wait_settled()  # the descent must start from where the plan thinks the arm is
        R_g = solved["grasp"][1]
        # The stretched arm sags 1-2 cm; descending from there puts a finger beside the object
        # instead of on it. Measure the sag here and aim the descent so the *measured*
        # fingertips land on the grasp point.
        sag_g = arm.grasp_pose()[0] - arm.kin.grasp(arm.q_cmd)[0]
        print(f"       sag at the pre-grasp {np.round(sag_g * 1e3, 0)} mm")
        cv2.imwrite(str(DEBUG / "grasp_wrist_before_close.png"), rig.grab_fresh(wrist_role).color)
        res = arm.move_linear(grasp.pos - sag_g, R_g, speed=args.descend_speed, correct=1, track_abort_rad=args.contact_abort_rad)
        report("grasp", grasp.pos, res)
        if np.linalg.norm(arm.grasp_pose()[0] - grasp.pos) > 0.025:
            print("[abort] the descent did not reach the grasp point (contact?) -- backing out without closing")
            arm.move_joints(q7(solved["lift"][0]), settle=0.0)
            if q_final is not None:
                arm.move_joints(q_final, settle=0.0)
            return
        arm.set_gripper(grasp.close, wait=args.close_wait)
        print(f"       gripper closed to {arm.gripper():.3f} (thin objects read close to 0 -- confirm with the camera)")
        cv2.imwrite(str(DEBUG / "grasp_top_after_close.png"), rig.grab_fresh("top").color)

        # --- carry: up out of the object and one curve over the box to the release probe ---
        q6, perr2, rerr2 = arm.kin.ik(targets["lift"], R_g, arm.q_cmd)
        if perr2 > 5e-3 or rerr2 > 0.05:
            q6 = solved["lift"][0]
        report("above the box", p_probe, arm.move_through([q7(q6), q7(solved["transit above box"][0]), q7(q_probe)]))
        arm.wait_settled()
        sag = arm.grasp_pose()[0] - arm.kin.grasp(arm.q_cmd)[0]
        print(f"       sag above the release point {np.round(sag * 1e3, 0)} mm")
        res = arm.move_linear(targets["place"] - sag, solved["place"][1], speed=args.descend_speed, correct=0)
        report("place", targets["place"], res)
        arm.set_gripper(1.0, wait=args.open_wait)
        # straight up out of the box, then one curve home
        # a joint move, not a Cartesian one: it is the same near-vertical path (same xy) but
        # paced by the slew limit, and the wrist re-orientation on the way out is large enough
        # that a fixed Cartesian speed outruns what the arm can track
        report("retreat", targets["retreat"], arm.move_joints(q7(solved["retreat"][0]), settle=0.0))
        arm.wait_settled()  # do not swing away from a pose the arm has not reached yet
        if q_final is not None:
            print(f"[done] to the {args.final_pose} pose")
            arm.move_through([q_final])
        (DEBUG / "last_pick.json").write_text(json.dumps({"object": obj.name, "start_xy": obj.center[:2].tolist(), "time": time.time()}))
    finally:
        arm.close()
        rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
