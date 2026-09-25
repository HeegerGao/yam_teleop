"""Language-guided pick and place: name an object, an arm puts it in the box.

    python scripts/box_packing/language_pick_place.py                  # type names as you go
    python scripts/box_packing/language_pick_place.py --objects lemon red_cube saw
    python scripts/box_packing/language_pick_place.py --from-saved /tmp/snap --stem rig --objects lemon

The words it knows (lang_perception.VOCAB): elephant, bread, shovel, saw, white_cup, hammer,
lemon, bear, pink, blue_cube, red_cube -- plus a few aliases (cup, rabbit, teddy, hacksaw,
trowel ...). A whole sentence is fine: "put the lemon and the red cube in the box" is two
picks, in that order. Type in the terminal; the window is for watching and for the recording
keys.

What happens on each name, all decided here rather than by clicking (compare click_pick.py):

  1. find it   -- GroundingDINO with one phrase at a time, kept to the side of the box the
                  object always lies on and to its colour; SAM for the outline where depth
                  cannot give one (the tools), depth for the heights where it can.
  2. grasp     -- top-down, fingers across the narrow side of what they close on, the width
                  read off the footprint; a plush toy that is too fat around the middle is
                  taken where it is narrowest, a tool by its coloured handle. Fingertips a
                  set depth below the top of the held part. Neighbours that a fingertip
                  would land on turn the fingers or move the grasp along.
  3. place     -- a spot inside the box where the held object *and the open fingers* fit
                  with a margin from the walls and from whatever is already in there (seen by
                  depth and by colour), filling from the corner on the arm's own side; the
                  object is let go a little above whatever is under it. Something longer than
                  the box is allowed to lie across the rim, fingers still inside.
  4. arm       -- whichever arm's reachable territory the object is in (the shaded map from
                  click_pick.py), the other one if that one cannot solve the whole motion.
  5. motion    -- click_pick.run_plan, unchanged: fly out, straight down, close on contact,
                  lift, arc over, lower, open only as wide as it opened to pick, retreat.

Every pick is drawn to calib/debug/lang_<name>.png before the arm moves, and --no-execute
stops there. --from-saved plans on a saved frame (Frame.save layout) with no arms or cameras
at all, which is how the planning is checked against a scene without touching anything.

Keys in the window:  0-9 = pick the object on that key (--keys; the table is printed at
start and drawn on the idle view)   s / e / x = start / end / throw away a recorded episode
d = discard the last one   space = go (with --confirm)   n = skip (with --confirm)
q = quit (both arms fold back to the home pose first)
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import bimanual_teleop_record as rec
import click_pick as cp
import lang_perception as lp
from arm import Arm, Kinematics, Workspace
from cameras import CameraRig, Frame
from click_model import ClickModel
from perception import Detector
from poses import WORKING_Q

WINDOW = "language pick place"
DEBUG = _HERE / "calib" / "debug"


@dataclass
class Args(cp.Args):
    objects: Tuple[str, ...] = ()
    """Names to pick, in order, then quit. Empty: read names from the terminal until q."""
    confirm: bool = False
    """Show the plan and wait for space in the window before every move (n skips it)."""
    from_saved: Optional[str] = None
    """Plan on a saved frame in this directory (Frame.save layout: <stem>_color.png,
    <stem>_depth.npy, <stem>_K.json) instead of live cameras and arms. No motion at all."""
    stem: str = "top"
    sam: bool = True
    """Outline objects with SAM (facebook/sam-vit-base). Off: colour/depth outlines only."""
    det_threshold: float = 0.2
    place_margin: float = 0.015
    """Room to leave between the placed object (fingers included) and the walls / contents."""
    drop_clear: float = 0.025
    """The held object's bottom this far above whatever is under it when the fingers open."""
    wall_inset: float = 0.012
    """How far inside the rim rectangle the box's usable floor starts (wall + rim slop)."""
    fill_from: str = "near"
    """Where in the box to put things first: near (the corner on the arm's side) | far | centre."""
    allow_overhang: bool = True
    """Something too long for the box may be laid across its rim, fingers still inside."""
    scan_frames: int = 5
    """Top frames to median for each look at the table."""
    keys: Tuple[str, ...] = (
        "blue_cube",
        "elephant",
        "bread",
        "shovel",
        "saw",
        "white_cup",
        "hammer",
        "lemon",
        "bear",
        "pink",
    )
    """Which object each digit key picks, in the order 0, 1, ... 9 (ten keys, so one of the
    eleven names is left off by default: red_cube -- swap any in with --keys)."""
    task: str = "language_pick_place"
    """Episodes land in <save-root>/<task>/episode_NNNN (see click_pick.py)."""


# ---------------------------------------------------------------------------
# planning the arm's side of it
# ---------------------------------------------------------------------------


class _Shim:
    def __init__(self, kin: Kinematics) -> None:
        self.kin = kin


def plan_arm(
    side: str, kin: Kinematics, g: lp.GraspPlan, p: lp.PlacePlan, rim_z: float, args: Args
) -> Dict[str, object]:
    """Everything click_pick.run_plan needs, from a grasp and a place -- the same solve as
    click_pick.plan_for, with the heights coming from the objects instead of the clicks."""
    from pick_object import solve_approach, vertical_headroom

    tilts = tuple(float(t) for t in np.arange(0.0, args.max_tilt + 0.1, 15.0))
    grasp = np.array([g.center[0], g.center[1], g.z])
    place = np.array([p.center[0], p.center[1], p.z])
    out: Dict[str, object] = {
        "side": side,
        "g": cp.Half(centre=g.center.copy(), axis=g.axis3, surface=g.surface_z, width=g.width),
        "p": cp.Half(centre=p.center.copy(), axis=p.axis3, surface=p.surface_z, width=g.width),
        "grip_open": g.grip_open,
        "release_open": g.grip_open,
        "grasp": grasp,
        "place": place,
    }
    try:
        q_grasp, R_g, tilt_g = kin.ik_reach(grasp, WORKING_Q, tilts=tilts, finger_axis=g.axis3)
    except ValueError:
        return {"side": side, "ok": False, "why": f"grasp point out of reach (up to {args.max_tilt:.0f} deg of lean)"}
    # a leaning wrist drops one fingertip: aim high enough that the lower one lands at the
    # intended depth instead of into the table (as click_pick)
    drop_g = abs(float(R_g[2, 1])) * g.grip_open * lp.OPEN_STROKE / 2
    if drop_g > 0.002:
        grasp = grasp + np.array([0.0, 0.0, drop_g])
        try:
            q_grasp, R_g, tilt_g = kin.ik_reach(grasp, WORKING_Q, tilts=tilts, finger_axis=g.axis3)
        except ValueError:
            return {"side": side, "ok": False, "why": "grasp point out of reach once raised for the lean"}
        out["grasp"] = grasp
    try:
        q_place, R_p, tilt_p = kin.ik_reach(place, WORKING_Q, tilts=tilts, finger_axis=p.axis3)
    except ValueError:
        return {
            "side": side,
            "ok": False,
            "why": f"release point out of reach (up to {args.max_tilt:.0f} deg of lean)",
        }
    drop_p = abs(float(R_p[2, 1])) * g.grip_open * lp.OPEN_STROKE / 2
    if drop_p > 0.002:
        place = place + np.array([0.0, 0.0, drop_p])
        try:
            q_place, R_p, tilt_p = kin.ik_reach(place, WORKING_Q, tilts=tilts, finger_axis=p.axis3)
        except ValueError:
            return {"side": side, "ok": False, "why": "release point out of reach once raised for the lean"}
        out["place"] = place
    out.update(tilt_g=tilt_g, tilt_p=tilt_p, drop_g=drop_g, drop_p=drop_p)
    transit_g = np.array([grasp[0], grasp[1], g.surface_z + args.transit_clear])
    # over the box the held object hangs below the fingertips, and it has to clear the rim
    transit_p = np.array([place[0], place[1], max(p.surface_z + args.transit_clear, rim_z + g.hang + 0.06)])
    try:
        q_tg, _, _ = kin.ik_reach(transit_g, WORKING_Q, finger_axis=g.axis3)
        q_tp, _, _ = kin.ik_reach(transit_p, WORKING_Q, finger_axis=p.axis3)
    except ValueError:
        return {"side": side, "ok": False, "why": "no way in over the top"}
    shim = _Shim(kin)
    head = vertical_headroom(shim, grasp, R_g, q_grasp)
    try:
        q_pre, p_pre, _ = solve_approach(shim, grasp, R_g, q_grasp, args.approach)
    except ValueError:
        return {"side": side, "ok": False, "why": "no vertical approach above the grasp"}
    out.update(
        ok=True,
        q_pre=q_pre,
        p_pre=p_pre,
        q_grasp=q_grasp,
        R_g=R_g,
        q_place=q_place,
        R_p=R_p,
        q_tg=q_tg,
        q_tp=q_tp,
        transit_g=transit_g,
        transit_p=transit_p,
        headroom=head,
    )
    return out


@dataclass
class Pick:
    """One planned pick, ready to run (or the reason it cannot be)."""

    name: str
    side: Optional[str]
    obj: Optional[lp.Found]
    box: Optional[lp.BoxView]
    g: Optional[lp.GraspPlan]
    p: Optional[lp.PlacePlan]
    plan: Optional[Dict[str, object]]
    why: str
    image: np.ndarray

    @property
    def ok(self) -> bool:
        return self.plan is not None and bool(self.plan.get("ok"))


class Planner:
    """Perception + planning for one look at the table, for every arm that has a calibration."""

    def __init__(
        self,
        models: Dict[str, ClickModel],
        kins: Dict[str, Kinematics],
        args: Args,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.models = models
        self.kins = kins
        self.args = args
        self.masks = masks or {}
        self.detector = Detector()
        self.sam = lp.Segmenter() if args.sam else None
        self.plane: Optional[Tuple[np.ndarray, float]] = None

    def owner(self, obj: lp.Found) -> Optional[str]:
        """The arm whose territory the object's centre pixel is in, if the map is known."""
        v, u = np.nonzero(obj.grasp_mask)
        cu, cv_ = int(np.median(u)), int(np.median(v))
        for side, m in self.masks.items():
            if m[min(m.shape[0] - 1, cv_), min(m.shape[1] - 1, cu)]:
                return side
        return None

    def plan(self, name: str, frame: Frame) -> Pick:
        args = self.args
        t0 = time.time()
        view = lp.TableView(frame, self.models, self.plane)
        self.plane = view.plane
        box: Optional[lp.BoxView] = None
        obj: Optional[lp.Found] = None
        try:
            box = view.find_box(self.detector)
            obj = view.find_object(
                name, self.detector, self.sam, box_center_u=box.center_u, min_score=args.det_threshold
            )
        except LookupError as e:
            img = lp.draw_scene(view, box, obj, None, None, None, [f"{name}: {e}"])
            return Pick(name, None, obj, box, None, None, None, str(e), img)
        print(
            f"[find] {name}: {obj.note}; top {obj.top_h * 100:.1f} cm, {obj.mask.sum()} px  ({time.time() - t0:.1f}s)"
        )
        for side in self.models:
            print(f"[box]  {side}: {box.frame(side, args.wall_inset).describe()}")
        # the arm whose territory it is in goes first, then the one on its side of the box
        owner = self.owner(obj)
        order = list(self.models)
        order.sort(key=lambda s: (s != owner, s != lp.VOCAB[name].side))
        tried: List[str] = []
        for side in order:
            kin = self.kins[side]
            try:
                obs = view.obstacle_points(side, exclude=obj.mask)
                grasps = lp.plan_grasps(view, obj, side, obstacles=obs, margin=args.grip_margin)
            except ValueError as e:
                tried.append(f"{side}: {e}")
                continue
            plan: Optional[Dict[str, object]] = None
            # the best grasp first; if the arm cannot solve it (a finger direction the wrist
            # cannot turn to out there), the next distinct one
            for g in grasps:
                try:
                    p = lp.plan_place(
                        view,
                        box,
                        g,
                        side,
                        margin=args.place_margin,
                        drop_clear=args.drop_clear,
                        wall_inset=args.wall_inset,
                        prefer=args.fill_from,
                    )
                except ValueError as e:
                    if not args.allow_overhang:
                        tried.append(f"{side}: {e}")
                        break
                    try:
                        p = lp.plan_place(
                            view,
                            box,
                            g,
                            side,
                            margin=args.place_margin,
                            drop_clear=args.drop_clear,
                            wall_inset=args.wall_inset,
                            prefer=args.fill_from,
                            overhang=True,
                        )
                    except ValueError as e2:
                        tried.append(f"{side}: {e2}")
                        break
                bf = box.frame(side, args.wall_inset)
                plan = plan_arm(side, kin, g, p, bf.rim_z, args)
                if plan.get("ok"):
                    break
                tried.append(f"{side}: {plan['why']} (fingers {g.note.split(', fingers ')[-1].split(',')[0]})")
                plan = None
            if plan is None:
                continue
            print(
                f"[grasp] {side}: ({g.center[0]:.3f}, {g.center[1]:.3f}, {g.z:.3f}) axis {np.round(g.axis, 2)} -- {g.note}"
            )
            print(
                f"[place] {side}: ({p.center[0]:.3f}, {p.center[1]:.3f}, {p.z:.3f}) axis {np.round(p.axis, 2)} -- {p.note}"
            )
            print(
                f"[plan]  {side} arm{' (its territory)' if side == owner else ''}: lean {float(plan['tilt_g']):.0f}/{float(plan['tilt_p']):.0f} deg, "
                f"lift headroom {float(plan['headroom']) * 100:.0f} cm, open {g.grip_open:.2f}  ({time.time() - t0:.1f}s)"
            )
            lines = [
                f"{name} -> {side} arm   grasp ({plan['grasp'][0]:.3f},{plan['grasp'][1]:.3f},{plan['grasp'][2]:.3f}) open {g.grip_open:.2f}  release ({plan['place'][0]:.3f},{plan['place'][1]:.3f},{plan['place'][2]:.3f})",
                g.note,
                p.note,
            ]
            img = lp.draw_scene(view, box, obj, g, p, side, lines)
            return Pick(name, side, obj, box, g, p, plan, "", img)
        why = "; ".join(tried) or "no arm available"
        img = lp.draw_scene(view, box, obj, None, None, None, [f"{name}: cannot do it", why[:150]])
        return Pick(name, None, obj, box, None, None, None, why, img)


# ---------------------------------------------------------------------------
# the live loop
# ---------------------------------------------------------------------------


def top_frame(rig: CameraRig, n: int) -> Frame:
    """A fresh top frame with the depth medianed over ``n`` frames (holes and flicker)."""
    frames = [rig.grab_fresh("top", min_frames=2) for _ in range(max(1, n))]
    stack = np.stack([f.depth for f in frames])
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    f = frames[-1]
    return Frame(f.role, f.color, depth, f.K, f.t)


def stdin_reader(q: "queue.Queue[str]", stop: threading.Event) -> None:
    """Lines typed in the terminal, handed to the display loop."""
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:  # EOF
            q.put("\x04")
            return
        q.put(line.strip())


def show(img: np.ndarray, args: Args, status: str = "") -> None:
    if status:
        cv2.putText(img, status, (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            img,
            status,
            (12, img.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (60, 60, 255) if status.startswith("REC") else (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    scale = args.display_width / img.shape[1]
    cv2.imshow(WINDOW, cv2.resize(img, (args.display_width, int(img.shape[0] * scale))))


def run_live(args: Args) -> None:
    models: Dict[str, ClickModel] = {}
    for side in args.sides:
        try:
            models[side] = ClickModel.load(side)
            print(f"[calib] {side}: {models[side].describe()}")
        except Exception as e:
            print(f"[calib] {side}: no click calibration ({e})")
    if not models:
        raise SystemExit("[error] no arm has a click calibration -- run calibrate_click.py first")
    roles = ("top", "left_wrist", "right_wrist") if args.record else ("top",)
    try:
        rig = CameraRig.open(roles=roles)
    except Exception as e:
        print(f"[cam] {e}; falling back to the top camera only (no recording)")
        args.record = False
        rig = CameraRig.open(roles=("top",))
    arms: Dict[str, Arm] = {}
    for side in models:
        try:
            arms[side] = Arm(
                side,
                cp.PORTS[side],
                max_joint_vel=args.max_joint_vel,
                execute=args.execute,
                workspace=Workspace(z=(0.03, 0.6)),
            )
        except Exception as e:
            print(f"[arm] {side} not available ({e})")
    if args.execute and args.start_at_working:
        for side, arm in arms.items():
            print(f"[start] {side} -> working pose")
            cp.go_working(arm, args)
    kins = {side: (arms[side].kin if side in arms else Kinematics()) for side in models}

    # the table plane in the camera frame, once (the camera does not move)
    try:
        from perception import Scene

        prior = Scene.from_calib("left").table_plane_cam()
    except Exception:
        prior = None
    time.sleep(0.5)
    plane = cp.table_plane_cam(rig.grab("top"), prior)
    print(f"[table] plane in the camera frame: n={np.round(plane[0], 3)} d={plane[1]:.3f}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    speaker = rec.Speaker(args.tts)
    recorder = cp.Recorder(args, rig, arms, speaker)
    if args.record:
        n = rec._next_episode_index(recorder.task_dir) if recorder.task_dir.exists() else 0
        print(f"[record] task '{args.task}' -> {recorder.task_dir} (next episode {n:04d}, {len(rig.cams)} cameras)")
        print("[record] press s in the window to start an episode, e to end it")
    # the reachable-territory map, from click_pick's cache (computed if missing)
    mask_holder: Dict[str, Dict[str, np.ndarray]] = {"masks": {}}
    f0 = rig.grab("top")
    if args.show_reach:
        pts_cam, uv = f0.point_cloud(f0.depth > 0.2)
        n_, d_ = plane
        flat = np.abs(pts_cam @ n_ + d_) < 0.15
        on_table = np.zeros(f0.depth.shape, bool)
        on_table[uv[flat, 1], uv[flat, 0]] = True
        on_table = cv2.morphologyEx(on_table.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(
            bool
        )
        threading.Thread(
            target=cp.fill_masks, args=(mask_holder, models, f0.depth.shape, args, on_table), daemon=True
        ).start()
    # The models take several seconds to load and the window must keep answering the
    # desktop meanwhile, or it gets flagged as not responding -- same for every pick below.
    holder: Dict[str, object] = {}

    def _load() -> None:
        try:
            holder["planner"] = Planner(models, kins, args)
        except Exception as e:  # surfaced below, from the main thread
            holder["error"] = e

    loader = threading.Thread(target=_load, name="load-models", daemon=True)
    loader.start()
    while loader.is_alive():
        show(f0.color.copy(), args, "loading the detector and SAM...")
        cv2.waitKey(50)
    if "planner" not in holder:
        raise SystemExit(f"[error] could not load the models: {holder.get('error')}")
    planner: Planner = holder["planner"]  # type: ignore[assignment]
    planner.plane = plane
    speaker.say("ready")

    todo: "queue.Queue[str]" = queue.Queue()
    stop = threading.Event()
    for name in args.objects:
        todo.put(name)
    keymap: Dict[int, str] = {}
    for i, name in enumerate(args.keys[:10]):
        hits = lp.parse_instruction(name)
        if hits:
            keymap[ord(str(i))] = hits[0]
        else:
            print(f"[keys] unknown object {name!r} on key {i} -- ignored")
    legend = "  ".join(f"{chr(k)}={n}" for k, n in sorted(keymap.items()))
    if args.objects:
        todo.put("\x04")
    else:
        threading.Thread(target=stdin_reader, args=(todo, stop), daemon=True).start()
        print(
            f"[ready] type an object name and press enter ({', '.join(lp.VOCAB)}), or a digit in the window; q in the window quits"
        )
    print(f"[keys] {legend}")
    # Shared with the worker: what to draw, the one-line status, whether a pick is under
    # way, and the answer to a --confirm prompt. The main thread never blocks on the arms.
    state: Dict[str, object] = {"img": None, "status": "", "busy": False, "confirm": None}
    confirm_evt = threading.Event()

    def pick_one(line: str, name: str) -> None:
        spoken = name.replace("_", " ")
        speaker.say(f"looking for the {spoken}")
        state["status"] = f"looking for {name}..."
        t0 = time.time()
        pick = planner.plan(name, top_frame(rig, args.scan_frames))
        state["img"] = pick.image
        cv2.imwrite(str(DEBUG / f"lang_{name}.png"), pick.image)
        if not pick.ok:
            print(f"[plan] {name}: cannot do it -- {pick.why}")
            speaker.say(f"I cannot pick the {spoken}")
            return
        plan, side = pick.plan, str(pick.side)
        print(f"[plan] {name}: {side} arm, {time.time() - t0:.1f}s; drawn to calib/debug/lang_{name}.png")
        if not args.execute or side not in arms:
            print("[dry run] --execute (and a live arm) to move")
            return
        if args.confirm:
            state["confirm"] = None
            confirm_evt.clear()
            state["status"] = "space = go, n = skip"
            while not confirm_evt.wait(0.1):
                if stop.is_set():
                    return
            if not state["confirm"]:
                print("[skip] not running that one")
                return
        speaker.say(f"picking the {spoken}")
        state["status"] = f"running: {name} with the {side} arm"
        ok, err = True, ""
        frame_start = recorder.n_frames
        t_start = time.time()
        try:
            cp.run_plan(plan, args, arms[side])
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            print(f"[run] failed -- {err}")
            speaker.say("the run failed")
            state["status"] = "run failed -- back to the working pose"
            for other, a in arms.items():
                print(f"[recover] {other} -> working pose")
                try:
                    cp.go_working(a, args)
                except Exception as e2:
                    print(f"[recover] {other} did not make it back ({type(e2).__name__}: {e2})")
        if recorder.active:
            g, p = pick.g, pick.p
            info: Dict[str, object] = {
                "instruction": line,
                "object": name,
                "arm": side,
                "grasp": np.asarray(plan["grasp"]).tolist(),
                "release": np.asarray(plan["place"]).tolist(),
                "finger_axis_grasp": g.axis3.tolist() if g else None,
                "finger_axis_release": p.axis3.tolist() if p else None,
                "grip_open": float(plan["grip_open"]),
                "grip_width_m": float(arms[side].q_cmd[6]) * lp.OPEN_STROKE,
                "lean_deg": [float(plan.get("tilt_g", 0)), float(plan.get("tilt_p", 0))],
                "grasp_note": g.note if g else "",
                "place_note": p.note if p else "",
                "frames": [frame_start, recorder.n_frames],
                "t_start": t_start,
                "t_end": time.time(),
                "completed": ok,
                "error": err,
            }
            recorder.note_run(info)
        speaker.say("done" if ok else "failed")

    def worker(line: str, names: List[str]) -> None:
        try:
            for name in names:
                if stop.is_set():
                    break
                pick_one(line, name)
        except Exception as e:  # a bug in the planner must not take the window down
            import traceback

            traceback.print_exc()
            print(f"[pick] failed -- {type(e).__name__}: {e}")
            state["status"] = f"failed: {e}"
            speaker.say("the pick failed")
        finally:
            state["busy"] = False

    quitting = False
    try:
        while True:
            planner.masks = mask_holder["masks"]
            frame = rig.grab("top")
            last_img = state["img"]
            img = last_img.copy() if last_img is not None else frame.color.copy()  # type: ignore[union-attr]
            if last_img is None:
                lines = [
                    "press a digit, or type an object name in the terminal"
                    + ("" if planner.masks else "   [reach map loading]"),
                    legend,
                    "q = arms home and quit",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(
                        img, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA
                    )
                    cv2.putText(
                        img, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA
                    )
            status = str(state["status"]) or recorder.status()
            if quitting:
                status = "finishing this pick, then both arms go home"
            show(img, args, status)
            key = cv2.waitKey(30) & 0xFF
            busy = bool(state["busy"])
            if quitting and not busy:
                break
            if key == ord("q"):
                if not busy:
                    break
                # never abandon an arm mid-motion: the pick under way runs to its end first
                print("[quit] finishing the current pick first")
                quitting = True
            if key in keymap and not quitting:
                print(f"[keys] {chr(key)} -> {keymap[key]}")
                todo.put(keymap[key])
            if key == ord("s"):
                recorder.start()
            if key == ord("e"):
                recorder.stop()
            if key == ord("x"):
                recorder.abort()
            if key == ord("d"):
                recorder.discard_last()
            if busy and str(state["status"]).startswith("space = go"):
                if key in (ord(" "), 13, 10):
                    state["confirm"] = True
                    confirm_evt.set()
                elif key == ord("n"):
                    state["confirm"] = False
                    confirm_evt.set()
            if busy or quitting:
                continue
            try:
                line = todo.get_nowait()
            except queue.Empty:
                continue
            if line == "\x04":
                break
            if not line:
                continue
            if line.strip().lower() in ("q", "quit", "exit"):
                break
            names = lp.parse_instruction(line)
            if not names:
                print(f"[input] nothing I know in {line!r} -- one of: {', '.join(lp.VOCAB)}")
                speaker.say("I do not know that object")
                continue
            state["busy"] = True
            state["status"] = ""
            threading.Thread(target=worker, args=(line, names), name="pick", daemon=True).start()
    finally:
        stop.set()
        if recorder.active:
            print("[record] quitting while recording -- saving the episode")
            recorder.stop({"stop_reason": "quit while recording"})
        if args.execute and args.park_on_exit and arms:
            speaker.say("going home")
            for side, arm in arms.items():
                print(f"[exit] {side} -> home pose")
                try:
                    cp.go_home(arm, args)
                except Exception as e:
                    print(f"[exit] {side} did not reach home ({type(e).__name__}: {e})")
        cv2.destroyAllWindows()
        for a in arms.values():
            a.close()
        rig.close()


# ---------------------------------------------------------------------------
# offline: plan on a saved frame
# ---------------------------------------------------------------------------


def run_saved(args: Args) -> None:
    src = Path(args.from_saved).expanduser()
    frame = Frame.load(src, args.stem, role="top")
    models = {side: ClickModel.load(side) for side in args.sides if (_HERE / "calib" / f"click_{side}.json").exists()}
    if not models:
        raise SystemExit("[error] no click calibration found")
    kins = {side: Kinematics(args.arm, args.version, args.follower_gripper) for side in models}
    masks: Dict[str, np.ndarray] = {}
    for side in models:
        path = cp.cache_path(side, args, frame.depth.shape)
        if path.exists():
            masks[side] = np.load(path)
    if masks:
        cp.split_territory(masks, models, frame.depth.shape, args)
        print(
            f"[reach] territory map from cache: {', '.join(f'{s}: {m.mean() * 100:.0f}%' for s, m in masks.items())}"
        )
    else:
        print("[reach] no cached territory map; arms are tried by the object's side of the box")
    planner = Planner(models, kins, args, masks)
    names = list(args.objects) or list(lp.VOCAB)
    DEBUG.mkdir(parents=True, exist_ok=True)
    for name in names:
        hits = lp.parse_instruction(name)
        if not hits:
            print(f"[input] unknown object {name!r}")
            continue
        for n in hits:
            pick = planner.plan(n, frame)
            out = DEBUG / f"lang_{n}.png"
            cv2.imwrite(str(out), pick.image)
            print(f"[{n}] {'OK -> ' + str(pick.side) + ' arm' if pick.ok else 'cannot: ' + pick.why}   ({out})")


def main(args: Args) -> None:
    if args.from_saved:
        run_saved(args)
        return
    procs: List["subprocess.Popen[bytes]"] = []
    try:
        cp.launch_followers(args, procs)
        run_live(args)
    finally:
        if procs:
            print(
                "[stop] the followers this run launched are exiting: motor torque goes to zero and the arms go limp."
            )
            rec._terminate(procs)


if __name__ == "__main__":
    main(tyro.cli(Args))
