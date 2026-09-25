"""Click-to-pick: point at the top camera's live image, the arms do the rest.

Click four points. The first two say where and how to grasp -- their midpoint is the grasp
point, the line between them is the direction the fingers open, and their distance sets how
wide the gripper opens. The next two say the same for the release. The tool then picks the arm
that can reach both, and runs: over the object -> straight down -> close -> straight up ->
across -> open. The release happens in the air, --release-height above the surface at the
release click, so nothing is ever driven down into whatever is being placed on or into. A run
that fails part way (a target the workspace box refuses, IK that gives out on the descent)
sends both arms back to the working pose rather than leaving them stalled where they stopped.

The motion is meant to look teleoperated rather than programmed. A pick is three streamed
paths, not nine point-to-point moves: fly out and drop onto the object, lift and carry it over
to the release, come back to the working pose -- each one continuous joint samples whose
corners have been rounded off and whose timing is set by a cap on how fast the fingertips are
allowed to travel (--eef-speed, 20 cm/s), bounded at its ends by a deceleration limit
(--accel) and slowed further over the last stretch onto a target. The arm only actually stops
where a person would: on the object with the fingers closing, and over the release with them
opening.

Everything is measured through the per-arm click calibration (calibrate_click.py): a pixel
plus a height maps to that arm's base coordinates directly, and the height comes from the
depth camera at the clicked pixel, so an object's own height sets the grasp depth.

    python scripts/box_packing/click_pick.py                 # both arms, live
    python scripts/box_packing/click_pick.py --no-execute    # plan and draw only

Both follower arms are started by this script (the recorder's own launch, followers only,
no leaders), so nothing else has to be running first. A port that is already served is
attached to instead of launched over, so scripts/run_policy_followers.sh still works --
pass --no-launch to require that.

Keys:  space / enter = run     r = clear the clicks     u = undo one click
       s = start recording an episode    e = end it (save)    x = throw it away
       [ / ]  grasp deeper / shallower    - / +  release lower / higher
       l / R / a = force the left arm / right arm / automatic
       q = quit (both arms fold down to the home pose first)
       d  discard the last saved episode

Recording is on the keys, not on the runs: nothing is written until you press ``s``, and the
episode ends when you press ``e``. Everything in between -- however many picks, and the
thinking time between them -- is one episode, which is what a multi-pick demonstration needs.
Episodes land in <save-root>/<task>/episode_NNNN/ in the same layout bimanual_teleop_record.py
writes -- one mp4 per camera, low_dim.npz, meta.json last -- so the clicked demonstrations and
the teleoperated ones can be read by the same tools; meta.json carries a ``runs`` list saying
what each pick in the episode was and which frames it spans.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

sys.path.insert(0, str(_HERE.parent))

import bimanual_teleop_record as rec  # noqa: E402  (EpisodeWriter, Speaker, episode numbering)
from arm import Arm, MoveResult, Workspace  # noqa: E402
from cameras import CameraRig, Frame  # noqa: E402
from can_channels import channel_for  # noqa: E402
from click_model import ClickModel  # noqa: E402
from poses import HOME_Q, WORKING_Q  # noqa: E402

PORTS = {"left": 1235, "right": 1234}
WINDOW = "click to pick"
OPEN_STROKE = 0.096
"""Fingertip separation at gripper command 1.0, metres."""


@dataclass
class Args:
    sides: Tuple[str, ...] = ("left", "right")
    """Arms available. An arm without calib/click_<side>.json is ignored."""
    execute: bool = True
    grasp_depth: float = 0.02
    """Fingertips this far below the surface height at the grasp click."""
    release_height: float = 0.05
    """Fingertips this far above the surface at the release click, where the object is let go.
    There is no descent on the release side at all: the arm flies over at --transit-clear,
    drops to this height and opens, so the object falls the last few centimetres."""
    transit_clear: float = 0.16
    """Flight height above the table."""
    approach: float = 0.08
    eef_speed: float = 0.20
    """Ceiling on fingertip speed, m/s -- the one number that sets the pace of everything.
    Roughly what a hand on the teleoperation leader does over a transit; the joint slew limit
    (--max-joint-vel) still applies on top and is what actually binds on the parts of a move
    the wrist dominates, and a turn in place is charged 10 cm per radian so it is bounded too.
    The approach onto a target is slower than this by construction -- see --descend-speed."""
    descend_speed: float = 0.06
    """Tip speed over the last --approach onto the grasp: the run eases down to this rather
    than arriving at full speed."""
    place_speed: float = 0.08
    """The same, coming down onto the release point."""
    accel: float = 0.35
    """How hard a streamed motion is allowed to pick up speed and shed it, m/s^2. This is what
    makes a start and a stop look like a hand rather than a step input -- and, at the end of a
    descent, what decides whether the arm settles onto the object or creeps the last centimetre
    for a second and a half looking like it has stopped to think."""
    start_speed: float = 0.03
    """Slowest a streamed motion is allowed to creep, m/s, unless the local cap is lower.
    The deceleration ramp is otherwise exactly zero at each end of a path, and since a path is
    sampled every few millimetres the first interval alone then takes 0.15 s -- the arm sitting
    still on the object it has just gripped, which is what a pause before the lift looks like.
    It also sets how fast the fingertips are still travelling when they arrive on a target;
    lower it if objects get nudged on touchdown."""
    path_smooth: float = 0.03
    """Corner-rounding radius, metres of path. A straight run is untouched by it; it only
    bites where a leg turns -- fly-out into descent, lift into carry -- which is where the
    right angles used to be. 0 keeps the corners square."""
    path_spacing: float = 0.004
    """How finely a path is sampled along its own travel before it is smoothed and streamed."""
    pin_ends: float = 0.02
    """Metres at each end of a path that the smoothing may not move, so an endpoint stays
    exactly on its target and the last of a descent stays exactly vertical."""
    carry_arc: float = 0.07
    """How high over the two transit heights the carry arcs, metres. Both ends sit the same
    clearance over the same table, so without this the object crosses dead level; a person
    lifts it clear, swings it over and lowers it in. Capped at a third of the distance covered,
    so a short move barely bulges, and dropped altogether if the apex is out of reach."""
    lift_straight: float = 0.05
    """The same, for the lift out of a grasp and the drop onto a release: this much of it is
    held exactly vertical before the curve across is allowed to start, so the object clears
    whatever it was sitting in before it is swung anywhere."""
    max_joint_vel: float = 0.6
    grip_margin: float = 0.02
    """Extra opening beyond the clicked width."""
    close: float = 0.0
    """Tightest the fingers may go. They normally stop earlier, on contact."""
    grip_lag: float = 0.035
    """How far the fingers may fall behind their command before that counts as contact. This
    is what stops the gripper crushing things: the gap between command and position *is* the
    squeezing force, and clamping to 0 on a rigid object asks for all of it."""
    squeeze: float = 0.012
    """How much tighter than the contact point to hold -- about 1 mm of finger travel, enough
    to hold a light object, gentle enough for a paper cup."""
    close_wait: float = 0.05
    """How long the fingers hold the squeeze before the lift begins, seconds. This is the one
    interval where the arm is deliberately stationary between gripping and lifting, and it is
    short on purpose: it only exists so the width this prints is settled, and the lift that
    follows starts slowly anyway, so the squeeze goes on building while the arm is moving."""
    open_wait: float = 0.3
    correct_rounds: int = 3
    """Residual-compensation rounds at the bottom of a descent. The soft wrist joints sag a
    centimetre under gravity and one round only takes out about half of it: measured landing
    error is 7.8 mm with one round, 3.3 with two, 1.7 with three, at 0.1 s each."""
    contact_abort_rad: float = 0.15
    task: str = "click_pick"
    """Episodes land in <save-root>/<task>/episode_NNNN, numbered on from what is there."""
    save_root: str = "~/yam_data"
    record: bool = True
    """Allow recording at all (opens the wrist cameras). Episodes are still started and ended
    by hand, with the s and e keys -- runs made outside an episode are not written."""
    record_fps: float = 30.0
    tts: bool = True
    """Speak what happened (spd-say), so you can watch the arms instead of the terminal."""
    display_width: int = 1280
    max_tilt: float = 45.0
    """How far the wrist may lean from vertical. A vertical wrist covers only the near part of
    the table; letting it lean opens up most of the rest. The grasp point is raised by however
    much the lean drops the lower fingertip, so the fingers still straddle the object rather
    than one of them digging into the table."""
    show_reach: bool = True
    """Shade the part of the image each arm can actually grasp in (green = left, orange = right)."""
    table_guess: float = 0.065
    """Fallback surface height when the depth at a clicked pixel is missing."""
    start_at_working: bool = True
    """Send both arms to the working pose at startup, before the table is measured."""
    park_on_exit: bool = True
    """On q, fold both arms down to the home pose before letting go of them. The followers this
    run launched exit right after, and that cuts motor torque -- home is where the arms rest
    unpowered, so they settle into it instead of dropping from wherever they were left."""
    launch: bool = True
    """Start a minimum_gello follower server per arm, so no other terminal is needed. Ports
    that are already served are attached to either way; --no-launch requires that."""
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = channel_for("follower_left")
    """LEFT follower CAN netdev. Default comes from the mapping table scripts/can_map.conf."""
    can_follower_right: str = channel_for("follower_right")
    """RIGHT follower CAN netdev (from scripts/can_map.conf)."""
    sim: bool = False
    """Launch the followers in MuJoCo instead of on the CAN buses (no hardware needed)."""


@dataclass
class Clicks:
    pts: List[Tuple[int, int]] = field(default_factory=list)
    scale: float = 1.0

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < 4:
            self.pts.append((int(x / self.scale), int(y / self.scale)))

    @property
    def ready(self) -> bool:
        return len(self.pts) == 4


def table_plane_cam(frame: Frame, prior: Optional[Tuple[np.ndarray, float]] = None) -> Tuple[np.ndarray, float]:
    """The table plane in the camera's own frame, refined on this frame.

    Measured straight from the depth image, so it needs no arm calibration; a prior (from the
    old top-camera calibration) keeps the fit off the back wall, which is the larger plane.
    """
    from calibrate_top import fit_plane

    pts, _ = frame.point_cloud((frame.depth > 0.3) & (frame.depth < 1.5))
    sub = pts[:: max(1, len(pts) // 60000)]
    if prior is None:
        return fit_plane(sub)
    n, d = prior
    for _ in range(2):
        inl = sub[np.abs(sub @ n + d) < 0.015]
        if len(inl) < 500:
            break
        cen = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
        n_new = vt[2]
        if n_new @ n < 0:
            n_new = -n_new
        n, d = n_new, float(-n_new @ cen)
    return n, d


def surface_height(
    model: ClickModel,
    frame: Frame,
    uv: Tuple[int, int],
    plane: Tuple[np.ndarray, float],
    fallback: float,
    radius: int = 4,
    ring: Tuple[int, int] = (45, 85),
) -> float:
    """Base-frame height of whatever the camera sees at that pixel.

    Measured against the table *right next to that pixel*, not against one plane fitted to the
    whole image: this depth camera reads the far corners of the table 1-2 cm high, which would
    go straight into the grasp depth. The ring of table pixels around the click is refitted and
    the height taken from that, so the bias cancels.
    """
    h, w = frame.depth.shape
    u, v = int(uv[0]), int(uv[1])
    win = frame.depth[max(0, v - radius) : min(h, v + radius + 1), max(0, u - radius) : min(w, u + radius + 1)]
    vals = win[win > 0]
    if vals.size < 5:
        return fallback
    p = frame.deproject(u, v, float(np.median(vals)))
    if p is None:
        return fallback
    n, d = plane
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - u) ** 2 + (yy - v) ** 2
    local = (r2 > ring[0] ** 2) & (r2 < ring[1] ** 2) & (frame.depth > 0.2)
    pts, _ = frame.point_cloud(local)
    if len(pts) > 200:
        flat = pts[np.abs(pts @ n + d) < 0.02]  # the table around it, not whatever stands there
        if len(flat) > 150:
            cen = flat.mean(0)
            _, _, vt = np.linalg.svd(flat[:: max(1, len(flat) // 4000)] - cen, full_matrices=False)
            n_loc = vt[2]
            if n_loc @ n < 0:
                n_loc = -n_loc
            n, d = n_loc, float(-n_loc @ cen)
    above = abs(float(p @ n + d))
    try:
        return model.base_z(above)
    except ValueError:
        return fallback


@dataclass
class Half:
    """One half of the task: where to go, how to hold, how deep."""

    centre: np.ndarray  # base xy at the working height
    axis: np.ndarray  # unit finger-opening direction, base frame
    surface: float  # base-frame height of the surface there
    width: float  # clicked distance between the two points, metres


def read_half(model: ClickModel, frame: Frame, a: Tuple[int, int], b: Tuple[int, int], args: Args, plane: Tuple[np.ndarray, float]) -> Half:
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    surface = surface_height(model, frame, (int(mid[0]), int(mid[1])), plane, args.table_guess)
    pa = model.xy_at(np.array(a, dtype=np.float64), surface)
    pb = model.xy_at(np.array(b, dtype=np.float64), surface)
    centre = (pa + pb) / 2.0
    d = pb - pa
    n = float(np.linalg.norm(d))
    axis = d / n if n > 1e-6 else np.array([1.0, 0.0])
    return Half(centre=centre, axis=np.array([axis[0], axis[1], 0.0]), surface=surface, width=n)


def plan_for(side: str, model: ClickModel, frame: Frame, clicks: Clicks, args: Args, plane: Tuple[np.ndarray, float], kin=None) -> Optional[Dict[str, object]]:
    """Everything the run needs for one arm, or None if it cannot do it."""
    if kin is None:
        from arm import Kinematics

        kin = _KIN.setdefault(side, Kinematics())
    g = read_half(model, frame, clicks.pts[0], clicks.pts[1], args, plane)
    p = read_half(model, frame, clicks.pts[2], clicks.pts[3], args, plane)
    grip_open = float(np.clip((g.width + args.grip_margin) / OPEN_STROKE, 0.15, 1.0))
    grasp_z = max(g.surface - args.grasp_depth, g.surface - 0.06)
    # The release is a fixed height above whatever is under the release click -- the object is
    # dropped from there rather than set down, so how deep the fingers sit on it does not enter
    # into it (and neither does how tall it is, which the top camera cannot see anyway).
    place_z = p.surface + args.release_height
    grasp = np.array([g.centre[0], g.centre[1], grasp_z])
    place = np.array([p.centre[0], p.centre[1], place_z])
    out: Dict[str, object] = {"side": side, "g": g, "p": p, "grip_open": grip_open, "grasp": grasp, "place": place}
    # Upright first, leaning only as far as it must: a vertical wrist keeps both fingertips at
    # the same height and lifts whatever is held straight up, but it only reaches the near part
    # of the table, so the solver walks out to --max-tilt when it has to.
    tilts = tuple(float(t) for t in np.arange(0.0, args.max_tilt + 0.1, 15.0))
    try:
        q_grasp, R_g, tilt_g = kin.ik_reach(grasp, WORKING_Q, tilts=tilts, finger_axis=g.axis)
    except ValueError:
        return {"side": side, "ok": False, "why": f"grasp point out of reach (up to {args.max_tilt:.0f} deg of lean)"}
    # a leaning wrist drops one fingertip: aim high enough that the lower one lands at the
    # intended depth instead of into the table
    drop_g = abs(float(R_g[2, 1])) * grip_open * OPEN_STROKE / 2
    if drop_g > 0.002:
        grasp = grasp + np.array([0.0, 0.0, drop_g])
        try:
            q_grasp, R_g, tilt_g = kin.ik_reach(grasp, WORKING_Q, tilts=tilts, finger_axis=g.axis)
        except ValueError:
            return {"side": side, "ok": False, "why": "grasp point out of reach once raised for the lean"}
        out["grasp"] = grasp
    try:
        q_place, R_p, tilt_p = kin.ik_reach(place, WORKING_Q, tilts=tilts, finger_axis=p.axis)
    except ValueError:
        return {"side": side, "ok": False, "why": f"release point out of reach (up to {args.max_tilt:.0f} deg of lean)"}
    drop_p = abs(float(R_p[2, 1])) * grip_open * OPEN_STROKE / 2
    if drop_p > 0.002:
        place = place + np.array([0.0, 0.0, drop_p])
        try:
            q_place, R_p, tilt_p = kin.ik_reach(place, WORKING_Q, tilts=tilts, finger_axis=p.axis)
        except ValueError:
            return {"side": side, "ok": False, "why": "release point out of reach once raised for the lean"}
        out["place"] = place
    out.update(tilt_g=tilt_g, tilt_p=tilt_p, drop_g=drop_g, drop_p=drop_p)
    transit_g = np.array([grasp[0], grasp[1], g.surface + args.transit_clear])
    transit_p = np.array([place[0], place[1], p.surface + args.transit_clear])
    try:
        q_tg, R_tg, _ = kin.ik_reach(transit_g, WORKING_Q, finger_axis=g.axis)
        q_tp, R_tp, _ = kin.ik_reach(transit_p, WORKING_Q, finger_axis=p.axis)
    except ValueError:
        return {"side": side, "ok": False, "why": "no way in over the top"}
    # how far the object can be lifted straight up, which is what a box makes tight
    from pick_object import solve_approach, vertical_headroom

    class _A:
        pass

    a = _A()
    a.kin = kin
    head = vertical_headroom(a, grasp, R_g, q_grasp)
    # The descent is a straight vertical line held at the grasp orientation, so it has to
    # *start* somewhere that orientation is reachable -- the transit pose above it is not
    # (a vertical wrist runs out around z = 0.20 out here). Solve the highest point above
    # each target that still holds it, and go there first.
    try:
        q_pre, p_pre, h_pre = solve_approach(a, grasp, R_g, q_grasp, args.approach)
    except ValueError:
        return {"side": side, "ok": False, "why": "no vertical approach above the grasp"}
    out.update(q_pre=q_pre, p_pre=p_pre)
    out.update(ok=True, q_grasp=q_grasp, R_g=R_g, q_place=q_place, R_p=R_p, q_tg=q_tg, q_tp=q_tp, transit_g=transit_g, transit_p=transit_p, headroom=head)
    return out


_KIN: Dict[str, object] = {}


def choose(plans: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """The arm that can do it with the most room to lift; ties go to the closer reach."""
    if not plans:
        return None
    return max(plans, key=lambda pl: (min(float(pl["headroom"]), 0.10), -float(np.linalg.norm(pl["grasp"][:2]))))


def reach_mask(model: ClickModel, kin, shape: Tuple[int, int], table_z: float, args: Args, step: int = 40, on_table: Optional[np.ndarray] = None) -> np.ndarray:
    """Which pixels this arm could grasp at, drawn as a coarse mask.

    A vertical wrist is what the picks rely on, and it only reaches part of the table -- being
    told that *after* clicking four points is a poor way to find out, so the reachable band is
    shaded on the live view.
    """
    h, w = shape
    grid = np.zeros(((h + step - 1) // step, (w + step - 1) // step), bool)
    z = table_z - args.grasp_depth
    from arm import rot_from_z_axis

    # A handful of fixed wrist orientations rather than the full ik_grasp search: the fingers
    # can always turn to one of them, and it is ~40x faster, which is the difference between a
    # one-second startup and a minute of staring at a blank window. The leans match what
    # plan_for will accept, so the shaded area is what can really be picked.
    seed = WORKING_Q[:6]
    rots = []
    for tilt in np.arange(0.0, args.max_tilt + 0.1, 15.0):
        for a in np.linspace(0, np.pi, 4, endpoint=False):
            lean = np.array([np.cos(a), np.sin(a), 0.0]) * np.sin(np.radians(tilt)) - np.array([0.0, 0.0, np.cos(np.radians(tilt))])
            rots.append(rot_from_z_axis(lean, np.array([np.cos(a + np.pi / 2), np.sin(a + np.pi / 2), 0.0])))
    for i, v in enumerate(range(0, h, step)):
        for j, u in enumerate(range(0, w, step)):
            xy = model.xy_at(np.array([u, v], dtype=np.float64), z)
            p3 = np.array([xy[0], xy[1], z])
            for R in rots:
                q6, perr, rerr = kin.ik(p3, R, seed)
                if perr < 5e-3 and rerr < 0.05:
                    grid[i, j] = True
                    break
    mask = cv2.resize(grid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask & on_table if on_table is not None else mask


def cache_path(side: str, args: Args, shape: Tuple[int, int]) -> Path:
    return _HERE / "calib" / f"reach_{side}_{shape[1]}x{shape[0]}_{round(args.grasp_depth * 1000)}mm_{round(args.max_tilt)}deg.npy"


def base_distance(model: ClickModel, shape: Tuple[int, int], z: float, step: int = 40) -> np.ndarray:
    """How far each pixel is from this arm's own base, in metres, as a coarse map."""
    h, w = shape
    grid = np.zeros(((h + step - 1) // step, (w + step - 1) // step), np.float32)
    for i, v in enumerate(range(0, h, step)):
        for j, u in enumerate(range(0, w, step)):
            grid[i, j] = float(np.linalg.norm(model.xy_at(np.array([u, v], dtype=np.float64), z)))
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_NEAREST)


def split_territory(masks: Dict[str, np.ndarray], models: Dict[str, ClickModel], shape: Tuple[int, int], args: Args) -> None:
    """Give every pixel to exactly one arm: whichever base it is nearer to, among those that
    can reach it. Overlapping shaded regions look like a choice you do not have, and left the
    picker free to send the far arm across the table for something under the near one's nose."""
    if len(masks) < 2:
        return
    dist = {side: base_distance(m, shape, (m.table_z or 0.08) - args.grasp_depth) for side, m in models.items() if side in masks}
    sides = list(dist)
    a, b = sides[0], sides[1]
    nearer_a = dist[a] <= dist[b]
    both = masks[a] & masks[b]
    masks[a] = masks[a] & (~both | nearer_a)
    masks[b] = masks[b] & (~both | ~nearer_a)


def fill_masks(holder: Dict[str, Dict[str, np.ndarray]], models: Dict[str, ClickModel], shape: Tuple[int, int], args: Args, on_table: Optional[np.ndarray] = None) -> None:
    """Load or compute each arm's reachable-region mask; runs in a background thread.

    The result is published by rebinding one key at the end -- filling a shared dict in place
    while the display loop iterates it is a crash waiting to happen (and was one).
    """
    from arm import Kinematics

    masks: Dict[str, np.ndarray] = {}
    for side, m in models.items():
        if m.table_z is None:
            continue
        path = cache_path(side, args, shape)
        calib = _HERE / "calib" / f"click_{side}.json"
        if path.exists() and path.stat().st_mtime > calib.stat().st_mtime:
            masks[side] = np.load(path)
            print(f"[reach] {side}: {masks[side].mean() * 100:.0f}% reachable (cached)")
            continue
        t = time.time()
        mask = reach_mask(m, Kinematics(), shape, m.table_z, args, on_table=on_table)
        masks[side] = mask
        np.save(path, mask)
        print(f"[reach] {side}: {mask.mean() * 100:.0f}% of the frame reachable ({time.time() - t:.0f}s, cached for next time)")
    split_territory(masks, models, shape, args)
    if len(masks) > 1:
        print("[reach] " + ", ".join(f"{s}: {m.mean() * 100:.0f}%" for s, m in masks.items()) + " after splitting the overlap by which base is nearer")
    holder["masks"] = masks


def draw(frame: Frame, clicks: Clicks, plan: Optional[Dict[str, object]], args: Args, note: str, note2: str = "", masks: Optional[Dict[str, np.ndarray]] = None, status: str = "") -> np.ndarray:
    img = frame.color.copy()
    if masks:
        tint = {"left": (0, 26, 0), "right": (26, 12, 0)}
        for side, m in list(masks.items()):
            edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            img[m] = np.clip(img[m].astype(np.int16) + np.array(tint.get(side, (0, 20, 20))), 0, 255).astype(np.uint8)
            img[edge] = (0, 255, 0) if side == "left" else (255, 160, 0)
    colors = [(0, 255, 255), (0, 255, 255), (255, 0, 255), (255, 0, 255)]
    for i, pt in enumerate(clicks.pts):
        cv2.drawMarker(img, pt, colors[i], cv2.MARKER_CROSS, 22, 2)
    if len(clicks.pts) >= 2:
        cv2.line(img, clicks.pts[0], clicks.pts[1], (0, 255, 255), 2)
        mid = ((clicks.pts[0][0] + clicks.pts[1][0]) // 2, (clicks.pts[0][1] + clicks.pts[1][1]) // 2)
        cv2.circle(img, mid, 6, (0, 255, 255), -1)
        cv2.putText(img, "grasp", (mid[0] + 10, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if len(clicks.pts) == 4:
        cv2.line(img, clicks.pts[2], clicks.pts[3], (255, 0, 255), 2)
        mid = ((clicks.pts[2][0] + clicks.pts[3][0]) // 2, (clicks.pts[2][1] + clicks.pts[3][1]) // 2)
        cv2.circle(img, mid, 6, (255, 0, 255), -1)
        cv2.putText(img, "release", (mid[0] + 10, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    lines = [
        "click 2 points on the object (centre + finger direction), then 2 for the release"
        + ("" if masks else "   [reachable areas still loading]"),
        f"grasp depth {args.grasp_depth * 100:.1f} cm ([ ])   release {args.release_height * 100:.1f} cm above the surface (- +)   {note}",
    ]
    if plan is not None:
        g, p = plan["g"], plan["p"]
        lines.append(
            f"{plan['side']} arm: grasp ({plan['grasp'][0]:.3f},{plan['grasp'][1]:.3f},{plan['grasp'][2]:.3f}) "
            f"open {plan['grip_open']:.2f} ({plan['grip_open'] * OPEN_STROKE * 100:.1f} cm)  "
            f"release ({plan['place'][0]:.3f},{plan['place'][1]:.3f},{plan['place'][2]:.3f})  "
            f"lean {float(plan.get('tilt_g', 0)):.0f}/{float(plan.get('tilt_p', 0)):.0f} deg  lift headroom {float(plan['headroom']) * 100:.0f} cm"
        )
    elif clicks.ready:
        lines.append(f"cannot do it -- {note2}" if note2 else "cannot do it (click inside a shaded area)")
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if status:
        y = 28 + 26 * len(lines) + 6
        colour = (60, 60, 255) if status.startswith("REC") else (200, 200, 200)
        cv2.putText(img, status, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, status, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, 2, cv2.LINE_AA)
    return img


class Recorder:
    """One episode per start/end key press, in the same layout bimanual_teleop_record.py writes.

    <save-root>/<task>/episode_NNNN/{top,left_wrist,right_wrist}.mp4 + low_dim.npz + meta.json,
    meta.json written last so its presence marks a complete episode. The low-dim stream is
    both arms' measured joints and EEF poses (state) and the commands actually streamed to
    them (action) -- the same keys as a teleop episode, so the same tools read both.

    An episode is *not* one pick: it runs from ``s`` to ``e`` and may hold any number of picks
    (and the idle clicking between them). Each pick is appended to ``runs`` with the frame span
    it covers, so meta.json still says what happened and where inside the episode.
    """

    def __init__(self, args: Args, rig: CameraRig, arms: Dict[str, Arm], speaker) -> None:
        self.args = args
        self.rig = rig
        self.arms = arms
        self.speaker = speaker
        self.task_dir = Path(args.save_root).expanduser() / args.task
        self.writer: Optional[rec.EpisodeWriter] = None
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last: Optional[Path] = None
        self.ep: Optional[Path] = None
        self.runs: List[Dict[str, object]] = []
        self.t0 = 0.0
        self.kin = next(iter(arms.values())).kin if arms else None

    @property
    def active(self) -> bool:
        return self.writer is not None

    @property
    def n_frames(self) -> int:
        return self.writer.n_frames if self.writer is not None else 0

    def status(self) -> str:
        """The one line the live view shows about recording."""
        if not self.args.record:
            return "recording off -- nothing will be written"
        if not self.active:
            return "not recording -- s starts an episode" + (f"   (last: {self.last.name})" if self.last else "")
        n = len(self.runs)
        return (
            f"REC {self.ep.name if self.ep else '?'}  {time.time() - self.t0:5.0f}s  "
            f"{n} pick{'' if n == 1 else 's'}   e = end, x = throw away"
        )

    def start(self) -> Optional[Path]:
        if not self.args.record:
            print("[record] recording is off (--no-record); nothing will be written")
            self.speaker.say("recording is off")
            return None
        if self.active:
            print("[record] already recording -- e ends this episode")
            return self.ep
        self.task_dir.mkdir(parents=True, exist_ok=True)
        index = rec._next_episode_index(self.task_dir)
        ep = self.task_dir / f"episode_{index:04d}"
        self.writer = rec.EpisodeWriter(ep, fps=round(self.args.record_fps))
        self.ep = ep
        self.runs = []
        self.t0 = time.time()
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._loop, name="record", daemon=True)
        self.thread.start()
        print(f"[record] {ep} -- recording; e ends the episode")
        self.speaker.say(f"recording {ep.name.replace('_', ' ')}")
        return ep

    def note_run(self, info: Dict[str, object]) -> None:
        """Remember what one pick inside the running episode was. Ignored when not recording."""
        if self.active:
            self.runs.append(info)

    def _sample(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for side in ("left", "right"):
            a = self.arms.get(side)
            if a is None:
                continue
            q = a.q()
            pos, quat = a.kin.grasp(q)
            out[f"joint_pos_{side}"] = q[:6].copy()
            out[f"gripper_{side}"] = float(q[6])
            out[f"eef_pos_{side}"] = pos
            out[f"eef_quat_{side}"] = quat
            cmd = a.q_cmd
            out[f"action_joint_pos_{side}"] = cmd[:6].copy()
            out[f"action_gripper_{side}"] = float(cmd[6])
        return out

    def _loop(self) -> None:
        dt = 1.0 / float(self.args.record_fps)
        writer = self.writer
        next_tick = time.monotonic()
        while not self.stop_flag.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(dt, next_tick - now))
                continue
            next_tick += dt
            frames = {}
            for role in self.rig.cams:
                f = self.rig.grab(role)
                frames[role] = (f.color, f.t)
            try:
                if self.writer is None:
                    return
                writer.add(self._sample(), frames)
            except Exception as e:  # a recording must never take the arms down with it
                print(f"[record] dropped a tick ({type(e).__name__}: {e})")
                if self.writer is None:
                    return

    def _halt(self) -> Optional[rec.EpisodeWriter]:
        """Stop the sampling thread and hand back the writer, or None if idle."""
        writer = self.writer
        if writer is None:
            return None
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        return writer

    def stop(self, extra: Optional[Dict[str, object]] = None) -> Optional[Path]:
        """End the episode and write it out. Metadata never costs us the recording."""
        writer = self._halt()
        if writer is None:
            print("[record] not recording -- s starts an episode")
            return None
        n = writer.n_frames
        if n < 2:
            writer.discard()
            print("[record] nothing recorded -- discarded")
            self.writer, self.ep = None, None
            return None
        meta: Dict[str, object] = {
            "run": "click_pick",
            "task": self.args.task,
            "n_runs": len(self.runs),
            "runs": self.runs,
            "duration_s": round(time.time() - self.t0, 2),
            "completed": bool(self.runs) and all(r.get("completed", True) for r in self.runs),
        }
        if extra:
            meta.update(extra)
        try:
            meta["args"] = {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(self.args).items()
            }
            path = writer.save(meta)
        except Exception as e:  # never lose the episode over a bad metadata field
            print(f"[record] metadata failed ({type(e).__name__}: {e}); saving without it")
            path = writer.save({"run": "click_pick", "task": self.args.task, "meta_error": str(e)})
        self.writer, self.ep = None, None
        self.last = path
        print(f"[record] saved {path} ({n} frames, {len(self.runs)} picks)")
        self.speaker.say(f"saved {path.name.replace('_', ' ')}")
        return path

    def abort(self) -> None:
        """Throw the episode being recorded away."""
        writer = self._halt()
        if writer is None:
            print("[record] not recording -- nothing to throw away")
            return
        name = self.ep.name if self.ep else "episode"
        writer.discard()
        self.writer, self.ep = None, None
        print(f"[record] threw {name} away")
        self.speaker.say(f"threw {name.replace('_', ' ')} away")

    def discard_last(self) -> None:
        if self.last is None or not self.last.exists():
            self.speaker.say("nothing to discard")
            return
        rec._retire_episode(self.last)
        print(f"[record] discarded {self.last.name}")
        self.speaker.say(f"discarded {self.last.name.replace('_', ' ')}")
        self.last = None


def _port_served(port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def launch_followers(args: Args, procs: List["subprocess.Popen[bytes]"]) -> None:
    """Bring up one minimum_gello follower server per arm -- the recorder's launch, no leaders.

    A port that is already served is attached to rather than launched over: another follower
    (scripts/run_policy_followers.sh) holds those arms powered, and a second server would die
    on bind while this tool happily drove the first one. Must run before any thread starts:
    _spawn_gello uses preexec_fn.
    """
    channels = {"left": args.can_follower_left, "right": args.can_follower_right}
    started: Dict[str, Tuple[int, "subprocess.Popen[bytes]"]] = {}
    missing: List[str] = []
    for side in args.sides:
        port = PORTS[side]
        if _port_served(port):
            print(f"[attach] {side} follower already serving port {port} -- using it, and leaving it running")
            continue
        if not args.launch:
            missing.append(f"{side} (port {port})")
            continue
        if not args.sim:
            rec._check_can_interface(channels[side])
        cmd = [
            "--arm",
            args.arm,
            "--version",
            str(args.version),
            "--can_channel",
            channels[side],
            "--gripper",
            args.follower_gripper,
            "--server_port",
            str(port),
        ]
        if args.sim:
            cmd.append("--sim")
        print(f"[launch] {side} follower on {channels[side]}, RPC port {port}")
        started[side] = (port, rec._spawn_gello(cmd))
        procs.append(started[side][1])
    if missing:
        raise SystemExit(
            "[error] --no-launch, but nothing is listening for: "
            + ", ".join(missing)
            + " -- start them (scripts/run_policy_followers.sh) or drop --no-launch"
        )
    for side, (port, proc) in started.items():
        rec._wait_for_port(port, proc=proc)
        print(f"[launch] {side} follower up (pid {proc.pid})")


def shaped(arm: Arm, path: np.ndarray, args: Args, pin: Optional[float] = None) -> np.ndarray:
    """Even out a raw path and round its corners off, ready to stream."""
    path = arm.resample_path(path, spacing=args.path_spacing)
    return arm.smooth_path(
        path,
        radius=args.path_smooth,
        pin=args.pin_ends if pin is None else pin,
        spacing=args.path_spacing,
    )


def go_working(arm: Arm, args: Args) -> None:
    """Bring one arm to the working pose as a single eased curve.

    Same shape as poses.go_working -- straight up first, in case the fingers are down inside a
    box, then across -- but built as one path and streamed under the tip-speed cap, so the
    return between picks is as unhurried as the picks. Already-there is a no-op.
    """
    if float(np.max(np.abs(arm.q()[:6] - WORKING_Q[:6]))) < 0.05:
        return
    ways: List[np.ndarray] = []
    p, _ = arm.grasp_pose()
    if p[2] < 0.30:
        try:
            q6, _, _ = arm.kin.ik_reach(p + np.array([0.0, 0.0, 0.08]), arm.q_cmd)
            ways.append(np.concatenate([q6, [arm.q_cmd[6]]]))
        except ValueError:
            pass
    ways.append(WORKING_Q.copy())
    arm.move_smooth(shaped(arm, arm.joint_curve(ways), args, pin=args.lift_straight), v_eef=args.eef_speed, accel=args.accel, start_speed=args.start_speed)
    arm.wait_settled()


def go_home(arm: Arm, args: Args) -> None:
    """Fold one arm down to the rest pose, by way of the working pose.

    Straight from wherever it is to the folded pose would sweep the arm across the table; the
    working pose is clear of everything (go_working lifts first when the fingers are down in a
    box), and the fold from there comes down beside the base rather than over the objects.
    """
    go_working(arm, args)
    arm.move_smooth(shaped(arm, arm.joint_curve([HOME_Q.copy()]), args), v_eef=args.eef_speed, accel=args.accel, start_speed=args.start_speed)
    arm.wait_settled()


def run_plan(plan: Dict[str, object], args: Args, arm: Arm) -> None:
    """Run one pick as three streamed paths: reach, carry, retreat.

    The arm comes to rest exactly where a person would let it: on the object with the fingers
    closing, and over the release point with them opening. Everything else -- the fly-out, the
    turn into the descent, the lift out of the grasp, the swing across, the way back to the
    working pose -- is one continuous curve at a bounded tip speed, so there are no waypoint
    stops and no square corners in between.
    """
    grasp, place = np.asarray(plan["grasp"]), np.asarray(plan["place"])
    R_g, R_p = np.asarray(plan["R_g"]), np.asarray(plan["R_p"])

    def q7(q6: np.ndarray, grip: float) -> np.ndarray:
        return np.concatenate([np.asarray(q6)[:6], [grip]])

    def say(name: str, target: np.ndarray, res: MoveResult) -> None:
        now = arm.grasp_pose()[0]
        print(f"[move] {name:16s} -> {np.round(target, 3)}  at {np.round(now, 3)}  err {np.linalg.norm(now - target) * 1e3:.0f} mm {res.aborted}")

    go_working(arm, args)
    grip = float(plan["grip_open"])
    # ---- 1. reach ---------------------------------------------------------------------
    # Out over the object, round into the approach line and straight down onto the grasp, all
    # as one path: the fingers open to the clicked width on the way rather than in a separate
    # move first, and the arm never comes to rest above the object the way it used to.
    transit = arm.joint_curve([q7(plan["q_tg"], grip), q7(plan["q_pre"], grip)])
    descent = arm.line_curve(grasp, R_g, q_start=q7(plan["q_pre"], grip), seeds=(np.asarray(plan["q_grasp"]),))
    # The descent has to stop on contact, the fly-out must not false-trigger on its own
    # dynamics: the threshold rides along in the path as an eighth column.
    limit = np.concatenate([np.full(len(transit), arm.track_abort_rad), np.full(len(descent) - 1, args.contact_abort_rad)])
    reach = shaped(arm, np.column_stack([np.vstack([transit, descent[1:]]), limit]), args)
    res = arm.move_smooth(reach, v_eef=args.eef_speed, v_slow=args.descend_speed, slow_len=args.approach + 0.06, accel=args.accel, start_speed=args.start_speed)
    # The streamed path ends on the commanded target; the soft wrist joints sag a centimetre
    # under gravity, so the measured fingertips are still short of it. This is what closes that.
    if not res.aborted:
        res = arm.settle_onto(grasp, R_g, rounds=args.correct_rounds, v_eef=args.descend_speed)
    say("grasp", grasp, res)
    grip_res = arm.grip_until_contact(
        target=args.close, lag_limit=args.grip_lag, squeeze=args.squeeze, hold_wait=args.close_wait
    )
    if grip_res["contacted"]:
        print(f"       gripped at {grip_res['width_m'] * 100:.1f} cm -- stopped on contact, squeezing {args.squeeze * 96:.0f} mm worth")
    else:
        print("       closed all the way without touching anything -- the grasp probably missed")
    # ---- 2. carry ---------------------------------------------------------------------
    # Straight up out of the grasp -- --lift-straight of it is held exactly vertical, so the
    # object clears whatever it was sitting in -- then one arc over to the release point and
    # down onto it. Not strict: however far up this pose can carry it is far enough, and the
    # curve that follows picks it up from wherever the lift got to.
    held = float(arm.q_cmd[6])
    p_now = arm.kin.grasp(arm.q_cmd)[0]
    lift = np.array([p_now[0], p_now[1], float(np.asarray(plan["transit_g"])[2])])
    try:
        rise = arm.line_curve(lift, R_g, q_start=q7(arm.q_cmd, held), strict=False)
    except ValueError as e:
        print(f"[move] no straight lift out of the grasp ({e}); curving out instead")
        rise = np.atleast_2d(q7(arm.q_cmd, held))
    # Both transit heights are the same clearance over the same table, so a curve through them
    # crosses dead level -- the one stretch of the run that still read as a machine. An apex
    # over the midpoint turns it into carrying something over: up, across, down.
    ways = [q7(plan["q_tp"], held), q7(plan["q_place"], held)]
    axis_g = np.asarray(plan["g"].axis) if hasattr(plan.get("g"), "axis") else None
    apex = arm.arch_point(rise[-1], np.asarray(plan["transit_p"]), R_g, height=args.carry_arc, finger_axis=axis_g)
    if apex is not None:
        ways.insert(0, q7(apex, held))
    else:
        print(f"[move] no room to arc the carry over by {args.carry_arc * 100:.0f} cm; going across level")
    carry = arm.joint_curve(ways, q_start=rise[-1])
    path = shaped(arm, np.vstack([rise, carry[1:]]), args, pin=args.lift_straight)
    res = arm.move_smooth(path, v_eef=args.eef_speed, v_slow=args.place_speed, slow_len=0.10, accel=args.accel, start_speed=args.start_speed)
    if not res.aborted:
        res = arm.settle_onto(place, R_p, rounds=args.correct_rounds, v_eef=args.place_speed)
    say("release", place, res)
    # How far to open to let go: all the way by default, but a plan may say less -- inside
    # a box beside other things, fingers thrown wide at the bottom of a descent would sweep
    # into them (language_pick_place.py opens to the width it picked up with).
    release_open = float(plan.get("release_open", 1.0))
    arm.set_gripper(release_open, wait=args.open_wait)
    # ---- 3. retreat -------------------------------------------------------------------
    # Up off the release and back to the working pose in one curve, ready for the next click.
    # The fingers finish opening on the way up, once clear of whatever was released into.
    retreat = arm.joint_curve([q7(plan["q_tp"], release_open), WORKING_Q.copy()], q_start=q7(arm.q_cmd, release_open))
    arm.move_smooth(shaped(arm, retreat, args, pin=args.lift_straight), v_eef=args.eef_speed, accel=args.accel, start_speed=args.start_speed)
    arm.wait_settled()
    print("[done] back at the working pose")


def run(args: Args) -> None:
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
    # open both arms once: constructing one costs a second (RPC handshake + MuJoCo model), and
    # doing it when you press space is most of the wait before anything moves
    arms: Dict[str, Arm] = {}
    for side in models:
        try:
            arms[side] = Arm(side, PORTS[side], max_joint_vel=args.max_joint_vel, execute=args.execute, workspace=Workspace(z=(0.03, 0.6)))
        except Exception as e:
            print(f"[arm] {side} not available ({e})")
    # Park both arms before anything looks at the table. Wherever they were left -- inside the
    # box, over the objects -- they would otherwise sit in the top camera's view and corrupt
    # both the table-plane fit and the reachability shading, and the first pick would start
    # with a long move from an unknown pose. One arm at a time, so the two never cross.
    if args.execute and args.start_at_working:
        for side, arm in arms.items():
            print(f"[start] {side} -> working pose")
            go_working(arm, args)

    try:
        from perception import Scene

        prior = Scene.from_calib("left").table_plane_cam()
    except Exception:
        prior = None
    time.sleep(0.5)
    plane = table_plane_cam(rig.grab("top"), prior)
    print(f"[table] plane in the camera frame: n={np.round(plane[0], 3)} d={plane[1]:.3f}")
    clicks = Clicks()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, clicks.on_mouse)
    forced: Optional[str] = None
    mask_holder: Dict[str, Dict[str, np.ndarray]] = {"masks": {}}
    cache: Dict[str, object] = {}
    speaker = rec.Speaker(args.tts)
    recorder = Recorder(args, rig, arms, speaker)
    if args.record:
        n = rec._next_episode_index(recorder.task_dir) if recorder.task_dir.exists() else 0
        print(f"[record] task '{args.task}' -> {recorder.task_dir} (next episode {n:04d}, {len(rig.cams)} cameras)")
        print("[record] press s to start an episode, e to end it -- picks made outside one are not written")
        speaker.say(f"task {args.task.replace('_', ' ')}, ready")
    if args.show_reach:
        # In a thread, and cached: this takes half a minute the first time, and computing it
        # before the first imshow left the window black and unresponsive.
        f0 = rig.grab("top")
        # only shade actual table: the reach test is a fixed height, so without this the back
        # wall gets shaded as if you could grasp there
        pts_cam, uv = f0.point_cloud(f0.depth > 0.2)
        n, d = plane
        flat = np.abs(pts_cam @ n + d) < 0.15
        on_table = np.zeros(f0.depth.shape, bool)
        on_table[uv[flat, 1], uv[flat, 0]] = True
        on_table = cv2.morphologyEx(on_table.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(bool)
        threading.Thread(target=fill_masks, args=(mask_holder, models, f0.depth.shape, args, on_table), daemon=True).start()
    try:
        while True:
            frame = rig.grab("top")
            masks = mask_holder["masks"]  # one snapshot per frame; the thread swaps it whole
            plan = None
            why = ""
            note = f"arm: {forced or 'auto'}"
            if clicks.ready:
                # Only re-plan when something actually changed: each plan is a few hundred IK
                # solves, and running that on every displayed frame is what made the window
                # crawl the moment the fourth point went down.
                key = (tuple(clicks.pts), round(args.grasp_depth, 4), round(args.release_height, 4), round(args.max_tilt, 1), forced)
                if key != cache.get("key"):
                    hint = draw(frame, clicks, None, args, note, "planning...", masks, recorder.status())
                    cv2.imshow(WINDOW, cv2.resize(hint, (args.display_width, int(hint.shape[0] * clicks.scale))))
                    cv2.waitKey(1)
                    tried = []
                    mid = ((clicks.pts[0][0] + clicks.pts[1][0]) // 2, (clicks.pts[0][1] + clicks.pts[1][1]) // 2)
                    for side, m in models.items():
                        if forced not in (None, side):
                            continue
                        mask = masks.get(side)
                        other = next((mk for sd, mk in masks.items() if sd != side), None)
                        inside = mask is None or bool(mask[min(mask.shape[0] - 1, mid[1]), min(mask.shape[1] - 1, mid[0])])
                        in_other = other is not None and bool(other[min(other.shape[0] - 1, mid[1]), min(other.shape[1] - 1, mid[0])])
                        if not inside and in_other:
                            continue  # the other arm owns this spot; no need to say anything
                        if not inside:
                            tried.append({"side": side, "ok": False, "why": "grasp point outside its reachable area"})
                            continue
                        t = plan_for(side, m, frame, clicks, args, plane, kin=arms[side].kin if side in arms else None)
                        if t is not None:
                            tried.append(t)
                    cands = [t for t in tried if t.get("ok")]
                    # the arm whose territory the grasp point is in goes first, so the picture
                    # and the picker agree; the other one is the fallback
                    owner = next((sd for sd, mk in masks.items() if mk[min(mk.shape[0] - 1, mid[1]), min(mk.shape[1] - 1, mid[0])]), None)
                    cands.sort(key=lambda t: (t["side"] != owner,))
                    cache["key"] = key
                    cache["plan"] = cands[0] if cands else None
                    cache["why"] = "" if cache["plan"] else ("; ".join(f"{t['side']}: {t['why']}" for t in tried) or "no arm available")
                plan, why = cache.get("plan"), cache.get("why", "")
            img = draw(frame, clicks, plan, args, note, why, masks, recorder.status())
            clicks.scale = args.display_width / img.shape[1]
            cv2.imshow(WINDOW, cv2.resize(img, (args.display_width, int(img.shape[0] * clicks.scale))))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                clicks.pts.clear()
                cache.clear()
            if key == ord("u") and clicks.pts:
                clicks.pts.pop()
            if key == ord("["):
                args.grasp_depth = min(0.06, args.grasp_depth + 0.005)
            if key == ord("]"):
                args.grasp_depth = max(0.0, args.grasp_depth - 0.005)
            if key == ord("-"):
                args.release_height = max(0.005, args.release_height - 0.005)
            if key in (ord("+"), ord("=")):
                args.release_height = min(0.20, args.release_height + 0.005)
            if key == ord("l"):
                forced = "left"
            if key == ord("R"):
                forced = "right"
            if key == ord("a"):
                forced = None
            if key == ord("d"):
                recorder.discard_last()
            if key == ord("s"):
                recorder.start()
            if key == ord("e"):
                recorder.stop()
            if key == ord("x"):
                recorder.abort()
            if key in (ord(" "), 13, 10) and plan is not None:
                print(f"[run] {plan['side']} arm  grasp {np.round(np.asarray(plan['grasp']), 3)}  release {np.round(np.asarray(plan['place']), 3)}")
                if args.execute and plan["side"] in arms:
                    cv2.imshow(
                        WINDOW,
                        cv2.resize(
                            draw(frame, clicks, plan, args, "running...", "", masks, recorder.status()),
                            (args.display_width, int(img.shape[0] * clicks.scale)),
                        ),
                    )
                    cv2.waitKey(1)
                    speaker.say("picking")
                    ok, err = True, ""
                    frame_start = recorder.n_frames  # where this pick starts inside the episode
                    t_start = time.time()
                    try:
                        run_plan(plan, args, arms[str(plan["side"])])
                    except Exception as e:
                        ok, err = False, f"{type(e).__name__}: {e}"
                        print(f"[run] failed -- {err}")
                        speaker.say("the run failed")
                        # Never leave the arms where the failure caught them -- stalled over an
                        # object at the pre-grasp pose, they sit in the top camera's view and the
                        # next pick starts from an unknown place. Back to the working pose, one
                        # arm at a time; the idle one is already there, so that costs nothing.
                        # go_working lifts straight up first, so it does not drag across the
                        # object it stopped over.
                        for other, a in arms.items():
                            print(f"[recover] {other} -> working pose")
                            try:
                                go_working(a, args)
                            except Exception as e2:
                                print(f"[recover] {other} did not make it back ({type(e2).__name__}: {e2})")
                    if recorder.active:
                        try:
                            info: Dict[str, object] = {
                                "arm": plan["side"],
                                "clicks": [list(pt) for pt in clicks.pts],
                                "grasp": np.asarray(plan["grasp"]).tolist(),
                                "release": np.asarray(plan["place"]).tolist(),
                                "finger_axis_grasp": np.asarray(plan["g"].axis).tolist(),
                                "finger_axis_release": np.asarray(plan["p"].axis).tolist(),
                                "grip_open": float(plan["grip_open"]),
                                "grip_width_m": float(arms[str(plan["side"])].q_cmd[6]) * OPEN_STROKE,
                                "lean_deg": [float(plan.get("tilt_g", 0)), float(plan.get("tilt_p", 0))],
                                "frames": [frame_start, recorder.n_frames],
                                "t_start": t_start,
                                "t_end": time.time(),
                                "completed": ok,
                                "error": err,
                            }
                        except Exception as e:  # a bad metadata field must not cost us the pick
                            print(f"[record] could not describe that pick ({type(e).__name__}: {e})")
                            info = {"arm": str(plan["side"]), "completed": ok, "error": err or str(e)}
                        recorder.note_run(info)
                else:
                    print("[dry run] --execute to move")
                clicks.pts.clear()
                cache.clear()
    finally:
        if recorder.active:
            print("[record] quitting while recording -- saving the episode")
            recorder.stop({"stop_reason": "quit while recording"})
        # Park before anything is closed, one arm at a time so the two never cross. Nothing
        # here may raise: the steps after it are what release the hardware.
        if args.execute and args.park_on_exit and arms:
            speaker.say("going home")
            for side, arm in arms.items():
                print(f"[exit] {side} -> home pose")
                try:
                    go_home(arm, args)
                except Exception as e:
                    print(f"[exit] {side} did not reach home ({type(e).__name__}: {e})")
        cv2.destroyAllWindows()
        for a in arms.values():
            a.close()
        rig.close()


def main(args: Args) -> None:
    # Followers first, before this process starts a single thread: _spawn_gello uses
    # preexec_fn (PR_SET_PDEATHSIG), which is only safe from a single-threaded parent -- and
    # that death signal is what stops the arms if this tool is killed outright.
    procs: List["subprocess.Popen[bytes]"] = []
    try:
        launch_followers(args, procs)
        run(args)
    finally:
        if procs:
            print("[stop] the followers this run launched are exiting: motor torque goes to zero and")
            print("[stop] the arms go limp. To keep them powered between runs, start them separately")
            print("[stop] (scripts/run_policy_followers.sh) and pass --no-launch.")
            rec._terminate(procs)


if __name__ == "__main__":
    main(tyro.cli(Args))
