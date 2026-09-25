"""Calibrate "which pixel of the top image is this arm's fingertip pair" -- one arm at a time.

What is fitted is **one camera pose in that arm's base frame**, from marked fingertip sightings
spread over the image and over three heights. A pixel plus a height then maps to base xy by
intersecting that pixel's ray with the plane at that height -- exactly, at any height, with no
extrapolation. The file written is still the two plane homographies click_model.ClickModel
reads; they are now *derived* from the fitted pose rather than fitted independently, which is
what makes them mutually consistent, and they are emitted at two heights chosen to bracket
everything the picks use, so nothing downstream ever extrapolates either.

Why it was rebuilt (measured on the calibration this replaces):

* Coverage. Both arms were marked over a strip spanning ~20% of the image height (left: 44% of
  the width, right: 28%), while the planner will happily accept clicks over 44% and 74% of the
  frame. A homography fitted on a strip is badly conditioned across it: leave-one-out error was
  4.8-8.1 mm, against the 2.5-4.2 mm the in-sample residual advertised.
* Height. Both planes sat *above* where grasping happens. A flat object is grasped at
  table - 2 cm, which with planes at 9 and 15 cm is an extrapolation to t = -0.5, outside a
  6 cm baseline, with each plane's error amplified ~1.6x. The old two-plane model and a single
  camera pose fitted to the same points disagreed by 9.6-15.3 mm on average -- up to 5 cm --
  at exactly the height a real grasp lands, while agreeing to ~1 mm in the middle of the band.

So: cover the frame, sample several heights, fit one pose, emit planes that bracket the work.

The grid is laid out in **image** space -- a lattice of pixels inside a margin from the frame
edge -- and each cell is mapped back to a base-frame target through whatever estimate exists so
far. That estimate does not have to be good: the target only has to be reachable and land in
frame, because what is actually measured is the *marked pixel* against the *measured*
fingertip pose. A rough prior is therefore enough to bootstrap an accurate calibration, and the
estimate is refitted after every point, so the grid self-corrects as it goes.

Marking the fingertips is done by hand, on purpose. Automatic detection was tried three ways
(colour difference between open and closed, depth background subtraction, and both together)
and all three are fragile here: closing the gripper swings the arm's *shadow*, which is a far
bigger change than the fingers themselves, and the gripper is small, black, and often near the
frame edge. The fingers are held closed, so the two tips coincide at the grasp point -- which
is true whatever the wrist lean, and is why this can use the same leaning wrists the picks do
instead of being confined to the small patch a vertical wrist reaches.

Both follower arms are started by this script, so nothing else has to be running first; a port
that is already served is attached to instead, so scripts/run_policy_followers.sh still works
(pass --no-launch to require that). Each arm is folded down to the home pose when it is done,
before the followers this run started are stopped -- that cuts motor torque, and home is where
the arms rest unpowered.

Controls per pose:  left click = mark the point between the fingertips   space/enter = accept
                    s = skip this pose (do it whenever the tips are hidden or unclear)
                    u = undo the click    q = stop this arm and fit what has been marked

    python scripts/box_packing/calibrate_click.py                      # both arms
    python scripts/box_packing/calibrate_click.py --sides left
    python scripts/box_packing/calibrate_click.py --sides left --append # add to what is there
"""

from __future__ import annotations

import json
import subprocess
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

import followers  # noqa: E402
from arm import Arm, Workspace  # noqa: E402
from cameras import CameraRig  # noqa: E402
from click_model import ClickModel  # noqa: E402
from collide import samples_clearance, scan_obstacles  # noqa: E402
from perception import Scene  # noqa: E402
from poses import WORKING_Q, go_home, go_working  # noqa: E402

PORTS = followers.PORTS
CALIB = _HERE / "calib"
WINDOW = "calibrate: click between the fingertips"
TABLE_Z_FALLBACK = 0.078
"""Used only if calib/table_z_<side>.json is missing -- run measure_table_z.py first."""


@dataclass
class Args:
    sides: Tuple[str, ...] = ("left", "right")
    heights: Tuple[float, ...] = (0.015, 0.085, 0.16)
    """Fingertip heights to mark at, metres **above that arm's measured table**. Three spread
    over the working range is what pins the camera's distance and tilt down; two nearly
    coincident planes are what made the old calibration guess at the ray directions."""
    cols: int = 7
    rows: int = 4
    """Image lattice per height. cols x rows cells are tried; the unreachable ones fall out, so
    ask for more than you expect to mark."""
    u_margin: int = 110
    """Pixels of the left and right edges left alone -- the fingertips must be *inside* the
    frame to be marked, and a gripper half off the edge cannot be marked accurately."""
    v_margin: int = 70
    """The same for the top and bottom edges."""
    min_spacing_px: float = 60.0
    """Skip a cell whose target is predicted within this of a point already marked at the same
    height. Two marks in the same place cost a pose and add nothing to the fit."""
    max_tilt: float = 45.0
    """How far the wrist may lean, degrees. The closed fingertips meet at the grasp point at
    any lean, so this costs no accuracy and buys most of the table: a vertical wrist only
    reaches a small patch, which is exactly why the old calibration covered so little."""
    finger_axis: Tuple[float, float] = (1.0, 0.0)
    """Direction the fingers open, base frame. Held the same at every pose so the wrist looks
    the same in every picture and the operator is always clicking at the same feature."""
    plane_low: float = -0.05
    plane_high: float = 0.25
    """The two planes written to the file, metres relative to the table. They are computed from
    the fitted camera pose, not measured, so they need not be reachable -- and being wide apart
    and bracketing everything the picks use (grasp at -0.02, release at +0.05, transit at
    +0.16) is what stops ClickModel ever having to extrapolate."""
    seed_poses: int = 6
    """How many poses the base-frame fan aims to mark on an arm with no calibration at all,
    before the image lattice takes over. Four is the arithmetic minimum; six leaves margin."""
    append: bool = False
    """Add to the points already in calib/click_<side>.json instead of starting over."""
    hop: float = 0.05
    """Rise this far before travelling to the next pose, and come down onto it. The poses are
    spread much wider than they used to be, and a joint-space move between two far-apart poses
    at the same height dips in the middle."""
    check_obstacles: bool = True
    """Skip poses whose path sweeps into something from the last scan. Clear the table first
    and this has nothing to do."""
    max_joint_vel: float = 0.7
    settle: float = 0.25
    gripper: float = 0.0
    """Fingers closed while marking, so the two tips coincide at the grasp point."""
    display_width: int = 1280
    execute: bool = True
    launch: bool = True
    """Start a minimum_gello follower server per arm, so no other terminal is needed. Ports
    that are already served are attached to either way; --no-launch requires that."""
    park_on_exit: bool = True
    """Fold each arm down to the home pose when it is done, before the followers stop."""
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = followers.default_channels()["left"]
    """LEFT follower CAN netdev. Default comes from the mapping table scripts/can_map.conf."""
    can_follower_right: str = followers.default_channels()["right"]
    """RIGHT follower CAN netdev (from scripts/can_map.conf)."""
    sim: bool = False
    """Launch the followers in MuJoCo instead of on the CAN buses (no hardware needed)."""


# --------------------------------------------------------------------------------------
# the running estimate
# --------------------------------------------------------------------------------------
class Estimator:
    """Pixel <-> base-frame map, refitted after every point marked.

    Three tiers, best first: a camera pose once the points span some height (exact at any
    height); a plane homography for a height that has four points of its own; and whatever
    calibration was already on disk. Only the *first* few poses of a fresh arm have none of
    these, and those come from a base-frame fan instead.
    """

    def __init__(self, K: np.ndarray, prior: Optional[ClickModel] = None) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.prior = prior
        self.uv: List[np.ndarray] = []
        self.xyz: List[np.ndarray] = []
        self.plane_of: List[float] = []
        """The height each point was *asked* for. Not the same as the height it was measured
        at -- IK leaves a millimetre or two and the wrist sags a few more -- and grouping by the
        measured value instead splits one plane into a group per point, which leaves every
        group below the four a homography needs. The fit itself uses the measured position;
        only the grouping uses this."""
        self.pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.planes: Dict[float, np.ndarray] = {}  # height -> pixel->xy homography

    def add(self, uv: np.ndarray, xyz: np.ndarray, plane: Optional[float] = None) -> None:
        self.uv.append(np.asarray(uv, dtype=np.float64))
        self.xyz.append(np.asarray(xyz, dtype=np.float64))
        self.plane_of.append(float(xyz[2]) if plane is None else float(plane))
        self.refit()

    def at_plane(self, z: float) -> List[int]:
        return [i for i, p in enumerate(self.plane_of) if abs(p - z) < 1e-3]

    def refit(self) -> None:
        if len(self.uv) >= 6:
            xyz = np.stack(self.xyz)
            if float(xyz[:, 2].max() - xyz[:, 2].min()) > 0.03:
                fit = fit_camera(np.stack(self.uv), xyz, self.K)
                if fit is not None:
                    self.pose = fit
        self.planes = {}
        for z in set(self.plane_of):
            sel = self.at_plane(z)
            if len(sel) >= 4:
                src = np.array([self.uv[i] for i in sel], np.float32).reshape(-1, 1, 2)
                dst = np.array([self.xyz[i][:2] for i in sel], np.float32).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src, dst, 0)
                if H is not None:
                    self.planes[z] = H

    # ---- pixel -> base ---------------------------------------------------------------
    def xy_at(self, uv: np.ndarray, z: float) -> Optional[np.ndarray]:
        if self.pose is not None:
            return unproject(self.pose, self.K, uv, z)[:2]
        near = self._nearest_plane(z)
        if near is not None:
            return _apply(self.planes[near], uv)
        if self.prior is not None:
            return self.prior.xy_at(np.asarray(uv, dtype=np.float64), z)
        return None

    # ---- base -> pixel (for the overlay and the spacing test) ------------------------
    def uv_of(self, xyz: np.ndarray) -> Optional[np.ndarray]:
        xyz = np.asarray(xyz, dtype=np.float64)
        if self.pose is not None:
            rvec, tvec = self.pose
            return cv2.projectPoints(xyz.reshape(1, 3), rvec, tvec, self.K, None)[0].reshape(2)
        near = self._nearest_plane(float(xyz[2]))
        if near is not None:
            return _apply(np.linalg.inv(self.planes[near]), xyz[:2])
        if self.prior is not None:
            a = _apply(np.linalg.inv(self.prior.H0), xyz[:2])
            b = _apply(np.linalg.inv(self.prior.H1), xyz[:2])
            t = (float(xyz[2]) - self.prior.z0) / (self.prior.z1 - self.prior.z0)
            return a + (b - a) * t
        return None

    def _nearest_plane(self, z: float) -> Optional[float]:
        if not self.planes:
            return None
        return min(self.planes, key=lambda k: abs(k - z))

    @property
    def quality(self) -> str:
        if self.pose is not None:
            return f"camera pose from {len(self.uv)} points"
        if self.planes:
            return f"homography at z={sorted(self.planes)} from {len(self.uv)} points"
        return "the calibration already on disk" if self.prior is not None else "nothing yet"


def _apply(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.asarray(H, dtype=np.float64) @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def fit_camera(uv: np.ndarray, xyz: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Camera pose (rvec, tvec) taking base-frame points to the camera, by PnP + refinement."""
    if len(uv) < 4:
        return None
    obj = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    except cv2.error:
        return None
    return rvec, tvec


def unproject(pose: Tuple[np.ndarray, np.ndarray], K: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
    """Where the ray through pixel ``uv`` crosses the plane at height ``z``, in the base frame."""
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    centre = (-R.T @ tvec).ravel()
    ray = R.T @ np.linalg.inv(K) @ np.array([float(uv[0]), float(uv[1]), 1.0])
    return centre + ray * ((z - centre[2]) / ray[2])


def plane_homography(pose: Tuple[np.ndarray, np.ndarray], K: np.ndarray, z: float) -> np.ndarray:
    """Pixel -> base xy on the plane at height ``z``, derived from the camera pose.

    For X = (x, y, z) with z fixed, K(RX + t) = K [R0 | R1 | R2 z + t] (x, y, 1), so the
    base->pixel map is that 3x3; this returns its inverse, which is what ClickModel stores.
    """
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    M = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z + tvec.ravel()])
    return np.linalg.inv(M)


def loo_error(uv: np.ndarray, xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Leave-one-out base-frame error, metres. The honest number: an in-sample residual on a
    handful of points flatters any model with eight free parameters per plane."""
    out = []
    for i in range(len(uv)):
        keep = np.ones(len(uv), bool)
        keep[i] = False
        pose = fit_camera(uv[keep], xyz[keep], K)
        if pose is None:
            continue
        out.append(float(np.linalg.norm(unproject(pose, K, uv[i], xyz[i, 2])[:2] - xyz[i, :2])))
    return np.array(out)


# --------------------------------------------------------------------------------------
# where to go
# --------------------------------------------------------------------------------------
def image_lattice(args: Args, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Cells spread over the frame, in a snake order so the arm does not fly back and forth."""
    h, w = shape
    us = np.linspace(args.u_margin, w - args.u_margin, args.cols)
    vs = np.linspace(args.v_margin, h - args.v_margin, args.rows)
    cells: List[Tuple[int, int]] = []
    for i, v in enumerate(np.rint(vs).astype(int)):
        for u in np.rint(us if i % 2 == 0 else us[::-1]).astype(int):
            cells.append((u.item(), v.item()))
    return cells


def seed_fan(z: float) -> List[np.ndarray]:
    """Base-frame poses to start a *fresh* arm from, when there is no estimate at all.

    A fan rather than a grid: each arm's visible wedge sits at a different bearing, so a fan
    covering the reachable annulus is the only shape that finds it without being told where it
    is. The operator skips the ones out of frame, which is why it is deliberately generous --
    a fan that only lands three poses in shot leaves the run with nothing to fit.
    """
    out = []
    for r in (0.28, 0.36, 0.44):
        for a in np.radians((-35.0, -18.0, 0.0, 18.0, 35.0)):
            out.append(np.array([r * np.cos(a), r * np.sin(a), z]))
    return out


def reachable(arm: Arm, target: np.ndarray, obstacles: Optional[np.ndarray], axis: np.ndarray, args: Args) -> Optional[np.ndarray]:
    """Joints that put the closed fingertips on ``target``, leaning as little as it can."""
    if not arm.workspace.contains(target):
        return None
    tilts = tuple(float(t) for t in np.arange(0.0, args.max_tilt + 0.1, 15.0))
    try:
        q6, _, _ = arm.kin.ik_reach(target, WORKING_Q, tilts=tilts, finger_axis=axis)
    except ValueError:
        return None
    q7 = np.concatenate([q6, [args.gripper]])
    if obstacles is not None and len(obstacles):
        gap, _ = samples_clearance(arm.kin, arm.spline([arm.q_cmd, q7], samples=12), obstacles, per_geom=600)
        if np.isfinite(gap) and gap < 0.02:
            return None
    return q7


def go_to(arm: Arm, q7: np.ndarray, target: np.ndarray, axis: np.ndarray, args: Args) -> bool:
    """Travel to a pose by way of a point above it, so nothing is dragged across the table."""
    ways: List[np.ndarray] = []
    tilts = tuple(float(t) for t in np.arange(0.0, args.max_tilt + 0.1, 15.0))
    try:
        q_up, _, _ = arm.kin.ik_reach(target + np.array([0.0, 0.0, args.hop]), arm.q_cmd, tilts=tilts, finger_axis=axis)
        ways.append(np.concatenate([q_up, [args.gripper]]))
    except ValueError:
        pass
    ways.append(q7)
    res = arm.move_through(ways)
    arm.wait_settled()
    time.sleep(args.settle)
    return not res.aborted


# --------------------------------------------------------------------------------------
# marking
# --------------------------------------------------------------------------------------
class Clicker:
    def __init__(self) -> None:
        self.point: Optional[Tuple[int, int]] = None
        self.scale = 1.0

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (int(x / self.scale), int(y / self.scale))


def mark_pose(
    rig: CameraRig,
    clicker: Clicker,
    header: List[str],
    args: Args,
    est: Estimator,
    measured: np.ndarray,
    aimed: Optional[Tuple[int, int]],
    plane: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    """Live view until the operator accepts a click, skips, or stops.

    The overlay shows every point marked so far (so the coverage being built is visible) and
    where the current estimate *thinks* the fingertips are (so a gross error is obvious before
    it goes into the fit rather than after).
    """
    clicker.point = None
    guess = est.uv_of(measured)
    here = set(est.at_plane(float(measured[2]) if plane is None else plane))
    while True:
        img = rig.grab("top").color.copy()
        for i, uv in enumerate(est.uv):
            cv2.circle(img, (int(uv[0]), int(uv[1])), 4, (0, 200, 0) if i in here else (90, 90, 90), -1)
        if aimed is not None:
            cv2.circle(img, aimed, 22, (255, 160, 0), 1)
        if guess is not None and np.all(np.isfinite(guess)):
            cv2.drawMarker(img, (int(guess[0]), int(guess[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 1)
        if clicker.point is not None:
            cv2.drawMarker(img, clicker.point, (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(img, clicker.point, 14, (0, 255, 255), 1)
        for i, line in enumerate(header):
            cv2.putText(img, line, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        clicker.scale = args.display_width / img.shape[1]
        cv2.imshow(WINDOW, cv2.resize(img, (args.display_width, int(img.shape[0] * clicker.scale))))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord(" "), 13, 10) and clicker.point is not None:
            return clicker.point
        if key == ord("u"):
            clicker.point = None
        if key == ord("s"):
            return None
        if key == ord("q"):
            raise KeyboardInterrupt


# --------------------------------------------------------------------------------------
def table_height(side: str) -> float:
    try:
        return float(json.loads((CALIB / f"table_z_{side}.json").read_text())["table_z"])
    except Exception:
        print(f"[{side}] no calib/table_z_{side}.json -- assuming the table is at {TABLE_Z_FALLBACK:.3f}; run measure_table_z.py")
        return TABLE_Z_FALLBACK


def load_points(side: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """The raw sightings from a previous run, for --append: (pixel, measured xyz, plane asked for)."""
    try:
        data = json.loads((CALIB / f"click_{side}.json").read_text())
    except Exception:
        return [], [], []
    pts = data.get("points")
    if pts:
        return (
            [np.array(q["uv"], float) for q in pts],
            [np.array(q["xyz"], float) for q in pts],
            [float(q.get("plane", q["xyz"][2])) for q in pts],
        )
    uv, xyz, plane = [], [], []  # a file from before points were stored: rebuild from the planes
    for z, v in data.get("planes", {}).items():
        for a, b in zip(v.get("uv", []), v.get("xy", []), strict=True):
            uv.append(np.array(a, float))
            xyz.append(np.array([b[0], b[1], float(z)], float))
            plane.append(float(z))
    return uv, xyz, plane


def seed_pass(
    side: str,
    arm: Arm,
    args: Args,
    rig: CameraRig,
    clicker: Clicker,
    est: Estimator,
    obstacles: Optional[np.ndarray],
    axis: np.ndarray,
    z: float,
) -> None:
    """Get a *fresh* arm to the point where the image lattice can take over.

    Only runs when there is no calibration on disk and no points yet: with nothing to map a
    pixel through, the targets have to be named in the arm's own frame. A fan rather than a
    grid, because each arm's visible wedge sits at a different bearing; the operator skips the
    ones that are out of frame, and four marks are enough for a homography to take over.
    """
    fan = seed_fan(z)
    print(f"[{side}] no estimate at all -- seeding from a {len(fan)}-pose base-frame fan at z={z:.3f}")
    for i, target in enumerate(fan):
        if len(est.uv) >= args.seed_poses:
            break
        q7 = reachable(arm, target, obstacles, axis, args)
        if q7 is None or not go_to(arm, q7, target, axis, args):
            continue
        measured = arm.grasp_pose()[0]
        header = [
            f"{side} arm   SEEDING {len(est.uv)}/{args.seed_poses}   fan pose {i + 1}/{len(fan)}",
            f"fingertips at ({measured[0]:.3f}, {measured[1]:.3f}, {measured[2]:.3f})",
            "click between the fingertips, then SPACE.   s = skip (do that if they are off screen)",
        ]
        uv = mark_pose(rig, clicker, header, args, est, measured, None, plane=z)
        if uv is not None:
            est.add(np.array(uv, dtype=np.float64), measured.copy(), plane=z)
    if len(est.uv) < 4:
        raise SystemExit(
            f"[{side}] the seeding fan only got {len(est.uv)} poses in shot, and four are the minimum to map a "
            "pixel through at all -- so the lattice that follows would have nothing to aim with. Check the top "
            "camera is looking at this arm's part of the table, then rerun; --seed-poses and the fan radii in "
            "seed_fan() are the knobs if its reachable area is somewhere unusual."
        )


def calibrate_arm(side: str, args: Args, rig: CameraRig, clicker: Clicker, obstacles: Optional[np.ndarray]) -> None:
    table_z = table_height(side)
    heights = [table_z + h for h in args.heights]
    K = rig.grab("top").K
    shape = rig.grab("top").color.shape[:2]
    try:
        prior: Optional[ClickModel] = ClickModel.load(side)
        print(f"[{side}] bootstrapping from the calibration on disk ({prior.describe()})")
    except Exception:
        prior = None
        print(f"[{side}] no calibration on disk -- starting from a base-frame fan")
    est = Estimator(K, prior)
    if args.append:
        uv0, xyz0, plane0 = load_points(side)
        for a, b, pl in zip(uv0, xyz0, plane0, strict=True):
            est.uv.append(a)
            est.xyz.append(b)
            est.plane_of.append(pl)
        est.refit()
        print(f"[{side}] --append: carrying {len(uv0)} points forward ({est.quality})")

    arm = Arm(side, PORTS[side], max_joint_vel=args.max_joint_vel, execute=args.execute, workspace=Workspace(z=(0.02, 0.6)))
    axis = np.array([args.finger_axis[0], args.finger_axis[1], 0.0])
    try:
        go_working(arm)
        arm.set_gripper(args.gripper, wait=0.2)
        cells = image_lattice(args, shape)
        if est.xy_at(np.array([shape[1] / 2, shape[0] / 2]), heights[0]) is None:
            seed_pass(side, arm, args, rig, clicker, est, obstacles, axis, heights[0])
        for hi, z in enumerate(heights):
            marked = 0
            for i, cell in enumerate(cells):
                # the target is recomputed from the estimate as it stands *now*: every point
                # marked improves it, so the later cells of a height land nearer where they
                # were asked for than the earlier ones did
                xy = est.xy_at(np.array(cell, dtype=np.float64), z)
                if xy is None:
                    continue
                target = np.array([xy[0], xy[1], z])
                here = [est.uv[j] for j in est.at_plane(z)]
                if here and min(float(np.linalg.norm(np.array(cell, float) - o)) for o in here) < args.min_spacing_px:
                    continue
                q7 = reachable(arm, target, obstacles, axis, args)
                if q7 is None:
                    print(f"  [{hi + 1}.{i + 1:02d}] {np.round(target, 3)}: out of reach or blocked")
                    continue
                if not go_to(arm, q7, target, axis, args):
                    print(f"  [{hi + 1}.{i + 1:02d}] {np.round(target, 3)}: move aborted")
                    continue
                measured = arm.grasp_pose()[0]
                header = [
                    f"{side} arm   height {hi + 1}/{len(heights)} (z={z:.3f}, {(z - table_z) * 100:+.0f} cm over the table)"
                    f"   marked {len(est.uv)} ({marked} here)   estimate: {est.quality}",
                    f"fingertips at ({measured[0]:.3f}, {measured[1]:.3f}, {measured[2]:.3f})   "
                    "magenta = where the estimate thinks they are",
                    "click between the fingertips, then SPACE.   s = skip, u = undo, q = stop and fit",
                ]
                uv = mark_pose(rig, clicker, header, args, est, measured, cell, plane=z)
                if uv is None:
                    print(f"  [{hi + 1}.{i + 1:02d}] skipped")
                    continue
                before = est.uv_of(measured)
                est.add(np.array(uv, dtype=np.float64), measured.copy(), plane=z)
                marked += 1
                miss = "" if before is None else f"  (the estimate was {np.linalg.norm(before - np.array(uv)):.0f} px out)"
                print(f"  [{hi + 1}.{i + 1:02d}] {np.round(measured, 3)} -> pixel {uv}{miss}")
            if marked == 0:
                print(f"[{side}] z={z:.3f}: nothing marked at this height -- every cell was out of reach, blocked, "
                      "already covered, or skipped. Lower --heights, or raise --max-tilt, if that was not intended.")
            print(f"[{side}] z={z:.3f} ({(z - table_z) * 100:+.0f} cm): {marked} points marked, {len(est.uv)} in total")
    except KeyboardInterrupt:
        print(f"[{side}] stopped by the operator -- fitting what has been marked")
    finally:
        # Park before the RPC connection is dropped, whatever happened above: the followers
        # this run started are stopped on the way out, and that cuts torque.
        try:
            if args.execute and args.park_on_exit:
                print(f"[{side}] -> home pose")
                go_home(arm)
            else:
                go_working(arm)
        except Exception as e:
            print(f"[{side}] did not make it back ({type(e).__name__}: {e})")
        arm.close()
    write_calibration(side, est, table_z, args, shape)


def write_calibration(side: str, est: Estimator, table_z: float, args: Args, shape: Tuple[int, int]) -> None:
    if len(est.uv) < 6:
        print(f"[{side}] only {len(est.uv)} points -- not enough to fit; nothing written")
        return
    uv, xyz = np.stack(est.uv), np.stack(est.xyz)
    pose = fit_camera(uv, xyz, est.K)
    if pose is None:
        print(f"[{side}] the camera pose would not fit; nothing written")
        return
    rvec, tvec = pose
    proj = cv2.projectPoints(xyz.reshape(-1, 1, 3), rvec, tvec, est.K, None)[0].reshape(-1, 2)
    rep = np.linalg.norm(proj - uv, axis=1)
    loo = loo_error(uv, xyz, est.K)
    R, _ = cv2.Rodrigues(rvec)
    centre = (-R.T @ tvec).ravel()
    hull = cv2.convexHull(uv.astype(np.float32))
    span = float(cv2.contourArea(hull)) / float(shape[0] * shape[1])
    print(f"[{side}] {len(uv)} points at heights {sorted({round(float(p), 3) for p in xyz[:, 2]})}")
    print(f"[{side}] camera at {np.round(centre, 3)} in the base frame; reprojection {rep.mean():.2f} px mean, {rep.max():.2f} max")
    print(f"[{side}] LEAVE-ONE-OUT error {loo.mean() * 1000:.1f} mm mean, {np.median(loo) * 1000:.1f} median, {loo.max() * 1000:.1f} max")
    print(f"[{side}] the marked points span {span * 100:.0f}% of the frame "
          f"(u {uv[:, 0].min():.0f}..{uv[:, 0].max():.0f}, v {uv[:, 1].min():.0f}..{uv[:, 1].max():.0f})")
    if loo.mean() > 0.006:
        print(f"[{side}] that is worse than 6 mm -- rerun with --append to add poses, especially away from where the points cluster")

    zs = (table_z + args.plane_low, table_z + args.plane_high)
    out: Dict[str, object] = {
        "side": side,
        "method": "camera pose (solvePnP) over marked fingertip sightings; planes derived from it",
        "table_z": table_z,
        "K": est.K.tolist(),
        "camera": {"rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(), "centre_base": centre.tolist()},
        "fit": {
            "n_points": len(uv),
            "reprojection_px_mean": float(rep.mean()),
            "leave_one_out_m": loo.tolist(),
            "frame_span_frac": span,
        },
        "points": [
            {"uv": a.tolist(), "xyz": b.tolist(), "plane": pl}
            for a, b, pl in zip(est.uv, est.xyz, est.plane_of, strict=True)
        ],
        "planes": {},
    }
    for z in zs:
        sel = np.argsort(np.abs(xyz[:, 2] - z))[: max(4, len(xyz) // len(zs))]
        pred = np.array([unproject(pose, est.K, uv[i], float(xyz[i, 2]))[:2] for i in sel])
        out["planes"][f"{z:.4f}"] = {
            "H": plane_homography(pose, est.K, z).tolist(),
            "residual_m": np.linalg.norm(pred - xyz[sel, :2], axis=1).tolist(),
            "uv": uv[sel].tolist(),
            "xy": xyz[sel, :2].tolist(),
        }
    path = CALIB / f"click_{side}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[{side}] wrote {path}: planes at z={zs[0]:.3f}/{zs[1]:.3f}, bracketing grasp, release and transit")


def calibrate(args: Args) -> None:
    CALIB.mkdir(exist_ok=True)
    try:
        obstacles = scan_obstacles(Scene.from_calib("left")) if args.check_obstacles else None
        if obstacles is not None:
            print(f"[calib] {len(obstacles)} obstacle points from the last scan (poses sweeping into them are skipped)")
    except Exception as e:
        obstacles = None
        print(f"[calib] no usable scan ({e}); clear the table before running this")

    rig = CameraRig.open(roles=("top",))
    clicker = Clicker()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, clicker.on_mouse)
    try:
        for side in args.sides:
            calibrate_arm(side, args, rig, clicker, obstacles)
    finally:
        cv2.destroyAllWindows()
        rig.close()


def main(args: Args) -> None:
    # Followers first, before this process starts a single thread: _spawn_gello uses
    # preexec_fn (PR_SET_PDEATHSIG), which is only safe from a single-threaded parent -- and
    # that death signal is what stops the arms if this tool is killed outright.
    procs: List["subprocess.Popen[bytes]"] = []
    try:
        followers.launch(
            args.sides,
            procs,
            arm=args.arm,
            version=args.version,
            gripper=args.follower_gripper,
            channels={"left": args.can_follower_left, "right": args.can_follower_right},
            sim=args.sim,
            allow_launch=args.launch,
        )
        calibrate(args)
    finally:
        followers.terminate(procs)


if __name__ == "__main__":
    main(tyro.cli(Args))
