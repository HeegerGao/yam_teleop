"""Does a joint-space path sweep into anything on the table?

A blanket "no link below z" rule is either unsafe or refuses perfectly good moves: unfolding
from the home pose dips the elbow to ~12 cm over the *near* table, where nothing stands. This
checks the arm's own mesh against the last scan's object points instead: a step is a collision
when an arm point is within ``margin_xy`` of an object point in xy *and* is lower than that
object's top plus ``margin_z``.

    obs = scan_obstacles(scene)                     # from calib/debug/scan_masks.png
    gap, s = path_clearance(kin, q_a, q_b, obs)     # gap < 0 means it hits
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from calibrate_top import ArmSurface  # noqa: E402
from cameras import Frame  # noqa: E402

DEBUG = _HERE / "calib" / "debug"


def scan_obstacles(scene, debug: Path = DEBUG, cell: float = 0.01, skip_ids: Tuple[int, ...] = ()) -> np.ndarray:
    """Nx3 base-frame points (1 cm grid) of everything the last scan found on the table."""
    labels = cv2.imread(str(debug / "scan_masks.png"), cv2.IMREAD_UNCHANGED)
    frame = Frame.load(debug, "scan_frame", role="top")
    mask = labels > 0
    for i in skip_ids:
        mask &= labels != i
    pts_cam, _ = frame.point_cloud(mask)
    pts = scene.to_base(pts_cam)
    keys, idx = np.unique(np.floor(pts / cell).astype(np.int64), axis=0, return_index=True)
    return pts[idx]


def path_clearance(
    kin, q_a: np.ndarray, q_b: np.ndarray, obstacles: np.ndarray, steps: int = 24, margin_xy: float = 0.03, per_geom: int = 1200
) -> Tuple[float, float]:
    """(smallest vertical gap between an arm point and the object under it, worst step in [0,1]).

    Negative means the arm passes through an object. Only arm points that have an object
    within ``margin_xy`` in xy are considered; the gap is arm z minus that object's top z.
    """
    surf = ArmSurface(kin, skip_bodies=("base", "link1"))
    tree = cKDTree(obstacles[:, :2])
    worst = (np.inf, 0.0)
    for i in range(steps + 1):
        s = i / steps
        pts = surf.points(q_a + (q_b - q_a) * s, per_geom=per_geom)
        near = tree.query_ball_point(pts[:, :2], margin_xy)
        for p, hits in zip(pts, near, strict=True):
            if not hits:
                continue
            gap = float(p[2] - obstacles[hits, 2].max())
            if gap < worst[0]:
                worst = (gap, s)
    return worst


def samples_clearance(kin, configs: np.ndarray, obstacles: np.ndarray, margin_xy: float = 0.03, per_geom: int = 900) -> Tuple[float, float]:
    """Same test as path_clearance but over an explicit list of joint configurations (a
    spline, say), so a curve is checked where it actually goes rather than where a straight
    interpolation would."""
    surf = ArmSurface(kin, skip_bodies=("base", "link1"))
    tree = cKDTree(obstacles[:, :2])
    worst = (np.inf, 0.0)
    for i, q in enumerate(configs):
        pts = surf.points(q, per_geom=per_geom)
        near = tree.query_ball_point(pts[:, :2], margin_xy)
        for p, hits in zip(pts, near, strict=True):
            if not hits:
                continue
            gap = float(p[2] - obstacles[hits, 2].max())
            if gap < worst[0]:
                worst = (gap, i / max(1, len(configs) - 1))
    return worst


def check_curve(kin, arm, waypoints, obstacles: Optional[np.ndarray], label: str, min_gap: float = 0.02, samples: int = 24) -> bool:
    """Check the spline the arm would actually fly through these waypoints."""
    if obstacles is None or not len(obstacles):
        return True
    path = arm.spline([arm.q_cmd] + list(waypoints), samples=samples)
    gap, s = samples_clearance(kin, path, obstacles)
    ok = gap >= min_gap
    if np.isfinite(gap):
        print(f"[clear] {label}: {gap * 100:+.1f} cm over the nearest object at {s * 100:.0f}% of the curve{'' if ok else '  <-- TOO CLOSE'}")
    return ok


def check_path(kin, q_a: np.ndarray, q_b: np.ndarray, obstacles: Optional[np.ndarray], label: str, min_gap: float = 0.02) -> bool:
    """Print and judge one path. True when it is clear (or there is nothing to hit)."""
    if obstacles is None or not len(obstacles):
        return True
    gap, s = path_clearance(kin, q_a, q_b, obstacles)
    ok = gap >= min_gap
    if np.isfinite(gap):
        print(f"[clear] {label}: {gap * 100:+.1f} cm over the nearest object at {s * 100:.0f}% of the way{'' if ok else '  <-- TOO CLOSE'}")
    return ok
