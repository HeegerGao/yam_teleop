"""Per-object grasp policies and the box placement spec.

A policy turns one scanned object (mask + 3D from scan_table.py) into a ``Grasp``: where the
fingertip centre goes, along which axis the fingers open, how deep below the object's top,
and whether the wrist has to be vertical. Policies are looked up by the object's name
(``POLICIES``); anything unknown falls back to a centred top-down grasp across the short side.

Placement inside the box is a ``PlaceSpec``: an offset from the box's centre in *image*
directions -- ``right`` and ``down`` in metres as seen in the top camera -- which is what a
person looking at scan_overlay.png naturally thinks in ("5 cm below the red dot"). It is
converted to the arm base frame with the camera's axes projected onto the table.

    grasp = plan_grasp(obj, scene)                # obj: a scan_result.json record + mask
    p_base = PlaceSpec(right=0.0, down=0.05).resolve(box, scene)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from perception import Scene


@dataclass
class TableObject:
    """One scanned object: the scan_result.json record plus its mask and per-pixel base xyz."""

    rec: Dict[str, Any]
    mask: np.ndarray  # HxW bool
    xyz: np.ndarray  # Nx3 base-frame points of the mask (valid depth only)
    height: np.ndarray  # N heights above the table

    @property
    def name(self) -> str:
        return str(self.rec.get("name", "object"))

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.rec["center_xyz"], dtype=np.float64)

    @property
    def top_z(self) -> float:
        return float(self.rec["top_z"])

    @property
    def table_z(self) -> float:
        return float(self.rec["table_z"])

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.rec["size"], dtype=np.float64)  # long, short, height

    @property
    def yaw(self) -> float:
        return float(self.rec["yaw"])

    def long_axis(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    def short_axis(self) -> np.ndarray:
        return np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])

    def top_band_center(self, band: float = 0.01) -> np.ndarray:
        """Centroid of the points within ``band`` of the object's top (rim / top face)."""
        sel = self.height >= self.height.max() - band
        return self.xyz[sel].mean(0) if sel.sum() >= 10 else self.center


@dataclass
class Grasp:
    pos: np.ndarray  # fingertip-centre target, base frame
    finger_axis: np.ndarray  # fingers open along this (base frame, sign-free)
    vertical_only: bool = False  # refuse a tilted wrist (a rim pinch needs one finger inside)
    approach: float = 0.10  # pre-grasp height above the object's top
    close: float = 0.0  # gripper command when closed
    note: str = ""
    bottom_z: float = 0.0  # where the object's bottom is (for the placement height)
    body_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    """Object centre minus fingertip centre (base xy) while held: a rim-pinched cup hangs a
    radius away from the fingers, so the release point is shifted by -body_offset."""

    @property
    def grip_above_bottom(self) -> float:
        return float(self.pos[2] - self.bottom_z)


def _min_grasp_z(obj: TableObject, below_top: float, floor_clear: float = 0.012) -> float:
    return max(obj.table_z + floor_clear, obj.top_z - below_top)


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def grasp_topdown_short_side(obj: TableObject, scene: Scene, below_top: float = 0.022, **_: Any) -> Grasp:
    """Generic: centre of the top band, fingers across the footprint's short side."""
    c = obj.top_band_center()
    return Grasp(
        pos=np.array([c[0], c[1], _min_grasp_z(obj, below_top)]),
        finger_axis=obj.short_axis(),
        bottom_z=obj.table_z,
        note=f"top-down across the {obj.size[1] * 100:.1f} cm side",
    )


def grasp_carrot(obj: TableObject, scene: Scene, **_: Any) -> Grasp:
    """A long thin thing: mid-length, fingers across the short side, low (it is only ~2.5 cm)."""
    c = obj.center
    return Grasp(
        pos=np.array([c[0], c[1], _min_grasp_z(obj, 0.014)]),
        finger_axis=obj.short_axis(),
        bottom_z=obj.table_z,
        note="mid-length, across the short side",
    )


def grasp_bread(obj: TableObject, scene: Scene, **_: Any) -> Grasp:
    c = obj.center
    return Grasp(
        pos=np.array([c[0], c[1], _min_grasp_z(obj, 0.03)]),
        finger_axis=obj.short_axis(),
        bottom_z=obj.table_z,
        note="mid-length, across the loaf",
    )


def grasp_plush(obj: TableObject, scene: Scene, **_: Any) -> Grasp:
    """Soft: go deep (3.5 cm below the top) and squeeze."""
    c = obj.top_band_center(band=0.02)
    return Grasp(
        pos=np.array([c[0], c[1], _min_grasp_z(obj, 0.035)]),
        finger_axis=obj.short_axis(),
        bottom_z=obj.table_z,
        note="deep top-down squeeze",
    )


def wall_clearance(container: TableObject, p: np.ndarray, wall: float = 0.012) -> float:
    """Distance from base-frame xy ``p`` to the nearest inner wall of ``container`` (its
    footprint minus the wall thickness); negative when outside."""
    d = p[:2] - container.center[:2]
    u = abs(float(d @ container.long_axis()[:2]))
    v = abs(float(d @ container.short_axis()[:2]))
    return float(min(container.size[0] / 2 - wall - u, container.size[1] / 2 - wall - v))


def grasp_cup_wall(obj: TableObject, scene: Scene, others: Optional[list] = None, container: Optional[TableObject] = None, **_: Any) -> Grasp:
    """Pinch the wall: one finger inside the cup, one outside, at a rim point away from the
    handle and with the most room for the outer finger.

    The rim is the top band; a circle fitted to it gives the body's centre and radius. The
    handle shows up as rim points well outside that circle (the footprint centroid is no use:
    the camera sees the far inner wall, which drags it away from the camera). Eight rim
    angles are scored: those within 50 deg of the handle are out, the rest by the distance
    from where the outer fingertip lands to the nearest other object."""
    rim = obj.xyz[obj.height >= obj.height.max() - 0.012][:, :2]
    if len(rim) < 20:
        rim = obj.xyz[:, :2]
    A = np.column_stack([rim[:, 0], rim[:, 1], np.ones(len(rim))])
    b = -(rim[:, 0] ** 2 + rim[:, 1] ** 2)
    # robust: fit, drop points far outside the circle (the handle), refit
    for _ in range(2):
        sol, *_r = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -sol[0] / 2, -sol[1] / 2
        r = float(np.sqrt(max(cx**2 + cy**2 - sol[2], 1e-6)))
        dist = np.linalg.norm(rim - [cx, cy], axis=1)
        keep = dist < r + 0.006
        rim, A, b = rim[keep], A[keep], b[keep]
    r = float(np.clip(r, 0.02, 0.06))
    centre = np.array([cx, cy])
    allrim = obj.xyz[obj.height >= obj.height.max() - 0.012][:, :2]
    d_all = np.linalg.norm(allrim - centre, axis=1)
    outliers = allrim[d_all > r + 0.01]
    handle_dir = None
    if len(outliers) >= 15:
        handle_dir = outliers.mean(0) - centre
        handle_dir /= np.linalg.norm(handle_dir)
    # candidate rim angles
    obstacles = None
    if others:
        obstacles = np.concatenate([o.xyz[:, :2] for o in others if o is not obj]) if len(others) > 1 else None
    best = None
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        radial = np.array([np.cos(ang), np.sin(ang)])
        if handle_dir is not None and float(radial @ handle_dir) > np.cos(np.radians(50)):
            continue
        p = centre + r * radial
        outer_tip = centre + (r + 0.048) * radial
        inner_tip = centre + (r - 0.048 - 0.010) * radial
        clearance = 1.0
        if obstacles is not None and len(obstacles):
            clearance = float(np.min(np.linalg.norm(obstacles - outer_tip, axis=1)))
        if container is not None:
            # inside a box: both open fingertips must clear the walls (the gripper body rides
            # above the rim, only the fingers go in). The footprint model over-estimates the
            # box (flaps), so the scanned wall points in ``others`` are the real test: the
            # outer fingertip must land >= 2.5 cm from anything, the wall's rim included.
            wall_c = min(wall_clearance(container, outer_tip), wall_clearance(container, inner_tip))
            if wall_c < 0.01 or clearance < 0.025:
                continue
            clearance = min(clearance, wall_c)
        # a rim point on the base-facing side is the easiest reach, and inside a box it is
        # what makes the lift-out work: the vertical wrist reaches highest close to the base,
        # and the forward tilt used above the box then swings the cup body *up*
        score = min(clearance, 0.12) - (0.08 if container is not None else 0.02) * radial[0]
        if best is None or score > best[0]:
            best = (score, p, radial, clearance)
    if best is None:
        raise ValueError("no rim point keeps both fingers clear of the box walls and the handle")
    _, p, radial, clearance = best
    # aim the fingertip centre 1 cm inside the rim: the inner tip then lands ~3.9 cm in from
    # the wall and the outer ~3.9 cm out, so a 1.5 cm position error still leaves one finger
    # on each side of the wall
    p = p - (0.018 if container is not None else 0.010) * radial
    # inside a box pinch deeper (fingertips ~1 cm above the cup's floor): the cup then hangs
    # only ~1 cm below the fingertips and clears the rim sooner on the way out
    z = _min_grasp_z(obj, 0.035 if container is not None else 0.025)
    return Grasp(
        pos=np.array([p[0], p[1], z]),
        finger_axis=np.array([radial[0], radial[1], 0.0]),  # radial: one tip in, one out
        vertical_only=True,
        approach=0.08,
        bottom_z=obj.table_z,
        body_offset=centre - p,
        note=(
            f"rim pinch, body centre ({cx:.3f},{cy:.3f}) r {r * 100:.1f} cm, handle "
            f"{'at ' + str(np.round(handle_dir, 2)) if handle_dir is not None else 'not seen'}, outer-finger clearance {clearance * 100:.1f} cm"
        ),
    )


def grasp_hammer(obj: TableObject, scene: Scene, **_: Any) -> Grasp:
    """Grab the handle: the thinner end along the long axis. Width is measured in bins along
    the axis; the head is the wide end, the grasp sits a third of the way in from the other."""
    ax = obj.long_axis()[:2]
    c = obj.center[:2]
    t = (obj.xyz[:, :2] - c) @ ax
    s = (obj.xyz[:, :2] - c) @ np.array([-ax[1], ax[0]])
    t_lo, t_hi = np.percentile(t, 2), np.percentile(t, 98)
    bins = np.linspace(t_lo, t_hi, 9)
    widths = []
    for i in range(8):
        sel = (t >= bins[i]) & (t < bins[i + 1])
        widths.append(np.percentile(s[sel], 95) - np.percentile(s[sel], 5) if sel.sum() > 5 else 0.0)
    head_at_hi = np.mean(widths[-3:]) > np.mean(widths[:3])
    t_grasp = t_lo + 0.3 * (t_hi - t_lo) if head_at_hi else t_hi - 0.3 * (t_hi - t_lo)
    p = c + ax * t_grasp
    return Grasp(
        pos=np.array([p[0], p[1], max(obj.table_z + 0.010, obj.table_z + 0.6 * min(obj.size[2], 0.03))]),
        finger_axis=np.array([-ax[1], ax[0], 0.0]),
        bottom_z=obj.table_z,
        note=f"handle ({'head at +' if head_at_hi else 'head at -'} end), across the handle",
    )


def grasp_rope(obj: TableObject, scene: Scene, **_: Any) -> Grasp:
    """Thick knot: the highest band's centroid, fingers across the local direction."""
    sel = obj.height >= obj.height.max() - 0.02
    c = obj.xyz[sel].mean(0)
    xy = obj.xyz[sel][:, :2] - c[:2]
    evals, evecs = np.linalg.eigh(xy.T @ xy / max(len(xy), 1))
    long_ = evecs[:, int(np.argmax(evals))]
    return Grasp(
        pos=np.array([c[0], c[1], _min_grasp_z(obj, 0.03)]),
        finger_axis=np.array([-long_[1], long_[0], 0.0]),
        bottom_z=obj.table_z,
        note="highest knot, across it",
    )


POLICIES: Dict[str, Callable[..., Grasp]] = {
    "white cup": grasp_cup_wall,
    "cup": grasp_cup_wall,
    "carrot": grasp_carrot,
    "bread": grasp_bread,
    "toy rabbit": grasp_plush,
    "stuffed animal": grasp_plush,
    "hammer": grasp_hammer,
    "rope": grasp_rope,
    "lemon": grasp_topdown_short_side,
    "red cube": grasp_topdown_short_side,
    "blue cube": grasp_topdown_short_side,
    "green handle": grasp_hammer,  # the shovel: grab its handle the same way
    "shovel": grasp_hammer,
}
NOT_PICKABLE = {"cardboard box", "black box"}


def plan_grasp(
    obj: TableObject, scene: Scene, style: Optional[str] = None, others: Optional[list] = None, container: Optional[TableObject] = None
) -> Grasp:
    """``style`` overrides the name lookup (e.g. 'wall' for a cup, 'topdown' for anything);
    ``others`` (all scanned objects) lets a policy keep its fingers clear of the neighbours;
    ``container`` says the object sits inside that box (fingers must clear its walls)."""
    if style == "wall":
        return grasp_cup_wall(obj, scene, others=others, container=container)
    if style == "topdown":
        return grasp_topdown_short_side(obj, scene)
    if obj.name in NOT_PICKABLE:
        raise ValueError(f"{obj.name} is not something to pick up")
    return POLICIES.get(obj.name, grasp_topdown_short_side)(obj, scene, others=others, container=container)


def inside_container(obj: TableObject, container: TableObject) -> bool:
    return wall_clearance(container, obj.center[:2]) > 0.0


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


@dataclass
class PlaceSpec:
    """Where in the box, as an offset from the box's centre in top-image directions."""

    right: float = 0.0
    """Metres towards image-right (as seen in scan_overlay.png)."""
    down: float = 0.0
    """Metres towards image-down (towards the camera / the arms)."""
    clear: float = 0.06
    """Object bottom this far above the box floor when the gripper opens."""
    box_floor: float = 0.006

    def image_axes(self, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
        """Camera x (image right) and y (image down) projected onto the table, base frame."""
        n = scene.n / np.linalg.norm(scene.n)
        axes = []
        for a in (scene.T[:3, 0], scene.T[:3, 1]):
            v = a - n * (a @ n)
            axes.append(v / np.linalg.norm(v))
        return axes[0], axes[1]

    def resolve(self, box: TableObject, scene: Scene) -> np.ndarray:
        """Base-frame xy of the release point (z = box floor)."""
        r, d = self.image_axes(scene)
        c = box.center[:2] + (r * self.right + d * self.down)[:2]
        return np.array([c[0], c[1], box.table_z + self.box_floor])

    def inside(self, box: TableObject, p: np.ndarray, margin: float = 0.05) -> bool:
        """Is the release point at least ``margin`` inside the box's footprint?"""
        d = p[:2] - box.center[:2]
        u = abs(d @ box.long_axis()[:2])
        v = abs(d @ box.short_axis()[:2])
        return bool(u < box.size[0] / 2 - margin and v < box.size[1] / 2 - margin)


def load_objects(result_json: Path, masks_png: Path, frame_dir: Path, scene: Scene) -> Dict[int, TableObject]:
    """Rebuild TableObjects from a scan's outputs (record + mask + xyz from the saved frame)."""
    import json

    import cv2

    from cameras import Frame

    recs = json.loads(Path(result_json).read_text())["objects"]
    labels = cv2.imread(str(masks_png), cv2.IMREAD_UNCHANGED)
    frame = Frame.load(frame_dir, "scan_frame", role="top")
    out: Dict[int, TableObject] = {}
    for rec in recs:
        m = labels == rec["id"]
        pts_cam, _ = frame.point_cloud(m)
        xyz = scene.to_base(pts_cam)
        h = scene.height(xyz)
        out[rec["id"]] = TableObject(rec, m, xyz, h)
    return out
