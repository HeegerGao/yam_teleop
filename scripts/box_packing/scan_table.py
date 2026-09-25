"""Scan the table: raise both arms out of the top camera's view, take one top frame, and
return every object on the table as a mask + centre pixel (+ base-frame 3D via the
calibration).

Three segmentation backends:
  hybrid (default) depth components split by SAM masks where one component holds several
                   objects (see segment_hybrid) -- the box stays whole, a cup against a cube
                   comes apart.
  depth (default)  points higher than --min-height above the calibrated table plane, cleaned
                   up and split into connected components. Model-free, runs in milliseconds,
                   and gives exactly "what stands on the table" -- the box included.
  sam              Segment Anything (facebook/sam-vit-base) prompted with a grid of points
                   over the table; masks that lie flat on the table are discarded using the
                   same height map. Slower, but follows object outlines in colour.

Outputs (in --out): scan_overlay.png (masks + centres + ids), scan_masks.png (uint16 label
image, 0 = background) and scan_result.json with, per object: id, area_px, center_uv, bbox,
center_xyz (base frame), top_z, size (long, short, height) and yaw.

    python scripts/box_packing/scan_table.py                    # arms stay where they are
    python scripts/box_packing/scan_table.py --backend sam
    python scripts/box_packing/scan_table.py --raise-arms       # lift the arms first
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Arm  # noqa: E402
from calibrate_top import ArmSurface  # noqa: E402
from cameras import CameraRig, Frame  # noqa: E402
from perception import Scene  # noqa: E402

PORTS = {"left": 1235, "right": 1234}
RAISED_Q = np.array([0.0, 1.57, 3.0, 0.0, 0.0, 0.0, 1.0])
"""Upper arm vertical, forearm straight up, gripper open and pointing up: a column above the
base, entirely outside the top camera's frustum (checked against the calibration) and clear
of the table."""


@dataclass
class Args:
    backend: str = "hybrid"
    """depth | sam | hybrid (depth components, split by SAM where they hold several objects)"""
    sides: Tuple[str, ...] = ("left", "right")
    raise_arms: bool = False
    """Lift both arms out of the view first. Off: at their home pose the arms are already
    outside the top camera's frustum, so the table is unobstructed."""
    return_home: bool = True
    """Bring the arms back to where they were after the frame is taken."""
    max_joint_vel: float = 0.5
    frames: int = 5
    """Top frames to median."""
    min_height: float = 0.012
    """Depth backend: points this far above the table are 'object'."""
    min_area: int = 250
    """Smallest component considered during segmentation, pixels."""
    min_object_area: int = 1200
    """Smallest object reported, pixels (smaller ones are table noise or loose fragments)."""
    merge_area: int = 3000
    """Labels smaller than this that touch a bigger label are merged into it (box flaps,
    hammer-head fragments)."""
    min_median_height: float = 0.006
    """An object whose mask is mostly at table level (median height below this) is noise.
    A hammer handle lying flat reads ~1 cm at its median, so this stays well below that."""
    sam_grid: int = 48
    """SAM backend: point-prompt spacing in pixels."""
    sam_min_iou: float = 0.85
    out: str = str(_HERE / "calib" / "debug")
    calib_side: str = "left"
    """Whose calibration expresses the 3D output (left arm base frame)."""
    names: bool = True
    """Name each mask with GroundingDINO (open-vocabulary detector) using --vocab."""
    vocab: Tuple[str, ...] = (
        "lemon", "bread", "carrot", "red cube", "blue cube", "white cup", "toy rabbit", "cardboard box",
        "hammer", "shovel", "rope", "green handle", "stuffed animal", "black box", "tin can", "orange",
        "banana", "apple", "bottle", "ball", "toy", "tool",
    )
    """Candidate names; a mask gets the best-matching detection or 'object'."""
    mask_arm: bool = True
    """Project the calibrated arm's own mesh (at its current joint pose) out of the frame, so
    it is never scanned as an object."""
    from_saved: bool = False
    """Re-run the segmentation on <out>/scan_frame_* instead of touching arms or cameras."""


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def check_raise_path(arm: Arm, q_from: np.ndarray, q_to: np.ndarray) -> Tuple[float, float]:
    """(lowest link z over the table, farthest link x) along the joint-space path."""
    surf = ArmSurface(arm.kin, skip_bodies=("base", "link1"))
    low, far = np.inf, -np.inf
    for s in np.linspace(0, 1, 20):
        pts = surf.points(q_from + (q_to - q_from) * s, per_geom=2000)
        over = pts[pts[:, 0] > 0.22]
        if len(over):
            low = min(low, float(over[:, 2].min()))
        far = max(far, float(pts[:, 0].max()))
    return low, far


def raise_arms(sides: Tuple[str, ...], vel: float) -> Dict[str, Tuple[Arm, np.ndarray]]:
    arms: Dict[str, Tuple[Arm, np.ndarray]] = {}
    for side in sides:
        arm = Arm(side, PORTS[side], max_joint_vel=vel)
        q0 = arm.q().copy()
        low, far = check_raise_path(arm, q0, RAISED_Q)
        print(f"[{side}] raising: lowest link over the table on the way z={low:.3f}, farthest reach x={far:.2f}")
        if low < 0.15 and far > 0.30:
            raise SystemExit(f"[{side}] the raise path sweeps low over the table -- refusing")
        arms[side] = (arm, q0)
    for side, (arm, _) in arms.items():
        res = arm.move_joints(RAISED_Q, settle=0.0)
        print(f"[{side}] raised: {'ok' if not res.aborted else res.aborted}")
    time.sleep(0.5)
    return arms


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------


def arm_pixels(frame: Frame, scene: "Scene", kin, q: np.ndarray, dilate: int = 21) -> np.ndarray:
    """Pixels covered by the arm at joint pose ``q``, from its own mesh through the
    calibration. Without this the gripper hanging into the frame is scanned as an object
    (it came back as a 15 cm 'tin can') and every path check then refuses to move."""
    from calibrate_top import ArmSurface, apply

    surf = ArmSurface(kin, skip_bodies=("base",))
    pts = surf.points(q, per_geom=4000)
    cam = scene.to_cam(pts)
    cam = cam[cam[:, 2] > 0.05]
    uv = frame.project(cam)
    h, w = frame.depth.shape
    m = np.zeros((h, w), np.uint8)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inside].astype(int)
    m[uv[:, 1], uv[:, 0]] = 1
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)).astype(bool)


def top_frame(rig: CameraRig, n: int) -> Frame:
    frames = [rig.grab_fresh("top", min_frames=2) for _ in range(n)]
    stack = np.stack([f.depth for f in frames])
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    f = frames[-1]
    return Frame(f.role, f.color, depth, f.K, f.t)


Extent = Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max of the table top


def height_map(frame: Frame, scene: Scene) -> Tuple[np.ndarray, np.ndarray, Extent]:
    """Per-pixel height above the table (NaN where no depth), base-frame xyz per pixel, and
    the table top's extent in base xy (so walls and curtains are not mistaken for objects).

    The plane is re-fitted on this very frame (RANSAC in the camera frame): the calibrated
    plane is good to a few mm at the landmarks but a small tilt error turns into a height
    gradient across a 60 cm table, which is exactly what a 1 cm threshold cannot tolerate."""
    h, w = frame.depth.shape
    valid = (frame.depth > 0.3) & (frame.depth < 1.5)
    v, u = np.nonzero(valid)
    z = frame.depth[v, u]
    pc = np.stack([(u - frame.K[0, 2]) / frame.K[0, 0] * z, (v - frame.K[1, 2]) / frame.K[1, 1] * z, z], axis=1)
    # refine the calibrated plane on this frame's own inliers (a free RANSAC can lock onto
    # the back wall, which is the larger plane in the image)
    n_cam, d_cam = scene.table_plane_cam()
    for _ in range(2):
        inl = pc[np.abs(pc @ n_cam + d_cam) < 0.015]
        cen = inl.mean(0)
        _, _, vt = np.linalg.svd(inl[:: max(1, len(inl) // 80000)] - cen, full_matrices=False)
        n_new = vt[2]
        if n_new @ n_cam < 0:
            n_new = -n_new
        n_cam, d_cam = n_new, float(-n_new @ cen)
    hc = pc @ n_cam + d_cam  # height above the refined plane (normal points at the camera)
    pb = scene.to_base(pc)
    H = np.full((h, w), np.nan, np.float32)
    H[v, u] = hc
    XYZ = np.full((h, w, 3), np.nan, np.float32)
    XYZ[v, u] = pb
    table = pb[np.abs(hc) < 0.006]
    extent = (
        float(np.percentile(table[:, 0], 0.5)),
        float(np.percentile(table[:, 0], 99.5)),
        float(np.percentile(table[:, 1], 0.5)),
        float(np.percentile(table[:, 1], 99.5)),
    )
    return H, XYZ, extent


def on_table_mask(XYZ: np.ndarray, extent: Extent, margin: float = 0.02, far_margin: float = 0.05) -> np.ndarray:
    """Near edge: none (the image's bottom edge cuts the table there, objects can straddle it).
    Far edge: generous, the back wall rises right behind it."""
    x, y = XYZ[..., 0], XYZ[..., 1]
    return (x > extent[0] - 0.05) & (x < extent[1] - far_margin) & (y > extent[2] + margin) & (y < extent[3] - margin)


def segment_depth(H: np.ndarray, XYZ: np.ndarray, extent: Extent, min_height: float, min_area: int, seed_height: float = 0.022, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Label image from the height map. Touching objects (a lemon against a cube) form one
    'above the table' blob, so it is split by a marker watershed: the seeds are the blobs at
    a higher threshold (each object's own top), grown downhill over the height map."""
    fg = (H > min_height) & (H < 0.35) & on_table_mask(XYZ, extent)
    if exclude is not None:
        fg &= ~exclude
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    seeds = (H > seed_height) & fg
    seeds = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, markers, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    markers = markers.astype(np.int32)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            markers[markers == i] = 0
    # low, flat blobs that never reach the seed height (a carrot, a thin tool) still count
    n2, low_lab, low_stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    next_id = int(markers.max()) + 1
    for i in range(1, n2):
        blob = low_lab == i
        if low_stats[i, cv2.CC_STAT_AREA] >= min_area and not np.any(markers[blob] > 0):
            markers[blob] = next_id
            next_id += 1
    # watershed over the height map (higher = 'deeper' basin), background marker outside fg
    hm = np.nan_to_num(H, nan=0.0)
    inv = np.clip(0.35 - hm, 0, 0.35) / 0.35
    img3 = cv2.cvtColor((inv * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers[~fg] = next_id  # background
    cv2.watershed(img3, markers)
    labels = np.where((markers > 0) & (markers < next_id) & fg, markers, 0)
    out = np.zeros(labels.shape, np.uint16)
    k = 0
    for i in np.unique(labels):
        if i == 0:
            continue
        m = labels == i
        if m.sum() >= min_area:
            k += 1
            out[m] = k
    return out


_SAM = None
_DETECTOR = None


def _sam():
    """SAM, loaded once per process (a reload costs ~2 s on every scan otherwise)."""
    global _SAM
    if _SAM is None:
        import torch
        from transformers import SamModel, SamProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM = (SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval(), SamProcessor.from_pretrained("facebook/sam-vit-base"), device)
    return _SAM


def segment_sam(frame: Frame, H: np.ndarray, XYZ: np.ndarray, extent: Extent, grid: int, min_iou: float, min_area: int, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """SAM with a point-prompt grid over the table; keeps masks that stand on the table."""
    import torch

    model, proc, device = _sam()
    h, w = frame.depth.shape
    on_table = on_table_mask(XYZ, extent)
    if exclude is not None:
        on_table &= ~exclude
    pts = [[float(u), float(v)] for v in range(grid // 2, h, grid) for u in range(grid // 2, w, grid) if on_table[v, u]]
    rgb = cv2.cvtColor(frame.color, cv2.COLOR_BGR2RGB)
    inputs = proc(rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_embeddings(inputs["pixel_values"])
    cands: List[Tuple[float, np.ndarray]] = []
    batch = 64
    for i in range(0, len(pts), batch):
        chunk = pts[i : i + batch]
        inp = proc(rgb, input_points=[[[p] for p in chunk]], return_tensors="pt").to(device)
        inp.pop("pixel_values")
        with torch.no_grad():
            out = model(image_embeddings=emb, **inp, multimask_output=True)
        masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        scores = out.iou_scores.cpu()[0]
        for j in range(masks.shape[0]):
            best = int(torch.argmax(scores[j]))
            m = masks[j, best].numpy().astype(bool)
            s = float(scores[j, best])
            area = int(m.sum())
            if s < min_iou or area < min_area or area > 0.25 * h * w:
                continue
            if exclude is not None and (m & exclude).sum() > 0.3 * area:
                continue  # the arm
            hm = H[m]
            hm = hm[~np.isnan(hm)]
            if hm.size < 20 or np.median(hm) < 0.008:  # flat on the table -> the table itself
                continue
            cands.append((s, m))
    cands.sort(key=lambda c: -c[0])
    kept: List[np.ndarray] = []
    for s, m in cands:
        if all((m & k).sum() / min(m.sum(), k.sum()) < 0.5 for k in kept):
            kept.append(m)
    out_lab = np.zeros((h, w), np.uint16)
    for i, m in enumerate(kept, start=1):
        out_lab[m & (out_lab == 0)] = i
    return out_lab


def segment_hybrid(frame: Frame, H: np.ndarray, XYZ: np.ndarray, extent: Extent, args: "Args", exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Depth components decide *what stands on the table* (robust, includes the box); SAM
    decides where one component is really several objects: a component is split when at
    least two SAM masks, each a decent share of it, lie inside -- a cup leaning on a cube --
    and left whole otherwise, so a box does not fall apart into walls and flaps."""
    depth_lab = segment_depth(H, XYZ, extent, args.min_height, args.min_area, exclude=exclude)
    sam_lab = segment_sam(frame, H, XYZ, extent, args.sam_grid, args.sam_min_iou, args.min_area, exclude=exclude)
    out = np.zeros_like(depth_lab)
    k = 0
    for i in range(1, int(depth_lab.max()) + 1):
        comp = depth_lab == i
        area = int(comp.sum())
        parts = []
        for j in range(1, int(sam_lab.max()) + 1):
            m = sam_lab == j
            inside = int((m & comp).sum())
            if inside >= 0.6 * m.sum() and inside >= 0.15 * area:
                parts.append(m & comp)
        covered = int(np.any(np.stack(parts), axis=0).sum()) if parts else 0
        # split only when the parts explain most of the component: SAM sees a cup and a cube
        # as two masks covering all of their blob, but a box only as a wall and a flap (the
        # floor is 'table' to it), which must not tear the box apart
        if len(parts) < 2 or covered < 0.7 * area:
            # ... except for something *standing inside* the component: a SAM mask that is a
            # small part of it and clearly taller than the rest (a cup on the box floor, which
            # the floor's depth noise can glue to the walls) is carved out as its own object
            rest = comp.copy()
            carved = []
            for j in range(1, int(sam_lab.max()) + 1):
                m = sam_lab == j
                inside = int((m & comp).sum())
                if inside < args.min_area or inside < 0.8 * m.sum() or inside > 0.2 * area:
                    continue
                # a box wall also has a low ring (table outside, floor inside) but is a long
                # thin strip; an object standing on the floor is compact
                ys, xs = np.nonzero(m & comp)
                if (np.ptp(xs) + 1) / max(1, np.ptp(ys) + 1) > 3.0 or (np.ptp(ys) + 1) / max(1, np.ptp(xs) + 1) > 3.0:
                    continue
                # an island: the mask is tall and the ring just outside it is at floor level
                ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
                hm, hr = H[m & comp], H[ring]
                hm, hr = hm[~np.isnan(hm)], hr[~np.isnan(hr)]
                if hm.size and hr.size > 50 and np.median(hm) > 0.02 and np.median(hr) < args.min_height + 0.008:
                    carved.append(m & comp)
                    rest &= ~m
            for m in carved:
                k += 1
                out[m] = k
            k += 1
            out[rest] = k
            continue
        # assign every component pixel to the nearest part (distance transform per part)
        dist = np.stack([cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, 3) for pm in parts])
        nearest = np.argmin(dist, axis=0)
        for pi in range(len(parts)):
            m = comp & (nearest == pi)
            if m.sum() >= args.min_area:
                k += 1
                out[m] = k
    # SAM masks the depth missed entirely (thin things: a hammer handle is 2 cm tall and gets
    # eaten by the morphology) are added as objects of their own
    for j in range(1, int(sam_lab.max()) + 1):
        full = sam_lab == j
        m = full & (out == 0)  # what the depth did not already claim
        hm = H[m]
        hm = hm[~np.isnan(hm)]
        # a thin object the depth largely missed (a hammer handle 2 cm tall): most of SAM's
        # mask is unclaimed and it does stand above the table
        if m.sum() >= max(args.min_area, 0.5 * full.sum()) and hm.size and np.median(hm) >= 0.008:
            k += 1
            out[m] = k
    return merge_fragments(out, args.merge_area)


def merge_fragments(labels: np.ndarray, merge_area: int, reach: int = 21) -> np.ndarray:
    """Small labels that touch a bigger one are absorbed by it: a box's flaps, the bits a
    dark hammer head or a shiny blade break into, all end up with the object they belong to.
    Repeats until nothing small touches anything bigger."""
    labels = labels.copy()
    kernel = np.ones((reach, reach), np.uint8)
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        area = dict(zip(ids.tolist(), counts.tolist()))
        merged = False
        for i in sorted(area, key=lambda i: area[i]):
            if area[i] >= merge_area:
                break
            ring = cv2.dilate((labels == i).astype(np.uint8), kernel).astype(bool) & (labels != i) & (labels > 0)
            if not ring.any():
                continue
            neigh, n_cnt = np.unique(labels[ring], return_counts=True)
            big = [(n_cnt[t], neigh[t]) for t in range(len(neigh)) if area[int(neigh[t])] > area[i]]
            if not big:
                continue
            target = int(max(big)[1])
            labels[labels == i] = target
            merged = True
            break
        if not merged:
            return labels


def describe(labels: np.ndarray, H: np.ndarray, XYZ: np.ndarray, scene: Scene, min_object_area: int = 1200, min_median_height: float = 0.006, extent: Extent = (-1.0, 9.0, -9.0, 9.0)) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Per-object records, and the label image renumbered 1..N over the objects kept."""
    objs: List[Dict[str, Any]] = []
    from scipy import ndimage

    for i in range(1, int(labels.max()) + 1):
        m = labels == i
        area = int(m.sum())
        if area == 0:
            continue
        vs, us = np.nonzero(m)
        cu, cv_ = float(us.mean()), float(vs.mean())
        # footprint from the hole-filled mask (a box with an object carved out of its floor
        # still has the box's own outline); heights stay from the mask itself
        filled = ndimage.binary_fill_holes(m) & ~np.isnan(XYZ[..., 0])
        pts = XYZ[filled if filled.sum() > 30 else m]
        pts = pts[~np.isnan(pts[:, 0])]
        hh = H[m]
        hh = hh[~np.isnan(hh)]
        entry: Dict[str, Any] = {
            "id": i,
            "area_px": area,
            "center_uv": [round(cu, 1), round(cv_, 1)],
            "bbox": [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())],
        }
        if len(pts) >= 20:
            c = pts.mean(0)
            xy = pts[:, :2] - c[:2]
            evals, evecs = np.linalg.eigh(xy.T @ xy / len(xy))
            ax = evecs[:, int(np.argmax(evals))]
            pl = xy @ ax
            ps = xy @ np.array([-ax[1], ax[0]])
            top_h = float(np.percentile(hh, 98))
            tz = scene.table_z(float(c[0]), float(c[1]))
            entry.update(
                center_xyz=[round(float(v), 4) for v in c],
                top_z=round(tz + top_h, 4),
                table_z=round(tz, 4),
                size=[round(float(np.percentile(pl, 98) - np.percentile(pl, 2)), 4), round(float(np.percentile(ps, 98) - np.percentile(ps, 2)), 4), round(top_h, 4)],
                yaw=round(float(np.arctan2(ax[1], ax[0])), 3),
            )
        if area < min_object_area or (hh.size and float(np.median(hh)) < min_median_height):
            continue
        if "size" in entry:
            long_, short, height = entry["size"]
            # vertical slivers (wall remnants, a box wall caught on its own) and anything taller
            # than a table object can be are not objects to pick
            if (short < 0.012 and height > 0.04) or height > 0.25 or long_ > 0.40:
                continue
            # a wall/curtain patch: its points are not on the table top's far half at all
            if float(np.median(pts[:, 0])) > extent[1] - 0.06:
                continue
        objs.append(entry)
    relabel = np.zeros(labels.shape, np.uint16)
    for k, o in enumerate(objs, start=1):
        relabel[labels == o["id"]] = k
        o["id"] = k
    return relabel, objs


def name_objects(frame: Frame, labels: np.ndarray, objs: List[Dict[str, Any]], vocab: Tuple[str, ...]) -> List[Any]:
    """Attach a name to every mask: GroundingDINO boxes for the vocabulary, each mask takes
    the detection that contains most of it (and is not several times bigger than it)."""
    global _DETECTOR
    if _DETECTOR is None:
        from perception import Detector

        _DETECTOR = Detector()
    dets = _DETECTOR.detect(frame.color, list(vocab), box_threshold=0.25, text_threshold=0.2)
    for o in objs:
        m = labels == o["id"]
        area = int(m.sum())
        x0, y0, x1, y1 = o["bbox"]
        mask_box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        best = None
        for d in dets:
            bx0, by0, bx1, by1 = [int(round(v)) for v in d.box]
            inside = int(m[max(0, by0) : by1 + 1, max(0, bx0) : bx1 + 1].sum())
            contain = inside / area
            det_area = max(1, (bx1 - bx0 + 1) * (by1 - by0 + 1))
            if contain < 0.6 or det_area > 4 * mask_box_area:
                continue
            score = contain * d.score * min(1.0, mask_box_area / det_area)
            if best is None or score > best[0]:
                best = (score, d.label, d.score)
        name = "object"
        if best:
            # GroundingDINO may merge phrases ('lemon orange'); keep the vocabulary entry
            # that shares the most words with what it said
            words = set(best[1].lower().split())
            name = max(vocab, key=lambda v: len(words & set(v.lower().split())))
        o["name"] = name
        o["name_score"] = round(best[2], 2) if best else 0.0
    return dets


CONTAINERS = ("cardboard box", "black box")


def carve_named(frame: Frame, labels: np.ndarray, objs: List[Dict[str, Any]], dets: List[Any], H: np.ndarray, XYZ: np.ndarray, min_height: float, min_area: int) -> Tuple[np.ndarray, bool]:
    """Split an object the detector sees but the geometry merged into a container.

    A cup standing on the box floor is only 6 mm above it, so the height map joins the two;
    GroundingDINO has no such trouble. Any detection that is not a container itself, sits
    mostly inside a container's mask, and stands above that mask's floor is carved out.
    """
    labels = labels.copy()
    by_id = {o["id"]: o for o in objs}
    changed = False
    for d in dets:
        if d.score < 0.45 or any(c.split()[-1] in d.label.lower() for c in CONTAINERS):
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in d.box]
        h, w = labels.shape
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
        region = np.zeros_like(labels, bool)
        region[y0 : y1 + 1, x0 : x1 + 1] = True
        ids, counts = np.unique(labels[region & (labels > 0)], return_counts=True)
        if not len(ids):
            continue
        host = int(ids[int(np.argmax(counts))])
        rec = by_id.get(host)
        if rec is None or rec.get("name") not in CONTAINERS:
            continue
        m = region & (labels == host)
        if m.sum() < min_area or m.sum() > 0.4 * int((labels == host).sum()):
            continue
        # Heights cannot tell a cup on the box floor from the box's own wall (the wall is the
        # taller of the two), so the test is geometric: the carved piece must sit well inside
        # the host's footprint, where a wall never is, and be compact rather than a strip.
        host_pts = XYZ[(labels == host) & ~np.isnan(XYZ[..., 0])]
        c = host_pts[:, :2].mean(0)
        xy = host_pts[:, :2] - c
        evals, evecs = np.linalg.eigh(xy.T @ xy / len(xy))
        ax_long = evecs[:, int(np.argmax(evals))]
        ax_short = np.array([-ax_long[1], ax_long[0]])
        half = np.array(
            [
                (np.percentile(xy @ ax_long, 98) - np.percentile(xy @ ax_long, 2)) / 2,
                (np.percentile(xy @ ax_short, 98) - np.percentile(xy @ ax_short, 2)) / 2,
            ]
        )
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        piece = lab == big
        bw, bh = stats[big, cv2.CC_STAT_WIDTH], stats[big, cv2.CC_STAT_HEIGHT]
        if piece.sum() < min_area or piece.sum() < 0.3 * bw * bh or max(bw / bh, bh / bw) > 3.0:
            continue
        p_pts = XYZ[piece & ~np.isnan(XYZ[..., 0])]
        if len(p_pts) < 30:
            continue
        d_xy = p_pts[:, :2].mean(0) - c
        inside = float(min(half[0] - abs(d_xy @ ax_long), half[1] - abs(d_xy @ ax_short)))
        if inside < 0.04:
            continue
        # The detection box also covers container floor around the object. The floor level is
        # the low mode of the host well inside its own footprint; keep only what stands above
        # it, or the shape fitted to this mask (a cup's rim circle) is pulled off centre.
        hx = XYZ[..., 0] - c[0]
        hy = XYZ[..., 1] - c[1]
        u = np.abs(hx * ax_long[0] + hy * ax_long[1])
        v = np.abs(hx * ax_short[0] + hy * ax_short[1])
        interior = (labels == host) & (u < half[0] - 0.04) & (v < half[1] - 0.04) & ~np.isnan(H)
        if interior.sum() > 200:
            floor = float(np.percentile(H[interior], 20))
            rim = float(rec.get("size", [0, 0, 0.06])[2])  # the container's own top
            # keep what stands clear of the floor but below the container's rim: the floor
            # drags a rim-circle fit off centre, and the wall behind the object -- being the
            # tallest thing in the crop -- would otherwise *be* the "top band"
            band = piece & (H > floor + 0.006) & (H < floor + rim - 0.008)
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
            if n2 > 1:
                b2 = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
                if st2[b2, cv2.CC_STAT_AREA] >= max(min_area, 0.25 * piece.sum()):
                    piece = lab2 == b2
        labels[piece] = int(labels.max()) + 1
        changed = True
        print(f"[scan] carved '{d.label}' ({d.score:.2f}) out of the {rec['name']} mask ({int(piece.sum())} px, {inside * 100:.0f} cm inside its footprint)")
    return labels, changed


def merge_by_name(labels: np.ndarray, objs: List[Dict[str, Any]], gap_px: int = 30, loose_area: int = 4000) -> Tuple[np.ndarray, bool]:
    """Second merging pass, once the masks have names: two masks with the same name that lie
    within ``gap_px`` of each other are one object (a hammer's head and handle), and a small
    unnamed mask ('object', below ``loose_area``) joins the nearest named mask within reach
    (a box flap, a blade edge). Returns (labels, changed)."""
    labels = labels.copy()
    by_id = {o["id"]: o for o in objs}
    changed = False
    for o in sorted(objs, key=lambda o: o["area_px"]):
        i = o["id"]
        m = labels == i
        if not m.any():
            continue
        dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
        best = None
        for j, other in by_id.items():
            if j == i or not (labels == j).any():
                continue
            gap = float(dt[labels == j].min())
            if gap > gap_px:
                continue
            same = other["name"] == o["name"] and o["name"] != "object"
            loose = o["name"] == "object" and other["name"] != "object" and o["area_px"] < loose_area
            if (same or loose) and (best is None or gap < best[0]):
                best = (gap, j)
        if best is not None:
            labels[m] = best[1]
            by_id[best[1]]["area_px"] += o["area_px"]
            changed = True
    return labels, changed


def draw(frame: Frame, labels: np.ndarray, objs: List[Dict[str, Any]]) -> np.ndarray:
    img = frame.color.copy()
    rng = np.random.default_rng(3)
    colors = rng.integers(60, 255, size=(int(labels.max()) + 1, 3))
    overlay = img.copy()
    for i in range(1, int(labels.max()) + 1):
        overlay[labels == i] = colors[i]
    img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    # red dot = mask centroid (mean pixel of the mask); label box at the mask's top-left,
    # nudged down when it would sit on top of a label already drawn
    taken: List[Tuple[int, int, int, int]] = []
    for o in objs:
        u, v = int(o["center_uv"][0]), int(o["center_uv"][1])
        cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        txt = f"{o['id']} {o.get('name', '')}".strip()
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x0, y0 = o["bbox"][0], max(th + 4, o["bbox"][1] - 4)
        x0 = min(x0, img.shape[1] - tw - 4)
        for _ in range(12):
            box = (x0, y0 - th - 4, x0 + tw + 4, y0 + 2)
            if all(box[2] < t[0] or box[0] > t[2] or box[3] < t[1] or box[1] > t[3] for t in taken):
                break
            y0 += th + 8
        taken.append(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), -1)
        cv2.putText(img, txt, (x0 + 2, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (u, v), (x0 + 2, y0), (0, 0, 255), 1, cv2.LINE_AA)
    return img


def scan(frame: Frame, scene: Scene, args: Args, exclude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    t = time.time()
    H, XYZ, extent = height_map(frame, scene)
    if args.backend == "depth":
        labels = merge_fragments(segment_depth(H, XYZ, extent, args.min_height, args.min_area, exclude=exclude), args.merge_area)
    elif args.backend == "sam":
        labels = segment_sam(frame, H, XYZ, extent, args.sam_grid, args.sam_min_iou, args.min_area, exclude=exclude)
    elif args.backend == "hybrid":
        labels = segment_hybrid(frame, H, XYZ, extent, args, exclude=exclude)
    else:
        raise SystemExit(f"unknown backend {args.backend}")
    labels, objs = describe(labels, H, XYZ, scene, args.min_object_area, args.min_median_height, extent)
    print(f"[scan] {args.backend}: {len(objs)} objects in {time.time() - t:.2f}s (table x {extent[0]:.2f}..{extent[1]:.2f}, y {extent[2]:.2f}..{extent[3]:.2f})")
    return labels, objs


def arm_exclusion(frame: Frame, scene: Scene, args: "Args") -> Optional[np.ndarray]:
    """Pixels of the calibrated arm at its current pose (read over RPC, nothing moves)."""
    if not args.mask_arm:
        return None
    try:
        from arm import Arm

        a = Arm(args.calib_side, PORTS[args.calib_side], execute=False)
        try:
            q = a.q()
        finally:
            a.close()
        m = arm_pixels(frame, scene, Kinematics_cache(), q)
        print(f"[scan] masking the {args.calib_side} arm out of the frame ({int(m.sum())} px, q={np.round(q[:6], 2)})")
        return m
    except Exception as e:
        print(f"[scan] could not mask the arm ({e})")
        return None


_KIN = None


def Kinematics_cache():
    global _KIN
    if _KIN is None:
        from arm import Kinematics

        _KIN = Kinematics()
    return _KIN


def main(args: Args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene.from_calib(args.calib_side)
    arms: Dict[str, Tuple[Arm, np.ndarray]] = {}
    rig: Optional[CameraRig] = None
    try:
        if args.from_saved:
            frame = Frame.load(out, "scan_frame", role="top")
        else:
            if args.raise_arms:
                arms = raise_arms(args.sides, args.max_joint_vel)
            rig = CameraRig.open(roles=("top",))
            frame = top_frame(rig, args.frames)
            frame.save(out, "scan_frame")
        exclude = arm_exclusion(frame, scene, args)
        labels, objs = scan(frame, scene, args, exclude=exclude)
        if args.names:
            dets = name_objects(frame, labels, objs, args.vocab)
            H, XYZ, extent = height_map(frame, scene)
            labels, carved = carve_named(frame, labels, objs, dets, H, XYZ, args.min_height, args.min_area)
            if carved:
                labels, objs = describe(labels, H, XYZ, scene, args.min_object_area, args.min_median_height, extent)
                name_objects(frame, labels, objs, args.vocab)
            for _ in range(3):  # name-aware merging, then re-measure and re-name
                labels, changed = merge_by_name(labels, objs)
                if not changed:
                    break
                H, XYZ, extent = height_map(frame, scene)
                labels, objs = describe(labels, H, XYZ, scene, args.min_object_area, args.min_median_height, extent)
                name_objects(frame, labels, objs, args.vocab)
        for o in objs:
            extra = f"  xyz ({o['center_xyz'][0]:+.3f}, {o['center_xyz'][1]:+.3f}, {o['center_xyz'][2]:+.3f})  size {np.round(np.array(o['size']) * 100, 1)} cm" if "size" in o else ""
            print(f"  #{o['id']:2d} {o.get('name', ''):16s} uv ({o['center_uv'][0]:6.1f}, {o['center_uv'][1]:6.1f})  {o['area_px']:6d} px{extra}")
        cv2.imwrite(str(out / "scan_overlay.png"), draw(frame, labels, objs))
        cv2.imwrite(str(out / "scan_masks.png"), labels)
        (out / "scan_result.json").write_text(json.dumps({"backend": args.backend, "objects": objs, "time": time.time()}, indent=1))
        print(f"[scan] wrote {out / 'scan_overlay.png'}, scan_masks.png, scan_result.json")
    finally:
        if rig is not None:
            rig.close()
        for side, (arm, q0) in arms.items():
            try:
                if args.return_home:
                    res = arm.move_joints(q0, settle=0.0)
                    print(f"[{side}] back: {'ok' if not res.aborted else res.aborted}")
            finally:
                arm.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
