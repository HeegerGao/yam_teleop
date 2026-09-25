"""Scene understanding and grasp/place planning for language_pick_place.py.

Everything here is geometry on one top-camera frame; nothing moves. Given the name of one of
the known objects it answers three questions -- where is it, how should the fingers take it,
and where in the box does it go -- in the base frame of whichever arm is asked, through that
arm's click calibration (click_model.py: pixel + height -> base xy).

    view = TableView(frame, models={"left": ClickModel.load("left"), ...})
    box = view.find_box(detector)                    # rim rectangle, floor, what is in it
    obj = view.find_object("hammer", detector, sam)  # mask + heights, side-checked
    g = plan_grasp(view, obj, "left")                # fingertip centre, finger axis, opening
    p = plan_place(view, box, g, "left", ...)        # a free spot the held object fits in

Why the mix of sensors: the top D435's depth is fine on the plush toys, cubes and the box,
and useless on the tools -- the hammer's handle reads *below* the table, the hacksaw's frame
is not there at all. So the footprint comes from colour (a SAM mask prompted with the
detector's box, or the handle's own colour), and the heights from depth where depth is
plausible and from a per-object prior where it is not. The objects are a fixed set, which is
what makes the priors honest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from cameras import Frame
from click_model import ClickModel
from perception import Detection, Detector

OPEN_STROKE = 0.096
"""Fingertip separation at gripper command 1.0, metres (as click_pick.py)."""


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    """What is known about one of the objects before looking."""

    prompts: Tuple[str, ...]
    """GroundingDINO phrases, tried one at a time (tiny-DINO is far better with one phrase)."""
    side: str
    """Which side of the box it always lies on: left | right (as seen in the top image)."""
    grasp: str = "center"
    """center: top-down at the centre of the part the fingers reach, across its narrow side.
    handle: hold the coloured handle (tools), fingers across it.
    deep:   like center but low and squeezing (plush toys)."""
    below_top: float = 0.02
    """Fingertips this far below the top of the grasped part."""
    height: float = 0.05
    """Prior height, used when depth does not see the object (and as a sanity floor)."""
    color: Optional[str] = None
    """Colour that must be visible in a candidate detection (rejects DINO's wrong boxes)."""
    handle_color: Optional[str] = None
    """The handle's colour: the grasp region is the largest blob of it inside the detection."""
    handle_height: float = 0.025
    """Handle diameter -- the fingertips close this far above the table."""
    along: float = 0.5
    """handle grasps: where along the handle, as a fraction from its free end."""
    min_width: float = 0.02
    trust_depth: bool = True
    """Whether the depth image is believed on this object at all."""
    extent: Optional[Tuple[float, float, float]] = None
    """handle grasps: the whole tool relative to the grasp point, metres -- (behind the
    fingers towards the handle's free end, ahead towards the head, half width). The tools
    lie tangled together and their grey parts vanish in depth, so their outline for the
    placement is this prior, not a measurement."""


VOCAB: Dict[str, Spec] = {
    "red_cube": Spec(("red cube", "red block"), "left", color="red", height=0.045),
    "blue_cube": Spec(("blue cube", "blue block"), "left", color="blue", height=0.045),
    "lemon": Spec(("lemon", "yellow lemon"), "left", color="yellow", height=0.055),
    "white_cup": Spec(("white cup", "cup", "mug"), "left", below_top=0.025, height=0.065),
    "bread": Spec(("bread", "loaf of bread"), "left", below_top=0.03, height=0.06),
    "bear": Spec(("brown bear", "teddy bear", "stuffed animal"), "left", grasp="deep", below_top=0.035, height=0.06),
    "pink": Spec(
        ("pink rabbit", "pink toy", "pink stuffed animal"),
        "left",
        grasp="deep",
        below_top=0.035,
        height=0.06,
        color="pink",
    ),
    "elephant": Spec(
        ("grey elephant", "stuffed elephant", "elephant toy", "elephant"),
        "right",
        grasp="deep",
        below_top=0.035,
        height=0.07,
    ),
    "hammer": Spec(
        ("yellow hammer", "hammer", "claw hammer"),
        "right",
        grasp="handle",
        handle_color="yellow",
        handle_height=0.025,
        along=0.45,
        height=0.03,
        trust_depth=False,
        extent=(0.11, 0.17, 0.06),
    ),
    "shovel": Spec(
        ("green handle", "trowel", "garden trowel", "shovel"),
        "right",
        grasp="handle",
        handle_color="green",
        handle_height=0.03,
        along=0.5,
        height=0.035,
        trust_depth=False,
        extent=(0.07, 0.20, 0.045),
    ),
    "saw": Spec(
        ("red handle", "hacksaw", "hand saw", "saw"),
        "right",
        grasp="handle",
        handle_color="red",
        handle_height=0.028,
        along=0.5,
        height=0.03,
        trust_depth=False,
        extent=(0.07, 0.27, 0.065),
    ),
}

ALIASES: Dict[str, str] = {
    "cup": "white_cup",
    "mug": "white_cup",
    "white cup": "white_cup",
    "rabbit": "pink",
    "bunny": "pink",
    "pink rabbit": "pink",
    "pink bunny": "pink",
    "teddy": "bear",
    "teddy bear": "bear",
    "brown bear": "bear",
    "trowel": "shovel",
    "spade": "shovel",
    "hacksaw": "saw",
    "red cube": "red_cube",
    "red block": "red_cube",
    "blue cube": "blue_cube",
    "blue block": "blue_cube",
    "loaf": "bread",
}


def parse_instruction(text: str) -> List[str]:
    """The vocabulary entries named in a sentence, in the order they appear.

    'put the lemon and the red cube in the box' -> ['lemon', 'red_cube']. Longest names are
    matched first so 'red cube' is not read as 'cube', and every hit is blanked out so it is
    not counted twice."""
    s = " " + " ".join(text.lower().replace("_", " ").replace(",", " ").split()) + " "
    names = {k.replace("_", " "): k for k in VOCAB}
    names.update(ALIASES)
    hits: List[Tuple[int, str]] = []
    for phrase, name in sorted(names.items(), key=lambda kv: -len(kv[0])):
        p = " " + phrase.replace("_", " ") + " "
        while True:
            i = s.find(p)
            if i < 0:
                break
            hits.append((i, name))
            s = s[:i] + " " * len(p) + s[i + len(p) :]
    hits.sort()
    out: List[str] = []
    for _, n in hits:
        if n not in out:
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

_HUE = {
    "red": ((0, 10), (170, 180)),
    "yellow": ((18, 38),),
    "green": ((35, 85),),
    "blue": ((95, 130),),
    "pink": ((140, 175),),
}


def color_mask(bgr: np.ndarray, name: str) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if name == "white":
        return (s < 60) & (v > 150)
    if name == "pink":  # pale: low saturation, but a definite hue
        m = np.zeros(h.shape, bool)
        for lo, hi in _HUE["pink"]:
            m |= (h >= lo) & (h <= hi)
        return m & (s > 35) & (v > 120)
    m = np.zeros(h.shape, bool)
    for lo, hi in _HUE[name]:
        m |= (h >= lo) & (h <= hi)
    return m & (s > 80) & (v > 60)


def non_table_mask(bgr: np.ndarray, sat: int = 60, darker: int = 45) -> np.ndarray:
    """Pixels that are not the white table: anything coloured, or clearly darker than the
    table *around it* -- the table is lit unevenly, so a fixed brightness threshold turned
    half of the left side into 'object'. The local table brightness is a coarse median."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    small = cv2.resize(v, (v.shape[1] // 8, v.shape[0] // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 21), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = (hsv[..., 1] > sat) | (v.astype(np.int16) < bg.astype(np.int16) - darker)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def largest_component(mask: np.ndarray, min_area: int = 1) -> Optional[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None
    return lab == i


def box_fraction(mask: np.ndarray, box: np.ndarray) -> float:
    """Fraction of a detection box covered by the mask."""
    x0, y0, x1, y1 = [round(v) for v in box]
    sub = mask[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
    return float(sub.mean()) if sub.size else 0.0


# ---------------------------------------------------------------------------
# SAM, prompted with a box (and a point), for the outline depth cannot give
# ---------------------------------------------------------------------------


class Segmenter:
    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str = "cuda") -> None:
        import torch
        from transformers import SamModel, SamProcessor

        t = time.time()
        self.torch = torch
        self.device = device
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.proc = SamProcessor.from_pretrained(model_id)
        self.emb = None
        self.emb_id: Optional[int] = None
        print(f"[sam] {model_id} on {device} in {time.time() - t:.1f}s")

    def _embed(self, bgr: np.ndarray) -> None:
        if self.emb_id == id(bgr) and self.emb is not None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            self.emb = self.model.get_image_embeddings(inputs["pixel_values"])
        self.emb_id = id(bgr)
        self.rgb = rgb

    def masks(
        self, bgr: np.ndarray, box: Optional[np.ndarray] = None, point: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """All three SAM masks for a box and/or a foreground point, (iou score, HxW bool),
        best first. The three are SAM's part / sub-part / whole readings of the prompt; a
        tool's handle is a 'part' and the whole tool is what a placement needs."""
        self._embed(bgr)
        kw = {}
        if box is not None:
            kw["input_boxes"] = [[[float(v) for v in box]]]
        if point is not None:
            kw["input_points"] = [[[[float(point[0]), float(point[1])]]]]
        inp = self.proc(self.rgb, return_tensors="pt", **kw).to(self.device)
        inp.pop("pixel_values")
        with self.torch.no_grad():
            out = self.model(image_embeddings=self.emb, **inp, multimask_output=True)
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
        )[0]
        scores = out.iou_scores.cpu()[0, 0]
        res = [(float(scores[k]), masks[0, k].numpy().astype(bool)) for k in range(masks.shape[1])]
        res.sort(key=lambda r: -r[0])
        return res

    def mask(
        self, bgr: np.ndarray, box: Optional[np.ndarray] = None, point: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Best SAM mask for a box and/or a foreground point, HxW bool."""
        self._embed(bgr)
        kw = {}
        if box is not None:
            kw["input_boxes"] = [[[float(v) for v in box]]]
        if point is not None:
            kw["input_points"] = [[[[float(point[0]), float(point[1])]]]]
        inp = self.proc(self.rgb, return_tensors="pt", **kw).to(self.device)
        inp.pop("pixel_values")
        with self.torch.no_grad():
            out = self.model(image_embeddings=self.emb, **inp, multimask_output=True)
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
        )[0]
        scores = out.iou_scores.cpu()[0, 0]
        best = int(self.torch.argmax(scores))
        return masks[0, best].numpy().astype(bool)


# ---------------------------------------------------------------------------
# the frame, its heights, and pixel -> base
# ---------------------------------------------------------------------------


def xy_at_many(model: ClickModel, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """ClickModel.xy_at for N pixels at N heights."""
    P = np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))]).astype(np.float64)
    a = (model.H0 @ P.T).T
    b = (model.H1 @ P.T).T
    a = a[:, :2] / a[:, 2:3]
    b = b[:, :2] / b[:, 2:3]
    t = ((np.asarray(z, dtype=np.float64) - model.z0) / (model.z1 - model.z0))[:, None]
    return a + (b - a) * t


def fit_plane(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    cen = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - cen, full_matrices=False)
    n = vt[2]
    return n, float(-n @ cen)


def _hull(xy: np.ndarray) -> np.ndarray:
    return cv2.convexHull(np.asarray(xy, dtype=np.float32)).reshape(-1, 2)


def _inside(xy: np.ndarray, hull: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as _Path

    return _Path(hull).contains_points(xy)


@dataclass
class Found:
    """One located object: its pixels, and a height for each of them."""

    name: str
    score: float
    box: np.ndarray  # detection xyxy
    mask: np.ndarray  # HxW bool, the whole object (placement footprint)
    grasp_mask: np.ndarray  # HxW bool, the part to hold (== mask unless it is a handle)
    h: np.ndarray  # HxW metres above the table under it; nan where unknown
    top_h: float  # height of the part to hold
    depth_seen: bool
    note: str = ""

    def pixels(self, mask: Optional[np.ndarray] = None, stride: int = 1) -> np.ndarray:
        m = self.mask if mask is None else mask
        v, u = np.nonzero(m)
        return np.column_stack([u, v])[::stride]


class TableView:
    """One top frame: the table plane, a height map, and the per-arm pixel->base mapping."""

    def __init__(
        self, frame: Frame, models: Dict[str, ClickModel], plane: Optional[Tuple[np.ndarray, float]] = None
    ) -> None:
        self.frame = frame
        self.models = models
        if plane is None:
            from click_pick import table_plane_cam

            try:
                from perception import Scene

                prior = Scene.from_calib("left").table_plane_cam()
            except Exception:
                prior = None
            plane = table_plane_cam(frame, prior)
        n, d = plane
        if d < 0:  # the camera must be on the positive side: heights above the table positive
            n, d = -n, -d
        self.plane = (np.asarray(n, dtype=np.float64), float(d))
        pts, uv = frame.point_cloud(frame.depth > 0.2)
        self.pts_cam = pts
        self.uv = uv
        s = pts @ self.plane[0] + self.plane[1]
        H = np.full(frame.depth.shape, np.nan, np.float32)
        H[uv[:, 1], uv[:, 0]] = s
        self.nontable = non_table_mask(frame.color)
        # One plane does not fit this depth image: the table reads a centimetre or more high
        # in places, which would turn bare table into 'obstacle'. The bias is measured per
        # block from pixels that look like table (near the plane, not coloured), smoothed,
        # and taken out; the local refits around each object do the same thing more finely.
        blk = 64
        h, w = H.shape
        nb_v, nb_u = (h + blk - 1) // blk, (w + blk - 1) // blk
        bias = np.full((nb_v, nb_u), np.nan, np.float32)
        tablelike = ~np.isnan(H) & (np.abs(H) < 0.03) & ~self.nontable
        for i in range(nb_v):
            for j in range(nb_u):
                sl = (slice(i * blk, (i + 1) * blk), slice(j * blk, (j + 1) * blk))
                vals = H[sl][tablelike[sl]]
                if vals.size > 200:
                    bias[i, j] = np.median(vals)
        if np.isfinite(bias).any():
            fill = float(np.nanmedian(bias))
            bias = np.where(np.isnan(bias), fill, bias).astype(np.float32)
            bias = cv2.GaussianBlur(bias, (3, 3), 0)
            bias_full = cv2.resize(bias, (w, h), interpolation=cv2.INTER_LINEAR)
            H = H - bias_full
        self.H = H
        # the table itself, for the local refits: near the plane, seen by the depth camera
        self.table = np.zeros(frame.depth.shape, bool)
        self.table[uv[:, 1], uv[:, 0]] = np.abs(H[uv[:, 1], uv[:, 0]]) < 0.02

    # ---- heights -----------------------------------------------------------------------
    def local_heights(self, box: np.ndarray, ring: Tuple[int, int] = (30, 110)) -> np.ndarray:
        """Heights inside a box, measured against the table right around it.

        The depth camera reads the far corners of the table a centimetre or two high; a
        plane fitted to the whole image would put that straight into the grasp height. The
        table pixels in a ring around the box are refitted and heights taken from that."""
        h, w = self.H.shape
        x0, y0, x1, y1 = [round(v) for v in box]
        X0, Y0 = max(0, x0 - ring[1]), max(0, y0 - ring[1])
        X1, Y1 = min(w - 1, x1 + ring[1]), min(h - 1, y1 + ring[1])
        sel = np.zeros((h, w), bool)
        sel[Y0 : Y1 + 1, X0 : X1 + 1] = True
        inner = np.zeros((h, w), bool)
        inner[max(0, y0 - ring[0]) : min(h, y1 + ring[0] + 1), max(0, x0 - ring[0]) : min(w, x1 + ring[0] + 1)] = True
        sel &= ~inner & self.table & ~self.nontable
        out = np.full((h, w), np.nan, np.float32)
        idx = sel[self.uv[:, 1], self.uv[:, 0]]
        n, d = self.plane
        if idx.sum() > 300:
            ring_pts = self.pts_cam[idx][:: max(1, idx.sum() // 6000)]
            n_loc, d_loc = fit_plane(ring_pts)
            if n_loc @ n < 0:
                n_loc, d_loc = -n_loc, -d_loc
            if float(np.degrees(np.arccos(np.clip(n_loc @ n, -1, 1)))) < 6.0:
                n, d = n_loc, d_loc
        inbox = np.zeros((h, w), bool)
        inbox[max(0, y0) : min(h, y1 + 1), max(0, x0) : min(w, x1 + 1)] = True
        idx = inbox[self.uv[:, 1], self.uv[:, 0]]
        s = self.pts_cam[idx] @ n + d
        uv = self.uv[idx]
        out[uv[:, 1], uv[:, 0]] = s
        return out

    # ---- pixel -> base -------------------------------------------------------------------
    def to_base(self, side: str, uv: np.ndarray, h: np.ndarray) -> np.ndarray:
        """(N,3) base-frame points of pixels at heights ``h`` above the table."""
        m = self.models[side]
        z = m.table_z + np.asarray(h, dtype=np.float64)
        xy = xy_at_many(m, np.asarray(uv, dtype=np.float64), z)
        return np.column_stack([xy, z])

    def to_pixel(self, side: str, p: np.ndarray) -> Tuple[int, int]:
        """Base point -> pixel, by inverting the click model's homography at that height."""
        m = self.models[side]
        z = float(p[2])
        # xy(uv) = a + (b - a) t is not itself a homography, so solve it with a few Newton steps
        uv = np.array([self.frame.color.shape[1] / 2.0, self.frame.color.shape[0] / 2.0])
        for _ in range(12):
            f = m.xy_at(uv, z) - np.asarray(p[:2])
            J = np.zeros((2, 2))
            for k in range(2):
                d = np.zeros(2)
                d[k] = 1.0
                J[:, k] = (m.xy_at(uv + d, z) - m.xy_at(uv - d, z)) / 2.0
            try:
                uv = uv - np.linalg.solve(J, f)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(f) < 1e-4:
                break
        return round(uv[0]), round(uv[1])

    def footprint(self, side: str, uv: np.ndarray, h_lo: float, h_hi: float, step: float = 0.004) -> np.ndarray:
        """Base-frame xy of what a set of pixels stands on, when their heights are only known
        to lie between ``h_lo`` and ``h_hi``.

        The depth camera's height on a small object is good to about a centimetre and has
        a slope across a flat top, and every centimetre of height error moves a pixel a
        centimetre along its ray. So the heights are not used at all: the pixels mapped at
        the lower bound over-estimate the footprint away from the camera, mapped at the upper
        bound they over-estimate it towards the camera, and the true footprint is (near
        enough) where the two agree. Returned as points on a grid inside that intersection."""
        uv = np.asarray(uv, dtype=np.float64)
        lo = xy_at_many(self.models[side], uv, np.full(len(uv), self.models[side].table_z + h_lo))
        if h_hi - h_lo < 0.003:
            return lo
        hi = xy_at_many(self.models[side], uv, np.full(len(uv), self.models[side].table_z + h_hi))
        hull_lo, hull_hi = _hull(lo), _hull(hi)
        mid = xy_at_many(self.models[side], uv, np.full(len(uv), self.models[side].table_z + 0.5 * (h_lo + h_hi)))
        mn = np.maximum(lo.min(0), hi.min(0))
        mx = np.minimum(lo.max(0), hi.max(0))
        if np.any(mx <= mn):
            return mid
        gx, gy = np.meshgrid(np.arange(mn[0], mx[0] + 1e-9, step), np.arange(mn[1], mx[1] + 1e-9, step), indexing="ij")
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        keep = _inside(grid, hull_lo) & _inside(grid, hull_hi)
        # a thin thing (a handle) narrower than the height band's shift has no intersection
        # worth the name; its mid-height mapping is the honest answer then
        area = keep.sum() * step * step
        if keep.sum() < 4 or area < 0.4 * min(cv2.contourArea(hull_lo), cv2.contourArea(hull_hi)):
            return mid
        return grid[keep]

    def obstacle_points(self, side: str, exclude: Optional[np.ndarray] = None, stride: int = 3) -> np.ndarray:
        """Everything standing on the table, as base-frame points: seen by depth (above 1 cm)
        or coloured against the white table (the tools), heights from depth where known."""
        m = (np.nan_to_num(self.H, nan=0.0) > 0.012) | self.nontable
        if exclude is not None:
            m &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        v, u = np.nonzero(m)
        uv = np.column_stack([u, v])[::stride]
        h = self.H[uv[:, 1], uv[:, 0]]
        h = np.where(np.isnan(h) | (h < 0.0), 0.015, np.minimum(h, 0.3))
        return self.to_base(side, uv, h)

    # ---- objects ---------------------------------------------------------------------
    def detect(self, detector: Detector, prompts: Sequence[str], threshold: float = 0.2) -> List[Detection]:
        out: List[Detection] = []
        for p in prompts:
            for d in detector.detect(self.frame.color, [p], box_threshold=threshold, text_threshold=threshold):
                if d.score >= threshold:
                    d.label = p
                    out.append(d)
        out.sort(key=lambda d: -d.score)
        return out

    def find_object(
        self,
        name: str,
        detector: Detector,
        sam: Optional[Segmenter],
        box_center_u: Optional[float] = None,
        min_score: float = 0.2,
        stride: int = 2,
    ) -> Found:
        """Locate one vocabulary object: DINO boxes for its phrases, filtered by the side of
        the box it must be on and by its colour, then a mask for the best one."""
        spec = VOCAB[name]
        H, W = self.H.shape
        dets = self.detect(detector, spec.prompts, threshold=min_score)
        cands: List[Tuple[float, Detection, str]] = []
        for d in dets:
            x0, y0, x1, y1 = d.box
            area = (x1 - x0) * (y1 - y0)
            if area > 0.25 * H * W or area < 400:
                continue
            cu = 0.5 * (x0 + x1)
            if box_center_u is not None and ((spec.side == "left") != (cu < box_center_u)):
                continue
            score = d.score
            colour = spec.color or spec.handle_color
            if colour is not None:
                frac = box_fraction(color_mask(self.frame.color, colour), d.box)
                if frac < 0.04:
                    continue
                score *= 0.5 + min(frac, 0.5)
            # a box that is mostly table is a box around several things -- the detector's
            # 'hammer' that spans the whole pile of tools
            filled = box_fraction(self.nontable | (np.nan_to_num(self.H, nan=0.0) > 0.01), d.box)
            if filled < 0.12:
                continue
            cands.append((score * (0.6 + min(filled, 0.4)), d, d.label))
        if not cands:
            raise LookupError(f"{name}: nothing found for {list(spec.prompts)} on the {spec.side} of the box")
        cands.sort(key=lambda c: -c[0])
        score, det, phrase = cands[0]
        x0, y0, x1, y1 = [round(v) for v in det.box]
        inbox = np.zeros((H, W), bool)
        inbox[max(0, y0 - 6) : min(H, y1 + 7), max(0, x0 - 6) : min(W, x1 + 7)] = True
        note = f"'{phrase}' {det.score:.2f}"
        grasp_mask: Optional[np.ndarray] = None
        if spec.handle_color is not None:
            grasp_mask = largest_component(color_mask(self.frame.color, spec.handle_color) & inbox, min_area=150)
            if grasp_mask is None:
                raise LookupError(f"{name}: no {spec.handle_color} handle inside the '{phrase}' detection")
            grasp_mask = cv2.morphologyEx(
                grasp_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
            ).astype(bool)
        hloc = self.local_heights(det.box)
        depth_blob = largest_component((np.nan_to_num(hloc, nan=0.0) > 0.01) & inbox, min_area=150)
        # the whole outline: SAM around the detection (seeded on the handle / the depth blob),
        # checked against the box it was asked about; otherwise whatever is coloured or
        # stands up inside that box
        mask: Optional[np.ndarray] = None
        seed_mask = grasp_mask if grasp_mask is not None else depth_blob
        point = None
        if seed_mask is not None:
            v, u = np.nonzero(seed_mask)
            point = (float(np.median(u)), float(np.median(v)))
        if spec.extent is not None:
            # a tool: the handle is all that is measured; the rest of it is the prior extent
            mask = grasp_mask.copy()
            note += ", outline from prior"
        elif sam is not None:
            try:
                good = []
                for sc, m in sam.masks(self.frame.color, box=det.box, point=point):
                    m = largest_component(m & inbox, min_area=100)
                    if m is None:
                        continue
                    inside = float((m & inbox).sum() / max(1, m.sum()))
                    on_seed = (
                        point is None
                        or m[int(point[1]), int(point[0])]
                        or (m & seed_mask).sum() > 0.5 * seed_mask.sum()
                    )
                    if inside > 0.9 and m.sum() < 1.6 * (x1 - x0) * (y1 - y0) and on_seed:
                        good.append((sc, m))
                if good:
                    mask = good[0][1]
                    note += ", sam"
            except Exception as e:
                print(f"[sam] failed on {name}: {type(e).__name__}: {e}")
        if mask is None:
            fallback = (self.nontable | (np.nan_to_num(hloc, nan=0.0) > 0.01)) & inbox
            fallback = cv2.morphologyEx(fallback.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(
                bool
            )
            if seed_mask is not None:
                _, lab = cv2.connectedComponents(fallback.astype(np.uint8), connectivity=8)
                ids = np.unique(lab[seed_mask])
                ids = ids[ids > 0]
                mask = np.isin(lab, ids) if len(ids) else seed_mask.copy()
            else:
                mask = largest_component(fallback, min_area=100)
            if mask is None:
                raise LookupError(f"{name}: the '{phrase}' detection holds nothing that stands out from the table")
            note += ", colour/depth outline"
        if grasp_mask is not None:
            mask |= grasp_mask
        else:
            grasp_mask = mask
        # heights: depth where it is plausible, the prior where it is not
        h = np.full((H, W), np.nan, np.float32)
        hv = hloc[mask]
        seen = ~np.isnan(hv) & (hv > 0.008) & (hv < 0.3)
        depth_ok = spec.trust_depth and seen.sum() > 0.3 * mask.sum() and seen.sum() > 100
        if depth_ok:
            h[mask] = np.where(seen, np.clip(hv, 0.0, 0.3), np.nan)
            top = (
                float(np.nanpercentile(h[grasp_mask], 95))
                if np.isfinite(h[grasp_mask]).sum() > 30
                else float(np.nanpercentile(h[mask], 95))
            )
            if top < 0.5 * spec.height:
                # the plush toys read low: the top is real, but do not trust it for the
                # descent -- the fingers would go through where depth says the toy ends
                note += f", depth top {top * 100:.1f} cm < prior {spec.height * 100:.0f}"
                top = max(top, 0.6 * spec.height)
            fill = float(np.nanmedian(h[mask]))
            h[mask & np.isnan(h)] = fill
        else:
            top = spec.handle_height if spec.handle_color else spec.height
            h[mask] = spec.height / 2.0
            if spec.handle_color:
                h[grasp_mask] = spec.handle_height / 2.0
            note += ", height from prior"
        return Found(
            name=name,
            score=float(score),
            box=det.box.copy(),
            mask=mask,
            grasp_mask=grasp_mask,
            h=h,
            top_h=float(top),
            depth_seen=bool(depth_ok),
            note=note,
        )

    # ---- the box -----------------------------------------------------------------------
    def find_box(
        self,
        detector: Detector,
        prompts: Sequence[str] = ("cardboard box", "box"),
        rim_range: Tuple[float, float] = (0.025, 0.09),
    ) -> "BoxView":
        dets = self.detect(detector, prompts, threshold=0.3)
        H, W = self.H.shape
        dets = [d for d in dets if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) > 0.02 * H * W]
        if not dets:
            raise LookupError("no cardboard box in view")
        det = dets[0]
        x0, y0, x1, y1 = [round(v) for v in det.box]
        hloc = self.local_heights(det.box, ring=(20, 90))
        inbox = np.zeros((H, W), bool)
        inbox[max(0, y0) : min(H, y1 + 1), max(0, x0) : min(W, x1 + 1)] = True
        wall = inbox & (hloc > rim_range[0]) & (hloc < rim_range[1])
        if wall.sum() < 300:
            raise LookupError("the box's rim is not visible in depth")
        # the rim itself: the top of the walls. The inner faces read badly in depth (a
        # vertical face at a grazing angle smears into the floor behind it), so only the
        # top band is used, at one height, for the rectangle
        rim_h = float(np.nanpercentile(hloc[wall], 90))
        rim = wall & (hloc > rim_h - 0.012)
        return BoxView(self, det, hloc, rim, inbox, rim_h)


class BoxView:
    """The box in the image; ``frame(side)`` gives its geometry in one arm's base frame."""

    def __init__(
        self, view: TableView, det: Detection, hloc: np.ndarray, rim: np.ndarray, inbox: np.ndarray, rim_h: float
    ) -> None:
        self.view = view
        self.det = det
        self.hloc = hloc
        self.rim = rim
        self.inbox = inbox
        self.center_u = 0.5 * (det.box[0] + det.box[2])
        self.rim_h = rim_h
        self._frames: Dict[str, "BoxFrame"] = {}

    def frame(self, side: str, wall_inset: float = 0.012) -> "BoxFrame":
        if side not in self._frames:
            self._frames[side] = BoxFrame(self, side, wall_inset)
        return self._frames[side]


@dataclass
class BoxFrame:
    """The box's interior as a rectangle in one arm's base frame, and what already sits in it."""

    box: BoxView
    side: str
    wall_inset: float
    center: np.ndarray = field(init=False)  # base xy
    u: np.ndarray = field(init=False)  # unit, along the long side
    v: np.ndarray = field(init=False)  # unit, along the short side
    half: np.ndarray = field(init=False)  # interior half sizes (along u, along v)
    floor_z: float = field(init=False)
    rim_z: float = field(init=False)
    contents: np.ndarray = field(init=False)  # (N,3) base points of whatever is inside
    grid: np.ndarray = field(init=False)  # occupancy at ``cell`` resolution, (nu, nv)
    heights: np.ndarray = field(init=False)  # top height per cell, metres above the floor
    cell: float = 0.005

    def __post_init__(self) -> None:
        view, side = self.box.view, self.side
        model = view.models[side]
        v, u = np.nonzero(self.box.rim)
        uv = np.column_stack([u, v])
        pts = view.to_base(side, uv, np.full(len(uv), self.box.rim_h))
        # the rim's outer rectangle; a 1 cm occupancy grid with a minimum count throws the
        # stray points out before the rectangle is fitted, which minAreaRect cannot do itself
        xy = pts[:, :2]
        cells = np.floor(xy / 0.01).astype(int)
        _, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        keep = counts[inv] >= 3
        xy = xy[keep] if keep.sum() > 100 else xy
        (cx, cy), (w, hgt), ang = cv2.minAreaRect(xy.astype(np.float32))
        a = np.radians(ang)
        uvec, vvec = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        if w < hgt:
            uvec, vvec, w, hgt = vvec, -uvec, hgt, w
        self.center = np.array([cx, cy])
        self.u, self.v = uvec, vvec
        self.half = np.array([w / 2 - self.wall_inset, hgt / 2 - self.wall_inset])
        self.rim_z = model.table_z + self.box.rim_h
        self.floor_z = model.table_z + 0.003
        # What is in it: only what lies *on the floor in the image*. The floor polygon is
        # the interior rectangle projected at floor height; the walls' inner faces fall
        # outside it (the far wall's face is above the far floor edge in the image), which
        # is what keeps their smeared depth from reading as contents.
        H, W = view.H.shape
        floor_mask = np.ones((H, W), np.uint8)
        # ... intersected with the same rectangle at rim height: the near wall hides the near
        # part of the floor, and the wall's own top would otherwise count as contents
        for z in (self.floor_z, self.rim_z):
            px = np.array([view.to_pixel(side, np.array([c[0], c[1], z])) for c in self.corners()], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [px], 1)
            floor_mask &= m
        floor_mask = cv2.erode(floor_mask, np.ones((21, 21), np.uint8)).astype(bool) & self.box.inbox
        self.floor_mask = floor_mask
        vv, uu = np.nonzero(floor_mask)
        uv_f = np.column_stack([uu, vv])[::2]
        h_f = self.box.hloc[uv_f[:, 1], uv_f[:, 0]]
        seen = ~np.isnan(h_f)
        if seen.sum() > 50:
            floor_h = float(np.percentile(h_f[seen], 30))
            self.floor_z = model.table_z + float(np.clip(floor_h, 0.0, 0.015))
        tall = seen & (h_f > (self.floor_z - model.table_z) + 0.012)
        cont = view.to_base(side, uv_f[tall], np.clip(h_f[tall], 0.0, 0.3)) if tall.sum() else np.zeros((0, 3))
        # coloured pixels depth does not see (a grey saw frame): put them at a nominal height.
        # cardboard is itself 'non-table', so only what is not cardboard-coloured counts
        hsv = cv2.cvtColor(view.frame.color, cv2.COLOR_BGR2HSV)
        card = (hsv[..., 0] >= 5) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 40) & (hsv[..., 1] < 210)
        nt = view.nontable & floor_mask & ~card
        nt = cv2.morphologyEx(nt.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
        vv, uu = np.nonzero(nt)
        uv_nt = np.column_stack([uu, vv])[::2]
        if len(uv_nt):
            p_nt = view.to_base(side, uv_nt, np.full(len(uv_nt), 0.02))
            cont = np.vstack([cont, p_nt]) if len(cont) else p_nt
        if len(cont):
            # the floor right along a wall reads a few centimetres high (the wall's depth
            # bleeds into it), so a band next to the walls is not believed
            l = self.local(cont[:, :2])
            cont = cont[(np.abs(l[:, 0]) < self.half[0] - 0.02) & (np.abs(l[:, 1]) < self.half[1] - 0.02)]
        self.contents = cont if len(cont) else np.zeros((0, 3))
        nu, nv = int(np.ceil(2 * self.half[0] / self.cell)), int(np.ceil(2 * self.half[1] / self.cell))
        self.grid = np.zeros((nu, nv), bool)
        self.heights = np.zeros((nu, nv), np.float32)
        if len(self.contents):
            l = self.local(self.contents[:, :2])
            i = np.clip(((l[:, 0] + self.half[0]) / self.cell).astype(int), 0, nu - 1)
            j = np.clip(((l[:, 1] + self.half[1]) / self.cell).astype(int), 0, nv - 1)
            self.grid[i, j] = True
            np.maximum.at(self.heights, (i, j), (self.contents[:, 2] - self.floor_z).astype(np.float32))
            self.grid = cv2.morphologyEx(
                self.grid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            ).astype(bool)

    def local(self, xy: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(xy) - self.center
        return np.column_stack([d @ self.u, d @ self.v])

    def world(self, loc: np.ndarray) -> np.ndarray:
        loc = np.atleast_2d(loc)
        return self.center + loc[:, :1] * self.u + loc[:, 1:2] * self.v

    def corners(self) -> np.ndarray:
        s = self.half
        return self.world(np.array([[-s[0], -s[1]], [s[0], -s[1]], [s[0], s[1]], [-s[0], s[1]]]))

    def clearance_map(self, margin: float) -> np.ndarray:
        """Distance (m) from each free cell to the nearest content or wall."""
        free = ~self.grid
        padded = np.zeros((free.shape[0] + 2, free.shape[1] + 2), np.uint8)
        padded[1:-1, 1:-1] = free
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * self.cell
        return dist

    def describe(self) -> str:
        return (
            f"interior {self.half[0] * 200:.1f} x {self.half[1] * 200:.1f} cm, centre ({self.center[0]:.3f}, {self.center[1]:.3f}), "
            f"long axis {np.degrees(np.arctan2(self.u[1], self.u[0])):+.0f} deg, floor z {self.floor_z:.3f}, rim z {self.rim_z:.3f}, "
            f"{self.grid.mean() * 100:.0f}% of the floor occupied"
        )


# ---------------------------------------------------------------------------
# grasp
# ---------------------------------------------------------------------------


@dataclass
class GraspPlan:
    side: str
    center: np.ndarray  # fingertip centre xy (base)
    axis: np.ndarray  # unit finger-opening direction (base xy)
    z: float  # fingertip height (base)
    surface_z: float  # top of the part being held (base z)
    width: float  # object width between the fingers
    grip_open: float  # gripper command to open to
    hang: float  # how far the object's bottom hangs below the fingertips
    foot: np.ndarray  # held object's footprint in the gripper frame: [a_min, a_max, b_min, b_max]
    # (a along the finger axis, b across), relative to the fingertip centre
    clearance: float  # nearest other thing to a fingertip, m
    note: str = ""

    @property
    def axis3(self) -> np.ndarray:
        return np.array([self.axis[0], self.axis[1], 0.0])


def _extent(vals: np.ndarray, lo: float = 3.0, hi: float = 97.0) -> Tuple[float, float]:
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def plan_grasp(view: TableView, obj: Found, side: str, **kw: object) -> GraspPlan:
    """The best of plan_grasps."""
    return plan_grasps(view, obj, side, **kw)[0]  # type: ignore[arg-type]


def plan_grasps(
    view: TableView,
    obj: Found,
    side: str,
    obstacles: Optional[np.ndarray] = None,
    margin: float = 0.02,
    floor_clear: float = 0.008,
    tip_radius: float = 0.012,
    min_clearance: float = 0.005,
    n_best: int = 4,
) -> List[GraspPlan]:
    """Where the fingers go, in one arm's frame -- the ``n_best`` workable answers, best
    first, each with a different finger direction, so that a caller whose arm cannot solve
    the first one has something else to try.

    The part to hold is looked at in the band the fingertips will reach into (everything at
    or above the grasp height). Its long direction is found, and a window is slid along it:
    at each spot, and for a few finger directions around 'across the narrow side', the width
    the fingers would close on and the room each fingertip has from the neighbours are
    measured. The spot nearest the middle that fits, with the most room, wins -- which is a
    cube's centre, a hammer's handle, and the narrowest part of a plush toy that is too fat
    to take round the middle."""
    spec = VOCAB[obj.name]
    model = view.models[side]
    uv_all = obj.pixels(stride=2)
    uv_g = obj.pixels(obj.grasp_mask, stride=1)
    if len(uv_g) < 20:
        raise ValueError(f"{obj.name}: too few pixels on the part to hold")

    if spec.grasp == "handle":
        top_z = model.table_z + spec.handle_height
        z = max(model.table_z + spec.handle_height * 0.45, model.table_z + floor_clear)
        slice_uv = uv_g
        h_lo, h_hi = 0.0, spec.handle_height
    else:
        top_z = model.table_z + obj.top_h
        z = max(top_z - spec.below_top, model.table_z + floor_clear)
        z = min(z, top_z - 0.01)
        h_g = obj.h[uv_g[:, 1], uv_g[:, 0]]
        band = np.isfinite(h_g) & (h_g >= (z - model.table_z) - 0.004)
        slice_uv = uv_g[band] if band.sum() >= 30 else uv_g
        h_lo, h_hi = z - model.table_z, obj.top_h
    # The whole object as a footprint on the table -- that is what the fingers have to clear
    # when they come down around it, wherever on its height they close. The slice the
    # fingertips actually reach into is only used to say which way it is long.
    foot_all = view.footprint(side, uv_all, 0.0, obj.top_h if spec.trust_depth else spec.height)
    xy = view.footprint(side, slice_uv, h_lo, h_hi) if spec.grasp == "handle" else foot_all
    cen = xy.mean(0)
    cov = (xy - cen).T @ (xy - cen) / max(len(xy), 1)
    evals, evecs = np.linalg.eigh(cov)
    long_axis = evecs[:, int(np.argmax(evals))]
    short_axis = np.array([-long_axis[1], long_axis[0]])
    t = (xy - cen) @ long_axis
    t_lo, t_hi = _extent(t, 2, 98)
    length = t_hi - t_lo
    if spec.grasp == "handle":
        # the free end is the handle end farther from the whole tool's centroid
        c_all = foot_all.mean(0)
        t_c = float((c_all - cen) @ long_axis)
        free_end, other = (t_lo, t_hi) if abs(t_lo - t_c) > abs(t_hi - t_c) else (t_hi, t_lo)
        # the preferred spot first, then the rest of the middle half of the handle: the
        # tools lie close together and a neighbour often sits where a fingertip would land
        t_pref = free_end + spec.along * (other - free_end)
        spots = [t_pref] + [
            float(v)
            for v in np.arange(t_lo + 0.25 * length, t_hi - 0.25 * length + 1e-9, 0.01)
            if abs(v - t_pref) > 0.005
        ]
        window = 0.015
        margin = min(margin, 0.016)
    elif length < 0.05:
        spots = [0.0]
        window = 0.5 * length + 0.01
    else:
        spots = [0.0] + [float(v) for v in np.arange(t_lo + 0.02, t_hi - 0.02 + 1e-9, 0.01)]
        window = 0.015
    obs_xy = obstacles[:, :2] if obstacles is not None and len(obstacles) else None
    base_ang = float(np.arctan2(short_axis[1], short_axis[0]))  # fingers across the narrow side
    trials = [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0, 90.0]
    found: List[Tuple[float, np.ndarray, np.ndarray, float, float, float, float, float]] = []
    why: List[str] = []
    if spec.grasp == "handle":
        trials = [d for d in trials if abs(d) <= 45.0]
    for t_c in spots:
        c0 = cen + long_axis * t_c
        for d_ang in trials:
            ang = base_ang + np.radians(d_ang)
            axis = np.array([np.cos(ang), np.sin(ang)])
            perp = np.array([-axis[1], axis[0]])
            # the strip the fingers close across: a band as wide as a fingertip, running
            # along the finger axis through this spot
            b_all = (xy - c0) @ perp
            win = xy[np.abs(b_all) < window]
            if len(win) < 12:
                continue
            a = (win - cen) @ axis
            b = (win - cen) @ perp
            a0, a1 = _extent(a, 1, 99)
            width = max(a1 - a0, spec.min_width)
            if width + 0.012 > OPEN_STROKE:
                if t_c == spots[0]:
                    why.append(f"{d_ang:+.0f} deg: {width * 100:.1f} cm too wide")
                continue
            centre = cen + axis * (0.5 * (a0 + a1)) + perp * float(np.median(b))
            opening = min(OPEN_STROKE, width + margin)
            clearance = 1.0
            if obs_xy is not None:
                for sgn in (-1.0, 1.0):
                    tip = centre + axis * (sgn * opening / 2)
                    clearance = min(clearance, float(np.min(np.linalg.norm(obs_xy - tip, axis=1))) - tip_radius)
            if clearance < min_clearance:
                if t_c == spots[0]:
                    why.append(f"{d_ang:+.0f} deg: a fingertip lands {max(clearance, 0) * 100:.1f} cm from something")
                continue
            score = min(clearance, 0.04) - 0.15 * width - 0.0004 * abs(d_ang) - 0.25 * abs(t_c - spots[0])
            found.append((score, axis, centre, width, opening, clearance, d_ang, t_c))
    if not found:
        raise ValueError(f"{obj.name}: no finger direction works -- " + "; ".join(why))
    found.sort(key=lambda f: -f[0])
    out: List[GraspPlan] = []
    for _, axis, centre, width, opening, clearance, d_ang, t_c in found:
        if any(abs(float(np.cross(axis, o.axis))) < 0.15 and np.linalg.norm(centre - o.center) < 0.01 for o in out):
            continue  # the same answer at the next spot along
        perp = np.array([-axis[1], axis[0]])
        rel = foot_all - centre
        a_lo, a_hi = _extent(rel @ axis, 1, 99)
        b_lo, b_hi = _extent(rel @ perp, 1, 99)
        if spec.extent is not None:
            # the tool from its prior: the head lies on from the handle's far end
            behind, ahead, hw = spec.extent
            head_dir = long_axis * np.sign(other - free_end)
            if float(head_dir @ perp) >= 0:
                b_lo, b_hi = -behind, ahead
            else:
                b_lo, b_hi = -ahead, behind
            a_lo, a_hi = min(a_lo, -hw), max(a_hi, hw)
        grip_open = float(np.clip(opening / OPEN_STROKE, 0.15, 1.0))
        where = "" if abs(t_c) < 0.005 else f", {t_c * 100:+.1f} cm along the length"
        out.append(
            GraspPlan(
                side=side,
                center=centre,
                axis=axis,
                z=float(z),
                surface_z=float(top_z),
                width=float(width),
                grip_open=grip_open,
                hang=float(z - model.table_z),
                foot=np.array([a_lo, a_hi, b_lo, b_hi]),
                clearance=float(clearance),
                note=f"{spec.grasp}{where}, fingers {d_ang:+.0f} deg off the narrow side, {width * 100:.1f} cm wide, open {opening * 100:.1f} cm, "
                f"tips {(z - model.table_z) * 100:.1f} cm over the table, tip clearance {min(clearance, 0.2) * 100:.1f} cm",
            )
        )
        if len(out) >= n_best:
            break
    return out


# ---------------------------------------------------------------------------
# place
# ---------------------------------------------------------------------------


@dataclass
class PlacePlan:
    side: str
    center: np.ndarray  # fingertip centre xy at release (base)
    axis: np.ndarray  # finger-opening direction at release
    z: float  # fingertip height at release (base)
    surface_z: float  # what the object is dropped onto (base z)
    yaw_change: float  # rotation of the finger axis from the grasp, rad
    clearance: float  # nearest content/wall to the placed footprint, m
    poly: np.ndarray  # (N,2) base xy outline of the placed footprint (fingers included)
    note: str = ""

    @property
    def axis3(self) -> np.ndarray:
        return np.array([self.axis[0], self.axis[1], 0.0])


def _rect_pts(a0: float, a1: float, b0: float, b1: float, step: float) -> np.ndarray:
    a = np.arange(a0, a1 + 1e-9, step) if a1 > a0 else np.array([a0])
    b = np.arange(b0, b1 + 1e-9, step) if b1 > b0 else np.array([b0])
    A, B = np.meshgrid(a, b, indexing="ij")
    return np.column_stack([A.ravel(), B.ravel()])


def plan_place(
    view: TableView,
    box: BoxView,
    g: GraspPlan,
    side: str,
    margin: float = 0.015,
    drop_clear: float = 0.025,
    finger_pad: float = 0.012,
    finger_width: float = 0.026,
    wall_inset: float = 0.012,
    prefer: str = "near",
    yaw_steps: int = 12,
    overhang: bool = False,
) -> PlacePlan:
    """A spot inside the box where the held object -- and the open fingers around it -- fit
    with ``margin`` to spare from the walls and from whatever is already in there.

    The footprint is the object's outline as measured at the grasp, carried rigidly by the
    fingers, plus the strip the open fingers occupy. Every finger-axis direction is tried
    (the wrist turns during the carry), and every position on a 1 cm grid; among the ones
    that fit, the nearest to the box corner on this arm's side wins, with a bonus for room
    to spare and for lying square to the box. The release height puts the object's bottom
    ``drop_clear`` above the highest thing under the footprint.

    ``overhang`` is the fallback for something longer than the box (the hammer, the saw):
    the object may then stick out over the walls as long as the fingers themselves come
    down inside, and it is let go from above the rim, to lie across it."""
    bf = box.frame(side, wall_inset=wall_inset)
    model = view.models[side]
    clear = bf.clearance_map(margin)
    nu, nv = bf.grid.shape
    a0, a1, b0, b1 = g.foot
    fa = g.grip_open * OPEN_STROKE / 2 + finger_pad
    # the corner to fill from
    corners = bf.corners()
    base = np.zeros(2)  # the arm's own base is the origin of its frame
    if prefer == "near":
        target = corners[int(np.argmin(np.linalg.norm(corners - base, axis=1)))]
    elif prefer == "far":
        target = corners[int(np.argmax(np.linalg.norm(corners - base, axis=1)))]
    else:
        target = bf.center
    target_loc = bf.local(target)[0]
    grasp_ang = float(np.arctan2(g.axis[1], g.axis[0]))
    u_ang = float(np.arctan2(bf.u[1], bf.u[0]))
    best = None
    tried = 0
    fits = 0
    step = 0.01
    us = np.arange(-bf.half[0] + 0.01, bf.half[0] - 0.01 + 1e-9, step)
    vs = np.arange(-bf.half[1] + 0.01, bf.half[1] - 0.01 + 1e-9, step)
    for k in range(yaw_steps):
        ang_loc = k * np.pi / yaw_steps  # finger axis relative to the box's long side
        ca, sa = np.cos(ang_loc), np.sin(ang_loc)
        R = np.array([[ca, -sa], [sa, ca]])
        # sample points of the footprint (object + finger strip), in the gripper frame
        obj_pts = _rect_pts(a0, a1, b0, b1, 0.005)
        fin_pts = _rect_pts(-fa, fa, -finger_width / 2, finger_width / 2, 0.005)
        pts = np.vstack([obj_pts, fin_pts]) @ R.T  # into box-local, centred on the tips
        is_finger = np.zeros(len(pts), bool)
        is_finger[len(obj_pts) :] = True
        # squareness: the object's long side (b, across the fingers) along a box axis
        square = abs(np.sin(2 * ang_loc)) < 1e-6
        for uu in us:
            for vv in vs:
                tried += 1
                p = pts + np.array([uu, vv])
                inside = (np.abs(p[:, 0]) < bf.half[0] - margin) & (np.abs(p[:, 1]) < bf.half[1] - margin)
                if not overhang and not inside.all():
                    continue
                if overhang and not inside[is_finger].all():
                    continue
                q = p[inside]
                i = np.clip(((q[:, 0] + bf.half[0]) / bf.cell).astype(int), 0, nu - 1)
                j = np.clip(((q[:, 1] + bf.half[1]) / bf.cell).astype(int), 0, nv - 1)
                c = float(np.min(clear[i, j])) if len(q) else 0.0
                if c < margin:
                    continue
                fits += 1
                dist = float(np.linalg.norm(np.array([uu, vv]) - target_loc))
                # the wrist turn from the grasp, so the arm is not asked for a full twist
                fin_ang = u_ang + ang_loc
                d_yaw = (fin_ang - grasp_ang + np.pi / 2) % np.pi - np.pi / 2
                frac_in = float(inside[~is_finger].mean())
                score = -dist + 2.0 * min(c, 0.03) + (0.01 if square else 0.0) - 0.004 * abs(d_yaw) + 0.3 * frac_in
                if best is None or score > best[0]:
                    best = (score, uu, vv, ang_loc, c, d_yaw, p, frac_in)
    if best is None:
        raise ValueError(
            f"{g.note.split(',')[0]} object does not fit anywhere in the box with {margin * 100:.1f} cm to spare "
            f"(footprint {(a1 - a0) * 100:.1f} x {(b1 - b0) * 100:.1f} cm, box {bf.half[0] * 200:.1f} x {bf.half[1] * 200:.1f} cm, "
            f"{bf.grid.mean() * 100:.0f}% occupied; {tried} spots tried)"
        )
    _, uu, vv, ang_loc, c, d_yaw, p, frac_in = best
    centre = bf.world(np.array([uu, vv]))[0]
    fin_ang = u_ang + ang_loc
    axis = np.array([np.cos(fin_ang), np.sin(fin_ang)])
    # what is under the footprint decides the release height
    i = np.clip(((p[:, 0] + bf.half[0]) / bf.cell).astype(int), 0, nu - 1)
    j = np.clip(((p[:, 1] + bf.half[1]) / bf.cell).astype(int), 0, nv - 1)
    under = float(np.max(bf.heights[i, j])) if bf.grid[i, j].any() else 0.0
    surface_z = bf.floor_z + under
    over = ""
    if frac_in < 0.999:
        # part of it will come to rest on the rim, so it has to be let go from above it
        surface_z = max(surface_z, bf.rim_z)
        over = f"; {(1 - frac_in) * 100:.0f}% of it overhangs the rim"
    z = surface_z + drop_clear + g.hang
    z = max(z, bf.floor_z + 0.01)
    poly = bf.world(p)
    hull = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2)
    return PlacePlan(
        side=side,
        center=centre,
        axis=axis,
        z=float(z),
        surface_z=float(surface_z),
        yaw_change=float(d_yaw),
        clearance=float(c),
        poly=hull,
        note=f"{fits} spots fit; chosen {c * 100:.1f} cm clear, fingers {np.degrees(ang_loc):.0f} deg to the box's long side "
        f"(wrist turns {np.degrees(d_yaw):+.0f} deg), dropped from {drop_clear * 100:.1f} cm onto {'the floor' if under < 0.005 else f'something {under * 100:.1f} cm tall'}{over}",
    )


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def draw_scene(
    view: TableView,
    box: Optional[BoxView],
    obj: Optional[Found],
    g: Optional[GraspPlan],
    p: Optional[PlacePlan],
    side: Optional[str],
    lines: Sequence[str] = (),
) -> np.ndarray:
    img = view.frame.color.copy()
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box.det.box]
        cv2.rectangle(img, (x0, y0), (x1, y1), (120, 90, 40), 1)
        s = side or next(iter(view.models))
        bf = box.frame(s)
        cs = bf.corners()
        pix = [view.to_pixel(s, np.array([c[0], c[1], bf.floor_z])) for c in cs]
        cv2.polylines(img, [np.array(pix, np.int32)], True, (255, 200, 0), 2)
        if len(bf.contents):
            for q in bf.contents[:: max(1, len(bf.contents) // 600)]:
                u, v = view.to_pixel(s, q)
                cv2.circle(img, (u, v), 2, (0, 0, 200), -1)
    if obj is not None:
        edge = obj.mask & ~cv2.erode(obj.mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[edge] = (0, 255, 0)
        if obj.grasp_mask is not obj.mask:
            edge = obj.grasp_mask & ~cv2.erode(obj.grasp_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            img[edge] = (0, 255, 255)
        x0, y0, x1, y1 = [int(v) for v in obj.box]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 180, 0), 1)
        cv2.putText(
            img,
            f"{obj.name} {obj.note}",
            (x0, max(14, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if g is not None and side is not None:
        tips = [np.array([*(g.center + sgn * g.axis * g.grip_open * OPEN_STROKE / 2), g.z]) for sgn in (-1, 1)]
        pa, pb = (view.to_pixel(side, t) for t in tips)
        cv2.line(img, pa, pb, (0, 255, 255), 2)
        for q in (pa, pb):
            cv2.drawMarker(img, q, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        c = view.to_pixel(side, np.array([*g.center, g.z]))
        cv2.circle(img, c, 5, (0, 255, 255), -1)
        cv2.putText(img, "grasp", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if p is not None and side is not None:
        pix = [view.to_pixel(side, np.array([q[0], q[1], p.surface_z])) for q in p.poly]
        cv2.polylines(img, [np.array(pix, np.int32)], True, (255, 0, 255), 2)
        tips = [
            np.array([*(p.center + sgn * p.axis * (g.grip_open if g else 0.5) * OPEN_STROKE / 2), p.z])
            for sgn in (-1, 1)
        ]
        pa, pb = (view.to_pixel(side, t) for t in tips)
        cv2.line(img, pa, pb, (255, 0, 255), 2)
        c = view.to_pixel(side, np.array([*p.center, p.z]))
        cv2.circle(img, c, 5, (255, 0, 255), -1)
        cv2.putText(img, "release", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return img
