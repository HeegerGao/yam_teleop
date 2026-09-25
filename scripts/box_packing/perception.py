"""Open-vocabulary object localisation for the box-packing task.

GroundingDINO (text -> 2D boxes) on a colour frame, then the aligned depth turns each box into
a 3D object: the points inside the box that stand higher than ``min_height`` above the table
plane are the object, and their centroid / extent / principal axis give a grasp. Everything is
returned in the arm base frame through the top-camera calibration (calibrate_top.py).

    det = Detector()                                    # loads grounding-dino-tiny on cuda
    scene = Scene.from_calib("left")                    # T_base_cam + table plane
    objs = scene.locate(frame, det, ["white cup", "cardboard box"])
    objs[0].center_base, objs[0].top_z, objs[0].yaw

The same code runs on a wrist frame given that camera's pose in the base frame (from FK and the
wrist hand-eye), which is how the final approach is refined.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cameras import Frame  # noqa: E402

CALIB_DIR = _HERE / "calib"


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray  # xyxy pixels


class Detector:
    """GroundingDINO-tiny through transformers. ``prompts`` are joined as 'a. b. c.'"""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        t = time.time()
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        print(f"[det] {model_id} on {device} in {time.time() - t:.1f}s")

    def detect(
        self, image_bgr: np.ndarray, prompts: Sequence[str], box_threshold: float = 0.3, text_threshold: float = 0.25
    ) -> List[Detection]:
        from PIL import Image

        text = ". ".join(p.strip().rstrip(".").lower() for p in prompts) + "."
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[img.size[::-1]]
        )[0]
        labels = list(res["text_labels"] if "text_labels" in res else res["labels"])
        # transformers >= 5 hands back [''] for the labels of an empty result, one entry
        # against zero scores, so the labels are cut to the detections that exist
        labels = labels[: len(res["scores"])]
        dets = [
            Detection(str(label), float(score), box.detach().cpu().numpy().astype(np.float64))
            for label, score, box in zip(labels, res["scores"], res["boxes"], strict=True)
        ]
        dets.sort(key=lambda d: -d.score)
        return dets

    @staticmethod
    def match(dets: List[Detection], prompt: str) -> List[Detection]:
        """Detections whose label overlaps the prompt words (GroundingDINO may return a subset
        or a merge of the phrase, e.g. 'cup' or 'white cup cardboard'), best first: most
        shared words, then score. 'blue cube' therefore prefers a 'blue cube' label over a
        'red cube' one even if the latter scores higher."""
        words = set(prompt.lower().replace(".", "").split())
        hits = [(len(words & set(d.label.lower().split())), d) for d in dets]
        hits = [(n, d) for n, d in hits if n > 0]
        hits.sort(key=lambda nd: (nd[0], nd[1].score), reverse=True)
        return [d for _, d in hits]

    @staticmethod
    def assign(dets: List[Detection], prompts: Sequence[str]) -> Dict[str, Detection]:
        """One detection per prompt, no detection used twice: prompts with a full-phrase
        match are served first, then by shared-word count and score."""
        cands = []
        for prompt in prompts:
            words = set(prompt.lower().replace(".", "").split())
            for d in dets:
                shared = len(words & set(d.label.lower().split()))
                if shared:
                    cands.append((shared == len(words), shared, d.score, prompt, id(d), d))
        cands.sort(key=lambda c: c[:3], reverse=True)
        out: Dict[str, Detection] = {}
        used = set()
        for _, _, _, prompt, did, d in cands:
            if prompt in out or did in used:
                continue
            out[prompt] = d
            used.add(did)
        return out

    @staticmethod
    def assign_verified(dets: List[Detection], prompts: Sequence[str], image_bgr: np.ndarray, min_color: float = 0.15) -> Dict[str, Detection]:
        """``assign`` plus two sanity filters: a colour word in the prompt must be visible in
        the box, and boxes that overlap another assigned box (IoU > 0.5) are duplicates of one
        object under two labels -- both are dropped as ambiguous."""
        out = Detector.assign(dets, prompts)
        for prompt in list(out):
            colors = [w for w in prompt.lower().split() if w in _COLOR_HUE or w == "white"]
            if colors and color_fraction(image_bgr, out[prompt].box, colors[0]) < min_color:
                del out[prompt]
        names = list(out)
        drop = set()
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if box_iou(out[a].box, out[b].box) > 0.5:
                    drop.update((a, b))
        return {n: d for n, d in out.items() if n not in drop}


_COLOR_HUE = {"red": (0, 10, 170, 180), "orange": (8, 22), "yellow": (22, 38), "green": (38, 85), "blue": (95, 130)}


def color_fraction(image_bgr: np.ndarray, box: np.ndarray, color: str) -> float:
    """Fraction of the box's central region whose HSV hue matches ``color`` (white: low
    saturation + high value). Used to reject GroundingDINO's duplicate boxes that carry the
    wrong colour word ('blue cube' drawn on the red cube)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    crop = image_bgr[y0 + h // 4 : y0 + 3 * h // 4 + 1, x0 + w // 4 : x0 + 3 * w // 4 + 1]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if color == "white":
        return float(np.mean((sat < 60) & (val > 150)))
    if color not in _COLOR_HUE:
        return 1.0
    r = _COLOR_HUE[color]
    in_hue = (hue >= r[0]) & (hue <= r[1])
    if len(r) == 4:
        in_hue |= (hue >= r[2]) & (hue <= r[3])
    return float(np.mean(in_hue & (sat > 80) & (val > 60)))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


@dataclass
class SceneObject:
    label: str
    score: float
    box: np.ndarray  # xyxy pixels in the source frame
    center_base: np.ndarray  # (3,) centroid of the object's points, base frame
    center_top: np.ndarray  # (3,) centroid of the *top band* of points (rim / top face) -- the
    # visible-surface centroid leans towards the camera, the top band does not
    top_z: float  # highest point above the table, base frame z
    bottom_z: float  # table height under the object, base frame z
    size: np.ndarray  # (3,) extent along principal axes (x-long, x-short, height)
    yaw: float  # base-frame yaw of the footprint's long axis
    n_points: int
    points_base: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def height(self) -> float:
        return self.top_z - self.bottom_z


class Scene:
    """Camera <-> base transform plus the table plane, for one camera."""

    def __init__(self, T_base_cam: np.ndarray, table_normal_base: np.ndarray, table_d_base: float) -> None:
        self.T = np.asarray(T_base_cam, dtype=np.float64)
        self.n = np.asarray(table_normal_base, dtype=np.float64)
        self.d = float(table_d_base)

    @classmethod
    def from_calib(cls, side: str = "left", path: Optional[Path] = None) -> "Scene":
        path = path or CALIB_DIR / f"top_{side}.json"
        data = json.loads(Path(path).read_text())
        scene = cls(np.array(data["T_base_cam"]), np.array(data["table_normal_base"]), data["table_d_base"])
        scene.plane_cam = (np.array(data["table_normal_cam"]), float(data["table_d_cam"]))
        return scene

    plane_cam: Optional[Tuple[np.ndarray, float]] = None

    def table_plane_cam(self) -> Tuple[np.ndarray, float]:
        """The table plane in this camera's frame, (n, d) with n pointing at the camera."""
        if self.plane_cam is not None:
            return self.plane_cam
        n_cam = self.T[:3, :3].T @ self.n
        p_cam = self.to_cam(np.array([0.0, 0.0, self.table_z(0.0, 0.0)]))[0]
        return n_cam, float(-n_cam @ p_cam)

    @classmethod
    def from_pose(cls, T_base_cam: np.ndarray, like: "Scene") -> "Scene":
        """Another camera (e.g. a wrist cam at its current FK pose) over the same table."""
        return cls(T_base_cam, like.n, like.d)

    def to_base(self, pts_cam: np.ndarray) -> np.ndarray:
        pts_cam = np.atleast_2d(pts_cam)
        return (self.T[:3, :3] @ pts_cam.T).T + self.T[:3, 3]

    def to_cam(self, pts_base: np.ndarray) -> np.ndarray:
        pts_base = np.atleast_2d(pts_base)
        return (self.T[:3, :3].T @ (pts_base - self.T[:3, 3]).T).T

    def height(self, pts_base: np.ndarray) -> np.ndarray:
        """Signed distance above the table plane."""
        return np.atleast_2d(pts_base) @ self.n + self.d

    def table_z(self, x: float, y: float) -> float:
        """Base-frame z of the table surface under (x, y)."""
        return float(-(self.d + self.n[0] * x + self.n[1] * y) / self.n[2])

    def pixel_to_table(self, frame: Frame, u: float, v: float) -> np.ndarray:
        """Intersect the pixel ray with the table plane -> base-frame point."""
        ray_cam = np.array([(u - frame.K[0, 2]) / frame.K[0, 0], (v - frame.K[1, 2]) / frame.K[1, 1], 1.0])
        o = self.T[:3, 3]
        r = self.T[:3, :3] @ ray_cam
        t = -(self.n @ o + self.d) / (self.n @ r)
        return o + t * r

    def locate(
        self,
        frame: Frame,
        detector: Detector,
        prompts: Sequence[str],
        min_height: float = 0.012,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        keep_points: bool = False,
    ) -> List[SceneObject]:
        dets = detector.detect(frame.color, prompts, box_threshold, text_threshold)
        return [obj for obj in (self.lift(frame, d, min_height, keep_points) for d in dets) if obj is not None]

    def lift(self, frame: Frame, det: Detection, min_height: float = 0.012, keep_points: bool = False) -> Optional[SceneObject]:
        """Turn a 2D detection into a 3D object using the points above the table inside its box."""
        x0, y0, x1, y1 = [int(round(v)) for v in det.box]
        h, w = frame.depth.shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros((h, w), bool)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        pts_cam, _ = frame.point_cloud(mask)
        if len(pts_cam) < 30:
            return None
        pts = self.to_base(pts_cam)
        hgt = self.height(pts)
        above = pts[(hgt > min_height) & (hgt < 0.5)]
        if len(above) < 30:
            return None
        # A box often clips a neighbour; keep only the connected footprint cluster (1 cm
        # cells) under the box centre -- or, if that cell is empty, the largest cluster.
        above = self._footprint_cluster(above, frame, det.box)
        if len(above) < 30:
            return None
        # de-noise: drop stray points far from the footprint
        med = np.median(above[:, :2], axis=0)
        r = np.linalg.norm(above[:, :2] - med, axis=1)
        above = above[r < np.percentile(r, 90) + 0.01]
        centroid = above.mean(0)
        xy = above[:, :2] - centroid[:2]
        cov = xy.T @ xy / max(len(xy), 1)
        evals, evecs = np.linalg.eigh(cov)
        long_axis = evecs[:, int(np.argmax(evals))]
        yaw = float(np.arctan2(long_axis[1], long_axis[0]))
        proj_long = xy @ long_axis
        proj_short = xy @ np.array([-long_axis[1], long_axis[0]])
        extent = np.array(
            [
                np.percentile(proj_long, 98) - np.percentile(proj_long, 2),
                np.percentile(proj_short, 98) - np.percentile(proj_short, 2),
                float(np.percentile(self.height(above), 98)),
            ]
        )
        bottom = self.table_z(float(centroid[0]), float(centroid[1]))
        h_above = self.height(above)
        band = above[h_above >= float(extent[2]) - 0.008]  # rim / top face only, not a handle
        center_top = band.mean(0) if len(band) >= 10 else centroid
        return SceneObject(
            label=det.label,
            score=det.score,
            box=det.box.copy(),
            center_base=centroid,
            center_top=center_top,
            top_z=bottom + float(extent[2]),
            bottom_z=bottom,
            size=extent,
            yaw=yaw,
            n_points=len(above),
            points_base=above if keep_points else None,
        )

    def _footprint_cluster(self, above: np.ndarray, frame: Frame, box: np.ndarray, cell: float = 0.01) -> np.ndarray:
        from scipy import ndimage

        xy = above[:, :2]
        lo = xy.min(0) - cell
        ij = np.floor((xy - lo) / cell).astype(int)
        grid = np.zeros(ij.max(0) + 2, dtype=bool)
        grid[ij[:, 0], ij[:, 1]] = True
        labels, n = ndimage.label(grid, structure=np.ones((3, 3)))
        if n <= 1:
            return above
        lab = labels[ij[:, 0], ij[:, 1]]
        # the cluster under the box centre: deproject that pixel and find its cell
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        p_cam = frame.deproject(cx, cy)
        chosen = 0
        if p_cam is not None:
            c = self.to_base(p_cam)[0, :2]
            cij = np.floor((c - lo) / cell).astype(int)
            if 0 <= cij[0] < labels.shape[0] and 0 <= cij[1] < labels.shape[1]:
                chosen = int(labels[cij[0], cij[1]])
        if chosen == 0:
            counts = np.bincount(lab, minlength=n + 1)
            counts[0] = 0
            chosen = int(np.argmax(counts))
        return above[lab == chosen]

    def draw(self, frame: Frame, objs: Sequence[SceneObject]) -> np.ndarray:
        img = frame.color.copy()
        for o in objs:
            x0, y0, x1, y1 = [int(v) for v in o.box]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            uv = frame.project(self.to_cam(o.center_base))[0]
            cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (0, 0, 255), -1)
            uv = frame.project(self.to_cam(o.center_top))[0]
            cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (255, 0, 255), -1)
            txt = f"{o.label} {o.score:.2f} ({o.center_base[0]:.2f},{o.center_base[1]:.2f}) h{o.height * 100:.0f}cm"
            cv2.putText(img, txt, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img


def summarize(objs: Sequence[SceneObject]) -> str:
    return "\n".join(
        f"  {o.label:18s} {o.score:.2f}  centre ({o.center_base[0]:+.3f}, {o.center_base[1]:+.3f}, {o.center_base[2]:+.3f})"
        f"  top z {o.top_z:+.3f}  size {np.round(o.size * 100, 1)} cm  yaw {np.degrees(o.yaw):+.0f}  pts {o.n_points}"
        for o in objs
    )
