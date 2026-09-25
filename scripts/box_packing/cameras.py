"""RealSense rig for the box-packing pipeline: colour + depth, aligned, per role.

Same role assignment as the recorder (non-D405 = ``top``, the D405s are wrists in serial
order) but every camera streams depth as well, aligned into the colour frame, and exposes its
intrinsics -- the perception code needs metric 3D points, not just pictures.

    rig = CameraRig.open()          # all three
    frame = rig.grab("top")         # Frame(color BGR, depth metres, K, ...)
    xyz = frame.deproject(u, v)     # 3D point in that camera's optical frame
    rig.close()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

ROLES = ("top", "left_wrist", "right_wrist")
_GEOMETRY = {"top": (1280, 720, 30), "left_wrist": (640, 480, 30), "right_wrist": (640, 480, 30)}


@dataclass
class Frame:
    role: str
    color: np.ndarray  # HxWx3 BGR uint8
    depth: np.ndarray  # HxW float32 metres, 0 = invalid, aligned to color
    K: np.ndarray  # 3x3 colour intrinsics
    t: float  # monotonic arrival time

    def deproject(self, u: float, v: float, z: Optional[float] = None) -> Optional[np.ndarray]:
        """3D point (camera optical frame: x right, y down, z forward) of pixel (u, v)."""
        if z is None:
            z = self.depth_at(u, v)
        if z is None or z <= 0:
            return None
        return np.array([(u - self.K[0, 2]) / self.K[0, 0] * z, (v - self.K[1, 2]) / self.K[1, 1] * z, z])

    def depth_at(self, u: float, v: float, radius: int = 3) -> Optional[float]:
        """Median valid depth in a small window -- single pixels on the D4xx are noisy/holey."""
        h, w = self.depth.shape
        u0, v0 = int(round(u)), int(round(v))
        win = self.depth[max(0, v0 - radius) : min(h, v0 + radius + 1), max(0, u0 - radius) : min(w, u0 + radius + 1)]
        vals = win[win > 0]
        return float(np.median(vals)) if vals.size else None

    def point_cloud(self, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(points Nx3, pixel indices Nx2 (u, v)) for valid depth, optionally within ``mask``."""
        valid = self.depth > 0
        if mask is not None:
            valid &= mask.astype(bool)
        v, u = np.nonzero(valid)
        z = self.depth[v, u]
        x = (u - self.K[0, 2]) / self.K[0, 0] * z
        y = (v - self.K[1, 2]) / self.K[1, 1] * z
        return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Nx3 camera-frame points -> Nx2 pixels."""
        pts = np.atleast_2d(pts)
        u = pts[:, 0] / pts[:, 2] * self.K[0, 0] + self.K[0, 2]
        v = pts[:, 1] / pts[:, 2] * self.K[1, 1] + self.K[1, 2]
        return np.stack([u, v], axis=1)

    def save(self, directory: Path, stem: Optional[str] = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = stem or self.role
        cv2.imwrite(str(directory / f"{stem}_color.png"), self.color)
        np.save(directory / f"{stem}_depth.npy", self.depth)
        (directory / f"{stem}_K.json").write_text(json.dumps(self.K.tolist()))

    @staticmethod
    def load(directory: Path, stem: str, role: Optional[str] = None) -> "Frame":
        directory = Path(directory)
        color = cv2.imread(str(directory / f"{stem}_color.png"))
        depth = np.load(directory / f"{stem}_depth.npy").astype(np.float32)
        K = np.array(json.loads((directory / f"{stem}_K.json").read_text()))
        return Frame(role or stem, color, depth, K, 0.0)


def _assign_roles() -> Dict[str, str]:
    devs = [
        (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()
    ]
    roles: Dict[str, str] = {}
    d405 = sorted(s for n, s in devs if "D405" in n.upper())
    other = [s for n, s in devs if "D405" not in n.upper()]
    if other:
        roles["top"] = other[0]
    for role, serial in zip(("left_wrist", "right_wrist"), d405, strict=False):
        roles[role] = serial
    return roles


class _Cam:
    def __init__(self, role: str, serial: str, geometry: Tuple[int, int, int]) -> None:
        self.role = role
        self.serial = serial
        w, h, fps = geometry
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipe = rs.pipeline()
        profile = self.pipe.start(cfg)
        self.scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        # light depth post-processing: fill small holes, drop flying pixels
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.last: Optional[Frame] = None
        self.n = 0

    def poll(self) -> None:
        frames = self.pipe.poll_for_frames()
        if not frames:
            return
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return
        depth = self.temporal.process(self.spatial.process(depth))
        d = np.asanyarray(depth.get_data()).astype(np.float32) * self.scale
        c = np.asanyarray(color.get_data()).copy()
        if d.shape != c.shape[:2]:
            d = cv2.resize(d, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.last = Frame(self.role, c, d, self.K, time.monotonic())
        self.n += 1

    def stop(self) -> None:
        try:
            self.pipe.stop()
        except Exception:
            pass


class CameraRig:
    """Background-polled cameras; ``grab`` returns the newest frame, ``grab_fresh`` waits for
    frames captured after the call (use after moving the arm, to avoid a stale image)."""

    def __init__(self, cams: Dict[str, _Cam]) -> None:
        self.cams = cams
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="rig-poll", daemon=True)
        self._thread.start()

    @classmethod
    def open(cls, roles: Iterable[str] = ROLES, geometry: Optional[Dict[str, Tuple[int, int, int]]] = None) -> "CameraRig":
        serials = _assign_roles()
        geometry = dict(_GEOMETRY, **(geometry or {}))
        cams: Dict[str, _Cam] = {}
        for role in roles:
            if role not in serials:
                raise RuntimeError(f"no camera for role {role} (found {serials})")
            cams[role] = _Cam(role, serials[role], geometry[role])
            print(f"[rig] {role}: {serials[role]} {geometry[role]}")
        rig = cls(cams)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(c.last is None for c in cams.values()):
            time.sleep(0.05)
        missing = [r for r, c in cams.items() if c.last is None]
        if missing:
            rig.close()
            raise RuntimeError(f"cameras delivered no frames: {missing}")
        return rig

    def _loop(self) -> None:
        while not self._stop.is_set():
            for cam in self.cams.values():
                with self._lock:
                    cam.poll()
            time.sleep(0.002)

    def grab(self, role: str) -> Frame:
        with self._lock:
            frame = self.cams[role].last
        assert frame is not None
        return frame

    def grab_fresh(self, role: str, min_frames: int = 5, timeout: float = 3.0) -> Frame:
        """A frame at least ``min_frames`` frames after now -- the temporal filter and auto
        exposure both need a few frames after a scene change."""
        with self._lock:
            n0 = self.cams[role].n
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.cams[role].n >= n0 + min_frames:
                    return self.cams[role].last  # type: ignore[return-value]
            time.sleep(0.01)
        return self.grab(role)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        for cam in self.cams.values():
            cam.stop()
