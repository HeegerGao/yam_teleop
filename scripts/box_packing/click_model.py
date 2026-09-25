"""Pixel <-> arm-base geometry from the two-plane click calibration (calibrate_click.py).

Two homographies, measured with the gripper at two known fingertip heights, are enough for
everything a click interface needs:

* ``xy_at(uv, z)`` -- where a pixel is, in base coordinates, at height ``z``. A camera ray is
  a straight line, so base xy varies affinely with height: interpolate (or extrapolate) the
  two planes. No camera pose, no lens model, no arm model.
* ``ray(uv)`` -- that line, as (point, direction).
* ``height(uv, depth)`` -- the base-frame height of a surface the depth camera sees at that
  pixel: the ray is intersected with the measured distance, using a camera centre recovered
  by least squares from the rays themselves.

The whole model is per arm, because each arm's base frame is its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class ClickModel:
    def __init__(self, planes: Dict[float, np.ndarray], table_z: Optional[float] = None) -> None:
        self.table_z = table_z
        zs = sorted(planes)
        if len(zs) < 2:
            raise ValueError("the click calibration needs two planes")
        self.z0, self.z1 = float(zs[0]), float(zs[-1])
        self.H0, self.H1 = np.asarray(planes[zs[0]], dtype=np.float64), np.asarray(planes[zs[-1]], dtype=np.float64)
        self._centre: Optional[np.ndarray] = None

    @classmethod
    def load(cls, side: str, calib_dir: Optional[Path] = None) -> "ClickModel":
        d = calib_dir or Path(__file__).resolve().parent / "calib"
        data = json.loads((d / f"click_{side}.json").read_text())
        table_z = None
        try:
            table_z = float(json.loads((d / f"table_z_{side}.json").read_text())["table_z"])
        except Exception:
            pass
        return cls({float(z): np.array(v["H"]) for z, v in data["planes"].items()}, table_z)

    @staticmethod
    def _apply(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
        p = H @ np.array([uv[0], uv[1], 1.0])
        return p[:2] / p[2]

    def xy_at(self, uv: np.ndarray, z: float) -> np.ndarray:
        a, b = self._apply(self.H0, uv), self._apply(self.H1, uv)
        t = (z - self.z0) / (self.z1 - self.z0)
        return a + (b - a) * t

    def point_at(self, uv: np.ndarray, z: float) -> np.ndarray:
        xy = self.xy_at(uv, z)
        return np.array([xy[0], xy[1], z])

    def ray(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(point on the ray at z0, unit direction pointing away from the camera)."""
        p0 = self.point_at(uv, self.z0)
        p1 = self.point_at(uv, self.z1)
        d = p0 - p1  # towards the camera is +z here; the ray goes down, so p1 -> p0 is 'up'
        return p1, d / np.linalg.norm(d)

    def base_z(self, height_above_table: float) -> float:
        """Base-frame height of a surface ``height_above_table`` above the table.

        Heights come from the depth camera, which measures them against the table plane it can
        see; this adds where that table is in the arm's own frame (measure_table_z.py). The
        camera pose is deliberately not reconstructed from the two planes: they sit only 6 cm
        apart, so the ray directions -- and any camera centre derived from them -- carry
        several degrees of error even though the planes themselves are good to a few mm.
        """
        if self.table_z is None:
            raise ValueError("no table height for this arm -- run measure_table_z.py")
        return float(self.table_z + height_above_table)

    def describe(self) -> str:
        t = "no table height" if self.table_z is None else f"table at z={self.table_z:.3f}"
        return f"planes at z={self.z0:.3f}/{self.z1:.3f}, {t}"
