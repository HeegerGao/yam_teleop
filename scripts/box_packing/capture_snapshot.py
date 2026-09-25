"""Grab one color+depth snapshot from every RealSense camera and save it to disk.

Saves per camera: <out>/<role>_color.png, <role>_depth.npy (uint16 mm, aligned to color),
<role>_intrinsics.json (color intrinsics after alignment). Roles are assigned like the
recorder: the non-D405 is 'top', D405s are wrists in serial order.

    python scripts/box_packing/capture_snapshot.py --out /tmp/snap
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))


@dataclass
class Args:
    out: str = "/tmp/snap"
    top_width: int = 1280
    top_height: int = 720
    wrist_width: int = 640
    wrist_height: int = 480
    warmup_frames: int = 30
    """Frames to discard while auto-exposure settles."""
    roles: Optional[list[str]] = None
    """Subset of top/left_wrist/right_wrist; default all."""


def assign_roles() -> Dict[str, str]:
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


def capture(serial: str, w: int, h: int, warmup: int) -> Dict[str, object]:
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(warmup):
            pipe.wait_for_frames(5000)
        frames = align.process(pipe.wait_for_frames(5000))
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()
        intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
    finally:
        pipe.stop()
    return {
        "color": color,
        "depth": depth,
        "depth_scale": scale,
        "intrinsics": {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy, "width": intr.width, "height": intr.height},
    }


def main(args: Args) -> None:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    roles = assign_roles()
    for role, serial in roles.items():
        if args.roles and role not in args.roles:
            continue
        w, h = (args.top_width, args.top_height) if role == "top" else (args.wrist_width, args.wrist_height)
        t = time.time()
        snap = capture(serial, w, h, args.warmup_frames)
        cv2.imwrite(str(out / f"{role}_color.png"), snap["color"])
        np.save(out / f"{role}_depth.npy", snap["depth"])
        meta = dict(snap["intrinsics"], depth_scale=snap["depth_scale"], serial=serial)
        (out / f"{role}_intrinsics.json").write_text(json.dumps(meta, indent=2))
        d = snap["depth"]
        print(f"[{role}] {serial} {w}x{h} in {time.time() - t:.1f}s; depth valid {np.mean(d > 0):.2f}, "
              f"median {np.median(d[d > 0]) * snap['depth_scale']:.3f} m; saved to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
