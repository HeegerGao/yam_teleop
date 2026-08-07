"""Simultaneously stream from every connected RealSense camera.

Each camera gets its own ``rs.pipeline`` bound to a specific serial number via
``config.enable_device(serial)`` -- that binding is what keeps multiple cameras
from fighting over the same device. The main loop then polls every pipeline
non-blockingly, so one slow camera never stalls the others.

Stream profiles are negotiated per camera rather than applied globally: models
differ in what they support (a D435i's RGB module reaches 1920x1080, a D405 tops
out at 1280x720), and a USB2 link cuts the list down further. The requested
geometry is an upper bound -- each camera opens at the closest profile it
actually advertises -- and ``--profiles`` overrides it per camera.

Examples:
    # Color only: D435i as high as it goes, both D405s at 640x480.
    realsense_multicam.py --no-depth --profiles D435=max D405=640x480@30

    # Depth + color everywhere at a common geometry.
    realsense_multicam.py --width 640 --height 480 --fps 30
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

Profile = Tuple[int, int, int]
"""(width, height, fps) of a concrete video stream profile."""

_PROFILE_RE = re.compile(r"^(?P<match>[^=]+)=(?P<spec>max|(?P<w>\d+)x(?P<h>\d+)@(?P<fps>\d+))$", re.IGNORECASE)

# Rough per-stream wire cost. Depth is z16 (2 B/px); raw color is 2 B/px too, but
# librealsense often negotiates MJPEG for color on constrained links, so this is
# only ever an upper bound.
_BYTES_PER_PIXEL = 2
_USB2_BUDGET_MB_S = 40.0
"""Practical isochronous ceiling for a whole USB 2.0 bus (~480 Mbps theoretical).
Cameras sharing one USB2 uplink share this budget."""

_STARVATION_GRACE_S = 3.0
"""Frames take a moment to start flowing; only judge throughput after this."""
_STARVED_FRACTION = 0.25
"""Below this share of the requested fps, a camera is treated as starved."""


@dataclass
class Args:
    width: int = 640
    height: int = 480
    fps: int = 30
    """Default geometry for any camera not named in --profiles. Treated as an
    upper bound -- a camera opens at the closest profile it supports, never larger."""
    profiles: Optional[List[str]] = None
    """Per-camera overrides, as MATCH=WxH@FPS or MATCH=max. MATCH is a serial
    number or a case-insensitive substring of the model name, e.g.
    'D435=max' 'D405=640x480@30'. First matching entry wins."""
    depth: bool = True
    """Stream depth (z16)."""
    color: bool = True
    """Stream color (bgr8)."""
    align: bool = False
    """Align depth into the color frame (requires both --depth and --color)."""
    serials: Optional[List[str]] = None
    """Only open these serials. Defaults to every connected camera."""
    display: bool = True
    """Show a tiled preview window. Disable for headless throughput tests."""
    main: Optional[str] = None
    """Camera shown as the large right-hand panel, matched like --profiles
    (serial or model substring). Defaults to whichever renders largest."""
    duration: float = 0.0
    """Stop after this many seconds. 0 means run until 'q' or Ctrl-C."""


def parse_overrides(entries: Optional[List[str]]) -> List[Tuple[str, Optional[Profile]]]:
    """Parse --profiles into ordered (match, profile) pairs. A None profile means
    'max', resolved per camera against what that camera advertises."""
    parsed: List[Tuple[str, Optional[Profile]]] = []
    for entry in entries or []:
        m = _PROFILE_RE.match(entry.strip())
        if not m:
            raise ValueError(f"bad --profiles entry {entry!r}; expected MATCH=WxH@FPS or MATCH=max")
        if m.group("spec").lower() == "max":
            parsed.append((m.group("match"), None))
        else:
            parsed.append((m.group("match"), (int(m.group("w")), int(m.group("h")), int(m.group("fps")))))
    return parsed


def match_override(cam: Dict, overrides: List[Tuple[str, Optional[Profile]]]) -> Optional[Tuple[str, Optional[Profile]]]:
    """First override whose MATCH is the camera's serial or a substring of its model name."""
    for match, profile in overrides:
        if match == cam["serial"] or match.lower() in cam["name"].lower():
            return match, profile
    return None


def discover(wanted: Optional[List[str]]) -> List[Dict]:
    """Return one descriptor per camera to open, carrying the rs.device so
    profiles can be negotiated against the real hardware."""
    found = [
        {
            "dev": dev,
            "serial": dev.get_info(rs.camera_info.serial_number),
            "name": dev.get_info(rs.camera_info.name),
            "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
        }
        for dev in rs.context().query_devices()
    ]
    if wanted is not None:
        by_serial = {c["serial"]: c for c in found}
        missing = [s for s in wanted if s not in by_serial]
        if missing:
            raise RuntimeError(f"requested serials not connected: {missing} (found {list(by_serial)})")
        found = [by_serial[s] for s in wanted]
    return found


def supported(dev: rs.device, stream: rs.stream, fmt: rs.format) -> Set[Profile]:
    """Every (w, h, fps) this device advertises for a stream/format, across all
    of its sensors -- a D405 exposes color on the Stereo Module, a D435i on a
    separate RGB Camera, so both are covered by sweeping sensors."""
    out: Set[Profile] = set()
    for sensor in dev.sensors:
        for prof in sensor.profiles:
            if prof.stream_type() != stream or prof.format() != fmt:
                continue
            video = prof.as_video_stream_profile()
            out.add((video.width(), video.height(), prof.fps()))
    return out


def pick_profile(profiles: Set[Profile], want: Optional[Profile], label: str) -> Profile:
    """Choose an advertised profile. ``want`` of None means the largest available
    (then the fastest at that size); otherwise the closest to the request,
    preferring one no larger so a fallback never silently inflates bandwidth."""
    if not profiles:
        raise RuntimeError(f"{label}: device advertises no profiles for this stream")
    if want is None:
        best = max(profiles, key=lambda p: (p[0] * p[1], p[2]))
        print(f"[prof] {label}: max -> {best[0]}x{best[1]}@{best[2]}")
        return best
    if want in profiles:
        return want
    target_area = want[0] * want[1]
    no_larger = [p for p in profiles if p[0] * p[1] <= target_area]
    pool = no_larger or profiles
    best = min(pool, key=lambda p: (abs(p[0] * p[1] - target_area), abs(p[2] - want[2])))
    print(f"[prof] {label}: {want[0]}x{want[1]}@{want[2]} unsupported -> using {best[0]}x{best[1]}@{best[2]}")
    return best


def resolve(cams: List[Dict], args: Args) -> None:
    """Fill in each camera's actual depth/color profiles, in place."""
    overrides = parse_overrides(args.profiles)
    default: Profile = (args.width, args.height, args.fps)
    for cam in cams:
        hit = match_override(cam, overrides)
        want = default if hit is None else hit[1]
        if hit is not None:
            shown = "max" if hit[1] is None else f"{hit[1][0]}x{hit[1][1]}@{hit[1][2]}"
            print(f"[prof] {cam['name']} {cam['serial']}: matched override {hit[0]}={shown}")
        label = f"{cam['name']} {cam['serial']}"
        cam["depth"] = (
            pick_profile(supported(cam["dev"], rs.stream.depth, rs.format.z16), want, f"{label} depth")
            if args.depth
            else None
        )
        cam["color"] = (
            pick_profile(supported(cam["dev"], rs.stream.color, rs.format.bgr8), want, f"{label} color")
            if args.color
            else None
        )


def enabled_profiles(cam: Dict) -> List[Profile]:
    """The camera's active stream profiles, depth first."""
    return [p for p in (cam["depth"], cam["color"]) if p is not None]


def expected_fps(cam: Dict) -> int:
    """A frameset only completes when every enabled stream has a frame, so the
    slowest enabled stream sets the rate."""
    return min(p[2] for p in enabled_profiles(cam))


def report_bandwidth_estimate(cams: List[Dict]) -> None:
    """Print a rough uncompressed-wire cost per USB2 camera.

    This is a loose upper bound, not a prediction: librealsense commonly
    negotiates MJPEG for color on constrained links, so real usage can be far
    below this, and what actually starves a camera is the shared upstream link
    (several cameras behind one hub contend for a single 480 Mbps uplink) rather
    than the raw byte count. Starvation is reported for real by ``check_starvation``
    once frames are flowing."""
    usb2 = [c for c in cams if c["usb"].startswith("2")]
    if not usb2:
        return
    total = 0.0
    for cam in usb2:
        pixels = sum(w * h * fps for w, h, fps in enabled_profiles(cam))
        rate = pixels * _BYTES_PER_PIXEL / 1e6
        total += rate
        print(f"[bw] {cam['name']} {cam['serial']} (USB{cam['usb']}): <={rate:.1f} MB/s uncompressed")
    print(
        f"[bw] {len(usb2)} camera(s) on USB2, <={total:.1f} MB/s total vs a ~{_USB2_BUDGET_MB_S:.0f} MB/s "
        f"per-bus ceiling (color is usually MJPEG-compressed, so actual use is lower)"
    )


def check_starvation(cam: Dict, windowed_fps: float) -> bool:
    """Report a camera that started fine but delivers almost nothing -- the actual
    signature of an oversubscribed USB link. Judged on a windowed rate, not a
    cumulative average, so startup dead time cannot mask a stall. Returns True
    when it warned, so the caller can fire this at most once per camera."""
    want = expected_fps(cam)
    if windowed_fps >= want * _STARVED_FRACTION:
        return False
    print(
        f"[bw] WARNING: {cam['name']} {cam['serial']} is delivering {windowed_fps:.1f}/{want} fps.\n"
        f"[bw]          The USB link is oversubscribed. Lower the profile, drop a stream, or move\n"
        f"[bw]          cameras off a shared hub onto separate USB3 ports."
    )
    return True


def open_pipelines(cams: List[Dict], args: Args) -> List[Dict]:
    """Start one pipeline per camera. On failure, stop whatever already started."""
    opened: List[Dict] = []
    try:
        for cam in cams:
            cfg = rs.config()
            cfg.enable_device(cam["serial"])  # <- binds this pipeline to exactly this camera
            if cam["depth"]:
                cfg.enable_stream(rs.stream.depth, *cam["depth"][:2], rs.format.z16, cam["depth"][2])
            if cam["color"]:
                cfg.enable_stream(rs.stream.color, *cam["color"][:2], rs.format.bgr8, cam["color"][2])
            pipe = rs.pipeline()
            try:
                pipe.start(cfg)
            except RuntimeError as e:
                raise RuntimeError(f"{cam['name']} ({cam['serial']}) failed to start: {e}") from e
            cam.update(
                pipe=pipe,
                aligner=rs.align(rs.stream.color) if (args.align and cam["depth"] and cam["color"]) else None,
                frames=0,
                frames_at_report=0,
                warned=False,
                last=None,
            )
            opened.append(cam)
            geom = " ".join(
                f"{tag} {p[0]}x{p[1]}@{p[2]}" for tag, p in (("depth", cam["depth"]), ("color", cam["color"])) if p
            )
            print(f"[open] {cam['name']} {cam['serial']} USB{cam['usb']}  {geom}")
        return opened
    except Exception:
        for cam in opened:
            cam["pipe"].stop()
        raise


def render(cam: Dict) -> Optional[np.ndarray]:
    """Build one camera's preview tile: color, and/or a depth colormap beside it.
    Depth and color can resolve to different geometries, so the color frame is
    scaled to the depth tile's height before they are joined."""
    frames = cam["last"]
    if frames is None:
        return None
    parts: List[np.ndarray] = []
    if cam["color"]:
        color = frames.get_color_frame()
        if color:
            parts.append(np.asanyarray(color.get_data()))
    if cam["depth"]:
        depth = frames.get_depth_frame()
        if depth:
            raw = np.asanyarray(depth.get_data())
            parts.append(cv2.applyColorMap(cv2.convertScaleAbs(raw, alpha=0.09), cv2.COLORMAP_JET))
    if not parts:
        return None
    height = parts[0].shape[0]
    scaled = [
        p if p.shape[0] == height else cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in parts
    ]
    tile = np.hstack(scaled)
    cv2.putText(tile, f"{cam['name']} {cam['serial']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return tile


def pick_main_index(cams: List[Dict], tiles: List[np.ndarray], match: Optional[str]) -> int:
    """Index of the camera to feature on the right. Falls back to the largest
    rendered tile so the highest-resolution camera leads by default."""
    if match:
        for i, cam in enumerate(cams):
            if match == cam["serial"] or match.lower() in cam["name"].lower():
                return i
    return max(range(len(tiles)), key=lambda i: tiles[i].shape[0] * tiles[i].shape[1])


def _resize_to_width(tile: np.ndarray, width: int) -> np.ndarray:
    return tile if tile.shape[1] == width else cv2.resize(tile, (width, round(tile.shape[0] * width / tile.shape[1])))


def _resize_to_height(tile: np.ndarray, height: int) -> np.ndarray:
    return tile if tile.shape[0] == height else cv2.resize(tile, (round(tile.shape[1] * height / tile.shape[0]), height))


def tile_grid(tiles: List[np.ndarray], main_index: int, max_width: int = 1600) -> np.ndarray:
    """Lay the main camera out as a full-height panel on the right, with the rest
    stacked in a column on its left.

    Both sides are scaled to a common height rather than padded, so the layout
    carries no dead black space when the cameras run at different resolutions
    (e.g. a 1080p D435i beside two VGA D405s)."""
    if len(tiles) == 1:
        grid = tiles[0]
    else:
        main = tiles[main_index]
        others = [t for i, t in enumerate(tiles) if i != main_index]
        column_width = min(t.shape[1] for t in others)
        column = np.vstack([_resize_to_width(t, column_width) for t in others])
        grid = np.hstack([_resize_to_height(column, main.shape[0]), main])
    return _resize_to_width(grid, max_width) if grid.shape[1] > max_width else grid


def main(args: Args) -> None:
    if not args.depth and not args.color:
        raise ValueError("nothing to stream: pass at least one of --depth / --color")
    if args.align and not (args.depth and args.color):
        raise ValueError("--align requires both --depth and --color")

    cams = discover(args.serials)
    if not cams:
        raise RuntimeError("no RealSense cameras found (check `lsusb | grep 8086` and cable/port)")
    print(f"[info] {len(cams)} camera(s): " + ", ".join(f"{c['name']} {c['serial']} USB{c['usb']}" for c in cams))

    resolve(cams, args)
    report_bandwidth_estimate(cams)
    cams = open_pipelines(cams, args)

    t0 = time.time()
    last_report = t0
    try:
        while True:
            for cam in cams:
                # poll_for_frames never blocks, so a stalled camera cannot hold up the rest.
                frames = cam["pipe"].poll_for_frames()
                if frames:
                    if cam["aligner"] is not None:
                        frames = cam["aligner"].process(frames)
                    cam["last"] = frames
                    cam["frames"] += 1

            now = time.time()
            if now - last_report >= 2.0:
                window = now - last_report
                parts = []
                for cam in cams:
                    fps = (cam["frames"] - cam["frames_at_report"]) / window
                    cam["frames_at_report"] = cam["frames"]
                    parts.append(f"{cam['serial']}:{fps:5.1f}fps")
                    # Warn at most once per camera, and only past the startup grace period.
                    if not cam["warned"] and now - t0 >= _STARVATION_GRACE_S:
                        cam["warned"] = check_starvation(cam, fps)
                print(f"[rate] {'  '.join(parts)}")
                last_report = now

            if args.display:
                rendered = [(cam, tile) for cam, tile in ((c, render(c)) for c in cams) if tile is not None]
                if rendered:
                    ready_cams, tiles = [list(x) for x in zip(*rendered, strict=True)]
                    main_index = pick_main_index(ready_cams, tiles, args.main)
                    cv2.imshow("RealSense multi-camera", tile_grid(tiles, main_index))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.duration and now - t0 >= args.duration:
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams:
            cam["pipe"].stop()
        if args.display:
            cv2.destroyAllWindows()
        print("[exit] pipelines stopped")


if __name__ == "__main__":
    main(tyro.cli(Args))
