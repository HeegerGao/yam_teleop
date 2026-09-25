"""Bimanual YAM teleoperation + single-operator episode recording.

Launches the four minimum_gello processes (leader/follower pair per side), opens the three
RealSense cameras (two D405 wrist cams + one D435 third-person cam), and runs an episode
recorder driven by a wireless keyboard/mouse so one person can start, operate, save,
discard, reset and continue without touching the terminal.

Controls (global via pynput on X11, plus the preview window when focused):
    Space / n / mouse LEFT   IDLE -> start recording;  RECORDING -> stop and SAVE
    f                        RECORDING -> stop and SAVE as a FAILED trajectory (episode_NNNN_failed)
    r / d / mouse RIGHT      RECORDING -> stop and DISCARD;  IDLE -> retract last saved episode
    q / Esc                  quit the session (a recording in progress is discarded; both arms
                             then open their grippers and fold down to the home rest pose
                             before the gello processes are killed)

On the way out the recorder parks the arms itself: grippers open, then one arm at a time
back to the folded home pose, and only then are the gello processes killed (killing them cuts
motor torque, and home is where an unpowered arm rests, so it settles there instead of dropping
from wherever teleop left it). The park is made over the follower RPC ports, so it needs the
leaders released -- an engaged leader owns its follower and the two command streams would fight.
If a leader is still engaged the recorder says so over TTS and waits, repeating the prompt, until
the handle is released (or --disengage-wait-s runs out, after which it skips the park).

Tip: put the wireless mouse on the floor and press it with your foot -- left click saves,
right click discards. Every state change is announced through TTS (spd-say), so you never
need to look at the screen. Engage/disengage teleop itself stays on the leader handle button.

Data layout (one directory per episode; discard = move to discarded/, videos only --
low_dim.npz and meta.json are written at save time). A failed trajectory (``f``) is a normal
save whose directory carries a ``_failed`` suffix and whose meta.json has ``failed: true``;
it still consumes an episode index, so numbering is unaffected:
    <save_root>/<task>/episode_0000/          # (or episode_0000_failed/)
        top.mp4  left_wrist.mp4  right_wrist.mp4   # color streams, one frame per tick
        low_dim.npz   # per side: joint_pos, eef_pos/quat, gripper (state); action_joint_pos,
                      # action_eef_pos/quat, action_eef_delta, action_gripper (action);
                      # engaged flags; t_mono/t_wall/cam_t_* timestamps
        meta.json     # written last -> its presence marks a complete episode; documents keys

The action channel is the leader's last command (via the follower server's get_last_command
RPC); when an arm is disengaged it falls back to that arm's measured joint pos, and the
engaged_* arrays record which one you got. EEF poses are the FK of the arm's terminal
``gripper`` mount body in each arm's own base frame (quat wxyz); action_eef_delta is the
commanded pose relative to the measured pose (base-frame dpos + base-frame axis-angle drot).

CAN buses are kernel-default names, assigned to arms by scripts/can_map.conf -- the one
mapping table, edited by hand when a reboot or replug moves the numbering. No udev persistent
names and no auto-resolution. Each adapter's un-cabled sibling netdev must stay DOWN --
sudo scripts/fix_can_links.sh puts every adapter in that state, and startup refuses to launch
if a sibling is UP (it would steal motor replies).

Usage:
    python scripts/bimanual_teleop_record.py --task pick_place
    python scripts/bimanual_teleop_record.py --sim --task smoke     # follower-only MuJoCo smoke test
    python scripts/bimanual_teleop_record.py --task x --no-launch   # attach to already-running gello processes
"""

import json
import logging
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np
import portal
import tyro

from i2rt.robots.utils import ArmType

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import realsense_multicam as rsm
from can_channels import channel_for

_MINIMUM_GELLO = _REPO_ROOT / "examples" / "minimum_gello" / "minimum_gello.py"
_CAMERA_ROLES = ("top", "left_wrist", "right_wrist")
_DISPLAY_MAX_ROW_ASPECT = 0.55
"""Upper bound on the preview camera row's height, as a fraction of --display-width, so the
window stays shorter than a 1080p screen even when only one camera is present."""
_ENGAGED_MAX_CMD_AGE_S = 0.3
"""A command younger than this means the leader is engaged and streaming."""
_INPUT_DEBOUNCE_S = 0.3
_CAM_FRESH_S = 0.5
"""All cameras must have delivered a frame this recently for an episode to start."""
_RPC_TIMEOUT_S = 2.0
"""Per-RPC deadline. Normal calls take milliseconds; without a deadline a wedged follower
server blocks the tick loop forever (portal futures wait unbounded by default)."""
_CLEANUP_STEP_TIMEOUT_S = 5.0
"""Every teardown step is given this long, then abandoned. Nothing on the way out may hang:
the step that matters (killing the gello processes, which drive the arms) is the last one."""
_HOME_Q = np.array([0.0, 0.02, 0.07, -0.13, 0.0, 0.0])
"""Folded rest pose, 6 arm joints: where the arm sits with its motors off (same pose as
box_packing/poses.py HOME_Q). Parking here means killing the gello processes drops the arm by
millimetres instead of from wherever teleop left it."""
_GRIPPER_OPEN = 1.0
"""Normalized gripper command for fully open -- released before the arm moves, so nothing stays
clamped when torque goes away."""
_PARK_SEND_HZ = 50.0
"""Command rate of the park slew. The follower holds the last command, so this only has to be
fast enough for the ramp to look continuous."""
_PARK_SETTLE_S = 0.5
"""Wait after each park ramp, so the arm reaches the command before the next step."""
_DISENGAGE_PROMPT_S = 5.0
"""How often the 'stop teleop' prompt is repeated while a leader is still engaged."""
_PARK_ENGAGED_CHECK_S = 0.3
"""How often the park slew re-checks that no leader has grabbed its follower back."""

_LIBC = None
try:
    import ctypes

    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:  # pragma: no cover -- non-glibc; the pdeathsig safety net is then skipped
    _LIBC = None


@dataclass
class Args:
    # task: str = "default_task"
    task: str = "box_folding"
    """Task name; episodes land in <save_root>/<task>/episode_NNNN."""
    save_root: str = "~/yam_data"

    # --- robot ---
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = channel_for("follower_left")
    """LEFT follower CAN netdev. Default comes from the mapping table scripts/can_map.conf."""
    can_follower_right: str = channel_for("follower_right")
    """RIGHT follower CAN netdev (from scripts/can_map.conf)."""
    can_leader_left: str = channel_for("leader_left")
    """LEFT leader CAN netdev (from scripts/can_map.conf)."""
    can_leader_right: str = channel_for("leader_right")
    """RIGHT leader CAN netdev (from scripts/can_map.conf). No udev persistent names and no
    serial-based auto-resolution any more: both kept binding to the wrong (un-cabled) channel
    of the dual-channel adapters."""
    port_left: int = 1235
    port_right: int = 1234
    bilateral_kp: float = 0.0
    launch: bool = True
    """Spawn the four minimum_gello processes. --no-launch attaches to already-running ones."""
    sim: bool = False
    """Followers run in MuJoCo, leaders are not launched. For testing without hardware."""

    # --- exit park ---
    park_on_exit: bool = True
    """On q (and on Ctrl-C), open both grippers and fold both arms down to the home rest pose
    before the gello processes are killed. --no-park-on-exit leaves the arms where they are."""
    park_joint_vel: float = 0.6
    """Joint slew limit (rad/s) of the park move. Only this move; teleop is unaffected."""
    park_gripper_vel: float = 1.0
    """Slew limit (normalized units/s) of the gripper release that precedes the park move."""
    disengage_wait_s: float = 180.0
    """How long to keep asking for the leader handle to be released before giving up on the
    park. A park move and an engaged leader are two command streams for one follower, so the
    park never starts underneath an engaged leader. A second Ctrl-C forces the exit anyway."""

    # --- cameras ---
    top_serial: Optional[str] = None
    """D435 third-person camera serial. Default: the only non-D405 camera found."""
    left_wrist_serial: Optional[str] = None
    right_wrist_serial: Optional[str] = None
    """D405 wrist camera serials. Default: the two D405s, assigned in serial order --
    check the labels in the preview window and use --swap-wrists if they are backwards."""
    swap_wrists: bool = False
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    """Color geometry for the wrist cameras."""
    top_width: int = 1920
    top_height: int = 1080
    top_fps: int = 30
    """Color geometry for the D435 third-person camera -- its full 1080p mode. Both this and
    the wrist geometry are upper bounds: a camera opens at the closest profile it advertises,
    never a larger one. 1080p is ~6x the wire cost of 640x480, so give the D435 a USB3 port
    that it does not share with the wrist cams, or it will starve (watch for the [bw] warning)."""
    allow_missing_cameras: bool = False
    """Keep going with fewer than three cameras (always allowed under --sim)."""

    # --- recording ---
    fps: int = 30
    """Episode tick rate: low-dim sampling and one video frame per tick."""
    display: bool = True
    display_width: int = 1600
    global_keys: bool = True
    """Global pynput keyboard listener. Disable if the keyboard is shared with other apps."""
    mouse: bool = True
    """Global mouse listener (left=save, right=discard) -- the 'foot pedal'."""
    tts: bool = True


# ---------------------------------------------------------------------------
# Operator IO
# ---------------------------------------------------------------------------


class Speaker:
    """Non-blocking TTS through spd-say; silently degrades to print."""

    def __init__(self, enabled: bool):
        self._cmd = shutil.which("spd-say") if enabled else None

    def say(self, text: str) -> None:
        print(f"[say] {text}")
        if self._cmd:
            try:
                subprocess.Popen([self._cmd, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logging.warning(f"spd-say failed: {e}")
                self._cmd = None


class ShutdownRequest:
    """Ctrl-C handling: first signal asks for a clean stop, second one forces the exit.

    Why not plain KeyboardInterrupt: it lands at an arbitrary bytecode, including in the
    middle of teardown, where it used to abort the cleanup *before* the gello processes were
    killed -- leaving them (and the arms they drive) running with no parent. Here the first
    SIGINT/SIGTERM/SIGHUP only sets a flag that the tick loop and the blocking startup waits
    poll, so shutdown always runs to the end.

    A second signal means the graceful path is stuck: signal the gello groups directly,
    give them a moment, kill the survivors and leave via os._exit -- deliberately skipping
    the atexit/finally machinery, since that is what would be hanging."""

    def __init__(self, procs: List["subprocess.Popen[bytes]"]):
        self._procs = procs
        self.requested = threading.Event()
        self._hits = 0
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum: int, frame: Any) -> None:
        self._hits += 1
        name = signal.Signals(signum).name
        if self._hits == 1:
            print(f"\n[exit] {name} -- shutting down (press Ctrl-C again to force)", flush=True)
            self.requested.set()
            return
        print(f"\n[exit] {name} again -- forcing: stopping the gello processes now", flush=True)
        for proc in self._procs:
            _signal_group(proc, signal.SIGINT)  # gives each one a chance to disable its motors
        deadline = time.monotonic() + 3.0
        for proc in self._procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for proc in self._procs:
            _signal_group(proc, signal.SIGKILL)
        os._exit(130)

    def raise_if_requested(self) -> None:
        """Abort a blocking startup wait. Raises where main() already expects an interrupt."""
        if self.requested.is_set():
            raise KeyboardInterrupt


class OperatorInput:
    """Collects toggle/discard/quit events from global pynput listeners and the cv2 window.

    Events are debounced at push time so pynput + cv2 double-reports and key auto-repeat
    collapse into one event; duplicate presses are therefore idempotent."""

    TOGGLE = "toggle"
    SAVE_FAILED = "save_failed"
    DISCARD = "discard"
    QUIT = "quit"

    def __init__(self, global_keys: bool, mouse: bool):
        self._events: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._last_accept = 0.0
        self._listeners: List[Any] = []
        if global_keys or mouse:
            try:
                from pynput import keyboard
                from pynput import mouse as pynput_mouse

                if global_keys:
                    self._listeners.append(keyboard.Listener(on_press=self._on_key))
                if mouse:
                    self._listeners.append(pynput_mouse.Listener(on_click=self._on_click))
                for listener in self._listeners:
                    listener.daemon = True
                    listener.start()
            except Exception as e:
                logging.warning(f"global input listeners unavailable ({e}); falling back to the preview window keys")
                self._listeners = []

    def _push(self, event: str) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_accept < _INPUT_DEBOUNCE_S:
                return
            self._last_accept = now
        self._events.put(event)

    def _on_key(self, key: Any) -> None:
        from pynput import keyboard

        if key in (keyboard.Key.space,):
            self._push(self.TOGGLE)
        elif key in (keyboard.Key.esc,):
            self._push(self.QUIT)
        elif isinstance(key, keyboard.KeyCode) and key.char:
            self.feed_char(key.char.lower())

    def _on_click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        if not pressed:
            return
        from pynput import mouse as pynput_mouse

        if button == pynput_mouse.Button.left:
            self._push(self.TOGGLE)
        elif button == pynput_mouse.Button.right:
            self._push(self.DISCARD)

    def feed_char(self, char: str) -> None:
        """Feed a key from any source (pynput chars and cv2.waitKey both land here)."""
        if char in ("n", " "):
            self._push(self.TOGGLE)
        elif char == "f":
            self._push(self.SAVE_FAILED)
        elif char in ("r", "d"):
            self._push(self.DISCARD)
        elif char in ("q", "\x1b"):
            self._push(self.QUIT)

    def pop(self) -> Optional[str]:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        for listener in self._listeners:
            listener.stop()


# ---------------------------------------------------------------------------
# Robot + camera IO
# ---------------------------------------------------------------------------


class FollowerClient:
    """Thin RPC client for one follower server: measured pos + last leader command, plus the
    position commands the exit park is made of (teleop itself is commanded by the leader)."""

    def __init__(self, name: str, port: int):
        self.name = name
        self._client = portal.Client(f"127.0.0.1:{port}")
        self.num_dofs: int = int(self._client.num_dofs().result(timeout=10.0))
        self._last_sent: Optional[np.ndarray] = None

    def read(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Returns (measured joint pos, action, engaged). The action is the leader's last
        command when it is fresh, else the measured pos (arm disengaged / never engaged)."""
        pos = np.asarray(self._client.get_joint_pos().result(timeout=_RPC_TIMEOUT_S), dtype=np.float64)
        cmd = self._client.get_last_command().result(timeout=_RPC_TIMEOUT_S)
        cmd_pos = np.asarray(cmd["pos"], dtype=np.float64)
        fresh = cmd_pos.shape == pos.shape and (time.time() - float(cmd["time"])) < _ENGAGED_MAX_CMD_AGE_S
        engaged = fresh and not self._is_own(cmd_pos)
        return pos, (cmd_pos if engaged else pos.copy()), engaged

    def _is_own(self, cmd_pos: np.ndarray) -> bool:
        """The server remembers *every* command it gets, ours included: during the exit park
        the freshest one is the ramp's own, which must not read as an engaged leader."""
        return (
            self._last_sent is not None
            and cmd_pos.shape == self._last_sent.shape
            and bool(np.array_equal(cmd_pos, self._last_sent))
        )

    def command(self, joint_pos: np.ndarray) -> None:
        """Absolute joint-position command (arm joints + normalized gripper). Only the exit
        park uses this -- during teleop the leader is the one commanding the follower."""
        q = np.asarray(joint_pos, dtype=np.float64)
        self._client.command_joint_pos(q).result(timeout=_RPC_TIMEOUT_S)
        # After the RPC returns: the server has already remembered it, so a leader command
        # arriving between the two is still seen as foreign by the next read().
        self._last_sent = q

    def close(self) -> None:
        # portal's default close() joins its socket thread with no timeout: if that thread is
        # wedged (server gone mid-call, unflushed send queue) the join never returns and the
        # whole teardown stops here, before the gello processes get killed.
        self._client.close(timeout=2.0)


class CameraRig:
    """Owns the RealSense pipelines and a poll thread caching each camera's latest frame."""

    def __init__(self, cams_by_role: Dict[str, Dict]):
        self.cams_by_role = cams_by_role
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, name="camera-poll", daemon=True)

    @property
    def roles(self) -> List[str]:
        return list(self.cams_by_role)

    def start(self) -> None:
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            for cam in self.cams_by_role.values():
                frames = cam["pipe"].poll_for_frames()
                if frames:
                    cam["last"] = frames
                    cam["last_arrival"] = time.monotonic()
                    cam["frames"] = cam.get("frames", 0) + 1
            time.sleep(0.002)

    def snapshot(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Latest color image (BGR copy) and device timestamp (ms) per role that has a frame."""
        out: Dict[str, Tuple[np.ndarray, float]] = {}
        for role in self.cams_by_role:
            frame = self.snapshot_role(role)
            if frame is not None:
                out[role] = frame
        return out

    def snapshot_role(
        self, role: str, *, after_timestamp: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Copy the latest color frame for one role, optionally only when it is new.

        The timestamp check happens before copying the pixels. This lets a high-rate recorder
        poll independently of a slower control loop without repeatedly copying a 1080p frame.
        """
        cam = self.cams_by_role.get(role)
        if cam is None:
            return None
        frames = cam["last"]
        if frames is None:
            return None
        timestamp = float(frames.get_timestamp())
        if after_timestamp is not None and timestamp == after_timestamp:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data()).copy(), timestamp

    def all_fresh(self, max_age_s: float = _CAM_FRESH_S) -> bool:
        now = time.monotonic()
        return all(now - cam.get("last_arrival", 0.0) < max_age_s for cam in self.cams_by_role.values())

    def frame_counts(self) -> Dict[str, int]:
        """Number of frames delivered to the poll loop for each role."""
        return {role: int(cam.get("frames", 0)) for role, cam in self.cams_by_role.items()}

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        for cam in self.cams_by_role.values():
            try:
                cam["pipe"].stop()
            except Exception:
                pass


def _assign_camera_roles(cams: List[Dict], args: Args) -> Dict[str, Dict]:
    """Map discovered cameras to top/left_wrist/right_wrist. Explicit serials win; otherwise
    the non-D405 is the top cam and the D405s become wrists in serial order."""
    by_serial = {c["serial"]: c for c in cams}
    roles: Dict[str, Dict] = {}
    explicit = {"top": args.top_serial, "left_wrist": args.left_wrist_serial, "right_wrist": args.right_wrist_serial}
    for role, serial in explicit.items():
        if serial is None:
            continue
        if serial not in by_serial:
            raise RuntimeError(f"--{role}_serial {serial} not connected (found {sorted(by_serial)})")
        roles[role] = by_serial.pop(serial)

    remaining = sorted(by_serial.values(), key=lambda c: c["serial"])
    if "top" not in roles:
        non_d405 = [c for c in remaining if "D405" not in c["name"].upper()]
        if non_d405:
            roles["top"] = non_d405[0]
            remaining.remove(non_d405[0])
    wrist_order = ["left_wrist", "right_wrist"]
    if args.swap_wrists:
        wrist_order.reverse()
    for role in wrist_order:
        if role not in roles and remaining:
            roles[role] = remaining.pop(0)

    for role in _CAMERA_ROLES:
        if role in roles:
            cam = roles[role]
            print(f"[cam] {role}: {cam['name']} {cam['serial']} (USB{cam['usb']})")
    missing = [r for r in _CAMERA_ROLES if r not in roles]
    if missing:
        msg = f"missing cameras for roles: {missing}"
        if args.sim or args.allow_missing_cameras:
            logging.warning(msg + " -- continuing without them")
        else:
            raise RuntimeError(msg + " (pass --allow-missing-cameras to record anyway)")
    return {role: roles[role] for role in _CAMERA_ROLES if role in roles}


def _rsm_args_for_role(args: Args, role: str) -> rsm.Args:
    """Color-only stream request for one camera role: the top cam gets its own geometry,
    every wrist cam shares the ``cam_*`` one."""
    w, h, fps = (
        (args.top_width, args.top_height, args.top_fps)
        if role == "top"
        else (args.cam_width, args.cam_height, args.cam_fps)
    )
    return rsm.Args(width=w, height=h, fps=fps, depth=False, color=True, align=False)


def _open_camera_rig(args: Args) -> CameraRig:
    try:
        cams = rsm.discover(None)
    except Exception as e:
        cams = []
        logging.warning(f"RealSense discovery failed: {e}")
    if not cams and not (args.sim or args.allow_missing_cameras):
        raise RuntimeError("no RealSense cameras found (check `lsusb | grep 8086` and cables)")
    roles = _assign_camera_roles(cams, args) if cams else {}
    if roles:
        cam_list = list(roles.values())
        # Profiles are negotiated per role -- the top cam runs at a different geometry from the
        # wrists -- so resolve() is called one camera at a time. open_pipelines() reads only
        # `align` off its Args, the geometry it opens comes from what resolve() stored per camera.
        for role, cam in roles.items():
            rsm.resolve([cam], _rsm_args_for_role(args, role))
        rsm.report_bandwidth_estimate(cam_list)
        rsm.open_pipelines(cam_list, _rsm_args_for_role(args, "top"))
        for cam in cam_list:
            cam["last_arrival"] = 0.0
    rig = CameraRig(roles)
    rig.start()
    return rig


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


class EEFKinematics:
    """Forward kinematics of the arm's terminal ``gripper`` mount body, on the arm-only MJCF.

    Both arms share one model instance (same hardware); calls are sequential in the record
    loop. Poses are expressed in the arm's own base frame, quaternions are wxyz."""

    def __init__(self, arm: str, version: int):
        arm_type = ArmType.from_string_name(arm)
        self._model = mujoco.MjModel.from_xml_path(arm_type.get_xml_path(version))
        self._data = mujoco.MjData(self._model)
        self._body_id = self._model.body("gripper").id
        self.n_arm_joints = self._model.nq

    def fk(self, joint_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(eef_pos(3), eef_quat(4, wxyz)) for the arm joints (extra dims, e.g. gripper, ignored)."""
        self._data.qpos[:] = joint_pos[: self.n_arm_joints]
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.xpos[self._body_id].copy(), self._data.xquat[self._body_id].copy()

    @staticmethod
    def pose_delta(pos_a: np.ndarray, quat_a: np.ndarray, pos_b: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
        """6D delta taking pose b (state) to pose a (target): base-frame position delta (3)
        stacked with the axis-angle of ``R_a @ R_b.T``, also in the base frame (3)."""
        quat_b_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_b_inv, quat_b)
        quat_delta = np.zeros(4)
        mujoco.mju_mulQuat(quat_delta, quat_a, quat_b_inv)
        rotvec = np.zeros(3)
        mujoco.mju_quat2Vel(rotvec, quat_delta, 1.0)
        return np.concatenate([pos_a - pos_b, rotvec])


# ---------------------------------------------------------------------------
# Episode storage
# ---------------------------------------------------------------------------


class EpisodeWriter:
    """Streams one episode to disk: a VideoWriter per camera plus in-memory low-dim buffers.

    meta.json is written last, so its presence marks a complete episode; discard tears the
    directory out into <task>/discarded/ instead of deleting it outright."""

    def __init__(self, ep_dir: Path, fps: int):
        self.ep_dir = ep_dir
        self.fps = fps
        self.ep_dir.mkdir(parents=True, exist_ok=False)
        self._writers: Dict[str, cv2.VideoWriter] = {}
        self._low: Dict[str, List[Any]] = {"t_mono": [], "t_wall": []}
        self._cam_ts: Dict[str, List[float]] = {}
        self._t0 = time.monotonic()

    @property
    def n_frames(self) -> int:
        return len(self._low["t_mono"])

    def add(self, sample: Dict[str, Any], frames: Dict[str, Tuple[np.ndarray, float]]) -> None:
        """Append one tick: every key of ``sample`` becomes an array in low_dim.npz, plus
        one video frame per camera. Samples must carry the same keys every tick."""
        self._low["t_mono"].append(time.monotonic() - self._t0)
        self._low["t_wall"].append(time.time())
        for key, value in sample.items():
            self._low.setdefault(key, []).append(value)
        for role, (img, ts) in frames.items():
            writer = self._writers.get(role)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.ep_dir / f"{role}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer for {role}")
                self._writers[role] = writer
                self._cam_ts[role] = []
            writer.write(img)
            self._cam_ts[role].append(ts)

    def _release_writers(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def save(self, meta: Dict[str, Any], failed: bool = False) -> Path:
        """Finalize the episode. ``failed=True`` marks it as a failed trajectory: meta.json gets
        ``failed: true`` and the directory is renamed with a ``_failed`` suffix."""
        self._release_writers()
        arrays = {k: np.asarray(v) for k, v in self._low.items()}
        for role, ts in self._cam_ts.items():
            arrays[f"cam_t_{role}"] = np.asarray(ts, dtype=np.float64)
        np.savez_compressed(self.ep_dir / "low_dim.npz", **arrays)
        meta = dict(
            meta,
            n_frames=self.n_frames,
            fps=self.fps,
            failed=failed,
            saved_at=datetime.now().isoformat(timespec="seconds"),
        )
        (self.ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if failed:
            failed_dir = self.ep_dir.with_name(f"{self.ep_dir.name}{FAILED_SUFFIX}")
            self.ep_dir.rename(failed_dir)
            self.ep_dir = failed_dir
        return self.ep_dir

    def discard(self) -> None:
        self._release_writers()
        _retire_episode(self.ep_dir)


FAILED_SUFFIX = "_failed"
"""Directory-name suffix of an episode saved with the ``f`` key (failed trajectory)."""


def _retire_episode(ep_dir: Path) -> None:
    """Move an episode directory into <task>/discarded/ with a timestamp suffix."""
    trash = ep_dir.parent / "discarded"
    trash.mkdir(exist_ok=True)
    shutil.move(str(ep_dir), str(trash / f"{ep_dir.name}_{datetime.now().strftime('%H%M%S')}"))


def _next_episode_index(task_dir: Path) -> int:
    indices = []
    for p in task_dir.glob("episode_*"):
        try:
            indices.append(int(p.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


# ---------------------------------------------------------------------------
# Process launching
# ---------------------------------------------------------------------------


def _adapter_serial(channel: str) -> Optional[str]:
    """USB serial of the adapter behind a CAN netdev (None for non-USB / missing).

    Only used to tell which two netdevs sit on the same physical adapter -- arm identity comes
    from the fixed channel mapping, not from serials."""
    try:
        dev = (Path("/sys/class/net") / channel / "device").resolve()
        return (dev.parent / "serial").read_text().strip()
    except OSError:
        return None


def _sibling_channels(channel: str) -> List[str]:
    """The other CAN netdev(s) of the same dual-channel adapter."""
    mine = _adapter_serial(channel)
    if mine is None:
        return []
    others = (p.name for p in sorted(Path("/sys/class/net").glob("can*")) if p.name != channel)
    return [name for name in others if _adapter_serial(name) == mine]


def _channel_is_up(channel: str) -> bool:
    try:
        flags = int((Path("/sys/class/net") / channel / "flags").read_text().strip(), 16)
        return bool(flags & 1)  # IFF_UP
    except (OSError, ValueError):
        return False


def _check_can_interface(interface: str) -> None:
    result = subprocess.run(["ip", "link", "show", interface], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"CAN interface {interface} not found -- the arm/channel mapping is fixed "
            f"(scripts/can_channels.py); run scripts/reset_all_can.sh"
        )
    if "state UP" not in result.stdout and "state UNKNOWN" not in result.stdout:
        raise RuntimeError(f"CAN interface {interface} exists but is not UP (try scripts/reset_all_can.sh)")
    # The un-cabled sibling channel of the same adapter must stay DOWN: while it is UP it
    # absorbs part of the motors' replies and the control loop sees motors drop out at random.
    up_siblings = [s for s in _sibling_channels(interface) if _channel_is_up(s)]
    if up_siblings:
        raise RuntimeError(
            f"{interface}: sibling channel(s) {up_siblings} of the same adapter are UP and will "
            f"steal motor replies -- run sudo scripts/fix_can_links.sh"
        )


def _wait_for_port(
    port: int,
    timeout_s: float = 90.0,
    proc: Optional["subprocess.Popen[bytes]"] = None,
    shutdown: Optional[ShutdownRequest] = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if shutdown is not None:
            shutdown.raise_if_requested()  # Ctrl-C during startup must not wait out the timeout
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"follower for port {port} exited with code {proc.returncode} before serving -- "
                "usually a dead CAN bus; run scripts/check_arms.py"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"follower server on port {port} did not come up within {timeout_s:.0f}s")


def _child_pdeathsig() -> None:
    """Run in the child between fork and exec: ask the kernel to SIGINT it when the recorder
    dies. Without this, anything that kills the recorder outright (SIGKILL, terminal hangup,
    a crash) leaves the gello processes -- and the arms they drive -- running unparented.

    preexec_fn is only safe in a single-threaded parent; gello processes are spawned before
    any camera/input/RPC thread starts, and libc is already loaded, so no work happens here
    beyond one syscall."""
    if _LIBC is not None:
        _LIBC.prctl(1, signal.SIGINT)  # PR_SET_PDEATHSIG


def _spawn_gello(cmd_args: List[str]) -> "subprocess.Popen[bytes]":
    cmd = [sys.executable, str(_MINIMUM_GELLO), *cmd_args]
    print(f"[launch] {' '.join(cmd)}")
    # Own session (= own process group) per gello process: teardown signals the whole group,
    # reaching the portal worker children that actually own the CAN hardware, and a terminal
    # Ctrl-C on the recorder no longer double-delivers SIGINT to them directly.
    # preexec_fn is flagged as unsafe with threads; it is suppressed below because gello
    # processes are spawned before this process starts any thread (see _child_pdeathsig).
    return subprocess.Popen(cmd, start_new_session=True, preexec_fn=_child_pdeathsig)  # noqa: PLW1509


def _launch_gello_processes(
    args: Args, procs: List["subprocess.Popen[bytes]"], shutdown: Optional[ShutdownRequest] = None
) -> None:
    """Followers first (they serve RPC), then leaders once both servers accept connections.

    Spawned processes are appended to ``procs`` (owned by the caller) as they start, so the
    caller's cleanup can reach them even when this function is interrupted mid-launch."""
    if not args.sim:
        for interface in (
            args.can_follower_left,
            args.can_follower_right,
            args.can_leader_left,
            args.can_leader_right,
        ):
            _check_can_interface(interface)
        print(
            "[can] follower-left "
            f"{args.can_follower_left}, follower-right {args.can_follower_right}, "
            f"leader-left {args.can_leader_left}, leader-right {args.can_leader_right} (scripts/can_map.conf)"
        )
        print("[launch] all CAN interfaces up")

    common = ["--arm", args.arm, "--version", str(args.version)]
    followers = [(args.can_follower_right, args.port_right), (args.can_follower_left, args.port_left)]
    follower_procs: Dict[int, subprocess.Popen[bytes]] = {}
    for can, port in followers:
        cmd = [*common, "--can_channel", can, "--gripper", args.follower_gripper, "--server_port", str(port)]
        if args.sim:
            cmd.append("--sim")
        follower_procs[port] = _spawn_gello(cmd)
        procs.append(follower_procs[port])
    for _, port in followers:
        _wait_for_port(port, proc=follower_procs[port], shutdown=shutdown)
    print("[launch] follower servers up")

    if shutdown is not None:
        shutdown.raise_if_requested()
    if not args.sim:
        leaders = [(args.can_leader_right, args.port_right), (args.can_leader_left, args.port_left)]
        for can, port in leaders:
            procs.append(
                _spawn_gello(
                    [
                        *common,
                        "--can_channel",
                        can,
                        "--gripper",
                        "yam_teaching_handle",
                        "--mode",
                        "leader",
                        "--server_port",
                        str(port),
                        "--bilateral_kp",
                        str(args.bilateral_kp),
                    ]
                )
            )


def _signal_group(proc: "subprocess.Popen[bytes]", sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)  # pid == pgid thanks to start_new_session=True
    except (ProcessLookupError, PermissionError):
        pass


def _terminate(procs: List["subprocess.Popen[bytes]"]) -> None:
    """Stop the gello process groups: SIGINT first so minimum_gello runs its own cleanup
    (stops workers, releases shared memory), SIGKILL the stragglers after a grace period.

    SIGTERM would be wrong here: Python's default SIGTERM action kills the main gello
    process without running its ``finally`` cleanup, orphaning the portal worker children
    that own the arms -- control (and terminal spam) would keep running."""
    for proc in procs:
        if proc.poll() is None:
            _signal_group(proc, signal.SIGINT)
    deadline = time.monotonic() + 8.0
    for proc in procs:
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            logging.warning(f"process {proc.pid} ignored SIGINT, killing its group")
    for proc in procs:
        _signal_group(proc, signal.SIGKILL)
        if proc.poll() is None:
            try:
                proc.wait(timeout=2.0)  # bounded: a process wedged in the kernel must not hang us
            except subprocess.TimeoutExpired:
                logging.warning(f"process {proc.pid} survived SIGKILL (stuck in a driver call?)")


def _cleanup_step(name: str, fn: Any, timeout: float = _CLEANUP_STEP_TIMEOUT_S) -> None:
    """Run one teardown step, abandoning it if it hangs.

    Each step calls into a library that can block indefinitely on a bad day (pynput's X11
    listener, librealsense's pipeline.stop, portal's socket thread). None of them is allowed
    to stop the steps that come after -- killing the gello processes is the one that matters."""
    error: List[BaseException] = []

    def run() -> None:
        try:
            fn()
        except BaseException as e:  # teardown reports, never propagates
            error.append(e)

    thread = threading.Thread(target=run, name=f"cleanup-{name}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logging.warning(f"[exit] {name} did not finish within {timeout:.0f}s -- moving on without it")
    elif error:
        logging.warning(f"[exit] {name} failed: {error[0]}")


# ---------------------------------------------------------------------------
# Exit park: grippers open, then home
# ---------------------------------------------------------------------------


def _engaged_sides(clients: Dict[str, FollowerClient]) -> List[str]:
    """Sides whose leader is currently streaming commands. Raises if a follower cannot be read."""
    return [side for side, client in clients.items() if client.read()[2]]


def _wait_disengaged(clients: Dict[str, FollowerClient], speaker: Speaker, timeout_s: float) -> bool:
    """Block until no leader is streaming commands, or the timeout passes (then False).

    An engaged leader owns its follower: a park move made underneath it is a second command
    stream for the same arm, and the arm would chase whichever RPC landed last. So this asks
    the operator -- out loud, since they are looking at the robot and not at the terminal --
    to let go of the handle and hold still, and repeats the prompt until they do."""
    deadline = time.monotonic() + timeout_s
    last_prompt = 0.0
    while True:
        try:
            engaged = _engaged_sides(clients)
        except Exception as e:
            logging.warning(f"[park] could not read the leaders ({e})")
            return False
        if not engaged:
            return True
        now = time.monotonic()
        if now - last_prompt > _DISENGAGE_PROMPT_S:
            speaker.say("stop teleop, release the leader handle and hold still")
            last_prompt = now
        print(f"[park] leader engaged on {', '.join(engaged)} -- waiting for the handle to be released", flush=True)
        if now > deadline:
            return False
        time.sleep(0.2)


def _ramp(client: FollowerClient, q_from: np.ndarray, q_to: np.ndarray, seconds: float) -> None:
    """Stream a cosine-eased joint-space ramp from ``q_from`` to ``q_to`` at _PARK_SEND_HZ.

    Raises if the side's leader re-engages mid-ramp: the operator grabbing the handle again
    outranks the park, and two command streams for one follower must never overlap."""
    dt = 1.0 / _PARK_SEND_HZ
    seconds = max(seconds, dt)
    t0 = time.monotonic()
    last_check = t0
    while True:
        now = time.monotonic()
        s = min(1.0, (now - t0) / seconds)
        client.command(q_from + (q_to - q_from) * (0.5 - 0.5 * np.cos(np.pi * s)))
        if now - last_check > _PARK_ENGAGED_CHECK_S:
            last_check = now
            if client.read()[2]:
                raise RuntimeError("leader re-engaged")
        if s >= 1.0:
            return
        time.sleep(dt)


def _park_side(client: FollowerClient, args: Args) -> None:
    """Open one arm's gripper, then fold that arm down to the home rest pose.

    The gripper goes first and on its own: whatever teleop was holding has to be let go before
    the arm carries it across the table, and the release is a slew of its own so it never snaps
    open. The ramp starts from the measured position rather than from the leader's last command,
    so the first command closes the arm's gravity sag instead of re-commanding a pose it was
    already failing to hold."""
    start = np.asarray(client.read()[0], dtype=np.float64)
    n_arm = min(6, client.num_dofs)
    if client.num_dofs > n_arm:
        released = start.copy()
        released[n_arm:] = _GRIPPER_OPEN
        _ramp(client, start, released, float(np.max(np.abs(released - start))) / args.park_gripper_vel)
        time.sleep(_PARK_SETTLE_S)
        start = released
    home = start.copy()
    home[:n_arm] = _HOME_Q[:n_arm]
    _ramp(client, start, home, float(np.max(np.abs(home[:n_arm] - start[:n_arm]))) / args.park_joint_vel)
    time.sleep(_PARK_SETTLE_S)
    err = float(np.max(np.abs(np.asarray(client.read()[0])[:n_arm] - home[:n_arm])))
    print(f"[park] {client.name} at home, worst joint {err:.3f} rad off", flush=True)


def _park_arms(clients: Dict[str, FollowerClient], speaker: Speaker, args: Args) -> None:
    """Release both grippers and fold both arms home, one arm at a time so the two never cross.

    Nothing here raises: the steps that release the hardware come after this one, and an arm
    left where it was is worse than an exit that never finishes."""
    if not clients:
        return
    if not _wait_disengaged(clients, speaker, args.disengage_wait_s):
        logging.warning("[park] a leader is still engaged -- leaving the arms where they are")
        speaker.say("still teleoperating, skipping the park")
        return
    speaker.say("opening grippers and going home")
    for side, client in clients.items():
        print(f"[park] {side} -> gripper open, home pose", flush=True)
        try:
            _park_side(client, args)
        except Exception as e:
            logging.warning(f"[park] {side} did not reach home ({type(e).__name__}: {e})")
            speaker.say(f"{side} arm did not reach home")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    recording: bool = False
    episode_index: int = 0
    saved_this_session: List[Path] = field(default_factory=list)
    n_frames: int = 0

    @property
    def n_failed(self) -> int:
        return sum(p.name.endswith(FAILED_SUFFIX) for p in self.saved_this_session)


def _cell(img: np.ndarray, width: int, height: int, label: str = "") -> np.ndarray:
    """Fit one camera frame into a width x height cell: scaled to fit, centered, padded with
    black. Letterboxing, never cropping -- the whole frame stays visible whatever its aspect
    ratio (the top cam is 16:9, the wrists 4:3). The label is drawn after scaling so it stays
    legible no matter how far the source was downscaled."""
    scale = min(width / img.shape[1], height / img.shape[0])
    w, h = max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - h) // 2, (width - w) // 2
    canvas[y : y + h, x : x + w] = cv2.resize(img, (w, h))
    if label:
        cv2.putText(canvas, label, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def _split(total: int, n: int) -> List[int]:
    """``n`` integer parts summing to exactly ``total`` (at least 1 each), so tiles laid out
    side by side always add up to the requested size instead of drifting by a rounding pixel."""
    edges = [round(i * total / n) for i in range(n + 1)]
    return [max(1, edges[i + 1] - edges[i]) for i in range(n)]


def _build_camera_row(frames: Dict[str, Tuple[np.ndarray, float]], width: int) -> np.ndarray:
    """Compose the camera tiles into one ``width``-wide image.

    The top view is the one the operator actually steers by, so it gets the whole left side at
    its full frame while the wrist views stack beside it. The layout solves for the single tile
    height H that makes ``top_width(H) + wrist_column_width(H / n) == width``, so the row fills
    the window exactly without any tile being cropped to fit."""
    tiles = [(role, frames[role][0]) for role in _CAMERA_ROLES if role in frames]
    if not tiles:
        return np.zeros((120, width, 3), dtype=np.uint8)
    top = next((img for role, img in tiles if role == "top"), None)
    wrists = [(role, img) for role, img in tiles if role != "top"]

    if top is None:
        # No top cam: fall back to one row of equal-width tiles.
        widths = _split(width, len(wrists))
        height = round(min(w * img.shape[0] / img.shape[1] for (_, img), w in zip(wrists, widths, strict=True)))
        row = np.hstack([_cell(img, w, height, role) for (role, img), w in zip(wrists, widths, strict=True)])
    elif not wrists:
        row = _cell(top, width, max(1, round(width * top.shape[0] / top.shape[1])), "top")
    else:
        n = len(wrists)
        a_top = top.shape[1] / top.shape[0]
        a_wrist = max(img.shape[1] / img.shape[0] for _, img in wrists)
        height = max(1, round(width / (a_top + a_wrist / n)))
        top_w = min(width - 1, max(1, round(a_top * height)))
        heights = _split(height, n)
        column = np.vstack(
            [_cell(img, width - top_w, h, role) for (role, img), h in zip(wrists, heights, strict=True)]
        )
        row = np.hstack([_cell(top, top_w, column.shape[0], "top"), column])

    cap = round(width * _DISPLAY_MAX_ROW_ASPECT)
    if row.shape[0] > cap:
        # A degenerate layout (one 4:3 tile alone) would otherwise make a window taller than the
        # screen: shrink the row to the cap and center it in the same width.
        row = _cell(row, width, cap)
    return row


def _build_display(
    frames: Dict[str, Tuple[np.ndarray, float]],
    state: SessionState,
    engaged: Dict[str, bool],
    width: int,
) -> np.ndarray:
    row = _build_camera_row(frames, width)

    bar = np.zeros((56, width, 3), dtype=np.uint8)
    if state.recording:
        cv2.circle(bar, (24, 28), 12, (0, 0, 255), -1)
        status = f"REC episode {state.episode_index}  frames {state.n_frames}"
        color = (0, 0, 255)
    else:
        status = f"IDLE  next episode {state.episode_index}"
        color = (0, 200, 0)
    eng = "  ".join(f"{side[0].upper()}{'#' if engaged.get(side) else '-'}" for side in ("left", "right"))
    cv2.putText(bar, status, (48, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        bar,
        f"engaged [{eng}]  saved {len(state.saved_this_session)} (failed {state.n_failed})"
        "   space/LMB: start-save  f: save-failed  r/RMB: discard  q: quit",
        (48, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    return np.vstack([bar, row])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _read_arms(clients: Dict[str, FollowerClient], kin: EEFKinematics) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """One low-dim sample across both arms: joint-space + EEF-space state and action.

    State per side: joint_pos (arm+gripper), eef_pos/eef_quat (FK of measured joints),
    gripper (measured, normalized). Action per side: action_joint_pos (leader command),
    action_eef_pos/quat (FK of the command), action_eef_delta (command pose relative to the
    measured pose: base-frame dpos + base-frame axis-angle drot), action_gripper."""
    sample: Dict[str, Any] = {}
    engaged_by_side: Dict[str, bool] = {}
    for side, client in clients.items():
        pos, cmd, engaged = client.read()
        eef_pos, eef_quat = kin.fk(pos)
        act_eef_pos, act_eef_quat = kin.fk(cmd)
        gripper_idx = kin.n_arm_joints
        sample[f"joint_pos_{side}"] = pos
        sample[f"eef_pos_{side}"] = eef_pos
        sample[f"eef_quat_{side}"] = eef_quat
        sample[f"gripper_{side}"] = pos[gripper_idx] if len(pos) > gripper_idx else np.nan
        sample[f"action_joint_pos_{side}"] = cmd
        sample[f"action_eef_pos_{side}"] = act_eef_pos
        sample[f"action_eef_quat_{side}"] = act_eef_quat
        sample[f"action_eef_delta_{side}"] = EEFKinematics.pose_delta(act_eef_pos, act_eef_quat, eef_pos, eef_quat)
        sample[f"action_gripper_{side}"] = cmd[gripper_idx] if len(cmd) > gripper_idx else np.nan
        sample[f"engaged_{side}"] = engaged
        engaged_by_side[side] = engaged
    return sample, engaged_by_side


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task_dir = Path(args.save_root).expanduser() / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    speaker = Speaker(args.tts)
    procs: List[subprocess.Popen[bytes]] = []
    clients: Dict[str, FollowerClient] = {}
    rig: Optional[CameraRig] = None
    inputs: Optional[OperatorInput] = None
    writer: Optional[EpisodeWriter] = None

    state = SessionState(episode_index=_next_episode_index(task_dir))
    dt = 1.0 / args.fps
    window = "bimanual teleop"

    # Installed before anything is launched, so a Ctrl-C during startup is handled the same
    # way as one during recording -- and so it can already reach the processes in ``procs``.
    shutdown = ShutdownRequest(procs)

    try:
        if args.launch:
            _launch_gello_processes(args, procs, shutdown)
        else:
            for port in (args.port_right, args.port_left):
                _wait_for_port(port, timeout_s=10.0, shutdown=shutdown)

        clients = {"left": FollowerClient("left", args.port_left), "right": FollowerClient("right", args.port_right)}
        print(f"[robot] followers up: left {clients['left'].num_dofs} dofs, right {clients['right'].num_dofs} dofs")
        rig = _open_camera_rig(args)
        kin = EEFKinematics(args.arm, args.version)
        inputs = OperatorInput(global_keys=args.global_keys, mouse=args.mouse)

        meta_base = {
            "task": args.task,
            "arm": args.arm,
            "version": args.version,
            "cameras": {role: cam["serial"] for role, cam in rig.cams_by_role.items()},
            # The roles no longer share one geometry, so record what each actually negotiated.
            "camera_color_profiles": {
                role: (f"{cam['color'][0]}x{cam['color'][1]}@{cam['color'][2]}" if cam.get("color") else None)
                for role, cam in rig.cams_by_role.items()
            },
            "arms": {
                "left": {"follower_can": args.can_follower_left, "leader_can": args.can_leader_left},
                "right": {"follower_can": args.can_follower_right, "leader_can": args.can_leader_right},
            },
            "action_source": "leader command via get_last_command; falls back to measured pos when disengaged",
            "low_dim_keys": {
                "joint_pos_<side>": "measured joint positions, 6 arm joints + gripper (normalized 0-1)",
                "eef_pos_<side> / eef_quat_<side>": "FK of measured arm joints: terminal mount pose in the "
                "arm's base frame, quat wxyz",
                "gripper_<side>": "measured gripper opening, normalized",
                "action_joint_pos_<side>": "leader joint-space command (absolute), same layout as joint_pos",
                "action_eef_pos_<side> / action_eef_quat_<side>": "FK of the leader command",
                "action_eef_delta_<side>": "commanded pose relative to measured pose: base-frame dpos (3) + "
                "base-frame axis-angle drot (3)",
                "action_gripper_<side>": "leader gripper command, normalized",
                "engaged_<side>": "True when the action came from a fresh leader command",
                "t_mono / t_wall / cam_t_<role>": "tick timestamps (s since episode start / unix s / camera ms)",
            },
            "sim": args.sim,
        }

        speaker.say(f"teleop ready, next episode {state.episode_index}")
        overruns = 0
        engaged: Dict[str, bool] = {"left": False, "right": False}

        while True:
            tick_start = time.monotonic()
            if shutdown.requested.is_set():
                if writer is not None:
                    writer.discard()
                    writer = None
                    speaker.say("interrupted, episode discarded")
                break
            event = inputs.pop()

            if event == OperatorInput.QUIT:
                if writer is not None:
                    writer.discard()
                    writer = None
                    speaker.say("discarded, quitting")
                else:
                    speaker.say("quitting")
                break

            if event == OperatorInput.TOGGLE:
                if writer is None:
                    if rig.cams_by_role and not rig.all_fresh():
                        speaker.say("cameras not ready")
                    else:
                        writer = EpisodeWriter(task_dir / f"episode_{state.episode_index:04d}", args.fps)
                        state.recording = True
                        speaker.say(f"recording {state.episode_index}")
                else:
                    ep_dir = writer.save(meta_base)
                    writer = None
                    state.recording = False
                    state.saved_this_session.append(ep_dir)
                    speaker.say(f"saved {state.episode_index}, total {len(state.saved_this_session)}")
                    state.episode_index += 1
            elif event == OperatorInput.SAVE_FAILED:
                if writer is not None:
                    ep_dir = writer.save(meta_base, failed=True)
                    writer = None
                    state.recording = False
                    state.saved_this_session.append(ep_dir)
                    speaker.say(
                        f"saved {state.episode_index} as failed, total {len(state.saved_this_session)}, "
                        f"failed {state.n_failed}"
                    )
                    state.episode_index += 1
                else:
                    speaker.say("not recording")
            elif event == OperatorInput.DISCARD:
                if writer is not None:
                    writer.discard()
                    writer = None
                    state.recording = False
                    speaker.say(f"discarded {state.episode_index}")
                elif state.saved_this_session:
                    retracted = state.saved_this_session.pop()
                    _retire_episode(retracted)
                    state.episode_index = _next_episode_index(task_dir)
                    speaker.say(f"retracted {retracted.name}")
                else:
                    speaker.say("nothing to retract")

            frames = rig.snapshot()
            if writer is not None:
                try:
                    sample, engaged = _read_arms(clients, kin)
                    writer.add(sample, frames)
                    state.n_frames = writer.n_frames
                except Exception as e:
                    logging.error(f"tick failed, discarding episode: {e}")
                    writer.discard()
                    writer = None
                    state.recording = False
                    speaker.say("robot connection lost, episode discarded")
            else:
                # Keep the engaged indicator live while idle (cheap: two RPCs per side).
                try:
                    _, engaged = _read_arms(clients, kin)
                except Exception:
                    engaged = {"left": False, "right": False}

            if args.display:
                cv2.imshow(window, _build_display(frames, state, engaged, args.display_width))
                key = cv2.waitKey(1) & 0xFF
                if key not in (255, 0):
                    inputs.feed_char(chr(key) if key < 128 else "")

            elapsed = time.monotonic() - tick_start
            if elapsed > dt * 1.5:
                overruns += 1
                if overruns % 100 == 1:
                    logging.warning(f"slow tick: {elapsed * 1000:.0f} ms (target {dt * 1000:.0f} ms)")
            else:
                time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        if writer is not None:
            writer.discard()
            writer = None
            speaker.say("interrupted, episode discarded")
    finally:
        # Every step is bounded and independent: the run must always reach _terminate, which
        # is what stops the arms. Ctrl-C during this stretch escalates via ShutdownRequest.
        print("[exit] stopping...", flush=True)
        if inputs is not None:
            _cleanup_step("input listeners", inputs.close)
        if args.display:
            # In the main thread: the window was created here, and GUI backends dislike being
            # torn down from another one.
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logging.warning(f"[exit] preview window: {e}")
        if rig is not None:
            _cleanup_step("cameras", rig.stop)
        # Park before anything is closed and long before _terminate: the gello processes are what
        # hold the motors, and killing them cuts torque wherever the arms happen to be. Not run
        # through _cleanup_step -- this one is allowed to take as long as its own ramps and its
        # disengage wait need, and a second Ctrl-C still forces the exit through ShutdownRequest.
        if args.park_on_exit:
            _park_arms(clients, speaker, args)
        for client in clients.values():
            _cleanup_step(f"{client.name} follower client", client.close)
        # Last and in the main thread: _terminate is bounded by construction, and this is the
        # step that must actually happen -- it is what stops the arms.
        _terminate(procs)
        n = len(state.saved_this_session)
        print(f"[exit] {n} episode(s) saved this session ({state.n_failed} failed) -> {task_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
