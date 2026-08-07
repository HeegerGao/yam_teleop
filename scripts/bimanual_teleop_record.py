"""Bimanual YAM teleoperation + single-operator episode recording.

Launches the four minimum_gello processes (leader/follower pair per side), opens the three
RealSense cameras (two D405 wrist cams + one D435 third-person cam), and runs an episode
recorder driven by a wireless keyboard/mouse so one person can start, operate, save,
discard, reset and continue without touching the terminal.

Controls (global via pynput on X11, plus the preview window when focused):
    Space / n / mouse LEFT   IDLE -> start recording;  RECORDING -> stop and SAVE
    r / d / mouse RIGHT      RECORDING -> stop and DISCARD;  IDLE -> retract last saved episode
    q / Esc                  quit the session (a recording in progress is discarded)

Tip: put the wireless mouse on the floor and press it with your foot -- left click saves,
right click discards. Every state change is announced through TTS (spd-say), so you never
need to look at the screen. Engage/disengage teleop itself stays on the leader handle button.

Data layout (one directory per episode; discard = move to discarded/, videos only --
low_dim.npz and meta.json are written at save time):
    <save_root>/<task>/episode_0000/
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

_MINIMUM_GELLO = _REPO_ROOT / "examples" / "minimum_gello" / "minimum_gello.py"
_CAMERA_ROLES = ("top", "left_wrist", "right_wrist")
_ENGAGED_MAX_CMD_AGE_S = 0.3
"""A command younger than this means the leader is engaged and streaming."""
_INPUT_DEBOUNCE_S = 0.3
_CAM_FRESH_S = 0.5
"""All cameras must have delivered a frame this recently for an episode to start."""


@dataclass
class Args:
    task: str = "default_task"
    """Task name; episodes land in <save_root>/<task>/episode_NNNN."""
    save_root: str = "~/yam_data"

    # --- robot ---
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = "can_f_white_l"
    can_follower_right: str = "can_f_white_r"
    can_leader_left: str = "can_leader_l"
    can_leader_right: str = "can_leader_r"
    port_left: int = 1235
    port_right: int = 1234
    bilateral_kp: float = 0.0
    launch: bool = True
    """Spawn the four minimum_gello processes. --no-launch attaches to already-running ones."""
    sim: bool = False
    """Followers run in MuJoCo, leaders are not launched. For testing without hardware."""

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


class OperatorInput:
    """Collects toggle/discard/quit events from global pynput listeners and the cv2 window.

    Events are debounced at push time so pynput + cv2 double-reports and key auto-repeat
    collapse into one event; duplicate presses are therefore idempotent."""

    TOGGLE = "toggle"
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
    """Thin RPC reader for one follower server: measured pos + last leader command."""

    def __init__(self, name: str, port: int):
        self.name = name
        self._client = portal.Client(f"127.0.0.1:{port}")
        self.num_dofs: int = int(self._client.num_dofs().result())

    def read(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Returns (measured joint pos, action, engaged). The action is the leader's last
        command when it is fresh, else the measured pos (arm disengaged / never engaged)."""
        pos = np.asarray(self._client.get_joint_pos().result(), dtype=np.float64)
        cmd = self._client.get_last_command().result()
        cmd_pos = np.asarray(cmd["pos"], dtype=np.float64)
        engaged = cmd_pos.shape == pos.shape and (time.time() - float(cmd["time"])) < _ENGAGED_MAX_CMD_AGE_S
        return pos, (cmd_pos if engaged else pos.copy()), engaged

    def close(self) -> None:
        self._client.close()


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
            time.sleep(0.002)

    def snapshot(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Latest color image (BGR copy) and device timestamp (ms) per role that has a frame."""
        out: Dict[str, Tuple[np.ndarray, float]] = {}
        for role, cam in self.cams_by_role.items():
            frames = cam["last"]
            if frames is None:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            out[role] = (np.asanyarray(color.get_data()).copy(), float(frames.get_timestamp()))
        return out

    def all_fresh(self, max_age_s: float = _CAM_FRESH_S) -> bool:
        now = time.monotonic()
        return all(now - cam.get("last_arrival", 0.0) < max_age_s for cam in self.cams_by_role.values())

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
        rsm_args = rsm.Args(
            width=args.cam_width, height=args.cam_height, fps=args.cam_fps, depth=False, color=True, align=False
        )
        cam_list = list(roles.values())
        rsm.resolve(cam_list, rsm_args)
        rsm.report_bandwidth_estimate(cam_list)
        rsm.open_pipelines(cam_list, rsm_args)
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

    def save(self, meta: Dict[str, Any]) -> Path:
        self._release_writers()
        arrays = {k: np.asarray(v) for k, v in self._low.items()}
        for role, ts in self._cam_ts.items():
            arrays[f"cam_t_{role}"] = np.asarray(ts, dtype=np.float64)
        np.savez_compressed(self.ep_dir / "low_dim.npz", **arrays)
        meta = dict(meta, n_frames=self.n_frames, fps=self.fps, saved_at=datetime.now().isoformat(timespec="seconds"))
        (self.ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self.ep_dir

    def discard(self) -> None:
        self._release_writers()
        _retire_episode(self.ep_dir)


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


def _check_can_interface(interface: str) -> None:
    result = subprocess.run(["ip", "link", "show", interface], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"CAN interface {interface} not found")
    if "state UP" not in result.stdout and "state UNKNOWN" not in result.stdout:
        raise RuntimeError(f"CAN interface {interface} exists but is not UP (try scripts/reset_all_can.sh)")


def _wait_for_port(port: int, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"follower server on port {port} did not come up within {timeout_s:.0f}s")


def _spawn_gello(cmd_args: List[str]) -> "subprocess.Popen[bytes]":
    cmd = [sys.executable, str(_MINIMUM_GELLO), *cmd_args]
    print(f"[launch] {' '.join(cmd)}")
    # Own session (= own process group) per gello process: teardown signals the whole group,
    # reaching the portal worker children that actually own the CAN hardware, and a terminal
    # Ctrl-C on the recorder no longer double-delivers SIGINT to them directly.
    return subprocess.Popen(cmd, start_new_session=True)


def _launch_gello_processes(args: Args) -> List["subprocess.Popen[bytes]"]:
    """Followers first (they serve RPC), then leaders once both servers accept connections."""
    if not args.sim:
        for interface in (
            args.can_follower_left,
            args.can_follower_right,
            args.can_leader_left,
            args.can_leader_right,
        ):
            _check_can_interface(interface)
        print("[launch] all CAN interfaces up")

    common = ["--arm", args.arm, "--version", str(args.version)]
    procs: List[subprocess.Popen[bytes]] = []
    followers = [(args.can_follower_right, args.port_right), (args.can_follower_left, args.port_left)]
    for can, port in followers:
        cmd = [*common, "--can_channel", can, "--gripper", args.follower_gripper, "--server_port", str(port)]
        if args.sim:
            cmd.append("--sim")
        procs.append(_spawn_gello(cmd))
    for _, port in followers:
        _wait_for_port(port)
    print("[launch] follower servers up")

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
    return procs


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
            proc.wait()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    recording: bool = False
    episode_index: int = 0
    saved_this_session: List[Path] = field(default_factory=list)
    n_frames: int = 0


def _build_display(
    frames: Dict[str, Tuple[np.ndarray, float]],
    state: SessionState,
    engaged: Dict[str, bool],
    width: int,
) -> np.ndarray:
    tiles = []
    for role in _CAMERA_ROLES:
        if role in frames:
            tile = frames[role][0].copy()
            cv2.putText(tile, role, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(tile)
    if tiles:
        height = min(t.shape[0] for t in tiles)
        row = np.hstack([cv2.resize(t, (round(t.shape[1] * height / t.shape[0]), height)) for t in tiles])
        if row.shape[1] != width:
            row = cv2.resize(row, (width, round(row.shape[0] * width / row.shape[1])))
    else:
        row = np.zeros((120, width, 3), dtype=np.uint8)

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
        f"engaged [{eng}]  saved {len(state.saved_this_session)}   space/LMB: start-save  r/RMB: discard  q: quit",
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

    try:
        if args.launch:
            procs = _launch_gello_processes(args)
        else:
            for port in (args.port_right, args.port_left):
                _wait_for_port(port, timeout_s=10.0)

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
        if inputs is not None:
            inputs.close()
        if args.display:
            cv2.destroyAllWindows()
        if rig is not None:
            rig.stop()
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass
        _terminate(procs)
        n = len(state.saved_this_session)
        print(f"[exit] {n} episode(s) saved this session -> {task_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
