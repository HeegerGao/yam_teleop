"""Real-robot eval of the π₀.₅ steering policies on the bimanual YAM.

Brings up the two follower arms (no leaders), the three RealSense cameras at the demo geometry
(top 1920x1080, wrists 640x480, 30 fps), and the policy server from steering_policy_server.py in its
own venv. Models, cameras and followers stay alive across episodes:

    1. (--goto-start) slow slew to the task's configured start pose (normally the demo frame-0 median)
    2. (--wait-for-start) the arms hold there while the operator resets the scene; SPACE or Enter
       starts the first episode, Esc quits. No Enter needed after q / Esc, and the prompt is repeated on
       screen, in the preview window and over TTS, because the follower logs scroll it away.
    3. closed loop at 30 Hz until q / --max-seconds:
         every tick   grab 3 frames + both joint states, stream the command, record one row
         plans        a 50-step chunk (1.67 s) from the newest observation, requested every
                      --replan-every ticks; ~130 ms of inference, so a chunk lands ~4-5 steps old
                      and is read on its own time axis, (now - t_obs) * 30, never from step 0
    4. q (or --max-seconds) ends the current episode:
         a. save the recording (the arms hold still meanwhile)
         b. open both grippers where they stand
         c. fold both arms to the rest pose, via the working pose
         d. move to this task's ready pose, reset policy history and wait for a fresh Enter
            (with --label-outcome, first press s / f to label the previous recording)
    5. Esc saves the episode, releases grippers and parks according to --park-pose, then
         powers off: the followers this script launched are stopped, which disables the motors
            (with --no-launch the followers keep running and the arms stay powered, folded)
    6. (--label-outcome) s / f / Enter in the terminal labels the last saved episode after exit

q works from the terminal the run was started in, from the preview window, and from anywhere on
the desktop (pynput, --no-global-keys to disable). q before the episode starts also folds and
returns to the ready pose. Esc during this return switches to the exit sequence.
Ctrl-C is the emergency stop: the arms hold their last
command, the recording is still saved, and the launched followers are powered off without a fold.

**Recording.** Same layout, resolutions, rate and low_dim keys as bimanual_teleop_record.py writes
for a demo, so eval episodes and demos load with the same code:

    <save_root>/steering_<task>/<model>/episode_NNNN[_failed]/
        top.mp4 (1920x1080)  left_wrist.mp4 (640x480)  right_wrist.mp4 (640x480)   30 fps, 1 frame/tick
        low_dim.npz   joint_pos_/eef_pos_/eef_quat_/gripper_<side>            measured state
                      action_joint_pos_/action_eef_pos_/action_eef_quat_/
                      action_eef_delta_/action_gripper_<side>                 command actually streamed
                      engaged_<side>                                          True while the policy drives
                      t_mono / t_wall / cam_t_<role>
                      + policy_joint_pos_<side> (chunk target before the safety limiter),
                        plan_age, plan_latency, plan_infer, plan_index
                      + plan_rejected (fresh chunk exceeded the task's switch-jump guard)
        meta.json     written last; outcome success/failure/null, checkpoint, timing stats, args

Before the first plan lands the action is the measured pose, like the recorder's disengaged
fallback. An episode labelled ``f`` gets the recorder's ``_failed`` suffix. Videos are encoded on a
writer thread so a slow 1080p encode never stretches a control tick; each row keeps the tick's own
timestamp.

**Safety.** ``--execute`` is required to move the arms. Commands are clamped to the URDF joint limits
and gripper 0-1, slew-limited by --max-joint-vel, and the run refuses to engage if the first target
is more than --max-start-error from the measured pose. For repeated episodes keep the followers alive:

    scripts/run_policy_followers.sh                                          # terminal 1
    python scripts/steering_policy_eval.py --task cloth --no-launch --execute   # terminal 2, repeat

Usage:
    python scripts/steering_policy_eval.py --task cloth                       # dry run, records
    python scripts/steering_policy_eval.py --task pingpong --execute
    python scripts/steering_policy_eval.py --task cup_pingpong --execute
    python scripts/steering_policy_eval.py --task cup_pingpong --model steeract_enc --execute
    python scripts/steering_policy_eval.py --task cup_pingpong --model steeract_dec --execute
    python scripts/steering_policy_eval.py --task cup_pingpong --model drvla --execute
    python scripts/steering_policy_eval.py --task cup_pingpong --model coast --execute
    python scripts/steering_policy_eval.py --task cloth --sim --allow-missing-cameras --execute

Offline plumbing check against the demos first: scripts/steering_policy_offline_check.py.
"""

from __future__ import annotations

import json
import logging
import queue
import select
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import bimanual_teleop_record as rec  # cameras, FK, episode writer, follower launch/teardown
from box_folding_policy_rollout import (
    FOLDED_Q,
    GRIPPER_OPEN,
    WORKING_Q,
    Ema,
    FollowerArm,
    SafetyLimiter,
    _check_attached_followers,
    _joint_range,
    _launch_followers,
    _slew_to,
    _stop_sequence,
)
from can_channels import channel_for
from steering_policy_models import RELEASE_ROOT, Model, new_episode, resolve_assets
from steering_policy_server import (
    DEFAULT_PYTHON,
    VIEWS,
    PolicyClient,
    demo_dir,
)

SIDES = ("left", "right")
_NDOF = 7  # 6 arm joints + normalised gripper, per side
_WINDOW = "steering policy eval"
_START_KEYS = (" ", "\n", "\r")
_QUIT_KEYS = ("\x1b",)
_PROMPT_REPEAT_S = 8.0
START_ADJUST: Dict[str, Dict[str, float]] = {
    "cloth": {"start_forward": 0.08, "start_down": 0.0, "start_tilt_to_base_deg": 0.0},
    "pingpong": {"start_forward": 0.0, "start_down": 0.05, "start_tilt_to_base_deg": 30.0},
    "cup_pingpong": {"start_forward": 0.0, "start_down": 0.0, "start_tilt_to_base_deg": 0.0},
    "push_cube": {"start_forward": 0.0, "start_down": 0.0, "start_tilt_to_base_deg": 0.0},
}
"""Per-task defaults for the start-pose adjustment (metres, metres, degrees). A flag given on the
command line wins; tasks not listed use the demo median unchanged."""

START_ADJUST_SIDES: Dict[str, Tuple[str, ...]] = {"push_cube": ("left",)}
"""Tasks whose start-pose Cartesian adjustment applies only to the listed active arms."""

POLICY_ACTIVE_SIDES: Dict[str, Tuple[str, ...]] = {"push_cube": ("left",)}
"""Tasks where the policy may command only the listed arms; other arms hold their demo start pose."""

_CUP_PINGPONG_START_POSE = {
    "left": np.array([0.123026, 0.883688, 1.204128, -0.562486, -0.007057, -0.051690, 0.999712]),
    "right": np.array([0.033761, 1.023690, 1.152247, -0.713169, -0.018883, -0.011635, 0.999711]),
}
MANUAL_START_POSES: Dict[str, Dict[str, np.ndarray]] = {
    "cup_pingpong": _CUP_PINGPONG_START_POSE,
}
"""Per-task arm poses that replace the corresponding side of the median demo frame-0 pose."""


@dataclass
class Args:
    # --- policy ---
    task: str = "cloth"
    """cloth | pingpong | cup_pingpong | push_cube. Picks the checkpoint and prompt, plus the demo
    directory used for the start pose (~/yam_data/<task>, except push_cube -> push_cube_shovel_correct).
    push_cube is a left-arm task: in every demo the right arm sits at the folded rest pose."""
    model: Model = "pi"
    """pi | steeract_enc (PaliGemma D5) | steeract_dec (expert D5) | drvla | coast."""
    coast_beta: float = 1.0
    """COAST strength in [0, 1]. 1 = released baseline; smaller values reduce the intervention.
    This changes the model variant, not the 0.8-rad start guard. Saved in episode metadata."""
    model_root: str = str(RELEASE_ROOT)
    """Downloaded release root; contains task directories and deployment/."""
    checkpoint: Optional[str] = None
    """Override the selected method's pretrained_model directory."""
    prompt: Optional[str] = None
    """Task string given to the model. Default: --task, as in plain inference."""
    pi05_python: str = str(DEFAULT_PYTHON)
    device: str = "cuda"
    num_inference_steps: Optional[int] = None
    """Flow steps. Default: the checkpoint's 10 (~130 ms a plan on the 5090)."""
    seed: Optional[int] = None
    steer: bool = False
    """Backward-compatible alias for --model steeract_enc."""

    # --- control ---
    execute: bool = False
    """Send commands to the arms. Off by default: plans, records, moves nothing."""
    fps: int = 30
    """Control, recording and chunk rate. The checkpoints were trained on 30 fps data; keep it."""
    replan_every: int = 10
    """Ticks between plan requests (the checkpoint's n_action_steps). A plan is ~4-5 ticks old on
    arrival, so 10 executes ~5 fresh steps of each 50-step chunk; the horizon leaves room for a slow plan."""
    blend_steps: int = 5
    """Linear cross-fade from the old chunk to the new one over this many ticks, so a replan never
    steps the command. 0 = switch hard."""
    max_seconds: float = 120.0
    max_plan_age: float = 3.0
    """Stop if the active chunk gets this old (s); its own horizon is 50/30 = 1.67 s."""
    max_joint_vel: float = 2.0
    """Slew limit per arm joint, rad/s. The demos' leader commands: p99 1.2 / p99.9 1.9 rad/s on cloth,
    p99 0.66 / p99.9 1.1 on pingpong, so 2.0 only bites on moves faster than the operator ever made."""
    max_gripper_vel: float = 6.0
    """Normalised gripper units/s. Cloth demos close at p99.9 ~6.5/s."""
    max_start_error: float = 0.8
    """Refuse to engage if the first target is farther than this (rad) from the measured pose."""
    max_plan_switch_jump: float = 0.4
    """For single-arm tasks, reject a fresh plan when its active-arm target differs from the old plan's
    target at the same instant by more than this many radians. The old plan remains active and its
    existing --max-plan-age timeout stops execution if safe replanning does not recover. 0 disables."""

    # --- episode start ---
    goto_start: bool = True
    """With --execute: before the loop, slew to the task's start pose. Most tasks use the median
    demo frame-0 pose; cup_pingpong uses its manually selected pose for both arms."""
    start_pose_from: str = "~/yam_data"
    """Demo root holding <task>/episode_*/low_dim.npz for --goto-start."""
    start_forward: Optional[float] = None
    """Move the task start pose's active-arm TCP (gripper mount / tcp_site) this many metres forward,
    +x of its arm base frame, toward the table. push_cube adjusts only the left arm; other tasks adjust
    both arms. Default per task (START_ADJUST): cloth 0.08 -- its demo median sits pulled back, TCP at
    x ~0.10 m, about as far forward as the folded rest pose, while the demos average ~0.24 m mid-task;
    pingpong and push_cube 0."""
    start_down: Optional[float] = None
    """Move the TCP this many metres down, -z of the base frame. Default per task: pingpong 0.05, cloth 0."""
    start_tilt_to_base_deg: Optional[float] = None
    """Rotate the gripper this many degrees toward the arm base, about the TCP: its pointing direction
    (TCP -> grasp_site, the mount's -z) turns toward the line from the TCP to the base origin, so a
    gripper pointing forward over the table curls its tip down and back. Roll about the pointing
    direction is unchanged. Default per task: pingpong 30, cloth 0.

    All three adjust the task's selected start pose (IK on the composed arm + gripper model,
    everything not asked for held fixed); all 0 = that median unchanged. Each of them also moves the
    policy's first state off its training data -- watch the [engage] start error."""
    start_speed_scale: float = 0.25
    """Fraction of the slew limits used for that move."""
    start_seconds: float = 15.0
    wait_for_start: bool = True
    """Wait for SPACE / Enter before the first episode. After q / timeout, always wait for a fresh
    Enter, even with --no-wait-for-start. q prepares again; Esc exits."""

    # --- episode end ---
    release_grippers: bool = True
    release_seconds: float = 3.0
    park_on_exit: bool = True
    """On Esc: park before power-off. q / timeout always folds and returns to the task ready pose."""
    park_pose: Literal["folded", "working", "start"] = "folded"
    """folded = the rest pose an unpowered arm sits in (via the working pose); working = stop up and
    clear of the table; start = where the arms were when the script connected."""
    park_speed_scale: float = 0.2
    park_seconds: float = 15.0
    label_outcome: bool = True
    """Require s / f for the saved episode before allowing the next Enter. Also ask after exit.
    --no-label-outcome allows starting the next episode without a label."""

    # --- recording ---
    record: bool = True
    save_root: str = "~/yam_eval"
    """Episodes are numbered under <save_root>/steering_<task>/<model>/episode_NNNN/."""
    run_name: Optional[str] = None
    """Optional label saved in meta.json; directories always use episode_NNNN numbering."""

    # --- robot ---
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = channel_for("follower_left")
    can_follower_right: str = channel_for("follower_right")
    port_left: int = 1235
    port_right: int = 1234
    launch: bool = True
    """Spawn the follower processes. --no-launch attaches to scripts/run_policy_followers.sh."""
    sim: bool = False

    # --- cameras (recorder fields and defaults = demo geometry) ---
    top_serial: Optional[str] = None
    left_wrist_serial: Optional[str] = None
    right_wrist_serial: Optional[str] = None
    swap_wrists: bool = False
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    top_width: int = 1920
    top_height: int = 1080
    top_fps: int = 30
    allow_missing_cameras: bool = False
    """Black frames for missing views. Plumbing tests only -- the policy is blind there."""

    # --- operator IO ---
    display: bool = True
    global_keys: bool = True
    """Also take q / Esc from anywhere on the desktop (pynput, X11). Starting is never global: a space
    typed into another window must not start the robot."""
    tts: bool = True
    """Announce ready / stopping over spd-say."""
    display_width: int = 1280
    display_every: int = 3
    """Redraw the preview every N ticks; drawing 1080p every tick costs control time."""
    log_every: int = 30


@dataclass
class Plan:
    chunk: np.ndarray  # [T, 14] raw absolute joint targets, step k = t0 + k/fps
    t0: float
    latency_s: float
    infer_s: float
    index: int

    def target(self, now: float, dt: float) -> np.ndarray:
        u = float(np.clip((now - self.t0) / dt, 0.0, self.chunk.shape[0] - 1))
        lo = int(u)
        hi = min(lo + 1, self.chunk.shape[0] - 1)
        w = u - lo
        return (1.0 - w) * self.chunk[lo] + w * self.chunk[hi]


def _pin_sides(values: np.ndarray, held_targets: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a copy with inactive arms replaced by their canonical demo targets."""
    pinned = np.asarray(values).copy()
    for side, held in held_targets.items():
        i = SIDES.index(side)
        pinned[i * _NDOF : (i + 1) * _NDOF] = held
    return pinned


class TickWriter(rec.EpisodeWriter):
    """EpisodeWriter fed from a background thread, with each row stamped at its control tick."""

    def __init__(self, ep_dir: Path, fps: int) -> None:
        super().__init__(ep_dir, fps)
        self._q: "queue.Queue[Optional[Tuple[float, float, Dict[str, Any], Dict[str, Tuple[np.ndarray, float]]]]]" = (
            queue.Queue()
        )
        self.max_backlog = 0
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._thread.start()

    def put(
        self, t_mono: float, t_wall: float, sample: Dict[str, Any], frames: Dict[str, Tuple[np.ndarray, float]]
    ) -> None:
        if self.error is not None:
            raise RuntimeError(f"episode writer failed: {self.error}")
        self._q.put((t_mono, t_wall, sample, frames))
        self.max_backlog = max(self.max_backlog, self._q.qsize())

    @property
    def backlog(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if self.error is not None:
                continue
            t_mono, t_wall, sample, frames = item
            try:
                self.add(sample, frames)
                self._low["t_mono"][-1] = t_mono - self._t0
                self._low["t_wall"][-1] = t_wall
            except BaseException as e:
                self.error = e

    def finish(self) -> None:
        """Drain the queue (every tick recorded gets written) and stop the thread."""
        self._q.put(None)
        self._thread.join()
        if self.error is not None:
            raise RuntimeError(f"episode writer failed: {self.error}")


class KeyWatcher:
    """Single keypresses, no Enter needed, from wherever the operator's hands are.

    * the **terminal** the run was started in, in cbreak mode (ISIG untouched, so Ctrl-C still works)
    * the **preview window**, via :meth:`feed` with the ``cv2.waitKey`` code
    * the **desktop**, via pynput -- q / Esc only, so a stop works with the focus anywhere but a
      space typed into some other window can never start the robot

    Keys queue up and :meth:`pop` hands them out; :meth:`drain` drops what was typed before a prompt.
    """

    def __init__(self, global_keys: bool) -> None:
        self._keys: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self.exit_requested = threading.Event()
        self._stopping = threading.Event()
        self._fd: Optional[int] = None
        self._saved: Optional[Any] = None
        self._listener: Optional[Any] = None
        self.sources: List[str] = []
        self._start_terminal()
        if global_keys:
            self._start_global()

    def _start_terminal(self) -> None:
        try:
            if not sys.stdin.isatty():
                return
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception as e:
            logging.debug(f"terminal key source unavailable: {e}")
            self._fd, self._saved = None, None
            return
        threading.Thread(target=self._read_terminal, name="keys-stdin", daemon=True).start()
        self.sources.append("terminal")

    def _read_terminal(self) -> None:
        while not self._stopping.is_set():
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if not ch:
                        return
                    self._push(ch.lower())
            except Exception:
                return

    def _start_global(self) -> None:
        try:
            from pynput import keyboard

            def on_press(key: Any) -> None:
                char = getattr(key, "char", None)
                if key == keyboard.Key.esc:
                    self._push("\x1b")
                elif char and char.lower() == "q":
                    self._push("q")

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            self.sources.append("desktop (q/Esc)")
        except Exception as e:
            logging.debug(f"global key source unavailable: {e}")
            self._listener = None

    def feed(self, key: int) -> None:
        """A key code from ``cv2.waitKey`` (255 = none)."""
        if key in (255, 0, -1):
            return
        self._push({13: "\n", 27: "\x1b"}.get(key, chr(key).lower() if key < 128 else ""))

    def _push(self, key: str) -> None:
        if key in _QUIT_KEYS:
            self.exit_requested.set()
        self._keys.put(key)

    def pop(self) -> Optional[str]:
        if self.exit_requested.is_set():
            return "\x1b"
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def drain(self, *, preserve_labels: bool = False) -> None:
        # Discard duplicate q / early Enter events, but retain labels typed during the return.
        retained = []
        while True:
            try:
                key = self._keys.get_nowait()
                if preserve_labels and key in ("s", "f"):
                    retained.append(key)
            except queue.Empty:
                break
        for key in retained:
            self._keys.put(key)

    def close(self) -> None:
        self._stopping.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._fd is not None and self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._fd, self._saved = None, None


def demo_start_pose(root: Path, task: str) -> Tuple[Optional[np.ndarray], int]:
    """Median first-frame joint pose [14] over the complete, non-failed demos of ``task``."""
    firsts = []
    for ep in sorted((root / task).glob("episode_*")):
        if ep.name.endswith(rec.FAILED_SUFFIX) or not (ep / "meta.json").is_file():
            continue
        low = np.load(ep / "low_dim.npz")
        firsts.append(np.concatenate([low["joint_pos_left"][0], low["joint_pos_right"][0]]))
    if not firsts:
        return None, 0
    return np.median(np.stack(firsts), axis=0), len(firsts)


def adjust_pose(
    q: np.ndarray,
    *,
    forward: float,
    down: float,
    tilt_to_base_deg: float,
    arm: str,
    version: int,
    gripper: str,
    joint_range: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """``q`` (arm joints + gripper) with the TCP moved and the gripper tilted toward the base.

    TCP = the ``gripper`` mount body (``tcp_site`` sits on its origin), in the arm's base frame.
    Target position: ``forward`` m along +x and ``down`` m along -z. Target orientation: the start
    orientation rotated ``tilt_to_base_deg`` about the TCP, about the axis ``d x v`` -- ``d`` the
    pointing direction (TCP -> ``grasp_site``), ``v`` the direction from the target TCP to the base
    origin -- which turns ``d`` toward ``v`` and leaves roll about ``d`` alone.

    The full 6-D pose is the IK target, so the 6-joint arm has a single local solution (joint 1
    turns too when the gripper sits off-centre). Damped least squares on the composed arm + gripper
    MJCF, which is the model that has the grasp point; clamped to the joint limits; gripper value kept.
    Returns ``(q, position error m, rotation error rad)`` so the caller can refuse an unreachable pose.
    """
    import mujoco

    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    xml = combine_arm_and_gripper_xml(
        ArmType.from_string_name(arm), GripperType.from_string_name(gripper), version=version
    )
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    body = model.body("gripper").id
    grasp = model.site("grasp_site").id
    n = joint_range.shape[0]

    def pose(joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data.qpos[:] = 0.0
        data.qpos[:n] = joints
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return data.xpos[body].copy(), data.xquat[body].copy()

    def rot_error(quat_target: np.ndarray, quat: np.ndarray) -> np.ndarray:
        inv, delta, vel = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(inv, quat)
        mujoco.mju_mulQuat(delta, quat_target, inv)
        mujoco.mju_quat2Vel(vel, delta, 1.0)
        return vel

    arm_q = np.asarray(q[:n], dtype=np.float64).copy()
    pos0, quat0 = pose(arm_q)
    target_pos = pos0 + np.array([float(forward), 0.0, -float(down)])
    target_quat = quat0.copy()
    if tilt_to_base_deg:
        pointing = data.site_xpos[grasp] - pos0
        to_base = -target_pos
        axis = np.cross(pointing, to_base)
        if np.linalg.norm(axis) < 1e-9:
            raise ValueError("the gripper already points straight at the base; the tilt axis is undefined")
        tilt = np.zeros(4)
        mujoco.mju_axisAngle2Quat(tilt, axis / np.linalg.norm(axis), np.radians(float(tilt_to_base_deg)))
        mujoco.mju_mulQuat(target_quat, tilt, quat0)

    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(500):
        pos, quat = pose(arm_q)
        err = np.concatenate([target_pos - pos, rot_error(target_quat, quat)])
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            break
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, body)
        jac = np.vstack([jac_pos[:, :n], jac_rot[:, :n]])
        arm_q = np.clip(
            arm_q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-2 * np.eye(6), err), joint_range[:, 0], joint_range[:, 1]
        )
    pos, quat = pose(arm_q)
    out = np.asarray(q, dtype=np.float64).copy()
    out[:n] = arm_q
    return out, float(np.linalg.norm(target_pos - pos)), float(np.linalg.norm(rot_error(target_quat, quat)))


def _grab(
    rig: rec.CameraRig, last: Dict[str, Tuple[np.ndarray, float]], args: Args
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Raw BGR frame + device timestamp per view; a view with no new frame repeats its last one."""
    snap = rig.snapshot()
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    for view in VIEWS:
        if view in snap:
            last[view] = snap[view]
        if view in last:
            out[view] = last[view]
        elif args.allow_missing_cameras or args.sim:
            hw = (args.top_height, args.top_width) if view == "top" else (args.cam_height, args.cam_width)
            out[view] = (np.zeros((*hw, 3), np.uint8), 0.0)
        else:
            raise RuntimeError(f"camera '{view}' has delivered no frame")
    return out


def _check_camera_geometry(rig: rec.CameraRig, image_hw: Dict[str, Any]) -> None:
    """The negotiated camera profile must match what the checkpoint was trained on."""
    for view, cam in rig.cams_by_role.items():
        if not cam.get("color") or view not in image_hw:
            continue
        w, h = int(cam["color"][0]), int(cam["color"][1])
        if (h, w) != tuple(image_hw[view]):
            raise RuntimeError(
                f"camera {view} opened at {w}x{h}, the checkpoint expects {image_hw[view][1]}x{image_hw[view][0]} "
                "-- give it a USB3 port with enough bandwidth"
            )


def _sample(
    kin: rec.EEFKinematics,
    pos: Dict[str, np.ndarray],
    cmd: Dict[str, np.ndarray],
    engaged: bool,
    target: Optional[np.ndarray],
    active: Optional[Plan],
    plan_rejected: bool,
    now: float,
) -> Dict[str, Any]:
    """One low_dim row, with the recorder's keys for the recorder's quantities."""
    g = kin.n_arm_joints
    row: Dict[str, Any] = {}
    for i, side in enumerate(SIDES):
        p, c = pos[side], cmd[side]
        eef_pos, eef_quat = kin.fk(p)
        a_pos, a_quat = kin.fk(c)
        row[f"joint_pos_{side}"] = p
        row[f"eef_pos_{side}"] = eef_pos
        row[f"eef_quat_{side}"] = eef_quat
        row[f"gripper_{side}"] = p[g]
        row[f"action_joint_pos_{side}"] = c
        row[f"action_eef_pos_{side}"] = a_pos
        row[f"action_eef_quat_{side}"] = a_quat
        row[f"action_eef_delta_{side}"] = rec.EEFKinematics.pose_delta(a_pos, a_quat, eef_pos, eef_quat)
        row[f"action_gripper_{side}"] = c[g]
        row[f"engaged_{side}"] = bool(engaged)
        row[f"policy_joint_pos_{side}"] = (
            np.full(_NDOF, np.nan) if target is None else np.asarray(target[i * _NDOF : (i + 1) * _NDOF], np.float64)
        )
    row["plan_age"] = float(now - active.t0) if active is not None else np.nan
    row["plan_latency"] = float(active.latency_s) if active is not None else np.nan
    row["plan_infer"] = float(active.infer_s) if active is not None else np.nan
    row["plan_index"] = int(active.index) if active is not None else -1
    row["plan_rejected"] = bool(plan_rejected)
    return row


def _read_line(shutdown: rec.ShutdownRequest, allowed: Tuple[str, ...]) -> Optional[str]:
    """One line from a cooked terminal, polling the Ctrl-C flag (a plain input() would not see it)."""
    if not sys.stdin.isatty():
        return None
    while True:
        shutdown.raise_if_requested()
        if select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline().strip().lower()
            if line in allowed:
                return line
            print(f"  (type one of {allowed!r} and Enter)")


def _relabel(ep_dir: Path, outcome: Optional[str]) -> Path:
    """Write the operator's verdict into an already-saved episode; a failure gets the _failed suffix."""
    meta_path = ep_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["outcome"] = outcome
    meta["failed"] = outcome == "failure"
    meta_path.write_text(json.dumps(meta, indent=2))
    if outcome == "failure" and not ep_dir.name.endswith(rec.FAILED_SUFFIX):
        failed_dir = ep_dir.with_name(f"{ep_dir.name}{rec.FAILED_SUFFIX}")
        ep_dir.rename(failed_dir)
        return failed_dir
    return ep_dir


def _preview(frames: Dict[str, Tuple[np.ndarray, float]], lines: List[str], width: int, color: Tuple[int, ...]) -> int:
    row = rec._build_camera_row(frames, width)
    panel = np.zeros((30 * len(lines) + 12, row.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 26 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.imshow(_WINDOW, np.vstack([panel, row]))
    return cv2.waitKey(1) & 0xFF


def _wait_for_start(
    keys: KeyWatcher,
    rig: rec.CameraRig,
    last_frames: Dict[str, Tuple[np.ndarray, float]],
    args: Args,
    shutdown: rec.ShutdownRequest,
    speaker: rec.Speaker,
    *,
    require_enter: bool = False,
    label_previous: Optional[Callable[[str], None]] = None,
) -> Literal["start", "retry", "exit"]:
    """After q, require a fresh Enter even when --no-wait-for-start was specified."""
    can_start = "terminal" in keys.sources or args.display
    if not can_start and not require_enter and label_previous is None:
        print("[start] no terminal and no preview window to press a key in -- starting right away")
        return "start"
    where = " / ".join(
        s for s in ("terminal" if "terminal" in keys.sources else "", "preview window" if args.display else "") if s
    )
    label_pending = label_previous is not None
    if label_pending:
        keys.drain(preserve_labels=True)
        speaker.say("label the last episode. s for success, f for failure")
    else:
        keys.drain()
        speaker.say("ready. press enter to start")
    label_status = ""
    last_prompt = -1e9
    while True:
        shutdown.raise_if_requested()
        now = time.monotonic()
        if now - last_prompt > _PROMPT_REPEAT_S:
            instruction = (
                "[label] LABEL PREVIOUS EPISODE: s = SUCCESS / f = FAILURE (no Enter needed). "
                "Enter cannot start until labeled."
                if label_pending
                else f"[start] READY {label_status}-- reset the scene, then press Enter to start."
            )
            print(
                f"\n{'=' * 78}\n{instruction} ({where or 'no start input'})\n"
                f"[start] q = fold and prepare again; Esc = fold and exit.\n{'=' * 78}",
                flush=True,
            )
            last_prompt = now
        key = keys.pop()
        if key in (("\n", "\r") if require_enter else _START_KEYS):
            if not label_pending:
                return "start"
            print("[label] Please press s (success) or f (failure) before starting the next episode.", flush=True)
            last_prompt = -1e9
        if key == "q":
            return "retry"
        if key in _QUIT_KEYS:
            return "exit"
        if label_pending and label_previous is not None and key in ("s", "f"):
            outcome = {"s": "success", "f": "failure"}[key]
            label_previous(outcome)
            label_pending = False
            label_status = f"(previous: {outcome.upper()}) "
            last_prompt = -1e9
            speaker.say(f"{outcome} saved. press enter for the next episode")
        if args.display:
            policy_name = f"{args.task} / {args.model}"
            lines = (
                [f"{policy_name}: LABEL PREVIOUS EPISODE", "s = SUCCESS    f = FAILURE    Esc = exit"]
                if label_pending
                else [f"{policy_name}: READY {label_status}", "Enter = start    q = prepare again    Esc = exit"]
            )
            keys.feed(_preview(_grab(rig, last_frames, args), lines, args.display_width, (0, 255, 255)))
        time.sleep(1.0 / 15.0)


def _return_to_ready(
    arms: Dict[str, FollowerArm],
    limiters: Dict[str, SafetyLimiter],
    ready: Dict[str, np.ndarray],
    args: Args,
    keep_moving: Callable[[], bool],
) -> bool:
    """Open, fold via the existing working waypoint, then reach the task pose without power-off."""
    if not args.execute:
        print("[retry] dry run -- no movement; waiting for the next Enter")
        return keep_moving()
    slow = {side: limiter.scaled(args.park_speed_scale) for side, limiter in limiters.items()}
    for side, arm in arms.items():
        slow[side].reset(arm.joint_pos())
    release = {side: limiter.last.copy() for side, limiter in slow.items()}
    for target in release.values():
        target[-1] = GRIPPER_OPEN
    phases = []
    if args.release_grippers:
        phases.append(("open grippers", release, args.release_seconds, args.park_speed_scale))
    for label, pose in (("working pose", WORKING_Q), ("folded rest pose", FOLDED_Q)):
        targets = {side: np.asarray(pose, dtype=np.float64)[:_NDOF].copy() for side in arms}
        for side, target in targets.items():
            target[-1] = GRIPPER_OPEN if args.release_grippers else slow[side].last[-1]
        phases.append((label, targets, args.park_seconds, args.park_speed_scale))
    phases.append(("task ready pose", ready, args.start_seconds, args.start_speed_scale))
    dt = 1.0 / args.fps
    try:
        for label, targets, seconds, scale in phases:
            print(f"[retry] moving to {label}", flush=True)
            for side in arms:
                slowed = limiters[side].scaled(scale)
                slowed.reset(slow[side].last)
                slow[side] = slowed
            deadline = time.monotonic() + seconds
            while True:
                if not keep_moving():
                    return False
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Did not reach {label}; refusing to start another episode")
                reached = True
                for side, arm in arms.items():
                    command = slow[side].step(targets[side], dt)
                    arm.command(command)
                    if np.max(np.abs(arm.joint_pos() - targets[side])) > 0.05:
                        reached = False
                    if np.max(np.abs(command - targets[side])) > 1e-3:
                        reached = False
                if reached:
                    break
                time.sleep(dt)
    finally:
        # Keep exit and subsequent policy commands continuous with the last reset command.
        for side in arms:
            limiters[side].reset(slow[side].last)
    return True


def run(args: Args, ckpt: Path) -> None:
    procs: List["subprocess.Popen[bytes]"] = []
    shutdown = rec.ShutdownRequest(procs)
    speaker = rec.Speaker(args.tts)
    policy: Optional[PolicyClient] = None
    rig: Optional[rec.CameraRig] = None
    arms: Dict[str, FollowerArm] = {}
    writer: Optional[TickWriter] = None
    keys: Optional[KeyWatcher] = None
    saved_dir: Optional[Path] = None
    saved_outcome: Optional[str] = None
    stop_reason = "startup"
    start_pose: Optional[np.ndarray] = None
    meta_base: Dict[str, Any] = {}
    latencies: List[float] = []
    infers: List[float] = []
    switch_jumps: List[float] = []
    rejected_plan_jumps: List[float] = []
    overruns = 0
    tick = 0
    dt = 1.0 / float(args.fps)

    demos = demo_dir(args.task)
    demo_pose: Optional[np.ndarray] = None
    n_demos = 0
    if args.execute or args.task in POLICY_ACTIVE_SIDES:
        demo_pose, n_demos = demo_start_pose(Path(args.start_pose_from).expanduser(), demos)
    held_sides = tuple(side for side in SIDES if side not in POLICY_ACTIVE_SIDES.get(args.task, SIDES))
    held_targets: Dict[str, np.ndarray] = {}
    if held_sides:
        if demo_pose is None:
            raise RuntimeError(
                f"{args.task} needs its demo start pose to canonicalize and hold {', '.join(held_sides)}, "
                f"but no demos were found under {args.start_pose_from}/{demos}"
            )
        held_targets = {
            side: demo_pose[SIDES.index(side) * _NDOF : (SIDES.index(side) + 1) * _NDOF].astype(np.float32).copy()
            for side in held_sides
        }
    active_arm_indices = np.array(
        [
            SIDES.index(side) * _NDOF + joint
            for side in POLICY_ACTIVE_SIDES.get(args.task, SIDES)
            for joint in range(6)
        ],
        dtype=np.int64,
    )

    def save_recording() -> Optional[Path]:
        """Drain the writer thread and finalise the episode on disk. Clears ``writer``."""
        nonlocal writer, saved_outcome
        w, writer = writer, None
        if w is None:
            return None
        w.finish()
        if not w.n_frames:
            w.discard()
            print("[record] nothing recorded -- discarded")
            return None
        meta = dict(
            meta_base,
            stop_reason=stop_reason,
            outcome=None,
            executed=bool(args.execute),
            plans=policy.n_plans if policy is not None else 0,
            ticks=tick,
            tick_overruns=overruns,
            writer_max_backlog=w.max_backlog,
            start_pose=None if start_pose is None else start_pose.tolist(),
            infer_ms=None
            if not infers
            else {"median": 1e3 * float(np.median(infers)), "max": 1e3 * float(np.max(infers))},
            latency_ms=None
            if not latencies
            else {"median": 1e3 * float(np.median(latencies)), "max": 1e3 * float(np.max(latencies))},
            switch_jump_rad=None
            if not switch_jumps
            else {"median": float(np.median(switch_jumps)), "max": float(np.max(switch_jumps))},
            rejected_plans={
                "count": len(rejected_plan_jumps),
                "max_jump_rad": None if not rejected_plan_jumps else float(np.max(rejected_plan_jumps)),
            },
            steering_telemetry=policy.steering_telemetry if policy is not None else [],
            args={
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in asdict(args).items()
            },
        )
        path = w.save(meta)
        saved_outcome = None
        print(f"[record] saved {path} ({w.n_frames} frames)", flush=True)
        return path

    try:
        # The model first: a missing checkpoint or a broken env fails before any arm is powered.
        prompt = args.prompt or args.task
        assets = resolve_assets(args.task, args.model, Path(args.model_root), str(ckpt))
        mode = args.model
        print(f"[policy] starting the pi05 server ({ckpt}, prompt {prompt!r}, {mode}) ...")
        policy = PolicyClient(
            ckpt,
            prompt,
            python=Path(args.pi05_python),
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            model=args.model,
            coast_beta=args.coast_beta,
            steer_deployment_root=assets.root / "deployment",
            steer_sae_root=assets.sae,
            steer_method_root=assets.method,
            selection=assets.selection if args.model in ("drvla", "coast") else None,
        )
        info = policy.start()
        print(
            f"[policy] ready: chunk {info['chunk_size']} @ {args.fps} fps, {info['num_inference_steps']} flow steps, "
            f"views {info['image_hw']}"
        )

        if args.launch:
            _launch_followers(args, procs, shutdown)
        else:
            _check_attached_followers(args, {"left": args.port_left, "right": args.port_right})
        arms = {side: FollowerArm(side, port) for side, port in (("left", args.port_left), ("right", args.port_right))}
        for side, arm in arms.items():
            if arm.num_dofs != _NDOF:
                raise RuntimeError(f"{side} follower has {arm.num_dofs} dofs, the policy needs {_NDOF}")
        kin = rec.EEFKinematics(args.arm, args.version)
        limits = _joint_range(args.arm, args.version)
        limiters = {s: SafetyLimiter(limits, args.max_joint_vel, args.max_gripper_vel) for s in SIDES}

        rig = rec._open_camera_rig(args)
        deadline = time.monotonic() + 10.0
        while rig.cams_by_role and not rig.all_fresh() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not rig.all_fresh() and not (args.allow_missing_cameras or args.sim):
            raise RuntimeError("cameras did not deliver frames (check USB bandwidth and cables)")
        _check_camera_geometry(rig, info["image_hw"])

        meta_base = {
            "run": "steering_policy_eval",
            "task": args.task,
            "model": args.model,
            "model_root": str(assets.root),
            "prompt": prompt,
            "checkpoint": str(ckpt),
            "steer": args.model != "pi",
            "policy_info": info,
            "arm": args.arm,
            "version": args.version,
            "cameras": {role: cam["serial"] for role, cam in rig.cams_by_role.items()},
            "camera_color_profiles": {
                role: (f"{cam['color'][0]}x{cam['color'][1]}@{cam['color'][2]}" if cam.get("color") else None)
                for role, cam in rig.cams_by_role.items()
            },
            "arms": {
                "left": {"follower_can": args.can_follower_left},
                "right": {"follower_can": args.can_follower_right},
            },
            "policy_active_sides": list(POLICY_ACTIVE_SIDES.get(args.task, SIDES)),
            "held_targets": {side: target.tolist() for side, target in held_targets.items()},
            "policy_state_overrides": {side: target.tolist() for side, target in held_targets.items()},
            "action_source": "policy chunk interpolated at the tick, after cross-fade and the safety limiter; "
            "measured pos before the policy engages; inactive single-task arms hold their demo start pose",
            "low_dim_keys": {
                "joint_pos_<side>": "measured joint positions, 6 arm joints + gripper (normalized 0-1)",
                "eef_pos_<side> / eef_quat_<side>": "FK of measured arm joints: terminal mount pose in the "
                "arm's base frame, quat wxyz",
                "gripper_<side>": "measured gripper opening, normalized",
                "action_joint_pos_<side>": "joint-space command streamed to the follower (absolute); measured pos "
                "while not engaged",
                "action_eef_pos_<side> / action_eef_quat_<side>": "FK of the command",
                "action_eef_delta_<side>": "commanded pose relative to measured pose: base-frame dpos (3) + "
                "base-frame axis-angle drot (3)",
                "action_gripper_<side>": "gripper command, normalized",
                "engaged_<side>": "True when the policy is driving (commands sent only with --execute)",
                "policy_joint_pos_<side>": "raw chunk target at the tick, before the cross-fade limiter (NaN before "
                "the first plan)",
                "plan_age / plan_latency / plan_infer / plan_index": "active chunk: age at the tick, observation-to-"
                "arrival, server inference time (s), plan counter",
                "plan_rejected": "a fresh plan exceeded --max-plan-switch-jump on an active arm and was ignored",
                "t_mono / t_wall / cam_t_<role>": "tick timestamps (s since episode start / unix s / camera ms)",
            },
            "sim": args.sim,
        }

        # From here on a key press is queued even while the start slew blocks.
        keys = KeyWatcher(global_keys=args.global_keys)
        if args.display:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        last_frames: Dict[str, Tuple[np.ndarray, float]] = {}

        # ---- episode start ----------------------------------------------------------------
        home = {side: arms[side].joint_pos() for side in SIDES}
        start_pose = np.concatenate([home[side][:_NDOF] for side in SIDES])
        for side in SIDES:
            limiters[side].reset(home[side])
        if args.execute:
            if demo_pose is not None or args.task in MANUAL_START_POSES:
                start_pose = start_pose if demo_pose is None else demo_pose.copy()
                print(f"[start] median frame 0 pose of {n_demos} demos in {demos}: {np.round(start_pose, 2)}")
                manual_poses = MANUAL_START_POSES.get(args.task, {})
                for side, pose in manual_poses.items():
                    i = SIDES.index(side)
                    start_pose[i * _NDOF : (i + 1) * _NDOF] = np.asarray(pose, dtype=np.float64)[:_NDOF]
                if manual_poses:
                    print(f"[start] manual {', '.join(manual_poses)} pose for {args.task}: {np.round(start_pose, 2)}")
                if args.start_forward or args.start_down or args.start_tilt_to_base_deg:
                    what = (
                        f"{args.start_forward * 100:g} cm forward, {args.start_down * 100:g} cm down, "
                        f"tilted {args.start_tilt_to_base_deg:g} deg toward the base"
                    )
                    adjusted = start_pose.copy()
                    ok = True
                    adjust_sides = START_ADJUST_SIDES.get(args.task, SIDES)
                    for i, side in enumerate(SIDES):
                        if side not in adjust_sides:
                            continue
                        q, pos_err, rot_err = adjust_pose(
                            start_pose[i * _NDOF : (i + 1) * _NDOF],
                            forward=args.start_forward,
                            down=args.start_down,
                            tilt_to_base_deg=args.start_tilt_to_base_deg,
                            arm=args.arm,
                            version=args.version,
                            gripper=args.follower_gripper,
                            joint_range=limits,
                        )
                        if pos_err > 5e-3 or rot_err > np.radians(2.0):
                            print(
                                f"[warn] {side}: {what} is not reachable (off by {pos_err * 1e3:.1f} mm, "
                                f"{np.degrees(rot_err):.1f} deg)"
                            )
                            ok = False
                        adjusted[i * _NDOF : (i + 1) * _NDOF] = q
                    if ok:
                        start_pose = adjusted
                        print(f"[start] adjusted {', '.join(adjust_sides)}: {what}: {np.round(start_pose, 2)}")
                    else:
                        print("[warn] keeping the unadjusted demo start pose -- reduce the --start-* adjustment")
                if args.goto_start:
                    print(f"[start] slewing to the start pose at {args.start_speed_scale:g}x the slew limit ...")
                    slow = {side: limiters[side].scaled(args.start_speed_scale) for side in SIDES}
                    targets = {"left": start_pose[:_NDOF], "right": start_pose[_NDOF:]}
                    if not _slew_to(arms, slow, targets, seconds=args.start_seconds, send_hz=args.fps):
                        print("[warn] did not reach the start pose in time -- continuing from where the arms are")
                    for side in SIDES:
                        limiters[side].last = slow[side].last.copy()
            else:
                print(f"[start] no task pose under {args.start_pose_from}/{demos}; using the initial measured pose")

        if held_targets:
            refs = ", ".join(f"{side}={np.round(target, 3)}" for side, target in held_targets.items())
            print(
                f"[safety] {args.task}: policy state and commands pinned to demo pose for {refs}; "
                f"reject active-arm plan switches > {args.max_plan_switch_jump:g} rad"
            )

        def label_previous(outcome: str) -> None:
            nonlocal saved_dir, saved_outcome
            if saved_dir is not None:
                saved_dir = _relabel(saved_dir, outcome)
                saved_outcome = outcome
                print(f"[label] {outcome} -> {saved_dir}")

        def keep_resetting() -> bool:
            shutdown.raise_if_requested()
            if args.display:
                keys.feed(
                    _preview(
                        _grab(rig, last_frames, args),
                        ["Returning to task ready pose -- Esc = exit"]
                        + (["After return: s = success / f = failure, then Enter"] if args.label_outcome else []),
                        args.display_width,
                        (0, 255, 255),
                    )
                )
            return not keys.exit_requested.is_set()

        ready = {"left": start_pose[:_NDOF].copy(), "right": start_pose[_NDOF:].copy()}
        episode_number = 0
        while True:
            tick = overruns = 0
            latencies.clear()
            infers.clear()
            switch_jumps.clear()
            rejected_plan_jumps.clear()
            meta_base["episode_number"] = episode_number
            for side in SIDES:
                limiters[side].reset(arms[side].joint_pos())
            started = "start"
            if args.wait_for_start or episode_number > 0:
                started = _wait_for_start(
                    keys,
                    rig,
                    last_frames,
                    args,
                    shutdown,
                    speaker,
                    require_enter=episode_number > 0,
                    label_previous=label_previous
                    if saved_dir is not None and saved_outcome is None and args.label_outcome
                    else None,
                )
            if started == "exit":
                stop_reason = "operator quit before start (Esc)"
            elif started == "retry":
                stop_reason = "operator retry (q)"
            else:
                # ---- closed loop --------------------------------------------------------------
                print(
                    f"[mode] {args.model}; "
                    f"{'EXECUTE -- the arms will move' if args.execute else 'DRY RUN -- no commands sent'}; "
                    f"q = save, fold and prepare next episode; Esc = fold and exit; Ctrl-C = emergency hold; "
                    f"max {args.max_seconds:.0f}s",
                    flush=True,
                )
                speaker.say("start")
                if args.record:
                    writer = new_episode(
                        Path(args.save_root), args.task, args.model, lambda path: TickWriter(path, fps=args.fps)
                    )
                    meta_base["session_dir"] = str(writer.ep_dir)
                    print(f"[record] {writer.ep_dir}")

                timing = {k: Ema() for k in ("grab", "state", "send", "tick")}
                active: Optional[Plan] = None
                previous: Optional[Plan] = None
                switch_t = 0.0
                engaged = False
                last_request_tick = -(10**9)
                stop_reason = "max-seconds"
                start = time.monotonic()
                next_tick = start

                while True:
                    shutdown.raise_if_requested()
                    key = keys.pop()
                    if key == "q" or key in _QUIT_KEYS:
                        stop_reason = "operator retry (q)" if key == "q" else "operator quit (Esc)"
                        break
                    wait = next_tick - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    now = time.monotonic()
                    t_wall = time.time()
                    if now - start > args.max_seconds:
                        break
                    late = now - next_tick
                    next_tick += dt
                    if late > dt:  # a stalled tick: resync instead of firing a burst of catch-up ticks
                        overruns += 1
                        next_tick = now + dt
                    tick += 1

                    mark = time.perf_counter()
                    frames = _grab(rig, last_frames, args)
                    timing["grab"].add(time.perf_counter() - mark)
                    mark = time.perf_counter()
                    pos = {side: arms[side].joint_pos() for side in SIDES}
                    timing["state"].add(time.perf_counter() - mark)
                    state = np.concatenate([pos["left"][:_NDOF], pos["right"][:_NDOF]]).astype(np.float32)

                    # newest plan
                    plan_rejected = False
                    done = policy.poll()
                    if done is not None:
                        t0, chunk, latency, infer = done
                        latencies.append(latency)
                        infers.append(infer)
                        fresh = Plan(chunk=chunk, t0=t0, latency_s=latency, infer_s=infer, index=policy.n_plans)
                        if active is not None and engaged:
                            old, new = active.target(now, dt), fresh.target(now, dt)
                            jump = float(np.max(np.abs((new - old)[active_arm_indices])))
                            switch_jumps.append(jump)
                            if held_targets and args.max_plan_switch_jump > 0 and jump > args.max_plan_switch_jump:
                                plan_rejected = True
                                rejected_plan_jumps.append(jump)
                                print(
                                    f"[safety] rejected plan {fresh.index}: active-arm switch jump {jump:.3f} rad "
                                    f"> {args.max_plan_switch_jump:.3f}; keeping plan {active.index}"
                                )
                        if not plan_rejected:
                            if not engaged:
                                target = fresh.target(now, dt)
                                err = float(np.max(np.abs((target - state)[active_arm_indices])))
                                print(f"[engage] first active-arm target is {err:.3f} rad from the measured pose")
                                if err > args.max_start_error:
                                    raise RuntimeError(
                                        f"first target is {err:.3f} rad away (limit {args.max_start_error:.2f}); "
                                        "no policy action sent. Check observation/checkpoint alignment and steering "
                                        "strength with offline replay. "
                                        + (
                                            f"COAST beta={args.coast_beta:g}; --coast-beta controls intervention "
                                            "strength (1 is the published baseline)."
                                            if args.model == "coast"
                                            else "Start from a demo-like pose (--goto-start)."
                                        )
                                    )
                                engaged = True
                            previous, active, switch_t = active, fresh, now

                    # next request
                    if not policy.busy and (active is None or tick - last_request_tick >= args.replan_every):
                        images = {view: cv2.cvtColor(frames[view][0], cv2.COLOR_BGR2RGB) for view in VIEWS}
                        policy_state = _pin_sides(state, held_targets)
                        policy.request(now, policy_state, images)
                        last_request_tick = tick

                    # command
                    target: Optional[np.ndarray] = None
                    if active is not None and engaged:
                        age = now - active.t0
                        if age > args.max_plan_age:
                            stop_reason = f"no fresh plan for {age:.1f}s -- planning has stopped"
                            break
                        target = active.target(now, dt)
                        blended = target
                        if previous is not None and args.blend_steps > 0:
                            w = float(np.clip((now - switch_t) / (args.blend_steps * dt), 0.0, 1.0))
                            if w < 1.0:
                                blended = (1.0 - w) * previous.target(now, dt) + w * target
                        if held_targets:
                            blended = _pin_sides(blended, held_targets)
                        mark = time.perf_counter()
                        cmd = {}
                        for i, side in enumerate(SIDES):
                            cmd[side] = limiters[side].step(blended[i * _NDOF : (i + 1) * _NDOF], dt)
                            if args.execute:
                                arms[side].command(cmd[side])
                        timing["send"].add(time.perf_counter() - mark)
                    else:
                        cmd = {side: pos[side].copy() for side in SIDES}

                    if writer is not None:
                        writer.put(
                            now, t_wall, _sample(kin, pos, cmd, engaged, target, active, plan_rejected, now), frames
                        )

                    if args.display and tick % max(1, args.display_every) == 0:
                        plan_age = "--" if active is None else f"{now - active.t0:.2f}s"
                        policy_name = f"{args.task} / {args.model}"
                        lines = [
                            f"{policy_name}  t {now - start:5.1f}s  plans {policy.n_plans}  plan age {plan_age}  "
                            f"{'EXECUTE' if args.execute else 'DRY RUN'}{'  REC' if writer is not None else ''}   q: next episode   Esc: exit"
                        ]
                        keys.feed(_preview(frames, lines, args.display_width, (255, 255, 255)))

                    timing["tick"].add(time.monotonic() - now)
                    if tick % max(1, args.log_every) == 0:
                        inf = f"{np.median(infers[-10:]) * 1e3:.0f}" if infers else "--"
                        lat = f"{np.median(latencies[-10:]) * 1e3:.0f}" if latencies else "--"
                        jump = f" jump={switch_jumps[-1]:.3f}rad" if switch_jumps else ""
                        rejected = f" rejected={len(rejected_plan_jumps)}" if rejected_plan_jumps else ""
                        backlog = f" rec_backlog={writer.backlog}" if writer is not None else ""
                        print(
                            f"[tick {tick:5d}] {'EXEC   ' if args.execute else 'DRY RUN'} t={now - start:5.1f}s "
                            f"plans={policy.n_plans} infer={inf}ms latency={lat}ms tick={timing['tick'].value:.1f}ms "
                            f"(grab {timing['grab'].value:.1f} state {timing['state'].value:.1f} "
                            f"send {timing['send'].value:.1f}) overruns={overruns} engaged={engaged}{jump}{rejected}{backlog}"
                        )

            if stop_reason not in ("operator retry (q)", "max-seconds"):
                break
            print("[retry] saving this episode, folding, then returning to the task ready pose", flush=True)
            # A plan already in flight must be consumed before the reset RPC. Never execute it.
            deadline = time.monotonic() + 30.0
            while policy.busy and keep_resetting():
                if policy.poll(timeout=0.05) is None and time.monotonic() >= deadline:
                    raise TimeoutError("Policy did not finish the outstanding plan before episode reset")
            if writer is not None:
                saved_dir = save_recording()
            if not _return_to_ready(arms, limiters, ready, args, keep_resetting):
                stop_reason = "operator quit during reset (Esc)"
                break
            policy.reset_episode()
            last_frames.clear()
            episode_number += 1

        # ---- stop sequence: save -> open grippers -> fold -> (finally) power off ---------------
        print(f"[stop] {stop_reason}", flush=True)
        speaker.say("stopping")
        if writer is not None:
            print("[stop] 1/4 saving the recording (arms hold still) ...", flush=True)
            try:
                saved_dir = save_recording()
            except Exception as e:  # a failed save must not keep the arms up
                print(f"[error] saving the recording failed: {e}")
        if args.execute and (args.release_grippers or args.park_on_exit):
            print("[stop] 2/4 opening grippers, 3/4 folding to the rest pose", flush=True)
            _stop_sequence(
                arms,
                limiters,
                home,
                release=args.release_grippers,
                park=args.park_on_exit,
                park_pose=args.park_pose,
                release_seconds=args.release_seconds,
                park_seconds=args.park_seconds,
                send_hz=args.fps,
                speed_scale=args.park_speed_scale,
            )

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("[stop] interrupted (Ctrl-C) -- no fold, the arms hold their last command until power-off")
    except BaseException as e:
        stop_reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        if keys is not None:
            rec._cleanup_step("restore terminal", keys.close)
        if writer is not None:  # the loop did not end normally: still keep what was recorded

            def _save_late() -> None:
                nonlocal saved_dir
                saved_dir = save_recording()

            rec._cleanup_step("save recording", _save_late, timeout=120.0)
        if args.display:
            rec._cleanup_step("close display", cv2.destroyAllWindows)
        for side, arm in arms.items():
            rec._cleanup_step(f"close {side} client", arm.close)
        if rig is not None:
            rec._cleanup_step("stop cameras", rig.stop)
        if policy is not None:
            rec._cleanup_step("stop policy server", policy.close, timeout=20.0)
        if procs:
            print("[stop] 4/4 powering off: stopping the followers, motors disabled", flush=True)
            rec._cleanup_step("terminate followers", lambda: rec._terminate(procs), timeout=15.0)
        elif arms:
            print("[stop] --no-launch: the followers keep running, the arms stay powered where they are")

    # Only reached after Esc / Ctrl-C; powered-down followers have already been cleaned up.
    if (
        saved_dir is not None
        and saved_outcome is None
        and args.label_outcome
        and sys.stdin.isatty()
        and not shutdown.requested.is_set()
    ):
        print("[label] outcome?  s + Enter = success,  f + Enter = failure,  Enter = leave unlabeled")
        try:
            answer = _read_line(shutdown, ("s", "f", ""))
        except KeyboardInterrupt:
            answer = ""
        outcome = {"s": "success", "f": "failure"}.get(answer or "")
        if outcome is not None:
            print(f"[label] {outcome} -> {_relabel(saved_dir, outcome)}")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0.0 <= args.coast_beta <= 1.0:
        raise SystemExit("[error] --coast-beta must be finite and in [0, 1]")
    if args.replan_every < 1:
        raise SystemExit("[error] --replan-every must be >= 1")
    if args.max_plan_switch_jump < 0:
        raise SystemExit("[error] --max-plan-switch-jump must be >= 0")
    for name in ("start_forward", "start_down", "start_tilt_to_base_deg"):
        if getattr(args, name) is None:
            setattr(args, name, START_ADJUST.get(args.task, {}).get(name, 0.0))
    if args.steer:
        if args.model not in ("pi", "steeract_enc"):
            raise SystemExit("[error] --steer conflicts with --model; use only --model")
        args.model = "steeract_enc"
    if args.model == "steeract_dec" and args.num_inference_steps not in (None, 10):
        raise SystemExit("[error] released steeract_dec requires exactly 10 denoising steps")
    try:
        assets = resolve_assets(args.task, args.model, Path(args.model_root), args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    ckpt = assets.checkpoint
    if not args.execute:
        print(
            "[warn] DRY RUN -- the policy plans and the episode is recorded, but nothing is sent. --execute moves the arms."
        )
    run(args, ckpt)


if __name__ == "__main__":
    main(tyro.cli(Args))
