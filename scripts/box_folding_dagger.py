"""Bimanual YAM HG-DAgger: the box_folding policy drives, the operator corrects from the leaders.

The two follower arms run the policy exactly as scripts/box_folding_policy_rollout.py does (a
32-step chunk every ``--replan-every`` observation ticks, interpolated and slew-limited), while
both leader arms are held up as *handles*: whenever the policy drives an arm its leader MIRRORS
that arm (a PD hold on the measured follower pose), so the handle is always where the arm is.
Press the handle button and that arm switches to the human -- its leader goes free (gravity
compensation) and the follower tracks it, like ordinary teleop -- press it again and the policy
takes the arm back. The two arms are independent: correcting the left one leaves the right one
under the policy, and the policy keeps re-planning from what it sees either way.

Every tick of every episode is written in the bimanual_teleop_record layout, plus a per-arm
``control_mode`` that says who was driving, so a DAgger episode can be trained on next to a
demonstration and the human frames (the DAgger signal) can be told from the policy frames.

Controls (one operator; keys are global via pynput/X11, plus the preview window when focused):
    Space / n / mouse LEFT    IDLE -> start an episode (policy engages, recording starts)
                              RUNNING -> stop and SAVE
    r / d / mouse RIGHT       RUNNING -> stop and DISCARD;  IDLE -> retract the last saved episode
    handle button (per arm)   toggle that arm between the policy and the human. Works in IDLE
                              too, as plain teleop, for putting the arms back between episodes.
    i                         toggle BOTH arms at once from the keyboard (the only way in --sim)
    h                         IDLE -> home: open the grippers, then fold both arms (--home-pose)
    q / Esc                   quit: home first (--park-on-exit), then tear down

Every state change is announced through TTS (spd-say), as in the recorder.

Data layout, identical to bimanual_teleop_record.py so the same tools read both:
    <save_root>/<task>/episode_NNNN/
        top.mp4  left_wrist.mp4  right_wrist.mp4   # camera streams, one frame per tick (fps)
        low_dim.npz                                # see meta.json["low_dim_keys"]
        meta.json                                  # written last; source = "dagger"

    Per side, same keys as the recorder: joint_pos / eef_pos / eef_quat / gripper (state) and
    action_joint_pos / action_eef_pos / action_eef_quat / action_eef_delta / action_gripper
    (the action that drove the arm this tick -- the leader pose while the human had it, the
    policy target while the policy had it) and engaged (True whenever someone was driving).
    Added for DAgger:
        control_mode_<side>     0 = hold (nobody driving), 1 = policy, 2 = human
        policy_joint_pos_<side> what the policy asked for this tick even when overridden (NaN
                                before the first plan)
        human_joint_pos_<side>  the leader pose + trigger this tick, whoever was driving
        command_joint_pos_<side> what was actually sent after the slew/joint limits
        plan_age / plan_latency  as in the rollout recording

Takeover is seamless because of the mirroring: with ``--leader-mirror`` (default) the follower
is commanded to the leader's ABSOLUTE pose, and the leader was already there. With
``--no-leader-mirror`` the handles hang free, so a takeover maps the leader RELATIVELY (deltas
from the pose the arm held at takeover) -- no jump, but the handle and the arm drift apart with
every correction. On takeover the gripper only starts following the trigger once the trigger
has come within ``--grip-sync-tol`` of the gripper's current opening, so grabbing a handle with
the trigger squeezed does not slam the gripper shut on the box.

When the human hands an arm back, the policy does not resume its old chunk (planned before the
correction, so its targets are stale): the arm holds while a fresh plan is requested and takes
over from that. Those few ticks are recorded as control_mode 0.

Prerequisites are the rollout's plus the two leaders on their CAN buses (scripts/can_map.conf):
    python scripts/check_arms.py                     # all four arms answer
    python scripts/check_leader_triggers.py          # triggers sweep 1.0 -> 0.0

Usage:
    python scripts/box_folding_dagger.py                         # e2e 150k, ~/yam_data/box_folding_dagger
    python scripts/box_folding_dagger.py --task box_folding_dagger_r2 --policy-root ~/box_folding_policy/demo_full  # 300k
    python scripts/box_folding_dagger.py --replan-every 4         # 0.4 s chunks, twice as reactive
    python scripts/box_folding_dagger.py --no-launch             # followers already up (run_policy_followers.sh)
    python scripts/box_folding_dagger.py --sim --allow-missing-cameras   # plumbing only, no leaders
    python scripts/box_folding_dagger.py --no-execute            # everything runs, nothing is sent

Only the goal-free ``e2e`` policy is supported: the hierarchical one needs a Wan subgoal
generation per episode (~8 s once the generator is loaded, but ~20 GB of VRAM held for the whole
session to keep it that way), which does not fit a collection loop.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import portal
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import bimanual_teleop_record as rec  # cameras, FK, episode writer, operator IO, gello teardown
import box_folding_policy_rollout as roll  # follower launch/attach, FollowerArm, SafetyLimiter, Plan
from arm_offsets import offsets_for_channel
from box_folding_policy import (
    CONTROL_DT,
    DEFAULT_TORCH_HOME,
    FOLDED_Q,
    GRIPPER_OPEN,
    IMAGE_HW,
    VIEW_ORDER,
    WORKING_Q,
    PolicyProcess,
    build_state,
    default_checkpoint,
    default_policy_root,
    resize_to_policy,
    split_action,
)
from can_channels import channel_for

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.utils import override_log_level

SIDES = ("left", "right")
N_ARM = 6
"""Arm joints per side; a follower command is these plus the normalised gripper."""

MODE_HOLD, MODE_POLICY, MODE_HUMAN = 0, 1, 2
"""``control_mode_<side>`` values in low_dim.npz."""
_MODE_NAMES = {MODE_HOLD: "hold", MODE_POLICY: "policy", MODE_HUMAN: "human"}


_LEADER_STATE_LEN = 10
"""Leader worker -> main: q(6), trigger (gripper command 0-1), button, t_wall, seq."""
_LEADER_CMD_LEN = 8
"""Main -> leader worker: mode (0 free / 1 mirror), mirror target q(6), kp scale."""
_LEADER_STALE_S = 0.5
"""A leader reading older than this means its worker has stopped; the arm it drives holds."""
_BUTTON_DEBOUNCE_S = 0.5
_HOME_ARRIVED_TOL = 1e-3


@dataclass
class Args:
    # --- recording ---
    task: str = "box_folding_dagger"
    """Episodes land in <save_root>/<task>/episode_NNNN. Keep it distinct from the demonstration
    task name: the same uploader reads both, and a DAgger episode is a different kind of data."""
    save_root: str = "~/yam_data"
    fps: int = 30
    """Tick rate: one low-dim row and one video frame per tick, the same 30 Hz the demonstrations
    were recorded at. The policy observes every third tick (its 10 fps timeline)."""

    # --- policy (same knobs as the rollout) ---
    policy_root: Optional[str] = None
    """The policy bundle. Default: the rollout's, ~/box_folding_policy/e2e -- the checkpoint
    trained on the first, smaller demonstration set (its README says 55 episodes) and stopped at
    150k steps, which is the one that behaves best on the real arms (see
    box_folding_policy.DEFAULT_POLICY_ROOT for the measurement).
    ``--policy-root ~/box_folding_policy/demo_full`` runs the 105-episode 300k checkpoint
    instead. This script is e2e-only: a goal policy would need the Wan planner alongside it."""
    checkpoint: Optional[str] = None
    """Explicit checkpoint. Default: the newest one in the bundle."""
    device: str = "cuda:0"
    torch_home: str = DEFAULT_TORCH_HOME
    flow_steps: int = 10
    radio_dtype: str = "float16"
    trunk_dtype: str = "float16"
    seed: Optional[int] = 0

    # --- control ---
    execute: bool = True
    """Send commands to the arms. --no-execute runs everything (policy, takeover logic, recording)
    but sends nothing -- a dry run of the plumbing."""
    replan_every: int = 8
    """Minimum observation ticks (10 Hz) between plans, i.e. a fresh chunk every 0.8 s. Shorter
    than the rollout's 24 on purpose: under DAgger the scene changes under the policy whenever
    the human corrects an arm, so it should act on recent observations rather than commit to a
    2.4 s plan. A plan costs ~105 ms in its own process, so 0.8 s leaves plenty of room.

    Both directions have a cost, and 8 is the middle of them. Lower (the rollout used to be 3)
    means only the first few steps of each 32-step chunk are ever executed, and those are the
    steps closest to the current pose -- the arms move more slowly and the policy re-decides
    instead of following through. Higher means acting on an observation up to that many ticks
    old, which under DAgger is an observation from before the human's last correction.

    yam_data/box_folding_dagger round 1 was collected at this default; see
    docs/guides/box-folding-dagger-data.md."""
    max_plan_age: float = 8.0
    """Abort if the active chunk gets this old while the policy is driving: planning has died."""
    max_joint_vel: float = 1.5
    """Slew limit per arm joint, rad/s, on everything sent -- policy and human alike."""
    max_gripper_vel: float = 3.0
    """Slew limit on the normalised gripper command, 1/s."""
    max_start_error: float = 0.8
    """Refuse to start an episode if the policy's first target is farther than this (rad) from the
    measured pose: the policy does not recognise the scene. Take over and move the arms to a
    demonstration-like start pose, or press h to home."""

    # --- leaders ---
    leader_mirror: bool = True
    """While the policy drives an arm, its leader is PD-held at that arm's measured pose, so a
    takeover starts with the handle already where the arm is (absolute mapping). Off: the
    leaders hang free and a takeover maps them relatively (deltas from the takeover pose)."""
    mirror_kp_scale: float = 1.0
    """Fraction of the leader's own nominal kp used for the mirror hold. 1.0 is the same PD the
    follower tracks with -- the reference behaviour. Lower it if the handles feel too stiff to
    grab; too low and the handle lags the arm, which becomes a jump on takeover."""
    mirror_vel: float = 1.5
    """Slew limit, rad/s, on the leader's mirror target. Matters at hand-back: the handle is
    wherever the operator let go, and walks to the arm at this speed instead of snapping."""
    grip_sync_tol: float = 0.15
    """On takeover the gripper ignores the trigger until the trigger is this close (normalised
    0-1) to the gripper's current opening, then follows it directly."""

    # --- homing (h key, and --park-on-exit) ---
    home_pose: Literal["folded", "working"] = "folded"
    """Where h / --park-on-exit take the arms after opening the grippers: the folded rest pose by
    way of the working pose (where an unpowered arm rests -- the default, since teardown zeroes
    every motor torque), or just the working pose."""
    park_on_exit: bool = True
    """On q, home the arms before the followers are torn down. Skipped after Ctrl-C and after an
    error, as in the rollout: an interrupt means stop now."""
    park_speed_scale: float = 0.5
    park_seconds: float = 8.0
    """Time budget per homing leg (release, working pose, folded pose)."""

    # --- robot ---
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = channel_for("follower_left")
    can_follower_right: str = channel_for("follower_right")
    can_leader_left: str = channel_for("leader_left")
    can_leader_right: str = channel_for("leader_right")
    """All four from scripts/can_map.conf, the one mapping table."""
    port_left: int = 1235
    port_right: int = 1234
    launch: bool = True
    """Spawn the two follower servers. --no-launch attaches to running ones
    (scripts/run_policy_followers.sh), which keeps the arms powered between sessions."""
    leaders: bool = True
    """Bring up the two leader arms in this process. --no-leaders (or --sim) runs without handles:
    takeover is then only the keyboard i, and a 'human' arm simply holds."""
    sim: bool = False
    """Followers in MuJoCo, no leaders, cameras optional. Plumbing test only."""

    # --- cameras (same fields and defaults as the recorder) ---
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

    # --- operator IO ---
    display: bool = True
    display_width: int = 1600
    global_keys: bool = True
    mouse: bool = True
    tts: bool = True
    log_every: int = 30
    """Print a status line every N ticks."""


# ---------------------------------------------------------------------------
# Leader arms: one worker process per side, talking through shared arrays
# ---------------------------------------------------------------------------


def _leader_worker(
    side: str,
    channel: str,
    arm: str,
    version: int,
    mirror_vel: float,
    state_shared: "portal.SharedArray",
    cmd_shared: "portal.SharedArray",
    stop_event: Any,
) -> None:
    """Owns one leader arm. Publishes (q, trigger, button) at hardware rate and either leaves the
    arm in gravity compensation (free, the human has it) or PD-holds it at the mirror target.

    Its own process for the same reason minimum_gello's leader is: the CAN control thread must
    not share a GIL with the cameras, the RPC clients and the tick loop. SIGINT is ignored here
    so the parent sequences the shutdown (stop_event, then close -> zero torque); if the parent
    vanishes anyway the ppid check ends the loop."""
    override_log_level()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = os.getppid()
    yam = get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name(arm),
        version=version,
        gripper_type=GripperType.YAM_TEACHING_HANDLE,
        # Same per-arm zero correction the followers get through minimum_gello: this leader's
        # joint angles become the followers' targets, so both ends must share one frame.
        joint_offsets=offsets_for_channel(channel),
    )
    try:
        nominal_kp = np.asarray(yam._kp, dtype=np.float64).copy()
        grav_kd = np.asarray(yam._grav_comp_kd, dtype=np.float64).copy()
        chain = yam.motor_chain
        mirroring = False
        mirror_q = np.zeros(N_ARM)
        seq = 0
        last = time.monotonic()
        while not stop_event.is_set() and os.getppid() == parent:
            now = time.monotonic()
            dt = min(now - last, 0.05)
            last = now
            q = np.asarray(yam.get_observations()["joint_pos"], dtype=np.float64)[:N_ARM]
            states = chain.get_same_bus_device_states()
            if states:  # populated after the first CAN read of the encoder
                enc = states[0]
                trigger = 1.0 - float(enc.position)  # minimum_gello's gripper command
                button = float(np.asarray(enc.io_inputs, dtype=np.float64).reshape(-1)[0] > 0.5)
                seq += 1
                state_shared.array[:] = [*q, trigger, button, time.time(), float(seq)]
            cmd = cmd_shared.array.copy()
            if cmd[0] > 0.5:
                if not mirroring:
                    # Start the hold where the handle IS, then walk to the target: a PD hold
                    # commanded straight at a distant target would yank the handle there.
                    mirror_q = q.copy()
                    yam.update_kp_kd(nominal_kp * float(cmd[7]), grav_kd)
                    mirroring = True
                step = mirror_vel * dt
                mirror_q = mirror_q + np.clip(cmd[1 : 1 + N_ARM] - mirror_q, -step, step)
                yam.command_joint_pos(mirror_q)
            elif mirroring:
                yam.enter_gravity_comp_idle()
                mirroring = False
            time.sleep(0.002)
    finally:
        yam.close()


@dataclass
class LeaderReading:
    q: np.ndarray  # 6 arm joints
    trigger: float  # gripper command, 1 = open
    button: bool
    age_s: float  # since the worker last published

    @property
    def human_target(self) -> np.ndarray:
        return np.concatenate([self.q, [self.trigger]])


class LeaderHandle:
    """Main-process side of one leader worker."""

    def __init__(self, side: str, channel: str, args: Args) -> None:
        self.side = side
        self.state = portal.SharedArray((_LEADER_STATE_LEN,), np.float64)
        self.state.array[:] = 0.0
        self.cmd = portal.SharedArray((_LEADER_CMD_LEN,), np.float64)
        self.cmd.array[:] = 0.0
        self.cmd.array[7] = float(args.mirror_kp_scale)
        self.stop_event = portal.mp.Event()
        self.proc = portal.Process(
            _leader_worker,
            side,
            channel,
            args.arm,
            args.version,
            args.mirror_vel,
            self.state,
            self.cmd,
            self.stop_event,
            name=f"leader-{side}",
        )
        self.proc.start()

    def wait_ready(self, timeout_s: float, shutdown: rec.ShutdownRequest) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            shutdown.raise_if_requested()
            if self.state.array[9] > 0:
                return
            if not self.proc.running:
                raise RuntimeError(f"{self.side} leader worker exited during startup (dead CAN bus? see its log)")
            time.sleep(0.1)
        raise TimeoutError(f"{self.side} leader did not publish within {timeout_s:.0f}s")

    def read(self) -> LeaderReading:
        s = self.state.array.copy()
        return LeaderReading(q=s[:N_ARM], trigger=float(s[6]), button=s[7] > 0.5, age_s=time.time() - float(s[8]))

    def free(self) -> None:
        self.cmd.array[0] = 0.0

    def mirror(self, q: np.ndarray) -> None:
        self.cmd.array[1 : 1 + N_ARM] = np.asarray(q, dtype=np.float64)[:N_ARM]
        self.cmd.array[0] = 1.0

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.proc.join(timeout=4.0)  # lets the worker run yam.close() (zero torque)
        finally:
            if self.proc.running:
                self.proc.kill()
            self.state.close()
            self.cmd.close()


# ---------------------------------------------------------------------------
# Operator IO
# ---------------------------------------------------------------------------


class DaggerInput(rec.OperatorInput):
    """The recorder's toggle/discard/quit plus h (home) and i (toggle both arms)."""

    HOME = "home"
    INTERVENE = "intervene"

    def feed_char(self, char: str) -> None:
        if char == "h":
            self._push(self.HOME)
        elif char == "i":
            self._push(self.INTERVENE)
        else:
            super().feed_char(char)


# ---------------------------------------------------------------------------
# Per-arm control state
# ---------------------------------------------------------------------------


@dataclass
class ArmControl:
    side: str
    limiter: roll.SafetyLimiter
    human: bool = False
    grip_synced: bool = False
    anchor_leader: Optional[np.ndarray] = None  # relative mapping: leader pose at takeover
    anchor_follower: Optional[np.ndarray] = None  # relative mapping: command at takeover
    waiting_plan: bool = False  # after hand-back: hold until a plan newer than handback_t
    handback_t: float = 0.0
    button_down: bool = False  # last seen level, so a held button toggles once
    last_button_t: float = 0.0
    n_takeovers: int = 0
    human_frames: int = 0

    def take_over(self, reading: Optional[LeaderReading], now: float, mirror: bool) -> None:
        self.human = True
        self.grip_synced = False
        self.n_takeovers += 1
        self.waiting_plan = False
        if reading is not None and not mirror:
            self.anchor_leader = reading.q.copy()
            base = self.limiter.last
            self.anchor_follower = (base[:N_ARM] if base is not None else reading.q).copy()

    def hand_back(self, now: float) -> None:
        self.human = False
        self.waiting_plan = True
        self.handback_t = now

    def human_command(self, reading: LeaderReading, mirror: bool, grip_sync_tol: float) -> np.ndarray:
        """The follower target for this tick while the human has the arm."""
        if mirror or self.anchor_leader is None or self.anchor_follower is None:
            q = reading.q
        else:
            q = self.anchor_follower + (reading.q - self.anchor_leader)
        grip = reading.trigger
        last = self.limiter.last
        if not self.grip_synced and last is not None:
            if abs(grip - float(last[N_ARM])) <= grip_sync_tol:
                self.grip_synced = True
            else:
                grip = float(last[N_ARM])  # hold until the trigger has caught up
        return np.concatenate([q, [grip]])


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _side_sample(
    side: str,
    pos: np.ndarray,
    action: np.ndarray,
    kin: rec.EEFKinematics,
    mode: int,
    policy_q: Optional[np.ndarray],
    human_q: Optional[np.ndarray],
    sent: Optional[np.ndarray],
) -> Dict[str, Any]:
    """One side's low-dim row. The recorder's keys mean the recorder's things; ``action_*`` is
    whatever drove the arm this tick (leader pose, policy target, or the measured pose on hold)."""
    nan7 = np.full(N_ARM + 1, np.nan)
    eef_pos, eef_quat = kin.fk(pos)
    act_pos, act_quat = kin.fk(action)
    g = kin.n_arm_joints
    return {
        f"joint_pos_{side}": pos,
        f"eef_pos_{side}": eef_pos,
        f"eef_quat_{side}": eef_quat,
        f"gripper_{side}": pos[g] if len(pos) > g else np.nan,
        f"action_joint_pos_{side}": action,
        f"action_eef_pos_{side}": act_pos,
        f"action_eef_quat_{side}": act_quat,
        f"action_eef_delta_{side}": rec.EEFKinematics.pose_delta(act_pos, act_quat, eef_pos, eef_quat),
        f"action_gripper_{side}": action[g] if len(action) > g else np.nan,
        f"engaged_{side}": mode != MODE_HOLD,
        f"control_mode_{side}": np.int8(mode),
        f"policy_joint_pos_{side}": nan7 if policy_q is None else np.asarray(policy_q, dtype=np.float64),
        f"human_joint_pos_{side}": nan7 if human_q is None else np.asarray(human_q, dtype=np.float64),
        f"command_joint_pos_{side}": nan7 if sent is None else np.asarray(sent, dtype=np.float64),
    }


def _meta(args: Args, rig: rec.CameraRig, policy: PolicyProcess, ckpt: Path) -> Dict[str, Any]:
    return {
        "task": args.task,
        "source": "dagger",
        "arm": args.arm,
        "version": args.version,
        "cameras": {role: cam["serial"] for role, cam in rig.cams_by_role.items()},
        "camera_color_profiles": {
            role: (f"{cam['color'][0]}x{cam['color'][1]}@{cam['color'][2]}" if cam.get("color") else None)
            for role, cam in rig.cams_by_role.items()
        },
        "arms": {
            "left": {"follower_can": args.can_follower_left, "leader_can": args.can_leader_left},
            "right": {"follower_can": args.can_follower_right, "leader_can": args.can_leader_right},
        },
        "policy": policy.description,
        "checkpoint": str(ckpt),
        "instruction": policy.info.get("instruction"),
        "action_source": "control_mode_<side>: 1 = policy chunk target, 2 = leader pose (human), "
        "0 = nobody driving (action falls back to the measured pos, engaged False)",
        "leader_mapping": "absolute (leader mirrors the follower)" if args.leader_mirror else "relative",
        "low_dim_keys": {
            "joint_pos_<side>": "measured joint positions, 6 arm joints + gripper (normalized 0-1)",
            "eef_pos_<side> / eef_quat_<side>": "FK of measured arm joints: terminal mount pose in the "
            "arm's base frame, quat wxyz",
            "gripper_<side>": "measured gripper opening, normalized",
            "action_joint_pos_<side>": "the command that drove the arm this tick (absolute), same layout "
            "as joint_pos: leader pose when control_mode is 2, policy target when 1, measured when 0",
            "action_eef_pos_<side> / action_eef_quat_<side>": "FK of action_joint_pos",
            "action_eef_delta_<side>": "commanded pose relative to measured pose: base-frame dpos (3) + "
            "base-frame axis-angle drot (3)",
            "action_gripper_<side>": "commanded gripper, normalized",
            "engaged_<side>": "True when someone (policy or human) drove the arm this tick",
            "control_mode_<side>": "0 hold, 1 policy, 2 human",
            "policy_joint_pos_<side>": "the policy's target this tick, also while overridden (NaN before "
            "the first plan)",
            "human_joint_pos_<side>": "leader arm joints + trigger this tick, whoever was driving (NaN "
            "without leaders)",
            "command_joint_pos_<side>": "what was sent to the follower after the slew/joint limits",
            "plan_age / plan_latency": "age of the active chunk (s) and how long it took to plan",
            "t_mono / t_wall / cam_t_<role>": "tick timestamps (s since episode start / unix s / camera ms)",
        },
        "sim": args.sim,
        "args": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in asdict(args).items()
        },
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _build_display(
    frames: Dict[str, Tuple[np.ndarray, float]], lines: List[str], width: int, rec_on: bool
) -> np.ndarray:
    tiles = []
    for role in rec._CAMERA_ROLES:
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
    bar = np.zeros((26 * len(lines) + 12, width, 3), dtype=np.uint8)
    if rec_on:
        cv2.circle(bar, (20, 22), 10, (0, 0, 255), -1)
    for i, line in enumerate(lines):
        color = (0, 0, 255) if (i == 0 and rec_on) else (220, 220, 220)
        cv2.putText(bar, line, (40, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return np.vstack([bar, row])


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

IDLE, ARMING, RUNNING, HOMING = "IDLE", "ARMING", "RUNNING", "HOMING"


@dataclass
class Session:
    state: str = IDLE
    episode_index: int = 0
    saved: List[Path] = field(default_factory=list)
    writer: Optional[rec.EpisodeWriter] = None
    quit_after_home: bool = False
    home_legs: List[Tuple[str, Dict[str, np.ndarray]]] = field(default_factory=list)
    leg_deadline: float = 0.0


def _home_legs(ctls: Dict[str, ArmControl], pose: str) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """Release the grippers where the arms stand, then walk to the home pose -- the rollout's stop
    sequence, expressed as targets so the tick loop can step them without blocking."""
    release: Dict[str, np.ndarray] = {}
    for side, ctl in ctls.items():
        hold = ctl.limiter.last.copy()
        hold[N_ARM] = GRIPPER_OPEN
        release[side] = hold
    legs = [("release grippers", release), ("working pose", {s: WORKING_Q.copy() for s in ctls})]
    if pose == "folded":
        legs.append(("folded rest pose", {s: FOLDED_Q.copy() for s in ctls}))
    return legs


def run(args: Args, ckpt: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task_dir = Path(args.save_root).expanduser() / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    speaker = rec.Speaker(args.tts)

    procs: List["subprocess.Popen[bytes]"] = []
    shutdown = rec.ShutdownRequest(procs)
    leaders: Dict[str, LeaderHandle] = {}
    arms: Dict[str, roll.FollowerArm] = {}
    rig: Optional[rec.CameraRig] = None
    policy: Optional[PolicyProcess] = None
    inputs: Optional[DaggerInput] = None
    session = Session(episode_index=rec._next_episode_index(task_dir))
    stop_reason = "startup"
    window = "box_folding dagger"
    use_leaders = args.leaders and not args.sim
    mirror = args.leader_mirror and use_leaders

    try:
        # ---- bring-up: followers, leaders, cameras, policy -------------------------------
        ports = {"left": args.port_left, "right": args.port_right}
        if args.launch:
            roll._launch_followers(args, procs, shutdown)
        else:
            roll._check_attached_followers(args, ports)
        if use_leaders:
            for interface in (args.can_leader_left, args.can_leader_right):
                rec._check_can_interface(interface)
            print(f"[can] leader-left {args.can_leader_left}, leader-right {args.can_leader_right}")
            # Before any thread exists in this process (camera poll, pynput, RPC sockets).
            for side, channel in (("right", args.can_leader_right), ("left", args.can_leader_left)):
                leaders[side] = LeaderHandle(side, channel, args)
        arms = {"left": roll.FollowerArm("left", args.port_left), "right": roll.FollowerArm("right", args.port_right)}
        print(f"[robot] followers connected ({', '.join(f'{s}:{a.num_dofs} dof' for s, a in arms.items())})")
        for handle in leaders.values():
            handle.wait_ready(60.0, shutdown)
        if leaders:
            print("[robot] leaders up" + (" -- mirroring the followers" if mirror else " -- hanging free"))

        kin = rec.EEFKinematics(args.arm, args.version)
        limits = roll._joint_range(args.arm, args.version)
        ctls = {s: ArmControl(s, roll.SafetyLimiter(limits, args.max_joint_vel, args.max_gripper_vel)) for s in SIDES}

        rig = rec._open_camera_rig(args)
        deadline = time.monotonic() + 10.0
        while not rig.all_fresh() and time.monotonic() < deadline:
            shutdown.raise_if_requested()
            time.sleep(0.1)
        if not rig.all_fresh() and not (args.allow_missing_cameras or args.sim):
            raise RuntimeError("cameras did not deliver frames (check USB bandwidth and cables)")

        print("[policy] starting the policy process (loads the checkpoint and RADIO) ...")
        policy = PolicyProcess(
            policy_root=args.policy_root,
            checkpoint=args.checkpoint,
            mode="e2e",
            device=args.device,
            flow_steps=args.flow_steps,
            radio_dtype=args.radio_dtype,
            trunk_dtype=args.trunk_dtype,
            torch_home=args.torch_home,
            seed=args.seed,
        )
        info = policy.start()
        history = roll.FrameHistory(int(info["history_len"]))
        print(f"[policy] {policy.description}")
        meta_base = _meta(args, rig, policy, ckpt)

        inputs = DaggerInput(global_keys=args.global_keys, mouse=args.mouse)
        print(
            f"[mode] {'EXECUTE -- the arms will move' if args.execute else 'DRY RUN -- nothing is sent'}; "
            "space: start/save, r: discard, handle button: take over / hand back, i: both, h: home, q: quit"
        )
        speaker.say(f"dagger ready, next episode {session.episode_index}")

        # ---- loop state --------------------------------------------------------------
        dt = 1.0 / args.fps
        obs_every = max(1, round(CONTROL_DT * args.fps))
        active: Optional[roll.Plan] = None
        force_replan = False
        tick = 0
        last_frames: Dict[str, Tuple[np.ndarray, float]] = {}
        last_state = np.zeros(28, np.float32)
        last_send: Optional[float] = None
        overruns = 0
        stop_reason = "operator quit"

        def end_episode(save: bool, why: str) -> None:
            nonlocal active
            writer = session.writer
            session.writer = None
            session.state = IDLE
            for ctl in ctls.values():
                ctl.waiting_plan = False
            active = None
            if writer is None:
                return
            if save and writer.n_frames:
                stats = {
                    s: {"takeovers": c.n_takeovers, "human_frames": c.human_frames, "frames": writer.n_frames}
                    for s, c in ctls.items()
                }
                ep_dir = writer.save(dict(meta_base, interventions=stats, plans=policy.n_plans))
                session.saved.append(ep_dir)
                speaker.say(f"saved {session.episode_index}, total {len(session.saved)}")
                session.episode_index += 1
            else:
                writer.discard()
                speaker.say(f"{why} {session.episode_index}")

        def set_human(side: str, flag: bool, reading: Optional[LeaderReading], now: float) -> None:
            ctl = ctls[side]
            if flag == ctl.human:
                return
            if flag:
                ctl.take_over(reading, now, mirror)
                speaker.say(f"{side} human")
            else:
                ctl.hand_back(now)
                speaker.say(f"{side} policy" if session.state in (ARMING, RUNNING) else f"{side} hold")

        while True:
            tick_start = time.monotonic()
            shutdown.raise_if_requested()
            now = tick_start
            tick += 1

            # ---- operator events ----------------------------------------------------
            event = inputs.pop()
            if event == DaggerInput.QUIT:
                if session.state in (ARMING, RUNNING):
                    end_episode(False, "discarded")
                if session.state == HOMING:
                    session.quit_after_home = True
                elif args.park_on_exit and args.execute and not any(c.human for c in ctls.values()):
                    for ctl in ctls.values():
                        if ctl.limiter.last is None:
                            ctl.limiter.reset(arms[ctl.side].joint_pos())
                    session.home_legs = _home_legs(ctls, args.home_pose)
                    session.leg_deadline = now + args.park_seconds
                    session.state = HOMING
                    session.quit_after_home = True
                    speaker.say("homing, then quitting")
                else:
                    speaker.say("quitting")
                    break
            elif event == DaggerInput.TOGGLE:
                if session.state == IDLE:
                    if rig.cams_by_role and not rig.all_fresh():
                        speaker.say("cameras not ready")
                    else:
                        session.writer = rec.EpisodeWriter(task_dir / f"episode_{session.episode_index:04d}", args.fps)
                        for ctl in ctls.values():
                            ctl.n_takeovers = 1 if ctl.human else 0
                            ctl.human_frames = 0
                            ctl.waiting_plan = False
                        session.state = ARMING
                        active = None
                        force_replan = True
                        speaker.say(f"recording {session.episode_index}")
                elif session.state in (ARMING, RUNNING):
                    end_episode(True, "saved")
            elif event == DaggerInput.DISCARD:
                if session.state in (ARMING, RUNNING):
                    end_episode(False, "discarded")
                elif session.state == IDLE and session.saved:
                    retracted = session.saved.pop()
                    rec._retire_episode(retracted)
                    session.episode_index = rec._next_episode_index(task_dir)
                    speaker.say(f"retracted {retracted.name}")
                else:
                    speaker.say("nothing to retract")
            elif event == DaggerInput.HOME:
                if session.state != IDLE:
                    speaker.say("home only when idle")
                elif any(c.human for c in ctls.values()):
                    speaker.say("hand both arms back first")
                elif not args.execute:
                    speaker.say("dry run, no homing")
                else:
                    for ctl in ctls.values():
                        if ctl.limiter.last is None:
                            ctl.limiter.reset(arms[ctl.side].joint_pos())
                    session.home_legs = _home_legs(ctls, args.home_pose)
                    session.leg_deadline = now + args.park_seconds
                    session.state = HOMING
                    speaker.say("homing")

            # ---- leaders: readings and handle buttons ---------------------------------
            readings: Dict[str, LeaderReading] = {}
            for side, handle in leaders.items():
                r = handle.read()
                readings[side] = r
                ctl = ctls[side]
                pressed = r.button and not ctl.button_down  # rising edge only
                ctl.button_down = r.button
                if pressed and session.state != HOMING and now - ctl.last_button_t > _BUTTON_DEBOUNCE_S:
                    ctl.last_button_t = now
                    set_human(side, not ctl.human, r, now)
                    if not ctl.human:
                        force_replan = True
            if event == DaggerInput.INTERVENE and session.state != HOMING:
                flag = not all(c.human for c in ctls.values())
                for side in SIDES:
                    set_human(side, flag, readings.get(side), now)
                if not flag:
                    force_replan = True

            # ---- cameras + policy observation --------------------------------------------
            frames = rig.snapshot()
            for view, item in frames.items():
                last_frames[view] = item
            obs_tick = tick % obs_every == 0
            if obs_tick:
                policy_frames: Dict[str, np.ndarray] = {}
                for view in VIEW_ORDER:
                    if view in last_frames:
                        policy_frames[view] = resize_to_policy(last_frames[view][0], bgr=True)
                    elif args.allow_missing_cameras or args.sim:
                        policy_frames[view] = np.zeros((*IMAGE_HW, 3), np.uint8)
                    else:
                        raise RuntimeError(f"camera '{view}' has delivered no frame")
                history.push(policy_frames)

            # ---- follower state (every tick: it is recorded, and the leaders mirror it) ------
            pos = {side: arms[side].joint_pos() for side in SIDES}
            fk = {side: kin.fk(pos[side]) for side in SIDES}
            last_state = build_state(
                pos["left"], pos["right"], fk["left"][0], fk["left"][1], fk["right"][0], fk["right"][1]
            )

            if obs_tick and session.state in (ARMING, RUNNING) and history.ready and not policy.busy:
                due = force_replan or (now - policy.last_request_t0 >= args.replan_every * CONTROL_DT)
                if active is None or due:
                    ids, pixels = history.snapshot()
                    policy.request(now, pixels, last_state, ids)
                    force_replan = False

            # ---- newest plan ----------------------------------------------------------------
            done = policy.poll()
            if done is not None:
                t0, chunk, latency, _cursor = done
                plan = roll.Plan(chunk=chunk, t0=t0, latency_s=latency)
                if session.state == ARMING:
                    target, _ = plan.target(now)
                    err = {
                        s: float(np.max(np.abs(split_action(target)[i][:N_ARM] - pos[s][:N_ARM])))
                        for i, s in enumerate(SIDES)
                        if not ctls[s].human
                    }
                    worst = max(err.values(), default=0.0)
                    print(f"[engage] first target is {worst:.3f} rad from the measured pose ({err})")
                    if worst > args.max_start_error:
                        print(
                            f"[engage] refused: {worst:.3f} > --max-start-error {args.max_start_error:.2f}; the policy "
                            "does not recognise this scene. Take over (handle button) or press h to home."
                        )
                        end_episode(False, "policy too far, discarded")
                    else:
                        active = plan
                        session.state = RUNNING
                        for ctl in ctls.values():
                            if not ctl.human:
                                ctl.limiter.reset(pos[ctl.side])
                        speaker.say("policy")
                elif session.state == RUNNING:
                    active = plan
                    for ctl in ctls.values():
                        if ctl.waiting_plan and t0 >= ctl.handback_t:
                            ctl.waiting_plan = False
                            ctl.limiter.reset(pos[ctl.side])
                            back = float(
                                np.max(
                                    np.abs(
                                        split_action(chunk[0])[SIDES.index(ctl.side)][:N_ARM] - pos[ctl.side][:N_ARM]
                                    )
                                )
                            )
                            print(f"[handback] {ctl.side}: policy resumes, first target {back:.3f} rad away")

            # ---- per-arm command --------------------------------------------------------------
            send_elapsed = dt if last_send is None else min(now - last_send, dt)
            last_send = now
            policy_target = active.target(now)[0] if active is not None else None
            policy_parts = (
                dict(zip(SIDES, split_action(policy_target), strict=True)) if policy_target is not None else {}
            )
            if session.state == RUNNING and active is not None and now - active.t0 > args.max_plan_age:
                if any(not c.human for c in ctls.values()):
                    raise RuntimeError(f"no fresh plan for {now - active.t0:.1f}s -- planning has stopped")

            modes: Dict[str, int] = {}
            actions: Dict[str, np.ndarray] = {}
            sent: Dict[str, Optional[np.ndarray]] = {}
            if session.state == HOMING:
                leg_name, leg_targets = session.home_legs[0]
                arrived = True
                for side, ctl in ctls.items():
                    cmd = ctl.limiter.step(leg_targets[side], send_elapsed * args.park_speed_scale)
                    if args.execute:
                        arms[side].command(cmd)
                    if float(np.max(np.abs(cmd - leg_targets[side][: cmd.size]))) > _HOME_ARRIVED_TOL:
                        arrived = False
                    modes[side], actions[side], sent[side] = MODE_HOLD, leg_targets[side], cmd
                if arrived or now > session.leg_deadline:
                    if not arrived:
                        print(f"[home] ran out of time on the {leg_name} leg -- moving on")
                    session.home_legs.pop(0)
                    session.leg_deadline = now + args.park_seconds
                    if not session.home_legs:
                        session.state = IDLE
                        speaker.say("home")
                        if session.quit_after_home:
                            break
            else:
                for side, ctl in ctls.items():
                    reading = readings.get(side)
                    target: Optional[np.ndarray] = None
                    mode = MODE_HOLD
                    if ctl.human:
                        mode = MODE_HUMAN
                        if reading is not None and reading.age_s < _LEADER_STALE_S:
                            target = ctl.human_command(reading, mirror, args.grip_sync_tol)
                        elif reading is not None:
                            logging.warning(f"{side} leader reading is {reading.age_s:.1f}s old -- holding")
                    elif session.state == RUNNING and side in policy_parts and not ctl.waiting_plan:
                        mode = MODE_POLICY
                        target = policy_parts[side]
                    if target is not None:
                        cmd = ctl.limiter.step(target, send_elapsed)
                        if args.execute:
                            arms[side].command(cmd)
                        sent[side] = cmd
                        actions[side] = np.asarray(target, dtype=np.float64)
                    else:
                        sent[side] = None
                        actions[side] = pos[side].copy()
                    modes[side] = mode
                    if mode == MODE_HUMAN:
                        ctl.human_frames += 1

            # ---- leaders: free while the human has them, else mirror the follower -----------
            for side, handle in leaders.items():
                if ctls[side].human or not mirror:
                    handle.free()
                else:
                    handle.mirror(pos[side][:N_ARM])

            # ---- record -------------------------------------------------------------------
            if session.writer is not None:
                sample: Dict[str, Any] = {
                    "plan_age": float(now - active.t0) if active is not None else np.nan,
                    "plan_latency": float(active.latency_s) if active is not None else np.nan,
                }
                for side in SIDES:
                    reading = readings.get(side)
                    sample.update(
                        _side_sample(
                            side,
                            pos[side],
                            actions[side],
                            kin,
                            modes[side],
                            policy_parts.get(side),
                            reading.human_target if reading is not None else None,
                            sent[side],
                        )
                    )
                session.writer.add(sample, frames)

            # ---- display / log --------------------------------------------------------------
            who = "  ".join(f"{s[0].upper()}:{_MODE_NAMES[modes[s]]}" for s in SIDES)
            if args.display:
                if session.state == HOMING:
                    head = f"HOMING ({session.home_legs[0][0]})"
                elif session.writer is not None:
                    head = f"REC episode {session.episode_index}  frames {session.writer.n_frames}  {session.state}"
                else:
                    head = f"IDLE  next episode {session.episode_index}"
                age = "--" if active is None else f"{now - active.t0:.2f}s"
                lines = [
                    head,
                    f"{who}   plan age {age}  plans {policy.n_plans}  saved {len(session.saved)}"
                    f"  {'EXEC' if args.execute else 'DRY RUN'}",
                    "space/LMB: start-save  r/RMB: discard  button/i: take over  h: home  q: quit",
                ]
                cv2.imshow(window, _build_display(frames, lines, args.display_width, session.writer is not None))
                key = cv2.waitKey(1) & 0xFF
                if key not in (255, 0):
                    inputs.feed_char(chr(key) if key < 128 else "")
            if tick % max(1, args.log_every) == 0:
                age = "--" if active is None else f"{now - active.t0:.2f}"
                print(f"[tick {tick:5d}] {session.state:7s} {who}  plan age {age} s  plans {policy.n_plans}")

            elapsed = time.monotonic() - tick_start
            if elapsed > dt * 1.5:
                overruns += 1
                if overruns % 100 == 1:
                    logging.warning(f"slow tick: {elapsed * 1000:.0f} ms (target {dt * 1000:.0f} ms)")
            else:
                time.sleep(max(0.0, dt - elapsed))

        print(f"[stop] {stop_reason}")
        if args.launch and args.execute:
            print("[stop] the follower processes are about to exit: motor torques go to zero, the arms go limp")

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("[stop] interrupted -- the followers hold their last command, the leaders go limp")
    except BaseException as e:
        stop_reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        if session.writer is not None:
            rec._cleanup_step("discard open episode", session.writer.discard)
            speaker.say("episode discarded")
        if inputs is not None:
            rec._cleanup_step("input listeners", inputs.close)
        if args.display:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logging.warning(f"[exit] preview window: {e}")
        if rig is not None:
            rec._cleanup_step("cameras", rig.stop)
        if policy is not None:
            rec._cleanup_step("policy process", policy.close, timeout=10.0)
        # Leaders before the followers: a leader is only ever held up by its own torque, so it is
        # freed while the operator is still holding the handle, not after the arms have dropped.
        for side, handle in leaders.items():
            rec._cleanup_step(f"{side} leader", handle.close, timeout=8.0)
        for side, arm in arms.items():
            rec._cleanup_step(f"{side} follower client", arm.close)
        if procs:
            rec._cleanup_step("terminate followers", lambda: rec._terminate(procs), timeout=15.0)
        print(f"[exit] {len(session.saved)} episode(s) saved this session -> {task_dir}")


def main(args: Args) -> None:
    if args.replan_every < 1:
        raise SystemExit("[error] --replan-every must be >= 1")
    if args.fps < 10 or args.fps % 10:
        raise SystemExit("[error] --fps must be a multiple of 10 (the policy observes on a 10 fps timeline)")
    args.policy_root = args.policy_root or default_policy_root("e2e")
    ckpt = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else default_checkpoint(Path(args.policy_root).expanduser().resolve(), "e2e")
    )
    if not ckpt.is_file():
        raise SystemExit(f"[error] checkpoint not found: {ckpt}")
    print(f"[policy] e2e: {ckpt}")
    if not args.execute:
        print("[warn] DRY RUN -- nothing is sent to the arms (policy or human); drop --no-execute to drive them")
    run(args, ckpt)


if __name__ == "__main__":
    main(tyro.cli(Args))
