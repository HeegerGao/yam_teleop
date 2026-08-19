"""Closed-loop rollout of the box_folding end-to-end flow policy on the bimanual YAM.

Brings up the two follower arms (no leaders -- the policy replaces the operator), opens the
three RealSense cameras, and runs the policy as a receding-horizon controller: an observation
every 100 ms, a fresh 32-step action chunk every ``--replan-every`` ticks, and interpolated
absolute joint commands streamed to both arms at ``--send-hz``.

    observe  10 Hz    ~5 ms  grab 3 frames, resize to 240x320, buffer them as uint8
    plan     ~3 Hz  ~190 ms  RADIO + prefix + 10 Euler steps, in a separate policy *process*
    send     30 Hz    <1 ms  interpolate the active chunk at (now - t0) / 0.1 s

The policy runs in its own process, not a thread: portal's client socket threads busy-spin while
they have traffic, and in-process that inflated a plan from ~210 ms to ~550 ms (and, on the real
robot, up to 1.45 s) by holding the GIL between CUDA launches. See PolicyProcess.

Planning runs off the control loop precisely because it is longer than one tick; the send loop
keeps streaming the previous chunk meanwhile. A chunk is never extrapolated past its horizon --
once it runs out the arms hold its last step, and only ``--max-plan-age`` of that ends the run.

**Safety.** ``--execute`` is required to actually move the arms; without it everything runs and
prints but no command is sent. Commands are clamped to the arm's URDF joint limits, the gripper
to 0-1, and slew-rate-limited by ``--max-joint-vel``. The run refuses to engage if the policy's
first target is more than ``--max-start-error`` from the measured pose (that means the policy
does not recognise the scene -- start from a pose like the demonstrations began in). Ctrl-C,
``q``/Esc in the preview window and ``--max-seconds`` all stop the arms and hold.

Prerequisites: the arms powered and on the CAN buses named in ``scripts/can_map.conf``, the three
cameras plugged in, and the policy bundle plus RADIO weights on disk (see box_folding_policy.py).

When this script launches the followers itself it also tears them down, and that sets every motor
torque to zero -- so the arms go limp at the end of *every* run, including a clean one. It parks
them back at the pose they started in first (--no-park-on-exit to skip), but for repeated runs
prefer keeping the followers alive in their own terminal:

    scripts/run_policy_followers.sh                                    # terminal 1, stays up
    python scripts/box_folding_policy_rollout.py --no-launch --execute # terminal 2, repeatable

Usage:
    python scripts/box_folding_policy_rollout.py                       # dry run, nothing moves
    python scripts/box_folding_policy_rollout.py --execute             # arms move
    python scripts/box_folding_policy_rollout.py --execute --max-seconds 60 --replan-every 2
    python scripts/box_folding_policy_rollout.py --sim --allow-missing-cameras  # plumbing only
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import portal
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import bimanual_teleop_record as rec  # cameras, FK, gello launch/teardown
from box_folding_policy import (
    CONTROL_DT,
    DEFAULT_POLICY_ROOT,
    DEFAULT_TORCH_HOME,
    VIEW_ORDER,
    PolicyProcess,
    build_state,
    resize_to_policy,
    split_action,
)
from can_channels import channel_for

SIDES = ("left", "right")
_RPC_TIMEOUT_S = 2.0


@dataclass
class Args:
    # --- policy ---
    policy_root: str = DEFAULT_POLICY_ROOT
    """Downloaded HF folder box_folding_policy/e2e (checkpoint, norm_stats, planning package)."""
    checkpoint: Optional[str] = None
    """Checkpoint inside <policy_root>/checkpoints; default is the newest one."""
    device: str = "cuda:0"
    torch_home: str = DEFAULT_TORCH_HOME
    """Torch Hub cache holding the c-radio_v3-h weights."""
    flow_steps: int = 10
    """Euler steps for the flow integration. 5/10/20 score the same on the replay check."""
    radio_dtype: str = "float32"
    """float32 | float16 | bfloat16 for the frozen RADIO tower."""
    seed: Optional[int] = 0
    """Seed for the flow noise. Fixed by default: it makes a run reproducible and consecutive
    chunks agree slightly better in their overlap (0.0164 vs 0.0174 rad), at no accuracy cost."""

    # --- control ---
    execute: bool = False
    """Send commands to the arms. Off by default: the run is a dry run that only prints."""
    send_hz: float = 30.0
    """Command stream rate. The chunk is on a 10 fps timeline and is interpolated up to this."""
    replan_every: int = 3
    """Minimum ticks between plans (3 = 0.3 s, i.e. ~3 Hz). The 32-step chunk covers 3.2 s, so
    this sets how much of it is ever executed open-loop; a plan costs ~160-210 ms, so 2 also
    keeps up (5 Hz, ~77% GPU) while 8 leans much harder on the open-loop tail."""
    max_seconds: float = 120.0
    """Hard stop for the rollout."""
    max_plan_age: float = 8.0
    """Abort if the active chunk gets this old (s) -- planning has died. Between the chunk's own
    3.2 s horizon and this, the arms simply hold the chunk's last step instead of stopping: an
    abort tears down the follower processes, which disables the motors and drops the arms."""
    max_joint_vel: float = 1.5
    """Slew limit per arm joint, rad/s, applied to the streamed command."""
    max_gripper_vel: float = 3.0
    """Slew limit for the normalised gripper command, 1/s."""
    max_start_error: float = 0.8
    """Refuse to engage if the first predicted target is farther than this (rad) from the
    measured pose. Raise it only if you know why the policy is asking for a big first move."""
    park_on_exit: bool = True
    """On a normal finish (--max-seconds or q), slew both arms back to the pose they started in
    before the followers are torn down, so the arms end up where you put them rather than
    wherever the last chunk left them. Skipped after Ctrl-C and after an error: an interrupt is
    the operator saying stop, not "make one more move"."""
    park_seconds: float = 3.0
    """Time budget for that return move (it is slew-rate-limited like any other command)."""
    warmup_ticks: int = 0
    """Extra observation ticks before the first plan; the history needs 4 either way."""

    # --- robot ---
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = channel_for("follower_left")
    """LEFT follower CAN netdev (from scripts/can_map.conf)."""
    can_follower_right: str = channel_for("follower_right")
    """RIGHT follower CAN netdev (from scripts/can_map.conf)."""
    port_left: int = 1235
    port_right: int = 1234
    launch: bool = True
    """Spawn the two follower processes. --no-launch attaches to already-running ones."""
    sim: bool = False
    """Followers run in MuJoCo. Plumbing test only -- the policy needs the real cameras."""

    # --- cameras (same fields and defaults as the recorder, so the rig comes up identically) ---
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
    """Continue with fewer than three cameras. The policy is blind on the missing view -- for
    smoke-testing the loop, never for a real rollout."""

    # --- display ---
    display: bool = True
    """Preview window with the three policy-resolution views; q/Esc stops the run."""
    display_width: int = 1200
    log_every: int = 10
    """Print a status line every N ticks."""


@dataclass
class Plan:
    """One action chunk and the observation time it was predicted from."""

    chunk: np.ndarray  # [32, 14] absolute leader joint commands
    t0: float  # monotonic time of the observation that produced it
    latency_s: float

    def target(self, now: float) -> Tuple[np.ndarray, float]:
        """Linearly interpolated command for ``now``, plus how far into the chunk that is."""
        u = float(np.clip((now - self.t0) / CONTROL_DT, 0.0, self.chunk.shape[0] - 1))
        lo = int(np.floor(u))
        hi = min(lo + 1, self.chunk.shape[0] - 1)
        w = u - lo
        return (1.0 - w) * self.chunk[lo] + w * self.chunk[hi], u


class FollowerArm:
    """One follower server: read the measured joint pos, write absolute joint commands."""

    def __init__(self, name: str, port: int) -> None:
        self.name = name
        self._client = portal.Client(f"127.0.0.1:{port}")
        self.num_dofs = int(self._client.num_dofs().result(timeout=10.0))

    def joint_pos(self) -> np.ndarray:
        return np.asarray(self._client.get_joint_pos().result(timeout=_RPC_TIMEOUT_S), dtype=np.float64)

    def command(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(np.asarray(joint_pos, dtype=np.float64)).result(timeout=_RPC_TIMEOUT_S)

    def close(self) -> None:
        self._client.close(timeout=2.0)


class Ema:
    """Exponential moving average of a millisecond timing, for the status line."""

    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = float(alpha)
        self.value = 0.0

    def add(self, seconds: float) -> None:
        ms = seconds * 1e3
        self.value = ms if self.value == 0.0 else (1 - self._alpha) * self.value + self._alpha * ms


class FrameHistory:
    """The last ``length`` ticks of camera frames, as the policy process wants them.

    Frames are kept as uint8 here rather than encoded on arrival: only the 4 frames a plan
    actually reads need RADIO features, and on the real robot a plan runs about once a second.
    Each tick gets an id so the policy process can reuse features for frames it already saw --
    including the repeated first frame, which is how the training set left-clamped episode starts.
    """

    def __init__(self, length: int) -> None:
        self._buf: Deque[Tuple[int, np.ndarray]] = deque(maxlen=int(length))
        self._length = int(length)
        self._next_id = 0

    def push(self, frames_rgb: Dict[str, np.ndarray]) -> None:
        stack = np.stack([frames_rgb[v] for v in VIEW_ORDER])  # [V,H,W,3] uint8
        self._next_id += 1
        for _ in range(self._length if not self._buf else 1):
            self._buf.append((self._next_id, stack))

    @property
    def ready(self) -> bool:
        return len(self._buf) == self._length

    def snapshot(self) -> Tuple[List[int], np.ndarray]:
        return [fid for fid, _ in self._buf], np.stack([img for _, img in self._buf])


class SafetyLimiter:
    """Joint-limit clamp + slew-rate limit on the outgoing command of one arm."""

    def __init__(self, joint_range: np.ndarray, max_joint_vel: float, max_gripper_vel: float) -> None:
        self._lo = np.concatenate([joint_range[:, 0], [0.0]])
        self._hi = np.concatenate([joint_range[:, 1], [1.0]])
        self._step = np.array([max_joint_vel] * joint_range.shape[0] + [max_gripper_vel], dtype=np.float64)
        self.last: Optional[np.ndarray] = None

    def reset(self, measured: np.ndarray) -> None:
        self.last = np.asarray(measured, dtype=np.float64)[: self._lo.size].copy()

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        t = np.clip(np.asarray(target, dtype=np.float64)[: self._lo.size], self._lo, self._hi)
        if self.last is None:
            self.last = t.copy()
            return t
        delta = np.clip(t - self.last, -self._step * dt, self._step * dt)
        self.last = self.last + delta
        return self.last.copy()


def _joint_range(arm: str, version: int) -> np.ndarray:
    """``[n_arm_joints, 2]`` limits from the arm-only MJCF -- the same model the FK uses."""
    import mujoco

    from i2rt.robots.utils import ArmType

    model = mujoco.MjModel.from_xml_path(ArmType.from_string_name(arm).get_xml_path(version))
    return np.asarray(model.jnt_range[: model.nq].copy(), dtype=np.float64)


def _listening_pids(port: int) -> List[int]:
    """PIDs of this user's processes listening on ``port``, straight from /proc.

    No ``ss``/``lsof`` dependency: read the listening socket's inode out of /proc/net/tcp*, then
    find which process has it open. Processes owned by another user are invisible here, which is
    fine -- the followers always run as the caller.
    """
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A = TCP_LISTEN
                continue
            try:
                if int(fields[1].rsplit(":", 1)[1], 16) == int(port):
                    inodes.add(fields[9])
            except ValueError:
                continue
    if not inodes:
        return []
    pids: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    pids.append(int(entry.name))
                    break
        except OSError:
            continue
    return pids


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
    except OSError:
        return "<gone>"


def _describe_port(port: int) -> str:
    return " | ".join(f"pid {pid}: {_cmdline(pid)}" for pid in _listening_pids(port)) or "<unknown process>"


def _check_ports_free(ports: Dict[str, int]) -> None:
    """Refuse to launch onto a port something else already serves.

    Otherwise the follower this script spawns dies with ``Address already in use`` while the
    *stale* server keeps the port -- and since the readiness check is only "is the port
    connectable", the rollout then happily drives whatever that old process is attached to
    (a MuJoCo sim, or nothing at all) while the real arms sit unpowered.
    """
    busy = {side: port for side, port in ports.items() if _listening_pids(port)}
    if not busy:
        return
    detail = "\n".join(f"  {side} port {port} -- {_describe_port(port)}" for side, port in busy.items())
    raise SystemExit(
        "[error] follower port(s) already in use, so the followers this run launches would die on bind:\n"
        f"{detail}\n"
        "Kill the process(es) above, or attach to them instead with --no-launch (it checks that "
        "they match this run)."
    )


def _check_attached_followers(args: Args, ports: Dict[str, int]) -> None:
    """--no-launch: make sure the servers already on those ports are the ones we mean to drive."""
    channels = {"left": args.can_follower_left, "right": args.can_follower_right}
    for side, port in ports.items():
        pids = _listening_pids(port)
        if not pids:
            raise SystemExit(
                f"[error] nothing is listening on the {side} follower port {port}; start the followers "
                "(scripts/run_policy_followers.sh) or drop --no-launch"
            )
        line = _cmdline(pids[0])
        if "minimum_gello" not in line:
            raise SystemExit(f"[error] {side} port {port} is served by something else -- {line}")
        if ("--sim" in line.split()) != bool(args.sim):
            kind = "a --sim follower" if "--sim" in line.split() else "a real-hardware follower"
            raise SystemExit(
                f"[error] {side} port {port} is served by {kind}, which does not match this run "
                f"({'--sim' if args.sim else 'real hardware'}) -- {line}"
            )
        if not args.sim and channels[side] not in line.split():
            raise SystemExit(
                f"[error] {side} port {port} is served by a follower on a different CAN bus (expected "
                f"{channels[side]}) -- {line}"
            )
        print(f"[attach] {side} follower on port {port}: pid {pids[0]}")


def _launch_followers(args: Args, procs: List["subprocess.Popen[bytes]"], shutdown: rec.ShutdownRequest) -> None:
    """The follower half of the recorder's launch: two gello processes, no leaders."""
    ports = {"left": args.port_left, "right": args.port_right}
    _check_ports_free(ports)
    if not args.sim:
        for interface in (args.can_follower_left, args.can_follower_right):
            rec._check_can_interface(interface)
        print(f"[can] follower-left {args.can_follower_left}, follower-right {args.can_follower_right}")

    common = ["--arm", args.arm, "--version", str(args.version)]
    started: Dict[int, subprocess.Popen[bytes]] = {}
    for can, port in ((args.can_follower_right, args.port_right), (args.can_follower_left, args.port_left)):
        cmd = [*common, "--can_channel", can, "--gripper", args.follower_gripper, "--server_port", str(port)]
        if args.sim:
            cmd.append("--sim")
        started[port] = rec._spawn_gello(cmd)
        procs.append(started[port])
    for port, proc in started.items():
        rec._wait_for_port(port, proc=proc, shutdown=shutdown)
        # Belt and braces: the port being connectable only proves *someone* serves it.
        if proc.poll() is not None:
            raise RuntimeError(f"follower for port {port} exited with code {proc.returncode} right after binding")
        if proc.pid not in _listening_pids(port):
            raise RuntimeError(
                f"port {port} is served by another process ({_describe_port(port)}), not the follower "
                f"we launched (pid {proc.pid})"
            )
    print("[launch] follower servers up")


def _read_state(arms: Dict[str, FollowerArm], kin: rec.EEFKinematics) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """The 28-D policy state and the per-side measured joint vectors behind it."""
    pos = {side: arms[side].joint_pos() for side in SIDES}
    fk = {side: kin.fk(pos[side]) for side in SIDES}
    state = build_state(pos["left"], pos["right"], fk["left"][0], fk["left"][1], fk["right"][0], fk["right"][1])
    return state, pos


def _grab_frames(rig: rec.CameraRig, last: Dict[str, np.ndarray], allow_missing: bool) -> Dict[str, np.ndarray]:
    """One 240x320 RGB frame per view; a view that has not delivered yet repeats its last frame."""
    snap = rig.snapshot()
    out: Dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        if view in snap:
            out[view] = resize_to_policy(snap[view][0], bgr=True)
            last[view] = out[view]
        elif view in last:
            out[view] = last[view]
        elif allow_missing:
            out[view] = np.zeros((240, 320, 3), np.uint8)
        else:
            raise RuntimeError(f"camera '{view}' has delivered no frame")
    return out


def _preview(frames: Dict[str, np.ndarray], lines: List[str], width: int) -> int:
    """Draw the three policy-resolution views with a status overlay; returns the pressed key."""
    strip = np.concatenate([cv2.cvtColor(frames[v], cv2.COLOR_RGB2BGR) for v in VIEW_ORDER], axis=1)
    scale = width / strip.shape[1]
    strip = cv2.resize(strip, (width, int(strip.shape[0] * scale)))
    for i, view in enumerate(VIEW_ORDER):
        x = int(i * width / len(VIEW_ORDER)) + 8
        cv2.putText(strip, view, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    panel = np.zeros((26 * len(lines) + 12, width, 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("box_folding policy", np.concatenate([strip, panel], axis=0))
    return cv2.waitKey(1) & 0xFF


def _park(
    arms: Dict[str, "FollowerArm"],
    limiters: Dict[str, "SafetyLimiter"],
    home: Dict[str, np.ndarray],
    *,
    seconds: float,
    send_hz: float,
) -> None:
    """Slew both arms back to ``home`` before teardown. Same rate limit as the rollout itself."""
    print(f"[park] returning to the start pose over up to {seconds:.1f}s")
    dt = 1.0 / float(send_hz)
    deadline = time.monotonic() + float(seconds)
    while time.monotonic() < deadline:
        done = True
        for side, arm in arms.items():
            cmd = limiters[side].step(home[side], dt)
            arm.command(cmd)
            if float(np.max(np.abs(cmd - home[side][: cmd.size]))) > 1e-3:
                done = False
        if done:
            break
        time.sleep(dt)


def run(args: Args) -> None:
    procs: List["subprocess.Popen[bytes]"] = []
    shutdown = rec.ShutdownRequest(procs)
    rig: Optional[rec.CameraRig] = None
    arms: Dict[str, FollowerArm] = {}
    policy: Optional[PolicyProcess] = None

    try:
        ports = {"left": args.port_left, "right": args.port_right}
        if args.launch:
            _launch_followers(args, procs, shutdown)
        else:
            _check_attached_followers(args, ports)
        arms = {"left": FollowerArm("left", args.port_left), "right": FollowerArm("right", args.port_right)}
        print(f"[robot] followers connected ({', '.join(f'{s}:{a.num_dofs} dof' for s, a in arms.items())})")

        kin = rec.EEFKinematics(args.arm, args.version)
        limits = _joint_range(args.arm, args.version)
        limiters = {s: SafetyLimiter(limits, args.max_joint_vel, args.max_gripper_vel) for s in SIDES}

        rig = rec._open_camera_rig(args)
        deadline = time.monotonic() + 10.0
        while not rig.all_fresh() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not rig.all_fresh() and not args.allow_missing_cameras:
            raise RuntimeError("cameras did not deliver frames (check USB bandwidth and cables)")

        print("[policy] starting the policy process (loads the checkpoint and RADIO) ...")
        policy = PolicyProcess(
            policy_root=args.policy_root,
            checkpoint=args.checkpoint,
            device=args.device,
            flow_steps=args.flow_steps,
            radio_dtype=args.radio_dtype,
            torch_home=args.torch_home,
            seed=args.seed,
        )
        info = policy.start()
        history = FrameHistory(int(info["history_len"]))
        print(f"[policy] {policy.description}")
        print(f"[policy] instruction: {info['instruction']!r}")
        print(
            f"[mode] {'EXECUTE -- the arms will move' if args.execute else 'DRY RUN -- no commands sent'}"
            f" for up to {args.max_seconds:.0f}s (--max-seconds), then the run ends"
        )

        last_frames: Dict[str, np.ndarray] = {}
        send_dt = 1.0 / float(args.send_hz)
        start = time.monotonic()
        next_tick = start
        next_send = start
        tick = 0
        engaged = False
        active: Optional[Plan] = None
        stop_reason = "max-seconds"
        last_state = np.zeros(28, np.float32)
        last_stale_warn = 0.0
        timing = {k: Ema() for k in ("grab", "state", "send")}

        home: Dict[str, np.ndarray] = {}
        for side in SIDES:
            home[side] = arms[side].joint_pos()
            limiters[side].reset(home[side])

        while True:
            shutdown.raise_if_requested()
            now = time.monotonic()
            if now - start > args.max_seconds:
                break

            # ---- observation tick (10 Hz) -------------------------------------------------
            if now >= next_tick:
                t_obs = next_tick
                next_tick += CONTROL_DT
                mark = time.perf_counter()
                frames = _grab_frames(rig, last_frames, args.allow_missing_cameras or args.sim)
                timing["grab"].add(time.perf_counter() - mark)
                history.push(frames)
                tick += 1

                # Start the next plan as soon as the policy process is free and the interval has
                # passed, instead of only on every Nth tick: on the real robot a plan can take
                # longer than the interval, and waiting for the next multiple then added a whole
                # extra period of staleness on top of the overrun.
                due = t_obs - policy.last_request_t0 >= args.replan_every * CONTROL_DT
                if history.ready and not policy.busy and tick > args.warmup_ticks and (active is None or due):
                    # The state is read here, not every tick: the policy conditions on the newest
                    # state only, and every RPC steals CPU from the control loop.
                    mark = time.perf_counter()
                    last_state, _ = _read_state(arms, kin)
                    timing["state"].add(time.perf_counter() - mark)
                    ids, pixels = history.snapshot()
                    policy.request(t_obs, pixels, last_state, ids)

                if args.display:
                    plan_age = "--" if active is None else f"{now - active.t0:.2f}s"
                    lines = [
                        f"tick {tick}  t {now - start:5.1f}s  plans {policy.n_plans}  plan age {plan_age}"
                        f"  {'EXECUTE' if args.execute else 'DRY RUN'}"
                    ]
                    for side in SIDES:
                        cmd = limiters[side].last
                        shown = "--" if cmd is None else np.array2string(cmd, precision=2, suppress_small=True)
                        lines.append(f"cmd {side[0].upper()} {shown}")
                    key = _preview(frames, lines, args.display_width)
                    if key in (ord("q"), 27):
                        stop_reason = "operator quit"
                        break

                if tick % max(1, args.log_every) == 0:
                    lat = f"{active.latency_s * 1e3:.0f}" if active else "--"
                    print(
                        f"[tick {tick:4d}] t={now - start:5.1f}s plans={policy.n_plans} plan={lat} ms "
                        f"grab={timing['grab'].value:.0f} state={timing['state'].value:.0f} "
                        f"send={timing['send'].value:.1f} ms engaged={engaged}"
                    )

            # ---- newest plan --------------------------------------------------------------
            done = policy.poll()
            if done is not None:
                t0, chunk, latency = done
                active = Plan(chunk=chunk, t0=t0, latency_s=latency)
                if not engaged:
                    target, _ = active.target(time.monotonic())
                    err = {
                        s: float(np.max(np.abs(split_action(target)[i][:6] - arms[s].joint_pos()[:6])))
                        for i, s in enumerate(SIDES)
                    }
                    worst = max(err.values())
                    print(f"[engage] first target is {worst:.3f} rad from the measured pose ({err})")
                    if worst > args.max_start_error:
                        raise RuntimeError(
                            f"first target is {worst:.3f} rad away (limit {args.max_start_error:.2f}); the "
                            "policy does not recognise this scene -- move the arms to a demo-like start "
                            "pose, or raise --max-start-error if you are sure"
                        )
                    engaged = True

            # ---- send (30 Hz) --------------------------------------------------------------
            if active is not None and engaged and time.monotonic() >= next_send:
                now = time.monotonic()
                next_send = max(now, next_send + send_dt)
                age = now - active.t0
                horizon_s = CONTROL_DT * active.chunk.shape[0]
                if age > args.max_plan_age:
                    stop_reason = f"no fresh plan for {age:.1f}s -- planning has stopped"
                    break
                if age > horizon_s and now - last_stale_warn > 1.0:
                    last_stale_warn = now
                    print(
                        f"[warn] active plan is {age:.1f}s old (> {horizon_s:.1f}s horizon) -- holding its "
                        "last step; raise --replan-every or lower --flow-steps if this persists"
                    )
                target, _ = active.target(now)
                mark = time.perf_counter()
                for side, part in zip(SIDES, split_action(target), strict=True):
                    cmd = limiters[side].step(part, send_dt)
                    if args.execute:
                        arms[side].command(cmd)
                timing["send"].add(time.perf_counter() - mark)

            # Sleep to the next scheduled event rather than spinning on a fixed 2 ms poll. (It is
            # not what limits plan latency -- that is GPU time shared with the 10 Hz RADIO encodes,
            # measured at ~450 ms per plan while the loop runs vs ~130 ms for a plan on its own --
            # but there is no reason to wake 500 times a second to find out nothing is due.)
            due_at = next_tick if active is None or not engaged else min(next_tick, next_send)
            time.sleep(float(np.clip(due_at - time.monotonic(), 0.0, 0.02)))

        print(f"[stop] {stop_reason}")
        if args.execute:
            if args.park_on_exit and engaged:
                _park(arms, limiters, home, seconds=args.park_seconds, send_hz=args.send_hz)
            if args.launch:
                print(
                    "[stop] the follower processes are about to exit, which sets every motor torque "
                    "to zero -- the arms go limp and drop. To keep them powered between runs, start "
                    "the followers separately (scripts/run_policy_followers.sh) and pass --no-launch."
                )
            else:
                print("[stop] the followers keep running, so the arms hold this pose")

    except KeyboardInterrupt:
        # Either a real Ctrl-C or ShutdownRequest.raise_if_requested() acting on the first signal.
        print("[stop] interrupted -- the arms hold their last command")
    finally:
        # Nothing on the way out may hang; the step that matters (killing the followers, which
        # drive the arms) is last, exactly as in the recorder.
        if args.display:
            rec._cleanup_step("close display", cv2.destroyAllWindows)
        for side, arm in arms.items():
            rec._cleanup_step(f"close {side} client", arm.close)
        if rig is not None:
            rec._cleanup_step("stop cameras", rig.stop)
        if policy is not None:
            rec._cleanup_step("stop policy process", policy.close, timeout=10.0)
        if procs:
            rec._cleanup_step("terminate followers", lambda: rec._terminate(procs), timeout=15.0)


def main(args: Args) -> None:
    if args.replan_every < 1:
        raise SystemExit("[error] --replan-every must be >= 1")
    if not args.execute:
        print("[warn] dry run: pass --execute to actually command the arms")
    run(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
