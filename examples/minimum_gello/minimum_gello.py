import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.utils import RateRecorder, override_log_level

DEFAULT_ROBOT_PORT = 11333
_MAX_DOFS = 16
"""Upper bound on robot DOF. The shared pos array is sized to this; the active
slice is exposed through an int Value the worker fills after constructing the robot."""

_WORKER_LOOP_PERIOD_S = 0.002
"""~500 Hz pacing for worker loops whose reads hit cached state (get_joint_pos /
get_info return instantly). Without it these loops peg a CPU core; the actual YAM
hardware/control thread updates at ~250 Hz, well below this cap, so no data is lost."""

_RPC_TIMEOUT_S = 2.0
"""Per-call ceiling for every follower RPC. Generous next to a sub-millisecond loopback
round trip: it is a liveness check, not a latency budget."""

_FOLLOWER_STARTUP_TIMEOUT_S = 60.0
"""How long the leader retries its first follower read before giving up. Covers a follower
that is still coming up (gripper calibration) when the leader starts."""

_RPC_ERRORS_BEFORE_RECONNECT = 20
"""Consecutive failed RPCs (~4 s at _RPC_TIMEOUT_S, less when they fail fast) before the
leader redials the follower. High enough to ride out a blip, low enough that a connection
which has stopped answering is replaced while the operator is still holding the leader."""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class ClientRobot(Robot):
    """A simple client for a leader robot.

    Every call is bounded and consumes its future. portal gives us two blocking paths that
    never time out and never log: ``Future.result()`` defaults to waiting forever, and
    ``Client.call`` parks in an untimed ``cond.wait`` loop once ``maxinflight`` (16) futures
    are still pending (portal/client.py:66). A fire-and-forget command whose response is lost
    therefore wedges the *caller* permanently and silently -- which is what froze both
    followers mid-teleop. Waiting for each response keeps at most one request in flight, so
    that limit is unreachable, and a stall surfaces as a TimeoutError the caller can act on."""

    def __init__(
        self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1", timeout: float = _RPC_TIMEOUT_S
    ) -> None:
        self._addr = f"{host}:{port}"
        self._timeout = timeout
        self._client = portal.Client(self._addr)

    def _connected(self) -> None:
        """Raise instead of blocking when the socket is down.

        The third untimed path in portal: ``Client.call`` sends before it ever builds a future,
        and that send waits on the connection with no timeout at all, so a follower that goes
        away parks the caller indefinitely. ``connect`` is the bounded form of the same wait."""
        if not self._client.connect(timeout=self._timeout):
            raise TimeoutError(f"not connected to {self._addr}")

    def num_dofs(self) -> int:
        self._connected()
        return self._client.num_dofs().result(timeout=self._timeout)

    def get_joint_pos(self) -> np.ndarray:
        self._connected()
        return self._client.get_joint_pos().result(timeout=self._timeout)

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._connected()
        self._client.command_joint_pos(joint_pos).result(timeout=self._timeout)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self._connected()
        self._client.command_joint_state(joint_state).result(timeout=self._timeout)

    def get_observations(self) -> Dict[str, np.ndarray]:
        self._connected()
        return self._client.get_observations().result(timeout=self._timeout)

    def reconnect(self) -> None:
        """Drop the socket and dial again, on a fresh Client.

        A timed-out request stays in ``Client.futures`` (only the receive path pops it), so a
        connection that has stopped answering never recovers on its own. ``close`` fails every
        pending future and clears that dict, so the replacement starts clean."""
        try:
            self.close()
        except Exception as e:  # a wedged socket must not stop us from dialing again
            logging.warning(f"[{self._addr}] closing the stale client failed: {e}")
        self._client = portal.Client(self._addr)

    def close(self) -> None:
        """Tear down the underlying portal client (background loop thread + socket)."""
        self._client.close(timeout=1.0)

    def __enter__(self) -> "ClientRobot":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, np.ndarray]:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        if encoder_obs is None:
            # Populated only after the first CAN read in the background driver thread.
            raise RuntimeError("CAN device states not ready yet")
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


# ---------------------------------------------------------------------------
# Args + Resources
# ---------------------------------------------------------------------------


@dataclass
class Args:
    arm: Literal["yam", "yam_pro", "yam_ultra", "big_yam", "no_arm"] = "yam"
    version: int = 1
    """Arm hardware revision (the v<N> model/config version)."""
    gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "flexible_4310", "yam_teaching_handle", "no_gripper"
    ] = "yam_teaching_handle"
    mode: Literal["follower", "leader", "visualizer_local", "visualizer_remote"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"
    bilateral_kp: float = 0.0
    sim: bool = False
    """Use SimRobot instead of real hardware."""
    ee_mass: Optional[float] = None
    """Override end-effector (link_6) mass in kg for gravity compensation. Defaults to the value in the XML."""
    joint_offsets: List[float] = field(default_factory=list)
    """Six per-joint zero corrections in RADIANS, added to what this arm reports. Left empty
    (the default) the corrections are looked up by CAN channel in scripts/arm_offsets.conf, so
    every launcher gets them without passing anything; pass six numbers to override that, or
    six zeros to opt out. See _resolve_joint_offsets."""


@dataclass
class Resources:
    """Resources accumulated during mode setup; ``cleanup`` drains them on exit.

    Mirrors the ``server_processes`` accumulator pattern from xdof/envs/launch.py:
    every spawn helper appends here so a single ``cleanup(resources)`` call tears
    down the mode regardless of how it exits."""

    processes: List[portal.Process] = field(default_factory=list)
    pos_shared: Optional[portal.SharedArray] = None
    stop_event: Optional[Any] = None


def _resolve_joint_offsets(args: "Args") -> Optional[np.ndarray]:
    """Zero-calibration correction for the arm this process is driving, in radians.

    Explicit ``--joint_offsets`` wins. Otherwise the arm is identified by its CAN channel
    through scripts/can_map.conf and its correction read from scripts/arm_offsets.conf --
    so a hand-edited table fixes every launcher at once instead of each one having to
    thread six numbers through. That table lives in the deployment's scripts/ directory,
    not in the library, so a checkout without it (or a --sim run) simply gets no correction.
    """
    if args.joint_offsets:
        offsets = np.asarray(args.joint_offsets, dtype=float)
        if offsets.shape != (6,):
            raise ValueError(f"--joint_offsets takes 6 values (one per arm joint), got {offsets.shape[0]}")
        return offsets
    if args.sim:
        return None  # sim has no motor zeros to disagree about
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from arm_offsets import offsets_for_channel
    except Exception as e:
        logging.debug(f"no per-arm zero-calibration table available ({e}); using zero offsets")
        return None
    offsets = offsets_for_channel(args.can_channel)
    if np.any(offsets):
        logging.info(f"{args.can_channel}: zero correction (deg) {np.round(np.rad2deg(offsets), 3).tolist()}")
    return offsets


def _spawn_into(processes: List[portal.Process], proc: portal.Process) -> portal.Process:
    """Start a portal.Process and append it to the cleanup pool."""
    proc.start()
    processes.append(proc)
    return proc


# ---------------------------------------------------------------------------
# Background workers. Each runs in its own portal.Process so the foreground
# loop (web IO server, web IO client, or MuJoCo viewer) never blocks on
# hardware or RPC.
# ---------------------------------------------------------------------------


def _yam_polling_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    n_dofs_value: Any,
    cmd_queue: Any,
    stop_event: Any,
    rate_name: str,
    enable_auto_recovery: bool = False,
) -> None:
    """Owns the YAM hardware. Streams joint_pos into ``pos_shared`` at hardware
    rate; if ``cmd_queue`` is provided, drains pending commands onto the YAM.
    Publishes the actual DOF count via ``n_dofs_value`` once the YAM is up.

    ``enable_auto_recovery`` is forwarded to the motor chain: with it off, one motor error
    kills the chain's control loop thread while this process keeps serving stale cached
    state (see run_follower)."""
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    yam = get_yam_robot(
        channel=args.can_channel,
        arm_type=arm_type,
        version=args.version,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=args.sim,
        enable_auto_recovery=enable_auto_recovery,
        joint_offsets=_resolve_joint_offsets(args),
    )
    n_dofs_value.value = yam.num_dofs()
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    while not stop_event.is_set():
        if cmd_queue is not None:
            # Drain to the newest command and drop any stale backlog. Commands arrive (over RPC)
            # at least as fast as this loop actuates them, so replaying every queued command in
            # FIFO order would make the follower lag further and further behind. Mirrors the
            # leader io loop's drain-to-newest (_run_leader_io_loop).
            cmd = None
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
            if cmd is not None:
                yam.command_joint_pos(cmd)
        try:
            pos = yam.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            rate.track()
        except Exception as e:
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)
        time.sleep(_WORKER_LOOP_PERIOD_S)  # pace: get_joint_pos is a cached read, so this loop would otherwise spin


def _rpc_polling_worker(
    server_port: int,
    server_host: str,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    rate_name: str,
) -> None:
    """Polls a remote follower over portal RPC and caches the latest joint_pos
    in ``pos_shared`` so the foreground viewer reads from memory."""
    override_log_level()
    client = ClientRobot(server_port, host=server_host)
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    while not stop_event.is_set():
        try:
            pos = client.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            rate.track()
        except Exception as e:
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)


def _leader_control_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    cmd_queue: Any,
    stop_event: Any,
) -> None:
    """Owns the leader YAM hardware and runs the bilateral control
    loop. Reads the latest follower pos from ``pos_shared`` (written by main's
    foreground web-IO loop) and pushes commands for the follower into
    ``cmd_queue`` (drained by main and forwarded over RPC). Never touches the
    network itself."""
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    yam = get_yam_robot(
        channel=args.can_channel,
        arm_type=arm_type,
        version=args.version,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=args.sim,
        joint_offsets=_resolve_joint_offsets(args),
    )
    robot = YAMLeaderRobot(yam)
    robot_current_kp = yam._kp

    control_rate = RateRecorder(name="yam-leader control", report_interval=1.0)
    control_rate.start()

    # The encoder device states are populated only after the first CAN read, so get_info() can
    # raise on startup. Retry until the first reading lands before entering the control loop.
    while not stop_event.is_set():
        try:
            current_joint_pos, current_button = robot.get_info()
            break
        except RuntimeError:
            time.sleep(0.05)
    else:
        return  # stop requested before the encoder became ready
    current_follower_joint_pos = pos_shared.array.copy()
    logging.info(f"Current leader joint pos: {current_joint_pos}")
    logging.info(f"Current follower joint pos: {current_follower_joint_pos}")

    def slow_move(joint_pos: np.ndarray, duration: float = 3.0) -> None:
        start_pos = pos_shared.array.copy()
        steps = 100
        dt = duration / steps
        for i in range(steps):
            cmd = joint_pos * i / steps + start_pos * (1 - i / steps)
            cmd_queue.put(cmd)
            time.sleep(dt)

    synchronized = False
    while not stop_event.is_set():
        control_rate.track()
        current_joint_pos, current_button = robot.get_info()
        if current_button[0] > 0.5:
            if not synchronized:
                robot.update_kp_kd(kp=robot_current_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                robot.command_joint_pos(current_joint_pos[:6])
                slow_move(current_joint_pos)
            else:
                logging.info("clear bilateral pd")
                robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                current_follower_joint_pos = pos_shared.array.copy()
                robot.command_joint_pos(current_follower_joint_pos[:6])
            synchronized = not synchronized
            while current_button[0] > 0.5:
                time.sleep(0.03)
                current_joint_pos, current_button = robot.get_info()

        current_follower_joint_pos = pos_shared.array.copy()

        if synchronized:
            cmd_queue.put(current_joint_pos)
            # bilateral feedback to the leader, proportional to bilateral_kp
            robot.command_joint_pos(current_follower_joint_pos[:6])

        # Pace the loop: get_info reads cached state, so without this the worker spins a core and
        # competes with the in-process robot_server control thread for the GIL, adding jitter.
        time.sleep(_WORKER_LOOP_PERIOD_S)


# ---------------------------------------------------------------------------
# Foreground helpers and loops
# ---------------------------------------------------------------------------


def _wait_for_dof(n_dofs_value: Any, timeout_s: float = 30.0) -> int:
    """Block until the background worker publishes the active DOF count."""
    deadline = time.time() + timeout_s
    while n_dofs_value.value == 0:
        if time.time() > deadline:
            raise TimeoutError("worker did not publish DOF count within timeout")
        time.sleep(0.05)
    return int(n_dofs_value.value)


def _put_latest_command(cmd_queue: Any, item: np.ndarray) -> None:
    """Non-blocking latest-wins enqueue onto a (bounded) command queue.

    Never blocks the caller: if the queue is full (the hardware worker has not drained it
    yet), drop the stale item and replace it with the newest, so the follower always actuates
    the freshest target. A setpoint stream is latest-wins, not a work queue."""
    while True:
        try:
            cmd_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                cmd_queue.get_nowait()
            except queue.Empty:
                pass


def _build_follower_server(
    server_port: int,
    n_dofs_value: Any,
    pos_shared: "portal.SharedArray",
    cmd_queue: Any,
) -> portal.Server:
    """Build the portal.Server that exposes the follower YAM over RPC. Returned
    unstarted; caller does ``server.start()`` to block."""
    io_rate = RateRecorder(name=f"ServerRobot[{server_port}] io", report_interval=1.0)
    io_rate.start()
    io_lock = threading.Lock()

    def _tracked(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            with io_lock:
                io_rate.track()
            return fn(*a, **kw)

        return wrapper

    def _get_pos() -> np.ndarray:
        return pos_shared.array[: n_dofs_value.value].copy()

    # Latest command received over RPC, kept so observers (e.g. an episode recorder) can log
    # the leader's target as the action instead of the follower's lagged measured position.
    last_cmd_lock = threading.Lock()
    last_cmd: Dict[str, Any] = {"pos": None, "time": 0.0}

    def _remember_cmd(joint_pos: np.ndarray) -> None:
        with last_cmd_lock:
            last_cmd["pos"] = joint_pos
            last_cmd["time"] = time.time()

    def _cmd_pos(joint_pos: np.ndarray) -> None:
        arr = np.asarray(joint_pos)
        _remember_cmd(arr)
        _put_latest_command(cmd_queue, arr)

    def _cmd_state(state: Dict[str, np.ndarray]) -> None:
        key = "position" if "position" in state else "pos"
        arr = np.asarray(state[key])
        _remember_cmd(arr)
        _put_latest_command(cmd_queue, arr)

    def _get_obs() -> Dict[str, np.ndarray]:
        return {"joint_pos": _get_pos()}

    def _get_last_command() -> Dict[str, np.ndarray]:
        """Last commanded joint pos and its wall-clock receive time. An empty ``pos``
        means no command has arrived yet (leader never engaged)."""
        with last_cmd_lock:
            pos, t = last_cmd["pos"], last_cmd["time"]
        if pos is None:
            return {"pos": np.zeros(0, dtype=np.float64), "time": np.float64(0.0)}
        return {"pos": pos.astype(np.float64), "time": np.float64(t)}

    server = portal.Server(server_port)
    logging.info(f"Robot Server Binding to {server_port}")
    server.bind("num_dofs", _tracked(lambda: n_dofs_value.value))
    server.bind("get_joint_pos", _tracked(_get_pos))
    server.bind("command_joint_pos", _tracked(_cmd_pos))
    server.bind("command_joint_state", _tracked(_cmd_state))
    server.bind("get_observations", _tracked(_get_obs))
    server.bind("get_last_command", _tracked(_get_last_command))
    return server


def _run_leader_io_loop(
    client_robot: ClientRobot,
    cmd_queue: Any,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    io_rate: RateRecorder,
) -> None:
    """Foreground loop for leader mode: drain commands from the bilateral
    control worker onto the follower over RPC, and pull the follower's latest
    joint_pos into ``pos_shared`` so the worker can read it for bilateral
    feedback."""
    consecutive_errors = 0
    while not stop_event.is_set():
        # Drain to the newest command and drop any stale backlog: the control worker can enqueue
        # faster than each blocking RPC round-trip drains, so replaying every queued command would
        # make the follower lag further and further behind.
        cmd = None
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break
        try:
            if cmd is not None:
                client_robot.command_joint_pos(cmd)
            pos = client_robot.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            io_rate.track()
            consecutive_errors = 0
        except Exception as e:
            # Ride out a transient follower/network blip. Without this the exception escapes to
            # run_leader, whose finally: cleanup() calls proc.kill() on the control worker while the
            # leader arm is energized in bilateral PD — an abrupt loss of control of a powered arm.
            consecutive_errors += 1
            logging.error(f"[yam-leader web-port io] error ({consecutive_errors} in a row): {e}")
            if consecutive_errors >= _RPC_ERRORS_BEFORE_RECONNECT:
                # Not a blip: this connection has stopped answering, and it will not resume.
                logging.error("[yam-leader web-port io] follower unresponsive, reconnecting")
                client_robot.reconnect()
                consecutive_errors = 0
            time.sleep(0.1)
        # Pace the loop. Unpaced it ran ~9.5 kHz, i.e. ~19k RPC/s at a follower server with a
        # single worker thread — 20x more than the ~410 Hz control worker produces setpoints at,
        # and enough pressure on portal's flow control that one stall wedged it for good.
        time.sleep(_WORKER_LOOP_PERIOD_S)


def _run_viewer_loop(
    arm_type: ArmType,
    gripper_type: GripperType,
    version: int,
    get_pos_fn: Callable[[], np.ndarray],
) -> None:
    """Foreground MuJoCo viewer. Pulls joint_pos from ``get_pos_fn`` each frame
    — typically a snapshot of a SharedArray populated by a background worker."""
    xml_path = combine_arm_and_gripper_xml(arm_type, gripper_type, version=version)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt: float = 0.01
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            joint_pos = get_pos_fn()
            nq = model.nq
            n = min(len(joint_pos), nq)
            data.qpos[:n] = joint_pos[:n]

            for j in range(model.njnt):
                adr = model.jnt_qposadr[j]
                if adr >= n:
                    continue
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                    lo, hi = model.jnt_range[j]
                    data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

            for i in range(model.neq):
                if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                adr1 = model.jnt_qposadr[model.eq_obj1id[i]]
                adr2 = model.jnt_qposadr[model.eq_obj2id[i]]
                coef = model.eq_data[i, :5]
                data.qpos[adr2] = np.polyval(coef[::-1], data.qpos[adr1])

            mujoco.mj_forward(model, data)
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(resources: Resources) -> None:
    """Tear down everything accumulated into ``resources``. Mirrors
    xdof/envs/launch.py's ``cleanup``: signal workers to stop, kill their
    processes, then release shared memory."""
    logging.info("exiting")
    if resources.stop_event is not None:
        resources.stop_event.set()
    for proc in resources.processes:
        proc.kill()
    if resources.pos_shared is not None:
        resources.pos_shared.close()
    logging.info("Processes terminated.")


# ---------------------------------------------------------------------------
# Per-mode runners. Each owns its own setup → foreground loop → cleanup, the
# same shape as ``main`` in xdof/envs/launch.py: build dependencies, call the
# control loop, wrap in try/finally with a single ``cleanup``.
# ---------------------------------------------------------------------------


def run_follower(args: Args) -> None:
    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    spawn = partial(_spawn_into, resources.processes)

    n_dofs_value = portal.mp.Value("i", 0)
    # maxsize=1 so the command queue holds only the latest setpoint; _cmd_pos/_cmd_state drop
    # stale commands on overflow (see _build_follower_server). A setpoint stream is latest-wins,
    # not a work queue — bounding it prevents any backlog from accumulating on this hop.
    cmd_queue = portal.mp.Queue(maxsize=1)

    spawn(
        portal.Process(
            _yam_polling_worker,
            args,
            resources.pos_shared,
            n_dofs_value,
            cmd_queue,
            resources.stop_event,
            "follower yam-hardware",
            # A follower error must not fail fast: the CAN frames only ever leave the motor
            # chain's own control-loop thread, and that thread dies on the first motor error
            # -- silently, since this process keeps serving the last cached joint_pos over RPC
            # and accepting commands nobody sends. The arm then holds its last MIT setpoint and
            # freezes mid-teleop. Let the chain clean+re-enable errored motors instead.
            True,  # enable_auto_recovery
            name="follower-yam-control",
        )
    )
    n_dofs = _wait_for_dof(n_dofs_value)
    logging.info(f"Follower YAM ready, num_dofs={n_dofs}")

    server = _build_follower_server(args.server_port, n_dofs_value, resources.pos_shared, cmd_queue)

    try:
        server.start()  # blocks
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("follower mode failed")
        raise
    finally:
        cleanup(resources)


def run_leader(args: Args) -> None:
    client_robot = ClientRobot(args.server_port, host=args.server_host)
    # Retry: the RPCs are timeout-bounded now, and the follower may still be calibrating its
    # gripper when the leader starts, which takes longer than one timeout.
    deadline = time.time() + _FOLLOWER_STARTUP_TIMEOUT_S
    while True:
        try:
            initial_follower_pos = client_robot.get_joint_pos()
            break
        except TimeoutError:
            if time.time() > deadline:
                raise TimeoutError(
                    f"follower on {args.server_host}:{args.server_port} did not answer within "
                    f"{_FOLLOWER_STARTUP_TIMEOUT_S:.0f}s"
                ) from None
            logging.info("waiting for the follower to answer...")
    logging.info(f"Initial follower joint pos: {initial_follower_pos}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    resources.pos_shared.array[:] = initial_follower_pos
    cmd_queue = portal.mp.Queue()

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _leader_control_worker,
            args,
            resources.pos_shared,
            cmd_queue,
            resources.stop_event,
            name="yam-leader-control",
        )
    )

    io_rate = RateRecorder(name="yam-leader web-port io", report_interval=1.0)
    io_rate.start()
    try:
        _run_leader_io_loop(client_robot, cmd_queue, resources.pos_shared, resources.stop_event, io_rate)
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("leader mode failed")
        raise
    finally:
        cleanup(resources)


def run_visualizer_local(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    n_dofs_value = portal.mp.Value("i", 0)

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _yam_polling_worker,
            args,
            resources.pos_shared,
            n_dofs_value,
            None,  # no cmd_queue — viewer doesn't send commands
            resources.stop_event,
            "viz-local yam-read",
            name="viz-local-yam",
        )
    )
    n_dofs = _wait_for_dof(n_dofs_value)
    logging.info(f"Viz YAM ready, num_dofs={n_dofs}")

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, args.version, get_pos)
    except Exception:
        logging.exception("visualizer_local mode failed")
        raise
    finally:
        cleanup(resources)


def run_visualizer_remote(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    with ClientRobot(args.server_port, host=args.server_host) as bootstrap:
        n_dofs = bootstrap.num_dofs()
    logging.info(f"Remote follower num_dofs={n_dofs}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _rpc_polling_worker,
            args.server_port,
            args.server_host,
            resources.pos_shared,
            resources.stop_event,
            "viz-remote rpc-poll",
            name="viz-remote-rpc",
        )
    )

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, args.version, get_pos)
    except Exception:
        logging.exception("visualizer_remote mode failed")
        raise
    finally:
        cleanup(resources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    # Make RateRecorder's logging.info output (and our own logging calls) visible.
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)

    if arm_type == ArmType.NO_ARM:
        # Every mode here needs an arm model (real YAM for follower/leader, an XML for the
        # visualizers); NO_ARM has no XML path and otherwise crashes deep inside the utils.
        raise ValueError("--arm 'no_arm' is not supported by minimum_gello; all modes require an arm")

    if args.mode == "leader" and args.sim:
        raise ValueError("Leader mode requires real hardware (--sim is not supported)")

    runners: Dict[str, Callable[[Args], None]] = {
        "follower": run_follower,
        "leader": run_leader,
        "visualizer_local": run_visualizer_local,
        "visualizer_remote": run_visualizer_remote,
    }
    runner = runners.get(args.mode)
    if runner is None:
        raise ValueError(f"Invalid mode: {args.mode}")
    runner(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
