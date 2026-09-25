"""Measure the zero-calibration offset between one side's leader and follower arm.

WHY: each DM motor's zero lives in its own firmware, saved once at assembly by holding the
joint at its nominal pose (i2rt/motor_config_tool/set_zero.py). Two physically identical
arms whose zeros were saved a couple of degrees apart report DIFFERENT joint angles for the
SAME physical pose. Teleop copies the leader's angles onto the follower verbatim
(examples/minimum_gello/minimum_gello.py::_leader_control_worker), so the difference shows up
as the follower sitting visibly rotated from the leader -- most obviously at the wrist, where
a couple of degrees of roll is easy to see. Nothing in the servo loop can detect it: the
follower reaches its commanded angle exactly, that angle just means something else on that arm.

This script measures the difference and prints the lines to paste into
scripts/arm_offsets.conf, which is where the correction is applied from.

Both arms come up gravity-compensated (the same state as run_leader_right.sh /
run_follower_right.sh) so you can move them freely by hand. Do NOT run it while teleop or a
policy rollout is running -- they want the same CAN buses.

TWO MODES
---------
--mode sweep (default, and the accurate one)
    For each joint in turn, push it to BOTH of its mechanical hard stops on BOTH arms while
    the script watches. The midpoint of a joint's mechanical range is a physical feature of
    the hardware, identical on two identical arms, so the difference of the two midpoints is
    the calibration offset -- with no dependence on your eye or on posing anything by hand.
    Sweep both arms over the same segment of travel; the script reports the range it saw on
    each so you can spot a joint you only pushed one way.

--mode pose
    The by-eye fallback: hold both arms in what looks like the same pose, press Enter, and
    the script averages the difference over a second. Fine for a sanity check or a joint
    whose stops you cannot reach, but it is only as good as your eye -- which is the very
    thing that noticed the problem, so prefer --mode sweep for the number you write down.

USAGE
    python scripts/leader_follower_offset.py --side right              # sweep joints 4 5 6
    python scripts/leader_follower_offset.py --side right --joints 6   # just the wrist roll
    python scripts/leader_follower_offset.py --side left --mode pose

READING THE RESULT
    The printed correction is for the FOLLOWER, i.e. it assumes the leader's zeros are the
    good ones -- which is the usual case, because the leader is what the operator holds and
    judges "straight" by. --correct leader flips that and prints the leader's correction
    instead, for when it is the leader that is visibly off. Only ever correct ONE arm of a
    pair, or you will chase the offset back and forth.

    Both arms come up with the CURRENT table applied, so the tool measures what is left after
    it and prints new TOTAL lines (table + residual) to replace that arm's lines. Re-run after
    editing: every corrected joint should then read ~0.0 residual.
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from arm_offsets import OFFSET_FILE, offsets_for
from can_channels import channel_for, label

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

_N_ARM_JOINTS = 6
_SAMPLE_PERIOD_S = 0.02
"""50 Hz. The arms publish cached state, so this is a display/averaging rate, not a control rate."""


@dataclass
class Args:
    side: Literal["left", "right"] = "right"
    """Which leader/follower pair to compare. CAN netdevs come from scripts/can_map.conf."""
    mode: Literal["sweep", "pose"] = "sweep"
    """sweep: hard-stop midpoints (accurate). pose: hold both arms alike and average (by eye)."""
    joints: List[int] = field(default_factory=lambda: [4, 5, 6])
    """Which arm joints (1..6) to measure. Defaults to the wrist, where an offset shows most."""
    correct: Literal["follower", "leader"] = "follower"
    """Which arm the printed correction is written for. Correct only one arm of a pair."""
    seconds: float = 1.0
    """--mode pose: how long to average once you press Enter."""
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    leader_gripper: str = "yam_teaching_handle"


class _ArmReader:
    """One gravity-compensated arm, with its joint angles polled into a shared buffer.

    Its own thread per arm: ``get_joint_pos`` is a cached read but the two arms sit on
    different CAN buses with independent control threads, and polling them from one loop
    would interleave their sample times."""

    def __init__(self, name: str, channel: str, arm: str, version: int, gripper: str, correction: np.ndarray):
        self.name = name
        self.channel = channel
        self.correction = correction
        applied = "" if not np.any(correction) else f", table correction (deg) {np.round(np.rad2deg(correction), 2).tolist()}"
        print(f"[{name}] bringing up on {label(channel)} (gravity compensation -- it will move freely{applied})")
        # The current table IS applied, exactly as every launcher applies it, so what this tool
        # measures is the RESIDUAL: zero once the table is right.
        self._robot = get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name(arm),
            version=version,
            gripper_type=GripperType.from_string_name(gripper),
            zero_gravity_mode=True,
            joint_offsets=correction,
        )
        self._pos = np.zeros(_N_ARM_JOINTS)
        self._have_sample = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True, name=f"{name}-poll")
        self._thread.start()
        self._wait_for_first_sample()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                pos = np.asarray(self._robot.get_joint_pos(), dtype=float)[:_N_ARM_JOINTS]
            except Exception as e:
                print(f"[{self.name}] read failed: {e}")
                time.sleep(0.1)
                continue
            with self._lock:
                self._pos = pos
                self._have_sample = True
            time.sleep(_SAMPLE_PERIOD_S)

    def _wait_for_first_sample(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not self._have_sample:
            if time.monotonic() > deadline:
                raise TimeoutError(f"[{self.name}] no joint reading within {timeout_s:.0f}s")
            time.sleep(0.05)

    def read(self) -> np.ndarray:
        with self._lock:
            return self._pos.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        close = getattr(self._robot, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f"[{self.name}] close failed: {e}")


def _prompt(text: str) -> None:
    """Block until Enter, so the operator drives the pace with both hands on the arms."""
    try:
        input(text)
    except EOFError:
        # Non-interactive stdin: nothing can drive the procedure, so stop rather than
        # spin through every step measuring an arm nobody moved.
        raise SystemExit("stdin closed -- this tool needs an operator at the arms") from None


def _watch_extremes(
    arms: Tuple[_ArmReader, _ArmReader], joint: int, stop_prompt: str
) -> Dict[str, Tuple[float, float]]:
    """Track each arm's min/max on ``joint`` (1-based) until the operator presses Enter.

    Returns ``{arm name: (min, max)}`` in radians. Sampling runs in this thread's own loop
    while a helper waits on stdin, so the live readout keeps updating as the operator moves."""
    idx = joint - 1
    lo = {a.name: np.inf for a in arms}
    hi = {a.name: -np.inf for a in arms}
    done = threading.Event()

    def _wait_enter() -> None:
        _prompt(stop_prompt)
        done.set()

    waiter = threading.Thread(target=_wait_enter, daemon=True)
    waiter.start()
    while not done.is_set():
        for a in arms:
            q = float(a.read()[idx])
            lo[a.name] = min(lo[a.name], q)
            hi[a.name] = max(hi[a.name], q)
        line = "  ".join(f"{a.name}: {np.rad2deg(lo[a.name]):+7.2f}..{np.rad2deg(hi[a.name]):+7.2f}" for a in arms)
        print(f"\r  joint{joint} travel (deg)  {line}   ", end="", flush=True)
        time.sleep(_SAMPLE_PERIOD_S)
    print()
    return {a.name: (lo[a.name], hi[a.name]) for a in arms}


def _sweep(leader: _ArmReader, follower: _ArmReader, joints: List[int]) -> Dict[int, float]:
    """Offset per joint from the midpoint of each arm's mechanical travel.

    Returns ``{joint: leader_midpoint - follower_midpoint}`` in radians -- how much the
    follower reads LOW relative to the leader at the same physical pose."""
    print(
        "\nSWEEP MODE\n"
        "  For each joint: move it slowly to one hard stop and then the other, on BOTH arms,\n"
        "  covering the same travel on each. The midpoint of the mechanical range is the same\n"
        "  physical place on two identical arms, so the difference of the midpoints is the\n"
        "  calibration offset. Do not force a stop -- rest against it, that is enough.\n"
    )
    offsets: Dict[int, float] = {}
    for joint in joints:
        _prompt(f"joint{joint}: press Enter, then sweep it stop-to-stop on both arms...")
        extremes = _watch_extremes((leader, follower), joint, "  ...press Enter when both arms have been swept: ")
        mids = {}
        for arm in (leader, follower):
            low, high = extremes[arm.name]
            span = high - low
            mids[arm.name] = 0.5 * (low + high)
            print(f"    {arm.name:<9} range {np.rad2deg(span):6.2f} deg, midpoint {np.rad2deg(mids[arm.name]):+7.3f}")
            if span < np.deg2rad(5.0):
                print(f"    WARNING: {arm.name} barely moved on joint{joint} -- that midpoint means nothing")
        span_gap = abs(
            (extremes[leader.name][1] - extremes[leader.name][0])
            - (extremes[follower.name][1] - extremes[follower.name][0])
        )
        if span_gap > np.deg2rad(5.0):
            # Different amounts of travel means the two sweeps did not cover the same segment,
            # so their midpoints are not the same physical place and the difference is noise.
            print(
                f"    WARNING: the two sweeps differ by {np.rad2deg(span_gap):.1f} deg of travel -- "
                f"redo joint{joint}, covering the same range on both arms"
            )
        offsets[joint] = mids[leader.name] - mids[follower.name]
        print(f"    -> follower reads {np.rad2deg(offsets[joint]):+.3f} deg low on joint{joint}\n")
    return offsets


def _pose(leader: _ArmReader, follower: _ArmReader, joints: List[int], seconds: float) -> Dict[int, float]:
    """Offset per joint from a single matched pose, averaged over ``seconds``."""
    print(
        "\nPOSE MODE\n"
        "  Hold both arms in the same pose by eye, then press Enter. This is only as accurate\n"
        "  as your eye -- use --mode sweep for a number you intend to write down.\n"
    )
    done = threading.Event()

    def _wait_enter() -> None:
        _prompt("  press Enter when both arms are posed alike: ")
        done.set()

    waiter = threading.Thread(target=_wait_enter, daemon=True)
    waiter.start()
    while not done.is_set():
        diff = leader.read() - follower.read()
        print(
            "\r  live difference (deg)  " + "  ".join(f"j{j}{np.rad2deg(diff[j - 1]):+7.2f}" for j in joints) + "   ",
            end="",
            flush=True,
        )
        time.sleep(_SAMPLE_PERIOD_S)
    print()

    samples = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        samples.append(leader.read() - follower.read())
        time.sleep(_SAMPLE_PERIOD_S)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    for j in joints:
        print(f"    joint{j}: follower reads {np.rad2deg(mean[j - 1]):+.3f} deg low (sd {np.rad2deg(std[j - 1]):.3f})")
    return {j: float(mean[j - 1]) for j in joints}


def _report(offsets: Dict[int, float], correct: str, side: str, current: np.ndarray) -> None:
    """Print the scripts/arm_offsets.conf lines that cancel the measured RESIDUAL.

    Both arms came up with the table already applied, so ``offsets`` is what is still left
    after it; the lines printed are the new TOTALS for the corrected arm (current table entry
    plus residual), to replace that arm's lines in the table, not to be added alongside them."""
    role = f"{correct}_{side}"
    # offsets[j] is (leader - follower). Correcting the follower means adding that difference
    # to what the follower reports; correcting the leader instead means removing it from the
    # leader. Either way the pair ends up agreeing -- correcting both would double it.
    sign = 1.0 if correct == "follower" else -1.0
    print("\n" + "=" * 70)
    print("Residual offsets (leader minus follower, same physical pose, current table applied):")
    for joint, value in sorted(offsets.items()):
        print(f"    joint{joint}: {np.rad2deg(value):+.3f} deg")
    significant = {j: v for j, v in offsets.items() if abs(np.rad2deg(v)) >= 0.3}
    if not significant:
        print("\nEverything is under 0.3 deg -- that is measurement noise. The table is right as it is.")
        return
    print(f"\nNew TOTAL lines for {OFFSET_FILE.name}, replacing {role}'s current lines:\n")
    print("# arm role         joint   degrees")
    for joint, value in sorted(significant.items()):
        total = np.rad2deg(current[joint - 1]) + sign * np.rad2deg(value)
        was = f"   # was {np.rad2deg(current[joint - 1]):+.2f}" if current[joint - 1] else ""
        print(f"{role:<18} {joint}      {total:+.2f}{was}")
    print("\nThen re-run this script: every corrected joint should read ~0.0 residual.")
    print("=" * 70)


def main(args: Args) -> None:
    bad = [j for j in args.joints if not 1 <= j <= _N_ARM_JOINTS]
    if bad:
        raise SystemExit(f"--joints must be in 1..{_N_ARM_JOINTS}, got {bad}")

    leader: Optional[_ArmReader] = None
    follower: Optional[_ArmReader] = None
    try:
        leader = _ArmReader(
            "leader",
            channel_for(f"leader_{args.side}"),
            args.arm,
            args.version,
            args.leader_gripper,
            offsets_for(f"leader_{args.side}"),
        )
        follower = _ArmReader(
            "follower",
            channel_for(f"follower_{args.side}"),
            args.arm,
            args.version,
            args.follower_gripper,
            offsets_for(f"follower_{args.side}"),
        )
        print(f"\nboth {args.side} arms up. Keep the workspace clear; Ctrl-C aborts.\n")
        if args.mode == "sweep":
            offsets = _sweep(leader, follower, args.joints)
        else:
            offsets = _pose(leader, follower, args.joints, args.seconds)
        _report(offsets, args.correct, args.side, offsets_for(f"{args.correct}_{args.side}"))
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        # Closing zeroes the motor torques, so both arms go limp: park them low first.
        for reader in (follower, leader):
            if reader is not None:
                reader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
