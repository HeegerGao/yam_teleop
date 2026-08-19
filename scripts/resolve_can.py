"""Look up an arm's CAN channel, or probe one channel for motor replies.

The arm -> channel mapping is read from scripts/can_map.conf, the one table (via
scripts/can_channels.py). Nothing resolves anything by adapter serial or udev persistent name
any more -- both kept binding to the un-cabled channel of the dual-channel adapters, which is
exactly what the table replaces.

Prints its answer on stdout and nothing else, so shell callers can capture it
(scripts/_yam_arm_common.sh for --arm, scripts/fix_can_links.sh for --channel).

Do NOT probe while teleop is running -- it shares the bus with the control loops.

Usage:
    python scripts/resolve_can.py --arm follower_left      # -> can4
    python scripts/resolve_can.py --channel can4           # -> how many of motors 1..N answer
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import can
import tyro
from can_channels import ARM_CHANNELS, channel_for

_DISABLE = bytes([0xFF] * 7 + [0xFD])


@dataclass
class Args:
    arm: Optional[str] = None
    """Arm to look up: one of follower_left, follower_right, leader_left, leader_right."""
    channel: Optional[str] = None
    """Probe this netdev instead and print how many of motors 1..N answered."""
    motors: int = 6
    """Probe motor IDs 1..N."""
    timeout: float = 0.2
    """Reply timeout per motor, in seconds."""


def channel_is_up(channel: str) -> bool:
    try:
        flags = int((Path("/sys/class/net") / channel / "flags").read_text().strip(), 16)
        return bool(flags & 1)  # IFF_UP
    except (OSError, ValueError):
        return False


def replying_ids(channel: str, motors: int, timeout: float) -> List[int]:
    """Which of motor IDs 1..N answer a (safe, no-op) disable frame on this channel."""
    try:
        bus = can.Bus(channel=channel, interface="socketcan")
    except OSError:
        return []
    replies: List[int] = []
    try:
        for motor_id in range(1, motors + 1):
            try:
                bus.send(can.Message(arbitration_id=motor_id, data=_DISABLE, is_extended_id=False))
            except can.CanOperationError:
                # ENOBUFS: the TX queue is jammed because nothing on the bus ACKs, and it stays
                # jammed until the link is bounced -- so this looks the same whether the port is
                # un-cabled or the arm went quiet. Bounce the link and re-probe to tell them apart.
                break
            if bus.recv(timeout=timeout) is not None:
                replies.append(motor_id)
    finally:
        bus.shutdown()
    return replies


def main(args: Args) -> None:
    if args.channel is not None:
        ids = replying_ids(args.channel, args.motors, args.timeout) if channel_is_up(args.channel) else []
        # Count on stdout (callers capture it); the IDs on stderr, where a gap pinpoints where
        # the daisy chain breaks.
        print(f"      answering IDs: {ids or 'none'}", file=sys.stderr)
        print(len(ids))
        return
    if args.arm is None:
        print(f"pass --arm (one of {sorted(ARM_CHANNELS)}) or --channel <netdev>", file=sys.stderr)
        raise SystemExit(2)
    try:
        print(channel_for(args.arm))
    except KeyError as e:
        print(f"resolve_can: {e}", file=sys.stderr)
        raise SystemExit(2) from e


if __name__ == "__main__":
    main(tyro.cli(Args))
