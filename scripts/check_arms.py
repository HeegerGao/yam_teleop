"""Quick health check for the four YAM CAN buses: is each arm powered and talking?

Sends the DM-motor *disable* frame (0xFD -- a safe no-op for an idle motor, and motors
reply to it either way) to motor IDs 1-3 on every bus and counts replies. A powered,
connected arm answers 3/3; a dead bus (arm unpowered, e-stop pressed, cable unplugged)
answers 0/3. Also reports the kernel CAN controller state (ERROR-ACTIVE is healthy;
BUS-OFF means run scripts/reset_all_can.sh).

Do NOT run while teleop is running -- the probe shares the bus with the control loops.

By default it probes the four channels assigned in scripts/can_map.conf -- the one table that
maps arms to buses. Pass --channels explicitly to sweep others; that is how you re-derive the
table after a reboot moved the numbering (see the table's own header).

Usage:
    python scripts/check_arms.py
    python scripts/check_arms.py --channels can6 can0
    python scripts/check_arms.py --channels can0 can1 can2 can3 can4 can5 can6 can7   # sweep
"""

import subprocess
from dataclasses import dataclass, field
from typing import List

import can
import tyro
from can_channels import ARM_CHANNELS, ARMS, label

_DISABLE = bytes([0xFF] * 7 + [0xFD])


@dataclass
class Args:
    channels: List[str] = field(default_factory=lambda: [ARM_CHANNELS[arm] for arm in ARMS])
    """CAN netdevs to probe. Default: the four arm buses from scripts/can_map.conf, in role order."""
    timeout: float = 0.3
    """Per-motor reply timeout in seconds."""
    motors: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Motor IDs to probe. The default checks bus health cheaply; use --motors 1 2 3 4 5 6 7
    on a follower (1-6 on a leader) to locate a daisy-chain break: the chain runs base->tip,
    so 'first N reply, rest silent' pinpoints the broken segment."""


def controller_state(channel: str) -> str:
    result = subprocess.run(["ip", "-details", "link", "show", channel], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return "INTERFACE MISSING"
    for line in result.stdout.splitlines():
        if "can state" in line:
            return line.strip().split()[2]
    return "UNKNOWN"


def probe(channel: str, motor_ids: List[int], timeout: float) -> List[int]:
    """Motor IDs that replied on this channel."""
    try:
        bus = can.Bus(channel=channel, interface="socketcan")
    except OSError as e:
        print(f"  cannot open {channel}: {e}")
        return []
    replies: List[int] = []
    try:
        for motor_id in motor_ids:
            try:
                bus.send(can.Message(arbitration_id=motor_id, data=_DISABLE, is_extended_id=False))
            except can.CanOperationError as e:
                # ENOBUFS: the TX queue is jammed with unacknowledged frames -- nothing on the bus
                # is ACKing, and it stays jammed until the link is bounced. A bounce alone only
                # clears the symptom: if the bus answers right after `ip link ... up` and then
                # jams again, the connection itself is intermittent (power / connector / cable).
                print(f"  {channel}: cannot transmit ({e}) -- TX queue jammed, run scripts/fix_can_links.sh")
                return []
            if bus.recv(timeout=timeout):
                replies.append(motor_id)
    finally:
        bus.shutdown()
    return replies


def main(args: Args) -> None:
    all_ok = True
    for channel in args.channels:
        state = controller_state(channel)
        if state == "INTERFACE MISSING":
            print(f"[FAIL] {label(channel)}: interface missing (USB-CAN adapter unplugged?)")
            all_ok = False
            continue
        replies = probe(channel, args.motors, args.timeout)
        missing = [m for m in args.motors if m not in replies]
        ok = not missing and state == "ERROR-ACTIVE"
        all_ok &= ok
        verdict = "OK  " if ok else "FAIL"
        hint = ""
        if not replies:
            hint = "  <- no motor replies: arm unpowered / e-stop / cable"
        elif missing:
            hint = f"  <- silent: {missing} -- daisy-chain break between motor {max(replies)} and {min(missing)}"
        if state != "ERROR-ACTIVE":
            hint += f"  <- controller {state}: run scripts/reset_all_can.sh"
        print(f"[{verdict}] {label(channel)}: motors {len(replies)}/{len(args.motors)}, controller {state}{hint}")
    print("all buses healthy" if all_ok else "some buses unhealthy -- fix before launching teleop")


if __name__ == "__main__":
    main(tyro.cli(Args))
