"""Survey every CAN netdev on the machine and say what is on it -- the tool for re-deriving
scripts/can_map.conf after a reboot, a replug, or an adapter swap.

For each netdev it prints the USB adapter it belongs to (serial + interface number), the link
state, and how many motors answer. Adapters are grouped: each USB-CAN adapter exposes two
netdevs, only one of which is cabled, and the pairing is what makes the numbering readable.

The motor count identifies the arm type: a leader (teaching handle, passive) answers IDs 1-6,
a follower (powered gripper) answers 1-7. Left vs right cannot be told from the bus -- bring
one arm up with scripts/run_leader_left.sh and see which one goes gravity-compensated, or
power just one arm at a time.

Only netdevs that are UP can be probed. Right after a boot the flow_base udev rule brings up
every adapter's interface-00 netdev, which is usually exactly the cabled ones; if a bus you
expect is DOWN, run `sudo scripts/reset_all_can.sh` first.

Do NOT run while teleop is running -- the probe shares the bus with the control loops.

Usage:
    python scripts/scan_can.py
    python scripts/scan_can.py --motors 9        # arms with more than 7 motors on the chain
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import can
import tyro
from can_channels import ARM_CHANNELS, ARMS, MAP_FILE

_DISABLE = bytes([0xFF] * 7 + [0xFD])
_SYS_NET = Path("/sys/class/net")


@dataclass
class Args:
    motors: int = 7
    """Probe motor IDs 1..N. 7 covers a follower (6 joints + gripper); a leader answers 6."""
    timeout: float = 0.2
    """Per-motor reply timeout in seconds."""


def netdevs() -> List[str]:
    """Every can* netdev, ordered by index so pairs read naturally."""
    names = [p.name for p in _SYS_NET.glob("can*")]
    return sorted(names, key=lambda n: (int(n[3:]) if n[3:].isdigit() else 0, n))


def usb_info(channel: str) -> Tuple[str, str]:
    """``(adapter serial, USB interface number)`` for a netdev, ``('?', '?')`` if unknown."""
    device = (_SYS_NET / channel / "device").resolve()
    serial = device.parent / "serial"
    interface = device / "bInterfaceNumber"
    return (
        serial.read_text().strip() if serial.exists() else "?",
        interface.read_text().strip() if interface.exists() else "?",
    )


def link_state(channel: str) -> str:
    """``'ERROR-ACTIVE'`` / ``'STOPPED'`` / ``'BUS-OFF'``; ``'DOWN'`` if the link is not up."""
    result = subprocess.run(["ip", "-details", "link", "show", channel], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return "MISSING"
    up = "state UP" in result.stdout or ",UP," in result.stdout.split(":", 2)[-1]
    for line in result.stdout.splitlines():
        if "can state" in line:
            state = line.strip().split()[2]
            return state if up else f"DOWN ({state})"
    return "UP" if up else "DOWN"


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
                break  # link down, or the TX queue jammed because nothing on the bus ACKs
            if bus.recv(timeout=timeout) is not None:
                replies.append(motor_id)
    finally:
        bus.shutdown()
    return replies


def arm_kind(count: int, motors: int) -> str:
    """What a motor count says about the arm on the bus."""
    if count == 0:
        return ""
    if count == 6:
        return "LEADER   (6 motors: 6 joints, passive handle)"
    if count == 7:
        return "FOLLOWER (7 motors: 6 joints + gripper)"
    if count < 6:
        return f"partial chain -- only {count} answered, daisy-chain break after motor {count}"
    return f"{count} motors (probed 1..{motors})"


def main(args: Args) -> None:
    channels = netdevs()
    if not channels:
        print("no CAN netdevs at all -- USB-CAN adapters unplugged, or gs_usb not loaded")
        return

    adapters: Dict[str, List[str]] = {}
    info: Dict[str, Tuple[str, str, str, List[int]]] = {}
    for channel in channels:
        serial, interface = usb_info(channel)
        state = link_state(channel)
        replies = replying_ids(channel, args.motors, args.timeout) if not state.startswith("DOWN") else []
        info[channel] = (serial, interface, state, replies)
        adapters.setdefault(serial, []).append(channel)

    current = {channel: arm for arm, channel in ARM_CHANNELS.items()}
    print(f"{len(adapters)} USB-CAN adapter(s), {len(channels)} netdev(s). Current table: {MAP_FILE}\n")
    for serial, group in adapters.items():
        print(f"adapter {serial}")
        for channel in group:
            _, interface, state, replies = info[channel]
            assigned = f"  [table: {current[channel]}]" if channel in current else ""
            note = arm_kind(len(replies), args.motors) if not state.startswith("DOWN") else "not probed (link down)"
            print(f"  {channel:<6} usb_if={interface}  {state:<14} {note}{assigned}")
        print()

    cabled = [c for c in channels if len(info[c][3]) >= 6]
    print("-- what to write in the table " + "-" * 48)
    if not cabled:
        print("No bus answered. Every arm unpowered / e-stopped, or every link is DOWN:")
        print("  sudo scripts/reset_all_can.sh     then re-run this scan")
        return
    print("These buses have an arm on them; fill in left/right yourself (the bus cannot tell you):\n")
    for channel in cabled:
        kind = "leader" if len(info[channel][3]) == 6 else "follower"
        print(f"    {channel[3:]:<4}{kind}_left | {kind}_right")
    print("\nTo tell left from right, bring one up and watch which arm moves, e.g.")
    print(f"    YAM_CAN_CHANNEL={cabled[0]} scripts/run_leader_left.sh")
    missing = [arm for arm in ARMS if ARM_CHANNELS[arm] not in cabled]
    if missing:
        print(f"\nThe table currently assigns {', '.join(missing)} to a bus that did not answer.")
    print(f"\nThen: edit {MAP_FILE}, and run")
    print("    sudo scripts/fix_can_links.sh && python scripts/check_arms.py")


if __name__ == "__main__":
    main(tyro.cli(Args))
