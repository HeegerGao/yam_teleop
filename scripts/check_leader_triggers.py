"""Read the leader teaching-handle trigger encoders and print raw counts + gripper command.

The leader's gripper command is ``1 - encoder.position`` (examples/minimum_gello/
minimum_gello.py::YAMLeaderRobot.get_info), where ``position`` is
``|raw_rad| / 0.7`` after clipping raw_rad to +-0.7 (PassiveEncoderReader.read_encoder).
The encoder reports a wrapped 12-bit absolute angle, so a handle whose zero sits a hair
above its rest angle reports ~4090 counts (~+6.27 rad) rather than ~-6 -- the same pose one
turn up. dm_driver folds the count into (-2048, 2048] before scaling; this script prints the
un-folded wire value next to the resulting command so a drifted zero stays visible.

At rest expect: counts within a few of 0 or 4096, gripper_cmd ~1.0 (open). Squeezing the
trigger should sweep gripper_cmd down to 0.0 (closed). A gripper_cmd pinned at 0.00 that
does not move with the trigger means that handle's encoder zero is off by more than the
0.7 rad trigger travel -- re-zero it with i2rt/utils/encoder_manager.py reset-zero-position.

Read-only: it only sends the encoder's passive read request, no motor commands.

Usage:
    python scripts/check_leader_triggers.py                 # both leaders, from can_map.conf
    python scripts/check_leader_triggers.py --side left
"""

import struct
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from can_channels import channel_for

from i2rt.motor_drivers.dm_driver import CanInterface, ReceiveMode
from i2rt.robots.get_robot import get_encoder_chain

_ENCODER_ID = 0x50E


def _raw_counts(can_interface: CanInterface) -> int:
    """The position field straight off the wire, before dm_driver's wrap folding."""
    message = can_interface._send_message_get_response(
        _ENCODER_ID,
        _ENCODER_ID,
        [0xFF, 0x02],
        expected_id=ReceiveMode.plus_one.get_receive_id(_ENCODER_ID),
        max_retry=15,
    )
    _device_id, position, _velocity, _digital_inputs = struct.unpack("!B h h B", message.data)
    return position


def main(side: Literal["left", "right", "both"] = "both", duration: Optional[float] = None) -> None:
    sides = ["left", "right"] if side == "both" else [side]
    arms = {s: channel_for(f"leader_{s}") for s in sides}
    print("Squeeze each trigger through its full travel; gripper_cmd should sweep 1.0 -> 0.0.")
    for s, ch in arms.items():
        print(f"  leader_{s}: {ch}")
    print()

    readers = {}
    for s, ch in arms.items():
        can_interface = CanInterface(channel=ch, use_buffered_reader=False)
        readers[s] = (can_interface, get_encoder_chain(can_interface))

    deadline = None if duration is None else time.time() + duration
    while deadline is None or time.time() < deadline:
        parts = []
        for s, (can_interface, chain) in readers.items():
            try:
                counts = _raw_counts(can_interface)
                state = chain.read_states()[0]
            except Exception as e:  # keep the other side readable when one bus is quiet
                parts.append(f"{s}: ERROR {type(e).__name__}")
                continue
            folded = (counts + 2048) % 4096 - 2048
            rad = folded * 2 * np.pi / 4096
            parts.append(
                f"{s}: counts={counts:+5d} rad={rad:+.3f} gripper_cmd={1 - state.position:.2f} btn={state.io_inputs}"
            )
        sys.stdout.write("\r" + " | ".join(parts) + "   ")
        sys.stdout.flush()
        time.sleep(0.05)
    print()


if __name__ == "__main__":
    tyro.cli(main)
