"""Python access to the one zero-calibration table, scripts/arm_offsets.conf.

Which arm reads a couple of degrees off, and on which joint, is NOT written down here --
it lives in ``scripts/arm_offsets.conf``, a ``<arm role> <joint> <degrees>`` table that this
module parses. Same shape as the CAN mapping table next to it (:mod:`can_channels`): one
hand-edited file, every script reads it, editing it re-points everything at once.

Why it exists: each DM motor's zero is stored in its own firmware, saved once at assembly.
Two identical arms zeroed a couple of degrees apart report different joint angles for the
same physical pose, and teleop copies the leader's angles onto the follower verbatim -- so
the follower sits visibly rotated. The correction here is applied at the motor chain
(``joint_offsets`` in :func:`i2rt.robots.get_robot.get_yam_robot`), which puts the arm's
reported angles, its commands, its FK and its gravity compensation in one consistent frame.

Sign: the value is ADDED to the angle the arm reports, i.e. it is how much that arm reads
too low at a given physical pose.

    from arm_offsets import offsets_for, offsets_for_channel
    offsets_for("follower_right")   # -> array of 6 radians, zeros when uncorrected
    offsets_for_channel("can0")     # -> same, resolved through scripts/can_map.conf
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from can_channels import ARMS, CHANNEL_ARMS

OFFSET_FILE = Path(__file__).resolve().parent / "arm_offsets.conf"
"""The table itself. The single place per-arm zero corrections are written down."""

N_ARM_JOINTS = 6
"""Corrections cover the six arm joints; the gripper has no comparable zero."""


def _parse(path: Path) -> Dict[str, np.ndarray]:
    """``{arm role: (6,) radians}`` from a ``<arm role> <joint> <degrees>`` table.

    Every role gets an entry, all-zero when the table says nothing about it, so callers
    never have to special-case an uncorrected arm."""
    offsets: Dict[str, np.ndarray] = {arm: np.zeros(N_ARM_JOINTS) for arm in ARMS}
    if not path.exists():
        # An absent table means "no arm is corrected" -- the same thing an empty one means.
        return offsets

    try:
        text = path.read_text()
    except OSError as e:
        raise RuntimeError(f"cannot read the zero-calibration table {path}: {e}") from e

    seen: set[tuple[str, int]] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError(f"{path}:{lineno}: expected '<arm role> <joint> <degrees>', got {raw.strip()!r}")
        arm, joint_str, degrees_str = fields
        if arm not in ARMS:
            raise RuntimeError(f"{path}:{lineno}: unknown arm role {arm!r} (expected one of {', '.join(ARMS)})")
        try:
            joint = int(joint_str)
        except ValueError:
            raise RuntimeError(f"{path}:{lineno}: {joint_str!r} is not a joint number (1..{N_ARM_JOINTS})") from None
        if not 1 <= joint <= N_ARM_JOINTS:
            raise RuntimeError(f"{path}:{lineno}: joint {joint} out of range (1..{N_ARM_JOINTS})")
        try:
            degrees = float(degrees_str)
        except ValueError:
            raise RuntimeError(f"{path}:{lineno}: {degrees_str!r} is not a number of degrees") from None
        if (arm, joint) in seen:
            raise RuntimeError(f"{path}:{lineno}: {arm} joint{joint} is corrected twice")
        # A correction this big is a mis-typed sign or a wrong unit, not a calibration:
        # a saved zero is off by degrees, and a genuine 2*pi wrap is already handled at
        # bring-up in get_robot.py. Refuse rather than silently fold the arm in half.
        if abs(degrees) > 30.0:
            raise RuntimeError(
                f"{path}:{lineno}: {degrees} deg is too large for a zero correction "
                f"(limit +-30) -- check the sign and the units"
            )
        seen.add((arm, joint))
        offsets[arm][joint - 1] = np.deg2rad(degrees)
    return offsets


ARM_OFFSETS: Dict[str, np.ndarray] = _parse(OFFSET_FILE)
"""arm role -> (6,) radians added to that arm's reported joint angles."""


def offsets_for(arm: str) -> np.ndarray:
    """Zero correction for ``arm``, in radians, one per arm joint. Zeros when uncorrected."""
    try:
        return ARM_OFFSETS[arm].copy()
    except KeyError:
        raise KeyError(f"unknown arm {arm!r} (expected one of {sorted(ARM_OFFSETS)})") from None


def offsets_for_channel(channel: str) -> np.ndarray:
    """Zero correction for whichever arm is cabled to ``channel`` per scripts/can_map.conf.

    An unknown channel is not an error: a bus that the mapping table does not mention is an
    arm this table has nothing to say about either, so it gets zeros."""
    arm = CHANNEL_ARMS.get(channel)
    return offsets_for(arm) if arm is not None else np.zeros(N_ARM_JOINTS)


def is_corrected(arm: str) -> bool:
    """Whether ``arm`` carries any non-zero correction -- for log lines worth printing."""
    return bool(np.any(offsets_for(arm)))


def table() -> str:
    """The corrections as one line per arm, in role order -- for --help text and banners."""
    lines: List[str] = []
    for arm in ARMS:
        degrees = np.rad2deg(ARM_OFFSETS[arm])
        if np.any(degrees):
            pretty = "  ".join(f"j{i + 1}{d:+.2f}" for i, d in enumerate(degrees) if d)
            lines.append(f"    {arm:<15} {pretty}")
        else:
            lines.append(f"    {arm:<15} (uncorrected)")
    return "\n".join(lines)


if __name__ == "__main__":
    # `python scripts/arm_offsets.py` prints the table -- a quick way to check what every
    # script currently believes, and it fails loudly if the table is malformed.
    # `python scripts/arm_offsets.py --channel can0` prints that channel's six corrections in
    # radians on one line, for shell launchers to splice into a --joint_offsets argument.
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--channel":
        print(" ".join(f"{x:.6f}" for x in offsets_for_channel(sys.argv[2])))
    elif len(sys.argv) == 1:
        print(f"{OFFSET_FILE}:")
        print(table())
    else:
        raise SystemExit("usage: arm_offsets.py [--channel canN]")
