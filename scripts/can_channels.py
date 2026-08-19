"""Python access to the one CAN mapping table, scripts/can_map.conf.

Which netdev each arm sits on is NOT written down here -- it lives in
``scripts/can_map.conf``, a four-line ``<can number> <arm role>`` table that this module
parses. The shell scripts read the same file through ``scripts/_can_map.sh``, so editing
that one table re-points every script in this directory. After a reboot or a replug,
re-check the numbering (the recipe is in the header of can_map.conf) and edit the table.

There are no udev persistent names and no serial-based auto-resolution any more: the udev
rule matched on adapter serial only, and every adapter is dual-channel, so the persistent
name kept landing on the un-cabled netdev of the pair.

The odd sibling of each cabled netdev must stay DOWN -- while it is UP it absorbs part of
the motors' replies and the arm looks half-dead on both channels. scripts/fix_can_links.sh
enforces that, finding the sibling by USB serial rather than by number.

    from can_channels import channel_for, label
    channel_for("follower_left")   # -> 'can4'
    label("can4")                  # -> 'can4 (follower_left)'
"""

from pathlib import Path
from typing import Dict

MAP_FILE = Path(__file__).resolve().parent / "can_map.conf"
"""The table itself. The single place the arm -> channel assignment is written down."""

ARMS = ("leader_left", "leader_right", "follower_left", "follower_right")
"""The four arm roles the table must assign, each exactly once."""


def _parse(path: Path) -> Dict[str, str]:
    """``{arm role: netdev}`` from a ``<can number> <arm role>`` table."""
    channels: Dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError as e:
        raise RuntimeError(f"cannot read the CAN mapping table {path}: {e}") from e

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError(f"{path}:{lineno}: expected '<can number> <arm role>', got {raw.strip()!r}")
        number, arm = fields
        channel = number if number.startswith("can") else f"can{number}"
        if not channel[3:].isdigit():
            raise RuntimeError(f"{path}:{lineno}: {number!r} is not a CAN netdev number (e.g. 4 or can4)")
        if arm not in ARMS:
            raise RuntimeError(f"{path}:{lineno}: unknown arm role {arm!r} (expected one of {', '.join(ARMS)})")
        if arm in channels:
            raise RuntimeError(f"{path}:{lineno}: arm {arm!r} is assigned twice")
        if channel in channels.values():
            raise RuntimeError(f"{path}:{lineno}: {channel} is assigned to two arms")
        channels[arm] = channel

    missing = [arm for arm in ARMS if arm not in channels]
    if missing:
        raise RuntimeError(f"{path}: no channel for {', '.join(missing)} -- all four arms must be in the table")
    return channels


ARM_CHANNELS: Dict[str, str] = _parse(MAP_FILE)
"""arm role -> CAN netdev, as read from :data:`MAP_FILE`."""

CHANNEL_ARMS: Dict[str, str] = {channel: arm for arm, channel in ARM_CHANNELS.items()}
"""Reverse lookup, for labelling a channel in diagnostics."""


def channel_for(arm: str) -> str:
    """CAN netdev cabled to ``arm`` (one of the keys of :data:`ARM_CHANNELS`)."""
    try:
        return ARM_CHANNELS[arm]
    except KeyError:
        raise KeyError(f"unknown arm {arm!r} (expected one of {sorted(ARM_CHANNELS)})") from None


def label(channel: str) -> str:
    """``'can4 (follower_left)'`` -- channel plus the arm it belongs to, for log lines."""
    arm = CHANNEL_ARMS.get(channel)
    return f"{channel} ({arm})" if arm else channel


def table() -> str:
    """The mapping as one line per arm, in role order -- for --help text and log banners."""
    return "\n".join(f"    {ARM_CHANNELS[arm]:<6} {arm}" for arm in ARMS)


if __name__ == "__main__":
    # `python scripts/can_channels.py` prints the table -- a quick way to check what every
    # script currently believes, and it fails loudly if the table is malformed.
    print(f"{MAP_FILE}:")
    print(table())
