"""Bring up (or attach to) the two follower RPC servers, and take them down without orphans.

Every tool in this package talks to the arms over a minimum_gello RPC server per side. This
starts them, so nothing else has to be running first, and a port that is already served is
attached to instead of launched over -- scripts/run_policy_followers.sh keeps the arms powered
between runs and must not be launched over, because the second server would die on bind while
the tool happily drove the first one.

Teardown goes through bimanual_teleop_record._terminate, which signals the whole process
*group* (SIGINT, then SIGKILL for stragglers). That matters more than it looks: minimum_gello
runs its motor chain in a multiprocessing child, and that grandchild is what actually owns the
CAN hardware. Signalling only the direct child -- which is what plain ``kill <pid>`` does --
leaves the grandchild reparented and still commanding the motors at full rate, with the RPC
port closed so it looks stopped. Launching through here cannot leave that behind.

    procs: List[subprocess.Popen[bytes]] = []
    try:
        followers.launch(("left", "right"), procs)   # before any thread starts
        ...
    finally:
        followers.terminate(procs)
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import bimanual_teleop_record as rec  # noqa: E402
from can_channels import channel_for  # noqa: E402

PORTS = {"left": 1235, "right": 1234}
"""RPC port per side, matching every other tool in this package."""


def default_channels() -> Dict[str, str]:
    """CAN netdev per side, from the one mapping table (scripts/can_map.conf)."""
    return {"left": channel_for("follower_left"), "right": channel_for("follower_right")}


def port_served(port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def launch(
    sides: Iterable[str],
    procs: List["subprocess.Popen[bytes]"],
    arm: str = "yam",
    version: int = 1,
    gripper: str = "linear_4310",
    channels: Optional[Dict[str, str]] = None,
    sim: bool = False,
    allow_launch: bool = True,
) -> None:
    """Start a follower server per side, or attach to one already serving that port.

    Must be called before this process starts any thread: _spawn_gello uses preexec_fn to set
    PR_SET_PDEATHSIG, which is what stops the arms if the tool is killed outright, and that is
    only safe from a single-threaded parent.

    Processes are appended to ``procs`` as they start, so a caller interrupted mid-launch can
    still tear down what did come up.
    """
    channels = channels or default_channels()
    started: Dict[str, int] = {}
    missing: List[str] = []
    for side in sides:
        port = PORTS[side]
        if port_served(port):
            print(f"[attach] {side} follower already serving port {port} -- using it, and leaving it running")
            continue
        if not allow_launch:
            missing.append(f"{side} (port {port})")
            continue
        if not sim:
            rec._check_can_interface(channels[side])
        cmd = [
            "--arm", arm,
            "--version", str(version),
            "--can_channel", channels[side],
            "--gripper", gripper,
            "--server_port", str(port),
        ]  # fmt: skip
        if sim:
            cmd.append("--sim")
        print(f"[launch] {side} follower on {channels[side]}, RPC port {port}")
        proc = rec._spawn_gello(cmd)
        procs.append(proc)
        started[side] = len(procs) - 1
    if missing:
        raise SystemExit(
            "[error] --no-launch, but nothing is listening for: "
            + ", ".join(missing)
            + " -- start them (scripts/run_policy_followers.sh) or drop --no-launch"
        )
    for side, i in started.items():
        rec._wait_for_port(PORTS[side], proc=procs[i])
        print(f"[launch] {side} follower up (pid {procs[i].pid})")


def terminate(procs: List["subprocess.Popen[bytes]"]) -> None:
    """Stop the followers this run started (see the note about process groups above)."""
    if not procs:
        return
    print("[stop] the followers this run launched are exiting: motor torque goes to zero and")
    print("[stop] the arms go limp. To keep them powered between runs, start them separately")
    print("[stop] (scripts/run_policy_followers.sh) and pass --no-launch.")
    rec._terminate(procs)
