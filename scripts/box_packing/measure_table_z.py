"""Measure where the table is, in each arm's own base frame, by touching it.

The click calibration gives xy from a pixel, but not how high anything is; heights come from
the depth camera, which measures them *above the table*. This ties the two together with one
number per arm: the base-frame z of the table surface, found by lowering the closed gripper
until the joints start lagging their command -- the same contact test the picks use.

Output: calib/table_z_<side>.json.

Both follower arms are started by this script, so nothing else has to be running first; a port
that is already served is attached to instead, so scripts/run_policy_followers.sh still works
(pass --no-launch to require that). Each arm is folded down to the home pose when it is done,
before the followers this run started are stopped -- that cuts motor torque, and home is where
the arms rest unpowered.

    python scripts/box_packing/measure_table_z.py --sides left right
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import followers  # noqa: E402
from arm import Arm, Workspace  # noqa: E402
from poses import WORKING_Q, go_home, go_working  # noqa: E402

PORTS = followers.PORTS
CALIB = _HERE / "calib"
SPOTS = {"left": (0.39, 0.0), "right": (0.36, 0.22)}
"""An empty patch of table each arm can reach with a vertical wrist (its own base frame)."""


@dataclass
class Args:
    sides: Tuple[str, ...] = ("left", "right")
    start_z: float = 0.13
    """Height to start the descent from."""
    min_z: float = 0.02
    speed: float = 0.02
    """Descent speed, m/s -- slow, so contact is caught within a millimetre or two."""
    contact_rad: float = 0.05
    """Joint lag over its starting value that counts as touching down."""
    tip_offset: float = 0.004
    """Fingertip radius: the tips bottom out this far above where their centre reads."""
    execute: bool = True
    launch: bool = True
    """Start a minimum_gello follower server per arm, so no other terminal is needed. Ports
    that are already served are attached to either way; --no-launch requires that."""
    park_on_exit: bool = True
    """Fold each arm down to the home pose when it is done, before the followers stop."""
    arm: str = "yam"
    version: int = 1
    gripper: str = "linear_4310"
    can_follower_left: str = followers.default_channels()["left"]
    """LEFT follower CAN netdev. Default comes from the mapping table scripts/can_map.conf."""
    can_follower_right: str = followers.default_channels()["right"]
    """RIGHT follower CAN netdev (from scripts/can_map.conf)."""
    sim: bool = False
    """Launch the followers in MuJoCo instead of on the CAN buses (no hardware needed)."""


def measure(args: Args) -> None:
    CALIB.mkdir(exist_ok=True)
    for side in args.sides:
        arm = Arm(side, PORTS[side], max_joint_vel=0.5, execute=args.execute, workspace=Workspace(z=(0.0, 0.6)))
        try:
            x, y = SPOTS[side]
            go_working(arm)
            arm.set_gripper(0.0, wait=0.3)
            start = np.array([x, y, args.start_z])
            q6, R = arm.kin.ik_grasp(start, np.array([0.0, 0.0, -1.0]), WORKING_Q, finger_axis=np.array([1.0, 0.0, 0.0]))
            arm.move_joints(np.concatenate([q6, [0.0]]), settle=0.0)
            arm.wait_settled()
            time.sleep(0.3)
            if not args.execute:
                print(f"[{side}] would touch down at ({x}, {y}) from z={args.start_z}")
                continue
            sag = arm.grasp_pose()[0] - arm.kin.grasp(arm.q_cmd)[0]
            print(f"[{side}] descending at ({x}, {y}); sag at the start {np.round(sag * 1e3, 0)} mm")
            target = np.array([x, y, args.min_z]) - sag
            res = arm.move_linear(target, R, speed=args.speed, correct=0, track_abort_rad=args.contact_rad)
            touched = arm.grasp_pose()[0]
            if not res.aborted:
                print(f"[{side}] reached {args.min_z:.3f} without contact -- the table is lower than that; not saving")
            else:
                table_z = float(touched[2] - args.tip_offset)
                print(f"[{side}] contact at {np.round(touched, 4)} ({res.aborted}) -> table z = {table_z:.4f}")
                (CALIB / f"table_z_{side}.json").write_text(json.dumps({"side": side, "table_z": table_z, "spot": [x, y]}, indent=1))
            up = touched + np.array([0.0, 0.0, 0.06])
            q6, perr, rerr = arm.kin.ik(up, R, arm.q_cmd)
            if perr < 5e-3:
                arm.move_joints(np.concatenate([q6, [0.0]]), settle=0.0)
            arm.set_gripper(1.0, wait=0.0)
            if args.park_on_exit and args.execute:
                print(f"[{side}] -> home pose")
                go_home(arm)
            else:
                go_working(arm)
        finally:
            arm.close()


def main(args: Args) -> None:
    # Followers first, before this process starts a single thread: _spawn_gello uses
    # preexec_fn (PR_SET_PDEATHSIG), which is only safe from a single-threaded parent -- and
    # that death signal is what stops the arms if this tool is killed outright.
    procs: List["subprocess.Popen[bytes]"] = []
    try:
        followers.launch(
            args.sides,
            procs,
            arm=args.arm,
            version=args.version,
            gripper=args.gripper,
            channels={"left": args.can_follower_left, "right": args.can_follower_right},
            sim=args.sim,
            allow_launch=args.launch,
        )
        measure(args)
    finally:
        followers.terminate(procs)


if __name__ == "__main__":
    main(tyro.cli(Args))
