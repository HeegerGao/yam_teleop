"""Move one or both arms to a named pose (see poses.py).

    python scripts/box_packing/goto_pose.py --pose working
    python scripts/box_packing/goto_pose.py --pose home --sides left      # end of session
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Arm  # noqa: E402
from collide import check_path, scan_obstacles  # noqa: E402
from perception import Scene  # noqa: E402
from poses import describe, named  # noqa: E402

PORTS = {"left": 1235, "right": 1234}


@dataclass
class Args:
    pose: str = "working"
    """working | home"""
    sides: Tuple[str, ...] = ("left", "right")
    max_joint_vel: float = 0.7
    min_gap: float = 0.02
    """Refuse the move if a link would pass within this of an object from the last scan."""
    execute: bool = True


def main(args: Args) -> None:
    q_target = named(args.pose)
    try:
        obstacles = scan_obstacles(Scene.from_calib("left"))
        print(f"[clear] {len(obstacles)} obstacle points from the last scan")
    except Exception as e:
        obstacles = None
        print(f"[clear] no usable scan to check against ({e}) -- moving without a collision check")
    for side in args.sides:
        arm = Arm(side, PORTS[side], max_joint_vel=args.max_joint_vel, execute=args.execute)
        try:
            q0 = arm.q().copy()
            print(f"[{side}] {args.pose}: {describe(arm.kin, q_target)}")
            if not check_path(arm.kin, q0, q_target, obstacles, f"{side} -> {args.pose}", args.min_gap):
                raise SystemExit(f"[{side}] that path would sweep into something on the table -- refusing")
            if args.execute:
                res = arm.move_joints(q_target, settle=0.0)
                arm.wait_settled()
                print(f"[{side}] at {np.round(arm.q()[:6], 3)} {'ok' if not res.aborted else res.aborted}")
        finally:
            arm.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
