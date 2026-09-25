"""The two canonical arm poses of the box-packing system.

WORKING: half-open, elbow up, fingers pointing forward-down above the table's near edge. This
is where a task starts and where it returns between objects: it is outside the top camera's
frustum (so a scan never sees the arm), every link is >= 40 cm above the table, and it is
about half the joint travel from a typical grasp, which makes each pick visibly quicker than
starting from the folded pose.

HOME: the folded rest pose. Only used when the session is over -- ``goto_pose.py --pose home``.

    from poses import WORKING_Q, HOME_Q, describe
"""

from __future__ import annotations

from typing import Dict

import numpy as np

WORKING_Q = np.array([0.0, 1.10, 1.90, -1.20, 0.0, 0.0, 1.0])
"""Half-open working pose (7-D: 6 arm joints + gripper, 1 = open)."""

HOME_Q = np.array([0.0, 0.02, 0.07, -0.13, 0.0, 0.0, 1.0])
"""Folded rest pose, close to where the arms sit unpowered."""

POSES: Dict[str, np.ndarray] = {"working": WORKING_Q, "home": HOME_Q}


def named(pose: str) -> np.ndarray:
    try:
        return POSES[pose].copy()
    except KeyError:
        raise SystemExit(f"unknown pose {pose!r} (expected one of {sorted(POSES)})") from None


def describe(kin, q: np.ndarray) -> str:
    """One line about a pose: fingertip position and the lowest link over the table."""
    from calibrate_top import ArmSurface

    surf = ArmSurface(kin, skip_bodies=("base", "link1"))
    p, R = kin.grasp(q)
    return f"tips {np.round(p, 3)}, finger dir {np.round(R[:, 2], 2)}, lowest link over the table {surf.min_link_z(q):.3f} m"


def go_working(arm, lift: float = 0.08, vel: float = 0.6) -> None:
    """Bring one arm to the working pose from wherever it is: straight up first (it may be
    inside a box), then one curve. Scans need this -- an arm parked over the table both
    occludes the objects and leaks into their masks."""
    import numpy as np

    q0 = arm.q()
    if float(np.max(np.abs(q0[:6] - WORKING_Q[:6]))) < 0.05:
        return
    p, _ = arm.grasp_pose()
    if p[2] < 0.30:
        try:
            q6, _, _ = arm.kin.ik_reach(p + np.array([0.0, 0.0, lift]), arm.q_cmd)
            arm.move_joints(np.concatenate([q6, [arm.q_cmd[6]]]), settle=0.0)
        except ValueError:
            pass
    arm.move_through([WORKING_Q])
    arm.wait_settled()


def go_home(arm, lift: float = 0.08, settle: float = 0.0) -> None:
    """Fold one arm down to the rest pose, by way of the working pose.

    Straight from wherever it is to the folded pose would sweep the arm across the table; the
    working pose is clear of everything (go_working lifts first when the fingers are down in a
    box), and the fold from there comes down beside the base rather than over the objects.

    This is where the arms belong at the end of a run. Whatever started the followers stops
    them on the way out, and that sets motor torque to zero -- home is the pose they rest in
    unpowered, so they settle into it instead of dropping from wherever they were left.
    """
    go_working(arm, lift=lift)
    arm.move_joints(HOME_Q, settle=settle)
    arm.wait_settled()
