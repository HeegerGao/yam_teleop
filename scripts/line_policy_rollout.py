"""Closed-loop rollout of the three YAM line-drawing e2e policies.

This is the line-policy entry point for :mod:`box_folding_policy_rollout`.  It deliberately
reuses that module's camera, follower, safety, recording, receding-horizon, and continuation
implementation; the policy checkpoints have the same observation/action contract.  The only
new choice is ``--line-type``:

    top       make a horizontal line at the top
    bottom    make a horizontal line at the bottom (``horizontal_down`` in the HF dataset)
    vertical  make a vertical line

The downloaded bundles default to ``~/line_policy/<line-type>``.  Each bundle must contain its
own ``norm_stats.json`` and ``checkpoints/goal_step_150000.pt`` because the three training runs
have different normalisation statistics.  ``--policy-root`` and ``--checkpoint`` still allow an
explicit bundle/checkpoint override.

Usage:
    python scripts/line_policy_rollout.py --line-type top
    python scripts/line_policy_rollout.py --line-type bottom \
        --execute --replan-every 12 --continuation rtc
    python scripts/line_policy_rollout.py --line-type vertical \
        --execute --replan-every 12 --continuation rtc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import box_folding_policy_rollout as rollout
import tyro

DEFAULT_LINE_POLICY_ROOT = "~/line_policy"


@dataclass
class Args(rollout.Args):
    """Line-policy selection plus all options supported by the shared rollout."""

    line_type: Literal["top", "bottom", "vertical"] = "top"
    """Which line policy to run. top/bottom are horizontal lines at that image location."""

    line_policy_root: str = DEFAULT_LINE_POLICY_ROOT
    """Parent of the downloaded top, bottom, and vertical policy bundles."""
    goto_start: bool = True
    """With --execute, slew to this line task's recorded start_pose.json before rollout."""


def _select_policy(args: Args) -> Path:
    """Apply the line-type default without overriding an explicitly selected policy bundle."""
    if args.policy_mode != "e2e":
        raise SystemExit("[error] line_policy_rollout supports only --policy-mode e2e")
    if args.checkpoint_preset != "default":
        raise SystemExit("[error] box-folding --checkpoint-preset values do not apply to line policies")

    if args.policy_root is None:
        args.policy_root = str(Path(args.line_policy_root).expanduser() / args.line_type)

    root = Path(args.policy_root).expanduser().resolve()
    if args.start_pose_file is None:
        args.start_pose_file = str(root / "start_pose.json")
    if args.checkpoint is None:
        # Pin the released checkpoint rather than silently selecting another file placed beside it.
        args.checkpoint = str(root / "checkpoints" / "goal_step_150000.pt")
    return Path(args.checkpoint).expanduser().resolve()


def main(args: Args) -> None:
    checkpoint = _select_policy(args)
    print(f"[line] {args.line_type}: {checkpoint}")
    rollout.main(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
