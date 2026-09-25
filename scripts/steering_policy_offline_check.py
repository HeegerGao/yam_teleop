"""Open-loop check of a steering π₀.₅ checkpoint against recorded demos -- no robot, no cameras.

Feeds recorded episodes from ``~/yam_data/<task>`` through exactly the path the real-robot eval uses
(PolicyClient -> steering_policy_server.py in the pi05 venv): the raw 14-D measured joint state and the
three mp4 frames decoded to RGB at their recorded resolution. Every ``--stride`` frames it compares
the first ``--horizon`` steps of the predicted chunk with the operator's recorded leader command
(``action_joint_pos_*``), and prints the same error for a "hold the current pose" baseline.

The policy was trained on these episodes, so this is a plumbing check, not a generalisation score:
a wrong view order, a BGR/RGB swap or a state layout mix-up shows up as an error no better than the
hold baseline; a correct pipeline is well under it.

Usage:
    python scripts/steering_policy_offline_check.py --task cloth
    python scripts/steering_policy_offline_check.py --task pingpong --episodes 0 10 20 --stride 15
    python scripts/steering_policy_offline_check.py --task cup_pingpong
    python scripts/steering_policy_offline_check.py --task cup_pingpong --steer
    python scripts/steering_policy_offline_check.py --task push_cube   # demos: ~/yam_data/push_cube_shovel_correct
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering_policy_models import RELEASE_ROOT, Model, new_episode, resolve_assets
from steering_policy_server import (
    DEFAULT_PYTHON,
    VIEWS,
    PolicyClient,
    demo_dir,
)

_ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
_GRIPPER = [6, 13]


@dataclass
class Args:
    task: str = "cloth"
    model: Model = "pi"
    coast_beta: float = 1.0
    """COAST conceptor strength in [0, 1]; 1 reproduces the published baseline."""
    model_root: str = str(RELEASE_ROOT)
    episodes: List[int] = field(default_factory=lambda: [0, 20])
    stride: int = 30
    """Plan every N recorded frames (30 = once per second)."""
    horizon: int = 10
    """Chunk steps compared against the recorded commands."""
    data_root: str = "~/yam_data"
    checkpoint: Optional[str] = None
    prompt: Optional[str] = None
    pi05_python: str = str(DEFAULT_PYTHON)
    seed: Optional[int] = 0
    steer: bool = False
    """Alias for --model steeract_enc."""
    max_decisions: Optional[int] = None
    """Limit each episode's replay; use 3 for a smoke test."""
    save_root: str = "~/yam_eval/offline"
    """Numbered task/model/episode_NNNN directories, including predicted chunks."""


def _read_frames(ep_dir: Path, keep: range) -> Dict[str, Dict[int, np.ndarray]]:
    """RGB frames at the ``keep`` indices only -- a whole 1080p episode is ~8 GB decoded."""
    wanted = set(keep)
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for view in VIEWS:
        cap = cv2.VideoCapture(str(ep_dir / f"{view}.mp4"))
        frames: Dict[int, np.ndarray] = {}
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in wanted:
                frames[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        out[view] = frames
    return out


def _create_output_directory(path: Path) -> Path:
    path.mkdir(exist_ok=False)
    return path


def main(args: Args) -> None:
    if args.stride < 1 or args.horizon < 1 or (args.max_decisions is not None and args.max_decisions < 1):
        raise ValueError("stride, horizon and max-decisions must be positive")
    if args.steer:
        if args.model not in ("pi", "steeract_enc"):
            raise ValueError("--steer conflicts with --model")
        args.model = "steeract_enc"
    assets = resolve_assets(args.task, args.model, Path(args.model_root), args.checkpoint)
    ckpt = assets.checkpoint
    prompt = args.prompt or args.task
    client = PolicyClient(
        ckpt,
        prompt,
        python=Path(args.pi05_python),
        seed=args.seed,
        model=args.model,
        coast_beta=args.coast_beta,
        steer_deployment_root=assets.root / "deployment",
        steer_sae_root=assets.sae,
        steer_method_root=assets.method,
        selection=assets.selection if args.model in ("drvla", "coast") else None,
    )
    print(f"[check] starting the policy server for {ckpt} ...")
    totals = {"policy_arm": [], "hold_arm": [], "policy_grip": [], "hold_grip": []}
    try:
        info = client.start()
        print(f"[check] {info}")
        if args.horizon > info["chunk_size"]:
            raise ValueError("--horizon exceeds the checkpoint chunk size")
        for ep in args.episodes:
            ep_dir = Path(args.data_root).expanduser() / demo_dir(args.task) / f"episode_{ep:04d}"
            if not (ep_dir / "meta.json").is_file():
                print(f"[episode {ep:04d}] no complete episode at {ep_dir} -- skipped")
                continue
            client.reset_episode()
            low = np.load(ep_dir / "low_dim.npz")
            state = np.concatenate([low["joint_pos_left"], low["joint_pos_right"]], axis=1).astype(np.float32)
            action = np.concatenate([low["action_joint_pos_left"], low["action_joint_pos_right"]], axis=1)
            plan_ticks = range(0, len(state) - args.horizon, args.stride)
            if args.max_decisions is not None:
                plan_ticks = plan_ticks[: args.max_decisions]
            frames = _read_frames(ep_dir, plan_ticks)
            n = len(state)
            errs: Dict[str, List[float]] = {k: [] for k in totals}
            chunks: List[np.ndarray] = []
            frame_indices: List[int] = []
            for t in plan_ticks:
                if any(t not in frames[v] for v in VIEWS):
                    break  # a video shorter than the low-dim stream
                chunk = client.infer(state[t], {v: frames[v][t] for v in VIEWS})
                chunks.append(chunk)
                frame_indices.append(t)
                gt = action[t : t + args.horizon]
                pe = np.abs(chunk[: args.horizon] - gt)
                he = np.abs(state[t][None] - gt)
                errs["policy_arm"].append(float(pe[:, _ARM].mean()))
                errs["hold_arm"].append(float(he[:, _ARM].mean()))
                errs["policy_grip"].append(float(pe[:, _GRIPPER].mean()))
                errs["hold_grip"].append(float(he[:, _GRIPPER].mean()))
            for k, v in errs.items():
                totals[k].extend(v)
            if not chunks:
                raise RuntimeError(f"No complete observations replayed from {ep_dir}")
            output = new_episode(Path(args.save_root), args.task, args.model, _create_output_directory)
            print(f"[record] {output}")
            np.savez_compressed(output / "actions.npz", actions=np.stack(chunks), frame_indices=frame_indices)
            (output / "meta.json").write_text(
                json.dumps(
                    {
                        "task": args.task,
                        "model": args.model,
                        "source_episode": str(ep_dir),
                        "executed": False,
                        "policy_info": info,
                        "seed": args.seed,
                        "stride": args.stride,
                        "horizon": args.horizon,
                        "errors": errs,
                        "steering_telemetry": client.steering_telemetry,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"[episode {ep:04d}] {len(errs['policy_arm'])} plans over {n} frames | arm joints MAE "
                f"policy {np.mean(errs['policy_arm']):.4f} rad vs hold {np.mean(errs['hold_arm']):.4f} | "
                f"gripper policy {np.mean(errs['policy_grip']):.4f} vs hold {np.mean(errs['hold_grip']):.4f}"
            )
    finally:
        client.close()
    if not totals["policy_arm"]:
        raise RuntimeError("No episodes evaluated; check --data-root and --episodes")
    print(
        f"[total] arm joints MAE policy {np.mean(totals['policy_arm']):.4f} rad vs hold "
        f"{np.mean(totals['hold_arm']):.4f} | gripper policy {np.mean(totals['policy_grip']):.4f} vs hold "
        f"{np.mean(totals['hold_grip']):.4f} (first {args.horizon} steps of each chunk)"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
