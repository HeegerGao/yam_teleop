"""One command that proves the whole ``demo_full`` stack runs on this machine, without a robot.

Every stage drives the *production* code path, so a pass here means the rollout script will get
the same behaviour -- this file contains no second implementation of anything:

``e2e``   :class:`box_folding_policy.BoxFoldingPolicy` in e2e mode on a recorded episode, scored
          against the recorded leader commands. Catches a broken observation contract.
``wan``   :func:`box_folding_wan_subgoals.generate_subgoals` -- the ~20 GB Wan2.2 + LoRA load and
          one 61-frame generation. Writes the 15 subgoals to ``--out-dir`` so ``goal`` (and a
          rollout, via ``--subgoal-dir``) can reuse them instead of regenerating.
``goal``  the same policy class in goal mode, fed those subgoals. Falls back to the bundle's
          ground-truth K=15 annotation when no generated set exists.

    uv run python scripts/box_folding_demo_full_smoke.py --stage all

For a full-episode score rather than a single anchor use ``box_folding_policy_replay.py``, which
takes the same ``--policy-mode`` and ``--subgoal-dir``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from box_folding_policy import (
    DEFAULT_GOAL_POLICY_ROOT,
    DEFAULT_TORCH_HOME,
    VIEW_ORDER,
    BoxFoldingPolicy,
    build_state,
    resize_to_policy,
)
from box_folding_wan_subgoals import (
    DEFAULT_WAN_MODELS,
    generate_subgoals,
    load_subgoals,
    save_subgoals,
)

DEFAULT_EPISODE = "~/yam_data/box_folding/episode_0000"
RECORD_FPS, POLICY_FPS = 30, 10
STRIDE = RECORD_FPS // POLICY_FPS


def read_episode(episode: Path, *, n_ticks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(pixels [L,V,240,320,3] uint8 RGB, state [28], actions [T,14])`` for the episode start.

    ``pixels`` is the 10 fps history ending at the ``n_ticks``-th tick -- what
    :meth:`BoxFoldingPolicy.observe` would have accumulated by then, left-clamping included.
    """
    per_view: Dict[str, List[np.ndarray]] = {}
    for view in VIEW_ORDER:
        cap = cv2.VideoCapture(str(episode / f"{view}.mp4"))
        got: List[np.ndarray] = []
        for i in range((n_ticks - 1) * STRIDE + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"{view}.mp4 ended after {i} frames")
            if i % STRIDE == 0:
                got.append(resize_to_policy(bgr, bgr=True))
        cap.release()
        per_view[view] = got

    pixels = np.stack([np.stack([per_view[v][t] for v in VIEW_ORDER]) for t in range(n_ticks)])
    z = np.load(episode / "low_dim.npz")
    idx = np.arange(0, int(z["joint_pos_left"].shape[0]), STRIDE)
    a = idx[n_ticks - 1]
    state = build_state(
        z["joint_pos_left"][a],
        z["joint_pos_right"][a],
        z["eef_pos_left"][a],
        z["eef_quat_left"][a],
        z["eef_pos_right"][a],
        z["eef_quat_right"][a],
    ).astype(np.float32)
    actions = np.concatenate([z["action_joint_pos_left"][idx], z["action_joint_pos_right"][idx]], axis=1)
    return pixels, state, actions.astype(np.float32)


def _first_top_frame(episode: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(episode / "top.mp4"))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frames in {episode / 'top.mp4'}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _annotated_subgoals(root: Path, episode: Path, n_wanted: int) -> np.ndarray:
    """The bundle's ground-truth K=15 subgoals for this episode, at policy resolution."""
    import json

    index = json.loads((root / "annotation" / "box_folding_subgoal_index_k15.json").read_text())
    key = next((k for k in (episode.name, episode.name.replace("episode", "box_folding")) if k in index), None)
    if key is None:
        raise SystemExit(f"[error] no K=15 annotation for {episode.name}; run --stage wan first")
    ticks = index[key]
    ticks = ticks["subgoal_frames"] if isinstance(ticks, dict) else ticks
    print(f"goal window from the ground-truth annotation {key}: ticks {ticks[:n_wanted]} ...")
    cap = cv2.VideoCapture(str(episode / "top.mp4"))
    wanted = {int(t) * STRIDE for t in ticks}
    got: Dict[int, np.ndarray] = {}
    i = 0
    while wanted and i <= max(wanted):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            got[i] = resize_to_policy(bgr, bgr=True)
        i += 1
    cap.release()
    last = max(got)
    return np.stack([got.get(int(t) * STRIDE, got[last]) for t in ticks])


def _run_policy(mode: str, args: "Args", subgoals: Optional[np.ndarray]) -> None:
    """Build the policy in ``mode``, feed it one history, print the chunk it asks for."""
    bundle = Path(args.bundle).expanduser().resolve()
    episode = Path(args.episode).expanduser().resolve()
    policy = BoxFoldingPolicy(
        bundle,
        mode=mode,
        flow_steps=args.flow_steps,
        torch_home=args.torch_home,
        seed=args.seed,
    )
    print(policy.describe())
    if mode == "goal":
        policy.set_goals(subgoals if subgoals is not None else _annotated_subgoals(bundle, episode, 6))

    pixels, state, actions = read_episode(episode, n_ticks=policy.history_len)
    for tick in range(policy.history_len):
        policy.observe({v: pixels[tick][i] for i, v in enumerate(VIEW_ORDER)}, state)
    started = time.perf_counter()
    chunk = policy.predict()
    latency = (time.perf_counter() - started) * 1e3

    ref = actions[policy.history_len - 1 : policy.history_len - 1 + chunk.shape[0]]
    n = min(len(ref), len(chunk))
    err = np.abs(chunk[:n] - ref[:n])
    cursor = f", subgoal {policy.goal_cursor + 1}/{policy.n_goals}" if policy.uses_goals else ""
    print(f"chunk {chunk.shape} in {latency:.0f} ms, range [{chunk.min():+.3f}, {chunk.max():+.3f}]{cursor}")
    print(f"vs the recorded leader commands over {n} steps: mean |err| {err.mean():.4f} rad")


@dataclass
class Args:
    stage: Literal["e2e", "wan", "goal", "all"] = "all"
    """Which model to exercise. ``all`` runs e2e, then wan, then goal on the generated subgoals."""
    bundle: str = DEFAULT_GOAL_POLICY_ROOT
    """The demo_full folder (checkpoints, configs, norm_stats, planning package)."""
    episode: str = DEFAULT_EPISODE
    """A recorded yam_data/box_folding episode to draw observations from."""
    out_dir: Optional[str] = None
    """Where ``wan`` writes its subgoal PNGs (default ``<bundle>/subgoals_<episode>``)."""
    wan_models: str = DEFAULT_WAN_MODELS
    """DiffSynth model base path holding Wan-AI/Wan2.2-TI2V-5B."""
    torch_home: str = DEFAULT_TORCH_HOME
    flow_steps: int = 10
    wan_steps: int = 40
    seed: int = 0


def main(args: Args) -> None:
    episode = Path(args.episode).expanduser().resolve()
    out_dir = Path(args.out_dir or Path(args.bundle).expanduser() / f"subgoals_{episode.name}").expanduser()

    if args.stage in ("e2e", "all"):
        print("\n=== stage e2e =========================================================")
        _run_policy("e2e", args, None)

    if args.stage in ("wan", "all"):
        print("\n=== stage wan =========================================================")
        started = time.perf_counter()
        subgoals = generate_subgoals(
            _first_top_frame(episode),
            policy_root=args.bundle,
            models=args.wan_models,
            steps=args.wan_steps,
            seed=args.seed,
        )
        paths = save_subgoals(subgoals, out_dir)
        print(f"{len(paths)} subgoals in {time.perf_counter() - started:.1f}s -> {out_dir}")

    if args.stage in ("goal", "all"):
        print("\n=== stage goal ========================================================")
        generated = None
        if out_dir.is_dir() and list(out_dir.glob("subgoal_*.png")):
            generated = np.stack([resize_to_policy(x, bgr=False) for x in load_subgoals(out_dir)])
            print(f"goal window from the generated subgoals in {out_dir}")
        _run_policy("goal", args, generated)

    print("\nall requested stages ran.")


if __name__ == "__main__":
    main(tyro.cli(Args))
