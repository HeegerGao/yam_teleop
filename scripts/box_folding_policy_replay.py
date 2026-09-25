"""Offline check of the box_folding policy against a recorded episode -- no robot, no cameras.

Replays one ``<save_root>/box_folding/episode_NNNN/`` directory through exactly the pipeline
``scripts/box_folding_policy_rollout.py`` runs on the real arms (same resize, same view order,
same 10 fps history, same 28-D state) and compares each predicted chunk with the leader commands
that were actually recorded.

This is the cheapest way to catch a broken observation contract: a swapped view, a BGR/RGB flip
or a wrong state layout does not raise, it just makes the predictions bad. Compare the reported
error against the ``hold`` baseline (predict "stay where you are"); a policy that is wired up
correctly beats it clearly, especially on the later chunk steps.

``--policy-mode goal`` checks the hierarchical policy the same way. Its subgoals come from
``--subgoal-dir`` when a Wan set has been generated for this episode, and otherwise from the
bundle's ground-truth K=15 annotation for it -- section 6 of the bundle README, and the only way
to tell a broken goal pathway from a bad Wan generation.

Usage:
    python scripts/box_folding_policy_replay.py                                  # episode_0000
    python scripts/box_folding_policy_replay.py --episode ~/yam_data/box_folding/episode_0007
    python scripts/box_folding_policy_replay.py --policy-mode goal               # ground-truth goals
    python scripts/box_folding_policy_replay.py --policy-mode goal --subgoal-dir ~/subgoals
    python scripts/box_folding_policy_replay.py --predict-every 5 --max-ticks 100
    python scripts/box_folding_policy_replay.py --save-plot /tmp/replay.png      # per-joint traces
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from box_folding_continuation import ContinuationConfig
from box_folding_policy import (
    CONTROL_DT,
    DEFAULT_TORCH_HOME,
    VIEW_ORDER,
    BoxFoldingPolicy,
    build_state,
    default_policy_root,
    resize_to_policy,
)

FRAME_STRIDE = 3
"""30 fps recording -> the policy's 10 fps timeline (``build_box_folding_policy_hdf5.py``)."""


@dataclass
class Args:
    episode: str = "~/yam_data/box_folding/episode_0000"
    """Recorded episode directory (top.mp4, left_wrist.mp4, right_wrist.mp4, low_dim.npz)."""
    policy_mode: Literal["e2e", "goal"] = "e2e"
    """e2e = the goal-free policy; goal = the one conditioned on a 6-subgoal window."""
    policy_root: Optional[str] = None
    """The downloaded bundle; default depends on --policy-mode (see the rollout script)."""
    checkpoint: Optional[str] = None
    """Explicit checkpoint. Default: the newest one in <policy_root>/<policy_mode>/."""
    subgoal_dir: Optional[str] = None
    """--policy-mode goal: a Wan set (subgoal_NN.png). Default: the bundle's K=15 ground truth
    for this episode, which separates a broken goal pathway from a bad generation."""
    goal_advance: Literal["similarity", "time"] = "similarity"
    """How the subgoal window walks forward; the same knob the rollout has."""
    goal_stage_seconds: float = 6.0
    device: str = "cuda:0"
    torch_home: str = DEFAULT_TORCH_HOME
    """Torch Hub cache holding the c-radio_v3-h weights."""
    flow_steps: int = 10
    """Euler steps for the flow integration; 5/10/20 score the same here."""
    radio_dtype: str = "float16"
    """float32 | float16 | bfloat16 for the frozen RADIO tower."""
    trunk_dtype: str = "float16"
    """float32 | float16 autocast for the DiT trunk."""
    seed: Optional[int] = 0
    """Seed for the flow noise, so repeated runs are comparable. None = fresh randomness."""
    predict_every: int = 10
    """Run a prediction every N ticks (10 ticks = 1 s of the 10 fps timeline)."""
    max_ticks: int = 0
    """Stop after this many 10 fps ticks; 0 = the whole episode."""
    horizon: int = 0
    """Chunk steps to score; 0 = the full 32-step chunk."""
    save_plot: Optional[str] = None
    """Write a per-joint predicted-vs-recorded plot here (needs matplotlib)."""
    continuation: Literal["none", "rtc", "bid", "legato"] = "none"
    """Sample each chunk as a continuation of the previous prediction (--predict-every steps
    earlier), as the rollout does; the summary then reports the jump at each simulated switch."""
    continuation_delay_steps: int = 2
    """Simulated inference delay d: the switch happens at step d of the new chunk."""
    rtc_beta: float = 5.0
    bid_samples: int = 16
    bid_mode_size: int = 3
    bid_decay: float = 0.9
    bid_weak_checkpoint: str = "auto"
    legato_ramp: Optional[int] = None


def _decode_views(ep: Path, stride: int) -> Dict[str, List[np.ndarray]]:
    """Every ``stride``-th frame of each view, already 240x320 RGB."""
    out: Dict[str, List[np.ndarray]] = {}
    for view in VIEW_ORDER:
        path = ep / f"{view}.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"missing video: {path}")
        cap = cv2.VideoCapture(str(path))
        frames: List[np.ndarray] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(resize_to_policy(frame, bgr=True))
            i += 1
        cap.release()
        out[view] = frames
        print(f"[replay] {view}: {i} frames -> {len(frames)} ticks")
    return out


def _states_and_actions(ep: Path, n_ticks: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """``(states[T,28], actions[T,14])`` on the 10 fps timeline, straight from ``low_dim.npz``."""
    z = np.load(ep / "low_dim.npz")
    idx = np.arange(0, int(z["joint_pos_left"].shape[0]), stride)[:n_ticks]
    states = np.stack(
        [
            build_state(
                z["joint_pos_left"][i],
                z["joint_pos_right"][i],
                z["eef_pos_left"][i],
                z["eef_quat_left"][i],
                z["eef_pos_right"][i],
                z["eef_quat_right"][i],
            )
            for i in idx
        ]
    )
    actions = np.concatenate([z["action_joint_pos_left"][idx], z["action_joint_pos_right"][idx]], axis=1)
    return states.astype(np.float32), actions.astype(np.float32)


def _chunk_target(actions: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """The recorded chunk for anchor ``t``, right-padded by repeating the last command."""
    idx = np.clip(np.arange(t, t + horizon), 0, actions.shape[0] - 1)
    return actions[idx]


def _annotated_subgoals(root: Path, ep: Path, frames: Dict[str, List[np.ndarray]]) -> np.ndarray:
    """The bundle's K=15 ground-truth subgoals for this episode, as ``[15,240,320,3]`` uint8 RGB.

    The annotation indexes the 10 fps timeline, which is the timeline ``frames`` is already on,
    so the top view is simply indexed -- no second decode and no stride arithmetic.
    """
    index = json.loads((root / "annotation" / "box_folding_subgoal_index_k15.json").read_text())
    key = next((k for k in (ep.name, ep.name.replace("episode", "box_folding")) if k in index), None)
    if key is None:
        raise SystemExit(
            f"[error] {ep.name} has no K=15 annotation (keys look like {list(index)[:3]}); "
            "pass --subgoal-dir with a generated set instead"
        )
    ticks = index[key]
    ticks = ticks["subgoal_frames"] if isinstance(ticks, dict) else ticks
    top = frames["top"]
    print(f"[replay] ground-truth subgoals for {key}: ticks {ticks}")
    return np.stack([top[min(int(t), len(top) - 1)] for t in ticks])


def main(args: Args) -> None:
    ep = Path(args.episode).expanduser()
    if not ep.is_dir():
        raise SystemExit(f"[error] episode not found: {ep}")
    args.policy_root = args.policy_root or default_policy_root(args.policy_mode)

    frames = _decode_views(ep, FRAME_STRIDE)
    n_ticks = min(len(f) for f in frames.values())
    if args.max_ticks:
        n_ticks = min(n_ticks, args.max_ticks)
    states, actions = _states_and_actions(ep, n_ticks, FRAME_STRIDE)
    n_ticks = min(n_ticks, states.shape[0])
    print(f"[replay] {ep.name}: {n_ticks} ticks @ {1 / CONTROL_DT:.0f} fps")

    policy = BoxFoldingPolicy(
        args.policy_root,
        checkpoint=args.checkpoint,
        mode=args.policy_mode,
        device=args.device,
        flow_steps=args.flow_steps,
        radio_dtype=args.radio_dtype,
        trunk_dtype=args.trunk_dtype,
        goal_advance=args.goal_advance,
        goal_stage_seconds=args.goal_stage_seconds,
        torch_home=args.torch_home,
        seed=args.seed,
        continuation=ContinuationConfig(
            method=args.continuation,
            rtc_beta=args.rtc_beta,
            bid_samples=args.bid_samples,
            bid_mode_size=args.bid_mode_size,
            bid_decay=args.bid_decay,
            bid_weak_checkpoint=args.bid_weak_checkpoint,
            legato_ramp=args.legato_ramp,
        ),
    )
    print(f"[replay] {policy.describe()}")
    if args.policy_mode == "goal":
        if args.subgoal_dir:
            from box_folding_wan_subgoals import load_subgoals

            subgoals = np.stack([resize_to_policy(img, bgr=False) for img in load_subgoals(args.subgoal_dir)])
            print(f"[replay] {len(subgoals)} generated subgoals from {args.subgoal_dir}")
        else:
            subgoals = _annotated_subgoals(Path(args.policy_root).expanduser().resolve(), ep, frames)
        policy.set_goals(subgoals)
    horizon = args.horizon or policy.action_chunk_len

    errs: List[np.ndarray] = []
    hold_errs: List[np.ndarray] = []
    anchors: List[int] = []
    preds: List[np.ndarray] = []
    latencies: List[float] = []
    jumps: List[float] = []
    prev_full: Optional[np.ndarray] = None
    d = int(args.continuation_delay_steps)

    for t in range(n_ticks):
        policy.observe({v: frames[v][t] for v in VIEW_ORDER}, states[t])
        if t % args.predict_every or not policy.ready:
            continue
        t0 = time.perf_counter()
        full = policy.predict(prev_chunk=prev_full, prev_offset=args.predict_every, delay_steps=d)
        if prev_full is not None:
            # The simulated switch: the executor leaves the old chunk for the new one at step d.
            h_full = full.shape[0]
            old = prev_full[min(args.predict_every + d, h_full - 1)]
            jumps.append(float(np.max(np.abs(np.delete(full[min(d, h_full - 1)] - old, [6, 13])))))
        prev_full = full
        chunk = full[:horizon]
        latencies.append((time.perf_counter() - t0) * 1e3)
        target = _chunk_target(actions, t, horizon)
        # "hold": the trivial policy that repeats the current leader command for the whole chunk.
        hold = np.repeat(actions[t][None, :], horizon, axis=0)
        errs.append(np.abs(chunk - target))
        hold_errs.append(np.abs(hold - target))
        anchors.append(t)
        preds.append(chunk)
        goal = f"  subgoal {policy.goal_cursor + 1}/{policy.n_goals}" if policy.uses_goals else ""
        print(
            f"  t={t:4d}  |err| mean {errs[-1].mean():.4f} (hold {hold_errs[-1].mean():.4f})  "
            f"step0 {errs[-1][0].mean():.4f}  step{horizon - 1} {errs[-1][-1].mean():.4f}  "
            f"{latencies[-1]:.0f} ms{goal}"
        )

    if not errs:
        raise SystemExit("[error] no predictions -- episode shorter than the history")

    e = np.stack(errs)  # [N, horizon, 14]
    h = np.stack(hold_errs)
    print(f"\n[summary] {len(errs)} chunks from {ep.name}, horizon {horizon}")
    print(f"  mean |err| rad   policy {e.mean():.4f}   hold {h.mean():.4f}   ratio {e.mean() / h.mean():.2f}")
    print(f"  step 0           policy {e[:, 0].mean():.4f}   hold {h[:, 0].mean():.4f}")
    print(f"  last step        policy {e[:, -1].mean():.4f}   hold {h[:, -1].mean():.4f}")
    per_dim = e.mean(axis=(0, 1))
    names = [f"L{j + 1}" for j in range(6)] + ["Lgrip"] + [f"R{j + 1}" for j in range(6)] + ["Rgrip"]
    print("  per-dim mean |err|: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(names, per_dim, strict=True)))
    print(f"  inference {np.mean(latencies):.0f} ms mean, {np.max(latencies):.0f} ms max")
    if jumps:
        print(
            f"  switch jump (continuation {args.continuation}, s={args.predict_every}, d={d}): "
            f"mean {np.mean(jumps):.4f}  median {np.median(jumps):.4f}  max {np.max(jumps):.4f} rad"
        )
    if e.mean() >= h.mean():
        print(
            "  [warn] the policy is no better than holding still -- check the view order, the "
            "BGR/RGB conversion and the state layout before touching the robot"
        )

    if args.save_plot:
        _plot(anchors, np.stack(preds), actions, names, Path(args.save_plot).expanduser())


def _plot(anchors: List[int], preds: np.ndarray, actions: np.ndarray, names: List[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    for d in range(14):
        ax = axes[d % 7][d // 7]
        ax.plot(np.arange(actions.shape[0]), actions[:, d], color="0.3", lw=1.0, label="recorded")
        for i, t in enumerate(anchors):
            ax.plot(np.arange(t, t + preds.shape[1]), preds[i, :, d], lw=1.0, alpha=0.8)
        ax.set_ylabel(names[d])
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1][0].set_xlabel("tick (10 fps)")
    axes[-1][1].set_xlabel("tick (10 fps)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
