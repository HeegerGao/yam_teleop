"""Where a Wan subgoal generation's seconds actually go, and what --wan-steps buys back.

The rollout's online replanning (``box_folding_policy_rollout.py --wan-replan-chunks``) is paced
by how long one generation takes, so this measures it directly instead of guessing: the pipeline
is loaded once and then run at several step counts, and the wall clock is fitted as

    seconds = fixed + steps * per_step

``fixed`` is everything that does not scale with the denoiser -- text encode, the VAE encode of
the conditioning frame, and above all the VAE *decode* of all 61 output frames, which is charged
once no matter how few steps ran. That is the term that decides whether cutting ``--wan-steps``
is worth anything: below the point where ``fixed`` dominates, halving the steps stops halving the
time. ``--no-tiled`` runs the same sweep with untiled VAE decoding, which is the one knob that
moves ``fixed``.

Run it on an idle GPU -- a rollout planning next to it inflates every number.

    python scripts/box_folding_wan_bench.py                     # 4/8/16/40 steps, tiled + untiled
    python scripts/box_folding_wan_bench.py --steps 8 16 --repeats 3
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from box_folding_policy import DEFAULT_GOAL_POLICY_ROOT
from box_folding_wan_subgoals import DEFAULT_WAN_MODELS, WAN_NUM_FRAMES, generate_with, load_pipeline


@dataclass
class Args:
    steps: Tuple[int, ...] = (4, 8, 16, 40)
    """Denoising-step counts to time. 40 is the rollout's default."""
    repeats: int = 2
    """Timed runs per step count. The first run of all is dropped as a warm-up."""
    tiled: bool = True
    """Tiled VAE decoding, as the rollout uses it."""
    also_untiled: bool = True
    """Repeat the sweep with tiled=False, to price the VAE decode."""
    compare: bool = True
    """Also score each step count's images against the highest one, same seed. Time is only half
    the question -- fewer steps is only a win if the subgoals it draws are the same subgoals."""
    out_dir: Optional[str] = None
    """Write each step count's set to <out_dir>/steps_NN/, to look at them rather than trust PSNR."""
    episode: str = "~/yam_data/box_folding/episode_0000"
    """Recorded episode whose first top frame conditions every generation."""
    image: Optional[str] = None
    """Condition on this image instead of --episode."""
    policy_root: str = DEFAULT_GOAL_POLICY_ROOT
    models: str = DEFAULT_WAN_MODELS
    seed: int = 0


def _conditioning_frame(args: Args) -> np.ndarray:
    if args.image:
        bgr = cv2.imread(str(Path(args.image).expanduser()))
        if bgr is None:
            raise SystemExit(f"[error] cannot read {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(str(Path(args.episode).expanduser() / "top.mp4"))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"[error] no frames in {args.episode}/top.mp4")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit(steps: List[int], seconds: List[float]) -> Tuple[float, float]:
    """Least-squares ``seconds = fixed + steps * per_step``."""
    a = np.vstack([np.ones(len(steps)), np.asarray(steps, float)]).T
    fixed, per_step = np.linalg.lstsq(a, np.asarray(seconds, float), rcond=None)[0]
    return float(fixed), float(per_step)


def main(args: Args) -> None:
    import torch

    top = _conditioning_frame(args)
    print(f"[bench] conditioning frame {top.shape}, {WAN_NUM_FRAMES} output frames")
    print(f"[bench] device {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    started = time.monotonic()
    pipe, prompt = load_pipeline(policy_root=args.policy_root, models=args.models)
    print(f"[bench] pipeline loaded in {time.monotonic() - started:.1f}s -- this is what keep_alive saves")

    def timed(steps: int, tiled: bool) -> float:
        torch.cuda.synchronize()
        t0 = time.monotonic()
        generate_with(pipe, prompt, top, steps=steps, seed=args.seed, cfg_scale=1.0, tiled=tiled)
        torch.cuda.synchronize()
        return time.monotonic() - t0

    timed(min(args.steps), args.tiled)  # warm-up: first-call autotuning is not the steady state
    for tiled in (True, False) if args.also_untiled else (args.tiled,):
        rows: List[Tuple[int, float]] = []
        print(f"\n[bench] tiled VAE decode = {tiled}")
        for steps in sorted(args.steps):
            runs = [timed(steps, tiled) for _ in range(max(1, args.repeats))]
            best = min(runs)
            rows.append((steps, best))
            print(f"  {steps:3d} steps  {best:6.2f}s   (runs: {', '.join(f'{r:.2f}' for r in runs)})")
        fixed, per_step = _fit([s for s, _ in rows], [t for _, t in rows])
        worst = max(t for _, t in rows)
        print(
            f"  fit: {fixed:.2f}s fixed + {per_step * 1e3:.0f} ms/step "
            f"-- at 40 steps the denoiser is {per_step * 40 / worst:.0%} of the time"
        )

    if not args.compare:
        return
    ref_steps = max(args.steps)
    print(f"\n[bench] images against {ref_steps} steps (same seed, so this is the schedule alone)")
    sets = {
        n: generate_with(pipe, prompt, top, steps=n, seed=args.seed, cfg_scale=1.0, tiled=args.tiled)
        for n in sorted(args.steps)
    }
    if args.out_dir:
        from box_folding_wan_subgoals import save_subgoals

        for n, imgs in sets.items():
            save_subgoals(imgs, Path(args.out_dir).expanduser() / f"steps_{n:02d}")
        print(f"  sets written under {Path(args.out_dir).expanduser()}")
    ref = sets[ref_steps].astype(np.float64)
    for n in sorted(args.steps):
        err = sets[n].astype(np.float64) - ref
        mae = float(np.abs(err).mean())
        mse = float((err**2).mean())
        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
        print(f"  {n:3d} steps  mean|delta| {mae:5.2f}/255   PSNR vs {ref_steps} steps {psnr:5.1f} dB")


if __name__ == "__main__":
    main(tyro.cli(Args))
