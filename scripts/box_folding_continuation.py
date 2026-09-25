"""Chunk-to-chunk continuation for the box_folding flow policies: RTC, BID and Legato.

A receding-horizon rollout switches from the chunk it is executing to a fresh one every
``--replan-every`` ticks. Sampled independently, the two chunks need not agree where they overlap,
and the arm jerks at the switch -- or, worse, the new sample picks a different mode (grasp from the
other side) half-way through a motion. Each method here makes the new chunk continue the old one:

    rtc     Real-Time Execution of Action Chunking Flow Policies (Black et al., 2025,
            arXiv:2506.07339). Training-free inpainting. Every Euler step is guided towards the
            previous chunk with a PiGDM vector-Jacobian product, weighted by a soft mask: 1 over the
            first ``d`` steps (the inference delay -- executed by the *old* chunk before the new one
            arrives), exponentially decaying over the rest of the overlap, 0 past it. Costs one
            backward pass per Euler step (~3x the denoise time).
    bid     Bidirectional Decoding (Liu et al., ICLR 2025, arXiv:2408.17355). Training-free
            rejection sampling. Draw ``N`` chunks in one batch and keep the one minimising
            backward coherence (decayed L2 distance to the previous chunk over the overlap) plus
            forward contrast (close to the K nearest other strong samples, far from the K nearest
            samples of a *weak*, early checkpoint). Without a weak checkpoint only the positive
            half of the contrast is used.
    legato  Learning Native Continuation for Action Chunking Flow Policies (arXiv:2602.12978).
            Start from ``w * A_ref + (1 - w) * noise`` and re-impose ``w * A_ref`` before every Euler
            step, with ``w`` = 1 over the delay then a linear ramp to 0. **Legato is a training
            method**: the network sees ``w`` as a 15th input channel and is trained on a reshaped
            velocity target so this per-step guidance is consistent with its dynamics. A checkpoint
            whose ``action_in`` takes ``action_dim + 1`` inputs is run natively; every checkpoint
            released so far takes ``action_dim``, and for those this is the guidance-only
            approximation (hard per-step inpainting, plus a final re-imposition of ``A_ref`` that a
            native model does not need). Expect it to be the weakest of the three until a Legato
            checkpoint exists.

Index conventions, shared by all three (``H`` = chunk length, 32):

    s  offset: steps of the previous chunk already elapsed when the new observation was taken.
       The previous chunk's steps ``s..H-1`` line up with the new chunk's ``0..H-s-1``; past its
       end the reference repeats its last step, which is what the executor holds there anyway.
    d  delay: new-chunk steps that will already have been executed by the old chunk when the new
       one arrives. The rollout estimates it from measured plan ages.

Everything operates in the policy's normalised action space ([-1, 1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

CONTINUATION_METHODS = ("none", "rtc", "bid", "legato")


@dataclass
class ContinuationConfig:
    """Which continuation method a policy samples with, and its knobs. Picklable (spawn-safe)."""

    method: str = "none"
    rtc_beta: float = 5.0
    """RTC's clip on the PiGDM guidance weight. The paper's default; it only binds at early flow
    times, where the unclipped weight goes to infinity."""
    bid_samples: int = 16
    """BID batch size N, drawn from the strong policy (and again from the weak one)."""
    bid_mode_size: int = 3
    """BID's K: forward contrast averages the K nearest positive / negative samples."""
    bid_decay: float = 0.9
    """BID's rho: backward-coherence weight rho**tau over the overlap."""
    bid_weak_checkpoint: Optional[str] = "auto"
    """Weak policy for BID's negative samples. auto = the lowest-step checkpoint next to the strong
    one (e2e bundle: goal_step_050000.pt), none = positive-only contrast, or a path."""
    legato_ramp: Optional[int] = None
    """Legato ramp length r. Default H - s - d, the paper's r + s + d = H."""

    def __post_init__(self) -> None:
        if self.method not in CONTINUATION_METHODS:
            raise ValueError(f"continuation method must be one of {CONTINUATION_METHODS}, got {self.method!r}")
        if self.bid_samples < 2:
            raise ValueError("bid_samples must be >= 2")
        if self.bid_mode_size < 1:
            raise ValueError("bid_mode_size must be >= 1")
        if not 0.0 < self.bid_decay <= 1.0:
            raise ValueError("bid_decay must be in (0, 1]")
        if self.rtc_beta <= 0.0:
            raise ValueError("rtc_beta must be > 0")

    def describe(self) -> str:
        if self.method == "rtc":
            return f"rtc (beta {self.rtc_beta:g})"
        if self.method == "bid":
            return f"bid (N {self.bid_samples}, K {self.bid_mode_size}, rho {self.bid_decay:g})"
        if self.method == "legato":
            return f"legato (ramp {'H-s-d' if self.legato_ramp is None else self.legato_ramp})"
        return "none"


# ---------------------------------------------------------------------------------- schedules
def reference_chunk(prev: np.ndarray, offset: int) -> np.ndarray:
    """``PadLast(prev[s:H])``: the previous chunk re-indexed onto the new chunk's timeline."""
    prev = np.asarray(prev)
    h = prev.shape[0]
    tail = prev[min(max(int(offset), 0), h - 1) :]
    pad = np.repeat(tail[-1:], h - tail.shape[0], axis=0)
    return np.concatenate([tail, pad], axis=0)


def rtc_weights(horizon: int, delay: int, offset: int) -> np.ndarray:
    """RTC's soft mask (eq. 5): 1 for i < d, ``c (e^c - 1) / (e - 1)`` with
    ``c = (H-s-i) / (H-s-d+1)`` for d <= i < H-s, 0 beyond."""
    h, d = int(horizon), int(np.clip(delay, 0, horizon))
    end = h - int(offset)
    w = np.zeros(h, dtype=np.float64)
    w[:d] = 1.0
    if end > d:
        i = np.arange(d, end, dtype=np.float64)
        c = (end - i) / (end - d + 1)
        w[d:end] = c * np.expm1(c) / (math.e - 1.0)
    return w


def legato_schedule(horizon: int, delay: int, offset: int, ramp: Optional[int] = None) -> np.ndarray:
    """Legato's ``w``: 1 over the delay, a linear ramp of length ``r`` down towards 0, then 0."""
    h, d = int(horizon), int(np.clip(delay, 0, horizon))
    r = max(0, h - int(offset) - d) if ramp is None else max(0, int(ramp))
    w = np.zeros(h, dtype=np.float64)
    w[:d] = 1.0
    i = np.arange(d, min(h, d + r), dtype=np.float64)
    w[d : d + i.size] = (d + r - i) / (r + 1)
    return w


def bid_backward_weights(horizon: int, offset: int, decay: float) -> np.ndarray:
    """``rho**tau`` over the overlap with the previous chunk (at least its first step)."""
    overlap = max(1, int(horizon) - int(offset))
    w = np.zeros(int(horizon), dtype=np.float64)
    w[:overlap] = float(decay) ** np.arange(overlap)
    return w


# ---------------------------------------------------------------------------------- samplers
Velocity = Callable[[Any, float], Any]
"""``(x [B,H,A], flow time tau) -> velocity [B,H,A]``; tau = 0 is noise, 1 is data."""


def euler(velocity: Velocity, x: Any, steps: int) -> Any:
    """The plain rectified-flow integration the policy has always used."""
    dt = 1.0 / steps
    for k in range(steps):
        x = x + dt * velocity(x, k * dt)
    return x


def rtc_sample(velocity: Velocity, x: Any, ref: Any, weights: Any, steps: int, beta: float) -> Any:
    """RTC guided inference (Algorithm 1): PiGDM guidance towards ``ref`` under the soft mask.

    ``x`` is the initial noise ``[1,H,A]``, ``ref`` ``[1,H,A]``, ``weights`` ``[H]``. Needs autograd
    through the velocity network, so it must not run under ``torch.inference_mode``.
    """
    import torch

    w = weights.view(1, -1, 1).to(x.dtype)
    dt = 1.0 / steps
    for k in range(steps):
        tau = k * dt
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            v = velocity(xg, tau).float()
            a1 = xg + (1.0 - tau) * v  # the one-step estimate of the clean chunk
            (g,) = torch.autograd.grad(a1, xg, grad_outputs=(ref - a1).detach() * w)
        if tau <= 0.0:
            gw = beta
        else:
            r2 = (1.0 - tau) ** 2 / (tau**2 + (1.0 - tau) ** 2)
            gw = min(beta, (1.0 - tau) / (tau * r2))
        x = x.detach() + dt * (v.detach() + gw * g)
    return x


def legato_sample(velocity: Velocity, x: Any, ref: Any, omega: Any, steps: int, *, native: bool) -> Any:
    """Legato inference (Algorithm 2): ``Y_k = (1-w) X_k + w A_ref``, ``X_{k+1} = Y_k + dt f(Y_k)``.

    ``X_0`` is the noise, so ``Y_0`` is the paper's action-noise mixture. A native model returns
    ``X_N``; a vanilla one gets ``A_ref`` re-imposed once more, because its last Euler step moves the
    frozen prefix off the reference (a native model is trained so that it does not).
    """
    w = omega.view(1, -1, 1).to(x.dtype)
    dt = 1.0 / steps
    for k in range(steps):
        y = (1.0 - w) * x + w * ref
        x = y + dt * velocity(y, k * dt)
    return x if native else (1.0 - w) * x + w * ref


def bid_select(
    strong: Any, weak: Optional[Any], ref: Optional[Any], back_weights: Optional[Any], mode_size: int
) -> Dict[str, Any]:
    """BID's criterion over ``strong [N,H,A]``: argmin of backward coherence + forward contrast.

    Distances are summed per-step L2 norms. Positives are the *other* strong samples (a sample is
    not its own neighbour), negatives the weak samples; each side averages its K nearest.
    """
    import torch

    n = strong.shape[0]

    def pairwise(a: Any, b: Any) -> Any:
        return torch.linalg.vector_norm(a[:, None] - b[None], dim=-1).sum(-1)  # [Na,Nb]

    pos = pairwise(strong, strong)
    pos.fill_diagonal_(float("inf"))
    forward = pos.topk(min(mode_size, n - 1), dim=1, largest=False).values.mean(1)
    if weak is not None:
        neg = pairwise(strong, weak)
        forward = forward - neg.topk(min(mode_size, weak.shape[0]), dim=1, largest=False).values.mean(1)
    backward = torch.zeros_like(forward)
    if ref is not None and back_weights is not None:
        backward = (torch.linalg.vector_norm(strong - ref, dim=-1) * back_weights.view(1, -1)).sum(-1)
    total = backward + forward
    best = int(total.argmin().item())
    return {
        "index": best,
        "chunk": strong[best : best + 1],
        "backward": float(backward[best].item()),
        "forward": float(forward[best].item()),
    }


def expand_prefix_cache(cached: Dict[str, Any], n: int) -> Dict[str, Any]:
    """A batch-1 prefix KV cache broadcast to batch ``n`` (views, no copy until attention concats)."""
    if n == 1:
        return cached
    cache = [(k.expand(n, *k.shape[1:]), v.expand(n, *v.shape[1:])) for k, v in cached["cache"]]
    return {**cached, "cache": cache}
