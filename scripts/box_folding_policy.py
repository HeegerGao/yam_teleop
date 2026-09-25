"""Runtime for the ``box_folding`` flow policies (trained on ``yam_data/box_folding``).

Shared by ``scripts/box_folding_policy_rollout.py`` (closed loop on the real arms) and
``scripts/box_folding_policy_replay.py`` (offline check against a recorded episode).

A policy bundle -- checkpoints, ``norm_stats.json``, the ``planning`` package and the frozen
DecisionNCE instruction embedding -- is a folder of ``ChongkaiGao/planning`` on HF, downloaded to
``--policy-root``. Two policies share this runtime, selected by ``mode``, checked against the
checkpoint, and defaulting to *different* bundles because they were not released together:

    e2e    goal-free: cameras and proprioception, nothing else. Default bundle
           ``~/box_folding_policy/e2e`` (HF ``box_folding_policy/e2e``), whose newest checkpoint
           is ``checkpoints/goal_step_150000.pt``. See DEFAULT_POLICY_ROOT for why this rather
           than demo_full's newer e2e checkpoint.
    goal   additionally conditioned on a 6-image window of the 15 subgoals Wan2.2 generates once
           at episode start (see box_folding_wan_subgoals.py). Call
           :meth:`BoxFoldingPolicy.set_goals` once before the first plan. Default bundle
           ``~/box_folding_policy/demo_full`` (HF ``box_folding_policy/demo_full``), the only one
           carrying a goal checkpoint and the Wan LoRA; see its ``INSTALL.md``.

Both layouts are understood: ``<root>/<mode>/*.pt`` (demo_full) and ``<root>/checkpoints/*.pt``
(the older e2e-only bundle). ``norm_stats.json`` belongs to its bundle -- the 55- and 105-episode
statistics differ -- so a checkpoint and a root must never be mixed across bundles.

Observation contract (fixed by training; every part of it fails silently when wrong):

    views     exactly 3 cameras, always in the order ``("left_wrist", "right_wrist", "top")``.
              That is alphabetical, which is what ``sorted(norm_stats["views"])`` handed the
              training dataset and therefore the order the view embedding learned.
    image     240x320 (HxW) RGB, area-interpolated resize, ImageNet mean/std normalised.
              The recorder hands out BGR frames (OpenCV / RealSense) while training decoded
              mp4 to RGB, so a live frame must be converted before it is resized.
    history   4 frames on a 10 fps timeline (t-3, t-2, t-1, t). The cameras run at 30 fps, so
              one observation every 3rd frame; at episode start the first frame is repeated.
    state     28-D: joint_pos_left(7), joint_pos_right(7), eef_pos_left(3), eef_quat_left(4),
              eef_pos_right(3), eef_quat_right(4) -- quaternions wxyz, poses the FK of the
              arm-only MJCF ``gripper`` body in each arm's own base frame, exactly what
              ``scripts/bimanual_teleop_record.py`` logs. Min-max normalised to [-1, 1].
    language  the single training instruction, byte-encoded (not tokenised).

Output: ``[32, 14]`` absolute **leader joint commands** on the same 10 fps timeline -- left 6
arm joints + gripper, then right -- de-normalised from [-1, 1] with ``action_p01``/``action_p99``.
The gripper entry is normalised openness in 0-1, the same unit the follower server takes.

RADIO (``c-radio_v3-h``) is encoded live inside :meth:`BoxFoldingPolicy.predict`, on demand: an
observation only buffers uint8 frames, and a plan encodes the (at most 12) images its history
snapshot has not already got features for. Encoding on every 10 Hz tick instead would spend GPU
on frames no plan reads, in contention with the planner.

Where a plan's time goes, measured on an RTX 5090 at the default ``--replan-every 24``, where
consecutive plans share no history frames and all 12 images are encoded fresh:

                         fp32    fp16     note
    RADIO tower       110 ms   31 ms     --radio-dtype, default float16
    prefix (3.6k tok)  22 ms   12 ms     --trunk-dtype, default float16
    10 Euler steps     75 ms   60 ms     --flow-steps; the cost is linear in the step count
    -------------------------------------------------------------------------------
    one plan          207 ms  103 ms

At a small --replan-every the RADIO line shrinks (3 ticks apart, one of the four history frames
is shared, so 9 images: 89 ms fp32 / 24 ms fp16) but plans run eight times as often.

The goal policy costs the same: its prefix is 3638 tokens against the e2e run's 3614, and its
subgoal features are encoded once by :meth:`set_goals` rather than per plan. fp16 moves the
predicted chunk by 0.004 normalised units against fp32 -- bfloat16 moved it by 0.25, so bf16 is
allowed for RADIO but refused for the trunk. Capturing the Euler loop in a CUDA graph is
bit-exact and saves a further 17 ms, but it needs the prefix KV copied into static buffers every
plan; at fp16 that is 15% of a plan for a lot of fragility, so it is deliberately not done.

The newest checkpoint for the mode is the default, ranked by step number.

``continuation`` (a :class:`box_folding_continuation.ContinuationConfig`) swaps the plain Euler
integration for RTC, BID or Legato, all of which make a new chunk continue the previous one; pass
that chunk to :meth:`BoxFoldingPolicy.predict_history` as ``prev_chunk``. See
box_folding_continuation.py for what each costs and what Legato needs from the checkpoint.

Weights: RADIO comes from Torch Hub, so ``TORCH_HOME`` must hold ``hub/NVlabs_RADIO_main`` and
``hub/checkpoints/c-radio_v3-h_half.pth.tar`` (~1.7 GB, downloaded on first use).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from box_folding_continuation import (
    ContinuationConfig,
    bid_backward_weights,
    bid_select,
    euler,
    expand_prefix_cache,
    legato_sample,
    legato_schedule,
    reference_chunk,
    rtc_sample,
    rtc_weights,
)

VIEW_ORDER: Tuple[str, ...] = ("left_wrist", "right_wrist", "top")
"""Alphabetical -- the order the view embedding was trained with. Never reorder."""
IMAGE_HW: Tuple[int, int] = (240, 320)
CONTROL_DT = 0.1
"""One action / one history frame per 100 ms: the canonical 10 fps timeline."""
LANGUAGE_PAD_TOKEN = 256
GRIPPER_CLOSED, GRIPPER_OPEN = 0.0, 1.0
"""The gripper's normalised command range, and which end is which.

``i2rt.robots.utils.JointMapper`` maps a command of ``x`` to ``x * (open - closed) + closed`` over
the per-robot ``gripper_limits = [closed, open]``, so 1.0 is fully open. The demonstrations agree:
every ``yam_data/box_folding`` episode starts at ~0.99 and drops to ~0.005 while gripping."""
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# The two canonical YAM poses a run ends at, 7-D (6 arm joints + normalised gripper). Copied from
# ``scripts/box_packing/poses.py`` -- same physical arm, same numbers, and that file is the one
# with the full rationale. If either is ever re-measured, change both.
WORKING_Q = np.array([0.0, 1.10, 1.90, -1.20, 0.0, 0.0, GRIPPER_OPEN])
"""Half-open, elbow up, clear of the table: the waypoint a fold goes through.

Folding straight down from wherever a rollout stopped would sweep the arm across the table and
through whatever it was just working on; from here the fold comes down beside the base."""
FOLDED_Q = np.array([0.0, 0.02, 0.07, -0.13, 0.0, 0.0, GRIPPER_OPEN])
"""The folded rest pose -- close to where the arm sits with no torque.

This is where a run should end. Tearing the followers down zeroes every motor torque, so an arm
left anywhere else drops from there; parked here it just settles."""
DEFAULT_POLICY_ROOT = "~/box_folding_policy/e2e"
"""Default bundle for ``mode="e2e"``: the 55-episode run (``checkpoints/goal_step_150000.pt``).

Deliberately *not* the newer ``demo_full`` e2e checkpoint. On the real arms the 105-episode 300k
model asks for about half the joint velocity and is the only one that produces idle ticks --
measured over ten rollouts on one afternoon, five per checkpoint, same rig and same precision --
and it is no better on the offline replay either:

    checkpoint            replay |err|   median commanded speed   ticks asking for no motion
    150k / 55 episodes       0.0136 rad         0.224 rad/s                 0.0%
    300k / 105 episodes      0.0207 rad         0.121 rad/s                 2.3%

Both replay figures are on ``episode_0000``, which is in *both* training sets, so they measure
fit rather than generalisation; the hardware columns are what the choice actually rests on.
Pass ``--policy-root ~/box_folding_policy/demo_full`` to run the newer one."""
DEFAULT_GOAL_POLICY_ROOT = "~/box_folding_policy/demo_full"
"""Default bundle for ``mode="goal"``: the only one with a goal checkpoint and the Wan LoRA."""
DEFAULT_TORCH_HOME = "~/torch_home"
POLICY_MODES = ("e2e", "goal")
"""``e2e`` = the goal-free policy; ``goal`` = the one conditioned on a Wan subgoal window."""


def default_policy_root(mode: str) -> str:
    """The bundle a mode uses when ``--policy-root`` is not given.

    They differ because the two policies do not live in the same download: only ``demo_full``
    ships a goal checkpoint, and only the older bundle holds the e2e checkpoint that is currently
    the better one on hardware.
    """
    if mode not in POLICY_MODES:
        raise ValueError(f"mode must be one of {POLICY_MODES}, got {mode!r}")
    return DEFAULT_POLICY_ROOT if mode == "e2e" else DEFAULT_GOAL_POLICY_ROOT


def resize_to_policy(image: np.ndarray, *, bgr: bool) -> np.ndarray:
    """One camera frame -> ``[240, 320, 3]`` uint8 RGB, matching the training preprocessing.

    Training decoded the mp4s with ``ffmpeg -vf scale=320:240:flags=area`` straight to rgb24;
    ``INTER_AREA`` is the same filter. The aspect ratio is *not* preserved (the 1920x1080 top
    camera is squeezed into 4:3) -- that squeeze is part of the contract, not a bug.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {image.shape}")
    h, w = IMAGE_HW
    out = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB) if bgr else out


def default_checkpoint(root: Path, mode: str) -> Path:
    """The highest-step checkpoint for ``mode`` under a policy bundle.

    Two layouts are accepted. The ``demo_full`` bundle keeps one directory per model
    (``<root>/e2e``, ``<root>/goal``); the older e2e-only bundle keeps a single
    ``<root>/checkpoints``, which is used for ``mode="e2e"`` when no ``e2e/`` directory exists.

    Ranked by the step number in the name, not by the name: the runs write ``goal_step_%06d``, so
    plain sorting works only until a seventh digit appears and ``goal_step_1000000.pt`` silently
    sorts below ``goal_step_950000.pt``. Anything without a step number sorts first, so a
    hand-named file never outranks a real checkpoint.
    """
    candidates = [root / mode] + ([root / "checkpoints"] if mode == "e2e" else [])
    for directory in candidates:
        ckpts = list(directory.glob("*.pt")) if directory.is_dir() else []
        if ckpts:

            def step_of(path: Path) -> Tuple[int, str]:
                digits = re.findall(r"\d+", path.stem)
                return (int(digits[-1]) if digits else -1, path.name)

            return max(ckpts, key=step_of)
    tried = ", ".join(str(d) for d in candidates)
    raise FileNotFoundError(f"no {mode} checkpoint found (looked in {tried})")


class BoxFoldingPolicy:
    """The flow policy plus its observation history, RADIO cache and (de)normalisation.

    Call :meth:`observe` once per 10 fps tick and :meth:`predict` whenever a fresh action chunk
    is wanted; ``predict`` is safe to call from a worker thread while the caller keeps feeding
    observations, and it works off a snapshot of the history taken under the lock.
    """

    def __init__(
        self,
        policy_root: Optional[str | Path] = None,
        *,
        checkpoint: Optional[str | Path] = None,
        mode: str = "e2e",
        device: str = "cuda:0",
        flow_steps: int = 10,
        radio_dtype: str = "float16",
        trunk_dtype: str = "float16",
        goal_advance: str = "similarity",
        goal_stage_seconds: float = 6.0,
        torch_home: Optional[str | Path] = DEFAULT_TORCH_HOME,
        seed: Optional[int] = None,
        continuation: Optional[ContinuationConfig] = None,
    ) -> None:
        if mode not in POLICY_MODES:
            raise ValueError(f"mode must be one of {POLICY_MODES}, got {mode!r}")
        self.mode = str(mode)
        self.root = Path(policy_root or default_policy_root(self.mode)).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"policy root not found: {self.root}")
        if torch_home:
            th = Path(torch_home).expanduser()
            if th.is_dir() or not os.environ.get("TORCH_HOME"):
                os.environ["TORCH_HOME"] = str(th)

        # The bundle's ``planning`` package is imported from the policy root; keep it last on the
        # path so it can never shadow a repo module of the same name.
        if str(self.root) not in sys.path:
            sys.path.append(str(self.root))

        import torch  # imported here so `--help` and the CLI arg parsing stay torch-free

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.flow_steps = int(flow_steps)
        # Half precision is the default for both towers because it is where the plan latency is.
        # Measured on this RTX 5090 for one plan's worth of work (9 new images at --replan-every 3):
        # RADIO 89 -> 24 ms, prefix 22 -> 12 ms, 10 denoise steps 75 -> 60 ms. Training itself
        # consumed fp16 RADIO features, and the replay check scores the same either way.
        # bfloat16 is accepted for RADIO but *not* for the trunk: it moved the predicted chunk by
        # 0.25 normalised units against fp32, where fp16 moved it by 0.0036.
        self.radio_dtype = getattr(torch, str(radio_dtype))
        if self.radio_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"radio_dtype must be float32/float16/bfloat16, got {radio_dtype}")
        self.trunk_dtype = getattr(torch, str(trunk_dtype))
        if self.trunk_dtype not in (torch.float32, torch.float16):
            raise ValueError(f"trunk_dtype must be float32/float16, got {trunk_dtype}")

        self.stats: Dict[str, Any] = json.loads((self.root / "norm_stats.json").read_text())
        if sorted(self.stats["views"]) != list(VIEW_ORDER):
            raise RuntimeError(f"norm_stats views {self.stats['views']} do not match {VIEW_ORDER}")
        if tuple(self.stats["image_size"]) != IMAGE_HW:
            raise RuntimeError(f"norm_stats image_size {self.stats['image_size']} != {IMAGE_HW}")
        self.instruction: str = str(self.stats["language_instruction"])
        a_lo = np.asarray(self.stats["action_p01"], np.float32)
        self._a_lo, self._a_span = a_lo, np.maximum(np.asarray(self.stats["action_p99"], np.float32) - a_lo, 1e-6)
        s_lo = np.asarray(self.stats["state_min"], np.float32)
        self._s_lo, self._s_span = s_lo, np.maximum(np.asarray(self.stats["state_max"], np.float32) - s_lo, 1e-6)

        ckpt = Path(checkpoint).expanduser() if checkpoint else default_checkpoint(self.root, self.mode)
        self.model, self.cfg, self.step, self.uses_goals = self._build(ckpt)
        self.history_len = int(self.model.history_len)
        self.action_chunk_len = int(self.model.action_chunk_len)
        self.action_dim = int(self.model.action_dim)
        self.state_dim = int(self.stats["state_dim"])
        self.checkpoint = ckpt
        self.goal_sequence_len = int(self.cfg["data"]["goal_sequence_len"]) if self.uses_goals else 0

        # --- continuation (RTC / BID / Legato) -------------------------------------------------
        self.continuation = continuation or ContinuationConfig()
        # A Legato-trained network reads the guidance schedule as one extra action channel.
        self.legato_native = int(self.model.action_in.in_features) == self.action_dim + 1
        self.weak_model: Optional[Any] = None
        self.weak_checkpoint: Optional[Path] = None
        if self.continuation.method == "bid":
            self.weak_checkpoint = self._weak_checkpoint(ckpt, self.continuation.bid_weak_checkpoint)
            if self.weak_checkpoint is not None:
                # No RADIO tower of its own: the strong model's features are handed to it.
                self.weak_model, _, _, _ = self._build(self.weak_checkpoint, load_radio_backbone=False)
        self.last_continuation: Dict[str, Any] = {}
        """What the last plan's sampler did (method, offset, delay, BID scores); for logging."""

        ids, lens = self._encode_instruction(self.instruction)
        self._lang_ids, self._lang_lens = ids.to(self.device), lens.to(self.device)
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device).manual_seed(int(seed))

        self._lock = threading.Lock()
        self._frames: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.history_len)
        """(id, [V,H,W,3] uint8 RGB) for the last ``history_len`` ticks."""
        self._radio_cache: Dict[int, Any] = {}
        """Frame id -> RADIO patches [V,P,C], kept only while that frame is in the history."""
        self._top_global: Dict[int, Any] = {}
        """Frame id -> the top view's RADIO summary [C], used to place the goal cursor."""
        self._next_id = 0
        self._scratch_id = -1
        self._state: Optional[np.ndarray] = None
        self.n_observations = 0

        # --- goal stream (mode="goal" only) --------------------------------------------------
        if goal_advance not in ("similarity", "time"):
            raise ValueError(f"goal_advance must be 'similarity' or 'time', got {goal_advance!r}")
        self.goal_advance = str(goal_advance)
        self.goal_stage_seconds = float(goal_stage_seconds)
        self._goal_local: Optional[Any] = None
        """``[K,P,C]`` RADIO patches for the whole subgoal sequence, encoded once by set_goals."""
        self._goal_global: Optional[Any] = None
        self._goal_unit: Optional[Any] = None
        """``[K,C]`` L2-normalised summaries, for the cosine match against the live top view."""
        self.n_goals = 0
        self.goal_cursor = 0
        """Index of the next *unreached* subgoal; the window is ``[cursor, cursor+goal_len)``."""
        self._goal_t0: Optional[float] = None

    def _uncached_id(self) -> int:
        """A fresh negative id: never reused, so the frame is encoded and then evicted."""
        self._scratch_id -= 1
        return self._scratch_id

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _weak_checkpoint(strong: Path, spec: Optional[str]) -> Optional[Path]:
        """BID's weak policy: ``auto`` = the lowest-step checkpoint beside the strong one, if any."""
        if spec is None or str(spec).lower() in ("", "none"):
            return None
        if str(spec).lower() != "auto":
            path = Path(spec).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"BID weak checkpoint not found: {path}")
            return path

        def step_of(path: Path) -> int:
            digits = re.findall(r"\d+", path.stem)
            return int(digits[-1]) if digits else -1

        others = [p for p in strong.parent.glob("*.pt") if p.resolve() != strong.resolve()]
        earlier = [p for p in others if 0 <= step_of(p) < step_of(strong)]
        return min(earlier, key=step_of) if earlier else None

    def _build(
        self, checkpoint: Path, *, load_radio_backbone: bool = True
    ) -> Tuple[Any, Dict[str, Any], Optional[int], bool]:
        """Mirror of the bundle's ``inference_example.build_policy``, with absolute asset paths.

        Everything structural comes from the checkpoint's own training config, so a newer
        checkpoint of the same run loads without touching this file. ``load_radio_backbone=True``
        because the robot has no precomputed RADIO cache, and ``resnet_pretrained=False`` because
        the trained ResNet34 weights are in the checkpoint.

        Whether the checkpoint has a goal stream is read from its **weights**, not its config: the
        e2e run sets ``model.use_goal_conditioning: false`` explicitly but the goal run simply
        omits the key (it is the builder's default), so the config alone cannot tell them apart.
        ``goal_step_emb`` exists only when the goal Perceiver was built.
        """
        from planning.model.flow_matching_goal_resnet_radio_dit_mv import (
            FlowMatchingGoalResNetRadioDiTMVPolicy,
        )

        blob = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, m, d = blob["cfg"], blob["cfg"]["model"], blob["cfg"]["data"]
        uses_goals = "goal_step_emb" in blob["model"]
        if uses_goals != (self.mode == "goal"):
            has = "a goal stream" if uses_goals else "no goal stream"
            raise RuntimeError(
                f"--policy-mode {self.mode} but {checkpoint.name} has {has}; pass the other "
                f"checkpoint (the demo_full bundle keeps them in <root>/e2e and <root>/goal)"
            )
        # Released task bundles name this cache after their task suite (for example
        # ``yam_line_decisionnce_t.npz``), while the older box-folding bundles predate that
        # convention and keep ``box_folding_decisionnce_t.npz`` at their root.  The embedding
        # table itself is restored from the checkpoint; the cache also supplies the ordered
        # instruction strings needed to select the correct row at inference time.
        configured_cache = Path(str(m.get("decisionnce_cache", "box_folding_decisionnce_t.npz"))).name
        cache_candidates = [self.root / configured_cache, self.root / "box_folding_decisionnce_t.npz"]
        decisionnce_cache = next((path for path in cache_candidates if path.is_file()), cache_candidates[0])
        model = FlowMatchingGoalResNetRadioDiTMVPolicy(
            state_dim=int(self.stats["state_dim"]),
            action_dim=int(self.stats["action_dim"]),
            history_len=int(d["history_len"]),
            action_chunk_len=int(d["action_chunk_len"]),
            d_model=int(m["d_model"]),
            nhead=int(m["nhead"]),
            num_layers=int(m["num_layers"]),
            dim_feedforward=int(m["dim_feedforward"]),
            dropout=float(m["dropout"]),
            use_state=True,
            use_language=True,
            radio_version=str(m["radio_version"]),
            radio_summary_dim=int(m["radio_summary_dim"]),
            radio_local_dim=int(m["radio_local_dim"]),
            radio_patch_tokens=300,
            radio_patch_grid=(IMAGE_HW[0] // 16, IMAGE_HW[1] // 16),
            radio_image_size=IMAGE_HW,
            freeze_radio=True,
            load_radio_backbone=load_radio_backbone,
            use_goal_conditioning=uses_goals,
            goal_sequence_len=int(d["goal_sequence_len"]),
            goal_latent_tokens=int(m.get("goal_latent_tokens", 4)),
            goal_perceiver_layers=int(m.get("goal_perceiver_layers", 2)),
            goal_drop_prob=float(m.get("goal_drop_prob", 0.3)),
            language_encoder_type="decisionnce",
            decisionnce_cache=str(decisionnce_cache),
            task_aux_loss_weight=0.0,
            normalize_cond_tokens=True,
            language_drop_prob=float(m["language_drop_prob"]),
            keep_one_conditioning=True,
            num_views=len(VIEW_ORDER),
            resnet_pretrained=False,
        )
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        # ``radio.*`` is the frozen tower, absent by design (training read cached features);
        # anything else missing means this builder and the checkpoint disagree.
        bad = [k for k in missing if not k.startswith("radio.")]
        if bad or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}")
        return model.to(self.device).eval(), cfg, blob.get("step"), uses_goals

    def _encode_instruction(self, text: str, max_chars: int = 256) -> Tuple[Any, Any]:
        """UTF-8 byte tokens padded to ``max_chars`` -- the exact encoding used in training."""
        raw = text.encode("utf-8")[:max_chars] or b" "
        ids = self.torch.full((1, max_chars), LANGUAGE_PAD_TOKEN, dtype=self.torch.long)
        ids[0, : len(raw)] = self.torch.tensor(list(raw), dtype=self.torch.long)
        return ids, self.torch.tensor([len(raw)], dtype=self.torch.long)

    # ------------------------------------------------------------------ normalisation
    def normalize_state(self, raw_state: np.ndarray) -> np.ndarray:
        s = np.asarray(raw_state, np.float32).reshape(-1)
        if s.shape[0] != self.state_dim:
            raise ValueError(f"state must be {self.state_dim}-D, got {s.shape[0]}")
        return 2.0 * (s - self._s_lo) / self._s_span - 1.0

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(action, np.float32) - self._a_lo) / self._a_span - 1.0

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action, np.float32) + 1.0) / 2.0 * self._a_span + self._a_lo

    # ------------------------------------------------------------------ goal stream
    def set_goals(self, goal_images_rgb: np.ndarray) -> int:
        """Encode the whole subgoal sequence once and start its cursor at 0.

        The RADIO features of the K subgoals never change during an episode -- Wan generates them
        once, conditioned on the first top frame -- so they are encoded here and only *sliced* per
        plan. Encoding the 6-image window inside every plan instead would add 16 ms (fp16) to each
        one for features that are bit-identical from plan to plan.

        Args:
            goal_images_rgb: ``[K, 240, 320, 3]`` uint8 RGB, already at the policy's resolution.
        """
        if not self.uses_goals:
            raise RuntimeError("this checkpoint has no goal stream (--policy-mode e2e)")
        torch = self.torch
        arr = np.asarray(goal_images_rgb)
        if arr.ndim != 4 or arr.shape[1:] != (*IMAGE_HW, 3) or arr.dtype != np.uint8:
            raise ValueError(f"goals must be [K,{IMAGE_HW[0]},{IMAGE_HW[1]},3] uint8 RGB, got {arr.shape} {arr.dtype}")
        if arr.shape[0] < self.goal_sequence_len:
            raise ValueError(f"need at least {self.goal_sequence_len} subgoals, got {arr.shape[0]}")
        unit = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(self.device).float().div_(255.0)
        local, global_ = self._encode_radio(unit)
        self._goal_local, self._goal_global = local, global_
        self._goal_unit = global_ / global_.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.n_goals = int(arr.shape[0])
        self.goal_cursor = 0
        self._goal_t0 = None
        return self.n_goals

    def _advance_cursor(self, top_global: Optional[Any], now: float) -> None:
        """Move the subgoal cursor forward for this plan. Monotonic: it never rewinds.

        ``similarity`` is the bundle README's robust option: cosine-match the live top view
        against the subgoal summaries and treat the best match as the stage just completed. The
        search is confined to ``[cursor-1, cursor+2]`` so one noisy frame -- and the domain gap
        between a camera frame and a Wan frame is real -- cannot skip the policy to the end of the
        task. ``time`` is the simple option: one stage per ``goal_stage_seconds``.
        """
        if self.n_goals == 0:
            return
        if self._goal_t0 is None:
            self._goal_t0 = now
        if self.goal_advance == "time":
            elapsed = now - self._goal_t0
            self.goal_cursor = max(self.goal_cursor, min(int(elapsed / self.goal_stage_seconds), self.n_goals - 1))
            return
        if top_global is None or self._goal_unit is None:
            return
        lo = max(0, self.goal_cursor - 1)
        hi = min(self.n_goals, self.goal_cursor + 2)
        q = top_global.reshape(-1)
        q = q / q.norm().clamp_min(1e-6)
        sims = self._goal_unit[lo:hi] @ q.to(self._goal_unit.dtype)
        reached = lo + int(sims.argmax().item())
        # The cursor is the next *unreached* subgoal, and it is capped one short of the end so the
        # window always has a real subgoal in it rather than K repeats of the final frame.
        self.goal_cursor = min(max(self.goal_cursor, reached + 1), self.n_goals - 1)

    def _goal_window(self) -> Tuple[Any, Any, Any]:
        """``(local [1,G,P,C], global [1,G,C], valid [1,G])`` for the cursor's window.

        The window is the next ``goal_sequence_len`` subgoals; past the end it repeats the last
        one, which is exactly the padding the training dataset applied.
        """
        torch = self.torch
        idx = [min(self.goal_cursor + i, self.n_goals - 1) for i in range(self.goal_sequence_len)]
        sel = torch.tensor(idx, device=self.device)
        return (
            self._goal_local.index_select(0, sel).unsqueeze(0),
            self._goal_global.index_select(0, sel).unsqueeze(0),
            torch.ones(1, self.goal_sequence_len, dtype=torch.bool, device=self.device),
        )

    # ------------------------------------------------------------------ observation
    def reset(self) -> None:
        """Forget the history; the next observation starts a fresh (left-clamped) episode.

        The subgoal features survive -- they belong to the episode's Wan generation, not to the
        observation history -- but their cursor goes back to the start of the task.
        """
        with self._lock:
            self._frames.clear()
            self._state = None
            self.n_observations = 0
        self._radio_cache.clear()
        self._top_global.clear()
        self.goal_cursor = 0
        self._goal_t0 = None

    def observe(self, frames_rgb: Dict[str, np.ndarray], raw_state: np.ndarray) -> None:
        """Push one 10 fps tick: ``{view: [240,320,3] uint8 RGB}`` plus the raw 28-D state.

        Pure bookkeeping -- the frames are kept as uint8 and nothing touches the GPU until
        :meth:`predict` needs them. Encoding every tick would burn ~34 ms of GPU per tick to
        produce features for frames no plan ever reads (a plan uses 4 of them and, on the real
        robot, runs about once a second), and it burns that GPU in the *caller's* thread, in
        contention with the planner that is trying to finish.

        The first tick of an episode fills all 4 history slots with itself, which is the same
        left-clamping the training dataset applied at episode start.
        """
        missing = [v for v in VIEW_ORDER if v not in frames_rgb]
        if missing:
            raise KeyError(f"missing views {missing}; need all of {VIEW_ORDER}")
        stack = np.stack([frames_rgb[v] for v in VIEW_ORDER])  # [V,H,W,3] uint8
        if stack.shape[1:3] != IMAGE_HW:
            raise ValueError(f"frames must be {IMAGE_HW} (HxW), got {stack.shape[1:3]}")
        if stack.dtype != np.uint8:
            raise TypeError(f"frames must be uint8 RGB, got {stack.dtype}")
        state = np.asarray(raw_state, np.float32).reshape(-1).copy()
        if state.shape[0] != self.state_dim:
            raise ValueError(f"state must be {self.state_dim}-D, got {state.shape[0]}")

        with self._lock:
            first = not self._frames
            self._next_id += 1
            # One id for the repeated first frame, so the left-clamped history encodes it once.
            for _ in range(self.history_len if first else 1):
                self._frames.append((self._next_id, stack))
            self._state = state
            self.n_observations += 1

    def _encode_radio(self, unit_images: Any) -> Tuple[Any, Any]:
        """Frozen RADIO: ``[N,3,H,W]`` in [0,1] -> ``([N,P,C], [N,C])`` patch tokens and summaries.

        The summary is free here (the tower produces it either way) and the goal cursor needs it,
        so both come back rather than only the patches the prefix consumes.
        """
        from planning.model.radio_encode_utils import radio_encode_unit_nchw_local_global

        torch = self.torch
        use_amp = self.radio_dtype is not torch.float32 and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.radio_dtype, enabled=use_amp):
            local, global_ = radio_encode_unit_nchw_local_global(
                self.model.radio, unit_images, microbatch=int(unit_images.shape[0])
            )
        return local.float().clone(), global_.float().clone()

    @property
    def ready(self) -> bool:
        with self._lock:
            return len(self._frames) == self.history_len and self._state is not None

    def last_state(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._state is None else self._state.copy()

    # ------------------------------------------------------------------ inference
    def predict(self, *, flow_steps: Optional[int] = None, **continuation: Any) -> np.ndarray:
        """Rectified-flow Euler integration -> ``[32, 14]`` absolute leader joint commands.

        Snapshots the history under the lock, encodes whatever RADIO features that snapshot is
        missing (frames shared with the previous plan are reused), then integrates. All of the
        GPU work therefore happens in this one call, so it can run in a planning thread while the
        caller keeps observing and sending.
        """
        with self._lock:
            if len(self._frames) < self.history_len or self._state is None:
                raise RuntimeError("no observation yet -- call observe() first")
            ids = [fid for fid, _ in self._frames]
            pixels = np.stack([img for _, img in self._frames])
            state = self._state.copy()
        return self.predict_history(pixels, state, frame_ids=ids, flow_steps=flow_steps, **continuation)

    def predict_history(
        self,
        pixels: np.ndarray,
        raw_state: np.ndarray,
        *,
        frame_ids: Optional[Sequence[int]] = None,
        flow_steps: Optional[int] = None,
        prev_chunk: Optional[np.ndarray] = None,
        prev_offset: int = 0,
        delay_steps: int = 0,
    ) -> np.ndarray:
        """The stateless form: ``[L,V,H,W,3]`` uint8 RGB (views in :data:`VIEW_ORDER` order) plus
        the raw 28-D state -> ``[32, 14]``.

        ``frame_ids`` lets the caller say which frames are the same image as last time, so their
        RADIO features are reused; without ids every frame is encoded.

        ``prev_chunk`` (``[32, 14]`` de-normalised, as this method returned it) is the chunk being
        executed, ``prev_offset`` how many of its steps had elapsed at this observation, and
        ``delay_steps`` how many steps of the new chunk the old one will still execute before it
        arrives. They are read only by the ``rtc``/``bid``/``legato`` continuation methods; without
        ``prev_chunk`` RTC and Legato fall back to plain sampling and BID to forward contrast alone.
        """
        torch = self.torch
        pixels = np.asarray(pixels)
        if pixels.shape != (self.history_len, len(VIEW_ORDER), *IMAGE_HW, 3):
            raise ValueError(
                f"pixels must be [{self.history_len},{len(VIEW_ORDER)},{IMAGE_HW[0]},{IMAGE_HW[1]},3], "
                f"got {pixels.shape}"
            )
        state = np.asarray(raw_state, np.float32).reshape(-1)
        ids = list(frame_ids) if frame_ids is not None else [self._uncached_id() for _ in range(len(pixels))]
        history = list(zip(ids, pixels, strict=True))

        # Frames repeat at episode start (left-clamping) and across overlapping plans, so encode
        # each distinct id once.
        missing = [(fid, img) for fid, img in history if fid not in self._radio_cache]
        seen: Dict[int, np.ndarray] = {}
        for fid, img in missing:
            seen.setdefault(fid, img)
        if seen:
            batch = np.concatenate([seen[fid] for fid in seen])  # [K*V,H,W,3]
            unit = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(self.device)
            unit = unit.float().div_(255.0)
            local, global_ = self._encode_radio(unit)  # [K*V,P,C], [K*V,C]
            v = len(VIEW_ORDER)
            top = VIEW_ORDER.index("top")
            for i, fid in enumerate(seen):
                self._radio_cache[fid] = local[i * v : (i + 1) * v]
                self._top_global[fid] = global_[i * v + top]
        live = {fid for fid, _ in history}
        for fid in [k for k in self._radio_cache if k not in live]:
            del self._radio_cache[fid]
            self._top_global.pop(fid, None)

        views = torch.from_numpy(np.ascontiguousarray(pixels.transpose(0, 1, 4, 2, 3))).to(self.device)
        views = views.float().div_(255.0)
        mean = torch.from_numpy(IMAGENET_MEAN).to(self.device)
        std = torch.from_numpy(IMAGENET_STD).to(self.device)
        views = ((views - mean) / std).unsqueeze(0)  # [1,L,V,3,H,W]
        radio = torch.stack([self._radio_cache[fid] for fid, _ in history]).unsqueeze(0)  # [1,L,V,P,C]

        st = torch.from_numpy(self.normalize_state(state)).unsqueeze(0).to(self.device)
        steps = int(flow_steps or self.flow_steps)

        goal_kwargs: Dict[str, Any] = {}
        if self.uses_goals:
            if self.n_goals == 0:
                raise RuntimeError("goal policy without subgoals -- call set_goals() first")
            newest = history[-1][0]
            self._advance_cursor(self._top_global.get(newest), time.monotonic())
            g_local, g_global, g_mask = self._goal_window()
            goal_kwargs = dict(goal_radio_local=g_local, goal_radio_global=g_global, goal_valid_mask=g_mask)

        use_amp = self.trunk_dtype is not torch.float32 and self.device.type == "cuda"
        prefix_kwargs = dict(
            language_token_ids=self._lang_ids, language_lengths=self._lang_lens, radio_local=radio, **goal_kwargs
        )
        ref = None
        if prev_chunk is not None and self.continuation.method != "none":
            ref_np = reference_chunk(self.normalize_action(prev_chunk), prev_offset)
            ref = torch.from_numpy(ref_np).to(self.device).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=self.trunk_dtype, enabled=use_amp):
            cached = self.model.encode_prefix(views, st, None, **prefix_kwargs)
            x = self._sample(cached, views, st, prefix_kwargs, ref, int(prev_offset), int(delay_steps), steps)
        return self.denormalize_action(x[0].float().cpu().numpy())

    def _velocity(self, model: Any, cached: Dict[str, Any], omega: Optional[Any] = None) -> Any:
        """``(x [B,H,A], tau) -> v`` for one model and prefix cache, batch-broadcasting the cache.

        A Legato-trained network takes the schedule as an extra channel; everything else feeds it
        zeros there (no guidance), so a native checkpoint still samples plainly under other methods.
        """
        torch = self.torch
        native = int(model.action_in.in_features) == self.action_dim + 1

        def velocity(x: Any, tau: float) -> Any:
            b = int(x.shape[0])
            t = torch.full((b,), float(tau), device=self.device)
            if native:
                w = omega if omega is not None else torch.zeros(self.action_chunk_len, device=self.device)
                x = torch.cat([x, w.view(1, -1, 1).expand(b, -1, 1).to(x.dtype)], dim=-1)
            return model.denoise(expand_prefix_cache(cached, b), x, t)

        return velocity

    def _sample(
        self,
        cached: Dict[str, Any],
        views: Any,
        st: Any,
        prefix_kwargs: Dict[str, Any],
        ref: Optional[Any],
        offset: int,
        delay: int,
        steps: int,
    ) -> Any:
        """Integrate the flow with the configured continuation method -> normalised ``[1,H,A]``."""
        torch = self.torch
        cfg = self.continuation
        h, a = self.action_chunk_len, self.action_dim
        info: Dict[str, Any] = {"method": cfg.method, "offset": offset, "delay": delay, "guided": ref is not None}
        self.last_continuation = info

        def noise(n: int) -> Any:
            return torch.randn(n, h, a, device=self.device, generator=self._generator)

        if cfg.method == "bid":
            n = int(cfg.bid_samples)
            strong = euler(self._velocity(self.model, cached), noise(n), steps).float()
            weak = None
            if self.weak_model is not None:
                weak_cached = self.weak_model.encode_prefix(views, st, None, **prefix_kwargs)
                weak = euler(self._velocity(self.weak_model, weak_cached), noise(n), steps).float()
            back = None
            if ref is not None:
                back = torch.from_numpy(bid_backward_weights(h, offset, cfg.bid_decay)).to(self.device).float()
            picked = bid_select(strong, weak, ref, back, int(cfg.bid_mode_size))
            info.update(index=picked["index"], backward=picked["backward"], forward=picked["forward"])
            return picked["chunk"]

        x = noise(1)
        if ref is None or cfg.method == "none":
            return euler(self._velocity(self.model, cached), x, steps)
        if cfg.method == "rtc":
            w = torch.from_numpy(rtc_weights(h, delay, offset)).to(self.device).float()
            return rtc_sample(self._velocity(self.model, cached), x, ref, w, steps, float(cfg.rtc_beta))
        if cfg.method == "legato":
            w = torch.from_numpy(legato_schedule(h, delay, offset, cfg.legato_ramp)).to(self.device).float()
            info["native"] = self.legato_native
            return legato_sample(self._velocity(self.model, cached, w), x, ref, w, steps, native=self.legato_native)
        raise ValueError(f"unknown continuation method {cfg.method!r}")

    def observe_and_predict(self, frames_rgb: Dict[str, np.ndarray], raw_state: np.ndarray) -> np.ndarray:
        self.observe(frames_rgb, raw_state)
        return self.predict()

    def describe(self) -> str:
        goal = (
            f", goals {self.goal_sequence_len}-window advanced by {self.goal_advance}"
            if self.uses_goals
            else ", goal-free"
        )
        cont = f", continuation {self.continuation.describe()}"
        if self.continuation.method == "bid":
            cont += f" weak {self.weak_checkpoint.name if self.weak_checkpoint else 'none (positive-only contrast)'}"
        if self.continuation.method == "legato":
            cont += " native" if self.legato_native else " GUIDANCE-ONLY (checkpoint not Legato-trained)"
        return (
            f"box_folding {self.mode} policy: {self.checkpoint.name} @ step {self.step}, device {self.device}, "
            f"history {self.history_len} @ {1 / CONTROL_DT:.0f} fps, chunk {self.action_chunk_len}x"
            f"{self.action_dim}, flow steps {self.flow_steps}, RADIO {str(self.radio_dtype).split('.')[-1]}, "
            f"trunk {str(self.trunk_dtype).split('.')[-1]}, views {list(VIEW_ORDER)}{goal}{cont}"
        )


def _policy_worker(conn: Any, cfg: Dict[str, Any]) -> None:
    """Child-process entry point: build the policy, then answer plan requests until told to stop."""
    import traceback

    try:
        policy = BoxFoldingPolicy(**cfg)
        conn.send(
            (
                "ready",
                policy.describe(),
                {
                    "history_len": policy.history_len,
                    "action_chunk_len": policy.action_chunk_len,
                    "action_dim": policy.action_dim,
                    "state_dim": policy.state_dim,
                    "instruction": policy.instruction,
                    "uses_goals": policy.uses_goals,
                    "goal_sequence_len": policy.goal_sequence_len,
                },
            )
        )
    except BaseException:
        conn.send(("error", traceback.format_exc()))
        return

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        except KeyboardInterrupt:
            # A Ctrl-C in the operator's terminal is delivered to the whole process group, so it
            # lands here too, in the blocking recv. Leave quietly: the parent has its own handler
            # and a traceback from the child on every interrupt is pure noise on top of it.
            return
        if msg[0] == "stop":
            return
        if msg[0] == "goals":
            try:
                conn.send(("goals", policy.set_goals(msg[1])))
            except BaseException:
                conn.send(("error", traceback.format_exc()))
                return
            continue
        if msg[0] != "plan":
            conn.send(("error", f"unknown message {msg[0]!r}"))
            return
        _, t0, ids, pixels, state, prev_chunk, prev_offset, delay_steps = msg
        try:
            started = time.monotonic()
            chunk = policy.predict_history(
                pixels, state, frame_ids=ids, prev_chunk=prev_chunk, prev_offset=prev_offset, delay_steps=delay_steps
            )
            elapsed = time.monotonic() - started
            conn.send(("plan", t0, chunk, elapsed, policy.goal_cursor, dict(policy.last_continuation)))
        except BaseException:
            conn.send(("error", traceback.format_exc()))
            return


class PolicyProcess:
    """The policy in its own process, talking over a pipe: request in, action chunk out.

    Not premature isolation -- it is the fix for a measured 2.5x slowdown. A ``portal.Client``
    socket thread busy-spins whenever it has traffic (portal 3.7.3, client_socket.py: ``writing``
    is only cleared inside the ``if self.sendq`` branch, so a stale wakeup leaves the loop
    polling with a 0 s timeout). Two follower clients therefore burn ~65% CPU each and hold the
    GIL between the planner's CUDA launches: measured in the rollout, a plan takes ~210 ms with
    the RPCs stubbed out and ~550 ms with them live, at any command rate. In its own process the
    planner has a quiet GIL and keeps the ~210 ms.

    One request is in flight at a time, so the pipe never backs up; a request carries its own
    ``[L,V,H,W,3]`` uint8 history (~2.7 MB, ~3 ms to ship), which also keeps the child stateless
    apart from its RADIO cache.
    """

    def __init__(self, **policy_kwargs: Any) -> None:
        self._kwargs = policy_kwargs
        self._proc: Optional[Any] = None
        self._conn: Optional[Any] = None
        self._busy = False
        self.info: Dict[str, Any] = {}
        self.description = ""
        self.n_plans = 0
        self.n_goals = 0
        self.goal_cursor = 0
        self.last_request_t0 = -1e9
        self.last_continuation: Dict[str, Any] = {}
        """The child's report on how the last plan was sampled (method, offset, delay, ...)."""

    def start(self, timeout: float = 300.0) -> Dict[str, Any]:
        """Spawn the child and block until the model is loaded (tens of seconds)."""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # CUDA cannot be inherited across fork
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_policy_worker, args=(child_conn, self._kwargs), name="box-folding-policy", daemon=True
        )
        self._proc.start()
        child_conn.close()
        if not self._conn.poll(timeout):
            raise TimeoutError(f"policy process did not come up within {timeout:.0f}s")
        try:
            msg = self._conn.recv()
        except EOFError as e:
            # spawn re-imports the caller's __main__; without an `if __name__ == "__main__":`
            # guard the child re-runs the whole program instead of the worker and dies here.
            raise RuntimeError(
                "policy process exited during startup -- is the caller's entry point guarded by "
                'if __name__ == "__main__": ?'
            ) from e
        if msg[0] == "error":
            raise RuntimeError(f"policy process failed to start:\n{msg[1]}")
        _, self.description, self.info = msg
        return self.info

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def request(
        self,
        t0: float,
        pixels: np.ndarray,
        raw_state: np.ndarray,
        frame_ids: Sequence[int],
        *,
        prev_chunk: Optional[np.ndarray] = None,
        prev_offset: int = 0,
        delay_steps: int = 0,
    ) -> bool:
        """Ask for a chunk from this history. False if a request is already in flight.

        ``prev_chunk``/``prev_offset``/``delay_steps`` feed the continuation method; see
        :meth:`BoxFoldingPolicy.predict_history`.
        """
        if self._busy or self._conn is None:
            return False
        prev = None if prev_chunk is None else np.ascontiguousarray(prev_chunk)
        self._conn.send(
            (
                "plan",
                float(t0),
                list(frame_ids),
                np.ascontiguousarray(pixels),
                np.asarray(raw_state),
                prev,
                int(prev_offset),
                int(delay_steps),
            )
        )
        self._busy = True
        self.last_request_t0 = float(t0)
        return True

    def set_goals(self, goal_images_rgb: np.ndarray, timeout: float = 120.0) -> int:
        """Ship the subgoal sequence to the child and block until it has encoded it.

        Synchronous on purpose: it runs once, before the control loop starts, and no plan can be
        in flight yet -- so it may use the same pipe without racing :meth:`poll`.
        """
        if self._conn is None:
            raise RuntimeError("policy process is not running")
        if self._busy:
            raise RuntimeError("cannot set goals while a plan is in flight")
        self._conn.send(("goals", np.ascontiguousarray(goal_images_rgb)))
        if not self._conn.poll(timeout):
            raise TimeoutError(f"policy process did not accept the subgoals within {timeout:.0f}s")
        msg = self._conn.recv()
        if msg[0] == "error":
            raise RuntimeError(f"policy process failed to encode the subgoals:\n{msg[1]}")
        self.n_goals = int(msg[1])
        return self.n_goals

    def poll(self) -> Optional[Tuple[float, np.ndarray, float, int]]:
        """``(t0, chunk, latency_s, goal_cursor)`` if a plan is ready, else None.

        Raises if the child died. ``goal_cursor`` is 0 for a goal-free policy.
        """
        if self._conn is None:
            return None
        if not self._conn.poll():
            if self._busy and not self.alive:
                raise RuntimeError("policy process died while planning")
            return None
        msg = self._conn.recv()
        self._busy = False
        if msg[0] == "error":
            raise RuntimeError(f"policy process failed:\n{msg[1]}")
        _, t0, chunk, latency, cursor, self.last_continuation = msg
        self.n_plans += 1
        self.goal_cursor = int(cursor)
        return float(t0), chunk, float(latency), int(cursor)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None


def build_state(
    joint_pos_left: np.ndarray,
    joint_pos_right: np.ndarray,
    eef_pos_left: np.ndarray,
    eef_quat_left: np.ndarray,
    eef_pos_right: np.ndarray,
    eef_quat_right: np.ndarray,
) -> np.ndarray:
    """The 28-D proprioception vector, in the one order the policy was trained on.

    Same concatenation as ``build_box_folding_policy_hdf5.py``: both 7-D joint vectors
    (6 arm joints + normalised gripper) first, then the two EEF poses (pos, quat wxyz).
    """
    parts: Sequence[np.ndarray] = (
        joint_pos_left,
        joint_pos_right,
        eef_pos_left,
        eef_quat_left,
        eef_pos_right,
        eef_quat_right,
    )
    state = np.concatenate([np.asarray(p, np.float32).reshape(-1) for p in parts])
    if state.shape[0] != 28:
        raise ValueError(f"state is {state.shape[0]}-D, expected 28 (7+7+3+4+3+4)")
    return state


def split_action(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``[14]`` leader command -> ``(left[7], right[7])``, each 6 arm joints + gripper openness."""
    a = np.asarray(action, np.float32).reshape(-1)
    if a.shape[0] != 14:
        raise ValueError(f"action must be 14-D, got {a.shape[0]}")
    return a[:7].copy(), a[7:].copy()
