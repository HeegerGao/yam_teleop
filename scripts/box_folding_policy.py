"""Runtime for the ``box_folding`` end-to-end flow policy (trained on ``yam_data/box_folding``).

Shared by ``scripts/box_folding_policy_rollout.py`` (closed loop on the real arms) and
``scripts/box_folding_policy_replay.py`` (offline check against a recorded episode).

The policy bundle -- checkpoint, ``norm_stats.json``, the ``planning`` package and the frozen
DecisionNCE instruction embedding -- is the HF folder ``box_folding_policy/e2e`` of
``ChongkaiGao/planning``, downloaded to ``--policy-root`` (default ``~/box_folding_policy/e2e``).

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
The policy is goal-free: no planner, no goal image, nothing but cameras and proprioception.

RADIO (``c-radio_v3-h``) is encoded live inside :meth:`BoxFoldingPolicy.predict`, on demand: an
observation only buffers uint8 frames, and a plan encodes the (at most 12) images its history
snapshot has not already got features for. Encoding on every 10 Hz tick instead would spend ~34 ms
of GPU per tick on frames no plan reads, in contention with the planner. One plan costs ~110 ms of
RADIO plus a
prefix + 10 Euler steps (~130 ms). Both numbers matter: they are why the rollout
plans in a worker thread instead of blocking its send loop. 10 flow steps rather than the 20 the
bundled example uses: on the replay check 5, 10 and 20 steps score the same (0.021-0.023 rad mean
error against the recorded leader commands), and halving them halves the GPU the planner has to
win back from the 10 Hz RADIO encodes it shares the device with.

Weights: RADIO comes from Torch Hub, so ``TORCH_HOME`` must hold ``hub/NVlabs_RADIO_main`` and
``hub/checkpoints/c-radio_v3-h_half.pth.tar`` (~1.7 GB, downloaded on first use).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

VIEW_ORDER: Tuple[str, ...] = ("left_wrist", "right_wrist", "top")
"""Alphabetical -- the order the view embedding was trained with. Never reorder."""
IMAGE_HW: Tuple[int, int] = (240, 320)
CONTROL_DT = 0.1
"""One action / one history frame per 100 ms: the canonical 10 fps timeline."""
LANGUAGE_PAD_TOKEN = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
DEFAULT_POLICY_ROOT = "~/box_folding_policy/e2e"
DEFAULT_TORCH_HOME = "~/torch_home"


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


class BoxFoldingPolicy:
    """The flow policy plus its observation history, RADIO cache and (de)normalisation.

    Call :meth:`observe` once per 10 fps tick and :meth:`predict` whenever a fresh action chunk
    is wanted; ``predict`` is safe to call from a worker thread while the caller keeps feeding
    observations, and it works off a snapshot of the history taken under the lock.
    """

    def __init__(
        self,
        policy_root: str | Path = DEFAULT_POLICY_ROOT,
        *,
        checkpoint: Optional[str | Path] = None,
        device: str = "cuda:0",
        flow_steps: int = 10,
        radio_dtype: str = "float32",
        torch_home: Optional[str | Path] = DEFAULT_TORCH_HOME,
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(policy_root).expanduser().resolve()
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
        # float16/bfloat16 for the frozen tower measured identical on the replay check (0.0225 rad)
        # and identical in GPU time on an RTX 5090, so fp32 stays the default; the knob is here
        # because training consumed fp16 features and a smaller GPU may still prefer half.
        self.radio_dtype = getattr(torch, str(radio_dtype))
        if self.radio_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"radio_dtype must be float32/float16/bfloat16, got {radio_dtype}")

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

        ckpt = Path(checkpoint).expanduser() if checkpoint else self._latest_checkpoint()
        self.model, self.cfg, self.step = self._build(ckpt)
        self.history_len = int(self.model.history_len)
        self.action_chunk_len = int(self.model.action_chunk_len)
        self.action_dim = int(self.model.action_dim)
        self.state_dim = int(self.stats["state_dim"])
        self.checkpoint = ckpt

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
        self._next_id = 0
        self._scratch_id = -1
        self._state: Optional[np.ndarray] = None
        self.n_observations = 0

    def _uncached_id(self) -> int:
        """A fresh negative id: never reused, so the frame is encoded and then evicted."""
        self._scratch_id -= 1
        return self._scratch_id

    # ------------------------------------------------------------------ construction
    def _latest_checkpoint(self) -> Path:
        ckpts = sorted((self.root / "checkpoints").glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints in {self.root / 'checkpoints'}")
        return ckpts[-1]

    def _build(self, checkpoint: Path) -> Tuple[Any, Dict[str, Any], Optional[int]]:
        """Mirror of the bundle's ``inference_example.build_policy``, with absolute asset paths.

        Everything structural comes from the checkpoint's own training config, so a newer
        checkpoint of the same run loads without touching this file. ``load_radio_backbone=True``
        because the robot has no precomputed RADIO cache, and ``resnet_pretrained=False`` because
        the trained ResNet34 weights are in the checkpoint.
        """
        from planning.model.flow_matching_goal_resnet_radio_dit_mv import (
            FlowMatchingGoalResNetRadioDiTMVPolicy,
        )

        blob = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, m, d = blob["cfg"], blob["cfg"]["model"], blob["cfg"]["data"]
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
            load_radio_backbone=True,
            use_goal_conditioning=False,
            goal_sequence_len=int(d["goal_sequence_len"]),
            language_encoder_type="decisionnce",
            decisionnce_cache=str(self.root / "box_folding_decisionnce_t.npz"),
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
        return model.to(self.device).eval(), cfg, blob.get("step")

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

    # ------------------------------------------------------------------ observation
    def reset(self) -> None:
        """Forget the history; the next observation starts a fresh (left-clamped) episode."""
        with self._lock:
            self._frames.clear()
            self._state = None
            self.n_observations = 0
        self._radio_cache.clear()

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

    def _encode_radio(self, unit_images: Any) -> Any:
        """Frozen RADIO patch tokens: ``[N,3,H,W]`` in [0,1] -> ``[N,P,C]`` float32."""
        from planning.model.radio_encode_utils import radio_encode_unit_nchw_local_global

        torch = self.torch
        use_amp = self.radio_dtype is not torch.float32 and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.radio_dtype, enabled=use_amp):
            local, _ = radio_encode_unit_nchw_local_global(
                self.model.radio, unit_images, microbatch=int(unit_images.shape[0])
            )
        return local.float().clone()

    @property
    def ready(self) -> bool:
        with self._lock:
            return len(self._frames) == self.history_len and self._state is not None

    def last_state(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._state is None else self._state.copy()

    # ------------------------------------------------------------------ inference
    def predict(self, *, flow_steps: Optional[int] = None) -> np.ndarray:
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
        return self.predict_history(pixels, state, frame_ids=ids, flow_steps=flow_steps)

    def predict_history(
        self,
        pixels: np.ndarray,
        raw_state: np.ndarray,
        *,
        frame_ids: Optional[Sequence[int]] = None,
        flow_steps: Optional[int] = None,
    ) -> np.ndarray:
        """The stateless form: ``[L,V,H,W,3]`` uint8 RGB (views in :data:`VIEW_ORDER` order) plus
        the raw 28-D state -> ``[32, 14]``.

        ``frame_ids`` lets the caller say which frames are the same image as last time, so their
        RADIO features are reused; without ids every frame is encoded.
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
            local = self._encode_radio(unit)  # [K*V,P,C]
            v = len(VIEW_ORDER)
            for i, fid in enumerate(seen):
                self._radio_cache[fid] = local[i * v : (i + 1) * v]
        live = {fid for fid, _ in history}
        for fid in [k for k in self._radio_cache if k not in live]:
            del self._radio_cache[fid]

        views = torch.from_numpy(np.ascontiguousarray(pixels.transpose(0, 1, 4, 2, 3))).to(self.device)
        views = views.float().div_(255.0)
        mean = torch.from_numpy(IMAGENET_MEAN).to(self.device)
        std = torch.from_numpy(IMAGENET_STD).to(self.device)
        views = ((views - mean) / std).unsqueeze(0)  # [1,L,V,3,H,W]
        radio = torch.stack([self._radio_cache[fid] for fid, _ in history]).unsqueeze(0)  # [1,L,V,P,C]

        st = torch.from_numpy(self.normalize_state(state)).unsqueeze(0).to(self.device)
        steps = int(flow_steps or self.flow_steps)
        with torch.no_grad():
            cached = self.model.encode_prefix(
                views,
                st,
                None,
                language_token_ids=self._lang_ids,
                language_lengths=self._lang_lens,
                radio_local=radio,
            )
            x = torch.randn(1, self.action_chunk_len, self.action_dim, device=self.device, generator=self._generator)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((1,), i * dt, device=self.device)
                x = x + dt * self.model.denoise(cached, x, t)
        return self.denormalize_action(x[0].float().cpu().numpy())

    def observe_and_predict(self, frames_rgb: Dict[str, np.ndarray], raw_state: np.ndarray) -> np.ndarray:
        self.observe(frames_rgb, raw_state)
        return self.predict()

    def describe(self) -> str:
        return (
            f"box_folding policy: {self.checkpoint.name} @ step {self.step}, device {self.device}, "
            f"history {self.history_len} @ {1 / CONTROL_DT:.0f} fps, chunk {self.action_chunk_len}x"
            f"{self.action_dim}, flow steps {self.flow_steps}, RADIO {str(self.radio_dtype).split('.')[-1]}, "
            f"views {list(VIEW_ORDER)}"
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
        if msg[0] == "stop":
            return
        if msg[0] != "plan":
            conn.send(("error", f"unknown message {msg[0]!r}"))
            return
        _, t0, ids, pixels, state = msg
        try:
            started = time.monotonic()
            chunk = policy.predict_history(pixels, state, frame_ids=ids)
            conn.send(("plan", t0, chunk, time.monotonic() - started))
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
        self.last_request_t0 = -1e9

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

    def request(self, t0: float, pixels: np.ndarray, raw_state: np.ndarray, frame_ids: Sequence[int]) -> bool:
        """Ask for a chunk from this history. False if a request is already in flight."""
        if self._busy or self._conn is None:
            return False
        self._conn.send(("plan", float(t0), list(frame_ids), np.ascontiguousarray(pixels), np.asarray(raw_state)))
        self._busy = True
        self.last_request_t0 = float(t0)
        return True

    def poll(self) -> Optional[Tuple[float, np.ndarray, float]]:
        """``(t0, chunk, latency_s)`` if a plan is ready, else None. Raises if the child died."""
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
        _, t0, chunk, latency = msg
        self.n_plans += 1
        return float(t0), chunk, float(latency)

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
