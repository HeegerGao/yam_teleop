"""The upper level of the hierarchical box_folding policy: Wan2.2 generates the subgoal images.

Wan2.2-TI2V-5B plus the bundle's LoRA (``wan_lora/lora_step_0028000_final.safetensors``, a DP-K15
subgoal-sequence generator) turns the *first* top-camera frame of an episode into a 61-frame
rollout; frames 4, 8, ... 60 are the 15 subgoals the goal-conditioned policy consumes six at a
time. The LoRA was only ever trained conditioned on a *first* frame, which is why the classic
mode is one generation per episode.

:class:`WanSubgoalProcess` also serves the rollout's **online replanning**: with
``keep_alive=True`` the child holds the loaded pipeline and generates again on demand, each time
conditioned on the top view as it is *then*. A mid-episode conditioning frame is outside what the
LoRA was trained on, so this is an experiment rather than a strictly better plan -- but it is the
only way the upper level can react to what the arms actually did.

Run it in its own process (:class:`WanSubgoalProcess`), not in the caller. Two reasons:

* **VRAM.** The pipeline needs ~20 GB; the policy needs ~3 GB more. One-shot, the child exits
  after generating and gives all of it back before the control loop starts. Kept alive for
  replanning, both stay resident -- ~25 GB of a 32 GB card, which fits, with the generation
  sharing the GPU with whatever plan overlaps it.
* **Wall clock.** Loading Wan takes ~11 s with the weights in page cache (much longer the first
  time they come off disk) and loading the policy ~30 s. Started together they overlap, so the
  rollout waits for the longer one instead of their sum. It is also why replanning keeps the
  child: a reload plus a 20 GB alloc/free per generation, on top of the ~7.7 s the generation
  itself takes, would not fit the window replanning runs on.

CLI, for generating a set offline (e.g. to replay a rollout with fixed subgoals):

    python scripts/box_folding_wan_subgoals.py --episode ~/yam_data/box_folding/episode_0000
    python scripts/box_folding_wan_subgoals.py --image top.png --out-dir /tmp/subgoals

Weights: ``--models`` must hold ``Wan-AI/Wan2.2-TI2V-5B`` (~32 GB with the umt5-xxl encoder) and
``Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl`` (the tokenizer). See the bundle's ``INSTALL.md``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from box_folding_policy import DEFAULT_GOAL_POLICY_ROOT

DEFAULT_WAN_MODELS = "~/box_folding_policy/models"
WAN_INPUT_WH = (512, 288)
"""(W, H) the LoRA generates at. The top frame is squeezed into it, aspect ratio not preserved."""
WAN_NUM_FRAMES = 61
SUBGOAL_INDICES = tuple(range(4, WAN_NUM_FRAMES, 4))  # 4, 8, ... 60 -- the 15 DP-K15 stages
LORA_GLOB = "wan_lora/*.safetensors"


def load_pipeline(
    *,
    policy_root: str | Path = DEFAULT_GOAL_POLICY_ROOT,
    models: str | Path = DEFAULT_WAN_MODELS,
    lora_alpha: float = 1.0,
) -> Tuple[Any, str]:
    """Load Wan2.2-TI2V-5B + the bundle's subgoal LoRA: ``(pipeline, prompt)``. ~20 GB, ~60 s.

    Split out of :func:`generate_subgoals` so a server process can hold the pipeline across
    several generations. Online replanning wants a fresh set every ~20 s and the generation
    itself costs ~7 s, so paying this load again each time is not an option.
    """
    import os

    root = Path(policy_root).expanduser().resolve()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(Path(models).expanduser().resolve()))
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    prompt = str(json.loads((root / "norm_stats.json").read_text())["language_instruction"])
    loras = sorted(root.glob(LORA_GLOB))
    if not loras:
        raise FileNotFoundError(f"no subgoal LoRA at {root / LORA_GLOB}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        # Without this, from_pretrained silently redirects the T5 encoder and the VAE to
        # DiffSynth's own converted-safetensors repo and re-downloads 14 GB that is already here.
        redirect_common_files=False,
    )
    pipe.load_lora(pipe.dit, str(loras[-1]), alpha=float(lora_alpha))
    return pipe, prompt


def generate_with(
    pipe: Any,
    prompt: str,
    top_rgb: np.ndarray,
    *,
    steps: int = 40,
    seed: int = 0,
    cfg_scale: float = 1.0,
    tiled: bool = True,
) -> np.ndarray:
    """One conditioning frame -> ``[15, 288, 512, 3]`` uint8 RGB subgoals, on a loaded pipeline.

    Args:
        pipe: the pipeline from :func:`load_pipeline`; reusable for as many generations as wanted.
        top_rgb: the top-camera frame to condition on, RGB uint8, any resolution.
        tiled: tile the VAE encode/decode. Cheaper in VRAM, dearer in time -- see
            scripts/box_folding_wan_bench.py, which is what this argument exists for.
    """
    from PIL import Image

    w, h = WAN_INPUT_WH
    frame = cv2.resize(np.asarray(top_rgb), (w, h), interpolation=cv2.INTER_AREA)
    video = pipe(
        prompt=prompt,
        input_image=Image.fromarray(frame),
        height=h,
        width=w,
        num_frames=WAN_NUM_FRAMES,
        cfg_scale=float(cfg_scale),
        num_inference_steps=int(steps),
        seed=int(seed),
        tiled=bool(tiled),
        # The raw VAE tensor, not a PIL list: one clamp is cheaper than 61 PIL round-trips and
        # only 15 of the frames are ever wanted.
        output_type="floatpoint",
    )
    frames = _to_uint8(video)
    if len(frames) < WAN_NUM_FRAMES:
        raise RuntimeError(f"Wan returned {len(frames)} frames, expected {WAN_NUM_FRAMES}")
    return np.stack([frames[i] for i in SUBGOAL_INDICES])


def generate_subgoals(
    top_rgb: np.ndarray,
    *,
    policy_root: str | Path = DEFAULT_GOAL_POLICY_ROOT,
    models: str | Path = DEFAULT_WAN_MODELS,
    steps: int = 40,
    seed: int = 0,
    cfg_scale: float = 1.0,
    lora_alpha: float = 1.0,
) -> np.ndarray:
    """One top frame -> ``[15, 288, 512, 3]`` uint8 RGB subgoals. Loads ~20 GB; call once.

    The load-and-generate convenience wrapper. To generate more than once, call
    :func:`load_pipeline` yourself and then :func:`generate_with` per conditioning frame.
    """
    pipe, prompt = load_pipeline(policy_root=policy_root, models=models, lora_alpha=lora_alpha)
    return generate_with(pipe, prompt, top_rgb, steps=steps, seed=seed, cfg_scale=cfg_scale)


def _to_uint8(video: Any) -> List[np.ndarray]:
    """``output_type='floatpoint'`` gives ``[1,3,T,H,W]`` in [-1,1]; DiffSynth's own quantiser
    uses the same min/max, so this is the identical mapping without the PIL detour."""
    import torch

    if isinstance(video, torch.Tensor):
        x = video.detach().float().cpu()
        x = x[0] if x.dim() == 5 else x
        if x.shape[0] == 3:
            x = x.permute(1, 2, 3, 0)  # [C,T,H,W] -> [T,H,W,C]
        arr = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
        return [arr[t] for t in range(arr.shape[0])]
    return [np.asarray(f.convert("RGB") if hasattr(f, "convert") else f, np.uint8) for f in video]


def save_subgoals(subgoals: np.ndarray, out_dir: Path) -> List[Path]:
    """Write ``subgoal_NN.png`` (RGB in, BGR on disk) and return the paths, in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, img in enumerate(np.asarray(subgoals)):
        path = out_dir / f"subgoal_{k:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        written.append(path)
    return written


def load_subgoals(directory: str | Path) -> np.ndarray:
    """Read a ``subgoal_NN.png`` set back as ``[K,H,W,3]`` uint8 RGB, in filename order."""
    d = Path(directory).expanduser().resolve()
    paths = sorted(d.glob("subgoal_*.png"))
    if not paths:
        raise FileNotFoundError(f"no subgoal_*.png in {d}")
    return np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])


def _worker(conn: Any, cfg: Dict[str, Any], keep_alive: bool) -> None:
    """Child entry point: load once, then answer generation requests until told to stop.

    One-shot (``keep_alive=False``) returns after the first hand-over, which frees its ~20 GB of
    VRAM before the control loop starts. Kept alive, it holds the pipeline and waits for the next
    conditioning frame -- the load, not the generation, is what makes replanning per-episode.
    """
    import traceback

    loaded: Optional[Tuple[Any, str]] = None
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:  # the parent closed the pipe without a "stop"
                return
            if msg[0] == "stop":
                return
            started = time.monotonic()
            if loaded is None:
                loaded = load_pipeline(
                    policy_root=cfg.get("policy_root", DEFAULT_GOAL_POLICY_ROOT),
                    models=cfg.get("models", DEFAULT_WAN_MODELS),
                    lora_alpha=cfg.get("lora_alpha", 1.0),
                )
            pipe, prompt = loaded
            subgoals = generate_with(
                pipe,
                prompt,
                msg[1],
                steps=cfg.get("steps", 40),
                seed=cfg.get("seed", 0),
                cfg_scale=cfg.get("cfg_scale", 1.0),
            )
            try:
                conn.send(("subgoals", subgoals, time.monotonic() - started))
            except (BrokenPipeError, OSError):
                # The run ended while this generation was still going: the parent hung up and
                # nobody wants the result. Leave quietly rather than dumping a traceback over
                # the teardown log.
                return
            if not keep_alive:
                return
    except KeyboardInterrupt:
        # Ctrl-C reaches the whole process group; the parent reports it, this child just leaves.
        return
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass


class WanSubgoalProcess:
    """Generate the subgoals in a child process.

    One-shot by default: the child exits as soon as the first set is handed over, giving its
    ~20 GB of VRAM back before the control loop starts. ``keep_alive=True`` keeps the pipeline
    loaded so every later :meth:`start` costs only the ~7 s generation -- what the rollout's
    online replanning needs, at the price of holding that VRAM for the whole run.

    Start it before the policy process so the two model loads overlap, then collect the result
    with :meth:`result` (blocking, for the first set) or :meth:`poll` (non-blocking, for a set
    asked for from inside a control loop).
    """

    def __init__(self, *, keep_alive: bool = False, **cfg: Any) -> None:
        self._cfg = cfg
        self._keep_alive = bool(keep_alive)
        self._proc: Optional[Any] = None
        self._conn: Optional[Any] = None
        self._busy = False
        self.seconds = 0.0
        """Wall-clock of the most recent generation; the first one includes the model load."""
        self.n_generations = 0

    @property
    def busy(self) -> bool:
        """A generation has been asked for and not collected yet."""
        return self._busy

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self, top_rgb: np.ndarray) -> None:
        """Ask for a subgoal set conditioned on this frame, spawning the child if needed."""
        import multiprocessing as mp

        if self._busy:
            raise RuntimeError("a generation is already in flight")
        if self._proc is None:
            ctx = mp.get_context("spawn")  # CUDA cannot be inherited across fork
            self._conn, child = ctx.Pipe()
            self._proc = ctx.Process(
                target=_worker, args=(child, self._cfg, self._keep_alive), name="wan-subgoals", daemon=True
            )
            self._proc.start()
            child.close()
        assert self._conn is not None
        self._conn.send(("generate", np.ascontiguousarray(top_rgb)))
        self._busy = True

    def result(self, timeout: float = 900.0) -> np.ndarray:
        """Block for the subgoals. Raises whatever the child raised."""
        if self._conn is None or not self._busy:
            raise RuntimeError("start() was never called")
        if not self._conn.poll(timeout):
            raise TimeoutError(f"Wan did not produce subgoals within {timeout:.0f}s")
        return self._take()

    def poll(self) -> Optional[np.ndarray]:
        """The subgoals if the child has finished, else None. Never blocks the caller's loop."""
        if self._conn is None or not self._busy:
            return None
        if not self._conn.poll():
            if not self.alive:
                raise RuntimeError("the Wan subgoal process died while generating")
            return None
        return self._take()

    def _take(self) -> np.ndarray:
        """Collect one finished generation, and reap the child unless it is being kept alive."""
        assert self._conn is not None
        msg = self._conn.recv()
        self._busy = False
        if msg[0] == "error":
            self.close()
            raise RuntimeError(f"Wan subgoal generation failed:\n{msg[1]}")
        _, subgoals, self.seconds = msg
        self.n_generations += 1
        if not self._keep_alive:
            self.close()
        return subgoals

    def close(self) -> None:
        busy = self._busy
        if self._conn is not None:
            try:
                self._conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        self._busy = False
        if self._proc is not None:
            # An idle child leaves as soon as it reads the "stop". One that is mid-generation is
            # still seconds from reading anything, and its result is worthless now, so it gets a
            # moment and then a signal -- teardown does not wait on work nobody wants.
            self._proc.join(timeout=1.0 if busy else 10.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5.0)
            self._proc = None


@dataclass
class Args:
    episode: Optional[str] = None
    """Recorded episode whose FIRST top.mp4 frame conditions the generation."""
    image: Optional[str] = None
    """A top-view image file instead of --episode. Exactly one of the two is required."""
    out_dir: Optional[str] = None
    """Where the PNGs land. Default: <episode>/subgoals, or ./subgoals for --image."""
    policy_root: str = DEFAULT_GOAL_POLICY_ROOT
    """Bundle holding wan_lora/*.safetensors and norm_stats.json (for the instruction)."""
    models: str = DEFAULT_WAN_MODELS
    """DiffSynth model base path holding Wan-AI/Wan2.2-TI2V-5B."""
    steps: int = 40
    """Wan denoising steps. 40 is what the LoRA was evaluated at; fewer is faster and blurrier."""
    seed: int = 0
    cfg_scale: float = 1.0
    """1.0 = no classifier-free guidance, which also halves the DiT work per step."""
    contact_sheet: bool = True
    """Also write subgoals.png, the 15 frames as one 5x3 grid, for a glance at the plan."""


def _first_top_frame(args: Args) -> np.ndarray:
    if bool(args.episode) == bool(args.image):
        raise SystemExit("[error] pass exactly one of --episode or --image")
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


def main(args: Args) -> None:
    top = _first_top_frame(args)
    out = Path(
        args.out_dir or (Path(args.episode).expanduser() / "subgoals" if args.episode else "subgoals")
    ).expanduser()
    started = time.monotonic()
    subgoals = generate_subgoals(
        top,
        policy_root=args.policy_root,
        models=args.models,
        steps=args.steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )
    paths = save_subgoals(subgoals, out)
    print(f"[wan] {len(paths)} subgoals in {time.monotonic() - started:.1f}s -> {out}")
    if args.contact_sheet:
        rows = [np.hstack(list(subgoals[i * 5 : (i + 1) * 5])) for i in range(3)]
        sheet = cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / "subgoals.png"), sheet)
        print(f"[wan] contact sheet -> {out / 'subgoals.png'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
