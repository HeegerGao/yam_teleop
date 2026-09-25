"""π₀.₅ policy as a local inference server, plus the client that drives it.

The checkpoints come from HF ``lzy001Yuki/SteeringPolicyReal`` (versioned releases under
``~/steering_policy_real/releases/<revision>/<task>/``). They need LeRobot at the commit they were
trained with (71a11efe, transformers 5.5.4), which does not fit in the i2rt venv (transformers 5.16,
diffsynth for box_folding). So the model lives in its own venv, ``~/steering_policy_real/pi05_env``
(Python 3.12), and runs as this server; the robot side (i2rt venv, Python 3.11) talks to it over a
``multiprocessing.connection`` socket on localhost.

That split is also what the robot side wants anyway: portal's client socket threads busy-spin and
steal the GIL from GPU work in the same process (see box_folding_policy.PolicyProcess).

Wire format: pickled tuples of plain Python types; numpy arrays travel as ``(dtype, shape, bytes)``
so nothing depends on both interpreters sharing a numpy pickle layout.

    request  ("plan", {"state": [14] float32, "images": {view: [H,W,3] uint8 RGB}})
    reply    ("ok", {"chunk": [50,14] float32 raw joint targets, "infer_s": float}) | ("error", str)

The default path is exactly the one PI05_STEERING.md documents: raw state and raw-resolution RGB
frames -> checkpoint preprocessor -> predict_action_chunk -> checkpoint postprocessor.  For
the three tasks, ``--model`` selects plain pi, D5 encoder/decoder, DrVLA or COAST hooks from
``STEERING.md``. The eval launcher resolves task-specific assets and rejects unavailable releases.
The socket contract stays the same, so robot control and safety do not fork.

Normally :class:`PolicyClient` spawns this script itself. To run it by hand:

    ~/steering_policy_real/pi05_env/bin/python scripts/steering_policy_server.py \
        --checkpoint ~/steering_policy_real/cloth/020000/pretrained_model --prompt cloth
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from steering_policy_models import Model

DEFAULT_ROOT = Path("~/steering_policy_real").expanduser()
DEFAULT_PYTHON = DEFAULT_ROOT / "pi05_env" / "bin" / "python"
DEFAULT_STEER_CHECKPOINT = DEFAULT_ROOT / "cup_pingpong" / "finetuned_ckpt" / "pretrained_model"
DEFAULT_STEER_DEPLOYMENT_ROOT = DEFAULT_ROOT / "deployment"
DEFAULT_STEER_SAE_ROOT = DEFAULT_ROOT / "cup_pingpong" / "sae"
DEFAULT_STEER_METHOD_ROOT = DEFAULT_ROOT / "cup_pingpong" / "method_v1_3"
VIEWS: Tuple[str, ...] = ("top", "left_wrist", "right_wrist")
"""Camera roles, named as in the recorder and as ``observation.images.<view>`` in the checkpoint."""
_AUTHKEY = b"steering-policy"
DEMO_DIRS: Dict[str, str] = {"push_cube": "push_cube_shovel_correct"}
"""Task name (checkpoint dir and prompt) -> the ~/yam_data folder its training demos came from, where
the two differ. push_cube was trained on push_cube_shovel_correct: 40 episodes, 15,439 frames, exactly
PI05_STEERING.md's count; push_cube_shovel_wrong (40 episodes, 18,291 frames) was not used."""


def demo_dir(task: str) -> str:
    """The ~/yam_data folder holding ``task``'s training demos."""
    return DEMO_DIRS.get(task, task)


def default_checkpoint(task: str, root: Path = DEFAULT_ROOT) -> Path:
    """The highest-step ``<root>/<task>/<step>/pretrained_model``."""
    steps = sorted(
        (p for p in (root / task).glob("*/pretrained_model") if p.parent.name.isdigit()),
        key=lambda p: int(p.parent.name),
    )
    if not steps:
        raise FileNotFoundError(f"no checkpoint under {root / task}/<step>/pretrained_model")
    return steps[-1]


def _pack(a: np.ndarray) -> Tuple[str, Tuple[int, ...], bytes]:
    a = np.ascontiguousarray(a)
    return (str(a.dtype), tuple(a.shape), a.tobytes())


def _unpack(t: Tuple[str, Tuple[int, ...], bytes]) -> np.ndarray:
    dtype, shape, data = t
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)


# ---------------------------------------------------------------------------
# Server (pi05_env)
# ---------------------------------------------------------------------------


@dataclass
class ServerArgs:
    checkpoint: str
    """The ``pretrained_model`` directory."""
    prompt: str
    """Task string from the training dataset (for example, ``cloth`` or ``cup_pingpong``)."""
    port: int = 18765
    device: str = "cuda"
    num_inference_steps: Optional[int] = None
    """Flow-matching steps. Default: the checkpoint's own (10)."""
    seed: Optional[int] = None
    """Seed the flow noise before every plan. None = fresh noise each plan."""
    steer: bool = False
    """Enable the released cup_pingpong D5 + maturity-gated SAE steering controller."""
    steer_deployment_root: str = str(DEFAULT_STEER_DEPLOYMENT_ROOT)
    steer_sae_root: str = str(DEFAULT_STEER_SAE_ROOT)
    steer_method_root: str = str(DEFAULT_STEER_METHOD_ROOT)
    model: Model = "pi"
    selection: Optional[str] = None
    coast_beta: float = 1.0
    """COAST conceptor strength in [0, 1]; 1 is the published baseline, 0 disables its effect."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.coast_beta <= 1.0:
            raise ValueError("--coast-beta must be finite and in [0, 1]")


@contextmanager
def _skip_random_init(torch: Any) -> Iterator[None]:
    """Build the model without filling its weights with random numbers first.

    ``PI05Policy.from_pretrained`` constructs the full PaliGemma + Gemma expert (~3 B params, float32,
    on the CPU) and every ``nn.Linear`` / ``nn.Embedding`` / transformers ``post_init`` fills its
    weights with ``uniform_`` / ``normal_`` -- only for the checkpoint to overwrite all of them a
    moment later. Profiled: 45 of the 50 s load were those two calls; reading the 9.4 GB safetensors
    is ~2 s. The load reports every key present, and parameters, buffers and plan output were checked
    bit-identical with and without this.
    """
    saved = (torch.Tensor.uniform_, torch.Tensor.normal_)

    def keep(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self

    torch.Tensor.uniform_ = keep
    torch.Tensor.normal_ = keep
    try:
        yield
    finally:
        torch.Tensor.uniform_, torch.Tensor.normal_ = saved


@contextmanager
def _accept_min_range_processor_config() -> Iterator[None]:
    """Load the newer released processor JSON with the pinned LeRobot runtime.

    The steering checkpoint adds ``min_range=0.001`` to its normalizer configs, while the training
    runtime pinned for these models predates that constructor argument.  The released steering
    inference also ignores ``min_range``.  Drop only that unsupported keyword while the two
    processors are instantiated, then restore their constructors immediately.
    """
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    normalizer_init = NormalizerProcessorStep.__init__
    unnormalizer_init = UnnormalizerProcessorStep.__init__

    def init_normalizer(self: Any, *args: Any, min_range: Optional[float] = None, **kwargs: Any) -> None:
        del min_range
        normalizer_init(self, *args, **kwargs)

    def init_unnormalizer(self: Any, *args: Any, min_range: Optional[float] = None, **kwargs: Any) -> None:
        del min_range
        unnormalizer_init(self, *args, **kwargs)

    NormalizerProcessorStep.__init__ = init_normalizer
    UnnormalizerProcessorStep.__init__ = init_unnormalizer
    try:
        yield
    finally:
        NormalizerProcessorStep.__init__ = normalizer_init
        UnnormalizerProcessorStep.__init__ = unnormalizer_init


class _Pi05:
    def __init__(self, args: ServerArgs) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Policy
        from lerobot.policies.utils import prepare_observation_for_inference

        self._torch = torch
        self._prepare = prepare_observation_for_inference
        self.args = args
        ckpt = str(Path(args.checkpoint).expanduser())
        t = time.monotonic()
        with _skip_random_init(torch):
            self.policy = PI05Policy.from_pretrained(ckpt, local_files_only=True)
        if args.num_inference_steps is not None:
            self.policy.config.num_inference_steps = int(args.num_inference_steps)
        self.policy.to(args.device).eval()
        with _accept_min_range_processor_config():
            self.pre, self.post = make_pre_post_processors(
                self.policy.config, ckpt, preprocessor_overrides={"device_processor": {"device": args.device}}
            )
        cfg = self.policy.config
        # [C,H,W] in the config -> [H,W] per view.
        self.image_hw = {
            k.removeprefix("observation.images."): tuple(int(x) for x in f.shape[1:])
            for k, f in cfg.input_features.items()
            if k.startswith("observation.images.")
        }
        self.info: Dict[str, Any] = {
            "model": "pi",
            "checkpoint": ckpt,
            "prompt": args.prompt,
            "chunk_size": int(cfg.chunk_size),
            "n_action_steps": int(cfg.n_action_steps),
            "num_inference_steps": int(cfg.num_inference_steps),
            "state_dim": int(cfg.input_features["observation.state"].shape[0]),
            "action_names": list(getattr(cfg, "action_feature_names", None) or []),
            "image_hw": self.image_hw,
            "load_s": 0.0,
        }
        zeros = {v: np.zeros((*hw, 3), np.uint8) for v, hw in self.image_hw.items()}
        for _ in range(2):  # the first plan pays for CUDA kernel selection
            self.plan(np.zeros(self.info["state_dim"], np.float32), zeros)
        self.info["load_s"] = time.monotonic() - t

    def plan(self, state: np.ndarray, images: Dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        if set(images) != set(self.image_hw):
            raise ValueError(f"views {sorted(images)} != checkpoint views {sorted(self.image_hw)}")
        obs: Dict[str, Any] = {"observation.state": np.asarray(state, np.float32).copy()}
        for view, img in images.items():
            if img.shape[:2] != self.image_hw[view] or img.dtype != np.uint8:
                raise ValueError(
                    f"{view}: got {img.shape} {img.dtype}, the checkpoint was trained on "
                    f"{self.image_hw[view]} uint8 -- open the camera at that resolution"
                )
            obs[f"observation.images.{view}"] = np.array(img, copy=True)
        if self.args.seed is not None:
            torch.manual_seed(int(self.args.seed))
        batch = self._prepare(obs, torch.device(self.args.device), task=self.args.prompt)
        batch = self.pre(batch)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
        actions = self.post(actions)
        return actions[0].detach().float().cpu().numpy()

    def telemetry(self) -> Optional[Dict[str, Any]]:
        return None

    def reset_episode(self) -> None:
        self.policy.reset()


class _SteeredPi05:
    """Official D5 encoder/decoder or fixed baseline hooks, with training-time preprocessing."""

    def __init__(self, args: ServerArgs) -> None:
        import importlib
        import json
        import types

        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference

        self.args = args
        deployment_root = Path(args.steer_deployment_root).expanduser().resolve()
        entrypoint = deployment_root / "infer_real_cup_pingpong_d5_maturity.py"
        if not entrypoint.is_file():
            raise FileNotFoundError(f"steering deployment not found: {entrypoint}")
        sys.path.insert(0, str(deployment_root))
        # The published real-robot bundle imports a descriptor-KNN branch from its all-purpose hook
        # module, but omits that branch's version_b_descriptors.py.  D5 maturity never calls it; a
        # fail-closed placeholder lets the released D5 hook import while still rejecting accidental
        # use of the unavailable branch.
        missing_descriptor = deployment_root / "analysis" / "version_b_descriptors.py"
        if not missing_descriptor.exists() and "analysis.version_b_descriptors" not in sys.modules:
            unavailable = types.ModuleType("analysis.version_b_descriptors")

            class UnavailableDescriptorConfig:
                def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                    raise RuntimeError("descriptor-KNN assets were not included in the steering release")

            def unavailable_describe(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("descriptor-KNN assets were not included in the steering release")

            unavailable.DescriptorConfig = UnavailableDescriptorConfig
            unavailable.describe = unavailable_describe
            sys.modules[unavailable.__name__] = unavailable
        module = importlib.import_module("infer_real_cup_pingpong_d5_maturity")

        ckpt = Path(args.checkpoint).expanduser().resolve()
        t = time.monotonic()
        model = "steeract_enc" if args.steer else args.model
        if model == "steeract_dec" and args.num_inference_steps not in (None, 10):
            raise ValueError("The released decoder controller requires exactly 10 denoising steps")
        family = "expert" if model in ("steeract_dec", "coast") else "paligemma"
        layers = tuple(f"{family}_L{i}" for i in (14, 15, 16))
        with _skip_random_init(torch):
            if model in ("drvla", "coast"):
                baseline = importlib.import_module("infer_real_cup_pingpong_baselines")
                if args.selection is None:
                    raise ValueError(f"--selection is required for {model}")
                self.policy = baseline.RealBaselinePolicy(
                    method=model,
                    checkpoint=ckpt,
                    sae_root=Path(args.steer_sae_root).expanduser(),
                    selection=Path(args.selection).expanduser(),
                    device=args.device,
                    beta=args.coast_beta,
                )
            else:
                self.policy = module.CupPingpongD5MaturityPolicy(
                    checkpoint=ckpt,
                    sae_root=Path(args.steer_sae_root).expanduser(),
                    method_root=Path(args.steer_method_root).expanduser(),
                    device=args.device,
                    num_steps=args.num_inference_steps or 10,
                    layers=layers,
                    expert_mode=model == "steeract_dec",
                )
        self.policy.policy.config.num_inference_steps = int(args.num_inference_steps or 10)
        self._torch = torch
        self._prepare = prepare_observation_for_inference
        with _accept_min_range_processor_config():
            self.pre, self.post = make_pre_post_processors(
                self.policy.policy.config,
                str(ckpt),
                preprocessor_overrides={"device_processor": {"device": args.device}},
            )

        cfg = json.loads((ckpt / "config.json").read_text())
        self.image_hw = {
            key.removeprefix("observation.images."): tuple(int(x) for x in feature["shape"][1:])
            for key, feature in cfg["input_features"].items()
            if key.startswith("observation.images.")
        }
        self.info: Dict[str, Any] = {
            "model": model,
            "coast_beta": args.coast_beta if model == "coast" else None,
            "checkpoint": str(ckpt),
            "prompt": args.prompt,
            "chunk_size": int(cfg["chunk_size"]),
            "n_action_steps": int(cfg["n_action_steps"]),
            "num_inference_steps": int(args.num_inference_steps or 10),
            "state_dim": int(cfg["input_features"]["observation.state"]["shape"][0]),
            "action_names": list(cfg.get("action_feature_names") or []),
            "image_hw": self.image_hw,
            "steer": True,
            "steer_layers": list(layers),
            "steer_inference_contract": "standard_pi05_processor_raw_14d",
            "steer_sae_root": str(Path(args.steer_sae_root).expanduser().resolve()),
            "steer_method_root": str(Path(args.steer_method_root).expanduser().resolve()),
            "selection": args.selection,
            "steer_deployment_root": str(deployment_root),
            "load_s": 0.0,
        }
        # Pay CUDA kernel-selection cost before the robot is powered, then clear the controller's
        # causal history so the real episode still begins at decision zero.
        zeros = {view: np.zeros((*hw, 3), np.uint8) for view, hw in self.image_hw.items()}
        for _ in range(2):
            self.plan(np.zeros(self.info["state_dim"], np.float32), zeros)
        self.reset_episode()
        self.info["load_s"] = time.monotonic() - t

    def plan(self, state: np.ndarray, images: Dict[str, np.ndarray]) -> np.ndarray:
        if set(images) != set(self.image_hw):
            raise ValueError(f"views {sorted(images)} != checkpoint views {sorted(self.image_hw)}")
        for view, image in images.items():
            if image.shape[:2] != self.image_hw[view] or image.dtype != np.uint8:
                raise ValueError(
                    f"{view}: got {image.shape} {image.dtype}, the checkpoint was trained on "
                    f"{self.image_hw[view]} uint8 -- open the camera at that resolution"
                )
        # Use the same official preprocessing path as plain PI0.5.  The release's standalone infer()
        # contradicts STEERING.md by padding the raw 14-D state to 32 prompt tokens; that changes the
        # base policy even while the maturity gate is off.  The hooks are already registered on this
        # exact model, so standard inference still applies D5 steering when the controller admits it.
        obs: Dict[str, Any] = {"observation.state": np.array(state, dtype=np.float32, copy=True)}
        for view, image in images.items():
            obs[f"observation.images.{view}"] = np.array(image, copy=True)
        if self.args.seed is not None:
            self._torch.manual_seed(int(self.args.seed))
        batch = self._prepare(obs, self._torch.device(self.args.device), task=self.args.prompt)
        batch = self.pre(batch)
        with self._torch.inference_mode():
            actions = self.policy.policy.predict_action_chunk(batch)
        actions = self.post(actions)
        return actions[0].detach().float().cpu().numpy()

    def telemetry(self) -> Dict[str, Any]:
        """Return layer counters and this decision's traces, including all expert denoising steps."""
        result: Dict[str, Any] = {}
        for layer, state in self.policy.telemetry().items():
            trace = state.get("trace", [])
            calls_per_decision = self.info["num_inference_steps"] if layer.startswith("expert_") else 1
            result[layer] = {
                "calls": state.get("calls", 0),
                "applied": state.get("applied", 0),
                "total_dose": state.get("total_dose", trace[-1].get("total_layer_dose", 0.0) if trace else 0.0),
                "trace": trace[-1] if trace else None,
                "step_traces": trace[-calls_per_decision:],
                "alpha": getattr(self.policy, "alpha", None),
                "beta": getattr(self.policy, "beta", None),
            }
        return result

    def reset_episode(self) -> None:
        self.policy.reset_episode()


def serve(args: ServerArgs) -> None:
    mode = "steeract_enc" if args.steer else args.model
    print(f"[server] loading {args.checkpoint} (prompt {args.prompt!r}, {mode}) ...", flush=True)
    model = _Pi05(args) if mode == "pi" else _SteeredPi05(args)
    print(f"[server] ready in {model.info['load_s']:.0f}s: {model.info}", flush=True)
    listener = Listener(("127.0.0.1", int(args.port)), authkey=_AUTHKEY)
    while True:
        conn = listener.accept()
        conn.send(("info", model.info))
        try:
            while True:
                try:
                    kind, payload = conn.recv()
                except EOFError:
                    break
                if kind == "close":
                    conn.close()
                    return
                if kind == "reset":
                    model.reset_episode()
                    conn.send(("ok", None))
                    continue
                if kind != "plan":
                    conn.send(("error", f"unknown request {kind!r}"))
                    continue
                try:
                    t = time.perf_counter()
                    state = _unpack(payload["state"])
                    images = {v: _unpack(img) for v, img in payload["images"].items()}
                    chunk = model.plan(state, images)
                    if (
                        chunk.shape != (model.info["chunk_size"], model.info["state_dim"])
                        or not np.isfinite(chunk).all()
                    ):
                        raise ValueError(
                            f"Invalid policy chunk: shape={chunk.shape}, finite={np.isfinite(chunk).all()}"
                        )
                    conn.send(
                        (
                            "ok",
                            {
                                "chunk": _pack(chunk.astype(np.float32)),
                                "infer_s": time.perf_counter() - t,
                                "steering": model.telemetry(),
                            },
                        )
                    )
                except Exception:
                    conn.send(("error", traceback.format_exc()))
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Client (i2rt venv) -- no torch / lerobot import on this side
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pdeathsig() -> None:
    """In the server child: SIGTERM it if the robot process dies, so 10 GB of VRAM is not orphaned."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


class PolicyClient:
    """Spawns the server in the pi05 venv and exchanges plans with it, one request in flight at a time."""

    def __init__(
        self,
        checkpoint: Path,
        prompt: str,
        *,
        python: Path = DEFAULT_PYTHON,
        device: str = "cuda",
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        port: Optional[int] = None,
        steer: bool = False,
        steer_deployment_root: Path = DEFAULT_STEER_DEPLOYMENT_ROOT,
        steer_sae_root: Path = DEFAULT_STEER_SAE_ROOT,
        steer_method_root: Path = DEFAULT_STEER_METHOD_ROOT,
        model: Model = "pi",
        selection: Optional[Path] = None,
        coast_beta: float = 1.0,
    ) -> None:
        if not 0.0 <= coast_beta <= 1.0:
            raise ValueError("--coast-beta must be finite and in [0, 1]")
        self.checkpoint = Path(checkpoint).expanduser()
        self.prompt = prompt
        self._python = Path(python).expanduser()
        self._device = device
        self._steps = num_inference_steps
        self._seed = seed
        self._port = port
        self._steer = bool(steer)
        self._model = model
        self._selection = selection
        self._coast_beta = coast_beta
        self._steer_deployment_root = Path(steer_deployment_root).expanduser()
        self._steer_sae_root = Path(steer_sae_root).expanduser()
        self._steer_method_root = Path(steer_method_root).expanduser()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._conn: Optional[Any] = None
        self._pending_t0: Optional[float] = None
        self._sent_at = 0.0
        self.info: Dict[str, Any] = {}
        self.n_plans = 0
        self.last_steering: Optional[Dict[str, Any]] = None
        self.steering_telemetry: List[Dict[str, Any]] = []

    def start(self, timeout: float = 600.0) -> Dict[str, Any]:
        if not self._python.exists():
            raise FileNotFoundError(f"pi05 interpreter not found: {self._python}")
        port = self._port or _free_port()
        cmd: List[str] = [
            str(self._python),
            str(Path(__file__).resolve()),
            "--checkpoint",
            str(self.checkpoint),
            "--prompt",
            self.prompt,
            "--port",
            str(port),
            "--device",
            self._device,
            "--model",
            self._model,
            "--coast-beta",
            str(self._coast_beta),
        ]
        if self._steps is not None:
            cmd += ["--num-inference-steps", str(self._steps)]
        if self._seed is not None:
            cmd += ["--seed", str(self._seed)]
        if self._steer:
            cmd += ["--steer"]
        if self._steer or self._model != "pi":
            cmd += [
                "--steer-deployment-root",
                str(self._steer_deployment_root),
                "--steer-sae-root",
                str(self._steer_sae_root),
                "--steer-method-root",
                str(self._steer_method_root),
            ]
        if self._selection is not None:
            cmd += ["--selection", str(self._selection)]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.pop("VIRTUAL_ENV", None)
        # Own session: a terminal Ctrl-C reaches the robot process, which then shuts this down in
        # order, instead of killing the model mid-plan underneath a moving arm.
        self._proc = subprocess.Popen(cmd, env=env, start_new_session=True, preexec_fn=_pdeathsig)  # noqa: PLW1509
        deadline = time.monotonic() + timeout
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(f"policy server exited with code {self._proc.returncode} during startup")
            try:
                self._conn = Client(("127.0.0.1", port), authkey=_AUTHKEY)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"policy server did not come up within {timeout:.0f}s") from None
                time.sleep(0.5)
        kind, self.info = self._conn.recv()
        assert kind == "info", kind
        return self.info

    @property
    def busy(self) -> bool:
        return self._pending_t0 is not None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def request(self, t0: float, state: np.ndarray, images_rgb: Dict[str, np.ndarray]) -> bool:
        """Fire a plan for an observation taken at monotonic time ``t0``. False if one is in flight."""
        if self.busy or self._conn is None:
            return False
        payload = {
            "state": _pack(np.asarray(state, np.float32)),
            "images": {v: _pack(np.asarray(img, np.uint8)) for v, img in images_rgb.items()},
        }
        self._conn.send(("plan", payload))
        self._pending_t0 = float(t0)
        self._sent_at = time.monotonic()
        return True

    def poll(self, timeout: float = 0.0) -> Optional[Tuple[float, np.ndarray, float, float]]:
        """``(t0, chunk [T,14], latency_s observation->arrival, infer_s)`` once the plan is back."""
        if self._pending_t0 is None or self._conn is None:
            return None
        if not self._conn.poll(timeout):
            if not self.alive:
                raise RuntimeError(f"policy server died (exit code {self._proc.returncode if self._proc else None})")
            return None
        kind, payload = self._conn.recv()
        t0, self._pending_t0 = self._pending_t0, None
        if kind != "ok":
            raise RuntimeError(f"policy server failed a plan:\n{payload}")
        self.n_plans += 1
        self.last_steering = payload.get("steering")
        if self.last_steering is not None:
            self.steering_telemetry.append({"plan_index": self.n_plans, "layers": self.last_steering})
        return t0, _unpack(payload["chunk"]).copy(), time.monotonic() - t0, float(payload["infer_s"])

    def infer(self, state: np.ndarray, images_rgb: Dict[str, np.ndarray], timeout: float = 30.0) -> np.ndarray:
        """Blocking plan, for offline checks."""
        self.request(time.monotonic(), state, images_rgb)
        done = self.poll(timeout)
        if done is None:
            raise TimeoutError(f"no plan within {timeout:.0f}s")
        return done[1]

    def reset_episode(self) -> None:
        """Reset controller history and telemetry before another offline episode."""
        if self.busy or self._conn is None:
            raise RuntimeError("Cannot reset an unconnected or busy policy")
        self._conn.send(("reset", None))
        if not self._conn.poll(30.0):
            raise TimeoutError("Policy episode reset timed out")
        kind, payload = self._conn.recv()
        if kind != "ok":
            raise RuntimeError(f"Policy reset failed: {payload}")
        self.n_plans = 0
        self.last_steering = None
        self.steering_telemetry.clear()

    def close(self) -> None:
        if self._conn is not None:
            try:
                if self.busy:
                    self._conn.poll(5.0)
                self._conn.send(("close", None))
                self._conn.close()
            except (OSError, EOFError):
                pass
            self._conn = None
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, signal.SIGKILL)
                self._proc.wait(timeout=5.0)
        self._proc = None


if __name__ == "__main__":
    import tyro

    sys.exit(serve(tyro.cli(ServerArgs)))
