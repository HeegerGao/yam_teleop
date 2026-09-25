"""Task/method asset selection and isolated evaluation directories (no GPU imports)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypeVar

REPO_ID = "lzy001Yuki/SteeringPolicyReal"
REVISION = "b42b04972a24d47b6266606050bd9ea1cdf56f0d"
LOCAL_ROOT = Path("~/steering_policy_real").expanduser()
RELEASE_ROOT = LOCAL_ROOT / "releases" / REVISION
Model = Literal["pi", "steeract_enc", "steeract_dec", "drvla", "coast"]
MODELS: tuple[Model, ...] = ("pi", "steeract_enc", "steeract_dec", "drvla", "coast")
TASKS = ("cup_pingpong", "cloth", "push_cube")
T = TypeVar("T")


@dataclass(frozen=True)
class Assets:
    task: str
    model: Model
    root: Path
    checkpoint: Path

    @property
    def sae(self) -> Path:
        return self.root / self.task / "sae"

    @property
    def method(self) -> Path:
        return self.root / self.task / "method_v1_3"

    @property
    def selection(self) -> Path:
        return self.root / self.task / self.model / "selection.json"

    @property
    def layers(self) -> tuple[str, ...]:
        family = "paligemma" if self.model in ("steeract_enc", "drvla") else "expert"
        return tuple(f"{family}_L{i}" for i in (14, 15, 16))

    def validate(self) -> None:
        required = [
            self.checkpoint / name
            for name in (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_3_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                "tokenizer/tokenizer.json",
            )
        ]
        if self.model != "pi":
            required.append(self.root / "deployment/infer_real_cup_pingpong_d5_maturity.py")
        if self.model.startswith("steeract"):
            required.append(self.method / "online_intensity_knn_banks.json")
            required.extend(self.method / "weights/D5" / layer / "T0_top1.npy" for layer in self.layers)
        if self.model in ("steeract_enc", "steeract_dec", "drvla"):
            required.extend(self.sae / layer / "topk_er4_k100.pt" for layer in self.layers)
        if self.model in ("drvla", "coast"):
            required.extend([self.selection, self.root / "deployment/infer_real_cup_pingpong_baselines.py"])
            if self.selection.is_file():
                manifest = json.loads(self.selection.read_text())
                if tuple(manifest["layers"]) != self.layers:
                    raise ValueError(f"Unexpected {self.model} layers in {self.selection}: {manifest['layers']}")
            suffix = "drvla_alpha100" if self.model == "drvla" else "coast_conceptor"
            required.extend(self.selection.parent / f"{layer}.{suffix}.npy" for layer in self.layers)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self.task}/{self.model} is unavailable; missing assets:\n  "
                + "\n  ".join(missing)
                + "\nDownload with scripts/steering_policy_download.py or supply --model-root with task-specific assets."
            )


def resolve_assets(task: str, model: Model = "pi", root: Path = RELEASE_ROOT, checkpoint: str | None = None) -> Assets:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", task) or model not in MODELS:
        raise ValueError(f"Invalid task/model: {task}/{model}")
    root = root.expanduser().resolve()
    subdir = "finetuned_ckpt" if model.startswith("steeract") else "020000"
    ckpt = Path(checkpoint).expanduser().resolve() if checkpoint else root / task / subdir / "pretrained_model"
    assets = Assets(task, model, root, ckpt)
    assets.validate()
    return assets


def new_episode(save_root: Path, task: str, model: Model, create: Callable[[Path], T]) -> T:
    """Create the next numbered episode directly below the model, without overwriting recordings.

    ``create`` must atomically create the directory with exist_ok=False (as EpisodeWriter does).
    Retry collisions so simultaneous evaluators cannot share an episode. Failed episodes count
    toward the next index; legacy timestamp directories are left untouched.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", task) or model not in MODELS:
        raise ValueError(f"Invalid task/model: {task}/{model}")
    parent = save_root.expanduser() / f"steering_{task}" / model
    parent.mkdir(parents=True, exist_ok=True)
    index = (
        max(
            (
                int(match[1])
                for path in parent.iterdir()
                if (match := re.fullmatch(r"episode_(\d+)(?:_failed)?", path.name))
            ),
            default=-1,
        )
        + 1
    )
    while True:
        path = parent / f"episode_{index:04d}"
        if path.with_name(f"{path.name}_failed").exists():
            index += 1
            continue
        try:
            return create(path)
        except FileExistsError:
            index += 1
