"""No hardware/GPU: prevent evaluation mixing and unintended fallback to another policy."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import steering_policy_eval as evaluation
from steering_policy_models import Assets, Model, new_episode, resolve_assets
from steering_policy_server import _SteeredPi05


def test_switching_back_continues_episode_numbering(tmp_path: Path) -> None:
    def create(path: Path) -> Path:
        path.mkdir(exist_ok=False)
        return path

    models: tuple[Model, ...] = ("pi", "steeract_enc", "pi", "pi", "steeract_dec", "drvla", "coast")
    outputs = [new_episode(tmp_path, "cup_pingpong", model, create) for model in models]
    assert len(set(outputs)) == len(models)
    for output, model in zip(outputs, models, strict=True):
        assert output.parent == tmp_path / "steering_cup_pingpong" / model
        assert output.is_dir()
    assert [p.name for p in outputs[:4]] == ["episode_0000", "episode_0000", "episode_0001", "episode_0002"]
    outputs[3].rename(outputs[3].with_name("episode_0002_failed"))
    assert new_episode(tmp_path, "cup_pingpong", "pi", create).name == "episode_0003"


@pytest.mark.parametrize("task", ["cloth", "push_cube"])
@pytest.mark.parametrize("model", ["drvla", "coast"])
def test_missing_assets_fail_before_robot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, model: Model
) -> None:
    def unexpected_run(*_args: object) -> None:
        pytest.fail("Robot workflow must not start with missing method assets")

    monkeypatch.setattr(evaluation, "run", unexpected_run)
    args = evaluation.Args(task=task, model=model, model_root=str(tmp_path), execute=True)
    with pytest.raises(SystemExit, match=f"{task}/{model} is unavailable"):
        evaluation.main(args)


def test_decoder_telemetry_keeps_all_ten_denoising_steps() -> None:
    traces = [{"denoising_step": i % 10, "total_layer_dose": float(i)} for i in range(20)]
    adapter = _SteeredPi05.__new__(_SteeredPi05)
    adapter.info = {"num_inference_steps": 10}
    adapter.policy = SimpleNamespace(telemetry=lambda: {"expert_L14": {"calls": 20, "trace": traces}})
    telemetry = adapter.telemetry()["expert_L14"]
    assert telemetry["step_traces"] == traces[-10:]
    assert telemetry["total_dose"] == 19.0


@pytest.mark.parametrize("task", ["cloth", "push_cube"])
@pytest.mark.parametrize("model", ["drvla", "coast"])
def test_new_baselines_resolve_task_specific_assets(tmp_path: Path, task: str, model: Model) -> None:
    checkpoint = tmp_path / task / "020000/pretrained_model"
    assets = Assets(task, model, tmp_path, checkpoint)
    required = [
        checkpoint / name
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
    required.extend(
        tmp_path / "deployment" / name
        for name in (
            "infer_real_cup_pingpong_d5_maturity.py",
            "infer_real_cup_pingpong_baselines.py",
        )
    )
    suffix = "drvla_alpha100" if model == "drvla" else "coast_conceptor"
    required.extend(assets.selection.parent / f"{layer}.{suffix}.npy" for layer in assets.layers)
    if model == "drvla":
        required.extend(assets.sae / layer / "topk_er4_k100.pt" for layer in assets.layers)
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assets.selection.write_text(json.dumps({"layers": assets.layers, "checkpoint": str(checkpoint)}))
    resolved = resolve_assets(task, model, tmp_path)
    assert resolved.checkpoint == checkpoint
    assert resolved.selection == tmp_path / task / model / "selection.json"
    assert resolved.sae == tmp_path / task / "sae"
    assert resolved.layers == tuple(f"{'paligemma' if model == 'drvla' else 'expert'}_L{i}" for i in (14, 15, 16))
