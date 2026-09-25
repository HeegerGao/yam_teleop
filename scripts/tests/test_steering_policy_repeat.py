"""Episode restart and exit tests, without cameras, CAN hardware or model weights."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import steering_policy_eval as ev


class Arm:
    num_dofs = 7

    def __init__(self, *_args: object) -> None:
        self.pose = np.zeros(7)
        self.commands: list[np.ndarray] = []

    def joint_pos(self) -> np.ndarray:
        return self.pose.copy()

    def command(self, pose: np.ndarray) -> None:
        self.pose = pose.copy()
        self.commands.append(pose.copy())

    def close(self) -> None:
        pass


def test_return_reaches_folded_before_task_pose() -> None:
    arms = {s: Arm() for s in ev.SIDES}
    limits = {s: ev.SafetyLimiter(np.tile([-10.0, 10.0], (6, 1)), 10000, 10000) for s in ev.SIDES}
    ready = {s: np.full(7, 0.3) for s in ev.SIDES}
    args = ev.Args(execute=True)
    assert ev._return_to_ready(arms, limits, ready, args, lambda: True)
    for side, arm in arms.items():
        assert len(arm.commands) == 4  # release -> working -> folded -> ready
        np.testing.assert_allclose(arm.commands[1][:6], ev.WORKING_Q[:6])
        np.testing.assert_allclose(arm.commands[2][:6], ev.FOLDED_Q[:6])
        np.testing.assert_allclose(arm.commands[3], ready[side])
        np.testing.assert_allclose(limits[side].last, ready[side])


def test_esc_cancels_return_before_next_pose() -> None:
    arms = {s: Arm() for s in ev.SIDES}
    limits = {s: ev.SafetyLimiter(np.tile([-10.0, 10.0], (6, 1)), 10000, 10000) for s in ev.SIDES}
    checks = iter((True, False))
    assert not ev._return_to_ready(
        arms, limits, {s: np.ones(7) for s in ev.SIDES}, ev.Args(execute=True), lambda: next(checks)
    )
    assert all(len(arm.commands) == 1 for arm in arms.values())


def test_esc_is_not_lost_when_clearing_queued_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev.KeyWatcher, "_start_terminal", lambda _self: None)
    keys = ev.KeyWatcher(global_keys=False)
    keys.feed(ord("q"))
    keys.feed(27)
    keys.drain()
    assert keys.pop() == "\x1b"


def test_retry_requires_enter_even_without_start_input(monkeypatch: pytest.MonkeyPatch) -> None:
    presses = iter((" ", "\n"))
    keys = SimpleNamespace(sources=[], drain=lambda: None, pop=lambda: next(presses))
    shutdown = SimpleNamespace(raise_if_requested=lambda: None)
    assert (
        ev._wait_for_start(
            keys, None, {}, ev.Args(display=False), shutdown, SimpleNamespace(say=lambda _: None), require_enter=True
        )
        == "start"
    )
    with pytest.raises(StopIteration):
        next(presses)  # SPACE did not start it; Enter was consumed.


def test_enter_cannot_skip_pending_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev.KeyWatcher, "_start_terminal", lambda _self: None)
    keys = ev.KeyWatcher(global_keys=False)
    presses = iter(("\n", "s", "\n"))
    monkeypatch.setattr(keys, "pop", lambda: next(presses))
    labels: list[str] = []
    result = ev._wait_for_start(
        keys,
        None,
        {},
        ev.Args(display=False),
        SimpleNamespace(raise_if_requested=lambda: None),
        SimpleNamespace(say=lambda _: None),
        require_enter=True,
        label_previous=labels.append,
    )
    assert result == "start"
    assert labels == ["success"]
    with pytest.raises(StopIteration):
        next(presses)


@pytest.mark.parametrize("key,outcome", [("s", "success"), ("f", "failure")])
def test_label_typed_during_return_is_saved_before_next_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    outcome: str,
) -> None:
    monkeypatch.setattr(ev.KeyWatcher, "_start_terminal", lambda _self: None)
    keys = ev.KeyWatcher(global_keys=False)
    keys.feed(ord(key))
    keys.feed(13)  # A premature Enter during the return must still be discarded.
    episode = tmp_path / "episode_0000"
    episode.mkdir()
    (episode / "meta.json").write_text('{"outcome": null, "failed": false}')
    saved: list[Path] = []
    original_pop = keys.pop

    def pop() -> str:
        value = original_pop()
        if value is None:
            assert saved, "The label queued during the return was lost"
            return "\n"  # A new Enter after the label has been saved.
        return value

    monkeypatch.setattr(keys, "pop", pop)
    result = ev._wait_for_start(
        keys,
        None,
        {},
        ev.Args(display=False),
        SimpleNamespace(raise_if_requested=lambda: None),
        SimpleNamespace(say=lambda _: None),
        require_enter=True,
        label_previous=lambda value: saved.append(ev._relabel(episode, value)),
    )
    assert result == "start"
    assert saved[0].name == ("episode_0000_failed" if outcome == "failure" else "episode_0000")
    assert json.loads((saved[0] / "meta.json").read_text())["outcome"] == outcome


def test_esc_can_exit_without_a_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev.KeyWatcher, "_start_terminal", lambda _self: None)
    keys = ev.KeyWatcher(global_keys=False)
    keys.feed(27)
    labels: list[str] = []
    assert (
        ev._wait_for_start(
            keys,
            None,
            {},
            ev.Args(display=False),
            SimpleNamespace(raise_if_requested=lambda: None),
            SimpleNamespace(say=lambda _: None),
            require_enter=True,
            label_previous=labels.append,
        )
        == "exit"
    )
    assert labels == []


def test_q_saves_resets_waits_then_esc_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(ev.KeyWatcher, "_start_terminal", lambda _self: None)
    keys = ev.KeyWatcher(global_keys=False)
    monkeypatch.setattr(ev, "KeyWatcher", lambda **_: keys)

    class Policy:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.n_plans = 0
            self.requests = 0
            self.busy = False
            self.steering_telemetry: list[dict[str, int]] = []

        def start(self) -> dict[str, Any]:
            events.append("load")
            return {"chunk_size": 50, "num_inference_steps": 10, "image_hw": {v: (16, 16) for v in ev.VIEWS}}

        def request(self, now: float, state: np.ndarray, _images: object) -> None:
            self.requests += 1
            self.busy = True
            self.pending = (now, np.tile(state, (50, 1)), 0.01, 0.01)
            if self.requests == 2:
                keys.feed(ord("q"))
            if self.requests == 4:
                keys.feed(27)

        def poll(self, timeout: float = 0) -> object:
            if not self.busy:
                return None
            self.busy = False
            self.n_plans += 1
            self.steering_telemetry.append({"plan_index": self.n_plans})
            return self.pending

        def reset_episode(self) -> None:
            assert not self.busy
            events.append("reset")
            self.n_plans = 0
            self.steering_telemetry.clear()

        def close(self) -> None:
            events.append("close")

    def return_to_ready(*_args: object) -> bool:
        events.append("fold_ready")
        assert len(list(tmp_path.rglob("meta.json"))) == 1
        return True

    def wait_for_start(*_args: object, **kwargs: object) -> str:
        assert kwargs["require_enter"] is True
        assert events[-2:] == ["fold_ready", "reset"]
        events.append("wait_enter")
        keys.drain()
        return "start"

    monkeypatch.setattr(ev, "PolicyClient", Policy)
    monkeypatch.setattr(ev, "_return_to_ready", return_to_ready)
    monkeypatch.setattr(ev, "_wait_for_start", wait_for_start)
    monkeypatch.setattr(ev, "_check_attached_followers", lambda *_: None)
    monkeypatch.setattr(ev, "FollowerArm", Arm)
    monkeypatch.setattr(ev, "demo_start_pose", lambda *_: (np.zeros(14), 1))
    monkeypatch.setattr(ev, "_stop_sequence", lambda *_args, **_kwargs: events.append("exit_fold"))
    monkeypatch.setattr(ev, "_check_camera_geometry", lambda *_: None)
    monkeypatch.setattr(ev, "_sample", lambda *_: {"value": np.zeros(7)})
    monkeypatch.setattr(ev, "_grab", lambda *_: {v: (np.zeros((16, 16, 3), np.uint8), 0.0) for v in ev.VIEWS})
    monkeypatch.setattr(ev.rec, "EEFKinematics", lambda *_: None)
    monkeypatch.setattr(
        ev.rec,
        "_open_camera_rig",
        lambda _: SimpleNamespace(
            cams_by_role={},
            all_fresh=lambda: True,
            stop=lambda: events.append("camera_close"),
        ),
    )
    monkeypatch.setattr(
        ev.rec,
        "ShutdownRequest",
        lambda _: SimpleNamespace(
            requested=threading.Event(),
            raise_if_requested=lambda: None,
        ),
    )
    monkeypatch.setattr(
        ev,
        "resolve_assets",
        lambda *_: SimpleNamespace(
            root=tmp_path,
            sae=tmp_path,
            method=tmp_path,
        ),
    )
    args = ev.Args(
        task="cup_pingpong",
        execute=True,
        sim=True,
        goto_start=False,
        wait_for_start=False,
        launch=False,
        display=False,
        global_keys=False,
        tts=False,
        label_outcome=False,
        replan_every=1,
        max_seconds=5,
        start_forward=0,
        start_down=0,
        start_tilt_to_base_deg=0,
        save_root=str(tmp_path),
    )
    ev.run(args, tmp_path)
    assert events == ["load", "fold_ready", "reset", "wait_enter", "exit_fold", "camera_close", "close"]
    records = sorted(
        (json.loads(p.read_text()) for p in tmp_path.rglob("meta.json")), key=lambda m: m["episode_number"]
    )
    assert len(records) == 2
    assert [m["stop_reason"] for m in records] == ["operator retry (q)", "operator quit (Esc)"]
    assert [m["ticks"] for m in records] == [2, 2]
    assert records[0]["session_dir"] != records[1]["session_dir"]
    assert Path(records[0]["session_dir"]) == tmp_path / "steering_cup_pingpong/pi/episode_0000"
    assert Path(records[1]["session_dir"]) == tmp_path / "steering_cup_pingpong/pi/episode_0001"
    assert records[1]["steering_telemetry"] == [{"plan_index": 1}]
