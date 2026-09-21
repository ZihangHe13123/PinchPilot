import json
import os
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from pinchpilot import app as app_module
from pinchpilot.cli import parser
from pinchpilot.demo import synthetic_finger
from pinchpilot.domain import EngineResult, InputEvent
from pinchpilot.single_finger import finger_features


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    win = app_module.MainWindow(tmp_path, demo=True)
    win.timer.stop()
    yield win
    win.close()


def select(window, mode):
    window.interaction_choice.setCurrentIndex(window.interaction_choice.findData(mode))


@pytest.mark.parametrize("mode", ["finger-flex", "finger-dwell"])
def test_mode_switch_and_camera_source_cannot_enable_native_or_old_training(
    window, monkeypatch, mode
):
    window.start_practice()
    native = window.output = Mock()
    window.engine.config.engage_ratio = 0.29
    select(window, mode)
    native.close.assert_called_once()
    assert not window.practice.active and window.output is None
    assert not window.live_button.isEnabled() and not window.record_button.isEnabled()
    assert not window.models_group.isEnabled()
    assert window.participant.isEnabled() and window.session.isEnabled()
    window.source = "camera"
    window.last_packet_at = time.monotonic()
    window.current_frame = synthetic_finger(window.last_packet_at)
    constructor = Mock(side_effect=AssertionError("single-finger attempted native output"))
    monkeypatch.setattr(app_module, "MouseOutput", constructor)
    window._toggle_live(True)
    window.start_recording()
    constructor.assert_not_called()
    assert window.output is None and window.recorder is None and window.countdown is None
    assert not list(window.record_dir.glob("*.jsonl"))
    select(window, "pinch")
    assert window.live_button.isEnabled() and window.record_button.isEnabled()
    assert window.models_group.isEnabled() and window.engine.config.engage_ratio == 0.29


@pytest.mark.parametrize("selected", ["pinch", "finger-flex"])
def test_dispatch_guards_both_active_mode_and_event_origin(window, selected):
    select(window, selected)
    native = window.output = Mock()
    window._dispatch(
        EngineResult("POINT", (0.5, 0.5), [InputEvent("down", 0.5, 0.5)], mode="finger-flex")
    )
    native.close.assert_called_once()
    native.emit.assert_not_called()
    assert window.output is None


def test_single_finger_uses_own_demo_and_does_not_call_loaded_classifier(window):
    select(window, "finger-flex")
    window.predictor = Mock()
    window.recognizer.setCurrentIndex(1)
    window._tick()
    assert window.camera_view.result.mode == "finger-flex"
    assert finger_features(window.current_frame) is not None
    window.predictor.predict.assert_not_called()
    window._emergency()
    assert not window.finger_engine.enabled
    window._tick()
    assert window.camera_view.result.state == "PAUSED"


def test_settings_end_task_and_drag_is_only_available_for_physical_mouse(window):
    select(window, "finger-dwell")
    assert window.task_choice.currentData() == "click"
    assert not window.task_choice.model().item(1).isEnabled()
    window.start_practice()
    window.finger_controls.dwell.setValue(1.1)
    assert not window.practice.active
    assert window.finger_engine.config.dwell_seconds == 1.1
    window.task_method.setCurrentIndex(1)
    assert window.task_choice.model().item(1).isEnabled()
    window.task_choice.setCurrentIndex(1)
    window.start_practice()
    assert window.practice.task == "drag" and not window.practice.virtual
    window.practice.stop()
    window.task_method.setCurrentIndex(0)
    assert window.task_choice.currentData() == "click"
    # Direct invocation cannot bypass the visual disabled option either.
    window.task_choice.setCurrentIndex(1)
    window.start_practice()
    assert not window.practice.active


@pytest.mark.parametrize("mode", ["finger-flex", "finger-dwell"])
def test_real_engine_completes_eight_synthetic_targets_with_mode_metadata(window, mode):
    select(window, mode)
    window.practice.resize(800, 500)
    window.start_practice()
    engine = window.finger_engine
    timestamp = 100.0

    def frames(n, **kwargs):
        nonlocal timestamp
        for _ in range(n):
            timestamp += 1 / 30
            window._dispatch(engine.process(synthetic_finger(timestamp, **kwargs)))

    frames(10)
    base = finger_features(synthetic_finger(timestamp)).relative
    for target in window.practice.target_list:
        dx, dy = tuple(
            (v - p) / engine.config.gain + r - b
            for v, p, r, b in zip(target, engine.pointer_anchor, engine.relative_anchor, base)
        )
        frames(40, dx=dx, dy=dy)
        if mode == "finger-flex":
            frames(6, dx=dx, dy=dy, bend=0.18)
            frames(6, dx=dx, dy=dy)
    assert not window.practice.active
    rows = [json.loads(line) for line in window.practice.path.read_text().splitlines()]
    assert rows[0]["interaction_mode"] == mode
    assert rows[0]["app_version"] == app_module.__version__
    assert rows[0]["source"] == "synthetic_demo"
    assert rows[0]["config"]["mode"] == mode
    assert rows[-1]["completed"] and rows[-1]["hits"] == 8 and rows[-1]["misses"] == 0
    window.event_file.flush()
    with open(window.event_file.name) as log:
        events = [json.loads(line) for line in log]
    assert len(events) == 16
    assert all(e["interaction_mode"] == mode and e["output"] == "preview" for e in events)


def test_cli_mode_selection_and_initial_window(application, tmp_path):
    args = parser().parse_args(["gui", "--interaction", "finger-dwell", "--demo"])
    win = app_module.MainWindow(tmp_path, demo=args.demo, interaction=args.interaction)
    try:
        win.timer.stop()
        win._tick()
        assert win.interaction_mode == "finger-dwell"
        assert win.camera_view.result.mode == "finger-dwell"
        assert not win.live_button.isEnabled()
    finally:
        win.close()
