import json
import os
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pinchpilot import app as app_module
from pinchpilot.domain import EngineResult, InputEvent
from pinchpilot.stability_probe import StabilityProbe
from pinchpilot.tripod_demo import synthetic_tripod


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    win = app_module.MainWindow(tmp_path, demo=True, interaction="tripod")
    win.timer.stop()
    yield win
    win.close()


def test_tripod_gui_uses_own_engine_and_preserves_native_and_training_boundaries(
    window, monkeypatch
):
    native = Mock(side_effect=AssertionError("unexpected native output"))
    monkeypatch.setattr(app_module, "MouseOutput", native)
    window._tick()
    assert window.camera_view.result.mode == "tripod"
    assert window.active_engine is window.tripod_engine
    assert not window.live_button.isEnabled() and not window.record_button.isEnabled()
    assert window.finger_controls.isHidden() and not window.tripod_controls.isHidden()
    window.source = "camera"
    window.last_packet_at = time.monotonic()
    window._toggle_live(True)
    window.start_recording()
    assert window.recorder is None and window.countdown is None
    native.assert_not_called()
    output = window.output = Mock()
    window._dispatch(
        EngineResult("TOUCHED", (0.5, 0.5), [InputEvent("down", 0.5, 0.5)], mode="tripod")
    )
    output.emit.assert_not_called()
    output.close.assert_called_once()


def test_tripod_settings_stop_trial_and_preserve_pinch_preferences(window):
    old_config = window.engine.config
    window.start_practice()
    window.tripod_controls.stability.setCurrentIndex(2)
    assert not window.practice.active
    assert window.tripod_engine.config.deadband == 0.014
    assert window.engine.config == old_config
    window.interaction_choice.setCurrentIndex(window.interaction_choice.findData("pinch"))
    assert window.active_engine is window.engine and window.tripod_controls.isHidden()


@pytest.mark.parametrize(
    "task,frozen_click",
    [
        ("click", False),
        ("click", True),
        ("right_click", False),
        ("right_click", True),
        ("drag", False),
    ],
)
def test_tripod_engine_to_eight_targets_and_logging(window, task, frozen_click):
    window.practice.resize(800, 500)
    window.task_choice.setCurrentIndex(window.task_choice.findData(task))
    window.start_practice()
    engine = window.tripod_engine
    timestamp = 100.0

    def frames(n, x=0.5, y=0.38, contact=0.85, grip=True, right_contact=0.85):
        nonlocal timestamp
        for _ in range(n):
            timestamp += 1 / 30
            window._dispatch(
                engine.process(
                    synthetic_tripod(
                        timestamp, x, y, grip=grip, contact=contact, right_contact=right_contact
                    )
                )
            )

    frames(10)
    current = np.array([0.5, 0.38])

    def move_to(target, contact=0.85):
        nonlocal current
        point = (
            np.array(engine.camera_anchor)
            + (np.array(target) - engine.pointer_anchor) * engine.config.span
        )
        for value in np.linspace(current, point, 20):
            frames(1, *value, contact=contact)
        frames(15, *point, contact=contact)
        current = point
        return point

    for target in window.practice.target_list:
        if task == "drag":
            point = move_to((0.35, 0.5))
            frames(16, *point, contact=0.10)
            assert engine.state == "DRAG" and window.practice.dragging
            point = move_to(target, contact=0.10)
            frames(8, *point)
            continue
        point = move_to(target)
        if frozen_click:
            frames(5, *point, grip=False)
            assert engine.state == "FROZEN"
        key = "right_contact" if task == "right_click" else "contact"
        frames(3, *(point + 0.002), grip=not frozen_click, **{key: 0.32})
        frames(6, *(point + 0.004), grip=not frozen_click, **{key: 0.10})
        frames(12, *point)
        current = point
    assert window.practice.index == 8 and window.practice.misses == 0
    assert not window.practice.active
    rows = [json.loads(line) for line in window.practice.path.read_text().splitlines()]
    assert rows[0]["interaction_mode"] == "tripod" and rows[0]["source"] == "synthetic_demo"
    assert rows[0]["config"]["deadband"] == 0.008
    assert rows[0]["app_version"] == app_module.__version__
    assert rows[0]["task"] == task
    assert all(row["button"] == row["expected_button"] for row in rows if row["type"] == "attempt")


def test_wrong_button_does_not_pass_right_task_and_cancelled_press_does_not_score(window):
    window.task_choice.setCurrentIndex(window.task_choice.findData("right_click"))
    window.start_practice()
    target = window.practice.target_list[0]
    window._dispatch(
        EngineResult(
            "CONTROL",
            target,
            [InputEvent("down", *target), InputEvent("up", *target)],
            mode="tripod",
        )
    )
    assert window.practice.index == 0 and window.practice.misses == 1
    window._dispatch(
        EngineResult(
            "RIGHT_TOUCHED",
            target,
            [InputEvent("right_down", *target), InputEvent("right_up", *target)],
            mode="tripod",
        )
    )
    assert window.practice.index == 1
    window.task_choice.setCurrentIndex(0)
    assert not window.practice.active
    window.start_practice()
    window._dispatch(EngineResult("PRESSED", target, [InputEvent("down", *target)], mode="tripod"))
    window._dispatch(
        EngineResult(
            "WAIT_GRIP", target, [InputEvent("up", *target)], mode="tripod", cancelled=True
        )
    )
    assert (
        window.practice.index == 0 and window.practice.misses == 0 and not window.practice.pressed
    )


@pytest.mark.parametrize(
    "action",
    ["pause", "stop", "end_task", "recenter", "mode", "settings", "task", "method", "emergency"],
)
def test_app_lifecycle_cancels_held_drag_without_success_or_stuck_button(window, action):
    window.task_choice.setCurrentIndex(window.task_choice.findData("drag"))
    window.start_practice()
    engine = window.tripod_engine
    for index in range(30):
        window._dispatch(
            engine.process(synthetic_tripod(100 + index / 30, contact=0.85 if index < 10 else 0.1))
        )
    assert engine.left_down and window.practice.pressed
    if action == "pause":
        window._pause(True)
    elif action == "stop":
        window.stop_all()
    elif action == "end_task":
        window.stop_practice()
    elif action == "recenter":
        window._recenter_finger()
    elif action == "mode":
        window.interaction_choice.setCurrentIndex(0)
    elif action == "settings":
        window.tripod_controls.drag_hold.setCurrentIndex(1)
    elif action == "task":
        window.task_choice.setCurrentIndex(0)
    elif action == "method":
        window.task_method.setCurrentIndex(1)
    else:
        window._emergency()
    assert not engine.left_down and not window.practice.pressed
    assert window.practice.index == 0 and window.practice.misses == 0


def test_physical_right_button_is_scored_as_right_and_legacy_modes_cannot_start_it(window):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    window.task_method.setCurrentIndex(1)
    window.task_choice.setCurrentIndex(window.task_choice.findData("right_click"))
    window.start_practice()
    x, y = window.practice.target_list[0]
    QTest.mouseClick(
        window.practice,
        Qt.MouseButton.RightButton,
        pos=QPoint(round(x * window.practice.width()), round(y * window.practice.height())),
    )
    assert window.practice.index == 1
    window.task_method.setCurrentIndex(0)
    window.interaction_choice.setCurrentIndex(0)
    assert not window.task_choice.model().item(2).isEnabled()
    window.task_choice.setCurrentIndex(2)
    assert window.task_choice.currentData() == "click"
    window.task_choice.blockSignals(True)
    window.task_choice.setCurrentIndex(2)
    window.task_choice.blockSignals(False)
    window.start_practice()
    assert not window.practice.active


def test_probe_rejects_demo_and_cancels_on_mode_change(window):
    window.start_probe()
    assert window.probe is None
    window.source = "camera"  # Simulated camera source; no hardware is opened.
    window.last_packet_at = time.monotonic()
    window.start_probe()
    assert window.probe.metadata["config"]["mode"] == "tripod"
    window.interaction_choice.setCurrentIndex(0)
    assert window.probe is None
    saved = list((window.report_dir / "jitter").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["cancelled"]


def probe_result(state="CONTROL", raw=(0.50, 0.50), pointer=(0.50, 0.50)):
    return EngineResult(state, pointer, mode="tripod", raw_pointer=raw)


def test_probe_only_counts_fresh_observed_control_and_reports_spread():
    probe = StabilityProbe(100, {"source": "synthetic_test"})
    for i in range(241):
        result = probe_result(raw=(0.5 + (i % 2) * 0.01, 0.5))
        probe.add(100 + i / 30, result)
        probe.add(100 + i / 30, result)
    report = probe.report(108)
    assert report["frames"] == 241 and report["sufficient"]
    assert report["valid_observed_seconds"] == pytest.approx(8)
    assert report["raw"]["rms_radius"] > 0.004
    assert report["output"]["rms_radius"] == 0


@pytest.mark.parametrize(
    "condition",
    ["PRESSED", "DRAG", "RIGHT_TOUCHED", "FROZEN", "WAIT_CLEAR", "edge", "stalled", "cancelled"],
)
def test_probe_rejects_misleading_zero_jitter_conditions(condition):
    probe = StabilityProbe(100, {})
    step = 0.4 if condition == "stalled" else 1 / 30
    for timestamp in np.arange(100, 108, step):
        result = probe_result(
            state=condition if condition.isupper() else "CONTROL",
            raw=(1.2, 0.5) if condition == "edge" else (0.5, 0.5),
        )
        probe.add(float(timestamp), result)
    report = probe.report(108, cancelled=condition == "cancelled")
    assert not report["sufficient"]


def test_probe_timer_saves_without_waiting_or_changing_output(window, monkeypatch):
    now = 100.0
    monkeypatch.setattr(app_module.time, "monotonic", lambda: now)
    window.source, window.last_packet_at = "camera", now
    window.start_probe()
    assert window.probe.starts_at == 102
    for i in range(241):
        window.probe.add(102 + i / 30, probe_result())
    now = 110
    window._probe_tick(now)
    assert window.probe is None and window.output is None
    report = json.loads(next((window.report_dir / "jitter").glob("*.json")).read_text())
    assert report["sufficient"] and not report["cancelled"]


def test_engine_reset_cancels_probe_instead_of_mixing_two_anchors(window):
    window.source, window.last_packet_at = "camera", time.monotonic()
    window.start_probe()
    window._disable_live()
    assert window.probe is None
    report = json.loads(next((window.report_dir / "jitter").glob("*.json")).read_text())
    assert report["cancelled"] and not report["sufficient"]
