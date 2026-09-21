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


@pytest.mark.parametrize("frozen_click", [False, True])
def test_tripod_engine_to_eight_targets_and_logging(window, frozen_click):
    window.practice.resize(800, 500)
    window.start_practice()
    engine = window.tripod_engine
    timestamp = 100.0

    def frames(n, x=0.5, y=0.38, contact=0.85, grip=True):
        nonlocal timestamp
        for _ in range(n):
            timestamp += 1 / 30
            window._dispatch(
                engine.process(synthetic_tripod(timestamp, x, y, grip=grip, contact=contact))
            )

    frames(10)
    current = np.array([0.5, 0.38])
    for target in window.practice.target_list:
        point = (
            np.array(engine.camera_anchor)
            + (np.array(target) - engine.pointer_anchor) * engine.config.span
        )
        for value in np.linspace(current, point, 20):
            frames(1, *value)
        frames(15, *point)
        if frozen_click:
            frames(5, *point, grip=False)
            assert engine.state == "FROZEN"
        frames(3, *(point + 0.002), contact=0.32, grip=not frozen_click)
        frames(4, *(point + 0.004), contact=0.10, grip=not frozen_click)
        frames(12, *point)
        current = point
    assert window.practice.index == 8 and window.practice.misses == 0
    assert not window.practice.active
    rows = [json.loads(line) for line in window.practice.path.read_text().splitlines()]
    assert rows[0]["interaction_mode"] == "tripod" and rows[0]["source"] == "synthetic_demo"
    assert rows[0]["config"]["deadband"] == 0.008
    assert rows[0]["app_version"] == app_module.__version__


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
    "condition", ["TOUCHED", "FROZEN", "WAIT_CLEAR", "edge", "stalled", "cancelled"]
)
def test_probe_rejects_misleading_zero_jitter_conditions(condition):
    probe = StabilityProbe(100, {})
    step = 0.4 if condition == "stalled" else 1 / 30
    for timestamp in np.arange(100, 108, step):
        result = probe_result(
            state=condition if condition in ("TOUCHED", "FROZEN", "WAIT_CLEAR") else "CONTROL",
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
