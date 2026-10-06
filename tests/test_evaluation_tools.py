import json
import os
from dataclasses import asdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from test_desktop_control import Clock, Rig

from pinchpilot.desktop import DesktopWindow
from pinchpilot.domain import EngineResult, HandFrame
from pinchpilot.trial_metrics import TrialMetrics
from pinchpilot.vision import Packet


def diagnostic(reason="moving"):
    return {
        "reason": reason,
        "label": reason,
        "state": "CONTROL",
        "values": {"grip_keep_margin": 0.2},
    }


def test_issue_marker_contains_only_recent_context_and_respects_logging_off(tmp_path):
    clock = Clock()
    metrics = TrialMetrics(tmp_path, clock)
    metrics.context_source = "camera"
    for i in range(450):
        clock.now = 100 + i / 30
        metrics.observe_diagnostic(diagnostic("moving" if i < 420 else "grip_unstable"), clock.now)
    path = metrics.mark_issue("jitter", "camera", {"span": 0.3})
    report = json.loads(path.read_text(encoding="utf-8"))
    assert metrics.file is None
    assert report["user_reported"] and not report["stores_keypoints"]
    assert report["context"][-1]["reason"] == "grip_unstable"
    assert all(
        clock.now - metrics.started - row["elapsed_s"] <= 10.001 for row in report["context"]
    )
    assert all(row["source"] == "camera" for row in report["context"])
    text = path.read_text(encoding="utf-8")
    assert not any(key in text for key in ('"landmarks"', '"rgb"', '"pointer"'))
    assert list(tmp_path.iterdir()) == [path]


def test_issue_marker_flushes_existing_log_without_new_file(tmp_path):
    metrics = TrialMetrics(tmp_path, Clock())
    metrics.set_logging(True, {}, "synthetic_demo")
    path = metrics.mark_issue("missed_click", "synthetic_demo", {})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "issue_marker"
    assert rows[-1]["source"] == "synthetic_demo"
    assert len(list(tmp_path.iterdir())) == 1
    metrics.close()


def test_expired_context_and_invalid_mark_category(tmp_path):
    clock = Clock()
    metrics = TrialMetrics(tmp_path, clock)
    metrics.observe_diagnostic(diagnostic(), clock.now)
    clock.now += 11
    path = metrics.mark_issue("other", "none", {})
    assert json.loads(path.read_text(encoding="utf-8"))["context"] == []
    with pytest.raises(ValueError):
        metrics.mark_issue("bad", "none", {})


def test_diagnostic_durations_are_read_only_on_snapshot(tmp_path):
    clock = Clock()
    metrics = TrialMetrics(tmp_path, clock)
    metrics.observe_diagnostic(diagnostic(), clock.now)
    clock.now += 2
    metrics.observe_diagnostic(diagnostic("grip_unstable"), clock.now)
    clock.now += 0.04
    one = metrics.snapshot(clock.now)
    assert one == metrics.snapshot(clock.now)
    assert one["reason_seconds"]["moving"] == 2
    assert one["reason_duration_s"] == 0.04


def test_stage_measurements_and_slot_overwrites_are_not_sensor_latency(tmp_path):
    clock = Clock()
    metrics = TrialMetrics(tmp_path, clock)
    metrics.reset_source("camera")
    packet = Packet(
        None,
        HandFrame(99.98),
        8,
        30,
        99.98,
        capture_ms=12,
        preprocess_ms=2,
        ready_at=99.99,
        overwritten_results=3,
    )
    metrics.observe(packet, EngineResult("WAIT_GRIP"), clock.now)
    metrics.observe(packet, EngineResult("WAIT_GRIP"), clock.now)
    data = metrics.snapshot(clock.now)
    assert data["timings"]["gui_wait"]["last_ms"] == 10
    assert data["timings"]["capture_read"]["last_ms"] == 12
    assert data["overwritten_results"] == 3
    assert data["timings"]["native_emit"]["last_ms"] is None
    metrics.reset_source("synthetic_demo")
    metrics.observe(packet, EngineResult("WAIT_GRIP"), clock.now)
    assert metrics.snapshot(clock.now)["timings"]["inference"]["samples"] == 0
    for value in (None, -1, float("nan"), float("inf")):
        metrics.observe_timing("engine", value)
    assert metrics.snapshot(clock.now)["timings"]["engine"]["samples"] == 0


def test_practice_releases_native_and_cannot_enable_until_closed(tmp_path):
    rig = Rig(tmp_path)
    rig.live()
    rig.frames(4, contact=0.1)
    assert rig.output.down
    rig.control.begin_practice()
    assert not rig.output.down and not rig.control.active
    before = list(rig.output.events)
    with pytest.raises(RuntimeError, match="测试台"):
        rig.control.enable()
    rig.frames(10)
    rig.frames(6, contact=0.1)
    assert rig.output.events == before
    rig.control.end_practice()
    assert not rig.control.active
    rig.control.enable()
    assert rig.control.active
    rig.control.close()


def test_pending_release_blocks_practice(tmp_path):
    rig = Rig(tmp_path)
    rig.live()
    rig.frames(4, contact=0.1)
    rig.output.fail_close = True
    with pytest.raises(RuntimeError, match="松键"):
        rig.control.begin_practice()
    assert not rig.control.practice_active
    rig.output.fail_close = False
    rig.control.close()


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_desktop_evaluation_controls_have_no_native_side_effects(application, tmp_path):
    rig = Rig(tmp_path / "reports/desktop_trials")
    window = DesktopWindow(tmp_path, controller=rig.control)
    window.timer.stop()
    try:
        window.open_practice()
        assert window.practice_window is not None
        assert rig.control.practice_active and not window.live_button.isEnabled()
        window.open_diagnostics()
        assert "相机关闭" in window.metrics_window.reason.text()
        assert not rig.created
        window.mark_issue()
        assert "已标记" in rig.control.notice
        window.stop_control()
        assert window.practice_window is None and not rig.control.practice_active
        window.open_practice()
        window._configure()
        assert window.practice_window is None and not rig.control.active
    finally:
        window.close()


def test_practice_start_failure_cannot_bypass_native_isolation(application, tmp_path, monkeypatch):
    from pinchpilot.practice_window import PracticeWindow

    rig = Rig(tmp_path)
    window = DesktopWindow(tmp_path, controller=rig.control)
    window.timer.stop()
    try:
        with monkeypatch.context() as patch:

            def fail(_self, _config):
                raise RuntimeError("injected initialization failure")

            patch.setattr(PracticeWindow, "set_config", fail)
            window.open_practice()
        assert window.practice_window is None and not rig.control.practice_active
        rig.control.set_source("camera")
        rig.frames()
        window.open_practice()
        assert window.practice_window is not None and rig.control.practice_active
        with pytest.raises(RuntimeError, match="测试台"):
            rig.control.enable()
        assert not rig.created
    finally:
        window.close()


def test_controller_diagnostics_and_timings_dont_change_engine_output(tmp_path):
    from pinchpilot.tripod import TripodEngine
    from pinchpilot.tripod_demo import tripod_demo_frame

    rig = Rig(tmp_path)
    rig.control.set_source("synthetic_demo")
    plain = TripodEngine(rig.control.engine.config)
    for i in range(900):
        rig.clock.now = 100 + (i + 1) / 30
        frame = tripod_demo_frame(rig.clock.now, i / 30)
        expected = plain.process(frame)
        assert rig.control.consume(Packet(None, frame, 0, 30, frame.timestamp))
        assert asdict(rig.control.result) == asdict(expected)
        rig.control.tick()
        assert asdict(rig.control.result) == asdict(plain.tick(rig.clock.now))
    assert not rig.created
    assert rig.control.metrics.snapshot(rig.clock.now)["timings"]["engine"]["samples"] == 180
