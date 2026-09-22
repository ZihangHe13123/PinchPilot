"""Offscreen test bench UI; no cameras or native input objects."""

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QWidget

from pinchpilot.domain import EngineResult, InputEvent
from pinchpilot.practice_window import PracticeWindow


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    item = PracticeWindow(tmp_path)
    item.show()
    application.processEvents()
    yield item
    item.close()
    application.processEvents()


def start(window, source="camera"):
    window.feed(EngineResult("CONTROL", (0.5, 0.5), grip=0.2), time.monotonic(), source)
    assert window.start_button.isEnabled()
    window.start_task()
    now = window.session.started + 0.01
    window.feed(EngineResult("CONTROL", (0.5, 0.5), grip=0.2), now, source)
    return now


def test_waits_for_input_and_start_locks_task_choice(window):
    assert not window.start_button.isEnabled() and window.session is None
    assert window.task_choice.count() == 5
    start(window)
    assert window.session.active and not window.task_choice.isEnabled()
    assert window.cancel_button.isEnabled()
    window.cancel_task()
    assert window.report_path.exists() and not window.session.active
    report = json.loads(window.report_path.read_text())
    assert report["reason"] == "user_cancelled" and report["source"] == "camera"


def test_completed_export_captures_config_not_later_mutation(window):
    config = {"pointer_basis": "unified", "span": 0.3}
    window.set_config(config)
    now = start(window, "synthetic_demo")
    config["span"] = 0.9
    while window.session.active:
        target = window.session.target
        now += 0.05
        window.feed(
            EngineResult("PRESSED", target, [InputEvent("down", *target)], grip=0.2),
            now,
            "synthetic_demo",
        )
        now += 0.05
        window.feed(
            EngineResult("CONTROL", target, [InputEvent("up", *target)], grip=0.2),
            now,
            "synthetic_demo",
        )
    report = json.loads(window.report_path.read_text())
    assert report["status"] == "completed" and report["evidence"] == "synthetic_engineering"
    assert report["engine_config"]["span"] == 0.3
    assert len(report["trials"]) == 5


def test_close_exports_partial_once_and_emits_signal(window):
    start(window)
    spy = QSignalSpy(window.closed)
    window.close()
    path = window.report_path
    window.close()
    report = json.loads(path.read_text())
    assert spy.count() == 1 and report["reason"] == "window_closed"
    assert len(list(path.parent.glob("practice_*.json"))) == 1


def test_source_change_is_visible_cancel_and_preserves_original_source(window):
    now = start(window)
    window.feed(EngineResult("CONTROL", (0.5, 0.5), grip=0.2), now + 0.05, "synthetic_demo")
    report = json.loads(window.report_path.read_text())
    assert report["reason"] == "source_changed" and report["source"] == "camera"
    assert "已取消" in window.metrics.text()


def test_invalidate_disables_start_cancels_held_and_does_not_refresh_input(window):
    now = start(window)
    point = window.session.target
    window.feed(EngineResult("PRESSED", point, [InputEvent("down", *point)], grip=0.2), now + 0.01)
    window.invalidate(now + 0.05, "camera")
    assert not window.start_button.isEnabled() and window.last_feed_time is None
    assert window.session.pressed is None and not window.session.active
    assert "输入已过期" in window.metrics.text()
    report = json.loads(window.report_path.read_text())
    assert report["reason"] == "stale_input" and report["input_interruptions"] == 1
    window.invalidate(now + 0.1, "camera")
    assert len(list(window.report_path.parent.glob("*.json"))) == 1


def test_invalidate_before_first_frame_cannot_start(window):
    window.invalidate(time.monotonic(), "camera")
    assert window.session is None and not window.start_button.isEnabled()


def test_export_failure_is_visible_and_retry_succeeds(window, tmp_path):
    start(window)
    blocker = tmp_path / "reports"
    blocker.write_text("not a folder")
    window.cancel_task()
    assert window.report_path is None and "结果尚未保存" in window.feedback.text()
    blocker.unlink()
    window.close()
    assert window.report_path.exists()


def test_small_window_controls_and_canvas_visible(window, application):
    window.resize(600, 540)
    application.processEvents()
    start(window)
    for key in ("click", "double_click", "drag", "reverse", "scroll"):
        window.cancel_task()
        window.task_choice.setCurrentIndex(window.task_choice.findData(key))
        start(window)
        application.processEvents()
        assert not window.canvas.grab().isNull()
        assert window.cancel_button.isVisible() and window.feedback.isVisible()
        assert window.feedback.geometry().bottom() <= window.height()
        assert window.canvas.height() >= 180


def test_repeated_parented_windows_are_deleted_after_close(application, tmp_path):
    parent = QWidget()
    try:
        for _ in range(3):
            item = PracticeWindow(tmp_path, parent)
            item.show()
            application.processEvents()
            assert len(parent.findChildren(PracticeWindow)) == 1
            item.close()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            application.processEvents()
            assert not parent.findChildren(PracticeWindow)
    finally:
        parent.close()
