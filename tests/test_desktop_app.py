import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider
from test_desktop_control import FakeMouse, Rig

from pinchpilot import desktop
from pinchpilot.desktop_control import DesktopController
from pinchpilot.domain import InputEvent
from pinchpilot.tripod_demo import synthetic_tripod
from pinchpilot.vision import Packet


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    rig = Rig(tmp_path / "reports/desktop_trials")
    win = desktop.DesktopWindow(tmp_path, controller=rig.control)
    win.timer.stop()
    win.rig = rig
    yield win
    rig.output.fail_close = False
    rig.output.fail_emit = ""
    win.close()


def test_startup_is_off_and_synthetic_demo_cannot_activate_mouse(window, monkeypatch):
    assert window.worker is None and window.controller.source == "none"
    assert not window.live_button.isChecked() and not window.live_button.isEnabled()
    assert window.logging.isChecked() and window.controller.metrics.file is not None
    window.start_demo()
    window.controller.output_factory = Mock(side_effect=AssertionError("native injection"))
    window._toggle_live(True)
    window.controller.output_factory.assert_not_called()
    assert not window.live_button.isChecked()


def test_controls_persist_only_preferences_and_restart_requires_explicit_enable(
    window, application
):
    window.rig.live()
    window.rig.frames(4, contact=0.1)
    assert window.rig.output.down
    window.right.setChecked(False)
    assert not window.rig.output.down and not window.controller.active
    window.drag.setChecked(False)
    window.scroll.setChecked(False)
    window.preview.setChecked(False)
    window.logging.setChecked(False)
    window.tripod_controls.stability.setCurrentIndex(2)
    saved = json.loads(window.settings_path.read_text())
    assert saved["right"] is False and saved["drag"] is False
    assert "active" not in saved and "source" not in saved
    second = desktop.DesktopWindow(window.workspace)
    try:
        second.timer.stop()
        assert not second.right.isChecked() and not second.drag.isChecked()
        assert not second.scroll.isChecked() and not second.controller.engine.config.scroll_enabled
        assert not second.preview.isChecked() and second.camera_view.isHidden()
        assert second.controller.metrics.file is None
        assert second.controller.engine.config.deadband == 0.014
        assert second.controller.source == "none" and not second.controller.active
    finally:
        second.close()


def test_settings_invalid_values_do_not_restore_control(application, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "desktop-settings.json").write_text(
        json.dumps({"right": "false", "span": 500, "camera_index": 999, "active": True})
    )
    win = desktop.DesktopWindow(tmp_path)
    try:
        assert win.right.isChecked() and win.controller.engine.config.span == 0.3
        assert win.camera_index.value() == 0 and not win.controller.active
    finally:
        win.close()


@pytest.mark.parametrize("span,percent", [(0.2, 150), (0.3, 100), (0.4, 75), (1.2, 25), (3.0, 10)])
def test_sensitivity_restores_old_and_continuous_settings(application, tmp_path, span, percent):
    data = tmp_path / "data"
    data.mkdir()
    (data / "desktop-settings.json").write_text(json.dumps({"span": span, "stability": 0.014}))
    win = desktop.DesktopWindow(tmp_path)
    try:
        win.timer.stop()
        assert win.tripod_controls.sensitivity.value() == percent
        assert win.controller.engine.config.span == pytest.approx(span)
        assert win.controller.engine.config.deadband == 0.014
        win.tripod_controls.sensitivity.setValue(42)
        saved = json.loads(win.settings_path.read_text())
        assert saved["span"] == pytest.approx(0.3 / 0.42)
        win.tripod_controls.restore_span(saved["span"])
        assert win.tripod_controls.sensitivity.value() == 42
        win.tripod_controls.reset_sensitivity.click()
        assert win.controller.engine.config.span == 0.3
    finally:
        win.close()


def test_slider_drag_previews_then_commits_and_releases_on_mouse_up(window, application):
    window.advanced_button.setChecked(True)
    window.show()
    application.processEvents()
    slider = window.tripod_controls.sensitivity
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, slider
    )
    start = handle.center()
    end = QPoint(max(12, start.x() - slider.width() // 3), start.y())
    window.rig.live()
    window.rig.frames(18, contact=0.1)
    assert window.rig.output.down
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(slider, end)
    assert slider.sliderPosition() < 75  # Slower than the old slowest setting.
    assert slider.value() == 100 and window.controller.active
    assert window.tripod_controls.sensitivity_value.text() != "1.00×"
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=end)
    assert slider.value() < 75 and not window.controller.active
    assert not window.rig.output.down
    assert window.controller.engine.config.span > 0.4


def test_scroll_reaches_output_metrics_and_settings_switch_stops_it(window):
    from pinchpilot.demo import synthetic_hand

    rig = window.rig
    rig.live()
    for i in range(70):
        rig.clock.now += 1 / 30
        frame = synthetic_hand(rig.clock.now, "scroll", y=0.5 - max(0, i - 12) * 0.0015)
        window.controller.consume(Packet(None, frame, 8.0, 30.0, frame.timestamp))
        window.controller.tick()
    assert window.controller.result.state == "SCROLL"
    wheel = [e for e in rig.output.events if e.kind == "scroll"]
    assert wheel and all(e.value > 0 for e in wheel)
    window._refresh()
    assert f"滚 {len(wheel)}" in window.metrics_labels["duration"].text()
    assert "SCROLL" in window.gesture.text()
    window.tripod_controls.scroll_speed.setCurrentIndex(0)
    assert not window.controller.active
    assert window.controller.engine.config.scroll_gain == 30
    window.scroll.setChecked(False)
    assert not window.controller.engine.config.scroll_enabled
    saved = json.loads(window.settings_path.read_text())
    assert saved["scroll"] is False and saved["scroll_speed"] == 30


def test_camera_permission_and_download_callbacks_cannot_restart_after_stop(
    window, monkeypatch, application
):
    callbacks = []
    monkeypatch.setattr(window, "_job", lambda fn, done, generation: callbacks.append(done))
    worker = Mock()
    monkeypatch.setattr(desktop, "CameraWorker", worker)
    window.start_camera()
    assert window.camera_pending
    window.stop_camera()
    callbacks[0](Path("test.task"))
    worker.assert_not_called()

    permissions = []
    monkeypatch.setattr(desktop, "request_camera_access", permissions.append)
    window.start_camera()
    callbacks[1](Path("test.task"))
    assert len(permissions) == 1
    window.stop_camera()
    permissions[0](True)
    application.processEvents()
    worker.assert_not_called()
    assert window.controller.source == "none"


def test_camera_permission_denial_and_old_failure_do_not_break_current_source(
    window, monkeypatch, application
):
    monkeypatch.setattr(desktop, "request_camera_access", lambda callback: callback(False))
    window._begin_camera(window.generation)
    application.processEvents()
    assert "未获准" in window.notice.text()
    window.start_demo()
    window._camera_failed(window.generation - 1, "late failure")
    assert window.controller.source == "synthetic_demo"
    assert "late failure" not in window.notice.text()


def test_camera_failure_stops_control_and_releases_held_button(window):
    window.rig.live()
    window.rig.frames(4, contact=0.1)
    window.worker = Mock(failure="camera disconnected")
    window.worker.is_alive.return_value = False
    window._tick()
    assert not window.controller.active and not window.rig.output.down
    assert window.controller.source == "none"
    assert "camera disconnected" in window.notice.text()


def test_expired_frame_cannot_enable_button_or_show_current_fps(window):
    window.rig.live()
    window.rig.frames(8)
    window._refresh()
    assert window.live_button.isChecked() and window.live_button.isEnabled()
    window.rig.clock.now += 0.3
    window.controller.tick()
    window._refresh()
    assert not window.live_button.isEnabled() and not window.live_button.isChecked()
    assert "0.0 FPS" in window.metrics_labels["fps"].text()


def test_close_cleans_up_and_failed_cleanup_keeps_window_available(window):
    window.rig.live()
    window.rig.frames(4, contact=0.1)
    window.rig.output.fail_close = True
    event = Mock()
    window.closeEvent(event)
    event.ignore.assert_called_once()
    assert window.controller.pending_release and not window.controller.active
    window.rig.output.fail_close = False
    window.closeEvent(event)
    event.accept.assert_called_once()
    assert not window.rig.output.down


def test_minimized_window_continues_heartbeat_but_skips_preview(application, tmp_path):
    output = FakeMouse()
    control = DesktopController(tmp_path / "reports", output_factory=lambda: output)
    win = desktop.DesktopWindow(tmp_path, controller=control)
    control.set_source("camera")

    class Worker:
        failure = None

        def pop(self):
            now = time.monotonic()
            return Packet(None, synthetic_tripod(now), 8.0, 30.0, now)

        def stop(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    win.worker = Worker()
    control.consume(win.worker.pop())
    control.enable()
    win.showMinimized()
    win.camera_view.set_frame = Mock()
    loop = QEventLoop()
    QTimer.singleShot(650, loop.quit)
    try:
        loop.exec()
        assert control.active
        win.camera_view.set_frame.assert_not_called()
        assert output.events
    finally:
        win.close()


def test_queued_model_job_completes_on_the_gui_thread(window, application):
    from PySide6.QtCore import QThread

    seen = []
    window._job(
        lambda: 7, lambda value: seen.append((value, QThread.currentThread())), window.generation
    )
    deadline = time.monotonic() + 3
    while window.jobs and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    assert not window.jobs and seen == [(7, application.thread())]


def test_metrics_show_native_sends_not_preview_gesture_counts(window):
    window.controller.set_source("camera")
    window.rig.frames(8)
    window.rig.frames(18, contact=0.1)
    window._refresh()
    assert "左 0" in window.metrics_labels["buttons"].text()
    window.rig.live()
    window.rig.frames(4, contact=0.1)
    window.rig.frames(8)
    window._refresh()
    assert "左 1" in window.metrics_labels["buttons"].text()
    assert window.rig.output.events.count(InputEvent("down", *window.rig.output.point)) == 1
