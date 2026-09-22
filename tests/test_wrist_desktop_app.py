"""Wrist UI/preferences tests use synthetic packets and FakeMouse only."""

import json
import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication
from test_desktop_control import Rig
from test_wrist import calibration, hand
from test_wrist_desktop_control import CALIBRATION, FakeWristCalibration

from pinchpilot import desktop
from pinchpilot.domain import EngineResult
from pinchpilot.tripod_controls import TripodControls
from pinchpilot.tripod_demo import synthetic_tripod
from pinchpilot.vision import Packet
from pinchpilot.widgets import CameraView


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    rig = Rig(tmp_path / "reports/desktop_trials")
    win = desktop.DesktopWindow(tmp_path, controller=rig.control)
    win.timer.stop()
    win.tripod_controls.pointer_basis.setCurrentIndex(
        win.tripod_controls.pointer_basis.findData("position")
    )
    win.rig = rig
    yield win
    rig.output.fail_close = False
    rig.output.fail_emit = ""
    win.close()


def select_basis(window, basis):
    combo = window.tripod_controls.pointer_basis
    combo.setCurrentIndex(combo.findData(basis))


def test_position_defaults_and_wrist_switch_preserve_position_tuning(window):
    controls = window.tripod_controls
    assert controls.pointer_basis.currentData() == "position"
    assert controls.motion_profile.currentData() == "precise"
    assert window.wrist_calibration_group.isHidden()
    controls.restore_noise(0.001, 0.002)
    controls.motion_profile.setCurrentIndex(controls.motion_profile.findData("adaptive"))
    position_box = window.controller.engine.active_box
    select_basis(window, "wrist")
    config = window.controller.engine.config
    assert config.pointer_basis == "wrist"
    assert (config.rest_noise_x, config.rest_noise_y) == (0, 0)
    assert (controls.rest_noise_x, controls.rest_noise_y) == (0.001, 0.002)
    assert controls.motion_profile.currentData() == "adaptive"
    assert not controls.motion_profile.isEnabled()
    assert "固定增益精细滤波" in controls.interaction_hint.text()
    assert "停腕即停" in controls.interaction_hint.text()
    assert window.controller.engine.active_box is None and window.camera_view.box is None
    assert not window.wrist_calibration_group.isHidden()
    assert window.calibration_button.isHidden() and not window.calibration_button.isEnabled()
    assert not window.live_button.isEnabled()
    select_basis(window, "position")
    config = window.controller.engine.config
    assert config.motion_profile == "adaptive" and controls.motion_profile.isEnabled()
    assert (config.rest_noise_x, config.rest_noise_y) == (0.001, 0.002)
    assert window.controller.engine.active_box == position_box == window.camera_view.box
    assert window.wrist_calibration_group.isHidden()
    assert not window.calibration_button.isHidden()
    assert not window.rig.created and not window.rig.output.events


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "invalid",
        [0] * 14,
        [True] + list(CALIBRATION[1:]),
        [float("nan")] + list(CALIBRATION[1:]),
        [0] * 15,
        CALIBRATION[:12] + CALIBRATION[9:12],
    ],
)
def test_wrist_restore_rejects_corruption_without_changing_good_value(application, invalid):
    controls = TripodControls()
    try:
        spy = QSignalSpy(controls.changed)
        assert controls.restore_wrist_calibration(list(CALIBRATION))
        assert not controls.restore_wrist_calibration(invalid)
        assert controls.wrist_calibration == CALIBRATION and spy.count() == 0
        assert controls.restore_wrist_calibration(())
        assert controls.wrist_calibration == () and spy.count() == 0
    finally:
        controls.close()


def test_wrist_preferences_restart_and_clear_keep_position_calibration(window):
    controls = window.tripod_controls
    controls.restore_noise(0.001, 0.002)
    controls.restore_wrist_calibration(CALIBRATION)
    controls.motion_profile.setCurrentIndex(controls.motion_profile.findData("adaptive"))
    select_basis(window, "wrist")
    saved = json.loads(window.settings_path.read_text())
    assert saved["pointer_basis"] == "wrist"
    assert saved["wrist_calibration"] == list(CALIBRATION)
    assert saved["rest_noise_x"] == 0.001 and saved["motion_profile"] == "adaptive"
    second_rig = Rig(window.workspace / "reports/restarted")
    second = desktop.DesktopWindow(window.workspace, controller=second_rig.control)
    try:
        second.timer.stop()
        config = second.controller.engine.config
        assert config.pointer_basis == "wrist" and config.wrist_calibration == CALIBRATION
        assert (config.rest_noise_x, config.rest_noise_y) == (0, 0)
        assert second.tripod_controls.rest_noise_x == 0.001
        assert second.tripod_controls.motion_profile.currentData() == "adaptive"
        assert not second.tripod_controls.motion_profile.isEnabled()
        assert "已恢复" in second.wrist_calibration_status.text()
        assert not second.controller.active and not second.controller.calibrating
        second.controller.set_source("camera")
        second_rig.frames()
        second._refresh()
        assert second.live_button.isEnabled()
        second.clear_wrist_calibration_button.click()
        assert second.controller.engine.config.wrist_calibration == ()
        assert not second.live_button.isEnabled()
        assert json.loads(second.settings_path.read_text())["wrist_calibration"] == []
        select_basis(second, "position")
        assert second.controller.engine.config.rest_noise_x == 0.001
        assert second.tripod_controls.motion_profile.currentData() == "adaptive"
        assert not second_rig.created and not second_rig.output.events
    finally:
        second.close()


@pytest.mark.parametrize(
    "stored,expected", [({}, "classic"), ({"motion_profile": "adaptive"}, "adaptive")]
)
def test_pre_wrist_settings_migrate_to_position(application, tmp_path, stored, expected):
    data = tmp_path / "data"
    data.mkdir()
    (data / "desktop-settings.json").write_text(json.dumps(stored))
    rig = Rig(tmp_path / "reports")
    win = desktop.DesktopWindow(tmp_path, controller=rig.control)
    try:
        win.timer.stop()
        assert win.controller.engine.config.pointer_basis == "position"
        assert win.controller.engine.config.motion_profile == expected
        assert win.controller.engine.config.wrist_calibration == ()
        assert not rig.created
    finally:
        win.close()


def test_corrupt_saved_wrist_calibration_requires_calibration_before_enable(application, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "desktop-settings.json").write_text(
        json.dumps(
            {
                "pointer_basis": "wrist",
                "motion_profile": "precise",
                "wrist_calibration": [0] * 15,
                "unified_mode_seen": True,
            }
        )
    )
    rig = Rig(tmp_path / "reports")
    win = desktop.DesktopWindow(tmp_path, controller=rig.control)
    try:
        win.timer.stop()
        rig.control.set_source("camera")
        rig.frames()
        win._refresh()
        assert win.controller.engine.config.pointer_basis == "wrist"
        assert win.controller.engine.config.wrist_calibration == ()
        assert not win.live_button.isEnabled()
        assert win.wrist_calibration_button.isEnabled()
        assert "尚未校准" in win.wrist_calibration_status.text()
        assert not rig.created
    finally:
        win.close()


def test_wrist_calibration_ui_progress_saves_once_and_requires_manual_enable(window, monkeypatch):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    select_basis(window, "wrist")
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    window.wrist_calibration_button.click()
    capture = window.controller.wrist_capture
    assert capture is not None and "取消" in window.wrist_calibration_button.text()
    assert not window.live_button.isEnabled() and not window.controller.active
    capture.stage = 1
    capture.reference_hint = lambda: "相对中立向右转腕 · 当前 15°"
    window.rig.frames(50)
    window._refresh()
    assert window.wrist_calibration_progress.value() == 50
    assert "50%" in window.wrist_calibration_status.text()
    assert "【② 向右】" in window.wrist_calibration_steps.text()
    assert "15°" in window.wrist_calibration_reference.text()
    assert "方向校准 · 系统输入已关闭" in window.gesture.text()
    assert "12 秒" not in window.wrist_calibration_button.text()
    assert window.framing_status_label.isHidden()
    assert not window.camera_view.frame_edges
    assert not window.framing_overlay.isVisible()
    save = Mock(wraps=window._save_settings)
    monkeypatch.setattr(window, "_save_settings", save)
    capture.complete = True
    window.rig.frames()
    window._refresh()
    window._refresh()
    save.assert_called_once()
    assert not window.controller.calibrating and not window.controller.active
    assert window.wrist_calibration_progress.value() == 100
    assert window.live_button.isEnabled() and not window.live_button.isChecked()
    assert "手动启用" in window.wrist_calibration_status.text()
    assert window.tripod_controls.wrist_calibration == CALIBRATION
    assert json.loads(window.settings_path.read_text())["wrist_calibration"] == list(CALIBRATION)
    assert not window.rig.created and not window.rig.output.events


def test_wrist_calibration_button_cancel_and_position_switch_remain_off(window, monkeypatch):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    select_basis(window, "wrist")
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    window.wrist_calibration_button.click()
    assert window.controller.calibrating
    window.wrist_calibration_button.click()
    assert not window.controller.calibrating
    assert window.wrist_calibration_progress.value() == 0
    window.wrist_calibration_button.click()
    assert window.controller.calibrating
    select_basis(window, "position")
    assert not window.controller.calibrating
    assert not window.controller.active and not window.rig.created
    assert window.wrist_calibration_group.isHidden()
    assert window.calibration_button.isEnabled()


def test_camera_preview_without_mapping_box_paints_pointer(application):
    view = CameraView()
    try:
        view.box = None
        view.set_frame(
            None, synthetic_tripod(0), EngineResult("MOVE", pointer=(0.4, 0.6), mode="tripod")
        )
        view.show()
        application.processEvents()
        assert not view.grab().isNull()
        view.box = (0.2, 0.2, 0.8, 0.8)
        view.update()
        application.processEvents()
        assert not view.grab().isNull()
    finally:
        view.close()


def test_real_wrist_capture_through_controller_persists_three_demonstrated_poses(window):
    """Ideal 3D geometry verifies wiring; it is not camera/MediaPipe evidence."""
    select_basis(window, "wrist")
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    window.wrist_calibration_button.click()

    def feed(pose):
        window.rig.clock.now += 1 / 30
        frame = hand(window.rig.clock.now, pose)
        window.controller.consume(Packet(None, frame, 0.0, 30.0, frame.timestamp))
        window.controller.tick()

    for stage, pose in enumerate(((0, 0, 0), (0, 0, 15), (15, 0, 0))):
        if stage == 2:
            for _ in range(40):
                feed((0, 0, 0))
        for _ in range(200):
            capture = window.controller.wrist_capture
            if capture is None or capture.stage > stage:
                break
            feed(pose)
        else:
            pytest.fail(window.controller.wrist_capture.hint(window.rig.clock.now))
    window._refresh()
    assert window.controller.engine.config.wrist_calibration == pytest.approx(calibration())
    saved = json.loads(window.settings_path.read_text())
    assert saved["wrist_calibration"] == pytest.approx(calibration())
    assert window.wrist_calibration_progress.value() == 100
    assert window.live_button.isEnabled() and not window.live_button.isChecked()
    assert not window.controller.active and not window.rig.created and not window.rig.output.events


@pytest.mark.parametrize("preview", [True, False])
def test_short_window_scroll_reveals_wrist_calibration_controls(window, application, preview):
    window.preview.setChecked(preview)
    window.resize(980, 610)
    window.show()
    window.advanced_button.setChecked(True)
    select_basis(window, "wrist")
    application.processEvents()
    window._reveal_wrist_calibration()
    application.processEvents()
    viewport = window.settings_area.viewport()
    assert window.settings_area.verticalScrollBar().maximum() > 0
    for widget in (
        window.wrist_pose_guide,
        window.wrist_calibration_progress,
        window.wrist_calibration_status,
    ):
        top = widget.mapTo(viewport, QPoint(0, 0)).y()
        assert top >= 0 and top + widget.height() <= viewport.height()
    window.settings_area.ensureWidgetVisible(window.wrist_calibration_button)
    application.processEvents()
    top = window.wrist_calibration_button.mapTo(viewport, QPoint(0, 0)).y()
    assert top >= 0 and top + window.wrist_calibration_button.height() <= viewport.height()


@pytest.mark.parametrize("previous_calibration", [False, True])
def test_wrist_failure_is_persistent_visible_and_keeps_old_calibration(
    window, application, monkeypatch, previous_calibration
):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    if previous_calibration:
        window.tripod_controls.restore_wrist_calibration(CALIBRATION)
    select_basis(window, "wrist")
    window.controller.set_source("camera")
    window.rig.frames()
    window.show()
    window._refresh()
    window.wrist_calibration_button.click()
    window.advanced_button.setChecked(False)
    capture = window.controller.wrist_capture
    cause = "向右转动仅 2.1°，低于 8°；请稍多转动手腕并停稳"
    capture.failure = cause
    capture.complete = True
    window.rig.frames()
    window._refresh()
    application.processEvents()
    assert window.advanced_button.isChecked()
    assert cause in window.wrist_calibration_status.text()
    assert window.wrist_calibration_status.styleSheet()
    assert window.wrist_calibration_progress.value() == 0
    assert not window.controller.active and not window.live_button.isChecked()
    if previous_calibration:
        assert window.controller.engine.config.wrist_calibration == CALIBRATION
        assert "保留此前校准" in window.wrist_calibration_status.text()
    viewport = window.settings_area.viewport()
    top = window.wrist_calibration_status.mapTo(viewport, QPoint(0, 0)).y()
    assert top >= 0 and top + window.wrist_calibration_status.height() <= viewport.height()
    window.settings_area.verticalScrollBar().setValue(0)
    window.controller.notice = "普通状态更新"
    window._refresh()
    window._refresh()
    application.processEvents()
    assert cause in window.wrist_calibration_status.text()
    assert window.settings_area.verticalScrollBar().value() == 0
    window.clear_wrist_calibration_button.click()
    assert window.controller.wrist_calibration_error == ""
    assert not window.wrist_calibration_status.styleSheet()
    assert "尚未校准" in window.wrist_calibration_status.text()
    assert not window.rig.created


def test_retry_replaces_failure_with_live_quality_hint(window, monkeypatch):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    select_basis(window, "wrist")
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    window.wrist_calibration_button.click()
    window.controller.wrist_capture.failure = "姿态不稳定"
    window.controller.wrist_capture.complete = True
    window.rig.frames()
    window._refresh()
    assert "姿态不稳定" in window.wrist_calibration_status.text()
    window.wrist_calibration_button.click()
    assert window.controller.wrist_calibration_error == ""
    assert not window.wrist_calibration_status.styleSheet()
    assert "腕动校准" in window.wrist_calibration_status.text()
    assert "姿态不稳定" not in window.wrist_calibration_status.text()
    assert not window.controller.active


def test_pose_demo_follows_quality_flags_and_stops_during_collection(
    window, application, monkeypatch
):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    select_basis(window, "wrist")
    window.controller.set_source("camera")
    window.rig.frames()
    window.show()
    window.advanced_button.setChecked(True)
    window._refresh()
    window.wrist_calibration_button.click()
    capture = window.controller.wrist_capture
    capture.stage = 1
    capture.collecting = False
    capture.need_neutral = False
    window._refresh()
    application.processEvents()
    assert window.wrist_pose_guide.stage == 1
    assert window.wrist_pose_guide.timer.isActive()
    capture.collecting = True
    window._refresh()
    assert not window.wrist_pose_guide.timer.isActive()
    capture.stage = 2
    capture.collecting = False
    capture.need_neutral = True
    window._refresh()
    assert window.wrist_pose_guide.stage == 2
    assert not window.wrist_pose_guide.timer.isActive()
    capture.need_neutral = False
    window._refresh()
    assert window.wrist_pose_guide.timer.isActive()
    assert window.controller.calibrating and not window.controller.active and not window.rig.created
