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
            {"pointer_basis": "wrist", "motion_profile": "precise", "wrist_calibration": [0] * 15}
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
    window.rig.clock.now = capture.started + 6
    window.rig.frames()
    window._refresh()
    assert window.wrist_calibration_progress.value() == 50
    assert "50%" in window.wrist_calibration_status.text()
    assert window.framing_status_label.isHidden()
    assert not window.camera_view.frame_edges
    assert not window.framing_overlay.isVisible()
    save = Mock(wraps=window._save_settings)
    monkeypatch.setattr(window, "_save_settings", save)
    window.rig.clock.now = capture.started + 12
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
    started = window.controller.wrist_capture.started
    for index in range(366):
        elapsed = (index + 0.1) / 30
        window.rig.clock.now = started + elapsed
        pose = ((0, 0, 0), (0, 0, 15), (15, 0, 0))[min(2, int(elapsed / 4))]
        frame = hand(window.rig.clock.now, pose)
        window.controller.consume(Packet(None, frame, 0.0, 30.0, frame.timestamp))
        window.controller.tick()
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
        window.wrist_calibration_button,
        window.wrist_calibration_progress,
        window.wrist_calibration_status,
    ):
        top = widget.mapTo(viewport, QPoint(0, 0)).y()
        assert top >= 0 and top + widget.height() <= viewport.height()
