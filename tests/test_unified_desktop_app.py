"""Unified desktop defaults/migration; only temporary settings and FakeMouse."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from test_desktop_control import Rig
from test_wrist_desktop_control import CALIBRATION

from pinchpilot import desktop
from pinchpilot.tripod_controls import TripodControls


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(application, tmp_path):
    windows = []

    def make(settings=None):
        if settings is not None:
            path = tmp_path / "data/desktop-settings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings))
        rig = Rig(tmp_path / "reports/desktop_trials")
        window = desktop.DesktopWindow(tmp_path, controller=rig.control)
        window.timer.stop()
        window.rig = rig
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.rig.output.fail_close = False
        window.rig.output.fail_emit = ""
        window.close()


def choose(window, basis):
    choice = window.tripod_controls.pointer_basis
    choice.setCurrentIndex(choice.findData(basis))


def test_fresh_desktop_defaults_to_unified_without_calibration(make_window):
    window = make_window()
    controls = window.tripod_controls
    config = window.controller.engine.config
    assert controls.pointer_basis.itemData(0) == "unified"
    assert controls.pointer_basis.currentText() == "兼容定位 · 平移＋转腕"
    assert config.pointer_basis == "unified" and config.motion_profile == "precise"
    assert config.wrist_calibration == () and (config.rest_noise_x, config.rest_noise_y) == (0, 0)
    assert not controls.motion_profile.isEnabled()
    assert not controls.motion_form.isRowVisible(controls.motion_profile)
    assert controls.motion_form.isRowVisible(controls.fixed_motion_profile)
    assert controls.fixed_motion_profile.text() == "固定精细（灵敏度可调）"
    assert controls.motion_form.isRowVisible(controls.basis_summary)
    assert "无需方向校准" in controls.basis_summary.text()
    assert "无需方向校准" in controls.interaction_hint.text()
    assert window.wrist_calibration_group.isHidden()
    assert all(widget.isHidden() for widget in window.rest_calibration_widgets)
    assert not window.controller.active and not window.rig.created
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    assert window.live_button.isEnabled()
    window.live_button.click()
    assert window.controller.active and window.rig.created == 1
    assert not window.controller.calibrating


def test_shared_controls_keep_other_demos_on_position_default(application):
    controls = TripodControls()
    try:
        assert controls.pointer_basis.itemData(0) == "unified"
        assert controls.configuration().pointer_basis == "position"
        assert controls.configuration().motion_profile == "classic"
    finally:
        controls.close()


def test_unified_fixed_profile_retains_position_choices_and_noise(make_window):
    window = make_window()
    controls = window.tripod_controls
    controls.restore_noise(0.001, 0.002)
    controls.restore_wrist_calibration(CALIBRATION)
    controls.motion_profile.setCurrentIndex(controls.motion_profile.findData("adaptive"))
    window._configure()
    config = window.controller.engine.config
    assert config.motion_profile == "precise" and config.wrist_calibration == CALIBRATION
    assert (config.rest_noise_x, config.rest_noise_y) == (0, 0)
    assert controls.motion_profile.currentData() == "adaptive"
    saved = json.loads(window.settings_path.read_text())
    assert saved["motion_profile"] == "adaptive" and saved["rest_noise_x"] == 0.001
    assert saved["wrist_calibration"] == list(CALIBRATION)
    choose(window, "position")
    config = window.controller.engine.config
    assert controls.motion_profile.isEnabled() and config.motion_profile == "adaptive"
    assert controls.motion_form.isRowVisible(controls.motion_profile)
    assert not controls.motion_form.isRowVisible(controls.fixed_motion_profile)
    assert not controls.motion_form.isRowVisible(controls.basis_summary)
    assert (config.rest_noise_x, config.rest_noise_y) == (0.001, 0.002)
    assert not window.calibration_button.isHidden()
    choose(window, "unified")
    assert window.calibration_button.isHidden() and not window.calibration_button.isEnabled()
    assert window.controller.engine.config.motion_profile == "precise"


@pytest.mark.parametrize("old_calibration", [[], list(CALIBRATION)])
def test_legacy_wrist_migrates_once_and_keeps_other_settings(make_window, old_calibration):
    original = {
        "pointer_basis": "wrist",
        "motion_profile": "adaptive",
        "wrist_calibration": old_calibration,
        "span": 0.4,
        "stability": 0.014,
        "contact": 0.34,
        "right_contact": 0.20,
        "right": False,
        "drag": False,
        "scroll": True,
        "rest_noise_x": 0.001,
        "rest_noise_y": 0.002,
    }
    window = make_window(original)
    config = window.controller.engine.config
    assert config.pointer_basis == "unified" and config.motion_profile == "precise"
    assert config.wrist_calibration == tuple(old_calibration)
    assert config.span == 0.4 and config.deadband == 0.014
    assert config.touch_ratio == 0.34 and config.right_touch_ratio == 0.20
    assert (
        not window.right.isChecked() and not window.drag.isChecked() and window.scroll.isChecked()
    )
    assert window.tripod_controls.rest_noise_x == 0.001
    assert "已切换兼容定位，无需方向校准，请手动启用鼠标" in window.notice.text()
    saved = json.loads(window.settings_path.read_text())
    assert saved["unified_mode_seen"] is True
    assert saved["pointer_basis"] == "unified" and saved["motion_profile"] == "adaptive"
    assert not window.controller.active and not window.rig.created
    window.close()
    restored = make_window()
    assert restored.controller.engine.config.pointer_basis == "unified"
    assert "已切换兼容定位" not in restored.notice.text()
    choose(restored, "wrist")
    restored.close()
    explicit_wrist = make_window()
    assert explicit_wrist.controller.engine.config.pointer_basis == "wrist"
    assert explicit_wrist.controller.engine.config.wrist_calibration == tuple(old_calibration)
    assert "已切换兼容定位" not in explicit_wrist.notice.text()
    assert not explicit_wrist.controller.active and not explicit_wrist.rig.created


@pytest.mark.parametrize("profile", ["classic", "precise", "adaptive"])
def test_existing_position_configuration_is_not_forced_to_unified(make_window, profile):
    window = make_window({"pointer_basis": "position", "motion_profile": profile})
    assert window.controller.engine.config.pointer_basis == "position"
    assert window.controller.engine.config.motion_profile == profile
    assert window.tripod_controls.motion_profile.isEnabled()
    assert "已切换兼容定位" not in window.notice.text()
    assert json.loads(window.settings_path.read_text())["unified_mode_seen"] is True


def test_explicit_new_wrist_choice_still_requires_calibration(make_window):
    window = make_window(
        {"pointer_basis": "wrist", "motion_profile": "precise", "unified_mode_seen": True}
    )
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    assert window.controller.engine.config.pointer_basis == "wrist"
    assert not window.wrist_calibration_group.isHidden()
    assert not window.tripod_controls.motion_form.isRowVisible(
        window.tripod_controls.motion_profile
    )
    assert window.tripod_controls.motion_form.isRowVisible(
        window.tripod_controls.fixed_motion_profile
    )
    assert not window.live_button.isEnabled()
    assert window.wrist_calibration_button.isEnabled()
    assert not window.rig.created


def test_switching_from_unified_releases_fake_mouse_and_never_reenables(make_window):
    window = make_window()
    window.controller.set_source("camera")
    window.rig.frames()
    window._refresh()
    window.live_button.click()
    assert window.controller.active
    choose(window, "position")
    assert not window.controller.active and not window.live_button.isChecked()
    choose(window, "unified")
    assert not window.controller.active and not window.live_button.isChecked()
    assert window.rig.created == 1
