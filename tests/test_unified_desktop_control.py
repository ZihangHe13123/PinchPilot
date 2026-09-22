"""Unified-mode lifecycle contracts with synthetic packets and FakeMouse only."""

from dataclasses import replace

import pytest
from test_desktop_control import Rig
from test_wrist_desktop_control import CALIBRATION, FakeWristCalibration

from pinchpilot.domain import InputEvent


@pytest.fixture
def rig(tmp_path):
    value = Rig(tmp_path)
    value.control.configure(replace(value.control.engine.config, pointer_basis="unified"))
    yield value
    value.output.fail_close = False
    value.output.fail_emit = ""
    value.control.close()


def preview(rig):
    rig.control.set_source("camera")
    rig.frames()


def hold_button(rig):
    rig.control.enable()
    # Exercise release ownership independently of the new motion algorithm.
    rig.control.session.emit([InputEvent("down", *rig.output.point)])
    assert rig.output.down and rig.control.active


def test_unified_preview_outputs_nothing_and_manual_enable_needs_no_wrist_calibration(rig):
    preview(rig)
    rig.frames(8)
    assert rig.control.engine.config.wrist_calibration == ()
    assert not rig.created and not rig.output.events and not rig.control.active
    rig.control.enable()
    assert rig.control.active and rig.created == 1
    assert rig.control.engine.pointer == rig.output.point
    assert rig.control.engine.config.wrist_calibration == ()
    assert not rig.control.calibrating


@pytest.mark.parametrize("source", ["none", "synthetic_demo", "camera"])
def test_unified_enable_still_requires_fresh_real_camera_frames(rig, source):
    rig.control.set_source(source)
    if source != "none":
        rig.frames()
    if source == "camera":
        rig.clock.now += rig.control.FRAME_TIMEOUT + 0.01
    with pytest.raises(RuntimeError):
        rig.control.enable()
    assert not rig.created and not rig.output.events and not rig.control.active


def test_unified_configuration_forces_precise_profile_and_clears_only_position_noise(rig):
    rig.control.configure(
        replace(
            rig.control.engine.config,
            pointer_basis="position",
            motion_profile="adaptive",
            rest_noise_x=0.001,
            rest_noise_y=0.002,
            wrist_calibration=CALIBRATION,
            span=0.4,
            touch_ratio=0.25,
        )
    )
    previous = rig.control.engine.config
    rig.control.configure(replace(previous, pointer_basis="unified"))
    assert rig.control.engine.config == replace(
        previous,
        pointer_basis="unified",
        motion_profile="precise",
        rest_noise_x=0.0,
        rest_noise_y=0.0,
    )
    assert not rig.control.active and not rig.created


@pytest.mark.parametrize("method", ["start_calibration", "start_wrist_calibration"])
def test_unified_rejects_both_legacy_calibration_entrypoints(rig, method):
    preview(rig)
    with pytest.raises(RuntimeError):
        getattr(rig.control, method)()
    assert not rig.control.calibrating
    assert rig.control.calibration is None and rig.control.wrist_capture is None
    assert not rig.control.active and not rig.created and not rig.output.events


@pytest.mark.parametrize("basis", ["unified", "position", "wrist"])
def test_configuration_or_mode_change_releases_and_requires_manual_reenable(rig, basis):
    preview(rig)
    hold_button(rig)
    rig.control.configure(replace(rig.control.engine.config, pointer_basis=basis, span=0.4))
    assert not rig.control.active and not rig.output.down
    assert rig.control.metrics.sent["up"] == 1
    assert rig.control.engine.config.pointer_basis == basis
    before = len(rig.output.events)
    rig.frames(8)
    assert len(rig.output.events) == before and rig.created == 1
    if basis == "unified":
        rig.control.enable()
        assert rig.control.active and rig.created == 2


def test_switching_from_wrist_capture_to_unified_cancels_and_preserves_saved_fit(rig, monkeypatch):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    rig.control.configure(
        replace(
            rig.control.engine.config,
            pointer_basis="wrist",
            wrist_calibration=CALIBRATION,
        )
    )
    preview(rig)
    rig.control.start_wrist_calibration()
    capture = rig.control.wrist_capture
    rig.frames(4)
    assert capture.frames and rig.control.calibrating
    rig.control.configure(replace(rig.control.engine.config, pointer_basis="unified"))
    assert rig.control.wrist_capture is None and not rig.control.calibrating
    assert rig.control.wrist_calibration_result is None and capture.result_calls == 0
    assert rig.control.engine.config.wrist_calibration == CALIBRATION
    assert rig.control.engine.config.motion_profile == "precise"
    assert not rig.control.active and not rig.created and not rig.output.events


def test_failed_release_after_unified_setting_change_blocks_reenable_until_cleanup(rig):
    preview(rig)
    hold_button(rig)
    rig.output.fail_close = True
    rig.control.configure(replace(rig.control.engine.config, span=0.4))
    assert not rig.control.active and rig.control.pending_release and rig.output.down
    with pytest.raises(RuntimeError, match="松键"):
        rig.control.enable()
    assert rig.created == 1
    rig.output.fail_close = False
    rig.control.session.poll()
    rig.control.tick()
    assert not rig.control.pending_release and not rig.output.down
    assert rig.control.metrics.sent["up"] == 1
    assert not rig.control.active


def test_unified_honors_stale_frame_stop_and_releases_held_button(rig):
    preview(rig)
    hold_button(rig)
    rig.clock.now += rig.control.FRAME_TIMEOUT + 0.01
    rig.control.tick()
    assert not rig.control.active and not rig.output.down
    assert rig.control.metrics.sent["up"] == 1
    assert rig.control.session.snapshot()["reason"] == "stale_camera"
