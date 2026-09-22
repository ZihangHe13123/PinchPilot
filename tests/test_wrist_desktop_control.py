"""Wrist calibration lifecycle tests; no camera or native input is used."""

import json
import math
from dataclasses import asdict, replace

import pytest
from test_desktop_control import Rig

from pinchpilot.domain import InputEvent
from pinchpilot.tripod_demo import synthetic_tripod
from pinchpilot.vision import Packet

# Neutral basis, demonstrated horizontal rotation, demonstrated vertical rotation.
CALIBRATION = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    math.radians(15),
    math.radians(15),
    0.0,
    0.0,
)


class FakeWristCalibration:
    """Keep lifecycle tests independent of the real pose-fitting algorithm."""

    DURATION = 12.0

    def __init__(self, started, config):
        self.started = started
        self.config = config
        self.frames = []
        self.failure = None
        self.parameters = CALIBRATION
        self.result_calls = 0

    def add(self, frame):
        self.frames.append(frame)

    def progress(self, now):
        return min(1.0, max(0.0, (now - self.started) / self.DURATION))

    def hint(self, now):
        return f"腕动校准 {self.progress(now):.0%}"

    def result(self):
        self.result_calls += 1
        if self.failure is not None:
            raise ValueError(self.failure)
        return self.parameters


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr("pinchpilot.desktop_control.WristCalibration", FakeWristCalibration)
    value = Rig(tmp_path)
    value.control.configure(replace(value.control.engine.config, pointer_basis="wrist"))
    yield value
    value.output.fail_close = False
    value.output.fail_emit = ""
    value.control.close()


def preview(rig, calibrated=False):
    if calibrated:
        rig.control.configure(replace(rig.control.engine.config, wrist_calibration=CALIBRATION))
    rig.control.set_source("camera")
    rig.frames()


def finish(rig):
    capture = rig.control.wrist_capture
    rig.clock.now = capture.started + capture.DURATION
    rig.frames()
    return capture


def hold_button(rig):
    """Inject only into FakeMouse; gesture geometry belongs to engine tests."""
    rig.control.enable()
    assert rig.control.active
    rig.control.session.emit([InputEvent("down", *rig.output.point)])
    assert rig.output.down


@pytest.mark.parametrize("source", ["none", "synthetic_demo", "camera"])
def test_wrist_capture_requires_fresh_camera_frames(rig, source):
    rig.control.set_source(source)
    if source == "synthetic_demo":
        rig.frames()
    with pytest.raises(RuntimeError):
        rig.control.start_wrist_calibration()
    assert rig.control.wrist_capture is None
    assert not rig.control.calibrating and not rig.created


def test_expired_camera_cannot_start_wrist_calibration(rig):
    preview(rig)
    rig.clock.now += rig.control.FRAME_TIMEOUT + 0.01
    with pytest.raises(RuntimeError):
        rig.control.start_wrist_calibration()
    assert not rig.control.calibrating and not rig.created


def test_uncalibrated_wrist_mode_cannot_create_an_output_session(rig):
    preview(rig)
    with pytest.raises(RuntimeError, match="校准"):
        rig.control.enable()
    assert not rig.control.active and not rig.created and not rig.output.events


def test_capture_receives_live_frames_and_applies_only_the_result(rig):
    preview(rig)
    previous = rig.control.engine.config
    rig.control.start_wrist_calibration()
    capture = rig.control.wrist_capture
    assert isinstance(capture, FakeWristCalibration)
    assert capture.config == previous
    assert rig.control.calibrating and rig.control.calibration is None
    assert rig.control.wrist_calibration_result is None
    assert not rig.control.active

    rig.clock.now += capture.DURATION / 2
    rig.frames()
    assert len(capture.frames) == 1
    assert capture.frames[0].timestamp == rig.clock.now
    assert 0 < rig.control.wrist_calibration_progress < 1
    assert rig.control.calibration_progress == 0
    assert rig.control.engine.config.wrist_calibration == ()
    assert capture.result_calls == 0

    finish(rig)
    assert capture.result_calls == 1
    assert rig.control.wrist_capture is None and not rig.control.calibrating
    assert rig.control.wrist_calibration_result == CALIBRATION
    assert rig.control.engine.config == replace(previous, wrist_calibration=CALIBRATION)
    assert not rig.control.active and not rig.created and not rig.output.events
    rig.frames(4)
    assert capture.result_calls == 1


def test_recalibration_releases_held_button_and_never_resumes_output(rig):
    preview(rig, calibrated=True)
    hold_button(rig)
    rig.control.start_wrist_calibration()
    assert not rig.output.down and not rig.control.active
    assert rig.control.metrics.sent["up"] == 1
    assert rig.control.calibrating
    before = len(rig.output.events)
    with pytest.raises(RuntimeError, match="校准"):
        rig.control.enable()
    assert rig.control.calibrating
    finish(rig)
    assert not rig.control.active and not rig.output.down
    assert len(rig.output.events) == before
    assert rig.created == 1
    rig.control.enable()
    assert rig.control.active and rig.created == 2


def test_release_failure_prevents_calibration_from_starting(rig):
    preview(rig, calibrated=True)
    hold_button(rig)
    rig.output.fail_close = True
    with pytest.raises(RuntimeError, match="松键"):
        rig.control.start_wrist_calibration()
    assert rig.control.pending_release and rig.output.down
    assert not rig.control.active and not rig.control.calibrating
    assert rig.control.wrist_capture is None
    assert rig.control.engine.config.wrist_calibration == CALIBRATION


@pytest.mark.parametrize(
    "reason", ["stop", "source", "config", "mode", "stale", "invalid_frame", "close"]
)
def test_interruptions_cancel_capture_and_keep_prior_parameters(rig, reason):
    preview(rig, calibrated=True)
    rig.control.start_wrist_calibration()
    capture = rig.control.wrist_capture
    rig.frames(4)
    assert 0 < rig.control.wrist_calibration_progress < 1
    if reason == "stop":
        rig.control.stop()
    elif reason == "source":
        rig.control.set_source("synthetic_demo")
    elif reason == "config":
        rig.control.configure(replace(rig.control.engine.config, span=0.4))
    elif reason == "mode":
        rig.control.configure(replace(rig.control.engine.config, pointer_basis="position"))
    elif reason == "stale":
        rig.clock.now += rig.control.FRAME_TIMEOUT + 0.01
        rig.control.tick()
    elif reason == "invalid_frame":
        frame = synthetic_tripod(rig.clock.now)
        assert not rig.control.consume(Packet(None, frame, 8, 30, frame.timestamp))
    else:
        assert rig.control.close()
    assert rig.control.wrist_capture is None and not rig.control.calibrating
    assert rig.control.wrist_calibration_result is None
    assert rig.control.engine.config.wrist_calibration == CALIBRATION
    assert not rig.control.active and not rig.output.events
    assert capture.result_calls == 0


def test_failed_fit_keeps_previous_calibration_and_mouse_disabled(rig):
    preview(rig, calibrated=True)
    previous = rig.control.engine.config
    rig.control.start_wrist_calibration()
    capture = rig.control.wrist_capture
    capture.failure = "舒适角域不足"
    finish(rig)
    assert capture.result_calls == 1
    assert not rig.control.calibrating and not rig.control.active
    assert rig.control.wrist_calibration_result is None
    assert rig.control.engine.config == previous
    assert capture.failure in rig.control.notice
    assert not rig.created and not rig.output.events


@pytest.mark.parametrize("parameters", [(), CALIBRATION[:-1], (float("nan"),) * 15])
def test_invalid_fit_cannot_replace_previous_parameters(rig, parameters):
    preview(rig, calibrated=True)
    previous = rig.control.engine.config
    rig.control.start_wrist_calibration()
    rig.control.wrist_capture.parameters = parameters
    finish(rig)
    assert not rig.control.calibrating and not rig.control.active
    assert rig.control.engine.config == previous
    assert rig.control.wrist_calibration_result is None
    assert "未应用" in rig.control.notice
    assert not rig.created and not rig.output.events


def test_switching_to_wrist_clears_position_noise_incompatible_with_angular_units(rig):
    rig.control.configure(
        replace(
            rig.control.engine.config,
            pointer_basis="position",
            motion_profile="precise",
            rest_noise_x=0.001,
            rest_noise_y=0.002,
        )
    )
    rig.control.configure(replace(rig.control.engine.config, pointer_basis="wrist"))
    assert rig.control.engine.config.rest_noise_x == 0.0
    assert rig.control.engine.config.rest_noise_y == 0.0


def test_wrist_and_position_calibration_entrypoints_are_separate(rig):
    preview(rig)
    with pytest.raises(RuntimeError):
        rig.control.start_calibration()
    assert not rig.control.calibrating
    rig.control.configure(
        replace(rig.control.engine.config, pointer_basis="position", motion_profile="precise")
    )
    with pytest.raises(RuntimeError):
        rig.control.start_wrist_calibration()
    assert not rig.control.calibrating
    rig.control.start_calibration()
    assert rig.control.calibrating and rig.control.calibration is not None
    assert rig.control.wrist_capture is None
    rig.frames(185)
    assert not rig.control.calibrating and not rig.control.active
    assert rig.control.calibration_result["samples"] >= 140
    assert rig.control.wrist_calibration_result is None
    assert rig.control.engine.config.wrist_calibration == ()
    assert not rig.created and not rig.output.events


def test_logging_saves_fitted_parameters_without_capture_frames(rig):
    preview(rig)
    metrics = rig.control.metrics
    metrics.set_logging(True, asdict(rig.control.engine.config), "camera")
    path = metrics.path
    rig.control.start_wrist_calibration()
    capture = rig.control.wrist_capture
    rig.frames(6)
    finish(rig)
    assert capture.frames and rig.control.wrist_calibration_result == CALIBRATION
    metrics.close()
    text = path.read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    fits = [row for row in rows if row["type"] == "wrist_calibration"]
    assert len(fits) == 1
    vectors = [value for value in fits[0].values() if isinstance(value, list)]
    assert vectors == [list(CALIBRATION)]
    for forbidden in ("landmarks", "world_landmarks", "trajectory", '"rgb"', '"frames"'):
        assert forbidden not in text
    assert rows[0]["stores_images"] is False
