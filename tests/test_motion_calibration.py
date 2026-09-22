from dataclasses import replace

import numpy as np
import pytest
from test_desktop_control import Rig

from pinchpilot.domain import HandFrame
from pinchpilot.motion_calibration import RestNoiseCalibration
from pinchpilot.tripod import TripodConfig
from pinchpilot.tripod_demo import synthetic_tripod


def measurement(kind="stationary"):
    config = TripodConfig(motion_profile="precise")
    fit = RestNoiseCalibration(100, config)
    rng = np.random.default_rng(17)
    for i in range(181):
        t = 100 + i / 30
        if kind == "moving":
            frame = synthetic_tripod(t, x=0.5 + i * 0.0004)
        elif kind == "slow_drift":
            # Only 16 logical px over the five-second recording; still movement.
            frame = synthetic_tripod(t, x=0.5 + i / 30 * 0.0005)
        elif kind == "missing":
            frame = HandFrame(t) if i % 3 else synthetic_tripod(t)
        elif kind == "touching":
            frame = synthetic_tripod(t, contact=0.1)
        else:
            frame = synthetic_tripod(t, noise=tuple(rng.normal(0, 0.0004, 2)))
        fit.add(frame)
    return fit


def test_statistical_calibration_reports_noise_and_no_recorded_landmarks():
    result = measurement().result()
    assert 0 < result["rest_noise_x"] < 0.002
    assert 0 < result["rest_noise_y"] < 0.002
    assert 1 < result["noise_px"] < 6
    assert result["samples"] >= 140
    assert "samples" in result and "landmarks" not in result and "trajectory" not in result


@pytest.mark.parametrize("kind", ["moving", "slow_drift", "missing", "touching"])
def test_bad_calibration_is_rejected_instead_of_using_movement_as_noise(kind):
    with pytest.raises(ValueError):
        measurement(kind).result()


@pytest.fixture
def rig(tmp_path):
    rig = Rig(tmp_path)
    rig.control.configure(
        replace(rig.control.engine.config, motion_profile="precise", rest_noise_x=0.0002)
    )
    yield rig
    rig.output.fail_close = False
    rig.control.close()


def test_calibration_stops_held_button_and_cannot_enable_or_emit_during_sampling(rig):
    rig.live()
    rig.frames(4, contact=0.1)
    assert rig.output.down
    rig.control.start_calibration()
    assert rig.control.calibrating and not rig.control.active and not rig.output.down
    before = len(rig.output.events)
    with pytest.raises(RuntimeError, match="校准期间"):
        rig.control.enable()
    assert rig.control.calibrating
    rig.frames(185)
    assert not rig.control.calibrating and not rig.control.active
    assert len(rig.output.events) == before
    assert rig.control.calibration_result["samples"] >= 140
    assert rig.control.engine.config.rest_noise_x == pytest.approx(0)


@pytest.mark.parametrize("reason", ["stop", "source", "config", "stale", "close"])
def test_interruptions_cancel_and_preserve_previous_calibration(rig, reason):
    rig.live()
    rig.control.start_calibration()
    rig.frames(60)
    assert 0 < rig.control.calibration_progress < 1
    if reason == "stop":
        rig.control.stop()
    elif reason == "source":
        rig.control.set_source("synthetic_demo")
    elif reason == "config":
        rig.control.configure(replace(rig.control.engine.config, span=0.4))
    elif reason == "stale":
        rig.clock.now += 0.3
        rig.control.tick()
    else:
        rig.control.close()
    assert not rig.control.calibrating and not rig.control.active
    assert rig.control.calibration_result is None
    assert rig.control.engine.config.rest_noise_x == 0.0002


def test_failure_does_not_replace_previous_noise_or_restore_mouse_control(rig):
    rig.live()
    rig.control.start_calibration()
    rig.frames(185, contact=0.1)
    assert not rig.control.calibrating and not rig.control.active
    assert rig.control.calibration_result is None
    assert "未应用" in rig.control.notice
    assert rig.control.engine.config.rest_noise_x == 0.0002


def test_calibration_cannot_start_from_demo_or_without_successful_release(rig):
    rig.control.set_source("synthetic_demo")
    rig.frames(8)
    with pytest.raises(RuntimeError):
        rig.control.start_calibration()
    rig.live()
    rig.frames(4, contact=0.1)
    rig.output.fail_close = True
    with pytest.raises(RuntimeError, match="松键"):
        rig.control.start_calibration()
    assert not rig.control.calibrating and rig.control.pending_release
