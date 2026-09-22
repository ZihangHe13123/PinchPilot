from dataclasses import replace

import numpy as np
import pytest
from test_tripod import Sequence

from pinchpilot.domain import HandFrame
from pinchpilot.motion import TunedPointer, tuning_radius
from pinchpilot.tripod import TripodConfig


def test_precise_small_corrections_are_not_swallowed_and_units_match_axes():
    cfg = TripodConfig(motion_profile="precise")
    displacements = []
    for axis, size in enumerate((cfg.screen_width, cfg.screen_height)):
        pointer = TunedPointer(cfg)
        pointer.update((0.5, 0.5), 0)
        target = [0.5, 0.5]
        target[axis] += 10 / size
        for i in range(1, 31):
            point = pointer.update(target, i / 30)
        displacements.append((point[axis] - 0.5) * size)
        assert abs(point[1 - axis] - 0.5) < 1e-10
    assert displacements == pytest.approx([8, 8], abs=0.01)


@pytest.mark.parametrize("profile", ["precise", "adaptive"])
@pytest.mark.parametrize(
    "axis,direction",
    [
        pytest.param(0, -1, id="left"),
        pytest.param(0, 1, id="right"),
        pytest.param(1, -1, id="top"),
        pytest.param(1, 1, id="bottom"),
    ],
)
def test_isolated_spike_and_offscreen_travel_do_not_block_edge_reversal(profile, axis, direction):
    pointer = TunedPointer(TripodConfig(motion_profile=profile))

    def position(offset):
        point = [0.5, 0.5]
        point[axis] += direction * offset
        return tuple(point)

    for i in range(60):
        point = pointer.update(position(0.06 if i == 30 else 0), i / 30)
        assert point == pytest.approx((0.5, 0.5))
    for i in range(60, 200):
        point = pointer.update(position((i - 60) * 0.01), i / 30)
        assert all(0 <= value <= 1 for value in point)
    edge = 1 if direction > 0 else 0
    assert point[axis] == edge
    assert point[1 - axis] == pytest.approx(0.5)
    for i in range(200, 215):
        point = pointer.update(position(1.39), i / 30)
        assert point[axis] == edge
    for i in range(215, 230):
        # The input remains far outside the screen after this small return stroke.
        point = pointer.update(position(1.39 - (i - 214) * 0.01), i / 30)
        assert all(0 <= value <= 1 for value in point)
    assert direction * (edge - point[axis]) > 0.05
    assert point[1 - axis] == pytest.approx(0.5)


def test_bounded_speed_gain_gives_more_travel_to_faster_motion():
    def run(frames, fps):
        pointer = TunedPointer(TripodConfig(motion_profile="adaptive"))
        pointer.update((0.5, 0.5), 0)
        for i in range(1, frames + 31):
            point = pointer.update((0.5 + 0.1 * min(i / frames, 1), 0.5), i / fps)
            assert 0.55 <= pointer.gain <= 1.35
        return point[0] - 0.5

    slow, fast = run(60, 30), run(6, 30)
    assert fast > slow * 1.4
    assert run(120, 60) == pytest.approx(slow, abs=0.002)


@pytest.mark.parametrize("profile", ["precise", "adaptive"])
def test_click_freeze_release_reanchor_drag_and_tracking_loss_remain_separate(profile):
    s = Sequence(TripodConfig(motion_profile=profile))
    s.frames(10)
    s.frames(20, x=0.53)
    before = s.engine.pointer
    s.frames(4, x=0.55, contact=0.1)
    assert s.engine.left_down and s.engine.pointer == before
    s.frames(10, x=0.55)
    assert s.engine.pointer == before and not s.engine.left_down
    s.frames(18, x=0.55, contact=0.1)
    assert s.engine.state == "DRAG"
    s.frames(20, x=0.56, contact=0.1)
    assert s.engine.pointer[0] > before[0] and s.engine.left_down
    result = s.frame(HandFrame(s.t + 1 / 30))
    assert [event.kind for event in result.events] == ["up"]
    assert not s.engine.left_down and s.engine.state == "WAIT_GRIP"


def test_calibrated_radius_scales_with_sensitivity_and_is_bounded():
    cfg = TripodConfig(motion_profile="precise", rest_noise_x=0.0005)
    base = tuning_radius(cfg)
    assert tuning_radius(replace(cfg, span=0.6)) < base
    assert tuning_radius(replace(cfg, rest_noise_x=0.02, rest_noise_y=0.02)) == 4.0
    assert tuning_radius(replace(cfg, deadband=0)) == 0


@pytest.mark.parametrize("profile", ["precise", "adaptive"])
def test_standard_calibration_preserves_small_forward_and_reverse_corrections(profile):
    cfg = TripodConfig(motion_profile=profile, rest_noise_x=0.02)
    pointer = TunedPointer(cfg)
    pointer.update((0.5, 0.5), 0)
    for i in range(1, 31):
        forward = pointer.update((0.5 + 9.6 / cfg.screen_width, 0.5), i / 30)
    for i in range(31, 61):
        reverse = pointer.update((0.5, 0.5), i / 30)
    assert (forward[0] - 0.5) * cfg.screen_width > 3
    assert (forward[0] - reverse[0]) * cfg.screen_width > 0.8


@pytest.mark.parametrize("profile", ["precise", "adaptive"])
def test_strong_calibration_suppresses_small_reversal_but_recovers_with_more_travel(profile):
    cfg = TripodConfig(motion_profile=profile, rest_noise_x=0.02, deadband=0.014)
    pointer = TunedPointer(cfg)
    pointer.update((0.5, 0.5), 0)
    for i in range(1, 31):
        forward = pointer.update((0.5 + 9.6 / cfg.screen_width, 0.5), i / 30)
    assert (forward[0] - 0.5) * cfg.screen_width > 1
    for i in range(31, 61):
        small_reverse = pointer.update((0.5, 0.5), i / 30)
    # Strong stability trades away this small reversal; it must not be described
    # as having the standard profile's 4-logical-pixel radius or micro-correction result.
    assert abs(forward[0] - small_reverse[0]) * cfg.screen_width < 1e-6
    for i in range(61, 91):
        larger_reverse = pointer.update((0.5 - 9.6 / cfg.screen_width, 0.5), i / 30)
    assert (forward[0] - larger_reverse[0]) * cfg.screen_width > 2.5
    assert larger_reverse[0] < 0.5


@pytest.mark.parametrize(
    "values",
    [
        {"motion_profile": "unknown"},
        {"rest_noise_x": float("nan")},
        {"rest_noise_y": -0.001},
        {"screen_width": 0},
        {"screen_height": np.inf},
    ],
)
def test_invalid_motion_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        replace(TripodConfig(), **values).validate()
