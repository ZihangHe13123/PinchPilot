"""Rigid synthetic hands exercise geometry/contracts, not real tracking accuracy."""

import math
from dataclasses import replace

import numpy as np
import pytest

from pinchpilot.domain import HandFrame
from pinchpilot.tripod import TripodConfig, TripodEngine, tripod_features
from pinchpilot.tripod_demo import synthetic_tripod
from pinchpilot.wrist import (
    WristCalibration,
    WristMapping,
    palm_rotation,
    rotation_vector,
    validate_wrist_calibration,
)


def rotation(vector):
    v = np.radians(vector)
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.eye(3)
    x, y, z = v / theta
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * k @ k


def hand(t, degrees=(0, 0, 0), shift=(0, 0, 0), scale=1.0, aspect=4 / 3, **kwargs):
    frame = synthetic_tripod(t, **kwargs)
    q = np.array(frame.landmarks) * [frame.aspect, 1, frame.aspect]
    center = q[0].copy()
    q = (q - center) @ rotation(degrees).T * scale + center + shift
    return replace(frame, landmarks=tuple(map(tuple, q / [aspect, 1, aspect])), aspect=aspect)


def calibration(right=15, up=15):
    return tuple(map(float, palm_rotation(hand(0)).ravel())) + (
        0.0,
        0.0,
        math.radians(right),
        math.radians(up),
        0.0,
        0.0,
    )


class WristSequence:
    def __init__(self, **config):
        self.engine = TripodEngine(
            TripodConfig(pointer_basis="wrist", wrist_calibration=calibration(), **config)
        )
        self.t = 0
        self.events = []
        self.result = self.engine.last_result

    def frames(self, n=1, **kwargs):
        for _ in range(n):
            self.t += 1 / 30
            self.result = self.engine.process(hand(self.t, **kwargs))
            self.events.extend(self.result.events)
        return self.result

    def ramp(self, target, n=30, **kwargs):
        for angle in np.linspace(0, target, n):
            self.frames(degrees=(0, 0, angle), **kwargs)


def test_geometry_and_grip_invariant_under_translation_scale_and_image_aspect():
    original = hand(0, degrees=(25, 10, -12))
    expected = palm_rotation(original)
    f = tripod_features(original, spatial_scale=True)
    for scale in (0.5, 0.9, 1.8):
        for aspect in (1.0, 4 / 3, 16 / 9):
            frame = hand(0, (25, 10, -12), (0.04, -0.03, 0.15), scale, aspect)
            assert palm_rotation(frame) == pytest.approx(expected)
            actual = tripod_features(frame, spatial_scale=True)
            assert (actual.grip, actual.contact, actual.right_contact) == pytest.approx(
                (f.grip, f.contact, f.right_contact)
            )


def test_rotation_no_longer_changes_grip_ratio_from_foreshortening():
    values = []
    for angle in (0, 20, 40, 60):
        frame = hand(0, (angle, 0, 0))
        assert tripod_features(frame, spatial_scale=True).grip == pytest.approx(0.1)
        values.append(tripod_features(frame).grip)
    assert values[-1] > values[0] * 1.25  # The retained baseline has this sensitivity.


def test_direction_calibration_does_not_turn_small_demonstrations_into_higher_gain():
    anchor = palm_rotation(hand(0))
    pose = palm_rotation(hand(0, (0, 0, 11.25)))
    for amplitude in (9, 15, 30):
        mapped = WristMapping(calibration(amplitude, amplitude))
        assert mapped.displacement(pose, anchor) == pytest.approx((0.075, 0), abs=1e-12)


@pytest.mark.parametrize("profile", ["classic", "precise", "adaptive"])
def test_wrist_uses_fixed_gain_and_holding_rotation_does_not_keep_moving(profile):
    endpoints = []
    for n in (10, 60):
        s = WristSequence(motion_profile=profile)
        s.frames(10)
        s.ramp(11.25, n=n)
        s.frames(45, degrees=(0, 0, 11.25))
        assert s.result.state == "CONTROL"
        assert s.result.raw_pointer == pytest.approx((0.75, 0.5))
        endpoint = s.result.pointer
        s.frames(180, degrees=(0, 0, 11.25))
        assert s.result.pointer == pytest.approx(endpoint, abs=1e-6)
        endpoints.append(endpoint)
    assert endpoints[0] == pytest.approx(endpoints[1], abs=2 / 1920)


def test_up_axis_and_negative_angles_are_directional():
    anchor = palm_rotation(hand(0))
    mapping = WristMapping(calibration())
    for vector, point in [
        ((0, 0, -9), (-0.06, 0)),
        ((9, 0, 0), (0, -0.06)),
        ((-9, 0, 0), (0, 0.06)),
    ]:
        assert mapping.displacement(palm_rotation(hand(0, vector)), anchor) == pytest.approx(point)


def test_small_finger_motion_and_whole_hand_translation_do_not_move_wrist_pointer():
    s = WristSequence()
    s.frames(10)
    for i in range(90):
        result = s.frames(x=0.5 + 0.02 * math.sin(i), shift=(0.025 * math.sin(i / 30), 0, 0))
        assert result.state == "CONTROL"
        assert result.pointer == pytest.approx((0.5, 0.5))


def test_click_freezes_and_clutch_reanchors_drag_without_new_button_press():
    s = WristSequence()
    s.frames(10)
    s.ramp(8)
    anchor = s.result.pointer
    s.frames(16, degrees=(0, 0, 8), contact=0.1)
    assert s.engine.left_down and s.result.state == "DRAG"
    assert s.result.pointer == anchor
    for angle in np.linspace(8, 16, 20):
        s.frames(degrees=(0, 0, angle), contact=0.1)
    drop = s.result.pointer
    assert drop[0] > anchor[0] + 0.15
    s.frames(5, degrees=(0, 0, 16), contact=0.1, grip=False)
    for angle in np.linspace(16, 0, 20):
        s.frames(degrees=(0, 0, angle), contact=0.1, grip=False)
    assert s.engine.left_down and s.result.pointer == drop
    s.frames(10, contact=0.1)
    assert s.result.pointer == drop and s.engine.motion_engaged
    s.ramp(5, contact=0.1)
    assert s.result.pointer[0] > drop[0] + 0.08
    s.frames(6, degrees=(0, 0, 5))
    assert [e.kind for e in s.events if e.kind != "move"] == ["down", "up"]


def test_screen_edge_reversal_responds_without_unwinding_hidden_travel():
    s = WristSequence()
    s.frames(10)
    s.ramp(40, n=80)
    s.frames(30, degrees=(0, 0, 40))
    assert s.result.pointer[0] == 1
    for angle in np.linspace(40, 35, 15):
        s.frames(degrees=(0, 0, angle))
    assert s.result.pointer[0] < 0.95


def test_stationary_rotation_noise_is_filtered_but_small_intentional_correction_survives():
    s = WristSequence()
    s.frames(10)
    raw, output = [], []
    for angle in np.random.default_rng(72).normal(0, 0.06, 300):
        s.frames(degrees=(0, 0, angle))
        assert s.result.state == "CONTROL"
        raw.append(s.result.raw_pointer[0])
        output.append(s.result.pointer[0])
    assert np.std(output) < np.std(raw) * 0.5
    s.frames(30)
    before = s.result.pointer[0]
    s.ramp(0.225, n=12)  # About 9.6 logical px on the synthetic 1920-wide screen.
    s.frames(45, degrees=(0, 0, 0.225))
    assert s.result.pointer[0] > before + 5 / 1920


@pytest.mark.parametrize("grip", [False, True])
def test_rotated_hand_can_click_right_and_frozen_left_without_pointer_kick(grip):
    s = WristSequence()
    s.frames(10, degrees=(20, 0, 0))
    anchor = s.result.pointer
    s.frames(8, degrees=(20, 0, 0), grip=grip)
    s.frames(8, degrees=(20, 0, 0), grip=grip, right_contact=0.1)
    s.frames(8, degrees=(20, 0, 0), grip=grip)
    s.frames(4, degrees=(20, 0, 0), grip=grip, contact=0.1)
    s.frames(8, degrees=(20, 0, 0), grip=grip)
    assert [e.kind for e in s.events if e.kind != "move"] == [
        "right_down",
        "right_up",
        "down",
        "up",
    ]
    assert s.result.pointer == anchor


def test_existing_v_scroll_remains_available_in_wrist_mode():
    from test_tripod_scroll import ScrollSequence

    s = ScrollSequence(TripodConfig(pointer_basis="wrist", wrist_calibration=calibration()))
    s.begin()
    assert s.wheel()
    anchor = s.engine.pointer
    s.v(40, y=0.42)
    s.events.clear()
    s.v(30, y=0.42)
    assert not s.events and s.engine.pointer == anchor


@pytest.mark.parametrize(
    "failure", ["missing", "pose_jump", "degenerate", "out_of_range", "swap", "timeout"]
)
def test_drag_failure_releases_once_and_cannot_repress_while_fingers_stay_closed(failure):
    s = WristSequence()
    s.frames(10)
    s.frames(16, contact=0.1)
    assert s.engine.left_down
    t = s.t + 1 / 30
    frame = hand(t, contact=0.1)
    if failure == "missing":
        frame = HandFrame(t)
    elif failure == "pose_jump":
        frame = hand(t, (0, 0, 40), contact=0.1)
    elif failure == "degenerate":
        p = np.array(frame.landmarks)
        p[[0, 5, 9, 13, 17], 0] = 0.5
        frame = replace(frame, landmarks=tuple(map(tuple, p)))
    elif failure == "out_of_range":
        frame = hand(t, (0, 0, 75), contact=0.1)
    elif failure == "swap":
        frame = replace(frame, handedness="Left")
    else:
        frame = replace(frame, timestamp=t + 0.4)
    result = s.engine.process(frame)
    assert result.cancelled and [e.kind for e in result.events] == ["up"]
    assert not s.engine.left_down
    s.t = frame.timestamp
    s.frames(30, contact=0.1)
    assert s.result.state == "WAIT_GRIP" and not s.engine.left_down
    assert not [e for e in s.result.events if e.kind == "up"]


def test_uncalibrated_mode_has_no_events_even_with_valid_hand():
    engine = TripodEngine(TripodConfig(pointer_basis="wrist"))
    for i in range(100):
        result = engine.process(hand(i / 30))
        assert not result.events and result.state == "WAIT_GRIP"
        assert "校准" in result.hint
    assert engine.active_box is None


def make_capture(poses=((0, 0, 0), (0, 0, 15), (15, 0, 0)), jitter=0, missing=False, swapped=False):
    capture = WristCalibration(0, TripodConfig(pointer_basis="wrist"))
    for i in range(360):
        t = (i + 0.1) / 30
        stage = min(2, int(t / 4))
        pose = np.array(poses[stage]) + np.array([0, 0, jitter * (-1) ** i])
        frame = hand(t, pose)
        if missing and i % 2:
            frame = HandFrame(t)
        if swapped and stage == 1:
            frame = replace(frame, handedness="Left")
        capture.add(frame)
    return capture


def test_calibration_uses_only_three_stable_supported_poses():
    capture = make_capture()
    assert capture.result() == pytest.approx(calibration())
    assert capture.progress(6) == 0.5 and capture.progress(15) == 1
    assert "回中立" in capture.hint(9)
    assert "采集" in capture.hint(3.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"poses": ((0, 0, 0), (0, 0, 3), (15, 0, 0))},
        {"poses": ((0, 0, 0), (0, 0, 40), (15, 0, 0))},
        {"poses": ((0, 0, 0), (0, 0, 15), (2, 0, 15))},
        {"jitter": 5},
        {"missing": True},
        {"swapped": True},
    ],
)
def test_bad_calibration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_capture(**kwargs).result()


@pytest.mark.parametrize("values", [None, [0] * 15, [True] * 15, [0] * 14, [float("nan")] * 15])
def test_invalid_saved_calibration_is_rejected(values):
    with pytest.raises(ValueError):
        validate_wrist_calibration(values)


def test_rotation_log_handles_zero_small_and_half_turn():
    assert rotation_vector(np.eye(3)) == pytest.approx((0, 0, 0))
    assert rotation_vector(rotation((1e-7, 0, 0))) == pytest.approx((math.radians(1e-7), 0, 0))
    assert np.linalg.norm(rotation_vector(rotation((180, 0, 0)))) == pytest.approx(math.pi)
