from dataclasses import replace

import numpy as np
import pytest

from pinchpilot.domain import HandFrame
from pinchpilot.filtering import StablePointer
from pinchpilot.tripod import TripodConfig, TripodEngine, tripod_features
from pinchpilot.tripod_demo import synthetic_tripod


class Sequence:
    def __init__(self, config=None):
        self.engine = TripodEngine(config)
        self.t = 100.0
        self.events = []

    def frames(self, n=1, **kwargs):
        for _ in range(n):
            self.t += 1 / 30
            self.result = self.engine.process(synthetic_tripod(self.t, **kwargs))
            self.events.extend(self.result.events)
        return self.result

    def clicks(self):
        return [e for e in self.events if e.kind == "down"]


def test_index_coordinates_do_not_enter_pointer_and_depth_separates_overlap():
    a = synthetic_tripod(1, contact=0.85)
    b = synthetic_tripod(1, contact=0.10)
    assert tripod_features(a).point == tripod_features(b).point
    p = list(b.landmarks)
    p[8] = (*p[8][:2], 0.15)
    assert tripod_features(replace(b, landmarks=tuple(p))).contact > 0.48
    assert tripod_features(b).contact < 0.26


def test_geometry_ratios_preserved_under_translation_scale_and_aspect():
    frame = synthetic_tripod(1, contact=0.3)
    base = tripod_features(frame)
    p = np.array(frame.landmarks)
    p = (p - [0.5, 0.5, 0]) * 0.7 + [0.6, 0.55, 0.01]
    p[:, [0, 2]] *= frame.aspect / (16 / 9)
    actual = tripod_features(replace(frame, landmarks=tuple(map(tuple, p)), aspect=16 / 9))
    assert (actual.grip, actual.contact) == pytest.approx((base.grip, base.contact))


def test_open_hand_or_initial_three_finger_contact_cannot_arm_or_click():
    s = Sequence()
    s.frames(20, grip=False)
    assert s.result.state == "WAIT_GRIP" and not s.result.events
    s.frames(30, contact=0.1)
    assert s.result.state == "WAIT_GRIP" and not s.clicks()
    s.frames(8)
    assert s.result.state == "CONTROL"


def test_hover_freezes_before_displacement_and_contact_clicks_before_lift():
    s = Sequence()
    s.frames(10)
    s.frames(20, x=0.53)
    anchor = s.engine.pointer
    s.frames(3, x=0.55, contact=0.33)
    assert s.result.state == "APPROACH" and s.engine.pointer == anchor
    s.frames(4, x=0.56, contact=0.12)
    assert s.result.state == "TOUCHED" and len(s.clicks()) == 1
    assert [e.kind for e in s.events if e.kind != "move"] == ["down", "up"]
    s.frames(45, x=0.56, contact=0.12)
    assert len(s.clicks()) == 1 and s.engine.pointer == anchor
    s.frames(8, x=0.56)
    assert s.result.state == "CONTROL" and s.engine.pointer == anchor
    s.frames(20, x=0.56)
    assert s.engine.pointer == anchor
    s.frames(4, x=0.56, contact=0.12)
    assert len(s.clicks()) == 2


@pytest.mark.parametrize(
    "interruption", ["release", "missing", "gap", "swap", "jump", "pause", "badtime"]
)
def test_pending_click_cancelled_and_contact_on_return_does_not_click(interruption):
    s = Sequence()
    s.frames(10)
    s.frames(1, contact=0.1)
    if interruption == "release":
        s.frames(1, grip=False, contact=0.1)
    elif interruption == "missing":
        s.engine.process(HandFrame(s.t + 0.01))
    elif interruption == "gap":
        s.t += 0.4
    elif interruption == "swap":
        s.engine.process(replace(synthetic_tripod(s.t + 0.01, contact=0.1), handedness="Left"))
    elif interruption == "jump":
        s.frames(1, x=0.8, contact=0.1)
    elif interruption == "pause":
        s.engine.set_enabled(False)
        s.engine.set_enabled(True)
    elif interruption == "badtime":
        s.engine.process(synthetic_tripod(float("nan"), contact=0.1))
    s.frames(20, contact=0.1)
    assert not s.clicks() and s.result.state == "WAIT_GRIP"


def test_noise_or_brief_exit_cannot_confirm_click_or_rearm():
    s = Sequence()
    s.frames(10)
    for _ in range(10):
        s.frames(1, contact=0.1)
        s.frames(1, contact=0.32)
    assert not s.clicks()
    s.frames(5, contact=0.1)
    assert len(s.clicks()) == 1
    for _ in range(10):
        s.frames(1)
        s.frames(1, contact=0.1)
    assert len(s.clicks()) == 1


def test_release_and_regrip_keep_cursor_without_jump():
    s = Sequence()
    s.frames(10)
    s.frames(20, x=0.54)
    before = s.engine.pointer
    s.frames(1, grip=False)
    s.frames(20, x=0.45)
    assert s.engine.pointer == before and s.result.state == "CONTROL"


def test_palm_scale_noise_does_not_move_the_pointer():
    s = Sequence()
    s.frames(10)
    for i in range(50):
        s.t += 1 / 30
        frame = synthetic_tripod(s.t)
        p = np.array(frame.landmarks)
        palm = p[[0, 5, 9, 13, 17]]
        p[[0, 5, 9, 13, 17]] = (palm - palm.mean(axis=0)) * (1.05 if i % 2 else 0.95) + palm.mean(
            axis=0
        )
        result = s.engine.process(replace(frame, landmarks=tuple(map(tuple, p))))
        assert result.pointer == (0.5, 0.5)
        assert result.state == "CONTROL"


@pytest.mark.parametrize("arm_frames", [5, 10])
def test_median_rejects_an_isolated_input_spike(arm_frames):
    s = Sequence()
    s.frames(arm_frames)
    s.frames(1, x=0.55)
    s.frames(10)
    assert s.engine.pointer == (0.5, 0.5)


def test_deadband_holds_rest_but_continuous_slow_movement_accumulates():
    p = StablePointer(radius=0.008)
    p.update((0.5, 0.5), 0)
    for i in range(1, 50):
        assert p.update((0.5 + 0.003 * (-1) ** i, 0.5), i / 30) == (0.5, 0.5)
    positions = [p.update((0.5 + i * 0.0005, 0.5), (50 + i) / 30)[0] for i in range(100)]
    assert positions[-1] > 0.535
    assert max(np.diff(positions)) < 0.002  # No snap when crossing the deadband.


def test_generated_rest_noise_is_reduced_and_motion_remains_responsive():
    s = Sequence()
    s.frames(10)
    rng = np.random.default_rng(32)
    raw, filtered = [], []
    for noise in rng.normal(0, 0.002, (240, 2)):
        r = s.frames(1, noise=tuple(noise))
        raw.append(r.raw_pointer)
        filtered.append(r.pointer)
    raw_rms = np.std(raw, axis=0)
    output_rms = np.std(filtered, axis=0)
    assert np.linalg.norm(output_rms) < np.linalg.norm(raw_rms) * 0.5
    assert not s.clicks()
    s.frames(8, x=0.53)
    assert s.engine.pointer[0] > 0.58  # A 10% step reaches most of its distance within ~267 ms.


@pytest.mark.parametrize(
    "change",
    [
        dict(span=0),
        dict(deadband=float("nan")),
        dict(touch_ratio=0.6),
        dict(grip_release=0.1),
        dict(confirm_seconds=0),
        dict(mode="pinch"),
    ],
)
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        TripodEngine(replace(TripodConfig(), **change))


def test_missing_degenerate_landmarks_and_tick_are_safe():
    assert tripod_features(HandFrame(1, ((0.0, 0.0, 0.0),) * 21)) is None
    s = Sequence()
    s.frames(10)
    s.frames(4, contact=0.1)
    assert len(s.clicks()) == 1
    assert not s.engine.tick(s.t + 0.01).events
    assert s.engine.tick(s.t + 0.4).state == "WAIT_GRIP"
