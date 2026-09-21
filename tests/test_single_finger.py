from dataclasses import replace

import numpy as np
import pytest

from pinchpilot.demo import synthetic_finger
from pinchpilot.domain import HandFrame
from pinchpilot.single_finger import FingerConfig, SingleFingerEngine, finger_features


class Sequence:
    def __init__(self, mode="finger-flex"):
        self.engine = SingleFingerEngine(FingerConfig(mode=mode))
        self.t = 100.0
        self.events = []

    def frames(self, n=1, **kwargs):
        for _ in range(n):
            self.t += 1 / 30
            self.result = self.engine.process(synthetic_finger(self.t, **kwargs))
            self.events.extend(self.result.events)
        return self.result

    def clicks(self):
        return [e for e in self.events if e.kind == "down"]


def test_relative_point_and_bend_invariant_to_translation_scale_and_aspect():
    frame = synthetic_finger(1, dx=0.12, dy=0.08, bend=0.16)
    baseline = finger_features(frame)
    assert baseline.bend == pytest.approx(0.16)
    for factor, aspect in [(0.65, 4 / 3), (1.3, 16 / 9)]:
        p = np.array(frame.landmarks)
        p = (p - [0.5, 0.5, 0]) * factor + [0.6, 0.55, 0.04]
        p[:, [0, 2]] *= frame.aspect / aspect
        actual = finger_features(replace(frame, landmarks=tuple(map(tuple, p)), aspect=aspect))
        assert actual.relative == pytest.approx(baseline.relative)
        assert actual.bend == pytest.approx(baseline.bend)


def test_degenerate_and_missing_landmarks_are_rejected():
    for frame in (
        HandFrame(1),
        HandFrame(1, ((0.0, 0.0, 0.0),) * 21),
        HandFrame(1, ((float("nan"), 0.0, 0.0),) * 21),
    ):
        assert finger_features(frame) is None


def test_small_finger_motion_points_while_palm_translation_does_not():
    s = Sequence()
    s.frames(10)
    assert s.engine.pointer == pytest.approx((0.5, 0.5))
    s.frames(30, x=0.6, y=0.55)
    assert s.engine.pointer == pytest.approx((0.5, 0.5))
    s.frames(45, dx=0.08, dy=-0.06, x=0.6, y=0.55)
    assert s.engine.pointer == pytest.approx((0.7, 0.35), abs=1e-5)
    assert not s.clicks()


def test_complete_flex_click_freezes_and_reanchors_cursor():
    s = Sequence()
    s.frames(10)
    s.frames(35, dx=0.08)
    before = s.engine.pointer
    s.frames(8, dx=0.08, bend=0.18)
    assert s.result.state == "FLEX_HOLD" and not s.clicks()
    assert s.engine.pointer == before
    s.frames(6, dx=0.08)
    assert [e.kind for e in s.events if e.kind != "move"] == ["down", "up"]
    assert s.clicks()[0].x == before[0]
    s.frames(20, dx=0.08)
    assert s.engine.pointer == before
    assert len(s.clicks()) == 1


@pytest.mark.parametrize("cancel", ["noise", "held", "missing", "deep", "pause", "timeout", "swap"])
def test_incomplete_or_interrupted_flex_never_clicks(cancel):
    s = Sequence()
    s.frames(10)
    s.frames(1 if cancel == "noise" else 6, bend=0.18)
    if cancel == "held":
        s.frames(60, bend=0.18)
    elif cancel == "missing":
        s.engine.process(HandFrame(s.t + 0.01))
    elif cancel == "deep":
        s.frames(1, bend=0.70)
    elif cancel == "pause":
        s.engine.set_enabled(False)
        s.engine.set_enabled(True)
    elif cancel == "timeout":
        s.t += 0.4
    elif cancel == "swap":
        s.t += 1 / 30
        s.engine.process(replace(synthetic_finger(s.t, bend=0.18), handedness="Left"))
    s.frames(12)
    assert not s.clicks()
    assert not any(e.kind == "up" for e in s.events)


def test_flex_requires_continuous_bend_confirmation_and_a_second_cycle():
    s = Sequence()
    s.frames(10)
    for _ in range(5):
        s.frames(1, bend=0.18)
        s.frames(1, bend=0.07)
    s.frames(5)
    assert not s.clicks()
    for _ in range(2):
        s.frames(5, bend=0.18)
        s.frames(6)
    assert len(s.clicks()) == 2


def test_dwell_requires_motion_then_clicks_once_and_rearms_after_leaving():
    s = Sequence("finger-dwell")
    s.frames(80)
    assert s.result.state == "MOVE_TO_ARM" and not s.clicks()
    s.frames(8, dx=0.08)
    assert 0 < s.result.progress < 1
    s.frames(55, dx=0.08)
    assert len(s.clicks()) == 1 and s.result.state == "DWELL_LOCKED"
    for _ in range(12):
        s.frames(1, dx=0.084)
        s.frames(1, dx=0.076)
    assert len(s.clicks()) == 1
    s.frames(40, dx=-0.08)
    assert len(s.clicks()) == 2
    assert [e.kind for e in s.events if e.kind != "move"] == ["down", "up"] * 2


def test_dwell_move_cancel_dropouts_and_return_do_not_complete_old_countdown():
    s = Sequence("finger-dwell")
    s.frames(10)
    s.frames(18, dx=0.08)
    s.frames(10, dx=-0.08)
    assert not s.clicks()
    s.engine.process(HandFrame(s.t + 0.01))
    s.frames(50, dx=-0.08)
    assert not s.clicks()
    assert s.result.state == "MOVE_TO_ARM"


def test_dwell_slow_drift_uses_fixed_anchor_not_frame_to_frame_motion():
    s = Sequence("finger-dwell")
    s.frames(10)
    for i in range(100):
        s.frames(1, dx=i * 0.0015)
    assert not s.clicks()


def test_dwell_offscreen_motion_cannot_click_at_clamped_edge():
    s = Sequence("finger-dwell")
    s.frames(10)
    s.frames(80, dx=0.25)
    assert s.engine.pointer[0] == 1
    assert not s.clicks()
    s.frames(40, dx=0.15)
    assert len(s.clicks()) == 1


def test_tick_preserves_progress_but_never_replays_or_advances_click():
    s = Sequence("finger-dwell")
    s.frames(10)
    s.frames(15, dx=0.08)
    observed = s.result.progress
    idle = s.engine.tick(s.t + 0.01)
    assert idle.progress == observed and not idle.events
    s.frames(15, dx=0.08)
    assert len(s.clicks()) == 1
    assert not s.engine.tick(s.t + 0.01).events
    assert s.engine.tick(s.t + 0.3).state == "WAIT_FINGER"


@pytest.mark.parametrize("mode", ["finger-flex", "finger-dwell"])
def test_reacquisition_preserves_pointer_and_does_not_fire(mode):
    s = Sequence(mode)
    s.frames(10)
    s.frames(8, dx=0.1)
    before = s.engine.pointer
    s.engine.process(HandFrame(s.t + 0.01))
    s.frames(12, dx=-0.1, x=0.4)
    assert s.engine.pointer == pytest.approx(before)
    assert not s.clicks()


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 1.0])
def test_bad_timestamps_cancel_pending_action(timestamp):
    s = Sequence()
    s.frames(10)
    s.frames(5, bend=0.18)
    assert not s.engine.process(synthetic_finger(timestamp)).events
    s.frames(10)
    assert not s.clicks()


@pytest.mark.parametrize(
    "change",
    [
        dict(gain=0),
        dict(flex_delta=float("nan")),
        dict(dwell_seconds=0.1),
        dict(rearm_radius=0.01),
        dict(mode="pinch"),
    ],
)
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        SingleFingerEngine(replace(FingerConfig(), **change))
