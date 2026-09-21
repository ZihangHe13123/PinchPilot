from dataclasses import replace

import numpy as np
import pytest
from test_tripod import Sequence

from pinchpilot.demo import synthetic_hand
from pinchpilot.domain import HandFrame
from pinchpilot.tripod import TripodConfig, tripod_features


class ScrollSequence(Sequence):
    def v(self, count=1, x=0.5, y=0.5, button=None):
        for _ in range(count):
            frame = synthetic_hand(self.t + 1 / 30, "scroll", x=x, y=y)
            if button:
                points = list(frame.landmarks)
                points[4] = points[8 if button == "left" else 16]
                frame = replace(frame, landmarks=tuple(points))
            self.frame(frame)
        return self.result

    def wheel(self):
        return [e.value for e in self.events if e.kind == "scroll"]

    def begin(self):
        self.frames(10)
        self.v(12)
        for y in np.linspace(0.5, 0.42, 40):
            self.v(y=y)
        assert self.result.state == "SCROLL"


@pytest.mark.parametrize("button", ["left", "right"])
def test_stationary_or_jittering_v_does_not_steal_frozen_clicks(button):
    s = ScrollSequence()
    s.frames(10)
    anchor = s.engine.pointer
    assert tripod_features(synthetic_hand(1, "scroll")).scroll_pose
    for noise in np.random.default_rng(42).normal(0, 0.0015, 90):
        s.v(y=0.5 + noise)
    assert s.result.state == "FROZEN" and not s.wheel()
    s.v(8, button=button)
    s.v(8)
    expected = ["down", "up"] if button == "left" else ["right_down", "right_up"]
    assert [e.kind for e in s.events if e.kind != "move"] == expected
    assert s.engine.pointer == anchor


def test_scroll_is_bidirectional_locks_pointer_and_stops_at_rest():
    s = ScrollSequence()
    s.begin()
    anchor = s.engine.pointer
    assert s.wheel() and all(v > 0 for v in s.wheel())
    s.v(40, y=0.42)
    s.events.clear()
    s.v(30, y=0.42)
    assert not s.events
    for y in np.linspace(0.42, 0.54, 50):
        s.v(y=y)
        assert s.engine.pointer == anchor
        assert all(e.kind == "scroll" for e in s.result.events)
    assert s.wheel() and all(v < 0 for v in s.wheel())


def test_v_can_start_scroll_without_an_extra_pointer_grip():
    s = ScrollSequence()
    s.v(12)
    assert s.result.state == "WAIT_GRIP" and not s.events
    for y in np.linspace(0.5, 0.42, 40):
        s.v(y=y)
    assert s.result.state == "SCROLL" and s.wheel()
    assert not any(e.kind != "scroll" for e in s.events)


@pytest.mark.parametrize("reason", ["pose", "missing", "gap", "swap", "jump", "pause", "reset"])
def test_scroll_ends_on_each_interruption_and_requires_new_intent(reason):
    s = ScrollSequence()
    s.begin()
    s.events.clear()
    if reason == "pose":
        s.frame(synthetic_hand(s.t + 1 / 30, "open", y=0.42))
    elif reason == "missing":
        s.frame(HandFrame(s.t + 1 / 30))
    elif reason == "gap":
        s.frame(synthetic_hand(s.t + 0.3, "scroll", y=0.42))
    elif reason == "swap":
        s.frame(replace(synthetic_hand(s.t + 1 / 30, "scroll", y=0.42), handedness="Left"))
    elif reason == "jump":
        s.v(x=0.8, y=0.42)
    elif reason == "pause":
        s.engine.set_enabled(False)
        s.engine.set_enabled(True)
    else:
        s.engine.reset()
    assert s.engine.state == "WAIT_GRIP" and not s.wheel()
    s.v(20, y=0.42)
    assert not s.wheel()
    s.frames(8, contact=0.1)
    assert not s.clicks()
    s.frames(10)
    assert s.result.state == "CONTROL"
    s.frames(4, contact=0.1)
    assert len(s.clicks()) == 1


def test_drag_and_disabled_scroll_cannot_send_wheel():
    s = ScrollSequence()
    s.frames(10)
    s.frames(18, contact=0.1)
    for y in np.linspace(0.5, 0.42, 40):
        s.v(y=y, button="left")
        assert s.engine.left_down
    assert not s.wheel()
    off = ScrollSequence(replace(TripodConfig(), scroll_enabled=False))
    off.frames(10)
    for y in np.linspace(0.5, 0.42, 40):
        off.v(y=y)
    assert off.result.state == "FROZEN" and not off.wheel()


def test_scroll_speed_is_independent_of_pointer_speed():
    totals = []
    for span, scroll_gain in ((0.15, 60), (3.0, 60), (0.3, 120)):
        s = ScrollSequence(replace(TripodConfig(), span=span, scroll_gain=scroll_gain))
        s.begin()
        totals.append(sum(s.wheel()))
    assert totals[0] == pytest.approx(totals[1])
    assert totals[2] == pytest.approx(2 * totals[0])
