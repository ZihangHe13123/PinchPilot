"""Unified-mode clutch/button safety on generated landmarks, with no OS input."""

from dataclasses import replace

import numpy as np
import pytest
from test_tripod import Sequence
from test_tripod_scroll import ScrollSequence
from test_wrist import hand

from pinchpilot.domain import HandFrame
from pinchpilot.tripod import TripodConfig, tripod_features
from pinchpilot.tripod_demo import synthetic_tripod


def uncertain(frame, ratio=0.50):
    p = np.array(frame.landmarks) * [frame.aspect, 1, frame.aspect]
    scale = (np.linalg.norm(p[5] - p[17]) + np.linalg.norm(p[0] - p[9])) / 2
    direction = p[12] - p[4]
    p[12] = p[4] + direction / np.linalg.norm(direction) * scale * ratio
    return replace(frame, landmarks=tuple(map(tuple, p / [frame.aspect, 1, frame.aspect])))


def sequence():
    s = Sequence(TripodConfig(pointer_basis="unified"))
    s.frames(10)
    assert s.result.state == "CONTROL"
    return s


def test_spatial_grip_does_not_change_left_or_right_contact_thresholds():
    for angle in (0, 20, 40, 60, 77):
        frame = hand(1, (0, angle, 0))
        old = tripod_features(frame)
        new = tripod_features(frame, spatial_grip=True)
        assert new.grip == pytest.approx(0.10)
        assert (new.contact, new.right_contact, new.scale) == (
            old.contact,
            old.right_contact,
            old.scale,
        )


def test_near_threshold_rigid_pinch_survives_gradual_side_turn():
    s = Sequence(TripodConfig(pointer_basis="unified"))
    for angle in [0] * 10 + list(np.linspace(0, 77, 90)):
        s.frame(uncertain(hand(s.t + 1 / 30, (0, angle, 0)), ratio=0.27))
    assert s.result.state == "CONTROL" and s.engine.motion_engaged
    old = tripod_features(uncertain(hand(1, (0, 77, 0)), ratio=0.27))
    assert old.grip > s.engine.config.grip_release  # The old denominator loses this grip.


def test_projected_overlap_does_not_override_estimated_depth_separation():
    s = Sequence(TripodConfig(pointer_basis="unified"))
    for _ in range(15):
        frame = synthetic_tripod(s.t + 1 / 30)
        p = np.array(frame.landmarks)
        scale = tripod_features(frame).scale
        p[12] = p[4] + [0, 0, 0.70 * scale / frame.aspect]
        s.frame(replace(frame, landmarks=tuple(map(tuple, p))))
    assert s.result.state == "WAIT_GRIP" and not s.events


@pytest.mark.parametrize("depth", [0.10, 0.20])
def test_bad_palm_depth_cannot_arm_visibly_open_hand_or_prevent_clutch(depth):
    def open_with_depth(t, x=0.5):
        frame = synthetic_tripod(t, grip=False, x=x)
        points = np.array(frame.landmarks)
        points[[0, 5, 9, 13, 17], 2] += depth * np.array([0, -1, 0.7, -0.5, 1])
        return replace(frame, landmarks=tuple(map(tuple, points)))

    initial = Sequence(TripodConfig(pointer_basis="unified"))
    for _ in range(15):
        initial.frame(open_with_depth(initial.t + 1 / 30))
    assert initial.result.state == "WAIT_GRIP" and not initial.events
    s = sequence()
    before = s.engine.pointer
    s.events.clear()
    s.frame(open_with_depth(s.t + 1 / 30))
    assert s.result.state == "FROZEN" and not s.engine.motion_engaged
    for _ in range(15):
        s.frame(open_with_depth(s.t + 1 / 30, x=0.54))
    assert s.engine.pointer == before and not s.events


def test_uncertain_grip_stops_immediately_and_recovers_without_full_rearm_or_jump():
    s = sequence()
    before = s.engine.pointer
    for x in (0.52, 0.54):
        result = s.frame(uncertain(synthetic_tripod(s.t + 1 / 30, x=x)))
        assert result.pointer == before
        assert not result.events
        assert s.engine.motion_engaged
    s.frames(x=0.56)
    assert s.engine.motion_engaged and s.result.state == "CONTROL"
    assert s.result.pointer == before
    assert s.result.raw_pointer == before
    s.frames(20, x=0.57)
    assert s.result.pointer[0] > before[0]


def test_sustained_uncertainty_becomes_real_clutch_and_clearly_open_is_immediate():
    s = sequence()
    for _ in range(5):
        s.frame(uncertain(synthetic_tripod(s.t + 1 / 30)))
    assert s.result.state == "FROZEN" and not s.engine.motion_engaged
    s.frames()
    assert s.result.state == "FROZEN"  # A real release requires normal rearming.
    s.frames(8)
    assert s.result.state == "CONTROL"
    s.frames(grip=False)
    assert s.result.state == "FROZEN" and not s.engine.motion_engaged


@pytest.mark.parametrize("button", ["left", "right"])
@pytest.mark.parametrize("uncertain_frames", [2, 3])
def test_contact_during_grip_uncertainty_cannot_accumulate_a_new_click(button, uncertain_frames):
    s = sequence()
    touch = {"contact" if button == "left" else "right_contact": 0.1}
    s.events.clear()
    for _ in range(uncertain_frames):
        s.frame(uncertain(synthetic_tripod(s.t + 1 / 30, **touch)))
    s.frames(10, **touch)  # Recovering the grip must not complete that contact.
    assert s.result.state == "WAIT_CLEAR"
    assert not [e for e in s.events if e.kind != "move"]
    assert not s.engine.left_down
    s.frames(8)
    s.frames(8, **touch)
    expected = ["down"] if button == "left" else ["right_down", "right_up"]
    assert [e.kind for e in s.events if e.kind != "move"] == expected


def test_short_and_double_clicks_right_click_and_drag_reclutch():
    s = sequence()
    for _ in range(2):
        s.frames(4, contact=0.1)
        s.frames(4)
    s.frames(4)  # Let the independent right-click clear gate rearm after left-up.
    s.frames(8, right_contact=0.1)
    s.frames(8)
    assert [e.kind for e in s.events if e.kind != "move"] == [
        "down",
        "up",
        "down",
        "up",
        "right_down",
        "right_up",
    ]
    s.events.clear()
    s.frames(18, contact=0.1)
    assert s.engine.left_down and s.result.state == "DRAG"
    s.frames(20, x=0.53, contact=0.1)
    anchor = s.engine.pointer
    s.frames(contact=0.1, grip=False)
    s.frames(10, x=0.48, contact=0.1, grip=False)
    assert s.engine.pointer == anchor and s.engine.left_down
    s.frames(10, x=0.48, contact=0.1)
    assert s.engine.pointer == anchor and s.engine.motion_engaged
    s.frames(20, x=0.49, contact=0.1)
    assert s.engine.pointer[0] > anchor[0]
    s.frames(4, x=0.49)
    assert [e.kind for e in s.events if e.kind != "move"] == ["down", "up"]


@pytest.mark.parametrize("failure", ["missing", "timeout", "swap", "jump", "badtime", "pause"])
def test_interruption_releases_drag_once_and_never_clicks_on_return(failure):
    s = sequence()
    s.frames(18, contact=0.1)
    assert s.engine.left_down
    if failure == "missing":
        result = s.frame(HandFrame(s.t + 1 / 30))
    elif failure == "timeout":
        result = s.engine.tick(s.t + 1)
    elif failure == "swap":
        result = s.frame(replace(synthetic_tripod(s.t + 1 / 30), handedness="Left"))
    elif failure == "jump":
        result = s.frame(hand(s.t + 1 / 30, shift=(0.3, 0, 0)))
    elif failure == "badtime":
        result = s.frame(synthetic_tripod(s.t))
    else:
        result = s.engine.set_enabled(False)
        s.engine.set_enabled(True)
    assert [e.kind for e in result.events] == ["up"]
    assert not s.engine.left_down
    s.events.clear()
    s.frames(15, contact=0.1)
    assert not s.events


def test_releasing_index_during_uncertain_grip_still_releases_held_button():
    s = sequence()
    s.frames(18, contact=0.1)
    before = s.engine.pointer
    s.events.clear()
    for _ in range(3):
        s.frame(uncertain(synthetic_tripod(s.t + 1 / 30, x=0.51)))
    assert s.engine.pointer == before
    assert [e.kind for e in s.events if e.kind != "move"] == ["up"]
    assert not s.engine.left_down


def test_v_scroll_and_stop_keep_the_existing_gesture():
    s = ScrollSequence(TripodConfig(pointer_basis="unified"))
    s.begin()
    assert s.wheel() and all(value > 0 for value in s.wheel())
    anchor = s.engine.pointer
    s.v(40, y=0.42)
    s.events.clear()
    s.v(30, y=0.42)
    assert not s.events and s.engine.pointer == anchor
    s.frame(HandFrame(s.t + 1 / 30))
    assert s.result.state == "WAIT_GRIP"
