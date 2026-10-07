from dataclasses import replace

import pytest

from pinchpilot.demo import synthetic_hand
from pinchpilot.hand_selection import ControlHandSelector
from pinchpilot.tripod import TripodEngine
from pinchpilot.tripod_demo import synthetic_tripod


def hand(t, side="Right", x=0.35, score=0.99):
    return replace(synthetic_hand(t, x=x), handedness=side, handedness_score=score)


def choose(selector, t, *hands):
    return selector.select(list(hands), t, 4 / 3)


@pytest.mark.parametrize("side,other", [("Right", "Left"), ("Left", "Right")])
def test_lock_survives_result_order_and_other_hand_confidence(side, other):
    selector = ControlHandSelector()
    for i in range(3):
        t = 1 + i / 30
        result = choose(selector, t, hand(t, side, score=0.8))
        assert bool(result.landmarks) == (i == 2)
    for i in range(3, 35):
        t = 1 + i / 30
        current, distractor = hand(t, side, score=0.8), hand(t, other, x=0.7)
        candidates = [current, distractor] if i % 2 else [distractor, current]
        assert choose(selector, t, *candidates) == current
    assert selector.locked_hand == side


def test_lost_control_hand_does_not_unlock_even_after_long_absence_and_needs_confirmation():
    selector = ControlHandSelector()
    for i in range(3):
        choose(selector, i / 30, hand(i / 30))
    for i in range(3, 100):
        t = i / 30
        assert not choose(selector, t, hand(t, "Left", x=0.35)).landmarks
    assert selector.locked_hand == "Right" and "等待右手" in selector.status
    for i in range(100, 103):
        t = i / 30
        result = choose(selector, t, hand(t, x=0.65))
        assert bool(result.landmarks) == (i == 102)


def test_initial_two_hands_require_a_choice_and_explicit_side_can_acquire():
    automatic, explicit = ControlHandSelector(), ControlHandSelector("Left")
    for i in range(4):
        t = i / 30
        right, left = hand(t), hand(t, "Left", x=0.7)
        assert not choose(automatic, t, right, left).landmarks
        selected = choose(explicit, t, right, left)
    assert automatic.locked_hand == "" and "指定" in automatic.status
    assert selected == left
    # A fresh camera session can deliberately choose the other hand.
    for i in range(3):
        t = i / 30
        choose(automatic, t, hand(t, "Left", x=0.7))
    assert automatic.locked_hand == "Left"


@pytest.mark.parametrize(
    "problem",
    [
        "low_score",
        "flipped_label",
        "same_label",
        "overlap",
        "gap",
        "jump",
        "missing",
        "nan",
        "repeated_stamp",
    ],
)
def test_uncertain_tracking_outputs_no_old_landmarks_and_reacquires(problem):
    selector = ControlHandSelector("Right")
    for i in range(3):
        choose(selector, i / 30, hand(i / 30))
    # A coarse clock can hand a new frame the stamp of the previous one.
    t = {"gap": 1.0, "repeated_stamp": 2 / 30}.get(problem, 0.10)
    current = hand(t)
    candidates = [current]
    if problem == "low_score":
        candidates = [replace(current, handedness_score=0.55)]
    elif problem == "flipped_label":
        candidates = [replace(current, handedness="Left")]
    elif problem == "same_label":
        candidates.append(hand(t, x=0.7))
    elif problem == "overlap":
        candidates.append(hand(t, "Left", x=0.37))
    elif problem == "jump":
        candidates = [hand(t, x=0.7)]
    elif problem == "missing":
        candidates = []
    elif problem == "nan":
        candidates = [replace(current, landmarks=((float("nan"), 0, 0),) * 21)]
    assert not choose(selector, t, *candidates).landmarks
    assert selector.locked_hand == "Right"
    for i in range(1, 5):
        result = choose(selector, t + i / 30, hand(t + i / 30))
    assert result.landmarks and result.handedness == "Right"


def test_disappearing_during_press_releases_without_another_hand_clicking():
    selector, engine = ControlHandSelector(), TripodEngine()
    events = []
    t = 10.0

    def frame(n, contact=0.85, other_only=False):
        nonlocal t
        for _ in range(n):
            t += 1 / 30
            control = synthetic_tripod(t, contact=contact)
            other = replace(
                control,
                handedness="Left",
                landmarks=tuple((x + 0.3, y, z) for x, y, z in control.landmarks),
            )
            candidates = [other] if other_only else [control]
            selected = choose(selector, t, *candidates)
            events.extend(engine.process(selected).events)

    frame(12)
    frame(4, contact=0.1)
    assert engine.left_down
    frame(20, other_only=True)
    frame(20, contact=0.1, other_only=True)
    assert not engine.left_down and engine.state == "WAIT_GRIP"
    assert [e.kind for e in events if e.kind != "move"] == ["down", "up"]
    frame(12, contact=0.1)  # Returning with fingers touching cannot click or arm.
    assert engine.state == "WAIT_GRIP"
    frame(12)
    assert engine.state == "CONTROL"
    frame(4, contact=0.1)
    frame(4)
    assert [e.kind for e in events if e.kind != "move"] == ["down", "up", "down", "up"]
