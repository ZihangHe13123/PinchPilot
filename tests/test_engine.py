"""Engineering fixtures only: no camera, mouse injection or performance claims."""

from dataclasses import replace

import pytest

from pinchpilot.demo import synthetic_hand
from pinchpilot.domain import EngineConfig, HandFrame, Prediction
from pinchpilot.engine import GestureEngine


class Sequence:
    def __init__(self, engine=None):
        self.engine = engine or GestureEngine()
        self.t = 100.0

    def frames(self, pose, n=5, **kwargs):
        events = []
        for _ in range(n):
            self.t += 1 / 30
            result = self.engine.process(synthetic_hand(self.t, pose, **kwargs))
            events.extend(result.events)
        return events


def pressed():
    seq = Sequence()
    seq.frames("open")
    seq.frames("pinch")
    assert seq.engine.down
    return seq


def test_held_pinch_never_repeats_click():
    seq = Sequence()
    assert not seq.frames("pinch", 12)
    assert seq.engine.state == "WAIT_OPEN"
    seq.frames("open")
    events = seq.frames("pinch", 60) + seq.frames("open", 10)
    assert [e.kind for e in events if e.kind != "move"] == ["down", "up"]
    assert not seq.engine.down


def test_single_frame_pinch_noise_is_not_click():
    seq = Sequence()
    seq.frames("open")
    assert all(e.kind != "down" for e in seq.frames("pinch", 1) + seq.frames("open"))


@pytest.mark.parametrize(
    "action", ["timeout", "pause", "reset", "invalid_time", "fist", "other", "hand_swap"]
)
def test_every_exit_releases_button_and_requires_rearm(action):
    seq = pressed()
    engine = seq.engine
    if action == "timeout":
        result = engine.tick(seq.t + 0.25)
        seq.t += 0.25
    elif action == "pause":
        result = engine.set_enabled(False)
        engine.set_enabled(True)
    elif action == "reset":
        result = engine.reset()
    elif action == "invalid_time":
        result = engine.process(synthetic_hand(seq.t - 1, "pinch"))
    else:
        seq.t += 0.033
        frame = synthetic_hand(seq.t, "fist" if action == "fist" else "pinch")
        if action == "hand_swap":
            frame = replace(frame, handedness="Left")
        result = engine.process(frame, Prediction("other", 0.9) if action == "other" else None)
    assert [e.kind for e in result.events].count("up") == 1
    assert not engine.down
    assert not any(e.kind == "down" for e in seq.frames("pinch"))


def test_stalled_stream_releases_on_first_returning_frame():
    seq = pressed()
    seq.t += 0.4
    result = seq.engine.process(synthetic_hand(seq.t, "pinch"))
    assert [e.kind for e in result.events] == ["up"]
    assert result.state == "WAIT_OPEN"


def test_missing_observation_breaks_press_confirmation():
    seq = Sequence()
    seq.frames("open")
    seq.frames("pinch", 2)
    seq.t += 0.04
    seq.engine.process(HandFrame(seq.t))
    assert not any(e.kind == "down" for e in seq.frames("pinch", 1))
    assert not seq.engine.down


def test_drag_enters_without_position_jump_then_tracks_delta():
    seq = pressed()
    anchor = seq.engine.pointer
    first_drag = None
    for _ in range(20):
        events = seq.frames("pinch", 1, x=0.7)
        if seq.engine.state == "DRAG":
            first_drag = next(e for e in events if e.kind == "move")
            break
    assert first_drag is not None
    assert (first_drag.x, first_drag.y) == pytest.approx(anchor)
    seq.frames("pinch", 20, x=0.72)
    assert seq.engine.pointer[0] > anchor[0]
    assert any(e.kind == "up" for e in seq.frames("open", x=0.72))


def test_scroll_emits_wheel_never_click_and_rearms():
    seq = Sequence()
    seq.frames("open")
    seq.frames("scroll")
    events = seq.frames("scroll", 10, y=0.4)
    assert any(e.kind == "scroll" for e in events)
    assert not any(e.kind in ("down", "up") for e in events)
    assert not any(e.kind == "down" for e in seq.frames("pinch", 20))
    assert seq.engine.state == "WAIT_OPEN"


def test_low_confidence_hold_expires_without_repeated_down():
    seq = pressed()
    events = []
    for _ in range(10):
        seq.t += 0.033
        events.extend(
            seq.engine.process(
                synthetic_hand(seq.t, "pinch"), Prediction("uncertain", 0.4, "forest")
            ).events
        )
    assert [e.kind for e in events] == ["up"]
    assert seq.engine.state == "WAIT_OPEN"


def test_bad_config_is_rejected_before_use():
    with pytest.raises(ValueError):
        GestureEngine(EngineConfig(engage_ratio=0.8, release_ratio=0.4))


def test_pointer_clamped_after_large_continuous_motion():
    seq = Sequence()
    seq.frames("open")
    for x in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
        seq.frames("open", x=x)
    assert seq.engine.pointer[0] == 1.0


def test_rule_threshold_gap_holds_button_until_release_threshold():
    from pinchpilot.features import extract

    seq = pressed()
    frame = synthetic_hand(seq.t, "pinch")
    points = list(frame.landmarks)
    # Set distance to the middle of engage/release thresholds.
    scale = 0.32 / extract(frame).pinch
    thumb, tip = points[4], points[8]
    points[4] = tuple(t + (a - t) * scale for a, t in zip(thumb, tip))
    events = []
    for _ in range(20):
        seq.t += 0.033
        events.extend(
            seq.engine.process(replace(frame, timestamp=seq.t, landmarks=tuple(points))).events
        )
    assert seq.engine.down
    assert not any(e.kind == "up" for e in events)
    assert any(e.kind == "up" for e in seq.frames("open"))


def motion_sequence(span=0.30, clutch=True):
    edge = (1 - span) / 2
    return Sequence(
        GestureEngine(
            EngineConfig(
                box_left=edge,
                box_top=edge,
                box_right=1 - edge,
                box_bottom=1 - edge,
                reanchor_on_open=clutch,
            )
        )
    )


def test_small_range_halves_hand_travel_for_same_screen_distance():
    # Settle the speed-adaptive filter before comparing displacement.
    small, original = motion_sequence(), motion_sequence(0.60)
    for seq in (small, original):
        seq.frames("open", x=0.5)
    small.frames("open", 90, x=0.53)
    original.frames("open", 90, x=0.56)
    assert small.engine.pointer == pytest.approx(original.engine.pointer, abs=1e-6)
    assert small.engine.pointer[0] == pytest.approx(0.60)


@pytest.mark.parametrize("initial", [None, (0.23, 0.81)])
def test_first_open_anchors_current_cursor_and_preview_box(initial):
    from pinchpilot.features import extract

    seq = motion_sequence()
    seq.engine.pointer = initial
    events = seq.frames("open", 10, x=0.68, y=0.41)
    target = initial or (0.5, 0.5)
    assert all((e.x, e.y) == pytest.approx(target) for e in events)
    left, top, right, bottom = seq.engine.active_box
    assert (right - left, bottom - top) == pytest.approx((0.3, 0.3))
    raw = extract(synthetic_hand(seq.t, "open", x=0.68, y=0.41)).pointer
    assert (left + target[0] * (right - left), top + target[1] * (bottom - top)) == pytest.approx(
        raw
    )


@pytest.mark.parametrize("rest", ["fist", "timeout", "hand_swap"])
def test_rest_releases_once_and_rearms_at_old_cursor(rest):
    seq = motion_sequence()
    seq.frames("open")
    seq.frames("pinch")
    anchor = seq.engine.pointer
    if rest == "fist":
        events = seq.frames("fist", x=0.30, y=0.35)
    elif rest == "timeout":
        seq.t += 0.25
        events = seq.engine.tick(seq.t).events
    else:
        seq.t += 1 / 30
        frame = replace(synthetic_hand(seq.t, "pinch"), handedness="Left")
        events = seq.engine.process(frame).events
    assert [e.kind for e in events] == ["up"]
    assert seq.engine.state == "WAIT_OPEN"
    assert not seq.frames("pinch", x=0.30, y=0.35)
    events = seq.frames("open", 15, x=0.30, y=0.35)
    assert events and all((e.x, e.y) == pytest.approx(anchor) for e in events)
    seq.frames("open", 40, x=0.33, y=0.35)
    assert seq.engine.pointer[0] == pytest.approx(anchor[0] + 0.1, abs=1e-5)
    assert any(e.kind == "down" for e in seq.frames("pinch", x=0.33, y=0.35))


def test_drag_release_does_not_snap_back_to_uncompensated_hand_position():
    seq = motion_sequence()
    seq.frames("open")
    seq.frames("pinch")
    seq.frames("pinch", 10, x=0.54)
    seq.frames("pinch", 30, x=0.57)
    assert seq.engine.state == "DRAG"
    last_drag = seq.engine.pointer
    events = seq.frames("open", 25, x=0.57)
    assert [e.kind for e in events].count("up") == 1
    assert seq.engine.pointer == pytest.approx(last_drag)
    assert all((e.x, e.y) == pytest.approx(last_drag) for e in events)


def test_fixed_mapping_remains_available_as_baseline():
    seq = motion_sequence(0.60, clutch=False)
    seq.frames("open")
    before = seq.engine.pointer
    seq.frames("fist", x=0.35)
    seq.frames("open", 60, x=0.35)
    assert seq.engine.pointer[0] == pytest.approx(before[0] - 0.25)
    assert seq.engine.active_box == pytest.approx((0.2, 0.2, 0.8, 0.8))


def test_small_range_does_not_multiply_scroll_distance():
    totals = []
    for span in (0.60, 0.30, 0.20):
        seq = motion_sequence(span)
        seq.frames("open")
        seq.frames("scroll", 30)
        events = seq.frames("scroll", 90, y=0.40)
        totals.append(sum(e.value for e in events if e.kind == "scroll"))
    # The filter and sub-step cutoff can differ slightly; total travel remains comparable.
    assert totals == pytest.approx([3.0] * 3, abs=0.05)
