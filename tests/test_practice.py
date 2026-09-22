"""Virtual task outcomes; these tests do not evaluate hand recognition."""

import json
import math

import pytest

from pinchpilot.domain import EngineResult, InputEvent
from pinchpilot.practice import TASKS, PracticeSession


class Rig:
    def __init__(self, task="click", source="camera"):
        self.now = 1.0
        self.session = PracticeSession(task, started=self.now, source=source)
        self.source = source
        self.feed()

    def feed(self, *events, state="CONTROL", point=None, cancelled=False, dt=0.05):
        self.now += dt
        self.session.feed(
            EngineResult(
                state, point or self.session.pointer, list(events), cancelled=cancelled, grip=0.2
            ),
            self.now,
            self.source,
        )

    def click(self, point=None):
        point = point or self.session.target
        self.feed(InputEvent("down", *point))
        self.feed(InputEvent("up", *point))


@pytest.mark.parametrize("task", ("click", "reverse"))
def test_click_tasks_finish_only_paired_hits(task):
    rig = Rig(task)
    rig.feed(InputEvent("up", *rig.session.target))
    assert rig.session.hits == rig.session.attempts == 0
    rig.click((0, 0))
    assert rig.session.misses == 1 and rig.session.index == 0
    while rig.session.active:
        rig.click()
    report = rig.session.summary(rig.now)
    assert report["completed_targets"] == len(TASKS[task].targets)
    assert report["status"] == "completed"
    assert report["paired_button_attempts"] == report["total_targets"] + 1
    assert len(report["trials"]) == report["total_targets"]
    assert report["duration_s"] > 0


def test_start_requires_observed_clear_not_inherited_press():
    session = PracticeSession("click", started=0)
    point = session.target
    session.feed(EngineResult("PRESSED", point, [InputEvent("down", *point)], grip=0.2), 0.01)
    session.feed(EngineResult("CONTROL", point, [InputEvent("up", *point)], grip=0.2), 0.02)
    assert session.hits == 0 and session.awaiting_clear
    session.feed(EngineResult("CONTROL", point, grip=0.2), 0.03)
    assert not session.awaiting_clear


def test_missing_hand_or_uncleared_click_cannot_arm_task():
    session = PracticeSession("click", started=0)
    session.feed(EngineResult("WAIT_GRIP", (0.5, 0.5)), 0.01)
    session.feed(EngineResult("WAIT_GRIP", (0.5, 0.5), grip=0.2, contact=0.2), 0.02)
    assert session.awaiting_clear
    session.feed(EngineResult("CONTROL", (0.5, 0.5), grip=0.2, contact=0.7), 0.03)
    assert not session.awaiting_clear


def test_fresh_missing_hand_discards_held_button_without_scoring():
    rig = Rig()
    point = rig.session.target
    rig.feed(InputEvent("down", *point))
    rig.session.feed(EngineResult("WAIT_GRIP", point), rig.now + 0.01)
    rig.feed(InputEvent("up", *point))
    assert rig.session.hits == 0 and rig.session.pressed is None
    assert rig.session.interrupted_inputs == 1


def test_press_outside_release_inside_and_wrong_button_do_not_hit():
    rig = Rig()
    point = rig.session.target
    rig.feed(InputEvent("down", 0, 0))
    rig.feed(InputEvent("up", *point))
    rig.feed(InputEvent("right_down", *point))
    rig.feed(InputEvent("right_up", *point))
    assert rig.session.hits == 0 and rig.session.misses == 2


def test_double_click_fixed_protocol_and_timeout():
    rig = Rig("double_click")
    rig.click()
    assert rig.session.hits == 0 and rig.session.first_click is not None
    for _ in range(12):
        rig.feed()
    rig.click()
    assert rig.session.hits == 0 and rig.session.double_timeouts == 1
    rig.click()
    assert rig.session.hits == 1
    while rig.session.active:
        rig.click()
        rig.click()
    assert rig.session.completed


def test_double_click_miss_resets_pair():
    rig = Rig("double_click")
    rig.click()
    rig.click((0, 0))
    rig.click()
    assert rig.session.hits == 0 and rig.session.misses == 1


def test_drag_requires_origin_and_engine_drag_ready():
    rig = Rig("drag")
    origin, target = rig.session.drag_origin, rig.session.target
    rig.feed(InputEvent("down", *origin), state="PRESSED")
    rig.feed(InputEvent("up", *target))
    assert rig.session.hits == 0 and rig.session.misses == 1
    rig.feed(InputEvent("down", *origin), state="PRESSED")
    rig.feed(InputEvent("move", *target), state="DRAG")
    rig.feed(InputEvent("up", *target))
    assert rig.session.hits == 1


def test_drag_origin_matches_visible_square_including_corners():
    rig = Rig("drag")
    origin = tuple(
        v + (rig.session.radius_px - 1) / dimension
        for v, dimension in zip(rig.session.drag_origin, rig.session.viewport)
    )
    rig.feed(InputEvent("down", *origin), state="PRESSED")
    rig.feed(state="DRAG")
    rig.feed(InputEvent("up", *rig.session.target))
    assert rig.session.hits == 1


def test_reverse_targets_do_not_overlap_on_small_canvas():
    session = PracticeSession("reverse", started=0, viewport=(360, 200))
    a, b = session.spec.targets[1:3]
    assert abs(a[0] - b[0]) * session.viewport[0] > session.radius_px * 2


@pytest.mark.parametrize("interruption", ("cancelled", "gap"))
def test_interruption_release_never_scores(interruption):
    rig = Rig()
    point = rig.session.target
    rig.feed(InputEvent("down", *point))
    rig.feed(
        InputEvent("up", *point),
        cancelled=interruption == "cancelled",
        dt=0.5 if interruption == "gap" else 0.05,
    )
    assert rig.session.hits == rig.session.attempts == rig.session.misses == 0
    assert rig.session.cancelled_contacts == 1 and rig.session.pressed is None


def test_source_change_cancels_instead_of_mixing_evidence():
    rig = Rig()
    rig.session.feed(EngineResult("CONTROL", (0.5, 0.5)), rig.now + 0.05, "synthetic_demo")
    report = rig.session.summary(rig.now + 1)
    assert not rig.session.active
    assert report["source"] == "camera" and report["reason"] == "source_changed"
    assert report["status"] == "cancelled"


def test_scroll_requires_continuous_still_window_in_target():
    rig = Rig("scroll")
    rig.feed(InputEvent("scroll", value=-11.25), state="SCROLL")
    assert rig.session.scroll_value == rig.session.scroll_target == 65
    rig.feed(state="SCROLL", dt=0.8)
    assert rig.session.hits == 0
    for _ in range(10):
        rig.feed(state="CONTROL")
    assert rig.session.hits == 0
    for _ in range(17):
        rig.feed(state="SCROLL")
    assert rig.session.hits == 1
    for _ in range(2):
        amount = (rig.session.scroll_value - rig.session.scroll_target) / rig.session.SCROLL_SCALE
        rig.feed(InputEvent("scroll", value=amount), state="SCROLL")
        for _ in range(17):
            rig.feed(state="SCROLL")
    assert rig.session.completed and rig.session.scroll_events == 3


def test_scroll_movement_inside_target_restarts_stop_window():
    rig = Rig("scroll")
    rig.feed(InputEvent("scroll", value=-11.25), state="SCROLL")
    for _ in range(10):
        rig.feed(state="SCROLL")
    rig.feed(InputEvent("scroll", value=0.1), state="SCROLL")
    for _ in range(10):
        rig.feed(state="SCROLL")
    assert rig.session.hits == 0


def test_duplicate_and_bad_input_do_not_award_hits():
    rig = Rig()
    point = rig.session.target
    rig.session.feed(
        EngineResult("CONTROL", point, [InputEvent("down", *point), InputEvent("up", *point)]),
        rig.now,
    )
    rig.feed(point=(math.nan, 0.5))
    assert rig.session.hits == 0 and rig.session.invalid_inputs == 2
    json.dumps(rig.session.summary(rig.now), allow_nan=False)


def test_cancel_summary_is_stable_and_contains_no_trajectory():
    rig = Rig(source="synthetic_demo")
    rig.click()
    rig.session.cancel(rig.now, "window_closed")
    first = rig.session.summary(rig.now)
    later = rig.session.summary(rig.now + 100)
    assert first == later
    assert first["evidence"] == "synthetic_engineering"
    assert not first["stores_pointer_trajectory"]
    assert "pointer" not in first and "landmarks" not in first
    assert first["completed_targets"] == 1 and first["status"] == "cancelled"
