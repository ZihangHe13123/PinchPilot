"""The guided session's schedule and the draft labels built from prompts and the Space key."""

import pytest

from pinchpilot import contact_protocol as protocol
from pinchpilot.contact_data import CHANNELS
from pinchpilot.contact_protocol import KEY, Step


def test_session_runs_about_fourteen_minutes_and_covers_every_contact_both_ways():
    schedule = protocol.timeline(protocol.session())
    assert 13 * 60 <= schedule[-1][1] <= 15 * 60
    assert all(a[1] == b[0] for a, b in zip(schedule, schedule[1:])) and schedule[0][0] == 0
    steps = [step for _, _, step in schedule]
    assert {step.round for step in steps} == set(range(len(protocol.POSES) + 1))
    for channel in range(len(CHANNELS)):
        seen = {step.labels[channel] for step in steps}
        assert 0 in seen and seen & {1, KEY}  # Every contact is asked for both apart and touching.
    assert all(KEY in {step.labels[channel] for step in steps} for channel in (1, 2))
    assert any(step.labels == (1, 0, 0) for step in steps)
    assert any(step.labels == (1, KEY, 0) for step in steps)  # Click while the grip is held.
    assert all(sum(label == KEY for label in step.labels) <= 1 for step in steps)


def test_position_follows_the_schedule_and_is_none_outside_it():
    schedule = protocol.timeline([Step("a", "A", 2, (0, 0, 0)), Step("b", "B", 3, (1, 0, 0))])
    assert [protocol.position(schedule, t) for t in (-0.1, 0, 1.99, 2, 4.99, 5)] == [
        None,
        0,
        0,
        1,
        1,
        None,
    ]
    with pytest.raises(ValueError):
        protocol.timeline([Step("bad", "B", 0, (0, 0, 0))])
    with pytest.raises(ValueError):
        protocol.timeline([Step("bad", "B", 1, (0, 0))])


def test_key_spans_pair_presses_ignore_repeats_and_close_a_held_key():
    events = [(1.0, True), (1.1, True), (1.5, False), (1.6, False), (3.0, True)]
    assert protocol.key_spans(events, 10.0) == [(1.0, 1.5), (3.0, 10.0)]
    assert protocol.key_spans([], 10.0) == []


def test_draft_labels_leave_margins_missing_hands_and_key_edges_unlabelled():
    schedule = protocol.timeline(
        [Step("hold", "捏住", 4, (1, 0, 0)), Step("tap", "点", 6, (1, KEY, 0))]
    )
    events = [(6.0, True), (6.5, False)]  # One touch, half a second, in the second step.
    times = [0.2, 1.0, 3.5, 3.8, 4.5, 5.5, 5.95, 6.2, 6.47, 6.6, 9.8, 12.0]
    present = [True] * len(times)
    present[2] = False
    labels = protocol.draft_labels(times, present, schedule, events)
    blank = (None, None, None)
    assert labels == [
        blank,  # 0.2: still inside SETTLE of the first prompt
        (1, 0, 0),  # 1.0: prompted hold
        blank,  # 3.5: no hand in the frame
        blank,  # 3.8: inside LEAD before the prompt ends
        blank,  # 4.5: inside SETTLE of the second prompt
        (1, 0, 0),  # 5.5: Space not held
        (1, None, 0),  # 5.95: within KEY_EDGE of the press
        (1, 1, 0),  # 6.2: Space held
        (1, None, 0),  # 6.47: within KEY_EDGE of the release
        (1, 0, 0),  # 6.6: released
        blank,  # 9.8: inside LEAD of the last prompt
        blank,  # 12.0: after the session
    ]
    assert protocol.draft_labels(times, present, [], events) == [blank] * len(times)


def test_space_only_labels_the_channel_the_step_asks_for():
    schedule = protocol.timeline([Step("near", "靠近但不碰", 5, (0, 0, 0))])
    labels = protocol.draft_labels([2.0], [True], schedule, [(1.5, True), (3.0, False)])
    assert labels == [(0, 0, 0)]


def test_a_step_that_got_no_space_mark_leaves_its_marked_channel_unlabelled():
    schedule = protocol.timeline(
        [
            Step("tap", "点", 4, (1, KEY, 0)),
            Step("near", "靠近但不碰", 4, (1, 0, 0)),
            Step("ring", "碰无名指", 4, (0, 0, KEY)),
            Step("late", "点", 4, (0, KEY, 0)),
        ]
    )
    events = [(9.5, True), (10.0, False)]  # Only the third step was marked.
    assert protocol.unmarked(schedule, events, 16.0) == [0, 3]
    assert protocol.unmarked(schedule, events, 12.0) == [0]  # Stopped before the last step.
    assert protocol.unmarked(schedule, [(3.9, True), (12.1, False)], 16.0) == []
    labels = protocol.draft_labels([2.0, 6.0, 9.7, 11.0, 14.0], [True] * 5, schedule, events)
    assert labels == [(1, None, 0), (1, 0, 0), (0, 0, 1), (0, 0, 0), (0, None, 0)]


def test_intervals_run_length_encode_every_frame_once():
    blank = (None, None, None)
    labels = [blank, blank, (1, 0, 0), (1, 0, 0), (1, 1, 0), blank]
    assert protocol.intervals(labels) == [
        {"start": 0, "end": 2, "middle": None, "index": None, "ring": None},
        {"start": 2, "end": 4, "middle": 1, "index": 0, "ring": 0},
        {"start": 4, "end": 5, "middle": 1, "index": 1, "ring": 0},
        {"start": 5, "end": 6, "middle": None, "index": None, "ring": None},
    ]
    assert protocol.intervals([]) == []
