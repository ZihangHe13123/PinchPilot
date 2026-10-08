"""The schematic hand that acts out each step, and its place in the capture window."""

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import QApplication

from pinchpilot import contact_demo as demo
from pinchpilot import contact_protocol as protocol
from pinchpilot.contact_capture import ContactCaptureWindow
from pinchpilot.contact_demo_view import GestureDemo, caption
from pinchpilot.contact_protocol import KEY, LEAD, SETTLE
from pinchpilot.tripod_demo import synthetic_tripod

STEPS = {step.name: step for step in protocol.session()}


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_fingers_reach_the_thumb_exactly_and_bones_keep_their_length():
    cases = {
        (0, 0, 0): (False, False, False),
        (1, 0, 0): (True, False, False),
        (0, 1, 0): (False, True, False),
        (1, 1, 0): (True, True, False),
        (0, 0, 1): (False, False, True),
        (1, 0, 1): (True, False, True),
        (0, demo.NEAR, 0): (False, False, False),  # Close, with a gap that can be seen.
        (1, demo.NEAR, 0): (True, False, False),
    }
    for closing, touching in cases.items():
        points = demo.hand_points(*closing)
        assert points.shape == (21, 3) and np.isfinite(points).all()
        gaps = demo.tip_gaps(points)
        assert tuple(gap < demo.TOUCHING for gap in gaps) == touching, closing
        assert all(gap > 2.5 * demo.TOUCHING for gap, on in zip(gaps, touching) if not on)
        for name, first in demo.FIRST_JOINT.items():
            for bone, length in enumerate(demo.BONES[name]):
                actual = np.linalg.norm(points[first + bone + 1] - points[first + bone])
                assert actual == pytest.approx(length, abs=1e-6)
    # Half way there the bones are still whole: fingers bend, they do not stretch.
    half = demo.hand_points(0.5, 0.5, 0.0)
    assert np.linalg.norm(half[8] - half[7]) == pytest.approx(demo.BONES["index"][2], abs=1e-6)
    assert demo.tip_gaps(demo.hand_points(1, 0, 0))[0] == pytest.approx(0.0, abs=1e-6)
    # Out-of-range amounts are clamped rather than drawn as an impossible hand.
    assert np.allclose(demo.hand_points(3, -1, 9), demo.hand_points(1, 0, 1))


def test_the_hand_shows_as_many_touches_as_the_prompt_asks_for():
    for step in STEPS.values():
        asked = re.search(r"约 (\d+) 次", step.text)
        if KEY in step.labels:
            assert demo.touches(step) == int(asked.group(1)), step.name
        elif "near" in step.name:
            assert demo.touches(step, demo.APPROACH) == int(asked.group(1)), step.name
    assert demo.TAP[1] == pytest.approx(2.6)  # "两三秒一次"


def test_space_is_asked_for_exactly_while_the_marked_finger_touches():
    for step in STEPS.values():
        moments = np.arange(0, step.seconds, 1 / 30)
        states = [demo.demo_state(step, moment) for moment in moments]
        gaps = [demo.tip_gaps(demo.hand_points(s.grip, s.index, s.ring)) for s in states]
        if KEY not in step.labels:
            expected = False if "near" in step.name else None
            assert all(state.space is expected for state in states), step.name
            # Whatever the step, a finger that is not asked to touch never does.
            for channel in (1, 2):
                assert all(gap[channel] > demo.TOUCHING for gap in gaps), step.name
            continue
        marked = step.labels.index(KEY)
        held = np.array([state.space for state in states])
        touching = np.array([gap[marked] < demo.TOUCHING for gap in gaps])
        assert (held <= touching).all() and held.any(), step.name
        # Every touch lies inside the part of the step that gets labels.
        assert moments[held].min() >= SETTLE and moments[held].max() <= step.seconds - LEAD
        starts = moments[np.flatnonzero(np.diff(held.astype(int)) == 1) + 1]
        assert len(starts) == demo.touches(step)
        if step.labels[0] == 1:  # The grip stays closed while the other finger taps.
            late = moments > 0.5
            assert all(gap[0] < demo.TOUCHING for gap, ok in zip(gaps, late) if ok), step.name


def test_grip_steps_and_moving_steps_look_like_what_they_ask_for():
    assert demo.demo_state(STEPS["grip_on"], 2.0).grip == 1.0
    assert demo.demo_state(STEPS["grip_off"], 0.0).grip == 1.0  # It opens from a closed grip.
    assert demo.demo_state(STEPS["grip_off"], 1.0).grip == 0.0
    assert demo.demo_state(STEPS["open"], 3.0) == demo.DemoState()
    moving = [demo.demo_state(STEPS["grip_move"], moment) for moment in np.arange(1, 9, 0.5)]
    assert all(state.grip == 1.0 for state in moving)
    assert len({state.shift for state in moving}) > 5
    turning = [demo.demo_state(STEPS["adjust"], moment).turn for moment in np.arange(0, 7, 0.5)]
    assert max(turning) > 10 and min(turning) < -10
    # A step from another protocol is drawn from its labels alone.
    other = protocol.Step("custom", "别的动作", 9.0, (1, 0, KEY))
    state = demo.demo_state(other, 1.5)
    assert state.grip == 1.0 and state.ring == 1.0 and state.space is True


def test_widget_draws_the_touch_the_space_key_and_a_left_hand(application):
    widget = GestureDemo()
    widget.resize(320, 330)
    widget.show_step(None, 0.0)
    blank = widget.grab().toImage()
    step = STEPS["grip_index_tap"]

    def picture(moment):
        widget.show_step(step, moment)
        return widget.grab().toImage()

    def key_colour(image):
        return QColor(image.pixel(image.width() // 2 - 80, image.height() - 22)).name()

    apart, touching = picture(0.6), picture(1.5)
    assert apart != blank and touching != apart
    assert widget.state.space is True and key_colour(touching) == "#2f8f5f"  # Held: green.
    assert key_colour(apart) == "#1a2a3a"
    widget.show_step(STEPS["open"], 1.0)
    assert widget.state.space is None

    def thumb_side(image):
        """Mean x of the thumb-coloured pixels, as a share of the width."""
        columns = [
            x
            for x in range(0, image.width(), 2)
            for y in range(0, image.height() - 60, 2)
            if QColor(image.pixel(x, y)).name() == "#eafff9"
        ]
        return sum(columns) / len(columns) / image.width()

    widget.show_step(STEPS["open"], 1.0)
    widget.show_step(STEPS["grip_on"], 0.0)  # Thumb highlighted and still resting, out at the side.
    right_hand = thumb_side(widget.grab().toImage())
    widget.mirror = True
    widget.update()
    left_hand = thumb_side(widget.grab().toImage())
    widget.mirror = False
    assert right_hand < 0.42 and left_hand > 0.58  # A left hand has its thumb on the right.
    assert "空格键变绿" in caption(step) and "不按空格" in caption(STEPS["index_near"])
    assert "一起切换" in caption(STEPS["grip_on"], several=True) and caption(None) == ""


def key(window, pressed, which=Qt.Key.Key_Space):
    kind = QEvent.Type.KeyPress if pressed else QEvent.Type.KeyRelease
    event = QKeyEvent(kind, which, Qt.KeyboardModifier.NoModifier)
    (window.keyPressEvent if pressed else window.keyReleaseEvent)(event)


def test_capture_window_plays_the_clip_while_reading_and_follows_it_while_recording(
    application, tmp_path
):
    window = ContactCaptureWindow(tmp_path, guided=True)
    window.timer.stop()
    window.test_now = 50.0
    window.clock = lambda: window.test_now
    window.protocol = [
        protocol.Step("grip_on", "捏住：拇指和中指", 3.0, (1, 0, 0), round=1),
        protocol.Step("grip_off", "松开", 2.5, (0, 0, 0), round=1),
        protocol.Step("grip_index_tap", "保持捏住。食指点拇指", 9.0, (1, KEY, 0), "按住空格", 1),
    ]

    def feed(seconds=1 / 30, mirrored=False):
        for _ in range(max(1, round(seconds * 30))):
            window.test_now += 1 / 30
            frame = synthetic_tripod(window.test_now)
            if mirrored:
                points = tuple((1 - x, y, z) for x, y, z in frame.landmarks)
                frame = type(frame)(frame.timestamp, points, frame.aspect, frame.handedness, 0.9)
            window.feed(frame, window.test_now)

    feed()
    assert window.demo.isHidden() and not window.demo_timer.isActive()
    window.consent.setChecked(True)
    window.start_recording()
    assert not window.demo.isHidden() and window.demo_timer.isActive()
    # Reading: the hand plays the whole clip, its two prompts in turn, then starts over.
    assert window.demo.step.name == "grip_on" and "一起切换" in window.demo_caption.text()
    feed(2.0)
    assert window.demo.step.name == "grip_on" and window.demo.state.grip == 1.0
    feed(2.0)
    assert window.demo.step.name == "grip_off" and window.demo.state.grip == 0.0
    feed(2.6)  # 5.5 s of clip and a second of pause later it has started again.
    assert window.demo.step.name == "grip_on"
    assert window.count == 0  # Still nothing recorded.
    # The example is drawn with the thumb on the side it has on screen.
    thumb_left = window.demo.mirror
    feed(mirrored=True)
    assert window.demo.mirror != thumb_left

    key(window, True)
    key(window, False)
    assert window.clip_running and window.begin_button.text() == "正在录这一段…"
    feed(mirrored=False)
    assert window.demo.mirror != thumb_left  # Decided before the first clip; it stays.
    feed(1.0)
    assert window.demo.step.name == "grip_on" and window.demo.state.grip == 1.0
    feed(2.6)  # Now the screen says "松开", and so does the hand.
    assert window.prompt_label.text() == "松开" and window.demo.step.name == "grip_off"
    assert window.demo.state.grip == 0.0
    feed(2.2)
    assert not window.clip_running and window.demo.step.name == "grip_index_tap"
    assert window._demo_step(window.test_now)[1] < 0.3  # The next clip plays from its start.
    assert "空格键变绿" in window.demo_caption.text()
    assert window.begin_button.text() == "开始这一段（空格）"

    feed(window.START_LOCK)
    key(window, True)
    key(window, False)
    started = window.test_now
    # Recording: the hand keeps the pace of the clip itself, touch by touch.
    seen = []
    while window.clip_running:
        feed()
        if window.clip_running:
            seen.append((window.test_now - started, window.demo.state.space))
    lit = [moment for moment, space in seen if space]
    assert lit and min(lit) == pytest.approx(demo.TAP[0], abs=0.05)
    assert sum(1 for (_, a), (_, b) in zip(seen, seen[1:]) if b and not a) == 3
    assert not window.recorder_active and window.stop_reason == "protocol_complete"
    assert window.demo.isHidden() and not window.demo_timer.isActive()
    window.close()
