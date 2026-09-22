from dataclasses import replace

import pytest
from test_desktop_control import Rig

from pinchpilot.domain import HandFrame
from pinchpilot.framing import FramingMonitor, FramingStatus, framing_hint


def hand(t, edges=(), margin=0.05, exclude=()):
    points = [(0.5, 0.5, 0.0)] * 21
    for edge in edges:
        axis, direction = {"left": (0, 0), "right": (0, 1), "top": (1, 0), "bottom": (1, 1)}[edge]
        point = list(points[4])
        point[axis] = margin if direction == 0 else 1 - margin
        points[4] = tuple(point)
    for index in exclude:
        points[index] = (-0.5, 1.4, 0)
    return HandFrame(t, tuple(points))


@pytest.mark.parametrize(
    "edges", [("left",), ("right",), ("top",), ("bottom",), ("left", "bottom")]
)
def test_warns_on_camera_edges_after_persistence_with_hysteresis(edges):
    monitor = FramingMonitor()
    assert monitor.update(hand(0)).state == "clear"
    for i in range(1, 7):
        result = monitor.update(hand(i / 30, edges))
        if i < 4:
            assert result.state == "clear"
    assert result == FramingStatus("edge", edges)
    for i in range(7, 17):
        assert monitor.update(hand(i / 30, edges, margin=0.10)) == result
    for i in range(17, 27):
        cleared = monitor.update(hand(i / 30, edges, margin=0.14))
    assert cleared == FramingStatus("clear")


def test_wrist_and_pinky_outside_image_and_one_bad_frame_do_not_warn():
    monitor = FramingMonitor()
    for i in range(30):
        frame = hand(i / 30, edges=("left",) if i == 12 else (), exclude=(0, 20))
        assert monitor.update(frame).state == "clear"


def test_tracking_loss_is_debounced_and_recovery_clears_old_direction():
    monitor = FramingMonitor()
    for i in range(6):
        monitor.update(hand(i / 30, edges=("bottom",)))
    assert monitor.status.state == "edge"
    assert monitor.update(HandFrame(6 / 30)).state == "edge"
    for i in range(7, 14):
        monitor.update(HandFrame(i / 30))
    assert monitor.status == FramingStatus("lost")
    assert monitor.update(hand(14 / 30)) == FramingStatus("clear")
    monitor.reset()
    assert monitor.update(HandFrame(1)) == FramingStatus("waiting")


@pytest.mark.parametrize("bad", ["nan_point", "incomplete", "untracked"])
def test_invalid_hand_cannot_generate_directional_alert(bad):
    monitor = FramingMonitor()
    frame = hand(0, ("left",))
    if bad == "nan_point":
        points = list(frame.landmarks)
        points[4] = (float("nan"), 0.5, 0)
        frame = replace(frame, landmarks=tuple(points))
    elif bad == "incomplete":
        frame = replace(frame, landmarks=frame.landmarks[:20])
    assert monitor.update(frame, tracked=bad != "untracked") == FramingStatus("waiting")


def test_clock_discontinuity_and_frame_gap_do_not_count_as_edge_persistence():
    monitor = FramingMonitor()
    for t in (1, 1.03, 2):
        assert monitor.update(hand(t, ("left",))).state == "clear"
    assert monitor.update(hand(2, ("left",))).state == "waiting"
    assert monitor.update(hand(float("nan"))).state == "waiting"


def test_guidance_preserves_drag_and_does_not_claim_loss_was_out_of_frame():
    status = FramingStatus("edge", ("bottom",))
    assert "保持食拇" in framing_hint(status, "DRAG")
    assert "收回 V" in framing_hint(status, "SCROLL")
    assert "完成当前点击" in framing_hint(status, "PRESSED")
    assert "松中指" in framing_hint(status, "CONTROL")
    lost = framing_hint(FramingStatus("lost"), "WAIT_GRIP")
    assert "暂未跟踪" in lost and "出画面" not in lost


def test_monitor_is_advisory_and_repositioning_does_not_move_cursor(tmp_path):
    rig = Rig(tmp_path)
    try:
        rig.live()
        # Walk to the frame edge with accepted small steps, without a tracking jump.
        for i in range(22):
            rig.frames(x=0.5 - i * 0.02)
        rig.frames(8, x=0.08)
        assert rig.control.framing_status.state == "edge"
        assert rig.control.active and rig.control.result.state == "CONTROL"
        pointer = rig.output.point
        rig.frames(3, x=0.08, grip=False)
        for i in range(22):
            rig.frames(x=0.08 + i * 0.02, grip=False)
        assert rig.output.point == pytest.approx(pointer)
        rig.frames(10, x=0.5)
        assert rig.output.point == pytest.approx(pointer)
        assert rig.control.result.state == "CONTROL"
        assert rig.control.framing_status.state == "clear"
        assert all(event.kind == "move" for event in rig.output.events)
        rig.frames(18, contact=0.1)
        assert rig.output.down
        rig.frames(8, missing=True)
        assert rig.control.framing_status.state == "lost"
        assert not rig.output.down and rig.control.active
        assert sum(event.kind == "up" for event in rig.output.events) == 1
        rig.frames(8, contact=0.1)
        assert not rig.output.down  # Returning with click held must not press again.
        rig.control.set_source("synthetic_demo")
        rig.frames(8, x=0.03)
        assert rig.control.framing_status.state == "waiting"
    finally:
        rig.control.close()


def test_stale_camera_discards_old_framing_state(tmp_path):
    rig = Rig(tmp_path)
    try:
        rig.live()
        rig.frames(8, x=0.03)
        assert rig.control.framing_status.state == "edge"
        rig.clock.now += 0.3
        rig.control.tick()
        assert rig.control.framing_status.state == "waiting"
        assert not rig.control.active
    finally:
        rig.control.close()
