import json
import math
import threading
from dataclasses import replace

import pytest

from pinchpilot.desktop_control import DesktopController
from pinchpilot.domain import HandFrame, InputEvent
from pinchpilot.mouse_session import MouseSession
from pinchpilot.tripod_demo import synthetic_tripod
from pinchpilot.vision import Packet

TICK = 0.015625  # One GetTickCount64() step: time.monotonic() on Windows before Python 3.13.


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class CoarseClock(Clock):
    # Time still passes in `now`; a reading only moves once per TICK.
    def __call__(self):
        return math.floor(self.now / TICK) * TICK


class FakeMouse:
    def __init__(self):
        self.down = self.right_down = self.escape = False
        self.point = (0.32, 0.61)
        self.events = []
        self.fail_emit = ""
        self.fail_close = False
        self.released = threading.Event()

    def position(self):
        return self.point

    def emergency_pressed(self):
        return self.escape

    def emit(self, event):
        if event.kind == self.fail_emit:
            raise OSError("injected native failure")
        self.events.append(event)
        if event.kind == "move":
            self.point = (event.x, event.y)
        if event.kind in ("down", "up"):
            self.down = event.kind == "down"
        if event.kind in ("right_down", "right_up"):
            self.right_down = event.kind == "right_down"

    def close(self):
        if self.fail_close:
            raise OSError("injected release failure")
        if self.down:
            self.emit(InputEvent("up", *self.point))
        if self.right_down:
            self.emit(InputEvent("right_up", *self.point))
        self.released.set()


class Rig:
    def __init__(self, directory, clock=None):
        self.clock = clock or Clock()
        self.output = FakeMouse()
        self.created = 0
        self.control = DesktopController(directory, self.clock, self.factory, background=False)

    def factory(self):
        self.created += 1
        return self.output

    def frames(self, count=1, missing=False, step=1 / 30, **options):
        for _ in range(count):
            self.clock.now += step
            stamp = self.clock()  # One clock reading per frame, as CameraWorker stamps it.
            frame = HandFrame(stamp) if missing else synthetic_tripod(stamp, **options)
            self.control.consume(Packet(None, frame, 8.0, 30.0, frame.timestamp))
            self.control.tick()
        return self.control.result

    def live(self):
        self.control.set_source("camera")
        self.frames()
        self.control.enable()
        self.frames(8)


@pytest.fixture
def rig(tmp_path):
    value = Rig(tmp_path)
    yield value
    value.output.fail_close = False
    value.output.fail_emit = ""
    value.control.close()


@pytest.mark.parametrize("source", ["none", "synthetic_demo", "camera"])
def test_enabling_requires_a_fresh_real_camera_not_merely_selected_source(rig, source):
    rig.control.set_source(source)
    if source == "synthetic_demo":
        rig.frames(8)
    with pytest.raises(RuntimeError):
        rig.control.enable()
    assert not rig.created and not rig.output.events


def test_preview_never_outputs_and_enable_anchors_current_os_cursor(rig):
    rig.control.set_source("camera")
    rig.frames(8)
    rig.frames(8, contact=0.1)
    assert not rig.output.events and not rig.created
    rig.control.enable()
    rig.frames(8, contact=0.1)
    assert not rig.output.events  # Held gesture cannot immediately click after enabling.
    rig.frames(8)
    assert rig.output.events[0] == InputEvent("move", 0.32, 0.61)
    rig.frames(12, x=0.53)
    assert rig.output.point[0] > 0.36
    rig.control.stop()
    count = len(rig.output.events)
    rig.frames(8, contact=0.1)
    assert len(rig.output.events) == count
    rig.output.point = (0.7, 0.2)
    rig.control.enable()
    rig.frames(8)
    assert rig.output.point == pytest.approx((0.7, 0.2))


def test_real_output_mapping_left_right_drag_and_loss_releases_once(rig):
    rig.live()
    rig.frames(4, contact=0.1)
    assert rig.output.down
    rig.frames(8)
    assert not rig.output.down
    rig.frames(8, right_contact=0.1)
    rig.frames(20, right_contact=0.1)
    rig.frames(8)
    assert [e.kind for e in rig.output.events if e.kind != "move"] == [
        "down",
        "up",
        "right_down",
        "right_up",
    ]
    rig.frames(18, contact=0.1)
    start = rig.output.point
    rig.frames(12, contact=0.1, x=0.53)
    assert rig.output.down and rig.output.point[0] > start[0]
    rig.frames(missing=True)
    assert not rig.output.down and rig.control.active
    assert rig.control.metrics.sent["down"] == 2
    assert rig.control.metrics.sent["up"] == 2
    assert rig.control.metrics.drags == 1
    rig.frames(20, contact=0.1)
    assert not rig.output.down
    rig.output.point = (0.1, 0.8)
    rig.frames(8)
    assert rig.output.point == pytest.approx((0.1, 0.8))


@pytest.mark.parametrize(
    "interruption", ["stale", "heartbeat", "escape", "config", "source", "close"]
)
def test_held_button_cleanup_for_each_desktop_lifecycle(rig, interruption):
    rig.live()
    rig.frames(18, contact=0.1)
    assert rig.output.down
    if interruption == "stale":
        rig.clock.now += 0.26
        rig.control.tick()
    elif interruption == "heartbeat":
        rig.clock.now += 0.5
        rig.control.session.poll()  # No Qt tick; watchdog releases independently.
    elif interruption == "escape":
        rig.output.escape = True
        rig.control.session.poll()
    elif interruption == "config":
        rig.control.configure(replace(rig.control.engine.config, span=0.2))
    elif interruption == "source":
        rig.control.set_source("synthetic_demo")
    else:
        rig.control.close()
    assert not rig.output.down and not rig.control.active
    rig.control.tick()
    before = len(rig.output.events)
    rig.frames(10, contact=0.1)
    assert len(rig.output.events) == before
    assert rig.control.metrics.sent["up"] == 1


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), 200, 0, 100])
def test_bad_or_replayed_frame_stops_native_output(rig, stamp):
    rig.live()
    rig.frames(4, contact=0.1)
    frame = synthetic_tripod(stamp)
    assert not rig.control.consume(Packet(None, frame, 8, 30, stamp))
    assert not rig.control.active and not rig.output.down


@pytest.mark.parametrize(
    "clock,apart,accepted",
    [(CoarseClock, 0.008, False), (CoarseClock, 0.016, True), (Clock, 0.008, True)],
)
def test_new_frame_inside_one_coarse_clock_tick_stops_control_like_a_replay(
    tmp_path, clock, apart, accepted
):
    # An unchanged stamp is all the controller sees of a replay, so a different frame
    # read within the same tick of a coarse clock stops control exactly like one.
    rig = Rig(tmp_path, clock())
    try:
        rig.live()
        rig.frames(18, contact=0.1)
        assert rig.control.active and rig.output.down
        # One frame 1 ms into a clock tick, then a different one `apart` seconds later.
        rig.frames(contact=0.1, step=TICK - rig.clock.now % TICK + 0.001)
        rig.frames(contact=0.1, x=0.52, step=apart)
        assert rig.control.metrics.dropped == (0 if accepted else 1)
        assert rig.control.active is accepted and rig.output.down is accepted
        if not accepted:
            assert rig.control.session.snapshot()["reason"] == "invalid_or_stale_frame"
    finally:
        rig.control.close()


def test_heartbeat_renewal_cannot_revive_a_stalled_session(rig):
    rig.live()
    rig.frames(4, contact=0.1)
    rig.clock.now += 0.5
    rig.frames(8)
    assert not rig.control.active and not rig.output.down
    assert rig.control.session.snapshot()["reason"] == "gui_heartbeat_timeout"


def test_switches_do_not_freeze_movement_or_turn_a_long_click_into_drag(rig):
    rig.control.configure(
        replace(rig.control.engine.config, right_enabled=False, drag_enabled=False)
    )
    rig.live()
    rig.frames(8, right_contact=0.1, x=0.53)
    assert rig.output.point[0] > 0.35
    assert not any(e.kind == "right_down" for e in rig.output.events)
    anchor = rig.output.point
    rig.frames(20, contact=0.1, x=0.53)
    rig.frames(12, contact=0.1, x=0.56)
    assert rig.control.result.state == "PRESSED"
    assert rig.control.result.progress is None
    assert rig.output.point == anchor and rig.output.down
    rig.frames(4, x=0.56)
    assert not rig.output.down


def test_failed_native_send_is_not_counted_and_held_button_is_released(rig):
    rig.live()
    rig.frames(4, contact=0.1)
    rig.output.fail_emit = "move"
    rig.frames(18, contact=0.1)
    assert not rig.control.active and not rig.output.down
    assert rig.control.metrics.sent["up"] == 1
    assert "injected native failure" in rig.control.notice


def test_failed_release_is_preserved_retried_and_blocks_reenable(rig):
    rig.live()
    rig.frames(4, contact=0.1)
    rig.output.fail_close = True
    assert not rig.control.close()
    assert rig.control.pending_release and rig.output.down
    with pytest.raises(RuntimeError, match="松键"):
        rig.control.enable()
    assert rig.control.metrics.sent["up"] == 0
    rig.output.fail_close = False
    rig.control.session.poll()
    rig.control.tick()
    assert not rig.control.pending_release and not rig.output.down
    assert rig.control.metrics.sent["up"] == 1
    assert rig.control.close()


def test_watchdog_thread_releases_even_without_a_qt_event_loop():
    output = FakeMouse()
    session = MouseSession(output)
    session.emit([InputEvent("down", 0.3, 0.6)])
    output.escape = True
    assert output.released.wait(1.5)
    assert not output.down and not session.snapshot()["active"]
    session.thread.join(timeout=1)
    assert not session.thread.is_alive()


def test_logging_toggle_and_telemetry_are_local_and_source_labelled(rig):
    metrics = rig.control.metrics
    metrics.set_logging(True, {"span": 0.3}, "none")
    first_path = metrics.path
    rig.live()
    rig.frames(4, contact=0.1)
    rig.frames(8)
    metrics.set_logging(False, {}, "camera")
    rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["stores_images"] is False
    assert rows[-1]["type"] == "session_end"
    assert rows[-1]["summary"]["sent"]["down"] == 1
    assert "landmarks" not in first_path.read_text(
        encoding="utf-8"
    ) and "rgb" not in first_path.read_text(encoding="utf-8")
    before = first_path.read_bytes()
    rig.frames(30)
    assert first_path.read_bytes() == before
    assert metrics.snapshot(rig.clock())["processed_fps"] == pytest.approx(30, abs=0.2)
    rig.clock.now += 0.3
    assert metrics.snapshot(rig.clock())["processed_fps"] == 0
    metrics.set_logging(True, {}, "camera")
    assert metrics.path != first_path


def test_dispatch_also_blocks_synthetic_source_if_ui_teardown_were_bypassed(rig):
    rig.live()
    rig.control.source = "synthetic_demo"
    before = len(rig.output.events)
    rig.frames(20, contact=0.1)
    assert not rig.control.active and len(rig.output.events) == before


def test_logging_io_error_does_not_interrupt_button_cleanup(rig):
    from unittest.mock import Mock

    rig.live()
    rig.frames(4, contact=0.1)
    failed_file = Mock()
    failed_file.write.side_effect = OSError("disk full")
    rig.control.metrics.file = failed_file
    rig.control.stop()
    assert not rig.output.down and not rig.control.active
    assert rig.control.metrics.file is None
    assert "disk full" in rig.control.metrics.error
    assert rig.control.metrics.sent["up"] == 1


def test_permission_and_position_failures_never_activate_native_control(rig):
    rig.control.set_source("camera")
    rig.frames()

    def denied():
        raise PermissionError("permission denied")

    rig.control.output_factory = denied
    with pytest.raises(PermissionError):
        rig.control.enable()
    assert not rig.control.active
    rig.control.output_factory = rig.factory
    rig.output.position = denied
    with pytest.raises(PermissionError):
        rig.control.enable()
    assert not rig.control.active and rig.output.released.is_set()
