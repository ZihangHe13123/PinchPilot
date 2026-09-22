import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pinchpilot import platform_io
from pinchpilot.domain import InputEvent
from pinchpilot.mouse_session import MouseSession


@pytest.fixture
def windows(monkeypatch):
    calls = []
    user32 = SimpleNamespace(
        GetSystemMetrics=lambda n: [1920, 1080][n],
        SetCursorPos=Mock(return_value=1),
        GetCursorPos=Mock(return_value=1),
        SendInput=Mock(return_value=1),
        GetAsyncKeyState=lambda _: 0,
    )
    monkeypatch.setattr(platform_io, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=user32), raising=False)
    output = platform_io.MouseOutput()

    def send(count, pointer, size):
        item = ctypes.cast(pointer, ctypes.POINTER(output.win_input)).contents
        calls.append((item.u.mi.dwFlags, item.u.mi.mouseData, size))
        return 1

    user32.SendInput.side_effect = send
    return output, user32, calls


def test_windows_input_layout_coordinates_release_and_signed_wheel(windows):
    output, native, calls = windows
    expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    assert ctypes.sizeof(output.win_input) == expected
    output.emit(InputEvent("move", 1, 1))
    native.SetCursorPos.assert_called_with(1919, 1079)
    output.emit(InputEvent("down"))
    output.close()
    output.close()
    output.emit(InputEvent("scroll", value=-1.5))
    assert [c[0] for c in calls] == [2, 4, 0x0800]
    assert calls[-1][1] == (-120 & 0xFFFFFFFF)
    assert not output.down


def test_windows_rejected_input_never_sets_down(windows):
    output, native, _ = windows
    native.SendInput.side_effect = None
    native.SendInput.return_value = 0
    with pytest.raises(OSError):
        output.emit(InputEvent("down"))
    assert not output.down


def test_fractional_scroll_counts_only_native_sends(windows):
    output, _, calls = windows
    session = MouseSession(output, background=False)
    session.emit([InputEvent("scroll", value=0.25)] * 3)
    assert not calls and session.snapshot()["sent"] == {}
    session.emit([InputEvent("scroll", value=0.35)])
    assert calls[-1][0:2] == (0x0800, 120)
    session.emit([InputEvent("scroll", value=-1.5)])
    assert calls[-1][0:2] == (0x0800, -120 & 0xFFFFFFFF)
    assert session.snapshot()["sent"]["scroll"] == 2
    session.stop("test_done")


@pytest.mark.parametrize(
    "position,expected", [((960, 540), (960 / 1919, 540 / 1079)), ((-50, 2000), (0, 1))]
)
def test_windows_cursor_read_is_normalized_clamped_and_never_emits(windows, position, expected):
    output, native, calls = windows

    def read(pointer):
        point = ctypes.cast(pointer, native.GetCursorPos.argtypes[0]).contents
        point.x, point.y = position
        return 1

    native.GetCursorPos.side_effect = read
    assert output.position() == pytest.approx(expected)
    native.SetCursorPos.assert_not_called()
    assert calls == []


def test_windows_cursor_read_failure_is_reported(windows):
    output, native, calls = windows
    native.GetCursorPos.return_value = 0
    with pytest.raises(OSError):
        output.position()
    assert calls == []


@pytest.fixture
def macos(monkeypatch):
    q = SimpleNamespace(
        CGDisplayBounds=lambda _: SimpleNamespace(
            origin=SimpleNamespace(x=0, y=0), size=SimpleNamespace(width=1440, height=900)
        ),
        CGMainDisplayID=lambda: 1,
        CGEventCreateMouseEvent=Mock(
            side_effect=lambda source, kind, point, button: {
                "kind": kind,
                "point": point,
                "button": button,
            }
        ),
        CGEventSetIntegerValueField=Mock(
            side_effect=lambda event, field, value: event.update(count=value)
        ),
        CGEventCreateScrollWheelEvent=Mock(return_value="scroll-event"),
        CGEventPost=Mock(),
        CGEventCreate=lambda _: "read-event",
        CGEventGetLocation=lambda _: (50, 60),
        kCGEventLeftMouseDragged=6,
        kCGEventMouseMoved=5,
        kCGEventLeftMouseDown=1,
        kCGEventLeftMouseUp=2,
        kCGEventRightMouseDragged=7,
        kCGEventRightMouseDown=3,
        kCGEventRightMouseUp=4,
        kCGMouseButtonLeft=0,
        kCGMouseButtonRight=1,
        kCGHIDEventTap=0,
        kCGScrollEventUnitLine=1,
        kCGMouseEventClickState=1,
    )
    monkeypatch.setitem(sys.modules, "Quartz", q)
    trust = SimpleNamespace(AXIsProcessTrusted=lambda: True)
    monkeypatch.setitem(sys.modules, "ApplicationServices", trust)
    monkeypatch.setitem(
        sys.modules,
        "AppKit",
        SimpleNamespace(NSEvent=SimpleNamespace(doubleClickInterval=lambda: 0.5)),
    )
    monkeypatch.setattr(platform_io, "sys", SimpleNamespace(platform="darwin"))
    clock = SimpleNamespace(now=10.0)
    output = platform_io.MouseOutput(clock=lambda: clock.now)
    return output, q, clock, trust


def test_mac_adapter_native_calls_are_mocked_and_permission_gated(macos):
    output, q, _, trust = macos
    trust.AXIsProcessTrusted = lambda: False
    with pytest.raises(PermissionError):
        platform_io.MouseOutput()
    q.CGEventPost.assert_not_called()
    trust.AXIsProcessTrusted = lambda: True
    output = platform_io.MouseOutput()
    assert output.position() == pytest.approx((50 / 1439, 60 / 899))
    q.CGEventPost.assert_not_called()
    output.emit(InputEvent("down", 0.5, 0.5))
    output.emit(InputEvent("move", 0.6, 0.5))
    assert q.CGEventCreateMouseEvent.call_args.args[1] == 6
    output.close()
    assert q.CGEventCreateMouseEvent.call_args.args[1:3] == (2, (50, 60))
    output.emit(InputEvent("right_down", 0.5, 0.5))
    assert q.CGEventCreateMouseEvent.call_args.args[1] == 3
    assert q.CGEventCreateMouseEvent.call_args.args[3] == 1
    output.emit(InputEvent("move", 0.6, 0.5))
    assert q.CGEventCreateMouseEvent.call_args.args[1] == 7
    output.close()
    assert q.CGEventCreateMouseEvent.call_args.args[1:] == (4, (50, 60), 1)
    assert not output.right_down
    q.CGEventPost.reset_mock()
    assert output.emit(InputEvent("scroll", value=0.4)) is False
    q.CGEventPost.assert_not_called()
    output.emit(InputEvent("scroll", value=0.7))
    q.CGEventCreateScrollWheelEvent.assert_called_with(None, 1, 1, 1)
    q.CGEventPost.assert_called_once_with(0, "scroll-event")
    output.emit(InputEvent("scroll", value=-1.2))
    q.CGEventCreateScrollWheelEvent.assert_called_with(None, 1, 1, -1)


def mac_click(macos, offset, point=(0.5, 0.5), button="left", hold=0.08):
    output, _, clock, _ = macos
    clock.now = 10.0 + offset
    prefix = "right_" if button == "right" else ""
    output.emit(InputEvent(prefix + "down", *point))
    clock.now += hold
    output.emit(InputEvent(prefix + "up", *point))


def mac_counts(native):
    return [
        call.args[1]["count"]
        for call in native.CGEventPost.call_args_list
        if isinstance(call.args[1], dict) and "count" in call.args[1]
    ]


def test_mac_fast_complete_clicks_report_double_and_triple_counts_on_both_edges(macos):
    output, native, clock, _ = macos
    mac_click(macos, 0)
    clock.now = 10.15
    output.emit(InputEvent("move", 0.501, 0.5))  # Small jitter stays in the same target.
    mac_click(macos, 0.25, point=(0.501, 0.5))
    mac_click(macos, 0.5)
    assert mac_counts(native) == [1, 1, 2, 2, 3, 3]
    assert not output.down


@pytest.mark.parametrize(
    "interruption", ["timeout", "distance", "move", "drag", "hold", "right", "scroll", "stop"]
)
def test_mac_separate_clicks_and_interrupted_sequences_are_not_double_clicks(macos, interruption):
    output, native, clock, _ = macos
    if interruption == "drag":
        output.emit(InputEvent("down", 0.5, 0.5))
        output.emit(InputEvent("move", 0.55, 0.5))
        output.emit(InputEvent("move", 0.5, 0.5))
        output.emit(InputEvent("up", 0.5, 0.5))
    else:
        mac_click(macos, 0, hold=0.7 if interruption == "hold" else 0.08)
    if interruption == "move":
        output.emit(InputEvent("move", 0.55, 0.5))
        output.emit(InputEvent("move", 0.5, 0.5))
    elif interruption == "right":
        mac_click(macos, 0.12, button="right")
    elif interruption == "scroll":
        output.emit(InputEvent("scroll", value=1))
    elif interruption == "stop":
        output.close()
    delay = 0.8 if interruption == "hold" else (0.6 if interruption == "timeout" else 0.3)
    point = (0.51, 0.5) if interruption == "distance" else (0.5, 0.5)
    mac_click(macos, delay, point=point)
    assert mac_counts(native)[-2:] == [1, 1]


def test_mac_failed_second_down_does_not_advance_or_hold_and_release_retry_keeps_count(macos):
    output, native, clock, _ = macos
    mac_click(macos, 0)
    clock.now = 10.25
    native.CGEventPost.side_effect = RuntimeError("post failed")
    with pytest.raises(RuntimeError):
        output.emit(InputEvent("down", 0.5, 0.5))
    assert not output.down
    native.CGEventPost.side_effect = None
    output.emit(InputEvent("down", 0.5, 0.5))
    assert native.CGEventPost.call_args.args[1]["count"] == 2
    native.CGEventPost.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError):
        output.close()
    assert output.down
    native.CGEventPost.side_effect = None
    output.close()
    assert not output.down
    assert native.CGEventPost.call_args.args[1]["count"] == 2
    mac_click(macos, 0.4)
    assert mac_counts(native)[-2:] == [1, 1]


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit/CGEvent inspection requires macOS")
def test_30fps_gesture_double_click_is_recognized_by_appkit_without_posting(monkeypatch):
    import AppKit
    import ApplicationServices
    import Quartz

    from pinchpilot.tripod import TripodEngine
    from pinchpilot.tripod_demo import synthetic_tripod

    monkeypatch.setattr(ApplicationServices, "AXIsProcessTrusted", lambda: True)
    monkeypatch.setattr(Quartz, "CGPreflightListenEventAccess", lambda: True)
    posted = Mock()
    monkeypatch.setattr(Quartz, "CGEventPost", posted)
    clock = SimpleNamespace(now=10.0)
    output = platform_io.MouseOutput(clock=lambda: clock.now)
    engine = TripodEngine()
    # Real engine debounce, 30 FPS, complete release between short contacts.
    for frames, contact in ((10, 0.85), (4, 0.10), (4, 0.85), (4, 0.10), (4, 0.85)):
        for _ in range(frames):
            clock.now += 1 / 30
            for event in engine.process(synthetic_tripod(clock.now, contact=contact)).events:
                output.emit(event)
    native_buttons = [
        call.args[1]
        for call in posted.call_args_list
        if Quartz.CGEventGetType(call.args[1])
        in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp)
    ]
    assert [AppKit.NSEvent.eventWithCGEvent_(event).clickCount() for event in native_buttons] == [
        1,
        1,
        2,
        2,
    ]
    assert not output.down


def test_windows_right_button_is_separate_and_close_releases_both_once(windows):
    output, native, calls = windows
    output.emit(InputEvent("right_down"))
    assert output.right_down and not output.down
    output.emit(InputEvent("right_up"))
    assert not output.right_down
    output.emit(InputEvent("down"))
    output.emit(InputEvent("right_down"))
    output.close()
    output.close()
    assert [row[0] for row in calls] == [8, 16, 2, 8, 4, 16]
    assert not output.down and not output.right_down
    native.SendInput.side_effect = None
    native.SendInput.return_value = 0
    with pytest.raises(OSError):
        output.emit(InputEvent("right_down"))
    assert not output.right_down


def test_close_attempts_second_button_even_if_first_release_fails(windows):
    output, native, calls = windows
    output.emit(InputEvent("down"))
    output.emit(InputEvent("right_down"))
    native.SendInput.side_effect = [0, 1]
    with pytest.raises(OSError):
        output.close()
    assert native.SendInput.call_count == 4
    assert output.down and not output.right_down


def test_failed_right_release_stays_tracked_until_cleanup(windows):
    output, native, calls = windows
    output.emit(InputEvent("right_down"))
    native.SendInput.side_effect = [0, 1]
    with pytest.raises(OSError):
        output.emit(InputEvent("right_up"))
    assert output.right_down
    output.close()
    assert not output.right_down
