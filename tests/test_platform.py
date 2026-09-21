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


def test_mac_adapter_native_calls_are_mocked_and_permission_gated(monkeypatch):
    q = SimpleNamespace(
        CGDisplayBounds=lambda _: SimpleNamespace(
            origin=SimpleNamespace(x=0, y=0), size=SimpleNamespace(width=1440, height=900)
        ),
        CGMainDisplayID=lambda: 1,
        CGEventCreateMouseEvent=Mock(return_value="event"),
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
    )
    monkeypatch.setitem(sys.modules, "Quartz", q)
    trust = SimpleNamespace(AXIsProcessTrusted=lambda: False)
    monkeypatch.setitem(sys.modules, "ApplicationServices", trust)
    monkeypatch.setattr(platform_io, "sys", SimpleNamespace(platform="darwin"))
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
