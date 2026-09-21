import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pinchpilot import platform_io
from pinchpilot.domain import InputEvent


@pytest.fixture
def windows(monkeypatch):
    calls = []
    user32 = SimpleNamespace(
        GetSystemMetrics=lambda n: [1920, 1080][n],
        SetCursorPos=Mock(return_value=1),
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


def test_mac_adapter_native_calls_are_mocked_and_permission_gated(monkeypatch):
    q = SimpleNamespace(
        CGDisplayBounds=lambda _: SimpleNamespace(
            origin=SimpleNamespace(x=0, y=0), size=SimpleNamespace(width=1440, height=900)
        ),
        CGMainDisplayID=lambda: 1,
        CGEventCreateMouseEvent=Mock(return_value="event"),
        CGEventPost=Mock(),
        CGEventCreate=lambda _: "read-event",
        CGEventGetLocation=lambda _: (50, 60),
        kCGEventLeftMouseDragged=6,
        kCGEventMouseMoved=5,
        kCGEventLeftMouseDown=1,
        kCGEventLeftMouseUp=2,
        kCGMouseButtonLeft=0,
        kCGHIDEventTap=0,
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
    output.emit(InputEvent("down", 0.5, 0.5))
    output.emit(InputEvent("move", 0.6, 0.5))
    assert q.CGEventCreateMouseEvent.call_args.args[1] == 6
    output.close()
    assert q.CGEventCreateMouseEvent.call_args.args[1:3] == (2, (50, 60))
