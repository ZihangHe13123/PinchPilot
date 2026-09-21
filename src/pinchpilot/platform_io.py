"""Platform adapters are imported only after the user enables OS output."""

import ctypes
import sys

from .domain import InputEvent


def enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        try:
            # Must run before QApplication; native cursor coordinates are physical pixels.
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass


class MouseOutput:
    def __init__(self):
        self.down = False
        self.scroll_remainder = 0.0
        if sys.platform == "darwin":
            import Quartz
            from ApplicationServices import AXIsProcessTrusted

            self.q = Quartz
            if not AXIsProcessTrusted():
                raise PermissionError(
                    "请在系统设置 → 隐私与安全性 → 辅助功能中授权启动本程序的 Terminal / 应用，然后重新启动"
                )
            if (
                hasattr(Quartz, "CGPreflightListenEventAccess")
                and not Quartz.CGPreflightListenEventAccess()
            ):
                raise PermissionError(
                    "全局 Esc 需要输入监控权限；请在隐私与安全性 → 输入监控中授权后重新启动"
                )
            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            self.bounds = (bounds.origin.x, bounds.origin.y, bounds.size.width, bounds.size.height)
        elif sys.platform == "win32":
            self.user32 = ctypes.windll.user32
            self.bounds = (0, 0, self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1))
            self._prepare_windows()
        else:
            raise RuntimeError("真实控制当前支持 macOS / Windows；其他系统可运行预览和数据工具")

    def _prepare_windows(self) -> None:
        # Win32 LONG/DWORD stay 32-bit on x64 (LLP64), unlike host C long on macOS.
        # Explicit sizes also allow ABI contract tests on non-Windows hosts.
        class MouseInput(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_int32),
                ("dy", ctypes.c_int32),
                ("mouseData", ctypes.c_uint32),
                ("dwFlags", ctypes.c_uint32),
                ("time", ctypes.c_uint32),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_uint16),
                ("wScan", ctypes.c_uint16),
                ("dwFlags", ctypes.c_uint32),
                ("time", ctypes.c_uint32),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class HardwareInput(ctypes.Structure):
            _fields_ = [
                ("uMsg", ctypes.c_uint32),
                ("wParamL", ctypes.c_uint16),
                ("wParamH", ctypes.c_uint16),
            ]

        class Union(ctypes.Union):
            _fields_ = [("mi", MouseInput), ("ki", KeyboardInput), ("hi", HardwareInput)]

        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_uint32), ("u", Union)]

        self.win_input, self.win_mouse = Input, MouseInput
        self.user32.SendInput.argtypes = (ctypes.c_uint32, ctypes.POINTER(Input), ctypes.c_int)
        self.user32.SendInput.restype = ctypes.c_uint32

    def _send_win(self, flag: int, data: int = 0) -> None:
        item = self.win_input()
        item.type = 0
        item.u.mi = self.win_mouse(0, 0, data & 0xFFFFFFFF, flag, 0, 0)
        if self.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(item)) != 1:
            raise OSError("Windows 拒绝输入注入；请在普通权限窗口测试，避免管理员窗口")

    def emergency_pressed(self) -> bool:
        if sys.platform == "win32":
            return bool(self.user32.GetAsyncKeyState(0x1B) & 0x8000)
        return bool(
            self.q.CGEventSourceKeyState(self.q.kCGEventSourceStateCombinedSessionState, 53)
        )

    def position(self) -> tuple[float, float]:
        """Read the cursor; never emit an event. Coordinates are primary-screen normalized."""
        if sys.platform == "darwin":
            point = self.q.CGEventGetLocation(self.q.CGEventCreate(None))
            x, y = point
        else:

            class Point(ctypes.Structure):
                _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]

            point = Point()
            self.user32.GetCursorPos.argtypes = (ctypes.POINTER(Point),)
            self.user32.GetCursorPos.restype = ctypes.c_int32
            if not self.user32.GetCursorPos(ctypes.byref(point)):
                raise OSError("Windows 无法读取当前光标位置")
            x, y = point.x, point.y
        left, top, width, height = self.bounds
        return (
            min(1.0, max(0.0, (x - left) / max(width - 1, 1))),
            min(1.0, max(0.0, (y - top) / max(height - 1, 1))),
        )

    def emit(self, event: InputEvent) -> None:
        left, top, width, height = self.bounds
        x, y = left + event.x * (width - 1), top + event.y * (height - 1)
        if event.kind == "scroll":
            self.scroll_remainder += event.value
            lines = int(self.scroll_remainder)
            if not lines:
                return
            self.scroll_remainder -= lines
            if sys.platform == "darwin":
                obj = self.q.CGEventCreateScrollWheelEvent(
                    None, self.q.kCGScrollEventUnitLine, 1, lines
                )
                self.q.CGEventPost(self.q.kCGHIDEventTap, obj)
            else:
                self._send_win(0x0800, lines * 120)
            return
        if event.kind not in ("move", "down", "up"):
            return
        if sys.platform == "darwin":
            q = self.q
            kind = {
                "move": q.kCGEventLeftMouseDragged if self.down else q.kCGEventMouseMoved,
                "down": q.kCGEventLeftMouseDown,
                "up": q.kCGEventLeftMouseUp,
            }[event.kind]
            obj = q.CGEventCreateMouseEvent(None, kind, (x, y), q.kCGMouseButtonLeft)
            q.CGEventPost(q.kCGHIDEventTap, obj)
        else:
            if event.kind == "move":
                if not self.user32.SetCursorPos(round(x), round(y)):
                    raise OSError("Windows 无法移动光标")
            else:
                self._send_win(0x0002 if event.kind == "down" else 0x0004)
        if event.kind in ("down", "up"):
            self.down = event.kind == "down"

    def close(self) -> None:
        if self.down:
            if sys.platform == "darwin":
                q = self.q
                point = q.CGEventGetLocation(q.CGEventCreate(None))
                obj = q.CGEventCreateMouseEvent(
                    None, q.kCGEventLeftMouseUp, point, q.kCGMouseButtonLeft
                )
                q.CGEventPost(q.kCGHIDEventTap, obj)
            else:
                self._send_win(0x0004)
            self.down = False


def camera_permission() -> str:
    if sys.platform != "darwin":
        return "请允许桌面应用访问摄像头；Windows 设置 → 隐私和安全性 → 摄像头"
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

        value = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        return {
            0: "尚未请求摄像头权限",
            1: "摄像头访问受限制",
            2: "摄像头权限已拒绝",
            3: "摄像头权限已授权",
        }.get(value, "未知权限状态")
    except ImportError:
        return "AVFoundation 未安装；请运行 uv sync"


def request_camera_access(completion) -> None:
    """Call from the main/UI thread; callback may run on an AVFoundation thread."""
    if sys.platform != "darwin":
        completion(True)
        return
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

    state = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    if state == 0:
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeVideo, lambda allowed: completion(bool(allowed))
        )
    else:
        completion(state == 3)
