"""Serialized native output with an Esc/GUI-heartbeat watchdog independent of Qt."""

import threading
import time
from collections import Counter


class MouseSession:
    HEARTBEAT_TIMEOUT = 0.40

    def __init__(self, output, clock=time.monotonic, background=True):
        self.output, self.clock = output, clock
        self.lock = threading.RLock()
        self.finished = threading.Event()
        self.active = True
        self.reason = ""
        self.error = ""
        self.pending_release = False
        self.sent = Counter()
        self.last_heartbeat = clock()
        self.thread = None
        if background:
            self.thread = threading.Thread(
                target=self._watch, name="pinchpilot-mouse-watchdog", daemon=True
            )
            self.thread.start()

    def _watch(self):
        while not self.finished.wait(0.04):
            self.poll()

    def poll(self):
        with self.lock:
            if self.pending_release:
                self._release()
            elif self.active:
                try:
                    if self.output.emergency_pressed():
                        self.stop("global_escape")
                    elif self.clock() - self.last_heartbeat >= self.HEARTBEAT_TIMEOUT:
                        self.stop("gui_heartbeat_timeout")
                except Exception as error:
                    self.stop("watchdog_error", str(error))

    def heartbeat(self):
        # Check the old deadline before renewing: a stalled UI must not revive output.
        self.poll()
        with self.lock:
            if self.active:
                self.last_heartbeat = self.clock()

    def emit(self, events):
        with self.lock:
            self.poll()
            if not self.active:
                return
            try:
                for event in events:
                    if self.output.emit(event) is not False:
                        self.sent[event.kind] += 1
            except Exception as error:
                self.stop("output_error", str(error))

    def position(self):
        with self.lock:
            if not self.active:
                raise RuntimeError("鼠标控制已停止")
            try:
                return self.output.position()
            except Exception as error:
                self.stop("position_error", str(error))
                raise

    def stop(self, reason, error=""):
        with self.lock:
            if self.active:
                self.reason = reason
            self.active = False
            if error:
                self.error = error
            if not self.finished.is_set():
                self._release()
            return not self.pending_release

    def _release(self):
        held = (self.output.down, self.output.right_down)
        try:
            self.output.close()
            if self.pending_release:
                self.error = "松键已恢复；鼠标控制保持关闭"
            self.pending_release = False
            self.finished.set()
        except Exception as error:
            self.pending_release = True
            self.error = f"松键失败，正在重试：{error}"
        finally:
            # Include successful cleanup releases, never count a failed native call.
            for was_held, still_held, kind in zip(
                held, (self.output.down, self.output.right_down), ("up", "right_up")
            ):
                if was_held and not still_held:
                    self.sent[kind] += 1

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active,
                "reason": self.reason,
                "error": self.error,
                "pending_release": self.pending_release,
                "left_held": self.output.down,
                "right_held": self.output.right_down,
                "sent": dict(self.sent),
            }
