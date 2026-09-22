"""Camera freshness, gesture engine and a guarded native session; independent of Qt."""

import math
import time
from collections import Counter
from dataclasses import asdict, replace

from .diagnostics import inspect_engine
from .framing import FramingMonitor, FramingStatus, framing_hint
from .motion_calibration import RestNoiseCalibration
from .mouse_session import MouseSession
from .platform_io import MouseOutput
from .trial_metrics import TrialMetrics
from .tripod import TripodEngine
from .wrist import WristCalibration


class DesktopController:
    FRAME_TIMEOUT = 0.25

    def __init__(self, directory, clock=time.monotonic, output_factory=None, background=True):
        self.clock = clock
        self.output_factory = output_factory or MouseOutput
        self.background = background
        self.engine = TripodEngine()
        self.metrics = TrialMetrics(directory, clock)
        self.session = None
        self.source = "none"
        self.last_capture = None
        self.result = self.engine.last_result
        self.sent_seen = Counter()
        self.stop_seen = ""
        self.notice = "启动相机预览，再启用鼠标控制。Esc 随时停止。"
        self.calibration = None
        self.calibration_result = None
        self.wrist_capture = None
        self.wrist_calibration_result = None
        self.wrist_calibration_error = ""
        self.framing = FramingMonitor()
        self.practice_active = False
        self.last_tick = None

    @property
    def framing_status(self):
        if self.source != "camera" or not self.fresh():
            return FramingStatus()
        return self.framing.status

    @property
    def framing_hint(self):
        return framing_hint(self.framing_status, self.result.state)

    @property
    def calibrating(self):
        return self.calibration is not None or self.wrist_capture is not None

    @property
    def calibration_progress(self):
        return self.calibration.progress(self.clock()) if self.calibration else 0.0

    @property
    def wrist_calibration_progress(self):
        return self.wrist_capture.progress(self.clock()) if self.wrist_capture else 0.0

    def start_calibration(self):
        if self.source != "camera" or not self.fresh():
            raise RuntimeError("先启动实时相机预览，再进行静止校准")
        if self.engine.config.pointer_basis != "position":
            raise RuntimeError(
                "静止噪声校准仅用于指尖位置对照；兼容定位无需此校准，腕动请使用三步方向校准"
            )
        if self.engine.config.motion_profile == "classic":
            raise RuntimeError("静止校准用于精细 / 自适应模式，请先切换移动模式")
        if not self.stop("calibration_started"):
            raise RuntimeError("上一次松键仍在重试，暂不能校准")
        self.calibration_result = None
        self.calibration = RestNoiseCalibration(self.clock(), self.engine.config)
        self.notice = (
            "准备 1 秒、静止采集 5 秒：支撑前臂，拇中捏住，食指和无名指移开。鼠标控制已关闭。"
        )

    def cancel_calibration(self, reason="user_cancelled"):
        if self.calibrating:
            wrist = self.wrist_capture is not None
            measurement = self.wrist_capture
            label = "腕动方向" if wrist else "静止"
            self.calibration = None
            self.wrist_capture = None
            self.notice = f"{label}校准已取消，保留原设置；鼠标控制保持关闭。"
            if wrist:
                cause = {
                    "stale_camera": "相机画面中断",
                    "invalid_or_stale_frame": "相机帧无效或过期",
                    "source_changed": "画面来源已切换",
                    "settings_changed": "控制设置已改变",
                    "window_closed": "窗口已关闭",
                }.get(reason, "已取消此次校准")
                self.wrist_calibration_error = (
                    f"腕动方向校准未完成：{cause}。"
                    f"最后采集情况：{measurement.hint(self.clock())}。"
                    "保留原设置；鼠标控制保持关闭。"
                )
                self.notice = self.wrist_calibration_error
                self.metrics.record(
                    "wrist_calibration_cancelled",
                    reason=reason,
                    diagnostics=self._wrist_diagnostics(measurement),
                )

    @staticmethod
    def _wrist_diagnostics(measurement):
        """Read only the public geometry summary, never capture frames or coordinates."""
        diagnostics = getattr(measurement, "diagnostics", None)
        try:
            summary = diagnostics() if callable(diagnostics) else {}
        except Exception:
            return {"unavailable": True}
        return summary if isinstance(summary, dict) else {}

    def start_wrist_calibration(self):
        if self.source != "camera" or not self.fresh():
            raise RuntimeError("先启动实时相机预览，再进行腕动方向校准")
        if self.engine.config.pointer_basis != "wrist":
            raise RuntimeError("请先将定位方式切换为腕动角度 · 旧实验")
        if not self.stop("wrist_calibration_started"):
            raise RuntimeError("上一次松键仍在重试，暂不能校准")
        self.wrist_calibration_error = ""
        self.wrist_calibration_result = None
        self.wrist_capture = WristCalibration(self.clock(), self.engine.config)
        self.metrics.record("wrist_calibration_started")
        self.notice = "腕动方向校准已开始，鼠标控制保持关闭。" + self.wrist_capture.hint(
            self.clock()
        )

    def _finish_wrist_calibration(self):
        measurement, self.wrist_capture = self.wrist_capture, None
        try:
            values = tuple(measurement.result())
            if len(values) != 15:
                raise ValueError("腕动方向校准结果不完整")
            config = replace(self.engine.config, wrist_calibration=values)
            config.validate()
        except (ValueError, TypeError) as error:
            retained = (
                "保留此前校准；本次结果未应用。"
                if self.engine.config.wrist_calibration
                else "本次未生成校准，请按上述原因调整后重试。"
            )
            self.wrist_calibration_error = (
                f"此次腕动方向校准未通过：{error}。{retained}鼠标控制保持关闭。"
            )
            self.notice = self.wrist_calibration_error
            self.metrics.record(
                "wrist_calibration_failed",
                reason=str(error),
                diagnostics=self._wrist_diagnostics(measurement),
            )
            return
        self.configure(config)
        self.wrist_calibration_error = ""
        self.wrist_calibration_result = values
        self.metrics.record(
            "wrist_calibration",
            parameters=list(values),
            scope="neutral and demonstrated directions",
        )
        self.notice = "腕动方向校准已应用；转动一段、光标移动一段，停腕即停。请手动启用鼠标控制。"

    def _finish_calibration(self):
        measurement, self.calibration = self.calibration, None
        try:
            result = measurement.result()
        except ValueError as error:
            self.notice = f"静止校准未应用：{error}"
            return
        config = replace(
            self.engine.config,
            rest_noise_x=result["rest_noise_x"],
            rest_noise_y=result["rest_noise_y"],
        )
        self.configure(config)
        self.calibration_result = result
        self.metrics.record("rest_noise_calibration", **result)
        self.notice = f"静止校准已应用（{result['samples']} 帧）；需要时重新启用鼠标控制。" + (
            "波动偏大，已限制抗抖范围，建议改善手部支撑或机位。" if result["limited"] else ""
        )

    @property
    def active(self):
        return bool(self.session and self.session.snapshot()["active"])

    @property
    def pending_release(self):
        return bool(self.session and self.session.snapshot()["pending_release"])

    def fresh(self, now=None):
        now = self.clock() if now is None else now
        return self.last_capture is not None and 0 <= now - self.last_capture < self.FRAME_TIMEOUT

    def enable(self):
        if self.practice_active:
            raise RuntimeError("测试台只控制虚拟光标；关闭测试台后再手动启用系统鼠标")
        if self.calibrating:
            raise RuntimeError("校准期间不能接管鼠标；请等待完成或取消校准")
        if not self.stop("re-enable"):
            raise RuntimeError("上一次松键仍在重试，暂不能启用")
        if self.source != "camera" or not self.fresh():
            raise RuntimeError("需要实时相机画面；演示、停流或过期画面不能控制鼠标")
        if self.engine.config.pointer_basis == "wrist" and not self.engine.config.wrist_calibration:
            raise RuntimeError("腕动模式尚未校准；请先完成三步腕动方向校准")
        self.session = None
        output = self.output_factory()
        self.session = MouseSession(output, self.clock, self.background)
        self.sent_seen = Counter()
        self.stop_seen = ""
        self.engine.pointer = self.session.position()
        self.result = self.engine.reset()
        self.notice = "鼠标控制已启用 · 拇中捏住接管 · Esc 停止"
        self.metrics.record("control_started", config=asdict(self.engine.config))

    def begin_practice(self):
        if not self.stop("practice_started"):
            raise RuntimeError("上一次松键仍在重试，暂不能打开测试台")
        self.practice_active = True
        self.engine.pointer = (0.5, 0.5)
        self.result = self.engine.reset()
        self.metrics.record("practice_started", source=self.source)
        self.notice = "测试台已打开 · 手势仅控制测试台里的虚拟光标。"

    def end_practice(self):
        self.stop("practice_closed")
        self.practice_active = False
        self.metrics.record("practice_closed")
        self.notice = "测试台已关闭；需要时请手动重新启用鼠标控制。"

    def mark_issue(self, category):
        path = self.metrics.mark_issue(category, self.source, asdict(self.engine.config))
        self.notice = f"已标记刚才的问题（最多 10 秒诊断摘要）：{path.name}"
        return path

    def _diagnostic(self, now, packet=None, reason=None, label=None):
        self.metrics.context_source = self.source
        self.metrics.context_control = self.active
        value = inspect_engine(self.engine, self.result, packet)
        if reason:
            value = {
                "reason": reason,
                "label": label or reason,
                "state": self.result.state,
                "values": {},
            }
        self.metrics.observe_diagnostic(value, now)

    def _sync_session(self):
        if self.session is None:
            return
        current = self.session.snapshot()
        counts = Counter(current["sent"])
        delta = counts - self.sent_seen
        self.metrics.sent.update(delta)
        buttons = {k: v for k, v in delta.items() if k not in ("move", "scroll")}
        if buttons:
            self.metrics.record("native_buttons", counts=buttons)
        if delta.get("scroll"):
            self.metrics.record("native_scroll", count=delta["scroll"])
        self.sent_seen = counts
        if not current["active"] and self.stop_seen != current["reason"]:
            self.stop_seen = current["reason"]
            self.result = self.engine.reset()
            self.metrics.record("control_stopped", **current)
            labels = {
                "global_escape": "已按 Esc 停止鼠标控制",
                "gui_heartbeat_timeout": "界面响应超时，鼠标控制已停止",
                "stale_camera": "相机画面已过期，鼠标控制已停止",
            }
            self.notice = labels.get(current["reason"], "鼠标控制已关闭")
        if current["error"]:
            self.notice = current["error"]

    def stop(self, reason="user_stop"):
        self.cancel_calibration(reason)
        released = True
        if self.session:
            released = self.session.stop(reason)
            self._sync_session()
        self.result = self.engine.reset()
        return released

    def set_source(self, source):
        if source not in ("none", "camera", "synthetic_demo"):
            raise ValueError("未知画面来源")
        self.stop("source_changed")
        self.source = source
        self.framing.reset()
        self.last_capture = None
        # An intentional source change is not a tracking failure.
        self.metrics.tracked = False
        self.metrics.frames.clear()
        self.metrics.inference.clear()
        self.metrics.last_capture = None
        self.metrics.reset_source(source)
        self.last_tick = None
        self.metrics.record("source", source=source)

    def configure(self, config):
        if config.pointer_basis == "unified":
            config = replace(config, motion_profile="precise", rest_noise_x=0.0, rest_noise_y=0.0)
        elif config.pointer_basis == "wrist":
            config = replace(config, rest_noise_x=0.0, rest_noise_y=0.0)
        config.validate()
        self.calibration_result = None
        self.wrist_calibration_result = None
        self.stop("settings_changed")
        self.engine = TripodEngine(config)
        self.result = self.engine.last_result
        self.metrics.record("configuration", config=asdict(config))

    def consume(self, packet):
        now = self.clock()
        t = packet.captured_at
        if self.active and self.source != "camera":
            self.stop("source_changed")
        if (
            self.source == "none"
            or not math.isfinite(t)
            or not math.isfinite(packet.frame.timestamp)
            or not 0 <= now - t < self.FRAME_TIMEOUT
            or abs(packet.frame.timestamp - t) > 0.001
            or (self.last_capture is not None and t <= self.last_capture)
        ):
            self.metrics.dropped += 1
            self.framing.reset()
            self.stop("invalid_or_stale_frame")
            self._diagnostic(
                now, reason="invalid_or_stale_frame", label="帧无效、重复或过期，控制已停止"
            )
            return False
        self.last_capture = t
        if self.calibration is not None:
            self.calibration.add(packet.frame)
        if self.wrist_capture is not None:
            self.wrist_capture.add(packet.frame)
        if self.session:
            self.session.heartbeat()
            self._sync_session()
        if self.active and self.engine.state == "WAIT_GRIP":
            # Reacquisition follows the real cursor if a physical mouse was used in between.
            self.engine.pointer = self.session.position()
        previous = self.engine.state
        engine_started = time.perf_counter()
        self.result = self.engine.process(packet.frame)
        self.metrics.observe_timing("engine", (time.perf_counter() - engine_started) * 1000)
        if self.source == "camera":
            self.framing.update(packet.frame, tracked=self.result.grip is not None)
        self.metrics.context_source = self.source
        self.metrics.observe(packet, self.result, now)
        if self.active:
            send_started = time.perf_counter()
            self.session.emit(self.result.events)
            if self.result.events:
                self.metrics.observe_timing(
                    "native_emit", (time.perf_counter() - send_started) * 1000
                )
            if previous != "DRAG" and self.result.state == "DRAG" and self.active:
                self.metrics.drags += 1
            self._sync_session()
        self._diagnostic(now, packet)
        return True

    def tick(self):
        now = self.clock()
        if self.last_tick is not None:
            self.metrics.observe_timing("gui_interval", (now - self.last_tick) * 1000)
        self.last_tick = now
        if self.wrist_capture is not None:
            if not self.fresh(now) or self.source != "camera":
                self.cancel_calibration("stale_camera")
                self.notice = "相机画面中断，腕动方向校准未应用；保留原设置。"
            elif self.wrist_capture.complete:
                self._finish_wrist_calibration()
        if self.calibration is not None:
            if not self.fresh(now) or self.source != "camera":
                self.cancel_calibration()
                self.notice = "相机画面中断，静止校准未应用；保留原设置。"
            elif self.calibration_progress >= 1.0:
                self._finish_calibration()
        if self.session:
            self.session.heartbeat()
            self._sync_session()
        if self.active and (self.source != "camera" or not self.fresh(now)):
            self.stop("stale_camera")
        self.result = self.engine.tick(now)
        if self.active:
            self.session.emit(self.result.events)
            self._sync_session()
        if not self.fresh(now):
            self.framing.reset()
            self.metrics.no_tracking()
            self._diagnostic(
                now,
                reason="stale_camera" if self.source == "camera" else "idle",
                label="画面过期或等待相机" if self.source == "camera" else "相机关闭",
            )
        self.metrics.sample(now, self.source, self.active)

    def close(self):
        if not self.stop("window_closed"):
            return False
        self.metrics.close()
        return True
