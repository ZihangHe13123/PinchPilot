"""Camera freshness, gesture engine and a guarded native session; independent of Qt."""

import math
import time
from collections import Counter
from dataclasses import asdict, replace

from .motion_calibration import RestNoiseCalibration
from .mouse_session import MouseSession
from .platform_io import MouseOutput
from .trial_metrics import TrialMetrics
from .tripod import TripodEngine


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

    @property
    def calibrating(self):
        return self.calibration is not None

    @property
    def calibration_progress(self):
        return self.calibration.progress(self.clock()) if self.calibration else 0.0

    def start_calibration(self):
        if self.source != "camera" or not self.fresh():
            raise RuntimeError("先启动实时相机预览，再进行静止校准")
        if self.engine.config.motion_profile == "classic":
            raise RuntimeError("静止校准用于精细 / 自适应模式，请先切换移动模式")
        if not self.stop("calibration_started"):
            raise RuntimeError("上一次松键仍在重试，暂不能校准")
        self.calibration_result = None
        self.calibration = RestNoiseCalibration(self.clock(), self.engine.config)
        self.notice = (
            "准备 1 秒、静止采集 5 秒：支撑前臂，拇中捏住，食指和无名指移开。鼠标控制已关闭。"
        )

    def cancel_calibration(self):
        if self.calibration is not None:
            self.calibration = None
            self.notice = "静止校准已取消，保留原设置；鼠标控制保持关闭。"

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
        if self.calibrating:
            raise RuntimeError("静止校准期间不能接管鼠标；请等待完成或取消校准")
        if not self.stop("re-enable"):
            raise RuntimeError("上一次松键仍在重试，暂不能启用")
        if self.source != "camera" or not self.fresh():
            raise RuntimeError("需要实时相机画面；演示、停流或过期画面不能控制鼠标")
        self.session = None
        output = self.output_factory()
        self.session = MouseSession(output, self.clock, self.background)
        self.sent_seen = Counter()
        self.stop_seen = ""
        self.engine.pointer = self.session.position()
        self.result = self.engine.reset()
        self.notice = "鼠标控制已启用 · 拇中捏住接管 · Esc 停止"
        self.metrics.record("control_started", config=asdict(self.engine.config))

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
        self.cancel_calibration()
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
        self.last_capture = None
        # An intentional source change is not a tracking failure.
        self.metrics.tracked = False
        self.metrics.frames.clear()
        self.metrics.inference.clear()
        self.metrics.last_capture = None
        self.metrics.record("source", source=source)

    def configure(self, config):
        config.validate()
        self.calibration_result = None
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
            self.stop("invalid_or_stale_frame")
            return False
        self.last_capture = t
        if self.calibration is not None:
            self.calibration.add(packet.frame)
        if self.session:
            self.session.heartbeat()
            self._sync_session()
        if self.active and self.engine.state == "WAIT_GRIP":
            # Reacquisition follows the real cursor if a physical mouse was used in between.
            self.engine.pointer = self.session.position()
        previous = self.engine.state
        self.result = self.engine.process(packet.frame)
        self.metrics.observe(packet, self.result, now)
        if self.active:
            self.session.emit(self.result.events)
            if previous != "DRAG" and self.result.state == "DRAG" and self.active:
                self.metrics.drags += 1
            self._sync_session()
        return True

    def tick(self):
        now = self.clock()
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
            self.metrics.no_tracking()
        self.metrics.sample(now, self.source, self.active)

    def close(self):
        if not self.stop("window_closed"):
            return False
        self.metrics.close()
        return True
