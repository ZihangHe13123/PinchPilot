"""Local performance/event summaries only: no frames, landmarks or desktop context."""

import json
import math
import time
from collections import Counter, deque
from datetime import datetime, timezone

from . import __version__

TIMING_LABELS = {
    "capture_read": "相机 read 调用",
    "preprocess": "镜像与颜色转换",
    "inference": "模型推理",
    "gui_wait": "推理完成到界面接收",
    "engine": "手势与光标计算",
    "native_emit": "本地鼠标发送调用",
    "gui_interval": "界面轮询间隔",
}


class TrialMetrics:
    def __init__(self, directory, clock=time.monotonic):
        self.directory, self.clock = directory, clock
        self.started = clock()
        self.frames = deque(maxlen=180)
        self.inference = deque(maxlen=180)
        self.sent = Counter()
        self.losses = self.dropped = self.drags = 0
        self.tracked = False
        self.last_capture = None
        self.state = "WAIT_GRIP"
        self.file = self.path = None
        self.error = ""
        self.last_sample = 0
        self.timings = {name: deque(maxlen=180) for name in TIMING_LABELS}
        self.overwritten = 0
        self.overwritten_seen = 0
        self.diagnostic = {"reason": "idle", "label": "等待画面", "state": self.state, "values": {}}
        self.reason_since = self.started
        self.reason_seconds = Counter()
        self.recent = deque(maxlen=400)
        self.last_context = -math.inf
        self.context_source = "none"
        self.context_control = False

    def observe_timing(self, name, milliseconds):
        if (
            name in self.timings
            and isinstance(milliseconds, (int, float))
            and math.isfinite(milliseconds)
            and milliseconds >= 0
        ):
            self.timings[name].append(float(milliseconds))

    def reset_source(self, source):
        for values in self.timings.values():
            values.clear()
        self.overwritten_seen = 0
        self.context_source = source
        self.context_control = False
        self.observe_diagnostic(
            {
                "reason": "source_changed",
                "label": "画面来源已切换",
                "state": "WAIT_GRIP",
                "values": {},
            },
            self.clock(),
        )

    def observe_diagnostic(self, diagnostic, now):
        changed = diagnostic["reason"] != self.diagnostic["reason"]
        if changed:
            duration = max(0.0, now - self.reason_since)
            self.reason_seconds[self.diagnostic["reason"]] += duration
            self.record(
                "diagnostic",
                previous_reason=self.diagnostic["reason"],
                previous_duration_s=round(duration, 3),
                **diagnostic,
            )
            self.reason_since = now
        self.diagnostic = diagnostic
        if changed or now - self.last_context >= 0.1:
            self.last_context = now
            self.recent.append(
                {
                    "elapsed_s": round(now - self.started, 3),
                    "source": self.context_source,
                    "control": self.context_control,
                    **diagnostic,
                    "timings_ms": {
                        key: round(values[-1], 3) for key, values in self.timings.items() if values
                    },
                }
            )
        self._trim_context(now)

    def _trim_context(self, now):
        while self.recent and now - self.started - self.recent[0]["elapsed_s"] > 10.0:
            self.recent.popleft()

    def mark_issue(self, category, source, config):
        allowed = {"jitter", "delay", "missed_click", "unintended_click", "tracking_loss", "other"}
        if category not in allowed:
            raise ValueError("未知的问题类型")
        now = self.clock()
        self._trim_context(now)
        payload = {
            "schema": "pinchpilot-issue-v1",
            "version": __version__,
            "category": category,
            "source": source,
            "config": config,
            "user_reported": True,
            "stores_images": False,
            "stores_keypoints": False,
            "context_window_s": 10,
            "context": list(self.recent),
            "summary": self.snapshot(now),
        }
        if self.file is not None:
            self.record("issue_marker", **payload)
            self._flush()
            if self.file is None:
                raise OSError(self.error)
            return self.path
        # This explicit user action does not turn regular logging back on.
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"issue_{time.time_ns()}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return path

    def record(self, kind, **values):
        if self.file is None:
            return
        try:
            self.file.write(
                json.dumps(
                    {"type": kind, "elapsed_s": round(self.clock() - self.started, 3), **values},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        except (OSError, ValueError) as error:
            self.error = f"测试记录已停止：{error}"
            self._close_file()

    def set_logging(self, enabled, config, source):
        self.close()
        if not enabled:
            return
        self.error = ""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / f"trial_{time.time_ns()}.jsonl"
            self.file = self.path.open("x", encoding="utf-8")
            self.record(
                "session",
                version=__version__,
                started_utc=datetime.now(timezone.utc).isoformat(),
                config=config,
                source=source,
                stores_images=False,
                stores_keypoints=False,
                metric_scope="processed frames and native API sends; not recognition accuracy",
                timing_scope="read-return and local software stages; not sensor-to-display latency",
            )
        except OSError as error:
            self.error = f"无法写入测试记录：{error}"
            self._close_file()

    def observe(self, packet, result, now):
        self.frames.append(now)
        self.last_capture = packet.captured_at
        if math.isfinite(packet.inference_ms) and packet.inference_ms >= 0:
            self.inference.append(packet.inference_ms)
        if self.context_source == "camera":
            self.observe_timing("inference", packet.inference_ms)
            self.observe_timing("capture_read", getattr(packet, "capture_ms", None))
            self.observe_timing("preprocess", getattr(packet, "preprocess_ms", None))
            ready = getattr(packet, "ready_at", None)
            if ready is not None and math.isfinite(ready) and packet.captured_at <= ready <= now:
                self.observe_timing("gui_wait", (now - ready) * 1000)
            overwritten = getattr(packet, "overwritten_results", 0)
            if isinstance(overwritten, int) and overwritten >= 0:
                self.overwritten += max(0, overwritten - self.overwritten_seen)
                self.overwritten_seen = overwritten
        tracked = result.grip is not None
        if self.tracked and not tracked:
            self.losses += 1
        self.tracked = tracked
        if result.state != self.state:
            self.record("state", state=result.state, cancelled=result.cancelled)
        self.state = result.state

    def no_tracking(self):
        if self.tracked:
            self.losses += 1
        self.tracked = False

    def snapshot(self, now):
        recent = [t for t in self.frames if now - t < 2]
        age = max(0, now - self.last_capture) if self.last_capture is not None else None
        fps = (
            (len(recent) - 1) / max(recent[-1] - recent[0], 0.001)
            if len(recent) > 1 and age is not None and age < 0.25
            else 0.0
        )
        ordered = sorted(self.inference)
        timings = {}
        for name, values in self.timings.items():
            ordered_values = sorted(values)
            timings[name] = {
                "samples": len(values),
                "last_ms": round(values[-1], 3) if values else None,
                "p50_ms": round(ordered_values[int((len(values) - 1) * 0.5)], 3)
                if values
                else None,
                "p95_ms": round(ordered_values[int((len(values) - 1) * 0.95)], 3)
                if values
                else None,
            }
        durations = dict(self.reason_seconds)
        durations[self.diagnostic["reason"]] = durations.get(self.diagnostic["reason"], 0) + max(
            0, now - self.reason_since
        )
        return {
            "elapsed_s": round(now - self.started, 1),
            "processed_fps": round(fps, 1),
            "inference_ms": round(self.inference[-1], 1) if self.inference else None,
            "inference_p95_ms": round(ordered[int((len(ordered) - 1) * 0.95)], 1)
            if ordered
            else None,
            "frame_age_ms": round(age * 1000, 1) if age is not None else None,
            "tracked": self.tracked and age is not None and age < 0.25,
            "tracking_losses": self.losses,
            "discarded_frames": self.dropped,
            "drag_starts": self.drags,
            "sent": dict(self.sent),
            "overwritten_results": self.overwritten,
            "diagnostic": self.diagnostic,
            "reason_duration_s": round(max(0, now - self.reason_since), 3),
            "reason_seconds": {key: round(value, 3) for key, value in durations.items()},
            "timings": timings,
        }

    def sample(self, now, source, control):
        if now - self.last_sample < 1:
            return
        self.last_sample = now
        self.record("sample", source=source, control=control, **self.snapshot(now))
        self._flush()

    def _flush(self):
        if self.file is not None:
            try:
                self.file.flush()
            except OSError as error:
                self.error = f"测试记录已停止：{error}"
                self._close_file()

    def _close_file(self):
        if self.file is not None:
            try:
                self.file.close()
            except OSError as error:
                self.error = f"测试记录关闭失败：{error}"
            self.file = None

    def close(self):
        self.record("session_end", summary=self.snapshot(self.clock()))
        self._close_file()
