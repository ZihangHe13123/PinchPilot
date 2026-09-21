"""Local performance/event summaries only: no frames, landmarks or desktop context."""

import json
import math
import time
from collections import Counter, deque
from datetime import datetime, timezone

from . import __version__


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
                metric_scope="processed frames and native API sends; not recognition accuracy",
            )
        except OSError as error:
            self.error = f"无法写入测试记录：{error}"
            self._close_file()

    def observe(self, packet, result, now):
        self.frames.append(now)
        self.last_capture = packet.captured_at
        if math.isfinite(packet.inference_ms) and packet.inference_ms >= 0:
            self.inference.append(packet.inference_ms)
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
        }

    def sample(self, now, source, control):
        if now - self.last_sample < 1:
            return
        self.last_sample = now
        self.record("sample", source=source, control=control, **self.snapshot(now))
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
