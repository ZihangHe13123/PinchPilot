import json
import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .domain import EngineResult, HandFrame

CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (17, 0),
]


def draw_progress(painter, point, progress):
    if progress is None:
        return
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#ffce83"), 3))
    painter.drawArc(
        QRectF(point.x() - 16, point.y() - 16, 32, 32), 90 * 16, -int(360 * 16 * progress)
    )


class CameraView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 300)
        self.image = None
        self.frame = None
        self.result = None
        self.demo = False
        self.box = (0.2, 0.2, 0.8, 0.8)

    def set_frame(self, rgb, frame: HandFrame, result: EngineResult, demo: bool = False):
        if rgb is not None:
            h, w, _ = rgb.shape
            self.image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        else:
            self.image = None
        self.frame, self.result, self.demo = frame, result, demo
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101e2d"))
        aspect = self.frame.aspect if self.frame else 4 / 3
        w = min(self.width(), self.height() * aspect)
        h = w / aspect
        area = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        if self.image is not None:
            p.drawImage(area, self.image)
            p.fillRect(area, QColor(8, 15, 25, 35))
        else:
            p.setPen(QPen(QColor("#203344"), 1))
            for i in range(1, 10):
                p.drawLine(
                    QPointF(area.left() + i * w / 10, area.top()),
                    QPointF(area.left() + i * w / 10, area.bottom()),
                )
                p.drawLine(
                    QPointF(area.left(), area.top() + i * h / 10),
                    QPointF(area.right(), area.top() + i * h / 10),
                )
        left, top, right, bottom = self.box
        p.save()
        p.setClipRect(area)
        p.setPen(QPen(QColor("#73b6c7"), 1, Qt.PenStyle.DashLine))
        p.drawRoundedRect(
            QRectF(
                area.left() + left * w, area.top() + top * h, (right - left) * w, (bottom - top) * h
            ),
            12,
            12,
        )
        p.restore()
        if self.frame and self.frame.landmarks:
            points = [
                QPointF(area.left() + x * w, area.top() + y * h) for x, y, _ in self.frame.landmarks
            ]
            p.setPen(QPen(QColor("#6de1d1"), 3))
            for a, b in CONNECTIONS:
                p.drawLine(points[a], points[b])
            p.setBrush(QColor("#d9fff5"))
            p.setPen(Qt.PenStyle.NoPen)
            for i, point in enumerate(points):
                p.drawEllipse(point, 5 if i in (4, 8) else 3, 5 if i in (4, 8) else 3)
            p.setPen(QPen(QColor("#ffce83"), 3))
            if self.result and self.result.mode != "pinch":
                for a, b in ((5, 6), (6, 7), (7, 8)):
                    p.drawLine(points[a], points[b])
            else:
                p.drawLine(points[4], points[8])
        elif self.image is None:
            p.setPen(QColor("#b5c8d9"))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "让手势成为输入\n\n启动摄像头，或先体验无相机演示",
            )
        p.setPen(QColor("#d9e8ef"))
        p.drawText(
            20,
            28,
            "合成动作演示 · 不代表识别效果"
            if self.demo
            else (
                "镜像预览 · 食指相对掌部定位"
                if self.result and self.result.mode != "pinch"
                else "镜像预览 · 虚线内映射至主屏"
            ),
        )
        if self.result and self.result.pointer:
            x, y = self.result.pointer
            point = QPointF(
                area.left() + (left + x * (right - left)) * w,
                area.top() + (top + y * (bottom - top)) * h,
            )
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.setBrush(
                QColor("#edac55") if self.result.state in ("PRESSED", "DRAG") else QColor("#087e83")
            )
            p.setClipRect(area)
            p.drawEllipse(point, 10, 10)
            draw_progress(p, point, self.result.progress)
        p.end()


class PracticeView(QWidget):
    progress = Signal(str)
    finished = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 300)
        self.active = False
        self.virtual = True
        self.pointer = (0.5, 0.5)
        self.pressed = False
        self.dragging = False
        self.task = "click"
        self.index = 0
        self.misses = 0
        self.rows = []
        self.path = None
        self.metadata = {}
        self.feedback_progress = None
        self.feedback_hint = ""
        self.target_list = [
            (0.75, 0.25),
            (0.25, 0.70),
            (0.80, 0.75),
            (0.30, 0.25),
            (0.50, 0.75),
            (0.80, 0.45),
            (0.20, 0.40),
            (0.50, 0.25),
        ]

    def start(self, directory: Path, metadata: dict, virtual: bool = True, task: str = "click"):
        if self.active:
            self.stop()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"task_{time.time_ns()}.jsonl"
        self.metadata = dict(
            metadata,
            task=task,
            virtual=virtual,
            evidence="interaction task; synthetic runs are engineering checks only",
        )
        self.path.write_text(
            json.dumps({"type": "metadata", **self.metadata}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.active, self.virtual, self.task = True, virtual, task
        self.index, self.misses, self.rows = 0, 0, []
        self.pressed = self.dragging = False
        self.pointer = tuple(metadata.get("initial_pointer") or (0.5, 0.5))
        self.feedback_progress = None
        self.started = self.trial_started = time.monotonic()
        self.progress.emit("第 1 / 8 个目标")
        self.update()

    def _target(self):
        return self.target_list[self.index % len(self.target_list)]

    def _inside(self, point, target, radius=24):
        return (
            math.hypot(
                (point[0] - target[0]) * self.width(), (point[1] - target[1]) * self.height()
            )
            <= radius
        )

    def _event(self, kind: str, point):
        self.pointer = point
        if not self.active:
            self.update()
            return
        if kind == "down":
            self.pressed = True
            self.press_point = point
            if self.task == "drag":
                self.dragging = self._inside(point, (0.35, 0.5), 28)
        elif kind == "up" and self.pressed:
            self.pressed = False
            if self.task == "click":
                hit = self._inside(point, self._target()) and self._inside(
                    self.press_point, self._target()
                )
            else:
                hit = self.dragging and self._inside(point, self._target(), 30)
            self.dragging = False
            now = time.monotonic()
            row = {
                "type": "attempt",
                "trial": self.index + 1,
                "hit": hit,
                "elapsed_s": now - self.trial_started,
                "pointer": point,
                "target": self._target(),
                "viewport": [self.width(), self.height()],
                "radius_px": 24 if self.task == "click" else 30,
            }
            self.rows.append(row)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if hit:
                self.index += 1
                self.trial_started = now
            else:
                self.misses += 1
            if self.index == len(self.target_list):
                self.stop(completed=True)
            else:
                self.progress.emit(f"第 {self.index + 1} / 8 个目标 · 失误 {self.misses}")
        self.update()

    def feed(self, result: EngineResult):
        if not self.virtual:
            return
        self.feedback_progress = result.progress
        self.feedback_hint = result.hint
        if result.pointer is not None:
            self.pointer = result.pointer
        for event in result.events:
            if event.kind in ("move", "down", "up"):
                self._event(event.kind, (event.x, event.y))
        self.update()

    def cancel_press(self):
        self.pressed = self.dragging = False
        self.feedback_progress = None
        self.feedback_hint = ""
        self.update()

    def stop(self, completed=False):
        if not self.active:
            return
        self.active = False
        self.cancel_press()
        summary = {
            "type": "summary",
            "completed": completed,
            "hits": self.index,
            "misses": self.misses,
            "duration_s": time.monotonic() - self.started,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        message = f"{'完成' if completed else '已结束'} · 命中 {self.index}/8 · 失误 {self.misses} · {summary['duration_s']:.1f} 秒"
        self.progress.emit(message)
        self.finished.emit(str(self.path))
        self.update()

    def mousePressEvent(self, event):
        if not self.virtual and event.button() == Qt.MouseButton.LeftButton:
            self._event(
                "down", (event.position().x() / self.width(), event.position().y() / self.height())
            )

    def mouseReleaseEvent(self, event):
        if not self.virtual and event.button() == Qt.MouseButton.LeftButton:
            self._event(
                "up", (event.position().x() / self.width(), event.position().y() / self.height())
            )

    def mouseMoveEvent(self, event):
        if not self.virtual:
            self._event(
                "move", (event.position().x() / self.width(), event.position().y() / self.height())
            )

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101e2d"))
        p.setPen(QColor("#b5c8d9"))
        if not self.active:
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "同一任务，比较不同输入方式\n\n点击目标或拖动方块，结果会保存到本地",
            )
        else:
            x, y = self._target()
            target = QPointF(x * self.width(), y * self.height())
            p.setPen(QPen(QColor("#6de1d1"), 3))
            p.setBrush(QColor("#174f54"))
            radius = 24 if self.task == "click" else 30
            p.drawEllipse(target, radius, radius)
            p.setPen(QColor("white"))
            p.drawText(
                QRectF(target.x() - 20, target.y() - 15, 40, 30),
                Qt.AlignmentFlag.AlignCenter,
                str(self.index + 1),
            )
            if self.task == "drag":
                px, py = self.pointer if self.dragging else (0.35, 0.5)
                p.setBrush(QColor("#edac55"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(
                    QRectF(px * self.width() - 20, py * self.height() - 20, 40, 40), 7, 7
                )
        if self.virtual:
            x, y = self.pointer
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.setBrush(QColor("#ffce83") if self.pressed else QColor("#299cac"))
            point = QPointF(x * self.width(), y * self.height())
            p.drawEllipse(point, 8, 8)
            draw_progress(p, point, self.feedback_progress)
            if self.feedback_hint:
                p.setPen(QColor("#c0d6e2"))
                p.drawText(
                    QRectF(16, 12, self.width() - 32, 52),
                    Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop,
                    self.feedback_hint,
                )
        p.end()
