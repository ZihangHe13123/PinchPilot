import json
import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen
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


def fit_to_screen(window, width, height):
    """Give a window its preferred size, but never larger than the screen it opens on.

    A 1080p laptop at 150% scaling has only about 680 usable pixels of height; a taller
    window would open with its lower controls off the screen.
    """
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = min(width, available.width() - 40)
        height = min(height, available.height() - 60)
    window.resize(width, height)


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
        self.frame_edges = ()

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
        if self.box is not None:
            left, top, right, bottom = self.box
            p.save()
            p.setClipRect(area)
            p.setPen(QPen(QColor("#73b6c7"), 1, Qt.PenStyle.DashLine))
            p.drawRoundedRect(
                QRectF(
                    area.left() + left * w,
                    area.top() + top * h,
                    (right - left) * w,
                    (bottom - top) * h,
                ),
                12,
                12,
            )
            p.restore()
        # Camera-frame edges are distinct from the pointer-mapping rectangle.
        strips = {
            "left": QRectF(area.left(), area.top(), 7, h),
            "right": QRectF(area.right() - 7, area.top(), 7, h),
            "top": QRectF(area.left(), area.top(), w, 7),
            "bottom": QRectF(area.left(), area.bottom() - 7, w, 7),
        }
        for edge in self.frame_edges:
            if edge in strips:
                p.fillRect(strips[edge], QColor("#ffc36b"))
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
            if self.result and self.result.mode == "tripod":
                middle = (points[4] + points[12]) / 2
                p.drawLine(points[4], points[12])
                p.setPen(QPen(QColor("#ffce83"), 2, Qt.PenStyle.DashLine))
                p.drawLine(points[8], points[4])
                p.setPen(QPen(QColor("#c1a1ff"), 2, Qt.PenStyle.DashLine))
                p.drawLine(points[16], points[4])
                p.setPen(QPen(QColor("#9eaebd"), 1))
                p.drawEllipse(middle, 5, 5)
            elif self.result and self.result.mode != "pinch":
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
                "腕动预览 · 稳定点表示屏幕位置"
                if self.box is None
                else "拇中中点：灰色原始点 · 蓝绿色稳定点"
                if self.result and self.result.mode == "tripod"
                else "镜像预览 · 食指相对掌部定位"
                if self.result and self.result.mode != "pinch"
                else "镜像预览 · 虚线内映射至主屏"
            ),
        )
        if self.result and self.result.pointer:
            left, top, right, bottom = self.box or (0.0, 0.0, 1.0, 1.0)
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
        self.press_button = None
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
        self.press_button = None
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
        button = "right" if kind.startswith("right_") else "left"
        if kind in ("down", "right_down"):
            if self.pressed:
                return
            self.pressed = True
            self.press_button = button
            self.press_point = point
            if self.task == "drag":
                self.dragging = button == "left" and self._inside(point, (0.35, 0.5), 28)
        elif kind in ("up", "right_up") and self.pressed and self.press_button == button:
            self.pressed = False
            self.press_button = None
            expected_button = "right" if self.task == "right_click" else "left"
            if self.task in ("click", "right_click"):
                hit = (
                    button == expected_button
                    and self._inside(point, self._target())
                    and self._inside(self.press_point, self._target())
                )
            else:
                hit = self.dragging and self._inside(point, self._target(), 30)
            self.dragging = False
            now = time.monotonic()
            row = {
                "type": "attempt",
                "trial": self.index + 1,
                "hit": hit,
                "button": button,
                "expected_button": expected_button,
                "elapsed_s": now - self.trial_started,
                "pointer": point,
                "target": self._target(),
                "viewport": [self.width(), self.height()],
                "radius_px": 30 if self.task == "drag" else 24,
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
                prefix = "右键命中 · " if hit and self.task == "right_click" else ""
                self.progress.emit(f"{prefix}第 {self.index + 1} / 8 个目标 · 失误 {self.misses}")
        self.update()

    def feed(self, result: EngineResult):
        if not self.virtual:
            return
        if result.cancelled:
            self.cancel_press()
        self.feedback_progress = result.progress
        self.feedback_hint = result.hint
        if result.pointer is not None:
            self.pointer = result.pointer
        for event in result.events:
            if event.kind in ("move", "down", "up", "right_down", "right_up"):
                self._event(event.kind, (event.x, event.y))
        self.update()

    def cancel_press(self):
        self.pressed = self.dragging = False
        self.press_button = None
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
        if not self.virtual and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._event(
                "right_down" if event.button() == Qt.MouseButton.RightButton else "down",
                (event.position().x() / self.width(), event.position().y() / self.height()),
            )

    def mouseReleaseEvent(self, event):
        if not self.virtual and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._event(
                "right_up" if event.button() == Qt.MouseButton.RightButton else "up",
                (event.position().x() / self.width(), event.position().y() / self.height()),
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
                "同一任务，比较不同输入方式\n\n选择左键、右键或拖拽任务，结果会保存到本地",
            )
        else:
            x, y = self._target()
            target = QPointF(x * self.width(), y * self.height())
            p.setPen(QPen(QColor("#c1a1ff" if self.task == "right_click" else "#6de1d1"), 3))
            p.setBrush(QColor("#174f54"))
            radius = 30 if self.task == "drag" else 24
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
