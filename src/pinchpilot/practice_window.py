"""Standalone virtual test bench; its host supplies already-computed engine results."""

import json
import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .practice import TASKS, PracticeSession


class PracticeCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 180)
        self.session = None
        self.pointer = (0.5, 0.5)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101e2d"))
        session = self.session
        if session is None:
            p.setPen(QColor("#c9dbe8"))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "选择任务，点「开始本项」\n测试只操作这里的虚拟光标",
            )
            return
        width, height = session.viewport
        scale = min(self.width() / width, self.height() / height)
        area = QRectF(
            (self.width() - width * scale) / 2,
            (self.height() - height * scale) / 2,
            width * scale,
            height * scale,
        )

        def point(coords):
            return QPointF(
                area.left() + coords[0] * area.width(), area.top() + coords[1] * area.height()
            )

        p.setClipRect(area)
        p.setPen(QPen(QColor("#355061"), 1))
        p.drawRect(area.adjusted(1, 1, -1, -1))
        if session.task == "scroll":
            x = area.center().x()
            top, span = area.top() + area.height() * 0.12, area.height() * 0.76
            target = top + session.scroll_target / 100 * span
            p.setPen(QPen(QColor("#6de1d1"), 28))
            half = session.SCROLL_TOLERANCE / 100 * span
            p.drawLine(QPointF(x, target - half), QPointF(x, target + half))
            p.setPen(QPen(QColor("#577084"), 2))
            p.drawLine(QPointF(x, top), QPointF(x, top + span))
            p.setPen(QPen(QColor("#ffffff"), 3))
            y = top + session.scroll_value / 100 * span
            p.drawLine(QPointF(x - 30, y), QPointF(x + 30, y))
            p.drawText(QRectF(x + 45, y - 14, 100, 28), f"{session.scroll_value:.0f}")
            p.setPen(QColor("#c9dbe8"))
            p.drawText(
                QRectF(area.left() + 12, area.top() + 12, area.width() - 24, 28),
                f"目标 {session.scroll_target:.0f} ± {session.SCROLL_TOLERANCE:.0f} · 绿色区域内停住",
            )
        else:
            target = point(session.target)
            radius = session.radius_px * scale
            p.setPen(QPen(QColor("#6de1d1"), 2))
            p.setBrush(QColor("#194b50"))
            p.drawEllipse(target, radius, radius)
            p.drawLine(target + QPointF(-4, 0), target + QPointF(4, 0))
            p.drawLine(target + QPointF(0, -4), target + QPointF(0, 4))
            if session.task == "drag":
                origin = point(session.drag_origin)
                p.setPen(QPen(QColor("#d8b4fc"), 2))
                p.setBrush(QColor("#4a3860"))
                center = point(session.pointer) if session.pressed == "left" else origin
                p.drawRect(QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2))
                p.drawText(
                    QRectF(origin.x() - 55, origin.y() + radius + 8, 110, 24),
                    Qt.AlignmentFlag.AlignCenter,
                    "从这里捏住拖动",
                )
            if session.task == "double_click" and session.first_click is not None:
                p.setPen(QColor("#ffcf82"))
                p.drawText(
                    QRectF(target.x() - 90, target.y() + radius + 8, 180, 28),
                    Qt.AlignmentFlag.AlignCenter,
                    "第一下已收到，再短捏一次",
                )
        cursor = point(self.pointer)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.setBrush(QColor("#ecae58" if session.pressed else "#3282ad"))
        p.drawEllipse(cursor, 6, 6)


class PracticeWindow(QWidget):
    closed = Signal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName("practice")
        self.workspace = Path(workspace)
        self.session = None
        self.source = None
        self.last_feed_time = None
        self.last_result = None
        self.report_path = None
        self.export_error = ""
        self.engine_config = {}
        self.run_config = {}
        self._closed = False
        self.setWindowTitle("PinchPilot · 虚拟测试台")
        self.resize(820, 570)
        self.setMinimumSize(440, 460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        title = QLabel("固定任务测试台 · 系统鼠标已暂停")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        layout.addWidget(title)
        self.scope = QLabel("等待主窗口输入 · 仅测试虚拟任务；不代表系统点击成功率或识别准确率。")
        self.scope.setWordWrap(True)
        layout.addWidget(self.scope)
        row = QHBoxLayout()
        self.task_choice = QComboBox()
        for key, task in TASKS.items():
            self.task_choice.addItem(task.title, key)
        row.addWidget(self.task_choice, 1)
        self.start_button = QPushButton("开始本项")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_task)
        row.addWidget(self.start_button)
        self.cancel_button = QPushButton("取消并保存")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.instruction = QLabel()
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)
        self.canvas = PracticeCanvas()
        layout.addWidget(self.canvas, 1)
        self.metrics = QLabel("尚未开始")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.feedback = QLabel(
            "结果只保存任务摘要，不保存图片、关键点或光标轨迹。关闭测试台后需手动启用系统鼠标。"
        )
        self.feedback.setWordWrap(True)
        self.feedback.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.feedback)
        self.task_choice.currentIndexChanged.connect(self._instruction)
        self._instruction()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._refresh)
        self.timer.start()
        screen = self.screen()
        if screen:
            available = screen.availableGeometry()
            self.resize(min(820, available.width() - 30), min(570, available.height() - 40))

    def _instruction(self):
        self.instruction.setText(TASKS[self.task_choice.currentData()].instruction)

    def set_config(self, config):
        """Capture serializable engine settings; no caller-owned mutable references."""
        if not isinstance(config, dict):
            raise ValueError("测试台配置必须是参数字典")
        self.engine_config = json.loads(json.dumps(config, allow_nan=False))

    def start_task(self):
        if self.source not in ("camera", "synthetic_demo") or self._closed:
            return
        if self.session and self.session.active:
            self.cancel_task()
        self.report_path = None
        self.export_error = ""
        self.run_config = json.loads(json.dumps(self.engine_config))
        self.session = PracticeSession(
            self.task_choice.currentData(),
            started=time.monotonic(),
            source=self.source,
            viewport=(max(100, self.canvas.width()), max(100, self.canvas.height())),
        )
        self.canvas.session = self.session
        self.feedback.setText("先松开点击手指，再开始动作 · 只操控虚拟光标；取消会保留已完成部分。")
        self._refresh()

    def feed(self, result, now, source="camera"):
        if self._closed:
            return
        self.source = source
        self.last_feed_time = now
        self.last_result = result
        if (
            result.pointer is not None
            and len(result.pointer) == 2
            and all(math.isfinite(v) and 0 <= v <= 1 for v in result.pointer)
        ):
            self.canvas.pointer = result.pointer
        if self.session and self.session.active:
            self.session.feed(result, now, source)
            self.canvas.pointer = self.session.pointer
            if not self.session.active:
                self._export(now)
            elif not self.session.awaiting_clear:
                self.feedback.setText(result.hint or "虚拟输入已就绪 · 取消会保留已完成部分。")
        self.canvas.update()
        self._refresh()

    def invalidate(self, now, source, reason="stale_input"):
        """Host reports stale/absent input; never renew freshness with a timer tick."""
        if self._closed:
            return
        if self.session and self.session.active:
            reason = "source_changed" if source != self.session.source else reason
            self.session.cancel(now, reason)
            self._export(now)
        self.source = source
        self.last_feed_time = None
        self.last_result = None
        self._refresh()

    def _export(self, now):
        if self.session is None or self.report_path is not None:
            return
        folder = self.workspace / "reports/practice"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"practice_{time.time_ns()}.json"
            with path.open("x", encoding="utf-8") as output:
                summary = self.session.summary(now)
                summary["engine_config"] = self.run_config
                json.dump(summary, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
            self.report_path = path
            self.export_error = ""
            self.feedback.setText(f"已保存：reports/practice/{path.name} · 不含图像或轨迹")
        except (OSError, ValueError) as error:
            self.export_error = str(error)
            self.feedback.setText(f"结果尚未保存：{error}。关闭窗口时会再尝试保存。")

    def cancel_task(self):
        if self.session and self.session.active:
            now = time.monotonic()
            self.session.cancel(now)
            self._export(now)
            self._refresh()

    def _refresh(self):
        now = time.monotonic()
        active = bool(self.session and self.session.active)
        recent = self.last_feed_time is not None and 0 <= now - self.last_feed_time < 0.5
        self.start_button.setEnabled(
            not active and recent and self.source in ("camera", "synthetic_demo")
        )
        self.cancel_button.setEnabled(active)
        self.task_choice.setEnabled(not active)
        label = {"camera": "相机识别输入", "synthetic_demo": "合成演示 · 仅工程验证"}.get(
            self.source, "等待主窗口输入"
        )
        self.scope.setText(
            f"{label}{' · 输入已中断' if self.source and not recent else ''} · 仅虚拟任务表现，不代表真实识别准确率。"
        )
        if self.session:
            summary = self.session.summary(now)
            status = "进行中" if active else "完成" if self.session.completed else "已取消"
            explanation = {
                "stale_input": "输入已过期，请恢复相机后重新开始",
                "source_changed": "输入来源已变化，请重新开始",
            }.get(self.session.reason, "")
            self.metrics.setText(
                f"{status} · 完成 {self.session.hits}/{self.session.total} · 未命中 {self.session.misses}"
                f" · {summary['duration_s']:.1f} 秒 · 输入中断 {self.session.interrupted_inputs}"
                + (" · 请先松开点击手指" if active and self.session.awaiting_clear else "")
                + (f" · {explanation}" if explanation else "")
            )
        self.canvas.update()

    def closeEvent(self, event):
        if not self._closed:
            self.timer.stop()
            if self.session:
                now = time.monotonic()
                self.session.cancel(now, "window_closed")
                self._export(now)
            self._closed = True
            self.closed.emit()
        super().closeEvent(event)
