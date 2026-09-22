"""Vector-only wrist instructions. Animation never controls calibration progress."""

import math
import time

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class WristPoseGuide(QWidget):
    """Neutral, rightward, upward, and completed pose illustrations (stages 0–3)."""

    LOOP_SECONDS = 4.8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 140)
        self.setAccessibleName("腕动校准动作示意")
        self.setToolTip("绘制的动作示意，不是摄像头画面或当前姿态；转到后保持不动即可。")
        self.stage = 0
        self._started = time.monotonic()
        self._stopped = False
        self._collecting = False
        self._neutral_return = False
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._animate)
        self._describe()

    def sizeHint(self):
        return QSize(300, 170)

    def set_stage(self, stage):
        if isinstance(stage, bool) or not isinstance(stage, int) or stage not in range(4):
            raise ValueError("示意步骤应为 0、1、2 或 3")
        if stage != self.stage or self._stopped:
            self.stage = stage
            self._started = time.monotonic()
            self._stopped = False
            self._collecting = False
            self._neutral_return = False
        self._describe()
        self._sync_timer()
        self.update()

    def set_collecting(self, collecting):
        """Hold the illustrated target while the independent sampler collects."""
        collecting = bool(collecting)
        if collecting != self._collecting:
            self._collecting = collecting
            self._started = time.monotonic()
            self._describe()
            self._sync_timer()
            self.update()

    def set_neutral_return(self, returning):
        """Show only the neutral pose until the sampler accepts the return."""
        returning = bool(returning)
        if returning != self._neutral_return:
            self._neutral_return = returning
            self._started = time.monotonic()
            self._describe()
            self._sync_timer()
            self.update()

    def stop(self):
        self._stopped = True
        self.timer.stop()

    def _describe(self):
        phase = (
            "先回到自然中立姿势，等待提示后再抬掌。"
            if self.stage == 2 and self._neutral_return
            else "正在采集，请保持不动，等本步通过。"
            if self._collecting
            else ""
        )
        self.setAccessibleDescription(
            (
                "中立示意：前臂有支撑，手指自然放松。",
                "俯视示意：前臂保持固定，以手腕为中心轻轻向右转，转到后停住。",
                "侧面示意：先回中立，前臂保持固定，以手腕为中心轻轻向上抬掌，转到后停住。",
                "示范完成，保持自然姿态。",
            )[self.stage]
            + "此图不是当前姿态，不必照抄手形或跟随循环。"
            + phase
        )

    def _sync_timer(self):
        if (
            self.isVisible()
            and not self.visibleRegion().isEmpty()
            and not self._stopped
            and not self._collecting
            and not (self.stage == 2 and self._neutral_return)
            and self.stage in (1, 2)
        ):
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.timer.stop()

    def _animate(self):
        if not self.isVisible() or self.visibleRegion().isEmpty() or self._stopped:
            self.timer.stop()
            return
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._started = time.monotonic()
        self._sync_timer()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    @classmethod
    def _motion(cls, elapsed):
        """Rest, turn, hold, return. The example deliberately dwells at its target."""
        phase = max(0.0, elapsed) % cls.LOOP_SECONDS
        if phase < 0.8:
            return 0.0
        if phase < 1.8:
            return (1 - math.cos(math.pi * (phase - 0.8))) / 2
        if phase < 3.7:
            return 1.0
        if phase < 4.5:
            return (1 + math.cos(math.pi * (phase - 3.7) / 0.8)) / 2
        return 0.0

    @staticmethod
    def _text(painter, x, y, text, size=11, color="#a8c2d0", bold=False):
        font = painter.font()
        font.setPixelSize(size)
        font.setBold(bold)
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(QPointF(x, y), text)

    @staticmethod
    def _arrow(painter, start, end):
        start, end = QPointF(*start), QPointF(*end)
        painter.setPen(QPen(QColor("#ffce83"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        for offset in (-0.58, 0.58):
            tip = end - QPointF(7 * math.cos(angle + offset), 7 * math.sin(angle + offset))
            painter.drawLine(end, tip)

    @staticmethod
    def _skin(painter, ghost=False):
        painter.setPen(
            QPen(
                QColor("#5f8392" if ghost else "#b99472"),
                1.3,
                Qt.PenStyle.DashLine if ghost else Qt.PenStyle.SolidLine,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush if ghost else QColor("#ead0af"))

    @classmethod
    def _top_hand(cls, painter, angle, ghost=False):
        painter.save()
        painter.translate(150, 121)
        painter.rotate(angle)
        path = QPainterPath(QPointF(-10, 5))
        path.lineTo(-14, -19)
        path.quadTo(-22, -27, -28, -29)
        path.cubicTo(-36, -33, -32, -43, -26, -39)
        path.lineTo(-15, -32)
        path.lineTo(-15, -55)
        path.cubicTo(-16, -66, -7, -67, -7, -55)
        path.lineTo(-5, -42)
        path.lineTo(-5, -67)
        path.cubicTo(-5, -78, 4, -78, 4, -67)
        path.lineTo(5, -43)
        path.lineTo(8, -62)
        path.cubicTo(9, -73, 18, -69, 16, -60)
        path.lineTo(15, -41)
        path.lineTo(20, -53)
        path.cubicTo(24, -63, 32, -57, 27, -48)
        path.lineTo(22, -25)
        path.quadTo(21, -7, 11, 5)
        path.closeSubpath()
        cls._skin(painter, ghost)
        painter.drawPath(path)
        if not ghost:
            painter.setPen(QPen(QColor("#bf9a79"), 1))
            crease = QPainterPath(QPointF(-9, -28))
            crease.quadTo(-3, -20, 11, -23)
            painter.drawPath(crease)
            painter.drawLine(QPointF(-7, -8), QPointF(8, -8))
        painter.restore()

    @classmethod
    def _side_hand(cls, painter, angle, ghost=False):
        painter.save()
        painter.translate(135, 114)
        painter.rotate(-angle)
        path = QPainterPath(QPointF(-2, -8))
        path.cubicTo(17, -11, 25, -16, 39, -12)
        path.cubicTo(49, -9, 59, -12, 68, -7)
        path.cubicTo(76, -2, 70, 7, 63, 5)
        path.lineTo(48, 3)
        path.cubicTo(48, 11, 42, 17, 35, 12)
        path.quadTo(19, 8, -2, 9)
        path.closeSubpath()
        cls._skin(painter, ghost)
        painter.drawPath(path)
        if not ghost:
            painter.setPen(QPen(QColor("#bf9a79"), 1))
            painter.drawLine(QPointF(39, -7), QPointF(66, -3))
            painter.drawLine(QPointF(42, -2), QPointF(64, 1))
            crease = QPainterPath(QPointF(24, 2))
            crease.quadTo(32, -3, 46, 4)
            painter.drawPath(crease)
        painter.restore()

    def _top_view(self, painter, amount):
        self._text(painter, 14, 43, "俯视示意", color="#7195aa")
        painter.setPen(QPen(QColor("#36536a"), 1))
        painter.setBrush(QColor("#1d3345"))
        painter.drawRoundedRect(QRectF(111, 130, 78, 20), 7, 7)
        self._skin(painter)
        painter.drawRoundedRect(QRectF(138, 112, 24, 36), 7, 7)
        if self.stage == 1:
            self._top_hand(painter, 0, ghost=True)
        self._top_hand(painter, amount * 25)
        if self.stage == 1:
            self._text(painter, 223, 60, "向右", color="#ffce83")
            self._arrow(painter, (221, 69), (258, 69))
        self._text(painter, 17, 105, "前臂放稳")
        painter.setPen(QPen(QColor("#7195aa"), 1))
        painter.drawLine(QPointF(73, 109), QPointF(128, 139))
        self._pivot(painter, 150, 121, 212, 117)

    def _side_view(self, painter, amount):
        self._text(painter, 14, 43, "侧面示意 · 先回正", color="#7195aa")
        painter.setPen(QPen(QColor("#36536a"), 1))
        painter.setBrush(QColor("#1d3345"))
        painter.drawRoundedRect(QRectF(34, 127, 99, 12), 5, 5)
        painter.drawLine(QPointF(24, 141), QPointF(274, 141))
        self._skin(painter)
        painter.drawRoundedRect(QRectF(37, 104, 101, 22), 9, 9)
        self._side_hand(painter, 0, ghost=True)
        self._side_hand(painter, amount * 27)
        if not self._neutral_return:
            self._text(painter, 241, 55, "向上", color="#ffce83")
            self._arrow(painter, (253, 104), (253, 65))
        self._text(painter, 39, 92, "前臂不抬起")
        self._pivot(painter, 135, 114, 167, 133)

    @classmethod
    def _pivot(cls, painter, x, y, label_x, label_y):
        painter.setPen(QPen(QColor("#71ddca"), 1))
        painter.drawLine(QPointF(x + 5, y), QPointF(label_x - 4, label_y - 4))
        painter.setBrush(QColor("#183d42"))
        painter.drawEllipse(QPointF(x, y), 4, 4)
        cls._text(painter, label_x, label_y, "手腕", color="#71ddca")

    def paintEvent(self, event):
        # Scrolling back into view can expose a widget without a showEvent.
        self._sync_timer()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#284154"), 1))
        painter.setBrush(QColor("#101f2e"))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 9, 9)
        scale = min(self.width() / 300, self.height() / 170)
        painter.translate((self.width() - 300 * scale) / 2, (self.height() - 170 * scale) / 2)
        painter.scale(scale, scale)
        self._text(painter, 13, 21, "动作示意 · 不必照抄手形", 12, "#dcecf3", True)
        self._text(painter, 266, 21, "完成" if self.stage == 3 else f"{self.stage + 1}/3")
        amount = 0.0
        if self.stage in (1, 2) and not (self.stage == 2 and self._neutral_return):
            amount = 1.0 if self._collecting else self._motion(time.monotonic() - self._started)
        if self.stage == 2:
            self._side_view(painter, amount)
        else:
            self._top_view(painter, amount)
        if self.stage == 2 and self._neutral_return:
            footer = "先回正 · 等待提示后再轻抬手掌"
        elif self._collecting:
            footer = "保持不动 · 等本步通过"
        elif self.stage in (0, 3):
            footer = "手指自然放松 · 不是当前姿态画面"
        else:
            footer = "转到后停住 · 等本步通过"
        self._text(painter, 15, 162, footer, 11)
