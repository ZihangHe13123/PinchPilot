"""Draws the schematic hand of ``contact_demo`` and the Space key it asks for."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import contact_demo

FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12)}
FINGERS.update(ring=(13, 14, 15, 16), pinky=(17, 18, 19, 20))
PALM = (0, 1, 5, 9, 13, 17)
VIEW = (25.0, 30.0)  # yaw and pitch of the drawing: palm in view, seen a little from above


def _font(base, change, bold=False):
    """A copy of `base` a few pixels larger or smaller, whichever unit it is sized in."""
    font = QFont(base)
    pixels = font.pixelSize() if font.pixelSize() > 0 else round(font.pointSizeF() * 4 / 3)
    font.setPixelSize(max(10, pixels + change))
    font.setBold(bold)
    return font


COLOURS = {"thumb": "#eafff9", "middle": "#6de1d1", "index": "#ffce83", "ring": "#c1a1ff"}
NAMES = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指"}
IDLE = "#2f6d70"
ORDER = ("thumb", "middle", "index", "ring")


class GestureDemo(QWidget):
    """An animated example of the step: what the hand does, and when Space is held."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 250)
        self.step = None
        self.state = contact_demo.DemoState()
        self.view = VIEW  # (yaw, pitch); the capture window turns it as each round asks.
        self.mirror = False  # True draws a left hand: thumb on the right.

    def show_step(self, step, moment):
        self.step = step
        self.state = contact_demo.demo_state(step, moment) if step is not None else self.state
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101e2d"))
        if self.step is None:
            p.end()
            return
        state = self.state
        points = contact_demo.hand_points(state.grip, state.index, state.ring)
        everything = np.vstack((points, contact_demo.WRIST_SIDES))
        flat, depth = contact_demo.project(everything, self.view[0] + state.turn, self.view[1])
        key_height, legend_height = 46, 24
        area = QRectF(
            8, legend_height + 6, self.width() - 16, self.height() - key_height - legend_height - 22
        )
        scale = min(area.width() / 1.75, area.height() / 1.98)
        side = -1 if self.mirror else 1
        centre = QPointF(
            area.center().x() + side * (0.08 + state.shift[0]) * scale,
            area.bottom() - 0.3 * scale - state.shift[1] * scale,
        )
        screen = [QPointF(centre.x() + side * x * scale, centre.y() - y * scale) for x, y in flat]
        wrist_left, wrist_right, arm_left, arm_right = screen[21:25]

        p.setPen(QPen(QColor("#2f6f74"), 2))
        p.setBrush(QColor(38, 92, 99))
        palm = [arm_left, wrist_left, screen[1], screen[5], screen[9], screen[13], screen[17]]
        p.drawPolygon(QPolygonF([*palm, wrist_right, arm_right]))
        # In this step: the thumb whenever something closes, and each finger asked to move.
        involved = {"middle"} if self.step.labels[0] == 1 else set()
        involved |= {"index"} if self.step.labels[1] not in (0, None) or state.index else set()
        involved |= {"ring"} if self.step.labels[2] not in (0, None) or state.ring else set()
        involved |= {"index"} if "near" in self.step.name else set()
        involved |= {"thumb"} if involved else set()
        # Far fingers first, so that near ones are drawn over them.
        order = sorted(FINGERS, key=lambda name: float(np.mean(depth[list(FINGERS[name])])))
        width = max(6.0, scale * 0.105)
        round_pen = (Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        for name in order:
            chain = [screen[joint] for joint in FINGERS[name]]
            colour = QColor(COLOURS[name] if name in involved else IDLE)
            for pen in (
                QPen(QColor("#0b1420"), width + 4, *round_pen),
                QPen(colour, width, *round_pen),
            ):
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolyline(QPolygonF(chain))
        gaps = contact_demo.tip_gaps(points)
        for gap, tip in zip(gaps, (12, 8, 16)):
            if gap < contact_demo.TOUCHING:
                spot = (screen[4] + screen[tip]) / 2
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(95, 211, 141, 80))
                p.drawEllipse(spot, width * 1.7, width * 1.7)
                p.setBrush(QColor("#5fd38d"))
                p.drawEllipse(spot, width * 0.7, width * 0.7)
        # Which finger is which: a legend, because names at the fingertips would overlap.
        p.setFont(_font(self.font(), -1))
        shown = [name for name in ORDER if name in involved]
        cell = min(86.0, (self.width() - 16) / max(1, len(shown)))
        left = (self.width() - cell * len(shown)) / 2
        for number, name in enumerate(shown):
            box = QRectF(left + number * cell, 6, cell, legend_height - 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(COLOURS[name]))
            p.drawRoundedRect(QRectF(box.left() + 4, box.center().y() - 5, 10, 10), 3, 3)
            p.setPen(QColor("#dce7f0"))
            p.drawText(
                box.adjusted(20, 0, 0, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                NAMES[name],
            )
        if not shown:
            p.setPen(QColor("#7f95a8"))
            p.drawText(
                QRectF(0, 6, self.width(), legend_height - 4),
                Qt.AlignmentFlag.AlignCenter,
                "手指互不接触",
            )

        key = QRectF(self.width() / 2 - 96, self.height() - key_height - 6, 192, key_height - 6)
        p.setFont(_font(self.font(), 1, bold=True))
        if state.space is None:
            p.setPen(QPen(QColor("#3b5063"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(key, 8, 8)
            p.setPen(QColor("#7f95a8"))
            p.drawText(key, Qt.AlignmentFlag.AlignCenter, "这一段不按空格")
        elif state.space:
            p.setPen(QPen(QColor("#5fd38d"), 2))
            p.setBrush(QColor("#2f8f5f"))
            p.drawRoundedRect(key.adjusted(0, 3, 0, 3), 8, 8)
            p.setPen(QColor("#ffffff"))
            p.drawText(key.adjusted(0, 3, 0, 3), Qt.AlignmentFlag.AlignCenter, "按住空格")
        else:
            p.setPen(QPen(QColor("#f0b35a"), 1.5))
            p.setBrush(QColor("#1a2a3a"))
            p.drawRoundedRect(key, 8, 8)
            p.setPen(QColor("#f0b35a"))
            text = "不要按空格" if "near" in self.step.name else "空格：松开"
            p.drawText(key, Qt.AlignmentFlag.AlignCenter, text)
        p.end()


def caption(step, several=False):
    """One line under the drawing: how to use it for this step."""
    if step is None:
        return ""
    if contact_demo.KEY in step.labels:
        return "跟着示意图的节奏做：指尖碰到时下面的空格键变绿，这时按住空格，离开就松开。"
    if "near" in step.name:
        return "示意图：手指靠近拇指但留一条缝，全程不按空格。"
    if several:
        return "示意图会和屏幕上的提示一起切换，跟着做。"
    return "示意图：这一段手的样子。"
