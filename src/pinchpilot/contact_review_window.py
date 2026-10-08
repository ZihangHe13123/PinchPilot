"""Window for reviewing a guided session's draft labels unit by unit. Opens no camera."""

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import contact_protocol, contact_review
from .app import STYLE
from .contact_data import CHANNELS
from .contact_review import ACCEPTED, AGREE, DISAGREE, DISCARDED, Margins
from .widgets import CONNECTIONS

LANES = ("中指", "食指", "无名指")
TIPS = (12, 8, 16)  # fingertip joints in the order of CHANNELS
PAD = 1.0  # seconds shown before and after the unit
TOUCH, APART, BLANK, SPACE = "#5fd38d", "#40566c", "#101a26", "#f0b35a"
STATE_COLORS = {
    None: "#2a3b4d",
    ACCEPTED: TOUCH,
    AGREE: TOUCH,
    DISCARDED: "#e0705c",
    DISAGREE: "#e0705c",
}
# name, label, (low, high, step) in milliseconds
SLIDERS = (
    ("shift", "空格标记提前", (-300, 500, 10)),
    ("edge", "按下松开前后留空", (30, 400, 10)),
    ("settle", "提示出现后留空", (200, 2500, 50)),
    ("lead", "提示结束前留空", (100, 1500, 50)),
)


class Timeline(QWidget):
    """Thumb-to-fingertip distances, the labels under them and the Space marks, over time."""

    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 330)
        self.session = self.labels = self.unit = None
        self.spans, self.scale, self.playhead = [], np.ones(3), 0.0

    def show_unit(self, session, labels, spans, unit):
        self.session, self.labels, self.spans, self.unit = session, labels, spans, unit
        with np.errstate(all="ignore"):
            top = np.nanpercentile(session.distances, 99, axis=0)
        self.scale = np.where(np.isfinite(top) & (top > 0), top, 1.0)
        self.update()

    def set_playhead(self, moment):
        self.playhead = moment
        self.update()

    def _plot(self):
        return QRectF(64, 26, max(1, self.width() - 76), max(1, self.height() - 50))

    def _moment(self, x):
        plot = self._plot()
        start, end = self.unit.start - PAD, self.unit.end + PAD
        return start + (end - start) * min(1.0, max(0.0, (x - plot.left()) / plot.width()))

    def mousePressEvent(self, event):
        if self.unit is not None:
            self.seek.emit(self._moment(event.position().x()))

    mouseMoveEvent = mousePressEvent

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0d1826"))
        if self.unit is None:
            p.end()
            return
        session, unit, plot = self.session, self.unit, self._plot()
        start, end = unit.start - PAD, unit.end + PAD

        def x(moment):
            return plot.left() + plot.width() * (moment - start) / (end - start)

        first, last = np.searchsorted(session.times, (start, end))
        times = session.times[first:last]
        key_row = 24
        lane = (plot.height() - key_row) / 3
        p.setPen(QColor("#8fa8bc"))
        p.drawText(
            QRectF(0, plot.top(), 58, key_row),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            "空格",
        )
        for down, up in self.spans:
            if up > start and down < end:
                box = QRectF(x(max(down, start)), plot.top() + 4, 0, key_row - 8)
                box.setRight(x(min(up, end)))
                p.fillRect(box, QColor(SPACE))
        for channel, name in enumerate(LANES):
            top = plot.top() + key_row + lane * channel
            area = QRectF(plot.left(), top + 5, plot.width(), lane - 10)
            p.fillRect(area, QColor("#122132"))
            p.setPen(QColor("#dce7f0"))
            p.drawText(
                QRectF(0, top, 58, lane),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                name,
            )
            band = QRectF(area.left(), area.bottom() - 12, area.width(), 12)
            values = self.labels[first:last, channel]
            if len(times):
                changes = np.flatnonzero(np.diff(values)) + 1
                for a, b in zip(np.append(0, changes), np.append(changes, len(values))):
                    piece = QRectF(x(times[a]), band.top(), 0, band.height())
                    piece.setRight(x(times[b]) if b < len(times) else x(times[-1]) + 2)
                    color = TOUCH if values[a] == 1 else APART if values[a] == 0 else BLANK
                    p.fillRect(piece, QColor(color))
                    if values[a] < 0:
                        p.fillRect(piece, QBrush(QColor("#3b4c5e"), Qt.BrushStyle.BDiagPattern))
            path, drawing = QPainterPath(), False
            curve = session.distances[first:last, channel] / self.scale[channel]
            for moment, value in zip(times, curve):
                if not np.isfinite(value):
                    drawing = False
                    continue
                point = QPointF(
                    x(moment), area.top() + 4 + (1 - min(1.0, value)) * (area.height() - 22)
                )
                path.lineTo(point) if drawing else path.moveTo(point)
                drawing = True
            p.setPen(QPen(QColor("#9fd6ff"), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for index in unit.steps:
            step_start, step_end, step = session.schedule[index]
            p.setPen(QPen(QColor("#5f7890"), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x(step_start), plot.top()), QPointF(x(step_start), plot.bottom()))
            if len(unit.steps) > 1:
                # Which prompt each part of a merged unit was recorded under.
                p.setPen(QColor("#b9cbd9"))
                p.drawText(
                    QRectF(x(step_start) + 4, plot.top(), x(step_end) - x(step_start) - 6, key_row),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    step.text.split("：")[0],
                )
        p.setPen(QPen(QColor("#5f7890"), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x(unit.end), plot.top()), QPointF(x(unit.end), plot.bottom()))
        shade = QColor(8, 14, 22, 150)
        p.fillRect(
            QRectF(plot.left(), plot.top(), x(unit.start) - plot.left(), plot.height()), shade
        )
        p.fillRect(
            QRectF(x(unit.end), plot.top(), plot.right() - x(unit.end), plot.height()), shade
        )
        p.setPen(QColor("#8fa8bc"))
        for second in range(int(np.ceil(unit.end - unit.start)) + 1):
            if second % 2 == 0:
                at = x(unit.start + second)
                p.drawText(
                    QRectF(at - 20, plot.bottom() + 2, 40, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{second}s",
                )
        p.drawText(
            QRectF(plot.left(), 2, plot.width(), 20),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "曲线：拇指尖到该指尖的距离，越低越近。色带：绿=接触，灰=未接触，斜纹=不给标签。",
        )
        if start <= self.playhead <= end:
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.drawLine(
                QPointF(x(self.playhead), plot.top()), QPointF(x(self.playhead), plot.bottom())
            )
        p.end()


class HandView(QWidget):
    """The recorded hand at one moment, zoomed to where it is during the unit."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.frame, self.box, self.row = None, (0.5, 0.5, 0.5), (-1, -1, -1)

    def show_hand(self, frame, box, row):
        self.frame, self.box, self.row = frame, box, row
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101e2d"))
        if self.frame is None or not self.frame.landmarks:
            p.setPen(QColor("#8fa8bc"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "这一帧没有检测到手")
            p.end()
            return
        cx, cy, half = self.box
        radius = min(self.width(), self.height()) / 2 - 10
        points = [
            QPointF(
                self.width() / 2 + (px * self.frame.aspect - cx) / half * radius,
                self.height() / 2 + (py - cy) / half * radius,
            )
            for px, py, _ in self.frame.landmarks
        ]
        p.setPen(QPen(QColor("#6de1d1"), 3))
        for a, b in CONNECTIONS:
            p.drawLine(points[a], points[b])
        for tip, value in zip(TIPS, self.row):
            pen = QPen(QColor(TOUCH if value == 1 else "#6f8497"), 3 if value == 1 else 1.5)
            if value != 1:
                pen.setStyle(Qt.PenStyle.DashLine if value == 0 else Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawLine(points[4], points[tip])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#d9fff5"))
        for index, point in enumerate(points):
            size = 5 if index in (4, 8, 12, 16) else 3
            p.drawEllipse(point, size, size)
        p.end()


class UnitStrip(QWidget):
    """One cell per unit, coloured by its decision; click to jump."""

    chosen = Signal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(22)
        self.states, self.current = [], 0

    def show_states(self, states, current):
        self.states, self.current = states, current
        self.update()

    def _cell(self):
        return min(22.0, self.width() / max(1, len(self.states)))

    def mousePressEvent(self, event):
        if self.states:
            self.chosen.emit(min(len(self.states) - 1, int(event.position().x() / self._cell())))

    def paintEvent(self, _):
        p = QPainter(self)
        cell = self._cell()
        for index, state in enumerate(self.states):
            box = QRectF(index * cell + 1, 3, max(1.0, cell - 2), 16)
            p.fillRect(box, QColor(STATE_COLORS[state]))
            if index == self.current:
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawRect(box)
        p.end()


class ReviewWindow(QWidget):
    """Review your own recordings, or check a sample of a teammate's reviewed labels.

    Which of the two it is follows from the names: the annotator's own recordings are
    reviewed in full, anyone else's are cross-checked.
    """

    def __init__(self, source, annotator, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("practice")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(STYLE)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle("PinchPilot · 标签检查")
        self.resize(1180, 800)
        self.annotator = annotator.strip()
        self.clock = time.monotonic
        self.session = self.labels = None
        self.mode = ""
        self.items, self.position = [], 0
        self.margins, self.decisions, self.notes = Margins(), {}, {}
        self.owner, self.reviewed, self.objections = "", {}, {}
        self.box = (0.5, 0.5, 0.5)
        self.playing, self.playhead, self.last_tick = True, 0.0, None
        self.error = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        self.picker = QComboBox()
        self.picker.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.picker.setMinimumWidth(520)
        top.addWidget(QLabel("录制"))
        top.addWidget(self.picker, 1)
        self.mode_label = QLabel()
        self.mode_label.setObjectName("chip")
        top.addWidget(self.mode_label)
        layout.addLayout(top)
        self.strip = UnitStrip()
        layout.addWidget(self.strip)
        self.title_label = QLabel("请选择一段录制")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-size:20px; font-weight:700; color:#ffffff;")
        layout.addWidget(self.title_label)
        self.info_label = QLabel()
        layout.addWidget(self.info_label)
        middle = QHBoxLayout()
        self.hand = HandView()
        self.timeline = Timeline()
        middle.addWidget(self.hand, 2)
        middle.addWidget(self.timeline, 5)
        layout.addLayout(middle, 1)

        self.sliders, self.slider_labels = {}, {}
        tuning = QHBoxLayout()
        for name, text, (low, high, step) in SLIDERS:
            column = QVBoxLayout()
            label = QLabel()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setRange(low // step, high // step)
            slider.setProperty("step", step)
            slider.valueChanged.connect(lambda _, key=name: self._slider_text(key))
            slider.sliderReleased.connect(self._margins_changed)
            column.addWidget(label)
            column.addWidget(slider)
            tuning.addLayout(column)
            self.sliders[name], self.slider_labels[name] = slider, label
        layout.addLayout(tuning)

        actions = QHBoxLayout()
        self.previous_button = QPushButton("← 上一个")
        self.accept_button = QPushButton("通过 (Enter)")
        self.accept_button.setObjectName("primary")
        self.reject_button = QPushButton("这一段作废 (X)")
        self.reject_button.setObjectName("stop")
        self.next_button = QPushButton("下一个 →")
        self.play_button = QPushButton("暂停 (空格)")
        self.note = QLineEdit()
        self.note.setPlaceholderText("不同意时可以写一句原因")
        self.note.setMaxLength(120)
        self.finish_button = QPushButton("完成并保存")
        for button in (
            self.previous_button,
            self.accept_button,
            self.reject_button,
            self.next_button,
            self.play_button,
            self.finish_button,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for widget in (
            self.previous_button,
            self.accept_button,
            self.reject_button,
            self.next_button,
            self.play_button,
        ):
            actions.addWidget(widget)
        actions.addWidget(self.note, 1)
        actions.addWidget(self.finish_button)
        layout.addLayout(actions)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)

        self.previous_button.clicked.connect(lambda: self.go(self.position - 1))
        self.next_button.clicked.connect(lambda: self.go(self.position + 1))
        self.accept_button.clicked.connect(lambda: self.decide(True))
        self.reject_button.clicked.connect(lambda: self.decide(False))
        self.play_button.clicked.connect(self.toggle_play)
        self.finish_button.clicked.connect(self.finish)
        self.note.editingFinished.connect(self._note_changed)
        self.note.returnPressed.connect(self.setFocus)
        self.strip.chosen.connect(self.go)
        self.timeline.seek.connect(self._seek)
        self.picker.activated.connect(lambda index: self.open_session(self.picker.itemData(index)))
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        source, first = Path(source), 0
        if source.is_dir():
            rows = contact_review.list_recordings(source)
            # Your own recordings first, and start with one that still needs reviewing.
            rows.sort(key=lambda row: row["participant"] != self.annotator)
            for row in rows:
                self.picker.addItem(self._picker_text(row), row["path"])
            mine = [
                index
                for index, row in enumerate(rows)
                if row["participant"] == self.annotator and row["status"] != "已检查"
            ]
            first = mine[0] if mine else 0
            if not rows:
                self.status_label.setText(f"这个文件夹里没有按提示录制的压缩包：{source}")
        else:
            self.picker.addItem(source.name, source)
        self._refresh()
        if self.picker.count():
            self.picker.setCurrentIndex(first)
            self.open_session(self.picker.itemData(first))

    @staticmethod
    def _picker_text(row):
        progress = f" {row['decided']}/{row['units']}" if row["status"] == "检查中" else ""
        return f"{row['participant']} · {row['path'].name} · {row['status']}{progress}"

    def _update_picker(self):
        """Keep the current entry's status in step with what was just saved."""
        rows = contact_review.list_recordings(self.session.path.parent)
        row = next((row for row in rows if row["path"] == self.session.path), None)
        for index in range(self.picker.count()):
            if row is not None and self.picker.itemData(index) == self.session.path:
                self.picker.setItemText(index, self._picker_text(row))

    # ---- Opening a recording.

    def open_session(self, path):
        self.session, self.items, self.error = None, [], ""
        self.status_label.setText("正在读取录制…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            session = contact_review.load_session(path)
            if session.participant == self.annotator:
                mode, owner = "review", self.annotator
                margins, decisions, notes = Margins(), {}, {}
                saved = contact_review.load_progress(session)
                if saved is not None:
                    _, margins, decisions = saved
                items = list(session.units)
                self.objections = contact_review.disagreements(session)
            else:
                mode = "crosscheck"
                owner, margins, reviewed, _ = contact_review.load_reviewed(session)
                sample = contact_review.sample_units(session, reviewed, self.annotator)
                items = [session.units[index] for index in sample]
                decisions, notes = contact_review.load_crosscheck(session, self.annotator)
                self.reviewed = reviewed
        except (ValueError, OSError) as error:
            self.error = str(error)
            return self._refresh()
        finally:
            QApplication.restoreOverrideCursor()
        self.session, self.mode, self.owner, self.items = session, mode, owner, items
        self.margins, self.decisions, self.notes = margins, dict(decisions), dict(notes)
        for name, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(round(getattr(margins, name) * 1000 / slider.property("step")))
            slider.blockSignals(False)
            self._slider_text(name)
        self._relabel()
        waiting = [i for i, unit in enumerate(items) if unit.index not in self.decisions]
        self.go(waiting[0] if waiting else 0)

    def _relabel(self):
        base = self.decisions if self.mode == "review" else self.reviewed
        self.labels = contact_review.frame_labels(self.session, self.margins, base)

    # ---- Moving around and playing.

    @property
    def unit(self):
        return self.items[self.position] if self.items else None

    def go(self, position):
        if not self.items:
            return self._refresh()
        self._note_changed()
        self.position = min(len(self.items) - 1, max(0, position))
        unit, session = self.unit, self.session
        inside = (session.times >= unit.start - PAD) & (session.times < unit.end + PAD)
        hands = [session.frames[i] for i in np.flatnonzero(inside & session.present)]
        if hands:
            xy = np.array([[(x * f.aspect, y) for x, y, _ in f.landmarks] for f in hands])
            low, high = xy.reshape(-1, 2).min(axis=0), xy.reshape(-1, 2).max(axis=0)
            self.box = (*((low + high) / 2), max(0.02, float((high - low).max()) / 2 * 1.1))
        spans = contact_protocol.key_spans(
            contact_review.shifted_events(session, self.margins), session.ended
        )
        self.timeline.show_unit(session, self.labels, spans, unit)
        self.note.blockSignals(True)
        self.note.setText(self.notes.get(unit.index, ""))
        self.note.blockSignals(False)
        self.playhead, self.last_tick = unit.start - 0.3, None
        self._refresh()

    def _seek(self, moment):
        self.playhead, self.last_tick = moment, None
        self._show_frame()

    def toggle_play(self):
        self.playing, self.last_tick = not self.playing, None
        self._refresh()

    def _tick(self):
        if self.unit is None or not self.playing:
            return
        now = self.clock()
        if self.last_tick is not None:
            self.playhead += now - self.last_tick
        self.last_tick = now
        if self.playhead > self.unit.end + 0.5:
            self.playhead = self.unit.start - 0.3
        self._show_frame()

    def _show_frame(self):
        session = self.session
        index = int(np.searchsorted(session.times, self.playhead, side="right")) - 1
        index = min(len(session.frames) - 1, max(0, index))
        self.hand.show_hand(session.frames[index], self.box, tuple(self.labels[index].tolist()))
        self.timeline.set_playhead(self.playhead)

    # ---- Decisions.

    def decide(self, positive):
        if self.unit is None:
            return
        if self.mode == "review":
            self.decisions[self.unit.index] = ACCEPTED if positive else DISCARDED
            self._relabel()
        else:
            self.decisions[self.unit.index] = AGREE if positive else DISAGREE
        self._save()
        waiting = [i for i, unit in enumerate(self.items) if unit.index not in self.decisions]
        later = [i for i in waiting if i > self.position]
        self.go(later[0] if later else waiting[0] if waiting else self.position)

    def _note_changed(self):
        if self.mode != "crosscheck" or self.unit is None:
            return
        text = self.note.text().strip()
        if text != self.notes.get(self.unit.index, ""):
            self.notes[self.unit.index] = text
            if self.unit.index in self.decisions:
                self._save()

    def _slider_text(self, name):
        slider, text = self.sliders[name], next(t for key, t, _ in SLIDERS if key == name)
        self.slider_labels[name].setText(f"{text} {slider.value() * slider.property('step')} 毫秒")

    def _margins_changed(self):
        if self.mode != "review" or self.session is None:
            return
        values = {
            name: slider.value() * slider.property("step") / 1000
            for name, slider in self.sliders.items()
        }
        margins = Margins(**values)
        if margins == self.margins:
            return
        self.margins = margins
        # A unit accepted under other margins has different labels now: it needs another look.
        cleared = [key for key, value in self.decisions.items() if value == ACCEPTED]
        for key in cleared:
            del self.decisions[key]
        self._relabel()
        self._save()
        self.go(self.position)
        if cleared:
            self.status_label.setText(f"参数变了，之前通过的 {len(cleared)} 段需要重新看一遍。")

    def _save(self):
        try:
            if self.mode == "review":
                contact_review.save_progress(
                    self.session, self.annotator, self.margins, self.decisions
                )
            else:
                contact_review.save_crosscheck(
                    self.session, self.annotator, self.decisions, self.notes
                )
            self.error = ""
            self._update_picker()
        except (ValueError, OSError) as error:
            self.error = f"保存失败：{error}"

    def finish(self):
        if self.mode != "review" or self.session is None:
            return
        try:
            target = contact_review.save_reviewed(
                self.session, self.annotator, self.margins, self.decisions
            )
        except (ValueError, OSError) as error:
            self.error = f"保存失败：{error}"
            return self._refresh()
        self.error = ""
        self.objections = {}  # They were about the labels this save has just replaced.
        self._update_picker()
        self._refresh()
        self.status_label.setText(
            f"已保存。把这两个文件上传到数据仓库里这段录制所在的文件夹：\n"
            f"{target.name}\n{self.session.sidecar('.review.json').name}"
        )

    # ---- Keys and display.

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.decide(True)
        elif key in (Qt.Key.Key_X, Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.decide(False)
        elif key == Qt.Key.Key_Left:
            self.go(self.position - 1)
        elif key == Qt.Key.Key_Right:
            self.go(self.position + 1)
        elif key == Qt.Key.Key_Space:
            self.toggle_play()
        else:
            super().keyPressEvent(event)

    def _refresh(self):
        ready, review = self.unit is not None, self.mode == "review"
        for widget in (
            self.previous_button,
            self.next_button,
            self.accept_button,
            self.reject_button,
            self.play_button,
        ):
            widget.setEnabled(ready)
        for slider in self.sliders.values():
            slider.setEnabled(ready and review)
        self.note.setVisible(ready and not review)
        self.finish_button.setVisible(review or not ready)
        self.accept_button.setText("通过 (Enter)" if review or not ready else "同意 (Enter)")
        self.reject_button.setText("这一段作废 (X)" if review or not ready else "不同意 (X)")
        self.play_button.setText("暂停 (空格)" if self.playing else "播放 (空格)")
        if not ready:
            self.mode_label.setText("")
            self.info_label.setText("")
            self.strip.show_states([], 0)
            self.finish_button.setEnabled(False)
            self.timeline.unit = None
            self.timeline.update()
            self.hand.show_hand(None, self.box, (-1, -1, -1))
            if self.error:
                self.title_label.setText("这段录制现在不能检查")
                self.status_label.setText(self.error)
            return
        unit, states = self.unit, [self.decisions.get(item.index) for item in self.items]
        done = sum(state is not None for state in states)
        bad = sum(state in (DISCARDED, DISAGREE) for state in states)
        self.strip.show_states(states, self.position)
        self.finish_button.setEnabled(review and done == len(states))
        self.mode_label.setText(
            f"检查自己的标签 · {self.annotator}"
            if review
            else f"抽查 {self.owner} 的标签 · 抽查者 {self.annotator}"
        )
        self.title_label.setText(unit.title)
        state = {
            None: "还没看",
            ACCEPTED: "已通过",
            DISCARDED: "已作废",
            AGREE: "同意",
            DISAGREE: "不同意",
        }[self.decisions.get(unit.index)]
        fingers = "、".join(
            LANES[i]
            for i in range(len(CHANNELS))
            if any(
                self.session.schedule[s][2].labels[i] == contact_protocol.KEY for s in unit.steps
            )
        )
        self.info_label.setText(
            f"第 {self.position + 1}/{len(self.items)} 段 · 第 {unit.round} 轮 · "
            f"{unit.end - unit.start:.0f} 秒 · "
            + (f"{fingers}的标签来自空格" if fingers else "标签来自屏幕提示")
            + f" · {state}"
            + "".join(
                f" · {checker} 抽查不同意" + (f"：{note}" if note else "")
                for checker, note in (self.objections.get(unit.index, []) if review else [])
            )
        )
        if self.error:
            self.status_label.setText(self.error)
        elif review:
            self.status_label.setText(
                f"已看 {done}/{len(states)}，其中作废 {bad}。"
                + (
                    "全部看完了，点「完成并保存」。"
                    if done == len(states)
                    else "进度会自动保存，可以随时关掉再继续。"
                )
            )
        else:
            self.status_label.setText(
                f"已看 {done}/{len(states)}，不同意 {bad}。"
                + (
                    f"抽查完成，结果已保存：{self.session.sidecar(f'.crosscheck.{self.annotator}.json').name}"
                    if done == len(states)
                    else "每个结论都会自动保存。"
                )
            )
        self._show_frame()

    def closeEvent(self, event):
        self.timer.stop()
        self._note_changed()
        super().closeEvent(event)


def run_review(source, annotator, smoke_seconds=None, screenshot=None):
    """Open the review window on a recording or a folder of recordings."""
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("需要用 --annotator 填写你自己的编号，例如 --annotator P02")
    source = Path(source)
    if not source.exists():
        raise ValueError(f"找不到文件或文件夹：{source}")
    app = QApplication.instance() or QApplication([])
    window = ReviewWindow(source, annotator)
    window.show()
    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        QTimer.singleShot(1200, lambda: window.grab().save(str(screenshot)))
    if smoke_seconds:
        QTimer.singleShot(int(smoke_seconds * 1000), window.close)
    return app.exec()
