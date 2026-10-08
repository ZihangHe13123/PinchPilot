"""Explicit, local keypoint collection; owns no camera, training or native input."""

import json
import math
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import contact_protocol
from .contact_data import (
    ContactRecorder,
    annotation_template,
    draft_annotation,
    read_contact_recording,
)
from .domain import HandFrame

STOP_LABELS = {
    "user_stop": "手动停止",
    "window_closed": "采集窗口关闭",
    "source_changed": "输入来源改变",
    "stale_input": "相机输入过期或中断",
    "input_gap": "连续输入间隔过长",
    "nonmonotonic_frame": "收到重复或倒序帧",
    "invalid_frame": "帧时间无效",
    "write_error": "录制写入失败",
    "consent_withdrawn": "已取消关键点保存同意",
    "protocol_complete": "已按提示录完",
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _session_name():
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class ContactCaptureWindow(QWidget):
    """Record only new, fresh camera frames after an explicit Start action.

    ``source='camera'`` is supplied by the trusted host; this window never opens
    hardware. A source label alone does not attest real-human provenance in tests.
    Empty HandFrames are part of the continuous recording, not filtered out.

    A guided session shows the steps of ``contact_protocol`` one by one and notes when the
    Space key is held. It saves draft labels from those, which still need human review.
    """

    closed = Signal()
    MAX_AGE = 0.25
    GUIDED_STALL = 5.0  # A guided session rides out a camera stall up to this many seconds.

    def __init__(self, workspace, parent=None, guided=False):
        super().__init__(parent, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName("practice")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.workspace = Path(workspace)
        self.clock = time.monotonic
        self.recorder = None
        self.source = "none"
        self.last_capture = self.last_input_timestamp = None
        self.recording_path = self.annotation_path = self.summary_path = None
        self.count = self.missing_count = 0
        self.started_utc = None
        self.stop_reason = ""
        self.error = ""
        self._closed = False
        self.protocol = contact_protocol.session()
        self.schedule = []
        self.protocol_start = None
        self.protocol_path = self.bundle_path = None
        self.identity = ("", "")
        self.key_events = []
        self.unmarked = []
        self.key_down = self.hand_present = False
        self.gaps = 0
        self.last_feed = None
        self.setWindowTitle("PinchPilot · ML 关键点采集")
        self.resize(*((640, 720) if guided else (560, 480)))
        self.setMinimumSize(440, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        heading = QLabel("ML 动作采集 · 系统鼠标已暂停")
        heading.setStyleSheet("font-size:18px; font-weight:600;")
        layout.addWidget(heading)
        scope = QLabel(
            "只保存当前控制手的连续 21 点坐标、时间与左右手分类；无手帧也保留。"
            "不保存图像，不会自动启动相机或训练模型。"
        )
        scope.setWordWrap(True)
        layout.addWidget(scope)
        self.guide_panel = QWidget()
        guide = QVBoxLayout(self.guide_panel)
        guide.setContentsMargins(0, 6, 0, 6)
        self.prompt_label = QLabel()
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setStyleSheet("font-size:26px; font-weight:700; color:#ffffff;")
        self.prompt_label.setMinimumHeight(76)  # Two lines, so a long step is never clipped.
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("font-size:15px; color:#f0b35a;")
        self.step_bar = QProgressBar()
        self.step_bar.setRange(0, 1000)
        self.step_bar.setTextVisible(False)
        self.guide_status = QLabel()
        self.key_label = QLabel()
        self.next_label = QLabel()
        self.next_label.setWordWrap(True)
        for item in (
            self.prompt_label,
            self.note_label,
            self.step_bar,
            self.guide_status,
            self.key_label,
            self.next_label,
        ):
            guide.addWidget(item)
        layout.addWidget(self.guide_panel)
        fields = QFormLayout()
        self.participant = QLineEdit("P01")
        self.participant.setMaxLength(80)
        self.participant.setPlaceholderText("使用匿名编号，例如 P01")
        self.session = QLineEdit(_session_name())
        self.session.setMaxLength(80)
        fields.addRow("用户编号", self.participant)
        fields.addRow("采集场次", self.session)
        layout.addLayout(fields)
        self.consent = QCheckBox("同意保存 21 点坐标，不记录图像；之后需人工标注")
        layout.addWidget(self.consent)
        minutes = round(contact_protocol.timeline(self.protocol)[-1][1] / 60)
        self.guided_box = QCheckBox(f"按屏幕提示录制，约 {minutes} 分钟；用空格标记手指接触")
        self.guided_box.setChecked(guided)
        layout.addWidget(self.guided_box)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("开始采集")
        self.stop_button = QPushButton("停止并保存")
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        self.status_label = QLabel("等待新鲜相机帧，尚未采集")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.output_label = QLabel("输出目录：data/contact_recordings")
        self.output_label.setWordWrap(True)
        self.output_label.setTextFormat(Qt.TextFormat.PlainText)
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.output_label)
        annotation_note = QLabel(
            "停止后会生成标签文件：按提示录制时是来自提示和空格键的草稿，否则是空白模板。"
            "两种都需要人工复核后才能训练，动作提示和程序的判断不能直接作为正确答案。"
            "只保存点击开始之后的画面关键点。关闭后需手动重新启用系统鼠标。"
        )
        annotation_note.setWordWrap(True)
        layout.addWidget(annotation_note)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(lambda: self.stop_recording())
        self.consent.toggled.connect(self._consent_changed)
        self.guided_box.toggled.connect(self._refresh)
        self.participant.textChanged.connect(self._refresh)
        self.session.textChanged.connect(self._refresh)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._refresh)
        self.timer.start()
        self._refresh()

    @property
    def recorder_active(self):
        return self.recorder is not None

    @property
    def guided_active(self):
        return self.recorder is not None and self.protocol_start is not None

    def _protocol_done(self, now):
        return self.guided_active and now - self.protocol_start >= self.schedule[-1][1]

    def _fresh(self, now):
        return (
            self.source == "camera"
            and _finite(now)
            and self.last_capture is not None
            and 0 <= now - self.last_capture <= self.MAX_AGE
        )

    def _consent_changed(self):
        if self.recorder_active and not self.consent.isChecked():
            self._finish("consent_withdrawn")
        self._refresh()

    def start_recording(self):
        if self._closed or self.recorder_active:
            return
        if not self.consent.isChecked() or not self._fresh(self.clock()):
            self.status_label.setText("尚未开始：请明确同意保存关键点，并等待新鲜相机帧。")
            return
        participant, session = self.participant.text().strip(), self.session.text().strip()
        if not participant or not session:
            self.status_label.setText("尚未开始：请填写用户编号和采集场次。")
            return
        path = self.workspace / "data/contact_recordings" / f"contact_{time.time_ns()}.jsonl"
        try:
            recorder = ContactRecorder(path, participant, session, source="camera")
        except (OSError, ValueError) as error:
            self.error = str(error)
            self.status_label.setText(f"无法开始采集：{error}")
            return
        self.recorder = recorder
        self.recording_path = path
        self.annotation_path = self.summary_path = None
        self.protocol_path = self.bundle_path = None
        self.identity = (participant, session)
        self.count = self.missing_count = self.gaps = 0
        self.key_events, self.unmarked, self.key_down = [], [], False
        self.stop_reason = self.error = ""
        self.started_utc = datetime.now(timezone.utc).isoformat()
        # Keep the focus off the checkboxes and buttons: a stray Space must not toggle consent
        # or press a button, during the recording or after it ends.
        self.setFocus()
        if self.guided_box.isChecked():
            self.schedule = contact_protocol.timeline(self.protocol)
            self.protocol_start = self.clock()
            # Space must reach this window whichever control has the focus.
            self.grabKeyboard()
        self._refresh()

    def feed(self, frame, now, source="camera"):
        if self._closed:
            return
        if source != "camera":
            self.invalidate(now, source)
            return
        self.source = source
        if not isinstance(frame, HandFrame) or not _finite(now) or not _finite(frame.timestamp):
            self.last_capture = None
            self._finish("invalid_frame")
            self._refresh()
            return
        if not 0 <= now - frame.timestamp <= self.MAX_AGE:
            self.invalidate(now, source)
            return
        if self.last_input_timestamp is not None and frame.timestamp <= self.last_input_timestamp:
            self.last_capture = None
            self._finish("nonmonotonic_frame")
            self._refresh()
            return
        previous = self.last_input_timestamp
        self.last_capture = self.last_input_timestamp = frame.timestamp
        self.last_feed = now
        self.hand_present = bool(frame.landmarks)
        gap = previous is not None and frame.timestamp - previous > self.MAX_AGE
        if self.recorder_active and gap and not self.guided_active:
            self._finish("input_gap")
        elif self.recorder_active:
            self.gaps += gap
            try:
                self.recorder.add(frame)
                self.count = self.recorder.count
                if not frame.landmarks:
                    self.missing_count += 1
            except (OSError, ValueError, TypeError, OverflowError) as error:
                self.error = str(error)
                self._finish("write_error")
        self._refresh()

    def invalidate(self, now, source="camera"):
        if self._closed:
            return
        reason = "source_changed" if source != "camera" else "stale_input"
        self.source = source
        if reason == "stale_input" and self.guided_active:
            # Not fresh any more, but the session goes on; _refresh ends a long stall.
            self.last_capture = None
            self._refresh()
            return
        self.last_capture = self.last_input_timestamp = None
        self._finish(reason)
        self._refresh()

    def stop_recording(self, reason="user_stop"):
        self._finish(reason)
        self._refresh()

    def _mark(self, pressed):
        if self.guided_active and pressed != self.key_down:
            self.key_down = pressed
            self.key_events.append((self.clock(), pressed))
            self._refresh()

    def keyPressEvent(self, event):
        if self.guided_active and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._mark(True)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.guided_active and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._mark(False)
            return
        super().keyReleaseEvent(event)

    def _draft(self, candidate, started):
        _, frames = read_contact_recording(self.recording_path)
        labels = contact_protocol.draft_labels(
            [frame.timestamp - started for frame in frames],
            [bool(frame.landmarks) for frame in frames],
            self.schedule,
            [(moment - started, pressed) for moment, pressed in self.key_events],
        )
        draft_annotation(
            self.recording_path,
            candidate,
            contact_protocol.intervals(labels),
            contact_protocol.NAME,
        )

    def _save_protocol(self, started, ended, reason):
        candidate = self.recording_path.with_suffix(".protocol.json")
        self.unmarked = contact_protocol.unmarked(
            self.schedule,
            [(moment - started, pressed) for moment, pressed in self.key_events],
            ended - started,
        )
        payload = {
            "schema": "pinchpilot-contact-protocol-v1",
            "protocol": contact_protocol.NAME,
            "completed": reason == "protocol_complete",
            "time_unit": "seconds from the start of the session",
            "clock_start": started,  # Same clock as the frame timestamps.
            "ended": ended - started,
            "unlabelled_margins": {
                "settle": contact_protocol.SETTLE,
                "lead": contact_protocol.LEAD,
                "key_edge": contact_protocol.KEY_EDGE,
            },
            "steps": [
                {
                    "start": start,
                    "end": end,
                    "round": step.round,
                    "name": step.name,
                    "text": step.text,
                    "labels": list(step.labels),
                }
                for start, end, step in self.schedule
            ],
            "space_key": [[moment - started, pressed] for moment, pressed in self.key_events],
            # Steps that ask for Space marks and got none; their marked channel has no label.
            "steps_without_space": self.unmarked,
        }
        with candidate.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        self.protocol_path = candidate

    def _bundle(self):
        """One zip of the session's files, so that there is a single file to hand over."""
        tag = "".join(
            f"_{part}"
            for part in (re.sub(r"[^A-Za-z0-9-]+", "-", text).strip("-") for text in self.identity)
            if part
        )
        candidate = self.recording_path.with_name(f"{self.recording_path.stem}{tag[:60]}.zip")
        files = (self.recording_path, self.annotation_path, self.protocol_path, self.summary_path)
        with zipfile.ZipFile(candidate, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if path is not None:
                    archive.write(path, path.name)
        self.bundle_path = candidate

    def _finish(self, reason):
        if not self.recorder_active:
            return
        recorder, self.recorder = self.recorder, None
        started, self.protocol_start = self.protocol_start, None
        guided = started is not None
        ended = self.clock()
        if guided:
            self.releaseKeyboard()
            if self.key_down:
                self.key_down = False
                self.key_events.append((ended, False))
        self.stop_reason = reason
        self.count = recorder.count
        errors = [self.error] if self.error else []
        try:
            recorder.close()
        except (OSError, ValueError) as error:
            errors.append(f"关闭录制文件失败：{error}")
        if self.count:
            candidate = self.recording_path.with_suffix(".labels.json")
            try:
                if guided:
                    self._draft(candidate, started)
                else:
                    annotation_template(self.recording_path, candidate)
                self.annotation_path = candidate
            except (OSError, ValueError) as error:
                errors.append(f"未能生成待标注模板：{error}")
        if guided:
            try:
                self._save_protocol(started, ended, reason)
            except (OSError, ValueError) as error:
                errors.append(f"提示记录未保存：{error}")
        candidate = self.recording_path.with_suffix(".capture.json")
        try:
            with candidate.open("x", encoding="utf-8") as output:
                json.dump(
                    {
                        "schema": "pinchpilot-contact-capture-v1",
                        "source": "camera",
                        "started_utc": self.started_utc,
                        "ended_utc": datetime.now(timezone.utc).isoformat(),
                        "stop_reason": reason,
                        "partial": reason != ("protocol_complete" if guided else "user_stop"),
                        "guided": guided,
                        "camera_gaps": self.gaps,
                        "frames_written": self.count,
                        "missing_hand_frames": self.missing_count,
                        "recording": self.recording_path.name,
                        "annotation": self.annotation_path.name if self.annotation_path else None,
                        "stores_images": False,
                        "stores_landmarks": True,
                        "labels_reviewed": False,
                        "errors": errors,
                    },
                    output,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                output.write("\n")
            self.summary_path = candidate
        except (OSError, ValueError) as error:
            errors.append(f"停止摘要未保存：{error}")
        if guided and self.count:
            try:
                self._bundle()
            except (OSError, ValueError) as error:
                errors.append(f"打包失败：{error}")
        self.error = "；".join(errors)
        if self.count:
            # The next recording in this window is another session, not more of this one.
            name = _session_name()
            self.session.setText(name if name != self.identity[1] else f"{name}_2")

    def _refresh_guide(self, now):
        self.guide_panel.setVisible(self.guided_box.isChecked())
        for item in (self.step_bar, self.guide_status, self.key_label, self.next_label):
            item.setVisible(self.guided_active)
        if not self.guided_active:
            if self.bundle_path and self.stop_reason == "protocol_complete":
                self.prompt_label.setText("这一次录完了")
                missed = (
                    f"有 {len(self.unmarked)} 个步骤没有收到空格，这些步骤的标签留空；多的话请重录。"
                    if self.unmarked
                    else ""
                )
                self.note_label.setText(
                    f"{missed}把这个文件上传到数据仓库：{self.bundle_path.name}"
                )
            elif self.bundle_path:
                self.prompt_label.setText("提前停止了，这一次不完整")
                self.note_label.setText(
                    f"请重新录一次完整的。已录的部分保存在：{self.bundle_path.name}"
                )
            else:
                self.prompt_label.setText("点「开始采集」后，这里会一步步提示动作")
                self.note_label.setText("提示要求时，手指碰到就按住空格，离开就松开。")
            return
        elapsed = now - self.protocol_start
        index = contact_protocol.position(self.schedule, elapsed)
        if index is None:
            return
        start, end, step = self.schedule[index]
        rounds = max(item.round for _, _, item in self.schedule)
        self.prompt_label.setText(step.text)
        self.note_label.setText(step.note)
        self.step_bar.setValue(int(1000 * (elapsed - start) / (end - start)))
        self.guide_status.setText(
            (f"第 {step.round}/{rounds} 轮" if step.round else "准备")
            + f" · 已用 {_clock_text(elapsed)} / {_clock_text(self.schedule[-1][1])}"
            + f" · 这一步还剩 {math.ceil(end - elapsed)} 秒"
        )
        hand = "手在画面内" if self.hand_present else "没有检测到手"
        self.key_label.setText(("空格：按住中" if self.key_down else "空格：未按") + f" · {hand}")
        self.key_label.setStyleSheet("color:#5fd38d; font-weight:600;" if self.key_down else "")
        following = self.schedule[index + 1][2].text if index + 1 < len(self.schedule) else "结束"
        self.next_label.setText(f"下一步：{following}")

    def _refresh(self):
        if self._closed:
            return
        now = self.clock()
        if self._protocol_done(now):
            self._finish("protocol_complete")
        fresh = self._fresh(now)
        if self.recorder_active and not fresh:
            stalled = self.last_feed is None or now - self.last_feed > self.GUIDED_STALL
            if not self.guided_active or stalled:
                self._finish("stale_input")
        self._refresh_guide(now)
        active = self.recorder_active
        self.start_button.setEnabled(
            not active
            and fresh
            and self.consent.isChecked()
            and bool(self.participant.text().strip())
            and bool(self.session.text().strip())
        )
        self.stop_button.setEnabled(active)
        self.participant.setEnabled(not active)
        self.session.setEnabled(not active)
        self.guided_box.setEnabled(not active)
        self.count_label.setText(f"已写入 {self.count} 帧 · 其中无手 {self.missing_count} 帧")
        if active and not fresh:
            self.status_label.setText("相机画面中断，等待恢复；超过 5 秒会停止并保存")
        elif active:
            self.status_label.setText("正在采集新的相机帧 · 不保存图像、不自动生成真值")
        elif self.stop_reason:
            self.status_label.setText(
                f"已停止：{STOP_LABELS.get(self.stop_reason, self.stop_reason)}。"
                + (f"保存异常：{self.error}" if self.error else "已保留采集文件。")
            )
        elif self.error:
            self.status_label.setText(f"无法开始采集：{self.error}")
        else:
            self.status_label.setText(
                "相机帧已就绪 · 勾选同意后点击开始" if fresh else "等待新鲜相机帧，尚未采集"
            )
        if self.bundle_path:
            self.output_label.setText(
                f"文件夹：{self.bundle_path.parent}\n上传这一个文件：{self.bundle_path.name}"
            )
        elif self.recording_path:
            paths = [f"关键点：{self.recording_path}"]
            if self.annotation_path:
                paths.append(f"待标注模板：{self.annotation_path}")
            elif not active and not self.count:
                paths.append("没有采到新帧，未生成标注模板。")
            self.output_label.setText("\n".join(paths))

    def closeEvent(self, event):
        if not self._closed:
            self.timer.stop()
            self._finish("window_closed")
            self._closed = True
            self.closed.emit()
        super().closeEvent(event)
