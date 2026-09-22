"""Explicit, local keypoint collection; owns no camera, training or native input."""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .contact_data import ContactRecorder, annotation_template
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
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ContactCaptureWindow(QWidget):
    """Record only new, fresh camera frames after an explicit Start action.

    ``source='camera'`` is supplied by the trusted host; this window never opens
    hardware. A source label alone does not attest real-human provenance in tests.
    Empty HandFrames are part of the continuous recording, not filtered out.
    """

    closed = Signal()
    MAX_AGE = 0.25

    def __init__(self, workspace, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName("practice")
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
        self.setWindowTitle("PinchPilot · ML 关键点采集")
        self.resize(560, 480)
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
        fields = QFormLayout()
        self.participant = QLineEdit("P01")
        self.participant.setMaxLength(80)
        self.participant.setPlaceholderText("使用匿名编号，例如 P01")
        self.session = QLineEdit(datetime.now().strftime("session_%Y%m%d_%H%M%S"))
        self.session.setMaxLength(80)
        fields.addRow("用户编号", self.participant)
        fields.addRow("采集场次", self.session)
        layout.addLayout(fields)
        self.consent = QCheckBox("同意保存 21 点坐标，不记录图像；之后需人工标注")
        layout.addWidget(self.consent)
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
            "停止后会生成空白标签模板，需要人工确认每段实际动作后才能训练。"
            "动作提示和程序的判断不能直接作为正确答案。"
            "只保存点击开始之后的画面关键点。关闭后需手动重新启用系统鼠标。"
        )
        annotation_note.setWordWrap(True)
        layout.addWidget(annotation_note)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(lambda: self.stop_recording())
        self.consent.toggled.connect(self._consent_changed)
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
        self.count = self.missing_count = 0
        self.stop_reason = self.error = ""
        self.started_utc = datetime.now(timezone.utc).isoformat()
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
        if (
            self.recorder_active
            and previous is not None
            and frame.timestamp - previous > self.MAX_AGE
        ):
            self._finish("input_gap")
        elif self.recorder_active:
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
        self.last_capture = self.last_input_timestamp = None
        self._finish(reason)
        self._refresh()

    def stop_recording(self, reason="user_stop"):
        self._finish(reason)
        self._refresh()

    def _finish(self, reason):
        if not self.recorder_active:
            return
        recorder, self.recorder = self.recorder, None
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
                annotation_template(self.recording_path, candidate)
                self.annotation_path = candidate
            except (OSError, ValueError) as error:
                errors.append(f"未能生成待标注模板：{error}")
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
                        "partial": reason != "user_stop",
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
        self.error = "；".join(errors)

    def _refresh(self):
        if self._closed:
            return
        fresh = self._fresh(self.clock())
        if self.recorder_active and not fresh:
            self._finish("stale_input")
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
        self.count_label.setText(f"已写入 {self.count} 帧 · 其中无手 {self.missing_count} 帧")
        if active:
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
        if self.recording_path:
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
