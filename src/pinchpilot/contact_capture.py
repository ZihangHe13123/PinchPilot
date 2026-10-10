"""Explicit, local keypoint collection; owns no camera, training or native input."""

import json
import math
import re
import statistics
import time
import zipfile
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
    palm_angle,
    read_contact_recording,
)
from .contact_demo import view_for
from .contact_demo_view import GestureDemo, caption
from .domain import HandFrame
from .widgets import fit_to_screen

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


def _session_name(side=False):
    return datetime.now().strftime(("side" if side else "session") + "_%Y%m%d_%H%M%S")


def _clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class ContactCaptureWindow(QWidget):
    """Record only new, fresh camera frames after an explicit Start action.

    ``source='camera'`` is supplied by the trusted host; this window never opens
    hardware. A source label alone does not attest real-human provenance in tests.
    Empty HandFrames are part of the continuous recording, not filtered out.

    A guided session goes through the clips of ``contact_protocol`` one at a time. The person
    reads what a clip asks for and starts it with Space; it stops by itself. Only frames of
    running clips are written. While a clip runs, the Space key marks touching fingers. The
    session saves draft labels from the prompts and those marks, which still need review.
    """

    closed = Signal()
    MAX_AGE = 0.25
    GUIDED_STALL = 5.0  # A running clip rides out a camera stall up to this many seconds.
    START_LOCK = 1.2  # After a clip ends, Space cannot start the next one for this long.

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
        self.clips = []
        self.clip_index = 0  # The clip waiting to be started, or running.
        self.clip_range = None  # (first, end) schedule entries of the running clip.
        self.last_range = None  # The same for the clip recorded last, until it is redone.
        self.last_marks = None  # Space holds counted in it, or None when it asked for none.
        self.discarded, self.redone = set(), 0
        self.notice = ""
        self.lock_until = 0.0
        self.recent_angles = deque(maxlen=9)  # palm angle of the last frames with a hand
        self.clip_angles = []  # palm angle of the frames written in the running clip
        self.step_angles = {}  # schedule index -> median palm angle of its clip
        self.step_poses = {}  # schedule index -> the pose its clip asked for
        self.last_angle = None  # (median angle, pose asked for) of the clip recorded last
        self.pose_warned = None  # clip index whose start was held back once for its pose
        self.last_feed = None
        self.setWindowTitle("PinchPilot · ML 关键点采集")
        fit_to_screen(self, *((920, 820) if guided else (560, 480)))
        self.setMinimumSize(440, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        heading = QLabel("ML 动作采集 · 系统鼠标已暂停")
        heading.setStyleSheet("font-size:18px; font-weight:600;")
        layout.addWidget(heading)
        self.scope_note = QLabel(
            "只保存当前控制手的连续 21 点坐标、时间与左右手分类；无手帧也保留。"
            "不保存图像，不会自动启动相机或训练模型。"
        )
        self.scope_note.setWordWrap(True)
        layout.addWidget(self.scope_note)
        self.guide_panel = QWidget()
        guide = QVBoxLayout(self.guide_panel)
        guide.setContentsMargins(0, 6, 0, 6)
        self.notice_label = QLabel()
        self.notice_label.setWordWrap(True)
        self.notice_label.setStyleSheet("font-size:18px; font-weight:700; color:#9fd6ff;")
        self.pose_label = QLabel()
        self.pose_label.setWordWrap(True)
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
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        guide.addWidget(self.notice_label)
        guide.addWidget(self.pose_label)
        columns, words, picture = QHBoxLayout(), QVBoxLayout(), QVBoxLayout()
        for item in (
            self.prompt_label,
            self.note_label,
            self.step_bar,
            self.guide_status,
            self.key_label,
            self.next_label,
            self.result_label,
        ):
            words.addWidget(item)
        words.addStretch()
        # Beside the words: a hand that acts the clip out at the pace it should be done.
        self.demo = GestureDemo()
        self.demo.setMinimumSize(210, 210)
        self.demo_caption = QLabel()
        self.demo_caption.setWordWrap(True)
        self.demo_caption.setStyleSheet("color:#9fb4c4;")
        picture.addWidget(self.demo, 1)
        picture.addWidget(self.demo_caption)
        columns.addLayout(words, 3)
        columns.addLayout(picture, 2)
        guide.addLayout(columns)
        self.demo_since = 0.0
        self.demo_timer = QTimer(self)
        self.demo_timer.setInterval(33)
        self.demo_timer.timeout.connect(lambda: self._refresh_demo(self.clock()))
        clip_buttons = QHBoxLayout()
        self.begin_button = QPushButton("开始这一段（空格）")
        self.begin_button.setObjectName("primary")
        self.redo_button = QPushButton("重录上一段（退格）")
        for button in (self.begin_button, self.redo_button):
            # Keys are handled by the window, so that Space never presses a button.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            clip_buttons.addWidget(button)
        guide.addLayout(clip_buttons)
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
        planned = contact_protocol.clips(self.protocol)
        minutes = round(sum(clip.seconds for clip in planned) / 60)
        self.guided_box = QCheckBox(
            f"按屏幕提示分段录制：共 {len(planned)} 段、约 {minutes} 分钟，每段读完提示再开始"
        )
        self.guided_box.setChecked(guided)
        layout.addWidget(self.guided_box)
        side = contact_protocol.clips(contact_protocol.side_session())
        minutes = round(sum(clip.seconds for clip in side) / 60)
        self.side_box = QCheckBox(
            f"只补录侧对：共 {len(side)} 段、约 {minutes} 分钟，给已经录过完整一次的人"
        )
        layout.addWidget(self.side_box)
        self.side_session = False  # Whether the running or last session was side-on only.
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
        self.annotation_note = annotation_note = QLabel(
            "停止后会生成标签文件：按提示录制时是来自提示和空格键的草稿，否则是空白模板。"
            "两种都需要人工复核后才能训练，动作提示和程序的判断不能直接作为正确答案。"
            "只保存点击开始之后的画面关键点。关闭后需手动重新启用系统鼠标。"
        )
        annotation_note.setWordWrap(True)
        layout.addWidget(annotation_note)
        layout.addStretch()
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(lambda: self.stop_recording())
        self.begin_button.clicked.connect(self.start_clip)
        self.redo_button.clicked.connect(self.redo_clip)
        self.consent.toggled.connect(self._consent_changed)
        self.guided_box.toggled.connect(self._refresh)
        self.side_box.toggled.connect(self._side_toggled)
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

    @property
    def clip_running(self):
        return self.guided_active and self.clip_range is not None

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
        self.schedule, self.clip_index, self.clip_range = [], 0, None
        self.last_range = self.last_marks = None
        self.discarded, self.redone, self.notice = set(), 0, ""
        self.lock_until = 0.0
        self.clip_angles, self.step_angles, self.step_poses = [], {}, {}
        self.last_angle = self.pose_warned = None
        self.stop_reason = self.error = ""
        self.started_utc = datetime.now(timezone.utc).isoformat()
        # Keep the focus off the checkboxes and buttons: a stray Space must not toggle consent
        # or press a button, during the recording or after it ends.
        self.setFocus()
        if self.guided_box.isChecked():
            self.side_session = self.side_box.isChecked()
            steps = contact_protocol.side_session() if self.side_session else self.protocol
            self.clips = contact_protocol.clips(steps)
            self.protocol_start = self.demo_since = self.clock()
            self.demo_timer.start()
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
        angle = palm_angle(frame)
        if angle is None:
            self.recent_angles.clear()
        else:
            self.recent_angles.append(angle)
            if self.clip_running:
                self.clip_angles.append(angle)
        if self.guided_active and not self.schedule and frame.landmarks:
            # Before the first clip, with the palm to the camera: draw the example as the
            # hand on screen looks, thumb on the same side.
            self.demo.mirror = frame.landmarks[4][0] > frame.landmarks[17][0]
        gap = previous is not None and frame.timestamp - previous > self.MAX_AGE
        if self.recorder_active and gap and not self.guided_active:
            self._finish("input_gap")
        elif self.guided_active and not self.clip_running:
            pass  # Between clips nothing is written; the person is reading the next prompt.
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

    def start_clip(self):
        """Begin recording the clip that is waiting. Its steps are scheduled from now."""
        now = self.clock()
        if not self.guided_active or self.clip_running or self.clip_index >= len(self.clips):
            return
        if not self._fresh(now):
            self.result_label.setText("相机画面还没准备好，稍等一下再开始。")
            return
        clip = self.clips[self.clip_index]
        if self._pose_ok(clip) is False and self.pose_warned != self.clip_index:
            # The hand is not held as this round asks. Say so once; a second start goes ahead,
            # because the angle is only an estimate.
            self.pose_warned = self.clip_index
            return self._refresh()
        self.notice = clip.notice or self.notice
        moment, first = now - self.protocol_start, len(self.schedule)
        for step in clip.steps:
            self.schedule.append((moment, moment + step.seconds, step))
            moment += step.seconds
        self.clip_range = (first, len(self.schedule))
        for index in range(first, len(self.schedule)):
            self.step_poses[index] = clip.pose
        self.clip_angles = []
        self._refresh()

    def _protocol_name(self):
        return contact_protocol.SIDE_NAME if self.side_session else contact_protocol.NAME

    def _side_toggled(self, side):
        """Name the session after what it records, unless the person typed a name."""
        if re.fullmatch(r"(session|side)_\d{8}_\d{6}(_2)?", self.session.text().strip()):
            self.session.setText(_session_name(side))
        self._refresh()

    @property
    def palm_now(self):
        """The palm's angle to the camera over the last few frames, or None."""
        return statistics.median(self.recent_angles) if len(self.recent_angles) >= 3 else None

    def _pose_ok(self, clip, angle=None):
        """Whether the hand is turned as the clip's round asks; None when that cannot be said."""
        angle = self.palm_now if angle is None else angle
        if clip.pose is None or angle is None:
            return None
        return clip.pose[0] <= angle <= clip.pose[1]

    def _end_clip(self):
        first, end = self.clip_range
        finished = self.schedule[end - 1][1]
        if self.key_down:
            self.key_down = False
            self.key_events.append((self.protocol_start + finished, False))
        begun = self.schedule[first][0]
        clip = self.clips[self.clip_index]
        holds = sum(
            pressed and begun <= moment - self.protocol_start < finished
            for moment, pressed in self.key_events
        )
        self.last_marks = holds if clip.uses_key else None
        angle = statistics.median(self.clip_angles) if self.clip_angles else None
        for index in range(first, end):
            self.step_angles[index] = angle
        self.last_angle = (angle, clip.pose)
        self.last_range, self.clip_range = self.clip_range, None
        self.clip_index += 1
        self.demo_since = self.clock()
        # Someone still tapping when the clip ends must not start the next one by accident.
        # A key that is simply held only repeats, and repeats are ignored anyway.
        self.lock_until = self.clock() + self.START_LOCK

    def redo_clip(self):
        """Throw away the clip recorded last and record it again."""
        if not self.guided_active or self.clip_running or self.last_range is None:
            return
        for index in range(*self.last_range):
            start, end, step = self.schedule[index]
            # Its frames stay in the file without labels; what it asked for is kept by name.
            self.schedule[index] = (start, end, replace(step, labels=(None,) * len(step.labels)))
            self.discarded.add(index)
        self.last_range = self.last_marks = self.last_angle = None
        self.clip_index -= 1
        self.redone += 1
        self.demo_since = self.clock()
        self._refresh()

    def _space(self, pressed):
        if self.clip_running:
            # The press that started the clip was never recorded, so its release is not either.
            if pressed != self.key_down:
                self.key_down = pressed
                self.key_events.append((self.clock(), pressed))
        elif pressed and self.clock() >= self.lock_until:
            self.start_clip()
        self._refresh()

    def keyPressEvent(self, event):
        if self.guided_active and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                return self._space(True)
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                return self.start_clip()
            if event.key() == Qt.Key.Key_Backspace:
                return self.redo_clip()
        if self.guided_active and event.key() == Qt.Key.Key_Space:
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.guided_active and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._space(False)
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
            self._protocol_name(),
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
            "protocol": self._protocol_name(),
            "completed": reason == "protocol_complete",
            "time_unit": "seconds from the start of the session",
            "clock_start": started,  # Same clock as the frame timestamps.
            "ended": ended - started,
            "unlabelled_margins": {
                "settle": contact_protocol.SETTLE,
                "lead": contact_protocol.LEAD,
                "key_edge": contact_protocol.KEY_EDGE,
            },
            "clips_planned": len(self.clips),
            "clips_recorded": self.clip_index,
            "clips_redone": self.redone,
            # Steps are listed as they were recorded, with the reading pauses between clips.
            # A clip that was redone stays in the list, marked and without labels.
            "steps": [
                {
                    "start": start,
                    "end": end,
                    "round": step.round,
                    "name": step.name,
                    "text": step.text,
                    "labels": list(step.labels),
                    "discarded": index in self.discarded,
                    # How the hand was really held: median angle between the palm and the
                    # camera over the clip, in degrees, and the range its round asked for.
                    "palm_angle": self.step_angles.get(index),
                    "pose": list(step_pose) if step_pose else None,
                }
                for index, (start, end, step), step_pose in (
                    (index, entry, self.step_poses.get(index))
                    for index, entry in enumerate(self.schedule)
                )
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
            self.demo_timer.stop()
            self.releaseKeyboard()
            if self.key_down:
                self.key_down = False
                self.key_events.append((ended, False))
            self.clip_range = None
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
            name = _session_name(self.side_box.isChecked())
            self.session.setText(name if name != self.identity[1] else f"{name}_2")

    def _demo_step(self, now):
        """(step, seconds into it) that the example hand should be showing."""
        if self.clip_running:
            elapsed = now - self.protocol_start
            first, end = self.clip_range
            index = next((i for i in range(first, end) if elapsed < self.schedule[i][1]), end - 1)
            return self.schedule[index][2], elapsed - self.schedule[index][0]
        # While the person reads, the hand acts the whole clip out, over and over.
        clip = self.clips[self.clip_index]
        moment = (now - self.demo_since) % (clip.seconds + 1.0)
        for step in clip.steps:
            if moment < step.seconds or step is clip.steps[-1]:
                return step, min(moment, step.seconds)
            moment -= step.seconds

    def _refresh_demo(self, now):
        if not self.guided_active or self.clip_index >= len(self.clips):
            return
        self.demo.show_step(*self._demo_step(now))

    def _refresh_guide(self, now, fresh):
        self.guide_panel.setVisible(self.guided_box.isChecked())
        session = self.guided_active
        # During a session the general explanations and the two choices that can no longer
        # change make room for the prompt and the hand.
        for item in (self.scope_note, self.annotation_note, self.guided_box, self.side_box):
            item.setVisible(not session)
        for item in (
            self.step_bar,
            self.guide_status,
            self.key_label,
            self.result_label,
            self.begin_button,
            self.redo_button,
            self.demo,
            self.demo_caption,
            self.pose_label,
        ):
            item.setVisible(session)
        if not session:
            self.notice_label.setVisible(False)
            self.next_label.setVisible(False)
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
                self.prompt_label.setText("点「开始采集」后，这里会一段一段提示动作")
                self.note_label.setText(
                    "每段先读提示，读完按空格开始，到时间自动停。"
                    "提示要求时，手指碰到就按住空格，离开就松开。"
                )
            return
        clip = self.clips[self.clip_index]
        rounds = max(item.round for item in self.clips)
        place = f"第 {self.clip_index + 1}/{len(self.clips)} 段"
        place = (
            f"侧对补录 · {place}" if self.side_session else f"第 {clip.round}/{rounds} 轮 · {place}"
        )
        notice = clip.notice or self.notice
        self.notice_label.setText(notice)
        self.notice_label.setVisible(bool(notice))
        self.demo.view = view_for(clip.pose)
        angle, fits = self.palm_now, self._pose_ok(clip)
        measured = "看不到手" if angle is None else f"现在约 {angle:.0f}°"
        if clip.pose is None:
            self.pose_label.setText(f"手掌和镜头的夹角：{measured}（这一轮不限）")
            self.pose_label.setStyleSheet("color:#9fb4c4;")
        else:
            wanted = (
                f"{clip.pose[0]}° 以上"
                if clip.pose[1] >= 90
                else f"{clip.pose[1]}° 以内"
                if clip.pose[0] <= 0
                else f"{clip.pose[0]}° 到 {clip.pose[1]}°"
            )
            mark = "" if fits is None else " ✓" if fits else " ✗ 还不对，请照上面的朝向调整"
            self.pose_label.setText(f"手掌和镜头的夹角：{measured}，这一轮需要 {wanted}{mark}")
            self.pose_label.setStyleSheet(
                "color:#f0b35a; font-weight:600;"
                if fits is False
                else "color:#5fd38d;"
                if fits
                else "color:#9fb4c4;"
            )
        hand = "手在画面内" if self.hand_present else "没有检测到手"
        self.key_label.setText(("空格：按住中" if self.key_down else "空格：未按") + f" · {hand}")
        self.key_label.setStyleSheet("color:#5fd38d; font-weight:600;" if self.key_down else "")
        self.begin_button.setEnabled(not self.clip_running and fresh)
        self.begin_button.setText("正在录这一段…" if self.clip_running else "开始这一段（空格）")
        self.redo_button.setEnabled(not self.clip_running and self.last_range is not None)
        self._refresh_demo(now)
        self.demo_caption.setText(caption(self.demo.step, len(clip.steps) > 1))
        if self.clip_running:
            elapsed = now - self.protocol_start
            first, end = self.clip_range
            begun, finished = self.schedule[first][0], self.schedule[end - 1][1]
            index = next((i for i in range(first, end) if elapsed < self.schedule[i][1]), end - 1)
            _, stop, step = self.schedule[index]
            self.prompt_label.setText(step.text)
            self.note_label.setText(step.note)
            self.step_bar.setValue(int(1000 * (elapsed - begun) / (finished - begun)))
            self.guide_status.setText(f"{place} · 正在录，还剩 {math.ceil(finished - elapsed)} 秒")
            self.next_label.setVisible(index + 1 < end)
            if index + 1 < end:
                self.next_label.setText(
                    f"{math.ceil(stop - elapsed)} 秒后：{self.schedule[index + 1][2].text}"
                )
            self.result_label.setText("")
            return
        self.prompt_label.setText(clip.text)
        self.note_label.setText(clip.steps[0].note if len(clip.steps) == 1 else "")
        self.step_bar.setValue(0)
        self.guide_status.setText(f"{place} · 这一段 {clip.seconds:.0f} 秒 · 读完后按空格开始")
        self.next_label.setVisible(False)
        self.result_label.setStyleSheet("")
        turned_wrong = (
            self.last_angle is not None
            and self._pose_ok(SimpleNamespace(pose=self.last_angle[1]), self.last_angle[0]) is False
        )
        if not fresh:
            self.result_label.setText("相机画面中断，恢复后才能开始这一段。")
        elif self.pose_warned == self.clip_index and fits is False:
            self.result_label.setStyleSheet("color:#f0b35a; font-weight:600;")
            self.result_label.setText(
                "手的朝向和这一轮要求的不一样，所以还没开始。调整好再按空格；"
                "确实摆不到的话，再按一次空格也会开始。"
            )
        elif turned_wrong:
            self.result_label.setStyleSheet("color:#f0b35a; font-weight:600;")
            self.result_label.setText(
                f"上一段手掌和镜头的夹角约 {self.last_angle[0]:.0f}°，不符合这一轮的要求。"
                "请调整朝向后按退格键重录。"
            )
        elif self.last_range is None:
            self.result_label.setText("上一段已作废，现在重录这一段。" if self.redone else "")
        elif self.last_marks is None:
            self.result_label.setText("上一段录完了。没做好可以按退格键重录。")
        elif self.last_marks:
            self.result_label.setText(
                f"上一段收到 {self.last_marks} 次空格。次数不对或没做好，可以按退格键重录。"
            )
        else:
            self.result_label.setStyleSheet("color:#f0b35a; font-weight:600;")
            self.result_label.setText(
                "上一段没有收到空格。请按退格键重录；如果按了却没收到，先点一下这个窗口。"
            )

    def _refresh(self):
        if self._closed:
            return
        now = self.clock()
        if (
            self.clip_running
            and now - self.protocol_start >= self.schedule[self.clip_range[1] - 1][1]
        ):
            self._end_clip()
            if self.clip_index >= len(self.clips):
                self._finish("protocol_complete")
        fresh = self._fresh(now)
        if self.recorder_active and not fresh:
            stalled = self.last_feed is None or now - self.last_feed > self.GUIDED_STALL
            # Between clips a stalled camera only blocks the next start.
            if not self.guided_active or (self.clip_running and stalled):
                self._finish("stale_input")
        self._refresh_guide(now, fresh)
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
        self.side_box.setEnabled(not active and self.guided_box.isChecked())
        self.count_label.setText(f"已写入 {self.count} 帧 · 其中无手 {self.missing_count} 帧")
        if active and not fresh:
            self.status_label.setText("相机画面中断，等待恢复；录制中超过 5 秒会停止并保存")
        elif self.guided_active and not self.clip_running:
            self.status_label.setText("两段之间不录制 · 读完提示再开始")
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
