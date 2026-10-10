"""Offscreen engineering fixtures only; the camera source is simulated, not human data."""

import json
import math
import os
import zipfile
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QWidget

from pinchpilot.contact_capture import ContactCaptureWindow
from pinchpilot.contact_data import palm_angle, read_contact_recording
from pinchpilot.contact_protocol import KEY, Step
from pinchpilot.domain import HandFrame
from pinchpilot.tripod_demo import synthetic_tripod


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    item = ContactCaptureWindow(tmp_path)
    item.test_now = 100.0
    item.clock = lambda: item.test_now
    item.timer.stop()
    yield item
    item.close()
    application.processEvents()


def feed(window, *, missing=False, source="camera", dt=1 / 30):
    window.test_now += dt
    frame = HandFrame(window.test_now) if missing else synthetic_tripod(window.test_now)
    window.feed(frame, window.test_now, source)
    return frame


def start(window):
    feed(window)
    window.consent.setChecked(True)
    window.start_recording()
    assert window.recorder_active


def test_needs_explicit_consent_identity_and_fresh_camera(window, tmp_path):
    assert not window.recorder_active and not window.start_button.isEnabled()
    feed(window)
    window.start_recording()
    assert not window.recorder_active and not (tmp_path / "data").exists()
    window.consent.setChecked(True)
    window.participant.setText(" ")
    window.start_recording()
    assert not window.recorder_active
    window.participant.setText("engineering-P01")
    feed(window, source="synthetic_demo")
    window.start_recording()
    assert not window.recorder_active
    feed(window)
    window.test_now += 0.251
    window.start_recording()
    assert not window.recorder_active
    feed(window)
    assert window.start_button.isEnabled()
    window.start_recording()
    assert window.recorder_active
    assert not window.participant.isEnabled() and not window.session.isEnabled()


def test_records_every_new_frame_including_missing_without_backfill_or_labels(window):
    start(window)
    expected = [feed(window), feed(window, missing=True), feed(window)]
    window.stop_recording()
    metadata, frames = read_contact_recording(window.recording_path)
    assert frames == expected and window.count == 3 and window.missing_count == 1
    assert metadata["source"] == "camera" and metadata["labels_inferred"] is False
    assert not metadata["stores_images"]
    labels = json.loads(window.annotation_path.read_text(encoding="utf-8"))
    assert labels["reviewed"] is False and labels["label_source"] == "unreviewed"
    assert labels["intervals"] == [
        {"start": 0, "end": 3, "middle": None, "index": None, "ring": None}
    ]
    report = json.loads(window.summary_path.read_text(encoding="utf-8"))
    assert report["frames_written"] == 3 and report["missing_hand_frames"] == 1
    assert report["stop_reason"] == "user_stop" and not report["partial"]


@pytest.mark.parametrize("failure", ["duplicate", "older", "stale", "future", "gap", "source"])
def test_discontinuity_stops_and_preserves_only_valid_prefix(window, failure):
    start(window)
    frame = feed(window)
    if failure == "duplicate":
        window.feed(frame, window.test_now)
    elif failure == "older":
        window.feed(replace(frame, timestamp=frame.timestamp - 0.01), window.test_now)
    elif failure == "stale":
        window.test_now += 0.26
        window.feed(frame, window.test_now)
    elif failure == "future":
        window.feed(replace(frame, timestamp=frame.timestamp + 1), window.test_now)
    elif failure == "gap":
        feed(window, dt=0.26)
    else:
        feed(window, source="synthetic_demo")
    assert not window.recorder_active
    assert read_contact_recording(window.recording_path)[1] == [frame]
    summary = json.loads(window.summary_path.read_text(encoding="utf-8"))
    assert summary["partial"] and summary["frames_written"] == 1
    assert window.annotation_path.exists()


def test_missing_frames_alone_are_valid_recording_content(window):
    start(window)
    frames = [feed(window, missing=True) for _ in range(4)]
    assert window.recorder_active
    window.stop_recording()
    assert read_contact_recording(window.recording_path)[1] == frames
    assert window.missing_count == 4


def test_timer_expiry_stops_without_inventing_missing_frame(window):
    start(window)
    frame = feed(window)
    window.test_now += 0.26
    window._refresh()
    assert not window.recorder_active and not window.start_button.isEnabled()
    assert window.stop_reason == "stale_input"
    assert read_contact_recording(window.recording_path)[1] == [frame]


def test_invalidate_does_not_refresh_or_create_a_fake_frame(window):
    start(window)
    frame = feed(window)
    window.invalidate(window.test_now + 0.05, "camera")
    assert not window.recorder_active and not window.start_button.isEnabled()
    assert window.last_capture is None
    assert read_contact_recording(window.recording_path)[1] == [frame]
    files = set(window.recording_path.parent.iterdir())
    window.invalidate(window.test_now + 0.1, "none")
    assert set(window.recording_path.parent.iterdir()) == files


def test_withdrawing_consent_stops_recording_and_requires_new_start(window):
    start(window)
    feed(window)
    window.consent.setChecked(False)
    assert not window.recorder_active and window.stop_reason == "consent_withdrawn"
    feed(window)
    assert window.count == 1
    assert not window.start_button.isEnabled()


def test_zero_frame_recording_has_no_fabricated_annotation(window):
    start(window)
    window.stop_recording()
    assert window.count == 0 and window.annotation_path is None
    assert len(window.recording_path.read_text(encoding="utf-8").splitlines()) == 1
    assert "没有采到新帧" in window.output_label.text()


def test_write_failure_is_visible_and_does_not_resume_automatically(window, monkeypatch):
    start(window)
    frame = feed(window)

    def fail(_):
        raise OSError("test disk full")

    monkeypatch.setattr(window.recorder, "add", fail)
    feed(window)
    assert not window.recorder_active and window.stop_reason == "write_error"
    assert "test disk full" in window.status_label.text()
    assert read_contact_recording(window.recording_path)[1] == [frame]
    feed(window)
    assert window.count == 1


def test_template_failure_preserves_recording_and_reports_unreviewed_error(window, monkeypatch):
    start(window)
    frame = feed(window)

    def fail(*_):
        raise OSError("test annotation unavailable")

    monkeypatch.setattr("pinchpilot.contact_capture.annotation_template", fail)
    window.stop_recording()
    assert read_contact_recording(window.recording_path)[1] == [frame]
    assert window.annotation_path is None
    assert "未能生成待标注模板" in window.status_label.text()
    assert json.loads(window.summary_path.read_text(encoding="utf-8"))["labels_reviewed"] is False


def test_start_failure_stays_visible_after_periodic_refresh(window, monkeypatch):
    feed(window)
    window.consent.setChecked(True)

    def fail(*args, **kwargs):
        raise OSError("test output permission denied")

    monkeypatch.setattr("pinchpilot.contact_capture.ContactRecorder", fail)
    window.start_recording()
    window._refresh()
    assert not window.recorder_active
    assert "test output permission denied" in window.status_label.text()


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), True])
def test_invalid_timestamp_stops_before_writing(window, stamp):
    start(window)
    frame = feed(window)
    window.feed(HandFrame(stamp), window.test_now)
    assert not window.recorder_active and window.stop_reason == "invalid_frame"
    assert read_contact_recording(window.recording_path)[1] == [frame]


def test_close_saves_partial_emits_once_and_destroys_child(application, tmp_path):
    parent = QWidget()
    item = ContactCaptureWindow(tmp_path, parent)
    item.test_now = 100.0
    item.clock = lambda: item.test_now
    start(item)
    frame = feed(item)
    spy = QSignalSpy(item.closed)
    item.close()
    path, summary = item.recording_path, item.summary_path
    item.close()
    assert spy.count() == 1 and not item.recorder_active
    assert read_contact_recording(path)[1] == [frame]
    assert json.loads(summary.read_text(encoding="utf-8"))["stop_reason"] == "window_closed"
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    assert not parent.findChildren(ContactCaptureWindow)
    parent.close()


# ---- Guided sessions: self-paced clips, the Space key and draft labels.

SHORT = [
    Step("pose", "这一轮的手部朝向：正对镜头", 8.0, (None, None, None), round=1),
    Step("hold", "捏住：拇指和中指", 2.0, (1, 0, 0), round=1),
    Step("tap", "保持捏住。食指点拇指", 3.0, (1, KEY, 0), "碰到时按住空格", 1),
]


@pytest.fixture
def guided(application, tmp_path):
    item = ContactCaptureWindow(tmp_path, guided=True)
    item.test_now = 100.0
    item.clock = lambda: item.test_now
    item.timer.stop()
    item.protocol = SHORT
    yield item
    item.close()
    application.processEvents()


def key(window, pressed, which=Qt.Key.Key_Space, repeat=False):
    kind = QEvent.Type.KeyPress if pressed else QEvent.Type.KeyRelease
    event = QKeyEvent(kind, which, Qt.KeyboardModifier.NoModifier, "", repeat)
    (window.keyPressEvent if pressed else window.keyReleaseEvent)(event)


def space(window, pressed, repeat=False):
    key(window, pressed, repeat=repeat)


def tap(window, which=Qt.Key.Key_Space):
    key(window, True, which)
    key(window, False, which)


def run(window, seconds):
    return [feed(window) for _ in range(round(seconds * 30))]


def begin(window):
    """Wait out the lock after the previous clip, then start the next one with Space."""
    run(window, window.START_LOCK + 0.1)
    tap(window)
    assert window.clip_running


def frame_labels(window):
    spec = json.loads(window.annotation_path.read_text(encoding="utf-8"))
    rows = []
    for interval in spec["intervals"]:
        rows += [(interval["middle"], interval["index"], interval["ring"])] * (
            interval["end"] - interval["start"]
        )
    return spec, rows


def test_guided_mode_is_opt_in_and_free_recording_ignores_the_space_key(window, guided):
    assert not window.guided_box.isChecked() and window.guide_panel.isHidden()
    assert guided.guided_box.isChecked() and not guided.guide_panel.isHidden()
    start(window)
    space(window, True)
    assert not window.guided_active and window.key_events == []
    window.stop_recording()
    assert window.protocol_path is None and window.bundle_path is None
    assert not json.loads(window.summary_path.read_text(encoding="utf-8"))["guided"]


def test_guided_session_waits_for_each_start_and_records_only_running_clips(guided):
    guided.participant.setText("P02")
    guided.session.setText("2")
    start(guided)
    began = guided.test_now
    assert guided.guided_active and not guided.clip_running
    assert not guided.guided_box.isEnabled() and guided.focusWidget() is guided
    # The first clip is on screen with the notice before it, and nothing is recorded yet.
    assert guided.prompt_label.text() == "捏住：拇指和中指"
    assert guided.notice_label.text() == "这一轮的手部朝向：正对镜头"
    assert "第 1/1 轮 · 第 1/2 段 · 这一段 2 秒 · 读完后按空格开始" in guided.guide_status.text()
    run(guided, 3.0)
    assert guided.count == 0 and guided.recorder.count == 0 and guided.schedule == []
    assert "两段之间不录制" in guided.status_label.text()

    tap(guided)
    first_start = guided.test_now - began
    assert guided.clip_running and "正在录，还剩 2 秒" in guided.guide_status.text()
    assert not guided.begin_button.isEnabled() and not guided.redo_button.isEnabled()
    run(guided, 2.1)
    assert not guided.clip_running and guided.clip_index == 1 and 59 <= guided.count <= 61
    written = guided.count
    assert guided.prompt_label.text() == "保持捏住。食指点拇指"
    assert guided.note_label.text() == "碰到时按住空格"
    assert "上一段录完了" in guided.result_label.text() and guided.redo_button.isEnabled()
    # Reading the next prompt takes as long as it takes; none of it is written.
    run(guided, 4.0)
    assert guided.count == written

    tap(guided)  # This press starts the clip; it is not a mark.
    second_start = guided.test_now - began
    assert guided.clip_running and guided.key_events == [] and not guided.key_down
    run(guided, 1.0)
    space(guided, True)
    space(guided, True, repeat=True)  # Auto-repeat while the key is held changes nothing.
    assert "按住中" in guided.key_label.text()
    run(guided, 0.5)
    space(guided, False, repeat=True)
    space(guided, False)
    assert "未按" in guided.key_label.text()
    run(guided, 1.6)
    assert not guided.recorder_active and guided.stop_reason == "protocol_complete"
    assert "已按提示录完" in guided.status_label.text()

    frames = read_contact_recording(guided.recording_path)[1]
    spec, rows = frame_labels(guided)
    assert spec["reviewed"] is False and spec["label_source"] == "protocol_draft"
    assert spec["protocol"] == "pinchpilot-guided-session-2"
    assert len(rows) == len(frames) and 149 <= len(frames) <= 152
    times = [frame.timestamp - began for frame in frames]
    # Frames exist only inside the two clips.
    assert all(
        first_start <= moment <= first_start + 2.05 or second_start <= moment <= second_start + 3.05
        for moment in times
    )
    touching = [moment - second_start for moment, row in zip(times, rows) if row == (1, 1, 0)]
    assert len(touching) >= 8 and 1.07 <= min(touching) and max(touching) < 1.47
    held = [moment for moment, row in zip(times, rows) if row == (1, 0, 0)]
    assert any(moment < first_start + 2 for moment in held)
    assert any(moment > second_start + 1.6 for moment in held)
    assert all(
        row == (None, None, None) for moment, row in zip(times, rows) if moment < first_start + 0.8
    )
    assert all(row[0] in (1, None) and row[2] in (0, None) for row in rows)

    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert record["completed"] and record["protocol"] == "pinchpilot-guided-session-2"
    assert (record["clips_planned"], record["clips_recorded"], record["clips_redone"]) == (2, 2, 0)
    steps = record["steps"]
    assert [(step["name"], step["discarded"]) for step in steps] == [
        ("hold", False),
        ("tap", False),
    ]
    assert steps[0]["start"] == pytest.approx(first_start)
    assert steps[0]["end"] == pytest.approx(first_start + 2.0)
    assert steps[1]["start"] == pytest.approx(second_start) and second_start > first_start + 6
    assert [pressed for _, pressed in record["space_key"]] == [True, False]
    assert record["space_key"][0][0] == pytest.approx(second_start + 1.0)
    assert record["space_key"][1][0] == pytest.approx(second_start + 1.5)
    assert record["steps_without_space"] == []
    summary = json.loads(guided.summary_path.read_text(encoding="utf-8"))
    assert summary["guided"] and not summary["partial"] and summary["camera_gaps"] == 0
    assert guided.bundle_path.name == f"{guided.recording_path.stem}_P02_2.zip"
    with zipfile.ZipFile(guided.bundle_path) as archive:
        assert sorted(archive.namelist()) == sorted(
            path.name
            for path in (
                guided.recording_path,
                guided.annotation_path,
                guided.protocol_path,
                guided.summary_path,
            )
        )
    assert "上传这一个文件" in guided.output_label.text()
    # A second recording in the same window does not reuse the session name.
    assert guided.session.text() not in ("", "2") and guided.session.isEnabled()
    assert guided.prompt_label.text() == "这一次录完了"
    assert guided.bundle_path.name in guided.note_label.text()


def test_enter_and_the_button_also_start_a_clip_and_a_dead_camera_blocks_it(guided):
    start(guided)
    guided.test_now += 2.0  # No frame for two seconds: the picture is stale.
    guided._refresh()
    assert guided.recorder_active and not guided.begin_button.isEnabled()
    assert "相机画面中断" in guided.result_label.text()
    tap(guided)
    assert not guided.clip_running and guided.schedule == []
    guided.test_now += 30.0  # A stalled camera between clips never ends the session.
    guided._refresh()
    assert guided.recorder_active
    feed(guided)
    assert guided.begin_button.isEnabled()
    tap(guided, Qt.Key.Key_Return)
    assert guided.clip_running
    run(guided, 2.1)
    run(guided, 0.2)
    guided.begin_button.click()
    assert guided.clip_running and guided.clip_index == 1


def test_a_clip_can_be_redone_and_its_first_take_keeps_no_labels(guided):
    start(guided)
    began = guided.test_now
    tap(guided, Qt.Key.Key_Backspace)  # Nothing to redo yet.
    assert guided.clip_index == 0 and guided.redone == 0
    tap(guided)
    tap(guided, Qt.Key.Key_Backspace)  # Not while a clip is running either.
    assert guided.clip_running and guided.redone == 0
    run(guided, 2.1)
    first_take = guided.count
    tap(guided, Qt.Key.Key_Backspace)
    assert guided.clip_index == 0 and guided.redone == 1
    assert guided.prompt_label.text() == "捏住：拇指和中指"
    assert "上一段已作废" in guided.result_label.text() and not guided.redo_button.isEnabled()
    tap(guided, Qt.Key.Key_Backspace)  # The take before that one is out of reach.
    assert guided.redone == 1
    begin(guided)
    run(guided, 2.1)
    guided.redo_button.click()
    assert guided.clip_index == 0 and guided.redone == 2
    begin(guided)
    run(guided, 2.1)
    begin(guided)
    run(guided, 1.0)
    space(guided, True)
    run(guided, 0.5)
    space(guided, False)
    run(guided, 1.7)
    assert guided.stop_reason == "protocol_complete"

    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert (record["clips_recorded"], record["clips_redone"]) == (2, 2)
    assert [(step["name"], step["discarded"]) for step in record["steps"]] == [
        ("hold", True),
        ("hold", True),
        ("hold", False),
        ("tap", False),
    ]
    assert [step["labels"] for step in record["steps"]] == [
        [None, None, None],
        [None, None, None],
        [1, 0, 0],
        [1, "key", 0],
    ]
    frames = read_contact_recording(guided.recording_path)[1]
    rows = frame_labels(guided)[1]
    kept_from = record["steps"][2]["start"]
    discarded = [row for frame, row in zip(frames, rows) if frame.timestamp - began < kept_from]
    assert len(discarded) >= 2 * first_take - 2 and set(discarded) == {(None, None, None)}
    assert (1, 0, 0) in rows and (1, 1, 0) in rows


def test_space_at_the_end_of_a_clip_cannot_start_the_next_one(guided):
    guided.protocol = [
        Step("tap", "食指点拇指", 3.0, (0, KEY, 0), round=1),
        Step("hold", "捏住", 5.0, (1, 0, 0), round=1),
    ]
    start(guided)
    began = guided.test_now
    tap(guided)
    clip_start = guided.test_now - began
    run(guided, 2.5)
    space(guided, True)  # Still holding when the clip stops by itself.
    run(guided, 0.6)
    assert not guided.clip_running and guided.clip_index == 1 and not guided.key_down
    assert [pressed for _, pressed in guided.key_events] == [True, False]
    assert guided.key_events[1][0] - began == pytest.approx(clip_start + 3.0)
    assert "上一段收到 1 次空格" in guided.result_label.text()
    space(guided, True, repeat=True)
    assert not guided.clip_running
    run(guided, guided.START_LOCK + 0.1)
    space(guided, True, repeat=True)  # The key has not been let go yet.
    assert not guided.clip_running
    space(guided, False)
    assert not guided.clip_running  # A release starts nothing.
    tap(guided)
    assert guided.clip_running and len(guided.key_events) == 2
    run(guided, 5.1)
    # Someone still tapping right after a clip ends: presses inside the lock are ignored.
    assert guided.stop_reason == "protocol_complete"


def test_taps_right_after_a_clip_are_ignored_until_the_lock_has_passed(guided):
    start(guided)
    tap(guided)
    run(guided, 2.05)
    assert not guided.clip_running and guided.clip_index == 1
    tap(guided)
    run(guided, guided.START_LOCK - 0.3)
    tap(guided)
    assert not guided.clip_running and guided.key_events == []
    run(guided, 0.4)
    tap(guided)
    assert guided.clip_running


def test_a_clip_without_space_marks_says_so_and_leaves_the_channel_unlabelled(guided):
    guided.protocol = [
        Step("tap", "食指点拇指", 3.0, (1, KEY, 0), round=1),
        Step("hold", "捏住", 5.0, (1, 0, 0), round=1),
    ]
    start(guided)
    tap(guided)
    run(guided, 3.1)
    assert "上一段没有收到空格。请按退格键重录" in guided.result_label.text()
    begin(guided)
    run(guided, 5.1)
    assert guided.stop_reason == "protocol_complete"
    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert record["steps_without_space"] == [0]
    rows = frame_labels(guided)[1]
    assert (1, 0, 0) in rows and (1, None, 0) in rows
    assert all(row[1] != 1 for row in rows)
    assert "有 1 个步骤没有收到空格" in guided.note_label.text()
    assert guided.bundle_path.name in guided.note_label.text()


def test_guided_stop_before_the_end_is_partial_and_releases_a_held_key(guided):
    start(guided)
    tap(guided)
    run(guided, 2.1)
    begin(guided)
    run(guided, 0.5)
    space(guided, True)
    feed(guided)
    guided.stop_recording()
    assert not guided.recorder_active and not guided.key_down and not guided.clip_running
    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert not record["completed"] and record["clips_recorded"] == 1
    assert [pressed for _, pressed in record["space_key"]] == [True, False]
    summary = json.loads(guided.summary_path.read_text(encoding="utf-8"))
    assert summary["partial"] and summary["stop_reason"] == "user_stop"
    assert guided.bundle_path.exists() and guided.annotation_path.exists()
    assert "不完整" in guided.prompt_label.text() and "重新录" in guided.note_label.text()
    tap(guided)  # After the session the key neither starts nor marks anything.
    assert len(guided.key_events) == 2 and not guided.recorder_active


def test_a_running_clip_rides_out_a_short_camera_stall_but_ends_on_a_long_one(guided):
    guided.protocol = [Step("hold", "捏住", 60.0, (1, 0, 0), round=1)]
    start(guided)
    tap(guided)
    first = [feed(guided) for _ in range(3)]
    guided.invalidate(guided.test_now + 0.3, "camera")
    guided.test_now += 1.0
    guided._refresh()
    assert guided.recorder_active and "中断" in guided.status_label.text()
    after = feed(guided)
    assert guided.recorder_active and guided.gaps == 1
    guided.test_now += guided.GUIDED_STALL + 0.1
    guided._refresh()
    assert not guided.recorder_active and guided.stop_reason == "stale_input"
    assert read_contact_recording(guided.recording_path)[1] == [*first, after]
    summary = json.loads(guided.summary_path.read_text(encoding="utf-8"))
    assert summary["partial"] and summary["camera_gaps"] == 1


def test_guided_session_still_ends_when_the_camera_source_changes(guided):
    start(guided)
    tap(guided)
    frame = feed(guided)
    feed(guided, source="synthetic_demo")
    assert not guided.recorder_active and guided.stop_reason == "source_changed"
    assert read_contact_recording(guided.recording_path)[1] == [frame]


def test_draft_failure_keeps_the_recording_and_the_record_of_the_session(guided, monkeypatch):
    start(guided)
    tap(guided)
    frame = feed(guided)

    def fail(*_):
        raise ValueError("test draft unavailable")

    monkeypatch.setattr("pinchpilot.contact_capture.draft_annotation", fail)
    guided.stop_recording()
    assert read_contact_recording(guided.recording_path)[1] == [frame]
    assert guided.annotation_path is None and guided.protocol_path.exists()
    assert "未能生成待标注模板" in guided.status_label.text()
    with zipfile.ZipFile(guided.bundle_path) as archive:
        assert guided.recording_path.name in archive.namelist()


# ---- How the hand is turned: measured from the world landmarks, checked against the round.


def turned(window, degrees, seconds=1 / 30):
    """Feed frames of a hand whose palm makes this angle with the camera."""
    angle = math.radians(degrees)
    world = [(0.0, 0.0, 0.0)] * 21
    for joint, (x, y) in ((5, (-0.03, -0.09)), (17, (0.03, -0.08))):
        world[joint] = (x * math.cos(angle), y, -x * math.sin(angle))
    frames = []
    for _ in range(max(1, round(seconds * 30))):
        window.test_now += 1 / 30
        frame = replace(synthetic_tripod(window.test_now), world_landmarks=tuple(world))
        window.feed(frame, window.test_now)
        frames.append(frame)
    return frames


def test_palm_angle_reads_how_far_the_hand_is_turned(guided):
    assert palm_angle(synthetic_tripod(1.0)) is None  # No world landmarks, nothing to say.
    for degrees in (0, 25, 60, 90):
        frame = turned(guided, degrees)[0]
        assert palm_angle(frame) == pytest.approx(degrees, abs=0.01)
    flipped = replace(
        frame, world_landmarks=tuple((-x, y, -z) for x, y, z in frame.world_landmarks)
    )
    assert palm_angle(flipped) == pytest.approx(90, abs=0.01)
    back = turned(guided, 20)[0]
    mirrored = tuple((x, y, -z) for x, y, z in back.world_landmarks)
    assert palm_angle(replace(back, world_landmarks=mirrored)) == pytest.approx(20, abs=0.01)


def test_a_round_that_asks_for_a_turned_hand_checks_it_and_records_what_was_done(guided):
    guided.protocol = [
        Step("pose", "这一轮的手部朝向：手侧过来", 8.0, (None, None, None), round=1, pose=(55, 90)),
        Step("open", "手张开", 5.0, (0, 0, 0), round=1),
        Step("grip_move", "捏住", 5.0, (1, 0, 0), round=1),
    ]
    start(guided)
    assert [clip.pose for clip in guided.clips] == [(55, 90), (55, 90)]
    assert guided.notice_label.text() == "这一轮的手部朝向：手侧过来"
    assert "看不到手" not in guided.pose_label.text() or guided.palm_now is None
    turned(guided, 22, 0.5)
    assert guided.palm_now == pytest.approx(22, abs=0.5)
    assert "现在约 22°，这一轮需要 55° 以上 ✗" in guided.pose_label.text()
    assert guided.demo.view[0] > 70  # The example hand is drawn side-on as well.

    tap(guided)  # Held back once: the hand is still facing the camera.
    assert not guided.clip_running and guided.schedule == []
    assert "所以还没开始" in guided.result_label.text()
    turned(guided, 22, 0.2)
    tap(guided)  # A second start goes ahead; the angle is only an estimate.
    assert guided.clip_running
    turned(guided, 22, 5.1)
    assert not guided.clip_running and guided.clip_index == 1
    assert "夹角约 22°，不符合这一轮的要求" in guided.result_label.text()

    tap(guided, Qt.Key.Key_Backspace)
    turned(guided, 70, guided.START_LOCK + 0.2)
    assert "现在约 70°，这一轮需要 55° 以上 ✓" in guided.pose_label.text()
    tap(guided)  # Turned as asked: starts at once.
    assert guided.clip_running
    turned(guided, 70, 5.1)
    assert "不符合" not in guided.result_label.text()
    turned(guided, 70, guided.START_LOCK + 0.2)
    tap(guided)
    turned(guided, 70, 5.1)
    assert guided.stop_reason == "protocol_complete"

    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    steps = record["steps"]
    assert [(step["name"], step["discarded"], step["pose"]) for step in steps] == [
        ("open", True, [55, 90]),
        ("open", False, [55, 90]),
        ("grip_move", False, [55, 90]),
    ]
    assert [round(step["palm_angle"]) for step in steps] == [22, 70, 70]


def test_rounds_without_a_pose_and_hands_without_an_angle_are_never_held_back(guided):
    start(guided)  # SHORT asks for no pose, and these frames carry no world landmarks.
    assert guided.clips[0].pose is None and guided.palm_now is None
    assert "看不到手" in guided.pose_label.text() and "这一轮不限" in guided.pose_label.text()
    tap(guided)
    assert guided.clip_running
    run(guided, 2.1)
    guided.stop_recording()
    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert record["steps"][0]["palm_angle"] is None and record["steps"][0]["pose"] is None
