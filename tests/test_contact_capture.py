"""Offscreen engineering fixtures only; the camera source is simulated, not human data."""

import json
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
from pinchpilot.contact_data import read_contact_recording
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


# ---- Guided sessions: on-screen steps, the Space key and draft labels.

SHORT = [
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


def space(window, pressed, repeat=False):
    kind = QEvent.Type.KeyPress if pressed else QEvent.Type.KeyRelease
    event = QKeyEvent(kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", repeat)
    (window.keyPressEvent if pressed else window.keyReleaseEvent)(event)


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


def test_guided_session_shows_each_step_and_saves_a_draft_from_prompts_and_space(guided):
    guided.participant.setText("P02")
    guided.session.setText("2")
    start(guided)
    began = guided.test_now
    assert guided.guided_active and not guided.guided_box.isEnabled()
    assert guided.focusWidget() is guided  # Space cannot reach the consent box or a button.
    assert (
        guided.prompt_label.text() == "捏住：拇指和中指"
        and "第 1/1 轮" in guided.guide_status.text()
    )
    for _ in range(90):
        feed(guided)
    assert guided.prompt_label.text() == "保持捏住。食指点拇指"
    assert guided.note_label.text() == "碰到时按住空格" and "结束" in guided.next_label.text()
    space(guided, True)
    space(guided, True, repeat=True)  # Auto-repeat while the key is held changes nothing.
    assert "按住中" in guided.key_label.text()
    for _ in range(15):
        feed(guided)
    space(guided, False, repeat=True)
    space(guided, False)
    assert "未按" in guided.key_label.text()
    for _ in range(100):
        if not guided.recorder_active:
            break
        feed(guided)
    assert not guided.recorder_active and guided.stop_reason == "protocol_complete"
    assert "已按提示录完" in guided.status_label.text()

    frames = read_contact_recording(guided.recording_path)[1]
    spec, rows = frame_labels(guided)
    assert spec["reviewed"] is False and spec["label_source"] == "protocol_draft"
    assert len(rows) == len(frames)
    times = [frame.timestamp - began for frame in frames]
    touching = [moment for moment, row in zip(times, rows) if row == (1, 1, 0)]
    assert len(touching) >= 8 and 3.07 <= min(touching) and max(touching) < 3.43
    held = [moment for moment, row in zip(times, rows) if row == (1, 0, 0)]
    assert any(moment < 2 for moment in held) and any(moment > 3.6 for moment in held)
    assert all(row == (None, None, None) for moment, row in zip(times, rows) if moment < 0.8)
    assert all(row[0] in (1, None) and row[2] in (0, None) for row in rows)

    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert record["completed"] and [step["name"] for step in record["steps"]] == ["hold", "tap"]
    assert [pressed for _, pressed in record["space_key"]] == [True, False]
    assert record["steps_without_space"] == []
    assert record["space_key"][0][0] == pytest.approx(3.0)
    assert record["space_key"][1][0] == pytest.approx(3.5)
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


def test_guided_session_without_any_space_mark_says_so_and_leaves_the_channel_unlabelled(guided):
    start(guided)
    for _ in range(200):
        if not guided.recorder_active:
            break
        feed(guided)
    assert guided.stop_reason == "protocol_complete"
    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert record["steps_without_space"] == [1]
    rows = frame_labels(guided)[1]
    assert (1, 0, 0) in rows and (1, None, 0) in rows
    assert all(row[1] != 1 for row in rows)
    assert "有 1 个步骤没有收到空格" in guided.note_label.text()
    assert guided.bundle_path.name in guided.note_label.text()


def test_guided_stop_before_the_end_is_partial_and_releases_a_held_key(guided):
    start(guided)
    for _ in range(75):
        feed(guided)
    space(guided, True)
    feed(guided)
    guided.stop_recording()
    assert not guided.recorder_active and not guided.key_down
    record = json.loads(guided.protocol_path.read_text(encoding="utf-8"))
    assert not record["completed"]
    assert [pressed for _, pressed in record["space_key"]] == [True, False]
    summary = json.loads(guided.summary_path.read_text(encoding="utf-8"))
    assert summary["partial"] and summary["stop_reason"] == "user_stop"
    assert guided.bundle_path.exists() and guided.annotation_path.exists()
    assert "不完整" in guided.prompt_label.text() and "重新录" in guided.note_label.text()
    space(guided, True)  # After the session the key is no longer recorded.
    assert len(guided.key_events) == 2


def test_guided_session_rides_out_a_short_camera_stall_but_ends_on_a_long_one(guided):
    guided.protocol = [Step("hold", "捏住", 60.0, (1, 0, 0), round=1)]
    start(guided)
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
    frame = feed(guided)
    feed(guided, source="synthetic_demo")
    assert not guided.recorder_active and guided.stop_reason == "source_changed"
    assert read_contact_recording(guided.recording_path)[1] == [frame]


def test_draft_failure_keeps_the_recording_and_the_record_of_the_session(guided, monkeypatch):
    start(guided)
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
