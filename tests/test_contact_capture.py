"""Offscreen engineering fixtures only; the camera source is simulated, not human data."""

import json
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QWidget

from pinchpilot.contact_capture import ContactCaptureWindow
from pinchpilot.contact_data import read_contact_recording
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
    labels = json.loads(window.annotation_path.read_text())
    assert labels["reviewed"] is False and labels["label_source"] == "unreviewed"
    assert labels["intervals"] == [
        {"start": 0, "end": 3, "middle": None, "index": None, "ring": None}
    ]
    report = json.loads(window.summary_path.read_text())
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
    summary = json.loads(window.summary_path.read_text())
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
    assert len(window.recording_path.read_text().splitlines()) == 1
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
    assert json.loads(window.summary_path.read_text())["labels_reviewed"] is False


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
    assert json.loads(summary.read_text())["stop_reason"] == "window_closed"
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    assert not parent.findChildren(ContactCaptureWindow)
    parent.close()
