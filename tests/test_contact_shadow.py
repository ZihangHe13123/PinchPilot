"""Synthetic shadow checks, with fake scores and no camera/native input."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pinchpilot import contact_shadow
from pinchpilot.contact_data import CHANNELS
from pinchpilot.contact_shadow import ContactShadow, run_shadow
from pinchpilot.domain import HandFrame
from pinchpilot.tripod_demo import synthetic_tripod


class FakePredictor:
    def __init__(self, scores=(0.9, 0.5, 0.1), source="synthetic"):
        self.metadata = {
            "schema": "pinchpilot-contact-model-v1",
            "model": "forest",
            "feature_version": "contact-v1",
            "feature_count": 67,
            "channels": list(CHANNELS),
            "window_config": {"frames": 8, "max_gap_seconds": 0.12, "max_window_seconds": 0.8},
            "source": source,
            "training_source": source,
            "evidence": "synthetic_engineering"
            if source == "synthetic"
            else "held_out_camera_windows",
        }
        self.scores = scores
        self.windows = []

    def predict_windows(self, windows):
        self.windows.append(windows.copy())
        return np.tile(self.scores, (len(windows), 1))


def frames(count=8, start=10.0):
    return [synthetic_tripod(start + i / 30) for i in range(count)]


def test_warmup_and_three_independent_scores_leave_frame_unchanged():
    predictor = FakePredictor()
    times = iter([1.0, 1.002])
    shadow = ContactShadow(predictor, clock=lambda: next(times))
    sequence = frames()
    original = deepcopy(sequence)
    rows = [shadow.process(frame) for frame in sequence]
    assert all(row["status"] == "warming" for row in rows[:7])
    assert all(all(value is None for value in row["scores"].values()) for row in rows[:7])
    assert len(predictor.windows) == 1
    assert predictor.windows[0].shape == (1, 8, 67)
    final = rows[-1]
    assert final["status"] == "ready"
    assert final["scores"] == dict(zip(CHANNELS, (0.9, 0.5, 0.1)))
    assert final["suggested_contact"] == dict(zip(CHANNELS, (True, True, False)))
    assert final["inference_ms"] == pytest.approx(2.0)
    assert final["rule_geometry"]["below_engage_or_touch_threshold"] == dict(
        zip(CHANNELS, (True, False, False))
    )
    assert sequence == original


@pytest.mark.parametrize(
    "break_frame,reason,status",
    [
        (HandFrame(10.27), "missing_hand", "unavailable"),
        (
            replace(synthetic_tripod(10.27), landmarks=((float("nan"), 0, 0),) * 21),
            "invalid_frame",
            "unavailable",
        ),
        (synthetic_tripod(10.2), "nonmonotonic_time", "unavailable"),
        (synthetic_tripod(10.6), "time_gap", "warming"),
        (replace(synthetic_tripod(10.27), handedness="Left"), "hand_changed", "warming"),
    ],
)
def test_discontinuities_clear_predictions_and_require_new_history(break_frame, reason, status):
    predictor = FakePredictor()
    shadow = ContactShadow(predictor)
    for frame in frames():
        shadow.push(frame)
    interrupted = shadow.push(break_frame)
    assert interrupted["status"] == status
    assert interrupted["reason"] == reason
    assert all(value is None for value in interrupted["scores"].values())
    assert all(value is None for value in interrupted["suggested_contact"].values())
    assert len(predictor.windows) == 1
    restart = max(break_frame.timestamp, 10.27) + 1 / 30
    resumed = frames(8, restart)
    if break_frame.handedness == "Left":
        resumed = [replace(frame, handedness="Left") for frame in resumed]
    rows = [shadow.push(frame) for frame in resumed]
    assert all(row["status"] == "warming" for row in rows[:6])
    assert rows[-1]["status"] == "ready"


def test_explicit_source_reset_and_future_frames_cannot_change_past_scores():
    first, second = ContactShadow(FakePredictor()), ContactShadow(FakePredictor())
    first_rows = [first.process(frame) for frame in frames(10)]
    second_rows = [second.process(frame) for frame in frames(10)]
    assert [row["scores"] for row in first_rows] == [row["scores"] for row in second_rows]
    saved = deepcopy(first_rows)
    first.process(synthetic_tripod(10.4, grip=False, contact=0.1))
    assert first_rows == saved
    first.reset("source_changed")
    assert first.last_reset_reason == "source_changed"
    rows = [first.process(frame) for frame in frames(8, 20)]
    assert all(row["status"] == "warming" for row in rows[:7])
    assert rows[-1]["status"] == "ready"


@pytest.mark.parametrize(
    "scores",
    [np.array([[float("nan"), 0.2, 0.3]]), np.zeros((1, 2)), np.array([[1.1, 0, 0]])],
)
def test_invalid_model_output_is_unavailable_and_does_not_reuse_positive_scores(scores):
    predictor = FakePredictor()
    shadow = ContactShadow(predictor)
    for frame in frames():
        assert shadow.process(frame)["status"] in ("warming", "ready")
    predictor.predict_windows = lambda _: scores
    result = shadow.process(synthetic_tripod(10.27))
    assert result["status"] == "unavailable"
    assert result["reason"] == "prediction_failed"
    assert all(value is None for value in result["scores"].values())
    assert shadow.process(synthetic_tripod(10.3))["status"] == "warming"


@pytest.mark.parametrize(
    "update",
    [
        {"channels": ["ring", "middle", "index"]},
        {"feature_count": 66},
        {"feature_version": "future-incompatible"},
        {"window_config": {"frames": 8}},
        {"window_config": {"frames": True, "max_gap_seconds": 0.12, "max_window_seconds": 0.8}},
        {
            "window_config": {
                "frames": 8,
                "max_gap_seconds": float("nan"),
                "max_window_seconds": 0.8,
            }
        },
    ],
)
def test_incompatible_metadata_is_rejected_instead_of_defaulted(update):
    predictor = FakePredictor()
    predictor.metadata.update(update)
    with pytest.raises(ValueError):
        ContactShadow(predictor)


def report_fixture(
    tmp_path, monkeypatch, *, recording_source="synthetic", model_source="synthetic"
):
    from pinchpilot import contact_models

    recording = tmp_path / "recording.jsonl"
    recording.write_text("input fixture remains unchanged", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    metadata = {
        "schema": "pinchpilot-contact-recording-v1",
        "source": recording_source,
        "participant": "synthetic-participant",
        "session": "fixture-session",
        "episode": "fixture-episode",
        # Report must not blindly echo unrelated source metadata.
        "landmarks": "not for report",
    }
    sequence = frames(10) + [HandFrame(10.34)]
    monkeypatch.setattr(contact_shadow, "read_contact_recording", lambda _: (metadata, sequence))
    predictor = FakePredictor(source=model_source)
    monkeypatch.setattr(contact_models, "ContactPredictor", lambda _: predictor)
    return recording, model, sequence


@pytest.mark.parametrize(
    "recording_source,model_source", [("synthetic", "camera"), ("camera", "synthetic")]
)
def test_report_preserves_sources_and_does_not_emit_landmarks_or_engine_events(
    tmp_path, monkeypatch, recording_source, model_source
):
    from pinchpilot.tripod import TripodEngine

    recording, model, sequence = report_fixture(
        tmp_path, monkeypatch, recording_source=recording_source, model_source=model_source
    )

    def forbidden(*_, **__):
        raise AssertionError("Shadow must not instantiate or process a gesture engine")

    monkeypatch.setattr(TripodEngine, "__init__", forbidden)
    monkeypatch.setattr(TripodEngine, "process", forbidden)
    output = tmp_path / "nested" / "report.json"
    before = recording.read_bytes()
    report = run_shadow(recording, model, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert recording.read_bytes() == before
    assert report["source"] == recording_source
    assert report["model_training_source"] == model_source
    assert report["synthetic_evidence_only"]
    assert report["summary"]["status_counts"] == {"warming": 7, "ready": 3, "unavailable": 1}
    assert report["summary"]["timed_predictions"] == 3
    assert len(report["trace"]) == len(sequence)
    for flag in (
        "desktop_realtime_integrated",
        "camera_opened",
        "os_events_sent",
        "gesture_engine_executed",
        "stores_images",
        "stores_keypoints",
    ):
        assert report[flag] is False
    assert "landmarks" not in output.read_text(encoding="utf-8")
    assert all("events" not in row for row in report["trace"])


def test_report_limit_and_input_protection_leave_existing_files_unchanged(tmp_path, monkeypatch):
    recording, model, _ = report_fixture(tmp_path, monkeypatch)
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")
    monkeypatch.setattr(contact_shadow, "MAX_REPORT_FRAMES", 5)
    with pytest.raises(ValueError, match="frames"):
        run_shadow(recording, model, output)
    assert output.read_text(encoding="utf-8") == "previous report"
    with pytest.raises(ValueError, match="input recording"):
        run_shadow(recording, model, recording)
    metadata = model / "metadata.json"
    metadata.write_text("existing model", encoding="utf-8")
    with pytest.raises(ValueError, match="model artifact"):
        run_shadow(recording, model, metadata)
    assert metadata.read_text(encoding="utf-8") == "existing model"


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_recording_read_and_legacy_source_are_preserved(tmp_path, monkeypatch, legacy):
    from pinchpilot import contact_models
    from pinchpilot.contact_data import ContactRecorder
    from pinchpilot.storage import Recorder

    if legacy:
        recorder = Recorder(tmp_path, "synthetic-participant", "fixture", "other")
    else:
        recorder = ContactRecorder(
            tmp_path / "continuous.jsonl", "synthetic-participant", "fixture", source="synthetic"
        )
    sequence = frames(10)
    for frame in sequence:
        recorder.add(frame)
    recorder.close()
    monkeypatch.setattr(contact_models, "ContactPredictor", lambda _: FakePredictor())
    report = run_shadow(recorder.path, tmp_path / "model", tmp_path / "report.json")
    assert report["source"] == ("legacy_unspecified" if legacy else "synthetic")
    assert report["recording_source_declared"] is not legacy
    assert [row["timestamp_s"] for row in report["trace"]] == [f.timestamp for f in sequence]
    assert report["summary"]["status_counts"] == {"warming": 7, "ready": 3}


def test_atomic_temp_path_cannot_overwrite_recording(tmp_path, monkeypatch):
    recording = tmp_path / "report.json.tmp"
    recording.write_text("source must survive", encoding="utf-8")
    with pytest.raises(ValueError, match="input recording"):
        run_shadow(recording, tmp_path / "model", tmp_path / "report.json")
    assert recording.read_text(encoding="utf-8") == "source must survive"


def test_shadow_import_and_inference_do_not_load_hardware_or_native_modules():
    script = r"""
import builtins
import numpy
original_import = builtins.__import__
forbidden = ('cv2', 'mediapipe', 'PySide6', 'Quartz', 'AppKit', 'ApplicationServices',
             'pinchpilot.platform_io', 'pinchpilot.vision')
def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in forbidden):
        raise AssertionError('forbidden dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from pinchpilot.contact_shadow import ContactShadow
from pinchpilot.tripod_demo import synthetic_tripod
class Fake:
    metadata = {'channels':['middle','index','ring'], 'feature_count':67,
                'feature_version':'contact-v1',
                'window_config':{'frames':8,'max_gap_seconds':.12,'max_window_seconds':.8}}
    def predict_windows(self, x):
        return numpy.array([[.8,.8,.1]])
shadow = ContactShadow(Fake())
for i in range(10):
    result = shadow.process(synthetic_tripod(i / 30))
assert result['status'] == 'ready'
assert result['suggested_contact']['middle'] and result['suggested_contact']['index']
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
