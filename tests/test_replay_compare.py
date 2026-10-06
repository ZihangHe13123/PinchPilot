"""Offline replay evidence checks; only temporary output and generated landmarks."""

import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace

import pytest

from pinchpilot import replay_compare
from pinchpilot.domain import HandFrame
from pinchpilot.replay_compare import ReplayCase, run_comparison, synthetic_cases
from pinchpilot.storage import SCHEMA, Recorder, read_recording
from pinchpilot.tripod import TripodConfig, TripodEngine
from pinchpilot.tripod_demo import synthetic_tripod


def test_fixed_seed_is_deterministic_and_every_profile_gets_identical_frames(tmp_path):
    first = run_comparison(None, tmp_path / "first.json", seed=935)
    second = run_comparison(None, tmp_path / "second.json", seed=935)
    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8")) == first
    assert first["source"] == "synthetic" and first["seed"] == 935
    assert not first["camera_opened"] and not first["os_events_sent"] and not first["models_loaded"]
    assert set(first["cases"]) == {
        "stationary_jitter",
        "translation",
        "edge_reversal",
        "click_drag",
        "tracking_loss",
    }
    for case in first["cases"].values():
        for profile in case["profiles"].values():
            assert profile["input_sha256"] == case["input_sha256"]
            assert profile["metrics"]["frames"] == case["input_frames"]
            assert "trace" not in profile
    changed = run_comparison(None, tmp_path / "changed.json", seed=936)
    assert (
        first["cases"]["stationary_jitter"]["input_sha256"]
        != changed["cases"]["stationary_jitter"]["input_sha256"]
    )
    assert (
        first["cases"]["translation"]["input_sha256"]
        == changed["cases"]["translation"]["input_sha256"]
    )


def test_actual_recording_format_and_trace_match_direct_engine_replay(tmp_path):
    recorder = Recorder(tmp_path / "recordings", "synthetic-participant", "fixture", "other")
    original = synthetic_cases()[3].frames  # Includes intentional click/drag freezes.
    for frame in original:
        recorder.add(frame)
    recorder.close()
    _, frames = read_recording(recorder.path)
    original_bytes = recorder.path.read_bytes()
    report = run_comparison(recorder.path, tmp_path / "recorded.json", include_trace=True)
    assert recorder.path.read_bytes() == original_bytes
    assert report["source"] == "recorded" and report["seed"] is None
    assert report["input"]["human_selected_gesture_label"] == "other"
    assert not report["input"]["label_used_as_intent_or_stationarity"]
    case = report["cases"]["recorded_sequence"]
    assert case["input_frames"] == len(frames)
    for name, profile in case["profiles"].items():
        engine = TripodEngine(TripodConfig(**report["configurations"][name]))
        engine.pointer = (0.5, 0.5)
        engine.reset()
        counts = Counter()
        for frame, row in zip(frames, profile["trace"], strict=True):
            result = engine.process(frame)
            assert row["timestamp_s"] == frame.timestamp
            assert row["state"] == result.state
            assert row["pointer_normalized_xy"] == pytest.approx(result.pointer)
            assert row["events"] == [asdict(event) for event in result.events]
            counts.update(event.kind for event in result.events)
        assert profile["metrics"]["generated_event_counts"] == dict(counts)
        assert profile["metrics"]["stationary_windows"] == []
        assert profile["metrics"]["generated_event_counts"]["down"] == 2
        assert profile["metrics"]["generated_event_counts"]["up"] == 2
        assert profile["metrics"]["drag_entries"] == 1


def test_loss_metrics_distinguish_input_interruptions_and_button_cancellations(tmp_path):
    report = run_comparison(None, tmp_path / "report.json", include_trace=True)
    case = report["cases"]["tracking_loss"]
    for profile in case["profiles"].values():
        metrics = profile["metrics"]
        interruptions = metrics["interruptions"]
        assert interruptions["input_missing_landmark_runs"] == 1
        assert interruptions["input_timestamp_gaps_at_least_timeout"] == 1
        assert interruptions["held_button_cancellations"] == 2
        assert (
            metrics["generated_event_counts"]["down"]
            == metrics["generated_event_counts"]["up"]
            == 2
        )
        assert not metrics["left_button_held_at_input_end"]
        assert sum(metrics["state_frame_counts"].values()) == metrics["frames"]
        assert profile["trace_sha256"] == replay_compare._digest(profile["trace"])


def test_intentional_freezing_is_excluded_from_shared_stationary_jitter(tmp_path, monkeypatch):
    frames = tuple(synthetic_tripod(i / 30, grip=i < 10) for i in range(30))
    case = ReplayCase(
        "intentional_freeze", "Deliberately release middle finger.", frames, (("hold", 10, 30),)
    )
    monkeypatch.setattr(replay_compare, "synthetic_cases", lambda seed: (case,))
    report = run_comparison(None, tmp_path / "frozen.json")
    for profile in report["cases"][case.name]["profiles"].values():
        window = profile["metrics"]["stationary_windows"][0]
        assert window["requested_frames"] == 20
        assert window["common_eligible_frames"] == 0
        assert window["output_jitter"] is None and window["raw_signal_jitter"] is None
        assert profile["metrics"]["interruptions"]["held_button_cancellations"] == 0


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [HandFrame(float("nan"))],
        [HandFrame(1), HandFrame(1)],
        [HandFrame(1, aspect=float("inf"))],
        [HandFrame(1, aspect=-1)],
        [HandFrame(1, handedness_score=2)],
        [HandFrame(1, landmarks=((0.1, 0.1, 0),))],
        [replace(synthetic_tripod(1), landmarks=((float("nan"), 0, 0),) * 21)],
    ],
)
def test_invalid_recordings_fail_without_overwriting_existing_report(tmp_path, frames):
    recording = tmp_path / "bad.jsonl"
    header = {
        "schema": SCHEMA,
        "label": "other",
        "participant": "fixture",
        "session": "test",
        "episode": "1",
    }
    rows = [header] + [{"type": "frame", **asdict(frame)} for frame in frames]
    recording.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "prior.json"
    output.write_text("prior result", encoding="utf-8")
    before = recording.read_bytes()
    with pytest.raises(ValueError):
        run_comparison(recording, output)
    assert output.read_text(encoding="utf-8") == "prior result"
    assert recording.read_bytes() == before


def test_report_cannot_overwrite_source_and_bad_seed_cannot_create_output(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("recording must remain intact", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        run_comparison(source, source)
    assert source.read_text(encoding="utf-8") == "recording must remain intact"
    output = tmp_path / "not-created.json"
    for seed in (-1, True, 1.2, 2**64):
        with pytest.raises(ValueError, match="seed"):
            run_comparison(None, output, seed=seed)
    assert not output.exists()


def test_hardware_and_model_loading_are_not_imported_or_used(tmp_path):
    # Check in a clean interpreter; parent test suites may already have imported Qt/OpenCV.
    code = """
import builtins
import pickle
import sys
from pathlib import Path
import numpy

original_import = builtins.__import__
forbidden = {'cv2', 'mediapipe', 'PySide6', 'Quartz', 'AppKit', 'ApplicationServices', 'joblib'}
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden or name in {'pinchpilot.platform_io', 'pinchpilot.vision'}:
        raise AssertionError('Hardware/model dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
def forbidden_pickle(*args, **kwargs):
    raise AssertionError('A serialized Python model was loaded')
builtins.__import__ = guarded_import
pickle.load = pickle.loads = forbidden_pickle
from pinchpilot.replay_compare import run_comparison
report = run_comparison(None, Path(sys.argv[1]))
assert not report['camera_opened'] and not report['os_events_sent']
assert not forbidden.intersection(sys.modules)
assert 'pinchpilot.platform_io' not in sys.modules
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "offline.json")],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert (tmp_path / "offline.json").is_file()
