import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from pinchpilot.contact_data import (
    CHANNELS,
    ContactRecorder,
    WindowBuilder,
    WindowConfig,
    annotation_template,
    contact_features,
    create_synthetic_contact_dataset,
    load_contact_dataset,
    read_contact_recording,
)
from pinchpilot.domain import HandFrame
from pinchpilot.storage import Recorder
from pinchpilot.tripod_demo import synthetic_tripod


def update(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_continuous_recording_keeps_missing_frames_and_templates_unknown(tmp_path):
    path = tmp_path / "recording.jsonl"
    writer = ContactRecorder(path, "P1", "S1")
    writer.add(synthetic_tripod(1))
    writer.add(HandFrame(1.1))
    with pytest.raises(ValueError, match="递增"):
        writer.add(synthetic_tripod(1))
    writer.close()
    metadata, frames = read_contact_recording(path)
    assert metadata["source"] == "camera" and not metadata["labels_inferred"]
    assert len(frames) == 2 and not frames[1].landmarks
    template = annotation_template(path, tmp_path / "labels.json")
    assert not template["reviewed"]
    assert all(template["intervals"][0][key] is None for key in CHANNELS)
    with pytest.raises(ValueError, match="不能覆盖"):
        annotation_template(path, tmp_path / "labels.json")


def test_features_invariant_to_rigid_translation_scale_and_aspect():
    frame = synthetic_tripod(1)
    points = np.array(frame.landmarks)
    moved = points * 0.8 + [0.1, 0.05, -0.2]
    actual = contact_features(replace(frame, landmarks=tuple(map(tuple, moved))), 0.033)
    expected = contact_features(frame, 0.033)
    assert actual.shape == (67,)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    changed = points.copy()
    aspect = 16 / 9
    changed[:, [0, 2]] *= frame.aspect / aspect
    resized = replace(frame, aspect=aspect, landmarks=tuple(map(tuple, changed)))
    np.testing.assert_allclose(contact_features(resized, 0.033), expected, atol=1e-6)


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("missing", "missing_hand"),
        ("gap", "time_gap"),
        ("duplicate", "nonmonotonic_time"),
        ("other_hand", "hand_changed"),
        ("invalid", "invalid_frame"),
    ],
)
def test_windows_never_bridge_missing_gap_reordered_or_different_hands(failure, reason):
    builder = WindowBuilder()
    for i in range(8):
        ready = builder.push(synthetic_tripod(1 + i / 30))
    assert ready.shape == (8, 67)
    t = 1 + 8 / 30
    next_frame = {
        "missing": HandFrame(t),
        "gap": synthetic_tripod(t + 0.2),
        "duplicate": synthetic_tripod(1 + 7 / 30),
        "other_hand": replace(synthetic_tripod(t), handedness="Left"),
        "invalid": HandFrame(float("nan")),
    }[failure]
    assert builder.push(next_frame) is None
    assert builder.reason == reason
    assert len(builder.history) <= 1


def test_window_snapshot_and_future_changes_do_not_rewrite_history():
    one, two = WindowBuilder(), WindowBuilder()
    for i in range(8):
        frame = synthetic_tripod(1 + i / 30)
        a, b = one.push(frame), two.push(frame)
    preserved = a.copy()
    one.push(synthetic_tripod(1 + 8 / 30, contact=0.1))
    two.push(synthetic_tripod(1 + 8 / 30, contact=0.85))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, preserved)
    assert np.isclose(a[-1, -1], 1 / 30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frames": True},
        {"frames": 1},
        {"max_gap_seconds": float("nan")},
        {"max_window_seconds": 0.01},
    ],
)
def test_window_configuration_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        WindowConfig(**kwargs)


def test_synthetic_data_split_before_windows_and_simultaneous_contacts(tmp_path):
    manifest = create_synthetic_contact_dataset(tmp_path)
    data = load_contact_dataset(manifest)
    assert data.metadata["source"] == "synthetic"
    assert [len(data.partitions[s].x) for s in ("train", "validation", "test")] == [699, 233, 233]
    assert np.any(np.all(data.partitions["test"].y == [1, 1, 1], axis=1))
    assert set(data.partitions["test"].groups).isdisjoint(data.partitions["train"].groups)
    assert data.window_config == WindowConfig()
    assert data.metadata["window_config"] == asdict(WindowConfig())
    for partition in data.partitions.values():
        assert partition.frame_indices.min() >= 7
    with pytest.raises(ValueError, match="空目录"):
        create_synthetic_contact_dataset(tmp_path)


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda d: d.update(reviewed=False), "复核"),
        (lambda d: d.update(recording_sha256="bad"), "SHA256"),
        (lambda d: d.update(label_source="human_reviewed"), "合成标签"),
        (lambda d: d["intervals"][0].update(index=True), "0、1"),
        (lambda d: d["intervals"][1].update(start=0), "重叠"),
    ],
)
def test_bad_or_unreviewed_labels_rejected(tmp_path, change, expected):
    manifest = create_synthetic_contact_dataset(tmp_path)
    labels = tmp_path / "synthetic_P1.labels.json"
    update(labels, change)
    with pytest.raises(ValueError, match=expected):
        load_contact_dataset(manifest)


def test_unknown_is_not_a_negative_label(tmp_path):
    manifest = create_synthetic_contact_dataset(tmp_path)
    update(tmp_path / "synthetic_P5.labels.json", lambda d: d["intervals"][0].update(index=None))
    data = load_contact_dataset(manifest)
    assert len(data.partitions["test"].x) == 210
    assert data.metadata["skipped"]["unknown_label"] == 23


def test_same_group_across_partitions_is_rejected_before_windowing(tmp_path):
    manifest = create_synthetic_contact_dataset(tmp_path)
    update(manifest, lambda d: d["recordings"].append({**d["recordings"][0], "split": "test"}))
    with pytest.raises(ValueError, match="分组泄漏"):
        load_contact_dataset(manifest)


def test_duplicate_episode_rejected_even_inside_training(tmp_path):
    manifest = create_synthetic_contact_dataset(tmp_path)
    update(manifest, lambda d: d["recordings"].append(d["recordings"][0]))
    with pytest.raises(ValueError, match="重复录制"):
        load_contact_dataset(manifest)


def test_legacy_labels_not_used_as_contact_ground_truth(tmp_path):
    recorder = Recorder(tmp_path, "P1", "S1", "pinch")
    recorder.add(synthetic_tripod(1))
    recorder.close()
    meta, frames = read_contact_recording(recorder.path)
    assert meta["source"] == "legacy_unspecified" and len(frames) == 1
    template = annotation_template(recorder.path, tmp_path / "labels.json")
    assert template["label_source"] == "unreviewed"
    assert template["intervals"][0]["index"] is None


def test_deterministic_fixture_content_and_windows(tmp_path):
    first = load_contact_dataset(create_synthetic_contact_dataset(tmp_path / "first"))
    second = load_contact_dataset(create_synthetic_contact_dataset(tmp_path / "second"))
    assert first.metadata["fingerprint"] == second.metadata["fingerprint"]
    for split in first.partitions:
        np.testing.assert_array_equal(first.partitions[split].x, second.partitions[split].x)


def test_identity_whitespace_cannot_create_a_separate_group(tmp_path):
    manifest = create_synthetic_contact_dataset(tmp_path)
    recording = tmp_path / "synthetic_P5.jsonl"
    rows = recording.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["participant"] = " synthetic_P1 "
    rows[0] = json.dumps(header)
    recording.write_text("\n".join(rows) + "\n", encoding="utf-8")
    import hashlib

    update(
        tmp_path / "synthetic_P5.labels.json",
        lambda d: d.update(recording_sha256=hashlib.sha256(recording.read_bytes()).hexdigest()),
    )
    with pytest.raises(ValueError, match="分组泄漏"):
        load_contact_dataset(manifest)


def test_degenerate_geometry_has_same_training_and_shadow_quality_gate():
    from pinchpilot.contact_data import rule_contacts

    builder = WindowBuilder()
    for index in range(10):
        frame = synthetic_tripod(1 + index / 30)
        points = np.asarray(frame.landmarks)
        points[:, 2] = points[:, 1] * 2
        points[:, :2] *= 0.08
        frame = replace(frame, landmarks=tuple(map(tuple, points)))
        assert rule_contacts(frame) is None
        assert contact_features(frame) is None
        assert builder.push(frame) is None
    assert builder.reason == "invalid_frame"


def test_annotation_creation_cannot_overwrite_a_recording_named_like_a_tempfile(tmp_path):
    path = tmp_path / "annotations.json.tmp"
    writer = ContactRecorder(path, "P1", "S1")
    writer.add(synthetic_tripod(1))
    writer.close()
    original = path.read_bytes()
    annotation_template(path, tmp_path / "annotations.json")
    assert path.read_bytes() == original
