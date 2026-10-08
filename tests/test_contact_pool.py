"""The pool of labelled recordings that cross-validation reads; synthetic frames only."""

import json
import shutil
import zipfile

import numpy as np
import pytest

from pinchpilot import contact_pool
from pinchpilot.contact_data import (
    CHANNELS,
    ContactRecorder,
    WindowBuilder,
    create_synthetic_contact_pool,
    draft_annotation,
    read_contact_recording,
)
from pinchpilot.contact_pool import load_contact_pool, summary
from pinchpilot.tripod_demo import synthetic_tripod


def guided(directory, stem, participant="P01", *, start=50.0, seed=0, bundle=True):
    """A camera-source recording with a protocol draft, as a guided session leaves it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jsonl"
    rng = np.random.default_rng(seed)
    recorder = ContactRecorder(path, participant, f"session-{stem}", source="camera")
    for index in range(60):
        touching = index >= 30
        recorder.add(
            synthetic_tripod(
                start + index / 30,
                contact=0.1 if touching else 0.8,
                noise=tuple(rng.normal(0, 0.0007, 2)),
            )
        )
    recorder.close()
    intervals = [
        {"start": 0, "end": 30, "middle": 1, "index": 0, "ring": 0},
        {"start": 30, "end": 60, "middle": 1, "index": 1, "ring": 0},
    ]
    labels = path.with_suffix(".labels.json")
    draft_annotation(path, labels, intervals, "pinchpilot-guided-session-1")
    protocol = path.with_suffix(".protocol.json")
    steps = [
        {"start": 0.0, "end": 1.0, "round": 1, "name": "grip_move"},
        {"start": 1.0, "end": 2.0, "round": 2, "name": "grip_index_hold"},
    ]
    protocol.write_text(
        json.dumps(
            {
                "schema": "pinchpilot-contact-protocol-v1",
                "protocol": "pinchpilot-guided-session-1",
                "clock_start": start,
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    if not bundle:
        return path
    archive = directory / f"{stem}_{participant}_session.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as target:
        for item in (path, labels, protocol):
            target.write(item, item.name)
            item.unlink()
    return archive


def review(labels, source, annotator="P01", index=1):
    spec = json.loads(source.read_text(encoding="utf-8"))
    spec.update(reviewed=True, label_source="human_reviewed", annotator=annotator)
    spec["intervals"][1]["index"] = index
    labels.write_text(json.dumps(spec), encoding="utf-8")


def test_pool_windows_match_the_window_builder_and_carry_person_round_and_step(tmp_path):
    create_synthetic_contact_pool(tmp_path, participants=3, sessions=2)
    pool = load_contact_pool(tmp_path)
    assert pool.metadata["source"] == "synthetic" and pool.metadata["labels"] == "reviewed"
    assert sorted(set(pool.participants.tolist())) == [f"synthetic_P0{i}" for i in (1, 2, 3)]
    assert len(pool.metadata["recordings"]) == 6 and pool.frames == 8

    first = pool.metadata["recordings"][0]
    _, frames = read_contact_recording(tmp_path / first["file"])
    builder, expected = WindowBuilder(), []
    for frame in frames:
        window = builder.push(frame)
        if window is not None:
            expected.append(window)
    rows = np.flatnonzero(pool.recordings == 0)
    assert len(rows) == first["windows"] == len(expected)
    assert np.array_equal(pool.windows(rows), np.stack(expected))
    # The fixture walks all eight label combinations twice, 24 frames each.
    assert set(map(tuple, pool.y[rows].tolist())) == {
        (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)
    }
    assert set(pool.rounds[rows].tolist()) == {1, 2}
    names = pool.metadata["step_names"]
    for row in rows[::37]:
        flags = "".join(str(value) for value in pool.y[row])
        assert names[pool.steps[row]] == f"synthetic_{flags}"
    report = summary(pool)
    assert report["windows"] == len(pool) and len(report["participants"]) == 3
    assert report["participants"]["synthetic_P01"]["positive_windows"]["ring"] > 0
    assert report["unlabelled_windows"] == 0 and len(pool.spare_last) == 0


def test_windows_without_a_label_are_kept_apart_as_keypoints_only(tmp_path):
    create_synthetic_contact_pool(tmp_path, participants=2, sessions=1)
    whole = load_contact_pool(tmp_path)
    labels = tmp_path / "synthetic_P02" / "synthetic_P02_s1.labels.json"
    spec = json.loads(labels.read_text(encoding="utf-8"))
    spec["intervals"][3]["index"] = None  # 24 frames whose index label is unknown.
    labels.write_text(json.dumps(spec), encoding="utf-8")
    pool = load_contact_pool(tmp_path)
    assert len(pool.spare_last) == 24 and len(pool) == len(whole) - 24
    assert set(pool.spare_participants.tolist()) == {"synthetic_P02"}
    assert summary(pool)["unlabelled_windows"] == 24
    # They are the same complete windows as before, only without an answer.
    missing = sorted(set(whole.last.tolist()) - set(pool.last.tolist()))
    assert missing == sorted(pool.spare_last.tolist())
    assert pool.gather(pool.spare_last).shape == (24, 8, 67)
    assert np.array_equal(pool.gather(pool.last[:5]), pool.windows(np.arange(5)))


def test_unreviewed_drafts_need_the_draft_switch_and_mark_the_pool(tmp_path):
    guided(tmp_path / "P01", "a", "P01")
    with pytest.raises(ValueError, match="还没有复核.*--draft"):
        load_contact_pool(tmp_path)
    pool = load_contact_pool(tmp_path, allow_draft=True)
    assert pool.metadata["labels"] == "draft" and pool.metadata["source"] == "camera"
    assert pool.metadata["recordings"][0]["protocol"] == "pinchpilot-guided-session-1"
    assert set(pool.rounds.tolist()) == {1, 2}
    assert pool.metadata["step_names"] == ["grip_move", "grip_index_hold"]
    assert pool.y[pool.rounds == 2][:, CHANNELS.index("index")].all()


def test_reviewed_labels_beside_a_zip_replace_the_draft_inside_it(tmp_path):
    archive = guided(tmp_path / "P01", "a", "P01")
    with zipfile.ZipFile(archive) as source:
        draft = tmp_path / "draft.json"
        draft.write_bytes(source.read("a.labels.json"))
    review(archive.with_name(f"{archive.stem}.labels.json"), draft, index=0)
    draft.unlink()
    pool = load_contact_pool(tmp_path)
    assert pool.metadata["labels"] == "reviewed" and len(pool.metadata["recordings"]) == 1
    assert not pool.y[:, CHANNELS.index("index")].any()  # The reviewer said: never touching.


def test_the_same_recording_as_zip_and_loose_files_counts_once(tmp_path):
    loose = guided(tmp_path / "P01", "a", "P01", bundle=False)
    with zipfile.ZipFile(tmp_path / "P01" / "a_P01_session.zip", "x") as target:
        for suffix in (".jsonl", ".labels.json", ".protocol.json"):
            target.write(loose.with_suffix(suffix), loose.with_suffix(suffix).name)
    pool = load_contact_pool(tmp_path, allow_draft=True)
    assert len(pool.metadata["recordings"]) == 1
    with pytest.raises(ValueError, match="1 段录制的标签还没有复核"):
        load_contact_pool(tmp_path)


def test_copies_shifted_in_time_and_mixed_sources_are_rejected(tmp_path):
    first = guided(tmp_path / "camera" / "P01", "a", "P01", bundle=False)
    guided(tmp_path / "camera" / "P02", "b", "P02", start=900.0, bundle=False)
    with pytest.raises(ValueError, match="重复录制"):
        load_contact_pool(tmp_path / "camera", allow_draft=True)
    shutil.rmtree(first.parent.parent / "P02")
    create_synthetic_contact_pool(tmp_path / "camera" / "synthetic", participants=1, sessions=1)
    with pytest.raises(ValueError, match="合成与真人"):
        load_contact_pool(tmp_path / "camera", allow_draft=True)


def test_broken_or_incomplete_inputs_are_named(tmp_path):
    with pytest.raises(ValueError, match="找不到数据文件夹"):
        load_contact_pool(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="没有带标签的录制"):
        load_contact_pool(tmp_path / "empty")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "broken.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="压缩包无法读取：broken.zip"):
        load_contact_pool(bad)
    partial = tmp_path / "partial"
    partial.mkdir()
    with zipfile.ZipFile(partial / "only_summary.zip", "x") as target:
        target.writestr("a.capture.json", "{}")
    with pytest.raises(ValueError, match="缺少录制或标签"):
        load_contact_pool(partial)


def test_cache_is_reused_until_an_input_changes(tmp_path, monkeypatch):
    data, cache = tmp_path / "data", tmp_path / "cache"
    archive = guided(data / "P01", "a", "P01")
    built = load_contact_pool(data, allow_draft=True, cache=cache)
    assert len(list(cache.glob("pool-*.npz"))) == 1

    def refuse(*args):
        raise AssertionError("the cached pool should have been used")

    with monkeypatch.context() as patch:
        patch.setattr(contact_pool, "_build", refuse)
        cached = load_contact_pool(data, allow_draft=True, cache=cache)
    for name in contact_pool.ARRAYS + contact_pool.SPARE:
        assert np.array_equal(getattr(cached, name), getattr(built, name))
    assert cached.metadata == built.metadata and cached.frames == built.frames

    with zipfile.ZipFile(archive) as source:
        draft = tmp_path / "draft.json"
        draft.write_bytes(source.read("a.labels.json"))
    review(archive.with_name(f"{archive.stem}.labels.json"), draft)
    changed = load_contact_pool(data, allow_draft=True, cache=cache)
    assert changed.metadata["labels"] == "reviewed"
    assert changed.metadata["fingerprint"] != built.metadata["fingerprint"]
    assert len(list(cache.glob("pool-*.npz"))) == 2
