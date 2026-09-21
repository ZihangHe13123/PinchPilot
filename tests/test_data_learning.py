from dataclasses import replace
from shutil import copyfile

import numpy as np
import pytest

from pinchpilot.demo import synthetic_hand
from pinchpilot.domain import EngineConfig, HandFrame
from pinchpilot.features import extract
from pinchpilot.learning import (
    Dataset,
    LearnedPredictor,
    _score,
    grouped_split,
    load_dataset,
    train,
)
from pinchpilot.storage import Recorder, calibrate, read_recording


def record(directory, participant="P1", label="open", n=30):
    rec = Recorder(directory, participant, "session1", label)
    for i in range(n):
        rec.add(synthetic_hand(i / 30, "fist" if label == "other" else label))
    rec.close()
    return rec.path


def test_features_ignore_translation_uniform_scale_and_aspect_conversion():
    original = synthetic_hand(0, "pinch")
    base = extract(original)
    points = tuple((x * 0.7 + 0.1, y * 0.7 - 0.04, z) for x, y, z in original.landmarks)
    assert extract(replace(original, landmarks=points)).vector == pytest.approx(base.vector)
    new_aspect = 16 / 9
    resized = tuple((x * original.aspect / new_aspect, y, z) for x, y, z in original.landmarks)
    assert extract(replace(original, landmarks=resized, aspect=new_aspect)).vector == pytest.approx(
        base.vector
    )


@pytest.mark.parametrize("points", [(), ((0, 0, 0),) * 21, ((float("nan"), 0, 0),) * 21])
def test_unusable_hand_rejected(points):
    assert extract(HandFrame(1, points)) is None


def test_record_roundtrip_and_timestamp_validation(tmp_path):
    rec = Recorder(tmp_path, "P1", "S1", "open")
    rec.add(synthetic_hand(1))
    rec.add(HandFrame(2))
    with pytest.raises(ValueError, match="递增"):
        rec.add(synthetic_hand(2))
    rec.close()
    meta, frames = read_recording(rec.path)
    assert meta["label_source"] == "human_selected"
    assert not meta["video_saved"]
    assert len(frames) == 2 and rec.valid_count == 1


def test_bad_label_and_duplicate_episode_rejected(tmp_path):
    with pytest.raises(ValueError):
        Recorder(tmp_path, "P1", "S1", "guessed_label")
    path = record(tmp_path)
    copyfile(path, tmp_path / "duplicate.jsonl")
    with pytest.raises(ValueError, match="重复"):
        load_dataset(tmp_path)


def test_calibration_requires_separable_human_labelled_shapes(tmp_path):
    record(tmp_path, label="open")
    record(tmp_path, label="pinch")
    config, report = calibrate(tmp_path, "P1", EngineConfig())
    assert report["pinch_q85"] < config.engage_ratio < config.release_ratio < report["open_q15"]
    with pytest.raises(ValueError, match="至少"):
        calibrate(tmp_path, "absent", EngineConfig())


def dataset():
    # Synthetic, temporary engineering fixture. Not stored as project research data.
    xs, ys, groups = [], [], []
    rng = np.random.default_rng(10)
    for group in ("P1", "P2", "P3", "P4"):
        for label in ("open", "pinch", "scroll", "other"):
            f = extract(synthetic_hand(0, "fist" if label == "other" else label))
            for _ in range(20):
                xs.append(np.asarray(f.vector) + rng.normal(0, 0.005, len(f.vector)))
                ys.append(label)
                groups.append(group)
    return Dataset(
        np.asarray(xs), np.asarray(ys), np.asarray(groups), np.asarray(ys), np.asarray(groups), {}
    )


def test_group_holdout_is_disjoint_and_single_group_fails():
    data = dataset()
    training, testing = grouped_split(data)
    assert not set(data.groups[training]) & set(data.groups[testing])
    data.groups[:] = "P1"
    with pytest.raises(ValueError, match="至少"):
        grouped_split(data)


@pytest.mark.parametrize("model", ["forest", "svm"])
def test_train_save_load_and_runtime_policy_agree(tmp_path, monkeypatch, model):
    import pinchpilot.learning as learning

    data = dataset()
    monkeypatch.setattr(learning, "load_dataset", lambda *_: data)
    report = train(tmp_path, tmp_path / "output", model=model)
    predictor = LearnedPredictor(tmp_path / "output/model.joblib")
    assert predictor.predict(extract(synthetic_hand(0, "pinch"))).label == "pinch"
    assert set(report["training_groups"]).isdisjoint(report["testing_groups"])
    _, testing = grouped_split(data)
    predicted, _ = learning.predict_with_rejection(
        predictor.estimator, data.x[testing], predictor.policy
    )
    assert _score(data.y[testing], predicted, report["labels"]) == report["learned"]
    assert (tmp_path / "output/confusion.png").stat().st_size > 1000


def test_confusion_counts_all_abstentions_and_outside_labels():
    result = _score(
        np.array(["open", "pinch"]), np.array(["other", "uncertain"]), ["open", "pinch"]
    )
    assert np.asarray(result["confusion_matrix"]).sum() == 2
    assert result["macro_f1"] == 0
    assert result["coverage"] == 0.5
