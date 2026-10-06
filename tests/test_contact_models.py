"""Synthetic engineering checks for isolated contact training and artifacts."""

import json
import shutil
from dataclasses import asdict
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from pinchpilot import contact_models
from pinchpilot.contact_data import CHANNELS, FEATURE_COUNT, FEATURE_VERSION, WindowConfig
from pinchpilot.contact_models import ContactPredictor, train_contacts


def dataset(seed=12, frames=8):
    rng = np.random.default_rng(seed)
    config = WindowConfig(frames=frames)
    partitions = {}
    declared = {}
    for split in ("train", "validation", "test"):
        labels = np.tile(np.array(list(product((0, 1), repeat=3)), dtype=np.int8), (6, 1))
        x = rng.normal(0, 0.1, (len(labels), frames, FEATURE_COUNT)).astype(np.float32)
        x[:, :, :3] += labels[:, None, :] * 2
        groups = np.array([split + "-person-a"] * 24 + [split + "-person-b"] * 24)
        partitions[split] = SimpleNamespace(
            x=x,
            y=labels.copy(),
            rules=labels.copy(),
            groups=groups,
            episodes=np.array([split] * len(labels)),
            frame_indices=np.arange(len(labels)),
            timestamps=np.arange(len(labels)) / 30,
        )
        declared[split] = sorted(set(groups))
    return SimpleNamespace(
        partitions=partitions,
        window_config=config,
        metadata={
            "source": "synthetic",
            "feature_version": FEATURE_VERSION,
            "window_config": asdict(config),
            "group_by": "participant",
            "groups": declared,
            "fingerprint": "f" * 64,
        },
    )


@pytest.fixture(scope="module")
def forest(tmp_path_factory):
    data = dataset()
    output = tmp_path_factory.mktemp("contact-forest")
    report = train_contacts(data, output, allow_synthetic=True)
    return data, output, report


def test_synthetic_training_requires_explicit_override(tmp_path):
    with pytest.raises(ValueError, match="合成数据默认拒绝"):
        train_contacts(dataset(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_training_cannot_overwrite_partial_or_unrelated_artifacts(tmp_path):
    previous = tmp_path / "metrics.json"
    previous.write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="新目录"):
        train_contacts(dataset(), tmp_path, allow_synthetic=True)
    assert previous.read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize("split", ("train", "validation", "test"))
@pytest.mark.parametrize("channel", (0, 1, 2))
def test_every_split_channel_needs_both_labels(tmp_path, split, channel):
    data = dataset()
    data.partitions[split].y[:, channel] = 0
    with pytest.raises(ValueError, match=f"{split} 的 {CHANNELS[channel]}"):
        train_contacts(data, tmp_path, allow_synthetic=True)


def test_overlapping_groups_are_rejected(tmp_path):
    data = dataset()
    data.partitions["test"].groups = np.array(["train-person-a"] * 48)
    data.metadata["groups"]["test"] = ["train-person-a"]
    with pytest.raises(ValueError, match="相同分组"):
        train_contacts(data, tmp_path, allow_synthetic=True)


def test_unknown_label_and_nonfinite_features_rejected(tmp_path):
    data = dataset()
    data.partitions["train"].y[0, 0] = -1
    with pytest.raises(ValueError, match="0/1"):
        train_contacts(data, tmp_path, allow_synthetic=True)
    data = dataset()
    data.partitions["test"].x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="有限实数"):
        train_contacts(data, tmp_path, allow_synthetic=True)


def test_forest_roundtrip_metrics_and_simultaneous_contacts(forest):
    data, output, report = forest
    predictor = ContactPredictor(output)
    scores = predictor.predict_windows(data.partitions["test"].x)
    assert scores.shape == (48, 3) and np.isfinite(scores).all()
    assert (scores[7] >= 0.5).all(), "Three contact heads must be allowed to activate together"
    assert predictor.metadata["source"] == predictor.metadata["training_source"] == "synthetic"
    assert report["evidence"] == "synthetic_engineering"
    for split, metrics in report["metrics"].items():
        assert metrics["windows"] == 48
        assert set(metrics["per_group"]) == set(data.partitions[split].groups)
        for name in CHANNELS:
            matrix = metrics["learned"]["channels"][name]["confusion_matrix"]
            assert sum(map(sum, matrix)) == 48
            assert metrics["rules"]["channels"][name]["f1"] == 1
    assert report == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "click" in report["limitations"][0]


def test_inference_shape_finite_and_empty(forest):
    _, output, _ = forest
    predictor = ContactPredictor(output / "metadata.json")
    assert predictor.predict_windows(np.empty((0, 8, 67))).shape == (0, 3)
    for x in (
        np.zeros((3, 67)),
        np.zeros((3, 7, 67)),
        np.full((3, 8, 67), np.inf),
        np.full((3, 8, 67), 1e100),
    ):
        with pytest.raises(ValueError):
            predictor.predict_windows(x)
    with pytest.raises(ValueError, match="模型目录"):
        ContactPredictor(output / "does-not-exist")


@pytest.mark.parametrize(
    "key,value",
    (
        ("feature_version", "old"),
        ("channels", ["index", "middle", "ring"]),
        ("source", "camera"),
        ("schema", "old"),
        ("model", "unknown"),
    ),
)
def test_bad_metadata_rejected_before_joblib_load(forest, tmp_path, monkeypatch, key, value):
    _, output, _ = forest
    shutil.copytree(output, tmp_path, dirs_exist_ok=True)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata[key] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        contact_models.joblib, "load", lambda _: pytest.fail("pickle should not load")
    )
    with pytest.raises(ValueError):
        ContactPredictor(tmp_path)


def test_changed_artifact_hash_is_rejected(forest, tmp_path):
    _, output, _ = forest
    shutil.copytree(output, tmp_path, dirs_exist_ok=True)
    with (tmp_path / "model.joblib").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        ContactPredictor(tmp_path)


def test_forest_does_not_import_torch(forest, monkeypatch):
    data, output, _ = forest
    monkeypatch.setattr(contact_models, "_torch", lambda: pytest.fail("forest imported torch"))
    ContactPredictor(output).predict_windows(data.partitions["test"].x)


def test_forest_does_not_fit_test_labels(forest, tmp_path):
    data, output, _ = forest
    changed = dataset()
    changed.partitions["test"].y = 1 - changed.partitions["test"].y
    train_contacts(changed, tmp_path, allow_synthetic=True)
    x = data.partitions["test"].x
    np.testing.assert_array_equal(
        ContactPredictor(output).predict_windows(x), ContactPredictor(tmp_path).predict_windows(x)
    )


@pytest.fixture(scope="module")
def cnn(tmp_path_factory):
    pytest.importorskip("torch")
    data = dataset()
    data.partitions["validation"].x[:, :, 10] += 100
    output = tmp_path_factory.mktemp("contact-cnn")
    report = train_contacts(data, output, model="cnn", epochs=3, allow_synthetic=True)
    return data, output, report


def test_cnn_roundtrip_train_only_normalization_and_epoch_selection(cnn):
    data, output, report = cnn
    predictor = ContactPredictor(output)
    scores = predictor.predict_windows(data.partitions["test"].x)
    assert scores.shape == (48, 3) and np.isfinite(scores).all()
    np.testing.assert_allclose(
        predictor.mean, data.partitions["train"].x.astype(np.float64).mean(axis=(0, 1))
    )
    history = report["training"]["history"]
    assert (
        report["training"]["selected_epoch"]
        == min(history, key=lambda row: row["validation_bce"])["epoch"]
    )
    labels = (scores >= 0.5).astype(np.int8)
    assert contact_models._score(data.partitions["test"].y, labels) == report["learned"]
    with np.load(output / "weights.npz", allow_pickle=False) as saved:
        assert all(not saved[name].dtype.hasobject for name in saved.files)


def test_cnn_seed_reproducible_and_test_cannot_select_model(cnn, tmp_path):
    data, output, report = cnn
    data2 = dataset()
    data2.partitions["validation"].x[:, :, 10] += 100
    data2.partitions["test"].x += 30
    changed = train_contacts(data2, tmp_path, model="cnn", epochs=3, allow_synthetic=True)
    assert changed["training"] == report["training"]
    x = data.partitions["train"].x
    np.testing.assert_array_equal(
        ContactPredictor(output).predict_windows(x), ContactPredictor(tmp_path).predict_windows(x)
    )


@pytest.mark.parametrize("frames", (2, 8, 12))
def test_cnn_last_output_can_see_oldest_frame(frames):
    torch = pytest.importorskip("torch")
    network = contact_models._make_cnn(frames)
    with torch.no_grad():
        for value in network.parameters():
            value.fill_(0.01)
        blank = torch.zeros(1, frames, FEATURE_COUNT)
        changed = blank.clone()
        changed[:, 0, :] = 1
        assert (network(changed) > network(blank)).all()


def test_existing_model_directory_is_not_silently_overwritten(forest):
    data, output, _ = forest
    with pytest.raises(ValueError, match="已有模型"):
        train_contacts(data, output, allow_synthetic=True)
