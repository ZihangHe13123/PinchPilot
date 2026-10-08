"""Offline, independent contact classifiers. No camera or desktop integration.

Random forest artifacts use joblib and must only be loaded from trusted local
sources. CNN artifacts contain numeric arrays only; PyTorch is imported lazily.
"""

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from . import __version__
from .contact_data import CHANNELS, FEATURE_COUNT, FEATURE_VERSION, WindowConfig

SCHEMA = "pinchpilot-contact-model-v1"
CNN_ARCHITECTURE = "causal-conv32-k3-kTminus2-last-v1"
SPLITS = ("train", "validation", "test")


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _window_config(config):
    if not isinstance(config, dict) or set(config) != {
        "frames",
        "max_gap_seconds",
        "max_window_seconds",
    }:
        raise ValueError("window_config 必须包含 frames/max_gap_seconds/max_window_seconds")
    return asdict(WindowConfig(**config))


def _windows(x, frames):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (frames, FEATURE_COUNT):
        raise ValueError(f"输入应为 [N, {frames}, {FEATURE_COUNT}] 因果窗口")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("窗口特征必须是有限实数")
    with np.errstate(over="ignore", invalid="ignore"):
        result = x.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("窗口特征超出 float32 有效范围")
    return result


def _validate_dataset(dataset, allow_synthetic):
    metadata = _json_copy(dataset.metadata)
    config = _window_config(asdict(dataset.window_config))
    if (
        metadata.get("window_config") != config
        or metadata.get("feature_version") != FEATURE_VERSION
    ):
        raise ValueError("数据特征版本或窗口配置与训练契约不一致")
    source = metadata.get("source")
    if source not in ("camera", "synthetic"):
        raise ValueError("训练数据必须明确标注 source=camera 或 synthetic")
    if source == "synthetic" and not allow_synthetic:
        raise ValueError("合成数据默认拒绝训练；仅工程验证可显式 allow_synthetic=True")
    if metadata.get("group_by") not in ("participant", "session"):
        raise ValueError("数据必须按 participant 或 session 分组")
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("数据缺少 fingerprint")
    if set(dataset.partitions) != set(SPLITS):
        raise ValueError("数据必须显式划分 train/validation/test")
    checked, seen = {}, set()
    for split in SPLITS:
        part = dataset.partitions[split]
        x = _windows(part.x, config["frames"])
        y, rules, groups = np.asarray(part.y), np.asarray(part.rules), np.asarray(part.groups)
        if len(x) == 0 or y.shape != (len(x), 3) or rules.shape != y.shape:
            raise ValueError(f"{split} 缺少窗口或 y/rules 不是 [N, 3]")
        for name, values in (("y", y), ("rules", rules)):
            if not np.isin(values, (0, 1)).all():
                raise ValueError(f"{split}.{name} 必须严格为 0/1，无未知标签")
        for i, channel in enumerate(CHANNELS):
            if set(y[:, i].tolist()) != {0, 1}:
                raise ValueError(f"{split} 的 {channel} 通道必须同时包含 0 和 1")
        if groups.shape != (len(x),) or any(
            not isinstance(g, str) or not g for g in groups.tolist()
        ):
            raise ValueError(f"{split}.groups 必须是 [N] 非空字符串")
        actual = set(groups.tolist())
        declared = metadata.get("groups", {}).get(split)
        if not isinstance(declared, list) or set(declared) != actual:
            raise ValueError(f"{split} 分组清单与实际窗口不一致")
        if actual & seen:
            raise ValueError("train/validation/test 存在相同分组，拒绝数据泄漏")
        seen.update(actual)
        checked[split] = (x, y.astype(np.int8), rules.astype(np.int8), groups)
    return checked, metadata, config


def _binary_metrics(y, predicted):
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "confusion_labels": [0, 1],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "support": len(y),
        "positive_support": int(np.sum(y)),
    }


def _score(y, predicted):
    channels = {name: _binary_metrics(y[:, i], predicted[:, i]) for i, name in enumerate(CHANNELS)}
    return {
        "channels": channels,
        "macro_f1": float(np.mean([row["f1"] for row in channels.values()])),
    }


def _partition_score(part, scores):
    _, y, rules, groups = part
    predicted = (scores >= 0.5).astype(np.int8)
    return {
        "windows": len(y),
        "learned": _score(y, predicted),
        "rules": _score(y, rules),
        "per_group": {
            group: {
                "learned": _score(y[groups == group], predicted[groups == group]),
                "rules": _score(y[groups == group], rules[groups == group]),
            }
            for group in sorted(set(groups.tolist()))
        },
    }


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "CNN 需要可选依赖 torch；请把命令开头改成 uv run --extra ml pinchpilot"
        ) from error
    return torch


def _last_output(x, conv1, conv2):
    """What two causal convolutions (zeros before the oldest frame) give at the last frame.

    x is [N, T, features]; conv1 has kernel 3 and conv2 kernel max(1, T - 2). The last
    output needs conv1 only at its final positions, so both layers are computed as matrix
    products over the same weights, which is many times faster on a CPU than Conv1d.
    """
    torch = _torch()
    count, kernel = x.shape[0], conv2.kernel_size[0]
    missing = kernel + 2 - x.shape[1]
    if missing > 0:
        x = torch.nn.functional.pad(x, (0, 0, missing, 0))
    x = x.unfold(1, 3, 1).reshape(count, kernel, -1)
    x = torch.relu(torch.nn.functional.linear(x, conv1.weight.flatten(1), conv1.bias))
    x = x.transpose(1, 2).reshape(count, -1)
    return torch.relu(torch.nn.functional.linear(x, conv2.weight.flatten(1), conv2.bias))


def _make_cnn(frames, seed=0, channels=32):
    torch = _torch()

    class CausalContactCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(FEATURE_COUNT, channels, 3)
            self.second_kernel = max(1, frames - 2)
            self.conv2 = torch.nn.Conv1d(channels, channels, self.second_kernel)
            self.head = torch.nn.Linear(channels, 3)

        def forward(self, x):
            return self.head(_last_output(x, self.conv1, self.conv2))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = CausalContactCNN().cpu()
    return network


@contextmanager
def _deterministic_cpu(torch):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def _normalize(x, mean, scale):
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = (x.astype(np.float64) - mean) / scale
        normalized = normalized.astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("标准化后的窗口超出有效范围")
    return normalized


def _cnn_scores(network, x, mean, scale):
    torch = _torch()
    network.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            values = torch.from_numpy(_normalize(x[start : start + 256], mean, scale))
            scores.append(torch.sigmoid(network(values)).numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0, 3), dtype=np.float32)


def _train_cnn(train, validation, seed, epochs):
    torch = _torch()
    x, y = train[:2]
    vx, vy = validation[:2]
    mean = x.astype(np.float64).mean(axis=(0, 1))
    scale = x.astype(np.float64).std(axis=(0, 1))
    scale = np.where(scale < 1e-6, 1.0, scale)
    inputs = torch.from_numpy(_normalize(x, mean, scale))
    labels = torch.from_numpy(y.astype(np.float32))
    valid_inputs = torch.from_numpy(_normalize(vx, mean, scale))
    valid_labels = torch.from_numpy(vy.astype(np.float32))
    network = _make_cnn(x.shape[1], seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    with _deterministic_cpu(torch):
        for epoch in range(1, epochs + 1):
            network.train()
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(x), 64):
                indices = order[start : start + 64]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(network(inputs[indices]), labels[indices])
                if not torch.isfinite(loss):
                    raise ValueError("CNN 训练损失不是有限值")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            network.eval()
            with torch.inference_mode():
                validation_loss = float(loss_fn(network(valid_inputs), valid_labels))
            if not math.isfinite(validation_loss):
                raise ValueError("CNN 验证损失不是有限值")
            history.append(
                {"epoch": epoch, "train_bce": loss_sum / len(x), "validation_bce": validation_loss}
            )
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = {
                    key: value.detach().clone() for key, value in network.state_dict().items()
                }
        network.load_state_dict(best_state, strict=True)
    return (
        network,
        mean,
        scale,
        {
            "selected_epoch": best_epoch,
            "history": history,
            "selection": "minimum validation BCE; no test selection",
            "epochs_requested": epochs,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "loss": "mean independent binary cross entropy with logits",
            "torch_version": torch.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
        },
    )


def train_contacts(
    dataset, output: Path, *, model="forest", seed=42, epochs=20, allow_synthetic=False
):
    if model not in ("forest", "cnn"):
        raise ValueError("接触模型应为 forest 或 cnn")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 - 3:
        raise ValueError("seed 必须是有效非负整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10000:
        raise ValueError("epochs 必须为 1–10000 的整数")
    parts, data_metadata, window = _validate_dataset(dataset, allow_synthetic)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录已有模型；请使用新目录保存本次实验")
    source = data_metadata["source"]
    metadata = {
        "schema": SCHEMA,
        "app_version": __version__,
        "library_versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "model": model,
        "feature_version": FEATURE_VERSION,
        "feature_count": FEATURE_COUNT,
        "channels": list(CHANNELS),
        "window_config": window,
        "source": source,
        "training_source": source,
        "evidence": "synthetic_engineering" if source == "synthetic" else "held_out_camera_windows",
        "dataset_fingerprint": data_metadata["fingerprint"],
        "group_by": data_metadata["group_by"],
        "groups": data_metadata["groups"],
        "seed": seed,
        "threshold": 0.5,
        "score_meaning": "Independent uncalibrated contact scores; not mutually exclusive and not click intent.",
    }
    if model == "forest":
        x, y = parts["train"][:2]
        heads = []
        for channel in range(3):
            head = RandomForestClassifier(
                n_estimators=160,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=seed + channel,
            )
            head.fit(x.reshape(len(x), -1), y[:, channel])
            heads.append(head)

        def predict(values):
            flat = values.reshape(len(values), -1)
            return np.column_stack([head.predict_proba(flat)[:, 1] for head in heads])

        training = {
            "selection": "fixed forest parameters; train only",
            "normalization": "none",
            "heads": 3,
            "n_estimators_per_head": 160,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": 1,
        }
        metadata["artifact"] = "model.joblib"
    else:
        metadata["library_versions"]["torch"] = str(_torch().__version__)
        network, mean, scale, training = _train_cnn(
            parts["train"], parts["validation"], seed, epochs
        )

        def predict(values):
            return _cnn_scores(network, values, mean, scale)

        training["normalization"] = "feature mean/std fitted on train windows only"
        metadata.update({"artifact": "weights.npz", "architecture": CNN_ARCHITECTURE})
    # Test is visited here once, after all fitting/epoch selection is complete.
    metrics = {split: _partition_score(parts[split], predict(parts[split][0])) for split in SPLITS}
    report = {
        **metadata,
        "training": training,
        "metrics": metrics,
        "learned": metrics["test"]["learned"],
        "rules": metrics["test"]["rules"],
        "limitations": [
            "Classifies contact at the final frame of a causal window, not clicks, intent, comfort or OS input.",
            "Overlapping windows are correlated; window counts are not independent participant counts.",
            "Decision threshold is fixed at 0.5; scores are not calibrated probabilities.",
            "No live control changes; matching a synthetic generator is engineering evidence only.",
            "Normalization and fitting use train only; CNN checkpoint selection uses validation only; test is scored once.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / metadata["artifact"]
    if model == "forest":
        joblib.dump({"schema": SCHEMA, "heads": heads}, artifact_path)
    else:
        arrays = {key: value.detach().cpu().numpy() for key, value in network.state_dict().items()}
        np.savez_compressed(artifact_path, mean=mean, scale=scale, **arrays)
    metadata["artifact_sha256"] = _digest(artifact_path)
    report["artifact_sha256"] = metadata["artifact_sha256"]
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


class ContactPredictor:
    """Load only trusted local forest directories: joblib can execute Python."""

    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            directory = path
        elif path.is_file() and path.name == "metadata.json":
            directory = path.parent
        else:
            raise ValueError("请提供存在的接触模型目录或 metadata.json")
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or (
            metadata.get("schema") != SCHEMA
            or metadata.get("model") not in ("forest", "cnn")
            or metadata.get("feature_version") != FEATURE_VERSION
            or metadata.get("feature_count") != FEATURE_COUNT
            or metadata.get("channels") != list(CHANNELS)
            or metadata.get("source") not in ("camera", "synthetic")
            or metadata.get("training_source") != metadata.get("source")
            or metadata.get("threshold") != 0.5
        ):
            raise ValueError("模型元数据版本、特征、通道或来源不兼容")
        expected_evidence = (
            "synthetic_engineering"
            if metadata["source"] == "synthetic"
            else "held_out_camera_windows"
        )
        if metadata.get("evidence") != expected_evidence:
            raise ValueError("模型来源与证据类型不一致")
        _window_config(metadata.get("window_config"))
        self.metadata = metadata
        self.frames = metadata["window_config"]["frames"]
        expected_artifact = "model.joblib" if metadata["model"] == "forest" else "weights.npz"
        if metadata.get("artifact") != expected_artifact:
            raise ValueError("模型文件名与类型不匹配")
        artifact = directory / expected_artifact
        if _digest(artifact) != metadata.get("artifact_sha256"):
            raise ValueError("模型文件哈希不匹配，可能损坏或混用了不同实验文件")
        if metadata["model"] == "forest":
            saved = joblib.load(artifact)
            if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
                raise ValueError("随机森林 artifact 格式不兼容")
            self.heads = saved.get("heads")
            if (
                not isinstance(self.heads, list)
                or len(self.heads) != 3
                or any(
                    not isinstance(head, RandomForestClassifier)
                    or getattr(head, "n_features_in_", None) != self.frames * FEATURE_COUNT
                    or not np.array_equal(getattr(head, "classes_", None), [0, 1])
                    for head in self.heads
                )
            ):
                raise ValueError("随机森林通道或特征形状不兼容")
        else:
            if metadata.get("architecture") != CNN_ARCHITECTURE:
                raise ValueError("CNN 架构版本不兼容")
            self.network = _make_cnn(self.frames)
            expected = self.network.state_dict()
            with np.load(artifact, allow_pickle=False) as saved:
                if set(saved.files) != set(expected) | {"mean", "scale"}:
                    raise ValueError("CNN 权重字段不兼容")
                arrays = {key: saved[key] for key in saved.files}
            if any(
                not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value)
                or not np.isfinite(value).all()
                for value in arrays.values()
            ):
                raise ValueError("CNN 权重或归一化包含无效数值")
            self.mean, self.scale = arrays.pop("mean"), arrays.pop("scale")
            if (
                self.mean.shape != (FEATURE_COUNT,)
                or self.scale.shape != (FEATURE_COUNT,)
                or (self.scale <= 0).any()
            ):
                raise ValueError("CNN 归一化参数形状或范围无效")
            if any(arrays[key].shape != tuple(expected[key].shape) for key in expected):
                raise ValueError("CNN 权重形状不匹配")
            torch = _torch()
            self.network.load_state_dict(
                {key: torch.from_numpy(value).to(torch.float32) for key, value in arrays.items()},
                strict=True,
            )
            self.network.eval()

    def predict_windows(self, x):
        x = _windows(x, self.frames)
        if len(x) == 0:
            return np.empty((0, 3), dtype=np.float32)
        if self.metadata["model"] == "forest":
            scores = np.column_stack(
                [head.predict_proba(x.reshape(len(x), -1))[:, 1] for head in self.heads]
            )
        else:
            scores = _cnn_scores(self.network, x, self.mean, self.scale)
        if (
            scores.shape != (len(x), 3)
            or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()
        ):
            raise ValueError("模型输出不是有效的三路接触分数")
        return scores
