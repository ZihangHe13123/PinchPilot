"""Grouped evaluation. Learned artifacts must be trusted local files (joblib)."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .domain import Features, Prediction
from .features import FEATURE_NAMES, FEATURE_VERSION, extract, rule_prediction
from .storage import read_recording, recording_files, save_json

REJECTION_POLICY = {"forest_probability": 0.70, "svm_margin": 0.25}


def predict_with_rejection(estimator, x, policy: dict) -> tuple[np.ndarray, np.ndarray]:
    """Shared offline/online decision policy; these scores are not calibrated probabilities."""
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x)
        confidence = scores.max(axis=1)
        predicted = estimator.classes_[scores.argmax(axis=1)]
        threshold = policy["forest_probability"]
    else:
        margins = np.asarray(estimator.decision_function(x))
        confidence = (
            np.abs(margins)
            if margins.ndim == 1
            else np.diff(np.sort(margins, axis=1)[:, -2:], axis=1).ravel()
        )
        predicted = estimator.predict(x)
        threshold = policy["svm_margin"]
    return np.where(confidence >= threshold, predicted, "uncertain"), confidence


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    rules: np.ndarray
    episodes: np.ndarray
    counts: dict


def load_dataset(directory: Path, group_by: str = "participant") -> Dataset:
    if group_by not in ("participant", "session"):
        raise ValueError("分组方式必须为 participant 或 session")
    x, y, groups, rules, episodes = [], [], [], [], []
    counts = {"files": 0, "frames": 0, "unusable_frames": 0}
    seen = set()
    for path in recording_files(directory):
        meta, frames = read_recording(path)
        episode = meta["episode"]
        if episode in seen:
            raise ValueError("发现重复 episode；避免复制同一录制导致数据泄漏")
        seen.add(episode)
        counts["files"] += 1
        group = meta["participant"]
        if group_by == "session":
            group += "/" + meta["session"]
        for frame in frames:
            counts["frames"] += 1
            f = extract(frame)
            if f is None:
                counts["unusable_frames"] += 1
                continue
            x.append(f.vector)
            y.append(meta["label"])
            groups.append(group)
            rules.append(rule_prediction(f).label)
            episodes.append(episode)
    if not x:
        raise ValueError("录制里没有可用的手部数据")
    return Dataset(
        np.asarray(x),
        np.asarray(y),
        np.asarray(groups),
        np.asarray(rules),
        np.asarray(episodes),
        counts,
    )


def grouped_split(data: Dataset, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if len(set(data.groups)) < 2:
        raise ValueError("至少需要两个独立分组；默认按参与者划分，同一人的不同帧不算独立测试")
    if not {"open", "pinch"}.issubset(set(data.y)):
        raise ValueError("训练至少需要 open 和 pinch 两类人工标注")
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=seed)
    labels = set(data.y)
    for train, test in splitter.split(data.x, data.y, groups=data.groups):
        if set(data.y[train]) == labels and set(data.y[test]) == labels:
            return train, test
    raise ValueError("无法让独立训练/测试分组都覆盖所有类别；请让每位参与者录制完整标签集")


def _score(y: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict:
    # Include every predicted class, including rule-only "other" and abstentions.
    matrix_labels = labels + sorted(set(predicted) - set(labels))
    return {
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "coverage": float(np.mean(predicted != "uncertain")),
        "classification_report": classification_report(
            y, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_labels": matrix_labels,
        "confusion_matrix": confusion_matrix(y, predicted, labels=matrix_labels).tolist(),
    }


def train(
    directory: Path,
    output: Path,
    group_by: str = "participant",
    model: str = "forest",
    seed: int = 42,
) -> dict:
    data = load_dataset(directory, group_by)
    training, testing = grouped_split(data, seed)
    if model == "forest":
        estimator = RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model == "svm":
        # Decision scores are used; avoid SVC's hidden frame-level probability CV.
        estimator = make_pipeline(StandardScaler(), SVC(C=3.0, class_weight="balanced"))
    else:
        raise ValueError("模型应为 forest 或 svm")
    estimator.fit(data.x[training], data.y[training])
    predicted, _ = predict_with_rejection(estimator, data.x[testing], REJECTION_POLICY)
    labels = sorted(set(data.y))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "feature_version": FEATURE_VERSION,
        "model": model,
        "seed": seed,
        "group_by": group_by,
        "training_groups": sorted(set(data.groups[training])),
        "testing_groups": sorted(set(data.groups[testing])),
        "training_frames": len(training),
        "testing_frames": len(testing),
        "counts": data.counts,
        "labels": labels,
        "rejection_policy": REJECTION_POLICY,
        "learned": _score(data.y[testing], predicted, labels),
        "learned_without_rejection": _score(
            data.y[testing], estimator.predict(data.x[testing]), labels
        ),
        "rules": _score(data.y[testing], data.rules[testing], labels),
        "limitations": [
            "Frame classification on held-out groups, not online event/comfort evaluation.",
            "Correlated frames are not independent participants.",
            "Test partition is not used for fitting or threshold selection. Runtime rejection policy is included in learned scores.",
            "RF vote scores and SVM margins are not calibrated probabilities. Rejection thresholds are fixed prototype settings.",
            "Rules use documented default thresholds; personalised comparison requires an additional controlled experiment.",
        ],
    }
    # The saved estimator is exactly the one evaluated, not a refit on held-out people.
    artifact = {
        "schema": "pinchpilot-model-v1",
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "estimator": estimator,
        "report": report,
    }
    joblib.dump(artifact, output / "model.joblib")
    save_json(output / "metrics.json", report)
    _plot(report, output / "confusion.png")
    return report


def _plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key in zip(axes, ("rules", "learned")):
        score = report[key]
        matrix = np.asarray(score["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        names = score["confusion_labels"]
        ax.set(
            xticks=range(len(names)),
            yticks=range(len(names)),
            xticklabels=names,
            yticklabels=names,
            xlabel="Predicted",
            ylabel="Human label",
            title=f"{key} | macro F1 {score['macro_f1']:.3f}",
        )
    fig.savefig(path, dpi=160)
    plt.close(fig)


class LearnedPredictor:
    def __init__(self, path: Path):
        artifact = joblib.load(path)
        if (
            artifact.get("schema") != "pinchpilot-model-v1"
            or artifact.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError("模型格式/特征版本不兼容")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("模型特征顺序不兼容")
        self.estimator = artifact["estimator"]
        self.report = artifact["report"]
        self.policy = self.report["rejection_policy"]
        if self.estimator.n_features_in_ != len(FEATURE_NAMES):
            raise ValueError("模型特征数错误")

    def predict(self, features: Features) -> Prediction:
        labels, scores = predict_with_rejection(self.estimator, [features.vector], self.policy)
        return Prediction(str(labels[0]), float(scores[0]), self.report["model"])
