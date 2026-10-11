"""Contact decisions for live control from a fitted rule or small CNN. NumPy only.

A fitted model is a folder with `model.json` and, for a network, `weights.npz`, written by
`save_live_model` from what `contact_cv.fit_for_live` returns. Nothing here needs PyTorch or
scikit-learn, so the desktop app can load it as it is.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import contact_rules
from .contact_data import CHANNELS, FEATURE_COUNT, FEATURE_VERSION, WindowBuilder, WindowConfig

SCHEMA = "pinchpilot-live-contacts-v1"
COORDINATES = 63
WEIGHTS = ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "head.weight", "head.bias")
FLAGS = ("flat_middle", "flat_index", "flat_ring", "nearest")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_live_model(record, arrays, directory):
    """Write a fitted model to a new or empty folder and return the folder."""
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("输出目录已有内容；请换一个新目录")
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    if arrays is not None:
        np.savez(directory / "weights.npz", **arrays)
        record["weights_sha256"] = _sha(directory / "weights.npz")
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    (directory / "model.json").write_text(text, encoding="utf-8")
    load_live_model(directory)  # What was written must be something the app accepts.
    return directory


class LiveModel:
    """A fitted rule or temporal CNN that decides windows of contact features."""

    def __init__(self, record, arrays=None):
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError(f"模型文件格式无效：schema 应为 {SCHEMA}")
        if (
            record.get("feature_version") != FEATURE_VERSION
            or record.get("channels") != list(CHANNELS)
            or record.get("threshold") != 0.5
        ):
            raise ValueError("模型的特征版本、通道或判定门槛与这个程序不一致")
        try:
            self.config = WindowConfig(**record["window_config"])
            options = record["options"]
            frames = options["frames"]
        except (KeyError, TypeError) as error:
            raise ValueError("模型文件缺少窗口设置或模型设置") from error
        if type(frames) is not int or not 1 <= frames <= self.config.frames:
            raise ValueError("模型用的帧数超出了窗口长度")
        self.record, self.options, self.kind = record, options, record.get("model")
        if self.kind == "rules":
            self._rule(record)
        elif self.kind == "cnn":
            self._network(arrays)
        elif self.kind == "finger_cnn":
            self._fingers(arrays)
        else:
            raise ValueError("实时控制目前只支持调过阈值的规则、时序 CNN 和手指共享 CNN")

    def _rule(self, record):
        cuts = record.get("thresholds")
        if (
            not isinstance(cuts, list)
            or len(cuts) != 3
            or any(type(cut) not in (int, float) or not math.isfinite(cut) for cut in cuts)
        ):
            raise ValueError("规则需要三个有限的阈值")
        if self.options.get("statistic") not in contact_rules.STATISTICS or any(
            type(self.options.get(flag, False)) is not bool for flag in FLAGS
        ):
            raise ValueError("规则的设置项无效")
        self.thresholds = np.array(cuts, dtype=np.float64)

    @staticmethod
    def _checked(arrays, names, shapes):
        """Weights with exactly these names, finite, in these shapes, and a usable scale."""
        if not isinstance(arrays, dict) or set(arrays) != set(names):
            raise ValueError("权重文件缺少字段或有多余字段")
        if any(
            not isinstance(value, np.ndarray)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
            for value in arrays.values()
        ):
            raise ValueError("权重里有无效数值")
        if any(arrays[name].shape != shape for name, shape in shapes.items()):
            raise ValueError("权重形状与设置不符")
        if (arrays["scale"] <= 0).any():
            raise ValueError("归一化参数无效")

    @staticmethod
    def _layers(arrays, prefix):
        """Two causal convolutions as the matrices their last output needs."""
        w1, w2 = arrays[f"{prefix}conv1.weight"], arrays[f"{prefix}conv2.weight"]
        return (
            w1.reshape(len(w1), -1).T.astype(np.float32),
            arrays[f"{prefix}conv1.bias"],
            w2.reshape(len(w2), -1).T.astype(np.float32),
            arrays[f"{prefix}conv2.bias"],
        )

    def _last_vector(self, x, layers):
        """What the two convolutions give at the last frame; x is normalized [N, T, F]."""
        first, first_bias, second, second_bias = layers
        count, frames, width = x.shape
        missing = self.kernel + 2 - frames
        if missing > 0:  # Zeros before the oldest frame, as in training.
            x = np.pad(x, ((0, 0), (missing, 0), (0, 0)))
        patches = np.stack([x[:, i : i + 3] for i in range(self.kernel)], axis=1)
        patches = patches.transpose(0, 1, 3, 2).reshape(count, self.kernel, width * 3)
        hidden = np.maximum(patches @ first + first_bias, 0)
        hidden = hidden.transpose(0, 2, 1).reshape(count, -1)
        return np.maximum(hidden @ second + second_bias, 0)

    def _fingers(self, arrays):
        shared, context = self.options.get("shared"), self.options.get("context")
        if type(shared) is not bool or type(context) is not bool:
            raise ValueError("手指共享 CNN 的 shared、context 设置无效")
        copies = 1 if shared else 3
        parts = ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias")
        names = [f"branches.{i}.{part}" for i in range(copies) for part in parts]
        names += [f"heads.{i}.{part}" for i in range(copies) for part in ("weight", "bias")]
        first = arrays.get("branches.0.conv1.weight") if isinstance(arrays, dict) else None
        channels = first.shape[0] if isinstance(first, np.ndarray) and first.ndim == 3 else 0
        frames, width = self.options["frames"], contact_rules.FINGER_FEATURES
        self.kernel = max(1, frames - 2)
        shapes = {"bias": (3,), "mean": (width,), "scale": (width,)}
        for i in range(copies):
            shapes[f"branches.{i}.conv1.weight"] = (channels, width, 3)
            shapes[f"branches.{i}.conv1.bias"] = (channels,)
            shapes[f"branches.{i}.conv2.weight"] = (channels, channels, self.kernel)
            shapes[f"branches.{i}.conv2.bias"] = (channels,)
            shapes[f"heads.{i}.weight"] = (1, channels * 2 if context else channels)
            shapes[f"heads.{i}.bias"] = (1,)
        self._checked(arrays, [*names, "bias", "mean", "scale"], shapes)
        self.mean = arrays["mean"].astype(np.float64)
        self.scale = arrays["scale"].astype(np.float64)
        self.branches = [self._layers(arrays, f"branches.{i}.") for i in range(copies)]
        self.heads = [
            (arrays[f"heads.{i}.weight"][0].astype(np.float32), arrays[f"heads.{i}.bias"][0])
            for i in range(copies)
        ]
        self.bias, self.context = arrays["bias"], context

    def _network(self, arrays):
        if self.options.get("inputs") not in ("all", "distances"):
            raise ValueError("时序 CNN 的 inputs 设置无效")
        self.first = COORDINATES if self.options["inputs"] == "distances" else 0
        width, frames = FEATURE_COUNT - self.first, self.options["frames"]
        first = arrays.get("conv1.weight") if isinstance(arrays, dict) else None
        channels = first.shape[0] if isinstance(first, np.ndarray) and first.ndim == 3 else 0
        shapes = {
            "conv1.weight": (channels, width, 3),
            "conv1.bias": (channels,),
            "conv2.weight": (channels, channels, max(1, frames - 2)),
            "conv2.bias": (channels,),
            "head.weight": (3, channels),
            "head.bias": (3,),
            "mean": (width,),
            "scale": (width,),
        }
        self._checked(arrays, [*WEIGHTS, "mean", "scale"], shapes)
        self.mean = arrays["mean"].astype(np.float64)
        self.scale = arrays["scale"].astype(np.float64)
        self.layers = self._layers(arrays, "")
        self.head = (arrays["head.weight"].T.astype(np.float32), arrays["head.bias"])
        self.kernel = max(1, frames - 2)

    def scores(self, windows):
        """[N, 3] contact scores in [0, 1] for [N, T, 67] windows; a rule gives 0 or 1."""
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (self.config.frames, FEATURE_COUNT):
            raise ValueError("窗口形状应为 [N, 帧数, 67]")
        if self.kind == "rules":
            touching = contact_rules.decide(windows, self.options, self.thresholds)
            return touching.astype(np.float32)
        recent = windows[:, -self.options["frames"] :]
        if self.kind == "finger_cnn":
            logits = self._finger_logits(recent)
        else:
            x = recent[:, :, self.first :]
            x = ((x.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
            logits = self._last_vector(x, self.layers) @ self.head[0] + self.head[1]
        return (1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.float32)

    def _finger_logits(self, recent):
        """One small network per finger over thumb and that finger; see contact_nets."""
        views = contact_rules.finger_inputs(recent)
        views = ((views.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        vectors = np.stack(
            [
                self._last_vector(views[:, i], self.branches[i % len(self.branches)])
                for i in range(3)
            ],
            axis=1,
        )
        if self.context:  # Each finger also sees the larger of the other two's values.
            a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
            others = np.stack((np.maximum(b, c), np.maximum(a, c), np.maximum(a, b)), axis=1)
            vectors = np.concatenate((vectors, others), axis=2)
        logits = [
            vectors[:, i] @ self.heads[i % len(self.heads)][0] + self.heads[i % len(self.heads)][1]
            for i in range(3)
        ]
        return np.stack(logits, axis=1) + self.bias

    def decide(self, windows):
        """[N, 3] booleans in the order of CHANNELS: touching or not."""
        return self.scores(windows) >= 0.5


def load_live_model(directory):
    """The fitted model in this folder, checked against its recorded digest and shapes."""
    directory = Path(directory)
    try:
        record = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError(f"读不到模型文件：{directory / 'model.json'}") from error
    arrays = None
    if isinstance(record, dict) and record.get("model") in ("cnn", "finger_cnn"):
        weights = directory / "weights.npz"
        if not weights.is_file() or _sha(weights) != record.get("weights_sha256"):
            raise ValueError("权重文件缺失，或与模型文件里记录的校验和不一致")
        try:
            with np.load(weights, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
        except (OSError, ValueError) as error:
            raise ValueError("权重文件无法读取") from error
    return LiveModel(record, arrays)


KINDS = {"rules": "调过的规则", "cnn": "时序 CNN", "finger_cnn": "手指共享 CNN"}
SUMMARY = (
    "model",
    "candidate",
    "trained_on",
    "excluded",
    "training_windows",
    "pool_fingerprint",
    "source",
    "labels",
    "evidence",
    "app_version",
    "weights_sha256",
)


def describe(model):
    """What a record of use should say about a loaded model: which one, fitted on whom."""
    return {key: model.record[key] for key in SUMMARY if key in model.record}


def label(model):
    """One line for a person choosing between models."""
    record = model.record
    people = record.get("trained_on") or []
    if record.get("excluded"):
        who = f"训练时不含 {record['excluded']}"
    elif people:
        who = f"用 {'、'.join(people)} 训练"
    else:
        who = "训练者未记录"
    name = f" · {record['candidate']}" if record.get("candidate") else ""
    return f"{KINDS[model.kind]} · {who}{name}"


def find_models(root):
    """Every loadable model one level below `root`: ([(label, folder, model)], [problems])."""
    root = Path(root)
    found, problems = [], []
    if not root.is_dir():
        return found, problems
    for folder in sorted(path for path in root.iterdir() if (path / "model.json").is_file()):
        try:
            model = load_live_model(folder)
        except ValueError as error:
            problems.append(f"{folder.name}：{error}")
            continue
        found.append((label(model), str(folder), model))
    return sorted(found, key=lambda item: item[0]), problems


class LiveContacts:
    """Feeds camera frames to a live model: one decision per frame once a window is full.

    `push` returns three booleans in the order of CHANNELS, or None while there is no
    complete window (no hand, the first frames after a gap, a changed hand). `reason` says
    which, in the words of WindowBuilder.
    """

    def __init__(self, model):
        self.model = model
        self.builder = WindowBuilder(model.config)

    @property
    def reason(self):
        return self.builder.reason

    def reset(self):
        self.builder.reset()

    def push(self, frame):
        window = self.builder.push(frame)
        if window is None:
            return None
        return self.model.decide(window[None])[0]
