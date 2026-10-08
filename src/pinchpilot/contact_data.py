"""Versioned contact labels and causal windows. No camera or native input access."""

import hashlib
import json
import math
import uuid
from collections import deque
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

from .domain import HandFrame
from .storage import read_recording, save_json
from .tripod import TripodConfig, tripod_features

CHANNELS = ("middle", "index", "ring")
FEATURE_VERSION = "contact-v1"
FEATURE_COUNT = 67
RECORDING_SCHEMA = "pinchpilot-contact-recording-v1"
ANNOTATION_SCHEMA = "pinchpilot-contact-annotations-v1"
MANIFEST_SCHEMA = "pinchpilot-contact-dataset-v1"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class WindowConfig:
    frames: int = 8
    max_gap_seconds: float = 0.12
    max_window_seconds: float = 0.8

    def __post_init__(self):
        if type(self.frames) is not int or not 2 <= self.frames <= 64:
            raise ValueError("窗口帧数必须在2到64之间")
        for value in (self.max_gap_seconds, self.max_window_seconds):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("窗口时间参数无效")
        if (
            not 0.02 <= self.max_gap_seconds <= 0.5
            or not self.max_gap_seconds <= self.max_window_seconds <= 3
        ):
            raise ValueError("窗口时间范围无效")


def _validate_frame(frame):
    if not isinstance(frame, HandFrame):
        raise ValueError("录制帧类型无效")
    for value in (frame.timestamp, frame.aspect, frame.handedness_score):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("帧包含无效数值")
    if (
        frame.aspect <= 0
        or not 0 <= frame.handedness_score <= 1
        or frame.handedness not in ("", "Left", "Right")
    ):
        raise ValueError("帧比例或左右手分类无效")
    if frame.landmarks:
        points = np.asarray(frame.landmarks, dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("关键点必须是有限的21×3坐标")
    if frame.world_landmarks:
        world = np.asarray(frame.world_landmarks, dtype=float)
        if not frame.landmarks or world.shape != (21, 3) or not np.isfinite(world).all():
            raise ValueError("世界坐标关键点必须随关键点一起出现，且为有限的21×3坐标")


def contact_features(frame, dt=0.0):
    """Palm-normalized estimated XYZ + 3 distance ratios + actual inter-frame dt."""
    _validate_frame(frame)
    if not frame.landmarks:
        return None
    # The first experiment compares both models and the existing geometry on
    # the same usable frames. Keep this quality domain identical in shadow.
    if tripod_features(frame, spatial_grip=True) is None:
        return None
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("帧间隔无效")
    points = np.asarray(frame.landmarks, dtype=float) * [frame.aspect, 1.0, frame.aspect]
    scale = (np.linalg.norm(points[5] - points[17]) + np.linalg.norm(points[0] - points[9])) / 2
    if scale < 0.025:
        return None
    local = (points - points[[0, 5, 9, 13, 17]].mean(axis=0)) / scale
    distances = [np.linalg.norm(points[4] - points[i]) / scale for i in (12, 8, 16)]
    vector = np.concatenate((local.ravel(), distances, [dt])).astype(np.float32)
    return vector if np.isfinite(vector).all() else None


def rule_contacts(frame):
    """Instantaneous thresholds only; deliberately not the interactive state machine."""
    f = tripod_features(frame, spatial_grip=True)
    if f is None:
        return None
    cfg = TripodConfig()
    projected = (
        math.hypot(
            (frame.landmarks[4][0] - frame.landmarks[12][0]) * frame.aspect,
            frame.landmarks[4][1] - frame.landmarks[12][1],
        )
        / f.scale
    )
    return np.array(
        [
            f.grip <= cfg.grip_engage and projected <= cfg.grip_release,
            f.contact <= cfg.touch_ratio,
            f.right_contact <= cfg.right_touch_ratio,
        ],
        dtype=np.int64,
    )


class WindowBuilder:
    def __init__(self, config=None):
        self.config = config or WindowConfig()
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.config.frames)
        self.last_time = None
        self.last_hand = ""
        self.reason = "reset"

    def push(self, frame):
        try:
            _validate_frame(frame)
            vector = contact_features(frame)
        except (ValueError, TypeError, OverflowError):
            self.reset()
            self.reason = "invalid_frame"
            return None
        if vector is None:
            self.reset()
            self.reason = "missing_hand" if not frame.landmarks else "invalid_frame"
            return None
        reason = "warming_up"
        if self.last_time is not None:
            dt = frame.timestamp - self.last_time
            if dt <= 0:
                self.reset()
                self.reason = "nonmonotonic_time"
                return None
            if dt > self.config.max_gap_seconds:
                self.reset()
                reason = "time_gap"
            elif frame.handedness != self.last_hand:
                self.reset()
                reason = "hand_changed"
        vector[-1] = frame.timestamp - self.last_time if self.last_time is not None else 0.0
        self.last_time, self.last_hand = frame.timestamp, frame.handedness
        self.history.append((frame.timestamp, vector))
        while (
            self.history and frame.timestamp - self.history[0][0] > self.config.max_window_seconds
        ):
            self.history.popleft()
            reason = "window_too_long"
        self.reason = reason
        if len(self.history) < self.config.frames:
            return None
        self.reason = "ready"
        return np.stack([row[1] for row in self.history])


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"JSON格式无效：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContactRecorder:
    """Explicit caller-owned recording, with no automatic labels or hardware access."""

    def __init__(self, path, participant, session, *, source="camera", episode=None):
        if source not in ("camera", "synthetic") or not all(
            isinstance(v, str) and v.strip() for v in (participant, session)
        ):
            raise ValueError("需要参与者编号、场次和有效来源")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        self.last_time = None
        self.count = 0
        self._write(
            {
                "type": "metadata",
                "schema": RECORDING_SCHEMA,
                "participant": participant.strip(),
                "session": session.strip(),
                "episode": episode.strip()
                if isinstance(episode, str) and episode.strip()
                else uuid.uuid4().hex,
                "source": source,
                "labels_inferred": False,
                "stores_images": False,
                "stores_landmarks": True,
            }
        )

    def _write(self, row):
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def add(self, frame):
        _validate_frame(frame)
        if self.last_time is not None and frame.timestamp <= self.last_time:
            raise ValueError("录制时间戳必须递增")
        self._write({"type": "frame", **asdict(frame)})
        self.last_time = frame.timestamp
        self.count += 1
        if self.count % 30 == 0:
            self.file.flush()

    def close(self):
        self.file.close()


def read_contact_recording(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            meta = json.loads(next(stream))
            if not isinstance(meta, dict):
                raise ValueError("文件头无效")
            if meta.get("schema") != RECORDING_SCHEMA:
                legacy, frames = read_recording(path)
                if not frames:
                    raise ValueError("录制文件没有帧")
                for frame in frames:
                    _validate_frame(frame)
                for key in ("participant", "session", "episode"):
                    if not isinstance(legacy.get(key), str) or not legacy[key].strip():
                        raise ValueError("文件头身份无效")
                    legacy[key] = legacy[key].strip()
                # Legacy format never certified a physical contact label or source.
                # Human annotations must explicitly attest source and review before training.
                return {**legacy, "source": "legacy_unspecified"}, frames
            if meta.get("source") not in ("camera", "synthetic") or not all(
                isinstance(meta.get(k), str) and meta[k].strip()
                for k in ("participant", "session", "episode")
            ):
                raise ValueError("文件头身份或来源无效")
            for key in ("participant", "session", "episode"):
                meta[key] = meta[key].strip()
            frames = []
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.pop("type") != "frame":
                    raise ValueError("记录类型无效")
                row["landmarks"] = tuple(tuple(point) for point in row["landmarks"])
                row["world_landmarks"] = tuple(
                    tuple(point) for point in row.get("world_landmarks", ())
                )
                frame = HandFrame(**row)
                _validate_frame(frame)
                if frames and frame.timestamp <= frames[-1].timestamp:
                    raise ValueError("时间戳不递增")
                frames.append(frame)
    except (StopIteration, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"录制文件无效：{path.name}: {error}") from error
    if not frames:
        raise ValueError("录制文件没有帧")
    return meta, frames


def annotation_template(recording, output):
    meta, frames = read_contact_recording(recording)
    path = Path(output)
    if path.exists():
        raise ValueError("标注模板已存在，不能覆盖已做的标注")
    result = {
        "schema": ANNOTATION_SCHEMA,
        "recording_sha256": _sha(recording),
        "recording_source": meta["source"],
        "reviewed": False,
        "label_source": "unreviewed",
        "annotator": "",
        "channels": list(CHANNELS),
        "interval_units": "zero_based_frame_index_half_open",
        "intervals": [{"start": 0, "end": len(frames), **{key: None for key in CHANNELS}}],
    }
    # Exclusive creation avoids save_json's predictable .tmp sibling colliding
    # with a caller's input recording or an unrelated existing file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def draft_annotation(recording, output, intervals, protocol):
    """Write unreviewed labels taken from a guided session's prompts and Space key.

    The file says so (`reviewed` false, `label_source` "protocol_draft"), and training
    refuses it until a person has reviewed it.
    """
    meta, frames = read_contact_recording(recording)
    path = Path(output)
    if path.exists():
        raise ValueError("标注文件已存在，不能覆盖已做的标注")
    last_end = 0
    for interval in intervals:
        start, end = interval.get("start"), interval.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not last_end <= start < end <= len(frames)
            or any(interval.get(channel) not in (0, 1, None) for channel in CHANNELS)
        ):
            raise ValueError("草稿标注区间无效")
        last_end = end
    result = {
        "schema": ANNOTATION_SCHEMA,
        "recording_sha256": _sha(recording),
        "recording_source": meta["source"],
        "reviewed": False,
        "label_source": "protocol_draft",
        "protocol": protocol,
        "annotator": "",
        "channels": list(CHANNELS),
        "interval_units": "zero_based_frame_index_half_open",
        "intervals": list(intervals),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def is_protocol_draft(spec):
    """Unreviewed labels that a guided session wrote from its prompts and the Space key."""
    return spec.get("reviewed") is False and spec.get("label_source") == "protocol_draft"


def _labels(recording, annotation, frames, source, allow_draft=False):
    """Per-frame labels, -1 where unknown.

    `allow_draft` also accepts a guided session's unreviewed draft. Only trial runs may ask
    for that; whatever they produce is not a result.
    """
    spec = _read_json(annotation)
    if spec.get("schema") != ANNOTATION_SCHEMA or spec.get("channels") != list(CHANNELS):
        raise ValueError("三指标注格式或通道不匹配")
    if spec.get("recording_sha256") != _sha(recording):
        raise ValueError("录制文件SHA256已改变，不能沿用旧标注")
    draft = allow_draft and is_protocol_draft(spec)
    if spec.get("reviewed") is not True and not draft:
        raise ValueError("需要已复核的接触标注；提示任务和规则输出不是真值")
    if source == "legacy_unspecified":
        source = spec.get("recording_source")
    if source not in ("camera", "synthetic") or spec.get("recording_source") != source:
        raise ValueError("标注与录制来源不匹配")
    expected = "synthetic_fixture" if source == "synthetic" else "human_reviewed"
    if not draft and (
        spec.get("label_source") != expected
        or not isinstance(spec.get("annotator"), str)
        or not spec["annotator"].strip()
    ):
        raise ValueError("标签来源或复核者无效；合成标签不能冒充真人标签")
    if spec.get("interval_units") != "zero_based_frame_index_half_open" or not isinstance(
        spec.get("intervals"), list
    ):
        raise ValueError("需要半开帧区间[start,end)标注")
    labels = np.full((len(frames), len(CHANNELS)), -1, dtype=np.int64)
    last_end = 0
    for interval in spec["intervals"]:
        if not isinstance(interval, dict):
            raise ValueError("标注区间必须是对象")
        start, end = interval.get("start"), interval.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not last_end <= start < end <= len(frames)
        ):
            raise ValueError("标注区间越界、重叠或未排序")
        for index, channel in enumerate(CHANNELS):
            value = interval.get(channel)
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError("接触标签只能是0、1或null")
            if value is not None:
                labels[start:end, index] = value
        last_end = end
    return labels, source


@dataclass
class ContactPartition:
    x: np.ndarray
    y: np.ndarray
    rules: np.ndarray
    groups: np.ndarray
    episodes: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray


@dataclass
class ContactDataset:
    partitions: dict[str, ContactPartition]
    metadata: dict
    window_config: WindowConfig


def load_contact_dataset(manifest, config=None):
    manifest = Path(manifest)
    spec = _read_json(manifest)
    if spec.get("schema") != MANIFEST_SCHEMA or spec.get("group_by") not in (
        "participant",
        "session",
    ):
        raise ValueError("数据清单格式或分组方式无效")
    entries = spec.get("recordings")
    if not isinstance(entries, list) or not entries:
        raise ValueError("数据清单需要recordings列表")
    config = config or WindowConfig()
    groups, episodes, contents, sources, prepared, fingerprints = {}, set(), set(), set(), [], []
    # Validate all partitions before deriving a single window.
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("split") not in SPLITS:
            raise ValueError("每段录制需要train/validation/test分区")
        try:
            recording = (manifest.parent / entry["recording"]).resolve()
            annotation = (manifest.parent / entry["annotation"]).resolve()
        except (KeyError, TypeError) as error:
            raise ValueError("录制或标注路径缺失") from error
        meta, frames = read_contact_recording(recording)
        labels, source = _labels(recording, annotation, frames, meta["source"])
        sources.add(source)
        if len(sources) > 1:
            raise ValueError("不能把合成与真人录制混入同一实验")
        group = (
            meta["participant"]
            if spec["group_by"] == "participant"
            else json.dumps([meta["participant"], meta["session"]])
        )
        split = entry["split"]
        if group in groups and groups[group] != split:
            raise ValueError("分组泄漏：同一参与者/场次跨分区")
        groups[group] = split
        rows = [
            {**asdict(f), "timestamp": round(f.timestamp - frames[0].timestamp, 9)} for f in frames
        ]
        content = hashlib.sha256(
            json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if meta["episode"] in episodes or content in contents:
            raise ValueError("重复录制/episode；复制或修改起始时间不能创建独立样本")
        episodes.add(meta["episode"])
        contents.add(content)
        fingerprints.append(
            {
                "recording_sha256": _sha(recording),
                "annotation_sha256": _sha(annotation),
                "split": split,
                "group": group,
            }
        )
        prepared.append((split, group, meta["episode"], frames, labels))
    columns = {
        split: {
            key: []
            for key in ("x", "y", "rules", "groups", "episodes", "frame_indices", "timestamps")
        }
        for split in SPLITS
    }
    skipped = {"unknown_label": 0, "warming_or_invalid": 0}
    for split, group, episode, frames, labels in prepared:
        builder = WindowBuilder(config)
        for index, frame in enumerate(frames):
            window = builder.push(frame)
            if window is None:
                skipped["warming_or_invalid"] += 1
                continue
            if (labels[index] < 0).any():
                skipped["unknown_label"] += 1
                continue
            rules = rule_contacts(frame)
            if rules is None:
                skipped["warming_or_invalid"] += 1
                continue
            row = columns[split]
            for key, value in (
                ("x", window),
                ("y", labels[index]),
                ("rules", rules),
                ("groups", group),
                ("episodes", episode),
                ("frame_indices", index),
                ("timestamps", frame.timestamp),
            ):
                row[key].append(value)
    partitions = {}
    for split, cols in columns.items():
        n = len(cols["x"])
        partitions[split] = ContactPartition(
            np.asarray(cols["x"], dtype=np.float32).reshape(n, config.frames, FEATURE_COUNT),
            np.asarray(cols["y"], dtype=np.int64).reshape(n, 3),
            np.asarray(cols["rules"], dtype=np.int64).reshape(n, 3),
            np.asarray(cols["groups"], dtype=str),
            np.asarray(cols["episodes"], dtype=str),
            np.asarray(cols["frame_indices"], dtype=np.int64),
            np.asarray(cols["timestamps"], dtype=float),
        )
    metadata = {
        "source": next(iter(sources)),
        "group_by": spec["group_by"],
        "feature_version": FEATURE_VERSION,
        "window_config": asdict(config),
        "groups": {s: sorted(k for k, v in groups.items() if v == s) for s in SPLITS},
        "files": fingerprints,
        "skipped": skipped,
        "recordings": len(prepared),
        "fingerprint": hashlib.sha256(
            json.dumps(fingerprints, sort_keys=True).encode()
        ).hexdigest(),
    }
    return ContactDataset(partitions, metadata, config)


def create_synthetic_contact_dataset(directory, seed=20260922):
    """Engineering fixture labels come from generator parameters, never human evidence."""
    from .tripod_demo import synthetic_tripod

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed需要为[0,2**32)整数")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("合成样例需要空目录，以免覆盖数据")
    rng = np.random.default_rng(seed)
    manifest = {"schema": MANIFEST_SCHEMA, "group_by": "participant", "recordings": []}
    for subject, split in enumerate(("train", "train", "train", "validation", "test")):
        name = f"synthetic_P{subject + 1}"
        path = directory / f"{name}.jsonl"
        recorder = ContactRecorder(
            path,
            name,
            "synthetic-session",
            source="synthetic",
            episode=f"synthetic-{seed}-{subject}",
        )
        intervals = []
        frame_index = 0
        try:
            for flags in product((0, 1), repeat=3):
                start = frame_index
                for _ in range(30):
                    stamp = 100 + frame_index / 30
                    frame = synthetic_tripod(
                        stamp,
                        grip=bool(flags[0]),
                        contact=0.10 if flags[1] else 0.85,
                        right_contact=0.10 if flags[2] else 0.85,
                        noise=tuple(rng.normal(0, 0.0007, 2)),
                    )
                    points = np.asarray(frame.landmarks)
                    points[:, :2] = (points[:, :2] - 0.5) * (0.88 + subject * 0.055) + 0.5
                    recorder.add(replace(frame, landmarks=tuple(map(tuple, points))))
                    frame_index += 1
                intervals.append({"start": start, "end": frame_index, **dict(zip(CHANNELS, flags))})
        finally:
            recorder.close()
        labels = directory / f"{name}.labels.json"
        annotation_template(path, labels)
        annotated = _read_json(labels)
        annotated.update(
            reviewed=True,
            label_source="synthetic_fixture",
            annotator="deterministic-generator",
            intervals=intervals,
        )
        save_json(labels, annotated)
        manifest["recordings"].append(
            {"split": split, "recording": path.name, "annotation": labels.name}
        )
    target = directory / "manifest.json"
    save_json(target, manifest)
    return target


def create_synthetic_contact_pool(
    directory, participants=3, sessions=2, seed=20260922, segment_frames=24
):
    """Several synthetic people with several sessions each, laid out like team recordings.

    For checking the cross-validation code only. Labels come from generator parameters.
    """
    from .tripod_demo import synthetic_tripod

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed需要为[0,2**32)整数")
    if not 1 <= participants <= 20 or not 1 <= sessions <= 10:
        raise ValueError("合成样例的人数或场次数无效")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("合成样例需要空目录，以免覆盖数据")
    rng = np.random.default_rng(seed)
    for person in range(1, participants + 1):
        name = f"synthetic_P{person:02d}"
        for session in range(1, sessions + 1):
            path = directory / name / f"{name}_s{session}.jsonl"
            recorder = ContactRecorder(
                path,
                name,
                f"synthetic-session-{session}",
                source="synthetic",
                episode=f"synthetic-pool-{seed}-{person}-{session}",
            )
            intervals, steps, index = [], [], 0
            try:
                for round_number in (1, 2):
                    for flags in product((0, 1), repeat=3):
                        start = index
                        # Each segment has its own gap sizes, so a threshold has to be learned.
                        contact = rng.uniform(0.06, 0.16) if flags[1] else rng.uniform(0.45, 0.9)
                        right = rng.uniform(0.06, 0.16) if flags[2] else rng.uniform(0.45, 0.9)
                        for _ in range(segment_frames):
                            frame = synthetic_tripod(
                                100 + index / 30,
                                grip=bool(flags[0]),
                                contact=contact,
                                right_contact=right,
                                noise=tuple(rng.normal(0, 0.0007, 2)),
                            )
                            points = np.asarray(frame.landmarks)
                            points[:, :2] = (points[:, :2] - 0.5) * (0.85 + person * 0.05) + 0.5
                            recorder.add(replace(frame, landmarks=tuple(map(tuple, points))))
                            index += 1
                        intervals.append(
                            {"start": start, "end": index, **dict(zip(CHANNELS, flags))}
                        )
                        steps.append(
                            {
                                "start": start / 30,
                                "end": index / 30,
                                "round": round_number,
                                "name": "synthetic_" + "".join(map(str, flags)),
                            }
                        )
            finally:
                recorder.close()
            labels = path.with_suffix(".labels.json")
            annotation_template(path, labels)
            annotated = _read_json(labels)
            annotated.update(
                reviewed=True,
                label_source="synthetic_fixture",
                annotator="deterministic-generator",
                intervals=intervals,
            )
            save_json(labels, annotated)
            save_json(
                path.with_suffix(".protocol.json"),
                {
                    "schema": "pinchpilot-contact-protocol-v1",
                    "protocol": "synthetic",
                    "clock_start": 100.0,
                    "steps": steps,
                },
            )
    return directory
