"""Every labelled contact recording as one pool of causal windows, for cross-validation.

Reads the zips that guided sessions produce and loose recording files. No camera, training
or native input access.
"""

import hashlib
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, contact_protocol
from .contact_data import (
    CHANNELS,
    FEATURE_COUNT,
    FEATURE_VERSION,
    WindowBuilder,
    WindowConfig,
    _labels,
    _read_json,
    _sha,
    is_protocol_draft,
    read_contact_recording,
    rule_contacts,
)

POOL_SCHEMA = "pinchpilot-contact-pool-v2"
PROTOCOL_SCHEMA = "pinchpilot-contact-protocol-v1"
MEMBER_SUFFIXES = (".jsonl", ".labels.json", ".protocol.json")


@dataclass
class ContactPool:
    """Windows are kept as one feature row per frame plus the row each window ends on."""

    features: np.ndarray  # [rows, 67], frames in recording order
    last: np.ndarray  # [N] row of each window's final frame
    y: np.ndarray  # [N, 3]
    rules: np.ndarray  # [N, 3] the product's fixed thresholds, for reference
    participants: np.ndarray  # [N] str
    recordings: np.ndarray  # [N] index into metadata["recordings"]
    rounds: np.ndarray  # [N] round of the guided session, 0 when unknown
    steps: np.ndarray  # [N] index into metadata["step_names"], -1 when unknown
    metadata: dict
    frames: int
    # Complete windows whose final frame has no usable label: keypoints without an answer.
    spare_last: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    spare_participants: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))

    def __len__(self):
        return len(self.last)

    def gather(self, rows):
        """[len(rows), frames, 67] windows ending on these feature rows, oldest frame first."""
        rows = np.asarray(rows, dtype=np.int64)
        return self.features[rows[:, None] + np.arange(1 - self.frames, 1)]

    def windows(self, indices):
        """The labelled windows with these indices."""
        return self.gather(self.last[np.asarray(indices, dtype=np.int64)])


def _member(archive, suffix):
    names = [
        info.filename
        for info in archive.infolist()
        if info.filename.endswith(suffix) and "/" not in info.filename and "\\" not in info.filename
    ]
    if len(names) > 1:
        raise ValueError(f"压缩包里有多个 {suffix} 文件")
    return names[0] if names else None


def _entries(source):
    """One entry per recording file found: where its recording, labels and protocol are."""
    source = Path(source)
    if not source.is_dir():
        raise ValueError(f"找不到数据文件夹：{source}")
    entries = []
    for path in sorted(source.rglob("*")):
        hidden = any(part.startswith(".") for part in path.relative_to(source).parts)
        if hidden or not path.is_file():
            continue
        if path.suffix == ".zip":
            reviewed = path.with_name(f"{path.stem}.labels.json")
            entries.append({"zip": path, "labels": reviewed if reviewed.is_file() else None})
        elif path.suffix == ".jsonl":
            labels = path.with_suffix(".labels.json")
            protocol = path.with_suffix(".protocol.json")
            if labels.is_file():
                entries.append(
                    {
                        "recording": path,
                        "labels": labels,
                        "protocol": protocol if protocol.is_file() else None,
                    }
                )
    if not entries:
        raise ValueError(f"文件夹里没有带标签的录制：{source}")
    return entries


def _key(entries, source, config, allow_draft):
    """Changes whenever any input file, the window settings or the draft switch changes."""
    files = []
    for entry in entries:
        for path in entry.values():
            if path is not None:
                files.append([path.relative_to(source).as_posix(), _sha(path)])
    payload = {
        "schema": POOL_SCHEMA,
        "app_version": __version__,
        "feature_version": FEATURE_VERSION,
        "window_config": asdict(config),
        "allow_draft": allow_draft,
        "files": sorted(files),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _unpack(entry, directory):
    """Paths of the recording, labels and protocol of an entry, extracting a zip if needed."""
    if "zip" not in entry:
        return entry["recording"], entry["labels"], entry["protocol"]
    paths = {}
    try:
        with zipfile.ZipFile(entry["zip"]) as archive:
            for suffix in MEMBER_SUFFIXES:
                name = _member(archive, suffix)
                if name is not None:
                    paths[suffix] = Path(directory) / f"recording{suffix}"
                    paths[suffix].write_bytes(archive.read(name))
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError(f"压缩包无法读取：{entry['zip'].name}: {error}") from error
    if ".jsonl" not in paths or (entry["labels"] is None and ".labels.json" not in paths):
        raise ValueError(f"压缩包里缺少录制或标签：{entry['zip'].name}")
    return paths[".jsonl"], entry["labels"] or paths[".labels.json"], paths.get(".protocol.json")


def _schedule(protocol):
    """(clock start, step starts, ends, rounds, names, protocol name) of a guided session."""
    if protocol is None:
        return None
    spec = _read_json(protocol)
    steps = spec.get("steps")
    try:
        if spec.get("schema") != PROTOCOL_SCHEMA or not steps:
            raise ValueError
        return (
            float(spec["clock_start"]),
            np.array([float(step["start"]) for step in steps]),
            np.array([float(step["end"]) for step in steps]),
            [int(step["round"]) for step in steps],
            [str(step["name"]) for step in steps],
            str(spec["protocol"]),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"提示记录格式无效：{Path(protocol).name}") from error


def _short_margins(label_spec):
    """Whether a review kept less than the default unlabelled time around changes."""
    margins = (label_spec.get("review") or {}).get("margins") or {}
    try:
        return (
            margins["settle"] < contact_protocol.SETTLE
            or margins["lead"] < contact_protocol.LEAD
            or margins["edge"] < contact_protocol.KEY_EDGE
        )
    except (KeyError, TypeError):
        return False


def _build(entries, source, config, allow_draft):
    columns = {key: [] for key in ("last", "y", "rules", "participants", "recordings")}
    columns.update(rounds=[], steps=[], spare_last=[], spare_participants=[])
    features, recordings, step_names = [], [], []
    seen, episodes, contents, sources, unreviewed = {}, set(), set(), set(), []
    skipped = {"unknown_label": 0, "warming_or_invalid": 0}
    for entry in entries:
        origin = entry.get("zip") or entry["recording"]
        name = origin.relative_to(source).as_posix()
        with tempfile.TemporaryDirectory() as scratch:
            recording, labels_path, protocol = _unpack(entry, scratch)
            digest = _sha(recording)
            label_spec = _read_json(labels_path)
            draft = is_protocol_draft(label_spec)
            if digest in seen:
                # The same recording as a zip and as loose files: keep reviewed labels if any.
                if not draft and recordings[seen[digest]]["labels"] == "draft":
                    raise ValueError(
                        f"同一段录制出现两次且标签不同，请只保留一份：{name}，"
                        f"{recordings[seen[digest]]['file']}"
                    )
                continue
            if draft and not allow_draft:
                seen[digest] = len(recordings)
                recordings.append({"file": name, "labels": "draft"})
                unreviewed.append(name)
                continue
            meta, frames = read_contact_recording(recording)
            labels, kind = _labels(recording, labels_path, frames, meta["source"], allow_draft)
            schedule = _schedule(protocol)
        sources.add(kind)
        if len(sources) > 1:
            raise ValueError("不能把合成与真人录制混入同一实验")
        builder, base, usable, positives = WindowBuilder(config), len(features), 0, np.zeros(3, int)
        stamps = []
        for index, frame in enumerate(frames):
            window = builder.push(frame)
            if builder.history and builder.history[-1][0] == frame.timestamp:
                features.append(builder.history[-1][1])
                stamps.append(frame.timestamp)
            if window is None:
                skipped["warming_or_invalid"] += 1
                continue
            rules = rule_contacts(frame)
            if rules is None:
                skipped["warming_or_invalid"] += 1
                continue
            if (labels[index] < 0).any():
                skipped["unknown_label"] += 1
                columns["spare_last"].append(len(features) - 1)
                columns["spare_participants"].append(meta["participant"])
                continue
            round_number, step = 0, -1
            if schedule is not None:
                elapsed = frame.timestamp - schedule[0]
                position = int(np.searchsorted(schedule[1], elapsed, side="right")) - 1
                if position >= 0 and elapsed < schedule[2][position]:
                    round_number = schedule[3][position]
                    if schedule[4][position] not in step_names:
                        step_names.append(schedule[4][position])
                    step = step_names.index(schedule[4][position])
            columns["last"].append(len(features) - 1)
            columns["y"].append(labels[index])
            columns["rules"].append(rules)
            columns["participants"].append(meta["participant"])
            columns["recordings"].append(len(recordings))
            columns["rounds"].append(round_number)
            columns["steps"].append(step)
            usable += 1
            positives += labels[index]
        content = hashlib.sha256(
            np.asarray(features[base:], dtype=np.float32).tobytes()
            + np.round(np.asarray(stamps) - (stamps[0] if stamps else 0), 6).tobytes()
        ).hexdigest()
        if meta["episode"] in episodes or (stamps and content in contents):
            raise ValueError(f"重复录制；复制或修改起始时间不能创建独立样本：{name}")
        episodes.add(meta["episode"])
        contents.add(content)
        seen[digest] = len(recordings)
        recordings.append(
            {
                "file": name,
                "participant": meta["participant"],
                "session": meta["session"],
                "episode": meta["episode"],
                "recording_sha256": digest,
                "labels": "draft" if draft else "reviewed",
                "short_margins": _short_margins(label_spec),
                "frames": len(frames),
                "windows": usable,
                "positive_windows": dict(zip(CHANNELS, positives.tolist())),
                "protocol": schedule[5] if schedule is not None else None,
            }
        )
    if unreviewed:
        listed = "、".join(unreviewed[:5]) + (" 等" if len(unreviewed) > 5 else "")
        raise ValueError(
            f"{len(unreviewed)} 段录制的标签还没有复核：{listed}。"
            "先看效果可以加 --draft，这样跑出来的数字不能当作结果"
        )
    count = len(columns["last"])
    if not count:
        raise ValueError("没有可用的窗口：检查标签是否全部留空")
    metadata = {
        "schema": POOL_SCHEMA,
        "source": next(iter(sources)),
        "labels": "draft" if any(r["labels"] == "draft" for r in recordings) else "reviewed",
        "feature_version": FEATURE_VERSION,
        "window_config": asdict(config),
        "recordings": recordings,
        "step_names": step_names,
        "skipped": skipped,
    }
    return ContactPool(
        np.asarray(features, dtype=np.float32).reshape(len(features), FEATURE_COUNT),
        np.asarray(columns["last"], dtype=np.int64),
        np.asarray(columns["y"], dtype=np.int8).reshape(count, 3),
        np.asarray(columns["rules"], dtype=np.int8).reshape(count, 3),
        np.asarray(columns["participants"], dtype=str),
        np.asarray(columns["recordings"], dtype=np.int32),
        np.asarray(columns["rounds"], dtype=np.int8),
        np.asarray(columns["steps"], dtype=np.int16),
        metadata,
        config.frames,
        np.asarray(columns["spare_last"], dtype=np.int64),
        np.asarray(columns["spare_participants"], dtype=str),
    )


ARRAYS = ("features", "last", "y", "rules", "participants", "recordings", "rounds", "steps")
SPARE = ("spare_last", "spare_participants")


def load_contact_pool(source, *, config=None, allow_draft=False, cache=None):
    """Build the pool from a folder of recordings, or reuse a cached copy of the same inputs.

    Labels must be reviewed. `allow_draft` also takes the unreviewed drafts of guided
    sessions and marks the pool as draft, for trial runs only.
    """
    source = Path(source).resolve()
    config = config or WindowConfig()
    entries = _entries(source)
    key = _key(entries, source, config, allow_draft)
    target = Path(cache) / f"pool-{key[:20]}.npz" if cache is not None else None
    if target is not None and target.is_file():
        try:
            with np.load(target, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                if metadata.get("fingerprint") == key:
                    arrays = [saved[name] for name in ARRAYS]
                    spare = [saved[name] for name in SPARE]
                    return ContactPool(*arrays, metadata, config.frames, *spare)
        except (OSError, ValueError, KeyError):
            pass  # An unreadable cache is rebuilt below.
    pool = _build(entries, source, config, allow_draft)
    pool.metadata["fingerprint"] = key
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            metadata=json.dumps(pool.metadata, ensure_ascii=False),
            **{name: getattr(pool, name) for name in ARRAYS + SPARE},
        )
    return pool


def summary(pool):
    """Who recorded what, for a quick look before training."""
    people = {}
    for item in pool.metadata["recordings"]:
        person = people.setdefault(
            item["participant"],
            {"recordings": 0, "windows": 0, "positive_windows": dict.fromkeys(CHANNELS, 0)},
        )
        person["recordings"] += 1
        person["windows"] += item["windows"]
        for channel in CHANNELS:
            person["positive_windows"][channel] += item["positive_windows"][channel]
    return {
        "source": pool.metadata["source"],
        "labels": pool.metadata["labels"],
        "windows": len(pool),
        "unlabelled_windows": len(pool.spare_last),
        # Reviewed with unlabelled margins below the defaults: to be reviewed again.
        "reviewed_with_short_margins": [
            item["file"] for item in pool.metadata["recordings"] if item["short_margins"]
        ],
        "participants": dict(sorted(people.items())),
        "recordings": pool.metadata["recordings"],
        "skipped": pool.metadata["skipped"],
    }
