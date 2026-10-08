import json
import math
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .domain import EngineConfig, HandFrame
from .features import FEATURE_VERSION, LABELS, extract

SCHEMA = "pinchpilot-recording-v1"


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w-]", "_", value.strip(), flags=re.UNICODE)[:60]
    if not value:
        raise ValueError("请填写参与者或场次名称")
    return value


class Recorder:
    def __init__(self, directory: Path, participant: str, session: str, label: str):
        if label not in LABELS:
            raise ValueError("无效的人工标签")
        self.participant, self.session = safe_name(participant), safe_name(session)
        self.label = label
        self.episode = uuid.uuid4().hex[:12]
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.participant}_{self.session}_{label}_{self.episode}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.count = self.valid_count = 0
        self.last_timestamp = None
        self._write(
            {
                "type": "metadata",
                "schema": SCHEMA,
                "participant": self.participant,
                "session": self.session,
                "episode": self.episode,
                "label": label,
                "label_source": "human_selected",
                "feature_version": FEATURE_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "video_saved": False,
            }
        )

    def _write(self, value: dict) -> None:
        self.file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def add(self, frame: HandFrame) -> None:
        if not math.isfinite(frame.timestamp):
            raise ValueError("无效时间戳")
        if frame.landmarks and extract(frame) is None:
            raise ValueError("无效关键点几何")
        if self.last_timestamp is not None and frame.timestamp <= self.last_timestamp:
            raise ValueError("录制帧必须按时间递增")
        self.last_timestamp = frame.timestamp
        self._write({"type": "frame", **asdict(frame)})
        self.count += 1
        self.valid_count += extract(frame) is not None
        if self.count % 30 == 0:
            self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()


def read_recording(path: Path) -> tuple[dict, list[HandFrame]]:
    frames = []
    with path.open(encoding="utf-8") as f:
        try:
            metadata = json.loads(next(f))
        except (StopIteration, ValueError) as e:
            raise ValueError(f"录制文件头无效：{path.name}") from e
        if metadata.get("schema") != SCHEMA or metadata.get("label") not in LABELS:
            raise ValueError(f"录制格式或标签不支持：{path.name}")
        for key in ("participant", "session", "episode"):
            if not metadata.get(key):
                raise ValueError(f"缺少 {key}：{path.name}")
        for number, line in enumerate(f, 2):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.pop("type") != "frame":
                    raise ValueError("记录类型无效")
                row["landmarks"] = tuple(tuple(p) for p in row["landmarks"])
                row["world_landmarks"] = tuple(tuple(p) for p in row.get("world_landmarks", ()))
                frame = HandFrame(**row)
                if not math.isfinite(frame.timestamp) or (
                    frames and frame.timestamp <= frames[-1].timestamp
                ):
                    raise ValueError("非递增时间戳")
                if frame.landmarks and extract(frame) is None:
                    raise ValueError("无效关键点几何")
                frames.append(frame)
            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(f"{path.name}:{number}: {e}") from e
    return metadata, frames


def recording_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise ValueError(f"未找到录制文件：{directory}")
    return files


def calibrate(directory: Path, participant: str, config: EngineConfig) -> tuple[EngineConfig, dict]:
    samples = {"open": [], "pinch": []}
    for path in recording_files(directory):
        meta, frames = read_recording(path)
        if meta["participant"] != safe_name(participant) or meta["label"] not in samples:
            continue
        # Cap each episode so a long recording cannot dominate calibration.
        valid = [extract(frame) for frame in frames]
        ratios = [f.pinch for f in valid if f is not None]
        if ratios:
            indices = np.linspace(0, len(ratios) - 1, min(60, len(ratios)), dtype=int)
            samples[meta["label"]].extend(ratios[i] for i in indices)
    if min(map(len, samples.values())) < 20:
        raise ValueError("需要同一参与者至少 20 帧张开和 20 帧捏合样本；请各录几次独立动作")
    close = float(np.quantile(samples["pinch"], 0.85))
    opened = float(np.quantile(samples["open"], 0.15))
    gap = opened - close
    if gap < 0.12:
        raise ValueError("张开与捏合样本重叠过多，请检查标签、摄像头角度并重录；原配置未改变")
    values = asdict(config)
    values.update(engage_ratio=close + gap * 0.25, release_ratio=close + gap * 0.60)
    result = EngineConfig(**values)
    result.validate()
    summary = {
        "participant": safe_name(participant),
        "samples": {k: len(v) for k, v in samples.items()},
        "pinch_q85": close,
        "open_q15": opened,
        "config": asdict(result),
        "evidence": "threshold calibration only; not a held-out performance evaluation",
    }
    return result, summary


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)
