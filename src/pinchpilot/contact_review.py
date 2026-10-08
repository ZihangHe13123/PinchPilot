"""Review of a guided session's draft labels. Pure logic: no Qt, camera or native input.

The reviewer goes through the session unit by unit and accepts or discards each one. One
shift for the whole recording moves the Space-key marks to undo reaction time, and the
unlabelled margins can be widened. Labels come only from the prompts, the key and these
decisions; nothing is fitted to the keypoints.

A second person can then check a sample of the accepted units and record whether they agree.
"""

import hashlib
import json
import math
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, contact_protocol
from .contact_data import (
    ANNOTATION_SCHEMA,
    CHANNELS,
    _read_json,
    _sha,
    contact_features,
    is_protocol_draft,
    read_contact_recording,
)
from .contact_pool import PROTOCOL_SCHEMA, _member
from .storage import save_json

REVIEW_SCHEMA = "pinchpilot-contact-review-v1"
CROSSCHECK_SCHEMA = "pinchpilot-contact-crosscheck-v1"
ACCEPTED, DISCARDED = "accepted", "discarded"
AGREE, DISAGREE = "agree", "disagree"
SHORT_STEP = 4.0  # Runs of shorter steps without Space marks are reviewed as one unit.
LIMITS = {"shift": (-0.3, 0.5), "settle": (0.2, 2.5), "lead": (0.1, 1.5), "edge": (0.03, 0.4)}


@dataclass(frozen=True)
class Margins:
    """How the draft is re-derived, in seconds; the same values for the whole recording."""

    shift: float = 0.0  # The Space marks are moved this much earlier (reaction time).
    settle: float = contact_protocol.SETTLE
    lead: float = contact_protocol.LEAD
    edge: float = contact_protocol.KEY_EDGE

    def __post_init__(self):
        for name, (low, high) in LIMITS.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"复核参数 {name} 超出范围")


@dataclass(frozen=True)
class Unit:
    """What the reviewer decides on at once: one step, or a run of short steps."""

    index: int
    steps: tuple  # indices into the session's schedule
    start: float
    end: float
    round: int
    title: str
    uses_key: bool


@dataclass
class Session:
    path: Path  # the zip of a guided session, or the .jsonl of loose files
    stem: str  # sidecar files are named after it
    participant: str
    session: str
    source: str
    protocol: str
    recording_sha256: str
    frames: list
    times: np.ndarray  # seconds from the start of the session
    present: np.ndarray
    distances: np.ndarray  # [frames, 3] thumb-to-fingertip ratios; NaN without a usable hand
    schedule: list
    events: list
    ended: float
    units: list

    def sidecar(self, suffix):
        return self.path.with_name(f"{self.stem}{suffix}")


def _units(schedule, ended):
    units, run = [], []

    def close():
        if run:
            first, last = schedule[run[0]], schedule[run[-1]]
            title = (
                first[2].text if len(run) == 1 else f"{first[2].text} / {schedule[run[1]][2].text}"
            )
            if len(run) > 2:
                title += f"，共 {len(run)} 步"
            units.append(
                Unit(
                    len(units),
                    tuple(run),
                    first[0],
                    last[1],
                    first[2].round,
                    title,
                    any(contact_protocol.KEY in schedule[i][2].labels for i in run),
                )
            )
            run.clear()

    for index, (start, end, step) in enumerate(schedule):
        if start >= ended or all(label is None for label in step.labels):
            close()
            continue
        short = end - start < SHORT_STEP and contact_protocol.KEY not in step.labels
        if not short or (run and schedule[run[0]][2].round != step.round):
            close()
        run.append(index)
        if not short:
            close()
    close()
    return units


def _read_protocol(path):
    record = _read_json(path)
    try:
        if record.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError
        schedule = contact_protocol.steps_from_record(record)
        events = [(float(moment), bool(pressed)) for moment, pressed in record["space_key"]]
        return (
            schedule,
            events,
            float(record["clock_start"]),
            float(record["ended"]),
            str(record["protocol"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"提示记录格式无效：{Path(path).name}") from error


def load_session(path):
    """Read a guided session from its zip, or from a .jsonl with its .protocol.json."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as scratch:
        if path.suffix == ".zip":
            members = {}
            try:
                with zipfile.ZipFile(path) as archive:
                    for suffix in (".jsonl", ".protocol.json"):
                        name = _member(archive, suffix)
                        if name is not None:
                            members[suffix] = Path(scratch) / f"recording{suffix}"
                            members[suffix].write_bytes(archive.read(name))
            except (zipfile.BadZipFile, OSError) as error:
                raise ValueError(f"压缩包无法读取：{path.name}: {error}") from error
            recording, protocol = members.get(".jsonl"), members.get(".protocol.json")
        elif path.suffix == ".jsonl" and path.is_file():
            recording, protocol = path, path.with_suffix(".protocol.json")
            protocol = protocol if protocol.is_file() else None
        else:
            raise ValueError(f"请选择按提示录制生成的 .zip 或 .jsonl：{path.name}")
        if recording is None or protocol is None:
            raise ValueError(f"这段录制没有提示记录，不是按提示录的：{path.name}")
        meta, frames = read_contact_recording(recording)
        digest = _sha(recording)
        schedule, events, clock_start, ended, name = _read_protocol(protocol)
    distances = np.full((len(frames), len(CHANNELS)), np.nan)
    for index, frame in enumerate(frames):
        vector = contact_features(frame)
        if vector is not None:
            distances[index] = vector[63:66]
    return Session(
        path,
        path.stem,
        meta["participant"],
        meta["session"],
        meta["source"],
        name,
        digest,
        frames,
        np.array([frame.timestamp - clock_start for frame in frames]),
        np.array([bool(frame.landmarks) for frame in frames]),
        distances,
        schedule,
        events,
        ended,
        _units(schedule, ended),
    )


def shifted_events(session, margins):
    return [(moment - margins.shift, pressed) for moment, pressed in session.events]


def frame_labels(session, margins, decisions):
    """[frames, 3] labels (0, 1, or -1 for none) under these margins and decisions."""
    labels = contact_protocol.label_array(
        session.times,
        session.present,
        session.schedule,
        shifted_events(session, margins),
        margins.settle,
        margins.lead,
        margins.edge,
    )
    labels[session.times >= session.ended] = -1
    for unit in session.units:
        if decisions.get(unit.index) == DISCARDED:
            labels[(session.times >= unit.start) & (session.times < unit.end)] = -1
    return labels


def pending(session, decisions):
    return [unit.index for unit in session.units if decisions.get(unit.index) is None]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _name(value, what):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"需要填写{what}的编号")
    return value.strip()


def _check_decisions(session, decisions, allowed):
    known = {unit.index for unit in session.units}
    if any(key not in known or value not in allowed for key, value in decisions.items()):
        raise ValueError("复核记录里的单元或结论无效")


def save_progress(session, annotator, margins, decisions):
    """Remember where the review stands, so that it can be closed and continued."""
    _check_decisions(session, decisions, (ACCEPTED, DISCARDED))
    save_json(
        session.sidecar(".review.json"),
        {
            "schema": REVIEW_SCHEMA,
            "recording_sha256": session.recording_sha256,
            "annotator": _name(annotator, "复核者"),
            "margins": asdict(margins),
            "decisions": {str(key): value for key, value in sorted(decisions.items())},
            "units": len(session.units),
            "complete": not pending(session, decisions),
            "updated_utc": _now(),
            "app_version": __version__,
        },
    )


def load_progress(session):
    """(annotator, margins, decisions) saved earlier for this recording, or None."""
    path = session.sidecar(".review.json")
    if not path.is_file():
        return None
    record = _read_json(path)
    try:
        if (
            record.get("schema") != REVIEW_SCHEMA
            or record.get("recording_sha256") != session.recording_sha256
        ):
            raise ValueError
        margins = Margins(**record["margins"])
        decisions = {int(key): value for key, value in record["decisions"].items()}
        _check_decisions(session, decisions, (ACCEPTED, DISCARDED))
        return _name(record["annotator"], "复核者"), margins, decisions
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"复核进度文件与这段录制不匹配：{path.name}") from error


def save_reviewed(session, annotator, margins, decisions):
    """Write the reviewed labels beside the recording; every unit must have a decision."""
    annotator = _name(annotator, "复核者")
    _check_decisions(session, decisions, (ACCEPTED, DISCARDED))
    left = pending(session, decisions)
    if left:
        raise ValueError(f"还有 {len(left)} 个单元没有看完")
    if session.source != "camera":
        raise ValueError("只有真人录制的标签可以标为已复核")
    target = session.sidecar(".labels.json")
    if target.is_file():
        # Beside loose files this name holds the draft; a zip keeps its draft inside.
        existing = _read_json(target)
        ours = isinstance(existing.get("review"), dict) or is_protocol_draft(existing)
        if not ours or existing.get("recording_sha256") != session.recording_sha256:
            raise ValueError(f"已有一份不是本工具写的标签，不能覆盖：{target.name}")
    labels = frame_labels(session, margins, decisions).tolist()
    rows = [tuple(None if value < 0 else value for value in row) for row in labels]
    discarded = [unit for unit in session.units if decisions[unit.index] == DISCARDED]
    save_json(
        target,
        {
            "schema": ANNOTATION_SCHEMA,
            "recording_sha256": session.recording_sha256,
            "recording_source": session.source,
            "reviewed": True,
            "label_source": "human_reviewed",
            "protocol": session.protocol,
            "annotator": annotator,
            "channels": list(CHANNELS),
            "interval_units": "zero_based_frame_index_half_open",
            "intervals": contact_protocol.intervals(rows),
            "review": {
                "schema": REVIEW_SCHEMA,
                "method": "draft from prompts and Space key; each unit accepted or discarded "
                "by the annotator; one shift and margins for the whole recording",
                "margins": asdict(margins),
                "units": len(session.units),
                "discarded_units": [
                    {"unit": unit.index, "round": unit.round, "title": unit.title}
                    for unit in discarded
                ],
                "completed_utc": _now(),
                "app_version": __version__,
            },
        },
    )
    save_progress(session, annotator, margins, decisions)
    return target


def load_reviewed(session):
    """(annotator, margins, decisions, sha256 of the file) of the reviewed labels."""
    path = session.sidecar(".labels.json")
    if not path.is_file():
        raise ValueError(f"这段录制还没有复核过的标签：{session.stem}")
    spec = _read_json(path)
    review = spec.get("review")
    try:
        if (
            spec.get("reviewed") is not True
            or spec.get("recording_sha256") != session.recording_sha256
            or review.get("schema") != REVIEW_SCHEMA
        ):
            raise ValueError
        discarded = {item["unit"] for item in review["discarded_units"]}
        decisions = {
            unit.index: DISCARDED if unit.index in discarded else ACCEPTED for unit in session.units
        }
        return (
            _name(spec["annotator"], "复核者"),
            Margins(**review["margins"]),
            decisions,
            _sha(path),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"这段录制还没有复核过的标签：{session.stem}") from error


def sample_units(session, decisions, checker, fraction=0.1, minimum=6):
    """The accepted units a second person checks: the same ones every time for that person."""
    accepted = [unit for unit in session.units if decisions.get(unit.index) == ACCEPTED]
    count = min(len(accepted), max(minimum, math.ceil(fraction * len(session.units))))
    seed = hashlib.sha256(f"{session.recording_sha256}:{_name(checker, '抽查者')}".encode())
    rng = np.random.default_rng(int(seed.hexdigest()[:16], 16))
    marked = [unit.index for unit in accepted if unit.uses_key]
    plain = [unit.index for unit in accepted if not unit.uses_key]
    rng.shuffle(marked)
    rng.shuffle(plain)
    # Space-marked units carry the timing risk, so they make up at least half of the sample.
    take = min(len(marked), max(math.ceil(count / 2), count - len(plain)))
    return sorted(marked[:take] + plain[: count - take])


def save_crosscheck(session, checker, verdicts, notes=None):
    """Record a second person's verdicts on their sample of the reviewed labels."""
    checker = _name(checker, "抽查者")
    annotator, _, decisions, labels_sha = load_reviewed(session)
    if checker == annotator:
        raise ValueError("抽查要由另一位组员来做")
    sample = sample_units(session, decisions, checker)
    if any(key not in sample or value not in (AGREE, DISAGREE) for key, value in verdicts.items()):
        raise ValueError("抽查记录里的单元或结论无效")
    notes = notes or {}
    units = {unit.index: unit for unit in session.units}
    target = session.sidecar(f".crosscheck.{checker}.json")
    save_json(
        target,
        {
            "schema": CROSSCHECK_SCHEMA,
            "recording_sha256": session.recording_sha256,
            "labels_sha256": labels_sha,
            "annotator": annotator,
            "checker": checker,
            "sample": sample,
            "verdicts": [
                {
                    "unit": index,
                    "round": units[index].round,
                    "title": units[index].title,
                    "verdict": verdicts[index],
                    "note": str(notes.get(index, "")).strip(),
                }
                for index in sample
                if index in verdicts
            ],
            "agreed": sum(value == AGREE for value in verdicts.values()),
            "complete": len(verdicts) == len(sample),
            "updated_utc": _now(),
            "app_version": __version__,
        },
    )
    return target


def load_crosscheck(session, checker):
    """(verdicts, notes) saved earlier by this checker for the current reviewed labels."""
    path = session.sidecar(f".crosscheck.{_name(checker, '抽查者')}.json")
    if not path.is_file():
        return {}, {}
    record = _read_json(path)
    if (
        record.get("schema") != CROSSCHECK_SCHEMA
        or record.get("recording_sha256") != session.recording_sha256
        or record.get("labels_sha256") != load_reviewed(session)[3]
    ):
        return {}, {}  # The labels were reviewed again since; the old verdicts do not apply.
    rows = record.get("verdicts", [])
    return (
        {row["unit"]: row["verdict"] for row in rows},
        {row["unit"]: row.get("note", "") for row in rows if row.get("note")},
    )


def disagreements(session):
    """{unit: [(checker, note)]} from cross-checks of the current reviewed labels."""
    try:
        current = load_reviewed(session)[3]
    except ValueError:
        return {}
    found = {}
    for path in sorted(session.path.parent.glob(f"{session.stem}.crosscheck.*.json")):
        try:
            record = _read_json(path)
            if record.get("labels_sha256") != current:
                continue
            for row in record["verdicts"]:
                if row["verdict"] == DISAGREE:
                    found.setdefault(int(row["unit"]), []).append(
                        (str(record["checker"]), str(row.get("note", "")))
                    )
        except (ValueError, KeyError, TypeError):
            continue
    return found


def list_recordings(folder):
    """Guided recordings under a folder with who made them and how far their review is."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"找不到文件夹：{folder}")
    rows = []
    for path in sorted(folder.rglob("*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                recording = _member(archive, ".jsonl")
                if recording is None or _member(archive, ".protocol.json") is None:
                    continue
                with archive.open(recording) as stream:
                    meta = json.loads(stream.readline().decode("utf-8"))
            participant = str(meta["participant"]).strip()
        except (zipfile.BadZipFile, OSError, ValueError, KeyError, TypeError):
            continue
        status, decided, units, discarded, margins = "未检查", 0, None, None, None
        progress = path.with_name(f"{path.stem}.review.json")
        labels = path.with_name(f"{path.stem}.labels.json")
        if progress.is_file():
            try:
                record = _read_json(progress)
                decided, units = len(record["decisions"]), int(record["units"])
                discarded = sum(value == DISCARDED for value in record["decisions"].values())
                margins = record["margins"]
                status = "已检查" if record.get("complete") and labels.is_file() else "检查中"
            except (ValueError, KeyError, TypeError, AttributeError):
                status = "进度文件损坏"
        checks = []
        for item in sorted(path.parent.glob(f"{path.stem}.crosscheck.*.json")):
            try:
                record = _read_json(item)
                checks.append(
                    {
                        "checker": str(record["checker"]),
                        "agreed": int(record["agreed"]),
                        "checked": len(record["verdicts"]),
                        "sample": len(record["sample"]),
                        "complete": bool(record["complete"]),
                        "current": labels.is_file() and record["labels_sha256"] == _sha(labels),
                    }
                )
            except (ValueError, KeyError, TypeError):
                continue
        rows.append(
            {
                "path": path,
                "participant": participant,
                "status": status,
                "decided": decided,
                "units": units,
                "discarded": discarded,
                "margins": margins,
                "crosschecks": checks,
            }
        )
    return rows
