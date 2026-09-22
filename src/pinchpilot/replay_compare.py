"""Deterministic landmark replay comparison, with no camera or OS input access.

Standalone synthetic run:
    python -m pinchpilot.replay_compare --output reports/replay-comparison.json

All three configurations use the current engine, not historical executable builds.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import __version__
from .domain import HandFrame
from .storage import read_recording, save_json
from .tripod import TripodConfig, TripodEngine
from .tripod_demo import synthetic_tripod

SCHEMA = "pinchpilot-replay-comparison-v1"
DEFAULT_SEED = 20260922
FPS = 30
PROFILES = (
    ("unified", "unified", "precise"),
    ("position_precise", "position", "precise"),
    ("position_classic", "position", "classic"),
)


@dataclass(frozen=True)
class ReplayCase:
    name: str
    description: str
    frames: tuple[HandFrame, ...]
    # Half-open frame-index intervals explicitly designated stationary by a fixture.
    stationary_windows: tuple[tuple[str, int, int], ...] = ()


def synthetic_cases(seed=DEFAULT_SEED):
    """Generated hand landmarks; no claim about real detector noise or user intent."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    rng = np.random.default_rng(seed)
    cases = []

    def begin():
        return [synthetic_tripod(100.0 + i / FPS) for i in range(30)]

    def append(frames, count=1, *, shift=(0.0, 0.0), missing=False, gap=0.0, **gesture):
        for index in range(count):
            timestamp = frames[-1].timestamp + 1 / FPS + (gap if index == 0 else 0)
            if missing:
                frames.append(HandFrame(timestamp))
            else:
                frame = synthetic_tripod(timestamp, **gesture)
                points = np.array(frame.landmarks)
                points[:, :2] += shift  # Rigid translation of the entire visible hand.
                frames.append(replace(frame, landmarks=tuple(map(tuple, points))))

    frames = begin()
    for noise in rng.normal(0, [0.00065, 0.00075], (360, 2)):
        append(frames, noise=tuple(noise))
    cases.append(
        ReplayCase(
            "stationary_jitter",
            "Fixed palm and intended midpoint; Gaussian fingertip XY noise, sigma=(.00065,.00075). "
            "30 clean arming frames; exclude another 60 disturbance-onset frames from jitter.",
            tuple(frames),
            (("stationary_after_settling", 90, len(frames)),),
        )
    )
    frames = begin()
    for fraction in np.linspace(0, 1, 60):
        append(frames, shift=(0.07 * fraction, -0.035 * fraction))
    append(frames, 45, shift=(0.07, -0.035))
    cases.append(ReplayCase("translation", "Rigid diagonal translation, then hold.", tuple(frames)))
    frames = begin()
    for offset in np.linspace(0, 0.20, 90):
        append(frames, shift=(offset, 0))
    append(frames, 30, shift=(0.20, 0))
    for offset in np.linspace(0.20, 0.17, 30):
        append(frames, shift=(offset, 0))
    for offset in np.linspace(0.17, 0, 60):
        append(frames, shift=(offset, 0))
    append(frames, 45)
    cases.append(
        ReplayCase(
            "edge_reversal",
            "Rigid rightward translation beyond the default mapping edge, then reverse and return. "
            "Clamped output means a closed input path need not return to its original screen point.",
            tuple(frames),
        )
    )
    frames = begin()
    append(frames, 4, contact=0.1)
    append(frames, 12)
    append(frames, 8, right_contact=0.1)
    append(frames, 12)
    append(frames, 16, contact=0.1)
    for offset in np.linspace(0, 0.04, 45):
        append(frames, shift=(offset, 0), contact=0.1)
    append(frames, 12, shift=(0.04, 0))
    cases.append(
        ReplayCase(
            "click_drag",
            "Short left contact, right contact, held left contact and rigid drag, then release. "
            "Intentional approach/press freezes are not tracking errors.",
            tuple(frames),
        )
    )
    frames = begin()
    append(frames, 16, contact=0.1)
    append(frames, 3, missing=True)
    append(frames, 10, contact=0.1)  # Returning with contact held cannot click again.
    append(frames, 12)
    append(frames, 16, contact=0.1)
    append(frames, gap=0.35, contact=0.1)  # Timeout with a held button.
    append(frames, 10, contact=0.1)
    append(frames, 12)
    cases.append(
        ReplayCase(
            "tracking_loss",
            "Missing landmarks during a drag, held-contact return, reacquisition, then a .35s "
            "extra timestamp gap during another held contact. No native button is pressed.",
            tuple(frames),
        )
    )
    return tuple(cases)


def _validate_frames(frames):
    if not frames:
        raise ValueError("Replay requires at least one HandFrame")
    previous = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, HandFrame):
            raise ValueError(f"Frame {index}: expected HandFrame")
        for name in ("timestamp", "aspect", "handedness_score"):
            value = getattr(frame, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Frame {index}: invalid {name}")
        if frame.aspect <= 0 or not 0 <= frame.handedness_score <= 1:
            raise ValueError(f"Frame {index}: invalid aspect or handedness score")
        if previous is not None and frame.timestamp <= previous:
            raise ValueError(f"Frame {index}: timestamps must strictly increase")
        if frame.handedness not in ("", "Left", "Right"):
            raise ValueError(f"Frame {index}: unsupported handedness")
        if frame.landmarks:
            points = np.asarray(frame.landmarks, dtype=float)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raise ValueError(f"Frame {index}: invalid landmark coordinates")
        previous = frame.timestamp


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _run(frames, config):
    engine = TripodEngine(config)
    engine.pointer = (0.5, 0.5)
    engine.reset()
    rows = []
    for index, frame in enumerate(frames):
        result = engine.process(frame)
        pointer = [float(v) for v in result.pointer] if result.pointer is not None else None
        raw = [float(v) for v in result.raw_pointer] if result.raw_pointer is not None else None
        if pointer is None or not all(math.isfinite(v) and 0 <= v <= 1 for v in pointer):
            raise ValueError(f"Engine returned an invalid pointer at frame {index}")
        if raw is not None and not all(math.isfinite(v) for v in raw):
            raise ValueError(f"Engine returned an invalid raw pointer at frame {index}")
        rows.append(
            {
                "frame": index,
                "timestamp_s": float(frame.timestamp),
                "elapsed_s": float(frame.timestamp - frames[0].timestamp),
                "landmarks_present": bool(frame.landmarks),
                "input_midpoint_camera_xy": (
                    np.asarray(frame.landmarks)[[4, 12], :2].mean(axis=0).tolist()
                    if frame.landmarks
                    else None
                ),
                "state": result.state,
                "pointer_normalized_xy": pointer,
                "raw_pointer_normalized_xy": raw,
                "motion_engaged": bool(engine.motion_engaged),
                "grip_pending": engine.grip_pending_since is not None,
                "left_button_held": bool(engine.left_down),
                "cancelled": bool(result.cancelled),
                "events": [asdict(event) for event in result.events],
            }
        )
    return rows


def _noise(points):
    if len(points) < 2:
        return None
    values = np.asarray(points, dtype=float)
    radius = np.linalg.norm(values - values.mean(axis=0), axis=1)
    return {
        "radial_rms_px": float(np.sqrt(np.mean(radius**2))),
        "radial_p95_px": float(np.quantile(radius, 0.95)),
        "axis_peak_to_peak_px": np.ptp(values, axis=0).tolist(),
    }


def _metrics(rows, frames, config, stationary):
    size = np.array([config.screen_width, config.screen_height])
    points = np.array([row["pointer_normalized_xy"] for row in rows]) * size
    events = Counter(event["kind"] for row in rows for event in row["events"])
    transitions = Counter()
    reacquisitions = drag_entries = 0
    for before, after in zip(rows, rows[1:]):
        if before["state"] != after["state"]:
            transitions[f"{before['state']} -> {after['state']}"] += 1
            reacquisitions += after["state"] == "WAIT_GRIP"
            drag_entries += after["state"] == "DRAG"
    missing_runs = sum(
        not frame.landmarks and (i == 0 or bool(frames[i - 1].landmarks))
        for i, frame in enumerate(frames)
    )
    windows = []
    for name, start, stop, eligible in stationary:
        selected = [rows[i] for i in eligible]
        windows.append(
            {
                "name": name,
                "frame_start_inclusive": start,
                "frame_stop_exclusive": stop,
                "requested_frames": stop - start,
                "common_eligible_frames": len(eligible),
                "excluded_frames": stop - start - len(eligible),
                "output_jitter": _noise(
                    [np.array(r["pointer_normalized_xy"]) * size for r in selected]
                ),
                "raw_signal_jitter": _noise(
                    [np.array(r["raw_pointer_normalized_xy"]) * size for r in selected]
                ),
            }
        )
    return {
        "frames": len(rows),
        "duration_s": float(frames[-1].timestamp - frames[0].timestamp),
        "output_path_length_px": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        "output_net_displacement_px": (points[-1] - points[0]).tolist(),
        "output_max_excursion_from_start_px": float(
            np.linalg.norm(points - points[0], axis=1).max()
        ),
        "generated_event_counts": dict(sorted(events.items())),
        "generated_scroll_signed_sum": float(
            sum(
                event["value"]
                for row in rows
                for event in row["events"]
                if event["kind"] == "scroll"
            )
        ),
        "state_frame_counts": dict(sorted(Counter(row["state"] for row in rows).items())),
        "state_transition_counts": dict(sorted(transitions.items())),
        "drag_entries": drag_entries,
        "left_button_held_at_input_end": rows[-1]["left_button_held"],
        "interruptions": {
            "input_missing_landmark_runs": missing_runs,
            "input_timestamp_gaps_at_least_timeout": sum(
                b.timestamp - a.timestamp >= config.tracking_timeout
                for a, b in zip(frames, frames[1:])
            ),
            "engine_reacquisition_entries": reacquisitions,
            "held_button_cancellations": sum(row["cancelled"] for row in rows),
        },
        "stationary_windows": windows,
    }


def run_comparison(
    recording: Path | None,
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    include_trace: bool = False,
    screen_width: float = 1920,
    screen_height: float = 1080,
    span: float = 0.30,
) -> dict:
    """Write one JSON report; output is a file, not a directory.

    A recording is read with storage.read_recording. Its gesture-class label is
    not a stationary interval, target location, or intention annotation. Thus a
    recorded replay has path/event/state metrics but no inferred jitter/error score.
    """
    output = Path(output)
    configs = {
        name: TripodConfig(
            pointer_basis=basis,
            motion_profile=profile,
            screen_width=screen_width,
            screen_height=screen_height,
            span=span,
        )
        for name, basis, profile in PROFILES
    }
    for config in configs.values():
        config.validate()
    if recording is None:
        cases = synthetic_cases(seed)
        source, input_info = "synthetic", {"generator": "replay_compare-v1", "fps": FPS}
    else:
        recording = Path(recording)
        if recording.resolve() == output.resolve():
            raise ValueError("Output must not overwrite the source recording")
        try:
            metadata, frames = read_recording(recording)
        except (OSError, UnicodeError, ValueError, TypeError, AttributeError, IndexError) as error:
            raise ValueError(f"Cannot replay recording {recording.name}: {error}") from error
        source = "recorded"
        input_info = {
            "path": str(recording.resolve()),
            "recording_schema": metadata["schema"],
            "human_selected_gesture_label": metadata["label"],
            "label_used_as_intent_or_stationarity": False,
        }
        cases = (
            ReplayCase(
                "recorded_sequence",
                "Frames and timestamps replayed without resampling.",
                tuple(frames),
            ),
        )
    for case in cases:
        _validate_frames(case.frames)
    report = {
        "schema": SCHEMA,
        "app_version": __version__,
        "numpy_version": np.__version__,
        "source": source,
        "seed": seed if source == "synthetic" else None,
        "rng": "numpy.default_rng/PCG64" if source == "synthetic" else None,
        "input": input_info,
        "initial_pointer_normalized_xy": [0.5, 0.5],
        "screen_coordinates": {
            "width": screen_width,
            "height": screen_height,
            "unit": "declared logical pixel",
        },
        "camera_opened": False,
        "os_events_sent": False,
        "models_loaded": False,
        "definitions": {
            "output_path_length_px": "Sum of Euclidean differences between consecutive per-frame pointer positions in the declared logical screen coordinates, including all gesture states. Not target error.",
            "output_net_displacement_px": "Last pointer minus first pointer; clamping and deliberate reanchoring can prevent a closed input path from returning to the initial screen point.",
            "stationary_jitter": "Radial RMS/p95 about each output's own mean, only inside explicitly declared synthetic stationary windows. The same indices are used for all profiles: CONTROL, motion engaged, no pending grip, valid raw pointer in every profile. Fewer than two eligible frames gives null.",
            "raw_signal_jitter": "The engine's mapped signal before pointer smoothing; unified palm assistance has already been applied. This is not the detector's unprocessed landmark noise.",
            "generated_event_counts": "InputEvents produced by the gesture engine; no OS API sends. A down/up pair is not proof of a successful application click or a double-click.",
            "interruptions": "Missing runs and time gaps describe the input. Engine entries into WAIT_GRIP can be deliberate (e.g. leaving scroll) and are not an error rate. held_button_cancellations counts EngineResult.cancelled, not all resets.",
            "state_frame_counts": "Observed gesture states, not ground-truth classifications. Approach/press/clutch freezes are normal interactions and are not scored as failures.",
        },
        "limitations": [
            "Synthetic fixtures are engineering checks, not real tracking accuracy, comfort, fatigue, or intent evidence.",
            "Recorded gesture labels alone do not identify stationary periods, desired targets, or expected click timings.",
            "All configurations run the current installed engine; this is not reproduction of historical release binaries.",
            "No camera, display, operating-system input latency, physical mouse movement, or application outcome is measured.",
            "No extra end-of-file release is injected; a held-button-at-end field describes the replay state only.",
        ],
        "configurations": {name: asdict(config) for name, config in configs.items()},
        "cases": {},
    }
    for case in cases:
        input_digest = _digest([asdict(frame) for frame in case.frames])
        traces = {name: _run(case.frames, config) for name, config in configs.items()}
        stationary = []
        for name, start, stop in case.stationary_windows:
            eligible = [
                i
                for i in range(start, stop)
                if all(
                    rows[i]["state"] == "CONTROL"
                    and rows[i]["motion_engaged"]
                    and not rows[i]["grip_pending"]
                    and rows[i]["raw_pointer_normalized_xy"] is not None
                    for rows in traces.values()
                )
            ]
            stationary.append((name, start, stop, eligible))
        compared = {}
        for name, rows in traces.items():
            compared[name] = {
                "input_sha256": input_digest,
                "trace_sha256": _digest(rows),
                "metrics": _metrics(rows, case.frames, configs[name], stationary),
            }
            if include_trace:
                compared[name]["trace"] = rows
        report["cases"][case.name] = {
            "description": case.description,
            "input_sha256": input_digest,
            "input_frames": len(case.frames),
            "profiles": compared,
        }
    # Return the same JSON-compatible types written to disk (rather than tuples/numpy scalars).
    report = json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/replay-comparison.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trace", action="store_true", help="Include per-frame generated output")
    args = parser.parse_args()
    report = run_comparison(args.recording, args.output, seed=args.seed, include_trace=args.trace)
    print(
        f"Saved {report['source']} comparison ({len(report['cases'])} cases): {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
