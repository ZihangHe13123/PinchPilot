"""Offline contact suggestions only: no gesture-engine or native-input execution."""

import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from .contact_data import (
    CHANNELS,
    FEATURE_COUNT,
    FEATURE_VERSION,
    WindowBuilder,
    WindowConfig,
    read_contact_recording,
    rule_contacts,
)
from .domain import HandFrame
from .storage import save_json
from .tripod import TripodConfig, tripod_features

SCHEMA = "pinchpilot-contact-shadow-v1"
MAX_REPORT_FRAMES = 100_000
SUGGESTION_THRESHOLD = 0.5
_WINDOW_FIELDS = {"frames", "max_gap_seconds", "max_window_seconds"}
_UNAVAILABLE_REASONS = {"missing_hand", "invalid_frame", "nonmonotonic_time"}
_RULE_CONFIG = TripodConfig(pointer_basis="unified", motion_profile="precise")


def _window_config(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Contact model metadata must be a dictionary")
    if (
        metadata.get("channels") != list(CHANNELS)
        or metadata.get("feature_count") != FEATURE_COUNT
        or metadata.get("feature_version") != FEATURE_VERSION
    ):
        raise ValueError(
            "Contact model channels or feature definition do not match the window input"
        )
    values = metadata.get("window_config")
    if not isinstance(values, dict) or set(values) != _WINDOW_FIELDS:
        raise ValueError("Contact model must declare all three window_config fields")
    if isinstance(values["frames"], bool) or not isinstance(values["frames"], int):
        raise ValueError("Contact model window frames must be an integer")
    for key in ("max_gap_seconds", "max_window_seconds"):
        value = values[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("Contact model window duration must be finite and numeric")
    try:
        config = WindowConfig(**values)
        # WindowBuilder owns the authoritative range/relationship validation.
        builder = WindowBuilder(config)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid contact model window_config") from error
    return config, builder


def _rule_geometry(frame):
    """Instantaneous ratios/thresholds, not state-machine decisions or human labels."""
    if not isinstance(frame, HandFrame):
        return None
    try:
        features = tripod_features(frame, spatial_grip=True)
        if features is None:
            return None
        projected = (
            math.hypot(
                (frame.landmarks[4][0] - frame.landmarks[12][0]) * frame.aspect,
                frame.landmarks[4][1] - frame.landmarks[12][1],
            )
            / features.scale
        )
        ratios = (features.grip, features.contact, features.right_contact)
        if not all(math.isfinite(value) for value in (*ratios, projected)):
            return None
        flags = rule_contacts(frame)
        if flags is None:
            return None
        return {
            "ratios": dict(zip(CHANNELS, map(float, ratios))),
            "projected_middle_ratio": float(projected),
            "below_engage_or_touch_threshold": dict(zip(CHANNELS, map(bool, flags))),
        }
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


class ContactShadow:
    """Observe causal windows without changing the caller's interaction engine.

    A real ContactPredictor validates its model schema and feature version when
    loading. This wrapper additionally checks its window/channel contract. Call
    reset on source, configuration, selected-hand identity or model changes.
    Empty/invalid frames and temporal breaks are handled by WindowBuilder.
    """

    def __init__(self, predictor, *, clock=None):
        self.predictor = predictor
        self.config, self.builder = _window_config(getattr(predictor, "metadata", None))
        self.clock = time.perf_counter if clock is None else clock
        self.last_reset_reason = "initial"

    def reset(self, reason="reset"):
        self.builder.reset()
        self.last_reset_reason = str(reason)[:160]

    def process(self, frame: HandFrame) -> dict:
        result = {
            "status": "warming",
            "reason": "warming_up",
            "scores": dict.fromkeys(CHANNELS),
            "suggested_contact": dict.fromkeys(CHANNELS),
            "inference_ms": None,
            "rule_geometry": _rule_geometry(frame),
        }
        try:
            window = self.builder.push(frame)
        except (TypeError, ValueError, OverflowError):
            self.reset("invalid_frame")
            result.update(status="unavailable", reason="invalid_frame")
            return result
        if window is None:
            reason = self.builder.reason
            result.update(
                status="unavailable" if reason in _UNAVAILABLE_REASONS else "warming",
                reason=reason,
            )
            return result
        started = self.clock()
        try:
            window = np.asarray(window)
            if window.shape != (self.config.frames, FEATURE_COUNT) or not np.isfinite(window).all():
                raise ValueError("Invalid contact window")
            scores = np.asarray(self.predictor.predict_windows(window[None, ...]), dtype=float)
            if scores.shape != (1, len(CHANNELS)) or not np.isfinite(scores).all():
                raise ValueError("Invalid contact score shape or values")
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError("Contact scores are outside [0, 1]")
        except Exception as error:
            # A failed observation must not leave a previous positive score alive.
            # Do not serialize arbitrary exception text: it can contain input data.
            self.reset("prediction_failed")
            result.update(
                status="unavailable",
                reason="prediction_failed",
                error_type=type(error).__name__,
            )
            return result
        elapsed = (self.clock() - started) * 1000
        result.update(
            status="ready",
            reason="ready",
            scores=dict(zip(CHANNELS, map(float, scores[0]))),
            suggested_contact=dict(zip(CHANNELS, map(bool, scores[0] >= SUGGESTION_THRESHOLD))),
            inference_ms=float(elapsed) if math.isfinite(elapsed) and elapsed >= 0 else None,
        )
        return result

    push = process


def run_shadow(recording: Path, model_path: Path, output: Path) -> dict:
    """Write a bounded, per-frame JSON shadow report, without images/landmarks.

    Model loading follows ContactPredictor's trusted-local RF / numeric CNN
    artifact policy. No input files are modified, and errors before completion
    leave an existing report unchanged. Timing is local inference execution, not
    gesture, capture, operating-system or display latency.
    """
    from .contact_models import ContactPredictor

    recording, model_path, output = map(Path, (recording, model_path, output))
    write_paths = {output.resolve(), output.with_suffix(output.suffix + ".tmp").resolve()}
    if recording.resolve() in write_paths:
        raise ValueError("Shadow output must not overwrite its input recording")
    model_directory = model_path if model_path.is_dir() else model_path.parent
    protected = {model_path.resolve()}
    protected.update(
        (model_directory / name).resolve()
        for name in ("metadata.json", "metrics.json", "model.joblib", "weights.npz")
    )
    if write_paths & protected:
        raise ValueError("Shadow output must not overwrite a model artifact")
    metadata, frames = read_contact_recording(recording)
    if not 0 < len(frames) <= MAX_REPORT_FRAMES:
        raise ValueError(f"Shadow recording must contain 1..{MAX_REPORT_FRAMES} frames")
    if metadata.get("source") not in ("camera", "synthetic", "legacy_unspecified"):
        raise ValueError("Contact recording source is unsupported")
    predictor = ContactPredictor(model_path)
    shadow = ContactShadow(predictor)
    model_metadata = predictor.metadata
    if model_metadata.get("source") not in ("camera", "synthetic"):
        raise ValueError("Contact model must declare camera or synthetic training source")
    rows = []
    for index, frame in enumerate(frames):
        observation = shadow.process(frame)
        rows.append({"frame_index": index, "timestamp_s": frame.timestamp, **observation})
    elapsed = [row["inference_ms"] for row in rows if row["inference_ms"] is not None]
    report = {
        "schema": SCHEMA,
        "app_version": __version__,
        "source": metadata["source"],
        "model_training_source": model_metadata["source"],
        "synthetic_evidence_only": metadata["source"] == "synthetic"
        or model_metadata["source"] == "synthetic",
        "recording_source_declared": metadata["source"] != "legacy_unspecified",
        "recording": {
            key: metadata[key]
            for key in ("schema", "participant", "session", "episode", "source")
            if key in metadata
        },
        "model": {
            key: model_metadata[key]
            for key in (
                "schema",
                "model",
                "feature_version",
                "feature_count",
                "channels",
                "source",
                "training_source",
                "evidence",
            )
            if key in model_metadata
        },
        "window_config": asdict(shadow.config),
        "suggestion_threshold": SUGGESTION_THRESHOLD,
        "rule_reference": {
            "pointer_basis": "unified",
            "grip_engage": _RULE_CONFIG.grip_engage,
            "projected_grip_veto": _RULE_CONFIG.grip_release,
            "index_touch": _RULE_CONFIG.touch_ratio,
            "ring_touch": _RULE_CONFIG.right_touch_ratio,
        },
        "summary": {
            "frames": len(rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "reason_counts": dict(Counter(row["reason"] for row in rows)),
            "timed_predictions": len(elapsed),
            "inference_p50_ms": float(np.median(elapsed)) if elapsed else None,
            "inference_p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
        },
        "desktop_realtime_integrated": False,
        "camera_opened": False,
        "os_events_sent": False,
        "gesture_engine_executed": False,
        "stores_images": False,
        "stores_keypoints": False,
        "definitions": {
            "scores": "Model scores in channel order, not calibrated contact probabilities or user-intent labels. Null while warming or unavailable.",
            "suggested_contact": "Independent score >= 0.5 flags. Multiple channels may be true. No click, drag, release or movement is generated.",
            "rule_geometry": "Instantaneous default unified-mode ratios and engage/touch comparisons, including the projected middle-grip veto. No temporal, hover, release, arming or button-state rules are executed.",
            "inference_ms": "Local window validation and predict_windows execution, excluding recording read, geometry observation, model loading, capture and display. Not end-to-end response latency.",
        },
        "limitations": [
            "Offline shadow only; not connected to desktop realtime control or OS input.",
            "Contact scores and rule observations are not human contact or intended-action ground truth.",
            "No contact accuracy, false-click rate, event latency, comfort or fatigue conclusion is produced.",
            "A synthetic recording or synthetic-trained model provides engineering evidence only.",
            "Legacy recordings have unverified source; legacy_unspecified is never treated as camera evidence.",
            "Selected-hand identity and source changes need caller reset; handedness alone is not a persistent hand identity.",
        ],
        "trace": rows,
    }
    save_json(output, report)
    return report
