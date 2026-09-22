"""Synthetic unified-pointer contracts, not evidence of real hand-tracking accuracy."""

import math
from dataclasses import replace

import numpy as np
import pytest

from pinchpilot.tripod import TripodConfig, TripodEngine
from pinchpilot.tripod_demo import synthetic_tripod

PALM = (0, 5, 9, 13, 17)
SCREEN = np.array([1920.0, 1080.0])


def hand(
    timestamp,
    *,
    shift=(0.0, 0.0),
    roll=0.0,
    scale=1.0,
    aspect=4 / 3,
    mirrored=False,
    finger_noise=(0.0, 0.0),
    palm_depth=0.0,
    degenerate=False,
):
    frame = synthetic_tripod(timestamp)
    points = np.array(frame.landmarks)
    xy = points[:, :2] * [frame.aspect, 1]
    pivot = xy[0].copy()
    angle = math.radians(roll)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    xy = (xy - pivot) @ rotation.T * scale + pivot
    xy += np.array(shift) * [frame.aspect, 1]
    points[:, :2] = xy / [aspect, 1]
    points[:, 2] *= frame.aspect / aspect * scale
    if degenerate:
        points[list(PALM), 0] = points[list(PALM), 0].mean()
    # Exercise a dangerous new depth-noise channel without changing any image x/y.
    points[list(PALM), 2] += palm_depth * np.array([0.0, -1.0, 0.7, -0.5, 1.0])
    points[[4, 12], :2] += finger_noise
    if mirrored:
        points[:, 0] = 1 - points[:, 0]
    return replace(
        frame,
        landmarks=tuple(map(tuple, points)),
        aspect=aspect,
        handedness="Left" if mirrored else "Right",
    )


def midpoint(frame):
    return np.array(frame.landmarks)[[4, 12], :2].mean(axis=0)


class Sequence:
    def __init__(self, basis="unified", **config):
        settings = {"motion_profile": "precise", **config}
        self.engine = TripodEngine(TripodConfig(pointer_basis=basis, **settings))
        self.timestamp = 10.0
        self.result = self.engine.last_result

    def step(self, **pose):
        self.timestamp += 1 / 30
        frame = hand(self.timestamp, **pose)
        self.result = self.engine.process(frame)
        return frame, self.result

    def hold(self, frames=10, **pose):
        for _ in range(frames):
            self.step(**pose)
        return self.result


@pytest.mark.parametrize("aspect", [4 / 3, 16 / 9])
def test_rigid_translation_preserves_exact_position_raw_mapping(aspect):
    unified, position = Sequence(), Sequence("position")
    for sequence in (unified, position):
        sequence.hold(aspect=aspect)
        assert sequence.result.state == "CONTROL"
    for dx, dy in ((0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03), (0, 0)):
        for factor in np.linspace(0, 1, 24):
            options = {"shift": (dx * factor, dy * factor), "aspect": aspect}
            unified.step(**options)
            position.step(**options)
            assert unified.result.state == "CONTROL"
            assert unified.result.raw_pointer == pytest.approx(
                position.result.raw_pointer, abs=1e-10
            )
            assert unified.result.pointer == pytest.approx(position.result.pointer, abs=1e-10)


@pytest.mark.parametrize("mirrored", [False, True], ids=["right-hand", "mirrored-left-hand"])
@pytest.mark.parametrize("direction", [-1, 1], ids=["leftward", "rightward"])
def test_in_plane_rotation_plus_translation_counts_the_midpoint_once(mirrored, direction):
    sequence = Sequence()
    frame, _ = sequence.step(mirrored=mirrored)
    reference = midpoint(frame)
    sequence.hold(mirrored=mirrored)
    for fraction in np.linspace(0, 1, 45):
        frame, result = sequence.step(
            mirrored=mirrored,
            roll=direction * 12 * fraction,
            shift=(direction * 0.025 * fraction, -0.008 * fraction),
            scale=1 + 0.08 * fraction,
        )
        expected = 0.5 + (midpoint(frame) - reference) / sequence.engine.config.span
        assert result.state == "CONTROL"
        assert result.raw_pointer == pytest.approx(expected, abs=1e-10)
    expected_sign = -direction if mirrored else direction
    assert expected_sign * (result.pointer[0] - 0.5) > 0.15


def test_palm_depth_noise_cannot_move_a_stationary_image_midpoint():
    sequence = Sequence()
    sequence.hold()
    for depth in np.random.default_rng(607).normal(0, 0.015, 240):
        _, result = sequence.step(palm_depth=float(depth))
        assert result.state == "CONTROL"
        assert result.raw_pointer == pytest.approx((0.5, 0.5), abs=1e-10)
        assert result.pointer == pytest.approx((0.5, 0.5), abs=1e-10)
        assert not any(event.kind != "move" for event in result.events)


def test_auxiliary_degeneracy_falls_back_without_requiring_wrist_calibration():
    sequence = Sequence()
    sequence.hold()
    for fraction in np.linspace(0, 1, 25):
        frame, result = sequence.step(degenerate=True, shift=(0.02 * fraction, 0))
        expected = 0.5 + (midpoint(frame) - midpoint(hand(0))) / sequence.engine.config.span
        assert result.state == "CONTROL" and sequence.engine.motion_engaged
        assert np.linalg.norm((np.array(result.raw_pointer) - expected) * SCREEN) <= 3.01
        assert not any(event.kind != "move" for event in result.events)
    # A large, valid image-plane rotation is outside the retired wrist-only 65° domain.
    sequence = Sequence()
    sequence.hold()
    for angle in np.linspace(0, 95, 75):
        _, result = sequence.step(roll=angle)
        assert result.state == "CONTROL" and sequence.engine.motion_engaged


def test_auxiliary_degeneracy_and_recovery_cannot_move_a_stationary_midpoint():
    sequence = Sequence()
    sequence.hold(30)
    anchor = sequence.result.pointer
    for degenerate, frames in ((True, 30), (False, 60)):
        for _ in range(frames):
            _, result = sequence.step(degenerate=degenerate)
            assert result.state == "CONTROL"
            assert result.raw_pointer == pytest.approx(anchor, abs=1e-10)
            assert result.pointer == pytest.approx(anchor, abs=1e-10)


def test_repeated_closed_rigid_paths_return_without_accumulating_drift():
    sequence = Sequence()
    sequence.hold()
    returns = []
    for _ in range(4):
        for phase in np.linspace(0, 2 * math.pi, 90):
            sequence.step(
                roll=8 * math.sin(phase),
                shift=(0.02 * math.sin(phase), 0.012 * (1 - math.cos(phase))),
            )
        sequence.hold(60)
        assert sequence.result.raw_pointer == pytest.approx((0.5, 0.5), abs=1e-9)
        returns.append(np.array(sequence.result.pointer))
    # Existing screen deadband permits a bounded 2px endpoint offset, not exact zero.
    assert max(np.linalg.norm((point - 0.5) * SCREEN) for point in returns) <= 2.05
    assert max(np.linalg.norm((point - returns[0]) * SCREEN) for point in returns) < 0.05


def fingertip_noise_metrics():
    unified, position = Sequence(), Sequence("position")
    unified.hold()
    position.hold()
    samples = {
        name: [] for name in ("unified_raw", "position_raw", "unified_output", "position_output")
    }
    noise = np.random.default_rng(935).normal(0, [0.00065, 0.00075], (360, 2))
    for index, sample in enumerate(noise):
        for label, sequence in (("unified", unified), ("position", position)):
            _, result = sequence.step(finger_noise=sample)
            assert result.state == "CONTROL"
            if index >= 60:
                samples[label + "_raw"].append(np.array(result.raw_pointer) * SCREEN)
                samples[label + "_output"].append(np.array(result.pointer) * SCREEN)
        raw_error = (np.array(unified.result.raw_pointer) - position.result.raw_pointer) * SCREEN
        assert np.linalg.norm(raw_error) <= 3.01
    metrics = {}
    for name, rows in samples.items():
        values = np.array(rows)
        metrics[name] = float(np.sqrt(np.mean(np.sum((values - values.mean(axis=0)) ** 2, axis=1))))
    # Intentional fingertip movement remains available despite the auxiliary model.
    result = unified.hold(90, finger_noise=(12 * 0.3 / SCREEN[0], 0))
    metrics["intentional_12px_final"] = (result.pointer[0] - 0.5) * SCREEN[0]
    return metrics


def test_bounded_assistance_reduces_synthetic_fingertip_noise_without_losing_small_motion():
    metrics = fingertip_noise_metrics()
    assert metrics["unified_raw"] < 0.9 * metrics["position_raw"], metrics
    assert metrics["unified_output"] < metrics["position_output"], metrics
    assert 8 <= metrics["intentional_12px_final"] <= 14, metrics


def test_unified_gain_does_not_depend_on_saved_adaptive_profile_or_motion_speed():
    endpoints = []
    for profile, frames in (("precise", 60), ("adaptive", 6)):
        sequence = Sequence(motion_profile=profile)
        sequence.hold()
        for offset in np.linspace(0, 0.05, frames):
            sequence.step(shift=(offset, 0))
        result = sequence.hold(60, shift=(0.05, 0))
        endpoints.append(result.pointer)
        assert result.raw_pointer == pytest.approx((0.5 + 0.05 / 0.3, 0.5), abs=1e-10)
    assert endpoints[0] == pytest.approx(endpoints[1], abs=0.1 / SCREEN[0])
