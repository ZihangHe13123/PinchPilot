"""Synthetic replay for engineering checks ONLY. Never a training dataset."""

import math

import numpy as np

from .domain import HandFrame


def synthetic_hand(
    timestamp: float, pose: str = "open", x: float = 0.5, y: float = 0.5
) -> HandFrame:
    points = [
        (0.0, 0.14, 0.0),
        (-0.065, 0.10, 0),
        (-0.12, 0.05, 0),
        (-0.17, -0.02, 0),
        (-0.22, -0.07, 0),
    ]
    for base, length in [(-0.065, 0.19), (-0.020, 0.23), (0.025, 0.21), (0.070, 0.16)]:
        folded = pose == "fist" or (pose == "scroll" and base > 0)
        if folded:
            points.extend(
                [(base, 0.02, 0), (base, -0.03, 0), (base + 0.01, 0.0, 0), (base + 0.015, 0.06, 0)]
            )
        else:
            points.extend(
                [
                    (base, 0.02, 0),
                    (base, -length * 0.35, 0),
                    (base, -length * 0.7, 0),
                    (base, -length, 0),
                ]
            )
    if pose == "pinch":
        # Thumb moves towards index. Palm stays independent of the click gesture.
        tip = points[8]
        points[4] = (tip[0] - 0.008, tip[1] + 0.005, 0)
    return HandFrame(
        timestamp, tuple((a / (4 / 3) + x, b + y, c) for a, b, c in points), 4 / 3, "Right", 0.99
    )


def demo_frame(timestamp: float, elapsed: float) -> HandFrame:
    phase = elapsed % 12
    pose = (
        "pinch" if 3 < phase < 3.4 or 6 < phase < 8.5 else ("scroll" if 9 < phase < 11 else "open")
    )
    if phase > 11.4:
        return HandFrame(timestamp)
    return synthetic_hand(
        timestamp, pose, 0.5 + 0.13 * math.sin(elapsed * 0.6), 0.5 + 0.08 * math.cos(elapsed * 0.8)
    )


def synthetic_finger(
    timestamp: float,
    dx: float = 0,
    dy: float = 0,
    bend: float = 0,
    x: float = 0.5,
    y: float = 0.60,
) -> HandFrame:
    """Articulated engineering fixture. dx/dy are palm-scale pointing offsets.

    A straight index rotates at its base to point; separate PIP/DIP flexion
    supplies the click signal. This is not a model of real tracking noise.
    """
    base = synthetic_hand(timestamp, "fist", x, y)
    points = np.array(base.landmarks) * [base.aspect, 1, base.aspect]
    scale = (
        np.linalg.norm(points[5, :2] - points[17, :2])
        + np.linalg.norm(points[0, :2] - points[9, :2])
    ) / 2
    vx, vy = dx * scale, -0.16 + dy * scale
    length = 0.21
    if vx * vx + vy * vy >= length * length:
        raise ValueError("Synthetic finger target exceeds its fixture reach")
    direction = np.array([vx, vy, -math.sqrt(length**2 - vx**2 - vy**2)]) / length
    normal = np.array([0.0, 0.0, -1.0])
    normal -= np.dot(normal, direction) * direction
    normal /= np.linalg.norm(normal)
    for joint, segment, angle in (
        (6, 0.08, 0),
        (7, 0.07, bend * math.pi),
        (8, 0.06, 2 * bend * math.pi),
    ):
        points[joint] = points[joint - 1] + segment * (
            direction * math.cos(angle) + normal * math.sin(angle)
        )
    points /= [base.aspect, 1, base.aspect]
    return HandFrame(timestamp, tuple(map(tuple, points)), base.aspect, base.handedness, 0.99)


def finger_demo_frame(timestamp: float, elapsed: float, mode: str) -> HandFrame:
    targets = [(0.0, 0.0), (0.10, -0.09), (-0.10, 0.09), (0.08, 0.06)]
    phase = max(0.0, elapsed - 0.6)
    segment = int(phase / 3) % len(targets)
    within = phase % 3
    a, b = targets[segment], targets[(segment + 1) % len(targets)]
    fraction = min(1.0, within / 0.7)
    dx, dy = tuple(v + (w - v) * fraction for v, w in zip(a, b))
    bend = 0.0
    if mode == "finger-flex" and 1.5 < within < 2.3:
        bend = 0.18 * min(1.0, (within - 1.5) / 0.2, (2.3 - within) / 0.2)
    return synthetic_finger(timestamp, dx, dy, bend)
