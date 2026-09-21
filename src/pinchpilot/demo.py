"""Synthetic replay for engineering checks ONLY. Never a training dataset."""

import math

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
