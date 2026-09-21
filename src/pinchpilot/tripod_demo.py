"""Generated landmarks only: exercises the gesture, never training evidence."""

import math

import numpy as np

from .demo import synthetic_hand
from .domain import HandFrame


def synthetic_tripod(timestamp, x=0.5, y=0.38, grip=True, contact=0.85, noise=(0, 0)):
    base = synthetic_hand(timestamp)
    p = np.array(base.landmarks)
    xy = p[:, :2] * [base.aspect, 1]
    scale = (np.linalg.norm(xy[5] - xy[17]) + np.linalg.norm(xy[0] - xy[9])) / 2
    gap = (0.10 if grip else 0.75) * scale / base.aspect
    for tip, sign in ((4, -1), (12, 1)):
        p[tip] = (x + sign * gap / 2 + noise[0], y + noise[1], 0)
    p[8] = (x + noise[0], y - contact * scale + noise[1], 0)
    for previous, tip in ((3, 4), (7, 8), (11, 12)):
        p[previous] = (p[previous - 1] + p[tip]) / 2
    return HandFrame(timestamp, tuple(map(tuple, p)), base.aspect, "Right", 0.99)


def tripod_demo_frame(timestamp, elapsed):
    phase = elapsed % 10
    if phase >= 8.5:
        return synthetic_tripod(timestamp, grip=False)
    x = 0.5 + 0.075 * math.sin(max(0, elapsed - 0.6) * 0.7)
    y = 0.38 + 0.045 * math.sin(max(0, elapsed - 0.6) * 1.1)
    contact = 0.12 if 3.2 < phase < 4.0 or 6.2 < phase < 6.6 else 0.85
    if 2.9 < phase <= 3.2 or 6.0 < phase <= 6.2:
        contact = 0.32
    noise = (0.0007 * math.sin(elapsed * 36), 0.0007 * math.cos(elapsed * 42))
    return synthetic_tripod(timestamp, x, y, contact=contact, noise=noise)
