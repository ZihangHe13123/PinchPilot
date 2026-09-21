import math

import numpy as np

from .domain import Features, HandFrame, Prediction

FEATURE_NAMES = (
    "thumb_index_distance",
    "thumb_middle_distance",
    "thumb_ring_distance",
    "thumb_pinky_distance",
    "index_angle",
    "middle_angle",
    "ring_angle",
    "pinky_angle",
    "index_reach",
    "middle_reach",
    "ring_reach",
    "pinky_reach",
)
FEATURE_VERSION = "geometry-v1-aspect-corrected"
LABELS = ("open", "pinch", "scroll", "other")


def extract(frame: HandFrame) -> Features | None:
    if len(frame.landmarks) != 21 or not math.isfinite(frame.aspect) or frame.aspect <= 0:
        return None
    p = np.asarray(frame.landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    xy = p[:, :2] * [frame.aspect, 1.0]
    scale = (np.linalg.norm(xy[5] - xy[17]) + np.linalg.norm(xy[0] - xy[9])) / 2
    if scale < 0.025:
        return None
    angles, reaches = [], []
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        a, b = xy[mcp] - xy[pip], xy[tip] - xy[pip]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-9:
            return None
        angles.append(float(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1)) / np.pi))
        reaches.append(float(np.linalg.norm(xy[tip] - xy[0]) / scale))
    distances = [float(np.linalg.norm(xy[4] - xy[t]) / scale) for t in [8, 12, 16, 20]]
    palm = p[[0, 5, 9, 13, 17], :2].mean(axis=0)
    scroll = angles[0] > 0.78 and angles[1] > 0.78 and angles[2] < 0.65 and angles[3] < 0.65
    fist = all(a < 0.60 for a in angles)
    return Features(
        tuple(distances + angles + reaches),
        distances[0],
        (float(palm[0]), float(palm[1])),
        scroll,
        fist,
    )


def rule_prediction(f: Features, engage: float = 0.26, release: float = 0.40) -> Prediction:
    if f.fist:
        return Prediction("other", 1.0)
    if f.pinch < engage:
        return Prediction("pinch", 1.0)
    if f.scroll_pose:
        return Prediction("scroll", 1.0)
    if f.pinch > release:
        return Prediction("open", 1.0)
    return Prediction("uncertain", 0.0)
