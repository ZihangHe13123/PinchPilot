"""Geometry on windows of contact features that training and live control share. NumPy only.

What the tuned rule measures, and the per-finger views the finger-shared CNN reads. Kept in
one place so that both sides compute them the same way.
"""

import numpy as np

from .contact_data import CHANNELS

THUMB_TIP = 4
# MediaPipe fingertip behind each output, in the order of CHANNELS.
FINGERTIPS = (12, 8, 16)
STATISTICS = ("last", "median", "mean", "min")


def ratios(x, options):
    """Thumb tip to each fingertip in 3-D over the recent frames, as one number per finger."""
    recent = x[:, -options["frames"] :, 63:66]
    if options["statistic"] == "last":
        return recent[:, -1]
    return getattr(np, options["statistic"])(recent, axis=1)


def distances(x, options):
    """What the rule compares with its cuts, [N, 3] in the order of CHANNELS.

    By default the 3-D distance ratio of the features. For a finger set to flat, the distance
    within the image plane instead: the same two points and the same palm unit, without the
    estimated depth.
    """
    flat = [options.get(f"flat_{channel}", False) for channel in CHANNELS]
    if not any(flat):
        return ratios(x, options)
    recent = x[:, -options["frames"] :]
    values = recent[:, :, 63:66].copy()
    points = recent[:, :, :63].reshape(len(x), recent.shape[1], 21, 3)
    for i, tip in enumerate(FINGERTIPS):
        if flat[i]:
            gap = points[:, :, THUMB_TIP, :2] - points[:, :, tip, :2]
            values[:, :, i] = np.linalg.norm(gap, axis=-1)
    if options["statistic"] == "last":
        return values[:, -1]
    return getattr(np, options["statistic"])(values, axis=1)


def nearer_only(touching, ratio):
    """Where index and ring both come out as touching, keep the one nearer to the thumb."""
    touching = touching.copy()
    both = touching[:, 1] & touching[:, 2]
    touching[both & (ratio[:, 1] > ratio[:, 2]), 1] = False
    touching[both & (ratio[:, 1] <= ratio[:, 2]), 2] = False
    return touching


def decide(x, options, thresholds):
    """Touching or not for each finger, [N, 3] booleans, with cuts fitted earlier."""
    touching = distances(x, options) <= np.asarray(thresholds)
    if options.get("nearest"):
        touching = nearer_only(touching, ratios(x, options))
    return touching


THUMB = (1, 2, 3, 4)
# MediaPipe joints of the finger behind each output, in the order of CHANNELS.
FINGERS = ((9, 10, 11, 12), (5, 6, 7, 8), (13, 14, 15, 16))
FINGER_FEATURES = 29


def finger_inputs(x):
    """[N, T, 67] windows -> [N, 3, T, 29]: for each finger, only the thumb and that finger.

    Per frame: the thumb's four joints (12), the finger's four joints (12), fingertip minus
    thumb tip (3), that finger's distance ratio (1) and the frame interval (1).
    """
    count, frames, _ = x.shape
    points = x[:, :, :63].reshape(count, frames, 21, 3)
    thumb = points[:, :, THUMB, :].reshape(count, frames, 12)
    views = []
    for channel, joints in enumerate(FINGERS):
        finger = points[:, :, joints, :]
        views.append(
            np.concatenate(
                (
                    thumb,
                    finger.reshape(count, frames, 12),
                    finger[:, :, 3] - points[:, :, 4],
                    x[:, :, 63 + channel : 64 + channel],
                    x[:, :, 66:67],
                ),
                axis=2,
            )
        )
    return np.stack(views, axis=1).astype(np.float32)
