"""Small image-only palm assistance for the same thumb-middle pointer point."""

import math

import numpy as np

from .filtering import AdaptiveFilter

PALM = (0, 5, 9, 13, 17)
MAX_CORRECTION_PX = 3.0


class CompatiblePoint:
    """Estimate one point, never add angular motion to observed displacement.

    A 2-D similarity fit predicts rigid palm movement. Only its non-rigid
    fingertip residual is filtered, with a small screen-space correction cap.
    Depth cannot create pointer motion. Poor palm geometry disables assistance,
    not tracking; the actual midpoint remains available to the gesture engine.
    """

    def __init__(self, frame, point, config):
        self.aspect = frame.aspect
        self.pixel_scale = np.array((config.screen_width, config.screen_height)) / config.span
        self.point = np.array(point) * [self.aspect, 1.0]
        palm = self._palm(frame)
        self.center = palm.mean(axis=0)
        self.reference = palm - self.center
        self.energy = float(np.sum(self.reference**2))
        eigenvalues = np.linalg.eigvalsh(self.reference.T @ self.reference)
        self.usable = eigenvalues[0] > 0.02 * eigenvalues[-1] and self.energy > 0.0005
        self.filters = [AdaptiveFilter(4.0, 0.0) for _ in range(2)]
        for f in self.filters:
            f.update(0.0, frame.timestamp)
        self.quality = 0.0
        self.correction_px = 0.0
        self.assisting = True

    @staticmethod
    def _palm(frame):
        return np.asarray(frame.landmarks, dtype=float)[list(PALM), :2] * [frame.aspect, 1.0]

    def update(self, frame, point):
        actual = np.array(point, dtype=float)
        self.quality = self.correction_px = 0.0
        if not self.usable or not math.isclose(frame.aspect, self.aspect):
            return tuple(actual)
        palm = self._palm(frame)
        center = palm.mean(axis=0)
        observed = palm - center
        # Row-vector proper rotation: reflections are not an alternate control axis.
        u, singular, vt = np.linalg.svd(self.reference.T @ observed)
        sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
        rotation = u @ np.diag((1.0, sign)) @ vt
        scale = float((singular[0] + sign * singular[1]) / self.energy)
        predicted = scale * self.reference @ rotation
        radius = math.sqrt(float(np.mean(np.sum(observed**2, axis=1))))
        error = math.sqrt(float(np.mean(np.sum((predicted - observed) ** 2, axis=1))))
        quality = max(0.0, min(1.0, 1.0 - error / max(radius * 0.12, 1e-9)))
        if not 0.4 <= scale <= 2.5 or radius < 0.012:
            quality = 0.0
        if quality == 0.0:
            # Invalid fits must not accumulate a fictitious residual that kicks
            # the pointer when the palm becomes observable again.
            self.assisting = False
            return tuple(actual)
        virtual = (center + scale * (self.point - self.center) @ rotation) / [self.aspect, 1.0]
        residual = actual - virtual
        if not self.assisting:
            self.filters = [AdaptiveFilter(4.0, 0.0) for _ in range(2)]
            self.assisting = True
        smooth = np.array([f.update(v, frame.timestamp) for f, v in zip(self.filters, residual)])
        correction = (smooth - residual) * self.pixel_scale * quality
        distance = float(np.linalg.norm(correction))
        correction *= min(1.0, MAX_CORRECTION_PX / max(distance, 1e-12))
        self.quality = quality
        self.correction_px = float(np.linalg.norm(correction))
        return tuple(actual + correction / self.pixel_scale)
