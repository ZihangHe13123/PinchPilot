"""Opt-in stationary noise measurement; samples stay in memory and never emit input."""

import math

import numpy as np

from .motion import MAX_CALIBRATED_RADIUS, ScreenSmoother
from .tripod import tripod_features


class RestNoiseCalibration:
    PREPARE_SECONDS = 1.0
    RECORD_SECONDS = 5.0
    DURATION = PREPARE_SECONDS + RECORD_SECONDS

    def __init__(self, started, config):
        self.started, self.config = started, config
        self.samples = []
        self.observed = 0

    def progress(self, now):
        return min(1.0, max(0.0, (now - self.started) / self.DURATION))

    def add(self, frame):
        elapsed = frame.timestamp - self.started
        if not self.PREPARE_SECONDS <= elapsed <= self.DURATION:
            return
        self.observed += 1
        features = tripod_features(frame)
        if (
            features is not None
            and features.grip <= self.config.grip_release
            and features.contact >= self.config.clear_ratio
            and features.right_contact >= self.config.right_clear_ratio
        ):
            self.samples.append((frame.timestamp, features.point))

    def result(self):
        if len(self.samples) < 60 or len(self.samples) / max(1, self.observed) < 0.8:
            raise ValueError("有效静止样本不足；保持拇中捏合，食指和无名指移开后重试")
        times = np.array([row[0] for row in self.samples])
        points = np.array([row[1] for row in self.samples])
        if times[-1] - times[0] < 4.0 or np.max(np.diff(times)) > 0.25:
            raise ValueError("相机采样不连续，请保持控制手可见后重试")
        edge = max(5, len(points) // 5)
        drift = np.linalg.norm(np.median(points[-edge:], axis=0) - np.median(points[:edge], axis=0))
        spread = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
        if drift > 0.02 or np.max(spread) > 0.04:
            raise ValueError("校准期间手移动过多，请支撑前臂、保持捏合点静止后重试")
        cfg = self.config
        drift_px = np.linalg.norm(
            (np.median(points[-edge:], axis=0) - np.median(points[:edge], axis=0))
            * np.array([cfg.screen_width, cfg.screen_height])
            / cfg.span
        )
        if drift_px > 6.0:
            raise ValueError("校准期间有持续漂移，请保持捏合点静止后重试；保留原有校准")
        smoother = ScreenSmoother(cfg.screen_width, cfg.screen_height)
        smooth = np.array(
            [smoother.update(tuple(v / cfg.span for v in point), t) for t, point in self.samples]
        )[5:]
        noise = np.percentile(np.abs(smooth - np.median(smooth, axis=0)), 95, axis=0)
        noise_px = math.hypot(*noise)
        if noise_px > 12:
            raise ValueError("静止波动太大，请先调整支撑、光照或相机位置；保留原有校准")
        return {
            "rest_noise_x": float(noise[0] * cfg.span / cfg.screen_width),
            "rest_noise_y": float(noise[1] * cfg.span / cfg.screen_height),
            "noise_px": round(noise_px, 3),
            "samples": len(self.samples),
            "valid_fraction": round(len(self.samples) / self.observed, 3),
            "limited": noise_px * 1.1 > MAX_CALIBRATED_RADIUS,
            "screen_width": cfg.screen_width,
            "screen_height": cfg.screen_height,
            "span": cfg.span,
            "scope": "stationary filtered spread; not accuracy or intent labels",
        }
