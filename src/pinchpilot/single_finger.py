"""Experimental, app-only finger pointing and clicks. No camera or OS access."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .domain import EngineResult, HandFrame, InputEvent, Prediction
from .filtering import AdaptiveFilter

FINGER_MODES = ("finger-flex", "finger-dwell")


@dataclass(frozen=True)
class FingerConfig:
    mode: str = "finger-flex"
    gain: float = 2.5
    flex_delta: float = 0.12
    dwell_seconds: float = 0.8
    dwell_radius: float = 0.018
    rearm_radius: float = 0.05
    arm_seconds: float = 0.18
    confirm_seconds: float = 0.07
    flex_timeout: float = 1.5
    tracking_timeout: float = 0.20

    def validate(self):
        if self.mode not in FINGER_MODES:
            raise ValueError("未知单指模式")
        if not all(math.isfinite(v) for k, v in vars(self).items() if k != "mode"):
            raise ValueError("单指配置包含无效数值")
        if not 0.5 <= self.gain <= 6 or not 0.05 <= self.flex_delta <= 0.25:
            raise ValueError("单指灵敏度超出范围")
        if not 0.4 <= self.dwell_seconds <= 3:
            raise ValueError("停留时间应为 0.4–3 秒")
        if not 0.005 <= self.dwell_radius < self.rearm_radius <= 0.15:
            raise ValueError("停留与重新准备范围无效")
        if not 0.1 <= self.arm_seconds <= 0.5 or not 0.02 <= self.confirm_seconds <= 0.2:
            raise ValueError("单指确认时间无效")
        if not 0.5 <= self.flex_timeout <= 3 or not 0.05 <= self.tracking_timeout <= 0.5:
            raise ValueError("单指超时配置无效")


@dataclass(frozen=True)
class FingerFeatures:
    relative: tuple[float, float]
    palm: tuple[float, float]
    scale: float
    aspect: float
    bend: float


def finger_features(frame: HandFrame) -> FingerFeatures | None:
    if len(frame.landmarks) != 21 or not math.isfinite(frame.aspect) or frame.aspect <= 0:
        return None
    p = np.asarray(frame.landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    xy = p[:, :2] * [frame.aspect, 1.0]
    scale = float((np.linalg.norm(xy[5] - xy[17]) + np.linalg.norm(xy[0] - xy[9])) / 2)
    if scale < 0.025:
        return None
    palm = p[[0, 5, 9, 13, 17], :2].mean(axis=0)
    relative = (p[8, :2] - palm) * [frame.aspect, 1.0] / scale
    # MediaPipe image-landmark z has roughly the same scale as image x.
    # These are geometric proxies, not measured anatomical joint angles.
    xyz = p * [frame.aspect, 1.0, frame.aspect]
    angles = []
    for a, b, c in ((5, 6, 7), (6, 7, 8)):
        u, v = xyz[a] - xyz[b], xyz[c] - xyz[b]
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm < 1e-9:
            return None
        angles.append(float(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))
    bend = 1 - sum(angles) / (2 * math.pi)
    return FingerFeatures(tuple(relative), tuple(palm), scale, frame.aspect, bend)


class SingleFingerEngine:
    """One active index finger; completed clicks are atomic down/up pairs.

    Coordinates are relative to the palm, in palm-scale units. There are no
    drag/scroll events and no learned labels. The GUI separately blocks OS output.
    """

    def __init__(self, config: FingerConfig | None = None):
        self.config = config or FingerConfig()
        self.config.validate()
        self.enabled = True
        self.pointer = None
        self.reset()

    def reset(self) -> EngineResult:
        self.state = "WAIT_FINGER" if self.enabled else "PAUSED"
        self.last_time = self.last_seen = None
        self.last_hand = ""
        self.features = None
        self.arm_since = None
        self.arm_samples = []
        self.neutral = 0.0
        self.relative_anchor = None
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.filters = [AdaptiveFilter(), AdaptiveFilter()]
        self.stroke_since = self.press_since = self.release_since = None
        self.dwell_since = None
        self.dwell_anchor = None
        self.unlock_anchor = None
        self.dwell_locked = True
        return self._result()

    def set_enabled(self, enabled: bool) -> EngineResult:
        self.enabled = enabled
        return self.reset()

    def tick(self, timestamp: float) -> EngineResult:
        if not math.isfinite(timestamp) or (
            self.last_seen is not None
            and timestamp - self.last_seen >= self.config.tracking_timeout
        ):
            return self.reset()
        # Keep the last observed progress between camera frames, without replaying
        # its click events or advancing a timer on unobserved input.
        return replace(self.last_result, events=[])

    @property
    def active_box(self) -> tuple[float, float, float, float]:
        f = self.features
        if f is None or self.relative_anchor is None:
            return (0.35, 0.2, 0.65, 0.5)
        origin = tuple(
            p + (r - a / self.config.gain) * f.scale / aspect
            for p, r, a, aspect in zip(
                f.palm, self.relative_anchor, self.pointer_anchor, (f.aspect, 1.0)
            )
        )
        return (
            *origin,
            origin[0] + f.scale / (self.config.gain * f.aspect),
            origin[1] + f.scale / self.config.gain,
        )

    def _result(self, events=None, progress=None, hint=None) -> EngineResult:
        hints = {
            "WAIT_FINGER": "舒展食指，保持片刻以定位；其余手指可放松",
            "PAUSED": "已暂停；点击恢复后重新定位",
            "POINT": "小幅移动食指；轻弯再恢复，完成一次点击",
            "FLEX_PENDING": "指针已冻结 · 轻弯食指",
            "FLEX_HOLD": "已识别轻弯 · 恢复食指完成点击",
            "RECOVER": "本次动作已取消 · 恢复食指后继续",
            "MOVE_TO_ARM": "先移开指针，再停住开始倒计时",
            "DWELL": "停留中 · 移开取消",
            "DWELL_LOCKED": "已点击 · 移开指针后才能再次点击",
        }
        result = EngineResult(
            self.state,
            self.pointer,
            events or [],
            Prediction(self.state.lower(), 1.0, "finger-rules") if self.features else None,
            progress=progress,
            hint=hint or hints.get(self.state, ""),
            mode=self.config.mode,
            bend=self.features.bend if self.features else None,
        )
        self.last_result = result
        return result

    def _anchor(self, f: FingerFeatures, t: float):
        self.relative_anchor = f.relative
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.pointer = self.pointer_anchor
        self.filters = [AdaptiveFilter(), AdaptiveFilter()]
        for filt, value in zip(self.filters, self.pointer):
            filt.update(value, t)

    def _move(self, f: FingerFeatures, t: float):
        raw = tuple(
            a + (v - r) * self.config.gain
            for a, v, r in zip(self.pointer_anchor, f.relative, self.relative_anchor)
        )
        self.pointer = tuple(
            min(1.0, max(0.0, filt.update(value, t))) for filt, value in zip(self.filters, raw)
        )
        return raw, InputEvent("move", *self.pointer)

    def _click(self):
        return [
            InputEvent("move", *self.pointer),
            InputEvent("down", *self.pointer),
            InputEvent("up", *self.pointer),
        ]

    def process(self, frame: HandFrame) -> EngineResult:
        t = frame.timestamp
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            return self.reset()
        self.tick(t)
        if not self.enabled:
            return self._result()
        f = finger_features(frame)
        if f is None:
            return self.reset()  # No countdown/click can span even a brief lost-hand frame.
        if self.features is not None and (
            (self.last_hand and frame.handedness and frame.handedness != self.last_hand)
            or math.dist(self.features.palm, f.palm) > 0.35
            or not 0.5 <= f.scale / self.features.scale <= 2.0
        ):
            self.reset()
        self.last_time = self.last_seen = t
        self.features, self.last_hand = f, frame.handedness
        if f.bend > 0.60:
            return self.reset()
        if self.state == "WAIT_FINGER":
            if f.bend > 0.23:
                self.arm_since, self.arm_samples = None, []
                return self._result()
            if self.arm_since is None:
                self.arm_since = t
            self.arm_samples.append(f.bend)
            if t - self.arm_since < self.config.arm_seconds:
                return self._result(hint="正在定位 · 请保持舒适的食指姿势")
            self.neutral = float(np.median(self.arm_samples))
            self._anchor(f, t)
            self.state = "POINT" if self.config.mode == "finger-flex" else "MOVE_TO_ARM"
            self.unlock_anchor = self.pointer
            return self._result([InputEvent("move", *self.pointer)])
        if self.config.mode == "finger-flex":
            return self._flex(f, t)
        return self._dwell(f, t)

    def _flex(self, f: FingerFeatures, t: float) -> EngineResult:
        delta = f.bend - self.neutral
        release = self.config.flex_delta * 0.30
        onset = self.config.flex_delta * 0.40
        if self.state == "RECOVER":
            if delta <= release:
                self._anchor(f, t)
                self.state = "POINT"
            return self._result()
        if self.state == "POINT":
            if delta < onset:
                _, move = self._move(f, t)
                return self._result([move])
            self.state = "FLEX_PENDING"
            self.stroke_since = t
            self.press_since = self.release_since = None
        # Freeze BEFORE moving this frame: flex motion must not steer the click.
        if t - self.stroke_since > self.config.flex_timeout:
            self.state = "RECOVER"
            return self._result()
        if self.state == "FLEX_PENDING":
            if delta >= self.config.flex_delta:
                if self.press_since is None:
                    self.press_since = t
                if t - self.press_since >= self.config.confirm_seconds:
                    self.state = "FLEX_HOLD"
            else:
                self.press_since = None
                if delta <= release:
                    self._anchor(f, t)
                    self.state = "POINT"
            return self._result(progress=0.5 if self.state == "FLEX_HOLD" else None)
        if delta <= release:
            if self.release_since is None:
                self.release_since = t
            if t - self.release_since >= self.config.confirm_seconds:
                events = self._click()
                self._anchor(f, t)
                self.state = "POINT"
                return self._result(events, hint="点击完成 · 继续小幅定位")
        else:
            self.release_since = None
        return self._result(progress=0.5)

    def _dwell(self, f: FingerFeatures, t: float) -> EngineResult:
        if abs(f.bend - self.neutral) > 0.10:
            return self.reset()
        raw, move = self._move(f, t)
        if self.dwell_locked:
            if math.dist(raw, self.unlock_anchor) <= self.config.rearm_radius:
                return self._result([move])
            self.dwell_locked = False
            self.dwell_anchor, self.dwell_since = raw, t
            self.state = "DWELL"
        # Use unfiltered, UNCLAMPED input and a fixed anchor. Slowly drifting
        # input, including motion beyond a screen edge, cannot masquerade as rest.
        if not all(0 <= v <= 1 for v in raw):
            self.dwell_anchor, self.dwell_since = None, None
            return self._result([move], hint="已到边缘 · 移回范围内再停留")
        if (
            self.dwell_anchor is None
            or math.dist(raw, self.dwell_anchor) > self.config.dwell_radius
        ):
            self.dwell_anchor, self.dwell_since = raw, t
        progress = min(1.0, (t - self.dwell_since) / self.config.dwell_seconds)
        if progress < 1:
            return self._result([move], progress)
        events = self._click()
        self.state = "DWELL_LOCKED"
        self.dwell_locked = True
        self.unlock_anchor = raw
        self.dwell_anchor = self.dwell_since = None
        return self._result(events, 1.0)
