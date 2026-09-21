"""Thumb-middle grip points; index approaches the pair to click, once per tap."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .domain import EngineResult, HandFrame, InputEvent, Prediction
from .filtering import StablePointer


@dataclass(frozen=True)
class TripodConfig:
    mode: str = "tripod"
    span: float = 0.30
    deadband: float = 0.008
    filter_cutoff: float = 3.0
    filter_beta: float = 4.0
    grip_engage: float = 0.28
    grip_release: float = 0.46
    touch_ratio: float = 0.26
    hover_ratio: float = 0.38
    clear_ratio: float = 0.48
    arm_seconds: float = 0.12
    confirm_seconds: float = 0.06
    tracking_timeout: float = 0.20
    max_jump: float = 0.10

    def validate(self):
        if self.mode != "tripod" or not all(
            math.isfinite(v) for k, v in vars(self).items() if k != "mode"
        ):
            raise ValueError("三指配置无效")
        if not 0.15 <= self.span <= 0.60 or not 0 <= self.deadband <= 0.03:
            raise ValueError("三指移动范围或抗抖强度无效")
        if not 0.5 <= self.filter_cutoff <= 10 or not 0 <= self.filter_beta <= 20:
            raise ValueError("三指滤波参数无效")
        if not 0.1 <= self.grip_engage < self.grip_release <= 0.8:
            raise ValueError("拇中捏合阈值无效")
        if not 0.08 <= self.touch_ratio < self.hover_ratio < self.clear_ratio <= 1:
            raise ValueError("食指触碰阈值无效")
        if not 0.06 <= self.arm_seconds <= 0.5 or not 0.03 <= self.confirm_seconds <= 0.2:
            raise ValueError("三指确认时长无效")
        if not 0.1 <= self.tracking_timeout <= 0.5 or not 0.04 <= self.max_jump <= 0.3:
            raise ValueError("三指追踪中断配置无效")


@dataclass(frozen=True)
class TripodFeatures:
    point: tuple[float, float]
    scale: float
    grip: float
    contact: float


def tripod_features(frame: HandFrame) -> TripodFeatures | None:
    if len(frame.landmarks) != 21 or not math.isfinite(frame.aspect) or frame.aspect <= 0:
        return None
    p = np.asarray(frame.landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    xyz = p * [frame.aspect, 1.0, frame.aspect]
    scale = float(
        (np.linalg.norm(xyz[5, :2] - xyz[17, :2]) + np.linalg.norm(xyz[0, :2] - xyz[9, :2])) / 2
    )
    if scale < 0.025:
        return None
    grip = float(np.linalg.norm(xyz[4] - xyz[12]) / scale)
    # Require proximity to BOTH tips, including estimated depth. This is a
    # landmark-distance proxy, not a physical contact/pressure sensor.
    contact = float(max(np.linalg.norm(xyz[8] - xyz[4]), np.linalg.norm(xyz[8] - xyz[12])) / scale)
    return TripodFeatures(tuple((p[4, :2] + p[12, :2]) / 2), scale, grip, contact)


class TripodEngine:
    def __init__(self, config: TripodConfig | None = None):
        self.config = config or TripodConfig()
        self.config.validate()
        self.enabled = True
        self.pointer = None
        self.reset()

    def reset(self) -> EngineResult:
        self.state = "WAIT_GRIP" if self.enabled else "PAUSED"
        self.last_time = self.last_seen = None
        self.last_hand = ""
        self.features = None
        self.camera_anchor = None
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.raw_pointer = None
        self.arm_since = self.touch_since = self.clear_since = None
        self.stabilizer = None
        return self._result()

    def set_enabled(self, enabled):
        self.enabled = enabled
        return self.reset()

    def tick(self, timestamp):
        if not math.isfinite(timestamp) or (
            self.last_seen is not None
            and timestamp - self.last_seen >= self.config.tracking_timeout
        ):
            return self.reset()
        return replace(self.last_result, events=[])

    @property
    def active_box(self):
        span = self.config.span
        center = self.camera_anchor or (0.5, 0.5)
        left, top = tuple(c - p * span for c, p in zip(center, self.pointer_anchor))
        return left, top, left + span, top + span

    def _result(self, events=None, hint=None):
        hints = {
            "WAIT_GRIP": "拇指与中指捏住，食指移开，接管指针",
            "CONTROL": "移动捏合点定位 · 食指碰入点击 · 松开拇中即休息",
            "APPROACH": "食指接近 · 指针已冻结 · 再靠近一点点击",
            "TOUCHED": "已点击 · 食指移开后继续定位",
            "PAUSED": "已暂停 · 恢复后重新捏住拇中",
        }
        f = self.features
        self.last_result = EngineResult(
            self.state,
            self.pointer,
            events or [],
            Prediction(self.state.lower(), 1.0, "tripod-rules") if f else None,
            hint=hint or hints[self.state],
            mode="tripod",
            grip=f.grip if f else None,
            contact=f.contact if f else None,
            raw_pointer=self.raw_pointer,
        )
        return self.last_result

    def _anchor(self, f, timestamp):
        self.camera_anchor = f.point
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.pointer = self.pointer_anchor
        self.raw_pointer = self.pointer_anchor
        cfg = self.config
        self.stabilizer = StablePointer(cfg.deadband, cfg.filter_cutoff, cfg.filter_beta)
        self.stabilizer.update(self.pointer, timestamp)

    def _disarm(self, f, frame, hint=None):
        self.reset()
        self.features = f
        self.last_time = self.last_seen = frame.timestamp
        self.last_hand = frame.handedness
        return self._result(hint=hint)

    def process(self, frame: HandFrame) -> EngineResult:
        t = frame.timestamp
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            return self.reset()
        self.tick(t)
        if not self.enabled:
            return self._result()
        f = tripod_features(frame)
        if f is None:
            return self.reset()
        if self.features is not None and (
            (self.last_hand and frame.handedness and self.last_hand != frame.handedness)
            or math.dist(f.point, self.features.point) > self.config.max_jump
        ):
            return self._disarm(f, frame, "追踪发生跳变 · 暂停移动，请移开食指重新捏住")
        self.last_time = self.last_seen = t
        self.features, self.last_hand = f, frame.handedness
        cfg = self.config
        if self.state == "WAIT_GRIP":
            if f.grip > cfg.grip_engage or f.contact < cfg.clear_ratio:
                self.arm_since = None
                return self._result()
            if self.arm_since is None:
                self.arm_since = t
            if t - self.arm_since < cfg.arm_seconds:
                return self._result(hint="正在接管 · 保持拇中捏合，食指在外侧")
            self._anchor(f, t)
            self.state = "CONTROL"
            return self._result([InputEvent("move", *self.pointer)])
        if f.grip > cfg.grip_release:
            return self._disarm(f, frame, "拇中已松开 · 指针停住，再捏合可继续")
        self.raw_pointer = tuple(
            a + (v - c) / cfg.span
            for a, v, c in zip(self.pointer_anchor, f.point, self.camera_anchor)
        )
        if self.state == "CONTROL":
            if f.contact > cfg.hover_ratio:
                point = self.stabilizer.update(self.raw_pointer, t)
                self.pointer = tuple(min(1.0, max(0.0, v)) for v in point)
                return self._result([InputEvent("move", *self.pointer)])
            # Freeze the previous pointer before this frame's contacting hand can
            # move it. Index coordinates NEVER enter the pointing calculation.
            self.state = "APPROACH"
            self.touch_since = self.clear_since = None
        if f.contact >= cfg.clear_ratio:
            self.touch_since = None
            if self.clear_since is None:
                self.clear_since = t
            if t - self.clear_since >= cfg.confirm_seconds:
                self._anchor(f, t)
                self.state = "CONTROL"
                self.clear_since = None
                return self._result([InputEvent("move", *self.pointer)])
            return self._result()
        self.clear_since = None
        if self.state == "APPROACH":
            if f.contact <= cfg.touch_ratio:
                if self.touch_since is None:
                    self.touch_since = t
                if t - self.touch_since >= cfg.confirm_seconds:
                    self.state = "TOUCHED"
                    return self._result(
                        [
                            InputEvent("move", *self.pointer),
                            InputEvent("down", *self.pointer),
                            InputEvent("up", *self.pointer),
                        ]
                    )
            else:
                self.touch_since = None
        return self._result()
