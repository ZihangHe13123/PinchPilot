"""Thumb-middle points, thumb-index presses/drags, thumb-ring right-clicks."""

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
    drag_hold_seconds: float = 0.30
    right_touch_ratio: float = 0.26
    right_hover_ratio: float = 0.38
    right_clear_ratio: float = 0.48
    right_confirm_seconds: float = 0.10
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
            raise ValueError("食拇触碰阈值无效")
        if not 0.06 <= self.arm_seconds <= 0.5 or not 0.03 <= self.confirm_seconds <= 0.2:
            raise ValueError("三指确认时长无效")
        if not 0.15 <= self.drag_hold_seconds <= 0.8:
            raise ValueError("拖拽等待时长无效")
        if (
            not 0.08
            <= self.right_touch_ratio
            < self.right_hover_ratio
            < self.right_clear_ratio
            <= 1
        ):
            raise ValueError("右键触碰阈值无效")
        if not 0.06 <= self.right_confirm_seconds <= 0.3:
            raise ValueError("右键确认时长无效")
        if not 0.1 <= self.tracking_timeout <= 0.5 or not 0.04 <= self.max_jump <= 0.3:
            raise ValueError("三指追踪中断配置无效")


@dataclass(frozen=True)
class TripodFeatures:
    point: tuple[float, float]
    palm: tuple[float, float]
    scale: float
    grip: float
    contact: float
    right_contact: float


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
    # Middle finger gates movement only. Contact is a thumb-index distance
    # proxy with estimated depth, not a physical contact/pressure sensor.
    contact = float(np.linalg.norm(xyz[8] - xyz[4]) / scale)
    return TripodFeatures(
        tuple((p[4, :2] + p[12, :2]) / 2),
        tuple(p[[0, 5, 9, 13, 17], :2].mean(axis=0)),
        scale,
        grip,
        contact,
        float(np.linalg.norm(xyz[16] - xyz[4]) / scale),
    )


class TripodEngine:
    def __init__(self, config: TripodConfig | None = None):
        self.config = config or TripodConfig()
        self.config.validate()
        self.enabled = True
        self.pointer = None
        self.left_down = False
        self.reset()

    def reset(self) -> EngineResult:
        cancelled = self.left_down
        events = [InputEvent("up", *(self.pointer or (0.5, 0.5)))] if cancelled else []
        self.left_down = False
        self.press_since = None
        self.state = "WAIT_GRIP" if self.enabled else "PAUSED"
        self.last_time = self.last_seen = None
        self.last_hand = ""
        self.features = None
        self.camera_anchor = None
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.raw_pointer = None
        self.motion_engaged = False
        self.right_ready = False
        self.right_clear_since = None
        self.require_both_clear = False
        self.arm_since = self.touch_since = self.clear_since = None
        self.stabilizer = None
        return self._result(events, cancelled=cancelled)

    def set_enabled(self, enabled):
        self.enabled = enabled
        return self.reset()

    def tick(self, timestamp):
        if not math.isfinite(timestamp) or (
            self.last_seen is not None
            and timestamp - self.last_seen >= self.config.tracking_timeout
        ):
            return self.reset()
        return replace(self.last_result, events=[], cancelled=False)

    @property
    def active_box(self):
        span = self.config.span
        center = self.camera_anchor or (0.5, 0.5)
        left, top = tuple(c - p * span for c, p in zip(center, self.pointer_anchor))
        return left, top, left + span, top + span

    def _result(self, events=None, hint=None, cancelled=False):
        hints = {
            "WAIT_GRIP": "拇指与中指捏住，食指移开，接管指针",
            "CONTROL": "拇中定位 · 食拇短捏左键、保持拖拽 · 拇无名指右键",
            "FROZEN": "位置已锁住 · 仍可左/右键 · 移开食指、捏中指继续移动",
            "APPROACH": "食指接近拇指 · 指针已冻结 · 继续靠近按下",
            "PRESSED": "左键已按下 · 松食指单击 · 保持到进度环满可拖拽",
            "DRAG": (
                "拖拽已就绪 · 保持食拇捏合移动 · 松食指放下"
                if self.motion_engaged
                else "拖拽位置锁住 · 捏中指可继续拖 · 松食指放下"
            ),
            "RIGHT_APPROACH": "无名指接近拇指 · 位置已冻结 · 食指保持分开",
            "RIGHT_TOUCHED": "右键已触发 · 松开无名指后可继续操作",
            "WAIT_CLEAR": (
                "动作已取消 · 请先分开食指、无名指与拇指，再试"
                if self.require_both_clear
                else "位置已锁住 · 请先分开食指和拇指，再捏合点击"
            ),
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
            right_contact=f.right_contact if f else None,
            progress=(
                min(1.0, (self.last_time - self.press_since) / self.config.drag_hold_seconds)
                if self.state == "PRESSED" and self.press_since is not None
                else None
            ),
            cancelled=cancelled,
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
        stopped = self.reset()
        self.features = f
        self.last_time = self.last_seen = frame.timestamp
        self.last_hand = frame.handedness
        return self._result(stopped.events, hint, cancelled=stopped.cancelled)

    def _move(self, timestamp):
        point = self.stabilizer.update(self.raw_pointer, timestamp)
        self.pointer = tuple(min(1.0, max(0.0, v)) for v in point)
        return self._result([InputEvent("move", *self.pointer)])

    def _idle(self, f, timestamp, events=None):
        self.touch_since = self.clear_since = self.arm_since = None
        self.require_both_clear = False
        events = list(events or [])
        if self.motion_engaged:
            self._anchor(f, timestamp)
            self.state = "CONTROL"
            events.append(InputEvent("move", *self.pointer))
        else:
            self.state = "FROZEN"
        return self._result(events)

    def _wait_clear(self, both=False):
        self.state = "WAIT_CLEAR"
        self.require_both_clear = both
        self.touch_since = self.clear_since = self.arm_since = None
        self.right_ready = False
        return self._result()

    def _update_right_ready(self, f, timestamp):
        cfg = self.config
        if self.left_down or f.contact < cfg.clear_ratio:
            self.right_ready = False
            self.right_clear_since = None
        elif f.right_contact >= cfg.right_clear_ratio:
            if self.right_clear_since is None:
                self.right_clear_since = timestamp
            if timestamp - self.right_clear_since >= cfg.confirm_seconds:
                self.right_ready = True
        else:
            self.right_clear_since = None

    def _pressed(self, f, timestamp):
        cfg = self.config
        if f.contact >= cfg.clear_ratio:
            # Freeze from the first separating frame, before confirming release.
            if self.clear_since is None:
                self.clear_since = timestamp
            if timestamp - self.clear_since >= cfg.confirm_seconds:
                self.left_down = False
                self.press_since = None
                return self._idle(f, timestamp, [InputEvent("up", *self.pointer)])
            return self._result()
        self.clear_since = None
        if self.state == "PRESSED" and timestamp - self.press_since >= cfg.drag_hold_seconds:
            # Discard finger-closing motion before allowing held movement.
            self.state = "DRAG"
            if self.motion_engaged:
                self._anchor(f, timestamp)
            return self._result()
        if self.state == "DRAG" and self.motion_engaged:
            return self._move(timestamp)
        return self._result()

    def _right(self, f, timestamp):
        cfg = self.config
        if f.contact < cfg.clear_ratio:
            return self._wait_clear(both=True)
        if f.right_contact >= cfg.right_clear_ratio:
            self.touch_since = None
            if self.clear_since is None:
                self.clear_since = timestamp
            if timestamp - self.clear_since >= cfg.confirm_seconds:
                return self._idle(f, timestamp)
            return self._result()
        self.clear_since = None
        if self.state == "RIGHT_APPROACH":
            if f.right_contact <= cfg.right_touch_ratio:
                if self.touch_since is None:
                    self.touch_since = timestamp
                if timestamp - self.touch_since >= cfg.right_confirm_seconds:
                    self.state = "RIGHT_TOUCHED"
                    self.right_ready = False
                    return self._result(
                        [
                            InputEvent("move", *self.pointer),
                            InputEvent("right_down", *self.pointer),
                            InputEvent("right_up", *self.pointer),
                        ]
                    )
            else:
                self.touch_since = None
        return self._result()

    def process(self, frame: HandFrame) -> EngineResult:
        t = frame.timestamp
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            return self.reset()
        expired = self.tick(t)
        result = self._process(frame)
        # A returned frame must not swallow an earlier timeout's button release.
        result.events = expired.events + result.events
        result.cancelled = expired.cancelled or result.cancelled
        return result

    def _process(self, frame):
        t = frame.timestamp
        if not self.enabled:
            return self._result()
        f = tripod_features(frame)
        if f is None:
            return self.reset()
        if self.features is not None and (
            (self.last_hand and frame.handedness and self.last_hand != frame.handedness)
            or math.dist(f.palm, self.features.palm) > self.config.max_jump
            or (
                self.motion_engaged
                and f.grip <= self.config.grip_release
                and math.dist(f.point, self.features.point) > self.config.max_jump
            )
        ):
            return self._disarm(f, frame, "追踪发生跳变 · 暂停移动，请移开食指重新捏住")
        self.last_time = self.last_seen = t
        self.features, self.last_hand = f, frame.handedness
        cfg = self.config
        right_near = self.right_ready and f.right_contact <= cfg.right_hover_ratio
        self._update_right_ready(f, t)
        if self.state == "WAIT_GRIP":
            if f.grip > cfg.grip_engage or f.contact < cfg.clear_ratio:
                self.arm_since = None
                return self._result()
            if self.arm_since is None:
                self.arm_since = t
            if t - self.arm_since < cfg.arm_seconds:
                return self._result(hint="正在接管 · 保持拇中捏合，食指在外侧")
            self._anchor(f, t)
            self.motion_engaged = True
            self.state = "CONTROL"
            return self._result([InputEvent("move", *self.pointer)])
        if self.motion_engaged and f.grip > cfg.grip_release:
            # Releasing the middle finger freezes before any midpoint update.
            # Do not turn a contact that started during this transition into a
            # click, or rearm a contact that has already clicked.
            self.motion_engaged = False
            self.raw_pointer = None
            self.arm_since = self.touch_since = self.clear_since = None
            if not self.left_down:
                if self.state == "RIGHT_TOUCHED":
                    return self._result()
                if self.state == "RIGHT_APPROACH" or self.require_both_clear or right_near:
                    return self._wait_clear(both=True)
                self.state = "FROZEN" if f.contact >= cfg.clear_ratio else "WAIT_CLEAR"
                return self._result()
        if not self.motion_engaged:
            if f.grip <= cfg.grip_engage and (
                (self.left_down and f.contact < cfg.clear_ratio)
                or (self.state == "FROZEN" and f.contact >= cfg.clear_ratio and not right_near)
            ):
                if self.arm_since is None:
                    self.arm_since = t
                if t - self.arm_since >= cfg.arm_seconds:
                    self._anchor(f, t)
                    self.motion_engaged = True
                    self.arm_since = None
                    if not self.left_down:
                        self.state = "CONTROL"
                        return self._result([InputEvent("move", *self.pointer)])
            else:
                self.arm_since = None
        else:
            self.raw_pointer = tuple(
                a + (v - c) / cfg.span
                for a, v, c in zip(self.pointer_anchor, f.point, self.camera_anchor)
            )
        if self.left_down:
            return self._pressed(f, t)
        if self.state in ("RIGHT_APPROACH", "RIGHT_TOUCHED"):
            return self._right(f, t)
        if self.state in ("CONTROL", "FROZEN"):
            if right_near:
                if f.contact < cfg.clear_ratio:
                    return self._wait_clear(both=True)
                self.state = "RIGHT_APPROACH"
                self.touch_since = self.clear_since = None
                return self._right(f, t)
            if f.contact > cfg.hover_ratio:
                if self.motion_engaged:
                    return self._move(t)
                return self._result()
            # Freeze the previous pointer before this frame's contacting hand can
            # move it. Index coordinates NEVER enter the pointing calculation.
            self.state = "APPROACH"
            self.touch_since = self.clear_since = None
        if f.contact >= cfg.clear_ratio and (
            not self.require_both_clear or f.right_contact >= cfg.right_clear_ratio
        ):
            self.touch_since = None
            if self.clear_since is None:
                self.clear_since = t
            if t - self.clear_since >= cfg.confirm_seconds:
                return self._idle(f, t)
            return self._result()
        self.clear_since = None
        if self.state == "APPROACH":
            if f.contact <= cfg.touch_ratio:
                if self.touch_since is None:
                    self.touch_since = t
                if t - self.touch_since >= cfg.confirm_seconds:
                    self.state = "PRESSED"
                    self.left_down = True
                    self.press_since = t
                    self.right_ready = False
                    return self._result(
                        [
                            InputEvent("move", *self.pointer),
                            InputEvent("down", *self.pointer),
                        ]
                    )
            else:
                self.touch_since = None
        return self._result()
