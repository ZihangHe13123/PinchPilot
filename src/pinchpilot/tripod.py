"""Thumb-middle points, thumb-index presses/drags, thumb-ring right-clicks."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .compatible import CompatiblePoint
from .domain import EngineResult, HandFrame, InputEvent, Prediction
from .features import extract
from .filtering import StablePointer
from .motion import TunedPointer
from .wrist import (
    MAX_FRAME_ANGLE,
    WristMapping,
    palm_rotation,
    rotation_vector,
    validate_wrist_calibration,
)


@dataclass(frozen=True)
class TripodConfig:
    mode: str = "tripod"
    motion_profile: str = "classic"
    pointer_basis: str = "position"
    wrist_calibration: tuple[float, ...] = ()
    screen_width: float = 1920.0
    screen_height: float = 1080.0
    rest_noise_x: float = 0.0
    rest_noise_y: float = 0.0
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
    right_enabled: bool = True
    drag_enabled: bool = True
    scroll_enabled: bool = True
    scroll_gain: float = 60.0
    scroll_confirm_seconds: float = 0.18
    scroll_activation_distance: float = 0.015
    scroll_deadband: float = 0.004
    tracking_timeout: float = 0.20
    max_jump: float = 0.10

    def validate(self):
        validate_wrist_calibration(self.wrist_calibration)
        if not all(
            isinstance(v, bool)
            for v in (self.right_enabled, self.drag_enabled, self.scroll_enabled)
        ):
            raise ValueError("右键、拖拽和滚轮开关必须为布尔值")
        if (
            self.mode != "tripod"
            or self.motion_profile not in ("classic", "precise", "adaptive")
            or self.pointer_basis not in ("position", "wrist", "unified")
            or not all(
                math.isfinite(v)
                for k, v in vars(self).items()
                if k not in ("mode", "motion_profile", "pointer_basis", "wrist_calibration")
            )
        ):
            raise ValueError("三指配置无效")
        if not (100 <= self.screen_width <= 32768 and 100 <= self.screen_height <= 32768):
            raise ValueError("主屏逻辑尺寸无效")
        if not (0 <= self.rest_noise_x <= 0.02 and 0 <= self.rest_noise_y <= 0.02):
            raise ValueError("静止噪声校准无效")
        if not 0.15 <= self.span <= 3.0 or not 0 <= self.deadband <= 0.03:
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
        if not 1 <= self.scroll_gain <= 180 or not 0.10 <= self.scroll_confirm_seconds <= 0.5:
            raise ValueError("滚动速度或确认时间无效")
        if (
            not 0.005 <= self.scroll_activation_distance <= 0.1
            or not 0 <= self.scroll_deadband <= 0.03
        ):
            raise ValueError("滚动启用位移或抗抖参数无效")
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
    scroll_pose: bool = False


def tripod_features(
    frame: HandFrame, *, spatial_scale=False, spatial_grip=False
) -> TripodFeatures | None:
    if len(frame.landmarks) != 21 or not math.isfinite(frame.aspect) or frame.aspect <= 0:
        return None
    p = np.asarray(frame.landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    xyz = p * [frame.aspect, 1.0, frame.aspect]
    dimension = 3 if spatial_scale else 2
    scale = float(
        (
            np.linalg.norm(xyz[5, :dimension] - xyz[17, :dimension])
            + np.linalg.norm(xyz[0, :dimension] - xyz[9, :dimension])
        )
        / 2
    )
    if scale < 0.025:
        return None
    grip_scale = (
        float((np.linalg.norm(xyz[5] - xyz[17]) + np.linalg.norm(xyz[0] - xyz[9])) / 2)
        if spatial_grip
        else scale
    )
    grip = float(np.linalg.norm(xyz[4] - xyz[12]) / grip_scale)
    # Middle finger gates movement only. Contact is a thumb-index distance
    # proxy with estimated depth, not a physical contact/pressure sensor.
    contact = float(np.linalg.norm(xyz[8] - xyz[4]) / scale)
    pose = extract(frame)
    return TripodFeatures(
        tuple((p[4, :2] + p[12, :2]) / 2),
        tuple(p[[0, 5, 9, 13, 17], :2].mean(axis=0)),
        scale,
        grip,
        contact,
        float(np.linalg.norm(xyz[16] - xyz[4]) / scale),
        bool(pose and pose.scroll_pose),
    )


class TripodEngine:
    def __init__(self, config: TripodConfig | None = None):
        self.config = config or TripodConfig()
        self.config.validate()
        self.wrist_mapping = (
            WristMapping(self.config.wrist_calibration)
            if self.config.pointer_basis == "wrist" and self.config.wrist_calibration
            else None
        )
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
        self.current_frame = None
        self.compatible_point = None
        self.grip_pending_since = None
        self.camera_anchor = None
        self.wrist_anchor = self.wrist_pose = None
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.raw_pointer = None
        self.motion_engaged = False
        self.right_ready = False
        self.right_clear_since = None
        self.require_both_clear = False
        self.arm_since = self.touch_since = self.clear_since = None
        self.stabilizer = None
        self.scroll_since = self.scroll_anchor = self.scroll_y = None
        self.scroll_filter = None
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
        if self.config.pointer_basis == "wrist":
            return None
        span = self.config.span
        center = self.camera_anchor or (0.5, 0.5)
        left, top = tuple(c - p * span for c, p in zip(center, self.pointer_anchor))
        return left, top, left + span, top + span

    def _result(self, events=None, hint=None, cancelled=False):
        hints = {
            "WAIT_GRIP": "拇指与中指捏住，食指移开，接管指针",
            "CONTROL": (
                "拇中定位 · 食拇短捏左键"
                + ("、保持拖拽" if self.config.drag_enabled else "")
                + (" · 拇无名指右键" if self.config.right_enabled else "")
            ),
            "FROZEN": (
                "位置已锁住 · 仍可左/右键 · 移开食指、捏中指继续移动"
                if self.config.right_enabled
                else "位置已锁住 · 仍可左键 · 移开食指、捏中指继续移动"
            ),
            "APPROACH": "食指接近拇指 · 指针已冻结 · 继续靠近按下",
            "PRESSED": (
                "左键已按下 · 松食指单击 · 保持到进度环满可拖拽"
                if self.config.drag_enabled
                else "左键已按下 · 松食指完成点击 · 拖拽已关闭"
            ),
            "DRAG": (
                "拖拽已就绪 · 保持食拇捏合移动 · 松食指放下"
                if self.motion_engaged
                else "拖拽位置锁住 · 捏中指可继续拖 · 松食指放下"
            ),
            "SCROLL": "V 手势滚动 · 拇指保持分开 · 上移向上翻、下移向下翻 · 收指停止",
            "RIGHT_APPROACH": "无名指接近拇指 · 位置已冻结 · 食指保持分开",
            "RIGHT_TOUCHED": "右键已触发 · 松开无名指后可继续操作",
            "WAIT_CLEAR": (
                "动作已取消 · 请先分开食指、无名指与拇指，再试"
                if self.require_both_clear
                else "位置已锁住 · 请先分开食指和拇指，再捏合点击"
            ),
            "PAUSED": "已暂停 · 恢复后重新捏住拇中",
        }
        if self.scroll_since is not None and self.state in ("WAIT_GRIP", "CONTROL", "FROZEN"):
            hints[self.state] = (
                "V 手势已准备 · 上下轻移滚动；捏中指可接管指针"
                if self.state == "WAIT_GRIP"
                else "V 手势已准备 · 上下轻移滚动；静止不翻页，仍可捏合点击"
            )
        f = self.features
        if self.config.pointer_basis == "wrist":
            hints["WAIT_GRIP"] = (
                "先完成三步腕动方向校准"
                if self.wrist_mapping is None
                else "前臂放稳 · 拇中捏住接管腕动，食指移开"
            )
            if self.state == "CONTROL":
                hints["CONTROL"] = "转腕移动 · 停腕停光标 · 松中指回位 · 食拇点击/保持拖拽"
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
                if self.config.drag_enabled
                and self.state == "PRESSED"
                and self.press_since is not None
                else None
            ),
            cancelled=cancelled,
        )
        return self.last_result

    def _anchor(self, f, timestamp):
        self.camera_anchor = f.point
        self.compatible_point = (
            CompatiblePoint(self.current_frame, f.point, self.config)
            if self.config.pointer_basis == "unified"
            else None
        )
        if self.wrist_pose is not None:
            self.wrist_anchor = self.wrist_pose.copy()
        self.pointer_anchor = self.pointer or (0.5, 0.5)
        self.pointer = self.pointer_anchor
        self.raw_pointer = self.pointer_anchor
        cfg = self.config
        if cfg.pointer_basis in ("wrist", "unified"):
            # Both trials use constant gain. Old raw-position noise calibration
            # must not silently change their different input/filtering paths.
            cfg = replace(cfg, motion_profile="precise", rest_noise_x=0.0, rest_noise_y=0.0)
        self.stabilizer = (
            StablePointer(cfg.deadband, cfg.filter_cutoff, cfg.filter_beta)
            if cfg.motion_profile == "classic"
            else TunedPointer(cfg)
        )
        self.stabilizer.update(self.pointer, timestamp)

    def _disarm(self, f, frame, hint=None):
        stopped = self.reset()
        self.features = f
        self.last_time = self.last_seen = frame.timestamp
        self.last_hand = frame.handedness
        return self._result(stopped.events, hint, cancelled=stopped.cancelled)

    def _move(self, timestamp):
        if self.grip_pending_since is not None:
            return self._result(hint="捏合暂不稳定 · 指针已暂停，捏稳后原地继续")
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
        if not cfg.right_enabled or self.left_down or f.contact < cfg.clear_ratio:
            self.right_ready = False
            self.right_clear_since = None
        elif f.right_contact >= cfg.right_clear_ratio:
            if self.right_clear_since is None:
                self.right_clear_since = timestamp
            if timestamp - self.right_clear_since >= cfg.confirm_seconds:
                self.right_ready = True
        else:
            self.right_clear_since = None

    def _scroll(self, f, frame):
        cfg = self.config
        pose = (
            cfg.scroll_enabled
            and not self.left_down
            and f.scroll_pose
            and f.grip > cfg.grip_release
            and f.contact >= cfg.clear_ratio
            and f.right_contact >= cfg.right_clear_ratio
        )
        if self.state == "SCROLL":
            if not pose:
                return self._disarm(f, frame, "滚动已停止 · 食指移开，重新捏中指接管")
            y = self.scroll_filter.update(f.palm, frame.timestamp)[1]
            delta = (self.scroll_y - y) * cfg.scroll_gain
            self.scroll_y = y
            events = (
                [InputEvent("scroll", value=max(-3.0, min(3.0, delta)))]
                if abs(delta) > 1e-6
                else []
            )
            return self._result(events)
        if not pose or self.state not in ("WAIT_GRIP", "CONTROL", "FROZEN"):
            self.scroll_since = self.scroll_anchor = None
            self.scroll_filter = None
            return None
        if self.scroll_since is None:
            self.scroll_since = frame.timestamp
            self.scroll_filter = StablePointer(
                cfg.scroll_deadband, cfg.filter_cutoff, cfg.filter_beta
            )
            self.scroll_anchor = self.scroll_filter.update(f.palm, frame.timestamp)[1]
            y = self.scroll_anchor
        else:
            y = self.scroll_filter.update(f.palm, frame.timestamp)[1]
        if (
            frame.timestamp - self.scroll_since >= cfg.scroll_confirm_seconds
            and abs(y - self.scroll_anchor) >= cfg.scroll_activation_distance
        ):
            # Static V must not steal frozen-position clicks. Commit only after deliberate
            # vertical motion, and discard the pose-forming/activation part of the stroke.
            self.state = "SCROLL"
            self.motion_engaged = self.right_ready = False
            self.raw_pointer = None
            self.pointer = self.pointer or (0.5, 0.5)
            self.arm_since = self.touch_since = self.clear_since = None
            self.scroll_y = y
            return self._result()
        return None

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
        if (
            cfg.drag_enabled
            and self.state == "PRESSED"
            and timestamp - self.press_since >= cfg.drag_hold_seconds
        ):
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
        cfg = self.config
        wrist = cfg.pointer_basis == "wrist"
        unified = cfg.pointer_basis == "unified"
        f = tripod_features(frame, spatial_scale=wrist, spatial_grip=unified)
        if f is None:
            return self.reset()
        # Estimated palm depth can inflate the 3-D scale. Visible fingertip
        # separation independently vetoes a grip; depth cannot authorize an
        # obviously open hand. Use a 2-D numerator here, not the old mixed ratio.
        projected_grip = (
            math.hypot(
                (frame.landmarks[4][0] - frame.landmarks[12][0]) * frame.aspect,
                frame.landmarks[4][1] - frame.landmarks[12][1],
            )
            / f.scale
            if unified
            else 0.0
        )
        visible_grip = not unified or projected_grip <= cfg.grip_release
        visibly_open = unified and projected_grip >= max(0.64, cfg.grip_release + 0.12)
        if wrist:
            if self.wrist_mapping is None:
                return self._disarm(f, frame, "先完成三步腕动方向校准，再启用鼠标")
            pose = palm_rotation(frame)
            if pose is None:
                return self._disarm(
                    f, frame, "掌部姿态暂不可用 · 调整机位，让掌部清晰可见后重新捏合"
                )
            if not self.wrist_mapping.in_range(pose):
                return self._disarm(f, frame, "转腕超出校准范围 · 回到自然姿势，移开食指后重新捏合")
            if (
                self.wrist_pose is not None
                and np.linalg.norm(rotation_vector(pose @ self.wrist_pose.T)) > MAX_FRAME_ANGLE
            ):
                return self._disarm(f, frame, "掌部朝向跳变 · 暂停并松键，请回到自然姿势重新捏合")
            self.wrist_pose = pose
        if self.features is not None and (
            (self.last_hand and frame.handedness and self.last_hand != frame.handedness)
            or math.dist(f.palm, self.features.palm) > self.config.max_jump
            or (
                self.motion_engaged
                and not wrist
                and f.grip <= self.config.grip_release
                and math.dist(f.point, self.features.point) > self.config.max_jump
            )
        ):
            return self._disarm(f, frame, "追踪发生跳变 · 暂停移动，请移开食指重新捏住")
        self.last_time = self.last_seen = t
        self.features, self.last_hand = f, frame.handedness
        self.current_frame = frame
        scrolling = self._scroll(f, frame)
        if scrolling is not None:
            return scrolling
        right_near = (
            cfg.right_enabled and self.right_ready and f.right_contact <= cfg.right_hover_ratio
        )
        self._update_right_ready(f, t)
        if self.state == "WAIT_GRIP":
            if f.grip > cfg.grip_engage or not visible_grip or f.contact < cfg.clear_ratio:
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
        release_grip = f.grip > cfg.grip_release or visibly_open
        if unified and self.motion_engaged:
            if release_grip:
                if self.grip_pending_since is None:
                    self.grip_pending_since = t
                # An uncertain grip never moves the pointer. A clearly open
                # middle finger still freezes immediately using the old clutch.
                release_grip = (
                    visibly_open
                    or f.grip >= max(0.64, cfg.grip_release + 0.12)
                    or t - self.grip_pending_since >= 0.08
                )
            elif self.grip_pending_since is not None:
                self.grip_pending_since = None
                self._anchor(f, t)  # Discard motion during the uncertain frames.
            if self.grip_pending_since is not None and not release_grip and not self.left_down:
                # The grace period is only for motion continuity, never extra
                # evidence for a new click. Preserve the original release-transition
                # rule: touching while middle contact is uncertain must clear first.
                self.touch_since = self.clear_since = None
                if self.state == "RIGHT_TOUCHED":
                    return self._result()
                if self.state == "RIGHT_APPROACH" or self.require_both_clear or right_near:
                    self._wait_clear(both=True)
                elif f.contact < cfg.clear_ratio:
                    self._wait_clear()
                return self._result(hint="捏合暂不稳定 · 指针已暂停，请先分开点击手指")
        if self.motion_engaged and release_grip:
            # Releasing the middle finger freezes before any midpoint update.
            # Do not turn a contact that started during this transition into a
            # click, or rearm a contact that has already clicked.
            self.motion_engaged = False
            self.grip_pending_since = None
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
            if (
                f.grip <= cfg.grip_engage
                and visible_grip
                and (
                    (self.left_down and f.contact < cfg.clear_ratio)
                    or (self.state == "FROZEN" and f.contact >= cfg.clear_ratio and not right_near)
                )
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
            point = (
                self.compatible_point.update(frame, f.point)
                if self.compatible_point is not None
                else f.point
            )
            displacement = (
                self.wrist_mapping.displacement(self.wrist_pose, self.wrist_anchor)
                if wrist
                else tuple(v - c for v, c in zip(point, self.camera_anchor))
            )
            self.raw_pointer = tuple(
                a + delta / cfg.span for a, delta in zip(self.pointer_anchor, displacement)
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
