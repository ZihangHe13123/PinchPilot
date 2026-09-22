"""Deterministic virtual interaction tasks, with no camera or native output."""

import math
from dataclasses import dataclass

from . import __version__


@dataclass(frozen=True)
class Task:
    title: str
    instruction: str
    targets: tuple[tuple[float, float], ...]
    radius_px: float = 15.0


TASKS = {
    "click": Task(
        "小目标点击",
        "移动到圆心，食拇短捏并松开；按下与松开都在圆内才命中。",
        ((0.3, 0.4), (0.72, 0.6), (0.4, 0.68), (0.68, 0.32), (0.5, 0.5)),
    ),
    "double_click": Task(
        "连续双击",
        "在同一圆内连续短捏两次，中间明确松开；两次松开的间隔不超过 0.50 秒。",
        ((0.35, 0.5), (0.65, 0.4), (0.5, 0.65)),
        22.0,
    ),
    "drag": Task(
        "拖拽放下",
        "在方块内捏住食拇，等拖拽就绪，再移入目标圆并松开。",
        ((0.7, 0.4), (0.65, 0.7), (0.72, 0.5)),
        25.0,
    ),
    "reverse": Task(
        "反向微调",
        "依次点击圆点；远移后会在反方向出现相邻小目标，观察能否及时停住再微调。",
        ((0.32, 0.5), (0.68, 0.5), (0.64, 0.5), (0.32, 0.5), (0.36, 0.5)),
        12.0,
    ),
    "scroll": Task(
        "滚动后停住",
        "保持 V 手势，将滚动条移进绿色区，再停住手至少 0.70 秒；保持 V 等本轮通过。",
        ((0.5, 0.5),) * 3,
    ),
}


class PracticeSession:
    """Coordinates map to the full virtual canvas, not to a desktop window rect.

    Pixel distances are task-canvas logical pixels fixed at session start. Reports
    describe task performance, never recognition accuracy or OS double-click success.
    """

    MAX_GAP = 0.30
    SCROLL_SCALE = 4.0
    SCROLL_TOLERANCE = 4.0
    SCROLL_HOLD = 0.70

    def __init__(self, task, *, started, source="camera", viewport=(720, 360)):
        if task not in TASKS or source not in ("camera", "synthetic_demo"):
            raise ValueError("未知任务或输入来源")
        if (
            not math.isfinite(started)
            or len(viewport) != 2
            or any(not math.isfinite(v) or v < 100 for v in viewport)
        ):
            raise ValueError("任务时间或画布尺寸无效")
        self.task, self.spec = task, TASKS[task]
        self.started = self.trial_started = float(started)
        self.source, self.viewport = source, tuple(viewport)
        self.last_time = None
        self.tracked = False
        self.ended = None
        self.reason = ""
        self.active = True
        self.awaiting_clear = True
        self.completed = False
        self.index = self.hits = self.misses = self.cancelled_contacts = 0
        self.invalid_inputs = self.interrupted_inputs = self.double_timeouts = 0
        self.attempts = self.scroll_events = 0
        self.pointer = (0.5, 0.5)
        self.pressed = None
        self.press_point = None
        self.drag_ready = False
        self.first_click = None
        self.double_click_interval = 0.5
        self.scroll_value = 20.0
        self.scroll_still_since = None
        self.rows = []
        self._trial_misses = self._trial_attempts = 0

    @property
    def total(self):
        return len(self.spec.targets)

    @property
    def target(self):
        return self.spec.targets[min(self.index, self.total - 1)]

    @property
    def radius_px(self):
        if self.task == "reverse":
            # Adjacent reverse targets must not overlap even on a small canvas.
            return min(self.spec.radius_px, self.viewport[0] * 0.018)
        return self.spec.radius_px

    @property
    def drag_origin(self):
        return (0.3, 0.5)

    @property
    def scroll_target(self):
        return (65.0, 35.0, 60.0)[min(self.index, 2)]

    def inside(self, point, target=None, radius=None):
        target = self.target if target is None else target
        radius = self.radius_px if radius is None else radius
        return math.hypot(*(self.viewport[i] * (point[i] - target[i]) for i in (0, 1))) <= radius

    def _clear_contact(self):
        if self.pressed is not None:
            self.cancelled_contacts += 1
        self.pressed = self.press_point = self.first_click = None
        self.drag_ready = False
        self.scroll_still_since = None

    def _advance(self, now):
        self.rows.append(
            {
                "trial": self.index + 1,
                "duration_s": round(now - self.trial_started, 4),
                "attempts": self.attempts - self._trial_attempts,
                "misses": self.misses - self._trial_misses,
            }
        )
        self.hits += 1
        self.index += 1
        self.trial_started = now
        self._trial_misses, self._trial_attempts = self.misses, self.attempts
        self.first_click = self.scroll_still_since = None
        if self.index == self.total:
            self.active, self.completed, self.ended = False, True, now
            self.reason = "completed"

    def _release(self, event, now):
        button = "right" if event.kind == "right_up" else "left"
        if button != self.pressed:
            return
        self.attempts += 1
        point = (event.x, event.y)
        origin = self.drag_origin if self.task == "drag" else self.target
        origin_inside = (
            all(
                abs((self.press_point[i] - origin[i]) * self.viewport[i]) <= self.radius_px
                for i in (0, 1)
            )
            if self.task == "drag"
            else self.inside(self.press_point, origin)
        )
        hit = (
            button == "left"
            and self.task != "scroll"
            and origin_inside
            and self.inside(point)
            and (self.task != "drag" or self.drag_ready)
        )
        self.pressed = self.press_point = None
        self.drag_ready = False
        if not hit:
            self.misses += 1
            self.first_click = None
        elif self.task == "double_click":
            if (
                self.first_click is not None
                and now - self.first_click <= self.double_click_interval
            ):
                self._advance(now)
            else:
                if self.first_click is not None:
                    self.double_timeouts += 1
                self.first_click = now
        else:
            self._advance(now)

    def feed(self, result, now, source="camera"):
        if not self.active:
            return
        if source != self.source:
            self.cancel(now, "source_changed")
            return
        if (
            not math.isfinite(now)
            or now < self.started
            or (self.last_time is not None and now <= self.last_time)
        ):
            self.invalid_inputs += 1
            return
        points = ([result.pointer] if result.pointer is not None else []) + [
            (event.x, event.y)
            for event in result.events
            if event.kind in ("move", "down", "up", "right_down", "right_up")
        ]
        if any(
            len(point) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in point)
            for point in points
        ) or any(not math.isfinite(event.value) for event in result.events):
            self.invalid_inputs += 1
            self._clear_contact()
            return
        if self.last_time is not None and now - self.last_time > self.MAX_GAP:
            self.interrupted_inputs += 1
            self._clear_contact()
        self.last_time = now
        if result.pointer is not None:
            self.pointer = tuple(result.pointer)
        if result.grip is None:
            if self.tracked:
                self.interrupted_inputs += 1
            self.tracked = False
            self._clear_contact()
            return
        self.tracked = True
        if result.cancelled:
            self._clear_contact()
            return
        if self.awaiting_clear:
            if (
                result.state in ("CONTROL", "FROZEN", "WAIT_GRIP")
                and (result.contact is None or result.contact >= 0.48)
                and not any(
                    event.kind in ("down", "up", "right_down", "right_up")
                    for event in result.events
                )
            ):
                self.awaiting_clear = False
            return
        if result.state == "DRAG" and self.pressed == "left":
            self.drag_ready = True
        for event in result.events:
            if not self.active:
                break
            if event.kind in ("move", "down", "up", "right_down", "right_up"):
                self.pointer = (event.x, event.y)
            if event.kind in ("down", "right_down"):
                if self.pressed is None:
                    self.pressed = "right" if event.kind == "right_down" else "left"
                    self.press_point = self.pointer
                    self.drag_ready = result.state == "DRAG"
            elif event.kind in ("up", "right_up"):
                self._release(event, now)
            elif event.kind == "scroll" and self.task == "scroll" and event.value != 0:
                self.scroll_events += 1
                self.scroll_value = max(
                    0, min(100, self.scroll_value - event.value * self.SCROLL_SCALE)
                )
                self.scroll_still_since = None
        if self.task == "scroll" and self.active:
            if (
                result.state != "SCROLL"
                or abs(self.scroll_value - self.scroll_target) > self.SCROLL_TOLERANCE
            ):
                self.scroll_still_since = None
            elif self.scroll_still_since is None:
                self.scroll_still_since = now
            elif now - self.scroll_still_since >= self.SCROLL_HOLD:
                self._advance(now)

    def cancel(self, now, reason="user_cancelled"):
        if self.active:
            if reason == "stale_input":
                self.interrupted_inputs += 1
            self._clear_contact()
            self.active = False
            self.ended = max(
                self.last_time or self.started, now if math.isfinite(now) else self.started
            )
            self.reason = reason

    def summary(self, now):
        duration = max(0, (self.ended if self.ended is not None else now) - self.started)
        return {
            "schema_version": 1,
            "app_version": __version__,
            "task": self.task,
            "source": self.source,
            "evidence": "synthetic_engineering"
            if self.source == "synthetic_demo"
            else "virtual_camera_task",
            "metric_scope": "Fixed virtual task outcomes; not recognition accuracy, OS input acceptance, or end-to-end latency.",
            "status": "completed" if self.completed else "running" if self.active else "cancelled",
            "reason": self.reason,
            "duration_s": round(duration, 4),
            "completed_targets": self.hits,
            "total_targets": self.total,
            "paired_button_attempts": self.attempts,
            "misses": self.misses,
            "cancelled_contacts": self.cancelled_contacts,
            "input_interruptions": self.interrupted_inputs,
            "invalid_inputs": self.invalid_inputs,
            "double_click_timeouts": self.double_timeouts,
            "scroll_events": self.scroll_events,
            "protocol": {
                "canvas_logical_size": self.viewport,
                "coordinate_space": "normalized entire virtual canvas",
                "target_radius_canvas_px": self.radius_px,
                "target_sequence": self.spec.targets,
                "double_release_interval_s": self.double_click_interval,
                "scroll_units_per_event_unit": self.SCROLL_SCALE,
                "scroll_target_sequence": (65, 35, 60),
                "scroll_tolerance": self.SCROLL_TOLERANCE,
                "scroll_still_s": self.SCROLL_HOLD,
                "maximum_input_gap_s": self.MAX_GAP,
            },
            "definitions": {
                "completed_targets": "Trials meeting this task's complete criterion; a double-click pair is one target.",
                "misses": "Paired button attempts outside the required target/button/drag-ready condition; interruptions excluded.",
                "duration_s": "Wall time from Start, including repositioning and interruptions; excludes time before Start.",
                "scroll_success": "Within target tolerance with no nonzero scroll events for 0.70 s, while fresh SCROLL states continue.",
            },
            "trials": list(self.rows),
            "stores_images": False,
            "stores_landmarks": False,
            "stores_pointer_trajectory": False,
        }
