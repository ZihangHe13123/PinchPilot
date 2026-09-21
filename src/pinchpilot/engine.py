"""Pure event state machine. It neither reads a camera nor touches the OS."""

import math

from .domain import EngineConfig, EngineResult, HandFrame, InputEvent, Prediction
from .features import extract, rule_prediction
from .filtering import AdaptiveFilter


class GestureEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.config.validate()
        self.enabled = True  # Preview engine only; OS output has its own explicit gate.
        self.state = "WAIT_OPEN"
        self.pointer = None
        self.last_seen = None
        self.last_time = None
        self.last_hand = ""
        self.raw_pointer = None
        self.down = False
        self.press_anchor = None
        self.press_input = None
        self.drag_origin = None
        self.scroll_y = None
        self.pending = None
        self.pending_since = None
        self.filters = [AdaptiveFilter(), AdaptiveFilter()]

    def _release(self) -> list[InputEvent]:
        events = []
        if self.down:
            x, y = self.pointer or (0.5, 0.5)
            events.append(InputEvent("up", x, y))
        self.down = False
        self.press_anchor = self.press_input = self.drag_origin = None
        self.scroll_y = None
        self.pending = self.pending_since = None
        return events

    def reset(self) -> EngineResult:
        events = self._release()
        self.state = "WAIT_OPEN" if self.enabled else "PAUSED"
        self.last_seen = self.last_time = None
        self.last_hand = ""
        self.raw_pointer = None
        self.filters = [AdaptiveFilter(), AdaptiveFilter()]
        return EngineResult(self.state, self.pointer, events)

    def set_enabled(self, enabled: bool) -> EngineResult:
        self.enabled = enabled
        return self.reset()

    def _confirmed(self, label: str, timestamp: float) -> bool:
        if self.pending != label:
            self.pending, self.pending_since = label, timestamp
        return timestamp - self.pending_since >= self.config.confirm_seconds

    def _move(self, p: tuple[float, float]) -> InputEvent:
        self.pointer = (min(1.0, max(0.0, p[0])), min(1.0, max(0.0, p[1])))
        return InputEvent("move", *self.pointer)

    def tick(self, timestamp: float) -> EngineResult:
        if (
            self.last_seen is not None
            and timestamp - self.last_seen >= self.config.tracking_timeout
        ):
            events = self._release()
            self.state = "WAIT_OPEN" if self.enabled else "PAUSED"
            self.last_seen = None
            self.filters = [AdaptiveFilter(), AdaptiveFilter()]
            return EngineResult(self.state, self.pointer, events)
        return EngineResult(self.state, self.pointer)

    def process(self, frame: HandFrame, prediction: Prediction | None = None) -> EngineResult:
        t = frame.timestamp
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            events = self._release()
            self.state = "WAIT_OPEN" if self.enabled else "PAUSED"
            return EngineResult(self.state, self.pointer, events)
        # A resumed/stalled stream must release BEFORE evaluating the returning hand.
        expired = self.tick(t).events
        self.last_time = t
        if not self.enabled:
            return EngineResult("PAUSED", self.pointer, expired)
        f = extract(frame)
        if f is None:
            # A confirmation needs continuously observed evidence, not two frames
            # separated by a short tracking dropout.
            self.pending = self.pending_since = None
            return EngineResult(self.state, self.pointer, expired)
        if (self.last_hand and frame.handedness and frame.handedness != self.last_hand) or (
            self.raw_pointer is not None
            and self.last_seen is not None
            and math.dist(self.raw_pointer, f.pointer) > 0.35
        ):
            expired += self._release()
            self.state = "WAIT_OPEN"
            self.filters = [AdaptiveFilter(), AdaptiveFilter()]
        self.last_hand, self.raw_pointer, self.last_seen = frame.handedness, f.pointer, t
        cfg = self.config
        p = (
            (f.pointer[0] - cfg.box_left) / (cfg.box_right - cfg.box_left),
            (f.pointer[1] - cfg.box_top) / (cfg.box_bottom - cfg.box_top),
        )
        p = tuple(filt.update(v, t) for filt, v in zip(self.filters, p))
        pred = prediction or rule_prediction(f, cfg.engage_ratio, cfg.release_ratio)
        label = pred.label
        if label not in ("open", "pinch", "scroll", "other", "uncertain"):
            label = "other"
        if (
            self.down
            and pred.source == "rules"
            and label == "uncertain"
            and f.pinch <= cfg.release_ratio
        ):
            # True hysteresis: once pressed, the geometric gap keeps the hold.
            # ML abstentions still take the conservative timed-release branch.
            label = "pinch"
        events = expired

        if f.fist or label == "other":
            events += self._release()
            self.state = "WAIT_OPEN"
        elif self.state == "WAIT_OPEN":
            # Geometry AND classifier must agree on an open hand to re-arm.
            if label == "open" and f.pinch > cfg.release_ratio and self._confirmed("arm", t):
                self.state = "POINT"
                self.pending = None
                events.append(self._move(p))
            elif label != "open":
                self.pending = None
        elif self.state == "SCROLL":
            if label == "scroll":
                if self.scroll_y is not None:
                    dy = (self.scroll_y - p[1]) * cfg.scroll_gain
                    if abs(dy) > 0.01:
                        events.append(InputEvent("scroll", value=dy))
                self.scroll_y = p[1]
                self.pending = None
            elif self._confirmed("end_scroll", t):
                self.state = "WAIT_OPEN"
                self.scroll_y = None
        elif self.state in ("PRESSED", "DRAG"):
            if label == "open":
                if self._confirmed("release", t):
                    events += self._release()
                    self.state = "POINT"
            elif label in ("uncertain", "scroll"):
                # Low-confidence predictions never hold an OS button indefinitely.
                if self._confirmed("invalid_press", t):
                    events += self._release()
                    self.state = "WAIT_OPEN"
            else:
                self.pending = None
                if self.state == "PRESSED" and math.dist(p, self.press_input) > cfg.drag_distance:
                    self.state = "DRAG"
                    self.drag_origin = p
                if self.state == "DRAG":
                    if cfg.stabilise:
                        target = tuple(
                            a + v - o for a, v, o in zip(self.press_anchor, p, self.drag_origin)
                        )
                    else:
                        target = p
                    events.append(self._move(target))
        else:  # POINT
            if label == "pinch":
                if self.pending != "press":
                    self.press_anchor = self.pointer or p
                    self.press_input = p
                if self._confirmed("press", t):
                    if not cfg.stabilise:
                        self.press_anchor = p
                    events.append(self._move(self.press_anchor))
                    events.append(InputEvent("down", *self.pointer))
                    self.down = True
                    self.state = "PRESSED"
                    self.pending = None
                elif not cfg.stabilise:
                    events.append(self._move(p))
            elif label == "scroll":
                if self._confirmed("scroll", t):
                    self.state = "SCROLL"
                    self.scroll_y = p[1]
                    self.pending = None
            elif label == "open":
                self.pending = None
                events.append(self._move(p))
            else:
                self.pending = None
        return EngineResult(self.state, self.pointer, events, pred, f.pinch)
