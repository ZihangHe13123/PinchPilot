"""Advisory camera-edge feedback; never changes cursor or button events."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FramingStatus:
    state: str = "waiting"
    edges: tuple[str, ...] = ()


class FramingMonitor:
    # Wrist/pinky can naturally leave the image in a supported-hand posture.
    POINTS = (4, 5, 8, 9, 12, 13, 16, 17)
    ENTER_MARGIN = 0.08
    EXIT_MARGIN = 0.12
    ENTER_SECONDS = 0.12
    CLEAR_SECONDS = 0.25
    LOST_SECONDS = 0.15
    MAX_GAP_SECONDS = 0.25

    def __init__(self):
        self.reset()

    def reset(self):
        self.status = FramingStatus()
        self.last_time = self.missing_since = self.candidate_since = None
        self.candidate = None
        self.seen = False

    def update(self, frame, tracked=True):
        t = frame.timestamp
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            self.reset()
            return self.status
        if self.last_time is not None and t - self.last_time > self.MAX_GAP_SECONDS:
            self.reset()
        self.last_time = t
        valid = (
            tracked
            and len(frame.landmarks) == 21
            and all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in frame.landmarks)
        )
        if not valid:
            self.candidate = self.candidate_since = None
            if self.missing_since is None:
                self.missing_since = t
            if self.seen and t - self.missing_since >= self.LOST_SECONDS:
                self.status = FramingStatus("lost")
            return self.status
        self.seen = True
        self.missing_since = None
        points = [frame.landmarks[i] for i in self.POINTS]
        margins = {
            "left": min(p[0] for p in points),
            "right": 1 - max(p[0] for p in points),
            "top": min(p[1] for p in points),
            "bottom": 1 - max(p[1] for p in points),
        }
        edges = tuple(
            edge
            for edge, margin in margins.items()
            if margin < (self.EXIT_MARGIN if edge in self.status.edges else self.ENTER_MARGIN)
        )
        if self.status.state in ("waiting", "lost"):
            self.status = FramingStatus("clear")
        candidate = FramingStatus("edge", edges) if edges else FramingStatus("clear")
        if candidate == self.status:
            self.candidate = self.candidate_since = None
        elif candidate != self.candidate:
            self.candidate, self.candidate_since = candidate, t
        elif t - self.candidate_since >= (self.ENTER_SECONDS if edges else self.CLEAR_SECONDS):
            self.status = candidate
            self.candidate = self.candidate_since = None
        return self.status


def framing_hint(status, gesture_state):
    if status.state == "waiting":
        return "请把控制手放入画面，先找一个前臂有支撑的位置。"
    if status.state == "lost":
        return "暂未跟踪到控制手 · 移回画面，食指移开、拇中捏住重新接管。"
    if status.state == "edge":
        labels = {"left": "左侧", "right": "右侧", "top": "上沿", "bottom": "下沿"}
        edge = "、".join(labels[value] for value in status.edges)
        if gesture_state == "DRAG":
            action = "保持食拇捏合，松中指换位，再捏中指继续拖。"
        elif gesture_state == "SCROLL":
            action = "收回 V 手势，将手移回画面中央，再继续滚动。"
        elif gesture_state == "PRESSED":
            action = "先完成当前点击，再松中指换位。"
        else:
            action = "松中指，将手移回画面中央，再捏住继续。"
        return f"手靠近画面{edge} · {action}"
    return "控制手在画面内 · 松中指可换位，捏回继续。"
