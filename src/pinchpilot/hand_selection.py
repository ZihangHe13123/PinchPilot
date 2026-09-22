"""Keep one control hand across detector reordering, disappearance and other hands."""

import math

from .domain import HandFrame


class ControlHandSelector:
    CONFIDENCE = 0.70  # Handedness classification only, not landmark visibility.
    CONFIRM_FRAMES = 3
    MAX_GAP = 0.20
    MAX_STEP = 0.12  # Aspect-corrected image coordinates, in image-height units.
    MIN_SEPARATION = 0.08

    def __init__(self, preferred_hand="auto"):
        if preferred_hand not in ("auto", "Left", "Right"):
            raise ValueError("控制手必须为 auto / Left / Right")
        self.locked_hand = "" if preferred_hand == "auto" else preferred_hand
        self.palm = None
        self.last_seen = None
        self.pending_hand = ""
        self.pending_palm = None
        self.pending_frames = 0
        self.status = self._waiting()

    def _waiting(self):
        label = {"Left": "左手", "Right": "右手"}.get(self.locked_hand)
        return f"等待{label} · 另一只手不会接管" if label else "先只露出要控制的一只手"

    @staticmethod
    def _palm(frame):
        if (
            len(frame.landmarks) != 21
            or any(len(point) != 3 for point in frame.landmarks)
            or frame.handedness not in ("Left", "Right")
            or not math.isfinite(frame.aspect)
            or frame.aspect <= 0
            or not all(math.isfinite(v) for point in frame.landmarks for v in point)
        ):
            return None
        points = [frame.landmarks[i] for i in (0, 5, 9, 13, 17)]
        return (
            sum(p[0] for p in points) / len(points) * frame.aspect,
            sum(p[1] for p in points) / len(points),
        )

    def _missing(self, timestamp, aspect, status=None):
        self.palm = self.pending_palm = None
        self.pending_hand = ""
        self.pending_frames = 0
        self.status = status or self._waiting()
        return HandFrame(timestamp, aspect=aspect)

    def select(self, frames, timestamp, aspect):
        candidates = [(frame, self._palm(frame)) for frame in frames]
        candidates = [(frame, palm) for frame, palm in candidates if palm is not None]
        eligible = [
            (frame, palm)
            for frame, palm in candidates
            if self.CONFIDENCE <= frame.handedness_score <= 1
            and (not self.locked_hand or frame.handedness == self.locked_hand)
        ]
        # Initial auto selection is intentional: expose one hand, or choose a side in UI.
        if not self.locked_hand and len(candidates) > 1:
            return self._missing(timestamp, aspect, "请先只露出控制手，或指定左手 / 右手")
        if len(eligible) != 1:
            return self._missing(timestamp, aspect)
        frame, palm = eligible[0]
        if any(
            other is not frame and math.dist(palm, other_palm) < self.MIN_SEPARATION
            for other, other_palm in candidates
        ):
            return self._missing(timestamp, aspect, "双手太近 · 分开后恢复控制手")
        continuous = (
            self.palm is not None
            and self.last_seen is not None
            and 0 < timestamp - self.last_seen <= self.MAX_GAP
            and math.dist(palm, self.palm) <= self.MAX_STEP
        )
        if not continuous:
            self.palm = None
            if (
                self.pending_hand != frame.handedness
                or self.pending_palm is None
                or self.last_seen is None
                or not 0 < timestamp - self.last_seen <= self.MAX_GAP
                or math.dist(palm, self.pending_palm) > self.MAX_STEP
            ):
                self.pending_frames = 0
            self.pending_hand, self.pending_palm = frame.handedness, palm
            self.pending_frames += 1
            self.last_seen = timestamp
            if self.pending_frames < self.CONFIRM_FRAMES:
                label = "左手" if frame.handedness == "Left" else "右手"
                self.status = f"正在确认{label}"
                return HandFrame(timestamp, aspect=aspect)
        self.locked_hand = frame.handedness
        self.palm, self.last_seen = palm, timestamp
        self.pending_palm = None
        self.pending_frames = 0
        label = "左手" if self.locked_hand == "Left" else "右手"
        self.status = f"已锁定{label} · 另一只手不参与操作"
        return frame
