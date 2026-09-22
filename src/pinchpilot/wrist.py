"""Calibrated palm rotation mapping; estimated geometry, not measured wrist angles."""

import math
from collections import Counter, deque

import numpy as np

from .domain import HandFrame

MAX_WORKING_ANGLE = math.radians(65)
MAX_FRAME_ANGLE = math.radians(25)
RADIANS_PER_SCREEN = math.radians(45)


def rotation_vector(rotation):
    """SO(3) log, including stable small-angle and pi cases."""
    angle = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-6:
        return skew / 2
    if math.pi - angle < 1e-5:
        # Calibration/working ranges reject this angle, but computing its
        # magnitude must not turn a half-turn into a false zero rotation.
        values, vectors = np.linalg.eigh((rotation + rotation.T) / 2)
        axis = vectors[:, np.argmax(values)]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis * angle
    return skew * (angle / (2 * math.sin(angle)))


def _palm_observation(frame: HandFrame):
    if not frame.landmarks:
        return None, "missing_hand"
    if not math.isfinite(frame.aspect) or frame.aspect <= 0:
        return None, "invalid_coordinates"
    p = np.asarray(frame.landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None, "invalid_coordinates"
    q = p * [frame.aspect, 1, frame.aspect]
    width = q[5] - q[17]
    length = q[[5, 9, 13, 17]].mean(axis=0) - q[0]
    w, length_norm = np.linalg.norm(width), np.linalg.norm(length)
    if min(w, length_norm) < 0.025 or not 0.25 <= w / length_norm <= 4:
        return None, "palm_size"
    # Estimated z alone must not rescue a nearly edge-on/degenerate projection.
    projected_area = abs(width[0] * length[1] - width[1] * length[0])
    if projected_area / (w * length_norm) < 0.15:
        return None, "edge_on"
    u = width / w
    v = length - np.dot(length, u) * u
    if np.linalg.norm(v) / length_norm < 0.3:
        return None, "degenerate"
    v /= np.linalg.norm(v)
    return np.column_stack((u, v, np.cross(u, v))), ""


def palm_rotation(frame: HandFrame):
    """Camera-axis palm basis from the wrist and MCPs; no fingertip positions."""
    return _palm_observation(frame)[0]


def validate_wrist_calibration(values):
    if not isinstance(values, (tuple, list)) or len(values) not in (0, 15):
        raise ValueError("腕动校准应为空或15个数值")
    if len(values) == 0:
        return
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("腕动校准包含无效数值")
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("腕动校准包含非有限值")
    neutral = a[:9].reshape(3, 3)
    if not np.allclose(neutral.T @ neutral, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(neutral), 1.0, atol=1e-5
    ):
        raise ValueError("腕动中立姿态无效")
    directions = a[9:].reshape(2, 3)
    magnitudes = np.linalg.norm(directions, axis=1)
    if np.any(magnitudes < math.radians(8)) or np.any(magnitudes > math.radians(35)):
        raise ValueError("示范转动需约8–35度：轻转一小段，避免只动手指")
    cosine = np.dot(directions[0], directions[1]) / np.prod(magnitudes)
    if abs(cosine) > math.cos(math.radians(45)):
        raise ValueError("向右和向上的转动太相似；回到中立后分别示范两个方向")


class WristMapping:
    def __init__(self, calibration):
        validate_wrist_calibration(calibration)
        if not calibration:
            raise ValueError("先完成三步腕动方向校准")
        data = np.array(calibration)
        self.neutral = data[:9].reshape(3, 3)
        directions = data[9:].reshape(2, 3)
        self.axes = np.linalg.pinv((directions / np.linalg.norm(directions, axis=1)[:, None]).T)

    def in_range(self, pose):
        return np.linalg.norm(rotation_vector(pose @ self.neutral.T)) <= MAX_WORKING_ANGLE

    def displacement(self, pose, anchor):
        angles = self.axes @ rotation_vector(pose @ anchor.T)
        # Equivalent input units keep the existing sensitivity and filter stages.
        return tuple(angles * [1, -1] * (0.3 / RADIANS_PER_SCREEN))


def mean_rotation(poses):
    u, _, vt = np.linalg.svd(np.mean(poses, axis=0))
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    return u @ correction @ vt


class WristCalibration:
    """Advance on enough valid, stable observations, never on a countdown."""

    REQUIRED_SAMPLES = 30
    WINDOW_SAMPLES = 36
    MAX_SAMPLE_GAP = 0.25
    MAX_SPREAD_DEGREES = 2.0
    STAGES = ("neutral", "right", "up")
    ISSUES = {
        "missing_hand": "尚未取得控制手，请把常用手放到镜头看得清的位置",
        "invalid_coordinates": "手部坐标暂不可用，请保持掌部清晰可见",
        "palm_size": "掌部太小或形状不可用，请靠近镜头一些",
        "edge_on": "掌部太侧斜，请稍调整手或机位，让掌部轮廓看得清",
        "degenerate": "掌部结构暂不稳定，请调整机位后保持自然手形",
        "hand_changed": "控制手改变，请继续使用开始校准时的那只手",
        "time_order": "帧顺序中断，正在等待新的连续画面",
    }

    def __init__(self, started, config):
        self.started, self.config = started, config
        self.stage = 0
        self.samples = [[], [], []]
        self.observed = [0, 0, 0]
        self.rejected = [Counter(), Counter(), Counter()]
        self.window = deque(maxlen=self.WINDOW_SAMPLES)
        self.means = []
        self.hand = ""
        self.hand_changed = self.time_order_invalid = False
        self.last_timestamp = None
        self.current_pose = None
        self.current_turn = self.direction_angle = None
        self.current_issue = self.ISSUES["missing_hand"]
        self.spread = None
        self.need_neutral = False
        self.collecting = False

    @property
    def complete(self):
        return self.stage == 3

    def progress(self, now):
        if self.complete:
            return 1.0
        count = sum(pose is not None for _, pose in self.window)
        # A full but unstable window is not a passed stage.
        return (self.stage + min(0.95, count / self.REQUIRED_SAMPLES)) / 3

    def reference_hint(self):
        if self.complete:
            return "三步已通过，可以回到舒服的姿势后手动启用鼠标。"
        if self.stage == 0:
            return "① 中立参考：前臂支撑，保持日常操作的自然手形；无需捏紧手指。"
        turn = (
            f"当前估计偏转 {self.current_turn:.1f}°"
            if self.current_turn is not None
            else "等待掌部朝向"
        )
        if self.stage == 1:
            return f"② 向右 →：从中立轻转整个手掌并停住，参考范围 8–35°；{turn}。"
        if self.need_neutral:
            return f"③ 先回中立 ◉：回到第①步姿态（估计偏转 ≤5°），再向上转；{turn}。"
        separation = (
            f"；与向右方向夹角 {self.direction_angle:.1f}°（参考 45–135°）"
            if self.direction_angle is not None
            else ""
        )
        return f"③ 向上 ↑：从中立轻转并停住，参考范围 8–35°；{turn}{separation}。"

    def hint(self, now):
        if self.complete:
            return "三步数据已收集完成；鼠标控制仍关闭。"
        count = min(self.REQUIRED_SAMPLES, sum(pose is not None for _, pose in self.window))
        detail = self.current_issue or "掌部有效，保持当前姿势即可"
        return f"第 {self.stage + 1}/3 步 · 有效姿态 {count}/{self.REQUIRED_SAMPLES} · {detail}"

    def _reject(self, timestamp, reason, message=None):
        self.rejected[self.stage][reason] += 1
        self.window.append((timestamp, None))
        self.current_issue = message or self.ISSUES[reason]

    def add(self, frame):
        if self.complete:
            return
        self.collecting = False
        t = frame.timestamp
        if not math.isfinite(t) or (self.last_timestamp is not None and t <= self.last_timestamp):
            self.time_order_invalid = True
            self.window.clear()
            self.current_issue = self.ISSUES["time_order"]
            self.rejected[self.stage]["time_order"] += 1
            return
        if self.last_timestamp is not None and t - self.last_timestamp > self.MAX_SAMPLE_GAP:
            self.window.clear()
        self.last_timestamp = t
        self.observed[self.stage] += 1
        self.spread = None
        pose, issue = _palm_observation(frame)
        self.current_pose = pose
        self.current_turn = self.direction_angle = None
        if pose is None:
            self._reject(t, issue)
            return
        if not self.hand:
            self.hand = frame.handedness
        elif frame.handedness and frame.handedness != self.hand:
            self.hand_changed = True
            self._reject(t, "hand_changed")
            return
        vector = None
        if self.stage:
            vector = rotation_vector(pose @ self.means[0].T)
            self.current_turn = math.degrees(np.linalg.norm(vector))
            if self.need_neutral:
                self.window.clear()
                if self.current_turn <= 5:
                    self.need_neutral = False
                    self.current_issue = "已回到中立，现在轻向上转整个手掌"
                else:
                    self._reject(
                        t, "return_neutral", "先回到第①步的自然姿态，再向上转；前两步已保留"
                    )
                return
            if not 8 <= self.current_turn <= 35:
                self._reject(
                    t, "turn_range", f"估计偏转 {self.current_turn:.1f}°；轻转至 8–35° 范围后停住"
                )
                return
            if self.stage == 2:
                right = rotation_vector(self.means[1] @ self.means[0].T)
                cosine = float(
                    np.clip(
                        np.dot(vector, right) / (np.linalg.norm(vector) * np.linalg.norm(right)),
                        -1,
                        1,
                    )
                )
                self.direction_angle = math.degrees(math.acos(cosine))
                if not 45 <= self.direction_angle <= 135:
                    self._reject(
                        t,
                        "direction_separation",
                        f"两个方向太相似（夹角 {self.direction_angle:.1f}°）；请示范不同于向右的向上转动",
                    )
                    return
        self.collecting = True
        self.window.append((t, pose))
        rows = [(at, p) for at, p in self.window if p is not None]
        self.current_issue = "掌部有效，保持当前姿势即可"
        if len(rows) < self.REQUIRED_SAMPLES:
            return
        times = np.array([at for at, _ in rows])
        if np.max(np.diff(times)) > self.MAX_SAMPLE_GAP:
            self.current_issue = "有效姿态有中断，正在收集新的一段连续数据"
            return
        # Use the newest full window so a noisy or moving attempt can recover.
        rows = rows[-self.REQUIRED_SAMPLES :]
        mean = mean_rotation([p for _, p in rows])
        deviations = [math.degrees(np.linalg.norm(rotation_vector(p @ mean.T))) for _, p in rows]
        self.spread = float(np.percentile(deviations, 95))
        if self.spread > self.MAX_SPREAD_DEGREES:
            self.current_issue = (
                f"估计朝向波动 {self.spread:.1f}°（参考 ≤2°）；停稳或调整机位，数据会自动继续收集"
            )
            return
        # Validate each fitted direction before saving the stage. A mean just
        # below the minimum would otherwise make the final step impossible.
        if self.stage == 1:
            turn = math.degrees(np.linalg.norm(rotation_vector(mean @ self.means[0].T)))
            if not 8 <= turn <= 35:
                self.current_issue = f"平均偏转 {turn:.1f}° 接近范围边界，请稍调整幅度后停稳"
                return
        # Validate the fitted means too, not only individual samples near a boundary.
        if self.stage == 2:
            candidate = self._parameters(self.means + [mean])
            try:
                validate_wrist_calibration(candidate)
            except ValueError as error:
                self.current_issue = str(error)
                return
        self.samples[self.stage] = rows
        self.means.append(mean)
        self.stage += 1
        self.window.clear()
        self.spread = None
        self.current_issue = "上一步已通过，请按参考调整到下一姿态"
        self.need_neutral = self.stage == 2
        self.collecting = False

    @staticmethod
    def _parameters(means):
        return tuple(
            float(v)
            for v in np.concatenate(
                (
                    means[0].ravel(),
                    rotation_vector(means[1] @ means[0].T),
                    rotation_vector(means[2] @ means[0].T),
                )
            )
        )

    def diagnostics(self):
        stages = []
        for i, name in enumerate(self.STAGES):
            rows = self.samples[i] or (
                [(t, p) for t, p in self.window if p is not None] if i == self.stage else []
            )
            times = [t for t, _ in rows]
            stages.append(
                {
                    "name": name,
                    "observed": self.observed[i],
                    "accepted": len(rows),
                    "rejected": dict(self.rejected[i]),
                    "span_s": round(times[-1] - times[0], 3) if len(times) > 1 else 0.0,
                    "max_gap_s": round(max(np.diff(times)), 3) if len(times) > 1 else 0.0,
                }
            )
        return {
            "scope": "calibration_geometry_summary",
            "stage": self.stage,
            "hand_changed": self.hand_changed,
            "time_order_invalid": self.time_order_invalid,
            "stages": stages,
            "current_spread_deg": self.spread,
            "estimated_turn_deg": self.current_turn,
            "direction_angle_deg": self.direction_angle,
            "current_issue": self.current_issue,
            "complete": self.complete,
        }

    def result(self):
        if not self.complete:
            raise ValueError(self.hint(self.last_timestamp or self.started))
        values = self._parameters(self.means)
        validate_wrist_calibration(values)
        return values
