"""Calibrated palm rotation mapping; estimated geometry, not measured wrist angles."""

import math

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


def palm_rotation(frame: HandFrame):
    """Camera-axis palm basis from the wrist and MCPs; no fingertip positions."""
    if not math.isfinite(frame.aspect) or frame.aspect <= 0:
        return None
    p = np.asarray(frame.landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    q = p * [frame.aspect, 1, frame.aspect]
    width = q[5] - q[17]
    length = q[[5, 9, 13, 17]].mean(axis=0) - q[0]
    w, length_norm = np.linalg.norm(width), np.linalg.norm(length)
    if min(w, length_norm) < 0.025 or not 0.25 <= w / length_norm <= 4:
        return None
    # Estimated z alone must not rescue a nearly edge-on/degenerate projection.
    projected_area = abs(width[0] * length[1] - width[1] * length[0])
    if projected_area / (w * length_norm) < 0.15:
        return None
    u = width / w
    v = length - np.dot(length, u) * u
    if np.linalg.norm(v) / length_norm < 0.3:
        return None
    v /= np.linalg.norm(v)
    return np.column_stack((u, v, np.cross(u, v)))


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
            raise ValueError("先完成12秒腕动方向校准")
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
    PREPARE_SECONDS = 3.0
    STAGE_SECONDS = 4.0
    DURATION = 12.0

    def __init__(self, started, config):
        self.started, self.config = started, config
        self.samples = [[], [], []]
        self.observed = [0, 0, 0]
        self.hand = None
        self.swapped = False
        self.last_timestamp = None

    def progress(self, now):
        return min(1.0, max(0.0, (now - self.started) / self.DURATION))

    def hint(self, now):
        elapsed = max(0.0, now - self.started)
        stage = min(2, int(elapsed / self.STAGE_SECONDS))
        instruction = (
            "前臂放稳，保持自然中立姿势",
            "从中立轻轻向右转腕，停住",
            "先回中立，再轻轻向上转腕，停住",
        )[stage]
        remaining = max(0, math.ceil(self.PREPARE_SECONDS - elapsed % self.STAGE_SECONDS))
        phase = f"准备 {remaining} 秒" if remaining else "正在采集，请保持不动"
        return f"{stage + 1}/3 · {instruction} · {phase}；拇中捏住，食指和无名指移开"

    def add(self, frame):
        # Imported lazily to keep the geometry usable from the gesture module.
        from .tripod import tripod_features

        t = frame.timestamp
        if not math.isfinite(t):
            return
        if self.last_timestamp is not None and t <= self.last_timestamp:
            self.swapped = True  # An interrupted sequence must be redone.
            return
        self.last_timestamp = t
        elapsed = t - self.started
        if not 0 <= elapsed < self.DURATION:
            return
        if frame.landmarks:
            if self.hand is None:
                self.hand = frame.handedness
            elif frame.handedness != self.hand:
                self.swapped = True
        stage = int(elapsed / self.STAGE_SECONDS)
        if elapsed % self.STAGE_SECONDS < self.PREPARE_SECONDS:
            return
        self.observed[stage] += 1
        features = tripod_features(frame, spatial_scale=True)
        pose = palm_rotation(frame)
        cfg = self.config
        if (
            features is not None
            and pose is not None
            and features.grip <= cfg.grip_release
            and features.contact >= cfg.clear_ratio
            and features.right_contact >= cfg.right_clear_ratio
        ):
            self.samples[stage].append((t, pose))

    def result(self):
        if self.swapped:
            raise ValueError("校准期间控制手或时间序列改变，请用同一只手重新采集")
        means = []
        for stage, rows in enumerate(self.samples):
            if len(rows) < 15 or len(rows) / max(1, self.observed[stage]) < 0.8:
                raise ValueError(f"第{stage + 1}步有效样本不足，请保持掌部可见和拇中捏合")
            times = np.array([t for t, _ in rows])
            if times[-1] - times[0] < 0.70 or np.max(np.diff(times)) > 0.15:
                raise ValueError(f"第{stage + 1}步采样不连续，请保持控制手可见")
            mean = mean_rotation([pose for _, pose in rows])
            deviations = [np.linalg.norm(rotation_vector(pose @ mean.T)) for _, pose in rows]
            if np.percentile(deviations, 95) > math.radians(2):
                raise ValueError(f"第{stage + 1}步姿态波动过大，请支撑前臂、停稳后采集")
            means.append(mean)
        values = tuple(
            float(v)
            for v in np.concatenate(
                (
                    means[0].ravel(),
                    rotation_vector(means[1] @ means[0].T),
                    rotation_vector(means[2] @ means[0].T),
                )
            )
        )
        validate_wrist_calibration(values)
        return values
