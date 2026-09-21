from dataclasses import dataclass, field


@dataclass(frozen=True)
class HandFrame:
    timestamp: float
    landmarks: tuple[tuple[float, float, float], ...] = ()
    aspect: float = 4 / 3
    handedness: str = ""
    # MediaPipe's handedness score is NOT a joint visibility/confidence score.
    handedness_score: float = 0.0


@dataclass(frozen=True)
class Features:
    vector: tuple[float, ...]
    pinch: float
    pointer: tuple[float, float]
    scroll_pose: bool
    fist: bool


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    source: str = "rules"


@dataclass(frozen=True)
class InputEvent:
    kind: str
    x: float = 0.0
    y: float = 0.0
    value: float = 0.0


@dataclass
class EngineResult:
    state: str
    pointer: tuple[float, float] | None = None
    events: list[InputEvent] = field(default_factory=list)
    prediction: Prediction | None = None
    pinch: float | None = None
    progress: float | None = None
    hint: str = ""
    mode: str = "pinch"
    bend: float | None = None
    grip: float | None = None
    contact: float | None = None
    raw_pointer: tuple[float, float] | None = None
    right_contact: float | None = None
    cancelled: bool = False


@dataclass
class EngineConfig:
    engage_ratio: float = 0.26
    release_ratio: float = 0.40
    confirm_seconds: float = 0.07
    tracking_timeout: float = 0.20
    drag_distance: float = 0.028
    stabilise: bool = True
    scroll_gain: float = 18.0
    # A deliberately limited camera box: small movements can reach the screen.
    box_left: float = 0.20
    box_top: float = 0.20
    box_right: float = 0.80
    box_bottom: float = 0.80
    reanchor_on_open: bool = False

    def validate(self) -> None:
        import math

        if not all(math.isfinite(float(v)) for v in vars(self).values()):
            raise ValueError("配置包含无效数值")
        if not 0.02 <= self.engage_ratio < self.release_ratio <= 2.0:
            raise ValueError("捏合阈值必须小于释放阈值")
        if not 0.02 <= self.confirm_seconds <= 0.5:
            raise ValueError("确认时间超出范围")
        if not 0.05 <= self.tracking_timeout <= 1.0:
            raise ValueError("丢手超时超出范围")
        if not 0.005 <= self.drag_distance <= 0.15:
            raise ValueError("拖拽阈值超出范围")
        if not (
            0 <= self.box_left < self.box_right <= 1 and 0 <= self.box_top < self.box_bottom <= 1
        ):
            raise ValueError("交互区域无效")
        if self.box_right - self.box_left < 0.15 or self.box_bottom - self.box_top < 0.15:
            raise ValueError("交互区域太小")
        if not 1 <= self.scroll_gain <= 100:
            raise ValueError("滚动增益超出范围")
