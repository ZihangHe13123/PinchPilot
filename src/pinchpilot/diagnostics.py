"""Read-only explanations of the current gesture gate, without camera data.

These are observed engine phases, not correctness labels or inferred intent.
Reset causes cannot be recovered from a reset engine; callers should retain
explicit capture/selector reasons separately. No hint text is parsed here.
"""

import math
from numbers import Real

REASON_LABELS = {
    "paused": "手势控制已暂停",
    "no_hand": "未检测到手",
    "invalid_hand_geometry": "手部数据不可用",
    "other_hand_only": "等待指定的控制手",
    "multiple_hands": "多只手待选择，请先只露出控制手",
    "ambiguous_control_hand": "控制手候选不唯一，暂缓接管",
    "hands_too_close": "双手太近，等待分开",
    "confirming_control_hand": "正在确认控制手",
    "handedness_uncertain": "左右手分类分数不足，暂缓接管",
    "tracking_unavailable": "暂无可用的控制手数据，具体原因未知",
    "wrist_calibration_needed": "腕动方向尚未校准",
    "grip_unstable": "拇中捏合暂不稳定，指针暂停",
    "grip_confirming": "正在确认拇中捏合",
    "wait_grip": "等待拇中捏合接管指针",
    "wait_index_clear": "等待食指与拇指分开",
    "wait_both_clear": "等待食指、无名指与拇指分开",
    "moving": "已接管，可移动指针",
    "clutch_frozen": "拇中已松开，位置锁住",
    "left_approach": "食指接近拇指，点击前冻结",
    "left_contact_confirming": "正在确认食拇接触",
    "contact_clear_confirming": "正在确认手指已分开",
    "left_pressed": "左键按住，等待松开或拖拽",
    "left_release_confirming": "正在确认左键松开，指针冻结",
    "dragging": "左键按住，可移动拖拽",
    "drag_frozen": "左键按住，拇中松开使拖拽暂停",
    "right_approach": "无名指接近拇指，右键前冻结",
    "right_contact_confirming": "正在确认拇指与无名指接触",
    "right_release_wait": "右键已触发，等待无名指移开",
    "scroll_ready": "V 手势已准备，等待上下移动",
    "scrolling": "V 手势滚动中，指针锁住",
    "unknown": "当前阶段无法解释",
}

HAND_REASONS = frozenset(
    (
        "no_hand",
        "invalid_hand_geometry",
        "other_hand_only",
        "multiple_hands",
        "ambiguous_control_hand",
        "hands_too_close",
        "confirming_control_hand",
        "handedness_uncertain",
    )
)


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _phase(engine, result, packet):
    state = getattr(result, "state", "")
    if state == "PAUSED":
        return "paused"
    if state not in (
        "WAIT_GRIP",
        "WAIT_CLEAR",
        "CONTROL",
        "FROZEN",
        "APPROACH",
        "PRESSED",
        "DRAG",
        "RIGHT_APPROACH",
        "RIGHT_TOUCHED",
        "SCROLL",
    ):
        return "unknown"
    if not _finite(getattr(result, "grip", None)):
        hand_reason = getattr(packet, "hand_reason", "")
        return hand_reason if hand_reason in HAND_REASONS else "tracking_unavailable"
    if getattr(engine, "grip_pending_since", None) is not None:
        return "grip_unstable"
    if state == "WAIT_CLEAR":
        return "wait_both_clear" if engine.require_both_clear else "wait_index_clear"
    if state == "SCROLL":
        return "scrolling"
    if state in ("PRESSED", "DRAG"):
        if engine.clear_since is not None:
            return "left_release_confirming"
        if state == "DRAG":
            return "dragging" if engine.motion_engaged else "drag_frozen"
        return "left_pressed"
    if state == "RIGHT_TOUCHED":
        return "right_release_wait"
    if state in ("APPROACH", "RIGHT_APPROACH"):
        if engine.clear_since is not None:
            return "contact_clear_confirming"
        side = "left" if state == "APPROACH" else "right"
        return (
            f"{side}_contact_confirming" if engine.touch_since is not None else f"{side}_approach"
        )
    if engine.scroll_since is not None:
        return "scroll_ready"
    if state in ("WAIT_GRIP", "FROZEN") and engine.arm_since is not None:
        return "grip_confirming"
    if state == "FROZEN":
        return "clutch_frozen"
    if state == "CONTROL":
        return "moving"
    if engine.config.pointer_basis == "wrist" and engine.wrist_mapping is None:
        return "wrist_calibration_needed"
    # WAIT_GRIP also follows a safety reset. It does not by itself identify
    # whether the preceding frame jumped, timed out, or changed handedness.
    return "wait_grip"


def inspect_engine(engine, result, packet=None) -> dict:
    """Return stable reason/label/state and a small whitelist of scalar values.

    Positive margins satisfy that numeric threshold: engage/keep grip,
    left/right touch, or left/right clear. Other geometry and timing gates
    still apply. A negative margin is the shortfall. ``compatible_quality``
    describes the latest palm-assist 2-D fit, not joint visibility or a
    calibrated probability; the fit may predate a frozen state. All values
    are finite and JSON-safe.
    """
    reason = _phase(engine, result, packet)
    values = {
        "motion_engaged": bool(getattr(engine, "motion_engaged", False)),
        "left_down": bool(getattr(engine, "left_down", False)),
        "cancelled": bool(getattr(result, "cancelled", False)),
    }
    config = getattr(engine, "config", None)
    for key, observation, threshold, direction in (
        ("grip_engage_margin", "grip", "grip_engage", -1),
        ("grip_keep_margin", "grip", "grip_release", -1),
        ("left_touch_margin", "contact", "touch_ratio", -1),
        ("left_clear_margin", "contact", "clear_ratio", 1),
        ("right_touch_margin", "right_contact", "right_touch_ratio", -1),
        ("right_clear_margin", "right_contact", "right_clear_ratio", 1),
    ):
        observed, boundary = getattr(result, observation, None), getattr(config, threshold, None)
        if _finite(observed) and _finite(boundary):
            margin = direction * (float(observed) - float(boundary))
            if math.isfinite(margin):
                values[key] = margin
    pending, now = getattr(engine, "grip_pending_since", None), getattr(engine, "last_time", None)
    if _finite(pending) and _finite(now) and now >= pending:
        elapsed = (float(now) - float(pending)) * 1000
        if math.isfinite(elapsed):
            values["grip_pause_ms"] = elapsed
    compatible = getattr(engine, "compatible_point", None)
    for key, attribute in (
        ("compatible_quality", "quality"),
        ("compatible_correction_px", "correction_px"),
    ):
        value = getattr(compatible, attribute, None)
        if _finite(value):
            values[key] = float(value)
    state = getattr(result, "state", "")
    return {
        "reason": reason,
        "label": REASON_LABELS[reason],
        "state": state if reason != "unknown" else "UNKNOWN",
        "values": values,
    }
