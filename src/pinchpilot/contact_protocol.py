"""The guided recording session: what to show, for how long, and the draft labels it implies.

Pure logic with no Qt, camera or file access. Prompts only organise a session. The labels
built here come from the prompt on screen and from the Space key the person holds while two
fingers touch; they are a draft for human review, never ground truth.
"""

from dataclasses import dataclass

import numpy as np

from .contact_data import CHANNELS

NAME = "pinchpilot-guided-session-1"
KEY = "key"  # In a step's labels: the Space key marks this channel.

# Frames near a change of prompt or of the Space key stay unlabelled, in seconds.
SETTLE = 0.8  # after a prompt appears: reading and reaction time
LEAD = 0.3  # before a prompt ends: people anticipate the next one
KEY_EDGE = 0.07  # around each press or release of the Space key

POSES = (
    "手掌正对镜头",
    "手向内转一些，大约 45 度",
    "手侧过来，拇指一侧朝向镜头",
    "手离镜头近一些或远一些",
    "换成你觉得最放松的姿势",
)
MARK = "碰到时按住空格，离开时松开"
HOLD = "按住期间一直按着空格"


@dataclass(frozen=True)
class Step:
    name: str
    text: str
    seconds: float
    labels: tuple  # One of 0, 1, None or KEY for each of CHANNELS (middle, index, ring).
    note: str = ""
    round: int = 0  # 1-based; 0 for the countdown before the first round.


def _round(number, pose):
    steps = [
        Step("pose", f"换姿势：{pose}", 8, (None, None, None), "可以把手移开再放回来"),
        Step("open", "手放松张开，手指互不接触", 6, (0, 0, 0)),
    ]
    for _ in range(4):
        steps.append(Step("grip_on", "捏住：拇指和中指", 3, (1, 0, 0)))
        steps.append(Step("grip_off", "松开", 2.5, (0, 0, 0)))
    steps += [
        Step("grip_move", "捏住拇指和中指，小幅移动手或手腕", 10, (1, 0, 0)),
        Step("grip_index_tap", "保持捏住。食指点拇指，大约 8 次", 22, (1, KEY, 0), MARK),
        Step(
            "grip_index_hold",
            "保持捏住。食指按住拇指约 2 秒再松开，大约 4 次",
            18,
            (1, KEY, 0),
            HOLD,
        ),
        Step("index_tap", "松开中指。只用食指点拇指，大约 6 次", 16, (0, KEY, 0), MARK),
        Step("ring_tap", "松开中指。拇指碰无名指，大约 5 次", 14, (0, 0, KEY), MARK),
        Step("ring_hold", "拇指按住无名指约 2 秒再松开，大约 3 次", 12, (0, 0, KEY), HOLD),
        Step("grip_ring_tap", "捏住拇指和中指。无名指也碰拇指，大约 4 次", 12, (1, 0, KEY), MARK),
        Step("index_near", "食指靠近拇指但不要碰到，来回约 5 次", 12, (0, 0, 0), "不要按空格"),
        Step(
            "grip_index_near",
            "捏住拇指和中指。食指靠近拇指但不要碰到，约 5 次",
            12,
            (1, 0, 0),
            "不要按空格",
        ),
        Step("adjust", "手自然动一动：转转手腕、换个位置，手指互不接触", 7, (0, 0, 0)),
    ]
    return [Step(s.name, s.text, s.seconds, s.labels, s.note, number) for s in steps]


def session():
    """The steps of one full session, about 14 minutes."""
    steps = [Step("ready", "准备：把一只手放进画面", 4, (None, None, None))]
    for number, pose in enumerate(POSES, 1):
        steps += _round(number, pose)
    return steps


def timeline(steps):
    """[(start, end, step)] in seconds from the start of the session."""
    result, start = [], 0.0
    for step in steps:
        if step.seconds <= 0 or len(step.labels) != len(CHANNELS):
            raise ValueError("步骤时长或标签数量无效")
        result.append((start, start + step.seconds, step))
        start += step.seconds
    return result


def position(schedule, elapsed):
    """Index of the step shown at `elapsed` seconds, or None before the start or after the end."""
    for index, (start, end, _) in enumerate(schedule):
        if start <= elapsed < end:
            return index
    return None


def key_spans(events, end):
    """Spans (down, up) from [(time, pressed)] events; a key still held is released at `end`."""
    spans, down = [], None
    for moment, pressed in events:
        if pressed and down is None:
            down = moment
        elif not pressed and down is not None:
            spans.append((down, moment))
            down = None
    if down is not None:
        spans.append((down, end))
    return spans


def unmarked(schedule, events, end):
    """Indices of the steps begun before `end` that ask for Space marks and got none."""
    spans = key_spans(events, end)
    return [
        index
        for index, (start, stop, step) in enumerate(schedule)
        if KEY in step.labels
        and start < end
        and not any(down < stop and up > start for down, up in spans)
    ]


def label_array(times, present, schedule, events, settle=SETTLE, lead=LEAD, edge=KEY_EDGE):
    """Draft labels as an int array [frames, 3]: 0, 1, or -1 for no label.

    `times` are frame times in seconds from the start of the session and `present` says
    whether the frame has a hand. A frame gets no label without a hand, outside the session,
    within `settle` after a prompt appears or `lead` before it ends, or within `edge` of a
    press or release of the Space key. A step that asks for Space marks and got none leaves
    that channel unlabelled: the key may not have reached the window, so "never touched"
    would be a guess.
    """
    times = np.asarray(times, dtype=float)
    labels = np.full((len(times), len(CHANNELS)), -1, dtype=np.int8)
    if not schedule or not len(times):
        return labels
    starts = np.array([start for start, _, _ in schedule])
    ends = np.array([end for _, end, _ in schedule])
    step = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(schedule) - 1)
    usable = (
        np.asarray(present, dtype=bool)
        & (times >= starts[step] + settle)
        & (times < ends[step] - lead)
    )
    spans = key_spans(events, schedule[-1][1])
    held = np.zeros(len(times), dtype=bool)
    near = np.zeros(len(times), dtype=bool)
    if spans:
        downs = np.array([down for down, _ in spans])
        ups = np.array([up for _, up in spans])
        last = np.searchsorted(downs, times, side="right") - 1
        held = (last >= 0) & (times < ups[np.clip(last, 0, None)])
        edges = np.sort(np.concatenate((downs, ups)))
        after = np.clip(np.searchsorted(edges, times), 0, len(edges) - 1)
        before = np.clip(after - 1, 0, None)
        near = np.minimum(np.abs(times - edges[after]), np.abs(times - edges[before])) < edge
    silent = np.zeros(len(schedule), dtype=bool)
    silent[unmarked(schedule, events, schedule[-1][1])] = True
    for channel in range(len(CHANNELS)):
        wanted = [item.labels[channel] for _, _, item in schedule]
        marked = np.array([label == KEY for label in wanted])[step]
        fixed = np.array([-1 if label in (None, KEY) else label for label in wanted])[step]
        from_key = np.where(near | silent[step], -1, held.astype(np.int8))
        labels[:, channel] = np.where(usable, np.where(marked, from_key, fixed), -1)
    return labels


def draft_labels(times, present, schedule, events):
    """One (middle, index, ring) tuple of 0, 1 or None for each frame; see `label_array`."""
    rows = label_array(times, present, schedule, events).tolist()
    return [tuple(None if value < 0 else value for value in row) for row in rows]


def steps_from_record(record):
    """The schedule [(start, end, step)] that a saved `.protocol.json` describes."""
    try:
        schedule = [
            (
                float(item["start"]),
                float(item["end"]),
                Step(
                    str(item["name"]),
                    str(item["text"]),
                    float(item["end"]) - float(item["start"]),
                    tuple(item["labels"]),
                    round=int(item["round"]),
                ),
            )
            for item in record["steps"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("提示记录里的步骤格式无效") from error
    if not schedule or any(
        len(step.labels) != len(CHANNELS)
        or any(label not in (0, 1, None, KEY) for label in step.labels)
        or not start < end
        for start, end, step in schedule
    ):
        raise ValueError("提示记录里的步骤格式无效")
    return schedule


def intervals(labels):
    """Run-length encode per-frame labels into the annotation file's half-open intervals."""
    result, start = [], 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            result.append({"start": start, "end": index, **dict(zip(CHANNELS, labels[start]))})
            start = index
    return result
