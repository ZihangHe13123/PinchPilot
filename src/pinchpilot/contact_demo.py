"""A schematic hand that acts out each step of the guided session, for the person to copy.

Pure geometry and timing; no Qt, camera or file access. The hand is a small 3-D skeleton
with the 21 MediaPipe joints. Each fingertip is given a target and the finger is bent to
reach it, so bones keep their length while fingers open and close. It is a drawing aid: none
of this is used for labels, features or control.
"""

import math
from dataclasses import dataclass

import numpy as np

from .contact_protocol import KEY

# A right hand seen as in the mirrored preview: palm towards the viewer (+z), fingers up
# (+y), thumb on the left (-x). Lengths are in palm widths, roughly.
WRIST = np.array([0.0, 0.0, 0.0])
# The two corners of the wrist and the end of the forearm stub, for drawing only.
WRIST_SIDES = np.array([[-0.26, 0.0, 0.0], [0.27, 0.0, 0.0], [-0.24, -0.3, 0.0], [0.25, -0.3, 0.0]])
THUMB_BASE = np.array([-0.36, 0.22, 0.08])
BASES = {
    "index": np.array([-0.30, 0.98, 0.0]),
    "middle": np.array([-0.08, 1.02, 0.0]),
    "ring": np.array([0.13, 0.97, 0.0]),
    "pinky": np.array([0.32, 0.86, 0.0]),
}
BONES = {
    "thumb": (0.40, 0.32, 0.27),
    "index": (0.44, 0.27, 0.21),
    "middle": (0.49, 0.31, 0.22),
    "ring": (0.45, 0.29, 0.21),
    "pinky": (0.34, 0.21, 0.19),
}
FIRST_JOINT = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
# Where each fingertip rests in a relaxed hand: loosely curled, ring and little finger more.
REST = {
    "thumb": THUMB_BASE + (-0.66, 0.66, 0.30),
    "index": BASES["index"] + (-0.08, 0.80, 0.26),
    "middle": BASES["middle"] + (0.0, 0.88, 0.28),
    "ring": BASES["ring"] + (0.05, 0.74, 0.36),
    "pinky": BASES["pinky"] + (0.09, 0.54, 0.34),
}
# Where fingertips meet the thumb.
GRIP = np.array([-0.20, 0.84, 0.62])  # thumb and middle finger
INDEX_ALONE = np.array([-0.36, 0.80, 0.58])  # thumb and index, middle finger open
INDEX_ON_GRIP = GRIP + (-0.07, 0.02, -0.01)  # index beside the middle finger on the thumb
RING_ALONE = np.array([-0.02, 0.76, 0.60])
RING_ON_GRIP = GRIP + (0.075, -0.03, -0.01)
NEAR = 0.6  # How far a finger goes towards the thumb when asked to come close, not touch.
TOUCHING = 0.09  # Tips this close are drawn as touching; a fingertip has some width.

# The rhythm the hand shows: (first touch, period, length of a touch) in seconds.
TAP = (1.2, 2.6, 0.7)
HOLD = (1.2, 4.0, 2.0)
APPROACH = (1.0, 2.3, 0.8)
RAMP = 0.25  # A finger takes this long to close or open.


@dataclass(frozen=True)
class DemoState:
    """What the hand is doing at one moment of a step."""

    grip: float = 0.0  # 0 open .. 1 thumb and middle finger together
    index: float = 0.0  # 0 away .. 1 on the thumb
    ring: float = 0.0
    space: bool | None = None  # None: this step never uses Space
    shift: tuple = (0.0, 0.0)  # the whole hand moved sideways and up
    turn: float = 0.0  # extra rotation about the forearm, in degrees


def _chain(base, target, lengths, pole, coupling):
    """Three bones from `base` whose tip reaches `target`, bulging towards `pole`."""
    l1, l2, l3 = lengths
    reach = target - base
    distance = float(np.linalg.norm(reach))

    def tip(bend):
        return (
            l1 + l2 * math.cos(bend) + l3 * math.cos((1 + coupling) * bend),
            -l2 * math.sin(bend) - l3 * math.sin((1 + coupling) * bend),
        )

    low, high = 0.0, 2.2
    wanted = min(distance, (l1 + l2 + l3) * 0.999)
    for _ in range(40):  # The tip comes closer to the base as the finger bends more.
        middle = (low + high) / 2
        if math.hypot(*tip(middle)) > wanted:
            low = middle
        else:
            high = middle
    bend = (low + high) / 2
    lift = -math.atan2(tip(bend)[1], tip(bend)[0])  # Turn the chain so its tip is on the axis.
    along = reach / distance
    side = pole - np.dot(pole, along) * along
    side = side / np.linalg.norm(side)
    points, angle, here = [base], lift, np.zeros(2)
    for length, step in zip(lengths, (0.0, bend, coupling * bend)):
        angle -= step
        here = here + length * np.array([math.cos(angle), math.sin(angle)])
        points.append(base + here[0] * along + here[1] * side)
    return points


def hand_points(grip=0.0, index=0.0, ring=0.0):
    """The 21 joints, [21, 3], for these amounts of closing. All arguments are 0..1."""
    grip, index, ring = (min(1.0, max(0.0, float(value))) for value in (grip, index, ring))
    free = 1 - grip
    # With the grip closed the thumb stays on the middle finger and the others come to it.
    thumb = REST["thumb"] + grip * (GRIP - REST["thumb"])
    thumb = thumb + free * index * (INDEX_ALONE - REST["thumb"])
    thumb = thumb + free * ring * (RING_ALONE - REST["thumb"])
    index_to = INDEX_ALONE + grip * (INDEX_ON_GRIP - INDEX_ALONE)
    ring_to = RING_ALONE + grip * (RING_ON_GRIP - RING_ALONE)
    targets = {
        "thumb": thumb,
        "index": REST["index"] + index * (index_to - REST["index"]),
        "middle": REST["middle"] + grip * (GRIP - REST["middle"]),
        "ring": REST["ring"] + ring * (ring_to - REST["ring"]),
        "pinky": REST["pinky"] + 0.25 * ring * (ring_to - REST["pinky"]) * 0.3,
    }
    points = np.zeros((21, 3))
    points[0] = WRIST
    for name, target in targets.items():
        thumb_like = name == "thumb"
        chain = _chain(
            THUMB_BASE if thumb_like else BASES[name],
            target,
            BONES[name],
            np.array([-1.0, 0.1, 0.5]) if thumb_like else np.array([0.0, 1.0, 0.25]),
            0.6 if thumb_like else 0.75,
        )
        first = FIRST_JOINT[name]
        points[first : first + 4] = chain
    return points


def project(points, yaw=0.0, pitch=0.0):
    """Turn the hand (degrees) and drop depth: ([21, 2] screen x right / y up, [21] depth)."""
    a, b = math.radians(yaw), math.radians(pitch)
    x = points[:, 0] * math.cos(a) + points[:, 2] * math.sin(a)
    z = -points[:, 0] * math.sin(a) + points[:, 2] * math.cos(a)
    y = points[:, 1] * math.cos(b) - z * math.sin(b)
    depth = points[:, 1] * math.sin(b) + z * math.cos(b)
    return np.column_stack((x, y)), depth


def _pulse(moment, seconds, rhythm):
    """(closing 0..1, touching) for touches repeated in this rhythm within a step."""
    first, period, length = rhythm
    if moment < first - RAMP:
        return 0.0, False
    number = math.floor((moment - first + RAMP) / period)
    start = first + number * period
    if start + length + RAMP > seconds - 0.3:  # No touch that the step would cut short.
        return 0.0, False
    since = moment - start
    if since < 0:
        return 1 + since / RAMP, False
    if since <= length:
        return 1.0, True
    return max(0.0, 1 - (since - length) / RAMP), False


def touches(step, rhythm=None):
    """How many touches the hand shows during a step."""
    rhythm = rhythm or (HOLD if "hold" in step.name else TAP)
    first, period, length = rhythm
    count = 0
    while first + count * period + length + RAMP <= step.seconds - 0.3:
        count += 1
    return count


def demo_state(step, moment):
    """What the hand does `moment` seconds into `step`."""
    moment = min(max(0.0, moment), step.seconds)
    closing = min(1.0, moment / 0.4)
    wants_grip = step.labels[0] == 1
    grip = closing if wants_grip else max(0.0, 1 - moment / 0.4) if step.name == "grip_off" else 0.0
    if step.name in ("grip_on", "grip_off"):
        return DemoState(grip=grip)
    if "near" in step.name:
        amount, _ = _pulse(moment, step.seconds, APPROACH)
        return DemoState(grip=grip, index=NEAR * amount, space=False)
    if KEY in step.labels:
        amount, touching = _pulse(moment, step.seconds, HOLD if "hold" in step.name else TAP)
        marked = step.labels.index(KEY)
        return DemoState(
            grip=grip,
            index=amount if marked == 1 else 0.0,
            ring=amount if marked == 2 else 0.0,
            space=touching,
        )
    if step.name == "grip_move":
        return DemoState(
            grip=grip,
            shift=(0.12 * math.sin(moment * 1.6), 0.07 * math.sin(moment * 2.3)),
            turn=10 * math.sin(moment * 1.1),
        )
    if step.name == "adjust":
        return DemoState(
            shift=(0.1 * math.sin(moment * 1.3), 0.06 * math.sin(moment * 2.0)),
            turn=22 * math.sin(moment * 1.5),
        )
    return DemoState(grip=grip)


def tip_gaps(points):
    """Distance from the thumb tip to the middle, index and ring fingertips."""
    return tuple(float(np.linalg.norm(points[4] - points[tip])) for tip in (12, 8, 16))
