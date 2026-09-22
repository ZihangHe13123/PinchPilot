"""Native click counts for macOS; never delay a click or move the pointer."""

import math
from copy import copy
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Click:
    button: str
    point: tuple[float, float]
    started: float
    count: int
    moved: bool = False


class ClickSequence:
    def __init__(self, interval: float, slop: float = 4.0):
        self.interval = interval
        # CoreGraphics display coordinates are logical points, including on Retina.
        self.slop = slop
        self.completed: Click | None = None
        self.held: dict[str, Click] = {}

    def prepare(self, kind, point, now):
        """Return proposed state and count; commit only after native output succeeds."""
        proposed = copy(self)
        proposed.held = self.held.copy()
        count = proposed._advance(kind, point, now)
        return proposed, count

    def _near(self, click, point):
        return math.dist(click.point, point) <= self.slop

    def _advance(self, kind, point, now):
        if kind == "move":
            if self.completed and not self._near(self.completed, point):
                self.completed = None
            for button, click in self.held.items():
                if not self._near(click, point):
                    self.held[button] = replace(click, moved=True)
            return 0

        button = "right" if kind.startswith("right_") else "left"
        if kind in ("down", "right_down"):
            previous = self.completed
            count = 1
            if (
                previous is not None
                and not self.held
                and previous.button == button
                and 0 <= now - previous.started <= self.interval
                and self._near(previous, point)
            ):
                count = previous.count + 1
            self.completed = None
            # Overlapping buttons cannot become a later multi-click sequence.
            self.held = {key: replace(click, moved=True) for key, click in self.held.items()}
            self.held[button] = Click(button, point, now, count, moved=bool(self.held))
            return count

        click = self.held.pop(button, None)
        self.completed = None
        if click is not None:
            if (
                not self.held
                and not click.moved
                and self._near(click, point)
                and 0 <= now - click.started <= self.interval
            ):
                self.completed = click
            return click.count
        return 1

    def interrupt(self):
        """Break the sequence but retain counts for matching cleanup/retry releases."""
        self.completed = None
        self.held = {key: replace(click, moved=True) for key, click in self.held.items()}

    def release_count(self, right):
        click = self.held.get("right" if right else "left")
        return click.count if click else 1

    def released(self, right):
        self.held.pop("right" if right else "left", None)
