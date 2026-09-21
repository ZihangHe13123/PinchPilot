import math
from collections import deque
from statistics import median


class AdaptiveFilter:
    """Speed-adaptive low-pass (One Euro family); timestamps are supplied by caller."""

    def __init__(self, minimum_cutoff: float = 1.6, beta: float = 0.30):
        self.minimum_cutoff = minimum_cutoff
        self.beta = beta
        self.last_time = None
        self.raw = None
        self.value = None
        self.velocity = 0.0

    @staticmethod
    def alpha(cutoff: float, dt: float) -> float:
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def update(self, value: float, timestamp: float) -> float:
        if self.last_time is None or timestamp - self.last_time > 0.3:
            self.last_time, self.raw, self.value = timestamp, value, value
            self.velocity = 0.0
            return value
        dt = max(timestamp - self.last_time, 1e-5)
        derivative = (value - self.raw) / dt
        a = self.alpha(1.0, dt)
        self.velocity += a * (derivative - self.velocity)
        a = self.alpha(self.minimum_cutoff + self.beta * abs(self.velocity), dt)
        self.value += a * (value - self.value)
        self.raw, self.last_time = value, timestamp
        return self.value


class StablePointer:
    """Median rejects isolated spikes; adaptive smoothing and a soft deadband
    suppress rest noise. The deadband only follows displacement beyond its radius,
    so crossing its boundary does not suddenly snap to the input point.
    """

    def __init__(self, radius=0.008, minimum_cutoff=3.0, beta=4.0):
        self.radius = radius
        self.history = deque(maxlen=3)
        self.filters = [AdaptiveFilter(minimum_cutoff, beta) for _ in range(2)]
        self.value = None

    def update(self, point, timestamp):
        if not self.history:
            self.history.extend([point] * self.history.maxlen)
        else:
            self.history.append(point)
        middle = tuple(median(p[axis] for p in self.history) for axis in range(2))
        smooth = tuple(f.update(v, timestamp) for f, v in zip(self.filters, middle))
        if self.value is None:
            self.value = smooth
        distance = math.dist(smooth, self.value)
        if distance > self.radius:
            fraction = (distance - self.radius) / distance
            self.value = tuple(a + (b - a) * fraction for a, b in zip(self.value, smooth))
        return self.value
