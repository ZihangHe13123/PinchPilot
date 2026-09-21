import math


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
