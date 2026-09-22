"""Optional screen-space pointer tuning. No camera, ML, or native input access."""

import math
from collections import deque
from statistics import median

from .filtering import AdaptiveFilter

MAX_CALIBRATED_RADIUS = 4.0


class ScreenSmoother:
    """Three-frame spike rejection followed by a pixel-scaled One Euro filter."""

    def __init__(self, width, height):
        self.size = (width, height)
        self.history = deque(maxlen=3)
        # beta is in logical screen units, unlike the normalized classic filter.
        self.filters = [AdaptiveFilter(6.0, 0.012) for _ in range(2)]

    def update(self, point, timestamp):
        pixel = tuple(v * size for v, size in zip(point, self.size))
        if not self.history:
            self.history.extend([pixel] * 3)
        else:
            self.history.append(pixel)
        middle = tuple(median(p[axis] for p in self.history) for axis in range(2))
        return tuple(f.update(v, timestamp) for f, v in zip(self.filters, middle))


def tuning_radius(config):
    noise = math.hypot(
        config.rest_noise_x * config.screen_width / config.span,
        config.rest_noise_y * config.screen_height / config.span,
    )
    # A noisy calibration cannot silently swallow arbitrarily large corrections.
    base = min(MAX_CALIBRATED_RADIUS, max(2.0, noise * 1.1))
    return base * config.deadband / 0.008


class TunedPointer:
    def __init__(self, config):
        self.size = (config.screen_width, config.screen_height)
        self.smoother = ScreenSmoother(*self.size)
        self.radius = tuning_radius(config)
        self.adaptive = config.motion_profile == "adaptive"
        self.intent = self.last_time = self.value = None
        self.gain = 1.0

    def update(self, point, timestamp):
        smooth = self.smoother.update(point, timestamp)
        if self.intent is None:
            self.intent = smooth
            self.value = tuple(min(size, max(0.0, v)) for v, size in zip(smooth, self.size))
            self.last_time = timestamp
            return tuple(v / size for v, size in zip(self.value, self.size))
        dt = max(timestamp - self.last_time, 1e-5)
        # Calibration measures this input space. Applying its radius after a
        # slow-speed gain would amplify the effective threshold by 1 / gain.
        intent = self.intent
        distance = math.dist(smooth, intent)
        if distance > self.radius:
            fraction = (distance - self.radius) / distance
            intent = tuple(a + (b - a) * fraction for a, b in zip(intent, smooth))
        delta = tuple(a - b for a, b in zip(intent, self.intent))
        speed = math.hypot(*delta) / dt
        if self.adaptive:
            u = min(1.0, max(0.0, (speed - 120.0) / (800.0 - 120.0)))
            self.gain = 0.55 + 0.80 * u * u * (3.0 - 2.0 * u)
        # Clamp the output, not the input reference: reversing at the screen
        # edge must not require unwinding off-screen hand travel.
        self.value = tuple(
            min(size, max(0.0, v + d * self.gain))
            for v, d, size in zip(self.value, delta, self.size)
        )
        self.intent, self.last_time = intent, timestamp
        return tuple(v / size for v, size in zip(self.value, self.size))
