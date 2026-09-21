"""Unlabelled, explicitly requested rest trace; never a classifier dataset."""

import math
from collections import Counter

import numpy as np


class StabilityProbe:
    def __init__(self, starts_at, metadata, duration=8.0):
        self.starts_at = starts_at
        self.duration = duration
        self.metadata = metadata
        self.rows = []

    def add(self, timestamp, result):
        if not self.starts_at <= timestamp <= self.starts_at + self.duration:
            return
        if self.rows and timestamp <= self.rows[-1]["timestamp"]:
            return
        raw, output = result.raw_pointer, result.pointer
        usable = (
            result.mode == "tripod"
            and result.state == "CONTROL"
            and raw is not None
            and output is not None
            and all(math.isfinite(v) and 0 <= v <= 1 for v in (*raw, *output))
        )
        self.rows.append(
            {
                "timestamp": timestamp,
                "state": result.state,
                "eligible": usable,
                "raw": raw,
                "output": output,
                "grip": result.grip,
                "contact": result.contact,
            }
        )

    @staticmethod
    def _spread(points):
        points = np.asarray(points)
        return {
            "rms_radius": float(
                np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
            ),
            "range_xy": np.ptp(points, axis=0).tolist(),
        }

    def report(self, ended_at, cancelled=False):
        selected = [row for row in self.rows if row["eligible"]]
        observed = sum(
            b["timestamp"] - a["timestamp"]
            for a, b in zip(self.rows, self.rows[1:])
            if a["eligible"] and b["eligible"] and 0 < b["timestamp"] - a["timestamp"] <= 0.2
        )
        coverage = len(selected) / max(1, len(self.rows))
        enough = (
            not cancelled
            and ended_at >= self.starts_at + self.duration
            and len(selected) >= 30
            and observed >= self.duration * 0.70
            and coverage >= 0.85
        )
        return {
            "type": "stationary_pointer_probe",
            "metadata": self.metadata,
            "instruction": "User was asked to hold thumb-middle grip still, index away; intent is not verified.",
            "units": "normalised screen x/y; spread about each trace mean, not accuracy or centimetres",
            "cancelled": cancelled,
            "sufficient": enough,
            "frames": len(self.rows),
            "eligible_frames": len(selected),
            "eligible_frame_fraction": coverage,
            "valid_observed_seconds": observed,
            "state_counts": dict(Counter(row["state"] for row in self.rows)),
            "raw": self._spread([row["raw"] for row in selected]) if selected else None,
            "output": self._spread([row["output"] for row in selected]) if selected else None,
            "rows": self.rows,
        }
