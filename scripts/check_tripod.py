"""Synthetic tripod UI, target task and declared-noise checks; no hardware."""

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from pinchpilot.app import MainWindow
from pinchpilot.stability_probe import StabilityProbe
from pinchpilot.tripod import TripodEngine
from pinchpilot.tripod_demo import synthetic_tripod


def noise_check(kind):
    engine = TripodEngine()
    probe = StabilityProbe(101, {"source": "synthetic_engineering", "noise_profile": kind})
    rng = np.random.default_rng(42)
    for i in range(300):
        timestamp = 100 + i / 30
        noise = rng.normal(0, 0.002, 2)
        if kind == "correlated":
            noise = rng.normal(0, 0.001, 2) + 0.005 * np.array([np.sin(i / 15), np.cos(i / 18)])
        if i < 30:
            noise *= 0  # Known neutral before the declared synthetic disturbance.
        result = engine.process(synthetic_tripod(timestamp, noise=tuple(noise)))
        probe.add(timestamp, result)
    report = probe.report(110)
    assert report["sufficient"]
    return report


def main():
    application = QApplication.instance() or QApplication([])
    directory = Path("reports/verification/tripod").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    window = MainWindow(directory / "task-workspace", demo=True, interaction="tripod")
    window.timer.stop()
    window.resize(1240, 860)
    window.show()
    engine = window.tripod_engine
    timestamp = 100.0

    def frames(n, point=(0.5, 0.38), contact=0.85):
        nonlocal timestamp
        for _ in range(n):
            timestamp += 1 / 30
            frame = synthetic_tripod(timestamp, *point, contact=contact)
            result = engine.process(frame)
            window._dispatch(result)
            window.camera_view.box = engine.active_box
            window.camera_view.set_frame(None, frame, result, True)
            window.metrics.setText(result.hint + "\n合成关键点 · 无相机或系统输入")

    try:
        window.start_practice()
        frames(10)
        current = np.array([0.5, 0.38])
        for index, target in enumerate(window.practice.target_list):
            point = (
                np.array(engine.camera_anchor)
                + (np.array(target) - engine.pointer_anchor) * engine.config.span
            )
            for value in np.linspace(current, point, 20):
                frames(1, value)
            frames(15, point)
            frames(3, point + 0.002, 0.32)
            if index == 0:
                application.processEvents()
                assert window.grab().save(str(directory / "tripod-preview.png"))
            frames(4, point + 0.004, 0.10)
            if index == 1:
                window.tabs.setCurrentIndex(1)
                application.processEvents()
                assert window.grab().save(str(directory / "tripod-practice.png"))
            frames(5, point)
            current = point
        assert window.practice.index == 8 and window.practice.misses == 0
        summary = {
            "source": "synthetic_engineering",
            "camera_opened": False,
            "os_events_sent": False,
            "targets_hit": 8,
            "misses": 0,
            "task_log": str(window.practice.path),
            "noise_profiles": {},
        }
        for kind in ("independent", "correlated"):
            report = noise_check(kind)
            (directory / f"noise-{kind}.json").write_text(json.dumps(report, indent=2))
            summary["noise_profiles"][kind] = {
                "raw_rms": report["raw"]["rms_radius"],
                "output_rms": report["output"]["rms_radius"],
            }
        summary["limits"] = (
            "Declared synthetic noise only. No real tracking, intent, fatigue or accuracy conclusion."
        )
        (directory / "checks.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        window.close()


if __name__ == "__main__":
    main()
