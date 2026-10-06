"""Deterministic end-to-end demo check. No camera, OS output or human claims."""

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pinchpilot.app import MainWindow
from pinchpilot.demo import synthetic_finger
from pinchpilot.single_finger import FINGER_MODES, finger_features


def main():
    application = QApplication.instance() or QApplication([])
    output = Path("reports/verification/single-finger").resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "evidence": "synthetic engineering checks only",
        "camera_opened": False,
        "os_events_sent": False,
        "modes": {},
    }
    for mode in FINGER_MODES:
        window = MainWindow(output / mode, demo=True, interaction=mode)
        window.timer.stop()
        window.resize(1240, 820)
        window.show()
        engine = window.finger_engine
        timestamp = 100.0

        def frames(count, **kwargs):
            nonlocal timestamp
            for _ in range(count):
                timestamp += 1 / 30
                frame = synthetic_finger(timestamp, **kwargs)
                result = engine.process(frame)
                window._dispatch(result)
                window.camera_view.box = engine.active_box
                window.camera_view.set_frame(None, frame, result, demo=True)
                window.metrics.setText(result.hint + "\n合成关键点 · 工程检查 · 无系统输入")

        try:
            window.start_practice()
            frames(10)
            frames(8, dx=0.10, dy=-0.10)
            if mode == "finger-flex":
                frames(5, dx=0.10, dy=-0.10, bend=0.18)
            application.processEvents()
            assert window.grab().save(str(output / f"{mode}-preview.png"))
            window.tabs.setCurrentIndex(1)
            application.processEvents()
            assert window.grab().save(str(output / f"{mode}-practice.png"))
            # A fresh trial isolates the full eight-target check from screenshots.
            window.start_practice()
            frames(10)
            base = finger_features(synthetic_finger(timestamp)).relative
            for target in window.practice.target_list:
                dx, dy = tuple(
                    (v - p) / engine.config.gain + r - b
                    for v, p, r, b in zip(
                        target, engine.pointer_anchor, engine.relative_anchor, base
                    )
                )
                frames(40, dx=dx, dy=dy)
                if mode == "finger-flex":
                    frames(6, dx=dx, dy=dy, bend=0.18)
                    frames(6, dx=dx, dy=dy)
            rows = [
                json.loads(line)
                for line in window.practice.path.read_text(encoding="utf-8").splitlines()
            ]
            result = rows[-1]
            assert result["completed"] and result["hits"] == 8 and result["misses"] == 0
            summary["modes"][mode] = {
                "hits": result["hits"],
                "misses": result["misses"],
                "completed": True,
                "log": str(window.practice.path),
                "timing_note": "Generated frames are processed faster than real time; task duration is not a user metric.",
            }
        finally:
            window.close()
    (output / "checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
