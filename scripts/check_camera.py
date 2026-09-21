"""Explicit camera check; requests permission on the UI thread and saves no images."""

import argparse
import json
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from pinchpilot.platform_io import camera_permission, request_camera_access
from pinchpilot.storage import save_json
from pinchpilot.vision import CameraWorker, fetch_model


class Check(QObject):
    authorized = Signal(bool)

    def __init__(self, args, model, application):
        super().__init__()
        self.args, self.model, self.application = args, model, application
        self.before = camera_permission()
        self.worker = None
        self.frames = self.hands = 0
        self.timings = []
        self.started = time.monotonic()
        self.ended = False
        self.result = {}
        self.authorized.connect(self.begin)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)
        QTimer.singleShot(0, lambda: request_camera_access(self.authorized.emit))

    def begin(self, allowed):
        if self.ended:
            return
        if not allowed:
            self.finish()
            return
        self.worker = CameraWorker(self.model, self.args.camera)
        self.worker.start()

    def tick(self):
        if self.worker:
            packet = self.worker.pop()
            if packet:
                self.frames += 1
                self.hands += bool(packet.frame.landmarks)
                self.timings.append(packet.inference_ms)
            if self.worker.failure:
                self.finish()
                return
        if time.monotonic() - self.started >= self.args.seconds:
            self.finish()

    def finish(self):
        self.ended = True
        self.timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=3)
        self.result = {
            "camera": self.args.camera,
            "requested_seconds": self.args.seconds,
            "frames_received": self.frames,
            "frames_with_landmarks": self.hands,
            "inference_mean_ms": sum(self.timings) / len(self.timings) if self.timings else None,
            "permission_before": self.before,
            "permission_after": camera_permission(),
            "failure": self.worker.failure
            if self.worker
            else "camera not started; permission denied or pending",
            "worker_stopped": not self.worker or not self.worker.is_alive(),
            "images_saved": False,
            "os_events_sent": False,
            "note": "Connectivity only. No accuracy or interaction evaluation.",
        }
        save_json(self.args.output, self.result)
        print(json.dumps(self.result, ensure_ascii=False, indent=2))
        self.application.quit()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--camera", type=int, default=0)
    cli.add_argument("--seconds", type=float, default=12)
    cli.add_argument("--output", type=Path, default=Path("reports/verification/camera.json"))
    args = cli.parse_args()
    if not 1 <= args.seconds <= 60:
        cli.error("seconds must be between 1 and 60")
    model = fetch_model()
    application = QApplication([])
    check = Check(args, model, application)
    application.exec()
    return (
        0
        if check.frames and not check.result.get("failure") and check.result["worker_stopped"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
