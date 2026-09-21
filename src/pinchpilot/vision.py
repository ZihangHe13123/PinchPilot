import hashlib
import json
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .domain import HandFrame

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def default_model_path() -> Path:
    return Path.home() / ".cache" / "pinchpilot" / "hand_landmarker_v1.task"


def fetch_model(path: Path | None = None) -> Path:
    path = path or default_model_path()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError(f"模型文件校验失败，请移除后重下：{path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temporary.stat().st_size < 1_000_000:
            raise ValueError("下载内容不完整")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError("官方模型文件校验失败；未覆盖本地模型")
        temporary.replace(path)
        path.with_suffix(".json").write_text(
            json.dumps({"url": MODEL_URL, "sha256": digest}, indent=2)
        )
    finally:
        temporary.unlink(missing_ok=True)
    return path


class Tracker:
    def __init__(self, model_path: Path):
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def process(self, rgb, timestamp: float) -> HandFrame:
        stamp = max(self.last_ms + 1, int(timestamp * 1000))
        self.last_ms = stamp
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), stamp
        )
        aspect = rgb.shape[1] / rgb.shape[0]
        if not result.hand_landmarks:
            return HandFrame(timestamp, aspect=aspect)
        points = tuple((p.x, p.y, p.z) for p in result.hand_landmarks[0])
        handedness = result.handedness[0][0]
        return HandFrame(timestamp, points, aspect, handedness.category_name, handedness.score)

    def close(self) -> None:
        self.detector.close()


@dataclass
class Packet:
    rgb: object
    frame: HandFrame
    inference_ms: float
    fps: float
    captured_at: float


class CameraWorker(threading.Thread):
    """Single latest-result slot; GUI never queues unbounded stale camera frames."""

    def __init__(self, model_path: Path, camera: int = 0):
        super().__init__(daemon=True, name="pinchpilot-camera")
        self.model_path, self.camera = model_path, camera
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.failure = None

    def pop(self) -> Packet | None:
        with self.lock:
            packet, self.latest = self.latest, None
        return packet

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        import cv2

        cap = tracker = None
        try:
            tracker = Tracker(self.model_path)
            if self.stop_event.is_set():
                return
            backend = (
                cv2.CAP_AVFOUNDATION
                if sys.platform == "darwin"
                else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            )
            cap = cv2.VideoCapture(self.camera, backend)
            if not cap.isOpened() and sys.platform == "win32":
                cap.release()
                cap = cv2.VideoCapture(self.camera, cv2.CAP_MSMF)
            if not cap.isOpened():
                raise RuntimeError("摄像头无法打开。检查相机编号、系统权限和其他占用相机的程序。")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            last = time.monotonic()
            fps = 0.0
            while not self.stop_event.is_set():
                ok, bgr = cap.read()
                captured = time.monotonic()
                if not ok:
                    raise RuntimeError("相机未返回画面，已停止控制")
                # Mirror exactly once, before both feature extraction and preview.
                rgb = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
                started = time.monotonic()
                frame = tracker.process(rgb, captured)
                inference_ms = (time.monotonic() - started) * 1000
                fps = 0.9 * fps + 0.1 / max(captured - last, 0.001)
                last = captured
                with self.lock:
                    self.latest = Packet(rgb, frame, inference_ms, fps, captured)
        except Exception as error:
            self.failure = str(error)
        finally:
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()
