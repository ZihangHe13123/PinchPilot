import hashlib
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pinchpilot import vision
from pinchpilot.domain import HandFrame


def test_model_download_verifies_bytes_and_cache(tmp_path, monkeypatch):
    payload = b"engineering model fixture" * 50000
    monkeypatch.setattr(vision, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    download = Mock(side_effect=lambda *a, **k: io.BytesIO(payload))
    monkeypatch.setattr(vision.urllib.request, "urlopen", download)
    path = vision.fetch_model(tmp_path / "test.task")
    assert path.read_bytes() == payload
    assert (
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["sha256"]
        == vision.MODEL_SHA256
    )
    vision.fetch_model(path)
    assert download.call_count == 1
    path.write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="校验失败"):
        vision.fetch_model(path)


def test_bad_download_does_not_publish_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad" * 400000)
    )
    with pytest.raises(ValueError, match="校验失败"):
        vision.fetch_model(tmp_path / "test.task")
    assert not list(tmp_path.glob("*.task"))
    assert not list(tmp_path.glob("*.download"))


def test_cancelled_worker_never_opens_camera(tmp_path, monkeypatch):
    detector = SimpleNamespace(close=Mock())
    monkeypatch.setattr(vision, "Tracker", lambda _, **kwargs: detector)
    cv = SimpleNamespace(VideoCapture=Mock())
    monkeypatch.setitem(sys.modules, "cv2", cv)
    worker = vision.CameraWorker(tmp_path)
    worker.stop()
    worker.run()
    cv.VideoCapture.assert_not_called()
    detector.close.assert_called_once()


def test_tracker_requests_both_hands_and_selects_control_after_detector_reordering(
    tmp_path, monkeypatch
):
    from pinchpilot.demo import synthetic_hand

    def result(order):
        return SimpleNamespace(
            hand_landmarks=[
                [
                    SimpleNamespace(x=x, y=y, z=z)
                    for x, y, z in synthetic_hand(1, x=position).landmarks
                ]
                for _, position in order
            ],
            handedness=[[SimpleNamespace(category_name=side, score=0.99)] for side, _ in order],
        )

    right, left = ("Right", 0.35), ("Left", 0.7)
    detector = SimpleNamespace(
        detect_for_video=Mock(
            side_effect=[result([right])] * 3 + [result([left, right]), result([left])]
        ),
        close=Mock(),
    )
    base_options = Mock(return_value="base")
    base_options.Delegate.CPU = "CPU"
    options = Mock(return_value="options")
    creator = Mock(return_value=detector)
    mp = SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=base_options,
            vision=SimpleNamespace(
                HandLandmarkerOptions=options,
                RunningMode=SimpleNamespace(VIDEO="VIDEO"),
                HandLandmarker=SimpleNamespace(create_from_options=creator),
            ),
        ),
        Image=Mock(return_value="image"),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    tracker = vision.Tracker(tmp_path / "model.task")
    options.assert_called_once()
    assert options.call_args.kwargs["num_hands"] == 2
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(4):
        selected = tracker.process(rgb, 1 + i / 30)
    assert selected.handedness == "Right"
    assert selected.landmarks == synthetic_hand(1, x=0.35).landmarks
    assert not tracker.process(rgb, 1 + 4 / 30).landmarks
    assert tracker.selector.locked_hand == "Right"
    tracker.close()
    detector.close.assert_called_once()


@pytest.mark.parametrize("frame_count", [1, 3])
def test_windows_capture_fallback_and_cleanup_without_hardware(tmp_path, monkeypatch, frame_count):
    failed = SimpleNamespace(isOpened=lambda: False, release=Mock())
    camera = SimpleNamespace(
        isOpened=lambda: True,
        release=Mock(),
        set=Mock(),
        read=Mock(side_effect=[(True, "rgb")] * frame_count + [(False, None)]),
    )
    capture = Mock(side_effect=[failed, camera])
    cv = SimpleNamespace(
        CAP_DSHOW=1,
        CAP_MSMF=2,
        CAP_ANY=0,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=6,
        VideoCapture=capture,
        flip=lambda value, _: value,
        cvtColor=lambda value, _: value,
    )
    detector = SimpleNamespace(
        process=lambda rgb, t: HandFrame(t),
        close=Mock(),
        selector=SimpleNamespace(status="等待右手"),
    )
    create_tracker = Mock(return_value=detector)
    monkeypatch.setattr(vision, "Tracker", create_tracker)
    monkeypatch.setattr(vision, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "cv2", cv)
    worker = vision.CameraWorker(tmp_path, preferred_hand="Right")
    worker.run()
    assert [call.args for call in capture.call_args_list] == [(0, 1), (0, 2)]
    assert "未返回画面" in worker.failure
    packet = worker.pop()
    assert packet.rgb == "rgb" and packet.hand_status == "等待右手"
    assert packet.overwritten_results == frame_count - 1
    assert packet.capture_ms >= 0 and packet.preprocess_ms >= 0
    assert packet.ready_at >= packet.captured_at
    assert packet.inference_ms <= (packet.ready_at - packet.captured_at) * 1000
    create_tracker.assert_called_once_with(tmp_path, preferred_hand="Right")
    assert worker.pop() is None
    failed.release.assert_called_once()
    camera.release.assert_called_once()
    detector.close.assert_called_once()
