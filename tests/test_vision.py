import hashlib
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

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
    assert json.loads(path.with_suffix(".json").read_text())["sha256"] == vision.MODEL_SHA256
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
    monkeypatch.setattr(vision, "Tracker", lambda _: detector)
    cv = SimpleNamespace(VideoCapture=Mock())
    monkeypatch.setitem(sys.modules, "cv2", cv)
    worker = vision.CameraWorker(tmp_path)
    worker.stop()
    worker.run()
    cv.VideoCapture.assert_not_called()
    detector.close.assert_called_once()


def test_windows_capture_fallback_and_cleanup_without_hardware(tmp_path, monkeypatch):
    failed = SimpleNamespace(isOpened=lambda: False, release=Mock())
    camera = SimpleNamespace(
        isOpened=lambda: True,
        release=Mock(),
        set=Mock(),
        read=Mock(side_effect=[(True, "rgb"), (False, None)]),
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
    detector = SimpleNamespace(process=lambda rgb, t: HandFrame(t), close=Mock())
    monkeypatch.setattr(vision, "Tracker", lambda _: detector)
    monkeypatch.setattr(vision, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "cv2", cv)
    worker = vision.CameraWorker(tmp_path)
    worker.run()
    assert [call.args for call in capture.call_args_list] == [(0, 1), (0, 2)]
    assert "未返回画面" in worker.failure
    assert worker.pop().rgb == "rgb"
    assert worker.pop() is None
    failed.release.assert_called_once()
    camera.release.assert_called_once()
    detector.close.assert_called_once()
