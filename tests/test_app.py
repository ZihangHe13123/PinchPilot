import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from pinchpilot import app as app_module
from pinchpilot.domain import EngineResult, InputEvent
from pinchpilot.widgets import PracticeView


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    win = app_module.MainWindow(tmp_path, demo=True)
    win.timer.stop()
    yield win
    win.close()


def test_demo_cannot_record_or_enable_native_input(window, monkeypatch):
    def forbidden():
        pytest.fail("synthetic source attempted OS input")

    monkeypatch.setattr(app_module, "MouseOutput", forbidden)
    window._tick()
    window.start_recording()
    window._toggle_live(True)
    assert window.recorder is None and window.countdown is None
    assert window.output is None
    assert not window.live_button.isChecked()
    assert not list(window.record_dir.glob("*.jsonl"))


def test_stopped_download_cannot_later_start_camera(window, monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(app_module, "default_model_path", lambda: tmp_path / "missing.task")
    monkeypatch.setattr(window, "_job", lambda fn, done: captured.append(done))
    monkeypatch.setattr(window, "_begin_camera", lambda: pytest.fail("camera restarted after stop"))
    window.start_camera()
    assert len(captured) == 1
    window.stop_all()
    captured[0](Path("downloaded.task"))
    assert window.source == "none"


def test_async_job_completion_runs_on_gui_thread(window, application):
    from PySide6.QtCore import QThread

    seen = []
    window._job(lambda: 7, lambda result: seen.append((result, QThread.currentThread())))
    deadline = time.monotonic() + 5
    while window.jobs and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert not window.jobs
    assert seen == [(7, application.thread())]


def test_practice_completes_and_labels_synthetic_evidence(application, tmp_path):
    view = PracticeView()
    view.resize(800, 500)
    view.start(tmp_path, {"source": "synthetic_demo"}, virtual=True)
    for point in view.target_list:
        view.feed(
            EngineResult("PRESSED", point, [InputEvent("move", *point), InputEvent("down", *point)])
        )
        view.feed(EngineResult("POINT", point, [InputEvent("up", *point)]))
    rows = [json.loads(line) for line in view.path.read_text().splitlines()]
    assert rows[0]["source"] == "synthetic_demo"
    assert rows[-1]["completed"] and rows[-1]["hits"] == 8
    assert not view.active and not view.pressed


def test_switch_profile_releases_output_before_replacing_engine(window):
    class FakeOutput:
        closed = False

        def close(self):
            self.closed = True

    old = FakeOutput()
    window.output = old
    window.reset_profile()
    assert old.closed and window.output is None
    assert window.engine.state == "WAIT_OPEN"


def test_camera_waits_for_permission_and_stop_cancels_start(window, monkeypatch, application):
    from PySide6.QtCore import QThread

    callbacks = []

    def request(callback):
        assert QThread.currentThread() == application.thread()
        callbacks.append(callback)

    monkeypatch.setattr(app_module, "request_camera_access", request)
    monkeypatch.setattr(
        app_module, "CameraWorker", lambda *_: pytest.fail("late permission started stopped camera")
    )
    window._begin_camera()
    assert window.permission_pending and window.worker is None
    window.stop_all()
    callbacks[0](True)
    application.processEvents()
    assert window.worker is None and not window.permission_pending


def test_camera_permission_denial_is_recoverable(window, monkeypatch, application):
    monkeypatch.setattr(app_module, "request_camera_access", lambda callback: callback(False))
    window._begin_camera()
    application.processEvents()
    assert not window.permission_pending and window.worker is None
    assert "未获准" in window.notice.text()
