import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from test_desktop_control import Rig

from pinchpilot.cli import main
from pinchpilot.desktop import DesktopWindow
from pinchpilot.domain import HandFrame
from pinchpilot.tripod_demo import synthetic_tripod


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_capture_controller_isolated_and_manual_reenable(tmp_path):
    rig = Rig(tmp_path)
    rig.live()
    rig.frames(4, contact=0.1)
    assert rig.output.down
    rig.control.begin_capture()
    assert not rig.control.active and not rig.output.down and rig.control.capture_active
    before = list(rig.output.events)
    rig.frames(10)
    with pytest.raises(RuntimeError, match="采集"):
        rig.control.enable()
    with pytest.raises(RuntimeError, match="采集"):
        rig.control.begin_practice()
    assert rig.output.events == before
    rig.control.end_capture()
    assert not rig.control.active
    rig.control.enable()
    assert rig.control.active
    rig.control.close()


def test_capture_open_only_cannot_record_or_activate_native(application, tmp_path):
    rig = Rig(tmp_path / "reports/desktop_trials")
    window = DesktopWindow(tmp_path, controller=rig.control)
    window.timer.stop()
    try:
        window.open_contact_capture()
        child = window.contact_capture_window
        assert child is not None and not child.recorder_active and rig.control.capture_active
        assert not child.start_button.isEnabled()
        assert not rig.created
        assert not list((tmp_path / "data").glob("contact_recordings/*.jsonl"))
        window.open_practice()
        assert window.contact_capture_window is None
        assert window.practice_window is not None and rig.control.practice_active
        window.open_contact_capture()
        assert window.practice_window is None and rig.control.capture_active
        window._configure()
        assert window.contact_capture_window is None and not rig.control.capture_active
    finally:
        window.close()


def test_capture_constructor_failure_cannot_leave_bypass(application, tmp_path, monkeypatch):
    import pinchpilot.contact_capture as module

    rig = Rig(tmp_path)
    window = DesktopWindow(tmp_path, controller=rig.control)
    window.timer.stop()
    try:
        original = module.ContactCaptureWindow

        def fail(*args, **kwargs):
            raise RuntimeError("injected initialization failure")

        monkeypatch.setattr(module, "ContactCaptureWindow", fail)
        window.open_contact_capture()
        assert window.contact_capture_window is None and not rig.control.capture_active
        monkeypatch.setattr(module, "ContactCaptureWindow", original)
        window.open_contact_capture()
        assert window.contact_capture_window is not None and rig.control.capture_active
        with pytest.raises(RuntimeError, match="采集"):
            rig.control.enable()
    finally:
        window.close()


def test_ml_cli_fixture_inspection_template_and_default_synthetic_refusal(tmp_path, capsys):
    root = tmp_path / "fixture"
    assert main(["ml-fixture", "--output", str(root)]) == 0
    capsys.readouterr()
    manifest = root / "manifest.json"
    assert main(["ml-inspect", str(manifest)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"] == "synthetic" and report["windows"]["test"] == 233
    recording = root / "synthetic_P1.jsonl"
    blank = tmp_path / "blank-labels.json"
    assert main(["ml-label-template", str(recording), "--output", str(blank)]) == 0
    assert not json.loads(blank.read_text())["reviewed"]
    assert main(["ml-train", str(manifest), "--output", str(tmp_path / "model")]) == 2
    assert "合成数据默认拒绝" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()


def test_windows_cannot_skip_missing_frame_into_existing_history():
    from pinchpilot.contact_data import WindowBuilder

    builder = WindowBuilder()
    for i in range(8):
        result = builder.push(synthetic_tripod(100 + i / 30))
    assert result is not None
    assert builder.push(HandFrame(100 + 8 / 30)) is None
    for i in range(7):
        assert builder.push(synthetic_tripod(100 + (9 + i) / 30)) is None
    assert builder.push(synthetic_tripod(100 + 16 / 30)) is not None
