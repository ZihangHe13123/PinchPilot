import hashlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from pinchpilot import vision


@pytest.fixture
def portable(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/windows/portable.py"
    spec = importlib.util.spec_from_file_location("portable_trial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in (
        "PINCHPILOT_MODEL_PATH",
        "MPLCONFIGDIR",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ):
        monkeypatch.setenv(key, "")
    # Register restoration even if configure() removes an inherited Qt setting.
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    return module


def install_fake_model(root, monkeypatch):
    model = root / "models/hand_landmarker_v1.task"
    model.parent.mkdir()
    model.write_bytes(b"offline model test fixture")
    monkeypatch.setattr(vision, "MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    return model


def test_portable_model_uses_local_bundle_without_downloading(portable, tmp_path, monkeypatch):
    model = install_fake_model(tmp_path, monkeypatch)
    download = Mock(side_effect=AssertionError("unexpected network access"))
    monkeypatch.setattr(vision.urllib.request, "urlopen", download)
    portable.configure(tmp_path)
    assert vision.default_model_path() == model
    assert vision.fetch_model() == model
    portable.model_check(tmp_path)
    download.assert_not_called()
    monkeypatch.delenv("PINCHPILOT_MODEL_PATH")
    assert vision.default_model_path() == Path.home() / ".cache/pinchpilot/hand_landmarker_v1.task"


def test_missing_model_blocks_start_and_saves_error(portable, tmp_path, monkeypatch):
    execute = Mock(side_effect=AssertionError("child should not launch"))
    monkeypatch.setattr(portable, "execute", execute)
    assert portable.main(["start"], root=tmp_path) == 1
    execute.assert_not_called()
    logs = list((tmp_path / "reports/startup").glob("*-launcher-error.log"))
    assert len(logs) == 1 and "Bundled model missing" in logs[0].read_text(encoding="utf-8")


def test_normal_start_is_explicit_desktop_without_auto_control(portable, tmp_path, monkeypatch):
    install_fake_model(tmp_path, monkeypatch)
    execute = Mock(return_value=0)
    monkeypatch.setattr(portable, "execute", execute)
    assert portable.main(["start"], root=tmp_path) == 0
    arguments, root, log = execute.call_args.args
    assert arguments == ["-m", "pinchpilot", "desktop", "--workspace", str(tmp_path)]
    assert root == tmp_path and log.parent == tmp_path / "reports/startup"


def test_child_failure_captures_unicode_stderr(portable, tmp_path):
    folder = tmp_path / "中文 路径"
    folder.mkdir()
    log = folder / "startup.log"
    code = portable.execute(
        ["-c", "import sys; print('依赖加载失败', file=sys.stderr); sys.exit(7)"], folder, log
    )
    assert code == 7 and "依赖加载失败" in log.read_text(encoding="utf-8")


def test_child_ignores_host_python_environment(portable, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/does-not-exist-host-python")
    monkeypatch.setenv("PYTHONPATH", "/does-not-exist-host-packages")
    log = tmp_path / "isolated.log"
    code = portable.execute(
        ["-c", "import sys; assert sys.flags.isolated; print('isolated')"], tmp_path, log
    )
    assert code == 0 and log.read_text(encoding="utf-8").strip() == "isolated"


@pytest.mark.parametrize("damaged", [False, True])
def test_diagnosis_does_not_touch_settings_or_fetch_missing_model(
    portable, tmp_path, monkeypatch, damaged
):
    model = install_fake_model(tmp_path, monkeypatch)
    if damaged:
        model.write_bytes(b"damaged")
    settings = tmp_path / "data/desktop-settings.json"
    settings.parent.mkdir()
    settings.write_text('{"span": 1.2}', encoding="utf-8")
    commands = []

    def execute(arguments, root, log, timeout):
        commands.append(arguments)
        assert root == tmp_path and timeout == 90
        if "--workspace" in arguments:
            assert arguments[arguments.index("--workspace") + 1] != str(tmp_path)
            assert "--demo" in arguments and "--smoke-seconds" in arguments
        return 0

    monkeypatch.setattr(portable, "execute", execute)
    assert portable.main(["diagnose"], root=tmp_path) == int(damaged)
    assert settings.read_text(encoding="utf-8") == '{"span": 1.2}'
    assert any("Tracker" in " ".join(command) for command in commands) is not damaged
    report = json.loads(
        next((tmp_path / "reports/diagnostics").rglob("diagnostics.json")).read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is not damaged
    assert not report["camera_opened"] and not report["os_events_sent"]
