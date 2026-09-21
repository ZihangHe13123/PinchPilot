"""Verify and execute an extracted Mac archive without camera or mouse input."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("archive", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = {"camera_opened": False, "os_events_sent": False}
    with tempfile.TemporaryDirectory(prefix="pinchpilot-中文 归档-") as temporary:
        subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), temporary], check=True)
        package = (Path(temporary) / archive.stem).resolve()
        info = json.loads((package / "BUILD_INFO.json").read_text())
        checks["version"] = info["version"]
        checks["source_commit"] = info["source_commit"]
        checks["source_dirty"] = info["source_dirty"]
        for name, target in json.loads((package / "SYMLINKS.json").read_text()).items():
            path = package / name
            assert path.is_symlink() and os.readlink(path) == target
            assert path.resolve().is_relative_to(package) and path.exists()
        checks["symlinks_restored"] = True
        files = 0
        for line in (package / "FILES.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = package / name
            with path.open("rb") as file:
                assert hashlib.file_digest(file, "sha256").hexdigest() == expected, name
            files += 1
        checks["file_hashes_verified"] = files
        for name in (
            "Start.command",
            "Demo.command",
            "Diagnose.command",
            "launch.sh",
            "runtime/bin/python3.11",
        ):
            assert os.access(package / name, os.X_OK), name
        checks["executable_modes_restored"] = True
        assert not (package / "data").exists() and not (package / "reports").exists()
        environment = os.environ | {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONHOME": "/nonexistent-pinchpilot-host-python",
            "PYTHONPATH": "/nonexistent-pinchpilot-host-packages",
            "VIRTUAL_ENV": "/nonexistent-pinchpilot-venv",
            "QT_QPA_PLATFORM": "offscreen",
        }
        python = package / "runtime/bin/python3.11"
        probe = (
            "import sys, json, pathlib, cv2, mediapipe, numpy, scipy, PySide6, Quartz, "
            "ApplicationServices, AVFoundation, pinchpilot; "
            "root=pathlib.Path(sys.executable).resolve().parents[2]; "
            "modules=[cv2,mediapipe,numpy,scipy,PySide6,Quartz,ApplicationServices,AVFoundation,pinchpilot]; "
            "locations={m.__name__:m.__file__ for m in modules}; "
            "assert all(pathlib.Path(p).resolve().is_relative_to(root) for p in locations.values()); "
            "assert pathlib.Path(sys.prefix).resolve()==root/'runtime'; "
            "assert sys.flags.isolated; "
            "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'imports':locations}))"
        )
        imported = subprocess.run(
            [str(python), "-I", "-B", "-X", "utf8", "-c", probe],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        (output / "imports.log").write_text(imported.stdout + imported.stderr)
        assert imported.returncode == 0, imported.stderr
        checks["isolated_imports"] = json.loads(imported.stdout)
        diagnostic = subprocess.run(
            [str(package / "Diagnose.command")],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=150,
        )
        (output / "launcher.log").write_text(diagnostic.stdout + diagnostic.stderr)
        reports = package / "reports/diagnostics"
        if reports.exists():
            shutil.copytree(reports, output / "diagnostics", dirs_exist_ok=True)
        assert diagnostic.returncode == 0, diagnostic.stdout + diagnostic.stderr
        result = json.loads(next(reports.rglob("diagnostics.json")).read_text())
        assert result["passed"] and not result["camera_opened"] and not result["os_events_sent"]
        assert not (package / "data/desktop-settings.json").exists()
        checks["diagnostics"] = result
        # Verify the normal desktop entry starts with the camera and output off too.
        environment["PINCHPILOT_MODEL_PATH"] = str(package / "models/hand_landmarker_v1.task")
        idle = subprocess.run(
            [
                str(python),
                "-I",
                "-B",
                "-X",
                "utf8",
                "-m",
                "pinchpilot",
                "desktop",
                "--workspace",
                str(Path(temporary) / "idle-workspace"),
                "--smoke-seconds",
                "2",
                "--screenshot",
                str(output / "idle-window.png"),
            ],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        (output / "idle.log").write_text(idle.stdout + idle.stderr)
        assert idle.returncode == 0, idle.stderr
        checks["idle_desktop_exit"] = idle.returncode
    with archive.open("rb") as file:
        checks["archive_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    checks["passed"] = True
    (output / "execution-checks.json").write_text(json.dumps(checks, indent=2))
    print(
        json.dumps(
            {
                "passed": True,
                "version": checks["version"],
                "files": files,
                "camera_opened": False,
                "os_events_sent": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
