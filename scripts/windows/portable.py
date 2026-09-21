"""Portable entry point; dependency failures are captured outside the GUI process."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path


def configure(root: Path) -> None:
    os.environ["PINCHPILOT_MODEL_PATH"] = str(root / "models/hand_landmarker_v1.task")
    os.environ["MPLCONFIGDIR"] = str(root / "data/matplotlib")
    # An unrelated Qt/Conda installation must not select this app's plugins.
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        os.environ.pop(key, None)
    if sys.platform == "win32":
        os.environ.pop("QT_QPA_PLATFORM", None)


def model_check(root: Path) -> None:
    from pinchpilot.vision import MODEL_SHA256

    path = root / "models/hand_landmarker_v1.task"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("Bundled model missing or damaged. Extract the complete ZIP again.")


def execute(arguments: list[str], root: Path, log: Path, timeout=None) -> int:
    with log.open("wb") as output:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-X", "utf8", *arguments],
                cwd=root,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            output.write(f"\n{type(error).__name__}: {error}\n".encode("utf-8"))
            return 1


def diagnose(root: Path, folder: Path) -> int:
    """No camera, user input injection, or changes to the normal trial settings."""
    report = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version,
        "camera_opened": False,
        "os_events_sent": False,
        "checks": [],
    }
    checks = [
        (
            "imports",
            [
                "-c",
                "import cv2, mediapipe, numpy, PySide6, sklearn; "
                "from PySide6.QtWidgets import QApplication; "
                "from pinchpilot.cli import environment_report; "
                "import json; print(json.dumps(environment_report(), indent=2))",
            ],
        ),
        (
            "model-inference",
            [
                "-c",
                "import numpy as np; from pinchpilot.vision import Tracker, fetch_model; "
                "tracker = Tracker(fetch_model()); "
                "frame = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8), 1.0); "
                "tracker.close(); print('CPU inference completed; no camera used')",
            ],
        ),
    ]
    if sys.platform == "win32":
        checks.append(
            (
                "win32-read-only",
                [
                    "-c",
                    "import ctypes; from pinchpilot.platform_io import "
                    "MouseOutput, enable_dpi_awareness; enable_dpi_awareness(); "
                    "mouse = MouseOutput(); print('Primary screen:', mouse.bounds); "
                    "print('Cursor:', mouse.position()); "
                    "assert ctypes.sizeof(mouse.win_input) == 40; "
                    "print('Win64 INPUT ABI: 40 bytes; no events sent')",
                ],
            )
        )
    try:
        model_check(root)
        report["checks"].append({"name": "model-integrity", "passed": True})
    except RuntimeError as error:
        report["checks"].append({"name": "model-integrity", "passed": False, "error": str(error)})
        # Missing models must not trigger fetch_model's network download.
        checks = [check for check in checks if check[0] != "model-inference"]
    with tempfile.TemporaryDirectory(prefix="pinchpilot-diagnostic-") as temporary:
        checks.append(
            (
                "desktop-demo",
                [
                    "-m",
                    "pinchpilot",
                    "desktop",
                    "--demo",
                    "--workspace",
                    temporary,
                    "--smoke-seconds",
                    "3",
                    "--screenshot",
                    str(folder / "demo-window.png"),
                ],
            )
        )
        for name, arguments in checks:
            print(f"Checking {name}...", flush=True)
            code = execute(arguments, root, folder / f"{name}.log", timeout=90)
            report["checks"].append({"name": name, "exit_code": code, "passed": code == 0})
    report["passed"] = all(item["passed"] for item in report["checks"])
    (folder / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Diagnostics {'passed' if report['passed'] else 'found an issue'}: {folder}")
    return 0 if report["passed"] else 1


def main(argv=None, root=None) -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=("start", "demo", "diagnose"))
    args = cli.parse_args(argv)
    root = Path(root or Path(__file__).resolve().parent).resolve()
    configure(root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = root / "reports" / ("diagnostics" if args.mode == "diagnose" else "startup")
    if args.mode == "diagnose":
        folder /= stamp
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if args.mode == "diagnose":
            return diagnose(root, folder)
        if args.mode == "start":
            model_check(root)
        log = folder / f"{stamp}.log"
        arguments = ["-m", "pinchpilot", "desktop", "--workspace", str(root)]
        if args.mode == "demo":
            arguments += ["--demo"]
        print("Starting PinchPilot. Close its window to exit.", flush=True)
        code = execute(arguments, root, log)
        if code:
            print(f"PinchPilot exited with code {code}. See {log}")
            suffix = ".cmd" if sys.platform == "win32" else ".command"
            print(f"Run Diagnose{suffix} and send its report folder with your feedback.")
        return code
    except Exception:
        details = traceback.format_exc()
        print(details)
        try:
            (folder / f"{stamp}-launcher-error.log").write_text(details, encoding="utf-8")
        except OSError:
            pass
        suffix = ".cmd" if sys.platform == "win32" else ".command"
        print(f"Extract the whole ZIP to a writable folder, then run Start{suffix} again.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
