"""Assemble a Windows x64 portable ZIP from verified, prebuilt distributions.

Run with `uv run python scripts/build_windows_portable.py` on macOS or Windows.
This is packaging, not cross-compilation or proof of Windows execution.
"""

import csv
import hashlib
import importlib.metadata
import json
import shutil
import struct
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags, parse_tag
from packaging.utils import canonicalize_name

from pinchpilot.vision import MODEL_SHA256, MODEL_URL, fetch_model

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12.10"
PYTHON_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
PYTHON_SHA256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"
VC_URL = (
    "https://download.visualstudio.microsoft.com/download/pr/"
    "ebdab8e5-1d7b-4d9f-a11b-cbb1720c3b12/"
    "843068991DAAA1F73AD9F6239BCE4D0F6A07A51F18C37EA2A867E9BECA71295C/VC_redist.x64.exe"
)
VC_SHA256 = "843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c"
TARGET_ENV = {
    "implementation_name": "cpython",
    "implementation_version": PYTHON_VERSION,
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_python_implementation": "CPython",
    "platform_release": "10",
    "platform_system": "Windows",
    "platform_version": "10.0",
    "python_full_version": PYTHON_VERSION,
    "python_version": "3.12",
    "sys_platform": "win32",
    "extra": "",
}


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(url: str, checksum: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == checksum:
        return path
    partial = path.with_suffix(path.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        if digest(partial) != checksum:
            raise ValueError(f"Checksum mismatch: {url}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def run(*arguments: str) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def wheel_inventory(site: Path) -> list[dict]:
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    versions = {canonicalize_name(d.metadata["Name"]): d.version for d in distributions}
    supported = set(cpython_tags((3, 12), ["cp312"], ["win_amd64"]))
    supported.update(compatible_tags((3, 12), "cp312", ["win_amd64"]))
    inventory = []
    for dist in sorted(distributions, key=lambda d: d.metadata["Name"].lower()):
        tags = [
            line[5:] for line in dist.read_text("WHEEL").splitlines() if line.startswith("Tag: ")
        ]
        if not any(parse_tag(tag) & supported for tag in tags):
            raise ValueError(f"Non-Windows/Python 3.12 wheel: {dist.metadata['Name']} {tags}")
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(TARGET_ENV):
                continue
            installed = versions.get(canonicalize_name(requirement.name))
            if installed is None or installed not in requirement.specifier:
                raise ValueError(f"Incomplete dependencies for {dist.metadata['Name']}: {text}")
        inventory.append(
            {"name": dist.metadata["Name"], "version": dist.version, "wheel_tags": tags}
        )
    return inventory


def pe_machine(path: Path) -> int:
    with path.open("rb") as file:
        if file.read(2) != b"MZ":
            raise ValueError(f"Not a Windows binary: {path}")
        file.seek(0x3C)
        location = struct.unpack("<I", file.read(4))[0]
        file.seek(location)
        if file.read(4) != b"PE\0\0":
            raise ValueError(f"Invalid PE header: {path}")
        return struct.unpack("<H", file.read(2))[0]


def audit(package: Path, site: Path) -> dict:
    required = [
        "Start.cmd",
        "Demo.cmd",
        "Diagnose.cmd",
        "portable.py",
        "README.txt",
        "runtime/python.exe",
        "runtime/python312.dll",
        "runtime/python312._pth",
        "runtime/Lib/site-packages/PySide6/QtWidgets.pyd",
        "runtime/Lib/site-packages/PySide6/plugins/platforms/qwindows.dll",
        "runtime/Lib/site-packages/pinchpilot/desktop.py",
        "models/hand_landmarker_v1.task",
        "prerequisites/VC_redist.x64.exe",
    ]
    for name in required:
        if not (package / name).is_file():
            raise ValueError(f"Missing package file: {name}")
    binaries = {"x64": 0, "other": []}
    for path in (package / "runtime").rglob("*"):
        if path.suffix.lower() not in (".exe", ".dll", ".pyd"):
            continue
        machine = pe_machine(path)
        if machine == 0x8664:
            binaries["x64"] += 1
        else:
            relative = path.relative_to(package).as_posix()
            # The upstream sounddevice wheel includes all three Windows variants.
            alternative = (machine == 0x14C and "32bit" in path.name) or (
                machine == 0xAA64 and "arm64" in path.name
            )
            if "_sounddevice_data/portaudio-binaries/" not in relative or not alternative:
                raise ValueError(f"Unexpected binary architecture {machine:#x}: {relative}")
            binaries["other"].append(relative)
    for path in package.rglob("*"):
        relative = path.relative_to(package)
        if any(part in (".venv", ".git", "__pycache__", ".DS_Store") for part in relative.parts):
            raise ValueError(f"Unexpected local build data: {relative}")
    if digest(package / "models/hand_landmarker_v1.task") != MODEL_SHA256:
        raise ValueError("Bundled model was changed")
    if (package / "data").exists() or (package / "reports").exists():
        raise ValueError("Do not redistribute a used package with tester data/settings")
    return {"distributions": wheel_inventory(site), "binaries": binaries, "contents_checked": True}


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    name = f"PinchPilot-{version}-Windows-x64"
    package = ROOT / "build/windows-x64" / name
    runtime = package / "runtime"
    site = runtime / "Lib/site-packages"
    cache = ROOT / "build/windows-downloads"
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("The build machine needs uv; recipients do not.")
    if (package / "data").exists() or (package / "reports").exists():
        raise ValueError("Build staging folder contains user data; choose a clean build folder")
    runtime.mkdir(parents=True, exist_ok=True)
    archive = download(PYTHON_URL, PYTHON_SHA256, cache / "python-3.12.10-embed-amd64.zip")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(runtime)
    (runtime / "python312._pth").write_text(
        "python312.zip\n.\nLib/site-packages\n..\nimport site\n", encoding="ascii"
    )
    install = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--python-version",
        PYTHON_VERSION,
        "--python-platform",
        "x86_64-pc-windows-msvc",
        "--target",
        str(site),
        "--only-binary",
        ":all:",
        "--link-mode",
        "copy",
        "--no-progress",
    ]
    run(*install, "--exact", "--require-hashes", "-r", "requirements.lock")
    run(uv, "build", "--wheel")
    wheel = ROOT / "dist" / f"pinchpilot-{version}-py3-none-any.whl"
    run(*install, "--no-deps", "--reinstall", str(wheel))
    # Cross-target uv generates host shebangs for unused dependency command-line tools.
    # Keep all libraries/licenses, but remove these machine-specific entry scripts.
    shutil.rmtree(site / "bin", ignore_errors=True)
    for path in list(site.rglob("__pycache__")):
        shutil.rmtree(path)
    for path in list(site.glob("*.whl")) + list(site.glob("*.dist-info/direct_url.json")):
        path.unlink()
    (site / ".lock").unlink(missing_ok=True)
    for record in site.glob("*.dist-info/RECORD"):
        with record.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.reader(file) if (site / row[0]).exists()]
        with record.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)
    for path in (ROOT / "scripts/windows").iterdir():
        if path.suffix == ".cmd":
            (package / path.name).write_bytes(
                path.read_text().replace("\n", "\r\n").encode("ascii")
            )
        elif path.name == "portable.py":
            shutil.copy2(path, package / path.name)
    for directory in ("models", "docs", "prerequisites"):
        (package / directory).mkdir(exist_ok=True)
    shutil.copy2(fetch_model(), package / "models/hand_landmarker_v1.task")
    shutil.copy2(
        download(VC_URL, VC_SHA256, cache / "vc_redist.x64.exe"),
        package / "prerequisites/VC_redist.x64.exe",
    )
    # A UTF-8 BOM lets older Windows Notepad display the Chinese guide correctly.
    for source, destination in (
        ("docs/WINDOWS_TRIAL.md", "README.txt"),
        ("docs/WINDOWS_FEEDBACK.md", "FEEDBACK.txt"),
        ("docs/DESKTOP_TRIAL.md", "docs/DESKTOP_TRIAL.md"),
        ("docs/WINDOWS_TRIAL.md", "docs/WINDOWS_TRIAL.md"),
        ("docs/VALIDATION.md", "docs/VALIDATION.md"),
        ("THIRD_PARTY.md", "docs/THIRD_PARTY.md"),
    ):
        (package / destination).write_text((ROOT / source).read_text(), encoding="utf-8-sig")
    shutil.copy2(ROOT / "requirements.lock", package / "docs/requirements.lock")
    result = audit(package, site)
    result.update(
        {
            "version": version,
            "target": "Windows 10/11 x64; CPython 3.12.10",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
            "windows_execution_tested_here": False,
            "camera_opened": False,
            "os_events_sent": False,
            "sources": {
                "python": {"url": PYTHON_URL, "sha256": PYTHON_SHA256},
                "vc_runtime_installer": {"url": VC_URL, "sha256": VC_SHA256},
                "model": {"url": MODEL_URL, "sha256": MODEL_SHA256},
                "dependency_lock_sha256": digest(ROOT / "requirements.lock"),
            },
            "licenses": [
                path.relative_to(package).as_posix()
                for path in sorted(package.rglob("*"))
                if path.is_file()
                and any(word in path.name.lower() for word in ("license", "copying", "notice"))
            ],
        }
    )
    (package / "BUILD_INFO.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Every shipped file is independently verifiable after transfer/extraction.
    hashes = [
        f"{digest(path)}  {path.relative_to(package).as_posix()}"
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "FILES.sha256"
    ]
    (package / "FILES.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    output = ROOT / "dist" / f"{name}.zip"
    temporary = output.with_suffix(".zip.partial")
    print(f"Compressing {len(hashes)} files...", flush=True)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, f"{name}/{path.relative_to(package).as_posix()}")
    with zipfile.ZipFile(temporary) as bundle:
        damaged = bundle.testzip()
        if damaged:
            raise ValueError(f"Archive CRC failure: {damaged}")
    temporary.replace(output)
    checksum = digest(output)
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    result["archive"] = {"name": output.name, "bytes": output.stat().st_size, "sha256": checksum}
    report = ROOT / "reports/verification" / f"windows-{version}"
    report.mkdir(parents=True, exist_ok=True)
    (report / "package-checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["archive"], indent=2))


if __name__ == "__main__":
    main()
