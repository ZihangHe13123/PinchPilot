"""Build the Apple Silicon archive from a pinned runtime and locked wheels."""

import csv
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

from build_windows_portable import audit_source_archive, digest, download
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags, mac_platforms, parse_tag
from packaging.utils import canonicalize_name

from pinchpilot.vision import MODEL_SHA256, MODEL_URL, fetch_model

ROOT = Path(__file__).resolve().parents[1]
PYTHON_NAME = "cpython-3.11.15+20260623-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260623/"
    "cpython-3.11.15%2B20260623-aarch64-apple-darwin-install_only.tar.gz"
)
PYTHON_SHA256 = "d2324bfd1a7b9fc44ccd884c3a2505bcab6691dbfd4f8270e10c50aaa4e19506"


def run(*arguments, **kwargs):
    return subprocess.run(arguments, cwd=ROOT, check=True, **kwargs)


def inventory(site):
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    versions = {canonicalize_name(d.metadata["Name"]): d.version for d in distributions}
    platforms = list(mac_platforms((14, 0), "arm64"))
    supported = set(cpython_tags((3, 11), ["cp311"], platforms))
    supported.update(compatible_tags((3, 11), "cp311", platforms))
    environment = default_environment() | {"extra": "", "python_full_version": "3.11.15"}
    result = []
    for dist in sorted(distributions, key=lambda d: d.metadata["Name"].lower()):
        tags = [
            line[5:] for line in dist.read_text("WHEEL").splitlines() if line.startswith("Tag: ")
        ]
        if not any(parse_tag(tag) & supported for tag in tags):
            raise ValueError(f"Wheel incompatible with macOS 14 arm64: {dist.metadata['Name']}")
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            installed = versions.get(canonicalize_name(requirement.name))
            if installed is None or installed not in requirement.specifier:
                raise ValueError(f"Missing dependency for {dist.metadata['Name']}: {text}")
        result.append({"name": dist.metadata["Name"], "version": dist.version, "wheel_tags": tags})
    return result


def macho_architectures(path):
    with path.open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            return set()
        magic = header[:4]
        if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
            return {struct.unpack("<I", header[4:])[0]}
        if magic in (b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"):
            return {struct.unpack(">I", header[4:])[0]}
        if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
            count = struct.unpack(">I", header[4:])[0]
            if count > 16:
                return set()  # A Java class can share the FAT_MAGIC prefix.
            stride = 32 if magic[-1] == 0xBF else 20
            return {struct.unpack(">I", file.read(stride)[:4])[0] for _ in range(count)}
        return set()


def audit(package, site):
    required = (
        "Start.command",
        "Demo.command",
        "Diagnose.command",
        "launch.sh",
        "portable.py",
        "runtime/bin/python3.11",
        "runtime/lib/python3.11/LICENSE.txt",
        "runtime/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms/libqcocoa.dylib",
        "runtime/lib/python3.11/site-packages/pinchpilot/compatible.py",
        "models/hand_landmarker_v1.task",
        "docs/COMPATIBLE_TRIAL.md",
        "README.txt",
    )
    for name in required:
        if not (package / name).is_file():
            raise ValueError(f"Missing package file: {name}")
    if (package / "data").exists() or (package / "reports").exists():
        raise ValueError("Staging directory contains trial settings or logs")
    links = {}
    binaries = 0
    for path in package.rglob("*"):
        relative = path.relative_to(package).as_posix()
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_relative_to(package.resolve()):
                raise ValueError(f"Broken/external symlink: {relative}")
            links[relative] = os.readlink(path)
        elif path.is_file():
            if path.name in (".DS_Store", "direct_url.json") or "__pycache__" in path.parts:
                raise ValueError(f"Unexpected build metadata: {relative}")
            architectures = macho_architectures(path)
            if architectures:
                if 0x0100000C not in architectures:
                    raise ValueError(f"Native binary lacks arm64: {relative}")
                binaries += 1
    if digest(package / "models/hand_landmarker_v1.task") != MODEL_SHA256:
        raise ValueError("Invalid bundled model")
    return {"distributions": inventory(site), "arm64_macho_files": binaries}, links


def archive(package, destination):
    temporary = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package.rglob("*")):
            name = f"{package.name}/{path.relative_to(package).as_posix()}"
            if path.is_symlink():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(info, os.readlink(path))
            elif path.is_file():
                bundle.write(path, name)
    with zipfile.ZipFile(temporary) as bundle:
        if damaged := bundle.testzip():
            raise ValueError(f"Archive CRC failure: {damaged}")
    temporary.replace(destination)
    checksum = digest(destination)
    destination.with_suffix(".zip.sha256").write_text(f"{checksum}  {destination.name}\n")
    return {"name": destination.name, "bytes": destination.stat().st_size, "sha256": checksum}


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("Build with uv's Python 3.11 on an Apple Silicon Mac")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("Commit the release sources before building the portable archive")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    name = f"PinchPilot-{version}-macOS-arm64"
    package = ROOT / "build/macos-arm64" / name
    runtime = package / "runtime"
    site = runtime / "lib/python3.11/site-packages"
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("The build machine needs uv; archive users do not")
    if (package / "data").exists() or (package / "reports").exists():
        raise ValueError("Refusing to package a used staging directory")
    package.mkdir(parents=True, exist_ok=True)
    source = download(PYTHON_URL, PYTHON_SHA256, ROOT / "build/macos-downloads" / PYTHON_NAME)
    with tempfile.TemporaryDirectory(dir=package.parent) as temporary:
        with tarfile.open(source) as bundle:
            bundle.extractall(temporary, filter="data")
        if runtime.exists():
            shutil.rmtree(runtime)
        shutil.move(str(Path(temporary) / "python"), runtime)
    # Keep only the runtime; package management runs on the build machine.
    for path in (runtime / "bin").iterdir():
        if path.name not in ("python", "python3", "python3.11"):
            path.unlink()
    install = [
        uv,
        "pip",
        "install",
        "--python",
        str(runtime / "bin/python3.11"),
        "--python-version",
        "3.11.15",
        "--python-platform",
        "aarch64-apple-darwin",
        "--target",
        str(site),
        "--only-binary",
        ":all:",
        "--link-mode",
        "copy",
        "--no-progress",
    ]
    environment = os.environ | {"MACOSX_DEPLOYMENT_TARGET": "14.0"}
    run(*install, "--exact", "--require-hashes", "-r", "requirements.lock", env=environment)
    run(uv, "build", "--wheel")
    run(
        *install,
        "--no-deps",
        str(ROOT / "dist" / f"pinchpilot-{version}-py3-none-any.whl"),
        env=environment,
    )
    shutil.rmtree(site / "bin", ignore_errors=True)
    for path in list(runtime.rglob("__pycache__")):
        shutil.rmtree(path)
    for path in list(site.glob("*.whl")) + list(site.glob("*.dist-info/direct_url.json")):
        path.unlink()
    (site / ".lock").unlink(missing_ok=True)
    for record in site.glob("*.dist-info/RECORD"):
        with record.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.reader(file) if (site / row[0]).exists()]
        with record.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)
    for path in (ROOT / "scripts/macos").iterdir():
        target = package / path.name
        shutil.copy2(path, target)
        target.chmod(0o755)
    shutil.copy2(ROOT / "scripts/windows/portable.py", package / "portable.py")
    for directory in ("models", "docs", "source"):
        (package / directory).mkdir(exist_ok=True)
    shutil.copy2(fetch_model(), package / "models/hand_landmarker_v1.task")
    for src, target in (
        ("docs/MACOS_ARCHIVE.md", "README.txt"),
        ("THIRD_PARTY.md", "docs/THIRD_PARTY.md"),
        ("requirements.lock", "docs/requirements.lock"),
    ):
        shutil.copy2(ROOT / src, package / target)
    # Some user guides link to design notes; retain the complete Markdown tree.
    for path in (ROOT / "docs").rglob("*.md"):
        target = package / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    run(
        "git",
        "archive",
        "--format=zip",
        "--prefix=PinchPilot/",
        "-o",
        str(package / "source" / f"PinchPilot-{version}-source.zip"),
        "HEAD",
    )
    result, links = audit(package, site)
    result["source_archive"] = audit_source_archive(package, site, version)
    result.update(
        {
            "version": version,
            "target": "macOS >=14, Apple Silicon arm64, CPython 3.11.15",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
            "sources": {
                "python": {"url": PYTHON_URL, "sha256": PYTHON_SHA256},
                "model": {"url": MODEL_URL, "sha256": MODEL_SHA256},
                "dependency_lock_sha256": digest(ROOT / "requirements.lock"),
            },
            "camera_opened": False,
            "os_events_sent": False,
            "licenses": [
                path.relative_to(package).as_posix()
                for path in sorted(package.rglob("*"))
                if path.is_file()
                and not path.is_symlink()
                and any(word in path.name.lower() for word in ("license", "copying", "notice"))
            ],
        }
    )
    (package / "BUILD_INFO.json").write_text(json.dumps(result, indent=2) + "\n")
    (package / "SYMLINKS.json").write_text(json.dumps(links, indent=2) + "\n")
    hashes = [
        f"{digest(p)}  {p.relative_to(package).as_posix()}"
        for p in sorted(package.rglob("*"))
        if p.is_file() and not p.is_symlink() and p.name != "FILES.sha256"
    ]
    (package / "FILES.sha256").write_text("\n".join(hashes) + "\n")
    print(f"Archiving {len(hashes)} files and {len(links)} symbolic links...", flush=True)
    result["archive"] = archive(package, ROOT / "dist" / f"{name}.zip")
    report = ROOT / "reports/verification" / f"macos-{version}"
    report.mkdir(parents=True, exist_ok=True)
    (report / "package-checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["archive"], indent=2))


if __name__ == "__main__":
    main()
