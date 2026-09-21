# Windows portable trial package

The user requested a packaged build to send to teammates for Windows testing. Target
Windows 10/11 x64 (Intel/AMD), with the existing desktop gestures and settings.

## Delivery

- A ZIP containing private CPython, locked Windows wheels, application code and the
  verified hand-landmarker model. Extract all files, then double-click `Start.cmd`.
- No Python, uv, package installation or model download at first launch. Windows
  system prerequisites are documented separately; keep an official VC++ runtime
  installer available for machines that need it, without running it automatically.
- Separate `Demo.cmd` and `Diagnose.cmd` entries. Demo never opens a camera or sends
  mouse input. Diagnostics test imports, bundled-model inference and a temporary
  demo window, writing a local report. Normal launch logs failures locally.
- Chinese quick start, gesture reference, short feedback template and third-party
  provenance/licenses. Ship no developer recordings, settings or machine logs.

## Approach

Use the official Python embeddable distribution with its isolated module search
path and all locked Windows dependency wheels. A source ZIP still needs developer
setup; a frozen executable requires an actual Windows build environment. The
portable directory can be assembled on this Mac using prebuilt Windows binaries.
Keep it reproducible with a build script, pinned Python/model checksums, locked
dependency hashes, an inventory and an archive checksum.

Portable launch chooses the bundled model through an explicit application-specific
environment variable. Normal Mac/source launches keep the existing model cache.
Existing camera-off / mouse-off startup and Esc emergency stop remain unchanged.
Package-specific changes are version 0.8.1.

## Validation and boundaries

Run existing tests and meaningful checks for bundled-model resolution, bootstrap
failure logging and diagnostic isolation. Verify ZIP contents, Win64 binary headers,
dependency closure, relative paths and absence of personal data. Package includes
an on-device diagnostic because macOS checks cannot establish Windows DLL loading,
camera compatibility or native mouse behavior. Do not claim Windows execution or
hardware acceptance until a teammate supplies those results.
