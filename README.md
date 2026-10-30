# PinchPilot

[中文说明](README.zh-CN.md)

PinchPilot turns an ordinary webcam into a mouse controlled by small finger movements. The forearm rests on the desk; a pinch of thumb and middle finger moves the pointer, and light taps of the index or ring finger on the thumb click, drag and right-click. It was built as the course project of Group 15 (Wei Yi, He Zihang, Chi Yongjun) in the NUS-ISS Pattern Recognition Systems practice module.

The pattern recognition problem: MediaPipe gives 21 hand keypoints in every frame, but not whether two fingertips are touching. PinchPilot decides, every frame, whether the middle, index and ring finger touch the thumb.

## Final version

**v0.20.0, submitted on 30 October 2026.** The three of us recorded twelve guided sessions (about 2.5 hours, 205,141 windows with reviewed labels; only hand keypoints are stored, never images) and compared four contact recognisers under one leave-one-person-out protocol. All settings were frozen on 13 October and tested once; a blinded pilot then chose the default model.

| Contact recogniser | Macro-F1 on held-out people |
|---|---|
| Tuned rule | 0.885 |
| Temporal CNN (default) | 0.863 |
| Finger-shared CNN | 0.860 |
| Random forest | 0.840 |
| Original fixed thresholds (baseline) | 0.771 |

Side-on hands are the hardest case: the fixed thresholds fall to 0.42 there, while the four trained recognisers stay between 0.68 and 0.74. In the blinded [pilot](docs/PILOT.md) all three of us ranked the temporal CNN first, with the lowest workload (NASA-TLX 26.4, against 29.2 for the tuned rule and 54.2 for the fixed thresholds), so it is the default; the tuned rule and the fixed thresholds remain selectable. Every number comes from the project's own result files; the dated records are in [docs/VALIDATION.md](docs/VALIDATION.md).

## Download and run

Portable archives are on the [v0.20.0 release](https://github.com/ZihangHe13123/PinchPilot/releases/tag/v0.20.0). They carry their own Python runtime, dependencies, hand model and both built-in contact models; no installation is needed.

- **macOS** (Apple silicon, macOS 14 or later): unzip `PinchPilot-0.20.0-macOS-arm64.zip`, keep the whole folder, and double-click `Start-English.command` (`Start.command` opens the Chinese interface). If macOS blocks it, right-click the file, choose Open and confirm.
- **Windows 10/11** (64-bit): extract `PinchPilot-0.20.0-Windows-x64.zip` completely, for example to `D:\PinchPilot`, and double-click `Start-English.cmd` (`Start.cmd` for Chinese). The Windows archive passed its build-time checks but has not been run on Windows hardware; the source version ran on two members' Windows laptops throughout data collection.
- **No camera:** `Demo.command` / `Demo.cmd` opens a synthetic demonstration.

From the source code, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
uv sync --locked
uv run pinchpilot desktop --lang en
```

`--lang en` (or the environment variable `PINCHPILOT_LANG=en`) shows every window in English; the default is Chinese. No GPU, account or API key is needed.

## Use

**Start camera → Enable mouse control → move the index away from the thumb and pinch thumb and middle finger to take the pointer.** The pointer starts from where the system cursor is. Rest the forearm on the desk where the camera sees the hand and fingertips.

| Action | Gesture |
|---|---|
| Move | Hold thumb and middle finger together and move the hand or wrist slightly |
| Click | A quick thumb–index tap; the pointer freezes while the index approaches, so the click lands where you aimed |
| Double click | Two taps at the same place, releasing the index in between |
| Right click | Index away, a light thumb–ring tap |
| Drag | Hold thumb and index for about 0.3 s, move, release the index to drop |
| Scroll | A V with index and middle finger, then move slightly up or down; fold the fingers to stop |
| Reposition | Release the middle finger: the pointer stays put; grip again to continue from there |
| Stop | Esc anywhere, or Stop control |

**Mini view:** F9, or the Mini view button, keeps only the camera preview and the current state in a small window on top of other windows; F9 again or a double click on the preview brings the full window back.

The whole camera image is used for detection; the dashed box in the preview is only the pointer's mapping area. If the hand often leaves the image, tilt the laptop screen. The frame edge warning tells you when the hand is near the edge.

**Settings** (Show feel settings): pointer speed 0.10× to 2.00×, jitter filter, drag delay, scroll speed, control hand (automatic lock, right only or left only), positioning mode, and the **contact model**: Temporal CNN · built-in (default), Tuned rule · built-in, Original rule · fixed distance thresholds, or a model you exported into the workspace's `data/contact_models/`. Changing a setting stops control; enable it again afterwards.

**Testbench:** Open the testbench runs six tasks on a virtual cursor (small-target click, double click, drag and drop, fine correction, right click, scroll and stop) while the system mouse stays paused.

## How it works

Camera frames go to MediaPipe Hand Landmarker; PinchPilot locks onto one control hand. Two paths run on every frame:

- the **pointer path** turns the thumb–middle grip into smooth pointer motion (filtering, clutch, freeze before a click);
- the **contact path** decides touches. A contact model looks at the last 8 frames (about 233 ms at 30 fps, never future frames; 67 features per frame: palm-normalised keypoints, three fingertip distance ratios, the frame interval) and decides only whether a finger touches. Releasing a button still follows the measured distance, so a model error can cost one click but never leave a button held.

Models are trained offline and exported as a folder (`model.json` plus `weights.npz`) that runs with NumPy in about 0.1 ms per frame; the application does not need PyTorch.

## Contact models: data, comparison and export

Run from the source code on a folder of guided recordings:

```sh
uv run pinchpilot ml-pool <recordings>
uv run --extra ml pinchpilot ml-cv <recordings> --settings experiments/contact/cnn.json
uv run --extra ml pinchpilot ml-cv <recordings> --settings experiments/contact/cnn.json --final-test
uv run --extra ml pinchpilot ml-export <recordings> --settings experiments/contact/cnn.json --candidate "旋转＋缩放＋噪声" --epochs 22 --output <folder>
```

Use `rules.json`, `forest.json` or `finger_cnn.json` for the other three recognisers; the four settings files are as frozen on 13 October 2026, and the candidate names inside them are in Chinese ("旋转＋缩放＋噪声" is rotation + scale + noise). Every report records the data fingerprint, the code version and the settings. Validation is nested leave-one-person-out: each person is held out in turn, and the other two choose the settings, one training while the other validates, over three seeds.

The recordings, reviewed labels and cross-checks are kept in the team's private data repository (a copy went with the course submission). The tools that made them are in the application: guided recording with an animated demonstration hand and a live palm-angle check, label review and cross-checking (`pinchpilot ml-review`).

## Tech stack

| Layer | Implementation | Purpose |
|---|---|---|
| Interface | Python 3.11, PySide6 | Desktop window, mini view, testbench, recording and review windows |
| Perception | OpenCV, MediaPipe Hand Landmarker | Camera image to 21 hand keypoints |
| Contact models | PyTorch (training, `--extra ml`), scikit-learn, NumPy | Tuned rule, random forest, temporal CNN, finger-shared CNN; NumPy inference at run time |
| Interaction | Python state machine | Grip, contacts, press and release, drag, scroll, safety pauses |
| System input | macOS Quartz, Windows Win32 | Pointer, buttons, scrolling, global Esc |
| Quality | pytest (818 tests), Ruff, GitHub Actions on macOS 15 and Windows Server 2022 | Every pull request |

## Permissions and common problems

- **macOS camera:** allow the launching app under System Settings → Privacy & Security → Camera, then start again.
- **macOS pointer control:** allow it under Accessibility; the global Esc key also needs Input Monitoring.
- **Windows camera:** allow desktop apps to use the camera under Settings → Privacy and security → Camera; close other programs that use it.
- **Windows input:** input injection may not reach windows running as administrator; PinchPilot does not elevate itself.
- **Jittery pointer or false pinches:** improve the lighting, keep the fingertips visible, lower the pointer speed or use the stronger jitter filter.

PinchPilot stores no video or images. Keypoint recordings, local logs and reports live in `data/` and `reports/` of the workspace, which are not part of the repository.

## Development

`main` accepts changes only through pull requests, which must pass the checks on macOS and Windows:

```sh
uv run ruff check src tests
uv run pytest -q
uv run pinchpilot desktop --demo --smoke-seconds 3
```

Portable archives are built with `uv run python scripts/build_macos_portable.py` and `scripts/build_windows_portable.py`.

## AI assistance

Anthropic Claude (Claude Code) and OpenAI Codex assisted with the implementation, tests, experiment tooling, documentation and figures. The team designed the gestures, the recording protocol and the pilot, recorded and reviewed all data, cross-checked each other's labels, took part in the pilot and reviewed the results. Generated code was accepted only after review and the automated tests on macOS and Windows. Every number in this README and in the validation record comes from a result file produced by the project's own tools.

## Documentation

Most documents are in Chinese; the [Chinese README](README.zh-CN.md) also covers the earlier interaction modes kept for comparison (palm-pointing baseline, single-finger demos, small-motion range).

- [Validation record](docs/VALIDATION.md): dated results of every test, study and release
- [Pilot](docs/PILOT.md): the blinded comparison of the three contact settings
- [Recording protocol](docs/RECORDING_PROTOCOL.md), [label review](docs/LABEL_REVIEW.md), [model comparison](docs/MODEL_COMPARISON.md)
- [Desktop guide](docs/DESKTOP_TRIAL.md), [Windows guide](docs/WINDOWS_TRIAL.md), [macOS archive](docs/MACOS_ARCHIVE.md)
- [Diagnostics and testbench](docs/EVALUATION_TOOLS.md)
- [Course proposal and first presentation](docs/course/README.md)
- [Third-party components](THIRD_PARTY.md)
