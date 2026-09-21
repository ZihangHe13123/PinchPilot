# References and third-party components

PinchPilot's Python application code is independently implemented for this coursework prototype. No Pawvis source code, branding, UI assets, or repository history is copied into this repository. This note records conceptual and technical references; it does not assert that pinch interaction, calibration or smoothing is novel.

| Reference | Use in this project |
|---|---|
| [Pawvis](https://github.com/alexandriax/pawvis/tree/039e6d8ede814cb943532b841e6c78f83fba1f62) | Prior product and architecture reference. Existing calibration, DTW templates and stabilization must be acknowledged. |
| [Meta hand interactions](https://developers.meta.com/horizon/documentation/unity/unity-handtracking-interactions/) | Pointer/pinch interaction concept; not reused SDK code. |
| [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) | Pretrained hand detection/landmarks. The model is not trained by this project. |
| [One Euro Filter](https://gery.casiez.net/1euro/) | Speed-adaptive low-pass filtering approach; cite Casiez, Roussel and Vogel (2012) in the report. |
| [Qt for Python](https://doc.qt.io/qtforpython-6/gettingstarted.html) | PySide6 desktop widgets. |
| [PyObjC ApplicationServices](https://pyobjc.readthedocs.io/en/latest/apinotes/ApplicationServices.html) | macOS permissions and native framework access. |
| [Win32 SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) | Windows mouse events, platform limitations and ABI. |
| [Win32 MOUSEINPUT](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput) | Left/right button transition flags used by the native adapter; checked against official documentation. |
| [Core Graphics CGEventType](https://developer.apple.com/documentation/coregraphics/cgeventtype) | Native mouse button and dragged-event types. |

The project's wheel/source package does not embed dependency binaries. The Windows
and Mac portable ZIPs additionally include upstream Python and dependency binaries,
their original license/notice files, and the verified hand model. `BUILD_INFO.json`
lists versions, provenance, hashes and license locations; `FILES.sha256` inventories
the distribution. Application Python source and replaceable dependency directories
remain available in `runtime/Lib/site-packages`.

Python's license is in `runtime/LICENSE.txt` (Windows) or
`runtime/lib/python3.11/LICENSE.txt` (Mac); each dependency's licenses remain in
its `.dist-info/licenses` or package directory. Qt/PySide6 source and distribution
information: [Qt for Python](https://code.qt.io/cgit/pyside/pyside-setup.git/),
[Qt source archives](https://download.qt.io/archive/qt/),
[Qt for Python licensing](https://doc.qt.io/qtforpython-6/licenses.html).
The optional Microsoft VC++ runtime installer is copied unchanged from its official
download endpoint and is only run if the tester chooses it. Its own installer
presents Microsoft's terms.

The Mac archive uses a pinned release of
[python-build-standalone](https://github.com/astral-sh/python-build-standalone),
the same Python runtime family used by uv. Its exact release asset and checksum are
recorded in `BUILD_INFO.json`. The archive retains upstream license files and uses
independently replaceable Python packages rather than a frozen executable.

Source launches fetch the hand-landmarker task separately; portable launches
use the bundled identical file. Its source is Google's versioned model storage:

`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

Expected SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

Google's [Hand Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
links this model family to the [hand tracking model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf),
which states Apache License 2.0. The Apache text is retained with MediaPipe's license
files. PinchPilot does not claim authorship of the pretrained model or change its
license. These bundled components are separate from the project's own coursework code.
