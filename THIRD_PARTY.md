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

Dependencies are installed from their own distributions with their accompanying licenses. A wheel/source package of this project does not embed their binaries. Review their license files when preparing a bundled executable, especially Qt/PySide6 distribution requirements.

The hand-landmarker task is fetched separately from Google's versioned model storage:

`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

Expected SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

The project does not grant a license to Google's model weights. Retain model provenance and consult the model's distribution terms when including weights in a coursework submission or other redistribution.
