"""The everyday trial window. Camera and system mouse always start switched off."""

import json
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .app import STYLE, Job
from .desktop_control import DesktopController
from .platform_io import camera_permission, enable_dpi_awareness, request_camera_access
from .tripod_controls import TripodControls
from .tripod_demo import tripod_demo_frame
from .vision import CameraWorker, Packet, default_model_path, fetch_model
from .widgets import CameraView


class DesktopWindow(QMainWindow):
    camera_authorized = Signal(int, bool)

    def __init__(self, workspace: Path, demo=False, controller=None):
        super().__init__()
        self.workspace = workspace.resolve()
        self.settings_path = self.workspace / "data" / "desktop-settings.json"
        self.controller = controller or DesktopController(self.workspace / "reports/desktop_trials")
        self.worker = None
        self.stopping_workers = []
        self.jobs = set()
        self.generation = 0
        self.camera_pending = False
        self.closing_requested = False
        self.demo_started = None
        self.last_render = 0
        self.setWindowTitle(f"PinchPilot {__version__} · 桌面测试版")
        self.resize(980, 700)
        self._build_ui()
        self.setStyleSheet(STYLE)
        self._restore_settings()
        self._connect_settings()
        self._configure()
        self._toggle_logging(self.logging.isChecked())
        self.camera_authorized.connect(self._camera_authorized, Qt.ConnectionType.QueuedConnection)
        self.escape = QShortcut(QKeySequence("Escape"), self)
        self.escape.activated.connect(self.stop_control)
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        if demo:
            self.start_demo()
        self._refresh()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel("PinchPilot")
        title.setObjectName("title")
        heading.addWidget(title)
        subtitle = QLabel(f"桌面测试版  {__version__}  ·  主屏幕")
        subtitle.setObjectName("subtitle")
        heading.addWidget(subtitle)
        heading.addStretch()
        self.chip = QLabel("预览 · 未接管鼠标")
        self.chip.setObjectName("chip")
        heading.addWidget(self.chip)
        layout.addLayout(heading)

        body = QHBoxLayout()
        body.setSpacing(18)
        left = QVBoxLayout()
        self.camera_view = CameraView()
        left.addWidget(self.camera_view, 1)
        self.gesture = QLabel()
        self.gesture.setObjectName("notice")
        self.gesture.setWordWrap(True)
        left.addWidget(self.gesture)
        grid = QGridLayout()
        self.metrics_labels = {}
        for i, (key, label) in enumerate(
            (
                ("fps", "处理帧率"),
                ("inference", "推理耗时"),
                ("age", "读帧后帧龄"),
                ("tracking", "手部跟踪"),
                ("buttons", "按下 / 拖拽就绪"),
                ("duration", "本次运行"),
            )
        ):
            widget = QLabel(label + "\n—")
            widget.setObjectName("metrics")
            widget.setMinimumWidth(135)
            self.metrics_labels[key] = widget
            grid.addWidget(widget, i // 3, i % 3)
        left.addLayout(grid)
        left.addStretch(0)
        body.addLayout(left, 1)

        side = QWidget()
        side.setObjectName("side")
        side.setFixedWidth(305)
        controls = QVBoxLayout(side)
        controls.setContentsMargins(0, 0, 0, 0)
        self.camera_button = QPushButton("启动相机")
        self.camera_button.setObjectName("primary")
        self.camera_button.clicked.connect(self._toggle_camera)
        self.live_button = QPushButton("启用鼠标控制")
        self.live_button.setObjectName("live")
        self.live_button.setCheckable(True)
        self.live_button.clicked.connect(self._toggle_live)
        self.stop_button = QPushButton("停止控制 · Esc")
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self.stop_control)
        for button in (self.camera_button, self.live_button, self.stop_button):
            controls.addWidget(button)
        self.right = QCheckBox("启用右键 · 拇指＋无名指")
        self.drag = QCheckBox("启用拖拽 · 食拇保持捏合")
        self.scroll = QCheckBox("启用滚轮 · V 手势上下移动")
        self.preview = QCheckBox("显示相机预览")
        self.on_top = QCheckBox("窗口置顶")
        self.logging = QCheckBox("保存本地测试记录")
        for option in (self.right, self.drag, self.scroll, self.preview, self.on_top, self.logging):
            option.setChecked(option is not self.on_top)
            controls.addWidget(option)
        privacy = QLabel("记录性能、状态和按钮事件；不保存相机图像。")
        privacy.setObjectName("subtitle")
        privacy.setWordWrap(True)
        controls.addWidget(privacy)
        folder = QPushButton("打开测试记录")
        folder.clicked.connect(self.open_reports)
        controls.addWidget(folder)
        self.advanced_button = QPushButton("展开手感设置 ▾")
        self.advanced_button.setCheckable(True)
        controls.addWidget(self.advanced_button)
        self.advanced = QScrollArea()
        self.advanced.setWidgetResizable(True)
        advanced_content = QWidget()
        advanced_content.setObjectName("side")
        advanced_layout = QVBoxLayout(advanced_content)
        self.tripod_controls = TripodControls()
        self.tripod_controls.probe_button.hide()
        advanced_layout.addWidget(self.tripod_controls)
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("摄像头编号"))
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        camera_row.addWidget(self.camera_index)
        advanced_layout.addLayout(camera_row)
        demo_button = QPushButton("查看合成演示")
        demo_button.clicked.connect(self.start_demo)
        advanced_layout.addWidget(demo_button)
        self.advanced.setWidget(advanced_content)
        self.advanced.hide()
        controls.addWidget(self.advanced, 1)
        self.advanced_button.toggled.connect(self._show_advanced)
        controls.addStretch()
        body.addWidget(side)
        layout.addLayout(body, 1)
        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.notice)

    def _show_advanced(self, enabled):
        self.advanced.setVisible(enabled)
        self.advanced_button.setText("收起手感设置 ▴" if enabled else "展开手感设置 ▾")

    def _choices(self):
        return {
            name: getattr(self.tripod_controls, name)
            for name in ("stability", "contact", "right_contact", "drag_hold", "scroll_speed")
        }

    def _restore_settings(self):
        if not self.settings_path.exists():
            return
        try:
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError("settings must be an object")
            for name in ("right", "drag", "scroll", "preview", "on_top", "logging"):
                value = settings.get(name)
                if isinstance(value, bool):
                    getattr(self, name).setChecked(value)
            for name, choice in self._choices().items():
                index = choice.findData(settings.get(name))
                if index >= 0:
                    choice.setCurrentIndex(index)
            self.tripod_controls.restore_span(settings.get("span", 0.3))
            index = settings.get("camera_index", 0)
            if type(index) is int and 0 <= index <= 8:
                self.camera_index.setValue(index)
        except (OSError, ValueError) as error:
            self.controller.notice = f"设置读取失败，使用默认设置：{error}"
        self.camera_view.setVisible(self.preview.isChecked())
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.on_top.isChecked())

    def _save_settings(self):
        settings = {
            name: getattr(self, name).isChecked()
            for name in ("right", "drag", "scroll", "preview", "on_top", "logging")
        }
        settings.update({name: choice.currentData() for name, choice in self._choices().items()})
        settings["span"] = self.tripod_controls.configuration().span
        settings["camera_index"] = self.camera_index.value()
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            temporary.replace(self.settings_path)
        except OSError as error:
            self.controller.notice = f"设置未保存：{error}"

    def _connect_settings(self):
        self.right.toggled.connect(self._configure)
        self.drag.toggled.connect(self._configure)
        self.scroll.toggled.connect(self._configure)
        self.tripod_controls.changed.connect(self._configure)
        self.tripod_controls.recenter.connect(self.stop_control)
        self.camera_index.valueChanged.connect(self._camera_changed)
        self.preview.toggled.connect(self._toggle_preview)
        self.on_top.toggled.connect(self._toggle_top)
        self.logging.toggled.connect(self._toggle_logging)

    def _configure(self):
        self.controller.configure(
            replace(
                self.tripod_controls.configuration(),
                right_enabled=self.right.isChecked(),
                drag_enabled=self.drag.isChecked(),
                scroll_enabled=self.scroll.isChecked(),
            )
        )
        self._save_settings()
        self._refresh()

    def _camera_changed(self):
        self.stop_camera()
        self._save_settings()

    def _toggle_preview(self, enabled):
        self.camera_view.setVisible(enabled)
        if not enabled:
            self.camera_view.image = None
        self._save_settings()

    def _toggle_top(self, enabled):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.show()
        self._save_settings()

    def _toggle_logging(self, enabled):
        self.controller.metrics.set_logging(
            enabled, asdict(self.controller.engine.config), self.controller.source
        )
        self.logging.blockSignals(True)
        self.logging.setChecked(self.controller.metrics.file is not None)
        self.logging.blockSignals(False)
        self._save_settings()

    def _toggle_camera(self):
        if self.worker or self.camera_pending:
            self.stop_camera()
        else:
            self.start_camera()

    def _job(self, fn, done, generation):
        job = Job(fn, self)
        self.jobs.add(job)
        job.succeeded.connect(done, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(
            lambda message: self._camera_failed(generation, message),
            Qt.ConnectionType.QueuedConnection,
        )
        job.finished.connect(lambda: self._job_finished(job), Qt.ConnectionType.QueuedConnection)
        job.start()

    def _job_finished(self, job):
        self.jobs.discard(job)
        job.deleteLater()
        if self.closing_requested and not self.jobs:
            self.close()

    def start_camera(self):
        self.stop_camera()
        if self.jobs or any(worker.is_alive() for worker in self.stopping_workers):
            self.controller.notice = "正在结束上一次相机/模型任务，请稍后再启动"
            return
        self.camera_pending = True
        generation = self.generation
        self.controller.notice = "正在校验手部模型；首次使用会下载约 8 MB…"
        self._job(fetch_model, lambda _: self._begin_camera(generation), generation)
        self._refresh()

    def _begin_camera(self, generation):
        if generation != self.generation:
            return
        self.controller.notice = "正在检查摄像头权限，请处理系统授权提示。"
        try:
            request_camera_access(lambda allowed: self.camera_authorized.emit(generation, allowed))
        except Exception as error:
            self._camera_failed(generation, str(error))

    def _camera_authorized(self, generation, allowed):
        if generation != self.generation:
            return
        if not allowed:
            self._camera_failed(generation, "摄像头权限未获准，请在系统隐私设置中授权启动程序。")
            return
        try:
            self.worker = CameraWorker(default_model_path(), self.camera_index.value())
            self.controller.set_source("camera")
            self.worker.start()
            self.camera_pending = False
            self.controller.notice = (
                "相机启动中…有画面后点击「启用鼠标控制」。" + camera_permission()
            )
        except Exception as error:
            self._camera_failed(generation, str(error))

    def _camera_failed(self, generation, message):
        if generation != self.generation:
            return
        self.stop_camera()
        self.controller.notice = message
        self._refresh()

    def stop_camera(self):
        self.generation += 1
        self.camera_pending = False
        self.controller.set_source("none")
        if self.worker:
            self.worker.stop()
            self.stopping_workers.append(self.worker)
            self.worker = None
        self.demo_started = None
        self.camera_view.set_frame(None, None, self.controller.result)
        self._refresh()

    def start_demo(self):
        self.stop_camera()
        self.controller.set_source("synthetic_demo")
        self.demo_started = time.monotonic()
        self.controller.notice = "合成演示 · 不接相机、不控制系统鼠标；数据不能当真人识别效果。"
        self._refresh()

    def _toggle_live(self, enabled):
        try:
            if enabled:
                self.controller.enable()
            else:
                self.stop_control()
        except Exception as error:
            self.controller.stop("enable_failed")
            self.controller.notice = str(error)
        self._refresh()

    def stop_control(self):
        self.controller.stop("user_stop")
        if not self.controller.pending_release:
            self.controller.notice = "鼠标控制已停止 · 相机可继续预览；再次启用会从当前光标接管。"
        self._refresh()

    def _tick(self):
        try:
            now = time.monotonic()
            self.stopping_workers = [w for w in self.stopping_workers if w.is_alive()]
            packet = None
            if self.worker:
                if self.worker.failure:
                    message = self.worker.failure
                    self.stop_camera()
                    self.controller.notice = message
                else:
                    packet = self.worker.pop()
            elif self.controller.source == "synthetic_demo":
                frame = tripod_demo_frame(now, now - self.demo_started)
                packet = Packet(None, frame, 0.0, 0.0, now)
            if packet is not None and self.controller.consume(packet):
                if self.preview.isChecked() and not self.isMinimized():
                    self.camera_view.box = self.controller.engine.active_box
                    self.camera_view.set_frame(
                        packet.rgb,
                        packet.frame,
                        self.controller.result,
                        self.controller.source == "synthetic_demo",
                    )
            self.controller.tick()
            if now - self.last_render >= 0.10:
                self._refresh()
                self.last_render = now
        except Exception as error:
            self.controller.stop("runtime_error")
            self.controller.notice = f"运行已停止：{error}"
            self._refresh()

    def _refresh(self):
        control = self.controller
        active = control.active
        self.live_button.blockSignals(True)
        self.live_button.setChecked(active)
        self.live_button.blockSignals(False)
        self.live_button.setText("关闭鼠标控制" if active else "启用鼠标控制")
        self.live_button.setEnabled(
            not control.pending_release and control.source == "camera" and control.fresh()
        )
        self.camera_button.setText("停止相机" if self.worker or self.camera_pending else "启动相机")
        source = {"none": "相机关闭", "camera": "实时相机", "synthetic_demo": "合成演示"}[
            control.source
        ]
        if control.source == "camera" and not control.fresh():
            source = "相机等待 / 画面过期"
        self.chip.setText(f"{source} · {'正在控制' if active else '未接管鼠标'}")
        snapshot = control.session.snapshot() if control.session else {}
        held = "左键按住" if snapshot.get("left_held") else "按键已松开"
        if snapshot.get("right_held"):
            held = "右键按住"
        self.gesture.setText(f"{control.result.state} · {held}\n{control.result.hint}")
        data = control.metrics.snapshot(control.clock())
        synthetic = control.source == "synthetic_demo"
        inference = data["inference_ms"]
        age = data["frame_age_ms"]
        values = {
            "fps": f"{'演示' if synthetic else '处理'}帧率\n{data['processed_fps']:.1f} FPS",
            "inference": "推理耗时\n"
            + ("—（合成）" if synthetic else ("—" if inference is None else f"{inference:.1f} ms")),
            "age": "读帧后帧龄\n" + ("—" if age is None or synthetic else f"{age:.0f} ms"),
            "tracking": f"{'合成手部' if synthetic else '手部跟踪'}\n{'可见' if data['tracked'] else '未跟踪'} · 丢失 {data['tracking_losses']}",
            "buttons": f"按下 / 拖拽就绪\n左 {data['sent'].get('down', 0)} · 右 {data['sent'].get('right_down', 0)} · 拖 {data['drag_starts']}",
            "duration": f"本次运行 / 滚轮发送\n{int(data['elapsed_s']) // 60:02d}:{int(data['elapsed_s']) % 60:02d} · 滚 {data['sent'].get('scroll', 0)}",
        }
        for key, value in values.items():
            self.metrics_labels[key].setText(value)
        message = control.notice
        if control.metrics.error:
            message += "\n" + control.metrics.error
        self.notice.setText(message)
        if control.metrics.error:
            self.logging.blockSignals(True)
            self.logging.setChecked(False)
            self.logging.blockSignals(False)

    def open_reports(self):
        directory = self.controller.metrics.directory
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def closeEvent(self, event):
        self.stop_camera()
        if not self.controller.close():
            self.controller.notice = "仍在重试松键，暂不能关闭；请先用实体鼠标松开按键。"
            self._refresh()
            event.ignore()
            return
        if self.jobs:
            self.closing_requested = True
            self.controller.notice = "鼠标控制已停止；模型任务结束后自动关闭。"
            self._refresh()
            event.ignore()
            return
        self.timer.stop()
        for worker in self.stopping_workers:
            worker.join(timeout=0.5)
        event.accept()


def run_desktop(workspace, demo=False, smoke_seconds=None, screenshot=None):
    workspace = workspace.resolve()
    (workspace / "data").mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(workspace / "data/desktop.lock"))
    if not lock.tryLock(0):
        raise RuntimeError("这个工作目录已有桌面测试版在运行，请使用已有窗口。")
    enable_dpi_awareness()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("PinchPilot Desktop")
    window = DesktopWindow(workspace, demo)
    available = app.primaryScreen().availableGeometry()
    window.resize(min(980, available.width() - 40), min(700, available.height() - 40))
    window.show()
    if screenshot:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        QTimer.singleShot(1500, lambda: window.grab().save(str(screenshot)))
    if smoke_seconds:
        QTimer.singleShot(int(smoke_seconds * 1000), window.close)
    previous_signals = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_signals[sig] = signal.signal(sig, lambda *_: QTimer.singleShot(0, window.close))
    try:
        return app.exec()
    finally:
        # Also covers application-level Quit, not just the window's close button.
        window.stop_camera()
        window.controller.close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        lock.unlock()
