"""The everyday trial window. Camera and system mouse always start switched off."""

import json
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .app import STYLE, Job
from .desktop_control import DesktopController
from .framing_overlay import FramingOverlay
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
        self.control_hand_status = "相机关闭"
        self._calibration_seen = None
        self._calibration_summary = ""
        self._wrist_calibration_seen = None
        self._wrist_calibration_summary = ""
        self.framing_overlay = FramingOverlay()
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
        hand_row = QHBoxLayout()
        hand_row.addWidget(QLabel("控制手"))
        self.control_hand = QComboBox()
        for label, value in (("自动锁定", "auto"), ("只用右手", "Right"), ("只用左手", "Left")):
            self.control_hand.addItem(label, value)
        self.control_hand.setToolTip(
            "自动：先只露出要控制的一只手，锁定后不会换手。\n"
            "切换选项会停止相机和鼠标控制；重新启动相机后生效。"
        )
        hand_row.addWidget(self.control_hand, 1)
        controls.addLayout(hand_row)
        self.hand_status = QLabel()
        self.hand_status.setObjectName("subtitle")
        self.hand_status.setWordWrap(True)
        controls.addWidget(self.hand_status)
        self.framing_status_label = QLabel()
        self.framing_status_label.setObjectName("notice")
        self.framing_status_label.setWordWrap(True)
        self.framing_status_label.setTextFormat(Qt.TextFormat.PlainText)
        controls.addWidget(self.framing_status_label)
        # Keep camera/control/hand selection visible while the settings scroll together.
        self.settings_area = QScrollArea()
        self.settings_area.setWidgetResizable(True)
        self.settings_area.setFrameShape(QScrollArea.Shape.NoFrame)
        settings_content = QWidget()
        settings_content.setObjectName("side")
        self.settings_area.setWidget(settings_content)
        controls.addWidget(self.settings_area, 1)
        controls = QVBoxLayout(settings_content)
        controls.setContentsMargins(0, 0, 0, 0)
        self.right = QCheckBox("启用右键 · 拇指＋无名指")
        self.drag = QCheckBox("启用拖拽 · 食拇保持捏合")
        self.scroll = QCheckBox("启用滚轮 · V 手势上下移动")
        self.preview = QCheckBox("显示相机预览")
        self.on_top = QCheckBox("窗口置顶")
        self.logging = QCheckBox("保存本地测试记录")
        self.edge_assist = QCheckBox("画面边缘提醒")
        self.edge_assist.setToolTip("提醒不会改变鼠标移动或点击；控制时也会在桌面显示提示。")
        for option in (
            self.edge_assist,
            self.right,
            self.drag,
            self.scroll,
            self.preview,
            self.on_top,
            self.logging,
        ):
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
        self.advanced = QWidget()
        self.advanced.setObjectName("side")
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.tripod_controls = TripodControls()
        self.tripod_controls.motion_profile.setCurrentIndex(
            self.tripod_controls.motion_profile.findData("precise")
        )
        self.tripod_controls.probe_button.hide()
        advanced_layout.addWidget(self.tripod_controls)
        self.calibration_button = QPushButton("6 秒静止校准")
        self.calibration_button.clicked.connect(self._toggle_calibration)
        advanced_layout.addWidget(self.calibration_button)
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setRange(0, 100)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setStyleSheet(
            "QProgressBar { min-height: 18px; border: 1px solid #31546b; border-radius: 4px;"
            "background: #10202e; color: #d8e6ed; text-align: center; }"
            "QProgressBar::chunk { background: #177f79; border-radius: 3px; }"
        )
        advanced_layout.addWidget(self.calibration_progress)
        self.calibration_status = QLabel()
        self.calibration_status.setObjectName("subtitle")
        self.calibration_status.setWordWrap(True)
        advanced_layout.addWidget(self.calibration_status)
        self.clear_calibration_button = QPushButton("清除静止校准")
        self.clear_calibration_button.clicked.connect(self._clear_calibration)
        advanced_layout.addWidget(self.clear_calibration_button)
        calibration_hint = QLabel(
            "支撑前臂，拇中捏住并保持静止，食指和无名指移开；校准期间暂停鼠标控制。"
            "仅测量静止抖动，不代表动作识别准确率。"
        )
        calibration_hint.setObjectName("subtitle")
        calibration_hint.setWordWrap(True)
        advanced_layout.addWidget(calibration_hint)
        self.rest_calibration_widgets = (
            self.calibration_button,
            self.calibration_progress,
            self.calibration_status,
            self.clear_calibration_button,
            calibration_hint,
        )
        self.wrist_calibration_group = QWidget()
        wrist_layout = QVBoxLayout(self.wrist_calibration_group)
        wrist_layout.setContentsMargins(0, 0, 0, 0)
        self.wrist_calibration_button = QPushButton("12 秒腕动方向校准")
        self.wrist_calibration_button.clicked.connect(self._toggle_wrist_calibration)
        wrist_layout.addWidget(self.wrist_calibration_button)
        self.wrist_calibration_progress = QProgressBar()
        self.wrist_calibration_progress.setRange(0, 100)
        self.wrist_calibration_progress.setValue(0)
        self.wrist_calibration_progress.setStyleSheet(self.calibration_progress.styleSheet())
        wrist_layout.addWidget(self.wrist_calibration_progress)
        self.wrist_calibration_status = QLabel()
        self.wrist_calibration_status.setObjectName("subtitle")
        self.wrist_calibration_status.setWordWrap(True)
        self.wrist_calibration_status.setTextFormat(Qt.TextFormat.PlainText)
        wrist_layout.addWidget(self.wrist_calibration_status)
        self.clear_wrist_calibration_button = QPushButton("清除腕动方向校准")
        self.clear_wrist_calibration_button.clicked.connect(self._clear_wrist_calibration)
        wrist_layout.addWidget(self.clear_wrist_calibration_button)
        wrist_hint = QLabel(
            "支撑前臂并保持拇中捏合，食指和无名指移开。"
            "按提示示范舒服的中立姿态、向右与向上转动后的姿态；每步准备 3 秒、记录 1 秒。"
            "只保存方向参数，不保存关键点；校准结束不会自动启用鼠标。"
        )
        wrist_hint.setObjectName("subtitle")
        wrist_hint.setWordWrap(True)
        wrist_layout.addWidget(wrist_hint)
        advanced_layout.addWidget(self.wrist_calibration_group)
        self.wrist_calibration_group.hide()
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("摄像头编号"))
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        camera_row.addWidget(self.camera_index)
        advanced_layout.addLayout(camera_row)
        demo_button = QPushButton("查看合成演示")
        demo_button.clicked.connect(self.start_demo)
        advanced_layout.addWidget(demo_button)
        self.advanced.hide()
        controls.addWidget(self.advanced)
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
        if enabled:
            QTimer.singleShot(0, self._reveal_sensitivity)

    def _reveal_sensitivity(self):
        if self.advanced.isVisible():
            self.settings_area.verticalScrollBar().setValue(self.advanced.y())
            self.settings_area.ensureWidgetVisible(self.tripod_controls.pointer_basis, 0, 35)

    def _choices(self):
        return {
            name: getattr(self.tripod_controls, name)
            for name in (
                "pointer_basis",
                "motion_profile",
                "stability",
                "contact",
                "right_contact",
                "drag_hold",
                "scroll_speed",
            )
        }

    def _restore_settings(self):
        if not self.settings_path.exists():
            return
        try:
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError("settings must be an object")
            if "motion_profile" not in settings:
                # Preserve existing users' motion until they opt into a new profile.
                self.tripod_controls.motion_profile.setCurrentIndex(
                    self.tripod_controls.motion_profile.findData("classic")
                )
            for name in ("right", "drag", "scroll", "preview", "on_top", "logging", "edge_assist"):
                value = settings.get(name)
                if isinstance(value, bool):
                    getattr(self, name).setChecked(value)
            for name, choice in self._choices().items():
                index = choice.findData(settings.get(name))
                if index >= 0:
                    choice.setCurrentIndex(index)
            self.tripod_controls.restore_span(settings.get("span", 0.3))
            self.tripod_controls.restore_noise(
                settings.get("rest_noise_x", 0.0), settings.get("rest_noise_y", 0.0)
            )
            self.tripod_controls.restore_wrist_calibration(settings.get("wrist_calibration", ()))
            index = settings.get("camera_index", 0)
            if type(index) is int and 0 <= index <= 8:
                self.camera_index.setValue(index)
            hand_index = self.control_hand.findData(settings.get("control_hand", "auto"))
            if hand_index >= 0:
                self.control_hand.setCurrentIndex(hand_index)
        except (OSError, ValueError) as error:
            self.controller.notice = f"设置读取失败，使用默认设置：{error}"
        self.camera_view.setVisible(self.preview.isChecked())
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.on_top.isChecked())

    def _save_settings(self):
        settings = {
            name: getattr(self, name).isChecked()
            for name in ("right", "drag", "scroll", "preview", "on_top", "logging", "edge_assist")
        }
        settings.update({name: choice.currentData() for name, choice in self._choices().items()})
        settings["span"] = self.tripod_controls.configuration().span
        settings["rest_noise_x"] = self.tripod_controls.rest_noise_x
        settings["rest_noise_y"] = self.tripod_controls.rest_noise_y
        settings["wrist_calibration"] = list(self.tripod_controls.wrist_calibration)
        settings["camera_index"] = self.camera_index.value()
        settings["control_hand"] = self.control_hand.currentData()
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
        self.control_hand.currentIndexChanged.connect(self._camera_changed)
        self.preview.toggled.connect(self._toggle_preview)
        self.on_top.toggled.connect(self._toggle_top)
        self.logging.toggled.connect(self._toggle_logging)
        self.edge_assist.toggled.connect(self._toggle_edge_assist)

    def _toggle_edge_assist(self, _):
        self._save_settings()
        self._refresh_framing()

    def _refresh_framing(self):
        control = self.controller
        status = control.framing_status
        enabled = self.edge_assist.isChecked() and not control.calibrating
        self.framing_status_label.setText(control.framing_hint)
        fresh_camera = control.source == "camera" and control.fresh()
        self.framing_status_label.setVisible(enabled and fresh_camera)
        edges = status.edges if enabled and fresh_camera and status.state == "edge" else ()
        if self.camera_view.frame_edges != edges:
            self.camera_view.frame_edges = edges
            self.camera_view.update()
        self.framing_overlay.set_hint(
            control.framing_hint,
            enabled
            and control.active
            and fresh_camera
            and not control.calibrating
            and not self.closing_requested
            and status.state in ("edge", "lost", "waiting"),
        )

    def _configure(self):
        screen = QApplication.primaryScreen()
        size = screen.size() if screen is not None else None
        self.controller.configure(
            replace(
                self.tripod_controls.configuration(),
                screen_width=float(size.width()) if size else 1920.0,
                screen_height=float(size.height()) if size else 1080.0,
                right_enabled=self.right.isChecked(),
                drag_enabled=self.drag.isChecked(),
                scroll_enabled=self.scroll.isChecked(),
            )
        )
        self._save_settings()
        self.camera_view.box = self.controller.engine.active_box
        self.camera_view.update()
        self._refresh()

    def _toggle_calibration(self):
        try:
            if self.controller.calibrating:
                self.controller.cancel_calibration()
            else:
                self.controller.start_calibration()
        except Exception as error:
            self.controller.notice = f"校准未启动：{error}"
        self._refresh()
        if self.controller.calibrating:
            QTimer.singleShot(0, self._reveal_calibration)

    def _reveal_calibration(self):
        if self.advanced.isVisible():
            self.settings_area.verticalScrollBar().setValue(
                self.advanced.y() + self.calibration_button.y() - 35
            )

    def _clear_calibration(self):
        self._calibration_seen = self.controller.calibration_result
        self._calibration_summary = ""
        self.tripod_controls.restore_noise(0.0, 0.0)
        self._configure()
        self.controller.notice = "已清除静止校准；再次启用鼠标控制后生效。"
        self._refresh()

    def _refresh_calibration(self):
        control = self.controller
        result = control.calibration_result
        if result is not None and result is not self._calibration_seen:
            self._calibration_seen = result
            if self.tripod_controls.restore_noise(result["rest_noise_x"], result["rest_noise_y"]):
                self._calibration_summary = f"已校准 · 采集时波动 {result['noise_px']:.1f} 逻辑像素 · {result['samples']} 帧"
                self._save_settings()
        wrist = self.tripod_controls.pointer_basis.currentData() == "wrist"
        for widget in self.rest_calibration_widgets:
            widget.setVisible(not wrist)
        calibrating = control.calibrating and control.wrist_capture is None
        classic = self.tripod_controls.motion_profile.currentData() == "classic"
        self.calibration_button.setText("取消静止校准" if calibrating else "6 秒静止校准")
        self.calibration_button.setToolTip(
            "静止校准适用于精细 / 自适应模式" if classic else "准备 1 秒、静止采集 5 秒"
        )
        self.calibration_button.setEnabled(
            calibrating
            or (
                not classic
                and not wrist
                and not control.calibrating
                and control.source == "camera"
                and control.fresh()
                and not control.pending_release
            )
        )
        progress = control.calibration_progress if calibrating else bool(self._calibration_summary)
        self.calibration_progress.setValue(round(progress * 100))
        calibrated = bool(self.tripod_controls.rest_noise_x or self.tripod_controls.rest_noise_y)
        self.clear_calibration_button.setEnabled(
            calibrated or calibrating or bool(self._calibration_summary)
        )
        if calibrating:
            text = "采集中 · 保持拇中捏合并静止；食指和无名指移开。"
        elif classic:
            text = "原版用于对照；静止校准适用于精细 / 自适应模式。"
        elif self._calibration_summary:
            text = self._calibration_summary
        elif calibrated:
            text = "已恢复静止校准 · 可在当前姿势下重新采集。"
        else:
            text = "尚未校准 · 先启动相机，并让控制手清晰可见。"
        self.calibration_status.setText(text)

    def _toggle_wrist_calibration(self):
        try:
            if self.controller.wrist_capture is not None:
                self.controller.cancel_calibration()
            else:
                self.controller.start_wrist_calibration()
        except Exception as error:
            self.controller.notice = f"腕动校准未启动：{error}"
        self._refresh()
        if self.controller.wrist_capture is not None:
            QTimer.singleShot(0, self._reveal_wrist_calibration)

    def _reveal_wrist_calibration(self):
        if self.advanced.isVisible():
            self.settings_area.ensureWidgetVisible(self.wrist_calibration_status, 0, 35)

    def _clear_wrist_calibration(self):
        self._wrist_calibration_seen = self.controller.wrist_calibration_result
        self._wrist_calibration_summary = ""
        self.tripod_controls.restore_wrist_calibration(())
        self._configure()
        self.controller.notice = "已清除腕动方向校准；重新完成 12 秒校准后才能启用腕动控制。"
        self._refresh()

    def _refresh_wrist_calibration(self):
        control = self.controller
        result = control.wrist_calibration_result
        if result is not None and result is not self._wrist_calibration_seen:
            self._wrist_calibration_seen = result
            if self.tripod_controls.restore_wrist_calibration(result):
                self._wrist_calibration_summary = "腕动方向已校准 · 请手动启用鼠标控制。"
                self._save_settings()
        wrist = self.tripod_controls.pointer_basis.currentData() == "wrist"
        self.wrist_calibration_group.setVisible(wrist)
        capturing = control.wrist_capture is not None
        calibrated = bool(self.tripod_controls.wrist_calibration)
        self.wrist_calibration_button.setText(
            "取消腕动方向校准" if capturing else "12 秒腕动方向校准"
        )
        self.wrist_calibration_button.setEnabled(
            capturing
            or (
                wrist
                and not control.calibrating
                and control.source == "camera"
                and control.fresh()
                and not control.pending_release
            )
        )
        progress = control.wrist_calibration_progress if capturing else float(calibrated)
        self.wrist_calibration_progress.setValue(round(progress * 100))
        self.clear_wrist_calibration_button.setEnabled(calibrated or capturing)
        if capturing:
            text = control.wrist_capture.hint(control.clock())
        elif self._wrist_calibration_summary and calibrated:
            text = self._wrist_calibration_summary
        elif calibrated:
            text = "已恢复腕动方向校准 · 停腕即停；姿势改变后可重新示范方向。"
        else:
            text = "腕动尚未校准，暂不能启用鼠标。先启动相机，再完成 12 秒方向校准。"
        self.wrist_calibration_status.setText(text)

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
        job.finished.connect(self._job_finished, Qt.ConnectionType.QueuedConnection)
        job.start()

    @Slot()
    def _job_finished(self):
        # Do not capture the soon-to-be-deleted QThread in its own signal callback.
        job = self.sender()
        job.succeeded.disconnect()
        job.failed.disconnect()
        job.finished.disconnect()
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
            self.worker = CameraWorker(
                default_model_path(), self.camera_index.value(), self.control_hand.currentData()
            )
            self.control_hand_status = "等待控制手 · 自动模式先只露出一只手"
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
        self.control_hand_status = "相机关闭"
        self.camera_view.set_frame(None, None, self.controller.result)
        self._refresh()

    def start_demo(self):
        self.stop_camera()
        self.controller.set_source("synthetic_demo")
        self.control_hand_status = "合成演示 · 不选择真人控制手"
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
                self.control_hand_status = packet.hand_status or self.control_hand_status
                if self.preview.isChecked() and not self.isMinimized():
                    self.camera_view.box = self.controller.engine.active_box
                    self.camera_view.set_frame(
                        packet.rgb,
                        packet.frame,
                        self.controller.result,
                        self.controller.source == "synthetic_demo",
                    )
            self.controller.tick()
            self._refresh_framing()
            if now - self.last_render >= 0.10:
                self._refresh()
                self.last_render = now
        except Exception as error:
            self.controller.stop("runtime_error")
            self.controller.notice = f"运行已停止：{error}"
            self._refresh()

    def _refresh(self):
        control = self.controller
        self._refresh_calibration()
        self._refresh_wrist_calibration()
        self._refresh_framing()
        active = control.active
        self.live_button.blockSignals(True)
        self.live_button.setChecked(active)
        self.live_button.blockSignals(False)
        self.live_button.setText("关闭鼠标控制" if active else "启用鼠标控制")
        self.live_button.setEnabled(
            not control.calibrating
            and not control.pending_release
            and control.source == "camera"
            and control.fresh()
            and (
                control.engine.config.pointer_basis != "wrist"
                or bool(control.engine.config.wrist_calibration)
            )
        )
        self.live_button.setToolTip(
            "请先完成 12 秒腕动方向校准"
            if control.engine.config.pointer_basis == "wrist"
            and not control.engine.config.wrist_calibration
            else ""
        )
        self.camera_button.setText("停止相机" if self.worker or self.camera_pending else "启动相机")
        source = {"none": "相机关闭", "camera": "实时相机", "synthetic_demo": "合成演示"}[
            control.source
        ]
        if control.source == "camera" and not control.fresh():
            source = "相机等待 / 画面过期"
        self.chip.setText(f"{source} · {'正在控制' if active else '未接管鼠标'}")
        self.hand_status.setText(self.control_hand_status)
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
        self.framing_overlay.set_hint("", False)
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
        self.framing_overlay.dispose()
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
        window.framing_overlay.dispose()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        lock.unlock()
