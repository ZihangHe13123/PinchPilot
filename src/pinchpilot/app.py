import json
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .demo import demo_frame, finger_demo_frame
from .domain import EngineConfig
from .engine import GestureEngine
from .features import extract
from .finger_controls import FingerControls
from .platform_io import MouseOutput, camera_permission, enable_dpi_awareness, request_camera_access
from .single_finger import SingleFingerEngine
from .storage import Recorder, calibrate, safe_name, save_json
from .vision import CameraWorker, default_model_path, fetch_model
from .widgets import CameraView, PracticeView

STYLE = """
QMainWindow, QWidget#root, QWidget#side, QWidget#practice { background: #0b1420; color: #dce7f0; }
QWidget { color: #dce7f0; font-size: 13px; }
QLabel#title { font-size: 27px; font-weight: 700; color: #effaf9; }
QLabel#subtitle { color: #91a8ba; font-size: 12px; }
QLabel#chip { color: #85e1ce; background: #123d3c; padding: 10px 14px; border-radius: 14px; }
QLabel#notice { color: #aac1d0; background: #142436; padding: 10px; border-radius: 8px; }
QLabel#metrics { color: #a9c5d4; background: #101f2e; padding: 12px; border-radius: 8px; }
QGroupBox { background: #122132; border: 1px solid #273b4e; border-radius: 10px;
            margin-top: 20px; padding: 16px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; top: 1px; color: #b2d4df; }
QPushButton { background: #243b50; border: 1px solid #365369; border-radius: 7px; padding: 9px 12px; }
QPushButton:hover { background: #304f67; }
QPushButton:disabled { color: #617484; background: #162536; border-color: #263645; }
QPushButton#primary { background: #167d78; border-color: #259a90; color: white; font-weight: 600; }
QPushButton#live:checked { background: #a45531; border-color: #db8753; color: white; }
QPushButton#stop { color: #ffb8a7; border-color: #765046; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #0d1a29; border: 1px solid #314a60;
                              border-radius: 5px; padding: 6px; selection-background-color: #167d78; }
QComboBox QAbstractItemView { background: #172b3d; color: #e0ebf4; selection-background-color: #225a62; }
QTabWidget::pane { border: 1px solid #263a4e; border-radius: 8px; }
QTabBar::tab { background: #162538; color: #8fa8bc; padding: 10px 20px; }
QTabBar::tab:selected { background: #244253; color: #aff7e9; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #122132; width: 7px; }
QScrollBar::handle:vertical { background: #365369; min-height: 24px; border-radius: 3px; }
QMessageBox { background: #172738; }
"""


class Job(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.succeeded.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    camera_authorized = Signal(int, bool)

    def __init__(self, workspace: Path, demo: bool = False, interaction: str = "pinch"):
        super().__init__()
        self.workspace = workspace.resolve()
        self.record_dir = self.workspace / "data" / "recordings"
        self.report_dir = self.workspace / "reports"
        self.engine = GestureEngine(
            EngineConfig(
                box_left=0.35,
                box_top=0.35,
                box_right=0.65,
                box_bottom=0.65,
                reanchor_on_open=True,
            )
        )
        self.finger_engine = SingleFingerEngine()
        self.worker = None
        self.stopping_workers = []
        self.jobs = set()
        self.output = None
        self.predictor = None
        self.recorder = None
        self.countdown = self.record_until = None
        self.source = "none"
        self.current_frame = None
        self.last_packet_at = 0.0
        self.event_count = 0
        self.training_busy = False
        self.download_busy = False
        self.source_generation = 0
        self.permission_pending = False
        self.camera_authorized.connect(self._camera_authorized, Qt.ConnectionType.QueuedConnection)
        log_dir = self.workspace / "data" / "events"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = (log_dir / f"events_{time.time_ns()}.jsonl").open("w", encoding="utf-8")
        self.setWindowTitle("PinchPilot · 手势交互实验室")
        self.resize(1240, 820)
        self._build_ui()
        self.setStyleSheet(STYLE)
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.escape = QShortcut(QKeySequence("Escape"), self)
        self.escape.activated.connect(self._emergency)
        self.set_notice(
            "先预览，再启用。张开手定位，拇指与食指捏合点击；捏住移动拖拽，V 手势上下移动滚动。"
        )
        index = self.interaction_choice.findData(interaction)
        if index < 0:
            raise ValueError("未知交互模式")
        self.interaction_choice.setCurrentIndex(index)
        if demo:
            self.start_demo()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(14)
        header = QHBoxLayout()
        names = QVBoxLayout()
        title = QLabel("PinchPilot")
        title.setObjectName("title")
        subtitle = QLabel("小幅定位 · 主动点击 · 停留点击     /     摄像头交互研究原型")
        subtitle.setObjectName("subtitle")
        names.addWidget(title)
        names.addWidget(subtitle)
        header.addLayout(names)
        header.addStretch()
        self.chip = QLabel("预览模式 · 系统光标不受控")
        self.chip.setObjectName("chip")
        header.addWidget(self.chip)
        outer.addLayout(header)
        body = QHBoxLayout()
        body.setSpacing(18)
        main = QVBoxLayout()
        self.tabs = QTabWidget()
        self.camera_view = CameraView()
        self.camera_view.box = self.engine.active_box
        self.tabs.addTab(self.camera_view, "实时预览")
        practice_page = QWidget()
        practice_page.setObjectName("practice")
        pl = QVBoxLayout(practice_page)
        self.practice = PracticeView()
        pl.addWidget(self.practice, 1)
        taskrow = QHBoxLayout()
        self.task_choice = QComboBox()
        self.task_choice.addItem("点击目标", "click")
        self.task_choice.addItem("拖拽到圆环", "drag")
        self.task_method = QComboBox()
        self.task_method.addItem("手势 · 应用内", True)
        self.task_method.addItem("鼠标 / 触控板", False)
        self.task_method.currentIndexChanged.connect(self._update_task_options)
        starttask = QPushButton("开始 8 个目标")
        starttask.clicked.connect(self.start_practice)
        endtask = QPushButton("结束")
        endtask.clicked.connect(lambda: self.practice.stop())
        for w in (self.task_choice, self.task_method, starttask, endtask):
            taskrow.addWidget(w)
        pl.addLayout(taskrow)
        self.task_status = QLabel("结果区分真实相机、物理输入与合成演示。先练习，再正式计时。")
        self.task_status.setWordWrap(True)
        pl.addWidget(self.task_status)
        self.practice.progress.connect(self.task_status.setText)
        self.practice.finished.connect(lambda path: self.set_notice(f"任务记录已保存：{path}"))
        self.tabs.addTab(practice_page, "点击与拖拽实验")
        main.addWidget(self.tabs, 1)
        self.metrics = QLabel("相机未启动    /    规则识别    /    等待张开手")
        self.metrics.setObjectName("metrics")
        self.metrics.setWordWrap(True)
        main.addWidget(self.metrics)
        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        self.notice.setMinimumHeight(60)
        main.addWidget(self.notice)
        body.addLayout(main, 1)

        panel = QWidget()
        panel.setObjectName("side")
        panel.setMaximumWidth(355)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(14)
        controls, layout = self._group("01  输入与控制")
        self.interaction_choice = QComboBox()
        self.interaction_choice.addItem("捏合 · 当前主方案", "pinch")
        self.interaction_choice.addItem("单指 · 轻弯点击 Demo", "finger-flex")
        self.interaction_choice.addItem("单指 · 停留点击 Demo", "finger-dwell")
        self.interaction_choice.currentIndexChanged.connect(self._change_interaction)
        layout.addWidget(self.interaction_choice)
        row = QHBoxLayout()
        row.addWidget(QLabel("相机编号"))
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 9)
        row.addWidget(self.camera_index)
        layout.addLayout(row)
        self.camera_button = QPushButton("启动摄像头")
        self.camera_button.setObjectName("primary")
        self.camera_button.clicked.connect(self.start_camera)
        layout.addWidget(self.camera_button)
        row = QHBoxLayout()
        self.demo_button = QPushButton("无相机演示")
        self.demo_button.clicked.connect(self.start_demo)
        stop = QPushButton("停止全部")
        stop.setObjectName("stop")
        stop.clicked.connect(self.stop_all)
        row.addWidget(self.demo_button)
        row.addWidget(stop)
        layout.addLayout(row)
        self.live_button = QPushButton("启用系统鼠标控制")
        self.live_button.setObjectName("live")
        self.live_button.setCheckable(True)
        self.live_button.toggled.connect(self._toggle_live)
        layout.addWidget(self.live_button)
        self.pause_button = QPushButton("暂停手势 · 可用 Esc 停止控制")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._pause)
        layout.addWidget(self.pause_button)
        self.stable_check = QCheckBox("捏合稳定定位（可关闭做对照）")
        self.stable_check.setChecked(True)
        self.stable_check.toggled.connect(self._change_stability)
        layout.addWidget(self.stable_check)
        self.motion_range = QComboBox()
        self.motion_range.addItem("小幅移动 · 30% 范围（推荐）", 0.30)
        self.motion_range.addItem("更小幅度 · 20% 范围", 0.20)
        self.motion_range.addItem("原始范围 · 60%（对照）", 0.60)
        self.motion_range.currentIndexChanged.connect(self._change_motion)
        layout.addWidget(self.motion_range)
        self.clutch_check = QCheckBox("握拳休息，张开后从原光标位置继续")
        self.clutch_check.setChecked(True)
        self.clutch_check.toggled.connect(self._change_motion)
        layout.addWidget(self.clutch_check)
        motion_tip = QLabel(
            "范围越小，移动越省，但也更敏感。手移出画面也可休息；重新分开拇指食指后继续。"
        )
        motion_tip.setWordWrap(True)
        motion_tip.setObjectName("subtitle")
        layout.addWidget(motion_tip)
        self.pinch_controls = (self.stable_check, self.motion_range, self.clutch_check, motion_tip)
        self.finger_controls = FingerControls()
        self.finger_controls.changed.connect(self._change_finger_settings)
        self.finger_controls.recenter.connect(self._recenter_finger)
        self.finger_controls.hide()
        layout.addWidget(self.finger_controls)
        helptext = QLabel(
            "首次启动可能请求相机权限。真实控制仅作用于主屏；录制与校准时自动回到预览。"
        )
        helptext.setWordWrap(True)
        helptext.setObjectName("subtitle")
        layout.addWidget(helptext)
        self.input_help = helptext
        side.addWidget(controls)

        capture, layout = self._group("02  采集与个人校准")
        self.capture_group = capture
        form = QFormLayout()
        self.participant = QLineEdit("P01")
        self.session = QLineEdit(self._session_name())
        self.label_choice = QComboBox()
        for title, value in [
            ("张开 / 定位", "open"),
            ("保持捏合", "pinch"),
            ("V 手势 / 滚动", "scroll"),
            ("其他自然动作", "other"),
        ]:
            self.label_choice.addItem(title, value)
        self.record_seconds = QSpinBox()
        self.record_seconds.setRange(3, 30)
        self.record_seconds.setValue(6)
        self.record_seconds.setSuffix(" 秒")
        form.addRow("参与者", self.participant)
        form.addRow("录制场次", self.session)
        form.addRow("人工标签", self.label_choice)
        form.addRow("每段时长", self.record_seconds)
        layout.addLayout(form)
        self.record_button = QPushButton("倒计时 3 秒后录制")
        self.record_button.clicked.connect(self.start_recording)
        layout.addWidget(self.record_button)
        row = QHBoxLayout()
        self.new_session = QPushButton("新场次")
        self.new_session.clicked.connect(lambda: self.session.setText(self._session_name()))
        self.calibrate_button = QPushButton("用本人的数据校准")
        self.calibrate_button.clicked.connect(self.run_calibration)
        row.addWidget(self.new_session)
        row.addWidget(self.calibrate_button)
        layout.addLayout(row)
        self.calibration_status = QLabel("默认阈值 · 捏合 0.26 / 释放 0.40")
        self.calibration_status.setWordWrap(True)
        layout.addWidget(self.calibration_status)
        row = QHBoxLayout()
        self.load_profile_button = QPushButton("加载本人校准")
        self.load_profile_button.clicked.connect(self.load_profile)
        self.reset_profile_button = QPushButton("恢复默认阈值")
        self.reset_profile_button.clicked.connect(self.reset_profile)
        row.addWidget(self.load_profile_button)
        row.addWidget(self.reset_profile_button)
        layout.addLayout(row)
        tip = QLabel(
            "录制开始前摆好所选手形并保持；其他类录自然动作。只保存关键点，不保存相机视频。"
        )
        tip.setWordWrap(True)
        tip.setObjectName("subtitle")
        layout.addWidget(tip)
        side.addWidget(capture)

        models, layout = self._group("03  模型与分组评测")
        self.models_group = models
        self.recognizer = QComboBox()
        self.recognizer.addItems(["规则基线", "已加载的 ML 模型"])
        self.recognizer.currentIndexChanged.connect(self._change_recognizer)
        layout.addWidget(self.recognizer)
        row = QHBoxLayout()
        self.model_choice = QComboBox()
        self.model_choice.addItem("Random Forest", "forest")
        self.model_choice.addItem("SVM", "svm")
        self.group_choice = QComboBox()
        self.group_choice.addItem("按参与者留出", "participant")
        self.group_choice.addItem("按独立场次留出", "session")
        row.addWidget(self.model_choice)
        row.addWidget(self.group_choice)
        layout.addLayout(row)
        self.train_button = QPushButton("训练并与规则比较")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)
        load = QPushButton("加载本项目训练的模型…")
        load.clicked.connect(self.load_model)
        layout.addWidget(load)
        self.model_status = QLabel("尚无自训练模型。先采集，至少两个独立分组。")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)
        folder = QPushButton("打开项目数据文件夹")
        folder.clicked.connect(self.open_data)
        layout.addWidget(folder)
        side.addWidget(models)
        side.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(375)
        scroll.setWidget(panel)
        body.addWidget(scroll)
        outer.addLayout(body, 1)
        self.setCentralWidget(root)

    @property
    def interaction_mode(self):
        return self.interaction_choice.currentData()

    @property
    def active_engine(self):
        return self.engine if self.interaction_mode == "pinch" else self.finger_engine

    @property
    def recognizer_name(self):
        return (
            self.recognizer.currentText() if self.interaction_mode == "pinch" else "单指规则 Demo"
        )

    def _refresh_interaction_controls(self):
        single = self.interaction_mode != "pinch"
        self.live_button.setEnabled(not single and self.source != "synthetic_demo")
        self.live_button.setText("单指 Demo · 仅应用内" if single else "启用系统鼠标控制")
        for widget in self.pinch_controls:
            widget.setVisible(not single)
        self.finger_controls.setVisible(single)
        if single:
            self.finger_controls.select_mode(self.interaction_mode)
        self.models_group.setEnabled(not single)
        self.capture_group.setTitle(
            "02  实验标识 · 单指暂不采集训练数据" if single else "02  采集与个人校准"
        )
        self.input_help.setText(
            "进入「点击与拖拽实验」开始点击任务。两种单指模式均不支持拖拽或滚动。"
            if single
            else "首次启动可能请求相机权限。真实控制仅作用于主屏；录制与校准时自动回到预览。"
        )
        available = not single and not (self.recorder or self.countdown)
        for widget in (
            self.label_choice,
            self.record_seconds,
            self.calibrate_button,
            self.load_profile_button,
            self.reset_profile_button,
        ):
            widget.setEnabled(available)
        self.record_button.setEnabled(not single)
        self._update_task_options()

    def _update_task_options(self, _=None):
        allow_drag = self.interaction_mode == "pinch" or not self.task_method.currentData()
        self.task_choice.model().item(1).setEnabled(allow_drag)
        if not allow_drag:
            self.task_choice.setCurrentIndex(0)

    def _change_interaction(self, _=None):
        self._disable_live()
        self.finish_recording()
        self.practice.stop()
        if self.interaction_mode != "pinch":
            self.finger_engine = SingleFingerEngine(
                self.finger_controls.configuration(self.interaction_mode)
            )
        self.pause_button.setChecked(False)
        self.active_engine.set_enabled(True)
        self.demo_started = time.monotonic()
        self._refresh_interaction_controls()
        self.camera_view.box = self.active_engine.active_box
        self.camera_view.set_frame(None, None, None)
        self._dispatch(self.active_engine.reset())
        self.set_notice(
            "已切换到"
            + self.interaction_choice.currentText()
            + "。单指请先支撑前臂，进入点击实验并开始 8 个目标；其余手指可自然放松。"
            if self.interaction_mode != "pinch"
            else "已恢复捏合主方案。先张开手定位，再捏合点击。"
        )

    def _change_finger_settings(self):
        if self.interaction_mode == "pinch":
            return
        pointer = self.finger_engine.pointer
        self._disable_live()
        self.practice.stop()
        self.finger_engine = SingleFingerEngine(
            self.finger_controls.configuration(self.interaction_mode)
        )
        self.finger_engine.pointer = pointer
        self.finger_engine.set_enabled(not self.pause_button.isChecked())
        self._dispatch(self.finger_engine.reset())
        self.set_notice("单指参数已更新，本轮任务已结束。舒展食指重新定位后再开始一轮。")

    def _recenter_finger(self):
        self._dispatch(self.finger_engine.reset())
        if self.practice.active:
            self.practice.stop()
        self.set_notice("指针留在原位。将手放到舒服的位置，舒展食指保持片刻；随后开始新一轮。")

    @staticmethod
    def _group(title):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(8)
        return box, layout

    @staticmethod
    def _session_name():
        return datetime.now().strftime("%m%d_%H%M") + "_" + uuid.uuid4().hex[:4]

    def set_notice(self, text):
        self.notice.setText(text)

    def _job(self, fn, done):
        job = Job(fn, self)
        self.jobs.add(job)
        job.succeeded.connect(done, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(self._job_failed, Qt.ConnectionType.QueuedConnection)
        job.finished.connect(lambda: self._job_finished(job), Qt.ConnectionType.QueuedConnection)
        job.start()

    def _job_finished(self, job):
        self.jobs.discard(job)
        job.deleteLater()
        self.training_busy = False
        self.download_busy = False
        self.train_button.setEnabled(not (self.recorder or self.countdown))
        self.camera_button.setEnabled(not self.permission_pending)

    def _job_failed(self, message):
        self.set_notice(message)

    def start_camera(self):
        if self.jobs or self.permission_pending:
            self.set_notice("请等待当前下载或训练完成；停止和演示仍可使用。")
            return
        self.stop_all()
        generation = self.source_generation
        if any(t.is_alive() for t in self.stopping_workers):
            self.set_notice("正在释放上一个相机连接，请稍后再次启动")
            return
        self.download_busy = True
        self.camera_button.setEnabled(False)
        self.set_notice("正在校验手部模型；首次运行会下载 Google 官方模型（约 8 MB）…")
        self._job(
            fetch_model,
            lambda _: self._begin_camera() if generation == self.source_generation else None,
        )

    def _begin_camera(self):
        self.permission_pending = True
        self.camera_button.setEnabled(False)
        self.set_notice("正在检查摄像头权限；首次使用请处理系统的相机授权提示。")
        generation = self.source_generation
        try:
            request_camera_access(lambda allowed: self.camera_authorized.emit(generation, allowed))
        except Exception as error:
            self.permission_pending = False
            self.camera_button.setEnabled(True)
            self.set_notice(f"无法请求摄像头权限：{error}")

    def _camera_authorized(self, generation, allowed):
        self.permission_pending = False
        self.camera_button.setEnabled(not self.jobs)
        if generation != self.source_generation:
            return
        if not allowed:
            self.set_notice(
                "摄像头权限未获准。请在系统设置中授权实际启动程序的应用，然后重新启动。"
            )
            return
        self.worker = CameraWorker(default_model_path(), self.camera_index.value())
        self.worker.start()
        self.source = "camera"
        self.last_packet_at = 0.0
        hint = (
            "先张开手进入定位，再尝试捏合。"
            if self.interaction_mode == "pinch"
            else ("先支撑前臂，让手部保持可见；舒展食指片刻定位，再进入应用内点击实验。")
        )
        self.set_notice(camera_permission() + "。" + hint)

    def start_demo(self):
        self.stop_all()
        self.source = "synthetic_demo"
        self.demo_started = time.monotonic()
        self.live_button.setEnabled(False)
        self.set_notice(
            "这是合成动作回放，用于检查界面与事件逻辑。不会控制系统光标，不能用于真实性能评估或数据采集。"
        )

    def _disable_live(self):
        if self.output is not None:
            try:
                self.output.close()
            except Exception as error:
                self.set_notice(f"释放鼠标失败：{error}。请手动松开鼠标并停止程序。")
            self.output = None
        self.live_button.blockSignals(True)
        self.live_button.setChecked(False)
        self.live_button.blockSignals(False)
        self.live_button.setText("启用系统鼠标控制")
        self.chip.setText("预览模式 · 系统光标不受控")
        self.engine.reset()
        self.finger_engine.reset()
        self.practice.cancel_press()
        if self.interaction_mode != "pinch":
            self.live_button.setText("单指 Demo · 仅应用内")
            self.chip.setText("单指 Demo · 应用内练习")

    def _toggle_live(self, checked):
        if not checked:
            self._disable_live()
            return
        if self.interaction_mode != "pinch":
            self._disable_live()
            self.set_notice("单指 Demo 仅用于应用内点击练习，暂不接管系统鼠标。")
            return
        if self.source != "camera" or self.recorder or self.countdown or self.training_busy:
            self._disable_live()
            self.set_notice("需要实时相机，且当前不能在录制、训练或演示状态。")
            return
        if time.monotonic() - self.last_packet_at > 0.25:
            self._disable_live()
            self.set_notice("尚无新鲜相机帧，请确认相机与权限。")
            return
        try:
            self.output = MouseOutput()
            if self.engine.config.reanchor_on_open:
                self.engine.pointer = self.output.position()
            self.engine.reset()
            self.live_button.setText("关闭系统鼠标控制")
            self.chip.setText("系统控制已启用 · Esc 停止")
            self.set_notice("张开手后开始。主屏控制；Esc 停止；手离开画面会结束按下状态。")
        except Exception as e:
            self._disable_live()
            self.set_notice(str(e))

    def _pause(self, checked):
        result = self.active_engine.set_enabled(not checked)
        self._dispatch(result)
        self.pause_button.setText("恢复手势预览" if checked else "暂停手势 · 可用 Esc 停止控制")

    def _emergency(self):
        self._disable_live()
        self.pause_button.setChecked(True)
        self.set_notice("已停止系统控制并暂停识别。恢复后先舒展手指重新定位。")

    def _change_stability(self, enabled):
        self._disable_live()
        self.practice.stop()
        self.engine.config.stabilise = enabled

    def _change_motion(self, _=None):
        self._disable_live()
        self.practice.stop()
        span = self.motion_range.currentData()
        edge = (1.0 - span) / 2
        config = replace(
            self.engine.config,
            box_left=edge,
            box_top=edge,
            box_right=1 - edge,
            box_bottom=1 - edge,
            reanchor_on_open=self.clutch_check.isChecked(),
        )
        pointer = self.engine.pointer
        self.engine = GestureEngine(config)
        self.engine.pointer = pointer
        self.pause_button.setChecked(False)
        self.camera_view.box = self.engine.active_box
        self.camera_view.update()
        hint = (
            "握拳可停住光标并调整手位，张开后从原位置继续。"
            if config.reanchor_on_open
            else "当前使用固定区域映射；恢复时光标会回到手位对应的位置。"
        )
        self.set_notice(f"已切换到 {span:.0%} 的移动范围，并回到预览。{hint}")

    def _change_recognizer(self, index):
        self._disable_live()
        self.practice.stop()
        if index and self.predictor is None:
            self.recognizer.setCurrentIndex(0)
            self.set_notice("尚未加载模型，请先采集并训练。")

    def stop_all(self):
        self.source_generation += 1
        self._disable_live()
        self.finish_recording()
        self.practice.stop()
        if self.worker:
            self.worker.stop()
            self.stopping_workers.append(self.worker)
            self.worker = None
        self.source = "none"
        self.current_frame = None
        self.camera_view.set_frame(None, None, None)
        self.metrics.setText("相机未启动    /    等待张开手")
        self.live_button.setEnabled(self.interaction_mode == "pinch")
        self.pause_button.setChecked(False)
        self.engine.set_enabled(True)
        self.finger_engine.set_enabled(True)
        self._dispatch(self.active_engine.reset())

    def _dispatch(self, result):
        if self.output and (self.interaction_mode != "pinch" or result.mode != "pinch"):
            self._disable_live()
            self.set_notice("单指事件已阻止发送到系统鼠标。请在应用内练习。")
            return
        self.practice.feed(result)
        for event in result.events:
            if event.kind != "move":
                self.event_count += 1
                self.event_file.write(
                    json.dumps(
                        {
                            "timestamp": time.monotonic(),
                            "app_version": __version__,
                            "source": self.source,
                            "output": "os" if self.output else "preview",
                            "recognizer": self.recognizer_name,
                            "interaction_mode": self.interaction_mode,
                            "stabilise": self.engine.config.stabilise
                            if result.mode == "pinch"
                            else True,
                            "config": asdict(self.active_engine.config),
                            "event": asdict(event),
                        }
                    )
                    + "\n"
                )
                self.event_file.flush()
            if self.output:
                try:
                    self.output.emit(event)
                except Exception as e:
                    self._disable_live()
                    self.set_notice(f"控制已停止：{e}")
                    break

    def _tick(self):
        now = time.monotonic()
        self.stopping_workers = [t for t in self.stopping_workers if t.is_alive()]
        if self.output:
            try:
                if self.output.emergency_pressed():
                    self._emergency()
            except Exception as e:
                self._disable_live()
                self.set_notice(f"停止键监测失败，已关闭系统控制：{e}")
        packet = self.worker.pop() if self.worker else None
        if self.worker and self.worker.failure:
            failure = self.worker.failure
            self.stop_all()
            self.set_notice(failure + " " + camera_permission())
        if self.source == "synthetic_demo":
            frame = (
                demo_frame(now, now - self.demo_started)
                if self.interaction_mode == "pinch"
                else finger_demo_frame(now, now - self.demo_started, self.interaction_mode)
            )
            rgb, inference, fps = None, 0.0, 50.0
        elif packet:
            if now - packet.captured_at > 0.25:
                self._dispatch(self.active_engine.tick(now))
                self.set_notice("相机结果过期，已丢弃；不会重放积压鼠标动作。")
                return
            frame, rgb, inference, fps = packet.frame, packet.rgb, packet.inference_ms, packet.fps
            self.last_packet_at = now
        else:
            self._dispatch(self.active_engine.tick(now))
            self._record_tick(now)
            return
        self.current_frame = frame
        features = extract(frame) if self.interaction_mode == "pinch" else None
        prediction = None
        if self.recognizer.currentIndex() and self.predictor and features:
            try:
                prediction = self.predictor.predict(features)
            except Exception as e:
                self._disable_live()
                self.recognizer.setCurrentIndex(0)
                self.set_notice(f"模型推理失败，已停止控制：{e}")
                return
        result = (
            self.engine.process(frame, prediction)
            if self.interaction_mode == "pinch"
            else self.finger_engine.process(frame)
        )
        self._dispatch(result)
        self.camera_view.box = self.active_engine.active_box
        self.camera_view.set_frame(rgb, frame, result, self.source == "synthetic_demo")
        ratio = f"{result.pinch:.3f}" if result.pinch is not None else "—"
        predicted = result.prediction.label if result.prediction else "未检测到手"
        detail = f"{result.state}    ·    {predicted}    ·    捏合比 {ratio}"
        if self.interaction_mode != "pinch":
            bend = f"{result.bend:.2f}" if result.bend is not None else "—"
            detail = f"{result.hint}\n{result.state}    ·    屈曲量（几何代理） {bend}"
        self.metrics.setText(
            detail
            + f"\n{fps:.1f} FPS    /    手部推理 {inference:.1f} ms    /    事件 {self.event_count}"
        )
        self._record_tick(now)
        if self.recorder and self.source == "camera":
            try:
                self.recorder.add(frame)
            except Exception as e:
                self.finish_recording()
                self.set_notice(f"录制已停止：{e}")

    def _record_controls(self, enabled):
        for widget in (
            self.participant,
            self.session,
            self.label_choice,
            self.record_seconds,
            self.new_session,
            self.calibrate_button,
            self.load_profile_button,
            self.reset_profile_button,
            self.motion_range,
            self.clutch_check,
            self.interaction_choice,
            self.finger_controls,
        ):
            widget.setEnabled(enabled)
        self.train_button.setEnabled(enabled and not self.training_busy)
        self._refresh_interaction_controls()

    def start_recording(self):
        if self.interaction_mode != "pinch":
            self.set_notice("单指 Demo 的动作不能录入捏合四分类数据。点击任务结果会单独保存。")
            return
        if self.training_busy:
            self.set_notice("请先等待训练结束，避免录制文件在训练读取时发生变化。")
            return
        if self.recorder or self.countdown:
            self.finish_recording()
            return
        if self.source != "camera" or self.current_frame is None:
            self.set_notice("先启动真实摄像头。合成演示不能录成训练数据。")
            return
        try:
            safe_name(self.participant.text())
            safe_name(self.session.text())
        except ValueError as e:
            self.set_notice(str(e))
            return
        self._disable_live()
        self.practice.stop()
        self.countdown = time.monotonic() + 3
        self._record_controls(False)
        self.record_button.setText("取消录制")

    def _record_tick(self, now):
        if self.countdown:
            remaining = self.countdown - now
            if remaining > 0:
                self.set_notice(
                    f"{int(remaining) + 1} 秒后录制「{self.label_choice.currentText()}」。请提前摆好手形并保持。"
                )
            else:
                try:
                    self.recorder = Recorder(
                        self.record_dir,
                        self.participant.text(),
                        self.session.text(),
                        self.label_choice.currentData(),
                    )
                    self.record_until = now + self.record_seconds.value()
                    self.countdown = None
                    self.record_button.setText("结束本段录制")
                except Exception as e:
                    self.finish_recording()
                    self.set_notice(str(e))
        if self.recorder:
            if now >= self.record_until:
                self.finish_recording()
            else:
                self.set_notice(
                    f"录制 {self.recorder.label} · 还剩 {self.record_until - now:.1f} 秒 · 有效帧 {self.recorder.valid_count}"
                )

    def finish_recording(self):
        if self.recorder:
            self.recorder.close()
            self.set_notice(
                f"已保存 {self.recorder.path.name} · {self.recorder.valid_count}/{self.recorder.count} 帧有可用手部；人工标签请与实际动作一致。"
            )
            self.recorder = None
        self.countdown = self.record_until = None
        if hasattr(self, "record_button"):
            self.record_button.setText("倒计时 3 秒后录制")
            self._record_controls(True)

    def run_calibration(self):
        self._disable_live()
        self.practice.stop()
        try:
            config, report = calibrate(self.record_dir, self.participant.text(), self.engine.config)
            self.engine = GestureEngine(config)
            self.pause_button.setChecked(False)
            save_json(
                self.workspace / "data" / "profiles" / f"{safe_name(self.participant.text())}.json",
                report,
            )
            self.calibration_status.setText(
                f"{self.participant.text()} · 捏合 {config.engage_ratio:.3f} / 释放 {config.release_ratio:.3f}"
            )
            self.set_notice(
                "个人阈值已应用并保存。它影响规则识别；不是 ML 微调，也不代表测试集效果。"
            )
        except Exception as e:
            self.set_notice(str(e))

    def load_profile(self):
        self._disable_live()
        self.practice.stop()
        try:
            path = (
                self.workspace / "data" / "profiles" / f"{safe_name(self.participant.text())}.json"
            )
            profile = json.loads(path.read_text(encoding="utf-8"))
            saved = EngineConfig(**profile["config"])
            saved.validate()
            config = replace(
                self.engine.config,
                engage_ratio=saved.engage_ratio,
                release_ratio=saved.release_ratio,
            )
            self.engine = GestureEngine(config)
            self.pause_button.setChecked(False)
            self.calibration_status.setText(
                f"{profile['participant']} · 捏合 {config.engage_ratio:.3f} / 释放 {config.release_ratio:.3f}"
            )
            self.set_notice("已加载个人校准。正式评测请用与校准不同的场次。")
        except Exception as e:
            self.set_notice(f"未加载校准：{e}")

    def reset_profile(self):
        self._disable_live()
        self.practice.stop()
        defaults = EngineConfig()
        self.engine = GestureEngine(
            replace(
                self.engine.config,
                engage_ratio=defaults.engage_ratio,
                release_ratio=defaults.release_ratio,
            )
        )
        self.pause_button.setChecked(False)
        self.calibration_status.setText("默认阈值 · 捏合 0.26 / 释放 0.40")
        self.set_notice("已恢复默认规则阈值，可与个人校准做对照。")

    def run_training(self):
        from .learning import train

        if self.jobs or self.recorder or self.countdown:
            self.set_notice("请先结束录制或等待当前后台任务完成。")
            return
        self._disable_live()
        self.practice.stop()
        self.training_busy = True
        self.train_button.setEnabled(False)
        output = self.report_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        group, model = self.group_choice.currentData(), self.model_choice.currentData()
        self.set_notice("正在按独立分组训练和评估；界面与停止操作仍可响应。")

        def done(report):
            self._load_model_path(output / "model.joblib")
            self.model_status.setText(
                f"留出测试 F1 {report['learned']['macro_f1']:.3f} · 规则 {report['rules']['macro_f1']:.3f}\n报告：{output.name}"
            )
            self.set_notice(f"训练完成。按 {group} 留出，尚不代表在线误触或舒适度。报告：{output}")

        self._job(lambda: train(self.record_dir, output, group, model), done)

    def load_model(self):
        self._disable_live()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "仅加载你信任的、本项目训练的模型",
            str(self.report_dir),
            "PinchPilot 模型 (*.joblib)",
        )
        if filename:
            self._load_model_path(Path(filename))

    def _load_model_path(self, path):
        from .learning import LearnedPredictor

        try:
            self.predictor = LearnedPredictor(path)
            self.recognizer.setCurrentIndex(1)
            self.model_status.setText(
                f"已加载 {self.predictor.report['model']}\n{path.parent.name}"
            )
        except Exception as e:
            self.set_notice(f"模型加载失败：{e}")

    def start_practice(self):
        self._disable_live()
        virtual = self.task_method.currentData()
        if virtual and self.source == "none":
            self.set_notice("手势任务需要先启动相机或演示。")
            return
        if (
            virtual
            and self.interaction_mode != "pinch"
            and self.task_choice.currentData() != "click"
        ):
            self.set_notice("单指 Demo 当前只支持点击目标。")
            return
        if virtual:
            self.active_engine.pointer = (0.5, 0.5)
            self.active_engine.reset()
        self.practice.start(
            self.report_dir / "tasks",
            {
                "participant": self.participant.text(),
                "app_version": __version__,
                "session": self.session.text(),
                "source": self.source if virtual else "physical_mouse_or_trackpad",
                "recognizer": self.recognizer_name,
                "interaction_mode": self.interaction_mode if virtual else "physical",
                "config": asdict(self.active_engine.config),
                "initial_pointer": self.active_engine.pointer if virtual else None,
            },
            virtual,
            self.task_choice.currentData(),
        )

    def open_data(self):
        from PySide6.QtCore import QUrl

        (self.workspace / "data").mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.workspace / "data")))

    def closeEvent(self, event):
        if self.jobs:
            self._disable_live()
            self.set_notice("后台任务即将完成，请稍后关闭；系统控制已停止。")
            event.ignore()
            return
        self.stop_all()
        self.timer.stop()
        for worker in self.stopping_workers:
            worker.join(timeout=2)
        self.event_file.close()
        event.accept()


def run_gui(
    workspace: Path,
    demo=False,
    smoke_seconds: float | None = None,
    screenshot: Path | None = None,
    interaction: str = "pinch",
):
    enable_dpi_awareness()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("PinchPilot")
    app.setOrganizationName("PinchPilot Research")
    window = MainWindow(workspace, demo, interaction)
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1240, available.width() - 40), min(820, available.height() - 40))
    window.show()
    if screenshot:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        QTimer.singleShot(1500, lambda: window.grab().save(str(screenshot)))
    if smoke_seconds:
        QTimer.singleShot(int(smoke_seconds * 1000), window.close)
    return app.exec()
