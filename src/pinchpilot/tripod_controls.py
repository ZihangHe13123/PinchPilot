import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .tripod import TripodConfig
from .wrist import validate_wrist_calibration


class TripodControls(QWidget):
    changed = Signal()
    recenter = Signal()
    probe = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.motion_form = form
        self.pointer_basis = QComboBox()
        self.pointer_basis.addItem("兼容定位 · 平移＋转腕", "unified")
        self.pointer_basis.addItem("指尖位置 · 原版对照", "position")
        self.pointer_basis.addItem("腕动角度 · 旧实验", "wrist")
        # Other demo windows retain their established position-mode default.
        self.pointer_basis.setCurrentIndex(self.pointer_basis.findData("position"))
        self.pointer_basis.setToolTip(
            "兼容定位沿用画面方向，平移和自然转腕均可移动，无需方向校准。原位置和角度模式可选作对照。"
        )
        self.wrist_calibration = ()
        self.motion_profile = QComboBox()
        for label, value in (
            ("原版 · 对照", "classic"),
            ("精细 · 小幅微调", "precise"),
            ("自适应 · 实验", "adaptive"),
        ):
            self.motion_profile.addItem(label, value)
        self.motion_profile.setToolTip(
            "原版保留已有移动手感；精细模式改善慢速微调；自适应模式随移动速度调整灵敏度。"
        )
        self.fixed_motion_profile = QLabel("固定精细（灵敏度可调）")
        self.basis_summary = QLabel("无需方向校准 · 整手平移与自然转腕连续控制")
        self.basis_summary.setWordWrap(True)
        self.basis_summary.setObjectName("subtitle")
        self.rest_noise_x = 0.0
        self.rest_noise_y = 0.0
        speed = QWidget()
        speed_layout = QVBoxLayout(speed)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        value_row = QHBoxLayout()
        self.sensitivity_value = QLabel("1.00×")
        value_row.addWidget(self.sensitivity_value)
        value_row.addStretch()
        self.reset_sensitivity = QPushButton("恢复默认")
        self.reset_sensitivity.setToolTip("恢复原默认速度 1.00×")
        value_row.addWidget(self.reset_sensitivity)
        speed_layout.addLayout(value_row)
        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(10, 200)
        self.sensitivity.setValue(100)
        self.sensitivity.setSingleStep(1)
        self.sensitivity.setPageStep(10)
        # Apply once on release, so changing speed cannot end its own gesture drag midway.
        self.sensitivity.setTracking(False)
        self.sensitivity.setToolTip("向左更慢；1.00× 为原默认，0.75× 为原最慢档")
        speed_layout.addWidget(self.sensitivity)
        limits = QLabel("慢 0.10×  ← →  快 2.00×")
        limits.setObjectName("subtitle")
        speed_layout.addWidget(limits)
        self.sensitivity.sliderMoved.connect(self._show_sensitivity)
        self.sensitivity.valueChanged.connect(self._sensitivity_changed)
        self.reset_sensitivity.clicked.connect(lambda: self.sensitivity.setValue(100))
        self.stability = QComboBox()
        for label, value in (
            ("轻 · 小动作更灵敏", 0.003),
            ("标准", 0.008),
            ("强 · 小动作会被抑制", 0.014),
        ):
            self.stability.addItem(label, value)
        self.stability.setCurrentIndex(1)
        self.contact = QComboBox()
        for label, value in (
            ("默认触碰", 0.26),
            ("宽松 · 提前判定", 0.34),
            ("严格 · 需更靠近", 0.20),
        ):
            self.contact.addItem(label, value)
        self.right_contact = QComboBox()
        for label, value in (
            ("默认触碰", 0.26),
            ("宽松 · 提前判定", 0.34),
            ("严格 · 需更靠近", 0.20),
        ):
            self.right_contact.addItem(label, value)
        self.drag_hold = QComboBox()
        for label, value in (
            ("0.30 秒 · 默认", 0.30),
            ("0.20 秒 · 较快", 0.20),
            ("0.50 秒 · 从容", 0.50),
        ):
            self.drag_hold.addItem(label, value)
        self.scroll_speed = QComboBox()
        for label, value in (("慢 · 精细翻动", 30.0), ("标准", 60.0), ("快 · 长页面", 120.0)):
            self.scroll_speed.addItem(label, value)
        self.scroll_speed.setCurrentIndex(1)
        form.addRow("定位方式", self.pointer_basis)
        form.addRow("移动模式", self.motion_profile)
        form.addRow("移动模式", self.fixed_motion_profile)
        form.addRow(self.basis_summary)
        form.addRow("移动灵敏度", speed)
        form.addRow("抗抖强度", self.stability)
        form.addRow("食拇接触", self.contact)
        form.addRow("右键触碰", self.right_contact)
        form.addRow("拖拽等待", self.drag_hold)
        form.addRow("滚动速度", self.scroll_speed)
        layout.addLayout(form)
        reset = QPushButton("重新定位 · 保留指针")
        reset.clicked.connect(self.recenter.emit)
        layout.addWidget(reset)
        self.probe_button = QPushButton("记录 8 秒静止抖动")
        self.probe_button.clicked.connect(self.probe.emit)
        layout.addWidget(self.probe_button)
        self.interaction_hint = QLabel(
            "拇中定位；食拇短捏松开是左键，保持到进度环满可拖，松食指放下。"
            "食指移开，拇指＋无名指轻捏是右键。松中指锁住位置，捏回可继续；Esc 暂停。"
            "V 手势、拇指分开，上下轻移滚动；收指停止。"
        )
        self.interaction_hint.setWordWrap(True)
        self.interaction_hint.setObjectName("subtitle")
        self._position_hint = self.interaction_hint.text()
        layout.addWidget(self.interaction_hint)
        self.pointer_basis.currentIndexChanged.connect(self._refresh_basis_hint)
        for choice in (
            self.pointer_basis,
            self.motion_profile,
            self.stability,
            self.contact,
            self.right_contact,
            self.drag_hold,
            self.scroll_speed,
        ):
            choice.currentIndexChanged.connect(lambda _: self.changed.emit())
        self._refresh_basis_hint(self.pointer_basis.currentIndex())

    def _refresh_basis_hint(self, _):
        basis = self.pointer_basis.currentData()
        wrist = basis == "wrist"
        self.motion_profile.setEnabled(basis == "position")
        self.motion_form.setRowVisible(self.motion_profile, basis == "position")
        self.motion_form.setRowVisible(self.fixed_motion_profile, basis != "position")
        self.motion_form.setRowVisible(self.basis_summary, basis == "unified")
        if wrist:
            self.motion_profile.setToolTip(
                "腕动采用固定增益精细滤波；下方灵敏度仍可调。切回位置模式恢复原选择。"
            )
            self.interaction_hint.setText(
                "先完成腕动方向校准。拇中捏住后，手腕转动一段、光标移动一段；"
                "停腕即停，松中指可回到舒服姿势再接管。"
                "腕动采用固定增益精细滤波，灵敏度仍可调。"
                "食拇点击/保持拖拽、拇无名指右键与 V 手势滚动不变；不会根据偏转持续移动。"
            )
        elif basis == "unified":
            self.motion_profile.setToolTip(
                "兼容定位采用固定增益精细滤波；灵敏度仍可调。切回位置对照后恢复原移动模式选择。"
            )
            self.interaction_hint.setText(
                "无需方向校准。拇中捏住后，整手平移和自然转腕可连续移动光标，方向沿用画面位置；"
                "手停光标就停，松中指可换位再接管。采用固定增益精细滤波，灵敏度仍可调。"
                "食拇点击/保持拖拽、拇无名指右键与 V 手势滚动不变。"
            )
        else:
            self.motion_profile.setToolTip(
                "原版保留已有移动手感；精细模式改善慢速微调；自适应模式随移动速度调整灵敏度。"
            )
            self.interaction_hint.setText(self._position_hint)

    def _show_sensitivity(self, value):
        self.sensitivity_value.setText(f"{value / 100:.2f}×")

    def _sensitivity_changed(self, value):
        self._show_sensitivity(value)
        self.changed.emit()

    def restore_span(self, span):
        """Read both legacy three-position settings and new continuous settings."""
        if (
            isinstance(span, bool)
            or not isinstance(span, (int, float))
            or not math.isfinite(span)
            or not 0.15 <= span <= 3.0
        ):
            return
        self.sensitivity.setValue(round(30 / span))

    def restore_noise(self, x, y):
        """Restore valid calibration without emitting a settings change."""
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 0.02
            for value in (x, y)
        ):
            return False
        self.rest_noise_x, self.rest_noise_y = float(x), float(y)
        return True

    def restore_wrist_calibration(self, values):
        """Restore only a validated empty/15-number calibration, without signals."""
        if not isinstance(values, (tuple, list)):
            return False
        try:
            candidate = tuple(values)
            validate_wrist_calibration(candidate)
        except (ValueError, TypeError):
            return False
        self.wrist_calibration = tuple(float(value) for value in candidate)
        return True

    def configuration(self):
        touch = self.contact.currentData()
        right_touch = self.right_contact.currentData()
        basis = self.pointer_basis.currentData()
        return TripodConfig(
            pointer_basis=basis,
            wrist_calibration=self.wrist_calibration,
            motion_profile="precise" if basis == "unified" else self.motion_profile.currentData(),
            rest_noise_x=self.rest_noise_x if basis == "position" else 0.0,
            rest_noise_y=self.rest_noise_y if basis == "position" else 0.0,
            span=30 / self.sensitivity.value(),
            deadband=self.stability.currentData(),
            touch_ratio=touch,
            hover_ratio=touch + 0.12,
            clear_ratio=touch + 0.22,
            right_touch_ratio=right_touch,
            right_hover_ratio=right_touch + 0.12,
            right_clear_ratio=right_touch + 0.22,
            drag_hold_seconds=self.drag_hold.currentData(),
            scroll_gain=self.scroll_speed.currentData(),
        )
