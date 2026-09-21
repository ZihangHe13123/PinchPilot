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


class TripodControls(QWidget):
    changed = Signal()
    recenter = Signal()
    probe = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
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
        hint = QLabel(
            "拇中定位；食拇短捏松开是左键，保持到进度环满可拖，松食指放下。"
            "食指移开，拇指＋无名指轻捏是右键。松中指锁住位置，捏回可继续；Esc 暂停。"
            "V 手势、拇指分开，上下轻移滚动；收指停止。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        for choice in (
            self.stability,
            self.contact,
            self.right_contact,
            self.drag_hold,
            self.scroll_speed,
        ):
            choice.currentIndexChanged.connect(lambda _: self.changed.emit())

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

    def configuration(self):
        touch = self.contact.currentData()
        right_touch = self.right_contact.currentData()
        return TripodConfig(
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
