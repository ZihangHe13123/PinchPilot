from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

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
        self.span = QComboBox()
        for label, value in (
            ("30% · 默认", 0.30),
            ("40% · 更易精调", 0.40),
            ("20% · 更省位移", 0.20),
        ):
            self.span.addItem(label, value)
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
        form.addRow("定位范围", self.span)
        form.addRow("抗抖强度", self.stability)
        form.addRow("食拇接触", self.contact)
        form.addRow("右键触碰", self.right_contact)
        form.addRow("拖拽等待", self.drag_hold)
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
        )
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        for choice in (self.span, self.stability, self.contact, self.right_contact, self.drag_hold):
            choice.currentIndexChanged.connect(lambda _: self.changed.emit())

    def configuration(self):
        touch = self.contact.currentData()
        right_touch = self.right_contact.currentData()
        return TripodConfig(
            span=self.span.currentData(),
            deadband=self.stability.currentData(),
            touch_ratio=touch,
            hover_ratio=touch + 0.12,
            clear_ratio=touch + 0.22,
            right_touch_ratio=right_touch,
            right_hover_ratio=right_touch + 0.12,
            right_clear_ratio=right_touch + 0.22,
            drag_hold_seconds=self.drag_hold.currentData(),
        )
