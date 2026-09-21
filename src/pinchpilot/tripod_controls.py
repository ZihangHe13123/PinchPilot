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
        form.addRow("定位范围", self.span)
        form.addRow("抗抖强度", self.stability)
        form.addRow("食拇接触", self.contact)
        layout.addLayout(form)
        reset = QPushButton("重新定位 · 保留指针")
        reset.clicked.connect(self.recenter.emit)
        layout.addWidget(reset)
        self.probe_button = QPushButton("记录 8 秒静止抖动")
        self.probe_button.clicked.connect(self.probe.emit)
        layout.addWidget(self.probe_button)
        hint = QLabel(
            "拇中捏住后移动；食指碰拇指点一次，分开再点。松中指锁住位置，仍可点击。"
            "先移开食指，再捏中指继续移动；Esc 暂停。前臂尽量有支撑。静止记录前留 2 秒准备。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        for choice in (self.span, self.stability, self.contact):
            choice.currentIndexChanged.connect(lambda _: self.changed.emit())

    def configuration(self):
        touch = self.contact.currentData()
        return TripodConfig(
            span=self.span.currentData(),
            deadband=self.stability.currentData(),
            touch_ratio=touch,
            hover_ratio=touch + 0.12,
            clear_ratio=touch + 0.22,
        )
