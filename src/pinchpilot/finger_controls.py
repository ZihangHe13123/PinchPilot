from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .single_finger import FingerConfig


class FingerControls(QWidget):
    changed = Signal()
    recenter = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.gain = QComboBox()
        for title, value in [("精细 · 1.5", 1.5), ("均衡 · 2.5", 2.5), ("更省位移 · 3.5", 3.5)]:
            self.gain.addItem(title, value)
        self.gain.setCurrentIndex(1)
        self.flex = QComboBox()
        for title, value in [("较轻 · 容易误触", 0.08), ("默认轻弯", 0.12), ("更明显的弯曲", 0.18)]:
            self.flex.addItem(title, value)
        self.flex.setCurrentIndex(1)
        self.dwell = QDoubleSpinBox()
        self.dwell.setRange(0.4, 2.0)
        self.dwell.setSingleStep(0.1)
        self.dwell.setValue(0.8)
        self.dwell.setSuffix(" 秒")
        self.flex_label = QLabel("点击幅度")
        self.dwell_label = QLabel("停留时间")
        form.addRow("移动灵敏度", self.gain)
        form.addRow(self.flex_label, self.flex)
        form.addRow(self.dwell_label, self.dwell)
        layout.addLayout(form)
        reset = QPushButton("重新定位 · 保留当前指针位置")
        reset.clicked.connect(self.recenter.emit)
        layout.addWidget(reset)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setObjectName("subtitle")
        layout.addWidget(self.instructions)
        self.gain.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.flex.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.dwell.valueChanged.connect(lambda _: self.changed.emit())

    def select_mode(self, mode):
        flex = mode == "finger-flex"
        self.flex.setVisible(flex)
        self.flex_label.setVisible(flex)
        self.dwell.setVisible(not flex)
        self.dwell_label.setVisible(not flex)
        click = (
            "定位后轻弯食指，再恢复，完成点击；弯曲期间指针冻结。"
            if flex
            else "先移动食指，再停住等进度环完成；点击后移开才能再次点击。"
        )
        self.instructions.setText(
            "先支撑前臂，让相机看清手部。舒展食指保持片刻，再小幅移动。"
            + click
            + "深弯或移出画面可休息；Esc 暂停。仅应用内点击练习。"
        )

    def configuration(self, mode):
        return FingerConfig(
            mode=mode,
            gain=self.gain.currentData(),
            flex_delta=self.flex.currentData(),
            dwell_seconds=self.dwell.value(),
        )
