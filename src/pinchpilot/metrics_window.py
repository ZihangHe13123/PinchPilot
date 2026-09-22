"""Read-only timing/diagnostic view, with no camera or native-input ownership."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .trial_metrics import TIMING_LABELS


class MetricsWindow(QDialog):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setStyleSheet("""
            QDialog { background: #0b1420; color: #dce7f0; }
            QLabel { color: #dce7f0; }
            QTableWidget { background: #101f2e; color: #c9dbe8; gridline-color: #314a60; }
            QHeaderView::section { background: #243b50; color: #dce7f0; padding: 5px; border: none; }
            QTableCornerButton::section { background: #243b50; border: none; }
        """)
        self.setWindowTitle("PinchPilot · 诊断与耗时")
        self.resize(700, 470)
        layout = QVBoxLayout(self)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        layout.addWidget(self.reason)
        self.table = QTableWidget(len(TIMING_LABELS), 4)
        self.table.setHorizontalHeaderLabels(["阶段", "最近 / ms", "P95 / ms", "样本数"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for row, label in enumerate(TIMING_LABELS.values()):
            self.table.setItem(row, 0, QTableWidgetItem(label))
        layout.addWidget(self.table)
        self.counts = QLabel()
        self.counts.setWordWrap(True)
        layout.addWidget(self.counts)
        scope = QLabel(
            "各阶段取最近最多 180 个样本；画面来源切换后重新统计。\n"
            "read 耗时包含等待摄像头返回；这里没有测量传感器曝光、系统实际响应或显示器刷新。"
            "各阶段 P95 不能相加作为端到端延迟。\n"
            "原因来自当前可观测状态，暂停可能是正常点击/松指；它不等于识别错误。"
        )
        scope.setWordWrap(True)
        layout.addWidget(scope)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        data = self.controller.metrics.snapshot(self.controller.clock())
        diagnostic = data["diagnostic"]
        source = {"camera": "相机", "none": "相机关闭", "synthetic_demo": "合成演示"}[
            self.controller.source
        ]
        self.reason.setText(
            f"{source} · {diagnostic['label']} · 持续 {data['reason_duration_s']:.1f} 秒\n"
            f"原因代码：{diagnostic['reason']} · 状态：{diagnostic['state']}"
        )
        for row, name in enumerate(TIMING_LABELS):
            values = data["timings"][name]
            for col, key in enumerate(("last_ms", "p95_ms", "samples"), 1):
                value = values[key]
                label = (
                    "—" if value is None else (str(value) if key == "samples" else f"{value:.2f}")
                )
                self.table.setItem(row, col, QTableWidgetItem(label))
        self.counts.setText(
            f"本次运行累计：最新结果槽覆盖 {data['overwritten_results']} 次 · "
            f"无效/过期帧丢弃 {data['discarded_frames']} 次 · 跟踪中断 {data['tracking_losses']} 次\n"
            "结果槽覆盖表示界面取走前被较新推理结果替换，不代表摄像头丢帧。"
        )
