"""Small non-interactive reminder, independent of the main window's visibility."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class FramingOverlay(QWidget):
    def __init__(self):
        # No QWidget parent: minimizing the settings window must not hide this tool.
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.disposed = False
        self.setObjectName("framingOverlay")
        self.setWindowTitle("PinchPilot · 手部位置提醒")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(
            "QWidget#framingOverlay { background: #1b2938; border: 1px solid #ad8651;"
            "border-radius: 9px; }"
            "QLabel { color: #edf3f7; background: transparent; border: none; font-size: 14px; }"
            "QLabel#framingTitle { color: #efc385; font-size: 12px; font-weight: 600; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 11, 15, 12)
        layout.setSpacing(5)
        title = QLabel("PinchPilot · 手部位置提醒")
        title.setObjectName("framingTitle")
        layout.addWidget(title)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setTextFormat(Qt.TextFormat.PlainText)
        self.hint.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.hint.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self.hint)
        QApplication.instance().aboutToQuit.connect(self.dispose)

    def paintEvent(self, event):
        # Translucent top-level widgets do not reliably paint stylesheet backgrounds.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#1b2938"))
        painter.setPen(QPen(QColor("#ad8651"), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 9, 9)

    def set_hint(self, text, visible):
        if self.disposed:
            return
        if not visible or not text:
            self.hide()
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            self.hide()
            return
        self.hint.setText(text)
        available = screen.availableGeometry()
        margin = min(16, available.width() // 8, available.height() // 8)
        self.setFixedWidth(min(440, max(1, available.width() - 2 * margin)))
        self.adjustSize()
        self.resize(self.width(), min(self.height(), available.height() - 2 * margin))
        self.move(
            available.left() + (available.width() - self.width()) // 2,
            available.top() + margin,
        )
        if not self.isVisible():
            self.show()

    def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        QApplication.instance().aboutToQuit.disconnect(self.dispose)
        self.close()
        self.deleteLater()
