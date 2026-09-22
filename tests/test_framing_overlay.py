import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QRect, Qt
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from pinchpilot import framing_overlay


@pytest.fixture
def overlay():
    app = QApplication.instance() or QApplication([])
    widget = framing_overlay.FramingOverlay()
    yield widget
    widget.dispose()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_reminder_requests_no_focus_and_native_input_transparency(overlay):
    flags = overlay.windowFlags()
    for flag in (
        Qt.WindowType.WindowTransparentForInput,
        Qt.WindowType.WindowDoesNotAcceptFocus,
        Qt.WindowType.WindowStaysOnTopHint,
        Qt.WindowType.FramelessWindowHint,
    ):
        assert flags & flag
    assert overlay.parentWidget() is None
    assert overlay.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not overlay.testAttribute(Qt.WidgetAttribute.WA_QuitOnClose)
    assert overlay.hint.textFormat() == Qt.TextFormat.PlainText


@pytest.mark.parametrize("available", [QRect(0, 25, 1512, 920), QRect(-800, 40, 360, 280)])
def test_reminder_stays_inside_available_screen_and_wraps(overlay, monkeypatch, available):
    monkeypatch.setattr(
        framing_overlay.QApplication,
        "primaryScreen",
        lambda: SimpleNamespace(availableGeometry=lambda: available),
    )
    message = "手靠近画面右侧、下沿 · 保持食拇捏合，松中指换位，再捏中指继续拖。"
    overlay.set_hint(message, True)
    QApplication.processEvents()
    assert overlay.isVisible() and available.contains(overlay.geometry())
    assert overlay.hint.text() == message
    assert overlay.hint.height() >= overlay.hint.heightForWidth(overlay.hint.width())
    overlay.set_hint(message, False)
    assert not overlay.isVisible()


def test_disposed_reminder_cannot_reappear_and_is_destroyed(overlay):
    overlay.set_hint("请把手移回画面", True)
    assert overlay.isVisible()
    overlay.dispose()
    assert overlay.disposed and not overlay.isVisible()
    overlay.set_hint("late update", True)
    assert not overlay.isVisible()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(overlay)


def test_reminder_paints_opaque_backplate_for_readability(overlay):
    overlay.set_hint("手靠近画面下沿 · 松中指，将手移回画面中央，再捏住继续。", True)
    QApplication.processEvents()
    pixmap = overlay.grab()
    image = pixmap.toImage()
    scale = pixmap.devicePixelRatio()
    background = image.pixelColor(int(6 * scale), int(overlay.height() / 2 * scale))
    assert background.alpha() == 255
    assert background.name() == "#1b2938"
    assert image.pixelColor(0, 0).alpha() == 0  # Rounded corners stay transparent.
