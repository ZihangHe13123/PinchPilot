"""Offscreen vector instruction checks; no camera, tracker, or native input."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from pinchpilot import wrist_guide


@pytest.fixture
def guide():
    app = QApplication.instance() or QApplication([])
    value = wrist_guide.WristPoseGuide()
    value.resize(300, 170)
    yield value
    value.stop()
    value.close()
    value.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_animation_is_visible_only_and_repeated_stage_does_not_restart(guide, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(wrist_guide.time, "monotonic", lambda: now[0])
    guide.set_stage(1)
    assert not guide.timer.isActive()
    guide.show()
    assert guide.timer.isActive()
    now[0] = 102.0
    guide.set_stage(1)
    assert guide._started == 100.0
    guide.hide()
    assert not guide.timer.isActive()
    guide.show()
    assert guide.timer.isActive() and guide._started == 102.0
    guide.stop()
    guide.hide()
    guide.show()
    assert not guide.timer.isActive()
    guide.set_stage(2)
    assert guide.timer.isActive()
    guide.set_stage(3)
    assert not guide.timer.isActive()


@pytest.mark.parametrize("size", [(300, 170), (280, 140)])
def test_all_stages_render_and_only_hand_moves(guide, monkeypatch, size):
    now = [100.0]
    monkeypatch.setattr(wrist_guide.time, "monotonic", lambda: now[0])
    guide.resize(*size)
    guide.show()
    images = []
    for stage in range(4):
        guide.set_stage(stage)
        now[0] += 2.5  # Target pose held still in both animated examples.
        QApplication.processEvents()
        image = guide.grab().toImage()
        assert not image.isNull()
        assert image.pixelColor(image.width() // 2, 5).alpha() == 255
        assert "不是当前姿态" in guide.accessibleDescription()
        images.append(image)
    assert images[0] != images[1] and images[1] != images[2]
    # In side view the forearm and its support do not translate or lift.
    guide.set_stage(2)
    neutral = guide.grab().toImage()
    now[0] += 2.5
    turned = guide.grab().toImage()
    scale = min(guide.width() / 300, guide.height() / 170)
    ratio = guide.devicePixelRatioF()
    x = int(((guide.width() - 300 * scale) / 2 + 40 * scale) * ratio)
    y = int(((guide.height() - 170 * scale) / 2 + 105 * scale) * ratio)
    width, height = int(60 * scale * ratio), int(33 * scale * ratio)
    assert neutral.copy(x, y, width, height) == turned.copy(x, y, width, height)
    assert neutral != turned


def test_motion_holds_at_target_and_stage_validation(guide):
    assert guide._motion(0.5) == 0
    assert guide._motion(2) == guide._motion(3.5) == 1
    assert guide._motion(4.7) == 0
    for stage in (-1, 4, True, "1"):
        with pytest.raises(ValueError):
            guide.set_stage(stage)


def test_sampling_holds_target_and_neutral_return_takes_priority(guide, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(wrist_guide.time, "monotonic", lambda: now[0])
    guide.show()
    guide.set_stage(2)
    assert guide.timer.isActive()
    guide.set_collecting(True)
    target = guide.grab().toImage()
    assert not guide.timer.isActive()
    now[0] += 20
    assert target == guide.grab().toImage()
    guide.set_neutral_return(True)
    neutral = guide.grab().toImage()
    assert neutral != target and not guide.timer.isActive()
    assert "等待提示后再抬掌" in guide.accessibleDescription()
    guide.set_neutral_return(False)
    assert target == guide.grab().toImage()
    guide.set_collecting(False)
    assert guide.timer.isActive()
