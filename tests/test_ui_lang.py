"""The English interface: the language switch, the catalogue, and windows with no Chinese left on screen."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pinchpilot import cli, contact_protocol, practice, ui_lang

TESTS = Path(__file__).resolve().parent


@pytest.fixture
def english(monkeypatch):
    monkeypatch.setattr(ui_lang, "LANG", "en")
    ui_lang._translate.cache_clear()
    ui_lang.MISSING.clear()
    yield
    ui_lang._translate.cache_clear()
    ui_lang.MISSING.clear()


def test_chinese_stays_the_default_and_untouched(monkeypatch):
    monkeypatch.delenv("PINCHPILOT_LANG", raising=False)
    assert ui_lang.resolve() == "zh"
    assert ui_lang.resolve("en") == "en"
    monkeypatch.setenv("PINCHPILOT_LANG", "en")
    assert ui_lang.resolve() == "en"
    assert ui_lang.resolve("zh") == "zh"
    assert ui_lang.LANG == "zh"
    assert ui_lang.tr("启动相机") == "启动相机"


def test_the_window_commands_take_a_language():
    for argv in (
        ["desktop", "--lang", "en"],
        ["gui", "--lang", "en"],
        ["ml-review", "x.zip", "--lang", "en"],
    ):
        assert cli.parser().parse_args(argv).lang == "en"
    assert cli.parser().parse_args(["desktop"]).lang is None


def test_fixed_and_composed_texts_translate(english):
    tr = ui_lang.tr
    assert tr("启动相机") == "Start camera"
    assert tr("实时相机 · 正在控制") == "Live camera · controlling"
    assert tr("按下 / 拖拽就绪\n左 2 · 右 1 · 拖 0") == "Presses / drags\nleft 2 · right 1 · drag 0"
    assert tr("第 3/5 轮 · 第 27/60 段") == "Round 3/5 · Segment 27/60"
    assert (
        tr("手掌和镜头的夹角：现在约 61°，这一轮需要 55° 以上 ✓")
        == "Palm-to-camera angle: now about 61°; this round needs 55° or more ✓"
    )
    assert tr("时序 CNN · 内置") == "Temporal CNN · built-in"
    assert tr("P01 · a.zip · 检查中 3/60") == "P01 · a.zip · in review 3/60"
    assert tr("手靠近画面右侧 · 松中指，将手移回画面中央，再捏住继续。") == (
        "Hand near the right edge · Release the middle finger, move the hand back to the centre, "
        "then pinch to continue."
    )
    assert tr("手靠近画面左侧、上沿 · 先完成当前点击，再松中指换位。").startswith(
        "Hand near the left, top edge · "
    )
    assert tr("已看 3/60，其中作废 1。进度会自动保存，可以随时关掉再继续。") == (
        "Seen 3/60, voided 1. Progress is saved automatically; you can close and continue later."
    )
    assert tr("  启动相机 ") == "  Start camera "
    assert tr("no Chinese here") == "no Chinese here"
    assert not ui_lang.MISSING
    assert tr("一段没有译文的话") == "一段没有译文的话"
    assert ui_lang.MISSING == {"一段没有译文的话"}


def test_every_recording_prompt_and_testbench_task_has_english(english):
    texts = []
    for steps in (contact_protocol.session(), contact_protocol.side_session()):
        for clip in contact_protocol.clips(steps):
            texts += (
                [clip.text, clip.notice]
                + [step.text for step in clip.steps]
                + [s.note for s in clip.steps]
            )
    for task in practice.TASKS.values():
        texts += [task.title, task.instruction]
    leftovers = {t for t in texts if t and ui_lang.HAN.search(ui_lang.tr(t))}
    assert not leftovers


def test_english_windows_show_no_chinese(tmp_path):
    """Open the desktop window (demo, settings, mini view), testbench, recording window and label review in English."""
    script = textwrap.dedent(
        """
        import re, sys
        from pathlib import Path
        from pinchpilot import ui_lang
        ui_lang.install("en")
        from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLabel, QWidget
        from pinchpilot.contact_review_window import ReviewWindow
        from pinchpilot.desktop import DesktopWindow
        from test_contact_review import SHORT, record

        HAN = re.compile(r"[\\u4e00-\\u9fff]")
        app = QApplication.instance() or QApplication([])
        root = Path(sys.argv[1])

        def shown(window):
            texts = []
            for w in [window] + window.findChildren(QWidget):
                texts += [w.toolTip(), w.windowTitle()]
                if isinstance(w, (QLabel, QAbstractButton)):
                    texts.append(w.text())
                if isinstance(w, QComboBox):
                    texts += [w.itemText(i) for i in range(w.count())]
            return {t for t in texts if HAN.search(t)}

        def settle(window):
            for _ in range(30):
                app.processEvents()
            window.grab()  # paints, so QPainter texts go through the catalogue too

        (root / "ws" / "data").mkdir(parents=True)
        desktop = DesktopWindow(root / "ws", demo=True)
        desktop.show()
        settle(desktop)
        desktop.advanced_button.click()
        desktop.toggle_mini(True)
        settle(desktop)
        desktop.toggle_mini(False)
        left = shown(desktop)
        desktop.open_practice()
        settle(desktop.practice_window)
        for i in range(desktop.practice_window.task_choice.count()):
            desktop.practice_window.task_choice.setCurrentIndex(i)
            settle(desktop.practice_window)
        left |= shown(desktop.practice_window)
        desktop.practice_window.close()
        app.processEvents()
        desktop.open_contact_capture()
        settle(desktop.contact_capture_window)
        left |= shown(desktop.contact_capture_window)
        desktop.contact_capture_window.close()
        desktop.close()
        missing = set(ui_lang.MISSING)
        ui_lang.MISSING.clear()
        recording = root / "recordings"
        recording.mkdir()
        record(recording)
        review = ReviewWindow(recording, "P01")
        review.show()
        settle(review)
        review.next_button.click()
        settle(review)
        review.accept_button.click()  # a saved decision puts "in review 1/N" in the picker
        settle(review)
        review_left = shown(review)
        review.close()
        # The review test's short protocol has prompts of its own, which are not interface text.
        found = set(ui_lang.MISSING)
        ui_lang.MISSING.clear()
        ui_lang._translate.cache_clear()
        for step in SHORT:
            ui_lang.tr(step.text), ui_lang.tr(step.note)
        prompts = set(ui_lang.MISSING)
        missing |= found - prompts
        left |= {t for t in review_left if all(p not in t for p in prompts)}
        print("LEFT", sorted(left))
        print("MISSING", sorted(missing))
        """
    )
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]),
    }
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = dict(
        line.split(" ", 1)
        for line in done.stdout.splitlines()
        if line.startswith(("LEFT", "MISSING"))
    )
    assert lines["LEFT"] == "[]"
    assert lines["MISSING"] == "[]"
