"""Label review, from a guided capture to reviewed labels and a cross-check. Offscreen;
the camera source is simulated, not human data."""

import json
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from pinchpilot import contact_protocol, contact_review
from pinchpilot.cli import main
from pinchpilot.contact_capture import ContactCaptureWindow
from pinchpilot.contact_pool import load_contact_pool
from pinchpilot.contact_protocol import KEY, Step
from pinchpilot.contact_review import (
    ACCEPTED,
    AGREE,
    DISAGREE,
    DISCARDED,
    Margins,
    frame_labels,
    list_recordings,
    load_crosscheck,
    load_progress,
    load_reviewed,
    load_session,
    pending,
    sample_units,
    save_crosscheck,
    save_progress,
    save_reviewed,
)
from pinchpilot.contact_review_window import ReviewWindow
from pinchpilot.tripod_demo import synthetic_tripod

SHORT = [
    Step("ready", "准备", 2.0, (None, None, None)),
    Step("open", "手放松张开", 5.0, (0, 0, 0), round=1),
    Step("grip_on", "捏住：拇指和中指", 3.0, (1, 0, 0), round=1),
    Step("grip_off", "松开", 2.5, (0, 0, 0), round=1),
    Step("grip_on", "捏住：拇指和中指", 3.0, (1, 0, 0), round=1),
    Step("grip_off", "松开", 2.5, (0, 0, 0), round=1),
    Step("grip_index_tap", "保持捏住。食指点拇指", 8.0, (1, KEY, 0), "碰到时按住空格", 1),
    Step("ring_tap", "拇指碰无名指", 8.0, (0, 0, KEY), "碰到时按住空格", 2),
    Step("index_near", "食指靠近拇指但不要碰到", 5.0, (0, 0, 0), "不要按空格", 2),
]
LAG = 0.12  # The simulated person presses Space this long after the fingers touch.


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def record(directory, participant="P01", lag=LAG, seed=1, stop_at=None):
    """Run a guided capture with a synthetic hand that follows the prompts.

    Returns the zip and, for every frame time from the start of the session, whether the
    index and the ring finger were really touching.
    """
    window = ContactCaptureWindow(directory, guided=True)
    window.timer.stop()
    window.protocol = SHORT
    now = [300.0 + seed]
    window.clock = lambda: now[0]
    rng = np.random.default_rng(seed)

    def key(pressed):
        kind = QEvent.Type.KeyPress if pressed else QEvent.Type.KeyRelease
        event = QKeyEvent(kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", False)
        (window.keyPressEvent if pressed else window.keyReleaseEvent)(event)

    def hand(grip, index, ring):
        return synthetic_tripod(
            now[0],
            grip=grip,
            contact=0.1 if index else 0.8,
            right_contact=0.1 if ring else 0.8,
            noise=tuple(rng.normal(0, 0.0007, 2)),
        )

    window.feed(hand(False, False, False), now[0])
    window.participant.setText(participant)
    window.consent.setChecked(True)
    window.start_recording()
    start, schedule = now[0], window.schedule
    waiting, before, truth = [], False, []
    while window.recorder_active:
        now[0] += 1 / 30
        elapsed = now[0] - start
        if stop_at is not None and elapsed >= stop_at:
            window.stop_recording()
            break
        position = contact_protocol.position(schedule, elapsed)
        grip = touching = False
        labels = (0, 0, 0)
        if position is not None:
            begin, end, step = schedule[position]
            labels = step.labels
            grip = labels[0] == 1
            if KEY in labels:
                touching = (
                    begin + 1.2 <= elapsed < end - 0.7 and (elapsed - begin - 1.2) % 2.0 < 0.6
                )
        if touching != before:
            waiting.append((now[0] + lag, touching))
            before = touching
        while waiting and waiting[0][0] <= now[0]:
            key(waiting.pop(0)[1])
        index, ring = touching and labels[1] == KEY, touching and labels[2] == KEY
        truth.append((elapsed, index, ring))
        window.feed(hand(grip, index, ring), now[0])
    bundle = window.bundle_path
    window.close()
    return bundle, truth


@pytest.fixture(scope="module")
def recorded(application, tmp_path_factory):
    return record(tmp_path_factory.mktemp("guided"))


@pytest.fixture
def session(recorded, tmp_path):
    """A private copy of the recorded zip, so that each test writes its own sidecars."""
    target = tmp_path / "P01" / recorded[0].name
    target.parent.mkdir()
    target.write_bytes(recorded[0].read_bytes())
    return load_session(target)


def accept_all(session, discard=()):
    return {unit.index: DISCARDED if unit.index in discard else ACCEPTED for unit in session.units}


def test_units_merge_runs_of_short_steps_and_leave_out_unlabelled_ones(session):
    assert session.participant == "P01" and session.source == "camera"
    assert session.protocol == contact_protocol.NAME
    assert [(unit.steps, unit.round, unit.uses_key) for unit in session.units] == [
        ((1,), 1, False),
        ((2, 3, 4, 5), 1, False),
        ((6,), 1, True),
        ((7,), 2, True),
        ((8,), 2, False),
    ]
    assert session.units[1].title == "捏住：拇指和中指 / 松开，共 4 步"
    assert session.units[1].start == 7.0 and session.units[1].end == 18.0
    full = contact_protocol.timeline(contact_protocol.session())
    units = contact_review._units(full, full[-1][1])
    assert len(units) == 60 and sum(unit.uses_key for unit in units) == 30
    assert [len(unit.steps) for unit in units[:3]] == [1, 8, 1]
    assert len(contact_review._units(full, 100.0)) < 12  # A session stopped in round one.
    assert np.isfinite(session.distances[session.present]).all()


def test_shift_undoes_the_reaction_delay_and_discarding_blanks_one_unit(session, recorded):
    truth = np.array([(index, ring) for _, index, ring in recorded[1]], dtype=bool)
    truth = truth[: len(session.times)]

    def wrongly_touching(margins):
        labels = frame_labels(session, margins, {})[: len(truth)]
        return int(
            np.sum((labels[:, 1] == 1) & ~truth[:, 0]) + np.sum((labels[:, 2] == 1) & ~truth[:, 1])
        )

    def wrongly_apart(margins):
        labels = frame_labels(session, margins, {})[: len(truth)]
        return int(
            np.sum((labels[:, 1] == 0) & truth[:, 0]) + np.sum((labels[:, 2] == 0) & truth[:, 1])
        )

    # Unshifted, the marks run late: frames after each release still say "touching", and
    # frames at the start of each touch still say "apart".
    assert wrongly_touching(Margins()) > 5 and wrongly_apart(Margins()) > 5
    aligned = Margins(shift=LAG)
    assert wrongly_touching(aligned) == 0 and wrongly_apart(aligned) == 0
    kept = frame_labels(session, aligned, {})
    assert (kept[:, 1] == 1).sum() > 20 and (kept[:, 2] == 1).sum() > 20

    unit = session.units[2]
    inside = (session.times >= unit.start) & (session.times < unit.end)
    dropped = frame_labels(session, aligned, {unit.index: DISCARDED})
    assert (dropped[inside] == -1).all() and np.array_equal(dropped[~inside], kept[~inside])
    assert np.array_equal(frame_labels(session, aligned, {unit.index: ACCEPTED}), kept)
    wider = frame_labels(session, Margins(shift=LAG, edge=0.2), {})
    assert (wider[:, 1] == -1).sum() > (kept[:, 1] == -1).sum()
    for bad in (dict(shift=0.9), dict(edge=0.0), dict(settle=float("nan")), dict(lead=True)):
        with pytest.raises(ValueError, match="超出范围"):
            Margins(**bad)


def test_reviewed_labels_need_every_unit_and_are_then_accepted_for_training(session):
    margins, decisions = Margins(shift=LAG), accept_all(session, discard={3})
    partial = {key: value for key, value in decisions.items() if key != 4}
    assert pending(session, partial) == [4]
    with pytest.raises(ValueError, match="还有 1 个单元没有看完"):
        save_reviewed(session, "P01", margins, partial)
    with pytest.raises(ValueError, match="复核者"):
        save_reviewed(session, " ", margins, decisions)
    with pytest.raises(ValueError, match="结论无效"):
        save_reviewed(session, "P01", margins, {**decisions, 0: "maybe"})
    assert not session.sidecar(".labels.json").exists()
    with pytest.raises(ValueError, match="还没有复核"):
        load_contact_pool(session.path.parent.parent)

    target = save_reviewed(session, "P01", margins, decisions)
    assert target == session.path.with_name(f"{session.path.stem}.labels.json")
    spec = json.loads(target.read_text(encoding="utf-8"))
    assert spec["reviewed"] is True and spec["label_source"] == "human_reviewed"
    assert spec["annotator"] == "P01" and spec["recording_sha256"] == session.recording_sha256
    review = spec["review"]
    assert review["margins"]["shift"] == LAG and review["units"] == 5
    assert [item["unit"] for item in review["discarded_units"]] == [3]
    assert review["discarded_units"][0]["title"] == "拇指碰无名指"

    pool = load_contact_pool(session.path.parent.parent)
    assert pool.metadata["labels"] == "reviewed" and len(pool.metadata["recordings"]) == 1
    assert pool.metadata["recordings"][0]["participant"] == "P01"
    # The discarded ring unit is gone, so no ring contact is left in this short session.
    assert pool.y[:, 1].any() and not pool.y[:, 2].any()
    assert load_reviewed(session)[:3] == ("P01", margins, decisions)
    # Reviewing again replaces the tool's own file; a foreign file is left alone.
    save_reviewed(session, "P01", margins, accept_all(session))
    assert load_reviewed(session)[2] == accept_all(session)
    target.write_text(json.dumps({"reviewed": True, "recording_sha256": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="不能覆盖"):
        save_reviewed(session, "P01", margins, decisions)


def test_progress_is_kept_between_sittings_and_tied_to_the_recording(session):
    assert load_progress(session) is None
    margins = Margins(shift=0.08, edge=0.1)
    save_progress(session, "P01", margins, {0: ACCEPTED, 2: DISCARDED})
    assert load_progress(load_session(session.path)) == (
        "P01",
        margins,
        {0: ACCEPTED, 2: DISCARDED},
    )
    record_path = session.sidecar(".review.json")
    saved = json.loads(record_path.read_text(encoding="utf-8"))
    assert saved["complete"] is False and saved["units"] == 5
    saved["recording_sha256"] = "0" * 64
    record_path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="不匹配"):
        load_progress(session)
    with pytest.raises(ValueError, match="结论无效"):
        save_progress(session, "P01", margins, {99: ACCEPTED})


def test_crosscheck_is_a_fixed_sample_by_another_person_tied_to_the_reviewed_labels(session):
    with pytest.raises(ValueError, match="还没有复核过的标签"):
        save_crosscheck(session, "P02", {})
    decisions = accept_all(session, discard={4})
    save_reviewed(session, "P01", Margins(shift=LAG), decisions)
    sample = sample_units(session, decisions, "P02")
    assert sample == [0, 1, 2, 3]  # Too few units to sample: every accepted one is checked.
    with pytest.raises(ValueError, match="另一位组员"):
        save_crosscheck(session, "P01", {})
    with pytest.raises(ValueError, match="结论无效"):
        save_crosscheck(session, "P02", {4: AGREE})
    target = save_crosscheck(session, "P02", {0: AGREE, 2: DISAGREE}, {2: " 第二下偏晚 "})
    assert target.name == f"{session.stem}.crosscheck.P02.json"
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["annotator"] == "P01" and saved["checker"] == "P02"
    assert saved["agreed"] == 1 and saved["complete"] is False and saved["sample"] == sample
    assert saved["verdicts"][1] == {
        "unit": 2,
        "round": 1,
        "title": "保持捏住。食指点拇指",
        "verdict": DISAGREE,
        "note": "第二下偏晚",
    }
    assert load_crosscheck(session, "P02") == ({0: AGREE, 2: DISAGREE}, {2: "第二下偏晚"})
    assert load_crosscheck(session, "P03") == ({}, {})
    # The owner reviews again: verdicts on the old labels no longer apply.
    save_reviewed(session, "P01", Margins(shift=0.05), decisions)
    assert load_crosscheck(session, "P02") == ({}, {})

    # On a full session the sample is about a tenth, fixed per checker, half Space-marked.
    full = contact_protocol.timeline(contact_protocol.session())
    whole = replace(session, units=contact_review._units(full, full[-1][1]))
    everything = accept_all(whole, discard={5, 6})
    first = sample_units(whole, everything, "P02")
    assert first == sample_units(whole, everything, "P02") and len(first) == 6
    assert first != sample_units(whole, everything, "P03")
    assert not {5, 6} & set(first)
    assert sum(whole.units[index].uses_key for index in first) >= 3


def test_folder_listing_follows_the_review(session):
    folder = session.path.parent.parent
    (folder / "P01" / "notes.zip").write_bytes(b"not a zip")
    assert [(row["participant"], row["status"]) for row in list_recordings(folder)] == [
        ("P01", "未检查")
    ]
    save_progress(session, "P01", Margins(), {0: ACCEPTED})
    row = list_recordings(folder)[0]
    assert (row["status"], row["decided"], row["units"]) == ("检查中", 1, 5)
    save_reviewed(session, "P01", Margins(shift=0.05), accept_all(session, discard={1}))
    save_crosscheck(session, "P03", {0: AGREE, 2: DISAGREE}, {2: "偏晚"})
    row = list_recordings(folder)[0]
    assert row["status"] == "已检查" and row["discarded"] == 1
    assert row["margins"]["shift"] == 0.05
    assert row["crosschecks"] == [
        {
            "checker": "P03",
            "agreed": 1,
            "checked": 2,
            "sample": 4,
            "complete": False,
            "current": True,
        }
    ]
    assert contact_review.disagreements(session) == {2: [("P03", "偏晚")]}
    # Reviewing again makes the earlier cross-check stale: it no longer counts as current.
    save_reviewed(session, "P01", Margins(shift=0.06), accept_all(session))
    assert list_recordings(folder)[0]["crosschecks"][0]["current"] is False
    assert contact_review.disagreements(session) == {}
    with pytest.raises(ValueError, match="找不到文件夹"):
        list_recordings(folder / "missing")


def test_a_recording_without_prompts_or_an_unknown_file_is_refused(session, tmp_path):
    with pytest.raises(ValueError, match="请选择按提示录制生成的"):
        load_session(tmp_path / "notes.txt")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="压缩包无法读取"):
        load_session(bad)
    import zipfile

    stripped = tmp_path / "stripped.zip"
    with zipfile.ZipFile(session.path) as source, zipfile.ZipFile(stripped, "x") as target:
        for info in source.infolist():
            if not info.filename.endswith(".protocol.json"):
                target.writestr(info, source.read(info))
    with pytest.raises(ValueError, match="不是按提示录的"):
        load_session(stripped)


def press(window, key):
    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


@pytest.fixture
def folder(recorded, tmp_path):
    target = tmp_path / "recordings" / "P01" / recorded[0].name
    target.parent.mkdir(parents=True)
    target.write_bytes(recorded[0].read_bytes())
    return tmp_path / "recordings"


def test_window_reviews_ones_own_recording_with_keys_and_saves_at_the_end(application, folder):
    window = ReviewWindow(folder, "P01")
    window.timer.stop()
    assert window.mode == "review" and len(window.items) == 5 and window.position == 0
    assert "检查自己的标签" in window.mode_label.text() and "未检查" in window.picker.itemText(0)
    assert (
        window.title_label.text() == "手放松张开" and "标签来自屏幕提示" in window.info_label.text()
    )
    assert not window.finish_button.isEnabled() and window.note.isHidden()
    press(window, Qt.Key.Key_Return)
    assert window.decisions == {0: ACCEPTED} and window.position == 1
    press(window, Qt.Key.Key_Right)
    assert window.position == 2 and "食指的标签来自空格" in window.info_label.text()
    press(window, Qt.Key.Key_X)
    assert window.decisions[2] == DISCARDED and window.position == 3
    unit = window.session.units[2]
    inside = (window.session.times >= unit.start) & (window.session.times < unit.end)
    assert (window.labels[inside] == -1).all()
    # Progress is on disk after every decision; reopening continues at the first open unit.
    window.close()
    window = ReviewWindow(folder, "P01")
    window.timer.stop()
    assert "检查中 2/5" in window.picker.itemText(0)
    assert window.decisions == {0: ACCEPTED, 2: DISCARDED} and window.position == 1
    press(window, Qt.Key.Key_Return)
    assert window.position == 3  # The next unit without a decision.
    press(window, Qt.Key.Key_Return)
    press(window, Qt.Key.Key_Return)
    assert not pending(window.session, window.decisions) and window.finish_button.isEnabled()
    assert "全部看完了" in window.status_label.text()
    window.finish()
    labels = window.session.sidecar(".labels.json")
    assert labels.is_file() and labels.name in window.status_label.text()
    assert load_contact_pool(folder).metadata["labels"] == "reviewed"
    window.close()


def test_window_margins_relabel_and_send_accepted_units_back(application, folder):
    window = ReviewWindow(folder, "P01")
    window.timer.stop()
    press(window, Qt.Key.Key_Return)
    press(window, Qt.Key.Key_X)
    before = window.labels.copy()
    slider = window.sliders["shift"]
    slider.setValue(round(LAG * 1000 / slider.property("step")))
    assert "空格标记提前 120 毫秒" in window.slider_labels["shift"].text()
    assert np.array_equal(window.labels, before)  # Nothing changes until the slider is let go.
    window._margins_changed()
    assert window.margins == Margins(shift=LAG) and not np.array_equal(window.labels, before)
    assert window.decisions == {1: DISCARDED}  # The discard stays; the accepted unit is open again.
    assert "需要重新看一遍" in window.status_label.text()
    assert load_progress(window.session)[1] == Margins(shift=LAG)
    # Playing moves the playhead with the clock and loops over the unit.
    window.clock = lambda: window.test_now
    window.test_now = 10.0
    window.go(0)
    window._tick()
    window.test_now += 2.0
    window._tick()
    assert window.playhead == pytest.approx(window.unit.start - 0.3 + 2.0)
    press(window, Qt.Key.Key_Space)
    for _ in range(2):
        window.test_now += 5.0
        window._tick()
    assert window.playhead == pytest.approx(window.unit.start - 0.3 + 2.0) and not window.playing
    press(window, Qt.Key.Key_Space)
    window._tick()
    window.test_now += 60.0
    window._tick()
    assert window.playhead == pytest.approx(window.unit.start - 0.3)
    window.close()


def test_window_cross_checks_a_teammates_reviewed_labels(application, folder):
    window = ReviewWindow(folder, "P02")
    assert window.unit is None and "还没有复核过的标签" in window.status_label.text()
    assert not window.accept_button.isEnabled()
    window.close()
    owner = load_session(next(folder.rglob("*.zip")))
    save_reviewed(owner, "P01", Margins(shift=LAG), accept_all(owner, discard={4}))

    window = ReviewWindow(folder, "P02")
    window.timer.stop()
    assert window.mode == "crosscheck" and [unit.index for unit in window.items] == [0, 1, 2, 3]
    assert "抽查 P01 的标签" in window.mode_label.text() and window.margins == Margins(shift=LAG)
    assert window.accept_button.text() == "同意 (Enter)" and not window.note.isHidden()
    assert not window.sliders["shift"].isEnabled() and window.finish_button.isHidden()
    press(window, Qt.Key.Key_Return)
    window.note.setText("松开标得太晚")
    press(window, Qt.Key.Key_X)
    assert window.decisions == {0: AGREE, 1: DISAGREE} and window.notes[1] == "松开标得太晚"
    assert window.note.text() == ""  # The next unit starts with an empty note.
    press(window, Qt.Key.Key_Return)
    press(window, Qt.Key.Key_Return)
    assert "抽查完成" in window.status_label.text() and "不同意 1" in window.status_label.text()
    saved = json.loads(owner.sidecar(".crosscheck.P02.json").read_text(encoding="utf-8"))
    assert saved["complete"] is True and saved["agreed"] == 3
    assert saved["verdicts"][1]["note"] == "松开标得太晚"
    # A cross-check never changes the owner's labels or review record.
    assert load_reviewed(owner)[2] == accept_all(owner, discard={4})
    window.close()

    # The owner sees the objection on that unit; saving a new review clears it.
    window = ReviewWindow(folder, "P01")
    window.timer.stop()
    window.go(1)
    assert "P02 抽查不同意：松开标得太晚" in window.info_label.text()
    window.go(0)
    assert "抽查不同意" not in window.info_label.text()
    window.finish()
    window.go(1)
    assert "抽查不同意" not in window.info_label.text()
    window.close()


def test_window_lists_ones_own_recordings_first_and_opens_one_that_needs_review(
    application, folder, tmp_path
):
    other, _ = record(tmp_path / "other", participant="P00", seed=9)
    target = folder / "P00" / other.name
    target.parent.mkdir()
    target.write_bytes(other.read_bytes())
    window = ReviewWindow(folder, "P01")
    window.timer.stop()
    assert [window.picker.itemText(i)[:3] for i in range(2)] == ["P01", "P00"]
    assert window.mode == "review" and window.session.participant == "P01"
    press(window, Qt.Key.Key_Return)
    assert "检查中 1/5" in window.picker.itemText(0)  # The list follows each saved decision.
    for _ in range(4):
        press(window, Qt.Key.Key_Return)
    window.finish()
    assert "已检查" in window.picker.itemText(0)
    window.open_session(window.picker.itemData(1))
    assert window.unit is None and "还没有复核过的标签" in window.status_label.text()
    window.close()


def test_review_command_opens_and_closes(application, folder, capsys):
    archive = next(folder.rglob("*.zip"))
    assert main(["ml-review", str(archive), "--annotator", "P01", "--smoke-seconds", "0.2"]) == 0
    assert main(["ml-review", str(folder / "missing"), "--annotator", "P01"]) == 2
    assert "找不到文件或文件夹" in capsys.readouterr().err
    assert main(["ml-review", str(archive), "--annotator", " "]) == 2
    assert "--annotator" in capsys.readouterr().err
    assert main(["ml-review", str(archive)]) == 2
    assert "--annotator" in capsys.readouterr().err
    assert main(["ml-review", str(folder), "--status"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [(row["participant"], row["status"]) for row in listed] == [("P01", "未检查")]


def test_a_session_stopped_early_can_still_be_reviewed(application, tmp_path):
    bundle, _ = record(tmp_path, participant="P03", seed=4, stop_at=20.5)
    session = load_session(bundle)
    assert session.ended == pytest.approx(20.5, abs=0.05)
    assert [unit.steps for unit in session.units] == [(1,), (2, 3, 4, 5), (6,)]
    labels = frame_labels(session, Margins(), {})
    assert (labels[session.times >= 20.5] == -1).all()
