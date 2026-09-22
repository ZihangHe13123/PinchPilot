import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_compatible_gestures import uncertain
from test_tripod import Sequence
from test_tripod_scroll import ScrollSequence

from pinchpilot.diagnostics import HAND_REASONS, inspect_engine
from pinchpilot.domain import EngineResult, HandFrame
from pinchpilot.tripod import TripodConfig, TripodEngine
from pinchpilot.tripod_demo import synthetic_tripod


def diagnostic(sequence, packet=None):
    return inspect_engine(sequence.engine, sequence.result, packet)


def reason(sequence):
    return diagnostic(sequence)["reason"]


def sequence():
    s = Sequence(TripodConfig(pointer_basis="unified"))
    s.frames(10)
    return s


def test_distinguishes_initial_arm_move_clutch_and_regrip_without_new_state_names():
    s = Sequence()
    s.frames(grip=False)
    assert reason(s) == "wait_grip"
    s.frames()
    assert reason(s) == "grip_confirming"
    s.frames(10)
    assert reason(s) == "moving"
    s.frames(grip=False)
    assert reason(s) == "clutch_frozen"
    s.frames()
    assert reason(s) == "grip_confirming"


def test_uncertain_grip_is_visible_even_while_engine_still_reports_control():
    s = sequence()
    for _ in range(2):
        s.frame(uncertain(synthetic_tripod(s.t + 1 / 30)))
    report = diagnostic(s)
    assert report["reason"] == "grip_unstable"
    assert report["state"] == "CONTROL"
    assert report["values"]["grip_pause_ms"] == pytest.approx(1000 / 30)
    assert report["values"]["grip_keep_margin"] < 0
    assert not s.result.events
    s.frames()
    assert reason(s) == "moving"
    assert "grip_pause_ms" not in diagnostic(s)["values"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_uncertain_contact_then_clear_reports_actual_wait_gate(side):
    s = sequence()
    kwargs = {"contact" if side == "left" else "right_contact": 0.1}
    s.frame(uncertain(synthetic_tripod(s.t + 1 / 30, **kwargs)))
    assert reason(s) == "grip_unstable"
    s.frames(10, **kwargs)
    assert reason(s) == ("wait_index_clear" if side == "left" else "wait_both_clear")


def test_left_hover_contact_press_drag_clutch_and_release_reasons():
    s = sequence()
    s.frames(contact=0.33)
    assert reason(s) == "left_approach"
    s.frames(contact=0.1)
    assert reason(s) == "left_contact_confirming"
    s.frames(3, contact=0.1)
    assert reason(s) == "left_pressed"
    s.frames(12, contact=0.1)
    assert reason(s) == "dragging"
    s.frames(grip=False, contact=0.1)
    assert reason(s) == "drag_frozen"
    s.frames(grip=False)
    assert reason(s) == "left_release_confirming"
    s.frames(4, grip=False)
    assert reason(s) == "clutch_frozen"


def test_right_approach_contact_and_release_gate_are_separate():
    s = sequence()
    s.frames(right_contact=0.33)
    assert reason(s) == "right_approach"
    s.frames(right_contact=0.1)
    assert reason(s) == "right_contact_confirming"
    s.frames(5, right_contact=0.1)
    assert reason(s) == "right_release_wait"


def test_aborted_contact_clear_confirmation_does_not_claim_a_click():
    s = sequence()
    s.frames(contact=0.33)
    s.frames()
    assert reason(s) == "contact_clear_confirming"
    assert not s.engine.left_down


def test_scroll_preparation_and_active_scrolling():
    s = ScrollSequence(TripodConfig(pointer_basis="unified"))
    s.v(12)
    assert reason(s) == "scroll_ready"
    s.begin()
    assert reason(s) == "scrolling"


@pytest.mark.parametrize("hand_reason", sorted(HAND_REASONS))
def test_missing_frame_uses_explicit_selector_reason_only(hand_reason):
    s = sequence()
    s.frame(HandFrame(s.t + 1 / 30))
    report = diagnostic(s, SimpleNamespace(hand_reason=hand_reason))
    assert report["reason"] == hand_reason
    assert isinstance(report["label"], str)
    assert "grip_keep_margin" not in report["values"]


def test_unavailable_reason_and_reset_cause_are_not_guessed_from_hint():
    s = sequence()
    s.frame(HandFrame(s.t + 1 / 30))
    s.result.hint = "双手太近 · 摄像头断开 · 捏合不稳定"
    report = diagnostic(s, SimpleNamespace(hand_reason="sensitive arbitrary unknown reason"))
    assert report["reason"] == "tracking_unavailable"
    assert "sensitive" not in json.dumps(report)
    s.frames(10)
    # A jump stores the current valid features and resets arming. Its precise
    # cause has been lost to this stateless inspector and must not be invented.
    s.frames(x=0.9)
    assert reason(s) == "wait_grip"


def test_stale_packet_reason_cannot_override_live_engine_state():
    s = sequence()
    assert diagnostic(s, SimpleNamespace(hand_reason="no_hand"))["reason"] == "moving"


def test_paused_is_not_mislabelled_as_no_hand():
    engine = TripodEngine()
    report = inspect_engine(engine, engine.set_enabled(False))
    assert report["reason"] == "paused"


def test_wrist_calibration_is_explicit_when_valid_hand_is_waiting():
    s = Sequence(TripodConfig(pointer_basis="wrist"))
    s.frames()
    assert reason(s) == "wrist_calibration_needed"


def test_margins_have_documented_direction_and_quality_is_a_scalar():
    s = sequence()
    report = diagnostic(s)
    values = report["values"]
    assert values["grip_engage_margin"] > 0
    assert values["grip_keep_margin"] > 0
    assert values["left_clear_margin"] > 0
    assert values["right_clear_margin"] > 0
    assert values["left_touch_margin"] < 0
    assert 0 <= values["compatible_quality"] <= 1
    assert 0 <= values["compatible_correction_px"] <= 3
    s.frames(contact=0.1)
    assert diagnostic(s)["values"]["left_touch_margin"] > 0


def test_inspection_cannot_mutate_engine_result_or_leak_positions():
    s = sequence()
    s.frames(4, x=0.5341, contact=0.1)
    before = pickle.dumps((s.engine, s.result))
    report = diagnostic(s)
    assert pickle.dumps((s.engine, s.result)) == before
    assert set(report) == {"reason", "label", "state", "values"}
    assert set(report["values"]) <= {
        "motion_engaged",
        "left_down",
        "cancelled",
        "grip_engage_margin",
        "grip_keep_margin",
        "left_touch_margin",
        "left_clear_margin",
        "right_touch_margin",
        "right_clear_margin",
        "grip_pause_ms",
        "compatible_quality",
        "compatible_correction_px",
    }
    assert all(isinstance(value, (float, bool)) for value in report["values"].values())
    json.dumps(report, allow_nan=False)


def test_nonfinite_values_are_omitted_and_unknown_state_stays_unknown():
    s = sequence()
    damaged = replace(s.result, grip=float("nan"), contact=float("inf"))
    report = inspect_engine(s.engine, damaged)
    assert report["reason"] == "tracking_unavailable"
    assert "grip_engage_margin" not in report["values"]
    assert "left_touch_margin" not in report["values"]
    json.dumps(report, allow_nan=False)
    report = inspect_engine(SimpleNamespace(), EngineResult("some unsupported state"))
    assert report["reason"] == "unknown"
    assert report["state"] == "UNKNOWN"
