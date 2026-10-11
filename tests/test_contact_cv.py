"""Leave-one-person-out comparison: settings, leakage, selection and the four models."""

import json
from pathlib import Path

import numpy as np
import pytest

from pinchpilot import contact_cv
from pinchpilot.cli import main
from pinchpilot.contact_cv import check_settings, markdown, read_settings, run_cv, write_report
from pinchpilot.contact_data import CHANNELS, FEATURE_COUNT
from pinchpilot.contact_pool import ContactPool

SETTINGS = Path(__file__).parents[1] / "experiments" / "contact"
FLAGS = ("flat_middle", "flat_index", "flat_ring", "nearest")
PEOPLE = ("P01", "P02", "P03")


def spec(model="forest", *candidates, **extra):
    return {
        "schema": contact_cv.SETTINGS_SCHEMA,
        "model": model,
        "candidates": list(candidates) or [{"name": "基础"}],
        **extra,
    }


def fake_pool(people=PEOPLE, count=96, seed=5, labels="reviewed", source="camera", spare=24):
    """Separable windows. Feature 0 carries the person's number so that a spy can tell
    whose windows it was given; the three distance features carry the labels. Each person
    has four rounds of two steps, and `spare` further windows without labels."""
    rng = np.random.default_rng(seed)
    frames, total = 8, len(people) * count
    combinations = np.array(
        [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int8
    )
    # In random order, so that taking every n-th window cannot drop a label value.
    y = rng.permutation(np.tile(combinations, (total // 8, 1)))
    features = rng.normal(0, 0.05, (total, frames, FEATURE_COUNT)).astype(np.float32)
    participants = np.repeat(np.array(people), count)
    for number, person in enumerate(people):
        mine = participants == person
        features[mine, :, 0] = number
        # Touching is a small distance, and every person sits at a slightly different scale.
        features[mine, :, 63:66] += np.where(y[mine, None, :] == 1, 0.1, 0.6) * (1 + 0.1 * number)
    extra = rng.normal(0, 0.05, (len(people) * spare, frames, FEATURE_COUNT)).astype(np.float32)
    extra[:, :, 0] = np.repeat(np.arange(len(people)), spare)[:, None]
    return ContactPool(
        np.concatenate((features, extra)).reshape(-1, FEATURE_COUNT),
        np.arange(total) * frames + frames - 1,
        y,
        y.copy(),
        participants,
        np.repeat(np.arange(len(people), dtype=np.int32), count),
        np.tile(np.repeat(np.array([1, 2, 3, 4], dtype=np.int8), count // 4), len(people)),
        np.tile(np.repeat(np.array([0, 1], dtype=np.int16), count // 8), len(people) * 4),
        {
            "source": source,
            "labels": labels,
            "feature_version": "contact-v1",
            "window_config": {"frames": 8, "max_gap_seconds": 0.12, "max_window_seconds": 0.8},
            "recordings": [{"protocol": "pinchpilot-guided-session-2"} for _ in people],
            "step_names": ["grip_move", "index_near"],
            "fingerprint": "f" * 64,
        },
        frames,
        (total + np.arange(len(people) * spare)) * frames + frames - 1,
        np.repeat(np.array(people), spare),
    )


def owners(windows):
    return {PEOPLE[int(value)] for value in np.unique(windows[:, -1, 0])}


def test_settings_fill_defaults_and_name_what_is_wrong():
    settings = check_settings(
        spec("forest", {"name": " 基础 "}, {"name": "深", "depth": 20}, seed=7)
    )
    assert settings["model"] == "forest" and settings["seed"] == 7
    assert [item["name"] for item in settings["candidates"]] == ["基础", "深"]
    assert settings["candidates"][0]["options"] == contact_cv.DEFAULTS["forest"]
    assert settings["candidates"][1]["options"]["depth"] == 20
    with pytest.raises(ValueError, match="不认识的项 channels；随机森林可用的项：.*trees"):
        check_settings(spec("forest", {"name": "a", "channels": 8}))
    for model, bad in (
        ("forest", {"trees": 5}),
        ("forest", {"depth": 12.0}),
        ("forest", {"stride": True}),
        ("cnn", {"learning_rate": 0}),
        ("cnn", {"positive_weight": "auto"}),
        ("rules", {"statistic": "max"}),
        ("finger_cnn", {"shared": 1}),
    ):
        with pytest.raises(ValueError, match="取值无效"):
            check_settings(spec(model, {"name": "a", **bad}))
    with pytest.raises(ValueError, match="不重复的 name"):
        check_settings(spec("forest", {"name": "a"}, {"name": "a "}))
    with pytest.raises(ValueError, match="model 应为"):
        check_settings(spec("svm"))
    with pytest.raises(ValueError, match="不认识的项：epochs"):
        check_settings(spec("cnn", epochs=3))
    # Pretraining belongs to the temporal CNN; the label share can be cut for every model.
    with pytest.raises(ValueError, match="不认识的项 pretrain_epochs"):
        check_settings(spec("finger_cnn", {"name": "a", "pretrain_epochs": 5}))
    for bad in ({"mask": 0.95}, {"pretrain_epochs": -1}, {"label_fraction": 0}):
        with pytest.raises(ValueError, match="取值无效"):
            check_settings(spec("cnn", {"name": "a", **bad}))
    for model in contact_cv.DEFAULTS:
        assert check_settings(spec(model, {"name": "a", "label_fraction": 0.25}))
    with pytest.raises(ValueError, match="schema"):
        check_settings({"model": "forest", "candidates": [{"name": "a"}]})
    assert check_settings(spec("cnn", {"name": "a", "positive_weight": "balanced"}))


def test_shipped_settings_files_are_valid_and_cover_the_two_finger_cnn_controls():
    models = {path.stem: read_settings(path) for path in sorted(SETTINGS.glob("*.json"))}
    assert set(models) == set(contact_cv.DEFAULTS)
    assert all(settings["model"] == name for name, settings in models.items())
    switches = {
        (item["options"]["shared"], item["options"]["context"])
        for item in models["finger_cnn"]["candidates"]
    }
    assert {(True, True), (False, True), (True, False)} <= switches
    pretrained = [item["options"]["pretrain_epochs"] for item in models["cnn"]["candidates"]]
    assert 0 in pretrained and max(pretrained) > 0  # The same CNN with and without it.
    # The rule is also compared with a depth-free distance and with one finger at a time.
    rules = [item["options"] for item in models["rules"]["candidates"]]
    assert any(item["flat_index"] and not item["flat_ring"] for item in rules)
    assert any(item["nearest"] for item in rules) and not any(rules[0][key] for key in FLAGS)
    # The temporal CNN is also compared with its hands turned in training, and on distances only.
    options = [item["options"] for item in models["cnn"]["candidates"]]
    assert any(item["rotate"] for item in options) and not options[0]["rotate"]
    assert {"all", "distances"} == {item["inputs"] for item in options}


def test_best_threshold_is_the_exact_f1_optimum():
    rng = np.random.default_rng(3)
    for _ in range(20):
        values = rng.normal(0.4, 0.2, 60).round(2)  # Rounded, so that values repeat.
        labels = (values + rng.normal(0, 0.15, 60) < 0.35).astype(np.int8)
        labels[0] = 1

        def f1(cut):
            called = values <= cut
            hits = np.sum(called & (labels == 1))
            return 2 * hits / (called.sum() + labels.sum())

        best = max(f1(value) for value in np.unique(values))
        assert f1(contact_cv._best_threshold(values, labels)) == pytest.approx(best)


def test_tuning_never_fits_or_selects_with_a_folds_held_out_person(monkeypatch):
    pool = fake_pool()
    calls = []
    real = contact_cv._fit_and_score

    def spy(model, options, seed, x, y, targets, select, keypoints=None):
        assert keypoints is None  # Nothing asked for pretraining here.
        calls.append((owners(x), set(targets), select))
        return real(model, options, seed, x, y, targets, select)

    monkeypatch.setattr(contact_cv, "_fit_and_score", spy)
    settings = check_settings(spec("rules", {"name": "a"}, {"name": "b", "frames": 3}))
    report = run_cv(pool, settings)
    assert report["test"] is None and report["evidence"] == "validation_only"
    # Two candidates, three ways to leave two people out: every fit is for validation.
    assert len(calls) == 6 and all(select for _, _, select in calls)
    for trained, scored, _ in calls:
        assert len(trained) == 1 and len(scored) == 2 and not trained & scored
    for candidate in report["candidates"]:
        fits = {(fit["held_out"], fit["validation"]) for fit in candidate["fits"]}
        assert fits == {(a, b) for a in PEOPLE for b in PEOPLE if a != b}
        for fit in candidate["fits"]:
            assert fit["held_out"] not in fit["trained_on"]
            assert fit["validation"] not in fit["trained_on"]
            assert fit["held_out"] != fit["validation"]


def test_each_fold_selects_by_its_own_validation_and_tests_once_with_that_choice(monkeypatch):
    pool = fake_pool()
    # Candidate "x" is perfect on P01 only; "y" is good on P02 and P03 only.
    wrong = {"x": {"P01": 0.0, "P02": 0.4, "P03": 0.4}, "y": {"P01": 0.4, "P02": 0.1, "P03": 0.1}}
    calls = []

    def fake(model, options, seed, x, y, targets, select, keypoints=None):
        name = "x" if options["frames"] == 1 else "y"
        calls.append((name, owners(x), set(targets), select, options["epochs"]))
        scores = {}
        for person, (_, labels) in targets.items():
            flipped = labels.copy()
            step = round(1 / wrong[name][person]) if wrong[name][person] else 0
            if step:
                flipped[::step] = 1 - flipped[::step]
            scores[person] = flipped.astype(np.float32)
        return scores, {
            "selected_epoch": {person: 4 if person == "P02" else 9 for person in targets}
        }

    monkeypatch.setattr(contact_cv, "_fit_and_score", fake)
    settings = check_settings(
        spec(
            "cnn",
            {"name": "x", "frames": 1, "epochs": 30},
            {"name": "y", "frames": 2, "epochs": 30},
        )
    )
    report = run_cv(pool, settings, final_test=True)
    chosen = {name: item["selected_for"] for name, item in zip("xy", report["candidates"])}
    # Leaving P01 out, its fold only sees P02 and P03, where "y" is better. A choice made
    # on everyone's validation would give every fold the same candidate.
    assert chosen == {"x": ["P02", "P03"], "y": ["P01"]}
    assert report["evidence"] == "final_test"
    tests = [call for call in calls if not call[3]]
    assert [(name, sorted(trained), scored) for name, trained, scored, _, _ in tests] == [
        ("y", ["P02", "P03"], {"P01"}),
        ("x", ["P01", "P03"], {"P02"}),
        ("x", ["P01", "P02"], {"P03"}),
    ]
    # Final fits have no validation person to stop on: they run for the mean epoch that the
    # fold's own validation fits selected (P02 -> 4, the others -> 9).
    assert [call[4] for call in tests] == [round((4 + 9) / 2), 9, round((9 + 4) / 2)]
    assert all(call[4] == 30 for call in calls if call[3])
    folds = report["test"]["folds"]
    assert [(fold["held_out"], fold["candidate"]) for fold in folds] == [
        ("P01", "y"),
        ("P02", "x"),
        ("P03", "x"),
    ]
    assert report["test"]["macro_f1"] == pytest.approx(np.mean([f["macro_f1"] for f in folds]))


def test_pretraining_sees_keypoints_of_the_fitted_people_only_labelled_or_not(monkeypatch):
    pool = fake_pool()
    calls = []

    def fake(model, options, seed, x, y, targets, select, keypoints=None):
        calls.append((options["pretrain_epochs"], owners(x), keypoints, set(targets), select))
        scores = {person: labels.astype(np.float32) for person, (_, labels) in targets.items()}
        return scores, {"selected_epoch": dict.fromkeys(targets, 3)}

    monkeypatch.setattr(contact_cv, "_fit_and_score", fake)
    settings = check_settings(
        spec(
            "cnn",
            {"name": "无预训练", "stride": 1},
            {"name": "预训练", "stride": 1, "pretrain_epochs": 4},
        )
    )
    run_cv(pool, settings, final_test=True)
    assert all(keypoints is None for epochs, _, keypoints, _, _ in calls if not epochs)
    with_keypoints = [call for call in calls if call[0]]
    assert len(with_keypoints) >= 3
    for _, fitted, keypoints, scored, select in with_keypoints:
        assert owners(keypoints) == fitted and not fitted & scored
        # 96 labelled and 24 unlabelled windows for each person fitted on.
        assert len(keypoints) == 120 * len(fitted)


def test_label_fraction_keeps_whole_segments_of_every_kind_of_step():
    pool = fake_pool()
    rows = np.flatnonzero(np.isin(pool.participants, ("P01", "P03")))
    assert np.array_equal(contact_cv._label_subset(pool, rows, 1.0, 1), rows)
    half = contact_cv._label_subset(pool, rows, 0.5, 1)
    assert len(half) == len(rows) // 2 and set(half) <= set(rows)
    assert np.array_equal(half, contact_cv._label_subset(pool, rows, 0.5, 1))

    def segments(chosen):
        return {(pool.recordings[i], pool.rounds[i], pool.steps[i]): 0 for i in chosen}.keys()

    kept = segments(half)
    # Twelve windows per segment: a segment is kept whole or not at all.
    assert len(kept) * 12 == len(half)
    # Each person has four segments of each of the two steps; half of each kind stays.
    assert sorted(step for _, _, step in kept) == [0] * 4 + [1] * 4
    tiny = contact_cv._label_subset(pool, rows, 0.01, 1)
    assert sorted({pool.steps[i] for i in tiny}) == [0, 1]  # Never drops a kind of step.
    assert any(
        set(contact_cv._label_subset(pool, rows, 0.5, seed)) != set(half) for seed in range(2, 8)
    )
    report = run_cv(
        pool, check_settings(spec("rules", {"name": "一半标签", "label_fraction": 0.5}))
    )
    assert {fit["training_windows"] for fit in report["candidates"][0]["fits"]} == {48}


def test_pretraining_learns_to_rebuild_blanked_keypoints_without_touching_the_head():
    torch = pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    rng = np.random.default_rng(0)
    # Windows with structure: a few hidden factors that drift over the eight frames.
    factors = np.cumsum(rng.normal(0, 0.3, (3000, 8, 6)), axis=1)
    x = factors @ rng.normal(0, 1, (6, FEATURE_COUNT)) + rng.normal(0, 0.1, (3000, 8, 67))
    x = ((x - x.mean(axis=(0, 1))) / x.std(axis=(0, 1))).astype(np.float32)
    settings = dict(epochs=5, mask=0.3, learning_rate=0.002, batch_size=128, seed=2)
    network = nets.temporal_cnn(8, 32, seed=2)
    before = {key: value.clone() for key, value in network.state_dict().items()}
    details = nets.pretrain(network, x, **settings)
    errors = [row["masked_error"] for row in details["history"]]
    assert details["windows"] == 3000 and len(errors) == 5
    assert errors[-1] < 0.5 * errors[0]  # A blanked value is no longer guessed as the mean.
    after = network.state_dict()
    assert not torch.equal(before["conv1.weight"], after["conv1.weight"])
    assert not torch.equal(before["conv2.weight"], after["conv2.weight"])
    assert torch.equal(before["head.weight"], after["head.weight"])
    again = nets.temporal_cnn(8, 32, seed=2)
    nets.pretrain(again, x, **settings)
    assert torch.equal(again.state_dict()["conv1.weight"], after["conv1.weight"])
    other = nets.temporal_cnn(8, 32, seed=2)
    nets.pretrain(other, x, **{**settings, "mask": 0.6})
    assert not torch.equal(other.state_dict()["conv1.weight"], after["conv1.weight"])

    # What is blanked is really hidden from the network. Here every frame's interval feature
    # is independent noise and everything else is zero, so a blanked value cannot be worked
    # out from what is left. A network that still saw it would simply copy it.
    noise = np.zeros((3000, 8, FEATURE_COUNT), dtype=np.float32)
    noise[:, :, 66] = rng.normal(0, 1, (3000, 8))
    hidden = nets.pretrain(
        nets.temporal_cnn(8, 32, seed=2),
        noise,
        **{**settings, "epochs": 12, "learning_rate": 0.005},
    )
    # The error is averaged over all blanked values; only 1 in 67 of them is not zero.
    assert hidden["history"][-1]["masked_error"] * FEATURE_COUNT > 0.9


def test_variation_options_belong_to_the_two_cnns_and_input_choice_to_the_temporal_one():
    filled = check_settings(spec("cnn", {"name": "a"}))["candidates"][0]["options"]
    assert (filled["inputs"], filled["rotate"], filled["scale_jitter"], filled["noise"]) == (
        "all",
        0,
        0.0,
        0.0,
    )
    assert check_settings(
        spec("cnn", {"name": "a", "rotate": 180, "scale_jitter": 0.15, "noise": 0.03})
    )
    assert check_settings(spec("cnn", {"name": "a", "inputs": "distances", "scale_jitter": 0.2}))
    for bad in (
        {"rotate": 181},
        {"rotate": -1},
        {"scale_jitter": 0.6},
        {"noise": -0.1},
        {"inputs": "ratios"},
        {"inputs": 1},
    ):
        with pytest.raises(ValueError, match="取值无效"):
            check_settings(spec("cnn", {"name": "a", **bad}))
    # Distances alone leave no coordinates to turn, to disturb or to blank and rebuild.
    for clash in ({"rotate": 30}, {"noise": 0.01}, {"pretrain_epochs": 2}):
        with pytest.raises(ValueError, match=f"inputs 是 distances.*{next(iter(clash))}"):
            check_settings(spec("cnn", {"name": "a", "inputs": "distances", **clash}))
    # The finger-shared CNN can be trained on varied hands too; it always sees coordinates.
    varied = {"rotate": 180, "scale_jitter": 0.15, "noise": 0.03}
    fingers = check_settings(spec("finger_cnn", {"name": "a", **varied}))
    assert {key: fingers["candidates"][0]["options"][key] for key in varied} == varied
    plain = check_settings(spec("finger_cnn", {"name": "a"}))["candidates"][0]["options"]
    assert (plain["rotate"], plain["scale_jitter"], plain["noise"]) == (0, 0.0, 0.0)
    with pytest.raises(ValueError, match="取值无效"):
        check_settings(spec("finger_cnn", {"name": "a", "rotate": 181}))
    for model, option in (
        ("finger_cnn", {"inputs": "distances"}),
        ("forest", {"rotate": 30}),
        ("forest", {"inputs": "distances"}),
        ("rules", {"rotate": 30}),
        ("rules", {"inputs": "distances"}),
    ):
        with pytest.raises(ValueError, match="不认识的项"):
            check_settings(spec(model, {"name": "a", **option}))


def test_augmenter_turns_whole_hands_and_keeps_what_a_turn_cannot_change():
    torch = pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    rng = np.random.default_rng(1)
    raw = rng.normal(0, 1, (40, 8, FEATURE_COUNT)).astype(np.float32)
    raw[:, :, 63:66] = np.abs(raw[:, :, 63:66]) + 0.1
    mean, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    x = torch.from_numpy((raw - mean) / scale)

    def varied(**options):
        return nets.augmenter(mean, scale, **options)(x).numpy() * scale + mean

    def cloud(values):  # Every landmark of every frame of a window, as one set of points.
        return values[:, :, :63].reshape(len(values), -1, 3).astype(np.float64)

    assert nets.augmenter(mean, scale) is None  # Nothing to vary.

    turned = varied(rotate=180, seed=3)
    assert np.allclose(turned[:, :, 63:], raw[:, :, 63:], atol=1e-4)  # Ratios and interval.
    assert not np.allclose(turned[:, :, :63], raw[:, :, :63], atol=0.2)
    before, after = cloud(raw), cloud(turned)
    # All frames of a window turn together about the palm centre: every inner product stays.
    assert np.allclose(
        before @ before.transpose(0, 2, 1), after @ after.transpose(0, 2, 1), atol=2e-3
    )
    turns = [np.linalg.lstsq(before[i], after[i], rcond=None)[0].T for i in range(40)]
    assert np.allclose([np.linalg.det(turn) for turn in turns], 1, atol=1e-3)  # No mirror.
    assert not np.allclose(turns[0], turns[1], atol=0.1)  # Each window has its own turn.

    def angle(turn):
        return np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))

    slight = cloud(varied(rotate=10, seed=3))
    angles = [angle(np.linalg.lstsq(before[i], slight[i], rcond=None)[0].T) for i in range(40)]
    assert 1 < np.mean(angles) and max(angles) <= 30.5  # At most 10 degrees about each axis.

    sized = varied(scale_jitter=0.2, seed=3)
    factor = sized[:, :, 63:66] / raw[:, :, 63:66]
    assert np.allclose(factor, factor[:, :1, :1], atol=2e-3)  # One factor for a window ...
    assert 0.8 - 1e-3 <= factor.min() and factor.max() <= 1.2 + 1e-3 and factor.std() > 0.05
    assert np.allclose(sized[:, :, :63], raw[:, :, :63] * factor[:, :1, :1], atol=2e-3)
    assert np.allclose(sized[:, :, 66], raw[:, :, 66], atol=1e-4)  # ... never the interval.

    change = varied(noise=0.05, seed=3) - raw
    assert np.allclose(change[:, :, 63:], 0, atol=1e-4)
    assert 0.045 < change[:, :, :63].std() < 0.055

    both = dict(rotate=90, scale_jitter=0.1, noise=0.02)
    assert np.array_equal(varied(**both, seed=3), varied(**both, seed=3))
    assert not np.array_equal(varied(**both, seed=3), varied(**both, seed=4))
    vary = nets.augmenter(mean, scale, **both, seed=3)
    assert not torch.equal(vary(x), vary(x))  # A window differs every time it comes up.

    # Windows of the three ratios and the interval can only change size.
    small_mean, small_scale = mean[63:], scale[63:]
    small = torch.from_numpy((raw[:, :, 63:] - small_mean) / small_scale)
    resized = nets.augmenter(small_mean, small_scale, scale_jitter=0.2, seed=3)(small).numpy()
    resized = resized * small_scale + small_mean
    factor = resized[:, :, :3] / raw[:, :, 63:66]
    assert np.allclose(factor, factor[:, :1, :1], atol=2e-3) and factor.std() > 0.05
    assert np.allclose(resized[:, :, 3], raw[:, :, 66], atol=1e-4)
    with pytest.raises(ValueError, match="没有坐标"):
        nets.augmenter(small_mean, small_scale, rotate=30)
    with pytest.raises(ValueError, match="特征数不对"):
        nets.augmenter(mean[:10], scale[:10], noise=0.1)


def test_only_training_batches_are_varied():
    pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (100, 8, FEATURE_COUNT)).astype(np.float32)
    y = (x[:, -1, 63:66] > 0).astype(np.int8)
    sizes = []

    def blank(batch):
        sizes.append(len(batch))
        return batch * 0

    fit = dict(epochs=3, learning_rate=0.01, batch_size=16, weight_decay=0.0, positive_weight=1.0)
    network = nets.temporal_cnn(8, 8, seed=1)
    details, _ = nets.train(network, x, y, **fit, seed=1, validation={"v": (x, y)}, augment=blank)
    assert sizes == [16, 16, 16, 16, 16, 16, 4] * 3  # Every training batch, nothing else.
    seen = nets.temporal_cnn(8, 8, seed=1)
    plain, _ = nets.train(seen, x, y, **fit, seed=1, validation={"v": (x, y)})
    # Trained on blanks the network learns nothing about x; validation still shows it x.
    blind = [row["validation_loss"]["v"] for row in details["history"]]
    sighted = [row["validation_loss"]["v"] for row in plain["history"]]
    assert sighted[-1] < 0.8 * sighted[0] and sighted[-1] < blind[-1]

    sizes.clear()
    settings = dict(epochs=2, mask=0.3, learning_rate=0.002, batch_size=32, seed=2)
    rebuilt = nets.pretrain(nets.temporal_cnn(8, 8, seed=1), x, **settings, augment=blank)
    assert sizes == [32, 32, 32, 4] * 2
    # What must be rebuilt is the varied window, here all zeros, which is easy.
    usual = nets.pretrain(nets.temporal_cnn(8, 8, seed=1), x, **settings)
    assert rebuilt["history"][-1]["masked_error"] < 0.2 * usual["history"][-1]["masked_error"]


def test_finger_views_are_cut_from_a_hand_that_was_varied_as_a_whole():
    torch = pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    rng = np.random.default_rng(6)
    raw = rng.normal(0, 1, (30, 8, FEATURE_COUNT)).astype(np.float32)
    raw[:, :, 63:66] = np.abs(raw[:, :, 63:66]) + 0.1
    views = nets.finger_inputs(raw)
    assert torch.equal(nets.finger_views(torch.from_numpy(raw)), torch.from_numpy(views))
    mean, scale = views.mean(axis=(0, 1, 2)), views.std(axis=(0, 1, 2))
    assert nets.finger_augmenter(mean, scale) is None  # Nothing to vary.
    with pytest.raises(ValueError, match="手指视图的特征数不对"):
        nets.finger_augmenter(mean[:5], scale[:5], rotate=30)

    def varied(**options):
        out = nets.finger_augmenter(mean, scale, **options)(torch.from_numpy(raw))
        return out.numpy() * scale + mean  # Back to the units of the views.

    # With next to no variation the result is the ordinary normalized view of the same hand.
    assert np.allclose(varied(scale_jitter=1e-9, seed=1), views, atol=1e-4)

    turned = varied(rotate=180, seed=1)
    assert turned.shape == views.shape == (30, 3, 8, 29)
    assert not np.allclose(turned[..., :24], views[..., :24], atol=0.2)
    assert np.allclose(turned[..., 27:], views[..., 27:], atol=1e-4)  # Ratio and interval.
    # One turn for the whole hand: the thumb is the same in all three views of a window, and
    # within a view every inner product among thumb joints, finger joints and their gap stays.
    assert np.allclose(turned[:, 0, :, :12], turned[:, 1, :, :12], atol=1e-5)
    assert np.allclose(turned[:, 0, :, :12], turned[:, 2, :, :12], atol=1e-5)

    def gram(values):
        points = values[..., :27].reshape(30, 3, 8 * 9, 3).astype(np.float64)
        return points @ points.transpose(0, 1, 3, 2)

    assert np.allclose(gram(turned), gram(views), atol=3e-3)
    # The gap feature is still fingertip minus thumb tip of the turned hand.
    assert np.allclose(turned[..., 24:27], turned[..., 21:24] - turned[..., 9:12], atol=1e-4)

    sized = varied(scale_jitter=0.2, seed=1)
    factor = sized[..., 27] / views[..., 27]
    assert np.allclose(factor, factor[:, :1, :1], atol=2e-3) and factor.std() > 0.05
    assert np.allclose(sized[..., :27], views[..., :27] * factor[:, :1, :1, None], atol=3e-3)
    assert np.array_equal(
        varied(rotate=90, noise=0.02, seed=4), varied(rotate=90, noise=0.02, seed=4)
    )


def test_finger_cnn_with_varied_hands_runs_the_whole_protocol():
    pytest.importorskip("torch")
    base = {"channels": 8, "epochs": 10, "learning_rate": 0.01, "batch_size": 16, "stride": 1}
    settings = check_settings(
        spec(
            "finger_cnn",
            {"name": "原样", **base},
            {"name": "旋转", **base, "rotate": 180},
            {"name": "都用", **base, "rotate": 90, "scale_jitter": 0.1, "noise": 0.01},
        )
    )
    report = run_cv(fake_pool(), settings, final_test=True)
    plain, turned, everything = report["candidates"]
    assert plain["parameters"] == turned["parameters"] == everything["parameters"]
    # Same seed and data: turning the hands in training is the only difference between these.
    assert turned["validation"] != plain["validation"]
    assert turned["validation"]["macro_f1"] > 0.8 and everything["validation"]["macro_f1"] > 0.6
    assert all(np.isfinite(fold["macro_f1"]) for fold in report["test"]["folds"])
    again = run_cv(fake_pool(), settings)
    assert [item["validation"] for item in again["candidates"]] == [
        item["validation"] for item in report["candidates"]
    ]
    text = markdown(report)
    assert text.count("rotate=180") == 1 and text.count("rotate=") == 2
    assert text.count("scale_jitter=0.1") == 1 and text.count("noise=0.01") == 1


def test_cnn_on_distances_only_and_with_varied_hands_runs_the_whole_protocol():
    pytest.importorskip("torch")
    base = {"channels": 8, "epochs": 12, "learning_rate": 0.01, "batch_size": 16, "stride": 1}
    settings = check_settings(
        spec(
            "cnn",
            {"name": "全部", **base},
            {"name": "只看距离比", **base, "inputs": "distances"},
            {"name": "旋转", **base, "rotate": 180},
            {"name": "旋转＋预训练", **base, "rotate": 180, "pretrain_epochs": 3},
            {"name": "都用", **base, "rotate": 90, "scale_jitter": 0.1, "noise": 0.01},
        )
    )
    report = run_cv(fake_pool(), settings, final_test=True)
    whole, distances, turned, both, everything = report["candidates"]
    assert distances["parameters"] < whole["parameters"] == turned["parameters"]
    assert distances["options"]["inputs"] == "distances" and turned["options"]["rotate"] == 180
    # The fake pool's labels are in the distances, which a turn leaves alone.
    assert distances["validation"]["macro_f1"] > 0.9 and turned["validation"]["macro_f1"] > 0.8
    assert everything["validation"]["macro_f1"] > 0.6
    # Same seed and data: turning the hands in training is the only difference between these.
    assert turned["validation"] != whole["validation"]
    assert both["pretraining"]["windows"] == 120
    assert all(np.isfinite(fold["macro_f1"]) for fold in report["test"]["folds"])
    again = run_cv(fake_pool(), settings)
    assert [item["validation"] for item in again["candidates"]] == [
        item["validation"] for item in report["candidates"]
    ]
    json.dumps(report, allow_nan=False)
    text = markdown(report)
    # Shown only for the candidates that use them.
    assert text.count("inputs=distances") == 1 and "inputs=all" not in text
    assert text.count("rotate=180") == 2 and text.count("rotate=") == 3
    assert text.count("scale_jitter=0.1") == 1 and text.count("scale_jitter=") == 1
    assert text.count("noise=0.01") == 1 and text.count("noise=") == 1


def test_pretrained_cnn_runs_the_whole_protocol_and_reports_the_reconstruction():
    pytest.importorskip("torch")
    base = {"channels": 8, "epochs": 12, "learning_rate": 0.01, "batch_size": 16, "stride": 1}
    settings = check_settings(
        spec("cnn", {"name": "无预训练", **base}, {"name": "预训练", **base, "pretrain_epochs": 3})
    )
    report = run_cv(fake_pool(), settings, final_test=True)
    plain, pretrained = report["candidates"]
    assert plain["pretraining"] is None
    assert pretrained["pretraining"]["windows"] == 120  # One person: 96 labelled + 24 not.
    assert np.isfinite(pretrained["pretraining"]["last_masked_error"])
    assert pretrained["options"]["pretrain_epochs"] == 3
    # Same seed and data: any difference in the scores comes from the pretraining.
    assert plain["validation"]["macro_f1"] != pretrained["validation"]["macro_f1"]
    assert pretrained["validation"]["macro_f1"] > 0.8
    json.dumps(report, allow_nan=False)
    text = markdown(report)
    assert "预训练用了 120 个窗口，遮住部分的重建误差从" in text
    assert text.count("mask=") == 1  # Shown only where pretraining is on.


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(labels="draft"), "草稿标签不能用来跑测试结果"),
        (dict(source="synthetic"), "合成数据默认拒绝"),
        (dict(people=("P01", "P02")), "至少需要 3 位参与者"),
    ],
)
def test_runs_that_would_mislead_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        run_cv(fake_pool(**change), check_settings(spec("rules")), final_test=True)


def test_a_person_without_both_labels_or_too_many_frames_is_refused():
    pool = fake_pool()
    pool.y[pool.participants == "P02", 2] = 0
    with pytest.raises(ValueError, match="P02 的无名指标签只有一种取值"):
        run_cv(pool, check_settings(spec("rules")))
    with pytest.raises(ValueError, match="frames 不能超过 8"):
        run_cv(fake_pool(), check_settings(spec("forest", {"name": "a", "frames": 9})))


def test_rules_and_forest_learn_the_separable_fixture_and_reports_are_plain_json(tmp_path):
    pool = fake_pool()
    for model, candidate in (
        ("rules", {"name": "中位数", "frames": 3, "statistic": "median"}),
        ("forest", {"name": "小", "trees": 12, "depth": 4, "stride": 2}),
    ):
        settings = check_settings(spec(model, candidate))
        report = run_cv(pool, settings, final_test=True)
        assert report["candidates"][0]["validation"]["macro_f1"] > 0.95
        assert report["test"]["macro_f1"] > 0.95 and len(report["test"]["folds"]) == 3
        assert report["fixed_rules_validation"]["macro_f1"] == 1.0  # The fixture's rules are y.
        assert set(report["candidates"][0]["validation"]["rounds"]) == {"1", "2", "3", "4"}
        text = write_report(report, settings, tmp_path / model)
        saved = json.loads((tmp_path / model / "report.json").read_text(encoding="utf-8"))
        assert saved == json.loads(json.dumps(report)) and saved["threshold"] == 0.5
        assert (tmp_path / model / "report.md").read_text(encoding="utf-8") == text
        assert json.loads((tmp_path / model / "settings.json").read_text(encoding="utf-8")) == (
            settings
        )
        assert "## 测试" in text and "| P02 | " in text
        assert "第 1 轮：手掌正对镜头" in text and "捏住拇指和中指，小幅移动手或手腕" in text
        with pytest.raises(ValueError, match="输出目录已有内容"):
            write_report(report, settings, tmp_path / model)


def test_rule_can_measure_in_the_image_plane_and_let_only_the_nearer_finger_touch():
    filled = check_settings(spec("rules", {"name": "a"}))["candidates"][0]["options"]
    assert not any(filled[key] for key in FLAGS)
    for bad in ({"flat_index": 1}, {"nearest": "yes"}, {"flat_ring": None}):
        with pytest.raises(ValueError, match="取值无效"):
            check_settings(spec("rules", {"name": "a", **bad}))
    for model in ("forest", "cnn", "finger_cnn"):
        for option in ({"flat_index": True}, {"nearest": True}):
            with pytest.raises(ValueError, match="不认识的项"):
                check_settings(spec(model, {"name": "a", **option}))

    def options(**changes):
        return check_settings(spec("rules", {"name": "a", **changes}))["candidates"][0]["options"]

    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (60, 8, FEATURE_COUNT)).astype(np.float32)
    points = x[:, :, :63].reshape(60, 8, 21, 3)

    def flat(tip, frames):  # Thumb tip to a fingertip without the depth coordinate.
        return np.linalg.norm(points[:, -frames:, 4, :2] - points[:, -frames:, tip, :2], axis=-1)

    assert np.array_equal(contact_cv._distances(x, options()), x[:, -1, 63:66])
    mixed = contact_cv._distances(x, options(flat_index=True, frames=3, statistic="median"))
    assert np.allclose(mixed[:, 1], np.median(flat(8, 3), axis=1), atol=1e-6)
    assert np.array_equal(mixed[:, [0, 2]], np.median(x[:, -3:, 63:66], axis=1)[:, [0, 2]])
    every = contact_cv._distances(x, options(flat_middle=True, flat_index=True, flat_ring=True))
    assert np.allclose(every, np.stack([flat(tip, 1)[:, 0] for tip in (12, 8, 16)], axis=1))

    # Depth that misleads: the index ratio is noise, the distance in the image carries the label.
    y = rng.permutation(np.tile(np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]), (15, 1)))
    x[:, :, 63:66] = np.where(y[:, None, :] == 1, 0.1, 0.6)
    x[:, :, 64] = rng.uniform(0.1, 0.6, (60, 8))
    x[:, :, 12:14] = 0  # Thumb tip, x and y.
    x[:, :, 24] = np.where(y[:, None, 1] == 1, 0.1, 0.6)  # Index fingertip, x.
    x[:, :, 25] = 0
    target = {"t": (x, y)}
    by_ratio, details = contact_cv._fit_and_score("rules", options(), 0, x, y, target, False)
    in_plane, _ = contact_cv._fit_and_score(
        "rules", options(flat_index=True), 0, x, y, target, False
    )
    assert set(details["thresholds"]) == set(CHANNELS)
    assert np.array_equal(in_plane["t"], y) and not np.array_equal(by_ratio["t"][:, 1], y[:, 1])
    assert np.array_equal(by_ratio["t"][:, [0, 2]], y[:, [0, 2]])

    # Index and ring both under their cuts: the thumb is on the nearer one. Ties go to the index.
    x[:, :, 63:66] = np.where(y[:, None, :] == 1, 0.1, 0.6)
    near = np.full((4, 8, FEATURE_COUNT), 0.6, dtype=np.float32)
    near[:, :, 64] = np.array([0.10, 0.20, 0.15, 0.10])[:, None]
    near[:, :, 65] = np.array([0.20, 0.10, 0.15, 0.60])[:, None]
    unused = np.zeros((4, 3), dtype=np.int8)
    plain, _ = contact_cv._fit_and_score("rules", options(), 0, x, y, {"n": (near, unused)}, False)
    one, _ = contact_cv._fit_and_score(
        "rules", options(nearest=True), 0, x, y, {"n": (near, unused)}, False
    )
    assert plain["n"][:, 1:].tolist() == [[1, 1], [1, 1], [1, 1], [1, 0]]
    assert one["n"][:, 1:].tolist() == [[1, 0], [0, 1], [1, 0], [1, 0]]
    assert one["n"][:, 0].tolist() == [0, 0, 0, 0]  # The middle finger is never part of this.

    report = run_cv(
        fake_pool(),
        check_settings(spec("rules", {"name": "近的", "nearest": True}, {"name": "原样"})),
    )
    text = markdown(report)
    assert text.count("nearest=True") == 1 and "flat_" not in text and "nearest=False" not in text


def test_report_says_when_numbers_are_not_results():
    settings = check_settings(spec("rules"))
    draft = markdown(run_cv(fake_pool(labels="draft"), settings))
    assert "草稿标签，仅供试跑" in draft and "## 测试" not in draft
    pool = fake_pool(source="synthetic")
    pool.metadata["recordings"] = [{"protocol": "synthetic"} for _ in PEOPLE]
    synthetic = markdown(run_cv(pool, settings, allow_synthetic=True))
    assert "合成数据，只验证程序能跑通" in synthetic and "手掌正对镜头" not in synthetic
    assert "草稿" not in markdown(run_cv(fake_pool(), settings))
    assert "短于默认值的留空" not in markdown(run_cv(fake_pool(), settings))
    pool = fake_pool()
    pool.metadata["recordings"][1]["short_margins"] = True
    assert "有 1 段录制是用短于默认值的留空复核的" in markdown(run_cv(pool, settings))


def test_finger_inputs_hold_only_the_thumb_and_one_finger():
    nets = pytest.importorskip("pinchpilot.contact_nets")
    x = np.arange(2 * 3 * FEATURE_COUNT, dtype=np.float32).reshape(2, 3, FEATURE_COUNT)
    views = nets.finger_inputs(x)
    assert views.shape == (2, 3, 3, nets.FINGER_FEATURES)
    points = x[:, :, :63].reshape(2, 3, 21, 3)
    for channel, joints in zip(range(3), ((9, 10, 11, 12), (5, 6, 7, 8), (13, 14, 15, 16))):
        view = views[:, channel]
        assert np.array_equal(view[:, :, :12], points[:, :, 1:5].reshape(2, 3, 12))
        assert np.array_equal(view[:, :, 12:24], points[:, :, list(joints)].reshape(2, 3, 12))
        assert np.array_equal(view[:, :, 24:27], points[:, :, joints[-1]] - points[:, :, 4])
        assert np.array_equal(view[:, :, 27], x[:, :, 63 + channel])
        assert np.array_equal(view[:, :, 28], x[:, :, 66])
    assert CHANNELS == ("middle", "index", "ring")  # The joint order above follows this.


def test_shared_finger_weights_treat_the_three_fingers_alike():
    torch = pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    views = torch.randn(5, 3, 8, nets.FINGER_FEATURES)
    swapped = views[:, [1, 0, 2]]
    for context in (True, False):
        shared = nets.finger_cnn(8, 12, seed=1, shared=True, context=context)
        separate = nets.finger_cnn(8, 12, seed=1, shared=False, context=context)
        with torch.inference_mode():
            # Swapping two fingers' inputs swaps their outputs, only with shared weights.
            assert torch.allclose(shared(swapped), shared(views)[:, [1, 0, 2]], atol=1e-6)
            assert not torch.allclose(separate(swapped), separate(views)[:, [1, 0, 2]], atol=1e-3)
        assert nets.parameter_count(separate) > 2.9 * nets.parameter_count(shared)
    alone = nets.finger_cnn(8, 12, seed=1, context=False)
    together = nets.finger_cnn(8, 12, seed=1, context=True)
    changed = views.clone()
    changed[:, 2] += 1.0
    with torch.inference_mode():
        # Without context an output sees its own finger only; with it, the others too.
        assert torch.equal(alone(changed)[:, :2], alone(views)[:, :2])
        assert not torch.equal(together(changed)[:, :2], together(views)[:, :2])
    assert nets.parameter_count(nets.finger_cnn(8, 24, 0)) == 5644
    assert nets.parameter_count(nets.temporal_cnn(8, 32, 0)) == 12739


def test_one_training_run_keeps_a_best_epoch_for_each_validation_set():
    torch = pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (256, 8, FEATURE_COUNT)).astype(np.float32)
    y = (x[:, -1, :3] > 0).astype(np.int8)
    easy = (x[:64], y[:64])
    reversed_labels = (x[64:128], 1 - y[64:128])  # Gets worse as the fit gets better.
    settings = dict(
        epochs=6, learning_rate=0.01, batch_size=32, weight_decay=0.0, positive_weight=1.0, seed=3
    )
    network = nets.temporal_cnn(8, 8, seed=3)
    details, states = nets.train(
        network, x, y, validation={"easy": easy, "reversed": reversed_labels}, **settings
    )
    assert details["selected_epoch"]["reversed"] == 1 < details["selected_epoch"]["easy"]
    assert set(states) == {"easy", "reversed"} and len(details["history"]) == 6
    last = nets.predict(network, x)
    network.load_state_dict(states["reversed"])
    assert not np.allclose(nets.predict(network, x), last)
    again = nets.temporal_cnn(8, 8, seed=3)
    nets.train(again, x, y, **settings)
    assert np.array_equal(nets.predict(again, x), last)  # Same seed, same weights.
    assert torch.get_num_threads() >= 1


@pytest.mark.parametrize("model", ["cnn", "finger_cnn"])
def test_networks_run_the_whole_protocol(model):
    pytest.importorskip("torch")
    settings = check_settings(
        spec(
            model,
            {
                "name": "小",
                "channels": 8,
                "epochs": 12,
                "learning_rate": 0.01,
                "batch_size": 16,
                "stride": 1,
            },
        )
    )
    report = run_cv(fake_pool(), settings, final_test=True)
    candidate = report["candidates"][0]
    assert candidate["parameters"] > 0 and candidate["validation"]["macro_f1"] > 0.8
    epochs = [fit["selected_epoch"] for fit in candidate["fits"]]
    assert len(epochs) == 6 and all(1 <= value <= 12 for value in epochs)
    for fold in report["test"]["folds"]:
        chosen = [
            f["selected_epoch"] for f in candidate["fits"] if f["held_out"] == fold["held_out"]
        ]
        assert fold["options"]["epochs"] == max(1, round(float(np.mean(chosen))))
    assert report["test"]["macro_f1"] > 0.8
    json.dumps(report, allow_nan=False)


def test_commands_run_on_the_synthetic_pool(tmp_path, capsys):
    data, cache = tmp_path / "pool", tmp_path / "cache"
    assert main(["ml-fixture", "--output", str(data), "--pool"]) == 0
    assert main(["ml-pool", str(data), "--cache", str(cache)]) == 0
    listing = json.loads(capsys.readouterr().out.split("\n", 4)[-1])
    assert listing["labels"] == "reviewed" and len(listing["participants"]) == 3
    command = [
        "ml-cv",
        str(data),
        "--settings",
        str(SETTINGS / "rules.json"),
        "--cache",
        str(cache),
    ]
    assert main([*command, "--output", str(tmp_path / "refused")]) == 2
    assert "合成数据默认拒绝" in capsys.readouterr().err
    assert not (tmp_path / "refused").exists()
    output = tmp_path / "run"
    assert main([*command, "--output", str(output), "--allow-synthetic"]) == 0
    shown = capsys.readouterr()
    assert "# 调过阈值的规则：验证结果" in shown.out and "## 测试" not in shown.out
    fits = 3 * len(read_settings(SETTINGS / "rules.json")["candidates"])
    assert f"[{fits}/{fits}]" in shown.err
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["test"] is None and report["evidence"] == "synthetic_engineering"
    assert main([*command, "--output", str(output), "--allow-synthetic"]) == 2
    assert "输出目录已有内容" in capsys.readouterr().err
