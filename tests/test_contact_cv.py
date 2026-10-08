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
            "recordings": [{"protocol": "pinchpilot-guided-session-1"} for _ in people],
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


def test_report_says_when_numbers_are_not_results():
    settings = check_settings(spec("rules"))
    draft = markdown(run_cv(fake_pool(labels="draft"), settings))
    assert "草稿标签，仅供试跑" in draft and "## 测试" not in draft
    pool = fake_pool(source="synthetic")
    pool.metadata["recordings"] = [{"protocol": "synthetic"} for _ in PEOPLE]
    synthetic = markdown(run_cv(pool, settings, allow_synthetic=True))
    assert "合成数据，只验证程序能跑通" in synthetic and "手掌正对镜头" not in synthetic
    assert "草稿" not in markdown(run_cv(fake_pool(), settings))


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
    assert "[9/9]" in shown.err
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["test"] is None and report["evidence"] == "synthetic_engineering"
    assert main([*command, "--output", str(output), "--allow-synthetic"]) == 2
    assert "输出目录已有内容" in capsys.readouterr().err
