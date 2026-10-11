"""Fitted rules and temporal CNNs for live control: export, the NumPy runtime, the frame feed."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pinchpilot import contact_cv, contact_live
from pinchpilot.cli import main
from pinchpilot.contact_cv import check_settings, fit_for_live
from pinchpilot.contact_data import FEATURE_COUNT
from pinchpilot.contact_live import LiveContacts, LiveModel, load_live_model, save_live_model
from pinchpilot.tripod_demo import synthetic_tripod

sys.path.insert(0, str(Path(__file__).parent))
from test_contact_cv import SETTINGS, fake_pool, spec  # noqa: E402

RULE = {"name": "规则", "flat_index": True, "nearest": True, "frames": 3, "statistic": "median"}
NET = {"channels": 8, "epochs": 6, "learning_rate": 0.01, "batch_size": 16, "stride": 1}


def windows(count=40, seed=3):
    return np.random.default_rng(seed).normal(0, 1, (count, 8, FEATURE_COUNT)).astype(np.float32)


def test_exported_rule_decides_windows_exactly_as_the_comparison_does(tmp_path):
    pool = fake_pool()
    settings = check_settings(spec("rules", RULE, {"name": "原样"}))
    record, arrays = fit_for_live(pool, settings, "规则", exclude="P02")
    assert arrays is None and record["model"] == "rules" and record["candidate"] == "规则"
    assert record["trained_on"] == ["P01", "P03"] and record["excluded"] == "P02"
    assert record["training_windows"] == 192 and record["evidence"] == "fitted_not_scored"
    assert record["options"]["flat_index"] is True and len(record["thresholds"]) == 3
    json.dumps(record, allow_nan=False)

    # The same fit, scored the way the comparison scores it.
    rows = np.flatnonzero(pool.participants != "P02")
    x = windows()
    expected, details = contact_cv._fit_and_score(
        "rules", record["options"], 42, pool.windows(rows), pool.y[rows], {"t": (x, None)}, False
    )
    assert record["thresholds"] == [details["thresholds"][name] for name in contact_cv.CHANNELS]
    model = load_live_model(save_live_model(record, arrays, tmp_path / "rule"))
    assert not (tmp_path / "rule" / "weights.npz").exists()
    assert np.array_equal(model.decide(x), expected["t"].astype(bool))
    assert np.array_equal(model.scores(x), expected["t"])

    everyone, _ = fit_for_live(pool, settings, "原样")
    assert everyone["trained_on"] == ["P01", "P02", "P03"] and everyone["excluded"] is None
    assert everyone["training_windows"] == 288


@pytest.mark.parametrize(
    "candidate",
    [
        {"name": "网络", **NET},
        {"name": "网络", **NET, "inputs": "distances", "frames": 4},
        {"name": "网络", **NET, "frames": 2, "rotate": 180, "scale_jitter": 0.1, "noise": 0.01},
        {"name": "网络", **NET, "pretrain_epochs": 2},
    ],
)
def test_exported_cnn_gives_the_scores_of_the_trained_network_without_torch(tmp_path, candidate):
    torch = pytest.importorskip("torch")
    from pinchpilot import contact_nets as nets

    pool = fake_pool()
    settings = check_settings(spec("cnn", candidate))
    record, arrays = fit_for_live(pool, settings, "网络", exclude="P01", epochs=4)
    assert record["options"]["epochs"] == 4 and record["trained_on"] == ["P02", "P03"]
    assert set(arrays) == {*contact_live.WEIGHTS, "mean", "scale"}
    assert record["parameters"] == sum(arrays[name].size for name in contact_live.WEIGHTS)

    # The scores the comparison computes for the same fit, with PyTorch.
    options = record["options"]
    rows, keypoints = contact_cv._training(
        pool,
        {p: np.flatnonzero(pool.participants == p) for p in ("P01", "P02", "P03")},
        ("P01",),
        options,
        42,
    )
    x = windows()
    expected, _ = contact_cv._fit_and_score(
        "cnn",
        options,
        42,
        pool.windows(rows),
        pool.y[rows],
        {"t": (x, None)},
        False,
        keypoints=keypoints,
    )
    model = load_live_model(save_live_model(record, arrays, tmp_path / "cnn"))
    assert np.allclose(model.scores(x), expected["t"], atol=1e-5)
    assert np.array_equal(model.decide(x), expected["t"] >= 0.5)
    assert 0.02 < model.decide(x).mean() < 0.98  # Both answers occur; the check is not empty.

    # And directly against the network rebuilt from the saved weights.
    first = 63 if options["inputs"] == "distances" else 0
    network = nets.temporal_cnn(options["frames"], options["channels"], 0, FEATURE_COUNT - first)
    network.load_state_dict({k: torch.from_numpy(arrays[k]) for k in contact_live.WEIGHTS})
    inputs = (x[:, -options["frames"] :, first:].astype(np.float64) - arrays["mean"]) / arrays[
        "scale"
    ]
    assert np.allclose(model.scores(x), nets.predict(network, inputs.astype(np.float32)), atol=1e-5)


def test_live_runtime_needs_neither_torch_nor_sklearn(tmp_path):
    record = {
        "schema": contact_live.SCHEMA,
        "model": "rules",
        "options": {"frames": 1, "statistic": "last"},
        "thresholds": [0.3, 0.3, 0.3],
        "feature_version": "contact-v1",
        "window_config": {"frames": 8, "max_gap_seconds": 0.12, "max_window_seconds": 0.8},
        "channels": ["middle", "index", "ring"],
        "threshold": 0.5,
    }
    save_live_model(record, None, tmp_path / "m")
    code = (
        "import sys\n"
        "from pinchpilot.contact_live import load_live_model\n"
        f"model = load_live_model({str(tmp_path / 'm')!r})\n"
        "assert model.kind == 'rules'\n"
        "print(sorted(name for name in ('torch', 'sklearn') if name in sys.modules))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_frames_become_one_decision_each_once_a_window_is_full():
    record = {
        "schema": contact_live.SCHEMA,
        "model": "rules",
        "options": {"frames": 1, "statistic": "last"},
        "thresholds": [0.36, 0.36, 0.36],
        "feature_version": "contact-v1",
        "window_config": {"frames": 8, "max_gap_seconds": 0.12, "max_window_seconds": 0.8},
        "channels": ["middle", "index", "ring"],
        "threshold": 0.5,
    }
    live = LiveContacts(LiveModel(record))

    def frame(i, grip=True, index=0.85):
        return synthetic_tripod(i / 30, grip=grip, contact=index)

    answers = [live.push(frame(i)) for i in range(8)]
    assert all(answer is None for answer in answers[:7]) and live.reason == "ready"
    assert answers[7].tolist() == [True, False, False]  # Thumb on the middle finger only.
    assert live.push(frame(8, index=0.10)).tolist() == [True, True, False]
    assert live.push(frame(9, grip=False)).tolist() == [False, False, False]
    # A gap longer than the window allows starts over: no decision until eight new frames.
    assert live.push(frame(40, index=0.10)) is None and live.reason == "time_gap"
    for i in range(41, 47):
        assert live.push(frame(i, index=0.10)) is None
    assert live.push(frame(47, index=0.10)).tolist() == [True, True, False]
    # No hand: nothing to decide, and the window starts again afterwards.
    assert live.push(replace(frame(48), landmarks=())) is None and live.reason == "missing_hand"
    assert live.push(frame(49)) is None and live.reason == "warming_up"
    live.reset()
    assert live.reason == "reset"


def test_what_cannot_be_exported_or_loaded_is_refused(tmp_path):
    pool = fake_pool()
    rules = check_settings(spec("rules", RULE))
    with pytest.raises(ValueError, match="没有名为「别的」的设置；可选：规则"):
        fit_for_live(pool, rules, "别的")
    with pytest.raises(ValueError, match="没有编号为 P09 的人"):
        fit_for_live(pool, rules, "规则", exclude="P09")
    with pytest.raises(ValueError, match="epochs 只用于时序 CNN"):
        fit_for_live(pool, rules, "规则", epochs=5)
    with pytest.raises(ValueError, match="草稿标签不能用来导出"):
        fit_for_live(fake_pool(labels="draft"), rules, "规则")
    with pytest.raises(ValueError, match="合成数据默认拒绝"):
        fit_for_live(fake_pool(source="synthetic"), rules, "规则")
    for model in ("forest", "finger_cnn"):
        with pytest.raises(ValueError, match="只支持调过阈值的规则和时序 CNN"):
            fit_for_live(pool, check_settings(spec(model, {"name": "规则"})), "规则")

    record, _ = fit_for_live(pool, rules, "规则")
    folder = save_live_model(record, None, tmp_path / "ok")
    with pytest.raises(ValueError, match="输出目录已有内容"):
        save_live_model(record, None, folder)
    for change, message in (
        ({"schema": "other"}, "schema 应为"),
        ({"feature_version": "contact-v0"}, "特征版本"),
        ({"threshold": 0.4}, "判定门槛"),
        ({"thresholds": [0.3, float("nan"), 0.3]}, "三个有限的阈值"),
        ({"thresholds": [0.3, 0.3]}, "三个有限的阈值"),
        ({"options": {**record["options"], "frames": 9}}, "帧数超出"),
        ({"options": {**record["options"], "statistic": "max"}}, "设置项无效"),
        ({"options": {**record["options"], "nearest": 1}}, "设置项无效"),
        ({"model": "forest"}, "只支持"),
        ({"window_config": None}, "缺少窗口设置"),
        ({"window_config": {"frames": 1}}, "窗口帧数"),
    ):
        with pytest.raises(ValueError, match=message):
            LiveModel({**record, **change})
    with pytest.raises(ValueError, match="读不到模型文件"):
        load_live_model(tmp_path / "missing")
    with pytest.raises(ValueError, match="窗口形状"):
        load_live_model(folder).decide(np.zeros((2, 7, FEATURE_COUNT), dtype=np.float32))


def test_tampered_or_mismatched_weights_are_refused(tmp_path):
    pytest.importorskip("torch")
    pool = fake_pool()
    record, arrays = fit_for_live(
        pool, check_settings(spec("cnn", {"name": "网络", **NET})), "网络"
    )
    folder = save_live_model(record, arrays, tmp_path / "cnn")
    saved = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    assert saved["weights_sha256"] == contact_live._sha(folder / "weights.npz")
    for change, message in (
        ({"conv1.weight": arrays["conv1.weight"][:, :, :2]}, "形状与设置不符"),
        ({"head.bias": np.array([0.0, np.inf, 0.0], dtype=np.float32)}, "无效数值"),
        ({"scale": np.zeros_like(arrays["scale"])}, "归一化参数无效"),
        ({"extra": np.zeros(1, dtype=np.float32)}, "缺少字段或有多余字段"),
    ):
        with pytest.raises(ValueError, match=message):
            LiveModel(saved, {**arrays, **change})
    with pytest.raises(ValueError, match="缺少字段"):
        LiveModel(saved, {k: v for k, v in arrays.items() if k != "mean"})
    with pytest.raises(ValueError, match="inputs 设置无效"):
        LiveModel({**saved, "options": {**saved["options"], "inputs": "some"}}, arrays)
    # A weights file swapped or changed after saving no longer matches its recorded digest.
    np.savez(folder / "weights.npz", **{**arrays, "head.bias": arrays["head.bias"] + 1})
    with pytest.raises(ValueError, match="校验和不一致"):
        load_live_model(folder)
    (folder / "weights.npz").unlink()
    with pytest.raises(ValueError, match="权重文件缺失"):
        load_live_model(folder)


def test_export_command_writes_a_model_the_app_can_load(tmp_path, capsys):
    data = tmp_path / "data"
    assert main(["ml-fixture", "--pool", "--output", str(data)]) == 0
    capsys.readouterr()
    command = [
        "ml-export",
        str(data),
        "--settings",
        str(SETTINGS / "rules.json"),
        "--candidate",
        "食指用画面内距离＋只取近的",
        "--exclude",
        "synthetic_P02",
        "--cache",
        str(tmp_path / "cache"),
    ]
    assert main([*command, "--output", str(tmp_path / "refused")]) == 2
    assert "合成数据默认拒绝" in capsys.readouterr().err
    assert not (tmp_path / "refused").exists()
    output = tmp_path / "model"
    assert main([*command, "--output", str(output), "--allow-synthetic"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["model"] == "rules" and shown["excluded"] == "synthetic_P02"
    assert shown["trained_on"] == ["synthetic_P01", "synthetic_P03"]
    assert shown["evidence"] == "fitted_not_scored"
    model = load_live_model(output)
    assert model.record["source"] == "synthetic" and model.options["nearest"] is True
    assert main([*command, "--output", str(output), "--allow-synthetic"]) == 2
    assert "输出目录已有内容" in capsys.readouterr().err
