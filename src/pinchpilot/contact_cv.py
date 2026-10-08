"""Leave-one-person-out comparison of contact models, one settings file per model.

Each person is held out once as the test person of a fold. Among the others, each person in
turn is the validation person and the model is fitted on the rest. Tuning reports those
validation scores only. A held-out person is scored only by `final_test`, with the candidate
setting that the fold's own validation chose.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import __version__, contact_protocol
from .contact_data import CHANNELS
from .contact_models import _binary_metrics, _normalize, _score

SETTINGS_SCHEMA = "pinchpilot-contact-settings-v1"
REPORT_SCHEMA = "pinchpilot-contact-cv-v1"
MODEL_NAMES = {
    "rules": "调过阈值的规则",
    "forest": "随机森林",
    "cnn": "时序 CNN",
    "finger_cnn": "手指共享 CNN",
}
CHANNEL_NAMES = {"middle": "中指", "index": "食指", "ring": "无名指"}
MAX_CANDIDATES = 24

_NETWORK = {
    "frames": 8,
    "stride": 2,
    "channels": 32,
    "epochs": 15,
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0.0,
    "positive_weight": 1.0,
}
# Every option a candidate may set, with the value used when it does not.
DEFAULTS = {
    "rules": {"frames": 1, "stride": 1, "statistic": "last"},
    "forest": {"frames": 8, "stride": 3, "trees": 160, "depth": 12, "min_leaf": 2},
    "cnn": dict(_NETWORK),
    "finger_cnn": {**_NETWORK, "channels": 24, "shared": True, "context": True},
}


def _number(integer, low, high):
    kinds = (int,) if integer else (int, float)
    return lambda value: type(value) in kinds and low <= value <= high


CHECKS = {
    "frames": _number(True, 1, 64),
    "stride": _number(True, 1, 30),
    "trees": _number(True, 10, 2000),
    "depth": _number(True, 1, 64),
    "min_leaf": _number(True, 1, 1000),
    "channels": _number(True, 4, 256),
    "epochs": _number(True, 1, 500),
    "learning_rate": _number(False, 1e-5, 1.0),
    "batch_size": _number(True, 8, 4096),
    "weight_decay": _number(False, 0.0, 1.0),
    "positive_weight": lambda value: (
        type(value) is str and value == "balanced" or _number(False, 0.05, 50.0)(value)
    ),
    "statistic": lambda value: type(value) is str and value in ("last", "median", "mean", "min"),
    "shared": lambda value: type(value) is bool,
    "context": lambda value: type(value) is bool,
}


def check_settings(spec):
    """Validated settings with every option filled in: {model, seed, candidates}."""
    if not isinstance(spec, dict) or spec.get("schema") != SETTINGS_SCHEMA:
        raise ValueError(f"设置文件格式无效：schema 应为 {SETTINGS_SCHEMA}")
    unknown = set(spec) - {"schema", "model", "seed", "note", "candidates"}
    if unknown:
        raise ValueError(f"设置文件里有不认识的项：{'、'.join(sorted(unknown))}")
    model = spec.get("model")
    if model not in DEFAULTS:
        raise ValueError(f"model 应为 {'、'.join(DEFAULTS)} 之一")
    seed = spec.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32 - 3:
        raise ValueError("seed 必须是有效的非负整数")
    listed = spec.get("candidates")
    if not isinstance(listed, list) or not 1 <= len(listed) <= MAX_CANDIDATES:
        raise ValueError(f"candidates 需要 1 到 {MAX_CANDIDATES} 组设置")
    candidates, names = [], set()
    for item in listed:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip() or name.strip() in names:
            raise ValueError("每组设置需要一个不重复的 name")
        name = name.strip()
        names.add(name)
        options = dict(DEFAULTS[model])
        for key, value in item.items():
            if key == "name":
                continue
            if key not in options:
                raise ValueError(
                    f"设置「{name}」里有不认识的项 {key}；"
                    f"{MODEL_NAMES[model]}可用的项：{'、'.join(options)}"
                )
            if not CHECKS[key](value):
                raise ValueError(f"设置「{name}」的 {key} 取值无效：{value!r}")
            options[key] = value
        candidates.append({"name": name, "options": options})
    return {"model": model, "seed": seed, "candidates": candidates}


def read_settings(path):
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"设置文件不是有效的 JSON：{path}") from error
    return check_settings(spec)


def _best_threshold(values, labels):
    """The cut on a distance (touching is small) with the highest F1 on these labels."""
    order = np.argsort(values, kind="stable")
    values, labels = values[order], labels[order]
    hits = np.cumsum(labels)
    # With everything up to a cut called touching: F1 = 2TP / (taken + positives).
    f1 = 2 * hits / (np.arange(1, len(values) + 1) + hits[-1])
    cuts = np.flatnonzero(np.append(values[1:] != values[:-1], True))
    best = cuts[np.argmax(f1[cuts])]
    above = values[best + 1] if best + 1 < len(values) else values[best]
    return float((values[best] + above) / 2)


def _distances(x, options):
    recent = x[:, -options["frames"] :, 63:66]
    if options["statistic"] == "last":
        return recent[:, -1]
    return getattr(np, options["statistic"])(recent, axis=1)


def _fit_and_score(model, options, seed, x, y, targets, select):
    """Fit on (x, y) and score the named target windows: ({name: scores}, details).

    `targets` maps a name to (windows, labels). With `select`, a network keeps for each
    target the epoch where that target's own loss was lowest, so only validation people may
    be passed that way. Without it every target is scored after the last epoch.
    """
    frames = options["frames"]
    for i, channel in enumerate(CHANNELS):
        if len(set(y[:, i].tolist())) < 2:
            raise ValueError(
                f"训练用的窗口里{CHANNEL_NAMES[channel]}标签只有一种取值；把 stride 调小或补充录制"
            )
    if model == "rules":
        distances = _distances(x, options)
        thresholds = np.array([_best_threshold(distances[:, i], y[:, i]) for i in range(3)])
        scores = {
            name: (_distances(windows, options) <= thresholds).astype(np.float32)
            for name, (windows, _) in targets.items()
        }
        return scores, {"thresholds": dict(zip(CHANNELS, thresholds.tolist()))}
    if model == "forest":

        def flat(windows):
            return windows[:, -frames:].reshape(len(windows), -1)

        heads = [
            RandomForestClassifier(
                n_estimators=options["trees"],
                max_depth=options["depth"],
                min_samples_leaf=options["min_leaf"],
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + channel,
            ).fit(flat(x), y[:, channel])
            for channel in range(3)
        ]
        scores = {
            name: np.column_stack([head.predict_proba(flat(windows))[:, 1] for head in heads])
            for name, (windows, _) in targets.items()
        }
        return scores, {}
    from . import contact_nets as nets

    if model == "finger_cnn":
        network = nets.finger_cnn(
            frames, options["channels"], seed, options["shared"], options["context"]
        )

        def prepare(windows):
            return nets.finger_inputs(windows[:, -frames:])
    else:
        network = nets.temporal_cnn(frames, options["channels"], seed)

        def prepare(windows):
            return windows[:, -frames:]

    inputs = prepare(x)
    axes = tuple(range(inputs.ndim - 1))
    mean = inputs.astype(np.float64).mean(axis=axes)
    scale = inputs.astype(np.float64).std(axis=axes)
    scale = np.where(scale < 1e-6, 1.0, scale)
    prepared = {
        name: (_normalize(prepare(windows), mean, scale), labels)
        for name, (windows, labels) in targets.items()
    }
    details, states = nets.train(
        network,
        _normalize(inputs, mean, scale),
        y,
        epochs=options["epochs"],
        learning_rate=options["learning_rate"],
        batch_size=options["batch_size"],
        weight_decay=options["weight_decay"],
        positive_weight=options["positive_weight"],
        seed=seed,
        validation=prepared if select else None,
    )
    scores = {}
    for name, (values, _) in prepared.items():
        if select:
            network.load_state_dict(states[name], strict=True)
        scores[name] = nets.predict(network, values)
    return scores, {key: details[key] for key in ("epochs", "selected_epoch", "parameters")}


def _breakdown(pool, indices, predicted):
    """Scores by round (hand pose) and error rates by step, where the session recorded them."""
    y, rounds, steps = pool.y[indices], pool.rounds[indices], pool.steps[indices]
    by_round, by_step = {}, {}
    for number in sorted(set(rounds.tolist()) - {0}):
        mask = rounds == number
        f1 = {
            channel: _binary_metrics(y[mask, i], predicted[mask, i])["f1"]
            if y[mask, i].any()
            else None
            for i, channel in enumerate(CHANNELS)
        }
        known = [value for value in f1.values() if value is not None]
        by_round[str(number)] = {
            "windows": int(mask.sum()),
            "f1": f1,
            "macro_f1": float(np.mean(known)) if known else None,
        }
    for step in sorted(set(steps.tolist()) - {-1}):
        mask = steps == step
        by_step[pool.metadata["step_names"][step]] = {
            "windows": int(mask.sum()),
            "error_rate": {
                channel: float(np.mean(y[mask, i] != predicted[mask, i]))
                for i, channel in enumerate(CHANNELS)
            },
        }
    return {"rounds": by_round, "steps": by_step}


def _result(pool, indices, scores):
    predicted = (scores >= 0.5).astype(np.int8)
    return {**_score(pool.y[indices], predicted), **_breakdown(pool, indices, predicted)}


def _mean(values):
    known = [value for value in values if value is not None]
    return float(np.mean(known)) if known else None


def _average(results):
    """Mean of a list of `_result` dictionaries, key by key."""
    rounds = sorted({number for item in results for number in item["rounds"]}, key=int)
    steps = sorted({name for item in results for name in item["steps"]})
    return {
        "macro_f1": _mean([item["macro_f1"] for item in results]),
        "channels": {
            channel: {
                key: _mean([item["channels"][channel][key] for item in results])
                for key in ("f1", "precision", "recall")
            }
            for channel in CHANNELS
        },
        "rounds": {
            number: _mean([item["rounds"].get(number, {}).get("macro_f1") for item in results])
            for number in rounds
        },
        "steps": {
            name: {
                channel: _mean(
                    [
                        item["steps"][name]["error_rate"][channel]
                        for item in results
                        if name in item["steps"]
                    ]
                )
                for channel in CHANNELS
            }
            for name in steps
        },
    }


def run_cv(pool, settings, *, final_test=False, allow_synthetic=False, progress=None):
    """Validation scores of every candidate, and with `final_test` the held-out test scores."""
    say = progress or (lambda text: None)
    meta = pool.metadata
    if meta["source"] == "synthetic" and not allow_synthetic:
        raise ValueError("合成数据默认拒绝；仅工程验证可加 --allow-synthetic")
    if final_test and meta["labels"] != "reviewed":
        raise ValueError("草稿标签不能用来跑测试结果；先完成标签检查")
    people = sorted(set(pool.participants.tolist()))
    if len(people) < 3:
        raise ValueError(f"留一人验证至少需要 3 位参与者的录制，现在只有 {len(people)} 位")
    index = {person: np.flatnonzero(pool.participants == person) for person in people}
    for person, rows in index.items():
        for i, channel in enumerate(CHANNELS):
            if len(set(pool.y[rows, i].tolist())) < 2:
                raise ValueError(
                    f"{person} 的{CHANNEL_NAMES[channel]}标签只有一种取值，无法训练或评分"
                )
    model, seed, candidates = settings["model"], settings["seed"], settings["candidates"]
    for candidate in candidates:
        if candidate["options"]["frames"] > pool.frames:
            raise ValueError(f"设置「{candidate['name']}」的 frames 不能超过 {pool.frames}")

    def training(excluded, stride):
        rows = np.concatenate([index[person] for person in people if person not in excluded])
        return np.sort(rows)[::stride]

    def target(person):
        return pool.windows(index[person]), pool.y[index[person]]

    # One fit serves two folds. Fitted without a and b, its score on a belongs to the fold
    # that holds out b, and its score on b to the fold that holds out a. A network keeps a
    # separate best epoch for each of the two, chosen by that person's loss alone.
    runs = [(a, b) for a in people for b in people if a < b]
    inner, count = [], 0
    for candidate in candidates:
        options, results = candidate["options"], {}
        for a, b in runs:
            started = time.perf_counter()
            rows = training((a, b), options["stride"])
            scores, details = _fit_and_score(
                model,
                options,
                seed,
                pool.windows(rows),
                pool.y[rows],
                {a: target(a), b: target(b)},
                select=True,
            )
            for validation, held_out in ((a, b), (b, a)):
                results[(held_out, validation)] = {
                    "held_out": held_out,
                    "validation": validation,
                    "trained_on": [p for p in people if p not in (a, b)],
                    "training_windows": len(rows),
                    "selected_epoch": details.get("selected_epoch", {}).get(validation),
                    **_result(pool, index[validation], scores[validation]),
                }
            count += 1
            say(
                f"[{count}/{len(candidates) * len(runs)}] {candidate['name']}："
                f"不含 {a}、{b} 的训练完成，{time.perf_counter() - started:.1f} 秒"
            )
        inner.append((results, details))
    selected = {
        held_out: int(
            np.argmax(
                [
                    np.mean([results[(held_out, v)]["macro_f1"] for v in people if v != held_out])
                    for results, _ in inner
                ]
            )
        )
        for held_out in people
    }
    reference = {
        person: _result(pool, index[person], pool.rules[index[person]]) for person in people
    }
    report = {
        "schema": REPORT_SCHEMA,
        "app_version": __version__,
        "model": model,
        "seed": seed,
        "source": meta["source"],
        "labels": meta["labels"],
        "evidence": "synthetic_engineering"
        if meta["source"] == "synthetic"
        else "draft_labels_trial_run"
        if meta["labels"] != "reviewed"
        else "final_test"
        if final_test
        else "validation_only",
        "pool_fingerprint": meta.get("fingerprint"),
        "feature_version": meta["feature_version"],
        "window_config": meta["window_config"],
        "participants": people,
        "protocols": sorted({item["protocol"] for item in meta["recordings"] if item["protocol"]}),
        "recordings": len(meta["recordings"]),
        "windows": len(pool),
        "threshold": 0.5,
        "candidates": [
            {
                "name": candidate["name"],
                "options": candidate["options"],
                "parameters": details.get("parameters"),
                "validation": _average(list(results.values())),
                "selected_for": [person for person in people if selected[person] == number],
                "fits": list(results.values()),
            }
            for number, (candidate, (results, details)) in enumerate(zip(candidates, inner))
        ],
        "fixed_rules_validation": _average(list(reference.values())),
        "test": None,
    }
    if not final_test:
        return report
    folds = []
    for held_out in people:
        candidate = candidates[selected[held_out]]
        options = dict(candidate["options"])
        epochs = [
            inner[selected[held_out]][0][(held_out, v)]["selected_epoch"]
            for v in people
            if v != held_out
        ]
        if all(value is not None for value in epochs):
            # No validation person is left to stop on, so use what the fold's validation chose.
            options["epochs"] = max(1, round(float(np.mean(epochs))))
        rows = training((held_out,), options["stride"])
        scores, _ = _fit_and_score(
            model,
            options,
            seed,
            pool.windows(rows),
            pool.y[rows],
            {held_out: target(held_out)},
            select=False,
        )
        folds.append(
            {
                "held_out": held_out,
                "candidate": candidate["name"],
                "options": options,
                "training_windows": len(rows),
                **_result(pool, index[held_out], scores[held_out]),
                "fixed_rules": reference[held_out],
            }
        )
        say(f"测试：留出 {held_out}，用设置「{candidate['name']}」")
    report["test"] = {
        "folds": folds,
        **_average(folds),
        "macro_f1_std": float(np.std([fold["macro_f1"] for fold in folds])),
        "fixed_rules": _average([fold["fixed_rules"] for fold in folds]),
    }
    return report


def _cell(value):
    return "–" if value is None else f"{value:.3f}"


def _scores_row(label, scores, tail=""):
    cells = [_cell(scores["macro_f1"])] + [
        _cell(scores["channels"][channel]["f1"]) for channel in CHANNELS
    ]
    return f"| {label} | " + " | ".join(cells) + (f" | {tail} |" if tail else " |")


def markdown(report):
    """The report as a table a person can read and paste into an issue."""
    model = MODEL_NAMES[report["model"]]
    lines = [f"# {model}：{'测试结果' if report['test'] else '验证结果'}", ""]
    if report["evidence"] == "draft_labels_trial_run":
        lines += ["> **草稿标签，仅供试跑。** 标签还没有检查，这些数字不能写进报告。", ""]
    elif report["evidence"] == "synthetic_engineering":
        lines += ["> **合成数据，只验证程序能跑通。** 这些数字不代表真人效果。", ""]
    lines += [
        f"数据：{len(report['participants'])} 人、{report['recordings']} 段录制、"
        f"{report['windows']} 个窗口。",
        "",
        "## 验证",
        "",
        "每次留出一人不参与，在其余的人里轮流用一人验证、用剩下的人训练。下表是各次验证的平均 F1。",
        "",
        "| 设置 | 平均 | 中指 | 食指 | 无名指 | 被几折选中 |",
        "|---|---|---|---|---|---|",
    ]
    for candidate in report["candidates"]:
        lines.append(
            _scores_row(
                candidate["name"], candidate["validation"], str(len(candidate["selected_for"]))
            )
        )
    lines.append(_scores_row("产品现有的固定规则（参考）", report["fixed_rules_validation"], "–"))
    lines += ["", "设置内容："]
    for candidate in report["candidates"]:
        options = "，".join(f"{key}={value}" for key, value in candidate["options"].items())
        size = f"；{candidate['parameters']} 个参数" if candidate["parameters"] else ""
        lines.append(f"- {candidate['name']}：{options}{size}")
    rounds = sorted({n for c in report["candidates"] for n in c["validation"]["rounds"]}, key=int)
    if rounds:
        lines += [
            "",
            "## 各轮的平均 F1",
            "",
            "| 设置 | " + " | ".join(f"第 {number} 轮" for number in rounds) + " |",
            "|---|" + "---|" * len(rounds),
        ]
        for candidate in report["candidates"]:
            cells = [_cell(candidate["validation"]["rounds"].get(number)) for number in rounds]
            lines.append(f"| {candidate['name']} | " + " | ".join(cells) + " |")
        if report["protocols"] == [contact_protocol.NAME]:
            poses = contact_protocol.POSES
            legend = [f"第 {number} 轮：{poses[int(number) - 1]}" for number in rounds]
            lines += ["", "每轮的手部朝向不同。" + "；".join(legend) + "。"]
    best = max(report["candidates"], key=lambda item: item["validation"]["macro_f1"])
    worst = sorted(
        (
            (rate, name, channel)
            for name, rates in best["validation"]["steps"].items()
            for channel, rate in rates.items()
            if rate is not None
        ),
        reverse=True,
    )[:6]
    if worst:
        texts = {step.name: step.text for step in contact_protocol.session()}
        lines += [
            "",
            f"## 错得最多的步骤（设置「{best['name']}」）",
            "",
            "| 步骤 | 手指 | 判错的比例 |",
            "|---|---|---|",
        ]
        for rate, name, channel in worst:
            lines.append(f"| {texts.get(name, name)} | {CHANNEL_NAMES[channel]} | {rate:.1%} |")
    if report["test"]:
        test = report["test"]
        lines += [
            "",
            "## 测试",
            "",
            "每一折用它自己的验证选出的设置，在除留出者以外的所有人上重新训练，再给留出的人打分。",
            "",
            "| 留出的人 | 用的设置 | 平均 | 中指 | 食指 | 无名指 | 固定规则的平均 |",
            "|---|---|---|---|---|---|---|",
        ]
        for fold in test["folds"]:
            label = f"{fold['held_out']} | {fold['candidate']}"
            lines.append(_scores_row(label, fold, _cell(fold["fixed_rules"]["macro_f1"])))
        lines.append(_scores_row("平均 | –", test, _cell(test["fixed_rules"]["macro_f1"])))
        lines += ["", f"各折平均 F1 的标准差：{test['macro_f1_std']:.3f}。"]
    return "\n".join(lines) + "\n"


def write_report(report, settings, directory):
    """Save the report as JSON and Markdown next to the settings that produced it."""
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("输出目录已有内容；请换一个新目录")
    directory.mkdir(parents=True, exist_ok=True)
    text = markdown(report)
    for name, value in (("report.json", report), ("settings.json", settings)):
        (directory / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (directory / "report.md").write_text(text, encoding="utf-8")
    return text
