"""Deterministic real-engine motion comparison; no camera, Qt or native input.

Run from the repository root: uv run python scripts/check_motion_tuning.py
All pixel figures use the declared synthetic screen coordinate system. They do
not measure a physical monitor, camera accuracy, user intent or comfort.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pinchpilot import __version__
from pinchpilot.motion_calibration import RestNoiseCalibration
from pinchpilot.tripod import TripodConfig, TripodEngine
from pinchpilot.tripod_demo import synthetic_tripod

FPS = 30
SEED = 20260922
WARMUP = 90
SCREEN = np.array([1920.0, 1080.0])
ORIGIN = np.array([0.5, 0.38])
SPAN = 0.30
PROFILES = ("classic", "precise", "adaptive")
REFERENCE_COMMIT = "6e80967de6ea0040c70e9e866133bdcc45a20233"
# SHA-256 of actual v0.8.2 engine results, rounded to 10 decimal places.
# Captured using the same fixtures below, not a duplicate filter implementation.
CLASSIC_REFERENCE = {
    "white_noise": "700ae56989bbbea9d7e7eb7314ba36e544f3ebb0612ff40c0ebd8e980cfc36dc",
    "correlated_drift": "9502e873c3a7376a2b64722b5bb87c6a8a9215b1d504734447123c1b01c601e2",
    "small_correction": "2d533e2b9266165e2a17b96086b3f9f149b2dd1048beaa66e9cbafaea1db07a4",
    "step_and_stop": "a50bb6c90870491834d1f734265d0570c2c78da96049a7bbbd70613897105e27",
    "single_spike": "5795a56f3f48d98aea81b485311590fce07a4015b1c799d0e3571e70e0f0ddc0",
    "edge_reverse": "1fe9af986357a56d605ef3be28e44d2ebb6057b1bebdceeaed7d9ec1a9becc87",
}


def fixtures():
    """Each profile receives identical camera-coordinate points and timestamps."""
    rng = np.random.default_rng(SEED)
    cases = {}

    def add(name, segments, conditions):
        points = [np.tile(ORIGIN, (WARMUP, 1))]
        phases = ["warmup"] * WARMUP
        for label, offsets in segments:
            offsets = np.asarray(offsets, dtype=float)
            points.append(offsets + ORIGIN)
            phases.extend([label] * len(offsets))
        cases[name] = {
            "points": np.concatenate(points),
            "phases": phases,
            "conditions": conditions,
        }

    def horizontal(values):
        return np.column_stack((values, np.zeros(len(values))))

    add(
        "white_noise",
        [("noise", rng.normal(0, 0.001, (360, 2)))],
        {"duration_s": 12, "sigma_camera_xy": [0.001, 0.001]},
    )
    t = np.arange(360) / FPS
    drift = np.column_stack((0.002 * np.sin(t * 1.8), 0.0015 * np.sin(t * 1.3)))
    add(
        "correlated_drift",
        [("noise", drift + rng.normal(0, 0.0004, (360, 2)))],
        {
            "duration_s": 12,
            "white_sigma_camera_xy": [0.0004, 0.0004],
            "sine_amplitude_camera_xy": [0.002, 0.0015],
            "sine_angular_frequency_rad_s_xy": [1.8, 1.3],
        },
    )
    add(
        "small_correction",
        [
            ("forward", horizontal(np.linspace(0, 0.0015, 13)[1:])),
            ("forward_hold", horizontal(np.full(45, 0.0015))),
            ("reverse", horizontal(np.linspace(0.0015, 0, 13)[1:])),
            ("reverse_hold", np.zeros((45, 2))),
        ],
        {"camera_delta_x": 0.0015, "ramp_s": 0.4, "hold_s": 1.5},
    )
    add(
        "step_and_stop",
        [("step_hold", horizontal(np.full(90, 0.03)))],
        {"camera_step_x": 0.03, "hold_s": 3},
    )
    add(
        "single_spike",
        [("spike", np.array([[0.045, 0]])), ("after_spike", np.zeros((90, 2)))],
        {"camera_spike_x": 0.045, "spike_frames": 1, "hold_after_s": 3},
    )
    add(
        "edge_reverse",
        [
            ("toward_edge", horizontal(np.linspace(0, 0.21, 91)[1:])),
            ("edge_hold", horizontal(np.full(45, 0.21))),
            ("reverse", horizontal(np.linspace(0.21, 0, 91)[1:])),
            ("reverse_hold", np.zeros((60, 2))),
        ],
        {"camera_max_x_offset": 0.21, "ramp_s": 3, "edge_hold_s": 1.5},
    )
    return cases


def run_case(engine, fixture):
    # DesktopController normally seeds this from the real cursor; use a declared
    # center position here without reading or moving the system cursor.
    engine.pointer = (0.5, 0.5)
    engine.reset()
    rows = []
    for i, (point, phase) in enumerate(zip(fixture["points"], fixture["phases"])):
        timestamp = 100.0 + i / FPS
        result = engine.process(synthetic_tripod(timestamp, *point))
        assert result.pointer is not None, "Engine did not provide a pointer"
        output = np.asarray(result.pointer)
        assert np.isfinite(output).all() and ((0 <= output) & (output <= 1)).all()
        assert all(event.kind == "move" for event in result.events), "Unexpected button event"
        if i >= WARMUP:
            assert result.state == "CONTROL", f"Fixture lost control at frame {i}"
        rows.append(
            {
                "frame": i,
                "time_s": i / FPS,
                "phase": phase,
                "input_camera_xy": list(map(float, point)),
                "output_normalized_xy": list(map(float, output)),
                "output_screen_xy": list(map(float, output * SCREEN)),
                "state": result.state,
                "event_kinds": [event.kind for event in result.events],
            }
        )
    return rows


def trajectory_digest(rows):
    values = [
        [
            row["state"],
            [round(v, 10) for v in row["output_normalized_xy"]],
            row["event_kinds"],
        ]
        for row in rows
    ]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def radius_metrics(points):
    radius = np.linalg.norm(points - points.mean(axis=0), axis=1)
    return {
        "rms_radius_px": float(np.sqrt(np.mean(radius**2))),
        "p95_radius_px": float(np.quantile(radius, 0.95)),
        "axis_range_px": np.ptp(points, axis=0).tolist(),
    }


def metrics(name, rows):
    output = np.array([row["output_screen_xy"] for row in rows])
    camera = np.array([row["input_camera_xy"] for row in rows])
    phases = np.array([row["phase"] for row in rows])
    origin = output[WARMUP - 1]
    result = {
        "observed_frames": len(rows) - WARMUP,
        "output_path_length_px": float(
            np.linalg.norm(np.diff(output[WARMUP:], axis=0), axis=1).sum()
        ),
        "final_displacement_px": (output[-1] - origin).tolist(),
        "max_excursion_from_start_px": float(
            np.linalg.norm(output[WARMUP:] - origin, axis=1).max()
        ),
    }
    if name in ("white_noise", "correlated_drift"):
        # Exclude one additional second for disturbance onset, beyond arming warmup.
        observed = slice(WARMUP + FPS, None)
        result["output_noise"] = radius_metrics(output[observed])
        result["input_noise_at_fixed_mapping_scale"] = radius_metrics(
            camera[observed] / SPAN * SCREEN
        )
        result["output_mean_offset_from_start_px"] = (
            output[observed].mean(axis=0) - origin
        ).tolist()
    elif name == "small_correction":
        forward = output[phases == "forward_hold"][-1]
        reverse = output[phases == "reverse_hold"][-1]
        input_fixed_px = 0.0015 / SPAN * SCREEN[0]
        result.update(
            fixed_mapping_input_displacement_px=float(input_fixed_px),
            forward_response_px=float(forward[0] - origin[0]),
            reverse_response_px=float(forward[0] - reverse[0]),
            settled_forward_effective_gain=float((forward[0] - origin[0]) / input_fixed_px),
        )
    elif name == "step_and_stop":
        trace = output[WARMUP:, 0] - origin[0]
        amplitude = float(trace[-1])

        def crossing(fraction):
            matching = np.flatnonzero(trace >= fraction * amplitude)
            return float(matching[0] / FPS * 1000) if len(matching) and amplitude > 0 else None

        after_200ms = int(round(0.2 * FPS))
        result.update(
            fixed_mapping_input_step_px=float(0.03 / SPAN * SCREEN[0]),
            actual_settled_displacement_px=amplitude,
            settled_effective_gain=float(amplitude / (0.03 / SPAN * SCREEN[0])),
            response_10_percent_of_own_final_ms=crossing(0.1),
            response_90_percent_of_own_final_ms=crossing(0.9),
            remaining_displacement_after_200ms_px=float(abs(trace[-1] - trace[after_200ms])),
            tail_path_after_200ms_px=float(np.abs(np.diff(trace[after_200ms:])).sum()),
        )
    elif name == "single_spike":
        result["fixed_mapping_input_spike_px"] = float(0.045 / SPAN * SCREEN[0])
    elif name == "edge_reverse":
        at_edge = output[phases == "edge_hold"][-1, 0]
        reverse = output[phases == "reverse", 0]
        recovery = np.flatnonzero(reverse <= at_edge - 1.0)
        result.update(
            edge_hold_x_px=float(at_edge),
            reverse_recovered_x_px=float(output[-1, 0]),
            reverse_displacement_px=float(at_edge - output[-1, 0]),
            reverse_to_first_1px_response_ms=float((recovery[0] + 1) / FPS * 1000)
            if len(recovery)
            else None,
        )
    return result


def check_mechanisms(cases, comparisons):
    checks = {}
    for profile in PROFILES:
        results = comparisons[profile]["cases"]
        assert results["single_spike"]["metrics"]["max_excursion_from_start_px"] <= 1.0
        assert results["edge_reverse"]["metrics"]["edge_hold_x_px"] >= SCREEN[0] - 1.0
        assert results["edge_reverse"]["metrics"]["reverse_displacement_px"] > 100.0
        checks[f"{profile}_finite_in_screen_no_button_events"] = True
        checks[f"{profile}_isolated_spike_suppressed_below_1px"] = True
        checks[f"{profile}_edge_reversal_recovers_over_100px"] = True
    for name in cases:
        actual = comparisons["classic"]["cases"][name]["trajectory_sha256"]
        assert actual == CLASSIC_REFERENCE[name], f"Classic trajectory changed: {name}"
    checks["classic_matches_v0_8_2_reference"] = True
    fine = comparisons["precise"]["cases"]["small_correction"]["metrics"]
    assert fine["forward_response_px"] > 1.0 and fine["reverse_response_px"] > 1.0
    checks["precise_small_forward_and_reverse_motion_exceeds_1px"] = True
    return checks


def calibrated_comparisons(cases, comparisons):
    """Fit on independent synthetic sessions, then evaluate the unchanged fixtures.

    Calibration samples have their own RNG seeds, timestamps and sine phases.
    No fitting or parameter selection observes the evaluation trajectories.
    """
    calibrations = {}
    selected = ("white_noise", "small_correction", "correlated_drift")
    for index, kind in enumerate(("white_noise", "correlated_drift")):
        seed = SEED + 100 + index
        rng = np.random.default_rng(seed)
        started = 1000.0
        seconds = np.arange(round(RestNoiseCalibration.DURATION * FPS) + 1) / FPS
        conditions = {"white_sigma_camera_xy": [0.001, 0.001]}
        noise = rng.normal(0, 0.001, (len(seconds), 2))
        if kind == "correlated_drift":
            phase = rng.uniform(0, 2 * np.pi, 2)
            noise = rng.normal(0, 0.0004, (len(seconds), 2))
            noise += np.column_stack(
                (
                    0.002 * np.sin(seconds * 1.8 + phase[0]),
                    0.0015 * np.sin(seconds * 1.3 + phase[1]),
                )
            )
            conditions = {
                "white_sigma_camera_xy": [0.0004, 0.0004],
                "sine_amplitude_camera_xy": [0.002, 0.0015],
                "sine_angular_frequency_rad_s_xy": [1.8, 1.3],
                "sine_phase_rad_xy": phase.tolist(),
            }
        config = TripodConfig(**comparisons["precise"]["configuration"])
        calibration = RestNoiseCalibration(started, config)
        calibration_rows = []
        for elapsed, offset in zip(seconds, noise):
            point = ORIGIN + offset
            frame = synthetic_tripod(started + elapsed, *point)
            calibration.add(frame)
            calibration_rows.append(
                {
                    "timestamp": frame.timestamp,
                    "input_camera_xy": point.tolist(),
                }
            )
        entry = {
            "source": "synthetic_engineering",
            "seed": seed,
            "fps": FPS,
            "started_timestamp": started,
            "duration_s": RestNoiseCalibration.DURATION,
            "conditions": conditions,
            "input_rows": calibration_rows,
            "fit_uses_evaluation_frames": False,
            "requested_evaluation_cases": list(selected),
            "profiles": {},
        }
        try:
            estimate = calibration.result()
        except ValueError as error:
            entry.update(status="rejected_not_applied", rejection_reason=str(error), estimate=None)
            calibrations[kind] = entry
            continue
        entry.update(status="accepted_applied", rejection_reason=None, estimate=estimate)
        for profile in ("precise", "adaptive"):
            base = TripodConfig(**comparisons[profile]["configuration"])
            calibrated = replace(
                base,
                rest_noise_x=estimate["rest_noise_x"],
                rest_noise_y=estimate["rest_noise_y"],
            )
            results = {}
            for name in selected:
                fixture = cases[name]
                rows = run_case(TripodEngine(calibrated), fixture)
                result = metrics(name, rows)
                previous = comparisons[profile]["cases"][name]["metrics"]
                if name == "small_correction":
                    change = {
                        "forward_response_change_px": result["forward_response_px"]
                        - previous["forward_response_px"],
                        "reverse_response_change_px": result["reverse_response_px"]
                        - previous["reverse_response_px"],
                    }
                else:
                    change = {
                        "output_noise_rms_change_px": result["output_noise"]["rms_radius_px"]
                        - previous["output_noise"]["rms_radius_px"]
                    }
                results[name] = {
                    "conditions": fixture["conditions"],
                    "trajectory_sha256": trajectory_digest(rows),
                    "metrics": result,
                    "change_from_uncalibrated_same_profile": change,
                    "rows": rows,
                }
            entry["profiles"][profile] = {"configuration": asdict(calibrated), "cases": results}
        calibrations[kind] = entry
    return calibrations


def calibration_markdown(calibrations, comparisons):
    lines = [
        "",
        "## 独立静止校准后的对照",
        "",
        "使用独立seed和1000秒起始的独立序列校准；测试仍使用原来的100秒起始序列，未在测试轨迹上拟合。",
        "只调用实际 RestNoiseCalibration，不调低其拒绝门槛；拒绝时不应用或伪造估计。",
        "表中“前→后”是同模式未校准到应用校准，减小噪声不等于细调一定改善。",
        "",
        "| 校准样本 | 状态 | 模式 | 估计波动/px | 白噪声RMS前→后/px | 相关漂移RMS前→后/px | 小修正正向前→后/px | 小修正反向前→后/px |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for kind, calibration in calibrations.items():
        if calibration["status"] == "rejected_not_applied":
            lines.append(f"| {kind} | 拒绝，未应用 | — | — | — | — | — | — |")
            continue
        estimate = calibration["estimate"]
        for profile, data in calibration["profiles"].items():
            before = comparisons[profile]["cases"]
            after = data["cases"]

            def noise_pair(name):
                initial = before[name]["metrics"]["output_noise"]["rms_radius_px"]
                final = after[name]["metrics"]["output_noise"]["rms_radius_px"]
                return f"{initial:.3f}→{final:.3f}"

            def response_pair(field):
                initial = before["small_correction"]["metrics"][field]
                final = after["small_correction"]["metrics"][field]
                return f"{initial:.3f}→{final:.3f}"

            state = "应用，触及上限" if estimate["limited"] else "应用"
            lines.append(
                f"| {kind} | {state} | {profile} | {estimate['noise_px']:.3f} | "
                f"{noise_pair('white_noise')} | {noise_pair('correlated_drift')} | "
                f"{response_pair('forward_response_px')} | {response_pair('reverse_response_px')} |"
            )
    lines.append("")
    for kind, calibration in calibrations.items():
        if calibration["status"] == "rejected_not_applied":
            lines.append(
                f"- {kind} 被拒绝：{calibration['rejection_reason']}。保留未校准对照，不填写校准后数字。"
            )
        else:
            estimate = calibration["estimate"]
            lines.append(
                f"- {kind}：独立seed={calibration['seed']}，有效样本{estimate['samples']}，"
                f"rest_noise_x={estimate['rest_noise_x']:.8f}、"
                f"rest_noise_y={estimate['rest_noise_y']:.8f}；估计单位为相机归一化坐标。"
            )
    lines.extend(
        [
            "",
            "该对照只考察两种声明噪声下的校准与取舍，不证明其他用户、姿势或机位能获得同样结果。",
            "没有要求校准后全面胜出的断言；有限值、屏幕边界和无按钮事件检查仍应用于每条轨迹。",
            "",
        ]
    )
    return lines


def markdown(report):
    lines = [
        f"# 移动调优合成对照 · {report['app_version']}",
        "",
        "来源：`synthetic_engineering`。未打开相机、未发送系统输入；不代表真人精度或舒适度。",
        "",
        "条件：30fps、1920×1080逻辑坐标、span=0.30、各模式相同输入，先静止预热3秒。",
        "噪声统计另排除扰动开始后1秒。像素单位是本报告声明的坐标，未测物理屏幕。",
        "",
        "| 模式 | 白噪声 RMS/px | 相关漂移 RMS/px | 小幅正向/反向响应 px | 阶跃实际位移/px | 达自身最终位移90%/ms | 阶跃200ms后余动/px | 靠边反向首次响应/ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        cases = report["profiles"][profile]["cases"]
        white = cases["white_noise"]["metrics"]["output_noise"]["rms_radius_px"]
        drift = cases["correlated_drift"]["metrics"]["output_noise"]["rms_radius_px"]
        small = cases["small_correction"]["metrics"]
        step = cases["step_and_stop"]["metrics"]
        edge = cases["edge_reverse"]["metrics"]
        lines.append(
            f"| {profile} | {white:.3f} | {drift:.3f} | "
            f"{small['forward_response_px']:.3f} / {small['reverse_response_px']:.3f} | "
            f"{step['actual_settled_displacement_px']:.3f} | "
            f"{step['response_90_percent_of_own_final_ms']:.1f} | "
            f"{step['tail_path_after_200ms_px']:.3f} | "
            f"{edge['reverse_to_first_1px_response_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 如何解释",
            "",
            "- 小幅修正在固定映射下为9.6px；阶跃在固定映射下为192px。自适应模式会改变增益，不能据其偏离固定终点声称不准确。",
            "- 阶跃响应时间相对每个模式自身最终位移计算，只测引擎，不包含摄像头、显示器、系统输入或人的动作时间。",
            "- 停止余动指阶跃输入保持不动200ms后剩余输出路径；不是拍摄到的真实鼠标延迟。",
            "- 靠边测试故意让固定映射超出边界，再反向移动；记录恢复等待，不能只报告最终能回来。",
            "- 合成手指点变化不代表真实手部整体运动；这里没有遮挡、识别错误、用户目标或点击意图标签。",
            "- 完整输入、输出轨迹、参数、seed和实际倍率保存在同目录 motion-comparison.json。",
            "",
            f"机制断言：{len(report['mechanism_checks'])}项通过。经典轨迹与 v0.8.2 的固定参考一致。",
            "",
        ]
    )
    lines.extend(calibration_markdown(report["calibrations"], report["profiles"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(f"reports/verification/motion-{__version__}")
    cases = fixtures()
    report = {
        "schema": "pinchpilot-motion-comparison-v2",
        "source": "synthetic_engineering",
        "app_version": __version__,
        "fps": FPS,
        "seed": SEED,
        "rng": "numpy.default_rng/PCG64",
        "numpy_version": np.__version__,
        "screen_coordinates": {"width": int(SCREEN[0]), "height": int(SCREEN[1])},
        "warmup_frames": WARMUP,
        "initial_pointer_normalized_xy": [0.5, 0.5],
        "camera_opened": False,
        "os_events_sent": False,
        "classic_reference_commit": REFERENCE_COMMIT,
        "classic_digest_decimal_places": 10,
        "profiles": {},
        "limits": [
            "Synthetic engine output only; no camera accuracy, user intent or fatigue evidence.",
            "Adaptive gain changes intended displacement; fixed-mapping deviation is not error.",
            "Response excludes camera, native input and display latency.",
            "No click/drag, occlusion or variable frame timing comparison.",
            "Calibration uses independent declared synthetic noise; not a human calibration study.",
        ],
    }
    for profile in PROFILES:
        config = TripodConfig(
            motion_profile=profile,
            span=SPAN,
            screen_width=int(SCREEN[0]),
            screen_height=int(SCREEN[1]),
            rest_noise_x=0.0,
            rest_noise_y=0.0,
        )
        compared = {}
        for name, case in cases.items():
            rows = run_case(TripodEngine(config), case)
            compared[name] = {
                "conditions": case["conditions"],
                "trajectory_sha256": trajectory_digest(rows),
                "metrics": metrics(name, rows),
                "rows": rows,
            }
        report["profiles"][profile] = {"configuration": asdict(config), "cases": compared}
    report["mechanism_checks"] = check_mechanisms(cases, report["profiles"])
    report["calibrations"] = calibrated_comparisons(cases, report["profiles"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "motion-comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = markdown(report)
    (directory / "motion-comparison.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved: {directory.resolve()}")


if __name__ == "__main__":
    main()
