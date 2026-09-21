"""Command-line tools never inject mouse events; only the GUI has that opt-in."""

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="pinchpilot", description="PinchPilot 摄像头捏合交互原型")
    sub = cli.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="打开桌面界面（默认仅预览）")
    gui.add_argument("--workspace", type=Path, default=Path.cwd())
    gui.add_argument("--demo", action="store_true", help="合成动作演示，不接相机、不控制系统")
    gui.add_argument(
        "--interaction",
        choices=("pinch", "tripod", "finger-flex", "finger-dwell"),
        default="pinch",
        help="捏合主方案 / 三指定位点击 / 已搁置的单指对照（实验模式仅应用内）",
    )
    gui.add_argument("--smoke-seconds", type=float, help="在指定秒数后关闭，用于界面检查")
    gui.add_argument("--screenshot", type=Path, help="保存本程序窗口的截图")
    doctor = sub.add_parser("doctor", help="检查运行环境；不打开摄像头")
    doctor.add_argument("--output", type=Path)
    download = sub.add_parser("fetch-model", help="下载 Google 官方手部模型")
    download.add_argument("--path", type=Path)
    inspect = sub.add_parser("inspect-data", help="统计关键点录制，不训练")
    inspect.add_argument("--data", type=Path, default=Path("data/recordings"))
    train = sub.add_parser("train", help="独立分组留出评测并保存模型")
    train.add_argument("--data", type=Path, default=Path("data/recordings"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--group-by", choices=("participant", "session"), default="participant")
    train.add_argument("--model", choices=("forest", "svm"), default="forest")
    train.add_argument("--seed", type=int, default=42)
    calibrate = sub.add_parser("calibrate", help="用指定参与者的数据校准规则阈值")
    calibrate.add_argument("--data", type=Path, default=Path("data/recordings"))
    calibrate.add_argument("--participant", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay", help="离线回放关键点并导出事件；不操作系统鼠标")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--profile", type=Path)
    replay.add_argument("--model", type=Path, help="仅使用本项目生成且可信的 joblib 文件")
    return cli


def environment_report() -> dict:
    from .platform_io import camera_permission
    from .vision import default_model_path

    versions = {}
    for name in (
        "pinchpilot",
        "numpy",
        "opencv-contrib-python",
        "mediapipe",
        "PySide6",
        "scikit-learn",
        "pyobjc-framework-Quartz",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed / not needed on this OS"
    result = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "versions": versions,
        "model_path": str(default_model_path()),
        "model_present": default_model_path().exists(),
        "camera_permission": camera_permission(),
        "camera_opened": False,
        "os_events_sent": False,
    }
    if sys.platform == "darwin":
        try:
            from ApplicationServices import AXIsProcessTrusted

            result["accessibility_trusted"] = bool(AXIsProcessTrusted())
        except ImportError:
            result["accessibility_trusted"] = None
    return result


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command is None:
        args = cli.parse_args(["gui"])
    try:
        if args.command == "gui":
            from .app import run_gui

            return run_gui(
                args.workspace, args.demo, args.smoke_seconds, args.screenshot, args.interaction
            )
        if args.command == "doctor":
            result = environment_report()
            if args.output:
                from .storage import save_json

                save_json(args.output, result)
        elif args.command == "fetch-model":
            from .vision import fetch_model

            result = {"model_path": str(fetch_model(args.path))}
        elif args.command == "inspect-data":
            from .learning import load_dataset

            data = load_dataset(args.data)
            result = {
                **data.counts,
                "usable_frames": len(data.y),
                "participants": sorted(set(data.groups)),
                "labels": dict(Counter(data.y)),
                "episodes": len(set(data.episodes)),
            }
        elif args.command == "train":
            from .learning import train

            result = train(args.data, args.output, args.group_by, args.model, args.seed)
        elif args.command == "calibrate":
            from .domain import EngineConfig
            from .storage import calibrate, save_json

            _, result = calibrate(args.data, args.participant, EngineConfig())
            save_json(args.output, result)
        elif args.command == "replay":
            from .domain import EngineConfig
            from .engine import GestureEngine
            from .features import extract
            from .storage import read_recording, save_json

            config = (
                EngineConfig(**json.loads(args.profile.read_text(encoding="utf-8"))["config"])
                if args.profile
                else EngineConfig()
            )
            engine = GestureEngine(config)
            predictor = None
            if args.model:
                from .learning import LearnedPredictor

                predictor = LearnedPredictor(args.model)
            metadata, frames = read_recording(args.recording)
            events = []
            for frame in frames:
                f = extract(frame)
                p = predictor.predict(f) if predictor and f else None
                for event in engine.process(frame, p).events:
                    events.append({"timestamp": frame.timestamp, **asdict(event)})
            for event in engine.reset().events:
                events.append({"timestamp": frames[-1].timestamp if frames else 0, **asdict(event)})
            result = {
                "recording": metadata,
                "config": asdict(config),
                "os_events_sent": False,
                "events": events,
                "note": "Single-label episodes may never arm the interaction; online evaluation needs continuous sequences.",
            }
            save_json(args.output, result)
            result = {"output": str(args.output), "events": len(events), "os_events_sent": False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"PinchPilot: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
