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
    desktop = sub.add_parser("desktop", help="日常桌面测试工具：拇中定位、左右键与拖拽")
    desktop.add_argument("--workspace", type=Path, default=Path.cwd())
    desktop.add_argument("--demo", action="store_true", help="合成演示，禁止系统鼠标输出")
    desktop.add_argument("--smoke-seconds", type=float)
    desktop.add_argument("--screenshot", type=Path)
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
    compare = sub.add_parser(
        "compare-tripod", help="同帧对比三种定位方案；默认使用合成序列，不操作系统鼠标"
    )
    compare.add_argument("--recording", type=Path, help="可选：既有关键点录制 JSONL")
    compare.add_argument("--output", type=Path, required=True, help="输出对比 JSON 文件")
    compare.add_argument("--seed", type=int, default=20260922)
    compare.add_argument("--trace", action="store_true", help="包含逐帧虚拟指针与事件")
    fixture = sub.add_parser("ml-fixture", help="生成明确标记的合成三指标注；仅工程验证")
    fixture.add_argument("--output", type=Path, required=True, help="需要一个空目录")
    fixture.add_argument("--seed", type=int, default=20260922)
    annotate = sub.add_parser("ml-label-template", help="为关键点录制生成未复核的空白接触标注")
    annotate.add_argument("recording", type=Path)
    annotate.add_argument("--output", type=Path, required=True)
    contact_inspect = sub.add_parser("ml-inspect", help="检查连续接触数据、人工标注和分组泄漏")
    contact_inspect.add_argument("manifest", type=Path)
    contact_train = sub.add_parser("ml-train", help="离线训练三路接触RF/CNN，不接管系统鼠标")
    contact_train.add_argument("manifest", type=Path)
    contact_train.add_argument("--output", type=Path, required=True)
    contact_train.add_argument("--model", choices=("forest", "cnn"), default="forest")
    contact_train.add_argument("--seed", type=int, default=42)
    contact_train.add_argument("--epochs", type=int, default=20)
    contact_train.add_argument(
        "--allow-synthetic", action="store_true", help="只为工程验证训练合成数据"
    )
    shadow = sub.add_parser("ml-shadow", help="离线旁路比较模型与几何规则；只加载可信的本地模型")
    shadow.add_argument("recording", type=Path)
    shadow.add_argument("--model", type=Path, required=True)
    shadow.add_argument("--output", type=Path, required=True)
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
            import Quartz
            from ApplicationServices import AXIsProcessTrusted

            result["accessibility_trusted"] = bool(AXIsProcessTrusted())
            result["input_monitoring_allowed"] = (
                bool(Quartz.CGPreflightListenEventAccess())
                if hasattr(Quartz, "CGPreflightListenEventAccess")
                else None
            )
        except ImportError:
            result["accessibility_trusted"] = None
            result["input_monitoring_allowed"] = None
    return result


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command is None:
        args = cli.parse_args(["gui"])
    try:
        if args.command == "desktop":
            from .desktop import run_desktop

            return run_desktop(args.workspace, args.demo, args.smoke_seconds, args.screenshot)
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
        elif args.command == "compare-tripod":
            from .replay_compare import run_comparison

            run_comparison(args.recording, args.output, seed=args.seed, include_trace=args.trace)
            result = {
                "output": str(args.output),
                "source": "recorded" if args.recording else "synthetic",
                "os_events_sent": False,
            }
        elif args.command == "ml-fixture":
            from .contact_data import create_synthetic_contact_dataset

            result = {
                "manifest": str(create_synthetic_contact_dataset(args.output, args.seed)),
                "source": "synthetic_engineering",
            }
        elif args.command == "ml-label-template":
            from .contact_data import annotation_template

            annotation_template(args.recording, args.output)
            result = {"output": str(args.output), "reviewed": False, "labels": "unknown"}
        elif args.command in ("ml-inspect", "ml-train"):
            from .contact_data import load_contact_dataset

            dataset = load_contact_dataset(args.manifest)
            if args.command == "ml-inspect":
                result = {
                    **dataset.metadata,
                    "windows": {k: len(v.x) for k, v in dataset.partitions.items()},
                }
            else:
                from .contact_models import train_contacts

                result = train_contacts(
                    dataset,
                    args.output,
                    model=args.model,
                    seed=args.seed,
                    epochs=args.epochs,
                    allow_synthetic=args.allow_synthetic,
                )
        elif args.command == "ml-shadow":
            from .contact_shadow import run_shadow

            run_shadow(args.recording, args.model, args.output)
            result = {
                "output": str(args.output),
                "os_events_sent": False,
                "scope": "offline model suggestions only",
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"PinchPilot: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
