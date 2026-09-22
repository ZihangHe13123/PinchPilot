# PinchPilot

用普通电脑摄像头进行捏合交互的课程研究原型。研究目标是：**以尽可能少的辅助动作和尽可能小的手部活动幅度，稳定完成桌面操控**，并检验误操作、响应速度和主观费力度。

**当前源码版本 v0.8.2：修复 macOS 双击标记，并持续锁定控制手，避免另一只手入镜时抢占。** 桌面工具支持拇中定位、食拇左键/双击/拖拽、拇无名指右键和真实滚轮，保留既有默认移动手感、抗抖和连续灵敏度滑条。完整步骤见 [桌面试用说明](docs/DESKTOP_TRIAL.md)。

## 立即启动桌面测试版

本次修复在源码启动版 v0.8.2；下列已交付的 v0.8.1 便携包按原样保留，尚不包含本次修复。

Windows 组员使用 `PinchPilot-0.8.1-Windows-x64.zip`：全部解压后双击 **Start.cmd**，无需安装 Python/uv，模型随包附带。详见 [Windows 试用说明](docs/WINDOWS_TRIAL.md)。

Mac 留档使用 `PinchPilot-0.8.1-macOS-arm64.zip`：适用于 M 系列 Mac、macOS 14+，解压后双击 **Start.command**。同样带齐运行环境和模型，并含源码快照；详见 [Mac 归档说明](docs/MACOS_ARCHIVE.md)。构建命令：`uv run python scripts/build_macos_portable.py`。

源码方式：Mac 双击 **launch_desktop_mac.command**；Windows 双击 **launch_desktop_windows.cmd**。或在项目目录执行：

```sh
uv run pinchpilot desktop
```

**启动相机 → 启用鼠标控制 → 食指移开、拇中轻捏接管。** 从当前系统光标位置开始。启动时相机和控制均关闭，手感设置会保留。先退出占用相机的旧 Demo；前臂支撑在桌面或扶手，让摄像头看清手和指尖。

| 操作 | 动作 |
|---|---|
| 移动 | 拇中捏住，小幅移动捏合点 |
| 左键 | 食拇短捏，松食指完成点击 |
| 双击 | 同一位置连续食拇短捏两次，中间松开食指；节奏遵循系统双击速度 |
| 右键 | 食指移开，拇指＋无名指轻捏 |
| 拖拽 | 食拇保持捏合，按下后约 0.30 秒进入拖拽；松食指放下 |
| 滚轮 | 食中指做 V，无名指/小指弯曲，拇指与指尖分开；保持片刻后上下轻移翻页，收指停止 |
| 锁定 / 换位 | 松中指锁住位置，仍可点击；捏回后从原位置继续 |
| 停止 | 全局 Esc 或窗口「停止控制」 |

右键、拖拽、滚轮、预览、置顶和本地记录都有开关。「展开手感设置」中的移动灵敏度滑条为 **0.10×～2.00×**，向左更慢，1.00× 是原默认速度，原最慢的 40% 档对应 0.75×；松开滑条才应用，支持恢复默认和保存旧设置。滚动速度单独调整。修改参数后重新启用鼠标控制。

窗口显示处理 FPS、推理耗时、读帧后帧龄、跟踪丢失及按钮/滚轮发送计数，不冒充真人准确率。默认只保存本地性能与事件摘要，不保存视频或图像。

右侧「控制手」可选自动锁定、只用右手或只用左手。自动模式先只露出要控制的一只手，看到锁定提示后另一只手可入镜；控制手离开时暂停、释放按键，不自动换到另一只手。换手需重新启动相机，或更改控制手选项后启动相机。双手太近、遮挡或左右手类别不明确时会暂停，分开并重新捏住拇中后恢复。

支持一只控制手、主屏幕，尚不支持跨屏。macOS 和 Windows 共用核心和界面，Windows 真机仍待验收。规则交互不等于已完成课程 ML 工作：当前使用 MediaPipe 预训练跟踪，没有项目真人训练数据或自训练深度网络；后续仍需数据、模型对比、动作成本与舒适度实验。

## 研究界面与旧基线

应用内目标任务、静止抖动诊断和原有采集/训练工具保留。Mac 使用 `launch_tripod_mac.command`，Windows 使用 `launch_tripod_windows.cmd`，或 `uv run pinchpilot gui --interaction tripod`。这个研究入口中的三指模式仍只操作应用内练习区；桌面控制使用上面的独立入口。

原捏合基线使用 `launch_mac.command` / `launch_windows.cmd` 或 `uv run pinchpilot gui`：手掌定位、食拇捏合点击/拖拽，V 手势滚动；它与三指方案的手势定义不同。旧单指因用户报告严重晃动和难用已搁置。

## 已暂搁置的单指 Demo

用户实测反馈为抖动很大、难以使用，因此暂停此方向。以下入口仅保留用于复现对照，不再作为优先试用方案；历史工程检查不代表真人可用。

先退出旧窗口，再双击 `launch_flex_mac.command` 试主动轻弯，或 `launch_dwell_mac.command` 试停留点击。Windows 使用对应的 `launch_flex_windows.cmd` / `launch_dwell_windows.cmd`。这些入口只选择模式，不自动打开相机。

启动摄像头 → 让前臂获得支撑、手部保持可见 → 舒展食指片刻 → 切到「点击与拖拽实验」→「开始 8 个目标」。也可以从右上方的模式选择框随时切换。**只需要主动操作食指，但不能只露出孤立指尖。** 如果舒适手位不在相机视野内，需先调整机位；本轮不声称软件已经解决悬臂疲劳。

| 模式 | 点击操作 | 反馈与参数 |
|---|---|---|
| 单指 · 轻弯点击 | 定位后轻弯，再恢复；恢复确认时点击一次 | 弯曲时冻结指针；可选较轻、默认、更明显的点击幅度 |
| 单指 · 停留点击 | 先移动，停住等进度环完成 | 默认 0.8 秒，可调 0.4–2 秒；移开取消，点击后需移开才能再点击 |

两种模式都有三档移动灵敏度、重新定位和 Esc 暂停。修改参数或切换模式结束当前任务，请重新开始一轮。练习结果保存在 `reports/tasks/`，含模式和参数；合成演示带独立来源标记。

```sh
uv run pinchpilot gui --interaction finger-flex
uv run pinchpilot gui --interaction finger-dwell
# 无相机回放仅用于查看交互逻辑：
uv run pinchpilot gui --interaction finger-dwell --demo
```

## 原捏合基线的小幅移动与休息

v0.2 默认选择「小幅移动 · 30% 范围」，即相机画面宽高各 30% 的手掌参考点位移映射到全屏。原来需要移动 60% 的画面距离；在相机、姿势不变时，同样屏幕位移所需的画面位移约减半。这不等于疲劳或实际手臂位移已经测得减半。

| 范围 | 相对原版的画面位移 | 适用方式 |
|---|---|---|
| 30%（默认） | 约一半 | 先用这个体验 |
| 20% | 约三分之一 | 想进一步减少位移时尝试；也更容易放大抖动 |
| 60% | 与原版一致 | 对照或需要较低灵敏度 |

默认勾选「握拳休息，张开后从原光标位置继续」：握拳或丢手暂停后，可先调整手位再张开。拖拽时握拳会释放左键、结束拖拽，不能在休息期间继续按住。松开捏合后也保留最后光标位置，减少回跳。启用系统控制时以当前主屏光标为起点；应用内任务以屏幕中心为起点。

虚线框会随重新接管的手位调整。若它靠近画面边缘，可握拳并把手移回相机看得清的位置。范围设置只改变指针灵敏度，滚轮增益保持原有尺度。切换范围/恢复开关会退出系统控制并结束当前任务；个人校准加载或恢复默认仅更改捏合阈值，保留移动设置。

完整对照原来的映射时，选择 **60% 并取消休息后恢复**。命令行离线回放的默认配置仍是原始固定映射；传入校准配置可指定其他映射。新版本不会热更新已运行的程序：先按 Esc 停止控制，退出旧窗口，再用启动脚本重新打开。

## Windows 源码安装与便携包构建

普通试用优先使用上面的便携 ZIP。维护者可运行 `uv run python scripts/build_windows_portable.py` 重建，产物在 `dist/`；构建需要联网，运行不需要下载 Python/模型。构建脚本核对官方运行时、模型与锁定依赖的哈希，并保留第三方许可证。可在 Mac 组装 Windows 预编译依赖，但这不代表已在 Windows 执行验证。

首版目标环境：Windows 10/11 x64、macOS Apple Silicon。当前依赖锁未覆盖 Intel Mac；Windows ARM 暂未列为验收目标。Windows 真机摄像头与输入测试仍待完成。

将整个源码包解压到普通目录，例如 `C:\Projects\PinchPilot`。先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，Windows 可运行：

```powershell
winget install --id=astral-sh.uv -e
```

重新打开终端，再双击 **launch_desktop_windows.cmd**。启动脚本会准备 Python 3.11 虚拟环境并按 `uv.lock` 安装依赖。首次安装需要联网。也可在两种系统上执行：

```sh
uv sync --locked
uv run pinchpilot doctor
uv run pinchpilot desktop --demo
uv run pinchpilot desktop
```

无需 GPU、CUDA、云端账号或 API key。不要把 `.venv` 从 Mac 复制到 Windows；每台电脑由 uv 创建自己的环境。项目目录尽量不要包含冒号；原课程文件夹名称中的冒号曾导致本机 uv 路径解析失败。

## 权限和常见问题

- **Mac 摄像头：** 在「系统设置 → 隐私与安全性 → 摄像头」允许实际启动进程访问。授权后重新启动程序。
- **Mac 系统控制：** 在「辅助功能」中授权启动本程序的 Terminal / 应用。相机权限与辅助功能权限是两项独立权限。`doctor` 只查询，不自动修改授权。
- **Mac 全局 Esc：** 程序也检查输入监控授权；如有提示，在「输入监控」中为启动进程授权并重启。
- **Windows 相机：** 开启「隐私和安全性 → 摄像头 → 允许桌面应用访问摄像头」。被会议软件占用时先释放摄像头，必要时更换相机编号。
- **Windows 输入：** 先在普通权限窗口测试。Win32 输入注入可能无法作用于管理员窗口。程序不自动提升权限。
- **光标太抖 / 误捏合：** 改善光线、减少指尖遮挡、先做个人校准，再采集不同角度的数据。滤波和时间确认会增加少量延迟，需要实测取舍。
- **看不到模型与训练：** 右侧控制栏可向下滚动。默认使用规则；有真实采集后才能训练自有分类器。

## 技术栈

| 层 | 实现 | 用途 |
|---|---|---|
| 桌面界面 | Python 3.11 + PySide6 | 共用界面、预览、任务与采集 |
| 感知 | OpenCV + MediaPipe Hand Landmarker | 相机图像到 21 个手部关键点 |
| 特征与模型 | NumPy + scikit-learn | 归一化几何特征、RF/SVM、分组评测 |
| 交互 | 纯 Python 状态机 | 张开后启用、确认、防抖、按下/释放生命周期 |
| 系统接口 | macOS Quartz / Windows Win32 | 光标移动、左右键、拖拽、旧基线滚动与停止键 |
| 实验产物 | JSONL、joblib、matplotlib | 数据、模型、指标、混淆矩阵 |

MediaPipe 是复用的预训练视觉模型；本项目训练的是关键点之上的手形分类器。个人校准调整规则阈值，不等同于模型微调。当前没有自训练深度网络。

## 数据与训练

完整流程见 [采集与实验协议](docs/EXPERIMENTS.md)。每位参与者用固定编号，场次间重新摆位；标签由录制者选择，不使用规则预测充当监督标签。

```sh
uv run pinchpilot inspect-data --data data/recordings
uv run pinchpilot train --data data/recordings --output reports/forest_01 --model forest
uv run pinchpilot train --data data/recordings --output reports/svm_01 --model svm
uv run pinchpilot calibrate --participant P01 --output data/profiles/P01.json
```

默认按参与者划分；同一人的帧不会同时出现在训练集与测试集。至少要有两个完整分组才能运行，正式研究建议更多参与者。`--group-by session` 检验新场次，不代表跨人泛化。训练会产生 `model.joblib`、`metrics.json`、`confusion.png`；报告包含拒识后的表现，并保留不拒识的诊断分数。只有你信任的本项目模型才应加载，因为 joblib 使用 Python pickle。

原始视频不保存、不上传。关键点录制位于 `data/recordings/`；校准位于 `data/profiles/`；离散交互事件位于 `data/events/`；评测和任务记录位于 `reports/`。这些目录默认不进入 Git 和源码交付包。

## 工程检查与阅读入口

```sh
uv run ruff check src tests
uv run pytest -q
uv run pinchpilot desktop --demo --smoke-seconds 3
uv run pinchpilot gui --demo --smoke-seconds 3
uv run python scripts/check_single_finger.py
uv run python scripts/check_tripod.py
```

- [项目定位、课程要求与里程碑](docs/PROJECT_PLAN.md)
- [桌面测试版启动、数据与开关](docs/DESKTOP_TRIAL.md)
- [采集与实验协议](docs/EXPERIMENTS.md)
- [验证记录与待实测事项](docs/VALIDATION.md)
- [设计与模块边界](docs/superpowers/specs/2026-09-21-pinchpilot-design.md)
- [小幅移动设计](docs/superpowers/specs/2026-09-21-small-motion-design.md)
- [单指 Demo 操作与试用记录](docs/SINGLE_FINGER_DEMOS.md)
- [三指定位点击与抗抖诊断](docs/TRIPOD_DEMO.md)
- [右键与拖拽体验顺序](docs/RIGHT_CLICK_DRAG_TRIAL.md)
- [少动作、小幅度的目标与指标](docs/superpowers/specs/2026-09-21-minimal-motion-design.md)
- [参考与第三方组件](THIRD_PARTY.md)

状态机与输入结构测试使用合成帧/模拟接口，不能视为真实摄像头识别率或 Windows 真机验收。CI 配置已经提供；当前仓库仅在本地，未执行远程 CI。
