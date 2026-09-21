# PinchPilot

用普通电脑摄像头进行捏合交互的课程研究原型。移动手掌定位；拇指与食指捏合点击，捏住移动拖拽；V 手势上下移动滚动。目标是让前臂可以获得支撑，用较小动作完成日常点击和拖动，并用实验检验效果。

**当前原型版本为 v0.2.0。** 已实现小幅移动与休息后恢复、预览、手势状态机、Mac/Windows 系统接口、人工标签采集、个人阈值校准、Random Forest/SVM 训练、独立分组评测，以及点击/拖拽任务。尚无项目真人训练数据；ML 的准确率、误触率和舒适度尚未得到验证。

## 立即启动

这台 Mac 的代码位于 `/Users/zhihang/Project/PinchPilot`，运行环境已安装。双击 **launch_mac.command**，或在项目目录执行：

```sh
uv run pinchpilot gui
```

首次试用先点击「无相机演示」查看界面；它使用合成关键点，不会读取相机或操作系统光标。点击「启动摄像头」开始真实预览。首次启动会请求相机权限，手部模型需要联网下载一次（约 8 MB），随后缓存在本机。

1. 前臂放在桌面或扶手上，让相机看清手掌及全部指尖。
2. 舒适地分开拇指与食指，小幅移动手掌观察光标；虚线框对应主屏全范围，不必把整个手掌放进框内。
3. 捏合并松开产生一次点击。保持捏合并移动，超过拖动阈值后进入拖拽。
4. 比 V 手势并上下移动滚动；退出滚动后先回到张开定位。
5. 握拳暂停移动，将手放回舒服的位置，再分开拇指食指继续；默认从刚才的光标位置接着操作。
6. 先在「点击与拖拽实验」中练习。要控制桌面时，再点击「启用系统鼠标控制」。

Esc 停止系统控制并暂停手势；也可点击「停止全部」。丢手超过 0.2 秒会释放按键，恢复后需先张开手。首版限定单手、主屏，不包含右键、缩放、多屏、动态手势网络和完整鼠标替代功能。

## 小幅移动与休息

v0.2 默认选择「小幅移动 · 30% 范围」，即相机画面宽高各 30% 的手掌参考点位移映射到全屏。原来需要移动 60% 的画面距离；在相机、姿势不变时，同样屏幕位移所需的画面位移约减半。这不等于疲劳或实际手臂位移已经测得减半。

| 范围 | 相对原版的画面位移 | 适用方式 |
|---|---|---|
| 30%（默认） | 约一半 | 先用这个体验 |
| 20% | 约三分之一 | 想进一步减少位移时尝试；也更容易放大抖动 |
| 60% | 与原版一致 | 对照或需要较低灵敏度 |

默认勾选「握拳休息，张开后从原光标位置继续」：握拳或丢手暂停后，可先调整手位再张开。拖拽时握拳会释放左键、结束拖拽，不能在休息期间继续按住。松开捏合后也保留最后光标位置，减少回跳。启用系统控制时以当前主屏光标为起点；应用内任务以屏幕中心为起点。

虚线框会随重新接管的手位调整。若它靠近画面边缘，可握拳并把手移回相机看得清的位置。范围设置只改变指针灵敏度，滚轮增益保持原有尺度。切换范围/恢复开关会退出系统控制并结束当前任务；个人校准加载或恢复默认仅更改捏合阈值，保留移动设置。

完整对照原来的映射时，选择 **60% 并取消休息后恢复**。命令行离线回放的默认配置仍是原始固定映射；传入校准配置可指定其他映射。新版本不会热更新已运行的程序：先按 Esc 停止控制，退出旧窗口，再用启动脚本重新打开。

## Windows 安装

首版目标环境：Windows 10/11 x64、macOS Apple Silicon。当前依赖锁未覆盖 Intel Mac；Windows ARM 暂未列为验收目标。Windows 真机摄像头与输入测试仍待完成。

将整个源码包解压到普通目录，例如 `C:\Projects\PinchPilot`。先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，Windows 可运行：

```powershell
winget install --id=astral-sh.uv -e
```

重新打开终端，再双击 **launch_windows.cmd**。启动脚本会准备 Python 3.11 虚拟环境并按 `uv.lock` 安装依赖。首次安装需要联网。也可在两种系统上执行：

```sh
uv sync --locked
uv run pinchpilot doctor
uv run pinchpilot gui --demo
uv run pinchpilot gui
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
| 系统接口 | macOS Quartz / Windows Win32 | 光标移动、左键、滚动与停止键 |
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
uv run pinchpilot gui --demo --smoke-seconds 3
```

- [项目定位、课程要求与里程碑](docs/PROJECT_PLAN.md)
- [采集与实验协议](docs/EXPERIMENTS.md)
- [验证记录与待实测事项](docs/VALIDATION.md)
- [设计与模块边界](docs/superpowers/specs/2026-09-21-pinchpilot-design.md)
- [小幅移动设计](docs/superpowers/specs/2026-09-21-small-motion-design.md)
- [参考与第三方组件](THIRD_PARTY.md)

状态机与输入结构测试使用合成帧/模拟接口，不能视为真实摄像头识别率或 Windows 真机验收。CI 配置已经提供；当前仓库仅在本地，未执行远程 CI。
