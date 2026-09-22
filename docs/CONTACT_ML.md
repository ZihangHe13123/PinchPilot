# 三指接触 ML · v0.13 第一阶段

本轮完成的是可运行的采集、训练和离线评测工具。兼容定位和日常鼠标仍使用原有规则；当前生成的RF/CNN模型只用合成样例验证流程，不能据此宣称真人准确率或舒适度提高。

## 训练什么

继续使用MediaPipe预训练Hand Landmarker取得21个关键点，在过去一小段关键点上训练三个独立输出：`middle`拇中捏合、`index`食拇接触、`ring`拇无名指接触。三个输出可以同时为真。不是重新训练整套视觉网络；MediaPipe的图像坐标/估计深度与真实接触、压力、用户意图也不等价。[官方输出说明](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

RF用三个独立二分类器，CNN用两层仅向过去填充的1D卷积，最后时间步输出三个分数。两者输入相同的8帧×67维特征。30fps时是约233ms历史，冷启动要收够8帧；已经连续运行时每来一帧便可预测，不等待未来帧。它仍可能产生识别响应延迟，需要后续真实事件标注评估。

特征是宽高比修正后的21×3掌中心/掌尺度归一化坐标、三组距离比、真实帧间隔。XY尺度退化、缺手、非递增时间、换手或超过0.12秒间隔会重置历史；窗口总时长不得超过0.8秒。本阶段训练和回放都要求原几何特征可用，严重侧掌或丢点不属于已解决范围。

单个接触分数不能直接替代现有“接近冻结→确认按下→保持拖拽→分开释放”状态机。手部接触也可能无意发生，本阶段不训练“有意点击”标签。

## 采集自己的动作

1. 退出旧窗口，使用源码桌面入口，确认标题0.13.0。启动相机预览、确认控制手。
2. 右侧可滚动设置区点「ML 关键点采集 · 默认不录制」。系统鼠标会停止，实体鼠标可继续操作按钮。
3. 填匿名用户编号和场次。编号保持一致；换录制片段不等于换人。勾选保存关键点后，点「开始采集」。
4. 做正常拇中移动、轻食拇点击、保持食拇、松中指/重新捏住、拇无名指、普通手部调整等片段。包括较小幅度和自然转腕，也保留不接触的负例。第一轮先检查机位与数据可见性，不要求一次录大量数据。
5. 点「停止并保存」。关闭采集窗后需要手动重新启用系统鼠标。Esc、切换控制设置、打开测试台也会结束采集窗口。

每次采集在 `data/contact_recordings/` 保存 `.jsonl`连续21点及时间、`.labels.json`空白标注模板、`.capture.json`帧数和停止原因。无手帧保留，不伪造停流期间帧。窗口关闭/画面过期/来源变化/重复帧/取消勾选都会停止保存，不自动恢复。没有视频、桌面截图、输入文本或上传，文件不自动加入Git。可在关闭采集后删除这三个同名前缀文件。

普通日常日志仍只保存摘要。打开采集窗本身不录制，合成演示不能启用该采集窗的录制按钮。训练工具不要求启动相机。

## 标注与分区

**尚未提供逐帧标注编辑器**。需要人工编辑生成的 `.labels.json`；我也可以按后续确定的人工标签协助整理，但不能用规则输出自动填成“正确答案”。不能可靠判断的区间保留`null`，接触转换的模糊边界也留空。仅有动作提示、按键时间或关键点接近不足以证明实际接触，需结合人工观察/参与者确认，并明确记录不确定范围。

区间使用从0起算的帧号，`start`包含、`end`不包含。三路各为0（未接触）、1（接触）、null（未知）；窗口末帧有任一路未知便不进入本期训练。审核完才将`reviewed`改为true、`label_source`改为`human_reviewed`，填写`annotator`。录制SHA256不得手动忽略或随意替换。

```json
"intervals": [
  {"start": 0, "end": 40, "middle": 1, "index": 0, "ring": 0},
  {"start": 40, "end": 50, "middle": null, "index": null, "ring": null},
  {"start": 50, "end": 70, "middle": 1, "index": 1, "ring": 0}
]
```

在数据目录新建manifest.json，文件路径相对于清单目录：

```json
{
  "schema": "pinchpilot-contact-dataset-v1",
  "group_by": "participant",
  "recordings": [
    {"split": "train", "recording": "p1.jsonl", "annotation": "p1.labels.json"},
    {"split": "validation", "recording": "p2.jsonl", "annotation": "p2.labels.json"},
    {"split": "test", "recording": "p3.jsonl", "annotation": "p3.labels.json"}
  ]
}
```

先固定参与者分区，再构建窗口。每一分区的三路都需包含正、负样本；至少3个独立分组只是让流程可运行的最低条件，不能当作充分的泛化样本量。同一人不能放入不同participant分区；仅研究个人跨场次时用`group_by: "session"`，此时按用户+场次划分，结论仅限跨场次。重复episode、复制录制、首尾空格变造的身份会被检测。

旧 `open/pinch/scroll/other` 录制可导入，但原标签不用作三路接触真值，且旧文件没有可靠来源标记。需单独生成并复核新标签、明确`recording_source`；普通trial日志没有关键点，不能导入训练。合成与真人数据禁止混入同一个实验。

## 训练和回放

普通桌面运行与RF不需要PyTorch；CNN作为可选`ml`依赖，Mac/Windows共用代码。命令在项目根目录执行：

```sh
uv run pinchpilot ml-inspect data/contact_recordings/manifest.json
uv run pinchpilot ml-train data/contact_recordings/manifest.json --output models/contact-rf --model forest
uv run --extra ml pinchpilot ml-train data/contact_recordings/manifest.json --output models/contact-cnn --model cnn --epochs 20
uv run --extra ml pinchpilot ml-shadow data/contact_recordings/p3.jsonl --model models/contact-cnn --output reports/contact-shadow.json
```

每次训练用一个新输出目录。RF保存`model.joblib`，只加载本项目生成且可信的本地模型；CNN保存数字数组`weights.npz`。二者附带参数、特征版本、分组、源数据指纹、模型哈希、依赖版本与`metrics.json`。CNN归一化只用训练集，按验证集loss选择epoch，测试集最后评分；模型保存的是被评测的那份权重，不在测试集上重新拟合。

报告包含各通道混淆矩阵、precision/recall/F1及各组结果；固定阈值0.5，分数未经概率校准。规则对照是当前默认兼容模式的瞬时几何门槛，未运行完整按键状态机。相邻窗口相关，不能把窗口数当作独立参与者数。

`ml-shadow`只在离线录制中记录模型建议、规则观察及本地推理耗时，不输出系统输入、不代表点击成功率。无手或冷启动期间分数为未知。实时桌面旁路、接触起止事件误差、误点击/漏点击、逐帧标注编辑器和真人A/B仍未完成。

## 无需真人数据的工程检查

```sh
uv run pinchpilot ml-fixture --output data/contact-synthetic
uv run pinchpilot ml-inspect data/contact-synthetic/manifest.json
uv run pinchpilot ml-train data/contact-synthetic/manifest.json --output models/contact-synthetic-rf --model forest --allow-synthetic
uv run --extra ml pinchpilot ml-train data/contact-synthetic/manifest.json --output models/contact-synthetic-cnn --model cnn --epochs 8 --allow-synthetic
```

合成目录需为空，合成训练默认拒绝，必须显式`--allow-synthetic`，报告会标明`synthetic_engineering`。这组样例刻意简单，规则基线很强；模型取得高分不证明可以改善真实摄像头表现。开发测试用`uv run --extra ml pytest -q`；不安装extra时CNN专用测试会跳过。

## 后续优化优先级

先用真实录制找出“小幅接触漏判、姿态改变误断、自然调整时误识别”等具体情况，再评估RF和CNN是否优于规则。需要有意点击标签才能衡量误触；需要目标轨迹才能监督去抖，当前接触数据不能替代这两种标注。点击前后的位移干扰、滤波滞后、相机取景和跨平台响应也继续按0.12测试台/诊断记录推进。
