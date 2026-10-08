# 三指接触 ML · v0.13 第一阶段

本轮完成的是可运行的采集、训练和离线评测工具。兼容定位和日常鼠标仍使用原有规则；当前生成的RF/CNN模型只用合成样例验证流程，不能据此宣称真人准确率或舒适度提高。

## 训练什么

继续使用MediaPipe预训练Hand Landmarker取得21个关键点，在过去一小段关键点上训练三个独立输出：`middle`拇中捏合、`index`食拇接触、`ring`拇无名指接触。三个输出可以同时为真。不是重新训练整套视觉网络；MediaPipe的图像坐标/估计深度与真实接触、压力、用户意图也不等价。[官方输出说明](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

RF用三个独立二分类器，CNN用两层仅向过去填充的1D卷积，最后时间步输出三个分数。两者输入相同的8帧×67维特征。30fps时是约233ms历史，冷启动要收够8帧；已经连续运行时每来一帧便可预测，不等待未来帧。它仍可能产生识别响应延迟，需要后续真实事件标注评估。

特征是宽高比修正后的21×3掌中心/掌尺度归一化坐标、三组距离比、真实帧间隔。XY尺度退化、缺手、非递增时间、换手或超过0.12秒间隔会重置历史；窗口总时长不得超过0.8秒。本阶段训练和回放都要求原几何特征可用，严重侧掌或丢点不属于已解决范围。

单个接触分数不能直接替代现有“接近冻结→确认按下→保持拖拽→分开释放”状态机。手部接触也可能无意发生，本阶段不训练“有意点击”标签。

## 采集自己的动作

v0.14.0 起默认按屏幕提示录制。组员录制时看一页的 [录制说明](RECORDING_PROTOCOL.md) 即可，这里说明程序做了什么。

1. 退出旧窗口，使用源码桌面入口，确认标题0.14.0。启动相机预览、确认控制手。
2. 右侧可滚动设置区点「ML 关键点采集 · 默认不录制」。系统鼠标会停止，实体鼠标可继续操作按钮。
3. 填匿名用户编号，编号保持一致。场次名每录完一次自动换新；换录制片段不等于换人。勾选保存关键点后，点「开始采集」。
4. 「按屏幕提示录制」默认勾选：程序按固定顺序提示动作，共5轮、约14分钟，每轮换一种手部朝向。内容包括拇中捏住与小幅移动、捏住时食指点击和按住、单独食指点击、拇无名指点击和按住、靠近但不接触的负例和自然调整。提示要求时，录制者用另一只手按住空格标记指尖接触。取消勾选则是不带提示的自由录制，片段由录制者自己安排。
5. 按提示录完会自动停止并保存；也可点「停止并保存」提前结束，这一次记为不完整。关闭采集窗后需要手动重新启用系统鼠标。Esc、切换控制设置、打开测试台也会结束采集窗口。

每次采集在 `data/contact_recordings/` 保存 `.jsonl`连续21点及时间、`.labels.json`标注文件、`.capture.json`帧数和停止原因。按提示录制时 `.labels.json` 是未复核的草稿而不是空白模板，另有 `.protocol.json` 记录每一步的起止时间、空格按下和松开的时间、没有收到空格的步骤，并把这四个文件打包成一个 `.zip` 便于交付。v0.14.0 起每帧同时保存MediaPipe的世界坐标关键点（米，仅供后续分析，现有特征和控制不读取），坐标保留6位小数。

无手帧保留，不伪造停流期间帧。窗口关闭/来源变化/重复帧/取消勾选都会停止保存，不自动恢复。画面过期或帧间隔过长在自由录制时同样停止保存；按提示录制时容忍5秒以内的相机卡顿，并在 `.capture.json` 的 `camera_gaps` 里计数，超过5秒才停止。没有视频、桌面截图、输入文本或上传，文件不自动加入Git。可在关闭采集后删除同名前缀的文件。

普通日常日志仍只保存摘要。打开采集窗本身不录制，合成演示不能启用该采集窗的录制按钮。训练工具不要求启动相机。

## 标注与分区

按提示录制的草稿用 `ml-review` 逐段复核，组员步骤见 [标签检查说明](LABEL_REVIEW.md)。自由录制的空白模板仍需人工编辑 `.labels.json`，没有逐帧编辑器。任何情况下都不能用规则输出自动填成“正确答案”。

按提示录制生成的草稿标明 `reviewed: false` 和 `label_source: "protocol_draft"`，只是复核的起点。`ml-train` 和正式结果拒绝未复核的标注；`ml-cv --draft` 可以用草稿试跑，输出标明草稿，并且不能计算测试结果。草稿里中指一路来自屏幕提示，食指和无名指来自空格。没有手的帧、提示出现后0.8秒、提示结束前0.3秒、每次按下或松开空格前后0.07秒都留空；需要空格的步骤如果整步没有收到空格，对应一路在这一步留空。
不能可靠判断的区间保留`null`，接触转换的模糊边界也留空。仅有动作提示、按键时间或关键点接近不足以证明实际接触，需结合人工观察/参与者确认，并明确记录不确定范围。

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

## 复核草稿标签（`ml-review`）

```sh
uv run pinchpilot ml-review <压缩包或文件夹> --annotator <编号>
uv run pinchpilot ml-review <文件夹> --status
```

**单元**：复核以单元为单位。提示记录里每个带标签的步骤是一个单元；连续的、短于4秒且不需要空格的步骤合成一个单元（四次“捏住/松开”合成一个）。完整的一次录制是60个单元，其中30个需要空格。

**复核者能做的事**只有三样：每个单元通过或作废；整段录制共用一个时间偏移，把空格标记整体提前，用来抵消反应时间；调整三个留空时长（提示出现后、提示结束前、按下松开前后）。标签按这些参数从提示记录和空格事件重新推出，作废的单元全部留空。改动参数会把已通过的单元退回，因为它们的标签变了。程序不按关键点或距离曲线修改任何标签：模型的输入就是这些关键点，那样会把特征泄漏进标签。

**文件**（都写在压缩包旁边，名字以压缩包名开头）：

- `.review.json`：复核进度，每次决定后自动保存。
- `.labels.json`：全部单元都有结论后才能写出，`reviewed: true`、`label_source: "human_reviewed"`，并带一个 `review` 块记录参数、单元数和作废的单元。`ml-pool` 与 `ml-cv` 优先读它。
- `.crosscheck.<编号>.json`：另一人的抽查结果。

**抽查**：编号与录制者不同的人打开时进入抽查。样本由录制哈希和抽查者编号决定，固定为 max(6, 单元数的十分之一) 个已通过的单元，其中至少一半需要空格。抽查者看到的是录制者的参数和标签，只给出同意或不同意。抽查结果绑定到当时那份 `.labels.json` 的哈希；录制者重新保存后旧结果失效。录制者再次打开时能看到针对当前标签的不同意意见。`--status` 列出每段录制的进度、参数、作废数和各次抽查的同意数。

**局限**：没有保存视频，复核看到的是关键点回放和拇指尖到各指尖的距离曲线，能发现时间没对齐、漏按和动作做错，不能证明真实接触。时间偏移整段只有一个值；反应时间在一次录制中变化较大时，只能加宽留空或作废单元。抽查样本小，同意比例只是粗略的一致性指标。

## 多人比较：留一人验证

组员使用步骤见 [模型训练与比较说明](MODEL_COMPARISON.md)，这里说明程序做了什么。`ml-train` 仍按清单里写死的 train/validation/test 训练并保存一个模型；多人比较用下面两条命令，不保存模型。

```sh
uv run pinchpilot ml-pool <录制文件夹> [--draft]
uv run [--extra ml] pinchpilot ml-cv <录制文件夹> --settings experiments/contact/<模型>.json [--draft] [--final-test]
```

**数据**：直接读文件夹里按提示录制生成的压缩包，以及散放的 `.jsonl` 加 `.labels.json`。压缩包旁边如有同名的 `<压缩包名>.labels.json`，就用这份复核过的标签代替包内草稿。同一段录制以两种形式出现只算一次；内容相同只是起始时间不同的副本、合成与真人混放都会被拒绝。窗口和特征与 `ml-train` 相同（8帧×67维），另外记下每个窗口属于哪个人、提示记录里的第几轮和哪一步。读入结果按输入文件的哈希缓存到 `data/cache/contact_pool/`。

**划分**：每人轮流作为留出者。对留出者 T，其余每人 V 轮流做验证，用除 T、V 以外的人训练。三个人时每组设置有6次验证（训练3次，每次训练同时给两个人打分，各自归入不含自己的那一折）。调参只输出这些验证分数。CNN 的训练轮数由验证者的损失决定，两个验证者各自保留自己最好的一轮。

**选择与测试**：每一折只按自己的验证结果在候选设置里选一组，所以留出者的数据不参与它那一折的任何选择。`--final-test` 才给留出者打分：用选中的设置在其余所有人上重新训练（CNN 的轮数取该折验证选出的平均值），每个留出者只打一次分。草稿标签不能运行 `--final-test`。

**四种模型**（设置文件里的 `model`）：

- `rules`：每根手指一个距离阈值，在训练者身上取F1最高的切点；可对最近几帧取中位数等统计量。
- `forest`：三个独立的随机森林，输入是拉平的最近几帧。
- `cnn`：与 `ml-train` 相同结构的因果卷积网络，宽度可调；32通道时12,739个参数。
- `finger_cnn`：把输入拆成中指、食指、无名指三份，每份只含拇指4个关节、该手指4个关节、指尖差、该手指的距离比和帧间隔（29维）。三份过同一个两层因果卷积（`shared`），输出前拼上另外两根手指向量的逐元素最大值（`context`）；24通道时5,644个参数。两个开关各自对应一个对照实验。

表里另有一行“产品现有的固定规则”，是 `rule_contacts` 在同样窗口上的分数，仅作参考。

**局限**：判定门槛固定为0.5，分数未校准。相邻窗口高度相关，窗口数不是独立样本数。只有三个人时，每折验证的训练只来自一个人，验证分数会低于最终用两个人训练的模型，适合比较设置，不适合当作最终表现。人工调设置时看到的是所有折的验证分数，这一点留出者的数据间接参与了，报告里要写明。每折选出的设置可能不同，测试表会列出各折实际用的那一组。

## 无需真人数据的工程检查

```sh
uv run pinchpilot ml-fixture --output data/contact-synthetic
uv run pinchpilot ml-inspect data/contact-synthetic/manifest.json
uv run pinchpilot ml-train data/contact-synthetic/manifest.json --output models/contact-synthetic-rf --model forest --allow-synthetic
uv run --extra ml pinchpilot ml-train data/contact-synthetic/manifest.json --output models/contact-synthetic-cnn --model cnn --epochs 8 --allow-synthetic
```

多人比较的合成样例用 `ml-fixture --pool` 生成（3人×2场次），再用 `ml-cv … --allow-synthetic` 运行。

合成目录需为空，合成训练默认拒绝，必须显式`--allow-synthetic`，报告会标明`synthetic_engineering`。这组样例刻意简单，规则基线很强；模型取得高分不证明可以改善真实摄像头表现。开发测试用`uv run --extra ml pytest -q`；不安装extra时CNN专用测试会跳过。

## 后续优化优先级

先用真实录制找出“小幅接触漏判、姿态改变误断、自然调整时误识别”等具体情况，再评估RF和CNN是否优于规则。需要有意点击标签才能衡量误触；需要目标轨迹才能监督去抖，当前接触数据不能替代这两种标注。点击前后的位移干扰、滤波滞后、相机取景和跨平台响应也继续按0.12测试台/诊断记录推进。
