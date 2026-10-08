# 模型训练与比较说明

每人负责一个模型。所有模型用同一批录制、同一种验证方式和同一条命令，区别只在各自的设置文件。你要做的是改设置文件、看验证结果、把最后的设置和结果交上来。

## 准备（只做一次）

1. 在 PinchPilot 文件夹里运行 `git pull`。
2. 把数据仓库下载到本机：`git clone https://github.com/ZihangHe13123/PinchPilot-data.git C:\Projects\PinchPilot-data`。以后每次开始前在这个文件夹里运行 `git pull`，拿到大家新交的录制。
3. 两个 CNN 模型要用 PyTorch：命令写成 `uv run --extra ml pinchpilot …`，第一次运行会自动下载，约几百 MB。规则和随机森林不需要。

下面的命令都在 PinchPilot 文件夹里运行。Mac 上把路径换成自己的写法即可。

## 先看有哪些数据

```
uv run pinchpilot ml-pool C:\Projects\PinchPilot-data\recordings --draft
```

它列出每个人交了几段录制、有多少可用的窗口、每根手指有多少接触样本。三个人都有录制之后才能跑验证。

## 跑验证

```
uv run pinchpilot ml-cv C:\Projects\PinchPilot-data\recordings --settings experiments\contact\forest.json --draft
```

- `--settings` 换成自己模型的文件：`rules.json`、`forest.json`、`cnn.json`、`finger_cnn.json`。
- `--draft` 表示接受还没检查的草稿标签。标签检查完成后去掉它，程序就只用检查过的标签。**带 `--draft` 跑出来的数字只用来试设置，不能写进报告。**
- 结果直接显示在屏幕上，同时保存到 `reports\contact_cv\模型_时间\`。其中 `report.md` 可以直接贴到 issue 里，`settings.json` 是这次用的设置。
- 第一次运行要读入全部录制，约 1 分钟；之后只要录制和标签没变就直接用缓存。

## 调设置

用文本编辑器打开 `experiments\contact\你的模型.json`。`candidates` 里每个 `{…}` 是一组设置，`name` 是你给它起的名字。改数字，或者复制一组改成新的，保存后重新运行上面的命令，表里会多出对应的行。一次最多 24 组。

| 项 | 含义 | 适用模型 |
|---|---|---|
| `frames` | 用最近几帧，1 到 8 | 全部 |
| `stride` | 训练时每几个窗口取一个；相邻窗口很像，取大一些训练更快 | 全部 |
| `statistic` | 对最近几帧的距离取什么：`last`、`median`、`mean`、`min` | 规则 |
| `trees`、`depth`、`min_leaf` | 树的数量、树的深度、叶子最少样本数 | 随机森林 |
| `channels` | 网络的宽度 | 两个 CNN |
| `epochs` | 最多训练几轮；程序会自动停在验证效果最好的那一轮 | 两个 CNN |
| `learning_rate`、`batch_size`、`weight_decay` | 学习率、每批样本数、权重衰减 | 两个 CNN |
| `positive_weight` | 接触样本的权重；填 `"balanced"` 按比例自动加权 | 两个 CNN |
| `shared` | 三根手指是否共用一套参数 | 手指共享 CNN |
| `context` | 每根手指的输出是否参考另外两根手指 | 手指共享 CNN |

写错项名或数值超出范围时，程序会指出是哪一组的哪一项。

## 怎么读结果

- **验证方式**：每次留出一人完全不参与。在剩下的人里，轮流用一人验证、用其余的人训练。三个人时一共 6 次验证，表里是它们的平均值。
- **平均 / 中指 / 食指 / 无名指**：F1 分数，1 最好。接触判定的门槛固定为 0.5。
- **被几折选中**：每一折按自己的验证结果挑一组设置，这一列是该组被挑中的次数。
- **产品现有的固定规则（参考）**：程序现在用的固定阈值在同样数据上的分数，作为对照。
- **各轮的平均 F1**：一次录制的 5 轮对应 5 种手部朝向，这里能看出哪种朝向容易错。
- **错得最多的步骤**：最好的那组设置在哪个动作、哪根手指上判错最多。

## 测试结果

留出的那个人的分数是测试结果，要加 `--final-test` 才会计算。**这个开关只在 10/23 统一运行一次，平时不要加**；调设置时反复看测试结果，测试就不再可信。标签没检查完时程序会拒绝运行它。

## 交什么

在自己模型的 issue 里留言：

1. 最终的设置文件内容。
2. 去掉 `--draft` 后跑出的 `report.md`。
3. 5–8 条要点：试过哪些设置、哪种朝向或哪个动作容易错。

## 没有真人数据时试跑

```
uv run pinchpilot ml-fixture --output data\contact-synthetic-pool --pool
uv run pinchpilot ml-cv data\contact-synthetic-pool --settings experiments\contact\forest.json --allow-synthetic
```

这是程序生成的假数据，只用来确认命令能跑通，分数没有意义。
