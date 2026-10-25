"""English interface: an opt-in translation of the Chinese display texts.

The interface is written in Chinese. For English readers (the course video, the report and its screenshots, graders)
the desktop tool, its testbench, the keypoint recording window and the label review can run in English with
`--lang en` or `PINCHPILOT_LANG=en`. `install("en")` wraps the Qt setters and constructors that take display text
(labels, buttons, check boxes, combo items, form rows, table headers, tool tips, window titles, placeholders, spin box
suffixes and QPainter.drawText), so no call site changes and Chinese stays exactly as before. Fixed texts come from
TEXTS; texts with numbers, names or paths from PATTERNS; composed texts are translated line by line and part by part
around " · ", " / ", " → ", "；" and "、". Only what Qt displays is translated: files, labels and reports keep their
Chinese text. A text that is not found stays Chinese and is remembered in MISSING, which a test checks.
"""

import os
import re
from functools import lru_cache

HAN = re.compile(r"[一-鿿]")
LANG = "zh"
MISSING = set()
_installed = False

TEXTS = {
    # desktop window
    "相机关闭": "Camera off",
    "预览 · 未接管鼠标": "Preview · mouse not controlled",
    "处理帧率": "Processing rate",
    "演示帧率": "Demo rate",
    "推理耗时": "Inference time",
    "读帧后帧龄": "Frame age after read",
    "手部跟踪": "Hand tracking",
    "合成手部": "Synthetic hand",
    "按下 / 拖拽就绪": "Presses / drags",
    "本次运行": "This run",
    "本次运行 / 滚轮发送": "This run / scroll events",
    "可见": "visible",
    "未跟踪": "not tracked",
    "—（合成）": "— (synthetic)",
    "启动相机": "Start camera",
    "停止相机": "Stop camera",
    "启用鼠标控制": "Enable mouse control",
    "关闭鼠标控制": "Disable mouse control",
    "停止控制 · Esc": "Stop control · Esc",
    "控制手": "Control hand",
    "只用右手": "Right hand only",
    "只用左手": "Left hand only",
    "自动锁定": "Lock automatically",
    "自动：先只露出要控制的一只手，锁定后不会换手。\n切换选项会停止相机和鼠标控制；重新启动相机后生效。": (
        "Auto: show only the hand you want to use; once locked it will not switch hands.\n"
        "Changing this stops the camera and mouse control; it applies when the camera restarts."
    ),
    "启用右键 · 拇指＋无名指": "Right click · thumb + ring finger",
    "启用拖拽 · 食拇保持捏合": "Drag · hold index on thumb",
    "启用滚轮 · V 手势上下移动": "Scroll · V gesture, move up or down",
    "显示相机预览": "Show camera preview",
    "窗口置顶": "Keep window on top",
    "保存本地测试记录": "Save local test logs",
    "画面边缘提醒": "Frame edge warning",
    "提醒不会改变鼠标移动或点击；控制时也会在桌面显示提示。": (
        "Warnings never change pointer movement or clicks; during control a hint also appears on the desktop."
    ),
    "记录性能、状态和按钮事件；不保存相机图像。": "Logs performance, states and button events; no camera images are saved.",
    "打开测试记录": "Open test logs",
    "打开交互测试台": "Open the testbench",
    "ML 关键点采集 · 默认不录制": "ML keypoint recording · off until you start",
    "查看诊断与分段耗时": "Diagnostics and stage timings",
    "刚才光标抖动": "The pointer just jittered",
    "刚才有迟滞": "It just lagged",
    "刚才点击没响应": "A click just failed",
    "刚才误点击": "It just clicked by mistake",
    "刚才丢手 / 断触": "The hand or contact just dropped",
    "其他问题": "Other problem",
    "标记刚才的问题 · F8": "Mark the last problem · F8",
    "保存最近最多 10 秒诊断摘要；不含图像或关键点。关闭日志时会单独保存。F8 仅在本应用前台有效。": (
        "Saves a diagnostic summary of at most the last 10 seconds, without images or keypoints; it is saved "
        "separately when logging is off. F8 works only while this app is in front."
    ),
    "展开手感设置 ▾": "Show feel settings ▾",
    "收起手感设置 ▴": "Hide feel settings ▴",
    "接触判定": "Contact model",
    "现有规则 · 固定的距离阈值": "Original rule · fixed distance thresholds",
    "谁来判断食指、无名指有没有碰到拇指。默认是内置的时序 CNN；模型只接管“碰到”这一步，松开仍看距离。自己的模型放在工作区的 data/contact_models 文件夹里，每个模型一个子文件夹。": (
        "What decides whether the index or ring finger touches the thumb. The default is the built-in temporal "
        "CNN; a model only decides the touch, release still uses the distance. Put your own models in "
        "data/contact_models in the workspace, one subfolder per model."
    ),
    "上次选的接触模型找不到了，已改回现有规则。": "The contact model chosen last time is gone; back to the original rule.",
    "6 秒静止校准": "6-second stillness calibration",
    "取消静止校准": "Cancel stillness calibration",
    "清除静止校准": "Clear stillness calibration",
    "支撑前臂，拇中捏住并保持静止，食指和无名指移开；校准期间暂停鼠标控制。仅测量静止抖动，不代表动作识别准确率。": (
        "Rest your forearm, pinch thumb and middle finger and hold still, index and ring finger away; mouse "
        "control pauses meanwhile. This measures resting jitter only, not recognition accuracy."
    ),
    "三步腕动方向校准": "Three-step wrist direction calibration",
    "取消腕动方向校准": "Cancel wrist direction calibration",
    "清除腕动方向校准": "Clear wrist direction calibration",
    "① 中立  →  ② 向右  →  ③ 向上": "① Neutral  →  ② Right  →  ③ Up",
    "① 中立": "① Neutral",
    "② 向右": "② Right",
    "③ 向上": "③ Up",
    "支撑前臂，让掌部清晰可见。按提示示范舒服的中立姿态、向右与向上转动后的姿态；每步数据足够且稳定才会继续。不按倒计时切换，可随时取消。只保存方向参数，不保存关键点；校准结束不会自动启用鼠标。侧视图仅演示动作，实际让镜头看清掌部。": (
        "Rest your forearm and keep the palm clearly visible. Show a comfortable neutral pose, then the wrist "
        "turned right and up, as prompted; each step continues only with enough steady data. There is no "
        "countdown and you can cancel at any time. Only the directions are saved, no keypoints, and the mouse is "
        "not enabled afterwards. The side view only illustrates the motion; let the camera see the palm."
    ),
    "摄像头编号": "Camera index",
    "查看合成演示": "Synthetic demo",
    "已切换兼容定位，无需方向校准，请手动启用鼠标。": (
        "Switched to compatible positioning; no direction calibration needed. Enable the mouse by hand."
    ),
    "已清除静止校准；再次启用鼠标控制后生效。": "Stillness calibration cleared; applies when mouse control is enabled again.",
    "准备 1 秒、静止采集 5 秒": "Get ready for 1 s, then hold still for 5 s",
    "静止校准适用于精细 / 自适应模式": "Stillness calibration applies to the fine and adaptive modes",
    "采集中 · 保持拇中捏合并静止；食指和无名指移开。": (
        "Recording · keep thumb and middle pinched and still; index and ring finger away."
    ),
    "原版用于对照；静止校准适用于精细 / 自适应模式。": (
        "The original mode is for comparison; stillness calibration applies to the fine and adaptive modes."
    ),
    "已恢复静止校准 · 可在当前姿势下重新采集。": "Stillness calibration restored · you can record again in this pose.",
    "尚未校准 · 先启动相机，并让控制手清晰可见。": "Not calibrated · start the camera and keep the control hand visible.",
    "已清除腕动方向校准；重新完成三步校准后才能启用腕动控制。": (
        "Wrist direction calibration cleared; finish the three steps again before using wrist control."
    ),
    "腕动方向已校准 · 请手动启用鼠标控制。": "Wrist directions calibrated · enable mouse control by hand.",
    "先自然放稳；再相对中立向右转腕；最后回中立后向上转腕。": (
        "Settle naturally first; then turn the wrist right from neutral; finally return and turn it up."
    ),
    "已恢复腕动方向校准 · 停腕即停；姿势改变后可重新示范方向。": (
        "Wrist calibration restored · a still wrist stops the pointer; show the directions again after a pose change."
    ),
    "腕动尚未校准，暂不能启用鼠标。先启动相机，再按三步提示采集。": (
        "Wrist control is not calibrated, so the mouse cannot be enabled yet. Start the camera and follow the three steps."
    ),
    "正在结束上一次相机/模型任务，请稍后再启动": "Finishing the previous camera or model task; start again in a moment",
    "正在校验手部模型；首次使用会下载约 8 MB…": "Checking the hand model; the first run downloads about 8 MB…",
    "正在检查摄像头权限，请处理系统授权提示。": "Checking camera permission; please answer the system prompt.",
    "摄像头权限未获准，请在系统隐私设置中授权启动程序。": (
        "Camera permission denied; allow the launching app in the system privacy settings."
    ),
    "等待控制手 · 自动模式先只露出一只手": "Waiting for the control hand · in auto mode show one hand only",
    "相机启动中…有画面后点击「启用鼠标控制」。": 'Starting the camera… once you see the picture, click "Enable mouse control".',
    "合成演示 · 不选择真人控制手": "Synthetic demo · no real control hand",
    "合成演示 · 不接相机、不控制系统鼠标；数据不能当真人识别效果。": (
        "Synthetic demo · no camera and no system mouse; its numbers say nothing about real recognition."
    ),
    "鼠标控制已停止 · 相机可继续预览；再次启用会从当前光标接管。": (
        "Mouse control stopped · the preview keeps running; enabling again takes over from the current pointer."
    ),
    "请先完成三步腕动方向校准": "Finish the three-step wrist direction calibration first",
    "合成演示": "Synthetic demo",
    "实时相机": "Live camera",
    "相机等待 / 画面过期": "Camera waiting / stale frames",
    "未接管鼠标": "mouse not controlled",
    "正在控制": "controlling",
    "左键按住": "left button held",
    "按键已松开": "buttons released",
    "右键按住": "right button held",
    "方向校准 · 系统输入已关闭": "Direction calibration · system input off",
    "仍在重试松键，暂不能关闭；请先用实体鼠标松开按键。": (
        "Still retrying a button release, so the window cannot close yet; release the button with a real mouse first."
    ),
    "鼠标控制已停止；模型任务结束后自动关闭。": "Mouse control stopped; the window closes when the model task ends.",
    "这个工作目录已有桌面测试版在运行，请使用已有窗口。": "The desktop tool is already running for this workspace; use that window.",
    "小窗模式 · F9": "Mini view · F9",
    "恢复完整窗口 · F9": "Full window · F9",
    "只留摄像头预览和当前状态，窗口置顶、占地小；双击预览或按 F9 恢复。": (
        "Keeps only the camera preview and the current state in a small window on top; double-click the preview "
        "or press F9 to restore."
    ),
    "已锁定右手": "Right hand locked",
    "已锁定左手": "Left hand locked",
    "已锁定控制手": "Control hand locked",
    "等待右手": "Waiting for the right hand",
    "等待左手": "Waiting for the left hand",
    "等待控制手": "Waiting for the control hand",
    "正在确认右手": "Checking the right hand",
    "正在确认左手": "Checking the left hand",
    "正在确认控制手": "Checking the control hand",
    "另一只手不参与操作": "the other hand is ignored",
    "另一只手不会接管": "the other hand will not take over",
    "先只露出要控制的一只手": "Show only the hand you want to use",
    "请先只露出控制手，或指定左手 / 右手": "Show only the control hand, or choose the left or right hand",
    "双手太近 · 分开后恢复控制手": "Hands too close · separate them to get the control hand back",
    "左手": "left hand",
    "右手": "right hand",
    # camera preview
    "让手势成为输入": "Let gestures be your input",
    "启动摄像头，或先体验无相机演示": "Start the camera, or try the demo without a camera",
    "合成动作演示 · 不代表识别效果": "Synthetic demo · not real recognition",
    "腕动预览 · 稳定点表示屏幕位置": "Wrist preview · the steady dot is the screen position",
    "拇中中点：灰色原始点 · 蓝绿色稳定点": "Thumb–middle midpoint: grey raw · teal smoothed",
    "镜像预览 · 食指相对掌部定位": "Mirrored preview · index relative to the palm",
    "镜像预览 · 虚线内映射至主屏": "Mirrored preview · the dashed box maps to the main screen",
    "第 1 / 8 个目标": "Target 1 / 8",
    "右键命中": "Right-click hit",
    "完成": "Done",
    "已结束": "Ended",
    "同一任务，比较不同输入方式": "One task, different ways of input",
    "选择左键、右键或拖拽任务，结果会保存到本地": "Pick a left-click, right-click or drag task; results are saved locally",
    # control states (tripod)
    "拇指与中指捏住，食指移开，接管指针": "Pinch thumb and middle finger, index away, to take the pointer",
    "拇中定位": "Thumb + middle moves",
    "食拇短捏左键": "a quick index pinch clicks",
    "食拇短捏左键、保持拖拽": "a quick index pinch clicks, holding drags",
    "拇无名指右键": "thumb + ring right-clicks",
    "位置已锁住 · 仍可左/右键 · 移开食指、捏中指继续移动": (
        "Position locked · left and right click still work · index away, pinch the middle finger to move on"
    ),
    "位置已锁住 · 仍可左键 · 移开食指、捏中指继续移动": (
        "Position locked · left click still works · index away, pinch the middle finger to move on"
    ),
    "食指接近拇指 · 指针已冻结 · 继续靠近按下": "Index approaching the thumb · pointer frozen · close in to press",
    "左键已按下 · 松食指单击 · 保持到进度环满可拖拽": (
        "Left button down · release the index to click · hold until the ring fills to drag"
    ),
    "左键已按下 · 松食指完成点击 · 拖拽已关闭": "Left button down · release the index to click · dragging is off",
    "拖拽已就绪 · 保持食拇捏合移动 · 松食指放下": "Dragging · keep the index on the thumb and move · release the index to drop",
    "拖拽位置锁住 · 捏中指可继续拖 · 松食指放下": (
        "Drag position locked · pinch the middle finger to keep dragging · release the index to drop"
    ),
    "V 手势滚动 · 拇指保持分开 · 上移向上翻、下移向下翻 · 收指停止": (
        "V-gesture scroll · keep the thumb apart · move up to scroll up, down to scroll down · fold the fingers to stop"
    ),
    "无名指接近拇指 · 位置已冻结 · 食指保持分开": "Ring finger approaching the thumb · position frozen · keep the index apart",
    "右键已触发 · 松开无名指后可继续操作": "Right click sent · release the ring finger to go on",
    "动作已取消 · 请先分开食指、无名指与拇指，再试": (
        "Action cancelled · move the index and ring finger away from the thumb, then try again"
    ),
    "位置已锁住 · 请先分开食指和拇指，再捏合点击": "Position locked · separate index and thumb first, then pinch to click",
    "已暂停 · 恢复后重新捏住拇中": "Paused · pinch thumb and middle again to resume",
    "V 手势已准备 · 上下轻移滚动；捏中指可接管指针": (
        "V gesture ready · move slightly up or down to scroll; pinch the middle finger to take the pointer"
    ),
    "V 手势已准备 · 上下轻移滚动；静止不翻页，仍可捏合点击": (
        "V gesture ready · move slightly up or down to scroll; holding still does not scroll, pinches still click"
    ),
    "先完成三步腕动方向校准": "Finish the three-step wrist direction calibration first",
    "前臂放稳 · 拇中捏住接管腕动，食指移开": "Rest the forearm · pinch thumb and middle to take wrist control, index away",
    "转腕移动 · 停腕停光标 · 松中指回位 · 食拇点击/保持拖拽": (
        "Turn the wrist to move · still wrist, still pointer · release the middle finger to reposition · "
        "index pinch clicks, holding drags"
    ),
    "捏合暂不稳定 · 指针已暂停，捏稳后原地继续": "Pinch unsteady · pointer paused; it carries on in place once the pinch is steady",
    "滚动已停止 · 食指移开，重新捏中指接管": "Scrolling stopped · index away, pinch the middle finger again to take over",
    "先完成三步腕动方向校准，再启用鼠标": "Finish the three-step wrist direction calibration before enabling the mouse",
    "掌部姿态暂不可用 · 调整机位，让掌部清晰可见后重新捏合": (
        "Palm pose unavailable · adjust the camera so the palm is clear, then pinch again"
    ),
    "转腕超出校准范围 · 回到自然姿势，移开食指后重新捏合": (
        "Wrist beyond the calibrated range · return to a natural pose, index away, then pinch again"
    ),
    "掌部朝向跳变 · 暂停并松键，请回到自然姿势重新捏合": (
        "Palm orientation jumped · paused and buttons released; return to a natural pose and pinch again"
    ),
    "追踪发生跳变 · 暂停移动，请移开食指重新捏住": "Tracking jumped · movement paused; move the index away and pinch again",
    "正在接管 · 保持拇中捏合，食指在外侧": "Taking over · keep thumb and middle pinched, index outside",
    "捏合暂不稳定 · 指针已暂停，请先分开点击手指": "Pinch unsteady · pointer paused; separate the clicking fingers first",
    # feel settings (tripod_controls)
    "兼容定位 · 平移＋转腕": "Compatible · move+turn",
    "指尖位置 · 原版对照": "Fingertip position · original, for comparison",
    "腕动角度 · 旧实验": "Wrist angle · old experiment",
    "兼容定位沿用画面方向，平移和自然转腕均可移动，无需方向校准。原位置和角度模式可选作对照。": (
        "Compatible positioning follows the image directions: moving the hand and turning the wrist both move "
        "the pointer, with no direction calibration. The original position and angle modes remain for comparison."
    ),
    "原版 · 对照": "Original · comparison",
    "精细 · 小幅微调": "Fine · small corrections",
    "自适应 · 实验": "Adaptive · experimental",
    "原版保留已有移动手感；精细模式改善慢速微调；自适应模式随移动速度调整灵敏度。": (
        "Original keeps the earlier feel; fine improves slow corrections; adaptive changes sensitivity with speed."
    ),
    "固定精细（灵敏度可调）": "Fixed fine (adjustable speed)",
    "无需方向校准 · 整手平移与自然转腕连续控制": "No direction calibration · hand movement and wrist turns both steer",
    "恢复默认": "Reset",
    "恢复原默认速度 1.00×": "Back to the default speed 1.00×",
    "向左更慢；1.00× 为原默认，0.75× 为原最慢档": "Left is slower; 1.00× is the default, 0.75× the old slowest setting",
    "慢 0.10×  ← →  快 2.00×": "Slow 0.10×  ← →  fast 2.00×",
    "轻 · 小动作更灵敏": "Light · small moves more responsive",
    "标准": "Standard",
    "强 · 小动作会被抑制": "Strong · small moves damped",
    "默认触碰": "Default touch",
    "宽松 · 提前判定": "Loose · decides earlier",
    "严格 · 需更靠近": "Strict · needs closer contact",
    "0.30 秒 · 默认": "0.30 s · default",
    "0.20 秒 · 较快": "0.20 s · quicker",
    "0.50 秒 · 从容": "0.50 s · relaxed",
    "快 · 长页面": "Fast · long pages",
    "慢 · 精细翻动": "Slow · fine scrolling",
    "定位方式": "Positioning",
    "移动模式": "Movement",
    "移动灵敏度": "Pointer speed",
    "抗抖强度": "Jitter filter",
    "食拇接触": "Index touch",
    "右键触碰": "Right-click touch",
    "拖拽等待": "Drag delay",
    "滚动速度": "Scroll speed",
    "重新定位 · 保留指针": "Recentre · keep the pointer",
    "记录 8 秒静止抖动": "Record 8 s of resting jitter",
    "拇中定位；食拇短捏松开是左键，保持到进度环满可拖，松食指放下。食指移开，拇指＋无名指轻捏是右键。松中指锁住位置，捏回可继续；Esc 暂停。V 手势、拇指分开，上下轻移滚动；收指停止。": (
        "Thumb + middle moves the pointer. A quick index pinch is a left click; hold until the ring fills to drag "
        "and release the index to drop. With the index away, a light thumb + ring pinch is a right click. "
        "Releasing the middle finger locks the position, pinching again continues; Esc pauses. A V gesture with "
        "the thumb apart scrolls when moved slightly up or down; fold the fingers to stop."
    ),
    "腕动采用固定增益精细滤波；下方灵敏度仍可调。切回位置模式恢复原选择。": (
        "Wrist control uses fixed-gain fine filtering; the speed below still applies. Switching back restores your choice."
    ),
    "先完成腕动方向校准。拇中捏住后，手腕转动一段、光标移动一段；停腕即停，松中指可回到舒服姿势再接管。腕动采用固定增益精细滤波，灵敏度仍可调。食拇点击/保持拖拽、拇无名指右键与 V 手势滚动不变；不会根据偏转持续移动。": (
        "Calibrate the wrist directions first. With thumb and middle pinched, each wrist turn moves the pointer a "
        "step; a still wrist stops it, and releasing the middle finger lets you return to a comfortable pose. "
        "Fixed-gain fine filtering, speed adjustable. Index clicks and drags, thumb + ring right-clicks and V "
        "scrolling are unchanged; holding a tilt does not keep the pointer moving."
    ),
    "兼容定位采用固定增益精细滤波；灵敏度仍可调。切回位置对照后恢复原移动模式选择。": (
        "Compatible positioning uses fixed-gain fine filtering; the speed still applies. Switching back restores "
        "your movement choice."
    ),
    "无需方向校准。拇中捏住后，整手平移和自然转腕可连续移动光标，方向沿用画面位置；手停光标就停，松中指可换位再接管。采用固定增益精细滤波，灵敏度仍可调。食拇点击/保持拖拽、拇无名指右键与 V 手势滚动不变。": (
        "No direction calibration. With thumb and middle pinched, moving the hand or turning the wrist moves the "
        "pointer, following the image directions; when the hand stops the pointer stops, and releasing the middle "
        "finger lets you reposition. Fixed-gain fine filtering, speed adjustable. Index clicks and drags, thumb + "
        "ring right-clicks and V scrolling are unchanged."
    ),
    # single-finger controls
    "均衡 · 2.5": "Balanced · 2.5",
    "更省位移 · 3.5": "Less movement · 3.5",
    "精细 · 1.5": "Fine · 1.5",
    "更明显的弯曲": "A clearer bend",
    "较轻 · 容易误触": "Light · easy to trigger",
    "默认轻弯": "Default light bend",
    "点击幅度": "Click depth",
    "停留时间": "Dwell time",
    "重新定位 · 保留当前指针位置": "Recentre · keep the current pointer position",
    "定位后轻弯食指，再恢复，完成点击；弯曲期间指针冻结。": "Bend the index lightly and straighten it to click; the pointer freezes meanwhile.",
    "先移动食指，再停住等进度环完成；点击后移开才能再次点击。": (
        "Move the index, then hold still until the ring completes; move away before clicking again."
    ),
    "先支撑前臂，让相机看清手部。舒展食指保持片刻，再小幅移动。": (
        "Rest your forearm and let the camera see the hand. Stretch the index for a moment, then move a little."
    ),
    "深弯或移出画面可休息；Esc 暂停。仅应用内点击练习。": "Bend deeply or leave the frame to rest; Esc pauses. Practice clicks in this app only.",
    # framing hints and desktop notices
    "PinchPilot · 手部位置提醒": "PinchPilot · hand position",
    "请把控制手放入画面，先找一个前臂有支撑的位置。": "Bring the control hand into view; find a position where the forearm rests.",
    "暂未跟踪到控制手 · 移回画面，食指移开、拇中捏住重新接管。": (
        "Control hand not tracked · move back into view, index away, pinch thumb and middle to take over again."
    ),
    "上沿": "top",
    "下沿": "bottom",
    "左侧": "left",
    "右侧": "right",
    "保持食拇捏合，松中指换位，再捏中指继续拖。": "Keep the index pinched, release the middle finger to reposition, pinch it again to keep dragging.",
    "收回 V 手势，将手移回画面中央，再继续滚动。": "Fold the V, move the hand back to the centre, then scroll again.",
    "先完成当前点击，再松中指换位。": "Finish the current click, then release the middle finger to reposition.",
    "松中指，将手移回画面中央，再捏住继续。": "Release the middle finger, move the hand back to the centre, then pinch to continue.",
    "控制手在画面内 · 松中指可换位，捏回继续。": "Control hand in view · release the middle finger to reposition, pinch again to continue.",
    "启动相机预览，再启用鼠标控制。Esc 随时停止。": "Start the camera preview, then enable mouse control. Esc stops at any time.",
    "请先关闭 ML 关键点采集窗口，再进行校准": "Close the ML keypoint recording window before calibrating",
    "先启动实时相机预览，再进行静止校准": "Start the live camera preview before the stillness calibration",
    "静止噪声校准仅用于指尖位置对照；兼容定位无需此校准，腕动请使用三步方向校准": (
        "Stillness calibration is for the fingertip comparison mode only; compatible positioning needs none, "
        "wrist control uses the three-step direction calibration"
    ),
    "静止校准用于精细 / 自适应模式，请先切换移动模式": "Stillness calibration is for the fine and adaptive modes; switch the movement mode first",
    "上一次松键仍在重试，暂不能校准": "The last button release is still being retried; cannot calibrate yet",
    "准备 1 秒、静止采集 5 秒：支撑前臂，拇中捏住，食指和无名指移开。鼠标控制已关闭。": (
        "Get ready for 1 s, then hold still for 5 s: rest the forearm, pinch thumb and middle, index and ring "
        "away. Mouse control is off."
    ),
    "腕动方向": "Wrist direction",
    "静止": "Stillness",
    "相机画面中断": "camera picture interrupted",
    "相机帧无效或过期": "camera frames invalid or stale",
    "画面来源已切换": "picture source changed",
    "控制设置已改变": "control settings changed",
    "窗口已关闭": "window closed",
    "已取消此次校准": "calibration cancelled",
    "先启动实时相机预览，再进行腕动方向校准": "Start the live camera preview before the wrist direction calibration",
    "请先将定位方式切换为腕动角度 · 旧实验": 'Switch the positioning to "Wrist angle · old experiment" first',
    "腕动方向校准已开始，鼠标控制保持关闭。": "Wrist direction calibration started; mouse control stays off.",
    "腕动方向校准结果不完整": "The wrist direction calibration is incomplete",
    "保留此前校准；本次结果未应用。": "The previous calibration is kept; this result was not applied.",
    "本次未生成校准，请按上述原因调整后重试。": "No calibration was made; adjust as described above and try again.",
    "腕动方向校准已应用；转动一段、光标移动一段，停腕即停。请手动启用鼠标控制。": (
        "Wrist calibration applied; each turn moves the pointer a step and a still wrist stops it. Enable mouse control by hand."
    ),
    "波动偏大，已限制抗抖范围，建议改善手部支撑或机位。": (
        "Jitter is high, so the filter range was limited; better hand support or camera placement would help."
    ),
    "ML 关键点采集窗口打开时，系统鼠标保持关闭": "The system mouse stays off while the ML keypoint recording window is open",
    "测试台只控制虚拟光标；关闭测试台后再手动启用系统鼠标": (
        "The testbench controls only its virtual pointer; enable the system mouse by hand after closing it"
    ),
    "校准期间不能接管鼠标；请等待完成或取消校准": "The mouse cannot be taken during calibration; wait for it to finish or cancel it",
    "上一次松键仍在重试，暂不能启用": "The last button release is still being retried; cannot enable yet",
    "需要实时相机画面；演示、停流或过期画面不能控制鼠标": "Needs a live camera picture; the demo, a stopped stream or stale frames cannot control the mouse",
    "腕动模式尚未校准；请先完成三步腕动方向校准": "Wrist mode is not calibrated; finish the three-step wrist direction calibration first",
    "鼠标控制已启用 · 拇中捏住接管 · Esc 停止": "Mouse control on · pinch thumb and middle to take over · Esc stops",
    "请先关闭 ML 关键点采集窗口": "Close the ML keypoint recording window first",
    "上一次松键仍在重试，暂不能打开测试台": "The last button release is still being retried; cannot open the testbench yet",
    "测试台已打开 · 手势仅控制测试台里的虚拟光标。": "Testbench open · gestures control only the testbench's virtual pointer.",
    "测试台已关闭；需要时请手动重新启用鼠标控制。": "Testbench closed; enable mouse control again by hand when needed.",
    "请先关闭交互测试台": "Close the testbench first",
    "上一次松键仍在重试，暂不能打开采集窗口": "The last button release is still being retried; cannot open the recording window yet",
    "ML 采集准备 · 系统鼠标已关闭；只有明确开始后才保存关键点。": (
        "ML recording ready · system mouse off; keypoints are saved only after you start."
    ),
    "ML 采集窗口已关闭；系统鼠标需手动重新启用。": "ML recording window closed; enable the system mouse again by hand.",
    "已按 Esc 停止鼠标控制": "Esc stopped mouse control",
    "界面响应超时，鼠标控制已停止": "The interface stopped responding, so mouse control stopped",
    "相机画面已过期，鼠标控制已停止": "Camera frames went stale, so mouse control stopped",
    "鼠标控制已关闭": "Mouse control off",
    "未知画面来源": "Unknown picture source",
    "帧无效、重复或过期，控制已停止": "Invalid, repeated or stale frame; control stopped",
    "相机画面中断，腕动方向校准未应用；保留原设置。": "Camera picture interrupted; wrist calibration not applied, previous settings kept.",
    "相机画面中断，静止校准未应用；保留原设置。": "Camera picture interrupted; stillness calibration not applied, previous settings kept.",
    "画面过期或等待相机": "Stale frames or waiting for the camera",
    "鼠标控制已停止": "Mouse control stopped",
    "松键已恢复；鼠标控制保持关闭": "Button release recovered; mouse control stays off",
    "请在系统设置 → 隐私与安全性 → 辅助功能中授权启动本程序的 Terminal / 应用，然后重新启动": (
        "Allow the Terminal or app that starts PinchPilot in System Settings → Privacy & Security → Accessibility, then restart"
    ),
    "全局 Esc 需要输入监控权限；请在隐私与安全性 → 输入监控中授权后重新启动": (
        "The global Esc key needs Input Monitoring; allow it in Privacy & Security → Input Monitoring and restart"
    ),
    "真实控制当前支持 macOS / Windows；其他系统可运行预览和数据工具": (
        "Real control supports macOS and Windows; other systems can run the preview and the data tools"
    ),
    "摄像头无法打开。检查相机编号、系统权限和其他占用相机的程序。": (
        "Cannot open the camera. Check the camera index, system permissions and other apps using the camera."
    ),
    "相机未返回画面，已停止控制": "The camera returned no picture; control stopped",
    "尚未请求摄像头权限": "camera permission not requested yet",
    "摄像头访问受限制": "camera access restricted",
    "摄像头权限已拒绝": "camera permission denied",
    "摄像头权限已授权": "camera permission granted",
    "未知权限状态": "unknown permission state",
    # contact models
    "现有规则": "Original rule",
    "调过的规则": "Tuned rule",
    "时序 CNN": "Temporal CNN",
    "手指共享 CNN": "Finger-shared CNN",
    "内置": "built-in",
    "训练者未记录": "trainer not recorded",
    "旋转＋缩放＋噪声": "rotation + scale + noise",
    "只看距离比": "distance ratios only",
    "随机旋转": "random rotation",
    "食指用画面内距离＋只取近的": "in-frame index distance + keep the nearer finger",
    # testbench
    "小目标点击": "Small-target click",
    "移动到圆心，食拇短捏并松开；按下与松开都在圆内才命中。": (
        "Move to the centre and pinch index to thumb briefly; it counts only if press and release are inside the circle."
    ),
    "连续双击": "Double click",
    "在同一圆内连续短捏两次，中间明确松开；两次松开的间隔不超过 0.50 秒。": (
        "Pinch twice in the same circle with a clear release between; the two releases at most 0.50 s apart."
    ),
    "拖拽放下": "Drag and drop",
    "在方块内捏住食拇，等拖拽就绪，再移入目标圆并松开。": (
        "Pinch index to thumb inside the square, wait until the drag is ready, move into the circle and release."
    ),
    "反向微调": "Fine correction",
    "依次点击圆点；远移后会在反方向出现相邻小目标，观察能否及时停住再微调。": (
        "Click the dots in turn; after a long move a small target appears in the opposite direction, to see "
        "whether you can stop in time and correct."
    ),
    "右键点击": "Right click",
    "移动到圆心，食指移开，拇指和无名指轻捏并松开；按下与松开都在圆内才命中，左键不算。": (
        "Move to the centre, index away, pinch thumb and ring finger lightly; press and release must be inside "
        "the circle, and left clicks do not count."
    ),
    "滚动后停住": "Scroll and stop",
    "保持 V 手势，将滚动条移进绿色区，再停住手至少 0.70 秒；保持 V 等本轮通过。": (
        "Hold the V gesture, move the bar into the green zone and keep the hand still for at least 0.70 s; keep "
        "the V until the round passes."
    ),
    "选择任务，点「开始本项」": 'Choose a task and click "Start task"',
    "测试只操作这里的虚拟光标": "The test moves only the virtual pointer here",
    "从这里捏住拖动": "Pinch here and drag",
    "第一下已收到，再短捏一次": "First tap received, pinch once more",
    "绿色区域内停住": "stop inside the green zone",
    "PinchPilot · 虚拟测试台": "PinchPilot · testbench",
    "固定任务测试台 · 系统鼠标已暂停": "Fixed-task testbench · system mouse paused",
    "等待主窗口输入 · 仅测试虚拟任务；不代表系统点击成功率或识别准确率。": (
        "Waiting for input from the main window · virtual tasks only; not system click success or recognition accuracy."
    ),
    "开始本项": "Start task",
    "取消并保存": "Cancel and save",
    "尚未开始": "Not started",
    "结果只保存任务摘要，不保存图片、关键点或光标轨迹。关闭测试台后需手动启用系统鼠标。": (
        "Only task summaries are saved, no images, keypoints or pointer paths. Enable the system mouse by hand "
        "after closing the testbench."
    ),
    "先松开点击手指，再开始动作 · 只操控虚拟光标；取消会保留已完成部分。": (
        "Release the clicking fingers before you start · virtual pointer only; cancelling keeps what is done."
    ),
    "虚拟输入已就绪 · 取消会保留已完成部分。": "Virtual input ready · cancelling keeps what is done.",
    "相机识别输入": "Camera input",
    "仅工程验证": "engineering check only",
    "等待主窗口输入": "Waiting for input from the main window",
    "输入已中断": "input interrupted",
    "仅虚拟任务表现，不代表真实识别准确率。": "virtual tasks only, not real recognition accuracy.",
    "进行中": "In progress",
    "已取消": "Cancelled",
    "输入已过期，请恢复相机后重新开始": "input went stale; restart after the camera recovers",
    "输入来源已变化，请重新开始": "input source changed; start again",
    "到 2 分钟上限，这一项按未完成记录": "2-minute limit reached; this task counts as unfinished",
    "请先松开点击手指": "release the clicking fingers first",
    # keypoint recording
    "PinchPilot · ML 关键点采集": "PinchPilot · ML keypoint recording",
    "ML 动作采集 · 系统鼠标已暂停": "ML gesture recording · system mouse paused",
    "只保存当前控制手的连续 21 点坐标、时间与左右手分类；无手帧也保留。不保存图像，不会自动启动相机或训练模型。": (
        "Saves only the control hand's 21 keypoints per frame, timestamps and handedness; frames without a hand "
        "are kept. No images are saved, and it never starts the camera or trains a model by itself."
    ),
    "开始这一段（空格）": "Start this segment (Space)",
    "重录上一段（退格）": "Redo the last segment (Backspace)",
    "正在录这一段…": "Recording this segment…",
    "使用匿名编号，例如 P01": "Use an anonymous ID, e.g. P01",
    "用户编号": "Participant ID",
    "采集场次": "Session",
    "同意保存 21 点坐标，不记录图像；之后需人工标注": "I agree to saving 21 keypoint coordinates, no images; labels are reviewed by hand later",
    "开始采集": "Start recording",
    "停止并保存": "Stop and save",
    "等待新鲜相机帧，尚未采集": "Waiting for fresh camera frames, not recording",
    "相机帧已就绪 · 勾选同意后点击开始": "Camera frames ready · tick the consent box, then start",
    "输出目录：data/contact_recordings": "Output folder: data/contact_recordings",
    "停止后会生成标签文件：按提示录制时是来自提示和空格键的草稿，否则是空白模板。两种都需要人工复核后才能训练，动作提示和程序的判断不能直接作为正确答案。只保存点击开始之后的画面关键点。关闭后需手动重新启用系统鼠标。": (
        "Stopping writes a label file: a draft from the prompts and the Space key for guided recordings, otherwise "
        "an empty template. Both need a human review before training; prompts and the program's own judgement are "
        "not ground truth. Only keypoints after you start are saved. Enable the system mouse by hand afterwards."
    ),
    "尚未开始：请明确同意保存关键点，并等待新鲜相机帧。": "Not started: agree to saving keypoints and wait for fresh camera frames.",
    "尚未开始：请填写用户编号和采集场次。": "Not started: fill in the participant ID and the session.",
    "相机画面还没准备好，稍等一下再开始。": "The camera picture is not ready yet; wait a moment.",
    "这一次录完了": "This recording is complete",
    "提前停止了，这一次不完整": "Stopped early; this recording is incomplete",
    "点「开始采集」后，这里会一段一段提示动作": 'After "Start recording", prompts appear here one segment at a time',
    "每段先读提示，读完按空格开始，到时间自动停。提示要求时，手指碰到就按住空格，离开就松开。": (
        "Read each prompt, then press Space to start; the segment stops on its own. When asked, hold Space while "
        "the fingertips touch and release it when they part."
    ),
    "看不到手": "no hand visible",
    "✗ 还不对，请照上面的朝向调整": "✗ not yet, turn the hand as described above",
    "手在画面内": "hand in view",
    "没有检测到手": "no hand detected",
    "空格：按住中": "Space: held",
    "空格：未按": "Space: up",
    "相机画面中断，恢复后才能开始这一段。": "The camera picture is interrupted; this segment starts after it recovers.",
    "手的朝向和这一轮要求的不一样，所以还没开始。调整好再按空格；确实摆不到的话，再按一次空格也会开始。": (
        "The hand is not turned as this round asks, so the segment has not started. Adjust and press Space; "
        "if you really cannot, pressing Space again starts it anyway."
    ),
    "上一段已作废，现在重录这一段。": "The last segment was discarded; record it again now.",
    "上一段录完了。没做好可以按退格键重录。": "The last segment is done. Press Backspace to redo it if needed.",
    "上一段没有收到空格。请按退格键重录；如果按了却没收到，先点一下这个窗口。": (
        "No Space press was received in the last segment. Press Backspace to redo it; if you did press it, click "
        "this window first."
    ),
    "相机画面中断，等待恢复；录制中超过 5 秒会停止并保存": "Camera picture interrupted, waiting; over 5 s while recording stops and saves",
    "两段之间不录制 · 读完提示再开始": "Nothing is recorded between segments · read the prompt, then start",
    "正在采集新的相机帧 · 不保存图像、不自动生成真值": "Recording new camera frames · no images, no automatic ground truth",
    "已保留采集文件。": "The recording files are kept.",
    "没有采到新帧，未生成标注模板。": "No new frames were recorded; no label template was written.",
    "手动停止": "stopped by hand",
    "采集窗口关闭": "recording window closed",
    "输入来源改变": "input source changed",
    "相机输入过期或中断": "camera input stale or interrupted",
    "连续输入间隔过长": "gap between frames too long",
    "收到重复或倒序帧": "repeated or out-of-order frames",
    "帧时间无效": "invalid frame time",
    "录制写入失败": "writing the recording failed",
    "已取消关键点保存同意": "consent to saving keypoints withdrawn",
    "已按提示录完": "all prompts recorded",
    "侧对补录": "Side-on session",
    # recording prompts (contact_protocol)
    "手掌正对镜头": "palm facing the camera",
    "手向内转一些，大约 45 度": "hand turned inwards, about 45 degrees",
    "手侧过来，拇指一侧朝向镜头": "hand side-on, thumb side towards the camera",
    "手离镜头近一些或远一些": "hand nearer to or farther from the camera",
    "换成你觉得最放松的姿势": "whatever pose feels most relaxed",
    "每次碰到时按住空格，离开时松开": "Hold Space at each touch, release when the fingers part",
    "按住期间一直按着空格": "Keep Space held while the fingers stay together",
    "慢慢做，两三秒一次": "slowly, one every two or three seconds",
    "这一轮的手部朝向": "Hand orientation for this round",
    "手放松张开，手指互不接触": "Hand relaxed and open, fingers not touching",
    "捏住：拇指和中指": "Pinch: thumb and middle finger",
    "松开": "Release",
    "捏住": "Pinch",
    "捏住拇指和中指，小幅移动手或手腕": "Pinch thumb and middle finger, move the hand or wrist a little",
    "保持捏住。食指点拇指，慢慢做，两三秒一次，大约 8 次": (
        "Keep pinching. Tap the index on the thumb, slowly, one every two or three seconds, about 8 times"
    ),
    "保持捏住。食指按住拇指约 2 秒再松开，大约 4 次": "Keep pinching. Hold the index on the thumb for about 2 s, then release, about 4 times",
    "松开中指。只用食指点拇指，慢慢做，两三秒一次，大约 6 次": (
        "Release the middle finger. Tap only the index on the thumb, slowly, one every two or three seconds, about 6 times"
    ),
    "松开中指。拇指碰无名指，慢慢做，两三秒一次，大约 5 次": (
        "Release the middle finger. Touch the thumb to the ring finger, slowly, one every two or three seconds, about 5 times"
    ),
    "拇指按住无名指约 2 秒再松开，大约 3 次": "Hold the thumb on the ring finger for about 2 s, then release, about 3 times",
    "捏住拇指和中指。无名指也碰拇指，慢慢做，两三秒一次，大约 4 次": (
        "Pinch thumb and middle finger. Touch the ring finger to the thumb too, slowly, one every two or three "
        "seconds, about 4 times"
    ),
    "不要按空格": "Do not press Space",
    "食指靠近拇指但不要碰到，来回约 5 次": "Bring the index close to the thumb without touching, about 5 times",
    "捏住拇指和中指。食指靠近拇指但不要碰到，约 5 次": (
        "Pinch thumb and middle finger. Bring the index close to the thumb without touching, about 5 times"
    ),
    "手自然动一动：转转手腕、换个位置，手指互不接触": "Move the hand naturally: turn the wrist, change position, fingers not touching",
    # demonstration hand in the recording window
    "拇指": "thumb",
    "食指": "index",
    "中指": "middle",
    "无名指": "ring",
    "手指互不接触": "Fingers not touching",
    "这一段不按空格": "No Space in this segment",
    "按住空格": "Hold Space",
    "空格：松开": "Space: release",
    "跟着示意图的节奏做：指尖碰到时下面的空格键变绿，这时按住空格，离开就松开。": (
        "Follow the demo hand's rhythm: when the fingertips touch, the Space key below turns green; hold Space "
        "then and release it when they part."
    ),
    "示意图：手指靠近拇指但留一条缝，全程不按空格。": "Demo: the finger comes close to the thumb but leaves a gap; no Space at all.",
    "示意图会和屏幕上的提示一起切换，跟着做。": "The demo changes together with the prompts on screen; follow along.",
    "示意图：这一段手的样子。": "Demo: the hand for this segment.",
    # label review
    "PinchPilot · 标签检查": "PinchPilot · label review",
    "录制": "Recording",
    "请选择一段录制": "Choose a recording",
    "← 上一个": "← Previous",
    "下一个 →": "Next →",
    "通过 (Enter)": "Pass (Enter)",
    "这一段作废 (X)": "Void this segment (X)",
    "同意 (Enter)": "Agree (Enter)",
    "不同意 (X)": "Disagree (X)",
    "暂停 (空格)": "Pause (Space)",
    "播放 (空格)": "Play (Space)",
    "不同意时可以写一句原因": "If you disagree, add a short reason",
    "完成并保存": "Finish and save",
    "已检查": "reviewed",
    "检查中": "in review",
    "未检查": "not reviewed",
    "进度文件损坏": "progress file damaged",
    "正在读取录制…": "Reading the recording…",
    "空格标记提前": "Space marks earlier",
    "空格标记推后": "Space marks later",
    "按下松开前后留空": "Margin around presses",
    "提示出现后留空": "Margin after a prompt",
    "提示结束前留空": "Margin before a prompt ends",
    "空格": "Space",
    "曲线：拇指尖到该指尖的距离，越低越近。色带：绿=接触，灰=未接触，斜纹=不给标签。": (
        "Curve: thumb tip to that fingertip, lower is closer. Band: green = touch, grey = no touch, stripes = no label."
    ),
    "这一帧没有检测到手": "No hand in this frame",
    "这段录制现在不能检查": "This recording cannot be reviewed now",
    "还没看": "not seen yet",
    "已通过": "passed",
    "已作废": "voided",
    "同意": "agreed",
    "不同意": "disagreed",
    "标签来自屏幕提示": "labels from the on-screen prompts",
    "全部看完了，点「完成并保存」。": 'All seen; click "Finish and save".',
    "进度会自动保存，可以随时关掉再继续。": "Progress is saved automatically; you can close and continue later.",
    "每个结论都会自动保存。": "Every decision is saved automatically.",
    "需要用 --annotator 填写你自己的编号，例如 --annotator P02": "Give your own ID with --annotator, e.g. --annotator P02",
    "标签和上次保存的一样，文件没有改动，不用再上传，已有的抽查仍然有效。": (
        "The labels match the last save; nothing changed, no upload needed, and existing cross-checks stay valid."
    ),
    "上次检查时把留空调得比默认值短，动作切换的瞬间也被打上了标签。留空已恢复默认，所有段都需要重新看一遍；之前因为对不上而作废的段，现在多半对得上了。": (
        "Last time the margins were set shorter than the default, so the moments between actions got labels too. "
        "The margins are back to default and every segment needs another look; segments voided for not matching "
        "will mostly match now."
    ),
    "复核者": "reviewer",
    "抽查者": "cross-checker",
    # diagnostics window
    "PinchPilot · 诊断与耗时": "PinchPilot · diagnostics and timings",
    "阶段": "Stage",
    "最近 / ms": "Latest / ms",
    "样本数": "Samples",
    "相机": "Camera",
    # wrist guide
    "腕动校准动作示意": "Wrist calibration motion",
    "绘制的动作示意，不是摄像头画面或当前姿态；转到后保持不动即可。": (
        "A drawn illustration, not the camera picture or your pose; turn and hold still."
    ),
    "先回到自然中立姿势，等待提示后再抬掌。": "Return to a natural neutral pose; raise the palm when prompted.",
    "正在采集，请保持不动，等本步通过。": "Recording; hold still until this step passes.",
    "中立示意：前臂有支撑，手指自然放松。": "Neutral: forearm supported, fingers relaxed.",
    "俯视示意：前臂保持固定，以手腕为中心轻轻向右转，转到后停住。": "Top view: forearm fixed, turn gently right at the wrist, then stop.",
    "侧面示意：先回中立，前臂保持固定，以手腕为中心轻轻向上抬掌，转到后停住。": (
        "Side view: back to neutral, forearm fixed, raise the palm gently at the wrist, then stop."
    ),
    "示范完成，保持自然姿态。": "Done; keep a natural pose.",
    "此图不是当前姿态，不必照抄手形或跟随循环。": "This is not your pose; no need to copy the hand shape or follow the loop.",
    "俯视示意": "Top view",
    "向右": "right",
    "前臂放稳": "forearm steady",
    "侧面示意 · 先回正": "Side view · back to neutral first",
    "向上": "up",
    "前臂不抬起": "forearm stays down",
    "手腕": "wrist",
    "动作示意 · 不必照抄手形": "Illustration · no need to copy the hand shape",
    "先回正 · 等待提示后再轻抬手掌": "Back to neutral · raise the palm gently when prompted",
    "保持不动 · 等本步通过": "Hold still · until this step passes",
    "手指自然放松 · 不是当前姿态画面": "Fingers relaxed · not your current pose",
    "转到后停住 · 等本步通过": "Turn and stop · until this step passes",
    "有效静止样本不足；保持拇中捏合，食指和无名指移开后重试": (
        "Not enough steady samples; keep thumb and middle pinched, index and ring away, and try again"
    ),
    "相机采样不连续，请保持控制手可见后重试": "Camera samples were not continuous; keep the control hand visible and try again",
    "校准期间手移动过多，请支撑前臂、保持捏合点静止后重试": "The hand moved too much; rest the forearm, keep the pinch still and try again",
    "校准期间有持续漂移，请保持捏合点静止后重试；保留原有校准": (
        "The hand kept drifting; keep the pinch still and try again. The previous calibration is kept"
    ),
    "静止波动太大，请先调整支撑、光照或相机位置；保留原有校准": (
        "Resting jitter is too large; improve support, lighting or camera placement first. The previous calibration is kept"
    ),
}

# Texts with numbers, names or paths. Each group is translated again, so a Chinese part inside comes out in English.
PATTERNS = [
    (r"PinchPilot (\S+) · 桌面测试版", "PinchPilot {0} · desktop"),
    (r"桌面测试版  (\S+)  ·  主屏幕", "Desktop  {0}  ·  main screen"),
    (r"设置读取失败，使用默认设置：(.*)", "Could not read the settings, using defaults: {0}"),
    (r"设置未保存：(.*)", "Settings not saved: {0}"),
    (
        r"接触模型无法加载，已改回现有规则：(.*)",
        "Could not load the contact model; back to the original rule: {0}",
    ),
    (r"校准未启动：(.*)", "Calibration not started: {0}"),
    (
        r"已校准 · 采集时波动 (\S+) 逻辑像素 · (\d+) 帧",
        "Calibrated · jitter {0} logical px · {1} frames",
    ),
    (r"腕动校准未启动：(.*)", "Wrist calibration not started: {0}"),
    (r"运行已停止：(.*)", "Stopped: {0}"),
    (r"问题标记未保存：(.*)", "Problem mark not saved: {0}"),
    (r"无法打开测试台：(.*)", "Cannot open the testbench: {0}"),
    (r"无法打开 ML 采集窗口：(.*)", "Cannot open the ML recording window: {0}"),
    (r"读不出来、已跳过：(.*)", "Unreadable, skipped: {0}"),
    (
        r"(.*)校准已取消，保留原设置；鼠标控制保持关闭。",
        "{0} calibration cancelled; settings kept, mouse control stays off.",
    ),
    (r"静止校准未应用：(.*)", "Stillness calibration not applied: {0}"),
    (
        r"静止校准已应用（(\d+) 帧）；需要时重新启用鼠标控制。",
        "Stillness calibration applied ({0} frames); enable mouse control again when needed.",
    ),
    (
        r"已标记刚才的问题（最多 10 秒诊断摘要）：(.*)",
        "Marked the last problem (diagnostics of at most 10 s): {0}",
    ),
    (r"松键失败，正在重试：(.*)", "Button release failed, retrying: {0}"),
    (
        r"模型文件校验失败，请移除后重下：(.*)",
        "The model file failed its check; remove it and download again: {0}",
    ),
    # Only the edge names: the advice after " · " is translated on its own.
    (r"手靠近画面([^ ·]+)", "Hand near the {0} edge"),
    (r"丢失 (\d+)", "lost {0}"),
    (r"左 (\d+)", "left {0}"),
    (r"右 (\d+)", "right {0}"),
    (r"拖 (\d+)", "drag {0}"),
    (r"滚 (\d+)", "scroll {0}"),
    (r"第 (\d+) / 8 个目标", "Target {0} / 8"),
    (r"失误 (\d+)", "misses {0}"),
    (r"命中 (\d+)/8", "hits {0}/8"),
    (r"训练时不含 (.+)", "trained without {0}"),
    (r"用 (.+) 训练", "trained on {0}"),
    (r"没有这个内置模型：(.*)", "No such built-in model: {0}"),
    (r"目标 (\S+) ± (\S+)", "target {0} ± {1}"),
    (r"已保存：(.*) · 不含图像或轨迹", "Saved: {0} · no images or paths"),
    (
        r"结果尚未保存：(.*)。关闭窗口时会再尝试保存。",
        "Results not saved yet: {0}. Saving is tried again when the window closes.",
    ),
    (r"完成 (\d+)/(\d+)", "done {0}/{1}"),
    (r"未命中 (\d+)", "misses {0}"),
    (r"输入中断 (\d+)", "input interruptions {0}"),
    (r"([\d.]+) 秒", "{0} s"),
    (
        r"按屏幕提示分段录制：共 (\d+) 段、约 (\d+) 分钟，每段读完提示再开始",
        "Guided recording: {0} segments, about {1} min; read each prompt before starting",
    ),
    (
        r"只补录侧对：共 (\d+) 段、约 (\d+) 分钟，给已经录过完整一次的人",
        "Side-on only: {0} segments, about {1} min, for people who already recorded a full session",
    ),
    (r"无法开始采集：(.*)", "Cannot start recording: {0}"),
    (
        r"有 (\d+) 个步骤没有收到空格，这些步骤的标签留空；多的话请重录。",
        "{0} steps received no Space press and stay unlabelled; record again if there are many.",
    ),
    (r"(.*)把这个文件上传到数据仓库：(.*)", "{0}Upload this file to the data repository: {1}"),
    (
        r"请重新录一次完整的。已录的部分保存在：(.*)",
        "Please record a complete session again. What was recorded is saved in: {0}",
    ),
    (r"第 (\d+)/(\d+) 段", "Segment {0}/{1}"),
    (r"第 (\d+)/(\d+) 轮", "Round {0}/{1}"),
    (r"第 (\d+) 轮", "round {0}"),
    (r"现在约 (\d+)°", "now about {0}°"),
    (r"手掌和镜头的夹角：(.+)（这一轮不限）", "Palm-to-camera angle: {0} (any angle this round)"),
    (
        r"手掌和镜头的夹角：(.+?)，这一轮需要 (.+?)(| ✓| ✗ 还不对，请照上面的朝向调整)",
        "Palm-to-camera angle: {0}; this round needs {1}{2}",
    ),
    (r"(\d+)° 以上", "{0}° or more"),
    (r"(\d+)° 以内", "within {0}°"),
    (r"(\d+)° 到 (\d+)°", "{0}° to {1}°"),
    (r"正在录，还剩 (\d+) 秒", "recording, {0} s left"),
    (r"(\d+) 秒后：(.+)", "In {0} s: {1}"),
    (r"这一段 (\d+) 秒", "this segment {0} s"),
    (r"读完后按空格开始", "press Space when ready"),
    (
        r"上一段手掌和镜头的夹角约 (\d+)°，不符合这一轮的要求。请调整朝向后按退格键重录。",
        "In the last segment the palm-to-camera angle was about {0}°, not what this round needs. Turn the hand and "
        "press Backspace to redo it.",
    ),
    (
        r"上一段收到 (\d+) 次空格。次数不对或没做好，可以按退格键重录。",
        "The last segment received {0} Space presses. If that is wrong or it went badly, press Backspace to redo it.",
    ),
    (r"已写入 (\d+) 帧", "{0} frames written"),
    (r"其中无手 (\d+) 帧", "{0} without a hand"),
    (r"已停止：([^。]+)。", "Stopped: {0}."),
    (r"保存异常：(.*)", "Saving failed: {0}"),
    (r"文件夹：(.*)", "Folder: {0}"),
    (r"上传这一个文件：(.*)", "Upload this one file: {0}"),
    (r"关键点：(.*)", "Keypoints: {0}"),
    (r"待标注模板：(.*)", "Label template: {0}"),
    (
        r"(关闭录制文件失败|未能生成待标注模板|提示记录未保存|停止摘要未保存|打包失败)：(.*)",
        "{0}: {1}",
    ),
    (r"关闭录制文件失败", "Closing the recording failed"),
    (r"未能生成待标注模板", "Could not write the label template"),
    (r"提示记录未保存", "Prompt log not saved"),
    (r"停止摘要未保存", "Stop summary not saved"),
    (r"打包失败", "Packing failed"),
    (
        r"屏幕会自动轮流显示提示，跟着做，共 (\d+) 步：(.+)",
        "Prompts change on their own; follow them. {0} steps: {1}",
    ),
    (r"这一轮的手部朝向：(.+)", "Hand orientation for this round: {0}"),
    (
        r"侧对补录 第 (\d+)/(\d+) 遍，手的朝向：(.+)",
        "Side-on session, pass {0}/{1}, hand orientation: {2}",
    ),
    (r"(.+)，共 (\d+) 步", "{0}, {1} steps"),
    (r"(.+)的标签来自空格", "{0} labels from the Space key"),
    (r"(.+) 抽查不同意(?:：(.*))?", "{0} disagreed in the cross-check{1}"),
    (r"已看 (\d+)/(\d+)，其中作废 (\d+)。", "Seen {0}/{1}, voided {2}."),
    (r"已看 (\d+)/(\d+)，不同意 (\d+)。", "Seen {0}/{1}, disagreed {2}."),
    (r"检查自己的标签 · (.+)", "Reviewing your own labels · {0}"),
    (r"抽查 (.+) 的标签 · 抽查者 (.+)", "Cross-checking {0}'s labels · checker {1}"),
    (r"(.+) (-?\d+) 毫秒", "{0} {1} ms"),
    (
        r"参数变了，之前通过的 (\d+) 段需要重新看一遍。",
        "The settings changed; {0} segments passed before need another look.",
    ),
    (r"保存失败：(.*)", "Saving failed: {0}"),
    (
        r"已保存。把这两个文件上传到数据仓库里这段录制所在的文件夹：",
        "Saved. Upload these two files to this recording's folder in the data repository:",
    ),
    (r"抽查完成，结果已保存：(.*)", "Cross-check finished and saved: {0}"),
    (r"找不到文件或文件夹：(.*)", "File or folder not found: {0}"),
    (r"这个文件夹里没有按提示录制的压缩包：(.*)", "No guided recordings in this folder: {0}"),
    (
        r"(.+) 检查这段录制时把留空调得比默认值短，需要重新检查。等对方重新保存并上传后再抽查。",
        "{0} reviewed this recording with margins shorter than the default, so it needs another review. "
        "Cross-check it after they save and upload again.",
    ),
    (r"还有 (\d+) 个单元没有看完", "{0} units are not seen yet"),
]
PATTERNS = [(re.compile(p), t) for p, t in PATTERNS]
SPLITS = (("\n", "\n"), (" · ", " · "), (" / ", " / "), (" → ", " → "), ("；", "; "), ("、", ", "))


def resolve(lang=None):
    """The interface language: the --lang value, else PINCHPILOT_LANG, else Chinese."""
    value = (lang or os.environ.get("PINCHPILOT_LANG") or "zh").strip().lower()
    return "en" if value.startswith("en") else "zh"


def tr(text):
    """The display text in the interface language (unchanged in Chinese or when nothing Chinese is in it)."""
    if LANG != "en" or not isinstance(text, str) or not HAN.search(text):
        return text
    return _translate(text)


@lru_cache(maxsize=8192)
def _translate(text):
    core = text.strip()
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    return lead + _lookup(core) + trail


def _lookup(text):
    if text in TEXTS:
        return TEXTS[text]
    if (
        text.endswith("。") and text[:-1] in TEXTS
    ):  # the same sentence with or without its full stop
        return TEXTS[text[:-1]] + "."
    if text + "。" in TEXTS:
        return TEXTS[text + "。"].rstrip(".")
    for pattern, template in PATTERNS:
        found = pattern.fullmatch(text)
        if found:
            return template.format(*(tr(g or "") for g in found.groups()))
    for sep, joined in SPLITS:
        if sep in text:
            return joined.join(tr(part) for part in text.split(sep))
    sentences = [x for x in re.split(r"(?<=[。！？])", text) if x]
    if len(sentences) > 1:
        return " ".join(tr(x) for x in sentences)
    MISSING.add(text)
    return text


def _wrap(cls, name, positions=None):
    original = getattr(cls, name)

    def translated(self, *args, **kwargs):
        args = tuple(
            tr(a) if positions is None or i in positions else a for i, a in enumerate(args)
        )
        return original(self, *args, **kwargs)

    translated.__wrapped__ = original
    setattr(cls, name, translated)


def _wrap_first(cls, name):
    """Translate only the first text argument (combo items carry user data after it)."""
    original = getattr(cls, name)

    def translated(self, *args, **kwargs):
        args, done = list(args), False
        for i, a in enumerate(args):
            if isinstance(a, str) and not done:
                args[i], done = tr(a), True
        return original(self, *args, **kwargs)

    translated.__wrapped__ = original
    setattr(cls, name, translated)


def _wrap_list(cls, name):
    original = getattr(cls, name)

    def translated(self, items, *args):
        return original(self, [tr(item) for item in items], *args)

    translated.__wrapped__ = original
    setattr(cls, name, translated)


def install(lang=None):
    """Choose the interface language; in English, translate everything Qt displays from now on.

    Call it before the windows are built. Chinese leaves Qt untouched.
    """
    global LANG, _installed
    LANG = resolve(lang)
    if LANG != "en" or _installed:
        return LANG
    from PySide6 import QtGui
    from PySide6 import QtWidgets as W

    for cls in (
        W.QLabel,
        W.QPushButton,
        W.QCheckBox,
        W.QRadioButton,
        W.QGroupBox,
        W.QTableWidgetItem,
    ):
        _wrap(cls, "__init__")
    for cls, names in (
        (W.QLabel, ("setText",)),
        (W.QAbstractButton, ("setText",)),
        (W.QGroupBox, ("setTitle",)),
        (W.QComboBox, ("setPlaceholderText",)),
        (W.QLineEdit, ("setPlaceholderText",)),
        (W.QPlainTextEdit, ("setPlaceholderText",)),
        (W.QWidget, ("setToolTip", "setWindowTitle", "setStatusTip")),
        (W.QSpinBox, ("setSuffix", "setPrefix")),
        (W.QDoubleSpinBox, ("setSuffix", "setPrefix")),
        (W.QProgressBar, ("setFormat",)),
        (W.QTableWidgetItem, ("setText",)),
        (QtGui.QPainter, ("drawText",)),
    ):
        for name in names:
            _wrap(cls, name)
    for name in ("addItem", "insertItem", "setItemText"):
        _wrap_first(W.QComboBox, name)
    _wrap_first(W.QFormLayout, "addRow")
    _wrap_first(W.QFormLayout, "insertRow")
    _wrap_list(W.QComboBox, "addItems")
    _wrap_list(W.QTableWidget, "setHorizontalHeaderLabels")
    _installed = True
    return LANG
