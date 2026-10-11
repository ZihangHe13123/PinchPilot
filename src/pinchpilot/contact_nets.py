"""Small networks over contact windows and one training loop. PyTorch is imported lazily."""

import math

import numpy as np

from .contact_models import _deterministic_cpu, _last_output, _make_cnn, _torch

COORDINATES = 63
# What is left of a frame's 67 features without the coordinates: three distance ratios and
# the frame interval.
DISTANCE_FEATURES = 4
THUMB = (1, 2, 3, 4)
# MediaPipe joints of the finger behind each output, in the order of CHANNELS.
FINGERS = ((9, 10, 11, 12), (5, 6, 7, 8), (13, 14, 15, 16))
FINGER_FEATURES = 29


def finger_inputs(x):
    """[N, T, 67] windows -> [N, 3, T, 29]: for each finger, only the thumb and that finger.

    Per frame: the thumb's four joints (12), the finger's four joints (12), fingertip minus
    thumb tip (3), that finger's distance ratio (1) and the frame interval (1).
    """
    count, frames, _ = x.shape
    points = x[:, :, :63].reshape(count, frames, 21, 3)
    thumb = points[:, :, THUMB, :].reshape(count, frames, 12)
    views = []
    for channel, joints in enumerate(FINGERS):
        finger = points[:, :, joints, :]
        views.append(
            np.concatenate(
                (
                    thumb,
                    finger.reshape(count, frames, 12),
                    finger[:, :, 3] - points[:, :, 4],
                    x[:, :, 63 + channel : 64 + channel],
                    x[:, :, 66:67],
                ),
                axis=2,
            )
        )
    return np.stack(views, axis=1).astype(np.float32)


def temporal_cnn(frames, channels, seed, features=COORDINATES + DISTANCE_FEATURES):
    """The causal CNN of contact_models with a chosen width; input [N, T, features]."""
    return _make_cnn(frames, seed, channels, features)


def augmenter(mean, scale, *, rotate=0.0, scale_jitter=0.0, noise=0.0, seed=0):
    """A function that varies normalized [N, T, W] windows the way hands and people vary.

    Each window gets one random turn of the whole hand about its palm centre, up to `rotate`
    degrees about each of the three axes; one size factor within ±`scale_jitter`, applied to
    the coordinates and the distance ratios alike; and noise of standard deviation `noise`
    on every coordinate. A turn changes no distance, so the ratios stay as recorded, and the
    frame interval is never changed. W is 67, or 4 for windows of ratios and interval only,
    which have no coordinates to turn. `mean` and `scale` are the normalization of the
    windows. Returns None when there is nothing to vary.
    """
    if not (rotate or scale_jitter or noise):
        return None
    torch = _torch()
    coordinates = len(mean) - DISTANCE_FEATURES
    if coordinates not in (0, COORDINATES):
        raise ValueError("窗口的特征数不对，无法做数据增强")
    if not coordinates and (rotate or noise):
        raise ValueError("只用距离比时没有坐标可以旋转或加噪声")
    centre = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
    spread = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    limit = math.radians(rotate)

    def turns(count):
        a, b, c = (torch.rand((3, count), generator=generator) * 2 - 1) * limit
        one, zero = torch.ones(count), torch.zeros(count)

        def matrix(*rows):
            return torch.stack(rows, dim=1).reshape(count, 3, 3)

        about_z = matrix(a.cos(), -a.sin(), zero, a.sin(), a.cos(), zero, zero, zero, one)
        about_x = matrix(one, zero, zero, zero, b.cos(), -b.sin(), zero, b.sin(), b.cos())
        about_y = matrix(c.cos(), zero, c.sin(), zero, one, zero, -c.sin(), zero, c.cos())
        return about_z @ about_x @ about_y

    def vary(batch):
        raw = batch * spread + centre
        count, frames, _ = raw.shape
        points, ratios, interval = raw[..., :coordinates], raw[..., coordinates:-1], raw[..., -1:]
        if coordinates:
            points = points.reshape(count, frames, COORDINATES // 3, 3)
            if rotate:
                # The same turn for every frame of a window: the hand turns, not the motion.
                points = points @ turns(count).transpose(1, 2)[:, None]
            if noise:
                points = points + torch.randn(points.shape, generator=generator) * noise
        if scale_jitter:
            factor = 1 + (torch.rand((count, 1, 1), generator=generator) * 2 - 1) * scale_jitter
            ratios = ratios * factor
            if coordinates:
                points = points * factor[..., None]
        points = points.reshape(count, frames, coordinates)
        return (torch.cat((points, ratios, interval), dim=2) - centre) / spread

    return vary


def finger_cnn(frames, channels, seed, shared=True, context=True):
    """One small causal CNN per finger; input [N, 3, T, 29] from `finger_inputs`.

    `shared` runs the three fingers through the same weights. `context` lets each output
    also see the other two fingers' vectors (their element-wise maximum).
    """
    torch = _torch()
    kernel = max(1, frames - 2)

    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(FINGER_FEATURES, channels, 3)
            self.conv2 = torch.nn.Conv1d(channels, channels, kernel)

        def forward(self, x):
            return _last_output(x, self.conv1, self.conv2)

    class FingerContactCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            copies = 1 if shared else 3
            width = channels * 2 if context else channels
            self.branches = torch.nn.ModuleList(Branch() for _ in range(copies))
            self.heads = torch.nn.ModuleList(torch.nn.Linear(width, 1) for _ in range(copies))
            # With shared weights this is all that lets the fingers differ in how often they touch.
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, x):
            count = x.shape[0]
            if shared:
                vectors = self.branches[0](x.reshape(count * 3, *x.shape[2:])).reshape(count, 3, -1)
            else:
                vectors = torch.stack([self.branches[i](x[:, i]) for i in range(3)], dim=1)
            if context:
                a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
                others = torch.stack(
                    (torch.maximum(b, c), torch.maximum(a, c), torch.maximum(a, b)), dim=1
                )
                vectors = torch.cat((vectors, others), dim=2)
            logits = [self.heads[0 if shared else i](vectors[:, i]) for i in range(3)]
            return torch.cat(logits, dim=1) + self.bias

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = FingerContactCNN().cpu()
    return network


def parameter_count(network):
    return sum(value.numel() for value in network.parameters())


def predict(network, x, batch=2048):
    """Independent contact scores in [0, 1], shape [N, 3]."""
    torch = _torch()
    network.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            scores.append(
                torch.sigmoid(network(torch.from_numpy(x[start : start + batch]))).numpy()
            )
    return np.concatenate(scores, axis=0) if scores else np.empty((0, 3), dtype=np.float32)


def pretrain(network, x, *, epochs, mask, learning_rate, batch_size, seed, augment=None):
    """Teach the temporal CNN's two convolutions to rebuild windows they see only in part.

    A fraction `mask` of each window is blanked: whole landmarks (their three coordinates) in
    single frames, and single values among the distance and interval features. The
    convolutions' vector at the last frame goes through a throwaway linear layer that must
    give back the blanked values. No labels are used. Inputs are normalized, so a blanked
    value is the feature's mean. The classification head is left untouched. `augment`, from
    `augmenter`, varies every batch first; the varied window is then what must be rebuilt.
    """
    torch = _torch()
    count, frames, width = x.shape
    inputs = torch.from_numpy(x)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        decoder = torch.nn.Linear(network.conv2.out_channels, frames * width)
        generator = torch.Generator().manual_seed(seed)
    parameters = [*network.conv1.parameters(), *network.conv2.parameters(), *decoder.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = []
    with _deterministic_cpu(torch):
        for epoch in range(1, epochs + 1):
            order = rng.permutation(count)
            total = weight = 0.0
            for start in range(0, count, batch_size):
                batch = inputs[order[start : start + batch_size]]
                if augment is not None:
                    batch = augment(batch)
                hidden = torch.rand((len(batch), frames, 21), generator=generator) < mask
                rest = torch.rand((len(batch), frames, width - 63), generator=generator) < mask
                blank = torch.cat((hidden.repeat_interleave(3, dim=2), rest), dim=2)
                rebuilt = decoder(
                    _last_output(batch.masked_fill(blank, 0.0), network.conv1, network.conv2)
                )
                error = (rebuilt.reshape(batch.shape) - batch) ** 2
                loss = (error * blank).sum() / blank.sum().clamp(min=1)
                if not torch.isfinite(loss):
                    raise ValueError("预训练损失不是有限值")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                weight += len(batch)
            history.append({"epoch": epoch, "masked_error": total / weight})
    return {"epochs": epochs, "mask": mask, "windows": count, "history": history}


def train(
    network,
    x,
    y,
    *,
    epochs,
    learning_rate,
    batch_size,
    weight_decay,
    positive_weight,
    seed,
    validation=None,
    augment=None,
):
    """Fit in place on normalized inputs for exactly `epochs`.

    `validation` maps a name to (inputs, labels). Each one is scored after every epoch, and
    the weights of its best epoch are returned under its name, so that one training run can
    serve several validation sets. The network itself ends with the last epoch's weights.
    `augment`, from `augmenter`, varies every training batch; validation windows are scored
    as they are.
    """
    torch = _torch()
    inputs = torch.from_numpy(x)
    labels = torch.from_numpy(y.astype(np.float32))
    weight = None
    if positive_weight == "balanced":
        rate = np.clip(y.mean(axis=0), 1e-6, 1 - 1e-6)
        weight = np.clip((1 - rate) / rate, 0.05, 20.0)
    elif positive_weight != 1:
        weight = np.full(3, float(positive_weight))
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=None if weight is None else torch.tensor(weight, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    validation = validation or {}
    best = {name: {"loss": float("inf"), "epoch": None, "state": None} for name in validation}
    history = []
    with _deterministic_cpu(torch):
        for epoch in range(1, epochs + 1):
            network.train()
            order = rng.permutation(len(x))
            total = 0.0
            for start in range(0, len(x), batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                batch = inputs[indices]
                if augment is not None:
                    batch = augment(batch)
                loss = loss_fn(network(batch), labels[indices])
                if not torch.isfinite(loss):
                    raise ValueError("训练损失不是有限值")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            row = {"epoch": epoch, "train_loss": total / len(x), "validation_loss": {}}
            network.eval()
            for name, (held_x, held_y) in validation.items():
                held = 0.0
                with torch.inference_mode():
                    for start in range(0, len(held_x), 2048):
                        part = slice(start, start + 2048)
                        value = loss_fn(
                            network(torch.from_numpy(held_x[part])),
                            torch.from_numpy(held_y[part].astype(np.float32)),
                        )
                        held += float(value) * len(held_x[part])
                held /= len(held_x)
                if not math.isfinite(held):
                    raise ValueError("验证损失不是有限值")
                row["validation_loss"][name] = held
                if held < best[name]["loss"]:
                    best[name] = {
                        "loss": held,
                        "epoch": epoch,
                        "state": {
                            key: value.detach().clone()
                            for key, value in network.state_dict().items()
                        },
                    }
            history.append(row)
    return {
        "epochs": epochs,
        "history": history,
        "parameters": parameter_count(network),
        "selected_epoch": {name: item["epoch"] for name, item in best.items()},
    }, {name: item["state"] for name, item in best.items()}
