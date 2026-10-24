# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N10: one dataset switch for the launcher's trainer and aggregators (cifar10, google_speech).

`hyperparameters.dataset_name` picks the spec; unset = cifar10 (byte-identical). Each spec gives the
model, the train/test sets, the stub input shape and the trainer's local optimizer.
google_speech reads SpeechCommands v0.02 like torchaudio's SPEECHCOMMANDS (same sorted walker and
validation/testing lists, so the 2024 split indices line up) without the torchaudio dependency.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

EX_ROOT = Path(__file__).resolve().parent
DATASETS_YAML = EX_ROOT.parent / "_metadata" / "datasets.yaml"
# In-repo fallbacks (where the data lived before data_roots); used only when no data root has the dataset.
LEGACY_DIRS = {"cifar10": EX_ROOT / "data", "google_speech": EX_ROOT.parent / "async_google_speech" / "data" / "data"}


def dataset_dir(name: str) -> Path:
    """<root>/<name> for the first root that has it: $FLAME_DATA_ROOT, then datasets.yaml data_roots
    (per-node scratch mounts), else the in-repo legacy dir."""
    roots = [os.environ["FLAME_DATA_ROOT"]] if os.environ.get("FLAME_DATA_ROOT") else []
    try:
        import yaml
        roots += yaml.safe_load(open(DATASETS_YAML)).get("data_roots") or []
    except (OSError, ValueError):
        pass
    for r in roots:
        if (Path(r) / name).is_dir():
            return Path(r) / name
    return LEGACY_DIRS[name]


def speech_root() -> Path:
    return dataset_dir("google_speech") / "SpeechCommands" / "speech_commands_v0.02"
SPEECH_TRAIN_SIZE = 84843  # v0.02 training subset; the 2024 split indices assume exactly this list
SPEECH_TEST_SIZE = 11005
SPEECH_LEN = 16000  # 1 s at 16 kHz; shorter clips are zero-padded
SPEECH_STUB_LEN = 160  # stub input: same model; ~0.27s per 1-core step (cifar 0.16s)
SPEECH_LABELS = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go",
    "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
)


class CifarNet(nn.Module):
    """async_cifar10's CNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet34_1D(nn.Module):
    """The 2024 google_speech model (async_google_speech/trainer/pytorch/main_resnet.py)."""

    def __init__(self, n_input=1, n_output=35):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(n_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_output)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [BasicBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers += [BasicBlock1D(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return F.log_softmax(self.fc(x), dim=1)


class SpeechCommands(data_utils.Dataset):
    """SpeechCommands v0.02 subset -> (float waveform [1, SPEECH_LEN], label index)."""

    def __init__(self, subset: str, root: Path = None):
        root = Path(root) if root else speech_root()
        if not (root / "testing_list.txt").exists():
            raise FileNotFoundError(
                f"SpeechCommands v0.02 not found at {root}. Copy it from a node that has it (gitignored, 5.5 GB):"
                f" put it under <data_root>/google_speech/ (datasets.yaml data_roots or $FLAME_DATA_ROOT), or extract"
                " http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz into that directory.")

        def _list(*names):
            out = []
            for n in names:
                with open(root / n) as f:
                    out += [os.path.normpath(str(root / line.strip())) for line in f if line.strip()]
            return out

        if subset == "testing":
            self.files = _list("testing_list.txt")
        elif subset == "validation":
            self.files = _list("validation_list.txt")
        elif subset == "training":
            excl = set(_list("validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in root.glob("*/*.wav"))
            self.files = [w for w in walker if "_nohash_" in w and "_background_noise_" not in w
                          and os.path.normpath(w) not in excl]
        else:
            raise ValueError(subset)
        want = {"training": SPEECH_TRAIN_SIZE, "testing": SPEECH_TEST_SIZE}.get(subset)
        if want is not None and len(self.files) != want:  # a partial copy would misalign every split index
            raise RuntimeError(f"SpeechCommands {subset} at {root} has {len(self.files)} clips, expected {want}"
                               " (incomplete copy?)")
        self.label_idx = {n: i for i, n in enumerate(SPEECH_LABELS)}

    def __len__(self):
        return len(self.files)

    def pcm16(self, i):
        """Raw clip as int16 [1, SPEECH_LEN] (zero-padded) and label; model_input() gives the float waveform."""
        from scipy.io import wavfile

        path = self.files[i]
        _, pcm = wavfile.read(path)
        if pcm.dtype != np.int16:
            raise ValueError(f"{path}: expected 16-bit PCM, got {pcm.dtype}")
        wav = torch.zeros(1, SPEECH_LEN, dtype=torch.int16)
        x = torch.from_numpy(pcm[:SPEECH_LEN].copy())
        wav[0, : x.numel()] = x
        return wav, self.label_idx[Path(path).parent.name]

    def __getitem__(self, i):
        wav, label = self.pcm16(i)
        return model_input(wav), label


CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def augment_batch(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """FX-D233: _cifar_train's RandomCrop(32, padding=4) + flip on a normalized NCHW batch, fresh draws per call."""
    b, c, h, w = x.shape
    fill = (-torch.tensor(CIFAR_MEAN, device=x.device) / torch.tensor(CIFAR_STD, device=x.device)).view(1, c, 1, 1)
    xp = fill.expand(b, c, h + 2 * pad, w + 2 * pad).clone()
    xp[:, :, pad:pad + h, pad:pad + w] = x
    i = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (i[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    out = xp[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(b, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def _cifar_train():
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10

    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), cifar_plain_tf()])
    return CIFAR10(str(dataset_dir("cifar10")), train=True, download=True, transform=tf)


def cifar_plain_tf():
    import torchvision.transforms as T

    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def _cifar_test():
    from torchvision.datasets import CIFAR10

    return CIFAR10(str(dataset_dir("cifar10")), train=False, download=True, transform=cifar_plain_tf())


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    shape: Tuple[int, ...]       # one sample
    stub_shape: Tuple[int, ...]  # harness stub sample (synthetic)
    model: Callable[[], nn.Module]
    train: Callable[[], data_utils.Dataset]
    test: Callable[[], data_utils.Dataset]
    optimizer: str               # trainer-local optimizer: sgd | adam


SPECS = {
    "cifar10": DatasetSpec("cifar10", 10, (3, 32, 32), (3, 32, 32), CifarNet, _cifar_train, _cifar_test, "sgd"),
    "google_speech": DatasetSpec("google_speech", 35, (1, SPEECH_LEN), (1, SPEECH_STUB_LEN), ResNet34_1D,
                                 lambda: SpeechCommands("training"), lambda: SpeechCommands("testing"), "adam"),
}


def spec_for(hp) -> DatasetSpec:
    """Spec named by `hyperparameters.dataset_name` (cifar10 when unset)."""
    name = getattr(hp, "dataset_name", None)
    if name is None and isinstance(getattr(hp, "__dict__", None), dict):
        name = hp.__dict__.get("dataset_name")
    name = str(name or "cifar10").lower().replace("-", "_")
    if name not in SPECS:
        raise ValueError(f"dataset_name={name!r} not in {sorted(SPECS)}")
    return SPECS[name]


def model_input(x: torch.Tensor) -> torch.Tensor:
    """int16 PCM -> torchaudio's normalized float (exact: int16 fits float32, /2^15 is exact); other inputs pass through."""
    return x.float() / 32768.0 if x.dtype == torch.int16 else x


def in_memory(ds: data_utils.Dataset) -> data_utils.Dataset:
    """Decode a deterministic (test) dataset into tensors once (FX-D64); `FLAME_EVAL_PRELOAD=0` keeps per-sample decode.
    Speech stays int16 (FX-N84: 336 vs 672 MiB); consumers apply model_input()."""
    if os.environ.get("FLAME_EVAL_PRELOAD", "1") == "0":
        return ds
    base, idx = (ds.dataset, ds.indices) if isinstance(ds, data_utils.Subset) else (ds, range(len(ds)))
    get = base.pcm16 if isinstance(base, SpeechCommands) else base.__getitem__
    xs, ys = zip(*(get(i) for i in idx))
    return data_utils.TensorDataset(torch.stack(xs), torch.as_tensor(ys))


def verify(name: str) -> str:
    """Gate check: the dataset resolves and is complete ("<name>: <dir> train=N test=M")."""
    spec = SPECS[name]
    tr, te = spec.train(), spec.test()
    return f"{name}: {dataset_dir(name)} train={len(tr)} test={len(te)}"
