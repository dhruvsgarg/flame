# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N98: `trainerAmp: bf16` autocasts the trainer's forward on CUDA only; the default stays fp32."""

import contextlib
import types

import torch

from flame.config import Hyperparameters
from main import PyTorchCifar10Trainer


def _trainer(amp, device):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.device = torch.device(device)
    t.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(trainer_amp=amp))
    return t


def test_bf16_on_cuda_only():
    ctx = _trainer("bf16", "cuda")._amp()
    assert isinstance(ctx, torch.autocast) and ctx.fast_dtype == torch.bfloat16
    assert isinstance(_trainer("bf16", "cpu")._amp(), contextlib.nullcontext)
    assert isinstance(_trainer("", "cuda")._amp(), contextlib.nullcontext)


def test_config_alias_and_default():
    assert Hyperparameters(rounds=1, epochs=1).trainer_amp == ""
    assert Hyperparameters(rounds=1, epochs=1, trainerAmp="bf16").trainer_amp == "bf16"
