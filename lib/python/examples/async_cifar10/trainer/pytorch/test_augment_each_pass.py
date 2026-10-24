# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D233: per-batch crop / flip for GPU-preloaded cifar (trainerAugmentEachPass)."""

from types import SimpleNamespace

import torch

import fl_data
from trainer.pytorch.main import PyTorchCifar10Trainer


def test_crop_flip_is_a_shifted_window_of_the_padded_image():
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    out = fl_data.augment_batch(x)
    assert out.shape == x.shape
    fill = (-torch.tensor(fl_data.CIFAR_MEAN) / torch.tensor(fl_data.CIFAR_STD)).view(3, 1, 1)
    xp = fill.expand(3, 40, 40).clone()
    for b in range(64):
        xp[:, 4:36, 4:36] = x[b]
        wins = [xp[:, i:i + 32, j:j + 32] for i in range(9) for j in range(9)]
        assert any(torch.equal(out[b], w) or torch.equal(out[b], w.flip(2)) for w in wins)


def test_draws_differ_per_call():
    torch.manual_seed(0)
    x = torch.randn(32, 3, 32, 32)
    assert not torch.equal(fl_data.augment_batch(x), fl_data.augment_batch(x))


def test_gate_is_cifar_only_and_off_by_default():
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace(dataset_name="cifar10"))
    assert not t._augment_each_pass
    t.config.hyperparameters.trainer_augment_each_pass = True
    assert t._augment_each_pass
    t.config.hyperparameters.dataset_name = "google_speech"
    assert not t._augment_each_pass
