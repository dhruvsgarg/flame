# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D233 / N143: per-batch train augmentation (cifar crop / flip, speech FedScale pipeline; trainerAugmentEachPass)."""

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


def test_gate_is_on_by_default_for_both_datasets_and_off_in_stub():
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace(dataset_name="cifar10"))
    assert t._augment_each_pass
    t.config.hyperparameters.trainer_augment_each_pass = False
    assert not t._augment_each_pass
    t.config.hyperparameters.trainer_augment_each_pass = True
    t.config.hyperparameters.dataset_name = "google_speech"
    assert t._augment_each_pass
    t.harness_mode = "stub"
    assert not t._augment_each_pass


def _speech_off(monkeypatch):
    monkeypatch.setattr(fl_data, "_speech_noise", lambda dev: torch.zeros(128, fl_data.SPEECH_LEN))


def test_speech_identity_when_no_transform_fires(monkeypatch):
    _speech_off(monkeypatch)
    x = torch.randn(8, 1, fl_data.SPEECH_LEN)
    assert torch.allclose(fl_data.augment_speech_batch(x, p=0.0), x)


def test_speech_shift_and_speed_keep_length_and_content(monkeypatch):
    _speech_off(monkeypatch)
    torch.manual_seed(0)
    x = torch.randn(16, 1, fl_data.SPEECH_LEN)
    out = fl_data.augment_speech_batch(x, p=1.0)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert not torch.allclose(out, x)
    assert (out.abs().amax(2) <= 1.1 * x.abs().amax(2) + 1e-5).all()  # amplitude ≤ 1.1, interpolation never overshoots


def test_speech_pure_shift_moves_the_clip(monkeypatch):
    _speech_off(monkeypatch)
    x = torch.randn(1, 1, fl_data.SPEECH_LEN)
    seq = iter([torch.tensor([False]), torch.tensor([False]), torch.tensor([True]), torch.tensor([False])])  # amp, speed, shift, noise
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.zeros(1) if next(seq) else torch.ones(1))
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.full(a[-1], 100, dtype=torch.long))
    out = fl_data.augment_speech_batch(x, p=0.5)
    assert torch.allclose(out[0, 0, :-100], x[0, 0, 100:]) and (out[0, 0, -100:] == 0).all()
