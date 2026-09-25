# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""_effective_delay_s: floor then trainingDelayFactor; unset = registry D unchanged."""

from types import SimpleNamespace

import pytest

from trainer.pytorch.main import PyTorchCifar10Trainer


def _make(**hp):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace(**hp))
    return t


def test_unset_is_raw():
    assert _make()._effective_delay_s(9.5) == 9.5


def test_factor_divides():
    assert _make(training_delay_factor=4.0)._effective_delay_s(8.0) == pytest.approx(2.0)


def test_floor_applies_before_factor():
    t = _make(training_delay_factor=2.0, training_delay_floor_s=6.0)
    assert t._effective_delay_s(4.0) == pytest.approx(3.0)


def test_stub_load_data_builds_synthetic_loader():
    import torch
    from memory_profiler import MemoryProfiler

    t = _make(harness_mode="stub", harness_samples=8)
    t.harness_mode = "stub"
    t.trainer_id = 3
    t.trainer_indices_list = list(range(100))
    t.device = torch.device("cpu")
    t.batch_size = 4
    t.training_delay_s = 1.0
    t.memory_profiler = MemoryProfiler(trainer_id="3", enabled=False)
    t.data_streaming_enabled = False
    t.data_streaming_full_after_s = 0.0
    t.stream_stagger_enabled = False
    t.stream_min_visible = 1
    t._stream_onset_s = 0.0
    t._stream_span_s = 0.0
    t.load_data()
    assert len(t.train_loader.dataset) == 8
    x, y = next(iter(t.train_loader))
    assert tuple(x.shape[1:]) == (3, 32, 32) and x.device.type == "cpu"
