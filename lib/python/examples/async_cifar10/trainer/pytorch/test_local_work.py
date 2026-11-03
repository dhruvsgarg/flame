# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N116: Felix's local work per task (localWork: "" | cap | split)."""

from types import SimpleNamespace

import pytest

from trainer.pytorch.main import PyTorchCifar10Trainer


def _make(local_work="", local_steps=None, beta=0.5, ms=40.0, kbps=6000.0, shard=320):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace(
        local_work=local_work, local_steps=local_steps, local_work_beta=beta))
    t.compute_ms, t.bandwidth_kbps, t.model_kbit, t.batch_size, t.epochs = ms, kbps, 16800.0, 32, 1
    t.trainer_indices_list = list(range(shard))
    return t


def test_off_keeps_configured_steps():
    assert _make(local_steps=20)._local_steps(10) == 20
    assert _make()._local_steps(10) is None


def test_cap_stops_at_one_epoch():
    assert _make("cap", local_steps=4)._local_steps(10) == 4
    assert _make("cap", local_steps=40)._local_steps(10) == 10


def test_split_sets_compute_to_beta_ratio_of_transfer():
    t = _make("split", beta=0.5)  # transfer 2 x 16800 / 6000 = 5.6 s; compute per sample 0.12 s -> 46.7 samples -> 1 step of 32
    assert t._local_steps(10) == 1
    slow_link = _make("split", beta=0.5, kbps=600.0)  # 56 s transfer -> 467 samples, capped at the epoch's 10 batches
    assert slow_link._local_steps(10) == 10
    assert _make("split", beta=0.25, kbps=1500.0)._local_steps(10) == 2  # 22.4 s x 1/3 / 0.12 = 62 samples


def test_task_time_charges_the_capped_samples():
    t = _make("cap", local_steps=40, shard=100)
    t._task_steps = t._local_steps(4)
    t._task_time_s("train")
    assert t._last_task_model["samples"] == 100
