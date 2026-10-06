# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D58: at most `gpu_train_slots` trainers train at once on one GPU (speech c=30 first wave OOMed an A40)."""

import types

import pytest
import torch

import main
from main import PyTorchCifar10Trainer


def _trainer(slots, gpu):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.device = torch.device("cuda", 0)
    t.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(gpu_train_slots=slots))
    return t


def test_caps_holders_per_gpu(monkeypatch):
    gpu = f"t{id(object())}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    a, b = _trainer(2, gpu)._acquire_gpu_slot(), _trainer(2, gpu)._acquire_gpu_slot()
    assert a is not None and b is not None and a.name != b.name

    def full(_s):
        raise TimeoutError
    monkeypatch.setattr(main.time, "sleep", full)
    with pytest.raises(TimeoutError):  # third holder waits
        _trainer(2, gpu)._acquire_gpu_slot()
    a.close()
    monkeypatch.undo()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    c = _trainer(2, gpu)._acquire_gpu_slot()  # a freed slot is reused
    assert c.name == a.name
    b.close(), c.close()


def test_uncapped_and_cpu_return_none():
    assert _trainer(0, "x")._acquire_gpu_slot() is None
    t = _trainer(2, "x")
    t.device = torch.device("cpu")
    assert t._acquire_gpu_slot() is None


def test_slot_released_after_cache_release():
    import inspect
    src = inspect.getsource(PyTorchCifar10Trainer.train)
    assert src.index("_acquire_gpu_slot()") < src.index("self._release_gpu_cache()") < src.index("_gpu_slot.close()")
