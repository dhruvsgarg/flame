# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""An idle trainer frees > 1 GB of cached CUDA blocks, incl. after the util_cf forward (speech felix OOM)."""

import torch

import main
from main import PyTorchCifar10Trainer


def _trainer(monkeypatch, reserved_gb, calls):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.device = torch.device("cuda", 0)
    monkeypatch.setattr(main.torch.cuda, "memory_reserved", lambda d: int(reserved_gb * (1 << 30)))
    monkeypatch.setattr(main.torch.cuda, "memory_allocated", lambda d: 0)
    monkeypatch.setattr(main.torch.cuda, "empty_cache", lambda: calls.append(1))
    return t


def test_frees_over_1gb(monkeypatch):
    calls = []
    _trainer(monkeypatch, 2.4, calls)._release_gpu_cache()
    assert calls == [1]


def test_keeps_small_cache(monkeypatch):
    calls = []
    _trainer(monkeypatch, 0.5, calls)._release_gpu_cache()
    assert calls == []


def test_cpu_noop():
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.device = torch.device("cpu")
    t._release_gpu_cache()


def test_release_runs_after_util_cf():
    import inspect
    src = inspect.getsource(PyTorchCifar10Trainer.train)
    assert src.index("_emit_util_disparity(") < src.rindex("_release_gpu_cache()")
