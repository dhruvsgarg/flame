# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D15: the first task must not time CUDA init. A CPU model never touches CUDA in the timed
weights_to_gpu phase; a CUDA trainer warms up in initialize() and keeps its weights."""

from types import SimpleNamespace

import pytest
import torch

from flame.mode.horizontal.syncfl.trainer import Trainer
from main import PyTorchCifar10Trainer


def test_cpu_model_is_not_on_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("timed")))
    t = SimpleNamespace(model=torch.nn.Linear(2, 2))
    assert Trainer._model_on_cuda(t) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_warmup_restores_weights():
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace())
    t.device, t.batch_size, t.trainer_id = torch.device("cuda:0"), 4, "t"
    t.model = t.data_spec.model().to(t.device)
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    t._warmup_device()
    after = t.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert Trainer._model_on_cuda(t) is True


def test_cpu_warmup_runs_backward_and_restores_weights():
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace())
    t.device, t.batch_size, t.trainer_id = torch.device("cpu"), 4, "t"
    t.model = t.data_spec.model()
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    t._warmup_device()
    assert all(torch.equal(before[k], v) for k, v in t.model.state_dict().items())
    assert all(p.grad is None for p in t.model.parameters())
