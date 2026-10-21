# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D100: localSteps cycles the data to exactly N mini-batches; FedScale utility and mean loss follow the sources."""

import math
from types import SimpleNamespace

import torch

from main import PyTorchCifar10Trainer


def _trainer(stat_utility="legacy", local_steps=None, prefix=0):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,)))
    t.train_loader = torch.utils.data.DataLoader(data, batch_size=4)  # 2 batches per pass
    t.model, t.device, t.loss_fn = torch.nn.Linear(4, 3), torch.device("cpu"), torch.nn.CrossEntropyLoss
    t.use_oort_loss_fn, t.batch_size, t._step_lr = "True", 4, 0.1
    t.config = SimpleNamespace(hyperparameters=SimpleNamespace(
        stat_utility=stat_utility, local_steps=local_steps, lr_batch_normalize=False, utility_prefix_samples=prefix))
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=0.1)
    t.memory_profiler = SimpleNamespace(log_component_memory=lambda *a: None)
    t.reset_stat_utility()
    t.reset_local_accuracy()
    return t


def test_max_batches_caps_a_pass():
    t = _trainer()
    assert t._train_epoch(1, max_batches=1)[0] == 1
    assert t._train_epoch(2)[0] == 2


def test_fedscale_utility_is_rms_loss_times_trained_samples():
    t = _trainer("fedscale")
    t._train_epoch(1, max_batches=1)
    assert t._util_samples == 4 and t._util_ema is not None
    expected = math.sqrt(float(t._util_ema)) * 4  # min(|D|=8, 4 trained)
    t.normalize_stat_utility(1)
    assert math.isclose(t._stat_utility, expected, rel_tol=1e-6)


def test_prefix_utility_is_the_received_models_loss():
    t = _trainer("fedscale", prefix=4)  # FX-N119: first batch only = loss before any local step
    w0 = [p.detach().clone() for p in t.model.parameters()]
    x, y = next(iter(t.train_loader))
    with torch.no_grad():
        per = torch.nn.functional.cross_entropy(torch.nn.functional.linear(x, *w0), y, reduction="none")
    t._train_epoch(1)
    t.normalize_stat_utility(1)
    assert t._util_prefix_n == 4
    assert math.isclose(t._stat_utility, math.sqrt(float(torch.square(per).mean())) * 8, rel_tol=1e-5)


def test_mean_training_loss_over_local_iterations():
    t = _trainer()
    t._train_epoch(1)
    t.finalize_local_accuracy()
    assert t._train_loss_batches == 2 and t._train_loss_mean is not None and t._train_loss_mean > 0
