# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D41: each eval reports a fresh utility (not summed onto the last) and keeps no autograd graph."""

from types import SimpleNamespace

import torch

from main import PyTorchCifar10Trainer


def _trainer():
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,)))
    t.train_loader = torch.utils.data.DataLoader(data, batch_size=4)
    t.model, t.device, t.loss_fn = torch.nn.Linear(4, 3), torch.device("cpu"), torch.nn.CrossEntropyLoss
    t.task_to_perform, t.client_notify, t.trainer_id = "eval", {"trace": "three_state"}, "t"
    t._refresh_avl_state = lambda: None
    t.avl_state = SimpleNamespace(value="AVL_EVAL")
    t.data_streaming_enabled, t.epochs, t.use_oort_loss_fn = False, 1, "True"
    t.training_delay_enabled, t.simulated = False, False
    return t


def test_eval_utility_is_fresh_and_graph_free():
    t = _trainer()
    t._stat_utility = 1e6  # a previous task's value must not leak in
    t.evaluate()
    first = float(t._stat_utility)
    assert first < 1e3
    assert not (torch.is_tensor(t._stat_utility) and t._stat_utility.requires_grad)
    t.evaluate()
    assert float(t._stat_utility) == first
