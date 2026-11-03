# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D63: an example aggregator evaluates on every `evalEveryNRounds` round; the commit stride must not halve it."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator

AGG = Path(__file__).resolve().parents[2] / "aggregator" / "pytorch"


def _load(name):
    sys.path.insert(0, str(AGG.parent.parent))
    spec = importlib.util.spec_from_file_location(name, AGG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name,cls", [("main_asyncfl_agg", "PyTorchCifar10Aggregator"),
                                      ("main_oort_sync_agg", None), ("main_fedavg_agg", None)])
def test_every_gated_round_evaluates(name, cls):
    mod = _load(name)
    klass = getattr(mod, cls) if cls else next(v for k, v in vars(mod).items()
                                                if isinstance(v, type) and hasattr(v, "evaluate") and v.__module__ == name)
    agg = klass.__new__(klass)
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(eval_every_n_rounds=20, eval_every_n_commits=2))
    agg.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2), torch.nn.LogSoftmax(dim=1))
    agg.device = agg.eval_device = "cpu"
    agg.test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)), batch_size=8)
    agg.loss_list = []
    seen = []
    agg._eval_emit = lambda r, loss, acc: (seen.append(r), TopAggregator._eval_release(agg))
    agg._check_target_stop = lambda *a: None
    for r in range(1, 81):
        agg._round = r
        agg.evaluate()
        done = getattr(agg, "_eval_done", None)
        if done is not None:
            done.wait(5)
    assert [r for r in seen if r != 1] == [20, 40, 60, 80]
