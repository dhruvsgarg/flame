"""FMoW releases the shared evaluation event on success and failure."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_always_releases_completion(fail, monkeypatch):
    path = Path(__file__).parents[1] / "aggregator/pytorch/main_fedavg_agg.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("fmow_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agg = module.PyTorchFMoWAggregator.__new__(module.PyTorchFMoWAggregator)
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(eval_every_n_rounds=1))
    agg._round = agg._rounds = 1
    agg.device = torch.device("cpu")
    agg.loss_list = []
    agg.test_loader = [(torch.ones(1, 1), torch.zeros(1, dtype=torch.long))]
    agg._eval_snapshot_model = lambda: Mock(side_effect=RuntimeError("eval failed")) if fail else torch.nn.Linear(1, 1)
    agg._eval_emit = Mock()
    agg._eval_inflight = True
    agg._eval_done = threading.Event()
    agg.evaluate()
    assert agg._eval_done.is_set()
    assert not agg._eval_inflight
    assert agg._eval_emit.call_count == (0 if fail else 1)
