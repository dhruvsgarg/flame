import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def aggregator(monkeypatch, **hp):
    path = Path(__file__).parents[1] / "aggregator/pytorch/main_fedavg_agg.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("fmow_host_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.PyTorchFMoWAggregator
    agg = cls.__new__(cls)
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(**hp))
    return agg


def test_host_receive_bound_does_not_use_modeled_link_delay(monkeypatch):
    agg = aggregator(monkeypatch, trainer_recv_wall_timeout_s=7)
    agg._sim_known_delay_s = {"a": 100000}
    assert agg._sim_recv_timeout_s(["a", "unseen"]) == 7


def test_missing_first_connection_fails_before_unbounded_barrier(monkeypatch):
    agg = aggregator(monkeypatch, min_trainers_join_timeout_s=2)
    channel = SimpleNamespace(await_join=Mock(return_value=True))
    with pytest.raises(TimeoutError, match="No trainer"):
        agg._await_training_channel(channel)
    channel.await_join.assert_called_once_with(timeout=2)


def test_missing_cohort_does_not_silently_start_partial_experiment(monkeypatch):
    agg = aggregator(monkeypatch, min_trainers_join_timeout_s=0.001, min_trainers_to_start=2)
    channel = SimpleNamespace(await_join=lambda **kw: False, _ends={"a": None})
    with pytest.raises(TimeoutError, match="1/2"):
        agg._await_training_channel(channel)


def test_incomplete_physical_cohort_cannot_change_modeled_winners():
    from tests.mode.test_sync_sim_ordering import _make_agg, FakeSyncChannel
    agg = _make_agg()
    agg.require_complete_sim_cohort = True
    ch = FakeSyncChannel({"a": 2.0, "b": 1.0}, ["a"])
    with pytest.raises(TimeoutError, match="b"):
        agg._sync_sim_recv_first_k(ch, ch.ends(), 1)
    assert agg._vclock.now == 0
