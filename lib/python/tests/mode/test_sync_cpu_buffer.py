"""CPU buffering preserves the weighted GPU reduction path."""
from unittest.mock import Mock

import pytest
import torch

from flame.mode.horizontal.syncfl.top_aggregator import MemCache
from flame.optimizer.fedavg import FedAvg
from flame.optimizer.train_result import TrainResult
from tests.mode.test_sync_result_lifecycle import Channel, make_runtime


def test_cpu_buffered_updates_preserve_weighted_delta_math():
    cache = MemCache({
        "a": TrainResult({"weight": torch.tensor([2.0])}, 1),
        "b": TrainResult({"weight": torch.tensor([6.0])}, 3),
    })
    result = FedAvg().do({"weight": torch.tensor([10.0])}, cache, total=4)
    assert result["weight"].item() == 15
    assert not cache


def test_sync_uses_buffering_hook():
    ch = Channel({"a": 1.0, "b": 2.0})
    agg = make_runtime(ch, 2)
    agg._weights_for_aggregation = Mock(side_effect=lambda weights: weights)
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    assert agg._weights_for_aggregation.call_count == 2
    assert agg.model.weight.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_updates_reduce_on_gpu_without_moving_the_cache():
    cached = torch.tensor([2.0])
    cache = MemCache({"a": TrainResult({"weight": cached}, 1)})
    result = FedAvg().do({"weight": torch.tensor([10.0], device="cuda")}, cache, total=1)
    assert result["weight"].device.type == "cuda"
    assert cached.device.type == "cpu"
    assert result["weight"].item() == 12
