# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N83: FedAvg `weighting` equal = Oort is_even_avg / FedScale 1/K; samples (default) unchanged."""

import pytest
import torch

from flame.mode.horizontal.syncfl.top_aggregator import MemCache
from flame.optimizer.fedavg import FedAvg
from flame.optimizer.fedscale_yogi import FedAvgYoGi
from flame.optimizer.train_result import TrainResult


def _cache():
    return MemCache(a=TrainResult({"w": torch.ones(1)}, 1, 1), b=TrainResult({"w": torch.full((1,), 2.0)}, 3, 1))


@pytest.mark.parametrize("weighting,want", [("samples", 1.75), ("equal", 1.5)])
def test_weighting(weighting, want):
    out = FedAvg(weighting=weighting).do({"w": torch.zeros(1)}, _cache(), total=4)
    assert out["w"].item() == pytest.approx(want)


def test_unknown_weighting_rejected():
    with pytest.raises(ValueError):
        FedAvg(weighting="size")


def test_fedavg_yogi_passes_weighting():
    assert FedAvgYoGi(0.005, 0.001, 0.0, 0.999, weighting="equal").weighting == "equal"
