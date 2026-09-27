# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D17: REFL aggregates a model with integer buffers (BatchNorm) and keeps each tensor's dtype."""

import torch
from diskcache import Cache

from flame.optimizer.refl import REFL
from flame.optimizer.train_result import TrainResult


def _bn_weights(w, nbt):
    return {"w": torch.full((2,), float(w)), "bn.num_batches_tracked": torch.tensor(nbt, dtype=torch.long)}


def test_refl_aggregates_integer_buffers(tmp_path):
    refl = REFL(deadline=100.0)
    base = _bn_weights(1.0, 10)
    with Cache(str(tmp_path)) as cache:
        cache["a"] = TrainResult(weights=_bn_weights(0.5, 5), count=10, end_id="a")
        cache["b"] = TrainResult(weights=_bn_weights(1.5, 6), count=10, end_id="b")
        out = refl.do(base, cache, total=20, version=1, round_duration=1.0)

    assert out["bn.num_batches_tracked"].dtype == torch.long
    assert out["w"].dtype == torch.float32
    assert int(out["bn.num_batches_tracked"]) in (15, 16)  # 10 + round(weighted mean of 5, 6)
    assert torch.allclose(out["w"], torch.full((2,), 2.0))  # 1 + mean(0.5, 1.5) at equal importance


def test_refl_gradient_policy_skips_integer_buffers(tmp_path):
    """YoGi acts on float tensors only (REFL: model.parameters()); integer buffers keep the average."""
    refl = REFL(deadline=100.0, gradient_policy="yogi")
    base = _bn_weights(1.0, 10)
    for i in range(2):
        with Cache(str(tmp_path / str(i))) as cache:
            cache["a"] = TrainResult(weights=_bn_weights(0.5, 5), count=10, end_id="a")
            base = refl.do(base, cache, total=10, version=i + 1, round_duration=1.0)
    assert base["bn.num_batches_tracked"].dtype == torch.long
    assert int(base["bn.num_batches_tracked"]) == 20
    assert base["w"].dtype == torch.float32
