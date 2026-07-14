# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""S1 (fluxtune_contributions.md §8.2): server-side momentum damping on the raw
per-parameter SGD update direction, gating the undamped direct-SGD step (F8)
implicated in the random-walk/position-locked-collapse instability there.
`_server_update_step` is the PURE transform (reads/writes only
`server_momentum`/`_server_momentum_buf`) driven from FedSGDAggregator.
aggregate()'s two update sites (natural commit + force-commit/MaxIterBypass --
both call it identically). These tests pin:
(1) momentum=0.0 (default) is byte-identical: returns `raw_update` unchanged,
    no buffer created -- the shared-code path all 3 baselines run stays
    unmodified unless `hyperparameters.server_momentum` is set;
(2) momentum>0 applies heavy-ball recursion across successive commits;
(3) buffers are independent per parameter index;
(4) the returned buffer is a clone, not an alias of the caller's tensor.
"""

import torch

from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator


class _MomentumAgg:
    """Minimal stand-in exposing only what `_server_update_step` reads."""

    def __init__(self, momentum=0.0):
        self.server_momentum = momentum
        self._server_momentum_buf = {}

    step = FedSGDAggregator._server_update_step


def test_zero_momentum_is_byte_identical():
    agg = _MomentumAgg(momentum=0.0)
    raw = torch.tensor([1.0, 2.0, 3.0])
    out = agg.step(0, raw)
    assert out is raw
    assert agg._server_momentum_buf == {}


def test_momentum_first_commit_equals_raw_update():
    agg = _MomentumAgg(momentum=0.9)
    raw = torch.tensor([1.0, 2.0])
    out = agg.step(0, raw)
    assert torch.allclose(out, raw)
    assert 0 in agg._server_momentum_buf


def test_momentum_accumulates_heavy_ball():
    agg = _MomentumAgg(momentum=0.5)
    out1 = agg.step(0, torch.tensor([1.0]))
    assert torch.allclose(out1, torch.tensor([1.0]))
    out2 = agg.step(0, torch.tensor([2.0]))
    # buf = momentum*buf + raw2 = 0.5*1.0 + 2.0 = 2.5
    assert torch.allclose(out2, torch.tensor([2.5]))


def test_momentum_buffers_are_independent_per_param():
    agg = _MomentumAgg(momentum=0.5)
    agg.step(0, torch.tensor([10.0]))
    agg.step(1, torch.tensor([1.0]))
    out0 = agg.step(0, torch.tensor([0.0]))
    out1 = agg.step(1, torch.tensor([0.0]))
    assert torch.allclose(out0, torch.tensor([5.0]))  # 0.5*10 + 0
    assert torch.allclose(out1, torch.tensor([0.5]))  # 0.5*1 + 0


def test_returned_buffer_is_not_aliased_to_caller_tensor():
    agg = _MomentumAgg(momentum=0.9)
    raw = torch.tensor([1.0])
    out = agg.step(0, raw)
    out.add_(100.0)
    assert raw.item() == 1.0
