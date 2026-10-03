# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N64: a stale delta must not drive BatchNorm running_var negative (NaN at eval)."""
import torch

from flame.optimizer.bn_buffers import clamp_running_var


def test_clamps_only_running_var():
    w = {"layer4.2.bn2.running_var": torch.tensor([0.5, -0.005]), "bn.running_mean": torch.tensor([-1.0]),
         "conv.weight": torch.tensor([-2.0])}
    assert clamp_running_var(w) == 1
    assert w["layer4.2.bn2.running_var"].min().item() == 0.0
    assert w["bn.running_mean"].item() == -1.0 and w["conv.weight"].item() == -2.0


def test_refl_add_deltas_clamps_and_knob_reverts():
    from flame.optimizer.refl import REFL
    for knob, want_min in ((True, 0.0), ("False", -0.3)):
        opt = REFL(clamp_running_var=knob)
        base = {"bn.running_var": torch.tensor([0.1, 0.2])}
        opt._add_deltas_to_base(base, {"bn.running_var": torch.tensor([-0.4, 0.0])})
        assert abs(base["bn.running_var"].min().item() - want_min) < 1e-6


def _agg(tmp_path, **kw):
    from diskcache import Cache
    from flame.optimizer.refl import REFL
    from flame.optimizer.train_result import TrainResult
    opt = REFL(deadline=100.0, stale_update=5, stale_factor=-2, **kw)
    base = {"w": torch.tensor([1.0]), "bn.running_var": torch.tensor([0.1])}
    with Cache(str(tmp_path)) as cache:
        cache["f"] = TrainResult(weights={"w": torch.tensor([0.2]), "bn.running_var": torch.tensor([-0.05])},
                                 count=10, end_id="f", staleness=0)
        cache["s"] = TrainResult(weights={"w": torch.tensor([0.2]), "bn.running_var": torch.tensor([-0.5])},
                                 count=10, end_id="s", staleness=2)
        return opt.do(base, cache, total=20, version=3, round_duration=1.0)


def test_refl_bn_stats_use_fresh_updates_only(tmp_path):
    out = _agg(tmp_path, clamp_running_var="False")
    assert abs(out["bn.running_var"].item() - 0.05) < 1e-6  # 0.1 + the fresh delta alone; the stale -0.5 is ignored
    assert out["w"].item() > 1.0  # parameters still average the stale update too


def test_refl_bn_fresh_only_knob_reverts(tmp_path):
    out = _agg(tmp_path, clamp_running_var="False", bn_fresh_only="False")
    assert out["bn.running_var"].item() < 0  # the old all-updates delta average goes negative
