# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""S5: FedBuff's server lr is the required `learning_rate` kwarg; no dataset_name table."""

import pytest
import torch

from flame.optimizer.fedbuff import FedBuff


def _make_fedbuff(**overrides):
    kwargs = dict(
        use_oort_lr="False",
        dataset_name="unknown-dataset",
        agg_rate_conf={"type": "old"},
    )
    kwargs.update(overrides)
    return FedBuff(**kwargs)


def _scale_add(fedbuff):
    base_weights = {"w": torch.zeros(3)}
    agg_goal_weights = {"w": torch.ones(3)}
    return fedbuff.scale_add_agg_weights(base_weights, agg_goal_weights, agg_goal=1)


class TestFedBuffLearningRate:
    def test_explicit_learning_rate_is_the_server_lr(self):
        fedbuff = _make_fedbuff(learning_rate=0.5, dataset_name="cifar-10")
        result = _scale_add(fedbuff)
        assert torch.allclose(result["w"], torch.full((3,), 0.5))

    def test_missing_learning_rate_fails_loudly(self):
        with pytest.raises(KeyError):
            _make_fedbuff(dataset_name="cifar-10")


class TestFedBuffNormRate:
    """FX-N109 I7: rate "norm" weights u_i / u_bar x 1/sqrt(1+s); the cycle's mean weight is FedBuff's."""

    def test_utility_reweights_within_cycle(self):
        from flame.mode.horizontal.syncfl.top_aggregator import MemCache
        from flame.optimizer.train_result import TrainResult
        fb = _make_fedbuff(learning_rate=1.0, agg_rate_conf={"type": "norm"})
        agg = None
        for u, d in ((1.0, 1.0), (3.0, 2.0)):
            agg = fb.do(agg, MemCache({"e": TrainResult({"w": torch.full((1,), d)}, 1, 5, u)}), total=1, version=5)
        out = fb.scale_add_agg_weights({"w": torch.zeros(1)}, agg, agg_goal=2)
        assert torch.allclose(out["w"], torch.tensor([(0.5 * 1.0 + 1.5 * 2.0) / 2]))
        assert fb._u_n == 0  # reset for the next cycle

    def test_staleness_discount_is_fedbuffs(self):
        from flame.mode.horizontal.syncfl.top_aggregator import MemCache
        from flame.optimizer.train_result import TrainResult
        fb = _make_fedbuff(learning_rate=1.0, agg_rate_conf={"type": "norm"})
        agg = fb.do(None, MemCache({"e": TrainResult({"w": torch.ones(1)}, 1, 2, 4.0)}), total=1, version=5)  # staleness 3
        out = fb.scale_add_agg_weights({"w": torch.zeros(1)}, agg, agg_goal=1)
        assert torch.allclose(out["w"], torch.tensor([0.5]))


class TestFedBuffValueRate:
    """C5: rate "value" weights samples^beta x (1+s)^-a, normalized to a mean weight of 1 per cycle."""

    def test_weights_by_samples_and_staleness(self):
        from flame.mode.horizontal.syncfl.top_aggregator import MemCache
        from flame.optimizer.train_result import TrainResult
        fb = _make_fedbuff(learning_rate=1.0, agg_rate_conf={"type": "value", "beta": 0.5, "a": 1.0})
        agg = None
        for n, ver, d in ((4, 5, 1.0), (16, 4, 2.0)):  # u = 2 / 1 = 2 and 4 / 2 = 2
            agg = fb.do(agg, MemCache({"e": TrainResult({"w": torch.full((1,), d)}, n, ver, 1.0)}), total=n, version=5)
        out = fb.scale_add_agg_weights({"w": torch.zeros(1)}, agg, agg_goal=2)
        assert torch.allclose(out["w"], torch.tensor([(1.0 + 2.0) / 2]))
        assert fb._u_n == 0


class TestFedBuffNormClip:
    """FX-D232: an update above norm_clip x the running median norm is scaled down to it; off by default."""

    def _run(self, norms, **kw):
        from flame.mode.horizontal.syncfl.top_aggregator import MemCache
        from flame.optimizer.train_result import TrainResult
        fb = _make_fedbuff(learning_rate=1.0, **kw)
        agg = None
        for n in norms:
            agg = fb.do(None, MemCache({"e": TrainResult({"w": torch.tensor([float(n)])}, 1, 5, 1.0)}), total=1, version=5)
        return agg["w"].item()

    def test_outlier_is_clipped_to_k_median(self):
        assert self._run([1.0] * 10 + [100.0], norm_clip=3) == pytest.approx(3.0)

    def test_off_by_default_and_before_ten_norms(self):
        assert self._run([1.0] * 10 + [100.0]) == pytest.approx(100.0)
        assert self._run([1.0] * 5 + [100.0], norm_clip=3) == pytest.approx(100.0)

    def test_typical_update_untouched(self):
        assert self._run([1.0] * 10 + [2.5], norm_clip=3) == pytest.approx(2.5)
