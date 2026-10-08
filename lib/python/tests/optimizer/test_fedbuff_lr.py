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
