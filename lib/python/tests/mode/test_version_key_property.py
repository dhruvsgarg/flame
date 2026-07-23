# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""`version_key` is the shared step-identity property on the syncfl
TopAggregator base (plain sync FL: no intra-round iteration axis).
fwdllm_aggregator overrides it with (model_version, iteration_per_data_id) --
data_id is not in the key since model_version bumps once per data-bin.
asyncfl does not override it -- same 2-tuple vocabulary, iteration always 0.
"""

from flame.mode.horizontal.asyncfl.top_aggregator import (
    TopAggregator as AsyncTopAggregator,
)
from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    TopAggregator as FwdllmTopAggregator,
)
from flame.mode.horizontal.syncfl.top_aggregator import (
    TopAggregator as SyncTopAggregator,
)


class _BareBase:
    version_key = SyncTopAggregator.version_key

    def __init__(self, round_):
        self._round = round_


class _BareFwdllm:
    version_key = FwdllmTopAggregator.version_key

    def __init__(self, model_version, data_id, iteration_per_data_id):
        self._model_version = model_version
        self.data_id = data_id
        self.iteration_per_data_id = iteration_per_data_id


def test_base_version_key_is_round_and_zero():
    assert _BareBase(3).version_key == (3, 0)


def test_fwdllm_version_key_drops_data_id():
    agg = _BareFwdllm(model_version=5, data_id=2, iteration_per_data_id=1)
    assert agg.version_key == (5, 1)


def test_fwdllm_version_key_ignores_data_id_changes():
    # data_id is a reporting/progress field only -- not part of the key.
    a = _BareFwdllm(model_version=5, data_id=2, iteration_per_data_id=1)
    b = _BareFwdllm(model_version=5, data_id=99, iteration_per_data_id=1)
    assert a.version_key == b.version_key


def test_asyncfl_inherits_the_base_version_key_unoverridden():
    # async_cifar10's fedbuff/felix/oort baselines share the base (round, 0)
    # vocabulary verbatim -- no asyncfl-specific override.
    assert AsyncTopAggregator.version_key is SyncTopAggregator.version_key
