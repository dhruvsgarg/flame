# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D128: the example aggregators evaluate an eval round once, on the pass that committed it."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EX = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EX / "async_cifar10"))
sys.path.insert(0, str(EX / "async_cifar10" / "aggregator" / "pytorch"))
from agg_common import ExampleAggregatorMixin  # noqa: E402


class _Agg(ExampleAggregatorMixin):
    def __init__(self, rnd, committed):
        self._round = rnd
        if committed is not None:
            self._round_committed = committed
        self.config = SimpleNamespace(hyperparameters=SimpleNamespace(eval_every_n_rounds=20))
        self.snapshots = 0

    def _eval_snapshot_model(self):
        self.snapshots += 1
        return None  # stop before the eval thread


@pytest.mark.parametrize("committed, evals", [(False, 0), (True, 1), (None, 1)])
def test_eval_round_runs_only_on_committed_pass(committed, evals):
    agg = _Agg(20, committed)  # None: a stack that doesn't track commits keeps the round gate alone
    agg.evaluate()
    assert agg.snapshots == evals


def test_non_eval_round_skips():
    agg = _Agg(21, True)
    agg.evaluate()
    assert agg.snapshots == 0
