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


def test_eval_copy_built_at_initialize(monkeypatch):
    # FX-D142: a lazy first-eval deepcopy to the GPU put CUDA context creation on round 1's critical path (real only).
    import torch
    import agg_common
    agg = _Agg(1, True)
    agg.harness_mode = "stub"
    agg._init_oracle_util = lambda _: None
    monkeypatch.setattr(agg_common.harness, "device_for", lambda _: torch.device("cpu"))
    monkeypatch.setattr(ExampleAggregatorMixin, "data_spec",
                        property(lambda self: SimpleNamespace(model=lambda: torch.nn.Linear(2, 2))))
    agg.initialize()
    assert agg._eval_model is not None and agg._eval_model is not agg.model


class _TimedAgg(_Agg):
    """FX-N112: evalEverySeconds gates on the leg's clock (`_avail_now`), whatever the round."""

    def __init__(self, now):
        super().__init__(rnd=7, committed=True)
        self.config.hyperparameters.eval_every_s = 300.0
        self.now = now

    def _avail_now(self):
        return self.now


@pytest.mark.parametrize("now, last_mark, evals", [(0.0, -1, 1), (299.0, 0, 0), (300.0, 0, 1), (1250.0, 1, 1), (1250.0, 4, 0)])
def test_time_cadence_evaluates_once_per_mark(now, last_mark, evals):
    agg = _TimedAgg(now)
    agg._eval_last_mark = last_mark
    agg.evaluate()
    assert agg.snapshots == evals


def test_ema_tracks_each_committed_version_and_serve_beta_is_one_sweep():
    # FX-N128: ema_b <- b ema_b + (1 - b) model per committed version; serveEma=sweep uses b = 1 - K / contributors
    import torch
    agg = _Agg(1, True)
    agg.config.hyperparameters = SimpleNamespace(evalEma="0.5", serveEma="sweep")
    agg.model, agg._agg_goal = torch.nn.Linear(1, 1, bias=False), 10
    agg._trainer_participation_in_round_count = {f"t{i}": 1 for i in range(40)}
    for w in (0.0, 2.0, 4.0):
        with torch.no_grad():
            agg.model.weight.fill_(w)
        agg._ema_step()
    assert agg._emas[0.5]["weight"].item() == pytest.approx(0.5 * (0.5 * 0 + 0.5 * 2) + 0.5 * 4)  # 2.5
    b = 1 - 10 / 40
    assert agg._serve_beta() == pytest.approx(b)
    assert agg._emas["serve"]["weight"].item() == pytest.approx(b * (b * 0 + (1 - b) * 2) + (1 - b) * 4)


def test_no_ema_knobs_touch_nothing():
    agg = _Agg(21, True)
    agg._ema_step()  # no model needed: off is a no-op (R9)
    assert agg._emas == {} and agg._serve is None
