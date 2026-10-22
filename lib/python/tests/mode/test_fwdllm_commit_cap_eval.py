# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Paper-run stop and eval plumbing: commit cap, cap-scaled stop-rule warm-up,
per-eval resampled test sample, final full-test eval."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator
from examples.fwdllm.expts.saturation_stop import (
    SAT_SLOPE_FALLBACK, SAT_WARMUP_FALLBACK, slope_horizon_commits, warmup_commits,
)
from flame import telemetry
from flame.mode.horizontal.syncfl import fwdllm_aggregator as fa
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


# ---- 4: warm-up scales with the commit cap ---------------------------------

def test_warmup_uncapped_is_unchanged():
    assert warmup_commits(0) == SAT_WARMUP_FALLBACK == 450
    assert slope_horizon_commits(0) == SAT_SLOPE_FALLBACK == 150


def test_warmup_scales_with_cap():
    assert (warmup_commits(0, commit_cap=1000), slope_horizon_commits(0, commit_cap=1000)) == (450, 150)
    assert (warmup_commits(0, commit_cap=400), slope_horizon_commits(0, commit_cap=400)) == (180, 60)


def test_probe_cadence_wins_over_cap():
    assert warmup_commits(100, commit_cap=400) == 300
    assert slope_horizon_commits(100, commit_cap=400) == 100


# ---- 1: commit cap ---------------------------------------------------------

def _stopper(cap, phi_stop="off"):
    a = SimpleNamespace(_stop_fired=None, _max_commits=cap, _commit_count=0,
                        _phi_stop=phi_stop, _B=0.0, _work_done=False,
                        _sat_det=None, _phi_stop_threshold=3.0,
                        _rho_schedule="exp", _budget_stop_frac=float("inf"), _b_max=1.0)
    a.check = lambda: FedSGDAggregator._check_budget_stop(a)
    return a


def test_cap_halts_even_with_phi_stop_off():
    a = _stopper(5)
    for n in range(1, 5):
        a._commit_count = n
        a.check()
        assert not a._work_done
    a._commit_count = 5
    a.check()
    assert a._work_done and a._stop_fired == "commit_cap"


def test_no_cap_is_inert():
    a = _stopper(None)
    a._commit_count = 10_000
    a.check()
    assert not a._work_done and a._stop_fired is None


def test_earlier_stop_reason_is_kept():
    a = _stopper(5, phi_stop="halt")
    a._stop_fired = "stall"
    a._commit_count = 5
    a.check()
    assert a._stop_fired == "stall"


# ---- 2 + 3: resampled eval, final full-test eval ---------------------------

class _Agg:
    eval_model = TopAggregator.eval_model
    compute_metrics = TopAggregator.compute_metrics
    _final_full_eval = TopAggregator._final_full_eval

    def log_memory(self, tag, device):
        pass

    def __init__(self, n=40, n_max=10, resample=True, final=True):
        self.device = torch.device("cpu")
        self.model = nn.Linear(4, 2)
        self.num_labels = 2
        self.args = SimpleNamespace(eval_batch_size=8, fp16=False)
        self.config = SimpleNamespace(hyperparameters=SimpleNamespace(
            eval_max_samples=n_max, eval_resample=resample, final_full_eval=final))
        g = torch.Generator().manual_seed(0)
        # input_ids carry the row index, so the rows scored are recoverable
        rows = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
        labels = torch.randint(0, 2, (n,), generator=g)
        filler = torch.zeros(n)
        ds = SimpleNamespace(tensors=[filler, rows, filler, filler, labels])
        self.test_global = SimpleNamespace(dataset=ds, examples=None)
        self._cached_test_data = None
        self.iteration_per_data_id = 0
        self._round = 1
        self.seen = []
        _fwd = self.model.forward
        self.model.forward = lambda x: (self.seen.append(x[:, 0].long().tolist()), _fwd(x))[1]

    def rows(self, **kw):
        self.seen = []
        out = self.eval_model(**kw)
        return sorted(r for b in self.seen for r in b), out


def test_resample_draws_differ_per_commit_and_repeat_per_key():
    a = _Agg()
    r1, _ = a.rows(sample_key=1)
    r2, _ = a.rows(sample_key=2)
    r1b, _ = a.rows(sample_key=1)
    assert len(r1) == len(r2) == 10
    assert r1 != r2 and r1 == r1b


def test_no_key_keeps_the_fixed_draw():
    a = _Agg()
    r_a, _ = a.rows()
    r_b, _ = a.rows()
    assert len(r_a) == 10 and r_a == r_b


def test_resample_off_is_the_fixed_draw():
    a = _Agg(resample=False)
    fixed, _ = a.rows()
    keyed, _ = a.rows(sample_key=7)
    assert fixed == keyed


def test_full_scores_every_row():
    a = _Agg()
    a.rows(sample_key=1)  # cache the subsample first
    rows, _ = a.rows(full=True)
    assert rows == list(range(40))


def test_final_eval_emits_full_set(monkeypatch):
    a = _Agg()
    got = []
    monkeypatch.setattr(telemetry, "is_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "emit", lambda ev, **f: got.append((ev, f)))
    a.seen = []
    a._final_full_eval()
    assert sorted(r for b in a.seen for r in b) == list(range(40))
    assert [ev for ev, _ in got] == ["agg_eval_final"]
    assert got[0][1]["n_rows"] == 40 and got[0][1]["test-accuracy"] is not None


def test_final_eval_off_by_default():
    a = _Agg(final=False)
    a._final_full_eval()
    assert a.seen == []


def test_final_eval_is_in_both_compose_chains():
    import inspect
    src = inspect.getsource(fa.TopAggregator.compose)
    assert src.count(">> task_final_eval\n") == 2


# ---- 7: rho0 set directly ----------------------------------------------------

def _rho(sched="exp", rho0=None, s=1.5, pool=50, p=806_912, G=10.0):
    a = SimpleNamespace(args=SimpleNamespace(rho0=rho0), _rho_schedule=sched, _gate_safety_s=s,
                        _g_rule=G, _pool_target=pool, _p_trainable=p, _rho_star=0.01)
    FedSGDAggregator._set_rho0(a, sched == "exp" and pool is not None)
    return a


def test_rho0_from_s_when_unset():
    a = _rho(s=2.81)
    assert abs(a._rho_star - 2.81 * (10 * 50 / 806_912) ** 0.5) < 1e-12 and a._rho0 is None


def test_rho0_overrides_s_and_s_is_implied():
    a = _rho(rho0=0.07, s=1.5)
    assert a._rho_star == 0.07
    assert abs(a._gate_safety_s - 0.07 / (10 * 50 / 806_912) ** 0.5) < 1e-9   # ~2.81


def test_rho0_pool_schedule():
    assert _rho(sched="pool", rho0=0.05)._rho_star == 0.05


def test_rho0_exp_without_pool():
    assert _rho(rho0=0.06, pool=None)._rho_star == 0.06


def test_rho0_ignored_by_const():
    a = _rho(sched="const", rho0=0.2, pool=None)
    assert a._rho_star == 0.01 and a._gate_safety_s == 1.5
