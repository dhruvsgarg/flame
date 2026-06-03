# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical real/sim parity checks (parity_checks.py).

Runs in the default suite on synthetic telemetry — no MQTT/GPU — so the parity
logic itself is verified independently of any live run. The opt-in end-to-end
check (test_real_sim_e2e_parity.py) reuses the same functions on real runs.
"""

import pathlib
import sys

import pytest

# parity_checks lives with the async_cifar10 example scripts.
_SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples" / "async_cifar10" / "scripts"
)
sys.path.insert(0, str(_SCRIPTS))
import parity_checks as pc  # noqa: E402


def _agg(selection=None, agg_rounds=None, agg_evals=None):
    return {
        "selection_train": selection or [],
        "agg_rounds": agg_rounds or [],
        "agg_evals": agg_evals or [],
    }


def _sel(round_, chosen, ts=0.0):
    return {"event": "selection", "task": "train", "round": round_,
            "ts": ts, "chosen": chosen}


def _round(round_, contributing, staleness, agg_goal_count=1, vclock=None, ts=0.0):
    e = {"event": "agg_round", "round": round_, "ts": ts,
         "contributing_trainers": contributing, "staleness": staleness,
         "agg_goal_count": agg_goal_count}
    if vclock is not None:
        e["vclock_now"] = vclock
    return e


class TestSelectionParity:
    def test_identical_is_exact(self):
        a = _agg(selection=[_sel(1, ["x", "y"]), _sel(2, ["y", "z"])])
        r = pc.selection_parity(a, a)
        assert r["ok"] and r["mean_jaccard"] == 1.0 and r["exact_match_frac"] == 1.0

    def test_disjoint_fails(self):
        real = _agg(selection=[_sel(1, ["a", "b"])])
        sim = _agg(selection=[_sel(1, ["c", "d"])])
        r = pc.selection_parity(real, sim)
        assert not r["ok"] and r["mean_jaccard"] == 0.0


class TestStalenessParity:
    def test_matching_ok(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0]), _round(1, ["b"], [1])])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0]), _round(1, ["b"], [1])])
        r = pc.staleness_parity(real, sim)
        assert r["ok"] and r["real_mean"] == r["sim_mean"]

    def test_negative_staleness_fails(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0])])
        sim = _agg(agg_rounds=[_round(1, ["a"], [-1])])  # impossible in either mode
        r = pc.staleness_parity(real, sim)
        assert not r["ok"] and r["all_nonnegative"] is False


class TestSimSendTs:
    def test_real_null_sim_increasing_ok(self):
        real_tr = {"aa": {"task_recv": [{"round": 2, "sim_send_ts": None}]}}
        sim_tr = {"aa": {"task_recv": [
            {"round": 2, "sim_send_ts": 5.0}, {"round": 3, "sim_send_ts": 11.0}]}}
        assert pc.sim_send_ts_ok(real_tr, sim_tr)["ok"]

    def test_sim_null_fails(self):
        sim_tr = {"aa": {"task_recv": [{"round": 2, "sim_send_ts": None}]}}
        assert not pc.sim_send_ts_ok({}, sim_tr)["ok"]


class TestGpuBudget:
    def test_within_budget_ok(self):
        tr = {"aa": {"trainer_round": [
            {"real_gpu_time_s": 1.0, "training_budget_s": 5.0},
            {"real_gpu_time_s": 2.0, "training_budget_s": 5.0}]}}
        assert pc.gpu_budget_ok(tr)["ok"]

    def test_overrun_fails(self):
        tr = {"aa": {"trainer_round": [
            {"real_gpu_time_s": 9.0, "training_budget_s": 5.0},
            {"real_gpu_time_s": 8.0, "training_budget_s": 5.0}]}}
        r = pc.gpu_budget_ok(tr)
        assert not r["ok"] and r["mean_overrun_frac"] == 1.0


class TestSimInvariants:
    def test_commit_order_monotone(self):
        ok = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=5.0),
                              _round(1, ["b"], [0], vclock=10.0)])
        assert pc.sim_commit_order_monotone(ok)["ok"]
        bad = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=10.0),
                               _round(1, ["b"], [0], vclock=5.0)])
        assert not pc.sim_commit_order_monotone(bad)["ok"]

    def test_agg_goal_cycles(self):
        ok = _agg(agg_rounds=[_round(1, ["a"], [0], agg_goal_count=1),
                              _round(1, ["b"], [0], agg_goal_count=2)])
        assert pc.agg_goal_cycles_ok(ok, agg_goal=2)["ok"]
        bad = _agg(agg_rounds=[_round(1, ["a"], [0], agg_goal_count=3)])
        assert not pc.agg_goal_cycles_ok(bad, agg_goal=2)["ok"]


def test_run_all_parity_smoke():
    a = _agg(selection=[_sel(1, ["a", "b"])],
             agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])
    tr = {"aa": {"task_recv": [], "trainer_round": []}}
    res = pc.run_all_parity(a, a, tr, tr, agg_goal=2)
    assert all(v["ok"] for v in res.values())
