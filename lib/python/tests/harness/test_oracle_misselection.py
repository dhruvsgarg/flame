# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13: offline replay metrics (scripts/oracle_misselection.py)."""

import pytest

from examples.async_cifar10.scripts.oracle_misselection import _split_key, selection_metrics


def _sel(chosen, per_trainer):
    return {"chosen": chosen, "per_trainer": per_trainer}


def test_best_pick_scores_perfect():
    true_u = {"a": 3.0, "b": 2.0, "c": 1.0}
    pt = {t: {"utility": u, "in_all_selected": False} for t, u in true_u.items()}
    m = selection_metrics(_sel(["a"], pt), true_u)
    assert m["hit_rate"] == 1.0 and m["rank_pct"] == 1.0 and m["regret_rel"] == 0.0
    assert m["spearman"] == pytest.approx(1.0)


def test_busy_trainers_leave_the_pool():
    # 'a' is in flight (in_all_selected), so the best pickable is 'b'.
    true_u = {"a": 9.0, "b": 2.0, "c": 1.0}
    pt = {"a": {"utility": 9.0, "in_all_selected": True},
          "b": {"utility": 0.1, "in_all_selected": False},
          "c": {"utility": 5.0, "in_all_selected": False}}
    m = selection_metrics(_sel(["c"], pt), true_u)
    assert m["pool"] == 2 and m["hit_rate"] == 0.0 and m["rank_pct"] == 0.0
    assert m["regret_rel"] == pytest.approx(0.5)


def test_split_key_prefers_oracle_config_then_run_name():
    assert _split_key("x", {"oracle_utility_injection": {"alpha": 1, "num_trainers": 50}}) == (1.0, 50)
    assert _split_key("/e/run_1_htiny_cpu_dbg_felix_n300_alpha0.1_syn_0_stream_sim", {}) == (0.1, 300)
