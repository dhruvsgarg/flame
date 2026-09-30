# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D25: real's algorithmic clock started at its FIRST COMMIT, sim's vclock at dispatch (0). A run whose first
commit comes late (unaware oort: 4 chained 90s abandons, first commit at 364s) graded K8 time-to-N real 587s vs
sim 963s although the two commit timelines matched (pool_fixB_verify_s2_kaylee, gs oort mobiperf)."""

from parity.checks import terminal_state_parity


def _side(first_commit_s, n, step, vclock, t0=1000.0):
    rounds = []
    for i in range(n):
        t = first_commit_s + i * step
        e = {"round": i + 1, "ts": t0 + t, "contributing_trainers": [f"t{i % 3}"], "task_to_perform": "train"}
        if vclock:
            e["vclock_now"] = t
        rounds.append(e)
    return {"agg_rounds": rounds, "selection_train": [{"round": 1, "ts": t0}]}


def test_late_first_commit_times_from_run_start():
    real, sim = _side(364.0, 60, 10.0, vclock=False), _side(364.0, 60, 10.0, vclock=True)
    r = terminal_state_parity(real, sim)
    assert r["ok"], r
    assert r["real_time_to_n_s"] == r["sim_vclock_to_n_s"]
