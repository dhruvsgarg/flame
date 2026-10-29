# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Each event invariant passes on a clean synthetic run and fails on one planted violation."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parity import event_invariants as ev  # noqa: E402

T1, T2 = "trainer_a", "trainer_b"


def _write_run(tmp_path, agg, trainers, hp=None, optimizer="fedbuff", c=2, log="stopping run\n"):
    d = tmp_path / "run"
    (d / "telemetry").mkdir(parents=True)
    with open(d / "telemetry" / "aggregator_x.jsonl", "w") as f:
        for e in agg:
            f.write(json.dumps(e) + "\n")
    for tid, evs in trainers.items():
        with open(d / "telemetry" / f"trainer_{tid}.jsonl", "w") as f:
            for e in evs:
                f.write(json.dumps({"end_id": tid, **e}) + "\n")
    cfg = {"hyperparameters": {"aggGoal": 2, "time_mode": "simulated", "max_experiment_runtime_s": 20,
                               **(hp or {})},
           "selector": {"kwargs": {"c": c}}, "optimizer": {"sort": optimizer}}
    (d / "aggregator_config.json").write_text(json.dumps(cfg))
    (d / "x_aggregator.log").write_text(log)
    return str(d)


def _clean_run():
    """Two trainers, two async rounds of agg_goal=2, sim clock 0 -> 20."""
    agg, trainers, ts = [], {T1: [], T2: []}, 0.0
    vclock = 0.0
    for rnd in (1, 2):
        agg.append({"event": "selection", "task": "train", "ts": ts, "chosen": [T1, T2], "num_chosen": 2,
                    "in_flight": 2, "per_trainer": {T1: {"in_pending_commit": False, "avl_state": "AVL_TRAIN"},
                                                    T2: {"in_pending_commit": False, "avl_state": "AVL_TRAIN"}}})
        for t in (T1, T2):
            ts += 0.1
            agg.append({"event": "dispatch", "task": "train", "end_id": t, "ts": ts, "sim_send_ts": vclock,
                        "round": rnd})
            trainers[t] += [
                {"event": "task_recv", "round": rnd, "ts": ts + 0.01},
                {"event": "trainer_round", "round": rnd, "ts": ts + 0.02, "task_to_perform": "train",
                 "real_gpu_time_s": 0.01, "sim_round_duration_s": 5.0, "training_budget_s": 5.0, "model_samples": 100, "model_kbit": 1000.0,
                 "sim_send_ts": vclock, "sim_completion_ts": vclock + 5.0},
                {"event": "task_send", "round": rnd, "ts": ts + 0.03, "task_to_perform": "train"},
            ]
        for i, t in enumerate((T1, T2)):
            ts += 0.1
            vclock += 5.0 if i == 0 else 5.0
            agg.append({"event": "agg_round", "task_to_perform": "train", "ts": ts, "round": rnd,
                        "contributing_trainers": [t], "staleness": [0], "vclock_now": vclock,
                        "commit_gap_s": 0.0})
    return agg, trainers


@pytest.fixture(autouse=True)
def _no_registry(monkeypatch):
    monkeypatch.setattr(ev, "_registry_devices", lambda: {T1: (10.0, 1000.0), T2: (10.0, 1000.0)})


def _status(run_dir, name):
    return ev.check_run(run_dir)["checks"][name]["status"]


def test_clean_run_passes(tmp_path):
    r = ev.check_run(_write_run(tmp_path, *_clean_run()))
    assert r["passed"], {k: v for k, v in r["checks"].items() if v["status"] not in ("PASS", "SKIP")}


def test_ev0_missing_stop_fails(tmp_path):
    assert _status(_write_run(tmp_path, *_clean_run(), log="no stop\n"), "EV0_clean_exit") == "FAIL"


def test_ev2_double_recv_fails(tmp_path):
    agg, tr = _clean_run()
    tr[T1].insert(1, {"event": "task_recv", "round": 9, "ts": tr[T1][0]["ts"] + 0.001})
    assert _status(_write_run(tmp_path, agg, tr), "EV2_task_alternation") == "FAIL"


def test_ev3_registry_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_registry_devices", lambda: {T1: (30.0, 1000.0), T2: (10.0, 1000.0)})
    assert _status(_write_run(tmp_path, *_clean_run()), "EV3_duration_model") == "FAIL"


def test_ev3_delay_factor_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_registry_devices", lambda: {T1: (60.0, 1000.0), T2: (60.0, 1000.0)})
    run = _write_run(tmp_path, *_clean_run(), hp={"trainingDelayFactor": 4.0})
    assert _status(run, "EV3_duration_model") == "PASS"


def test_ev3_utility_pass_charge_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_registry_devices", lambda: {T1: (10.0, 1000.0), T2: (10.0, 1000.0)})
    agg, tr = _clean_run()
    for evs in tr.values():
        for e in evs:
            if e["event"] == "trainer_round":  # FX-N109 I9: budget = D + D/20, reported apart
                e.update(training_budget_s=5.25, sim_round_duration_s=5.25, utility_pass_s=0.25,
                         sim_completion_ts=e["sim_send_ts"] + 5.25)
    assert _status(_write_run(tmp_path, agg, tr), "EV3_duration_model") == "PASS"


@pytest.mark.parametrize("val_pass, status", [(0.1, "PASS"), (None, "FAIL")])
def test_ev3_held_out_pass_charge(tmp_path, monkeypatch, val_pass, status):
    # FX-N127 P5: the held-out report's forward pass is charged and reported apart; unreported it is a D mismatch
    monkeypatch.setattr(ev, "_registry_devices", lambda: {T1: (10.0, 1000.0), T2: (10.0, 1000.0)})
    agg, tr = _clean_run()
    for evs in tr.values():
        for e in evs:
            if e["event"] == "trainer_round":
                e.update(training_budget_s=5.1, sim_round_duration_s=5.1, val_pass_s=val_pass, sim_completion_ts=e["sim_send_ts"] + 5.1)
    assert _status(_write_run(tmp_path, agg, tr), "EV3_duration_model") == status


def test_ev5_commit_without_send_fails(tmp_path):
    agg, tr = _clean_run()
    tr[T2] = [e for e in tr[T2] if e["event"] != "task_send"]
    assert _status(_write_run(tmp_path, agg, tr), "EV5_commit_accounting") == "FAIL"


def test_ev7_short_round_fails(tmp_path):
    agg, tr = _clean_run()
    first_commit = next(i for i, e in enumerate(agg) if e["event"] == "agg_round")
    del agg[first_commit]
    assert _status(_write_run(tmp_path, agg, tr), "EV7_agg_goal_cadence") == "FAIL"


def test_ev7_dynamic_k_round_passes(tmp_path):
    agg, tr = _clean_run()
    first_commit = next(i for i, e in enumerate(agg) if e["event"] == "agg_round")
    del agg[first_commit]
    for e in agg:
        if e["event"] == "agg_round" and e.get("round") == 1:
            e["agg_goal"] = 1  # FX-N109 I5: the controller set K = 1 for round 1
    assert _status(_write_run(tmp_path, agg, tr), "EV7_agg_goal_cadence") == "PASS"


def test_ev8_choosing_over_cap_fails(tmp_path):
    agg, tr = _clean_run()
    agg[0]["in_flight"] = 3
    assert _status(_write_run(tmp_path, agg, tr), "EV8_concurrency_cap") == "FAIL"


def test_ev8_held_over_cap_choosing_zero_passes(tmp_path):
    agg, tr = _clean_run()
    agg.insert(1, {"event": "selection", "task": "train", "ts": 0.05, "chosen": [], "num_chosen": 0,
                   "in_flight": 5})
    assert _status(_write_run(tmp_path, agg, tr), "EV8_concurrency_cap") == "PASS"


def test_ev9_chosen_while_pending_fails(tmp_path):
    agg, tr = _clean_run()
    agg[0]["per_trainer"][T1]["in_pending_commit"] = True
    assert _status(_write_run(tmp_path, agg, tr), "EV9_selector_state") == "FAIL"


def test_ev9_carried_unavail_straggler_passes(tmp_path):
    agg, tr = _clean_run()
    agg[0]["per_trainer"][T2]["avl_state"] = "UN_AVL"
    agg[0].update(exploit_ids=[T1], explore_ids=[])  # T2 is carried in flight, not a new pick
    run = _write_run(tmp_path, agg, tr, hp={"avail_select_filter": True})
    assert _status(run, "EV9_selector_state") == "PASS"
    agg[0]["exploit_ids"] = [T1, T2]
    assert _status(_write_run(tmp_path / "b", agg, tr, hp={"avail_select_filter": True}),
                   "EV9_selector_state") == "FAIL"


def test_ev10_redispatch_while_outstanding_fails(tmp_path):
    agg, tr = _clean_run()
    d = next(e for e in agg if e["event"] == "dispatch")
    agg.append({**d, "ts": d["ts"] + 0.001})
    assert _status(_write_run(tmp_path, agg, tr), "EV10_dispatch_one_in_flight") == "FAIL"


def test_ev10_redispatch_after_withhold_passes(tmp_path):
    agg, tr = _clean_run()
    d = next(e for e in agg if e["event"] == "dispatch")
    agg += [{**d, "ts": d["ts"] + 0.001},
            {"event": "withheld_delivery", "end_id": d["end_id"], "ts": d["ts"] + 0.002}]
    assert _status(_write_run(tmp_path, agg, tr), "EV10_dispatch_one_in_flight") == "PASS"


def test_ev24_dispatch_while_computing_fails_unless_abandoned(tmp_path):
    agg, tr = _clean_run()
    assert _status(_write_run(tmp_path / "a", agg, tr), "EV24_busy_until_completion") == "PASS"
    r = next(e for e in tr[T1] if e["event"] == "trainer_round")
    early = {**r, "sim_send_ts": r["sim_completion_ts"] - 1.0, "sim_completion_ts": r["sim_completion_ts"] + 5.0}
    assert _status(_write_run(tmp_path / "b", agg, {**tr, T1: tr[T1] + [early]}), "EV24_busy_until_completion") == "FAIL"
    ab = {"event": "abandon_timeout", "end_id": T1, "sim_send_ts": r["sim_send_ts"], "vclock_now": r["sim_send_ts"] + 1}
    assert _status(_write_run(tmp_path / "c", agg + [ab], {**tr, T1: tr[T1] + [early]}), "EV24_busy_until_completion") == "PASS"


def test_ev11_clock_backwards_fails(tmp_path):
    agg, tr = _clean_run()
    commits = [e for e in agg if e["event"] == "agg_round"]
    commits[-1]["vclock_now"] = 1.0
    assert _status(_write_run(tmp_path, agg, tr), "EV11_vclock") == "FAIL"


def test_ev11_sub_slack_pastdating_fails(tmp_path):
    agg, tr = _clean_run()
    next(e for e in agg if e["event"] == "agg_round")["commit_gap_s"] = 0.5  # FX-D23: below the old 2s slack
    assert _status(_write_run(tmp_path, agg, tr), "EV11_vclock") == "FAIL"


def test_ev1_counts_rounds_not_updates(tmp_path):
    # Two rounds pass at any agg_goal (old 2 x agg_goal failed oort syn_50_3st at 8 rounds, K=5, on both sides).
    agg, tr = _clean_run()
    assert _status(_write_run(tmp_path, agg, tr, hp={"aggGoal": 5}), "EV1_progress") == "PASS"
    one = [e for e in agg if e["event"] != "agg_round"] + [next(e for e in agg if e["event"] == "agg_round")]
    assert _status(_write_run(tmp_path / "b", one, tr), "EV1_progress") == "FAIL"


def test_ev12_short_run_fails(tmp_path):
    run = _write_run(tmp_path, *_clean_run(), hp={"max_experiment_runtime_s": 1000})
    assert _status(run, "EV12_reached_budget") == "FAIL"


def test_ev14_nan_loss_fails(tmp_path):
    agg, tr = _clean_run()
    agg.append({"event": "agg_eval", "ts": 99, "round": 2, "test-accuracy": 0.1, "test-loss": float("nan")})
    assert _status(_write_run(tmp_path, agg, tr), "EV14_eval_sane") == "FAIL"


def test_ev14_zero_accuracy_passes(tmp_path):
    agg, tr = _clean_run()
    agg.append({"event": "agg_eval", "ts": 99, "round": 2, "test-accuracy": 0.0, "test-loss": 0.0})
    assert _status(_write_run(tmp_path, agg, tr), "EV14_eval_sane") == "PASS"


def test_broken_check_is_error_not_crash(tmp_path, monkeypatch):
    def boom(run):
        raise RuntimeError("x")
    monkeypatch.setattr(ev, "CHECKS", [boom])
    r = ev.check_run(_write_run(tmp_path, *_clean_run()))
    assert r["checks"]["boom"]["status"] == "ERROR" and not r["passed"]


@pytest.mark.parametrize("hp,want", [({}, "FAIL"),
                                     ({"trackTrainerAvail": {"trace": "syn_50_3st"}}, "WARN"),
                                     ({"client_notify": {"trace": "mobiperf_3st"}}, "WARN")])
def test_ev13_stall_is_warn_off_syn_0_for_either_trace_key(tmp_path, hp, want):
    agg, tr = _clean_run()
    [e for e in agg if e["event"] == "agg_round"][-1]["ts"] += 1000
    assert _status(_write_run(tmp_path, agg, tr, hp=hp), "EV13_no_stall") == want


def test_ev15_duplicate_send_fails(tmp_path):
    agg, tr = _clean_run()
    tr[T1].append({"event": "task_send", "round": 1, "ts": 99, "task_to_perform": "train"})
    assert _status(_write_run(tmp_path, agg, tr), "EV15_one_task_per_version") == "FAIL"


def test_ev15_same_version_redispatch_needs_timeout_and_policy(tmp_path):
    agg, tr = _clean_run()
    d = [e for e in agg if e["event"] == "dispatch"][-1]
    agg.append({"event": "abandon_timeout", "end_id": d["end_id"], "reason": "abandon_timeout_vclock", "ts": 50})
    agg.append({**d, "ts": 51})
    assert _status(_write_run(tmp_path, agg, tr), "EV15_one_task_per_version") == "FAIL"
    run = _write_run(tmp_path / "b", agg, tr, hp={"task_retry_policy": "fixed"})
    assert _status(run, "EV15_one_task_per_version") == "PASS"


def test_ev15_eval_after_train_same_version_fails_train_after_eval_passes(tmp_path):
    agg, tr = _clean_run()
    d = [e for e in agg if e["event"] == "dispatch" and e.get("task") == "train"][-1]
    bad = agg + [{**d, "task": "eval", "ts": 60}]
    assert _status(_write_run(tmp_path, bad, tr), "EV15_one_task_per_version") == "FAIL"
    first = next(e for e in agg if e["event"] == "dispatch" and e.get("task") == "train")
    ok = [{**first, "task": "eval", "ts": first["ts"] - 0.05}] + agg  # eval first, then train, same version
    assert _status(_write_run(tmp_path / "b", ok, tr), "EV15_one_task_per_version") == "PASS"


def test_ev15_trainer_eval_then_train_same_version_passes(tmp_path):
    agg, tr = _clean_run()
    tr[T1].insert(0, {"event": "task_send", "round": 1, "ts": 0.0, "task_to_perform": "eval"})
    assert _status(_write_run(tmp_path, agg, tr), "EV15_one_task_per_version") == "PASS"
    tr[T1].append({"event": "task_send", "round": 1, "ts": 99, "task_to_perform": "eval"})
    assert _status(_write_run(tmp_path / "b", agg, tr), "EV15_one_task_per_version") == "FAIL"


def test_ev15_duplicate_commit_fails(tmp_path):
    agg, tr = _clean_run()
    c = next(e for e in agg if e["event"] == "agg_round")
    agg.append({**c, "ts": 98, "round": c["round"] + 1, "staleness": [1]})  # same (end, version)
    assert _status(_write_run(tmp_path, agg, tr), "EV15_one_task_per_version") == "FAIL"


def test_ev7_sync_empty_round_fails(tmp_path):
    agg, tr = _clean_run()
    c = [e for e in agg if e["event"] == "agg_round"][-1]
    agg.append({**c, "ts": 99, "round": 3, "contributing_trainers": [], "staleness": []})
    assert _status(_write_run(tmp_path, agg, tr, optimizer="fedavg"), "EV7_agg_goal_cadence") == "FAIL"


def _sync_run(contrib_per_round):
    agg, tr = _clean_run()
    agg = [e for e in agg if e["event"] != "agg_round"]
    for rnd, contrib in enumerate(contrib_per_round, 1):
        agg.append({"event": "agg_round", "task_to_perform": "train", "ts": 5.0 + rnd, "round": rnd,
                    "contributing_trainers": contrib, "staleness": [0] * len(contrib), "vclock_now": 10.0 * rnd})
    return agg, tr


@pytest.mark.parametrize("contrib,status", [([[T1, T2], [T1, T2]], "PASS"), ([[T1, T2], [T1]], "FAIL")])
def test_ev7_sync_wait_k_commits_k(tmp_path, contrib, status):
    # FX-N37: with syncWaitForK a version never commits fewer than agg_goal updates.
    agg, tr = _sync_run(contrib)
    run = _write_run(tmp_path, agg, tr, optimizer="fedavg", hp={"syncWaitForK": True})
    assert _status(run, "EV7_agg_goal_cadence") == status


def test_ev5_sends_after_last_commit_are_not_lost(tmp_path):
    agg, tr = _clean_run()
    for r in (8, 9):  # two uploads cut off by the stop
        tr[T1].append({"event": "task_send", "round": r, "ts": 1e9 + r, "task_to_perform": "train"})
    assert _status(_write_run(tmp_path, agg, tr), "EV5_commit_accounting") == "PASS"


def test_ev12_trace_starved_run_reached_budget_passes(tmp_path):
    # FX-D12: a faithful starvation keeps selecting on the clock; the budget is consumed.
    agg, tr = _clean_run()
    agg.append({"event": "selection", "task": "train", "ts": 9.0, "chosen": [], "num_chosen": 0,
                "vclock_now": 100.0})
    run = _write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 100})
    assert _status(run, "EV12_reached_budget") == "PASS"


def test_ev12_starvation_jump_to_budget_passes(tmp_path):
    # FX-N31: a sync sim starved with no selection jumps the vclock to budget; run_end records it.
    agg, tr = _clean_run()
    agg.append({"event": "run_end", "ts": 9.0, "round": 2, "work_done": True, "vclock_now": 100.0})
    run = _write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 100})
    assert _status(run, "EV12_reached_budget") == "PASS"


def test_ev12_real_budget_stop_after_last_commit_passes(tmp_path):
    # FX-D21: P3 oort real commits until 199s and stops on its 240s wall budget; run_end marks the stop.
    agg, tr = _clean_run()
    t0 = min(e["ts"] for e in agg if e.get("event") == "selection")
    real = {"time_mode": "real", "max_experiment_runtime_s": 1000}
    assert _status(_write_run(tmp_path / "a", agg, tr, hp=real), "EV12_reached_budget") == "FAIL"
    agg.append({"event": "run_end", "ts": t0 + 1000.0, "round": 2, "work_done": True, "vclock_now": None})
    assert _status(_write_run(tmp_path / "b", agg, tr, hp=real), "EV12_reached_budget") == "PASS"


def test_ev3_frozen_trainer_clock_fails(tmp_path):
    # S1 injected bug freeze_trainer_clock: the trainer keeps its first stamp.
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round" and e["round"] == 2:
            e["sim_send_ts"], e["sim_completion_ts"] = 0.0, 5.0
    assert _status(_write_run(tmp_path, agg, tr), "EV3_duration_model") == "FAIL"


def test_ev3_reported_delivery_lag_is_not_a_leg(tmp_path):
    # FX-D108: a per-recipient delivery lag moves sct; reported, EV3 still holds; unreported, it is caught.
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round" and e["round"] == 2:
            e["sim_completion_ts"] += 0.7
    assert _status(_write_run(tmp_path / "a", agg, tr), "EV3_duration_model") == "FAIL"
    for e in tr[T1]:
        if e["event"] == "trainer_round" and e["round"] == 2:
            e["sim_delivery_lag_s"] = 0.7
    assert _status(_write_run(tmp_path / "b", agg, tr), "EV3_duration_model") == "PASS"


def _unavail_T1(monkeypatch):
    from sortedcontainers import SortedDict
    monkeypatch.setattr(ev, "_ground_truth", lambda run: {T1: SortedDict({0.0: "UN_AVL", 50.0: "AVL_TRAIN"}),
                                                          T2: SortedDict()})


def test_ev16_unavailable_commit_without_withhold_fails(tmp_path, monkeypatch):
    # S1 injected bug order_by_sct: T1 is UN_AVL at its sct yet committed with no withheld delivery.
    _unavail_T1(monkeypatch)
    assert _status(_write_run(tmp_path, *_clean_run()), "EV16_withheld_delivery") == "FAIL"


def test_ev16_withheld_at_next_avail_passes(tmp_path, monkeypatch):
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 50.0})
    assert _status(_write_run(tmp_path, agg, tr), "EV16_withheld_delivery") == "PASS"


def test_ev16_late_withheld_delivery_fails(tmp_path, monkeypatch):
    # FX-N54: a due delivery the sim starved past commits long after delivery_ts.
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 500.0})
    assert _status(_write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 1000}), "EV16_withheld_delivery") == "FAIL"


def test_ev16_delivery_at_budget_commits_at_final_round_passes(tmp_path, monkeypatch):
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 500.0})
    assert _status(_write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 50}), "EV16_withheld_delivery") == "PASS"


def test_ev16_sync_delivery_awaiting_unlogged_close_passes(tmp_path, monkeypatch):
    # FX-D136: a sync delivery joins the open round; its close is not logged yet in a live prefix or a cut final round.
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 500.0})
    run = _write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 1000}, optimizer="oort")
    assert _status(run, "EV16_withheld_delivery") == "PASS"


@pytest.mark.parametrize("closes, status", [((50.0, 600.0), "PASS"), ((60.0, 600.0), "FAIL")])
def test_ev16_sync_delivery_due_at_a_close_joins_the_next(tmp_path, monkeypatch, closes, status):
    # F42-T12 (b): a close at the delivery's instant took its K first; the delivery commits in the next version.
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 500.0})
    agg += [{"event": "agg_round", "task_to_perform": "train", "ts": 10 + i, "round": 3 + i, "contributing_trainers": [T2],
             "staleness": [0], "vclock_now": c, "commit_gap_s": 0.0} for i, c in enumerate(closes)]
    run = _write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 1000}, optimizer="oort")
    assert _status(run, "EV16_withheld_delivery") == status


@pytest.mark.parametrize("t2_sct, status", [(20.0, "PASS"), (55.0, "FAIL")])
def test_ev16_sync_round_cut_before_delivery_joins_the_next(tmp_path, monkeypatch, t2_sct, status):
    # F42g REFL speech: a round whose last arrival (cut) precedes the delivery closes later but can't take it.
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 500.0})
    tr[T2].append({"event": "trainer_round", "ts": 9, "round": 3, "sim_completion_ts": t2_sct})
    agg += [{"event": "agg_round", "task_to_perform": "train", "ts": 10 + i, "round": 3 + i, "contributing_trainers": [T2],
             "staleness": [i], "vclock_now": c, "commit_gap_s": 0.0} for i, c in enumerate((60.0, 600.0))]
    run = _write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 1000}, optimizer="oort")
    assert _status(run, "EV16_withheld_delivery") == status


def test_ev16_dropped_stale_sync_delivery_is_not_a_late_commit(tmp_path, monkeypatch):
    # F42-T12: a stale withheld delivery a no-stale sync stack drops (FX-N57) never commits, however late it is ingested.
    _unavail_T1(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 50.0})
    agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": 30.0, "round": 8, "staleness": 5,
                "delivery_ts": 50.0, "actual_commit_ts": 500.0})
    agg.append({"event": "agg_round", "task_to_perform": "train", "ts": 10, "round": 8, "contributing_trainers": [T2],
                "staleness": [0], "vclock_now": 60.0, "commit_gap_s": 0.0})
    run = _write_run(tmp_path, agg, tr, hp={"max_experiment_runtime_s": 1000}, optimizer="oort")
    assert _status(run, "EV16_withheld_delivery") == "PASS"


def _gated_real_run(tmp_path, dispatch_ends):
    # T1's update sits behind its send-gate over ts 10-20; a sync selection at 15 still lists it in flight.
    agg = [{"event": "selection", "task": "train", "ts": 15.0, "chosen": [T1, T2]}]
    agg += [{"event": "dispatch", "task": "train", "end_id": t, "ts": 15.0, "time_mode": "real"} for t in dispatch_ends]
    tr = {T1: [{"event": "task_send", "ts": 20.0, "send_gate_wait_s": 10.0}], T2: []}
    return _write_run(tmp_path, agg, tr, hp={"time_mode": "real"})


def test_ev17_carried_inflight_pick_passes(tmp_path):
    assert _status(_gated_real_run(tmp_path, [T2]), "EV17_real_gate_repick") == "PASS"


def test_ev17_redispatch_while_gated_fails(tmp_path):
    assert _status(_gated_real_run(tmp_path, [T1, T2]), "EV17_real_gate_repick") == "FAIL"


def test_ev17_without_dispatch_events_reads_selections(tmp_path):
    assert _status(_gated_real_run(tmp_path, []), "EV17_real_gate_repick") == "FAIL"


@pytest.mark.parametrize("abandon_lag, want", [(0.0, "PASS"), (61.0, "FAIL")])
def test_ev20_abandon_on_time(tmp_path, abandon_lag, want):
    # FX-D155: T2 dispatched 2 ms after T1; real abandoned it 61 s late.
    agg = [{"event": "dispatch", "task": "train", "end_id": T1, "round": 15, "ts": 0.0},
           {"event": "dispatch", "task": "train", "end_id": T2, "round": 15, "ts": 0.002},
           {"event": "abandon_timeout", "end_id": T1, "round": 15, "ts": 90.0},
           {"event": "abandon_timeout", "end_id": T2, "round": 15, "ts": 90.002 + abandon_lag},
           {"event": "agg_round", "round": 15, "ts": 200.0}]
    run = _write_run(tmp_path, agg, {T1: [], T2: []}, hp={"time_mode": "real"}, optimizer="fedavg")
    assert _status(run, "EV20_abandon_on_time") == want


def test_ev18_model_finite(tmp_path):
    agg, tr = _clean_run()
    agg += [{"event": "model_health", "round": r, "weight_norm": 10.0, "nonfinite": 0, "bn_buf_max": 2.0} for r in (1, 2)]
    assert _status(_write_run(tmp_path, agg, tr), "EV18_model_finite") == "PASS"
    agg.append({"event": "model_health", "round": 3, "weight_norm": 10.0, "nonfinite": 4, "bn_buf_max": 2.0})
    assert _status(_write_run(tmp_path / "b", agg, tr), "EV18_model_finite") == "FAIL"


def test_ev18_negative_running_var_fails(tmp_path):
    agg, tr = _clean_run()
    agg.append({"event": "model_health", "round": 1, "weight_norm": 1.0, "nonfinite": 0, "bn_buf_max": 2.0, "bn_var_min": -0.005})
    assert _status(_write_run(tmp_path, agg, tr), "EV18_model_finite") == "FAIL"


_STREAM = {"data_streaming": {"enabled": "True", "initial_frac": 0.1, "full_data_available_after_s": 20}}


def _streamed(plant=None):
    """_clean_run's trainer tasks carrying the schedule's own visible count (100 samples)."""
    agg, trainers = _clean_run()
    for evs in trainers.values():
        for e in evs:
            if e["event"] == "trainer_round":
                t = e["sim_send_ts"]
                e.update(stream_clock_s=t, total_samples=100, visible_samples=int(10 + 90 * t / 20))
    if plant:
        plant(trainers)
    return agg, trainers


def test_ev19_stream_schedule(tmp_path):
    assert _status(_write_run(tmp_path / "off", *_clean_run()), "EV19_stream_schedule") == "SKIP"
    assert _status(_write_run(tmp_path / "ok", *_streamed(), hp=_STREAM), "EV19_stream_schedule") == "PASS"

    def off_schedule(tr):
        [e for e in tr[T1] if e["event"] == "trainer_round"][1]["visible_samples"] = 99

    def clock_drift(tr):
        [e for e in tr[T2] if e["event"] == "trainer_round"][0]["stream_clock_s"] = 0.5

    for name, plant in (("sched", off_schedule), ("clock", clock_drift)):
        assert _status(_write_run(tmp_path / name, *_streamed(plant), hp=_STREAM), "EV19_stream_schedule") == "FAIL"


@pytest.mark.parametrize("logged_lag, want", [(0.3, "PASS"), (0.0, "FAIL")])
def test_ev19_sim_stream_clock_at_train_start(tmp_path, logged_lag, want):
    # FX-D160: sim reads the stream at send + delivery lag; a clock past send by an unlogged lag still fails.
    def lagged(tr):
        for evs in tr.values():
            for e in evs:
                if e["event"] == "trainer_round":
                    t = e["sim_send_ts"] + 0.3
                    e.update(stream_clock_s=t, visible_samples=int(10 + 90 * t / 20), sim_delivery_lag_s=logged_lag)
    assert _status(_write_run(tmp_path, *_streamed(lagged), hp=_STREAM), "EV19_stream_schedule") == want


def test_ev3_sim_duration_includes_measured_overhead(tmp_path):
    # C13 (FX-N73): sim duration = max(gpu, D) + the trainer's measured overhead; sct moves with it.
    agg, tr = _clean_run()
    for evs in tr.values():
        for e in evs:
            if e["event"] == "trainer_round":
                e.update(trainer_overhead_s=0.3, sim_round_duration_s=5.3, sim_completion_ts=e["sim_send_ts"] + 5.3)
    assert _status(_write_run(tmp_path, agg, tr), "EV3_duration_model") == "PASS"
    for evs in tr.values():
        for e in evs:
            if e["event"] == "trainer_round":
                e["trainer_overhead_s"] = 0.0
    assert _status(_write_run(tmp_path / "b", agg, tr), "EV3_duration_model") == "FAIL"


def test_doomed_only_prefix_safe_fails(tmp_path):
    # A live prefix (no stop line, short of budget) is not doomed; a NaN eval is; EV11 pastdating alone is not.
    agg, tr = _clean_run()
    assert ev.doomed(_write_run(tmp_path / "a", agg, tr, hp={"max_experiment_runtime_s": 1000}, log="")) == []
    nan = agg + [{"event": "agg_eval", "ts": 99, "round": 2, "test-accuracy": 0.1, "test-loss": float("nan")}]
    assert [d.split(":")[0] for d in ev.doomed(_write_run(tmp_path / "b", nan, tr))] == ["EV14"]
    next(e for e in agg if e["event"] == "agg_round")["commit_gap_s"] = 0.5
    assert ev.doomed(_write_run(tmp_path / "c", agg, tr)) == []


def test_doomed_skips_ev23_pick_count(monkeypatch):
    # A live prefix can log a selection before its `picked` check-in (B2 1027_c3 fedbuff; the full log PASSes).
    cfg = {"hyperparameters": {"avail_knowledge": "checkin", "time_mode": "simulated"}}
    ck = lambda k, t: {"event": "checkin", "kind": k, "end_id": "a", "t": t}
    pick = {"event": "selection", "task": "train", "round": 1, "chosen": ["a"]}
    runs = {"short": [ck("pooled", 10), pick], "break": [ck("picked", 20), pick]}
    monkeypatch.setattr(ev, "EARLY", (ev.ev23_checkin_protocol,))
    monkeypatch.setattr(ev, "_ground_truth", lambda r: None)
    monkeypatch.setattr(ev, "load_run", lambda d: {"dir": d, "cfg": cfg, "agg": runs[d]})
    assert ev.doomed("short") == []
    assert [d.split(":")[0] for d in ev.doomed("break")] == ["EV23"]


def _eval_trace(monkeypatch):
    # T1: TRAIN -> EVAL [0,50) -> TRAIN; T2 always TRAIN.
    from sortedcontainers import SortedDict
    monkeypatch.setattr(ev, "_ground_truth", lambda run: {T1: SortedDict({0.0: "AVL_EVAL", 50.0: "AVL_TRAIN"}),
                                                          T2: SortedDict({0.0: "AVL_TRAIN"})})


def test_ev16_train_commit_in_avl_eval_unheld_fails(tmp_path, monkeypatch):
    # FX-N97: AVL_EVAL is not deliverable for a train update.
    _eval_trace(monkeypatch)
    assert _status(_write_run(tmp_path, *_clean_run()), "EV16_withheld_delivery") == "FAIL"


def test_ev16_train_withheld_to_avl_train_passes(tmp_path, monkeypatch):
    _eval_trace(monkeypatch)
    agg, tr = _clean_run()
    for e in tr[T1]:
        if e["event"] == "trainer_round":
            agg.append({"event": "withheld_delivery", "ts": 9, "end_id": T1, "sct": e["sim_completion_ts"],
                        "delivery_ts": 50.0, "actual_commit_ts": 50.0, "task": "train"})
    assert _status(_write_run(tmp_path / "a", agg, tr), "EV16_withheld_delivery") == "PASS"


def _ev21_run(tmp_path, monkeypatch, chosen_state="AVL_TRAIN", evals=(), hp=None, sends=(), cfg_eval=0.0, done=True):
    _eval_trace(monkeypatch)
    agg = [{"event": "selection", "task": "train", "ts": 1.0, "round": 1, "chosen": [T2], "per_trainer": {
        T2: {"avl_state": chosen_state}}}]
    agg += [{"event": "selection", "task": "eval", "ts": 2.0, "round": 1, "chosen": [T1],
             "per_trainer": {T1: {"avl_state": st}}} for st in evals]
    agg += [{"event": "run_end", "ts": 3.0}] * done
    tr = {T1: [{"event": "task_send", "task_to_perform": "train", "send_gate_sct": sct, "send_gate_wait_s": w}
               for sct, w in sends], T2: []}
    run = _write_run(tmp_path, agg, tr, hp={"time_mode": "real", **(hp or {})})
    cfg = json.load(open(os.path.join(run, "aggregator_config.json")))
    cfg["selector"]["kwargs"]["evalGoalFactor"] = cfg_eval
    json.dump(cfg, open(os.path.join(run, "aggregator_config.json"), "w"))
    return run


def test_ev21_aware_train_pick_outside_avl_train_fails(tmp_path, monkeypatch):
    assert _status(_ev21_run(tmp_path / "a", monkeypatch), "EV21_avl_eval_semantics") == "PASS"
    run = _ev21_run(tmp_path / "b", monkeypatch, chosen_state="AVL_EVAL")
    assert _status(run, "EV21_avl_eval_semantics") == "FAIL"
    run = _ev21_run(tmp_path / "c", monkeypatch, chosen_state="AVL_EVAL", hp={"avail_select_filter": "False"})
    assert _status(run, "EV21_avl_eval_semantics") == "PASS"  # unaware picks anyone (its algorithm)


def test_ev21_relisted_held_pick_is_not_a_dispatch(tmp_path, monkeypatch):
    # gs T1 oort_star syn_50_3st: a top-up selection re-lists round 56's held pick after it went UN_AVL
    run = _ev21_run(tmp_path, monkeypatch)
    agg = os.path.join(run, "telemetry", [f for f in os.listdir(os.path.join(run, "telemetry")) if f.startswith("aggregator")][0])
    ev = [json.loads(l) for l in open(agg)]
    ev.insert(1, {"event": "selection", "task": "train", "ts": 1.5, "round": 1, "chosen": [T2], "per_trainer": {T2: {"avl_state": "UN_AVL"}}})
    open(agg, "w").write("".join(json.dumps(e) + "\n" for e in ev))
    assert _status(run, "EV21_avl_eval_semantics") == "PASS"
    ev[1]["round"] = 2  # PR36 oort_star: async re-lists an in-flight/withheld pick in later rounds
    open(agg, "w").write("".join(json.dumps(e) + "\n" for e in ev))
    assert _status(run, "EV21_avl_eval_semantics") == "PASS"
    for free in ({"event": "agg_round", "ts": 1.2, "round": 1, "contributing_trainers": [T2]},
                 {"event": "abandon_timeout", "ts": 1.2, "end_id": T2}):
        open(agg, "w").write("".join(json.dumps(e) + "\n" for e in ev[:1] + [free] + ev[1:]))
        assert _status(run, "EV21_avl_eval_semantics") == "FAIL"  # re-picked after its slot freed: a dispatch


@pytest.mark.parametrize("sct, wait, want", [(60.0, 0.0, "PASS"), (10.0, 0.0, "FAIL"), (10.0, 40.0, "PASS")])
def test_ev21_real_train_send_needs_avl_train(tmp_path, monkeypatch, sct, wait, want):
    run = _ev21_run(tmp_path, monkeypatch, sends=[(sct, wait)])
    assert _status(run, "EV21_avl_eval_semantics") == want


def test_ev21_eval_tasks_are_felix_only_and_reach_avl_eval(tmp_path, monkeypatch):
    felix = dict(cfg_eval=1.0)
    assert _status(_ev21_run(tmp_path / "a", monkeypatch, evals=["AVL_EVAL"] * 25, **felix), "EV21_avl_eval_semantics") == "PASS"
    assert _status(_ev21_run(tmp_path / "b", monkeypatch, evals=["AVL_EVAL"] * 25), "EV21_avl_eval_semantics") == "FAIL"  # not Felix
    assert _status(_ev21_run(tmp_path / "c", monkeypatch, evals=["UN_AVL"], **felix), "EV21_avl_eval_semantics") == "FAIL"
    assert _status(_ev21_run(tmp_path / "d", monkeypatch, evals=["AVL_TRAIN"] * 25, **felix), "EV21_avl_eval_semantics") == "FAIL"  # never AVL_EVAL
    assert _status(_ev21_run(tmp_path / "e", monkeypatch, evals=["AVL_TRAIN"] * 25, done=False, **felix), "EV21_avl_eval_semantics") == "PASS"  # live prefix


def _ev22(b, hp, agg=(), sel=None, phase="F42_s50u_ctl"):
    cfg = {"hyperparameters": {"simUnavailability": True, **hp}, "selector": {"kwargs": sel or {}}}
    return ev.ev22_availability_knowledge({"dir": f"/x/run_1_{phase}_dbg_{b}_n75_alpha0.1_syn_50_3st_sim", "cfg": cfg,
                                           "agg": list(agg)})


def test_ev22_r27_knowledge_per_baseline():
    assert _ev22("felix", {"proactive_inflight_evict": "True"})["status"] == "PASS"
    assert _ev22("felix", {"avail_knowledge": "checkin"}, phase="F42_s50u_fck")["status"] == "PASS"  # K1 ablation
    assert _ev22("felix", {"avail_knowledge": "checkin"})["status"] == "FAIL"
    assert _ev22("oort_star", {"avail_select_filter": "True", "proactive_inflight_evict": "False"})["status"] == "PASS"
    assert _ev22("refl", {"avail_select_filter": "True"})["status"] == "FAIL"  # F41's REFL: current-state filter
    assert _ev22("refl", {"avail_knowledge": "checkin"}, sel={"avail_predictor": "prophet"})["status"] == "PASS"
    assert _ev22("refl", {"avail_knowledge": "oracle"}, sel={"avail_predictor": "oracle"})["status"] == "PASS"  # REFL*
    assert _ev22("feddance", {"avail_knowledge": "none", "avail_select_filter": "False"})["status"] == "FAIL"
    assert _ev22("feddance", {"avail_knowledge": "none"}, phase="F42_s50u_fdblind")["status"] == "PASS"
    assert _ev22("feddance", {"avail_knowledge": "oracle"}, phase="F42_s50u_fdstar")["status"] == "PASS"  # FedDance*
    assert _ev22("fedbuff", {"avail_knowledge": "oracle"}, phase="F42_s50u_fbstar")["status"] == "PASS"  # FedBuff*
    assert _ev22("fedbuff", {"avail_knowledge": "oracle"})["status"] == "FAIL"
    assert _ev22("fedbuff", {"avail_knowledge": "oracle"}, phase="F42C_s50u_fbstar_hstub")["status"] == "PASS"  # harness suffix


def _ev23(agg, sel=(), trace=None):
    cfg = {"hyperparameters": {"avail_knowledge": "checkin", "time_mode": "simulated"}}
    run = {"dir": "/x/run_1_F42_s50u_ctl_dbg_fedbuff_n75_sim", "cfg": cfg, "agg": list(agg) + list(sel)}
    orig = ev._ground_truth
    ev._ground_truth = lambda r: trace
    try:
        return ev.ev23_checkin_protocol(run)
    finally:
        ev._ground_truth = orig


def test_ev23_checkin_protocol():
    from sortedcontainers import SortedDict
    tr = {"a": SortedDict({0.0: "AVL_TRAIN", 100.0: "UN_AVL", 300.0: "AVL_TRAIN"})}
    ck = lambda k, t, **f: {"event": "checkin", "kind": k, "end_id": "a", "t": t, **f}
    pick = {"event": "selection", "task": "train", "round": 1, "chosen": ["a"]}
    ok = [ck("pooled", 10), ck("picked", 20), ck("rejected", 50, reconnect_at=200), ck("missed", 200), ck("pooled", 300)]
    assert _ev23(ok, [pick], tr)["status"] == "PASS"
    assert _ev23([ck("picked", 20)], [pick], tr)["status"] == "FAIL"  # no waiting check-in
    assert _ev23([ck("rejected", 50, reconnect_at=200), ck("pooled", 150)], (), tr)["status"] == "FAIL"  # ignored pace steering
    assert _ev23([ck("pooled", 150)], (), tr)["status"] == "FAIL"  # checked in while UN_AVL
    assert _ev23([ck("missed", 50)], (), tr)["status"] == "FAIL"  # missed while AVL_TRAIN
    assert _ev23([ck("pooled", 10)], [pick], tr)["status"] == "FAIL"  # a pick the protocol never saw


def test_ev22_unaware_run_that_excludes_unavail_fails():
    sel = {"event": "selection", "task": "train", "excluded_by": {"unavail": 3}}
    assert _ev22("fedbuff", {"avail_knowledge": "none", "avail_select_filter": "False"}, agg=[sel])["status"] == "FAIL"
    evi = {"event": "abandon_timeout", "reason": "aware_boundary_eviction"}
    assert _ev22("oort_star", {"avail_knowledge": "oracle"}, agg=[evi])["status"] == "FAIL"
