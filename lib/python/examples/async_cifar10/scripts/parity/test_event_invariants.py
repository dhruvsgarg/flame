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
                 "real_gpu_time_s": 0.01, "sim_round_duration_s": 5.0, "training_budget_s": 5.0,
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
    monkeypatch.setattr(ev, "_registry_delays", lambda: {T1: 5.0, T2: 5.0})


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
    monkeypatch.setattr(ev, "_registry_delays", lambda: {T1: 9.0, T2: 5.0})
    assert _status(_write_run(tmp_path, *_clean_run()), "EV3_duration_model") == "FAIL"


def test_ev3_delay_factor_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_registry_delays", lambda: {T1: 20.0, T2: 20.0})
    run = _write_run(tmp_path, *_clean_run(), hp={"trainingDelayFactor": 4.0})
    assert _status(run, "EV3_duration_model") == "PASS"


def test_ev5_commit_without_send_fails(tmp_path):
    agg, tr = _clean_run()
    tr[T2] = [e for e in tr[T2] if e["event"] != "task_send"]
    assert _status(_write_run(tmp_path, agg, tr), "EV5_commit_accounting") == "FAIL"


def test_ev7_short_round_fails(tmp_path):
    agg, tr = _clean_run()
    first_commit = next(i for i, e in enumerate(agg) if e["event"] == "agg_round")
    del agg[first_commit]
    assert _status(_write_run(tmp_path, agg, tr), "EV7_agg_goal_cadence") == "FAIL"


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


def test_ev11_clock_backwards_fails(tmp_path):
    agg, tr = _clean_run()
    commits = [e for e in agg if e["event"] == "agg_round"]
    commits[-1]["vclock_now"] = 1.0
    assert _status(_write_run(tmp_path, agg, tr), "EV11_vclock") == "FAIL"


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
                                     ({"trackTrainerAvail": {"trace": "syn_50"}}, "WARN"),
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
    agg.append({"event": "abandon_timeout", "end_id": d["end_id"], "reason": "abandon_90s_vclock", "ts": 50})
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
