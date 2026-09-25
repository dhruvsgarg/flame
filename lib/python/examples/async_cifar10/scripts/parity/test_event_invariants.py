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
            agg.append({"event": "dispatch", "task": "train", "end_id": t, "ts": ts, "sim_send_ts": vclock})
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


def test_ev10_redispatch_while_outstanding_fails(tmp_path):
    agg, tr = _clean_run()
    d = next(e for e in agg if e["event"] == "dispatch")
    agg.append({**d, "ts": d["ts"] + 0.001})
    assert _status(_write_run(tmp_path, agg, tr), "EV10_dispatch_one_in_flight") == "FAIL"


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


def test_broken_check_is_error_not_crash(tmp_path, monkeypatch):
    def boom(run):
        raise RuntimeError("x")
    monkeypatch.setattr(ev, "CHECKS", [boom])
    r = ev.check_run(_write_run(tmp_path, *_clean_run()))
    assert r["checks"]["boom"]["status"] == "ERROR" and not r["passed"]
