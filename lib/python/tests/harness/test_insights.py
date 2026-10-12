# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FELIX_INSIGHTS: scripts/insights.py metrics on a synthetic run."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import insights  # noqa: E402


def _run(tmp_path, name, events, n=4):
    d = tmp_path / name
    (d / "telemetry").mkdir(parents=True)
    (d / "telemetry" / "aggregator_x.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (d / "telemetry" / "trainer_t1.jsonl").write_text(json.dumps({"event": "wire", "bytes": 500}) + "\n")
    (d / "aggregator_config.json").write_text(json.dumps({"hyperparameters": {"min_trainers_to_start": n}}))
    return str(d)


def _events():
    sel = lambda t, eval_, held, blocked: {"event": "selection", "task": "train", "ts": t, "vclock_now": t,
                                           "num_eligible": 4, "num_chosen": 1, "concurrency": 2,
                                           "in_flight_train": 2 - eval_, "in_flight_eval": eval_,
                                           "train_slots_held_by_eval": held, "train_blocked_by_eval": blocked,
                                           "explore_ids": ["t1"] if t == 0 else [], "exploit_ids": ["t2"]}
    return [
        sel(0.0, 0, 0, False), sel(10.0, 1, 1, True), sel(20.0, 0, 0, False),
        {"event": "selection", "task": "eval", "ts": 5.0, "num_eligible": 2, "num_chosen": 1},
        {"event": "agg_round", "ts": 9.0, "vclock_now": 9.0, "contributing_trainers": ["t1", "t2"], "staleness": [0, 1]},
        {"event": "agg_round", "ts": 19.0, "vclock_now": 19.0, "contributing_trainers": ["t1"], "staleness": [7]},
        {"event": "agg_round", "ts": 15.0, "task_to_perform": "eval", "contributing_trainers": ["t3"], "staleness": [0]},
        {"event": "abandon_timeout", "ts": 12.0, "reason": "abandon_90s_vclock"},
        {"event": "wire", "ts": 1.0, "dir": "send", "bytes": 2_000_000},
        {"event": "wire", "ts": 2.0, "dir": "recv", "bytes": 1_000_000},
        {"event": "agg_timing", "ts": 9.0, "recv_wait_s": 1.0, "ingest_s": 0.5, "commit_s": 0.1, "other_s": 0.0, "cycle_s": 1.6},
    ]


def test_metrics(tmp_path):
    r = insights.analyze(_run(tmp_path, "run_20261012_000000_X_sim", _events()))
    assert r["I-A2"]["contributors"] == 2 and r["I-A2"]["coverage"] == 0.5
    assert r["I-A3"]["train"]["chosen"] == 3 and r["I-A3"]["train"]["choice_ratio_mean"] == 4.0
    assert r["I-A5"] == {"explore": 1, "exploit": 3, "explore_share": 0.25}
    assert r["I-B1"]["aggregated"] == 3 and r["I-B1"]["waste_share"] == 0.0 and r["I-B1"]["abandons"] == {"abandon_90s_vclock": 1}
    assert r["I-C1"]["updates"] == 3 and abs(r["I-C1"]["gt5_share"] - 1 / 3) < 1e-9
    assert r["I-E1"]["mb_per_aggregated_update"] == 1.0 and r["I-E1"]["trainer_side_mb"] == 0.0005
    f1 = r["I-F1"]
    assert f1["c"] == 2 and f1["rho"] == 0.25 and f1["blocked_time_share"] == 0.5 and f1["eval_tasks"] == 1
    assert r["I-G1"]["span_s"] == 19.0


def test_pairs_and_plot(tmp_path):
    rs = [insights.analyze(_run(tmp_path, f"run_20261012_000000_X_{m}", _events())) for m in ("real", "sim")]
    assert list(insights.pairs(rs)) == ["X"]
    insights.plot(rs, str(tmp_path / "p.png"))
    assert (tmp_path / "p.png").stat().st_size > 0
