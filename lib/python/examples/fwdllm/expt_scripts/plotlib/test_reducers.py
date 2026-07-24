"""Regression tests for BRIDGE_DESIGN.md §2b reducer-gap fixes (N3/N5/N6).

Run:  python -m pytest plotlib/test_reducers.py -q
  or:  python plotlib/test_reducers.py
"""

from __future__ import annotations

import json
import os
import sys

_PLOTLIB = os.path.dirname(os.path.abspath(__file__))
if _PLOTLIB not in sys.path:
    sys.path.insert(0, _PLOTLIB)

import reducers as R  # noqa: E402


# --------------------------------------------------------------------------- #
# N6 — sync "one round span" grouping
# --------------------------------------------------------------------------- #
def test_round_spans_group_contiguous_same_data_id():
    rounds = [
        {"ts": 100.0, "data_id": 0, "is_async": False},
        {"ts": 102.0, "data_id": 0, "is_async": False},
        {"ts": 105.0, "data_id": 0, "is_async": False},   # commit -> span 5
        {"ts": 106.0, "data_id": 1, "is_async": False},
        {"ts": 110.0, "data_id": 1, "is_async": False},   # span 4
    ]
    assert R._compute_round_spans(rounds) == [5.0, 4.0]


def test_round_spans_single_cycle_round_is_zero_not_dropped():
    """A round that commits on its FIRST cycle (no retries) is a real, valid
    (if degenerate) session -- must appear as 0.0, not be silently skipped."""
    rounds = [{"ts": 50.0, "data_id": 0, "is_async": False}]
    assert R._compute_round_spans(rounds) == [0.0]


def test_round_spans_excludes_async_cycles():
    rounds = [
        {"ts": 1.0, "data_id": 0, "is_async": True},
        {"ts": 2.0, "data_id": 0, "is_async": True},
    ]
    assert R._compute_round_spans(rounds) == []


def test_round_spans_revisited_data_id_is_a_new_span_not_merged():
    """data_id wraps every epoch (round_transition_indices' own premise) -- a
    LATER epoch's data_id=0 must start its own span, not extend the first."""
    rounds = [
        {"ts": 0.0, "data_id": 0, "is_async": False},
        {"ts": 1.0, "data_id": 0, "is_async": False},   # span 1 (epoch 1, bin 0)
        {"ts": 2.0, "data_id": 1, "is_async": False},   # span 0 (single-cycle bin 1)
        {"ts": 3.0, "data_id": 0, "is_async": False},   # new epoch's bin 0 starts
        {"ts": 5.0, "data_id": 0, "is_async": False},   # span 2 (epoch 2, bin 0)
    ]
    assert R._compute_round_spans(rounds) == [1.0, 0.0, 2.0]


# --------------------------------------------------------------------------- #
# N3 — converge.json discovery
# --------------------------------------------------------------------------- #
def test_find_converge_json_matches_by_agg_telemetry_path(tmp_path):
    run_dir = tmp_path / "run_x"
    (run_dir / "telemetry").mkdir(parents=True)
    agg_file = run_dir / "telemetry" / "aggregator_abc.jsonl"
    agg_file.write_text("")

    smoke_logs = tmp_path / "smoke_logs"
    (smoke_logs / "20260101_000000").mkdir(parents=True)
    cj_path = smoke_logs / "20260101_000000" / "converge_fwdllm_n10_smoke_sim.json"
    payload = {"converged": True, "target_accuracy": 0.84, "window": 20,
               "time_to_converge_wall_s": 123.4, "agg_telemetry": str(agg_file)}
    cj_path.write_text(json.dumps(payload))

    found = R._find_converge_json(str(run_dir), smoke_logs_dir=str(smoke_logs))
    assert found is not None
    assert found["time_to_converge_wall_s"] == 123.4


def test_find_converge_json_none_when_no_watcher_ran(tmp_path):
    run_dir = tmp_path / "run_y"
    (run_dir / "telemetry").mkdir(parents=True)
    (run_dir / "telemetry" / "aggregator_abc.jsonl").write_text("")
    smoke_logs = tmp_path / "smoke_logs"
    smoke_logs.mkdir()
    assert R._find_converge_json(str(run_dir), smoke_logs_dir=str(smoke_logs)) is None


def test_find_converge_json_ignores_a_different_runs_file(tmp_path):
    run_dir = tmp_path / "run_z"
    (run_dir / "telemetry").mkdir(parents=True)
    (run_dir / "telemetry" / "aggregator_abc.jsonl").write_text("")

    other_run_agg = tmp_path / "run_other" / "telemetry" / "aggregator_def.jsonl"
    other_run_agg.parent.mkdir(parents=True)
    other_run_agg.write_text("")

    smoke_logs = tmp_path / "smoke_logs"
    (smoke_logs / "ts").mkdir(parents=True)
    (smoke_logs / "ts" / "converge_other.json").write_text(
        json.dumps({"converged": True, "agg_telemetry": str(other_run_agg)}))

    assert R._find_converge_json(str(run_dir), smoke_logs_dir=str(smoke_logs)) is None


# --------------------------------------------------------------------------- #
# N5 — net_wait_frac / busy_frac decomposition (via _read_trainers + load_run)
# --------------------------------------------------------------------------- #
def _write_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def test_load_run_computes_net_wait_frac_alongside_busy_frac(tmp_path):
    run_dir = tmp_path / "run_n5"
    tdir = run_dir / "telemetry"
    tdir.mkdir(parents=True)
    _write_jsonl(tdir / "aggregator_1.jsonl", [
        {"event": "agg_eval", "ts": 0.0, "round": 0, "data_id": 0,
         "test-accuracy": 0.1, "test-loss": 2.0},
    ])
    # one trainer: span 10.0, 6s busy, 3s mqtt-wait -> idle should be ~1.0
    _write_jsonl(tdir / "trainer_a.jsonl", [
        {"event": "trainer_round", "ts": 0.0, "round": 0, "data_id": 0,
         "gpu_compute_s": 3.0, "mqtt_fetch_s": 1.5},
        {"event": "trainer_round", "ts": 10.0, "round": 0, "data_id": 0,
         "gpu_compute_s": 3.0, "mqtt_fetch_s": 1.5},
    ])
    rr = R.load_run(str(run_dir), post_peak_grace_s=None)
    assert rr is not None
    assert rr.busy_frac == [0.6]
    assert rr.net_wait_frac == [0.3]


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
