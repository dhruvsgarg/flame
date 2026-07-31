"""Unit tests for replicate_floor.py — the reproducibility-floor measurement
that calibrates the DIST parity tolerances.

Synthetic run dirs only; no telemetry from a live run is required.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replicate_floor as rf  # noqa: E402


def _write_run(tmp_path, name, cycles, seed=1234, max_runtime_s=3600):
    """cycles: list of (round, data_id, iteration, var, committed)."""
    d = tmp_path / name
    (d / "telemetry").mkdir(parents=True)
    json.dump({"hyperparameters": {"seed": seed, "max_runtime_s": max_runtime_s}},
              open(d / "aggregator_config.json", "w"))
    with open(d / "telemetry" / "aggregator_x.jsonl", "w") as fh:
        for i, (rd, did, it, var, committed) in enumerate(cycles):
            fh.write(json.dumps({
                "event": "agg_round", "round": rd, "ts": float(i),
                "cycle_data_id": did, "iteration_per_data_id": it,
                "var": var, "var_good_enough": committed}) + "\n")
    return str(d)


def _run(n_bins, iters, var, tmp_path, name, **kw):
    cycles = []
    for b in range(n_bins):
        for it in range(iters):
            cycles.append((1, b, it, var, it == iters - 1))
    return _write_run(tmp_path, name, cycles, **kw)


class TestMetrics:
    def test_reads_the_four_graded_quantities(self, tmp_path):
        p = _run(5, 4, 0.9, tmp_path, "run_20260101_000000_x_n10_smoke_syn_0_real")
        m = rf.metrics(p)
        assert m["committed_bins"] == 5
        assert m["cycles"] == 20
        assert m["iters_per_bin"] == pytest.approx(4.0)
        assert m["mean_var"] == pytest.approx(0.9)

    def test_none_on_a_run_with_no_agg_rounds(self, tmp_path):
        d = tmp_path / "run_20260101_000000_x_n10_smoke_syn_0_real"
        (d / "telemetry").mkdir(parents=True)
        open(d / "telemetry" / "aggregator_x.jsonl", "w").close()
        assert rf.metrics(str(d)) is None

    def test_uncommitted_trailing_bin_is_not_counted(self, tmp_path):
        # A run cut off by max_runtime_s mid variance-check emits cycles for a bin
        # it never committed (§D-13) -- that bin is not progress.
        cycles = [(1, 0, 0, 0.5, True), (1, 1, 0, 0.5, False)]
        p = _write_run(tmp_path, "run_20260101_000000_x_n10_smoke_syn_0_real", cycles)
        assert rf.metrics(p)["committed_bins"] == 1


class TestGrouping:
    def test_only_same_duration_runs_are_pooled(self, tmp_path):
        # iters/bin rises with run length, so pooling durations measures the
        # training curve rather than reproducibility.
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(9, 6, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
             max_runtime_s=5400)
        groups = rf.discover(str(tmp_path), ["b"], "real")
        assert len(groups) == 2
        assert all(len(v) == 1 for v in groups.values())

    def test_same_duration_runs_group_together(self, tmp_path):
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        groups = rf.discover(str(tmp_path), ["b"], "real")
        assert len(groups) == 1 and len(next(iter(groups.values()))) == 2

    def test_mode_is_respected(self, tmp_path):
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_sim")
        assert len(rf.discover(str(tmp_path), ["b"], "real")) == 1
        assert len(rf.discover(str(tmp_path), ["b"], "sim")) == 1

    def test_baseline_token_is_exact(self, tmp_path):
        # The same footgun run_parity.py guards: a `fwdllm` filter must not
        # capture `fwdllm_plus`.
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_fwdllm_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_fwdllm_plus_n10_smoke_syn_0_real")
        groups = rf.discover(str(tmp_path), ["fwdllm"], "real")
        assert {k[0] for k in groups} == {"fwdllm"}


class TestSpread:
    def test_identical_runs_have_a_zero_floor(self):
        assert rf._spread([10.0, 10.0, 10.0]) == 0.0

    def test_spread_is_relative_to_the_larger_value(self):
        assert rf._spread([105.0, 110.0]) == pytest.approx(5 / 110)


class TestVerdicts:
    def _out(self, tmp_path, capsys, **kw):
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real"])
        return capsys.readouterr().out

    def test_tolerance_below_the_floor_is_called_out(self, tmp_path, capsys):
        # 5 vs 9 bins = 44% spread; every tolerance here is far under it.
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(9, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        out = self._out(tmp_path, capsys)
        assert "BELOW FLOOR" in out

    def test_reproducible_runs_report_ok(self, tmp_path, capsys):
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        out = self._out(tmp_path, capsys)
        assert "BELOW FLOOR" not in out and "OK" in out

    def test_mixed_seeds_are_flagged_not_silently_pooled(self, tmp_path, capsys):
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
             seed=999)
        assert "MIXED SEEDS" in self._out(tmp_path, capsys)

    def test_single_run_is_not_a_floor(self, tmp_path, capsys):
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        assert rf.main(["--experiments-dir", str(tmp_path), "--mode", "real"]) == 1
        assert "No replicate groups" in capsys.readouterr().out
