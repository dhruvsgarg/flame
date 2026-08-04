"""Unit tests for replicate_floor.py — the reproducibility-floor measurement
that calibrates the DIST parity tolerances.

Synthetic run dirs only; no telemetry from a live run is required.
"""

import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replicate_floor as rf  # noqa: E402


def _write_run(tmp_path, name, cycles, seed=1234, max_runtime_s=3600,
               achieved_s=None):
    """cycles: list of (round, data_id, iteration, var, committed).

    `ts` is WALL-CLOCK, as in real telemetry -- spread over `achieved_s`. Keying
    it to the cycle index would make a run that completed fewer bins look
    truncated, the exact distinction `achieved_span_s` draws.
    """
    d = tmp_path / name
    (d / "telemetry").mkdir(parents=True)
    json.dump({"hyperparameters": {"seed": seed, "max_runtime_s": max_runtime_s}},
              open(d / "aggregator_config.json", "w"))
    span = float(max_runtime_s if achieved_s is None else achieved_s)
    step = span / max(1, len(cycles) - 1)
    with open(d / "telemetry" / "aggregator_x.jsonl", "w") as fh:
        for i, (rd, did, it, var, committed) in enumerate(cycles):
            fh.write(json.dumps({
                "event": "agg_round", "round": rd, "ts": i * step,
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


class TestTruncatedLegIsNotAReplicate:
    """A run killed early still reports the `max_runtime_s` it was asked for.
    `run_20260801_232459_felix_round` stopped at 5571s of 7200s and was pooled
    with two 6949s legs, inflating the floor. Group on the ACHIEVED span."""

    def _out(self, tmp_path, capsys, *args):
        rf.main(["--experiments-dir", str(tmp_path), *args])
        return capsys.readouterr().out

    def test_short_leg_is_dropped_and_named(self, tmp_path, capsys):
        _run(9, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(9, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_real",
             achieved_s=1800)                      # killed at half the target
        out = self._out(tmp_path, capsys)
        assert "20260103_000000  DROPPED" in out
        assert "n_replicates=2" in out
        # The two full legs are identical -> a zero floor, not the 44% the
        # truncated leg would have manufactured.
        assert "BELOW FLOOR" not in out

    def test_full_length_legs_are_kept_even_with_different_bin_counts(self, tmp_path, capsys):
        # Fewer bins in the SAME wall time is a real reproducibility gap, not a
        # truncation -- it must still be graded.
        _run(9, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        out = self._out(tmp_path, capsys)
        assert "DROPPED" not in out
        assert "BELOW FLOOR" in out

    def test_span_tol_is_tunable(self, tmp_path, capsys):
        _run(9, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(9, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        _run(9, 4, 0.9, tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_real",
             achieved_s=3400)                      # 5.6% short
        assert "DROPPED" in self._out(tmp_path, capsys)
        assert "DROPPED" not in self._out(tmp_path, capsys, "--span-tol", "0.10")

    def test_dropping_below_two_legs_says_why(self, tmp_path, capsys):
        # Must not degrade to a bare "No replicate groups found".
        _run(9, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
             achieved_s=1800)
        out = self._out(tmp_path, capsys)
        assert "DROPPED" in out and "no floor for this group" in out


class TestFloorDurationMustMatchTheRungsWindow:
    """A baseline can now have ON replicate groups at two durations (fluxtune has
    7200s and 14400s). The floor must come from the length the rung grades, so the
    "longest ON group wins" rule needs an explicit override and a loud warning."""

    def _on(self, tmp_path, name, n_bins, iters, **kw):
        p = _run(n_bins, iters, 0.9, tmp_path, name, **kw)
        open(os.path.join(p, "x_trainers.log"), "w").write("jvp_eval_mode=True\n")
        return p

    def test_duration_selects_the_group(self, tmp_path, capsys):
        self._on(tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_real", 9, 4,
                 max_runtime_s=7200)
        self._on(tmp_path, "run_20260104_000000_b_n10_smoke_syn_0_real", 9, 4,
                 max_runtime_s=7200)
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real",
                 "--duration", "3600"])
        out = capsys.readouterr().out
        assert "max_runtime_s=3600" in out and "max_runtime_s=7200" not in out

    def test_two_on_durations_warn_when_unselected(self, tmp_path, capsys):
        self._on(tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_real", 9, 4,
                 max_runtime_s=7200)
        self._on(tmp_path, "run_20260104_000000_b_n10_smoke_syn_0_real", 9, 4,
                 max_runtime_s=7200)
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real",
                 "--profile-out", str(tmp_path / "floors")])
        out = capsys.readouterr().out
        assert "ON groups at" in out and "--duration" in out
        prof = yaml.safe_load(open(tmp_path / "floors" / "b.yaml"))
        assert prof["max_runtime_s"] == 7200        # longest, as documented

    def test_selected_duration_writes_that_floor_and_no_warning(self, tmp_path, capsys):
        self._on(tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_real", 9, 4,
                 max_runtime_s=7200)
        self._on(tmp_path, "run_20260104_000000_b_n10_smoke_syn_0_real", 9, 4,
                 max_runtime_s=7200)
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real",
                 "--duration", "3600", "--profile-out", str(tmp_path / "floors")])
        assert "ON groups at" not in capsys.readouterr().out
        prof = yaml.safe_load(open(tmp_path / "floors" / "b.yaml"))
        assert prof["max_runtime_s"] == 3600

    def test_single_duration_never_warns(self, tmp_path, capsys):
        self._on(tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real", 5, 4)
        self._on(tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real", 5, 4)
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real",
                 "--profile-out", str(tmp_path / "floors")])
        assert "ON groups at" not in capsys.readouterr().out
