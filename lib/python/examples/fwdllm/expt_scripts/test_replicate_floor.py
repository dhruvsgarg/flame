"""Unit tests for replicate_floor.py — the reproducibility-floor measurement
that calibrates the DIST parity tolerances.

Synthetic run dirs only; no telemetry from a live run is required.
"""

import glob
import json
import re
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replicate_floor as rf  # noqa: E402


def _write_run(tmp_path, name, cycles, seed=1234, max_runtime_s=3600,
               achieved_s=None, vclock_s=None):
    """cycles: list of (round, data_id, iteration, var, committed).

    `ts` is WALL-CLOCK, as in real telemetry -- spread over `achieved_s`. Keying
    it to the cycle index would make a run that completed fewer bins look
    truncated, the exact distinction `achieved_span_s` draws.

    `vclock_s` emits `vclock_now` spread over that span, as a SIM leg does. Real
    legs leave it None, which is what makes the axis choice self-describing.
    """
    d = tmp_path / name
    (d / "telemetry").mkdir(parents=True)
    json.dump({"hyperparameters": {"seed": seed, "max_runtime_s": max_runtime_s}},
              open(d / "aggregator_config.json", "w"))
    span = float(max_runtime_s if achieved_s is None else achieved_s)
    step = span / max(1, len(cycles) - 1)
    vstep = None if vclock_s is None else float(vclock_s) / max(1, len(cycles) - 1)
    with open(d / "telemetry" / "aggregator_x.jsonl", "w") as fh:
        for i, (rd, did, it, var, committed) in enumerate(cycles):
            e = {"event": "agg_round", "round": rd, "ts": i * step,
                 "cycle_data_id": did, "iteration_per_data_id": it,
                 "var": var, "var_good_enough": committed}
            if vstep is not None:
                e["vclock_now"] = i * vstep
            fh.write(json.dumps(e) + "\n")
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


class TestFloorIsMeasuredOnTheRungsOwnWindow:
    """§D-53: a run-level mean over each leg's FULL run is not the window `v1`
    grades. `v1` truncates both legs to the matched logical budget, so a leg that
    ran further contributes bins the rung never sees — and the floor comes out
    too small exactly where the residual is largest."""

    def test_identical_legs_have_a_zero_floor(self, tmp_path):
        a = _run(6, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        b = _run(6, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        floors = rf.rung_floors([a, b])
        assert floors["iters_per_bin"] == pytest.approx(0.0)

    def test_the_rung_ignores_a_tail_the_run_level_mean_averages_in(self, tmp_path):
        """Leg B commits the same 6 bins, then keeps going at a different cadence.
        The matched prefix is those 6 bins — where the legs agree exactly — so the
        rung reads 0 while the run-level mean reads the tail."""
        cycles_a = [(1, b, it, 0.9, it == 3) for b in range(6) for it in range(4)]
        cycles_b = cycles_a + [(1, b, it, 0.9, it == 9)
                               for b in range(6, 9) for it in range(10)]
        a = _write_run(tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real", cycles_a)
        b = _write_run(tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real", cycles_b)
        assert rf.rung_floors([a, b])["iters_per_bin"] == pytest.approx(0.0)
        assert rf._spread([rf.metrics(a)["iters_per_bin"],
                           rf.metrics(b)["iters_per_bin"]]) > 0.3

    def test_floor_is_the_max_over_every_pair(self, tmp_path):
        # Max-pairwise (§D-57): the worst pair sets the floor, not the mean pair.
        a = _run(6, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        b = _run(6, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        c = _run(6, 8, 0.9, tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_real")
        floors = rf.rung_floors([a, b, c])
        assert floors["iters_per_bin"] == pytest.approx(0.5)     # 4 vs 8 iters

    def test_run_level_flag_restores_the_old_estimator(self, tmp_path, capsys):
        cycles_a = [(1, b, it, 0.9, it == 3) for b in range(6) for it in range(4)]
        cycles_b = cycles_a + [(1, b, it, 0.9, it == 9)
                               for b in range(6, 9) for it in range(10)]
        _write_run(tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real", cycles_a)
        _write_run(tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real", cycles_b)
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real"])
        windowed = capsys.readouterr().out
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real", "--run-level"])
        assert windowed != capsys.readouterr().out

    def test_eval_tagged_agg_rounds_are_not_cadence_events(self, tmp_path):
        """The checker routes `task_to_perform=eval` agg_rounds to `eval_commits`.
        A floor built on the raw event stream would grade a different event set
        than the rung it calibrates."""
        p = _run(4, 3, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        raw = len(rf._agg_events(p))
        with open(os.path.join(p, "telemetry", "aggregator_x.jsonl"), "a") as fh:
            fh.write(json.dumps({"event": "agg_round", "round": 1, "ts": 99.0,
                                 "task_to_perform": "eval", "cycle_data_id": 0,
                                 "iteration_per_data_id": 0, "var": 0.9,
                                 "var_good_enough": True}) + "\n")
        assert len(rf._agg_events(p)) == raw + 1
        assert len(rf.checker_agg(p)["agg_rounds"]) == raw


class TestCodeVersionGrouping:
    """§D-70: two legs are replicates only if they ran the same CODE. `fwdllm`'s
    two sim legs straddle a charge re-profile and pooled to an 18% "floor" on a
    baseline whose real floor is 0.0%."""

    def test_same_commit_never_differs(self):
        assert rf.code_differs("abc123def", "abc123def") == (False, "same commit")

    def test_a_missing_commit_is_assumed_different(self):
        """Refusing to pool is the safe error."""
        assert rf.code_differs(None, "abc123def")[0] is True
        assert rf.code_differs("abc123def", "")[0] is True

    def test_an_unknown_commit_is_assumed_different(self):
        assert rf.code_differs("0" * 9, "1" * 9)[0] is True

    def test_docs_only_commits_do_not_split_a_group(self):
        """The whole point: this doc is committed constantly, and a docs-only
        commit between two legs must not make them different runs."""
        differs, why = rf.code_differs("2282f4b65", "09251f782")
        assert differs is False and "docs" in why

    def test_a_real_code_change_does_split(self):
        differs, why = rf.code_differs("779cdee23", "09251f782")
        assert differs is True and "run-affecting" in why

    def test_launcher_invoked_scripts_are_run_affecting(self):
        """Derived from run_sequential.sh so the deny-list cannot drift: any
        expt_scripts python the LAUNCHER runs must count as run-affecting."""
        sh = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_sequential.sh")
        invoked = set(re.findall(r"([a-z_]+)\.py", open(sh, errors="ignore").read()))
        here = {os.path.basename(p)[:-3]
                for p in glob.glob(os.path.join(os.path.dirname(sh), "*.py"))}
        for name in invoked & here:
            path = f"lib/python/examples/fwdllm/expt_scripts/{name}.py"
            assert not rf._RUN_IRRELEVANT.search(path), (
                f"{name}.py is invoked by run_sequential.sh but is treated as "
                f"run-irrelevant; add it to LAUNCHER_INVOKES")

    def test_analysis_scripts_are_not_run_affecting(self):
        for name in ("run_parity", "replicate_floor", "compare_baselines"):
            assert rf._RUN_IRRELEVANT.search(
                f"lib/python/examples/fwdllm/expt_scripts/{name}.py")

    def test_charge_profiles_are_run_affecting(self):
        """The file that caused the 18-point artifact must never be filtered."""
        assert not rf._RUN_IRRELEVANT.search(
            "lib/python/examples/fwdllm/sim_charge_profiles/fwdllm.yaml")

    def test_floors_and_reports_are_not(self):
        assert rf._RUN_IRRELEVANT.search(
            "lib/python/examples/fwdllm/parity_floors/fwdllm.yaml")
        assert rf._RUN_IRRELEVANT.search("lib/python/examples/fwdllm/README.md")

    def test_a_charge_reprofile_does_not_split_REAL_legs(self):
        # A charge profile only reaches a SIM run (§F-1), so committing one must
        # not split a real-side group. bdbde72b7 -> f2b7de071 changed exactly the
        # four block-1 charge profiles plus deny-listed analysis files.
        differs_real, why = rf.code_differs("bdbde72b7", "f2b7de071", mode="real")
        assert differs_real is False, why
        assert "sim charges" in why

    def test_the_same_pair_still_splits_SIM_legs(self):
        differs_sim, why = rf.code_differs("bdbde72b7", "f2b7de071", mode="sim")
        assert differs_sim is True, why

    def test_mode_none_stays_conservative(self):
        assert rf.code_differs("bdbde72b7", "f2b7de071")[0] is True

    def test_largest_same_code_keeps_the_biggest_cluster(self, tmp_path):
        def leg(name, sha):
            p = _run(5, 4, 0.9, tmp_path, name)
            yaml.safe_dump({"git_info": {"commit": sha, "clean": True}},
                           open(os.path.join(p, "snapshot.yaml"), "w"))
            return p
        leg("run_20260101_000000_b_n10_smoke_syn_0_real", "779cdee23")
        leg("run_20260102_000000_b_n10_smoke_syn_0_real", "779cdee23")
        leg("run_20260103_000000_b_n10_smoke_syn_0_real", "09251f782")
        legs = [(t, str(tmp_path / f"run_2026010{i}_000000_b_n10_smoke_syn_0_real"))
                for i, t in enumerate(["20260101_000000", "20260102_000000",
                                       "20260103_000000"], start=1)]
        kept, dropped, sha = rf.largest_same_code(legs)
        assert len(kept) == 2 and len(dropped) == 1 and sha == "779cdee23"

    def test_legs_without_a_snapshot_still_group(self, tmp_path):
        _run(5, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real")
        legs = [(t, str(tmp_path / f"run_{t}_b_n10_smoke_syn_0_real"))
                for t in ("20260101_000000", "20260102_000000")]
        kept, dropped, _sha = rf.largest_same_code(legs)
        assert len(kept) == 2 and dropped == []


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


class TestTruncationIsJudgedOnTheLegsOwnClock:
    """A SIM leg's WALL span measures the host, not the work (§D-73). Judging
    truncation on it drops a complete leg for running on a quieter node -- which
    is what cost `fedbuff_round` its third sim replicate, on the one baseline
    whose sim floor the run batch existed to measure."""

    def _out(self, tmp_path, capsys, *args):
        rf.main(["--experiments-dir", str(tmp_path), *args])
        return capsys.readouterr().out

    def test_wall_fast_sim_leg_is_kept_because_its_vclock_is_full_length(
            self, tmp_path, capsys):
        # The live case: three sim legs, all vclock 7200, one 6% faster in wall.
        for i, wall in enumerate((2226, 2090, 2169)):
            _run(9, 4, 0.9, tmp_path,
                 f"run_2026010{i + 1}_000000_b_n10_smoke_syn_0_sim",
                 max_runtime_s=7200, achieved_s=wall, vclock_s=7200)
        out = self._out(tmp_path, capsys, "--mode", "sim")
        assert "DROPPED" not in out
        assert "n_replicates=3" in out

    def test_a_genuinely_short_sim_leg_is_still_dropped(self, tmp_path, capsys):
        # The guard must keep working on the axis that measures sim's work.
        for i in (1, 2):
            _run(9, 4, 0.9, tmp_path,
                 f"run_2026010{i}_000000_b_n10_smoke_syn_0_sim",
                 max_runtime_s=7200, achieved_s=2200, vclock_s=7200)
        _run(5, 4, 0.9, tmp_path, "run_20260103_000000_b_n10_smoke_syn_0_sim",
             max_runtime_s=7200, achieved_s=2200, vclock_s=3600)   # half the work
        out = self._out(tmp_path, capsys, "--mode", "sim")
        assert "20260103_000000  DROPPED" in out
        assert "vclock span" in out
        assert "n_replicates=2" in out

    def test_real_legs_are_unaffected_and_still_read_wall(self, tmp_path, capsys):
        # Real emits no vclock_now, so the axis and every existing verdict hold.
        _run(9, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real")
        _run(5, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
             achieved_s=1800)
        out = self._out(tmp_path, capsys)
        assert "20260102_000000  DROPPED" in out
        assert "wall span" in out

    def test_a_group_missing_one_vclock_falls_back_to_wall_for_all(self, tmp_path):
        # Mixing axes would compare 7200 against 2200 and drop the wall leg every
        # time, so the whole group must fall back rather than half-convert.
        a = _run(9, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_sim",
                 max_runtime_s=7200, achieved_s=2200, vclock_s=7200)
        b = _run(9, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_sim",
                 max_runtime_s=7200, achieved_s=2200)          # legacy, no vclock
        rows, axis = rf.leg_spans([("20260101_000000", a), ("20260102_000000", b)])
        assert axis == "wall"
        assert all(abs(s - 2200) < 1 for _, _, s in rows)


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


class TestTheProfileCarriesBothSidesFloors:
    """§D-61: the gate needs real's floor AND sim's. They are measured by separate
    invocations, so writing one must not clobber the other."""

    def _on(self, tmp_path, name, n_bins, iters, **kw):
        p = _run(n_bins, iters, 0.9, tmp_path, name, **kw)
        open(os.path.join(p, "x_trainers.log"), "w").write("jvp_eval_mode=True\n")
        return p

    def _both_modes(self, tmp_path, sim_iters=(4, 5)):
        for i in (1, 2):
            self._on(tmp_path, f"run_2026010{i}_000000_b_n10_smoke_syn_0_real", 9, 4)
        # Sim legs disagree on the GRADED quantity, the way live sim legs do;
        # differing only in bin count leaves iters/bin identical and floor 0.
        for i, it in zip((3, 4), sim_iters):
            self._on(tmp_path, f"run_2026010{i}_000000_b_n10_smoke_syn_0_sim", 9, it,
                     vclock_s=3600)
        out = str(tmp_path / "floors")
        for mode in ("real", "sim"):
            rf.main(["--experiments-dir", str(tmp_path), "--mode", mode,
                     "--profile-out", out])
        return yaml.safe_load(open(tmp_path / "floors" / "b.yaml"))

    def test_sim_pass_keeps_the_real_floor(self, tmp_path, capsys):
        prof = self._both_modes(tmp_path)
        capsys.readouterr()
        assert "metrics" in prof and "sim_metrics" in prof
        assert prof["metrics"]["iters_per_bin"] == 0.0        # identical real legs
        assert prof["source_runs"] == ["20260101_000000", "20260102_000000"]
        assert prof["sim_source_runs"] == ["20260103_000000", "20260104_000000"]

    def test_the_sim_side_records_its_own_spread(self, tmp_path, capsys):
        prof = self._both_modes(tmp_path)
        capsys.readouterr()
        # Divergent sim legs must show a floor the real-only profile never had.
        assert prof["sim_metrics"]["iters_per_bin"] > 0.0
        assert prof["sim_n_replicates"] == 2

    def test_real_pass_after_sim_keeps_the_sim_floor(self, tmp_path, capsys):
        self._both_modes(tmp_path)
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real",
                 "--profile-out", str(tmp_path / "floors")])
        capsys.readouterr()
        prof = yaml.safe_load(open(tmp_path / "floors" / "b.yaml"))
        assert "sim_metrics" in prof and "sim_source_runs" in prof


class TestTheVclockTimeFamilyHasAMeasurableFloor:
    """§D-72: `throughput`, `terminal_state` and `total_commits` BAIL on two real
    legs for want of `vclock_now`, so they were left on hand-typed gates. The rung
    bails; the quantity does not — real's own wall-time-to-N is measurable from
    two real legs, and it separates the pinned baselines (0.0%) from the unpinned
    (up to 6.7% same-code, 13.6% pooled across code) exactly where gates fire."""

    def test_the_three_metrics_are_measured_by_calling_the_rung(self):
        """Not reimplemented here (§D-53). A `gap` of None would mean the metric
        silently fell back to a run-level spread on the wrong window."""
        for metric in ("throughput_rel", "time_to_n", "trainers_at_n"):
            assert rf._CALIBRATES[metric][3] is not None

    def test_throughput_is_no_longer_gated_on_work_volume(self):
        """`committed_bins` is how MUCH work got done; the rung grades how LONG it
        took. Wrong metric, and on fedbuff_round they differ 2.7x."""
        assert "committed_bins" not in rf._CALIBRATES
        assert rf._CALIBRATES["throughput_rel"][0] == "throughput"

    def test_a_pair_of_real_legs_produces_a_time_floor(self, tmp_path):
        """Two legs, same bins, different wall spans: the floor is the span gap.
        Before this the rung bailed and the metric did not exist."""
        a = _run(6, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real",
                 achieved_s=1000)
        b = _run(6, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
                 achieved_s=1200)
        floors = rf.rung_floors([a, b])
        assert floors["time_to_n"] == pytest.approx(1 - 1000 / 1200, abs=1e-3)
        assert floors["throughput_rel"] == pytest.approx(0.167, abs=1e-3)

    def test_identical_legs_floor_at_zero(self, tmp_path):
        """The built-in negative control: the pinned baselines measure 0.0%."""
        a = _run(6, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real",
                 achieved_s=1000)
        b = _run(6, 4, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
                 achieved_s=1000)
        floors = rf.rung_floors([a, b])
        assert floors["time_to_n"] == pytest.approx(0.0)
        assert floors["trainers_at_n"] == pytest.approx(0.0)

    def test_the_floor_follows_the_field_the_rung_decided_on(self, tmp_path):
        """`throughput` switches between `rel_diff` and `matched_window_rel_diff`
        by baseline. Reading the other one grades a window the rung never used
        (§D-71) — the same bug that understated v2's floor by 2.3x."""
        import parity.checks as C
        a = _run(6, 4, 0.9, tmp_path, "run_20260101_000000_b_n10_smoke_syn_0_real",
                 achieved_s=1000)
        b = _run(6, 5, 0.9, tmp_path, "run_20260102_000000_b_n10_smoke_syn_0_real",
                 achieved_s=1200)
        res = C.throughput_parity(rf.checker_agg(a), rf.checker_agg(b),
                                  same_mode=True)
        assert res["matched_window_rel_diff"] != res["rel_diff"]
        assert rf.rung_floors([a, b])["throughput_rel"] == pytest.approx(
            res[res["decided_on"]], abs=1e-3)


class TestSyn0FloorPooling:
    """§D-63: `fedbuff_it_oracular` and `fedbuff_it_unaware` resolve to configs
    differing in exactly one key — `trackTrainerAvail`, measured provably inert at
    syn_0 (eligible_pool_reduction 0.0/0.0, the oracle removes nobody). Their
    floors differ 3x (13.8% vs 4.9%) purely by which extremes landed under which
    name (§D-67), and unpooled `_unaware` has n=1: no floor at all."""

    @staticmethod
    def _legs(tmp_path, **kw):
        for i, (name, iters) in enumerate([("fedbuff_it_oracular", 4),
                                           ("fedbuff_it_unaware", 6)]):
            for j in range(2):
                _run(6, iters, 0.9, tmp_path,
                     f"run_2026010{i}_00000{j}_{name}_n10_smoke_syn_0_real", **kw)

    def test_the_two_names_pool_into_one_group(self, tmp_path):
        self._legs(tmp_path)
        groups = rf.discover(str(tmp_path), None, "real")
        assert [k[0] for k in groups] == ["fedbuff_it"]
        assert len(next(iter(groups.values()))) == 4

    def test_pooling_is_syn_0_only(self):
        """Phase 2 deletes this alias. It must not survive into a trace where
        `trackTrainerAvail` actually bites."""
        assert rf.pool_name("fedbuff_it_oracular", "syn_0") == "fedbuff_it"
        assert rf.pool_name("fedbuff_it_oracular", "fedscale") == "fedbuff_it_oracular"

    def test_an_unpooled_baseline_is_untouched(self):
        assert rf.pool_name("fedbuff_round", "syn_0") == "fedbuff_round"
        assert rf.pool_members("fedbuff_round") == ["fedbuff_round"]

    def test_either_member_name_selects_the_whole_group(self, tmp_path):
        """A floor asked for by one name and answered from half its legs is the
        n=1 problem this exists to fix."""
        self._legs(tmp_path)
        groups = rf.discover(str(tmp_path), ["fedbuff_it_unaware"], "real")
        assert len(next(iter(groups.values()))) == 4
        assert list(rf.discover(str(tmp_path), ["fedbuff_it"], "real")) == list(groups)

    def test_no_pool_restores_the_separate_names(self, tmp_path):
        self._legs(tmp_path)
        groups = rf.discover(str(tmp_path), None, "real", pool=False)
        assert sorted(k[0] for k in groups) == ["fedbuff_it_oracular",
                                                "fedbuff_it_unaware"]

    def test_the_pooled_floor_lands_in_every_members_own_file(self, tmp_path):
        """The checker looks a floor up by the baseline it is GRADING, and the
        parity rows stay separate — so a group floor has to reach both files."""
        out = tmp_path / "floors"
        self._legs(tmp_path, max_runtime_s=7200)
        for p in glob.glob(str(tmp_path / "run_*")):
            open(os.path.join(p, "x_trainers.log"), "w").write("jvp_eval_mode=True\n")
        rf.main(["--experiments-dir", str(tmp_path), "--mode", "real",
                 "--profile-out", str(out), "--any-code"])
        written = sorted(os.path.basename(p) for p in glob.glob(str(out / "*.yaml")))
        assert written == ["fedbuff_it_oracular.yaml", "fedbuff_it_unaware.yaml"]
        prof = yaml.safe_load(open(out / "fedbuff_it_unaware.yaml"))
        assert prof["n_replicates"] == 4
        # Say WHY one name's file carries four source runs, two of them another's.
        assert prof["pooled_from"] == ["fedbuff_it_oracular", "fedbuff_it_unaware"]
        assert yaml.safe_load(open(out / "fedbuff_it_oracular.yaml")) == prof

    def test_the_checker_finds_a_pooled_groups_floor(self, tmp_path, monkeypatch):
        """`--control` labels a pair with the POOLED name, which has no file of
        its own. Grading a control at nominal while the board is floor-gated makes
        the two disagree by construction (§D-65)."""
        import run_parity
        out = tmp_path / "floors"
        out.mkdir()
        yaml.safe_dump({"metrics": {"time_to_n": 0.11}},
                       open(out / "fedbuff_it_unaware.yaml", "w"))
        monkeypatch.setattr(run_parity, "_FLOOR_DIR", out)
        assert run_parity._floors("fedbuff_it/syn_0") == {"time_to_n": 0.11}
        assert run_parity._floors("felix_round/syn_0") is None
