"""Unit tests for run_parity.py's discovery and parallel-grading plumbing.

Grading itself is covered by tests/mode/test_parity_checks.py; this file covers
only what run_parity.py adds on top — pair discovery, the worker-count policy,
and the guarantee that parallel and serial produce identical results in the same
order.
"""

import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_parity as rp  # noqa: E402


def _mk_run(tmp_path, name, n_bins=4, agg_goal=2, vclock=False, jvp=None,
            max_runtime_s=None):
    d = tmp_path / name
    (d / "telemetry").mkdir(parents=True)
    if jvp is not None:                       # the knob lives only in the trainer log
        (d / "x_trainers.log").write_text(f"[JVP_EVAL_MODE] jvp_eval_mode={jvp}\n")
    json.dump({"hyperparameters": {"aggGoal": agg_goal, "seed": 1234,
                                   "max_runtime_s": max_runtime_s}},
              open(d / "aggregator_config.json", "w"))
    with open(d / "telemetry" / "aggregator_a.jsonl", "w") as fh:
        for b in range(n_bins):
            e = {"event": "agg_round", "round": 1, "ts": float(b * 10),
                 "cycle_data_id": b, "iteration_per_data_id": 0,
                 "var_good_enough": True, "var": 0.5,
                 "contributing_trainers": ["a"], "staleness": [0],
                 "agg_goal_count": agg_goal, "is_async": True,
                 "intrinsic_span_s": 20.0}
            if vclock:
                e["vclock_now"] = float(b * 10)
            fh.write(json.dumps(e) + "\n")
    return str(d)


def _pair(tmp_path, base, n_real=4, n_sim=4):
    _mk_run(tmp_path, f"run_20260101_000000_{base}_n10_smoke_syn_0_real", n_real)
    _mk_run(tmp_path, f"run_20260102_000000_{base}_n10_smoke_syn_0_sim", n_sim,
            vclock=True)


class TestDefaultJobs:
    def test_never_exceeds_the_pair_count(self):
        assert rp._default_jobs(1) == 1
        assert rp._default_jobs(3) <= 3

    def test_always_at_least_one(self):
        assert rp._default_jobs(0) >= 1

    def test_capped_by_available_ram(self, monkeypatch):
        # 9 GB available at ~4.5 GB/pair -> 2 workers, even with 100 pairs.
        meminfo = "MemTotal:  10000000 kB\nMemAvailable:  9000000 kB\n"
        real_open = open

        def fake_open(path, *a, **k):
            if str(path) == "/proc/meminfo":
                import io
                return io.StringIO(meminfo)
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)
        assert rp._default_jobs(100) == 2


class TestDiscovery:
    def test_finds_the_latest_pair_per_baseline(self, tmp_path):
        _pair(tmp_path, "felix_round")
        _mk_run(tmp_path, "run_20260103_000000_felix_round_n10_smoke_syn_0_sim",
                6, vclock=True)
        found = rp._discover(str(tmp_path))
        assert found[("felix_round", "syn_0")]["sim"][0] == "20260103_000000"

    def test_exact_baseline_token(self, tmp_path):
        _pair(tmp_path, "fwdllm")
        _pair(tmp_path, "fwdllm_plus")
        found = rp._discover(str(tmp_path))
        assert ("fwdllm", "syn_0") in found and ("fwdllm_plus", "syn_0") in found

    def test_skips_a_newer_real_whose_flag_differs(self, tmp_path):
        """An OFF control landing after the ON reals must not become the pair."""
        _mk_run(tmp_path, "run_20260101_000000_felix_it_n10_smoke_syn_0_real", jvp=True)
        _mk_run(tmp_path, "run_20260102_000000_felix_it_n10_smoke_syn_0_sim",
                vclock=True, jvp=True)
        _mk_run(tmp_path, "run_20260103_000000_felix_it_n10_smoke_syn_0_real", jvp=False)
        slot = rp._discover(str(tmp_path))[("felix_it", "syn_0")]
        assert slot["real"][0] == "20260101_000000"
        assert [s[0] for s in slot["_flag_skipped"]] == ["20260103_000000"]

    def test_takes_the_latest_real_when_the_flag_matches(self, tmp_path):
        _mk_run(tmp_path, "run_20260101_000000_felix_it_n10_smoke_syn_0_real", jvp=True)
        _mk_run(tmp_path, "run_20260102_000000_felix_it_n10_smoke_syn_0_sim",
                vclock=True, jvp=True)
        _mk_run(tmp_path, "run_20260103_000000_felix_it_n10_smoke_syn_0_real", jvp=True)
        slot = rp._discover(str(tmp_path))[("felix_it", "syn_0")]
        assert slot["real"][0] == "20260103_000000"
        assert slot["_flag_skipped"] == []

    def test_falls_back_to_latest_when_no_real_matches(self, tmp_path):
        """Never grade nothing: an unmatched flag still pairs, and says so."""
        _mk_run(tmp_path, "run_20260101_000000_felix_it_n10_smoke_syn_0_real", jvp=False)
        _mk_run(tmp_path, "run_20260102_000000_felix_it_n10_smoke_syn_0_sim",
                vclock=True, jvp=True)
        slot = rp._discover(str(tmp_path))[("felix_it", "syn_0")]
        assert slot["real"][0] == "20260101_000000" and slot["_flag"][0] is True


class TestParallelGrading:
    """Pairs are independent (own inputs, own output JSON, no shared state), so
    the parallel path must be observationally identical to the serial one."""

    @staticmethod
    def _run(tmp_path, capsys, jobs):
        rp.main(["--experiments-dir", str(tmp_path), "--yes", "--jobs", str(jobs),
                 "--baselines", "felix_round", "fedbuff_round", "fluxtune"])
        return capsys.readouterr().out

    def test_parallel_matches_serial_output(self, tmp_path, capsys):
        for b in ("felix_round", "fedbuff_round", "fluxtune"):
            _pair(tmp_path, b, n_real=5, n_sim=7)
        serial = self._run(tmp_path, capsys, 1)
        parallel = self._run(tmp_path, capsys, 3)
        # strip the worker-count banner, which is the one line that must differ
        strip = lambda s: "\n".join(l for l in s.split("\n") if "worker(s)" not in l)
        assert strip(serial) == strip(parallel)

    def test_results_stay_in_requested_order(self, tmp_path, capsys):
        for b in ("felix_round", "fedbuff_round", "fluxtune"):
            _pair(tmp_path, b)
        out = self._run(tmp_path, capsys, 3)
        order = [l.split("/")[0].strip() for l in out.split("\n")
                 if l.strip().startswith(("felix_round/", "fedbuff_round/",
                                          "fluxtune/")) and "pass /" in l]
        assert order == ["felix_round", "fedbuff_round", "fluxtune"]

    def test_each_pair_writes_its_own_report(self, tmp_path, capsys):
        for b in ("felix_round", "fedbuff_round"):
            _pair(tmp_path, b)
        self._run(tmp_path, capsys, 2)
        reports = os.listdir(tmp_path / "_parity_reports")
        assert len(reports) == 2 and len(set(reports)) == 2


class TestCoverageIsSurfaced:
    def test_budget_line_is_printed(self, tmp_path, capsys):
        _pair(tmp_path, "felix_round", n_real=5, n_sim=10)
        rp.main(["--experiments-dir", str(tmp_path), "--yes",
                 "--baselines", "felix_round"])
        out = capsys.readouterr().out
        assert "budget:" in out and "matched units grade" in out

    def test_low_coverage_is_called_out(self, tmp_path, capsys):
        # sim did 4x real's work -> 25% coverage, under the 50% floor.
        _pair(tmp_path, "felix_round", n_real=5, n_sim=20)
        rp.main(["--experiments-dir", str(tmp_path), "--yes",
                 "--baselines", "felix_round"])
        assert "LOW, windowed rungs unreliable" in capsys.readouterr().out


class TestControlIsSameModeBothSides:
    """The real<->real CONTROL (§D-55): a red rung is not evidence until the same
    comparison between config-identical legs comes back clean. It needs no sim
    leg, so it is the cheapest evidence in the pipeline."""

    @staticmethod
    def _leg(tmp_path, name, n_bins=4, jvp=True, max_runtime_s=7200, vclock=False):
        return _mk_run(tmp_path, name, n_bins=n_bins, jvp=jvp, vclock=vclock,
                       max_runtime_s=max_runtime_s)

    def test_every_pair_of_a_config_is_graded(self, tmp_path):
        for i in (1, 2, 3):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real")
        (key, kept, _dr, _cd, _ax), = rp._control_groups(str(tmp_path), ["felix_it"], "real")
        assert len(kept) == 3          # 3 legs -> 3 pairs

    def test_a_lone_leg_is_not_a_control(self, tmp_path):
        self._leg(tmp_path, "run_20260101_000000_felix_it_n10_smoke_syn_0_real")
        assert rp._control_groups(str(tmp_path), ["felix_it"], "real") == []

    def test_an_off_group_never_pools_with_an_on_one(self, tmp_path):
        """§D-45: an OFF leg is a different training config. Pooling its pairs
        into the roll-up would grade the flag, not the pipeline's noise."""
        for i in (1, 2):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real")
        for i in (3, 4):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real",
                      jvp=False)
        on = rp._control_groups(str(tmp_path), ["felix_it"], "real", jvp="on")
        assert len(on) == 1 and on[0][0][4] is True
        assert len(rp._control_groups(str(tmp_path), ["felix_it"], "real",
                                      jvp="any")) == 2

    def test_duration_selects_the_group(self, tmp_path):
        for i in (1, 2):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real")
        for i in (3, 4):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real",
                      max_runtime_s=14400)
        groups = rp._control_groups(str(tmp_path), ["felix_it"], "real",
                                    duration=14400)
        assert len(groups) == 1 and groups[0][0][3] == 14400

    def test_sim_mode_replicates_the_other_side(self, tmp_path):
        for i in (1, 2):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_sim",
                      vclock=True)
        assert len(rp._control_groups(str(tmp_path), ["felix_it"], "sim")) == 1
        assert rp._control_groups(str(tmp_path), ["felix_it"], "real") == []

    def test_it_grades_and_exits_zero(self, tmp_path, capsys):
        """A control REPORTS; it never gates. Exiting nonzero would truncate any
        `&&` chain it sits in (§D-51) for the normal outcome."""
        for i in (1, 2):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real",
                      n_bins=6)
        assert rp.main(["--experiments-dir", str(tmp_path), "--control", "--yes",
                        "--jobs", "1", "--baselines", "felix_it"]) == 0
        out = capsys.readouterr().out
        assert "CONTROL pairs" in out and "fails/pairs" in out

    def test_no_replicates_says_so(self, tmp_path, capsys):
        self._leg(tmp_path, "run_20260101_000000_felix_it_n10_smoke_syn_0_real")
        assert rp.main(["--experiments-dir", str(tmp_path), "--control", "--yes",
                        "--baselines", "felix_it"]) == 1
        assert "No replicate groups" in capsys.readouterr().out


class TestControlBailsAreNotFails:
    """§D-56: a rung reading a sim-only field cannot be graded with two real legs.
    It bails, and the bail is shaped exactly like a fail — counting it would put
    six permanently-red rungs in every control table."""

    def test_a_missing_vclock_stamp_reads_as_unreadable(self):
        assert rp._bailed_for_sim_field(
            {"ok": False, "note": "K10: no vclock_now in sim agg_round events"})
        assert rp._bailed_for_sim_field(
            {"ok": False, "issues": ["sim/0370: null sim_send_ts (missing)"]})

    def test_a_genuine_fail_is_not_a_bail(self):
        assert not rp._bailed_for_sim_field(
            {"ok": False, "mean_rel_diff": 0.14, "vclock_now": 123})
        assert not rp._bailed_for_sim_field({"ok": True, "note": "no vclock_now"})

    def test_bails_are_counted_apart_from_fails(self):
        done = [("b", {"throughput": {"ok": False, "note": "no vclock_now in sim"},
                       "v1_iter_per_data_id": {"ok": False, "tier": "DIST"},
                       "staleness": {"ok": True}}, "j", (1, 2, 0), [])]
        tally = rp._control_report(done, [("b@real a~b", "x", "y")])
        assert tally["throughput"] == {"fail": 0, "pairs": 0, "unreadable": 1,
                                       "skip": 0, "where": []}
        assert tally["v1_iter_per_data_id"]["fail"] == 1
        assert tally["staleness"]["fail"] == 0 and tally["staleness"]["pairs"] == 1

    def test_warn_only_rungs_do_not_count_as_fails(self):
        """The control must count a fail exactly as the scoreboard does, or the
        two tables disagree about the same rung."""
        done = [("b", {"inter_arrival_order": {"ok": False, "tier": "DIST"}},
                 "j", (0, 1, 0), [])]
        tally = rp._control_report(done, [("b@real a~b", "x", "y")],
                                   warn_only={"inter_arrival_order"})
        assert tally["inter_arrival_order"]["fail"] == 0


class TestPairsMustMatchOnRunLength:
    """A baseline can have real legs at two durations (fluxtune has 2h and 4h).
    The latest real is then the WRONG pair for a 2h sim leg — it grades the run
    length, and the budget coverage collapses to ~50%."""

    def test_skips_a_newer_real_of_a_different_duration(self, tmp_path):
        _mk_run(tmp_path, "run_20260101_000000_fluxtune_n10_smoke_syn_0_real",
                jvp=True, max_runtime_s=7200)
        _mk_run(tmp_path, "run_20260102_000000_fluxtune_n10_smoke_syn_0_sim",
                vclock=True, jvp=True, max_runtime_s=7200)
        _mk_run(tmp_path, "run_20260103_000000_fluxtune_n10_smoke_syn_0_real",
                jvp=True, max_runtime_s=14400)
        slot = rp._discover(str(tmp_path))[("fluxtune", "syn_0")]
        assert slot["real"][0] == "20260101_000000"
        assert [s[0] for s in slot["_flag_skipped"]] == ["20260103_000000"]

    def test_takes_the_latest_real_at_the_same_duration(self, tmp_path):
        _mk_run(tmp_path, "run_20260101_000000_fluxtune_n10_smoke_syn_0_real",
                jvp=True, max_runtime_s=7200)
        _mk_run(tmp_path, "run_20260102_000000_fluxtune_n10_smoke_syn_0_sim",
                vclock=True, jvp=True, max_runtime_s=7200)
        _mk_run(tmp_path, "run_20260103_000000_fluxtune_n10_smoke_syn_0_real",
                jvp=True, max_runtime_s=7200)
        slot = rp._discover(str(tmp_path))[("fluxtune", "syn_0")]
        assert slot["real"][0] == "20260103_000000"
        assert slot["_flag_skipped"] == []


class TestControlAndFloorAreTheSameMeasurement:
    """§D-65: a real↔real control and a replicate floor measure the same thing, so
    they must select the same legs. The control had no same-CODE filter while the
    floor did, and the two disagreed by 4.5x on `fedbuff_it`'s time-to-N — 13.6%
    against 3.0%. The gap was code drift (§D-70) reported as pipeline noise."""

    @staticmethod
    def _leg(tmp_path, name, sha):
        d = tmp_path / name
        (d / "telemetry").mkdir(parents=True)
        json.dump({"hyperparameters": {"seed": 1234, "max_runtime_s": 7200}},
                  open(d / "aggregator_config.json", "w"))
        (d / "snapshot.yaml").write_text(
            f"git_info:\n  commit: {sha}\n  clean: true\n", encoding="utf-8")
        (d / "x_trainers.log").write_text("jvp_eval_mode=True\n")
        with open(d / "telemetry" / "aggregator_x.jsonl", "w") as fh:
            for i in range(6):
                fh.write(json.dumps({"event": "agg_round", "round": 1, "ts": i * 100.0,
                                     "cycle_data_id": i, "iteration_per_data_id": 0,
                                     "var": 0.9, "var_good_enough": True}) + "\n")
        return str(d)

    def test_a_leg_on_other_code_is_not_a_control_replicate(self, tmp_path):
        for i, sha in ((1, "aaaaaaaaa"), (2, "aaaaaaaaa"), (3, "bbbbbbbbb")):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real", sha)
        (_key, kept, _dr, code_dropped, _ax), = rp._control_groups(
            str(tmp_path), ["felix_it"], "real")
        assert len(kept) == 2 and len(code_dropped) == 1
        assert code_dropped[0][0] == "20260103_000000"

    def test_any_code_pools_them_back(self, tmp_path):
        for i, sha in ((1, "aaaaaaaaa"), (2, "aaaaaaaaa"), (3, "bbbbbbbbb")):
            self._leg(tmp_path, f"run_2026010{i}_000000_felix_it_n10_smoke_syn_0_real", sha)
        (_key, kept, _dr, code_dropped, _ax), = rp._control_groups(
            str(tmp_path), ["felix_it"], "real", any_code=True)
        assert len(kept) == 3 and not code_dropped


class TestFloorGateIsTwoSided:
    """A real↔sim residual draws one leg from each side, so a gate sized on the
    real floor alone assumes sim is deterministic (§D-61). It is not: measured
    same-code at n=3, `fedbuff_round` reproduces to 0.7% real and 21.2% sim."""

    def _profile(self, tmp_path, monkeypatch, **prof):
        monkeypatch.setattr(rp, "_FLOOR_DIR", tmp_path)
        (tmp_path / "b.yaml").write_text(yaml.safe_dump(prof))
        return rp._floors("b/syn_0")

    def test_the_wider_side_sets_the_floor(self, tmp_path, monkeypatch):
        f = self._profile(tmp_path, monkeypatch,
                          metrics={"iters_per_bin": 0.007},
                          sim_metrics={"iters_per_bin": 0.212})
        assert f["iters_per_bin"] == pytest.approx(0.212)

    def test_real_still_wins_where_it_is_the_wider_side(self, tmp_path, monkeypatch):
        f = self._profile(tmp_path, monkeypatch,
                          metrics={"iters_per_bin": 0.159},
                          sim_metrics={"iters_per_bin": 0.082})
        assert f["iters_per_bin"] == pytest.approx(0.159)

    def test_a_profile_with_no_sim_side_grades_exactly_as_before(
            self, tmp_path, monkeypatch):
        # Every baseline outside the run batch is still real-only; none may move.
        f = self._profile(tmp_path, monkeypatch, metrics={"iters_per_bin": 0.03})
        assert f == {"iters_per_bin": 0.03}

    def test_a_metric_measured_on_one_side_only_is_not_dropped(
            self, tmp_path, monkeypatch):
        f = self._profile(tmp_path, monkeypatch,
                          metrics={"iters_per_bin": 0.03},
                          sim_metrics={"mean_var": 0.09})
        assert f == {"iters_per_bin": 0.03, "mean_var": 0.09}

    def test_no_metrics_at_all_reads_as_no_floor(self, tmp_path, monkeypatch):
        assert self._profile(tmp_path, monkeypatch, max_runtime_s=7200) is None
