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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_parity as rp  # noqa: E402


def _mk_run(tmp_path, name, n_bins=4, agg_goal=2, vclock=False, jvp=None):
    d = tmp_path / name
    (d / "telemetry").mkdir(parents=True)
    if jvp is not None:                       # the knob lives only in the trainer log
        (d / "x_trainers.log").write_text(f"[JVP_EVAL_MODE] jvp_eval_mode={jvp}\n")
    json.dump({"hyperparameters": {"aggGoal": agg_goal, "seed": 1234}},
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
        assert slot["real"][0] == "20260101_000000" and slot["_flag"] is True


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
