# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical real/sim parity checks (parity_checks.py).

Runs in the default suite on synthetic telemetry — no MQTT/GPU — so the parity
logic itself is verified independently of any live run. The opt-in end-to-end
check (test_real_sim_e2e_parity.py) reuses the same functions on real runs.
"""

import math
import pathlib
import sys

import pytest

# parity_checks lives with the async_cifar10 example scripts.
_SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples" / "async_cifar10" / "scripts"
)
sys.path.insert(0, str(_SCRIPTS))
import parity_checks as pc  # noqa: E402


def _agg(selection=None, agg_rounds=None, agg_evals=None):
    return {
        "selection_train": selection or [],
        "agg_rounds": agg_rounds or [],
        "agg_evals": agg_evals or [],
    }


def _sel(round_, chosen, ts=0.0):
    return {"event": "selection", "task": "train", "round": round_,
            "ts": ts, "chosen": chosen}


def _round(round_, contributing, staleness, agg_goal_count=1, vclock=None, ts=0.0):
    e = {"event": "agg_round", "round": round_, "ts": ts,
         "contributing_trainers": contributing, "staleness": staleness,
         "agg_goal_count": agg_goal_count}
    if vclock is not None:
        e["vclock_now"] = vclock
    return e


class TestSelectionParity:
    def test_identical_is_exact(self):
        a = _agg(selection=[_sel(1, ["x", "y"]), _sel(2, ["y", "z"])])
        r = pc.selection_parity(a, a)
        assert r["ok"] and r["mean_jaccard"] == 1.0 and r["exact_match_frac"] == 1.0

    def test_disjoint_fails(self):
        real = _agg(selection=[_sel(1, ["a", "b"])])
        sim = _agg(selection=[_sel(1, ["c", "d"])])
        r = pc.selection_parity(real, sim)
        assert not r["ok"] and r["mean_jaccard"] == 0.0


class TestStalenessParity:
    def test_matching_ok(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0]), _round(1, ["b"], [1])])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0]), _round(1, ["b"], [1])])
        r = pc.staleness_parity(real, sim)
        assert r["ok"] and r["real_mean"] == r["sim_mean"]

    def test_negative_staleness_fails(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0])])
        sim = _agg(agg_rounds=[_round(1, ["a"], [-1])])  # impossible in either mode
        r = pc.staleness_parity(real, sim)
        assert not r["ok"] and r["all_nonnegative"] is False


class TestSimSendTs:
    def test_real_null_sim_increasing_ok(self):
        real_tr = {"aa": {"task_recv": [{"round": 2, "sim_send_ts": None}]}}
        sim_tr = {"aa": {"task_recv": [
            {"round": 2, "sim_send_ts": 5.0}, {"round": 3, "sim_send_ts": 11.0}]}}
        assert pc.sim_send_ts_ok(real_tr, sim_tr)["ok"]

    def test_sim_null_fails(self):
        sim_tr = {"aa": {"task_recv": [{"round": 2, "sim_send_ts": None}]}}
        assert not pc.sim_send_ts_ok({}, sim_tr)["ok"]


class TestGpuBudget:
    def test_within_budget_ok(self):
        tr = {"aa": {"trainer_round": [
            {"real_gpu_time_s": 1.0, "training_budget_s": 5.0},
            {"real_gpu_time_s": 2.0, "training_budget_s": 5.0}]}}
        assert pc.gpu_budget_ok(tr)["ok"]

    def test_overrun_fails(self):
        tr = {"aa": {"trainer_round": [
            {"real_gpu_time_s": 9.0, "training_budget_s": 5.0},
            {"real_gpu_time_s": 8.0, "training_budget_s": 5.0}]}}
        r = pc.gpu_budget_ok(tr)
        assert not r["ok"] and r["mean_overrun_frac"] == 1.0


class TestSimInvariants:
    def test_commit_order_monotone(self):
        ok = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=5.0),
                              _round(1, ["b"], [0], vclock=10.0)])
        assert pc.sim_commit_order_monotone(ok)["ok"]
        bad = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=10.0),
                               _round(1, ["b"], [0], vclock=5.0)])
        assert not pc.sim_commit_order_monotone(bad)["ok"]

    def test_agg_goal_cycles(self):
        ok = _agg(agg_rounds=[_round(1, ["a"], [0], agg_goal_count=1),
                              _round(1, ["b"], [0], agg_goal_count=2)])
        assert pc.agg_goal_cycles_ok(ok, agg_goal=2)["ok"]
        bad = _agg(agg_rounds=[_round(1, ["a"], [0], agg_goal_count=3)])
        assert not pc.agg_goal_cycles_ok(bad, agg_goal=2)["ok"]


class TestLogicalSequence:
    def test_commit_sequence_order(self):
        a = _agg(agg_rounds=[
            _round(2, ["b"], [1], agg_goal_count=2),
            _round(2, ["a"], [0], agg_goal_count=1),  # earlier in logical order
        ])
        seq = pc.commit_sequence(a)
        assert [s["end"] for s in seq] == ["a", "b"]  # sorted by agg_goal_count

    def test_first_divergence_localizes(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0], 1), _round(1, ["b"], [0], 2)])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0], 1), _round(1, ["c"], [5], 2)])
        d = pc.first_divergence(real, sim)
        assert d["index"] == 1  # first mismatch at the 2nd committed update

    def test_first_divergence_none_when_equal(self):
        a = _agg(agg_rounds=[_round(1, ["a"], [0], 1), _round(1, ["b"], [0], 2)])
        assert pc.first_divergence(a, a)["index"] is None


def test_run_all_parity_smoke():
    a = _agg(selection=[_sel(1, ["a", "b"])],
             agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])
    tr = {"aa": {"task_recv": [], "trainer_round": []}}
    res = pc.run_all_parity(a, a, tr, tr, agg_goal=2)
    # Identical inputs ⇒ no parity check fails. field_coverage may flag the
    # deliberately-sparse fixture's missing telemetry fields (a coverage signal,
    # not a parity divergence), and data-less checks SKIP — both are excluded.
    for name, v in res.items():
        if name == "field_coverage" or v.get("status") == "SKIP":
            continue
        assert v["ok"], f"{name}: {v}"


# ── §3.H clock / throughput new checks ──────────────────────────────────────

def _round_speed(round_, contributing, staleness, agg_goal_count=1,
                 vclock=None, ts=0.0, speed=None):
    """Helper: agg_round event with optional vclock_now and trainer_speed_s."""
    e = _round(round_, contributing, staleness, agg_goal_count, vclock, ts)
    if speed is not None:
        e["trainer_speed_s"] = speed if isinstance(speed, list) else [speed]
    return e


class TestVclockTelemetryPresent:
    def test_present_passes(self):
        sim = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=10.0)])
        r = pc.vclock_telemetry_present(sim)
        assert r["ok"]

    def test_absent_fails(self):
        sim = _agg(agg_rounds=[_round(1, ["a"], [0])])  # no vclock_now
        r = pc.vclock_telemetry_present(sim)
        assert not r["ok"] and "ZERO vclock_now" in r["note"]


class TestThroughputParity:
    def _make_matched_pair(self):
        """Build real + sim where both do ~10 rounds in 100s of their respective time."""
        # real: 10 rounds, wall 0..100 s (10 s/round)
        real_rounds = [
            _round(r, ["a"], [0], ts=float(r * 10))
            for r in range(1, 11)
        ]
        real = _agg(agg_rounds=real_rounds)
        # sim: 10 rounds, vclock 0..100 s (10 s/round virtual)
        sim_rounds = [
            _round(r, ["a"], [0], vclock=float(r * 10), ts=float(r * 1))
            for r in range(1, 11)
        ]
        sim = _agg(agg_rounds=sim_rounds)
        return real, sim

    def test_matched_passes(self):
        real, sim = self._make_matched_pair()
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        assert r["ok"], r

    def test_diverged_fails(self):
        """Motivating case: sim does 41 rounds in 410 s vclock; real does 67 in 670 s wall."""
        # sim: 41 rounds, 26 s/round virtual → 41/1066 throughput
        sim_rounds = [_round(r, ["a"], [0], vclock=float(r * 26), ts=float(r))
                      for r in range(1, 42)]
        # real: 67 rounds, 15.6 s/round wall → 67/1045 throughput
        real_rounds = [_round(r, ["a"], [0], ts=float(r * 15.6))
                       for r in range(1, 68)]
        real = _agg(agg_rounds=real_rounds)
        sim = _agg(agg_rounds=sim_rounds)
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        assert not r["ok"], r

    def test_no_vclock_fails(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0], ts=10.0)])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0])])  # no vclock
        r = pc.throughput_parity(real, sim)
        assert not r["ok"] and "K10" in r.get("note", "")


class TestIntrinsicSpanAnchor:
    """#6: clock-rate rungs anchor REAL on `intrinsic_span_s` (barrier+eval)
    instead of raw wall ts (which bundles a transport artifact the sim omits).
    With vs without the field must flip the verdict; async_cifar10 (no field)
    stays byte-identical via the raw-ts fallback."""

    def _pair(self, with_intrinsic):
        # 10 data_ids: 20 s/data_id genuine work; REAL wall adds +15 s/data_id
        # transport (35/unit), but intrinsic_span_s reports the clean 20.
        real_rounds, sim_rounds = [], []
        for d in range(1, 11):
            re = {"event": "agg_round", "round": 1, "ts": float(d * 35),
                  "data_id": d, "cycle_data_id": d,
                  "contributing_trainers": ["a"], "staleness": [0],
                  "agg_goal_count": 1}
            if with_intrinsic:
                re["intrinsic_span_s"] = 20.0
            real_rounds.append(re)
            sim_rounds.append({"event": "agg_round", "round": 1, "ts": float(d),
                               "vclock_now": float(d * 20), "data_id": d,
                               "cycle_data_id": d, "contributing_trainers": ["a"],
                               "staleness": [0], "agg_goal_count": 1,
                               "intrinsic_span_s": 20.0})
        return _agg(agg_rounds=real_rounds), _agg(agg_rounds=sim_rounds)

    def test_intrinsic_anchor_passes(self):
        real, sim = self._pair(with_intrinsic=True)
        assert pc.throughput_parity(real, sim, tol_rel=0.05)["ok"]
        assert pc.per_round_advance_parity(real, sim)["ok"]
        assert pc.total_commits_parity(real, sim)["ok"]
        assert pc.terminal_state_parity(real, sim)["ok"]
        wd = pc.wall_disparity(real, sim)
        assert wd["anchor"] == "intrinsic_span"
        assert wd["mean_abs_disparity_s"] < 1.0, wd

    def test_raw_wall_fallback_fails_and_is_byte_identical(self):
        # Field absent -> real anchors on raw ts (35/unit) vs sim vclock
        # (20/unit): the #6 gap. Also the async_cifar10 path.
        real, sim = self._pair(with_intrinsic=False)
        assert not pc.throughput_parity(real, sim, tol_rel=0.05)["ok"]
        assert not pc.per_round_advance_parity(real, sim)["ok"]
        wd = pc.wall_disparity(real, sim)
        assert wd["anchor"] == "wall_ts"
        assert wd["max_abs_disparity_s"] > 100.0, wd  # 15 s/unit artifact, cumulative


class TestIntrinsicSpanAsyncOverlap:
    """fluxtune's async cycles OVERLAP in real wall-time (multiple cohorts
    commit concurrently, unlike sync's one round in flight), so
    cumulative-summing each cycle's own intrinsic_span_s as if sequential
    races far ahead of raw wall. `is_async` must force the raw-wall fallback
    (same as async_cifar10, which never emits intrinsic_span_s), not the
    sync cumulative-sum anchor."""

    def _pair(self, is_async):
        # 100 commits, each an 80s barrier+eval span, but real cycles OVERLAP
        # ~4x in wall time (4 concurrent trainers) so raw wall only advances
        # 20s/commit -- matching sim's vclock 1:1. (100 not 10: keeps the
        # boundary rounding edge <=1%, well under the 5% tol.)
        real_rounds, sim_rounds = [], []
        for d in range(1, 101):
            real_rounds.append({
                "event": "agg_round", "round": 1, "ts": float(d * 20),
                "data_id": d, "cycle_data_id": d,
                "contributing_trainers": ["a"], "staleness": [0],
                "agg_goal_count": 1, "intrinsic_span_s": 80.0,
                "is_async": is_async,
            })
            sim_rounds.append({
                "event": "agg_round", "round": 1, "ts": float(d),
                "vclock_now": float(d * 20), "data_id": d,
                "cycle_data_id": d, "contributing_trainers": ["a"],
                "staleness": [0], "agg_goal_count": 1,
                "intrinsic_span_s": 80.0, "is_async": is_async,
            })
        return _agg(agg_rounds=real_rounds), _agg(agg_rounds=sim_rounds)

    def test_async_falls_back_to_raw_wall(self):
        real, sim = self._pair(is_async=True)
        # raw-wall-anchored real (20s/commit) matches sim's vclock (20s/commit)
        r = pc.total_commits_parity(real, sim, tol_rel=0.05)
        assert r["ok"], r
        t = pc.throughput_parity(real, sim, tol_rel=0.05)
        assert t["ok"], t

    def test_sync_still_uses_cumulative_intrinsic_sum(self):
        # Same synthetic overlap, but is_async=False: cycles aren't supposed
        # to overlap for sync in the first place, so the cumulative sum
        # (which races to 80s/commit vs sim's 20s/commit vclock) correctly
        # flags this as a real divergence rather than silently masking it.
        real, sim = self._pair(is_async=False)
        r = pc.total_commits_parity(real, sim, tol_rel=0.05)
        assert not r["ok"], r


class TestSimSpeedup:
    """sim_speedup [DIAG] (#13): sim must run virtual time at least as fast as
    wall (sim_rate >= 1). K7's sane-range [0.01,100] check passes a slowdown;
    this rung catches it."""

    def test_healthy_speedup_passes(self):
        # sim: vclock 0..100 virtual-s in 0..10 wall-s (10x speedup);
        # real: same work took 0..100 wall-s.
        sim = _agg(agg_rounds=[
            _round(r, ["a"], [0], vclock=float(r * 10), ts=float(r))
            for r in range(1, 11)])
        real = _agg(agg_rounds=[
            _round(r, ["a"], [0], ts=float(r * 10)) for r in range(1, 11)])
        r = pc.sim_speedup(real, sim)
        assert r["ok"] and r["is_speedup"], r
        assert r["sim_rate"] >= 1.0
        assert r["wall_speedup"] > 1.0  # sim finished faster than real

    def test_slowdown_fails(self):
        # Slowdown shape: vclock reaches ~213 while wall burns ~566
        # (sim_rate ~0.38 < 1) -> sim is broken (#13).
        sim = _agg(agg_rounds=[
            _round(r, ["a"], [0], vclock=float(r * 213.0 / 24),
                   ts=float(r * 566.0 / 24))
            for r in range(1, 25)])
        real = _agg(agg_rounds=[
            _round(r, ["a"], [0], ts=float(r * 558.0 / 14))
            for r in range(1, 15)])
        r = pc.sim_speedup(real, sim)
        assert not r["ok"] and not r["is_speedup"], r
        assert r["sim_rate"] < 1.0
        assert "SLOWDOWN" in r["note"]

    def test_prefers_wall_elapsed_s_over_ts_span(self):
        # When agg_round carries wall_elapsed_s (the re-anchored measure), it is
        # used instead of the ts epoch span.
        sim_rounds = [
            _round(r, ["a"], [0], vclock=float(r * 10), ts=float(r * 1000 + r))
            for r in range(1, 11)]
        for e in sim_rounds:
            e["wall_elapsed_s"] = float(e["round"])  # 1..10, unlike the ts span
        sim = _agg(agg_rounds=sim_rounds)
        real = _agg(agg_rounds=[
            _round(r, ["a"], [0], ts=float(r * 10)) for r in range(1, 11)])
        r = pc.sim_speedup(real, sim)
        assert r["sim_wall_s"] == 10.0  # from wall_elapsed_s, not the huge ts span
        assert r["sim_rate"] >= 1.0

    def test_no_vclock_skips(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0], ts=10.0)])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0], ts=5.0)])  # no vclock
        r = pc.sim_speedup(real, sim)
        assert r.get("status") == "SKIP"


class TestPerRoundAdvanceParity:
    def test_matched_passes(self):
        # Both advance 10s per round
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 10), ts=float(r))
                                 for r in range(1, 11)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert r["ok"], r

    def test_diverged_fails(self):
        # sim: 26 s/round vclock; real: 15 s/round wall → 73% relative diff
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 15))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 26), ts=float(r))
                                 for r in range(1, 11)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert not r["ok"], r


class TestPerRoundAdvanceCentralEscape:
    """The matched-window central-tendency escape (real_coord branch only,
    i.e. sync + `intrinsic_span_s`): sim's Δvclock is whole-second quantized
    while real's Δwall spreads continuously, so a thin shape-only tail (e.g.
    a cold round-1 GPU warmup) can trip grid-KS even when the mean AND the
    per-round ratio MEDIAN both match -- must still pass. A genuine advance
    divergence moves the median/mean too and must still fail."""

    def _pair(self, n_spiked):
        # 20 advances, all real=10.0; sim=10.0 except `n_spiked` of them at
        # 11.4 (a 14% tail on those rounds only) -- large enough a FRACTION
        # to trip grid-KS but small enough in magnitude to keep mean+median
        # within band.
        real_rounds, sim_rounds = [], []
        for r in range(1, 22):
            real_rounds.append({"event": "agg_round", "round": r,
                                "ts": float(r * 10), "intrinsic_span_s": 10.0,
                                "contributing_trainers": ["a"], "staleness": [0],
                                "agg_goal_count": 1})
            spiked = r > (21 - n_spiked)
            step = 11.4 if spiked else 10.0
            sim_rounds.append({"event": "agg_round", "round": r, "ts": float(r),
                               "vclock_now": None,  # filled below
                               "contributing_trainers": ["a"], "staleness": [0],
                               "agg_goal_count": 1})
        # cumulative sim vclock from the per-round steps above
        run = 0.0
        for i, e in enumerate(sim_rounds):
            spiked = (i + 1) > (21 - n_spiked)
            run += 11.4 if spiked else 10.0
            e["vclock_now"] = run
        return _agg(agg_rounds=real_rounds), _agg(agg_rounds=sim_rounds)

    def test_thin_tail_fails_grid_ks_but_escapes_on_median_and_mean(self):
        real, sim = self._pair(n_spiked=5)   # 5/20 = 25% of rounds spiked
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert r["matched_window_ks_stat"] > 0.2           # grid-KS alone fails
        assert r["central_escape_ok"] is True
        assert r["ok"], r

    def test_systemic_shift_fails_even_with_escape_available(self):
        # ALL rounds spiked -> the median itself moves, so the escape's own
        # gate (ratio_med within band) correctly refuses to fire.
        real, sim = self._pair(n_spiked=21)
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.05)
        assert r["central_escape_ok"] is False
        assert not r["ok"], r


class TestOverlapFactor:
    def test_matched_passes(self):
        # Both: speed=20s, advance=15s → overlap ≈ 1.33 for both
        real = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], ts=float(r * 15), speed=20.0)
            for r in range(1, 11)
        ])
        sim = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], vclock=float(r * 15), ts=float(r), speed=20.0)
            for r in range(1, 11)
        ])
        r = pc.overlap_factor(real, sim, tol=0.3)
        assert r["ok"], r

    def test_no_overlap_sim_fails(self):
        # sim: speed≈advance (1.0 overlap); real: speed=20, advance=12 (1.67 overlap)
        real = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], ts=float(r * 12), speed=20.0)
            for r in range(1, 11)
        ])
        sim = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], vclock=float(r * 20), ts=float(r), speed=20.0)
            for r in range(1, 11)
        ])
        r = pc.overlap_factor(real, sim, tol=0.3)
        assert not r["ok"], r


class TestTotalCommitsParity:
    def test_matched_passes(self):
        # Both: 5 commits spanning 0-40s of their respective time
        # real: ts = 0,10,20,30,40  (wall_elapsed=40); sim: vclock = 0,10,20,30,40
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float((r - 1) * 10))
                                  for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float((r - 1) * 10),
                                       ts=float(r))
                                 for r in range(1, 6)])
        r = pc.total_commits_parity(real, sim, tol_rel=0.05)
        assert r["ok"], r

    def test_diverged_fails(self):
        # sim: 3 commits, vclock 0,10,20 → final_vclock=20
        # real: 6 commits at ts 0,5,10,15,20,25 → wall_elapsed=25, V=min(20,25)=20
        # n_sim: vclock ≤ 20 → 3; n_real: ts-t0=0,5,10,15,20,25 ≤ 20 → 5 → diverged
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float((r - 1) * 5))
                                  for r in range(1, 7)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float((r - 1) * 10),
                                       ts=float(r))
                                 for r in range(1, 4)])
        r = pc.total_commits_parity(real, sim, tol_rel=0.02)
        assert not r["ok"], r


class TestTerminalStateParity:
    def test_matched_passes(self):
        # Both modes: 10 rounds on a normalized 0..90s timeline → rel_diff 0.
        # (real wall normalizes by t0=min(ts); sim vclock is used directly, so the
        # sim vclocks must span 0..90 to match real's normalized 0..90 — otherwise
        # the V=min cutoff drops sim's last round and yields a spurious 10% edge.)
        real = _agg(agg_rounds=[_round(r, ["a", "b"], [0, 0], ts=float(r * 10))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a", "b"], [0, 0], vclock=float((r - 1) * 10),
                                       ts=float(r))
                                 for r in range(1, 11)])
        r = pc.terminal_state_parity(real, sim)
        assert r["ok"], r

    def test_diverged_fails(self):
        # sim: 5 rounds in 130s vclock; real: 10 rounds in 100s wall
        # V = min(130, 100) = 100; sim has 4 rounds ≤100, real has 10
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 26),
                                       ts=float(r))
                                 for r in range(1, 6)])
        r = pc.terminal_state_parity(real, sim, rounds_tol=0.10)
        assert not r["ok"], r


def _fwd_cycle(data_id, ts, vclock=None, round_=1, committed=True,
               barrier=None, var=None, iteration=0):
    """fwdllm agg_round with the fields the clock/variance rungs read."""
    e = {"event": "agg_round", "round": round_, "ts": ts,
         "cycle_data_id": data_id, "var_good_enough": committed,
         "contributing_trainers": ["a"], "staleness": [0], "agg_goal_count": 1,
         "iteration_per_data_id": iteration, "is_async": True}
    if vclock is not None:
        e["vclock_now"] = vclock
    if barrier is not None:
        e["intrinsic_span_s"] = barrier
    if var is not None:
        e["var"] = var
    return e


class TestMatchedBudgetLapWrap:
    """`round` bumps one bin BEFORE `cycle_data_id` wraps, so a real lap runs
    (1,148) -> (2,149) -> (2,0) -> (2,1). The tuple key is therefore NOT monotone
    and a max-key ceiling both mis-orders progress and admits work only one side
    did (felix_round: 9 bins sim committed and real never did, under a budget the
    checker called matched)."""

    @staticmethod
    def _lap(n_tail, ts0=0.0, vclock=False):
        """Bins (1,8),(1,9), then the wrap (2,10),(2,0),(2,1)... n_tail deep."""
        seq = [(1, 8), (1, 9), (2, 10)] + [(2, i) for i in range(n_tail)]
        out = []
        for i, (rd, did) in enumerate(seq):
            t = ts0 + i * 10.0
            out.append(_fwd_cycle(did, ts=t, round_=rd,
                                  vclock=(t if vclock else None)))
        return out

    def test_budget_counts_only_bins_both_sides_committed(self):
        # sim ran 4 bins deeper into lap 2 than real. Old max-key ceiling was
        # (2,10) on BOTH sides -- which sorts above every (2,0..3) -- so `<= N`
        # admitted sim's extra bins and graded two different workloads.
        real = _agg(agg_rounds=self._lap(2))
        sim = _agg(agg_rounds=self._lap(6, vclock=True))
        N, prog_fn = pc._matched_logical_budget(real["agg_rounds"], sim["agg_rounds"])
        assert N == 5, N                      # (1,8),(1,9),(2,10),(2,0),(2,1)
        n_sim_in = sum(1 for e in sim["agg_rounds"]
                       if (p := prog_fn(e)) is not None and p <= N)
        n_real_in = sum(1 for e in real["agg_rounds"]
                        if (p := prog_fn(e)) is not None and p <= N)
        assert n_sim_in == n_real_in == 5      # same work on both sides

    def test_ordinals_follow_time_not_key_sort(self):
        # (2,10) happens BEFORE (2,0) but sorts after it. The ordinal must be
        # chronological, or every windowed rung mis-attributes the lap boundary.
        real = _agg(agg_rounds=self._lap(2))
        sim = _agg(agg_rounds=self._lap(2, vclock=True))
        _, prog_fn = pc._matched_logical_budget(real["agg_rounds"], sim["agg_rounds"])
        by_key = {(e["round"], e["cycle_data_id"]): prog_fn(e)
                  for e in real["agg_rounds"]}
        assert by_key[(2, 10)] == 3 and by_key[(2, 0)] == 4 and by_key[(2, 1)] == 5

    def test_time_to_n_stops_at_the_shared_prefix(self):
        # The regression this fixes: sim's clock was read at ITS deadline rather
        # than at the last shared bin, so `total_commits` compared full run
        # lengths and called a 4.7% deadline artifact a throughput measurement.
        real = _agg(agg_rounds=self._lap(2))
        sim = _agg(agg_rounds=self._lap(6, vclock=True))
        r = pc.total_commits_parity(real, sim)
        assert r["matched_logical_budget_n"] == 5
        assert r["sim_vclock_to_n_s"] == 40.0     # bin 5 of 5, not sim's last bin
        assert r["ok"], r

    def test_divergent_bin_order_truncates_the_budget(self):
        # If the two sides commit DIFFERENT bins, the budget must stop at the
        # first disagreement rather than pretend the suffix is comparable.
        real = _agg(agg_rounds=[_fwd_cycle(d, ts=float(d * 10)) for d in range(6)])
        sim = _agg(agg_rounds=[_fwd_cycle(d if d < 3 else d + 7, ts=float(d * 10),
                                          vclock=float(d * 10)) for d in range(6)])
        N, _ = pc._matched_logical_budget(real["agg_rounds"], sim["agg_rounds"])
        assert N == 3


class TestOverlapFactorPipeliningDepth:
    """K4 grades pipelining depth = per-cycle barrier / per-cycle clock advance.
    Its old numerator (`_per_round_max_speed`) keyed on FL `round`, which fwdllm
    holds static for a whole lap -- so it returned the run-global max trainer,
    identical on both sides, and the rung silently restated its own denominator."""

    @staticmethod
    def _run(n, barrier, adv, vclock=False):
        return _agg(agg_rounds=[
            _fwd_cycle(d, ts=d * adv, vclock=(d * adv if vclock else None),
                       barrier=barrier)
            for d in range(n)])

    def test_matched_pipelining_passes(self):
        r = pc.overlap_factor(self._run(20, 20.0, 4.0),
                              self._run(20, 20.0, 4.0, vclock=True))
        assert r["ok"], r
        assert r["real_overlap_factor"] == pytest.approx(5.0, rel=0.1)

    def test_sim_serializes_what_real_pipelines_fails(self):
        # Identical barriers; sim advances its clock 25% more per cycle. This is
        # fluxtune's signature (real 5.09 vs sim 3.95) and no vclock charge fixes
        # it -- charging more moves sim further the wrong way.
        r = pc.overlap_factor(self._run(20, 20.0, 4.0),
                              self._run(20, 20.0, 5.0, vclock=True))
        assert not r["ok"], r
        assert r["sim_overlap_factor"] < r["real_overlap_factor"]

    def test_numerator_is_per_cycle_not_run_global(self):
        # Static `round` + a per-cycle barrier that differs between modes must be
        # SEEN. Under the old per-round max both sides read the same global max.
        real = self._run(20, 30.0, 4.0)
        sim = self._run(20, 15.0, 4.0, vclock=True)
        r = pc.overlap_factor(real, sim)
        assert r["real_mean_barrier_s"] == pytest.approx(30.0)
        assert r["sim_mean_barrier_s"] == pytest.approx(15.0)
        assert not r["ok"], r

    def test_gates_as_exact_not_diag(self):
        assert pc.CHECK_META["overlap_factor"]["role"] == "MECHANISM"
        assert "overlap_factor" in pc.CHECK_META["overhead_residual"]["deps"]


class TestVarDriftParity:
    """V2b routes the V2 investigation: a flat offset is a per-cycle mechanism;
    a monotone drift means the two models are on diverging trajectories and the
    per-cycle hunt is the wrong one."""

    @staticmethod
    def _pair(var_fn_real, var_fn_sim, n=60):
        real = _agg(agg_rounds=[_fwd_cycle(d, ts=float(d), var=var_fn_real(d))
                                for d in range(n)])
        sim = _agg(agg_rounds=[_fwd_cycle(d, ts=float(d), vclock=float(d),
                                          var=var_fn_sim(d)) for d in range(n)])
        return real, sim

    def test_flat_ratio_reads_as_level_offset(self):
        real, sim = self._pair(lambda d: 1.0, lambda d: 0.9)
        r = pc.var_drift_parity(real, sim)
        assert r["verdict"] == "level_offset"
        assert r["ok"], r                     # a level offset is not "drift"
        assert r["mean_ratio"] == pytest.approx(0.9, rel=0.02)

    def test_progressive_divergence_is_flagged(self):
        # real's var climbs, sim's stays flat -- fedbuff_round's signature.
        real, sim = self._pair(lambda d: 1.0 + d * 0.03, lambda d: 1.0)
        r = pc.var_drift_parity(real, sim)
        assert r["verdict"] == "progressive_drift"
        assert not r["ok"], r
        assert r["last_third_ratio"] < r["first_third_ratio"]
        assert abs(r["trend_rho"]) >= 0.6

    def test_identical_reads_flat(self):
        real, sim = self._pair(lambda d: 1.0 + (d % 3) * 0.1,
                               lambda d: 1.0 + (d % 3) * 0.1)
        r = pc.var_drift_parity(real, sim)
        assert r["verdict"] == "flat" and r["ok"]

    def test_never_gates_the_verdict(self):
        assert pc.CHECK_META["v2b_var_drift"]["role"] == "DIAG"


class TestMatchedBudgetCoverage:
    """8 of the 88 rungs window on `_matched_logical_budget`. Their numbers are
    only as trustworthy as the fraction of the run the budget covers, so the
    fraction is graded once and stamped on every one of them."""

    @staticmethod
    def _run(bins, ts0=0.0, vclock=False):
        return _agg(agg_rounds=[
            _fwd_cycle(b, ts=ts0 + i * 10.0,
                       vclock=(ts0 + i * 10.0 if vclock else None))
            for i, b in enumerate(bins)])

    def test_equal_runs_are_full_coverage(self):
        r = pc.matched_budget_coverage_parity(
            self._run(range(10)), self._run(range(10), vclock=True))
        assert r["ok"] and r["truncation"] == "none"
        assert r["real_coverage"] == 1.0 and r["sim_coverage"] == 1.0

    def test_overrun_truncates_only_the_faster_side(self):
        # Parity fixes the WORK: real's 10 bins are fully graded, sim's extra 5
        # have no counterpart and are dropped. The slower side is always 100%.
        r = pc.matched_budget_coverage_parity(
            self._run(range(10)), self._run(range(15), vclock=True))
        assert r["ok"] and r["truncation"] == "overrun"
        assert r["real_coverage"] == 1.0
        assert r["sim_coverage"] == pytest.approx(10 / 15, abs=1e-3)

    def test_sequence_divergence_is_a_hard_fail_not_a_windowing_artifact(self):
        # Same COUNT on both sides, but they committed different bins from
        # position 4 on -- the budget stops early for a reason no amount of
        # running longer would fix.
        real = self._run(list(range(10)))
        sim = self._run([0, 1, 2, 3, 40, 41, 42, 43, 44, 45], vclock=True)
        r = pc.matched_budget_coverage_parity(real, sim)
        assert not r["ok"]
        assert r["truncation"] == "sequence_divergence"
        assert r["first_divergence"]["position"] == 5
        assert r["first_divergence"]["real_unit"] != r["first_divergence"]["sim_unit"]

    def test_fails_when_coverage_falls_below_the_floor(self):
        # sim did 3x real's work: the windowed rungs would grade a third of it.
        r = pc.matched_budget_coverage_parity(
            self._run(range(10)), self._run(range(30), vclock=True))
        assert not r["ok"], r
        assert r["min_coverage"] == pytest.approx(1 / 3, abs=1e-3)

    def test_degraded_flag_trips_before_the_hard_fail(self):
        r = pc.matched_budget_coverage_parity(
            self._run(range(10)), self._run(range(14), vclock=True))
        assert r["ok"]                      # 71% still above the 50% floor
        assert r["degraded"] is True        # but below the 80% confidence bar

    def test_coverage_is_stamped_on_every_windowed_rung(self):
        real = self._run(range(20))
        sim = self._run(range(30), vclock=True)
        res = pc.run_all_parity(real, sim, {}, {}, agg_goal=1)
        stamped = {k for k, v in res.items()
                   if isinstance(v, dict) and "budget_coverage" in v}
        # every rung reporting a matched budget must carry its coverage
        windowed = {k for k, v in res.items()
                    if isinstance(v, dict)
                    and v.get("matched_logical_budget_n") is not None}
        assert windowed and stamped == windowed, (windowed - stamped)
        for k in stamped:
            assert res[k]["budget_coverage"]["min"] == pytest.approx(2 / 3, abs=1e-3)
            assert res[k]["low_budget_coverage"] is True

    def test_no_stamp_when_coverage_is_healthy(self):
        real = self._run(range(20))
        sim = self._run(range(20), vclock=True)
        res = pc.run_all_parity(real, sim, {}, {}, agg_goal=1)
        for v in res.values():
            if isinstance(v, dict) and "budget_coverage" in v:
                assert "low_budget_coverage" not in v

    def test_gates_as_a_stage0_control(self):
        meta = pc.CHECK_META["matched_budget_coverage"]
        assert meta["stage"] == 0 and meta["role"] == "CONTROL"


class TestMatchedLogicalBudget:
    """The logical-budget primitive + the U2/K8 reshape from count-at-clock-V to
    TIME-to-N. N = min(final progress each side) on the progress axis; the
    throughput/clock signal is real's algorithmic-time-to-N vs sim's vclock-to-N
    (PARITY.md §1.5). Extra units a faster side ran PAST the shared prefix must be
    absorbed; a genuine clock or trainer-set divergence within N must still fail."""

    def test_primitive_round_axis_is_min_final_round(self):
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r)) for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r), ts=float(r))
                               for r in range(1, 9)])
        N, prog_fn = pc._matched_logical_budget(real["agg_rounds"], sim["agg_rounds"])
        assert N == 5  # min(5, 8)
        assert prog_fn(real["agg_rounds"][0]) == 1

    def test_primitive_data_id_axis_is_committed_bin_count(self):
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d)) for d in range(5)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d), ts=float(d))
                               for d in range(8)])
        N, _ = pc._matched_logical_budget(real["agg_rounds"], sim["agg_rounds"])
        # data_id axis: N is the COUNT of bins both sides committed, in order
        # (5 = data_id 0..4), not a max key -- see the primitive's docstring.
        assert N == 5

    def test_primitive_none_when_a_side_has_no_progress(self):
        real = _agg(agg_rounds=[])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])
        N, prog_fn = pc._matched_logical_budget(real["agg_rounds"], sim["agg_rounds"])
        assert N is None and prog_fn is None

    def test_extra_units_past_shared_prefix_are_absorbed(self):
        # sim ran to 10 rounds, real to 5, at a MATCHED per-round rate. N=5, and
        # time-to-N matches on the prefix -- sim's extra rounds 6..10 must not fail.
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float((r - 1) * 10),
                                      ts=float(r)) for r in range(1, 11)])
        r = pc.total_commits_parity(real, sim, tol_rel=0.05)
        assert r["ok"], r
        assert r["matched_logical_budget_n"] == 5

    def test_time_to_n_gap_within_prefix_fails(self):
        # sim reaches the same N=5 rounds at HALF the virtual time real's clock
        # says the work takes -- a genuine clock divergence on the shared prefix.
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float((r - 1) * 5),
                                      ts=float(r)) for r in range(1, 6)])
        assert not pc.total_commits_parity(real, sim, tol_rel=0.05)["ok"]

    def test_terminal_state_trainer_set_divergence_fails_with_time_matched(self):
        # Time-to-N matches, but sim's contributing-trainer SET over the first N
        # units (5 distinct) diverges from real's (2) -- K8's live count dimension.
        real = _agg(agg_rounds=[_round(r, ["a", "b"], [0, 0], ts=float(r * 10))
                                for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, [f"t{r}"], [0], vclock=float((r - 1) * 10),
                                      ts=float(r)) for r in range(1, 6)])
        res = pc.terminal_state_parity(real, sim)
        assert not res["ok"], res
        assert res["time_rel_diff"] <= 0.05, res      # time matched
        assert res["trainers_rel_diff"] > 0.05, res   # trainer set is what fails


def _fwd_round(data_id, contributing, vclock=None, ts=0.0, round_=1):
    """fwdllm-style agg_round: `round` static (by default), progress on the
    committed `data_id` axis (cycle_data_id). `round_` lets a caller simulate a
    run long enough to complete a lap over `total_data_bins` and tick `round`."""
    e = {"event": "agg_round", "round": round_, "ts": ts,
         "cycle_data_id": data_id, "var_good_enough": True,
         "contributing_trainers": contributing, "staleness": [0],
         "agg_goal_count": 1}
    if vclock is not None:
        e["vclock_now"] = vclock
    return e


class TestProgressAxisRekey:
    """The clock family must measure progress on the axis the run advances.
    fwdllm keeps `round` at 1 and advances committed `data_id`, so a rung keyed
    on `round` divides by a stuck counter. The re-key auto-detects the axis;
    async_cifar10 (round-advancing) stays byte-identical."""

    def test_throughput_counts_data_ids_not_static_round(self):
        # 10 committed data_ids, round pinned at 1, matched 10 units / 100s.
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float((d + 1) * 10))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float((d + 1) * 10),
                                          ts=float(d + 1))
                               for d in range(10)])
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        # Re-keyed to data_id: 10 units, NOT collapsed to 1 stuck round.
        assert r["sim_rounds"] == 10 and r["real_rounds"] == 10, r
        assert r["ok"], r

    def test_throughput_divergence_caught_on_data_id_axis(self):
        # sim crawls (10 data_ids in 200s vclock) vs real (10 in 100s wall) -> 2x.
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float((d + 1) * 10))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float((d + 1) * 20),
                                          ts=float(d + 1))
                               for d in range(10)])
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        assert not r["ok"], r

    def test_terminal_state_time_to_n_measured_on_data_id_axis(self):
        real = _agg(agg_rounds=[_fwd_round(d, ["a", "b"], ts=float((d + 1) * 10))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a", "b"], vclock=float(d * 10),
                                          ts=float(d + 1))
                               for d in range(10)])
        r = pc.terminal_state_parity(real, sim)
        # Previously round-keyed & degenerate (round static); now the data_id axis
        # yields a real time-to-N on both sides.
        assert r["real_time_to_n_s"] > 0 and r["sim_vclock_to_n_s"] > 0, r
        assert r["ok"], r

    def test_total_commits_time_to_n_ignores_variance_retry(self):
        # A variance-FAIL retry emits 2 cycles on the SAME data_id; it must not
        # advance the logical budget N -- N stays data_id 1 and time-to-N is the
        # time to reach it (10s both sides), unaffected by the retry.
        real = _agg(agg_rounds=[
            _fwd_round(0, ["a"], ts=0.0),
            _fwd_round(0, ["a"], ts=5.0),   # retry, same data_id
            _fwd_round(1, ["a"], ts=10.0),
        ])
        sim = _agg(agg_rounds=[
            _fwd_round(0, ["a"], vclock=0.0, ts=1.0),
            _fwd_round(0, ["a"], vclock=5.0, ts=2.0),
            _fwd_round(1, ["a"], vclock=10.0, ts=3.0),
        ])
        r = pc.total_commits_parity(real, sim, tol_rel=0.05)
        assert r["matched_logical_budget_n"] == 2, r   # bins 0 and 1
        assert r["real_time_to_n_s"] == 10.0 and r["sim_vclock_to_n_s"] == 10.0, r
        assert r["ok"], r

    def test_normal_fl_still_keyed_on_round(self):
        # >1 distinct round -> axis stays "round"; identical to pre-re-key.
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 10),
                                      ts=float(r)) for r in range(1, 11)])
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        assert r["sim_rounds"] == 10, r
        assert r["ok"], r

    # --- The 4 advance rungs re-key too via _per_round_advances: keyed on
    # `round` they SKIP fwdllm ("<2 rounds"); on data_id they measure advances. ---

    def test_advance_rung_measures_on_data_id_axis(self):
        # Matched 10 s/data_id both sides -> advances measured, PASSes (not SKIP).
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 10))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 10),
                                          ts=float(d)) for d in range(10)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert "run too short" not in r.get("note", ""), r
        assert r["ok"], r

    def test_advance_rung_catches_data_id_rate_gap(self):
        # sim 26 s/data_id vclock vs real 15 s/data_id wall -> divergence caught.
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 15))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 26),
                                          ts=float(d)) for d in range(10)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert not r["ok"], r

    def test_advance_mean_on_round_axis_unchanged(self):
        # async_cifar10 (round-advancing): the sim mean advance is measured on
        # `round` exactly as before the re-key (+7 vclock per round).
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 7))
                                for r in range(1, 8)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 7),
                                      ts=float(r)) for r in range(1, 8)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert r["sim_mean_advance_s"] == 7.0, r
        assert r["real_mean_advance_s"] == 7.0, r

    def test_advance_mean_on_data_id_when_round_static(self):
        # fwdllm: round pinned at 1 -> mean advance is measured per data_id
        # (+4 vclock per committed data_id), not collapsed to a single stuck bin.
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 4))
                                for d in range(5)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 4),
                                          ts=float(d)) for d in range(5)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert r["sim_mean_advance_s"] == 4.0, r

    def test_axis_and_lap_disambiguation_when_only_sim_completes_a_lap(self):
        """On a long enough run the fast side (sim) can complete a full
        total_data_bins-length lap (`round` ticks 1->2, `cycle_data_id` wraps
        back to 0) while the slow side (real) never leaves round=1. The old
        axis heuristic picked per-side, so sim got keyed on `round` (one
        giant advance) while real stayed on `data_id` (many small ones) --
        incommensurate units, a spurious FAIL. Both sides must key on
        `data_id` whenever present, and the (round, data_id) composite key
        must keep sim's two laps distinct."""
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 10))
                                for d in range(10)])
        sim_events = (
            [_fwd_round(d, ["a"], vclock=float(d * 10), ts=float(d), round_=1)
             for d in range(10)]
            + [_fwd_round(d, ["a"], vclock=float((d + 10) * 10), ts=float(d + 10),
                          round_=2)
               for d in range(10)]
        )
        sim = _agg(agg_rounds=sim_events)
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert r["n_real_rounds"] == 9, r
        assert r["n_sim_rounds"] == 19, r
        assert r["ok"], r
        assert r["sim_mean_advance_s"] == pytest.approx(10.0), r
        assert r["real_mean_advance_s"] == pytest.approx(10.0), r


class TestConvergenceLapDisambiguation:
    """convergence_parity (C1/C2) has the same raw-data_id-collision exposure as
    the advance rungs (TestProgressAxisRekey): sim eval events can span
    round={1,2} while real stays at round=1. Keying the eval curve on raw
    `data_id` alone lets a sim lap-2 (more-trained) checkpoint silently
    overwrite lap-1's entry at the same nominal data_id, mismatching training
    amount. The (round, data_id) composite key excludes sim's lap-2 evals
    from the real/sim key intersection, comparing only genuinely matched
    progress."""

    def test_sim_lap2_eval_does_not_leak_into_lap1_comparison(self):
        real = _agg(agg_evals=[
            {"event": "agg_eval", "round": 1, "data_id": d,
             "test-accuracy": 0.5 + d * 0.01, "test-loss": 0.1}
            for d in range(10)
        ])
        sim = _agg(agg_evals=(
            [{"event": "agg_eval", "round": 1, "data_id": d,
              "test-accuracy": 0.5 + d * 0.01, "test-loss": 0.1}
             for d in range(10)]
            # lap 2: same nominal data_id values, much further trained -- must
            # NOT be compared against real's lap-1 checkpoints at those ids.
            + [{"event": "agg_eval", "round": 2, "data_id": d,
                "test-accuracy": 0.99, "test-loss": 0.01}
               for d in range(10)]
        ))
        r = pc.convergence_parity(real, sim, acc_tol=0.05)
        assert r["ok"], r
        assert r["avg_accuracy_diff"] == pytest.approx(0.0), r
        assert r["eval_rounds_compared"] == 10, r


class TestWallDisparity:
    """#6: a DIAG rung reporting |real_wall - sim_vclock| per matched progress
    unit. Never gates; surfaces the residual to drive to ~0."""

    def test_zero_disparity_when_clocks_match(self):
        # real advances 10 wall-s/data_id, sim 10 vclock-s/data_id -> residual 0.
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 10))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 10),
                                          ts=float(d)) for d in range(10)])
        r = pc.wall_disparity(real, sim)
        assert r["axis"] == "data_id"
        assert r["mean_abs_disparity_s"] == 0.0, r
        assert r["max_abs_disparity_s"] == 0.0, r
        assert r["n_matched_units"] == 10

    def test_surfaces_the_rate_gap(self):
        # real 35 wall-s/data_id vs sim 8 vclock-s/data_id -> growing residual,
        # but the rung still "ok" (DIAG never fails).
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 35))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 8),
                                          ts=float(d)) for d in range(10)])
        r = pc.wall_disparity(real, sim)
        assert r["ok"] is True  # DIAG: informational only
        # by the last of 10 data_ids: |9*35 - 9*8| = 243
        assert r["max_abs_disparity_s"] > 100.0, r
        assert r["mean_abs_disparity_s"] > 0.0

    def test_aligns_when_run_does_not_start_at_unit_zero(self):
        # matched units start at data_id 3; cumulative-from-first-matched keeps
        # residual 0 despite the nonzero vclock/ts origin.
        real = _agg(agg_rounds=[_fwd_round(d, ["a"], ts=float(d * 10))
                                for d in range(3, 8)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 10),
                                          ts=float(d)) for d in range(3, 8)])
        r = pc.wall_disparity(real, sim)
        assert r["mean_abs_disparity_s"] == 0.0, r

    def test_skips_when_too_few_matched_units(self):
        real = _agg(agg_rounds=[_fwd_round(0, ["a"], ts=0.0)])
        sim = _agg(agg_rounds=[_fwd_round(0, ["a"], vclock=0.0, ts=0.0)])
        r = pc.wall_disparity(real, sim)
        assert r.get("status") == "SKIP", r
        assert r["ok"] is True

    def test_round_axis_for_normal_fl(self):
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 5))
                                for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 5),
                                      ts=float(r)) for r in range(1, 6)])
        r = pc.wall_disparity(real, sim)
        assert r["axis"] == "round"
        assert r["mean_abs_disparity_s"] == 0.0, r


class TestFailsafeRealComputeSim:
    """A real-compute sim (fwdllm runs the real forward-grad pass in sim mode)
    has sim wall >> vclock by construction, so K5 must compare wall against the
    RUN wall budget, not the vclock (else it false-fails)."""

    def test_real_compute_sim_skips_without_run_budget(self):
        # data_id axis auto-detects real_compute_sim; wall ≫ vclock but no run
        # budget passed -> SKIP, not a false INV failure.
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 8),
                                          ts=float(d * 35)) for d in range(6)])
        r = pc.failsafe_ok(sim, budget_s=None)
        assert r.get("status") == "SKIP", r
        assert r["ok"] is True

    def test_real_compute_sim_uses_run_budget_when_given(self):
        # 6 data_ids, wall ends at 5*35=175s; run budget 3600s -> within budget.
        sim = _agg(agg_rounds=[_fwd_round(d, ["a"], vclock=float(d * 8),
                                          ts=float(d * 35)) for d in range(6)])
        r = pc.failsafe_ok(sim, budget_s=3600.0)
        assert r.get("status") != "SKIP", r
        assert r["real_compute_sim"] is True
        assert r["ok"] is True  # 175s wall << 3600s run budget

    def test_cheap_compute_sim_keeps_vclock_fallback(self):
        # round-advancing (async_cifar10): wall≈vclock, no run budget -> vclock
        # fallback unchanged (byte-identical).
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 10),
                                      ts=float(r * 10)) for r in range(1, 7)])
        r = pc.failsafe_ok(sim, budget_s=None)
        assert r["real_compute_sim"] is False
        assert r.get("status") != "SKIP"
        assert r["ok"] is True  # wall == vclock -> 0 overshoot


class TestFieldCoverageAlias:
    """fwdllm's trainer emits gpu/budget under different names; the coverage
    matrix must accept either spelling instead of false-FAILing."""

    def test_fwdllm_field_aliases_count_as_covered(self):
        agg = _agg(agg_rounds=[{"event": "agg_round", "round": 1, "ts": 0.0,
                                "vclock_now": 1.0, "trainer_speed_s": [1.0],
                                "staleness": [0], "stat_utility": [1.0],
                                "contributing_trainers": ["t1"]}])
        sel = {"num_eligible": 5, "avail_composition": {"a": 1},
               "num_chosen": 3}
        agg["selection_train"] = [sel]
        # trainer emits the fwdllm spellings only.
        tr = {"t1": {"trainer_round": [
            {"real_gpu_time_s": 1.2, "sim_round_duration_s": 2.3}]}}
        r = pc.field_coverage(agg, agg, tr, tr)
        gpu = r["matrix"]["trainer_round.gpu_compute_s"]
        bud = r["matrix"]["trainer_round.training_budget_s"]
        assert gpu["real"] == 1.0 and gpu["sim"] == 1.0, r
        assert bud["real"] == 1.0 and bud["sim"] == 1.0, r
        assert "trainer_round.gpu_compute_s(real)" not in r["violations"], r


class TestTrainerSpeedParity:
    def test_identical_passes(self):
        rounds_with_speed = [
            _round_speed(r, ["a"], [0], speed=28.0) for r in range(1, 6)
        ]
        agg = _agg(agg_rounds=rounds_with_speed)
        r = pc.trainer_speed_parity(agg, agg, ks_tol=0.1)
        assert r["ok"]

    def test_out_of_support_tail_fails(self):
        # Jun-16 support-guard semantics: P3 fails when sim produces speeds BEYOND
        # real's support (the genuine speed-model bug — oort's old 56s→sim tail).
        real_rounds = [_round_speed(r, ["a"], [0], speed=11.0) for r in range(1, 11)]
        sim_rounds = [_round_speed(r, ["a"], [0], speed=56.0) for r in range(1, 11)]
        real = _agg(agg_rounds=real_rounds)
        sim = _agg(agg_rounds=sim_rounds)
        r = pc.trainer_speed_parity(real, sim)
        assert not r["ok"]
        assert r["support_ratio"] > 1.0 + r["support_tol"]

    def test_faster_sim_within_support_defers_to_mix(self):
        # sim faster than real, same support direction (wall-capture / faster
        # selection mix) is NOT a speed-model bug — P3 passes, A2c owns the mix.
        real_rounds = [_round_speed(r, ["a"], [0], speed=11.0) for r in range(1, 11)]
        sim_rounds = [_round_speed(r, ["a"], [0], speed=7.0) for r in range(1, 11)]
        r = pc.trainer_speed_parity(_agg(agg_rounds=real_rounds),
                                    _agg(agg_rounds=sim_rounds))
        assert r["ok"], r


class TestBudgetNotCap:
    def test_no_cap_provided_skips(self):
        a = _agg(agg_rounds=[_round(100, ["a"], [0], ts=100.0)])
        r = pc.budget_not_cap(a, a)
        assert r["ok"]

    def test_under_cap_passes(self):
        a = _agg(agg_rounds=[_round(50, ["a"], [0], ts=100.0)])
        r = pc.budget_not_cap(a, a, rounds_cap=1000)
        assert r["ok"]

    def test_hit_cap_warns(self):
        a = _agg(agg_rounds=[_round(1000, ["a"], [0], ts=100.0)])
        r = pc.budget_not_cap(a, a, rounds_cap=1000, budget_s=10800.0)
        assert not r["ok"] and r.get("warnings")


class TestRunAllParityExtended:
    """Smoke test: run_all_parity returns a result for every expected new key."""

    _NEW_KEYS = [
        "vclock_telemetry", "throughput", "per_round_advance",
        "overlap_factor", "total_commits", "terminal_state",
        "trainer_speed",
    ]

    def test_new_keys_present(self):
        a = _agg(
            selection=[_sel(1, ["a", "b"])],
            agg_rounds=[
                _round_speed(r, ["a"], [0], vclock=float(r * 10),
                              ts=float(r), speed=12.0)
                for r in range(1, 6)
            ],
        )
        tr = {"aa": {"task_recv": [], "trainer_round": []}}
        res = pc.run_all_parity(a, a, tr, tr, agg_goal=2)
        for key in self._NEW_KEYS:
            assert key in res, f"missing key: {key}"

    def test_overall_verdict_fails_on_k2(self):
        """A pair where sim overcharges virtual time should fail overall verdict."""
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 15))
                                  for r in range(1, 68)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 26),
                                       ts=float(r))
                                 for r in range(1, 42)])
        tr: dict = {}
        res = pc.run_all_parity(real, sim, tr, tr)
        passed, roots, downstream, _warnings = pc.overall_verdict(res)
        assert not passed
        failures = set(roots) | set(downstream)
        assert "throughput" in failures or "per_round_advance" in failures


def _sel_pool(round_, speeds, ts=0.0):
    """selection event with a per_trainer pool carrying speed_s (for A2b)."""
    per_trainer = {f"t{i:03d}": {"speed_s": sp, "utility": None, "selected": i < 10}
                   for i, sp in enumerate(speeds)}
    return {"event": "selection", "task": "train", "round": round_, "ts": ts,
            "num_candidates": len(speeds), "num_eligible": len(speeds),
            "per_trainer": per_trainer}


class TestEligibleSpeedComposition:
    """A2b: eligible-pool speed-composition parity (catches what A2's count misses)."""

    def test_matched_pool_passes(self):
        pool = [3.0, 5.0, 8.0, 12.0, 20.0] * 6
        real = _agg(selection=[_sel_pool(r, pool) for r in range(1, 6)])
        sim = _agg(selection=[_sel_pool(r, pool) for r in range(1, 6)])
        res = pc.eligible_speed_composition_parity(real, sim)
        assert res["ok"], res

    def test_diverged_pool_fails(self):
        # Same eligible-set SIZE (30) in both, but sim pool is slow-skewed (the refl
        # signature: slow clients re-enter sim's pool). A2b must FAIL on composition.
        fast = [2.0, 3.0, 4.0, 5.0, 6.0] * 6   # real: fast-skewed pool
        slow = [10.0, 12.0, 14.0, 18.0, 22.0] * 6  # sim: slow-skewed pool
        real = _agg(selection=[_sel_pool(r, fast) for r in range(1, 6)])
        sim = _agg(selection=[_sel_pool(r, slow) for r in range(1, 6)])
        res = pc.eligible_speed_composition_parity(real, sim)
        assert not res["ok"], res
        assert res["real_mean_pool_speed_s"] < res["sim_mean_pool_speed_s"]

    def test_no_per_trainer_skips(self):
        real = _agg(selection=[_sel(1, ["a"])])
        sim = _agg(selection=[_sel(1, ["a"])])
        res = pc.eligible_speed_composition_parity(real, sim)
        assert res["ok"] and res.get("status") == "SKIP", res


class TestInflightResidenceEvent:
    """The oort in-flight residence telemetry builder (PARITY §4.x fine-tuning)."""

    def test_builder_shape(self):
        from flame.telemetry.events import build_inflight_residence, EVENT_INFLIGHT_RESIDENCE
        ev, f = build_inflight_residence(
            round_num=7, time_mode="sim", in_flight_before=16, in_flight_after=13,
            committed_fresh=10, cleaned=3, stale_rejected=0,
            residence_rounds=[0, 1, 2], carried_over_ages=[0, 0, 1])
        assert ev == EVENT_INFLIGHT_RESIDENCE
        assert f["time_mode"] == "sim" and f["in_flight_before"] == 16
        assert f["residence_rounds"] == [0, 1, 2]
        # optional fields omitted when None
        ev2, f2 = build_inflight_residence(
            round_num=1, time_mode="real", in_flight_before=13, in_flight_after=13)
        assert "residence_rounds" not in f2 and "cleaned" not in f2

    def test_builder_paired_commit_class(self):
        """residence_staleness / residence_was_fresh ride alongside residence_rounds
        (paired 1:1) to decompose the residence-shape gap by commit class (refl A2)."""
        from flame.telemetry.events import build_inflight_residence
        ev, f = build_inflight_residence(
            round_num=7, time_mode="real", in_flight_before=16, in_flight_after=13,
            residence_rounds=[3, 3, 5], residence_staleness=[0, 2, 4],
            residence_was_fresh=[True, False, False])
        assert f["residence_staleness"] == [0, 2, 4]
        assert f["residence_was_fresh"] == [True, False, False]
        assert len(f["residence_staleness"]) == len(f["residence_rounds"])
        # omitted when not supplied
        _, f2 = build_inflight_residence(
            round_num=1, time_mode="sim", in_flight_before=13, in_flight_after=13,
            residence_rounds=[1])
        assert "residence_staleness" not in f2


# ── FwdLLM variance-cadence layer (V/DK/G rungs) ──────────

def _cadence(cycle_data_id, cycle_iteration, var, var_threshold,
             var_good_enough, force_commit_planned=False, grad_pool_size=None,
             cached_v_size=None, agg_goal=None, round_=1, ts=0.0, **extra):
    """One fwdllm agg-goal-boundary (variance gate) agg_round event."""
    e = {"event": "agg_round", "round": round_, "ts": ts,
         "cycle_data_id": cycle_data_id, "cycle_iteration": cycle_iteration,
         "var": var, "var_threshold": var_threshold,
         "var_good_enough": var_good_enough,
         "force_commit_planned": force_commit_planned}
    for k, v in (("grad_pool_size", grad_pool_size),
                 ("cached_v_size", cached_v_size), ("agg_goal", agg_goal)):
        if v is not None:
            e[k] = v
    e.update(extra)
    return e


def _cadence_run(iters_per_data, var_threshold=1.0, pass_var=0.5, fail_var=2.0,
                 agg_goal=3, round_=1):
    """Series where data_id d takes iters_per_data[d] cycles: (n-1) variance
    FAILs (var>thr) then one PASS (var<=thr). grad_pool grows each retry and is
    at its max on the committing cycle; cached_v grows across the FAIL rollbacks.

    `round_` is the lap counter -- pass it to build a run that WRAPPED past
    `total_data_bins` and re-visited the same `cycle_data_id`s (§D-19).
    """
    events = []
    for d, n in enumerate(iters_per_data):
        for it in range(n):
            committed = (it == n - 1)
            events.append(_cadence(
                cycle_data_id=d, cycle_iteration=it,
                var=(pass_var if committed else fail_var),
                var_threshold=var_threshold, var_good_enough=committed,
                grad_pool_size=(it + 1) * agg_goal, cached_v_size=it,
                agg_goal=agg_goal, round_=round_))
    return events


class TestSelectionDetailMatchedWindow:
    """S3/S4 contract: same number of cohort re-draws AND same cohort size, at
    matched WORK. A round-cadence baseline re-draws once per round, so the event
    COUNT is part of the contract, not just `num_chosen`.

    Counted at the matched logical budget (§D-4) so the rung grades the selector
    rather than how far each side got: felix_round's real leg lapped and re-drew
    at 4752-4760s, after it had already passed the budget, reading as "real 2.73
    vs sim 30.0 chosen" when over the work both sides actually did the two are
    identical (§D-17). That progress difference is real and is graded -- by
    `throughput`, which fails on the same pair."""

    @staticmethod
    def _sd(num_chosen, ts):
        return {"event": "selection", "task": "train", "ts": ts,
                "num_chosen": num_chosen, "in_flight": 30, "effective_c": 30}

    def _side(self, sel, n_bins, ts_per_bin=1.0):
        rounds = [_cadence(d, 0, 0.5, 1.0, True, ts=(d + 1) * ts_per_bin)
                  for d in range(n_bins)]
        return _agg(selection=sel, agg_rounds=rounds)

    def test_out_of_budget_redraw_excluded(self):
        # Real reaches the budget at bin 4 (ts=5.0) then re-draws its cohort in
        # small backfills at ts>5.0; sim never gets there. Inside the window the
        # two are identical.
        real = self._side([self._sd(30, 0.5)]
                          + [self._sd(2, t) for t in (6.0, 6.1, 6.2)], n_bins=8)
        sim = self._side([self._sd(30, 0.5)], n_bins=5)
        r = pc.selection_detail_parity(real, sim)
        assert r["ok"], r
        assert r["real_mean_chosen"] == 30.0 and r["sim_mean_chosen"] == 30.0, r
        assert r["n_real_selections"] == 1, r

    def test_unwindowed_would_have_failed(self):
        # Same data, budget removed: proves the window is what changes the
        # verdict, not a tolerance change.
        real = self._side([self._sd(30, 0.5)]
                          + [self._sd(2, t) for t in (6.0, 6.1, 6.2)], n_bins=8)
        real_no_budget = _agg(selection=real["selection_train"], agg_rounds=[])
        sim_no_budget = _agg(selection=[self._sd(30, 0.5)], agg_rounds=[])
        r = pc.selection_detail_parity(real_no_budget, sim_no_budget)
        assert not r["ok"] and r["real_mean_chosen"] == 9.0, r

    def test_in_budget_divergence_still_fails(self):
        # The window must not become a blanket amnesty: a genuine granularity
        # difference INSIDE the budget still has to fail.
        real = self._side([self._sd(30, 0.5)], n_bins=5)
        sim = self._side([self._sd(2, t) for t in (0.5, 1.5, 2.5)], n_bins=5)
        r = pc.selection_detail_parity(real, sim)
        assert not r["ok"] and r["sim_mean_chosen"] == 2.0, r

    def test_one_redraw_each_of_the_same_size_passes(self):
        """Operator ruling: a 1.5h run IS how this baseline operates. One round,
        one cohort draw per side, same size -> PASS. Trivial, but true to the
        system: nothing about the selector diverged."""
        real = self._side([self._sd(30, 0.5)], n_bins=5)
        sim = self._side([self._sd(30, 0.5)], n_bins=5)
        r = pc.selection_detail_parity(real, sim)
        assert r["ok"] and "underpowered" not in r, r

    def test_redraw_COUNT_mismatch_fails_even_when_cohort_size_matches(self):
        """Rounds must match too: 1 re-draw vs 4, both of 30 trainers, is a
        divergence (`fedbuff_round`'s shape -- sim crossed the round boundary
        inside the graded window and re-drew, real never did)."""
        real = self._side([self._sd(30, 0.5)], n_bins=5)
        sim = self._side([self._sd(30, t) for t in (0.5, 1.5, 2.5, 3.5)], n_bins=5)
        r = pc.selection_detail_parity(real, sim)
        assert not r["ok"] and r["rel_diff_n_selections"] == 0.75, r
        assert r["real_mean_chosen"] == r["sim_mean_chosen"] == 30.0, r

    def test_full_run_counts_reported_so_the_window_hides_nothing(self):
        real = self._side([self._sd(30, 0.5)]
                          + [self._sd(2, t) for t in (6.0, 6.1, 6.2)], n_bins=8)
        sim = self._side([self._sd(30, 0.5)], n_bins=5)
        r = pc.selection_detail_parity(real, sim)
        assert r["ok"], r
        assert r["full_run_n_real_selections"] == 4, r
        assert r["full_run_n_sim_selections"] == 1, r

    def test_window_never_empties_a_side_out(self):
        # If the budget would leave a side with no selections at all, fall back
        # to the full run rather than manufacturing a pass on zero data.
        real = self._side([self._sd(30, 9.0)], n_bins=5)   # all past the budget
        sim = self._side([self._sd(2, 9.0)], n_bins=5)
        r = pc.selection_detail_parity(real, sim)
        assert r["n_real_selections"] == 1 and not r["ok"], r


class TestV1IterPerDataId:
    def test_matched_passes(self):
        real = _agg(agg_rounds=_cadence_run([1, 2, 1, 3, 1]))
        sim = _agg(agg_rounds=_cadence_run([1, 2, 1, 3, 1]))
        r = pc.iters_per_data_id_parity(real, sim)
        assert r["ok"] and r["real_mean_iters"] == r["sim_mean_iters"], r

    def test_diverged_fails(self):
        # sim compounds: every data_id needs 3 cycles vs real's 1 (a
        # contributing-set/order divergence surfacing as more retries).
        real = _agg(agg_rounds=_cadence_run([1, 1, 1, 1, 1]))
        sim = _agg(agg_rounds=_cadence_run([3, 3, 3, 3, 3]))
        r = pc.iters_per_data_id_parity(real, sim)
        assert not r["ok"] and r["sim_mean_iters"] > r["real_mean_iters"], r

    def test_iters_binning_exact_for_force_commit(self):
        # A data_id that force-commits after 2 fails still counts 3 cycles: the
        # commit event carries cycle_data_id of the SAME data_id (pre-advance).
        # Keyed per VISIT -- (round, cycle_data_id) -- not raw data_id (§D-19).
        cycles = [
            _cadence(0, 0, 2.0, 1.0, False),
            _cadence(0, 1, 2.0, 1.0, False),
            _cadence(0, 2, 2.0, 1.0, True, force_commit_planned=True),  # forced
        ]
        assert pc._iters_per_data_id(cycles) == {(1, 0): 3}

    def test_wrapped_data_id_counted_as_two_visits_not_one_bin(self):
        """§D-19: `cycle_data_id` wraps each lap. Two visits to bin 0 in
        different rounds are two bins of 2 and 3 cycles -- NOT one bin of 5,
        which is what the old raw-data_id key produced (and which both summed
        the counts and divided by too few bins, hiding a real gap)."""
        cycles = [
            _cadence(0, 0, 2.0, 1.0, False, round_=1),
            _cadence(0, 1, 0.5, 1.0, True, round_=1),
            _cadence(0, 0, 2.0, 1.0, False, round_=2),
            _cadence(0, 1, 2.0, 1.0, False, round_=2),
            _cadence(0, 2, 0.5, 1.0, True, round_=2),
        ]
        assert pc._iters_per_data_id(cycles) == {(1, 0): 2, (2, 0): 3}

    def test_incomplete_trailing_visit_excluded(self):
        """A run stopped by `max_runtime_s` mid variance-check leaves a visit
        that never passed the gate; its truncated cycle count is not "iterations
        needed to close" and must not enter the distribution (§D-13)."""
        cycles = [
            _cadence(0, 0, 2.0, 1.0, False, round_=1),
            _cadence(0, 1, 0.5, 1.0, True, round_=1),   # completed
            _cadence(1, 0, 2.0, 1.0, False, round_=1),  # deadline hit here
        ]
        assert pc._iters_per_data_id(cycles) == {(1, 0): 2}

    def test_no_completed_visit_falls_back_to_unfiltered(self):
        """Fixtures/baselines that never set the commit field must not collapse
        to an empty distribution -- mirrors `_per_progress_last_event`."""
        cycles = [
            {"event": "agg_round", "round": 1, "cycle_data_id": 0},
            {"event": "agg_round", "round": 1, "cycle_data_id": 0},
        ]
        assert pc._iters_per_data_id(cycles) == {(1, 0): 2}

    def test_truncated_to_matched_budget_so_extra_real_bins_dont_count(self):
        """The felix_round shape end-to-end: real laps past `total_data_bins`
        and re-visits early bins; sim never gets there. Those extra visits are
        beyond the matched logical budget and must not enter either mean --
        otherwise real's late, differently-priced bins pull its average and the
        comparison stops being like-for-like (§D-4)."""
        real = _agg(agg_rounds=(_cadence_run([2, 2, 2, 2])
                                + _cadence_run([9, 9], round_=2)))  # 2nd lap
        sim = _agg(agg_rounds=_cadence_run([3, 3, 3, 3]))
        r = pc.iters_per_data_id_parity(real, sim)
        # The 2nd-lap visits (9 cycles each) are outside the budget: real is 2.0,
        # not (2*4 + 9*2)/6 = 4.33.
        assert r["n_real_data_ids"] == 4 and r["real_mean_iters"] == 2.0, r
        assert r["sim_mean_iters"] == 3.0 and not r["ok"], r

    def test_non_fwdllm_skips(self):
        a = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])  # no cadence fields
        r = pc.iters_per_data_id_parity(a, a)
        assert r["ok"] and r.get("status") == "SKIP", r


class TestV1bItersMovingAvg:
    """V1b: the MOVING-AVERAGE trajectory of iters-per-data_id must track tightly
    over the whole run -- catches a run-length DRIFT that v1's pooled KS+mean is
    blind to (identical pooled stats, divergent trajectory)."""

    def test_matched_passes(self):
        seq = [1, 2, 1, 3, 1, 2, 1, 1, 2, 3] * 6  # 60 data_ids
        real = _agg(agg_rounds=_cadence_run(seq))
        sim = _agg(agg_rounds=_cadence_run(seq))
        r = pc.iters_per_data_id_moving_avg_parity(real, sim, window=10)
        assert r["ok"] and r["ma_max_abs_dev"] == 0.0, r

    def test_late_run_drift_fails_even_when_pooled_stats_match(self):
        # Construct the exact case v1 misses: the SAME multiset of iteration
        # counts (identical pooled histogram + mean, so v1 KS+mean PASS), but the
        # ORDER differs -- sim front-loads the cheap data_ids and back-loads the
        # expensive ones, so its moving average drifts above real's late-run.
        base = ([1] * 30) + ([3] * 30)          # cheap-then-expensive
        real = _agg(agg_rounds=_cadence_run(base))
        sim = _agg(agg_rounds=_cadence_run(list(reversed(base))))  # expensive-then-cheap
        # v1 (pooled) cannot tell them apart:
        v1 = pc.iters_per_data_id_parity(real, sim)
        assert v1["ok"] and v1["real_mean_iters"] == v1["sim_mean_iters"], v1
        # v1b (trajectory) catches the drift:
        r = pc.iters_per_data_id_moving_avg_parity(real, sim, window=10)
        assert not r["ok"] and r["ma_max_abs_dev"] > 1.0, r

    def test_small_jitter_within_tight_bound_passes(self):
        # Per-data_id counts differ by an occasional +/-1 (fp16 jitter), but the
        # smoothed average stays within the tight band -> PASS (exact not required).
        real_seq = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 5
        sim_seq = [2, 3, 2, 1, 2, 2, 3, 1, 2, 2] * 5   # same mean, local wobble
        real = _agg(agg_rounds=_cadence_run(real_seq))
        sim = _agg(agg_rounds=_cadence_run(sim_seq))
        r = pc.iters_per_data_id_moving_avg_parity(real, sim, window=10)
        assert r["ok"], r

    def test_non_fwdllm_skips(self):
        a = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])
        r = pc.iters_per_data_id_moving_avg_parity(a, a)
        assert r["ok"] and r.get("status") == "SKIP", r

    @staticmethod
    def _async_run(iters):
        ev = []
        for d, n in enumerate(iters):
            for it in range(n):
                ev.append(_lcyc(d, it, ["a"], 0.5, var_good=(it == n - 1),
                                goal=10, is_async=True))
        return ev

    def test_async_stochastic_gates_ma_shadow_keeps_cum_drift(self):
        # fluxtune regime: async + stochastic selector. The per-data_id iter count
        # is noisy and DECORRELATED between modes (boundary-race cascade), so the
        # MA curves can't shadow -- gated. Same iter multiset in a different ORDER
        # (large MA dev, equal cumulative mean) must now PASS.
        sel = [_selc(1, ["a"], 20)]                       # subset -> stochastic
        base = ([1] * 20) + ([5] * 20)
        real = _agg(selection=sel, agg_rounds=self._async_run(base))
        sim = _agg(selection=sel, agg_rounds=self._async_run(list(reversed(base))))
        r = pc.iters_per_data_id_moving_avg_parity(real, sim, window=10)
        assert r["ma_shadow_gated"] is True
        assert r["ma_max_abs_dev"] > 1.0                  # shadow genuinely diverges
        assert r["cum_mean_rel_diff"] <= 0.05             # but cumulative mean matches
        assert r["ok"]

    def test_async_stochastic_still_fails_on_cumulative_drift(self):
        # Gating the MA shadow does NOT gate a real throughput drift: sim doing
        # materially more iters/data_id overall still fails via the cum-mean guard.
        sel = [_selc(1, ["a"], 20)]
        real = _agg(selection=sel, agg_rounds=self._async_run([2] * 40))
        sim = _agg(selection=sel, agg_rounds=self._async_run([3] * 40))  # +50%
        r = pc.iters_per_data_id_moving_avg_parity(real, sim, window=10)
        assert r["ma_shadow_gated"] is True
        assert r["cum_mean_rel_diff"] > 0.05
        assert not r["ok"]


class TestV2VarTrajectory:
    def test_matched_passes(self):
        real = _agg(agg_rounds=_cadence_run([2, 2, 2]))
        sim = _agg(agg_rounds=_cadence_run([2, 2, 2]))
        assert pc.var_trajectory_parity(real, sim)["ok"]

    def test_diverged_fails(self):
        # Same iteration counts but the variance *signal* differs (grad-pool
        # accumulation-order bug): sim's var values sit far from real's.
        real = _agg(agg_rounds=[_cadence(d, 0, 0.4, 1.0, True) for d in range(8)])
        sim = _agg(agg_rounds=[_cadence(d, 0, 5.0, 1.0, True) for d in range(8)])
        assert not pc.var_trajectory_parity(real, sim)["ok"]


class TestV4ForceCommitRate:
    def test_matched_passes(self):
        real = _agg(agg_rounds=_cadence_run([1, 1, 1, 1]))
        sim = _agg(agg_rounds=_cadence_run([1, 1, 1, 1]))
        r = pc.force_commit_rate_parity(real, sim)
        assert r["ok"] and r["sim_force_commit_rate"] == 0.0, r

    def test_diverged_fails(self):
        # sim hits the iteration cap far more often (chronic variance divergence).
        real = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True) for d in range(10)])
        sim = _agg(agg_rounds=[
            _cadence(d, 0, 2.0, 1.0, True, force_commit_planned=True)
            for d in range(10)])
        r = pc.force_commit_rate_parity(sim, real)  # order-agnostic
        assert not r["ok"], r


class TestV5VariancePassRatio:
    def test_matched_passes(self):
        real = _agg(agg_rounds=_cadence_run([1, 2, 1, 2]))
        sim = _agg(agg_rounds=_cadence_run([1, 2, 1, 2]))
        assert pc.variance_pass_ratio_parity(real, sim)["ok"]

    def test_force_commit_excluded_from_pass(self):
        # A force-commit (var>thr, committed only because the cap fired) is NOT a
        # genuine variance pass -> real (genuine) and sim (forced) diverge on V5
        # even though both "committed" every cycle.
        real = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True) for d in range(10)])
        sim = _agg(agg_rounds=[
            _cadence(d, 0, 2.0, 1.0, True, force_commit_planned=True)
            for d in range(10)])
        r = pc.variance_pass_ratio_parity(real, sim)
        assert not r["ok"]
        assert r["real_pass_ratio"] == 1.0 and r["sim_pass_ratio"] == 0.0, r


class TestV3CachedVPool:
    def test_matched_passes(self):
        real = _agg(agg_rounds=_cadence_run([2, 3, 2]))
        sim = _agg(agg_rounds=_cadence_run([2, 3, 2]))
        assert pc.cached_v_pool_parity(real, sim)["ok"]

    def test_absent_skips(self):
        real = _agg(agg_rounds=[_cadence(0, 0, 0.5, 1.0, True)])
        sim = _agg(agg_rounds=[_cadence(0, 0, 0.5, 1.0, True)])
        r = pc.cached_v_pool_parity(real, sim)
        assert r["ok"] and r.get("status") == "SKIP", r


class TestDK1AggGoalTrajectory:
    def test_constant_k_skips(self):
        real = _agg(agg_rounds=_cadence_run([1, 1, 1], agg_goal=3))
        sim = _agg(agg_rounds=_cadence_run([1, 1, 1], agg_goal=3))
        r = pc.agg_goal_trajectory_parity(real, sim)
        assert r["ok"] and r.get("status") == "SKIP" and "disabled" in r["note"], r

    def test_varying_k_matched_passes(self):
        real = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True, agg_goal=k)
                                for d, k in enumerate([3, 3, 4, 5, 4])])
        sim = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True, agg_goal=k)
                               for d, k in enumerate([3, 3, 4, 5, 4])])
        assert pc.agg_goal_trajectory_parity(real, sim)["ok"]


class TestDK3EligibleEndsMetric:
    def test_absent_skips_with_note(self):
        real = _agg(agg_rounds=_cadence_run([1, 1]))
        sim = _agg(agg_rounds=_cadence_run([1, 1]))
        r = pc.eligible_ends_metric_parity(real, sim)
        assert r["ok"] and r.get("status") == "SKIP" and "deferred" in r["note"], r

    def test_present_diverged_fails(self):
        real = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True, n_eligible_train=10)
                                for d in range(6)])
        sim = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True, n_eligible_train=2)
                               for d in range(6)])
        assert not pc.eligible_ends_metric_parity(real, sim)["ok"]


class TestG1GradNorm:
    def test_absent_skips_with_note(self):
        real = _agg(agg_rounds=_cadence_run([1, 1]))
        sim = _agg(agg_rounds=_cadence_run([1, 1]))
        r = pc.grad_norm_parity(real, sim)
        assert r["ok"] and r.get("status") == "SKIP" and "deferred" in r["note"], r


class TestG2GradPoolSize:
    def test_matched_passes(self):
        real = _agg(agg_rounds=_cadence_run([1, 2, 1, 2]))
        sim = _agg(agg_rounds=_cadence_run([1, 2, 1, 2]))
        assert pc.grad_pool_size_parity(real, sim)["ok"]

    def test_diverged_fails(self):
        # sim accumulates far larger pools before committing (K x V1 rollup).
        real = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True, grad_pool_size=3)
                                for d in range(8)])
        sim = _agg(agg_rounds=[_cadence(d, 0, 0.5, 1.0, True, grad_pool_size=30)
                               for d in range(8)])
        assert not pc.grad_pool_size_parity(real, sim)["ok"]


class TestRunAllParityFwdllm:
    _CADENCE_KEYS = [
        "v1_iter_per_data_id", "v2_var_trajectory", "v3_cached_v_pool",
        "v4_force_commit_rate", "v5_variance_pass_ratio",
        "dk1_agg_goal_trajectory", "dk2_dynamic_c", "dk3_eligible_ends_metric",
        "g1_grad_norm", "g2_grad_pool_size",
    ]

    def test_keys_present_and_identical_passes(self):
        a = _agg(selection=[_sel(1, ["a", "b"])],
                 agg_rounds=_cadence_run([1, 2, 1, 3, 1]))
        tr = {"aa": {"task_recv": [], "trainer_round": []}}
        res = pc.run_all_parity(a, a, tr, tr, agg_goal=3)
        for key in self._CADENCE_KEYS:
            assert key in res, f"missing cadence key: {key}"
            v = res[key]
            # identical real==sim ⇒ every cadence rung PASSes or SKIPs cleanly
            assert v["ok"], f"{key}: {v}"

    def test_non_fwdllm_run_skips_all_cadence(self):
        a = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r))
                             for r in range(1, 4)])
        tr: dict = {}
        res = pc.run_all_parity(a, a, tr, tr)
        for key in self._CADENCE_KEYS:
            assert res[key].get("status") == "SKIP", f"{key} should SKIP: {res[key]}"

    def test_cadence_divergence_fails_verdict(self):
        real = _agg(agg_rounds=_cadence_run([1, 1, 1, 1, 1, 1]))
        sim = _agg(agg_rounds=_cadence_run([3, 3, 3, 3, 3, 3]))
        tr: dict = {}
        res = pc.run_all_parity(real, sim, tr, tr, agg_goal=3)
        passed, roots, downstream, _w = pc.overall_verdict(res)
        assert not passed
        assert "v1_iter_per_data_id" in (set(roots) | set(downstream))


class TestAggRoundCadenceEmission:
    """The agg_round builder carries the cadence fields through `extra` (the
    aggregator snapshots them pre-mutation)."""

    def test_cadence_fields_land_in_event(self):
        from flame.telemetry.events import build_agg_round, EVENT_AGG_ROUND
        ev, f = build_agg_round(
            round_num=4, agg_goal=3, agg_goal_count=3,
            extra={"cycle_data_id": 7, "cycle_iteration": 2, "var": 0.8,
                   "var_threshold": 1.0, "var_good_enough": True,
                   "force_commit_planned": False, "grad_pool_size": 9,
                   "cached_v_size": 2})
        assert ev == EVENT_AGG_ROUND
        assert f["cycle_data_id"] == 7 and f["cycle_iteration"] == 2
        assert f["grad_pool_size"] == 9 and f["cached_v_size"] == 2


# ── FwdLLM async residence rungs R1 / W1 ──

def _cyc(cycle_data_id, intervals, contributing=None):
    """A committed cadence cycle carrying per-contributor [dispatch, commit]
    intervals. `intervals` = list of (end, dispatch_ts, commit_ts)."""
    ci = [{"end": e, "dispatch_ts": d, "commit_ts": c} for (e, d, c) in intervals]
    return {"event": "agg_round", "round": 1, "ts": 0.0,
            "cycle_data_id": cycle_data_id, "var_good_enough": True,
            "agg_goal_count": len(ci),
            "contributing_trainers": contributing or [e for (e, _, _) in intervals],
            "contributor_intervals": ci}


def _trainers_with_rounds(counts):
    """{short_id: {"trainer_round": [ ...n events ]}} for W1 forward-pass counts."""
    return {sid: {"trainer_round": [{"event": "trainer_round"} for _ in range(n)]}
            for sid, n in counts.items()}


def _dispatch(peer, ts, payload_kind="weights", version_key=(0, 0)):
    mv, it = version_key
    return {"event": "comm", "direction": "agg_to_trainer", "peer_id": peer,
            "ts": ts, "payload_kind": payload_kind,
            "model_version": mv, "iteration_per_data_id": it}


def _agg_comm(dispatches, resolves=None):
    """`dispatches` = list of (peer, ts), (peer, ts, payload_kind), or
    (peer, ts, payload_kind, version_key) -> comm_dispatch, each tagged with
    its own `version_key` = (model_version, iteration_per_data_id) (defaults
    to (0, 0) if omitted). `resolves` = list of (peer, ts) or
    (peer, ts, version_key) -> agg_rounds entries whose contributor_intervals
    name that peer via `dispatch_version_key` (R1 is scoped to version_key --
    a resolve only clears the SAME version_key it names). Always tags
    is_async=True (R1 is async-only) even with no resolves, via a marker
    round with no contributor."""
    d = _agg()
    d["comm_dispatch"] = [_dispatch(*args) for args in dispatches]
    resolve_rounds = []
    for r in (resolves or []):
        peer, ts = r[0], r[1]
        vk = r[2] if len(r) > 2 else (0, 0)
        resolve_rounds.append({
            "event": "agg_round", "ts": ts, "is_async": True,
            "contributor_intervals": [{"end": peer, "dispatch_version_key": list(vk)}],
        })
    d["agg_rounds"] = [{"event": "agg_round", "ts": 0.0, "is_async": True}] + resolve_rounds
    return d


class TestR1InflightOverlap:
    """R1: no trainer may have the SAME version_key (model_version,
    iteration_per_data_id) outstanding twice -- a DISPATCH for a version_key
    while an EARLIER dispatch for that EXACT version_key hasn't yet been
    RESOLVED by a variance-gate evaluation naming it via
    `dispatch_version_key`. Scoped to version_key (not "any unresolved
    dispatch") because a trainer's stale grad is legitimately consumed
    (down-weighted) by a LATER cycle's evaluation while the trainer is
    handed genuinely NEW work in parallel -- that's not a violation. Also
    subsumes the earlier REPLY-based rejection: `var_bad` resampling
    naturally gets a fresh `iteration_per_data_id`, so it's never flagged."""

    def test_dispatch_after_resolve_passes(self):
        # A and B are each re-dispatched a 2nd time (a NEW version_key), but
        # only AFTER a variance-gate evaluation resolved their 1st.
        agg = _agg_comm(
            dispatches=[("A", 0.0, "weights", (0, 0)), ("B", 0.0, "weights", (0, 0)),
                        ("A", 6.0, "weights", (1, 0)), ("B", 6.0, "weights", (1, 0))],
            resolves=[("A", 5.0, (0, 0)), ("B", 5.0, (0, 0))],
        )
        r = pc.inflight_overlap_parity(agg, agg)
        assert r["ok"]
        assert r["real_overlap_frac"] == 0.0 and r["sim_overlap_frac"] == 0.0

    def test_sim_overlap_fails_with_clean_real(self):
        # Real: A's 2nd dispatch (6.0, SAME version_key (0,0)) comes after
        # the eval resolving the 1st (5.0) -> clean.
        real_agg = _agg_comm(dispatches=[("A", 0.0, "weights", (0, 0)),
                                          ("A", 6.0, "weights", (0, 0))],
                              resolves=[("A", 5.0, (0, 0))])
        # Sim: A re-dispatched at 2.0 for the SAME version_key (0,0) BEFORE
        # the eval resolving it (5.0) ran -> overlap = genuine duplicate work.
        sim_agg = _agg_comm(dispatches=[("A", 0.0, "weights", (0, 0)),
                                         ("A", 2.0, "weights", (0, 0))],
                             resolves=[("A", 5.0, (0, 0))])
        r = pc.inflight_overlap_parity(real_agg, sim_agg)
        assert not r["ok"]
        assert r["real_overlap_frac"] == 0.0
        assert r["sim_overlap_frac"] > 0.0

    def test_skips_without_comm_telemetry(self):
        # Sync baselines / runs predating the comm dispatch telemetry.
        agg = _agg()
        r = pc.inflight_overlap_parity(agg, agg)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_var_bad_same_version_key_is_a_real_violation(self):
        # var_bad IS new work, not a passive ping -- two of them for the
        # SAME version_key with no intervening resolve is a genuine
        # violation (duplicate work on the same iteration).
        agg = _agg_comm(dispatches=[("A", 0.0, "var_bad", (0, 0)),
                                     ("A", 2.0, "var_bad", (0, 0))])
        r = pc.inflight_overlap_parity(agg, agg)
        assert not r["ok"]
        assert r["sim_overlap_frac"] > 0.0

    def test_fedbuff_carried_surplus_new_version_key_is_not_a_violation(self):
        # A trainer's stale grad (dispatched at version_key (56, 12)) hasn't
        # resolved yet when handed genuinely NEW work for CURRENT cycle
        # (57, 10) -- FedBuff legitimately consumes the stale grad later
        # (down-weighted), so this must NOT be flagged.
        agg = _agg_comm(
            dispatches=[("A", 0.0, "weights", (56, 12)),
                        ("A", 41.7, "weights", (57, 10))],
            resolves=[("A", 44.2, (56, 12))],
        )
        r = pc.inflight_overlap_parity(agg, agg)
        assert r["ok"]
        assert r["real_overlap_frac"] == 0.0 and r["sim_overlap_frac"] == 0.0


class TestW1ComputeConservation:
    """W1 [DIAG]: forward passes per committed grad; a large sim excess over
    real = wasted recompute (the residence violation), localizes to R1."""

    def test_matched_ratio_passes(self):
        real_agg = _agg(agg_rounds=[_cyc(0, [("A", 0.0, 5.0), ("B", 0.0, 5.0)])])
        sim_agg = _agg(agg_rounds=[_cyc(0, [("A", 0.0, 5.0), ("B", 0.0, 5.0)])])
        # 2 committed each; ~1.5 forward passes per commit in both modes.
        real_tr = _trainers_with_rounds({"A": 2, "B": 1})
        sim_tr = _trainers_with_rounds({"A": 2, "B": 1})
        r = pc.compute_conservation_parity(real_agg, sim_agg, real_tr, sim_tr)
        assert r["ok"]
        assert r["real_fwd_per_commit"] == r["sim_fwd_per_commit"]

    def test_sim_recompute_excess_fails(self):
        real_agg = _agg(agg_rounds=[_cyc(0, [("A", 0.0, 5.0), ("B", 0.0, 5.0)])])
        sim_agg = _agg(agg_rounds=[_cyc(0, [("A", 0.0, 5.0), ("B", 0.0, 5.0)])])
        # Same 2 commits both modes, but sim ran ~2x the forward passes (the
        # residence-bug direction: sim OVER-computes -> violation).
        real_tr = _trainers_with_rounds({"A": 1, "B": 1})   # 2 fwd / 2 commit = 1.0
        sim_tr = _trainers_with_rounds({"A": 3, "B": 3})    # 6 fwd / 2 commit = 3.0
        r = pc.compute_conservation_parity(real_agg, sim_agg, real_tr, sim_tr)
        assert not r["ok"]
        assert r["sim_excess_rel"] > 0

    def test_sim_under_computing_is_ok(self):
        # W1 is ASYMMETRIC: sim doing FEWER forward passes than real (the async
        # start-tail, not wasted recompute) must NOT fail.
        real_agg = _agg(agg_rounds=[_cyc(0, [("A", 0.0, 5.0), ("B", 0.0, 5.0)])])
        sim_agg = _agg(agg_rounds=[_cyc(0, [("A", 0.0, 5.0), ("B", 0.0, 5.0)])])
        real_tr = _trainers_with_rounds({"A": 3, "B": 3})   # 3.0 fwd/commit
        sim_tr = _trainers_with_rounds({"A": 1, "B": 1})    # 1.0 fwd/commit
        r = pc.compute_conservation_parity(real_agg, sim_agg, real_tr, sim_tr)
        assert r["ok"] and r["sim_excess_rel"] < 0

    def test_skips_without_data(self):
        empty = _agg(agg_rounds=[])
        r = pc.compute_conservation_parity(empty, empty, {}, {})
        assert r.get("status") == "SKIP" and r["ok"]


class TestR1W1Registered:
    """Both rungs run in run_all_parity and are wired into the causal registry
    (R1 upstream of V1 -- the dep chain proving cadence is downstream)."""

    def test_present_in_run_all_and_meta(self):
        assert "r1_inflight_overlap" in pc.CHECK_META
        assert "w1_compute_conservation" in pc.CHECK_META
        assert pc.CHECK_META["w1_compute_conservation"]["deps"] == ("r1_inflight_overlap",)

    def test_r1_is_upstream_of_the_cadence_chain(self):
        """REACHABILITY, not a direct edge: `v1c_iter_drift_rate` was inserted
        between R1 and V1, so R1 reaches V1 one hop further down. What the
        registry must preserve is that cadence stays DOWNSTREAM of R1."""
        seen, stack = set(), ["v1_iter_per_data_id"]
        while stack:
            node = stack.pop()
            for dep in pc.CHECK_META.get(node, {}).get("deps", ()):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        assert "r1_inflight_overlap" in seen
        assert "v1c_iter_drift_rate" in seen


def _agg_rd(rows):
    """agg dict carrying `redispatch_decomp` tripwire rows:
    (outstanding, target, retask)."""
    return {"selection_train": [], "agg_rounds": [], "agg_evals": [],
            "redispatch_decomp": [
                {"event": "redispatch_decomp", "ts": float(i), "end_id": "A",
                 "outstanding_at_dispatch": o, "concurrency_target": t,
                 "retask_before_close": rt}
                for i, (o, t, rt) in enumerate(rows)]}


def _agg_ci(intervals, target=30):
    """agg dict whose agg_rounds carry `contributor_intervals` [(dispatch, commit)]
    plus one tripwire row to supply `concurrency_target` (both modes share the
    same configured `c`, so either side's row can name it)."""
    return {"selection_train": [], "agg_evals": [],
            "agg_rounds": [{"event": "agg_round", "round": 1, "ts": 0.0,
                            "contributor_intervals": [
                                {"end": f"e{i}", "dispatch_ts": a, "commit_ts": b}
                                for i, (a, b) in enumerate(intervals)]}],
            "redispatch_decomp": [
                {"event": "redispatch_decomp", "ts": 0.0, "end_id": "A",
                 "outstanding_at_dispatch": 1, "concurrency_target": target,
                 "retask_before_close": False}]}


def _iv_round(intervals, ts=0.0):
    """agg_round carrying contributor_intervals: [(end, dispatch, commit), ...]."""
    return {"event": "agg_round", "round": 1, "ts": ts, "cycle_data_id": 0,
            "var_good_enough": True, "contributing_trainers": [i[0] for i in intervals],
            "staleness": [0], "agg_goal_count": 1,
            "contributor_intervals": [
                {"end": e, "dispatch_ts": a, "commit_ts": b} for e, a, b in intervals]}


class TestKsBlindToLevelShift:
    """KS barely moves under a uniform shift, so a rung graded on KS ALONE cannot
    see a systematic level divergence -- the failure mode `v2_var_trajectory`
    documents and guards, which several sibling rungs never got."""

    def test_grad_norm_level_shift_fails_despite_clean_ks(self):
        def _run(scale, vclock=False):
            return _agg(agg_rounds=[
                _fwd_cycle(d, ts=float(d), vclock=(float(d) if vclock else None))
                | {"grad_norm": [100.0 * scale + d, 101.0 * scale + d]}
                for d in range(40)])
        r = pc.grad_norm_parity(_run(1.0), _run(1.25, vclock=True))
        assert r["mean_rel_diff"] > 0.1
        assert not r["ok"], r

    def test_grad_norm_matched_passes(self):
        run = _agg(agg_rounds=[
            _fwd_cycle(d, ts=float(d)) | {"grad_norm": [100.0 + d]}
            for d in range(40)])
        assert pc.grad_norm_parity(run, run)["ok"]


class TestSelectionSpeedBiasGradesTheBias:
    """The rung is NAMED for the bias and already computed it -- it just never
    graded it. Both modes draw from an identical pool, so a bias difference is
    the selector's, not the input's."""

    @staticmethod
    def _sel_events(selected_speeds, pool_speeds):
        per = {}
        for i, sp in enumerate(pool_speeds):
            per[f"p{i}"] = {"speed_s": sp, "selected": False}
        for i, sp in enumerate(selected_speeds):
            per[f"s{i}"] = {"speed_s": sp, "selected": True}
        return _agg(selection=[{"event": "selection", "task": "train", "round": 1,
                                "ts": 0.0, "chosen": [f"s{i}" for i in
                                                      range(len(selected_speeds))],
                                "per_trainer": per}])

    def test_sim_speed_biased_against_its_own_pool_fails(self):
        pool = [10.0, 12.0, 14.0, 16.0]
        real = self._sel_events([12.0, 14.0], pool)      # ~pool mean
        sim = self._sel_events([16.0, 16.0], pool)       # biased slow
        r = pc.selection_speed_bias_parity(real, sim)
        assert r["bias_rel_diff"] > 0.10
        assert not r["ok"], r

    def test_matched_bias_passes(self):
        pool = [10.0, 12.0, 14.0, 16.0]
        run = self._sel_events([12.0, 14.0], pool)
        r = pc.selection_speed_bias_parity(run, run)
        assert r["ok"] and r["bias_rel_diff"] == pytest.approx(0.0, abs=1e-6)

    def test_ks_failure_still_fails(self):
        pool = [10.0, 12.0, 14.0, 16.0]
        real = self._sel_events([10.0, 10.0, 10.0, 10.0], pool)
        sim = self._sel_events([16.0, 16.0, 16.0, 16.0], pool)
        assert not pc.selection_speed_bias_parity(real, sim)["ok"]


class TestSlotUtilization:
    """PEAK answers 'did anyone exceed c'. It cannot answer 'did anyone fail to
    USE c', and that second failure cost fluxtune 16% of its throughput while
    `concurrency_cap` read a clean 30/30 on both modes."""

    @staticmethod
    def _steady(n_busy, span=100.0, c_marker=None, target_c=30):
        """n_busy ends held continuously, plus one brief spike to `c_marker` so
        PEAK matches across modes while MEAN does not. `redispatch_decomp`
        carries the run's own `c` (what the cap grades against)."""
        iv = [(f"e{i}", 0.0, span) for i in range(n_busy)]
        if c_marker:
            iv += [(f"spike{i}", 0.0, 0.5) for i in range(c_marker - n_busy)]
        run = _agg(agg_rounds=[_iv_round(iv)])
        run["redispatch_decomp"] = [
            {"event": "redispatch_decomp", "outstanding_at_dispatch": n_busy,
             "concurrency_target": target_c}]
        return run

    def test_matched_utilization_passes(self):
        r = pc.slot_utilization_parity(self._steady(20), self._steady(20))
        assert r["ok"], r
        assert r["real_mean_inflight"] == pytest.approx(20, abs=0.5)

    def test_chronic_underfill_fails_even_when_peak_matches(self):
        # THE fluxtune signature: identical peak, very different mean.
        real = self._steady(29, c_marker=30)
        sim = self._steady(24, c_marker=30)
        cap = pc.concurrency_cap_ok(real, sim)
        util = pc.slot_utilization_parity(real, sim)
        assert cap["real_peak_inflight"] == cap["sim_peak_inflight"] == 30
        assert not util["ok"], util
        assert util["under_filling_mode"] == "sim"

    def test_names_the_under_filling_side(self):
        r = pc.slot_utilization_parity(self._steady(15), self._steady(25))
        assert r["under_filling_mode"] == "real"

    def test_occupancy_is_time_weighted_not_event_weighted(self):
        # 2 ends busy the whole span, plus 20 ends busy for 1% of it. An
        # event-weighted mean would read ~20; time-weighted reads ~2.
        iv = [("a", 0.0, 100.0), ("b", 0.0, 100.0)]
        iv += [(f"blip{i}", 50.0, 51.0) for i in range(20)]
        one = _agg(agg_rounds=[_iv_round(iv)])
        r = pc.slot_utilization_parity(one, one)
        assert r["real_mean_inflight"] < 3.0, r
        assert r["ok"]

    def test_skips_without_contributor_intervals(self):
        assert pc.slot_utilization_parity(_agg(), _agg())["status"] == "SKIP"

    def test_gates_as_a_mechanism_downstream_of_the_cap(self):
        meta = pc.CHECK_META["slot_utilization"]
        assert meta["role"] == "MECHANISM" and "concurrency_cap" in meta["deps"]


class TestConcurrencyCapReportsCentralOccupancy:
    def test_cap_reports_mean_and_median_alongside_peak(self):
        real = TestSlotUtilization._steady(29, c_marker=30)
        sim = TestSlotUtilization._steady(24, c_marker=30)
        r = pc.concurrency_cap_ok(real, sim)
        # The INV verdict still rests on peak, and peak is clean...
        assert r["ok"], r
        # ...but the central statistics now make the divergence visible.
        assert r["real_median_inflight"] != r["sim_median_inflight"]
        assert r["real_mean_inflight"] > r["sim_mean_inflight"]

    def test_peak_still_decides_the_invariant(self):
        over = TestSlotUtilization._steady(40, c_marker=None)
        ok = TestSlotUtilization._steady(20, c_marker=30)
        r = pc.concurrency_cap_ok(ok, over)
        assert not r["ok"] and "sim" in r["offending_modes"]


class TestConcurrencyCapTripwire:
    """[INV, per mode] ends in flight at any instant <= that mode's own selector
    `c`. Single-side decidable -- graded per mode, never as a diff.

    Graded as peak INTERVAL OVERLAP on each mode's own clock (§D-20), not as the
    aggregator's pending-set size at dispatch instants: that reading counted
    committed-but-not-yet-released ends (phantom +1) and, sampling only at
    dispatch instants, missed a genuine 2x breach entirely."""

    def test_both_within_cap_passes(self):
        # 3 fully overlapping intervals, c=30.
        agg = _agg_ci([(0.0, 10.0), (1.0, 11.0), (2.0, 12.0)])
        r = pc.concurrency_cap_ok(agg, agg)
        assert r["ok"] and r["offending_modes"] == []
        assert r["real_peak_inflight"] == 3 and r["target_c"] == 30

    def test_sim_breach_fails_and_names_sim_only(self):
        real = _agg_ci([(0.0, 10.0), (20.0, 30.0)], target=1)   # never overlaps
        sim = _agg_ci([(0.0, 10.0), (1.0, 11.0)], target=1)     # overlaps -> 2 > 1
        r = pc.concurrency_cap_ok(real, sim)
        assert not r["ok"] and r["offending_modes"] == ["sim"], r
        assert r["sim_peak_inflight"] == 2 and r["real_peak_inflight"] == 1, r

    def test_real_breach_fails_the_real_side(self):
        real = _agg_ci([(0.0, 10.0), (1.0, 11.0)], target=1)
        sim = _agg_ci([(0.0, 10.0), (20.0, 30.0)], target=1)
        r = pc.concurrency_cap_ok(real, sim)
        assert not r["ok"] and r["offending_modes"] == ["real"], r

    def test_skips_without_contributor_intervals(self):
        r = pc.concurrency_cap_ok(_agg(), _agg())
        assert r.get("status") == "SKIP" and r["ok"]

    def test_pre_tripwire_real_leg_is_gradeable_from_intervals_alone(self):
        """The old rung reported real UNGRADED whenever its leg predated the
        dispatch tripwire. `contributor_intervals` is emitted by every leg, so
        real is now graded from its own telemetry -- and the target falls back to
        sim's row, since both modes run the same configured `c`."""
        real = {"selection_train": [], "agg_evals": [],
                "agg_rounds": [{"event": "agg_round", "round": 1, "ts": 0.0,
                                "contributor_intervals": [
                                    {"end": "a", "dispatch_ts": 0.0, "commit_ts": 9.0},
                                    {"end": "b", "dispatch_ts": 1.0, "commit_ts": 8.0},
                                ]}]}          # no redispatch_decomp rows at all
        sim = _agg_ci([(0.0, 10.0)], target=1)
        r = pc.concurrency_cap_ok(real, sim)
        assert r["ungraded_modes"] == [] and r["real_peak_inflight"] == 2, r
        assert not r["ok"] and r["offending_modes"] == ["real"], r

    def test_boundary_double_cohort_is_caught(self):
        """fedbuff_round's shape: a second cohort of DIFFERENT ends dispatched
        while the first is still in flight -> 2x the cap."""
        first = [(0.0, 100.0)] * 30
        second = [(50.0, 150.0)] * 30      # overlaps the first by 50
        sim = _agg_ci(first + second)      # _agg_ci names every end uniquely
        r = pc.concurrency_cap_ok(_agg_ci([(0.0, 1.0)]), sim)
        assert not r["ok"] and r["offending_modes"] == ["sim"], r
        assert r["sim_peak_inflight"] == 60, r

    def test_counts_distinct_ends_not_intervals(self):
        """One end holding two concurrent dispatches is ONE busy trainer, so it
        must not inflate the cap reading -- counting intervals reported
        fedbuff_round at 61 where only 35 distinct ends were in flight."""
        agg = {"selection_train": [], "agg_evals": [],
               "agg_rounds": [{"event": "agg_round", "round": 1, "ts": 0.0,
                               "contributor_intervals": [
                                   {"end": "a", "dispatch_ts": 0.0, "commit_ts": 10.0},
                                   {"end": "a", "dispatch_ts": 1.0, "commit_ts": 11.0},
                                   {"end": "b", "dispatch_ts": 2.0, "commit_ts": 12.0},
                               ]}],
               "redispatch_decomp": [
                   {"event": "redispatch_decomp", "ts": 0.0, "end_id": "A",
                    "outstanding_at_dispatch": 1, "concurrency_target": 30,
                    "retask_before_close": False}]}
        r = pc.concurrency_cap_ok(_agg_ci([(0.0, 1.0)]), agg)
        assert r["sim_peak_inflight"] == 2, r        # not 3
        assert r["sim_n_self_overlap_dispatches"] == 1, r

    def test_same_end_concurrent_dispatch_fails_even_under_the_cap(self):
        """§F-25: one instruction per version_key. Two live dispatches to one end
        is a violation regardless of how far below `c` the run is."""
        agg = {"selection_train": [], "agg_evals": [],
               "agg_rounds": [{"event": "agg_round", "round": 1, "ts": 0.0,
                               "contributor_intervals": [
                                   {"end": "a", "dispatch_ts": 0.0, "commit_ts": 10.0},
                                   {"end": "a", "dispatch_ts": 1.0, "commit_ts": 11.0},
                               ]}],
               "redispatch_decomp": [
                   {"event": "redispatch_decomp", "ts": 0.0, "end_id": "A",
                    "outstanding_at_dispatch": 1, "concurrency_target": 30,
                    "retask_before_close": False}]}
        r = pc.concurrency_cap_ok(_agg_ci([(0.0, 1.0)]), agg)
        assert not r["ok"] and r["offending_modes"] == ["sim"], r
        assert r["sim_peak_inflight"] == 1 <= r["target_c"], r


class TestRetaskBeforeCloseTripwire:
    """[INV, per mode] rate of dispatches to an end that already contributed to
    the still-open agg cycle must be 0 -- the direct §D-15 root-cause detector."""

    def test_zero_rate_passes(self):
        agg = _agg_rd([(5, 30, False), (6, 30, False)])
        r = pc.retask_before_close_ok(agg, agg)
        assert r["ok"] and r["sim_retask_frac"] == 0.0

    def test_any_sim_retask_fails(self):
        real = _agg_rd([(5, 30, False), (6, 30, False)])
        sim = _agg_rd([(5, 30, True), (6, 30, False)])
        r = pc.retask_before_close_ok(real, sim)
        assert not r["ok"] and r["offending_modes"] == ["sim"]
        assert r["sim_retask_frac"] == 0.5 and r["real_retask_frac"] == 0.0

    def test_skips_without_tripwire_telemetry(self):
        r = pc.retask_before_close_ok(_agg(), _agg())
        assert r.get("status") == "SKIP" and r["ok"]

    def test_old_real_leg_reads_ungraded_not_a_measured_zero(self):
        r = pc.retask_before_close_ok(_agg(), _agg_rd([(5, 30, False)]))
        assert r["ungraded_modes"] == ["real"]
        assert r["real_retask_frac"] is None and r["sim_retask_frac"] == 0.0
        assert r["ok"]


class TestDispatchTripwiresRegistered:
    """Both tripwires are wired into the causal registry upstream of R1: a
    contribution-level R1 pass does not clear the dispatch path (§D-15 trap 3)."""

    def test_present_in_meta_with_dispatch_upstream_of_r1(self):
        assert pc.CHECK_META["concurrency_cap"]["deps"] == ()
        assert pc.CHECK_META["retask_before_close"]["deps"] == ("concurrency_cap",)
        assert "retask_before_close" in pc.CHECK_META["r1_inflight_overlap"]["deps"]


def _lcyc(data_id, iteration, cohort, var, var_good=False, force=False, goal=3,
          is_async=False):
    """One fwdllm variance-cadence cycle event (cohort in receive/commit order)."""
    return {"event": "agg_round", "round": 1,
            "cycle_data_id": data_id, "iteration_per_data_id": iteration,
            "contributing_trainers": list(cohort), "var": var,
            "var_good_enough": var_good, "force_commit_planned": force,
            "agg_goal_count": goal, "staleness": [0] * len(cohort),
            "is_async": is_async}


class TestParticipationParityFwdllmWindowing:
    """S2 (participation_parity) normally keys its matched-window on `round`
    (increments per cohort for felix/oort/fedbuff); fwdllm's round is coarse
    (advances only once every data_id finishes), so it keys on cycle position
    instead -- otherwise every cohort landed in the same round-bucket and the
    window degenerated to n=1, comparing full-run totals unmatched."""

    def test_fwdllm_matched_window_ignores_pure_throughput_gap(self):
        # Real completes 3 cohorts, sim completes 6 -- same shape, pure
        # throughput gap. Round-keying would compare real's 3-cohort total
        # against sim's full 6-cohort total unmatched (a false failure);
        # cycle-keying matches on the first 3 of each.
        real = _agg(agg_rounds=[_lcyc(i, 1, ["a", "b"], 0.5) for i in range(3)])
        sim = _agg(agg_rounds=[_lcyc(i, 1, ["a", "b"], 0.5) for i in range(6)])
        r = pc.participation_parity(real, sim)
        assert r["n_rounds_matched"] == 3
        assert r["ok"], r

    def test_fwdllm_matched_window_catches_real_shape_divergence(self):
        # Same cohort size and total commits both modes (9 each), but the
        # participation SHAPE differs sharply: real concentrates on one
        # trainer, sim spreads evenly across three -- a genuine divergence the
        # matched window must still catch (KS is on the count-VALUE
        # distribution, so this needs a real skew, not just relabeled counts).
        real = _agg(agg_rounds=[_lcyc(i, 1, ["a", "a", "a"], 0.5) for i in range(3)])
        sim = _agg(agg_rounds=[_lcyc(i, 1, ["a", "b", "c"], 0.5) for i in range(3)])
        r = pc.participation_parity(real, sim)
        assert r["n_rounds_matched"] == 3
        assert not r["ok"], r

    def test_non_fwdllm_still_windows_by_round(self):
        # async_cifar10-shape events (no cycle_data_id/var_good_enough) keep the
        # original round-keyed behavior, unchanged.
        real = _agg(agg_rounds=[_round(0, ["a", "b"], [0, 0]),
                                _round(1, ["a", "b"], [0, 0])])
        sim = _agg(agg_rounds=[_round(0, ["a", "b"], [0, 0]),
                               _round(1, ["a", "b"], [0, 0])])
        r = pc.participation_parity(real, sim)
        assert r["n_rounds_matched"] == 2
        assert r["ok"], r


class TestCohortSequence:
    """L1 cohort_sequence_parity: the ordered per-aggregation logical sequence
    (set + receive-ORDER + cadence + var value) must be IDENTICAL. EXACT, ungated
    (real receive-order is deterministic in both modes by design)."""

    def test_identical_sequence_passes(self):
        cyc = [_lcyc(0, 1, ["a", "b", "c"], 0.9),
               _lcyc(0, 2, ["a", "b", "c"], 0.28, var_good=True),
               _lcyc(1, 1, ["a", "b", "c"], 0.5)]
        r = pc.cohort_sequence_parity(_agg(agg_rounds=list(cyc)),
                                      _agg(agg_rounds=list(cyc)))
        assert r["ok"], r
        assert r["order_match_frac"] == 1.0 and r["var_match_frac"] == 1.0

    def test_reordered_cohort_same_set_is_benign_for_sync(self):
        # SYNC receive-ORDER is SOFT (fedavg order-invariant, ties
        # canonicalize) -- a same-set/same-var reorder no longer fails,
        # though order_match_frac still surfaces it for diagnosis.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.9, is_async=False)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["c", "a", "b"], 0.9, is_async=False)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"], r
        assert r["set_match_frac"] == 1.0 and r["order_match_frac"] == 0.0
        assert r["order_gates_ok"] is False

    def test_reordered_cohort_same_set_FAILS_for_async(self):
        # #N: the same reorder DOES fail when the cycle is ASYNC -- order is
        # only SOFT for sync.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.9, is_async=True)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["c", "a", "b"], 0.9, is_async=True)])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"], r
        assert r["order_gates_ok"] is True
        assert r["first_divergence"]["order_ok"] is False

    def test_different_cohort_FAILS(self):
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.9)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "d"], 0.9)])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"] and r["set_match_frac"] == 0.0

    def test_async_boundary_cascade_tolerated_distributionally(self):
        # A boundary arrival race shifts one trainer across each cohort boundary
        # (fast-late vs slow-early), so each cycle overlaps 9/10 with the other
        # mode but is never exact -- must pass DISTRIBUTIONALLY (set_overlap_frac
        # >= tol), not fail as a mix bug (which S2/participation_parity owns).
        base = [f"t{i}" for i in range(11)]         # cohorts of 10 from 11 ids
        real = _agg(agg_rounds=[
            _lcyc(0, 1, base[0:10], 0.5, is_async=True, goal=10),
            _lcyc(0, 2, base[1:11], 0.4, is_async=True, goal=10)])
        sim = _agg(agg_rounds=[
            _lcyc(0, 1, base[1:11], 0.5, is_async=True, goal=10),   # shifted by one
            _lcyc(0, 2, base[0:10], 0.4, is_async=True, goal=10)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"], r
        assert r["set_match_frac"] == 0.0 and r["set_dist_frac"] == 1.0
        assert r["set_overlap_frac"] >= 0.8

    def test_async_low_overlap_still_FAILS(self):
        # A genuine selection divergence (overlap < tol) is NOT absorbed --
        # distributional grading tolerates boundary races, not real mix bugs.
        # NB: no selector telemetry -> deterministic fallback -> ENFORCED (the
        # stochastic-selector gate below is what changes this).
        real = _agg(agg_rounds=[_lcyc(0, 1, [f"t{i}" for i in range(10)], 0.5,
                                      is_async=True, goal=10)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, [f"t{i}" for i in range(5, 15)], 0.5,
                                     is_async=True, goal=10)])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"] and r["set_overlap_frac"] < 0.8

    def test_async_stochastic_selector_gates_identity_composition(self):
        # fluxtune regime: async + stochastic-SUBSET selector (num_chosen<pool).
        # The marginal cohort slot is a physical-arrival vs modeled-sct boundary
        # race that cascades, decorrelating index-paired membership to the
        # independent-draw floor -- unattainable, not a bug. composition +
        # first-bin SET gate to diagnostic; COUNT stays enforced, S2 owns the
        # mix catch.
        base = [f"t{i}" for i in range(20)]
        sel = [_selc(1, base[:10], 20)]                     # subset -> stochastic
        real = _agg(selection=sel, agg_rounds=[
            _lcyc(0, 1, base[0:10], 0.5, is_async=True, goal=10)])
        sim = _agg(selection=sel, agg_rounds=[
            _lcyc(0, 1, base[8:18], 0.5, is_async=True, goal=10)])  # low overlap
        r = pc.cohort_sequence_parity(real, sim)
        assert r["identity_gated"] is True
        assert r["composition"]["gated_stochastic"] is True
        assert r["ok"], r                                   # gated -> passes on COUNT
        assert r["composition"]["independent_draw_floor"] is not None

    def test_async_stochastic_still_enforces_count(self):
        # Gating IDENTITY does not gate THROUGHPUT: a cohort-COUNT drift beyond
        # tol still fails even for a stochastic async selector.
        base = [f"t{i}" for i in range(20)]
        sel = [_selc(1, base[:10], 20)]
        real = _agg(selection=sel, agg_rounds=[
            _lcyc(0, i, base[0:10], 0.5, is_async=True, goal=10) for i in range(2)])
        sim = _agg(selection=sel, agg_rounds=[
            _lcyc(0, i, base[0:10], 0.5, is_async=True, goal=10) for i in range(20)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["identity_gated"] is True
        assert not r["ok"] and not r["count"]["ok"]

    def test_var_divergence_FAILS_even_with_matched_order(self):
        # Identical cohort+order, var off by >0.1% -> the RNG-desync tell.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.371605)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.371067)])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"] and r["var_match_frac"] == 0.0
        assert r["first_divergence"]["var_ok"] is False

    def test_cadence_shift_FAILS(self):
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a"], 0.5), _lcyc(1, 0, ["a"], 0.2, var_good=True)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a"], 0.5), _lcyc(0, 2, ["a"], 0.4)])  # extra iter
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"] and r["cadence_match_frac"] < 1.0

    def test_tier_exact_and_enforced(self):
        # EXACT tier -> enforced FAIL even under --lenient (not a DIST/DIAG warn).
        r = pc.cohort_sequence_parity(
            _agg(agg_rounds=[_lcyc(0, 1, ["a"], 0.5)]),
            _agg(agg_rounds=[_lcyc(0, 1, ["b"], 0.5)]))
        assert r["tier"] == "EXACT" and r["ok"] is False
        passed, roots, _down, _warn = pc.overall_verdict(
            {"cohort_sequence": r}, lenient=True)
        assert not passed and "cohort_sequence" in roots

    def test_skips_on_non_fwdllm(self):
        # No cadence fields (async_cifar10 shape) -> clean SKIP, byte-identical.
        plain = {"event": "agg_round", "round": 0, "ts": 0.0,
                 "contributing_trainers": ["a"], "staleness": [0]}
        rd = _agg(agg_rounds=[plain])
        r = pc.cohort_sequence_parity(rd, rd)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_max_bin_windows_to_first_bin(self):
        # Cadence matches on bin 0, diverges on bin 1 (var_good flips) ->
        # --max-bin 0 passes; the default (cap=1) window still sees bin 1 and fails.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                                _lcyc(1, 1, ["a", "b"], 0.2, var_good=True)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                               _lcyc(1, 1, ["a", "b"], 0.5, var_good=False)])
        assert pc.cohort_sequence_parity(real, sim, max_bin=0)["ok"]
        assert not pc.cohort_sequence_parity(real, sim)["ok"]

    def test_cadence_divergence_beyond_bin1_not_enforced_by_default(self):
        # cadence/var EXACT is HARD only through bin 1 (float-nondeterminism
        # wall starts ~bin 7) -- a divergence beyond it does NOT fail
        # cohort_sequence; that's v1/v2/v4/v5's (DISTRIBUTIONAL) job.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                                _lcyc(7, 2, ["a", "b"], 0.26, var_good=False)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                               _lcyc(7, 2, ["a", "b"], 0.31, var_good=True)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"], r
        assert r["cadence_var_order_max_bin"] == 1

    def test_set_divergence_beyond_bin1_not_enforced_by_default(self):
        # A boundary-race SET mismatch past bin-1 is a cascade artifact, not a
        # bug, so `first_bin_logical_ok` stays unenforced past the wall.
        # `composition` (separate, full-run) still grades it distributionally,
        # so overall `ok` correctly fails here -- that's composition's job,
        # not a regression of the bin-1 cap this test targets.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                                _lcyc(7, 2, ["a", "b"], 0.5)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                               _lcyc(7, 2, ["a", "c"], 0.5)])  # different SET
        r = pc.cohort_sequence_parity(real, sim)
        assert r["first_bin_logical_ok"], r
        assert r["cadence_var_order_max_bin"] == 1

    def test_set_divergence_within_bin1_still_enforced(self):
        # The achievable window itself stays HARD -- a genuine divergence at
        # or before max_bin still fails (test_different_cohort_FAILS covers
        # the single-cycle case; this checks a 2-cycle run where BOTH cycles
        # are within the default bin-1 window).
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                                _lcyc(1, 2, ["a", "b"], 0.5)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                               _lcyc(1, 2, ["a", "c"], 0.5)])  # different SET
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        assert r["set_match_frac"] < 1.0
        assert r["set_divergence"]["cycle_index"] == 1

    def test_present_in_run_all_and_meta(self):
        assert "cohort_sequence" in pc.CHECK_META
        assert pc.CHECK_META["cohort_sequence"]["deps"]


class TestCohortSequenceCountLogicalBudget:
    """`count` filters to the matched LOGICAL budget N (cohorts whose progress <= N,
    the common data_id prefix), not raw full-run totals -- a run still going in one
    mode must not false-fail count. On this axis `count` is rolled-up V1: it fires
    only when a side does MORE aggregation cycles to reach the SAME data_ids."""

    def _cyc(self, data_id, ts=None, vclock=None, iteration=1):
        e = _lcyc(data_id, iteration, ["a"], 0.5)
        if ts is not None:
            e["ts"] = ts
        if vclock is not None:
            e["vclock_now"] = vclock
        return e

    def test_pure_throughput_gap_absorbed_by_logical_budget(self):
        # real: 10 cohorts over data_id 0..9. sim: 15 cohorts over data_id 0..14
        # -- same 1-cohort-per-data_id shape, sim just kept running past the shared
        # prefix. Raw counts (10 vs 15) would false-fail; filtered to N=data_id 9,
        # both hold exactly 10 (1 cohort/data_id each).
        real = _agg(agg_rounds=[self._cyc(i, ts=float(i)) for i in range(10)])
        sim = _agg(agg_rounds=[self._cyc(i, vclock=float(i)) for i in range(15)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["count"]["ok"], r["count"]
        assert r["count"]["n_real_cohorts"] == 10
        assert r["count"]["n_sim_cohorts"] == 10
        assert r["count"]["matched_logical_budget_n"] == 10

    def test_extra_cycles_to_reach_same_data_ids_fails(self):
        # Genuine cohort-count divergence on the logical axis: sim does 2
        # aggregation cycles per data_id (variance retries) to reach the SAME
        # data_ids 0..9 real reaches in 1 each -- 20 vs 10 cohorts at N=data_id 9.
        real = _agg(agg_rounds=[self._cyc(d, ts=float(d)) for d in range(10)])
        sim_rounds = []
        for d in range(10):
            sim_rounds.append(self._cyc(d, vclock=float(2 * d), iteration=1))
            sim_rounds.append(self._cyc(d, vclock=float(2 * d + 1), iteration=2))
        sim = _agg(agg_rounds=sim_rounds)
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["count"]["ok"], r["count"]
        assert r["count"]["n_real_cohorts"] == 10 and r["count"]["n_sim_cohorts"] == 20


# Real trainer_registry.yaml task_ids (lib/python/examples/_metadata) with known
# raw training_delay_s, used to exercise the tie-window contention logic below
# without mocking the registry: 370=4.0s, 375=5.0s (1.0s apart -- boundary tie),
# 371=16.0s (far from 370 -- never a tie).
_TID_370 = "505f9fc483cf4df68a2409257b5fad7d3c580370"
_TID_375 = "505f9fc483cf4df68a2409257b5fad7d3c580375"
_TID_371 = "505f9fc483cf4df68a2409257b5fad7d3c580371"


def _with_delay_cfg(agg: dict, divisor: float = 1.0, floor_s: float = 0.0) -> dict:
    agg["training_delay_factor"] = divisor
    agg["training_delay_floor_s"] = floor_s
    return agg


class TestCohortSequenceTieWindow:
    """A committed-cohort divergence at a near-degenerate fast class is an
    arrival race, not a bug, when every differing trainer's EXPECTED delay
    (registry, divisor-scaled) is within `tie_window_s` of the others' --
    granted a TIE instead of a hard fail. Ungrantable (no delay model, or an
    unknown trainer) stays strict."""

    def test_set_swap_within_tie_window_is_granted(self):
        # 370 (4.0s) <-> 375 (5.0s): 1.0s apart, exactly at the default window.
        real = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, ["372", "373", _TID_370], 0.5)]))
        sim = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, ["372", "373", _TID_375], 0.5)]))
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"], r
        assert r["set_match_frac"] == 0.0 and r["set_tie_frac"] == 1.0
        assert r["set_divergence"] is None
        assert r["delay_model_available"] is True

    def test_set_swap_beyond_tie_window_still_fails(self):
        # 370 (4.0s) vs 371 (16.0s): 12.0s apart, far outside the window.
        real = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, ["372", "373", _TID_370], 0.5)]))
        sim = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, ["372", "373", _TID_371], 0.5)]))
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        assert r["set_tie_frac"] == 0.0
        assert r["set_divergence"] is not None

    def test_order_swap_within_tie_window_is_granted_for_async(self):
        real = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, [_TID_370, _TID_375], 0.5, is_async=True)]))
        sim = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, [_TID_375, _TID_370], 0.5, is_async=True)]))
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"], r
        assert r["first_divergence"] is None

    def test_order_swap_beyond_tie_window_still_fails_for_async(self):
        real = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, [_TID_370, _TID_371], 0.5, is_async=True)]))
        sim = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, [_TID_371, _TID_370], 0.5, is_async=True)]))
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        assert r["first_divergence"]["order_ok"] is False

    def test_unknown_trainer_falls_back_to_strict_even_with_divisor(self):
        # Neither "unknown_x" nor "unknown_y" is in the registry -- the tie
        # can't be assessed, so it isn't granted (no silent free pass).
        real = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, [_TID_370, "unknown_x"], 0.5)]))
        sim = _with_delay_cfg(_agg(agg_rounds=[
            _lcyc(0, 1, [_TID_370, "unknown_y"], 0.5)]))
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        assert r["set_tie_frac"] == 0.0

    def test_no_delay_model_falls_back_to_strict(self):
        # No training_delay_factor on either side (pre-knob run) -> exp_map is
        # None -> byte-identical to the pre-tie-window behavior.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["372", "373", _TID_370], 0.5)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["372", "373", _TID_375], 0.5)])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        assert r["delay_model_available"] is False


def _sel_full(ts=None, vclock_now=None, per_trainer=None, chosen=None):
    return {"event": "selection", "task": "train", "round": 1,
            "ts": ts, "vclock_now": vclock_now,
            "chosen": chosen or [], "per_trainer": per_trainer or {}}


def _cyc_with_boundary(data_id, iteration, cohort, var, commit_ts_by_end):
    """_lcyc plus contributor_intervals (needed for _cohort_boundary_ts) --
    the LAST entry in `cohort` is the boundary trainer (mirrors real
    fwdllm telemetry, where the agg-goal-th commit closes the cycle)."""
    c = _lcyc(data_id, iteration, cohort, var)
    c["contributor_intervals"] = [
        {"end": end, "commit_ts": ts} for end, ts in commit_ts_by_end.items()
    ]
    return c


class TestTrainerSpeedIdentityGating:
    """P3b: per-trainer SPEED identity is registry-assigned -> always enforced.
    Per-trainer UTILITY is loss-on-current-model (path-dependent), so for a
    stochastic subset selector it is a gated diagnostic (utility_parity owns the
    distribution)."""

    @staticmethod
    def _sel_pt(per_trainer, chosen=("a",), ncand=10):
        e = _selc(1, list(chosen), ncand)      # subset (ncand>chosen) -> stochastic
        e["per_trainer"] = per_trainer
        return e

    def test_utility_identity_gated_for_stochastic(self):
        real = _agg(selection=[self._sel_pt({"a": {"speed_s": 8.0, "utility": 6.0}})] * 3)
        sim = _agg(selection=[self._sel_pt({"a": {"speed_s": 8.0, "utility": 9.0}})] * 3)
        r = pc.trainer_speed_identity_parity(real, sim)
        assert r["utility"]["gated_stochastic"] is True
        assert r["utility"]["ok"] is False       # divergence still REPORTED
        assert r["speed_s"]["gated_stochastic"] is False
        assert r["ok"]                            # but utility identity does not gate

    def test_speed_identity_enforced_even_when_stochastic(self):
        real = _agg(selection=[self._sel_pt({"a": {"speed_s": 8.0, "utility": 6.0}})] * 3)
        sim = _agg(selection=[self._sel_pt({"a": {"speed_s": 12.0, "utility": 6.0}})] * 3)
        r = pc.trainer_speed_identity_parity(real, sim)
        assert not r["speed_s"]["ok"] and not r["ok"]   # speed is registry-fixed


class TestCohortFirstCommitRaceDiagnostic:
    """`first_commit_race_diagnostic` reports whether a SET divergence is
    explained by a trainer's first-ever exploring transition racing the
    cohort boundary (real=raw FIFO jitter, sim=clean sct-order). Must NEVER
    affect `ok`/`set_ok`, and must NOT explain away a divergence that isn't
    actually a near-tie."""

    # A single-cycle cohort list means the differing member (458/405) can't
    # appear "nearby" in the other mode's list, so _cohort_set_tie_ok
    # correctly doesn't grant a tie here -- set_divergence is populated and
    # this diagnostic actually runs.

    def _scale_event(self):
        # LATE event (scanned last by _run_utility_rank_gap_scale) with a
        # clean, evenly-spaced utility distribution -> noise-floor scale=1.0.
        return _sel_full(ts=200.0, per_trainer={
            "u1": {"utility": 1.0}, "u2": {"utility": 2.0},
            "u3": {"utility": 3.0}, "u4": {"utility": 4.0}, "u5": {"utility": 5.0},
        })

    def test_near_tie_is_explained(self):
        real = _agg(agg_rounds=[_cyc_with_boundary(
            0, 1, ["372", "373", "458"], 0.5,
            {"372": 100.0, "373": 101.0, "458": 102.0})],
            selection=[
                _sel_full(ts=50.0, per_trainer={
                    "372": {"utility": 5.0}, "373": {"utility": 4.5},
                    "405": {"utility": 4.55}, "458": {"utility": None},
                }),
                # 458 explores at ts=101.8 -- 0.2s from the boundary (102.0),
                # well within the default 1.0s tie_window_s.
                _sel_full(ts=101.8, per_trainer={"458": {"utility": 4.6}}),
                self._scale_event(),
            ])
        sim = _agg(agg_rounds=[_cyc_with_boundary(
            0, 1, ["372", "373", "405"], 0.5,
            {"372": 100.0, "373": 101.0, "405": 102.0})])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]  # diagnostic must NOT flip this
        assert r["set_divergence"] is not None
        diag = r["first_commit_race_diagnostic"]
        assert diag["applicable"] is True
        assert diag["gate1_structural_ok"] is True
        assert "458" in diag["gate1_recent_explore_hits"]
        assert diag["gate2_margin_ok"] is True
        assert diag["explained"] is True

    def test_large_margin_is_NOT_explained(self):
        # Same structural race (458 explores right at the boundary), but its
        # replacement (405) is nowhere near real's own selection cutoff --
        # a genuinely worse candidate, not a close call. Must NOT be excused.
        real = _agg(agg_rounds=[_cyc_with_boundary(
            0, 1, ["372", "373", "458"], 0.5,
            {"372": 100.0, "373": 101.0, "458": 102.0})],
            selection=[
                _sel_full(ts=50.0, per_trainer={
                    "372": {"utility": 5.0}, "373": {"utility": 4.5},
                    "405": {"utility": 0.1},   # far below the cutoff (~4.5)
                    "458": {"utility": None},
                }),
                _sel_full(ts=101.8, per_trainer={"458": {"utility": 4.6}}),
                self._scale_event(),
            ])
        sim = _agg(agg_rounds=[_cyc_with_boundary(
            0, 1, ["372", "373", "405"], 0.5,
            {"372": 100.0, "373": 101.0, "405": 102.0})])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        diag = r["first_commit_race_diagnostic"]
        assert diag["gate1_structural_ok"] is True   # the race precondition still holds...
        assert diag["gate2_margin_ok"] is False       # ...but the margin doesn't -> no free pass
        assert diag["explained"] is False

    def test_stale_exploring_transition_is_NOT_explained(self):
        # 458 explored LONG before the boundary (not a race at all) -- even
        # with a small margin, gate 1 alone must block "explained".
        real = _agg(agg_rounds=[_cyc_with_boundary(
            0, 1, ["372", "373", "458"], 0.5,
            {"372": 100.0, "373": 101.0, "458": 102.0})],
            selection=[
                _sel_full(ts=10.0, per_trainer={
                    "372": {"utility": 5.0}, "373": {"utility": 4.5},
                    "405": {"utility": 4.55}, "458": {"utility": 4.6},
                }),
                self._scale_event(),
            ])
        sim = _agg(agg_rounds=[_cyc_with_boundary(
            0, 1, ["372", "373", "405"], 0.5,
            {"372": 100.0, "373": 101.0, "405": 102.0})])
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        diag = r["first_commit_race_diagnostic"]
        assert diag["gate1_structural_ok"] is False
        assert diag["explained"] is False

    def test_diagnostic_absent_when_cohorts_match(self):
        # No set divergence -> the diagnostic isn't computed at all (None),
        # not a vacuous "applicable": False on a real pass.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.5)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b", "c"], 0.5)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"]
        assert r["first_commit_race_diagnostic"] is None


class TestDrainWallBudget:
    """New commit-stage invariant: sim must NEVER cost more wall-clock than
    real at the drain/commit stage -- generalizes the #15 diagnostic (a
    phantom drain-gate stall, previously only visible via debug counters)
    into a standing rung. `barrier_wait_s`/`drain_spread` are ONE-SIDED
    (sim <= real*(1+tol)) -- sim being FASTER than real is always healthy.
    `drain_tail_s` is the exception: reclassified DIST (matched percentile
    band, see TestDrainTailBand below) because it's SHARED cohort-merge
    compute both modes genuinely perform, not a real-only transport wait
    sim should collapse -- sim being suspiciously FASTER there can fail
    (see test_drain_tail_dist_band_* below)."""

    def _cyc(self, barrier=None, drain=None, proc_ts=None):
        e = {"event": "agg_round", "round": 1}
        if barrier is not None:
            e["barrier_wait_s"] = barrier
        if drain is not None:
            e["drain_tail_s"] = drain
        if proc_ts is not None:
            e["contributor_intervals"] = [
                {"end": f"t{i}", "processing_wall_ts": t}
                for i, t in enumerate(proc_ts)
            ]
        return e

    def test_matched_wall_passes(self):
        real = _agg(agg_rounds=[self._cyc(barrier=2.0, drain=1.0, proc_ts=[0.0, 0.5])])
        sim = _agg(agg_rounds=[self._cyc(barrier=1.9, drain=0.9, proc_ts=[0.0, 0.4])])
        r = pc.drain_wall_budget_parity(real, sim)
        assert r["ok"], r

    def test_sim_transport_excess_FAILS(self):
        # drain_tail_s is a real-transport artifact sim should collapse to ~0;
        # sim taking noticeably MORE than real is a stall, not benign noise.
        real = _agg(agg_rounds=[self._cyc(drain=0.1)])
        sim = _agg(agg_rounds=[self._cyc(drain=5.0)])
        r = pc.drain_wall_budget_parity(real, sim)
        assert not r["ok"]
        assert not r["components"]["drain_tail_s"]["ok"]

    def test_sim_drain_spread_excess_FAILS(self):
        # #15 shape: sim's drain loop takes far longer to get through an
        # already-ready cohort than real's did.
        real = _agg(agg_rounds=[self._cyc(proc_ts=[0.0, 0.3, 0.6])])
        sim = _agg(agg_rounds=[self._cyc(proc_ts=[0.0, 15.0, 30.0])])
        r = pc.drain_wall_budget_parity(real, sim)
        assert not r["ok"]
        assert not r["components"]["drain_spread"]["ok"]

    def test_sim_faster_than_real_passes(self):
        # barrier_wait_s/drain_spread: sim being FASTER than real is always
        # healthy. drain_tail_s stays matched here (0.5 vs 0.45, within
        # band) -- it is NOT exempted by "sim is faster" (see
        # TestDrainTailBand below for what happens when it drops to ~0).
        real = _agg(agg_rounds=[self._cyc(barrier=5.0, drain=0.5, proc_ts=[0.0, 4.0])])
        sim = _agg(agg_rounds=[self._cyc(barrier=0.01, drain=0.45, proc_ts=[0.0, 0.0])])
        r = pc.drain_wall_budget_parity(real, sim)
        assert r["ok"], r

    def test_drain_tail_dist_band_not_exempted_by_sim_being_faster(self):
        # In contrast to barrier_wait_s/drain_spread above: sim dropping
        # drain_tail_s to ~0 while real measures 2.0s is NOT "sim is
        # healthily faster" -- it's sim skipping shared compute it should
        # be charging, and the DIST band correctly fails it.
        real = _agg(agg_rounds=[self._cyc(drain=2.0)])
        sim = _agg(agg_rounds=[self._cyc(drain=0.0)])
        r = pc.drain_wall_budget_parity(real, sim)
        assert not r["ok"]
        assert not r["components"]["drain_tail_s"]["ok"]

    def test_drain_tail_dist_band_absorbs_contention_blip(self):
        # drain_tail_s is SHARED cohort-merge compute (not transport), graded
        # DISTRIBUTIONALLY on a percentile band -- a single sim-host GPU/
        # memory-contention blip (1 of 40 cycles) pushes the MEAN well past a
        # one-sided sim<=real*(1+tol) budget (0.1*1.25=0.125, floored to 0.5;
        # sim mean lands ~0.6) but must still pass because P50/P90/P95 match.
        real = _agg(agg_rounds=[self._cyc(drain=0.1) for _ in range(40)])
        sim = _agg(agg_rounds=[self._cyc(drain=0.1) for _ in range(39)]
                              + [self._cyc(drain=20.0)])
        r = pc.drain_wall_budget_parity(real, sim)
        assert r["ok"], r
        dt = r["components"]["drain_tail_s"]
        assert dt["sim_mean_s"] > 0.5           # a one-sided budget would fail this
        assert dt["pctl_band"]["bands"]["p95"]["rel"] == 0.0

    def test_drain_tail_dist_band_still_catches_systemic_shift(self):
        # Contrast with the blip above: a shift affecting the BULK of the
        # distribution (not one outlier) must still fail.
        real = _agg(agg_rounds=[self._cyc(drain=0.1) for _ in range(40)])
        sim = _agg(agg_rounds=[self._cyc(drain=2.0) for _ in range(40)])
        r = pc.drain_wall_budget_parity(real, sim)
        assert not r["ok"]
        assert not r["components"]["drain_tail_s"]["ok"]

    def test_skips_without_telemetry(self):
        real = _agg(agg_rounds=[{"event": "agg_round", "round": 1}])
        sim = _agg(agg_rounds=[{"event": "agg_round", "round": 1}])
        r = pc.drain_wall_budget_parity(real, sim)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_present_in_run_all_and_meta(self):
        assert "drain_wall_budget" in pc.CHECK_META
        assert pc.CHECK_META["drain_wall_budget"]["deps"]


def _phase_round(**phases):
    e = {"event": "trainer_round"}
    e.update(phases)
    return e


class TestTrainerPhaseWallBudget:
    """Trainer-compute-stage twin of drain_wall_budget: sim must never cost
    more real wall-clock than real on the phases it should collapse
    (dispatch/local-copy overhead), one-sided (sim <= real*(1+tol))."""

    def _tr(self, real_phases, sim_phases):
        real = {"t1": {"trainer_round": [_phase_round(**real_phases)]}}
        sim = {"t1": {"trainer_round": [_phase_round(**sim_phases)]}}
        return real, sim

    def test_matched_wall_passes(self):
        real, sim = self._tr(
            {"pre_train_s": 0.2, "post_train_s": 0.1,
             "weights_to_ram_s": 0.05, "weights_to_gpu_s": 0.05},
            {"pre_train_s": 0.19, "post_train_s": 0.09,
             "weights_to_ram_s": 0.04, "weights_to_gpu_s": 0.04},
        )
        r = pc.trainer_phase_wall_budget_ok(real, sim)
        assert r["ok"], r

    def test_sim_overhead_excess_FAILS(self):
        real, sim = self._tr({"pre_train_s": 0.01}, {"pre_train_s": 3.0})
        r = pc.trainer_phase_wall_budget_ok(real, sim)
        assert not r["ok"]
        assert not r["components"]["pre_train_s"]["ok"]

    def test_sim_faster_than_real_passes(self):
        real, sim = self._tr({"pre_train_s": 2.0, "post_train_s": 1.0},
                              {"pre_train_s": 0.0, "post_train_s": 0.0})
        r = pc.trainer_phase_wall_budget_ok(real, sim)
        assert r["ok"], r

    def test_mqtt_fetch_reported_but_never_gates(self):
        # Apples-to-oranges (real network I/O vs sim in-mem cache): surfaced
        # for diagnosis but a huge mqtt excess alone must never fail `ok`.
        real, sim = self._tr({"pre_train_s": 0.1, "mqtt_fetch_s": 0.1},
                              {"pre_train_s": 0.1, "mqtt_fetch_s": 50.0})
        r = pc.trainer_phase_wall_budget_ok(real, sim)
        assert r["ok"], r
        assert r["components"]["mqtt_fetch_s"]["gates_ok"] is False
        assert r["components"]["mqtt_fetch_s"]["ok"] is False

    def test_skips_without_telemetry(self):
        real = {"t1": {"trainer_round": [{"event": "trainer_round"}]}}
        sim = {"t1": {"trainer_round": [{"event": "trainer_round"}]}}
        r = pc.trainer_phase_wall_budget_ok(real, sim)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_present_in_run_all_and_meta(self):
        assert "trainer_phase_wall_budget" in pc.CHECK_META


class TestPctlBandOk:
    """`pctl_band_ok`: central+P90/P95 band agreement for a WALL/timing
    distribution, deliberately blind to the tail beyond `qs` (P99/max) --
    the escape that lets a genuinely-matched distribution survive a
    sim-host GPU/memory-contention blip (§F-10)."""

    def test_matched_distributions_pass(self):
        r = pc.pctl_band_ok([1.0] * 50, [1.0] * 50)
        assert r["ok"], r
        assert all(b["ok"] for b in r["bands"].values())

    def test_upper_tail_outlier_ignored(self):
        # A handful of extreme values beyond P95 (contention blip) must not
        # move the graded quantiles -- p99 is reported but never gates.
        real = [1.0] * 100
        sim = [1.0] * 96 + [50.0] * 4          # 4% tail, below the P95 cut
        r = pc.pctl_band_ok(real, sim, qs=(50, 90, 95), tol_rel=0.1)
        assert r["ok"], r
        assert r["bands"]["p95"]["rel"] == 0.0
        assert r["p99_diag"]["sim"] == 50.0    # surfaced, not gating

    def test_genuine_central_shift_fails(self):
        # A shift affecting the BULK of the distribution (not a thin tail)
        # must still fail -- the band ignores the tail, not the shape.
        r = pc.pctl_band_ok([1.0] * 100, [2.0] * 100, tol_rel=0.3)
        assert not r["ok"]
        assert not r["bands"]["p50"]["ok"]

    def test_min_abs_floor_absorbs_small_absolute_gap(self):
        # A large RELATIVE gap on a sub-noise absolute magnitude (3ms vs
        # 6ms) is irrelevant -- min_abs floors it to a pass.
        r = pc.pctl_band_ok([0.003] * 20, [0.006] * 20, tol_rel=0.1, min_abs=0.01)
        assert r["ok"], r

    def test_min_abs_does_not_mask_a_large_gap(self):
        r = pc.pctl_band_ok([0.003] * 20, [5.0] * 20, tol_rel=0.1, min_abs=0.01)
        assert not r["ok"]

    def test_empty_distribution_skips(self):
        r = pc.pctl_band_ok([], [1.0])
        assert r["ok"] and r.get("status") == "SKIP"


class TestStepTimingBreakdown:
    """Fine-grained per-function GPU-compute decomposition: DISTRIBUTIONAL
    (KS) match, not a one-sided bound -- genuine shared compute, mode-
    invariant per principle #1."""

    def _st(self, func, real_durs, sim_durs):
        real = {"t1": {"step_timing": [
            {"event": "step_timing", "func": func, "duration_s": d} for d in real_durs]}}
        sim = {"t1": {"step_timing": [
            {"event": "step_timing", "func": func, "duration_s": d} for d in sim_durs]}}
        return real, sim

    def test_matched_distribution_passes(self):
        real, sim = self._st("jvp_eval", [0.01] * 20, [0.01] * 20)
        r = pc.step_timing_breakdown_parity(real, sim)
        assert r["ok"], r
        assert r["by_func"]["jvp_eval"]["ok"]

    def test_diverged_function_FAILS_and_is_named(self):
        real, sim = self._st("jvp_eval", [0.01] * 10, [0.05] * 10)
        r = pc.step_timing_breakdown_parity(real, sim)
        assert not r["ok"]
        assert r["worst_func"] == "jvp_eval"
        assert not r["by_func"]["jvp_eval"]["ok"]

    def test_skips_without_telemetry(self):
        real = {"t1": {"step_timing": []}}
        sim = {"t1": {"step_timing": []}}
        r = pc.step_timing_breakdown_parity(real, sim)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_present_in_run_all_and_meta(self):
        assert "step_timing_breakdown" in pc.CHECK_META

    def test_degenerate_bulk_with_one_outlier_is_skipped_not_scored(self):
        # A near-zero function can have an occasional GC/cache-miss outlier
        # that pushes the MAX a hair over the degenerate threshold while the
        # bulk (p99) is still quantization noise -- must SKIP, not score the
        # dither as a KS "divergence".
        real_durs = [5e-5] * 999 + [9e-4]       # p99 ~5e-5, max 9e-4
        sim_durs = [5e-5] * 999 + [1.2e-3]      # p99 ~5e-5, max 1.2e-3 (over old threshold)
        real, sim = self._st("tb_batch_to_device", real_durs, sim_durs)
        r = pc.step_timing_breakdown_parity(real, sim)
        assert r["ok"], r
        assert r["by_func"]["tb_batch_to_device"].get("status") == "SKIP"

    def test_genuine_divergence_spanning_many_samples_still_fails(self):
        # A divergence affecting a real FRACTION of samples (not a lone
        # outlier) must still be caught even though every value is tiny --
        # the p99 gate tolerates one-in-a-thousand noise, not a systematic
        # shift.
        real_durs = [5e-5] * 500 + [2e-3] * 500   # p99 well above threshold
        sim_durs = [5e-5] * 1000
        real, sim = self._st("tb_batch_to_device", real_durs, sim_durs)
        r = pc.step_timing_breakdown_parity(real, sim)
        assert not r["ok"]
        assert r["by_func"]["tb_batch_to_device"].get("status") != "SKIP"

    def test_real_only_funcs_reported_but_never_gate(self):
        # _emulate_training_delay/pause_execution are commented real-only
        # sleeps (sim skips them); _fetch_weights/recv_wrapper are the same
        # MQTT recv phase_mqtt_fetch already treats as diagnostic-only. A
        # huge, expected real-vs-sim gap here must not fail `ok` alone.
        for func in pc._STEP_TIMING_REAL_ONLY_FUNCS:
            real, sim = self._st(func, [10.0] * 10, [0.0001] * 10)
            r = pc.step_timing_breakdown_parity(real, sim)
            assert r["ok"], (func, r)
            assert r["by_func"][func]["gates_ok"] is False
            assert r["by_func"][func]["ok"] is False

    def test_band_escape_absorbs_contention_tail_ks_and_mean_both_fail(self):
        # A thin (3%) upper-tail contention blip on the sim host inflates
        # both KS and the mean far past their tolerances, but the
        # CENTRAL+P90/P95 band (pctl_band_ok) still matches -- must pass via
        # the band escape, not KS or mean alone.
        real_durs = [1.0] * 200
        sim_durs = [1.0] * 194 + [50.0] * 6   # 3% tail
        real, sim = self._st("aggregate_grads", real_durs, sim_durs)
        r = pc.step_timing_breakdown_parity(real, sim, ks_tol=0.02)
        entry = r["by_func"]["aggregate_grads"]
        assert entry["ok"], entry
        assert entry["ks_stat"] > 0.02          # KS alone would fail
        assert entry["mean_rel_diff"] > 0.05    # mean alone would fail
        assert entry["pctl_band"]["ok"]         # band is what carries it

    def test_real_only_func_does_not_mask_a_genuine_divergence(self):
        real = {"t1": {"step_timing": (
            [{"event": "step_timing", "func": "pause_execution", "duration_s": d} for d in [1.0] * 10]
            + [{"event": "step_timing", "func": "jvp_eval", "duration_s": d} for d in [0.01] * 10])}}
        sim = {"t1": {"step_timing": (
            [{"event": "step_timing", "func": "pause_execution", "duration_s": d} for d in [0.0] * 10]
            + [{"event": "step_timing", "func": "jvp_eval", "duration_s": d} for d in [0.05] * 10])}}
        r = pc.step_timing_breakdown_parity(real, sim)
        assert not r["ok"]
        assert not r["by_func"]["jvp_eval"]["ok"]


class TestAggStepTimingEvalModelExempt:
    """`eval_model` runs on a daemon thread (off the vclock, off the critical
    path); its real<->sim wall gap is pure GPU contention (sim trainers never
    sleep the delay -> sim GPUs denser). It is reported but excluded from gating,
    same mechanism as the real-only-sleep funcs. It must NOT be able to mask a
    genuine divergence in an ON-path aggregator function."""

    def _agg(self, funcs):
        # funcs: {name: [durations]} -> a flat aggregator step_timing list.
        st = [{"event": "step_timing", "func": f, "duration_s": d}
              for f, ds in funcs.items() for d in ds]
        return {"step_timing": st}

    def test_eval_model_gap_reported_but_does_not_gate(self):
        # eval_model 30s sim vs 11s real (gap > the aggregator rung's widened
        # mean_tol_rel), everything else matched -> rung PASSES (eval_model
        # exempt) but still reports diverged.
        real = self._agg({"eval_model": [10.8] * 20, "aggregate": [0.1] * 20})
        sim = self._agg({"eval_model": [30.0] * 20, "aggregate": [0.1] * 20})
        r = pc.agg_step_timing_breakdown_parity(real, sim)
        assert r["ok"], r
        assert r["by_func"]["eval_model"]["gates_ok"] is False
        assert r["by_func"]["eval_model"]["ok"] is False  # still reported as diverged

    def test_eval_model_exemption_does_not_mask_on_path_divergence(self):
        # aggregate (on the critical path) genuinely diverges -> rung still FAILS,
        # even though eval_model is exempt.
        real = self._agg({"eval_model": [10.8] * 20, "aggregate": [0.1] * 20})
        sim = self._agg({"eval_model": [17.2] * 20, "aggregate": [0.4] * 20})
        r = pc.agg_step_timing_breakdown_parity(real, sim)
        assert not r["ok"], r
        assert r["by_func"]["aggregate"]["ok"] is False

    def test_eval_model_in_exemption_set(self):
        assert "eval_model" in pc._AGG_STEP_TIMING_OFF_CRITICAL_PATH_FUNCS

    def test_distribute_weights_sync_real_only_gap_reported_but_does_not_gate(self):
        # _distribute_weights_sync holds the same real-only time.sleep(0.1)
        # pad as its async twin -- a large real<->sim gap here must not fail
        # the rung, same mechanism as _distribute_weights_async.
        real = self._agg({"_distribute_weights_sync": [0.15] * 20, "aggregate": [0.1] * 20})
        sim = self._agg({"_distribute_weights_sync": [0.05] * 20, "aggregate": [0.1] * 20})
        r = pc.agg_step_timing_breakdown_parity(real, sim)
        assert r["ok"], r
        assert r["by_func"]["_distribute_weights_sync"]["gates_ok"] is False
        assert r["by_func"]["_distribute_weights_sync"]["ok"] is False


class TestAggregationComputeWall:
    """Aggregation-stage wall-clock EQUALITY (DIAG, two-sided): unlike
    drain_wall_budget, aggregate_fedavg_s/eval_s are genuine shared compute --
    the target is a MATCH, so sim being either faster OR slower fails it."""

    def _cyc(self, fedavg=None, ev=None):
        e = {"event": "agg_round", "round": 1}
        if fedavg is not None:
            e["aggregate_fedavg_s"] = fedavg
        if ev is not None:
            e["eval_s"] = ev
        return e

    def _agg_with_fedavg(self, vals):
        return _agg(agg_rounds=[self._cyc(fedavg=v) for v in vals])

    def test_matched_wall_passes(self):
        # Slightly-shifted but overlapping distributions -- realistic jitter,
        # not degenerate point masses (a constant-per-cycle value would give
        # KS=1.0 regardless of how close the means are).
        real = self._agg_with_fedavg([1.28, 1.29, 1.30, 1.31, 1.32])
        sim = self._agg_with_fedavg([1.29, 1.30, 1.31, 1.32, 1.33])
        r = pc.aggregation_compute_wall_parity(real, sim)
        assert r["ok"], r

    def test_sim_slower_FAILS(self):
        # Two-sided: sim taking noticeably MORE genuine compute time fails,
        # same as sim taking noticeably LESS would (both are suspicious here).
        real = self._agg_with_fedavg([1.28, 1.29, 1.30, 1.31, 1.32])
        sim = self._agg_with_fedavg([30.0, 31.0, 32.0, 33.0, 34.0])
        r = pc.aggregation_compute_wall_parity(real, sim)
        assert not r["ok"]
        assert not r["components"]["aggregate_fedavg_s"]["ok"]

    def test_sim_faster_also_FAILS(self):
        real = self._agg_with_fedavg([1.28, 1.29, 1.30, 1.31, 1.32])
        sim = self._agg_with_fedavg([0.01, 0.02, 0.03, 0.04, 0.05])
        r = pc.aggregation_compute_wall_parity(real, sim)
        assert not r["ok"]
        assert not r["components"]["aggregate_fedavg_s"]["ok"]

    def test_never_hard_fails_is_diag(self):
        assert pc.CHECK_META["aggregation_compute_wall"]["role"] == "DIAG"

    # --- graded basis: what reached the CLOCK, not sim's contended span ---

    def _charges(self, label, charged, source="profiled", n=5):
        return [{"event": "vclock_charge", "label": label,
                 "charged_s": charged, "span_s": 99.0,
                 "charge_source": source} for _ in range(n)]

    def test_profiled_charge_is_graded_not_sims_own_span(self):
        """The measured shape: sim's raw fedavg wall sits at a flat co-location
        floor well above real's, but the vclock is folded a real-profiled
        constant instead -- so the raw span is a quantity the design discarded
        and grading it fails a run that is actually charging correctly."""
        real = self._agg_with_fedavg([0.060, 0.061, 0.059, 0.062, 0.058])
        sim = self._agg_with_fedavg([0.088, 0.090, 0.089, 0.091, 0.087])
        sim["vclock_charges"] = self._charges("fedavg", 0.0645)
        r = pc.aggregation_compute_wall_parity(real, sim)
        c = r["components"]["aggregate_fedavg_s"]
        assert c["graded_basis"] == "sim_charged"
        assert c["sim_charged_mean_s"] == 0.0645
        assert r["ok"], c

    def test_contention_inflation_is_reported_not_hidden(self):
        """§D-1 must stay visible: the span/charge ratio is the co-location
        inflation, reported even though it no longer gates."""
        real = self._agg_with_fedavg([0.060, 0.061, 0.059, 0.062, 0.058])
        sim = self._agg_with_fedavg([0.088, 0.090, 0.089, 0.091, 0.087])
        sim["vclock_charges"] = self._charges("fedavg", 0.0645)
        c = pc.aggregation_compute_wall_parity(real, sim)["components"]["aggregate_fedavg_s"]
        assert c["sim_wall_inflation_x"] == round(0.089 / 0.0645, 2)
        assert c["sim_mean_s"] == 0.089  # raw span still reported

    def test_a_genuinely_wrong_charge_still_FAILS(self):
        """Switching the basis must not make the rung toothless -- a profile
        that mis-prices real's cost is exactly what this should now catch."""
        real = self._agg_with_fedavg([0.060, 0.061, 0.059, 0.062, 0.058])
        sim = self._agg_with_fedavg([0.088, 0.090, 0.089, 0.091, 0.087])
        sim["vclock_charges"] = self._charges("fedavg", 0.5)  # 8x real
        r = pc.aggregation_compute_wall_parity(real, sim)
        assert not r["ok"]
        assert r["components"]["aggregate_fedavg_s"]["graded_basis"] == "sim_charged"

    def test_live_charge_keeps_grading_the_raw_span(self):
        """`live` means sim folded its OWN span, so the span IS what the clock
        saw -- the yaml is missing sim_charge_profile_path (§D-18) and the raw
        comparison is the right one to fail on."""
        real = self._agg_with_fedavg([0.051, 0.052, 0.050, 0.053, 0.049])
        sim = self._agg_with_fedavg([0.096, 0.097, 0.095, 0.098, 0.094])
        sim["vclock_charges"] = self._charges("fedavg", 0.096, source="live")
        r = pc.aggregation_compute_wall_parity(real, sim)
        c = r["components"]["aggregate_fedavg_s"]
        assert c["graded_basis"] == "sim_wall"
        assert not r["ok"]

    def test_uncharged_keeps_grading_the_raw_span(self):
        real = self._agg_with_fedavg([0.051, 0.052, 0.050, 0.053, 0.049])
        sim = self._agg_with_fedavg([0.096, 0.097, 0.095, 0.098, 0.094])
        sim["vclock_charges"] = self._charges("fedavg", 0.0, source="none")
        c = pc.aggregation_compute_wall_parity(real, sim)["components"]["aggregate_fedavg_s"]
        assert c["graded_basis"] == "sim_wall"

    def test_no_charge_ledger_at_all_is_the_old_behaviour(self):
        """Legacy runs recorded before the ledger existed must still grade."""
        real = self._agg_with_fedavg([1.28, 1.29, 1.30, 1.31, 1.32])
        sim = self._agg_with_fedavg([1.29, 1.30, 1.31, 1.32, 1.33])
        r = pc.aggregation_compute_wall_parity(real, sim)
        assert r["components"]["aggregate_fedavg_s"]["graded_basis"] == "sim_wall"
        assert r["ok"]

    def test_other_charge_labels_do_not_leak_into_fedavg(self):
        real = self._agg_with_fedavg([0.060, 0.061, 0.059, 0.062, 0.058])
        sim = self._agg_with_fedavg([0.088, 0.090, 0.089, 0.091, 0.087])
        sim["vclock_charges"] = self._charges("drain_tail", 0.2783)
        c = pc.aggregation_compute_wall_parity(real, sim)["components"]["aggregate_fedavg_s"]
        assert c["graded_basis"] == "sim_wall"

    def test_skips_without_telemetry(self):
        real = _agg(agg_rounds=[{"event": "agg_round", "round": 1}])
        sim = _agg(agg_rounds=[{"event": "agg_round", "round": 1}])
        r = pc.aggregation_compute_wall_parity(real, sim)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_present_in_run_all_and_meta(self):
        assert "aggregation_compute_wall" in pc.CHECK_META


class TestVarTrajectoryMeanGuard:
    """V2 must fail a systematic mean offset that KS alone misses (a uniform ~1%
    shift barely moves the CDF -> KS~0 but grads have desynced)."""

    def test_systematic_offset_fails_despite_low_ks(self):
        base = [0.9, 0.5, 0.42, 0.31, 0.6, 0.48]
        real = _agg(agg_rounds=[_lcyc(0, i, ["a"], v) for i, v in enumerate(base)])
        sim = _agg(agg_rounds=[_lcyc(0, i, ["a"], v * 1.05) for i, v in enumerate(base)])
        r = pc.var_trajectory_parity(real, sim)
        assert r["mean_rel_diff"] > r["mean_tol_rel"], r
        assert not r["ok"], r

    def test_matched_var_passes(self):
        base = [0.9, 0.5, 0.42, 0.31]
        agg = _agg(agg_rounds=[_lcyc(0, i, ["a"], v) for i, v in enumerate(base)])
        assert pc.var_trajectory_parity(agg, agg)["ok"]


def _selc(round_, chosen, num_candidates, selector="random", ts=0.0):
    """Selection event WITH the num_chosen/num_candidates count fields the
    full-cohort determinism gate reads (real runs always emit these)."""
    return {"event": "selection", "task": "train", "round": round_, "ts": ts,
            "selector": selector, "chosen": list(chosen),
            "num_chosen": len(chosen), "num_candidates": num_candidates}


class TestSelectionDeterminismGate:
    """Set/sequence selection rungs ENFORCE when selection is provably
    deterministic (full-cohort K>=pool, or a declared deterministic selector)
    and gate to a trivial pass otherwise -- data-driven from num_chosen/
    num_candidates so it self-splits fwdllm (enforce) vs fluxtune / fwdllm_plus
    (gate) and self-disables under scarcity."""

    def test_full_cohort_helper(self):
        full = _agg(selection=[_selc(1, ["a", "b", "c"], 3),
                               _selc(2, ["a", "b", "c"], 3)])
        subset = _agg(selection=[_selc(1, ["a", "b", "c"], 10)])  # fluxtune-like
        assert pc._full_cohort_selection(full)
        assert not pc._full_cohort_selection(subset)

    def test_full_cohort_ungates_and_enforces(self):
        # fwdllm syn_0: everyone selected -> deterministic -> ENFORCED. A genuine
        # per-round contributing-set divergence must now FAIL (was invisible).
        real = _agg(selection=[_selc(1, ["a", "b", "c"], 3)],
                    agg_rounds=[_round(1, ["a", "b", "c"], [0, 0, 0])])
        sim = _agg(selection=[_selc(1, ["a", "b", "c"], 3)],
                   agg_rounds=[_round(1, ["a", "b", "x"], [0, 0, 0])])  # x != c
        assert pc._selection_is_deterministic(real, sim)
        agg_seq = pc.aggregation_sequence_parity(real, sim)
        assert agg_seq["gated"] is False and agg_seq["ok"] is False

    def test_full_cohort_matched_is_genuine_pass(self):
        a = _agg(selection=[_selc(1, ["a", "b", "c"], 3)],
                 agg_rounds=[_round(1, ["a", "b", "c"], [0, 0, 0])])
        agg_seq = pc.aggregation_sequence_parity(a, a)
        assert agg_seq["gated"] is False and agg_seq["ok"] is True

    def test_subset_selection_stays_gated(self):
        # fluxtune agg_goal=3 < pool=10: stochastic subset -> gated trivial pass
        # even when the sim picks a DIFFERENT subset (cohort_sequence owns that).
        real = _agg(selection=[_selc(1, ["a", "b", "c"], 10)],
                    agg_rounds=[_round(1, ["a", "b", "c"], [0, 0, 0])])
        sim = _agg(selection=[_selc(1, ["d", "e", "f"], 10)],
                   agg_rounds=[_round(1, ["d", "e", "f"], [0, 0, 0])])
        assert not pc._selection_is_deterministic(real, sim)
        agg_seq = pc.aggregation_sequence_parity(real, sim)
        assert agg_seq["gated"] is True and agg_seq["ok"] is True

    def test_asymmetric_eligibility_stays_gated(self):
        # fwdllm_plus #7: real sees fewer eligible than the pool -> not full
        # cohort in real -> gated (don't hard-fail a real-side eligibility gap).
        real = _agg(selection=[_selc(1, ["a", "b"], 5)])       # 2 of 5 chosen
        sim = _agg(selection=[_selc(1, ["a", "b", "c", "d", "e"], 5)])  # full
        assert not pc._selection_is_deterministic(real, sim)

    def test_legacy_no_counts_falls_back_to_selector_rule(self):
        # No count telemetry + unknown selector -> old behavior: ENFORCED (so the
        # pre-existing disjoint-fails test semantics are preserved, no regression).
        real = _agg(selection=[_sel(1, ["a", "b"])])
        sim = _agg(selection=[_sel(1, ["c", "d"])])
        assert pc._selection_is_deterministic(real, sim)  # enforce on unknown
        assert not pc.selection_parity(real, sim)["ok"]


def _tr_round(overran, data_id=0, it=0, gpu=1.0, budget=2.0):
    return {"real_gpu_time_s": gpu, "training_budget_s": budget,
            "training_overran": overran, "data_id": data_id,
            "iteration_per_data_id": it}


class TestTimingOverrun:
    """Ovr: order-determinism tell. A high overrun fraction means gpu > modeled
    D -> arrival order can flip -> cohort_sequence/v2 breaks are a TIMING-MODEL
    limit, not a sim ordering bug. DIAG (never fails)."""

    def test_no_overrun_verdict(self):
        tr = {"a": {"trainer_round": [_tr_round(False), _tr_round(False)]}}
        r = pc.timing_overrun(tr, tr)
        assert r["ok"] and r["tier"] == "DIAG"
        assert r["real_overrun_frac"] == 0.0
        assert "attainable" in r["verdict"]

    def test_overrun_flags_and_reports_first_bin(self):
        # fluxtune-like: some rounds overrun; earliest at (data_id, iter).
        real = {"a": {"trainer_round": [
            _tr_round(False, 0, 0), _tr_round(True, 0, 1), _tr_round(True, 2, 0)]}}
        sim = {"a": {"trainer_round": [_tr_round(False, 0, 0)]}}
        r = pc.timing_overrun(real, sim)
        assert r["ok"]  # DIAG never fails the scoreboard
        assert r["real_overrun_frac"] == pytest.approx(2 / 3, abs=1e-3)
        assert r["real_first_overrun"] == [0, 1]
        assert "OVERRUN" in r["verdict"]

    def test_skips_without_telemetry(self):
        tr = {"a": {"trainer_round": [{"real_gpu_time_s": 1.0}]}}  # no overran field
        r = pc.timing_overrun(tr, tr)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_present_in_run_all_and_meta(self):
        assert "timing_overrun" in pc.CHECK_META
        tr = {"a": {"trainer_round": [_tr_round(False)]}}
        res = pc.run_all_parity(_agg(), _agg(), tr, tr)
        assert "timing_overrun" in res


def _agg_wall_round(agg_s, eval_s, wall_elapsed_s):
    return {"event": "agg_round", "aggregate_fedavg_s": agg_s,
            "eval_s": eval_s, "wall_elapsed_s": wall_elapsed_s}


class TestVclockFoldDiagnostic:
    """`aggregation_compute_wall_parity` reports the CUMULATIVE
    `aggregate_fedavg_s` as a fraction of total wall, both modes -- the direct
    measurement of how much real compute this rung's per-cycle check covers,
    at a glance instead of only per-cycle means. DIAG tier (never gates the
    scoreboard verdict under --strict), but `ok` is still a real per-cycle
    equality read -- these tests don't assert on it, only on the new dict."""

    def test_reports_cumulative_totals_and_fractions(self):
        # sim: 3 cycles of 2.0s aggregate() each, wall ends at 100s -> 6/100.
        real = _agg(agg_rounds=[
            _agg_wall_round(1.5, 8.0, 30.0),
            _agg_wall_round(1.5, 8.0, 60.0),
            _agg_wall_round(1.5, 8.0, 90.0),
        ])
        sim = _agg(agg_rounds=[
            _agg_wall_round(2.0, 8.0, 33.0),
            _agg_wall_round(2.0, 8.0, 66.0),
            _agg_wall_round(2.0, 8.0, 100.0),
        ])
        r = pc.aggregation_compute_wall_parity(real, sim)
        diag = r["vclock_fold_diagnostic"]
        assert diag["sim_total_aggregate_fedavg_s"] == pytest.approx(6.0)
        assert diag["real_total_aggregate_fedavg_s"] == pytest.approx(4.5)
        assert diag["sim_total_wall_s"] == pytest.approx(100.0)
        assert diag["real_total_wall_s"] == pytest.approx(90.0)
        assert diag["sim_uncredited_fraction"] == pytest.approx(0.06)
        assert diag["real_uncredited_fraction"] == pytest.approx(0.05)

    def test_absent_when_wall_telemetry_missing(self):
        # aggregate_fedavg_s present but no wall_elapsed_s -> fraction is None,
        # not a crash or a fabricated 0.
        real = _agg(agg_rounds=[
            {"event": "agg_round", "aggregate_fedavg_s": 1.0, "eval_s": 1.0}])
        sim = _agg(agg_rounds=[
            {"event": "agg_round", "aggregate_fedavg_s": 1.0, "eval_s": 1.0}])
        r = pc.aggregation_compute_wall_parity(real, sim)
        diag = r["vclock_fold_diagnostic"]
        assert diag["sim_uncredited_fraction"] is None
        assert diag["real_uncredited_fraction"] is None
        assert diag["sim_total_aggregate_fedavg_s"] == pytest.approx(1.0)

    def test_skip_status_has_no_fold_diagnostic_key(self):
        r = pc.aggregation_compute_wall_parity(_agg(), _agg())
        assert r.get("status") == "SKIP"
        assert "vclock_fold_diagnostic" not in r


class TestStepTimingWorstFuncNamesAGatingFunc:
    """`worst_func` ranked over EVERY func, including the ones exempted from
    gating. `_emulate_training_delay` is a modeled sleep the sim skips, so its
    KS is 1.0 by construction and it won every ranking -- naming it as the worst
    func of a failure actually caused by `_make_model_functional` reads as the
    divergence having moved when nothing moved."""

    def _tr(self, func_vals):
        """{trainer: [step_timing events]} for {func: [durations]}."""
        evs = [{"event": "step_timing", "func": f, "duration_s": v, "ts": i}
               for f, vals in func_vals.items() for i, v in enumerate(vals)]
        return {"t0": {"step_timing": evs}}

    def _run(self):
        # Exempted sleep: real-only, KS 1.0. Gating func: a genuine small gap.
        real = self._tr({"_emulate_training_delay": [10.0, 11.0, 12.0, 13.0, 14.0],
                         "_make_model_functional": [0.034, 0.035, 0.036, 0.037, 0.038]})
        sim = self._tr({"_emulate_training_delay": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "_make_model_functional": [0.030, 0.031, 0.031, 0.032, 0.033]})
        return pc.step_timing_breakdown_parity(real, sim)

    def test_worst_func_is_not_the_exempted_sleep(self):
        r = self._run()
        assert r["worst_func"] != "_emulate_training_delay"

    def test_worst_func_is_the_gating_one(self):
        assert self._run()["worst_func"] == "_make_model_functional"

    def test_raw_ranking_still_reported(self):
        """The all-func winner is kept, just no longer mistakable for a cause."""
        assert self._run()["worst_func_incl_exempt"] == "_emulate_training_delay"

    def test_failing_gating_funcs_listed(self):
        r = self._run()
        assert "_emulate_training_delay" not in r["failing_gating_funcs"]

    def test_exempted_sleep_still_does_not_gate(self):
        """The exemption itself must be untouched -- sim skipping a modeled
        sleep is correct behaviour, never a divergence."""
        real = self._tr({"_emulate_training_delay": [10.0, 11.0, 12.0, 13.0, 14.0]})
        sim = self._tr({"_emulate_training_delay": [0.0, 0.0, 0.0, 0.0, 0.0]})
        r = pc.step_timing_breakdown_parity(real, sim)
        assert r["ok"], r
        assert r["by_func"]["_emulate_training_delay"]["gates_ok"] is False


class TestIterDriftRate:
    """V1c grades the per-bin RATE of cadence divergence, not its level.

    `iterations_per_data_id` is an algorithm output that legitimately grows near
    convergence -- parity claims only that both modes need the SAME number. But
    the variance gate is a feedback loop, so a divergence compounds and the
    pooled level becomes a function of run length: `felix_round` read +1.1% at
    3600s and +19.4% at 7200s on unchanged code. The slope does not move with
    the stopping point, so it is what can carry a fixed verdict.
    """

    @staticmethod
    def _run(iters_fn, n_units=200, vclock=False):
        """One agg_round per cycle; `iters_fn(unit)` cycles spent on each bin."""
        rounds, ts = [], 0.0
        for u in range(n_units):
            for it in range(int(round(iters_fn(u)))):
                ts += 1.0
                rounds.append(_fwd_cycle(u, ts=ts,
                                         vclock=(ts if vclock else None),
                                         committed=(it == int(round(iters_fn(u))) - 1),
                                         iteration=it))
        return _agg(agg_rounds=rounds)

    def test_identical_cadence_is_flat(self):
        r = pc.iter_drift_rate_parity(self._run(lambda u: 10),
                                      self._run(lambda u: 10, vclock=True))
        assert r["verdict"] == "flat" and r["ok"], r
        assert r["lambda_per_100_units"] == pytest.approx(0.0, abs=1e-9)

    def test_level_offset_with_no_trend_passes(self):
        """`fedbuff_round`'s shape: sim runs a steady ~5% under real with no
        trend. The LEVEL trips 5%-tolerance rungs, but there is no compounding
        mechanism to find -- the residual is inside the replicate floor (§D-24),
        so this rung must not call it a divergence."""
        r = pc.iter_drift_rate_parity(
            self._run(lambda u: 15 + (u % 3)),
            self._run(lambda u: 14 + (u % 3), vclock=True))
        assert r["verdict"] == "flat" and r["ok"], r
        assert r["sim_mean_iters"] < r["real_mean_iters"]

    def test_compounding_divergence_fails(self):
        """`felix_round`'s shape: the ratio climbs monotonically across the run."""
        r = pc.iter_drift_rate_parity(
            self._run(lambda u: 10),
            self._run(lambda u: 10 * (1.0 + u * 0.002), vclock=True))
        assert r["verdict"] == "diverging" and not r["ok"], r
        assert r["lambda_per_100_units"] > 0
        assert r["last_bin_ratio"] > r["first_bin_ratio"]
        assert abs(r["t_stat"]) > r["t_crit"]

    def test_slope_is_duration_invariant(self):
        """The point of the rung: one underlying divergence must report the same
        lambda whether the run stopped early or late. Compounding is exponential
        in progress, so the generator is too -- and the recovered slope should
        match the planted one at both lengths. A LEVEL rung cannot do this: its
        mean gap keeps growing with the run."""
        planted = 0.002                                  # per unit
        def _sim(u):
            return 10 * math.exp(planted * u)
        short = pc.iter_drift_rate_parity(
            self._run(lambda u: 10, n_units=100),
            self._run(_sim, n_units=100, vclock=True))
        long = pc.iter_drift_rate_parity(
            self._run(lambda u: 10, n_units=200),
            self._run(_sim, n_units=200, vclock=True))
        for r in (short, long):
            assert r["lambda_per_100_units"] == pytest.approx(planted * 100, rel=0.1), r
        # ... while the LEVEL gap a fixed tolerance would grade roughly doubles,
        # which is why no constant tolerance on it can be right at two lengths.
        def _excess(r):
            return r["sim_mean_iters"] / r["real_mean_iters"] - 1.0
        assert _excess(long) > _excess(short) * 1.8

    def test_significant_but_below_floor_does_not_gate(self):
        """A slope can clear significance on a long run and still be physically
        negligible. Both the t-test AND the replicate floor must trip."""
        r = pc.iter_drift_rate_parity(
            self._run(lambda u: 10),
            self._run(lambda u: 10 * (1.0 + u * 0.000002), vclock=True),
            lambda_floor_per_100=0.05)
        assert r["ok"], r
        assert r["verdict"] in ("flat", "significant_below_floor")

    def test_is_the_root_the_level_rungs_depend_on(self):
        meta = pc.CHECK_META
        assert "v1c_iter_drift_rate" in meta["v1_iter_per_data_id"]["deps"]
        assert "v1c_iter_drift_rate" in meta["v2_var_trajectory"]["deps"]
        assert meta["v1c_iter_drift_rate"]["role"] == "MECHANISM"


class TestSimClockBasis:
    """§D-31: grade what the sim CLOCK consumed, never sim's own contended wall."""

    @staticmethod
    def _charge(label, source, charged, span, kind=None, n=4):
        e = {"event": "vclock_charge", "label": label, "charge_source": source,
             "charged_s": charged, "span_s": span}
        if kind:
            e["payload_kind"] = kind
        return [dict(e) for _ in range(n)]

    def test_profiled_label_reports_the_charge_not_the_span(self):
        sim = _agg()
        sim["vclock_charges"] = self._charge("drain_tail", "profiled", 0.2783, 0.3462)
        b = pc.sim_clock_basis(sim)["by_label"]["drain_tail"]
        assert b["source"] == "profiled"
        assert b["charged_mean_s"] == pytest.approx(0.2783)
        assert b["span_mean_s"] == pytest.approx(0.3462)

    def test_payload_kinds_are_not_pooled(self):
        """`redispatch_turnaround` carries a charged `weights` and an uncharged
        `var_bad`. Pooling by bare label reported the whole thing as uncharged,
        because `var_bad` outnumbers `weights` ~4:1."""
        sim = _agg()
        sim["vclock_charges"] = (
            self._charge("redispatch_turnaround", "profiled", 0.06, 0.31, kind="weights", n=4)
            + self._charge("redispatch_turnaround", "none", 0.0, 0.02, kind="var_bad", n=16))
        by = pc.sim_clock_basis(sim)["by_label"]
        assert by["redispatch_turnaround.weights"]["source"] == "profiled"
        assert by["redispatch_turnaround.var_bad"]["source"] == "none"

    def test_agg_wall_gates_only_when_a_label_charges_live(self):
        profiled, live = _agg(), _agg()
        profiled["vclock_charges"] = self._charge("fedavg", "profiled", 0.0645, 0.088)
        live["vclock_charges"] = self._charge("fedavg", "live", 0.088, 0.088)
        assert pc.sim_clock_basis(profiled)["agg_wall_gates"] is False
        assert pc.sim_clock_basis(live)["agg_wall_gates"] is True

    def test_compute_binds_frac_zero_when_delay_dominates(self):
        """D ~7s against ~0.3s of JVP: sim's compute never binds sct's max(), so
        trainer step spans never reach the clock."""
        sim = _agg(agg_rounds=[{"event": "agg_round", "trainer_speed_s": [7.0, 7.0]}])
        trainers = {"t1": {"step_timing": [
            {"func": "train_with_data_id", "duration_s": 0.3} for _ in range(10)]}}
        assert pc.sim_clock_basis(sim, trainers)["compute_binds_frac"] == 0.0

    def test_drain_tail_graded_on_the_charge_when_profiled(self):
        """The mispricing detector: a charge far from THIS baseline's own real
        span is a stale/shared profile, and it lands straight on the vclock."""
        def _dt(vals):
            return _agg(agg_rounds=[{"event": "agg_round", "drain_tail_s": v} for v in vals])
        real = _dt([0.10, 0.10, 0.11, 0.10, 0.10])
        sim = _dt([0.38, 0.39, 0.38, 0.39, 0.38])
        sim["vclock_charges"] = self._charge("drain_tail", "profiled", 0.2783, 0.387)
        c = pc.drain_wall_budget_parity(real, sim)["components"]["drain_tail_s"]
        assert c["graded_basis"] == "sim_charged"
        assert not c["ok"], c            # 0.278 charged against a 0.101 real cost
        assert c["sim_wall_inflation_x"] is not None

    def test_step_timing_rungs_report_when_the_clock_discards(self):
        sim = _agg(agg_rounds=[{"event": "agg_round", "trainer_speed_s": [7.0]}])
        sim["vclock_charges"] = self._charge("fedavg", "profiled", 0.0645, 0.088)
        sim["step_timing"] = [{"func": "_compute_var", "duration_s": 0.023} for _ in range(50)]
        real = _agg()
        real["step_timing"] = [{"func": "_compute_var", "duration_s": 0.010} for _ in range(50)]
        r = pc.agg_step_timing_breakdown_parity(real, sim)
        assert r["ok"] and r["tier"] == "DIAG", r
        assert "discarded" in r["clock_basis"]


class TestChargeCoverage:
    """The standing audit: a mispriced or uncharged span announces itself on the
    run that introduces it, instead of surfacing later as a rung that flipped."""

    @staticmethod
    def _side(label, source, charged, span, n=6):
        a = _agg()
        a["vclock_charges"] = [{"event": "vclock_charge", "label": label,
                                "charge_source": source, "charged_s": charged,
                                "span_s": span} for _ in range(n)]
        return a

    def test_flags_a_charge_far_from_this_baselines_own_real(self):
        real = self._side("drain_tail", "none", 0.0, 0.1011)
        sim = self._side("drain_tail", "profiled", 0.2783, 0.3869)
        r = pc.charge_coverage(real, sim)
        assert r["mispriced"] is True
        assert r["worst_mispriced_label"] == "drain_tail"
        assert r["worst_charge_vs_real_x"] == pytest.approx(2.75, abs=0.02)

    def test_matched_charge_is_not_flagged(self):
        real = self._side("drain_tail", "none", 0.0, 0.2137)
        sim = self._side("drain_tail", "profiled", 0.2137, 0.3462)
        assert pc.charge_coverage(real, sim)["mispriced"] is False

    def test_uncharged_labels_are_named(self):
        real = self._side("fedavg", "none", 0.0, 0.05)
        sim = self._side("fedavg", "none", 0.0, 0.08)
        assert pc.charge_coverage(real, sim)["uncharged_labels"] == ["fedavg"]

    def test_never_gates(self):
        real = self._side("drain_tail", "none", 0.0, 0.1)
        sim = self._side("drain_tail", "profiled", 9.9, 0.4)
        assert pc.charge_coverage(real, sim)["ok"] is True
        assert pc.CHECK_META["charge_coverage"]["role"] == "DIAG"


class TestVarTrajectoryMatchedBudgetOnAsync:
    """§D-4: `v2` must compare the work BOTH sides did. It used to apply the
    matched-budget truncation only when `_real_intrinsic_clock` returned a
    coordinate, which it never does for an async baseline -- so async graded the
    pooled run and compared real's 94 bins against sim's 99."""

    @staticmethod
    def _run(n, var_fn, vclock=False):
        return _agg(agg_rounds=[
            _fwd_cycle(d, ts=float(d), vclock=(float(d) if vclock else None),
                       var=var_fn(d)) for d in range(n)])

    def test_sims_overrun_tail_does_not_decide_the_verdict(self):
        # Matched over the shared 40 bins; sim's extra 10 are late/high-var.
        real = self._run(40, lambda d: 1.0 + d * 0.01)
        sim = self._run(50, lambda d: 1.0 + d * 0.01, vclock=True)
        r = pc.var_trajectory_parity(real, sim)
        assert r["matched_window_mean_rel_diff"] < r["mean_rel_diff"]
        assert r["ok"], r

    def test_a_genuine_matched_window_gap_still_fails(self):
        real = self._run(40, lambda d: 1.0)
        sim = self._run(40, lambda d: 1.6, vclock=True)
        assert not pc.var_trajectory_parity(real, sim)["ok"]


def _seld(round_, chosen, ts=0.0):
    """A selection event carrying the scalars selection_detail grades."""
    e = _sel(round_, chosen, ts=ts)
    e.update({"num_chosen": len(chosen), "in_flight": 30, "effective_c": 30})
    return e


class TestSelectionDetailRepicks:
    """`mean_chosen` averages a bimodal burst (one draw of c, then single-trainer
    top-ups), so it reports how many top-ups fired rather than who they went to.
    felix_round's sim read 2.41 vs real 2.73 while the actual defect was 35 picks
    filling 30 slots. The re-pick count reports that quantity directly (§D-32)."""

    @staticmethod
    def _pair(sim_boundary):
        real = _agg(selection=[_seld(0, [f"t{i}" for i in range(30)])]
                    + [_seld(2, [f"t{i}"], ts=1.0) for i in range(5)])
        sim = _agg(selection=[_seld(0, [f"t{i}" for i in range(30)])]
                   + [_seld(2, [c], ts=1.0) for c in sim_boundary])
        return real, sim

    def test_clean_boundary_reports_zero_repicks(self):
        real, sim = self._pair([f"t{i}" for i in range(5)])
        r = pc.selection_detail_parity(real, sim)
        assert r["real_repicks_in_round"] == 0
        assert r["sim_repicks_in_round"] == 0
        assert "repick_note" not in r

    def test_duplicate_pick_in_one_boundary_is_reported(self):
        # sim re-picks t0 twice inside the SAME round-2 re-draw.
        real, sim = self._pair(["t0", "t1", "t2", "t3", "t0"])
        r = pc.selection_detail_parity(real, sim)
        assert r["real_repicks_in_round"] == 0
        assert r["sim_repicks_in_round"] == 1
        assert "DIVERGE" in r["repick_note"]

    def test_event_driven_reselection_is_not_counted(self):
        # `round` never advances, so a burst is indistinguishable from the run:
        # every legitimate re-pick would otherwise read as a duplicate.
        run = _agg(selection=[_seld(0, ["a", "b"], ts=float(i)) for i in range(20)])
        r = pc.selection_detail_parity(run, run)
        assert r["sim_repicks_in_round"] is None
        assert "N/A" in r["repick_note"]

    def test_repicks_never_flip_the_verdict(self):
        clean_r, clean_s = self._pair([f"t{i}" for i in range(5)])
        dup_r, dup_s = self._pair(["t0", "t1", "t2", "t3", "t0"])
        assert (pc.selection_detail_parity(clean_r, clean_s)["ok"]
                == pc.selection_detail_parity(dup_r, dup_s)["ok"])


class TestInterArrivalOrderPower:
    """ρ must be read with its bucket count: on the fwdllm family `round` is
    coarse, so a 7200s run yields TWO buckets and a 0.66 is unresolved, not a
    well-powered fail."""

    def test_coarse_round_is_flagged_underpowered(self):
        rounds = [_round(1, ["a", "b", "c"], [0, 0, 0]),
                  _round(2, ["c", "b", "a"], [0, 0, 0])]
        r = pc.inter_arrival_order_parity(_agg(agg_rounds=rounds),
                                          _agg(agg_rounds=rounds))
        assert r["n_rounds"] == 2 and r["underpowered"] is True
        assert r["bucket_sizes"]["real"] == [3, 3]

    def test_many_rounds_is_not_flagged(self):
        rounds = [_round(i, ["a", "b"], [0, 0]) for i in range(1, 8)]
        r = pc.inter_arrival_order_parity(_agg(agg_rounds=rounds),
                                          _agg(agg_rounds=rounds))
        assert r["n_rounds"] == 7 and r["underpowered"] is False
