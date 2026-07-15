# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical real/sim parity checks (parity_checks.py).

Runs in the default suite on synthetic telemetry — no MQTT/GPU — so the parity
logic itself is verified independently of any live run. The opt-in end-to-end
check (test_real_sim_e2e_parity.py) reuses the same functions on real runs.
"""

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
    """§Next-roots (simulate_fwdllm.md): fluxtune's async cycles OVERLAP in
    real wall-time (multiple cohorts commit concurrently, unlike sync's one
    round in flight), so cumulative-summing each cycle's own intrinsic_span_s
    as if sequential races far ahead of raw wall -- confirmed ~3.8-4x on live
    fluxtune logs (20764.7s cumulative vs 5372s raw at a 5400s run's end).
    `is_async` must force the raw-wall fallback (same as async_cifar10, which
    never emits intrinsic_span_s), not the sync cumulative-sum anchor."""

    def _pair(self, is_async):
        # 100 commits, each a genuine 80s barrier+eval span, but real cycles
        # OVERLAP ~4x in wall time (dispatched to ~4 concurrent trainers) so
        # raw wall only advances 20s/commit -- matching sim's vclock 1:1.
        # (100, not 10: keeps a matched-V boundary rounding edge to <=1%,
        # well under the 5% tol, instead of dominating a 10-commit sample.)
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

    def test_terminal_state_data_ids_at_V_nonzero(self):
        real = _agg(agg_rounds=[_fwd_round(d, ["a", "b"], ts=float((d + 1) * 10))
                                for d in range(10)])
        sim = _agg(agg_rounds=[_fwd_round(d, ["a", "b"], vclock=float(d * 10),
                                          ts=float(d + 1))
                               for d in range(10)])
        r = pc.terminal_state_parity(real, sim)
        # Previously real_rounds_at_V==0 (round static); now counts data_ids.
        assert r["real_rounds_at_V"] > 0 and r["sim_rounds_at_V"] > 0, r
        assert r["ok"], r

    def test_total_commits_counts_distinct_data_ids(self):
        # A variance-FAIL retry emits 2 cycles on the SAME data_id; the commit
        # count must be distinct data_ids (2), not raw cycles (3).
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
        assert r["n_sim_commits"] == 2 and r["n_real_commits"] == 2, r
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
        """Regression, 2026-07-14 7200s fwdllm/fwdllm_plus runs: on a long enough
        run the fast side (sim) can complete a full total_data_bins-length lap
        (`round` ticks 1->2, `cycle_data_id` wraps back to 0) while the slow side
        (real) never leaves round=1. The old axis heuristic picked per-side
        ("round" if >1 distinct value seen on THAT side), so sim got keyed on
        `round` (one giant "advance" bundling ~10 commits) while real stayed on
        `data_id` (9 small ones) -- incommensurate units, both a spurious FAIL
        and a fabricated overhead_residual. Both sides must key on `data_id`
        whenever it's present at all, and the (round, data_id) composite key
        must keep sim's two laps distinct instead of colliding on raw value."""
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
    the advance rungs (TestProgressAxisRekey): 2026-07-14 fwdllm/fwdllm_plus 7200s
    runs showed sim eval events spanning round={1,2} while real stayed at
    round=1. Keying the eval curve on raw `data_id` alone lets a sim lap-2
    (more-trained) checkpoint silently overwrite lap-1's entry at the same
    nominal data_id, so the real<->sim comparison at that key mismatches training
    amount. The (round, data_id) composite key excludes sim's lap-2 evals from
    the real/sim key intersection (real has no `(2, *)` key), comparing only
    genuinely matched progress."""

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
                 agg_goal=3):
    """Series where data_id d takes iters_per_data[d] cycles: (n-1) variance
    FAILs (var>thr) then one PASS (var<=thr). grad_pool grows each retry and is
    at its max on the committing cycle; cached_v grows across the FAIL rollbacks.
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
                agg_goal=agg_goal))
    return events


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
        cycles = [
            _cadence(0, 0, 2.0, 1.0, False),
            _cadence(0, 1, 2.0, 1.0, False),
            _cadence(0, 2, 2.0, 1.0, True, force_commit_planned=True),  # forced
        ]
        assert pc._iters_per_data_id(cycles) == {0: 3}

    def test_non_fwdllm_skips(self):
        a = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])  # no cadence fields
        r = pc.iters_per_data_id_parity(a, a)
        assert r["ok"] and r.get("status") == "SKIP", r


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


def _dispatch(peer, ts, payload_kind="weights"):
    return {"event": "comm", "direction": "agg_to_trainer", "peer_id": peer,
            "ts": ts, "payload_kind": payload_kind}


def _agg_comm(dispatches, resolves=None):
    """`dispatches` = list of (peer, ts) or (peer, ts, payload_kind) ->
    comm_dispatch. `resolves` = list of (peer, ts) -> agg_rounds entries
    whose contributor_intervals name that peer (the variance-gate evaluation
    that actually consumes the trainer's outstanding contribution). Always
    tags is_async=True (R1 is async-only, see inflight_overlap_parity) even
    with no resolves, via a marker round with no contributor."""
    d = _agg()
    d["comm_dispatch"] = [_dispatch(*args) for args in dispatches]
    d["agg_rounds"] = [{"event": "agg_round", "ts": 0.0, "is_async": True}] + [
        {"event": "agg_round", "ts": ts, "is_async": True,
         "contributor_intervals": [{"end": peer}]}
        for (peer, ts) in (resolves or [])
    ]
    return d


class TestR1InflightOverlap:
    """R1 [INV]: no trainer may have two simultaneously-UNRESOLVED dispatches
    (a DISPATCH before the trainer's prior dispatch was RESOLVED by a
    variance-gate evaluation). NOT the trainer's own reply landing: confirmed
    2026-07-15 by tracing one trainer's full dispatch/reply/eval timeline
    end-to-end that real always orders eval-before-next-dispatch (clean by
    channel construction -- real's network latency means the async
    variance-gate evaluation has usually already run by the time a reply
    lands) while sim's near-zero-latency control path can dispatch again
    before that evaluation catches up -- a genuine sim-only race. `var_bad`
    ("keep training, submit a new round of perturbations") counts as a real
    dispatch too, not a passive ping -- confirmed by the same trace."""

    def test_dispatch_after_resolve_passes(self):
        # A and B are each re-dispatched a 2nd time, but only AFTER a
        # variance-gate evaluation resolved their 1st contribution.
        agg = _agg_comm(
            dispatches=[("A", 0.0), ("B", 0.0), ("A", 6.0), ("B", 6.0)],
            resolves=[("A", 5.0), ("B", 5.0)],
        )
        r = pc.inflight_overlap_parity(agg, agg)
        assert r["ok"]
        assert r["real_overlap_frac"] == 0.0 and r["sim_overlap_frac"] == 0.0

    def test_sim_overlap_fails_with_clean_real(self):
        # Real: A's 2nd dispatch (6.0) is after the eval resolving the 1st (5.0) -> clean.
        real_agg = _agg_comm(dispatches=[("A", 0.0), ("A", 6.0)], resolves=[("A", 5.0)])
        # Sim: A re-dispatched at 2.0 BEFORE the eval resolving its 1st
        # dispatch (5.0) ran -> overlap = the genuine residence violation.
        sim_agg = _agg_comm(dispatches=[("A", 0.0), ("A", 2.0)], resolves=[("A", 5.0)])
        r = pc.inflight_overlap_parity(real_agg, sim_agg)
        assert not r["ok"]
        assert r["real_overlap_frac"] == 0.0
        assert r["sim_overlap_frac"] > 0.0

    def test_skips_without_comm_telemetry(self):
        # Sync baselines / runs predating the comm dispatch telemetry.
        agg = _agg()
        r = pc.inflight_overlap_parity(agg, agg)
        assert r.get("status") == "SKIP" and r["ok"]

    def test_var_bad_is_a_real_dispatch(self):
        # var_bad IS new work ("keep training, submit a new round of
        # perturbations against the current model"), not a passive ping --
        # two of them with no intervening resolve is a genuine violation.
        agg = _agg_comm(dispatches=[("A", 0.0, "var_bad"), ("A", 2.0, "var_bad")])
        r = pc.inflight_overlap_parity(agg, agg)
        assert not r["ok"]
        assert r["sim_overlap_frac"] > 0.0


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
        assert "r1_inflight_overlap" in pc.CHECK_META["v1_iter_per_data_id"]["deps"]
        assert pc.CHECK_META["w1_compute_conservation"]["deps"] == ("r1_inflight_overlap",)


def _lcyc(data_id, iteration, cohort, var, var_good=False, force=False, goal=3,
          is_async=False):
    """One fwdllm variance-cadence cycle event (cohort in receive/commit order)."""
    return {"event": "agg_round", "round": 1,
            "cycle_data_id": data_id, "iteration_per_data_id": iteration,
            "contributing_trainers": list(cohort), "var": var,
            "var_good_enough": var_good, "force_commit_planned": force,
            "agg_goal_count": goal, "staleness": [0] * len(cohort),
            "is_async": is_async}


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
        # #N: SYNC receive-ORDER is SOFT (fedavg order-invariant, K-D31
        # canonicalizes ties) -- a same-set/same-var reorder no longer fails,
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
        # #N: cadence/var EXACT is HARD only through bin 1 (float-
        # nondeterminism wall starts ~bin 7) -- a divergence beyond it does
        # NOT fail cohort_sequence; that's v1/v2/v4/v5's (DISTRIBUTIONAL) job.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                                _lcyc(7, 2, ["a", "b"], 0.26, var_good=False)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                               _lcyc(7, 2, ["a", "b"], 0.31, var_good=True)])
        r = pc.cohort_sequence_parity(real, sim)
        assert r["ok"], r
        assert r["cadence_var_order_max_bin"] == 1

    def test_set_divergence_beyond_bin1_still_enforced(self):
        # #N: cohort SET is HARD over the ENTIRE run, uncapped -- a genuine
        # divergence (fluxtune's #1d) must still fail no matter the data_id.
        real = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                                _lcyc(7, 2, ["a", "b"], 0.5)])
        sim = _agg(agg_rounds=[_lcyc(0, 1, ["a", "b"], 0.5),
                               _lcyc(7, 2, ["a", "c"], 0.5)])  # different SET
        r = pc.cohort_sequence_parity(real, sim)
        assert not r["ok"]
        assert r["set_match_frac"] < 1.0
        assert r["set_divergence"]["cycle_index"] == 1

    def test_present_in_run_all_and_meta(self):
        assert "cohort_sequence" in pc.CHECK_META
        assert pc.CHECK_META["cohort_sequence"]["deps"]


class TestDrainWallBudget:
    """New commit-stage invariant: sim must NEVER cost more wall-clock than
    real at the drain/commit stage -- generalizes the #15 diagnostic (a
    phantom drain-gate stall, previously only visible via debug counters)
    into a standing rung. One-sided (sim <= real*(1+tol)), not a two-sided
    distribution match -- sim being FASTER than real is always healthy."""

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
        # The whole point of this rung: sim being FASTER than real is
        # healthy, never a fail.
        real = _agg(agg_rounds=[self._cyc(barrier=5.0, drain=2.0, proc_ts=[0.0, 4.0])])
        sim = _agg(agg_rounds=[self._cyc(barrier=0.01, drain=0.0, proc_ts=[0.0, 0.0])])
        r = pc.drain_wall_budget_parity(real, sim)
        assert r["ok"], r

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
    """K-D41: `aggregation_compute_wall_parity` reports the CUMULATIVE
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
