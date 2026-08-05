# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""DIST tolerances sized against the measured same-seed replicate floor (D-24).

The motivating defect: `v1_iter_per_data_id` shipped a 15% gate calibrated when
the floor was 13.3%. Removing the noise source dropped the floor to 0.6% and the
gate stayed at 15%, so a 5.4% real↔sim gap — 9x the floor — graded green.
"""
import pytest

from parity.checks import floor_gated_tol


class TestFloorGatedTol:
    def test_no_floor_leaves_the_nominal_alone(self):
        """A baseline with no replicate is graded exactly as before, never on
        some other baseline's floor (D-36)."""
        assert floor_gated_tol(0.15, None) == (0.15, None)

    def test_tightens_toward_a_small_floor(self):
        """The motivating case: 15% nominal over a 0.6% floor."""
        tol, why = floor_gated_tol(0.15, 0.0058)
        assert why is None
        assert tol == pytest.approx(0.02)          # min_abs, not 3x0.0058
        assert 0.054 > tol                          # the gap that used to pass

    def test_scales_with_a_larger_floor(self):
        tol, why = floor_gated_tol(0.15, 0.041)
        assert why is None
        assert tol == pytest.approx(0.123)          # 3 x floor, under nominal

    def test_never_loosens_past_the_nominal(self):
        """A noisy baseline must not silently WIDEN its gate."""
        tol, why = floor_gated_tol(0.05, 0.016)
        assert tol <= 0.05 and why is None

    def test_floor_at_or_above_tolerance_is_ungradeable(self):
        """Not a pass and not a fail — a verdict there is a coin flip."""
        tol, why = floor_gated_tol(0.02, 0.066)
        assert tol == 0.02
        assert why and "grades noise" in why

    def test_min_abs_is_a_floor_on_the_tightening(self):
        """A near-zero floor must not manufacture an unmeetable gate."""
        tol, why = floor_gated_tol(0.15, 0.0)
        assert tol == pytest.approx(0.02) and why is None


class TestRunAllParityWiring:
    """The gate must reach the rungs and stamp what it did, or a re-grade is
    unreadable months later."""

    def test_ungradeable_rungs_become_skips(self):
        from parity.checks import floor_gated_tol as f
        # v2's 2% nominal under fluxtune's 6.6% ON floor
        _, why = f(0.02, 0.066)
        assert why is not None, "fluxtune v2 must refuse to grade"

    def test_felix_round_v1_would_fail_after_gating(self):
        tol, _ = floor_gated_tol(0.15, 0.0058)
        assert 0.054 > tol, "the 5.4% gap must no longer pass"


class TestAbsoluteUnitFloors:
    """Three of the gated fields are absolute — a moving-average deviation in
    ITERATIONS, an accuracy gap in POINTS, a slope per 100 progress units. The
    global 2% `min_abs` is a relative quantity and means nothing in those units,
    so each field carries its own."""

    def test_min_abs_is_per_field(self):
        # v1b's worst-case MA deviation: a 0.75-iteration nominal must not be
        # tightened to 0.02 iterations by the relative default.
        assert floor_gated_tol(0.75, 0.0, min_abs=0.75)[0] == pytest.approx(0.75)
        assert floor_gated_tol(0.75, 0.0)[0] == pytest.approx(0.02)

    def test_a_measured_floor_still_tightens_within_its_own_units(self):
        # 3 x a 0.1-iteration floor sits under the 0.75 nominal.
        tol, why = floor_gated_tol(0.75, 0.1, min_abs=0.75)
        assert why is None and tol == pytest.approx(0.75)
        tol, why = floor_gated_tol(2.5, 0.1, min_abs=0.75)
        assert why is None and tol == pytest.approx(0.75)

    def test_the_measured_control_makes_v1c_ungradeable_where_it_fired(self):
        """`v1c`'s docstring calibration (max |t| 1.61 over six pairs) is
        falsified at n=3: |lambda| reaches 0.20/100 real↔real on fedbuff_it_unaware
        against a 0.05 floor, with t=5.14. Above nominal ⇒ refuse to grade."""
        _tol, why = floor_gated_tol(0.05, 0.1979, min_abs=0.05)
        assert why and "grades noise" in why

    def test_a_quiet_baseline_keeps_v1c_gradeable(self):
        # fwdllm's pinned legs: zero drift real↔real, so the rung still gates.
        assert floor_gated_tol(0.05, 0.0, min_abs=0.05) == (0.05, None)


class TestBoundedLoosen:
    """`v2`'s floor (1.8-1.9%) had caught up with a 2.0% nominal that was never
    calibrated, so it graded its own noise and fired on 3 of 9 config-identical
    real pairs. `loosen_cap` lets a rung widen — but only so far, or a degrading
    sim would widen its own gate without bound."""

    def test_a_rung_without_the_cap_never_loosens(self):
        assert floor_gated_tol(0.02, 0.019)[0] == pytest.approx(0.02)

    def test_v2_loosens_to_the_cap_not_to_3x_floor(self):
        # 3 x 1.9% = 5.7%, capped at 2 x nominal = 4%.
        tol, why = floor_gated_tol(0.02, 0.019, loosen_cap=2.0)
        assert why is None and tol == pytest.approx(0.04)

    def test_a_quiet_baseline_is_not_loosened(self):
        """fwdllm's 0.2% floor must leave the 2% gate exactly where it was."""
        tol, why = floor_gated_tol(0.02, 0.002, loosen_cap=2.0)
        assert why is None and tol == pytest.approx(0.02)

    def test_a_floor_above_even_the_cap_still_refuses_to_grade(self):
        """fluxtune's 6.6% mean_var floor clears 4%, so it stays a SKIP."""
        tol, why = floor_gated_tol(0.02, 0.066, loosen_cap=2.0)
        assert tol == pytest.approx(0.04)
        assert why and "grades noise" in why

    def test_the_cap_bounds_the_widening(self):
        for floor in (0.05, 0.5, 5.0):
            tol, _why = floor_gated_tol(0.02, floor, loosen_cap=2.0)
            assert tol <= 0.04


class TestSelectionCountIsWorkVolume:
    """`selection_detail` failed only on `rel_diff_n_selections` (7.5-13.8%) while
    `rel_diff_chosen` and `rel_diff_inflight` read EXACTLY 0.0 on the same pairs —
    chosen/cycle is 10.01 on every leg of every baseline. The count is cadence,
    not selector behaviour, so it takes the cadence floor (§D-64)."""

    def test_the_count_gate_is_separate_from_the_behaviour_gate(self):
        import inspect

        from parity.checks import selection_detail_parity
        p = inspect.signature(selection_detail_parity).parameters
        assert "tol_n_selections" in p
        assert p["tol_n_selections"].default == p["tol_chosen"].default == 0.05

    def test_the_count_shares_v1s_gate_not_its_own(self):
        """Its own 5% nominal is BELOW the cadence floor on the noisy baselines,
        so floor-gating alone would only turn it into a SKIP. It is v1's number,
        so it takes v1's tolerance — felix_it 7.5% and fedbuff_it_oracular 13.8%
        both clear a v1 gate, and neither clears 5%."""
        v1_tol, why = floor_gated_tol(0.15, 0.049)      # fedbuff_it_oracular
        assert why is None and 0.138 < v1_tol
        own_tol, own_why = floor_gated_tol(0.05, 0.056)  # felix_it, own nominal
        assert own_why is not None, "its own gate can only ever SKIP here"

    def test_a_zero_floor_baseline_keeps_the_tight_count_gate(self):
        """fwdllm's pinned legs: the count stays gated at min_abs, so a genuine
        selector regression there still fails."""
        tol, why = floor_gated_tol(0.05, 0.0, min_abs=0.02)
        assert why is None and tol == pytest.approx(0.02)


class TestOneQuantityOneTolerance:
    """`cohort_sequence.count` and `v1b.cum_mean_rel` are `v1`'s number rolled up
    — identical to 3 decimals on all nine baselines over 19 control pairs. Grading
    them at their own hand-typed 5% failed the SAME measurement that `v1` passed
    (§D-22)."""

    @staticmethod
    def _tols(floors):
        import inspect

        from parity.checks import (cohort_sequence_parity,
                                   iters_per_data_id_moving_avg_parity,
                                   iters_per_data_id_parity)
        v1 = inspect.signature(iters_per_data_id_parity).parameters[
            "mean_tol_rel"].default
        eff, _ = floor_gated_tol(v1, floors.get("iters_per_bin"))
        nom_c = inspect.signature(cohort_sequence_parity).parameters[
            "count_tol"].default
        nom_b = inspect.signature(iters_per_data_id_moving_avg_parity).parameters[
            "cum_mean_rel_tol"].default
        return eff, nom_c, nom_b

    def test_the_hand_typed_gates_were_tighter_than_v1s(self):
        eff, nom_c, nom_b = self._tols({"iters_per_bin": 0.049})
        assert nom_c == nom_b == 0.05
        assert eff > nom_c, "v1's gated tolerance must be the looser one"

    def test_fedbuff_it_oracular_no_longer_splits_on_one_number(self):
        """0.116 measured, one gate: pass on all three rungs or fail on all
        three — never pass as v1 and fail as cohort."""
        eff, _nc, _nb = self._tols({"iters_per_bin": 0.049})
        assert 0.116 < eff


class TestTheVclockTimeFamily:
    """`throughput`, `terminal_state` and `total_commits` BAIL real↔real for want
    of `vclock_now`, so §D-56 called them uncontrollable and all three kept
    hand-typed gates. But the RUNG bails; the QUANTITY does not (§D-72) — real's
    own time-to-N spreads to 6.7% between same-code replicate legs under an
    8% gate (13.6% pooled across code versions), and
    `fedbuff_round` fails trainers at 7.3% against a 5% gate whose own floor is
    5.5%. `fedbuff_it_unaware` PASSES on a 13.6% floor: luck, not parity."""

    @staticmethod
    def _default(fn, param):
        import inspect
        return inspect.signature(fn).parameters[param].default

    def test_the_bail_stays_the_default_for_a_real_sim_pair(self):
        """`same_mode` is opt-in. A SIM leg carrying no vclock is a broken run and
        must still fail loudly, never get quietly graded on its wall clock."""
        from parity.checks import (terminal_state_parity, throughput_parity,
                                   total_commits_parity)
        for fn in (throughput_parity, terminal_state_parity, total_commits_parity):
            assert self._default(fn, "same_mode") is False

    def test_two_real_legs_are_gradeable_in_same_mode(self):
        """The floor measurement itself: no vclock on either side, a number out."""
        from parity.checks import terminal_state_parity
        a = {"agg_rounds": [{"event": "agg_round", "round": 1, "ts": t,
                             "contributing_trainers": ["1"]} for t in (0.0, 100.0)]}
        b = {"agg_rounds": [{"event": "agg_round", "round": 1, "ts": t,
                             "contributing_trainers": ["1"]} for t in (0.0, 110.0)]}
        assert terminal_state_parity(a, b)["ok"] is False        # bails, §D-56
        res = terminal_state_parity(a, b, same_mode=True)
        assert res["time_rel_diff"] == pytest.approx(0.091, abs=1e-3)

    def test_the_trainers_gate_would_be_ungradeable_without_the_cap(self):
        """fedbuff_round's 5.5% trainers floor EXCEEDS the 5% nominal, and one
        ungradeable field takes the whole rung down — a SKIP would stop grading
        the time half too. The cap keeps it gradeable at 10%."""
        _tol, why = floor_gated_tol(0.05, 0.055)
        assert why is not None, "without the cap the rung refuses to grade"
        tol, why = floor_gated_tol(0.05, 0.055, loosen_cap=2.0)
        assert why is None and tol == pytest.approx(0.10)
        assert 0.073 < tol, "the 7.3% residual is inside its own floor's 3x"

    def test_a_pinned_baseline_keeps_the_tight_time_gate(self):
        """fwdllm's measured time-to-N floor is 0.0%: the negative control. The
        gate must TIGHTEN there, or floor-gating would only ever excuse fails."""
        tol, why = floor_gated_tol(0.08, 0.0)
        assert why is None and tol == pytest.approx(0.02)

    def test_throughput_grades_a_time_ratio_not_work_volume(self):
        """It was floor-gated on `committed_bins`. On fedbuff_round those read
        1.6% while the quantity it actually decides on reads 4.3% — a gate sized
        from the wrong metric by 2.7x."""
        by_volume, _ = floor_gated_tol(0.08, 0.0155)
        by_time, _ = floor_gated_tol(0.08, 0.043)
        assert by_volume < by_time <= 0.08

    def test_the_rung_names_the_field_its_verdict_came_from(self):
        """`throughput` reports a full-run `rel_diff` and a matched-work one, and
        decides on the matched-work figure (§D-75). A floor read off the other
        grades a window the rung never decided on (§D-71), so it states which it
        used and `replicate_floor` reads that key rather than guessing."""
        from parity.checks import throughput_parity
        a = {"agg_rounds": [{"event": "agg_round", "round": r, "ts": r * 10.0,
                             "contributing_trainers": ["1"]} for r in range(1, 8)]}
        b = {"agg_rounds": [{"event": "agg_round", "round": r, "ts": r * 12.0,
                             "contributing_trainers": ["1"]} for r in range(1, 8)]}
        res = throughput_parity(a, b, same_mode=True)
        assert res["decided_on"] == "matched_window_rel_diff"
        assert res["ok"] == (res[res["decided_on"]] <= res["tol"])


class TestTotalCommitsIsTerminalStatesTimeHalf:
    """U2 recomputes K8's time dimension — same matched N, same clocks, same
    number (`sim_vclock_to_n 7192.8` vs `real_time_to_n 6335.6`, both reporting
    0.119 against 0.08). Gating both counts ONE measurement twice (§D-64, 4th
    instance after cohort_sequence, v1b and selection_detail)."""

    def test_the_two_rungs_report_the_same_number(self):
        from parity.checks import terminal_state_parity, total_commits_parity
        a = {"agg_rounds": [{"event": "agg_round", "round": 1, "ts": t,
                             "contributing_trainers": ["1"]} for t in (0.0, 100.0)]}
        b = {"agg_rounds": [{"event": "agg_round", "round": 1, "ts": t,
                             "contributing_trainers": ["1"]} for t in (0.0, 130.0)]}
        k8 = terminal_state_parity(a, b, same_mode=True)
        u2 = total_commits_parity(a, b, same_mode=True)
        assert u2["rel_diff"] == pytest.approx(k8["time_rel_diff"], abs=1e-3)
        assert u2["sim_vclock_to_n_s"] == k8["sim_vclock_to_n_s"]

    def test_u2_reports_but_does_not_gate(self):
        """One measurement, one verdict — on K8, which also grades trainers."""
        from parity.checks import _WARN_ONLY_CHECKS
        assert "total_commits" in _WARN_ONLY_CHECKS
        assert "terminal_state" not in _WARN_ONLY_CHECKS

    def test_a_u2_fail_is_a_warn_not_a_root_cause(self):
        from parity.checks import _classify
        res = {"ok": False, "tier": "EXACT"}
        assert _classify("total_commits", res, False, False) == "warn"
        assert _classify("terminal_state", res, False, False) == "fail"

    def test_the_floor_reaches_the_rung_and_is_stamped(self):
        """End-to-end through `run_all_parity`: a re-grade months later has to be
        able to see WHICH floor sized the gate, not just the verdict."""
        from parity.test_ladder import AGG_GOAL, _build_mode

        from parity.checks import run_all_parity
        real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
        sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
        res = run_all_parity(real_agg, sim_agg, real_tr, sim_tr, agg_goal=AGG_GOAL,
                             floors={"time_to_n": 0.067, "trainers_at_n": 0.055,
                                     "throughput_rel": 0.043})
        k8 = res["terminal_state"]
        assert k8["replicate_floor_rel"] == {"time_to_n": 0.067,
                                             "trainers_at_n": 0.055}
        # time: 3 x 6.7% clears the 8% nominal, so it holds at nominal; trainers
        # loosen to the 2x cap because their floor has overtaken the 5% nominal.
        assert k8["floor_gated_tol"] == {"time_to_n": 0.08, "trainers_at_n": 0.1}
        assert k8["time_tol"] == 0.08 and k8["trainers_tol"] == pytest.approx(0.1)
        assert res["throughput"]["replicate_floor_rel"] == {"throughput_rel": 0.043}
        assert res["total_commits"]["tol"] == k8["time_tol"], "U2 shares K8's gate"


class TestTheSharedCadenceCountVotesOnce:
    """`v1_iter_per_data_id`, `cohort_sequence.count` and
    `selection_detail.rel_diff_n_selections` are ONE number — identical to 3
    decimals on all nine baselines over 19 control pairs — and since §D-64 they
    also share ONE floor-sized tolerance. Letting all three vote made one
    measurement produce three fails: 6 of the board's 12 were two numbers."""

    def test_selection_detail_reports_the_count_without_voting(self):
        """Its selector bounds still gate; only the cadence count steps down."""
        from parity.checks import selection_detail_parity
        base = {"round": 1, "task": "train", "event": "selection",
                "num_chosen": 10, "in_flight": 10, "effective_c": 10,
                "chosen": ["1"] * 10}
        real = {"selection_train": [dict(base, ts=float(i)) for i in range(10)],
                "agg_rounds": []}
        sim = {"selection_train": [dict(base, ts=float(i)) for i in range(12)],
               "agg_rounds": []}
        res = selection_detail_parity(real, sim, tol_n_selections=0.05)
        assert res["rel_diff_n_selections"] > 0.05
        assert res["n_selections_ok"] is False, "the sub-verdict is still reported"
        assert res["n_selections_owned_by"] == "v1_iter_per_data_id"
        assert res["ok"] is True, "a count gap alone must not fail this rung"

    def test_a_genuine_selector_gap_still_fails_it(self):
        """The collapse must not disarm what this rung uniquely owns."""
        from parity.checks import selection_detail_parity
        def evs(n_chosen, n):
            return [{"round": 1, "task": "train", "event": "selection",
                     "ts": float(i), "num_chosen": n_chosen, "in_flight": n_chosen,
                     "effective_c": n_chosen, "chosen": ["1"] * n_chosen}
                    for i in range(n)]
        res = selection_detail_parity({"selection_train": evs(10, 10), "agg_rounds": []},
                                      {"selection_train": evs(5, 10), "agg_rounds": []})
        assert res["rel_diff_chosen"] > 0.15 and res["ok"] is False

    def test_cohort_sequences_verdict_is_invariant_to_the_count_gate(self):
        """The property, stated directly: moving `count_tol` across the measured
        gap flips the count's own sub-verdict and must NOT move the rung's. On an
        async stochastic baseline `count` was its only enforced bound, so the
        rung's whole verdict WAS v1's number."""
        import sys
        from pathlib import Path
        _fw = Path(__file__).resolve().parents[3] / "fwdllm" / "expt_scripts"
        sys.path.insert(0, str(_fw))
        import replicate_floor as rf
        from test_replicate_floor import _run

        from parity.checks import cohort_sequence_parity
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        a = rf.checker_agg(_run(6, 4, 0.9, tmp,
                                "run_20260101_000000_b_n10_smoke_syn_0_real"))
        b = rf.checker_agg(_run(6, 6, 0.9, tmp,
                                "run_20260102_000000_b_n10_smoke_syn_0_sim"))
        tight = cohort_sequence_parity(a, b, count_tol=0.01)
        loose = cohort_sequence_parity(a, b, count_tol=0.99)
        assert tight["count"]["ok"] is False and loose["count"]["ok"] is True
        assert tight["ok"] == loose["ok"], "the count must not move the verdict"
        assert tight["count_owned_by"] == "v1_iter_per_data_id"


class TestThroughputGradesMatchedWork:
    """§D-75/§D-76: per-unit time rises 2.1-2.7x across quintiles, so each side's
    OWN full run is a different window. Real finished 191 units to sim's 183 on
    `fedbuff_it_oracular`; its 8 extra units are its most expensive, and averaging
    them in understated the residual (fedbuff_round 0.024 -> 0.080)."""

    @staticmethod
    def _legs(real_units, sim_units, step):
        """Both sides advance `step` s/unit, so a correct matched-work comparison
        reads 0.0 however many units each side got to."""
        def side(n, vclock):
            out = []
            for r in range(1, n + 1):
                e = {"event": "agg_round", "round": r,
                     "contributing_trainers": ["1"]}
                # cost RISES with progress: r*step, the trend that makes an
                # unequal-length mean the wrong number.
                t = sum(i * step for i in range(1, r + 1))
                e["ts"] = t
                if vclock:
                    e["vclock_now"] = t
                out.append(e)
            return {"agg_rounds": out}
        return side(real_units, False), side(sim_units, True)

    def test_a_longer_real_no_longer_biases_the_mean(self):
        from parity.checks import throughput_parity
        real, sim = self._legs(60, 40, step=1.0)
        res = throughput_parity(real, sim)
        assert res["decided_on"] == "matched_window_rel_diff"
        # Same per-unit schedule on both sides -> matched work reads ~0 ...
        assert res["matched_window_rel_diff"] < 0.01
        # ... while each side's own full-run mean does NOT, because real ran on
        # into the expensive tail. That gap is the defect this closes.
        assert res["rel_diff"] > 0.2

    def test_the_quantile_tuple_is_reported(self):
        """A tail difference has to be visible to be measurable later."""
        from parity.checks import throughput_parity
        real, sim = self._legs(40, 40, step=1.0)
        pu = throughput_parity(real, sim)["per_unit_s"]
        assert set(pu) == {"p50", "p90", "p99"}
        assert set(pu["p90"]) == {"real", "sim", "rel_diff"}

    def test_a_pure_shape_difference_does_not_gate(self):
        """Same mean, same median, sim's p90 30% heavier — the case the mean is
        blind to. The rung must PASS and still expose the gap in the tuple: a tail
        has no replicate floor yet, so gating it would be a hand-typed threshold,
        the mistake this family just finished undoing (§D-74/§D-76)."""
        from parity.checks import throughput_parity
        # n units yield n-1 advances, so the leading step is dropped: build the
        # advance sequences directly. Both mean 10 and median 10; sim's spread is
        # symmetric, which moves only its tail.
        real = {"agg_rounds": [{"event": "agg_round", "round": r, "ts": t,
                                "contributing_trainers": ["1"]}
                               for r, t in enumerate(_cum([10] + [10] * 19), 1)]}
        sim = {"agg_rounds": [{"event": "agg_round", "round": r, "ts": t,
                               "vclock_now": t, "contributing_trainers": ["1"]}
                              for r, t in enumerate(
                                  _cum([10] + [7] * 9 + [10] + [13] * 9), 1)]}
        res = throughput_parity(real, sim)
        assert res["matched_window_rel_diff"] < 0.01, "means are identical"
        assert res["per_unit_s"]["p50"]["rel_diff"] < 0.02
        assert res["per_unit_s"]["p90"]["rel_diff"] > 0.2
        assert res["ok"] is True, "a tail alone must not fail an ungated bound"


def _cum(steps):
    t, out = 0.0, []
    for s in steps:
        t += s
        out.append(t)
    return out


class TestThresholdProvenance:
    """Every enforced threshold must declare WHERE its number came from.

    The taxonomy is decided by one question — does this quantity differ between
    two runs that should be identical? INVARIANT: no, so the floor is zero and the
    gate is exact. CALIBRATED: yes, so the gate must come from a measured floor.
    POLICY: there is no paired comparison at all, so no floor is definable and the
    number is an engineering choice.

    There is no "hand-typed" class because that is the FAILURE state, and it is
    what produced every defect this batch fixed. This test is the ratchet that
    stops the next one: a new rung cannot land unclassified."""

    def test_every_rung_declares_its_provenance(self):
        from parity.checks import (CHECK_META, THRESHOLD_PROVENANCE,
                                   UNCLASSIFIED_PROVENANCE)
        missing = set(CHECK_META) - set(THRESHOLD_PROVENANCE) - UNCLASSIFIED_PROVENANCE
        assert not missing, (
            f"new rung(s) {sorted(missing)} must declare INVARIANT / CALIBRATED / "
            f"POLICY in THRESHOLD_PROVENANCE — or be named in "
            f"UNCLASSIFIED_PROVENANCE with a reason")

    def test_the_registry_has_no_phantom_rungs(self):
        from parity.checks import (CHECK_META, THRESHOLD_PROVENANCE,
                                   UNCLASSIFIED_PROVENANCE)
        phantom = (set(THRESHOLD_PROVENANCE) | UNCLASSIFIED_PROVENANCE) - set(CHECK_META)
        assert not phantom, f"registry names non-rungs: {sorted(phantom)}"

    def test_unclassified_only_ever_shrinks(self):
        """A ratchet, not a target. Classifying one is a one-line PR; letting the
        list grow is how the backlog became 62 gates nobody had derived."""
        from parity.checks import UNCLASSIFIED_PROVENANCE
        assert len(UNCLASSIFIED_PROVENANCE) <= 11, (
            "unclassified rungs must not grow — classify the new one instead")

    def test_an_invariant_is_never_floor_gated(self):
        """Floor-gating a structural property licenses drift in something that
        must never drift: its floor is zero by construction, so a measured floor
        could only ever loosen it."""
        import inspect

        from parity.checks import (INVARIANT, THRESHOLD_PROVENANCE,
                                   run_all_parity)
        src = inspect.getsource(run_all_parity)
        import re
        block = src[src.index("_floor_specs = {"):src.index("_tol, _ungradeable")]
        gated = set(re.findall(r'"(\w+)":\s*\(', block))
        bad = [r for r in gated
               if THRESHOLD_PROVENANCE.get(r, (None, None))[0] == INVARIANT]
        assert not bad, f"INVARIANT rungs must not be floor-gated: {bad}"

    def test_every_floor_gated_rung_is_declared_calibrated(self):
        """The registry and the wiring must not drift apart."""
        import inspect
        import re

        from parity.checks import CALIBRATED, THRESHOLD_PROVENANCE, run_all_parity
        src = inspect.getsource(run_all_parity)
        block = src[src.index("_floor_specs = {"):src.index("_tol, _ungradeable")]
        for rung in re.findall(r'"(\w+)":\s*\(', block):
            cls, floor = THRESHOLD_PROVENANCE[rung]
            assert cls == CALIBRATED, f"{rung} is floor-gated but declared {cls}"
            assert floor is not None, f"{rung} is floor-gated but declares no metric"

    def test_the_debt_is_reported_not_hidden(self):
        """44 CALIBRATED rungs still have no measured floor. That is the honest
        number, and the batches are what shrink it — a gate nobody derived is
        either too tight (grades noise) or too loose (passes divergence), and
        0-fail on a control cannot tell you which (§D-24)."""
        from parity.checks import calibration_debt
        debt = calibration_debt()
        assert "per_round_advance" in debt, "its KS 0.2 passed a 27% tail gap"
        assert len(debt) <= 44
