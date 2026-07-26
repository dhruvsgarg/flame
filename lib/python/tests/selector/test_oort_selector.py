# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Oort selector tests."""

import json
from datetime import timedelta

import pytest

from flame.selector.oort import OortSelector
from flame.selector.async_oort import AsyncOortSelector
from flame.selector.properties import (
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_STAT_UTILITY,
)


@pytest.fixture
def oort():
    return OortSelector(aggr_num=3)


@pytest.fixture
def async_oort():
    # Minimal kwargs required by AsyncOortSelector.__init__; the framework is
    # patched to PyTorch in conftest (flame.selector.async_oort included there).
    return AsyncOortSelector(
        c=30,
        aggGoal=10,
        evalGoalFactor=0.5,
        roundNudgeType="last_train",
        selectType="default",
    )


class TestOortInit:
    def test_defaults(self, oort):
        assert oort.aggr_num == 3
        assert oort.num_of_ends == int(3 * 1.3)
        assert isinstance(oort.selected_ends, set)
        assert oort.ordered_updates_recv_ends == []
        assert 0.0 < oort.exploration_factor <= 1.0


class TestOortColdStart:
    def test_first_round_random_selection(self, oort, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert len(result) == oort.num_of_ends
        for end_id in result:
            assert end_id in ends

    def test_in_flight_excluded_next_round(
        self, oort, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        first_picked = set(oort.selected_ends)

        channel_props["round"] = 2
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in first_picked or end_id in oort.selected_ends


class TestOortIdempotentWithinRound:
    def test_same_round_returns_same_set(self, oort, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        r1 = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        r2 = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert set(r1.keys()) == set(r2.keys())


class TestVersionKeySymmetry:
    """oort mirrors async_oort's optional no-repeat-this-tuple filter, for
    interface symmetry. Inert unless a caller passes both kwargs -- no
    current caller does (sync's round-scoped `selected_ends` guard already
    prevents a within-round re-pick)."""

    def test_absent_kwargs_are_noop(self, oort, make_ends, channel_props):
        ends = make_ends(count=5, prefix="t")
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert len(result) == oort.num_of_ends

    def test_matching_version_key_excludes_candidate(self, oort, channel_props, make_ends):
        ends = make_ends(count=1, prefix="t")
        end_id = next(iter(ends))
        result = oort.select(
            ends,
            channel_props,
            trainer_unavail_list=[],
            task_to_perform="train",
            agg_version_key=(1, 0),
            trainer_version_keys={end_id: (1, 0)},
        )
        assert result == {}

    def test_mismatched_version_key_still_selects(self, oort, channel_props, make_ends):
        ends = make_ends(count=1, prefix="t")
        end_id = next(iter(ends))
        result = oort.select(
            ends,
            channel_props,
            trainer_unavail_list=[],
            task_to_perform="train",
            agg_version_key=(1, 0),
            trainer_version_keys={end_id: (0, 0)},
        )
        assert end_id in result


class TestTemporalUncertaintyFidelity:
    """UCB temporal term keys on the agg round of the end's last RECEIVED update
    (PROP_LAST_RETURNED_ROUND, stamped at receipt by the aggregator), reference
    Oort/REFL — registration-initialized so the bonus is defined for every client.
    Supersedes the D5 last-selected stamping (see PARITY D7)."""

    def test_registration_init_fires(self, oort, make_ends):
        from flame.selector.properties import PROP_LAST_RETURNED_ROUND

        ends = make_ends(count=4, prefix="t", stat_utility=1.0)
        bonus = oort.calculate_temporal_uncertainty_of_trainer(ends, "t0", 50)
        assert bonus > 0
        assert ends["t0"].get_property(PROP_LAST_RETURNED_ROUND) == 50

    def test_older_receipt_gets_larger_bonus(self, oort, make_ends):
        from flame.selector.properties import PROP_LAST_RETURNED_ROUND

        ends = make_ends(count=2, prefix="t", stat_utility=1.0)
        ends["t0"].set_property(PROP_LAST_RETURNED_ROUND, 5)
        ends["t1"].set_property(PROP_LAST_RETURNED_ROUND, 95)
        old = oort.calculate_temporal_uncertainty_of_trainer(ends, "t0", 100)
        recent = oort.calculate_temporal_uncertainty_of_trainer(ends, "t1", 100)
        assert old > recent > 0

    def test_disabled_zeroes_term(self, make_ends):
        from flame.selector.oort import OortSelector

        sel = OortSelector(aggr_num=3, enable_temporal=False)
        ends = make_ends(count=2, prefix="t", stat_utility=1.0)
        assert sel.calculate_temporal_uncertainty_of_trainer(ends, "t0", 100) == 0.0


class TestRoundPreferredDuration:
    """Guards the Jun-15 parity fix: pref must be the round_threshold-th
    PERCENTILE of candidate durations (reference Oort sorts the list before
    indexing; the FLAME reimplementation had dropped the sort, so `pref` was an
    arbitrary dict-position duration -> the oort selector-scoring divergence)."""

    def _ends_with_durations(self, make_ends, durations_s):
        # Insert in the GIVEN (unsorted) order so a missing sort is detectable.
        ends = make_ends([f"e{i}" for i in range(len(durations_s))])
        for (eid, e), d in zip(ends.items(), durations_s):
            e.set_property(PROP_CLIENT_TASK_TRAIN_DURATION, timedelta(seconds=d))
        return ends

    def test_round_preferred_duration_is_sorted_percentile(self, oort, make_ends):
        # Deliberately unsorted insertion order. round_threshold default = 30.
        durations = [50, 10, 40, 20, 30]
        ends = self._ends_with_durations(make_ends, durations)
        oort.round_threshold = 30
        idx = int(len(durations) * 30 / 100.0)  # = 1
        expected = sorted(durations)[idx]        # sorted=[10,20,30,40,50] -> 20
        pref = oort.calculate_round_preferred_duration(ends)
        assert pref.total_seconds() == expected
        # The buggy (unsorted) impl would have returned insertion-order[idx] = 10.
        assert pref.total_seconds() != durations[idx]

    def test_pref_is_monotone_in_threshold(self, oort, make_ends):
        durations = [5, 9, 1, 7, 3, 11, 13, 2, 6, 8]
        ends = self._ends_with_durations(make_ends, durations)
        prev = -1.0
        for thr in (10, 30, 50, 70, 90):
            oort.round_threshold = thr
            cur = oort.calculate_round_preferred_duration(ends).total_seconds()
            assert cur >= prev, f"pref not monotone in threshold at {thr}"
            prev = cur

    def test_threshold_100_is_unbounded(self, oort, make_ends):
        ends = self._ends_with_durations(make_ends, [10, 20, 30])
        oort.round_threshold = 100.0
        assert oort.calculate_round_preferred_duration(ends).total_seconds() == 99999


class TestAsyncRoundPreferredDuration:
    """Same Jun-15 sort guard for AsyncOortSelector (felix stack). The async
    selector carries its OWN copy of calculate_round_preferred_duration, so the
    sort fix must be guarded independently of OortSelector's."""

    def _ends_with_durations(self, make_ends, durations_s):
        ends = make_ends([f"e{i}" for i in range(len(durations_s))])
        for (eid, e), d in zip(ends.items(), durations_s):
            e.set_property(PROP_CLIENT_TASK_TRAIN_DURATION, timedelta(seconds=d))
        return ends

    def test_round_preferred_duration_is_sorted_percentile(
        self, async_oort, make_ends
    ):
        durations = [50, 10, 40, 20, 30]
        ends = self._ends_with_durations(make_ends, durations)
        async_oort.round_threshold = 30
        idx = int(len(durations) * 30 / 100.0)  # = 1
        expected = sorted(durations)[idx]        # sorted=[10,20,30,40,50] -> 20
        pref = async_oort.calculate_round_preferred_duration(ends)
        assert pref.total_seconds() == expected
        assert pref.total_seconds() != durations[idx]  # buggy unsorted -> 10

    def test_pref_is_monotone_in_threshold(self, async_oort, make_ends):
        durations = [5, 9, 1, 7, 3, 11, 13, 2, 6, 8]
        ends = self._ends_with_durations(make_ends, durations)
        prev = -1.0
        for thr in (10, 30, 50, 70, 90):
            async_oort.round_threshold = thr
            cur = async_oort.calculate_round_preferred_duration(ends).total_seconds()
            assert cur >= prev, f"pref not monotone in threshold at {thr}"
            prev = cur

    def test_threshold_100_is_unbounded(self, async_oort, make_ends):
        ends = self._ends_with_durations(make_ends, [10, 20, 30])
        async_oort.round_threshold = 100.0
        pref = async_oort.calculate_round_preferred_duration(ends)
        assert pref.total_seconds() == 99999


class TestAsyncOortSystemUtilTelemetry:
    """AsyncOortSelector never emitted round_preferred_duration_s/round_threshold/
    sys_util_mean/pref_binds -- sync oort.py had it, async didn't, so a real
    vs sim divergence in the Oort speed-penalty couldn't be directly observed,
    only inferred. Guards the ported telemetry."""

    def test_selection_emits_pref_and_system_util_fields(
        self, tmp_path, async_oort, make_ends
    ):
        from flame import telemetry
        from flame.channel import (
            KEY_CH_SELECT_REQUESTER, KEY_CH_STATE, VAL_CH_STATE_SEND,
        )

        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            async_oort.round_threshold = 30  # low percentile -> forces binding
            durations = [8, 10, 14, 20, 22, 26, 28, 30, 32, 36]
            ends = make_ends([f"t{i}" for i in range(len(durations))])
            for (eid, e), d in zip(ends.items(), durations):
                e.set_property(
                    PROP_CLIENT_TASK_TRAIN_DURATION, timedelta(seconds=d)
                )
                # PROP_STAT_UTILITY None routes the end to unexplored_end_ids,
                # and calculate_total_utility short-circuits before
                # recomputing round_preferred_duration -- must be set for the
                # scored (exploitation) path.
                e.set_property(PROP_STAT_UTILITY, 1.0)

            channel_props = {
                "round": 5,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }
            # model_version=5 (non-zero) so this takes the scored path
            # (calculate_total_utility -> calculate_round_preferred_duration),
            # not the model_version==0 cold-start random branch.
            async_oort.select(
                ends, channel_props, trainer_unavail_list=[],
                task_to_perform="train", agg_version_key=(5, 0),
            )

            events = [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
            ]
            sels = [e for e in events if e["event"] == "selection"]
            assert len(sels) == 1
            s = sels[0]

            # round_preferred_duration_s must match the sorted-percentile
            # value TestAsyncRoundPreferredDuration proves the pure function
            # computes -- confirms select() wires it into telemetry, not
            # leaves it None.
            expected_pref = async_oort.calculate_round_preferred_duration(
                ends
            ).total_seconds()
            assert s["round_preferred_duration_s"] == expected_pref
            assert s["round_threshold"] == 30

            # pref is well below max duration, so at least one end should be
            # penalized -- sys_util_mean/pref_binds catches a "penalty never
            # binds" regression.
            assert s["sys_util_mean"] is not None
            assert s["pref_binds"] is True
            assert 0.0 < s["frac_penalized"] <= 1.0
        finally:
            telemetry.shutdown()

    def test_singleton_filtered_ends_still_uses_full_pool_for_pref(
        self, tmp_path, async_oort, make_ends
    ):
        """Async dispatches one freed trainer per SEND call, so
        calculate_round_preferred_duration used to see a population of one --
        the percentile trivially returns that candidate's own duration, so it
        can never exceed pref. Reference Oort computes the percentile from
        ALL tracked clients, not just this round's feasible subset. Fix:
        calculate_total_utility's duration population is now the full
        `connected_ends`, not `filtered_ends`.

        Here only one of ten registered ends is eligible (the rest unavail)
        -- the slowest one (36s). Pre-fix this singleton would trivially set
        its own pref, never binding. Post-fix, pref reflects the full
        10-trainer spread, so the straggler gets penalized.
        """
        from flame import telemetry
        from flame.channel import (
            KEY_CH_SELECT_REQUESTER, KEY_CH_STATE, VAL_CH_STATE_SEND,
        )

        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            async_oort.round_threshold = 10.0  # the real default
            durations = [8, 10, 14, 20, 22, 26, 28, 30, 32, 36]
            ends = make_ends([f"t{i}" for i in range(len(durations))])
            for (eid, e), d in zip(ends.items(), durations):
                e.set_property(
                    PROP_CLIENT_TASK_TRAIN_DURATION, timedelta(seconds=d)
                )
                e.set_property(PROP_STAT_UTILITY, 1.0)

            # Only the slowest trainer (t9, 36s) is eligible this round --
            # mirrors async's one-at-a-time dispatch cadence.
            unavail = [f"t{i}" for i in range(9)]

            channel_props = {
                "round": 5,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }
            async_oort.select(
                ends, channel_props, trainer_unavail_list=unavail,
                task_to_perform="train", agg_version_key=(5, 0),
            )

            events = [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
            ]
            sels = [e for e in events if e["event"] == "selection"]
            assert len(sels) == 1
            s = sels[0]

            # Pre-fix this would be 36.0 (the singleton's own duration,
            # self-referentially always <= itself). Post-fix it's the 10th
            # percentile of the FULL 10-trainer pool.
            assert s["round_preferred_duration_s"] < 36.0
            t9_util = (s.get("per_trainer") or {}).get("t9", {}).get("system_util")
            assert t9_util is not None and t9_util < 1.0
        finally:
            telemetry.shutdown()


class TestRewardNormalization:
    """Guards PARITY D2 (Jun-16): the statistical reward must be normalized+clipped
    into ~[0,1] (reference Oort get_norm) before the temporal/UCB term is added, so
    exploration is not inert. Reference: oort/oort.py get_norm:394 + score:292-295."""

    def test_norm_stats_matches_reference_get_norm(self):
        from flame.selector import scoring
        rewards = [10.0, 30.0, 20.0, 40.0, 100.0]
        _min, _range, clip = scoring.oort_norm_stats(rewards, clip_bound=0.9)
        s = sorted(rewards)
        # reference: clip = sorted[min(int(n*clip_bound), n-1)]; min*0.999; range floored
        assert clip == s[min(int(len(s) * 0.9), len(s) - 1)]
        assert _min == s[0] * 0.999
        assert _range == max(s[-1] - _min, 1e-4)

    def test_normalize_reward_clips_and_scales(self):
        from flame.selector import scoring
        _min, _range, clip = scoring.oort_norm_stats([10.0, 20.0, 100.0], clip_bound=0.5)
        # clip_value = sorted[int(3*0.5)=1] = 20 -> a raw 100 is clipped to 20
        assert scoring.oort_normalize_reward(100.0, _min, _range, clip) == \
            scoring.oort_normalize_reward(20.0, _min, _range, clip)
        # normalized reward is bounded ~[0,1]
        n = scoring.oort_normalize_reward(10.0, _min, _range, clip)
        assert 0.0 <= n <= 1.0

    def test_calculate_total_utility_normalizes_believed_I(self, oort, make_ends):
        # Raw stat-utilities ~70; after D2 the audit's believed_I must land in ~[0,1].
        from flame.selector.properties import PROP_STAT_UTILITY, PROP_END_ID, PROP_UTILITY
        ids = [f"e{i}" for i in range(5)]
        ends = make_ends(ids)
        raws = [66.0, 68.0, 70.0, 72.0, 74.0]
        for (eid, e), r in zip(ends.items(), raws):
            e.set_property(PROP_STAT_UTILITY, r)
            e.set_property(PROP_CLIENT_TASK_TRAIN_DURATION, timedelta(seconds=10))
        util_list = [{PROP_END_ID: eid, PROP_UTILITY: r} for eid, r in zip(ids, raws)]
        assert oort.normalize_reward is True
        oort.calculate_total_utility(util_list, ends, round=5)
        for comp in oort._audit_components.values():
            assert 0.0 <= comp["believed_I"] <= 1.0, comp

    def test_normalization_can_be_disabled(self, make_ends):
        from flame.selector.properties import PROP_END_ID, PROP_UTILITY, PROP_CLIENT_TASK_TRAIN_DURATION
        sel = OortSelector(aggr_num=3, normalize_reward=False)
        ids = [f"e{i}" for i in range(3)]
        ends = make_ends(ids)
        for eid, e in ends.items():
            e.set_property(PROP_CLIENT_TASK_TRAIN_DURATION, timedelta(seconds=10))
        raws = [66.0, 70.0, 74.0]
        util_list = [{PROP_END_ID: eid, PROP_UTILITY: r} for eid, r in zip(ids, raws)]
        sel.calculate_total_utility(util_list, ends, round=5)
        # raw believed_I retained when disabled
        assert any(c["believed_I"] > 1.0 for c in sel._audit_components.values())


class TestAlgorithmHyperparams:
    """Guards Jun-16 fidelity pass: Oort algorithm knobs default to the paper
    (standalone Oort) and are overridable per-baseline via kwargs (REFL fork)."""

    def test_defaults_match_oort_paper(self, oort):
        from flame.selector.scoring import OORT_PAPER_DEFAULTS as d
        assert oort.round_threshold == d["round_threshold"]   # 10 (was 30)
        assert oort.clip_bound == d["clip_bound"]             # 0.98 (was 0.95)
        assert oort.cut_off_util == d["cut_off_util"]         # 0.7 (was 0.95)
        assert oort.alpha == d["round_penalty"]               # 2.0
        assert oort.exploration_factor_decay == d["exploration_decay"]  # 0.95
        assert oort.min_exploration_factor == d["exploration_min"]      # 0.2

    def test_refl_fork_overrides_apply(self):
        # The values the refl baseline config passes to match the REFL fork.
        sel = OortSelector(aggr_num=3, round_threshold=30, clip_bound=0.9,
                           cut_off_util=0.05, exploration_decay=0.98,
                           exploration_min=0.3)
        assert sel.round_threshold == 30
        assert sel.clip_bound == 0.9
        assert sel.cut_off_util == 0.05
        assert sel.exploration_factor_decay == 0.98
        assert sel.min_exploration_factor == 0.3

    def test_cutoff_util_thresholds_high_utility_boundary(self, oort, make_ends):
        # cutoff must scale the (exploitLen-th HIGHEST) utility by cut_off_util,
        # not an arbitrary low-end value. ascending list of utilities.
        from flame.selector.properties import PROP_END_ID, PROP_UTILITY
        utils = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ul = [{PROP_END_ID: f"e{i}", PROP_UTILITY: u} for i, u in enumerate(utils)]
        oort.exploration_factor = 0.0          # exploitLen = num_of_ends
        oort.cut_off_util = 0.5
        n = len(utils)
        # exploit_len = int(n*(1-0)) = n -> index = n-1-n < 0 -> clamps to 0 (lowest)
        cut = oort.cutoff_util(ul, n)
        assert cut == 0.5 * utils[0]
        # with high exploration, exploitLen small -> boundary near the TOP
        oort.exploration_factor = 0.8
        exploit_len = int(n * (1.0 - oort.exploration_factor))  # float-faithful to code
        cut2 = oort.cutoff_util(ul, n)
        assert cut2 == 0.5 * utils[n - 1 - exploit_len]
        assert cut2 > cut  # boundary moved up toward higher utilities


class TestPacerFidelity:
    """Guards §S.pacer: pacer() must faithfully port reference Oort
    (third_party/Oort/oort/oort.py:184-199) — a FLAT plateau relaxes
    round_threshold, a SHARP change tightens it, keyed on the current round.
    The earlier port raised on any dip and never lowered (monotonic ratchet)."""

    def _seed_history(self, oort, last_vals, curr_vals):
        # two pacer_step windows of mean exploited utility
        oort.exploitation_util_history = list(last_vals) + list(curr_vals)

    def test_flat_plateau_relaxes(self, oort):
        oort.pacer_step, oort.pacer_delta, oort.round_threshold = 2, 5.0, 10.0
        # last sum 20, curr sum 21 -> |Δ|=1 <= 0.1*20=2 -> RELAX
        self._seed_history(oort, [10.0, 10.0], [10.0, 11.0])
        oort.pacer(round=4)            # 4 >= 2*step and 4 % step == 0
        assert oort.round_threshold == 15.0

    def test_sharp_change_tightens(self, oort):
        oort.pacer_step, oort.pacer_delta, oort.round_threshold = 2, 5.0, 30.0
        # last sum 20, curr sum 200 -> |Δ|=180 >= 5*20=100 -> TIGHTEN
        self._seed_history(oort, [10.0, 10.0], [100.0, 100.0])
        oort.pacer(round=4)
        assert oort.round_threshold == 25.0

    def test_moderate_change_no_move(self, oort):
        oort.pacer_step, oort.pacer_delta, oort.round_threshold = 2, 5.0, 30.0
        # last 20, curr 30 -> |Δ|=10, between 0.1*20=2 and 5*20=100 -> NO MOVE
        self._seed_history(oort, [10.0, 10.0], [15.0, 15.0])
        oort.pacer(round=4)
        assert oort.round_threshold == 30.0

    def test_tighten_floored_at_pacer_delta(self, oort):
        oort.pacer_step, oort.pacer_delta, oort.round_threshold = 2, 5.0, 5.0
        self._seed_history(oort, [10.0, 10.0], [100.0, 100.0])
        oort.pacer(round=4)
        assert oort.round_threshold == 5.0  # max(delta, thr-delta) floors here

    def test_no_move_off_cadence_or_warmup(self, oort):
        oort.pacer_step, oort.pacer_delta, oort.round_threshold = 2, 5.0, 10.0
        self._seed_history(oort, [10.0, 10.0], [10.0, 11.0])
        oort.pacer(round=3)            # 3 % 2 != 0 -> no move
        assert oort.round_threshold == 10.0
        oort.pacer(round=2)            # 2 < 2*step(4) warmup -> no move
        assert oort.round_threshold == 10.0

    def test_async_oort_pacer_faithful(self, async_oort):
        # felix/fluxtune's shared AsyncOortSelector.pacer must use the same
        # two-branch reference logic, keyed on the passed current_round.
        async_oort.pacer_step, async_oort.pacer_delta = 2, 5.0
        async_oort.exploitation_util_history = [10.0, 10.0, 10.0, 11.0]
        async_oort.round_threshold = 10.0
        async_oort.pacer(current_round=4)        # FLAT |Δ|=1 <= 2 -> relax
        assert async_oort.round_threshold == 15.0
        async_oort.exploitation_util_history = [10.0, 10.0, 100.0, 100.0]
        async_oort.round_threshold = 30.0
        async_oort.pacer(current_round=4)        # SHARP |Δ|=180 >= 100 -> tighten
        assert async_oort.round_threshold == 25.0
        async_oort.round_threshold = 30.0
        async_oort.pacer(current_round=3)         # off-cadence
        assert async_oort.round_threshold == 30.0

    def test_async_oort_pacer_once_per_round(self, async_oort):
        # Once-per-round guard: a burst of same-model_version select() calls
        # must fire the pacer's state transition at most once, not per call.
        async_oort.pacer_step, async_oort.pacer_delta = 2, 5.0
        async_oort.exploitation_util_history = [10.0, 10.0, 10.0, 11.0]
        async_oort.round_threshold = 10.0
        assert async_oort._last_pacer_round is None
        for _ in range(5):  # simulate 5 select() calls for the SAME round
            if 4 != async_oort._last_pacer_round:
                async_oort.pacer(current_round=4)
                async_oort._last_pacer_round = 4
        assert async_oort.round_threshold == 15.0  # moved ONCE, not 5x (would be 35.0)


class TestOortCleanup:
    def test_cleanup_recvd_ends_clears_inflight(self, oort, make_ends):
        oort.selected_ends.update(["a", "b", "c"])
        oort.ordered_updates_recv_ends = ["a", "b"]
        oort._cleanup_recvd_ends(make_ends(["a", "b", "c"]))
        assert oort.selected_ends == {"c"}
        assert oort.ordered_updates_recv_ends == []


class TestChallenge13SendStateCleanup:
    """The 'invalid prior selection' cleanup in _handle_send_state must key off
    CONNECTED membership, not availability-eligibility (Challenge 13). An
    in-flight trainer that merely went UN_AVL / wrong-task-type is absent from
    the filtered eligible pool but still connected & computing — it must stay in
    selected_ends. Only a genuinely disconnected end (gone from the channel) is
    removed. Concurrency is sized so extra==0 and the method returns right after
    cleanup, isolating the membership check."""

    def test_unavailable_inflight_retained_when_connected_pool_passed(
        self, async_oort, make_ends
    ):
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": {"t1", "t2"}}
        connected = make_ends(["t1", "t2", "t3"])   # all still connected
        eligible = {}                               # all currently unavailable
        out = async_oort._handle_send_state(
            ends=eligible, concurrency=2, channel_props={},
            connected_ends=connected,
        )
        assert out == {}
        # t1/t2 unavailable but connected → in-flight tracking preserved.
        assert async_oort.selected_ends["agg"] == {"t1", "t2"}

    def test_disconnected_inflight_removed(self, async_oort, make_ends):
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": {"t1", "t2"}}
        connected = make_ends(["t1"])               # t2 genuinely gone
        async_oort._handle_send_state(
            ends={}, concurrency=1, channel_props={},
            connected_ends=connected,
        )
        assert async_oort.selected_ends["agg"] == {"t1"}

    def test_fallback_to_eligible_when_no_connected_pool(
        self, async_oort, make_ends
    ):
        # Backward-compat: with connected_ends omitted the check falls back to
        # `ends` — the pre-fix behavior. Guards the default-arg contract.
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": {"t1"}}
        async_oort._handle_send_state(
            ends=make_ends(["t1"]), concurrency=1, channel_props={},
        )
        assert async_oort.selected_ends["agg"] == {"t1"}


class TestRecvStateNeverWritesNewSelections:
    """`_handle_recv_state` must be read-only over `selected_ends`: it reports
    who send-state already dispatched, minus anyone who has replied, and
    never picks new candidates itself -- that's `_handle_send_state`'s job.
    A prior version resampled fresh candidates here when `selected_ends` was
    empty (a `None` vs `"none"` comparison bug), claiming ends into
    `all_selected` before they were ever sent anything -> permanent deadlock.
    The fallback is removed; empty `selected_ends` in -> empty result out.
    """

    def test_empty_selected_ends_returns_empty_never_touched(
        self, async_oort, make_ends
    ):
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": set()}   # nothing currently in flight
        async_oort.all_selected = {}
        ends = make_ends(count=5, prefix="t")        # KEY_END_STATE never set -> None

        result = async_oort._handle_recv_state(ends=ends, concurrency=5)

        assert result == {}
        assert async_oort.selected_ends["agg"] == set()
        assert async_oort.all_selected == {}

    def test_empty_selected_ends_returns_empty_even_with_real_state(
        self, async_oort, make_ends
    ):
        from flame.end import KEY_END_STATE, VAL_END_STATE_NONE

        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": set()}
        async_oort.all_selected = {}
        ends = make_ends(count=3, prefix="t")
        for e in ends.values():
            e.set_property(KEY_END_STATE, VAL_END_STATE_NONE)

        result = async_oort._handle_recv_state(ends=ends, concurrency=3)

        # No resample fallback at all -> a real state doesn't matter either.
        assert result == {}
        assert async_oort.selected_ends["agg"] == set()
        assert async_oort.all_selected == {}

    def test_only_reports_already_selected_ends_minus_recvd(
        self, async_oort, make_ends
    ):
        from flame.end import KEY_END_STATE, VAL_END_STATE_RECVD

        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": {"t0", "t1"}}  # send-state already dispatched
        async_oort.all_selected = {"t0": 0.0, "t1": 0.0}
        ends = make_ends(["t0", "t1"])
        ends["t0"].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)  # t0 replied

        result = async_oort._handle_recv_state(ends=ends, concurrency=2)

        assert set(result) == {"t1"}
        assert async_oort.selected_ends["agg"] == {"t1"}
        # all_selected untouched here -- cleared by the aggregator after
        # processing t0's grad (_cleanup_recvd_ends), not by this function.
        assert async_oort.all_selected == {"t0": 0.0, "t1": 0.0}


class TestPendingCommitExcludedFromSelection:
    """async_oort must exclude the aggregator's VIRTUAL in-flight set
    (`_agg_pending_commit_ref`, bound live to the fwdllm aggregator's
    `_sim_pending_commit`) from selection eligibility -- so a returned-but-
    uncommitted trainer is NEVER re-dispatched even when `all_selected` has been
    pruned by a physical event (the recv-fifo re-select loop / RECVD-NONE cleanup
    a slow sim triggers). This hardens over the all_selected-only guard (an unset
    ref -> byte-identical).
    """

    def test_pending_end_excluded_even_when_all_selected_empty(
        self, async_oort, make_ends
    ):
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": set()}
        async_oort.all_selected = {}                 # guard pruned by a physical event
        async_oort._agg_pending_commit_ref = {"t3"}  # agg still holds t3 in flight
        ends = make_ends(count=5, prefix="t")

        result = async_oort._handle_send_state(
            ends=ends, concurrency=5, channel_props={"round": 1},
            trainer_unavail_list=[], task_to_perform="train",
            agg_version_key=(1, 0, 0), trainer_version_keys={},
            connected_ends=ends,
        )

        # t3 is still outstanding in virtual time -> must not be re-picked.
        assert "t3" not in result
        # the other four are eligible -> selection still works (no over-restriction).
        assert len(result) >= 1
        assert set(result).issubset({"t0", "t1", "t2", "t4"})

    def test_all_pending_yields_no_selection(self, async_oort, make_ends):
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": set()}
        async_oort.all_selected = {}
        ends = make_ends(count=4, prefix="t")
        async_oort._agg_pending_commit_ref = set(ends)  # every trainer in flight

        result = async_oort._handle_send_state(
            ends=ends, concurrency=4, channel_props={"round": 1},
            trainer_unavail_list=[], task_to_perform="train",
            agg_version_key=(1, 0, 0), trainer_version_keys={},
            connected_ends=ends,
        )
        assert result == {}   # nobody eligible -> no re-dispatch-while-in-flight

    def test_empty_ref_is_byte_identical_noop(self, async_oort, make_ends):
        """Real / async_cifar10 never populate the ref -> selection is unchanged."""
        async_oort.requester = "agg"
        async_oort.selected_ends = {"agg": set()}
        async_oort.all_selected = {}
        # no _agg_pending_commit_ref attribute at all -> getattr default {}
        ends = make_ends(count=5, prefix="t")

        result = async_oort._handle_send_state(
            ends=ends, concurrency=5, channel_props={"round": 1},
            trainer_unavail_list=[], task_to_perform="train",
            agg_version_key=(1, 0, 0), trainer_version_keys={},
            connected_ends=ends,
        )
        assert len(result) >= 1
        assert set(result).issubset(set(ends))
