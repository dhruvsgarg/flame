# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""REFL_Oort selector tests."""

import pytest

from flame.selector.refl_oort import REFLOortSelector
from flame.selector.properties import PROP_LAST_RETURNED_ROUND


@pytest.fixture
def refl_oort():
    return REFLOortSelector(aggr_num=3, avail_priority=0)


class TestREFLTemporalFidelity:
    """PARITY/fidelity (Jun 23): refl's UCB temporal term was DEAD (refl_oort.select()
    let it divide by a None time_stamp; telemetry 0/7513 nonzero), so refl scored on
    stat_util alone (speed-uniform). Reference Oort/REFL key the bonus on the last-RECEIVED
    round (`time_stamp = self.epoch`, PROP_LAST_RETURNED_ROUND stamped at receipt),
    registration-initialized so it is never None. `enable_temporal=False` = ablation."""

    def test_enabled_by_default(self, refl_oort):
        assert refl_oort.enable_temporal is True

    def test_registration_init_fires(self, refl_oort, make_ends):
        # A never-returned candidate (no PROP_LAST_RETURNED_ROUND) is lazily initialized
        # to the current round (reference registers at current epoch) -> bonus is >0.
        ends = make_ends(count=4, prefix="t", stat_utility=1.0)
        eid = "t0"
        bonus = refl_oort.calculate_temporal_uncertainty_of_trainer(ends, eid, 50)
        assert bonus > 0
        assert ends[eid].get_property(PROP_LAST_RETURNED_ROUND) == 50

    def test_uses_received_round(self, refl_oort, make_ends):
        # An older last-RECEIVED round yields a larger bonus than a recent one
        # (under-selected / slower-returning clients are explored more).
        ends = make_ends(count=2, prefix="t", stat_utility=1.0)
        ends["t0"].set_property(PROP_LAST_RETURNED_ROUND, 5)    # returned long ago
        ends["t1"].set_property(PROP_LAST_RETURNED_ROUND, 95)   # returned recently
        old = refl_oort.calculate_temporal_uncertainty_of_trainer(ends, "t0", 100)
        recent = refl_oort.calculate_temporal_uncertainty_of_trainer(ends, "t1", 100)
        assert old > recent > 0

    def test_disabled_zeroes_term(self, make_ends):
        sel = REFLOortSelector(aggr_num=3, avail_priority=0, enable_temporal=False)
        ends = make_ends(count=2, prefix="t", stat_utility=1.0)
        assert sel.calculate_temporal_uncertainty_of_trainer(ends, "t0", 100) == 0.0


class TestREFLOortInit:
    def test_defaults(self, refl_oort):
        assert refl_oort.aggr_num == 3
        assert refl_oort.avail_priority == 0
        assert refl_oort.avail_tracker is None
        assert isinstance(refl_oort.selected_ends, set)


class TestREFLOortFiltersUnavail:
    def test_unavail_excluded(self, refl_oort, make_ends, channel_props):
        ends = make_ends(count=6, prefix="t")
        unavail = ["t0", "t1"]
        result = refl_oort.select(
            ends, channel_props, trainer_unavail_list=unavail, task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in unavail


class TestREFLOortInflightExcluded:
    def test_inflight_not_reselected_next_round(
        self, refl_oort, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        refl_oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        first = set(refl_oort.selected_ends)

        channel_props["round"] = 2
        refl_oort.newly_selected_this_round = set()
        result = refl_oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in first


class TestREFLExploresUnexplored:
    """FX-N49: REFL fork getTopK explores UNEXPLORED clients; the port sampled exploration
    from explored ones only, so cifar GPU refl picked the same 3 of 100 for 20 rounds."""

    def _run(self, sel, ends, rounds):
        chosen = set()
        for r in range(1, rounds + 1):
            picked = sel.select(ends, {"round": r, "cur_time": 0.0}, trainer_unavail_list=[],
                                task_to_perform="train", trainer_version_keys={}, agg_version_key=r)
            for e in picked:  # the round returns: stamp utility, free the slot
                ends[e].set_property("stat_utility", 1.0)
            sel.selected_ends = set()
            chosen |= set(picked)
        return chosen

    def test_rotates_through_cohort(self, make_ends):
        sel = REFLOortSelector(aggr_num=3, avail_priority=0, exploration_factor=0.9,
                               exploration_decay=0.98, exploration_min=0.2)
        sel.num_of_ends = 3
        chosen = self._run(sel, make_ends(count=100, prefix="t"), 20)
        assert len(chosen) >= 30, len(chosen)  # the unfaithful port gave 3

    def test_exploit_capped_below_explored(self, make_ends):
        # Reference: exploitLen = min(k - int(k*exploration), len(explored) - 1).
        sel = REFLOortSelector(aggr_num=3, avail_priority=0, exploration_factor=0.0,
                               exploration_decay=1.0, exploration_min=0.0)
        ends = make_ends(count=10, prefix="t")
        for e in ("t0", "t1", "t2"):
            ends[e].set_property("stat_utility", 1.0)
        picked = sel._select_with_oort_ucb(ends, set(ends), 3, 5)
        assert len(sel._last_exploit) == 2 and len(sel._last_explore) == 1
        assert len(set(picked)) == 3

    def test_priority_fill_is_random_then_oort(self, make_ends):
        sel = REFLOortSelector(aggr_num=3, avail_priority=0)
        ends = make_ends(count=10, prefix="t")
        picked = sel._select_priority_fill(ends, ["t0"], [f"t{i}" for i in range(1, 10)], 3, 2)
        assert "t0" in picked and len(set(picked)) == 3
