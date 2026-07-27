# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Contract for `AsyncSelectorBase`, run against every selector built on it.

Each construct here previously existed only in `async_oort.py`; `fedbuff.py`
and `async_random.py` carried drifted copies missing several of them, which is
what made `fedbuff_it_*` re-pick the same trainer on 34% of real commits vs
0.9% in sim (simulate_fwdllm.md R-A). Parameterizing over the subclasses is
what keeps them from drifting apart again.
"""

import time

import pytest

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.config import TrainerAvailState
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD
from flame.selector.async_base import AsyncSelectorBase, SelectContext
from flame.selector.async_random import AsyncRandomSelector
from flame.selector.fedbuff import FedBuffSelector
from flame.selector.properties import PROP_AVL_STATE

BUILDERS = [
    lambda **kw: FedBuffSelector(_seed=7, c=kw.pop("c", 4), aggGoal=2, **kw),
    lambda **kw: AsyncRandomSelector(_seed=7, c=kw.pop("c", 4), aggGoal=2, **kw),
]


@pytest.fixture(params=BUILDERS, ids=["fedbuff", "async_random"])
def build(request):
    return request.param


def _send_props(round_num=1):
    return {
        KEY_CH_STATE: VAL_CH_STATE_SEND,
        KEY_CH_SELECT_REQUESTER: "agg",
        "round": round_num,
    }


class TestSharedMechanism:
    def test_all_subclasses_share_the_base(self, build):
        assert isinstance(build(), AsyncSelectorBase)

    def test_selected_ends_is_keyed_by_requester(self, build, make_ends):
        sel = build()
        sel.select(make_ends(count=10, prefix="t"), _send_props(), [])
        # async selectors keep {requester: set}, not the bare set sync ones use
        assert isinstance(sel.selected_ends, dict)
        assert isinstance(sel.selected_ends["agg"], set)

    def test_respects_concurrency(self, build, make_ends):
        sel = build(c=3)
        chosen = sel.select(make_ends(count=10, prefix="t"), _send_props(), [])
        assert len(chosen) == 3

    def test_second_call_does_not_exceed_concurrency(self, build, make_ends):
        sel = build(c=3)
        ends = make_ends(count=10, prefix="t")
        first = sel.select(ends, _send_props(), [])
        second = sel.select(ends, _send_props(), [])
        assert len(first) == 3
        assert second == {}  # slots full, nothing more to dispatch

    def test_unavail_list_is_excluded(self, build, make_ends):
        sel = build(c=3)
        ends = make_ends(count=6, prefix="t")
        blocked = ["t0", "t1", "t2"]
        chosen = sel.select(ends, _send_props(), blocked)
        assert not set(chosen) & set(blocked)


class TestVersionKeyRePickGuard:
    """R-A: a trainer that already contributed to this exact version_key must
    not be dispatched to again. `fedbuff.py` swallowed both kwargs in **kwargs
    and never read them."""

    def test_same_version_key_contributor_is_skipped(self, build, make_ends):
        sel = build(c=5)
        ends = make_ends(count=6, prefix="t")
        agg_key = (3, 0)
        already = {"t0": agg_key, "t1": agg_key, "t2": agg_key}

        chosen = sel.select(
            ends, _send_props(), [],
            agg_version_key=agg_key, trainer_version_keys=already,
        )
        assert not set(chosen) & set(already)

    def test_older_version_key_contributor_is_eligible(self, build, make_ends):
        sel = build(c=6)
        ends = make_ends(count=6, prefix="t")
        stale = {eid: (2, 0) for eid in ends}

        chosen = sel.select(
            ends, _send_props(), [],
            agg_version_key=(3, 0), trainer_version_keys=stale,
        )
        assert len(chosen) == 6  # a stale key is no bar to re-selection


class TestPendingCommitGuard:
    """R1: `all_selected` alone stops being enough once a buffered return frees
    the channel slot early, so the aggregator's pending-commit set is consulted
    too (bound live via `_agg_pending_commit_ref`)."""

    def test_pending_commit_ends_are_not_redispatched(self, build, make_ends):
        sel = build(c=5)
        ends = make_ends(count=6, prefix="t")
        sel._agg_pending_commit_ref = {"t0", "t1"}

        chosen = sel.select(ends, _send_props(), [])
        assert not set(chosen) & {"t0", "t1"}

    def test_timeout_reclaim_also_clears_pending_commit(self, build, make_ends):
        sel = build(c=2)
        pending = {"stale"}
        sel._agg_pending_commit_ref = pending
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"stale"}}
        sel.all_selected = {"stale": time.time() - 1000}

        sel._reclaim_timed_out_ends(sel.selected_ends["agg"])
        # left behind, the end would be un-re-pickable forever despite the
        # reclaim freeing the concurrency slot
        assert "stale" not in pending


class TestAbandonClockIsVirtualInSim:
    """#1c: the abandon timeout must run on the clock the trainer commits on --
    virtual in sim -- or a slow sim evicts a still-outstanding trainer."""

    def test_vclock_now_drives_the_timeout(self, build, make_ends):
        sel = build()
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"t0"}}
        # stamped at virtual t=0; wall clock is ~1.7e9, so a wall-keyed
        # timeout would evict instantly
        sel.all_selected = {"t0": 0.0}

        sel._sim_now_s = 10.0  # only 10 virtual seconds elapsed
        sel._reclaim_timed_out_ends(sel.selected_ends["agg"])
        assert "t0" in sel.all_selected

        sel._sim_now_s = 10_000.0  # now well past send_timeout_wait_s
        sel._reclaim_timed_out_ends(sel.selected_ends["agg"])
        assert "t0" not in sel.all_selected

    def test_dispatch_stamp_uses_the_same_clock(self, build, make_ends):
        sel = build(c=2)
        props = _send_props()
        props["vclock_now"] = 500.0
        chosen = sel.select(make_ends(count=4, prefix="t"), props, [])
        assert all(sel.all_selected[e] == 500.0 for e in chosen)

    def test_send_timeout_is_configurable(self, build):
        assert build().send_timeout_wait_s == 90
        assert build(send_timeout_wait_s=300).send_timeout_wait_s == 300


class TestAvailabilityEligibility:
    def test_ineligible_avl_state_is_skipped(self, build, make_ends):
        sel = build(c=5)
        ends = make_ends(count=4, prefix="t")
        ends["t0"].set_property(PROP_AVL_STATE, TrainerAvailState.UN_AVL.value)

        chosen = sel.select(ends, _send_props(), [])
        assert "t0" not in chosen

    def test_none_avl_state_stays_eligible(self, build, make_ends):
        """Trainers without availability tracking must not be filtered out."""
        sel = build(c=4)
        ends = make_ends(count=4, prefix="t")
        assert len(sel.select(ends, _send_props(), [])) == 4

    def test_unknown_state_in_config_raises(self):
        with pytest.raises(ValueError, match="unknown state"):
            FedBuffSelector(
                _seed=1, c=2, aggGoal=1,
                task_eligible_states={"train": ["NOT_A_STATE"]},
            )


class TestRecvStateIsReadOnly:
    """A prior fedbuff `_handle_recv_state` resampled here, racing send-state
    dispatch; it must only report who is outstanding."""

    def test_recv_never_selects_new_ends(self, build, make_ends):
        """Mid-run (all_selected non-empty): RECV must not resample."""
        sel = build(c=3)
        ends = make_ends(count=8, prefix="t")
        sel.requester = "agg"
        sel.selected_ends = {"agg": set()}
        stamp = time.time()
        sel.all_selected = {"t7": stamp}  # something else already in flight

        recv_props = dict(_send_props(), **{KEY_CH_STATE: VAL_CH_STATE_RECV})
        assert sel.select(ends, recv_props, []) == {}
        assert sel.selected_ends["agg"] == set()
        assert sel.all_selected == {"t7": stamp}

    def test_recv_bootstraps_when_nothing_ever_dispatched(self, build, make_ends):
        """`channel.one_end()` caller (`allow_recv_bootstrap=True`), fresh
        selector: RECV must bootstrap-select -- a trainer's first call is
        RECV, not SEND, by protocol."""
        sel = build(c=3)
        ends = make_ends(count=1, prefix="agg")
        sel.requester = "trainer-x"
        sel.selected_ends = {"trainer-x": set()}
        sel.all_selected = {}

        recv_props = {
            KEY_CH_STATE: VAL_CH_STATE_RECV,
            KEY_CH_SELECT_REQUESTER: "trainer-x",
            "round": 1,
        }
        chosen = sel.select(ends, recv_props, [], allow_recv_bootstrap=True)
        assert set(chosen) == {"agg0"}
        assert sel.selected_ends["trainer-x"] == {"agg0"}

    def test_recv_never_bootstraps_for_a_dispatcher(self, build, make_ends):
        """`channel.ends()` caller (default, no `allow_recv_bootstrap`), fresh
        selector: must stay read-only. Bootstrapping here (the aggregator's
        RECV racing its own first SEND) froze `fedbuff_round`'s sim vclock at
        0.0 forever -- nothing dispatched, nothing to reclaim on."""
        sel = build(c=3)
        ends = make_ends(count=8, prefix="t")
        sel.requester = "agg"
        sel.selected_ends = {"agg": set()}
        sel.all_selected = {}

        recv_props = dict(_send_props(), **{KEY_CH_STATE: VAL_CH_STATE_RECV})
        assert sel.select(ends, recv_props, []) == {}
        assert sel.selected_ends["agg"] == set()
        assert sel.all_selected == {}

    def test_recv_drops_ends_already_received(self, build, make_ends):
        sel = build(c=3)
        ends = make_ends(count=4, prefix="t")
        sel.select(ends, _send_props(), [])
        in_flight = sorted(sel.selected_ends["agg"])
        ends[in_flight[0]].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)

        recv_props = dict(_send_props(), **{KEY_CH_STATE: VAL_CH_STATE_RECV})
        outstanding = sel.select(ends, recv_props, [])
        assert in_flight[0] not in outstanding
        assert set(outstanding) == set(in_flight[1:])

    def test_recv_order_is_process_stable(self, build, make_ends):
        sel = build(c=4)
        ends = make_ends(count=6, prefix="t")
        sel.select(ends, _send_props(), [])
        recv_props = dict(_send_props(), **{KEY_CH_STATE: VAL_CH_STATE_RECV})
        assert list(sel.select(ends, recv_props, [])) == sorted(
            sel.selected_ends["agg"]
        )


class TestCleanupDrainsEverything:
    """fedbuff capped the drain at `agg_goal`, permanently orphaning the excess
    each cycle and deadlocking once K changed dynamically."""

    def test_cleanup_recvd_ends_drains_all_not_agg_goal(self, build, make_ends):
        sel = build(c=6)  # aggGoal=2, so a capped drain would free only 2
        ends = make_ends(count=6, prefix="t")
        sel.select(ends, _send_props(), [])

        in_flight = sorted(sel.selected_ends["agg"])
        assert len(in_flight) == 6
        for eid in in_flight:
            ends[eid].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
        sel.ordered_updates_recv_ends = list(in_flight)

        sel._cleanup_recvd_ends(ends)

        assert sel.selected_ends["agg"] == set()
        assert sel.all_selected == {}
        assert sel.ordered_updates_recv_ends == []
        assert all(
            ends[e].get_property(KEY_END_STATE) == VAL_END_STATE_NONE
            for e in in_flight
        )

    def test_cleanup_provided_ends_frees_only_the_given_ends(self, build, make_ends):
        sel = build(c=4)
        ends = make_ends(count=4, prefix="t")
        sel.select(ends, _send_props(), [])
        target, *rest = sorted(sel.selected_ends["agg"])
        ends[target].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)

        sel._cleanup_provided_ends({target: ends[target]}, ends)

        assert target not in sel.selected_ends["agg"]
        assert target not in sel.all_selected
        assert set(rest) <= set(sel.all_selected)

    def test_cleanup_removed_ends_keeps_an_end_that_already_returned(
        self, build, make_ends
    ):
        sel = build(c=4)
        ends = make_ends(count=4, prefix="t")
        sel.select(ends, _send_props(), [])
        returned = sorted(sel.selected_ends["agg"])[0]
        sel.ordered_updates_recv_ends = [returned]

        sel._cleanup_removed_ends(returned)
        # its participation is complete; the drain owns it now
        assert returned in sel.all_selected

    def test_cleanup_removed_ends_frees_an_end_that_never_returned(
        self, build, make_ends
    ):
        sel = build(c=4)
        ends = make_ends(count=4, prefix="t")
        sel.select(ends, _send_props(), [])
        gone = sorted(sel.selected_ends["agg"])[0]

        sel._cleanup_removed_ends(gone)

        assert gone not in sel.all_selected
        assert gone not in sel.selected_ends["agg"]
        assert sel.track_selected_trainers_which_left[gone] == 1


class TestDisconnectedSelectionsFreeTheirSlot:
    def test_departed_end_leaves_selected_ends(self, build, make_ends):
        sel = build(c=4)
        ends = make_ends(count=4, prefix="t")
        sel.select(ends, _send_props(), [])
        gone = sorted(sel.selected_ends["agg"])[0]

        remaining = {k: v for k, v in ends.items() if k != gone}
        sel._drop_disconnected_selections(sel.selected_ends["agg"], remaining)

        assert gone not in sel.selected_ends["agg"]
        # deliberately left in all_selected: it may already have participated
        assert gone in sel.all_selected

    def test_unavailable_but_connected_end_keeps_its_slot(self, build, make_ends):
        """Challenge 13: eligibility filtering must not be mistaken for
        disconnection, or the aggregator forgets it is waiting on an
        in-flight trainer that merely went UN_AVL."""
        sel = build(c=4)
        ends = make_ends(count=4, prefix="t")
        sel.select(ends, _send_props(), [])
        in_flight = sorted(sel.selected_ends["agg"])
        for eid in in_flight:
            ends[eid].set_property(PROP_AVL_STATE, TrainerAvailState.UN_AVL.value)

        # empty eligible pool, full connected pool
        sel._handle_send_state(
            ends={}, concurrency=4,
            ctx=SelectContext(task_to_perform="train", connected_ends=ends),
        )
        assert sel.selected_ends["agg"] == set(in_flight)


class TestChoiceIsPoolSizeIndependent:
    """`_keyed_topk` ranks each id by a key derived only from itself, so a
    trainer's incidental presence can't shift anyone else's draw -- the failure
    mode of index-based `random.sample`/reservoir sampling across real vs sim."""

    def test_extra_candidate_does_not_reorder_the_others(self, build):
        sel = build()
        ctx = SelectContext(agg_version_key=(1, 0))
        small = {f"t{i}": None for i in range(8)}
        large = dict(small, extra_one=None, extra_two=None)

        ranked_small = sel._choose(small, 8, ctx)
        ranked_large = [e for e in sel._choose(large, 10, ctx) if e in small]
        assert ranked_small == ranked_large

    def test_selection_is_seed_reproducible(self, build):
        ctx = SelectContext(agg_version_key=(1, 0))
        ends = {f"t{i}": None for i in range(20)}
        assert build()._choose(ends, 5, ctx) == build()._choose(ends, 5, ctx)

    def test_version_key_advance_rerolls_the_order(self, build):
        sel = build()
        ends = {f"t{i}": None for i in range(20)}
        first = sel._choose(ends, 5, SelectContext(agg_version_key=(1, 0)))
        second = sel._choose(ends, 5, SelectContext(agg_version_key=(2, 0)))
        assert first != second


class TestSelectorsDrawIndependently:
    def test_fedbuff_and_async_random_differ_under_the_same_seed(self):
        ctx = SelectContext(agg_version_key=(1, 0))
        ends = {f"t{i}": None for i in range(30)}
        fedbuff = FedBuffSelector(_seed=7, c=4, aggGoal=2)._choose(ends, 8, ctx)
        arand = AsyncRandomSelector(_seed=7, c=4, aggGoal=2)._choose(ends, 8, ctx)
        assert fedbuff != arand
