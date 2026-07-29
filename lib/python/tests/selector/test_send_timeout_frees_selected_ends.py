# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression: the 90s SEND_TIMEOUT_WAIT_S abandon in async_oort.py/fedbuff.py
must free the end from selected_ends, not just all_selected -- recv_ends
derives from selected_ends, so leaving it there hangs recv_fifo forever
(UNAVAILABILITY_DESIGN.md, Open A)."""

import time

import pytest

from flame.selector.async_base import SelectContext


def _oort_ctx(**kw):
    kw.setdefault("task_to_perform", "train")
    kw.setdefault("channel_props", {"round": 1})
    kw.setdefault("trainer_unavail_list", [])
    kw.setdefault("agg_version_key", (1, 0, 0))
    kw.setdefault("trainer_version_keys", {})
    return SelectContext(**kw)


class _StopAfterAbandon(Exception):
    """Raised from a stubbed pacer() to inspect state right after the abandon
    block runs, without driving async_oort's heavier downstream selection
    (same technique as TestCoolingHoldsConcurrency in test_async_sim_ordering.py)."""


def _boom(*args, **kwargs):
    raise _StopAfterAbandon()


class TestAsyncOortSendTimeoutFreesSelectedEnds:
    @staticmethod
    def _stub_selector():
        from flame.selector.async_oort import AsyncOortSelector

        sel = AsyncOortSelector.__new__(AsyncOortSelector)
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"stale"}}
        sel.all_selected = {"stale": time.time() - 100}  # past 90s SEND_TIMEOUT_WAIT_S
        sel.ordered_updates_recv_ends = []
        sel.track_trainer_timeouts = {}
        sel._last_pacer_round = None
        sel.pacer = _boom
        return sel

    def test_stale_end_leaves_both_all_selected_and_selected_ends(self, make_ends):
        sel = self._stub_selector()
        ends = make_ends(["stale", "fresh"])

        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(ends, 2, _oort_ctx())

        assert "stale" not in sel.all_selected
        assert "stale" not in sel.selected_ends["agg"]

    def test_fresh_end_untouched(self, make_ends):
        sel = self._stub_selector()
        sel.all_selected["fresh"] = time.time()  # well within the 90s window
        sel.selected_ends["agg"].add("fresh")
        ends = make_ends(["stale", "fresh", "third"])

        with pytest.raises(_StopAfterAbandon):
            # 2 already selected -> extra=1, past the extra==0 short-circuit
            sel._handle_send_state(ends, 3, _oort_ctx())

        assert "fresh" in sel.all_selected
        assert "fresh" in sel.selected_ends["agg"]


class TestAsyncOortSendTimeoutDropsPendingCommitRef:
    """Once `_sim_hold_busy_slots` folds `_agg_pending_commit_ref` into its own
    reconciliation, a trainer abandoned here but left in that set would stay
    permanently un-re-pickable -- the SEND_TIMEOUT_WAIT_S reclaim would
    appear to work (all_selected/selected_ends clear) while the aggregator's
    virtual in-flight set quietly keeps the slot occupied forever."""

    @staticmethod
    def _stub_selector():
        from flame.selector.async_oort import AsyncOortSelector

        sel = AsyncOortSelector.__new__(AsyncOortSelector)
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"stale"}}
        sel.all_selected = {"stale": time.time() - 100}  # past 90s SEND_TIMEOUT_WAIT_S
        sel.ordered_updates_recv_ends = []
        sel.track_trainer_timeouts = {}
        sel._last_pacer_round = None
        sel.pacer = _boom
        sel._agg_pending_commit_ref = {"stale", "other"}
        return sel

    def test_abandoned_end_dropped_from_pending_commit_ref(self, make_ends):
        sel = self._stub_selector()
        ends = make_ends(["stale", "fresh"])

        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(ends, 2, _oort_ctx())

        assert "stale" not in sel._agg_pending_commit_ref
        assert "other" in sel._agg_pending_commit_ref  # unrelated entry untouched

    def test_noop_when_no_pending_commit_ref_bound(self, make_ends):
        # Real fwdllm binds this ref via _sim_hold_busy_slots; other selectors
        # (felix/async_cifar10) never set the attribute at all.
        sel = self._stub_selector()
        del sel._agg_pending_commit_ref
        ends = make_ends(["stale", "fresh"])

        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(ends, 2, _oort_ctx())

        assert "stale" not in sel.all_selected


class TestAsyncOortSendTimeoutIsConfigurable:
    """The bare 90s constant evicted a genuinely-busy fwdllm trainer as
    abandoned. `send_timeout_wait_s` is now configurable (getattr fallback to
    the original constant, so existing baselines are unaffected)."""

    @staticmethod
    def _stub_selector(send_timeout_wait_s):
        from flame.selector.async_oort import AsyncOortSelector

        sel = AsyncOortSelector.__new__(AsyncOortSelector)
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"slow"}}
        # 100s since dispatch: past the 90s default, within a 300s budget.
        sel.all_selected = {"slow": time.time() - 100}
        sel.ordered_updates_recv_ends = []
        sel.track_trainer_timeouts = {}
        sel.send_timeout_wait_s = send_timeout_wait_s
        sel._last_pacer_round = None
        sel.pacer = _boom
        return sel

    def test_configured_timeout_holds_a_trainer_the_default_would_evict(
        self, make_ends
    ):
        sel = self._stub_selector(send_timeout_wait_s=300)
        ends = make_ends(["slow"])

        # concurrency=2 with 1 already selected -> extra=1 regardless of
        # whether "slow" gets evicted, so execution reaches pacer() either way
        # (mirrors test_fresh_end_untouched's not-evicted case above).
        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(ends, 2, _oort_ctx())

        assert "slow" in sel.all_selected
        assert "slow" in sel.selected_ends["agg"]

    def test_default_90s_still_evicts_when_unconfigured(self, make_ends):
        sel = self._stub_selector(send_timeout_wait_s=90)
        ends = make_ends(["slow"])

        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(ends, 1, _oort_ctx())

        assert "slow" not in sel.all_selected
        assert "slow" not in sel.selected_ends["agg"]


class TestFedBuffSendTimeoutFreesSelectedEnds:
    @staticmethod
    def _stub_selector():
        from flame.selector.fedbuff import FedBuffSelector

        sel = FedBuffSelector(c=2, aggGoal=1)
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"stale"}}
        sel.all_selected = {"stale": time.time() - 100}  # past 90s SEND_TIMEOUT_WAIT_S
        return sel

    def test_stale_end_leaves_both_all_selected_and_selected_ends(self, make_ends):
        sel = self._stub_selector()
        ends = make_ends(["stale", "fresh"])

        # Inspect state right after the abandon block, before the freed slot
        # can be re-filled in the same call -- FedBuff now shares async_oort's
        # ordering (reclaim BEFORE `extra` is computed), so a reclaimed end is
        # immediately re-pickable. `_pre_choose` is the first hook past the
        # reclaim, so it stands in for async_oort's stubbed pacer above.
        from flame.selector.async_base import SelectContext

        sel._pre_choose = _boom
        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(
                ends=ends,
                concurrency=1,
                ctx=SelectContext(task_to_perform="train", connected_ends=ends),
            )

        assert "stale" not in sel.all_selected
        assert "stale" not in sel.selected_ends["agg"]
