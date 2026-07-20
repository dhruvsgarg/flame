# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-1: intra-databin weight-resend suppression.

Within a data-bin the model_version is constant and the full WEIGHTS+GRAD_POOL
payload is byte-identical across iterations, yet the aggregator used to re-send
it to the same trainers every iteration because the WEIGHTS-vs-VAR=bad guard
keyed off the return-driven `_trainer_last_model_version` map, which never marks
an actively-training trainer current -> `is_stale` stays True -> full re-send.

Landed as a config flag (simulate_fwdllm.md §G, 07-19), then validated 0%
redundant weight-sends on a live pair and promoted to unconditional aggregator
behavior (§G, 07-19 pm) -- a trainer that already has the current
version_key/model_version's weights is never re-sent them, full stop, as an
invariant of the version-tracking logic, not an opt-in.

`_should_send_full_weights` is the SHARED decision used by both the sync and the
async distribute loops (their parity is the regression guard). These tests pin:
(1) a within-cycle repeat is always downgraded to VAR=bad, never a first send;
(2) a commit (`var_good_enough`) always ships weights the first time;
(3) the full distribute-loop contract: exactly ONE weights send per trainer per
    data-bin, reverting to weights after the cycle set is cleared on a commit;
(4) `_warn_if_redundant_weights_resend` is the regression tripwire for a future
    call site that bypasses `_should_send_full_weights`.
"""

import logging

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _DistribAgg:
    """Minimal stand-in exposing only what `_should_send_full_weights` /
    `_warn_if_redundant_weights_resend` read."""

    def __init__(self, var_good, sent=None, model_version=1):
        self.var_good_enough = var_good
        self._weights_sent_this_cycle = set(sent or [])
        self._model_version = model_version

    decide = TopAggregator._should_send_full_weights
    warn_if_resend = TopAggregator._warn_if_redundant_weights_resend


# --- (1)/(2) core decision ---------------------------------------------------

def test_commit_branch_first_send_weights_repeat_suppressed():
    # The dominant redundancy: repeated distribute calls in the SAME data-bin all
    # take the var_good_enough=True branch. First send per end -> weights; a repeat
    # within the cycle -> VAR=bad (the set is checked BEFORE the var_good branch).
    agg = _DistribAgg(var_good=True, sent=set())
    assert agg.decide("a", is_stale=False) is True     # first send -> weights
    agg._weights_sent_this_cycle.add("a")              # caller records it
    assert agg.decide("a", is_stale=False) is False    # repeat same cycle -> VAR=bad
    assert agg.decide("a", is_stale=True) is False      # still suppressed if stale too
    assert agg.decide("b", is_stale=False) is True      # a NEW end still gets weights


def test_first_send_is_weights_repeat_is_varbad():
    agg = _DistribAgg(var_good=False, sent=set())
    # first time this cycle, stale -> weights (nothing in the set yet)
    assert agg.decide("a", is_stale=True) is True
    agg._weights_sent_this_cycle.add("a")            # caller records the send
    # re-dispatched same cycle, still stale by the return-map -> VAR=bad now
    assert agg.decide("a", is_stale=True) is False
    # a different, not-yet-sent end still gets weights
    assert agg.decide("b", is_stale=True) is True


def test_current_by_return_map_is_varbad():
    # not stale and not a commit -> var_bad
    agg = _DistribAgg(var_good=False, sent=set())
    assert agg.decide("a", is_stale=False) is False


# --- (3) full distribute-loop contract: exactly one weights send / bin -----

def _run_cycle(agg, dispatches):
    """Mimic the distribute loop's set-management for a sequence of dispatches
    within ONE data-bin. Returns the list of 'weights'/'var_bad' decisions."""
    kinds = []
    for end, is_stale in dispatches:
        if agg.decide(end, is_stale):
            kinds.append("weights")
            agg._weights_sent_this_cycle.add(end)
        else:
            kinds.append("var_bad")
    return kinds


def test_exactly_one_weights_per_trainer_per_databin():
    # fwdllm pattern: same K=3 trainers, 4 iterations, always is_stale by the map.
    agg = _DistribAgg(var_good=False, sent=set())
    dispatches = [(e, True) for _ in range(4) for e in ("a", "b", "c")]
    kinds = _run_cycle(agg, dispatches)
    # first 3 (one per trainer) are weights; the remaining 9 are var_bad.
    assert kinds[:3] == ["weights", "weights", "weights"]
    assert set(kinds[3:]) == {"var_bad"}
    assert kinds.count("weights") == 3          # exactly once per unique trainer


def test_exactly_one_weights_per_trainer_var_good_branch():
    # The fwdllm pattern: sync loop distribute>>aggregate re-runs ~10x per data-bin,
    # each distribute taking the var_good_enough=True branch for the SAME K=3
    # trainers. Each trainer gets the model exactly once per data-bin.
    agg = _DistribAgg(var_good=True, sent=set())
    dispatches = [(e, False) for _ in range(10) for e in ("a", "b", "c")]
    kinds = _run_cycle(agg, dispatches)
    assert kinds[:3] == ["weights", "weights", "weights"]
    assert set(kinds[3:]) == {"var_bad"}
    assert kinds.count("weights") == 3


def test_weights_resume_after_cycle_cleared_on_commit():
    agg = _DistribAgg(var_good=False, sent=set())
    _run_cycle(agg, [("a", True)])
    assert agg.decide("a", is_stale=True) is False       # suppressed mid-bin
    agg._weights_sent_this_cycle.clear()                 # model_version advanced
    assert agg.decide("a", is_stale=True) is True        # new bin -> weights again


# --- (4) regression tripwire -------------------------------------------------

def test_warn_if_redundant_weights_resend_is_silent_on_the_happy_path(caplog):
    # `_should_send_full_weights` gates this in every real call site, so the
    # warning must NOT fire for an end that hasn't been sent yet.
    agg = _DistribAgg(var_good=True, sent=set())
    with caplog.at_level(logging.WARNING):
        agg.warn_if_resend("a")
    assert caplog.records == []


def test_warn_if_redundant_weights_resend_fires_on_violation(caplog):
    # Simulates a future call site bypassing `_should_send_full_weights`: an end
    # already in `_weights_sent_this_cycle` must produce a WARNING, not silence.
    agg = _DistribAgg(var_good=True, sent={"a"})
    with caplog.at_level(logging.WARNING):
        agg.warn_if_resend("a")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "INVARIANT VIOLATION" in caplog.records[0].message
