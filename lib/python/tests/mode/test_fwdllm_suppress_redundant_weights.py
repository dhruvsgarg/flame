# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-1 (charter EXPTS_CHARTER §5c): intra-databin weight-resend suppression.

Within a data-bin the model_version is constant and the full WEIGHTS+GRAD_POOL
payload is byte-identical across iterations. The telemetry showed the aggregator
re-sending that identical payload to the same trainers every iteration (~90% of
fwdllm weight-bytes, ~71% of fluxtune) because the WEIGHTS-vs-VAR=bad guard keyed
off the return-driven `_trainer_last_model_version` map, which never marks an
actively-training trainer current -> `is_stale` stays True -> full re-send.

`_should_send_full_weights` is the SHARED decision used by both the sync and the
async distribute loops (their parity is the regression guard). These tests pin:
(1) flag OFF is byte-identical to the legacy `var_good_enough or is_stale` rule;
(2) flag ON downgrades a within-cycle repeat to VAR=bad and never a first send;
(3) a commit (`var_good_enough`) always ships weights;
(4) the full distribute-loop contract: exactly ONE weights send per trainer per
    data-bin, reverting to weights after the cycle set is cleared on a commit.
"""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _DistribAgg:
    """Minimal stand-in exposing only what `_should_send_full_weights` reads."""

    def __init__(self, suppress, var_good, sent=None):
        self._suppress_redundant_weights = suppress
        self.var_good_enough = var_good
        self._weights_sent_this_cycle = set(sent or [])

    decide = TopAggregator._should_send_full_weights


# --- (1) flag OFF == legacy (var_good_enough or is_stale) ------------------

def test_flag_off_is_legacy():
    off_varbad = _DistribAgg(suppress=False, var_good=False)
    assert off_varbad.decide("a", is_stale=True) is True   # stale -> weights
    assert off_varbad.decide("a", is_stale=False) is False  # current -> var_bad
    off_commit = _DistribAgg(suppress=False, var_good=True)
    assert off_commit.decide("a", is_stale=False) is True   # commit -> weights
    assert off_commit.decide("a", is_stale=True) is True


def test_flag_off_ignores_sent_set():
    # Even if an end is in the set, flag OFF must not suppress (byte-identical).
    off = _DistribAgg(suppress=False, var_good=False, sent={"a"})
    assert off.decide("a", is_stale=True) is True


# --- (2)/(3) flag ON semantics ---------------------------------------------

def test_commit_branch_first_send_weights_repeat_suppressed():
    # The dominant redundancy: repeated distribute calls in the SAME data-bin all
    # take the var_good_enough=True branch. First send per end -> weights; a repeat
    # within the cycle -> VAR=bad (the set is checked BEFORE the var_good branch).
    on = _DistribAgg(suppress=True, var_good=True, sent=set())
    assert on.decide("a", is_stale=False) is True     # first send -> weights
    on._weights_sent_this_cycle.add("a")              # caller records it
    assert on.decide("a", is_stale=False) is False    # repeat same cycle -> VAR=bad
    assert on.decide("a", is_stale=True) is False      # still suppressed if stale too
    assert on.decide("b", is_stale=False) is True      # a NEW end still gets weights


def test_flag_off_commit_always_weights():
    # With the flag OFF the set is never consulted -> legacy commit-branch behavior.
    off = _DistribAgg(suppress=False, var_good=True, sent={"a"})
    assert off.decide("a", is_stale=False) is True
    assert off.decide("a", is_stale=True) is True


def test_first_send_is_weights_repeat_is_varbad():
    on = _DistribAgg(suppress=True, var_good=False, sent=set())
    # first time this cycle, stale -> weights (nothing in the set yet)
    assert on.decide("a", is_stale=True) is True
    on._weights_sent_this_cycle.add("a")            # caller records the send
    # re-dispatched same cycle, still stale by the return-map -> VAR=bad now
    assert on.decide("a", is_stale=True) is False
    # a different, not-yet-sent end still gets weights
    assert on.decide("b", is_stale=True) is True


def test_current_by_return_map_is_varbad():
    # not stale and not a commit -> var_bad regardless of flag
    on = _DistribAgg(suppress=True, var_good=False, sent=set())
    assert on.decide("a", is_stale=False) is False


# --- (4) full distribute-loop contract: exactly one weights send / bin -----

def _run_cycle(agg, dispatches):
    """Mimic the distribute loop's set-management for a sequence of dispatches
    within ONE data-bin. Returns the list of 'weights'/'var_bad' decisions."""
    kinds = []
    for end, is_stale in dispatches:
        if agg.decide(end, is_stale):
            kinds.append("weights")
            if agg._suppress_redundant_weights:
                agg._weights_sent_this_cycle.add(end)
        else:
            kinds.append("var_bad")
    return kinds


def test_exactly_one_weights_per_trainer_per_databin():
    # fwdllm pattern: same K=3 trainers, 4 iterations, always is_stale by the map.
    agg = _DistribAgg(suppress=True, var_good=False, sent=set())
    dispatches = [(e, True) for _ in range(4) for e in ("a", "b", "c")]
    kinds = _run_cycle(agg, dispatches)
    # first 3 (one per trainer) are weights; the remaining 9 are var_bad.
    assert kinds[:3] == ["weights", "weights", "weights"]
    assert set(kinds[3:]) == {"var_bad"}
    assert kinds.count("weights") == 3          # exactly once per unique trainer
    # legacy (flag off) would have sent 12 weights -> 9 redundant.
    off = _DistribAgg(suppress=False, var_good=False, sent=set())
    assert _run_cycle(off, dispatches).count("weights") == 12


def test_exactly_one_weights_per_trainer_var_good_branch():
    # The fwdllm pattern: sync loop distribute>>aggregate re-runs ~10x per data-bin,
    # each distribute taking the var_good_enough=True branch for the SAME K=3
    # trainers. Legacy shipped weights every time (10x redundant); the fix ships
    # each trainer the model exactly once per data-bin.
    agg = _DistribAgg(suppress=True, var_good=True, sent=set())
    dispatches = [(e, False) for _ in range(10) for e in ("a", "b", "c")]
    kinds = _run_cycle(agg, dispatches)
    assert kinds[:3] == ["weights", "weights", "weights"]
    assert set(kinds[3:]) == {"var_bad"}
    assert kinds.count("weights") == 3
    off = _DistribAgg(suppress=False, var_good=True, sent=set())
    assert _run_cycle(off, dispatches).count("weights") == 30   # legacy: all redundant


def test_weights_resume_after_cycle_cleared_on_commit():
    agg = _DistribAgg(suppress=True, var_good=False, sent=set())
    _run_cycle(agg, [("a", True)])
    assert agg.decide("a", is_stale=True) is False       # suppressed mid-bin
    agg._weights_sent_this_cycle.clear()                 # model_version advanced
    assert agg.decide("a", is_stale=True) is True        # new bin -> weights again
