# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression test: FedBuffSelector was missing `_cleanup_provided_ends`,
which channel.py's `cleanup_provided_ends()` calls on every selector
unconditionally. Every other async selector (async_random, async_oort)
implements it; FedBuffSelector never did -- invisible until the first real
run pairing it with fwdllm's aggregator (the fedbuff_round/fedbuff_it_unaware/
fedbuff_it_oracular smoke tests, 2026-07-23), which crashed immediately on
the first received gradient:
    AttributeError: 'FedBuffSelector' object has no attribute
    '_cleanup_provided_ends'
Same bug CLASS as the AsyncOortSelector/_cleanup_recvd_end crash covered in
test_fwdllm_duplicate_contribution.py -- selector interface methods aren't
enforced by a common base class, so a missing one only surfaces at runtime.
"""

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_SEND,
)
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD
from flame.selector.fedbuff import FedBuffSelector


def test_cleanup_provided_ends_exists_and_resets_state(make_ends):
    """Direct regression for the AttributeError: the method must exist and
    must actually free the end (state reset + removed from selected_ends/
    all_selected) so it becomes eligible for resampling."""
    sel = FedBuffSelector(_seed=1, c=4, aggGoal=2)
    ends = make_ends(count=10, prefix="t")
    cp = {
        KEY_CH_STATE: VAL_CH_STATE_SEND,
        KEY_CH_SELECT_REQUESTER: "agg",
        "round": 1,
    }
    selected = sel.select(ends, cp, trainer_unavail_list=[])
    assert selected, "fixture selection produced nothing to clean up"

    end_id = next(iter(selected))
    ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
    assert end_id in sel.all_selected
    assert end_id in sel.selected_ends[sel.requester]

    # This call raised AttributeError before the fix.
    sel._cleanup_provided_ends({end_id: ends[end_id]}, ends)

    assert ends[end_id].get_property(KEY_END_STATE) == VAL_END_STATE_NONE
    assert end_id not in sel.selected_ends[sel.requester]
    assert end_id not in sel.all_selected


def test_cleanup_provided_ends_only_touches_the_given_ends(make_ends):
    """Ends not in the cleanup batch must be left alone."""
    sel = FedBuffSelector(_seed=1, c=4, aggGoal=2)
    ends = make_ends(count=10, prefix="t")
    cp = {
        KEY_CH_STATE: VAL_CH_STATE_SEND,
        KEY_CH_SELECT_REQUESTER: "agg",
        "round": 1,
    }
    selected = sel.select(ends, cp, trainer_unavail_list=[])
    assert len(selected) >= 2, "need at least 2 selected ends for this test"

    ids = list(selected)
    cleanup_id, untouched_id = ids[0], ids[1]
    ends[cleanup_id].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)

    sel._cleanup_provided_ends({cleanup_id: ends[cleanup_id]}, ends)

    assert untouched_id in sel.all_selected
    assert untouched_id in sel.selected_ends[sel.requester]
