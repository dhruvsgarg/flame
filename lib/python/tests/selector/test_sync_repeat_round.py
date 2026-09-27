# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N31: a sync round that did not advance (FX-D10) re-dispatches afresh, never to ends
already tasked at that version (FX-D9); same-round RECV lookups keep the round cache."""

import pytest

from flame.channel import KEY_CH_STATE, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND


def _oort():
    from flame.selector.oort import OortSelector
    return OortSelector(aggr_num=2)


def _refl_oort():
    from flame.selector.refl_oort import REFLOortSelector
    return REFLOortSelector(aggr_num=2, avail_priority=0)


def _feddance():
    from flame.selector.feddance import FedDanceSelector
    return FedDanceSelector(aggr_num=2)


@pytest.fixture(params=[_oort, _refl_oort, _feddance], ids=["oort", "refl_oort", "feddance"])
def selector(request):
    return request.param()


def _dispatch(sel, ends, rnd, tasked):
    props = {"round": rnd, "cur_time": 0.0, KEY_CH_STATE: VAL_CH_STATE_SEND}
    return set(sel.select(ends, props, trainer_unavail_list=[], task_to_perform="train",
                          agg_version_key=(rnd, 0),
                          trainer_version_keys={e: (rnd, 0) for e in tasked}))


def test_repeat_round_dispatch_skips_tasked_ends(selector, make_ends):
    ends = make_ends(count=10, prefix="t")
    first = _dispatch(selector, ends, 5, set())
    assert first
    selector.selected_ends.clear()  # withhold/abandon freed every slot; nothing committed
    again = _dispatch(selector, ends, 5, first)
    assert again and not (again & first)


def test_same_round_recv_keeps_cache(selector, make_ends):
    ends = make_ends(count=10, prefix="t")
    first = _dispatch(selector, ends, 5, set())
    props = {"round": 5, "cur_time": 0.0, KEY_CH_STATE: VAL_CH_STATE_RECV}
    assert set(selector.select(ends, props, trainer_unavail_list=[],
                               task_to_perform="train")) == first
