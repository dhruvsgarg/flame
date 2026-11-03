"""FX-N82: REFL `adapt_selection` (aggregator.py:640-648): fewer new picks when stale updates land next round."""
from datetime import timedelta
from types import SimpleNamespace

import pytest

from flame.mode.horizontal.oort.top_aggregator import TopAggregator
from flame.selector.oort import PROP_CLIENT_TASK_TRAIN_DURATION, PROP_ROUND_START_TIME
from flame.selector.properties import PROP_SIM_SEND_TS
from flame.selector.refl_oort import REFLOortSelector


class _Agg(TopAggregator):
    check_and_sleep = evaluate = initialize = load_data = train = None


class _Chan:
    def __init__(self, sel, props):
        self._selector, self._props = sel, props
        self._ends = {e: None for e in props}

    def get_end_property(self, end, key):
        return self._props.get(end, {}).get(key)


def _agg(round_num, now, stale_max=-1):
    a = _Agg.__new__(_Agg)
    a.simulated, a._round = True, round_num
    a._vclock = SimpleNamespace(now=now)
    a.optimizer = SimpleNamespace(stale_update_max=stale_max)
    return a


def _end(version, sent, dur=None):
    p = {PROP_ROUND_START_TIME: (version, None), PROP_SIM_SEND_TS: sent}
    if dur is not None:
        p[PROP_CLIENT_TASK_TRAIN_DURATION] = timedelta(seconds=dur)
    return p


def test_formula_matches_reference():
    sel = REFLOortSelector(aggr_num=10, adapt_selection=1)
    assert sel.num_of_ends == 13
    assert sel.adapt_num_to_sample(13, 0) == 13
    assert sel.adapt_num_to_sample(13, 3) == 10
    assert sel.adapt_num_to_sample(13, 9) == 6  # cap 0.5 binds
    sel.adapt_selection_cap = 0
    assert sel.adapt_num_to_sample(13, 20) == 1
    assert REFLOortSelector(aggr_num=10).adapt_num_to_sample(13, 3) == 13  # off by default


def test_overcommit_rounds_like_reference():
    assert REFLOortSelector(aggr_num=15).num_of_ends == 20  # round(19.5); Oort's int() gives 19


def test_hard_mode_rejected():
    with pytest.raises(ValueError):
        REFLOortSelector(aggr_num=10, adapt_selection=2)


def test_stale_due_counts_landings_within_window():
    sel = REFLOortSelector(aggr_num=10, adapt_selection=1)
    props = {"fresh": _end(5, 100.0, 5), "due": _end(4, 90.0, 20), "late": _end(4, 90.0, 100),
             "unknown": _end(3, 80.0), "old": _end(1, 95.0, 10)}
    sel.selected_ends = set(props)
    a = _agg(5, now=100.0, stale_max=3)
    a._adapt_round_len = 15.0
    # due lands at 110 <= 115; late at 190; unknown takes the median (20) -> 100; old exceeds stale_update 3
    assert a._stale_due(_Chan(sel, props)) == 2


def test_picks_decided_once_per_version_and_k_follows():
    sel = REFLOortSelector(aggr_num=10, adapt_selection=1)
    props = {f"s{i}": _end(1, 0.0, 10) for i in range(4)}
    sel.selected_ends = set(props)
    ch = _Chan(sel, props)
    a = _agg(2, now=5.0)
    assert a._adapt_picks(ch, 13) == 13 and a._version_k(10) == 10  # no round length yet
    a._round, a._adapt_round_len = 3, 10.0
    assert a._adapt_picks(ch, 13) == 9 and a._version_k(10) == 9
    sel.selected_ends = set()
    assert a._adapt_picks(ch, 13) == 9  # top-up within the version keeps its decision
    a._round = 4
    assert a._version_k(10) == 10


def test_round_length_moving_average():
    sel = REFLOortSelector(aggr_num=10, adapt_selection=1)
    ch = _Chan(sel, {})
    a = _agg(1, now=0.0)
    a._adapt_picks(ch, 13)
    a._vclock.now = 40.0
    a._adapt_note_commit(ch)
    assert a._adapt_round_len == 40.0
    a._vclock.now = 60.0
    a._adapt_note_commit(ch)
    assert a._adapt_round_len == pytest.approx(0.75 * 20 + 0.25 * 40)
