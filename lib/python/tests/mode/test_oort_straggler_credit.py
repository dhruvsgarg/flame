"""FX-N83: a dropped overcommitted straggler scores the mean utility of the version it missed (Oort
param_server.py:347-352); REFL with stale updates keeps its own utility."""
from types import SimpleNamespace

from flame.mode.horizontal.oort.top_aggregator import TopAggregator
from flame.selector.oort import PROP_STAT_UTILITY


class _Agg(TopAggregator):
    check_and_sleep = evaluate = initialize = load_data = train = None


class _Chan:
    def __init__(self):
        self.props = {}

    def set_end_property(self, end, key, value):
        self.props[(end, key)] = value


def _agg(optimizer):
    a = _Agg.__new__(_Agg)
    a.optimizer, a._version_mean_util = optimizer, {4: 2.5}
    return a


def test_oort_credits_version_mean():
    ch = _Chan()
    _agg(SimpleNamespace())._credit_dropped_straggler(ch, "t1", 4)
    assert ch.props[("t1", PROP_STAT_UTILITY)] == 2.5


def test_unknown_version_keeps_own_utility():
    ch = _Chan()
    _agg(SimpleNamespace())._credit_dropped_straggler(ch, "t1", 3)
    assert not ch.props


def test_refl_with_stale_updates_keeps_own_utility():
    ch = _Chan()
    _agg(SimpleNamespace(stale_update_max=-1))._credit_dropped_straggler(ch, "t1", 4)
    assert not ch.props
