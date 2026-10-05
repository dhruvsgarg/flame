"""FX-D53: a REFL stale update aggregates but does not fill the round's K; oort (no stale accepted) is unchanged."""
from types import SimpleNamespace

from flame.mode.horizontal.oort.top_aggregator import TopAggregator


class _Agg(TopAggregator):
    check_and_sleep = evaluate = initialize = load_data = train = None


def _agg(stale_max, knob=None):
    a = _Agg.__new__(_Agg)
    a.optimizer = SimpleNamespace(stale_update_max=stale_max) if stale_max is not None else SimpleNamespace()
    hp = SimpleNamespace() if knob is None else SimpleNamespace(refl_fresh_k=knob)
    a.config = SimpleNamespace(hyperparameters=hp)
    return a


def test_refl_counts_fresh_only():
    a = _agg(5)
    assert a._counts_toward_k(0) and not a._counts_toward_k(2)


def test_refl_knob_off_counts_stale():
    assert _agg(5, knob="false")._counts_toward_k(2)


def test_non_refl_counts_everything_accepted():
    assert _agg(None)._counts_toward_k(1)
