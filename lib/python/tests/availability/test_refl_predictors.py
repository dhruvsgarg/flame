"""F42b: REFL* (trace oracle) and REFL-fc (seasonal on-device forecaster) answer REFL's is_available on the leg's clock."""
from flame.availability import refl_predictors as rp

DAY = rp.DAY_S


def _pred(cls, ev, off=0.0):
    p = cls.__new__(cls)
    rp.REFLAvailabilityTracker.__init__(p)
    p._ev = {"c": ([t for t, _ in ev], [s for _, s in ev])}
    p._off, p._scale = off, 1.0
    return p


def test_avl_train_share():
    t, s = [0.0, 100.0], ["AVL_TRAIN", "UN_AVL"]
    assert rp._avl_train_share(t, s, 50, 150) == 0.5
    assert rp._avl_train_share(t, s, 0, 100) == 1.0


def test_oracle_reads_the_future_window():
    ev = [(0.0, "AVL_TRAIN"), (DAY + 600, "UN_AVL"), (DAY + 1200, "AVL_TRAIN")]
    o = _pred(rp.TraceOracle, ev, off=DAY)
    assert o.is_available("c", 0, 600, 1) and not o.is_available("c", 600, 600, 1)
    assert not o.is_available("c", 0, 600, 2)  # second slot [600, 1200) is UN_AVL


def test_forecaster_uses_only_past_days():
    # day 0 at 600-1200 s: UN_AVL; day 1 the same window is AVL_TRAIN (the forecaster must not see it)
    ev = [(0.0, "AVL_TRAIN"), (600, "UN_AVL"), (1200, "AVL_TRAIN")]
    f = _pred(rp.SeasonalForecaster, ev, off=DAY)
    assert not f.is_available("c", 600, 600, 1)  # yesterday's same window was off
    assert f.is_available("c", 3600, 600, 1)
    assert _pred(rp.SeasonalForecaster, ev).is_available("c", 600, 600, 1)  # day 0: no history → available


def test_prophet_learns_the_daily_pattern_from_past_days_only():
    # AVL_TRAIN 09:00-17:00 every day for 3 days; asked on day 3 (history = days 0-2)
    ev = [e for d in range(4) for e in ((d * DAY + 9 * 3600, "AVL_TRAIN"), (d * DAY + 17 * 3600, "UN_AVL"))]
    f = _pred(rp.ProphetForecaster, [(0.0, "UN_AVL")] + ev)
    f.step_s, f._fit, f.fits = 900.0, {}, 0
    noon, night = 3 * DAY + 12 * 3600, 3 * DAY + 2 * 3600
    assert f.p_trace("c", noon, noon, noon + 1800) > 0.7
    assert f.p_trace("c", noon, night, night + 1800) < 0.3
    assert f.fits == 1  # one fit per trace day
    assert f.p_trace("c", 3600.0, 3600.0, 7200.0) is None  # day 0: no full day of history
