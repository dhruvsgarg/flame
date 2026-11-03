"""FX-N127 (V3): pace_gain spends c while held-out gain per update is measurable, halves it when it is not; K stays put."""
import pytest

from flame.selector.dynamic_kc_controller import DynamicKCController
from flame.selector.dynamic_kc_policy import PaceGainPolicy, build_policy


def _rep(t, gain, n=3, du=10, noise=0.0):
    return dict(t=t, n_pair=n, du=du, d_nll=-(gain * n * du) + noise)


def _step(p, c, t, reps):
    return p.compute_new_c(c, {"now": t, "c_config": 16, "val_reports": reps})


def test_snr_cuts_without_gain_and_ramps_up_on_gain():
    p = PaceGainPolicy(tau_s=400, law="snr", c_min=2)
    assert _step(p, 16, 0, []) is None  # first call opens the clock
    flat = [_rep(t, 0.0, noise=(-1) ** t * 0.5) for t in range(1, 100)]
    assert _step(p, 16, 100, flat) == 8  # z ~ 0: halve
    assert _step(p, 8, 150, []) is None  # one decision per tau / 4
    assert _step(p, 8, 200, []) == 4 and _step(p, 4, 300, []) == 2 and _step(p, 2, 400, []) == 2  # floor c_min
    grow = [_rep(t, 0.01, noise=(-1) ** t * 0.01) for t in range(500, 600)]
    assert _step(p, 2, 600, grow) == 4 and _step(p, 4, 700, []) == 8  # measurable gain: x2 per decision, old reports age out
    assert p.state()["pace_z"] >= 2


def test_ratio_law_follows_gain_relative_to_peak():
    p = PaceGainPolicy(tau_s=400, law="ratio", c_min=2)
    _step(p, 16, 0, [])
    assert _step(p, 16, 100, [_rep(t, 0.02, noise=(-1) ** t * 0.01) for t in range(1, 100)]) == 16
    assert _step(p, 16, 600, [_rep(t, 0.005, noise=(-1) ** t * 0.01) for t in range(500, 600)]) == 4  # a quarter of the peak


def test_unpaired_reports_are_ignored_and_bad_law_rejected():
    p = PaceGainPolicy(tau_s=400)
    _step(p, 16, 0, [])
    assert _step(p, 16, 200, [dict(t=1, n_pair=0, du=5, d_nll=0.0), dict(t=2, n_pair=3, du=0, d_nll=0.0)]) is None
    with pytest.raises(ValueError):
        PaceGainPolicy(law="pid")


def test_controller_keeps_k_when_c_falls_if_asked():
    for follows, k in ((True, 4), (False, 10)):
        ctl = DynamicKCController(build_policy("noop", {}), k_init=10, c_init=15, k_min=1, k_max=100, c_min=1, c_max=15,
                                  k_follows_c=follows)
        ctl.policy.compute_new_c = lambda c, m: 4
        assert ctl.step({}) == (k, 4)
