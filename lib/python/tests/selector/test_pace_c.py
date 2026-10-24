"""F42d: pace_c lowers c when federated-validation loss rises at flat accuracy, restores it on progress; K <= c follows."""
from flame.selector.dynamic_kc_controller import DynamicKCController
from flame.selector.dynamic_kc_policy import PaceCPolicy, build_policy


def _win(p, c, t, loss, acc, n=300):
    return p.compute_new_c(c, {"now": t, "c_config": 16, "val_probes": [(t, loss * n, int(acc * n), n)]})


def test_pace_c_throttles_on_overfit_and_recovers():
    p = PaceCPolicy(window_s=100, min_n=100, rho=0.1, beta=0.5, c_floor=0.25, delta=0.01)
    assert _win(p, 16, 0, 2.0, 0.30) is None  # window opens
    assert _win(p, 16, 100, 1.8, 0.40) == 16  # progress: stays at c_config
    assert _win(p, 16, 200, 1.7, 0.45) == 16
    assert _win(p, 16, 300, 2.4, 0.45) == 16  # loss up, but accuracy still past its best: keep c
    c = _win(p, 16, 400, 2.6, 0.45)
    assert c == 8  # loss well above its min, accuracy flat
    assert _win(p, 8, 500, 2.8, 0.45) == 4 and _win(p, 4, 600, 3.0, 0.45) == 4  # floor c/4
    assert _win(p, 4, 700, 1.0, 0.47) == 4  # 2-window smoothing: one good window is not yet a recovery
    assert _win(p, 4, 800, 1.0, 0.50) == 8 and _win(p, 8, 900, 1.0, 0.52) == 16  # new data → progress → back up


def test_pace_c_waits_for_enough_probe_samples():
    p = PaceCPolicy(window_s=10, min_n=500)
    assert _win(p, 16, 0, 2.0, 0.3) is None and _win(p, 16, 20, 2.0, 0.3, n=100) is None


def test_controller_keeps_k_below_c():
    ctl = DynamicKCController(build_policy("noop", {}), k_init=10, c_init=15, k_min=1, k_max=100, c_min=1, c_max=15)
    ctl.policy.compute_new_c = lambda c, m: 4
    assert ctl.step({}) == (4, 4)
