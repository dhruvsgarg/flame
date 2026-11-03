# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N109 I5: update statistics and the backprop dynamic K / c policies."""

import pytest
import torch

from flame.selector.dynamic_kc_controller import DynamicKCController
from flame.selector.dynamic_kc_policy import build_policy
from flame.selector.update_stats import UpdateStats, cos_split, flatten


def test_noise_scale_recovers_tr_sigma_over_g2():
    torch.manual_seed(0)
    d, n, sigma = 2000, 400, 0.5
    g = torch.ones(d) * 0.1  # |G|^2 = 20, tr(Σ) = d σ² = 500 -> B = 25
    st = UpdateStats(keep_vectors=False)
    for _ in range(n):
        st.add(g + sigma * torch.randn(d))
    out = st.close()
    assert out["n"] == n
    assert out["noise_scale"] == pytest.approx(25, rel=0.15)
    assert st.close() == {"n": 0}  # close resets


def test_cos_split_stale_half_points_elsewhere():
    st = UpdateStats()
    for t in (0, 0):
        st.add(torch.tensor([1.0, 0.0]), staleness=t)
    for t in (5, 5):
        st.add(torch.tensor([0.0, 1.0]) * 0.1, staleness=t)
    assert cos_split(st.close()["cos_by_staleness"]) < 0


def test_flatten_skips_keys_and_ints():
    w = {"a": torch.ones(2), "bn.running_mean": torch.ones(3), "n": torch.tensor(4)}
    assert flatten(w, lambda k: "running" in k).numel() == 2


def test_schedule_interpolates_and_holds():
    p = build_policy("schedule", {"k_points": [[0, 3], [1000, 10]], "c_points": [[0, 30], [1000, 15]]})
    assert p.compute_new_k(5, {"n_updates": 500}) == 6  # 6.5 rounds to even
    assert p.compute_new_k(5, {"n_updates": 5000}) == 10
    assert p.compute_new_c(15, {"n_updates": 0}) == 30


def test_availability_cap_keeps_choice():
    p = build_policy("availability_cap", {"rho": 0.5})
    assert p.compute_new_c(15, {"n_eligible_train": 75, "c_config": 15}) == 15
    assert p.compute_new_c(15, {"n_eligible_train": 12, "c_config": 15}) == 6


def test_controller_caps_k_at_c_and_c_at_config():
    ctl = DynamicKCController(build_policy("schedule", {"k_points": [[0, 20]], "c_points": [[0, 40]]}),
                              k_init=10, c_init=15, k_min=1, k_max=100, c_min=1, c_max=15)
    assert ctl.step({"n_updates": 0}) == (15, 15)


def test_aimd_k_steps_down_while_loss_holds_and_doubles_on_rise():
    p = build_policy("aimd_k", {})
    flat = [1.0, 1.1, 0.9, 1.0]
    assert p.compute_new_k(10, {"cycle_losses": flat, "n_trainers": 4}) is None  # first window: no reference yet
    assert p.compute_new_k(10, {"cycle_losses": flat, "n_trainers": 4}) == 9  # no rise: larger step
    assert p.compute_new_k(9, {"cycle_losses": [3.0, 3.1], "n_trainers": 4}) is None  # window still filling
    assert p.compute_new_k(9, {"cycle_losses": [3.0, 3.2], "n_trainers": 4}) == 18  # rise past 2 SE: back off


def test_aimd_k_memory_holds_twice_the_unstable_k():
    p = build_policy("aimd_k", {"memory": True})  # FX-N121
    p.compute_new_k(4, {"cycle_losses": [1.0, 1.1, 0.9, 1.0], "n_trainers": 4})
    assert p.compute_new_k(4, {"cycle_losses": [3.0, 3.1, 2.9, 3.0], "n_trainers": 4}) == 8  # rise at K 4
    assert p.compute_new_k(8, {"cycle_losses": [2.0, 2.1, 1.9, 2.0], "n_trainers": 4}) == 8  # no rise, but floor 2 x 4


def test_aimd_k_stall_pairs_each_client_with_itself():
    p = build_policy("aimd_k", {"stall": True, "memory": True})  # Q-AIMD
    hard = {"a": 3.0, "b": 1.0, "c": 2.0, "d": 0.5}  # 6x spread across clients: only paired differences see progress
    m = lambda d: {"cycle_client_losses": [(e, q + d + 0.01 * i) for i, (e, q) in enumerate(hard.items())], "n_trainers": 8}
    assert p.compute_new_k(10, m(0.0)) is None  # first contact: no pairs yet
    assert p.compute_new_k(10, m(-0.2)) == 9  # every client improved: larger step
    assert p.compute_new_k(9, {"cycle_client_losses": [(e, q - 0.2 + x) for (e, q), x in
                                                       zip(hard.items(), (0.01, -0.01, 0.02, -0.02))], "n_trainers": 8}) == 10  # flat
    assert p.compute_new_k(10, m(0.3)) == 20 and p.floor == 20  # rise: back off, remember


def test_felix_c_scales_by_availability_and_phase():
    m = {"c_config": 60, "n_eligible_train": 150, "n_trainers": 300, "loss_phase": 0.5}
    assert build_policy("felix_c", {"avail": "true"}).compute_new_c(60, m) == 30
    assert build_policy("felix_c", {"phase": "true"}).compute_new_c(60, m) == 30
    assert build_policy("felix_c", {"avail": "true", "phase": "true"}).compute_new_c(60, m) == 15
    assert build_policy("felix_c", {"phase": "true"}).compute_new_c(60, dict(m, loss_phase=None)) == 60
    ctl = DynamicKCController(build_policy("felix_c", {"phase": "true"}), k_init=15, c_init=60, k_min=1, k_max=100,
                              c_min=15, c_max=60, k_follows_c=True)
    assert ctl.step(dict(m, loss_phase=0.05)) == (15, 15)  # floored at K


def test_felix_c_pool_bound_binds_only_on_a_small_trainable_pool():
    m = {"c_config": 60, "n_eligible_train": 150, "n_trainers": 300, "loss_phase": 0.5}
    pool = build_policy("felix_c", {"pool": 0.5})
    assert pool.compute_new_c(60, m) == 60  # 0.5 x 150 = 75 > 60: no bind
    assert pool.compute_new_c(60, dict(m, n_eligible_train=60)) == 30  # MobiPerf-like pool
    assert build_policy("felix_c", {"pool": 0.5, "phase": "true"}).compute_new_c(60, dict(m, n_eligible_train=60)) == 30


def test_loss_phase_tracks_peak():
    from flame.selector.loss_phase import LossPhase
    p = LossPhase(window=2)
    p.add(2.0)
    assert p.phi == 1.0  # window not full
    p.add(2.0)
    p.add(1.0)
    p.add(1.0)
    assert p.phi == 0.5
    p.add(3.0)
    p.add(3.0)
    assert p.phi == 1.0  # new data: new peak


def test_felix_c_pool_ramp_trades_throughput_for_choice_as_phase_falls():
    m = {"c_config": 60, "n_eligible_train": 60, "n_trainers": 300}
    ramp = build_policy("felix_c", {"pool": 0.5, "ramp": "true"})
    assert ramp.compute_new_c(60, dict(m, loss_phase=1.0)) == 60  # early: full throughput
    assert ramp.compute_new_c(60, dict(m, loss_phase=0.5)) == 45
    assert ramp.compute_new_c(60, dict(m, loss_phase=0.0)) == 30  # fit: the pool bound
    assert ramp.compute_new_c(60, dict(m, n_eligible_train=150, loss_phase=0.0)) == 60  # big pool: never binds
