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


def test_noise_scale_k_ema_and_guards():
    p = build_policy("noise_scale_k", {"kappa": 0.5, "ema": 0.5})
    assert p.compute_new_k(10, {"noise_scale": float("inf")}) is None
    assert p.compute_new_k(10, {"noise_scale": 20}) == 10
    assert p.compute_new_k(10, {"noise_scale": 40}) == 15  # EMA 30 x 0.5


def test_availability_cap_keeps_choice():
    p = build_policy("availability_cap", {"rho": 0.5})
    assert p.compute_new_c(15, {"n_eligible_train": 75, "c_config": 15}) == 15
    assert p.compute_new_c(15, {"n_eligible_train": 12, "c_config": 15}) == 6


def test_controller_caps_k_at_c_and_c_at_config():
    ctl = DynamicKCController(build_policy("schedule", {"k_points": [[0, 20]], "c_points": [[0, 40]]}),
                              k_init=10, c_init=15, k_min=1, k_max=100, c_min=1, c_max=15)
    assert ctl.step({"n_updates": 0}) == (15, 15)


def test_noise_scale_k_sweep_ema_weights_k_over_n():
    p = build_policy("noise_scale_k", {})  # kappa 1, ema "sweep"
    assert p.compute_new_k(5, {"noise_scale": 10, "n_trainers": 50}) == 10
    assert p.compute_new_k(10, {"noise_scale": 30, "n_trainers": 50}) == 14  # 0.8 x 10 + 0.2 x 30


def test_aimd_k_steps_down_while_loss_holds_and_doubles_on_rise():
    p = build_policy("aimd_k", {})
    flat = [1.0, 1.1, 0.9, 1.0]
    assert p.compute_new_k(10, {"cycle_losses": flat, "n_trainers": 4}) is None  # first window: no reference yet
    assert p.compute_new_k(10, {"cycle_losses": flat, "n_trainers": 4}) == 9  # no rise: larger step
    assert p.compute_new_k(9, {"cycle_losses": [3.0, 3.1], "n_trainers": 4}) is None  # window still filling
    assert p.compute_new_k(9, {"cycle_losses": [3.0, 3.2], "n_trainers": 4}) == 18  # rise past 2 SE: back off
