# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Flags for the simplified controller: fixed pool gate, pool-derived rho, GL decay off."""

import math

import torch

from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A
from examples.fwdllm.expts.saturation_stop import SaturationDetector

P_TRAIN, S, P = 450340, 1.5, 10.0


def _stub(gate="fixed", n=0, target=50, schedule="pool", rho=0.06):
    a = object.__new__(A)
    a._commit_gate = gate; a._pool_target = target
    a._gate_safety_s = S; a._gate_rho_ref = "annealed"
    a._server_step_rule = "trust_ratio"; a._rho_schedule = schedule
    a._rho_star = rho; a._rho_exp = 0.25; a._commit_count = 0
    a._p_trainable = P_TRAIN; a._g_rule = P; a._last_rho = None
    a._n_eff_scalar = float(n) * 0.98  # n_eff must not matter under `fixed`
    a.grad_for_var_check_list = [torch.zeros(1)] * n
    a.var = torch.tensor(5.0); a.var_threshold = 0.3
    return a


def test_fixed_gate_commits_on_upload_count_only():
    assert _stub(n=49)._gate_satisfied() is False
    assert _stub(n=50)._gate_satisfied() is True


def test_pool_schedule_is_constant_rho():
    rho = S * math.sqrt(P * 50 / P_TRAIN)
    a = _stub(rho=rho)
    assert abs(rho - 0.0500) < 1e-3
    for t in (0, 150, 800):
        a._commit_count = t
        assert a._rho_star_now() == rho


def _feed(det, accs):
    for i, acc in enumerate(accs):
        if det.update(2 * i, acc):
            return det.fired_at
    return None


def test_decay_off_never_fires_on_a_decline():
    drop = [0.8] * 30 + [0.7] * 60
    assert _feed(SaturationDetector(0, slope_horizon=10), drop) is not None
    assert _feed(SaturationDetector(0, slope_horizon=10, decay=False), drop) is None
