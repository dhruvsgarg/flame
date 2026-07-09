# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-3: gradient-aware aggregation.

The default fedbuff `new` rate rescales an update's MAGNITUDE by staleness×utility
but cannot refuse a wrong DIRECTION, so anti-aligned JVP estimates still get
averaged. The `grad_aware` rate weights by direction (alignment gate) and
optionally reliability (inverse-variance), BOUNDED so the result never exceeds the
base rate -> the effective server LR never inflates, so no LR re-tune is needed
for stability.

`_grad_aware_rate` (pure scalar) and `_cosine_flat` (tensor cosine vs the running
aggregate) are the testable primitives. These pin:
(1) both mechanisms off -> passthrough (== base rate);
(2) alignment gate: no-op when aligned, linear down-weight when anti-aligned, 0 at
    cos=-1, None cos -> no-op, and a non-zero align_floor gates weakly-aligned too;
(3) inverse-variance: no-op at/under the reference, shrinks above it, None -> no-op;
(4) the result is always <= base (never inflates LR);
(5) cosine sign/degenerate behavior.
"""

import torch

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator

rate = TopAggregator._grad_aware_rate
cosf = TopAggregator._cosine_flat


# --- (1) both off -> passthrough -------------------------------------------

def test_both_off_is_passthrough():
    assert rate(0.7, cos=-1.0, var_i=99.0, var_ref=0.3,
                align_gate=False, inverse_var=False) == 0.7


# --- (2) alignment gate (S2) -----------------------------------------------

def test_align_noop_when_aligned():
    # cos >= align_floor -> factor 1.
    assert rate(0.8, cos=0.5, var_i=None, var_ref=0.3, align_gate=True) == 0.8
    assert rate(0.8, cos=0.0, var_i=None, var_ref=0.3, align_gate=True) == 0.8


def test_align_downweights_anti_aligned():
    # floor 0: factor = (cos+1)/1 for cos<0. cos=-0.5 -> 0.5, cos=-1 -> 0.
    assert rate(1.0, cos=-0.5, var_i=None, var_ref=0.3, align_gate=True) == 0.5
    assert rate(1.0, cos=-1.0, var_i=None, var_ref=0.3, align_gate=True) == 0.0
    assert rate(0.6, cos=-0.5, var_i=None, var_ref=0.3, align_gate=True) == 0.3


def test_align_none_cos_is_noop():
    assert rate(0.9, cos=None, var_i=None, var_ref=0.3, align_gate=True) == 0.9


def test_align_floor_gates_weakly_aligned():
    # floor 0.2: cos=0.1 is below the floor -> gated. factor=(0.1+1)/(0.2+1)=1.1/1.2.
    r = rate(1.0, cos=0.1, var_i=None, var_ref=0.3, align_gate=True, align_floor=0.2)
    assert abs(r - (1.1 / 1.2)) < 1e-9
    # cos above the floor -> untouched.
    assert rate(1.0, cos=0.3, var_i=None, var_ref=0.3, align_gate=True,
                align_floor=0.2) == 1.0


# --- (3) inverse-variance (S1) ---------------------------------------------

def test_inverse_var_noop_at_or_under_ref():
    # var_i <= var_ref -> min(1, ref/var) == 1.
    assert rate(1.0, cos=None, var_i=0.2, var_ref=0.3,
                align_gate=False, inverse_var=True) == 1.0


def test_inverse_var_shrinks_above_ref():
    # var_i=0.6, ref=0.3 -> ~0.5 (eps negligible).
    r = rate(1.0, cos=None, var_i=0.6, var_ref=0.3,
             align_gate=False, inverse_var=True, var_eps=0.0)
    assert abs(r - 0.5) < 1e-9


def test_inverse_var_none_is_noop():
    assert rate(0.7, cos=None, var_i=None, var_ref=0.3,
                align_gate=False, inverse_var=True) == 0.7


def test_align_and_inverse_var_compose():
    # base 1.0 * align(cos=-0.5 -> .5) * invvar(var .6/ref .3 -> .5) = .25
    r = rate(1.0, cos=-0.5, var_i=0.6, var_ref=0.3,
             align_gate=True, inverse_var=True, var_eps=0.0)
    assert abs(r - 0.25) < 1e-9


# --- (4) never inflates the LR ---------------------------------------------

def test_never_exceeds_base():
    for cos in (-1.0, -0.3, 0.0, 0.9, None):
        for var_i in (0.0, 0.3, 5.0, None):
            r = rate(0.5, cos=cos, var_i=var_i, var_ref=0.3,
                     align_gate=True, inverse_var=True)
            assert r <= 0.5 + 1e-12


# --- (5) cosine primitive ---------------------------------------------------

def _np(names):  # minimal (name, param) list; param unused by _cosine_flat
    return [(n, None) for n in names]


def test_cosine_aligned_and_opposite():
    g = {"w": torch.tensor([1.0, 0.0, 0.0])}
    run_same = [torch.tensor([2.0, 0.0, 0.0])]
    run_opp = [torch.tensor([-2.0, 0.0, 0.0])]
    assert abs(cosf(g, run_same, _np(["w"])) - 1.0) < 1e-6
    assert abs(cosf(g, run_opp, _np(["w"])) + 1.0) < 1e-6


def test_cosine_none_on_zero_running():
    g = {"w": torch.tensor([1.0, 1.0])}
    assert cosf(g, [torch.zeros(2)], _np(["w"])) is None


def test_cosine_multi_tensor_orthogonal():
    g = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 0.0])}
    run = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])]
    assert abs(cosf(g, run, _np(["a", "b"]))) < 1e-6
