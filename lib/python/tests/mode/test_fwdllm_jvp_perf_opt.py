# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Fluxtune JVP perf-opt (simulate_fwdllm.md §L) — the retained optimizations
must be BIT-IDENTICAL to the un-optimized grads, so enabling `jvp_perf_opt` on
fluxtune never changes training fidelity or real↔sim parity.

The core claim is `calculate_jvp(..., trainable_idx=T)` == `calculate_jvp(...)`
when the perturbation `v` is zero on the non-T (frozen) params: `p - h·0 = p`
exactly, so both paths feed the model identical param values. These tests pin
that (the trainer-level gating just chooses which path to call)."""

import pytest

torch = pytest.importorskip("torch")

from examples.fwdllm.trainer.forward_training.fwdgrad_utils import calculate_jvp


def _linear_func(params):
    # a deterministic scalar of the params (order-independent) so the JVP is
    # well-defined and precision-stable for an exact-equality check
    return sum(p.sum() for p in params)


def _params_and_v(n=6, frozen=(1, 3, 4), shape=(4, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    params = [torch.randn(shape, generator=g) for _ in range(n)]
    v = [torch.randn(shape, generator=g) for _ in range(n)]
    for i in frozen:              # frozen params get a ZERO perturbation
        v[i] = torch.zeros(shape)
    trainable_idx = [i for i in range(n) if i not in frozen]
    return params, v, trainable_idx


class TestTrainableOnlyFD:
    def test_bit_identical_to_all_param_fd(self):
        params, v, tidx = _params_and_v()
        l0, j0 = calculate_jvp(_linear_func, params, v)                       # legacy
        l1, j1 = calculate_jvp(_linear_func, params, v, trainable_idx=tidx)   # perf-opt
        assert torch.equal(j0, j1), (j0, j1)
        assert torch.equal(l0, l1)

    def test_none_is_the_legacy_path(self):
        params, v, _ = _params_and_v()
        a = calculate_jvp(_linear_func, params, v)
        b = calculate_jvp(_linear_func, params, v, trainable_idx=None)
        assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])

    def test_frozen_params_untouched_by_perturbation(self):
        # a func that reads a FROZEN param heavily — its contribution must be
        # identical in both signs (v=0 there), so it cancels out of the JVP
        params, v, tidx = _params_and_v(frozen=(0,))
        def f(p):
            return (p[0] ** 3).sum() + p[2].sum()   # p[0] frozen, nonlinear
        _, j_all = calculate_jvp(f, params, v)
        _, j_opt = calculate_jvp(f, params, v, trainable_idx=tidx)
        assert torch.equal(j_all, j_opt)

    def test_covers_extra_zero_indices_safely(self):
        # trainable_idx may include an index whose v happens to be 0 (a trainable
        # param that drew a zero dir) — still p-h·0=p, so no divergence
        params, v, tidx = _params_and_v(frozen=(2,))
        v[0] = torch.zeros_like(v[0])          # a "trainable" idx with zero dir
        l0, j0 = calculate_jvp(_linear_func, params, v)
        l1, j1 = calculate_jvp(_linear_func, params, v, trainable_idx=tidx)
        assert torch.equal(j0, j1) and torch.equal(l0, l1)
