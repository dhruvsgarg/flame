# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-2: variance-plateau stopping policy.

At the α=1 operating point the achievable variance floor (~0.45) sits ABOVE the
commit gate (0.30), so a data-bin crosses the gate only on a noise dip and grinds
many iterations while the denoised estimate has long plateaued. The 'plateau'
policy commits a bin early once its per-bin variance curve flattens (relative
drop over the last N cycles < rel_delta) while var is still above threshold --
shipping the denoised estimate instead of a lucky noise sample.

`_should_force_commit_on_plateau` is the PURE decision (reads only instance
attrs, newest-var-last history in `var_prev_iter_list`), driven from
FedSGDAggregator.aggregate(). These tests pin:
(1) policy off / None / fixed_cap never forces a plateau commit (byte-identical off);
(2) it fires only when the curve has flattened AND var is still above threshold;
(3) it stays quiet while the curve is still dropping fast, or before N+1 samples;
(4) N (patience) and rel_delta (tolerance) sensitivity flips the decision;
(5) degenerate guards (non-positive baseline) return False.
"""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _PlateauAgg:
    """Minimal stand-in exposing only what `_should_force_commit_on_plateau` reads."""

    def __init__(self, policy, hist, N=3, eps=0.10, thr=0.30):
        self._var_stopping_policy = policy
        self.var_prev_iter_list = list(hist)   # newest var last
        self._var_plateau_patience = N
        self._var_plateau_rel_delta = eps
        self.var_threshold = thr

    decide = TopAggregator._should_force_commit_on_plateau


# --- (1) policy off/None/fixed_cap => never a plateau commit (byte-identical) ---

def test_policy_off_never_fires():
    flat = [0.50, 0.49, 0.485, 0.484]   # a clearly flattened curve
    for policy in (None, "off", "fixed_cap"):
        agg = _PlateauAgg(policy=policy, hist=flat, N=3, eps=0.10)
        assert agg.decide() is False


# --- (2) plateau fires on a flat curve with var still above threshold ----------

def test_plateau_fires_on_flat_curve_above_threshold():
    # last-N (N=3) window: 0.50 -> 0.484, rel drop = 0.032 < eps=0.10 ; var>thr.
    agg = _PlateauAgg(policy="plateau", hist=[0.9, 0.50, 0.49, 0.485, 0.484],
                      N=3, eps=0.10, thr=0.30)
    assert agg.decide() is True


def test_plateau_quiet_while_still_dropping_fast():
    # last-N window: 1.0 -> 0.55, rel drop = 0.45 >= eps=0.10 -> still improving.
    agg = _PlateauAgg(policy="plateau", hist=[2.0, 1.0, 0.8, 0.55],
                      N=3, eps=0.10, thr=0.30)
    assert agg.decide() is False


def test_plateau_quiet_when_var_already_under_threshold():
    # Curve flat AND var<=thr -> the natural gate commits; the plateau rule must NOT
    # claim it (avoids double-attribution / a spurious 'plateau' reason).
    agg = _PlateauAgg(policy="plateau", hist=[0.40, 0.30, 0.29, 0.285],
                      N=3, eps=0.10, thr=0.30)
    assert agg.decide() is False


# --- (3) insufficient history --------------------------------------------------

def test_no_fire_before_n_plus_one_samples():
    # len == N -> not enough to look back N steps (need > N).
    agg = _PlateauAgg(policy="plateau", hist=[0.49, 0.485, 0.484], N=3, eps=0.10)
    assert agg.decide() is False
    # one more sample -> now len == N+1, and it's flat -> fires.
    agg2 = _PlateauAgg(policy="plateau", hist=[0.50, 0.49, 0.485, 0.484],
                       N=3, eps=0.10, thr=0.30)
    assert agg2.decide() is True


# --- (4) N (patience) and eps (tolerance) sensitivity --------------------------

def test_eps_sensitivity_flips_decision():
    # window 0.50 -> 0.46 over N=3, rel drop = 0.08.
    hist = [0.7, 0.50, 0.48, 0.47, 0.46]
    assert _PlateauAgg("plateau", hist, N=3, eps=0.05).decide() is False  # 0.08 !< 0.05
    assert _PlateauAgg("plateau", hist, N=3, eps=0.10).decide() is True   # 0.08 <  0.10


def test_patience_window_sensitivity():
    # A curve that fell early then flattened. Short window (N=2) sees only the flat
    # tail -> fires; long window (N=4) still spans the early steep drop -> quiet.
    hist = [1.0, 0.5, 0.47, 0.465, 0.462]
    assert _PlateauAgg("plateau", hist, N=2, eps=0.10).decide() is True
    assert _PlateauAgg("plateau", hist, N=4, eps=0.10).decide() is False


# --- (5) degenerate guards -----------------------------------------------------

def test_nonpositive_baseline_guard():
    # A zero N-steps-back value would divide by zero; guard returns False.
    agg = _PlateauAgg(policy="plateau", hist=[0.0, 0.1, 0.1, 0.1], N=3, eps=0.10)
    assert agg.decide() is False
