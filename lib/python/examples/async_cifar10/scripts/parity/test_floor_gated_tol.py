# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""DIST tolerances sized against the measured same-seed replicate floor (D-24).

The motivating defect: `v1_iter_per_data_id` shipped a 15% gate calibrated when
the floor was 13.3%. Removing the noise source dropped the floor to 0.6% and the
gate stayed at 15%, so a 5.4% real↔sim gap — 9x the floor — graded green.
"""
import pytest

from parity.checks import floor_gated_tol


class TestFloorGatedTol:
    def test_no_floor_leaves_the_nominal_alone(self):
        """A baseline with no replicate is graded exactly as before, never on
        some other baseline's floor (D-36)."""
        assert floor_gated_tol(0.15, None) == (0.15, None)

    def test_tightens_toward_a_small_floor(self):
        """The motivating case: 15% nominal over a 0.6% floor."""
        tol, why = floor_gated_tol(0.15, 0.0058)
        assert why is None
        assert tol == pytest.approx(0.02)          # min_abs, not 3x0.0058
        assert 0.054 > tol                          # the gap that used to pass

    def test_scales_with_a_larger_floor(self):
        tol, why = floor_gated_tol(0.15, 0.041)
        assert why is None
        assert tol == pytest.approx(0.123)          # 3 x floor, under nominal

    def test_never_loosens_past_the_nominal(self):
        """A noisy baseline must not silently WIDEN its gate."""
        tol, why = floor_gated_tol(0.05, 0.016)
        assert tol <= 0.05 and why is None

    def test_floor_at_or_above_tolerance_is_ungradeable(self):
        """Not a pass and not a fail — a verdict there is a coin flip."""
        tol, why = floor_gated_tol(0.02, 0.066)
        assert tol == 0.02
        assert why and "grades noise" in why

    def test_min_abs_is_a_floor_on_the_tightening(self):
        """A near-zero floor must not manufacture an unmeetable gate."""
        tol, why = floor_gated_tol(0.15, 0.0)
        assert tol == pytest.approx(0.02) and why is None


class TestRunAllParityWiring:
    """The gate must reach the rungs and stamp what it did, or a re-grade is
    unreadable months later."""

    def test_ungradeable_rungs_become_skips(self):
        from parity.checks import floor_gated_tol as f
        # v2's 2% nominal under fluxtune's 6.6% ON floor
        _, why = f(0.02, 0.066)
        assert why is not None, "fluxtune v2 must refuse to grade"

    def test_felix_round_v1_would_fail_after_gating(self):
        tol, _ = floor_gated_tol(0.15, 0.0058)
        assert 0.054 > tol, "the 5.4% gap must no longer pass"
