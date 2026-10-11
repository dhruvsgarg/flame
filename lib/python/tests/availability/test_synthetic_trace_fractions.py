# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""A synthetic trace's name is its full-day unavailable fraction (FX-N35: the old syn_20 was 10.8%)."""

import pytest

from flame.availability.trace import effective_unavailability


@pytest.mark.parametrize("name", ["syn_10", "syn_20", "syn_50"])
def test_full_day_fraction_matches_name(name):
    assert effective_unavailability(name, 86400) == pytest.approx(int(name[4:]) / 100, abs=0.02)


@pytest.mark.parametrize("name", ["syn_10", "syn_20", "syn_50"])
def test_spans_the_mobiperf_horizon_at_its_level(name):
    # Extended from 24 h (then all UN_AVL) to mobiperf's 149 h span with the same chain (FX-D18).
    from flame.availability.trace import load_trace
    mobi_end = max(load_trace("mobiperf_2st", f"trainer_{i:03d}").peekitem(-1)[0] for i in range(1, 301))
    assert max(load_trace(name, f"trainer_{i:03d}").peekitem(-1)[0] for i in range(1, 301)) >= mobi_end
    assert effective_unavailability(name, 530000) == pytest.approx(int(name[4:]) / 100, abs=0.02)


@pytest.mark.parametrize("name", ["syn_10", "syn_20", "syn_50"])
def test_no_all_available_head(name):
    # Trace time starts at the join barrier, so no head is needed for registration (FX-D18): 10 min already
    # sees the named fraction (syn_10/syn_50 used to be 0% until 600s).
    assert effective_unavailability(name, 600) == pytest.approx(int(name[4:]) / 100, abs=0.03)


def test_mobiperf_has_no_injected_head():
    from flame.availability.trace import load_trace, state_at
    from flame.config import TrainerAvailState
    down = sum(state_at(load_trace("mobiperf_2st", f"trainer_{i:03d}"), 0.0) == TrainerAvailState.UN_AVL
               for i in range(1, 301))
    assert down > 200  # ~90% unavailable at t=0 (was 0 before the +300s shift)
