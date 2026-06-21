# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""One-in-flight-per-trainer invariant (simInflightResidence) for the felix
async stack.

Real keeps a trainer out of selection (VAL_CH_STATE_SEND) from the moment it is
dispatched until its update returns AND is aggregated — measured 0% overlapping
in-flight intervals over a full run. Sim, where a trainer is freed instantly
(no train sleep), without this gate re-dispatched a still-in-flight fast trainer
13.9% of the time; that overwrote the per-end ``_sim_inflight_expected`` entry so
the earlier update was lost to the sct gate (gate_holds=0) and committed
past-dated (the K3b/U3/U6 residual tail, commit_gap up to 119s, staleness 34).

These tests pin that with the flag ON, every dispatched-but-not-yet-committed
trainer (``_sim_inflight_expected`` keys) is added to the selector's unavailable
list, and that the gate is a no-op when off / in real mode / with nothing in
flight.
"""

import pytest

import flame.mode.horizontal.asyncfl.top_aggregator as async_mod
from tests.mode.test_async_staggered_redispatch import _DistChannel, _make_dist_agg


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    # Avoid needing a live ML framework: weights_to_device is identity here.
    monkeypatch.setattr(async_mod, "weights_to_device", lambda w, d: w)


class TestInflightResidence:
    def test_outstanding_trainers_held_out_of_selection(self):
        # e2 is free; e1 and e3 are in flight (dispatched, not yet committed).
        ch = _DistChannel(["e2"])
        agg = _make_dist_agg(ch, staggered=False)
        agg._sim_inflight_residence = True
        agg._sim_inflight_expected = {"e1": 150.0, "e3": 160.0}
        agg._distribute_weights("tag", "train")
        assert set(ch.unavail) == {"e1", "e3"}

    def test_flag_off_holds_nobody(self):
        ch = _DistChannel(["e2"])
        agg = _make_dist_agg(ch, staggered=False)
        agg._sim_inflight_residence = False
        agg._sim_inflight_expected = {"e1": 150.0, "e3": 160.0}
        agg._distribute_weights("tag", "train")
        assert ch.unavail == []

    def test_real_mode_unaffected(self):
        ch = _DistChannel(["e2"])
        agg = _make_dist_agg(ch, staggered=False, simulated=False)
        agg._sim_inflight_residence = True
        agg._sim_inflight_expected = {"e1": 150.0}
        agg._distribute_weights("tag", "train")
        assert ch.unavail == []

    def test_no_inflight_is_noop(self):
        ch = _DistChannel(["e1", "e2"])
        agg = _make_dist_agg(ch, staggered=False)
        agg._sim_inflight_residence = True
        agg._sim_inflight_expected = {}
        agg._distribute_weights("tag", "train")
        assert ch.unavail == []

    def test_excluded_trainer_keeps_its_inflight_entry(self):
        # Holding a trainer out of selection must NOT touch its outstanding
        # _sim_inflight_expected entry (that is the gate's record of the update
        # still owed); only a commit pops it.
        ch = _DistChannel(["e2"])
        agg = _make_dist_agg(ch, staggered=False)
        agg._sim_inflight_residence = True
        agg._sim_inflight_expected = {"e1": 150.0}
        agg._distribute_weights("tag", "train")
        assert agg._sim_inflight_expected["e1"] == 150.0
        assert "e1" not in ch.sent  # not re-dispatched
