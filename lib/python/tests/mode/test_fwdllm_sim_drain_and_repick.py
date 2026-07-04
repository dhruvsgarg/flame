# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression for the 2026-07-04 fluxtune sim deadlock (simulate_fwdllm.md §K-D17).

Two independent bugs conspired to freeze the fluxtune async sim after one cohort:

  Bug A -- drain gated on channel RECV state, not on the sct reorder buffer.
    _aggregate_grads_async early-returned ("no ends yet") whenever the channel
    reported no end in RECV. But _sim_recv_min_grad greedily drains ALL ready
    channel messages into _sim_buffer on its first call (emptying RECV), so the
    remaining already-received grads were stranded in the buffer and the loop
    never popped them: 10 grads received, agg_goal=3 never met. The sim commit
    path's readiness must key on _sim_buffer / _sim_inflight_expected, not on the
    real transport's RECV bookkeeping (principle #8: real-transport artifact).

  Bug B -- the async_oort re-pick triplet was stamped at DISPATCH.
    Stamping the whole cohort at the current _curr_agg_version made every
    dispatched trainer match the aggregator's version; since the version only
    advances at a commit boundary (never reached, see Bug A), async_oort's
    filter excluded the entire pool (filtered_ends=0) -> no re-dispatch, ever.
    The triplet is now stamped on grad RETURN, so an in-flight-but-not-returned
    trainer stays eligible (its compute slot already guards it), and the pool is
    never frozen before the first commit.
"""

import torch

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer


# --------------------------------------------------------------------------- #
# Bug A -- the sim drain loop must not stall on empty RECV while the buffer is
# non-empty.
# --------------------------------------------------------------------------- #
class _GateChannel:
    def __init__(self, recv_ends):
        self._recv_ends = recv_ends

    def ends(self, state=None):
        return self._recv_ends


class _CM:
    def __init__(self, channel):
        self._channel = channel

    def get_by_tag(self, tag):
        return self._channel


class _DrainGateAgg:
    """Binds the real _aggregate_grads_async and spies _sim_recv_min_grad so we
    observe ONLY the entry-guard decision (drain vs. early-return)."""

    _aggregate_grads_async = TopAggregator._aggregate_grads_async

    def __init__(self, recv_ends, buffer_scts=(), inflight=(), simulated=True):
        self.simulated = simulated
        self._sim_buffer = SimReorderBuffer()
        for i, s in enumerate(buffer_scts):
            self._sim_buffer.add(f"b{i}", float(s), None)
        self._sim_inflight_expected = {e: 1.0 for e in inflight}
        self.cm = _CM(_GateChannel(recv_ends))
        self.drain_calls = []

    def _sim_recv_min_grad(self, channel, recv_ends):
        # Spy: record the call and return "nothing committable" so
        # _aggregate_grads_async returns at `if not msg` without needing the
        # heavy _process_single_trainer_message path.
        self.drain_calls.append(list(recv_ends))
        return None, ("", 0)


class TestDrainGateNotBlockedByEmptyRecv:
    def test_buffered_grads_drain_with_no_recv_ends(self):
        """The core deadlock: RECV empty but the reorder buffer still holds grads
        -> the loop MUST proceed to drain (call _sim_recv_min_grad), not stall."""
        agg = _DrainGateAgg(recv_ends=None, buffer_scts=(10.0, 20.0, 30.0))
        agg._aggregate_grads_async("param-channel")
        assert agg.drain_calls == [[]], (
            "buffered grads must drain even when the channel reports no RECV ends"
        )

    def test_inflight_expected_keeps_loop_alive_with_no_recv_ends(self):
        """A trainer still computing (in the in-flight gate) must also keep the
        loop draining so the buffered minimum is not stranded behind it."""
        agg = _DrainGateAgg(recv_ends=None, inflight=("A", "B"))
        agg._aggregate_grads_async("param-channel")
        assert agg.drain_calls == [[]]

    def test_early_return_when_truly_idle(self):
        """RECV empty AND buffer empty AND nothing in flight -> genuinely nothing
        to do, so the loop still early-returns (no busy-spin on the drain)."""
        agg = _DrainGateAgg(recv_ends=None)
        agg._aggregate_grads_async("param-channel")
        assert agg.drain_calls == []

    def test_recv_ends_present_always_drains(self):
        agg = _DrainGateAgg(recv_ends=["A"])
        agg._aggregate_grads_async("param-channel")
        assert agg.drain_calls == [["A"]]

    def test_real_mode_ignores_sim_buffer(self):
        """Real path must stay byte-identical: a populated _sim_buffer must NOT
        make the real loop proceed when RECV is empty."""
        agg = _DrainGateAgg(
            recv_ends=None, buffer_scts=(10.0,), simulated=False
        )
        agg._aggregate_grads_async("param-channel")
        assert agg.drain_calls == []


# --------------------------------------------------------------------------- #
# Bug B -- the re-pick triplet is stamped on grad RETURN, not at dispatch.
# --------------------------------------------------------------------------- #
class _RepickChannel:
    def __init__(self):
        self._props = {}
        self.provided_cleaned_up = []

        class _Sel:
            ordered_updates_recv_ends = []

        self._selector = _Sel()

    def set_end_property(self, end, key, value):
        self._props[(end, key)] = value

    def get_end_property(self, end, key):
        return self._props.get((end, key))

    def cleanup_provided_ends(self, end):
        self.provided_cleaned_up.append(end)

    def cleanup_recvd_end(self, end):
        pass


class _RepickAgg:
    """Drives the real _process_single_trainer_message far enough to observe the
    triplet-stamp side effect. aggregate_grads_from_trainers is stubbed (the
    stamp lands before it) and telemetry stays disabled by default."""

    process = TopAggregator._process_single_trainer_message

    def __init__(self, residence=True, simulated=True, curr_ver=(5, 2, 1)):
        self.simulated = simulated
        self._sim_inflight_residence = residence
        self._curr_agg_version = curr_ver
        self._trainer_state_dict = {}
        self._per_agg_trainer_list = []
        self._agg_goal_cnt = 0
        self._round = 1
        self.data_id = 2
        self.iteration_per_data_id = 1
        self.is_async = True
        self._model_version = 5
        self._sim_contrib_intervals = None
        self._round_cache_activity_ts = {}
        self._updates_in_queue = 0
        self._updates_received = {}
        self._trainer_last_model_version = {}
        self.grad_pool = []

    def aggregate_grads_from_trainers(self, *a, **k):
        pass  # stub: the stamp executes before this call


def _valid_grad_msg():
    return {
        MessageType.MODEL_VERSION: 5,
        MessageType.GRADIENTS: torch.zeros(1),
        MessageType.GRADIENTS_FOR_VAR_CHECK: torch.zeros(1),
        MessageType.STAT_UTILITY: 1.0,
        MessageType.DATASET_SIZE: 8,
    }


class TestRepickTripletStampedOnReturn:
    def test_returning_trainer_is_stamped_at_current_agg_version(self):
        agg = _RepickAgg(residence=True, curr_ver=(5, 2, 1))
        ch = _RepickChannel()

        assert agg.process(ch, _valid_grad_msg(), "t1", timestamp=0) is True
        # A RETURNED trainer records the triplet it just contributed at, so
        # async_oort skips re-picking it until the agg version advances.
        assert agg._trainer_state_dict == {"t1": (5, 2, 1)}

    def test_residence_off_leaves_triplet_map_empty(self):
        """Sync baselines (residence off): empty map -> async_oort filter inert
        -> byte-identical to Batch 1."""
        agg = _RepickAgg(residence=False)
        ch = _RepickChannel()

        assert agg.process(ch, _valid_grad_msg(), "t1", timestamp=0) is True
        assert agg._trainer_state_dict == {}

    def test_not_simulated_leaves_triplet_map_empty(self):
        agg = _RepickAgg(residence=True, simulated=False)
        ch = _RepickChannel()

        assert agg.process(ch, _valid_grad_msg(), "t1", timestamp=0) is True
        assert agg._trainer_state_dict == {}
