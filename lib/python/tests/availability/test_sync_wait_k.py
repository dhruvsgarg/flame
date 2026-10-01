# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N37 substrate: with syncWaitForK a withheld sync pick keeps its slot until dispatch+90s (real
cannot see the withhold), the sim wakes at the next timeout/delivery, and real stops a round's recv
at the first pending timeout so distribute can replace that pick."""

from types import SimpleNamespace

from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability
from flame.selector.properties import PROP_ROUND_START_TIME, PROP_SIM_SEND_TS
from flame.sim.virtual_clock import SimReorderBuffer

_DOWN = SortedDict({0.0: "AVL_TRAIN", 10.0: "UN_AVL", 200.0: "AVL_TRAIN"})
_UP = SortedDict({0.0: "AVL_TRAIN"})


class _End:
    def __init__(self):
        self.props = {}

    def set_property(self, k, v):
        self.props[k] = v


class _Channel:
    def __init__(self, ends):
        self._selector = SimpleNamespace(selected_ends=set())
        self._ends = {e: _End() for e in ends}

    def has(self, e):
        return e in self._ends

    def get_end_property(self, e, k):
        return self._ends[e].props.get(k)

    def dispatch(self, e, rnd, t):
        self._selector.selected_ends.add(e)
        self._ends[e].props[PROP_ROUND_START_TIME] = (rnd, None)
        self._ends[e].props[PROP_SIM_SEND_TS] = t


class _Agg(ClientAvailability):
    def __init__(self, wait_k=True, now=0.0):
        self.trainer_event_dict = {"a": _DOWN, "b": _UP, "c": _UP}
        self.pending_withheld = {}
        self._sim_withheld_payload = {}
        self._sim_withheld_delivering = {}
        self._sim_buffer = SimReorderBuffer()
        self.proactive_inflight_evict = False
        self.simulated = True
        self._round = 3
        self.config = SimpleNamespace(hyperparameters=SimpleNamespace(
            sync_wait_for_k=wait_k, max_experiment_runtime_s=1000))
        self._now = now

    def _avail_now(self):
        return self._now


def _withheld_pick(wait_k):
    agg, ch = _Agg(wait_k, now=20.0), _Channel(["a", "b", "c"])
    ch.dispatch("a", 3, 5.0)
    ch.dispatch("b", 3, 5.0)
    assert agg._sim_withhold_if_unavail(ch, "a", 20.0, ({}, ("a", None)))  # down at sct=20
    return agg, ch


def test_withheld_pick_holds_slot_until_timeout():
    agg, ch = _withheld_pick(wait_k=True)
    assert "a" in ch._selector.selected_ends and agg.pending_withheld["a"] == 200.0
    assert agg._sync_version_inflight(ch) == {"a", "b"}
    agg._now = 95.0  # age 90: not yet
    agg._abandon_stalled(ch)
    assert "a" in ch._selector.selected_ends
    agg._now = 95.5
    agg._abandon_stalled(ch)
    assert "a" not in ch._selector.selected_ends and agg.pending_withheld["a"] == 200.0
    assert "a" not in agg._withheld_slot_held


def test_flag_off_frees_at_sct():
    agg, ch = _withheld_pick(wait_k=False)
    assert "a" not in ch._selector.selected_ends


def test_async_unaware_withheld_holds_slot_until_timeout():
    """FX-N56: an async unaware baseline's withheld trainer keeps its slot until dispatch+90s, as in real."""
    agg, ch = _Agg(wait_k=False, now=20.0), _Channel(["a", "b", "c"])
    agg._sim_hold_withheld_slot = True
    agg._sim_inflight_expected = {"a": 20.0}
    ch.dispatch("a", 3, 5.0)
    assert agg._sim_withhold_if_unavail(ch, "a", 20.0, ({}, ("a", None)))
    assert "a" in ch._selector.selected_ends and "a" in agg._withheld_slot_held
    assert "a" not in agg._sim_inflight_expected
    assert "a" not in agg.withheld_held_ends()  # in flight, not yet an owed identity (real's label)
    agg._now = 95.5
    agg._abandon_stalled(ch)
    assert "a" not in ch._selector.selected_ends and agg.pending_withheld["a"] == 200.0
    assert "a" in agg.withheld_held_ends()


def test_sim_wakes_at_first_timeout_or_delivery():
    agg, ch = _withheld_pick(wait_k=True)
    agg._sim_buffer.add("b", 30.0, ({}, ("b", None)))  # b arrived: not a wake source
    assert abs(agg._sim_sync_next_wake(ch) - 95.0) < 1e-3  # a's timeout precedes its delivery (200)
    agg._sim_buffer.pop_min()
    agg._vclock = SimpleNamespace(now=20.0, advance=lambda t: setattr(agg._vclock, "now", max(agg._vclock.now, t)))
    agg._sim_sync_wait(ch)
    assert agg._vclock.now > 95.0


def test_reinject_clears_hold():
    agg, ch = _withheld_pick(wait_k=True)
    agg._now = 200.0
    agg._sim_reinject_ready_withheld()
    assert "a" not in agg._withheld_slot_held and agg._sim_buffer.has("a")


def test_accepted_ends_leave_version_inflight():
    agg, ch = _withheld_pick(wait_k=True)
    agg._sync_accepted_ends().add("b")
    ch.dispatch("c", 2, 0.0)  # an older version's straggler is not this version's pick
    assert agg._sync_version_inflight(ch) == {"a"}


def test_real_deadline_earliest_pending_timeout():
    agg, ch = _Agg(now=50.0), _Channel(["a", "b", "c"])
    agg.simulated, agg.agg_start_time_ts = False, 1000.0
    agg._avail_send_ts = lambda ch_, e: {"a": 0.0, "b": 10.0, "c": 30.0}[e]
    assert agg._real_round_recv_deadline(ch, ["a", "b", "c"]) == 1000.0 + 120.0
    assert agg._real_round_recv_deadline(ch, ["a", "b", "c"], earliest=True) == 1000.0 + 90.0
    agg._now = 95.0  # a is past its timeout: the next pending one is b
    assert agg._real_round_recv_deadline(ch, ["a", "b", "c"], earliest=True) == 1000.0 + 100.0


def test_replied_pick_is_not_abandoned():
    # An accepted or stale-rejected pick stays in selected_ends until commit; it must not become a phantom delivery.
    agg, ch = _Agg(now=100.0), _Channel(["b", "c"])
    agg.simulated = False
    ch.dispatch("b", 3, 0.0)
    ch.dispatch("c", 3, 0.0)
    ch._selector.ordered_updates_recv_ends = ["c"]
    agg._sync_accepted_ends().add("b")
    agg._avail_send_ts = lambda ch_, e: 0.0
    agg._abandon_stalled(ch)
    assert ch._selector.selected_ends == {"b", "c"} and not agg.pending_withheld


def test_redispatch_rearms_timeout():
    agg, ch = _Agg(now=100.0), _Channel(["b"])
    ch.dispatch("b", 3, 0.0)
    agg._task_timeout_at = {"b": 50.0}  # abandoned at an earlier dispatch, re-picked at t=60
    agg._avail_send_ts = lambda ch_, e: 60.0
    agg._now = 151.0
    agg._abandon_stalled(ch)
    assert "b" not in ch._selector.selected_ends


def test_real_deadline_never_collapses_to_now():
    agg, ch = _Agg(now=200.0), _Channel(["a"])
    agg.simulated, agg.agg_start_time_ts = False, 1000.0
    agg._avail_send_ts = lambda ch_, e: 0.0  # only a long-timed-out pick left
    assert agg._real_round_recv_deadline(ch, ["a"], earliest=True) >= 1000.0 + 200.0 + 0.5


def test_due_withheld_pick_not_reselectable_until_delivered():
    # A re-pick at vclock == delivery_ts collided with the reinjected older update in the per-end buffer.
    agg, ch = _withheld_pick(wait_k=True)
    agg._now = 200.0  # a's delivery is due but not yet reinjected
    assert "a" in agg.withheld_held_ends()
    agg._sim_reinject_ready_withheld()
    assert "a" in agg.withheld_held_ends()  # FX-N38: reinjected, still uncommitted
    agg._sim_take_withheld_delivering("a")  # its commit
    assert "a" not in agg.withheld_held_ends()


def test_accepted_late_update_blocks_repick_at_version():
    # P2 feddance: a late v111 update accepted at v124 let a top-up re-pick that end at v124; its reply stranded.
    from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator

    agg, ch = _Agg(), _Channel(["a", "b"])
    agg.version_key, agg._task_ledger = (3, 0), {("b", "train"): (1, 0.0, 0)}  # b last tasked at v1
    agg._sync_accepted_ends().add("b")
    assert TopAggregator._task_version_keys(agg, ch, "train") == {"b": (3, 0)}
    agg.config.hyperparameters.sync_wait_for_k = False
    assert TopAggregator._task_version_keys(agg, ch, "train") == {}


def test_awaited_includes_earlier_version_leftover():
    # P2 feddance: a pick left in flight by a commit went unread until its 90s abandon (queue_wait 88s).
    agg, ch = _withheld_pick(wait_k=True)
    agg._sync_accepted_ends().add("b")
    ch.dispatch("c", 2, 0.0)
    assert agg._sync_awaited(ch) == {"a", "c"} and agg._sync_version_inflight(ch) == {"a"}


def test_selector_reclaimed_held_pick_keeps_identity_hold():
    """FX-N56: once a selector timeout frees a held slot, the end is an owed identity, not re-pickable (EV10)."""
    agg, ch = _Agg(wait_k=False, now=20.0), _Channel(["a", "b", "c"])
    agg._sim_hold_withheld_slot = True
    ch.dispatch("a", 3, 5.0)
    assert agg._sim_withhold_if_unavail(ch, "a", 20.0, ({}, ("a", None)))
    agg._withheld_slot_held.intersection_update(set())  # what distribute does once the selector reclaimed "a"
    assert "a" in agg.withheld_held_ends()
