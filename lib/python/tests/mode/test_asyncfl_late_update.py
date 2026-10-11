# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-L27: in real asyncfl, a late (abandoned-then-withheld) update from an older dispatch must not mark the
end RECVD while its newer dispatch is outstanding; else the newer reply waits unread until the next dispatch."""

from datetime import datetime, timedelta
from types import SimpleNamespace

from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from tests.mode.test_asyncfl_duplicate_contribution import _ConcreteAgg


class _End:
    def __init__(self):
        self.props = {KEY_END_STATE: VAL_END_STATE_RECVD}  # recv just marked it

    def set_property(self, k, v):
        self.props[k] = v


def _run(sent_versions, recv_version, simulated=False):
    t0 = datetime(2026, 1, 1)
    sent = {v: t0 + timedelta(seconds=i) for i, v in enumerate(sent_versions)}
    agg = SimpleNamespace(simulated=simulated, _track_trainer_version_duration_s={"t1": {"sent_wts_version_ts": sent}})
    ch = SimpleNamespace(_ends={"t1": _End()}, has=lambda e: True)
    TopAggregator._keep_newer_dispatch_inflight(agg, ch, "t1", recv_version)
    return ch._ends["t1"].props[KEY_END_STATE]


def test_late_update_keeps_newer_dispatch_inflight():
    assert _run([181, 353], 181) == VAL_END_STATE_NONE


def test_reply_to_latest_dispatch_frees_end():
    assert _run([181, 353], 353) == VAL_END_STATE_RECVD


def test_sim_untouched():
    assert _run([181, 353], 181, simulated=True) == VAL_END_STATE_RECVD


def _agg(timed_out=None, ledger=None, rx=None, withheld=None, agg_start=1000.0):
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg.simulated, agg.agg_start_time_ts = False, agg_start
    agg._task_ledger, agg._last_rx_ts, agg.pending_withheld = ledger or {}, rx or {}, withheld or {}
    return agg, SimpleNamespace(_selector=SimpleNamespace(timed_out_at=timed_out or {}))


def test_arrived_abandoned_end_is_received():
    # P2 fedbuff: an abandoned end's late update sat 6s until the end's next dispatch.
    agg, _ = _agg()
    ch = SimpleNamespace(ends_with_pending_rx=lambda: {"t9", "t1", "gone"}, has=lambda e: e != "gone", _selector=None)
    assert agg._with_arrived_ends(ch, ["t1", "t2"]) == ["t1", "t2", "t9"]
    assert agg._with_arrived_ends(ch, None) == ["t1", "t9"]


def test_owed_end_is_received_before_it_arrives():
    # cifar P8 fedbuff: a timed-out end's late update arrived mid-recv and waited 29s for the next call.
    ledger = {("t5", "train"): [318, 400.0, 0], ("t6", "train"): [318, 400.0, 0], ("t7", "train"): [330, 500.0, 0]}
    agg, ch = _agg(timed_out={"t5": 490.0, "t6": 490.0, "t7": 490.0},  # selector: avail clock
                   ledger=ledger, rx={"t6": 495.0}, withheld={"t8": 600.0})
    # t5 owed; t6 answered after its timeout; t7 re-dispatched after its timeout; t8 evicted (withheld).
    assert agg._owed_ends(ch) == {"t5", "t8"}


def test_timed_out_at_uses_the_avail_clock_in_real():
    agg, ch = _agg(timed_out={"t1": 90.0})
    agg._task_timeout_at = {"t2": 95.0}
    assert agg._timed_out_at(ch) == {"t1": 90.0, "t2": 95.0}


def test_selector_timeout_stamp_holds_owed_end_in_real():
    # Run 4 EV17: the selector stamps channel vclock_now (avail clock) in real; re-basing it as epoch held nothing.
    from flame.selector.fedbuff import FedBuffSelector
    sel = FedBuffSelector(_seed=7, c=4, aggGoal=2)
    sel.all_selected, sel.ordered_updates_recv_ends = {"t5": 400.0}, []
    sel._sim_now_s = 491.0  # = channel_props["vclock_now"] = agg._avail_now()
    sel._reclaim_timed_out_ends(set())
    agg, _ = _agg(ledger={("t5", "train"): [318, 400.0, 0]})
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace())
    assert agg.real_owed_held_ends(SimpleNamespace(_selector=sel)) == {"t5"}


def test_owed_end_is_held_out_of_selection_in_real():
    # FX-N50: real re-picked a trainer whose update was still behind its send-gate (5 of 9 on P2 fedbuff).
    ledger = {("t5", "train"): [318, 400.0, 0]}
    agg, ch = _agg(timed_out={"t5": 490.0}, ledger=ledger)
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace())
    assert agg.real_owed_held_ends(ch) == {"t5"}
    agg._note_real_receipt = TopAggregator._note_real_receipt.__get__(agg)
    agg._avail_now = lambda: 520.0
    agg._note_real_receipt("t5")  # its reply lands: identity released
    assert agg.real_owed_held_ends(ch) == set()


def test_owed_hold_is_real_only_and_switchable():
    agg, ch = _agg(timed_out={"t5": 490.0}, ledger={("t5", "train"): [318, 400.0, 0]})
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(real_hold_owed_ends=False))
    assert agg.real_owed_held_ends(ch) == set()
    agg.config.hyperparameters.real_hold_owed_ends = True
    agg.simulated = True
    assert agg.real_owed_held_ends(ch) == set()


def test_real_dispatch_is_logged_for_ev17(monkeypatch):
    from flame import telemetry
    calls = []
    monkeypatch.setattr(telemetry, "is_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "emit", lambda ev, **f: calls.append((ev, f)))
    agg, _ = _agg()
    agg._round, agg._avail_now = 7, lambda: 12.0
    agg._record_task_dispatch("t1", "train")
    assert calls == [("dispatch", {"round": 7, "end_id": "t1", "task": "train", "time_mode": "real"})]
    agg.simulated = True  # sim's asyncfl emits its own richer dispatch event
    agg._record_task_dispatch("t1", "train")
    assert len(calls) == 1


def _eval_reply(last_sent):
    from flame.mode.message import MessageType
    from tests.mode.test_asyncfl_duplicate_contribution import _FakeChannel, _make_agg
    freed = []
    ch = _FakeChannel([({MessageType.STAT_UTILITY: 0.5, MessageType.MODEL_VERSION: 1}, ("t1", 0.0))])
    ch._selector = SimpleNamespace(remove_from_selected_ends=lambda ends, e: freed.append(e),
                                   _cleanup_removed_ends=lambda e: None)
    ch._ends = {"t1": _End()}
    agg = _make_agg()
    agg.cm = SimpleNamespace(get_by_tag=lambda t: ch)
    agg._last_task_sent = {"t1": last_sent}
    agg._aggregate_weights("param-channel")
    return freed


def test_eval_reply_keeps_newer_train_in_flight():
    # FX-N38: cifar T3_syn_50 felix sim re-picked 0370 at v436 while its v435 train (sent after the eval) was buffered.
    assert _eval_reply("train") == []
    assert _eval_reply("eval") == ["t1"]
