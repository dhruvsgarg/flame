# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D9: one task per (trainer, task, model version) — aggregator ledger + retry policy,
selector timeout stamp, and the trainer-side discard of an already-answered request."""

import types
from types import SimpleNamespace

import pytest

import flame.mode.horizontal.asyncfl.top_aggregator as async_mod
from flame.mode.message import MessageType
from tests.mode.test_async_staggered_redispatch import _DistChannel, _make_dist_agg


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    monkeypatch.setattr(async_mod, "weights_to_device", lambda w, d: w)


def _agg(policy="none", backoff=10.0):
    ch = _DistChannel(["e1", "e2"])
    agg = _make_dist_agg(ch, staggered=False)  # vclock=100, round=5
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(
        task_retry_policy=policy, task_retry_backoff_s=backoff))
    agg._avail_now = lambda: agg._vclock.now
    agg._sim_committed = set()
    return agg, ch


class TestLedgerAndRetry:
    def test_dispatch_records_ledger_and_blocks_same_version(self):
        agg, ch = _agg()
        agg._distribute_weights("tag", "train")
        assert agg._task_ledger[("e1", "train")] == [5, 100.0, 0]
        assert agg._task_version_keys(ch, "train") == {"e1": (5, 0), "e2": (5, 0)}
        assert agg._task_version_keys(ch, "eval") == {}  # another task is another request

    def test_version_advance_releases(self):
        agg, ch = _agg()
        agg._distribute_weights("tag", "train")
        agg._round = 6
        assert agg._task_version_keys(ch, "train") == {}

    def test_none_never_retries_even_after_timeout(self):
        agg, ch = _agg("none")
        agg._distribute_weights("tag", "train")
        agg._task_timeout_at = {"e1": 190.0}
        agg._vclock.advance(10_000.0)
        assert "e1" in agg._task_version_keys(ch, "train")

    def test_fixed_retries_after_backoff_only_for_timeouts(self):
        agg, ch = _agg("fixed", backoff=10.0)
        agg._distribute_weights("tag", "train")
        agg._task_timeout_at = {"e1": 190.0}  # e2: aware eviction / no timeout -> never
        agg._vclock.advance(195.0)
        assert set(agg._task_version_keys(ch, "train")) == {"e1", "e2"}
        agg._vclock.advance(200.0)
        assert set(agg._task_version_keys(ch, "train")) == {"e2"}

    def test_exponential_doubles_per_retry(self):
        agg, ch = _agg("exponential", backoff=10.0)
        agg._distribute_weights("tag", "train")
        agg._distribute_weights("tag", "train")  # a retry at the same version
        assert agg._task_ledger[("e1", "train")][2] == 1
        agg._task_timeout_at = {"e1": 190.0}
        agg._vclock.advance(205.0)  # 15s < 10 * 2**1
        assert "e1" in agg._task_version_keys(ch, "train")
        agg._vclock.advance(210.0)
        assert "e1" not in agg._task_version_keys(ch, "train")

    def test_timeout_before_dispatch_does_not_count(self):
        agg, ch = _agg("fixed", backoff=0.0)
        agg._task_timeout_at = {"e1": 50.0}  # stale, from an older dispatch
        agg._distribute_weights("tag", "train")
        assert "e1" in agg._task_version_keys(ch, "train")

    def test_bad_policy_raises(self):
        agg, ch = _agg("sometimes")
        with pytest.raises(ValueError):
            agg._task_version_keys(ch, "train")

    def test_keys_reach_the_selector(self):
        agg, ch = _agg()
        seen = {}
        orig = ch.ends
        def _ends(state, task=None, **kw):
            seen.update(kw)
            return orig(state, task, **kw)
        ch.ends = _ends
        agg._distribute_weights("tag", "train")
        agg._distribute_weights("tag", "train")
        assert seen["trainer_version_keys"] == {"e1": (5, 0), "e2": (5, 0)}
        assert seen["agg_version_key"] == (5, 0)


class TestSelectorGuardAndTimeoutStamp:
    def _sel(self):
        from flame.selector.fedbuff import FedBuffSelector
        s = FedBuffSelector.__new__(FedBuffSelector)
        s.all_selected, s.ordered_updates_recv_ends = {"e1": 0.0}, []
        s.track_trainer_timeouts, s.send_timeout_wait_s, s._sim_now_s = {}, 90, 500.0
        return s

    def test_reclaim_stamps_timed_out_at(self):
        s = self._sel()
        sel_ends = {"e1"}
        s._reclaim_timed_out_ends(sel_ends)
        assert s.timed_out_at == {"e1": 500.0} and sel_ends == set()

    def test_guard_drops_same_version(self):
        from flame.selector.async_base import SelectContext
        s = self._sel()
        s.all_selected = {}
        s._task_eligible_states = {"train": ["AVL_TRAIN"]}
        ends = {e: SimpleNamespace(get_property=lambda k: None) for e in ("e1", "e2")}
        ctx = SelectContext(task_to_perform="train", agg_version_key=(5, 0),
                            trainer_version_keys={"e1": (5, 0), "e2": (4, 0)},
                            channel_props={}, connected_ends=ends, trainer_unavail_list=[])
        assert set(s._eligible_candidates(ends, ctx)) == {"e2"}


class TestTrainerDiscard:
    def _trainer(self, answered):
        from flame.mode.horizontal.syncfl.trainer import Trainer

        class _T(Trainer):
            def check_and_sleep(self): pass
            def evaluate(self): pass
            def initialize(self): pass
            def load_data(self): pass
            def train(self): pass
        t = _T.__new__(_T)
        t.trainer_id, t._round, t.task_to_perform, t._work_done = "t", 3, "train", False
        t._responded_version = dict(answered)
        t._phase_times, t._phase_vclock_s = {}, {}
        t.datasampler = SimpleNamespace(handle_metadata_from_aggregator=lambda m: None)
        cleaned = []
        sel = SimpleNamespace(ordered_updates_recv_ends=[])
        ch = SimpleNamespace(await_join=lambda: None, one_end=lambda s: "agg",
                             _selector=sel, cleanup_recvd_ends=lambda: cleaned.append(1))
        t.cm = SimpleNamespace(get_by_tag=lambda tag: ch)
        return t, ch, cleaned

    def _fetch(self, t, ch, msg):
        ch.recv = lambda end: (msg, None)
        t._fetch_weights("fetch")

    def test_same_task_same_version_is_discarded(self):
        t, ch, cleaned = self._trainer({"train": 5})
        self._fetch(t, ch, {MessageType.ROUND: 5, MessageType.TASK_TO_PERFORM: "train"})
        assert t.fetch_success is False and t._round == 3 and cleaned == [1]

    def test_older_version_is_discarded(self):
        t, ch, _ = self._trainer({"train": 5})
        self._fetch(t, ch, {MessageType.ROUND: 4, MessageType.TASK_TO_PERFORM: "train"})
        assert t.fetch_success is False

    def test_eval_after_train_same_version_is_accepted(self):
        t, ch, _ = self._trainer({"train": 5})
        self._fetch(t, ch, {MessageType.ROUND: 5, MessageType.TASK_TO_PERFORM: "eval"})
        assert t.fetch_success is True and t.task_to_perform == "eval"

    def test_newer_version_is_accepted(self):
        t, ch, _ = self._trainer({"train": 5})
        self._fetch(t, ch, {MessageType.ROUND: 6, MessageType.TASK_TO_PERFORM: "train"})
        assert t.fetch_success is True and t._round == 6

    def test_eot_is_never_discarded(self):
        t, ch, _ = self._trainer({"train": 5})
        self._fetch(t, ch, {MessageType.ROUND: 5, MessageType.EOT: True})
        assert t.fetch_success is True and t._work_done is True
