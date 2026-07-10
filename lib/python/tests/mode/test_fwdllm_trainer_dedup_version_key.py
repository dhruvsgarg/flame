# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""§M Step 1(c)/2: _fetch_weights' dedup guard keys on version_key
(model_version, iteration), not the old (data_id, iteration) -- data_id wraps
at total_data_bins, so a bare (data_id, iteration) match false-positives
across model_versions that recycle the same data_id, silently dropping a
genuine new dispatch (no grad sent, no abort_training log reason). Since
model_version bumps once per data-bin (§M Step 2), data_id is redundant in
the key and no longer compared.
"""

from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer
from flame.mode.message import MessageType


class _FakeSelector:
    def __init__(self):
        self.ordered_updates_recv_ends = []


class _FakeChannel:
    def __init__(self, msg):
        self._msg = msg
        self._selector = _FakeSelector()
        self.cleanup_calls = 0

    def await_join(self):
        pass

    def one_end(self, state):
        return "end_1"

    def recv(self, end_id):
        return self._msg, None

    def cleanup_recvd_ends(self):
        self.cleanup_calls += 1


class _FakeChannelManager:
    def __init__(self, channel):
        self._channel = channel

    def get_by_tag(self, tag):
        return self._channel


class _FakeTrainer:
    _fetch_weights = Trainer._fetch_weights

    def __init__(self, channel, model_version, data_id, iteration_per_data_id):
        self.cm = _FakeChannelManager(channel)
        self.trainer_id = "trainer_1"
        self.fetch_success = False
        self._work_done = False
        self.abort_training = False
        self.data_id = data_id
        self.iteration_per_data_id = iteration_per_data_id
        self._round = 1
        self._model_version = model_version
        self.time_mode = "real"


def _msg(model_version, data_id, iteration_per_data_id):
    return {
        MessageType.MODEL_VERSION: model_version,
        MessageType.DATA_ID: data_id,
        MessageType.ITERATION_PER_DATA_ID: iteration_per_data_id,
    }


class TestVersionKeyDedup:
    def test_same_tuple_same_model_version_aborts(self):
        # genuine duplicate: identical version_key -> abort + cleanup.
        channel = _FakeChannel(_msg(model_version=1, data_id=5, iteration_per_data_id=0))
        t = _FakeTrainer(channel, model_version=1, data_id=5, iteration_per_data_id=0)
        t._fetch_weights("fetch")

        assert t.abort_training is True
        assert channel.cleanup_calls == 1
        assert channel._selector.ordered_updates_recv_ends == ["end_1"]

    def test_same_data_id_iteration_different_model_version_does_not_abort(self):
        # data_id wraparound: (data_id, iteration) recycles across model_versions
        # -- the bug this fix closes. Must NOT abort.
        channel = _FakeChannel(_msg(model_version=7, data_id=5, iteration_per_data_id=0))
        t = _FakeTrainer(channel, model_version=1, data_id=5, iteration_per_data_id=0)
        t._fetch_weights("fetch")

        # _fetch_weights also calls cleanup_recvd_ends() unconditionally at the
        # end of a normal (non-aborted) pass -- that's independent of the dedup
        # guard under test here; what matters is abort_training stayed False.
        assert t.abort_training is False
        assert t.fetch_success is True

    def test_new_iteration_updates_local_state(self):
        # New iteration within the same data-bin (model_version unchanged) ->
        # not a duplicate.
        channel = _FakeChannel(_msg(model_version=1, data_id=5, iteration_per_data_id=1))
        t = _FakeTrainer(channel, model_version=1, data_id=5, iteration_per_data_id=0)
        t._fetch_weights("fetch")

        assert t.abort_training is False
        assert t.data_id == 5
        assert t.iteration_per_data_id == 1

    def test_data_id_is_not_compared_once_model_version_and_iteration_match(self):
        # data_id is a reporting/progress field only, not part of version_key --
        # model_version bumping once per data-bin already makes it redundant.
        channel = _FakeChannel(_msg(model_version=1, data_id=999, iteration_per_data_id=0))
        t = _FakeTrainer(channel, model_version=1, data_id=5, iteration_per_data_id=0)
        t._fetch_weights("fetch")

        assert t.abort_training is True
