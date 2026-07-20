# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""`_distribute_weights_async` never stamped `channel.properties["vclock_now"]`
before dispatching, unlike asyncfl/top_aggregator.py's own dispatch path
(simulate_fwdllm.md §B, 2026-07-20). AsyncOortSelector reads this field to
stamp its in-flight abandon-timeout clock (`_sim_now_s`/`_abandon_clock_now`)
and to tag `selection_train` telemetry -- fluxtune's sim runs silently read
`None` always, so the abandon-timeout fell back to wall-clock `time.time()`
instead of virtual time, and every selection event's `vclock_now` was absent.
Covers the fix: the property is now set from `self.vclock_now` before
`channel.ends()` is called.
"""
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeChannelManager:
    def __init__(self, channel):
        self._channel = channel

    def get_by_tag(self, tag):
        return self._channel


class _FakeChannel:
    def __init__(self):
        self.properties = {}
        self.vclock_now_seen_at_ends_call = "NOT_CALLED"

    def await_join(self):
        return True

    def set_curr_unavailable_trainers(self, trainer_unavail_list=None):
        pass

    def ends(self, **kwargs):
        # Snapshot what the selector would have seen, then short-circuit the
        # rest of _distribute_weights_async (payload prep etc. -- irrelevant
        # to this fix) by returning nothing.
        self.vclock_now_seen_at_ends_call = self.properties.get("vclock_now")
        return {}


class _FakeAggregator:
    _distribute_weights_async = TopAggregator._distribute_weights_async

    def __init__(self, vclock_now, channel, simulated=True):
        self.simulated = simulated
        self.trainer_event_dict = None
        self.cm = _FakeChannelManager(channel)
        self.version_key = (1, 0)
        self._trainer_state_dict = {}
        self.data_id = 0
        self._vclock_now = vclock_now
        self.weights = None

    @property
    def vclock_now(self):
        return self._vclock_now

    def get_global_model_params(self):
        return {}


class TestDistributeWeightsAsyncVclockStamp:
    def test_stamps_channel_properties_before_ends_call(self):
        channel = _FakeChannel()
        agg = _FakeAggregator(vclock_now=42.5, channel=channel)

        agg._distribute_weights_async("some_tag")

        assert channel.vclock_now_seen_at_ends_call == 42.5

    def test_real_mode_stamps_none_not_a_stale_value(self):
        channel = _FakeChannel()
        agg = _FakeAggregator(vclock_now=None, channel=channel, simulated=False)

        agg._distribute_weights_async("some_tag")

        assert channel.vclock_now_seen_at_ends_call is None
