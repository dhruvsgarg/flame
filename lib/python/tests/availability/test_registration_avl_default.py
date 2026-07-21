# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Registration (`Channel.add`) stamps PROP_AVL_STATE=AVL_TRAIN so a just-joined
end is never read as UNKNOWN before the first selection stamps it -- avoids a
join-timing-dependent avail_composition count that diverged real vs sim. It's
only a default:

  * availability-AWARE (client_notify / oracular): `_avail_stamp_end_states`
    overwrites it every selection from the dynamic trace.
  * availability-UNAWARE (no trace): the stamp is a no-op, so the end stays
    AVL_TRAIN (e.g. syn_0).
"""

import asyncio
import types

from sortedcontainers import SortedDict

from flame.channel import Channel
from flame.availability.client_availability import ClientAvailability
from flame.config import TrainerAvailState
from flame.selector.properties import PROP_AVL_STATE


def _trace(*pairs):
    d = SortedDict()
    for ts, state in pairs:
        d[float(ts)] = state
    return d


class _Backend:
    def create_tx_task(self, name, end_id):
        pass


class _MockChannel:
    """Minimal channel with the property bag `_avail_stamp_end_states` touches."""

    def __init__(self, ends):
        self._ends = {e: None for e in ends}
        self._props = {}

    def set_end_property(self, end_id, key, value):
        self._props[(end_id, key)] = value

    def get_end_property(self, end_id, key):
        return self._props.get((end_id, key))


class _Harness(ClientAvailability):
    """Minimal ClientAvailability host to drive _avail_stamp_end_states."""

    def __init__(self, trainer_event_dict, now):
        self.trainer_event_dict = trainer_event_dict
        self._now = now

    def _avail_now(self):
        return self._now


class TestRegistrationDefaultAvlTrain:
    def test_registration_stamps_avl_train(self):
        # The real Channel.add must leave a fresh end at AVL_TRAIN, not None.
        async def scenario():
            ch = Channel.__new__(Channel)
            ch._name = "ch"
            ch._ends = {}
            ch._backend = _Backend()
            ch._end_state_info = {}
            ch.await_join_event = asyncio.Event()
            await ch.add("t1")
            return ch.get_end_property("t1", PROP_AVL_STATE)

        assert asyncio.run(scenario()) == TrainerAvailState.AVL_TRAIN

    def test_aware_trace_overrides_registration_default(self):
        # A trace saying UN_AVL at the current time must move the end off the
        # AVL_TRAIN default (aware baseline: client_notify / oracular).
        ch = _MockChannel(["t1"])
        ch.set_end_property("t1", PROP_AVL_STATE, TrainerAvailState.AVL_TRAIN)  # registration default
        trace = _trace((0, "AVL_TRAIN"), (300, "UN_AVL"), (600, "AVL_TRAIN"))
        h = _Harness(trainer_event_dict={"t1": trace}, now=400.0)  # inside the down window
        h._avail_stamp_end_states(ch)
        assert ch.get_end_property("t1", PROP_AVL_STATE) == TrainerAvailState.UN_AVL

    def test_aware_end_without_trace_keeps_avl_train(self):
        # Aware run, but this end has no trace entry -> defaults to AVL_TRAIN.
        ch = _MockChannel(["t1"])
        h = _Harness(trainer_event_dict={"other": _trace((0, "AVL_TRAIN"))}, now=400.0)
        h._avail_stamp_end_states(ch)
        assert ch.get_end_property("t1", PROP_AVL_STATE) == TrainerAvailState.AVL_TRAIN

    def test_unaware_keeps_registration_avl_train(self):
        # Availability-unaware (trainer_event_dict=None): stamp is a no-op, so
        # the registration-time AVL_TRAIN persists unchanged.
        ch = _MockChannel(["t1"])
        ch.set_end_property("t1", PROP_AVL_STATE, TrainerAvailState.AVL_TRAIN)  # registration default
        h = _Harness(trainer_event_dict=None, now=400.0)
        h._avail_stamp_end_states(ch)
        assert ch.get_end_property("t1", PROP_AVL_STATE) == TrainerAvailState.AVL_TRAIN
