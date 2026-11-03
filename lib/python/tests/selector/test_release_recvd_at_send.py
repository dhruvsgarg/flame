# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D24: real refilled a freed async slot one arrival late -- SEND counted an ingested (RECVD) end as in flight
until the next RECV call dropped it (refill delay 0.35-0.55s vs sim 0.01s, pool_fixA_verify)."""

import time

from flame.end import KEY_END_STATE, VAL_END_STATE_RECVD
from flame.selector.async_base import SelectContext
from flame.selector.fedbuff import FedBuffSelector


class _Stop(Exception):
    pass


def _extra_after_send(make_ends, release):
    sel = FedBuffSelector.__new__(FedBuffSelector)
    sel.requester = "agg"
    sel.selected_ends = {"agg": {"a", "b"}}
    sel.all_selected = {"a": time.time(), "b": time.time()}
    sel.release_recvd_at_send = release
    ends = make_ends(["a", "b", "c"])
    ends["a"].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)  # a's update was just ingested
    ctx = SelectContext(task_to_perform="train", channel_props={"round": 1}, trainer_unavail_list=[],
                        agg_version_key=(1, 0, 0), trainer_version_keys={})
    sel._reclaim_timed_out_ends = lambda s: None

    def _stop(c):
        raise _Stop(c.extra)
    sel._model_version_for = _stop
    try:
        sel._handle_send_state(ends, 2, ctx)
    except _Stop as e:
        return e.args[0], sel.selected_ends["agg"]
    return 0, sel.selected_ends["agg"]  # extra 0: SEND returned before choosing


def test_ingested_slot_refills_at_send(make_ends):
    extra, in_flight = _extra_after_send(make_ends, release=True)
    assert extra == 1 and in_flight == {"b"}


def test_legacy_waits_for_next_recv(make_ends):
    extra, in_flight = _extra_after_send(make_ends, release=False)
    assert extra == 0 and in_flight == {"a", "b"}
