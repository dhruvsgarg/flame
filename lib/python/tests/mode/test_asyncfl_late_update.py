# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-L27: in real asyncfl, a late (abandoned-then-withheld) update from an older dispatch must not mark the
end RECVD while its newer dispatch is outstanding; else the newer reply waits unread until the next dispatch."""

from datetime import datetime, timedelta
from types import SimpleNamespace

from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator


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
