# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FELIX_INSIGHTS I-E1: the channel emits one `wire` event per message, only when telemetry is on."""

import json

from flame import telemetry
from flame.channel import _emit_wire
from flame.telemetry.events import EVENT_WIRE


def test_wire_event_written(tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    try:
        _emit_wire("send", "t1", 1234)
        _emit_wire("recv", "t1", 99)
    finally:
        telemetry.shutdown()
    evs = [json.loads(line) for line in (tmp_path / "aggregator.jsonl").read_text().splitlines()]
    assert [(e["event"], e["dir"], e["peer"], e["bytes"]) for e in evs] == [
        (EVENT_WIRE, "send", "t1", 1234), (EVENT_WIRE, "recv", "t1", 99)]


def test_wire_noop_when_disabled():
    assert not telemetry.is_enabled()
    _emit_wire("send", "t1", 1)
