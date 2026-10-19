# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from flame.link.config import LinkLayerConfig
from flame.link.delay import LinkDelayModel
from flame.link.transfer import estimate_transfer

MBPS = 1e6


class _FakeRuntime:
    def __init__(self, rate_bps, direction):
        self.cfg = LinkLayerConfig.from_dict({"enabled": True, "affectsDelay": True})
        self.rate = np.asarray(rate_bps, dtype=float)
        self.direction = direction
        self.calls = []

    def transfer(self, satellite_index, start_s, nbytes):
        self.calls.append(start_s)
        return estimate_transfer(
            self.rate.__getitem__, len(self.rate), start_s, nbytes, lambda k: 0.0,
            direction=self.direction,
        )


def test_upload_starts_after_download_and_compute():
    up = _FakeRuntime([8 * MBPS] * 100, "uplink")
    down = _FakeRuntime([8 * MBPS] * 100, "downlink")
    legs = LinkDelayModel(up, down).round_legs(0, 10.0, 5.0, 1_000_000)  # 1 s per leg
    assert up.calls == [10.0]
    assert down.calls == [pytest.approx(16.0)]
    assert legs.comm_s == pytest.approx(2.0)
    assert legs.as_telemetry()["link_upload_start_s"] == pytest.approx(16.0)


def test_upload_waits_for_pass_after_compute():
    rate = [8 * MBPS] * 3 + [0.0] * 7 + [8 * MBPS] * 5
    up, down = _FakeRuntime(rate, "uplink"), _FakeRuntime(rate, "downlink")
    legs = LinkDelayModel(up, down).round_legs(0, 0.0, 2.0, 1_000_000)
    # download 0-1 s, compute to 3 s, pass resumes at 10 s
    assert legs.upload.wait_s == pytest.approx(7.0)
    assert legs.comm_s == pytest.approx(1.0 + 7.0 + 1.0)


def test_unreachable_satellite_contributes_no_comm_delay():
    up, down = _FakeRuntime([0.0] * 5, "uplink"), _FakeRuntime([0.0] * 5, "downlink")
    legs = LinkDelayModel(up, down).round_legs(0, 0.0, 1.0, 10)
    assert not legs.complete
    assert legs.comm_s == 0.0


def test_disabled_unless_enabled_and_affects_delay():
    assert LinkDelayModel.from_link_layer_dict(None) is None
    assert LinkDelayModel.from_link_layer_dict({"enabled": True}) is None
    assert LinkDelayModel.from_link_layer_dict({"affectsDelay": True}) is None


def test_payload_bytes_configured_or_measured():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(10, 10)  # 110 float32 params
    up = _FakeRuntime([1.0], "uplink")
    m = LinkDelayModel(up, _FakeRuntime([1.0], "downlink"))
    assert m.payload_bytes(model) == 440
    up.cfg.transfer.payload_bytes = 1234
    assert m.payload_bytes(model) == 1234
