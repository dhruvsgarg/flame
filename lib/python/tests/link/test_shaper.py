# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from flame.link.config import LinkLayerConfig
from flame.link.emulation import shaper as shaper_mod
from flame.link.transfer import TransferEstimate


class _FakeClient:
    def __init__(self, *args):
        self.proxies, self.toxics, self.deleted = {}, {}, []

    def delete_proxy(self, name):
        self.deleted.append(name)
        if name not in self.proxies:
            raise RuntimeError("DELETE failed: 404")
        del self.proxies[name]

    def create_proxy(self, name, listen, upstream):
        self.proxies[name] = (listen, upstream)

    def set_toxic(self, proxy, name, type_, stream, attributes):
        self.toxics[name] = (type_, stream, attributes)


def _est(throughput_mbps, propagation_s, complete=True):
    return TransferEstimate("downlink", "direct", 1, 0.0, 0.0, 1.0, propagation_s,
                            1.0 + propagation_s, throughput_mbps, 1, complete)


@pytest.fixture
def shaper(monkeypatch):
    monkeypatch.setattr(shaper_mod, "ToxiproxyClient", _FakeClient)
    monkeypatch.setattr(shaper_mod.atexit, "register", lambda f: None)
    cfg = LinkLayerConfig.from_dict({"toxiproxy": {"listenPortBase": 30000, "bandwidthScale": 0.5}})
    return shaper_mod.ToxiproxyShaper(cfg, 7, "localhost")


def test_proxy_per_satellite_to_default_mqtt_port(shaper):
    assert shaper.broker_address == "127.0.0.1:30007"
    assert shaper.client.proxies["flame_sat007"] == ("127.0.0.1:30007", "localhost:1883")


def test_shape_sets_latency_and_scaled_bandwidth(shaper):
    shaper.shape("upstream", _est(8.0, 0.0123))
    assert shaper.client.toxics["latency_upstream"] == ("latency", "upstream", {"latency": 12, "jitter": 0})
    # 8 Mbps = 1000 KB/s, scaled by 0.5
    assert shaper.client.toxics["bandwidth_upstream"] == ("bandwidth", "upstream", {"rate": 500})


def test_incomplete_estimate_leaves_stream_unshaped(shaper):
    shaper.shape("downstream", _est(0.0, math.inf, complete=False))
    assert shaper.client.toxics == {}


def test_close_deletes_proxy(shaper):
    shaper.close()
    assert "flame_sat007" not in shaper.client.proxies
