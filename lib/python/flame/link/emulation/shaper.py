# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Per-trainer Toxiproxy shaping of real MQTT traffic from link estimates.

Each trainer reaches the broker through its own proxy
(listen_port_base + satellite_index). Before each transfer the trainer
shapes one stream from its `TransferEstimate`:
  upstream   (trainer -> broker) = update upload
  downstream (broker -> trainer) = model download
as latency = propagation delay and bandwidth = effective transmit rate.

Pass-scheduling waits (`wait_s`) are NOT shaped: a latency toxic delays
every byte, including MQTT keepalives, so hour-scale waits would drop the
connection. The trainer sleeps those instead.
"""

import atexit
import logging

from flame.link.config import LinkLayerConfig
from flame.link.emulation.toxiproxy import ToxiproxyClient
from flame.link.transfer import TransferEstimate

logger = logging.getLogger(__name__)

DEFAULT_MQTT_PORT = 1883


class ToxiproxyShaper:
    def __init__(self, cfg: LinkLayerConfig, satellite_index: int, broker: str):
        tp = cfg.toxiproxy
        self.client = ToxiproxyClient(tp.api_host, tp.api_port)
        self.bandwidth_scale = tp.bandwidth_scale
        self.proxy = f"flame_sat{satellite_index:03d}"
        host, _, port = broker.partition(":")
        upstream = f"{host}:{port or DEFAULT_MQTT_PORT}"
        self.broker_address = f"{tp.listen_host}:{tp.listen_port_base + satellite_index}"
        try:
            self.client.delete_proxy(self.proxy)  # stale proxy from a previous run
        except RuntimeError:
            pass
        self.client.create_proxy(self.proxy, self.broker_address, upstream)
        atexit.register(self.close)
        logger.info(f"[LINK_SHAPER] {self.proxy}: {self.broker_address} -> {upstream}")

    def shape(self, stream: str, est: TransferEstimate) -> None:
        """Apply `est`'s propagation latency and transmit rate to `stream`."""
        if not est.complete:
            return
        latency_ms = int(round(est.propagation_s * 1000))
        rate_kbps = max(1, int(est.throughput_mbps * 125 * self.bandwidth_scale))  # KB/s
        self.client.set_toxic(self.proxy, f"latency_{stream}", "latency", stream,
                              {"latency": latency_ms, "jitter": 0})
        self.client.set_toxic(self.proxy, f"bandwidth_{stream}", "bandwidth", stream,
                              {"rate": rate_kbps})

    def close(self) -> None:
        try:
            self.client.delete_proxy(self.proxy)
        except Exception:
            pass
