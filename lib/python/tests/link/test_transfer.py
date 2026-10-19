# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path

import numpy as np
import pytest

from flame.link.config import LinkLayerConfig
from flame.link.transfer import estimate_transfer

MBPS = 1e6


def _est(rate, start_s, nbytes, prop=0.0):
    rate = np.asarray(rate, dtype=float)
    return estimate_transfer(rate.__getitem__, len(rate), start_s, nbytes, lambda k: prop)


def test_visible_whole_transfer_fits_in_one_step():
    e = _est([8 * MBPS] * 10, 2.0, 500_000, prop=0.003)  # 4 Mbit at 8 Mbps
    assert e.complete
    assert e.wait_s == 0.0
    assert e.transmit_s == pytest.approx(0.5)
    assert e.total_s == pytest.approx(0.503)
    assert e.throughput_mbps == pytest.approx(8.0)


def test_waits_for_next_pass():
    rate = [0.0] * 5 + [8 * MBPS] * 5
    e = _est(rate, 1.25, 1_000_000)  # 8 Mbit -> 1 s once visible at t=5
    assert e.wait_s == pytest.approx(3.75)
    assert e.transmit_s == pytest.approx(1.0)


def test_pauses_and_resumes_across_passes():
    rate = [8 * MBPS, 8 * MBPS, 0.0, 0.0, 8 * MBPS, 8 * MBPS]
    e = _est(rate, 0.0, 3_000_000)  # 24 Mbit = 3 visible seconds
    assert e.transmit_s == pytest.approx(3.0)
    assert e.wait_s == pytest.approx(2.0)


def test_time_varying_rate_integrates_bits():
    rate = [4 * MBPS, 8 * MBPS, 16 * MBPS]
    e = _est(rate, 0.0, 2_000_000)  # 16 Mbit: 4 + 8 + 4 (0.25 s of 16)
    assert e.transmit_s == pytest.approx(2.25)


def test_wraps_past_end_of_series():
    rate = [8 * MBPS, 0.0, 0.0]
    e = _est(rate, 1.0, 1_000_000)  # wraps to k=0 at t=3
    assert e.wait_s == pytest.approx(2.0)
    assert e.transmit_s == pytest.approx(1.0)


def test_never_visible_is_incomplete():
    e = _est([0.0] * 4, 0.0, 10)
    assert not e.complete
    assert math.isinf(e.total_s)


def test_config_defaults_and_overrides():
    cfg = LinkLayerConfig.from_dict({"routing": "isl", "transfer": {"efficiency": 0.5}})
    assert cfg.routing == "isl"
    assert cfg.transfer.efficiency == 0.5
    assert cfg.transfer.payload_bytes is None
    assert LinkLayerConfig.from_dict({}).routing == "direct"
    assert LinkLayerConfig.from_dict({}).enforcement == "sleep"


_ECEF = Path(__file__).resolve().parents[2] / "examples/_metadata/leo/ecef.npz"


@pytest.mark.skipif(not _ECEF.exists(), reason="ecef.npz not available")
def test_runtime_direct_transfer_on_real_pass():
    pytest.importorskip("itur")
    from flame.link.runtime import LinkRuntime

    rt = LinkRuntime.from_link_layer_dict(
        {"enabled": True, "satelliteEcefPath": str(_ECEF)}, direction="downlink"
    )
    # Satellite 289 is visible at t=3214 (docs/link-layer.md worked example).
    e = rt.transfer(289, 3214.0, 100 * 1024 * 1024)
    assert e.complete and e.wait_s == 0.0
    assert e.propagation_s == pytest.approx(508.3128 / 299792.458, rel=1e-3)
    assert 1000 < e.throughput_mbps < 2200


_LATENCY = _ECEF.parent / "latency.npz"


@pytest.mark.skipif(not _LATENCY.exists(), reason="latency.npz not available")
def test_runtime_isl_relays_when_not_in_view():
    pytest.importorskip("itur")
    from flame.link.runtime import LinkRuntime

    raw = {
        "enabled": True,
        "satelliteEcefPath": str(_ECEF),
        "routing": "isl",
        "isl": {"latencyPath": str(_LATENCY), "perHopProcessingMs": 1.0},
    }
    rt = LinkRuntime.from_link_layer_dict(raw, direction="downlink")
    z = np.load(_LATENCY)
    never = int(np.flatnonzero(~z["direct_visible"].any(axis=0))[0])

    e = rt.transfer(never, 100.0, 10 * 1024 * 1024)
    hops = int(z["hop_count"][100, never])
    assert e.complete and e.routing == "isl" and e.wait_s == 0.0
    assert e.hops == hops > 1
    assert e.propagation_s == pytest.approx(z["latency_ms"][100, never] / 1000 + (hops - 1) * 1e-3)

    # In view: same as direct (own RF link, one hop).
    e = rt.transfer(289, 3214.0, 10 * 1024 * 1024)
    assert e.hops == 1 and e.propagation_s == pytest.approx(508.3128 / 299792.458, rel=1e-3)
