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


@pytest.mark.skipif(not _ECEF.exists(), reason="ecef.npz not available")
def test_isl_gateway_is_sticky_and_switches_by_margin():
    from flame.link.propagation import elevation_deg, geodetic_to_ecef_km
    from flame.link.runtime import LinkRuntime

    raw = {"enabled": True, "satelliteEcefPath": str(_ECEF), "routing": "isl", "isl": {"switchMarginDeg": 10.0, "gatewaySelection": "sticky"}}
    rt = LinkRuntime.from_link_layer_dict(raw, direction="downlink")
    gw = rt.gateway_schedule()
    cfg = rt.cfg
    gs = geodetic_to_ecef_km(cfg.ground_station.lat_deg, cfg.ground_station.lon_deg, 0.0, cfg.geometry.earth_radius_km)

    def elev(k, s):
        return elevation_deg(rt.ecef_km[k, s], gs)

    assert (gw >= 0).all()  # Svalbard always has a satellite in view
    switches = np.flatnonzero(gw[1:] != gw[:-1]) + 1
    assert 0 < len(switches) < len(gw) / 30  # held for tens of seconds, not per second
    for k in switches:
        old, new = int(gw[k - 1]), int(gw[k])
        if elev(k, old) >= cfg.min_elevation_deg:  # old still in view: only a 10 deg better one replaces it
            assert elev(k, new) >= elev(k, old) + 10.0


@pytest.mark.skipif(not _LATENCY.exists(), reason="latency.npz not available")
def test_isl_best_rate_gateway_repicks_each_step():
    pytest.importorskip("itur")
    from flame.link.runtime import LinkRuntime

    base = {"enabled": True, "satelliteEcefPath": str(_ECEF), "routing": "isl"}
    best = LinkRuntime.from_link_layer_dict({**base, "isl": {"latencyPath": str(_LATENCY)}})
    sticky = LinkRuntime.from_link_layer_dict({**base, "isl": {"latencyPath": str(_LATENCY), "gatewaySelection": "sticky"}})
    assert best.cfg.isl.gateway_selection == "best_rate"
    for k in range(0, 600, 60):
        g, rate = best._relay(k)
        assert g >= 0 and rate >= sticky._relay(k)[1]


def test_per_direction_bandwidth_overrides_shared():
    from flame.link.budget import compute_link_budget

    cfg = LinkLayerConfig.from_dict({"frequency": {"uplinkBandwidthMhz": 500, "downlinkBandwidthMhz": 120}})
    assert cfg.frequency.bandwidth_mhz_for("uplink") == 500
    assert cfg.frequency.bandwidth_mhz_for("downlink") == 120
    shared = LinkLayerConfig.from_dict({"frequency": {"bandwidthMhz": 300}})
    assert shared.frequency.bandwidth_mhz_for("uplink") == shared.frequency.bandwidth_mhz_for("downlink") == 300

    pytest.importorskip("itur")
    sat = np.array([6371.0 + 550.0, 0.0, 0.0])  # overhead a GS at (0, 0)
    gs = {"groundStation": {"latDeg": 0.0, "lonDeg": 0.0}}
    narrow = compute_link_budget(sat, LinkLayerConfig.from_dict({**gs, "frequency": {"downlinkBandwidthMhz": 120}}), "downlink")
    wide = compute_link_budget(sat, LinkLayerConfig.from_dict({**gs, "frequency": {"downlinkBandwidthMhz": 480}}), "downlink")
    assert wide.throughput_mbps > narrow.throughput_mbps
    assert wide.snr_db == pytest.approx(narrow.snr_db - 10 * np.log10(4))  # 4x bandwidth = 4x noise


def test_channels_multiply_throughput_not_noise():
    pytest.importorskip("itur")
    from flame.link.budget import compute_link_budget

    sat = np.array([6371.0 + 550.0, 0.0, 0.0])
    gs = {"groundStation": {"latDeg": 0.0, "lonDeg": 0.0}}
    one = compute_link_budget(sat, LinkLayerConfig.from_dict(gs), "downlink")
    eight = compute_link_budget(sat, LinkLayerConfig.from_dict({**gs, "frequency": {"downlinkChannels": 8}}), "downlink")
    up = compute_link_budget(sat, LinkLayerConfig.from_dict({**gs, "frequency": {"downlinkChannels": 8}}), "uplink")
    assert eight.snr_db == pytest.approx(one.snr_db)  # per-channel SNR unchanged
    assert eight.throughput_mbps == pytest.approx(8 * one.throughput_mbps)
    assert up.throughput_mbps < 2 * one.throughput_mbps  # uplink still 1 channel
