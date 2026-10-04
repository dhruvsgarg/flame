# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared ground-station CLI handling and vectorized per-satellite link
budget series, used by both link_budget_sweep.py and
plot_link_vs_elevation.py so a "pass" and an "elevation curve" are always
computed the same way."""

import math
import warnings

import astropy.units as u
import itur
import numpy as np
import pandas as pd

from flame.link.config import LinkLayerConfig
from flame.link.propagation import geodetic_to_ecef_km, ionospheric_loss_db

SPEED_OF_LIGHT_KM_S = 299_792.458

GROUND_STATION_PRESETS = {
    "svalbard": (78.9243, 11.9231),
    "north_pole": (90.0, 0.0),
    "south_pole": (-90.0, 0.0),
}


def add_ground_station_args(parser):
    """One ground station per run -- a named preset or a custom lat/lon."""
    g = parser.add_argument_group("ground station (exactly one required)")
    g.add_argument("--ground-station", choices=list(GROUND_STATION_PRESETS), help="named preset")
    g.add_argument("--lat", type=float, help="custom ground station latitude, deg")
    g.add_argument("--lon", type=float, help="custom ground station longitude, deg")
    g.add_argument("--name", default="custom", help="label for a --lat/--lon station (used in filenames/titles)")


def resolve_ground_station(args, parser):
    """Returns (name, lat, lon) for exactly one station; errors otherwise.

    Never loops over multiple ground stations -- one run, one station.
    """
    if args.ground_station and (args.lat is not None or args.lon is not None):
        parser.error("pass --ground-station OR --lat/--lon, not both")
    if args.ground_station:
        lat, lon = GROUND_STATION_PRESETS[args.ground_station]
        return args.ground_station, lat, lon
    if args.lat is not None and args.lon is not None:
        return args.name, args.lat, args.lon
    parser.error("specify a ground station: --ground-station <preset> or --lat/--lon <deg>")


def make_config(lat, lon, min_elevation_deg):
    cfg = LinkLayerConfig()
    cfg.min_elevation_deg = min_elevation_deg
    cfg.ground_station.lat_deg, cfg.ground_station.lon_deg = lat, lon
    return cfg


def _atmospheric_db(elev_deg, freq_ghz, cfg):
    """Vectorized ITU-R P.676 slant-path loss (same call as propagation.atmospheric_loss_db)."""
    a = cfg.atmospheric
    clamped = np.maximum(elev_deg, a.min_elevation_for_airmass_deg)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="itur.*")
        att = itur.models.itu676.gaseous_attenuation_slant_path(
            freq_ghz * u.GHz,
            clamped * u.deg,
            a.water_vapor_density_g_m3 * u.g / u.m**3,
            a.pressure_hpa * u.hPa,
            a.temperature_k * u.K,
            mode="approx",
        )
    return np.asarray(att.value, dtype=float)


def _snr_db(range_km, elev_deg, cfg, freq_ghz):
    """Vectorized mirror of flame.link.budget._snr_db. Returns (snr, fspl, atmos, iono)."""
    # Same formula as propagation.free_space_path_loss_db, which is scalar-only.
    fspl = 20.0 * np.log10(range_km) + 20.0 * np.log10(freq_ghz) + cfg.pathloss.fspl_constant_db
    atmos = _atmospheric_db(elev_deg, freq_ghz, cfg)
    iono = ionospheric_loss_db(cfg.ionospheric.loss_db)
    total = fspl + atmos + iono + cfg.losses.implementation_loss_db + cfg.losses.rain_margin_db
    rx_dbw = cfg.satellite.eirp_dbw + cfg.ground_station.antenna_gain_dbi - total
    noise_dbw = 10.0 * math.log10(
        cfg.physical.boltzmann_j_per_k
        * cfg.ground_station.system_noise_temp_k
        * cfg.frequency.bandwidth_mhz
        * 1e6
    )
    return rx_dbw - noise_dbw, fspl, atmos, iono


def _shannon_mbps(snr_db, bandwidth_mhz):
    return bandwidth_mhz * np.log2(1.0 + 10.0 ** (snr_db / 10.0))


def satellite_series(cfg, sat_ecef_km, time_s):
    """Per-timestep downlink link budget for one satellite's full (T, 3)
    ECEF track against cfg.ground_station. Rows below the horizon
    (elevation <= 0) are dropped; rows below cfg.min_elevation_deg are kept
    with visible=False and 100% throughput loss, matching
    flame.link.budget.compute_link_budget.
    """
    freq, bw = cfg.frequency.downlink_ghz, cfg.frequency.bandwidth_mhz
    gs = geodetic_to_ecef_km(
        cfg.ground_station.lat_deg,
        cfg.ground_station.lon_deg,
        cfg.ground_station.alt_m,
        cfg.geometry.earth_radius_km,
    )
    gs_norm = np.linalg.norm(gs)
    zenith = gs / gs_norm

    los = sat_ecef_km - gs
    rng = np.linalg.norm(los, axis=1)
    elev = 90.0 - np.degrees(np.arccos(np.clip(los @ zenith / rng, -1.0, 1.0)))
    keep = elev > 0
    if not keep.any():
        return pd.DataFrame()
    sat_k, rng_k, elev_k, time_k = sat_ecef_km[keep], rng[keep], elev[keep], time_s[keep]
    visible = elev_k >= cfg.min_elevation_deg

    alt_k = np.maximum(np.linalg.norm(sat_k, axis=1) - gs_norm, cfg.min_reference_range_km)
    snr_ref, *_ = _snr_db(alt_k, np.full_like(alt_k, 90.0), cfg, freq)
    tput_ref = _shannon_mbps(snr_ref, bw)

    snr, fspl, atmos, iono = _snr_db(rng_k, elev_k, cfg, freq)
    tput = _shannon_mbps(snr, bw)
    total = fspl + atmos + iono + cfg.losses.implementation_loss_db + cfg.losses.rain_margin_db
    nan = np.nan

    return pd.DataFrame(
        {
            "time_s": time_k,
            "visible": visible,
            "elevation_deg": elev_k,
            "slant_range_km": rng_k,
            "one_way_delay_ms": rng_k / SPEED_OF_LIGHT_KM_S * 1e3,
            "rtt_ms": 2.0 * rng_k / SPEED_OF_LIGHT_KM_S * 1e3,
            "fspl_db": np.where(visible, fspl, nan),
            "atmospheric_loss_db": np.where(visible, atmos, nan),
            "ionospheric_loss_db": np.where(visible, iono, nan),
            "total_loss_db": np.where(visible, total, nan),
            "snr_db": np.where(visible, snr, nan),
            "snr_db_zenith_ref": snr_ref,
            "snr_loss_db": np.where(visible, snr_ref - snr, nan),
            "throughput_mbps": np.where(visible, tput, 0.0),
            "throughput_mbps_zenith_ref": tput_ref,
            "throughput_loss_mbps": np.where(visible, np.maximum(tput_ref - tput, 0.0), tput_ref),
            "throughput_loss_pct": np.where(
                visible, np.maximum(0.0, (tput_ref - tput) / tput_ref * 100.0), 100.0
            ),
        }
    )
