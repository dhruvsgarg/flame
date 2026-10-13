# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Ground-station <-> satellite RF link budget.

Combines free-space + atmospheric + ionospheric loss into an SNR figure,
reports SINR degradation relative to a best-case zenith (directly overhead)
pass at the same altitude/frequency/config, and converts both into a
Shannon-capacity throughput number so callers get a direct throughput-loss
percentage. See docs/link-layer.md for the full
call flow and a walked-through numeric example.

`compute_link_budget` computes one timestep; `compute_link_budget_series`
precomputes every timestep for one satellite in a single pass, since a
satellite's full trajectory (and therefore its whole link-budget time
series) is already known the moment its position file is loaded -- callers
that drive a round loop should precompute once and index by timestep
rather than calling `compute_link_budget` again every round.

Every numeric input to this module's formulas is a `LinkLayerConfig` field
(config.py) -- there are no fixed physical/model constants left in this
file. The two literals that remain (`90.0` for "zenith" and the `1e6`/`1e-6`
unit-conversion factors) are not tunable parameters: 90 degrees *is* the
definition of directly overhead (changing it would redefine what "the
reference pass" means, not configure the model), and the unit conversions
are dimensional bookkeeping (MHz->Hz, Hz->Mbps), not modeling choices.

Formula references:
  - Link budget / SNR: EIRP + Gr - losses - N, the  satellite-link
    budget https://ccsds.org/Pubs/401x0b32.pdf
    (https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-14-202308-I!!PDF-E.pdf).
  - Thermal noise floor N = k*T*B (Johnson-Nyquist noise power in abandwidth B)
  - Shannon-Hartley channel capacity C = B*log2(1+SNR)
    https://doi.org/10.1002/j.1538-7305.1948.tb01338.x.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from flame.link.config import LinkLayerConfig
from flame.link.propagation import (
    atmospheric_loss_db as compute_atmospheric_loss_db,
    elevation_deg as compute_elevation_deg,
    free_space_path_loss_db as compute_free_space_path_loss_db,
    geodetic_to_ecef_km,
    ionospheric_loss_db as compute_ionospheric_loss_db,
    slant_range_km as compute_slant_range_km,
)

# The definition of "directly overhead" for the zenith-reference case below
# -- not a tunable model parameter (see module docstring).
_ZENITH_ELEVATION_DEG = 90.0


@dataclass
class LinkBudgetResult:
    elevation_deg: float
    slant_range_km: float
    visible: bool
    fspl_db: float
    atmospheric_loss_db: float
    ionospheric_loss_db: float
    total_loss_db: float
    snr_db: float
    snr_db_zenith_ref: float
    sinr_degradation_db: float
    throughput_mbps: float
    throughput_mbps_zenith_ref: float
    throughput_loss_pct: float


def _snr_db(
    slant_range_km: float, elevation_deg: float, freq_ghz: float, cfg: LinkLayerConfig
) -> Tuple[float, float, float, float, float]:
    """Returns (snr_db, free_space_path_loss_db, atmospheric_loss_db, ionospheric_loss_db, total_path_loss_db)."""
    free_space_path_loss_db = compute_free_space_path_loss_db(
        slant_range_km, freq_ghz, cfg.pathloss.fspl_constant_db
    )
    atmospheric_loss_db = compute_atmospheric_loss_db(
        elevation_deg,
        freq_ghz,
        cfg.atmospheric.water_vapor_density_g_m3,
        cfg.atmospheric.pressure_hpa,
        cfg.atmospheric.temperature_k,
        cfg.atmospheric.min_elevation_for_airmass_deg,
    )
    ionospheric_loss_db = compute_ionospheric_loss_db(cfg.ionospheric.loss_db)
    total_path_loss_db = (
        free_space_path_loss_db
        + atmospheric_loss_db
        + ionospheric_loss_db
        + cfg.losses.implementation_loss_db
        + cfg.losses.rain_margin_db
    )

    # Link budget equation: received_power = EIRP + Gr - total_path_loss (all dB/dBW/dBi).
    received_power_dbw = (
        cfg.satellite.eirp_dbw + cfg.ground_station.antenna_gain_dbi - total_path_loss_db
    )
    bandwidth_hz = cfg.frequency.bandwidth_mhz * 1e6
    # Johnson-Nyquist thermal noise floor: N = k*T*B, in dBW.
    thermal_noise_floor_dbw = 10.0 * math.log10(
        cfg.physical.boltzmann_j_per_k * cfg.ground_station.system_noise_temp_k * bandwidth_hz
    )
    snr_db = received_power_dbw - thermal_noise_floor_dbw
    return snr_db, free_space_path_loss_db, atmospheric_loss_db, ionospheric_loss_db, total_path_loss_db


def _shannon_mbps(snr_db: float, bandwidth_mhz: float) -> float:
    """Shannon-Hartley channel capacity, C = B*log2(1+SNR), in Mbps."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    bandwidth_hz = bandwidth_mhz * 1e6
    return bandwidth_hz * math.log2(1.0 + snr_linear) / 1e6


def compute_link_budget(
    sat_ecef_km: np.ndarray,
    cfg: LinkLayerConfig,
    direction: str = "downlink",
) -> LinkBudgetResult:
    """Compute one round's link budget for a satellite at `sat_ecef_km`
    against `cfg.ground_station`.
    """
    freq_ghz = cfg.frequency.downlink_ghz if direction == "downlink" else cfg.frequency.uplink_ghz

    ground_station_ecef_km = geodetic_to_ecef_km(
        cfg.ground_station.lat_deg,
        cfg.ground_station.lon_deg,
        cfg.ground_station.alt_m,
        cfg.geometry.earth_radius_km,
    )
    elevation_deg = compute_elevation_deg(sat_ecef_km, ground_station_ecef_km)
    slant_range_km = compute_slant_range_km(sat_ecef_km, ground_station_ecef_km)

    # Zenith reference: same altitude, directly overhead -- minimum possible
    # slant range and elevation=90 deg (airmass=1), i.e. the best-case pass
    # for this satellite/ground-station pair. Everything else is reported
    # as degradation relative to this.
    satellite_altitude_km = max(
        float(np.linalg.norm(np.asarray(sat_ecef_km, dtype=float)) - np.linalg.norm(ground_station_ecef_km)),
        cfg.min_reference_range_km,
    )
    zenith_reference_snr_db, *_ = _snr_db(satellite_altitude_km, _ZENITH_ELEVATION_DEG, freq_ghz, cfg)
    zenith_reference_throughput_mbps = _shannon_mbps(zenith_reference_snr_db, cfg.frequency.bandwidth_mhz)

    if elevation_deg < cfg.min_elevation_deg:
        return LinkBudgetResult(
            elevation_deg=elevation_deg,
            slant_range_km=slant_range_km,
            visible=False,
            fspl_db=float("nan"),
            atmospheric_loss_db=float("nan"),
            ionospheric_loss_db=float("nan"),
            total_loss_db=float("nan"),
            snr_db=float("-inf"),
            snr_db_zenith_ref=zenith_reference_snr_db,
            sinr_degradation_db=float("inf"),
            throughput_mbps=0.0,
            throughput_mbps_zenith_ref=zenith_reference_throughput_mbps,
            throughput_loss_pct=100.0,
        )

    snr_db, free_space_path_loss_db, atmospheric_loss_db, ionospheric_loss_db, total_path_loss_db = _snr_db(
        slant_range_km, elevation_deg, freq_ghz, cfg
    )
    throughput_mbps = _shannon_mbps(snr_db, cfg.frequency.bandwidth_mhz)
    sinr_degradation_db = zenith_reference_snr_db - snr_db
    throughput_loss_pct = (
        max(0.0, (zenith_reference_throughput_mbps - throughput_mbps) / zenith_reference_throughput_mbps * 100.0)
        if zenith_reference_throughput_mbps > 0
        else 0.0
    )

    return LinkBudgetResult(
        elevation_deg=elevation_deg,
        slant_range_km=slant_range_km,
        visible=True,
        fspl_db=free_space_path_loss_db,
        atmospheric_loss_db=atmospheric_loss_db,
        ionospheric_loss_db=ionospheric_loss_db,
        total_loss_db=total_path_loss_db,
        snr_db=snr_db,
        snr_db_zenith_ref=zenith_reference_snr_db,
        sinr_degradation_db=sinr_degradation_db,
        throughput_mbps=throughput_mbps,
        throughput_mbps_zenith_ref=zenith_reference_throughput_mbps,
        throughput_loss_pct=throughput_loss_pct,
    )


def compute_link_budget_series(
    sat_ecef_km_series: np.ndarray,
    cfg: LinkLayerConfig,
    direction: str = "downlink",
) -> List[LinkBudgetResult]:
    return [
        compute_link_budget(sat_ecef_km_series[t], cfg, direction)
        for t in range(sat_ecef_km_series.shape[0])
    ]
