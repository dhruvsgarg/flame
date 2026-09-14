# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""RF link-budget physics: free-space + atmospheric + ionospheric path loss,
and the throughput-loss fraction they imply relative to a reference SNR.

Pure functions, no I/O, no example-specific state -- callers (e.g.
examples/async_cifar10/leo/ground_station.py) supply distance/elevation and
own the config values. All loss values are in dB unless noted.
"""

import math
from dataclasses import dataclass


def fspl_db(distance_km: float, freq_mhz: float) -> float:
    """Free-space path loss (dB). Standard Friis-derived form for
    distance in km and frequency in MHz."""
    if distance_km <= 0 or freq_mhz <= 0:
        raise ValueError(f"distance_km and freq_mhz must be > 0, got {distance_km}, {freq_mhz}")
    return 20.0 * math.log10(distance_km) + 20.0 * math.log10(freq_mhz) + 32.44


def atmospheric_loss_db(elevation_deg: float, zenith_db: float = 0.2) -> float:
    """Gaseous (oxygen + water vapor) attenuation, ITU-R P.676 cosecant
    approximation: loss scales as 1/sin(elevation) relative to the
    zenith (90 deg) value. `zenith_db` is the clear-sky zenith loss at the
    link's frequency (~0.1-0.3 dB is typical at Ku-band).

    Elevation is clamped to a 5 deg floor -- the cosecant law diverges near
    the horizon and real links don't operate below a minimum elevation mask.
    """
    elev = max(elevation_deg, 5.0)
    return zenith_db / math.sin(math.radians(elev))


def ionospheric_loss_db(
    elevation_deg: float, freq_mhz: float, tec_tecu: float = 50.0
) -> float:
    """Ionospheric attenuation, simplified TEC-based model.

    Ionospheric absorption scales as ~1/freq^2 and, like the atmosphere,
    roughly as 1/sin(elevation) for the slant path. At Ku-band (>10 GHz)
    this term is negligible (<0.05 dB) -- it's kept here so the model is
    complete and reusable at lower bands (L/S-band) where it dominates.

    tec_tecu: total electron content in TEC units (1 TECU = 1e16 el/m^2).
    Coefficient (40.3 in SI) is the standard ionospheric refraction constant;
    scaled here into an empirical dB-loss proxy, not a phase-delay formula.
    """
    elev = max(elevation_deg, 5.0)
    tec_el_per_m2 = tec_tecu * 1e16
    loss_linear_proxy = 40.3 * tec_el_per_m2 / (freq_mhz * 1e6) ** 2
    loss_db = 10.0 * math.log10(1.0 + loss_linear_proxy)
    return loss_db / math.sin(math.radians(elev))


def total_path_loss_db(
    distance_km: float,
    elevation_deg: float,
    freq_mhz: float,
    atmo_zenith_db: float = 0.2,
    tec_tecu: float = 50.0,
) -> "PathLossBreakdown":
    """Sum of free-space + atmospheric + ionospheric loss, with breakdown."""
    fspl = fspl_db(distance_km, freq_mhz)
    atmo = atmospheric_loss_db(elevation_deg, atmo_zenith_db)
    iono = ionospheric_loss_db(elevation_deg, freq_mhz, tec_tecu)
    return PathLossBreakdown(fspl_db=fspl, atmospheric_db=atmo, ionospheric_db=iono)


def shannon_throughput_mbps(snr_db: float, bandwidth_hz: float) -> float:
    """Shannon capacity (Mbps) at the given SNR and channel bandwidth --
    the absolute-throughput counterpart to throughput_loss_fraction's ratio
    (which cancels bandwidth out; this is what makes the configured
    bandwidth, e.g. Starlink's 240 MHz Ku channel, load-bearing rather than
    a stored-but-unused constant)."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    return bandwidth_hz * math.log2(1.0 + snr_linear) / 1.0e6


def throughput_loss_fraction(
    extra_loss_db: float, ref_snr_db: float, floor: float = 0.0
) -> float:
    """Fractional Shannon-capacity loss caused by `extra_loss_db` of path
    loss beyond the reference condition `ref_snr_db` was measured/assumed at.

    C/C_ref = log2(1 + SNR) / log2(1 + SNR_ref), where
    SNR = SNR_ref / 10^(extra_loss_db/10) (extra loss reduces linear SNR
    proportionally; noise floor is treated as constant).

    Returns a fraction in [floor, 1.0]. extra_loss_db <= 0 (i.e. this link
    is no worse than the reference) returns 0.0 -- never a throughput *gain*.
    """
    if extra_loss_db <= 0:
        return max(0.0, floor)
    ref_snr_linear = 10.0 ** (ref_snr_db / 10.0)
    snr_linear = ref_snr_linear / (10.0 ** (extra_loss_db / 10.0))
    c_ref = math.log2(1.0 + ref_snr_linear)
    c = math.log2(1.0 + snr_linear)
    loss_frac = 1.0 - (c / c_ref)
    return min(1.0, max(floor, loss_frac))


@dataclass(frozen=True)
class PathLossBreakdown:
    fspl_db: float
    atmospheric_db: float
    ionospheric_db: float

    @property
    def total_db(self) -> float:
        return self.fspl_db + self.atmospheric_db + self.ionospheric_db


@dataclass(frozen=True)
class LinkBudgetConfig:
    """Configurable link-budget parameters. All fields overridable via
    hyperparameters.link_budget in an experiment/baseline config."""

    enabled: bool = False
    downlink_freq_mhz: float = 12000.0  # Ku-band downlink, ~10.7-12.7 GHz
    channel_bandwidth_hz: float = 240.0e6  # SpaceX FCC filing: 8x240 MHz Ku channels
    ref_snr_db: float = 20.0  # see docs/leo_link_budget.md for derivation
    ref_distance_km: float = 550.0  # reference slant range the ref_snr_db was set at
    ref_elevation_deg: float = 40.0  # reference elevation the ref_snr_db was set at
    atmo_zenith_db: float = 0.2
    tec_tecu: float = 50.0
    throughput_loss_threshold: float = 0.5  # drop if modeled loss exceeds this fraction
    min_elevation_deg: float = 5.0  # below this, satellite is not visible at all

    @classmethod
    def from_dict(cls, d: dict) -> "LinkBudgetConfig":
        """Build from a plain dict (e.g. config.hyperparameters.link_budget),
        ignoring unknown keys and falling back to defaults for missing ones.
        """
        known = {f: getattr(cls, f) for f in cls.__dataclass_fields__}
        merged = {**known, **{k: v for k, v in (d or {}).items() if k in known}}
        return cls(**merged)

    def reference_loss_db(self) -> float:
        """Path loss at the reference condition ref_snr_db was set at --
        subtracted from a candidate link's own path loss to get the *extra*
        loss that actually degrades throughput relative to that reference.
        """
        breakdown = total_path_loss_db(
            self.ref_distance_km,
            self.ref_elevation_deg,
            self.downlink_freq_mhz,
            self.atmo_zenith_db,
            self.tec_tecu,
        )
        return breakdown.total_db
