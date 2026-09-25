# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Ground-station / satellite geometry and RF propagation loss models.

Everything a caller needs to go from two positions to a total path loss in
dB lives here: geometry (ECEF conversion, slant range, elevation angle) and
the three loss mechanisms `flame.link.budget` combines into an SNR figure
(free-space, atmospheric, ionospheric). SNR/throughput/degradation math
itself lives in `budget.py`, not here -- this module only ever returns a
distance, an angle, or a loss in dB.

References, one per section below:
  - Free-space path loss: Friis Equation ITU-R P.525-5
    (https://ieeexplore.ieee.org/document/1697062/), standardized as
    "Calculation of free-space attenuation"
    (https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.525-5-202411-I!!PDF-E.pdf).
  - Atmospheric (tropospheric gas) loss: `atmospheric_loss_db`, ITU-R
    P.676, "Attenuation by atmospheric gases"
    (https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-11-201609-I!!PDF-E.pdf),
    computed directly by the ITU-Rpy library https://itu-rpy.readthedocs.io/en/latest/,
  - Ionospheric loss: a fixed, directly-configured margin,
"""

import warnings

import astropy.units as u
import itur
import numpy as np

# ---------------------------------------------------------------------------
# Geometry -- spherical-Earth ECEF math. Uses the same convention as
# examples/fmow/setup/captures.py (EARTH_RADIUS_KM = 6371.0) rather than a
# full WGS84 ellipsoid, so slant range / elevation stay consistent with the
# capture-event geometry already used elsewhere in the repo. Good to a few
# km at LEO altitudes -- not survey-grade. Config field:
# LinkLayerConfig.geometry.earth_radius_km.
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def geodetic_to_ecef_km(
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Spherical-Earth geodetic (lat, lon, alt) -> ECEF, in km."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = earth_radius_km + alt_m / 1000.0
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def slant_range_km(sat_ecef_km: np.ndarray, gs_ecef_km: np.ndarray) -> float:
    """Straight-line distance between satellite and ground station, km."""
    return float(np.linalg.norm(np.asarray(sat_ecef_km, dtype=float) - np.asarray(gs_ecef_km, dtype=float)))


def elevation_deg(sat_ecef_km: np.ndarray, gs_ecef_km: np.ndarray) -> float:
    """Elevation angle of the satellite above the ground station's local horizon.

    90 deg = directly overhead, 0 deg = on the horizon, negative = below it
    (not visible). Pure vector geometry (dot product between the
    line-of-sight and local-vertical/zenith directions) -- no additional
    physical constants beyond the ECEF positions themselves, so there is
    nothing here to make configurable.
    """
    sat = np.asarray(sat_ecef_km, dtype=float)
    gs = np.asarray(gs_ecef_km, dtype=float)
    line_of_sight = sat - gs
    los_norm = np.linalg.norm(line_of_sight)
    gs_norm = np.linalg.norm(gs)
    if los_norm == 0 or gs_norm == 0:
        return 90.0
    zenith = gs / gs_norm  # local "up" unit vector at the ground station
    cos_zenith_angle = float(np.dot(line_of_sight / los_norm, zenith))
    zenith_angle_deg = np.degrees(np.arccos(np.clip(cos_zenith_angle, -1.0, 1.0)))
    return 90.0 - zenith_angle_deg


def lookup_satellite_ecef(ecef_km_array: np.ndarray, satellite_index: int, timestep: int) -> np.ndarray:
    """Look up one satellite's ECEF position at one timestep.

    `ecef_km_array` is `(timesteps, satellites, xyz)` km -- the same
    indexing convention used by examples/fmow/setup/captures.py and by the
    existing lat/lon lookups in both fmow's and async_cifar10's trainer
    main.py (`coords[timestep, satellite_index, :]`).
    """
    t = min(max(int(timestep), 0), ecef_km_array.shape[0] - 1)
    return np.asarray(ecef_km_array[t, satellite_index, :], dtype=float)


# ---------------------------------------------------------------------------
# Free-space path loss (Friis transmission equation). Config field:
# LinkLayerConfig.pathloss.fspl_constant_db.
# ---------------------------------------------------------------------------

# Constant term for FSPL(dB) = 20log10(d_km) + 20log10(f_GHz) +
# FSPL_CONSTANT_DB, derived from FSPL = (4*pi*d*f/c)^2 with d in km, f in
# GHz, c = speed of light: 20log10(4*pi*1e3*1e9/299792458) = 92.45.
FSPL_CONSTANT_DB = 92.45


def free_space_path_loss_db(
    distance_km: float, freq_ghz: float, fspl_constant_db: float = FSPL_CONSTANT_DB
) -> float:
    """FSPL(dB) = 20*log10(d_km) + 20*log10(f_GHz) + fspl_constant_db.

    Standard Friis free-space path loss with distance in km and frequency
    in GHz.
    """
    if distance_km <= 0 or freq_ghz <= 0:
        raise ValueError("distance_km and freq_ghz must be positive")
    return 20.0 * np.log10(distance_km) + 20.0 * np.log10(freq_ghz) + fspl_constant_db


# ---------------------------------------------------------------------------
# Atmospheric (tropospheric gas) attenuation via ITU-R P.676, computed
# directly by the ITU-Rpy library (https://itu-rpy.readthedocs.io/en/latest/,
# a core dependency of this package -- not optional). No configured zenith
# figure to scale -- the "approx" method takes frequency, elevation, water
# vapor density, pressure and temperature directly and returns the
# slant-path loss already elevation-scaled. Config fields:
# LinkLayerConfig.atmospheric.{min_elevation_for_airmass_deg,
# water_vapor_density_g_m3, pressure_hpa, temperature_k}.
# ---------------------------------------------------------------------------

# Elevation (deg) floor: ITU-Rpy's "approx" method is only valid for
# elevation in [5, 90] deg, so elevation is clamped up to this before
# calling it. LinkLayerConfig.min_elevation_deg would normally keep real
# callers well above this anyway.
MIN_ELEVATION_FOR_AIRMASS_DEG = 5.0


def atmospheric_loss_db(
    elevation_deg: float,
    freq_ghz: float,
    water_vapor_density_g_m3: float,
    pressure_hpa: float,
    temperature_k: float,
    min_elevation_for_airmass_deg: float = MIN_ELEVATION_FOR_AIRMASS_DEG,
) -> float:
    """Slant-path gaseous absorption via ITU-R P.676, in dB. Computed
    directly by the ITU-Rpy library
    (https://itu-rpy.readthedocs.io/en/latest/,
    `itur.models.itu676.gaseous_attenuation_slant_path`).
    """
    if elevation_deg <= 0:
        return float("inf")
    clamped_elev = max(elevation_deg, min_elevation_for_airmass_deg)
    with warnings.catch_warnings():
        # ITU-Rpy warns whenever elevation is outside (5, 90) even at the
        # boundary values this module's own clamp produces -- expected and
        # harmless on essentially every call (this runs once per satellite
        # per timestep in the default precompute path), so silence it here
        # rather than at every one of thousands of calls per trainer run.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="itur.*")
        attenuation = itur.models.itu676.gaseous_attenuation_slant_path(
            freq_ghz * u.GHz,
            clamped_elev * u.deg,
            water_vapor_density_g_m3 * u.g / u.m**3,
            pressure_hpa * u.hPa,
            temperature_k * u.K,
            mode="approx",
        )
    return float(attenuation.value)


# ---------------------------------------------------------------------------
# Ionospheric attenuation -- a fixed, directly-configured margin rather than
# a computed formula. Ionospheric effects (dominant at L/S-band, governed
# by Total Electron Content and frequency -- see ITU-R P.531, "Ionospheric
# propagation data and prediction methods required for the design of
# satellite networks and systems",
# https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.531-16-202509-I!!PDF-E.pdf)
# are small enough at Ku/Ka band that a configured constant is simpler and
# no less accurate than a from-scratch TEC/frequency model would be at
# this frequency range. Config field: LinkLayerConfig.ionospheric.loss_db.
# ---------------------------------------------------------------------------

# Representative Ku-band ionospheric margin, in dB -- a fixed default, not
# derived from a formula. Set hyperparameters.linkLayer.ionospheric.lossDb
# directly for a different value.
IONOSPHERIC_LOSS_DB = 0.005


def ionospheric_loss_db(loss_db: float = IONOSPHERIC_LOSS_DB) -> float:
    return float(loss_db)
