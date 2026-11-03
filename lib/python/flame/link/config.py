# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Configurable link-layer parameters for the RF budget.

Frequency / power / antenna-gain defaults are grounded in a real Ku-band
NGSO satellite system's public FCC filings (pulled via web search,
2026-09-20):
  - Downlink 10.7-12.7 GHz / Uplink 14.0-14.5 GHz
    (FCC STA application 1423-EX-ST-2024,
    https://apps.fcc.gov/els/GetAtt.html?id=355143)
  - 240 MHz per channel, 8 channels, 10 MHz guard bands
    (Ku-band downlink channelization measurements, arXiv:2210.11578,
    https://arxiv.org/pdf/2210.11578)
  - Max EIRP 38.2 dBW; user-terminal Rx/Tx antenna gain 35.8/37.2 dBi at
    boresight, 31.5/32.2 dBi at max slant (same FCC STA filing)

Atmospheric loss is a real ITU-R P.676 gaseous-attenuation computation via
the ITU-Rpy library. Ionospheric loss is a fixed value.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeometryConfig:
    # Mean Earth radius 6371.0088 km
    earth_radius_km: float = 6371.0


@dataclass
class PhysicalConfig:
    # Boltzmann constant, J/K -- CODATA 2018 exact value (SI redefinition):
    # https://physics.nist.gov/cgi-bin/cuu/Value?k
    boltzmann_j_per_k: float = 1.380649e-23


@dataclass
class PathLossConfig:
    # Free-space path loss constant term for FSPL(dB) = 20log10(d_km) +
    # 20log10(f_GHz) + fspl_constant_db, derived from the Friis
    # transmission equation FSPL = (4*pi*d*f/c)^2 with d in km and f in GHz:
    # 20log10(4*pi*1e3*1e9/c) = 92.45 for c = 299,792,458 m/s. See H.T.
    # Friis, "A Note on a Simple Transmission Formula," Proc. IRE,
    # 34(5):254-256, 1946 (https://ieeexplore.ieee.org/document/1697062/)
    # and ITU-R P.525-5, "Calculation of free-space attenuation"
    # (https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.525-5-202411-I!!PDF-E.pdf).
    fspl_constant_db: float = 92.45


@dataclass
class GroundStationConfig:
    # Default: Svalbard Satellite Station (SvalSat), Norway, 78.9243 N,
    # 11.9231 E -- a real, identifiable placeholder location, not a claim
    # about any specific gateway site, chosen because it's empirically a
    # near-optimal site for a near-polar LEO constellation: a grid search
    # over this repo's own _metadata/leo/ecef.npz trajectory data found
    # +-80 deg latitude gives ~3x the visible satellites and total contact
    # time of an equatorial/mid-latitude site (e.g. 61 -> 180 of 300
    # satellites ever visible at a 25 deg elevation mask), because
    # near-polar orbital planes converge near the poles -- exactly why real
    # polar-orbiting satellite operators site ground stations at Svalbard
    # or McMurdo. Override with your own ground station's coordinates.
    lat_deg: float = 78.9243
    lon_deg: float = 11.9231
    alt_m: float = 0.0
    antenna_gain_dbi: float = 35.8  # FCC-filed Rx boresight gain
    system_noise_temp_k: float = 200.0  # representative Ku-band VSAT receiver


@dataclass
class SatelliteRfConfig:
    eirp_dbw: float = 38.2  # FCC-filed max EIRP
    antenna_gain_dbi: float = 32.2  # FCC-filed max-slant Tx gain


@dataclass
class FrequencyConfig:
    downlink_ghz: float = 11.7  # mid of FCC 10.7-12.7 GHz DL band
    uplink_ghz: float = 14.25  # mid of FCC 14.0-14.5 GHz UL band
    bandwidth_mhz: float = 240.0  # FCC-filed per-channel bandwidth


@dataclass
class LossConfig:
    implementation_loss_db: float = 1.0
    rain_margin_db: float = 0.0  # optional additive margin, off by default


@dataclass
class AtmosphericConfig:
    # Atmospheric loss is always computed via ITU-R P.676 gaseous
    # attenuation, using the ITU-Rpy library
    # (https://itu-rpy.readthedocs.io/en/latest/, a core dependency of this
    # package -- not optional). See propagation.py's `atmospheric_loss_db`.
    #
    # Elevation (deg) floor: ITU-Rpy's "approx" method is only valid for
    # elevation in [5, 90] deg, so elevation is clamped up to this value
    # first. LinkLayerConfig.min_elevation_deg would normally keep real
    # callers well above this anyway.
    min_elevation_for_airmass_deg: float = 5.0
    # ICAO/US Standard Atmosphere sea-level defaults (15degC, 1013.25 hPa)
    # plus a representative mid-latitude water vapor density; override for
    # a specific ground station's climate.
    water_vapor_density_g_m3: float = 7.5
    pressure_hpa: float = 1013.25
    temperature_k: float = 288.15


@dataclass
class IonosphericConfig:
    # Fixed ionospheric loss margin, in dB -- not derived from a TEC/
    # frequency formula (see propagation.py's docstring for why). Default
    # is a representative Ku-band figure; override directly for a
    # different value rather than tuning formula inputs.
    loss_db: float = 0.005


def _sub(cls_, raw_sub: Optional[dict], key_map: dict):
    """Build a nested dataclass from a raw dict, keyed by wire-format alias.

    Any alias missing or explicitly None falls back to that field's
    dataclass default -- callers only need to specify the fields they want
    to override.
    """
    raw_sub = raw_sub or {}
    kwargs = {
        field_name: raw_sub[alias]
        for alias, field_name in key_map.items()
        if raw_sub.get(alias) is not None
    }
    return cls_(**kwargs)


@dataclass
class LinkLayerConfig:
    # When False (default), link-budget results are telemetry-only
    enabled: bool = False
    affects_delay: bool = False
    # When to compute the link budget:
    #   "precompute" (default) -- compute the whole satellite trajectory's
    #     link budget once, at load time if full trajectory is already known.
    #   "per_round" -- call compute_link_budget from scratch every round
    #     instead of precomputing.
    # Any other value falls back to "precompute".
    compute_mode: str = "precompute"
    min_elevation_deg: float = 25.0  # typical LEO satellite elevation mask (approximate)
    # Numerical floor (km) under the "zenith reference" slant range, so the
    # reference-case computation never divides by (near-)zero for a
    # satellite whose ECEF position happens to coincide with the ground
    # station's zenith point. Purely a numerical guard, not a physical
    # parameter -- default is far below any real LEO altitude.
    min_reference_range_km: float = 1.0
    # Shared LEO position dataset -- same directory the existing
    # satellite_coordinates_path (geodetic.npz, lat/lon only) already points
    # at; this is the sibling ecef.npz, needed for altitude/slant range.
    satellite_ecef_path: str = "lib/python/examples/_metadata/leo/ecef.npz"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    pathloss: PathLossConfig = field(default_factory=PathLossConfig)
    ground_station: GroundStationConfig = field(default_factory=GroundStationConfig)
    satellite: SatelliteRfConfig = field(default_factory=SatelliteRfConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    atmospheric: AtmosphericConfig = field(default_factory=AtmosphericConfig)
    ionospheric: IonosphericConfig = field(default_factory=IonosphericConfig)

    @property
    def precompute(self) -> bool:
        return self.compute_mode != "per_round"

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "LinkLayerConfig":
        """Build from the raw `hyperparameters.linkLayer` dict. Any subset
        of fields may be present; everything else falls back to the
        documented default above."""
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            affects_delay=bool(raw.get("affectsDelay", False)),
            compute_mode=str(raw.get("computeMode", cls.compute_mode)),
            min_elevation_deg=float(raw.get("minElevationDeg", cls.min_elevation_deg)),
            min_reference_range_km=float(
                raw.get("minReferenceRangeKm", cls.min_reference_range_km)
            ),
            satellite_ecef_path=str(raw.get("satelliteEcefPath", cls.satellite_ecef_path)),
            geometry=_sub(
                GeometryConfig,
                raw.get("geometry"),
                {"earthRadiusKm": "earth_radius_km"},
            ),
            physical=_sub(
                PhysicalConfig,
                raw.get("physical"),
                {"boltzmannJPerK": "boltzmann_j_per_k"},
            ),
            pathloss=_sub(
                PathLossConfig,
                raw.get("pathloss"),
                {"fsplConstantDb": "fspl_constant_db"},
            ),
            ground_station=_sub(
                GroundStationConfig,
                raw.get("groundStation"),
                {
                    "latDeg": "lat_deg",
                    "lonDeg": "lon_deg",
                    "altM": "alt_m",
                    "antennaGainDbi": "antenna_gain_dbi",
                    "systemNoiseTempK": "system_noise_temp_k",
                },
            ),
            satellite=_sub(
                SatelliteRfConfig,
                raw.get("satellite"),
                {"eirpDbw": "eirp_dbw", "antennaGainDbi": "antenna_gain_dbi"},
            ),
            frequency=_sub(
                FrequencyConfig,
                raw.get("frequency"),
                {
                    "downlinkGhz": "downlink_ghz",
                    "uplinkGhz": "uplink_ghz",
                    "bandwidthMhz": "bandwidth_mhz",
                },
            ),
            losses=_sub(
                LossConfig,
                raw.get("losses"),
                {
                    "implementationLossDb": "implementation_loss_db",
                    "rainMarginDb": "rain_margin_db",
                },
            ),
            atmospheric=_sub(
                AtmosphericConfig,
                raw.get("atmospheric"),
                {
                    "minElevationForAirmassDeg": "min_elevation_for_airmass_deg",
                    "waterVaporDensityGM3": "water_vapor_density_g_m3",
                    "pressureHpa": "pressure_hpa",
                    "temperatureK": "temperature_k",
                },
            ),
            ionospheric=_sub(
                IonosphericConfig,
                raw.get("ionospheric"),
                {"lossDb": "loss_db"},
            ),
        )
