# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Ground-station / satellite-link gate for the async_cifar10 LEO framing.

Wires flame.availability.rf_link_budget's physics onto the constellation
already checked into examples/_metadata/leo/ecef.npz (300 satellites, the
same one trainer/pytorch/main.py loads via satellite_coordinates_path) and a
fixed ground-station position.

The registry-based end-to-satellite mapping exists because a channel `end`
id equals the trainer's MQTT task_id (a per-run hex string from
trainer_registry.yaml), not its integer trainer_id -- so a live lookup
through the same registry the launcher used to assign satellite_index is
required to go from `end` back to a row in ecef.npz.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pyproj import Transformer

from flame.availability.rf_link_budget import (
    LinkBudgetConfig,
    shannon_throughput_mbps,
    throughput_loss_fraction,
    total_path_loss_db,
)

logger = logging.getLogger(__name__)

_GEODETIC_TO_ECEF = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)


def link_budget_enabled(link_budget_cfg: dict) -> bool:
    """Parse the `enabled` flag, tolerating both a real bool and this
    codebase's "True"/"False" string convention (str(x) is never falsy for
    a non-empty string, so a plain `bool(...)` would misparse "False")."""
    value = (link_budget_cfg or {}).get("enabled", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


@dataclass(frozen=True)
class LinkMetrics:
    eligible: bool
    reason: str
    distance_km: Optional[float] = None
    elevation_deg: Optional[float] = None
    path_loss_db: Optional[float] = None
    throughput_loss_frac: Optional[float] = None
    throughput_mbps: Optional[float] = None
    reference_throughput_mbps: Optional[float] = None


@dataclass(frozen=True)
class GroundStation:
    """Fixed ground-station position. lat/lon in degrees, alt in meters."""

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    @property
    def ecef_km(self) -> np.ndarray:
        x_m, y_m, z_m = _GEODETIC_TO_ECEF.transform(self.lon_deg, self.lat_deg, self.alt_m)
        return np.array([x_m, y_m, z_m]) / 1000.0

    @classmethod
    def from_dict(cls, d: dict) -> "GroundStation":
        return cls(
            lat_deg=float(d.get("lat_deg", 0.0)),
            lon_deg=float(d.get("lon_deg", 0.0)),
            alt_m=float(d.get("alt_m", 0.0)),
        )


def range_and_elevation(gs_ecef_km: np.ndarray, sat_ecef_km: np.ndarray) -> tuple:
    """Slant range (km) and elevation angle (deg) from ground station to
    satellite. Spherical-Earth approximation (zenith = radial direction at
    the ground station's own ECEF position) -- adequate given the
    atmospheric/ionospheric terms are themselves coarse approximations.
    """
    los = sat_ecef_km - gs_ecef_km
    distance_km = float(np.linalg.norm(los))
    zenith = gs_ecef_km / np.linalg.norm(gs_ecef_km)
    los_unit = los / distance_km
    cos_zenith_angle = float(np.clip(np.dot(zenith, los_unit), -1.0, 1.0))
    zenith_angle_deg = np.degrees(np.arccos(cos_zenith_angle))
    elevation_deg = 90.0 - zenith_angle_deg
    return distance_km, elevation_deg


def load_task_id_to_satellite_index(registry_path: Path) -> dict:
    """{task_id_hex: satellite_index} from trainer_registry.yaml.

    satellite_index = trainer_id - 1, mirroring
    flame.launch.spawner.ConfigGenerator.generate_trainer_config's
    hyperparameters.satellite_index assignment -- the two must agree since
    ecef.npz's satellite axis is 0-indexed by (trainer_id - 1).
    """
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    mapping = {}
    for meta in registry.get("trainers", {}).values():
        mapping[str(meta["task_id"])] = int(meta["trainer_id"]) - 1
    return mapping


class SatelliteLinkGate:
    """Evaluates RF link quality between the ground station and a satellite
    (identified by its channel `end` / task_id) at a given timeline second.
    """

    def __init__(
        self,
        ecef_path: Path,
        ground_station: GroundStation,
        task_id_to_satellite_index: dict,
        config: LinkBudgetConfig,
    ):
        data = np.load(ecef_path)
        self._ecef_km = data["ecef_km"]  # [T, N, 3]
        self._n_timesteps = self._ecef_km.shape[0]
        self.ground_station = ground_station
        self._gs_ecef_km = ground_station.ecef_km
        self.task_index = task_id_to_satellite_index
        self.config = config
        self._ref_loss_db = config.reference_loss_db()

    @classmethod
    def from_config(
        cls,
        ecef_path: Path,
        registry_path: Path,
        ground_station_dict: dict,
        link_budget_dict: dict,
    ) -> "SatelliteLinkGate":
        return cls(
            ecef_path=ecef_path,
            ground_station=GroundStation.from_dict(ground_station_dict or {}),
            task_id_to_satellite_index=load_task_id_to_satellite_index(registry_path),
            config=LinkBudgetConfig.from_dict(link_budget_dict or {}),
        )

    @classmethod
    def from_config_for_satellite(
        cls,
        ecef_path: Path,
        ground_station_dict: dict,
        link_budget_dict: dict,
    ) -> "SatelliteLinkGate":
        """Trainer-side variant: no task_id registry needed since a trainer
        already knows its own satellite_index (call evaluate_index directly).
        """
        return cls(
            ecef_path=ecef_path,
            ground_station=GroundStation.from_dict(ground_station_dict or {}),
            task_id_to_satellite_index={},
            config=LinkBudgetConfig.from_dict(link_budget_dict or {}),
        )

    def _satellite_ecef_km(self, satellite_index: int, t_s: float) -> np.ndarray:
        idx = min(max(int(round(t_s)), 0), self._n_timesteps - 1)
        return self._ecef_km[idx, satellite_index]

    def evaluate_index(self, satellite_index: int, t_s: float) -> LinkMetrics:
        """Core physics evaluation, keyed directly by satellite (row) index
        into ecef.npz -- shared by both evaluate() (aggregator, keyed by
        channel `end`) and a trainer's own self-check (which already knows
        its own satellite_index and doesn't need the end->index registry).
        """
        sat_ecef_km = self._satellite_ecef_km(satellite_index, t_s)
        distance_km, elevation_deg = range_and_elevation(self._gs_ecef_km, sat_ecef_km)

        if elevation_deg < self.config.min_elevation_deg:
            return LinkMetrics(
                eligible=False,
                reason="below_horizon",
                distance_km=distance_km,
                elevation_deg=elevation_deg,
            )

        breakdown = total_path_loss_db(
            distance_km,
            elevation_deg,
            self.config.downlink_freq_mhz,
            self.config.atmo_zenith_db,
            self.config.tec_tecu,
        )
        extra_loss_db = breakdown.total_db - self._ref_loss_db
        loss_frac = throughput_loss_fraction(extra_loss_db, self.config.ref_snr_db)
        eligible = loss_frac <= self.config.throughput_loss_threshold
        ref_mbps = shannon_throughput_mbps(
            self.config.ref_snr_db, self.config.channel_bandwidth_hz
        )
        return LinkMetrics(
            eligible=eligible,
            reason="ok" if eligible else "throughput_loss_exceeded",
            distance_km=distance_km,
            elevation_deg=elevation_deg,
            path_loss_db=breakdown.total_db,
            throughput_loss_frac=loss_frac,
            throughput_mbps=ref_mbps * (1.0 - loss_frac),
            reference_throughput_mbps=ref_mbps,
        )

    def evaluate(self, end: str, t_s: float) -> LinkMetrics:
        satellite_index = self.task_index.get(str(end))
        if satellite_index is None:
            # Registry mismatch or an end the gate doesn't recognize (e.g. a
            # test double) -- never gate on missing data.
            return LinkMetrics(eligible=True, reason="unknown_end")
        return self.evaluate_index(satellite_index, t_s)

    def is_eligible(self, end: str, t_s: float) -> bool:
        return self.evaluate(end, t_s).eligible
