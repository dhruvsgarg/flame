# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Which ends can the ground station reach right now (direct routing).

Used by the aggregator when `linkLayer.direct.visibleOnly` is set: ends whose
satellite is below the elevation mask at selection time are kept out of the
selection pool. Satellites that never rise above the mask in the trajectory are
never visible, so they are excluded for the whole run.
"""

import logging
from typing import Callable, Dict, Iterable, Optional, Set

import numpy as np
import yaml

from flame.link.propagation import geodetic_to_ecef_km

logger = logging.getLogger(__name__)


def registry_satellite_map(path: str) -> Dict[str, int]:
    """task_id -> satellite index (trainer_id - 1) from a trainer registry yaml.

    Matches launch/spawner.py, which sets `satellite_index = trainer_id - 1`.
    Returns {} (with a warning) if the file can't be read.
    """
    try:
        with open(path, encoding="utf-8") as f:
            trainers = yaml.safe_load(f)["trainers"]
        return {str(t["task_id"]): int(t["trainer_id"]) - 1 for t in trainers.values()}
    except Exception as e:
        logger.warning(f"[LINK_VISIBLE_ONLY] could not read trainer registry {path!r}: {e}")
        return {}


def visibility_matrix(ecef_km, cfg, min_elevation_deg: Optional[float] = None):
    """bool [timestep, satellite]: above the elevation mask from the configured station.

    Pure geometry (the same `elevation >= min_elevation_deg` test the link budget uses
    for `visible`), vectorised, so it needs none of the per-satellite ITU series.
    """
    mask = cfg.min_elevation_deg if min_elevation_deg is None else min_elevation_deg
    gs = cfg.ground_station
    g = geodetic_to_ecef_km(gs.lat_deg, gs.lon_deg, gs.alt_m, cfg.geometry.earth_radius_km)
    los = np.asarray(ecef_km, dtype=float) - g
    up = g / np.linalg.norm(g)
    cos_zen = (los @ up) / np.linalg.norm(los, axis=2)
    elevation_deg = 90.0 - np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))
    return elevation_deg >= mask


def invisible_ends(
    visible,
    ends: Iterable[str],
    satellite_index_of: Callable[[str], Optional[int]],
    t_s: float,
) -> Set[str]:
    """Ends whose satellite is not visible at `t_s` seconds into the trajectory.

    `visible` is a `visibility_matrix`. `t_s` wraps modulo the trajectory length, the
    same way transfers do. Ends with no known satellite index are kept eligible.
    """
    n = visible.shape[0]
    t = int(t_s) % n if n else 0
    return {
        end
        for end in ends
        if (sat := satellite_index_of(end)) is not None and not visible[t, int(sat)]
    }
