# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Mode-agnostic link-budget runtime, shared by trainer and aggregator.

`LinkLayerConfig.compute_mode` selects *when* a satellite's link budget is
computed -- "precompute" (whole trajectory computed once, then indexed) or
"per_round" (recomputed from scratch every call) -- but the branching
between those two modes was duplicated per caller. `LinkRuntime` is that
branch, written once: `result_for(satellite_index, timestep)` is the same
call whether the caller is a trainer (which only ever asks for its own one
satellite) or an aggregator (which asks for many, one per connected
trainer).

In "precompute" mode, a satellite's full timestep series is computed and
cached the first time that satellite_index is requested, not eagerly for
every satellite in the position file at construction time -- a trainer only
ever has one satellite_index, so this is identical to eager precompute for
it; an aggregator only pays for the satellites that actually connect.
"""

from typing import Dict, List, Optional

import numpy as np

from flame.link.budget import LinkBudgetResult, compute_link_budget, compute_link_budget_series
from flame.link.config import LinkLayerConfig
from flame.link.propagation import lookup_satellite_ecef


class LinkRuntime:
    def __init__(self, cfg: LinkLayerConfig, ecef_km: np.ndarray, direction: str = "downlink"):
        self.cfg = cfg
        self.ecef_km = ecef_km  # (timesteps, num_satellites, 3) km
        self.direction = direction
        self._series_cache: Dict[int, List[LinkBudgetResult]] = {}

    @classmethod
    def from_link_layer_dict(
        cls, raw: Optional[dict], direction: str = "downlink"
    ) -> Optional["LinkRuntime"]:
        cfg = LinkLayerConfig.from_dict(raw)
        if not cfg.enabled:
            return None
        ecef_km = np.load(cfg.satellite_ecef_path)["ecef_km"]
        return cls(cfg, ecef_km, direction)

    def result_for(self, satellite_index: int, timestep: int) -> LinkBudgetResult:
        t = min(max(int(timestep), 0), self.ecef_km.shape[0] - 1)
        if self.cfg.precompute:
            series = self._series_cache.get(satellite_index)
            if series is None:
                series = compute_link_budget_series(
                    self.ecef_km[:, satellite_index, :], self.cfg, self.direction
                )
                self._series_cache[satellite_index] = series
            return series[t]
        sat_ecef = lookup_satellite_ecef(self.ecef_km, satellite_index, t)
        return compute_link_budget(sat_ecef, self.cfg, self.direction)
