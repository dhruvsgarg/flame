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

from typing import Dict, List, Optional, Tuple

import numpy as np

from flame.link.budget import LinkBudgetResult, compute_link_budget, compute_link_budget_series
from flame.link.config import LinkLayerConfig
from flame.link.propagation import geodetic_to_ecef_km, lookup_satellite_ecef
from flame.link.transfer import TransferEstimate, estimate_transfer


class LinkRuntime:
    def __init__(self, cfg: LinkLayerConfig, ecef_km: np.ndarray, direction: str = "downlink"):
        self.cfg = cfg
        self.ecef_km = ecef_km  # (timesteps, num_satellites, 3) km
        self.direction = direction
        self._series_cache: Dict[int, List[LinkBudgetResult]] = {}
        self._rate_cache: Dict[int, np.ndarray] = {}
        self._isl = None  # latency.npz arrays, loaded on first isl transfer
        self._relay_cache: Dict[int, Tuple[int, float]] = {}  # k -> (gateway, rate bps)
        self._gateway: Optional[np.ndarray] = None  # relay gateway per timestep, built on first isl relay

    @classmethod
    def from_link_layer_dict(
        cls, raw: Optional[dict], direction: str = "downlink"
    ) -> Optional["LinkRuntime"]:
        cfg = LinkLayerConfig.from_dict(raw)
        if not cfg.enabled:
            return None
        ecef_km = np.load(cfg.satellite_ecef_path)["ecef_km"]
        return cls(cfg, ecef_km, direction)

    def _series(self, satellite_index: int) -> List[LinkBudgetResult]:
        series = self._series_cache.get(satellite_index)
        if series is None:
            series = compute_link_budget_series(
                self.ecef_km[:, satellite_index, :], self.cfg, self.direction
            )
            self._series_cache[satellite_index] = series
        return series

    def result_for(self, satellite_index: int, timestep: int) -> LinkBudgetResult:
        t = min(max(int(timestep), 0), self.ecef_km.shape[0] - 1)
        if self.cfg.precompute:
            return self._series(satellite_index)[t]
        sat_ecef = lookup_satellite_ecef(self.ecef_km, satellite_index, t)
        return compute_link_budget(sat_ecef, self.cfg, self.direction)

    def _own_rate_bps(self, satellite_index: int) -> np.ndarray:
        rate_bps = self._rate_cache.get(satellite_index)
        if rate_bps is None:
            eff = self.cfg.transfer.efficiency
            rate_bps = np.array([r.throughput_mbps * 1e6 * eff for r in self._series(satellite_index)])
            self._rate_cache[satellite_index] = rate_bps
        return rate_bps

    def _isl_arrays(self) -> dict:
        if self._isl is None:
            z = np.load(self.cfg.isl.latency_path)
            self._isl = {k: z[k] for k in ("latency_ms", "hop_count", "direct_visible")}
            if self._isl["latency_ms"].shape[0] != self.ecef_km.shape[0]:
                raise ValueError(
                    f"{self.cfg.isl.latency_path} has {self._isl['latency_ms'].shape[0]} "
                    f"timesteps, satellite ECEF has {self.ecef_km.shape[0]}"
                )
        return self._isl

    def gateway_schedule(self) -> np.ndarray:
        """Relay gateway satellite per timestep (-1 = none in view).

        Sticky: picked by longest remaining visibility, kept until it leaves
        view or another in-view satellite is `isl.switch_margin_deg` higher.
        """
        if self._gateway is None:
            gs = self.cfg.ground_station
            gs_ecef = geodetic_to_ecef_km(gs.lat_deg, gs.lon_deg, gs.alt_m, self.cfg.geometry.earth_radius_km)
            los = self.ecef_km - gs_ecef
            up = gs_ecef / np.linalg.norm(gs_ecef)
            cos_zen = (los @ up) / np.linalg.norm(los, axis=2)
            elev = 90.0 - np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))
            vis = elev >= self.cfg.min_elevation_deg
            n, num_sats = vis.shape
            # Seconds of visibility left from each timestep; two sweeps so passes wrap like transfers do.
            remaining = np.zeros((n, num_sats), dtype=np.int64)
            run = np.zeros(num_sats, dtype=np.int64)
            for k in range(2 * n - 1, -1, -1):
                run = np.where(vis[k % n], run + 1, 0)
                if k < n:
                    remaining[k] = np.minimum(run, n)
            margin = self.cfg.isl.switch_margin_deg
            gateway = np.full(n, -1, dtype=np.int64)
            g = -1
            for k in range(n):
                in_view = np.flatnonzero(vis[k])
                if in_view.size == 0:
                    g = -1
                elif g < 0 or not vis[k, g]:
                    g = int(in_view[np.argmax(remaining[k, in_view])])
                else:
                    best = int(in_view[np.argmax(elev[k, in_view])])
                    if elev[k, best] >= elev[k, g] + margin:
                        g = best
                gateway[k] = g
            self._gateway = gateway
        return self._gateway

    def _relay(self, k: int) -> Tuple[int, float]:
        """(gateway, RF rate bps) for relaying at k per `isl.gateway_selection`; (-1, 0) if none."""
        relay = self._relay_cache.get(k)
        if relay is None:
            if self.cfg.isl.gateway_selection == "best_rate":
                candidates = np.flatnonzero(self._isl_arrays()["direct_visible"][k])
            else:
                candidates = [int(self.gateway_schedule()[k])]
            g, best = -1, 0.0
            for s in candidates:
                if s < 0:
                    continue
                r = compute_link_budget(self.ecef_km[k, s], self.cfg, self.direction)
                if r.visible and r.throughput_mbps > best:
                    g, best = int(s), r.throughput_mbps
            relay = (g, best * 1e6 * self.cfg.transfer.efficiency)
            self._relay_cache[k] = relay
        return relay

    def _relay_rate_bps(self, k: int) -> float:
        return self._relay(k)[1]

    def transfer(self, satellite_index: int, start_s: float, nbytes: int) -> TransferEstimate:
        """Time to move `nbytes` in this runtime's direction, starting at
        `start_s` (seconds since the trajectory origin), per `cfg.routing`:
          direct -- this satellite's own RF link; waits for its next pass.
          isl    -- own RF link when in view, else the gateway's RF link
                    (per isl.gateway_selection), capped at the ISL rate, with latency.npz path delay and hops.
        """
        series = self._series(satellite_index)
        own = self._own_rate_bps(satellite_index)
        c_km_s = self.cfg.transfer.speed_of_light_km_s

        def direct_prop(k):
            return series[k].slant_range_km / c_km_s

        if self.cfg.routing != "isl":
            return estimate_transfer(
                own.__getitem__, len(own), start_s, nbytes, direct_prop,
                direction=self.direction, routing="direct",
            )

        isl = self._isl_arrays()
        isl_bps = self.cfg.isl.throughput_mbps * 1e6
        per_hop_s = self.cfg.isl.per_hop_processing_ms / 1000.0

        def rate(k):
            return own[k] if own[k] > 0 else min(self._relay_rate_bps(k), isl_bps)

        def prop(k):
            if own[k] > 0:
                return direct_prop(k)
            hops = int(isl["hop_count"][k, satellite_index])
            return float(isl["latency_ms"][k, satellite_index]) / 1000.0 + (hops - 1) * per_hop_s

        def hops(k):
            return 1 if own[k] > 0 else int(isl["hop_count"][k, satellite_index])

        def gateway(k):
            return -1 if own[k] > 0 else self._relay(k)[0]

        return estimate_transfer(
            rate, len(own), start_s, nbytes, prop, hops,
            direction=self.direction, routing="isl", gateway_at=gateway,
        )

