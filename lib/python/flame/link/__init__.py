# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Configurable  RF link budget.

1.Computes elevation/range from satellite + ground-station position 
2.C alcualte free-space + atmospheric + ionospheric loss
3. Reports the resulting SINR degradation and throughput loss.
"""
from flame.link.budget import LinkBudgetResult, compute_link_budget, compute_link_budget_series
from flame.link.config import LinkLayerConfig
from flame.link.propagation import (
    elevation_deg,
    geodetic_to_ecef_km,
    lookup_satellite_ecef,
    slant_range_km,
)
from flame.link.runtime import LinkRuntime

__all__ = [
    "LinkBudgetResult",
    "LinkLayerConfig",
    "LinkRuntime",
    "compute_link_budget",
    "compute_link_budget_series",
    "elevation_deg",
    "geodetic_to_ecef_km",
    "lookup_satellite_ecef",
    "slant_range_km",
]
