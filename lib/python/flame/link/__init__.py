# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Configurable  RF link budget.

1.Computes elevation/range from satellite + ground-station position 
2.C alcualte free-space + atmospheric + ionospheric loss
3. Reports the resulting SINR degradation and throughput loss.
"""
import importlib

from flame.link.config import LinkLayerConfig

# Import-light: everything except config loads on first access (PEP 562).
_LAZY = {
    "LinkBudgetResult": "budget",
    "compute_link_budget": "budget",
    "compute_link_budget_series": "budget",
    "elevation_deg": "propagation",
    "geodetic_to_ecef_km": "propagation",
    "lookup_satellite_ecef": "propagation",
    "slant_range_km": "propagation",
    "LinkRuntime": "runtime",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f"flame.link.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
