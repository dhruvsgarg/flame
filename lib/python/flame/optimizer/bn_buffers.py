# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""BatchNorm running_var is a variance: base + averaged deltas against a stale base can go negative (FX-N64)."""


def clamp_running_var(weights: dict) -> int:
    """Clamp every `*running_var` tensor to >= 0 in place; returns how many entries were clamped."""
    n = 0
    for k, v in weights.items():
        if k.endswith("running_var") and hasattr(v, "clamp_"):
            n += int((v < 0).sum())
            v.clamp_(min=0)
    return n


def is_bn_stat(key: str) -> bool:
    """BatchNorm running mean/var: non-additive state (num_batches_tracked is a counter and stays additive)."""
    return key.endswith("running_mean") or key.endswith("running_var")
