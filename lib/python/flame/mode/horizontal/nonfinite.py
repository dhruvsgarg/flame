# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D111: an update with non-finite weights or utility is a failed client task: never aggregated, never scored."""

import logging
import math
from typing import Optional

from flame import telemetry
from flame.mode.message import MessageType

logger = logging.getLogger(__name__)


def hp_of(agg):
    return getattr(getattr(agg, "config", None), "hyperparameters", None)


def nonfinite_reason(msg: dict, weights, hp=None) -> Optional[str]:
    """'utility' / 'weights' when that part of the update is NaN/inf, else None (or when `reject_nonfinite_updates` is off)."""
    if hp is not None and getattr(hp, "reject_nonfinite_updates", True) is False:
        return None
    u = msg.get(MessageType.STAT_UTILITY)
    if u is not None and not math.isfinite(float(u)):
        return "utility"
    if isinstance(weights, dict):
        import torch

        for v in weights.values():
            if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all()):
                return "weights"
    return None


def reject(end: str, reason: str, round_num, version) -> None:
    logger.warning(f"[NONFINITE_REJECT] {end[-4:]} v{version} at r{round_num}: non-finite {reason}; update dropped")
    telemetry.emit("update_rejected", end_id=end, round=round_num, version=version, reason=f"nonfinite_{reason}")
