# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D111: a NaN/inf update or utility is a failed task: dropped, never scored; the knob reverts it."""

from types import SimpleNamespace

import torch

from flame.mode.horizontal.nonfinite import nonfinite_reason
from flame.mode.message import MessageType


def test_nonfinite_weights_or_utility_are_flagged():
    ok = {"w": torch.ones(2), "n": torch.tensor(3)}
    assert nonfinite_reason({MessageType.STAT_UTILITY: 2.3}, ok) is None
    assert nonfinite_reason({MessageType.STAT_UTILITY: float("nan")}, ok) == "utility"
    assert nonfinite_reason({}, {"w": torch.tensor([1.0, float("inf")])}) == "weights"


def test_knob_off_keeps_the_update():
    off = SimpleNamespace(reject_nonfinite_updates=False)
    assert nonfinite_reason({MessageType.STAT_UTILITY: float("nan")}, None, off) is None
