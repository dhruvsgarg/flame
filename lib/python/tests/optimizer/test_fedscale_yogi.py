# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D100: FedScale YoGi (REFL/Oort forks) steps parameters only; Oort's variant has no momentum."""

import torch

from flame.config import OptimizerType
from flame.optimizer.fedscale_yogi import FedAvgYoGi, FedScaleYoGi
from flame.optimizers import optimizer_provider


def test_oort_variant_matches_fork_formula():
    y = FedScaleYoGi(eta=0.005, tau=0.001, momentum=0.0, v_decay=0.999)
    last = {"w": torch.zeros(2), "bn.running_mean": torch.zeros(2)}
    y.step(last, {"w": torch.full((2,), 0.1), "bn.running_mean": torch.ones(2)})  # init: v = g^2 = 0.01
    g = torch.full((2,), 0.2)
    out = y.step(last, {"w": g, "bn.running_mean": torch.ones(2)})
    v = 0.01 - 0.001 * 0.04 * torch.sign(torch.tensor(0.01 - 0.04))  # Oort utils/yogi.py
    assert torch.allclose(out["w"], 0.005 / (torch.sqrt(v) + 0.001) * g)
    assert torch.equal(out["bn.running_mean"], torch.ones(2))  # buffers keep the aggregate


def test_fedavg_yogi_is_registered():
    opt = optimizer_provider.get(OptimizerType.FEDAVG_YOGI, yogi_eta=0.005, yogi_tau=0.001, yogi_momentum=0.0,
                                 yogi_v_decay=0.999)
    assert isinstance(opt, FedAvgYoGi)


def test_first_step_passes_through_unless_normalized():
    # FX-D107: off = upstream (plain first update); on = eta * g / (|g| + tau), bounded by eta.
    last, cur = {"w": torch.zeros(2)}, {"w": torch.tensor([100.0, -0.002])}
    assert torch.equal(FedScaleYoGi(0.005, 0.001, 0.0, 0.999).step(last, cur)["w"], cur["w"])
    out = FedScaleYoGi(0.005, 0.001, 0.0, 0.999, normalize_first=True).step(last, cur)["w"]
    assert torch.allclose(out, 0.005 / (cur["w"].abs() + 0.001) * cur["w"])
    assert out.abs().max() <= 0.005
