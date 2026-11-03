# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedScale-family server YoGi (FX-N74): one implementation for REFL and Oort.

REFL core/utils/yogi.py: momentum = yogi_beta (0.9), v decay = yogi_beta2 (0.99).
Oort training/utils/yogi.py: no momentum (beta2=-1), v decay = beta (0.999) -> momentum=0, v_decay=0.999.
Both step model.parameters() only: BatchNorm stats and integer buffers keep the plain aggregate.
"""

from .bn_buffers import is_bn_stat
from .fedavg import FedAvg


class FedScaleYoGi:
    def __init__(self, eta: float, tau: float, momentum: float, v_decay: float, normalize_first: bool = False):
        self.eta, self.tau, self.momentum, self.v_decay = eta, tau, momentum, v_decay
        self.normalize_first = normalize_first  # FX-D107: first step scaled by eta/(|g|+tau), not passed through
        self.v_t, self.delta_t = None, None

    def step(self, last: dict, current: dict) -> dict:
        """New global = last + YoGi(current - last) on parameters; buffers pass through from `current`."""
        import torch

        keys = [k for k, v in current.items() if v.is_floating_point() and not is_bn_stat(k)]
        diff = {k: current[k] - last[k] for k in keys}
        if self.v_t is None:  # first call: initialise state, apply the plain update (both forks)
            self.v_t = {k: d ** 2 for k, d in diff.items()}
            self.delta_t = {k: d.clone() for k, d in diff.items()}
            step = ({k: self.eta / (d.abs() + self.tau) * d for k, d in diff.items()}
                    if self.normalize_first else diff)
        else:
            step = {}
            for k, g in diff.items():
                g2 = g ** 2
                self.delta_t[k] = self.momentum * self.delta_t[k] + (1.0 - self.momentum) * g
                self.v_t[k] = self.v_t[k] - (1.0 - self.v_decay) * g2 * torch.sign(self.v_t[k] - g2)
                step[k] = self.eta / (torch.sqrt(self.v_t[k]) + self.tau) * self.delta_t[k]
        out = dict(current)
        for k in keys:
            out[k] = last[k] + step[k]
        return out


class FedAvgYoGi(FedAvg):
    """FedAvg aggregate, then a FedScale YoGi server step (Oort runs YoGi or Prox, never plain FedAvg; FX-N74)."""

    def __init__(self, yogi_eta: float, yogi_tau: float, yogi_momentum: float, yogi_v_decay: float,
                 yogi_normalize_first: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._yogi = FedScaleYoGi(yogi_eta, yogi_tau, yogi_momentum, yogi_v_decay,
                                  normalize_first=str(yogi_normalize_first).lower() == "true")

    def do(self, base_weights, cache, *, total: int = 0, version: int = 0, **kwargs):
        last = {k: v.clone() for k, v in base_weights.items()}  # FedAvg.do accumulates into base_weights
        new = super().do(base_weights, cache, total=total, version=version, **kwargs)
        return None if new is None else self._yogi.step(last, new)
