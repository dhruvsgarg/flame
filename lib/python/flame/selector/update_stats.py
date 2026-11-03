# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""FX-N109 I5: per-cycle update statistics for dynamic K / c, track-agnostic.

Feed each update vector of one aggregation cycle (a backprop delta, or a forward-gradient estimate); `close()` returns the
gradient-noise-scale estimate B = tr(Σ) / |G|² (McCandlish et al. 2018) and how the updates align with their mean by
staleness. B is the cycle size past which a larger K buys little: K << B wastes nothing on noise, K >> B wastes updates.
"""

from typing import Dict, Iterable, Optional

import torch


def flatten(weights: Dict[str, torch.Tensor], skip: Optional[callable] = None) -> torch.Tensor:
    """One float32 vector over a weights dict's float tensors (keys sorted; `skip(key)` drops e.g. BN statistics)."""
    parts = [v.detach().reshape(-1).float() for k, v in sorted(weights.items())
             if torch.is_tensor(v) and v.is_floating_point() and not (skip and skip(k))]
    return torch.cat(parts) if parts else torch.zeros(0)


class UpdateStats:
    """Accumulates one cycle's update vectors; O(model) memory (running sum, plus vectors only for alignment)."""

    def __init__(self, keep_vectors: bool = True):
        self.keep_vectors = keep_vectors
        self.reset()

    def reset(self) -> None:
        self._sum: Optional[torch.Tensor] = None
        self._sq = 0.0
        self._n = 0
        self._vecs: list = []  # (staleness, vector)

    def add(self, vec: torch.Tensor, staleness: int = 0) -> None:
        self._sum = vec.clone() if self._sum is None else self._sum.add_(vec)
        self._sq += float(vec.dot(vec))
        self._n += 1
        if self.keep_vectors:
            self._vecs.append((staleness, vec))

    def close(self) -> dict:
        """Cycle statistics, then reset. S = mean |Δ_i|², M = |mean Δ|²; unbiased |G|², tr(Σ) need n ≥ 2."""
        n, out = self._n, {"n": self._n}
        if n >= 2:
            mean = self._sum / n
            m, s = float(mean.dot(mean)), self._sq / n
            g2, tr = (n * m - s) / (n - 1), n * (s - m) / (n - 1)
            out.update(sq_mean=s, mean_sq=m, g2=g2, tr_sigma=tr, noise_scale=(tr / g2) if g2 > 0 else float("inf"))
            if self._vecs:
                mn = mean.norm()
                cos = [(t, float(v.dot(mean) / (v.norm() * mn))) for t, v in self._vecs if v.norm() > 0 and mn > 0]
                out["cos_by_staleness"] = cos
        self.reset()
        return out


def cos_split(cos_by_staleness: Iterable) -> Optional[float]:
    """Mean cos(Δ_i, Δ̄) of the stalest half minus the freshest half; < 0 = stale updates point elsewhere."""
    c = sorted(cos_by_staleness)
    if len(c) < 2:
        return None
    h = len(c) // 2
    return sum(x for _, x in c[-h:]) / h - sum(x for _, x in c[:h]) / h
