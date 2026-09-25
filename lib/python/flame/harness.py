# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""No-GPU local harness modes (ROBUST_FL_READINESS S1).

``harness_mode`` (hyperparameter, both roles):
  off      -- production path, byte-identical.
  stub     -- CPU, synthetic seeded data, no dataset download; exercises every FL
              code path (dispatch, ordering, clock, selection, availability) in ms.
  tiny_cpu -- CPU, a real-data prefix of ``harness_samples`` per trainer, so
              utility/loss signals are meaningful for selector dynamics.
"""

import hashlib

import torch
import torch.utils.data as data_utils

MODES = ("off", "stub", "tiny_cpu")
DEFAULT_SAMPLES = {"stub": 32, "tiny_cpu": 64}
DEFAULT_TEST_SAMPLES = {"stub": 256, "tiny_cpu": 512}


def _hp(hp, *keys, default=None):
    for k in keys:
        v = getattr(hp, k, None)
        if v is None and isinstance(getattr(hp, "__dict__", None), dict):
            v = hp.__dict__.get(k)
        if v is not None:
            return v
    return default


def harness_mode(hp) -> str:
    """Normalized harness mode from a hyperparameters object; 'off' if unset."""
    mode = str(_hp(hp, "harness_mode", "harnessMode", default="off")).strip().lower()
    if mode not in MODES:
        raise ValueError(f"harness_mode={mode!r} not in {MODES}")
    return mode


def harness_samples(hp, mode: str) -> int:
    return int(_hp(hp, "harness_samples", "harnessSamples", default=DEFAULT_SAMPLES.get(mode, 0)))


def harness_test_samples(hp, mode: str) -> int:
    return int(
        _hp(hp, "harness_test_samples", "harnessTestSamples",
            default=DEFAULT_TEST_SAMPLES.get(mode, 0))
    )


def device_for(mode: str) -> torch.device:
    if mode != "off":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stable_seed(key) -> int:
    """Process-independent seed (Python's hash() is salted per process)."""
    return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % (2**31)


def synthetic_dataset(n: int, shape, num_classes: int, seed_key, label_skew: float = 0.8):
    """Seeded TensorDataset; `label_skew` of labels are the key's dominant class
    (non-IID across trainers), the rest uniform."""
    g = torch.Generator().manual_seed(stable_seed(seed_key))
    data = torch.randn((n, *shape), generator=g)
    dominant = stable_seed(("label", seed_key)) % num_classes
    uniform = torch.randint(0, num_classes, (n,), generator=g)
    pick = torch.rand(n, generator=g) < label_skew
    targets = torch.where(pick, torch.full_like(uniform, dominant), uniform)
    return data_utils.TensorDataset(data, targets)


def prefix_indices(indices, k: int):
    """First k of a trainer's split (deterministic real-data subset)."""
    return list(indices)[: max(1, k)]
