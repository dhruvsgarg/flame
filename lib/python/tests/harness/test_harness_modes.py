# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""flame.harness: mode parsing, CPU forcing, deterministic synthetic data."""

from types import SimpleNamespace

import pytest
import torch

from flame import harness


def test_mode_defaults_off():
    assert harness.harness_mode(SimpleNamespace()) == "off"


@pytest.mark.parametrize("key", ["harness_mode", "harnessMode"])
def test_mode_reads_both_key_styles(key):
    assert harness.harness_mode(SimpleNamespace(**{key: "STUB"})) == "stub"


def test_mode_rejects_unknown():
    with pytest.raises(ValueError):
        harness.harness_mode(SimpleNamespace(harness_mode="gpu_fast"))


def test_harness_modes_force_cpu():
    assert harness.device_for("stub").type == "cpu"
    assert harness.device_for("tiny_cpu").type == "cpu"


def test_sample_defaults_and_override():
    assert harness.harness_samples(SimpleNamespace(), "stub") == 32
    assert harness.harness_samples(SimpleNamespace(harness_samples=8), "stub") == 8
    assert harness.harness_test_samples(SimpleNamespace(), "tiny_cpu") == 512


def test_synthetic_dataset_deterministic_per_key():
    a = harness.synthetic_dataset(16, (3, 4, 4), 10, seed_key=7)
    b = harness.synthetic_dataset(16, (3, 4, 4), 10, seed_key=7)
    c = harness.synthetic_dataset(16, (3, 4, 4), 10, seed_key=8)
    assert torch.equal(a.tensors[0], b.tensors[0]) and torch.equal(a.tensors[1], b.tensors[1])
    assert not torch.equal(a.tensors[0], c.tensors[0])


def test_synthetic_dataset_is_label_skewed():
    ds = harness.synthetic_dataset(400, (1,), 10, seed_key="t1", label_skew=0.8)
    counts = torch.bincount(ds.tensors[1], minlength=10)
    assert counts.max().item() / 400 > 0.7


def test_prefix_indices():
    assert harness.prefix_indices([5, 6, 7, 8], 2) == [5, 6]
    assert harness.prefix_indices([5], 0) == [5]
