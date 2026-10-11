# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13: the online oracle reconstructs exactly the data a harness trainer holds
(trainer/pytorch/main.py load_data + its sha256(trainer_id) arrival order)."""

import hashlib
import os
from types import SimpleNamespace

import torch
import yaml

from examples.async_cifar10.aggregator.pytorch.oracle_utility import OracleUtilityProvider
from flame import harness

_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "_metadata")


def _provider(mode, k):
    hp = SimpleNamespace(harness_mode=mode, harness_samples=k,
                         oracle_utility_injection={"enabled": "True", "alpha": 0.1, "num_trainers": 300},
                         data_streaming={"enabled": "True", "full_data_available_after_s": 100})
    return OracleUtilityProvider(SimpleNamespace(hyperparameters=hp), data_root=None)


def _first_trainer():
    reg = yaml.safe_load(open(f"{_META}/trainer_registry.yaml"))["trainers"]
    splits = yaml.safe_load(open(f"{_META}/dataset_splits/cifar10_alpha0.1_n300.yaml"))["trainer_data_splits"]
    key = next(k for k in reg if k in splits)
    return str(reg[key]["task_id"]), list(splits[key])


def _trainer_order(tid, n):
    seed = int(hashlib.sha256(tid.encode()).hexdigest(), 16) % (2**31)
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def test_tiny_cpu_uses_the_trainers_prefix_in_its_arrival_order():
    tid, idx = _first_trainer()
    info = _provider("tiny_cpu", 64)._build_table(0.1, 300)[tid]
    pool = harness.prefix_indices(idx, 64)
    assert info["total"] == len(pool)
    assert info["arrival_global_idx"].tolist() == torch.tensor(pool)[_trainer_order(tid, len(pool))].tolist()


def test_stub_uses_the_trainers_synthetic_pool():
    tid, idx = _first_trainer()
    info = _provider("stub", 32)._build_table(0.1, 300)[tid]
    n = min(len(idx), 32)
    data, targets = harness.synthetic_dataset(n, (3, 32, 32), 10, seed_key=tid).tensors
    assert info["total"] == n
    assert torch.equal(info["local"][0], data) and torch.equal(info["local"][1], targets)
    assert info["arrival_global_idx"].tolist() == _trainer_order(tid, n).tolist()


def test_inject_lists_candidates_without_running_the_selector():
    from examples.async_cifar10.aggregator.pytorch.oracle_utility import OracleInjectMixin
    seen = []

    class _Prov:
        enabled = True
        def inject(self, agg, channel, end_ids, task):
            seen.extend(end_ids)

    class _Ch:
        def all_ends(self):
            return ["a", "b"]
        def ends(self, *a, **k):
            raise AssertionError("ends() runs the selector")

    m = OracleInjectMixin()
    m._oracle_util = _Prov()
    m._inject_oracle_utilities(_Ch(), "train")
    assert seen == ["a", "b"]


def test_utility_memoized_per_model_version(monkeypatch):
    import examples.async_cifar10.aggregator.pytorch.oracle_utility as ou
    calls = []
    monkeypatch.setattr(ou, "_oort_utility_acc", lambda *a, **k: (calls.append(1) or (1.0, 0.5)))
    prov = _provider("stub", 32)
    tid, _ = _first_trainer()
    props = {}
    ch = SimpleNamespace(set_end_property=lambda e, k, v: props.__setitem__((e, k), v))
    agg = SimpleNamespace(simulated=False, device="cpu", model=None, _round=3)
    prov.inject(agg, ch, [tid], "train")
    prov.inject(agg, ch, [tid], "train")
    assert len(calls) == 1 and props[(tid, ou.PROP_STAT_UTILITY)] == 1.0
    agg._round = 4
    prov.inject(agg, ch, [tid], "train")
    assert len(calls) == 2


def test_stream_clock_matches_the_trainers_in_both_modes(monkeypatch):
    # Real trainers unlock by wall since AGG_START_TS; a 0 clock ranked every real pick on a 1-sample prefix.
    from examples.async_cifar10.aggregator.pytorch import oracle_utility as ou
    monkeypatch.setattr(ou.time, "time", lambda: 1_000.0)
    assert ou._stream_now(SimpleNamespace(simulated=False, agg_start_time_ts=880.0)) == 120.0
    assert ou._stream_now(SimpleNamespace(simulated=True, _vclock=SimpleNamespace(now=42.0))) == 42.0
