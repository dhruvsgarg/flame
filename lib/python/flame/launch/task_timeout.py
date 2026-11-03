# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N115: a baseline's task timeout from its own local work, T = 1.5 x the longest modeled task over the reference population.

Task time is FX-D197's FedScale model, 3 x samples x ms/sample + 2 model transfers / bandwidth; samples = local_steps x batch, else
local_epochs x shard. 1.5 x is FedBuff / Papaya's timeout factor (the old 90 s = 1.5 x 60 s). The max, not a percentile, so no
device is structurally unable to finish a task.
"""

import os
from functools import lru_cache

import yaml

from flame.launch import baseline_reference

META = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "_metadata")
TIMEOUT_FACTOR = 1.5


@lru_cache(maxsize=None)
def _registry():
    reg = yaml.safe_load(open(os.path.join(META, "trainer_registry.yaml"), encoding="utf-8"))["trainers"]
    return {f"trainer_{t['trainer_id']:03d}": (float(t["compute_ms_per_sample"]), float(t["bandwidth_kbps"]))
            for t in reg.values()}


@lru_cache(maxsize=None)
def _shards(split: str):
    s = yaml.safe_load(open(os.path.join(META, "dataset_splits", f"{split}.yaml"), encoding="utf-8"))
    return {k: len(v) for k, v in s["trainer_data_splits"].items()}


def task_times_s(baseline: str, dataset: str, prof: dict, ref: dict = None) -> list:
    """Modeled train-task time of every trainer in the dataset's reference split, at the baseline's local work."""
    d = baseline_reference.entry(ref or baseline_reference.load(), baseline)["datasets"][dataset]
    steps = (d.get("local_steps") or {}).get("v")
    batch, epochs = d["batch_size"]["v"], (d.get("local_epochs") or {}).get("v", 1)
    kbit, reg = float(prof["model_kbit"]), _registry()
    out = []
    for tid, shard in _shards(prof["reference_split"]).items():
        ms, kbps = reg[tid]
        samples = steps * batch if steps is not None else epochs * shard
        out.append(3 * samples * ms / 1000 + 2 * kbit / kbps)
    return out


def task_timeout_s(baseline: str, dataset: str, prof: dict, ref: dict = None) -> float:
    return TIMEOUT_FACTOR * max(task_times_s(baseline, dataset, prof, ref))
