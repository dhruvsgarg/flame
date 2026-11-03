# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N74: turn `_metadata/baseline_reference.yaml` into one experiment-config overlay per (baseline, dataset)."""

import math
import os

import yaml

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "_metadata", "baseline_reference.yaml")
K_OVER_N_FLOOR = 0.05  # operator 10-08: sync K >= 5% of the population
REQUIRED = ("batch_size", "learning_rate", "optimizer", "lr_decay", "server")  # + local_steps | local_epochs
_TRAINER = {"local_steps": "localSteps", "local_epochs": "epochs", "batch_size": "batchSize",
            "learning_rate": "learningRate", "optimizer": "trainerOptimizer", "stat_utility": "statUtility",
            "lr_batch_normalize": "lrBatchNormalize", "momentum": "trainerMomentum", "weight_decay": "trainerWeightDecay",
            "clip_grad_norm": "trainerClipGradNorm"}


def load(path: str = PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def entry(ref: dict, baseline: str) -> dict:
    e = ref.get(baseline)
    return ref[e["same_as"]] if e and "same_as" in e else e


def agg_goal(ref: dict, baseline: str, n: int):
    """Sync K = ceil(max(source K/N, floor) * n); None for async baselines (no source fraction)."""
    frac = (entry(ref, baseline) or {}).get("participation", {}).get("k_over_n")
    return None if frac is None else max(1, math.ceil(max(frac, K_OVER_N_FLOOR) * n))


def overlay(ref: dict, baseline: str, dataset: str, n: int) -> dict:
    """Experiment-config overlay (trainer hyperparameters, optimizer, selector kwargs, agg_goal); {} if unlisted."""
    e = entry(ref, baseline)
    if not e:
        return {}
    d = e["datasets"][dataset]
    hp = {_TRAINER[k]: v["v"] for k, v in d.items() if k in _TRAINER}
    if "local_steps" in d:
        hp["epochs"] = 1  # the step count binds; trainer cycles its data until it is reached
    decay = d["lr_decay"]["v"]
    hp.update({"lrDecayEnabled": False} if decay == "none" else
              {"lrDecayEnabled": True, "lrDecayFactor": decay["factor"], "lrDecayEpoch": decay["every"],
               "minLearningRate": decay["min"]})
    sel = {**(e.get("selector") or {}).get("v", {}), **(d.get("selector") or {}).get("v", {})}
    co = {"optimizer": d["server"]["v"]}
    if sel:
        co["selector"] = {"kwargs": sel}
    out = {"trainer": {"hyperparameters": hp}, "aggregator": {"config_overrides": co}}
    k = agg_goal(ref, baseline, n)
    if k is not None:
        out["aggregator"]["agg_goal"] = k
    return out
