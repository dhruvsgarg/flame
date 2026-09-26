#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13 offline oracle replay: true vs believed utility along a run's own trajectory.

For each checkpoint (round r, stream time t) it rebuilds every trainer's visible data prefix
exactly as the online oracle does (aggregator/pytorch/oracle_utility.py, harness-aware), scores
it with the round-r global model, and joins that to each train `selection` event at round r.

Writes under <run>/analysis/:
  oracle_true_utility.csv   round, stream_s, trainer, visible, true_util, rms_loss, believed_util
                            (rms_loss = true_util / visible: the loss term, free of data growth)
  oracle_misselection.csv   round, sel_idx, pool, k, hit_rate, rank_pct, regret_rel, spearman

  hit_rate   = |chosen ∩ true top-k of the pickable pool| / k   (misselection = 1 - hit_rate)
  rank_pct   = mean true-utility percentile of the chosen within the pool (1 = best)
  regret_rel = 1 - sum(true of chosen) / sum(true of top-k)
  spearman   = rank correlation believed vs true over the pool

Usage: oracle_misselection.py <run_dir> [<run_dir> ...] [--sample-size 256]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGG = os.path.join(_HERE, "..", "aggregator", "pytorch")
if _AGG not in sys.path:
    sys.path.insert(0, _AGG)

from oracle_utility import OracleUtilityProvider, _oort_utility_acc, _visible_count  # noqa: E402

_DATA_ROOT = os.path.join(_HERE, "..", "data")


def _load_hp(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "aggregator_config.json")) as f:
        return json.load(f).get("hyperparameters", {})


def _split_key(run_dir: str, hp: dict) -> tuple[float, int]:
    """(alpha, n) of the Dirichlet split file: oracle config first, else the run name."""
    oi = hp.get("oracle_utility_injection") or {}
    if "alpha" in oi and "num_trainers" in oi:
        return float(oi["alpha"]), int(oi["num_trainers"])
    m = re.search(r"_n(\d+)_alpha([0-9.]+)_", os.path.basename(run_dir.rstrip("/")))
    if not m:
        raise ValueError(f"cannot infer split (alpha, n) for {run_dir}")
    return float(m.group(2)), int(m.group(1))


def build_provider(run_dir: str, sample_size: int) -> OracleUtilityProvider:
    hp = dict(_load_hp(run_dir))
    alpha, n = _split_key(run_dir, hp)
    hp["oracle_utility_injection"] = {"enabled": "True", "alpha": alpha, "num_trainers": n,
                                      "sample_size": sample_size}
    os.environ["FLAME_TELEMETRY_DIR"] = os.path.join(run_dir, "telemetry")  # its metadata_location
    prov = OracleUtilityProvider(SimpleNamespace(hyperparameters=SimpleNamespace(**hp)), _DATA_ROOT)
    prov._ensure(prov.alpha, prov.num_trainers)
    return prov


def train_selections(run_dir: str) -> dict[int, list[dict]]:
    """round -> train selection events with a pick, in log order."""
    out: dict[int, list[dict]] = {}
    for p in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("event") == "selection" and e.get("task") == "train" and e.get("chosen"):
                    out.setdefault(int(e["round"]), []).append(e)
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = (ra.norm() * rb.norm()).item()
    return (ra @ rb).item() / den if den else None


def selection_metrics(sel: dict, true_u: dict[str, float]) -> dict | None:
    """Score one selection's picks against the true utilities of its pickable pool."""
    pt = sel.get("per_trainer") or {}
    chosen = [c for c in sel["chosen"] if c in true_u]
    pool = [t for t, v in pt.items()
            if t in true_u and (t in chosen or not v.get("in_all_selected"))]
    k = len(chosen)
    if not k or len(pool) < k:
        return None
    ranked = sorted(pool, key=lambda t: true_u[t], reverse=True)
    top = set(ranked[:k])
    pos = {t: i for i, t in enumerate(ranked)}
    denom = max(len(pool) - 1, 1)
    top_sum = sum(true_u[t] for t in top)
    believed = [pt[t].get("utility") for t in pool]
    rho = (_spearman([float(x) for x in believed], [true_u[t] for t in pool])
           if all(x is not None for x in believed) else None)
    return {
        "pool": len(pool), "k": k,
        "hit_rate": len(top & set(chosen)) / k,
        "rank_pct": sum(1.0 - pos[c] / denom for c in chosen) / k,
        "regret_rel": 1.0 - sum(true_u[c] for c in chosen) / top_sum if top_sum > 0 else 0.0,
        "spearman": rho,
    }


def replay(run_dir: str, sample_size: int = 256) -> tuple[str, str]:
    from main_asyncfl_agg import Net  # the aggregator's model

    prov = build_provider(run_dir, sample_size)
    sels = train_selections(run_dir)
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    u_path = os.path.join(out_dir, "oracle_true_utility.csv")
    m_path = os.path.join(out_dir, "oracle_misselection.csv")
    model = Net()
    in_run = {t for ss in sels.values() for s in ss for t in (s.get("per_trainer") or {})}
    torch.manual_seed(0)  # sample_size subsampling is seeded per run
    with open(u_path, "w", newline="") as uf, open(m_path, "w", newline="") as mf:
        uw = csv.writer(uf)
        uw.writerow(["round", "stream_s", "trainer", "visible", "true_util", "rms_loss", "believed_util"])
        mw = csv.DictWriter(mf, ["round", "sel_idx", "pool", "k", "hit_rate", "rank_pct",
                                 "regret_rel", "spearman"])
        mw.writeheader()
        for ck in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "round_*.pt"))):
            st = torch.load(ck, map_location="cpu", weights_only=False)
            rnd, t = int(st["round"]), float(st.get("sim_time_s") or 0.0)
            model.load_state_dict(st["state_dict"])
            believed = {}
            for s in sels.get(rnd, []):
                believed.update({k: v.get("utility") for k, v in (s.get("per_trainer") or {}).items()})
            true_u = {}
            for tid, info in prov._table.items():
                if in_run and tid not in in_run:
                    continue
                vis = _visible_count(t, info["onset_s"], info["span_s"], info["total"], prov.min_visible)
                g = info["arrival_global_idx"][:vis]
                imgs, targets = info["local"] or (prov._imgs, prov._targets)
                true_u[tid], _ = _oort_utility_acc(model, imgs[g], targets[g], norm_n=vis,
                                                   device=torch.device("cpu"), sample_size=prov.sample_size)
                uw.writerow([rnd, f"{t:.1f}", tid, vis, f"{true_u[tid]:.5f}", f"{true_u[tid] / vis:.5f}",
                                 "" if believed.get(tid) is None else f"{float(believed[tid]):.5f}"])
            for i, s in enumerate(sels.get(rnd, [])):
                m = selection_metrics(s, true_u)
                if m:
                    mw.writerow({"round": rnd, "sel_idx": i, **{k: ("" if v is None else round(v, 5))
                                                                for k, v in m.items()}})
    return u_path, m_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sample-size", type=int, default=256)
    a = ap.parse_args()
    for r in a.runs:
        if not glob.glob(os.path.join(r, "checkpoints", "round_*.pt")):
            print(f"SKIP {r}: no checkpoints")
            continue
        u, m = replay(r, a.sample_size)
        print(f"{r}\n  {u}\n  {m}")


if __name__ == "__main__":
    main()
