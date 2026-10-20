#!/usr/bin/env python3
"""FX-N117 (D3): does a client's loss carry progress signal beyond its sample count? Stored telemetry only, no runs.

  utility_evidence.py POOL... [--baselines felix,fedbuff]

Per async sim leg, per window between two test evals: progress = relative test-loss drop; features = log of the local samples
the window aggregated (S), mean log(q_i / test loss) over its updates (Q: how unfit the aggregated clients were relative to the
global model; q_i = utility / shard, the FedScale rms loss), mean staleness (τ), log test loss at the window start (phase).
Prints, pooled over legs, the partial correlation of progress with Q given (S, τ, phase), with a bootstrap 90% interval: Q's
signal beyond samples. Also the rank agreement (Spearman) of |D| x loss with |D| alone at each selection: how much the loss term
reorders today's picks.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import rows_from_pools  # noqa: E402
from selection_audit import spearman  # noqa: E402


def _trainer_tasks(run_dir):
    """(end_id, model version) -> (local samples, shard size) of each train task."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        end = os.path.basename(f)[len("trainer_"):-len(".jsonl")]
        for line in open(f, "rb"):
            if b'"trainer_round"' in line:
                e = json.loads(line)
                if e.get("task_to_perform", "train") == "train" and e.get("dataset_size"):
                    out[(end, e["round"])] = (e.get("model_samples") or e["dataset_size"], e["dataset_size"])
    return out


def windows(run_dir):
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not f:
        return [], []
    tasks = _trainer_tasks(run_dir)
    shard = {end: d for (end, _), (_, d) in tasks.items()}
    evals, rounds, agree = [], {}, []
    for line in open(f[0], "rb"):
        if b'"agg_eval"' in line:
            e = json.loads(line)
            if e.get("test-loss") is not None:
                evals.append((e["round"], float(e["test-loss"])))
        elif b'"agg_round"' in line:
            e = json.loads(line)
            if e.get("event") != "agg_round":
                continue
            for end, u, tau in zip(e.get("contributing_trainers") or [], e.get("stat_utility") or [], e.get("staleness") or []):
                t = tasks.get((end, e["round"] - (tau or 0)))
                if t and u:
                    rounds.setdefault(e["round"], []).append((u / t[1], tau or 0, t[0]))  # q = utility / shard: rms loss
        elif b'"selection"' in line and b'"utility": ' in line:
            e = json.loads(line)
            pt = [(v["utility"], shard[k]) for k, v in (e.get("per_trainer") or {}).items() if v.get("utility") and k in shard]
            if len(pt) >= 5:
                agree.append(spearman([u for u, _ in pt], [d for _, d in pt]))
    evals.sort()
    out = []
    for (r0, l0), (r1, l1) in zip(evals, evals[1:]):
        ups = [x for r in range(r0 + 1, r1 + 1) for x in rounds.get(r, [])]
        if len(ups) < 3 or l0 <= 0 or l1 <= 0:
            continue
        out.append(dict(progress=(l0 - l1) / l0, phase=math.log(l0), S=math.log(sum(s for *_, s in ups)),
                        Q=float(np.mean([math.log(max(q, 1e-9) / l0) for q, _, _ in ups])),
                        tau=float(np.mean([t for _, t, _ in ups]))))
    return out, [x for x in agree if x is not None and x == x]


def partial_corr(rows):
    X = np.array([[1, r["S"], r["tau"], r["phase"]] for r in rows])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = res(np.array([r["progress"] for r in rows])), res(np.array([r["Q"] for r in rows]))
    return float(a @ b / math.sqrt((a @ a) * (b @ b))) if a @ a > 0 and b @ b > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--baselines", default="felix,fedbuff")
    a = ap.parse_args(argv)
    want = set(a.baselines.split(","))
    by = {}
    legs = [(b, sim) for _ph, b, _tr, _real, sim in rows_from_pools(a.pools) if b in want and sim and os.path.isdir(sim)]
    with Pool(min(32, os.cpu_count() or 1)) as pool:  # L19: one process per leg
        res = pool.map(windows, [sim for _, sim in legs])
    for (b, sim), (w, ag) in zip(legs, res):
        d = by.setdefault((b, "speech" if "google_speech" in sim else "cifar"), {"rows": [], "agree": [], "legs": 0})
        d["rows"] += w
        d["agree"] += ag
        d["legs"] += 1
    rng = random.Random(0)
    print("| baseline · dataset | legs | windows | partial r(progress, Q | S, τ, phase) [90% CI] | Spearman(|D|·loss, |D|) per selection |")
    print("|---|---|---|---|---|")
    for (b, ds), d in sorted(by.items()):
        rows = d["rows"]
        if len(rows) < 10:
            continue
        boot = sorted(partial_corr([rng.choice(rows) for _ in rows]) for _ in range(500))
        ag = sorted(d["agree"])
        med = f"{ag[len(ag) // 2]:+.2f}" if ag else "–"
        print(f"| {b} · {ds} | {d['legs']} | {len(rows)} | {partial_corr(rows):+.2f} [{boot[25]:+.2f}, {boot[474]:+.2f}] | {med} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
