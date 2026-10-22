#!/usr/bin/env python3
"""Accuracy vs progress Λ for one run, and the s it implies (EXPERIMENTS.md §2.1).

Λ_t = Σ ρ_t·cos_theory_t over `server_update` records (needs `--server-update-audit`).
The logged `progress_lambda` is 2B/s, which equals Λ only at constant s; under the
recipe (fixed N, decaying ρ) it understates Λ, so it is not used.

For the recipe, Λ_∞ = 4·lnΦ_∞/s, so the largest s whose budget Φ* still reaches
Λ_need is s* = 4·lnΦ*/Λ_need. Λ_need is read at each target and at peak − 0.5 pt.

    ./lambda_need.py <run_dir> --targets 0.60 0.65 [--phi-star 2] [--window 5]
"""
import argparse
import bisect
import glob
import json
import math
import os


def load(run_dir):
    path = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0]
    commits, evals = [], []
    for line in open(path):
        if '"server_update"' not in line and '"agg_eval"' not in line:
            continue
        e = json.loads(line)
        if e["event"] == "server_update":
            commits.append((e["ts"], e["commit_count"], e["rho"] * e["cos_theory"], e["phi"]))
        else:
            evals.append((e["ts"], e["test-accuracy"]))
    commits.sort()
    return commits, sorted(evals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--targets", type=float, nargs="*", default=[])
    ap.add_argument("--phi-star", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=5, help="evals in the trailing mean")
    a = ap.parse_args()

    commits, evals = load(a.run_dir)
    ts = [c[0] for c in commits]
    lam, run = [], 0.0
    for _, _, dl, _ in commits:
        run += dl
        lam.append(run)
    # each eval scores the model after the last commit before it
    pts = []
    for i, (t, _) in enumerate(evals):
        j = max(0, bisect.bisect_right(ts, t) - 1)
        w = [acc for _, acc in evals[max(0, i - a.window + 1): i + 1]]
        pts.append((commits[j][1], lam[j], commits[j][3], sum(w) / len(w)))

    c_end = commits[-1]
    print(f"{os.path.basename(a.run_dir.rstrip('/'))}: {len(commits)} commits, "
          f"Λ_end {lam[-1]:.3f}, Φ_end {c_end[3]:.3f}")
    peak = max(pts, key=lambda p: p[3])
    print(f"  peak {peak[3]:.4f} at c{peak[0]}: Λ {peak[1]:.3f}, Φ {peak[2]:.3f}")
    k = 4.0 * math.log(a.phi_star)
    for tgt in a.targets + [round(peak[3] - 0.005, 4)]:
        hit = next((p for p in pts if p[3] >= tgt), None)
        if hit is None:
            print(f"  acc ≥ {tgt}: not reached")
            continue
        print(f"  acc ≥ {tgt}: c{hit[0]}, Λ {hit[1]:.3f}, Φ {hit[2]:.3f} -> "
              f"s* = {k / hit[1]:.2f} (Φ* {a.phi_star:g})")


if __name__ == "__main__":
    main()
