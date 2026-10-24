#!/usr/bin/env python3
"""F42 (FELIX Ledger C8): offline replay of overfit-aware pacing of c on finished Felix legs (ROBUST R26: model before code).

  pacing_replay.py POOL... [--arms e0 c4 ctl] [--rho 0.1] [--beta 0.75] [--cmin 0.25] [--sigma 0.03] [--draws 5]

Per eval window: signal = the leg's test loss × (1 + N(0, σ)) (an ideal federated validation probe), mean of the last 2 windows. Above
(1 + ρ) × its running min → c × β (floor cmin × c); at or below it → c × 2 (cap c). Bytes ∝ c (task times unchanged). Accuracy is
pessimistic: the original leg's accuracy at the matched cumulative bytes (= updates), so a plateau's data growth never helps the replay.
Deltas are against the same mapping at fixed c, which removes its eval-lag artifact.
"""
import argparse
import bisect
import glob
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import curve, dataset_of, rows_from_pools  # noqa: E402
from felix_claims import traffic  # noqa: E402


def load(run_dir):
    """(eval hours, accuracy %, test loss, cumulative-GB curve [(h, GB)]) of one leg."""
    c, _ = curve(run_dir)
    tr = traffic(run_dir)
    tot, s = [], 0.0
    for t, b in tr:
        s += b
        tot.append((t / 3600, s / 1e9))
    loss = []
    with open(glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0], "rb") as fh:
        for line in fh:
            if b'"agg_eval"' in line:
                loss.append(json.loads(line).get("test-loss"))
    return [t / 3600 for t, _, _ in c], [100 * a for *_, a in c], [x for x in loss if x is not None], tot


def replay(leg, rho, beta, cmin, sigma, seed):
    hs, acc, loss, tot = leg
    xs = [t for t, _ in tot]
    gb_at = lambda h: tot[bisect.bisect_right(xs, h) - 1][1] if xs and h >= xs[0] else 0.0
    h_at = lambda g: next((t for t, x in tot if x >= g), tot[-1][0])
    rnd = random.Random(seed)
    f, g, sig, out, mn = 1.0, 0.0, [], [], float("inf")
    for i in range(len(hs) - 1):
        g += f * (gb_at(hs[i + 1]) - gb_at(hs[i]))
        j = max(k for k, t in enumerate(hs) if t <= h_at(g) + 1e-9)
        out.append(acc[j])
        sig.append(loss[min(j + 1, len(loss) - 1)] * (1 + rnd.gauss(0, sigma)))  # the loss of the model Felix actually has
        s = st.mean(sig[-2:])
        mn = min(mn, s)
        f = max(cmin, f * beta) if s > (1 + rho) * mn else min(1.0, 2 * f)
    return g, st.mean(out[-3:]), st.mean([acc[0]] + out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--arms", nargs="*", default=["e0", "c4", "ctl"])
    ap.add_argument("--rho", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--cmin", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--draws", type=int, default=5)
    a = ap.parse_args(argv)
    print("| leg | dataset | bytes saved | final Δ pp | AUC Δ pp |\n|---|---|---|---|---|")
    for phase, b, _, _, sim in rows_from_pools(a.pools):
        if b != "felix" or not sim or phase.rsplit("_", 1)[-1] not in a.arms:
            continue
        leg = load(sim)
        g0, f0, a0 = replay(leg, 1e9, a.beta, a.cmin, 0.0, 0)
        r = [replay(leg, a.rho, a.beta, a.cmin, a.sigma, s) for s in range(a.draws)]
        print(f"| {phase} | {dataset_of(sim)} | {1 - st.mean(x[0] for x in r) / g0:.0%} | {st.mean(x[1] for x in r) - f0:+.1f} "
              f"| {st.mean(x[2] for x in r) - a0:+.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
