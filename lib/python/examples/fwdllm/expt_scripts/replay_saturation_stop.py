#!/usr/bin/env python3
"""Replay rows E + E2: where the saturation stop fires, and what it costs.

    ./replay_saturation_stop.py                        # the three cached 08-21 runs
    ./replay_saturation_stop.py --warmups 300,400,450,600
    ./replay_saturation_stop.py --runs RUN_DIR [RUN_DIR ...]

Reads `writeup_figs/data/<ds>_anchor.json` by default -- `acc_budget` is already
the exact commit/B/Lambda/accuracy join (`extract.py:acc_vs_budget`), so nothing
re-scans the ~25 GB in `experiments/`. `--runs` re-extracts from run dirs.

The gate this script IS (buildplan §3, rows E and E2):

  * fires at commit 1,184 / 1,084 / 1,126 on N1 / N2 / N3, within 0.008 of peak
  * warm-up 450 = 3 x the 150-commit probe cadence reproduces all three, so the
    warm-up is a multiple with no free parameter and not a constant fitted to
    these curves
  * warm-up 300 false-fires on yahoo at commit 346 and 0.24 accuracy -- the
    reason the warm-up exists at all

Also prints the fixed-Phi rail (row S) on the same curves, since the two stops
ship as an OR and the argument for that is that either alone is nearly right.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fwdllm.expts.saturation_stop import (  # noqa: E402
    SAT_GL_THRESHOLD, SAT_PATIENCE, SAT_SMOOTH_WINDOW,
    SaturationDetector, warmup_commits,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "writeup_figs", "data")
DATASETS = ("agnews", "yahoo", "yelp-p")
PROBE_CADENCE = 150          # b_max_probe_every on every P-4 controller run
PHI_RAIL = 3.0               # row S's re-derived rail


def load_cached(ds, tag="anchor"):
    """(commit, B, Lambda, acc) rows, in emission order."""
    with open(os.path.join(DATA, f"{ds}_{tag}.json")) as fh:
        d = json.load(fh)
    return d["run"], [tuple(r) for r in d["acc_budget"]]


def smooth(v, w=SAT_SMOOTH_WINDOW):
    return [sum(v[max(0, i - w + 1):i + 1]) / len(v[max(0, i - w + 1):i + 1])
            for i in range(len(v))]


def replay(rows, warmup, threshold, patience):
    det = SaturationDetector(warmup, threshold, patience)
    for commit, _B, _lam, acc in rows:
        if det.update(commit, acc):
            return det.fired_at
    return None


def phi_cross(rows, rail=PHI_RAIL):
    for commit, B, _lam, _acc in rows:
        if math.exp(B) >= rail:
            return int(commit)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dirs to extract instead of the cached json")
    ap.add_argument("--warmups", default="300,400,450,600")
    ap.add_argument("--threshold", type=float, default=SAT_GL_THRESHOLD)
    ap.add_argument("--patience", type=int, default=SAT_PATIENCE)
    ap.add_argument("--cadence", type=int, default=PROBE_CADENCE)
    args = ap.parse_args()

    if args.runs:
        sys.path.insert(0, os.path.join(HERE, "writeup_figs"))
        from extract import acc_vs_budget  # noqa: E402
        curves = [(os.path.basename(r), acc_vs_budget(r)) for r in args.runs]
    else:
        curves = []
        for ds in DATASETS:
            run, rows = load_cached(ds)
            curves.append((ds, rows))

    warmups = [int(w) for w in args.warmups.split(",")]
    tied = warmup_commits(args.cadence)
    print(f"threshold={args.threshold} patience={args.patience} "
          f"window={SAT_SMOOTH_WINDOW}  warm-up tied to cadence: "
          f"3 x {args.cadence} = {tied}\n")

    hdr = f"{'run':<10} {'peak':>7} {'@commit':>8} {'end':>7} " + \
          " ".join(f"{'warm ' + str(w):>12}" for w in warmups) + f" {'Phi>=3':>8}"
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for name, rows in curves:
        acc = [r[3] for r in rows]
        commits = [r[0] for r in rows]
        sm = smooth(acc)
        ip = max(range(len(sm)), key=lambda i: sm[i])
        cells = []
        for w in warmups:
            c = replay(rows, w, args.threshold, args.patience)
            if c is None:
                cells.append(f"{'never':>12}")
                continue
            j = min(range(len(commits)), key=lambda i: abs(commits[i] - c))
            cells.append(f"{c:>6} {sm[j] - sm[ip]:>+6.3f}")
        print(f"{name:<10} {sm[ip]:>7.4f} {commits[ip]:>8.0f} {sm[-1]:>7.4f} "
              + " ".join(cells) + f" {str(phi_cross(rows)):>8}")
        # E2's gate: the tied warm-up must not move any fire commit.
        if tied in warmups and replay(rows, tied, args.threshold, args.patience) \
                != replay(rows, 400, args.threshold, args.patience):
            ok = False
            print(f"  !! {name}: warm-up {tied} moves the fire commit off 400's")
    print("\nRow E2 gate:", "PASS -- the warm-up is a multiple, not a fit" if ok
          else "FAIL -- 400 and the tied warm-up disagree; the plateau is narrower "
               "than §5.5 claims")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
