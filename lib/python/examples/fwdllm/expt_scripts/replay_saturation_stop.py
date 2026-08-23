#!/usr/bin/env python3
"""Replay rows E + E2: where the saturation stop fires, and what it costs.

    ./replay_saturation_stop.py                        # the three cached 08-21 runs
    ./replay_saturation_stop.py --warmups 300,400,450,600
    ./replay_saturation_stop.py --runs RUN_DIR [RUN_DIR ...]

Reads `writeup_figs/data/<ds>_anchor.json` by default -- `acc_budget` is already
the exact commit/B/Lambda/accuracy join (`extract.py:acc_vs_budget`), so nothing
re-scans the ~25 GB in `experiments/`. `--runs` re-extracts from run dirs.

The gate this script IS (buildplan §3, rows E and E2):

  * fires at commit 1,194 / 1,084 / 1,179 on N1 / N2 / N3, within 0.008 of peak
  * on the six curves the rule was NOT sized on it never fires before that run's
    own peak -- without the progress term it fires 0.153 below on yahoo_control
  * both horizons are multiples of the probe cadence (warm-up 3x, progress 1x)
    and are independent: sweeping one must not move the other

Also prints the fixed-Phi rail (row S), since the two stops ship as an OR.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fwdllm.expts.saturation_stop import (  # noqa: E402
    SAT_GL_THRESHOLD, SAT_PATIENCE, SAT_SMOOTH_WINDOW,
    SaturationDetector, slope_horizon_commits, warmup_commits,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "writeup_figs", "data")
DATASETS = ("agnews", "yahoo", "yelp-p")
# `_anchor` sized the rule; `_controller` and `_control` are the six curves it has
# never seen, and the progress term exists because of what they said (§5.5).
TAGS = ("anchor", "controller", "control")
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


def replay(rows, warmup, threshold, patience, horizon=None):
    # `horizon` is held FIXED while `warmup` sweeps -- both ride the cadence, and
    # letting one move with the other is what E2's gate exists to catch.
    det = SaturationDetector(warmup, threshold, patience,
                             slope_horizon=horizon)
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
            for tag in TAGS:
                try:
                    _run, rows = load_cached(ds, tag)
                except FileNotFoundError:
                    continue
                curves.append((f"{ds}_{tag}", rows))

    warmups = [int(w) for w in args.warmups.split(",")]
    tied = warmup_commits(args.cadence)
    horizon = slope_horizon_commits(args.cadence)
    print(f"threshold={args.threshold} patience={args.patience} "
          f"window={SAT_SMOOTH_WINDOW}  cadence {args.cadence} => "
          f"warm-up 3x = {tied}, progress horizon 1x = {horizon}\n")

    hdr = f"{'run':<20} {'peak':>7} {'@commit':>8} {'end':>7} " + \
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
            c = replay(rows, w, args.threshold, args.patience, horizon)
            if c is None:
                cells.append(f"{'never':>12}")
                continue
            j = min(range(len(commits)), key=lambda i: abs(commits[i] - c))
            cells.append(f"{c:>6} {sm[j] - sm[ip]:>+6.3f}")
        print(f"{name:<20} {sm[ip]:>7.4f} {commits[ip]:>8.0f} {sm[-1]:>7.4f} "
              + " ".join(cells) + f" {str(phi_cross(rows)):>8}")
        # E2's gate: the tied warm-up must not move any fire commit.
        if tied in warmups and replay(rows, tied, args.threshold, args.patience,
                                      horizon) \
                != replay(rows, 400, args.threshold, args.patience, horizon):
            ok = False
            print(f"  !! {name}: warm-up {tied} moves the fire commit off 400's")
    print("\nRow E2 gate:", "PASS -- the warm-up is a multiple, not a fit" if ok
          else "FAIL -- 400 and the tied warm-up disagree; the plateau is narrower "
               "than §5.5 claims")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
