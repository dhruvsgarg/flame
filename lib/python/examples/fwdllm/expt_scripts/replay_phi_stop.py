#!/usr/bin/env python3
"""C-1's acceptance test (fl_fwd_ft_buildplan.md 0.5): for every arm that
learned, what accuracy would a controller have given up had it stopped the
first time smoothed `Phi` crossed a threshold? Reproduces
fl_fwd_ft_practice.md P4.1.

    ./replay_phi_stop.py RUN_DIR [RUN_DIR ...]
    ./replay_phi_stop.py --thresholds 2.3,2.5,2.7,3.0 RUN_DIR [RUN_DIR ...]

Accuracy is smoothed over an 11-eval TRAILING window before peak/crossing are
read off it -- the smoothing rule is part of the spec: raw crossings gave a
false agreement at Phi ~= 2.95 on the two G-1b arms (P4.2). `Phi` itself is
exact from `rho` (never re-derived from smoothed weight norms) and follows
`exp((1+beta)/(1-beta)*B)` once `server_momentum` > 0.

Edge cases (buildplan 0.5):
  (a) An arm that never crosses a threshold counts as accuracy given up = 0,
      not skipped -- the controller was correct not to stop.
  (b) An arm that never learned (peak(smoothed) < 0.80) is EXCLUDED and the
      exclusion is printed -- a sinking condition without its precondition is
      how G-1 became unreadable (P6).
  (c) Under server_momentum > 0, Phi follows the momentum-corrected form.
"""
import argparse
import glob
import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_scoring import slice_run, meta, load, enrich  # noqa: E402

SMOOTH_WINDOW = 11
PEAK_MIN = 0.80
DEFAULT_THRESHOLDS = [2.3, 2.5, 2.7, 3.0]


def smooth(values, window=SMOOTH_WINDOW):
    """Trailing moving average: out[i] = mean(values[max(0,i-window+1):i+1])."""
    out, dq, s = [], deque(), 0.0
    for v in values:
        dq.append(v)
        s += v
        if len(dq) > window:
            s -= dq.popleft()
        out.append(s / len(dq))
    return out


def phi_of(row, beta):
    return math.exp((1 + beta) / (1 - beta) * row["B"]) if beta else math.exp(row["B"])


def first_crossing_ts(rows, threshold, beta):
    for row in rows:
        if phi_of(row, beta) >= threshold:
            return row["ts"]
    return None


def smoothed_acc_at_or_after(evals, sm, ts):
    for i, e in enumerate(evals):
        if e["ts"] >= ts:
            return sm[i]
    return sm[-1]


def score_arm(rid, rows, evals, thresholds, beta):
    accs = [e.get("test-accuracy") or 0 for e in evals]
    if not accs:
        print(f"  excluded {rid}: no eval records")
        return None
    # (b) the RAW peak decides "did this arm learn" -- matching P4's own peak
    # column, so an arm sitting a smoothing-window's width below 0.80 (e.g.
    # 035045, raw peak 0.801) is not dropped by a boundary artifact. The
    # SMOOTHED curve is used throughout for the given-up accounting, per spec.
    raw_peak = max(accs)
    if raw_peak < PEAK_MIN:
        print(f"  excluded {rid}: peak={raw_peak:.3f} < {PEAK_MIN} -- never learned")
        return None
    sm = smooth(accs)
    peak = max(sm)
    out = {"rid": rid, "peak": peak, "raw_peak": raw_peak, "T": len(rows)}
    for th in thresholds:
        ts = first_crossing_ts(rows, th, beta)
        # (a) never crosses -> counted as correct, given up 0 -- not skipped
        given_up = 0.0 if ts is None else max(0.0, peak - smoothed_acc_at_or_after(evals, sm, ts))
        out[th] = given_up
        out[f"{th}_crossed"] = ts is not None
    out["no_stop"] = max(0.0, peak - sm[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    ap.add_argument("--cache", default="/tmp/fwd_replay_cache")
    a = ap.parse_args()
    thresholds = [float(t) for t in a.thresholds.split(",")]

    results = []
    for run in a.runs:
        rid, path, cfg = slice_run(run, a.cache)
        if not path:
            print(f"  skipped {rid}: no telemetry", file=sys.stderr)
            continue
        m = meta(cfg, run)
        commits, evals = load(path)
        if not commits:
            print(f"  skipped {rid}: no server_update records", file=sys.stderr)
            continue
        rows = enrich(m, commits)
        beta = float(cfg.get("hyperparameters", {}).get("server_momentum") or 0.0)
        r = score_arm(rid, rows, evals, thresholds, beta)
        if r:
            results.append(r)

    if not results:
        print("no scorable arms (all excluded or missing telemetry)")
        return

    print(f"\n{'rid':9s}{'T':>6s}{'peak':>7s}" +
          "".join(f"{'phi='+str(t):>10s}" for t in thresholds) + f"{'no_stop':>10s}")
    for r in results:
        print(f"{r['rid']:9s}{r['T']:6d}{r['peak']:7.3f}" +
              "".join(f"{r[t]:10.4f}" for t in thresholds) + f"{r['no_stop']:10.4f}")

    print(f"\n{'stopping rule':22s}{'mean given up':>15s}{'worst arm':>12s}{'n crossed':>11s}")
    for t in thresholds:
        vals = [r[t] for r in results]
        n_crossed = sum(1 for r in results if r[f"{t}_crossed"])
        print(f"{'stop at Phi = ' + str(t):22s}{sum(vals)/len(vals):15.4f}"
              f"{max(vals):12.4f}{n_crossed:11d}/{len(results)}")
    vals = [r["no_stop"] for r in results]
    print(f"{'no stop -- run to end':22s}{sum(vals)/len(vals):15.4f}{max(vals):12.4f}"
          f"{len(results):11d}/{len(results)}")


if __name__ == "__main__":
    main()
