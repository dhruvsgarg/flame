#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Characterize the per-databin variance-decay curve from a run's telemetry.

Charter §5c Opt-2: before designing the variance-gate `stopping_policy`
(fixed_cap / plateau / adaptive), measure the actual decay curve -- where var
starts, how fast it falls, where it plateaus, and whether the plateau drops as
training progresses.

Streams the aggregator telemetry, reconstructs each data-bin (robust to the
`data_id` cycling footgun via COMMIT counting -- see audit_weight_redundancy.py),
extracts each bin's (iteration -> var) curve, and runs two counterfactual sweeps:

  * fixed_cap sweep  -- for a cap K: iterations/forward-passes saved, bins
    affected, and how much worse the committed var is vs the natural commit var
    (the accuracy risk of committing early).
  * plateau sweep    -- for (patience N, tol eps): where a diminishing-returns
    rule fires and at what var, vs the fixed cap.

Usage:
  python characterize_variance_curve.py <run_dir_or_agg_jsonl> [--caps 7,8,10,12,15]
      [--plateau-N 3 --plateau-eps 0.05,0.10,0.15] [--json OUT.json]

Pure read; streaming/prefilter style so it runs over multi-GB logs.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict


def _agg_jsonl(path):
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "telemetry", "aggregator_*.jsonl"))
        if not hits:
            sys.exit(f"no aggregator_*.jsonl under {path}/telemetry/")
        return hits[0]
    return path


def _pct(xs, q):
    """Nearest-rank percentile (q in [0,100]); None on empty."""
    if not xs:
        return None
    s = sorted(xs)
    if q <= 0:
        return s[0]
    if q >= 100:
        return s[-1]
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def load_curves(path, max_databins=None):
    """Stream agg_round events -> {databin_id: [(iteration, var, committed), ...]}.

    Data-bin id is the running commit count: a bin ends when an agg_round commits
    (`var_good_enough=True`). Uses `cycle_iteration` (unambiguous, unlike the
    post-mutation `iteration_per_data_id`) as the intra-bin axis. `max_databins`
    caps to the first N committed bins for like-for-like truncation across runs.

    Also returns each committed bin's `commit_reason` (Opt-2 telemetry:
    natural / cap / plateau; None on legacy runs).
    """
    f = _agg_jsonl(path)
    databin = 0
    curves = defaultdict(list)  # databin -> [(iter, var, committed)]
    reasons = {}                # databin -> commit_reason on the committing cycle
    thr_seen = set()
    with open(f) as fh:
        for line in fh:
            if '"agg_round"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "agg_round":
                continue
            var = e.get("var")
            it = e.get("cycle_iteration")
            if it is None:
                it = e.get("iteration_per_data_id")
            committed = bool(e.get("var_good_enough"))
            thr = e.get("var_threshold")
            if thr is not None:
                thr_seen.add(thr)
            if var is not None:
                curves[databin].append((it, float(var), committed))
            if committed:
                reasons[databin] = e.get("commit_reason")
                databin += 1
                if max_databins is not None and databin >= max_databins:
                    break
    # sort each bin by iteration; drop a trailing open (never-committed) bin
    out = {}
    for b, pts in curves.items():
        if max_databins is not None and b >= max_databins:
            continue
        out[b] = sorted(pts, key=lambda p: (p[0] if p[0] is not None else 0))
    threshold = min(thr_seen) if thr_seen else None
    return out, threshold, reasons


def _plateau_onset(vars_, N, eps):
    """First iteration index i (>=N) where the relative var drop over the last N
    steps is < eps -- i.e. diminishing returns. Returns (onset_idx, var_at_onset)
    or (None, None) if it never plateaus within the bin."""
    for i in range(N, len(vars_)):
        prev = vars_[i - N]
        cur = vars_[i]
        if prev <= 0:
            continue
        rel_drop = (prev - cur) / prev
        if rel_drop < eps:
            return i, cur
    return None, None


def characterize(curves, threshold, caps, plateau_N, plateau_eps_list, reasons=None):
    committed_bins = {b: pts for b, pts in curves.items() if any(c for _, _, c in pts)}
    n_bins = len(committed_bins)

    # Opt-2 commit-reason split (natural gate / max-iter cap / plateau); None on
    # legacy runs (policy off) -> reported as "natural(legacy)".
    reason_counts = defaultdict(int)
    if reasons:
        for b in committed_bins:
            r = reasons.get(b)
            reason_counts["natural(legacy)" if r is None else r] += 1

    per_bin = []  # list of dicts, in databin order
    for b in sorted(committed_bins):
        pts = committed_bins[b]
        vars_ = [v for _, v, _ in pts]
        n_iters = len(pts)
        commit_idx = next((i for i, (_, _, c) in enumerate(pts) if c), n_iters - 1)
        v0 = vars_[0]
        v_commit = vars_[commit_idx]
        v_min = min(vars_)
        # plateau onset at the *reference* (N, eps[0]) for characterization
        onset, v_onset = _plateau_onset(vars_, plateau_N, plateau_eps_list[0])
        per_bin.append({
            "databin": b,
            "n_iters": n_iters,
            "commit_idx": commit_idx,       # 0-based iteration at which it committed
            "var_initial": v0,
            "var_commit": v_commit,
            "var_min": v_min,
            "plateau_onset": onset,
            "var_at_plateau": v_onset,
        })

    def dist(key):
        xs = [d[key] for d in per_bin if d[key] is not None]
        return {"p10": _pct(xs, 10), "p50": _pct(xs, 50),
                "p90": _pct(xs, 90), "mean": (st.mean(xs) if xs else None), "n": len(xs)}

    summary = {k: dist(k) for k in
               ["n_iters", "commit_idx", "var_initial", "var_commit",
                "var_min", "plateau_onset", "var_at_plateau"]}

    # --- evolution over training: split databins into thirds by order ---
    thirds = {}
    order = sorted(committed_bins)
    if order:
        third = max(1, len(order) // 3)
        buckets = {"early": order[:third], "mid": order[third:2 * third], "late": order[2 * third:]}
        idx = {d["databin"]: d for d in per_bin}
        for name, bs in buckets.items():
            iters = [idx[b]["n_iters"] for b in bs]
            vcommit = [idx[b]["var_commit"] for b in bs]
            vmin = [idx[b]["var_min"] for b in bs]
            onset = [idx[b]["plateau_onset"] for b in bs if idx[b]["plateau_onset"] is not None]
            thirds[name] = {
                "n_bins": len(bs),
                "median_n_iters": _pct(iters, 50),
                "median_var_commit": _pct(vcommit, 50),
                "median_var_min": _pct(vmin, 50),
                "median_plateau_onset": _pct(onset, 50),
            }

    # --- counterfactual fixed_cap sweep ---
    # A cap K forces commit at iteration index min(K-1, natural_commit_idx). We only
    # SAVE work on bins whose natural commit happened later than K-1.
    cap_sweep = []
    total_natural_iters = sum(d["n_iters"] for d in per_bin)
    for K in caps:
        saved = 0
        affected = 0
        for d in per_bin:
            nat_idx = d["commit_idx"]
            if nat_idx > K - 1:            # bin grinds past the cap -> we cut it
                affected += 1
                saved += nat_idx - (K - 1)
        cap_sweep.append({
            "cap": K,
            "bins_affected": affected,
            "frac_bins_affected": (affected / n_bins) if n_bins else None,
            "iters_saved": saved,
            "frac_iters_saved": (saved / total_natural_iters) if total_natural_iters else None,
        })

    # need the raw vars to report forced-commit var quality; recompute cleanly
    cap_quality = []
    for K in caps:
        forced = []
        for b in sorted(committed_bins):
            pts = committed_bins[b]
            vars_ = [v for _, v, _ in pts]
            commit_idx = next((i for i, (_, _, c) in enumerate(pts) if c), len(pts) - 1)
            if commit_idx > K - 1 and len(vars_) >= K:
                forced.append(vars_[K - 1])
        cap_quality.append({
            "cap": K,
            "var_at_cap_p50": _pct(forced, 50),
            "var_at_cap_p90": _pct(forced, 90),
            "var_at_cap_max": _pct(forced, 100),
            "n": len(forced),
        })

    # --- counterfactual plateau sweep ---
    plateau_sweep = []
    for eps in plateau_eps_list:
        fired = 0
        fire_idx = []
        fire_var = []
        for b in sorted(committed_bins):
            vars_ = [v for _, v, _ in committed_bins[b]]
            onset, v = _plateau_onset(vars_, plateau_N, eps)
            if onset is not None:
                fired += 1
                fire_idx.append(onset)
                fire_var.append(v)
        plateau_sweep.append({
            "patience_N": plateau_N,
            "eps": eps,
            "bins_fired": fired,
            "frac_bins_fired": (fired / n_bins) if n_bins else None,
            "fire_iter_p50": _pct(fire_idx, 50),
            "fire_iter_p90": _pct(fire_idx, 90),
            "var_at_fire_p50": _pct(fire_var, 50),
            "var_at_fire_p90": _pct(fire_var, 90),
        })

    return {
        "n_committed_bins": n_bins,
        "var_threshold": threshold,
        "total_natural_iters": total_natural_iters,
        "commit_reasons": dict(reason_counts),
        "summary": summary,
        "evolution_thirds": thirds,
        "cap_sweep": cap_sweep,
        "cap_quality": cap_quality,
        "plateau_sweep": plateau_sweep,
    }


def _fmt(x, nd=3):
    return "None" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_report(r):
    print(f"\n=== Variance-decay curve characterization ===")
    print(f"committed data-bins: {r['n_committed_bins']}   "
          f"var_threshold: {_fmt(r['var_threshold'])}   "
          f"total iters (compute): {r['total_natural_iters']}")
    if r.get("commit_reasons"):
        split = "  ".join(f"{k}={v}" for k, v in sorted(r["commit_reasons"].items()))
        print(f"commit reasons: {split}")

    print(f"\n-- per-bin distribution (p10 / p50 / p90 / mean) --")
    s = r["summary"]
    rows = [
        ("iters per bin", "n_iters", 1),
        ("commit iter idx", "commit_idx", 1),
        ("var @ iter 0", "var_initial", 3),
        ("var @ commit", "var_commit", 3),
        ("var min in bin", "var_min", 3),
        ("plateau onset iter", "plateau_onset", 1),
        ("var @ plateau", "var_at_plateau", 3),
    ]
    for label, key, nd in rows:
        d = s[key]
        print(f"  {label:20s}: {_fmt(d['p10'],nd):>8} / {_fmt(d['p50'],nd):>8} / "
              f"{_fmt(d['p90'],nd):>8} / {_fmt(d['mean'],nd):>8}   (n={d['n']})")

    print(f"\n-- evolution over training (databins split in thirds) --")
    ev = r["evolution_thirds"]
    print(f"  {'phase':6s} {'bins':>5} {'med_iters':>10} {'med_var_commit':>15} "
          f"{'med_var_min':>12} {'med_onset':>10}")
    for name in ["early", "mid", "late"]:
        if name in ev:
            d = ev[name]
            print(f"  {name:6s} {d['n_bins']:>5} {_fmt(d['median_n_iters'],1):>10} "
                  f"{_fmt(d['median_var_commit'],3):>15} {_fmt(d['median_var_min'],3):>12} "
                  f"{_fmt(d['median_plateau_onset'],1):>10}")

    print(f"\n-- fixed_cap sweep (iters/forward-passes saved) --")
    print(f"  {'cap':>4} {'bins_hit':>9} {'%bins':>7} {'iters_saved':>12} {'%iters':>8}")
    for d in r["cap_sweep"]:
        print(f"  {d['cap']:>4} {d['bins_affected']:>9} "
              f"{_fmt(100*(d['frac_bins_affected'] or 0),1):>7} {d['iters_saved']:>12} "
              f"{_fmt(100*(d['frac_iters_saved'] or 0),1):>8}")
    print(f"\n-- fixed_cap quality (var we'd COMMIT at on capped bins; lower=safer) --")
    print(f"  {'cap':>4} {'var@cap p50':>12} {'p90':>8} {'max':>8}   (natural commit var p50~thr)")
    for d in r["cap_quality"]:
        print(f"  {d['cap']:>4} {_fmt(d['var_at_cap_p50'],3):>12} "
              f"{_fmt(d['var_at_cap_p90'],3):>8} {_fmt(d['var_at_cap_max'],3):>8}   (n={d['n']})")

    print(f"\n-- plateau rule sweep (patience N, tol eps) --")
    print(f"  {'N':>3} {'eps':>6} {'bins_fired':>11} {'%':>6} "
          f"{'fire_iter p50/p90':>18} {'var@fire p50/p90':>18}")
    for d in r["plateau_sweep"]:
        fi = f"{_fmt(d['fire_iter_p50'],0)}/{_fmt(d['fire_iter_p90'],0)}"
        vf = f"{_fmt(d['var_at_fire_p50'],3)}/{_fmt(d['var_at_fire_p90'],3)}"
        print(f"  {d['patience_N']:>3} {_fmt(d['eps'],2):>6} {d['bins_fired']:>11} "
              f"{_fmt(100*(d['frac_bins_fired'] or 0),1):>6} {fi:>18} {vf:>18}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run dir or aggregator_*.jsonl")
    ap.add_argument("--caps", default="7,8,10,12,15",
                    help="comma list of fixed_cap values to sweep")
    ap.add_argument("--plateau-N", type=int, default=3, help="plateau patience window")
    ap.add_argument("--plateau-eps", default="0.05,0.10,0.15",
                    help="comma list of relative-drop tolerances to sweep")
    ap.add_argument("--max-databins", type=int, default=None,
                    help="only analyze the first N committed data-bins (like-for-like truncation)")
    ap.add_argument("--json", default=None, help="write full result as JSON here")
    a = ap.parse_args()

    caps = [int(x) for x in a.caps.split(",") if x.strip()]
    eps_list = [float(x) for x in a.plateau_eps.split(",") if x.strip()]

    curves, threshold, reasons = load_curves(a.run, max_databins=a.max_databins)
    if not curves:
        sys.exit("no agg_round events with var found")
    r = characterize(curves, threshold, caps, a.plateau_N, eps_list, reasons=reasons)
    print_report(r)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
