#!/usr/bin/env python3
"""FX-N109: per-arm medians of a knob-sweep pool (F39A, F39B, F39C, F39E, F39W) against its control arm.

  knob_sweep.py POOL... [--json]

Per leg: max accuracy, mean accuracy over the last 30% of the leg's clock (smooths one-eval noise), time and model bytes to 40% / 50% (cifar) or 40% / 60% (speech), versions, mean staleness,
top-20% update share, eval picks. Per arm: median over replicates and how many replicates reached each threshold.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import TARGETS, _agg_events, curve, dataset_of, rows_from_pools  # noqa: E402
from felix_claims import leg_stats  # noqa: E402

ARM = re.compile(r"(?:gs_)?F39[ABCSEW]_(\w+?)_r(\d+)$")


def leg_metrics(run_dir):
    ds = dataset_of(run_dir)
    ths = (0.4, TARGETS[ds])
    st = leg_stats(run_dir, ths)
    if not st:
        return None
    versions, stale, counts, evals = 0, [], collections.Counter(), 0
    for e in _agg_events(run_dir):
        if e.get("event") == "agg_round":
            versions = max(versions, e.get("round") or 0)
            stale += [s for s in e.get("staleness") or [] if s is not None]
            counts.update(e.get("contributing_trainers") or [])
    for f in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[:1]:  # _agg_events drops selection
        with open(f, "rb") as fh:
            for line in fh:
                if b'"selection"' in line and b'"eval"' in line and (e := json.loads(line)).get("task") == "eval":
                    evals += e.get("num_chosen") or 0
    tot = sum(counts.values())
    top = sorted(counts.values(), reverse=True)[:max(1, len(counts) // 5)]
    c, _ = curve(run_dir)
    tail = [a for t, _r, a in c if t >= 0.7 * c[-1][0]]
    out = {"max": st["max"], "last30": statistics.mean(tail), "versions": versions, "stale": statistics.mean(stale) if stale else None,
           "top20": sum(top) / tot if tot else None, "eval_picks": evals}
    for th in ths:
        t, b = st[th]
        out[f"t{int(th * 100)}"] = t / 60 if t is not None else None
        out[f"gb{int(th * 100)}"] = b / 1e9 if b is not None else None
    return out


def collect(pools):
    arms = collections.defaultdict(list)  # (dataset, baseline, arm) -> [metrics]
    for phase, b, _tr, _real, sim in rows_from_pools(pools):
        m = ARM.match(phase)
        if m and sim and os.path.isdir(sim) and (lm := leg_metrics(sim)):
            arms[(dataset_of(sim), b, m.group(1))].append(lm)
    return arms


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def table(arms):
    keys = ["max", "last30", "t40", "gb40", "versions", "stale", "top20", "eval_picks"]
    rows = []
    for (ds, b, arm), ms in sorted(arms.items()):
        tg = f"t{int(TARGETS[ds] * 100)}"
        r = {"dataset": ds, "baseline": b, "arm": arm, "reps": len(ms),
             **{k: med(m[k] for m in ms) for k in keys + [tg, "gb" + tg[1:]]},
             "reach40": sum(m["t40"] is not None for m in ms), "reach_target": sum(m[tg] is not None for m in ms)}
        rows.append(r)
    return rows


def fmt(v, nd=1):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = table(collect(a.pools))
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    ctl = {(r["dataset"], r["baseline"]): r for r in rows if r["arm"] in ("ctl", "K10c15")}  # K10c15 = F39B control
    print("| dataset | baseline | arm | reps | max % | last-30% % | min to 40% (reach) | GB to 40% | min to target (reach) | GB to target "
          "| versions | staleness | top-20% share | eval picks | Δ min to 40% vs ctl |")
    print("|" + "---|" * 15)
    for r in rows:
        tg = f"t{int(TARGETS[r['dataset']] * 100)}"
        c = ctl.get((r["dataset"], r["baseline"]))
        d = r["t40"] - c["t40"] if c and r["t40"] is not None and c["t40"] is not None else None
        print(f"| {r['dataset']} | {r['baseline']} | {r['arm']} | {r['reps']} | {fmt(r['max'] and r['max'] * 100)} | {fmt(r['last30'] and r['last30'] * 100)} "
              f"| {fmt(r['t40'])} ({r['reach40']}) | {fmt(r['gb40'])} | {fmt(r[tg])} ({r['reach_target']}) "
              f"| {fmt(r['gb' + tg[1:]])} | {fmt(r['versions'], 0)} | {fmt(r['stale'], 2)} | {fmt(r['top20'], 2)} "
              f"| {fmt(r['eval_picks'], 0)} | {fmt(d)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
