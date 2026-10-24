#!/usr/bin/env python3
"""FX-N129 gate B2 (V5): bench legs vs the harness sim legs of the same cell (phase, baseline, trace), over the matched vclock.

  bench_validate.py --bench BENCH_POOL --harness HARNESS_POOLS... [--json-out J]

Per cell (bench seeds pooled, harness seeds pooled): AUC = mean test accuracy over the matched horizon, train updates / h,
GB on the wire, mean train task time (charged budget). PASS when |AUC delta| <= max(1.5 pp, either side's seed range),
updates / h and task time within 10% / 5%, bytes within 5%. Exit 1 on any FAIL.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

AUC_PP, UPD, GB, TASK = 1.5, 0.10, 0.05, 0.05


def leg_stats(run_dir, horizon=None):
    """(end vclock, [(t, acc)], train commits, wire bytes, train budgets) up to `horizon`."""
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0]
    t, end, evals, commits, nbytes, pts = 0.0, 0.0, [], 0, 0, []
    with open(f, "rb") as fh:
        for line in fh:
            if b'"agg_round"' in line:
                e = json.loads(line)
                t = float(e.get("vclock_now") or t)
                if horizon is not None and t > horizon:
                    break
                end = t
                if e.get("task_to_perform") != "eval":  # async: one train update per event; sync: one round of K
                    commits += len(e.get("staleness") or [])
            elif b'"agg_eval"' in line:
                e = json.loads(line)
                if e.get("test-accuracy") is not None:
                    evals.append((t, float(e["test-accuracy"])))
            elif b'"wire"' in line:
                nbytes += json.loads(line)["bytes"]
    budgets = []
    for tf in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        with open(tf, "rb") as fh:
            for line in fh:
                if b'"trainer_round"' in line:
                    r = json.loads(line)
                    if r.get("task_to_perform") == "train" and (horizon is None or (r.get("sim_send_ts") or 0) <= horizon):
                        budgets.append(float(r.get("training_budget_s") or 0))
    return end, evals, commits, nbytes, budgets


def cells(pools, sim_col="sim_dir"):
    out = {}
    for p in pools:
        for f in glob.glob(os.path.join(p, "*", "summary.tsv")):
            phase = os.path.basename(os.path.dirname(f))
            for r in csv.DictReader(open(f), delimiter="\t"):
                d = r.get(sim_col) or ""
                if d and d != "-" and os.path.isdir(d):
                    out.setdefault((re.sub(r"_r\d+$", "", phase), r["baseline"], r["trace"]), []).append(d)
    return out


def summarize(dirs, horizon):
    rows = []
    for d in dirs:
        end, ev, n, b, bud = leg_stats(d, horizon)
        h = max(end, 1e-9) / 3600
        rows.append(dict(auc=st.mean(a for _, a in ev) if ev else None, upd_h=n / h, gb=b / 1e9, task=st.mean(bud) if bud else None))
    pick = lambda k: [r[k] for r in rows if r[k] is not None]
    return {k: (st.mean(pick(k)), max(pick(k)) - min(pick(k))) if pick(k) else (None, None) for k in rows[0]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--harness", nargs="+", required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    bench, harness = cells(a.bench), cells(a.harness)
    out, bad = [], 0
    print("| cell | n b/h | AUC b / h (Δ pp) | upd/h b / h | GB b / h | task s b / h | verdict |\n|---|---|---|---|---|---|---|")
    for key in sorted(bench):
        if key not in harness:
            continue
        horizon = min(leg_stats(d)[0] for d in bench[key] + harness[key])
        b, h = summarize(bench[key], horizon), summarize(harness[key], horizon)
        rel = lambda k: abs(b[k][0] - h[k][0]) / h[k][0] if h[k][0] else 0.0
        d_auc = 100 * (b["auc"][0] - h["auc"][0]) if b["auc"][0] is not None and h["auc"][0] is not None else None
        tol = max(AUC_PP, 100 * (b["auc"][1] or 0), 100 * (h["auc"][1] or 0))
        miss = ([f"AUC {d_auc:+.1f}>{tol:.1f}"] if d_auc is None or abs(d_auc) > tol else []) + \
            ([f"upd {100 * rel('upd_h'):.0f}%"] if rel("upd_h") > UPD else []) + \
            ([f"GB {100 * rel('gb'):.0f}%"] if rel("gb") > GB else []) + \
            ([f"task {100 * rel('task'):.0f}%"] if b["task"][0] and rel("task") > TASK else [])
        bad += bool(miss)
        f = lambda x, s=1: "–" if x is None else f"{x:.{s}f}"
        print(f"| {' '.join(key)} | {len(bench[key])}/{len(harness[key])} | {f(100 * b['auc'][0])} / {f(100 * h['auc'][0])} "
              f"({f(d_auc)}) | {f(b['upd_h'][0], 0)} / {f(h['upd_h'][0], 0)} | {f(b['gb'][0])} / {f(h['gb'][0])} | "
              f"{f(b['task'][0])} / {f(h['task'][0])} | {'PASS' if not miss else 'FAIL ' + ', '.join(miss)} |")
        out.append(dict(cell=key, horizon_s=horizon, bench=b, harness=h, miss=miss))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
