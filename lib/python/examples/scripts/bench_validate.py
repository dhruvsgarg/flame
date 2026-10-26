#!/usr/bin/env python3
"""FX-N129 gate B2 (V5): bench legs vs the harness sim legs of the same cell (phase, baseline, trace), over the matched vclock.

  bench_validate.py --bench BENCH_POOL --harness HARNESS_POOLS... [--json-out J]

Per cell (bench seeds pooled, harness seeds pooled): AUC = mean test accuracy over the matched horizon, train updates / h,
GB on the wire, mean train task time (charged budget). PASS when |AUC delta| <= max(1.5 pp, either side's seed range),
updates / h, task time and bytes within 10% / 5% / 5% or either side's relative seed range (Q-B2BAR). fwdllm legs (FLUXTUNE FT-N18) add iterations per data_id at commit,
variance pass rate and commits / h, each within 10%. Exit 1 on any FAIL.
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
FWD = {"iters": 0.10, "var_pass": 0.10, "commits_h": 0.10}  # FT-N18: the fwdllm cadence bands


def leg_stats(run_dir, horizon=None):
    """(end vclock, [(t, acc)], train commits, wire bytes, train budgets, fwdllm cadence) up to `horizon`."""
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0]
    t, end, evals, commits, nbytes, pts = 0.0, 0.0, [], 0, 0, []
    iters, var_checks, var_pass = [], 0, 0
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
                if e.get("var") is not None:  # fwdllm: one variance check per agg round
                    var_checks += 1
                    var_pass += bool(e.get("var_good_enough"))
                if e.get("commit_reason") is not None:
                    iters.append(int(e.get("cycle_iteration") or 0))  # the committed cycle's, not the reset counter
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
    h = max(end, 1e-9) / 3600
    fwd = dict(iters=st.mean(iters), var_pass=var_pass / var_checks if var_checks else None,
               commits_h=len(iters) / h) if iters else {}
    return end, evals, commits, nbytes, budgets, fwd


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
        end, ev, n, b, bud, fwd = leg_stats(d, horizon)
        h = max(end, 1e-9) / 3600
        rows.append(dict(auc=st.mean(a for _, a in ev) if ev else None, upd_h=n / h, gb=b / 1e9, task=st.mean(bud) if bud else None,
                         **{k: fwd.get(k) for k in FWD}))
    pick = lambda k: [r[k] for r in rows if r[k] is not None]
    return {k: (st.mean(pick(k)), max(pick(k)) - min(pick(k))) if pick(k) else (None, None) for k in rows[0]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--harness", nargs="+", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--upd", type=float, default=UPD, help="updates / h tolerance (FX-N151: 0.05 for the sync stacks)")
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
        # Q-B2BAR: a flat bar can fail on replicate noise, so it widens to the larger seed range (as for AUC); needs >= 2 legs/side.
        spread = lambda k: max((x[k][1] / x[k][0] if x[k][0] else 0.0) for x in (b, h))
        bar = lambda k, flat: max(flat, spread(k))
        d_auc = 100 * (b["auc"][0] - h["auc"][0]) if b["auc"][0] is not None and h["auc"][0] is not None else None
        tol = max(AUC_PP, 100 * (b["auc"][1] or 0), 100 * (h["auc"][1] or 0))
        miss = ([f"AUC {d_auc:+.1f}>{tol:.1f}"] if d_auc is None or abs(d_auc) > tol else []) + \
            ([f"upd {100 * rel('upd_h'):.0f}%"] if rel("upd_h") > bar("upd_h", a.upd) else []) + \
            ([f"GB {100 * rel('gb'):.0f}%"] if rel("gb") > bar("gb", GB) else []) + \
            ([f"task {100 * rel('task'):.0f}%"] if b["task"][0] and rel("task") > bar("task", TASK) else []) + \
            [f"{k} {100 * rel(k):.0f}%" for k, tol_k in FWD.items() if b[k][0] is not None and h[k][0] is not None and rel(k) > tol_k]
        bad += bool(miss)
        f = lambda x, s=1: "–" if x is None else f"{x:.{s}f}"
        print(f"| {' '.join(key)} | {len(bench[key])}/{len(harness[key])} | {f(100 * b['auc'][0])} / {f(100 * h['auc'][0])} "
              f"({f(d_auc)}) | {f(b['upd_h'][0], 0)} / {f(h['upd_h'][0], 0)} | {f(b['gb'][0])} / {f(h['gb'][0])} | "
              f"{f(b['task'][0])} / {f(h['task'][0])} | {'PASS' if not miss else 'FAIL ' + ', '.join(miss)} |")
        if b["iters"][0] is not None and h["iters"][0] is not None:
            print(f"|   fwdllm: iters/data_id {f(b['iters'][0], 2)} / {f(h['iters'][0], 2)}, var pass "
                  f"{f(b['var_pass'][0], 3)} / {f(h['var_pass'][0], 3)}, commits/h {f(b['commits_h'][0])} / {f(h['commits_h'][0])} | | | | | |")
        out.append(dict(cell=key, horizon_s=horizon, bench=b, harness=h, miss=miss))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
