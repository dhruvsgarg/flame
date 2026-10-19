#!/usr/bin/env python3
"""Felix claims gate (FX-N108; FELIX Scoreboard "Felix claims"): per dataset x scenario, every baseline's time and traffic to target.

  felix_claims.py POOL [POOL ...] [--runs RUN ...]

C1 every baseline reaches target; on speech a baseline below target passes as round-bound (operator 10-18) when it is learning
(median eval loss falls ≥ LOSS_DROP first → last third, accuracy rises) or, under unavailability, when it is near-degenerate but every
abandon fires at the task timeout and its slot is re-dispatched (FX-N106) · C2 Felix's time-to-target lowest by >= MARGIN · C3 Felix's model traffic (down + up) to target
lowest · C4 Felix's max-accuracy drop vs syn_0 smallest among baselines that reach target. A claim no leg can grade (nobody reaches
target) prints n/a; anything else that fails prints a WARNING line. Real legs win over sims for the same cell. Clock = the leg's own
(accuracy_table.curve); traffic = aggregator `wire` messages >= 100 kB (models), sim wall stamps mapped onto its vclock.
"""
import argparse
import bisect
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import TARGETS, _agg_events, curve, dataset_of, rows_from_pools  # noqa: E402

BIG = 100_000
MARGIN = 0.9  # "significantly lower": Felix <= 0.9 x the best other baseline
ROUND_BOUND = {"google_speech", "cifar10"}  # operator 10-18 (speech), 10-19 (cifar): 2 h legs may end below target
LOSS_DROP = 0.05  # "not flat": last-third median eval loss ≥ 5% under the first third's (median: one spike can't flip it)
MIN_EVALS = 6
REDISPATCH_S = 60.0  # FX-N106: a timed-out slot is re-dispatched within this (median)


def round_bound(run_dir, scenario):
    """'learning' / 'degenerate' when a below-target leg counts as round-bound, else None."""
    ev = _agg_events(run_dir)
    evals = sorted((e["round"], e["test-loss"], e["test-accuracy"]) for e in ev
                   if e.get("event") == "agg_eval" and e.get("test-loss") is not None and e.get("test-accuracy") is not None)
    if len(evals) >= MIN_EVALS:
        k = len(evals) // 3
        med = lambda xs, i: sorted(x[i] for x in xs)[len(xs) // 2]
        if med(evals[-k:], 1) <= (1 - LOSS_DROP) * med(evals[:k], 1) and med(evals[-k:], 2) > med(evals[:k], 2):
            return "learning"
        return None
    if scenario == "syn_0":
        return None
    ab, disp = [], []
    for f in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[:1]:
        with open(f, "rb") as fh:
            for line in fh:
                if b'"abandon_timeout"' in line or b'"dispatch"' in line:
                    e = json.loads(line)
                    (ab if e["event"] == "abandon_timeout" else disp).append(e)
    disp_ts = sorted(e["ts"] for e in disp)
    gaps = sorted(disp_ts[i] - e["ts"] for e in ab if (i := bisect.bisect_right(disp_ts, e["ts"])) < len(disp_ts))
    ages = [e.get("age_s") or 0.0 for e in ab]
    on_time = bool(ages) and max(ages) - min(ages) < 1.0  # every abandon at the one task timeout
    return "degenerate" if gaps and on_time and gaps[len(gaps) // 2] <= REDISPATCH_S else None


def traffic(run_dir):
    """[(t_s on the leg's clock, bytes)] of model-carrying messages, both directions."""
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not f:
        return []
    sim = run_dir.rstrip("/").endswith("_sim")
    wires, pts, t0 = [], [], None
    with open(f[0], "rb") as fh:
        for line in fh:
            if b'"wire"' in line:
                e = json.loads(line)
                if e.get("bytes", 0) >= BIG:
                    wires.append((e["ts"], e["bytes"]))
            elif b'"agg_round"' in line and sim:
                e = json.loads(line)
                if e.get("vclock_now") is not None:
                    pts.append((e["ts"], float(e["vclock_now"])))
            elif b'"trace_origin"' in line:
                t0 = json.loads(line).get("origin_ts")
    if sim:
        pts.sort()
        xs = [p[0] for p in pts]
        return [(pts[max(bisect.bisect_right(xs, ts) - 1, 0)][1], b) for ts, b in wires] if pts else []
    t0 = t0 or (wires[0][0] if wires else 0)
    return [(ts - t0, b) for ts, b in wires]


def leg_stats(run_dir, ths):
    c, _ = curve(run_dir)
    if not c:
        return None
    tr = traffic(run_dir)
    out = {"max": max(a for *_, a in c)}
    for th in ths:
        t = next((t for t, _, a in c if a >= th), None)
        out[th] = (t, sum(b for tt, b in tr if tt <= t) if t is not None else None)
    return out


def scenario(src, trace):
    """lin / eve for stream legs (G0T, G3 lin), else the trace."""
    return "lin" if "_lin_" in src else ("eve" if "_eve_" in src else trace)


def collect(pools, runs):
    cells = {}  # (dataset, scenario, baseline) -> run_dir, real first
    legs = list(rows_from_pools(pools)) + [(os.path.basename(r), "", "", r, "") for r in runs]
    for phase, b, tr, real, sim in legs:
        for d in (real, sim):
            if not d or not os.path.isdir(d):
                continue
            name = os.path.basename(d.rstrip("/"))
            b = b or re.search(r"_dbg_(?:google_speech_)?(\w+?)_n\d+", name).group(1)
            tr = tr or next((t for t in ("syn_50_3st", "syn_20_3st", "mobiperf_3st", "syn_0") if t in name), "?")
            k = (dataset_of(d), scenario(f"{phase} {name}", tr), b)
            if k not in cells or (cells[k].endswith("_sim") and not d.endswith("_sim")):
                cells[k] = d
    return cells


def grade(cells):
    rows, warnings = [], []
    stats = {k: leg_stats(d, (TARGETS[k[0]], 0.4)) for k, d in cells.items()}
    bound = {k: round_bound(d, k[1]) for k, d in cells.items()
             if k[0] in ROUND_BOUND and stats.get(k) and stats[k][TARGETS[k[0]]][0] is None}
    for ds, sc in sorted({(k[0], k[1]) for k in cells}):
        tg = TARGETS[ds]
        bs = {k[2]: v for k, v in stats.items() if k[:2] == (ds, sc) and v}
        if "felix" not in bs:
            continue
        reach = {b for b, v in bs.items() if v[tg][0] is not None}
        others = {b: v for b, v in bs.items() if b != "felix"}

        def best(th, i):
            vals = [(v[th][i], b) for b, v in others.items() if v[th][i] is not None]
            return min(vals) if vals else None
        f = bs["felix"]
        rb = {b: bound[(ds, sc, b)] for b in bs if bound.get((ds, sc, b))}
        miss = sorted(set(bs) - reach - set(rb))
        c1 = (f"{'✅' if not miss else '❌'} {len(reach)}/{len(bs)} reach"
              + (f", round-bound {', '.join(f'{b} ({v})' for b, v in sorted(rb.items()))}" if rb else "")
              + (f", miss {', '.join(miss)}" if miss else ""))
        verdict = {}
        for name, th, i in (("C2", tg, 0), ("C2'", 0.4, 0), ("C3", tg, 1), ("C3'", 0.4, 1)):
            mine, other = f[th][i], best(th, i)
            if mine is None:
                verdict[name] = "n/a" if other is None else f"❌ felix never, {other[1]} {other[0] / (60 if i == 0 else 1e9):.0f}"
            elif other is None:
                verdict[name] = "✅ only felix"
            else:
                ok = mine <= MARGIN * other[0] if i == 0 else mine <= other[0]
                unit, div = ("min", 60) if i == 0 else ("GB", 1e9)
                verdict[name] = f"{'✅' if ok else '❌'} {mine / div:.1f} vs {other[1]} {other[0] / div:.1f} {unit}"
        base = {b: stats.get((ds, "syn_0", b)) for b in bs}
        drops = {b: (base[b]["max"] - v["max"]) for b, v in bs.items() if base.get(b) and b in reach | {"felix"}}
        if sc == "syn_0" or "felix" not in drops:
            verdict["C4"] = "–"
        elif "felix" not in reach:
            verdict["C4"] = "n/a (felix below target)"
        else:
            worst = min((d, b) for b, d in drops.items() if b != "felix") if len(drops) > 1 else None
            verdict["C4"] = ("✅ only felix" if worst is None else
                             f"{'✅' if drops['felix'] <= worst[0] else '❌'} {-100 * drops['felix']:+.1f} vs {worst[1]} {-100 * worst[0]:+.1f} pp")
        rows.append((ds, sc, c1, verdict))
        for k, v in [("C1", c1), *verdict.items()]:
            if v.startswith("❌"):
                warnings.append(f"WARNING {ds} {sc} {k}: {v}")
    return rows, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*")
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    rows, warnings = grade(collect(a.pools, a.runs))
    print("| dataset | scenario | C1 reach | C2 TTA target | C2' TTA 40% | C3 traffic target | C3' traffic 40% | C4 Δ max |")
    print("|---|---|---|---|---|---|---|---|")
    for ds, sc, c1, v in rows:
        print(f"| {ds} | {sc} | {c1} | " + " | ".join(v[k] for k in ("C2", "C2'", "C3", "C3'", "C4")) + " |")
    for w in warnings:
        print(w)
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
