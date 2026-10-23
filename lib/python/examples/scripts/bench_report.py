#!/usr/bin/env python3
"""FX-N131: per arm x cell over seeds (bench or harness pools): AUC, time and bytes to targets, final accuracy, total GB.

  bench_report.py POOL... [--targets 0.35,0.4] [--json-out R.json] [--ceiling C.json ...]

AUC = mean test accuracy over the leg's horizon (the Scoreboard's); TTA = first (3-eval mean for speech) eval >= target on the
leg's clock; GB to target = wire bytes (both directions) up to that eval; final = mean accuracy over the last 10% of the horizon.
Cells print mean and seed range (max - min); `--ceiling` adds the centralized ceiling at the leg's end (scripts/ceiling.py).
"""
import argparse
import csv
import glob
import json
import multiprocessing
import os
import re
import statistics as st
import sys

TARGETS = {"cifar10": (0.35, 0.40), "google_speech": (0.25, 0.30)}


def leg(run_dir, event=b'"agg_eval"'):
    """[(t, acc, cumulative bytes)] and the end clock, streamed in event order (t = the last commit's vclock).
    `event` b'"agg_eval_raw"' reads a serveEma leg's unserved model (FX-N128)."""
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0]
    t, nbytes, pts = 0.0, 0, []
    wire, vclk, acc = re.compile(rb'"bytes": (\d+)'), re.compile(rb'"vclock_now": ([-0-9.e]+)'), re.compile(rb'"test-accuracy": ([-0-9.e]+)')
    with open(f, "rb") as fh:
        for line in fh:
            if b'"wire"' in line:
                nbytes += int(wire.search(line).group(1))
            elif b'"agg_round"' in line:
                m = vclk.search(line)
                t = float(m.group(1)) if m else t
            elif event in line and (m := acc.search(line)):
                pts.append((t, float(m.group(1)), nbytes))
    return pts, t, nbytes


def stats(run_dir, targets, smooth, event=b'"agg_eval"'):
    pts, end, nbytes = leg(run_dir, event)
    if not pts:
        return None
    acc = [a for _, a, _ in pts]
    sm = [st.mean(acc[max(0, i - smooth + 1):i + 1]) for i in range(len(acc))]
    out = dict(auc=st.mean(acc), final=st.mean(a for t, a, _ in pts if t >= 0.9 * end) if end else acc[-1], gb=nbytes / 1e9,
               end_h=end / 3600)
    for th in targets:
        hit = next((i for i, a in enumerate(sm) if a >= th), None)
        out[f"tta{th:g}"] = pts[hit][0] / 60 if hit is not None else None
        out[f"gb{th:g}"] = pts[hit][2] / 1e9 if hit is not None else None
    return out


def cells(pools):
    """(dataset, trace, phase arm, baseline) -> run dirs; bench layout and harness summary.tsv alike."""
    out = {}
    for p in pools:
        for f in glob.glob(os.path.join(p, "*", "summary.tsv")):
            phase = os.path.basename(os.path.dirname(f))
            for r in csv.DictReader(open(f), delimiter="\t"):
                d = r.get("sim_dir") or ""
                if d and d != "-" and os.path.isdir(d):
                    ds = "google_speech" if "google_speech" in d else "cifar10"
                    arm = re.sub(r"^(gs_)?F\d+\w*?_s\d+u?_", "", re.sub(r"_r\d+$", "", phase))
                    out.setdefault((ds, r["trace"], arm, r["baseline"]), []).append(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--targets", default="")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    rows, cs = [], sorted(cells(a.pools).items())
    tgt = lambda ds: [float(x) for x in a.targets.split(",") if x] or TARGETS[ds]
    raw = [((ds, trace, arm + ":raw", b), dirs) for (ds, trace, arm, b), dirs in cs if arm.endswith("_srv")]  # FX-N128
    jobs = [(d, tgt(ds), 3 if ds == "google_speech" else 1, ev) for (ds, _, arm, _), dirs in cs + raw for d in dirs
            for ev in [b'"agg_eval_raw"' if arm.endswith(":raw") else b'"agg_eval"']]
    with multiprocessing.Pool(min(32, len(jobs) or 1)) as pool:  # L19: one process per leg
        res = dict(zip(((j[0], j[3]) for j in jobs), pool.starmap(stats, jobs)))
    for (ds, trace, arm, b), dirs in cs + raw:
        ev = b'"agg_eval_raw"' if arm.endswith(":raw") else b'"agg_eval"'
        legs = [s for d in dirs if (s := res[(d, ev)])]
        if not legs:
            continue
        agg = {}
        for k in legs[0]:
            v = [x[k] for x in legs if x[k] is not None]
            agg[k] = (st.mean(v), max(v) - min(v), len(v)) if v else (None, None, 0)
        rows.append(dict(dataset=ds, trace=trace, arm=arm, baseline=b, n=len(legs), **agg))
    f = lambda m, s=1, pct=False: "–" if m[0] is None else (f"{100 * m[0]:.1f}±{100 * m[1] / 2:.1f}" if pct else f"{m[0]:.{s}f}")
    for (ds, trace) in sorted({(r["dataset"], r["trace"]) for r in rows}):
        ths = [k[3:] for k in rows[0] if k.startswith("tta")]
        print(f"\n## {ds} {trace}\n| arm | baseline | seeds | AUC % | final % | " + " | ".join(f"TTA {t} min (hit) | GB→{t}" for t in ths)
              + " | GB |\n|" + "---|" * (6 + 2 * len(ths)))
        for r in sorted((r for r in rows if (r["dataset"], r["trace"]) == (ds, trace)), key=lambda r: -(r["auc"][0] or 0)):
            print(f"| {r['arm']} | {r['baseline']} | {r['n']} | {f(r['auc'], pct=True)} | {f(r['final'], pct=True)} | "
                  + " | ".join(f"{f(r['tta' + t], 0)} ({r['tta' + t][2]}/{r['n']}) | {f(r['gb' + t])}" for t in ths)
                  + f" | {f(r['gb'])} |")
    if a.json_out:
        json.dump(rows, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
