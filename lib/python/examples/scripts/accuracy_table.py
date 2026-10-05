#!/usr/bin/env python3
"""Accuracy vs time per leg, real beside sim (FELIX_READINESS accuracy table, FX-N74).

  accuracy_table.py POOL_DIR [POOL_DIR ...]            # every pair in the pools (summary.tsv real_dir / sim_dir)
  accuracy_table.py --runs RUN_DIR [RUN_DIR ...]       # single legs

Time axis = the run's own clock from the join barrier: sim vclock (interpolated from agg_round.vclock_now), real wall
since `trace_origin` (else the first dispatch/selection). Reports max accuracy, accuracy at fixed windows (last eval at or
before t) and time to each dataset target (cifar10 0.5, google_speech 0.6).
"""
import argparse
import bisect
import csv
import glob
import json
import os
import sys

TARGETS = {"cifar10": 0.5, "google_speech": 0.6}
WINDOWS_MIN = (15, 30, 45, 60, 90, 120)


def _agg_events(run_dir):
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not f:
        return []
    with open(f[0]) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def curve(run_dir):
    """[(t_s, round, acc)] on the leg's own clock."""
    ev = _agg_events(run_dir)
    if not ev:
        return []
    sim = run_dir.rstrip("/").endswith("_sim")
    evals = [e for e in ev if e.get("event") == "agg_eval" and e.get("test-accuracy") is not None]
    if sim:
        pts = sorted((e["round"], float(e["vclock_now"])) for e in ev
                     if e.get("event") == "agg_round" and e.get("vclock_now") is not None and e.get("round") is not None)
        rounds = [r for r, _ in pts]

        def t_of(e):
            if not pts:
                return None
            i = min(bisect.bisect_left(rounds, e["round"]), len(pts) - 1)
            return pts[i][1]
    else:
        orig = [e["origin_ts"] for e in ev if e.get("event") == "trace_origin" and e.get("origin_ts")]
        start = [e["ts"] for e in ev if e.get("event") in ("dispatch", "selection")]
        t0 = orig[0] if orig else (min(start) if start else ev[0]["ts"])

        def t_of(e):
            return e["ts"] - t0
    return [(t, e["round"], float(e["test-accuracy"])) for e in evals if (t := t_of(e)) is not None]


def summarize(c, target):
    if not c:
        return None
    out = {"n_evals": len(c), "max_acc": max(a for _, _, a in c), "last_t_min": c[-1][0] / 60}
    for w in WINDOWS_MIN:
        prior = [a for t, _, a in c if t <= w * 60]
        out[f"acc@{w}m"] = prior[-1] if prior and c[-1][0] >= w * 60 * 0.95 else None
    hit = next((t for t, _, a in c if a >= target), None)
    out["t_to_target_min"] = hit / 60 if hit is not None else None
    return out


def dataset_of(run_dir):
    return "google_speech" if "google_speech" in os.path.basename(run_dir.rstrip("/")) else "cifar10"


def fmt(x, pct=True):
    if x is None:
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.0f}"


def rows_from_pools(pools):
    for p in pools:
        for f in sorted(glob.glob(os.path.join(p, "*", "summary.tsv"))):
            for r in csv.DictReader(open(f), delimiter="\t"):
                yield os.path.basename(os.path.dirname(f)), r["baseline"], r["trace"], r.get("real_dir") or "", r.get("sim_dir") or ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    hdr = ["phase", "baseline", "trace", "mode", "evals", "max%"] + [f"@{w}m%" for w in WINDOWS_MIN] + ["t→target(min)"]
    out = []
    legs = [(os.path.basename(r), "", "", r, "") for r in a.runs] + list(rows_from_pools(a.pools))
    for phase, b, tr, real, sim in legs:
        for mode, d in (("real", real), ("sim", sim)):
            if not d or not os.path.isdir(d):
                continue
            s = summarize(curve(d), TARGETS[dataset_of(d)])
            if s is None:
                continue
            out.append({"phase": phase, "baseline": b, "trace": tr, "mode": mode, "run_dir": d, **s})
    print("\t".join(hdr))
    for r in out:
        print("\t".join([r["phase"], r["baseline"], r["trace"], r["mode"], str(r["n_evals"]), fmt(r["max_acc"])]
                        + [fmt(r[f"acc@{w}m"]) for w in WINDOWS_MIN] + [fmt(r["t_to_target_min"], pct=False)]))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
