#!/usr/bin/env python3
"""Per-leg status of the fwdllm setup smokes: commits, iterations / commit, s / commit, eval loss / acc trend, errors."""
import glob
import json
import os
import re
import sys

B = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(B, "..", "experiments")


def legs():
    for d in sorted(glob.glob(f"{B}/bench_*/*/runs/*/run_*")):
        yield "bench " + d.split("/")[-4], d
    for d in sorted(glob.glob(f"{EXP}/run_*_fwdsm_*")):
        yield "harness " + re.sub(r".*fwdsm_(.*?)_fwdllm.*", r"\1", d), d


def errors(d):
    n = 0
    for f in glob.glob(f"{d}/*.log") + glob.glob(f"{d}/../leg.out"):
        with open(f, errors="replace") as fh:
            n += sum(1 for l in fh if re.search(r"Traceback|out of memory|OutOfMemoryError", l))
    return n


print(f"{'leg':34s} {'commits':>7s} {'it/c':>5s} {'s/c':>6s} {'evals':>5s}  eval loss / acc (first -> last 3)  err")
for name, d in legs():
    tel = glob.glob(f"{d}/telemetry/aggregator_*.jsonl")
    if not tel:
        print(f"{name:34s} (no telemetry yet)")
        continue
    ups, its, ev = [], 0, []
    with open(tel[0]) as fh:
        for l in fh:
            try:
                e = json.loads(l)
            except ValueError:
                continue
            k = e.get("event")
            if k == "server_update":
                ups.append(e["ts"])
            elif k == "agg_round":
                its += 1
            elif k == "agg_eval":
                ev.append((e.get("test-loss"), e.get("test-accuracy")))
    sc = (ups[-1] - ups[0]) / (len(ups) - 1) if len(ups) > 1 else float("nan")
    fmt = lambda x: f"{x[0]:.3f}/{x[1]:.3f}"
    trend = (fmt(ev[0]) + " -> " + " ".join(fmt(x) for x in ev[-3:])) if ev else "-"
    print(f"{name:34s} {len(ups):7d} {its / max(1, len(ups)):5.1f} {sc:6.0f} {len(ev):5d}  {trend}  {errors(d)}")
