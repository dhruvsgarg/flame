#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Time-stripped logical diff of a real/sim pair (PARITY_READINESS C4, Q6): the first selection and
the first commit where the two legs stop taking the same steps, ignoring every timestamp.

  logical_diff.py POOL_DIR [--baselines felix oort] [--context 3]   # every pair in a pool
  logical_diff.py --pair REAL_DIR SIM_DIR

A selection is (round, chosen set); a commit is (round, contributing trainers, staleness). The
selection event's fingerprints tell input drift (eligible/decision differ) from RNG desync (same).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys


def _events(run_dir):
    f = next(iter(glob.glob(f"{run_dir}/telemetry/aggregator_*.jsonl")), None)
    return [json.loads(line) for line in open(f)] if f else []


def trace(run_dir):
    ev = _events(run_dir)
    sel = [e for e in ev if e.get("event") == "selection" and e.get("task") == "train"]
    com = [e for e in ev if e.get("event") == "agg_round" and e.get("task_to_perform", "train") == "train"]
    return sel, com


def _skey(e):
    return e.get("round"), tuple(sorted(x[-4:] for x in e.get("chosen") or []))


def _ckey(e):
    return e.get("round"), tuple(sorted(x[-4:] for x in e.get("contributing_trainers") or [])), tuple(
        e.get("staleness") or [])


def diff(real_dir, sim_dir, context=3, out=sys.stdout):
    (rs, rc), (ss, sc) = trace(real_dir), trace(sim_dir)
    print(f"  selections real {len(rs)} sim {len(ss)} · commits real {len(rc)} sim {len(sc)}", file=out)
    first = {}
    for name, a, b, key in (("selection", rs, ss, _skey), ("commit", rc, sc, _ckey)):
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if key(x) != key(y)), None)
        if i is None and len(a) != len(b):
            i = min(len(a), len(b))
        first[name] = i
        print(f"  first {name} divergence: {'none' if i is None else '#' + str(i)}", file=out)
        if i is None:
            continue
        for j in range(max(0, i - context), min(i + context, len(a), len(b))):
            x, y = a[j], b[j]
            tag = ""
            if name == "selection":
                fp = lambda e: (e.get("eligible_fingerprint"), e.get("decision_fingerprint"))
                tag = ("  inputs differ" if fp(x) != fp(y) else "  same inputs -> RNG desync") if key(x) != key(y) else ""
            print(f"    {j:4d} real {key(x)} | sim {key(y)}{tag}", file=out)
    return first


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?")
    ap.add_argument("--pair", nargs=2, metavar=("REAL", "SIM"))
    ap.add_argument("--baselines", nargs="*")
    ap.add_argument("--context", type=int, default=3)
    a = ap.parse_args(argv)
    if a.pair:
        diff(*a.pair, context=a.context)
        return 0
    for f in sorted(glob.glob(f"{a.pool}/*/summary.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if a.baselines and r["baseline"] not in a.baselines:
                continue
            if not r.get("real_dir") or r.get("real_dir") == "-" or not r.get("sim_dir"):
                continue
            print(f"== {f.split('/')[-2]} {r['trace']} {r['baseline']}")
            diff(r["real_dir"], r["sim_dir"], context=a.context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
