#!/usr/bin/env python3
"""FX-N144: Felix's dynamic c per leg over time, from `dynamic_kc` telemetry: c, loss phase φ, trainable count, the pool bound
(pool x trainable) and τ, averaged per bin. Shows when and how fast c ramps down (φ falls as clients fit their data) and up
(the trainable pool grows).

  dynamic_c_report.py POOL... [--bin-h 1] [--pool 0.5]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_report import cells  # noqa: E402


def rows(path):
    out = subprocess.run(["grep", '"event": "dynamic_kc"', path], capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--bin-h", type=float, default=1.0)
    ap.add_argument("--pool", type=float, default=0.5, help="pool fraction for the bound column")
    a = ap.parse_args(argv)
    for (ds, trace, arm, bl), dirs in sorted(cells(a.pools).items()):
        for d in dirs:
            rs = [r for f in glob.glob(os.path.join(d, "telemetry", "aggregator_*.jsonl")) for r in rows(f) if r.get("c_config")]
            if not rs or all(r["c"] == r["c_config"] for r in rs):
                continue  # fixed c: nothing to report
            print(f"\n{ds} {trace} {arm} {bl}  (c_config {rs[0]['c_config']}, K {rs[0]['k']})")
            print("  h     c    φ     trainable  bound  τ")
            bins = {}
            for r in rs:
                bins.setdefault(int(r["now"] / 3600 / a.bin_h), []).append(r)
            for b in sorted(bins):
                g = bins[b]
                m = lambda k: sum(x.get(k) or 0 for x in g) / len(g)
                print(f"  {b * a.bin_h:4.1f}  {m('c'):4.0f}  {m('loss_phase'):.2f}  {m('n_eligible_train'):5.0f}      "
                      f"{a.pool * m('n_eligible_train'):5.0f}  {m('avg_staleness'):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
