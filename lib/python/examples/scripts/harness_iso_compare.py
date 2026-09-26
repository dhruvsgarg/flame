#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N22 P6 isolation control: the same legs run solo vs packed in a pool must look alike.

  harness_iso_compare.py <solo pool dir> <packed pool dir>

Per leg (baseline x real/sim): event verdict, commits, rounds reached, mean staleness, commit rate
(wall for real, vclock for sim), real queue_wait p99, and the pair's parity score. EQUIVALENT when event
verdicts match, rounds / staleness / rate move < TOL, parity scores move < 0.05 and real queue_wait p99
stays under max(2x solo, 0.25s). A single pair can't separate packing from run-to-run noise: a
NOT-EQUIVALENT line is a prompt to replicate, not a verdict on its own (L12).
"""

import csv
import glob
import json
import statistics
import subprocess
import sys
from pathlib import Path

EX_SCRIPTS = Path(__file__).resolve().parents[1] / "async_cifar10" / "scripts"  # analyze_send_recv_lag.py
TOL = 0.10


def leg_stats(run_dir: str, mode: str) -> dict:
    f = (glob.glob(f"{run_dir}/telemetry/aggregator_*.jsonl") or [None])[0]
    if not f:
        return {}
    rows = []
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "agg_round":
            rows.append(r)
    if not rows:
        return {"commits": 0}
    clock = [r.get("sim_completion_ts_recv") if mode == "sim" else r.get("ts") for r in rows]
    clock = [c for c in clock if isinstance(c, (int, float))]
    span = (max(clock) - min(clock)) if len(clock) > 1 else 0.0
    st = [r["staleness"] for r in rows if isinstance(r.get("staleness"), (int, float))]
    out = {"commits": len(rows), "rounds": max(r.get("round", 0) for r in rows),
           "staleness": statistics.mean(st) if st else 0.0, "rate": len(rows) / span if span else 0.0}
    if mode == "real":
        q = subprocess.run([sys.executable, str(EX_SCRIPTS / "analyze_send_recv_lag.py"), run_dir, "--queue-wait"],
                           capture_output=True, text=True).stdout.strip().splitlines()
        p99 = [t.split("=")[1] for t in (q[-1].split() if q else []) if t.startswith("p99=")]
        out["qw_p99"] = float(p99[0]) if p99 else None
    return out


def pool_rows(root: str) -> dict:
    """{(baseline, mode): (run_dir, ev verdict, parity score)} from a pool's phase summaries."""
    out = {}
    for tsv in glob.glob(f"{root}/ISO/summary.tsv"):
        with open(tsv, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                for mode in ("real", "sim"):
                    if r.get(f"{mode}_dir"):
                        out[(r["baseline"], mode)] = (r[f"{mode}_dir"], r[f"ev_{mode}"], r.get("score", ""))
    return out


def rel(a, b) -> float:
    return abs(a - b) / max(abs(a), 1e-9)


def main(solo: str, packed: str) -> int:
    s_rows, p_rows = pool_rows(solo), pool_rows(packed)
    bad = 0
    print(f"{'leg':<16} {'metric':<10} {'solo':>10} {'packed':>10}  verdict")
    for key in sorted(set(s_rows) & set(p_rows)):
        (sd, sev, ssc), (pd, pev, psc) = s_rows[key], p_rows[key]
        a, b = leg_stats(sd, key[1]), leg_stats(pd, key[1])
        checks = [("events", sev, pev, sev == pev)]
        for m in ("rounds", "staleness", "rate"):
            if m in a and m in b:
                checks.append((m, round(a[m], 3), round(b[m], 3), rel(a[m], b[m]) < TOL))
        if ssc and psc:
            checks.append(("parity", ssc, psc, abs(float(ssc) - float(psc)) < 0.05))
        if a.get("qw_p99") is not None and b.get("qw_p99") is not None:
            checks.append(("qw_p99", a["qw_p99"], b["qw_p99"], b["qw_p99"] <= max(2 * a["qw_p99"], 0.25)))
        for m, x, y, ok in checks:
            bad += not ok
            print(f"{key[0] + '/' + key[1]:<16} {m:<10} {str(x):>10} {str(y):>10}  {'ok' if ok else 'DIFF'}")
    missing = sorted(set(s_rows) ^ set(p_rows))
    if missing:
        print(f"legs in only one pool: {missing}")
    print("EQUIVALENT" if not bad and not missing else f"NOT EQUIVALENT ({bad} diff(s)) — replicate before concluding")
    return 0 if not bad else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
