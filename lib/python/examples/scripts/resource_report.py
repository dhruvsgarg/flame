#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Resource ledger report (FX-D140): reads `<pool>/resources.jsonl` written by `harness_pool`.

  resource_report.py POOL [POOL ...]              # per leg: RAM peak vs estimate, GB/trainer, GPU peak, leak suspects
  resource_report.py POOL [POOL ...] --timeline   # node RAM merged across pools, with leg start/end events
  resource_report.py POOL [POOL ...] --util       # FX-N100: utilization, throughput (legs/h) and interference flags

Pools sharing a node merge on one time axis, so a tight moment names every pool's running legs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_pool import leak_suspect  # noqa: E402

TIGHT_GB = 50  # node MemAvailable below this is a near-OOM moment


def load(pools):
    rows = []
    for p in pools:
        f = Path(p) / "resources.jsonl"
        if not f.exists():
            print(f"(no ledger in {p})", file=sys.stderr)
            continue
        for line in open(f):
            try:
                rows.append({**json.loads(line), "pool": Path(p).name})
            except json.JSONDecodeError:
                continue  # a torn last line of a live pool
    return sorted(rows, key=lambda r: r["t"])


def legs_table(rows) -> str:
    start = {r["jid"]: r for r in rows if r["ev"] == "start"}
    roles = {}  # jid -> role GB at the leg's peak sample
    for r in rows:
        if r["ev"] == "sample":
            for jid, leg in r["legs"].items():
                tot = sum(leg["ram_gb"].values())
                if tot >= sum(roles.get(jid, {}).values()):
                    roles[jid] = leg["ram_gb"]
    out = ["| leg | n | est GB | peak GB | agg / trainers GB | GB/trainer | GPU GB | 2nd-half growth | flag |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["ev"] != "end":
            continue
        s, ro = start.get(r["jid"], {}), roles.get(r["jid"], {})
        n = s.get("n") or 0
        per = ro.get("trainer", 0.0) / n if n else 0.0
        flags = [f for f, on in (("RAM_OVER", r["ram_peak_gb"] > r["mem_est_gb"]),
                                 ("LEAK?", leak_suspect(r["ram_peak_gb"], r["ram_growth_gb"]))) if on]
        out.append(f"| {r['jid']} | {n} | {r['mem_est_gb']:.0f} | {r['ram_peak_gb']:.1f} | "
                   f"{ro.get('agg', 0.0):.1f} / {ro.get('trainer', 0.0):.1f} | {per:.2f} | {r['gpu_peak_gb']:.1f} | "
                   f"{r['ram_growth_gb']:+.1f} | {' '.join(flags)} |")
    return "\n".join(out)


def timeline(rows) -> str:
    out, live = [], {}
    for r in rows:
        if r["ev"] == "start":
            live[r["jid"]] = r
            out.append(f"{r['t']:.0f} START {r['pool']}/{r['jid']} est {r['mem_est_gb']:.0f} GB")
        elif r["ev"] == "end":
            live.pop(r["jid"], None)
            out.append(f"{r['t']:.0f} END   {r['pool']}/{r['jid']} rc={r['rc']} peak {r['ram_peak_gb']:.0f} GB")
        elif r["ev"] == "sample":
            avail = r["node"]["mem_avail_gb"]
            mark = " TIGHT" if avail < TIGHT_GB else ""
            used = {j: round(sum(v["ram_gb"].values())) for j, v in r["legs"].items()}
            out.append(f"{r['t']:.0f} node avail {avail:.0f} GB reserved {r['node']['reserved_gb']:.0f}{mark} "
                       f"{r['pool']} legs {used}")
    return "\n".join(out)


SLOW_X = 1.25  # a leg running this x its solo estimate is interference-suspect (R24)


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def util(rows, gpus=None, gpu_gb: float = 45.0, mem_total_gb: float = 504.0) -> str:
    """FX-N100: time-weighted node CPU/RAM/GPU use over the batch window, legs and budget-hours finished per wall hour."""
    starts = {r["jid"]: r for r in rows if r["ev"] == "start"}
    ends = [r for r in rows if r["ev"] == "end"]
    if not starts or not ends:
        return "(no finished legs)"
    t0, t1 = min(r["t"] for r in starts.values()), max(r["t"] for r in ends)
    wall_h = max(1e-6, (t1 - t0) / 3600)
    buckets = {}
    for r in rows:
        if r["ev"] == "sample" and t0 <= r["t"] <= t1:
            buckets.setdefault(int(r["t"] // 30), []).append(r["node"])
    cpu, ram, sm, mem, idle = [], [], [], [], []
    ncpu = os.cpu_count() or 1
    for nodes in buckets.values():
        n = nodes[-1]
        ncpu = n.get("ncpu", ncpu) or ncpu
        cpu.append(n.get("cpu_busy", n.get("load1", 0.0)))
        ram.append(n.get("ram_used_gb", mem_total_gb - n.get("mem_avail_gb", mem_total_gb)))
        ids = [str(g) for g in (gpus or n.get("gpu_used_gb", {}).keys())]
        g_sm = [float(n.get("gpu_sm", {}).get(i, n.get("gpu_sm", {}).get(int(i), 0.0)) or 0.0) for i in ids] if n.get("gpu_sm") else []
        g_mem = [float(n.get("gpu_used_gb", {}).get(i, 0.0)) for i in ids]
        if g_sm:
            sm.append(sum(g_sm) / len(g_sm))
        if g_mem:
            mem.append(100 * sum(g_mem) / (gpu_gb * len(g_mem)))
            idle.append(sum(m < 2.0 for m in g_mem) / len(g_mem))
    legs = [r for r in ends if r.get("rc") == 0 and starts.get(r["jid"], {}).get("mode") != "grade"]
    budget_h = sum(starts[r["jid"]].get("runtime_s", 0) for r in legs) / 3600
    leg_h = sum(r.get("dur_s", 0) for r in legs) / 3600
    slow = [(r["jid"], r["dur_s"], starts[r["jid"]].get("est_s")) for r in legs
            if starts[r["jid"]].get("est_s") and r.get("dur_s", 0) > SLOW_X * starts[r["jid"]]["est_s"]]
    over = [(r["jid"], r["overran"], r["train_tasks"]) for r in legs if r.get("train_tasks") and r["overran"] > 0.02 * r["train_tasks"]]
    tasks = sum(r.get("train_tasks", 0) for r in legs)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    out = [f"window {wall_h:.2f} h, {len(buckets)} samples, {len(legs)} legs finished (rc=0, no grades)",
           f"throughput: {len(legs) / wall_h:.1f} legs/h · "
           + (f"{budget_h / wall_h:.2f} budget-h/h" if budget_h else "budget-h/h n/a (pre-FX-N100 ledger)")
           + f" · concurrency {leg_h / wall_h:.1f} legs",
           f"CPU busy: mean {mean(cpu):.1f} / p95 {_pct(cpu, 0.95):.1f} of {ncpu} ({100 * mean(cpu) / ncpu:.0f}%)",
           f"RAM used: mean {mean(ram):.0f} / peak {max(ram, default=0):.0f} of {mem_total_gb:.0f} GB",
           f"GPU SM: mean {mean(sm):.0f}% / p95 {_pct(sm, 0.95):.0f}%" if sm else "GPU SM: not recorded (pre-FX-N100 ledger)",
           f"GPU mem: mean {mean(mem):.0f}% · GPUs idle (< 2 GB) {100 * mean(idle):.0f}% of GPU-time",
           f"overruns (PL19): {sum(r.get('overran', 0) for r in legs)}/{tasks} train tasks; legs > 2%: {len(over)}"]
    out += [f"  {j}: {o}/{n}" for j, o, n in over[:12]]
    out += [f"slow legs (> {SLOW_X}x estimate): {len(slow)}"] + [f"  {j}: {d:.0f}s vs est {e:.0f}s" for j, d, e in slow[:12]]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--util", action="store_true")
    ap.add_argument("--gpus", default="", help="--util: GPU ids the batch may use (default: all in the ledger)")
    a = ap.parse_args(argv)
    rows = load(a.pools)
    if a.util:
        print(util(rows, [g for g in a.gpus.split(",") if g] or None))
    else:
        print(timeline(rows) if a.timeline else legs_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
