#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Resource ledger report (FX-D140): reads `<pool>/resources.jsonl` written by `harness_pool`.

  resource_report.py POOL [POOL ...]              # per leg: RAM peak vs estimate, GB/trainer, GPU peak, leak suspects
  resource_report.py POOL [POOL ...] --timeline   # node RAM merged across pools, with leg start/end events

Pools sharing a node merge on one time axis, so a tight moment names every pool's running legs.
"""
from __future__ import annotations

import argparse
import json
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--timeline", action="store_true")
    a = ap.parse_args(argv)
    rows = load(a.pools)
    print(timeline(rows) if a.timeline else legs_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
