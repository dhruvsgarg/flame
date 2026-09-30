#!/usr/bin/env python3
"""FX-D23: profile the sim's non-compute charges from REAL legs into a `sim_charge_registry` YAML.

Per aggregator stack (asyncfl | oort_sync | fedavg, from baselines.yaml), from each real leg's aggregator log:
  completion_leg   = download + upload per update: [LAG_DECOMP] agg_to_trainer + post_wait + mqtt_lag + queue_wait
  dispatch_latency = commit -> next send: sync stacks = last update of version v -> [DISTRIBUTE_TIMING] round v+1;
                     asyncfl = [DISTRIBUTE_TIMING] - the latest update arrival before it
The launcher (debug_run.sh) applies both to every baseline of that stack. Only this script writes the numbers.

Usage: profile_felix_charges.py --dataset D --harness H --out sim_charge_profiles/<H>_<D>.yaml <pool | *_real dir> ...
"""
import argparse
import glob
import os
import re
import socket
import statistics as st
import subprocess
from collections import defaultdict
from datetime import date, datetime

import yaml

LAG_KEYS = ("agg_to_trainer_s", "post_wait_s", "mqtt_lag_s", "queue_wait_s")
BASELINES = os.path.join(os.path.dirname(__file__), "..", "_metadata", "baselines.yaml")


def _ts(line):
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()


def stack_of(baseline: str) -> str:
    """The aggregator code path (asyncfl | oort_sync | fedavg): each pays its own ingest/commit cost."""
    main = yaml.safe_load(open(BASELINES))["baselines"][baseline]["example"]["aggregator_main"]
    return os.path.basename(main)[len("main_"):-len("_agg.py")]


def leg_samples(log: str, stack: str):
    """(completion_leg samples, dispatch_latency samples) from one real aggregator log."""
    legs, lat = [], []
    last_by_version, dist_by_round, last_recv = {}, {}, None
    for line in open(log, errors="ignore"):
        if "[LAG_DECOMP]" in line:
            vals = [re.search(k + r"=(-?[0-9.]+)", line) for k in LAG_KEYS]
            if all(vals):
                legs.append(sum(float(v.group(1)) for v in vals))
            m = re.search(r"version=(\d+)", line)
            if m:
                last_by_version[int(m.group(1))] = _ts(line)
            last_recv = _ts(line)
        elif "[DISTRIBUTE_TIMING]" in line:
            m = re.search(r"round=(\d+)", line)
            if stack == "asyncfl" and last_recv is not None:
                lat.append(_ts(line) - last_recv)
                last_recv = None
            elif m:
                dist_by_round.setdefault(int(m.group(1)), _ts(line))
    if stack != "asyncfl":
        lat = [dist_by_round[v + 1] - t for v, t in last_by_version.items()
               if v + 1 in dist_by_round and dist_by_round[v + 1] > t]
    return legs, lat


def _entry(xs):
    return {"mean_s": round(st.mean(xs), 4), "p50_s": round(st.median(xs), 4), "n": len(xs), "charge": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True, choices=("cifar10", "google_speech"))
    ap.add_argument("--harness", required=True, help="stub | tiny_cpu | gpu (run-name tag _h<harness>_; gpu = none)")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    runs = set()
    for p in a.dirs:
        if p.rstrip("/").endswith("_real"):
            runs.add(p.rstrip("/"))
        for f in glob.glob(f"{p}/**/*_grade/summary.tsv", recursive=True):
            rows = [l.rstrip("\n").split("\t") for l in open(f)]
            col = rows[0].index("real_dir") if rows and "real_dir" in rows[0] else None
            runs |= {r[col] for r in rows[1:] if col is not None and len(r) > col and os.path.isdir(r[col])}
    tag = "" if a.harness == "gpu" else f"_h{a.harness}_"
    runs = sorted(r for r in runs if ("google_speech" in r) == (a.dataset == "google_speech")
                  and (tag in r if tag else "_hstub_" not in r and "_htiny_cpu_" not in r))
    pooled = {"completion_leg": defaultdict(list), "dispatch_latency": defaultdict(list)}
    used = []
    for r in runs:
        m = re.search(r"_(felix|fedbuff|oort_star|oort|refl|feddance|oracle)_n\d+", r)
        logs = glob.glob(f"{r}/*aggregator.log")
        if not m or not logs:
            continue
        stack = stack_of(m.group(1))
        legs, lat = leg_samples(logs[0], stack)
        pooled["completion_leg"][stack] += legs
        pooled["dispatch_latency"][stack] += lat
        used.append(os.path.basename(r))
    out = {label: {s: _entry(xs) for s, xs in by.items() if xs} for label, by in pooled.items()}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["_meta"] = {"dataset": a.dataset, "harness": a.harness, "host": socket.gethostname().split(".")[0], "commit": commit, "date": str(date.today()),
                    "tool": "profile_felix_charges.py", "runs": used}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    yaml.safe_dump(out, open(a.out, "w"), sort_keys=False)
    for label in ("completion_leg", "dispatch_latency"):
        print(label, {s: (e["mean_s"], e["n"]) for s, e in out.get(label, {}).items()})
    print(f"{len(used)} real legs -> {a.out}")


if __name__ == "__main__":
    main()
