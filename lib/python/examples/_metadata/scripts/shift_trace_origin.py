#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Drop the all-available head of availability traces by moving trace time 0 to `shift` s (FX-D18).

The head (syn_10/syn_50: 600 s, mobiperf: 300 s) was meant to let trainers register, but trace time already
starts at the join barrier (real: `_mark_join_barrier_done`; sim: vclock at round 0), so it only removed
unavailability from the start of every run. Each series becomes [[0, state at shift], [t - shift, s] for t > shift].

  shift_trace_origin.py synthetic syn_10=600 syn_50=2400
  shift_trace_origin.py mobiperf 300
"""

import argparse
import bisect
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent.parent / "availability_traces"


def shift_events(ev: list, s: float) -> list:
    ts = [t for t, _ in ev]
    i = bisect.bisect_right(ts, s) - 1
    head = [[0, ev[i][1] if i >= 0 else "AVL_TRAIN"]]
    return head + [[t - s, st] for t, st in ev if t > s]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", choices=["synthetic", "mobiperf"])
    ap.add_argument("shifts", nargs="+", help="synthetic: name=seconds ...; mobiperf: seconds (all devices, all variants)")
    a = ap.parse_args()
    path = DIR / f"{a.file}_traces.yaml"
    d = yaml.safe_load(path.read_text())
    tr = d["traces"]
    if a.file == "synthetic":
        for spec in a.shifts:
            name, s = spec.split("=")
            s = int(s)
            e = tr[name]
            e["pattern"] = shift_events(e["pattern"], s)
            e["per_trainer"]["n300"] = {k: shift_events(v, s) for k, v in e["per_trainer"]["n300"].items()}
            e["description"] = f"{e['description']}; origin shifted +{s}s (all-available head dropped)"
    else:
        s = int(a.shifts[0])
        for dev in tr.values():
            for k in [k for k in dev if k.startswith("states_")]:
                dev[k] = shift_events(dev[k], s)
        d["description"] = f"{d['description']}; origin shifted +{s}s (the injected all-available head dropped)"
    path.write_text(yaml.safe_dump(d, default_flow_style=False, sort_keys=False, width=4096))
    print(f"shifted {path.name}: {a.shifts}")


if __name__ == "__main__":
    main()
