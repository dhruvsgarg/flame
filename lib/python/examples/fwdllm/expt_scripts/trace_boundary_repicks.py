#!/usr/bin/env python3
"""Was an end re-dispatched at a round boundary while STILL IN FLIGHT?

`selection_detail.sim_repicks_in_round` counts re-picks; this says whether they
are a defect. A re-pick after the end returned is normal churn -- a re-pick with
no `agg_round` commit from that end in between is over-dispatch, i.e. the
`all_selected` in-flight guard did not hold. Prints the vclock at each pick,
since a frozen vclock across repeats points at §D-30 (sim compresses the
inter-call gap real spreads over wall time).

    python trace_boundary_repicks.py <run_dir> [--round N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys


def _events(run_dir: str) -> list:
    paths = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not paths:
        sys.exit(f"no aggregator telemetry under {run_dir}")
    out = []
    for line in open(paths[0], errors="replace"):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--round", type=int, default=None,
                    help="boundary round to inspect (default: the busiest one)")
    args = ap.parse_args(argv)

    evs = _events(args.run_dir)
    sel = [e for e in evs if e.get("event") == "selection" and e.get("task") == "train"]
    if not sel:
        sys.exit("no train selection events")

    per_round = collections.Counter(e.get("round") for e in sel)
    rd = args.round if args.round is not None else per_round.most_common(1)[0][0]
    boundary = [e for e in sel if e.get("round") == rd]
    print(f"{os.path.basename(args.run_dir)}\n  selection events per round: {dict(per_round)}"
          f"\n  inspecting round {rd}: {len(boundary)} calls, "
          f"{sum(e.get('num_chosen') or 0 for e in boundary)} picks")
    if len(per_round) < 2:
        print("  NOTE: event-driven reselection (one round bucket) -- re-picks here "
              "are not distinguishable from normal churn.")

    seen, dups = {}, []
    for i, e in enumerate(boundary):
        for c in e.get("chosen") or []:
            if c in seen:
                dups.append((c, seen[c], i))
            else:
                seen[c] = i
    print(f"  unique ends {len(seen)}, re-picks {len(dups)}")

    for c, first, second in dups:
        a, b = boundary[first], boundary[second]
        between = [x for x in evs if a["ts"] < x["ts"] <= b["ts"]]
        commits = [x for x in between if x.get("event") == "agg_round"
                   and c in (x.get("contributing_trainers") or [])]
        verdict = "CHURN (returned in between)" if commits else "OVER-DISPATCH (never returned)"
        print(f"\n  end ...{c[-4:]}  call #{first} -> #{second}   vclock "
              f"{a.get('vclock_now')} -> {b.get('vclock_now')}"
              f"\n     in_flight {a.get('in_flight')} -> {b.get('in_flight')}, "
              f"eff_c {a.get('effective_c')}, commits from this end in between: {len(commits)}"
              f"\n     => {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
