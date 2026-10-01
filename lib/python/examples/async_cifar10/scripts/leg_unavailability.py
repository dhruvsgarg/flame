#!/usr/bin/env python
"""FX-N35: the UN_AVL share a leg actually saw, from its own selection telemetry (avail_composition).

  leg_unavailability.py RUN_DIR [RUN_DIR ...]   # one line per dir: '<frac> <n_selections>'
"""
import glob
import json
import sys


def leg_unavailability(run_dir: str):
    """(mean UN_AVL fraction over train selections, n selections); (None, 0) without composition."""
    files = glob.glob(f"{run_dir}/telemetry/aggregator_*.jsonl")
    if not files:
        return None, 0
    fracs = []
    with open(files[0]) as f:
        for line in f:
            if '"selection"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            comp = e.get("avail_composition") if e.get("event") == "selection" and e.get("task") == "train" else None
            total = sum(comp.values()) if comp else 0
            if total:
                fracs.append(comp.get("UN_AVL", 0) / total)
    return (sum(fracs) / len(fracs) if fracs else None), len(fracs)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        frac, n = leg_unavailability(d)
        print(f"{'-' if frac is None else f'{frac:.3f}'} {n}")
