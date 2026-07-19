#!/usr/bin/env python3
"""P1 item 3/7 (simulate_fwdllm.md §B, 2026-07-19): exact per-data_id iteration-
count comparison, real vs sim, over the FULL run and for ANY baseline (fluxtune/
fwdllm/fwdllm_plus). Generalizes the manual telemetry-archaeology that first
surfaced fluxtune's data_id=0 mismatch (real 4 iters vs sim 5, checked by hand
over only the first 5 rounds) into a repeatable, whole-run, cross-baseline tool.

`v1_iter_per_data_id`/`v1b_iters_moving_avg` (checks.py) already gate on this
quantity, but only as a POOLED distribution/moving-average -- neither prints
"data_id=17: real=4 sim=6, MISMATCH" for a specific data_id. This script reads
the same `agg_round`/`cycle_data_id` telemetry those rungs already consume (via
the shared `load_agg_jsonl`/`_iters_per_data_id` helpers, no new telemetry
field) and reports the exact per-data_id diff plus first-mismatch-onset.

Usage: python analyze_iters_per_data_id.py --real-run <dir> --sim-run <dir>
Both args accept a run dir path or a bare name under experiments/.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

_LIB_PYTHON = Path(__file__).resolve().parents[3]  # lib/python
_PARITY_SCRIPTS = _LIB_PYTHON / "examples" / "async_cifar10" / "scripts"
for _p in (str(_LIB_PYTHON), str(_PARITY_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from parity.checks import load_agg_jsonl, _fwd_cadence_cycles, _iters_per_data_id  # noqa: E402

_EXP = Path(__file__).resolve().parent.parent / "experiments"


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    p2 = _EXP / run
    if p2.is_dir():
        return p2
    raise SystemExit(f"run dir not found: {run}")


def _load(run_dir: Path) -> dict:
    tel = glob.glob(os.path.join(str(run_dir), "telemetry", "aggregator_*.jsonl"))
    if not tel:
        raise SystemExit(f"no aggregator_*.jsonl under {run_dir}/telemetry")
    return load_agg_jsonl(tel[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-run", required=True)
    ap.add_argument("--sim-run", required=True)
    args = ap.parse_args()

    real_dir, sim_dir = _resolve_run_dir(args.real_run), _resolve_run_dir(args.sim_run)
    real, sim = _load(real_dir), _load(sim_dir)

    r_iters = _iters_per_data_id(_fwd_cadence_cycles(real))
    s_iters = _iters_per_data_id(_fwd_cadence_cycles(sim))
    if not r_iters or not s_iters:
        raise SystemExit("no cycle_data_id in cadence events on one/both sides "
                          "-- non-fwdllm run or pre-instrumentation telemetry")

    shared_ids = sorted(set(r_iters) & set(s_iters))
    # Only compare data_ids BOTH sides actually reached -- a data_id past
    # whichever side ran fewer total rounds is a run-length artifact, not an
    # iteration-count divergence, and would otherwise dominate the count.
    mismatches = [d for d in shared_ids if r_iters[d] != s_iters[d]]

    print(f"real: {real_dir.name}")
    print(f"sim:  {sim_dir.name}")
    print(f"data_ids: real={len(r_iters)} sim={len(s_iters)} shared={len(shared_ids)}")
    print(f"mismatches (on shared data_ids only): {len(mismatches)}/{len(shared_ids)}")

    if mismatches:
        first = mismatches[0]
        print(f"\nfirst mismatch: data_id={first} real={r_iters.get(first)} "
              f"sim={s_iters.get(first)}")
        print("\nall mismatches (data_id: real -> sim):")
        for d in mismatches:
            print(f"  {d}: {r_iters.get(d)} -> {s_iters.get(d)}")
    else:
        print("\nNo mismatches -- every shared data_id took the same number of "
              "iterations in real and sim.")


if __name__ == "__main__":
    main()
