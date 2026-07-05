#!/usr/bin/env python3
"""Time-STRIPPED logical parity diff, real vs sim, up to a data-bin cap.

Separates the two parity concerns (simulate_fwdllm.md principles #14/#15):
does the sim take the SAME steps in the SAME order as real, IGNORING every
wall/vclock timestamp? Only once logical parity holds do we chase the time
dimension (sim_rate). Reads the ALREADY-BANKED aggregator telemetry — no re-run.

The authoritative record is the `agg_round` event stream (it carries data_id,
iteration_per_data_id, the receive-ordered contributing_trainers, and the
variance decision). We restrict to data_id <= --max-bin (default 1: bin 0 has
many iterations/aggregations already) and compare, per aggregation cycle:
  - receive SET   : which trainers committed together (async cohort composition)
  - receive ORDER : the arrival order (only meaningful for async; a sync barrier
                    commits the whole cohort so order within a set is irrelevant)
  - cadence tuple : (data_id, iter, agg_goal_count, var_good_enough, force_commit)
plus iterations-to-clear-bin-0 and the per-cycle variance trajectory.

Usage:  python logical_parity.py [--max-bin 1] [--baselines fluxtune ...]
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

_EXP = Path(__file__).resolve().parent.parent / "experiments"


def _short(t): return t[-3:] if len(t) >= 3 else t


def _agg_trace(run_dir: Path, max_bin: int):
    f = next(iter(glob.glob(str(run_dir / "telemetry" / "aggregator_*.jsonl"))), None)
    out = []
    for line in open(f):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "agg_round":
            continue
        did = e.get("data_id")
        if did is None or did > max_bin:
            continue
        out.append({
            "data_id": did,
            "iter": e.get("iteration_per_data_id"),
            "goal": e.get("agg_goal_count"),
            "order": [_short(t) for t in (e.get("contributing_trainers") or [])],
            "var_good": e.get("var_good_enough"),
            "force": e.get("force_commit_planned"),
            "var": e.get("var"),
        })
    return out


def _find_pair(baseline: str):
    def pick(variant):
        cands = [p for p in glob.glob(str(_EXP / f"run_*_{baseline}_n*_smoke_*_{variant}"))
                 if f"_{baseline}_n" in os.path.basename(p)]  # fwdllm !=> fwdllm_plus
        return Path(sorted(cands)[-1]) if cands else None
    return pick("real"), pick("sim")


def _iters_to_clear_bin0(trace):
    return sum(1 for x in trace if x["data_id"] == 0)


def _cmp(baseline: str, max_bin: int):
    rdir, sdir = _find_pair(baseline)
    print(f"\n==================== {baseline} (data bin <= {max_bin}) ====================")
    print(f"  real: {rdir.name if rdir else None}")
    print(f"  sim : {sdir.name if sdir else None}")
    if not (rdir and sdir):
        print("  MISSING pair"); return
    r, s = _agg_trace(rdir, max_bin), _agg_trace(sdir, max_bin)

    n = min(len(r), len(s))
    set_match = sum(1 for i in range(n) if sorted(r[i]["order"]) == sorted(s[i]["order"]))
    ord_match = sum(1 for i in range(n) if r[i]["order"] == s[i]["order"])
    cad = lambda x: (x["data_id"], x["iter"], x["goal"], x["var_good"], x["force"])
    cad_match = sum(1 for i in range(n) if cad(r[i]) == cad(s[i]))

    print(f"  aggregations compared: {n}  (real={len(r)}, sim={len(s)})")
    print(f"  receive-SET   identical: {set_match}/{n}")
    print(f"  receive-ORDER identical: {ord_match}/{n}   (sync barrier: order-within-set is benign)")
    print(f"  cadence       identical: {cad_match}/{n}   (data_id,iter,goal,var_good,force)")
    print(f"  iters to clear data bin 0:  real={_iters_to_clear_bin0(r)}  sim={_iters_to_clear_bin0(s)}")

    verdict = ("LOGICAL PARITY" if cad_match == n and set_match == n else
               "CADENCE PARITY (cohorts differ)" if cad_match == n else
               "LOGICAL DIVERGENCE")
    print(f"  => {verdict}")

    if cad_match != n or set_match != n:
        print(f"  {'REAL':<44} | SIM")
        for i in range(max(len(r), len(s))):
            rr = (r[i]["data_id"], r[i]["iter"], tuple(sorted(r[i]["order"])),
                  r[i]["var_good"], round(r[i]["var"] or 0, 3)) if i < len(r) else ""
            ss = (s[i]["data_id"], s[i]["iter"], tuple(sorted(s[i]["order"])),
                  s[i]["var_good"], round(s[i]["var"] or 0, 3)) if i < len(s) else ""
            flag = "" if (rr and ss and rr[2] == ss[2]) else "  <- cohort differs"
            print(f"  {str(rr):<44} | {ss}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bin", type=int, default=1)
    ap.add_argument("--baselines", nargs="+", default=["fwdllm", "fwdllm_plus", "fluxtune"])
    a = ap.parse_args()
    for b in a.baselines:
        _cmp(b, a.max_bin)
