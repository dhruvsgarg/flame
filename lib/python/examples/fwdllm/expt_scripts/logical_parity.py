#!/usr/bin/env python3
"""Time-STRIPPED logical parity diff, real vs sim, up to a data-bin cap.

Does the sim take the SAME steps in the SAME order as real, IGNORING every
wall/vclock timestamp? Logical parity is checked before the time dimension
(sim_rate). Reads already-banked aggregator telemetry — no re-run.

The authoritative record is the `agg_round` event stream (data_id,
iteration_per_data_id, receive-ordered contributing_trainers, variance decision).
Restricts to data_id <= --max-bin (default 1; pass e.g. --max-bin 15 to sweep
further and see WHERE a divergence onsets/grows/plateaus) and compares, per
aggregation cycle:
  - receive SET   : which trainers committed together (async cohort composition)
  - receive ORDER : arrival order (only meaningful for async; a sync barrier
                    commits the whole cohort, so order within a set is irrelevant)
  - cadence tuple : (data_id, iter, agg_goal_count, var_good_enough, force_commit)
plus iterations-to-clear-bin-0 and the per-cycle variance trajectory.

Also compares, per (data_id, iteration) SELECTION call, cohort SIZE (num_eligible/
num_chosen) — a finer axis than receive-SET/cadence: a reselect call that always
picks its whole eligible pool (fwdllm's `random` selector) can pass every
receive-SET/cadence check while still diverging on how MANY trainers were
eligible/chosen at that call (this is what the aggregate `selection_detail` rung
caught for fwdllm_plus's sim_rate regression — see simulate_fwdllm.md — well
after full-length telemetry archaeology; this axis surfaces it directly, at
bin-1 scope, from already-banked logs).

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


def _selection_trace(run_dir: Path, max_bin: int):
    """Per-(data_id, iteration) SELECTION-call cohort size, train task only.

    Only meaningful for baselines that thread `agg_version_key` through
    `channel.ends()` (fwdllm/fwdllm_plus — see flame/selector/random.py's
    `_extra["data_id"]`/`_extra["iteration_per_data_id"]`); returns {} when
    absent (e.g. fluxtune's async_oort path has no per-iteration reselect
    concept, so this axis is a no-op there, not a false SKIP).
    """
    f = next(iter(glob.glob(str(run_dir / "telemetry" / "aggregator_*.jsonl"))), None)
    out = {}
    for line in open(f):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "selection" or e.get("task") != "train":
            continue
        did, it = e.get("data_id"), e.get("iteration_per_data_id")
        if did is None or it is None or did > max_bin:
            continue
        # A (data_id, iteration) key can recur across multiple reselect calls
        # (fwdllm_plus's reselect_each_iteration=True re-invokes select() many
        # times before the iteration advances) — keep every call, ordered.
        out.setdefault((did, it), []).append({
            "num_candidates": e.get("num_candidates"),
            "num_eligible": e.get("num_eligible"),
            "num_chosen": e.get("num_chosen"),
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

    _cmp_cohort_size(rdir, sdir, max_bin)


def _cmp_cohort_size(rdir: Path, sdir: Path, max_bin: int):
    """SELECTION-call cohort-SIZE axis (see `_selection_trace`): distinct from
    receive-SET/cadence above — a call can always exhaust its eligible pool
    (SET/cadence-consistent) while the pool SIZE itself diverges real vs sim."""
    r_sel, s_sel = _selection_trace(rdir, max_bin), _selection_trace(sdir, max_bin)
    if not r_sel and not s_sel:
        return  # baseline has no per-iteration reselect telemetry (e.g. fluxtune)
    keys = sorted(set(r_sel) | set(s_sel))
    diffs = []
    first_mismatch = None
    for k in keys:
        rc = [c["num_chosen"] for c in r_sel.get(k, [])]
        sc = [c["num_chosen"] for c in s_sel.get(k, [])]
        re_ = [c["num_eligible"] for c in r_sel.get(k, [])]
        se_ = [c["num_eligible"] for c in s_sel.get(k, [])]
        rn, sn = len(rc), len(sc)
        r_mean = sum(rc) / rn if rn else 0.0
        s_mean = sum(sc) / sn if sn else 0.0
        if rn != sn or abs(r_mean - s_mean) > 1e-9:
            diffs.append((k, rn, sn, r_mean, s_mean))
            if first_mismatch is None:
                first_mismatch = k
    print(f"  --- cohort-SIZE axis (selection-call granularity, per data_id/iter) ---")
    print(f"  (data_id,iter) keys compared: {len(keys)}  "
          f"identical calls-and-mean-chosen: {len(keys) - len(diffs)}/{len(keys)}")
    if first_mismatch is not None:
        print(f"  FIRST cohort-size divergence at (data_id,iter)={first_mismatch} "
              f"-- run with a larger --max-bin to see if it's an isolated blip "
              f"or grows/plateaus")
        for k, rn, sn, rm, sm in diffs[:10]:
            print(f"    {k}: real calls={rn} mean_chosen={rm:.2f}  |  "
                  f"sim calls={sn} mean_chosen={sm:.2f}")
        if len(diffs) > 10:
            print(f"    ... and {len(diffs) - 10} more")
    else:
        print(f"  => COHORT-SIZE PARITY (every reselect call picked the same "
              f"count real vs sim)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bin", type=int, default=1)
    ap.add_argument("--baselines", nargs="+", default=["fwdllm", "fwdllm_plus", "fluxtune"])
    a = ap.parse_args()
    for b in a.baselines:
        _cmp(b, a.max_bin)
