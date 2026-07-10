#!/usr/bin/env python3
"""Explain #7's eligible-pool gap (real ~5.3 vs sim ~10.0 mean eligible,
fwdllm_plus @ syn_0) from banked telemetry -- no re-run needed (principle #11a).

Root cause (confirmed by this script on the banked n=10 pair): NOT a sim bug.
Two already-known, intentionally real-only mechanisms compound with
`reselect_each_iteration=True`:

  1. K-D11 "commit 1 per pass": once a fresh agg_goal-sized cohort is
     (re)selected, `sync_collect_and_accumulate_grads` clamps the REAL recv
     loop to drain exactly 1 grad before handing control back to
     `_select_ends_respecting_reselect_gate` (anti-deadlock, real-transport
     queue-persistence discipline). Sim's `_sync_sim_recv_first_k` always
     drains the full cohort in one call -- no such clamp applies.
     -> every real reselection immediately after a full(=agg_goal) cohort
        selection sees exactly 1 freshly-freed trainer eligible; sim always
        sees the full cohort (10) again.
  2. `_fetch_weights`/`recv_wrapper` (real MQTT wait, #11) and
     `_emulate_training_delay` (K-D29 intentional remainder-wait) dominate a
     real trainer's wall-clock cycle; actual JVP compute
     (`_train_one_batch`/`_perform_training`) is mode-invariant. Sim skips
     both waits by design (the vclock absorbs them), so its cohort turns
     over far faster in wall-clock terms -- explaining fwdllm_plus's
     ~4x real/sim wall-time ratio alongside the eligible-count gap.

Usage: python profile_eligibility_gap.py --real <run_dir> --sim <run_dir>
"""
from __future__ import annotations
import argparse, glob, json, statistics
from collections import defaultdict
from pathlib import Path

_EXP = Path(__file__).resolve().parent.parent / "experiments"

# Funcs that are real-transport-only or intentionally-real-only waits (K-D37's
# _STEP_TIMING_REAL_ONLY_FUNCS class) vs genuine mode-invariant compute.
_WAIT_FUNCS = ("_fetch_weights", "recv_wrapper", "_emulate_training_delay", "pause_execution")
_COMPUTE_FUNCS = ("_train_one_batch", "_perform_training", "train_model")


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    p2 = _EXP / run
    if p2.is_dir():
        return p2
    raise SystemExit(f"run dir not found: {run}")


def _load_jsonl(path: Path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _selection_train_events(run_dir: Path):
    f = next(iter(glob.glob(str(run_dir / "telemetry" / "aggregator_*.jsonl"))), None)
    if f is None:
        raise SystemExit(f"no aggregator_*.jsonl under {run_dir / 'telemetry'}")
    evs = [e for e in _load_jsonl(Path(f))
           if e.get("event") == "selection" and e.get("task") == "train"]
    return sorted(evs, key=lambda e: e["ts"])


def _step_timing_by_func(run_dir: Path):
    by_func = defaultdict(list)
    for f in glob.glob(str(run_dir / "telemetry" / "trainer_*.jsonl")):
        for e in _load_jsonl(Path(f)):
            if e.get("event") == "step_timing" and e.get("duration_s") is not None:
                by_func[e["func"]].append(e["duration_s"])
    return by_func


def _eligibility_summary(sel, label):
    ne = [e.get("num_eligible") for e in sel if e.get("num_eligible") is not None]
    ts = [e["ts"] for e in sel]
    dts = [b - a for a, b in zip(ts, ts[1:])]
    agg_goal = max(ne) if ne else 0  # cohort concurrency (10 at syn_0/n10)
    post_full = [b.get("num_eligible") for a, b in zip(sel, sel[1:])
                 if a.get("num_eligible") == agg_goal]
    print(f"{label}: n_selection_events={len(sel)} mean_eligible={statistics.mean(ne):.2f} "
          f"mean_inter_select_dt={statistics.mean(dts):.2f}s")
    print(f"{label}: eligible value on the FIRST reselect after a full({agg_goal})-cohort "
          f"selection: {post_full[:15]}{'...' if len(post_full) > 15 else ''}")


def _wait_vs_compute_summary(run_dir: Path, label):
    by_func = _step_timing_by_func(run_dir)
    print(f"{label} step_timing means (s/call):")
    for func in _WAIT_FUNCS + _COMPUTE_FUNCS:
        durs = by_func.get(func)
        tag = "WAIT " if func in _WAIT_FUNCS else "COMPUTE"
        if durs:
            print(f"  [{tag}] {func:<24} n={len(durs):<5} mean={statistics.mean(durs):.3f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="real run dir path or name under experiments/")
    ap.add_argument("--sim", required=True, help="sim run dir path or name under experiments/")
    args = ap.parse_args()

    real_dir = _resolve_run_dir(args.real)
    sim_dir = _resolve_run_dir(args.sim)

    r_sel = _selection_train_events(real_dir)
    s_sel = _selection_train_events(sim_dir)
    print(f"=== eligibility cadence (#7) ===")
    _eligibility_summary(r_sel, "real")
    _eligibility_summary(s_sel, "sim ")

    print(f"\n=== per-trainer step_timing: wait vs compute ===")
    _wait_vs_compute_summary(real_dir, "real")
    _wait_vs_compute_summary(sim_dir, "sim ")

    print(
        "\nConclusion: if compute means (real vs sim) are close and WAIT means are "
        "real-only/real >> sim, the eligible-count + throughput gap is fully explained "
        "by K-D11 (commit-1-per-pass) + K-D29 (remainder-wait) + #11 (MQTT fetch-weights "
        "wait) -- real-transport-only mechanisms, not a sim bug."
    )


if __name__ == "__main__":
    main()
