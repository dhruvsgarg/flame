#!/usr/bin/env python3
"""Regenerates `sim_charge_registry` YAML entries from a real run's own
`vclock_charge` telemetry (FWDLLM_DESIGN.md §P). Read that section first --
this script is the ONLY sanctioned way to write the numeric fields; the
`charge:` flag is a human decision this script never flips on its own.

Why this exists: a hand-derived profiled constant (e.g. §O's
`training_delay_factor`) goes stale silently whenever hardware, harness
overhead, or the code path it measures changes, and needed 3 manual
re-derivations. This script makes "re-profile after a hardware/setup change"
a one-line rerun instead of a fresh notebook derivation each time -- also the
reference implementation for porting a NEW example onto sim: point it at
that example's own real run to seed its own registry.

Usage:
    python profile_sim_charges.py --real-run <run_dir> [--real-run <run_dir> ...] \\
        --out ../sim_charge_profiles/<baseline>.yaml --only-observed \\
        [--enable redispatch_turnaround.weights]

Scans each real run's `telemetry/aggregator_*.jsonl` for `vclock_charge`
events with `time_mode == "real"`, pools samples per (label, payload_kind)
across all given runs, and writes mean/p50/p90/n + provenance into `--out`.
Existing entries' `charge:`/`rationale:` are preserved on refresh; a brand
new (label, payload_kind) is written with `charge: false` (review before
enabling) unless named in `--enable`.

`redispatch_turnaround` is a special case (simulate_fwdllm.md §B, 07-29
overcharge finding): its `post_close_overhead_wall_s` is CUMULATIVE from a
shared round-close wall-ts across every trainer redispatched in that batch
(`build_redispatch_decomp`'s own docstring: "now - round-close wall ts"), not
a standalone per-trainer cost -- the aggregator dispatches trainers in one
serial loop, so the k-th trainer's reading already includes the 1..k-1
trainers' dispatch wall ahead of it. Pooling those raw readings into one flat
mean (the generic path above) measures roughly HALF of a typical batch's
total span and then charges that flat amount to EVERY trainer in the batch,
overcharging by the batch size (confirmed 7.3-8.3x on fedbuff_round/
felix_round's 5400s pair). Fixed by grouping `redispatch_decomp` events per
(data_id, iteration_per_data_id) batch, sorting by ts, and pooling the FIRST
DIFFERENCE between consecutive positions -- the true per-trainer marginal
cost the charging loop actually adds one trainer at a time.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict
from datetime import date

import yaml


def _load_real_spans(run_dir: str) -> dict:
    """{(label, payload_kind or '_default'): [span_s, ...]} from one real run.
    Excludes `redispatch_turnaround` -- see `_load_real_redispatch_marginal`."""
    out = defaultdict(list)
    files = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not files:
        raise SystemExit(f"no telemetry/aggregator_*.jsonl under {run_dir}")
    with open(files[0]) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "vclock_charge" or e.get("time_mode") != "real":
                continue
            if e.get("label") == "redispatch_turnaround":
                continue
            span = e.get("span_s")
            if span is None:
                continue
            out[(e["label"], e.get("payload_kind") or "_default")].append(span)
    return out


def _load_real_redispatch_marginal(run_dir: str) -> dict:
    """{("redispatch_turnaround", payload_kind or '_default'): [marginal_s, ...]}
    from one real run's `redispatch_decomp` events -- see module docstring.
    Groups by (data_id, iteration_per_data_id) (one redispatch batch), sorts
    by ts (loop dispatch order, mixed payload kinds), and takes the first
    difference of `post_close_overhead_wall_s` between consecutive positions
    (position 0's own cumulative reading is already its marginal cost, since
    nothing precedes it in the batch)."""
    batches = defaultdict(list)
    files = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not files:
        raise SystemExit(f"no telemetry/aggregator_*.jsonl under {run_dir}")
    with open(files[0]) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "redispatch_decomp" or e.get("time_mode") != "real":
                continue
            span = e.get("post_close_overhead_wall_s")
            if span is None:
                continue
            key = (e.get("data_id"), e.get("iteration_per_data_id"))
            batches[key].append(
                (e.get("ts", 0.0), e.get("payload_kind") or "_default", span))

    out = defaultdict(list)
    for evts in batches.values():
        evts.sort(key=lambda t: t[0])
        prev = 0.0
        for _, pk, span in evts:
            out[("redispatch_turnaround", pk)].append(max(span - prev, 0.0))
            prev = span
    return out


def _pctl(vals: list, q: float) -> float:
    s = sorted(vals)
    return s[min(int(len(s) * q), len(s) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-run", action="append", required=True, dest="real_runs",
                     help="real run dir (repeatable); pooled across all given")
    ap.add_argument("--out", required=True, help="registry YAML to update/create")
    ap.add_argument("--enable", action="append", default=[],
                     help="label.payload_kind to flip charge:true (repeatable)")
    ap.add_argument("--only-observed", action="store_true",
                     help="drop entries the source runs never emitted. Use for a "
                          "PER-BASELINE profile: a baseline that never performs an "
                          "op must not carry a number for it, or the entry keeps a "
                          "stale value under fresh provenance (sync fwdllm inherited "
                          "`redispatch_turnaround` this way). Omit when refreshing a "
                          "pooled multi-baseline registry.")
    args = ap.parse_args()

    pooled = defaultdict(list)
    for run_dir in args.real_runs:
        for key, spans in _load_real_spans(run_dir).items():
            pooled[key].extend(spans)
        for key, spans in _load_real_redispatch_marginal(run_dir).items():
            pooled[key].extend(spans)

    registry = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            registry = yaml.safe_load(f) or {}

    today = date.today().isoformat()
    enable = set(args.enable)
    for (label, payload_kind), spans in pooled.items():
        prior = registry.setdefault(label, {}).get(payload_kind, {})
        key = f"{label}.{payload_kind}" if payload_kind != "_default" else label
        registry[label][payload_kind] = {
            "charge": True if key in enable else bool(prior.get("charge", False)),
            "mean_s": round(st.mean(spans), 4),
            "p90_s": round(_pctl(spans, 0.9), 4),
            "n": len(spans),
            "source_runs": sorted({os.path.basename(r) for r in args.real_runs}),
            "profiled_at": today,
            "rationale": prior.get("rationale", "new candidate -- review before enabling"),
        }

    if args.only_observed:
        registry = {
            label: {pk: e for pk, e in entries.items() if (label, pk) in pooled}
            for label, entries in registry.items()
        }
        registry = {label: e for label, e in registry.items() if e}

    with open(args.out, "w") as f:
        f.write(
            "# Real-profiled vclock charges for sim-only costs sim can't measure\n"
            "# live (FWDLLM_DESIGN.md §P). Regenerate via expt_scripts/\n"
            "# profile_sim_charges.py -- hand-edit only `charge:`/`rationale:`.\n"
        )
        yaml.safe_dump(registry, f, sort_keys=True, default_flow_style=False)

    for (label, pk), spans in pooled.items():
        print(f"{label}.{pk}: n={len(spans)} mean_s={st.mean(spans):.4f} "
              f"p90_s={_pctl(spans, 0.9):.4f}")


if __name__ == "__main__":
    main()
