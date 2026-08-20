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


# Top-level key holding provenance rather than charges. Readers skip it; it is
# underscored so it can never collide with a telemetry label.
_META = "_meta"


def _source_datasets(run_dirs) -> set:
    """What each source run was actually trained on, off its own config."""
    out = set()
    for d in run_dirs:
        try:
            with open(os.path.join(d, "aggregator_config.json")) as fh:
                out.add(json.load(fh)["hyperparameters"].get("dataset") or "unknown")
        except (OSError, ValueError, KeyError):
            out.add("unknown")
    return out


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


# A first-difference marginal above this multiple of the DIRECTLY measured
# per-dispatch cost is inter-arrival waiting, not work. 3x leaves room for the
# queueing/serialisation the step_timing span excludes.
_REDISPATCH_MARGINAL_MAX_X = 3.0
# A single sample may carry at most this share of a category's pooled total before
# its mean is a stall rather than a cost. Floored at 4/n so tiny pools aren't flagged
# for arithmetic alone (at n=8 any sample can be 1/8 of the total).
_OUTLIER_MAX_SHARE = 0.25
_OUTLIER_MIN_N = 8
# Share of the pool the concentration test looks at, so a repeating stall cannot be
# diluted below the threshold just by running longer.
_OUTLIER_TOP_FRAC = 0.01


def _load_real_dispatch_cost(run_dir: str) -> float | None:
    """Mean real `_distribute_weights_*` duration -- the per-dispatch cost measured
    DIRECTLY, as an independent cross-check on the first-difference marginal (§D-22)."""
    vals = []
    for path in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")):
        with open(path) as f:
            for line in f:
                if '"step_timing"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (e.get("event") == "step_timing"
                        and str(e.get("func", "")).startswith("_distribute_weights")
                        and e.get("duration_s") is not None):
                    vals.append(float(e["duration_s"]))
        break
    return (sum(vals) / len(vals)) if vals else None


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

    # The first-difference marginal assumes a batch is ONE serial dispatch burst.
    # It is, on the round baselines (cohort pinned, dispatched together after the
    # boundary clear) -- but NOT where dispatch is event-driven as slots free: there
    # a batch's timestamps span most of a cycle and the "marginal" is the gap between
    # trainers RETURNING. Measured: fluxtune's marginal 0.318s against a
    # cycle/dispatches of 0.400s and a directly-measured dispatch cost of 0.032s.
    # Cross-check against that direct cost and refuse to overwrite rather than write
    # a number that is 10-40x the work it claims to price (simulate_fwdllm.md §E).
    direct = [c for c in (_load_real_dispatch_cost(r) for r in args.real_runs) if c]
    direct_mean = (sum(direct) / len(direct)) if direct else None
    contaminated = set()
    if direct_mean:
        for key, spans in pooled.items():
            if key[0] == "redispatch_turnaround" and spans:
                m = st.mean(spans)
                if m > _REDISPATCH_MARGINAL_MAX_X * direct_mean:
                    contaminated.add(key)
                    print(f"WARN {key[0]}.{key[1]}: marginal {m:.4f}s is "
                          f"{m / direct_mean:.1f}x the measured per-dispatch cost "
                          f"{direct_mean:.4f}s -- event-driven dispatch, so this is "
                          f"inter-arrival waiting. Keeping the prior value.")

    # `mean_s` is what gets charged, so one warmup stall in a short run prices the
    # whole category. Score the mass the TOP 1% of samples carry: ~1% steady-state,
    # near 1 when stalls dominate. 2026-08-17 yelp-p `fedavg` = 2.6127s off ONE
    # 114.44s sample (91% of the total), median 0.0476s, a ~50x over-charge. NOT a
    # mean>p90 test -- that fires on the healthy agnews entry too (0.0563 > 0.0525).
    #
    # Top-k, not top-1, because a longer run LAUNDERS a repeating stall: yahoo's
    # every-25-commit cos probe put three ~116s samples in 489, each 21% of the
    # total, so a top-1 test at 25% passed a 2.7x overcharge that it had caught at
    # n=47 (2026-08-20). k scales with n, so what fires is concentration, not size.
    for key, spans in pooled.items():
        if key in contaminated or len(spans) < _OUTLIER_MIN_N:
            continue
        _total = sum(spans)
        _k = max(1, int(len(spans) * _OUTLIER_TOP_FRAC))
        _top = sorted(spans, reverse=True)[:_k]
        _share = (sum(_top) / _total) if _total > 0 else 0.0
        if _share > max(_OUTLIER_MAX_SHARE, 4.0 * _k / len(spans)):
            contaminated.add(key)
            print(f"WARN {key[0]}.{key[1]}: the top {_k} sample(s) "
                  f"(max {max(spans):.4f}s) carry {_share:.0%} of the pooled total "
                  f"(n={len(spans)}, mean {st.mean(spans):.4f}s, "
                  f"median {st.median(spans):.4f}s) -- outlier-dominated, not a "
                  f"steady-state cost. Keeping the prior value; re-profile off a "
                  f"real run that does not do that work.")

    today = date.today().isoformat()
    enable = set(args.enable)
    for (label, payload_kind), spans in pooled.items():
        if (label, payload_kind) in contaminated:
            continue
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
        keep = set(pooled) | contaminated      # a kept-prior entry is still observed
        registry = {
            label: {pk: e for pk, e in entries.items() if (label, pk) in keep}
            for label, entries in registry.items() if label != _META
        }
        registry = {label: e for label, e in registry.items() if e}

    # Which dataset these charges price. Per-pass cost scales with max_seq_length,
    # so charging a yahoo sim leg against agnews numbers mis-prices its vclock --
    # run_sequential.sh's preflight reads this tag to refuse exactly that, and an
    # untagged (pre-2026-08-17) profile is treated as agnews, which every one is.
    registry[_META] = {"datasets": sorted(_source_datasets(args.real_runs)),
                       "profiled_at": today}

    with open(args.out, "w") as f:
        f.write(
            "# Real-profiled vclock charges for sim-only costs sim can't measure\n"
            "# live (FWDLLM_DESIGN.md §P). Regenerate via expt_scripts/\n"
            "# profile_sim_charges.py -- hand-edit only `charge:`/`rationale:`.\n"
        )
        yaml.safe_dump(registry, f, sort_keys=True, default_flow_style=False)

    # Is this run long enough to have priced the leg, or did it just sample a phase?
    # A steady-state cost is flat across the run; one that drifts means the arm was
    # still warming up, or something periodic is inside the span. Read this before
    # trusting a profile taken off a short arm.
    for (label, pk), spans in sorted(pooled.items()):
        q = [spans[i * len(spans) // 5:(i + 1) * len(spans) // 5] for i in range(5)]
        q = [st.mean(b) for b in q if b]
        spread = (max(q) / min(q)) if q and min(q) > 0 else float("inf")
        flag = "" if spread <= 1.2 else (
            "  <-- %.1fx across the run: still warming up, or something periodic "
            "is inside the span" % spread)
        print(f"{label}.{pk}: n={len(spans)} mean_s={st.mean(spans):.4f} "
              f"p90_s={_pctl(spans, 0.9):.4f} median_s={st.median(spans):.4f}")
        print("    quintile means: " + " ".join(f"{v:.4f}" for v in q) + flag)


if __name__ == "__main__":
    main()
