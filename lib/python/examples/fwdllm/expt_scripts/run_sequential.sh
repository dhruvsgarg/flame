#!/bin/bash
# Drive the fwdllm real<->sim launcher pairs (fwdllm, fwdllm_plus, fluxtune)
# for the parity sign-off runs. Thin driver over the shared harness
# examples/scripts/expt_runner.sh -- the conda activation, launch+progress loop,
# and log-health assertions live there; this file owns only what's
# fwdllm-specific: the baseline->(real yaml, sim yaml) map, the knob patching,
# and the tier/check spec fed to the pre-flight gate.
#
# What changed vs the old real-only sequential runner:
#   * --mode {sim|real|both} now drives the _sim sibling yamls too and pairs
#     each baseline's real+sim runs (default: both -- that's the parity run).
#   * run names carry a _real / _sim tag so scripts.parity.cli can glob the pair
#     (*baseline*real* / *baseline*sim*).
#   * --delays {on|off} sets enable_training_delays IDENTICALLY on both sides of
#     a pair (K-D8: smokes keep D=0; the convergence/parity run needs D>0 in
#     BOTH real and sim together -- mismatched D would be a false divergence).
#   * A pre-flight gate prints the hyperparameters in three volatility tiers and
#     refuses infeasible configs (see examples/scripts/expt_runner.py). --dry-run
#     shows the table + checks and exits without launching; --yes skips the
#     confirm for a real (GPU) run; --force overrides a blocking check.
#
# Usage (from anywhere):
#   run_sequential.sh [--mode sim|real|both] [--delays on|off]
#       [--max-runtime-s 600] [--max-data-id 10] [--num-trainers N] [--num-gpus N]
#       [--c C] [--c-async C] [--k K] [--agg-goal N] [--min-initial-trainers N]
#       [--partition-method NAME] [--avail-trace NAME | --avail-traces N1,N2]
#       [--only name1,name2] [--stop-on-fail] [--dry-run] [--yes] [--force]
#       [--show-all]
#
#   --mode           which time_mode variant(s) to run per baseline (default both).
#   --delays         enable_training_delays for BOTH sides of a pair (default off=D=0).
#   --max-runtime-s  wall/vclock cap per run (default 600 = 10 min).
#   --max-data-id    stop a run once data_id reaches this (default 10).
#   --num-trainers   override trainer.num_trainers (default: each YAML's own, 10).
#   --num-gpus       override execution.num_gpus (default: each YAML's own).
#   --c / --c-async / --k / --agg-goal / --min-initial-trainers
#                    selector/aggregator knobs (see the per-flag notes below).
#   --partition-method  override hyperparameters.partition_method both sides.
#   --var-threshold  set the variance-pass gate threshold (hyperparameters.var_threshold)
#                    on both sides. It VARIES with data heterogeneity, so it's a
#                    review-every-run knob (shown in tier ①), not a fixed default.
#   --max-iter-per-data-id  set the force-commit cap (max_iterations_per_data_id)
#                    on both sides (review-every-run, tier ①).
#   --avail-trace / --avail-traces  availability trace(s); Phase 1 uses syn_0.
#   --only           comma-separated baseline subset (default all three).
#   --after          comma-separated post-launch hooks to run once all launches
#                    finish: parity (scripts.parity.cli --batch on the real/sim
#                    pairs), sanity (extract_sanity_checks.py per run dir), plot
#                    (analyze_run.py over the produced telemetry). e.g. --after parity,sanity
#   --stop-on-fail   abort remaining runs on first non-zero exit.
#   --dry-run        show the pre-flight table + checks, generate cfgs, DON'T launch.
#   --yes            don't prompt to confirm a real (GPU) run.
#   --force          launch even if a pre-flight check is BLOCKING (error).
#   --show-all       expand tier ③ (config-baked) + list passing checks.
#
# Per-flag knob notes (unchanged semantics):
#   --c        sets selector.kwargs.c (+ minInitialTrainers + agg_goal unless
#              --agg-goal/--min-initial-trainers override those per-field).
#   --c-async  sets selector.kwargs.c only for the async baseline (fluxtune).
#   --agg-goal sets aggregator.agg_goal directly (fans into hyperparameters.aggGoal
#              + selector aggGoal/aggr_num per runner.py) independent of --c.
#   --partition-method  must be a group name in agnews_partition.h5
#              (e.g. niid_label_clients=100_alpha=0.1); default "uniform" (IID).
set -u

# repo paths (portable across nodes/checkouts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # .../examples/fwdllm
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"       # flame/
AC10_DIR="$REPO_ROOT/lib/python/examples/async_cifar10"  # hosts scripts.parity.cli

# shared harness: conda activation, launch+ticker, log asserts, preflight bridge
# shellcheck source=../../scripts/expt_runner.sh
source "$REPO_ROOT/lib/python/examples/scripts/expt_runner.sh"

expt_activate_conda            # no default env: require an active env / FLAME_CONDA_ENV
expt_pin_pythonpath "$REPO_ROOT"

# defaults
MODE="both"
DELAYS="off"
MAX_RUNTIME_S=600
MAX_DATA_ID=10
# Companion "was this passed on the command line?" flags. Needed because MODE/
# DELAYS/MAX_RUNTIME_S/MAX_DATA_ID have non-empty defaults, so their value alone
# can't tell "operator passed it (override -> green)" from "defaulted". (The
# empty-default knobs like SEL_C/AGG_GOAL/VAR_THRESHOLD don't need this: non-empty
# already means "passed".)
MODE_SET=0; DELAYS_SET=0; MAX_RUNTIME_S_SET=0; MAX_DATA_ID_SET=0
STOP_ON_FAIL=0
NUM_TRAINERS=""
NUM_GPUS=""
SEL_C=""
SEL_C_ASYNC=""
SEL_K=""
AGG_GOAL=""
MIN_INIT_TRAINERS=""
AVAIL_TRACE=""
AVAIL_TRACES=""
PARTITION_METHOD=""
VAR_THRESHOLD=""       # variance-pass gate threshold; varies with data heterogeneity -> review every run
MAX_ITER_PER_DATA_ID=""  # force-commit cap (max_iterations_per_data_id); review every run
ONLY=""
AFTER=""          # comma list of post-launch hooks: parity,sanity,plot (see after_* below)
DRY_RUN=0
ASSUME_YES=0
FORCE=0
SHOW_ALL=0

usage() {
  echo "usage: $0 [--mode sim|real|both] [--delays on|off] [--max-runtime-s S] [--max-data-id N]" >&2
  echo "          [--num-trainers N] [--num-gpus N] [--c C] [--c-async C] [--k K] [--agg-goal N]" >&2
  echo "          [--min-initial-trainers N] [--partition-method NAME]" >&2
  echo "          [--var-threshold F] [--max-iter-per-data-id N]" >&2
  echo "          [--avail-trace NAME | --avail-traces N1,N2] [--only n1,n2] [--stop-on-fail]" >&2
  echo "          [--dry-run] [--yes] [--force] [--show-all]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)                 MODE="$2"; MODE_SET=1; shift 2 ;;
    --delays)               DELAYS="$2"; DELAYS_SET=1; shift 2 ;;
    --max-runtime-s)        MAX_RUNTIME_S="$2"; MAX_RUNTIME_S_SET=1; shift 2 ;;
    --max-data-id)          MAX_DATA_ID="$2"; MAX_DATA_ID_SET=1; shift 2 ;;
    --num-trainers)         NUM_TRAINERS="$2"; shift 2 ;;
    --num-gpus)             NUM_GPUS="$2"; shift 2 ;;
    --c)                    SEL_C="$2"; shift 2 ;;
    --c-async)              SEL_C_ASYNC="$2"; shift 2 ;;
    --k)                    SEL_K="$2"; shift 2 ;;
    --agg-goal)             AGG_GOAL="$2"; shift 2 ;;
    --min-initial-trainers) MIN_INIT_TRAINERS="$2"; shift 2 ;;
    --avail-trace)          AVAIL_TRACE="$2"; shift 2 ;;
    --avail-traces)         AVAIL_TRACES="$2"; shift 2 ;;
    --partition-method)     PARTITION_METHOD="$2"; shift 2 ;;
    --var-threshold)        VAR_THRESHOLD="$2"; shift 2 ;;
    --max-iter-per-data-id) MAX_ITER_PER_DATA_ID="$2"; shift 2 ;;
    --only)                 ONLY="$2"; shift 2 ;;
    --after)                AFTER="$2"; shift 2 ;;
    --stop-on-fail)         STOP_ON_FAIL=1; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    --yes)                  ASSUME_YES=1; shift ;;
    --force)                FORCE=1; shift ;;
    --show-all)             SHOW_ALL=1; shift ;;
    *) echo "ERROR: unknown arg '$1'" >&2; usage ;;
  esac
done
case "$MODE" in sim|real|both) ;; *) echo "ERROR: --mode must be sim|real|both (got '$MODE')" >&2; exit 2 ;; esac
case "$DELAYS" in on|off) ;; *) echo "ERROR: --delays must be on|off (got '$DELAYS')" >&2; exit 2 ;; esac

# baseline -> (real yaml : sim yaml). Plain baseline names, independent of the
# "n10" baked into each source filename.
ALL_RUNS=(
  "fwdllm:$SCRIPT_DIR/fwdllm_n10_smoke.yaml:$SCRIPT_DIR/fwdllm_n10_smoke_sim.yaml"
  "fwdllm_plus:$SCRIPT_DIR/fwdllm_plus_n10_smoke.yaml:$SCRIPT_DIR/fwdllm_plus_n10_smoke_sim.yaml"
  "fluxtune:$SCRIPT_DIR/fluxtune_n10_smoke.yaml:$SCRIPT_DIR/fluxtune_n10_smoke_sim.yaml"
)

if [ -n "$ONLY" ]; then
  RUNS=()
  IFS=',' read -ra ONLY_NAMES <<< "$ONLY"
  for want in "${ONLY_NAMES[@]}"; do
    found=0
    for entry in "${ALL_RUNS[@]}"; do
      if [ "${entry%%:*}" = "$want" ]; then RUNS+=("$entry"); found=1; break; fi
    done
    if [ "$found" = "0" ]; then
      echo "ERROR: --only name '$want' not recognized. Valid: ${ALL_RUNS[*]%%:*}" >&2; exit 2
    fi
  done
else
  RUNS=("${ALL_RUNS[@]}")
fi

# trace list: --avail-traces wins; else single --avail-trace (may be empty).
if [ -n "$AVAIL_TRACES" ]; then TRACE_CSV="$AVAIL_TRACES"; else TRACE_CSV="$AVAIL_TRACE"; fi

LOGDIR="$SCRIPT_DIR/smoke_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOGDIR"
GPUS_VISIBLE="$( (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || echo 0)"
MANIFEST="$LOGDIR/manifest.tsv"
SPEC_JSON="$LOGDIR/spec.json"

# ---- PHASE A: generate all patched cfgs, build the tier/check spec, gate ----
# One python step so the operator sees the WHOLE matrix (all baselines x traces x
# variants) once, then confirms once. Writes per-run cfgs + a launch manifest,
# renders the tiered table via the shared expt_runner.render_and_gate, and exits
# 2 if any check is blocking.
RUN_TSV="$LOGDIR/_runs.tsv"; : > "$RUN_TSV"
for entry in "${RUNS[@]}"; do
  bl="${entry%%:*}"; rest="${entry#*:}"; real_y="${rest%%:*}"; sim_y="${rest#*:}"
  printf '%s\t%s\t%s\n' "$bl" "$real_y" "$sim_y" >> "$RUN_TSV"
done

EXPT_RUNNER_DIR="$EXPT_RUNNER_DIR" \
MODE="$MODE" DELAYS="$DELAYS" MAX_RUNTIME_S="$MAX_RUNTIME_S" MAX_DATA_ID="$MAX_DATA_ID" \
NUM_TRAINERS="$NUM_TRAINERS" NUM_GPUS="$NUM_GPUS" SEL_C="$SEL_C" SEL_C_ASYNC="$SEL_C_ASYNC" \
SEL_K="$SEL_K" AGG_GOAL="$AGG_GOAL" MIN_INIT_TRAINERS="$MIN_INIT_TRAINERS" \
PARTITION_METHOD="$PARTITION_METHOD" TRACE_CSV="$TRACE_CSV" GPUS_VISIBLE="$GPUS_VISIBLE" \
VAR_THRESHOLD="$VAR_THRESHOLD" MAX_ITER_PER_DATA_ID="$MAX_ITER_PER_DATA_ID" \
MODE_SET="$MODE_SET" DELAYS_SET="$DELAYS_SET" MAX_RUNTIME_S_SET="$MAX_RUNTIME_S_SET" MAX_DATA_ID_SET="$MAX_DATA_ID_SET" \
LOGDIR="$LOGDIR" MANIFEST="$MANIFEST" RUN_TSV="$RUN_TSV" DRY_RUN="$DRY_RUN" SHOW_ALL="$SHOW_ALL" \
EXAMPLE_DIR="$EXAMPLE_DIR" AC10_DIR="$AC10_DIR" \
python - <<'PY'
import os, sys, copy, yaml, json
sys.path.insert(0, os.environ["EXPT_RUNNER_DIR"])
import expt_runner

env = os.environ.get
MODE = env("MODE"); DELAYS = env("DELAYS")
MAX_RUNTIME_S = int(env("MAX_RUNTIME_S")); MAX_DATA_ID = int(env("MAX_DATA_ID"))
NUM_TRAINERS = env("NUM_TRAINERS") or ""
NUM_GPUS = env("NUM_GPUS") or ""
SEL_C = env("SEL_C") or ""; SEL_C_ASYNC = env("SEL_C_ASYNC") or ""; SEL_K = env("SEL_K") or ""
AGG_GOAL = env("AGG_GOAL") or ""; MIN_INIT = env("MIN_INIT_TRAINERS") or ""
PART = env("PARTITION_METHOD") or ""
VAR_THRESHOLD = env("VAR_THRESHOLD") or ""; MAX_ITER = env("MAX_ITER_PER_DATA_ID") or ""
# "was it passed on the command line?" (override -> green) for the defaulted flags
MODE_SET = env("MODE_SET") == "1"; DELAYS_SET = env("DELAYS_SET") == "1"
MAX_RUNTIME_S_SET = env("MAX_RUNTIME_S_SET") == "1"; MAX_DATA_ID_SET = env("MAX_DATA_ID_SET") == "1"
GPUS_VISIBLE = int(env("GPUS_VISIBLE") or "0")
LOGDIR = env("LOGDIR"); MANIFEST = env("MANIFEST")
DRY_RUN = env("DRY_RUN") == "1"; SHOW_ALL = env("SHOW_ALL") == "1"
delays_on = (DELAYS == "on")

traces = [t for t in (env("TRACE_CSV") or "").replace(",", " ").split()] or [""]
multi_trace = len(traces) > 1

variants = {"real": 0, "sim": 1} if MODE == "both" else {MODE: (0 if MODE == "real" else 1)}

runs = []  # (baseline, real_yaml, sim_yaml)
with open(env("RUN_TSV")) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line:
            runs.append(line.split("\t"))

manifest = []            # (name, cfg_path, variant, budget_s)
per_baseline = {}        # baseline -> resolved knobs (for the tier ② rows)
checks = []


def patch(exp, run_key, variant, trace):
    h = exp["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = MAX_RUNTIME_S
    h["max_data_id_progress"] = MAX_DATA_ID
    # enable_training_delays: SAME on both sides of a pair (K-D8).
    exp["trainer"]["enable_training_delays"] = delays_on
    if PART:
        h["partition_method"] = PART
        exp["trainer"]["config_overrides"]["hyperparameters"]["partition_method"] = PART
    # Variance-cadence knobs (review-every-run). Only patched when explicitly set,
    # so an unset run keeps the code/trainer default (surfaced as "(D)" below).
    if VAR_THRESHOLD:
        h["var_threshold"] = float(VAR_THRESHOLD)
    if MAX_ITER:
        h["max_iterations_per_data_id"] = int(MAX_ITER)
    if NUM_TRAINERS:
        exp["trainer"]["num_trainers"] = int(NUM_TRAINERS)
    if NUM_GPUS:
        exp["execution"]["num_gpus"] = int(NUM_GPUS)
    kwargs = exp["aggregator"]["config_overrides"]["selector"]["kwargs"]
    is_async = (run_key == "fluxtune")
    if SEL_C:
        kwargs["c"] = int(SEL_C)
        if not MIN_INIT:
            kwargs["minInitialTrainers"] = int(NUM_TRAINERS) if NUM_TRAINERS else int(SEL_C)
        if not AGG_GOAL:
            exp["aggregator"]["agg_goal"] = int(SEL_C)  # legacy: agg_goal matches c
    if SEL_C_ASYNC and is_async:
        kwargs["c"] = int(SEL_C_ASYNC)
    if SEL_K:
        kwargs["k"] = int(SEL_K)
    if AGG_GOAL:
        exp["aggregator"]["agg_goal"] = int(AGG_GOAL)
    if MIN_INIT:
        kwargs["minInitialTrainers"] = int(MIN_INIT)
    if trace:
        exp["trainer"].setdefault("availability", {})["mode"] = trace
        t_hp = exp["trainer"].setdefault("config_overrides", {}).setdefault("hyperparameters", {})
        t_hp.setdefault("client_notify", {})["trace"] = trace
        h.setdefault("trackTrainerAvail", {})["trace"] = trace
    # name / job id: carry a _real|_sim tag so scripts.parity.cli can glob the pair.
    n = int(NUM_TRAINERS) if NUM_TRAINERS else exp["trainer"].get("num_trainers", 10)
    parts = [run_key, f"n{n}", "smoke"]
    if trace:
        parts.append(trace)
    parts.append(variant)
    name = "_".join(parts)
    exp["name"] = name
    exp["aggregator"]["config_overrides"]["job"]["id"] = name
    return name


for trace in traces:
    for run_key, real_y, sim_y in runs:
        for variant, _idx in variants.items():
            src = real_y if variant == "real" else sim_y
            if not os.path.exists(src):
                checks.append({"name": f"source yaml exists ({run_key} {variant})",
                               "level": "error", "detail": f"missing: {src}"})
                continue
            cfg = yaml.safe_load(open(src, encoding="utf-8"))
            exps = cfg.get("experiments", [])
            for exp in exps:
                name = patch(exp, run_key, variant, trace)
            cfg["experiments"] = exps
            out = os.path.join(LOGDIR, f"{name}.yaml")
            yaml.safe_dump(cfg, open(out, "w", encoding="utf-8"), sort_keys=False)
            manifest.append((name, out, variant, MAX_RUNTIME_S))

            # record resolved knobs from the (first) patched experiment for display
            e0 = exps[0]
            h0 = e0["aggregator"]["config_overrides"]["hyperparameters"]
            kw0 = e0["aggregator"]["config_overrides"]["selector"]["kwargs"]
            per_baseline.setdefault(run_key, {
                "c": kw0.get("c"), "k": kw0.get("k"),
                "agg_goal": e0["aggregator"].get("agg_goal"),
                "min_init": kw0.get("minInitialTrainers"),
                "n_trainers": e0["trainer"].get("num_trainers"),
                "n_gpus": e0.get("execution", {}).get("num_gpus"),
                "partition": h0.get("partition_method"),
                "delays": e0["trainer"].get("enable_training_delays"),
                "async": (run_key == "fluxtune"),
            })

with open(MANIFEST, "w") as fh:
    for name, out, variant, budget in manifest:
        fh.write(f"{name}\t{out}\t{variant}\t{budget}\n")

# ---------------- build the tiered spec ----------------
# Row colour convention (legend in the subtitle):
#   🟢 set  = value came from a command-line flag -> OVERRIDES the yaml (even if
#             the yaml happens to agree). Shown WITHOUT "(D)".
#   🟡 warn = review/attention (a review-every-run knob still on its default).
#      (D) + dim = plain yaml/code default, not overridden this run.
def dflt(val, overridden):
    # inline "(D)" marker for the bundled per-baseline knob strings (tier ②)
    return f"{val}" if overridden else f"{val} (D)"

def scalar_row(label, val, overridden, note=None, review=False):
    if overridden:                       # operator passed the flag -> override
        d = {"label": label, "value": f"{val}", "level": "set"}
    elif review:                         # defaulted but must be eyeballed each run
        d = {"label": label, "value": f"{val} (D)", "level": "warn"}
    else:                                # plain default
        d = {"label": label, "value": f"{val} (D)", "level": "ok"}
    if note:
        d["note"] = note
    return d

tiers = []
# ① review every run
trace_overridden = any(traces)
trace_val = " ".join(traces) if trace_overridden else "syn_0"
# mode: single-sided always warns (parity needs both), regardless of override.
if MODE != "both":
    mode_row = {"label": "mode", "value": MODE, "level": "warn", "note": "single-sided: parity needs both"}
else:
    mode_row = scalar_row("mode", MODE, MODE_SET, note="real+sim pair (--mode)")
tier1 = {"name": "① REVIEW EVERY RUN", "rows": [
    mode_row,
    {"label": "baselines", "value": " ".join(rk for rk, *_ in runs)},
    # The two similarly-named-but-DIFFERENT knobs, disambiguated + on their own rows:
    scalar_row("max_runtime_s", MAX_RUNTIME_S, MAX_RUNTIME_S_SET, note="wall/vclock cap (--max-runtime-s)"),
    scalar_row("max_data_id_progress", MAX_DATA_ID, MAX_DATA_ID_SET,
               note="STOP condition: stop when data_id reaches this (--max-data-id)"),
    scalar_row("trace", trace_val, trace_overridden,
               note=("Phase 1 is syn_0 (100% avail)" if any(t and t != "syn_0" for t in traces) else "100% availability")),
    scalar_row("enable_training_delays", str(delays_on).lower(), DELAYS_SET,
               note=f"modeled training delay {'ON (D>0)' if delays_on else 'OFF (D=0)'}; matched on BOTH sides — K-D8"),
    # var_threshold / max_iterations_per_data_id vary with data heterogeneity ->
    # review-every-run (warn when defaulted). NOTE: max_iters_per_data_id is the
    # FORCE-COMMIT cap and is NOT the same as max_data_id_progress (the stop) above.
    scalar_row("var_threshold", VAR_THRESHOLD if VAR_THRESHOLD else "unset",
               bool(VAR_THRESHOLD), review=True,
               note="variance-pass gate; varies w/ data heterogeneity (--var-threshold). unset ⇒ trainer/code default"),
    scalar_row("max_iters_per_data_id", MAX_ITER if MAX_ITER else "unset",
               bool(MAX_ITER), review=True,
               note="FORCE-COMMIT cap (--max-iter-per-data-id) — NOT the max_data_id_progress stop above. unset ⇒ code default"),
]}
tiers.append(tier1)

# ② per-baseline -- an ALIGNED TABLE (baselines = rows, knobs = columns). The
# renderer highlights any column whose value differs across the baselines (those
# are the ones to eyeball); columns identical across all 3 stay dim (expected).
tier2_cols = [
    ("c", "c"), ("agg_goal", "agg_goal"), ("k", "k"),
    ("min_init", "minInit"), ("n_trainers", "n_trainers"),
    ("n_gpus", "n_gpus"), ("partition", "part"),
]
overridden2 = []
if bool(SEL_C) or bool(SEL_C_ASYNC): overridden2.append("c")
if bool(AGG_GOAL) or bool(SEL_C):    overridden2.append("agg_goal")
if bool(SEL_K):        overridden2.append("k")
if bool(MIN_INIT):     overridden2.append("min_init")
if bool(NUM_TRAINERS): overridden2.append("n_trainers")
if bool(NUM_GPUS):     overridden2.append("n_gpus")
if bool(PART):         overridden2.append("partition")
rows2 = []
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    rows2.append({"name": rk, "cells": {
        "c": b.get("c"), "agg_goal": b.get("agg_goal"), "k": b.get("k"),
        "min_init": b.get("min_init"), "n_trainers": b.get("n_trainers"),
        "n_gpus": b.get("n_gpus"), "partition": b.get("partition"),
    }})
tiers.append({"name": "② PER-BASELINE (moderate)",
              "table": {"columns": tier2_cols, "rows": rows2,
                        "overridden": overridden2}})

# ③ config-baked
tiers.append({"name": "③ RARELY CHANGED", "collapsed": True, "rows": [
    {"label": "env", "value": os.environ.get("CONDA_DEFAULT_ENV", "?")},
    {"label": "gpus_visible", "value": str(GPUS_VISIBLE)},
    {"label": "example_dir", "value": env("EXAMPLE_DIR")},
    {"label": "logdir", "value": LOGDIR},
]})

# ---------------- feasibility checks ----------------
# D matched across each pair (by construction, but assert it visibly).
if MODE == "both":
    checks.append({"name": "enable_training_delays matched across every real/sim pair",
                   "level": "ok", "detail": f"D={'>0' if delays_on else '0'} both sides"})
# agg_goal <= c (more required than concurrently selected -> stall).
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    c, g = b.get("c"), b.get("agg_goal")
    if isinstance(c, int) and isinstance(g, int) and g > c:
        checks.append({"name": f"agg_goal <= c ({rk})", "level": "error",
                       "detail": f"agg_goal={g} > c={c} — selected trainers would be stranded"})
    else:
        checks.append({"name": f"agg_goal <= c ({rk})", "level": "ok", "detail": f"agg_goal={g} c={c}"})
# (No k-vs-agg_goal check: in the random selector, send-side selection/concurrency
# is driven by `c` (required_trainers = min(len(ends), c - in_use)); `k` is the
# RECV-side batch size (num_ends_to_remove = min(..., self.k)), NOT a selection
# cap -- so k < agg_goal is fine, the barrier still collects agg_goal grads across
# RECV passes. c <= num_trainers and agg_goal <= c are the binding invariants.)
# num_gpus <= visible.
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    g = b.get("n_gpus")
    if isinstance(g, int) and GPUS_VISIBLE and g > GPUS_VISIBLE:
        checks.append({"name": f"num_gpus <= gpus_visible ({rk})", "level": "error",
                       "detail": f"num_gpus={g} > visible={GPUS_VISIBLE}"})
# num_trainers >= minInitialTrainers.
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    n, mi = b.get("n_trainers"), b.get("min_init")
    if isinstance(n, int) and isinstance(mi, int) and n < mi:
        checks.append({"name": f"num_trainers >= minInitialTrainers ({rk})", "level": "error",
                       "detail": f"num_trainers={n} < minInitialTrainers={mi} — join barrier never clears"})
# non-uniform partition: can't verify the H5 group from here.
if PART and PART != "uniform":
    checks.append({"name": "partition group exists in agnews_partition.h5", "level": "warn",
                   "detail": f"verify group '{PART}' exists"})
# parity pairing naming (only meaningful for a both-mode matrix).
if MODE == "both":
    ok_pair = all(any(n.endswith("_real") for n, *_ in manifest) and
                  any(n.endswith("_sim") for n, *_ in manifest) for _ in [0])
    checks.append({"name": "run names carry _real/_sim tags for parity glob",
                   "level": "ok" if ok_pair else "error",
                   "detail": "scripts.parity.cli globs *baseline*real* / *baseline*sim*"})

goal_for_parity = AGG_GOAL or (SEL_C or "10")
next_cmd = (f"(cd {env('AC10_DIR')} && python -m scripts.parity.cli --batch "
            f"--experiments-dir {env('EXAMPLE_DIR')}/experiments "
            f"--baselines {' '.join(rk for rk, *_ in runs)} --agg-goal {goal_for_parity})")

spec = {
    "title": "FWDLLM RUN",
    "subtitle": f"mode={MODE}  {len(manifest)} run(s)   ·   🟢 set=flag override · (D)=yaml/code default",
    "dry_run": DRY_RUN,
    "tiers": tiers,
    "checks": checks,
    "next": next_cmd,
}
json.dump(spec, open(env("MANIFEST") + ".spec.json", "w"), indent=2)
rc = expt_runner.render_and_gate(spec, show_all=SHOW_ALL)
sys.exit(rc)
PY
GATE_RC=$?

# ---- gate decision ----
# render_and_gate returns 0 (ok) or 2 (blocking check). Anything else means the
# pre-flight step itself failed (e.g. a bad YAML or a spec-builder bug) -- abort
# rather than silently launch on an unvalidated config.
if [ "$GATE_RC" -ne 0 ] && [ "$GATE_RC" -ne 2 ]; then
  echo "ERROR: pre-flight step failed (exit $GATE_RC) -- see traceback above. Nothing launched." >&2
  exit "$GATE_RC"
fi
if [ "$GATE_RC" -eq 2 ] && [ "$FORCE" != "1" ]; then
  echo "Pre-flight BLOCKED (exit 2). Fix the config or pass --force to override. Nothing launched." >&2
  exit 2
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "--dry-run: generated cfgs in $LOGDIR (manifest: $MANIFEST). Nothing launched."
  exit 0
fi
# Real (GPU) run confirmation unless --yes.
if [ "$ASSUME_YES" != "1" ]; then
  read -r -p "Launch the runs above? [y/N] " _ans < /dev/tty || _ans=""
  case "$_ans" in y|Y|yes|YES) ;; *) echo "Aborted (no --yes / declined). Nothing launched."; exit 0 ;; esac
fi

# baseline names (for parity --batch and the summary), derived from RUNS.
RUNS_BASELINES=""
for _e in "${RUNS[@]}"; do RUNS_BASELINES="$RUNS_BASELINES ${_e%%:*}"; done
RUNS_BASELINES="${RUNS_BASELINES# }"

# ---- post-launch hooks (--after ...), dispatched by expt_dispatch_after ----
# Each is a shell function the shared harness calls by name; they own the
# fwdllm-specific command (parity CLI path / sanity extractor / plotter).
after_parity() {
  # Real<->sim parity battery on the pairs just produced. The parity engine
  # lives under async_cifar10/scripts (shared, fwdllm rungs registered in it).
  ( cd "$AC10_DIR" && python -m scripts.parity.cli --batch \
      --experiments-dir "$EXAMPLE_DIR/experiments" \
      --baselines $RUNS_BASELINES --agg-goal "${AGG_GOAL:-${SEL_C:-10}}" \
      --json-out "$LOGDIR/parity_<baseline>.json" )
}
after_sanity() {
  # Per-run sanity signals (selection / data_id / eval-per-data_id / partition).
  local d
  while IFS= read -r d; do
    [ -d "$d" ] || continue
    python "$SCRIPT_DIR/extract_sanity_checks.py" "$d" || true
  done < <(find "$EXAMPLE_DIR/experiments" -maxdepth 1 -type d -name "run_*" -newer "$MANIFEST" 2>/dev/null)
}
after_plot() {
  # Best-effort: analyze_run over the telemetry produced this session.
  local ar="$REPO_ROOT/scripts/analysis/analyze_run.py" d
  [ -f "$ar" ] || { echo "  [after:plot] $ar not found — skipping" >&2; return 0; }
  while IFS= read -r d; do
    [ -d "$d/telemetry" ] || continue
    python "$ar" "$d/telemetry" --out "$LOGDIR/plots_$(basename "$d")" || true
  done < <(find "$EXAMPLE_DIR/experiments" -maxdepth 1 -type d -name "run_*" -newer "$MANIFEST" 2>/dev/null)
}

# ---- PHASE B: launch each generated cfg sequentially ----
declare -A RESULT DURATION_S
ORDERED_KEYS=()
STOP_ALL=0
cd "$REPO_ROOT" || exit 1
while IFS=$'\t' read -r name cfg variant budget; do
  [ -n "$name" ] || continue
  start_ts=$(date +%s)
  expt_launch "$name" "$cfg" "$EXAMPLE_DIR" "$budget" 1 "$LOGDIR"
  rc=$?
  DURATION_S[$name]=$(( $(date +%s) - start_ts ))
  ORDERED_KEYS+=("$name")
  # Health verdict (COMPLETED / CRASH / NO_AGG_ROUNDS / WALL_CEILING) is the
  # source of truth for the summary -- NOT the launcher exit code, which is 0
  # even when the aggregator subprocess crashed (run_experiment swallows the
  # child's non-zero exit). This keeps the per-run line and the summary in
  # agreement, and never calls a mere completion "PASS" (PASS is for checks).
  expt_assert_run "$EXAMPLE_DIR" "$EXPT_LAST_MARKER" "$name"
  if [ "$rc" -ne 0 ]; then
    # Launcher itself failed: surface that, but keep the health word if the
    # scan caught a more specific cause (e.g. CRASH) than a bare exit code.
    case "${EXPT_LAST_HEALTH:-}" in
      COMPLETED|NO_MARKER|"") RESULT[$name]="LAUNCH_EXIT=$rc" ;;
      *)                      RESULT[$name]="${EXPT_LAST_HEALTH}(exit=$rc)" ;;
    esac
  else
    RESULT[$name]="${EXPT_LAST_HEALTH:-COMPLETED}"
  fi
  if [ "$rc" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    echo "--stop-on-fail set; aborting remaining runs."; STOP_ALL=1; break
  fi
done < "$MANIFEST"

# ---- post-launch hooks ----
[ -n "$AFTER" ] && [ "$STOP_ALL" != "1" ] && expt_dispatch_after "$AFTER"

echo ""
echo "=== Summary ==="
for key in "${ORDERED_KEYS[@]}"; do
  printf "  %-40s %-15s %ss\n" "$key" "${RESULT[$key]:-SKIPPED}" "${DURATION_S[$key]:-0}"
done
echo "Logs:     $LOGDIR"
echo "Run dirs: $EXAMPLE_DIR/experiments/run_*"
echo "Parity:   $(cd "$AC10_DIR" && echo "(cd $AC10_DIR && python -m scripts.parity.cli --batch --experiments-dir $EXAMPLE_DIR/experiments --baselines ${RUNS[*]%%:*} --agg-goal ${AGG_GOAL:-${SEL_C:-10}})")"
