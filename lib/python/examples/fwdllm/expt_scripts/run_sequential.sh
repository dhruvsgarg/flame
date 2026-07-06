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
#   --max-data-id    stop a run once data_id reaches this (default 9999, i.e.
#                    effectively unbounded -> the run is governed by --max-runtime-s.
#                    Pass a small value (e.g. 10, 3) ONLY when you deliberately want
#                    a short data-id-capped run. Defaulting high avoids the trap of a
#                    "1h" run silently stopping early at a low data-id cap.
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
# Default high so a run is governed by --max-runtime-s, NOT a silent low data-id cap.
# (An overnight "1h" run once stopped at data_id=10 because the default was 10; the
# real cost of a too-high default is zero since --max-runtime-s still bounds the run.)
MAX_DATA_ID=9999
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
TARGET_ACC=""          # convergence stop (EXPERIMENTS.md WS2): terminate when the last
                       # --converge-window data bins are ALL >= this test accuracy.
CONVERGE_WINDOW=""     # W consecutive-bin window for the convergence stop (default 20 when --target-acc set)
STALL_WINDOW_S=""      # stall guard: terminate EARLY if best acc hasn't gained --stall-min-delta
                       # within this many wall s (empty/0 = off unless registry/--run-set sets it)
STALL_MIN_DELTA=""     # accuracy gain that counts as progress (default 0.01 = 1%)
DELAY_FACTOR=""        # training_delay_factor: divides the registry 4-18s delay. Default (trainer_base) is 10 (=> 0.4-1.8s); pass 1 for the FULL modeled delay (simulate_fwdllm.md #12). Fans to BOTH roles via runner.py.
RUN_SET=""        # load the SHARED condition from experiments.yaml run_sets[NAME]
                  # (single source of truth for multi-node runs; CLI flags override)
ONLY=""
AFTER=""          # comma list of post-launch hooks: parity,sanity,plot (see after_* below)
DRY_RUN=0
ASSUME_YES=0
FORCE=0
SHOW_ALL=0
CLEAN=0           # --clean: auto-kill stray workers from a prior run (default: abort if dirty)

usage() {
  echo "usage: $0 [--mode sim|real|both] [--delays on|off] [--max-runtime-s S] [--max-data-id N]" >&2
  echo "          [--num-trainers N] [--num-gpus N] [--c C] [--c-async C] [--k K] [--agg-goal N]" >&2
  echo "          [--min-initial-trainers N] [--partition-method NAME]" >&2
  echo "          [--var-threshold F] [--max-iter-per-data-id N] [--delay-factor F]" >&2
  echo "          [--target-acc A] [--converge-window W] [--stall-window-s S] [--stall-min-delta D]" >&2
  echo "          [--run-set NAME] [--avail-trace NAME | --avail-traces N1,N2] [--only n1,n2] [--stop-on-fail]" >&2
  echo "          [--dry-run] [--yes] [--force] [--show-all] [--clean]" >&2
  echo "    --clean  auto-kill stray FL workers from a prior/crashed run before each" >&2
  echo "             launch (default: verify clean & ABORT if the node is dirty)." >&2
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
    --target-acc)           TARGET_ACC="$2"; shift 2 ;;
    --converge-window)      CONVERGE_WINDOW="$2"; shift 2 ;;
    --stall-window-s)       STALL_WINDOW_S="$2"; shift 2 ;;
    --stall-min-delta)      STALL_MIN_DELTA="$2"; shift 2 ;;
    --delay-factor)         DELAY_FACTOR="$2"; shift 2 ;;
    --run-set)              RUN_SET="$2"; shift 2 ;;
    --only)                 ONLY="$2"; shift 2 ;;
    --after)                AFTER="$2"; shift 2 ;;
    --stop-on-fail)         STOP_ON_FAIL=1; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    --yes)                  ASSUME_YES=1; shift ;;
    --force)                FORCE=1; shift ;;
    --show-all)             SHOW_ALL=1; shift ;;
    --clean)                CLEAN=1; shift ;;
    *) echo "ERROR: unknown arg '$1'" >&2; usage ;;
  esac
done
case "$MODE" in sim|real|both) ;; *) echo "ERROR: --mode must be sim|real|both (got '$MODE')" >&2; exit 2 ;; esac
case "$DELAYS" in on|off) ;; *) echo "ERROR: --delays must be on|off (got '$DELAYS')" >&2; exit 2 ;; esac

# --run-set NAME: pull the SHARED condition from experiments.yaml so every node in
# a multi-node run launches the SAME condition from ONE source of truth — only
# --only (the baseline subset) differs per node. Explicit CLI flags still WIN;
# the registry only fills knobs the operator left unset. Combined with the
# condition_fp printed in the gate, this is the anti-misconfig backbone: define
# the condition once, verify the fingerprint matches across nodes.
if [ -n "$RUN_SET" ]; then
  REG="$(EXAMPLE_DIR="$EXAMPLE_DIR" RUN_SET="$RUN_SET" python3 - <<'PY'
import os, sys, yaml
p = os.path.join(os.environ["EXAMPLE_DIR"], "experiments.yaml")
try:
    reg = yaml.safe_load(open(p, encoding="utf-8"))
except Exception as e:
    sys.stderr.write(f"ERROR: cannot read {p}: {e}\n"); sys.exit(3)
rs = (reg.get("run_sets") or {}).get(os.environ["RUN_SET"])
if not rs:
    valid = list((reg.get("run_sets") or {}).keys())
    sys.stderr.write(f"ERROR: run_set '{os.environ['RUN_SET']}' not in {p}. Valid: {valid}\n"); sys.exit(3)
c = rs.get("condition", {}) or {}
d = reg.get("defaults", {}) or {}
C = c.get("C", {})
def emit(k, v):
    if v is not None: print(f"REG_{k}={v}")
emit("N", c.get("N")); emit("K", c.get("K"))
if isinstance(C, dict):
    emit("C_SYNC", C.get("sync")); emit("C_ASYNC", C.get("async"))
elif C not in (None, {}):
    emit("C_SYNC", C); emit("C_ASYNC", C)
emit("PART", c.get("partition_method")); emit("TRACE", c.get("avail_trace"))
_dl = c.get("delays")
if _dl is not None:
    emit("DELAYS", "on" if _dl in (True, "on", "ON", "true", 1) else "off")
emit("DELAY_FACTOR", c.get("delay_factor"))
emit("TARGET_ACC", c.get("target_accuracy")); emit("CONVERGE_WINDOW", c.get("converge_window"))
emit("STALL_WINDOW_S", c.get("stall_window_s", d.get("stall_window_s")))
emit("STALL_MIN_DELTA", c.get("stall_min_delta", d.get("stall_min_delta")))
emit("MAX_RUNTIME_S", c.get("max_runtime_s", d.get("max_runtime_s")))
emit("MAX_DATA_ID", c.get("max_data_id_progress", d.get("max_data_id_progress")))
PY
)"
  rc=$?; if [ "$rc" -ne 0 ]; then echo "$REG" >&2; exit "$rc"; fi
  eval "$REG"   # defines REG_* shell vars from the registry condition
  # empty-default knobs: empty ⇒ operator didn't set ⇒ fill from registry
  [ -z "$NUM_TRAINERS" ]      && [ -n "${REG_N:-}" ]               && NUM_TRAINERS="$REG_N"
  [ -z "$SEL_K" ]             && [ -n "${REG_K:-}" ]               && SEL_K="$REG_K"
  [ -z "$SEL_C" ]             && [ -n "${REG_C_SYNC:-}" ]          && SEL_C="$REG_C_SYNC"
  [ -z "$SEL_C_ASYNC" ]       && [ -n "${REG_C_ASYNC:-}" ]         && SEL_C_ASYNC="$REG_C_ASYNC"
  [ -z "$PARTITION_METHOD" ]  && [ -n "${REG_PART:-}" ]            && PARTITION_METHOD="$REG_PART"
  [ -z "$AVAIL_TRACE" ] && [ -z "$AVAIL_TRACES" ] && [ -n "${REG_TRACE:-}" ] && AVAIL_TRACE="$REG_TRACE"
  [ -z "$TARGET_ACC" ]        && [ -n "${REG_TARGET_ACC:-}" ]      && TARGET_ACC="$REG_TARGET_ACC"
  [ -z "$CONVERGE_WINDOW" ]   && [ -n "${REG_CONVERGE_WINDOW:-}" ] && CONVERGE_WINDOW="$REG_CONVERGE_WINDOW"
  [ -z "$DELAY_FACTOR" ]      && [ -n "${REG_DELAY_FACTOR:-}" ]    && DELAY_FACTOR="$REG_DELAY_FACTOR"
  [ -z "$STALL_WINDOW_S" ]    && [ -n "${REG_STALL_WINDOW_S:-}" ]  && STALL_WINDOW_S="$REG_STALL_WINDOW_S"
  [ -z "$STALL_MIN_DELTA" ]   && [ -n "${REG_STALL_MIN_DELTA:-}" ] && STALL_MIN_DELTA="$REG_STALL_MIN_DELTA"
  # non-empty-default knobs: apply registry only when the operator didn't pass the flag
  if [ "$DELAYS_SET" = "0" ] && [ -n "${REG_DELAYS:-}" ]; then DELAYS="$REG_DELAYS"; fi
  if [ "$MAX_RUNTIME_S_SET" = "0" ] && [ -n "${REG_MAX_RUNTIME_S:-}" ]; then MAX_RUNTIME_S="$REG_MAX_RUNTIME_S"; fi
  if [ "$MAX_DATA_ID_SET" = "0" ] && [ -n "${REG_MAX_DATA_ID:-}" ]; then MAX_DATA_ID="$REG_MAX_DATA_ID"; fi
  echo "run-set '$RUN_SET' loaded from experiments.yaml (explicit CLI flags override registry)."
fi

# A convergence run (target-acc set) is governed by accuracy, not the clock, so a
# short wall default would kill a legitimately-learning run early. Default the wall
# ceiling to 48h whenever a target is set and the operator neither passed
# --max-runtime-s nor got a value from a run-set (still the hardcoded 600 default).
if [ -n "$TARGET_ACC" ] && [ "$MAX_RUNTIME_S_SET" = "0" ] && [ "$MAX_RUNTIME_S" = "600" ]; then
  MAX_RUNTIME_S=172800   # 48h
fi

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
VAR_THRESHOLD="$VAR_THRESHOLD" MAX_ITER_PER_DATA_ID="$MAX_ITER_PER_DATA_ID" DELAY_FACTOR="$DELAY_FACTOR" \
TARGET_ACC="$TARGET_ACC" CONVERGE_WINDOW="$CONVERGE_WINDOW" \
STALL_WINDOW_S="$STALL_WINDOW_S" STALL_MIN_DELTA="$STALL_MIN_DELTA" \
MODE_SET="$MODE_SET" DELAYS_SET="$DELAYS_SET" MAX_RUNTIME_S_SET="$MAX_RUNTIME_S_SET" MAX_DATA_ID_SET="$MAX_DATA_ID_SET" \
LOGDIR="$LOGDIR" MANIFEST="$MANIFEST" RUN_TSV="$RUN_TSV" DRY_RUN="$DRY_RUN" SHOW_ALL="$SHOW_ALL" \
EXAMPLE_DIR="$EXAMPLE_DIR" AC10_DIR="$AC10_DIR" \
python - <<'PY'
import os, sys, copy, yaml, json, hashlib
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
DELAY_FACTOR = env("DELAY_FACTOR") or ""
TARGET_ACC = env("TARGET_ACC") or ""; CONVERGE_WINDOW = env("CONVERGE_WINDOW") or ""
STALL_WINDOW_S = env("STALL_WINDOW_S") or ""; STALL_MIN_DELTA = env("STALL_MIN_DELTA") or ""
# "was it passed on the command line?" (override -> green) for the defaulted flags
MODE_SET = env("MODE_SET") == "1"; DELAYS_SET = env("DELAYS_SET") == "1"
MAX_RUNTIME_S_SET = env("MAX_RUNTIME_S_SET") == "1"; MAX_DATA_ID_SET = env("MAX_DATA_ID_SET") == "1"
GPUS_VISIBLE = int(env("GPUS_VISIBLE") or "0")
LOGDIR = env("LOGDIR"); MANIFEST = env("MANIFEST")
DRY_RUN = env("DRY_RUN") == "1"; SHOW_ALL = env("SHOW_ALL") == "1"
delays_on = (DELAYS == "on")

# Availability trace(s). Default to syn_0 (Phase-1, 100% availability) when the
# operator passes no --avail-trace, so patch() ALWAYS sets the mode EXPLICITLY on
# every baseline (trainer availability.mode + aggregator trackTrainerAvail +
# client_notify -- lines below) rather than silently inheriting each yaml's own
# `mode:`. This is what makes the printed "trace" row match what actually runs:
# the resolved value is patched into the launched cfg, not just displayed.
_trace_raw = [t for t in (env("TRACE_CSV") or "").replace(",", " ").split()]
trace_set = bool(_trace_raw)              # operator passed --avail-trace(s)?
traces = _trace_raw or ["syn_0"]          # Phase-1 default: 100% availability
multi_trace = len(traces) > 1

variants = {"real": 0, "sim": 1} if MODE == "both" else {MODE: (0 if MODE == "real" else 1)}

# Baseline-distinguishing internals (selector algorithm / optimizer / sync|async)
# come from the shared catalog _metadata/baselines.yaml, merged at LAUNCH — NOT
# from the per-run YAML the operator edits. Surface them in the review table so a
# mis-picked baseline (e.g. a sync selector where async was intended) is caught
# BEFORE the run, not after. Best-effort: if the catalog can't be read, the
# columns show "?" rather than blocking.
_BL_INTERNALS = {}
try:
    _bl_path = os.path.join(env("EXAMPLE_DIR"), "..", "_metadata", "baselines.yaml")
    _bl = yaml.safe_load(open(_bl_path, encoding="utf-8"))
    _bl = _bl.get("baselines", _bl)
    for _name, _b in (_bl or {}).items():
        _agg = (_b or {}).get("aggregator", {}) or {}
        _sel = _agg.get("selector", {}) or {}
        _opt = _agg.get("optimizer", {}) or {}
        _is_async = bool((_sel.get("kwargs", {}) or {}).get("is_async"))
        _BL_INTERNALS[_name] = {
            "selector": _sel.get("sort", "?"),
            "optimizer": _opt.get("sort", "?"),
            "async": "async" if _is_async else "sync",
        }
except Exception:
    _BL_INTERNALS = {}

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
    # training_delay_factor (simulate_fwdllm.md #12): divides the registry 4-18s
    # delay (trainer_base default 10 => 0.4-1.8s). Set it on the TRAINER
    # hyperparameters; runner.py fans the same value into the aggregator so both
    # roles agree. Only patched when explicitly passed (else the base default).
    if DELAY_FACTOR:
        exp["trainer"].setdefault("hyperparameters", {})
        exp["trainer"]["hyperparameters"]["training_delay_factor"] = float(DELAY_FACTOR)
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
                # RESOLVED availability mode read back from the PATCHED cfg (what
                # actually launches), so the table can't show a stale default.
                "avail": e0["trainer"].get("availability", {}).get("mode"),
                "async": (run_key == "fluxtune"),
                # baseline-distinguishing internals from the shared catalog
                "selector": _BL_INTERNALS.get(run_key, {}).get("selector", "?"),
                "optimizer": _BL_INTERNALS.get(run_key, {}).get("optimizer", "?"),
                "sync_async": _BL_INTERNALS.get(run_key, {}).get("async", "?"),
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

# --- condition fingerprint (TWO-NODE SAFETY) ----------------------------------
# A short hash over the SHARED condition axes that MUST match for a valid cross-
# baseline comparison. agg_goal / selector / optimizer legitimately DIFFER per
# baseline, so they are EXCLUDED. Uses RESOLVED partition/trace (what actually
# runs), not just the flags. Print it on every node: if node A and node B show
# the SAME fingerprint, they launched the same condition — the single check that
# catches a mistyped flag on the second node before the runs diverge.
_res_parts = sorted({str(b.get("partition")) for b in per_baseline.values()})
_res_traces = sorted({str(b.get("avail")) for b in per_baseline.values()})
_cond = {
    "N": NUM_TRAINERS or "yaml", "K": SEL_K or "yaml",
    "C_sync": SEL_C or "yaml", "C_async": SEL_C_ASYNC or SEL_C or "yaml",
    "partition": _res_parts, "trace": _res_traces,
    "delays": "on" if delays_on else "off",
    "delay_factor": DELAY_FACTOR or "base",
    "target_acc": TARGET_ACC or "none",
    "stall_window_s": STALL_WINDOW_S or "off", "stall_min_delta": STALL_MIN_DELTA or "off",
    "converge_window": (CONVERGE_WINDOW or "20") if TARGET_ACC else "none",
    "max_runtime_s": MAX_RUNTIME_S, "max_data_id": MAX_DATA_ID,
}
_cond_fp = hashlib.sha256(json.dumps(_cond, sort_keys=True).encode()).hexdigest()[:8]

tiers = []
# ① review every run
# The trace row reflects the RESOLVED per-baseline availability (read back from
# the patched cfgs), NOT a hardcoded default -- so "what is printed" == "what
# runs". If every baseline resolved to the same mode, show it; otherwise flag
# the divergence and defer to the per-baseline table (tier ②).
trace_overridden = trace_set
_resolved_avails = {b.get("avail") for b in per_baseline.values() if b.get("avail")}
if len(_resolved_avails) == 1:
    trace_val = next(iter(_resolved_avails))
elif _resolved_avails:
    trace_val = "MIXED: " + ",".join(sorted(a or "?" for a in _resolved_avails)) + " (see ②)"
else:
    trace_val = " ".join(traces)
# mode: single-sided always warns (parity needs both), regardless of override.
if MODE != "both":
    mode_row = {"label": "mode", "value": MODE, "level": "warn", "note": "single-sided: parity needs both"}
else:
    mode_row = scalar_row("mode", MODE, MODE_SET, note="real+sim pair (--mode)")
tier1 = {"name": "① REVIEW EVERY RUN", "rows": [
    {"label": "condition_fp", "value": _cond_fp, "level": "set",
     "note": "TWO-NODE CHECK: same fingerprint on every node ⇒ same shared condition "
             "(N/K/C/partition/trace/delays/target_acc/caps). Differs ⇒ a knob was mistyped."},
    mode_row,
    {"label": "baselines", "value": " ".join(rk for rk, *_ in runs)},
    # The two similarly-named-but-DIFFERENT knobs, disambiguated + on their own rows:
    scalar_row("max_runtime_s", MAX_RUNTIME_S, MAX_RUNTIME_S_SET, note="wall/vclock cap (--max-runtime-s)"),
    scalar_row("max_data_id_progress", MAX_DATA_ID, MAX_DATA_ID_SET,
               note="STOP condition: stop when data_id reaches this (--max-data-id)"),
    scalar_row("trace", trace_val, trace_overridden,
               note=("resolved availability actually patched into each launched cfg; "
                     + ("100% availability (Phase 1)" if _resolved_avails == {"syn_0"}
                        else "NON-syn_0 — unavailability (Phase 2+)"))),
    scalar_row("enable_training_delays", str(delays_on).lower(), DELAYS_SET,
               note=(f"modeled training delay {'ON (D>0)' if delays_on else 'OFF (D=0)'}; "
                     f"delay_factor={DELAY_FACTOR or 'base(10)'} (divides yaml base 4-18s delay); matched BOTH sides — K-D8")),
    # var_threshold / max_iterations_per_data_id vary with data heterogeneity ->
    # review-every-run (warn when defaulted). NOTE: max_iters_per_data_id is the
    # FORCE-COMMIT cap and is NOT the same as max_data_id_progress (the stop) above.
    scalar_row("var_threshold", VAR_THRESHOLD if VAR_THRESHOLD else "unset",
               bool(VAR_THRESHOLD), review=True,
               note="variance-pass gate; varies w/ data heterogeneity (--var-threshold). unset ⇒ trainer/code default"),
    scalar_row("max_iters_per_data_id", MAX_ITER if MAX_ITER else "unset",
               bool(MAX_ITER), review=True,
               note="FORCE-COMMIT cap (--max-iter-per-data-id) — NOT the max_data_id_progress stop above. unset ⇒ code default"),
    # Convergence stop (EXPERIMENTS.md WS2): terminate when the last W data bins
    # are ALL >= target accuracy. When set, max_runtime_s/max_data_id become
    # SAFETY CAPS (a non-converging run -> DID_NOT_CONVERGE). unset ⇒ time/data-id bound only.
    scalar_row("target_acc", TARGET_ACC if TARGET_ACC else "unset",
               bool(TARGET_ACC), review=True,
               note=("convergence stop: last %s bins all >= this (--target-acc). "
                     "unset ⇒ NO accuracy stop, only max_runtime_s/max_data_id"
                     % (CONVERGE_WINDOW or "20"))),
    # Stall guard: early-terminate a not-learning run before the wall ceiling.
    scalar_row("stall_guard",
               (f"<{STALL_MIN_DELTA or '0.01'} acc in {int(float(STALL_WINDOW_S))//3600}h→STALLED"
                if STALL_WINDOW_S and float(STALL_WINDOW_S) > 0 else "off"),
               bool(STALL_WINDOW_S), review=True,
               note=("terminate EARLY if best acc gains < stall_min_delta within stall_window_s "
                     "(--stall-window-s/--stall-min-delta). off ⇒ run to convergence or the wall ceiling")),
]}
tiers.append(tier1)

# ② per-baseline -- an ALIGNED TABLE (baselines = rows, knobs = columns). The
# renderer highlights any column whose value differs across the baselines (those
# are the ones to eyeball); columns identical across all 3 stay dim (expected).
tier2_cols = [
    ("sync_async", "mode"), ("selector", "selector"), ("optimizer", "optim"),
    ("c", "c"), ("agg_goal", "agg_goal"), ("k", "k"),
    ("min_init", "minInit"), ("n_trainers", "n_trainers"),
    ("n_gpus", "n_gpus"), ("partition", "part"), ("avail", "avail"),
]
overridden2 = []
if bool(SEL_C) or bool(SEL_C_ASYNC): overridden2.append("c")
if bool(AGG_GOAL) or bool(SEL_C):    overridden2.append("agg_goal")
if bool(SEL_K):        overridden2.append("k")
if bool(MIN_INIT):     overridden2.append("min_init")
if bool(NUM_TRAINERS): overridden2.append("n_trainers")
if bool(NUM_GPUS):     overridden2.append("n_gpus")
if bool(PART):         overridden2.append("partition")
if trace_set:          overridden2.append("avail")
rows2 = []
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    rows2.append({"name": rk, "cells": {
        "sync_async": b.get("sync_async"), "selector": b.get("selector"),
        "optimizer": b.get("optimizer"),
        "c": b.get("c"), "agg_goal": b.get("agg_goal"), "k": b.get("k"),
        "min_init": b.get("min_init"), "n_trainers": b.get("n_trainers"),
        "n_gpus": b.get("n_gpus"), "partition": b.get("partition"),
        "avail": b.get("avail"),
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
# agg_goal MATCHES across baselines (operator invariant 2026-07-06): the aggregation
# batch size should be identical for a fair head-to-head; a mismatch is almost always
# an unintended fan from --c/--c-async. Warn (visible), don't block (an experiment
# MAY intentionally vary it -- but then it's an eyeballed choice, not a silent one).
_goals = {rk: per_baseline.get(rk, {}).get("agg_goal") for rk in (r[0] for r in runs)}
_gset = {g for g in _goals.values() if g is not None}
if len(_gset) > 1:
    checks.append({"name": "agg_goal matches across baselines", "level": "warn",
                   "detail": f"agg_goal differs: {_goals} — intended? (fair comparison expects one value)"})
elif _gset:
    checks.append({"name": "agg_goal matches across baselines", "level": "ok",
                   "detail": f"all baselines agg_goal={next(iter(_gset))}"})
# Availability consistency + sync-barrier liveness: surface the RESOLVED per-
# baseline trace (what actually runs), and BLOCK a full-participation sync
# barrier under a non-syn_0 trace -- agg_goal == n_trainers can never assemble if
# any trainer is unavailable, so the real barrier waits to the wall cap (the
# fwdllm_plus / K-D20 stall; Stage C's wait bounds the log but still can't
# complete when full participation is required under scarcity).
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    av, g, n, is_async = b.get("avail"), b.get("agg_goal"), b.get("n_trainers"), b.get("async")
    if av and av != "syn_0":
        if (not is_async) and isinstance(g, int) and isinstance(n, int) and g >= n:
            checks.append({"name": f"availability liveness ({rk})", "level": "error",
                           "detail": f"trace={av} + sync agg_goal={g} >= n_trainers={n}: "
                                     f"barrier can't assemble under unavailability → stall"})
        else:
            checks.append({"name": f"availability ({rk})", "level": "warn",
                           "detail": f"trace={av} — non-syn_0 unavailability (Phase 2+); confirm intended"})
    else:
        checks.append({"name": f"availability ({rk})", "level": "ok", "detail": f"trace={av}"})
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

# Convergence stop (EXPERIMENTS.md WS2): export so expt_launch arms the watcher.
# Only when --target-acc was passed; otherwise runs stay governed by their caps
# (default behavior unchanged). Window defaults to 20 bins.
if [ -n "$TARGET_ACC" ]; then
  export EXPT_TARGET_ACC="$TARGET_ACC"
  export EXPT_CONVERGE_WINDOW="${CONVERGE_WINDOW:-20}"
  [ -n "$STALL_WINDOW_S" ]  && export EXPT_STALL_WINDOW_S="$STALL_WINDOW_S"
  [ -n "$STALL_MIN_DELTA" ] && export EXPT_STALL_MIN_DELTA="$STALL_MIN_DELTA"
fi

# ---- PHASE B: launch each generated cfg sequentially ----
declare -A RESULT DURATION_S
ORDERED_KEYS=()
STOP_ALL=0
cd "$REPO_ROOT" || exit 1

# Backstop trap for Ctrl+C landing OUTSIDE a run (between baselines, after-hooks);
# expt_launch installs its own thorough handler during each run. Sweep + exit.
_rs_interrupt() {
  trap - INT TERM
  echo "" >&2
  echo "[$(date '+%F %T')] INTERRUPT — aborting run-set, sweeping any workers ..." >&2
  pkill -TERM -f 'flame.launch.run_experiment' 2>/dev/null || true
  pkill -9 -f 'trainer/forward_training'  2>/dev/null || true
  pkill -9 -f 'trainer/pytorch/main.py'   2>/dev/null || true
  pkill -9 -f 'aggregator/pytorch/main_'  2>/dev/null || true
  pkill -9 -f converge_watch.py           2>/dev/null || true
  exit 130
}
trap _rs_interrupt INT TERM

[ "$CLEAN" = "1" ] && export EXPT_AUTOCLEAN=1   # --clean -> preflight kills stragglers instead of aborting
while IFS=$'\t' read -r name cfg variant budget; do
  [ -n "$name" ] || continue
  trap _rs_interrupt INT TERM   # re-arm: expt_launch clears its trap on return
  # Clean-slate guard: refuse to launch on top of a prior run's stray workers.
  if ! expt_assert_clean_slate "$name"; then
    RESULT[$name]="DIRTY_ABORT"; ORDERED_KEYS+=("$name"); DURATION_S[$name]=0
    echo "  [$name] aborting: node not clean (use --clean to auto-kill stragglers)." >&2
    STOP_ALL=1; break
  fi
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
  if [ "${EXPT_LAST_HEALTH:-}" = "CONVERGED" ] || [ "${EXPT_LAST_HEALTH:-}" = "STALLED" ]; then
    # Watcher kills the run's process group -> rc is the SIGKILL code (expected),
    # NOT a failure. Report the clean verdict (CONVERGED / STALLED) without exit noise.
    RESULT[$name]="${EXPT_LAST_HEALTH}"
  elif [ "$rc" -ne 0 ]; then
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
