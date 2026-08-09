#!/bin/bash
# Drive the fwdllm real<->sim launcher pairs (9-baseline FluxTune matrix, see
# BASELINES.md: fwdllm/fwdllm_it_unaware/fwdllm_it_oracular,
# fedbuff_round/fedbuff_it_unaware/fedbuff_it_oracular, felix_round/felix_it,
# fluxtune) for parity runs. Thin driver over examples/scripts/expt_runner.sh;
# owns the fwdllm baseline->(real yaml, sim yaml) map, knob patching, and the
# pre-flight gate. All 9 baselines now have BOTH a real and a sim yaml, so
# --mode both works over the full set (the 6 net-new ones got their real yamls
# in 1f419505; BRIDGE_DESIGN.md decision #3's sim-only rule applied only while
# those were missing). The "source yaml exists" check below still blocks any
# baseline whose side is absent.
# --mode both pairs each baseline's real+sim (names tagged _real/_sim so
# scripts.parity.cli globs the pair); --delays sets enable_training_delays
# IDENTICALLY both sides (mismatched D = false divergence). Pre-flight prints a
# tiered table + refuses infeasible configs (--dry-run preview, --yes skip
# confirm, --force override a block).
#
# delays/delay-divisor/delay-floor default to each baseline's settled value
# (BASELINE_DELAY_DEFAULTS below) instead of a global off -- a forgotten --delays
# used to silently collapse sim throughput (simulate_fwdllm.md §G, 07-18m).
#
# Usage (from anywhere):
#   run_sequential.sh [--mode sim|real|both] [--delays on|off]
#       [--max-runtime-s 600] [--max-data-id 10] [--num-trainers N] [--num-gpus N] [--gpu-ids N1,N2,...]
#       [--c C] [--c-async C] [--k K] [--agg-goal N] [--min-initial-trainers N]
#       [--partition-method NAME] [--avail-trace NAME | --avail-traces N1,N2]
#       [--only name1,name2] [--stop-on-fail] [--dry-run] [--yes] [--force]
#       [--show-all]
#
#   --mode           time_mode variant(s) per baseline (default both).
#   --delays         enable_training_delays BOTH sides (default: per-baseline, see above).
#   --max-runtime-s  wall/vclock cap per run (default 600).
#   --max-data-id    stop when data_id reaches this (default 9999 = unbounded).
#   --num-trainers / --num-gpus  override trainer.num_trainers / execution.num_gpus.
#   --gpu-ids        explicit CUDA-ordinal allowlist (e.g. 1,2,3,4,5,6,7 to skip a
#                    known-bad GPU 0); overrides range(num_gpus) for both trainer
#                    round-robin and the aggregator's pinned GPU. Implies
#                    num_gpus=len(list) unless --num-gpus is also given.
#   --c              selector.kwargs.c (+ minInitialTrainers + agg_goal unless overridden).
#   --c-async        selector.kwargs.c for async baselines (fedbuff_round/it_*,
#                    felix_round/it, fluxtune) only.
#   --k / --agg-goal  BOTH set aggregator.agg_goal -- K and agg_goal are the same
#                    knob (selector.kwargs.k is dead; nothing reads it).
#                    --agg-goal wins over --k, which wins over --c's fallback.
#   --min-initial-trainers / --min-initial-frac  join barrier before first selection:
#                    absolute count, or floor(F*N). DEFAULT = N (wait for ALL trainers ->
#                    set-exact initial cohort real<->sim). frac<1 tolerates stragglers but
#                    reintroduces a pool-size race; N blocks forever if a trainer never joins.
#   --partition-method  hyperparameters.partition_method (agnews_partition.h5 group; default uniform/IID).
#   --var-threshold / --max-iter-per-data-id  variance gate / force-commit cap (review each run, tier ①).
#   --avail-trace / --avail-traces  availability trace(s); Phase 1 uses syn_0.
#   --only           baseline subset (default all three).
#   --after          post-launch hooks: parity,sanity,plot.
#   --stop-on-fail   abort remaining runs on first non-zero exit.
#   --dry-run / --yes / --force / --show-all  preview / skip-confirm / override-block / expand table.
# Other knobs (see usage()): --var-stopping-policy --agg-rate-type --target-acc
# --converge-window --stall-* --delay-divisor --delay-floor --run-set --clean.
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
DELAYS="off"    # placeholder when DELAYS_SET=0 -- python resolves the real
                # per-baseline default (BASELINE_DELAY_DEFAULTS)
MAX_RUNTIME_S=600
MAX_DATA_ID=9999       # high => --max-runtime-s governs (not a silent data-id cap)
# "was this passed?" flags for non-empty-default knobs (value alone can't tell override from default)
MODE_SET=0; DELAYS_SET=0; MAX_RUNTIME_S_SET=0; MAX_DATA_ID_SET=0
STOP_ON_FAIL=0
NUM_TRAINERS=""
NUM_GPUS=""
GPU_IDS=""      # comma-separated CUDA ordinal allowlist (skip a known-bad GPU); sets
                # execution.gpu_ids and, unless --num-gpus is also given, num_gpus=len(ids)
SEL_C=""
SEL_C_ASYNC=""
SEL_K=""
AGG_GOAL=""
MIN_INIT_TRAINERS=""
MIN_INIT_FRAC=""
AVAIL_TRACE=""
AVAIL_TRACES=""
PARTITION_METHOD=""
VAR_THRESHOLD=""       # variance-pass gate threshold; varies with data heterogeneity -> review every run
SERVER_UPDATE_AUDIT="" # I-1 audit: per-commit ||delta||/||w||. OFF by default -- never on a replicate leg
POOL_SPLIT_HALF_AUDIT="" # L1 audit: per-commit pool split-half cosine. OFF -- adds a pass over (params x uploads)
LEARNING_RATE=""       # server step size (aggregator hyperparameters); empty => trainer_base.yaml (0.01)
PERTURBATION_COUNT=""  # P: probes per trainer per iteration (trainer hyperparameters); empty => code default 10
PROBE_COMBINE=""       # S-H: select|mean -- how the P probes become one upload; empty => code default select
MAX_ITER_PER_DATA_ID=""  # force-commit cap (max_iterations_per_data_id); review every run
VAR_STOPPING_POLICY=""   # Opt-2: off|fixed_cap|plateau (empty => baselines.yaml, fluxtune=plateau)
AGG_RATE_TYPE=""         # Opt-3: grad_aware|new (empty => baselines.yaml, fluxtune=grad_aware; new=FeLiX)
TARGET_ACC=""          # convergence stop: terminate when last --converge-window bins all >= this acc
CONVERGE_WINDOW=""     # W-bin window for the convergence stop (default 20 when --target-acc set)
SIM_WALL_CEILING_S=""  # sim-mode REAL-wall-clock outer safety (hyperparameters.sim_wall_ceiling_s).
                        # unset => code default = max_runtime_s * 20 (fwdllm_aggregator.py
                        # SIM_WALL_CEILING_FACTOR) -- with max_runtime_s in VIRTUAL/vclock seconds
                        # (sim mode), that default is ~20x too loose to bound REAL run duration
                        # (e.g. 48h vclock budget -> 40-DAY real-wall failsafe). Set this explicitly
                        # whenever the vclock budget alone doesn't bound how long the run can
                        # occupy GPUs in real time (e.g. an overnight launch).
STALL_WINDOW_S=""      # stall guard: terminate EARLY if no progress within this many wall s (--stall-window-s/-h)
STALL_MIN_DELTA=""     # accuracy gain that counts as progress (default 0.01 = 1%)
STALL_ON=""            # signal that resets the idle clock: acc|loss|either (default either)
LOSS_MIN_REL_DELTA=""  # relative test-loss drop vs running-best that counts as progress (default 0.01)
DELAY_FACTOR=""        # DIVISOR on the registry 4-18s delay (>1 shortens, <1 lengthens). unset =>
                        # BASELINE_DELAY_DEFAULTS. --delay-divisor
DELAY_FLOOR=""         # floor on the RAW registry delay, before the divisor. unset =>
                        # BASELINE_DELAY_DEFAULTS. FWDLLM_DESIGN.md §O
RUN_SET=""             # load SHARED condition from experiments.yaml run_sets[NAME] (CLI flags override)
ONLY=""
AFTER=""               # post-launch hooks: parity,sanity,plot
DRY_RUN=0
ASSUME_YES=0
FORCE=0
SHOW_ALL=0
CLEAN=0           # --clean: auto-kill stray workers from a prior run (default: abort if dirty)

usage() {
  echo "usage: $0 [--mode sim|real|both] [--delays on|off] [--max-runtime-s S] [--max-data-id N]" >&2
  echo "          [--num-trainers N] [--num-gpus N] [--gpu-ids N1,N2,...] [--c C] [--c-async C] [--k K] [--agg-goal N]" >&2
  echo "          [--min-initial-trainers N] [--partition-method NAME]" >&2
  echo "          [--var-threshold F] [--max-iter-per-data-id N] [--delay-divisor F (=--delay-factor; DIVISOR, <1 lengthens)]" >&2
  echo "          [--delay-floor F (floor on raw registry delay, applied before the divisor)]" >&2
  echo "    --delays/--delay-divisor/--delay-floor default to each baseline's settled value" >&2
  echo "          (BASELINE_DELAY_DEFAULTS in this script); pass explicitly only to override." >&2
  echo "          [--server-update-audit] [--pool-split-half-audit]" >&2
  echo "          [--learning-rate F] [--perturbation-count N] [--probe-combine select|mean]" >&2
  echo "          [--target-acc A] [--converge-window W] [--stall-window-s S | --stall-window-h H] [--stall-min-delta D]" >&2
  echo "          [--stall-on acc|loss|either] [--loss-min-rel-delta R]" >&2
  echo "          [--sim-wall-ceiling-s S | --sim-wall-ceiling-h H]  REAL-wall-clock outer safety" >&2
  echo "          (sim mode only; unset => max_runtime_s(vclock-s) * 20 -- far too loose to bound" >&2
  echo "          REAL duration, e.g. 48h vclock => 40-day real failsafe. Set for an unattended run." >&2
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
    --gpu-ids)              GPU_IDS="$2"; shift 2 ;;
    --c)                    SEL_C="$2"; shift 2 ;;
    --c-async)              SEL_C_ASYNC="$2"; shift 2 ;;
    --k)                    SEL_K="$2"; shift 2 ;;
    --agg-goal)             AGG_GOAL="$2"; shift 2 ;;
    --min-initial-trainers) MIN_INIT_TRAINERS="$2"; shift 2 ;;
    --min-initial-frac)     MIN_INIT_FRAC="$2"; shift 2 ;;
    --avail-trace)          AVAIL_TRACE="$2"; shift 2 ;;
    --avail-traces)         AVAIL_TRACES="$2"; shift 2 ;;
    --partition-method)     PARTITION_METHOD="$2"; shift 2 ;;
    --var-threshold)        VAR_THRESHOLD="$2"; shift 2 ;;
    --server-update-audit)  SERVER_UPDATE_AUDIT=1; shift ;;
    --pool-split-half-audit) POOL_SPLIT_HALF_AUDIT=1; shift ;;
    --learning-rate)        LEARNING_RATE="$2"; shift 2 ;;
    --perturbation-count)   PERTURBATION_COUNT="$2"; shift 2 ;;
    --probe-combine)        PROBE_COMBINE="$2"; shift 2 ;;
    --max-iter-per-data-id) MAX_ITER_PER_DATA_ID="$2"; shift 2 ;;
    --var-stopping-policy)  case "$2" in off|fixed_cap|plateau) ;; *) echo "ERROR: --var-stopping-policy must be off|fixed_cap|plateau (got '$2')" >&2; exit 2 ;; esac
                            VAR_STOPPING_POLICY="$2"; shift 2 ;;
    --agg-rate-type)        case "$2" in grad_aware|new|old) ;; *) echo "ERROR: --agg-rate-type must be grad_aware|new|old (got '$2')" >&2; exit 2 ;; esac
                            AGG_RATE_TYPE="$2"; shift 2 ;;
    --target-acc)           TARGET_ACC="$2"; shift 2 ;;
    --converge-window)      CONVERGE_WINDOW="$2"; shift 2 ;;
    --sim-wall-ceiling-s)   SIM_WALL_CEILING_S="$2"; shift 2 ;;
    --sim-wall-ceiling-h)   # hours alias, mirrors --stall-window-h
      case "$2" in ''|*[!0-9.]*|*.*.*) echo "ERROR: --sim-wall-ceiling-h needs a number of hours (got '$2')" >&2; exit 2 ;; esac
      SIM_WALL_CEILING_S="$(awk "BEGIN{printf \"%d\", ($2)*3600}")"; shift 2 ;;
    --stall-window-s)       STALL_WINDOW_S="$2"; shift 2 ;;
    --stall-window-h)       # ergonomic hours alias -> seconds (e.g. --stall-window-h 6 => 21600)
      case "$2" in ''|*[!0-9.]*|*.*.*) echo "ERROR: --stall-window-h needs a number of hours (got '$2')" >&2; exit 2 ;; esac
      STALL_WINDOW_S="$(awk "BEGIN{printf \"%d\", ($2)*3600}")"; shift 2 ;;
    --stall-min-delta)      STALL_MIN_DELTA="$2"; shift 2 ;;
    --stall-on)             case "$2" in acc|loss|either) ;; *) echo "ERROR: --stall-on must be acc|loss|either (got '$2')" >&2; exit 2 ;; esac
                            STALL_ON="$2"; shift 2 ;;
    --loss-min-rel-delta)   LOSS_MIN_REL_DELTA="$2"; shift 2 ;;
    --delay-divisor|--delay-factor)  DELAY_FACTOR="$2"; shift 2 ;;
    --delay-floor)           DELAY_FLOOR="$2"; shift 2 ;;
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

# --run-set NAME: pull the SHARED condition from experiments.yaml so every node in a
# multi-node run launches the same condition from one source of truth (only --only
# differs per node). Explicit CLI flags WIN; the registry fills only unset knobs.
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
emit("STALL_ON", c.get("stall_on", d.get("stall_on")))
emit("LOSS_MIN_REL_DELTA", c.get("loss_min_rel_delta", d.get("loss_min_rel_delta")))
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
  [ -z "$STALL_ON" ]          && [ -n "${REG_STALL_ON:-}" ]        && STALL_ON="$REG_STALL_ON"
  [ -z "$LOSS_MIN_REL_DELTA" ] && [ -n "${REG_LOSS_MIN_REL_DELTA:-}" ] && LOSS_MIN_REL_DELTA="$REG_LOSS_MIN_REL_DELTA"
  # non-empty-default knobs: apply registry only when the operator didn't pass the flag.
  # Also mark DELAYS_SET so resolve_delay_settings() doesn't then fall through to
  # BASELINE_DELAY_DEFAULTS and discard the registry's value.
  if [ "$DELAYS_SET" = "0" ] && [ -n "${REG_DELAYS:-}" ]; then DELAYS="$REG_DELAYS"; DELAYS_SET=1; fi
  if [ "$MAX_RUNTIME_S_SET" = "0" ] && [ -n "${REG_MAX_RUNTIME_S:-}" ]; then MAX_RUNTIME_S="$REG_MAX_RUNTIME_S"; fi
  if [ "$MAX_DATA_ID_SET" = "0" ] && [ -n "${REG_MAX_DATA_ID:-}" ]; then MAX_DATA_ID="$REG_MAX_DATA_ID"; fi
  echo "run-set '$RUN_SET' loaded from experiments.yaml (explicit CLI flags override registry)."
fi

# A convergence run (target-acc) is governed by accuracy, not the clock, so bump the
# wall ceiling to 48h when a target is set and --max-runtime-s is still the 600 default.
if [ -n "$TARGET_ACC" ] && [ "$MAX_RUNTIME_S_SET" = "0" ] && [ "$MAX_RUNTIME_S" = "600" ]; then
  MAX_RUNTIME_S=172800   # 48h
fi

# baseline -> (real yaml : sim yaml). Plain baseline names, independent of the
# "n10"/"n100" baked into each source filename. Every entry's two sides must
# agree on the shared condition (partition_method above all) -- a mismatch is a
# parity divergence with no clock cause. The pre-flight "source yaml exists"
# check blocks a missing side rather than silently launching one-sided.
ALL_RUNS=(
  "fwdllm:$SCRIPT_DIR/fwdllm_n100_smoke.yaml:$SCRIPT_DIR/fwdllm_n100_smoke_sim.yaml"
  "fwdllm_it_unaware:$SCRIPT_DIR/fwdllm_it_unaware_n100_smoke.yaml:$SCRIPT_DIR/fwdllm_it_unaware_n100_smoke_sim.yaml"
  "fwdllm_it_oracular:$SCRIPT_DIR/fwdllm_it_oracular_n100_smoke.yaml:$SCRIPT_DIR/fwdllm_it_oracular_n100_smoke_sim.yaml"
  # NAME IS MISLEADING: "n10" here does NOT mean 10 trainers -- these two files
  # are the full n=100/c=30/agg_goal=10 production scale (matches every other
  # baseline below), just never renamed. Don't swap these for the "n15" files:
  # fedbuff_round_n15_smoke.yaml/felix_round_n15_smoke.yaml are a DELIBERATELY
  # reduced-scale repro (n=15/c=10/agg_goal=5) built solely to cheaply debug the
  # TIMING_OVERRUN/compute-contention question (see that file's own header) --
  # a special-purpose debug tool this session used for fast iteration, not a
  # replacement for the production condition every other baseline runs at.
  "fedbuff_round:$SCRIPT_DIR/fedbuff_round_n10_smoke.yaml:$SCRIPT_DIR/fedbuff_round_n10_smoke_sim.yaml"
  "fedbuff_it_unaware:$SCRIPT_DIR/fedbuff_it_unaware_n10_smoke.yaml:$SCRIPT_DIR/fedbuff_it_unaware_n10_smoke_sim.yaml"
  "fedbuff_it_oracular:$SCRIPT_DIR/fedbuff_it_oracular_n10_smoke.yaml:$SCRIPT_DIR/fedbuff_it_oracular_n10_smoke_sim.yaml"
  "felix_round:$SCRIPT_DIR/felix_round_n10_smoke.yaml:$SCRIPT_DIR/felix_round_n10_smoke_sim.yaml"
  "felix_it:$SCRIPT_DIR/felix_it_n10_smoke.yaml:$SCRIPT_DIR/felix_it_n10_smoke_sim.yaml"
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
# One python step: the operator sees the whole matrix (baselines x traces x variants)
# and confirms once. Writes per-run cfgs + a launch manifest, renders the tiered
# table via expt_runner.render_and_gate, and exits 2 if any check is blocking.
RUN_TSV="$LOGDIR/_runs.tsv"; : > "$RUN_TSV"
for entry in "${RUNS[@]}"; do
  bl="${entry%%:*}"; rest="${entry#*:}"; real_y="${rest%%:*}"; sim_y="${rest#*:}"
  printf '%s\t%s\t%s\n' "$bl" "$real_y" "$sim_y" >> "$RUN_TSV"
done

EXPT_RUNNER_DIR="$EXPT_RUNNER_DIR" \
MODE="$MODE" DELAYS="$DELAYS" MAX_RUNTIME_S="$MAX_RUNTIME_S" MAX_DATA_ID="$MAX_DATA_ID" \
NUM_TRAINERS="$NUM_TRAINERS" NUM_GPUS="$NUM_GPUS" GPU_IDS="$GPU_IDS" SEL_C="$SEL_C" SEL_C_ASYNC="$SEL_C_ASYNC" \
SEL_K="$SEL_K" AGG_GOAL="$AGG_GOAL" MIN_INIT_TRAINERS="$MIN_INIT_TRAINERS" MIN_INIT_FRAC="$MIN_INIT_FRAC" \
PARTITION_METHOD="$PARTITION_METHOD" TRACE_CSV="$TRACE_CSV" GPUS_VISIBLE="$GPUS_VISIBLE" \
VAR_THRESHOLD="$VAR_THRESHOLD" MAX_ITER_PER_DATA_ID="$MAX_ITER_PER_DATA_ID" DELAY_FACTOR="$DELAY_FACTOR" \
SERVER_UPDATE_AUDIT="$SERVER_UPDATE_AUDIT" POOL_SPLIT_HALF_AUDIT="$POOL_SPLIT_HALF_AUDIT" \
LEARNING_RATE="$LEARNING_RATE" PERTURBATION_COUNT="$PERTURBATION_COUNT" \
  PROBE_COMBINE="$PROBE_COMBINE" \
DELAY_FLOOR="$DELAY_FLOOR" \
VAR_STOPPING_POLICY="$VAR_STOPPING_POLICY" AGG_RATE_TYPE="$AGG_RATE_TYPE" \
TARGET_ACC="$TARGET_ACC" CONVERGE_WINDOW="$CONVERGE_WINDOW" \
SIM_WALL_CEILING_S="$SIM_WALL_CEILING_S" \
STALL_WINDOW_S="$STALL_WINDOW_S" STALL_MIN_DELTA="$STALL_MIN_DELTA" \
STALL_ON="$STALL_ON" LOSS_MIN_REL_DELTA="$LOSS_MIN_REL_DELTA" \
MODE_SET="$MODE_SET" DELAYS_SET="$DELAYS_SET" MAX_RUNTIME_S_SET="$MAX_RUNTIME_S_SET" MAX_DATA_ID_SET="$MAX_DATA_ID_SET" \
LOGDIR="$LOGDIR" MANIFEST="$MANIFEST" RUN_TSV="$RUN_TSV" DRY_RUN="$DRY_RUN" SHOW_ALL="$SHOW_ALL" \
EXAMPLE_DIR="$EXAMPLE_DIR" AC10_DIR="$AC10_DIR" \
python - <<'PY'
import os, sys, copy, yaml, json, hashlib, glob, re
sys.path.insert(0, os.environ["EXPT_RUNNER_DIR"])
import expt_runner

env = os.environ.get
MODE = env("MODE"); DELAYS = env("DELAYS")
MAX_RUNTIME_S = int(env("MAX_RUNTIME_S")); MAX_DATA_ID = int(env("MAX_DATA_ID"))
NUM_TRAINERS = env("NUM_TRAINERS") or ""
NUM_GPUS = env("NUM_GPUS") or ""
GPU_IDS = env("GPU_IDS") or ""
SEL_C = env("SEL_C") or ""; SEL_C_ASYNC = env("SEL_C_ASYNC") or ""; SEL_K = env("SEL_K") or ""
AGG_GOAL = env("AGG_GOAL") or ""; MIN_INIT = env("MIN_INIT_TRAINERS") or ""
MIN_INIT_FRAC = env("MIN_INIT_FRAC") or ""
PART = env("PARTITION_METHOD") or ""
VAR_THRESHOLD = env("VAR_THRESHOLD") or ""; MAX_ITER = env("MAX_ITER_PER_DATA_ID") or ""
SERVER_UPDATE_AUDIT = env("SERVER_UPDATE_AUDIT") or ""
POOL_SPLIT_HALF_AUDIT = env("POOL_SPLIT_HALF_AUDIT") or ""
LEARNING_RATE = env("LEARNING_RATE") or ""; PERTURBATION_COUNT = env("PERTURBATION_COUNT") or ""
PROBE_COMBINE = env("PROBE_COMBINE") or ""
VAR_STOPPING_POLICY = env("VAR_STOPPING_POLICY") or ""; AGG_RATE_TYPE = env("AGG_RATE_TYPE") or ""
DELAY_FACTOR = env("DELAY_FACTOR") or ""
DELAY_FLOOR = env("DELAY_FLOOR") or ""
STALL_ON = env("STALL_ON") or ""; LOSS_MIN_REL_DELTA = env("LOSS_MIN_REL_DELTA") or ""
TARGET_ACC = env("TARGET_ACC") or ""; CONVERGE_WINDOW = env("CONVERGE_WINDOW") or ""
SIM_WALL_CEILING_S = env("SIM_WALL_CEILING_S") or ""
STALL_WINDOW_S = env("STALL_WINDOW_S") or ""; STALL_MIN_DELTA = env("STALL_MIN_DELTA") or ""
# "was it passed on the command line?" (override -> green) for the defaulted flags
MODE_SET = env("MODE_SET") == "1"; DELAYS_SET = env("DELAYS_SET") == "1"
MAX_RUNTIME_S_SET = env("MAX_RUNTIME_S_SET") == "1"; MAX_DATA_ID_SET = env("MAX_DATA_ID_SET") == "1"
GPUS_VISIBLE = int(env("GPUS_VISIBLE") or "0")
LOGDIR = env("LOGDIR"); MANIFEST = env("MANIFEST")
DRY_RUN = env("DRY_RUN") == "1"; SHOW_ALL = env("SHOW_ALL") == "1"
delays_on = (DELAYS == "on")

# Settled per-baseline training-delay condition so operators stop re-typing
# --delays/--delay-divisor/--delay-floor every launch. CLI flags still win
# when explicitly passed. `factor` is validated at 7200s scale (simulate_
# fwdllm.md §A). fwdllm/fwdllm_it_oracular's `floor` re-derived 07-19 pm
# (FWDLLM_DESIGN.md §O, same 1.3x-over-observed-max-compute formula as
# fluxtune's 7.0->4.0): the old 11.0 predated the harness-overhead-removal fix
# and was never re-checked against post-fix compute (2.72s/3.18s max, floor >=
# 1.3*1.63*3.18=6.74s) -- pending a validation run to confirm 0 TIMING_OVERRUN.
# fwdllm_it_unaware reuses fwdllm's exact values, not a guess: FWDLLM_DESIGN.md
# §O derives `factor` from select_perturbation_using_jvp (C1) forward-pass-unit
# count -- fwdllm_it_unaware has C1 off (code default), same as fwdllm, unlike
# fluxtune (C1 on) -- and `floor` from *observed compute under that baseline's
# own concurrency* -- fwdllm_it_unaware is sync C=10, same regime fwdllm was
# profiled at. Both axes match fwdllm exactly, so its number transfers cleanly.
# The 5 async net-new baselines (fedbuff_round/it_*, felix_round/it) borrow
# fwdllm's number (operator decision, 07-24): C1 (JVP) is off for all of them,
# same as fwdllm, so per-update compute matches fwdllm, not fluxtune -- despite
# their C=30 async concurrency matching fluxtune's regime, not fwdllm's C=10
# sync one. No real-mode yaml exists for these to profile floor against
# independently (sim-only per BRIDGE_DESIGN.md decision #3), so this is the
# operator's considered choice, not a placeholder.
# CAVEAT (07-26): fwdllm's 7.0 and fluxtune's 4.0 floors were profiled when
# those two ran `uniform`; all 9 baselines are now niid alpha=1, which shifts
# per-client partition size and hence observed compute. Re-derive both against
# post-switch compute if TIMING_OVERRUN shows up.
BASELINE_DELAY_DEFAULTS = {
    "fluxtune":           {"delays": True, "factor": 0.48, "floor": 4.0},
    "fwdllm":             {"delays": True, "factor": 1.63, "floor": 7.0},
    "fwdllm_it_oracular": {"delays": True, "factor": 1.63, "floor": 7.0},
    "fwdllm_it_unaware":  {"delays": True, "factor": 1.63, "floor": 7.0},
    "fedbuff_round":      {"delays": True, "factor": 1.63, "floor": 7.0},
    "fedbuff_it_unaware": {"delays": True, "factor": 1.63, "floor": 7.0},
    "fedbuff_it_oracular": {"delays": True, "factor": 1.63, "floor": 7.0},
    "felix_round":        {"delays": True, "factor": 1.63, "floor": 7.0},
    "felix_it":           {"delays": True, "factor": 1.63, "floor": 7.0},
}


def resolve_delay_settings(run_key):
    """(delays_on, factor_str, floor_str) for `run_key`: explicit CLI wins,
    else this baseline's settled default, else the OFF/base-code fallback for
    a baseline with no entry (new/unregistered baseline)."""
    bl = BASELINE_DELAY_DEFAULTS.get(run_key, {})
    _on = delays_on if DELAYS_SET else bl.get("delays", delays_on)
    _factor = DELAY_FACTOR or (str(bl["factor"]) if "factor" in bl else "")
    _floor = DELAY_FLOOR or (str(bl["floor"]) if "floor" in bl else "")
    return _on, _factor, _floor

# Availability trace(s). Default syn_0 (Phase-1, 100% avail) when no --avail-trace,
# so patch() sets the mode EXPLICITLY on every baseline (rather than inheriting
# each yaml's `mode:`) — the printed "trace" row then matches what actually runs.
_trace_raw = [t for t in (env("TRACE_CSV") or "").replace(",", " ").split()]
trace_set = bool(_trace_raw)              # operator passed --avail-trace(s)?
traces = _trace_raw or ["syn_0"]          # Phase-1 default: 100% availability
multi_trace = len(traces) > 1

variants = {"real": 0, "sim": 1} if MODE == "both" else {MODE: (0 if MODE == "real" else 1)}

# Baseline internals (selector/optimizer/sync|async) from the shared catalog
# _metadata/baselines.yaml (merged at launch, not the per-run yaml). Surfaced in
# the review table to catch a mis-picked baseline pre-run. "?" if unreadable.
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
sim_profiles = {}        # baseline -> sim_charge_profile_path (provenance check)
checks = []


def patch(exp, run_key, variant, trace):
    h = exp["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = MAX_RUNTIME_S
    h["max_data_id_progress"] = MAX_DATA_ID
    # The yamls' fixed 10800s watchdog silently TRUNCATES a longer real leg, which
    # still reports the duration it asked for (§D-44). Keep it above the budget,
    # never lower it. Real only -- sim's wall cap is sim_wall_ceiling_s.
    if variant != "sim":
        try:
            _wd = float(h.get("max_experiment_runtime_s") or 0.0)
        except (TypeError, ValueError):
            _wd = 0.0
        h["max_experiment_runtime_s"] = int(max(_wd, float(MAX_RUNTIME_S) + 1800.0))
    # sim_wall_ceiling_s: REAL-wall-clock outer safety, sim mode only (max_runtime_s
    # is VIRTUAL/vclock seconds there -- see the flag's own help text). Only set when
    # the operator passes it; unset keeps the code default (max_runtime_s * 20).
    if SIM_WALL_CEILING_S and variant == "sim":
        h["sim_wall_ceiling_s"] = float(SIM_WALL_CEILING_S)
    # enable_training_delays: SAME on both sides of a pair (K-D8) -- resolve_delay_settings
    # is a pure function of run_key, so real/sim calls for one baseline always agree.
    _bl_delays_on, _bl_delay_factor, _bl_delay_floor = resolve_delay_settings(run_key)
    exp["trainer"]["enable_training_delays"] = _bl_delays_on
    # training_delay_factor (#12): DIVISOR on the registry 4-18s delay; runner.py
    # fans it to both roles. CLI wins; else this baseline's settled default.
    if _bl_delay_factor:
        exp["trainer"].setdefault("hyperparameters", {})
        exp["trainer"]["hyperparameters"]["training_delay_factor"] = float(_bl_delay_factor)
    # training_delay_floor_s (FWDLLM_DESIGN.md §O): floor on the RAW delay, before the divisor.
    if _bl_delay_floor:
        exp["trainer"].setdefault("hyperparameters", {})
        exp["trainer"]["hyperparameters"]["training_delay_floor_s"] = float(_bl_delay_floor)
    if PART:
        h["partition_method"] = PART
        exp["trainer"]["config_overrides"]["hyperparameters"]["partition_method"] = PART
        # dirichlet_alpha is cosmetic for path-style data (nothing trainer-side
        # reads it) but it names the run, so an alpha sweep would emit N runs all
        # tagged alpha1. Re-derive it from the group actually selected.
        _m = re.search(r"alpha=([0-9.]+)", PART)
        if _m:
            _a = float(_m.group(1))
            exp["trainer"].setdefault("dataset", {})["dirichlet_alpha"] = (
                int(_a) if _a.is_integer() else _a  # 100 -> alpha100, not alpha100p0
            )
    # Variance-cadence knobs (review-every-run). Only patched when explicitly set,
    # so an unset run keeps the code/trainer default (surfaced as "(D)" below).
    if VAR_THRESHOLD:
        h["var_threshold"] = float(VAR_THRESHOLD)
    # I-1 audit telemetry: opt-in per run, never on a leg that pairs into a floor (§D-45).
    if SERVER_UPDATE_AUDIT:
        h["server_update_audit"] = True
    # L1 pool-agreement audit: a second pass over the pool, so its own flag.
    if POOL_SPLIT_HALF_AUDIT:
        h["pool_split_half_audit"] = True
    # Divergence sweeps (FLUXTUNE_PROBE_PLAN.md §4b). Both default to the
    # inherited value, so an unset run is byte-identical to today.
    # learning_rate is read aggregator-side (FedSgdAggregator._prepare_round_state);
    # perturbation_count is P, read trainer-side (trainer/main.py).
    if LEARNING_RATE:
        h["learning_rate"] = float(LEARNING_RATE)
    if PERTURBATION_COUNT:
        exp["trainer"]["config_overrides"]["hyperparameters"]["perturbation_count"] = int(PERTURBATION_COUNT)
    # S-H: trainer-side, so it must go in the TRAINER copy -- the aggregator's
    # copy of probe knobs is never read (same trap as select_perturbation_using_jvp).
    if PROBE_COMBINE:
        exp["trainer"]["config_overrides"]["hyperparameters"]["probe_combine"] = PROBE_COMBINE
    if MAX_ITER:
        h["max_iterations_per_data_id"] = int(MAX_ITER)
    # Opt-2/Opt-3 ablation toggles (charter 4-run 2x2). Written into the per-run
    # config_overrides, which WIN over the baselines.yaml catalog at launch. A full
    # agg_rate_conf dict is written per type so the result is deterministic
    # regardless of merge depth (the felix `new` branch needs scale/a_exp/b_exp).
    if VAR_STOPPING_POLICY:
        h["var_stopping_policy"] = VAR_STOPPING_POLICY
    if AGG_RATE_TYPE:
        _opt = exp["aggregator"]["config_overrides"].setdefault("optimizer", {})
        _ok = _opt.setdefault("kwargs", {})
        if AGG_RATE_TYPE == "grad_aware":
            _ok["agg_rate_conf"] = {
                "type": "grad_aware", "base": "new", "align_gate": True,
                "align_floor": 0.0, "inverse_var": False, "var_ref": 0.3,
                "scale": 0.4, "a_exp": 0.25, "b_exp": 0.1,
            }
        else:  # new (FeLiX) or old
            _ok["agg_rate_conf"] = {
                "type": AGG_RATE_TYPE, "scale": 0.4, "a_exp": 0.25, "b_exp": 0.1,
            }
    if NUM_TRAINERS:
        exp["trainer"]["num_trainers"] = int(NUM_TRAINERS)
    if NUM_GPUS:
        exp["execution"]["num_gpus"] = int(NUM_GPUS)
    if GPU_IDS:
        _ids = [int(x) for x in GPU_IDS.split(",") if x.strip() != ""]
        exp["execution"]["gpu_ids"] = _ids
        if not NUM_GPUS:  # keep the displayed n_gpus honest with the actual pool size
            exp["execution"]["num_gpus"] = len(_ids)
    kwargs = exp["aggregator"]["config_overrides"]["selector"]["kwargs"]
    is_async = _BL_INTERNALS.get(run_key, {}).get("async") == "async"
    if SEL_C:
        kwargs["c"] = int(SEL_C)
    if SEL_C_ASYNC and is_async:
        kwargs["c"] = int(SEL_C_ASYNC)
    # K IS agg_goal. `selector.kwargs.k` is read by NOTHING in flame, so --k
    # (and the registry's `K`) silently no-opped; agg_goal is the single source
    # of truth the runner fans into hyperparameters.aggGoal + selector.kwargs
    # aggGoal/aggr_num (runner.py:674). Precedence: --agg-goal > --k > the
    # legacy "agg_goal follows --c" fallback.
    _goal = AGG_GOAL or SEL_K or (SEL_C if SEL_C else "")
    if _goal:
        exp["aggregator"]["agg_goal"] = int(_goal)
    # minInitialTrainers join barrier. DEFAULT = N: the gate fires at
    # ends_count >= threshold, so threshold < N admits a nondeterministic surplus
    # (98 vs 99, join-vs-poll race) -> divergent seeded first cohort; threshold=N
    # caps at the ceiling -> set-exact. CAVEAT: at N the first selection blocks
    # until all N register -- one silent no-show and the run never progresses;
    # --min-initial-frac <1 tolerates stragglers (reintroducing the race).
    if MIN_INIT:
        kwargs["minInitialTrainers"] = int(MIN_INIT)
    elif MIN_INIT_FRAC and NUM_TRAINERS:
        import math as _math
        kwargs["minInitialTrainers"] = max(1, _math.floor(float(MIN_INIT_FRAC) * int(NUM_TRAINERS)))
    elif NUM_TRAINERS:
        kwargs["minInitialTrainers"] = int(NUM_TRAINERS)
    elif SEL_C:
        kwargs["minInitialTrainers"] = int(SEL_C)
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
            if variant == "sim":
                sim_profiles[run_key] = h0.get("sim_charge_profile_path")
            kw0 = e0["aggregator"]["config_overrides"]["selector"]["kwargs"]
            _t_hp0 = (e0["trainer"].get("config_overrides", {})
                      .get("hyperparameters", {}))
            per_baseline.setdefault(run_key, {
                "c": kw0.get("c"),
                "agg_goal": e0["aggregator"].get("agg_goal"),
                "min_init": kw0.get("minInitialTrainers"),
                "n_trainers": e0["trainer"].get("num_trainers"),
                "n_gpus": e0.get("execution", {}).get("num_gpus"),
                "gpu_ids": e0.get("execution", {}).get("gpu_ids"),
                "partition": h0.get("partition_method"),
                "delays": e0["trainer"].get("enable_training_delays"),
                # H13 A/B knob: yaml-only, so condition_fp cannot see it (§F-18).
                # Captured per variant and cross-checked below.
                "jvp_eval_mode": {},
                "delay_factor": e0["trainer"].get("hyperparameters", {}).get("training_delay_factor"),
                "delay_floor": e0["trainer"].get("hyperparameters", {}).get("training_delay_floor_s"),
                # RESOLVED availability mode read back from the PATCHED cfg (what
                # actually launches), so the table can't show a stale default.
                "avail": e0["trainer"].get("availability", {}).get("mode"),
                "async": _BL_INTERNALS.get(run_key, {}).get("async") == "async",
                # baseline-distinguishing internals from the shared catalog
                "selector": _BL_INTERNALS.get(run_key, {}).get("selector", "?"),
                "optimizer": _BL_INTERNALS.get(run_key, {}).get("optimizer", "?"),
                "sync_async": _BL_INTERNALS.get(run_key, {}).get("async", "?"),
            })

            per_baseline[run_key]["jvp_eval_mode"][variant] = _t_hp0.get(
                "jvp_eval_mode", "ABSENT")

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
# delays/factor/floor now legitimately differ BY BASELINE, so fingerprint the
# resolved per-baseline tuples read back from the patched cfgs, not one CLI value.
_res_delays = sorted(
    f"{rk}:{b.get('delays')}/{b.get('delay_factor')}/{b.get('delay_floor')}"
    for rk, b in per_baseline.items()
)
_cond = {
    "N": NUM_TRAINERS or "yaml", "K": SEL_K or "yaml",
    "C_sync": SEL_C or "yaml", "C_async": SEL_C_ASYNC or SEL_C or "yaml",
    "partition": _res_parts, "trace": _res_traces,
    "delays": _res_delays,
    "target_acc": TARGET_ACC or "none",
    "stall_window_s": STALL_WINDOW_S or "off", "stall_min_delta": STALL_MIN_DELTA or "off",
    "stall_on": STALL_ON or "either", "loss_min_rel_delta": LOSS_MIN_REL_DELTA or "0.01",
    "converge_window": (CONVERGE_WINDOW or "20") if TARGET_ACC else "none",
    "max_runtime_s": MAX_RUNTIME_S, "max_data_id": MAX_DATA_ID,
    "sim_wall_ceiling_s": SIM_WALL_CEILING_S or "default",
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
# Resolved (delays, factor, floor) differs by baseline now -- show one value
# only if every baseline agrees, else defer to tier ② (same pattern as trace/avail).
_resolved_delay_vals = {(b.get("delays"), b.get("delay_factor"), b.get("delay_floor"))
                         for b in per_baseline.values()}
_delays_overridden = DELAYS_SET or bool(DELAY_FACTOR) or bool(DELAY_FLOOR)
if len(_resolved_delay_vals) == 1:
    _don, _dfac, _dflr = next(iter(_resolved_delay_vals))
    delays_row = scalar_row(
        "enable_training_delays", f"{'on' if _don else 'off'} (factor={_dfac}, floor={_dflr})",
        _delays_overridden, note="matched both sides of every pair — K-D8")
else:
    delays_row = scalar_row(
        "enable_training_delays", "MIXED — see ②", _delays_overridden,
        note="differs by baseline (BASELINE_DELAY_DEFAULTS); see tier ② for each")
tier1 = {"name": "① REVIEW EVERY RUN", "rows": [
    {"label": "condition_fp", "value": _cond_fp, "level": "set",
     "note": "TWO-NODE CHECK: same fp on both nodes ⇒ same condition "
             "(N/K/C/partition/trace/delays/caps). Differs ⇒ mistyped knob."},
    mode_row,
    {"label": "baselines", "value": " ".join(rk for rk, *_ in runs)},
    # The two similarly-named-but-DIFFERENT knobs, disambiguated + on their own rows:
    scalar_row("max_runtime_s", MAX_RUNTIME_S, MAX_RUNTIME_S_SET,
               note=("--max-runtime-s: REAL mode = wall-clock seconds; SIM mode = VIRTUAL/vclock "
                     "seconds, NOT wall -- see sim_wall_ceiling_s below for the real-wall cap")),
    scalar_row("max_data_id_progress", MAX_DATA_ID, MAX_DATA_ID_SET,
               note="STOP condition: stop when data_id reaches this (--max-data-id)"),
    scalar_row("trace", trace_val, trace_overridden,
               note=("resolved availability actually patched into each launched cfg; "
                     + ("100% availability (Phase 1)" if _resolved_avails == {"syn_0"}
                        else "NON-syn_0 — unavailability (Phase 2+)"))),
    delays_row,
    # var_threshold / max_iterations_per_data_id vary with data heterogeneity ->
    # review-every-run (warn when defaulted). NOTE: max_iters_per_data_id is the
    # FORCE-COMMIT cap and is NOT the same as max_data_id_progress (the stop) above.
    scalar_row("var_threshold", VAR_THRESHOLD if VAR_THRESHOLD else "unset",
               bool(VAR_THRESHOLD), review=True,
               note="variance-pass gate; varies w/ data heterogeneity. unset ⇒ code default"),
    scalar_row("max_iters_per_data_id", MAX_ITER if MAX_ITER else "unset",
               bool(MAX_ITER), review=True,
               note="FORCE-COMMIT cap, NOT the max_data_id_progress stop above. unset ⇒ code default"),
    scalar_row("var_stopping_policy", VAR_STOPPING_POLICY if VAR_STOPPING_POLICY else "default",
               bool(VAR_STOPPING_POLICY), review=True,
               note="Opt-2: off|fixed_cap|plateau. default ⇒ baselines.yaml (fluxtune=plateau)"),
    scalar_row("agg_rate_type", AGG_RATE_TYPE if AGG_RATE_TYPE else "default",
               bool(AGG_RATE_TYPE), review=True,
               note="Opt-3: grad_aware|new. default ⇒ baselines.yaml (fluxtune=grad_aware, new=FeLiX)"),
    # Convergence stop (EXPERIMENTS.md WS2): terminate when the last W data bins
    # are ALL >= target accuracy. When set, max_runtime_s/max_data_id become
    # SAFETY CAPS (a non-converging run -> DID_NOT_CONVERGE). unset ⇒ time/data-id bound only.
    scalar_row("target_acc", TARGET_ACC if TARGET_ACC else "unset",
               bool(TARGET_ACC), review=True,
               note=("stop when last %s bins all >= target. unset ⇒ time/data-id bound only"
                     % (CONVERGE_WINDOW or "20"))),
    # REAL-wall-clock outer safety for sim mode (max_runtime_s there is VIRTUAL
    # seconds). unset ⇒ code default = max_runtime_s(vclock-s) * 20, which for a
    # 48h vclock budget is a 40-DAY real failsafe -- effectively no bound. Always
    # reviewed (not just when MODE includes sim) since --mode both patches both.
    scalar_row("sim_wall_ceiling_s",
               SIM_WALL_CEILING_S if SIM_WALL_CEILING_S
               else f"unset ⇒ {int(MAX_RUNTIME_S) * 20}s (={MAX_RUNTIME_S}s*20)",
               bool(SIM_WALL_CEILING_S), review=True,
               note="REAL-wall-clock cap in sim mode (max_runtime_s there is vclock-seconds, not wall)"),
    # Stall guard: early-terminate a not-learning run before the wall ceiling.
    scalar_row("stall_guard",
               ((lambda _on, _h: {
                   "acc":    f"<{STALL_MIN_DELTA or '0.01'} acc in {_h}h→STALLED",
                   "loss":   f"<{float(LOSS_MIN_REL_DELTA or '0.01')*100:g}% loss in {_h}h→STALLED",
                   "either": f"<{STALL_MIN_DELTA or '0.01'} acc AND <{float(LOSS_MIN_REL_DELTA or '0.01')*100:g}% loss in {_h}h→STALLED",
                 }[_on])(STALL_ON or "either", int(float(STALL_WINDOW_S))//3600)
                if STALL_WINDOW_S and float(STALL_WINDOW_S) > 0 else "off"),
               bool(STALL_WINDOW_S), review=True,
               note=("early-terminate if no PROGRESS within stall_window_s on the armed signal "
                     "(acc/loss vs running-best). off ⇒ run to convergence or the wall ceiling")),
]}
tiers.append(tier1)

# ② per-baseline -- an ALIGNED TABLE (baselines = rows, knobs = columns). The
# renderer highlights any column whose value differs across the baselines (those
# are the ones to eyeball); columns identical across all 3 stay dim (expected).
tier2_cols = [
    ("sync_async", "mode"), ("selector", "selector"), ("optimizer", "optim"),
    ("c", "c"), ("agg_goal", "agg_goal (K)"),
    ("min_init", "minInit"), ("n_trainers", "n_trainers"),
    ("n_gpus", "n_gpus"), ("partition", "part"), ("avail", "avail"),
    ("delays", "delays (factor/floor)"),
]
if GPU_IDS:
    tier2_cols.append(("gpu_ids", "gpu_ids"))
overridden2 = []
if bool(SEL_C) or bool(SEL_C_ASYNC): overridden2.append("c")
if bool(AGG_GOAL) or bool(SEL_K) or bool(SEL_C): overridden2.append("agg_goal")
if bool(MIN_INIT):     overridden2.append("min_init")
if bool(NUM_TRAINERS): overridden2.append("n_trainers")
if bool(NUM_GPUS):     overridden2.append("n_gpus")
if bool(GPU_IDS):      overridden2.append("gpu_ids")
if bool(PART):         overridden2.append("partition")
if trace_set:          overridden2.append("avail")
if _delays_overridden: overridden2.append("delays")
rows2 = []
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    _don = b.get("delays")
    _gpu_ids_val = b.get("gpu_ids")
    rows2.append({"name": rk, "cells": {
        "sync_async": b.get("sync_async"), "selector": b.get("selector"),
        "optimizer": b.get("optimizer"),
        "c": b.get("c"), "agg_goal": b.get("agg_goal"),
        "min_init": b.get("min_init"), "n_trainers": b.get("n_trainers"),
        "n_gpus": b.get("n_gpus"), "partition": b.get("partition"),
        "avail": b.get("avail"),
        "delays": (f"{'on' if _don else 'off'} ({b.get('delay_factor')}/{b.get('delay_floor')})"
                   if _don else "off"),
        "gpu_ids": ",".join(str(g) for g in _gpu_ids_val) if _gpu_ids_val else "-",
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
# D matched across each pair (by construction), per baseline since it can now
# differ ACROSS baselines (BASELINE_DELAY_DEFAULTS).
if MODE == "both":
    for rk in (r[0] for r in runs):
        _don, _dfac, _dflr = resolve_delay_settings(rk)
        checks.append({"name": f"enable_training_delays matched across real/sim pair ({rk})",
                       "level": "ok",
                       "detail": f"D={'>0' if _don else '0'} both sides (factor={_dfac or 'base'}, floor={_dflr or '0.0'})"})
# H13 `jvp_eval_mode` (simulate_fwdllm.md §B): a yaml-only knob, so `condition_fp`
# cannot detect it going missing -- the exact §F-18 failure mode. It changes what is
# TRAINED, so a real/sim pair that disagrees grades two different experiments.
for rk in (r[0] for r in runs):
    _jv = per_baseline.get(rk, {}).get("jvp_eval_mode", {})
    _vals = set(_jv.values())
    if not _jv:
        pass
    elif "ABSENT" in _vals and len(_vals) > 1:
        checks.append({"name": f"jvp_eval_mode present on both legs ({rk})",
                       "level": "error",
                       "detail": f"declared on one leg only: {_jv}"})
    elif len(_vals) > 1:
        checks.append({"name": f"jvp_eval_mode matched across real/sim pair ({rk})",
                       "level": "error", "detail": f"MISMATCH: {_jv}"})
    elif _vals == {"ABSENT"}:
        checks.append({"name": f"jvp_eval_mode ({rk})", "level": "warn",
                       "detail": "not declared -> code default True (dropout off in the JVP)"})
    elif _vals == {False}:
        checks.append({"name": f"jvp_eval_mode ({rk})", "level": "warn",
                       "detail": "explicitly OFF -> dropout LIVE in the JVP, the H13 defect"})
    else:
        checks.append({"name": f"jvp_eval_mode ({rk})", "level": "ok",
                       "detail": f"{_vals.pop()} on every leg"})

def leg_jvp_eval_mode(run_dir):
    """Was this finished leg trained with dropout off inside the JVP? Only the
    trainer log records it (simulate_fwdllm.md §B.6)."""
    for lg in glob.glob(os.path.join(run_dir, "*trainers.log")):
        try:
            with open(lg, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if "jvp_eval_mode=" in line:
                        return "jvp_eval_mode=True" in line
                    if i > 50000:      # the knob logs at trainer init or never
                        break
        except OSError:
            continue
    return False


# sim charge profile provenance: every charged entry must have been profiled from
# a real run of THIS baseline. A shared family-wide profile silently mis-prices the
# vclock -- one constant was 1.08-2.75x each baseline's own real drain_tail, i.e.
# 0.6-3.4% of sim's clock, always making sim look slower (simulate_fwdllm.md §D-18).
for rk in (r[0] for r in runs):
    prof = sim_profiles.get(rk)
    if not prof:
        continue
    _repo_root = os.path.abspath(os.path.join(env("EXAMPLE_DIR", ""), "..", "..", "..", ".."))
    path = prof if os.path.isabs(prof) else os.path.join(_repo_root, prof)
    if not os.path.exists(path):
        checks.append({"name": f"sim charge profile exists ({rk})", "level": "error",
                       "detail": f"missing: {prof}"})
        continue
    try:
        _pf = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except Exception as _e:
        checks.append({"name": f"sim charge profile readable ({rk})", "level": "error",
                       "detail": f"{prof}: {_e}"})
        continue
    _foreign, _dates = [], set()
    for _lbl, _entries in _pf.items():
        for _pk, _e in (_entries or {}).items():
            if not _e.get("charge"):
                continue
            _dates.add(_e.get("profiled_at"))
            # `cross_baseline: true` is a DECLARED exemption, not an inferred one:
            # `redispatch_turnaround.weights` is deliberately shared because its
            # marginal is only valid under burst dispatch (§E). Declaring it keeps
            # the gate meaningful for everything that must be self-sourced.
            if _e.get("cross_baseline"):
                continue
            # `_<rk>_n` not a bare substring: "fwdllm" is a prefix of
            # "fwdllm_it_unaware", so a plain `in` would accept a sibling's profile.
            if not any(f"_{rk}_n" in str(s) for s in (_e.get("source_runs") or [])):
                _foreign.append(f"{_lbl}.{_pk}")
    if _foreign:
        checks.append({"name": f"sim charge profile provenance ({rk})", "level": "error",
                       "detail": f"{prof}: charged entries not profiled from a {rk} real run: "
                                 f"{', '.join(sorted(_foreign))}"})
    else:
        checks.append({"name": f"sim charge profile provenance ({rk})", "level": "ok",
                       "detail": f"{os.path.basename(path)} profiled {'/'.join(sorted(d for d in _dates if d))} from {rk} real"})

    # ...and from its CURRENT reals. A profile from an older training config
    # mis-prices the vclock, and sim selects work against that clock, so the leg
    # grades the PROFILE not the code -- worth 5.4% of cadence on one baseline
    # (§A.1 stage CH, §D-50). --force overrides.
    _src = set()
    for _lbl, _entries in _pf.items():
        for _pk, _e in (_entries or {}).items():
            if _e.get("charge") and not _e.get("cross_baseline"):
                _src.update(str(s) for s in (_e.get("source_runs") or []))
    _newest_src = max((s.split("run_")[-1][:15] for s in _src), default="")
    _expt = os.path.join(env("EXAMPLE_DIR", ""), "experiments")
    _reals = [d for d in glob.glob(os.path.join(_expt, f"run_*_{rk}_n*_real"))
              if os.path.basename(d).split("run_")[-1][:15] > _newest_src]
    # Only a same-config real can stale a profile; a deliberate flag-OFF control
    # landing later is not a reason to re-profile.
    _want = per_baseline.get(rk, {}).get("jvp_eval_mode", {}).get("sim", "ABSENT")
    _want = True if _want == "ABSENT" else _want
    _newer = sorted(os.path.basename(d) for d in _reals
                    if leg_jvp_eval_mode(d) == _want)
    if _newer and _newest_src:
        checks.append({"name": f"sim charge profile is CURRENT ({rk})", "level": "error",
                       "detail": f"{len(_newer)} real leg(s) newer than the profile "
                                 f"(newest source {_newest_src}): {', '.join(_newer[-2:])}. "
                                 f"Re-run profile_sim_charges.py, or --force if the newer "
                                 f"reals are a different config on purpose."})
    elif _newest_src:
        checks.append({"name": f"sim charge profile is CURRENT ({rk})", "level": "ok",
                       "detail": f"sourced from this baseline's newest real ({_newest_src})"})

# agg_goal <= c (more required than concurrently selected -> stall).
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    c, g = b.get("c"), b.get("agg_goal")
    if isinstance(c, int) and isinstance(g, int) and g > c:
        checks.append({"name": f"agg_goal <= c ({rk})", "level": "error",
                       "detail": f"agg_goal={g} > c={c} — selected trainers would be stranded"})
    else:
        checks.append({"name": f"agg_goal <= c ({rk})", "level": "ok", "detail": f"agg_goal={g} c={c}"})
# agg_goal MATCHES across baselines: identical batch size for a fair head-to-head;
# a mismatch is usually an unintended --c/--c-async fan. Warn, don't block.
_goals = {rk: per_baseline.get(rk, {}).get("agg_goal") for rk in (r[0] for r in runs)}
_gset = {g for g in _goals.values() if g is not None}
if len(_gset) > 1:
    checks.append({"name": "agg_goal matches across baselines", "level": "warn",
                   "detail": f"agg_goal differs: {_goals} — intended? (fair comparison expects one value)"})
elif _gset:
    checks.append({"name": "agg_goal matches across baselines", "level": "ok",
                   "detail": f"all baselines agg_goal={next(iter(_gset))}"})
# Availability liveness: BLOCK a full-participation sync barrier (agg_goal >=
# n_trainers) under a non-syn_0 trace — it can never assemble if any trainer is
# unavailable, so the barrier stalls to the wall cap (fwdllm_it_oracular / K-D20).
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
# (No k-vs-agg_goal check: `c` drives send-side concurrency, `k` is the RECV batch
# size, not a selection cap — so k < agg_goal is fine. Binding: c<=n, agg_goal<=c.)
# num_gpus <= visible.
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    g = b.get("n_gpus")
    if isinstance(g, int) and GPUS_VISIBLE and g > GPUS_VISIBLE:
        checks.append({"name": f"num_gpus <= gpus_visible ({rk})", "level": "error",
                       "detail": f"num_gpus={g} > visible={GPUS_VISIBLE}"})
# gpu_ids sanity: in-range (vs nvidia-smi count) and no duplicates. This is a
# coarse sanity check only -- an ordinal can be within range but still faulted
# (dropped from CUDA's own enumeration); runner.py's health preflight is the
# real gate, this just catches a mistyped list before launch.
for rk in (r[0] for r in runs):
    b = per_baseline.get(rk, {})
    ids = b.get("gpu_ids")
    if not ids:
        continue
    if len(set(ids)) != len(ids):
        checks.append({"name": f"gpu_ids duplicates ({rk})", "level": "error",
                       "detail": f"gpu_ids={ids} has duplicate ordinal(s)"})
    if GPUS_VISIBLE and any((i < 0 or i >= GPUS_VISIBLE) for i in ids):
        checks.append({"name": f"gpu_ids in range ({rk})", "level": "error",
                       "detail": f"gpu_ids={ids} has an ordinal outside [0, {GPUS_VISIBLE})"})
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
# render_and_gate: 0=ok, 2=blocking check. Anything else = the pre-flight step
# itself failed (bad YAML / spec bug) -> abort, don't launch an unvalidated config.
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

# Convergence stop: export so expt_launch arms the watcher (only when --target-acc
# passed; else runs stay governed by their caps). Window defaults to 20 bins.
if [ -n "$TARGET_ACC" ]; then
  export EXPT_TARGET_ACC="$TARGET_ACC"
  export EXPT_CONVERGE_WINDOW="${CONVERGE_WINDOW:-20}"
  [ -n "$STALL_WINDOW_S" ]  && export EXPT_STALL_WINDOW_S="$STALL_WINDOW_S"
  [ -n "$STALL_MIN_DELTA" ] && export EXPT_STALL_MIN_DELTA="$STALL_MIN_DELTA"
  [ -n "$STALL_ON" ]          && export EXPT_STALL_ON="$STALL_ON"
  [ -n "$LOSS_MIN_REL_DELTA" ] && export EXPT_LOSS_MIN_REL_DELTA="$LOSS_MIN_REL_DELTA"
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
  # Health verdict (COMPLETED / CRASH / NO_AGG_ROUNDS / WALL_CEILING) is the source
  # of truth for the summary, NOT the launcher exit code -- which is 0 even when the
  # aggregator subprocess crashed (run_experiment swallows the child's non-zero exit).
  expt_assert_run "$EXAMPLE_DIR" "$EXPT_LAST_MARKER" "$name"
  if [ "${EXPT_LAST_HEALTH:-}" = "CONVERGED" ] || [ "${EXPT_LAST_HEALTH:-}" = "STALLED" ] \
     || [ "${EXPT_LAST_HEALTH:-}" = "TIMEOUT_KILLED" ]; then
    # Watcher/backstop kills the process group -> rc is the SIGKILL code, NOT a
    # launcher failure; report the clean verdict. TIMEOUT_KILLED = the run never
    # self-stopped at its budget (a hang) and was force-killed (simulate_fwdllm.md §A).
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
