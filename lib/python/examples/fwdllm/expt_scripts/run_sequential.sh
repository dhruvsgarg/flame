#!/bin/bash
# Run multiple fwdllm YAMLs (fwdllm, fwdllm_plus, fluxtune) one after
# another, in a single conda env, logging each run separately.
#
# Each YAML carries the Phase 13 auto-termination bar
# (MIGRATION_TO_LAUNCHER_FWDLLM.md): stop once data_id reaches a threshold,
# or after a wall-time cap, whichever comes first. This script overrides
# that wall-time cap (and the data_id cap) per invocation via
# --max-runtime-s/--max-data-id, generating a patched copy of each YAML
# rather than editing the originals.
#
# Usage (from anywhere):
#   run_sequential.sh [--max-runtime-s 600] [--max-data-id 10]
#       [--num-trainers N] [--num-gpus N] [--c C] [--k K] [--stop-on-fail]
#       [--only name1,name2]
#
#   --max-runtime-s  wall-clock cap in seconds for each run (default: 600 = 10 min)
#   --max-data-id    stop a run once data_id reaches this value (default: 10)
#   --num-trainers   override trainer.num_trainers (default: each YAML's own, 10)
#   --num-gpus       override execution.num_gpus (default: each YAML's own, 1).
#                     Each YAML's default of 1 GPU is sized for that default
#                     10-trainer count -- scaling --num-trainers up without
#                     also scaling this crams every trainer process onto one
#                     GPU and OOMs it almost immediately.
#   --c              override selector.kwargs.c + minInitialTrainers + agg_goal
#                     (agg_goal matches c so no selected trainer goes stranded,
#                     same rationale as MIGRATION_TO_LAUNCHER_FWDLLM.md Phase 14)
#   --k              override selector.kwargs.k
#   --stop-on-fail   abort the remaining runs as soon as one exits non-zero
#                    (default: run all three regardless, report at the end)
#   --only           comma-separated subset of baselines to execute, e.g.
#                     --only fwdllm_plus,fluxtune
#                     (default: all three -- fwdllm, fwdllm_plus, fluxtune)
#                     These are plain baseline names, independent of
#                     --num-trainers -- the "n10" in each source YAML's
#                     filename is just that file's own default trainer
#                     count, not part of the run's identity.
set -u

# --- robust conda activation (same pattern as scripts/debug_run.sh) ---
ENVNAME="${FLAME_CONDA_ENV:-aish_smoke_flame}"
CB=""
if command -v conda >/dev/null 2>&1; then
  CB="$(conda info --base 2>/dev/null)"
elif [ -n "${CONDA_EXE:-}" ]; then
  CB="$(dirname "$(dirname "$CONDA_EXE")")"
fi
if [ -z "$CB" ] || [ ! -f "$CB/etc/profile.d/conda.sh" ]; then
  for c in "$HOME/miniconda3" "/coc/scratch/${USER%??}/miniconda3" \
           "/coc/scratch/$USER/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && CB="$c" && break
  done
fi
if [ -z "$CB" ] || [ ! -f "$CB/etc/profile.d/conda.sh" ]; then
  echo "ERROR: conda not found. Activate '$ENVNAME' yourself or set CONDA_EXE." >&2; exit 1
fi
source "$CB/etc/profile.d/conda.sh"
conda activate "$ENVNAME" || { echo "ERROR: 'conda activate $ENVNAME' failed" >&2; exit 1; }
echo "conda: base=$CB env=$ENVNAME python=$(which python)"

# repo paths (portable across nodes/checkouts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # .../examples/fwdllm
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../../.." && pwd)"       # flame/

# defaults
MAX_RUNTIME_S=600   # 10 minutes
MAX_DATA_ID=10
STOP_ON_FAIL=0
NUM_TRAINERS=""   # empty = leave each YAML's own value
NUM_GPUS=""       # empty = leave each YAML's own value
SEL_C=""
SEL_K=""
ONLY=""           # empty = run all three

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-runtime-s) MAX_RUNTIME_S="$2"; shift 2 ;;
    --max-data-id)   MAX_DATA_ID="$2"; shift 2 ;;
    --num-trainers)  NUM_TRAINERS="$2"; shift 2 ;;
    --num-gpus)      NUM_GPUS="$2"; shift 2 ;;
    --c)             SEL_C="$2"; shift 2 ;;
    --k)             SEL_K="$2"; shift 2 ;;
    --stop-on-fail)  STOP_ON_FAIL=1; shift ;;
    --only)          ONLY="$2"; shift 2 ;;
    *) echo "usage: $0 [--max-runtime-s SECONDS] [--max-data-id N] [--num-trainers N] [--num-gpus N] [--c C] [--k K] [--stop-on-fail] [--only name1,name2]" >&2; exit 2 ;;
  esac
done

LOGDIR="$SCRIPT_DIR/smoke_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOGDIR"

# Patch hyperparameters.max_runtime_s / max_data_id_progress, and optionally
# num_trainers / selector c+k+minInitialTrainers+agg_goal, in a copy of the
# YAML rather than the original -- keeps the checked-in smoke configs stable
# while letting this script's caller pick the scale per invocation.
# run_key is the plain baseline name (e.g. "fwdllm_plus", from RUNS/--only
# below), NOT the source YAML's own "n10"-suffixed name -- it's the basis
# for exp["name"], so the run directory it produces is never stale/wrong
# regardless of what naming convention the source YAML file happens to use.
patch_yaml() {
  python - "$1" "$2" "$3" "$MAX_RUNTIME_S" "$MAX_DATA_ID" "$NUM_TRAINERS" "$NUM_GPUS" "$SEL_C" "$SEL_K" <<'PY'
import sys, yaml
src, dst, run_key, max_runtime_s, max_data_id, num_trainers, num_gpus, sel_c, sel_k = sys.argv[1:10]
cfg = yaml.safe_load(open(src))
for exp in cfg.get("experiments", []):
    h = exp["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = int(max_runtime_s)
    h["max_data_id_progress"] = int(max_data_id)
    if num_trainers:
        exp["trainer"]["num_trainers"] = int(num_trainers)
        # exp["name"] feeds the run directory name (run_<ts>_<name>).
        # Derive it from run_key (this run's plain baseline identity) and
        # the actual trainer count, instead of copying the source YAML's
        # own checked-in name verbatim -- that name encodes only that
        # file's own default trainer count (10), so left alone, a
        # 100-trainer run's directory would stay misleadingly named
        # "..._n10_smoke".
        exp["name"] = f"{run_key}_n{num_trainers}_smoke"
    if num_gpus:
        exp["execution"]["num_gpus"] = int(num_gpus)
    kwargs = exp["aggregator"]["config_overrides"]["selector"]["kwargs"]
    if sel_c:
        kwargs["c"] = int(sel_c)
        kwargs["minInitialTrainers"] = int(num_trainers) if num_trainers else int(sel_c)
        # agg_goal matches c so no selected trainer goes uncounted/stranded
        # (MIGRATION_TO_LAUNCHER_FWDLLM.md Phase 14).
        exp["aggregator"]["agg_goal"] = int(sel_c)
    if sel_k:
        kwargs["k"] = int(sel_k)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
}

# Keys are plain baseline names -- independent of --num-trainers and of
# whatever scale is baked into each source YAML's own filename/checked-in
# default. The mapping to the actual YAML file lives only here.
ALL_RUNS=(
  "fwdllm:$SCRIPT_DIR/fwdllm_n10_smoke.yaml"
  "fwdllm_plus:$SCRIPT_DIR/fwdllm_plus_n10_smoke.yaml"
  "fluxtune:$SCRIPT_DIR/fluxtune_n10_smoke.yaml"
)

if [ -n "$ONLY" ]; then
  RUNS=()
  IFS=',' read -ra ONLY_NAMES <<< "$ONLY"
  for want in "${ONLY_NAMES[@]}"; do
    found=0
    for entry in "${ALL_RUNS[@]}"; do
      if [ "${entry%%:*}" = "$want" ]; then
        RUNS+=("$entry")
        found=1
        break
      fi
    done
    if [ "$found" = "0" ]; then
      echo "ERROR: --only name '$want' not recognized. Valid names: ${ALL_RUNS[*]%%:*}" >&2
      exit 2
    fi
  done
else
  RUNS=("${ALL_RUNS[@]}")
fi

declare -A RESULT
declare -A DURATION_S

CHILD_PID=""
cleanup() {
  echo ""
  echo "Interrupted. Killing child (PID=${CHILD_PID:-none})..."
  [ -n "$CHILD_PID" ] && kill -- -"$CHILD_PID" 2>/dev/null
  exit 130
}
trap cleanup INT TERM

cd "$REPO_ROOT" || exit 1
echo "=== fwdllm sequential run: ${#RUNS[@]} runs (${RUNS[*]%%:*}), max_runtime_s=$MAX_RUNTIME_S max_data_id=$MAX_DATA_ID num_trainers=${NUM_TRAINERS:-<yaml default>} num_gpus=${NUM_GPUS:-<yaml default>} c=${SEL_C:-<yaml default>} k=${SEL_K:-<yaml default>}, logs in $LOGDIR ==="

for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  src_cfg="${entry#*:}"
  cfg="$LOGDIR/${name}.yaml"
  log="$LOGDIR/${name}.out"
  patch_yaml "$src_cfg" "$cfg" "$name"

  start_ts=$(date +%s)
  python -m flame.launch.run_experiment "$cfg" --example-dir "$EXAMPLE_DIR" \
      < /dev/null > "$log" 2>&1 &
  CHILD_PID=$!
  echo "[$(date '+%F %T')] START $name (PID=$CHILD_PID) -> $cfg (log: $log)"
  echo "  (to kill: kill -9 $CHILD_PID   or Ctrl+C)"
  wait "$CHILD_PID"
  rc=$?
  CHILD_PID=""
  end_ts=$(date +%s)
  DURATION_S[$name]=$((end_ts - start_ts))
  if [ $rc -eq 0 ]; then
    RESULT[$name]="PASS"
  else
    RESULT[$name]="FAIL(exit=$rc)"
  fi
  echo "[$(date '+%F %T')] DONE  $name -> ${RESULT[$name]} (${DURATION_S[$name]}s)"

  if [ $rc -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    echo "--stop-on-fail set; aborting remaining runs."
    break
  fi
done

echo ""
echo "=== Summary ==="
for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  printf "  %-25s %-15s %ss\n" "$name" "${RESULT[$name]:-SKIPPED}" "${DURATION_S[$name]:-0}"
done
echo "Logs: $LOGDIR"
echo "Run dirs: $EXAMPLE_DIR/experiments/run_*"
