#!/bin/bash
# Run the three fwdllm smoke YAMLs (fwdllm, fwdllm_plus, fluxtune) one after
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
#   run_smoke_sequential.sh [--max-runtime-s 600] [--max-data-id 10] [--stop-on-fail]
#
#   --max-runtime-s  wall-clock cap in seconds for each run (default: 600 = 10 min)
#   --max-data-id    stop a run once data_id reaches this value (default: 10)
#   --stop-on-fail   abort the remaining runs as soon as one exits non-zero
#                    (default: run all three regardless, report at the end)
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-runtime-s) MAX_RUNTIME_S="$2"; shift 2 ;;
    --max-data-id)   MAX_DATA_ID="$2"; shift 2 ;;
    --stop-on-fail)  STOP_ON_FAIL=1; shift ;;
    *) echo "usage: $0 [--max-runtime-s SECONDS] [--max-data-id N] [--stop-on-fail]" >&2; exit 2 ;;
  esac
done

LOGDIR="$SCRIPT_DIR/smoke_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOGDIR"

# Patch hyperparameters.max_runtime_s / max_data_id_progress in a copy of the
# YAML rather than the original -- keeps the checked-in smoke configs stable
# while letting this script's caller pick the cap per invocation.
patch_yaml() {
  python - "$1" "$2" "$MAX_RUNTIME_S" "$MAX_DATA_ID" <<'PY'
import sys, yaml
src, dst, max_runtime_s, max_data_id = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = yaml.safe_load(open(src))
for exp in cfg.get("experiments", []):
    h = exp["aggregator"]["config_overrides"]["hyperparameters"]
    h["max_runtime_s"] = max_runtime_s
    h["max_data_id_progress"] = max_data_id
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
}

RUNS=(
  "fwdllm_n10_smoke:$SCRIPT_DIR/fwdllm_n10_smoke.yaml"
  "fwdllm_plus_n10_smoke:$SCRIPT_DIR/fwdllm_plus_n10_smoke.yaml"
  "fluxtune_n10_smoke:$SCRIPT_DIR/fluxtune_n10_smoke.yaml"
)

declare -A RESULT
declare -A DURATION_S

cd "$REPO_ROOT" || exit 1
echo "=== fwdllm smoke sequence: ${#RUNS[@]} runs, max_runtime_s=$MAX_RUNTIME_S max_data_id=$MAX_DATA_ID, logs in $LOGDIR ==="

for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  src_cfg="${entry#*:}"
  cfg="$LOGDIR/${name}.yaml"
  log="$LOGDIR/${name}.out"
  patch_yaml "$src_cfg" "$cfg"

  start_ts=$(date +%s)
  echo "[$(date '+%F %T')] START $name -> $cfg (log: $log)"
  python -m flame.launch.run_experiment "$cfg" --example-dir "$EXAMPLE_DIR" \
      < /dev/null > "$log" 2>&1
  rc=$?
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
