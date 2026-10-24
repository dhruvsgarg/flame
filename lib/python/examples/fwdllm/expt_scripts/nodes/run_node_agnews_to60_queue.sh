#!/usr/bin/env bash
# Preview of every DistilBERT · AG News run in RUN_PLAN §3 (P1, C1-C3, C4 syn_0, R1/R2), each stopped at TARGET acc (0.60).
# Purpose: see every ablation figure's shape cheaply; not scored runs (TTA at 0.86 is DNF by design).
# Recipe = RUN_PLAN §2 distilbert_agnews cell. Skipped: G0 (real legs), update_utility/phi/Adam arms (need code),
# mobiperf_3st_50 (FT-N5).
# Lanes: one shared queue, drained by LANE_GPUS lanes in parallel (jayne: 2 × 4 A40, each lane on its GPUs' NUMA node).
# Each lane = taskset cores + private mosquitto + FLAME_RUN_TAG (sweeps stay in-lane) + FLAME_RUN_LABEL (run-dir lookup).
#   tmux new-window -t dg_flame -n ag60 "bash nodes/run_node_agnews_to60_queue.sh 2>&1 | tee ../ag60_queue_\$(date +%Y%m%d_%H%M).log"
# Env: TARGET (0.60), POOL_N (50), ARMS="A0 A6 ..." runs only those arms (labels below), START (queue index to begin at).
# ARM_TIMEOUT_MIN (60): clean wall-cap stop per arm (final eval kept); +15 min hard kill if it hangs; the lane then moves on.
# NODE_DRY_RUN=1 checks every arm's flags in seconds without waiting or launching (one lane).
set -u
. "$(dirname "$0")/_node_lib.sh"
export EXPT_GPU_ALLOW_PEER=1  # the launcher's check reads every GPU; the other lane sits outside GPUS
# Starvation predicates target controller arms; a fixed pool and FwdLLM's var gate can trip them falsely.
export NODE_WATCH_ARGS="--trips-floor 0 --i-floor-frac 1.01"

LANE_GPUS=(0,1,2,3 4,5,6,7)
LANE_CPUS=(0-31,64-95 32-63,96-127)  # NUMA node of each lane's GPUs (nvidia-smi topo -m)
TARGET=${TARGET:-0.60}
ARM_TIMEOUT_MIN=${ARM_TIMEOUT_MIN:-60}
export NODE_TIMEOUT_S=$(( ARM_TIMEOUT_MIN * 60 + 900 ))
export NODE_ABORT_ON_FAULT=0  # a crashed arm is logged `failed`; the lane moves on
N=${POOL_N:-50}
# Window 1: agg_eval data_ids step by the eval stride, and converge_watch.py only counts contiguous ids.
COMMON=(--mode sim --yes --clean --force --num-trainers 100
        --dataset agnews --partition-method niid_label_clients=100_alpha=1 --adapter-reduction-factor 16
        --train-batch-size 8 --server-update-audit --no-cos-ground-truth-audit --eval-every-n-commits 4
        --saturation-stop --sat-stall-frac 0.003 --max-commits 1000 --final-full-eval
        --max-runtime-s 432000 --sim-wall-ceiling-s "$(( ARM_TIMEOUT_MIN * 60 ))" --target-acc "$TARGET" --converge-window 1)

E1=(--probe-combine mean --perturbation-count 10)
E2=(--server-step-rule trust_ratio --rho-schedule exp --rho0 0.07 --t-res 283
    --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt)
E3=(--commit-gate fixed --pool-target "$N" --var-stopping-policy off)

FWD=(--only fwdllm "${COMMON[@]}")
# §2 recipe minus pool size, schedule and weights, which the arms set.
FLUX=(--only fluxtune "${COMMON[@]}" --agg-goal 10 --c 30 "${E1[@]}"
      --server-step-rule trust_ratio --rho0 0.07 --commit-gate fixed
      --var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt)
RECIPE=(--rho-schedule exp --t-res 283 --agg-rate-type uniform)

declare -a ORDER=(); declare -A DESC=() ARGS=()
arm () {  # arm <label> <description> [launcher flags...] -- queue order = priority order
  local label="$1"; shift
  if [ -n "${ARMS:-}" ] && [[ " $ARMS " != *" $label "* ]]; then return 0; fi
  ORDER+=("$label"); DESC[$label]="$1"; shift; ARGS[$label]="$(printf '%q ' "$@")"
}

# P1 / C1 endpoints
arm A0     "FwdLLM"                          "${FWD[@]}"
arm A6     "FluxTune (N $N)"                 "${FLUX[@]}" --pool-target "$N" "${RECIPE[@]}"
# P2 · C1 singles
arm A1     "FwdLLM + E1 mean-of-10"          "${FWD[@]}" "${E1[@]}"
arm A2     "FwdLLM + E2 trust ratio"         "${FWD[@]}" "${E2[@]}"
arm A3     "FwdLLM + E3 fixed pool"          "${FWD[@]}" "${E3[@]}"
arm A4     "FedBuff-IT (S1 async)"           --only fedbuff_it_unaware "${COMMON[@]}"
# P5 · C2 ladder middle rungs (A1 is rung 1); P2 pairs
arm A12    "FwdLLM + E1 + E2"                "${FWD[@]}" "${E1[@]}" "${E2[@]}"
arm A123   "FwdLLM + E1 + E2 + E3 (sync)"    "${FWD[@]}" "${E1[@]}" "${E2[@]}" "${E3[@]}"
arm A13    "FwdLLM + E1 + E3"                "${FWD[@]}" "${E1[@]}" "${E3[@]}"
arm A23    "FwdLLM + E2 + E3"                "${FWD[@]}" "${E2[@]}" "${E3[@]}"
arm A14    "FedBuff-IT + E1 (S1 + mean)"     --only fedbuff_it_unaware "${COMMON[@]}" "${E1[@]}"
# P6 · C3 agg_criteria
arm Cnt    "n_target pool"                   "${FLUX[@]}" "${RECIPE[@]}" --commit-gate n_target
arm CN10   "fixed N 10"                      "${FLUX[@]}" --pool-target 10 "${RECIPE[@]}"
arm CN20   "fixed N 20"                      "${FLUX[@]}" --pool-target 20 "${RECIPE[@]}"
arm CN90   "fixed N 90"                      "${FLUX[@]}" --pool-target 90 "${RECIPE[@]}"
# P6 · C3 step_size (rm a 0.11: Φ 1.98 at commit 1,000)
arm Spool  "const ρ (pool)"                  "${FLUX[@]}" --pool-target "$N" --rho-schedule pool --agg-rate-type uniform
arm Srm    "ρ0·t^-0.11 (rm)"                 "${FLUX[@]}" --pool-target "$N" --rho-schedule rm --rho-exp 0.11 --agg-rate-type uniform
arm Smom   "server momentum 0.5"             "${FLUX[@]}" --pool-target "$N" "${RECIPE[@]}" --server-momentum 0.5
# P6 · C3 agg_weights
arm Wnew   "staleness ω (Felix new)"         "${FLUX[@]}" --pool-target "$N" --rho-schedule exp --t-res 283 --agg-rate-type new
arm Wga    "grad_aware ω"                    "${FLUX[@]}" --pool-target "$N" --rho-schedule exp --t-res 283 --agg-rate-type grad_aware
# P7 · C4 syn_0 row (mobiperf_3st_50 blocked on FT-N5)
arm C4fb   "FedBuff(P) fedbuff_round"        --only fedbuff_round "${COMMON[@]}"
arm C4fx   "Felix(P) felix_round"            --only felix_round "${COMMON[@]}"
# R1/R2 ρ_max (N 90, as P0)
arm R1     "ρ0 0.10, T_res 139, N 90"        "${FLUX[@]}" --pool-target 90 --rho0 0.10 --rho-schedule exp --t-res 139 --agg-rate-type uniform
arm R2     "ρ0 0.14, T_res 71, N 90"         "${FLUX[@]}" --pool-target 90 --rho0 0.14 --rho-schedule exp --t-res 71 --agg-rate-type uniform

wait_gpus_free () {  # no compute process and < 1 GB used on every GPU in GPUS
  [ "${NODE_DRY_RUN:-0}" = "1" ] && return 0
  echo "=== waiting for GPUs $GPUS to be free  $(date -Is) ==="
  until [ -z "$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader -i "$GPUS")" ] \
        && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" \
           | awk '$1 > 1000 {busy=1} END {exit busy}'; do sleep 30; done
}

arm_status () {  # failed (< 5 commits) | reached | timeout | stopped (stall rule, watchdog, hard kill)
  local run="$1"
  [ "${2:-0}" != "0" ] && { echo failed; return; }
  grep -qs '\[SIM_WALL_CEILING\]' "$run"/*aggregator.log && { echo timeout; return; }
  grep -hos '"test-accuracy": [0-9.]*' "$run"/telemetry/aggregator_*.jsonl \
    | awk -v t="$TARGET" '$2 >= t {r = 1} END {print r ? "reached" : "stopped"}'
}

next_arm () {  # pop the next queue index (shared by every lane)
  local i
  exec 8>"$QDIR/queue.lock"; flock 8
  i=$(cat "$QDIR/next"); echo $((i + 1)) > "$QDIR/next"
  flock -u 8; exec 8>&-
  echo "$i"
}

lane_stop () {  # TERM every process carrying this lane's run tag, then the broker
  local p
  for p in $(pgrep -u "$(id -u)"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "FLAME_RUN_TAG=$FLAME_RUN_TAG" \
      && [ "$p" != "$$" ] && kill -TERM "$p" 2>/dev/null
  done
  [ -n "${BROKER_PID:-}" ] && kill "$BROKER_PID" 2>/dev/null
  exit 130
}

lane () {  # worker: drain the queue on lane $LANE
  GPUS=${LANE_GPUS[$LANE]}
  export FLAME_SLOT_GPUS=$GPUS  # else the aggregator takes an idle-looking GPU outside --gpu-ids (runner.py:322)
  export FLAME_RUN_TAG="${QDIR##*/}_L$LANE"
  BROKER_PID=""
  if [ "${NODE_DRY_RUN:-0}" != "1" ]; then
    local port; port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
    printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
      "$port" "$QDIR/mosquitto_L$LANE.log" > "$QDIR/mosquitto_L$LANE.conf"
    /usr/sbin/mosquitto -c "$QDIR/mosquitto_L$LANE.conf" >/dev/null 2>&1 &
    BROKER_PID=$!
    for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || { echo "!!! [L$LANE] private broker failed on $port" >&2; exit 5; }
    export FLAME_MQTT_BROKER="localhost:$port"
  fi
  trap lane_stop INT TERM
  trap '[ -n "$BROKER_PID" ] && kill "$BROKER_PID" 2>/dev/null' EXIT
  echo "=== [L$LANE] gpus=$GPUS cpus=$(taskset -cp $$ | awk '{print $NF}') broker=${FLAME_MQTT_BROKER:-none} tag=$FLAME_RUN_TAG ==="

  local i label args
  while i=$(next_arm); [ "$i" -lt "${#ORDER[@]}" ]; do
    label=${ORDER[$i]}
    export FLAME_RUN_LABEL="ag60_$label"  # run dir = run_<ts>_ag60_<label>_<exp>
    eval "args=(${ARGS[$label]})"
    wait_gpus_free
    # Space launches >= 3 s apart: run_sequential names its smoke_logs/ dir by the second.
    ( flock 7; sleep 3 ) 7>"$QDIR/launch.lock"
    node_run "ag60 L$LANE" "$label · ${DESC[$label]}" "${args[@]}" --gpu-ids "$GPUS"; local rc=$?
    if [ "${NODE_DRY_RUN:-0}" != "1" ]; then
      local run; run="$(ls -1dt "$RUNS"/run_*_ag60_"$label"_* | head -1)"
      printf '%s\t%s\tL%s\t%s\t%s\n' "$label" "${DESC[$label]}" "$LANE" "$(basename "$run")" "$(arm_status "$run" "$rc")" \
        >> "$QDIR/runs.tsv"
    fi
  done
  echo "=== [L$LANE] queue empty $(date -Is) ==="
}

if [ -n "${LANE:-}" ]; then lane; exit; fi

# Coordinator: one queue dir, one worker per lane (its own bash, so node_run's set -m groups work as before).
QDIR="$EXPT_DIR/ag60_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$QDIR"; echo "${START:-0}" > "$QDIR/next"
export QDIR
nl=${#LANE_GPUS[@]}; [ "${NODE_DRY_RUN:-0}" = "1" ] && nl=1
echo "=== [ag60] ${#ORDER[@]} arms from index ${START:-0} on $nl lane(s), target $TARGET, timeout ${ARM_TIMEOUT_MIN} min; queue dir $QDIR  $(date -Is) ==="
pids=()
for ((l = 0; l < nl; l++)); do
  LANE=$l taskset -c "${LANE_CPUS[$l]}" bash "$0" > >(sed -u "s/^/[L$l] /") 2>&1 &
  pids+=($!)
done
trap 'kill -TERM "${pids[@]}" 2>/dev/null; wait; exit 130' INT TERM
wait "${pids[@]}"
echo "=== [ag60] done $(date -Is); arm -> run dir: $QDIR/runs.tsv ==="
