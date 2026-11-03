#!/usr/bin/env bash
# DistilBERT · Yahoo paper cell (4th cell), jayne GPUs 0-1, fully queued:
#   1 real_fwdllm  2 real_flux    30-min real legs -> charge profiles {fwdllm,fluxtune}_yahoo_distilbert.yaml
#   3 screen       2 short fwdllm sim legs (lr/var combos A, B) on bench, 1 GPU each; pick_screen.py picks the winner
#   4 full_fwdllm  5 full_flux    paper legs on bench, 2 GPUs each, run until 0.65 x2 (then full-test eval); no wall or commit cap
# Model-suffixed profiles leave SmolLM's fluxtune_yahoo.yaml alone. Each stage writes .done_<stage>; a rerun skips done stages.
# Launch: setsid nohup bash pipeline.sh > pipeline.log 2>&1 < /dev/null &   Stop: kill -TERM -- -<pgid>
set -u
D="$(cd "$(dirname "$0")" && pwd)"
EX="$(cd "$D/../.." && pwd)"
FW="$EX/fwdllm"
PROF="$FW/sim_charge_profiles"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 FLAME_CONDA_ENV=test_fwdllm EXPT_GPU_ALLOW_PEER=1
GPUS=${GPUS:-0,1}
REAL_S=${REAL_S:-1800}
SCREEN_H=${SCREEN_H:-1.5}
SCREEN_A=${SCREEN_A:-"0.01 0.3"}   # lr var: AG News fwdllm values
SCREEN_B=${SCREEN_B:-"0.003 0.3"}
log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
done_ () { [ -e "$D/.done_$1" ]; }
mark () { touch "$D/.done_$1"; log "stage $1 done"; }
die () { log "ABORT: $*"; exit 1; }

COMMON=(--dataset yahoo --partition-method niid_label_clients=100_alpha=1 --train-batch-size 32 --send-timeout-s 3600
        --delay-floor 4.0 --no-cos-ground-truth-audit)
FWD=(--only fwdllm --delay-divisor 4.78)
FLUX=(--only fluxtune --delay-divisor 0.478 --fd-scale-invariant --probe-combine mean --perturbation-count 10
      --server-step-rule trust_ratio --commit-gate fixed --pool-target 50 --rho-schedule exp --rho0 0.07 --t-res 283
      --agg-rate-type uniform --var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt)
# sim legs: n 100, C 30 / K 10 (fluxtune), eval 10% resampled every 4 commits; screens stop at SCREEN_H, full legs on FULL
SIM=(--mode sim --num-trainers 100 --phi-stop-threshold 1e9
     --eval-max-samples 6000 --eval-resample --eval-every-n-commits 4 --max-runtime-s 1000000000 --force)

real_leg () {  # real_leg <label> <arm flags...>: 30-min real leg on its own broker, as charge_profile_20261025/run.sh
  local label=$1 port broker; shift
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
    "$port" "$D/mosquitto_${label}.log" > "$D/mosquitto_${label}.conf"
  /usr/sbin/mosquitto -c "$D/mosquitto_${label}.conf" >/dev/null 2>&1 &
  broker=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
  log "real leg $label gpus=$GPUS broker=$port"
  (cd "$EX" && FLAME_MQTT_BROKER="localhost:$port" FLAME_SLOT_GPUS="$GPUS" FLAME_RUN_TAG="dy_${label}" FLAME_RUN_LABEL="dy_${label}" \
    taskset -c 0-63 bash "$FW/expt_scripts/run_sequential.sh" --mode real "${COMMON[@]}" "$@" --num-trainers 30 \
    --eval-max-samples 1000 --eval-resample --eval-every-n-commits 4 --max-runtime-s "$REAL_S" --gpu-ids "$GPUS" --yes \
    < /dev/null > "$D/${label}.log" 2>&1)
  log "real leg $label exit=$?"
  kill "$broker" 2>/dev/null
}

profile () {  # profile <label> <seed profile> <out>: refresh a copy of the seed so its charge:/rationale: flags carry over
  local run
  run=$(ls -dt "$FW"/experiments/run_*_dy_"$1"_*_real 2>/dev/null | head -1)
  [ -n "$run" ] || die "no real run dir for dy_$1"
  grep -q '"vclock_charge"' "$run"/telemetry/aggregator_*.jsonl 2>/dev/null || die "$run has no vclock_charge events"
  cp "$PROF/$2" "$PROF/$3"
  (cd "$FW/expt_scripts" && python profile_sim_charges.py --real-run "$run" --out "$PROF/$3" --only-observed) || die "profile $3"
  log "profile $3 from $run"
}

render () {  # render <name> <profile> <flags...>: dry-run yaml -> $D/<name>.yaml, renamed and pointed at <profile>
  local name=$1 prof=$2 out src y; shift 2
  out=$(cd "$EX" && bash "$FW/expt_scripts/run_sequential.sh" "${COMMON[@]}" "$@" --dry-run 2>&1)
  src=$(grep -oE 'generated cfgs in [^ ]+' <<<"$out" | awk '{print $4}')
  y=$(ls "$src"/*_sim.yaml 2>/dev/null | head -1)
  [ -f "$y" ] || { echo "$out" | tail -20; die "render $name"; }
  sed -e "s#sim_charge_profile_path: .*#sim_charge_profile_path: $PROF/$prof#" \
      -e "s#\(name\|id\): \([a-z]*_yahoo_n100_smoke_syn_0_sim\)#\1: ${name}_\2#" "$y" > "$D/$name.yaml"
  rm -rf "$src"
  grep -q "sim_charge_profile_path: $PROF/$prof" "$D/$name.yaml" || die "render $name: profile path not set"
}

bench () {  # bench <tag> <gpus-per-leg> <timeout s> <job lines...>
  local tag=$1 gpl=$2 to=$3; shift 3
  printf '%s\n' "$@" > "$D/jobs_$tag.tsv"
  log "bench $tag start"
  (cd "$EX" && python scripts/bench_pool.py --jobs "$D/jobs_$tag.tsv" --out "$D/POOL_$tag" --gpus "$GPUS" \
    --gpus-per-leg "$gpl" --per-gpu 1 --timeout-s "$to" > "$D/bench_$tag.log" 2>&1 < /dev/null)
  log "bench $tag exit=$? (log bench_$tag.log)"
}
job () { printf '%s\t%s\tsyn_0\t%s\tgpu_gb=4\tram_gb=12\tworkers=8' "$1" "$2" "$D/$1.yaml"; }

stop () {  # trainers run in their own sessions, so also stop every process carrying a dy_ tag
  trap - INT TERM; log "stopping"
  local p tagged=""
  for p in $(pgrep -u "$(id -u)"); do
    tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -q '^FLAME_RUN_TAG=dy_' && tagged="$tagged $p"
  done
  kill -TERM $tagged $(jobs -p) 2>/dev/null; sleep 10; kill -KILL $tagged 2>/dev/null; exit 130
}
trap stop INT TERM

log "pipeline start gpus=$GPUS"
done_ real_fwdllm || { real_leg fwd_real "${FWD[@]}"; profile fwd_real fwdllm.yaml fwdllm_yahoo_distilbert.yaml; mark real_fwdllm; }
done_ real_flux || { real_leg flux_real "${FLUX[@]}" --c 20 --c-async 20 --agg-goal 10
                     profile flux_real fluxtune.yaml fluxtune_yahoo_distilbert.yaml; mark real_flux; }

if ! done_ screen; then
  read -r lrA varA <<<"$SCREEN_A"; read -r lrB varB <<<"$SCREEN_B"
  render scrA fwdllm_yahoo_distilbert.yaml "${FWD[@]}" "${SIM[@]}" --learning-rate "$lrA" --var-threshold "$varA" --sim-wall-ceiling-h "$SCREEN_H"
  render scrB fwdllm_yahoo_distilbert.yaml "${FWD[@]}" "${SIM[@]}" --learning-rate "$lrB" --var-threshold "$varB" --sim-wall-ceiling-h "$SCREEN_H"
  bench screen 1 "$(python3 -c "print(int(($SCREEN_H + 0.5) * 3600))")" "$(job scrA fwdllm)" "$(job scrB fwdllm)"
  win=$(python "$D/pick_screen.py" "$D/POOL_screen" scrA scrB) || die "screen pick"
  case $win in scrA) echo "$SCREEN_A" > "$D/winner.txt" ;; scrB) echo "$SCREEN_B" > "$D/winner.txt" ;; *) die "pick: $win" ;; esac
  log "screen winner $win: lr var = $(cat "$D/winner.txt")"
  mark screen
fi

FULL=(--target-stop 0.65 --target-stop-window 2 --final-full-eval --sim-wall-ceiling-h 720)  # 720 h: default is 2 h; target binds
TO=2592000  # 30 d: bench's own timeout only, so the target stop binds
if ! done_ full_fwdllm; then
  read -r lr var < "$D/winner.txt"
  render P_fwdllm fwdllm_yahoo_distilbert.yaml "${FWD[@]}" "${SIM[@]}" --learning-rate "$lr" --var-threshold "$var" "${FULL[@]}"
  bench full_fwdllm 2 "$TO" "$(job P_fwdllm fwdllm)"
  mark full_fwdllm
fi
if ! done_ full_flux; then
  render P_fluxtune fluxtune_yahoo_distilbert.yaml "${FLUX[@]}" "${SIM[@]}" --c 30 --agg-goal 10 "${FULL[@]}"
  bench full_flux 2 "$TO" "$(job P_fluxtune fluxtune)"
  mark full_flux
fi
log "pipeline done"
