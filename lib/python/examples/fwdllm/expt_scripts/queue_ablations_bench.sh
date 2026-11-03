#!/usr/bin/env bash
# DistilBERT · AG News ablation runs (RUN_PLAN §3.2 runs 8, 10-22, 26) queued on the bench, legs in parallel.
# One bench_pool.py queue: each leg takes GPUS_PER_LEG GPUs, so floor(#GPUS / GPUS_PER_LEG) legs run at once; the rest wait.
# Stops: target 0.86 x2, plateau (stall), 1000-commit cap, TIMEOUT_H. Replayed 10-28: a15/a16 stop at 684/528, a08/a10 unaffected.
# Each arm = §2 FluxTune recipe (N 50, rho0 0.07, T_res 283, bs 8, full test every 4 commits) plus its change.
# Usage:  OUT=paper_ab_20261028 bash queue_ablations_bench.sh [arm ...]        (no args: every arm in ORDER)
#   GPUS=0,1,2,3,4,5,6,7  GPUS_PER_LEG=2  TIMEOUT_H=4  DRY=1 (render yamls + jobs file only)
# Launch: setsid nohup bash queue_ablations_bench.sh > $OUT/queue.log 2>&1 < /dev/null &    Stop: kill -TERM -- -<pgid>
# a07 stops on target only (no cap, stall, wall ceiling or rounds stop); run with TIMEOUT_H=0 (no bench timeout).
# Not included: run 27 (needs a yaml override whose key is not in the rendered yaml), run 9 (variance gate under async, never run), 23-25 (FT-N5 trace).
set -u
FW="$(cd "$(dirname "$0")/.." && pwd)"
EX="$(cd "$FW/.." && pwd)"
OUT="$(realpath -m "${OUT:-$FW/paper_ab_$(date +%Y%m%d)}")"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
GPUS_PER_LEG=${GPUS_PER_LEG:-2}
TIMEOUT_H=${TIMEOUT_H:-4}
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 FLAME_CONDA_ENV=test_fwdllm EXPT_GPU_ALLOW_PEER=1
log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die () { log "ABORT: $*"; exit 1; }
mkdir -p "$OUT"

COMMON=(--dataset agnews --partition-method niid_label_clients=100_alpha=1 --train-batch-size 8 --send-timeout-s 3600
        --delay-floor 4.0 --no-cos-ground-truth-audit)
SIM=(--mode sim --num-trainers 100 --phi-stop-threshold 1e9 --max-commits 1000 --final-full-eval --eval-every-n-commits 4
     --target-stop 0.86 --target-stop-window 2
     --phi-stop halt --budget-stop-frac 0 --saturation-stop --sat-stall-frac 0.003  # plateau stop: warm-up 450, horizon 150 (scale with cap)
     --sim-wall-ceiling-h 6 --max-runtime-s 1000000000 --force)
FWD=(--only fwdllm --delay-divisor 0.478 --learning-rate 0.01 --var-threshold 0.3)
FLUX=(--only fluxtune --delay-divisor 0.478 --fd-scale-invariant --probe-combine mean --perturbation-count 10
      --server-step-rule trust_ratio --commit-gate fixed --pool-target 50 --rho-schedule exp --rho0 0.07 --t-res 283
      --agg-rate-type uniform --var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt
      --c 30 --agg-goal 10)
# arm -> "<baseline>|<flags>|<yaml post-edit: sed script or empty>". Flags override the base via argparse's last-wins; the edits
# in arms without the dropped flag are explicit below (flags are removed from FLUX with drop()).
declare -A BASE FLAGS POST
drop () { # drop <flag> [<flag>...]: FLUX minus these flags and their values
  local out=() skip=0 a f
  for a in "${FLUX[@]}"; do
    if [ $skip = 1 ]; then skip=0; case $a in --*) ;; *) continue ;; esac; fi
    for f in "$@"; do [ "$a" = "$f" ] && { skip=1; continue 2; }; done
    out+=("$a")
  done
  echo "${out[*]}"
}
BASE[a08]=fluxtune; FLAGS[a08]="$(drop --server-step-rule --rho-schedule --rho0 --t-res) --server-step-rule raw_sgd --learning-rate 0.01"
BASE[a10]=fwdllm;   FLAGS[a10]="--only fwdllm --delay-divisor 0.478 --fd-scale-invariant --probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --commit-gate fixed --pool-target 50 --rho-schedule exp --rho0 0.07 --t-res 283 --var-stopping-policy off"
BASE[a11]=fedbuff_it_unaware; FLAGS[a11]="--only fedbuff_it_unaware --delay-divisor 0.478 --learning-rate 0.01 --var-threshold 0.3 --c-async 30"
BASE[a12]=felix_round;   FLAGS[a12]="--only felix_round --delay-divisor 0.478 --c-async 30"
BASE[a13]=fedbuff_round; FLAGS[a13]="--only fedbuff_round --delay-divisor 0.478 --c-async 30"
BASE[a15]=fluxtune; FLAGS[a15]="${FLUX[*]} --rho0 0.14 --t-res 71"
BASE[a16]=fluxtune; FLAGS[a16]="${FLUX[*]} --rho0 0.28 --t-res 18"
BASE[a14]=fluxtune; FLAGS[a14]="${FLUX[*]} --rho0 0.10 --t-res 139"
BASE[a17]=fluxtune; FLAGS[a17]="${FLUX[*]} --pool-target 10"
BASE[a18]=fluxtune; FLAGS[a18]="${FLUX[*]} --pool-target 20"
BASE[a19]=fluxtune; FLAGS[a19]="${FLUX[*]} --pool-target 90"
BASE[a20]=fluxtune; FLAGS[a20]="$(drop --commit-gate --pool-target) --commit-gate n_target"
BASE[a21]=fluxtune; FLAGS[a21]="$(drop --rho-schedule --t-res) --rho-schedule pool"
BASE[a22]=fluxtune; FLAGS[a22]="${FLUX[*]} --server-momentum 0.8"
# Run 7 (- E1): one cos-picked probe of P 10 vs the last commit, every bin; 2 passes, so fwdllm's divisor 4.78.
SIM_TARGET=(--mode sim --num-trainers 100 --phi-stop-threshold 1e9 --max-commits 1000000000 --final-full-eval --eval-every-n-commits 4
            --target-stop 0.86 --target-stop-window 2 --phi-stop off --budget-stop-frac 0
            --sim-wall-ceiling-h 1000000 --max-runtime-s 1000000000 --force)
declare -A TARGET_ONLY
BASE[a07]=fluxtune; FLAGS[a07]="$(drop --probe-combine --delay-divisor --phi-stop) --probe-combine select --perturbation-select cos_every_bin --delay-divisor 4.78 --phi-stop off"; TARGET_ONLY[a07]=1
POST[a07]='s/^\( *rounds:\) 50$/\1 1000000/'
BASE[a26]=fluxtune; FLAGS[a26]="$(drop --probe-combine) --probe-combine select"
# Pitch arms (_metadata/temp_fxtune_abl.md): SIM_LONG = no target / stall stop, per-commit server_update + cos(G,g) audit.
SIM_LONG=(--mode sim --num-trainers 100 --phi-stop-threshold 1e9 --final-full-eval --eval-every-n-commits 4
          --phi-stop halt --budget-stop-frac 0 --sim-wall-ceiling-h 8 --max-runtime-s 1000000000 --force
          --server-update-audit --cos-ground-truth-audit --cos-probe-batch-size 1024)
declare -A LONG
BASE[q1]=fluxtune; FLAGS[q1]="$(drop --rho-schedule --t-res) --rho-schedule pool --max-commits 1500 --cos-probe-every 50"; LONG[q1]=1
BASE[q2]=fluxtune; FLAGS[q2]="${FLUX[*]} --t-res 566 --max-commits 1500 --cos-probe-every 50"; LONG[q2]=1  # Phi_budget 4
BASE[q3]=fluxtune; FLAGS[q3]="${FLUX[*]} --t-res 849 --max-commits 2000 --cos-probe-every 50"; LONG[q3]=1  # Phi_budget 8
BASE[q4]=fwdllm;   FLAGS[q4]="${FWD[*]} --max-commits 700 --cos-probe-every 25"; LONG[q4]=1
BASE[q5a]=fluxtune; FLAGS[q5a]="${FLUX[*]} --pool-target 10 --max-commits 300 --cos-probe-every 10"; LONG[q5a]=1
BASE[q5b]=fluxtune; FLAGS[q5b]="${FLUX[*]} --pool-target 90 --max-commits 300 --cos-probe-every 10"; LONG[q5b]=1
# P1, P2, P3 (RUN_PLAN §3.2), then a13, a11, a12 last (baselines, never run before)
ORDER=(a08 a10 a15 a16 a17 a19 a21 a14 a18 a20 a22 a26 a13 a11 a12)
ARMS=("${@:-${ORDER[@]}}")

render () { # render <arm>: dry-run yaml -> $OUT/<arm>.yaml, ids renamed so legs do not collide
  local arm=$1 o src y sim=("${SIM[@]}")
  [ -n "${LONG[$arm]:-}" ] && sim=("${SIM_LONG[@]}")
  [ -n "${TARGET_ONLY[$arm]:-}" ] && sim=("${SIM_TARGET[@]}")
  o=$(cd "$EX" && bash "$FW/expt_scripts/run_sequential.sh" "${COMMON[@]}" "${sim[@]}" ${FLAGS[$arm]} --dry-run 2>&1)
  src=$(grep -oE 'generated cfgs in [^ ]+' <<<"$o" | awk '{print $4}')
  y=$(ls "$src"/*_sim.yaml 2>/dev/null | head -1)
  [ -f "$y" ] || { echo "$o" | tail -20; die "render $arm"; }
  sed -e "s#\(name\|id\): \([a-z_]*_agnews_n100_smoke_syn_0_sim\)#\1: ${arm}_\2#" "$y" > "$OUT/$arm.yaml"
  rm -rf "$src"
  if [ -n "${POST[$arm]:-}" ]; then
    sed -i -e "${POST[$arm]}" "$OUT/$arm.yaml"
  fi
}

: > "$OUT/jobs.tsv"
for a in "${ARMS[@]}"; do
  [ -n "${BASE[$a]:-}" ] || die "unknown arm $a (known: ${ORDER[*]})"
  render "$a"
  printf '%s\t%s\tsyn_0\t%s\tgpu_gb=4\tram_gb=12\tworkers=8\n' "$a" "${BASE[$a]}" "$OUT/$a.yaml" >> "$OUT/jobs.tsv"
  log "rendered $a (${BASE[$a]})"
done
[ "${DRY:-0}" = 1 ] && { log "DRY: jobs in $OUT/jobs.tsv"; exit 0; }

log "bench start: ${#ARMS[@]} arms, gpus=$GPUS, $GPUS_PER_LEG per leg, timeout ${TIMEOUT_H} h"
trap 'log stopping; kill -TERM $(jobs -p) 2>/dev/null; exit 130' INT TERM
(cd "$EX" && python scripts/bench_pool.py --jobs "$OUT/jobs.tsv" --out "$OUT/POOL" --gpus "$GPUS" \
  --gpus-per-leg "$GPUS_PER_LEG" --per-gpu 1 --timeout-s "$(python3 -c "print(int($TIMEOUT_H*3600) or 10**9)")" \
  > "$OUT/bench.log" 2>&1 < /dev/null) &
wait
log "bench exit (log $OUT/bench.log; runs under $OUT/POOL/<arm>/runs/)"
