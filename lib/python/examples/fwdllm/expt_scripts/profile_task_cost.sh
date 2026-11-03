#!/usr/bin/env bash
# Per-task A40 GPU cost for the D divisor (paper_expts_fluxtune/D_CALIBRATION.md): one fluxtune trainer alone on one GPU,
# sim mode (no sleeps), trainer-side recipe settings (P 10, mean, bs, model, LoRA), eval capped so it can't share the GPU's time.
# Lanes: DistilBERT bs 8 / bs 32 (anchor), RoBERTa-L bs 32, SmolLM2-1.7B bs 32. Writes OUT/summary.tsv. One Ctrl+C stops all.
# Usage: [GPUS="4 5 6 7"] [WALL_S=600] [OUT=dir] profile_task_cost.sh [--dry-run]
set -u
EX="$(cd "$(dirname "$0")/.." && pwd)"
B="${OUT:-$EX/task_cost_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$B"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 FLAME_CONDA_ENV=test_fwdllm
DRY=${1:-}
WALL_S=${WALL_S:-600}
read -r -a GPUS <<< "${GPUS:-4 5 6 7}"

ONE=(--only fluxtune --mode sim --num-trainers 1 --c 1 --c-async 1 --agg-goal 1 --min-initial-trainers 1
     --fd-scale-invariant --probe-combine mean --perturbation-count 10
     --eval-every-n-commits 1000000 --eval-max-samples 100
     --max-runtime-s 1000000000 --sim-wall-ceiling-s "$WALL_S" --allow-stale-profile --no-cos-ground-truth-audit
     --force)  # the wall-budget preflight wants the vclock cap to bind; here the wall ceiling ends each leg by design
AGNEWS=(--dataset agnews --partition-method niid_label_clients=100_alpha=1)
YELP=(--dataset yelp-p --partition-method niid_label_clients=100_alpha=1 --model-type roberta-large --peft-method lora --lora-r 6)
YAHOO=(--dataset yahoo --partition-method niid_label_clients=100_alpha=1 --model-type llama
       --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 --adapter-reduction-factor 16)

lane () {  # lane <name> <gpu> <cpus> <flags...>
  local name=$1 gpu=$2 cpus=$3 port; shift 3
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
    "$port" "$B/mosquitto_$name.log" > "$B/mosquitto_$name.conf"
  /usr/sbin/mosquitto -c "$B/mosquitto_$name.conf" >/dev/null 2>&1 &
  local broker=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
  echo "=== $name gpu=$gpu cpus=$cpus broker=$port $(date -Is) ==="
  FLAME_MQTT_BROKER="localhost:$port" FLAME_SLOT_GPUS="$gpu" FLAME_RUN_TAG="ptc_$name" FLAME_RUN_LABEL="ptc_$name" \
    taskset -c "$cpus" bash "$EX/expt_scripts/run_sequential.sh" "${ONE[@]}" "$@" --gpu-ids "$gpu" --yes $DRY \
    < /dev/null > "$B/$name.log" 2>&1
  echo "=== $name exit=$? $(date -Is) ==="
  kill "$broker" 2>/dev/null
}

stop () {  # trainers run in their own sessions, so also stop every process carrying a lane tag
  trap - INT TERM; echo "stopping"
  local p tagged=""
  for p in $(pgrep -u "$(id -u)"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^FLAME_RUN_TAG=ptc_' && tagged="$tagged $p"
  done
  kill -TERM $tagged $(jobs -p) 2>/dev/null; sleep 10; kill -KILL $tagged 2>/dev/null; exit 130
}
trap stop INT TERM

cd "$EX"
lane distil_bs8   "${GPUS[0]}" 64-79   "${AGNEWS[@]}" --train-batch-size 8 &
[ -n "$DRY" ] || sleep 3  # run_sequential names its smoke_logs dir by the second
lane distil_bs32  "${GPUS[1]}" 80-95   "${AGNEWS[@]}" --train-batch-size 32 &
[ -n "$DRY" ] || sleep 3
lane roberta_bs32 "${GPUS[2]}" 96-111  "${YELP[@]}" --train-batch-size 32 &
[ -n "$DRY" ] || sleep 3
lane smollm_bs32  "${GPUS[3]}" 112-127 "${YAHOO[@]}" --train-batch-size 32 &
wait
[ -n "$DRY" ] && exit 0

# Per lane: GPU s/task after 3 warm-up tasks; ratio vs DistilBERT bs 32; divisor = 0.478 / (ratio · bs / 8).
python3 - "$EX/experiments" "$B/summary.tsv" <<'EOF'
import glob, re, statistics, sys
exp, out = sys.argv[1], sys.argv[2]
cost = {}
for name in ("distil_bs8", "distil_bs32", "roberta_bs32", "smollm_bs32"):
    runs = sorted(glob.glob(f"{exp}/run_*ptc_{name}_*"))
    if not runs:
        continue
    log = glob.glob(f"{runs[-1]}/*trainers.log")[0]
    g = [float(x) for x in re.findall(r"charged to vclock; gpu=([0-9.]+)s", open(log).read())][3:]
    if len(g) >= 2:
        cost[name] = (len(g), statistics.mean(g), statistics.stdev(g))
with open(out, "w") as f:
    f.write("lane\ttasks\tgpu_s_per_task\tcv\tratio_vs_distil_bs32\tfluxtune_divisor\tfwdllm_divisor\n")
    anchor = cost.get("distil_bs32", (0, None, 0))[1]
    for name, (n, mu, sd) in cost.items():
        bs = int(name.rsplit("bs", 1)[1])
        r = mu / anchor if anchor else float("nan")
        div = 0.478 if name == "distil_bs8" else 0.478 / (r * bs / 8)
        f.write(f"{name}\t{n}\t{mu:.4f}\t{sd / mu:.3f}\t{r:.3f}\t{div:.5f}\t{10 * div:.4f}\n")
print(open(out).read())
EOF
echo "=== all done $(date -Is) ==="
