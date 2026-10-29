#!/usr/bin/env bash
# V1/V2 harness side: the jobs_identity.tsv YAMLs as harness sim legs (seed 1234 = bench --seeds 1), private broker each.
#   run_identity_harness.sh <V-phase regex> <gpu list> <cpu list>
set -u
B="$(cd "$(dirname "$0")" && pwd)"
EX="$(cd "$B/.." && pwd)"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
pat=$1 gpus=$2 cpus=$3
read -r phase _ _ src _ < <(grep -P "^$pat" "$B/jobs_identity.tsv")
y="$B/harness_$phase.yaml"
python3 - "$src" "$y" "$gpus" <<'PY'
import sys, yaml
src, out, gpus = sys.argv[1], sys.argv[2], [int(g) for g in sys.argv[3].split(",")]
d = yaml.safe_load(open(src))
for e in d["experiments"]:
    e["aggregator"]["config_overrides"]["hyperparameters"]["seed"] = 1234
    e["execution"]["gpu_ids"], e["execution"]["num_gpus"] = gpus, len(gpus)
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PY
port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
  "$port" "$B/mosquitto_$phase.log" > "$B/mosquitto_$phase.conf"
/usr/sbin/mosquitto -c "$B/mosquitto_$phase.conf" >/dev/null 2>&1 &
broker=$!
for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
echo "=== $phase gpus=$gpus broker=$port $(date -Is) ==="
cd "$EX"
FLAME_MQTT_BROKER="localhost:$port" FLAME_SLOT_GPUS="$gpus" FLAME_RUN_TAG="fwdsm_$phase" FLAME_RUN_LABEL="fwdsm_$phase" \
  taskset -c "$cpus" python -m flame.launch.run_experiment "$y" --example-dir "$EX" < /dev/null > "$B/harness_$phase.out" 2>&1
echo "=== $phase exit=$? $(date -Is) ==="
kill "$broker" 2>/dev/null
