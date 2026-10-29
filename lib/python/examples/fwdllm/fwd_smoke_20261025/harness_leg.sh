#!/usr/bin/env bash
# One harness sim leg with a private broker: harness_leg.sh <src yaml> <name> <gpu list> <cpu list>
# Seed 1234 (= bench --seeds 1); minInitialTrainers = n, so the first cohort is drawn from every trainer, as on the bench.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
EX="$(cd "$B/.." && pwd)"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
src=$1 name=$2 gpus=$3 cpus=$4
y="$B/harness_$name.yaml"
python3 - "$src" "$y" "$gpus" <<'PY'
import sys, yaml
src, out, gpus = sys.argv[1], sys.argv[2], [int(g) for g in sys.argv[3].split(",")]
d = yaml.safe_load(open(src))

def walk(x, n):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "minInitialTrainers":
                x[k] = n
            else:
                walk(v, n)
    elif isinstance(x, list):
        for v in x:
            walk(v, n)

for e in d["experiments"]:
    e["aggregator"]["config_overrides"]["hyperparameters"]["seed"] = 1234
    e["execution"]["gpu_ids"], e["execution"]["num_gpus"] = gpus, len(gpus)
    walk(e, int(e["trainer"]["num_trainers"]) if "num_trainers" in e["trainer"] else None)
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PY
grep -q "minInitialTrainers: None" "$y" && { echo "!!! $name: num_trainers not found"; exit 1; }
port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
  "$port" "$B/mosquitto_$name.log" > "$B/mosquitto_$name.conf"
/usr/sbin/mosquitto -c "$B/mosquitto_$name.conf" >/dev/null 2>&1 &
broker=$!
for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
echo "=== $name gpus=$gpus broker=$port $(date -Is) ==="
cd "$EX"
FLAME_MQTT_BROKER="localhost:$port" FLAME_SLOT_GPUS="$gpus" FLAME_RUN_TAG="fwdsm_$name" FLAME_RUN_LABEL="fwdsm_$name" \
  taskset -c "$cpus" python -m flame.launch.run_experiment "$y" --example-dir "$EX" < /dev/null > "$B/harness_$name.out" 2>&1
echo "=== $name exit=$? $(date -Is) ==="
kill "$broker" 2>/dev/null
