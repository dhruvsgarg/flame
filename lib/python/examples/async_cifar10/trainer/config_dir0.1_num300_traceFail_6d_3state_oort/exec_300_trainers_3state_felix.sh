#!/bin/bash

# Source the bashrc file to ensure conda is set up
source /serenity/scratch/dgarg/anaconda3/etc/profile.d/conda.sh

conda activate dg_flame
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
export PATH="/serenity/scratch/dgarg/anaconda3/envs/dg_flame/bin:$PATH"

# Default values
NUM_AVAIL_GPUS=8
NOTIFY_ENABLED="False"
NOTIFY_TRACE="syn_0"

# Parse optional args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --notify_enabled) NOTIFY_ENABLED="$2"; shift ;;
        --notify_trace) NOTIFY_TRACE="$2"; shift ;;
        --gpus) NUM_AVAIL_GPUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Update all trainer_*.json files using Python
for json_file in trainer_*.json; do
    python3 - <<EOF
import json

fname = "$json_file"
with open(fname) as f:
    data = json.load(f)

data["hyperparameters"]["client_notify"]["enabled"] = "$NOTIFY_ENABLED"
data["hyperparameters"]["client_notify"]["trace"] = "$NOTIFY_TRACE"

with open(fname, "w") as f:
    json.dump(data, f, indent=4)
EOF
done

# Start trainers
for ((X=1; X<=300; X++)); do
    ASSIGN_TO_GPU=$(( X % NUM_AVAIL_GPUS ))
    echo "ASSIGN_TO_GPU=${ASSIGN_TO_GPU} value for X=${X}" 

    CUDA_VISIBLE_DEVICES="${ASSIGN_TO_GPU}" python ../pytorch/main.py --config "trainer_${X}.json" &

    sleep 1
done
