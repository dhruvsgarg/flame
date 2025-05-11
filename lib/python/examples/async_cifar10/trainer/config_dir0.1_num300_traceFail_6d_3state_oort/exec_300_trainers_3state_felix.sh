#!/bin/bash

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

# Update all trainer_*.json files
for json_file in trainer_*.json; do
    jq --arg enabled "$NOTIFY_ENABLED" \
       --arg trace "$NOTIFY_TRACE" \
       '.hyperparameters.client_notify.enabled = $enabled | .hyperparameters.client_notify.trace = $trace' \
       "$json_file" > tmp.json && mv tmp.json "$json_file"
done

# Start trainers
for ((X=1; X<=300; X++)); do
    ASSIGN_TO_GPU=$(( X % NUM_AVAIL_GPUS ))
    echo "ASSIGN_TO_GPU=${ASSIGN_TO_GPU} value for X=${X}" 

    CUDA_VISIBLE_DEVICES="${ASSIGN_TO_GPU}" python ../pytorch/main.py --config "trainer_${X}.json" &

    sleep 1
done
