#!/bin/bash

# Script to parse FLAME training logs and extract stat_utility metrics
# Usage: ./extract_stat_utility.sh <path_to_log_file>

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_log_file>"
    exit 1
fi

LOG_FILE=$1
OUTPUT_CSV="stat_utility_comparison.csv"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found at $LOG_FILE"
    exit 1
fi

echo "Extracting stat_utility metrics from $LOG_FILE..."
echo "TrainerID,Iteration,DataBin,BinStatUtility,FullDatasetStatUtility,FullSamples" > $OUTPUT_CSV

# We will use awk to parse the logs
# The relevant log lines look like this:
# 1. stat_utility - normalized for trainerId: {trainer_id} = {stat_utility}
# 2. full_dataset_stat_utility for trainerId: {trainer_id} is {full_stat_utility} over {samples} samples
# 3. We also want to capture Iteration and DataBin where possible, often found in training context logs

# A robust approach using grep and python or awk to correlate lines per round is ideal.
# Here is a python one-liner inside bash to handle the stateful parsing of logs:

python3 -c "
import sys
import re
import csv

log_file = sys.argv[1]
output_file = sys.argv[2]

# Regex patterns
bin_stat_pattern = re.compile(r'stat_utility - normalized for trainerId: (\S+) = ([\d\.]+)')
full_stat_pattern = re.compile(r'full_dataset_stat_utility for trainerId: (\S+) is ([\d\.]+) over (\d+) samples')
iter_data_pattern = re.compile(r'completed training for trainer id: (\S+), data_id = (\d+)')
fetch_pattern = re.compile(r'FETCH WEIGHTS complete for trainer_id (\S+), round: (\d+), data id: (\d+)')

trainer_states = {}

with open(log_file, 'r') as f, open(output_file, 'w', newline='') as out_csv:
    writer = csv.writer(out_csv)
    # Header already written by bash, but we can write it again if we want.
    # We rely on bash writing the header.
    
    for line in f:
        # Match bin level stat utility
        m1 = bin_stat_pattern.search(line)
        if m1:
            tid = m1.group(1)
            if tid not in trainer_states: trainer_states[tid] = {}
            trainer_states[tid]['bin_stat'] = m1.group(2)
            
        # Match full dataset stat utility
        m2 = full_stat_pattern.search(line)
        if m2:
            tid = m2.group(1)
            if tid not in trainer_states: trainer_states[tid] = {}
            trainer_states[tid]['full_stat'] = m2.group(2)
            trainer_states[tid]['samples'] = m2.group(3)
            
        # Match iteration context (fetch weights tells us round and data id early on)
        m3 = fetch_pattern.search(line)
        if m3:
            tid = m3.group(1)
            if tid not in trainer_states: trainer_states[tid] = {}
            trainer_states[tid]['round'] = m3.group(2)
            trainer_states[tid]['data_bin'] = m3.group(3)
            
        # Match end of training round to emit row
        m4 = iter_data_pattern.search(line)
        if m4:
            tid = m4.group(1)
            if tid in trainer_states and 'bin_stat' in trainer_states[tid] and 'full_stat' in trainer_states[tid]:
                # Construct row
                rnd = trainer_states[tid].get('round', 'N/A')
                dbin = trainer_states[tid].get('data_bin', m4.group(2))
                bstat = trainer_states[tid].get('bin_stat', '')
                fstat = trainer_states[tid].get('full_stat', '')
                samples = trainer_states[tid].get('samples', '')
                
                writer.writerow([tid, rnd, dbin, bstat, fstat, samples])
                
                # Clear for next round
                trainer_states.pop(tid, None)

print(f'Done! Output saved to {output_file}')
" "$LOG_FILE" "$OUTPUT_CSV"

echo "Results written to $OUTPUT_CSV"
