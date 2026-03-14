#!/bin/bash

# Script to parse FLAME training logs and extract stat_utility metrics
# This is a wrapper around the extract_stat_utility.py script.

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_log_file> [output_csv_file]"
    exit 1
fi

LOG_FILE=$1
OUTPUT_CSV=${2:-"stat_utility_comparison.csv"}

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found at $LOG_FILE"
    exit 1
fi

# Run the python script located in the same directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/extract_stat_utility.py"

python "$PYTHON_SCRIPT" "$LOG_FILE" "$OUTPUT_CSV"
