#!/bin/bash

# Source the bashrc file to ensure conda is set up
source /serenity/scratch/dgarg/anaconda3/etc/profile.d/conda.sh

check_accuracy() {
  local log_file=$1
  local threshold=$2
  local accuracy_values
  accuracy_values=$(grep -oP 'test accuracy: [0-9]+/[0-9]+ \(\K[0-9]+\.[0-9]+' "$log_file" | tail -n 20)

  # Check if we have at least 20 accuracy values
  if [ $(echo "$accuracy_values" | wc -l) -lt 20 ]; then
    echo "Less than 20 accuracy values found."
    return 1  # Condition not met
  fi

  # Initialize total and count
  local total=0
  local count=0

  # Calculate the total of the last 20 accuracy values
  for value in $accuracy_values; do
    total=$(python -c "print($total + $value)")
    count=$((count + 1))
  done

  # Calculate the average
  local average=$(python -c "print($total / $count)")

  # Get the last accuracy value
  local last_value=$(echo "$accuracy_values" | tail -n 1)

  # Perform the comparison using Python
  result=$(python -c "print(float($average) >= float($threshold) and float($last_value) >= float($threshold))")

  # Check the result of the comparison
  if [ "$result" == "True" ]; then
    return 0  # Condition met
  else
    return 1  # Condition not met
  fi
}

# Function to terminate main.py
terminate_main_py() {
  pkill -f main.py
  pkill -f main_oort_agg.py
}

# Check for the correct number of arguments
if [ "$#" -ne 1 ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') Usage: $0 <node-name>"
  exit 1
fi

node_name=$1

# Fixed alpha value
alpha=0.1
threshold=0.70

aggType="fedavg"
selType="oort"
awareMode="unaware"

# Availability traces to run in oracular mode
availability_traces=("syn20")

for trace in "${availability_traces[@]}"; do
  echo "$(date +'%Y-%m-%d %H:%M:%S') Starting experiment for trace=${trace}, mode=${awareMode}, alpha=${alpha} on node=${node_name}..."
  start_time=$(date +%s)

  conda activate dg_flame
  pkill -f main.py
  pkill -f main_oort_agg.py
  sleep 10
  echo "$(date +'%Y-%m-%d %H:%M:%S') Waited for cleanup to complete"

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
  cd /home/dgarg39/flame/lib/python/examples/async_cifar10/aggregator

  # Ensure log directories exist
  mkdir -p /home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs
  mkdir -p /home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/trainer_logs

  timestamp=$(date +%d_%m_%H_%M)
  agg_log_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs/agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}.log"
  config_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/configs/oort_n300_unaware_14may25.json"
  wandb_run_name="agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}"

  echo "Created aggregator log file: ${agg_log_file}"
  python pytorch/main_oort_agg.py "$config_file" --log_to_wandb --wandb_run_name "$wandb_run_name" > "$agg_log_file" 2>&1 &
  agg_pid=$!
  echo "Aggregator PID: $agg_pid"
  sleep 15
  echo "$(date +'%Y-%m-%d %H:%M:%S') Waited after aggregator start"

  conda activate dg_flame
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
  cd /home/dgarg39/flame/lib/python/examples/async_cifar10/trainer/config_dir0.1_num300_traceFail_6d_3state_oort/
  echo "Inside trainer folder for trace=${trace}"

  trainer_log_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/trainer_logs/log_trainer_${node_name}_${timestamp}_${alpha}_${aggType}_${selType}_${awareMode}_${trace}.log"
  echo "Created trainer log file: ${trainer_log_file}"
  bash exec_300_trainers_3state_felix.sh --notify_enabled False --notify_trace "$trace" > "$trainer_log_file" 2>&1 &
  trainer_pid=$!
  echo "Trainer PID: $trainer_pid"
  echo "$(date +'%Y-%m-%d %H:%M:%S') All trainers successfully started"

  while true; do
    if check_accuracy "$agg_log_file" "$threshold"; then
      terminate_main_py
      sleep 30
      break
    else
      sleep 60
    fi
  done

  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  elapsed_human=$(printf '%02dh:%02dm:%02ds\n' $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60)))
  echo "$(date +'%Y-%m-%d %H:%M:%S') Finished experiment for trace=${trace} in ${awareMode} mode on node=${node_name}. Time taken: ${elapsed_human}"
  echo "$(date +'%Y-%m-%d %H:%M:%S') Sleeping for a minute before next experiment"
  sleep 60
done