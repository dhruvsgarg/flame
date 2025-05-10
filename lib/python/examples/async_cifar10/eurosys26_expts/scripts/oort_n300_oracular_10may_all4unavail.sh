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
threshold=0.70

baseline_names=("syncfl_oort")
aggType="fedavg"
selType="oort"
aware_modes=("syn0" "syn20" "syn50" "mobiperf" "unaware")
alpha=0.1
trainer_dir_suffix="_6d_3state_oort"

for baseline_name in "${baseline_names[@]}"; do
  for awareType in "${aware_modes[@]}"; do
    echo "$(date +'%Y-%m-%d %H:%M:%S') Starting experiment with alpha=${alpha}, awareType=${awareType}, node=${node_name}"
    start_time=$(date +%s)

    conda activate dg_flame
    terminate_main_py
    sleep 10

    # Aggregator
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
    cd /home/dgarg39/flame/lib/python/examples/async_cifar10/aggregator
    timestamp=$(date +%d_%m_%H_%M)
    config_file="oort_n300_oracular_9may25_${awareType}.json"
    agg_log_dir="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs"
    agg_log_file="${agg_log_dir}/agg_${node_name}_${timestamp}_alpha${alpha}_${aggType}_${selType}_${awareType}.log"
    wandb_run_name="agg_${node_name}_${timestamp}_alpha${alpha}_${aggType}_${selType}_${awareType}_c13_1.3k"

    echo "Created aggregator log file: ${agg_log_file}"
    python pytorch/main_oort_agg.py "/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/configs/${config_file}" \
      --log_to_wandb --wandb_run_name "$wandb_run_name" > "$agg_log_file" 2>&1 &
    agg_pid=$!
    echo "Aggregator PID: $agg_pid"
    sleep 15

    # Trainer
    conda activate dg_flame
    cd "/home/dgarg39/flame/lib/python/examples/async_cifar10/trainer/config_dir0.1_num300_traceFail${trainer_dir_suffix}/"
    echo "Going inside trainer folder: config_dir0.1_num300_traceFail${trainer_dir_suffix}"
    trainer_log_dir="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/trainer_logs"
    trainer_log_file="${trainer_log_dir}/log_trainer_${node_name}_${timestamp}_${alpha}_${aggType}_${selType}_${awareType}.log"
    echo "Created trainer log file: ${trainer_log_file}"
    bash exec_300_trainers_2state.sh > "$trainer_log_file" 2>&1 &
    trainer_pid=$!
    echo "Trainer PID: $trainer_pid"
    echo "$(date +'%Y-%m-%d %H:%M:%S') All trainers successfully started"

    # Monitor the log file
    while true; do
      if check_accuracy "$agg_log_file" "$threshold"; then
        # If condition is met, terminate main.py and wait for 30 seconds
        terminate_main_py
        sleep 30  # Wait for all trainers and aggregator to stop
        break
      else
        # If not, check again after a minute
        sleep 60
      fi
    done

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    elapsed_human=$(printf '%02dh:%02dm:%02ds\n' $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60)))

    echo "$(date +'%Y-%m-%d %H:%M:%S') Finished experiment with alpha=${alpha}, awareType=${awareType}, baseline=${baseline_name}. Time taken: ${elapsed_human}"
    echo "$(date +'%Y-%m-%d %H:%M:%S') Sleeping for a minute before next experiment"
    sleep 60
  done
done
