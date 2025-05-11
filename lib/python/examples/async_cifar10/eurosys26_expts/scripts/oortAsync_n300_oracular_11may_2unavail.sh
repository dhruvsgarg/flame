#!/bin/bash

# Source the bashrc file to ensure conda is set up
source /serenity/scratch/dgarg/anaconda3/etc/profile.d/conda.sh

check_accuracy() {
  local log_file=$1
  local threshold=$2
  local accuracy_values
  accuracy_values=$(grep -oP 'test accuracy: [0-9]+/[0-9]+ \(\K[0-9]+\.[0-9]+' "$log_file" | tail -n 20)

  if [ $(echo "$accuracy_values" | wc -l) -lt 20 ]; then
    echo "Less than 20 accuracy values found."
    return 1
  fi

  local total=0
  local count=0

  for value in $accuracy_values; do
    total=$(python -c "print($total + $value)")
    count=$((count + 1))
  done

  local average=$(python -c "print($total / $count)")
  local last_value=$(echo "$accuracy_values" | tail -n 1)

  result=$(python -c "print(float($average) >= float($threshold) and float($last_value) >= float($threshold))")

  if [ "$result" == "True" ]; then
    return 0
  else
    return 1
  fi
}

terminate_main_py() {
  pkill -f main.py
  pkill -f main_oort_agg.py
}

if [ "$#" -ne 1 ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') Usage: $0 <node-name>"
  exit 1
fi

node_name=$1
alpha=0.1
threshold=0.70
aggType="fedbuff"
selType="oortAsync"
awareMode="oracular"

availability_traces=("syn20" "mobiperf")

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

  mkdir -p /home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs
  mkdir -p /home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/trainer_logs

  timestamp=$(date +%d_%m_%H_%M)
  agg_log_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs/agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}.log"
  config_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/configs/oortAsync_n300_oracular_9may25_${trace}.json"
  wandb_run_name="agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}_c13_1.3k"

  echo "Created aggregator log file: ${agg_log_file}"
  python pytorch/main.py "$config_file" --log_to_wandb --wandb_run_name "$wandb_run_name" > "$agg_log_file" 2>&1 &
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
  bash exec_300_trainers_2state.sh > "$trainer_log_file" 2>&1 &
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