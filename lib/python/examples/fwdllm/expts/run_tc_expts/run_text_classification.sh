client_num_per_round=$1
LR=$2
FL_ALG=$3

pkill -f fl_main.py
sleep 10  # Wait for the system to stabilize
C_LR=0.01
S_LR=0.1
ROUND=3
WORKER_NUM=1
model_type=distilbert
model_name=distilbert-base-uncased
# model_type=bert
# model_name=bert-base-uncased
# model_type=bert
# model_name=bert-large-uncased
# model_type=albert
# model_name=albert-base-v2
# model_type=roberta-large
# model_name=roberta-large
# model_type=deberta
# model_name=microsoft/deberta-xlarge
train_batch_size=8
DATA_NAME=agnews
# fold_name=${model_type}_${DATA_NAME}

if [ $model_type = "distilbert" ];then
  peft_method=adapter
else
  peft_method=bitfit
fi

PARTITION_METHOD="uniform"
if [ $DATA_NAME = "agnews" ];then
  max_seq_length=64
  frequency_of_the_test=1
elif [ $DATA_NAME = "20news" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yelp-p" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yahoo" ];then
  max_seq_length=256
  frequency_of_the_test=5
  PARTITION_METHOD="uniform_client_10000"
else
  max_seq_length=256
  frequency_of_the_test=1
fi


LOG_FILE="fedavg_transformer_tc.log"
CI=0

REPO_PATH=/home/dgarg39/flame
DATA_DIR=$REPO_PATH/lib/python/examples/fwdllm/fednlp_data/

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file
if [ $FL_ALG = "FedAvg" ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fed_avg_main_tc.py \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --frequency_of_the_test $frequency_of_the_test \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    > ./log/new/fedavg_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}.log 2>&1
elif [ $FL_ALG = FedSgd ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --frequency_of_the_test $frequency_of_the_test \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    > ./log/new/fedsgd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_full.log 2>&1
else
  # Run aggregator/main.py once with logging
  python $REPO_PATH/lib/python/examples/fwdllm/aggregator/fl_main.py \
    --config "$REPO_PATH/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/aggregator.json" \
    > ./log/new/test_agg_fedFwd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_numerical_$(date +%d_%m_%H_%M).log 2>&1 &

  # Sleep for 10 seconds so that agg sets everything up before trainer starts
  sleep 10
  
  # Run trainer/main.py 100 times, each with a unique log file
  NUM_AVAIL_GPUS=8

  # Run trainer/main.py 100 times, each with a unique log file
  for X in $(seq 1 7)
  do
    # Assign GPUs in a round-robin fashion
    ASSIGN_TO_GPU=$(( X % NUM_AVAIL_GPUS ))

    echo "Running client $X on GPU $ASSIGN_TO_GPU"
    CUDA_VISIBLE_DEVICES="${ASSIGN_TO_GPU}" python $REPO_PATH/lib/python/examples/fwdllm/trainer/fl_main.py \
      --config "$REPO_PATH/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/trainer_${X}.json" \
      >> ./log/new/test_trainer_fedFwd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_numerical_$(date +%d_%m_%H_%M).log 2>&1 &
    sleep 2
  done

  # Wait for all processes to finish
  wait
fi