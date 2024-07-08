#!/bin/bash

# TODO: Pass this from args
NUM_AVAIL_GPUS=8

# Loop from 1 to 100
for ((X=1; X<=100; X++)); do
    # Load balance the trainers across available GPUs
    ASSIGN_TO_GPU=$(( $X % $NUM_AVAIL_GPUS))
    echo "ASSIGN_TO_GPU=${ASSIGN_TO_GPU} value for X=${X}" 

    # Run the Python command with the corresponding trainer_X.json file
    CUDA_VISIBLE_DEVICES="${ASSIGN_TO_GPU}" python ../pytorch/main_resnet.py "trainer_${X}.json" &
    
    # Set the time gap based on the condition
    # if [ $X -eq 1 ] || [ $X -eq 2 ]; then
    #     sleep 30  # 30 seconds gap for the first two commands
    # else
    #     sleep 5   # 5 seconds gap for subsequent commands
    # fi
done