#!/bin/bash

# Loop from 1 to 100
for ((X=1; X<=100; X++)); do
    # Run the Python command with the corresponding trainer_X.json file
    python ../pytorch/main.py "trainer_${X}.json" &
    
    # Set the time gap based on the condition
    # if [ $X -eq 1 ] || [ $X -eq 2 ]; then
    #     sleep 30  # 30 seconds gap for the first two commands
    # else
    #     sleep 5   # 5 seconds gap for subsequent commands
    # fi
done