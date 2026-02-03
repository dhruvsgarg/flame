#!/bin/bash
# Phase 2 Test: Spawn 5 trainers using new spawner system

cd "$(dirname "$0")/.."

echo "======================================================================="
echo "PHASE 2 TEST: Spawning 5 trainers with new system"
echo "======================================================================="

# Make sure metadata exists
if [ ! -d "metadata" ]; then
    echo "Error: metadata/ directory not found. Run Phase 1 first."
    exit 1
fi

# Make sure configs exist
if [ ! -f "configs/trainer_base.yaml" ]; then
    echo "Error: configs/trainer_base.yaml not found."
    exit 1
fi

echo ""
echo "Test configuration:"
echo "  - Trainers: 5 (IDs 1-5)"
echo "  - Alpha: 0.1"
echo "  - Availability: mobiperf_2st"
echo "  - GPUs: 8"
echo ""

# Run spawner in test mode
python3 launch/spawner.py \
    --alpha 0.1 \
    --availability mobiperf_2st \
    --num-gpus 8 \
    --test

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✓ Test completed successfully"
else
    echo "✗ Test failed with exit code $exit_code"
fi

exit $exit_code
