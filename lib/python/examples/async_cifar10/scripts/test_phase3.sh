#!/bin/bash
# Phase 3 End-to-End Test Script
# Tests complete experiment orchestration with small scale (5 trainers)

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Phase 3 End-to-End Test"
echo "========================================="
echo ""
echo "Test configuration:"
echo "  - 5 trainers (IDs 1-5)"
echo "  - Dirichlet alpha = 0.1"
echo "  - Availability: syn_0 (always available)"
echo "  - 2 GPUs"
echo ""

# Create minimal test config
TEST_CONFIG="$EXAMPLE_DIR/experiments/configs/test_phase3_mini.yaml"
cat > "$TEST_CONFIG" << 'EOF'
experiments:
  - name: test_oort_syn0_mini
    
    trainer:
      num_trainers: 5
      start_id: 1
      dataset:
        dirichlet_alpha: 0.1
      availability:
        mode: syn_0
    
    aggregator:
      selector: oort
      tracking_mode: oracular
      config_template: expt_scripts_2026/configs/oort_n300_oracular_9may25_syn0.json
    
    execution:
      num_gpus: 2
      sleep_between_spawns: 2.0
      aggregator_warmup_time: 5.0
EOF

echo "Created test config: $TEST_CONFIG"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -d "$EXAMPLE_DIR/metadata" ]; then
    echo "ERROR: metadata/ directory not found"
    echo "Run Phase 1 extraction first: python3 scripts/extract_metadata.py"
    exit 1
fi

if [ ! -f "$EXAMPLE_DIR/configs/trainer_base.yaml" ]; then
    echo "ERROR: trainer_base.yaml not found"
    echo "Phase 2 setup incomplete"
    exit 1
fi

echo "✓ Prerequisites met"
echo ""

# Run experiment
echo "========================================="
echo "Starting experiment..."
echo "========================================="
echo ""
echo "Log files will be created in:"
echo "  experiments/run_<timestamp>_test_oort_syn0_mini/"
echo ""
echo "Press Ctrl+C to stop the experiment"
echo ""

cd "$EXAMPLE_DIR"
python3 launch/run_experiment.py experiments/configs/test_phase3_mini.yaml

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="
