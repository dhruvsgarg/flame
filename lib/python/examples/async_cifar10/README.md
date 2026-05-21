# Async CIFAR-10 Federated Learning

Asynchronous federated learning on CIFAR-10 with 300 trainers, demonstrating client selection strategies (Oort, FedBuff), availability tracking, and non-IID data distributions.

## Quick Start

```bash
# 1. Set up the conda env (Python 3.11 + flame + examples extras).
#    Run from the repo root:
bash scripts/setup_env.sh my_flame_env

# 2. Check MQTT broker is running.
systemctl is-active mosquitto || pgrep mosquitto

# 3. Run an experiment via the YAML launcher.
conda activate my_flame_env
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/felix_n10_alpha100_syn20_smoke.yaml
```

Logs land in `experiments/run_<timestamp>_<name>/` (aggregator + trainers,
plus the merged `aggregator_config.json` for reproducibility).

## What This Example Does

Simulates a federated learning environment with:
- **1 Aggregator**: Central server coordinating training
- **300 Trainers**: Clients with local CIFAR-10 data partitions
- **MQTT Communication**: Message broker for parameter exchange
- **Smart Selection**: Oort strategy picks best trainers based on utility
- **Availability Traces**: Simulates real-world trainer unavailability (syn0/20/50, MobiPerf)
- **Non-IID Data**: Dirichlet sampling (alpha=0.1) creates heterogeneous data distributions

The experiment runs until reaching 70% test accuracy, testing 4 availability scenarios automatically.

## Directory Structure

```
async_cifar10/
├── aggregator/               # Central server
│   ├── pytorch/main_oort_agg.py
│   └── *.json               # Legacy aggregator configs
├── trainer/                  # Client trainers
│   └── pytorch/main.py
├── configs/
│   └── trainer_base.yaml    # Per-example trainer template
├── expt_scripts_2026/
│   ├── felix_*.yaml         # Experiment YAMLs (launcher inputs)
│   └── configs/             # Aggregator config templates
└── data/                    # CIFAR-10 (auto-downloaded by trainers)

# Shared across examples (sibling at examples/_metadata):
examples/_metadata/
├── trainer_registry.yaml             # n=300 device population
├── availability_traces/              # mobiperf + synthetic
├── dataset_splits/                   # per-(dataset, alpha, N) splits
├── baselines.yaml                    # felix / refl / feddance / oort
└── aggregator_base.json              # generic aggregator boilerplate
```

## Manual Setup

If `scripts/setup_env.sh` doesn't work for your system:

```bash
conda create -n my_flame_env python=3.11 -y
conda activate my_flame_env
pip install -e lib/python[examples,dev]

# Check MQTT broker
systemctl is-active mosquitto 2>/dev/null || pgrep mosquitto
```

Required deps (installed automatically by the extras above):
- core (flame): paho-mqtt, pydantic, mlflow, grpcio, protobuf, PyYAML, ...
- examples: torch, torchvision, sortedcontainers, wandb
- wandb, sortedcontainers
- All Flame library dependencies

## Running Experiments

### Using Automated Script (Recommended)

```bash
cd eurosys26_expts/scripts
./oort_n300_oracular_10may_all4unavail.sh my_node_name
```

The script:
- Starts aggregator and 300 trainers
- Monitors accuracy, stops at 70%
- Runs 4 traces: syn0 (no failures), syn20, syn50, mobiperf
- Saves logs to `eurosys26_expts/{agg,trainer}_logs/`

### Manual Execution

**Terminal 1 (Aggregator)**:
```bash
conda activate my_flame_env  # Use your environment name
cd aggregator
python pytorch/main_oort_agg.py ../eurosys26_expts/configs/oort_n300_oracular_9may25_syn0.json
```

**Terminal 2 (Trainers)**:
```bash
conda activate my_flame_env  # Use your environment name
cd trainer/config_dir0.1_num300_traceFail_6d_3state_oort/
bash exec_300_trainers_2state.sh  # Distributes 300 trainers across 8 GPUs
```

**Single Trainer (Testing)**:
```bash
CUDA_VISIBLE_DEVICES=0 python ../pytorch/main.py --config trainer_1.json
```

## Key Configuration Parameters

**Aggregator config** (`eurosys26_expts/configs/*.json`):
```json
{
  "hyperparameters": {
    "aggGoal": 10,              // Trainer updates before global aggregation
    "trackTrainerAvail": {
      "type": "ORACULAR",       // ORACULAR (knows availability) or UNAWARE
      "trace": "avl_events_syn_0"  // syn0/20/50 or mobiperf
    }
  },
  "selector": {
    "sort": "oort",             // oort, random, fedbuff
    "kwargs": {"aggr_num": 10}  // Trainers selected per round
  }
}
```

**Trainer directories** (data distribution):
- `config_dir0.1_num300_*` - Highly non-IID (realistic)
- `config_dir1_num300_*` - Moderately non-IID
- `config_dir100_num300_*` - Nearly IID

## Monitoring

```bash
# Watch aggregator
tail -f eurosys26_expts/agg_logs/agg_*.log | grep "test accuracy"

# Watch trainers
tail -f eurosys26_expts/trainer_logs/log_trainer_*.log

# Weights & Biases (if configured)
# Visit: https://wandb.ai/your-username/ft-distr-ml
```

**Expected progress**: ~30% (round 50) → ~60% (round 200) → 70% (stop)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MQTT connection failed | Check if mosquitto is running: `systemctl is-active mosquitto` or `pgrep mosquitto`. Contact admin if not running. |
| CUDA out of memory | Edit `exec_*_trainers.sh`, increase `NUM_AVAIL_GPUS` or reduce `batchSize` |
| Import error: `flame` | `cd ../../ && pip install -e .` |
| Processes hang | `pkill -f main.py && pkill -f main_oort_agg.py` |

## References

- [Oort Paper (OSDI'21)](https://www.usenix.org/conference/osdi21/presentation/lai)
- [FedBuff Paper](https://arxiv.org/abs/2106.06639)
- [Flame Documentation](https://github.com/cisco-open/flame/tree/main/docs)
