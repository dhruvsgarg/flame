# Metadata

This folder contains trainer metadata and satellite latency data used in async federated learning experiments.

## Contents

- `trainer_registry.yaml` — per-trainer metadata (computation time, satellite index, RTT params, speed class)
- `baselines.yaml` — baseline experiment configurations
- `aggregator_base.json` — base aggregator configuration
- `availability_traces/` — trainer availability traces
- `dataset_splits/` — dataset index assignments per trainer
- `leo/` — LEO satellite latency traces
  - `satellite_latencies.npy` — propagation latencies from a ground station in Norway to 150 LEO satellites over 10800 timesteps (units: ms, range: ~1.7–85.5ms). Generated using https://github.gatech.edu/vbhosale6/planet-latency
- `scripts/` — scripts for exploring or modifying metadata
  - `explore_latencies.py` — visualize and analyze satellite_latencies.npy
  - `update_registry.py` — update fields in trainer_registry.yaml

## Regenerating satellite_latencies.npy

Use the scripts at https://github.gatech.edu/vbhosale6/planet-latency to regenerate propagation latencies from a ground station location of your choice.
