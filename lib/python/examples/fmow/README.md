# FMoW on Flame

FMoW (Functional Map of the World) is a satellite imagery dataset with 62 scene
categories, such as airports and amusement parks. This example uses its
WILDS-preprocessed RGB release to simulate federated learning across a satellite
constellation. Each trainer represents one satellite, with access to images
determined by its orbital path and elapsed simulation time.

## Getting started

With the GPU environment, MQTT broker, and orbital inputs ready, run from the
repository root:

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all
python -m flame.launch.run_experiment \
  lib/python/examples/fmow/exports/fmow_fedavg_smoke_n10.yaml
```

Setup downloads the dataset if needed and regenerates captures and configured
availability traces. The [smoke experiment](exports/fmow_fedavg_smoke_n10.yaml) runs
10 trainers for 50 rounds on two GPUs with synthetic `syn_0` availability.

See [Getting started](docs/getting-started.md) for prerequisites, input files,
and configuration. The [200-trainer experiment](exports/fmow_fedavg_n200.yaml)
uses eight GPUs, 500 rounds, and satellite contact availability.

## How it works

1. [Setup](setup/setup_fmow.py) prepares the dataset and uses precomputed orbits
   to generate image capture times and ground-station contact windows.
2. Each [trainer](trainer/pytorch/main.py) adds newly captured images to its
   local dataset before training. Images remain available for later rounds,
   so the dataset grows over time. A satellite can initially have no images.
3. The [aggregator](aggregator/pytorch/main_fedavg_agg.py) combines trainer
   updates with synchronous FedAvg and evaluates the global model on FMoW's
   test split.

An image is considered captured when a satellite passes within the configured
radius of its location. These events are saved in `captures.npz`.
The [model](dependencies/model.py) is ImageNet-pretrained DenseNet161 with group
normalization; only denseblock4, norm5, and the classifier are trained.

## Satellite availability

Availability describes when a satellite has a usable ground-station contact.
Setup calculates these windows from orbital positions, station locations,
minimum elevation, and required remaining contact time. Both training roles
read the resulting available/unavailable events from `satellite_traces.yaml`.

This file records when each satellite can communicate with a ground station.
It does not specify how fast data transfers or how long training takes. See the
[configuration reference](docs/config-reference.md#satellite-availability)
for file formats, runtime settings, and trace limits.

## Estimate data coverage

The coverage tool estimates how many unique training images satellites capture
and how many are held by a satellite that can communicate with a ground station
after capture. It uses saved
captures and availability traces without running training. See the
[data coverage guide](docs/data-coverage.md) for commands and how to read the results.

## Documentation

| Topic | Document |
| --- | --- |
| How FMoW runs on Flame and the changes made to support it | [Implementation details](docs/technical-overview.md) |
| What each configuration setting does and how to change it | [Configuration reference](docs/config-reference.md) |
| Estimating data coverage | [Data coverage](docs/data-coverage.md) |
| Known runtime issues and proposed changes | [Fix roadmap](docs/fmow-fix-roadmap.md) |
