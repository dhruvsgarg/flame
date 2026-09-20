# FMoW implementation

## What is FMoW?

FMoW (Functional Map of the World) contains satellite images labeled with 62
scene categories, including airports, race tracks, and amusement parks.
This example uses WILDS-preprocessed FMoW-rgb v1.1, with `train`, `val`,
`test`, and `seq` splits. Images are 224x224 PNGs named
`images/rgb_img_{i}.png`, where `i` is the row in `rgb_metadata.csv`.
The CSV's `img_path` column is not used to locate these files.

Each trainer represents a LEO satellite. Its data comes from captures along
its orbit, rather than a fixed partition. The model is ImageNet-pretrained
DenseNet161 with group normalization; only denseblock4, norm5, and the
classifier are trainable.

## Layout

Paths are relative to the FMoW directory.

```text
configs/
  fmow_config.yaml                 Dataset, orbit, capture, and contact settings
  trainer_base.yaml                Flame trainer template
dependencies/
  model.py                         Model construction and layer freezing
  fmow_dataset.py                  Image loading, labels, splits, transforms
  satellite_availability.py        Contact trace reader for both roles
setup/
  config.py                        Configuration schema and path resolution
  setup_fmow.py                    Setup commands
  download_fmow_dataset.sh          Download and extraction
  captures.py                      Generate captures.npz
  generate_ground_stations.py       Generate station coordinates
  generate_satellite_availability.py Generate contact events
trainer/pytorch/main.py             PyTorchFMoWTrainer
aggregator/pytorch/main_fedavg_agg.py PyTorchFMoWAggregator
exports/                           Experiment YAMLs
scripts/data_coverage.py            Optional coverage estimate
```

## Configuration wiring

Both role directories symlink `model.py`, `fmow_dataset.py`, `config.py`,
and `satellite_availability.py`, allowing direct imports of the shared code.
`metadata` is a symlink to `../_metadata`.

Set `fmow_config_path` in [trainer_base.yaml](../configs/trainer_base.yaml)
and separately under the experiment's
`aggregator.config_overrides.hyperparameters`. The aggregator has its own
configuration merge and never reads the trainer template. Dataset, orbital,
and availability paths inside the FMoW config resolve from the repository root.

## How a round works

1. [captures.py](../setup/captures.py) records the first time each satellite
   comes within the capture radius of a training image. `captures.npz` contains
   `events` as `(time, image row)`, `offsets`, and `sat_names`.
2. The [trainer](../trainer/pytorch/main.py) loads its event slice using
   `offsets[satellite_index:satellite_index + 2]`. Before each training call,
   `_admit_captured_images()` adds events through the current task time to
   `image_buffer`; `_rebuild_train_loader()` rebuilds the growing dataset.
3. Local SGD produces parameter deltas. The shared Flame trainer omits frozen
   parameters from uploads; empty clients send metadata without weights.
4. The [aggregator](../aggregator/pytorch/main_fedavg_agg.py) uses the sync
   `TopAggregator`, random selection, and FedAvg. It evaluates on the test split.

## Generating satellite availability

[setup_fmow.py](../setup/setup_fmow.py) runs download, captures, ground stations,
and availability in order, or accepts an individual step and `--config`.
Orbital files must already exist. Omitting `ground_stations` skips both
station and availability trace generation.

The availability generator combines satellite positions in `ecef.npz` with
station coordinates in `ground_stations.yaml` to find usable communication windows.
It applies elevation and remaining-window thresholds. Both roles read `availability.trace_path`
when `satellite` mode is selected. The
[reader](../dependencies/satellite_availability.py) gives the trainer an event
list and the aggregator timestamp-sorted events keyed by registry endpoint ID.
Satellite index 0 maps to `trainer_001`; missing entries fail. The trace assumes
one-second samples, ends unavailable, and contains no latency or bandwidth data.

## Changes made for FMoW

Supporting FMoW required changes to trainer timing, update payloads, message
assembly, selector cleanup, and evaluation. The shared Flame changes also affect
other workloads using those components.

#### Trainer timing and uploads

- **Send jitter:** The [FMoW trainer](../trainer/pytorch/main.py) sleeps for
  `random.uniform(0, 1)` seconds after training in simulated mode, spreading
  physical upload times. This is a hardcoded wall-clock delay with no config
  field. It runs after `_sim_completion_ts` is calculated and is not added to
  that timestamp. The `stream_rate_jitter` setting is separate.
- **Smaller updates:** The [shared trainer](../../../flame/mode/horizontal/syncfl/trainer.py)
  removes frozen parameters from uploaded deltas. Empty clients send model
  version, dataset size, utility, and accuracy metadata without weights.
- **Optional heartbeats:** An omitted `heartbeats` section defaults to disabled.

#### Transport and selector bookkeeping

- [ChunkStore](../../../flame/backend/chunk_store.py) buffers chunks by sequence
  number, ignores duplicates in the current buffer, and assembles complete
  messages in order. The default chunk size is 4 MiB.
- [ChunkManager](../../../flame/backend/chunk_manager.py) resets incomplete
  transfers older than 30 seconds from their first chunk when its queue wait
  times out. Transfer IDs and retransmission remain absent; consecutive messages
  can still mix after a missing chunk.
- [RandomSelector](../../../flame/selector/random.py) releases recorded responders
  from `selected_ends` and `all_selected` at round completion. The
  [sync aggregator](../../../flame/mode/horizontal/syncfl/top_aggregator.py)
  records responses without weights too, invokes cleanup when aggregation returns
  no model, and logs eligibility and in-flight IDs on starvation. This does not
  resolve every surplus-result case described in the [roadmap](fmow-fix-roadmap.md).
- [DefaultSelector](../../../flame/selector/default.py) discards departed
  endpoints through `_cleanup_removed_ends`.

#### Evaluation

The [FMoW aggregator](../aggregator/pytorch/main_fedavg_agg.py) limits intermediate
evaluation to the first five test batches and normalizes metrics by samples
actually evaluated. The last round bypasses the cadence filter and runs the full
test pass synchronously. Evaluation logs its start and progress and clears
`_eval_inflight` after success; the completion-event issue remains in the roadmap.

#### Setup, experiment, and analysis

The [200-trainer export](../exports/fmow_fedavg_n200.yaml) uses eight GPUs,
500 rounds, aggregation goal 175, periodic evaluation/checkpoints, and satellite
availability. It sets `eval_every_n_commits: 1` and a 200-trainer startup threshold.
The checked-in station config and generated files use one station.

The optional [coverage script](../scripts/data_coverage.py) estimates unique
captured/reachable images for a selected satellite count.
[Setup tests](../setup/tests/test_setup_fmow.py) cover optional generation and paths;
[coverage tests](../scripts/tests/test_data_coverage.py) cover contact boundaries,
duplicate captures, and satellite subsets.

For commands and settings, see [setup](getting-started.md),
[configuration](config-reference.md), and the optional [coverage script](data-coverage.md).
