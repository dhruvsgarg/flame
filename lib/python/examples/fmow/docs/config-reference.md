# FMoW configuration and availability

[fmow_config.yaml](../configs/fmow_config.yaml) supplies dataset, orbit, capture,
and ground-station settings to setup and both training roles. The experiment
YAML separately chooses trainer count, GPUs, aggregation, and availability mode.

## Configuration at a glance

This example shows every supported field, using the checked-in settings with
omitted defaults made explicit. Edit these values in `fmow_config.yaml`.

```yaml
dataset:
  root_dir: lib/python/examples/fmow/data/fmow
  num_classes: 62
  image_size: 224             # Pixels

satellites:
  leo_dir: lib/python/examples/fmow/metadata/leo

capture:
  radius: 15.0               # Kilometers

ground_stations:             # Omit this section to keep existing stations and contacts
  num_stations: 1            # Checked-in value; schema default is 8
  seed: 0                   # Station placement seed
  elevation_angle: 10.0      # Degrees above the horizon
  min_window: 90.0           # Required remaining contact time in seconds

availability:
  trace_path: lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml
```

See [fields and defaults](#fields-and-defaults) for optional-section behavior.
Trainer count, GPUs, rounds, and learning rate are set separately in an
[experiment YAML](../exports/fmow_fedavg_n200.yaml).

## Satellite availability

Satellite availability is a saved schedule of usable ground-station contacts.
[generate_satellite_availability.py](../setup/generate_satellite_availability.py)
combines `ecef.npz` and `ground_stations.yaml`: a satellite is eligible when at
least one station meets the elevation threshold and remaining-window requirement.

The generated `satellite_traces.yaml` stores generation settings and per-trainer
`[time, state]` events, for example:

```yaml
trainers:
  trainer_001:
    - [0, UN_AVL]
    - [120, AVL_TRAIN]
    - [300, UN_AVL]
```

Here, satellite 0 has a usable contact from second 120 up to, but excluding,
second 300. `AVL_TRAIN` means eligible; `UN_AVL` means unavailable.
Availability does not guarantee that a trainer is selected or has captured data.

### Which file contains what?

| Input | Meaning | Reader or generator |
| --- | --- | --- |
| `leo/geodetic.npz`, `leo/ecef.npz` | Precomputed satellite positions. | [captures.py](../setup/captures.py), [availability generator](../setup/generate_satellite_availability.py) |
| `leo/captures.npz` | First capture time for each image on each satellite; arrays `events`, `offsets`, `sat_names`. | [Trainer](../trainer/pytorch/main.py) |
| `leo/ground_stations.yaml` | Station coordinates. | [Station generator](../setup/generate_ground_stations.py), availability generator |
| `availability_traces/satellite_traces.yaml` | Available/unavailable contact events. No latency or bandwidth values. | [SatelliteAvailability](../dependencies/satellite_availability.py) |
| [trainer_registry.yaml](../../_metadata/trainer_registry.yaml) | Trainer identity and configured compute delay, including `training_delay_s`. | [Launcher](../../../flame/launch/spawner.py) |
| [trainer_base.yaml](../configs/trainer_base.yaml) and experiment export | Broker, training settings, and selected availability mode. | Launcher and runtime roles |

The first three `leo/` paths use `satellites.leo_dir`; the trace path is configured
separately. Defaults resolve through `fmow/metadata` to shared `examples/_metadata`.

The current FMoW path uses MQTT for message transport. Satellite availability
does not calculate radio-link latency, bandwidth, or transfer duration.
Configured compute delay is also separate from contact availability.

### Runtime wiring and limits

The [200-trainer export](../exports/fmow_fedavg_n200.yaml) selects
`trainer.availability.mode: satellite` and sets aggregator
`sim_unavailability: 'True'` and `availability_trace: satellite`.
Both roles load `availability.trace_path` through their `fmow_config_path`.
The reader maps satellite index 0 to `trainer_001`; the aggregator resolves
endpoint IDs through the trainer registry. Missing trace/registry entries fail
rather than defaulting to available. The reader does not repeat the schedule.

The [smoke export](../exports/fmow_fedavg_smoke_n10.yaml) uses synthetic `syn_0`
instead of the geometric trace.

The generator uses sample indices as seconds and truncates `min_window` to an
integer sample count. Use one-second samples starting at zero. Capture generation
uses `time_s`; trainer position lookup also assumes one-second indexing.
`min_window` trims the end of each station's visible interval. With the default
90 seconds, that final portion is ineligible even if the satellite remains visible.
Every usable interval ends with `UN_AVL`, including at the trace horizon.
Setting elevation to `-90` and minimum window to `0` covers all input samples,
but does not create an infinite trace.

## Fields and defaults

Paths below resolve from the repository root, regardless of config location or
working directory. Absolute and `~` paths also work. The CLI `--config` argument
follows normal command-line path rules.

Omitted fields use the [schema defaults](../setup/config.py) below; `{}` is a
valid config.

#### Dataset

- `root_dir`: Download destination and runtime dataset directory.
  Default: `lib/python/examples/fmow/data/fmow`.
- `num_classes`: Number of model output classes. Default: `62`.
- `image_size`: Runtime image transform size in pixels. Default: `224`.

#### Satellites

- `leo_dir`: Directory containing orbital inputs, captures, and the station file.
  Default: `lib/python/examples/fmow/metadata/leo`.

Changing `leo_dir` does not update `availability.trace_path` or the trainer's
`satellite_coordinates_path` hyperparameter; align them explicitly.

#### Capture

- `radius`: Image capture radius in kilometers. Default: `15.0`.
  This is independent of ground-station contact.

#### Ground stations

- `num_stations`: Positive station count, with coordinates sampled uniformly
  over the sphere. Default: `8`; the checked-in config uses `1`.
- `seed`: Station placement seed. Default: `0`.
- `elevation_angle`: Minimum elevation for visibility, in degrees. Default: `10.0`.
- `min_window`: Required remaining visibility at a station, in seconds. Default: `90.0`.

Omit this section or set it to `null` to skip station and availability generation.
Use `ground_stations: {}` to enable both with defaults.

#### Availability

- `trace_path`: Output file for the generated availability trace and input file for runtime
  availability. Default:
  `lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml`.

Omit this section, set it to `null`, or omit `trace_path` to use the default path.
A path alone does not enable generation.

For commands and required orbital array shapes, see [Getting started](getting-started.md).
