# Prepare and run FMoW

Run commands from the repository root. For training, first prepare the GPU
environment and MQTT broker using the [Python setup guide](../../../README.md).

## Inputs

Setup needs `numpy`, `PyYAML`, `pandas`, and `scipy`; downloading uses
`bash`, `curl`, and `tar`. Allow space for the roughly 54 GB dataset archive
plus extracted files.

Edit [fmow_config.yaml](../configs/fmow_config.yaml) and place these existing
orbital files in `satellites.leo_dir`. Setup does not generate orbits.

| File | Required contents | Used for |
| --- | --- | --- |
| `geodetic.npz` | `coords`: latitude/longitude degrees, shape `(timesteps, satellites, 2)`; `time_s`; `sat_names`. | Image captures and trainer positions. |
| `ecef.npz` | `ecef_km`: Earth-centered coordinates in km, shape `(timesteps, satellites, 3)`; `time_s`. | Ground-station contacts. |

Both files must use the same satellite order. Supply one-second samples starting
at zero: the availability generator uses sample indices as seconds.
See [configuration and availability](config-reference.md) for defaults and timing limits.

## Generate inputs

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all
```

Append `--config /path/to/fmow_config.yaml` to use another config. With no
command, setup defaults to `all`.

| Step, or standalone command | Inputs and output |
| --- | --- |
| `download` | Downloads/extracts FMoW. Skips if both metadata CSVs and `images/` exist; this checks presence, not integrity. |
| `captures` | Uses `rgb_metadata.csv` columns `split`, `lat`, and `lon`, plus `geodetic.npz`; overwrites `leo_dir/captures.npz`. |
| `ground-stations` | Uses configured station count and seed; overwrites `leo_dir/ground_stations.yaml`. |
| `availability` | Uses that station file, `ecef.npz`, elevation, and minimum window; overwrites `availability.trace_path`. |

`all` runs these steps in order. Omitting `ground_stations` skips the last two
and preserves their existing files. Standalone steps do not run prerequisites.
A failure stops setup; outputs from earlier steps remain.
The `fmow/metadata` symlink points to shared `examples/_metadata`.

To keep manually specified stations, run only `availability`. To regenerate
stations and contacts without touching captures, run `ground-stations` followed
by `availability`. Both commands require a `ground_stations` config section.
Setup generates outputs for all satellites in the orbital files.

## Launch

Setup does not create the trainer registry, configure the broker, or launch training.
Check GPU counts and paths in the chosen export before running:

```bash
python -m flame.launch.run_experiment \
  lib/python/examples/fmow/exports/fmow_fedavg_smoke_n10.yaml
```

The smoke config uses 10 trainers, two GPUs, 50 rounds, and synthetic `syn_0`.
The [full export](../exports/fmow_fedavg_n200.yaml) uses 200 trainers, eight GPUs,
500 rounds, and `satellite` availability; use its path to run that experiment.

Both roles must point to the same `fmow_config_path`. A custom orbital directory
also requires updating the trainer's `satellite_coordinates_path`. Every traced
satellite needs a registry entry. The [coverage tool](data-coverage.md) can check
data reachability before training; the [fix roadmap](fmow-fix-roadmap.md) lists
known runtime concerns.
