# Estimate training-data coverage

[data_coverage.py](../scripts/data_coverage.py) counts unique captured and
reachable training images without running training or writing output files.

After [setup](getting-started.md), run from the repository root:

```bash
python lib/python/examples/fmow/scripts/data_coverage.py --num-satellites 200
```

This selects satellite indices 0 through 199 for the current full experiment.
Without `--num-satellites`, the script counts entries in
`fmow/metadata/trainer_registry.yaml`, currently 300. It ignores the registry's
`num_trainers` header and does not read the experiment YAML. Inputs must contain
at least the requested count.

Append `--config /path/to/fmow_config.yaml` for another config. After changing
capture radius or station settings, regenerate the corresponding inputs before
analysis. Use separate output paths to preserve multiple scenarios.

## Inputs and results

The script needs `numpy` and `PyYAML` and reads:

| Input | Purpose |
| --- | --- |
| `satellites.leo_dir/captures.npz` | Capture times per image and satellite. |
| `satellites.leo_dir/geodetic.npz` | Satellite ordering and reporting timeline. |
| `availability.trace_path` | Saved contact events. |
| `dataset.root_dir/rgb_metadata.csv` | Training-image IDs and percentage denominator. |

It reports every 30 minutes and at the final orbital time.

| Column | Meaning |
| --- | --- |
| Captured | Unique training images captured before the reporting time. |
| Reachable | Captured images with a usable contact on any capturing satellite at or after capture, before the reporting time. |
| No contact yet | Captured images with no such contact yet. |

Images remain eligible after capture while a satellite waits for contact.
Duplicates count once globally; a later capture by another satellite can offer
an earlier contact. Events exactly at a reporting boundary count next interval.

Reachable coverage is an optimistic upper bound: it excludes training and
transmission delay, selection, and aggregation. It does not measure accepted
updates or accuracy. The script prints the saved trace description, but captures
do not store their generation radius, so it cannot verify that saved inputs
match a changed config.
