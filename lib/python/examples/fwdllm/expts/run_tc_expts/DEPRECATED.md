# DEPRECATED (2026-06-25)

`run_text_classification.sh` and `launch_single_run.py` directly invoke
`trainer/fl_main.py` / `aggregator/fl_main.py` with hand-rolled `envsubst`
JSON templates from `json_scripts/`. These entrypoints are superseded by
`trainer/main.py` and `aggregator/main_fedfwd_agg.py`, which take
`--config-json` from the launcher instead of a templated file.

Use the YAML-based launcher instead:

    python -m flame.launch.run_experiment \
        lib/python/examples/fwdllm/expt_scripts/<experiment>.yaml

Baselines (selector + optimizer + FedFwd hyperparameters) live in
`examples/_metadata/baselines.yaml`; experiments declare `baseline: <name>`
and override only what they need.

## `json_scripts/` is no longer load-bearing (as of Phase 7 step P3)

`flame/mode/horizontal/syncfl/fwdllm_aggregator.py:read_trainer_unavailability()`
previously hardcoded a glob over `json_scripts/trainer_*.json` to build the
oracular trainer-availability event dict. Phase 7 step P3 rewrote it to read
`_metadata/trainer_registry.yaml` + `_metadata/availability_traces/*.yaml`
instead (mirroring `async_cifar10/aggregator/pytorch/main_oort_sync_agg.py`'s
pattern), so the `fwdllm_plus`/oracular baseline no longer depends on this
directory. `json_scripts/` is now fully superseded and safe to delete
whenever someone cleans up this directory -- it has been left in place here
only because deleting it wasn't itself part of Phase 7's scope, not because
anything still reads it.
