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

## `json_scripts/` is still load-bearing for ORACULAR tracking

`flame/mode/horizontal/syncfl/fwdllm_aggregator.py:read_trainer_unavailability()`
hardcodes a glob over `json_scripts/trainer_*.json` to build the oracular
trainer-availability event dict. Until that function is rewritten to read
`_metadata/trainer_registry.yaml` + `_metadata/availability_traces/*.yaml`
(see the `fedfwd_oracular` baseline's description in `baselines.yaml`), do
NOT delete `json_scripts/` -- only the shell/Python launch scripts in this
directory are deprecated. The JSON config files remain a live dependency.
