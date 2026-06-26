# Deprecated / safe-to-delete files (fwdllm)

Manifest of files superseded by the `flame.launch` YAML migration (see
`MIGRATION_TO_LAUNCHER_FWDLLM.md`) and the single-env dependency reconciliation
(Phase 8). **Nothing here is deleted yet** — this is the checklist for a
dedicated cleanup commit. Paths are relative to
`lib/python/examples/fwdllm/`.

## A. Stale dependency snapshot

- `req.txt` — a full `pip freeze` dump, not an essential-deps list. Its
  `adapter-transformers==3.1.0` / `tokenizers==0.12.1` pins don't build on
  modern toolchains and are not needed. Superseded by the `[examples]` extra
  in `lib/python/setup.py` (modern `transformers` + the standalone `adapters`
  add-on). Already headed with a DEPRECATED banner.

## B. Legacy launch path (`expts/run_tc_expts/`)

Superseded by `python -m flame.launch.run_experiment <expt_scripts/*.yaml>`.
See `expts/run_tc_expts/DEPRECATED.md` for the rationale.

- `expts/run_tc_expts/run_text_classification.sh` — legacy `envsubst` +
  background-process launcher invoking `trainer/fl_main.py` /
  `aggregator/fl_main.py`.
- `expts/run_tc_expts/launch_single_run.py` — legacy single-run launcher.
- `expts/run_tc_expts/run_three_parallel.sh` — orchestrates `launch_single_run.py`
  (dead once it's gone).
- `expts/run_tc_expts/run_three_experiments.sh` — orchestrates the legacy
  launchers (dead once they're gone).
- `expts/run_tc_expts/fedavg_main_tc.py` — legacy non-FedFwd entrypoint.
  **Verify** nothing imports it before deleting.
- `expts/run_tc_expts/PARALLEL_RUN_DESIGN.md` — design notes for the deprecated
  parallel runner (currently untracked in git).
- `expts/run_tc_expts/mpi_host_file` — stale runtime artifact (single hostname).

## C. Legacy hand-rolled configs

- `expts/run_tc_expts/json_scripts/` (155 files) — hand-rolled
  `trainer_*.json` / `aggregator*.json` templates. No longer load-bearing as
  of Phase 7 P3 (`read_trainer_unavailability()` now reads `_metadata/`).
  **CAVEAT:** `MIGRATION_TO_LAUNCHER_FWDLLM.md` (Status & P8) references these
  for an *optional* `client_idx` parity check. Before deleting, either run that
  parity check, or drop the two `json_scripts/trainer_*.json` references from
  the migration doc in the same commit.

## D. Runtime / cache artifacts (optional cleanup, not source)

Regenerated on each run; safe to remove but not "deprecated code" per se.

- `expts/run_tc_expts/cache_dir/`
- `expts/run_tc_expts/log/`
- `expts/run_tc_expts/benchmark-runs-gaurav/` — verify ownership/intent first.

## Do NOT delete (still load-bearing — listed to prevent accidental removal)

- `expts/initializer.py` — on the launcher path (`trainer/main.py` /
  `aggregator/main_fedfwd_agg.py` call `create_model`).
- `trainer/fl_main.py`, `aggregator/fl_main.py` — retained for manual
  debugging (Phase 2); not on the launcher path but intentionally kept.
- `expts/run_tc_expts/gpu_mapping.yaml` — `configs/trainer_base.yaml` still
  references `gpu_mapping_file: gpu_mapping.yaml`. Confirm the launcher's
  resolved path before assuming it's unused.

## Pre-deletion checklist (for the cleanup commit)

- [ ] `grep -rn` the repo (incl. `tests/`, `*.yaml`, `*.sh`) for each path
      above; expect zero references outside the deprecation docs.
- [ ] Resolve the `json_scripts/` ↔ P8 parity-check caveat (section C).
- [ ] Update/remove `expts/run_tc_expts/DEPRECATED.md` and this manifest once
      their subjects are gone (or keep one as a tombstone).
- [ ] Confirm `gpu_mapping.yaml` is genuinely unused before removing it.
