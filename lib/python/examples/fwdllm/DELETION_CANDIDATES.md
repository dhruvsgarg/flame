# fwdllm legacy-code deletion candidates (JSON/MPI launch path — separate deletion PR)

**Durable tracking doc for the deletion PR.** `PR_CLEANUP_PLAN.md` (working notes) was removed; its
still-relevant content folded into `../MIGRATING_TO_LAUNCHER.md` (status pointer:
`MIGRATION_TO_LAUNCHER_FWDLLM.md`). This is the survivor — keep it current; it is the manifest for the
deletion PR.

**Precondition NOW MET.** The launcher migration is merged (`#68` launcher-migration readiness via 3
real n=100 runs, `#72` async simulator + fluxtune opts). The live path is `flame.launch` + the YAML
experiments in `expt_scripts/` (`aggregator/main_fedfwd_agg.py`, `trainer/main.py` →
`load_config_from_argv()`), which imports only `expts/initializer.py` from the legacy tree. The
JSON/MPI launch path below is now fully superseded and safe to delete in a **dedicated follow-up PR**
(kept out of the migration PRs so the launcher had time to prove out in real use — it has).

**Why now (operator ask 2026-07-09):** deprecate the JSON-based running. The stale per-trainer /
per-aggregator JSONs risk a **behaviour regression** if someone launches from them by mistake; there
is no reason to keep a second, divergent config path once the launcher works.

Verified via repo-wide grep: nothing on the live path imports any file below; the only references are
inside the doomed files themselves (or commented out). Deletion is safe.

---

## A. Confirmed safe to delete

### A1. Entire `expts/run_tc_expts/` directory
The whole legacy MPI/JSON launch tree. Contents:
- `json_scripts/` — **155 files**: 5 legacy aggregator variants (`aggregator.json`,
  `aggregator_base.json`, `aggregator_async_base.json`, `aggregator_async_dynk.json`,
  `aggregator_async_maxiter.json`) + `trainer_0.json … trainer_149.json` (150). Superseded exploratory
  variants — only `aggregator.json`→`fluxtune_dynkc` and `aggregator_dynamic_kc.json`→`fluxtune` were
  carried forward into `_metadata/baselines.yaml`; no 5th baseline depends on the rest. **These are the
  "redundant trainer/aggregator JSONs" the deprecation targets.**
- `fedavg_main_tc.py` — legacy argparse+MPI entrypoint (not imported by the live path).
- `launch_single_run.py`, `run_three_experiments.sh`, `run_three_parallel.sh`,
  `run_text_classification.sh` — legacy MPI launch scripts.
- `gpu_mapping.yaml`, `mpi_host_file` — MPI/topology artifacts. `gpu_mapping.yaml` is **vestigial**:
  the only readers are `*/fl_main.py` (commented out) and `initializer.py`'s dead argparse builder;
  the live trainer hard-sets its device (`FedSgdTrainer.py:391`, spawner pin authoritative — K-D33).
- `cache_dir/` — **already gitignored** (253M local data cache, untracked) → no git action.

### A2. Legacy `fl_main.py` entrypoints
- `aggregator/fl_main.py`, `trainer/fl_main.py` — old argparse+MPI-era `Config(args.config)` wiring,
  superseded by `main_fedfwd_agg.py` / `main.py`. Dead via repo-wide grep (only self-references + the
  doomed `run_tc_expts/` scripts + one comment in `main_fedfwd_agg.py:96`). Same origin commit
  `d29a2f7f`; kept in copy/paste lockstep until the migration updated only the live `__main__` blocks.

## B. Coupled cleanups — do IN THE SAME PR so nothing dangles after deletion

- **`configs/trainer_base.yaml`** — remove the now-dead `gpu_mapping_file: gpu_mapping.yaml` +
  `gpu_mapping_key: mapping_myMap` keys (they point at the deleted `run_tc_expts/gpu_mapping.yaml`; no
  live reader — device pinning is the spawner's job, K-D33). Leaving them would be a dangling reference.
- **`expt_scripts/diagnose_partition_binning.py`** — `DEF_CACHE` defaults to
  `…/run_tc_expts/cache_dir`; repoint to the current cache location (or drop the default). Path-only
  default, overridable, points at gitignored data → not a blocker, but tidy it here.
- **`aggregator/main_fedfwd_agg.py:96`** — comment still references the legacy `fl_main.py`; update the
  wording (cosmetic).

## C. Do NOT delete — confirmed still-live dependencies of the new path
- `expts/initializer.py` — imported by `aggregator/main_fedfwd_agg.py` (`set_seed, create_model`) and
  `trainer/main.py`. Its argparse `--gpu_mapping_*` defaults are dead for the live path (the live
  entrypoints call named functions, not the parser) but the module itself is live — **keep**.
- `configs/aggregator_base.json` — **live**: loaded by the current smoke YAMLs
  (`expt_scripts/*_n10_smoke.yaml`) and stamped into run snapshots. NOT a legacy JSON — keep.

## D. Post-deletion check
```bash
cd lib/python/examples/fwdllm
grep -rn "run_tc_expts\|fl_main\|fedavg_main_tc\|gpu_mapping" --include=*.py --include=*.sh --include=*.yaml . | grep -v __pycache__
# expect: only expts/initializer.py's (dead) argparse defaults, if the key-cleanup in §B is deferred
bash expt_scripts/run_sequential.sh --run-set main --only fwdllm --mode real --clean --yes   # smoke: still launches
```
