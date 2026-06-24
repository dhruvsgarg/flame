# Migrate `fwdllm` onto the `flame.launch` YAML launcher

This document is the living runbook for migrating `fwdllm` onto
`flame.launch.run_experiment`, per `lib/python/examples/MIGRATING_TO_LAUNCHER.md`.
It is kept up to date (Progress line + checkboxes) as each step's checkpoint
passes — read the Progress line and checklist below before resuming work on
this migration in any new session.

## Progress: 0 / 15 steps complete

- [ ] 1. Fix `_validate_stack` regex/`_ASYNC_STACKS` for `fwdllm_aggregator` (Phase 1a)
- [ ] 2. Add `DatasetConfig.path_style` + `skip_index_splits` plumbing (Phase 1b)
- [ ] 3. Add `TrainerConfig.client_idx_modulo` + per-trainer override path (Phase 1b)
- [ ] 4. Fix `_sweep_stragglers()` pattern list for fwdllm paths (Phase 1c)
- [ ] 5. Create `fwdllm/metadata -> ../_metadata` symlink (Phase 2d)
- [ ] 6. Create `fwdllm/configs/trainer_base.yaml` (Phase 2c)
- [ ] 7. Create `fwdllm/trainer/main.py` (Phase 2a)
- [ ] 8. Fix mobiperf trace-name mismatch in `FedSgdTrainer.py` (Phase 2a)
- [ ] 9. Add `trainer_round` telemetry emission in `FedSgdTrainer.py` (Phase 2a)
- [ ] 10. Create `fwdllm/aggregator/main_fedfwd_agg.py` (Phase 2b)
- [ ] 11. Add `fedfwd_async_random_dynkc` + `fedfwd_oracular` to `_metadata/baselines.yaml` (Phase 3)
- [ ] 12. Create `fedfwd_async_random_n10_smoke.yaml` (Phase 4)
- [ ] 13. Add `expts/run_tc_expts/DEPRECATED.md` (Phase 5)
- [ ] 14. Static + dry-run verification (Verification §1, §2)
- [ ] 15. Live 10-trainer smoke test + parity check vs. legacy script (Verification §3, §4)

## Checkpoint rules

- **One step at a time.** Do not start step *N+1* until step *N*'s checkpoint
  has passed.
- **Checkpoint after every step**, before checking its box:
  - Code-change steps (1–4, 7–11): the file imports/parses cleanly, and where
    applicable, the specific behavior the step claims to fix is exercised once
    (e.g. after step 1, re-run the `_validate_stack` repro from Verification
    §1 and confirm it now passes instead of raising).
  - New-file steps (5, 6, 12, 13): the file exists at the right path with the
    right shape (symlink resolves; YAML parses via `yaml.safe_load`).
  - Verification steps (14, 15): the run actually executed and its specific
    pass/fail criteria (Verification section below) were checked, not just
    "command exited 0."
- **On checkpoint failure**: stop, fix the issue, re-run the same checkpoint —
  never proceed to the next step with a known-failing checkpoint.
- **After each passing checkpoint**, update both the `Progress: N / 15 steps
  complete` line and that step's checkbox in the same edit, so the file is
  never out of sync with actual progress.
- This file is the source of truth for resuming work across sessions — before
  starting any work on this migration, read its Progress line and checklist
  first rather than assuming where things left off.

---

## Context

`fwdllm` (FedFwd text-classification on agnews/DistilBERT) is the last unmigrated
example per `lib/python/examples/MIGRATING_TO_LAUNCHER.md` §9/§Status table. It
still runs through hand-rolled JSON configs + `expts/run_tc_expts/run_text_classification.sh`
(`envsubst` templating, manual `pkill`/background-process orchestration, log-grep
accuracy watchdog). The goal is to bring it onto `flame.launch.run_experiment`
like `async_cifar10`, reusing the shared `_metadata/` bundle, baselines catalog,
and telemetry/snapshot infrastructure — and to surface exactly which parts of
this migration have **no precedent** in the async_cifar10 reference migration,
since fwdllm's architecture differs from cifar10's in ways the existing doc's
§9 only partially anticipated.

Two corrections to MIGRATING_TO_LAUNCHER.md §9, confirmed by direct code reading:

1. §9's aggregator table proposes splitting into per-stack files
   (`main_asyncfl_agg.py`, `main_oort_sync_agg.py`) mirroring async_cifar10.
   **This is wrong for fwdllm.** `aggregator/FedSgdAggregator.py` extends
   `flame.mode.horizontal.syncfl.fwdllm_aggregator.TopAggregator` — a single
   FedFwd-specific class (dynamic_kc controller, JVP/forward-mode variance
   gating, `max_iterations_per_data_id`), not the generic swappable
   `asyncfl`/`syncfl`/`oort` hierarchy. There is no second stack to split into.
   **fwdllm gets exactly one aggregator entrypoint.**
2. `flame/launch/runner.py:_validate_stack()` (line 381) regex-matches
   `r"from flame\.mode\.horizontal\.(\w+)\.top_aggregator import"` against the
   aggregator main file's text, defaulting to `stack="syncfl"` on no match.
   Confirmed via direct `re.search` test: this **does not match**
   `from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator`
   (module name is `fwdllm_aggregator`, not `top_aggregator`). fwdllm's existing
   selector is `async_random` (in `_ASYNC_SELECTORS`), so with the default
   `stack="syncfl"` fallback, `_validate_stack` raises `ValueError` **before any
   process spawns** — a real, confirmed blocker, not a hypothetical.

---

## Phase 1 — Launcher-side fixes (`flame/launch/*`) — must land first

### 1a. Fix `_validate_stack` to recognize fwdllm's aggregator module

File: `flame/launch/runner.py`, `_validate_stack()` (line 381) + `_ASYNC_STACKS` (line 378).

Detect `fwdllm_aggregator` as its own stack key (not folded into `"syncfl"`),
and add it to `_ASYNC_STACKS` — honest about FedFwd's aggregation being
event-driven/variance-gated (legitimately async-compatible), and isolated from
the existing `top_aggregator` regex so no other example's matching changes:

```python
_ASYNC_STACKS = {"asyncfl", "coord_asyncfl", "fwdllm"}

def _validate_stack(self, agg_main_path, agg_cfg):
    text = Path(agg_main_path).read_text()
    if re.search(r"from flame\.mode\.horizontal\.\w+\.fwdllm_aggregator import", text):
        stack = "fwdllm"
    else:
        m = re.search(r"from flame\.mode\.horizontal\.(\w+)\.top_aggregator import", text)
        stack = m.group(1) if m else "syncfl"
    ...
```

**Checkpoint:** re-run the static repro from Verification §1 — must raise
`ValueError` before this fix, must pass after.

### 1b. Per-trainer `client_idx` injection — explicit field, not eval-based formulas

Confirmed blockers in `flame/launch/spawner.py`:
- `ConfigGenerator.generate_trainer_config()` (line 93) unconditionally calls
  `self.metadata.get_dataset_split(...)` (line 121), which does
  `self.dataset_splits[split_key][...]` (spawner.py:60) — a hard `KeyError` for
  any dataset with no `dataset_splits/<name>_alpha<a>_n<N>.yaml` file. fwdllm's
  H5-path dataset has no such file and never will.
- The override-application loop (lines 166–177) does plain static dict
  assignment only — no per-trainer computation is possible today. Confirmed by
  reading the loop directly: every `value` is copied identically to all trainers.

**Design choice:** rather than the more general but riskier route of an
eval-evaluated formula string in YAML, add two narrow, explicit fields:

- `DatasetConfig.path_style: bool = False` (`flame/launch/experiment_config.py`,
  `DatasetConfig` at line 14) — when `True`, skip the index-split lookup
  entirely.
- `TrainerConfig.client_idx_modulo: Optional[int] = None` — when set, the
  runner computes `client_idx = (trainer_id - 1) % client_idx_modulo` per
  trainer and injects it as a per-trainer dotted-key override, instead of a
  generic expression language. This is declarative, trivially testable, and
  covers the actual need (wrap N trainers onto M data partitions) without
  introducing `eval()` over YAML-sourced strings.

Implementation:
- `spawner.py: generate_trainer_config()` — gate the `get_dataset_split()` call
  (and the `trainer_indices_list` injection) behind a new `skip_index_splits:
  bool = False` parameter. **(Step 2 checkpoint: dry-run call with
  `skip_index_splits=True` on a fake dataset name does not `KeyError`.)**
- `spawner.py: TrainerSpawner.spawn_trainer()`/`spawn_all()` — thread
  `skip_index_splits` through to `generate_trainer_config()`.
- `runner.py: run_experiment()` — where `config_overrides` is assembled
  (line ~239, right where `"job.id"`/`"hyperparameters.training_delay_enabled"`
  are set today): if `exp.trainer.client_idx_modulo` is set, loop per
  `trainer_id` and call `generate_trainer_config(..., **{**config_overrides,
  "hyperparameters.client_idx": (trainer_id - 1) % exp.trainer.client_idx_modulo})`
  individually instead of the current single `spawn_all(...)` call that passes
  identical overrides to every trainer — i.e. `TrainerSpawner.spawn_all()` needs
  a path where the per-trainer override dict can vary by `trainer_id` (a small
  signature change: accept an optional `per_trainer_overrides: Dict[int, Dict]`
  merged on top of the shared `**overrides` for that specific trainer_id).
  **(Step 3 checkpoint: matches Verification §2 — 10 distinct client_idx
  values 0–9, no `trainer_indices_list` key present.)**
- Pass `skip_index_splits=exp.trainer.dataset.path_style` through the same
  call chain (`run_experiment()` → `spawn_all()` → `spawn_trainer()` →
  `generate_trainer_config()`).

### 1c. Fix `_sweep_stragglers()` path patterns

File: `flame/launch/runner.py:_sweep_stragglers()` (line 487). Confirmed
hardcoded patterns `("trainer/pytorch/main.py", "aggregator/pytorch/main_")`
(line 494) and the GPU-drain poll's `pgrep -f "trainer/pytorch/main.py"`
(line 505) — fwdllm's new entrypoints have no `pytorch/` subdirectory. Extract
the pattern list to a class-level tuple and add fwdllm's two entrypoint paths
(decided in Phase 2: `trainer/main.py`, `aggregator/main_fedfwd_agg.py`); widen
the `pgrep` call to loop over the same pattern list rather than a single
hardcoded string.

**Checkpoint (step 4):** start a dummy background process whose argv contains
`fwdllm/trainer/main.py`, call `_sweep_stragglers()`, confirm it's killed.

---

## Phase 2 — fwdllm-side changes

New layout (additive — nothing deleted yet):
```
lib/python/examples/fwdllm/
├── MIGRATION_TO_LAUNCHER_FWDLLM.md           # this document
├── metadata -> ../_metadata                  # NEW symlink
├── configs/trainer_base.yaml                 # NEW
├── trainer/main.py                           # NEW launcher entrypoint
├── aggregator/main_fedfwd_agg.py             # NEW, single entrypoint (no per-stack split)
└── expt_scripts/fedfwd_async_random_n10_smoke.yaml   # NEW
```
`trainer/fl_main.py` and `aggregator/fl_main.py` are kept (manual `--config
<file>` debugging path, per the doc's "Keep" list), not deleted.

### Step 5. Metadata symlink
`ln -s ../_metadata lib/python/examples/fwdllm/metadata` (mirrors
`async_cifar10/metadata` exactly). **Checkpoint:** `readlink -f` resolves to
`examples/_metadata`.

### Step 6. `configs/trainer_base.yaml`
Static template with FedFwd's full hyperparameter set (dataset/model/FL/
availability blocks — transcribed from `json_scripts/trainer_1.json` and
`aggregator.json`), placeholders for `data_file_path`/`partition_file_path`
(injected via `config_overrides`, not `_metadata/dataset_splits/`) and
`client_idx` (injected per-trainer via Phase 1b's `client_idx_modulo`
mechanism). Include an inline comment documenting the mobiperf trace-name
wrinkle (step 8). **Checkpoint:** `yaml.safe_load()` parses it; all keys from
the existing `trainer_1.json`'s `hyperparameters` block are present.

### Step 7. `trainer/main.py`
Mirror `async_cifar10/trainer/pytorch/main.py`'s `main()` shape, but reuse all
existing fwdllm wiring (model/data-manager/preprocessor construction) unchanged
from `trainer/fl_main.py`:
- Replace `argparse --config (required)` + `Config(args.config)` with
  `from flame.launch.cli import load_config_from_argv; config =
  load_config_from_argv()`.
- Add a side-channel `--time_mode` parser (`add_help=False`,
  `parse_known_args()`, safe alongside `load_config_from_argv()`'s own
  `parse_known_args()`). FedFwd has **no simulated-clock concept**
  (`FedSgdTrainer.py`'s `_emulate_training_delay()` always sleeps real
  wall-clock time) — accept `--time_mode` only so the launcher's unconditional
  `--time_mode <value>` argv injection (`spawner.py` line ~283) doesn't crash
  with "unrecognized argument"; log+ignore if `simulated` is passed. Document
  this as deferred future work, not retrofitted here.
- Keep the existing `notify_trainer_avail` thread spawn (already implemented,
  no change).
**Checkpoint:** `python trainer/main.py --config-json '{...minimal...}' --time_mode real` parses argv without error and reaches model construction (does not need to complete a full round).

### Step 8. Fix mobiperf trace-name mismatch
`FedSgdTrainer.py`'s `client_notify["trace"]` dispatch matches long-form
strings (`"avl_events_mobiperf_2st"`) for mobiperf but the spawner's
`availability_mode`/`client_notify.trace` convention uses short names
(`"mobiperf_2st"`) — patch the trainer's match to accept both forms (one-line
fix, removes a footgun for future experiment authors).
**Checkpoint:** unit-test both `"mobiperf_2st"` and `"avl_events_mobiperf_2st"`
resolve to the same trace lookup.

### Step 9. Telemetry: `trainer_round`
Add `telemetry.configure(role="trainer", end_id=str(task_id))` at startup in
`trainer/main.py`. Emit `trainer_round` events (via
`flame.telemetry.events.build_trainer_round`) from inside `FedSgdTrainer.py`'s
per-round completion path (locate the exact call site during implementation —
near where loss/accuracy become available for the variance check), gated by
`telemetry.is_enabled()`.
**Checkpoint:** with `FLAME_TELEMETRY_DIR` set to a temp dir, one training
round produces a `trainer_<id>.jsonl` line with `event: trainer_round`.

### Step 10. `aggregator/main_fedfwd_agg.py`
Single entrypoint (see Phase-1 rationale). Mirror `main_asyncfl_agg.py`'s
`__main__` shape:
```python
parser.add_argument("--log_to_wandb", action="store_true")
parser.add_argument("--wandb_run_name", type=str)
args, _ = parser.parse_known_args()
config = load_config_from_argv()
# ... existing fl_main.py model/data wiring, unchanged ...
aggregator = FedSGDAggregator(...)
if args.log_to_wandb:
    initialize_wandb(run_name=args.wandb_run_name)   # already defined in fl_main.py, just gate the call
telemetry.configure(role="aggregator", end_id=config.job.job_id)
aggregator.compose(); aggregator.run()
```
`initialize_wandb()` already exists in `aggregator/fl_main.py` but is never
called unconditionally — gating is additive, not a removal.
**Checkpoint:** running with `--config-json` and no `--log_to_wandb` does not
import/call `wandb.init()` (verify by stubbing/monkeypatching `wandb.init` to
raise, confirm it's never hit).

---

## Phase 3 — Step 11: `_metadata/baselines.yaml` additions

One primary baseline, `fedfwd_async_random_dynkc`, transcribed faithfully from
the **existing** `json_scripts/aggregator.json` (selector `async_random` +
`dynamic_kc` block with `k_min/k_max: 5/15`, `c_min/c_max: 15`, optimizer
`fedbuff`, `var_threshold: 0.3`, `max_iterations_per_data_id: 15`) — this is
the current production default, ported as-is, not redesigned.

A second, clearly-flagged `fedfwd_oracular` baseline (async_oort selector +
ORACULAR tracking) carries an explicit warning in its `description:` field:
`flame/mode/horizontal/syncfl/fwdllm_aggregator.py:read_trainer_unavailability()`
(line 480) hardcodes a glob over
`examples/fwdllm/expts/run_tc_expts/json_scripts/trainer_*.json` to build the
oracular event dict — a **library-level**, not example-level, dependency on the
legacy JSON directory. This baseline only stays correct while `json_scripts/`
remains on disk; rewriting `read_trainer_unavailability()` to read
`_metadata/trainer_registry.yaml` + `_metadata/availability_traces/*.yaml`
(mirroring `main_oort_sync_agg.py`'s pattern for cifar10) is flagged as a
follow-up, not in the critical path for the primary (`client_notify`-based,
non-oracular) baseline.

**Checkpoint:** `yaml.safe_load()` parses `baselines.yaml` with both new keys
present; `load_baselines()` returns them; no existing baseline entries changed
(diff-check).

---

## Phase 4 — Step 12: smoke-test experiment YAML

`lib/python/examples/fwdllm/expt_scripts/fedfwd_async_random_n10_smoke.yaml`:
10 trainers, `baseline: fedfwd_async_random_dynkc`, `trainer.dataset.path_style:
true`, `trainer.client_idx_modulo: 100` (gives client_idx 0–9, a valid, distinct
subset of the 100 H5 partitions), `time_mode: real`, scaled-down selector
kwargs (`c`, `aggGoal`, `minInitialTrainers`, `dynamic_kc.{k,c}_{min,max}`) to
fit 10 trainers, `rounds: 50` for a short run, `data_file_path`/
`partition_file_path` injected via `config_overrides.hyperparameters`.

**Checkpoint:** `load_experiment_config()` parses it into one `ExperimentConfig`
with `num_trainers == 10` and `baseline == "fedfwd_async_random_dynkc"`.

---

## Phase 5 — Step 13: legacy decommission

Add `expts/run_tc_expts/DEPRECATED.md` pointing at the launcher path (mirrors
the existing pattern in async_cifar10's deprecated dirs), explicitly stating:
**do not delete `json_scripts/*.json`** — only `run_text_classification.sh`
and `launch_single_run.py` are deprecated. The JSON files remain a load-bearing
dependency of `read_trainer_unavailability()` (Phase 3) until that function is
rewritten. This is a deliberate deviation from the doc's normal "delete legacy
JSON dirs" checklist step, and should be called out in review.

**Checkpoint:** file exists; `json_scripts/*.json` files still present and
unmodified (diff-check against pre-migration state).

---

## New aspects required for fwdllm that are NOT present in async_cifar10's migration

1. **Custom FedFwd-specific `TopAggregator`/`Trainer` base classes** (not the
   generic swappable `asyncfl`/`syncfl`/`oort` hierarchy) — forces the
   `_validate_stack` regex/stack-set fix (step 1) and a single aggregator
   entrypoint instead of one-per-stack.
2. **Per-trainer `client_idx` injection** computed from `trainer_id`, vs.
   cifar10's static index-list lookup — no existing spawner mechanism supports
   this; requires the new `client_idx_modulo` field + per-trainer override path
   (steps 2–3).
3. **H5 path-style dataset** vs. index-based — `generate_trainer_config()`
   unconditionally `KeyError`s on path-style datasets; requires the new
   `skip_index_splits`/`path_style` escape hatch (step 2).
4. **FedFwd-specific hyperparameters with no cifar10 analog**: `var_threshold`
   (model-type-dependent default), `dynamic_kc` (adaptive K/C controller),
   `max_iterations_per_data_id`, `forward_mode`/`perturbation_sampling`/
   `select_perturbation_using_jvp` (JVP forward-mode gradient estimation),
   `peft_method`/`use_adapter`/`freeze_layers`/`fp16` — all newly transcribed
   into `baselines.yaml` and `trainer_base.yaml` (steps 6, 11).
5. **Single aggregator entrypoint** instead of per-stack split — direct
   consequence of #1; corrects MIGRATING_TO_LAUNCHER.md §9's guidance.
6. **No simulated-clock support** — FedFwd always uses real wall-clock delay;
   `--time_mode simulated` is accepted but a documented no-op for now (step 7).
7. **CPU-pinning perf risk for transformer workloads** — cifar10's CNN trainer
   tolerates 1-core pinning; fwdllm's DistilBERT/forward-mode training is much
   heavier per-trainer. No code change required, but flag for smoke-test
   wall-clock observation (step 15) before scaling to n=150.
8. **`_sweep_stragglers()` path-pattern mismatch** — fwdllm's entrypoints have
   no `pytorch/` subdirectory; requires the pattern-list extension (step 4).
9. **Library-level legacy path dependency** — `read_trainer_unavailability()`
   globs `json_scripts/trainer_*.json` directly inside
   `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` (not example-side code);
   constrains the Phase 5 decommission scope in a way async_cifar10 never had
   to deal with (its equivalent method is a stubbed no-op).
10. **Trace-name string-format mismatch** for mobiperf traces (long-form
    `"avl_events_mobiperf_2st"` expected by `FedSgdTrainer.py` vs. short-form
    `"mobiperf_2st"` used by the spawner/cifar10 convention) — one-line fix
    (step 8), no cifar10 equivalent since its schema and trace values were
    co-designed.

---

## Verification (Steps 14–15)

1. **Static, no spawning**: load the smoke YAML, resolve `fedfwd_async_random_dynkc`,
   build the merged aggregator config, and call `_validate_stack()` directly —
   must raise `ValueError` *before* the step-1 fix (reproducing the blocker)
   and pass *after*.
2. **Dry-run config generation**: call `ConfigGenerator.generate_trainer_config()`
   for trainer_id 1–10 with `skip_index_splits=True` and the modulo-based
   `client_idx` override; assert 10 distinct values 0–9 and that
   `trainer_indices_list` is absent.
3. **10-trainer live smoke test**: `python -m flame.launch.run_experiment
   lib/python/examples/fwdllm/expt_scripts/fedfwd_async_random_n10_smoke.yaml`.
   Check the aggregator/trainer logs run cleanly for several rounds, that
   `telemetry/trainer_*.jsonl` contain `trainer_round` events, and that
   `aggregator_config.json` shows the expected merged `selector.sort`,
   `dynamic_kc`, `var_threshold` values.
4. **Parity check vs. legacy**: run `run_text_classification.sh` with a
   matching 10-trainer subset and compare round-over-round loss/accuracy trend
   against the new launcher run (exact bit-parity not expected, but trend
   should match within the same rounds) — specifically confirm `client_idx =
   (trainer_id-1) % 100` reproduces the same H5 partition assignments as
   `json_scripts/trainer_1.json…trainer_10.json`'s hardcoded `client_idx`
   values.

### Critical files
- `flame/launch/runner.py` (`_validate_stack`, `_sweep_stragglers`, `run_experiment`'s `config_overrides` assembly)
- `flame/launch/spawner.py` (`ConfigGenerator.generate_trainer_config`, `TrainerSpawner.spawn_trainer`/`spawn_all`)
- `flame/launch/experiment_config.py` (`DatasetConfig`, `TrainerConfig`)
- `examples/_metadata/baselines.yaml`
- `examples/fwdllm/aggregator/FedSgdAggregator.py`, `examples/fwdllm/trainer/forward_training/FedSgdTrainer.py`
- `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` (`read_trainer_unavailability`)
