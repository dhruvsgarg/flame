# Migrate `fwdllm` onto the `flame.launch` YAML launcher

Living runbook for migrating `fwdllm` onto `flame.launch.run_experiment`, per
`lib/python/examples/MIGRATING_TO_LAUNCHER.md`. Read **Status & next step**
below before resuming work in any new session — it is always current.

## Status & next step

**Phases 1–7 are DONE.** The only remaining work is **Phase 7 step P8: live
smoke tests**, blocked on an environment gap (below). Everything else in
this document is historical record, kept for context — skip to it only if
you need to understand *why* something is the way it is.

**Run P8 once you have a host with fwdllm's pinned ML stack installed:**

```bash
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke.yaml
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fwdllm_plus_n10_smoke.yaml
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke.yaml
```

Each covers one of the three owner-spec baselines (10 trainers, ~50 rounds).
Pass criteria for each: aggregator + trainer processes start and don't crash;
≥3–5 rounds complete; `telemetry/trainer_*.jsonl` has `trainer_round` events;
`aggregator_config.json` shows the expected merged `selector.sort` /
`is_async` / `optimizer.sort` for that baseline (see the matrix in Phase 7).
Optional parity check: confirm `client_idx = (trainer_id-1) % 100` reproduces
the same H5 partitions as the legacy `json_scripts/trainer_*.json`.

**The blocker — fwdllm's pinned ML stack is not installable in this sandbox:**
- `req.txt` pins `adapter-transformers==3.1.0`, which pins `tokenizers==0.12.1`.
- `tokenizers==0.12.1`'s Rust source is from ~2022 and trips
  `invalid_reference_casting`, a lint that became **deny-by-default** in
  every `rustc` available here — it fails to compile, full stop.
- This sandbox has mainline `transformers==4.54.0` instead, which lacks the
  adapter API (`AdamW` import, custom adapter classes) that
  `examples/fwdllm/trainer/forward_training/tc_transformer_trainer_distribute.py`
  imports at module load time.
- Effect: **both** the trainer and aggregator entrypoints die at that import
  line, before `__main__` runs — before any argv/config code from this
  migration ever executes. Confirmed pre-existing and not migration-caused:
  the untouched legacy `trainer/fl_main.py` dies at the identical line for
  the identical reason.
- To unblock: a host with either an older `rustc` (predates the
  `invalid_reference_casting` deny-by-default change) or a prebuilt
  `tokenizers==0.12.1` wheel, with `adapter-transformers==3.1.0` actually
  installed (not mainline `transformers`).
- Everything upstream of this wall was still verified: argv/config parsing
  (`load_config_from_argv`, `--time_mode`/`--log_to_wandb` side-channel
  parsing), config generation/merging for all four baselines, and
  `_validate_stack` against the real entrypoint file are all covered by the
  pytest suite (`lib/python/tests/launch`, `lib/python/tests/mode`,
  `lib/python/tests/optimizer`, `lib/python/tests/selector` — 315+ tests,
  all green) and by the three smoke-test YAMLs' static dry-run (also
  pytest-covered, `tests/launch/test_config_generator.py::TestFwdllmSmokeYamlsResolve`).
  P8 is purely an "actually run it on real GPUs with the real ML stack" gap,
  not an unverified-code gap.

## Document conventions

- **Sections are chronological** (oldest first). When you finish new work,
  append a new `## PhaseN` section at the bottom, just before the trailing
  "Reference" appendix — never insert above existing phases.
- **Status tags** at each phase/step heading: `[DONE]`, `[IN PROGRESS]`,
  `[NEXT]`, `[BLOCKED: <reason>]`. Exactly one item in the whole document
  should be `[NEXT]` at a time.
- **Progress counters belong to their own phase**, not a global counter at
  the top of the file — the **Status & next step** section above is the one
  place a reader needs to check for current state.
- Keep entries terse once a phase is done: a sentence or two on what
  shipped and where, not a replay of the investigation that led there.

---

## Context

`fwdllm` (FedFwd text-classification on agnews/DistilBERT) was the last
unmigrated example per `MIGRATING_TO_LAUNCHER.md` §9. It ran through
hand-rolled JSON configs + `expts/run_tc_expts/run_text_classification.sh`
(`envsubst` templating, manual `pkill`/background-process orchestration).
Goal: bring it onto `flame.launch.run_experiment` like `async_cifar10`,
reusing `_metadata/`, the baselines catalog, and telemetry/snapshot infra.

Two corrections to `MIGRATING_TO_LAUNCHER.md` §9, found by direct code
reading:

1. §9 proposes splitting aggregator entrypoints per-stack
   (`main_asyncfl_agg.py`, `main_oort_sync_agg.py`), mirroring
   `async_cifar10`. **Wrong for fwdllm** — `FedSgdAggregator.py` extends
   `flame.mode.horizontal.syncfl.fwdllm_aggregator.TopAggregator`, a single
   FedFwd-specific class (dynamic_kc, JVP/forward-mode variance gating,
   `max_iterations_per_data_id`), not the generic swappable
   `asyncfl`/`syncfl`/`oort` hierarchy. **fwdllm gets exactly one aggregator
   entrypoint.**
2. `runner.py:_validate_stack()` originally regex-matched only
   `from flame\.mode\.horizontal\.(\w+)\.top_aggregator import`, defaulting
   to `stack="syncfl"` on no match — which doesn't match
   `fwdllm_aggregator`'s module name, so it raised `ValueError` before any
   process spawned. (This class of bug recurred later — see Phase 7 step P1.)

---

## Phase 1 — Launcher-side fixes (`flame/launch/*`) [DONE]

- **1a.** `_validate_stack()`: detect `fwdllm_aggregator` as its own stack
  key, added to `_ASYNC_STACKS` (later revised — Phase 7 step P1).
- **1b.** Per-trainer `client_idx` injection without `eval()`-based formulas:
  added `DatasetConfig.path_style: bool` (skips the index-split lookup for
  H5/path-style datasets) and `TrainerConfig.client_idx_modulo: int`
  (runner computes `client_idx = (trainer_id-1) % client_idx_modulo` and
  injects it as a per-trainer override). Threaded through
  `spawner.py`/`runner.py`/`experiment_config.py`.
- **1c.** `_sweep_stragglers()`: fwdllm's entrypoints have no `pytorch/`
  subdirectory, so its pkill/pgrep patterns were extracted to class-level
  tuples (`_STRAGGLER_PATTERNS`, `_TRAINER_STRAGGLER_PATTERNS`) and fwdllm's
  paths added, with no regression to cifar10's patterns.

## Phase 2 — fwdllm-side changes [DONE]

New, additive (nothing legacy deleted): `metadata -> ../_metadata` symlink;
`configs/trainer_base.yaml` (transcribed from `json_scripts/trainer_1.json` +
`aggregator.json`); `trainer/main.py` and `aggregator/main_fedfwd_agg.py`
(launcher entrypoints using `load_config_from_argv()`, replacing the legacy
`--config <file>` argparse path — `trainer/fl_main.py`/`aggregator/fl_main.py`
kept for manual debugging); the mobiperf trace-name short/long-form fix in
`FedSgdTrainer.py`; `trainer_round` telemetry emission at the end of
`train_with_data_id()`'s successful-training path.

All argv/config-parsing logic was verified directly; reaching live model
construction hits the pinned-ML-stack wall described in **Status & next
step** above — confirmed pre-existing, not migration-caused.

## Phase 3 — Original baseline catalog [DONE → superseded by Phase 7 P5]

Added two baselines to `_metadata/baselines.yaml`: `fedfwd_async_random_dynkc`
(transcribed as-is from the legacy `aggregator.json`) and `fedfwd_oracular`
(flagged with a known dependency on legacy `json_scripts/` for availability).
**Both were retired in Phase 7** (`fwdllm`/`fwdllm_plus`/`fluxtune` replace
them; `fedfwd_async_random_dynkc`'s shape survives, renamed, as
`fluxtune_dynkc`). Do not write new experiments against the old names.

## Phase 4 — Original smoke-test YAML [DONE → superseded by Phase 7 P6]

`fedfwd_async_random_n10_smoke.yaml` (10 trainers, `baseline:
fedfwd_async_random_dynkc`). **Renamed/replaced in Phase 7** by
`fwdllm_n10_smoke.yaml` + two siblings — see Phase 7 step P6.

## Phase 5 — Legacy decommission [DONE, partially superseded]

Added `expts/run_tc_expts/DEPRECATED.md`: only `run_text_classification.sh`/
`launch_single_run.py` are deprecated, `json_scripts/*.json` files were kept
on disk (at the time, still load-bearing for `read_trainer_unavailability()`).
**Phase 7 step P3** rewrote that method to read `_metadata/` instead, so
`json_scripts/` is no longer load-bearing — `DEPRECATED.md` was updated
accordingly, but the directory itself was left in place (deleting it wasn't
in scope).

## Smoke Tests A–D [DONE]

| Test | What it checked | Result |
|---|---|---|
| A | `_validate_stack` fix: fwdllm+async selector → no raise; cifar10 regressions still caught | PASSED |
| B | `load_experiment_config()` + `load_baselines()` resolve the Phase 4 smoke YAML | PASSED |
| C | Trainer/aggregator entrypoints parse `--config-json`/`--time_mode` without crashing (crash, if any, must be at import time, not argv-parsing time) | PASSED |
| D | Full static config-generation dry run for 10 trainers: distinct `client_idx` 0–9, no `KeyError`, no `trainer_indices_list` for the path-style dataset | PASSED |

## New aspects required for fwdllm with no `async_cifar10` precedent (reference)

1. Custom FedFwd-specific `TopAggregator`/`Trainer` base classes (not the
   generic `asyncfl`/`syncfl`/`oort` hierarchy) → single aggregator
   entrypoint, special-cased `_validate_stack` detection.
2. Per-trainer `client_idx` from `trainer_id`, not a static index-list
   lookup → `client_idx_modulo` (Phase 1b).
3. H5 path-style dataset → `skip_index_splits`/`path_style` (Phase 1b).
4. FedFwd-only hyperparameters: `var_threshold`, `dynamic_kc`,
   `max_iterations_per_data_id`, `forward_mode`/`perturbation_sampling`/
   `select_perturbation_using_jvp`, `peft_method`/`use_adapter`/
   `freeze_layers`/`fp16`.
5. Single aggregator entrypoint (consequence of #1).
6. No simulated-clock support — `--time_mode simulated` accepted but a
   documented no-op.
7. CPU-pinning perf risk for transformer workloads vs. cifar10's CNN — flag
   for wall-clock observation once live runs are possible, before scaling
   past n=10.
8. `_sweep_stragglers()` path-pattern mismatch (Phase 1c).
9. Library-level legacy path dependency in
   `fwdllm_aggregator.read_trainer_unavailability()` — resolved in Phase 7 P3.
10. mobiperf trace long-form/short-form mismatch (Phase 2) — no cifar10
    equivalent since its schema and traces were co-designed.

---

## Phase 6 — Baseline taxonomy design: fwdllm / fwdllm+ / fluxtune [DONE]

**Owner's spec** (verbatim taxonomy; "fwdllm+" below is the registered key
`fwdllm_plus`):

| Axis | **fwdllm** (syncfl) | **fwdllm+** (syncfl) | **fluxtune** (asyncfl) |
|---|---|---|---|
| Trainer availability | unaware | oracular | 3-tier client_notify |
| Selection | random per round | random per iteration | async_oort |
| Aggregation | federated averaging | federated averaging | fedbuff-based |
| Perturbation | randomly generated | randomly generated | greedy JVP-based |

Owner's note on fluxtune: selection (async_oort) and aggregation (fedbuff)
should be parameterized separately from `felix` (async_cifar10's
async_oort+fedbuff baseline) rather than forked, so the codebase stays clean
across the CNN/speech family (felix etc.) and the LLM forward-mode family
(fwdllm/fwdllm_plus/fluxtune).

**Decisions (D1–D6), each implemented in the matching Phase 7 step:**

- **D1 (→ P1).** `_validate_stack` must key fwdllm's async-ness off the
  selector's `is_async` kwarg, not stack-set membership — the fwdllm
  aggregator itself already dispatches sync vs. async purely on that kwarg
  (`fwdllm_aggregator.py` `is_async`-driven dispatch), so `_validate_stack`
  should agree with it instead of assuming "fwdllm == always async."
- **D2 (→ P2).** `fedbuff.py`'s learning-rate dispatch only knows
  `cifar-10`/`google-speech` (`fedbuff.py:233-256`) — add an explicit
  `learning_rate` kwarg that wins when present, falls back to the existing
  table otherwise. The one real "separate fluxtune from felix" code change;
  felix itself needs no fork.
- **D3 (→ P3).** `fwdllm_aggregator.read_trainer_unavailability()`
  (`fwdllm_aggregator.py:480-489`, pre-Phase-7) hardcoded a glob over legacy
  `json_scripts/trainer_*.json` — rewrite to read
  `_metadata/trainer_registry.yaml` + `_metadata/availability_traces/*.yaml`,
  porting `async_cifar10/aggregator/pytorch/main_oort_sync_agg.py`'s pattern.
  Required for a clean `fwdllm_plus` (ORACULAR availability).
- **D4 (→ P4).** "Random per round" (fwdllm) vs. "random per iteration"
  (fwdllm_plus) has no existing toggle — `_distribute_weights_sync` calls
  `channel.ends(VAL_CH_STATE_SEND)` (re-selecting) on **every** iteration
  today, i.e. per-iteration is the current behavior and per-round is new.
  Add `reselect_each_iteration: bool` (hyperparameter, default `true` =
  unchanged behavior); when `false`, cache the round's first selection and
  reuse it until `self._round` advances.
- **D5.** fluxtune's "3-tier client_notify" = `client_notify.enabled: "True"`
  + a `*_3st_*` trace, with `trackTrainerAvail.enabled: "False"` —
  client_notify is the *only* availability signal (not layered with
  ORACULAR/HEARTBEAT; those are independent mechanisms).
- **D6 (→ P5).** Retire `fedfwd_async_random_dynkc`/`fedfwd_oracular`;
  replace with `fwdllm`/`fwdllm_plus`/`fluxtune`. Preserve
  `fedfwd_async_random_dynkc`'s adaptive-K/C shape as `fluxtune_dynkc` (maps
  to none of the three owner-spec names, but is a real, distinct research
  config worth keeping for parity).

**Addendum, given verbally during Phase 7 implementation:** felix always
runs fixed K/C; fluxtune's `async_oort` must support **both** fixed and
adaptive (`dynamic_kc`) K/C, switched purely by config — see Phase 7 step P5.

---

## Phase 7 — Implementation of Phase 6's taxonomy

**[DONE: P1–P7] [NEXT: P8 — blocked, see Status & next step at the top]**

### Target baseline matrix

| Baseline | stack (`is_async`) | selector | optimizer | availability | perturbation | reselect |
|---|---|---|---|---|---|---|
| **fwdllm** | sync (`false`) | `random` | `fedavg` | unaware | `perturbation_sampling` on, JVP off | per-round (`false`) |
| **fwdllm_plus** | sync (`false`) | `random` | `fedavg` | ORACULAR (`_metadata` trace) | `perturbation_sampling` on, JVP off | per-iteration (`true`) |
| **fluxtune** | async (`true`) | `async_oort` | `fedbuff` (explicit `learning_rate`) | 3-tier `client_notify` | JVP-based | n/a |
| **fluxtune_dynkc** | async (`true`) | `async_random` + `dynamic_kc` | `fedbuff` | off | `perturbation_sampling` on | n/a |

`fluxtune`'s `selector.kwargs.dynamic_kc` defaults to `enabled: false` (fixed
K/C, like felix) but can be flipped to adaptive K/C per experiment —
`DynamicKCController` is selector-agnostic, so no fork was needed.

### P1 — `_validate_stack` selector-driven for fwdllm [DONE]

`flame/launch/runner.py`: removed `"fwdllm"` from `_ASYNC_STACKS`; for
`stack == "fwdllm"`, `is_async_stack` now comes from
`selector_cfg["kwargs"].get("is_async", False)` instead of set membership.
Tests: `tests/launch/test_runner_paths.py::TestValidateStack` (6 cases incl.
a cifar10 regression guard).

**Real bug found and fixed:** `examples/fwdllm/aggregator/main_fedfwd_agg.py`
imports `FedSGDAggregator` (which extends `TopAggregator` in a *separate*
file), not `TopAggregator` directly — so `_validate_stack`'s regex never
matched the real entrypoint and silently fell back to `stack="syncfl"` for
every fwdllm baseline (Phase 1a's fix was only ever verified against
synthetic fixture text, never the real file). Fixed with a same-file marker
import in `main_fedfwd_agg.py` (`from
flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator  # noqa:
F401`, commented as to why). Locked in by
`test_real_fwdllm_entrypoint_detected_for_both_sync_and_async`, which runs
`_validate_stack` against the actual repo file. **Lesson: any future
`_validate_stack` test must run against the real entrypoint file, not just
synthetic text — synthetic-only tests missed this for an entire session.**

### P2 — Explicit `learning_rate` kwarg on FedBuff [DONE]

`flame/optimizer/fedbuff.py`: `self.learning_rate = kwargs.get("learning_rate", None)`;
both `_scale_add_agg_weights_pytorch` and the tensorflow twin use it when set,
else fall back to the existing `dataset_name` table unchanged (felix/oracle
untouched). Tests: `tests/optimizer/test_fedbuff_lr.py` (3 cases).

### P3 — Oracular availability from `_metadata` [DONE]

`flame/mode/horizontal/syncfl/fwdllm_aggregator.py:read_trainer_unavailability()`
rewritten to mirror `main_oort_sync_agg.py` (registry + mobiperf/synthetic
trace lookup, keyed by `task_id`); `metadata_dir` defaults to a
`__file__`-resolved `_METADATA_DIR` but is overridable for tests. Dropped now-
unused `glob`/`ast`/`json` imports. Tests:
`tests/mode/test_fwdllm_oracular_avail.py` (asserts a monkeypatched
`glob.glob` that raises is never hit). `DEPRECATED.md` updated per D3.

### P4 — `reselect_each_iteration` flag [DONE]

Extracted `_select_ends_respecting_reselect_gate(channel, task_to_perform)`
out of `_distribute_weights_sync` specifically so it's unit-testable without
a full `TopAggregator`. Cache invalidation at the round boundary falls out
for free from comparing `self._round_selected_ends_round == self._round` —
no separate invalidation code. Confined to the sync path; async (fluxtune)
untouched. Tests: `tests/mode/test_fwdllm_reselection.py` (3 cases, incl. an
empty/`None` selection never being cached).

### P5 — Rewrite baseline catalog [DONE]

`examples/_metadata/baselines.yaml`: deleted `fedfwd_async_random_dynkc`/
`fedfwd_oracular`; added `fwdllm`/`fwdllm_plus`/`fluxtune`/`fluxtune_dynkc`
per the matrix. `fluxtune_dynkc` deliberately keeps the exact legacy
selector/optimizer shape (incl. its known `dataset_name: google-speech`
artifact) for parity — not updated to use P2's `learning_rate` kwarg.
Verified: `load_baselines()` returns exactly the four new keys;
felix/oracle/refl/feddance/oort/fedbuff/fedavg are dict-equal to the
pre-Phase-7 catalog (diff-checked programmatically).

### P6 — Trainer config coverage + smoke YAMLs [DONE]

`trainer_base.yaml` already exposed every needed knob with defaults that
make *unaware fwdllm* correct out of the box — no new keys needed; two stale
comments fixed (mobiperf long/short-form note, old baseline names next to
the `selector:`/`optimizer:` placeholders). `*_3st_*` mobiperf traces already
existed in `_metadata/availability_traces/`. Smoke YAMLs: deleted
`fedfwd_async_random_n10_smoke.yaml`; added all **three**
`fwdllm_n10_smoke.yaml` / `fwdllm_plus_n10_smoke.yaml` /
`fluxtune_n10_smoke.yaml` (10 trainers each, baseline-appropriate scaled
selector kwargs) so every owner-spec baseline has real/sim parity coverage,
not just the cheapest one. All three verified via
`ConfigGenerator.generate_trainer_config()` + `_validate_stack` against the
real entrypoint, and locked in by
`tests/launch/test_config_generator.py::TestFwdllmSmokeYamlsResolve`.

### P7 — Pytest sweep [DONE]

`tests/launch/test_baselines.py::TestFwdllmBaselines` (7 tests: all four
baselines present, retired keys gone, selector/optimizer/availability/
perturbation/`reselect_each_iteration`/`dynamic_kc` match the matrix,
`fluxtune_dynkc` preserves the legacy shape). `tests/launch/test_config_generator.py::TestFwdllmEndToEndConfigGeneration`
(parametrized over all four baselines: trainer config generates without
`KeyError`, matrix values survive the merge, aggregator merge passes
`_validate_stack` against the real entrypoint). `tests/mode/test_baseline_readiness.py`
needed no changes — it enumerates `baselines.yaml` dynamically, so the four
new baselines were automatically covered.

Full sweep: `pytest lib/python/tests/launch lib/python/tests/mode
lib/python/tests/optimizer lib/python/tests/selector` → all green
(300+ tests).

### P8 — Live smoke tests [NEXT — see **Status & next step** at the top of this document for the exact commands and the full blocker writeup]

Supersedes the old Smoke Test E. Run all three smoke YAMLs
(`fwdllm`/`fwdllm_plus`/`fluxtune`) on a host with fwdllm's pinned ML stack
installed; assert ≥3–5 rounds complete, `trainer_round` telemetry present,
and `aggregator_config.json` shows the matrix's expected merged values for
that baseline. Optional: parity-check `client_idx` against the legacy
`json_scripts/trainer_*.json` values.

---

## Reference: critical files

- `flame/launch/runner.py` — `_validate_stack`, `_sweep_stragglers`,
  `run_experiment`'s `config_overrides` assembly.
- `flame/launch/spawner.py` — `ConfigGenerator.generate_trainer_config`,
  `TrainerSpawner.spawn_trainer`/`spawn_all`.
- `flame/launch/experiment_config.py` — `DatasetConfig`, `TrainerConfig`.
- `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` —
  `_weighted_aggregation_enabled`, `_select_ends_respecting_reselect_gate`,
  `read_trainer_unavailability`, `check_trainer_availability`.
- `flame/optimizer/fedbuff.py` — learning-rate dispatch.
- `flame/selector/random.py` vs `async_random.py` vs `async_oort.py`.
- `examples/_metadata/baselines.yaml` — `felix` (CNN/speech family) vs.
  `fwdllm`/`fwdllm_plus`/`fluxtune`/`fluxtune_dynkc` (LLM forward-mode family).
- `examples/fwdllm/aggregator/FedSgdAggregator.py`,
  `examples/fwdllm/trainer/forward_training/FedSgdTrainer.py`.
- `examples/fwdllm/trainer/forward_training/tc_transformer_trainer_distribute.py`
  — perturbation strategies; also where the pinned-ML-stack import wall lives.
- `examples/fwdllm/expt_scripts/{fwdllm,fwdllm_plus,fluxtune}_n10_smoke.yaml`.
