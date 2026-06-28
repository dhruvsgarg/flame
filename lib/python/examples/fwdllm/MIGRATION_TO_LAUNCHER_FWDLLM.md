# Migrate `fwdllm` onto the `flame.launch` YAML launcher

Living runbook for migrating `fwdllm` onto `flame.launch.run_experiment`, per
`lib/python/examples/MIGRATING_TO_LAUNCHER.md`. Read **Status & next step**
below before resuming work in any new session — it is always current.

## Status & next step

**Phases 1–13 are DONE.** Phase 12 ran the first live P8 smoke tests on
2026-06-27 for all three baselines and triaged the results — two real bugs
found and fixed (both verified against the actual failure logs, with new
regression tests, full suite green: see Phase 12). Several deeper issues were
diagnosed but **deliberately left unfixed pending a decision from the repo
owner** (scope/risk too high to guess at silently) — see Phase 12's
"Flagged, open" list. Phase 13 gave the smoke YAMLs a real auto-termination
bar (`max_data_id_progress` / `max_runtime_s`, whichever fires first),
found+fixed a second real bug live (the async path's inner loop didn't
honor `_work_done`), and **re-verified all three smoke YAMLs self-terminate
promptly** (`fwdllm_n10_smoke` 948s, `fwdllm_plus_n10_smoke` 625s,
`fluxtune_n10_smoke` 487s post-fix, all clean — see Phase 13 for the full
live trace). **The `[NEXT]` action is the telemetry/plots design work Phase
12 scoped out** (its "Flagged, open" issues #3/#4/#6/#7 remain unfixed by
design — see Phase 12 — and #7 in particular needs owner input before any
code is written).

**fwdllm smoke-test pass/fail bar is different from async_cifar10's.**
async_cifar10 judges a smoke run by rounds completed, because cifar10 rounds
are fast. **fwdllm must not be judged by rounds** — one fwdllm "round" is
`total_data_bins` (hardcoded to 150) data-id completions, each potentially
needing up to `max_iterations_per_data_id` (15) real forward-mode training
iterations; `rounds: 50` in the smoke YAMLs would in principle mean 7500
data-id completions, far beyond smoke-test scope. **Phase 13 made this an
actual auto-terminating stop condition** (`max_data_id_progress: 10` /
`max_runtime_s: 1800`, whichever fires first, set in all three smoke YAMLs)
instead of the old "watch `grep IterProgress` by hand and kill it" bar —
judge a smoke run a **pass** if it reaches `data_id=10` before the 30-minute
ceiling, and a **fail** if the 30-minute ceiling fires first (stuck on a low
`data_id` the whole time counts as a fail even if the process exited
cleanly via the time cap, not a crash).

Phase 11's blocker (three missing `async_cifar10` split files whose absence
was masked by the pre-fix n300 fallback) is resolved: all three were
regenerated via `async_cifar10/scripts/gen_dirichlet_split.py` and the full
test suite (326 passed, 7 skipped) remains green. The full sweep
(`examples/_metadata/dataset_splits/` vs every non-`path_style` YAML in
`expt_scripts_2026/`) now shows no missing files.

**Re-run P8 in the single `dg_flame` env (Phase 8 already applied, Phase 12
fixes already applied):**

```bash
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke.yaml
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fwdllm_plus_n10_smoke.yaml
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke.yaml
```

Each covers one of the three owner-spec baselines (10 trainers, ~50 rounds).
Pass criteria for each: aggregator + trainer processes start and don't crash;
**≥5 distinct `data_id` values reached** (not rounds — see above);
`telemetry/trainer_*.jsonl` has `trainer_round` events;
`aggregator_config.json` shows the expected merged `selector.sort` /
`is_async` / `optimizer.sort` for that baseline (see the matrix in Phase 7),
**and** `hyperparameters.aggGoal` (+ `selector.kwargs.aggGoal`/`aggr_num` for
fluxtune) matches that YAML's top-level `agg_goal` (2/2/3 respectively, per
Phase 10 — confirm the fan-out actually landed, since this is exactly the
class of bug Phase 10 fixed). Also confirm the run now **terminates on its
own** within the launcher's watchdog window (Phase 12 fixed the dead
`_work_done` loop for the sync path; the async/hybrid compose path and the
trainer-side message-drop issue are still open — see Phase 12).
Optional parity check: confirm `client_idx = (trainer_id-1) % 100` reproduces
the same H5 partitions as the legacy `json_scripts/trainer_*.json`.

**What the blocker was (now RESOLVED in Phase 8) — corrected diagnosis
(supersedes the old "tokenizers==0.12.1 won't compile" writeup, which was
traced to the stale `req.txt` pip-freeze, not the code's real needs). Full
analysis + what shipped in Phase 8 below.** In short:
- `dg_flame` (the dev env shared with `async_cifar10`: py3.11, torch 2.12,
  numpy 2.4) is simply **missing** the NLP/data stack — `transformers`,
  `tokenizers`, `huggingface-hub`, `h5py`, `pandas`, `scikit-learn`. Nothing
  *conflicts*; `async_cifar10` (CNN/torchvision) never pulled them in.
- Both entrypoints (`trainer/main.py`, `aggregator/main_fedfwd_agg.py` — both
  import `expts/initializer.py` and call `create_model`) die at two
  module-level imports, **neither of which needs the un-buildable
  `adapter-transformers==3.1.0` fork**:
  1. `from transformers import AdamW`
     (`tc_transformer_trainer_distribute.py:17`) — removed from transformers
     top-level in v4.x; fix is `from torch.optim import AdamW`.
  2. `from transformers.adapters import LoRAConfig` (`expts/initializer.py:35`)
     — the old fork API; only *used* under `peft_method=lora`, but imported at
     module level so it breaks regardless. Make lazy, or swap to the modern
     standalone `adapters` package.
- `functorch` is **fine** on torch 2.12 (the shim still ships
  `make_functional_with_buffers`; verified live).
- PEFT is **functionally required** — forward-gradient perturbations are only
  generated for `requires_grad` params (`tc_..._distribute.py:302`), so the
  full 66M-param DistilBERT won't converge. Production uses **`adapter`** PEFT
  (verified: every `json_scripts/trainer_*.json` has `peft_method: adapter` on
  `distilbert`), so the adapter library is genuinely exercised — see Phase 8.
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

**[DONE: P1–P7] [P8: env unblocked by Phase 8 — now the `[NEXT]`, pending GPUs + agnews H5 data]**

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

### P8 — Live smoke tests [DONE (first pass) → triaged in Phase 12, re-run is the new [NEXT]]

Supersedes the old Smoke Test E. The env blocker is resolved (Phase 8). First
live run of all three smoke YAMLs happened 2026-06-27 — see Phase 12 for full
triage (two bugs fixed, several deeper issues flagged open). The original
"≥3–5 rounds complete" pass criterion in this step is **superseded** by
Phase 12's data-id-based bar (see **Status & next step** above) — one fwdllm
round is too coarse a unit to use as a smoke-test pass condition.

---

## Phase 8 — Single-env dependency reconciliation [DONE]

Goal: run `fwdllm` (DistilBERT forward-mode FL) in the **same single dev env
already used by `async_cifar10`** (`dg_flame`), with no separate per-example
env and no torch/numpy downgrade.

### Shipped

- **Durable dep spec** (not a one-off install): the NLP forward-mode stack was
  folded into the `[examples]` extra in `lib/python/setup.py`
  (`transformers>=4.57,<4.58`, `adapters>=1.3,<1.4`, `h5py`, `pandas`,
  `scikit-learn`, `setproctitle`), so the existing canonical command
  `pip install -e lib/python[examples,dev]` now provisions **one** env that
  runs both async_cifar10 and fwdllm. Docs updated (`docs/prerequisites.md`,
  both quickstarts, `examples/fwdllm/README.md`); `req.txt` headed with a
  DEPRECATED banner pointing at `setup.py`.
- **Code edits (the modern-`adapters` path, keeping `peft_method: adapter`):**
  `tc_..._distribute.py` now imports `AdamW` from `torch.optim`;
  `expts/initializer.py` imports `adapters`/`BnConfig`/`LoRAConfig` from the
  `adapters` package, calls `adapters.init(model)` in the adapter/lora
  branches, and builds the bottleneck adapter as `BnConfig(**adapter_config)`
  (legacy dict keys map 1:1).
- **Verified on `dg_flame`** (py3.11 / torch 2.12 / numpy 2.4): all four
  deps installed from **prebuilt wheels** (no Rust/tokenizers compile —
  transformers 4.57.6 / adapters 1.3.0 / tokenizers 0.22.2 / h5py 3.16.0);
  both launcher entrypoints (`trainer/main.py`, `aggregator/main_fedfwd_agg.py`)
  now import cleanly past the old wall; and `create_model`'s adapter path
  freezes the base model leaving only adapter params trainable (~0.27%),
  active for the forward pass. The remaining work is purely the live GPU run
  (Phase 7 P8), which also needs the real agnews H5 data on the host.

### Analysis & rationale (why the above)

### Findings (live inspection of `dg_flame` + the real launcher import chain)

- **No conflicts, only absences.** `dg_flame` = py3.11, torch **2.12**, numpy
  **2.4**, torchvision/wandb. It has **none** of `transformers`, `tokenizers`,
  `huggingface-hub`, `h5py`, `pandas`, `scikit-learn`. `async_cifar10`
  (CNN/torchvision) never needed them, so nothing is broken — they're missing.
- **`req.txt` is a stale full `pip freeze`, not an essential-deps list.** Its
  `adapter-transformers==3.1.0` → `tokenizers==0.12.1` pin (the previously
  cited Rust-compile wall) is **not** actually required by the launcher path.
- **Real launcher import surface** (both `trainer/main.py` and
  `aggregator/main_fedfwd_agg.py` → `tc_transformer_trainer_distribute` +
  `expts/initializer.py`, both calling `create_model`):
  - `from transformers import AdamW` (`tc_..._distribute.py:17`) — removed from
    transformers top-level in v4.x. Fix: `from torch.optim import AdamW`.
  - `from transformers.adapters import LoRAConfig` (`initializer.py:35`,
    module-level) — old `adapter-transformers` *fork* API; this is the import
    both entrypoints actually die on. Only *used* under `peft_method=lora`;
    `add_adapter`/`train_adapter` used under `peft_method ∈ {adapter, lora}`.
    Default config = `peft_method: adapter` (`trainer_base.yaml:65`).
  - `functorch` — **fine** on torch 2.12 (shim ships
    `make_functional_with_buffers`; verified live).
  - `transformers.tokenization_bert` (`span_extraction_utils.py:8`, pre-4.0) —
    dead on the text-classification path (not imported by `main.py`). Ignore.
- **PEFT is functionally mandatory.** Perturbations are generated only for
  `v.requires_grad` params (`tc_..._distribute.py:302`), so forward-grad over
  the full 66M-param DistilBERT won't converge — *some* PEFT must leave only a
  small trainable subset.
- **Production = `adapter` PEFT on DistilBERT (verified, not assumed).** Every
  legacy `json_scripts/trainer_*.json` carries `"model_type": "distilbert"` +
  `"peft_method": "adapter"`, and `trainer_base.yaml:65` transcribes that
  default. (`run_text_classification.sh:61-65` *looks* like it switches
  distilbert→adapter / else→bitfit, but that shell var is **dead** — never
  exported, never re-read; the JSON drives it.) So the adapter library is
  **actually exercised** at runtime via `add_adapter`/`train_adapter`
  (`initializer.py:116-138`) — it cannot simply be skipped without changing the
  workload.

### Recommended single-env path: modern `adapters` add-on (no torch/numpy downgrade)

Keep `peft_method: adapter` (= what's been running). Install the standalone
**`adapters`** package — the maintained successor to `adapter-transformers`; an
add-on *on top of* mainline `transformers`, **not** the un-buildable fork, and
it ships prebuilt wheels. **This is the only thing we skip from the old stack:
`adapter-transformers==3.1.0` and its `tokenizers==0.12.1` pin.**

Definitely need (install into `dg_flame`): modern `transformers` (pulls
prebuilt `tokenizers` + `huggingface-hub` wheels — no Rust build), `adapters`,
`h5py`, `pandas`, `scikit-learn`. Can skip: the `adapter-transformers` fork,
the `tokenizers==0.12.1` pin, standalone `functorch` (torch-2.12 shim covers
it), the whole stale `req.txt`, and the span/QA path's pre-4.0
`transformers.tokenization_bert` (dead on the TC path).

Code edits (small, contained):
1. `from transformers import AdamW` → `from torch.optim import AdamW`
   (`tc_..._distribute.py:17`).
2. In `create_model` (`initializer.py`): `import adapters; adapters.init(model)`
   after `from_pretrained`; `from adapters import LoRAConfig, BnConfig`; build
   the bottleneck adapter as `BnConfig(**adapter_config)` (the raw dict's keys
   — `mh_adapter`/`output_adapter`/`reduction_factor`/`non_linearity`/… — map
   ~1:1 to `BnConfig`). `add_adapter`/`train_adapter` then work unchanged.
   `LoRAConfig` now resolves from `adapters` (the `peft_method=lora` branch),
   so the old `from transformers.adapters import LoRAConfig` line is replaced
   outright — no lazy-guard needed once `adapters` is installed.

**Fallback / simplification (only if `adapters` integration proves fiddly):**
`peft_method: bitfit` (`initializer.py:139-142`) is pure PyTorch
(`requires_grad=False` on all but bias+classifier) and needs **no** adapter
library — but it is **not** the production workload, so use it only as a
last-resort way to prove the launcher path end-to-end, not for real runs.

**Not recommended:** a separate fwdllm env (abandons the single-env goal and
still wouldn't build the old fork here), or dropping PEFT entirely (forward-grad
won't converge — see the `requires_grad` finding).

---

## Phase 9 — Config-template design invariant (live-run fixes) [DONE]

### Design invariant: `configs/` owns per-example config templates

Established during live P8 smoke runs. Applicable to all examples, not just
fwdllm — record it here for the eventual cross-example rule-set doc.

**Rule:** `examples/<example>/configs/` holds all config templates for that
example (trainer + aggregator base). `examples/_metadata/` is for shared FL
metadata only (baselines, availability traces, dataset splits, trainer
registry). Config templates do not belong there.

**Rationale:** The shared `_metadata/aggregator_base.json` contains only
generic FL fields (rounds, batchSize, selector/optimizer stubs). fwdllm's
aggregator calls `create_model()` and needs 25+ NLP/model fields that are
example-specific, not generic. Putting them in the shared template would
pollute it for every other example. The `configs/trainer_base.yaml` precedent
(established in Phase 2) already defines the right home — aggregator configs
follow the same pattern.

**Current state:**
- `fwdllm/configs/trainer_base.yaml` — trainer template (Phase 2)
- `fwdllm/configs/aggregator_base.json` — aggregator template (Phase 9); smoke
  YAMLs use `config_template: ../configs/aggregator_base.json`
- `async_cifar10/configs/trainer_base.yaml` — trainer template (pre-existing)
- `async_cifar10/metadata/aggregator_base.json` — dead copy of the shared base
  (unused; smoke YAMLs still point to `../_metadata/aggregator_base.json`);
  migration target when async_cifar10 gets a `configs/aggregator_base.json`
- `_metadata/aggregator_base.json` — legacy shared base; retires once all
  examples migrate to per-example `configs/`

**YAML migration path:** `runner.py` now accepts both `.json` and `.yaml`
config templates (extension-sniffed). When the runner's own YAML support was
added, `configs/aggregator_base.json` can be renamed to `.yaml` per example
without changing the runner. The `_metadata/aggregator_base.json` should be
the last file to migrate.

### P8 live-run bug fixes (also shipped here)

Four crashes found during P8 and fixed:
1. `avl_events_syn_train_*` traces (`FedSgdTrainer.__init__`): these three
   legacy custom traces are read unconditionally but have no `_metadata/`
   equivalent and were never injected by the spawner. Fixed: defaults added to
   `trainer_base.yaml` (`[[0, "AVL_TRAIN"]]` = syn_0 / always-available).
2. Aggregator `model_name` / NLP model fields missing: the generic
   `_metadata/aggregator_base.json` has no NLP fields; aggregator crashed on
   `config.hyperparameters.model_name`. Fixed: `configs/aggregator_base.json`
   with all 25 NLP/model fields (model_name/type, peft_method, fp16, etc.).
3. `ast.literal_eval` on already-parsed lists (`FedSgdTrainer.py`): YAML
   delivers `avl_events_*` as Python lists; the legacy JSON path delivered
   string-encoded lists. Fixed: `_parse_avl_events()` helper (Phase 8 crash
   batch, already shipped).
4. Aggregator `manual_seed` missing: fixed by putting it in
   `configs/aggregator_base.json` (not in the smoke YAML overrides, which is
   the wrong layer for a stable default).

---

## Phase 10 — `agg_goal` plumbing fix (launcher-wide, not fwdllm-only) [DONE]

Found while re-checking the three smoke YAMLs before the live P8 run: setting
`aggregator.agg_goal` in an experiment YAML had **no effect on the real
run**. `AggregatorConfig.agg_goal` (`experiment_config.py`) was only ever
read by `execution_config_generator.py`/`snapshot.py` for display — never
merged into `agg_cfg`, the dict actually written to `aggregator_config.json`
and handed to the spawned aggregator. The real aggregation-goal lives at
`hyperparameters.aggGoal` (config_template → baseline → config_overrides),
plus a *second*, independent sink at `selector.kwargs.{aggGoal,aggr_num}`
inside the selector classes — two names for the same concept depending on
selector family (`fedbuff`/`async_random`/`async_oort`/`oracle` read
`aggGoal`; `oort`/`refl_oort`/`feddance` read `aggr_num`).

Concretely, before this fix: `fwdllm_n10_smoke.yaml`/
`fwdllm_plus_n10_smoke.yaml` set `agg_goal: 2` at the top level (dead) and
never overrode `hyperparameters.aggGoal` anywhere, so the real value stayed
at the template default of **10**. `fluxtune_n10_smoke.yaml` set `agg_goal: 3`
(dead) and *did* override `selector.kwargs.aggGoal: 3` (real), but
`hyperparameters.aggGoal` stayed at the template default of **10** — the
selector only ever admits ~3 concurrent contributions while the aggregator's
own wait-for-N threshold wants 10, a likely hang.

**Fix** (`flame/launch/experiment_config.py`, `flame/launch/runner.py`):
`AggregatorConfig.agg_goal` default changed `int = 10` → `Optional[int] =
None`. `_build_aggregator_config()` now appends a final merge layer when it's
set, fanning the one value into `hyperparameters.aggGoal` +
`selector.kwargs.aggGoal` + `selector.kwargs.aggr_num` together (harmless
no-op for selectors that don't read a given key — `Selector.kwargs` is an
unvalidated freeform dict). `agg_goal: None` (the new default) leaves lower
layers untouched, so YAMLs that already work via `config_template`/baseline
without ever mentioning `agg_goal` are unaffected. Also added
`_validate_selector_label()`: `aggregator.selector` is a descriptive label
only (log filename/snapshot records, never merged into `agg_cfg`) — the
runner now raises if it disagrees with the real merged selector, instead of
letting every record of the run carry a silently wrong label.

`fluxtune_n10_smoke.yaml`'s redundant `config_overrides.selector.kwargs.
aggGoal: 3` was removed (now subsumed by the top-level `agg_goal: 3`); the
sibling async_cifar10 smoke YAMLs that manually triplicated the value
(`felix_n10_alpha100_syn20_smoke.yaml`, `oort_n10_alpha100_syn0_smoke.yaml`,
`refl_n10_alpha100_syn0_smoke.yaml`) were simplified the same way. See
`MIGRATING_TO_LAUNCHER.md` §4 ("real config vs. descriptive labels") for the
durable rule. Tests:
`tests/launch/test_runner_paths.py::TestAggGoalFanOut`,
`::TestSelectorLabelValidation`; full `tests/launch tests/mode
tests/optimizer tests/selector` sweep re-run green (318 passed).

---

## Phase 11 — Audit for the same bug shape elsewhere (launcher-wide) [DONE]

**Blocker resolution (appended):** the three split files that went missing
once the `dataset_name`/`num_trainers` forwarding fix was applied were
generated via `async_cifar10/scripts/gen_dirichlet_split.py` (option 1 from
the original recommendation):

```
python gen_dirichlet_split.py --alpha 100.0 --num-trainers 10 --data-root ../data
python gen_dirichlet_split.py --alpha 0.1   --num-trainers 32 --data-root ../data
python gen_dirichlet_split.py --alpha 0.1   --num-trainers 10 --data-root ../data
```

All three are now in `examples/_metadata/dataset_splits/`. A sweep of every
non-`path_style` YAML in `expt_scripts_2026/` against the split directory
shows no remaining gaps. Full test suite re-run: **326 passed, 7 skipped**.

---

Following Phase 10, asked: are there other fields with the same disease
(declared on `ExperimentConfig`, parsed from YAML without error, but never
actually forwarded to whatever consumes it)? Audited every field on
`TrainerConfig`/`AggregatorConfig`/`ExecutionConfig`/`ExampleConfig`/
`MetadataPaths` and every selector's `kwargs[...]` read against what the
launcher actually merges/forwards. Two more real instances found and fixed;
one false lead ruled out. **None of this affects fwdllm directly** — both
real bugs live in the trainer-side dataset-split lookup, which fwdllm never
reaches (`path_style: true` skips it entirely) — but it's recorded here
since the fix is in shared `flame/launch/*` code fwdllm also runs through,
per `MIGRATING_TO_LAUNCHER.md`'s "what to check before adding a field"
checklist (§10, item 8b).

1. **`TrainerConfig.dataset.name`/`num_trainers` never forwarded to the
   dataset-split lookup [fixed].** `ConfigGenerator.generate_trainer_config()`
   takes `dataset_name="cifar10", num_trainers=300` to pick which
   `<dataset>_alpha<a>_n<N>.yaml` file to read, but `runner.py`'s call into
   `TrainerSpawner.spawn_all()` → `spawn_trainer()` never passed either —
   every experiment silently read the 300-trainer cifar10 split file
   regardless of its real `trainer.dataset.name`/`trainer.num_trainers`.
   Verified the actual data impact on existing runs was nil (every
   `cifar10_alpha0.1_n48.yaml` entry is byte-identical to its
   `cifar10_alpha0.1_n300.yaml` counterpart — the split-generation script
   assigns indices per `trainer_id` independent of total `N`), but the code
   contradicted its own naming scheme and would have silently mis-partitioned
   data the moment a differently-generated split file showed up. Fixed by
   threading `dataset_name`/`num_trainers` through `spawn_trainer()` →
   `spawn_all()` → `runner.py`'s call site, and fixed
   `execution_config_generator.py`'s `dataset_split_key` (previously
   hardcoded the literal string `"cifar10_..."` regardless of the real
   dataset — already wrong today for any fwdllm record, just never
   noticed because that field is display-only). **Irrelevant to fwdllm's
   real runs** (`path_style: true` skips `get_dataset_split()` outright —
   see Phase 1b) but shares the exact fix and is in the same shared code
   fwdllm's launcher invocation passes through. Tests:
   `tests/launch/test_config_generator.py::TestTrainerSpawnerForwardsDatasetIdentity`
   (3 cases, including a default-preserving case for non-launcher callers).
2. **`MetadataPaths.dataset_splits_dir`/`.traces_dir` were fully dead
   [removed].** Declared on the dataclass, parsed from YAML without error,
   but `MetadataLoader` (`spawner.py`) only ever accepts a single root
   `metadata_dir` and hardcodes the `dataset_splits/`/`availability_traces/`
   subdirectory names under it — there was no wiring for a per-component
   override at all. No YAML in the repo set either field (confirmed by
   grep), so this was a dormant landmine, not an active bug. Removed rather
   than wired up, since nothing needs the capability today; re-add only
   alongside an actual `MetadataLoader` change to accept the overrides.
3. **False lead, ruled out: selector kwarg naming (`k`/`c`/`is_async`/
   `evalGoalFactor`/`roundNudgeType`/`selectType`).** Grepped every
   `kwargs["..."]` read across all selector implementations
   (`flame/selector/*.py`) for a second `aggGoal`/`aggr_num`-style split
   name. Every other kwarg is spelled identically everywhere it's read —
   `aggGoal`/`aggr_num` (already handled in Phase 10's fan-out) is the only
   instance of this particular disease in the selector layer.

`MIGRATING_TO_LAUNCHER.md` gained a durable checklist item (§10, 8b) and a
new §1 subsection ("`dataset.name`/`num_trainers` must be threaded through,
not defaulted") documenting finding #1 above, cross-referenced from §4's
existing "real config vs. descriptive labels" note from Phase 10.

---

## Phase 12 — First live P8 smoke-test triage (2026-06-27) [DONE: fixes 1–2.
Items 3–7: FLAGGED, open — need an owner decision before fixing]

Ran all three smoke YAMLs for real on GPUs with the real agnews H5 data:
`run_20260627_163015_fwdllm_n10_smoke`, `run_20260627_164344_fwdllm_plus_n10_smoke`,
`run_20260627_181243_fluxtune_n10_smoke` (paths under
`examples/fwdllm/experiments/`). Results: fluxtune crashed with an
`AttributeError`; fwdllm_plus ran for >1.5 hours without terminating;
fwdllm got stuck on `data_id=0` and was manually killed (`KeyboardInterrupt`)
after ~12 minutes. All three were triaged by reading the actual aggregator +
trainer logs line-by-line (not guessed at) — findings below.

### 1. [FIXED] fluxtune crash: `channel.cleanup_recvd_end()` is sync-selector-only, called unconditionally on the async path too

**Crash:** `AttributeError: 'AsyncOortSelector' object has no attribute
'_cleanup_recvd_end'` in `_process_single_trainer_message`, immediately on
the first received gradient message.

**Root cause:** `channel.cleanup_recvd_end(end)` (singular,
`flame/channel.py`) unconditionally calls `self._selector._cleanup_recvd_end(end, ...)`.
Only `RandomSelector` (`flame/selector/random.py`) implements that singular
per-end method — `AsyncOortSelector`/`AsyncRandomSelector` (used by
fluxtune/fluxtune_dynkc) only implement the batch `_cleanup_recvd_ends`
(plural) / `_cleanup_provided_ends` path.
`fwdllm_aggregator.py:_process_single_trainer_message` has **three** call
sites for this cleanup; one of them (the stale-update-reject branch) already
had the right `is_async` guard —
`if self.is_async: channel.cleanup_provided_ends(end) else: channel.cleanup_recvd_end(end)`
— but the other two (the duplicate-contribution guard, and the success
path) called the sync-only `cleanup_recvd_end(end)` unconditionally. fwdllm
and fwdllm_plus never hit this because they're sync (`is_async=False`,
`RandomSelector`); fluxtune is the only baseline that's async, so it's the
only one that ever exercised the missing branch.

**Fix:** `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` — both
remaining call sites now use the same `is_async` branch as the
already-correct third one. **Tests:**
`tests/mode/test_fwdllm_duplicate_contribution.py` (added
`test_async_path_uses_cleanup_provided_ends_not_cleanup_recvd_end`,
reproducing the exact production crash against a fake selector; updated the
existing fixture to accept an `is_async` flag).

### 2. [FIXED] fwdllm/fwdllm_plus never terminate — `self._work_done` is never set anywhere in fwdllm_aggregator.py

**This is the real cause of "doesn't terminate," independent of any
variance-check or availability stall below.** Every other FL stack in this
codebase (`syncfl/top_aggregator.py`, `lifl_coord_syncfl/coordinator.py`,
`coord_syncfl/coordinator.py`, `distributed/trainer.py`, ...) sets
`self._work_done = self._round > self._rounds` (or an equivalent) somewhere,
which is what makes `Loop(loop_check_fn=lambda: self._work_done)` ever exit.
fwdllm's `TopAggregator` extends `flame.mode.horizontal.asyncfl.top_aggregator.TopAggregator`
(not `syncfl`'s, despite living under the `syncfl/` module path — same class
hierarchy quirk already flagged at the top of this doc), and **neither
class ever sets `self._work_done = True`, nor ever reads
`self._rounds`/`hyperparameters.rounds` at all.** The composer loop runs
forever; `rounds: 50` in the smoke YAMLs was silently dead. The only things
that ever stopped a run were a manual kill or the launcher's external
watchdog (`runner.py`'s `aggregator_spawner.wait(timeout=...)`, itself
disabled unless `max_runtime_s`/`sim_wall_ceiling_s` is set — none of the
three smoke YAMLs set either).

Verified live: `aggregator_fwdllm_plus_n10_smoke` ran 16:44→18:11 (>1.5h)
continuously active (steady ~11k log lines/min the whole time, not idle),
never crashing, never exiting on its own.

**Fix:** `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`'s
`_process_aggregation_goal_met`, in the round-rollover branch (`if
self.data_id == self.total_data_bins:`), now also sets `self._work_done =
self._round > self.config.hyperparameters.rounds` and logs when it fires.
**Tests:** `tests/mode/test_fwdllm_rounds_termination.py` (4 cases: sets
`_work_done` once rounds are exhausted, leaves it unset/False while rounds
remain, and confirms the check is never touched on the non-rollover or
variance-check-failed branches).

**Caveat — only the sync compose path was fixed.** This fix is in
`_process_aggregation_goal_met`, called from both the sync and the
hybrid/async compose paths, so it should cover all three baselines'
round-rollover. But the **sync compose path's `c.tasklet("inform_end_of_training")`
is commented out** (`fwdllm_aggregator.py`, end of the `else:` branch of
`compose()`) — so even with the aggregator now exiting its own loop on
schedule, trainers are never sent a clean `EOT` broadcast on the sync path
(fwdllm/fwdllm_plus). They'll instead be force-killed by the launcher's
hardcoded `trainer_spawner.wait_all(timeout_per_trainer=30.0)` grace period
in `runner.py`. Good enough for "the experiment as a whole terminates," not
a clean per-trainer shutdown. Left as-is (re-enabling a commented-out
tasklet block with several alternate commented variants nearby looked like
an active WIP decision point, not something to flip blindly without a live
GPU re-run to confirm it doesn't break the sync distribute loop).

### 3. [FLAGGED, open] fwdllm_plus real hang: ORACULAR `mobiperf_2st` trace + only 10 trainers can leave fewer than `aggGoal` simultaneously available for very long real-time stretches

Even after fix #2 makes the run terminate eventually (once `rounds` is hit or
the watchdog fires), this is a **separate, real liveness problem**: with
`reject_stale_updates=False` and `aggGoal=2`, the run can go for a very long
real-clock time making zero progress if fewer than 2 of the 10 trainers are
ever simultaneously available.

**Evidence:** `random.py`'s `select` log (`available ends: ...`) shows the
run starting with all 10 trainers available, but by the run's final ~75
minutes (continuously, every ~0.5–1s poll) **exactly one** trainer
(`...580370`) was ever in the available set — never two. With `aggGoal=2`,
no aggregation can ever complete in that state; the selector just logs
`Waiting on 1, need 10 more to maintain concurrency 10` /
`0 new selection less than concurrency 10` forever.

**Why:** `mobiperf_2st`/`mobiperf_3st_50` (`_metadata/availability_traces/mobiperf_traces.yaml`)
are **real per-device MobiPerf availability traces**, one independent
on/off timeline per `device_{trainer_id:03d}`, looked up 1:1 by
`trainer_id`. They are not percentage/synthetic traces. A 10-trainer smoke
test only samples 10 of these real device timelines; whether ≥2 of those 10
specific devices are online at the same real wall-clock moment is governed
by real MobiPerf data, not anything we control, and `time_mode: real` means
no time acceleration — a long mutual-offline stretch in the real trace data
is a long real wait, with no guarantee it ever recovers before `rounds`/the
watchdog ends the run.

**Not fixed — needs an owner decision**, because the right fix is an
experiment-design choice, not a code bug:
  - Use a synthetic/always-available trace (`syn_0`, like plain `fwdllm`) for
    the **smoke test** specifically, reserving `mobiperf_2st`/`_3st_*` for
    longer, non-smoke ORACULAR/3-tier experiments where a multi-hour budget
    is expected anyway.
  - Or pick 10 specific `device_*` IDs known to have good simultaneous
    coverage in the trace data, instead of the default `trainer_id` 1–10.
  - Or lower the smoke YAML's `aggGoal` to 1 (changes what's being tested).
  - Or accept it and just budget more real wall-clock time + set
    `max_runtime_s` for ORACULAR smoke runs.

### 4. [FLAGGED, open] Distribute-loop has no back-pressure → duplicate-broadcast storm → channel silently drops legitimate retraining submissions

This is the most likely reason **plain fwdllm** (no availability trace,
`syn_0`-equivalent default) still stalled on `data_id=0` after only 5
variance-check iterations, separate from issue #3.

**Evidence chain (from the fwdllm_n10_smoke run):**
- `_distribute_weights_sync`/`_aggregate_grads_sync` are re-invoked on
  **every composer tick with no gating** — nothing waits for trainers to
  respond to broadcast N before sending broadcast N+1. Observed: ~2,082
  "Model distributed to clients" broadcasts in 12.5 minutes, all for the
  same `(round=1, data_id=0)`, i.e. one every ~0.35s, the entire time the
  run was "stuck."
- This is by design at the trainer level (a documented, deliberate
  duplicate-guard: a trainer's `_fetch_weights` compares the incoming
  message's `iteration_per_data_id` against its own; if equal, it's a
  resend of something already handled and is aborted/discarded — see the
  comment at `fwdllm_aggregator.py:712-717`). But each FAILED variance
  check increments `iteration_per_data_id` and re-broadcasts to **all**
  selected trainers (not just the ones who haven't responded yet), so all
  10 trainers retrain and resend on every iteration, not just the 2 needed
  for `aggGoal`.
- Trace of one trainer (`...580372`) across `data_id=0`: it genuinely
  retrained and called `_send_grads` **6 separate times** (6 distinct grad
  hashes logged, one per iteration 0 through ~5), but the aggregator's own
  `_updates_received` counter only ever recorded **1** contribution from
  this trainer for the entire `data_id=0` cycle. 8 of its legitimately
  recomputed retraining submissions were never counted.
- `_streamer_for_recv_fifo` (`channel.py`) logs
  `[RECV_FIFO] Skipping end_id ... - already has active task` repeatedly
  for the same end — i.e. when a second message from the same end arrives
  before the first one's "active task" is cleared, **the channel skips
  (drops) it rather than queuing it**. Combined with the broadcast storm
  above, a trainer that resends faster than the aggregator drains its
  queue loses messages outright, silently.
- Net effect: which 2 (of 10) trainers' contributions get counted toward a
  given iteration's `aggGoal` is effectively a race, not a deterministic
  selection. Convergence of the variance check (whether it ever reaches
  `var_good_enough` within `max_iterations_per_data_id`) depends on this
  race resolving favorably, which it didn't within the iterations this run
  got to before being killed.

**Not fixed — this is a structural concurrency/flow-control gap spanning
`fwdllm_aggregator.py`'s distribute loop and `channel.py`'s chunked-recv
dedup, pre-dating the launcher migration, and risky to change blindly
without a live GPU re-run to confirm the fix doesn't change FwdLLM's actual
training semantics.** Recommended directions for whoever picks this up:
  - Add back-pressure to `_distribute_weights_sync`: don't re-broadcast to
    an end that already has a message in flight / already responded this
    iteration, instead of broadcasting to the full selected set every tick.
  - And/or change `_streamer_for_recv_fifo`'s "already has active task"
    branch to queue rather than drop, so a fast-resending trainer's later
    messages aren't silently lost.
  - Either change needs a live re-run to confirm data_id throughput
    actually improves and nothing else (e.g. fluxtune's async paths, which
    share `channel.py`) regresses.

### 5. [FLAGGED, design gap] `total_data_bins` is hardcoded to 150, never read from config — makes "rounds" a bad smoke-test unit

`self.total_data_bins = 150` is a hardcoded literal in
`fwdllm_aggregator.py`'s init, not sourced from `config`/the H5 partition
file. One fwdllm "round" = 150 data-id completions. The smoke YAMLs'
`rounds: 50` therefore nominally means 7,500 data-id completions to reach
"50 rounds done" — nowhere close to smoke-test scope. This is why
**Status & next step** above replaces "rounds complete" with "≥5 distinct
`data_id`s reached" as the smoke-test bar. Not fixed; flagged because a
real fix (e.g. exposing `total_data_bins` as a config override, or a
separate small-N smoke-specific cap) is an API design choice, not picked
unilaterally here.

### 6. [FLAGGED, minor] `timer_decorator`'s `(Round=.., DataId=.., Iter=..)` log suffix can show a later call's state than the one it's reporting on

`flame/monitor/runtime.py:timer_decorator` reads `self.fwd_llm_stage`
**after** the wrapped function returns, not a snapshot taken when the
function was entered. Since `self.fwd_llm_stage` is shared mutable state
that later calls (on the same single-threaded composer loop) overwrite
before the decorator gets to log, a decorator line for, e.g., `_send_grads`
can print `Iter=5` even though that particular `_send_grads` call actually
ran for iteration 4 — the next `_fetch_weights` call had already advanced
`fwd_llm_stage` by the time the decorator's `logger.info` executed. This
caused real confusion during this triage (initially looked like a
concurrency race; it isn't — everything here runs on `MainThread`). Purely
a logging/observability inaccuracy, not a behavior bug; worth a future
cleanup (snapshot the stage at function entry, not in the wrapper after
return) but not fixed here since it's tangential to the actual hangs.

### 7. [FLAGGED, design input needed] Telemetry/plots for FwdLLM's data-id/iteration/variance dynamics — WORK IN PROGRESS, needs owner input before building

**Current state (verified, not assumed):** the aggregator-side telemetry
file (`telemetry/aggregator_<job>.jsonl`) was **completely empty (0 events)**
in all three smoke runs — `fwdllm_aggregator.py` never calls
`telemetry.emit(...)` anywhere; the rich `[IterProgress]`/variance-check
state that drove this entire triage only exists as plain-text `logger.info`
calls, not structured events. Trainer-side telemetry does exist
(`trainer_round` events, `selection` events) and **already carries**
`data_id`/`iteration_per_data_id`/`model_version` in its `extra` fields
(`FedSgdTrainer.py`'s `train_with_data_id`), so that half is in better shape
than the aggregator half. The post-run plotting pipeline
(`runner.py:_run_post_analysis()`) is the generic one built for
async_cifar10 — `selection`/`availability`/`sanity`/`system` plot
categories, all centered on **rounds** as the unit of progress.

**What maps cleanly onto the existing (cifar10-shared) plots, as-is:**
selection fairness/composition, availability dynamics, send/recv-lag and
other system-timing sanity plots — these are generic FL concepts and should
be reusable for fwdllm baselines without change once aggregator-side events
exist to drive them.

**What has no cifar10 analog and currently has *no* telemetry backing it at
all** (this is the gap worth designing for):
  - `data_id`/`iteration_per_data_id` progress and throughput (the unit
    that actually matters for fwdllm's pacing — see issue #5).
  - Variance-check convergence: `var` vs `var_threshold` across iterations
    of a given `data_id`, pass/fail/force-commit outcome.
  - Perturbation/JVP signals (`jvp_for_snr_check`, `grad_for_var_check`) and
    `perturbation_sampling`/`select_perturbation_using_jvp` behavior.
  - `model_version` cadence relative to `data_id` (when
    `inc_model_version_per_data_id` is set) rather than relative to round.
  - `fluxtune_dynkc`'s adaptive K/C trace over time
    (`DynamicKCController.step()`'s history) — currently only logged, not
    captured as telemetry.
  - Per-(data_id, iteration) accepted-vs-attempted submission counts per
    trainer — would have made issue #4's message-drop directly visible from
    a plot instead of requiring manual log archaeology.

**Initial ideas (genuinely unverified — confirm with the user before
building any of this):**
  a. A new aggregator-side telemetry event (e.g. `agg_data_id`, parallel to
     the existing `build_agg_round`/`build_agg_eval` builders in
     `flame/telemetry/events.py`), emitted once per
     `_process_aggregation_goal_met` call: round, data_id,
     iteration_per_data_id, var, var_threshold, var_good_enough,
     force_commit, agg_goal_cnt, elapsed wall time since data_id start.
  b. A "variance convergence" plot per data_id: iteration on the x-axis,
     var (and the threshold line) on the y-axis — shows how many
     iterations each data_id actually needed and whether/when it
     converged.
  c. A "data-id throughput over wall-clock time" plot — the fwdllm analog
     of cifar10's round-over-time plots, and the natural metric to
     automate the "≥5 data_ids" smoke-test bar instead of grepping logs by
     hand.
  d. A per-trainer "accepted vs. attempted" plot per data_id, fed by issue
     #4's accounting once it exists.
  e. A K/C-over-time plot for `fluxtune_dynkc`.

These are starting points only. Before any of (a)–(e) gets built: ask the
user which signals they actually want plotted (this list is not
exhaustive and some items may not be worth the effort), whether any
existing cifar10-style plot categories should be reused unmodified, and
whether the new aggregator-telemetry event should land before or after
issues #3/#4 above are resolved (since #4 in particular would change what
"accepted vs. attempted" actually means).

---

## Phase 13 — Smoke-test auto-termination: `max_data_id_progress` + `max_runtime_s` [DONE]

Phase 12 item 5 flagged that `rounds` is a useless smoke-test stop unit
(`total_data_bins=150` is hardcoded, so one round is 150 data-id
completions) and **Status & next step** above had been working around it
with a manual, post-hoc bar (`grep IterProgress` for ≥5 distinct `data_id`
values). That bar was never wired up as an actual stop condition — runs
still had to be watched and killed by hand (as Phase 12's own triage was).

**Fix:** `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` gained
`_check_early_stop_conditions()`, called from `_distribute_weights` (shared
by both the sync and async/hybrid compose paths, and invoked on **every**
composer tick regardless of whether an aggregation goal is ever met — unlike
the Phase 12 #2 rounds-based stop in `_process_aggregation_goal_met`, which
only ever runs on the round-rollover branch of a *successful* aggregation).
Two new optional `hyperparameters` (both `None`/off by default, so
unmodified production YAMLs are unaffected):

- `max_data_id_progress`: stop once `self.data_id` reaches this value.
- `max_runtime_s`: stop once this many wall-clock seconds have elapsed since
  `agg_start_time_ts` (the aggregator's own start). This hyperparameter
  already existed and was already read by `runner.py` to size the launcher's
  external watchdog timeout — but nothing on the fwdllm aggregator side ever
  *consumed* it (fwdllm never calls the inherited `increment_round()`, which
  is where `syncfl/top_aggregator.py` implements this cap for every other
  stack), so it was silently dead for fwdllm/fwdllm_plus/fluxtune until now.

Whichever cap is hit first sets `self._work_done = True`, which is what
actually exits the composer `Loop`. All three smoke YAMLs
(`fwdllm_n10_smoke.yaml`, `fwdllm_plus_n10_smoke.yaml`,
`fluxtune_n10_smoke.yaml`) now set `max_data_id_progress: 10` and
`max_runtime_s: 1800` (30 min) alongside the existing `rounds: 50`, so a
smoke run now terminates on its own — by data-id progress or a 30-minute
wall-clock ceiling, whichever comes first — instead of relying on a human to
watch logs and kill it. `rounds: 50` remains as an outer safety net only.

A sequential runner script was also added:
`expt_scripts/run_smoke_sequential.sh` (conda-activates an env, then runs
the three smoke YAMLs back to back, logging each to its own
`smoke_logs/<timestamp>/<name>.{yaml,out}`). It patches `max_runtime_s`/
`max_data_id_progress` into a generated copy of each YAML per invocation
(`--max-runtime-s`, `--max-data-id`, default 600s/10) rather than editing
the checked-in YAMLs, and supports `--stop-on-fail`.

**Live re-run (2026-06-28) found and fixed a second bug, in the same area:**
First pass, all three smoke YAMLs were run via the script with a tightened
600s/data_id=10 cap (for a faster smoke iteration than the YAMLs' checked-in
1800s). `fwdllm_n10_smoke` (948s) and `fwdllm_plus_n10_smoke` (625s) — both
**sync** baselines — passed and self-terminated cleanly. `fluxtune_n10_smoke`
(**async**) did not: its own log showed `_check_early_stop_conditions()`
correctly logging `max_data_id_progress=10 reached` at 02:36:57, only ~4.5
minutes after the aggregator started — but the process kept running for
another ~26 minutes until the launcher's external watchdog (sized at
budget + 1200s = 1800s here) force-killed it as a suspected deadlock.

**Root cause:** `compose()`'s async/hybrid path nests a nested loop inside
the outer one — `loop(task_reset_agg_goal_vars >>
asyncfl_loop(task_put_train >> task_get_weights) >> ...)`. The outer `loop`
checks `self._work_done` (good), but the inner `asyncfl_loop` only checked
`self._agg_goal_cnt == self._agg_goal`. Setting `_work_done` inside
`_distribute_weights` (called from `task_put_train`, *inside* the inner
loop) has no effect on the inner loop's own exit test, so the inner loop
just keeps spinning — and the outer loop never gets a turn to observe
`_work_done` — until an aggregation goal happens to complete on its own.
This is the same family of issue as Phase 12 #4 (slow/blocked contributions
on the async path), just hitting the *new* early-stop flag instead of the
pre-existing rounds-based one.

**Fix:** extracted the inner loop's exit test into
`_async_inner_loop_done()` (`fwdllm_aggregator.py`,
`return self._agg_goal_cnt == self._agg_goal or self._work_done`), used in
`compose()` in place of the old inline lambda. **Tests:**
`tests/mode/test_fwdllm_early_stop_conditions.py` (13 cases total — the
original 10 for `_check_early_stop_conditions`/`_distribute_weights`, plus
3 new in `TestAsyncInnerLoopExitsOnWorkDone` reproducing this exact
scenario: exits on `_work_done` alone, exits on agg-goal alone, stays open
when neither holds). Full sweep re-run green: `pytest lib/python/tests/launch
lib/python/tests/mode lib/python/tests/optimizer lib/python/tests/selector`
→ 344 passed, 7 skipped.

**Re-verified live after the fix:** `fluxtune_n10_smoke` alone, same 600s/
data_id=10 caps — `max_data_id_progress=10 reached` logged at 03:35:20
(~5 min in), process exited cleanly with **no** watchdog/deadlock warning
this time, total script time 487s (~8 min, including spawn/grace/
post-analysis overhead). All three smoke YAMLs now self-terminate promptly
on the data-id cap rather than relying on the 30-minute external watchdog.

**Environment note (unrelated to the code, but cost real time during this
verification):** the `dg_flame` conda env's editable `flame` install points
at a *different* checkout (`/home/dgarg39/flame`, an older branch predating
even Phase 1 of this migration — no `DatasetConfig.path_style`, etc.). Running
the smoke script with `dg_flame` fails instantly with
`TypeError: DatasetConfig.__init__() got an unexpected keyword argument
'path_style'`, before touching any GPU. Use `aish_smoke_flame` (or
`FLAME_CONDA_ENV=<env>` to the script) for this checkout —
`run_smoke_sequential.sh` now defaults to `aish_smoke_flame`.

Not addressed here (still open, per Phase 12): issues #3 (ORACULAR
mutual-offline stalls) and #4 (distribute-loop broadcast storm / message
drop) remain real liveness problems on the async path — a run can still
take the full cap to make any data-id progress if contributions are
sufficiently delayed/dropped. This phase guarantees a run that *does* reach
a stop condition actually exits promptly; it does not fix how long it takes
to reach one.

---

## Reference: critical files

- `flame/launch/runner.py` — `_validate_stack`, `_sweep_stragglers`,
  `run_experiment`'s `config_overrides` assembly.
- `flame/launch/spawner.py` — `ConfigGenerator.generate_trainer_config`,
  `TrainerSpawner.spawn_trainer`/`spawn_all`.
- `flame/launch/experiment_config.py` — `DatasetConfig`, `TrainerConfig`.
- `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` —
  `_weighted_aggregation_enabled`, `_select_ends_respecting_reselect_gate`,
  `read_trainer_unavailability`, `check_trainer_availability`,
  `_check_early_stop_conditions`/`_async_inner_loop_done` (Phase 13).
- `examples/fwdllm/expt_scripts/run_smoke_sequential.sh` — runs the three
  smoke YAMLs back to back with overridable `max_runtime_s`/
  `max_data_id_progress` caps (Phase 13); defaults to the `aish_smoke_flame`
  conda env (the one whose editable `flame` install points at this
  checkout).
- `flame/optimizer/fedbuff.py` — learning-rate dispatch.
- `flame/selector/random.py` vs `async_random.py` vs `async_oort.py`.
- `examples/_metadata/baselines.yaml` — `felix` (CNN/speech family) vs.
  `fwdllm`/`fwdllm_plus`/`fluxtune`/`fluxtune_dynkc` (LLM forward-mode family).
- `examples/fwdllm/aggregator/FedSgdAggregator.py`,
  `examples/fwdllm/trainer/forward_training/FedSgdTrainer.py`.
- `examples/fwdllm/trainer/forward_training/tc_transformer_trainer_distribute.py`
  — perturbation strategies; also where the pinned-ML-stack import wall lives.
- `examples/fwdllm/expt_scripts/{fwdllm,fwdllm_plus,fluxtune}_n10_smoke.yaml`.
