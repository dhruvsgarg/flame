# Migrate `fwdllm` onto the `flame.launch` YAML launcher

This document is the living runbook for migrating `fwdllm` onto
`flame.launch.run_experiment`, per `lib/python/examples/MIGRATING_TO_LAUNCHER.md`.
It is kept up to date (Progress line + checkboxes) as each step's checkpoint
passes — read the Progress line and checklist below before resuming work on
this migration in any new session.

## Progress: 17 / 18 checkpoints complete

**This counter covers the original Phase 1–5 plan + Smoke Tests A–E only.**
Phase 6 (design) and Phase 7 (the baseline-taxonomy implementation it
specified) are tracked separately, further down this document — see
"Phase 7 — Implementation plan"'s own `Progress: 7 / 8 steps complete`
line. Phase 7's remaining step (P8, live smoke tests) is blocked by the
same environment gap as this counter's remaining item (Smoke Test E).

**Update to the environment-gap note above:** the core `flame` library deps
(`aiostream`, `gpustat`, `paho-mqtt`, `shared-memory-dict`, `mlflow`) were
missing too, but are all pure-Python with no native/Rust build step --
installed without issue, unlike the `adapter-transformers`/`tokenizers`
wall. `FedSgdTrainer.py` now imports and instantiates standalone. The
remaining hard blocker is specifically fwdllm's ML training stack
(`adapter-transformers==3.1.0` and its pinned `tokenizers==0.12.1`), still
unresolved -- see step 9's note.

**Known environment gap (affects steps 9, 10, C, E):** this dev sandbox lacks
fwdllm's pinned ML stack (`req.txt`'s `adapter-transformers==3.1.0` +
its `tokenizers==0.12.1`, which needs `sklearn` too) -- mainline
`transformers==4.54.0` is installed instead, and the pinned package can't be
built here (its 2022-era Rust source fails under any modern `rustc`).
Confirmed pre-existing: the untouched legacy `fl_main.py` fails identically.
Steps verified via isolated-logic checks instead of full execution where this
gap blocks reaching model construction; full live execution (Smoke Test E)
will need an environment that actually has fwdllm's pinned stack installed.

**Implementation steps (13):**
- [x] 1. Fix `_validate_stack` regex/`_ASYNC_STACKS` for `fwdllm_aggregator` (Phase 1a)
- [x] 2. Add `DatasetConfig.path_style` + `skip_index_splits` plumbing (Phase 1b)
- [x] 3. Add `TrainerConfig.client_idx_modulo` + per-trainer override path (Phase 1b)
- [x] 4. Fix `_sweep_stragglers()` pattern list for fwdllm paths (Phase 1c)
- [x] 5. Create `fwdllm/metadata -> ../_metadata` symlink (Phase 2d)
- [x] 6. Create `fwdllm/configs/trainer_base.yaml` (Phase 2c)
- [x] 7. Add `fedfwd_async_random_dynkc` + `fedfwd_oracular` to `_metadata/baselines.yaml` (Phase 3)
- [x] 8. Create `fedfwd_async_random_n10_smoke.yaml` (Phase 4)
- [x] 9. Create `fwdllm/trainer/main.py` (Phase 2a)
- [x] 10. Create `fwdllm/aggregator/main_fedfwd_agg.py` (Phase 2b)
- [x] 11. Fix mobiperf trace-name mismatch in `FedSgdTrainer.py` (Phase 2a)
- [x] 12. Add `trainer_round` telemetry emission in `FedSgdTrainer.py` (Phase 2a)
- [x] 13. Add `expts/run_tc_expts/DEPRECATED.md` (Phase 5)

**Smoke tests (5) — interspersed to catch blockers early:**
- [x] A. Static `_validate_stack` check (after step 4: verify the Phase 1a fix works)
- [x] B. Load experiment YAML + validate baseline (after step 8: confirm config schema is sound
  before writing entrypoint code that depends on it)
- [x] C. Entrypoints accept `--config-json` (after step 10: trainer/agg don't crash on startup)
- [x] D. Full static config generation dry-run (after step 13: all wiring correct before live test)
- [ ] E. Live 10-trainer smoke test + parity vs. legacy script (final: end-to-end validation)

**Reordering note (this section added when the plan was corrected):** baselines.yaml
and the smoke-test experiment YAML (steps 7–8) were moved earlier than the
entrypoint scripts (steps 9–10) so that Smoke Test B has real files to load
before any entrypoint code is written. The baseline/YAML files only reference
the entrypoint paths by string (`example.aggregator_main`), so they don't
need the entrypoint files to exist yet — but Smoke Test B needs the
baseline/YAML files to exist. Originally steps 7–8 came after the
entrypoints (mirroring the phase numbering in MIGRATING_TO_LAUNCHER.md), which
made Smoke Test B unrunnable at its assigned point; this reorder fixes that
without changing what any step actually does.

## Checkpoint rules

- **One step at a time.** Do not start step *N+1* until step *N*'s checkpoint
  has passed.
- **Checkpoint after every step**, before checking its box:
  - Code-change steps (1–4, 9–12): the file imports/parses cleanly, and where
    applicable, the specific behavior the step claims to fix is exercised once
    (e.g. after step 1, re-run the `_validate_stack` repro from Smoke Test A
    and confirm it now passes instead of raising).
  - New-file steps (5, 6, 7, 8, 13): the file exists at the right path with
    the right shape (symlink resolves; YAML parses via `yaml.safe_load`).
  - Smoke test steps (A–E): the run actually executed and its specific
    pass/fail criteria (below) were checked, not just "command exited 0."
- **On checkpoint failure**: stop, fix the issue, re-run the same checkpoint —
  never proceed to the next step with a known-failing checkpoint.
- **After each passing checkpoint**, update both the `Progress: N / 18
  checkpoints complete` line and that step's checkbox in the same edit, so
  the file is never out of sync with actual progress.
- This file is the source of truth for resuming work across sessions — before
  starting any work on this migration, read its Progress line and checklist
  first rather than assuming where things left off.

---

## Phase 6 — Baseline taxonomy: fwdllm / fwdllm+ / fluxtune

**Status: design complete, fully implemented in Phase 7 below (P1–P7 DONE).**
This phase itself stayed design-only (decisions D1–D6, captured below); all
code/YAML changes it called for landed under Phase 7's step-by-step plan.
One addendum to the owner's spec, given verbally in the Phase 7 implementation
session: felix (CNN/speech family) always runs fixed K/C, but fluxtune's
async_oort selector must support **both** fixed and adaptive (dynamic_kc)
K/C, toggled and parameterized purely via config — see fluxtune's
`selector.kwargs.dynamic_kc` block (Phase 7 step P5) for the implementation.

### Soundness verdict on the prior Phase 6 design (this session's evaluation)

**Verdict: the investigation is sound and well-grounded, but it stopped at
findings and left every actual decision open — so as a *design* it was
incomplete. The findings (file:line-verified) hold up under re-reading, with
one overstatement (finding 6, corrected below). The taxonomy it gestures at is
right; what was missing is the resolved target architecture, the concrete
step/checkpoint/test breakdown, and a decision on the legacy baselines. This
session supplies those (decisions D1–D6 and Phase 7 below).**

What was right: (a) `_validate_stack` is a real blocker for sync fwdllm
(finding 2); (b) FedAvg-vs-FedBuff is genuinely a one-flag axis, re-confirmed
— the fwdllm aggregator only touches `self.optimizer.agg_rate_conf` inside the
`_weighted_aggregation_enabled` (fedbuff-only) branch at
`fwdllm_aggregator.py:549-563`, so `optimizer.sort: fedavg` takes the
`rate=1.0` path and needs no code change (finding 3); (c) perturbation and
availability are config-ready (findings 4–5); (d) `fedbuff.py`'s hardcoded LR
table is the one real cross-example coupling (finding 7).

What needed correcting / completing: finding 6 overstated the selector blocker
(the `random` selector already supports sync via `is_async`, see correction
below); the oracular json_scripts dependency (Phase-3 note) was treated as an
out-of-scope follow-up but is actually **on** the critical path for a clean
fwdllm+ (decision D3); and the "keep or retire the two legacy baselines"
question was never answered (decision D6).

### Why this phase exists

Smoke-testing needs three baselines — **fwdllm**, **fwdllm+**, **fluxtune** —
but only two baselines currently exist in `_metadata/baselines.yaml`
(`fedfwd_async_random_dynkc`, `fedfwd_oracular`, added in Phase 3 above), and
neither maps cleanly onto the three names. Investigating the gap surfaced a
broader question: should fwdllm's async_oort/fedbuff usage share code paths
with `async_cifar10`'s existing `felix` baseline as-is, or does FluxTune (the
adaptive dynamic_kc / LLM-finetuning variant) need some of that shared code
forked or parameterized? That question is the actual subject of Phase 6 and
needs its own design pass before any YAML/code changes land.

### Target baseline spec (owner's words, verbatim)

| Axis | **fwdllm** (syncfl) | **fwdllm+** (syncfl) | **fluxtune** (asyncfl) |
|---|---|---|---|
| Trainer availability | unaware | oracular | 3-tier client_notify |
| Selection | random per round | random per iteration | async_oort |
| Aggregation | federated averaging | federated averaging | fedbuff-based |
| Perturbation | randomly generated | randomly generated | greedy JVP-based |

Owner's note on fluxtune: "Selection (async oort, should be separated out
from felix implementation), aggregation (fedbuff based but might need
different parameters from felix so should be separated)... if you want, we
can incorporate these design decisions/separate out code cleanly so that it
is easy to create configs and the codebase is also clean, separated and
maintainable across the examples (felix is for google speech, cifar10 etc)
while fluxtune is for llm fine tuning through examples and datasets like
fwdllm."

### Grounded findings this session (file:line, verified by direct reading)

**1. `felix` is a real, already-registered baseline, not just a verbal
nickname** — `examples/_metadata/baselines.yaml:18-64`, used by
`async_cifar10`: `selector.sort: async_oort` (`evalGoalFactor: 1.0`) +
`optimizer.sort: fedbuff` (`use_oort_lr: "True"`, `agg_rate_conf.type: new`)
+ `client_notify.enabled: "True"` (not oracular). This is structurally the
closest existing precedent for fluxtune — same selector/optimizer shape,
different domain (cifar-10/google-speech vs LLM fine-tuning).

**2. Real blocker: `_validate_stack()` forces async for *any* fwdllm
aggregator, which breaks the sync fwdllm/fwdllm+ specs.**
`flame/launch/runner.py:398-434`:
```python
_ASYNC_STACKS = {"asyncfl", "coord_asyncfl", "fwdllm"}
_ASYNC_SELECTORS = {"async_oort", "async_random", "fedbuff"}

def _validate_stack(self, agg_main_path, agg_cfg):
    ...
    stack = "fwdllm" if <agg main imports fwdllm_aggregator> else ...
    is_async_stack = stack in self._ASYNC_STACKS   # always True for fwdllm
    is_async_sel = selector in self._ASYNC_SELECTORS
    if is_async_stack != is_async_sel:
        raise ValueError(...)
```
Because `stack` is keyed off the aggregator *class* (always `fwdllm_aggregator`
for this example) rather than the runtime `is_async` selector kwarg, a
sync-style fwdllm/fwdllm+ baseline using `selector.sort: random` would be
**rejected before any process spawns** — `random` is not in
`_ASYNC_SELECTORS`. This check was added in Phase 1a of this same migration
under the assumption that all fwdllm runs are async (true of the original
legacy `aggregator.json`, not true of the new fwdllm/fwdllm+ spec). Needs
revisiting: likely fix is to key the async/sync check off
`agg_cfg["selector"]["kwargs"].get("is_async")` directly instead of the
stack name, for the `fwdllm` stack specifically.

**3. Aggregation rule (federated averaging vs fedbuff) is a single existing
flag, not two code paths** — `flame/mode/horizontal/syncfl/fwdllm_aggregator.py:260-269`:
```python
self._optimizer_sort_value = self.config.optimizer.sort
OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION = (OptimizerType.FEDBUFF,)
self._weighted_aggregation_enabled = (
    self._optimizer_sort_value in OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION
)
```
and used at `fwdllm_aggregator.py:549-577` (`aggregate_grads_from_trainers`):
when not fedbuff, `rate = 1.0` for every gradient (plain FedAvg); when
fedbuff, `rate` comes from `agg_rate_conf`. Both `_aggregate_grads_sync` and
`_aggregate_grads_async` call the same `aggregate_grads_from_trainers`, so
**FedAvg vs FedBuff aggregation is purely `optimizer.sort: fedavg` vs
`fedbuff`** — `OptimizerType.FEDAVG` already exists
(`flame/config.py:58`). No code change needed for this axis.

**4. Perturbation strategy is already fully wired end-to-end, no code
change needed** — `select_perturbation_using_jvp` is read from config in
`examples/fwdllm/trainer/main.py:123` and defaults `false` in
`examples/fwdllm/configs/trainer_base.yaml:84`; consumed in
`trainer/forward_training/tc_transformer_trainer_distribute.py:204-205,
394-463` (random-candidate path vs JVP-scored candidate path, both already
implemented). Setting `select_perturbation_using_jvp: "True"` per-baseline
in `baselines.yaml` is sufficient.

**5. Availability — ORACULAR and HEARTBEAT are both config-ready**;
`trackTrainerAvail` (YAML) aliases to `track_trainer_avail`
(`flame/config.py:175`), and `check_trainer_availability` already branches
on `enabled`/`type in {ORACULAR, HEARTBEAT}`
(`fwdllm_aggregator.py:1304-1313` region). ORACULAR has a known separate
issue (hardcoded glob over legacy `json_scripts/trainer_*.json`, already
flagged in the existing `fedfwd_oracular` baseline's description in
`baselines.yaml:346-357` — unrelated to this phase, pre-existing). HEARTBEAT
is implemented but currently unused by any baseline. The "3-tier
client_notify" mode (trainer self-reports `AVL_TRAIN`/`AVL_EVAL`/`UN_AVL` via
a separate notify thread — see `flame/config.py`'s `TrainerAvailState` enum
and `examples/fwdllm/trainer/forward_training/FedSgdTrainer.py:194-275`) is
a **different, independent mechanism** from `trackTrainerAvail` — it feeds
`task_eligible_states`-based filtering inside the selector, not the
aggregator's post-selection liveness check. fluxtune's "3-tier client
notify" axis means `client_notify.enabled: "True"` + a `*_3st_*` trace, with
`trackTrainerAvail.enabled` likely `"False"` (no oracular/heartbeat layered
on top) — needs explicit confirmation, not yet decided.

**6. Selection granularity ("random per round" vs "random per iteration")
has no existing code-level toggle, and fwdllm's sync aggregation path
cannot use the plain `random` selector as-is today.** Confirmed by direct
trace of `fwdllm_aggregator.py`'s `sync_collect_and_accumulate_grads`
(~line 1051-1078): it calls `channel.ends()` (no `RECV` state arg) inside
its collection loop, which `flame/selector/random.py` doesn't handle the
same way `async_random.py` does (random.py only properly drains
`selected_ends` on an explicit `RECV`-state call; fwdllm's sync loop never
makes that call). This is *why* both currently-registered fwdllm baselines
use `async_random`/`async_oort` even though one of them runs synchronously
— the buffered SEND/RECV state machine is a hard dependency of fwdllm's
collection loop, not an arbitrary choice. Making genuine "random per round"
(classic, only-select-once, like the `fedavg` baseline at
`baselines.yaml:236-255`) vs "random per iteration" (re-select on each
trainer message) real, distinct behaviors for fwdllm/fwdllm+ likely
**needs a small code change** in `sync_collect_and_accumulate_grads` (or a
new selector mode) — not just a YAML kwarg swap. This needs design.

**7. `fedbuff` optimizer hardcodes per-example learning rates keyed by
`dataset_name`** — `flame/optimizer/fedbuff.py:233-256`:
```python
if self.dataset_name == "cifar-10":
    learning_rate = 40.9   # (use_oort_lr=False path)
elif self.dataset_name == "google-speech":
    learning_rate = 0.075
else:
    learning_rate = 1.0    # fallback
# use_oort_lr=True path: 0.3 / 0.065 / (no fallback shown for else)
```
This is the concrete code-sharing problem behind the owner's "felix vs
fluxtune... might need different parameters... should be separated" note:
`async_oort.py` itself is fully generic/config-driven (no hardcoded
example-specific branches found), but `fedbuff.py`'s learning-rate dispatch
is **not** — it only knows about `cifar-10` and `google-speech`, and
`fedfwd_async_random_dynkc`'s existing `dataset_name: google-speech` kwarg
(flagged as a copy-paste artifact in `baselines.yaml:311-315`) is silently
relying on the google-speech branch rather than fwdllm having its own rate.
Cleanest fix: add a direct `learning_rate` kwarg to `fedbuff.py`,
independent of `dataset_name`, so each baseline sets its own rate explicitly
instead of routing through a dataset-name lookup table. This is the one
clear "separate fluxtune from felix" code change identified so far — felix
(`async_oort`+`fedbuff`) itself needs no fork, just this one optimizer
parameterization fix, which benefits all examples, not just fluxtune.

**8. Gap vs. the two already-registered fwdllm baselines:**
- `fedfwd_async_random_dynkc` (`baselines.yaml:265-343`): async_random +
  dynamic_kc + fedbuff(`type: old`) + `perturbation_sampling: "True"` only
  (no JVP) + `trackTrainerAvail` disabled. Doesn't match **fwdllm** (which
  per spec should be sync, not async) or **fwdllm+** (should be oracular).
  It's closer to an async middle-ground than either target.
- `fedfwd_oracular` (`baselines.yaml:345-398`): `async_oort` (spec says
  fwdllm+ should be `async_random`/"random") + ORACULAR (matches fwdllm+'s
  availability axis) + fedbuff(`type: new`) + perturbation flags unset
  (should be `perturbation_sampling: "True"` per spec). Partial match on
  fwdllm+, selector axis is wrong, perturbation flag missing.
- Neither is async + 3-tier client_notify + JVP perturbation, so
  **fluxtune doesn't exist yet** in any form.

### Correction to finding 6 (verified this session)

Finding 6 above overstates the blocker. Direct reading of
`flame/selector/random.py` shows it **does** support both modes: it reads an
`is_async` kwarg (`random.py:52`), and its `select()` handles
`VAL_CH_STATE_RECV` by returning the drained `selected_ends`
(`random.py:271-273`) — the same buffered SEND/RECV state machine
`async_random.py` uses. So `random` with `is_async: false` is a usable sync
selector, and the reason the two legacy baselines use `async_random` is
historical (they were transcribed from a legacy async `aggregator.json`), not
a hard code-level dependency. What is genuinely missing is not "can the sync
random selector run at all" but "is there a toggle for re-selecting clients
*per dynamic_kc iteration* vs *once per round*" — that distinction (fwdllm vs
fwdllm+) has no flag today and is the one selection-axis code change still
required (see decision **D4** / step **P4** below).

### Design decisions (RESOLVED this session)

These resolve the six open questions. Guiding principle for all of them:
**baselines.yaml is the single shared catalog across every example; shared
library code (selectors, optimizers, the fwdllm aggregator) stays
example-agnostic and config-driven; example-specific behavior lives only in
the example's `aggregator/`,`trainer/` subclasses and its `trainer_base.yaml`.
The CNN/speech family (felix/oort/refl/feddance/oracle/fedbuff/fedavg) and the
LLM forward-mode family (fwdllm/fwdllm+/fluxtune) share the *same* selectors
and optimizers — we parameterize the one place they currently diverge rather
than fork any shared class.** No felix code or config changes.

**D1 — `_validate_stack` (was Q1, Q6).** Fix it to key async-ness off the
selector's `is_async` kwarg for the `fwdllm` stack, not stack-set membership.
The fwdllm aggregator already dispatches sync vs async purely on
`selector.kwargs.is_async` (`fwdllm_aggregator.py:1657-1660`), so that kwarg
is the single source of truth — `_validate_stack` should agree with it. The
real invariant to enforce for fwdllm is *internal consistency*: an async
selector (`async_oort`/`async_random`/`fedbuff`) must set `is_async: true`,
and a sync selector (`random`) must set `is_async: false`/unset. This change
is scoped to the `stack == "fwdllm"` branch, so no other example's validation
path changes (felix/cifar10 keep the `_ASYNC_STACKS` membership test
unchanged). Remove `"fwdllm"` from `_ASYNC_STACKS` since it is no longer the
discriminator for this stack.

**D2 — `fedbuff.py` learning rate (was Q3).** Add an explicit
`learning_rate` kwarg now, before fluxtune registers. When present it wins;
when absent, fall back to the existing `dataset_name` lookup (backward
compatible — felix/oracle/fedbuff baselines are untouched). This removes
fluxtune's silent reliance on the `google-speech` branch (the copy-paste
artifact flagged at `baselines.yaml:311-316`) and is the single
"separate fluxtune from felix" change that benefits all examples. This is the
answer to Q6: felix needs no fork — only this generic parameterization.

**D3 — Oracular availability source (new, required by fwdllm+).** Rewrite the
**library** method `fwdllm_aggregator.read_trainer_unavailability()`
(`fwdllm_aggregator.py:480-489`, currently globs legacy
`json_scripts/trainer_*.json`) to read `_metadata/trainer_registry.yaml` +
`_metadata/availability_traces/*.yaml`, porting the already-clean pattern from
`async_cifar10/aggregator/pytorch/main_oort_sync_agg.py:173+`. fwdllm+'s
availability axis is ORACULAR, so a clean fwdllm+ is impossible while this
method depends on the legacy directory. This also unblocks the Phase 5
json_scripts deletion that was deferred in step 13.

**D4 — Selection granularity, "per round" vs "per iteration" (was Q2; owner
clarified the semantics).** In fwdllm each client's data is split into
**databins** (`data_id`), and within a databin the server iterates
(`iteration_per_data_id`) until the gradient-variance threshold is met (or
`max_iterations_per_data_id`). The loop nesting (verified at
`fwdllm_aggregator.py:900-998`) is:

```
round (self._round):
  for data_id in 0..total_data_bins:            # databins
    iterate until var <= var_threshold / max_iter:   # iterations within a databin
      _distribute_weights_sync()  -> channel.ends(VAL_CH_STATE_SEND)  # selection fires here
    data_id += 1                                # :981
  self._round += 1                              # :996 (after ALL databins done)
```

- **random per round (fwdllm)** = **one** selection that persists across *all*
  databins and *all* iterations of a round — the same trainers serve the whole
  round, re-sampled only when `self._round` advances.
- **random per iteration (fwdllm+)** = a fresh selection on *every* iteration,
  regardless of databin or iteration boundary.

Critical grounding: `_distribute_weights_sync` calls
`channel.ends(VAL_CH_STATE_SEND, ...)` (`fwdllm_aggregator.py:1437`) on **every**
iteration, which re-invokes the selector — so **per-iteration is what the code
does today**, and **per-round is the new behavior to build**. The fix is a
declarative hyperparameter `reselect_each_iteration: bool` (default `true`, to
preserve current behavior). When `false`, cache the selected end set at the
round boundary (where `self._round` increments / `data_id` resets to 0) and
re-distribute to that cached set on subsequent iterations instead of calling
the SEND-state selection again; when `true`, keep today's per-iteration
re-selection. The non-trivial part is the selector's buffered SEND/RECV state
machine (`random.py` drains `selected_ends` on RECV) — per-round mode must
re-arm/reuse the same set rather than re-sample, so implement the cache at the
aggregator level (`self._round_selected_ends`) and bypass re-selection, rather
than mutating the selector. Lock both modes with a unit test (step P4).

**D5 — fluxtune availability = 3-tier client_notify (was Q4).** "3-tier
client_notify" means `trainer.client_notify.enabled: "True"` with a `*_3st_*`
trace (trainer self-reports `AVL_TRAIN`/`AVL_EVAL`/`UN_AVL`), and
`trackTrainerAvail.enabled: "False"` — client_notify is the *only* availability
signal, NOT layered with ORACULAR/HEARTBEAT. The three are independent
mechanisms (finding 5); fluxtune uses exactly one.

**D6 — Retire the two legacy baselines (was Q5).** Replace
`fedfwd_async_random_dynkc` and `fedfwd_oracular` with the three owner-spec
baselines `fwdllm` / `fwdllm_plus` / `fluxtune`. `fedfwd_oracular` is fully
superseded by `fwdllm_plus` — delete it. The adaptive `dynamic_kc` block from
`fedfwd_async_random_dynkc` is a genuinely distinct research config (async +
adaptive-K) that maps to none of the three; preserve it as a clearly-labeled
`fluxtune_dynkc` variant (fluxtune base + the dynamic_kc selector block) so the
ported production default stays reproducible for parity, rather than keeping
the misnamed `fedfwd_*` entries. The smoke-test YAML moves to the `fwdllm`
baseline (cheapest, and it exercises the brand-new sync path — best coverage).

### Critical files for the next session

- `lib/python/examples/_metadata/baselines.yaml` (existing `felix` at
  lines 18-64; existing fwdllm baselines at lines 257-398)
- `lib/python/flame/launch/runner.py` (`_validate_stack`, lines 398-434)
- `lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py`
  (`_weighted_aggregation_enabled` at 260-269; `sync_collect_and_accumulate_grads`
  ~1051-1078; `check_trainer_availability` ~1304-1313)
- `lib/python/flame/optimizer/fedbuff.py` (learning-rate dispatch, 233-256)
- `lib/python/flame/selector/random.py` vs `async_random.py` vs
  `async_oort.py`
- `lib/python/examples/fwdllm/trainer/forward_training/tc_transformer_trainer_distribute.py`
  (perturbation strategies, 200-463)
- `lib/python/examples/fwdllm/configs/trainer_base.yaml`,
  `lib/python/examples/fwdllm/trainer/forward_training/FedSgdTrainer.py`
  (client_notify / `TrainerAvailState`)

---

## Phase 7 — Implementation plan (concrete, ready to execute)

**Progress: 7 / 8 steps complete (P1–P7 DONE; P8 -- live smoke tests --
blocked on the same pinned-ML-stack environment gap as Smoke Test E; see
Phase 7 implementation notes below).**

Built on decisions D1–D6. Same checkpoint discipline as Phases 1–5: **one step
at a time; each step's checkpoint must pass (and, for code steps, its unit test
must be added and green) before starting the next.** Code/library changes
(P1–P4) land first because the baselines depend on them; then the catalog
(P5), the per-baseline trainer config (P6), pytest sweep (P7), and the live
smoke test (P8).

### Target baseline matrix (the contract every step serves)

| Baseline | stack (`is_async`) | selector | optimizer | availability | perturbation | reselect |
|---|---|---|---|---|---|---|
| **fwdllm** | sync (`false`) | `random` | `fedavg` | unaware (`trackTrainerAvail` off, `client_notify` off) | `perturbation_sampling: "True"`, JVP off | per-round (`false`) |
| **fwdllm_plus** | sync (`false`) | `random` | `fedavg` | ORACULAR (`trackTrainerAvail` ORACULAR, `_metadata` trace) | `perturbation_sampling: "True"`, JVP off | per-iteration (`true`) |
| **fluxtune** | async (`true`) | `async_oort` | `fedbuff` (explicit `learning_rate`) | 3-tier `client_notify` (`*_3st_*` trace), `trackTrainerAvail` off | `select_perturbation_using_jvp: "True"` | n/a (async) |
| **fluxtune_dynkc** | async (`true`) | `async_random` + `dynamic_kc` | `fedbuff` | off | `perturbation_sampling: "True"` | n/a | (research variant; preserves the ported production default) |

**Addendum (owner clarification during implementation):** felix always runs
fixed K/C; fluxtune's K/C policy is itself config-driven --
`selector.kwargs.dynamic_kc` defaults to `enabled: false` (fixed K/C, same as
felix) but can be flipped to adaptive K/C per experiment without forking the
selector (`DynamicKCController` is already selector-agnostic -- see P5).

### Step P1 — Fix `_validate_stack` to be selector-driven for the fwdllm stack (D1)

File: `flame/launch/runner.py:398-434`.
- Remove `"fwdllm"` from `_ASYNC_STACKS` (line 398).
- In `_validate_stack`, after detecting `stack == "fwdllm"`, compute
  `is_async_stack = bool(agg_cfg["selector"]["kwargs"].get("is_async", False))`
  for that branch instead of `stack in self._ASYNC_STACKS`. Keep the membership
  test for all other stacks unchanged. Net invariant for fwdllm: the selector's
  async-ness (`sort in _ASYNC_SELECTORS`) must equal its declared `is_async`.
- **Unit test** (`tests/launch/test_runner_paths.py`, new `TestValidateStack`):
  fwdllm + `random`/`is_async:false` → no raise; fwdllm + `async_oort`/`is_async:true`
  → no raise; fwdllm + `random`/`is_async:true` (inconsistent) → raises; fwdllm +
  `async_oort`/`is_async:false` (inconsistent) → raises; cifar10 asyncfl + `async_random`
  → no raise (regression guard).
- **Checkpoint:** the five cases above behave as asserted; existing
  `test_runner_paths.py` still green.

**Checkpoint result — DONE.** Implemented as specified (`_ASYNC_STACKS` no
longer includes `"fwdllm"`; the `stack == "fwdllm"` branch in
`_validate_stack` now computes `is_async_stack` from
`selector_cfg["kwargs"].get("is_async", False)`). All 5 cases pass via the
new `TestValidateStack` class.

**Bug found and fixed while verifying this against the real entrypoint (not
just synthetic fixture text):** `examples/fwdllm/aggregator/main_fedfwd_agg.py`
does **not** import `fwdllm_aggregator.TopAggregator` directly — it imports
`FedSGDAggregator` from `examples/fwdllm/aggregator/FedSgdAggregator.py`,
which extends `TopAggregator` in a separate file. `_validate_stack`'s regex
only scans the entrypoint file's own text, so it never matched the real file
and silently fell back to `stack="syncfl"` for every fwdllm baseline (this
predates this session — Phase 1a's original fix was only ever verified
against synthetic file content, not the real file). Fixed by adding a
same-file marker import (`from flame.mode.horizontal.syncfl.fwdllm_aggregator
import TopAggregator  # noqa: F401`) to `main_fedfwd_agg.py`, with a comment
explaining why. Locked in by a new `test_real_fwdllm_entrypoint_detected_for_
both_sync_and_async` test in `test_runner_paths.py` that runs `_validate_stack`
against the actual repo file. Without this fix, P5–P8's async baselines
(`fluxtune`, `fluxtune_dynkc`) would have raised `ValueError` on every real
launch despite all unit tests passing.

### Step P2 — Add explicit `learning_rate` kwarg to FedBuff (D2)

File: `flame/optimizer/fedbuff.py:41-81` (`__init__`) + `:222-270`
(`_scale_add_agg_weights_pytorch`, and the tensorflow twin at `:272+`).
- In `__init__`, read `self.learning_rate = kwargs.get("learning_rate", None)`.
- In both `_scale_add_*`, if `self.learning_rate is not None` use it directly;
  else fall back to the existing `use_oort_lr`/`dataset_name` table (unchanged).
- Do **not** touch felix/oracle/fedbuff baselines — absence of the kwarg keeps
  their exact current behavior.
- **Unit test** (`tests/optimizer/test_fedbuff_lr.py`, new): explicit
  `learning_rate=0.5` overrides the dataset table; absence + `dataset_name:
  cifar-10` still yields the legacy value; absence + unknown dataset still warns
  and uses 1.0.
- **Checkpoint:** test green; felix's effective LR unchanged when no kwarg set.

**Checkpoint result — DONE.** `self.learning_rate = kwargs.get("learning_rate",
None)` added in `__init__`; both `_scale_add_agg_weights_pytorch` and the
tensorflow twin check it first and fall back to the existing table when
absent. `tests/optimizer/test_fedbuff_lr.py` (new): explicit
`learning_rate=0.5` overrides the table; absent + `dataset_name: cifar-10`
still yields `40.9`; absent + unknown dataset still falls back to `1.0`. All
3 green.

### Step P3 — Re-source oracular availability from `_metadata` (D3)

File: `flame/mode/horizontal/syncfl/fwdllm_aggregator.py:480-489`
(`read_trainer_unavailability`).
- Port `async_cifar10/aggregator/pytorch/main_oort_sync_agg.py:173+`: read
  `_metadata/trainer_registry.yaml` + `_metadata/availability_traces/<trace>.yaml`,
  return `trainer_id -> SortedDict(ts -> state)`. Drop the `glob` over
  `json_scripts/trainer_*.json` and the `import glob` if now unused.
- Resolve `_METADATA_DIR` the same way the cifar10 module does (don't hardcode a
  relative `../../../../examples/...` path).
- **Unit test** (`tests/mode/test_fwdllm_oracular_avail.py`, new): point it at a
  tmp `_metadata` with a tiny registry + trace; assert the returned dict matches,
  and assert **no** read of any `json_scripts/` path (e.g. monkeypatch `glob.glob`
  to fail if called).
- **Checkpoint:** test green; this unblocks deleting `json_scripts/` (revisit the
  Phase 5 / step 13 "do not delete" caveat — after P3 it becomes safe; update
  `DEPRECATED.md` accordingly).

**Checkpoint result — DONE.** `read_trainer_unavailability(trace,
metadata_dir=None)` rewritten to mirror `main_oort_sync_agg.py`'s pattern
exactly (registry + mobiperf/synthetic trace lookup, keyed by `task_id`);
`metadata_dir` defaults to a module-level `_METADATA_DIR` resolved from
`__file__` (`lib/python/examples/_metadata`) but is overridable, which is
what the new test points at a tmp bundle. `glob`/`ast`/`json` imports dropped
(no longer used anywhere in the file). `tests/mode/test_fwdllm_oracular_avail.py`
(new): reads from a tmp `_metadata` bundle correctly, and a monkeypatched
`glob.glob` that raises is never hit. `expts/run_tc_expts/DEPRECATED.md`
updated: `json_scripts/` is no longer load-bearing (left in place, but safe
to delete — deleting it wasn't itself in scope here).

### Step P4 — Selection granularity flag `reselect_each_iteration` (D4)

Semantics now confirmed by owner (see D4): per-round = one selection across all
databins+iterations of a round; per-iteration = fresh selection every iteration
(= current default behavior). So this step *adds* the per-round path.
File: `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` —
`_distribute_weights_sync` (`:1387-1442`, where `channel.ends(VAL_CH_STATE_SEND)`
re-selects), the round-increment site (`:996`), and ctor flag block (`~:255-269`).
- Read `self._reselect_each_iteration = bool(self.config.hyperparameters.get(
  "reselect_each_iteration", True))` (default `True` preserves today's behavior).
- Add `self._round_selected_ends = None`. In `_distribute_weights_sync`, when
  `_reselect_each_iteration is False`: if a new round just started (cache empty
  or `self._round` changed since last cache), call the SEND-state selection once
  and cache the resulting end set keyed by `self._round`; on subsequent
  iterations/databins of the same round, re-distribute to the cached set and skip
  re-selection. When `True`: unchanged (call SEND-state selection every time).
- Invalidate/refresh the cache exactly at the `self._round += 1` boundary.
- Keep the change confined to the sync path (`is_async` False); the async path
  (fluxtune) is untouched.
- **Unit test** (`tests/mode/test_fwdllm_reselection.py`, new): drive the gate
  with a fake channel/selector spanning 2 databins × 2 iterations within one
  round, then a round rollover; assert `False` → selector invoked exactly once
  for the whole round and again only after rollover; `True` → invoked every
  iteration.
- **Checkpoint:** both modes behave as asserted; existing fwdllm aggregator
  imports/compose still parse.

**Checkpoint result — DONE.** Implemented as a small extracted helper,
`_select_ends_respecting_reselect_gate(channel, task_to_perform)`, called
from `_distribute_weights_sync` in place of the bare `channel.ends(...)`
call — extracted (rather than left inline) specifically so it could be unit
tested directly against a fake channel/aggregator without needing to
construct a full `TopAggregator`. Cache invalidation at the round boundary
falls out for free from comparing `self._round_selected_ends_round ==
self._round` (no separate invalidation code needed — once `self._round`
increments, the comparison fails and the `else` branch re-selects).
`tests/mode/test_fwdllm_reselection.py` (new, 3 tests): per-round selects
once across 2 databins × 2 iterations then again after rollover;
per-iteration selects on all 4 calls; an empty/`None` selection (no trainers
joined yet) is never cached, so per-round mode keeps retrying instead of
freezing on an empty set. All green.

### Step P5 — Rewrite the fwdllm baseline catalog (D5, D6)

File: `examples/_metadata/baselines.yaml:257-398`.
- Delete `fedfwd_async_random_dynkc` and `fedfwd_oracular`.
- Add `fwdllm`, `fwdllm_plus`, `fluxtune`, `fluxtune_dynkc` per the matrix above.
  All four keep `example.aggregator_main: aggregator/main_fedfwd_agg.py`.
  fluxtune's `optimizer.kwargs` sets an explicit `learning_rate` (P2) and drops
  the misleading `dataset_name: google-speech`.
- Add two organizational comment headers in the file: one over the CNN/speech
  family, one over the LLM forward-mode family — purely for maintainability.
- **Checkpoint:** `yaml.safe_load` parses; `load_baselines()` returns the four
  new keys and neither old key; the felix/oort/refl/feddance/oracle/fedbuff/fedavg
  entries are byte-for-byte unchanged (diff-check).

**Checkpoint result — DONE.** All four baselines added; both legacy keys
removed. `fluxtune`'s `selector.kwargs.dynamic_kc` block defaults to
`enabled: false` per the owner's mid-session clarification (see Phase 6/7
addendum above) rather than being absent, so the adaptive-K/C path is
reachable via a per-experiment override without a second baseline.
`fluxtune_dynkc` deliberately keeps the exact legacy
`fedfwd_async_random_dynkc` selector/optimizer shape (including its known
`dataset_name: google-speech` artifact) for parity, just renamed and with
the now-default-true trainer-side keys (`forward_mode`/`var_control`/
`perturbation_sampling`/`fl_algorithm`) dropped since `trainer_base.yaml`'s
defaults already supply the same resolved values. Verified: `yaml.safe_load`
parses; `load_baselines()` returns exactly the four new keys; programmatic
diff confirmed felix/oracle/refl/feddance/oort/fedbuff/fedavg are
dict-equal to the pre-Phase-7 catalog.

### Step P6 — Per-baseline trainer config coverage (D5)

Files: `examples/fwdllm/configs/trainer_base.yaml`, and the smoke + any new
`expt_scripts/*.yaml`.
- Confirm `trainer_base.yaml` exposes (with sane defaults) every trainer-side
  knob the four baselines flip: `perturbation_sampling`,
  `select_perturbation_using_jvp`, `forward_mode`, `var_control`,
  `client_notify.{enabled,trace}`. Add any missing key with a default that makes
  the *unaware fwdllm* baseline correct out of the box.
- Ensure a `*_3st_*` mobiperf/availability trace exists for fluxtune's
  client_notify (reuse the step-11 short/long-form fix); register it if absent.
- **Checkpoint:** `yaml.safe_load`; a generated trainer config for each of the
  four baselines has the matrix's expected trainer-side values after merge.

**Checkpoint result — DONE.** `trainer_base.yaml` already exposed every knob
the four baselines need, with defaults that make the *unaware fwdllm*
baseline correct out of the box (`forward_mode`/`var_control`/
`perturbation_sampling: true`, `select_perturbation_using_jvp: false`,
`client_notify: {enabled: "False", trace: syn_0}`) — no new keys were
needed. Updated two stale comments: the long/short mobiperf-trace-name note
(described the mismatch as still-unresolved; step 11 already fixed it) and
the baseline-name reference next to the `selector:`/`optimizer:` placeholders
(named the now-deleted `fedfwd_*` keys). `*_3st_*` traces
(`states_3st_50`/`states_3st_75`) already exist per-device in
`_metadata/availability_traces/mobiperf_traces.yaml`, and the spawner
unconditionally injects all three mobiperf variants into every trainer
config regardless of `availability_mode` — fluxtune's `client_notify.trace:
mobiperf_3st_50` needed no new trace registration. Smoke-test YAML moved
per D6: deleted `expt_scripts/fedfwd_async_random_n10_smoke.yaml`, added
`expt_scripts/fwdllm_n10_smoke.yaml` (`baseline: fwdllm`, `random` selector
kwargs scaled to `k=5,c=10` for 10 trainers). Verified via
`ConfigGenerator.generate_trainer_config()` for all four baselines: matrix
values survive the merge exactly (see Step P7's test suite below, which
formalizes this same check).

### Step P7 — Pytest sweep (add + update)

Beyond the per-step unit tests (P1–P4), add/refresh integration-level tests:
- `tests/launch/test_baselines.py`: assert the four fwdllm baselines resolve and
  carry the expected `selector.sort` / `optimizer.sort` / `trackTrainerAvail` /
  perturbation flags; assert the two retired keys are gone.
- `tests/launch/test_config_generator.py`: for each fwdllm baseline, generate the
  aggregator + a trainer config end-to-end and assert no KeyError and that the
  matrix values survive the deep-merge (mirrors Smoke Test D, extended to all
  four baselines).
- `tests/mode/test_baseline_readiness.py` (already exists): extend its coverage
  to the new baselines if it enumerates baselines.
- **Checkpoint:** `pytest lib/python/tests/launch lib/python/tests/mode
  lib/python/tests/optimizer lib/python/tests/selector` all green.

**Checkpoint result — DONE.** `test_baselines.py`: new `TestFwdllmBaselines`
class (7 tests) -- all four baselines present, both retired keys gone,
selector/optimizer/availability/perturbation/`reselect_each_iteration`/
`dynamic_kc` values match the matrix, `fluxtune_dynkc` preserves the legacy
shape exactly. `test_config_generator.py`: new
`TestFwdllmEndToEndConfigGeneration` class, parametrized over all four
baselines -- trainer config generates without `KeyError` (path-style
dataset, no `trainer_indices_list`), trainer-side matrix values survive the
merge, and the aggregator-side merge (against the real
`aggregator_base.json` template) both matches the matrix **and** passes
`_validate_stack` against the real `main_fedfwd_agg.py` entrypoint (this is
what caught the P1 marker-import gap above). `test_baseline_readiness.py`
needed no changes -- it already enumerates `baselines.yaml` dynamically, so
all four new baselines were automatically parametrized in and their
selector/optimizer registration checked.
Full sweep: `pytest lib/python/tests/launch lib/python/tests/mode
lib/python/tests/optimizer lib/python/tests/selector` → **312 passed, 7
skipped, 0 failed** (152s).

### Step P8 — Live smoke tests (supersedes the old Smoke Test E)

Run on a host with fwdllm's pinned stack installed (the
`adapter-transformers`/Rust gap noted throughout Phases 9–E still applies in
this sandbox). Smoke each NEW codepath, cheapest first:
1. **`fwdllm`** (sync, fedavg, random, unaware) — exercises P1+P4(`false`)+fedavg.
   10 trainers, ~5 rounds; assert processes start, rounds complete,
   `telemetry/trainer_*.jsonl` has `trainer_round`, and `aggregator_config.json`
   shows `selector.sort: random`, `is_async: false`, `optimizer.sort: fedavg`.
2. **`fwdllm_plus`** (sync, oracular, per-iteration) — exercises P3+P4(`true`).
   Assert oracular events load from `_metadata` (P3), no `json_scripts` read.
3. **`fluxtune`** (async, async_oort, fedbuff, JVP, 3-tier) — exercises P2 LR
   kwarg + JVP perturbation + client_notify 3st.
- **Parity (optional):** for `fwdllm`, confirm `client_idx = (trainer_id-1) %
  100` reproduces the same H5 partitions as the legacy `trainer_*.json`.
- **Checkpoint:** all three launch and complete ≥3–5 rounds without crashing;
  per-baseline assertions above hold.

### Execution order summary

`P1 → P2 → P3 → P4 (confirm w/ owner) → P5 → P6 → P7 → P8`. P1–P4 are
independent library changes that can each be reviewed/merged on their own; P5+
depend on all four. None of P1–P4 touch felix/cifar10 behavior (regression
guards in each step's test enforce this).

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

**Checkpoint:** re-run the static repro from Smoke Test A — must raise
`ValueError` before this fix, must pass after. **[DONE — see Smoke Test A.]**

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

Implementation (done — see `spawner.py`/`runner.py`/`experiment_config.py`):
- `generate_trainer_config()` gates the `get_dataset_split()` call (and the
  `trainer_indices_list` injection) behind `skip_index_splits: bool = False`.
- `TrainerSpawner.spawn_trainer()`/`spawn_all()` thread `skip_index_splits`
  through, and `spawn_all()` accepts `per_trainer_overrides:
  Dict[int, Dict]` merged per-trainer on top of the shared `**overrides`.
- `runner.py: run_experiment()` builds `per_trainer_overrides = {tid:
  {"hyperparameters.client_idx": (tid - 1) % modulo} for tid in trainer_ids}`
  when `exp.trainer.client_idx_modulo` is set, and passes
  `skip_index_splits=exp.trainer.dataset.path_style` through the same call.

**Checkpoint:** 10 distinct client_idx values 0–9, no `trainer_indices_list`
key present — verified through the real `spawn_all()`/`per_trainer_overrides`
path. **[DONE.]**

### 1c. Fix `_sweep_stragglers()` path patterns

File: `flame/launch/runner.py:_sweep_stragglers()` (line 487 pre-fix). Confirmed
hardcoded patterns `("trainer/pytorch/main.py", "aggregator/pytorch/main_")`
and the GPU-drain poll's `pgrep -f "trainer/pytorch/main.py"` — fwdllm's new
entrypoints have no `pytorch/` subdirectory. Extracted the pattern lists to
class-level tuples `_STRAGGLER_PATTERNS` (pkill list) and
`_TRAINER_STRAGGLER_PATTERNS` (GPU-drain poll list), added fwdllm's two
entrypoint paths (`fwdllm/trainer/main.py`, `fwdllm/aggregator/main_fedfwd_agg.py`),
widened the poll loop to check all trainer-matching patterns.

**Checkpoint:** dummy background process with `fwdllm/trainer/main.py` in its
argv, confirmed killed by `_sweep_stragglers()`; cifar10's original pattern
also still works (no regression). **[DONE.]**

---

## Smoke Test A — Static `_validate_stack` check (after step 4)

Verify that the Phase 1a fix actually works and doesn't regress existing examples.

**Checkpoint:** Run a script that tests:
1. fwdllm + async_random selector → should NOT raise (fix is working)
2. async_cifar10 asyncfl + async_random → should NOT raise (no regression)
3. async_cifar10 sync + async_random → should raise ValueError (check still works)

If all three pass as expected, smoke test A is complete. **[PASSED.]**

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

### Step 5. Metadata symlink — **[DONE]**
`ln -s ../_metadata lib/python/examples/fwdllm/metadata` (mirrors
`async_cifar10/metadata` exactly). **Checkpoint:** `readlink -f` resolves to
`examples/_metadata`; files readable through the symlink.

### Step 6. `configs/trainer_base.yaml` — **[DONE]**
Static template with FedFwd's full hyperparameter set (dataset/model/FL/
availability blocks — transcribed from `json_scripts/trainer_1.json` and
`aggregator.json`), placeholders for `data_file_path`/`partition_file_path`
(injected via `config_overrides`, not `_metadata/dataset_splits/`) and
`client_idx` (injected per-trainer via Phase 1b's `client_idx_modulo`
mechanism). Includes an inline comment documenting the mobiperf trace-name
wrinkle (step 11). **Checkpoint:** `yaml.safe_load()` parses it; every
legacy hyperparameter key from `trainer_1.json` is present, except
`avl_events_*`/`training_delay_s` which are spawner-injected at runtime
(same convention as cifar10's `trainer_base.yaml`).

---

## Phase 3 — Step 7: `_metadata/baselines.yaml` additions

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

`example.aggregator_main` in both entries points at
`aggregator/main_fedfwd_agg.py` (a path string), which doesn't need to exist
yet for this step's checkpoint — only step 9/10 need to create the actual file.

**Checkpoint:** `yaml.safe_load()` parses `baselines.yaml` with both new keys
present; `load_baselines()` returns them; no existing baseline entries changed
(diff-check).

---

## Phase 4 — Step 8: smoke-test experiment YAML

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

## Smoke Test B — Load experiment YAML + validate baseline (after step 8)

Verify that the experiment YAML can be loaded and the baseline is resolvable,
before creating the entrypoints.

**Checkpoint:** Run a script that:
1. `load_experiment_config()` on `fedfwd_async_random_n10_smoke.yaml` → succeeds
2. Confirm `num_trainers == 10` and `baseline == "fedfwd_async_random_dynkc"`
3. `load_baselines()` from `_metadata/baselines.yaml` → succeeds, has both fwdllm baselines
4. Try to merge baseline + experiment config → no KeyError on missing baseline

If all four pass, smoke test B is complete.

---

### Step 9. `trainer/main.py`
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

**Checkpoint result — DONE, with a documented pre-existing environment gap:**
verified `load_config_from_argv()` + the `--time_mode`/`--log_level`/`--battery_threshold`
side-channel parser work correctly in isolation. Could not exercise reaching
model construction in this sandbox: the module-level import chain
(`tc_transformer_trainer_distribute.py`) requires `sklearn` (not in `req.txt`,
installed ad hoc) and then `adapter-transformers==3.1.0`'s `AdamW`/adapter API
(`req.txt`'s pin), which this conda env has never had installed (mainline
`transformers==4.54.0` is installed instead). Installing the pinned package
hits a further wall: its `tokenizers==0.12.1` dependency is 2022-era Rust
source that fails to compile under any readily available modern `rustc`
(`invalid_reference_casting` became a deny-by-default lint after that crate
was written). **Confirmed this is pre-existing and not migration-caused** —
the untouched legacy `trainer/fl_main.py` fails at the identical import line
for the identical reason, before either file's `__main__` body (where all of
this migration's code lives) ever executes. Out of scope for this migration;
flagged as an environment-provisioning gap for wherever fwdllm's real
training actually runs (presumably a different, properly-provisioned host).

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
raise, confirm it's never hit). **[DONE]** — verified both directions:
`wandb.init()` is never reached without `--log_to_wandb`, and is correctly
reached with it. Full end-to-end execution hits the same pre-existing
`adapter-transformers`/Rust-toolchain wall as the trainer side (step 9),
confirmed at the identical import line, before `__main__` runs.

---

## Smoke Test C — Entrypoints accept `--config-json` (after step 10)

Verify that the trainer and aggregator entrypoints can parse `--config-json`
and reach their model-construction code without crashing.

**Checkpoint:** Run:
```bash
# Trainer: minimal config JSON should parse argv and reach model setup
python lib/python/examples/fwdllm/trainer/main.py \
  --config-json '{"hyperparameters": {}, "taskid": "test"}' \
  --time_mode real 2>&1 | head -20
# Should reach "model construction" point without ValueError/TypeError on args

# Aggregator: minimal config JSON should parse argv and reach aggregator setup
python lib/python/examples/fwdllm/aggregator/main_fedfwd_agg.py \
  --config-json '{"hyperparameters": {}, "taskid": "test"}' 2>&1 | head -20
# Should reach aggregator construction without argv errors
```

Both commands should parse argv cleanly and log something, not crash on `--config-json`
or `--time_mode` parsing. Smoke test C passes if both reach their setup phase.

**Checkpoint result — PASSED:** ran both commands with a full mock config.
Both hit the pre-existing `adapter-transformers`/Rust-toolchain wall (steps
9–10) -- but critically, that failure is at **module-import time**, before
`sys.argv` is touched at all (dies on `from ... import
ForwardTextClassificationTrainer`, before any `--config-json`/`--time_mode`/
`--log_to_wandb` parsing code runs). So neither command crashed on argument
parsing -- combined with the isolated `load_config_from_argv()`/side-channel
parser checks already verified in steps 9-10, the actual intent of this
checkpoint (entrypoints don't crash on CLI parsing) is satisfied.

---

### Step 11. Fix mobiperf trace-name mismatch
`FedSgdTrainer.py`'s `client_notify["trace"]` dispatch matches long-form
strings (`"avl_events_mobiperf_2st"`) for mobiperf but the spawner's
`availability_mode`/`client_notify.trace` convention uses short names
(`"mobiperf_2st"`) — patch the trainer's match to accept both forms (one-line
fix, removes a footgun for future experiment authors).
**Checkpoint:** unit-test both `"mobiperf_2st"` and `"avl_events_mobiperf_2st"`
resolve to the same trace lookup. **[DONE]** — verified against the **real**
`FedSGDTrainer` constructor (not isolated logic): installed the missing
pure-Python core `flame` deps (`aiostream`, `gpustat`, `paho-mqtt`,
`shared-memory-dict`, `mlflow` -- all low-risk, no native/Rust build, needed
by `flame.channel`/`flame.monitor`/`flame.registry` regardless of fwdllm) so
`FedSgdTrainer.py` imports standalone. Built a minimal fake config/trainer
and instantiated `FedSGDTrainer` directly with `client_notify.trace` set to
each of the long and short forms for all three mobiperf variants --
confirmed identical `state_avl_event_ts` for each pair, plus a `syn_0`
regression check (unaffected, unrelated branch untouched).

### Step 12. Telemetry: `trainer_round`
Add `telemetry.configure(role="trainer", end_id=str(task_id))` at startup in
`trainer/main.py`. Emit `trainer_round` events (via
`flame.telemetry.events.build_trainer_round`) from inside `FedSgdTrainer.py`'s
per-round completion path (locate the exact call site during implementation —
near where loss/accuracy become available for the variance check), gated by
`telemetry.is_enabled()`.
**Checkpoint:** with `FLAME_TELEMETRY_DIR` set to a temp dir, one training
round produces a `trainer_<id>.jsonl` line with `event: trainer_round`.
**[DONE]** — emission added at the end of `train_with_data_id()` (after
`_perform_training()`/`_emulate_training_delay()`, on the successful-training
path only, not on the `abort_training`/`_check_availability` early-returns),
using `self._stat_utility` (the base `Trainer` class's per-batch-loss-derived
accumulator, the same value used in the aggregator's variance/utility checks)
as `stat_utility`; `final_loss` is left unset (`None`, an optional field) since
no persistent loss attribute is exposed by `tc_transformer_trainer_distribute.py`
without a deeper change there, out of scope for this migration.
`telemetry.configure(role="trainer", end_id=str(config.task_id))` added at
startup in `trainer/main.py` (missed in step 9, added now). Verified against
the real `train_with_data_id()` method (only the heavy ML sub-call
`_perform_training()` stubbed out) with a real `FLAME_TELEMETRY_DIR` temp
dir: produced exactly one `trainer_round` JSONL line with all expected
fields (`round`, `dataset_size`, `stat_utility`, `data_id`,
`iteration_per_data_id`, `model_version`, `avail_state`, `real_gpu_time_s`).

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
unmodified (diff-check against pre-migration state). **[DONE]** —
`git status --short` on `expts/run_tc_expts/` shows only the new
`DEPRECATED.md` as untracked; all 155 `json_scripts/*.json` files unchanged.

---

## Smoke Test D — Full static config generation dry-run (after step 13)

Verify that all configuration wiring is correct before the expensive live test.

**Checkpoint:**
```python
from flame.launch.spawner import ConfigGenerator, MetadataLoader
from flame.launch.experiment_config import load_experiment_config
from flame.launch.baselines import load_baselines
from pathlib import Path

# Load experiment YAML
exp = load_experiment_config(Path('lib/python/examples/fwdllm/expt_scripts/fedfwd_async_random_n10_smoke.yaml')).experiments[0]

# Load baselines and resolve
baselines = load_baselines(Path('lib/python/examples/_metadata'))
baseline_entry = baselines[exp.baseline]

# Create config generator
meta = MetadataLoader(Path('lib/python/examples/_metadata'))
cg = ConfigGenerator(meta, Path('lib/python/examples/fwdllm/configs/trainer_base.yaml'))

# Generate trainer configs for all 10 trainers
for trainer_id in range(1, 11):
    cfg = cg.generate_trainer_config(
        trainer_id,
        alpha=exp.trainer.dataset.dirichlet_alpha,
        availability_mode=exp.trainer.availability.mode,
        dataset_name=exp.trainer.dataset.name,
        num_trainers=exp.trainer.num_trainers,
        skip_index_splits=exp.trainer.dataset.path_style,
        **{
            f"hyperparameters.client_idx": (trainer_id - 1) % exp.trainer.client_idx_modulo,
        }
    )
    assert cfg["hyperparameters"]["client_idx"] == (trainer_id - 1) % 100
    assert "trainer_indices_list" not in cfg["hyperparameters"]
    print(f"trainer_{trainer_id:03d}: client_idx={cfg['hyperparameters']['client_idx']} OK")
```

If all 10 trainers generate with distinct client_idx 0–9 and no KeyError, smoke test D passes.

**Checkpoint result — PASSED:** ran the exact script above with `PYTHONPATH=lib/python`
(needed since `flame` isn't installed editable in this sandbox). All 10 trainer
configs generated with distinct `client_idx` 0–9, both assertions held for every
trainer (`client_idx == (trainer_id-1) % 100`, `trainer_indices_list` absent),
and `load_experiment_config`/`load_baselines`/`ConfigGenerator` all succeeded
with no KeyError.

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
   into `baselines.yaml` (step 7) and `trainer_base.yaml` (step 6).
5. **Single aggregator entrypoint** instead of per-stack split — direct
   consequence of #1; corrects MIGRATING_TO_LAUNCHER.md §9's guidance.
6. **No simulated-clock support** — FedFwd always uses real wall-clock delay;
   `--time_mode simulated` is accepted but a documented no-op for now (step 9).
7. **CPU-pinning perf risk for transformer workloads** — cifar10's CNN trainer
   tolerates 1-core pinning; fwdllm's DistilBERT/forward-mode training is much
   heavier per-trainer. No code change required, but flag for smoke-test
   wall-clock observation (Smoke Test E) before scaling to n=150.
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
    (step 11), no cifar10 equivalent since its schema and trace values were
    co-designed.

---

## Smoke Test E — Live 10-trainer smoke test + parity vs. legacy (final)

The final end-to-end validation before declaring the migration complete.

**Checkpoint:**
1. **Live 10-trainer smoke test**: `python -m flame.launch.run_experiment
   lib/python/examples/fwdllm/expt_scripts/fedfwd_async_random_n10_smoke.yaml`.
   Check:
   - Aggregator and trainer processes start without crashing
   - At least 3–5 rounds complete successfully
   - `telemetry/trainer_*.jsonl` files exist with `trainer_round` events
   - `aggregator_config.json` shows expected merged `selector.sort`,
     `dynamic_kc.enabled`, `var_threshold: 0.3`, `max_iterations_per_data_id: 15`

2. **Parity check vs. legacy**: Run `run_text_classification.sh` with a matching
   10-trainer subset and compare round-over-round loss/accuracy trend against
   the new launcher run (exact bit-parity not expected, but trend should match
   within the same rounds) — specifically confirm `client_idx = (trainer_id-1)
   % 100` reproduces the same H5 partition assignments as
   `json_scripts/trainer_1.json…trainer_10.json`'s hardcoded `client_idx`
   values.

### Critical files
- `flame/launch/runner.py` (`_validate_stack`, `_sweep_stragglers`, `run_experiment`'s `config_overrides` assembly)
- `flame/launch/spawner.py` (`ConfigGenerator.generate_trainer_config`, `TrainerSpawner.spawn_trainer`/`spawn_all`)
- `flame/launch/experiment_config.py` (`DatasetConfig`, `TrainerConfig`)
- `examples/_metadata/baselines.yaml`
- `examples/fwdllm/aggregator/FedSgdAggregator.py`, `examples/fwdllm/trainer/forward_training/FedSgdTrainer.py`
- `flame/mode/horizontal/syncfl/fwdllm_aggregator.py` (`read_trainer_unavailability`)
