# PR-readiness plan for `launcher-script-fwdllm`

Working plan for cleanup before opening the PR. See conversation history /
this file for full context if resuming cold.

---

## NEXT SESSION — run these on a healthy node (read this first)

**Context**: `jayne.cc.gatech.edu`'s GPU driver is wedged (confirmed twice
this branch's history — TSO not yet fixed as of this session). Do not run
anything GPU-touching there. `shepherd.cc.gatech.edu` is confirmed healthy
(same `/home/dgarg39/flame` checkout, passwordless SSH, `aish_smoke_flame`
conda env) — use it, or any other healthy node with this checkout.

All static verification (comment cleanup, doc consolidation, JSON→YAML field
comparison) is **done** — see "Completed this session" below. What's left
needs a live GPU run: (A) confirm the 3 launcher baselines actually train
end-to-end, (B) confirm the runtime-injected config matches the migrated
data, (C) decide the 5 open questions below, (D) commit + open the PR.

### 0. Pre-flight

```bash
ssh shepherd.cc.gatech.edu
cd /home/dgarg39/flame && git status   # confirm you're on launcher-script-fwdllm, working tree matches this session's edits
conda activate aish_smoke_flame        # or: export FLAME_CONDA_ENV=<your env>
nvidia-smi                             # sanity: no stuck D-state python/nvidia-smi processes from other users
```

### 1. Test suite (deferred from last session)

```bash
cd /home/dgarg39/flame
python -m pytest lib/python/tests/mode/test_fwdllm_oracular_avail.py -q
python -m pytest lib/python/tests/launch lib/python/tests/mode -q
```
Both must be green (Workstream A was comment/doc-only; Workstream B added
data, not logic — no regressions expected, but this wasn't confirmed on a
healthy node yet).

### 2. Minimal 3-baseline smoke test

Deliberately kept at the **checked-in n10 default**, not scaled to 100+ —
scaling up is what caused the original GPU-undersubscription/OOM incident on
`jayne`, and 10 trainers is already enough to exercise real gradient
computation, aggregation, the staleness-policy gate, and (for
`fwdllm_plus`/`fluxtune`) real availability-trace lookups. `--max-data-id 2`
is the fastest bar that still forces at least one full data_id transition
(proves the var-check/staleness-gate logic actually fires, not just that
processes spawn) with `--max-runtime-s` as a safety net so a hang doesn't
run forever.

```bash
cd /home/dgarg39/flame/lib/python/examples/fwdllm/expt_scripts
./run_sequential.sh --max-data-id 2 --max-runtime-s 300
# ~3 runs x up to 5 min each = well under 20 min total if all three behave.
# Do NOT pass --num-trainers/--num-gpus (leave each YAML's own small n10/8-GPU default).
```

### 3. Verification checklist (run after step 2)

```bash
cd /home/dgarg39/flame/lib/python/examples/fwdllm/expt_scripts
LOGDIR=$(ls -td smoke_logs/*/ | head -1)
echo "Using $LOGDIR"

# (a) No crashes, no leftover zombie trainers (the jayne-incident signature)
for f in "$LOGDIR"*.out; do
  echo "=== $f ==="
  grep -c "Traceback\|ERROR" "$f"
  grep "did not exit within" "$f" | wc -l   # must be 0
  grep "stopping run\|max_data_id_progress=.*reached\|max_runtime_s=.*reached" "$f" | tail -3
done

# (b) Aggregator's own field-provenance report (already built into the
# launcher's stdout) -- confirms every hyperparameter's source and catches
# any field still stuck on PLACEHOLDER. Read the "[aggregator] field
# provenance:" block near the top of each *.out file.
grep -A60 "field provenance" "$LOGDIR"fwdllm.out | head -70
grep -A60 "field provenance" "$LOGDIR"fwdllm_plus.out | head -70
grep -A60 "field provenance" "$LOGDIR"fluxtune.out | head -70

# (c) No PLACEHOLDER leaked into any generated aggregator config
for f in ../experiments/run_*/aggregator_config.json; do
  echo "=== $f ==="; grep -c PLACEHOLDER "$f"   # must be 0
done

# (d) Per-trainer availability trace assignment actually happened (spot
# check trainer logs for "Set avl_events_..." / "Set avl_events_mobiperf_..."
# lines -- confirms _metadata trace injection, not just config generation)
grep "Set avl_events" ../experiments/run_*fwdllm_plus*/*.log | head -5
grep "Set avl_events" ../experiments/run_*fluxtune*/*.log | head -5
```

If everything above is clean (no tracebacks, no PLACEHOLDER, no stuck
processes, at least one data_id transition per baseline, provenance report
shows fields coming from the expected source), that's sufficient evidence
that **(i) the three baselines work** and **(ii) the migrated data is wired
correctly at runtime**, on top of this session's static field-by-field
comparison (see below). No need to re-derive the static comparison — it's
already done.

### 4. Open questions — resolve with the user before finalizing

Found while doing the static JSON→YAML comparison this session (none are
blockers for the smoke test above, but should be resolved before the PR is
final):

1. **`configs/aggregator_base.json`'s `hyperparameters.rounds: 1000`** — no
   baseline in `_metadata/baselines.yaml` overrides `rounds`, so any
   *non-smoke* (real) launcher run defaults to 1000 rounds, not the legacy
   production value of 300 (`json_scripts/aggregator.json`'s `rounds: 300`).
   Intentional bump, or should this be corrected to 300 (or made explicit
   per-baseline)?
2. **`configs/aggregator_base.json` has dead/leftover boilerplate**: a
   top-level `"dataset"` field pointing at an MNIST URL and a
   `"dependencies": ["numpy >= 1.2.0"]` field — neither is read by any
   fwdllm code path (confirmed via grep); both look copy-pasted from a
   different example's template. Also `hyperparameters.batchSize: 32` /
   `learningRate: 0.01` (flame's generic camelCase fields) sit alongside the
   real `train_batch_size: 8` / `learning_rate: 0.01` fields fwdllm actually
   reads, with mismatched values — harmless today (unused) but confusing.
   OK to clean up in this PR, or defer to a follow-up?
3. **Three legacy aggregator JSON variants have no 1:1 new-baseline
   equivalent**: `json_scripts/aggregator_async_base.json`,
   `aggregator_async_dynk.json`, `aggregator_async_maxiter.json` (all
   `async_random` + `fedbuff`, varying dynamic_kc/maxiter settings). Only
   `aggregator.json` (→ `fluxtune_dynkc`) and `aggregator_dynamic_kc.json`
   (→ `fluxtune`, with the dataset_name→explicit-learning_rate fix already
   documented in §9) were carried forward. Are the other three safe to treat
   as superseded exploratory variants (no preservation needed), or does one
   need to become a 5th named baseline?
4. **`aggregator/fl_main.py` and `trainer/fl_main.py`** (not under
   `expts/`) are unreferenced by `_metadata/baselines.yaml` or any smoke
   YAML — look like the pre-launcher-migration predecessors to
   `aggregator/main_fedfwd_agg.py` / `trainer/main.py`. Deletion candidates,
   but flagged separately/lower-confidence from the list below since they
   sit outside the clearly-legacy `expts/run_tc_expts/` tree — confirm
   before adding to the deletion PR.
5. Confirm the smoke-test results in §3 above look right to you (data_id
   actually progressed, no unexpected staleness-policy rejections spamming
   the log, etc.) — I can't eyeball a live run from here.

### 5. Deletion candidates (separate PR — do NOT delete yet)

Confirmed **safe to delete** (legacy MPI/JSON launch path, superseded by
`flame.launch` + the YAML experiments in `expt_scripts/`; nothing in the new
path imports these):
- `lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/` (155 files:
  5 legacy aggregator `*.json` variants + `trainer_0.json..trainer_149.json`)
- `lib/python/examples/fwdllm/expts/run_tc_expts/run_three_experiments.sh`
- `lib/python/examples/fwdllm/expts/run_tc_expts/run_three_parallel.sh`
- `lib/python/examples/fwdllm/expts/run_tc_expts/run_text_classification.sh`
- `lib/python/examples/fwdllm/expts/run_tc_expts/launch_single_run.py`
- `lib/python/examples/fwdllm/expts/run_tc_expts/gpu_mapping.yaml`
- `lib/python/examples/fwdllm/expts/run_tc_expts/mpi_host_file`
- `lib/python/examples/fwdllm/expts/run_tc_expts/fedavg_main_tc.py`
  (confirmed: not imported by anything else, unlike its sibling
  `initializer.py` below)

**Do NOT delete** — confirmed still-live dependencies of the new path:
- `lib/python/examples/fwdllm/expts/initializer.py` — imported directly by
  `aggregator/main_fedfwd_agg.py` and `trainer/main.py` (the new launcher
  entrypoints), not just the legacy `fedavg_main_tc.py`/`fl_main.py`.

**Lower-confidence, needs your confirmation** (open question 4 above):
`aggregator/fl_main.py`, `trainer/fl_main.py`.

**Not code, no action needed**: `expts/run_tc_expts/cache_dir/` is already
gitignored (253M local data cache, untracked).

Write the actual deletion PR only after this smoke test confirms the YAML
path fully replaces the JSON path's behavior, and open questions 1-4 above
are resolved.

---

## Completed this session (crisped)

**Workstream A — comment trim + doc consolidation.** Done, uncommitted.
- Added a "Lessons from smoke-testing fwdllm" subsection to
  `MIGRATING_TO_LAUNCHER.md` §9 (CUDA-before-logging, var-check n<2 guard,
  `staleness_policy` modes, `run_sequential.sh` conventions, `syn_train_*`
  traces pointer).
- `MIGRATION_TO_LAUNCHER_FWDLLM.md` reduced to a pointer stub at §9 + `git
  log`, matching the repo's `DEPRECATED.md` convention.
- Trimmed verbose/changelog-style comments in `run_sequential.sh` and
  `fwdllm_aggregator.py`, then swept the repo for now-dangling
  `MIGRATION_TO_LAUNCHER_FWDLLM.md Phase N`/`decision DN` references (~19
  files: `baselines.yaml`, `trainer_base.yaml`, `README.md`,
  `main_fedfwd_agg.py`, `FedSgdTrainer.py`, all 3 smoke YAMLs,
  `aggregator_spawner.py`, `experiment_config.py`, 6 test-file docstrings).
  `async_cifar10/` and `dynamic_kc_design.md`'s own "Phase N" text left
  alone (unrelated, legitimate design-doc numbering). All edited
  Python/YAML/shell re-parsed clean (`ast.parse`/`yaml.safe_load`/`bash -n`).
- **A4 (pytest) not yet run** — see §1 above.

**Workstream B — port missing `syn_train_*` availability traces.** Done,
uncommitted.
- `_metadata/availability_traces/synthetic_traces.yaml`: 3 new trace keys
  (`syn_train_100_eval_0_unavail_0`, `syn_train_90_eval_10_unavail_0`,
  `syn_train_50_eval_30_unavail_20`), ported from the 150-trainer JSON
  source, trainers 151-300 wrap 1-150. Verified against source with a
  temporary migrate/verify script pair (run, confirmed pass, deleted).
- `configs/trainer_base.yaml`: updated the stale "have no `_metadata`
  equivalents" comment. Spawner auto-injection for these 3 keys is
  explicitly NOT wired up (future work, out of scope for this PR).

**Static JSON→YAML migration verification** (no GPU needed, done this
session):
- `trainer_base.yaml` vs legacy `trainer_1.json`: field-by-field match
  confirmed, only cosmetic diffs (string vs. int types on 2 fields, and the
  default `client_notify.trace` differs — `syn_0` vs. the source's
  `avl_events_syn_train_100_eval_0_unavail_0` — but both are the same
  always-available degenerate trace, functionally identical).
- `mobiperf_traces.yaml`: confirmed it has the `states_2st`/`states_3st_50`/
  `states_3st_75` per-device sub-keys that `fwdllm_aggregator.py`'s
  `_TRACE_KEY_TO_MOBIPERF_SUB` expects — this data was already correctly in
  place pre-session (not part of Workstream B).
- `fluxtune_dynkc` baseline vs. legacy `json_scripts/aggregator.json`:
  confirmed faithful transcription (same `async_random` selector +
  `dynamic_kc` config).
- A stuck-but-informative prior run
  (`expt_scripts/smoke_logs/20260630_122734/fwdllm_plus.out`, the actual
  `jayne`-incident run at n100 scale) already shows `fwdllm_plus`'s
  generated `aggregator_config.json` fully resolved with zero
  `PLACEHOLDER` leaks and all fields from the expected source — strong
  evidence config generation itself is correct; only real training progress
  was never confirmed (killed by the wedged GPU before any rounds
  completed). That's what §2-3 above is for.
- Open questions from this pass are in §4 above, not resolved unilaterally.

---

## Original task context (for full history)

1. **Comment verbosity** across the branch (changelog-style narration
   instead of durable *why*) and two overlapping migration docs — resolved
   by Workstream A above.
2. **Missing trace data**: 2 of 3 `syn_train_*` traces had real per-trainer
   data sitting unused in the legacy JSON that was never ported to
   `_metadata/` — resolved by Workstream B above.

User-approved decisions (from earlier in this branch's work):
- Trainers 151-300 (no source data) get the 150 source trainers' traces
  doubled/wrapped, preserving the named distribution rather than a
  placeholder.
- Migration/verification scripts are temporary: write, run, confirm pass,
  delete.
- `MIGRATION_TO_LAUNCHER_FWDLLM.md`'s durable content folds into §9 without
  duplication; the file itself becomes a pointer stub, not a silent delete.

---

## Verification (end-to-end, before opening the PR)

- [ ] §1-3 above green on a healthy node.
- [ ] Open questions in §4 resolved.
- [ ] Commit Workstream A (~19 files) and Workstream B (2 files) — probably
  as two separate commits, matching this branch's one-commit-per-logical-
  change granularity. Decide whether `PR_CLEANUP_PLAN.md` belongs in the PR
  or stays a local note; `expt_scripts/smoke_logs/` stays untracked either
  way.
- [ ] Final `git diff dg-fork-main...HEAD --stat` reviewed for leftover
  noise. **Note**: `dg-fork-main` is the real branch point (50 commits
  back) — `origin/main` on this fork is ~890 commits ahead/~415 behind
  HEAD from unrelated upstream sync drift and diffing against it pulls in
  the whole repo, not this branch's changes.
- [ ] Deletion candidates in §5 above written up as a **separate** PR, not
  bundled into this one.
