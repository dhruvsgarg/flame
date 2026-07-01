# PR-readiness plan for `launcher-script-fwdllm`

Working plan for cleanup before opening the PR. This file will be **deleted
once the PR merges** — the one thing worth keeping past that point is
[`DELETION_CANDIDATES.md`](DELETION_CANDIDATES.md), which already has its own
copy of the legacy-code deletion list for the follow-up PR.

---

## Status: smoke-tested green on shepherd. One thing left before merge.

All three baselines ran end-to-end on `shepherd.cc.gatech.edu`
(`test_fwdllm` conda env) via
`./run_sequential.sh --max-data-id 2 --max-runtime-s 300`:

```
fwdllm                    PASS            436s
fwdllm_plus                PASS            186s
fluxtune                   PASS            261s
```

Full pytest suite green: `test_fwdllm_oracular_avail.py` (2 passed) +
`tests/launch` + `tests/mode` (188 passed, 7 skipped, 0 failed).

All 4 open questions from the static JSON→YAML comparison pass are now
resolved (see below), and the only remaining item is the longer soak run.

### Bugs found and fixed this session (all committed + pushed to
`launcher-script-fwdllm`)

1. **`run_sequential.sh` importing a stale `flame` checkout.** The active
   conda env's `pip install -e` editable pointed at a different clone
   (`/home/dgarg39/aish_test/flame`, missing `flame.launch` entirely), so the
   script picked up the wrong code regardless of which repo it ran from.
   Fixed by exporting `PYTHONPATH="$REPO_ROOT/lib/python:..."` so this
   checkout always wins. Also dropped the hardcoded `aish_smoke_flame`
   conda-env default — `ENVNAME` now comes from `FLAME_CONDA_ENV` or the
   shell's own `CONDA_DEFAULT_ENV`. (`e086cc52`)
2. **Sync-mode `FwdLLM` aggregator never broadcast EOT.** `compose()`'s
   sync branch rebuilt its tasklet chain from scratch and dropped
   `inform_end_of_training` (present only as a stale comment), unlike the
   async branch. Trainers had no way to learn the aggregator was done.
   Fixed by adding `>> c.tasklet("inform_end_of_training")` to the sync
   chain, mirroring the async branch. (`4c588843`)
3. **`wait_all()` waited on trainers sequentially.** Even with EOT now
   broadcast, `channel.await_join()` only catches peers already joined at
   broadcast time and has no timeout of its own — any trainer whose
   fetch/upload call lands a moment after the aggregator has broadcast+left
   will hang forever. This is a real, currently-unfixed race in the
   channel/trainer shutdown protocol (see "known limitation" below), so
   `wait_all()`'s 30s-per-trainer force-kill is still the actual mechanism
   that unblocks things — and it ran once per trainer in a loop, so a full
   miss across all 10 trainers cost up to `30s * 10 = 300s` of dead time
   between runs. Fixed by polling all trainer processes concurrently
   against one shared deadline, capping the worst case at ~30s regardless
   of trainer count. (`5968b0f2`)

### Verification checklist confirmed on the run above

- No tracebacks/`ERROR` in any of the 3 `.out` logs. Nonzero
  "did not exit within" counts (10 / 2 / 4) are **expected**, not a failure
  signal — see bug #3 above; the fix caps the cost, it doesn't eliminate the
  race. Each baseline's aggregator log shows a clean
  `max_data_id_progress=2 reached; stopping run.`
- Field-provenance report present in all 3 `.out` files.
- Zero `PLACEHOLDER` leaks in any generated `aggregator_config.json`.
- Per-trainer availability wiring confirmed, with one initial false alarm
  worth recording: `fwdllm_plus` trainers all log `Set avl_events_syn_0`
  even though the YAML says `availability.mode: mobiperf_2st`. This is
  correct, not a bug — `fwdllm_plus` uses ORACULAR tracking (aggregator-side,
  reads `mobiperf_2st` directly — confirmed via the aggregator log's
  `Loaded availability traces for 300 trainers (trace=mobiperf_2st)`) and
  deliberately disables the trainer's own `client_notify` self-reporting
  (`metadata/baselines.yaml` sets `client_notify.trace: syn_0` on purpose
  for this baseline). `fluxtune` (which does use `client_notify` as its
  availability signal) correctly logs `Set avl_events_mobiperf_3st_50`.
  `availability.mode` in the trainer YAML only controls which trace *data*
  gets injected into the hyperparameters dict — it does not itself set
  `client_notify.trace`; that's a separate, per-baseline knob by design.

### Open questions — resolved

1. **`rounds: 1000` in `configs/aggregator_base.json`** — keep as-is
   (user call: retain 1000, do not revert to the legacy 300).
2. **Dead/confusing fields in `configs/aggregator_base.json`** — cleaned up:
   removed the unused top-level `"dependencies"` field (Optional in the
   schema, safe to drop entirely) and the unused `hyperparameters.batchSize`/
   `learningRate` (confirmed dead: they alias into `Hyperparameters.batch_size`/
   `learning_rate` per `flame/config.py`, but no fwdllm code path reads those
   fields — only `flame/mode/horizontal/scaffold/trainer.py` and
   `flame/datasampler/fedbalancer.py` do, neither used by fwdllm). The
   top-level `"dataset"` field could *not* be removed outright — it's a
   required (non-Optional) field on the generic `ExperimentConfig` schema
   that fwdllm's `hyperparameters.dataset` shadows/duplicates but doesn't
   replace — so its value was changed from a misleading MNIST URL to an
   explicit "unused, required by generic schema" placeholder string instead.
   Verified by round-tripping the edited JSON through the real
   `flame.config.Config` parser (not just `json.load` — confirms pydantic
   validation still passes, not just that the file is syntactically valid).
3. **3 legacy async aggregator JSON variants** (`aggregator_async_base.json`,
   `aggregator_async_dynk.json`, `aggregator_async_maxiter.json`) — confirmed
   no 5th baseline needed; these are superseded exploratory variants. Added
   explicitly to `DELETION_CANDIDATES.md` (they were already inside the
   `json_scripts/` bucket there, just no longer an open question).
4. **`aggregator/fl_main.py` / `trainer/fl_main.py`** — confirmed dead via a
   repo-wide grep pass (see `DELETION_CANDIDATES.md` for the full evidence
   trail): only referenced by the already-doomed `expts/run_tc_expts/`
   scripts, use the old argparse+MPI-era `Config(args.config)` wiring
   instead of `flame.launch.cli.load_config_from_argv()`, and share origin
   history with `main_fedfwd_agg.py`/`main.py` (which superseded them).
   Added to `DELETION_CANDIDATES.md`.

### Known limitation (not fixed, tracked for later)

The `await_join()` race itself (bug #3 above) is only mitigated, not fixed —
trainers still rely on the launcher's force-kill rather than exiting cleanly
on their own. Worth a follow-up (e.g. a timeout on `await_join()` in
`fwdllm_trainer.py`'s `_fetch_weights`/`_send_grads`), but that's a change to
shared `flame` channel/trainer code used beyond this example — a separate PR,
not blocking this one.

---

## What's left before this PR can merge

1. **Overnight soak run** (in progress) — the `--max-data-id 2` smoke test
   above only proves the plumbing works end-to-end (processes spawn,
   aggregate, exit cleanly, no crashes), not that training converges.
   Launch command (30 trainers, most-heterogeneous available `agnews`
   partition, 2h wall-clock cap, data_id cap set high enough it won't be the
   thing that stops the run):
   ```bash
   cd lib/python/examples/fwdllm/expt_scripts
   ./run_sequential.sh --num-trainers 30 --max-runtime-s 7200 \
       --max-data-id 100000 \
       --partition-method "niid_label_clients=100_alpha=0.1"
   ```
   `niid_label_clients=100_alpha=0.1` is confirmed the most heterogeneous
   split available for `agnews` in the 100-client group (smaller Dirichlet
   alpha = more skewed; the H5 file's 100-client group only goes down to
   0.1 — `1000`-client group also exists with its own alpha ladder down to
   0.5, not used here since trainer count is 30). This replaces the
   smoke-test default of `partition_method: uniform` (IID), which was
   deliberately chosen for smoke tests to isolate launcher-mechanics
   validation from data-skew effects — not appropriate for a convergence
   check. `--partition-method` is a new `run_sequential.sh` flag added this
   session; it overrides `hyperparameters.partition_method` on both the
   trainer and aggregator sides (must match). `--num-gpus` doesn't need
   overriding — each YAML already defaults to all 8 available GPUs.
   Note: `selector.kwargs.c`/`k`/`minInitialTrainers` stay at each YAML's
   own default (10) even at 30 trainers unless `--c`/`--k` are also passed —
   that's fine for this soak run (a random 10-of-30 subset per round), not
   a bug.
   After it finishes: confirm loss/accuracy trends look sane over the run,
   not just that it exits 0.
2. **Final diff review**: `git diff dg-fork-main...HEAD --stat` (not
   `origin/main` — that's ~890 commits ahead/~415 behind from unrelated
   upstream sync drift on this fork and pulls in the whole repo). Confirm no
   leftover noise before opening the PR.
3. **Deletion PR** — tracked in `DELETION_CANDIDATES.md`, do as a follow-up
   once this PR merges, not bundled into it.

`expt_scripts/smoke_logs/` and the async_cifar10 experiment/parity artifacts
currently sitting untracked in `git status` are local run output, not part
of this PR — leave untracked.

---

## Session history (for full context if resuming cold)

**Workstream A — comment trim + doc consolidation.** Committed (`64099d7c`).
- Added a "Lessons from smoke-testing fwdllm" subsection to
  `MIGRATING_TO_LAUNCHER.md` §9 (CUDA-before-logging, var-check n<2 guard,
  `staleness_policy` modes, `run_sequential.sh` conventions, `syn_train_*`
  traces pointer).
- `MIGRATION_TO_LAUNCHER_FWDLLM.md` reduced to a pointer stub at §9 + `git
  log`, matching the repo's `DEPRECATED.md` convention.
- Trimmed verbose/changelog-style comments in `run_sequential.sh` and
  `fwdllm_aggregator.py`, then swept ~19 files for now-dangling
  `MIGRATION_TO_LAUNCHER_FWDLLM.md Phase N`/`decision DN` references.

**Workstream B — port missing `syn_train_*` availability traces.**
Committed (`152b7cd5`).
- `metadata/availability_traces/synthetic_traces.yaml`: 3 new trace keys
  (`syn_train_100_eval_0_unavail_0`, `syn_train_90_eval_10_unavail_0`,
  `syn_train_50_eval_30_unavail_20`), ported from the 150-trainer JSON
  source, trainers 151-300 wrap 1-150.
- `configs/trainer_base.yaml`: updated the stale "have no `metadata`
  equivalents" comment. Spawner auto-injection for these 3 keys is
  explicitly NOT wired up (future work, out of scope for this PR).

**Static JSON→YAML migration verification** (no GPU needed):
- `trainer_base.yaml` vs legacy `trainer_1.json`: field-by-field match
  confirmed, only cosmetic diffs.
- `mobiperf_traces.yaml`: confirmed `states_2st`/`states_3st_50`/
  `states_3st_75` sub-keys present as `_TRACE_KEY_TO_MOBIPERF_SUB` expects.
- `fluxtune_dynkc` baseline vs legacy `json_scripts/aggregator.json`:
  confirmed faithful transcription.

**Live GPU smoke test + bugfixes on shepherd.** See top of this file.

**This session — open questions resolved, `aggregator_base.json` cleaned up,
`--partition-method` flag added, `DELETION_CANDIDATES.md` created.** See top
of this file.

User-approved decisions (from earlier in this branch's work):
- Trainers 151-300 (no source data) get the 150 source trainers' traces
  doubled/wrapped, preserving the named distribution rather than a
  placeholder.
- Migration/verification scripts are temporary: write, run, confirm pass,
  delete.
- `MIGRATION_TO_LAUNCHER_FWDLLM.md`'s durable content folds into §9 without
  duplication; the file itself becomes a pointer stub, not a silent delete.
