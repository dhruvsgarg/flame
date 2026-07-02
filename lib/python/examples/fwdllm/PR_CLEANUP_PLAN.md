# PR-readiness plan for `launcher-script-fwdllm`

Working plan for cleanup before opening the PR. This file will be **deleted
once the PR merges** — the one thing worth keeping past that point is
[`DELETION_CANDIDATES.md`](DELETION_CANDIDATES.md), which already has its own
copy of the legacy-code deletion list for the follow-up PR.

---

## Status: smoke-tested green on shepherd; more since — see "What's left"

**3 more real bugs found and fixed** since the smoke test below, via a
post-migration baseline-behavior investigation (see
[`MIGRATION_TO_LAUNCHER_FWDLLM.md`](MIGRATION_TO_LAUNCHER_FWDLLM.md) Parts
2/3/7). Convergence soak run still not done. One scope question needs a
decision (telemetry/analysis tooling work). See "What's left before this PR
can merge" below for the current, up-to-date checklist — the smoke-test
results directly below are historical (still accurate, just no longer the
full picture).

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

### Bugs found and fixed since (post-migration baseline-behavior
investigation — full detail, evidence, and verification in
[`MIGRATION_TO_LAUNCHER_FWDLLM.md`](MIGRATION_TO_LAUNCHER_FWDLLM.md), not
duplicated here)

Running the three baselines at longer duration/higher trainer count (past
what the `--max-data-id 2` plumbing smoke test above exercises) surfaced
**three more real correctness bugs**, all committed + pushed:

4. **`fluxtune` full deadlock at scale** (Part 2, `ba622a38`): a stuck
   trainer's `SEND_TIMEOUT_WAIT_S` reclaim in `async_oort.py` was gated
   behind the very concurrency lock it was supposed to free (ordering bug)
   and, even when reached, only freed half the accounting it needed to
   (completeness bug). Reproduced at n=30/~90min, not at n=10/10min — a
   scale/time-triggered condition the original smoke test was too short to
   catch. Same two-part bug found and fixed identically in
   `async_random.py`/`fedbuff.py` (selectors used by other baselines sharing
   the pattern, not just fluxtune).
5. **Generic asyncfl aggregator duplicate-contribution gap** (Part 3,
   `e76d54f1`): found while verifying bug #4's fix didn't let a trainer
   double-contribute in one cycle — `asyncfl/top_aggregator.py` had no
   per-cycle dedup guard (fwdllm's own aggregator already did). Newly
   reachable only after fixing #4 (a stuck trainer's slot never used to
   reopen at all).
6. **fwdllm's own stale reselection-cache gap** (Part 3, `4d8d3281`): a
   departed trainer was correctly forgotten by the selector but never
   pruned from the aggregator's own per-round `_round_selected_ends` cache,
   so a round could stall forever waiting for a contribution that could
   never arrive. Only manifests under real mid-round departure (a churny
   availability trace), not `syn_0` — matches a TODO already flagged (but
   unverified) in `../MIGRATING_TO_LAUNCHER.md` §9.
7. **`max_runtime_s`/`max_data_id_progress` starvation under a real
   availability trace** (Part 7, `9c28f230`): `_aggregate_grads_async`
   (fluxtune) and `sync_collect_and_accumulate_grads` (fwdllm/fwdllm_plus)
   both called `channel.recv_fifo()` with no timeout (default = block
   forever). A real 10-min `fluxtune` smoke test under `mobiperf_3st_50`
   hung 20+ minutes and had to be manually killed — the early-stop check
   only runs from the *other* side of the composer's put/aggregate loop, so
   it never got a chance to fire while blocked. Fixed by bounding both
   calls to `RECV_TIMEOUT_WAIT_S` (30s), matching the pattern
   `asyncfl/top_aggregator.py` already uses for the same reason.
   **Re-validation on GPU in progress** — see MIGRATION_TO_LAUNCHER_FWDLLM.md
   Part 7 for live status.

All 4 of these (plus the original 3) are covered by regression tests; full
suite currently: **472 passed, 7 skipped, 0 failed**.

### Scope question: is the telemetry/analysis tooling work part of this PR?

The same investigation also did substantial work on
`scripts/analysis/analyze_run.py` and `flame/telemetry/events.py` (both
shared, example-agnostic files, not fwdllm-specific) plus fwdllm's own
telemetry emission (`fwdllm_aggregator.py`, `FedSgdTrainer.py`,
`selector/random.py`) — see MIGRATION_TO_LAUNCHER_FWDLLM.md Parts 5/6.
None of it is a correctness fix; it's what makes fwdllm's runs analyzable
at all (`plots/performance/` etc. were structurally empty before Part 5) and
brings its plot coverage roughly to parity with async_cifar10's. **Not
determined**: should this land in the same PR as the launcher port + bug
fixes above, or split into a follow-up PR? Arguments either way:
- **Same PR**: it's still fwdllm-onboarding work, discovered via the same
  investigation, and reviewers will want the plots to sanity-check the bug
  fixes anyway.
- **Separate PR**: it touches shared, non-fwdllm files
  (`analyze_run.py`/`events.py`), which is a different blast radius/review
  audience than an example-specific port, and is logically a distinct
  concern (analysis tooling vs. training-loop correctness).

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

1. **GPU re-validation of the Part 7 fix** (in progress right now —
   `run_20260701_234833_fluxtune_n10_smoke`, started 23:48 EDT,
   `--only fluxtune --max-runtime-s 600 --max-data-id 200` under the same
   `mobiperf_3st_50` trace that exposed the bug). Confirms the aggregator
   now self-terminates at/near its budget instead of hanging 20+ minutes.
   Check `MIGRATION_TO_LAUNCHER_FWDLLM.md` Part 7 for the outcome once it
   finishes — this is a quick prerequisite check before investing in item 2.
2. **Convergence soak run — still not done.** No run in
   `experiments/` uses `partition_method=niid_label_clients...` (checked;
   every run so far is the smoke-test default `uniform`/IID). The
   `--max-data-id 2` plumbing smoke test only proves processes spawn,
   aggregate, and exit cleanly — not that training converges, and it
   predates bugs #4-7 above, so it never exercised the code paths those
   fixes touch under real duration. Same command as before, now doubling as
   the long-duration validation for the deadlock (#4) and starvation (#7)
   fixes:
   ```bash
   cd lib/python/examples/fwdllm/expt_scripts
   ./run_sequential.sh --num-trainers 30 --max-runtime-s 7200 \
       --max-data-id 100000 \
       --partition-method "niid_label_clients=100_alpha=0.1"
   ```
   (`niid_label_clients=100_alpha=0.1`: most heterogeneous split available
   for `agnews` at 30 trainers — see prior reasoning below if resuming
   cold.) After it finishes: confirm (a) it self-terminates at/near 7200s
   without manual intervention (validates #7 at real duration/trainer
   count, not just the 10-min/10-trainer check in item 1), (b) no deadlock
   recurrence (validates #4), and (c) loss/accuracy trends look sane, not
   just that it exits 0 — `python3 scripts/analysis/analyze_run.py
   <run_dir>/telemetry` now produces `plots/performance/accuracy_over_rounds.pdf`
   for this (Part 5/6 telemetry work), previously impossible.
3. **Decide the Part 5/6 scope question** (same PR vs. follow-up) — see
   above. Affects what "the diff" in item 4 actually contains.
4. **Final diff review — diff base needs correcting first.**
   `git diff dg-fork-main...HEAD --stat` (the previously-planned command)
   shows **6753 files changed, 253894 insertions(+), 2688502 deletions(-)**
   — not a clean "this PR" diff. Traced why: `dg-fork-main` predates not
   just fwdllm's port but an *earlier, separate* ~20-commit initiative
   (async_cifar10's own launcher/streaming/telemetry port, e.g. `c885ae0b`
   "YAML launcher + shared metadata...", `5ffed033` "Streaming data, more
   telemetry...") that this branch was built on top of but that isn't part
   of *this* PR's actual content. The commit right before fwdllm's own
   migration starts is `35f8d654` ("update launcher script with latest
   changes of async cifar") — `15ac3fab` ("fwdllm migration to launcher
   plan") is the first commit after it. `git diff 35f8d654...HEAD --stat`
   gives a far more plausible **72 files changed, 95233 insertions(+), 1006
   deletions(-)**. Recommend using `35f8d654` (or the identical
   `15ac3fab~1`) as the diff base — but this is a call about what the
   intended PR boundary actually is (does the async_cifar10 launcher work
   already exist independently on the real merge target, or does it need
   to ride along?), not something to assume. Confirm before opening the PR.
   Also confirm no leftover noise (`expt_scripts/smoke_logs/`,
   `PR_CLEANUP_PLAN.md` itself once merged, etc.).
5. **Deletion PR** — tracked in `DELETION_CANDIDATES.md`, do as a follow-up
   once this PR merges, not bundled into it.

**Unrelated, but noticed while checking git state for item 4**: `git remote
-v` shows the `origin` remote URL has a GitHub personal access token
embedded in plaintext (`https://ghp_...@github.com/...`). That's readable by
anything that can read `.git/config` and tends to leak into tool
output/logs (as it just did here). Worth rotating that token and switching
to SSH or a credential helper instead — unrelated to this PR's mergeability,
flagging since it came up.

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

**Post-migration baseline-behavior investigation (7 parts, all committed +
pushed) — bugs #4-7 above, plus telemetry/analysis tooling work (Parts
5/6, scope TBD — see above).** Full detail lives in
`MIGRATION_TO_LAUNCHER_FWDLLM.md`, not duplicated here; see that file's own
Parts 1-7 and its top-of-file summary for the complete account.

User-approved decisions (from earlier in this branch's work):
- Trainers 151-300 (no source data) get the 150 source trainers' traces
  doubled/wrapped, preserving the named distribution rather than a
  placeholder.
- Migration/verification scripts are temporary: write, run, confirm pass,
  delete.
- ~~`MIGRATION_TO_LAUNCHER_FWDLLM.md`'s durable content folds into §9
  without duplication; the file itself becomes a pointer stub, not a silent
  delete.~~ **Superseded**: that file is now an active living doc again (the
  post-migration investigation above, Parts 1-7) — it's a real, current
  bug-hunting log, not a stub. Don't fold/delete it without re-checking
  this decision first.
