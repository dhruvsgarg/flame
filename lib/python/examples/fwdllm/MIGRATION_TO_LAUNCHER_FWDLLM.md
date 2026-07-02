# fwdllm baseline-behavior investigation (post-migration)

The launcher migration itself is complete — see
[`../MIGRATING_TO_LAUNCHER.md`](../MIGRATING_TO_LAUNCHER.md) § 9 for the
mechanics (dataset path-style handling, `client_idx` injection, the
single-aggregator-entrypoint pattern, custom stopping criteria, availability
traces, NLP dependencies, the baseline taxonomy, staleness policy,
`run_sequential.sh` conventions). For the original phase-by-phase migration
notes (bugs found, decisions made, smoke-test results), see
`git log -- lib/python/examples/fwdllm/MIGRATION_TO_LAUNCHER_FWDLLM.md`.

This doc tracks a *new* phase: two rounds of runs across the three baselines
(`fwdllm`, `fwdllm_plus`, `fluxtune`) surfaced very different progress rates.
That investigation found and fixed a real deadlock bug (and two related
correctness gaps), plus a second, fwdllm-specific gap in the per-round
reselection cache. **All four fixes are implemented, covered by regression
tests, committed, and pushed** (`ba622a38`, `e76d54f1`, `4d8d3281`,
`dc2a166b`, `644a4b86`); the full suite passes (426 passed, 7 skipped, 0
failed). A 3-baseline (`fluxtune`, `fwdllm_plus`, `fwdllm`) 1.5h GPU
experiment (`n=100`, `c=30`, `aggGoal=10`, `syn_0`) **ran to completion**
against these fixes (launched by the user outside this session) — no
deadlock recurred. A *second*, orthogonal gap was found and fixed while that
ran: fwdllm's telemetry/analysis tooling was not example-agnostic and had
real, confirmed holes — see Part 5, now complete (P5.1–P5.7), verified
against real GPU telemetry from this same experiment. **Remaining open
item**: Part 5's changes are implemented and tested but not yet committed
(see "Files touched this session"). Living doc — update as findings land.

## ✅ RESUMED (2026-07-01 ~21:25 EDT) — Part 5 complete, GPU experiment finished

Picked back up from the 18:23 EDT pause (prior block preserved in git history
of this file). Two things had changed on disk in the meantime, independent of
this session: the 1.5h/n100 GPU experiment (Part 4) had **finished all three
baselines**, and its final baseline (fwdllm) already had a real
`analyze_run.py` run against it (`plots/summary.txt` timestamped 19:56 EDT) —
apparently run by the user outside this session, since the working tree's
uncommitted code (the only place the manifest cross-check exists) produced
it. That run is strong, real evidence, not synthetic — see below.

**1. Manifest wiring — VERIFIED, for real, against real GPU telemetry.**
The `plots/summary.txt` for `run_20260701_182242_fwdllm_n100_smoke` (fwdllm's
own n=100 baseline, 100 trainers, finished 19:54:51 EDT) shows a clean
`manifest event_categories check`: all 8 declared categories read `ok`, zero
`MISMATCH`/`DRIFT` lines — `performance`/`sanity`/`selection`/`system`/
`aggregation` populated as declared, `selection/why`/`availability` partial
as declared, `insights` not_populated as declared. Event counts: 835
`trainer_round`, 828 `selection`, 72 `agg_round`, 30 `agg_eval` — P5.2's
aggregator telemetry and P5.5's `random.py` selection-emission fix are both
confirmed live and correct on real GPU data, not just synthetic/unit-test
data. Independently re-verified the loader itself directly (not just via the
plots it produced): `_find_manifest_path()`/`configure_from_manifest()`
against this same telemetry dir correctly resolves
`lib/python/examples/fwdllm/telemetry_manifest.yaml`, sets
`MODEL_PARAM_COUNT=450340`/`MODEL_MB≈1.80`/
`_PROGRESS_HIERARCHY=[data_id(150), iteration_per_data_id(15)]`; the same
check against a real `async_cifar10` telemetry dir (no manifest there)
returns `None`/leaves the async_cifar10-shaped defaults untouched. Also
re-ran `analyze_run.py` fresh against a real async_cifar10 run
(`run_20260630_010520_dbg_felix_n300_alpha0.1_syn_0_stream_sim`) as a
regression check: completes cleanly, full plot set, and — correctly — no
manifest-check section in `summary.txt` at all (since `manifest` is `None`,
`write_summary()`'s `if declared:` guard skips it, not a crash).

**2. New regression tests**: `lib/python/tests/analysis/test_manifest.py`
(9 tests) covering `_find_manifest_path()`/`load_manifest()`/
`configure_from_manifest()` — manifest found vs. absent vs. malformed YAML
(doesn't crash, logs a warning), globals set correctly from a manifest vs.
left at defaults when absent, and the `--model-params` CLI-override guard
(a manifest's `model_param_count` must never clobber it). Uses a
pytest-fixture snapshot/restore of the module-level globals so this file
can't leak state into `test_progress_key.py` or vice versa. Full suite
re-run: **451 passed, 7 skipped, 0 failed** (442 + 9 new).

**3. P5.7 (onboarding checklist) — DONE**: new `scripts/analysis/README.md`
— the 8 plot categories and what telemetry each needs, the manifest schema
(annotated, referencing fwdllm's own manifest as the worked example), a
7-step "wiring telemetry for a new example" checklist, and a "gotchas found
while onboarding fwdllm" section (the trainer-side `selection.selector`
channel-implementation artifact from P5.1; the stale-reselection-cache class
of bug from Part 3).

**4. GPU experiment (Part 4) — COMPLETE, all 3 baselines finished:**
`fluxtune` 15:19→16:49 EDT, `fwdllm_plus` 16:50→18:21 EDT, `fwdllm`
18:22→19:54 EDT (each ran its full `--max-runtime-s 5400` budget). Spot-check
findings (event counts / distinct `data_id` spread pulled directly from each
run's saved telemetry, not re-run):
   - **`fluxtune`: no deadlock.** 9515 aggregator `selection` events and 9625
     trainer-side `trainer_round` events over the full ~90min at n=100 — a
     world apart from the pre-fix n30 run (2 selection events total, 0-byte
     telemetry, full stall). Zero `waiting for someone to join channel`
     lines in the *aggregator* log (the trainer-log-side count of that
     message, ~10k across 100 trainer processes, is expected startup-poll
     chatter, not a stall symptom, given how much real work also happened).
     Strong practical evidence Part 2's fix holds at 3x the scale it was
     validated at (n30→n100) — **not** a substitute for reading the actual
     accuracy curve, but the deadlock signature from Part 2 (near-zero
     selection events, 0-byte aggregator telemetry) is definitively absent.
   - **`fwdllm_plus`: the n30 round-2 throttle (200 vs 7336 trainer_round
     events) did not reproduce at n100.** This run produced 12691
     trainer_round events and reached **all 150/150 distinct `data_id`
     values** (i.e. completed at least one full round and moved into a
     second) — compare the n30 run's presumed handful of data_ids implied
     by only 200 total events. Framing this as "not reproduced under this
     config" rather than "root-caused": the n30 run also differed in
     avail-trace/aggGoal (see Part 1's config diff table) before
     `run_sequential.sh`'s override flags existed, so this doesn't
     conclusively rule in or out any specific mechanism — just that the
     symptom is gone under the controlled (`syn_0`, `--agg-goal`-explicit)
     n100 config. Downgrading from "deferred, active investigation" to
     "not currently reproducing, no further action planned unless it
     resurfaces."
   - **`fwdllm`: healthy, in line with n30 scaling.** Reached 31 distinct
     `data_id` values (0..30) in its 1.5h/n100 budget — slower per-data_id
     progress than the n30 run's 51-in-90min (expected: same `aggGoal=10`
     serving 100 trainers instead of 30 means more selection contention per
     data_id), and this is also the run behind finding #1 above (manifest
     wiring confirmed against its real telemetry).
   - Not done: reading the actual accuracy/loss curves for a
     learning-progress verdict (as opposed to a deadlock/throttle verdict) —
     out of scope for the telemetry-tooling workstream this session focused
     on; `plots/performance/accuracy_over_rounds.pdf` exists now for fwdllm
     (P5.2) and is the artifact to open for that follow-up.

---

## TL;DR — where things stand

- **Root cause of the `fluxtune` full-stall: FOUND AND FIXED, and now
  practically re-validated on GPU** at n=100 (3x the scale it stalled at) —
  no deadlock signature (near-zero selection events, 0-byte telemetry)
  recurred; see the 2026-07-01 ~21:25 EDT update below for the numbers. See
  "Fixes applied" below for the code change itself.
- **Same bug class found and fixed in 2 sibling selectors** (`async_random.py`,
  `fedbuff.py` selector) that weren't part of the original 3-baseline
  comparison but share the same code pattern.
- **A related invariant gap in the generic asyncfl aggregator was found and
  fixed** while double-checking the fix didn't break "a trainer contributes
  at most once per round."
- **A second, distinct gap in fwdllm's own per-round reselection cache was
  found and is now FIXED** — see "fwdllm's OWN gap" in Part 3 below.
- **`fwdllm_plus`'s 200-vs-7336-round throttle did not reproduce at n=100**
  post-fix (12691 trainer_round events, all 150 data_ids reached) — treating
  as resolved-in-practice under the controlled config, not formally
  root-caused (see 2026-07-01 ~21:25 EDT update for the caveat).
- **All 5 commits from Parts 2-4 pushed; Part 5's telemetry/analysis changes
  are complete but still uncommitted** (working tree only — see "Files
  touched this session"). The 1.5h, 3-baseline (`fluxtune`, `fwdllm_plus`,
  `fwdllm`) GPU experiment (Part 4) has **finished** — all three baselines
  ran their full budget with no deadlock.
- **Telemetry/analysis tooling parity — DONE.** All of Part 5 (P5.1-P5.7) is
  now complete, including verification against real GPU telemetry (not just
  synthetic) — see Part 5 and the 2026-07-01 ~21:25 EDT update above.
- **P5.1 (telemetry cross-contamination anomaly): ROOT-CAUSED, not a leak.**
  Confirmed a pre-existing, launcher-wide (not fwdllm-specific) gap: every
  baseline's `trainer:` block in `baselines.yaml` never overrides
  `selector`, so a trainer's own channel-local selector stays on its
  example's `trainer_base.yaml` placeholder regardless of which selector
  the baseline's aggregator really uses. Confirmed functionally benign
  (trainer's channel always has exactly 1 candidate — its aggregator).
  Telemetry data is trustworthy; not fixed in `baselines.yaml` (deliberate
  scope call — see Part 5 finding #3).
- **P5.6 (smoke-test the `progress_key()` round-collapse fix): DONE.** Also
  discovered and corrected a **process error**: a prior version of this doc
  claimed the `progress_key()` fix was already "landed" in
  `scripts/analysis/analyze_run.py` — it was not; the function didn't exist
  anywhere in the working tree or git history. Implemented it for real this
  session, wired into 3 verified single-stream call sites, and confirmed
  against real saved telemetry that it fixes the round-collapse (51
  distinct x-values vs. 1) with no regression on a real async_cifar10 run.
- **P5.2 (fwdllm aggregator emits zero telemetry): DONE.** fwdllm's
  aggregator now emits `agg_eval`/`agg_round` telemetry from
  `_process_aggregation_goal_met`, unlocking `plots/performance/` for
  fwdllm for the first time (previously structurally impossible regardless
  of any analyzer fix — finding #2). Verified via 7 new unit tests
  exercising the real method, a synthetic-telemetry end-to-end run of
  `analyze_run.py` producing non-degenerate `accuracy_over_rounds.pdf`, and
  a full-suite re-run (438 passed, 7 skipped, 0 failed). Also extended
  `progress_key()` to cover `agg_round`'s own round-collapse risk once it
  started carrying real data. **This code will be picked up by the
  currently-running GPU experiment's next baseline (fwdllm)** — see Part 5
  P5.2 for why that's safe (fresh subprocess per baseline, telemetry
  emission is additive-only and try/except-guarded, no change to training
  behavior).

---

## Part 1 — Original evidence (why we started digging)

### Runs

Round 1 — quick smoke, `run_sequential.sh` defaults (`--max-runtime-s 600`,
10 trainers):
- `experiments/run_20260630_232046_fwdllm_n10_smoke`
- `experiments/run_20260630_232803_fwdllm_plus_n10_smoke`
- `experiments/run_20260630_233108_fluxtune_n10_smoke`

Round 2 — longer, up to ~90 min, 30 trainers (launched with old
`run_sequential.sh`, before the new flags below existed — no `--agg-goal`/
`--avail-trace` overrides, so each baseline kept its own checked-in
avail-trace/aggGoal/concurrency):
- `experiments/run_20260701_001330_fwdllm_n30_smoke`
- `experiments/run_20260701_024504_fwdllm_plus_n30_smoke`
- `experiments/run_20260701_051556_fluxtune_n30_smoke`

### Baseline config diff (round 2, n30, before the new override flags existed)

| | `fwdllm` | `fwdllm_plus` | `fluxtune` |
|---|---|---|---|
| selector | `random` | `random` | `async_oort` |
| tracking_mode | `default` | `oracular` | `client_notify` |
| is_async | false | false | true |
| optimizer | `fedavg` | `fedavg` | `fedbuff` |
| avail trace | `syn_0` (full avail) | `mobiperf_2st` | `mobiperf_3st_50` |
| aggGoal | 10 | 2 | 3 |
| staleness_policy | `exact` | `round_data_id` | `none` |
| aggr_num (concurrency) | 10 | 2 | 3 |
| minInitialTrainers | 30 (n/a, sync) | — | 10 |

### Progress-rate comparison

| run | trainer_round events | selection events | wall budget | verdict |
|---|---:|---:|---|---|
| fwdllm n10 smoke | 48 | 40 | 10 min | healthy |
| fwdllm_plus n10 smoke | 51 | 50 | 10 min | healthy |
| fluxtune n10 smoke | 75 | 127 | 10 min | healthy |
| fwdllm n30 (round 2) | **7336** | 7332 | ~90 min | healthy, scales as expected |
| fwdllm_plus n30 (round 2) | **200** | 200 | ~90 min | **throttled — not yet root-caused** |
| fluxtune n30 (round 2) | **0** | 0 | ~90 min (agg ran 05:16→08:00+) | **fully dead — root-caused, see Part 2** |

The `fluxtune` n30 run's `telemetry/aggregator_fluxtune_n30_smoke.jsonl` is a
literal 0-byte file — not "slow", *zero* events for the entire run. That it
worked fine at n10/10-min but produced nothing at n30/~90min was the key
clue that this was a scale/race-triggered deadlock, not "async_oort doesn't
work under the launcher" (an April pre-launcher `fwdllm_aggregator.py` run in
`expts/run_tc_expts/log/new/...adaptive_k...log` shows the same selector
aggregating fine for hours, so the algorithm itself was always sound).

---

## Part 2 — `fluxtune` root cause: CONFIRMED AND FIXED

### Evidence trail (from `run_20260701_051556_fluxtune_n30_smoke`'s aggregator/trainer logs)

1. The aggregator's `select()` only reached the real selection body (past
   `AbstractSelector.enforce_min_start`) **twice in the whole ~2.5h run**,
   both at `05:16:43`, the moment the 10th trainer joined
   (`minInitialTrainers=10`). Both hits show
   `len(ends): 10, c: 8, effective_c: 8, chosen concurrency: 8`
   (`async_oort.py` select-state logging) — i.e. it dispatched to 8 trainers
   **once**, then never called `select()`'s send path again.
2. All **30/30** trainers were permanently stuck at process start on
   `_fetch_weights: waiting for someone to join channel`
   (`fwdllm_trainer.py:187`) — including the 8 "selected" above. None ever
   received the initial global weights; zero training ever happened.
3. `async_oort.py`'s `_cleanup_removed_ends` warned 25 times with "remove
   check from all_selected failed" as trainers churned availability states —
   a downstream symptom, not the cause.
4. The aggregator log also showed `_aggregate_grads_async: no ends yet`
   repeating on a ~2.5 min tick forever — the aggregation loop was alive and
   polling, just never finding anything to aggregate.

### Root cause

`flame/selector/async_oort.py`'s `_handle_send_state` has a `SEND_TIMEOUT_WAIT_S`
(90s) mechanism whose entire purpose is: "a selected trainer that never
returns an update in 90s gets forgotten, so it becomes eligible to be
selected again." It had **two compounding bugs**:

1. **Ordering bug**: the reclaim block ran *after* the `if extra == 0: return {}`
   early-return. Once concurrency saturated (all slots occupied by stuck
   trainers), the code path that was supposed to free those slots could
   never execute — the unlock was gated behind the very lock it was meant to
   release.
2. **Incompleteness bug**: even when reached, the reclaim only did
   `del self.all_selected[end]` — it never touched
   `self.selected_ends[self.requester]`, which is what `extra`'s concurrency
   accounting (`extra = concurrency - len(selected_ends) - cooling_count`)
   actually counts. So even a reclaim that ran would leave the slot
   permanently "occupied" from `extra`'s point of view.

Together: once the very first batch of trainers was selected and (for
reasons still not fully explained at the channel/MQTT level, but downstream
of this) never completed their first fetch/train/upload cycle, `extra` got
pinned at 0 forever. No further `select()` call ever did real work again, so
none of the other 20 trainers (who join after the first 10) ever got their
first task either — hence the 30/30 "waiting for someone to join channel"
stall and the 0-byte telemetry file.

**Confirms the bug is real and was previously reachable/broken (not just
theoretical)**: it reproduced at n30/~90min scale but *not* at the earlier
n10/10min smoke (75 rounds, healthy) — consistent with a slot-exhaustion
condition that only manifests once enough concurrency saturates and enough
wall-clock passes for the 90s timeout window to matter.

### Fix applied

`flame/selector/async_oort.py` — `_handle_send_state`: moved the entire
`SEND_TIMEOUT_WAIT_S` reclaim block from after the `extra==0` short-circuit
to the very top of the method (right after `selected_ends = self.selected_ends[self.requester]`
is obtained, before the "invalid selections" cleanup and before `extra` is
computed). Added `selected_ends.discard(end)` alongside the existing
`del self.all_selected[end]`.

**The exact same two-part bug existed in `flame/selector/async_random.py`**
(used by the `fluxtune_dynkc` baseline and potentially async_cifar10
baselines) — fixed identically (moved block + added `selected_ends.discard`).

**`flame/selector/fedbuff.py` (the FedBuff *selector*, used by async_cifar10's
`fedbuff` baseline — distinct from the FedBuff *optimizer* fluxtune uses) had
only the incompleteness bug** (no early-return gates it, so no reordering was
needed) — fixed by adding `selected_ends.discard(end)` next to
`del self.all_selected[end]`.

**`flame/selector/random.py`** (the *sync* selector used by `fwdllm`/
`fwdllm_plus`) already had this correct (frees both `self.selected_ends` and
`self.all_selected` together) — used as the reference pattern for the fixes
above. **`refl_oort.py`/`feddance.py` are sync selectors with no
`SEND_TIMEOUT_WAIT_S` mechanism at all** — confirmed via grep, not affected,
no changes needed there.

### Regression tests added

`lib/python/tests/mode/test_async_sim_ordering.py` — new class
`TestSendTimeoutReclaimsConcurrencySlot` (2 tests):
- `test_stale_end_freed_from_both_dicts_and_unblocks_selection` — a stale
  (>90s) selected end must be removed from **both** `all_selected` and
  `selected_ends`, and this must happen before pacer()/the rest of
  selection runs (proven via a pacer tripwire, matching the existing
  `TestCoolingHoldsConcurrency` test's style in the same file).
- `test_fresh_end_not_reclaimed_still_short_circuits` — control case: a
  recently-selected (not-yet-stale) end must NOT be reclaimed; concurrency
  stays saturated and the short-circuit still fires.

All pre-existing tests in that file (14) + these 2 = 16 pass.

---

## Part 3 — the invariant check (triggered by explicit user concern)

User's concern, verbatim intent: *"a trainer can at most participate only
once in a round... ensure this through tests and across baselines."*
Rightly so — the fix above changes *when* a trainer becomes eligible for
reselection, so it was worth re-verifying nothing double-counts a
contribution.

### fwdllm / fwdllm_plus / fluxtune: SAFE, independently enforced

All three share one aggregator entrypoint
(`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`). Its
`_process_single_trainer_message` has a dedicated guard: `_per_agg_trainer_list`
tracks which ends already contributed to the *current*
(round, data_id, iteration_per_data_id) cycle and rejects any repeat,
**regardless of what the selector's `all_selected`/`selected_ends` think**.
This is orthogonal to the selector-level reclaim fix and was already covered
by `lib/python/tests/mode/test_fwdllm_duplicate_contribution.py` (pre-existing,
untouched). Confirmed by reading the code (not just the test) — this guard
predates this session's changes and is unaffected by them.

### Generic asyncfl aggregator (non-fwdllm baselines): GAP FOUND AND FIXED

`flame/mode/horizontal/asyncfl/top_aggregator.py` (used by async_cifar10's
`fedbuff`/`async_random`-selector baselines) had **no equivalent per-cycle
dedup guard** — only an optional `reject_stale_updates` model-version check
(default off, and even when on, only catches a duplicate after the version
has already advanced, not within the same still-open window). Before this
session's selector fix, this was moot: a stuck trainer's slot never actually
reopened (same broken reclaim as `async_oort.py`), so double-selection while
an original response was still in flight was **not reachable in practice**.
After fixing the reclaim, it becomes reachable — a genuine, newly-exposed gap.

**Fix applied** — new `_agg_cycle_contributed_ends: set` guard in
`top_aggregator.py`, mirroring fwdllm's pattern:
- Initialized in `internal_init()` (near `self._agg_goal_cnt = 0`).
- Reset in `_reset_agg_goal_variables()` (this runs once per outer round via
  the `Loop`/`Tasklet` composer wiring in `compose()` — `task_reset_agg_goal_vars`
  fires before the inner `asyncfl_loop` that repeats
  `task_put_train >> task_put_eval >> task_get_weights` until
  `_agg_goal_cnt == self._agg_goal` — confirmed this is the correct
  per-aggregation-cycle reset point, analogous to fwdllm's cycle boundary).
- Checked at the top of the `WEIGHTS`/`WEIGHTS_BYTES` message branch in
  `_aggregate_weights`: if `end` already in the set, log + call
  `channel.cleanup_provided_ends(end)` + return early (no counting).
  Note in passing: there's a pre-existing "check2" comment right below this
  in the same function (`recv_wts_version` already-seen check) that only
  **logs an error** but doesn't actually reject — a second half-implemented
  guard, not touched this session, worth knowing about if this area gets
  revisited.
- Marked contributed at `self.cache[end] = tres` (the point a contribution
  is actually accepted into the aggregation).

**Regression tests added** — new file
`lib/python/tests/mode/test_asyncfl_duplicate_contribution.py` (3 tests, all
passing): duplicate from the same end in the same cycle is ignored and not
double-counted; different ends both count; `_reset_agg_goal_variables` clears
the guard for the next cycle. Uses a minimal real `torch.nn.Linear(1,1)`
model + tensor weights (this aggregator's `_aggregate_weights` needs real
`weights_to_model_device` calls, unlike fwdllm's gradient-based path) with a
fake channel/selector stub, agg_goal set high enough that finalization
(`optimizer.scale_add_agg_weights`/`_update_model`) is never reached — keeps
the test from needing to stub that heavier path.

### fwdllm's OWN gap: FOUND AND FIXED

Full mechanics, from tracing the code (not guessed):

**Confirmed hierarchy**: `round` > `data_id` (databin) > `iteration_per_data_id`.
A round has `total_data_bins` data_ids; each data_id can take multiple
iterations (variance-check retries) before advancing; the round only
advances once all its data_ids are done
(`fwdllm_aggregator.py:_process_aggregation_goal_met`).

**Reselection granularity** (`_select_ends_respecting_reselect_gate`,
`fwdllm_aggregator.py` ~line 1516):
- `fwdllm` (`reselect_each_iteration=False`): accumulates picks into a
  per-round cache `self._round_selected_ends` until it hits `agg_goal`, then
  **reuses that exact cached list for the rest of the round** — only reset
  when `self._round` itself advances. Reselection granularity = the round.
- `fwdllm_plus` (`reselect_each_iteration=True`): calls the selector fresh
  every single call, i.e. every iteration. Finest possible granularity.

**Is the 90s timeout enabled for fwdllm/fwdllm_plus?** Yes — both use
`selector/random.py`, whose `SEND_TIMEOUT_WAIT_S` reclaim was *already*
correct (reference implementation used to fix the async selectors above).
**But**: that check only runs inside `select()`'s SEND branch, and fwdllm's
cache-reuse path (`reselect_each_iteration=False`) **bypasses calling
`select()` at all** once the round's batch is assembled. So for fwdllm,
mid-round, a silently-stuck-but-still-"available" trainer's 90s eviction
never fires — only re-armed when the round advances. For fwdllm_plus, since
`select()` runs fresh every iteration, the 90s check is continuously live.

**Explicit departure** (trainer reports `UN_AVL`, or genuinely disconnects)
is handled by a *separate*, always-active, push-triggered path for both:
`channel.py`'s state-transition/`remove()` handling calls
`_selector._cleanup_removed_ends(end_id)` directly, not gated by reselect
granularity. `random.py`'s implementation of this correctly frees both
`selected_ends` and `all_selected`.

**The gap**: even though `_cleanup_removed_ends` correctly forgets a
departed trainer at the *selector* level, **nothing prunes it from the
*aggregator's own* `_round_selected_ends` cache** (`fwdllm_aggregator.py`).
Checked exhaustively via grep — `_round_selected_ends` is only ever
fully reset (round-advance) or appended to (new candidates found); there is
no code path that removes one specific departed end_id from it. So
`_select_ends_respecting_reselect_gate`'s cache-size check
(`len(self._round_selected_ends) >= target`) still reports "full" even
though one cached member has actually left — it keeps returning the stale
list (including the departed end_id) for the rest of the round, and never
re-invokes the selector to backfill the freed slot. **The round would stall
waiting for a contribution that can never arrive.**

This exactly matches an existing flagged-but-unverified TODO already in
`../MIGRATING_TO_LAUNCHER.md` § 9 ("confirm this doesn't starve progress
under the oracular/mobiperf availability traces at full (300+) trainer
scale"). It hasn't been hit in the runs above because all smoke tests so far
used `syn_0` (always-available) — this gap only manifests under real
mid-round departure, which needs a churny availability trace to exercise.

**Fix applied** — new `_prune_departed_from_round_cache` helper in
`fwdllm_aggregator.py`, called from
`_select_ends_respecting_reselect_gate` (only on the
`reselect_each_iteration=False` path, right after the round-rollover reset
and before the cache-size check). For each end currently in
`self._round_selected_ends`, it drops the end if either:
- `channel.has(end)` is `False` (fully disconnected — the `channel.remove()`
  case), or
- `channel.get_end_property(end, PROP_END_AVL_STATE) == TrainerAvailState.UN_AVL`
  (still connected but explicitly reported unavailable — the
  `channel.update_state()` case).

Pruning shrinks `self._round_selected_ends` below `self._agg_goal`, so the
existing cache-size check (`len(self._round_selected_ends) >= target`) now
correctly reports "not full" and falls through to
`channel.ends(VAL_CH_STATE_SEND, task_to_perform)`, which the existing
accumulate-path merge logic backfills into the cache. No changes needed to
the accumulate/backfill logic itself — it already treated a short cache as
"needs more," it just never used to *get* short.

**Regression tests added** — `lib/python/tests/mode/test_fwdllm_reselection.py`,
new class `TestStaleCachePruning` (3 tests, all passing):
`test_disconnected_end_pruned_and_backfilled` (channel.has()==False case),
`test_un_avl_end_pruned_and_backfilled` (UN_AVL-property case), and
`test_still_present_end_not_pruned` (control: no departure → no pruning, no
extra selector re-query). Extended the file's existing `_FakeChannel` with
`has()`/`get_end_property()`/`_removed`/`_unavail` to model both departure
modes, and bound the new `_prune_departed_from_round_cache` onto
`_FakeAggregator` the same way the file already binds `select`/
`_rearm_recv_eligibility`.

**vs. async (fluxtune) for comparison**: no round/batch concept at all —
concurrency `c` is a rolling window; the instant any slot frees (successful
contribution, explicit departure, or the 90s timeout fixed in Part 2), the
very next `select()` call can immediately backfill it, since there's no
cache-reuse gate suppressing the call. Far more elastic under churn, at the
cost of the staleness/version bookkeeping (`staleness_policy` +
`_per_agg_trainer_list`) needed to know which async response belongs to
which cycle.

---

## Part 4 — tooling changes (controlled cross-baseline comparison)

`lib/python/examples/fwdllm/expt_scripts/run_sequential.sh` gained new flags
so a single invocation can run all 3 baselines with a genuinely controlled
comparison (previously `--c` conflated concurrency and agg_goal, and there
was no way to override the availability trace):

- `--agg-goal N` — sets `aggregator.agg_goal` directly (fans into
  `hyperparameters.aggGoal` + `selector.kwargs.aggGoal`/`aggr_num` per
  `runner.py`), independent of `--c`. Legacy behavior (agg_goal == c) is
  preserved when `--agg-goal` is omitted.
- `--c-async N` — overrides `selector.kwargs.c` **only** for the async
  baseline (`fluxtune`, detected by `run_key == "fluxtune"` — the only async
  entry in `ALL_RUNS`), letting sync baselines use `--c` (concurrency ==
  agg_goal) while fluxtune overcommits concurrency independent of agg_goal.
- `--min-initial-trainers N` — overrides `selector.kwargs.minInitialTrainers`
  directly, independent of `--num-trainers`/`--c`.
- `--avail-trace NAME` — overrides the availability trace for **all three**
  baselines: `trainer.availability.mode` (cosmetic/consistency),
  `trainer.config_overrides.hyperparameters.client_notify.trace`
  (fluxtune's *real* signal), and
  `aggregator.config_overrides.hyperparameters.trackTrainerAvail.trace`
  (fwdllm_plus's *real* ORACULAR signal). Use `syn_0` to isolate
  selection/aggregation bugs from trace-driven scarcity/churn.

Verified via a dry-run simulation of `patch_yaml`'s python heredoc against
the actual checked-in YAMLs (not an actual GPU run) — confirmed output
matches intent exactly:
`--num-trainers 30 --num-gpus 8 --c 10 --c-async 30 --agg-goal 10 --min-initial-trainers 30 --avail-trace syn_0 --k 5`
produces: fwdllm c=10/agg_goal=10/minInitialTrainers=30/trace=syn_0;
fwdllm_plus c=10/agg_goal=10/minInitialTrainers=30/trace=syn_0 (overriding
its checked-in `mobiperf_2st`); fluxtune c=30/agg_goal=10/minInitialTrainers=30/
trace=syn_0 (overriding its checked-in `mobiperf_3st_50`).

### Follow-up: the actual 1.5h/n100 invocation, and a real bug caught in verification

User asked for `aggGoal=10, c=30, n=100`, per-baseline duration 1.5h, order
`fluxtune, fwdllm_plus, fwdllm`, controlled (`syn_0`) config. Dry-run
verification against the checked-in YAMLs caught a real footgun before any
GPU time was spent:

**`--max-data-id`'s script default (10) is not a per-round safety valve —
it's a hard, run-ending cap that fires almost immediately.** `total_data_bins
= 150` is hardcoded in `fwdllm_aggregator.py`'s `internal_init` (all three
baselines share this aggregator class), and `data_id` counts 0..149 *within*
a single round before resetting — it does NOT track completed rounds.
`_check_early_stop_conditions` sets `self._work_done = True` **permanently**
(not just "end this round") the moment `data_id >= max_data_id_progress`.
Since `run_sequential.sh`'s `patch_yaml` writes `max_data_id_progress`
unconditionally (default 10) regardless of whether `--max-data-id` is passed,
every invocation that doesn't explicitly override it would die a few minutes
into round 0 — nowhere near the intended 1.5h. Fix: pass `--max-data-id 200`
(anything > 150) so `--max-runtime-s` is the only real stop condition.

Also addressed: `--min-initial-trainers` was left unset, which made the
`--c`-derived default equal `--num-trainers` (100) — i.e. the aggregator
would refuse to do any real work until *all* 100 trainers joined, burning
budget if joining is staggered, and with zero tolerance for a trainer that
fails to join at all. Per user request, set explicitly to 5% below `n`:
`--min-initial-trainers 95`.

**Verified final invocation** (dry-run confirmed against all 3 YAMLs —
`num_trainers=100`, `c=30`, `agg_goal=10`, `max_runtime_s=5400`,
`max_data_id_progress=200`, `minInitialTrainers=95`, `syn_0` wired into all
three real availability-signal fields for every baseline):

```bash
lib/python/examples/fwdllm/expt_scripts/run_sequential.sh \
  --only fluxtune,fwdllm_plus,fwdllm \
  --num-trainers 100 \
  --c 30 \
  --agg-goal 10 \
  --avail-trace syn_0 \
  --max-runtime-s 5400 \
  --max-data-id 200 \
  --min-initial-trainers 95
```

**Status: COMPLETE.** Launched by the user outside this session, after the
above verification; all three baselines ran their full `--max-runtime-s
5400` budget back-to-back (`fluxtune` 15:19→16:49 EDT, `fwdllm_plus`
16:50→18:21 EDT, `fwdllm` 18:22→19:54 EDT) with no deadlock or early
termination. See the 2026-07-01 ~21:25 EDT update near the top of this doc
for per-baseline findings. `--num-gpus` was left at each YAML's checked-in
default (8) — not overridden, since that's a physical-resource call, not a
config-correctness one.

---

## Part 5 — telemetry & analysis parity, made example-agnostic (NEW, in progress)

### Why this is its own workstream, not a fwdllm bugfix

The original ask ("fluxtune's `plots/` dir is empty, go fix it") looked like
a one-off tooling gap. It isn't. `scripts/analysis/analyze_run.py` (2348
lines) was written *for* async_cifar10 and generalized only informally: it
hardcodes async_cifar10's model size, and assumes "round" is both (a) the
only progress unit and (b) fine-grained (advances every aggregation). Neither
holds for fwdllm. Since more examples will onboard onto `flame.launch` after
this one, the fix needs to separate "core, example-agnostic analysis" from
"example-specific analysis," with a declared contract new examples must
satisfy — not another one-off patch. This section is the plan; subtask
checkboxes track state as work lands.

### Evidence gathered so far (real telemetry, not guessed)

Pulled from `experiments/run_20260701_001330_fwdllm_n30_smoke` (round-2,
pre-fix, but telemetry-instrumentation-wise representative of today's code
too — nothing in Parts 2–4's fixes touched telemetry):

1. **Round-granularity collapse, confirmed severe.** One trainer's
   `trainer_round` events (734 total over the ~90min run) have `round == 1`
   for *every single event* — it never advances, because a round only
   completes once all 150 data bins finish, which this run never reached.
   Meanwhile `data_id` spans 51 distinct values in the same window. Any
   `analyze_run.py` plot that buckets by `round` (~70 call sites reference
   `round` as a grouping/x-axis key) collapses fwdllm-family runs onto a
   single point. `data_id`/`iteration_per_data_id` are already present on
   the event — this is a plotting-code gap, not a missing-telemetry-field
   gap.
2. **fwdllm's aggregator emits zero telemetry, period.** Grepped every
   `fwdllm_aggregator.py` method — no `telemetry.emit()` call anywhere.
   Confirmed against real output: `aggregator_fwdllm_n30_smoke.jsonl` is 0
   lines. Only trainer-side `trainer_round`/`selection` events exist. This
   is why only 4 of 8 plot categories (`selection/`, `sanity/`, `system/`,
   `availability/`) had any files in the real run's `plots/` dir —
   `performance/` (accuracy/loss, needs `agg_eval`) and `insights/` (needs
   `agg_round`) can't exist without aggregator-side instrumentation, no
   matter what the analyzer does. Contrast: `asyncfl/top_aggregator.py`
   (async_cifar10's base class) already calls `telemetry.emit()` for
   `agg_round`/`utility_belief` inside `_aggregate_weights` — fwdllm's
   `_aggregate_grads_sync`/`_aggregate_grads_async` are separate methods
   that never call anything equivalent.
3. **Anomaly from evidence gathering: ROOT-CAUSED (P5.1), refutes the leak
   hypothesis.** Every trainer's `selection` event in
   `run_20260701_001330_fwdllm_n30_smoke` names `"selector":
   "FedBuffSelector"` (733/733 selection events, in every one of the 10
   trainer telemetry files sampled) with identical `chosen`/
   `eligible_fingerprint`/`decision_fingerprint` values across all of them.
   Confirmed this is **not** `FLAME_TELEMETRY_DIR` leaking from an unrelated
   pytest run (`test_selector_contract.py`/`test_selection_determinism.py`
   don't call `telemetry.emit()`/`configure()` at all — grepped, no hits —
   and the env var isn't exported in any shell profile). It is real,
   legitimately-emitted telemetry from a genuine, reproducible config gap:

   - `flame/channel_manager.py:join()` (~line 157) has **every role**
     (aggregator *and* trainer) construct its **own** local `Selector`
     instance for a channel from `self._config.selector.sort`/`.kwargs` —
     each side does its own local "who do I send to / fetch from"
     resolution via `Channel.select()` (`flame/channel.py` ~line 221/233).
   - The aggregator's `selector.sort` is correctly deep-merged from the
     baseline (`random` for fwdllm, confirmed in
     `aggregator_config.json:103`).
   - The **trainer's** `selector.sort`/`kwargs` are never touched by any
     baseline. Checked all of `examples/_metadata/baselines.yaml`'s
     `trainer:` blocks (fwdllm, fwdllm_plus, fluxtune, and — for
     comparison — felix/oracle/refl/feddance/oort/fedbuff/fedavg): **none**
     override `selector` under `trainer:`, only `hyperparameters`. So every
     trainer's channel-local selector stays on whatever
     `<example>/configs/trainer_base.yaml` hardcodes as a placeholder —
     for fwdllm that's `sort: fedbuff, kwargs: {c: 5, aggGoal: 1}` (see
     `trainer_base.yaml:127-131`, explicitly commented "placeholders --
     real values come from the baseline ... deep-merged in by the launcher
     before these defaults" — the comment is aspirational; the deep-merge
     for `selector` specifically never happens for `trainer:`).
   - **Confirmed pre-existing and NOT fwdllm-specific**: async_cifar10's own
     `configs/trainer_base.yaml` has the identical placeholder pattern
     (`sort: fedbuff, kwargs: {c: 20, aggGoal: 1}`), and none of its
     baselines (`felix`, `oracle`, etc., which use `async_oort` at the
     aggregator) override `trainer.selector` either. This is a launcher-wide
     gap present since before fwdllm onboarded, just never noticed because
     nobody had looked at a trainer-side `selection` event's `selector`
     field before this investigation.
   - **Confirmed functionally benign** (traced, not guessed):
     `flame/selector/fedbuff.py:select()` does
     `concurrency = min(len(ends), self.c)`. A trainer's channel to its
     aggregator always has exactly 1 "other" end (`len(ends) == 1`), so
     `concurrency` resolves to 1 regardless of the placeholder `c`/`aggGoal`
     values — the trainer always trivially selects its one aggregator peer.
     This matches the observed telemetry exactly: `num_candidates: 1`,
     `num_eligible: 1`, `chosen` = the aggregator's fixed taskid, unchanged
     across all 733 events (also independently confirmed the aggregator's
     taskid `49d06b7526964db86cf37c70e8e0cdb6bd7aa742` is a real, checked-in
     constant in `configs/aggregator_base.json` — not a leaked test id, just
     a coincidence that grep first surfaced it in an unrelated example's
     `run.py`).

   **Practical implication**: telemetry data is trustworthy — this is
   deterministic misconfiguration, not cross-process/cross-run
   contamination, so nothing here blocks building analysis on existing
   telemetry. The one caveat for P5.5/analysis work: a trainer-role
   `selection` event's `"selector"` field is a channel-implementation
   artifact (always the example's `trainer_base.yaml` placeholder,
   currently `"FedBuffSelector"` for every fwdllm-family baseline) and does
   **not** reflect the real FL-level selection algorithm — that's only
   truthfully reported on the **aggregator**-role selection/config side.
   Any cross-baseline comparison that groups by "selector" must read it
   from the aggregator's config/telemetry, not from trainer telemetry.
   **Not fixed** (deliberately, scope call): correcting `trainer.selector`
   in `baselines.yaml` project-wide touches every example/baseline, not
   just fwdllm's three, and buys no behavioral difference (per the benign
   analysis above) — flagged as an optional low-priority cleanup, not
   pursued in this workstream.

### Design direction (proposed, not yet built)

- **Progress-axis abstraction**: don't hardcode `data_id`/
  `iteration_per_data_id` field names into `analyze_run.py`. Instead, have
  each example declare its progress hierarchy (ordered fields + bounds,
  e.g. fwdllm: `[round, data_id(<150), iteration_per_data_id(<15)]`;
  async_cifar10: `[round]`, i.e. today's behavior, unchanged) so the
  analyzer can fold sub-round fields into a composite ordinal generically.
  Async_cifar10 needs zero changes under this design (empty/default
  hierarchy = current behavior).
- **Per-example analysis manifest**: a small declared config (new file per
  example, e.g. `lib/python/examples/<example>/telemetry_manifest.yaml` or
  similar) carrying: model param count (replaces the hardcoded
  `MODEL_PARAM_COUNT = 537610` async_cifar10 constant), the progress
  hierarchy above, and which event categories the example actually
  populates (so missing `plots/performance/` is a documented, expected gap
  rather than a silent one when an aggregator genuinely doesn't emit
  `agg_eval`).
- **Core vs. example-specific plots**: classify each of the 8 existing plot
  categories (`performance`, `sanity`, `selection`, `insights`, `system`,
  `availability`, `selection/why`, `aggregation`) as core-as-is,
  core-needs-progress-key-fix, core-needs-model-size-fix, or
  example-specific (new plots unique to an example's mechanics, e.g.
  fwdllm's variance-check retry rate, which no other example has).
- **Onboarding contract**: once the above exists, write it up as a
  checklist future examples must satisfy (wire `telemetry.configure` at
  both aggregator/trainer startup, emit at least one `agg_eval`/`agg_round`
  per cycle, declare a manifest if progress isn't a plain round, run
  `analyze_run.py` on a smoke run and confirm no category is silently
  empty). Likely lands as a new `scripts/analysis/README.md` section or a
  new `MIGRATING_TO_LAUNCHER.md` subsection, referenced from both.

### Interim fix landed: `progress_key()` in `analyze_run.py` (NOT P5.3 — a
narrower, hardcoded stopgap for finding #1 only)

**Correction to this doc's own prior entry**: an earlier version of this
section described `progress_key()` as already landed, with a specific list
of wired call sites. That was inaccurate — `grep progress_key
scripts/analysis/analyze_run.py` (and `git log`/`git status`) turned up
nothing; the function did not exist in the working tree or in history. It
had been described but never actually written. Implemented for real this
session; see below for what's genuinely there now (a narrower set of call
sites than the earlier, inaccurate description claimed).

`progress_key(r)` (right after `by_event`, `scripts/analysis/analyze_run.py`)
returns plain `round` when a record has no `data_id` field (async_cifar10 —
unaffected, identical output to before), or `round * 200 + data_id` when it
does (fwdllm's `trainer_round` records only — `200 > total_data_bins=150`
keeps ordering correct). Wired into exactly three call sites, all of which
bucket **only** `EVENT_TRAINER_ROUND` records (verified single-stream, no
cross-event join):
- `trainer_rounds_by_round` (feeds `overrun_rate_over_rounds.pdf` and the
  data-unlock curve in `perf_plots`)
- `sim_time_by_round`
- the round-time-split loop in `system_plots` (feeds
  `trainer_time_split_over_rounds.pdf`/`_cdf.pdf`/`_overall.pdf` — this is in
  `system_plots`, not `sanity_plots` as the earlier, inaccurate entry said)

**Deliberately NOT touched, and confirmed correctly excluded**:
`_participation_heatmap` and `_state_fraction_plots` — despite the earlier,
inaccurate entry claiming these were wired, they cross-match
`EVENT_TRAINER_ROUND` (`trained`) against `EVENT_SELECTION` (`evalsel`) on
exact `round` equality to build a single combined per-round set; folding
`data_id` into only the trainer side would have silently broken that join
(SELECTION records don't carry `data_id` — see finding #3's caveat). Also
correctly left alone: `sanity_plots`' aggregator-observed-vs-trainer-reported
join (`agg_obs`/`tr_rep`), `selection_plots`' `actual_util` lookup keyed by
`(round, end_id)`, and every `EVENT_SELECTION`/`EVENT_AGG_ROUND`/
`EVENT_AGG_EVAL`/`EVENT_UTIL_DISPARITY`/`EVENT_AVAIL_CHANGE` site (all
currently empty or data_id-less for fwdllm regardless — see finding #2 — so
`progress_key` would be a no-op there anyway).

**P5.6 done, for real this time**: ran
`python3 scripts/analysis/analyze_run.py <telemetry_dir> --out <scratch>`
against the saved `run_20260701_001330_fwdllm_n30_smoke` telemetry (the
`python3`-invocation tool issue from the prior session is gone — `python3`
works fine now). It completes cleanly, writes 30 artifacts including
`overrun_rate_over_rounds.pdf` and `trainer_time_split_over_rounds*.pdf`.
Verified the actual data feeding those plots, not just that the file wrote
without error: plain `round` collapses all 733 `trainer_round` events for
one trainer onto a single x-value (`{1}`), while `progress_key()` spreads
the same events across 51 distinct values (`200..250` — `1*200 + data_id`
for `data_id` 0..50), confirming the collapse is fixed. Also ran the
analyzer against a real saved async_cifar10 run
(`run_20260630_095400_dbg_refl_n300_alpha0.1_syn_0_stream_sim`) as a
regression check — completes cleanly, full plot set produced, confirming
`data_id`-less records still degrade to plain `round` with no behavior
change.

**Regression tests added**: `lib/python/tests/analysis/test_progress_key.py`
(new file, 5 tests) — async_cifar10 no-op case, fwdllm's round-major/
data_id-minor ordering, the 200-multiplier's no-collision guarantee against
`total_data_bins=150`, and the concrete round-collapse-is-fixed regression
(733 synthetic events all at `round==1` → 1 plain-round bucket vs. 51
`progress_key` buckets, mirroring the real run's numbers). Full suite
re-run with these added: **431 passed, 7 skipped, 0 failed** (426 + 5 new).

This is intentionally narrower than P5.3/P5.4's declared design (no
per-example manifest, `data_id`/`total_data_bins=150` are hardcoded into the
analyzer rather than declared by the example) — it unblocks the specific,
demonstrated collapse in finding #1 now; P5.3/P5.4 remain the right
generalized fix for when more `flame.launch` examples onboard.

### Subtasks (status tracked here as work lands)

- [x] **P5.1 — Root-cause the telemetry cross-contamination anomaly.**
  **DONE, leak hypothesis refuted.** Not `FLAME_TELEMETRY_DIR` leaking
  across processes — confirmed real, reproducible telemetry from a
  pre-existing, launcher-wide (not fwdllm-specific) gap: trainer-side
  channel selectors are never overridden by any baseline's `trainer:`
  block in `baselines.yaml`, so they stay on each example's
  `trainer_base.yaml` placeholder (`fedbuff` for both fwdllm and
  async_cifar10). Confirmed functionally benign (trainer's channel always
  has exactly 1 candidate — its aggregator — so selection is trivial
  regardless of selector class/kwargs). See Part 5 finding #3 for the full
  trace. **Telemetry data is trustworthy; nothing blocks building analysis
  on it.** Caveat carried forward to P5.5: trainer-role `selection.selector`
  is a channel-implementation artifact, not the real FL selector — read
  that from the aggregator side instead. Not fixed in `baselines.yaml`
  (deliberate scope call, no behavioral upside).
- [x] **P5.2 — Instrument `fwdllm_aggregator.py` for `agg_eval`/`agg_round`
  telemetry** — **DONE.** Landed in `_process_aggregation_goal_met` (the
  single per-cycle aggregation entrypoint shared by all 3 baselines), not
  `_aggregate_grads_sync`/`_aggregate_grads_async` as originally guessed --
  those methods only accumulate a gradient into `self.grad`, they don't
  complete a cycle; `_process_aggregation_goal_met` is where the cycle
  actually finishes (variance check, eval, data_id/round advance), matching
  where the existing per-cycle bookkeeping (`_per_agg_trainer_list` reset,
  `_log_and_reset_model_version_stats`) already lives.
  - `agg_eval`: emitted right after the existing `self.eval_model()` call,
    only on the variance-check-passed branch (the only branch that
    evaluates) — fields `test-loss`/`test-accuracy`/`mcc` (matching
    `syncfl/top_aggregator.py`'s `_eval_emit` field-name convention exactly,
    so `analyze_run.py`'s existing `accuracy_by_round`/`loss_by_round`
    extractors work with zero analyzer changes) plus `data_id`/
    `iteration_per_data_id` so the eval can be placed on `progress_key()`'s
    axis. Unlocks `plots/performance/` for fwdllm (structurally impossible
    before — finding #2).
  - `agg_round`: emitted once per completed aggregation cycle (both the
    variance-passed and variance-failed paths — a cycle completes either
    way), using per-cycle contributor/staleness/utility/speed lists
    snapshotted at the top of the method (`_cycle_contributors`,
    `_cycle_staleness`, etc.) *before* `self._per_agg_trainer_list` is reset
    and *before* `self._model_version` potentially advances later in the
    same call — staleness is computed against the pre-cycle model version
    the contributors actually trained against, not whatever it becomes
    after this cycle's advance. Unlocks `plots/insights/`/`plots/system/`
    comm- and staleness-derived plots for fwdllm.
  - Both wrapped in `if telemetry.is_enabled(): try: ... except Exception:
    logger.debug(...)`, matching the existing telemetry-must-never-break-
    training convention used everywhere else in the codebase.
  - **Follow-on analyzer fix required and landed**: turning on `agg_round`
    exposed the *same* round-collapse problem P5.3's `progress_key()` fixed
    for `trainer_round` — `queue_depth_over_rounds.pdf`, `staleness_cdf.pdf`'s
    over-rounds companion, and `commit_cadence_over_rounds.pdf` all bucket
    `EVENT_AGG_ROUND` by plain `round`. Confirmed each is single-stream (no
    cross-event join) and wired `progress_key()` into all three. Also fixed
    `time_to_target()`'s `sim_map` lookup (was comparing a progress_key-keyed
    dict against a plain-round key — dead code for fwdllm before P5.2 since
    `agg_eval` was always empty, live and silently-broken after).
  - **Deliberately NOT changed**: `accuracy_by_round`/`loss_by_round` keep
    plain-`round` keys, even though `agg_eval` records now carry `data_id`.
    `comm_vs_accuracy_series` cross-joins `accuracy_by_round`'s output
    against `cumulative_comm_by_round` (SELECTION-derived, no `data_id`) via
    a `round <=` comparison — folding `data_id` into only the accuracy side
    would silently return the run's total comm for every accuracy point
    instead of the correctly-scoped partial cumulative comm. Net effect:
    `accuracy_over_rounds.pdf`/`accuracy_over_simtime.pdf` will still show
    fwdllm's eval curve collapsed onto few round-buckets even though the
    underlying telemetry is fully fine-grained (`data_id` present on every
    record) — a known, deliberate plotting-only gap, left for P5.3's
    generalized manifest-driven version rather than special-cased here.
  - **Verified two ways**: (1) unit tests exercising the *real*
    `_process_aggregation_goal_met` method (not mocked out) against a
    minimal stand-in with a real `torch.nn.Linear(1,1)` model — see below;
    (2) generated synthetic fwdllm-shaped telemetry (`agg_round`/`agg_eval`
    records with `data_id`, matching what the instrumented aggregator will
    actually emit) and ran `analyze_run.py` against it end-to-end: produced
    `plots/performance/accuracy_over_rounds.pdf`/`loss_over_rounds.pdf` for
    the first time ever for an fwdllm-shaped run, plus confirmed
    `agg_round`'s `progress_key()` spread across 20 distinct buckets (vs. 1
    under plain `round`) while `accuracy_by_round` stayed at 1 bucket as
    intended (the documented, deliberate tradeoff above).
  - **Regression tests added**: `lib/python/tests/mode/test_fwdllm_agg_telemetry.py`
    (new file, 7 tests, all passing) — `agg_eval` emitted only on the
    variance-pass path with correct pre-increment `data_id`; `agg_round`
    emitted on both outcomes; staleness computed against the pre-cycle
    (not post-advance) model version; contributor list captured before the
    method's own reset; both events are true no-ops (no emit call, no file
    written) when telemetry is disabled. Full suite re-run: **438 passed, 7
    skipped, 0 failed** (431 + 7 new).
  - **This code is live for the currently-running GPU experiment's next
    baseline.** `run_sequential.sh --only fluxtune,fwdllm_plus,fwdllm ...`
    (Part 4) spawns each baseline as a fresh `python3` subprocess only when
    that baseline's turn starts — fwdllm_plus (in progress as of this
    writing, ~18:05, budget ends ~18:21) doesn't re-import already-loaded
    modules, so it's unaffected; the next baseline, fwdllm, spawns fresh and
    *will* pick up this code from disk. Telemetry emission is additive/
    observational only (never touches model weights or aggregation math, and
    every emit call is try/except-guarded), so this does not change fwdllm's
    training behavior, timing, or results relative to the already-completed
    fluxtune/fwdllm_plus runs in this same batch — it only adds telemetry
    files those completed runs don't have.
- [~] **P5.3 — Design + land the progress-axis abstraction** in
  `analyze_run.py` (or wherever it ends up living) per "Design direction"
  above. Must be a no-op for async_cifar10 (regression risk: don't break
  its existing, working plots). **Partial, now actually implemented (see
  correction above — a prior draft of this doc claimed this was already
  done when the code didn't exist)**: a hardcoded `progress_key()` stopgap
  is landed and smoke-tested (see "Interim fix landed" above), wired into
  exactly 3 call sites (`trainer_rounds_by_round`, `sim_time_by_round`,
  `system_plots`' round-time split) — narrower than the prior draft
  claimed (`_participation_heatmap`/`_state_fraction_plots` correctly
  excluded, not wired, since they cross-join against `EVENT_SELECTION`).
  The generalized, manifest-driven version (per-example declared
  hierarchy, not a hardcoded `data_id`/150 constant) is still open.
- [x] **P5.4 — Design + land the per-example manifest** (model param count,
  progress hierarchy, expected event categories). **DONE.**
  `lib/python/examples/fwdllm/telemetry_manifest.yaml` declares all three;
  `analyze_run.py` gained `_find_manifest_path()`/`load_manifest()`/
  `configure_from_manifest()`; async_cifar10 stays on defaults (no manifest
  file for it, confirmed via `_find_manifest_path()` returning `None` for a
  real async_cifar10 telemetry dir). Verified against real GPU telemetry
  (see 2026-07-01 ~21:25 EDT update) and 9 new unit tests in
  `lib/python/tests/analysis/test_manifest.py`.
- [x] **P5.5 — Audit all 8 plot categories** against fwdllm's actual
  telemetry (post P5.1–P5.2) and classify per "Core vs. example-specific
  plots" above. **DONE.** Findings recorded directly in
  `telemetry_manifest.yaml`'s `event_categories` map (with reasoning
  comments per category) rather than a separate table — that map *is* the
  audit result, and `write_summary()`'s `MISMATCH`/`DRIFT` check keeps it
  honest against real runs going forward. Also found and fixed the
  `random.py` selection-emission gap in the course of this audit (see the
  paused-block/finding write-up above). No fwdllm-specific plots (e.g.
  variance-check retry rate) were added — none of the 8 categories needed a
  new plot to become non-degenerate once P5.2/P5.5's emission gaps were
  fixed, so this was deferred as unnecessary rather than skipped.
- [x] **P5.6 — Run `analyze_run.py` against a real fwdllm smoke run
  post-fixes** and confirm every populated event category produces a
  non-degenerate plot. **DONE** — the `python3`-invocation issue from the
  prior session is gone. Ran against saved
  `run_20260701_001330_fwdllm_n30_smoke` telemetry: 30 artifacts written
  cleanly; directly verified the `overrun_rate_over_rounds.pdf`/
  `trainer_time_split_over_rounds*.pdf` data now spans 51 distinct
  `progress_key` x-values (`200..250`) vs. 1 collapsed value under plain
  `round`. Also ran against a real saved async_cifar10 run as a regression
  check — clean, full plot set, no behavior change. **Still open**: compare
  against a fresh fluxtune run once the Part 2 deadlock fix is
  GPU-validated (its `plots/` dir couldn't be tested before since the run
  produced zero telemetry) — that requires the currently-running GPU
  experiment to finish.
- [x] **P5.7 — Write the onboarding contract/checklist** (README section)
  once P5.1–P5.6 establish what "done" looks like, so the next new example
  doesn't silently repeat this gap. **DONE** — `scripts/analysis/README.md`
  (new file): plot-category table, manifest schema, 7-step wiring checklist,
  and a gotchas section (trainer-side `selection.selector` artifact,
  stale-reselection-cache bug class).

**All of P5.1–P5.7 are done.** P5.3 remains intentionally narrower than its
original "fully generalized" design (see its own entry above) — the
interim `progress_key()` stopgap plus P5.4's manifest cover every need
found so far; a further generalization is not currently blocking anything.

---

## Immediate next steps (in priority order)

1. ~~Fix fwdllm's `_round_selected_ends` stale-cache gap~~ — **DONE** (Part 3).
2. ~~Run the full test suite~~ — **DONE**: `python3 -m pytest lib/python/tests/`
   passes clean, 426 passed, 7 skipped, 0 failed, including all new tests
   from Parts 2–3.
3. ~~Commit + push~~ — **DONE**, 5 commits (see TL;DR).
4. ~~Verify `run_sequential.sh` configs for the 1.5h, 3-baseline run~~ —
   **DONE** (Part 4 follow-up); caught the `--max-data-id` early-termination
   bug during verification, before any GPU time was spent. **Experiment
   ran to completion, all 3 baselines, no deadlock.**
5. ~~Part 5's telemetry/analysis parity plan~~ — **DONE, all of P5.1–P5.7.**
   See Part 5's subtask checklist and the 2026-07-01 ~21:25 EDT update near
   the top of this doc for the full account, including verification against
   real GPU telemetry (not just synthetic).
6. ~~Once the running GPU experiment finishes: check whether it passed
   cleanly...~~ — **DONE.** All 3 baselines finished with no deadlock;
   `fwdllm_plus`'s round-count throttle did not reproduce at n=100;
   `analyze_run.py` confirmed against fwdllm's real n=100 telemetry
   (`plots/performance/` populates, manifest check clean). See the
   2026-07-01 ~21:25 EDT update for the numbers.
7. **Not done, and the only concrete open item**: commit + push Part 5's
   changes (`fwdllm_aggregator.py`'s telemetry emission, `random.py`'s
   selection-emission fix, `analyze_run.py`'s manifest/progress-key changes,
   the new manifest/README/test files — see "Files touched this session").
   Also open: reading fwdllm's actual accuracy/loss curves for a
   learning-progress verdict (separate from the deadlock/throttle
   verdict this session focused on).

## Files touched this session (for a clean diff review)

- `lib/python/flame/selector/async_oort.py` — reclaim ordering + completeness fix
- `lib/python/flame/selector/async_random.py` — same fix
- `lib/python/flame/selector/fedbuff.py` — completeness-only fix
- `lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py` — new
  `_agg_cycle_contributed_ends` duplicate-contribution guard
- `lib/python/tests/mode/test_async_sim_ordering.py` — +2 tests
  (`TestSendTimeoutReclaimsConcurrencySlot`)
- `lib/python/tests/mode/test_asyncfl_duplicate_contribution.py` — new file, 3 tests
- `lib/python/examples/fwdllm/expt_scripts/run_sequential.sh` — new flags
  (`--agg-goal`, `--c-async`, `--min-initial-trainers`, `--avail-trace`)
- `lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py` — new
  `_prune_departed_from_round_cache` helper, wired into
  `_select_ends_respecting_reselect_gate` (Part 3's `_round_selected_ends`
  stale-cache fix)
- `lib/python/tests/mode/test_fwdllm_reselection.py` — extended
  `_FakeChannel`/`_FakeAggregator` with departure modeling, +3 tests
  (`TestStaleCachePruning`)
- `scripts/analysis/analyze_run.py` — new `progress_key()` helper (Part 5's
  interim round-granularity fix), wired into `sim_time_by_round`,
  `trainer_rounds_by_round`, `system_plots`' round-time-split loop, and (P5.2
  follow-on) `queue_depth_over_rounds.pdf`/`staleness_over_rounds.pdf`/
  `commit_cadence_over_rounds.pdf`'s `EVENT_AGG_ROUND` groupings plus
  `time_to_target()`'s `sim_map` lookup. Smoke-tested against real fwdllm +
  async_cifar10 telemetry (P5.6) and synthetic fwdllm-shaped `agg_eval`/
  `agg_round` telemetry (P5.2). Not yet committed.
- `lib/python/tests/analysis/test_progress_key.py` — new file, 5 tests
  covering the async_cifar10 no-op case, fwdllm's round-major/data_id-minor
  ordering, the 200-multiplier's no-collision guarantee vs.
  `total_data_bins=150`, and the concrete round-collapse-is-fixed
  regression (733-event single-round collapse -> 51 distinct buckets).
- `lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py` — P5.2:
  new `agg_eval`/`agg_round` telemetry emission in
  `_process_aggregation_goal_met` (imports `flame.telemetry` +
  `build_agg_eval`/`build_agg_round`). Not yet committed; this is the change
  the currently-running GPU experiment's fwdllm baseline will pick up.
- `lib/python/tests/mode/test_fwdllm_agg_telemetry.py` — new file, 7 tests
  exercising the real `_process_aggregation_goal_met` method against a
  minimal stand-in (real `torch.nn.Linear(1,1)` model, stubbed
  aggregate/eval_model).
- `lib/python/flame/selector/random.py` — P5.5: added `emit_selection(...)`
  call in `select()`'s SEND branch (fwdllm/fwdllm_plus's real aggregator-side
  selector had never emitted `selection` telemetry at all).
- `lib/python/tests/selector/test_random_selection_telemetry.py` — new file,
  4 tests, verified against the real `select()` method.
- `lib/python/examples/fwdllm/telemetry_manifest.yaml` — new file, P5.4's
  per-example manifest (`model_param_count`, `progress_hierarchy`,
  `event_categories`).
- `scripts/analysis/analyze_run.py` — P5.4: `_find_manifest_path()`,
  `load_manifest()`, `configure_from_manifest()`; `write_summary()`'s
  manifest `event_categories` cross-check (`MISMATCH`/`DRIFT` detection).
  Verified against real fwdllm + async_cifar10 GPU telemetry. Still not yet
  committed (see "Immediate next steps").
- `lib/python/tests/analysis/test_manifest.py` — new file, 9 tests covering
  the manifest loader's found/absent/malformed-YAML/CLI-override-guard
  behavior.
- `scripts/analysis/README.md` — new file, P5.7's onboarding contract:
  plot-category table, manifest schema, wiring checklist, gotchas.
- This doc.
