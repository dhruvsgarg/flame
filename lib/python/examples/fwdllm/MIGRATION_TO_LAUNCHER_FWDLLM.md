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
reselection cache. **All four fixes are implemented and covered by
regression tests; the full suite passes (426 passed, 7 skipped, 0 failed).
Nothing has been committed yet and no GPU experiments have been (re-)run
against the fixes below** — that's the immediate next step. Living doc —
update as findings land.

## TL;DR — where things stand

- **Root cause of the `fluxtune` full-stall: FOUND AND FIXED** (code change,
  not yet experimentally re-validated on GPU). See "Fixes applied" below.
- **Same bug class found and fixed in 2 sibling selectors** (`async_random.py`,
  `fedbuff.py` selector) that weren't part of the original 3-baseline
  comparison but share the same code pattern.
- **A related invariant gap in the generic asyncfl aggregator was found and
  fixed** while double-checking the fix didn't break "a trainer contributes
  at most once per round."
- **A second, distinct gap in fwdllm's own per-round reselection cache was
  found and is now FIXED** — see "fwdllm's OWN gap" in Part 3 below.
- **`fwdllm_plus`'s 200-vs-7336-round throttle is still not conclusively
  root-caused** — plausible-but-unconfirmed hypothesis: `mobiperf_2st` trace
  scarcity + `aggGoal=2` pacing, not a bug. Needs an expected-throughput
  baseline to confirm either way.
- **Telemetry/plotting harness parity (fluxtune's missing `plots/` dir)**:
  not started.
- **No commits made this session.** Next: commit → push → user runs GPU
  experiments against a list of goals/tests to pass (to be defined at that
  point).

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

**Not yet done**: actually running this (no GPU experiments run this
session, per explicit instruction — the user will run experiments after the
doc/commit/push step, with a list of goals/tests to pass to be defined at
that point).

---

## Immediate next steps (in priority order)

1. ~~Fix fwdllm's `_round_selected_ends` stale-cache gap~~ — **DONE** (Part 3).
2. ~~Run the full test suite~~ — **DONE**: `python3 -m pytest lib/python/tests/`
   passes clean, 426 passed, 7 skipped, 0 failed, including all new tests
   from Parts 2–3.
3. **Commit + push** — not yet done, next up.
4. **Then**, with the user: define the list of goals/tests the GPU
   experiments need to pass (throughput expectations, no-deadlock checks
   across all 3 baselines under the new `run_sequential.sh` flags, etc.) and
   actually run them — this hasn't happened yet this session.
5. Still open, lower priority: root-cause `fwdllm_plus`'s 200-round throttle
   (needs an expected-throughput baseline from trace density × aggGoal, not
   yet built) and telemetry/plotting harness parity for `fluxtune` (its
   `plots/` dir was empty/missing in the round-2 run — not yet investigated
   why `scripts/analysis/analyze_run.py` produced nothing for it).

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
- This doc.
