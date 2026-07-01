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
experiment (`n=100`, `c=30`, `aggGoal=10`, `syn_0`) is **currently running**
against these fixes (launched by the user outside this session). While that
runs, a *second*, orthogonal gap was found: fwdllm's telemetry/analysis
tooling is not example-agnostic and has real, confirmed holes — see Part 5.
Living doc — update as findings land.

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
  root-caused** — deferred: waiting on the current GPU run to see if it
  persists post-fix before spending more effort here.
- **All 5 commits pushed.** A 1.5h, 3-baseline (`fluxtune`, `fwdllm_plus`,
  `fwdllm`) GPU experiment is currently running (see Part 4 for the exact
  invocation and the `--max-data-id` gotcha caught during verification).
- **NEW: telemetry/analysis tooling is not example-agnostic — real gaps
  confirmed, plan being built** — see Part 5. This is independent of
  whether the running GPU experiment finds more baseline bugs, so it's the
  active work item while that experiment runs.

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

**Status: this experiment is currently running** (launched by the user
outside this session, after the above verification). `--num-gpus` was left
at each YAML's checked-in default (8) — not overridden, since that's a
physical-resource call, not a config-correctness one.

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
3. **Unexplained anomaly, flagged not chased**: one trainer's `selection`
   event in that same file names `"selector": "FedBuffSelector"` — not
   fwdllm's actual `random` selector — with identical `chosen`/
   `eligible_fingerprint`/`decision_fingerprint` values repeated verbatim
   across *every* trainer's file in the run. Fingerprint fields strongly
   resemble `test_selector_contract.py`/`test_selection_determinism.py`
   determinism-test output, not live FL traffic. Leading hypothesis (not
   confirmed): `FLAME_TELEMETRY_DIR` leaking across processes (e.g. set in
   a shell that later ran the test suite) let an unrelated pytest run
   append into this run's telemetry file. **This needs to be root-caused
   before any new analysis is trusted** — if telemetry files can be silently
   contaminated by unrelated processes, that's a correctness problem for the
   whole measurement pipeline, independent of fwdllm.

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

Rather than leave finding #1 (round-granularity collapse) unaddressed while
P5.3's generalized manifest design is still just a proposal, a small,
hardcoded fix landed directly in `analyze_run.py`: a `progress_key(r)`
helper (right after `by_event`) that returns plain `round` when a record has
no `data_id` field (async_cifar10 — unaffected, identical output to before),
or `round * 200 + data_id` when it does (fwdllm's `trainer_round` records
only — `200 > total_data_bins=150` keeps ordering correct). Wired into every
call site that buckets `EVENT_TRAINER_ROUND` records by round for an
`_over_rounds`-style plot or a per-round dict later used as a plot x-axis:
`trainer_rounds_by_round` (feeds `overrun_rate_over_rounds.pdf` and the data
unlock curve), `sim_time_by_round`, and the per-round dicts inside
`sanity_plots` (`trainer_time_split_over_rounds.pdf`),
`_participation_heatmap`, and `_state_fraction_plots` (the
`trainer_time_allocation_*` plots — confirmed populated for fwdllm today).

**Deliberately NOT touched**: sites that cross-match two different event
streams by exact round equality (e.g. `sanity_plots`' aggregator-observed-
vs-trainer-reported overhead comparison at the `agg_obs`/`tr_rep` join, and
`selection_plots`' `actual_util` lookup keyed by `(round, end_id)`) — folding
`data_id` into only one side of such a join would silently break the match.
Also not touched: every `EVENT_SELECTION`-only call site (see finding #3 —
selection events don't carry `data_id` today anyway, so `progress_key` is a
no-op there; changing them was pointless until #3 is resolved), and every
`EVENT_AGG_ROUND`/`EVENT_AGG_EVAL`/`EVENT_UTIL_DISPARITY`/`EVENT_AVAIL_CHANGE`
site (all currently empty for fwdllm regardless — see finding #2 — so there
was nothing to fix yet; `progress_key` degrades to plain `round` for these
either way since their builders never set `data_id`).

**Not yet done**: a live run of `analyze_run.py` against real fwdllm
telemetry to visually confirm the previously-degenerate plots now show
multiple distinct x-values (blocked this session by an unrelated tool/sandbox
issue preventing `python3` invocation — needs to happen before trusting this
fix, see P5.6). Manually traced through every edited call site's semantics
instead (checked for warmup-exclusion logic and cross-stream joins that
needed the *raw* round preserved separately from the plotting key).

This is intentionally narrower than P5.3/P5.4's declared design (no
per-example manifest, `data_id`/`total_data_bins=150` are hardcoded into the
analyzer rather than declared by the example) — it unblocks the specific,
demonstrated collapse in finding #1 now; P5.3/P5.4 remain the right
generalized fix for when more `flame.launch` examples onboard.

### Subtasks (status tracked here as work lands)

- [ ] **P5.1 — Root-cause the telemetry cross-contamination anomaly.**
  Confirm/refute the `FLAME_TELEMETRY_DIR` leak hypothesis (check whether
  it's exported in the shell profile vs. only ever passed as a scoped
  subprocess env var by the launcher; check whether
  `test_selector_contract.py`/`test_selection_determinism.py` call
  `telemetry.emit()` when telemetry happens to be enabled). Fix the
  isolation gap if confirmed. **Blocks trusting any analysis built on
  existing telemetry data**, so this goes first.
- [ ] **P5.2 — Instrument `fwdllm_aggregator.py` for `agg_eval`/`agg_round`
  telemetry**, mirroring `asyncfl/top_aggregator.py`'s call sites, adapted
  to fwdllm's sync/data_id-driven loop (`_process_aggregation_goal_met`,
  `_aggregate_grads_sync`/`_aggregate_grads_async`). Unlocks
  `plots/performance/` and `plots/insights/` for fwdllm, which are
  currently structurally empty regardless of any plotting-code fix.
- [~] **P5.3 — Design + land the progress-axis abstraction** in
  `analyze_run.py` (or wherever it ends up living) per "Design direction"
  above. Must be a no-op for async_cifar10 (regression risk: don't break
  its existing, working plots). **Partial**: a hardcoded `progress_key()`
  stopgap landed (see "Interim fix landed" above) covering `trainer_round`-
  sourced plots only — the generalized, manifest-driven version (per-example
  declared hierarchy, not a hardcoded `data_id`/150 constant) is still open.
- [ ] **P5.4 — Design + land the per-example manifest** (model param count,
  progress hierarchy, expected event categories). Wire fwdllm's values in;
  leave async_cifar10 on defaults.
- [ ] **P5.5 — Audit all 8 plot categories** against fwdllm's actual
  telemetry (post P5.1–P5.2) and classify per "Core vs. example-specific
  plots" above; add any genuinely fwdllm-specific plots identified (e.g.
  variance-check retry rate) as new, explicitly-labeled example-specific
  additions, not core changes.
- [ ] **P5.6 — Run `analyze_run.py` against a real fwdllm smoke run
  post-fixes** and confirm every populated event category produces a
  non-degenerate plot (multiple distinct x-values, not one collapsed
  point). Compare against a fresh fluxtune run once the Part 2 deadlock fix
  is GPU-validated (its `plots/` dir couldn't be tested at all before now
  since the run itself produced zero telemetry). **Blocked this session**:
  couldn't invoke `python3` at all (tool/sandbox issue, unrelated to the
  code) to smoke-test the `progress_key()` stopgap above against the saved
  `run_20260701_001330_fwdllm_n30_smoke` telemetry — do this first before
  relying on the new plots.
- [ ] **P5.7 — Write the onboarding contract/checklist** (README section)
  once P5.1–P5.6 establish what "done" looks like, so the next new example
  doesn't silently repeat this gap.

None of P5.1–P5.7 started yet. This is independent of the currently-running
GPU experiment (Part 4) — proceeding with it now rather than waiting, per
user direction.

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
   currently running.**
5. **Active: Part 5's telemetry/analysis parity plan.** A scoped
   `progress_key()` stopgap for finding #1 (round-granularity collapse) is
   implemented in `analyze_run.py` (not yet GPU/smoke-verified — blocked by
   a `python3`-invocation tool issue this session, see P5.6). **Not
   started**: P5.1 (contamination root-cause — still blocks trusting
   `selection`-based plots), P5.2 (aggregator telemetry emission), P5.4
   (per-example manifest), P5.5, P5.7.
6. Once the running GPU experiment finishes: check whether it passed
   cleanly (no deadlocks/stalls across all 3 baselines) and whether
   `fwdllm_plus`'s round-count throttle persists post-fix. If it persists,
   revisit root-causing it (expected-throughput baseline from trace density
   × aggGoal, not yet built) — deferred until we have that data point.

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
  interim round-granularity fix); not yet committed, not yet smoke-tested
  against real telemetry (blocked by a tool issue, see P5.6)
- This doc.
