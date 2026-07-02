# Sim Unavailability -- Design & Staged Plan

## NEXT STEP (Jul 1/2 -- Open A + Open B both ROOT-CAUSED AND FIXED (code + tests); live re-confirmation run is next; PR still blocked)

**`run_20260701_233518_..._syn_20_stream_real` (n=100, `--runtime-s 700`, the exact re-confirmation run this
doc was waiting on) reproduced the hang and pinned both Open A and Open B down to concrete bugs, not just
hypotheses. Both are now fixed in code with regression tests (not yet re-validated on a live run).**

### Open A -- real-mode 90s abandon never freed `selected_ends` -- FIXED

Timeline: distribute progress froze at round 199 (23:48:07); the aggregator then looped 30s `recv_fifo`
timeouts on the same ~30 in-flight ends for another 5 minutes with zero progress; it never self-stopped --
it was ended by an external `KeyboardInterrupt` at 23:53:00 (`Uncaught exception: ... KeyboardInterrupt` in
the agg log), ~340s past its 700s budget. **Root cause:** `_sim_evict_unavail_inflight` (D.1) is working
exactly as coded -- `EVICT_DEBUG` shows `skip_still_avl` for 100% of ~4511 checks across the whole run,
because these stuck ends genuinely aren't UN_AVL per the trace (D.1 only evicts UN_AVL ends, correctly).
The real bug is one layer up: real mode's *only* other stall-recovery path -- the native 90s
`SEND_TIMEOUT_WAIT_S` abandon inside `async_oort.py`/`fedbuff.py`'s `select()` (`_handle_send_state`) -- fired
34 times in this run (`"Removing end ... from self.all_selected since havent got its update in 90"`,
23:49:07-23:50:37) but **only ever did `del self.all_selected[end]`** (`async_oort.py:1546-1547`,
`fedbuff.py:554-555`). It never called `_avail_free_slot_ledger`/`free_stalled_slot`/
`remove_from_selected_ends` -- the thing that actually clears `selected_ends`. Since `channel.ends
(VAL_CH_STATE_RECV)` (`recv_ends`) is derived from `selected_ends`, not `all_selected` (Fix 1's own finding),
the stuck end never left `recv_ends` even after its 90s "abandon" -- the `recv_fifo` loop kept re-timing-out
on the identical end_id forever. `recv_ends` therefore never emptied, and `_aggregate_weights`'s
`max_experiment_runtime_s` self-stop check (`asyncfl/top_aggregator.py:634-671`) is nested inside
`if not recv_ends:` -- unreachable -- so there was no code path left that could end the run on its own. This
predates Batch 3/4 entirely (the 90s block is old code, copy-pasted identically in both selector files) -- it
was masked until now because this is the first real n=100 run where a trainer both (a) genuinely stalls past
90s and (b) isn't UN_AVL, so D.1 correctly declines to touch it and the old 90s path was the only thing left
that was supposed to, but didn't.

**Fix (landed):** `async_oort.py`'s and `fedbuff.py`'s 90s abandon block now also calls
`selected_ends.discard(end)` (the same local `selected_ends` var the block already had in scope, matching the
"invalid prior selection" cleanup a few lines above it) right after `del self.all_selected[end]`, so the end
actually leaves `selected_ends`/`recv_ends`, not just `all_selected`. **Tests:**
`tests/selector/test_send_timeout_frees_selected_ends.py` (3 tests, both selectors, verified to fail without
the fix and pass with it, including a race-safety fix: `extra` is computed *before* the abandon loop runs in
both selectors, so freeing a slot can let the reservoir-sampling step immediately re-select the very same end
in the same call if `extra > 0` -- the fedbuff test pins `concurrency` so `extra == 0` to isolate the abandon
behavior deterministically from that separate, correct, RNG-dependent redispatch behavior).

### Open B -- trainer process never read its own `per_trainer` trace -- FIXED

**Root cause (not timing-related -- confirmed by direct code inspection, not a race):**
`MetadataLoader.get_synthetic_trace(trace_name)` (`flame/launch/spawner.py`), which
`ConfigGenerator.generate_trainer_config` calls to bake `avl_events_syn_20` etc. into every trainer's spawn
config, **never took a `trainer_id` parameter at all** -- it unconditionally returned
`synthetic_traces.yaml`'s shared `pattern` entry for every trainer, regardless of registry identity. This is
a different code path from the aggregator's own trace loading (`flame.availability.trace.load_trace`, used by
`ClientAvailability.read_trainer_unavailability`), which *does* correctly resolve
`per_trainer.get(trainer_key) or pattern` -- so the aggregator's belief about each trainer's timeline was
always individualized while the trainer's own local `avl_state` machine (`trainer/pytorch/main.py`) never
was. Confirmed directly against `synthetic_traces.yaml`: `trainer_054`'s assigned `per_trainer` entry starts
its first transition at t=13800s, but the live run showed it transition UN_AVL at t~600s -- exactly
`syn_20`'s shared `pattern`, not its own trace. No timing/cohort-join dependency is involved -- trace baking
happens synchronously in `generate_trainer_config`, fully before any process spawns or joins.

**Fix (landed):** `get_synthetic_trace` now takes an optional `trainer_id`; when given, it delegates to
`flame.availability.trace.load_trace` (the same canonical per-trainer resolver the aggregator already uses)
instead of reading `pattern` directly, and both call sites in `generate_trainer_config` now pass their
already-available `trainer_id` through. **Tests:**
`tests/launch/test_config_generator.py::TestSyntheticTracePerTrainer` (5 tests against the real shared
metadata bundle -- trainer_054 gets its own trace not the shared pattern, different trainers get different
traces, `generate_trainer_config`'s baked-in hyperparameter reflects it end-to-end; verified to fail without
the fix).

**Also added (per user request, defense-in-depth for this bug class going forward):** trainer startup now
logs `[AVAIL_TRACE] trainer_id=... trace=... n_events=... first_events=... trace_hash=...`
(`trainer/pytorch/main.py`, right after `state_avl_event_ts` is set) -- an md5 of the resolved event list, so
a live run where many trainers share an identical `trace_hash` is visible directly in the aggregator/trainer
logs without needing to cross-reference `synthetic_traces.yaml` by hand the way this session did.

### What's left

Neither fix has been re-validated on a live run yet. **In flight (Jul 2)**: a shorter first-pass confirmation
before the full 7h campaign — felix + fedbuff, syn_50 only, both modes, `--runtime-s 1800`:

```
cd lib/python/examples/async_cifar10
bash scripts/debug_run.sh --baselines 'felix fedbuff' --mode both --trace syn_50 --runtime-s 1800
```

Then `python -m scripts.parity.cli --batch --experiments-dir experiments --baselines felix fedbuff --agg-goal 10`.
Expect: (i) felix real self-stops cleanly (`"stopping run"` in the log, no external kill needed), (ii) each
trainer's `[AVAIL_TRACE]` log line shows a distinct `trace_hash` from its neighbors, not a shared one,
(iii) Batch 4's fixes 2/3 (A6, K6, A7-commit) hold now that a run can actually reach a clean end state. If
this passes, follow up with the full cross-baseline campaign (both traces, `--runtime-s 3150`, n=300 parity
default):

```
bash scripts/debug_run.sh --baselines 'felix fedbuff' --mode both --trace 'syn_20 syn_50' --runtime-s 3150
```

<details>
<summary>Original Open A/B write-up (Jul 1, before the above root-cause) -- kept for history</summary>

**Phase 5/6 (Jul 1) found 3 issues (see "Phase 5/6 results" below); fixes 2 and 3 held up on unit tests, but
the live re-confirmation run for fix 1 (felix real TIMEOUT) found the fix is necessary but not sufficient --
felix real still doesn't self-stop.** Two things are open, tracked separately so fixing one doesn't get
credited to the other by accident:

**Open A -- D.1 eviction still misses most stalled trainers in real mode.** The fix below (un-nesting D.1 from
`if self.simulated:`) is real and does something -- some evictions now happen where zero did before -- but a
fresh felix-real run (`run_20260701_223602_..._stream_real`, n=100, syn_20) still hung: stuck at round 198 for
10+ minutes, 21 distinct trainers piled up unresponsive, only 5 `AWARE_EVICT` events fired total (all in two
tight bursts exactly at the trace's t=600s/t=1200s boundaries), had to be killed externally (SIGINT/SIGTERM
in the trainer log at the same instant the aggregator log goes silent -- no internal `"stopping run"` ever
logged). **Two from-scratch reproductions against the real `AsyncOortSelector` + `ClientAvailability` code
(not mocks)** -- one stalled trainer, one 40/100 simultaneously UN_AVL -- both show eviction working correctly
every cycle, so the mechanism is sound in isolation; something about the live run's actual conditions differs
from both repros. Added temporary diagnostic logging (`[EVICT_DEBUG]` in `_sim_evict_unavail_inflight`,
`client_availability.py`) to pin the exact skip reason (still-available / buffered / already-committed / no
trace) on the next live run instead of guessing further -- **remove this logging once root-caused.** A fresh
short real run (`--runtime-s 700`, just past the t=600s boundary) is in flight to nail this down.

**Open B -- NEW, separate bug found while diagnosing A: trace loading silently falls back to the shared
`pattern` for most trainers instead of each trainer's individually-assigned `per_trainer` entry.**
Hand-verified on two trainers in the same run: `...0423` (registry key `trainer_054`) is assigned a trace
whose first transition is at t=13800s, `...0374` (`trainer_005`) at t=34200s -- neither should ever go UN_AVL
within a 900s run -- yet both actually transitioned UN_AVL at t~600s and back at t~1200s, exactly matching
`syn_20`'s shared `pattern` entry (`load_trace`'s fallback, `flame/availability/trace.py`:
`per_trainer.get(trainer_key) or pattern`), not their own assigned entry. This explains why so many trainers
(21+) pile up simultaneously at the same boundary -- most of the n=100 cohort appears to be silently sharing
one timeline instead of each having its own. **Not yet root-caused** (why does `per_trainer.get(trainer_key)`
come back falsy for most of them, when the static YAML clearly has non-empty entries for both checked keys)
or fixed -- deliberately not touched yet, investigating Open A first so the two don't get tangled together.
This is pre-existing (not something Batch 3/4 introduced) -- it was likely masked until now because this is
the first real run at n=100 scale where D.1 eviction/the send-gate were both live and correct enough to
expose the downstream effect.

**Fixes 2 and 3 below are unaffected by A/B and remain code-complete + unit-tested** (12 tests, `tests/` 586
pass / 7 skip, `scripts/parity/` + `trainer/pytorch/` 125 pass). They still need their own live
re-confirmation once A/B are resolved and a clean felix-real run exists to check A6/K6/A7-commit against.

**PR is blocked on A and B, not just "needs a re-run."** Do not raise it yet.

**Checklist to reach PR-ready (work top to bottom; this is the state as of commit `8722aed2`):**
- [ ] **Waiting on user**: a short felix-real run (`--runtime-s 700`, n=100, syn_20 -- just past the t=600s
  boundary, don't need the full 900s) with the `[EVICT_DEBUG]`/`[AWARE_EVICT]` diagnostic logging already
  committed. If this hasn't arrived yet in a fresh session, don't re-run it yourself unprompted -- check with
  the user first (their env has been the one with a reachable MQTT broker; this repo's own env does not).
- [ ] **Open A**: read the `[EVICT_DEBUG]` lines from that run (`grep -E "EVICT_DEBUG|AWARE_EVICT" <agg log>`)
  and root-cause exactly which skip branch (`skip_still_avl` / `skip_buf` / `skip_committed_or_withheld` /
  `skip_no_trace`) is firing for the stuck trainers, or whether `inflight` itself excludes them (in which case
  the bug is upstream of `_sim_evict_unavail_inflight`, e.g. in `_avail_inflight_ends`'s read of
  `selected_ends`, or the SELECTION_CHECK `_track_trainer_version_duration_s` bookkeeping in
  `asyncfl/top_aggregator.py` -- flagged as an untested hypothesis, not confirmed). Fix + add a regression
  test that would have caught it (the two existing manual repros in this session's transcript, not committed,
  are a starting point but didn't reproduce the bug -- a new test needs to actually reproduce it first).
- [ ] **Open B**: root-cause why `flame/availability/trace.py:load_trace`'s `per_trainer.get(trainer_key) or
  pattern` fallback is returning falsy for most trainers in a real n=100 run, when the static
  `synthetic_traces.yaml` has valid non-empty entries for every checked key. Suspect areas: whether
  `trainer_key` passed into `load_trace` at runtime matches the registry's key format exactly, whether
  `--num-trainers 100` cohort-shrinking touches trace assignment, or an `lru_cache`/`base_dir` mismatch
  between the aggregator's and a component's trace loading. Fix + regression test.
- [ ] Remove the `[EVICT_DEBUG]` temporary logging once Open A is root-caused (it's marked "TEMP DIAGNOSTIC"
  in the code, `client_availability.py`, `_sim_evict_unavail_inflight`).
- [ ] Once A + B are fixed: one clean felix real+sim run (n=100, syn_20, `--runtime-s 900`, matching the
  original Phase 5 shape) to confirm (i) felix real self-stops cleanly (`"stopping run"` in the log, no
  external kill needed), (ii) each trainer's observed transitions match its own assigned trace, not the
  shared fallback.
- [ ] Re-run `scripts.parity.cli --batch` on that clean run and confirm fixes 2/3 (A6, K6, A7-commit) actually
  hold on real telemetry -- every real run so far has hit Open A before getting far enough to check this.
- [ ] Only then: Open Items #1 (legacy `trackTrainerAvail` cleanup) and #2 (mobiperf live exercise) -- both
  pre-date this session, listed in full under "Open items -- pick up in order" near the end of this doc -- are
  still separate prerequisites for Open item #3 (PR write-up itself).

</details>

1. **Fix 1 — felix real-mode TIMEOUT (D.1 proactive eviction, sim-only by accident). ⚠️ Necessary but NOT
   sufficient — see Open A above; live re-confirmation found felix real still hangs.**
   `_sim_evict_unavail_inflight` (D.1, the trace-read boundary eviction felix alone uses) was called only
   inside `if self.simulated:`, alongside `_sim_abandon_stalled` (which genuinely *is* sim-only — real mode
   already has a native wall-clock abandon in the selector itself, `SEND_TIMEOUT_WAIT_S=90` in
   `async_oort.py`/`fedbuff.py`). But D.1 has no sim dependency at all — it reads `_avail_now()` (already
   mode-dispatching) and the selector's own `selected_ends`, both equally valid in real mode. Root cause of
   the hang: `async_oort`'s selector derives `channel.ends(VAL_CH_STATE_RECV)` (what `recv_ends` the
   aggregator's `recv_fifo` waits on) **directly from `selected_ends`** (its own docstring: "In 'recv' state,
   it chooses all ends from `self.selected_ends`") — so a trainer stuck in `selected_ends` forever (never
   evicted, since D.1 never ran in real mode) never left `recv_ends` either, hanging the 30s `recv_fifo` loop
   indefinitely with no way to reach the existing `max_experiment_runtime_s` check (which only runs once
   `recv_ends` goes empty). **Fix:** un-nested D.1 from the `if self.simulated:` block in all three
   `top_aggregator.py` stacks (asyncfl/oort/syncfl) — it's a no-op everywhere except felix (the only baseline
   with `proactive_inflight_evict=True`), so this is a zero-behavior-change for oort/refl/feddance/fedbuff.
   **Tests:** `tests/mode/test_proactive_evict_call_site.py` (6 tests) — D.1 fires in real mode for all three
   stacks, the sim-only abandon doesn't.
2. **Fix 2 — A6/A4dur/K6, sim-mode trainer clock frozen between dispatches.** Two-part fix, both zero-new-comms
   (`v1` stays "trace-read, pull, no extra messages" — see dead-end #9):
   (a) `check_and_update_state_avl()` (`trainer/pytorch/main.py`) now stamps `avail_change.sim_now` with the
   transition's own scheduled trace-timestamp (`state_avl_event_ts[0][0]`, popped just before use) instead of
   `self._sim_now()` at processing time. The trainer already has the *entire* trace loaded locally from
   init — it doesn't need "what time is it now" to know *when* a transition occurred, the trace already says
   so exactly. This makes every recorded transition correct regardless of how late the catch-up runs.
   (b) That alone doesn't help a trainer that's never dispatched again before the run ends — its internal
   `avl_state`/queue never gets a chance to advance at all (`_refresh_avl_state`'s catch-up loop is gated on
   `_sim_now() >= due`, and `_sim_now()` itself is what's frozen). `inform_end_of_training`
   (`syncfl/top_aggregator.py`, inherited by asyncfl/oort) already `channel.broadcast()`s an EOT message to
   *every* connected end regardless of dispatch state — piggybacked the aggregator's final `_avail_now()`
   onto it (sim mode + gate-on only; byte-identical broadcast payload otherwise), and `_fetch_weights`
   (`syncfl/trainer.py`, shared by all three stacks) now calls `_refresh_avl_state()` (hasattr-guarded —
   example-specific hook) right after processing `EOT`. One last wake-up flushes every queued transition with
   its own correct due-timestamp (per fix (a)) before the trainer exits. **Resolves Challenges §5 item 19
   (K6)** the same way: `...0376`/`...0391`'s `sim_send_ts==0` was this exact mechanism (frozen at their one
   early dispatch), not selector-starvation or a checker false-positive. **Tests:**
   `trainer/pytorch/test_avail_change_due_ts.py` (3 tests, due-ts stamping, both modes) +
   `tests/mode/test_eot_avail_catchup.py` (6 tests, EOT broadcast payload gating + trainer-side wake-up hook).
3. **Fix 3 — A7 commit-checkpoint, event-sparse observation scored as if continuous.** NOT the same root
   cause as fix 2 (commit-checkpoint beliefs are individually correct at their own timestamp — the aggregator
   reads live `_avail_now()`, never a frozen value). The bug was in the **checker**: `_fidelity_score`
   (`scripts/parity/checks.py`) always extrapolated ("`_pad_tail`") the last observation forward to the run's
   full `span`, which is right for A6/A7-selection (continuously/densely refreshed every dispatch/round) but
   wrong for A7-commit — "commit" only samples a belief when an actual commit happens, and a trainer that
   legitimately stops committing (typically *because* it went UN_AVL — precisely the state this check exists
   to catch) leaves a silent tail that isn't drift, just absence of a later observation. **Fix:** new
   `extrapolate_tail` parameter on `_fidelity_score`, `False` for the commit-checkpoint call site only —
   truncates the scoring window to `[t_start, last observed t]`, symmetric with the pre-existing
   start-side truncation (`t_start = first observed t`, added when A7-commit's "no seed belief before the
   first commit" case was designed in T3.3). Still catches genuine drift *within* the observed window (see
   the kept `test_a7_commit_checkpoint_fails_on_injected_lag_drift` regression). **Tests:**
   `scripts/parity/test_agg_belief_fidelity.py::test_a7_commit_checkpoint_does_not_extrapolate_past_last_commit`.

**Next: one live re-confirmation run** (felix syn_20 real+sim, n=100, same shape as the Phase 5 run that
found these — a fresh run is in flight as of this edit) — check felix real self-stops cleanly (no TIMEOUT),
then re-run `scripts.parity.cli --batch` and confirm A6/K6 clear and A7-commit's error drops substantially.
Do **not** proceed with branch cleanup / PR write-up (Open item #3) until that confirmation lands.

<details>
<summary>Batch 3 history (B2.0.3 root-cause, T3.0–T3.5 build-out) — collapsed, superseded</summary>

B2.0.3's confirmation failure was root-caused to a confirmed structural bug, independent of the B2.0.3
join-barrier fix: every trainer in every `debug_run.sh`-launched run — real **and** sim, all 6 baselines —
was silently running its own local availability state machine against the trivial always-available `syn_0`
trace regardless of the run's actual `--trace`, because `debug_run.sh`'s trace substitution only ever
patched the *aggregator's* trace config, never the *trainer's* `client_notify.trace`. Fixed (T3.1a) and
regression-tested. T3.0 (shared `AGG_START_TS` origin broadcast) and T3.1b (`_refresh_avl_state()`
mode-dispatch cleanup) landed alongside it. T3.2–T3.5 built the four new absolute (vs. ground-truth)
fidelity checks (A6/A7/A8/K11), each unit-tested against synthetic data only — Phase 5/6 above is the first
time they saw real telemetry, and found the two Batch 4 gaps. Full detail in each task's own section below.

**This whole investigation is symptomatic of a broader gap, not a one-off bug**: every availability parity
check that existed before this session (A1/A3/A4/A5) compares real against sim *to each other* — none
compared either mode against the raw ground-truth trace file directly. That's exactly how both a real-mode
dead-code gap (item 20) and a sim-mode frozen-clock gap (Batch 4 finding 2) could go uncaught for this long.
</details>

---

## Preamble — what this is, what's done, how to verify (read first)

**Goal.** Model client *unavailability* (devices dropping offline mid-training) in the FLAME FL
simulator so a fast **simulated** run (virtual clock, no real sleeps) reproduces what a **real** run
(wall-clock, MQTT, true delays) does — **sim/real parity** — for every baseline, with the feature
**config-gated and default-OFF** (byte-identical to today when off).

**What was built (v1).** A shared availability substrate (`flame/availability/trace.py` +
`ClientAvailability`) mixed into the syncfl base and inherited by asyncfl, so all baselines share one
trace-read effect path:
- **Send-time gate, deliver-late-stale.** A trainer that goes UN_AVL mid-flight *keeps computing*; its
  upload is gated at send-time (real) / buffered to `delivery_ts = max(sct, next_avail)` (sim) and
  committed later as a stale update. Nothing is cancelled or dropped.
- **Two ledgers, never conflated.** Slot ledger (90 s vclock *abandon* frees the in-flight slot) +
  delivery ledger (`pending_withheld[end]=delivery_ts`, commits through the existing staleness gate).
- **Proactive in-flight eviction** — **felix only** (the one fully-aware baseline): frees a slot the
  trace shows UN_AVL at the next selection boundary, no 90 s wait.
- **Starvation / vclock-advance under scarcity.** When the eligible pool is too small to start a round,
  sim advances the vclock to the next availability transition instead of spinning. **B2.0.2 FIXED (T0).**
- **Parity ladder** (`scripts/parity/`) — availability rungs A1/A3/A4/A4dur, withheld_delivery,
  abandon_timeout, starvation_advance, eligible_pool_reduction.

**Two orthogonal axes per baseline (keep separate — see Baseline matrix below).**
1. **Knowledge at selection** (`avail_select_filter`): does the selector read the trace to avoid
   *selecting* trainers currently UN_AVL? aware = yes, unaware = select blind.
2. **In-flight slot-free timing** (`proactive_inflight_evict`): when a *dispatched* trainer goes UN_AVL
   mid-round, free its slot at the next boundary (proactive, felix only) or wait the 90 s vclock abandon
   (reactive-90s, everyone else). Aware-at-selection ≠ in-flight eviction.

The knowledge *model* (how the agg learns state) is **trace-read** for all v1 baselines; message-transport
(`client_notify`) and predictive models are Stage H.

**Hardest parts (where the bodies are buried).**
1. **A3 time-base CONTROL** — sim vclock and real wall must share one origin (`agg_start`); every other
   availability rung is gated on A3. Hard gate.
2. **Two-ledger discipline** — order withheld commits by `delivery_ts`, never `sct` (past-dating bug).
3. **Empty per-task pool corrupts shared `selected_ends`** (Challenge 13) — silent hang; cleanup must key
   off the *connected* pool, not the availability-filtered one.
4. **Per-baseline in-flight accounting** (Challenge 15) — in-flight is NOT a constant: oort over-selects
   (overcommitment), async holds > agg_goal at high concurrency, sync-FedAvg clears each round. Starvation
   threshold and scenario sizing must be derived per baseline.
5. **Starvation must self-terminate** — under perpetual scarcity at the trace end-horizon the sim spins
   without advancing the budget (**BUG B2.0.2**, found at feddance n=18).

**How to verify (real AND sim).**
1. **Regression:** `syn_0` (always-available) byte-identical gate ON vs OFF.
2. **Unit tests:** `cd lib/python && conda run -n dg_flame python -m pytest tests/` +
   `examples/async_cifar10 && conda run -n dg_flame python -m pytest scripts/parity/`. 0 failures.
3. **Smoke:** `scripts/debug_run.sh --baselines <b> --mode both --runtime-s 1800 --trace <syn_20|syn_50> --num-trainers <n>`.
4. **Parity:** `python -m scripts.parity.cli --batch --experiments-dir experiments --baselines <b> --agg-goal <g>`.
   Read **A3 first** (CONTROL gate); if A3 FAILs ignore A1/A2/A4. ROOT-CAUSE = lowest broken rung.
5. **Per-mode sanity:** sim → `[SIM_STARVATION]` only under genuine scarcity, `[VCLOCK_PROGRESS]` advancing,
   **self-stops** at `max_experiment_runtime_s` (`"stopping run"`) — NOT via `[SIM_WALL_CEILING]`. real →
   withheld-then-delivered (not dropped), completes within wall budget, `"stopping run"` present.
6. **Accuracy** (C1/C2) is only trustworthy once parity holds.

---

## Working agreement (standing — read every session)

1. **Common first, one baseline first.** Land shared/library changes once, drive a single reference
   baseline (oort/refl async-vs-sync, felix only for proactive-evict). Don't fan out until it behaves.
2. **Short runs to debug, long runs to confirm.** Gate on unit tests + syn_0 byte-identity + shortest
   syn_20 smoke. Long runs confirm; never find first bugs.
3. **Local deterministic tests before runs.** Prefer a synthetic-trace pytest that exhibits the bug over a
   long run that hunts for it.
4. **Keep this doc crisp.** Completed stages: mechanism + where it lives + exit (2–3 lines). Full detail
   only for active/next. Dead-ends in §6.
5. **No stale content.** The moment a section's content is superseded — a bug gets fixed, a task lands, a
   prediction gets confirmed/contradicted by real data, a "next step" gets taken — collapse it to a 2–3
   line resolved/historical note (or delete it outright if nothing future-facing depends on it) **in the
   same edit**, not "later." A section that still reads as an open plan/risk/blocker after the thing it
   describes is done is a bug in this doc. Active/next work gets full detail; everything else gets a
   one-line pointer to where the outcome landed.
6. **Whole-doc crisp pass on every edit, not just the touched section.** Don't only append/collapse the
   section you're updating — re-read the doc top to bottom and push down anything else the new result
   also supersedes (a run that finished, a prediction it resolved, a watch-list item it answered). The
   top of the doc and the Status section should carry only **open issues** (what was tried, what worked,
   what didn't) and **next steps**; run commands, wall-time estimates, and per-run hypothesis tables for
   work that already finished belong in raw logs, not here — once a run completes, only its *findings*
   (especially new/unexpected ones) earn a place in the doc.

---

## Baseline matrix (CANONICAL — supersedes scattered categorization)

| baseline | sync/async | agg base / entry | knowledge @ selection (`avail_select_filter`) | in-flight slot-free (`proactive_inflight_evict`) | config-gate (as actually run) |
|---|---|---|---|---|---|
| **felix** | **async** | `asyncfl` (← syncfl) / `main_asyncfl_agg.py` | ✅ aware | ✅ **proactive** (felix only) | `simUnavailability` |
| **fedbuff** | **async** | `asyncfl` / `main_asyncfl_agg.py` | ❌ unaware | ❌ reactive-90s | `simUnavailability` |
| **oort** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | ❌ unaware | ❌ reactive-90s | `simUnavailability`⁺ |
| **oort_star** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | ✅ aware | ❌ reactive-90s | `simUnavailability`⁺ |
| **refl** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | ✅ aware | ❌ reactive-90s | `simUnavailability`⁺ |
| **feddance** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | ✅ aware | ❌ reactive-90s | `simUnavailability` |

⁺ oort/oort_star/refl's `baselines.yaml` catalog entry still carries the legacy `trackTrainerAvail:
{enabled: True, type: ORACULAR}` block (pre-dates this project). In every run launched via
`debug_run.sh --trace` (i.e. every run this project has actually executed), the substitution script sets
`simUnavailability=True` for them too — `_init_availability`'s `if sim_unavail:` branch wins, so they're on
the same modern path as the other three. The legacy ORACULAR block only matters as a fallback if the parity
YAML is ever loaded *without* that substitution (`simUnavailability` is never statically set in
`felix_oort_refl_feddance_alpha0.1_parity.yaml` for these three) — untested territory, and the reason this
hasn't been cleaned up: removing the legacy block outright risks silently disabling availability in that
unsubstituted path (no error, just an inert trace) rather than a one-line cleanup. Flagged, not yet fixed.

**Notes.** (1) `ClientAvailability` is defined in `flame/availability/client_availability.py` and mixed into
`syncfl/top_aggregator.py` (`class TopAggregator(ClientAvailability, Role)`); asyncfl extends it (`class
TopAggregator(SyncTopAgg)`); oort's base also carries it — so all six get the same substrate. (2) "aware using trace" is v1; the knowledge model becomes message-transport / predictive in Stage H,
but the select-filter / in-flight-evict *behavior* is unchanged. (3) **felix is the only baseline that de-selects
an in-flight trainer** when it goes UN_AVL; the four aware-at-selection-only baselines (oort_star/refl/feddance and
— at selection — nobody for unaware oort/fedbuff) still hit the 90 s abandon for mid-round drop-offs.

### Flag reference (T1 ✅ — split replaced the conflated `_availability_aware`; see Status)

- `avail_select_filter: bool` — selector excludes currently-UN_AVL trainers from the **selection** pool
  (`get_curr_task_ineligible_trainers`). ON: felix/oort_star/refl/feddance. OFF: oort/fedbuff.
- `proactive_inflight_evict: bool` — gates `_sim_evict_unavail_inflight` (in-flight boundary eviction).
  ON: **felix only**. OFF: everyone else (reactive-90s).
- `tracking_mode` — knowledge-model axis: `trace_read` (v1, live) | `client_notify` (Stage H) | `predictive`
  (future). Replaces the `oracular` value at concept/log level (YAML field *value* compat kept).

---

## Status (Jul 1 — Batch 3 landed + Phase 5/6 run+analyzed; Batch 4's 3 fixes landed+unit-tested, live re-confirmation next, PR still blocked)

B2.0.3's join-barrier fix was masking a SECOND, bigger bug — `debug_run.sh` never wired the trainer's own
trace, universal across all 6 baselines — ✅ ROOT-CAUSED + FIXED (T3.0/T3.1a/T3.1b). **T3.2** (trainer
trace-fidelity vs. ground truth, A6 rung), **T3.3** (aggregator belief-tracking vs. ground truth, A7
rung, both selection and commit checkpoints), **T3.4** (trainer-side [SEND_GATE] delay decomposition,
A8 rung), and **T3.5** (aggregator commit-promptness invariant, K11) are now all done:
`scripts/parity/ground_truth.py` (trace-name resolver + duration-weighted range queries +
`expected_send_gate_wait`, shared across T3.2/T3.3/T3.4), `trainer_trace_fidelity_parity` +
`agg_belief_fidelity_parity` + `send_gate_wait_fidelity_parity` + `commit_promptness_parity` in `checks.py`
(A6/A7 sharing a `_fidelity_score`/`_fidelity_result` core; A8/K11 are scalar-error checks instead — DIST
and INV tier respectively — since they each compare one timing value per event, not a state-fraction
vector), a `sim_now` field on `avail_change` telemetry (T3.2), a new `agg_belief_change` event +
`_record_avail_belief`/`_record_commit_belief` hooks on `ClientAvailability` (T3.3),
`send_gate_wait_s`/`send_gate_sct` fields on `task_send` telemetry (T3.4), and an `actual_commit_ts` field
on the existing `withheld_delivery` event (T3.5) — real findings of missing instrumentation (or, for T3.4, a
wrong host event) found while building each check, see ▶ NEXT STEP for detail including T3.5's — and
per-trainer ground-truth-vs-observed timeline + fidelity-error CDF plots (A6/A7), a wait-duration CDF +
observed-vs-expected scatter (A8), and a commit-slack histogram with a marked zero-line (K11) in
`analyze_run.py`. **Phase 5/6 (Jul 1) done** — K6 resolved via A6 exactly as planned (Challenges §5 item 19),
but A6 itself (and A4dur/K6) turned up a genuine sim-mode trainer-clock-freeze gap, and felix real hit a new
TIMEOUT from an unrelated asyncfl budget-check bug — both are Batch 4, see ▶ NEXT STEP for the full writeup.

**A/B/C/C.6/D/E ✅ CONFIRMED syn_20. F.2 ✅ FIXED (B2.0.2 starvation self-termination).
syn_0 ✅. oort n=25 syn_50 ✅ (starvation fires, K1/K3a PASS). B2.0.1 real recv-barrier ✅ FIXED + confirmed.
T0–T5 pre-work ✅ COMPLETE. T5-smoke ✅ COMPLETE (23/23 PASS, Jun 30). A4dur ✅ FIXED + CONFIRMED (fresh
felix syn_20 pair, `mean_err=0.0`). B2.0.3 (real-mode join-ramp baked into the trace-read clock) ✅
ROOT-CAUSED + FIXED. Tests green: `tests/` 568 pass / 7 skip, `examples/async_cifar10/` `scripts/parity/` +
`trainer/pytorch/` 102 pass.**

**T0 ✅** B2.0.2 starvation self-termination fixed (syncfl/oort/asyncfl); budget check >= ; real-mode wall
guard; 10 regression tests in `test_starvation_termination.py`.

**T1 ✅** `_availability_aware` → `avail_select_filter` + `proactive_inflight_evict` (two-axis flag split);
`[ORACULAR]` → `[TRACE_READ]`; `oracular_trainer_avail_check` → `_trace_read_avail_check`.

**T2 ✅** Per-baseline flags set in parity YAML: felix (filter+evict), oort (off/off), refl/feddance (filter/off).

**T3 ✅** oort_star + fedbuff scaffolded in parity YAML (sim+real entries, 12 total); debug_run.sh updated
to include `oort_star fedbuff` in smoke defaults.

**T4 ✅** 41-test state-fidelity suite in `tests/availability/test_state_fidelity.py` covering
T-state-exact, T-eval-pool, T-withhold-deliver, T-aware-vs-reactive, T-starvation-sync.

**T5 pre-work ✅ (Jun 29)** Overnight-run blockers cleared:
- `oort_star` added to `baselines.yaml` (critical: was missing → `ValueError` at run start).
- `availability_trace: syn_0` added to fedbuff aggregator HP in parity YAML (belt-and-suspenders: `_init_availability` now finds trace via top-level key).
- A5 `state_timeline_agreement` wired into `checks.py` + `report.py` (per-(trainer,t) state agreement, DIST tier, 0.95 tol).
- 28-test config-wiring suite `tests/launch/test_baseline_wiring.py`: verifies all 6 baselines have correct HP after merge, catches `oort_star`-missing class of bug.

### B2.0.2 ✅ RESOLVED (found feddance n=18 syn_50, Jun 29; fixed T0; confirmed n=300 T5-smoke, Jun 30)

Perpetual scarcity at the trace end-horizon pinned vclock at budget; a strict-`>` budget check never
tripped, so the F.2 starvation branch spun until killed by `[SIM_WALL_CEILING]` instead of stopping. Fixed
in `flame/mode/horizontal/{syncfl,oort,asyncfl}/top_aggregator.py` (`>` → `>=` budget check; real branch
also consults wall budget). 10 regression tests in `test_starvation_termination.py`. Holds at n=300 across
syn_0/20/50 (23/23 PASS, zero `SIM_WALL_CEILING`) — closed.

### B2.0.3 ✅ RESOLVED (join-ramp clock skew + the bigger bug it was masking; Jul 1)

Two stacked bugs, both closed: (1) real's trace-read clock included the n=300 MQTT join ramp (~300s),
reading the trace ~300s ahead of sim — fixed by `_mark_join_barrier_done()` re-anchoring
`agg_start_time_ts` in real mode (`syncfl/top_aggregator.py`, `test_join_barrier_reanchor.py`, 6 tests).
(2) That fix alone didn't clear confirmation — root cause was `debug_run.sh` never wiring the trainer's
own trace at all (real trainers ran against always-available `syn_0` regardless of `--trace`), fixed as
Batch 3 T3.1a above. Both superseded by T3.0's shared canonical origin and T3.2's trace-fidelity check,
which now confirm real trainers track the trace with correct timing, not just the right file.

### Completed stages (mechanism + where it lives + exit)
- **A/B ✅** Substrate + A3 time-base CONTROL (origin `agg_start` both modes). A3 PASS oort/felix syn_20.
- **C ✅** Send-time gate, vclock 90 s abandon, `delivery_ts` ordering, `free_stalled_slot` (`ClientAvailability`).
  felix 49/49; oort 39/48 (pre-existing §7).
- **C.6 ✅** `_avail_stamp_end_states` writes `PROP_AVL_STATE` pre-selection. A4dur PASS. 5 availability plots.
- **D ✅** Proactive in-flight eviction `_sim_evict_unavail_inflight` (felix-gated). Task-aware eligibility with
  `_trace_has_avl_eval` 2-state guard. Real send-gate confirmed (withheld n=7, accept_frac=1.0).
- **E ✅** Syncfl path (feddance+refl): abandon/evict/stamp + `_sync_sim_recv_first_k` withhold drain. Accept-stale
  (FedAvg has no staleness gate). feddance 46/47 syn_20.
- **F.2 ✅ code / ⚠ B2.0.2** Unified pre-selection return-early starvation pattern in all three aggregators.
  oort n=25 ✅ (1 event, K1/K3a PASS); feddance starvation blocked by B2.0.2.
- **G.1 ✅** `starvation_advance` rung in `checks.py`+`report.py`; +4 tests.
- **B2.0.1 ✅** Real syncfl recv-barrier bounded with `timeout=min(90 s, remaining budget)`. Confirmed feddance
  n=20 syn_50 (real self-stopped, 54 rounds, no watchdog). Challenge 16 closed.
- **B2.0 ✅** oort cohort-floor guardrail (`[COHORT_FLOOR]` warn + clamp). The other 3 pre-ramp items were
  non-issues (see §6 / §7).

---

## ▶ Batch 2 — ordered task sequence (pick up in order, outside this conversation)

> Each task: scope, files, exit. Land T0 first (it blocks runs). T1–T4 are local/code (no long runs). T5 is the
> run campaign. "Across stages before across baselines, long runs last."

### T0–T4 ✅ DONE — see condensed entries in Status section above

Full scope/files/exit detail for T0 (B2.0.2 fix), T1 (rename + flag split), T2 (baseline categorization),
T3 (oort_star/fedbuff scaffold), T4 (41-test state-fidelity suite + A5 rung) lived here while active; now
superseded by the one-line summaries under **Status** (top of doc) and **Completed stages**, and by §6/§7
where a specific decision needs a longer-lived home. Not duplicating here — this section is a pointer only.

### T5-smoke ✅ COMPLETE (Jun 30) — 23/23 PASS @ n=300, gate cleared for full T5 campaign

`experiments/smoke_20260630_0931/` + `smoke_20260630_1040/`: pytest (536p/7s), syn_0 sim ×6, syn_20 sim+real
×6, syn_50 sim+real ×{feddance,oort}. 0 FAIL/ERROR/TIMEOUT, 0 `SIM_WALL_CEILING`, all self-stopped;
oort_star/fedbuff scaffolding ran clean. An ad-hoc parity pass on this output found two issues, both
root-caused and fixed the same week — **A4dur** (`duty_cycle_duration`): real-mode selection telemetry
never stamped `vclock_now`, falling back to a `ts-t0` origin ~300s off from sim's trace-anchored one; fixed
by stamping `vclock_now` via `_avail_now()` in both modes at all unconditional call sites, confirmed
`mean_err=0.0` on a fresh felix syn_20 pair. **feddance syn_50 A3 failure** — the same ~300s join-ramp
origin skew, root-caused and fixed as B2.0.3 above.

### T5 — Parity-table campaign (syn_0 → syn_20 → syn_50 → mobiperf, sim+real, all six baselines)
Produce a `PARITY.md`-style table (rows = baselines, cols = traces × modes, cells = pass/tot + ROOT). Order:
1. **syn_0** regression (byte-identity gate ON/OFF) — all six.
2. **syn_20** then **syn_50** n=300, 3 h, sim+real — all six. (T5-smoke was 900–1800s wall, short vs. this
   3h target; `duty_cycle_duration` and the feddance syn_50 A3 regression are both root-caused + fixed, see
   T5-smoke and B2.0.3 above — confirm on fresh runs first.)
3. **mobiperf:** `mobiperf_2st` → `mobiperf_3st_50`/`_3st_75` (3-state → AVL_EVAL + D.2 eval-pool + Challenge 13
   live exercise). Calibrate HELD rungs (`observation_lag`, `Aa` eligible_pool_reduction) once real data exists.
4. **§7 sweep:** resolve/re-classify each row with long-run data (oort K3b/A2/P3; feddance A3 syn_50, A2;
   U5 ρ watch; C2 loss noise).
- **Exit / sign-off:** per-baseline parity green at syn_50 + mobiperf n=300 (A3 gate open, A2/A4/A4dur/A5 PASS);
  `starvation_advance` populated where the scenario admits it (or deprioritized per Challenge 15); all §7 rows
  resolved; HELD rungs calibrated.

### Stage H (FUTURE — out of scope)

Two independent future knowledge-model upgrades, both replacing `trace_read` on the `tracking_mode` axis;
effect logic (select-filter / in-flight-evict) is unchanged by either — only how the agg learns state changes:

- **H.1 Message-transport** (`client_notify` ON for aware baselines): trainers push avl-state changes over
  MQTT instead of the aggregator reading the trace directly + continuous/event-scheduled vclock clamp.
  Re-measure `observation_lag` (must be ≈0) once live. C.5/D.1 hook already built for it.
- **H.2 Predictive**: a learned/heuristic model of trainer availability (no ground-truth trace read or
  message push) — the `predictive` value on `tracking_mode`. Not designed yet.

---

## ▶ Batch 3 — Trace-Fidelity & Delay-Enforcement Overhaul (✅ DONE incl. Phase 5/6 — Jul 1; Batch 4 fixes landed, live re-confirmation next)

### Why this batch exists

Every availability parity check that existed before Jul 1 (A1/A3/A4/A5) compares **real against sim to
each other**. None of them compares either mode against the **raw ground-truth trace file** directly.
That blind spot is exactly how the real-mode send-time gate (Challenges §5 item 20) could be completely
dead code — never withholding a single update, for every baseline, for the whole project so far — without
any existing check ever catching it: both modes could (and did) diverge from ground truth in different
ways while the *relative* checks stayed superficially plausible, or misattributed the divergence to a
clock-origin issue (B2.0.3) instead of a missing enforcement mechanism.

This batch adds **absolute** (vs. ground truth) fidelity checks at three levels — trainer state,
aggregator belief, and delay/commit-timing enforcement — alongside the existing relative checks, which
stay valuable for a different question ("do the two modes agree with each other?", not "is either one
correct?"). Building both is deliberate: A5 catching "real and sim disagree" doesn't tell you *which one*
is wrong; A6/A7 (below) will.

**Design principle (per session working agreement — clean, not layered).** One shared ground-truth-lookup
module, reused by every new check (trainer, aggregator, delay) rather than four bespoke trace-parsing
implementations. One shared canonical time origin between trainer and aggregator, both modes, rather than
each side computing its own (that's literally what caused B2.0.3). One mode-dispatching function per
mechanism (e.g. a single `_refresh_avl_state()` that reads `_sim_now()` or wall-since-origin depending on
`self.simulated`) rather than parallel sim/real code paths that can silently drift apart — which is how
item 20 happened in the first place (a sim-only path with a `return` guard, and the real path just never
built). Every new telemetry field lands on the shared per-mode phase-timing infra already in place
(`_PHASE_FIELDS` in `checks.py`, the T_ phase report block) rather than inventing a parallel reporting
mechanism.

**Sequencing.** T3.0, T3.1a, T3.1b, T3.2, T3.3, T3.4, and T3.5 are all ✅ **done** (Jul 1) — T3.1a
(config-wiring fix) was independent and landed first; T3.0 (shared origin) then unblocked T3.1b (the
trainer-side avl-refresh cleanup, which needs T3.0's origin to be meaningful in real mode); T3.2 (trainer
trace-fidelity) built on all three; T3.3 (aggregator belief-tracking) reused T3.2's `ground_truth.py` module
and its `_fidelity_score`/`_fidelity_result` core (refactored out during T3.3, now shared by both); T3.4
(trainer delay decomposition) also extended `ground_truth.py` (`expected_send_gate_wait`) but is a
scalar-error DIST check, not a state-fraction one, so it does NOT reuse `_fidelity_score` — see T3.4's
section for why; T3.5 (aggregator commit-promptness) turned out to depend on T3.3's `compute_delivery_ts`
mechanism (already generalized as "max of active gates") more than on a NEW belief-store concept — see
T3.5's section. T3.1a was originally planned as just "T3.1" before implementation revealed the real bug was
elsewhere — see its section below for the full correction. **Batch 3 is done; Phase 5 (one real run) is
next**, not a further code task. Land one baseline (felix — async, aware, proactive-evict,
the tightest/simplest case to validate against) all the way through T3.2–T3.5 before fanning out to refl →
oort (validates the unaware/commit-only-checkpoint path) → feddance (the baseline that surfaced all of
this) → oort_star/fedbuff → full 6-baseline sweep, per the standing working agreement (§ "Common first, one
baseline first"). Trace ramp: syn_0 (trivial, should show ~perfect fidelity — a new regression gate,
doubly so now that T3.1a makes trainer trace assignment trustworthy) → syn_20 → syn_50 → mobiperf (exercises
the AVL_EVAL
belief checkpoint, folds into Open item #2).

### ▶ Implementation phases before the first real run (read this before resuming — Jul 1)

**Principle (per session working agreement — test locally before running, but don't run once per task
either).** T3.2 needs zero new instrumentation — `avail_change` telemetry already exists and, as of
T3.0/T3.1a/T3.1b, is now genuinely correct in real mode. T3.3/T3.4/T3.5 each need new telemetry emitted
*during* a run before there's anything to analyze — but the checker/plot code for all four (A6/A7/A8/K11)
can be built and unit-tested against **synthetic** data without running anything real, exactly like T3.0/
T3.1's tests. So: land all of Phases 1–4 below (telemetry + ground-truth infra + checkers + plots, all four
tasks) with full unit coverage first, **then exactly one real run** (Phase 5) that exercises T3.0/T3.1's
fix live *and* generates telemetry for all four new checks at once, then analyze (Phase 6). Don't run
before Phase 4 is done — a run before then only re-confirms T3.0/T3.1, which is lower value than getting
the new checks' real data in the same run.

**Phase 1 — Telemetry instrumentation (code only, unit-testable standalone, no run needed):**
- T3.3 ✅ done, but **this line was also wrong** (third instance of the same class of gap as T3.2's, and
  the Batch 3 T3.2 correction above): the plan called for a fresh `agg_belief_change` emission at BOTH the
  `selection` checkpoint (`_avail_stamp_end_states`) and the `commit` checkpoint. Building it found the
  `selection` half already has a telemetry trail (`PROP_AVL_STATE` → `selection_train.per_trainer.avl_state`
  via `emit_selection`) — only `commit` needed new instrumentation (`_record_commit_belief`, called from
  `_sim_withhold_if_unavail` in sim and a new passive call in each stack's real receive loop). See T3.3's
  section for the full writeup.
- T3.4 ✅ done, but **this line was wrong too** (fourth instance of the same class of gap): the plan called
  for wiring `send_gate_wait_s` into `_PHASE_FIELDS`/`trainer_round`. Building it found `trainer_round` is
  emitted BEFORE `_send_weights` runs for that round (tasklet order `train → put`), so `_phase_times`
  merged there structurally cannot see the wait loop's timing — it lands on `task_send` instead (already
  the event `wall_send_ts` uses for the identical reason). See T3.4's section for the full writeup.
- T3.5 ✅ done, and **this line turned out to be over-scoped rather than wrong**: the plan called for new
  `earliest_legally_committable_time` bookkeeping alongside `delivery_ts`/`sct`. Building it found
  `delivery_ts` (via the pre-existing `compute_delivery_ts`) already IS that value — v1 only has one gate
  type (availability), so "max of whichever gates are active" collapses to the single existing computation;
  no new bookkeeping field was needed, just an `actual_commit_ts` stamp on the existing `withheld_delivery`
  event to compare it against. See T3.5's section for the full writeup, including a self-caught overclaim
  (an `_advance_sim_clock`/emit reorder that turned out to be provably a no-op in two of the three stacks).
- T3.2: **this line was wrong** — turned out `avail_change` needed one new field too (`sim_now`, the
  trainer's own trace-time-basis clock at the transition; its wall `ts` alone can't be compared against a
  trace indexed in trace-seconds). Found and fixed while implementing T3.2 (Jul 1) — see its section below.
  Standing reminder, same shape as the B2.0.3/item-20 one at the top of this doc: a design assumption
  written into this doc as settled still needs verifying against actual telemetry before building on it.

**Phase 2 — Ground-truth reader (shared infra, build once, reuse in T3.2/T3.3/T3.4).** Investigated this
session — **don't reimplement trace loading, it already exists in exactly the shape needed:**
- `flame/availability/trace.py:load_trace(trace_name, trainer_key, base_dir=None) -> SortedDict[ts→state]`
  is the canonical per-trainer trace loader (already used aggregator-side).
- `flame/availability/client_availability.py:read_trainer_unavailability(trace, base_dir) -> dict` builds
  the full `task_id → SortedDict` map by reading `examples/_metadata/trainer_registry.yaml` (trainer_key
  → task_id) and calling `load_trace` per entry. **Confirmed this session**: `task_id` in the registry
  (e.g. `505f9fc483cf4df68a2409257b5fad7d3c580370`) is byte-identical to the `trainer_id`/`end_id` used
  everywhere in telemetry — no translation layer needed, key straight into it.
- `read_trainer_unavailability` is a `ClientAvailability` method but doesn't reference `self` beyond
  calling `load_trace` — **worth a small refactor** (extract it to a free function in `trace.py` itself,
  have `ClientAvailability` call the free function) so `scripts/parity/ground_truth.py` can call it
  directly without instantiating/hacking around the mixin class. Small, low-risk, do it as part of T3.2.
- **New, still needed:** a trace-*name* resolver — given a run dir, which of the 3 possible config keys
  holds the trace name actually used? Confirmed this session it genuinely varies by baseline, mirroring
  `debug_run.sh`'s own 3-way substitution branch: oort/refl → `hyperparameters.trackTrainerAvail.trace`;
  feddance/fedbuff → `hyperparameters.availability_trace`; would need one more spot-check for the
  HP-level-`client_notify` branch (felix/oort_star, unconfirmed which of the two patterns they land in —
  check `aggregator_config.json` for a fresh run before coding this resolver, don't assume the doc's
  Baseline matrix flags are a perfect predictor of the persisted key). `ground_truth.py` should therefore
  do: try `trackTrainerAvail.trace`, else `hyperparameters.client_notify.trace`, else `availability_trace`,
  in that order (same priority as `debug_run.sh`'s branch) — reading from `aggregator_config.json` in the
  run dir, not re-deriving from the baseline name.

So `ground_truth.py` ends up thin: a trace-name resolver (new, small) + a call into
`read_trainer_unavailability`/`load_trace` (existing, just reused) + a query-time-range wrapper around
`state_at`/`next_avail_after` (existing). Far less new code than the original plan implied — worth
re-reading T3.2's "Shared module" note below with this in mind before starting.

**Phase 3 — Checker rungs (buildable/testable with synthetic data, independent of Phase 5):**
A6 `trainer_trace_fidelity` (T3.2), A7 `agg_belief_fidelity` (T3.3), A8 `send_gate_wait_fidelity` (T3.4),
K11 `commit_promptness` (T3.5) — see each task's own section below for the check design. Each gets a
synthetic-trace + injected-drift unit test (same pattern as T3.0/T3.1's tests) before touching real data.

**Phase 4 — Plots.** Per-trainer timeline overlays (A6/A7, ground-truth band vs. observed band), fidelity
histograms, `send_gate_wait_s` vs. ground-truth-expected scatter (A8), commit-slack histogram (K11) — see
each task section for detail. Extend existing plot infra (C.6's availability plots, the T_ phase plots)
rather than building new plotting machinery.

**Phase 5 — One real run. ✅ DONE (Jul 1).** felix + refl + oort, syn_20, n=100, sim+real
(`experiments/phase5_20260701_1616/`, smoke-suite step 4). **Phase 6 — Analyze. ✅ DONE (Jul 1).**
`python -m scripts.parity.cli --batch --baselines felix refl oort --agg-goal 10`. Both phases found real,
useful things — see "Phase 5/6 results — Jul 1" immediately below for the full writeup; summary: 5/6 runs
PASS (1 TIMEOUT, felix real — new bug, Batch 4 finding 1), K6 resolved (Batch 4 finding 2, not the
starvation/false-positive ambiguity item 19 originally posed), A7-commit-checkpoint still open.

---

## Phase 5/6 results — Jul 1 (Batch 4 gating findings)

`experiments/phase5_20260701_1616/` (felix/refl/oort, syn_20, n=100, sim+real): 5 PASS, 1 TIMEOUT (felix
real), 0 FAIL/ERROR. Root-caused 3 issues, all fixed same-day — mechanism/fix/tests live under "Fix 1/2/3"
above (▶ NEXT STEP), not duplicated here:
- **Finding 1** → Fix 1: felix real never self-stopped — `max_experiment_runtime_s` only checked inside
  `if not recv_ends:`, and `recv_ends` derives from `selected_ends`, which D.1 eviction (the only thing that
  frees a stalled slot without a 90s wait) never touched because it was gated `if self.simulated:`.
- **Finding 2** → Fix 2: sim trainer clock (`_sim_now()`) freezes between dispatches, so `avail_change`
  telemetry misses any trace transition while idle — also resolves Challenges §5 item 19 (K6) as the same
  mechanism, not the starvation/false-positive hypotheses item 19 originally posed.
- **Finding 3** → Fix 3: A7 commit-checkpoint failure was a checker bug, not an FL-system bug —
  `_fidelity_score` extrapolated a trainer's last commit forward to the run's full span instead of
  truncating there, scoring legitimate silence (trainer went UN_AVL) as drift.

A5/K9/K5/C1/C2/U3/U4 all PASS in this run — the three findings are narrow, not a wholesale parity
regression; nothing here contradicts A/B/C/C.6/D/E's existing ✅ CONFIRMED status.

---

### T3.0 ✅ DONE (Jul 1) — Shared canonical time origin (trainer ⟷ aggregator, real mode)

A real-mode trainer computing its own process-start origin reintroduced a per-trainer join-ramp skew (same
class as B2.0.3, moved to the trainer side). Fixed: new `MessageType.AGG_START_TS`, stamped by the
aggregator at every dispatch (all 3 stacks) and cached by the trainer as `_sim_now()`'s real-mode origin.
Files: `flame/mode/message.py`, `{syncfl,oort,asyncfl}/top_aggregator.py`, `syncfl/trainer.py`,
`examples/async_cifar10/trainer/pytorch/main.py`. 11 tests (`test_agg_start_ts_broadcast.py` +
`test_agg_start_ts_sim_now.py`). Exit met.

---

### T3.1a ✅ DONE (Jul 1) — debug_run.sh never wired the trainer's own trace (the actual §5 item 20 cause)

Every trainer in every `debug_run.sh`-launched run (real+sim, all 6 baselines) initialized against the
always-available `syn_0` trace regardless of `--trace` — the substitution only ever patched the
aggregator's HP, never `trainer.config_overrides.hyperparameters.client_notify.trace` (what `main.py`
actually reads). Sim's gating is aggregator-side so it was unaffected; real's send-gate is entirely
trainer-side, so it was silently inert the whole time. Fixed in `scripts/debug_run.sh` (now also sets
`client_notify.trace` for all 6 baselines, verified as a true deep-merge so `.enabled` isn't clobbered). 9
tests (`tests/launch/test_debug_run_trace_substitution.py`, execs the actual heredoc out of the script, not
a reimplementation). Exit met — closes item 20's wiring half (the `trackTrainerAvail` cleanup half is still
open, see Open items).

---

### T3.1b ✅ DONE (Jul 1) — Real-mode `_refresh_avl_for_sim` cleanup

Renamed to `_refresh_avl_state`, dropped its now-redundant `if not self.simulated: return` guard (`_sim_now()`
already mode-dispatches as of T3.0) — closes the "two independently-evolving copies" gap that let T3.1a's
bug hide. `examples/async_cifar10/trainer/pytorch/main.py`. 4 tests (`test_refresh_avl_state.py`, includes
an explicit regression guard that real-mode `_refresh_avl_state()` is not a no-op). Exit met.

---

### T3.2 ✅ DONE (Jul 1) — Trainer trace-fidelity check + plot ("Pillar 1")

Per-trainer duration-weighted TVD between observed (`avail_change` telemetry) and ground-truth trace state,
independently per mode (A6, DIST tier — never compared cross-mode, that's A5). Needed one new field,
`avail_change.sim_now` (trace-time-basis clock at the transition — wall `ts` alone isn't comparable to a
trace indexed in trace-seconds). New shared module `scripts/parity/ground_truth.py` (trace-name resolver +
duration-weighted range queries, reused by T3.3/T3.4). Plot: `trace_fidelity_plots()` in `analyze_run.py`.
18 tests (`test_ground_truth.py`). Exit met for code+tests; the ≥0.95 numeric bar itself did NOT pass on
the first real run — found the sim-mode frozen-clock gap fixed by Batch 4 fix 2 (Phase 5/6 results), not a
checker bug.

---

### T3.3 ✅ DONE (Jul 1) — Aggregator belief-tracking abstraction + fidelity check ("Pillar 2")

What the aggregator *believes* about each trainer's availability, tagged by checkpoint (`selection`/`commit`)
and scored against ground truth per mode (A7, DIST tier, 4 result keys). Selection-checkpoint reuses the
existing `PROP_AVL_STATE`/`emit_selection` trail (no new telemetry); only `commit` needed a new hook
(`_record_commit_belief` in `client_availability.py`, wired into sim's `_sim_withhold_if_unavail` and a new
passive real-mode call in each stack's receive loop, guarded `if not self.simulated:`). Shares
`_fidelity_score`/`_fidelity_result` core with A6 (refactored out during this task). Plot:
`agg_belief_fidelity_plots()`. 17 tests (`scripts/parity/test_agg_belief_fidelity.py` +
`tests/availability/test_agg_belief.py`). Exit met for code+tests; commit-checkpoint's ≥0.95 bar did NOT
pass on the first real run — root cause was a checker bug, fixed as Batch 4 fix 3 (Phase 5/6 results).

---

### T3.4 ✅ DONE (Jul 1) — Trainer-side delay decomposition telemetry ("Pillar 3", trainer half)

`send_gate_sct`/`send_gate_wait_s` (real-mode only): trace-time clock and wall-time actually spent blocked
in the UN_AVL send-gate wait loop. Rides on `task_send`, not `trainer_round`/`_PHASE_FIELDS` —
`trainer_round` fires before `_send_weights` runs for that round, so the wait loop is structurally invisible
to it (same reason `wall_send_ts` already lives on `task_send`). New `expected_send_gate_wait()` in
`ground_truth.py`; new A8 `send_gate_wait_fidelity` check (scalar-error DIST, not `_fidelity_score` — one
duration per event, not a state-fraction vector). Plot: `send_gate_wait_plots()`. 16 tests across
`test_ground_truth.py`/`test_send_gate_wait_fidelity.py`/`test_send_gate_wait.py`. Exit fully met, including
the numeric bar — A8 passed clean on the first real run (`mean_err_s≈0.0–0.27`, `frac_within_tol=1.0`).

---

### T3.5 ✅ DONE (Jul 1) — Aggregator commit-promptness invariant ("Pillar 3", aggregator half)

`commit_slack_s = actual_commit_ts − delivery_ts` should be ≈0 for every withheld-then-delivered commit;
flags early (committed before legally available, hard FAIL) vs. late (held longer than necessary)
separately. Scoped to the withheld population only — normal commits' "slack" against their own `sct` just
measures queue depth, not a real bug. No new bookkeeping needed: `delivery_ts` (existing
`compute_delivery_ts`) already IS the "earliest legally committable time" v1 needs; the only new field is
`actual_commit_ts` on the existing `withheld_delivery` event. New K11 `commit_promptness` check (INV tier),
sim-only (real emits no `withheld_delivery`). Also reordered `_advance_sim_clock`/`_emit_withheld_delivery`
in syncfl/oort's drain loops for consistency with asyncfl, though provably a no-op today (see
`_emit_withheld_delivery`'s docstring for the accounting). Plot: `commit_promptness_plots()` (histogram,
zero-line marked). 9 tests (`test_availability_rungs.py` + `test_agg_belief.py`). Exit met for code+tests;
K11's numeric bar SKIPped on the first real run — zero withheld_delivery events at that scale/duration,
needs a run where the gate actually engages (e.g. feddance, or a longer syn_20/syn_50 window).

---

## Parity rungs (availability tier)

- **A1** `avail_composition` — per-state counts, binned. **A3** `trace_time_base_consistency` — CONTROL hard
  gate (dep K3). **A4** `per_trainer_duty_cycle`. **A4dur** duration-weighted TVD (pass `mean_err≤0.05`,
  `frac_within_tol(τ=0.10)≥0.95`). **A5** `state_timeline_agreement` (T4) — per-(trainer,t) exact match.
  A1/A3/A4/A4dur/A5 are all **relative** (real vs sim to each other) — see Batch 3 for the absolute
  (vs. ground-truth trace) counterparts below.
- **A6** `trainer_trace_fidelity` (Batch 3 T3.2, NEW) — per-trainer observed-vs-ground-truth-trace,
  duration-weighted, independently per mode (no real-vs-sim comparison). **A7** `agg_belief_fidelity`
  (Batch 3 T3.3, NEW) — same, but for the aggregator's *belief* about each trainer (mechanism-agnostic:
  trace_read today, client_notify/predictive later), tagged by checkpoint (`selection`/`commit`). **A8**
  `send_gate_wait_fidelity` (Batch 3 T3.4, NEW, real-mode only) — observed send-gate wait vs.
  ground-truth-expected wait per event.
- **withheld_delivery** — dist of `delivery_ts − sct` + staleness + accept/reject split.
- **abandon_timeout** — count/timing of 90 s vclock abandons; fails loud on wall-clock leak.
- **K11** `commit_promptness` (Batch 3 T3.5, NEW, INV tier) — per-event hard invariant: actual commit time
  vs. earliest-legally-committable time, generic over gate reason; supersedes `withheld_delivery`/
  `abandon_timeout` as the primary promptness gate (those stay as secondary distributional diagnostics).
- **eligible_pool_reduction** (`Aa`, HELD), **observation_lag** (HELD pre-Batch-3; A7 makes it live in v1
  trace_read mode too, not just HELD-for-Stage-H) — calibrate at T5.
- **starvation_advance** — vclock jumps under scarcity, count + timing.
- **Ramp:** syn_0 → syn_20 → syn_50 → mobiperf_*.

---

## v1 core decisions (resolved)

- **Knowledge model:** trace-read for all; one shared trace + `state_at(trainer, vclock)` + one effect path.
- **Mid-flight UN_AVL = compute-completes, gate the send, deliver-late (stale).** Real: gate at send-time. Sim:
  buffer at `delivery_ts = max(sct, next_avail_ts)`.
- **Two ledgers, never conflated:** slot ledger (frees `selected_ends`) + delivery ledger (`pending_withheld`,
  commits stale through existing staleness gate). Order commits by `(delivery_ts, end_id)`, never `sct`.
- **Busy ≠ unavailable ≠ withheld** — three distinct non-pool states. Never route busy→UN_AVL.
- **All availability time on the vclock in sim.** Never wall, never a frozen per-trainer clock.
- **Config-gated, default OFF** → byte-identical. `simUnavailability` is the gate for all 6 baselines as
  actually run (see Baseline matrix ⁺ note — oort/oort_star/refl's legacy `trackTrainerAvail` ORACULAR block
  is now vestigial in every `debug_run.sh`-launched run, not a functionally different gate).
- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) × **busy?** × **has in-flight update?** are orthogonal.
  `syn_0/20/50` are 2-state (no AVL_EVAL); `_trace_has_avl_eval` guard collapses D.2 for them.

---

## 5. Challenges / land-mines

1. ✅ Ordering on `delivery_ts`, not `sct` — U6/U3 validated.
2. ✅ A3 time-base drift — hard CONTROL gate; 90 s abandon re-clocked to vclock.
3. ⚠️ A2 two-tolerance trap — bimodal sim vs smoother real → KS shape artifact; means match; improving with run
   length (0.437→0.338). Expect ≤0.2 at n=300/3 h.
4. ✅ Busy ≠ unavailable ≠ withheld — three ledgers.
5. ✅ Real send-gate fidelity — withheld-then-delivered (not drop); accept_frac=1.0.
6. ✅ Determinism — commit ordered by `(delivery_ts, end_id)`.
7. ✅ Compound straggler + UN_AVL cross-product — unit-tested.
8. ✅ AVL_EVAL inert for oort (dispatches 0 eval); `_trace_has_avl_eval` guard.
9. ✅ Staleness on sync changes cohort — K8/U2 movement expected; reuse existing threshold.
10. ✅ Scarcity advance must not skip events — `_next_avail_vclock()` = min(transitions, withheld deliveries).
11. ✅ Regression discipline — syn_0 byte-identity every stage.
12. ✅ Library mixin spans examples — `ClientAvailability`+`trace.py` in `flame/`; never example-local copy.
13. ✅ Empty per-task pool corrupts shared `selected_ends` — ROOT-FIXED: `_handle_send_state` cleanup keys off
    `connected_ends` (not the availability-filtered pool) in all three selectors; +3 tests. `_trace_has_avl_eval`
    guard KEPT (defense-in-depth). **Needs live exercise at mobiperf_3st (T5).**
14. ✅ Scarcity threshold mismatch — fixed by F.2 unified pre-selection pattern.
15. ⚠️ **Per-baseline in-flight accounting + scenario sizing.** In-flight is NOT constant across baselines:
    - **oort (sync, over-selects):** `in_flight ≈ overcommitment·agg_goal − completed` (~0.3·agg_goal extra);
      effective starve trigger `n − unavail < desired_selection (=13)`.
    - **felix/fedbuff (async):** concurrency-bound, `in_flight` can exceed agg_goal; trigger uses the async pool.
    - **refl/feddance (sync FedAvg):** `on_round_completed` clears `selected_ends`; FedDance returns *partial*
      selections, so `eligible ≈ (1−unavail_frac)·n`; trigger `unavail > n − agg_goal`.
    Manage in-flight per baseline — do NOT assume "sync has no in-flight term." Scenario sizing: syn_50 caps at
    ~43 % unavail, so feddance straddle window is narrow (n≈19); pick `n ≈ threshold ÷ (1 − unavail_frac)`.
16. ✅ Real-mode aggregate-recv hang (B2.0.1) — syncfl real `recv_fifo` bounded with `timeout=min(90 s, budget)`.
17. ✅ Sim starvation self-termination (B2.0.2) — perpetual scarcity at the trace end-horizon pinned vclock at
    budget; strict-`>` budget check never tripped → spun until wall ceiling. Fixed T0, confirmed n=300.
18. ✅/⚠️ Real-mode trace-clock included join-ramp dead-time (B2.0.3) — `agg_start_time_ts` stamped before
    the trainer-join wait, so real's `_avail_now()` read the trace ~300s ahead of sim/ground-truth at
    n=300. Fix landed (`_mark_join_barrier_done()` re-anchor) and is correct, but was masking a SECOND,
    bigger bug (item 20) that alone explains why the n=100 confirmation run didn't clear. See B2.0.3
    section above.
19. ✅ **RESOLVED + FIXED (Jul 1, Batch 4 fix 2) — oort `K6 sim_send_ts`, 2/100 sim trainers.** Neither of
    the two original hypotheses (selector-starvation bug / checker false-positive) — the real cause was
    sim-mode `Trainer._sim_now()` returning the frozen `_sim_send_ts` from its last dispatch, so a trainer
    whose only dispatch lands before the vclock's first advance freezes at `0` forever, identical in
    mechanism to A6's mid-run missed-transition failures, just triggered at the earliest possible point in
    the run instead of partway through. Fixed alongside A6 (due-timestamp stamping + EOT final wake-up); live
    re-confirmation pending. See "Phase 5/6 results — Jul 1" (▶ NEXT STEP) for the full writeup.
20. ✅ **ROOT-CAUSED + FIXED (Jul 1) — `debug_run.sh` never wired the trainer's own trace, universal across
    all 6 baselines, both modes.** Every trainer in every `debug_run.sh`-launched run logged
    `Set avl_events_syn_0` at init regardless of the run's actual `--trace` (`syn_20`/`syn_50`/etc).
    `debug_run.sh`'s `--trace` substitution only ever patched the *aggregator's* trace config
    (`h["trackTrainerAvail"]["trace"]` / `h["client_notify"]["trace"]` / `h["availability_trace"]`) or a
    vestigial `trainer.availability.mode` field nothing reads — never
    `trainer.config_overrides.hyperparameters.client_notify.trace`, which is what `main.py` actually uses
    to build `state_avl_event_ts`. Sim's effective gating is aggregator-side (reads the aggregator's
    correctly-substituted trace) so it was unaffected — 90 genuine withheld/stale-bonus commits in sim vs.
    zero in real for the same feddance syn_20 pair, real staleness a perfect `0.0` point mass. This is what
    B2.0.3's confirmation run was actually catching (via the K3b/A3 symptom), not a residual clock-origin
    issue. **Fixed:** `scripts/debug_run.sh` now also sets the trainer-side `client_notify.trace`.
    Regression-tested: `tests/launch/test_debug_run_trace_substitution.py` (9/9 pass), extracting and
    exec'ing the actual heredoc source so the test can't drift from what runs in production. Batch 3 T3.1a.
    (An earlier diagnosis this same session blamed a dead-code guard in `_refresh_avl_for_sim` — that guard
    is real but wasn't the proximate cause, a separate always-on background thread bypasses it; see Batch 3
    T3.1b for that now-secondary cleanup, and the ▶ NEXT STEP note on verifying against logs before trusting
    a root cause.)

---

## 6. Dead-ends (settled — do not retry)

- **busy → UN_AVL routing** — three distinct states.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch ts) — stuck UN_AVL forever; read `_vclock.now`.
- **Wall-clock in sim** for selection gate / 90 s abandon — wall barely advances vs vclock.
- **Per-tick MQTT broadcast** — comms storm; v1 = trace-read pull (zero comms).
- **Ordering withheld commits by `sct`** — past-dating; order by `(delivery_ts, end_id)`.
- **Forking withhold/abandon per stack** — single shared `ClientAvailability`.
- **A4 bare transition fraction** — brittle in trace-read mode; replaced by A4dur + Aa.
- **D.2 excluding AVL_TRAIN from eval on 2-state traces** — empty eval pool wiped `selected_ends`; fixed by
  `_trace_has_avl_eval` guard.
- **Subtracting starvation vclock-jumps from the budget** — breaks parity (real polls scarcity on wall budget;
  sim vclock jump consumes virtual budget symmetrically — they match). Size `--runtime-s` accordingly. *(B2.0.2
  is the opposite problem — failing to advance/stop at all — not a reason to revisit this.)*

---

## 7. Known parity failures (non-blocking — resolve at T5 with long-run data)

| Check | Baseline | Status | Verdict |
|---|---|---|---|
| A4dur `duty_cycle_duration` | all 6 (library-level) | ✅ ROOT-CAUSED + FIXED + CONFIRMED (Jun 30) | Real selection events never carried `vclock_now`; A4dur's real-mode time-base fell back to a join-ramp-skewed origin. Fixed: stamp `vclock_now` for real too via `_avail_now()`. Confirmed on a fresh felix syn_20 pair: `mean_err=0.0`. Closed. |
| A3 `avail_timebase` | feddance | n=300: syn_20 PASS (0.108), syn_50 FAIL (0.227) — root-caused as **B2.0.3**, fix landed. n=100 confirm (Jul 1): **still FAILs both**, worse — syn_20 0.667, syn_50 0.275 — ✅ true root-cause found + fixed (Challenges §5 item 20, Batch 3 T3.1a) | The join-barrier fix (B2.0.3) is real and correct but wasn't the only bug. True cause: `debug_run.sh` never substituted the trainer's own `client_notify.trace`, so real trainers ran their send-gate against a trivial always-available trace while sim (aggregator-driven) correctly used the real one — the two modes ran different dynamics whenever a trainer goes UN_AVL mid-flight. Fixed + regression-tested. Re-confirmation still gated on T3.0/T3.2 landing (clock-origin + fidelity verification). See B2.0.3 section + Batch 3. |
| K3b `overhead_residual` | feddance | syn_20 residual=10.96s rel=0.369; syn_50 residual=2.93s rel=0.1 (tol 0.1) | ✅ Explained, not a clock-advance bug or scale artifact: sim's withheld/stale-bonus commits (absent in real, per above) carry their own training duration into a round's telemetry without correspondingly advancing the vclock. Resolves once Batch 3 T3.0/T3.1b/T3.2 land and both modes run the same dynamics on a verified-consistent clock. |
| K6 `sim_send_ts` | oort | ✅ RESOLVED + FIXED (Jul 1, Batch 4 fix 2) | Sim-mode trainer clock freezes on last dispatch — same mechanism as A6's missed transitions, not selector-starvation or a checker false-positive. Fixed: due-ts stamping + EOT final wake-up. Live re-confirmation pending. See Challenges §5 item 19. |
| A6 `trainer_trace_fidelity` (sim) | felix, refl, oort | ✅ ROOT-CAUSED + FIXED (Jul 1, Batch 4 fix 2) | Sim-mode trainer clock frozen between dispatches — 12/100 trainers per run missed a trace transition entirely. Not a checker bug (hand-verified). Fixed, unit-tested; live re-confirmation pending — see "Phase 5/6 results." |
| A7 `agg_belief_fidelity` (commit checkpoint) | felix, refl, oort | ✅ ROOT-CAUSED + FIXED (Jul 1, Batch 4 fix 3) | Checker bug, not a system bug: `_fidelity_score` extrapolated the last commit belief across the run's full span, wrongly scoring a trainer's post-last-commit silence (typically because it went UN_AVL) as stale drift. Fixed with a truncated `extrapolate_tail=False` window for the commit checkpoint. Live re-confirmation pending — see "Phase 5/6 results." |
| TIMEOUT — asyncfl real self-stop | felix (+ fedbuff) | ✅ ROOT-CAUSED + FIXED (Jul 1, Batch 4 fix 1) | D.1 proactive eviction was sim-only by accident, so a real-mode stalled trainer never left `selected_ends`/`recv_ends`, hanging the aggregator past its own budget. Fixed by un-nesting D.1 from the sim-only gate. Live re-confirmation pending — see "Phase 5/6 results." |
| A2 `eligibility` KS | oort | 0.437→0.338 (1.5h→3h); FAIL again @ syn_20 n=300 smoke | Bimodal-vs-smooth shape artifact; means match; improving with run length but not yet resolved at n=300/short-run. Investigate alongside A4dur. |
| A2 `eligibility` KS | feddance | FAIL @ syn_20 n=300 smoke | Earlier "clears at n=300" (n=25 data) not confirmed by smoke run — re-opened, investigate alongside A4dur. |
| K3b `overhead_residual` | oort | rel≈0.116 | Run-length sensitive; P3 gates at n=300. Investigate T5. |
| P3 `trainer_speed` | oort | ratio=1.153 (tol 1.15) | Marginal tail at n=300; gates K3b. Investigate T5. |
| `throughput` | oort | FAIL @ syn_20 + syn_50, n=300 smoke | Baseline-specific; lower priority than A4dur/A2. |
| `avail_composition`/`commit_visibility`/`total_commits` | fedbuff | FAIL @ syn_20, n=300 smoke | Baseline-specific; lower priority. |
| C2 `loss` | feddance | avg_diff≈0.16 (few eval pts) | Early-training noise at α=0.1; K8/C1/utility PASS. |
| U5 `inter-arrival` ρ | feddance | 0.659→0.381 (syn_20→50) | Watch at mobiperf. |

---

## Open items — pick up in order

1. **Legacy `trackTrainerAvail` cleanup for oort/oort_star/refl — flagged, NOT done.** Their
   `baselines.yaml` catalog entries still carry `trackTrainerAvail: {enabled: True, type: ORACULAR}`. In
   every actual `debug_run.sh --trace`-launched run this resolves to the same `simUnavailability=True`
   path as the other 3 baselines (a merge-order quirk in `debug_run.sh`'s `--trace` substitution, not a
   static config value — see Baseline matrix ⁺ note for the exact mechanism). Do **not** just zero out
   `enabled`/`type` in `baselines.yaml` — that risks silently disabling availability (no error) for any
   invocation that doesn't go through that substitution script (e.g. the static parity YAML loaded
   directly). Safe path: (a) add `simUnavailability: true` statically wherever oort/oort_star/refl's
   trace gets set, so they're self-sufficient independent of the substitution-script quirk; (b) only then
   zero the legacy `enabled`/`type`; (c) there's no pytest coverage of `debug_run.sh`'s generator logic,
   so this needs an actual generation + short real run to verify, not just unit tests.
2. **Minimal mobiperf live exercise before declaring "mature enough to port"** — see recommendation below.
   Not yet run: the entire mobiperf (3-state, AVL_EVAL) path is untested live in this whole project so
   far (only syn_0/20/50, which are 2-state and collapse AVL_EVAL away). Challenge 13's fix (§5 item 13)
   explicitly still says "needs live exercise at mobiperf_3st."
3. **PR workflow — BLOCKED, not ready.** Gated on: (a) items 1–2 above, and (b) **Batch 4 live
   re-confirmation** (see ▶ NEXT STEP above) — Phase 5/6 (the real run Batch 3 was itself gating on) found
   three real gaps, all now fixed and unit-tested (code + 12 tests, both suites green) but not yet confirmed
   on a live run: (i) felix/fedbuff real-mode never self-stopped when only some trainers stalled (D.1
   proactive eviction was sim-only by accident), (ii) sim-mode trainers' own clock froze while idle, so
   trainer-side fidelity telemetry silently missed transitions (A6/K6/A4dur symptom — fixed via due-timestamp
   stamping + an EOT final wake-up), and (iii) the A7 commit-checkpoint checker wrongly extrapolated belief
   across a trainer's post-last-commit silence (fixed with a truncated scoring window). Once a fresh
   felix/refl/oort real+sim run confirms all three (no TIMEOUT, A6/K6 clear, A7-commit's error drops): clean-diff PR for this branch → write up the
   design decisions that were kept (durable, what's in the doc now) → write up decisions rejected / not
   pursued (currently scattered across §6 Dead-ends + inline "why not X" notes — worth a final sweep to
   make sure nothing rejected got lost) → fold both into this doc (already mostly done by §5/§6/§7) → then
   carry the relevant parts into `lib/python/examples/fwdllm/simulate_fwdllm.md`, which **already exists**
   and already defers unavailability to "follows `UNAVAILABILITY_DESIGN.md` as its template" — so the
   destination is wired, just needs the actual content once this doc is in PR-ready shape.

### Recommendation: minimal bar before porting fwdllm (not yet acted on)

Don't gate the port on the full 3h × 6-baseline × 4-trace T5 campaign — that's for paper-quality parity
numbers, not "is the substrate solid enough to build on." Smaller bar, roughly in order:
1. **Batch 3 (T3.0–T3.5), at least through felix** — A4dur ✅ confirmed clean; B2.0.3/feddance-A3 traced to
   a confirmed real-mode dead-code gap (Challenges §5 item 20), not a clock residual — the fix and the new
   absolute (vs. ground-truth) fidelity checks are the actual bar now, not a bare re-confirmation run. See
   ▶ NEXT STEP / Batch 3 section.
2. **mobiperf_2st** (simplest 3-state trace) for at least one async baseline (felix) and one sync baseline
   (refl or oort_star) — not full 3h, ~30–45min is enough to exercise the AVL_EVAL split and Challenge
   13's empty-pool cleanup *live* for the first time, and to run Batch 3's checks against a 3-state trace.
3. Skip the full syn_50/mobiperf sweep across all 6 baselines and both modes as a porting gate — separable,
   can run in parallel with/after the port starts, not a blocker before it.
