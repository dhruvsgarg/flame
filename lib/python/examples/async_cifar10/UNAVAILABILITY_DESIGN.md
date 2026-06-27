# Sim Unavailability — Design & Staged Plan

## Working agreement (standing instructions — read every session)

**Implement breadth-first across stages on ONE baseline with short runs; batch the long parity runs at
the end. Never block forward implementation on a long run.** This is the explicit guard against
serializing the whole plan behind a sequence of 45-min/3h runs.

1. **Common first, one baseline first.** Land shared/library-level changes once (in
   `flame/availability/`, the mixin, the two commit loops), then drive each mechanism through with a
   single reference baseline — **oort/refl** for async/unaware mechanisms, **felix** only where a
   mechanism is aware-specific. Don't fan out to every baseline until the mechanism behaves on the
   reference one.
2. **Short runs to debug, long runs to confirm.** Gate *forward implementation* on cheap signals only:
   unit tests, a `syn_0` byte-identity regression, and the *shortest* `syn_20` smoke that actually
   exercises the mechanism (~1800s vclock — syn_20's first `UN_AVL` is at t=600s, so a 300s run
   validates nothing). Discover bugs here. A long run (toward the 3h K2 benchmark) is for *confirming* a
   finished mechanism, never for finding its first bugs.
3. **Across stages before across baselines, long runs last.** Rough the mechanism stack in across
   stages on the reference baseline (short smokes throughout) → widen to the other baselines (short
   smokes, fix what breaks) → only then, once functionality is broadly in place and *expected* to pass,
   launch the **batched** longer parity runs across baselines. One long run should confirm several
   stages at once, not one stage each.
4. **Keep this doc crisp and in-place.** Update sections in place; don't append status logs.
   Completed stages compress to a few lines (mechanism + where it lives + exit met). Keep full detail
   only for not-yet-built stages. A short dead-ends/what-didn't-work ledger may be appended (§6/§9),
   but completed-stage prose gets *shorter* over time, not longer.

## Status (Jun 27)

- **Stage A/B COMPLETE.** Substrate — `flame/availability/trace.py` resolver + `AvailabilityMixin` —
  and the A3 time-base CONTROL landed and validated (A3 `max_rel_diff=0.033 ≤ 0.20` on oort syn_20). v1
  scope **locked** Jun 25: *oracular trace-read for ALL baselines; `client_notify` deferred to Stage H*.
  Built **library-level so it spans examples** (async_cifar10, fwdllm —
  [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) §7), not bolted onto one example.
- **Stage C live-wiring COMPLETE; validation partial.** Send-gate withhold + late re-commit (C.2) and
  vclock 90s abandon (C.3) live as a **shared core in `AvailabilityMixin`** (`_sim_withhold_if_unavail`,
  `_sim_pop_committable`, `_sim_reinject_ready_withheld`, `_sim_abandon_stalled`, robust to both
  selector shapes), called from `asyncfl/_sim_recv_min` (single pop) and `oort/_sim_drain_buffer` (pop
  loop, composed AFTER the §4.9 carry-over gate per Challenge 7). 436/436 unit tests pass; gate-off
  byte-identity preserved (helpers no-op without `_init_availability`).
- **First oort syn_20 sim+real pair (Jun 27): 47/49 enforced PASS** ([parity_oort.json](parity_oort.json)).
  Availability rungs clean — A1–A4 PASS, withheld fired (**n=2**, mean delay 599s, accept_frac 1.0),
  invariants hold. Sole enforced FAIL: **K2 throughput** (sim 6.71 vs real 6.31 s/round, rel_diff=0.061
  vs tol 0.05); `terminal_state` suppressed downstream. K3/K3b PASS. Same short-run signature PARITY.md
  documents at 100% availability (closes by ~3h; this run was ~25 min) — but that history predates the
  wiring, so confirm it (see Next actions), don't assume it away.
- **Two gaps the run exposed:** (a) `avail_composition` is **all-UNKNOWN** — per-trainer
  `PROP_AVL_STATE` identity is discarded before emit; A4 + three plots read trainer-local `avail_change`
  instead of the aggregator's belief (blind in pure-oracular mode) → **Stage C.6** fixes tracking +
  duration rungs + plots. (b) `abandon_timeout` is **SKIP** (no 90s abandon fired) and withheld is only
  n=2 — the C.3 abandon path and the withheld distribution are barely exercised; a heavier trace
  (syn_50) or longer run is needed to populate them (§7).
- **Stage C.6 IN PROGRESS (Jun 27, this session) — resume here.** C.6.1/C.6.2/C.6.3(partial) landed and
  unit-tested; C.6.4 written but **not yet verified**. Stage D not started. See the
  "**Stage C.6 — session handoff**" box at the top of the Stage C.6 section below for the exact
  resume point, what's tested, and what's not.

## ▶ Next actions (per the working agreement — implement; don't block on the long run)

Proceed on the reference baselines without waiting on a long run:
1. **Finish Stage C.6** — see the handoff box in §5 Stage C.6 below for the precise remaining steps
   (verify C.6.4, then the syn_0 byte-identity + short syn_20 smoke gate for the whole stage).
2. **Stage D** aware boundary eviction on **felix** (turn the dormant C.5 hook on) — NOT STARTED this
   session — same short-smoke discipline, do after C.6 is verified.
3. **Cheap K2 disambiguation first (do this before any 3h run):** run a **~25-min `syn_0`** (100%
   avail) oort pair and check K2 there. If syn_0 also shows ~6% at 25 min, K2 is the known length
   artifact — *independent of availability* — and no 3h availability run is needed to clear it. Only if
   syn_0's K2 is clean at 25 min does a longer syn_20 run become necessary (it would mean the wiring
   moved K2).

Then **batch the long runs once**: a longer oort syn_20 confirmation doubles as C.6 end-to-end
validation; widen to felix/feddance smokes; then the cross-baseline parity pass. Do not pay for a long
run per stage.

Short oort syn_20 smoke recipe (the right shape for the short iteration loop):

```bash
cd lib/python/examples/async_cifar10
# sim+real oort pair on syn_20. runtime MUST clear the first down window.
scripts/debug_run.sh --baselines oort --mode both --runtime-s 1800 --trace syn_20
# then check (auto-finds the sim/real pair by baseline tag):
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines oort --agg-goal 10
#   or explicit:  python -m scripts.parity.cli --real experiments/run_..._real \
#                    --sim experiments/run_..._sim --agg-goal 10 --json-out parity_oort_syn20.json
```

**Why `--runtime-s 1800` is the floor (not a 5-min smoke):** syn_20's first `UN_AVL` is at vclock
t=600s (0/300 down before; ~30/300 after). A 300s run reaches no unavailability ⇒ nothing is exercised.
The sim must reach ≥~900 vclock-s; 1800 gives multiple down windows (≈30 min wall, confirmed sufficient
Jun 27). The K2 *confirmation* run goes longer (toward 3h) — but try the cheap syn_0 disambiguator
(Next actions §3) first.

**Prerequisites (read first):** [PARITY.md](PARITY.md) §1–§2 (the causal ladder, role/tier tags,
dependency gating) and its §3 mechanism reference (`_vclock`, §3.drain, §3.resid, §4.5/§4.9, §S.dur).
This feature extends that ladder; every new rung and mechanism below assumes that vocabulary.

**Goal:** let trainers drop in/out of `AVL_TRAIN` / `AVL_EVAL` / `UN_AVL` per their traces inside
the sim, emitting correct client-side effects on the **virtual clock**, without (a) breaking the
parity already won at 100% availability, (b) a per-tick MQTT broadcast storm, or (c) frozen-clock
deadlocks.

---

## v1 SCOPE (read this before anything else)

The single biggest simplification, decided Jun 25: **for v1, the aggregator reads the shared trace
directly (oracular) for EVERY baseline — aware and unaware alike — and `client_notify` is OFF.**
That collapses what used to be the doc's central asymmetry (oracle-read for unaware vs
trainer→agg event message for aware) into **one oracular knowledge path**. The aware/unaware
distinction then reduces to *when and how a stalled slot is freed* (a timing/trigger difference),
**not** to *how the agg learns* a transition.

| | v1 (this spec) | End goal (Stage H, FUTURE) |
|---|---|---|
| How agg learns a transition | **oracular trace read** (all baselines) | aware: real `avl_*` trainer→agg message; unaware: still oracular |
| When availability is applied | **selection boundaries only** (no mid-round clamp) | continuous / event-scheduled at the exact transition vclock |
| Aware mid-flight slot-free | **deferred** (hook built, dormant) → behaves like unaware in v1 | proactive eviction the instant the message lands |
| `client_notify` | OFF | ON for aware baselines |

Everything below is written for v1 unless tagged **[Stage H]**. The architecture keeps the
state-resolution + effect logic identical so the future swap changes only *transport/timing*, not
*effect* (extensibility is a hard requirement).

---

## 0. What already exists (design *with* the grain)

The tree already has **three** availability paths; the 100%-avail runs left them dormant
(`trainer_event_dict`/`trainer_unavail_durations` default `None`). Reconcile these, don't add a fourth.

| # | Path | Where | Time-base | Drives | v1 role |
|---|---|---|---|---|---|
| **A. Agg pull (event trace)** | `get_curr_unavail_trainers()` binary-searches each trainer's `trainer_event_dict` (SortedDict `ts→state`) | oort `top_aggregator.py:643`; example dup `main_oort_sync_agg.py:298` | `_vclock.now` (sim) / `time.time()−agg_start` (real) | `channel.set_curr_unavailable_trainers` at selection | **THE v1 path (all baselines)** |
| **B. Agg pull (duration windows)** | `oracular_trainer_avail_check(end)` tests `(start,dur)` | `asyncfl/top_aggregator.py:1226` | same | per-pick veto | folded onto A (event trace strictly more expressive) |
| **C. Trainer push (notifications)** | `check_and_update_state_avl` pops trace events → `channel.update_trainer_state` → backend → `Channel.update_state` | `trainer/pytorch/main.py:330` + `channel.py:1056` | `_sim_now()` = last dispatch `_sim_send_ts` (sim) / wall (real) | MQTT message | **OFF in v1** → Stage H |

**Two anchoring facts:**
1. **A/B already key on `_vclock.now` in sim** (comment at `asyncfl/top_aggregator.py`:
   *"wall-clock would barely advance vs the sim timeline, so every unavailability window would be
   missed"*). Aggregator-pull on the virtual clock = the no-comms, deterministic, never-freezes path.
   **v1 makes this the source of truth for all baselines.**
2. **C has a frozen-clock defect in sim:** `_sim_now()` returns `_sim_send_ts`, which only updates
   when the trainer is *dispatched*. An unselected trainer never advances → never pops events → never
   notifies; a trainer that goes `UN_AVL` can't be selected → can't advance → **stuck `UN_AVL`
   forever**. v1 sidesteps C entirely for *selection* (fully agg-driven). The trainer still needs a
   correct "now" for the **send-time delivery gate** (below) and telemetry → fix `_sim_now()`/use
   `_vclock.now` (Stage A.3); never reintroduce the frozen per-trainer clock.

---

## 1. Core decisions (all resolved)

### Source of truth — ORACULAR for all baselines in v1 (the asymmetry moved, it didn't vanish)
- **v1:** one shared trace + one `state_at(trainer, vclock)` resolver + one effect path, read by the
  aggregator (oracular) for **oort, refl, felix, feddance alike**. "What the agg believes" and "what
  the trainer is" cannot disagree because both read the identical object on the same clock.
- The only per-baseline difference in v1 is **the trigger/timing of freeing a stalled slot**
  (next §): aware frees proactively at the next selection boundary; unaware frees reactively at the
  90s abandon deadline. **Same effect, different latency.** `availability_aware: bool` selects which.
- **[Stage H]** The end-goal asymmetry returns as *transport*: aware baselines learn via a real
  `avl_*` message (real-time, can free mid-round); unaware stay oracular. The effect logic is
  identical, so Stage H swaps transport only. Per-tick broadcast (agg pings everyone each step) is
  **rejected** (comms-heavy, induces sub-optimal decisions).

### Mid-flight unavailability = COMPUTE-COMPLETES, GATE THE *SEND*, then DELIVER-LATE (stale)
This is the corrected model (Jun 25). It is **NOT** a mid-compute interrupt and **NOT** a lost update.

- **The trainer never stops computing.** Even if the agg (wrongly, at dispatch) thought it was
  available, the trainer runs the train/eval task to completion. What is gated is the **upload**:
  *a trainer whose state at SEND time is `UN_AVL` must not send its update* (it isn't reachable on
  the wire). It **holds the completed result and sends it once it is `AVL_*` again** — now **stale**.
- **Where the gate lives (sim/real asymmetry — both yield the same logical delivery instant):**
  - **Real:** add a *send gate* on the trainer's upload path — block/defer the upload until
    `avl_state ∈ {AVL_TRAIN, AVL_EVAL}`. **This is a documented real-side change** (today the trainer
    only gates at *task start*, `trainer/pytorch/main.py:684,1029`; v1 moves the gate to *send time*).
    Compute still runs; only the send waits. We emulate "can't send while offline" rather than
    implementing real MQTT send/recv drops.
  - **Sim:** no wall-block. The agg-side buffer holds the completed update and commits it at
    **`delivery_ts = max(sct, next_avail_ts)`**, where `next_avail_ts` is the next `AVL_*` window from
    the resolver. It commits as a **stale** contribution (feeds Stage-5/6 staleness), **no recompute.**

### 90s abandon + withhold-deliver are BOTH true — two separate ledgers
The aggregator-side abandon timeout and the trainer-side withhold are **orthogonal and simultaneously
correct** (resolved Jun 25). They touch different ledgers; the discipline is to never conflate them
(PARITY.md Challenge 4):

- **Slot ledger (in-flight count).** `SEND_TIMEOUT_WAIT_S = 90` already exists
  (`asyncfl/top_aggregator.py:59`, enforced at `:406` via `time.time() >= deadline`). At the
  **vclock** 90s deadline the agg *stops blocking* on a stalled trainer, **frees it from
  `selected_ends`/in-flight, and a replacement becomes selectable.** This is the existing
  `RECV_TIMEOUT_WAIT_S` abandon path **re-clocked to `_vclock.now`** (it is wall today — a parity
  hazard, see Challenge 2 / §S.dur). Heuristic basis: train takes ≤60s, so >90s ⇒ assume offline.
- **Delivery ledger (pending withheld).** Tracked **separately** as `pending_withheld[end] =
  delivery_ts`. When it commits it runs through the **baseline's existing staleness gate** — async
  (felix/oort) **accept-stale** (fedbuff weighting); `reject_stale_updates="True"` / sync (feddance)
  **reject if over tolerance** (`asyncfl/top_aggregator.py:735`). **Reuse the existing threshold — do
  NOT invent a new scalar** (Q-new-2, E.2).

### Three correctness invariants (assert these in tests)
1. **No double-count.** Once a stalled trainer is freed (slot ledger), it is no longer in-flight; when
   its withheld update later commits you get **one extra stale contribution** — fine for async
   (aggGoal-based), but the in-flight counter must not go negative and staleness must be the true
   round-delta (§4.5/§4.9 accounting surface).
2. **Cannot re-select a still-down trainer.** No special-casing: the oracular selection gate
   (`get_curr_unavail_trainers`) excludes a trainer while its trace says `UN_AVL`, so any replacement
   is necessarily a *different, available* trainer; the original only re-enters the pool when it flips
   `AVL_*` — about when its withheld update delivers. **Extend `pending_after` to exclude the held end
   until `delivery_ts`, not just `sct`** (§4.5).
3. **Aware vs unaware = trigger only.** Aware frees the slot proactively at the next selection
   boundary (trace shows `UN_AVL`); unaware frees it reactively at the 90s vclock deadline. Single
   code path with an `availability_aware` flag; identical downstream effect (free → replace →
   late-stale-commit).

### Busy ≠ unavailable ≠ withheld — three distinct non-pool states
PARITY.md dead-end: do **NOT** route busy→`UN_AVL`. Busy = `AVL_*` but occupied (hold slot, returns
on time). Unavailable = excluded from selection / freed after abandon. Withheld = result exists,
delivery deferred to `delivery_ts`. Separate states, separate ledgers (Challenge 4).

### Timing, comms, determinism (v1)
- **Boundary sampling, no mid-round clamp.** v1 resolves availability at **selection boundaries**
  (the agg re-reads `state_at` when it selects), exactly as `get_curr_unavail_trainers` already does.
  A mid-round transition becomes visible at the **next** selection boundary. **The mid-round
  event-clamp (`min(next_sct, next_transition_ts)`) is DEFERRED to Stage H** — it is only needed when
  an aware baseline must react the instant a message lands. Dropping it is the main v1 speedup.
- **All availability time is on the vclock in sim** (selection gate, the 90s abandon, `delivery_ts`,
  `next_avail_ts`, telemetry) — never wall, never a frozen per-trainer clock (§S.dur lesson; A3).
- **Determinism.** Oracular pull is deterministic given trace+clock. The abandon + withheld-delivery
  commits must order by **`delivery_ts`** for held ends (Challenge 1) to preserve `SEED=1234`
  real+sim parity. **[Stage H]** the message path must order events by vclock with a defined tie-break
  (Challenge 6).

### Felix mid-flight eviction — hook built, dormant in v1
Aware baselines *should* free the in-flight slot the instant they learn the trainer went `UN_AVL`
(client_notify is real-time; a trace reader can only act at the next boundary and otherwise must
stall). v1 **defers proactive mid-flight eviction** but **must build the eviction effect behind an
abstraction** so it can later be triggered by either the trace (oracular boundary read) or a real
`avl_*` message **without changing the effect logic**. In v1 felix therefore behaves like unaware
(slot freed at the boundary / 90s deadline). It **must** be finishable — design for the swap, ship it
dormant.

### Trace representation
- **One event-trace representation** (`AVL_TRAIN/AVL_EVAL/UN_AVL`, strictly more expressive than
  duration-windows; derive windows if a path still needs them). **Prefer 3-state**; 2-state
  ({avail, unavail}) is allowed but **limited-utility for aware baselines** (felix/fluxtune act on the
  `AVL_TRAIN↔AVL_EVAL` split a 2-state trace collapses; oort is indifferent). Surface granularity in
  telemetry. For v1's first target (oort/refl, unaware) 2-state is sufficient; felix wants 3-state.
- **Single-source the trace + resolver** (like `client_duration.py` in §S.dur): one
  `flame/availability/trace.py` object read identically by the trainer (send-gate/telemetry) and the
  agg (oracular driver). Traces (`mobiperf_2st/3st`, `syn_0`=100%, `syn_20`, `syn_50`) cover all
  n=300, loaded by **name from a config-pointed store** — the canonical
  `examples/_metadata/availability_traces` already exists — so the library code never hardcodes
  `examples/` (Q-new-3).

### Config-gating
**Everything config-gated, default OFF** ⇒ byte-identical to today's 46/46 scoreboard. Master
`sim_unavailability` gate (default False) + per-baseline `availability_aware: bool` +
`availability_trace: <name>` (reconcile with the existing `client_notify["trace"]` /
`client_notify["enabled"]` surface — don't add a parallel fourth knob; see §8).

---

## 2. First-principles factors (the why behind each decision)

- **F1 Clock authority.** One monotonic vclock owns "now"; every availability decision (selection
  gate, 90s abandon, `delivery_ts`, telemetry) is indexed by it, never wall, never a per-trainer
  frozen clock. The trace is **sim-seconds since `agg_start`** (single global origin, both modes); sim
  (vclock) and real (wall-elapsed since `agg_start`) must index the SAME windows — parity rung **A3**
  (the REFL HIGH-1 hazard), a CONTROL for the whole feature.
- **F2 Source of truth** — **v1: oracular for all baselines** (§1). Both aware and unaware share one
  trace + one resolver + one effect; they differ only in slot-free trigger. **[Stage H]** aware moves
  to `avl_*` transport, effect unchanged.
- **F3 Event semantics (what a transition CAUSES).** `→UN_AVL`: excluded from new selection (oracle
  gate) AND its in-flight slot freed — proactively at the boundary (aware) or at the 90s vclock
  deadline (unaware); the in-flight update is **withheld, not discarded**. `AVL_TRAIN→AVL_EVAL`:
  train-pool removal, eval-eligible only (inert where a baseline dispatches 0 eval, Challenge 8).
  `→AVL_TRAIN`: re-enters pool + triggers the withheld delivery. Effects are produced deterministically
  at the **selection boundary** in v1, not continuously (Stage H).
- **F4 Compute-completes / gate-the-send / deliver-late** (§1). Return-stage fates: on-time /
  straggler-hold / **withheld-then-delivered (stale)**. NO permanent cancellation of the *result* (the
  90s abandon frees the *slot*, the *update* still arrives and is accept/reject-gated).
  `_sim_hold_busy_slots` and oort `pending_after`/carry-over must keep a withheld end accounted (out of
  the pool until `delivery_ts`) until its delayed delivery commits.
- **F5 Comms** — v1: **zero** (oracular pull for all). **[Stage H]** aware: bounded by # real
  transitions; per-tick broadcast rejected.
- **F6 Timing** — v1: **boundary-sampled, lag = "until next selection boundary"; mid-round clamp
  deferred.** `observation_lag` ≈ 0 is **not** a v1 invariant (it becomes one at Stage H when
  notifications land); v1 measures the boundary lag and confirms it matches real's oracular cadence.
- **F7 Regression surface (won mechanisms).** §3.resid `_sim_hold_busy_slots`; §4.5 `pending_after`
  (exclude held end until **`delivery_ts`**) / §4.9 carry-over (withheld end must NOT re-enter pool
  during the down window, but its delivery must still commit, stale); §S.pacer/§S.dur/A2c selector
  inputs (fewer candidates move the `pref` percentile — expect A2/S2/A2c shift; score only
  genuinely-eligible trainers); per-baseline return stages each gain a "trainer abandoned/withheld"
  branch; the 90s abandon re-clocked to the vclock. All config-gated ⇒ default-off keeps byte-identity.
- **F8 Determinism** — oracular-pull seed-stable; commit ordered by `delivery_ts` for held ends.
  **[Stage H]** message path ordered by vclock (Challenge 6).
- **F9 Trace** (§1).
- **F10 Starvation.** `oort/top_aggregator` `max_retries` wait-retry fires when too few are available;
  in sim the wait must **advance the vclock** (jump to next availability event / next in-flight
  `delivery_ts`), not spin on wall (Stage F).

---

## 3. Concepts to keep crisp (naming discipline, PARITY.md)

- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) × **busy/occupied?** × **has in-flight update?**
  — three orthogonal axes, never conflate (the busy→UN_AVL dead-end).
- **send-time gate** (the trainer/agg won't *deliver* an update produced by a now-`UN_AVL` trainer)
  vs **task-start gate** (today's real behavior). v1 moves to send-time; compute always completes.
- **slot ledger** (in-flight count; freed at boundary/90s) vs **delivery ledger**
  (`pending_withheld`; commits at `delivery_ts`, accept/reject by existing staleness gate) — two
  ledgers, never one (Challenge 4).
- **transition instant** (vclock the state changes) vs **observation instant** (vclock the agg acts);
  v1 lag = "until next selection boundary", measured not assumed-0 (Stage H → 0 for aware).
- Return fates: on-time / straggler-hold / withheld-then-delivered (stale). No result cancellation.
- **agg awareness** (`availability_aware`: does it free the slot proactively?) ⊥ **how it learns**
  (v1: oracular for all; Stage H: message for aware).
- `_sim_now()` must stop meaning "last dispatch ts" — availability reads the **global vclock**
  (selection is fully agg-driven; the trainer uses the vclock only for the send-gate + telemetry).

---

## 4. New parity rungs (Stage 2 = Availability; extend the ladder)

Each new mechanism leaves the finest-grained check that localizes it (PARITY.md Growth rule).
- **A1 avail_composition** (exists, trivial at 100%): now match per-state counts over the run, binned.
- **A3 trace_time_base_consistency** `[NEW]` (CONTROL/DIST, dep K3): same trace → same windows both
  modes (origin = `agg_start`, both modes). **Hard gate — do not read A1/A2/A4 until A3 passes** (the
  REFL HIGH-1 / Challenge 2 lesson).
- **A4 per_trainer_duty_cycle** `[NEW]` (MECHANISM/DIST, dep A3): on/off fraction per trainer matches.
- **transition_effect** `[NEW]`: counts of slot-frees (boundary + 90s-abandon),
  withheld-then-delivered updates, AVL_TRAIN→AVL_EVAL demotions; sim vs real.
- **withheld_delivery** `[NEW]`: dist of `delivery_ts − sct` (down-window delay) + resulting staleness
  + **accept/reject split** (cross-checks F4 against Stage-5/6 U3 and the staleness gate).
- **abandon_timeout** `[NEW]`: count + timing of 90s vclock abandons; sim vs real (must be on the
  vclock — fails loudly if wall leaks in).
- **observation_lag** `[NEW]`: transition→effect lag (tests F6). v1: matches real's boundary cadence
  (NOT asserted ≈0); **[Stage H]** ≈0 for aware.
- **eligible_pool_reduction** `[NEW]`: A2 (`num_eligible`) tracks real's reduction, not just at 100%.
- **Regression guard:** re-run the syn_0 90-min all-baseline parity with availability OFF → current
  scoreboard byte-for-byte (config-gating proof).
- **Ramp:** `syn_0` (regression) → **`syn_20` (first validation target)** → `syn_50` → `mobiperf_*`.
  Shortest run per effect (run-length budget table); reserve long runs for C1/C2. First mechanism pass
  = 45-min "one rung" budget, 5-min smoke first.

---

## 5. Staged implementation + testing plan

One mechanism per stage, all config-gated, default OFF. **Per the working agreement, each stage has two
exit bars, kept separate:**
- **Implementation exit (gates forward work):** unit tests + a `syn_0` byte-identity regression +
  (where it changes dynamics) a *short* syn_20 smoke showing the mechanism fires. Cheap, per-stage.
- **Parity exit (batched, NOT per-stage):** the 45-min/long sim-vs-real rung pass. Deferred to the
  batched confirmation phase once the mechanism is in place across the reference baselines — one long
  run confirms several stages. The "Validation" lines below feed that batched pass; they are not a
  gate to start the next stage's implementation.

**v1 builds the unaware-shaped oracular path for ALL baselines** (oort/refl first, then felix/feddance
behaving identically modulo the dormant eviction hook); proactive aware eviction + continuous
scheduling are **Stage H**. Context-free names (`_ts`/`_time_s`, `_round`).

### Stage A — Substrate: one trace, one resolver, one clock, one library mixin ✅ COMPLETE (Jun 25)
Single-sourced resolver + mixin landed; file-level spec in §8. `flame/availability/trace.py`
(`load_trace`/`state_at`/`next_avail_after`, one `bisect_right` replacing three inlined copies) +
`flame/availability/availability_mixin.py` (`AvailabilityMixin`: consolidated the three dup
`read_trainer_unavailability`/`get_curr_unavail_trainers` copies, `_avail_now()` on the vclock,
`free_stalled_slot` dormant hook). Mixed into `syncfl/TopAggregator` ⇒ all four stacks inherit. Config
surface (`sim_unavailability`/`availability_aware`/`availability_trace_dir`) added, default OFF ⇒
`trainer_event_dict=None` ⇒ byte-identical. **Exit met:** syn_0 5-min all-baseline smoke clean, 0 errors.

### Stage B — A3 time-base CONTROL (gate for everything above it) ✅ COMPLETE (Jun 26)
A3 `trace_time_base_consistency` + A4 `per_trainer_duty_cycle` rungs landed (origin = `agg_start`, both
modes). **Exit met:** A3 PASS on oort syn_20 smoke (`max_rel_diff=0.033 ≤ 0.20`); A1/A2/A2b/A2c +
K3/P3/T2/U3/U4/U6/C1/C2 PASS. Pre-existing (not B-caused) failures noted at the time: K3b
`overhead_residual`, S3/4 `num_chosen`, Sr `residence` — all rooted in the Sx `system_util` KS
divergence (utility beliefs differ → cascades into selection count + carry-over).

> **Carried forward from B (now owned by Stage C.6):** A4 as written counts transition *fraction*, not
> time-in-state, and reads trainer-local `avail_change` (blind in oracular mode). The duration-weighted
> trace-grounded replacements once sketched here as `Aa`/`A4b` are fully specified in **C.6.3** as
> `Aa`/`A4dur`. The three B-era A4 loader/format bugs were fixed in the Stage-C checker pass (§ Stage C
> "Parity rungs — LANDED").

### Stage C — ORACULAR driver + send-time delivery gate + vclock abandon (oort, refl; then ALL) ✅ WIRED, validation partial

**Mechanism LANDED on both stacks, sim side** (436/436 unit tests; default OFF ⇒ byte-identical). The
C.2/C.3 core is a **single shared effect in `AvailabilityMixin`** (Challenge 12 — not forked per stack);
each commit loop calls in (asyncfl `_sim_recv_min`, single pop; oort `_sim_drain_buffer`, pop loop):
- `compute_delivery_ts` / `free_stalled_slot` / `withheld_held_ends` / `ready_withheld` /
  `commit_withheld` — the delivery-ledger substrate (order by `(delivery_ts, end_id)`).
- `_sim_withhold_if_unavail` (per-update send-gate), `_sim_pop_committable` (asyncfl pop),
  `_sim_reinject_ready_withheld` (re-add due payloads), `_sim_abandon_stalled` (C.3 vclock abandon,
  generic over both selector shapes), `_emit_withheld_delivery` (telemetry).
- Status by sub-item: **C.1** oracular selection gate ✅; **C.2** send-time withhold + late stale
  re-commit ✅; **C.3** vclock 90s abandon (aggregator-side, not the inert wall selector) ✅; **C.4**
  `delivery_ts` ordering ✅; **C.5** `free_stalled_slot` single eviction effect (reused by D and H) ✅.
- Parity rungs LANDED in `scripts/parity/checks.py`: `withheld_delivery` (sim-side structural
  invariants), `abandon_timeout` (CONTROL — **fails loud on a wall-clock leak**), `eligible_pool_reduction`.
  Plus the A4 loader/format fixes and trace-name normalization (`avl_events_syn_20`→`syn_20`, which had
  silently resolved 0 traces ⇒ gate-OFF before).

**Non-obvious invariants to preserve (the parts that bite if forgotten — keep crisp, don't re-expand):**
- Withheld pops **do not advance the vclock** — only an actually-committed update drives
  `_advance_sim_clock`. A re-injected delivery commits at `delivery_ts ≤ vclock` ⇒ it lands in the
  `"withheld"` past-dating bucket (intended stale, not a bug); that bucket is what keeps `withheld_delivery`
  separable from real past-dating regressions (U6).
- **Invariant 1:** an abandoned end whose update later physically arrives must not re-register — guarded by
  `end in committed/pending_withheld`. **Invariant 2:** `withheld_held_ends()` is unioned into the
  unavailable list so a held end stays out of the pool until `delivery_ts`. Both unit-asserted.
- The oort loop must apply its §4.9 carry-over (still-computing) gate **before** `_sim_withhold_if_unavail`
  (Challenge 7); a re-injected delivery is exempt from both gates.

**Jun 27 oort syn_20 result** (`parity_oort.json`, summarized in Status): 47/49 PASS, availability rungs
clean, sole FAIL = K2 throughput (short-run signature). See Status + Next actions for the K2 follow-up.

**Validation REMAINING (feeds the batched parity pass, NOT a gate to start C.6/D):**
- **Confirm K2** via the cheap syn_0 disambiguator first (Next actions §3), then a longer run only if needed.
- **Real-side trainer send-gate** (`trainer/pytorch/main.py`): move the task-start skip (`:684`, `:1029`)
  to a SEND-time gate (compute completes; block upload until `AVL_*`); finish the `_sim_now()`→`_vclock.now`
  fix. Needed to confirm real *withholds-then-delivers* rather than dropping the update (Challenge 5) before
  C.2 is fully signed off. Can trail the sim wiring.
- **felix/feddance activation** needs the `sim_unavailability` master gate plumbed through the spawner
  (asyncfl uses `tracking_mode: client_notify`, not the legacy `trackTrainerAvail` ORACULAR path oort/refl
  ride) — §7. oort/refl configs are ready (`oort_n300_oracular_9may25_syn20.json`,
  `refl_n300_syn20_prob0.7.json`).
- **Batched-pass exit:** A1/A2/A3/A4 PASS, withheld/abandon rungs populated (needs a heavier trace/longer
  run — abandon is SKIP and withheld n=2 at syn_20/25-min), no new past-dating beyond `"withheld"` (U6),
  K2/K3b hold vs a syn_20 real reference; felix/feddance smoke confirms they ride the same path.

### Stage C.6 — Aggregator-side time-in-state tracking, fidelity rungs, plotting fixes

> **Session handoff (Jun 27, paused mid-C.6.4 — resume here).**
> Landed + unit-tested (flame `tests/`: 436/436 +7 skipped; async_cifar10 `scripts/parity/`: 63/63 —
> both green as of this checkpoint, BEFORE the C.6.4 edits below):
> - **C.6.1 DONE.** `per_trainer[end_id]["avl_state"]` in `flame/selector/__init__.py::emit_selection`
>   (universal, one line). `vclock_now` plumbed: `channel.properties["vclock_now"]` set in
>   `oort/top_aggregator.py` (new) and `syncfl/top_aggregator.py` (new) alongside the existing
>   `asyncfl/top_aggregator.py` site; consumed via `channel_props.get("vclock_now")` at every
>   `emit_selection` call site in `oort.py` (3), `refl_oort.py` (1), `feddance.py` (1), `fedbuff.py` (1)
>   (`async_oort.py` already had it).
> - **C.6.2 DONE.** `scripts/parity/avail_state_series.py` — `build_trainer_state_series` /
>   `state_fractions` / `run_span` / `total_variation_distance`. Unit tests:
>   `scripts/parity/test_avail_state_series.py` (11 tests).
> - **C.6.3 PARTIAL.** `A4dur` landed as `duration_duty_cycle_parity` in `checks.py` (registered in
>   `run_all_parity` as `results["duty_cycle_duration"]` and in `CHECK_META`), tests added to
>   `test_availability_rungs.py`. **`Aa` and `observation_lag` NOT built** — both need trace-loading
>   plumbing (trace name + `base_dir` + an end_id→trainer_key mapping) that doesn't exist anywhere in
>   `checks.py`/`cli.py` today; scoped out rather than building it untested under time pressure. Pick
>   these up as a follow-up (§7) once a run exists to calibrate against, per the original `observation_lag`
>   HELD rationale.
> - **C.6.4 WRITTEN, NOT YET VERIFIED — resume HERE.** Edited `/home/dgarg39/flame/scripts/analysis/
>   analyze_run.py` (repo-root `scripts/analysis/`, NOT under `lib/python/examples/async_cifar10/` —
>   the path in this doc's spec below is wrong; the file is example-agnostic and only imports
>   `flame.telemetry.events`, so it does NOT import `scripts/parity/avail_state_series.py` — that lives
>   inside the async_cifar10 example tree and would be a layering violation for a generic script. Added a
>   local `_avl_state_by_round(records)` helper instead (same underlying data source — `per_trainer[
>   end_id]["avl_state"]` on `selection` events — just a round-indexed dict-of-dict rather than the
>   time-indexed list C.6.2 uses, since these plots are round-indexed and `analyze_run.py` can't depend
>   on the example's `parity` package). Rewrote `availability_plots` (3-state funnel incl. eval pool, a
>   real `availability_dynamics.pdf` in the dynamic branch — previously never produced — transition-based
>   churn instead of sample-count churn), `_participation_heatmap` (6-state legend, was 5/binary
>   avail-aware), `_state_fraction_plots` (same source swap). **Remaining before this is done:**
>   1. `python -c "import ast; ast.parse(open('analyze_run.py').read())"` (or just run it) — never
>      syntax-checked after the edit.
>   2. Run it against an existing telemetry dir (e.g. `lib/python/examples/async_cifar10/experiments/
>      run_20260627_113505_dbg_oort_n300_alpha0.1_syn_20_stream_sim`) to confirm no crashes. That run
>      predates C.6.1, so `avl_state` will be absent and the new plots will be empty/skip — this only
>      proves crash-safety, not visual correctness.
>   3. The doc's "per-trainer fidelity plot" (`err_t` sorted bar/CDF, A4dur's visual companion) listed
>      under C.6.4 below was **not** started.
>   4. Visual correctness needs a fresh short syn_20 smoke (the one in "Next actions" / recipe below) —
>      do that AFTER step 1–2 pass cleanly, not before.
> - **Stage D NOT STARTED.**
> - Todo list at pause (for continuity): C.6.1 ✅, C.6.2 ✅, C.6.3 (A4dur ✅ / Aa+observation_lag deferred),
>   C.6.4 🔶 in progress, "run unit tests + syn_0 byte-identity + short syn_20 smoke" ⬜, Stage D ⬜.

**Implement before Stage D's parity validation** (D needs these fidelity rungs to score against), but
*implementation does not block on any long run* — unit tests + syn_0 byte-identity + a short syn_20
smoke are sufficient to land it. Triggered by the Jun 27 read: `A4` (duty-cycle) and three plots
(`availability_dynamics`, `selection_funnel_over_rounds`,
`participation_heatmap`) all read trainer-local `avail_change` telemetry (`build_avail_change`, only
emitted from `trainer/pytorch/main.py:351`) — that's incidental client-side bookkeeping, not what the
aggregator believed when it acted, and it's blind in pure-oracular (unaware) mode. This is the `Aa`/`A4b`
gap §7 already named but never built. The fix reuses data the aggregator already computes rather than
adding new telemetry: `flame/selector/__init__.py::emit_selection` already loops every candidate's
`PROP_AVL_STATE` into `avail_composition` counts — it just discards the per-trainer identity before
emitting. Restore that, and one shared resolver function feeds both the new parity rungs and the plot
fixes (single-source discipline, Challenge 12).

**C.6.1 — Tracking (telemetry, additive, no behavior change).**
- `per_trainer[end_id]["avl_state"] = state_name` — one line in `emit_selection`
  (`flame/selector/__init__.py`), next to where `avail_composition` is built. Selector-level code: fires
  identically for oort/refl/felix/feddance, unaware/aware, oracular/client_notify — no per-baseline fork.
- Add `vclock_now=self._vclock.now` to the `extra` dict at each `emit_selection` call site when
  `self.simulated` (mirrors how `agg_round` already carries `vclock_now`; real needs no change — every
  event already gets a wall `ts` from `TelemetryWriter.emit`, and `wall_elapsed = ts - t0` is the same
  approximation `K8`/`A3` already use).
- **Exit:** byte-identical aggregator behavior (telemetry-only diff); confirm via a syn_0 smoke that
  nothing currently reads the new field.

**C.6.2 — Shared resolver: `trainer_state_series`.**
New helper (e.g. `scripts/parity/avail_state_series.py`, imported by both `checks.py` and
`analyze_run.py`): from `selection` events, build per-trainer `[(t, avl_state), ...]` forward-fill series
on the same time-base both modes already use elsewhere (`vclock_now` sim / `ts - t0` real). Integrate
dwell time between samples → per-trainer fraction-of-run-in-state vector `{AVL_TRAIN, AVL_EVAL, UN_AVL}`
(sums to 1). One function, no duplicated forward-fill logic between the checker and the plotter.
**Landed as `scripts/parity/avail_state_series.py`** (async_cifar10 example tree, time-indexed, used by
`checks.py`). `analyze_run.py` lives at repo-root `scripts/analysis/` and is example-agnostic (imports
only `flame.telemetry.events`), so it can't depend on this example's `parity` package — it gets its own
round-indexed `_avl_state_by_round` helper (C.6.4) reading the identical `per_trainer[...]["avl_state"]`
field. Same source, not literally the same function — see the C.6 session-handoff box above.

**C.6.3 — New parity rungs (`checks.py`, append-only, `CHECK_META`-registered).**
- **`Aa`** (aggregator availability accuracy) — *within one mode*, agg-observed fraction vector vs the
  trace's own `state_at()` ground truth over the same span. Should be ~0 in v1 (oracular) by
  construction; a regression/plumbing sanity check (wrong trace, stale cache, mistimed clock), not a
  real-vs-sim comparison. Runs separately for real and sim.
- **`A4dur`** (duration-weighted duty-cycle parity, real vs sim — the doc's long-deferred `A4b`) —
  per-trainer error = total-variation distance between real/sim fraction vectors (`err_t = 0.5 * Σ_s
  |real_s − sim_s|`, one scalar in [0,1] per trainer). **Population rollup is a distribution, not a
  single number**: report `mean_err`, `p50/p90/p99`, and `frac_trainers_within_tol(τ=0.10)`. **Pass
  rule:** `mean_err ≤ 0.05 AND frac_within_tol ≥ 0.95` — two conditions because a systematic small drift
  (mean) and a real diverging subset (tail fraction) are different failure modes; neither alone is
  robust (a bare `max`, which is what today's `A4` effectively does, is exactly the brittleness being
  fixed). Mirrors the existing "distribution + robust summary" precedent already in this report (`U6`'s
  "KS uninformative on point-mass — passed on mean"). Keep the existing transition-count `A4` alongside
  — it catches a different failure mode (transitions stopping entirely) cheaply.
- **`observation_lag`** (named in §7, HELD) — now buildable from the same series: per trace transition,
  lag until the next selection-boundary sample reflects it. v1 target = matches real's boundary cadence
  (not ≈0; that's Stage H).
- **Tests:** `scripts/parity/test_availability_rungs.py`, synthetic real/sim fixtures with known
  fraction vectors — assert TVD/rollup arithmetic, assert `Aa` is ~0 on a clean oracular fixture and
  non-zero when deliberately desynced.

**C.6.4 — Plotting fixes (`scripts/analysis/analyze_run.py`).**
All four re-point from `EVENT_AVAIL_CHANGE` to `trainer_state_series` (C.6.2):
- `availability_dynamics.pdf` — currently only written in the degenerate static-availability branch;
  the dynamic branch (our syn_20 case) never produces this file at all. Add the real plot: 3-state
  population fraction over time, real vs sim overlaid.
- `selection_funnel_over_rounds.pdf` — currently filters out `task != "train"` (silently drops the
  `AVL_EVAL` pool) and shows 3 flat scalars with no exclusion-reason breakdown. Decompose via
  `avail_composition` (already 3-state, already on every selection event): candidates → (− UN_AVL,
  oracular gate) → eligible → (− busy/in-flight) → chosen; train and eval both included, 3-state labels
  throughout.
- `_participation_heatmap` — swap source, extend the discrete legend from binary unavailable/available
  to the 3 states.
- `_state_fraction_plots` — same source swap (its category shape is already right:
  train/eval/idle_train/idle_eval/unavail).
- New: per-trainer fidelity plot — sorted bar/CDF of `err_t` across the population, the visual companion
  to `A4dur`'s `mean`/`p90`/`frac_within_tol`.

**Sequencing (incremental, fast-iteration-first):** C.6.1 → C.6.2 → C.6.3 (+ unit tests) → C.6.4 → only
then re-run oort syn_20 to validate end-to-end, ideally reusing the longer K2-confirmation run above so
we're not paying for a second long run. Write/adjust code, validate with unit tests and short/cheap
checks first, fix what's found there — spend the long run once the functionality is mostly complete and
expected to pass; long runs are for confirming, not for discovering, the first round of bugs.

**Exit:** `Aa` ~0 both modes; `A4dur` mean/tail numbers cross-checked against the existing `A4`/`A3`
results (which already pass on this data) for rough consistency; all 5 plots visually correct on one
run dir.

### Stage D — [Stage H precursor] AWARE proactive eviction (felix; fluxtune if in scope)
*(Was the v0 "aware immediate-event driver"; in the v1 plan this is the point where the dormant hook
turns ON for proactive boundary eviction, still oracular. True `avl_*` transport is Stage H.)*
- **D.1** Turn on proactive slot-free at the boundary for `availability_aware` baselines (hook from
  C.5): on a `→UN_AVL` observed at selection, free the slot + register the withheld delivery
  separately (`pending_withheld`), reconcile with §3.resid `_sim_hold_busy_slots` — slot freed *and*
  pending tracked, no leak, no double-count (Challenge 4 / invariant 1).
- **D.2** `AVL_TRAIN↔AVL_EVAL`: task-type eligibility via the resolver; only meaningful where the
  baseline dispatches eval (sync oort dispatches 0 — felix-relevant; flag inert baselines, Challenge 8).
- **D.3** Late withheld update = accept-stale (async) through felix's existing staleness path.
- **(D.x deferred to Stage H)** continuous/event-scheduled timing + the `min(next_sct,
  next_transition_ts)` clamp — only needed when reacting the instant a message lands.
- **Tests:** boundary eviction frees slot + tracks pending (no leak); replacement selectable; AVL_EVAL
  gates task type; late update commits stale.
- **Validation:** syn_20, 45-min, felix. A1/A2/A3/A4, transition-effect counts vs real,
  withheld-delivery, U3, U6. **Exit:** mechanism rungs PASS at syn_20; then a full-length run to
  re-confirm felix C1/C2 under unavailability.

### Stage E — SYNC baselines + staleness-gated rejection (feddance)
- **E.1** Apply C/D to the sync path (barrier re-selects the cohort each round).
- **E.2** Staleness rejection: a late withheld update exceeding feddance's tolerance is **dropped**
  (baseline's existing rule, no new threshold). Changes round composition → expect K8/U2 movement;
  validate it's faithful (Challenge 9).
- **E.3** §6.u6 barrier-anchor: unavailability changes which K form the barrier; the barrier-anchored
  U6 lag must be computed over the *actually contributing* cohort.
- **Tests + syn_20 45-min feddance.** **Exit:** A-rungs + U3/U6 + K8 PASS.

### Stage F — Starvation / clock-advance under scarcity (F10)
- **F.1** In the `max_retries` wait-retry, when no one is selectable, **advance the vclock to the next
  availability event (or next in-flight `delivery_ts`)** rather than wall-sleeping. Clamp to the
  nearest of {next transition, next `delivery_ts`, next `sct`}; guard against an all-unavailable window
  spinning forever (Challenge 10).
- **Validation:** syn_50, 45-min (heavier unavailability triggers scarcity), all baselines. **Exit:**
  no stalls; K1 monotone; round cadence faithful at syn_50.

### Stage G — Ladder integration + ramp + sign-off
- **G.1** Land all new rungs in `scripts/parity/{checks.py,report.py}` with deps (A1 now enforced, A3,
  A4, transition_effect, withheld_delivery, abandon_timeout, observation_lag, eligible_pool_reduction).
  Append-only.
- **G.2** Ramp: syn_0 → syn_20 → syn_50 → mobiperf_*. **G.3** Per-baseline sign-off, now *with*
  availability.

### Stage H (FUTURE) — true `avl_*` message transport + continuous scheduling
Turn `client_notify` back ON for aware baselines: swap the oracular boundary read for real
trainer→agg `avl_*` messages, processed **immediately** (mid-round), **without changing the effect
logic** (the C.5/D.1 hook was built for exactly this). Add the continuous/event-scheduled vclock clamp
(`min(next_sct, next_transition_ts)`). Preserve determinism by ordering events on the vclock with a
defined tie-break (Challenge 6). Re-measure `observation_lag` (now must be ≈0 for aware).

---

## 6. Challenges / land-mines

1. **Ordering must key on `delivery_ts`, not `sct`, for withheld updates** (high risk of
   re-introducing past-dating). §3.drain's min-`sct` gate and §4.9 carry-over assume `commit order ==
   sct order`; a withheld delivery commits at `max(sct,next_avail) > sct`. Re-validate U6/U3 after C.
2. **A3 time-base drift is the silent killer** — AND the 90s abandon must move to the vclock with it.
   If sim vclock and real wall-elapsed advance at different rates, the same trace (and the same 90s
   deadline) fire at different real moments → every higher rung diverges and mislocalizes. Hard CONTROL
   gate; do not read A1/A2/A4 until A3 passes. The abandon timeout is wall today
   (`asyncfl/top_aggregator.py:406`) — re-clocking it is part of Stage C, tested by `abandon_timeout`.
3. **Two-tolerance trap on the eligible pool (A2 vs S3/4).** With availability ON,
   `eligible = candidates − in_flight − unavailable`; a small gap can fail A2's tight KS while S3/4
   in_flight passes. Decompose the channel first; don't chase A2 as a separate bug.
4. **Busy vs unavailable vs withheld = three distinct non-pool states; slot ledger ⊥ delivery ledger.**
   The §3.resid dead-end (busy→UN_AVL) ramped in-flight to ~300. The 90s abandon **frees** the slot
   but the withheld delivery is **still tracked** and **still commits** — slot accounting and delivery
   accounting are separate ledgers; conflating them leaks slots, double-counts, or drives in-flight
   negative (invariant 1).
5. **Real-side send-gate fidelity.** v1 adds a trainer send gate (block upload until `AVL_*`) and
   models delivery at `max(sct, next_avail)`. Confirm real felix/oort actually withhold-then-deliver
   under this gate (and don't, e.g., drop the socket and lose the update) on a real syn_20 run; if real
   loses the update instead of delivering it stale, the model is wrong (becomes a lost-update path).
   **[Stage H]** also measure real's `avl_*` notification lag — if non-trivial, lag-0 reflection won't
   match and Stage H timing must model it on the vclock.
6. **Determinism / event tie-break at a shared vclock instant.** Transition, commit (incl. withheld
   `delivery_ts`), and selection events can coincide. Define a total order (e.g. transitions < commits
   < selections, then by `trainer_id`) so `SEED=1234` real+sim parity + exact rungs stay enforceable.
   v1 only needs the commit ordering (Challenge 1); the full order is **[Stage H]**.
7. **Compound states with existing carry-over.** An oort §4.9 carried-over straggler that ALSO goes
   UN_AVL, or a §3.resid held slot whose trainer flips AVL_TRAIN→AVL_EVAL, are real cases. Enumerate
   the (avail_state × occupied × in-flight × withheld) cross-product and assert each cell in tests.
8. **AVL_EVAL may be inert for some baselines** (sync oort dispatches 0 eval). A 3-state trace's eval
   windows then do nothing; report which baselines exercise the eval split (ties to the 2-state
   limited-utility flag, F9).
9. **Staleness-rejection on sync changes round composition (feddance).** Dropping over-stale late
   updates shifts K8/U2/round-count — expect movement, validate it's faithful, reuse the existing
   threshold (no new scalar).
10. **Scarcity clock-advance must not stall or fast-forward past events** (F10/Stage F). Clamp the jump
    to the nearest of {next transition, next `delivery_ts`, next `sct`} and terminate on an
    all-unavailable window. Guard K1 monotone + the K5 failsafe ceiling.
11. **Regression discipline.** Every stage re-runs the syn_0 90-min all-baseline parity and must hold
    the scoreboard byte-for-byte before its syn_20 validation counts. A stage perturbing another
    baseline serializes (one baseline per run round).
12. **Library mixin spans examples — don't fork it.** The `AvailabilityMixin` + `trace.py` live in
    `flame/` and are mixed into all four `TopAggregator`s and reused by fwdllm. Resist re-adding an
    example-local `read_trainer_unavailability`; the three existing copies are being deleted, not
    forked.

---

## 7. Open follow-ups (note here as work lands)

- **felix master-gate plumbing (asyncfl activation):** oort/refl activate via the legacy
  `trackTrainerAvail` ORACULAR path; felix/fedbuff use `tracking_mode: client_notify` and so need the
  `sim_unavailability: true` master gate (+ `availability_trace`) emitted by the spawner into the asyncfl
  JSON config. Until then `_init_availability` returns `trainer_event_dict=None` for felix (gate OFF). The
  commit-loop wiring is already in place; this is config-compilation only. Needed before the felix/feddance
  Stage-C smoke.
- **`observation_lag` rung (HELD):** transition→effect boundary-cadence lag (F6). Needs a trace↔selection
  join (resolve each selection's `vclock_now` against the per-trainer trace, measure lag to the next
  boundary where the effect lands) and a real syn_20 reference to calibrate; deferred until run data
  exists so the check isn't written blind. v1 target = matches real's boundary cadence (NOT ≈0).
- **`A4b` trace-vs-dispatch duration validator** — **promoted to Stage C.6.3 as `A4dur`** (spec there).
  Kept here only as a back-pointer; build it in C.6.
- **Abandon / withheld under-exercised at syn_20 (Jun 27):** `abandon_timeout` is SKIP (no 90s abandon
  fired) and withheld was only n=2 over 25 min — the C.3 abandon path and the withheld distribution have
  almost no run data. Populate them with a **heavier trace (syn_50)** and/or a longer run before treating
  those rungs as validated. (Train ≤60s rarely crosses 90s, so abandon may need a trace whose `UN_AVL`
  lands mid-compute and persists.)
- **K2 cheap disambiguator (do before any 3h run):** a ~25-min **syn_0** oort pair — if K2 is ~6% there
  too, K2 is the length artifact independent of availability and no long availability run is needed
  (Next actions §3).
- **Real send-gate confirmation (Challenge 5):** on a real syn_20 run, verify the trainer
  withholds-then-delivers (stale) rather than dropping the update when it goes `UN_AVL` at send time.
  Decides whether the `max(sct,next_avail)` model is faithful before C.2 is signed off.
- **Q-new-2 sync confirmation:** verify feddance's existing staleness threshold is the right rejection
  gate on a real syn_20 run before E.2 (don't assume the async tolerance transfers).
- **[Stage H] Real notification lag (Challenge 5):** measure on a real felix run before turning
  `client_notify` back on; decides whether lag-0 reflection is admissible or lag must be modeled.
- *(Append new open questions/info needs here as the substrate lands — keep this the single ledger.)*

---

## 8. v1 Implementation Spec — file-level (Stage A + C **LANDED**; 8.3/8.4 still feed forward work)

All paths relative to `lib/python/`.

### 8.1–8.2 Library substrate ✅ LANDED — `flame/availability/{trace.py, availability_mixin.py}`
Code is the source of truth; kept here only as a map. `trace.py`: `load_trace` /
`state_at` (`bisect_right−1`) / `next_avail_after` / `compute_delivery_ts`, single resolver replacing the
three inlined copies, `base_dir` from config (default `examples/_metadata/availability_traces`).
`availability_mixin.py` (`AvailabilityMixin`, mixed into all four library `TopAggregator`s via
`syncfl/TopAggregator`): `_init_availability` (sets `trainer_event_dict=None` when gate off ⇒
byte-identical), `_avail_now()` (vclock in sim / `time.time()−agg_start` real),
`get_curr_unavail_trainers`, the delivery-ledger + `free_stalled_slot` eviction effect, and the live
commit-loop helpers (Stage C list above). Examples inherit for free (async_cifar10, fwdllm).

### 8.3 Trainer send-gate: `trainer/pytorch/main.py` — ⏳ OUTSTANDING (the one un-landed v1 piece)
- Replace the **task-start** skip (`:684-706`, `:1029-1040`) intent with a **send-time** gate on the
  upload path: compute always runs to completion; immediately before putting the update on the
  channel, if `state_at(trace, _vclock.now)` is `UN_AVL`, **hold** and (real) block until `AVL_*` /
  (sim) let the agg-side buffer commit at `delivery_ts`. Document the real-side change in the
  trainer's docstring + a `[SEND_GATE]` log line.
- Fix `_sim_now()` (A.3): availability uses `_vclock.now`, not `_sim_send_ts`; never reintroduce the
  frozen per-trainer clock.

### 8.4 Config surface (`flame/config.py` + spawner)
- Master `sim_unavailability: bool = False` (the §1 gate; off ⇒ byte-identical).
- Per-baseline `availability_aware: bool` and `availability_trace: str`. **Reconcile with the existing
  `client_notify` dict** (`config.py:180`, `client_notify["trace"]`/`["enabled"]`): reuse
  `client_notify["trace"]` as `availability_trace` and keep `client_notify["enabled"]="False"` in v1
  (notifications stay OFF; Stage H flips it). Do NOT add a parallel fourth knob (Challenge 12 spirit).
- `availability_trace_dir` (optional) → `base_dir` for the resolver; default
  `examples/_metadata/availability_traces`.

### 8.5 Telemetry ✅ LANDED (on the vclock)
`abandon_timeout`, `withheld_delivery` (`delivery_ts−sct`, staleness, accept/reject) builders in
`flame/telemetry/events.py`. Still TODO in C.6.1: per-trainer `avl_state` + `vclock_now` on
`emit_selection` (the all-UNKNOWN `avail_composition` gap).

### 8.6 Tests ✅ LANDED (436/436)
Resolver determinism + `syn_0`-inert; `get_curr_unavail_trainers` identical across all four aggregators;
the three ledger invariants; `delivery_ts` ordering; gate-off byte-identity. Outstanding: the real
send-gate tests (8.3) and C.6 rung tests.

### 8.7 Exit criteria (status)
Stage A ✅ (syn_0 smoke clean) · Stage B ✅ (A3 PASS, syn_20) · Stage C: mechanism ✅ / batched parity
exit pending (Stage-C "Validation REMAINING"). Remember the two-bar split (§5 intro): implementation
exit gates the next stage; the parity exit is a batched long-run pass, not a per-stage gate.

---

## 9. Dead-ends (settled — do not retry)

The single ledger of approaches already tried and rejected, so completed-stage prose can stay crisp and
nobody re-derives them. (Cross-refs to the live Challenges in §6.)

- **busy → `UN_AVL` routing** (Challenge 4): ramped in-flight toward ~300. Busy/unavailable/withheld are
  three distinct non-pool states with separate ledgers; never collapse them.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch `_sim_send_ts`): an unselected/`UN_AVL`
  trainer never advances ⇒ stuck `UN_AVL` forever. Availability reads the global `_vclock.now`.
- **Wall-clock in sim** (selection gate *and* the 90s abandon): wall barely advances vs the vclock ⇒ every
  window/deadline is missed. Everything availability-related is on the vclock (Challenge 2).
- **Per-tick MQTT broadcast** (agg pings everyone each step): comms storm + induces sub-optimal decisions.
  v1 is oracular pull (zero comms); Stage H uses bounded `avl_*` messages.
- **Ordering withheld commits by `sct`** instead of `delivery_ts`: re-introduces past-dating (a withheld
  delivery commits at `> sct`). Order by `(delivery_ts, end_id)` (Challenge 1).
- **Forking withhold/abandon per stack**: rejected for a single shared `AvailabilityMixin` effect
  (Challenge 12); the two commit loops call in, they don't reimplement.
- **A4 counting transition *fraction*** (a bare max over `avail_change`): brittle, blind in oracular mode.
  Replaced by duration-weighted `A4dur` + `Aa` (C.6.3).
- **Silent-OFF trace-name mismatch** (`avl_events_syn_20` resolved 0 traces ⇒ gate appeared on but did
  nothing): fixed by name normalization in `trace.py`. Watch for this whenever a new trace is added.
