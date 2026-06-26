# Sim Unavailability — Design & Staged Plan

**Status:** **Stage A COMPLETE** (Jun 25) — substrate implemented, 389/389 unit tests pass.
Smoke test (syn_0 90-min all-baseline byte-identity) pending on training node. v1 scope **locked**
(Jun 25) — *oracular trace-read for ALL baselines, `client_notify` deferred*. Same feature templates
into fwdllm ([simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) §7) — the substrate is built
**library-level so it spans examples** (async_cifar10, fwdllm), not bolted onto one example.

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

One mechanism per stage, each gated by its own tests + a syn_0 byte-identity regression + (where it
changes dynamics) a short syn_20 run. All config-gated, default OFF. **v1 builds the unaware-shaped
oracular path for ALL baselines** (oort/refl first, then felix/feddance behaving identically modulo
the dormant eviction hook); the proactive aware eviction + continuous scheduling are **Stage H**.
Context-free names (`_ts`/`_time_s`, `_round`).

### Stage A — Substrate: one trace, one resolver, one clock, one library mixin ✅ COMPLETE (Jun 25)
See §8 for the file-level spec. Summary:
- **A.1** ✅ `flame/availability/trace.py`: `load_trace`, `state_at`, `next_avail_after` implemented.
  Single `bisect_right` resolver replaces the three inlined copies; lru_cache for YAML files.
- **A.2** ✅ `flame/availability/availability_mixin.py`: `AvailabilityMixin` consolidates the three dup
  `read_trainer_unavailability` copies (deleted from `fwdllm_aggregator.py`, `main_oort_sync_agg.py`,
  `main_asyncfl_agg.py`); `get_curr_unavail_trainers` (deleted from `syncfl/top_aggregator.py` body and
  `main_oort_sync_agg.py` wall-clock override); `_avail_now()` uses vclock in sim; `free_stalled_slot`
  dormant hook built. Mixed into `syncfl/top_aggregator.py` → all four stacks inherit automatically.
- **A.3** Config surface: `sim_unavailability`, `availability_aware`, `availability_trace_dir` added to
  `flame/config.py`. Gate-off default ⇒ `trainer_event_dict=None` ⇒ byte-identical.
- **A.4** `_init_availability(config)` called from `syncfl/TopAggregator.internal_init()`; supports both
  new `sim_unavailability` gate and legacy `track_trainer_avail["enabled"]` path.
- **Unit tests:** 389/389 pass. **Smoke test:** syn_0 90-min all-baseline byte-identity — **PENDING
  on training node** (submit with `scripts/run_parity.sh --trace syn_0`).
- **Exit:** syn_0 90-min all-baseline parity holds the scoreboard byte-for-byte → then proceed to Stage B.

### Stage B — A3 time-base CONTROL (gate for everything above it)
- **B.1** A3 `trace_time_base_consistency` (CONTROL/DIST, dep K3): resolved on/off windows align
  between modes within tolerance, origin = `agg_start`. **B.2** A4 `per_trainer_duty_cycle` (dep A3).
- **Tests + a 5-min syn_20 smoke** (checker-side, validates instantly vs stored dirs).
  **Exit:** A3 PASS on a syn_20 smoke for oort.

### Stage C — ORACULAR driver + send-time delivery gate + vclock abandon (oort, refl; then ALL)
- **C.1** Activate `get_curr_unavail_trainers()` via the Stage-A resolver on `_vclock.now` →
  `channel.set_curr_unavailable_trainers`. **Gates new selection only.**
- **C.2 Send-time withhold-deliver.** An in-flight trainer entering `UN_AVL` is **not** interrupted;
  its modeled update is **held and delivered at `delivery_ts = max(sct, next_avail_ts)`**, committing
  **stale**. Real-side: add the trainer **send gate** (block upload until `AVL_*`) — documented real
  change. Held end excluded from the pool until `delivery_ts` (extend §4.5 `pending_after`:
  `vclock < delivery_ts`, not `< sct`).
- **C.3 Vclock abandon (slot ledger).** Re-clock `SEND_TIMEOUT_WAIT_S`/`RECV_TIMEOUT_WAIT_S` from
  `time.time()` to `_vclock.now`. At the 90s vclock deadline, **free the stalled trainer from
  in-flight → replacement selectable** (existing abandon path). Keep the **delivery ledger** separate
  (`pending_withheld[end] = delivery_ts`); the late update still commits and is accept/reject-gated by
  the baseline's existing staleness rule.
- **C.4 Ordering.** drain/commit must key on **`delivery_ts`** for held ends (`delivery_ts > sct`),
  or past-dating reappears (Challenge 1).
- **C.5 Eviction hook (dormant).** Build the slot-free effect behind an abstraction callable by either
  the boundary read (v1) or a future `avl_*` message (Stage H). Aware baselines free at the boundary;
  unaware at the 90s abandon — single path, `availability_aware` flag.
- **Tests:** compute completes but no send while `UN_AVL`; withheld delivers at `max(sct,next_avail)`,
  commits stale; accept-stale (async) vs reject-over-tolerance (sync) honored; held end excluded until
  `delivery_ts`; 90s abandon on the **vclock** frees the slot + replacement selectable; no
  double-count / in-flight never negative (invariant 1); still-down trainer never re-selected
  (invariant 2); determinism.
- **Validation:** syn_20, 45-min, oort+refl first, then a felix/feddance smoke confirming they ride
  the same path. Read A1, A2 (eligible-pool reduction), A4, withheld-delay dist + staleness (U3),
  abandon_timeout. **Exit:** A1/A2/A3/A4 PASS, no new past-dating (U6), K2/K3b hold vs a syn_20 real
  reference.

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

- **Real send-gate confirmation (Challenge 5):** on a real syn_20 run, verify the trainer
  withholds-then-delivers (stale) rather than dropping the update when it goes `UN_AVL` at send time.
  Decides whether the `max(sct,next_avail)` model is faithful before C.2 is signed off.
- **Q-new-2 sync confirmation:** verify feddance's existing staleness threshold is the right rejection
  gate on a real syn_20 run before E.2 (don't assume the async tolerance transfers).
- **[Stage H] Real notification lag (Challenge 5):** measure on a real felix run before turning
  `client_notify` back on; decides whether lag-0 reflection is admissible or lag must be modeled.
- *(Append new open questions/info needs here as the substrate lands — keep this the single ledger.)*

---

## 8. v1 Implementation Spec (Stage A + C) — file-level

Concrete enough to start in a fresh context. All paths relative to `lib/python/`.

### 8.1 New library module: `flame/availability/trace.py`
- `TrainerAvailState` enum already exists (`trainer/pytorch/main.py` imports it) — import/relocate to
  `flame/availability/` so both sides share one definition.
- `load_trace(trace_name: str, trainer_id: str, *, base_dir: str | None = None) ->
  SortedDict[float, str]` — loads the per-trainer `ts→state` event series from
  `base_dir or examples/_metadata/availability_traces/<trace_name>/...`. `base_dir` comes from config,
  never hardcoded. `syn_0` ⇒ empty/`AVL_TRAIN`-only series (inert).
- `state_at(trace: SortedDict, t: float) -> TrainerAvailState` — `bisect_right(t) - 1`, the single
  copy of the search inlined today in `get_curr_unavail_trainers` (`main_oort_sync_agg.py:311`),
  `oracular_trainer_avail_check` (`asyncfl/top_aggregator.py:1226`), and trainer
  `check_and_update_state_avl` (`trainer/pytorch/main.py:336`).
- `next_avail_after(trace, t) -> float` — next `ts` whose state ∈ {AVL_TRAIN, AVL_EVAL} at/after `t`;
  feeds `delivery_ts`. Returns +inf only if the trace never recovers (caller guards, Challenge 10).

### 8.2 New library mixin: `flame/availability/availability_mixin.py`
- `class AvailabilityMixin:` providing, against `self`:
  - `_init_availability(config)` — reads config (8.4), sets `self.trainer_event_dict` (None when
    `sim_unavailability` off ⇒ all current behavior unchanged), `self.availability_aware`,
    `self.availability_trace`, `self._avail_base_dir`. Call from each aggregator `__init__`.
  - `_avail_now() -> float` — `self._vclock.now` (sim) / `time.time() - self.agg_start_time_ts`
    (real). Single time source; never wall in sim.
  - `read_trainer_unavailability(trace=None)` — consolidates the **three** dup copies
    (`fwdllm_aggregator.py:481`, `main_oort_sync_agg.py:173`, `main_asyncfl_agg.py:158`); delete those.
  - `get_curr_unavail_trainers() -> list[str]` — `state_at(... , self._avail_now())` per trainer;
    replaces the example copy at `main_oort_sync_agg.py:298`. Library callers
    (`oort/top_aggregator.py:643,688`) keep working.
  - `free_stalled_slot(channel, end, *, reason)` — **the dormant eviction hook.** Frees `end` from
    `selected_ends`/in-flight AND registers `self.pending_withheld[end] = delivery_ts`. Called by
    (a) the 90s vclock abandon for everyone (C.3), and (b) the boundary eviction for
    `availability_aware` baselines (D.1). One effect path for both triggers and for the future
    `avl_*` message (Stage H).
  - `pending_withheld: dict[str, float]` and the commit/order helper keyed on `delivery_ts` (C.4).
- **Mix into all four** library `TopAggregator`s: `flame/mode/horizontal/oort/top_aggregator.py:59`,
  `asyncfl/top_aggregator.py:79`, `syncfl/top_aggregator.py:101`, `syncfl/fwdllm_aggregator.py`.
  Example aggregators (`PyTorchCifar10Aggregator(OracleInjectMixin, TopAggregator)`) inherit for free;
  fwdllm's `FedSGDAggregator(TopAggregator)` inherits via its library base — both examples covered.

### 8.3 Trainer send-gate: `trainer/pytorch/main.py`
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

### 8.5 Telemetry (on the vclock)
`avail_change` (move off wall), `agg_observed_state` (per-trainer belief at each selection),
`abandon_timeout` (vclock 90s frees), `withheld_delivery` (`delivery_ts−sct`, staleness,
accept/reject), trace granularity per run. These back rungs A1/A4/transition_effect/withheld_delivery/
abandon_timeout/observation_lag.

### 8.6 Tests (Stage A + C exit gate)
- Resolver: `state_at`/`next_avail_after` determinism; parity with the old inlined searches on a fixed
  trace; `syn_0` ⇒ inert.
- Mixin: `get_curr_unavail_trainers` identical across all four aggregators on a shared fixture;
  frozen-clock deadlock cannot recur.
- Send-gate: compute completes but no send while `UN_AVL`; withheld delivers at `max(sct,next_avail)`,
  commits **stale**; accept-stale (async) vs reject-over-tolerance (sync).
- Two ledgers (the three invariants): 90s **vclock** abandon frees the slot + replacement selectable;
  in-flight never negative, no double-count; held end excluded until `delivery_ts`; still-`UN_AVL`
  trainer never re-selected.
- Commit ordering keyed on `delivery_ts` (no past-dating).
- **Config-gating:** `sim_unavailability=False` ⇒ byte-identical (the regression guard).

### 8.7 Exit criteria
Stage A: syn_0 90-min all-baseline parity holds the scoreboard byte-for-byte.
Stage B: A3 PASS on a syn_20 smoke (oort).
Stage C: A1/A2/A3/A4 PASS on syn_20 45-min (oort+refl), withheld/abandon rungs populated, no new
past-dating (U6), K2/K3b hold vs a syn_20 real reference; felix/feddance smoke confirms they ride the
same oracular path (dormant eviction). Then proceed to Stage D.
