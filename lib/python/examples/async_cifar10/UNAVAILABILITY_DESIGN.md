# Sim Unavailability — Design & Staged Plan

## Preamble — what this is, what's done, how to verify (read first)

**Goal.** Model client *unavailability* (devices dropping offline mid-training) in the FLAME FL
simulator so a fast **simulated** run (virtual clock, no real sleeps) reproduces what a **real** run
(wall-clock, MQTT, true delays) does — **sim/real parity** — for every baseline, with the feature
**config-gated and default-OFF** (byte-identical to today when off).

**What was built (v1).** A shared availability substrate (`flame/availability/trace.py` +
`AvailabilityMixin`) mixed into the syncfl base and inherited by asyncfl, so all baselines share one
trace-read effect path:
- **Send-time gate, deliver-late-stale.** A trainer that goes UN_AVL mid-flight *keeps computing*; its
  upload is gated at send-time (real) / buffered to `delivery_ts = max(sct, next_avail)` (sim) and
  committed later as a stale update. Nothing is cancelled or dropped.
- **Two ledgers, never conflated.** Slot ledger (90 s vclock *abandon* frees the in-flight slot) +
  delivery ledger (`pending_withheld[end]=delivery_ts`, commits through the existing staleness gate).
- **Proactive in-flight eviction** — **felix only** (the one fully-aware baseline): frees a slot the
  trace shows UN_AVL at the next selection boundary, no 90 s wait.
- **Starvation / vclock-advance under scarcity.** When the eligible pool is too small to start a round,
  sim advances the vclock to the next availability transition instead of spinning. **⚠ BUG B2.0.2 open
  — see Status.**
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

---

## Baseline matrix (CANONICAL — supersedes scattered categorization)

| baseline | sync/async | agg base / entry | knowledge @ selection (`avail_select_filter`) | in-flight slot-free (`proactive_inflight_evict`) | config-gate |
|---|---|---|---|---|---|
| **felix** | **async** | `asyncfl` (← syncfl) / `main_asyncfl_agg.py` | ✅ aware | ✅ **proactive** (felix only) | `simUnavailability` |
| **fedbuff** | **async** | `asyncfl` / `main_asyncfl_agg.py` | ❌ unaware | ❌ reactive-90s | `simUnavailability` |
| **oort** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | ❌ unaware | ❌ reactive-90s | legacy `trackTrainerAvail` |
| **oort_star** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | ✅ aware | ❌ reactive-90s | legacy `trackTrainerAvail` |
| **refl** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | ✅ aware | ❌ reactive-90s | `simUnavailability` |
| **feddance** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | ✅ aware | ❌ reactive-90s | `simUnavailability` |

**Notes.** (1) `AvailabilityMixin` lives at `syncfl/top_aggregator.py` (`class TopAggregator(AvailabilityMixin, Role)`);
asyncfl extends it (`class TopAggregator(SyncTopAgg)`); oort's base also carries it — so all six get the same
substrate. (2) "aware using trace" is v1; the knowledge model becomes message-transport / predictive in Stage H,
but the select-filter / in-flight-evict *behavior* is unchanged. (3) **felix is the only baseline that de-selects
an in-flight trainer** when it goes UN_AVL; the four aware-at-selection-only baselines (oort_star/refl/feddance and
— at selection — nobody for unaware oort/fedbuff) still hit the 90 s abandon for mid-round drop-offs.

### Flag redesign (replaces the conflated `_availability_aware`) — task T1

Today a single `_availability_aware` HP gates `_sim_evict_unavail_inflight`. Split into two unambiguous flags:
- `avail_select_filter: bool` — selector excludes currently-UN_AVL trainers from the **selection** pool
  (`get_curr_task_ineligible_trainers`). ON: felix/oort_star/refl/feddance. OFF: oort/fedbuff.
- `proactive_inflight_evict: bool` — gates `_sim_evict_unavail_inflight` (in-flight boundary eviction).
  ON: **felix only**. OFF: everyone else (reactive-90s).

`tracking_mode` becomes the **knowledge-model** axis: `trace_read` (v1) | `client_notify` (Stage H) |
`predictive` (future). Replaces the `oracular` value at concept/log level (YAML field *value* compat kept).

---

## Status (Jun 30 — Batch 2. ⚠ NEW BLOCKING BUG B2.0.2 from feddance n=18. Recategorization + new baselines + local test suite scoped as T1–T5 below.)

**A/B/C/C.6/D/E ✅ CONFIRMED syn_20. F.2 CODE-COMPLETE but starvation self-termination BROKEN (B2.0.2).
syn_0 ✅. oort n=25 syn_50 ✅ (starvation fires, K1/K3a PASS). B2.0.1 real recv-barrier ✅ FIXED + confirmed.
Tests green: 456 pass / 7 skip + ladder 24/24.**

### ⚠ B2.0.2 — Sim starvation does not self-terminate at the trace end-horizon (BLOCKING; found feddance n=18 syn_50, Jun 29)

**Symptom.** sim ran **3443 `[SIM_STARVATION]` spins** (rounds 68→3511), every line `vclock→1800.0`, and was
killed by `[SIM_WALL_CEILING]` at 30 min wall instead of self-stopping. Real also lacked `"stopping run"`
(trained 136 rounds, died ~23 min). The `[SIM_WALL_CEILING] ... Sim slower than real (bug iii-c)` message is a
**misdiagnosis** — cause is the spin, not per-round slowness.

**Root cause.** Under perpetual scarcity at the **trace end-horizon**, `_next_avail_vclock()` returns
`1800.0` (= trace horizon = budget). Once vclock is already 1800, `_vclock.advance(1800)` is a no-op and the
guard `_nxt > vclock.now` is false → the F.2 path **returns to spin without advancing**. `increment_round`
runs each spin but its budget stop uses strict `>`, and vclock is pinned **exactly at** budget → `1800 > 1800`
is false → never stops. Empty 0-duration rounds accumulate nothing → ~3400 spins until the wall guillotine.

**Fix (scoped — T0, do FIRST, before any long run; F.2 path is shared syncfl/oort/asyncfl).**
1. In the F.2 starvation branch: if `_next_avail_vclock()` returns `None`/≤ vclock.now **or** vclock ≥ budget,
   **stop the run** (set the budget-stop flag / fall through to `increment_round`'s terminal path) instead of
   returning to spin.
2. Make the budget check `>=` (vclock == budget must stop).
3. Real branch: the `else: time.sleep(0.5); return` path also bypasses the budget check under perpetual
   scarcity — ensure real consults the wall budget before sleeping.
4. Regression: add a deterministic pytest (trace that exhausts mid-scarcity) asserting both modes hit
   `"stopping run"`, NOT the wall ceiling.

**Exit:** feddance perpetual-scarcity run self-stops at `max_experiment_runtime_s` in both modes; no
`[SIM_WALL_CEILING]`; spin count bounded.

### Scenario note (secondary): syn_50 caps at ~43 % unavail, sync starvation is finicky

syn_50's connected-cohort unavail peaks at **129/300 ≈ 43 %** (not 50 %). For feddance (`agg_goal=10`),
`eligible ≈ (1−unavail_frac)·n`: n=25 → min 11 (never starves); n=20 → min 11 (never); n=18 → eligible hits 0
(perpetual). The straddle window is narrow. Starvation is **already proven on oort** (n=25, 1 event) and
unit-tested (G.1) — feddance live starvation is a *validation nicety, not a code gate*. After B2.0.2, one
n=19 try; if it won't cleanly straddle, **deprioritize** (sync FL rarely starves by construction — Challenge 15).

### Completed stages (mechanism + where it lives + exit)
- **A/B ✅** Substrate + A3 time-base CONTROL (origin `agg_start` both modes). A3 PASS oort/felix syn_20.
- **C ✅** Send-time gate, vclock 90 s abandon, `delivery_ts` ordering, `free_stalled_slot` (`AvailabilityMixin`).
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

### T0 — Fix B2.0.2 starvation self-termination (BLOCKING)
- **Scope:** see B2.0.2 above (4 sub-items).
- **Files:** `flame/mode/horizontal/{syncfl,oort,asyncfl}/top_aggregator.py` (F.2 starvation branch);
  `flame/mode/horizontal/*/top_aggregator.py` `increment_round` budget check (`>` → `>=`);
  new `tests/.../test_starvation_termination.py`.
- **Exit:** deterministic pytest green; feddance perpetual-scarcity smoke self-stops both modes (no wall ceiling).

### T1 — G.4 rename + two-axis flag redesign (no-op refactor + flag split; land early)
- **Renames:** `oracular_trainer_avail_check` → `_trace_read_avail_check` (5 refs:
  `asyncfl/top_aggregator.py`, `syncfl/fwdllm_aggregator.py`, `availability/trace.py`, callers); log tag
  `[ORACULAR]` → `[TRACE_READ]` (`availability_mixin.py:266`); comments/docstrings "oracular" → "trace-read".
  **Keep** YAML field *values* `trackTrainerAvail.type: ORACULAR` and `tracking_mode` for config compat.
- **Flag split:** replace `_availability_aware` with `avail_select_filter` (gates
  `get_curr_task_ineligible_trainers` selection filtering) + `proactive_inflight_evict` (gates
  `_sim_evict_unavail_inflight`). Wire `__init__`/`_init_availability` to read both HPs.
- **tracking_mode enum:** `trace_read | client_notify | predictive` (knowledge-model axis); document.
- **Files:** `flame/availability/{availability_mixin.py,trace.py}`, the three `top_aggregator.py`,
  `syncfl/fwdllm_aggregator.py`; metadata template `_metadata/aggregator_base.json`.
- **Exit:** unit tests green; syn_0 byte-identity (gate OFF) preserved; `[TRACE_READ]` in logs.

### T2 — Baseline categorization fixes (docs + yaml + code) to match the Baseline matrix
- Set per-baseline flags in `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml`:
  felix `avail_select_filter+proactive_inflight_evict`; oort none; refl/feddance `avail_select_filter` only.
- Verify selector→base mapping (felix→async_oort/asyncfl, oort→oort/sync, refl→refl_oort/sync,
  feddance→feddance/sync). Fix any doc/comment that still says "oort = async" or "feddance/refl = unaware".
- **Exit:** matrix reflected in config + code comments; smoke each baseline gate-ON syn_20 (no behavior regression).

### T3 — Scaffold new baselines: oort_star (aware sync) + fedbuff (unaware async)
- **oort_star:** oort-sync base + `avail_select_filter=True`, `proactive_inflight_evict=False`. New parity-yaml
  entries (sim+real) + config. Selector = oort with select-filter enabled.
- **fedbuff:** asyncfl base + `FedBuffSelector` (`flame/selector/fedbuff.py` exists) + both flags OFF (unaware).
  New parity-yaml entries + config + `main_asyncfl_agg.py` wiring if needed.
- **Exit:** both run gate-OFF (byte-identical regression) and gate-ON syn_20 smoke; appear in the parity CLI batch.

### T4 — Local state-fidelity test suite (catch bugs before runs) — point 7
New `scripts/parity/test_state_fidelity.py` (deterministic, no cluster), driving a tiny synthetic trace through
sim and a mocked-real path, asserting identical state timelines:
- **T-state-exact:** scripted AVL_TRAIN→UN_AVL→AVL_TRAIN; `state_at(trace,t)` and stamped `PROP_AVL_STATE` match
  the trace at every selection boundary, both modes.
- **T-eval-pool (3-state):** AVL_EVAL trainer excluded from train dispatch, included in eval, both modes (closes
  the AVL_EVAL coverage gap before mobiperf).
- **T-withhold-deliver:** mid-flight UN_AVL → withheld→delivered-stale (not dropped), identical `delivery_ts`.
- **T-aware-vs-reactive:** same trace, proactive-evict ON (felix) vs OFF → proactive frees the slot one boundary
  earlier; quantify the *expected* divergence.
- **T-starvation-sync:** scripted scarcity → exactly one vclock advance to the right transition; K1 monotone;
  self-terminates (regression for B2.0.2).
- **New rung A5 `state_timeline_agreement`:** per-trainer exact-state agreement (binned time, KS=0) — closes the
  "no per-(trainer,t) exact agreement" gap (A3/A4 only aggregate).
- **Run T4 across all six baselines** on synthetic + mobiperf traces.
- **Exit:** suite green for all six; A5 wired into `checks.py`/`report.py`.

### T5 — Parity-table campaign (syn_0 → syn_20 → syn_50 → mobiperf, sim+real, all six baselines)
Produce a `PARITY.md`-style table (rows = baselines, cols = traces × modes, cells = pass/tot + ROOT). Order:
1. **syn_0** regression (byte-identity gate ON/OFF) — all six.
2. **syn_20** then **syn_50** n=300, 3 h, sim+real — all six. (Existing small-n runs are starvation smokes, not
   parity runs; re-run at n=300.) Expect §7 join-ramp artifacts (A3/A2) to clear at n=300.
3. **mobiperf:** `mobiperf_2st` → `mobiperf_3st_50`/`_3st_75` (3-state → AVL_EVAL + D.2 eval-pool + Challenge 13
   live exercise). Calibrate HELD rungs (`observation_lag`, `Aa` eligible_pool_reduction) once real data exists.
4. **§7 sweep:** resolve/re-classify each row with long-run data (oort K3b/A2/P3; feddance A3/A2 expect cleared;
   U5 ρ watch; C2 loss noise).
- **Exit / sign-off:** per-baseline parity green at syn_50 + mobiperf n=300 (A3 gate open, A2/A4/A4dur/A5 PASS);
  `starvation_advance` populated where the scenario admits it (or deprioritized per Challenge 15); all §7 rows
  resolved; HELD rungs calibrated.

### Stage H (FUTURE — out of scope)
Message-transport (`client_notify` ON for aware) + continuous/event-scheduled vclock clamp + predictive
knowledge model. Re-measure `observation_lag` (must be ≈0). Effect logic unchanged (C.5/D.1 hook built for it).

---

## Parity rungs (availability tier)

- **A1** `avail_composition` — per-state counts, binned. **A3** `trace_time_base_consistency` — CONTROL hard
  gate (dep K3). **A4** `per_trainer_duty_cycle`. **A4dur** duration-weighted TVD (pass `mean_err≤0.05`,
  `frac_within_tol(τ=0.10)≥0.95`). **A5** `state_timeline_agreement` (NEW, T4) — per-(trainer,t) exact match.
- **withheld_delivery** — dist of `delivery_ts − sct` + staleness + accept/reject split.
- **abandon_timeout** — count/timing of 90 s vclock abandons; fails loud on wall-clock leak.
- **eligible_pool_reduction** (`Aa`, HELD), **observation_lag** (HELD) — calibrate at T5.
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
- **Config-gated, default OFF** → byte-identical. `simUnavailability` (felix/fedbuff/refl/feddance) or legacy
  `trackTrainerAvail` (oort/oort_star).
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
12. ✅ Library mixin spans examples — `AvailabilityMixin`+`trace.py` in `flame/`; never example-local copy.
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
17. ⚠️ **Sim starvation self-termination (B2.0.2)** — perpetual scarcity at the trace end-horizon pins vclock at
    budget; strict-`>` budget check never trips → spin until wall ceiling. Fix in T0.

---

## 6. Dead-ends (settled — do not retry)

- **busy → UN_AVL routing** — three distinct states.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch ts) — stuck UN_AVL forever; read `_vclock.now`.
- **Wall-clock in sim** for selection gate / 90 s abandon — wall barely advances vs vclock.
- **Per-tick MQTT broadcast** — comms storm; v1 = trace-read pull (zero comms).
- **Ordering withheld commits by `sct`** — past-dating; order by `(delivery_ts, end_id)`.
- **Forking withhold/abandon per stack** — single shared `AvailabilityMixin`.
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
| K3b `overhead_residual` | oort | rel≈0.116 | Run-length sensitive; P3 gates at n=300. Investigate T5. |
| A2 `eligibility` KS | oort | 0.437→0.338 (1.5h→3h) | Bimodal-vs-smooth shape artifact; means match; improving. |
| P3 `trainer_speed` | oort | ratio=1.153 (tol 1.15) | Marginal tail at n=300; gates K3b. Investigate T5. |
| C2 `loss` | feddance | avg_diff≈0.16 (few eval pts) | Early-training noise at α=0.1; K8/C1/utility PASS. |
| U5 `inter-arrival` ρ | feddance | 0.659→0.381 (syn_20→50) | Watch at mobiperf. |
| A4dur | feddance | ✅ RESOLVED — PASS | F.2 stamps pre-selection; n=25 pair A4dur PASS. |
| A3 `avail_timebase` | feddance | 0.245 (n=25), 0.315 (n=20) | Small-n join-ramp artifact (divergence only in first 1–2 deciles, then ~0); means match. Expect clear at n=300. |
| A2 `eligibility` KS | feddance | 0.509 (n=25), 0.286 (n=20) | Same join-ramp + gated by A3; real ramps faster, sim applies unavailability correctly. Clears at n=300. |
