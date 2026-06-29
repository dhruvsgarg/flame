# Sim Unavailability — Design & Staged Plan

## Preamble — what this is, what's done, how to verify (read first)

**Goal.** Model client *unavailability* (devices dropping offline mid-training) in the FLAME FL
simulator such that a fast **simulated** run (virtual clock, no real sleeps) reproduces what a
**real** run (wall-clock, MQTT, true delays) does — i.e. **sim/real parity** — for every baseline,
with the feature **config-gated and default-OFF** (byte-identical to today when off).

**What was built (v1).** A shared availability substrate driven by a per-trainer **trace** the
aggregator reads (`trainer_event_dict`), mixed into all four `TopAggregator`s via `AvailabilityMixin`:
- **Send-time gate, deliver-late-stale.** A trainer mid-flight that goes unavailable *keeps computing*;
  its upload is gated at send-time (real) / buffered to `delivery_ts = max(sct, next_avail)` (sim) and
  committed later as a stale update. Nothing is cancelled or dropped.
- **Two ledgers, never conflated.** Slot ledger (90 s vclock *abandon* frees the in-flight slot) +
  delivery ledger (`pending_withheld[end]=delivery_ts`, commits through the existing staleness gate).
- **Proactive eviction** for the aware baseline (felix): free a slot the trace shows UN_AVL at the next
  selection boundary, no 90 s wait.
- **Starvation / vclock-advance under scarcity.** When the eligible pool is too small to start a round,
  sim advances the vclock to the next availability transition instead of spinning.
- **Parity ladder** (`scripts/parity/`) — availability rungs A1/A3/A4/A4dur, withheld_delivery,
  abandon_timeout, starvation_advance, eligible_pool_reduction — on top of the existing clock/selector rungs.

**Baselines covered.** `felix` (aware, async), `oort` (unaware, async), `feddance` + `refl`
(unaware, **sync**). Async vs sync differ in *slot-free timing* only; the knowledge model is trace-read
for all (v1 keeps `client_notify` OFF — message-transport is the future Stage H).

**Hardest parts (where the bodies are buried).**
1. **A3 time-base CONTROL** — sim vclock and real wall must share one origin (`agg_start`); every other
   availability rung is meaningless until A3 passes. It is a *hard gate*.
2. **Two-ledger discipline** — ordering withheld commits by `delivery_ts`, never `sct` (past-dating bug).
3. **Empty per-task pool corrupts shared `selected_ends`** (Challenge 13) — a silent hang; the cleanup
   must key off the *connected* pool, not the availability-filtered one.
4. **Sync vs async starvation threshold** (Challenge 15) — async subtracts in-flight, sync does not; the
   same scenario starves differently. Scenario design (cohort size vs unavailability %) is itself a trap:
   too small ⇒ *perpetual* starvation (degenerate), too large ⇒ never starves.
5. **Sim/real symmetry of budget consumption under scarcity** — *currently broken in real mode*, see Status.

**What we run, and how to verify correctness (real AND sim).**
1. **Regression first:** `syn_0` (always-available) must be **byte-identical** with the gate ON vs OFF.
2. **Unit tests:** `cd lib/python && conda run -n dg_flame python -m pytest tests/availability tests/selector`
   (226) + `examples/async_cifar10 && pytest scripts/parity/test_ladder.py` (24).
3. **Smoke a baseline in BOTH modes:**
   `scripts/debug_run.sh --baselines <b> --mode both --runtime-s 1800 --trace <syn_20|syn_50> --num-trainers <n>`.
4. **Parity check:** `python -m scripts.parity.cli --batch --experiments-dir experiments --baselines <b> --agg-goal <g>`.
   The report prints a **ROOT-CAUSE** (lowest broken rung with passing upstreams). Read A3 first; if A3
   FAILs, ignore A1/A2/A4 (they are gated). Mechanism rungs: A1 (composition), A2 (eligibility), A4/A4dur
   (duty-cycle), withheld_delivery, abandon_timeout, starvation_advance.
5. **Per-mode sanity (independent of parity):**
   - *sim* agg log: `[SIM_STARVATION]` only under genuine scarcity, `[VCLOCK_PROGRESS]` advancing, and the
     run **self-stops** at `max_experiment_runtime_s` (`"stopping run"`).
   - *real* agg log: withheld-then-delivered (not dropped), `accept_frac` sane, completes **within** the
     wall budget. **A real run that exceeds the budget or completes 0 rounds is a hang — see Status #1.**
6. **Training-performance / accuracy** is read from the C-rungs (C1 accuracy, C2 loss) + the accuracy/loss
   curves emitted by `analyze_run.py` (see `PLOTTING.md`). Parity must hold *before* accuracy numbers are
   trusted — a diverging clock or eligibility makes the accuracy curve meaningless.

---

## Working agreement (standing instructions — read every session)

**Implement breadth-first across stages on ONE baseline with short runs; batch the long parity runs at
the end. Never block forward implementation on a long run.**

1. **Common first, one baseline first.** Land shared/library-level changes once, drive through a single
   reference baseline — oort/refl for async/unaware, felix only for aware-specific. Don't fan out until it
   behaves on the reference.
2. **Short runs to debug, long runs to confirm.** Gate forward work on unit tests + syn_0 byte-identity +
   shortest syn_20 smoke (~1800s vclock). Long runs confirm; never find first bugs.
3. **Across stages before across baselines, long runs last.** Stack mechanisms across stages on the reference
   baseline, then widen to other baselines, then batch longer parity runs. One long run confirms several stages.
4. **Keep this doc crisp.** Completed stages: mechanism + where it lives + exit met (2–3 lines max). Full
   detail only for active and next stages. Dead-ends in §9.

---

## Status (Jun 29 — Batch 2 in progress. ✅ Real-mode recv-barrier hang FIXED; feddance n≈20 syn_50 ready to run)

**A/B/C/C.6/D ✅ CONFIRMED syn_20. E ✅ CONFIRMED syn_20. F.2 ✅ CODE-COMPLETE. syn_0 ✅.
oort n=25 syn_50 1800s ✅ CONFIRMED (starvation fires). Batch 2 B2.0 ✅ (cohort-floor guardrail + Challenge 13 root-fix shipped; the 3 scoped pre-ramp fixes were all non-issues — see B2.0).**

**✅ B2.0.1 FIXED (found + fixed Jun 29; was blocking ALL real-mode parity at syn_50+ small-n):**
**The real-mode aggregate recv barrier had no timeout.** `syncfl/_aggregate_weights` called
`channel.recv_fifo(ends, first_k=agg_goal)` with no `timeout`; under unavailability the withheld trainers
never send, so fewer than `first_k` arrive and the barrier blocked forever (real feddance n=12:
15:40:55→16:27:20 ≈ 46 min on a 30 min budget, 0 completed rounds, killed only by the runner watchdog).
**Not the scarcity poll** — pure scarcity self-terminates via `increment_round` (the **sim** n=12 run proved
this, ending cleanly at round 4). oort/asyncfl already bounded their real recv; only the syncfl base lacked
it. Fix: `timeout = min(trainer_recv_wall_timeout_s=90 s, remaining budget)`, mirroring oort. See B2.0.1.

**Secondary (scenario, not a bug): n=12 syn_50 is DEGENERATE for feddance.** The doc's "~3 unavailable"
math was wrong — syn_50 ⇒ ~50 % unavailable ⇒ eligible ≈ 6 ≪ threshold 10 ⇒ *perpetual* starvation, no
training. sim n=12 confirmed this (3 starvations → vclock 1200→2400 → budget-stopped at round 4, ~0
training). Feddance starvation needs the pool to **oscillate around** the threshold: target **n≈20 syn_50**
(mean eligible ≈10, straddles) or n≈22–24 for mostly-training-with-occasional-starvation. See Challenge 15.

- **A/B ✅** Substrate (`flame/availability/trace.py` + `AvailabilityMixin`) + A3 time-base CONTROL. Exit: A3 PASS oort/felix syn_20.
- **C ✅ CONFIRMED** Oracular gate, send-time withhold, vclock 90s abandon, `delivery_ts` ordering, `free_stalled_slot`. felix 49/49; oort 39/48 (3 pre-existing failures, see §7).
- **C.6 ✅ CONFIRMED** `_avail_stamp_end_states`, A4dur PASS (felix 0.0024, oort 0.0029), 5 availability plots.
- **D ✅ CONFIRMED** Proactive eviction (felix), task-aware eligibility with `_trace_has_avl_eval` guard, accept-stale withheld. Real send-gate confirmed (withheld n=7, accept_frac=1.0).
- **E ✅ CONFIRMED syn_20** Syncfl path (feddance+refl): abandon/evict/stamp + `_sync_sim_recv_first_k` withhold drain. feddance 46/47 (C2 emergent noise, not mechanism; A-rungs/U3/U6/K8/U2 PASS).
- **F.2 ✅ CODE-COMPLETE** Unified pre-selection return-early pattern in all three aggregators (see §5/Stage F). **syn_0 ✅**: Fst PASS (no starvation), C1/C2 diff=0.0, K1/K5 PASS. **oort n=25 syn_50 1800s ✅**: Fst PASS (1 starvation advance, jump 74.3s = 5× mean), K1 monotone, K3a PASS, 46/53 (P3/K2/K3 pre-existing §7). **feddance n=25 syn_50**: mechanism fires withhold (n=4, accept_frac=1.0), K1/K2/K3/P3 all PASS, BUT Fst 0 starvation advances — sync threshold gap (see §5/Challenge 15). **feddance n=12 syn_50 ran Jun 29 → degenerate (sim: all-starvation, budget-stop round 4; real: HUNG 46 min, 0 rounds → surfaced B2.0.1). Re-run at n≈20 after B2.0.1.**
- **G.1 ✅** `starvation_advance` rung in `checks.py` + `report.py`; +4 starvation unit tests.

---

## ▶ Next actions

### ✅ B2.0.1 — Real-mode recv-barrier timeout under unavailability (FIXED Jun 29)

**Symptom:** real feddance n=12 syn_50 ran ~46 min on a 30 min budget, then died only to the external
runner watchdog (`budget+1200 s`). The agg log went **silent after 12 s** (one `feddance select`, then
nothing) — a *blocking* wait, not a busy-spin.

**Actual root cause (different from the first hypothesis — corrected here).** The hang was NOT the
pre-selection scarcity poll. Pure scarcity (0 selected) self-terminates fine: `_distribute_weights`
returns early → `_aggregate_weights` hits `if not ends: sleep+return` → `increment_round` runs → budget
fires (this is exactly why the **sim** n=12 run ended cleanly at round 4). The real hang was the
**aggregate recv barrier**: round 1 dispatched 12 trainers, ~6 went UN_AVL and withheld their uploads, and
`syncfl/_aggregate_weights` called `channel.recv_fifo(ends, first_k=agg_goal)` **with no `timeout`** → it
blocked forever waiting for the 10th of 12. Sim avoids this via `_sync_sim_recv_first_k(..., timeout=grace)`.
**oort and asyncfl already passed a real-recv `timeout`** (oort `trainer_recv_wall_timeout_s`, asyncfl
`RECV_TIMEOUT_WAIT_S`) — **only the syncfl base lacked it**, which is why oort completed and feddance hung.

**Fix (`syncfl/top_aggregator.py:_aggregate_weights`, real branch — mirrors oort exactly).** Bound the real
`recv_fifo` with `timeout = min(trainer_recv_wall_timeout_s [default 90 s], remaining experiment budget)`.
90 s ≫ the ~18 s max trainer compute, so live stragglers still land; under unavailability the round
proceeds with whatever arrived after the timeout, then `increment_round`'s budget check runs and the run
self-stops *at* the budget. WALL-CLOCK only; sim path untouched; harmless with the gate OFF (all `first_k`
arrive well within 90 s). The `recv_fifo` timeout terminator `(None, ("", now))` is already handled by the
loop's `if not msg: continue`.

**Verification.** Full suite green except 7 **pre-existing** failures (`test_sync_sim_ordering`,
`test_sim_barrier` — fixtures missing `_sim_buffer`, fail identically on clean HEAD); parity ladder 24/24;
Challenge 13 tests pass. No unit test added for the recv path itself — it is a faithful copy of the
already-shipped oort timeout and would need heavy `recv_fifo`/channel mocking for low marginal value; the
n=20 run is the live confirmation.

**Exit:** real feddance n≈20 syn_50 self-stops at `max_experiment_runtime_s` (`"stopping run"` in the agg
log), not via the watchdog; completes real training rounds with partial cohorts.

**Not done (deferred, lower priority): sim no-vclock-advance edge case.** If `_next_avail_vclock()` ever
returns `None`/≤now while eligible < threshold, the sim scarcity branch would `return` without advancing or
sleeping → tight busy-loop (still bounded by `increment_round`'s vclock budget each iteration, so it
*terminates*, just hot). Not observed (sim always had a future transition). Add a guard if a long sim run
ever pegs a core under scarcity.

### Stage F exit — feddance starvation (scenario CORRECTED to n≈20 syn_50)

**oort n=25 ✅ DONE** (starvation fires, 1 event, K1/K3a PASS). Feddance n=25 did not starve (eligible
min=11 > 10); feddance **n=12 was the wrong correction** — syn_50 ⇒ ~6 unavailable ⇒ eligible ≈ 6 ≪ 10 ⇒
*perpetual* starvation, no training (see Status / Challenge 15). **Relaunch after B2.0.1 lands:**

```bash
cd lib/python/examples/async_cifar10
# Target: eligible oscillates around agg_goal=10 → mix of starvation + training.
scripts/debug_run.sh --baselines feddance --mode both --runtime-s 1800 --trace syn_50 --num-trainers 20
```

**Why n≈20 (not 12, not 25):** feddance threshold = `agg_goal=10`; sync clears `selected_ends` each round
so `eligible = n − unavail`. syn_50 ⇒ unavail ≈ 0.5 n ⇒ eligible ≈ 0.5 n. For eligible to straddle 10
(starve sometimes, train otherwise) ⇒ n ≈ 20. n=25 ⇒ eligible ≈ 12–19 (never starves); n=12 ⇒ eligible
≈ 6 (always starves). If n=20 still skews one way, nudge: more starvation → n=18; more training → n=22–24.

**⚠️ n=12 is never valid here. oort floor is n ≳ 16** (`desired_selection=13`; n<13 starves every round —
the cohort-floor guardrail now clamps + warns, but the run is still degenerate). Pick n per *baseline
threshold ÷ availability*, not a fixed number.

**Previous bug fixes (Jun 29, still apply):** (1) `debug_run.sh` injects `simUnavailability=True` for
feddance (was missing → `trainer_event_dict=None` → vclock deadlock). (2) `runner.py` logs aggregator
exit code + `PYTHONFAULTHANDLER=1`. (3) **Teardown abort** (`Fatal Python error: Aborted` after
`channel leave done`) is cosmetic — runner does not gate success on exit code (B2.0 #2).

**After run completes:**
```bash
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines feddance --agg-goal 10
```

**Stage F exit criteria:** `starvation_advance` populated (n_jumps > 0) for feddance AND ≥1 real training
round completes (not all-starvation); `[SIM_STARVATION]` in sim agg log; K1 monotone; both modes self-stop
at the budget; parity report ROOT-CAUSE is not a starvation/budget rung.

### Batch 2 — Long-run parity campaign + deferred-fix sweep (DETAILED)

**Theme:** This is the "long runs last" phase of the working agreement. The mechanism code (A–G) is complete across all three stacks (asyncfl, syncfl-feddance, syncfl-refl). Batch 2 does NOT add new mechanisms — it (1) lands a handful of cheap deferred code fixes, (2) widens validation to the least-tested baseline (refl), (3) runs the full-cohort (n=300) long parity campaign across syn_50 → mobiperf, and (4) resolves or re-classifies the §7 known failures with real long-run data.

**Entry gate (UPDATED Jun 29):** ✅ **B2.0.1 (real-mode recv-barrier timeout) is FIXED** — real runs now bound the aggregate recv under unavailability, so real/sim parity pairs at syn_50+ small-n are trustworthy. Stage F *code* is confirmed firing on oort (async) + unit-tested for syncfl (G.1); feddance live-confirmation (now at n≈20, not n=12) is a validation nicety, **not a code gate** — refl/oort/felix Batch 2 work does NOT wait on it. **Next concrete step: run feddance n≈20 syn_50 `--mode both` to confirm the fix + starvation mix.**

#### ▷ Decision tree — feddance n=12 syn_50 (resolve when logs arrive)

| Outcome | Signal | Action |
|---|---|---|
| **(A) Pass** | `starvation_advance` populated (n_jumps > 0), K1 monotone, training rounds interleave, no stall/crash | Stage F fully closed on all stacks. Proceed to Batch 2 at full confidence. |
| **(B) Mechanism bug** | starvation fires BUT K1 non-monotone / event skipped / stall / abandon-path crash / teardown-abort masks a real failure | **BLOCKS Batch 2.** syncfl F.2 code is shared with refl — must fix before refl/feddance long runs. Debug with short syn_50 n=12 runs + unit tests; do not burn a 3h run to find the bug. |
| **(C) Won't trigger** | clean run, withhold/abandon fire, but eligible never < 10 (peak simultaneous unavail ≤ 2) | One tuning retry (n=10, or syn_50→a denser trace). If still no trigger: **deprioritize** — accept oort-confirmed + syncfl unit-test (G.1) as sufficient starvation proof, record here, move on. The n=300 mobiperf runs may trigger it naturally; if not, that's acceptable (sync FL rarely starves by construction — Challenge 15). |
| **(D) Known small-n noise only** | A3/A2 fail with join-ramp signature (§7), starvation otherwise fine | Ignore — clears at n=300 (B2.2). Does not affect Stage F exit. |

**Recommendation:** treat Stage F as code-complete now; let feddance n=12 either confirm (A) or be deprioritized (C). Only outcome (B) holds up Batch 2.

#### B2.0 — Pre-ramp code work ✅ DONE (verified Jun 29 against the n=25 syn_50 runs)

The three "cheap fixes" originally scoped here were investigated against live code/telemetry and **all three turned out to be non-issues** — the code is already correct. The only real change was a defensive guardrail. Detail (so this is not re-litigated):

1. **A4dur syncfl — ALREADY PASSES, no fix needed.** The §7 "syncfl stamps post-selection → SKIP" claim was stale: F.2's restructure already stamps `_avail_stamp_end_states` pre-selection (`syncfl/top_aggregator.py:896`, before `channel.ends(VAL_CH_STATE_SEND)` at :957). Confirmed: running the parity CLI on the feddance n=25 syn_50 pair gives **A4dur PASS** (avl_state present 71/71 sim, 67/67 real). K3b also PASS for feddance. §7 rows updated.
2. **Teardown abort — cosmetic, not a bookkeeping bug.** `runner.py:300-302` logs the aggregator exit code as a *warning only* (`exit=N ⚠`) and proceeds to post-analysis + the next batch regardless. Run success is judged from telemetry/log content, never the exit code. The `Fatal Python error: Aborted` is a C++-level torch/MQTT/grpc teardown abort that fires *after* "stopping run" + `channel leave done` — purely shutdown noise. No code change; not worth chasing.
3. **Starvation-budget semantics — current behavior is parity-CORRECT; changing it would BREAK parity.** In real mode the scarcity path is `time.sleep(0.5)` + retry (`syncfl/top_aggregator.py:949-951`), so real consumes *wall* budget while polling through scarcity, and `increment_round` measures `elapsed = time.time() - agg_start`. In sim the starvation vclock jump consumes *virtual* budget symmetrically (`increment_round` measures `elapsed = vclock.now`). They match. Subtracting starvation skips from the sim budget (the originally-"preferred" option) would let sim run more training rounds than real for the same budget → divergence. **Recorded as a dead-end (§6).** The oort n=12 budget exhaustion was 100% the invalid-cohort footgun (#4), not a budget bug.
4. **✅ NEW — oort cohort-floor guardrail (implemented).** `oort/top_aggregator.py`: the starvation gate now clamps its threshold to `min(desired_selection, len(connected))` and emits a one-time `[COHORT_FLOOR]` warning when `desired_selection > connected`. Prevents the n<desired_selection footgun (n=12 with desired_selection=13 starved every round → vclock storm → budget exhausted at round 4, zero training) from silently degenerating; the run now proceeds with the available cohort and the misconfig is loud. No-op when the cohort is adequately sized (the normal case). 24/24 parity-ladder tests pass.

#### B2.1 — refl: first dedicated unavailability validation (widen baselines)

refl shares the syncfl E+F code with feddance but has **no dedicated unavailability run confirmation** (only the parity-fidelity work pre-dates this branch). Before committing refl to a 3h run: refl syn_20 then syn_50 short smokes (sim+real), confirm E (abandon/evict/stamp/withhold-drain) + F (starvation pattern, same `agg_goal` threshold as feddance) + A-rungs. This is the breadth-across-baselines step; it also re-confirms the syncfl path that B2.0 #1 touched.

#### B2.2 — syn_50 n=300 long runs (3h, sim+real), per baseline

felix, oort, refl, feddance. **First true full-cohort unavailability validation.** Note: the existing Jun 29 syn_50 runs are small-n starvation smokes (oort n=12, feddance n=25) — they are NOT the n=300 parity runs and must be re-run at n=300. Expected resolutions:
- **A3/A2 join-ramp artifacts clear** (§7 feddance A3=0.245, A2 KS=0.509 at n=25): at n=300 the first-decile bin no longer dominates. Confirm A3 PASS gates open, then read A2/A4.
- **C.3 90s-vclock abandon exercises** on oort/refl (longer train times cross 90s where syn_20/n=48 did not).
- **oort K3b/A2/P3 re-evaluated at length** (§7): K3b `overhead_residual` ~0.116 and A2 KS were run-length-sensitive (0.437→0.338 across 1.5h→3h). Confirm trend to PASS or root-cause if they plateau.

#### B2.3 — mobiperf ramp (3h, sim+real), per baseline

Order: `mobiperf_2st` (real 2-state schedule — closest to syn) → `mobiperf_3st_50` / `mobiperf_3st_75` (3-state, introduces **AVL_EVAL**). The 3-state traces activate code paths dormant in v1's 2-state syn traces:
- **D.2 eval-pool path** un-guards (`_trace_has_avl_eval` now true) — D.2 logic gets first real exercise.
- **Challenge 13 — root-fix already landed (Jun 29), needs live exercise here.** `_handle_send_state` now keys its cleanup off the full connected pool (`connected_ends`), so an empty per-task eligible pool no longer corrupts shared `selected_ends` (see §5/Challenge 13; +3 unit tests). mobiperf_3st is the first trace that exercises the all-AVL_EVAL → empty-train-pool case live — confirm no hang and correct in-flight retention.
- **HELD rungs calibrate here**: `observation_lag` and `Aa` (eligible_pool_reduction) need real run data to set tolerances — build them once B2.2/B2.3 data exists.

#### B2.4 — Deferred-fix sweep (resolve §7 with long-run data)

After B2.2/B2.3 surface real numbers: close out each §7 row — confirm cleared, root-cause if persistent, or formally re-classify as expected. Targets: oort K3b/A2/P3, feddance A3/A2 (expect cleared), feddance U5 inter-arrival ρ (watch at mobiperf), feddance C2 loss noise.

#### B2.5 — G.4 terminology cleanup (no-op refactor, LAST)

Only after all baselines are green: rename `oracular_trainer_avail_check` → `_trace_read_avail_check`; update log/comment "ORACULAR" references (keep the YAML field *value* `ORACULAR` as-is); consolidate legacy-gate + simUnavail-gate paths. Pure refactor — defer until nothing else is in flight.

**Batch 2 exit / sign-off:** per-baseline parity green at syn_50 + mobiperf n=300 (A3 gate open, A2/A4/A4dur PASS); `starvation_advance` populated where the scenario admits it (or deprioritization recorded per decision tree); all §7 rows resolved or re-classified; Challenge 13 root-fixed; HELD rungs calibrated; doc trimmed to completed-stage form.

### Stage H (FUTURE — out of Batch 2 scope)

- **Message-transport + continuous scheduling** (see §4 Stage H): turn `client_notify` ON for aware baselines; swap trace-read for real `avl_*` messages; re-measure `observation_lag` (must be ≈0). Effect logic unchanged — the C.5/D.1 hook was built for this.
- **Real notification lag**: measure on a real felix run before turning `client_notify` back on.

---

## v1 Scope

### Three axes — keep them separate

| Axis | Term | v1 value | Stage H |
|---|---|---|---|
| How the agg learns trainer state | **knowledge model** | **trace-read** (all baselines — agg binary-searches `trainer_event_dict`) | aware → **message-transport** (`avl_*` msgs) |
| When the agg frees a stalled slot | **slot-free timing** | **proactive** (felix: next selection boundary) OR **reactive-90s** (oort/refl/feddance: 90s vclock) | proactive → instant-on-message for aware |
| Which config knob activates | **config-gate** | **legacy-gate** (oort/refl: `trackTrainerAvail.type: ORACULAR`) OR **simUnavail-gate** (felix/feddance: `simUnavailability: True`) | no change |

**"ORACULAR" in code/YAML = only the legacy config field name.** All v1 baselines are trace-read. Use "trace-read" for the knowledge model.

v1 uses trace-read for every baseline; `client_notify` OFF. The proactive/reactive-90s distinction is slot-free timing only, not knowledge.

| | v1 (this spec) | Stage H (FUTURE) |
|---|---|---|
| Knowledge model | trace-read (all) | aware: message-transport; unaware: trace-read |
| When applied | selection boundaries only | continuous / event-scheduled |
| `client_notify` | OFF | ON for aware |

---

## 1. Core decisions (all resolved)

- **Knowledge model:** trace-read for all baselines. One shared trace + `state_at(trainer, vclock)` + one effect path in `AvailabilityMixin`. Per-baseline difference = slot-free timing only.
- **Mid-flight unavailability = compute-completes, gate the send, deliver-late (stale).** Trainer never stops computing. Upload gated at send-time (real) / agg-side buffer at `delivery_ts = max(sct, next_avail_ts)` (sim).
- **Two ledgers, never conflated:** slot ledger (90s vclock abandon, frees `selected_ends`) + delivery ledger (`pending_withheld[end] = delivery_ts`, commits stale through existing staleness gate).
- **Three invariants:** (1) no double-count — freed slot ≠ cancelled update; (2) withheld end stays out of pool until `delivery_ts`; (3) proactive vs reactive-90s = trigger only, identical downstream effect.
- **Busy ≠ unavailable ≠ withheld** — three distinct non-pool states, separate ledgers. Do NOT route busy→UN_AVL.
- **All availability time on the vclock in sim.** Never wall, never frozen per-trainer clock.
- **Config-gated, default OFF** → byte-identical. Legacy-gate (oort/refl) or simUnavail-gate (felix/feddance/fedbuff).

---

## 2. Concepts to keep crisp

- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) × **busy?** × **has in-flight update?** — orthogonal, never conflate.
- **send-time gate** (v1) vs **task-start gate** (old real behavior). Compute always completes.
- **slot ledger** (freed at boundary/90s) vs **delivery ledger** (commits at `delivery_ts`).
- **transition instant** vs **observation instant**; v1 lag = until next selection boundary.
- Return fates: on-time / straggler-hold / withheld-then-delivered (stale). No result cancellation.
- `syn_0/syn_20/syn_50` are **2-state** (AVL_TRAIN/UN_AVL only); `_trace_has_avl_eval` guard collapses D.2 for these traces.

---

## 3. Parity rungs (availability tier)

- **A1** `avail_composition`: per-state counts, binned. **A3** `trace_time_base_consistency` — hard gate, CONTROL (dep K3). **A4** `per_trainer_duty_cycle`. **A4dur** duration-weighted TVD (pass: `mean_err ≤ 0.05`, `frac_within_tol(τ=0.10) ≥ 0.95`).
- **withheld_delivery**: dist of `delivery_ts − sct` + staleness + accept/reject split.
- **abandon_timeout**: count + timing of 90s vclock abandons. Fails loud on wall-clock leak.
- **eligible_pool_reduction** (`Aa`): agg-observed fraction vs trace ground truth (HELD — needs run data).
- **observation_lag** (HELD): transition→effect boundary lag.
- **starvation_advance**: vclock jumps under scarcity, count + timing.
- **Ramp:** `syn_0` (regression) → `syn_20` (first validation) → `syn_50` → `mobiperf_*`.

---

## 4. Staged plan

### A ✅ — Substrate
`flame/availability/trace.py` + `AvailabilityMixin` (mixed into all four `TopAggregator`s). Default OFF → byte-identical. Exit: syn_0 clean.

### B ✅ — A3 time-base CONTROL
A3/A4 rungs, origin = `agg_start` both modes. Exit: A3 PASS oort syn_20.

### C ✅ CONFIRMED — Oracular driver + send-time gate + vclock abandon
`AvailabilityMixin` shared effect (not forked per stack): `compute_delivery_ts`, `free_stalled_slot`, `withheld_held_ends`, `_sim_withhold_if_unavail`, `_sim_pop_committable`, `_sim_reinject_ready_withheld`, `_sim_abandon_stalled`, `_emit_withheld_delivery`. felix 49/49 syn_20; oort 39/48 (pre-existing failures, §9.1).

### C.6 ✅ CONFIRMED — Aggregator tracking + plots
`_avail_stamp_end_states` writes `PROP_AVL_STATE` pre-selection (was all-UNKNOWN). A4dur PASS. Five availability plots in `analyze_run.py`.

### D ✅ CONFIRMED — Aware proactive eviction (felix)
`_sim_evict_unavail_inflight` (felix-gated, sim-only). Task-aware eligibility (`get_curr_task_ineligible_trainers`) with `_trace_has_avl_eval` 2-state guard. Real send-gate confirmed (withheld n=7, accept_frac=1.0).

### E ✅ CONFIRMED syn_20 — Sync baselines (feddance + refl)
Syncfl `_distribute_weights` (abandon/evict/stamp) + `_sync_sim_recv_first_k` (withhold + bonus drain). Accept-stale path (E.2: FedAvg has no staleness gate). feddance 46/47 syn_20 (C2 emergent noise; A-rungs/U3/U6/K8/U2 PASS; Challenge 9 ✅).

### F ✅ F.2 CODE-COMPLETE — Starvation / vclock-advance under scarcity

`_next_avail_vclock()` mixin helper returns `min(next_avail_transition_ts, next_pending_withheld_delivery_ts)`.

**F.2 unified pre-selection return-early pattern** (all three aggregators):

```python
_in_flight = getattr(channel._selector, 'selected_ends', set())
num_eligible = len(set(channel._ends.keys()) - set(curr_unavail_trainer_list) - _in_flight)

if num_eligible < threshold:  # oort: desired_selection=13; syncfl: agg_goal=10
    if self.simulated and self.trainer_event_dict is not None:
        _nxt = self._next_avail_vclock()
        if _nxt is not None and _nxt > self._vclock.now:
            self._vclock.advance(_nxt)
            self._sim_abandon_stalled(channel)
            # re-stamp at new vclock
            curr_unavail_trainer_list = self.get_curr_task_ineligible_trainers(task)
            _held = self.withheld_held_ends()
            if _held:
                curr_unavail_trainer_list = list(set(curr_unavail_trainer_list) | _held)
            channel.set_curr_unavailable_trainers(trainer_unavail_list=curr_unavail_trainer_list)
            self._avail_stamp_end_states(channel)
            channel.properties["vclock_now"] = self._vclock.now
        logger.info(f"[SIM_STARVATION] round={self._round} eligible={num_eligible} < {threshold}; vclock→{_nxt}")
    else:
        time.sleep(0.5)
    return  # outer run() loop retries non-blocking

selected_ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
```

**What was wrong and what F.2 fixed:**
- Oort had `min_required=5` (50% of agg_goal) + `max_retries=5` + "proceed anyway" fallback + 2s blocking sleep. Fixed: threshold=`desired_selection=int(aggr_num×overcommitment)=13`, unbounded, non-blocking.
- SyncFL fired only at pool=0 post-selection (`if not selected_ends:`). FedDanceSelector returns partial selections (e.g., 3/10) which are non-empty → never caught. Fixed: `agg_goal=10` PRE-selection.
- AsyncFL had `time.sleep(0.5)` at no-recv-ends path in both modes. Fixed: sim path advances vclock.

**n=48 syn_50 smoke (Jun 28):** no stalls ✅, K1 cadence ✅, starvation not populated — correct (staggered n300 schedules: min_avail=24 at n=48 >> both thresholds).

**n=25 syn_50 1800s (Jun 29 — both baselines):**
- **oort** ✅: 1 `[SIM_STARVATION]` at round=121 (eligible=12 < 13); Fst PASS (jump=74.3s, 5× mean); K1/K3a PASS; 46/53 (P3/K2/K3 pre-existing §7).
- **feddance**: mechanism correct (withheld n=4, accept_frac=1.0, K1/K2/K3/P3 PASS), BUT 0 starvation events. Root: sync FL clears `selected_ends` at round boundary → eligible = n − unavail (no in_flight term) ≈ 25 − 6 = 19 at peak, min observed=11 > threshold=10. The n=25 reasoning assumed oort-style in_flight reduction, which doesn't apply to sync. A3/A2 fail at n=25 due to join-ramp artifact (not a time-base bug, see §7).

**Exit (pending B2.0.1 hang fix + feddance n≈20 syn_50 smoke — n=12 was degenerate, see Status/Challenge 15):** `starvation_advance` populated for feddance AND ≥1 real training round; `[SIM_STARVATION]` log lines; K1 monotone; both modes self-stop at the budget.

### G.1 ✅ — Ladder integration
`starvation_advance` rung in `checks.py` + `report.py`; 67/67 parity tests (+4 starvation tests).

### G.2/G.3 — Ramp + sign-off (Batch 2, pending Stage F exit)
syn_50 → mobiperf, all baselines, 3h. Per-baseline sign-off. **Full detail + contingency decision tree in ▶ Next actions → "Batch 2 — Long-run parity campaign" (B2.0–B2.5).**

### G.4 — Terminology cleanup (after Batch 2)
Rename `oracular_trainer_avail_check` → `_trace_read_avail_check`; update log/comment "ORACULAR" references (keep YAML field value); consolidate config-gate paths. No-op refactor — defer until all baselines confirmed passing.

### H (FUTURE) — Message-transport + continuous scheduling
Turn `client_notify` ON for aware baselines: swap trace-read for real `avl_*` trainer→agg messages, processed mid-round. Add event-scheduled vclock clamp. Effect logic unchanged (C.5/D.1 hook was built for this). Re-measure `observation_lag` (must be ≈0).

---

## 5. Challenges / land-mines

Resolved challenges are noted briefly; open ones have full detail.

1. ✅ **Ordering on `delivery_ts`, not `sct`** — withheld commits at `max(sct, next_avail) > sct`. Fixed; U6/U3 validated.
2. ✅ **A3 time-base drift** — hard CONTROL gate; 90s abandon re-clocked to vclock. Do not read A1/A2/A4 until A3 passes.
3. ⚠️ **A2 two-tolerance trap:** `eligible = candidates − in_flight − unavailable`; bimodal sim distribution (avail windows) vs smoother real → KS shape artifact. Means match (real=47.1, sim=47.3); not a mechanism bug. KS improving with run length (0.437→0.338). Expect ≤0.2 at Batch 2 3h run.
4. ✅ **Busy ≠ unavailable ≠ withheld** — three ledgers, never conflated.
5. ✅ **Real send-gate fidelity** — confirmed withheld-then-delivered (not drop); n=7 accept_frac=1.0.
6. ✅ **Determinism** — commit ordered by `(delivery_ts, end_id)`. [Stage H] full vclock tie-break.
7. ✅ **Compound states with carry-over** — oort §4.9 straggler + UN_AVL cross-product covered in unit tests.
8. ✅ **AVL_EVAL inert for oort** — oort dispatches 0 eval; `_trace_has_avl_eval` guard handles 2-state traces.
9. ✅ **Staleness on sync changes cohort** — K8/U2 movement expected; reuse existing threshold, no new scalar.
10. ✅ **Scarcity advance must not skip events** — `_next_avail_vclock()` = min(transitions, withheld deliveries). K1 guarded.
11. ✅ **Regression discipline** — syn_0 byte-identity on every stage before syn_20 validation.
12. ✅ **Library mixin spans examples** — `AvailabilityMixin` + `trace.py` in `flame/`; never re-add example-local copy.
13. ✅ **Empty per-task pool corrupts shared `selected_ends`** — ROOT-CAUSE FIXED (Jun 29, ahead of the mobiperf ramp). `_handle_send_state`'s "invalid prior selection" cleanup checked `end_id not in ends` where `ends` was the availability-filtered eligible pool, so an in-flight trainer that merely went UN_AVL (or wrong task-type) was dropped from `selected_ends` though still connected; an empty per-task eligible pool (all AVL_EVAL on a 3-state trace, or the 2-state eval path) wiped ALL in-flight tracking across the shared train+eval `selected_ends` → hang. Fix: added `connected_ends` param to `_handle_send_state` in all three selectors (`async_oort`, `async_random`, `fedbuff`); cleanup now keys off the full connected pool, new-candidate selection still uses `eligible_ends`. Falls back to `ends` when omitted (backward-compat). The interim `_trace_has_avl_eval` 2-state guard is KEPT (defense-in-depth; it also drives task-type partitioning). +3 regression tests (`TestChallenge13SendStateCleanup`). **Still needs live exercise at mobiperf_3st (B2.3).**
14. ✅ **Scarcity threshold mismatch** — oort `min_required=5` + `max_retries` + proceed-anyway; syncfl pool=0 post-selection. Fixed by F.2 unified pattern.
15. ⚠️ **Sync vs async starvation threshold gap + scenario-sizing trap** — async (oort): `eligible = n − unavail − in_flight`; `in_flight ≈ agg_goal` so threshold is effectively `n − unavail > desired_selection + agg_goal`. Sync (feddance): `on_round_completed` clears `selected_ends` before next `_distribute_weights` → `eligible = n − unavail`; starvation fires when `unavail > n − agg_goal`. At n=25 syn_50, peak unavail=14 < 15 (=25−10) → never fires. **The n=12 "fix" was WRONG** (assumed ~3 unavailable; syn_50 ⇒ ~6 ⇒ eligible≈6 ≪ 10 ⇒ *perpetual* starvation, ~0 training — confirmed Jun 29). **Correct scenario = make eligible straddle the threshold:** `eligible ≈ 0.5 n` at syn_50, so **n ≈ 20** centers it on agg_goal=10 (n=18 for more starvation, n=22–24 for mostly-training). Generalize: pick **n ≈ threshold ÷ (1 − unavail_frac)**, not a fixed number; per-baseline (oort floor n ≳ 16).
16. ✅ **Real-mode aggregate-recv hang under unavailability** (FIXED Jun 29 — B2.0.1). The syncfl base called `channel.recv_fifo(ends, first_k=agg_goal)` in real mode with **no `timeout`**; when withheld (unavailable) trainers don't send, fewer than `first_k` arrive and the barrier blocks forever (real feddance n=12: ~46 min, 0 rounds, killed by the watchdog). NOT the scarcity poll — pure scarcity self-terminates via `increment_round` (sim proved this at round 4). oort/asyncfl already bounded their real recv; only syncfl lacked it. Fix: `timeout = min(trainer_recv_wall_timeout_s=90 s, remaining budget)` on the real `recv_fifo`, mirroring oort. Distinct from B2.0 #3 (which is *what counts* against the budget — parity-correct); this is the *real recv having no abandon* under withholding.

---

## 6. Dead-ends (settled — do not retry)

- **busy → UN_AVL routing**: ramped in-flight to ~300; busy/unavailable/withheld are three distinct states.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch `_sim_send_ts`): stuck UN_AVL forever. Availability reads `_vclock.now`.
- **Wall-clock in sim** for selection gate or 90s abandon: wall barely advances vs vclock → every deadline missed.
- **Per-tick MQTT broadcast**: comms storm + sub-optimal decisions. v1 = oracular pull (zero comms).
- **Ordering withheld commits by `sct`**: re-introduces past-dating. Fixed: order by `(delivery_ts, end_id)`.
- **Forking withhold/abandon per stack**: single shared `AvailabilityMixin` (Challenge 12).
- **A4 counting bare transition fraction**: brittle, blind in oracular mode. Replaced by `A4dur` + `Aa`.
- **Silent-OFF trace-name mismatch** (`avl_events_syn_20` → `syn_20`): fixed in `trace.py` name normalization.
- **D.2 excluding AVL_TRAIN from eval on 2-state traces**: made eval pool permanently empty → `_handle_send_state` cleanup wiped `selected_ends` across both tasks → run hangs exit-code 0. Fixed by `_trace_has_avl_eval` guard.
- **Subtracting starvation vclock-jumps from `max_experiment_runtime_s`** (Jun 29, B2.0 #3): would break sim/real parity. Real mode polls scarcity (`time.sleep(0.5)`+retry) and consumes WALL budget; sim's vclock jump consumes virtual budget symmetrically — they match by design. Discounting sim jumps would let sim run more training rounds than real per budget. The budget counts scarcity wait by design; size `--runtime-s` accordingly and pick a valid cohort (oort: n ≥ desired_selection, see cohort-floor guardrail).

---

## 7. Known parity failures (non-blocking — investigate at Batch 2)

| Check | Baseline | Status | Verdict |
|---|---|---|---|
| K3b `overhead_residual` | oort | rel≈0.116 consistently | Was PASS at 1.5h → run-length sensitive. Root-cause unclear (P3 gates it at n=300). Investigate at Batch 2. |
| A2 `eligibility` KS | oort | 0.437 (1800s) → 0.338 (3600s) | Shape artifact: bimodal sim vs smoother real distribution. Means match. Improving with run length. |
| P3 `trainer_speed` | oort | ratio=1.153 at n=300 (tol 1.15) | Marginal tail divergence at full cohort. Possibly noise; gates K3b. Investigate at Batch 2. |
| C2 `loss` | feddance | avg_diff≈0.16 (2–3 eval pts) | Emergent early-training noise at α=0.1. K8/C1/utility PASS; not a mechanism gap. |
| U5 `inter-arrival` ρ | feddance | ρ=0.381 at syn_50 (non-enforced) | Worsened vs syn_20 (0.659). Watch at mobiperf. |
| A4dur | feddance | ✅ RESOLVED — PASS | Was wrongly diagnosed as "stamps post-selection". F.2 already stamps pre-selection (`syncfl:896` < `:957`). Parity CLI on n=25 syn_50 pair → A4dur PASS (avl_state 71/71 sim, 67/67 real). K3b also PASS for feddance. |
| A3 `avail_timebase` | feddance | max_rel_diff=0.245 at n=25 | Join-ramp artifact: real trainers all connect by round 2 (eligible=25 immediately), sim ramps 17→25. First-decile bin dominates at small n. Not a time-base bug. Expect clear at Batch 2 n=300. |
| A2 `eligibility` KS | feddance | KS=0.509 at n=25 (gated by A3) | Same join-ramp artifact + downstream of A3 FAIL. real_mean=20.5, sim_mean=18.6 (sim correctly applies unavailability; real ramps faster). Not a mechanism gap. |
