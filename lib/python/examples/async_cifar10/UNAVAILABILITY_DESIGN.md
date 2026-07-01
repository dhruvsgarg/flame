# Sim Unavailability — Design & Staged Plan

## ▶ NEXT STEP (Jun 30 — run this to confirm the A4dur + B2.0.3 fixes)

Launch from `lib/python/examples/async_cifar10`:

```bash
bash scripts/smoke_suite.sh \
  --steps 4,5 \
  --starvation-baselines feddance \
  --runtime-syn20-s 900 --runtime-syn50-s 1800 \
  --output-dir experiments/smoke_confirm_$(date +%Y%m%d_%H%M) \
  --background
```

Unattended — `--background` re-execs the suite via `nohup`+`disown` and returns immediately; it survives
the launching shell/SSH session closing. Prints a PID, a `nohup.log` to `tail -f`, and the eventual
`report.txt` path. 14 runs (12 for the 6 syn_20 sim+real pairs, 2 for feddance syn_50 sim+real), ~1–3h
wall depending on GPU load. When done: re-run `python -m scripts.parity.cli` on the fresh pairs and
confirm `duty_cycle_duration` (A4dur) and `avail_timebase` (A3, feddance syn_50) both clear — see
T5-smoke below for full context on both fixes.

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

## Status (Jun 30 — T5-smoke ✅ COMPLETE 23/23 PASS @ n=300; A4dur fix ✅ CONFIRMED live; B2.0.3
join-barrier trace-clock offset ✅ ROOT-CAUSED + FIXED; next: user re-runs the 6 syn_20 smoke pairs
(picks up both fixes), then isolate/confirm feddance syn_50 A3 on that fresh data, then full T5 — see
T5-smoke below)

**A/B/C/C.6/D/E ✅ CONFIRMED syn_20. F.2 ✅ FIXED (B2.0.2 starvation self-termination).
syn_0 ✅. oort n=25 syn_50 ✅ (starvation fires, K1/K3a PASS). B2.0.1 real recv-barrier ✅ FIXED + confirmed.
T0–T5 pre-work ✅ COMPLETE. T5-smoke ✅ COMPLETE (23/23 PASS, Jun 30). A4dur ✅ FIXED + CONFIRMED (fresh
felix syn_20 pair, `mean_err=0.0`). B2.0.3 (real-mode join-ramp baked into the trace-read clock) ✅
ROOT-CAUSED + FIXED. Tests green: 542 pass / 7 skip.**

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

### B2.0.3 ✅ RESOLVED (found via feddance syn_50 A3, Jun 30; fixed same day; confirmation run pending)

Real's trace-read clock (`agg_start_time_ts`, `_avail_now()`) included the n=300 MQTT join ramp
(~300s), reading the availability trace ~300s ahead of sim/ground-truth. Fixed in
`syncfl/top_aggregator.py`: `_mark_join_barrier_done()` re-anchors `agg_start_time_ts` to real time at
join-barrier resolution (real mode only, self-correcting to actual join duration). 6 regression tests
in `test_join_barrier_reanchor.py`. See T5-smoke section above for full root-cause writeup; needs a
fresh feddance syn_50 real+sim pair to confirm A3 clears — folds into the pending smoke re-run.

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

Two suite invocations covered all 23 jobs with no gap (`experiments/smoke_20260630_0931/` = steps 1–2,
`experiments/smoke_20260630_1040/` = steps 4–5): pytest (536p/7s), syn_0 sim ×6, syn_20 sim+real ×6, syn_50
sim+real ×{feddance,oort}. **0 FAIL/ERROR/TIMEOUT, 0 `SIM_WALL_CEILING`, all self-stopped.** oort_star/fedbuff
(new scaffolding) ran clean with no `ValueError` — T3 confirmed live, not just unit-tested. B2.0.2 holds at
full n=300 scale across all three traces. Command/per-run/wall-time tables superseded by this summary; raw
logs are in the two `experiments/smoke_*/` dirs if needed. `scripts/smoke_suite.sh` grep-count crash hit
during this run is already fixed (see commit history).

**Ad-hoc parity pass on the smoke output** (`python -m scripts.parity.cli`, n=300, 900–1800s wall — short
vs. the 3h T5 target, so DIST/EMRG-tier misses are expected, not a gate; no baseline fails a
structural/K-tier check). Two findings, both root-caused and fixed same-week (below); neither has been
re-verified yet on a fresh full-scale smoke pass (the old telemetry predates both fixes):
- `duty_cycle_duration` (A4dur) failed at syn_20 — ✅ root-caused, fixed, **and confirmed live** (see below).
- feddance syn_50 failed A3 — ✅ root-caused as **B2.0.3** and fixed (see below); confirmation run pending.

#### A4dur `duty_cycle_duration` ✅ ROOT-CAUSED + FIXED + CONFIRMED (Jun 30) — single cause, all 6 baselines

Real-mode selection telemetry never stamped `vclock_now` (hardcoded `None` at 5 call sites across
`syncfl`/`asyncfl`/`oort` `top_aggregator.py`, gated `if self.simulated:`), so A4dur's duration-weighted
real/sim comparison fell back to `ts - t0` (t0 = the run's *first selection event*) instead of the trace's
true `agg_start`-anchored origin. Confirmed on telemetry: the same syn_20 transition lands at real
`ts-t0=300.7s` vs sim `vclock_now=600.2s`, producing an identical 0.1666 TVD error for all 34/300 syn_20
trainers that transition — that ~300s gap turned out to be the same join-ramp mechanism separately
root-caused below as **B2.0.3**, but this fix (telemetry only) was sufficient on its own to clear A4dur.
Confirmed library-level (not per-baseline) by re-pairing oort/feddance's own syn_20 real+sim runs directly:
`mean_err=0.018–0.021`, matching felix/refl/oort_star/fedbuff exactly — their worse-looking numbers in
`parity_oort.json`/`parity_feddance.json` were a stale-file artifact (the JSON is overwritten per CLI run;
the persisted file held their *syn_50* result, not syn_20).

**Fix:** `channel.properties["vclock_now"]` now stamped via `ClientAvailability._avail_now()` for both modes
(was sim-only) at the 3 unconditional call sites; the 2 sim-only starvation-branch sites were already
correct, left untouched. `avail_state_series.py` now prefers `vclock_now` in both modes, falling back to
`ts - t0` only for telemetry recorded before this fix. New regression test
`test_build_series_real_prefers_vclock_now_over_ts_minus_t0`.

**Confirmed** on a fresh felix syn_20 real+sim pair (n=20, runtime=180s —
`run_20260630_181633/181759_dbg_felix_n300_alpha0.1_syn_20_stream_{sim,real}`): first real selection event
now carries `vclock_now=15.66s` (small, non-null, no join-ramp offset — n=20 joins fast). Parity checker:
`duty_cycle_duration` `mean_err=0.0`, `frac_within_tol=1.0`; `avail_timebase` (A3) also clears (0.17 < 0.2).
Closed — no further action.

#### B2.0.3 real-mode trace-clock included join-ramp dead-time ✅ ROOT-CAUSED + FIXED (Jun 30)

Root cause of the feddance syn_50 A3 (`avail_timebase`) failure (0.227 > 0.2 tol), found by comparing
real/sim `num_eligible` against the syn_50 trace's own ground truth (n=300, direct trace count): the
300→221-trainer drop is defined at trace-time **t≈600s**. Sim reads it correctly (`vclock_now≈600–620s`
at the drop). **Real reads it ~300s early** (wall-elapsed `t≈283–312s` at the drop) — the same ~300s
gap the A4dur investigation already surfaced (`ts-t0=300.7s` vs `vclock_now=600.2s`) but had attributed
entirely to the missing telemetry stamp.

Actual mechanism: `agg_start_time_ts` is stamped at aggregator `__init__`
(`syncfl/top_aggregator.py`, in `internal_init`), *before* `channel.await_join()` /
`_await_min_trainers()` run. Trainer processes take real wall-clock time to spawn and connect in
**both** modes (sim only virtualizes training *sleeps*, not process startup) — at n=300 this join
ramp is ~300s. Real's `_avail_now()` (`client_availability.py`) reads
`time.time() - agg_start_time_ts`, so that ~300s of pure connection-establishment dead time is
silently baked into every subsequent trace read: real ends up ~300s further into the trace than it
should be at any given round. Sim's `_avail_now()` reads `_vclock.now`, which only starts advancing at
round-0 selection, so sim never pays this cost — vclock=0 already means "round 0 is starting," while
real's wall-clock=0 means "aggregator process just started," a materially earlier point. syn_50's dense,
high-scarcity transition schedule (43% peak unavailability) moved enough trainer-mass across a state
boundary within that 300s gap to blow A3's 0.2 tolerance; syn_20's sparser schedule apparently didn't
(0.108, still under tolerance) — so this bug was silently present in every real n=300 run, just not
loud enough to fail A3 except at syn_50.

**Fix** (`syncfl/top_aggregator.py`): new `_mark_join_barrier_done()` helper, called from every exit
path of `_await_min_trainers()` (barrier-disabled early return, cohort-satisfied return, timeout
fallback) in place of the old `self._join_barrier_done = True` inline. In real mode only, it also
re-stamps `self.agg_start_time_ts = time.time()` — self-correcting to however long the join actually
takes (not tied to today's ~300s figure at n=300; shrinks automatically if the join ramp is later
reduced). Sim mode is untouched (its origin was already correct), so this cannot affect the syn_0
byte-identity gate. `oort`/`asyncfl` inherit `agg_start_time_ts` and `_await_min_trainers` from
`syncfl.TopAggregator` — one fix point covers all six baselines. 6 new regression tests in
`tests/mode/test_join_barrier_reanchor.py` (real re-anchors, sim doesn't, one-shot doesn't re-fire).
Full suite green: 542 pass / 7 skip.

**Verification: pending** — needs a fresh feddance syn_50 real+sim pair (old telemetry predates the
fix). Folds naturally into the next-steps smoke re-run below, since that re-run already needs fresh
real+sim pairs for the A4dur confirmation.

**Next steps:**
1. Re-run the 6 syn_20 sim+real smoke pairs (picks up both the A4dur and B2.0.3 fixes) — user running
   outside this conversation. Re-check all 6 through the parity checker once done.
2. Run a fresh feddance syn_50 real+sim pair and confirm A3 clears (B2.0.3 fix verification).
3. Then run the full T5 overnight campaign (n=300, 3h, all traces, mobiperf) — scope/order below.

### T5 — Parity-table campaign (syn_0 → syn_20 → syn_50 → mobiperf, sim+real, all six baselines)
Produce a `PARITY.md`-style table (rows = baselines, cols = traces × modes, cells = pass/tot + ROOT). Order:
1. **syn_0** regression (byte-identity gate ON/OFF) — all six.
2. **syn_20** then **syn_50** n=300, 3 h, sim+real — all six. (T5-smoke was 900–1800s wall, short vs. this
   3h target; `duty_cycle_duration` and the feddance syn_50 A3 regression are both root-caused + fixed —
   confirm both on fresh runs first, see T5-smoke next steps above and §7.)
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
18. ✅ Real-mode trace-clock included join-ramp dead-time (B2.0.3) — `agg_start_time_ts` stamped before the
    trainer-join wait, so real's `_avail_now()` read the trace ~300s ahead of sim/ground-truth at n=300.
    Fixed via `_mark_join_barrier_done()` re-anchor; confirmation run pending.

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
| A3 `avail_timebase` | feddance | syn_20 PASS @ n=300 (0.108). syn_50 FAILed (0.227) — ✅ ROOT-CAUSED as **B2.0.3** + FIXED, confirmation run pending | Real's trace-read origin (`agg_start_time_ts`) included the ~300s n=300 MQTT join ramp, reading the trace ~300s ahead of sim/ground-truth; syn_50's dense transitions exposed it, syn_20's sparser ones stayed under tolerance. Fixed via `_mark_join_barrier_done()` re-anchor in `syncfl/top_aggregator.py` — see T5-smoke section above. Needs a fresh syn_50 real+sim pair to confirm A3 clears. |
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
3. **PR workflow** (once 1–2 are in reasonable shape, and T5-smoke's two pending fix-confirmation runs —
   see T5-smoke above — are green): clean-diff PR for this branch → write up the design decisions that
   were kept (durable, what's in the doc now) → write up decisions rejected / not pursued (currently
   scattered across §6 Dead-ends + inline "why not X" notes — worth a final sweep to make sure nothing
   rejected got lost) → fold both into this doc (already mostly done by §5/§6/§7) → then carry the
   relevant parts into `lib/python/examples/fwdllm/simulate_fwdllm.md`, which **already exists** and
   already defers unavailability to "follows `UNAVAILABILITY_DESIGN.md` as its template" — so the
   destination is wired, just needs the actual content once this doc is in PR-ready shape.

### Recommendation: minimal bar before porting fwdllm (not yet acted on)

Don't gate the port on the full 3h × 6-baseline × 4-trace T5 campaign — that's for paper-quality parity
numbers, not "is the substrate solid enough to build on." Smaller bar, roughly in order:
1. T5-smoke's two pending fix-confirmation runs (A4dur confirmed; B2.0.3/feddance-syn_50-A3 confirmation
   still pending) — cheap, already in motion.
2. **mobiperf_2st** (simplest 3-state trace) for at least one async baseline (felix) and one sync baseline
   (refl or oort_star) — not full 3h, ~30–45min is enough to exercise the AVL_EVAL split and Challenge
   13's empty-pool cleanup *live* for the first time. Highest-value remaining gap precisely because it's
   never been run, and the kind of bug better found in cifar10 than mid-port.
3. Skip the full syn_50/mobiperf sweep across all 6 baselines and both modes as a porting gate — separable,
   can run in parallel with/after the port starts, not a blocker before it.
