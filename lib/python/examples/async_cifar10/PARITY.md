# Real / Sim Parity — Methodical Causal Ladder

> **DEPRECATED — reference only.** The single source of truth is [ROBUST_FL_READINESS.md](../_metadata/ROBUST_FL_READINESS.md) → [FELIX_READINESS.md](../_metadata/FELIX_READINESS.md). This file only gets trimmed from here on as its content moves there; don't add to it.

Living doc for the parity checker. Kept in sync with
`scripts/parity/checks.py` (checks), `scripts/parity/report.py` (stage grouping +
verdict), and the pytest suite.

**Scope: this is the shared parity methodology, not one example's.** The ladder (§1), roles/
tiers/dependency-gating and mechanism reference (§3) are
**example-agnostic**. The async_cifar10 rung catalog (§2) is the reference instance; a second
example appends its own rung catalog rather than forking the method. **fwdllm** (forward-gradient,
variance-gated dynamic-K FL) is catalogued in **§F** — its live parity status/fixes are in
`examples/fwdllm/simulate_fwdllm.md` (references §F for rung definitions); its build plan, structural
deltas, roadmap, and calibration work are in `examples/fwdllm/FWDLLM_DESIGN.md`.

**Comparator — give two run dirs, get a report JSON:**
```bash
cd lib/python/examples/async_cifar10
PYTHONIOENCODING=utf-8 python scripts/parity_check.py \
  --real experiments/<real_run_dir> --sim experiments/<sim_run_dir> \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity_<baseline>_<tag>.json
# batch: --batch --experiments-dir experiments --baselines felix oort refl feddance
```
`--budget-s` = the run's `--runtime-s`; `--lenient` demotes DIST fails to warnings.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Launch runs** (node-agnostic): `bash scripts/debug_run.sh --baselines 'oort refl'
--runtime-s 3600 --mode both`. Reads `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml`
(each baseline × sim/real), seeds real+sim identically (`SEED=1234`), applies per-baseline
sim fixes. Split across machines via `--baselines`.

**Tests** (no cluster; under lib/python with `dg_flame`): `pytest tests/mode/
tests/selector/test_oort_selector.py tests/sim/ examples/async_cifar10/scripts/parity/`.

---

## §1  The parity ladder (methodology)
An FL run is a pipeline; each round flows the same stages in both modes:
`clock/time-base → availability → selection → dispatch+training → return+ordering →
aggregation → utility → emergent outcomes`. Parity must hold at every stage; if it breaks at
stage N, every stage above also diverges — those are **consequences, not bugs**. The checker
finds the **lowest broken rung** (earliest stage with sound inputs but diverging output) = the
root.

**Three roles** (tag each check): **CONTROL** confirms a stage's *input* is identical (failing
= fix the input model); **MECHANISM** confirms one *transformation* is modeled (failing with
passing controls = the localized bug, the prize); **EMERGENT** an aggregate outcome (never
localizes alone — walk *down* the ladder, never fix an emergent directly).

**Two axes:** STAGE (0–9, drives diagnosis) × TIER (drives verdict): `INV` (sim invariant,
hard FAIL), `EXACT` (tight tolerance, hard FAIL), `DIST` (distributional, FAIL unless
`--lenient`), `DIAG` (informational, feeds root-cause).

**Dependency gating:** each check declares upstream prerequisites. The engine walks rungs
bottom-up, labels the lowest enforced FAIL with all-passing upstreams **ROOT-CAUSE**, demotes
higher FAILs whose chain contains a failed check to **DOWNSTREAM**. `deps` names the
*strongest causal link*, not a generic base (TC1 gates only K10 — a missing field makes a
check SKIP, not FAIL).

**Growth rule:** every root-caused bug leaves behind the most fine-grained check that would
have localized it, at its stage with deps. Checks are **append-only** (a redundant check is a
future regression guard). Split a coarse check into one assertion per mechanism.

### §1.5  Comparison axis — grade on the LOGICAL budget (`matched_virtual_budget` deleted 2026-07-23)
**Parity is graded to a fixed LOGICAL budget N (min committed `data_id`s / FL rounds both sides
reached, §F-2), never to a matched virtual-time window.** `_matched_virtual_budget` is deleted;
`_matched_logical_budget(real, sim) → (N, prog_fn)` replaces it (checks.py).

*Why V violated parity.* The old `V = min(final_sim_vclock, final_real_wall)` truncated both streams
to commit-time ≤ V — treating **sim's virtual vclock and real's wall as one interchangeable axis**, but
whether `vclock ≈ wall` (`sim_rate`, §F-10) is *the parity question itself*. Normalizing along the axis
under test is circular. V existed only because pairs launch on a **fixed wall-clock budget** (7200s
each) → different logical endpoints → mismatched counts. Both fixed-wall comparisons are wrong in
opposite directions:
- **Matched-window (V) MASKS throughput divergence.** The fwdllm/fwdllm_plus sync leg read ~at-parity
  under V while HIDING a real **1.57× throughput gap** (68 vs 110 databins/7200s), found only by
  reading raw databins/wall and fixed on the real side (`drain_ready`, `var_bad` dedup). See §H.
- **Raw full-run counts PENALIZE sim's legitimate speedup** (`sim_rate ≥ 1` → more cohorts in equal
  wall, fluxtune 109 vs 118). And V doesn't even remove a real divergence: fluxtune ran 1795 vs 1648
  cycles *inside* the matched V.

*The fix (§F-2/§F-12).* Fix the WORK (progress ≤ N, the common prefix) and let TIME be the measured
output. `_matched_logical_budget` returns N on the progress axis (`data_id` tuple for fwdllm, `round`
for async); checks truncate to `prog(e) ≤ N`. The clock/throughput signal becomes **real's
algorithmic-time-to-N vs sim's vclock-to-N** (`_time_to_progress`), a clean ratio — prove-on-first-bin
(§F-12) is its N=1 case.

*What migrated (landed 07-23; the 5 checks that used V):*
- `total_commits` (U2), `terminal_state` (K8): count is trivially N at fixed N → **reshaped to
  time-to-N** (rel_diff ≤ 5%). K8 keeps its live **trainer-set-over-N** count dimension.
- `v2_var_trajectory`, `utility`, `cohort_sequence.count`: distributional/count truncation swapped
  clock-≤-V → progress-≤-N. `cohort_sequence` now deps on `v1_iter_per_data_id` (its `count` is
  rolled-up V1).
- **Untouched (already logical):** `throughput` (K2, full-run rate = units/time) and `per_round_advance`
  (K3) use a min-*count* matched window, not V.
- **Still open:** pairs are still launched to a wall budget; the checker truncates to N post-hoc, which
  is correct, but a fixed-N *launcher* termination would remove the wasted tail — verify how
  `run_parity.py` ends a run before changing it.

## §2  The ladder — check catalog
`[NEW]` = to implement; else exists in checks.py. "Isolates" = what a FAIL means when its
upstreams pass. "Dep" = upstream prerequisites.

**Stage 0 — Telemetry coverage** (gate for everything)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| TC1 `[NEW]` | Field coverage matrix | CONTROL/INV | a downstream-read field missing/sparse in one mode (explains every SKIP) | — |
| K10 | vclock_now present (sim) | CONTROL/INV | sim never stamps vclock | TC1 |

**Stage 1 — Clock / time-base** (the foundation; most bugs live here)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K1 | vclock monotone (sim) | MECHANISM/INV | vclock goes backwards | K10 |
| K7 | sim_rate in [0.01,100] | MECHANISM/INV | vclock/wall absurd | K10 |
| P3 | trainer_speed_s distribution | CONTROL/DIST | speed-model *input* differs | — |
| K3a `[NEW]` | Modeled-compute advance | MECHANISM/EXACT | advance **formula** (K-th fastest async / max-of-K sync), overhead excluded | P3,K1 |
| K3b `[NEW]` | Overhead residual | MECHANISM/EXACT | `real_advance − sim_advance` ≈ 0 (missing per-commit overhead) | K3a |
| K4 | Overlap factor | MECHANISM/DIAG | sim misses inter-round async pipelining | P3,K1 |
| K3 | Per-round advance dist | EMERGENT/EXACT | K3a+K3b+K4 rollup | K3a,K3b,K4 |
| K2 | Rounds-per-virtual-second | EMERGENT/EXACT | throughput rollup | K3 |

> Decomposition is the point: P3✓ K3a✓ **K3b✗** K4✓ → pure missing overhead. K3b
> cross-validates at Stage 4 (mqtt_fetch): trainer-level overhead = K3b residual × agg_goal.

**Stage 2 — Availability** (indexed by the clock; gated on Stage 1)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| A1 | avail_composition parity | MECHANISM/DIST | per-state counts diverge | — |
| A2 | num_eligible / num_candidates | MECHANISM/DIST | eligible-set size diverges | A1 |
| A3 `[NEW]` | Trace time-base consistency | CONTROL/DIST | availability indexed by different clocks (REFL HIGH-1) | K3 |
| A4 `[NEW]` | Per-trainer duty-cycle | MECHANISM/DIST | on/off fraction differs even when set sizes match | A3 |

> A2-fail + A3-pass = fix the clock first; A2-fail + A3-fail = fix the trace lookup.

**Stage 3 — Selection**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| S3/4 | num_chosen / in_flight / effective_c | MECHANISM/DIST | selector picks a different count | A2 |
| A2c | selected-vs-pool speed bias | MECHANISM/DIST | scoring bias diverges with pool matched (oort) vs pool itself (A2b, refl) | A2b |
| Sx | selector score-term localize | DIAG | which utility term drives a mix split (oort believed_I/temporal/system_util) | A2b |
| Sd | preferred-duration penalty bind | MECHANISM/DIST | oort speed-penalty binding freq + reconstructed `pref` (caught D1: real 80% vs sim 46%) | A2b |
| S2 | Participation frequency | EMERGENT/DIST | per-trainer chosen-count diverges | S3/4 |
| S1 | Per-round Jaccard | DIAG | exact set identity (WARN for stochastic) | A2 |

**Stage 4 — Dispatch & training** (per-trainer timing; the overhead source)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| T2 `[NEW]` | training_budget_s dist | CONTROL/DIST | speed-model *input* differs | — |
| T_pre/T_w2g/T_gpu/T_w2r/T_post `[NEW]` | per-phase splits | MECHANISM/DIST | one timing phase each (T_gpu dep T2) | — |
| T_mqtt `[NEW]` | mqtt_fetch_s phase | MECHANISM/DIST | per-commit MQTT overhead (cross-checks K3b) | — |
| T3 | GPU budget respected | MECHANISM/INV | real GPU overruns modeled budget | T2 |
| K6 | sim_send_ts correctness | CONTROL/INV | sim dispatch ts not stamped/advancing | K10 |

> Split the one `trainer_phase` DIAG blob into per-phase DIST sub-checks so the report names
> the diverging phase; keep the combined table for at-a-glance reading.

**Stage 5 — Update return & ordering**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U5 | Inter-arrival order (Spearman) | MECHANISM/DIST (WARN) | arrival rank within a round diverges | K3,S3/4 |
| U4 | agg_goal_count cycles 1..K | MECHANISM/INV | lost/double-counted update | — |

**Stage 6 — Aggregation**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U6 | Commit visibility lag | MECHANISM/DIST | aggregator-clock delay READY→COMMITTED diverges (sim past-dating); upstream of staleness. Same metric both modes (sim `vclock−sct`, real `wall commit−arrival`) | K3 |
| U3 | Staleness distribution | MECHANISM/DIST | staleness diverges (async: downstream of clock under-charge) | K3,U5,U6 |
| P1 | Aggregation sequence | EMERGENT/DIST (WARN) | per-round contributing set diverges | S2,U5 |

**Stage 7 — Statistical utility**: F1-3 per-trainer utility dists (EMERGENT/DIST; dep S2,T_gpu,U3).
**Stage 8 — Emergent**: K8 terminal-state @ matched V (dep K2,S2); U2 total commits @ V (dep
K2,U4); C1 accuracy curve, C2 `[NEW]` loss curve (dep F1-3,K8).
**Stage 9 — Budget/stop sanity**: K9 stopped-by-budget-not-cap (WARN); K5 failsafe ceiling
(sim wall overshoot >20%).

---

## §3  Mechanism reference — landed sim fixes
The sim does **real GPU compute** but stamps a *modeled* completion `sct` (no wall sleep).
**Overriding principle:** parity ≠ goal, a *correct* simulator is; real is the reference only
after `validate_real` shows it admissible (concurrency 28.8/c30, double-dispatch 0). Never
tune sim to a wrong real. All mechanisms below are config-gated (flag-off ⇒ byte-identical)
and guarded by tests.

### Clock & ordering (felix async, validated)
- **Overhead → 0** (`simCommitOverheadSeconds=0`): clock TRACKS completions
  (`vclock = max(vclock, sct)`), not an overhead ramp.
- **Drain by physical READINESS** (`_sim_recv_min`): admit any in-flight end whose message
  physically arrived into the reorder buffer → slow trainers buffer as futures, commit in
  `sct` order (staleness 7.2→3.5).
- **`realDistributeSettleSeconds=0`**: removes a real-only 2×`sleep(0.1)`/commit (advance 4.1,
  staleness 2.8).

### §3.drain  sct-ordered DIRECT drain (felix async; LANDED Jun 20)
`simSctOrderedDrain`. The async clock under-advanced because in-flight updates (instant in
sim) ingested via the `recv_fifo` streamer could be **stranded** out of the reorder buffer's
view (background task + shared `_rx_queue` + per-end dedup + grace timeout). The clock advanced
off the incomplete buffer and lapped stranded lower-`sct` updates → past-dating (`queue_wait`
26s/90.6% >5s, staleness 15 vs 2.8; 1.4 s/rd vs 3.85). Fix: drain each live in-flight end's rx
queue **directly** so the buffer is a complete snapshot and the min-`sct` gate + clock-jump
clamp commit in true order (`commit_gap≈0`). `End.get_ready_nowait` (non-blocking, peek-aware)
+ `Channel.drain_ready` (pull raw on backend loop, decode off-loop so `cloudpickle.loads`
can't stall the pump); `_sim_recv_min` drains `recv_ends ∪ _sim_inflight_expected`. Sync
untouched (`recv_fifo(first_k=len(ends))` barrier already waits for the whole cohort). Guard:
`test_async_sct_ordered_drain.py`. **Validated Jun 21:** `commit_gap` median 0, staleness
15→5.1, advance 1.4→3.04. Residual = overlapping re-dispatch → §3.resid.

### §3.resid  One-in-flight-per-trainer (felix async; `inflightResidence`; LANDED Jun 21)
**Invariant:** a trainer is re-pickable ONLY after its update returns AND is committed. Real
satisfies it by construction (channel holds an in-flight trainer out of `VAL_CH_STATE_SEND`
until aggregated — real 0% overlap). Felix sim freed trainers instantly → a fast trainer
re-selected while its prior update was in flight (sim 13.9%), overwriting `_sim_inflight_expected`
→ earlier update untracked, invisible to the `sct` gate (`gate_holds=0`) → lapped → past-dated
(14% tail, staleness 34). **First attempt (busy → UN_AVL) was WRONG** — see Dead ends.
**Correct fix:** a busy trainer HOLDS its concurrency slot in `selected_ends` (like real) until
commit. `_sim_hold_busy_slots` (`asyncfl/top_aggregator.py`, called from `_aggregate_weights`
at agg-goal) holds `pending_ends() ∪ set(_sim_inflight_expected)` in `selected_ends`/
`all_selected`/`_sim_pending_commit` (a SLOT, not unavail); released on commit in
`_sim_recv_min`. Bounds concurrency (`extra = c − len(selected_ends)`) AND excludes from the
pool. Sim only; sync uses §4.5 unavail path (correct — barrier re-selects). Guard:
`test_async_inflight_residence.py`. **VALIDATED Jun 21** (`…154600…sim` 90min): **46/46**, K3b
0.82→**−0.08**, advance 3.04→**3.93** (real 3.85), staleness 5.11→**2.83** (real 2.79), U6
mean_diff 3.91→**0.016s**. The whole K3b→{K2,U3,U6,K8,U2} cluster cleared together — one root.

### §3.evt  Event-driven re-dispatch (felix; FALSIFIED Jun 20, superseded by §3.drain)
`simStaggeredRedispatch` (gated, kept off). Pushed each commit's advanced vclock to a freed-slot
FIFO to re-stamp re-dispatched `sim_send_ts` and regain stagger. **Falsified:** advance got
worse (1.93→1.38) — the injected stagger is bounded by the clock advance it's meant to create
(≤0.55s injectable, circular). Real root was commit-side (§3.drain). Do not enable with
`simSctOrderedDrain`. Guard kept: `test_async_staggered_redispatch.py`.

### §3.async  Async ≠ sync selector knobs — do NOT inherit Oort *paper* defaults
`third_party/Oort` is sync-only; the paper defaults are SYNC values. Applied to the async
`AsyncOortSelector` (felix) they regress it (overlap 10.9× vs 6.6×) because the overlap model
is calibrated to the selected MIX and sync knobs narrow it. Root theme: many Oort knobs are
**per-round**, but "a round" differs in async (one `agg_goal` batch) vs sync (a barrier), and
async runs ~2–3× more of them.
- **`round_threshold`** (speed penalty) protects a SYNC barrier; async has no barrier →
  largely inert. Felix starts at **70**. *(Correction Jun 24, §S.pacer: the earlier "the pacer
  only raises toward 100, so the start washes out" described the BUGGY raise-only ratchet — that
  ratchet is exactly why the penalty went inert. The pacer is now the faithful two-branch
  controller and train-gated, so round_threshold no longer monotonically washes out; felix's
  penalty becomes active and its 46/46 must be re-validated.)*
- **`exploration_decay`** applied per-round; sync 0.95 floors exploration in ~29 async rounds.
  Felix uses **0.999** (`0.9999` ≈ never exploits — rejected).
- temporal/UCB `√(0.1·log(round)/last_selected)` auto-inflates with round count; pacer cadence
  fires more often in wall-time; staleness weighting is async-only.
- Principled generalization (not done): re-parameterize per-round terms by wall-time /
  samples-seen. Until then knobs are config-driven, anchored to the real run's spacing.

### §4.5  refl/oort — `sct`-gated pool exclusion (`inflightResidence`, validated)
A trainer modeled as still computing (`vclock < sct`) must NOT re-enter the eligible pool. In
`oort/top_aggregator._distribute_weights`, `_sim_buffer.pending_after(vclock)` is added to
`trainer_unavail_list` (the *unavailable* path, NOT `selected_ends` — which would re-dispatch
and reset `sct`); released at `vclock ≥ sct`. Fixed refl's pool composition (A2b 12.4→~6.5 =
real). Guard: `test_virtual_clock.py::test_pending_after_*`, `TestSimInflightResidence`.

### §4.9  oort — `sct`-gated carry-over (`simInflightCarryover`)
Sync oort over-selects (×1.3), closes at agg_goal=10, leaving ~3 slowest computing. Real keeps
them in `selected_ends` across rounds (`in_flight_after` 3.3); sim's update arrives at once,
gets stale-rejected, frees its slot → drains to 0.15. Fix: a prior-round straggler
(`_round − MODEL_VERSION > 0`) with `sct > vclock_round_start` is held (not yielded, not
clock-advanced), re-buffered in `selected_ends`, commits a few rounds later. Three follow-on
bugs fixed: (1) **lost straggler** — re-buffer ran after the yield-loop the caller abandons;
wrapped in `try/finally` (Jun 16). (2) **threshold creep** — `vclock_round_start` re-read each
`_oort_sim_recv` call crept forward across block-for-K retries; `_aggregate_weights` now pins
`self._round_start_vclock` once (Jun 17). (3) **block-for-K-fresh starvation** — the second
poll loop ran `while not self.simulated`, so sim got one pass and skipped not-yet-ready fresh
trainers → committed stale; removed the gate (confirmed 2.5h `sim_committed_fresh=10`). Guard:
`TestSimInflightCarryover`. *(The residual carry-over "decay" was the A2c scorer-input root —
see Settled roots; NOT a gate or speed-tail bug.)*

### §S.dur  oort/refl real selector duration = intrinsic client stamps (LANDED Jun 23 PM)
The selector's per-client speed signal `PROP_CLIENT_TASK_TRAIN_DURATION` feeds
`system_util = (pref/duration)^α`, where `pref` = round_threshold-th percentile of candidate
durations. **Real recorded `WALL_SEND_TS − dispatch`**, which still carried the dispatch→recv
**delivery lag** (`WALL_RECV_TS − dispatch`): the aggregator stamps `dispatch` at selection, but
a slow client held one-in-flight receives the weights later, so the lag is large for stragglers
(+8s for D≥20). That inflated real's slow-trainer durations, raised `pref` (real 10.7s vs sim
7.0s), so real UNDER-penalized slow clients → committed a ~0.77s-slower mix → per-round max-of-K
10.09 vs sim 9.69 → +0.5s/rd advance → fewer rounds = the **oort+refl K2 root**. The earlier
read-wait fix (§Settled oort) anchored the RECV side on a client stamp (`WALL_SEND_TS`) but left
the DISPATCH side aggregator-anchored. **Fix:** anchor BOTH ends on client stamps —
`_real_client_task_train_duration` returns `WALL_SEND_TS − WALL_RECV_TS` = the client's intrinsic
compute+sleep = exactly D (proven: trainer's own `wall_send − wall_recv == budget`; real GPU
~0.01s) = sim's `max(gpu,D)`. Real-path only; selector-input only (clock/`sct` untouched, so
K3a/K3b on committed updates unaffected); fresh returns barely move (delivery lag ~0.07s), only
the stale-straggler recordings (where the divergence lived) are corrected. Shared by oort+refl
(one oort aggregator). Fallbacks: `WALL_SEND − dispatch`, then `recv − dispatch`, when a client
stamp is absent. Telemetry `[CLIENT_DUR_STRIP]` (intrinsic vs delivery_lag per stale record).
**SINGLE-SOURCED:** the definition lives in `flame/mode/horizontal/client_duration.py`
(`real_client_task_train_duration`); oort's `_real_client_task_train_duration` is now a thin
wrapper, and asyncfl/felix (line ~951, was `recv−dispatch`), syncfl/feddance+fedavg (line ~519,
was `wall_lag_s`), and fwdllm all call it (each falling back to its prior agg-anchored measure
only when a client stamp is missing). syncfl's `_real_task_dur` (`WALL_SEND−dispatch`) U6
barrier anchor (§6.u6) is a SEPARATE quantity — left untouched. Guards:
`test_client_duration.py` (helper contract), `test_sync_sim_ordering.py::TestStaleTrainerPropsRecorded`
(oort path + fallback). **VALIDATED Jun 24 3h:** CLIENT_DUR_STRIP confirms real `intrinsic_s≈D`,
`delivery_lag≈0.02s` (the +8s lag the fix was scoped against was already negligible in this run);
refl throughput family CLOSED (K2 rel 0.047), felix no regression at 3h. For OORT it exposed the
§S.pacer SIM-side root (real `pref` cleaned to 8.58, sim sits at 10.14) — see §S.pacer.

### §S.pacer  oort — dynamic `round_threshold` PACER was an UNFAITHFUL port (ROOT-CAUSED + FIXED Jun 24; validate next run)
The Oort speed penalty `system_util = min(1,(pref/duration)^α)` only binds when a candidate's
duration exceeds `pref = round_preferred_duration`, the **`round_threshold`-th percentile** of
candidate durations (`oort.py:496`). `round_threshold` is **DYNAMIC**, set by `pacer()`. **The
flame base `OortSelector.pacer()` was an UNFAITHFUL port** of reference Oort
(`third_party/Oort/oort/oort.py:184-199`): the reference makes TWO symmetric moves keyed on the
exploited-utility trend over the last two `pacer_step` windows — a FLAT plateau (`|Δ| ≤ 0.1·last`)
RELAXES (`round_threshold += pacer_delta`), a SHARP change (`|Δ| ≥ 5·last`) TIGHTENS
(`round_threshold = max(pacer_delta, round_threshold − pacer_delta)`). flame's port instead raised
on **ANY** dip (`last > curr`) and had **NO** decrease branch → a monotonic ratchet to 100,
hypersensitive to per-round utility noise. Because the noise timing/magnitude differs between sim
and real (different stochastic utility trajectories), the ratchet drifted to different levels and,
never recovering, the divergence COMPOUNDED. Jun 24 3h: pref grew in both modes (real Q1→Q4
4.53→15.19, sim 4.31→19.49) but **sim faster** — Q3 sim 11.48/real 9.09, Q4 sim 19.49/real 15.19,
sim disabling the penalty (`pref=99999`) in 31 late rounds (real 0). So sim bound 73.6%/round vs
real 94.1% → ~1.5s-slower max-of-K (10.5 vs 8.99) → +1.2s/rd advance (K3b −1.22s) → fewer rounds
(K2). **Tells it's the pacer not the input:** candidate-pool static speed (12.13), selected MEAN
(8.04/8.16, A2c PASS), residence (PASS) all match; only the per-round `pref` percentile diverged,
and it's `round_threshold` (the pacer) that sets the percentile. CLIENT_DUR_STRIP confirms the
§S.dur input is clean (intrinsic≈D). **FIX LANDED:** faithful reference port in
`OortSelector.pacer(round)` — both branches on the 0.1 / 5× bands, keyed on the current `round`
(reference's `training_round`) with a `pacer_step > 0` guard. The `REFLOortSelector.pacer`
override (which was ALREADY faithful — flat/sharp, 0.1/5×) is REMOVED so oort+refl share the one
reference-matching base. `round_threshold` is added to the selection telemetry `extra` (was only
inferable from `pref`). Guard `tests/selector/test_oort_selector.py::TestPacerFidelity`.
**Selector-side, both modes identically** (changes the BASELINE toward the reference, not parity-
tuning) — the hypothesis is that a self-correcting symmetric controller, fed matched utility
distributions (F1-3 pooled KS 0.037), keeps sim/real `round_threshold` tracking instead of
ratcheting apart. **VALIDATE on the next 3h run** via the new `round_threshold` field (binned by
run-fraction: no monotonic-to-100, no sim 99999), Sd binding real≈sim, K3b/K2/K8/U2 close. Do NOT
tune `pacer_delta`/`pacer_step` or touch the §S.dur input. *(Earlier read floated a §3.async
"re-parameterize by wall-time" generalization — SUPERSEDED: the reference pacer is also per-round
and self-corrects, so the gap was the unfaithful port, not the round-indexing.)*

**Reference cross-check (Jun 24):** the fix matches BOTH references — `third_party/Oort/oort/oort.py`
:184-199 AND the `third_party/REFL/thirdparty/oort/oort.py`:176-201 fork are BYTE-IDENTICAL on the
pacer (same 0.1 / 5× bands, flat→raise / sharp→lower, `training_round`-keyed; REFL only differs in
`round_threshold=30` default, already config'd via D3). So refl inheriting the faithful base is
correct against its OWN reference, not just Oort's.

**felix (`AsyncOortSelector`) — SAME bug, ALSO fixed (validate next run).** felix is a separate
class (no async-Oort reference) but the pacer is the SAME concept, so it must match the reference's
two-branch logic. Two defects: (1) the identical flat/sharp bug (raised on any dip, no decrease
branch → monotonic ratchet to 100, turning the speed penalty OFF — felix only "passed" because the
penalty was thus rendered largely inert, see §3.async); (2) felix has TWO selector hands (train +
eval) on ONE instance and `pacer()` fired in `_handle_send_state` for BOTH, but `self.round` /
`exploitation_util_history` advance only on TRAIN, so an eval call re-ran the pacer off a STALE
round. **Fix:** faithful two-branch `AsyncOortSelector.pacer()` (kept in-class) + the call is now
TRAIN-GATED (matches the reference's *training*-selector pacer). Guard
`TestPacerFidelity::test_async_oort_pacer_faithful`. ⚠ **This changes felix's round_threshold
dynamics** — the penalty is now active (oscillates near its start) instead of ratcheting to off —
so felix's 46/46 MUST be re-validated on the next run; if it shifts, the old pass partly relied on
the inert penalty (a real finding, not a clean regression).

### §6.u6  syncfl real U6 barrier-anchor (feddance/fedavg; LANDED Jun 22)
`update_visibility_lag_s` on the REAL strict-sync path was wrong: `_update_visibility_lag`
evaluated `committed = datetime.now()` **per-message inside the recv loop**, so it measured
arrival→ingestion (~0.02s/update) — NOT the barrier wait. A strict barrier applies all K at ONE
post-loop instant (`optimizer.do`), so an early finisher's true visibility lag = barrier − its
own completion. Real updates do physically arrive spread (`[MSG_ARRIVAL]` 33→48s within a round,
`queue_depth=0`); the per-message metric was just blind to it (sim 15.5s vs real 0.02s = U6
FAIL). **Fix:** real anchors on the single round barrier — `_barrier_anchored_lags(durs)` returns
`max_dur − dur_i` with `dur = WALL_SEND_TS − dispatch` (client task-train duration, the
dispatch-relative completion matching sim's `sct`); sim is unchanged (`vclock−sct`, vclock is
already advanced to the barrier). **Why only feddance, not oort/refl:** oort/refl use the
oort overlay's STREAMING commit — `_oort_sim_recv` pops in `sct` order and `_advance_sim_clock`
tracks each pop, so each update commits at `vclock≈own sct` → lag≈0 in BOTH modes (real commits
first-K-to-arrive near arrival too). The barrier wait only exists for a baseline that waits for
the slowest of its cohort (feddance). So oort's per-message helper stays correct and is left
untouched; only `syncfl._aggregate_weights` (feddance + fedavg base) is barrier-anchored.
**Validated against STORED real logs (no rerun):** recomputed lag mean 15.65/min 0/max 53 ≈ sim
15.48/0/52. Pure telemetry (no dynamics/staleness effect). Guard:
`test_sync_sim_ordering.py::test_barrier_anchored_lags_*`. Pending: confirming real feddance rerun.

## §5  Checker corrections (stochastic / observability classes)
Once dynamics match, some residual FAILs were the checker enforcing exact identity on
quantities a stochastic/in-memory sim can't reproduce (tell: byte-identical across runs despite
large dynamics changes). All principled, guarded, append-only; a future *deterministic*
selector still gets exact enforcement via `DETERMINISTIC_SELECTORS`.
- **P1 aggregation_sequence** → WARN for stochastic (S2 participation is the enforced invariant).
- **S2 participation** → enforce participation BY SPEED CLASS (`speed_class_tvd`, registry
  `speed_class`), NOT per-trainer identity, for stochastic selectors (Jun 24). The per-trainer
  `matched_count_ks` is path-dependent: a stochastic weighted-exploit selector builds a persistent
  core whose SIZE/concentration/speed-composition match but whose individual MEMBERS diverge
  (refl: 63 of ~120 shared, speed-matched). What the POLICY fixes is the speed-class distribution
  (refl TVD 0.026); per-SECOND buckets re-expose the identity noise (TVD 0.187, sign-alternating),
  so coarse `speed_class` is the right granularity. `matched_count_ks` kept as diagnostic;
  `DETERMINISTIC_SELECTORS` still get exact identity enforcement. Tell it's identity-not-bias:
  A2c/K8 pass and the mode-specific cores are speed-matched.
- **F1-3 utility** → pooled KS (per-trainer KS=1.0 was mechanical for n≤2; means identical).
- **phase_mqtt_fetch** → DIAG (real wall-clock `channel.recv()` wait in BOTH modes, not an
  in-mem shortcut; dispatch cadence/aggregator-side overhead differ real vs sim and aren't
  vclock-modeled in either, so deliberately off the virtual clock and non-gating).
- **trainer_speed / eligible_speed / selection_bias** → integer-grid / metadata-pool.

## Discrepancy ledger — flame vs reference Oort
flame has ONE `OortSelector` for both `oort` (matches `third_party/Oort`) and `refl` (matches
`third_party/REFL` fork). The references differ on defaults, so each baseline's knobs are
config-driven (`selector.kwargs`), defaulting to the Oort paper (`OORT_PAPER_DEFAULTS`) with
refl overriding.

| # | discrepancy | resolution |
|---|---|---|
| D1 | `pref` not sorted | FIXED (sort added) — port bug; validated on oort |
| D2 | stat-utility not normalized/clipped | FIXED (`scoring.oort_normalize_reward`, config) |
| D3 | `round_threshold` | config: oort/felix=10 (paper), refl=30 (fork) |
| D4 | `cut_off_util` + cutoff-index | FIXED: config (0.7 paper / 0.05 refl); index thresholds the exploit-boundary score (was inert) |
| D5 | temporal time-base | **SUPERSEDED by D7 (Jun 23) for oort+refl.** Jun-16 fix stamped at selection (`_record_last_selected_round`) to dodge commit-order dependence; D7 found that was still unfaithful (reference keys the UCB term on last-RECEIPT round + registration-init, not dispatch round) and replaced it with `PROP_LAST_RETURNED_ROUND` — the last-selected machinery is removed. felix (`AsyncOortSelector`, separate class) keeps its own last-selected path, still deferred. |
| D6 | `clip_bound` | config: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now fork's cut_off_util-weighted `np.random.choice` |
| D7 | UCB temporal-uncertainty `time_stamp` | **FIXED (Jun 23, §S.temporal).** Reference Oort+REFL: `sc += sqrt(0.1·log(round)/time_stamp)`, `time_stamp=self.epoch` (agg round of last RECEIPT), init at registration → never None, always contributes, up-weights under-selected/slower-returning clients. flame bug: refl's term was DEAD (0/7513) — `refl_oort.select()` let it divide by a None `time_stamp`; oort used last-SELECTED (dispatch round) with a None→0 guard. **Fix:** `PROP_LAST_RETURNED_ROUND` stamped at every receipt (fresh+stale) in oort/top_aggregator = agg round; selector reads it, registration-init lazy to current round; both oort+refl. `enable_temporal` kwarg (default True; False = ablation only). The legacy last-SELECTED machinery (`_record_last_selected_round`, D5) is REMOVED from OortSelector (no baseline used it); D5's MODEL_VERSION value was for staleness, not this UCB term. felix AsyncOortSelector is a separate class — untouched. |
| D8 | `pacer()` round_threshold adaptation | **FIXED (Jun 24, §S.pacer).** Reference (`oort.py:184-199`) makes TWO symmetric moves on the exploited-utility trend: FLAT `|Δ|≤0.1·last` → `round_threshold += pacer_delta`, SHARP `|Δ|≥5·last` → `round_threshold = max(pacer_delta, −pacer_delta)`, keyed on `training_round`. flame's base `OortSelector.pacer()` raised on ANY dip (`last > curr`) with NO decrease branch → monotonic ratchet to 100, noise-sensitive → sim/real `round_threshold` diverged & back-half-compounded (the oort §S.pacer root). **Fix:** faithful both-branch port keyed on the current round, `pacer_step>0` guard; `REFLOortSelector.pacer` override (already faithful) REMOVED so oort+refl share the base; `round_threshold` added to selection telemetry. Guard `TestPacerFidelity`. **Cross-checked vs the REFL fork too** (`third_party/REFL/thirdparty/oort/oort.py`:176-201, byte-identical pacer, only round_threshold=30 default differs). **felix `AsyncOortSelector` (separate class) had the SAME bug + fired the pacer on its eval hand off a stale round → ALSO fixed** (faithful two-branch, train-gated; `test_async_oort_pacer_faithful`); changes felix dynamics → re-validate its 46/46 next run. |

---

## §F  FwdLLM extension -- variance-gated dynamic-K (forward-gradient FL)

The method above (§1-§5) is example-agnostic; this is fwdllm's rung catalog. Rungs not
redefined here are inherited from §2 unchanged. `[NEW]` = to implement. fwdllm's live real↔sim parity
status/fixes are `examples/fwdllm/simulate_fwdllm.md` (this section is the rung reference it points at);
its build plan (staged implementation, files, exit criteria, design decisions) is
`examples/fwdllm/FWDLLM_DESIGN.md`.

### §F.1  How FwdLLM differs (drives every new rung)
- **Aggregates GRADIENTS (JVPs), not weights.** Trainers send forward-gradient estimates; the
  aggregator accumulates them into `grad_pool` and applies a server-LR SGD step at commit.
  Gradient **values** depend on real GPU compute (run for real in sim) -> mode-invariant given
  identical input + perturbation seed. What differs across modes is **which** gradients arrive,
  **in what order**, against **which model version** = clock + selection + ordering fidelity.
  This is what makes the §2 ladder applicable to fwdllm at all.
- **Commit cadence is ENDOGENOUS (variance-gated dynamic-K).** At each `_agg_goal` (=K) boundary,
  `aggregate()` computes `var`; `var <= var_threshold` -> **commit** (server step, eval,
  `data_id += 1`, `model_version += 1`, clear `cached_v`); else **roll back**, push grads to
  `cached_v`, `iteration_per_data_id += 1`, **retry the same data_id**;
  `max_iterations_per_data_id` force-commits despite a failed variance gate. So
  updates-per-`model_version` is a **random variable** of the gradient-variance trajectory -- the
  `model_version` clock is not a fixed function of update count. A round is the outer loop over
  data bins; `_round += 1` only when `data_id == total_data_bins` (=150).
- **Dynamic K and C.** `DynamicKCController.step(metrics)` may change K (`_agg_goal`) and C
  (concurrency) from observed metrics (var-pass ratio, eligible-ends) after each agg-goal cycle.
- **Eval per variance-pass.** `eval_model()` runs on **every** committed data_id, not a fixed
  schedule; its modeled delay must be stamped separately (a stale-eval `sct` past-dates the clock).
- **Progress axis is `data_id`** (committed variance passes), not raw update count -- all
  throughput/terminal rungs re-key to `data_id`.

> **Crux:** in async_cifar10 clock and commit-count are loosely coupled; in fwdllm the commit
> (model-version) cadence is a *feedback function of gradient variance over the accumulated pool*.
> The sim must reproduce not just **when** updates arrive (clock) but the **variance trajectory**
> gating each commit. Variance is mode-invariant **iff** the contributing set + order +
> model-version of gradients matches -- reducing fwdllm parity back to clock + selection + ordering
> parity, **plus** a new variance-cadence verification layer (§F.4).

### §F.2  Baseline matrix (fwdllm)
Filled from the landed `examples/fwdllm/expt_scripts/*_n10_smoke.yaml`. Maps onto the
async_cifar10 availability taxonomy (`avail_select_filter` / `proactive_inflight_evict`,
Unavailability §Baseline matrix).

| baseline | sync/async | selector | agg | tracking_mode / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify` (3-tier, mobiperf_3st_50) | -- | 3 | disabled (fixed K/C) |
| **fwdllm** | sync | `random` | fedavg | `default` (unaware) | per-round | 10 (=c; all selected required) | -- |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` (`_metadata`, mobiperf_2st) | per-iteration (`reselect_each_iteration=True`) | 2 | -- |

Decides scope: Stage 3 oort rungs (Sx/Sd/S2) run only for **fluxtune** (`async_oort`); the two
`random`-selector baselines skip them. Stage 5 sync-barrier rungs run for **fwdllm**/**fwdllm_plus**
(sync); fluxtune streams per-message. `client_notify` (fluxtune) is async_cifar10's deferred Stage-H
tracking model -- see simulate_fwdllm.md decision D1.

### §F.3  Modified rungs (async_cifar10 meaning -> fwdllm redefinition)
| ID | async_cifar10 | FwdLLM redefinition |
|---|---|---|
| **K3a** | K-th fastest async commit advance | advance of `model_version` **per committed data_id** (clock delta between successive **variance passes**) |
| **K3b** | overhead residual ~= 0 | same, measured on the **variance-pass** boundary |
| **K2** | model versions / vsec | **committed data_ids / vsec** (throughput of *successful* variance passes) |
| **U3** | version gap at commit | gap of each contributing gradient vs `model_version` at the cycle it lands -- spread across **multiple iterations per data_id** |
| **K8 / U2** | @ matched model_version | @ matched **data_id** (the meaningful progress axis) |

### §F.4  New rungs -- the variance-cadence layer (the fwdllm prize)
Append-only, deps let the engine localize. Never fix an EMERGENT rung directly (§1): walk down to
the lowest variance-cadence rung whose inputs are matched.

**Stage 6' -- Variance-gated aggregation cadence** (dep Stage 5 ordering + Stage 1 clock)

| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| **V1** `[NEW]` | iterations-per-data_id dist (realized dynamic K) | MECHANISM/DIST | # accumulation cycles to pass variance => contributing set/order diverged | U5,U4 |
| **V2** `[NEW]` | per-cycle `var` trajectory (at each agg-goal) | MECHANISM/DIST | variance *signal* diverges with matched inputs => grad-pool composition/order differs | V1 |
| **V3** `[NEW]` | `cached_v` pool size over time | MECHANISM/DIAG | rollback/cache bookkeeping diverges | V1 |
| **V4** `[NEW]` | force-commit freq (`max_iterations_per_data_id` bypass rate) | MECHANISM/DIST | cap hit at a different rate => chronic variance divergence | V1 |
| **V5** `[NEW]` | variance-pass ratio per window | EMERGENT/DIST | rollup feeding DynamicKC | V1,V2 |

**Stage 3' -- Dynamic K/C trajectory** (dep Stage 3 selection + V5)

| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| **DK1** `[NEW]` | K (`_agg_goal`) trajectory | MECHANISM/DIST | DynamicKC sees different metrics => K diverges (feeds back into cadence) | V5 |
| **DK2** `[NEW]` | C (`dynamic_c`) trajectory | MECHANISM/DIST | concurrency target diverges | V5,S3/4 |
| **DK3** `[NEW]` | eligible-ends-count metric fed to policy | CONTROL/DIST | the policy *input* differs (fix input, not policy) | A2 |

**Stage 7' -- Forward-gradient quality**

| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| **G1** `[NEW]` | per-update grad/JVP norm or SNR dist | EMERGENT/DIST | grad *quality* diverges (should be ~mode-invariant; FAIL = perturbation seed/order leaked) | S2,T_gpu |
| **G2** `[NEW]` | grad_pool size at commit (realized contributions) | EMERGENT/DIST | rollup of V1 x K | V1,DK1 |

> **Decomposition:** `K2`(throughput)x but `K3a`(per-pass advance)ok -> clock fine, commit count
> diverged -> walk to V1/V5. `V1`x + `V2`ok-given-matched-input -> the *inputs* to variance differ
> -> walk to U5/S2. `V2`x with V1 inputs matched -> a true grad-pool accumulation-order bug.
> **DynamicKC coupling:** validate DK3 (policy *input*) before DK1/DK2 -- a diverging input means
> fix the metric, not the policy (CONTROL before MECHANISM). `var_threshold` /
> `max_iterations_per_data_id` are **baseline-defining config knobs, not parity levers** -- a
> cadence gap is always an upstream set/order/clock divergence.
