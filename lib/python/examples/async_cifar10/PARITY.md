# Real / Sim Parity — Methodical Causal Ladder

Living document for the async_cifar10 parity checker.
Kept in sync with `scripts/parity/checks.py` (check functions),
`scripts/parity/report.py` (stage grouping + verdict), and the pytest suite.

**Real/sim comparator — give the two run dirs, get a report JSON:**
```bash
cd lib/python/examples/async_cifar10
# single baseline: point at the real + sim run dirs
PYTHONIOENCODING=utf-8 python scripts/parity_check.py \
  --real experiments/<real_run_dir> \
  --sim  experiments/<sim_run_dir> \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity_<baseline>_<tag>.json

# all baselines at once: auto-discovers the latest real/sim pair per tag
PYTHONIOENCODING=utf-8 python scripts/parity_check.py --batch \
  --experiments-dir experiments --baselines felix oort refl feddance \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity.json
```
`--budget-s` = the run's `--runtime-s`. Add `--lenient` to demote DIST fails to
warnings; prints a stage-grouped report + root-cause banner.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Launch runs** (node-agnostic; any baselines/mode/duration on any machine):
```bash
bash scripts/debug_run.sh --baselines 'oort refl' --runtime-s 3600 --mode both
```
Reads `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml` (every
baseline × sim/real), seeds real+sim identically (`SEED=1234`, `SEED=none` to
disable), and applies the per-baseline sim fixes. Split across machines by
passing different `--baselines`.

**Readiness/regression tests** (no cluster; run under lib/python with `dg_flame`):
`pytest tests/mode/ tests/selector/test_oort_selector.py tests/sim/
examples/async_cifar10/scripts/parity/` — guards baseline wiring, the in-memory
cache, serialize-once, sim ordering (barrier/residence/carry-over), the overhead
model, deterministic seeding, the pass/total scoreboard, and every checker rung.
Last green: **147 pass / 7 skip**.

---

## Workflow policy: minimize time & runs to parity

The objective is parity in the fewest wall-hours and cluster runs. Rules:

1. **Run real only when needed.** Real is the reference; once a baseline's real
   run is admissible and stored, re-run real only when a change affects the *real*
   path. Sim-only changes validate against the stored real dir.
2. **Over-instrument telemetry deliberately.** Emit more signals than any single
   check needs if they capture runtime behavior and speed up root-causing — cheap
   to log, expensive to re-run for. (E.g. per-round `inflight_residence`, the
   `SIM_CLOCK_DIAG` past-dating counters localized oort + felix from stored runs.)
3. **Root-cause per baseline, then scope the fix to its blast radius.** Common
   cause across baselines → fix once. Independent fix that cannot affect others →
   land it. Fix that *could* perturb another baseline → serialize it (one baseline
   per run round) so attribution stays clean.
4. **A fix touching the real path → re-run both** real and sim; a sim-only fix →
   re-run sim, reuse stored real.
5. **Shortest run that exhibits the issue.** Don't default to 3–4h. Use the
   minimum duration that surfaces the check under test; reserve long runs for
   `C1`/`C2` convergence or round-count-compounding residuals only.
6. **Crisp code comments** — one sentence at most; let tests document behavior.
7. **Context-free variable names** (see Naming discipline below).

---

## Doc policy: ONE status section, not one per run

This doc used to grow a new `## Status (date — ...)` section per check-in,
left in place "for the record." That's bloat — old scoreboards are
regenerable from `parity_check.py --batch` and old hypothesis tables just
restate what the next rerun already overwrote. **From now on: there is
exactly one `## Status` section, dated to the most recent rerun, fully
replacing the previous one.** Durable lessons that don't expire with the next
rerun (triage rules, dead ends, validated mechanisms) live in their own named
sections below status and get *updated in place*, not appended to.

---

## Status (Jun 17 — refl 1h rerun complete: D5 VALIDATED; speed-tail/selection-mix residual)

Baselines not yet rerun (felix, feddance) still reference the 3h seeded runs:
`experiments/run_20260616_0{0,2,5}*_real`, sim dirs `run_20260616_110530_..._felix`
and `_131048_..._feddance`. **oort 1h rerun** (unchanged this session): real=`run_20260616_175804_dbg_oort_n300_alpha0.1_syn0_stream_real`,
sim=`run_20260616_174557_dbg_oort_n300_alpha0.1_syn0_stream_sim`, report
`experiments/parity_oort_20260616_1h.json`. **refl 1h rerun (NEW, this session)**:
real=`run_20260616_223801_dbg_refl_n300_alpha0.1_syn0_stream_real`,
sim=`run_20260616_223807_dbg_refl_n300_alpha0.1_syn0_stream_sim`, report
`experiments/parity_refl_20260616_1h.json` (`--budget-s 3600 --agg-goal 10`). The
crash-fixed selector ran clean to completion; **D5 confirmed** (`temporal KS=0.0`,
`participation` cleared, `Sr` carry-over/residence matched).

| baseline | score | lowest broken rung | root cause | status |
|---|---|---|---|---|
| **refl** | **38/44** *(1h rerun, post D5 + crash fix)* | `overhead_residual` (S1) + `eligibility` (S2) | **NOT a mechanism bug — a thinner slow-speed tail in sim.** P3 max-of-K speed sim=7.75s vs real=8.64s; `trainer_speed` max sim=21.0s vs real=29.17s (p99 matches: 9.0/10.05, grid_KS=0.087). The modeled speed model lacks real's slow tail; per-round max-of-K picks it up → sim advance 1.39 vs real 1.59 (`K3b` rel=0.121, **implied per-commit overhead=0.019s≈0** so NOT missing overhead) → sim 2574 rounds vs real 2077 → `K2`/`K8`/`U2` ~12% downstream. `A2` num_eligible KS=0.448 but means match 1.1% (247.3/244.5) with `A2b` composition KS=0.002 + `A3` time-base passing → point-mass/shape artifact. | **D5 VALIDATED** (`temporal KS=0.0`; `participation` no longer root; `Sr` `in_flight_after` real=52.7/sim=54.45, `committed_fresh` 10=10, `residence_rounds` 4.06/4.19). **Crash fix validated** (selector ran 2574 rounds, no `AttributeError`). Remaining residual is the stochastic speed-tail/selection-mix — **same class as feddance A2c**, not a fresh break. Deprioritized below oort carry-over. |
| **felix** | **33/42** | `overhead_residual`/`overlap_factor` | overlap-model collapse: sim 12.7× vs real 6.8×, advance 2.25 vs real 4.02. **Diagnostic mined this session (`run_20260616_110530` sim):** the bottleneck is **past-dating, not re-dispatch spacing** — `pastdated_commits=67401/92000` (**73%**), mean past-date gap **~626s**, max **9659s** (≈ whole run); `vclock_lead_over_buf` mean 280s. A population of slow-trainer updates commits thousands of vclock-s LATE with **zero clock advance** (`vclock=max(vclock,sct)` and sct≪vclock), packing many rounds into no time → overlap inflates. This is why both `simRedispatchGapSeconds` scalars (0, 0.6) were inert — they space same-trainer re-dispatch, not the cross-trainer past-dated burst. **SEED FOUND**: the gate's expected completion used the running-mean budget (~12s) as the unseen-trainer default — NOT a lower bound, so it overshoots fast trainers (actual ~5s), the gate doesn't hold, and the clock laps them; once one past-dated entry lands, `buffered_min` stays small and the gate (`min_stuck<buffered_min`) is inert forever. | **SEED FIX LANDED (sim-only)**: unseen-trainer default → running MINIMUM (`_sim_budget_min`), a true lower bound, so the gate never laps an unseen trainer. Tests: `TestExpectedCompletionLowerBound`. **Awaiting SHORT sim-only run** (reuse stored real) to confirm past-dating drops + overlap→~6.8×. NB removes the seed (~300 first-appearances); if past-dating persists there are other seeds (round-1 transient / re-dispatch) → then design buffer-aging pacing. |
| **oort** | **36/44** *(1h rerun, pre-fix; rerun pending)* | `residence` (carry-over) + `preferred_duration` | **ROOT FOUND + FIXED this session.** Carry-over `inflight_after` sim=0.82 vs real=4.24. Telemetry mining of the existing sim run (`run_20260616_174557`) localized it: per-round `in_flight_after` **starts at 4.31 (= real's 4.24) in decile 0 and decays to ~0 by mid-run** — not a uniform absorb. Cause: the carry-over gate holds prior-round stragglers with `sct > vclock_round_start`, but `vclock_round_start` was re-read as `self._vclock.now` **inside each `_oort_sim_recv` call**, and the block-for-K-fresh retry loop spawns a fresh generator per pass *after* the clock advanced → the threshold **creeps forward mid-round**, committing stragglers that should carry. Early rounds (few retries) ≈ no creep → match real; later rounds creep → drain. `preferred_duration`/`selection_detail`/`overhead_residual` are all downstream of this. | **FIXED**: pin the threshold to the round-start vclock, captured once per round in `_aggregate_weights` (`self._round_start_vclock`); `_oort_sim_recv` reads it via `getattr(..., self._vclock.now)`. Gated behind `simInflightCarryover` (oort-only) → **refl byte-neutral** (refl uses `simInflightResidence`, carryover=False, gate is dead code). Guard added: `TestSimInflightCarryover::test_pinned_threshold_holds_straggler_across_retry`. Suite 153 pass/7 skip. **Awaiting 1h rerun** (target `inflight_after`→~4). |
| **feddance** | **41/43** | `selection_bias` (A2c) + downstream `convergence_loss` | not a regression: the already-known deferred selection-mix bias (sim picks ~0.4–0.5s-faster trainers) tipped just over its .2/.15 bars on this seed, same magnitude as always | still open, deprioritized below refl/oort |

**Landed this session.**
- **felix past-dating SEED fix (NEW, sim-only)**: the gate's unseen-trainer
  expected-completion default was the running mean (overshoots fast trainers, laps
  them, seeds the past-dated backlog that makes the gate inert); changed to the
  running minimum (true lower bound). Guard `TestExpectedCompletionLowerBound`.
  Awaiting a short sim-only run to confirm the cascade breaks.
- **oort carry-over decay FIXED (NEW)**: pinned the §4.9 carry-over threshold to
  the round-start vclock so the block-for-K-fresh retry loop can't creep it forward
  mid-round. Root-caused by mining per-round `in_flight_after` from the existing sim
  run (decile 0 = 4.31 ≈ real, decaying to 0). Oort-gated, refl-neutral, guarded.
- **refl 1h rerun analysis (NEW)**: D5 + crash fix both validated against a clean
  2574-round sim run. refl's remaining residual localized to the speed-tail/
  selection-mix (see table + durable lesson below); reclassified as the feddance
  A2c class, deprioritized below oort.
- **oort block-for-K-fresh** (prior session, validated): `committed_fresh` sim→10 = real ✓. Mechanism: removed `while not self.simulated` gate from `_aggregate_weights`'s second poll loop; retry calls `_oort_sim_recv` (re-probes persistent `SimReorderBuffer`); `progressed` flag stops spinning.
- **refl selector crash fix** (prior session, now validated): `REFLOortSelector` used `self.round` (5 sites: guard at `select()` entry, assignment at `select()` exit, pacer condition, pacer update, pacer close). Parent's D5 rename to `self._last_selection_round` left these as dangling attrs; all updated.

### Next implementation steps, in priority order

> **oort carry-over (residence) is the open root cause.** The block-for-K-fresh
> and D5 fixes are validated. The try/finally §4.9 is wired but block-for-K-fresh's
> extended grace window is likely absorbing the trainers that should carry over,
> so they resolve as fresh rather than crossing the round boundary as stragglers.
> **refl is now validated** (D5 + crash fix); its residual is stochastic
> speed-tail/selection-mix (feddance class), not a fresh mechanism break.
> felix and feddance are unchanged.

1. **oort carry-over fix — LANDED this session, awaiting rerun.** Root was the
   carry-over threshold creeping forward across block-for-K-fresh retries (per-call
   `self._vclock.now`); fixed by pinning to the round-start vclock. Next action is
   just the rerun + parity check (target `inflight_after` sim→~4):
   `bash scripts/debug_run.sh --baselines oort --runtime-s 3600 --mode both`.
   - **oort**: expect `inflight_after` sim → ~4, `residence`/`preferred_duration`/
     `selection_bias` clear (downstream of the carry-over gap once fixed).
     Temporal term timing-stable (D5).
2. **refl speed-tail (deprioritized, feddance class).** No mechanism bug:
   `K3b`/`K2`/`K8`/`U2` all trace to sim's modeled speed model lacking real's
   slow tail (max-of-K sim 7.75 vs real 8.64; `trainer_speed` max 21.0 vs 29.17).
   Either (a) accept as stochastic selection-mix and let `K3b` ride its EXACT bar,
   or (b) widen the speed-model tail to reproduce real's slow draws. Sequence
   after oort + felix; close alongside feddance A2c (same class).
3. **felix D5 (deferred decision).** Same write-timing bug at
   `asyncfl/top_aggregator.py:516`; fix = stamp at selection in
   `AsyncOortSelector` (mind `round_nudge_type`: `last_train` reads this prop,
   `last_eval` reads `PROP_LAST_EVAL_ROUND` instead — confirm which felix uses
   before touching). Sequence after felix's overlap-model work to keep that
   attributable, or land standalone since it's orthogonal.
4. **felix overlap-model — SEED FIX LANDED this session, awaiting short run.**
   Root (mined `run_20260616_110530` sim): 74% past-dated commits, `gate_holds=0`
   all run. The gate (`min_stuck<buffered_min`) goes inert once a past-dated backlog
   forms (backlog keeps `buffered_min` small ≤ any future straggler). SEED: the
   unseen-trainer expected-completion default was the running MEAN (~12s), which
   overshoots fast trainers → gate laps them → first past-dated entry → cascade.
   Fix landed: default → running MINIMUM (true lower bound). **Next:** SHORT
   sim-only felix run (reuse stored real, rule iv/v) → check `pastdated_commits`
   frac and `overlap_factor`→~6.8×. If still high, other seeds remain (round-1
   transient, re-dispatch) → then buffer-aging pacing (mind: no overhead-on-clock).
5. **feddance A2c/C2 selection-mix bias.** Lowest priority — close once 1–4
   land, since it's a known, bounded, already-passing-at-the-margin gap, not
   a fresh break.

### Durable lessons (kept; update in place, don't append)

- **`Sdet` triage rule.** `eligible_match≈0` **with matching aggregates** =
  stochastic-class, PASS as-is; `eligible_match≈0` **with a diverging clock**
  = genuine, fix the clock; `eligible_match` high but `decision_match≈0` = the
  **values** diverge, not the set. Localized all four baselines' roots.
- **`participation`/S2 vs per-round Jaccard.** Round-to-round draw mismatch
  (`Sdet`/`S1`) is expected for a stochastic selector and not itself a bug.
  But a **systematic per-trainer skew over the full run** (S2 `matched_count_ks`
  large, `max_diff` not averaging out by run's end) is not noise — it means
  something is biasing *which* trainers win, not just *when* (this is how
  refl's `participation` finding was distinguished from normal stochastic
  variance — see Status table above).
- **`P3 mean_overhead` is wall-capture, not a speed-model bug**, when
  sub-second, opposite-sign across baselines, and `grid_KS`/`training_delay_s`
  metadata match — only trust a `P3` FAIL when `grid_KS` also fails.
- **`K3b` ≠ "missing overhead" when `implied_per_commit_overhead_s`≈0.** The check
  is *named* overhead_residual, but its residual is `real_advance − sim_advance`,
  which is also moved by the per-round **max-of-K speed** order statistic. If the
  implied per-commit overhead is sub-0.05s yet `K3b` fails, read the K3a `max_speed`
  line: a sim/real gap there (refl: sim 7.75 vs real 8.64; `trainer_speed` max 21 vs
  29 with p99 matching) means the advance gap is a **thinner sim speed-tail / selection
  mix**, not omitted overhead. Do NOT add a `simCommitOverheadSeconds` scalar to chase
  it (dead end); fix the speed-tail model or accept it as the feddance A2c class.
- **Run length matters.** 45min exercises every mechanism check but isn't
  enough for `C1`/`C2` convergence sign-off or to catch round-count-compounding
  clock residuals / low-frequency eligibility-shape drift (refl's 3h-only
  `K2` regression, surfaced only at 3h despite a clean 45min pass).
- **Checker-side fixes validate instantly against stored run dirs; only sim
  *mechanism* changes need a new cluster rerun.** Land and verify all checker
  corrections against existing dirs first, then batch mechanism changes into
  one rerun.

### Dead ends — do NOT retry

- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- `version_at(sct)` staleness relabel (fedbuff consumes the *real* number; inert).
- Adding `mqtt_fetch` (~57 s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` for felix (sim over-overlaps; the gap is a real mechanism).
- `simRedispatchGapSeconds=0.6` for felix — tested Jun 16, no measurable effect; don't retune this scalar further, go to the buffer-aging model.
- Tuning sim to a *wrong* real, or any scalar fudge where a mechanism is called for.
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0), the
  felix post-compute leg as a "bug" (it's serial-aggregator scheduling), per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent by nature).
- Expecting seeding to align per-round sets: `Sdet eligible_match`≈0 is *expected*
  for stochastic selectors whose inputs (clock-indexed availability) drift; judge the
  aggregates (S2/participation), not the per-round draw. Only a *diverging clock*
  or a *systematic per-trainer skew* (not round noise) makes it a genuine bug.
- A scalar fudge for the `P3 mean_overhead` ~1 s offset (wall-capture, opposite signs
  across baselines, grid-passing). Widen the bar or score on `training_delay_s`; don't
  bias the speed model to chase it.

### Naming discipline (instruction — apply when touching baseline code)

State and variable names must be **context-free**: a reader should not need the
surrounding code to know what a name refers to. Round/version/time confusion has
caused real bugs here (D5), so:
- A name ending `_round` is a **round index** (int), never a timestamp. Use
  `_ts`/`_time_s` for times. Don't name a round int `_stamp`/`timestamp`.
- Qualify *whose* round: aggregator's global counter vs the selector's last-run
  round vs a per-trainer property are different things. (Done for oort+refl:
  selector `self.round` → `self._last_selection_round`; the per-trainer property
  read is `end_last_selection_round`. **Deferred:** the base aggregator
  `self._round` → `self._agg_round` — it's a `TopAggregator` attribute, so that's
  a dedicated all-baseline pass, not scopeable to one baseline.)
- A local should say *what it is*, not just its type-shape — `trainer_model_version`
  (the version an update was trained on) is kept precisely because it names the
  provenance; a vaguer `trained_round` was rejected.
- Don't paper over an ambiguous name with a comment — rename it. Comments explain
  *why*, names carry *what*.
- Scope renames to the baseline you're in (oort+refl share `OortSelector`); felix
  (`AsyncOortSelector`) and others are separate passes.

---

## §1  Philosophy: the parity ladder

An FL run is a **pipeline**. Each round flows through the same stages in both
real and sim mode:

```
clock/time-base → availability → selection → dispatch+training
   → update-return+ordering → aggregation → utility → emergent outcomes
```

Parity must hold at *every* stage. If it breaks at stage N, every stage above N
also diverges — but those upper failures are **consequences, not bugs**. The job
of the checker is to find the **lowest broken rung**: the earliest stage whose
own inputs are sound but whose output diverges. That stage holds the root cause.

This replaces the old "severity-ordered symptom list." Severity tells you what
hurts; the ladder tells you *why*, and does so automatically.

### Three roles every check plays

Tag each check with the role it serves in localization:

- **CONTROL** — confirms an *input* to a stage is identical across modes
  (e.g. trainer_speed_s, training_budget_s, telemetry coverage). A failing
  control means the sim's inputs differ; fix the input model, not the stage.
- **MECHANISM** — confirms a *single transformation* inside a stage is modeled
  (e.g. per-commit overhead, inter-round overlap, availability time-base). A
  failing mechanism with passing controls is a *localized* bug — the prize.
- **EMERGENT** — an aggregate outcome (throughput, terminal state, convergence,
  utility). These are what we ultimately care about, but they never localize on
  their own; they only tell you *something* below them broke.

Debugging rule: an EMERGENT failure is a prompt to walk *down* the ladder to the
mechanism/control checks beneath it. Never fix an emergent symptom directly.

### Two-axis classification

Every check has two orthogonal labels:

- **STAGE** (0–9 below): where in the causal pipeline it sits. Determines
  ordering and dependency.
- **TIER** (enforcement strictness, unchanged from today):
  - `INV`  — sim-mode invariant; FAIL is always a hard FAIL.
  - `EXACT`— must match within tight tolerance; hard FAIL.
  - `DIST` — distributional match; FAIL unless `--lenient`.
  - `DIAG` — diagnostic only; never FAILs (informational), but feeds root-cause.

STAGE drives diagnosis; TIER drives the pass/fail verdict. They are independent.

### Dependency gating (the part that makes checks build on each other)

Each check declares its **upstream prerequisites** — the checks whose passing is
required for this check to be *meaningful*. The verdict engine then:

1. Walks rungs bottom-up.
2. Finds the lowest stage with an enforced FAIL whose upstreams all PASS →
   labels it **ROOT-CAUSE**.
3. Tags every higher enforced FAIL whose upstream chain contains a failed check
   as **DOWNSTREAM (of <root>)**, demoted from the headline failure list.

Result: one run prints "ROOT-CAUSE: stage-1 overhead residual (K3b); 7 downstream
failures suppressed" instead of nine equally-loud FAILs you have to triage by hand.

`deps` must name the *strongest causal link*, not a generic base. In particular
TC1 (coverage) is **not** a universal ancestor — it gates only K10, because a
missing field makes a downstream check SKIP (handled locally), not FAIL. Wiring
every check to depend on TC1 would wrongly demote independent failures (e.g. a
real trainer_speed gap) to "downstream" whenever any *unrelated* field is absent.

### Growth rule

Every time a parity bug is root-caused, leave behind the **most fine-grained
check that would have localized it to the responsible mechanism**, placed at its
causal stage with its upstream dependencies declared. Checks are append-only:
never delete one to "clean up." A check that is currently redundant becomes a
regression guard the next time the simulator changes.

When a single coarse check can be split into independent mechanisms, **split it**
— one assertion per mechanism. A blob KS over six timing phases tells you "timing
is off"; six per-phase KS checks tell you "the MQTT-fetch phase is off, the rest
match." Always prefer the latter.

---

## §2  The ladder

Stages run foundational → emergent. Within a stage, controls/mechanisms precede
the emergent rollup. `[NEW]` = to implement; everything else exists in checks.py.
"Isolates" = the one thing this check tells you when it fails *and its upstreams
pass*. "Dep" = upstream prerequisites.

### Stage 0 — Telemetry coverage  *(gate for everything)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| TC1 `[NEW]` | Field coverage matrix | CONTROL/INV | A field a downstream check reads is missing/sparse in one mode — explains every downstream SKIP at once | — |
| K10 | vclock_now present (sim) | CONTROL/INV | Sim path never stamps vclock (sync aggregator today) | TC1 |

> TC1 generalizes K10: for *each* field consumed downstream (vclock_now,
> staleness, trainer_speed_s, avail_composition, num_eligible, the phase fields,
> stat_utility, sim_send_ts), report presence count + density per mode. One table
> turns "9 mysterious SKIPs" into "these 3 fields are absent in sim."

### Stage 1 — Clock / time-base  *(the foundation; most parity bugs live here)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K1 | vclock monotone (sim) | MECHANISM/INV | vclock goes backwards | K10 |
| K7 | sim_rate in [0.01,100] | MECHANISM/INV | vclock/wall absurd | K10 |
| P3 | trainer_speed_s distribution | CONTROL/DIST | The *input* to the clock model differs (speed model itself wrong) | — |
| K3a `[NEW]` | Modeled-compute advance | MECHANISM/EXACT | sim Δvclock vs the speed order-statistic the round-close formula *should* produce (K-th fastest in-flight for async; max-of-K for sync) — tests the advance **formula** with overhead excluded | P3,K1 |
| K3b `[NEW]` | Overhead residual | MECHANISM/EXACT | `real_advance − sim_advance` per round ≈ 0 — the missing per-commit MQTT/dispatch overhead (CRITICAL-1). Pass once `sim_commit_overhead_s` is modeled | K3a |
| K4 | Overlap factor | MECHANISM/DIAG | Sim doesn't model inter-round async pipelining | P3,K1 |
| K3 | Per-round advance distribution | EMERGENT/EXACT | Sum of K3a+K3b+K4 diverges (rollup) | K3a,K3b,K4 |
| K2 | Rounds-per-virtual-second | EMERGENT/EXACT | Throughput diverges (rollup) | K3 |

> The decomposition is the whole point. Today K3 (per_round_advance) lumps
> formula + overhead + overlap into one FAIL. Split it: if **P3 passes, K3a
> passes, K3b fails, K4 passes** → the bug is *pure missing overhead*, nothing
> else. That single sentence is what CRITICAL-1 took a paragraph of prose to say.
> K3b is also cross-validated at Stage 4 (mqtt_fetch phase): real overhead seen
> at the trainer level should equal K3b residual × agg_goal.

### Stage 2 — Availability  *(indexed by the clock — so gated on Stage 1)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| A1 | avail_composition parity | MECHANISM/DIST | Per-state available counts diverge | — |
| A2 | num_eligible / num_candidates | MECHANISM/DIST | Eligible-set size diverges | A1 |
| A3 `[NEW]` | Trace time-base consistency | CONTROL/DIST | Availability trace indexed by *different* clocks (sim=vclock, real=wall) — the REFL HIGH-1 bug. Compare each trainer's first/last-available time mapped through its mode's clock | K3 |
| A4 `[NEW]` | Per-trainer duty-cycle | MECHANISM/DIST | A trainer's on/off fraction differs even when set sizes match; needs avail_change events | A3 |

> A2's failure on REFL is *downstream* of the clock (sim runs at vclock_rate
> 0.274 → hits different trace windows). A3 makes that explicit: it fails only
> when the time-base mapping itself is wrong, so A2-fail + A3-pass = "fix the
> clock first," A2-fail + A3-fail = "fix the trace lookup."

### Stage 3 — Selection  *(given the eligible set)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| S3/4 | num_chosen / in_flight / effective_c | MECHANISM/DIST | Selector picks a different count | A2 |
| A2c `[Jun14c]` | selected-vs-pool speed bias | MECHANISM/DIST | Selector's revealed *speed preference* (`bias=selected−pool`) diverges with the pool matched → **selector-scoring** (oort), vs pool itself diverging → **composition** (A2b, refl) | A2b |
| Sx `[Jun14c]` | selector score-term localize | DIAG | *Which* utility-score term drives a mix split (oort believed_I/temporal/system_util; feddance V/I/A/U) — pinpoints e.g. oort `system_util` | A2b |
| Sd `[Jun16]` | preferred-duration penalty bind | MECHANISM/DIST | Oort speed-penalty **binding frequency** per round (≥1 selected w/ `system_util<1`) + reconstructed `pref` median — the D1 unsorted-`pref` guard (caught: real 80 % vs sim 46 %). Works on pre-instrumentation runs (reconstructs `pref=dur·√system_util`). | A2b |
| S2 | Participation frequency | EMERGENT/DIST | Per-trainer chosen-count diverges | S3/4 |
| S1 | Per-round Jaccard | DIAG | Exact set identity (gated WARN for stochastic selectors) | A2 |

### Stage 4 — Dispatch & training  *(per-trainer timing; the overhead source)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| T2 `[NEW]` | training_budget_s distribution | CONTROL/DIST | The *input* to the speed model differs | — |
| T_pre `[NEW]` | pre_train_s phase | MECHANISM/DIST | one phase | — |
| T_w2g `[NEW]` | weights_to_gpu_s phase | MECHANISM/DIST | one phase | — |
| T_gpu `[NEW]` | gpu_compute_s phase | MECHANISM/DIST | one phase | T2 |
| T_mqtt `[NEW]` | mqtt_fetch_s phase | MECHANISM/DIST | per-commit MQTT overhead at trainer level (cross-checks K3b) | — |
| T_w2r `[NEW]` | weights_to_ram_s phase | MECHANISM/DIST | one phase | — |
| T_post `[NEW]` | post_train_s phase | MECHANISM/DIST | one phase | — |
| T3 | GPU budget respected | MECHANISM/INV | real GPU time overruns modeled budget | T2 |
| K6 | sim_send_ts correctness | CONTROL/INV | sim dispatch timestamps not stamped/advancing | K10 |

> Today `trainer_phase` is one DIAG blob. Split into one DIST sub-check per phase
> so the report says exactly which phase diverges. Keep the blob's combined table
> in the report for at-a-glance reading, but each phase asserts independently.

### Stage 5 — Update return & ordering
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U5 | Inter-arrival order (Spearman) | MECHANISM/DIST (gated WARN) | Arrival rank within a round diverges | K3,S3/4 |
| U4 | agg_goal_count cycles 1..K | MECHANISM/INV | Lost/double-counted update per round | — |

### Stage 6 — Aggregation
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U3 | Staleness distribution | MECHANISM/DIST | Staleness diverges (async: directly downstream of clock under-charge) | K3,U5 |
| P1 | Aggregation sequence | EMERGENT/DIST (gated for stochastic) | Per-round contributing set diverges | S2,U5 |

### Stage 7 — Statistical utility
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| F1-3 | Per-trainer utility distributions | EMERGENT/DIST | Utility diverges (downstream of selection+training+staleness) | S2,T_gpu,U3 |

### Stage 8 — Emergent outcomes  *(the headline numbers; depend on ~everything)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K8 | Terminal-state parity at matched V | EMERGENT/EXACT | rounds/trainers at matched virtual budget diverge | K2,S2 |
| U2 | Total commits at matched V | EMERGENT/EXACT | commit count at V diverges | K2,U4 |
| C1 | Accuracy curve by FL round | EMERGENT/DIST | accuracy diverges | F1-3,K8 |
| C2 `[NEW]` | Loss curve by FL round | EMERGENT/DIST | loss diverges (tracked separately from acc) | F1-3,K8 |

### Stage 9 — Budget / stop sanity  *(meta; orthogonal to causal chain)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K9 | Stopped by budget, not rounds cap | INV (WARN) | Comparison truncated by `rounds` cap | — |
| K5 | Failsafe ceiling | INV | Sim wall overshoot > 20% of budget | — |

---

## §3  Mechanism reference — implemented sim fixes

The simulator does **real GPU compute** but stamps a *modeled* completion time
`sct` (it does not sleep the trainer's wall budget). Parity work is making the
sim's clock, ordering, and availability behave as the real pipeline would at that
`sct`. The validated mechanisms below are config-gated and guarded; the
**Overriding principle**: parity ≠ goal, a *correct* simulator is — real is the
reference only after `validate_real` shows it admissible (done: concurrency
28.8/c30, double-dispatch 0); never tune sim to a wrong real.

### Clock & ordering (felix async stack, validated)
- **Overhead → 0** (`simCommitOverheadSeconds=0`): the clock TRACKS completions
  (`vclock = max(vclock, sct)`) instead of being a pure overhead ramp.
- **Drain by physical READINESS, not predicted completion** (`_sim_recv_min`):
  admit any in-flight end whose message has physically arrived into the reorder
  buffer, so slow trainers buffer as futures and commit in `sct` order (staleness
  7.2 → 3.5). Probing the LIVE in-flight set (`recv_fifo` pops min-`sct`) reorders
  the past-dated tail.
- **`realDistributeSettleSeconds=0`**: removes a real-only 2×`sleep(0.1)`/commit so
  real holds ~c computing instead of being artificially slowed (advance 4.1,
  staleness 2.8).
- **`simRedispatchGapSeconds`** (post-commit re-dispatch leg, slot-held by cooling):
  spaces completions without counting toward the committed update's staleness.
  `0.6` zeroed the over-advance → throughput family green. *Residual:* a
  gap↔staleness coupling means one knob can't hit both advance and staleness; felix
  is HELD pending a buffer-aging investigation, not a scalar.

### §3.async  Async ≠ sync selector knobs — do NOT inherit the Oort *paper* defaults
`third_party/Oort` is **sync-only**; there is no async Oort reference, so the paper
defaults (`OORT_PAPER_DEFAULTS`, e.g. `round_threshold 10`, `exploration_decay .95`)
are SYNC values. Applying them to the async `AsyncOortSelector` (felix) regressed it
(overlap 10.9× vs real 6.6×, staleness 8.4 vs 2.8) because the sim overlap model is
calibrated to the selected MIX, and the sync knobs narrow that mix. Root theme:
**many Oort knobs are parameterized *per round*, but "a round" is a different unit in
async (one `agg_goal` batch) than sync (a full barrier), and async runs ~2–3× more of
them.** Inheriting sync values therefore misbehaves:

- **`round_threshold` (speed penalty)** — exists to protect a SYNC barrier (round =
  max-of-K; a straggler blocks everyone). Async/fedbuff has no barrier (stragglers
  commit stale later) → the penalty should be largely **inert**. Felix uses **70**
  (broad mix). NB the pacer ([async_oort.py:546](../../flame/selector/async_oort.py#L546))
  only ever *raises* it toward 100, so the start value washes out over a long run.
- **`exploration_decay`** — applied once **per round**; async's higher round count
  collapses a sync-tuned decay almost immediately (0.95 → exploration floored in ~29
  rounds). Felix uses **0.999** (still reaches an exploitation phase across ~1150
  rounds). `0.9999` ≈ permanent exploration (never exploits) — rejected.
- **temporal/UCB** `√(0.1·log(round_num)/last_selected_round)` — `log(round_num)`
  inflates with async's round count (more exploration pressure, automatically).
- **pacer cadence** (`pacer_step` rounds) — fires more often in wall-time in async.
- **staleness weighting** — async-only (sync has none); confirm fedbuff down-weights.
- **D5 temporal time-base** (FIXED Jun 16, oort+refl) — the issue was *write
  timing*, not value: stamping `PROP_LAST_SELECTED_ROUND` at commit let its
  visible value ride commit ordering (sim-regular vs real-jittery). Now stamped
  at selection (value = selection round, unchanged). Matters more in async,
  where selection and commit decouple. felix (`AsyncOortSelector`) deferred.

Principled generalization (not yet done): re-parameterize the per-round terms by
**wall-time or samples-seen** so they're invariant to round semantics. Until then,
async knobs are config-driven and anchored to the real run's spacing, NOT the paper.
**Felix's 70/0.999 is a new operating point** — the first 3 h run is its first parity
test there; if overlap/staleness still diverge, the next move is the overlap-model
(buffer-aging) re-tune, not more knob changes.

### §4.5  refl — `sct`-gated pool exclusion (`simInflightResidence`, validated)
A trainer that has physically sent but is modeled as still computing (`vclock < sct`)
must NOT re-enter the eligible pool — in real it is busy. In `oort/top_aggregator.
_distribute_weights`, the still-computing set (`_sim_buffer.pending_after(vclock)`)
is added to `trainer_unavail_list` (the *unavailable* path, NOT `selected_ends` —
which would re-dispatch and reset `sct`); released at `vclock ≥ sct` (budget ≤ ~56s).
Fixed refl's pool composition (A2b 12.4 → ~6.5 = real), flipping all emergent checks
green. Guard: `test_virtual_clock.py::test_pending_after_*`,
`test_sync_sim_ordering.py::TestSimInflightResidence`.

### §4.9  oort — `sct`-gated carry-over (`simInflightCarryover`)
**Jun-15 seeded run:** under-fired (`in_flight_after` 0.15→1.43 vs real 4.58).
**Root found + FIXED (Jun 16):** not a tuning gap — a **lost-straggler bug**. The
held stragglers were re-buffered in a loop *after* the yield-loop, but the caller
abandons this generator the moment `agg_goal` fresh updates are accepted, so it
never ran and `held_over` was dropped each round. Now wrapped in `try/finally`
(the `GeneratorExit` on `gen.close()` runs the re-buffer).
The sync-oort aggregator over-selects (×1.3) and closes a round at agg_goal=10,
leaving the ~3 slowest still computing. In **real** they stay in `selected_ends`
in-flight across rounds (`in_flight_after` 3.3); in **sim** the update arrives at
once, gets stale-rejected (prior `MODEL_VERSION`), and frees its slot → sim drains
to 0.15. **Distinct from §4.5**: §4.5 gates pool *re-entry*; §4.9 gates the
*cleanup/commit*. In `oort/top_aggregator._oort_sim_recv`: a prior-round straggler
(`_round − MODEL_VERSION > 0`) with `sct > vclock_round_start` is held (not yielded,
not clock-advanced), re-buffered so it stays in `selected_ends` (carried in-flight),
and commits a few rounds later once the clock passes its `sct`. Enabled for the oort
sim block. Guard: `test_sync_sim_ordering.py::TestSimInflightCarryover`.

**Third mechanism, FIXED (Jun 17): carry-over threshold creep.** The 1h rerun
showed carry-over still under-firing (`inflight_after` sim 0.82 vs real 4.24).
Mining per-round `in_flight_after` from the existing sim run showed it **starts at
4.31 (≈ real 4.24) and decays to ~0 by mid-run** — a progressive collapse, not a
uniform absorb. Root: the gate held a prior-round straggler when `sct >
vclock_round_start`, but `vclock_round_start` was re-read as `self._vclock.now`
*inside each* `_oort_sim_recv` call. The block-for-K-fresh retry loop creates a
fresh generator per pass *after* earlier passes advanced the clock, so the
threshold crept forward within a single round and committed stragglers whose `sct`
fell between the true round start and the advanced clock. Early rounds (few
retries) barely creep → match real; later rounds creep hard → drain. Fix:
`_aggregate_weights` pins `self._round_start_vclock = self._vclock.now` once, before
the first `_oort_sim_recv` call; the gate reads that pinned value. Guard:
`TestSimInflightCarryover::test_pinned_threshold_holds_straggler_across_retry`.

**Second mechanism, FIXED (Jun 16, same day): block-for-K-fresh.** Even with
carry-over correct, the 3h rerun showed `committed_fresh` sim 7.24 vs real 10
— a *starvation*, not a carry-over bug. Root: `_aggregate_weights`'s second
poll loop only ran `while not self.simulated`, so a sim round got exactly one
`_oort_sim_recv` pass; if a fresh (this-round) trainer's message wasn't ready
within the adaptive `grace` window (`4× EMA of past full-drain time`,
`syncfl/top_aggregator.py`), it was silently skipped this round and re-probed
next round — by which point `self._round` had advanced, so it commits **stale**
instead of fresh. Real's equivalent loop just keeps polling the same `end_ids`
indefinitely. Fix: removed the `not self.simulated` gate so sim also retries
— calling `_oort_sim_recv` again (re-probing the same persistent
`SimReorderBuffer`, giving slow-but-alive trainers another `grace` window)
instead of `channel.recv_fifo` — bounded by a `progressed` flag so a pass that
accepts nothing stops instead of spinning. Guard: existing
`tests/sim`/`tests/mode` suite, 149 pass / 7 skip (was 147/7). **Awaiting a 3h
cluster rerun to confirm `committed_fresh`→~10** (logical fix validated by
code trace + unit tests, not yet by a fresh telemetry run).

### §5  Checker corrections (stochastic / observability classes)
Once the sim *dynamics* match, some residual FAILs were the checker enforcing exact
identity on quantities a stochastic / in-memory simulator cannot reproduce
(diagnostic tell: byte-identical across runs despite large dynamics changes). All
are principled, guarded, append-only — a future *deterministic* selector still gets
exact enforcement via `DETERMINISTIC_SELECTORS`:
- **P1 aggregation_sequence** → WARN for stochastic selectors (exact per-round set
  identity unattainable; S2 participation is the enforced invariant).
- **F1-3 utility** → enforce the POOLED KS (per-trainer KS=1.0 was mechanical for
  n≤2 samples; means were identical).
- **phase_mqtt_fetch** → DIAG (in-mem cache wall time, deliberately off the virtual
  clock).
- **trainer_speed / eligible_speed / selection_bias** → integer-grid / metadata-pool
  (see Status → checker corrections).

### Discrepancy ledger — flame vs reference Oort (per-baseline)
flame has ONE `OortSelector` inherited by both the `oort` baseline (should match
standalone Oort, `third_party/Oort`) and `refl` (should match the REFL fork,
`third_party/REFL`). The two references differ on defaults, so each baseline's
knobs are config-driven (`selector.kwargs`), defaulting to the Oort paper
(`scoring.OORT_PAPER_DEFAULTS`) with refl overriding to the fork.

| # | discrepancy | resolution |
|---|---|---|
| D1 | `pref` not sorted | FIXED (sort added) — was a port bug; validated on oort |
| D2 | stat-utility not normalized/clipped | FIXED (`scoring.oort_normalize_reward`, config `normalize_reward`/`clip_bound`) |
| D3 | `round_threshold` | config-driven: oort/felix=10 (paper), refl=30 (fork) |
| D4 | `cut_off_util` + cutoff-index | FIXED: config (0.7 paper / 0.05 refl); index now thresholds the exploit-boundary score (was inert) |
| D5 | temporal time-base | **FIXED (Jun 16) for oort + refl** — the VALUE (selection round == `MODEL_VERSION`) was always correct and matches both refs' `time_stamp` (engagement-round, which in sync == selection round). The bug was the *write timing*: the aggregator wrote it at **commit**, making a candidate's visible value depend on commit ordering (sim sct-regular vs real FIFO-jittery). Fix: stamp at **selection** in the selector (`oort.py::_record_last_selected_round`), remove the commit-write. Guard `TestLastSelectedRoundStamp`. **felix not yet done** — same commit-write at `asyncfl/top_aggregator.py:516`, separate `AsyncOortSelector`; deferred (mind `round_nudge_type`). |
| D6 | `clip_bound` | config-driven: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now the fork's cut_off_util-augmented utility-weighted `np.random.choice` |
