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

## Status (Jun 16 — 3h validation + scoreboard reframe)

The 3 h re-checks are in: `experiments/run_20260616_*` (real/sim pairs per
baseline, `--budget-s 10800`), reports `experiments/parity_3h_*.json`. The
checker now emits an **enforced pass/total scoreboard** (`summary` key in the
JSON, footer of every report, `pass/tot` column in `--batch`) so the distance to
parity is tracked and regenerable, not hand-maintained here. **Re-deciphering the
3h JSONs reframed two roots** (oort = one root not six; feddance = selection-mix
not wall-capture).

| baseline | score (enforced pass/tot) | lowest broken rung | what it actually is |
|---|---|---|---|
| **feddance** | **43/43 — PASS ✅** | — | **first fully-green baseline.** `P3`/`T2` reclassified to support-guard (the speed FAIL was selection-mix, not a model bug: `A2b` pool KS=0, sim just selects faster); `K3` per-round-advance moved to grid-KS (raw .715 was pure quantization — identical p10..p90). All closed with **no rerun** against the stored 3h dir. ⚠ the deferred selection-mix (sim picks ~1.7 s faster) is a real fidelity gap to revisit (may move end-to-end perf) |
| **refl** | **40/46** | `K2` throughput (rel .075) | genuine: sim 1.36 s/round vs real 1.47 — a 0.11 s/round under-charge. `K3b` per-round residual is .077 (under its own .1 bar) but compounds over 7937 rounds into K2's .05 aggregate bar. → model the per-commit overhead (sim change). `A2` also FAILs on eligible-pool *shape* (KS .453, mean 1% apart) |
| **oort** | **34/44** | `Sr` residence + `P3` slow tail (one root, several downstream) | **single root: carry-over collapses after warmup.** `[SIM_CARRYOVER]` logs: first ~10 rounds carry 3–7 stragglers, then only 112/908 carry any → mean in_flight_after 0.31 vs real 3.52. Sim commits **6.65 fresh/round vs real's 10**; the rest arrive in a stale burst once vclock catches up, committing with inflated modeled times → sim speed p99=27 s/max=47 s vs real 22/23.8 (`P3` support-guard now catches this; the old grid-KS .067 missed it). `selection_bias` (−2.65), `Sd` (.517), `throughput`, `P3` are all DOWNSTREAM — fix carry-over → they recover |
| **felix** | **33/42** | `K3b` overhead + `S3/4` | overlap-model collapse (the documented long pole; independent async stack). overlap 12.6× vs real 6.8×, staleness 13.7 vs 2.8, accuracy diff .106. Async knobs are exhausted; buffer-aging re-tune is next |

> Scoreboard caveat: the denominator is the *enforced* universe (EXACT/INV +
> DIST under default rules); DIAG/WARN/SKIP are excluded. Regenerate with
> `parity_check.py --batch` — the `pass/tot` column and per-run footer are the
> source of truth, not this table. (Denominators shifted 1–2 from the earlier
> table as the support-guard/grid-KS reclassifications changed which checks
> enforce vs warn.)

### Pinned next steps (post-3h) — grouped by whether a rerun is needed

**Key lever for "fewer reruns": checker-side changes re-validate against the
existing `run_20260616_*` dirs instantly. Only sim *mechanism* changes need a new
cluster run.** Plan accordingly: land all checker corrections + verify against the
stored dirs, then batch the sim-mechanism changes into ONE seeded rerun.

**No rerun (checker, validate against stored 3h dirs) — DONE:**
- **feddance `P3`/`T2`/`K3` — CLOSED, feddance is 43/43.** Landed against the
  stored 3h dir (no rerun): `P3`/`T2` now enforce *support containment* (sim must
  not produce speeds beyond real's range), with the grid/mean KS demoted to a
  selection-mix diagnostic owned by `A2c`; `K3` per-round-advance enforces a grid
  KS (the raw .715 was pure sim-quantization-vs-wall-jitter). Guards:
  `test_trainer_speed_support_guard_tolerates_mix_catches_tail`,
  `test_per_round_advance_grid_ks_tolerates_quantization`,
  `TestTrainerSpeedParity::test_out_of_support_tail_fails`. **Bonus:** the support
  guard is *more* sensitive than the old grid-KS — it caught an oort sim slow tail
  (p99 27 s vs real 22) the grid-KS (.067) had missed.

**Needs a seeded rerun (sim mechanism) — LANDED as hypotheses, batch into ONE run.**
Launch: `bash scripts/debug_run.sh --baselines 'refl felix oort' --runtime-s 10800
--mode both` (feddance is already green; include it to regression-check if desired).
Each change is config-gated/reversible; re-run the checker (`parity_check.py --batch`)
against the new dirs and read the `pass/tot` column.

| baseline | hypothesis landed | where | expected effect | confidence |
|---|---|---|---|---|
| **refl** | `simCommitOverheadSeconds: 0.011` — model the missing per-commit MQTT/dispatch overhead (residual 0.11 s/round ÷ aggGoal 10). Applied in `_advance_sim_clock` (`max(vclock,sct)+overhead`, drift-instrumented; the documented `K3b` mechanism, NOT a ramp) | [parity yaml refl sim block](expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml) | sim advance 1.36→~1.47 s/round → `K2`/`U2` close (rel .075→<.05). Refine value from next-run `K3b` residual | **high** (one-line config, existing `test_sim_commit_overhead.py` guard) |
| **felix** | `simRedispatchGapSeconds: 0.6` — post-commit re-dispatch cooldown spaces future-dated commits (`[SIM_CLOCK_DIAG] buf_future`) without inflating staleness | [parity yaml felix sim block](expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml) | advance 2.25→toward 4.02 → overlap 12.6×→toward 6.8×. **Watch staleness** (13.7→2.8): doc §3.async flags a gap↔staleness coupling — if overlap drops but staleness stays high, the deeper fix is the buffer-aging model (pace future-dated commits), not this scalar | **medium** (reversible; doc flags knobs may be insufficient) |
| **oort** | **diagnosis refined, behavioral fix DEFERRED (no-regression).** Carry-over stays on (`simInflightCarryover: true`). Root is NOT just hold-duration — late-run rounds are *starved of fresh commits* (`committed_fresh` 10→5.92 over the run; `in_flight_after` 4.67→0.02). Sim's `_oort_sim_recv` generator stops when the buffer drains instead of **blocking until aggGoal fresh accepted** the way real's `recv_fifo` does. Fix = re-probe/advance on buffer-empty until aggGoal fresh commit (a careful change to the *shared* sync aggregator — would touch refl, so not rushed) | [oort/top_aggregator._oort_sim_recv](../../flame/mode/horizontal/oort/top_aggregator.py#L62) | this rerun's `[SIM_BARRIER]` logs (probed/buf_depth/drained_all per round) confirm whether the starvation is grace/probe vs selection; then land block-for-K-fresh next session | **diagnosis high / fix next session** |

Why oort's fix is deferred not rushed: `_oort_sim_recv` and `_aggregate_weights`
are shared by oort AND refl (only the carry-over *gate* is oort-flag-gated). A
block-for-K-fresh change alters the recv loop both use, so it needs its own focused
implementation + guard, validated so it doesn't regress refl's now-good clock. The
batched rerun above gives the `[SIM_BARRIER]` data to land it confidently. Also
lower-priority: refl `A2` eligible-pool *shape* (KS .453, mean matches) — a
distribution/periodicity gap, not magnitude; A2b composition beneath it passes.

### Resolved checker decision — feddance `P3`/`T2` semantics (Option A, landed)

`P3`/`T2` pooled the **selected** trainers' speeds, so they absorbed selection
mix. Chosen (user, Jun 16): **make them selection-independent** via a support
guard, demoting the frequency/mean KS to a diagnostic `A2c` owns. ⚠ **Flagged to
revisit for higher fidelity:** this *defers* a real gap — sim selecting ~1.7 s
faster trainers can move end-to-end perf metrics (accuracy/convergence) later, so
once the sim-mechanism baselines land, come back and close the feddance utility-
scoring bias (`A2c` bias sim −1.23 vs real +0.49) rather than leaving it deferred.
Rejected: keeping P3/T2 enforced as-is (would chase the mix via reruns at a tight
bar `A2c` already passes at .2).

---

## Status (Jun 15 — seeded 45-min run results) — condensed

Seeded oort/refl/felix/feddance pairs (`seed=1234`, 2700 s, agg_goal 10, oort §4.9
carry-over ON): `experiments/run_20260615_16*/17*`. Superseded by the Jun 16 3h
run above (refl's "near-clean" call didn't hold at 3h scale, and the oort/feddance
fixes landed here read differently at 3h) — kept only for the one-change-per-baseline
record and durable cross-cutting reads.

**One change per baseline — landed from this run, re-checked by the Jun 16 3h run:**

| baseline | change | where |
|---|---|---|
| **oort** | carry-over re-buffer now in a `try/finally` so held stragglers survive the caller abandoning the recv generator early — the §4.9 under-fire was a lost-straggler bug, not tuning | [oort/top_aggregator.py:122](../../flame/mode/horizontal/oort/top_aggregator.py#L122) |
| **felix** | **async-relaxed** knobs `round_threshold 10→70`, `exploration_decay .95→.999` (NOT the sync paper; see §3.async) | [parity yaml felix blocks](expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml) |
| **refl** | gate `Sd` to WARN when the penalty is **inactive in real** (`r_frac=0`, no reconstructable `pref`) — the PROP_ROUND_DURATION None-density artifact (A2b/A2c class). oort (`r_frac>0`) stays enforced | [checks.py preferred_duration_parity](scripts/parity/checks.py) |
| **feddance** | widen `P3 mean_overhead` bar 0.5→1.5 s (sub-grid, opposite-sign, metadata-matched wall-capture; grid_KS stays the backstop) | [checks.py trainer_speed_parity](scripts/parity/checks.py) |

**Durable cross-cutting reads (still true at 3h):**
- **`Sdet` triage rule.** `eligible_match≈0` **with matching aggregates** = stochastic-class,
  PASS as-is; `eligible_match≈0` **with a diverging clock** = genuine, fix the clock;
  `eligible_match` high but `decision_match≈0` = the **values** diverge, not the set.
  This is how all four baselines' roots were localized, both at 45min and at 3h.
- **`P3 mean_overhead` is wall-capture, not a speed-model bug**, when it's sub-second,
  opposite-sign across baselines, and `grid_KS`/`training_delay_s` metadata match — only
  trust a `P3` FAIL when `grid_KS` also fails (oort's 56s tail at 45min was the one
  genuine case; it resolved by the 3h run).
- **Run length matters.** 45min already exercises `Sdet` + every mechanism check, but
  is not enough for `C1`/`C2` convergence sign-off, and (learned at 3h) not enough to
  catch round-count-compounding clock residuals or low-frequency eligibility-shape
  drift — both surfaced only in refl's 3h run despite a clean 45min pass.

### Dead ends — do NOT retry

- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- `version_at(sct)` staleness relabel (fedbuff consumes the *real* number; inert).
- Adding `mqtt_fetch` (~57 s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` for felix (sim over-overlaps; the gap is a real mechanism).
- Tuning sim to a *wrong* real, or any scalar fudge where a mechanism is called for.
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0), the
  felix post-compute leg as a "bug" (it's serial-aggregator scheduling), per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent by nature).
- Expecting seeding to align per-round sets: `Sdet eligible_match`≈0 is *expected*
  for stochastic selectors whose inputs (clock-indexed availability) drift; judge the
  aggregates (S2/participation), not the per-round draw. Only a *diverging clock*
  (oort/felix) makes `eligible_match`≈0 a genuine bug.
- A scalar fudge for the `P3 mean_overhead` ~1 s offset (wall-capture, opposite signs
  across baselines, grid-passing). Widen the bar or score on `training_delay_s`; don't
  bias the speed model to chase it (it would break the genuine oort 56 s tail signal).

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
- **D5 temporal time-base** (deferred) — `last_selected_round` vs round-last-*updated*
  matters more in async, where selection and update decouple.

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
(the `GeneratorExit` on `gen.close()` runs the re-buffer). Pending 3 h validation.
Below is the mechanism as designed.
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
| D5 | temporal time-base | **DEFERRED** — flame uses `last_selected_round`; refs use round-last-UPDATED. Needs an aggregator-stamped prop; subtle, high blast radius. Flagged in code. |
| D6 | `clip_bound` | config-driven: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now the fork's cut_off_util-augmented utility-weighted `np.random.choice` |
