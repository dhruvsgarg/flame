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
`--budget-s` = the run's `--runtime-s` (e.g. 12600 for 3.5h). Add `--lenient` to
demote DIST fails to warnings; prints a stage-grouped report + root-cause banner.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Readiness/regression tests** (no cluster; run under lib/python):
`pytest tests/mode/test_baseline_readiness.py tests/mode/test_sim_barrier.py
tests/mode/test_async_sim_ordering.py tests/mode/test_sync_sim_ordering.py
tests/mode/test_sim_commit_overhead.py` — guards baseline wiring, the in-memory
cache, serialize-once, sim-recv barrier ordering, and the overhead model.

### Status: **Jun 17b — DETERMINISM/SEEDING landed across all selectors (dedicated per-selector RNG + decision-fingerprint telemetry + Sdet check). refl P3+A2b checker fixes in. UNRUN — next: SEEDED 2 h pairs. Tests 271 + 23 green.**

> **Determinism / seeding (this session, the crux of "are we comparing like with like?").** The Jun-17 refl/felix runs were **unseeded** — selection uses `np.random.choice` (utility-weighted exploit) + `random.sample/randrange` (exploration), and no `seed` was set, so real and sim took **independent stochastic paths**. Two REAL runs would have diverged from each other by the same envelope as real-vs-sim → the residual participation (.554) / `believed_I` (felix .522) gaps are NOT sim bugs, they're path drift. **Proof it's not telemetry:** speed-CLASS selection rates already match (refl fast .344/.345 …); only per-individual identity drifts; the stochastic null (same-mode even/odd) is .04–.08 but that null shares utility state so it understates the true cross-independent-run floor.
>
> **Fix — make selection a pure function of (state, seed), comparable across runs/modes:**
> - **Dedicated per-selector RNG.** `AbstractSelector.__init__` builds `self._rng` (`numpy.random.RandomState`) + `self._pyrng` (`random.Random`) seeded from `config.hyperparameters.seed`, threaded as the reserved `_seed` kwarg by `channel_manager` (`:155`). EVERY selector draw now uses these, never the process-global RNG — so selection is **insulated** from other `np.random` consumers (model init, datasampler): a residual real/sim divergence under a shared seed is a genuine INPUT divergence, not RNG desync. Replaced all global calls in `oort/async_oort/refl_oort/fedbuff/random/async_random` (incl. a missed `random.randrange` in fedbuff); set→list conversions feeding a draw are now `sorted()` for cross-process-stable order. (Import the RNG classes directly — `from random import Random` — to avoid the `flame/selector/random.py` submodule shadowing `import random` in the package `__init__`.)
> - **Config:** `Hyperparameters.seed` (config.py); `debug_run.sh` sets the SAME `seed` (default 1234, `SEED=<n>`/`SEED=none`) on every experiment so real+sim of each baseline match. `[SEED]` (aggregator) + `[SELECTOR_SEED]` (per selector) logs confirm it fired.
> - **Telemetry + check (`Sdet` `decision_determinism`, Stage-3 DIAG):** each selection event now carries `seed`, `eligible_fingerprint` (the candidate SET), `decision_fingerprint` (set + per-candidate utility/speed + k). The check matches rounds and reports `eligible/decision/chosen_match_frac` with a verdict that **splits the two outcomes the user asked for**:
>   - *seeding WORKED* → `decision_match≈1 & chosen_match≈1` → residual participation/utility gaps are NOT stochastic; look elsewhere (real prize).
>   - *RNG desync* → `decision_match≈1 but chosen_match≪1` → a selector still hitting the global RNG / seed not threaded.
>   - *INPUT divergence* → `decision_match≪1` → candidate set/utilities differ BEFORE the draw (availability/utility/ordering) → fix upstream; drop to `eligible_match` to see if it's the SET or the VALUES.
>   Guards: `test_selection_determinism.py` (same-seed reproducible; global-RNG-perturbation insulation; unseeded still stochastic); `decision_determinism` SKIPs on old unseeded runs.
>
> **Next run = SEEDED real+sim pairs** (user shortened the loop). Read `Sdet` first: it tells you, per baseline, whether seeding closed the gap or where it didn't. felix also still has the paper-defaults async-overlap regression (below) — a seeded felix isolates how much of its advance-collapse is selection-mix (should recover) vs the clock model (won't).
>
> **Run length — 1 h is enough for the fast loop (what shrinks vs what you lose).** At 1 h: refl ≈ 2.4k rounds (~24k commits, ~50 evals), felix ≈ 870 rounds (~17 evals). **Fine at 1 h** (steady state reached after ~10–15 min warmup; ample KS samples): determinism `Sdet` (per-round from round 0), clock/throughput (K2/K3/K8/U2), staleness, P3, eligible_speed, availability, selection mechanism, AND **participation** — pre-seeding this needed a long horizon because divergence accumulated path-dependently (the Jun-13 2700 s "all-green" was a horizon artifact), but **seeding makes a real participation divergence show up immediately**, so it's no longer horizon-gated. felix's advance/overlap/staleness regression is steady-state → visible at 1 h. **Weakens at 1 h:** convergence C1/C2 (acc/loss) — the real/sim gap GROWS with training, so a 1 h PASS is inconclusive (one-sided) while a 1 h FAIL is still real; larger warmup fraction. With seeding, once `decision_match≈1`, convergence is a *lagging* confirmation (same decisions → same model), so short runs lose little. **Loop on 1 h reading Sdet + mechanism checks; reserve ONE 3–4 h run for final convergence sign-off when short runs are green.**

### Status: **Jun 17 — fidelity-pass runs IN (refl/felix; oort still running). refl: §4.5 VALIDATED + 2 checker corrections (P3 grid, A2b metadata) → root now participation/preferred_duration (stochastic class). felix: REGRESSED (paper defaults broke async overlap). Tests 23 parity green.**

> **Ran the overnight fidelity-pass pairs** (4 h, agg-goal 10, budget 14400): refl sim `run_20260614_233140` / real `run_20260615_022706`; felix sim `run_20260614_233119` / real `run_20260615_024940`. oort still running. `parity_refl_fresh.json` / `parity_felix_fresh.json`.
>
> **refl — §4.5 residence VALIDATED + emergent green.** Residence fires every round (holds median 52 of ~67 in-flight). Throughput/K8/U2 rel **.007**, staleness 3.00/2.95 (KS .024), accuracy .021, pooled utility .026 — ALL PASS. Two **checker corrections** this session (data-backed, principled, guarded):
>   - **P3 trainer_speed → modeled-grid KS.** Raw KS .194 was *entirely* sub-second wall-capture jitter: sim reports exact-integer `training_delay_s`, real = same integer + ~.04–.08 s sleep/settle leg (§3m), which the virtual clock excludes by design (§5c). Integer-rounded KS **.021**, histograms near-identical (max bin diff .018). Fix: enforce KS at the integer grid + a `mean_overhead_s` guard (≥.5 s systematic offset still fails). `trainer_speed_parity`, `test_trainer_speed_tolerates_wall_capture_jitter`. P3 now PASS.
>   - **A2b eligible_speed → metadata reconstruction (user decision).** The pool gap (real 7.0 / sim 12.2, KS .39) was an **observability artifact**: real leaves PROP_ROUND_DURATION = None for ~158/300 non-completers, so pooling observed `speed_s` samples only fast completers in real vs the whole population in sim. num_eligible (244/246), in_flight (68.6/66.4), selected-speed (6.68/6.45), A3 timebase (.01) all match. Reconstructing pool speed from the static `training_delay_s` registry → real 12.13 / sim 12.13, **KS .000** (pool is genuinely identical). Fix: `_trainer_delay_map()` + `eligible_speed_composition_parity` uses metadata (falls back to observed). A2b now PASS.
>   - **Remaining refl roots = participation (S2) + preferred_duration (Sd); convergence_loss downstream (.189 vs .15).** participation matched-count-KS **.554** is GENUINE (stochastic null = real even/odd .076, sim .035 — but that null shares utility state so understates the true cross-independent-run floor). Speed-CLASS selection rates MATCH (fast .344/.345, medium .435/.422, slow .148/.184, very_slow .074/.050) → it's which-INDIVIDUAL-within-class drift from the now-stochastic REFL-fork weighted exploit. Same class as P1/utility (§5a/§5b, gated) but participation was deliberately ungated. preferred_duration: real binds 0 % / sim 43.5 % (magnitude tiny — system_util KS .042) because the *real selector* computes `pref` over the sparse observed durations while sim sees dense modeled ones. **OPEN DECISION: gate S2/Sd as stochastic/observability class, or chase as genuine.**
>
> **felix — REGRESSED from the paper-defaults pass (the flagged risk fired).** With P3 ✓ (grid KS .073) and the speed penalty never binding either mode, sim over-overlaps: advance **2.13 s vs real 4.07** → 2× rounds (6944/3468), staleness **17.3 vs 2.79**, overlap 13.4× vs 6.7×. Real (also paper defaults) stayed 4.07 — only SIM collapsed, so it's a sim-side sensitivity, not a both-sides shift. Driver: the sync-Oort paper defaults (round_threshold 30→10, exploration_decay .98→.95, D2 norm, cutoff index) changed the selected mix (`believed_I` KS **.522**, real .571/sim .375), and the async sct/redispatch-overlap model (tuned for the OLD mix, overhead 0.0 per §3c) no longer spaces completions. **`third_party/Oort` is the SYNC standalone Oort — there is NO async reference**, so felix's paper knobs are sync-Oort's misapplied to the async selector. User directive: get the real baseline params right first, then make sim obey. Real params produce a valid real run but aren't anchored to any async reference. **OPEN DECISION (user chose investigate-first): revert felix async knobs to the pre-pass values that gave parity, vs keep paper defaults and re-tune the sim async-overlap model.**

### Status: **Jun 16b — FULL base-algorithm fidelity pass landed (D1–D6 + 2 structural) across oort/refl/felix; UNRUN. Tests 246 + 22 green.**

> **Goal (user, Jun 16):** make oort/refl/felix true to their base implementations' VALUES, cleanly + extensibly;
> accept that current parity perf may shift. **What landed (all config-driven, default = the Oort paper):**
> - **Canonical defaults** `scoring.OORT_PAPER_DEFAULTS` (standalone Oort `argParser.py`): round_threshold 10,
>   round_penalty/α 2.0, clip_bound 0.98, cut_off_util 0.7, pacer 20/5, exploration 0.9/decay 0.95/min 0.2.
>   `OortSelector` + `AsyncOortSelector` read every knob via `kwargs.get(..., paper)`. **oort + felix now use the
>   paper** (were 30/0.95/0.95/decay 0.98) — these are FIDELITY changes that move their behavior.
> - **refl overrides → REFL fork** via `selector.kwargs` in `OVERNIGHT_node2.yaml` (sim+real): round_threshold 30,
>   clip_bound 0.9, cut_off_util 0.05, exploration_decay 0.98, exploration_min 0.3.
> - **D1 sort, D2 reward-normalization** (prior Jun-16 work) — now also config-faithful (clip_bound per-baseline).
> - **Structural fix 1 — `cutoff_util`** (oort + async): was hardcoded `0.95 *` an arbitrary near-bottom index →
>   now `cut_off_util *` the exploitLen-th-HIGHEST score (ref `oort.py:329`). The factor was previously inert.
> - **Structural fix 2 — refl exploitation**: was deterministic top-k (ignored cut_off_util) → now the REFL fork's
>   `cut_off_util`-augmented, utility-WEIGHTED `np.random.choice` (`thirdparty/oort/oort.py:316-355`). refl is now
>   genuinely stochastic (parity treats it as such; P1/utility already gated).
> - **debug_run.sh** reads `OVERNIGHT_{node}.yaml` — both updated (also fixed a `§` char that broke `yaml.safe_load`).
>
> **Deferred (1 item, documented):** **D5 temporal time-base** — flame uses `last_selected_round`; refs use the
> round the util was last UPDATED (on completion). Needs a new aggregator-stamped property across real+sim; held
> as the one cross-cutting change (subtle effect, high blast radius). Flagged in code at the temporal call site.
>
> **Hypothesis for the overnight (perf WILL move; that's expected):**
> - **oort**: round_threshold 30→10 lowers `pref` → speed penalty binds more → picks faster (was the parity bug
>   direction anyway). cut_off_util 0.95→0.7 + the index fix widen exploitation → more utility-weighted spread.
>   D1+D2 add the sort + meaningful temporal. Net: oort sim/real should converge AND match the paper. Watch
>   `preferred_duration` binding (target real≈sim≈high), A2c bias, K2/terminal.
> - **refl**: stays §4.5-green on emergent expected; now stochastic-faithful, so per-round identity stays gated.
>   round_threshold already 30 (unchanged). Watch loss/P3 don't regress.
> - **felix**: paper knobs (10/0.7/0.98) shift its selection; throughput family may wobble — re-confirm it holds.
>
> **Validation criteria carried forward:** the §4.5 / D1 / D2 criteria below still apply; add "selector kwargs
> in the generated config match the per-baseline reference" and "felix throughput family still green".

### Status: **Jun 16 — refl §4.5 VALIDATED (emergent all-green); oort sort fix CODE-COMPLETE + guarded, STILL UNRUN; reference tally done (D-ledger corrected)**

> **Read first (supersedes the Jun-15 block below for the "what's run" question).** Three things this session:
>
> **1. refl §4.5 (`simInflightResidence`) WORKED.** Fresh sim `run_20260614_152648` vs the overnight real
> `run_20260614_034934` (4 h, agg-goal 10, `--budget-s 14400`): the hold fires every round and at steady
> state holds **52 of ~67 in-flight** (median; buf_depth median 67) — my first "~14 held" read was just the
> warmup. **Every EMERGENT check flipped to PASS**: convergence_loss **.1735→.1446 ✓**, K8 rounds rel .021 ✓,
> U2 commits rel .021 ✓, C1 acc .021 ✓, C2 loss ✓, staleness KS .021 ✓, K2/K3/K3b ✓. Remaining refl FAILs are
> **P3 trainer_speed (KS .191, root)** and **A2 eligibility (point-mass, passed-on-mean)**; A2b pool only moved
> **12.41→11.89** (still ≠ real 6.43) and participation is DOWNSTREAM-suppressed. **A2b's residual is NOT
> under-holding** — it's that A2b compares sim's *modeled* `PROP_ROUND_DURATION` pool vs real's *wall*
> round_duration pool (same asymmetry flagged for `system_util`), compounded by the P3 speed-shape gap. refl's
> next lever is **P3 (the speed model / its observed-sample shape), not more pool exclusion.** `parity_refl_152648.json`.
>
> **2. oort sort fix is CODE-COMPLETE and UNRUN.** Correction to the Jun-15 note: the oort pair
> (sim `152630` / real `081648`) referenced below is the **PRE-fix diagnostic run** — the sorted-`pref` code
> has NOT been exercised by any run yet. So `pref` median **22** / `system_util` sim .907 vs real .865 /
> A2c bias **+0.32** measured on `152630` is the **buggy baseline**, not a validation. The fix + guards landed
> this session (see DONE); it needs the next oort run to validate.
>
> **3. Reference tally done (user cloned `third_party/REFL` + `third_party/Oort`).** D-ledger corrected below
> — notably flame's `round_threshold=30` **MATCHES REFL** (D3 is not a refl bug), and `cut_off_util` differs
> from **both** references. D1 (sort) and D2 (no reward normalization) are confirmed real port errors.

### Status: **Jun 15 — oort ROOT CAUSE FOUND = unsorted `pref` bug (a real port error) — SESSION PAUSED mid-implementation**

> **Read first. This supersedes the Jun-14 oort "selector-scoring / `system_util`" diagnosis below by
> explaining its mechanism.** The oort `system_util` divergence is a **FLAME reimplementation bug**, found
> entirely from the EXISTING Jun-14 oort pair (sim `run_20260614_152630` vs real `run_20260614_081648`) —
> **no new instrumentation or re-run was needed to root-cause it.** A re-run is only to VALIDATE the fix.

**The bug.** `OortSelector.calculate_round_preferred_duration` builds a list literally named
`sorted_round_duration` and indexes it at the `round_threshold` percentile — **but never calls `.sort()`**.
So `pref` (the speed cutoff feeding `system_util = (pref/duration)^alpha`) was an arbitrary dict-position
duration, not the percentile. Reference Oort sorts: `third_party/Oort/oort/oort.py:272`
(`sortedDuration = sorted([...]); round_prefer_duration = sortedDuration[min(int(len·thr/100), len-1)]`).

**Proof / decomposition (from existing telemetry; `speed_s` in `emit_selection` IS `PROP_ROUND_DURATION`,
present BOTH modes — reconstruct `pref = speed_s·sqrt(system_util)` since `alpha=2`; within-round spread = 0.000):**
- Emitted sim `pref` median **22 s** vs the TRUE sorted 30th-pct **7 s**; bug shifts `pref` >3 s in **68 %** of rounds.
- Speed penalty binds in **80 % of real rounds but only 46 % of sim rounds** (54 % of sim rounds penalize
  *nobody*). Selected durations match (real 8.75 / sim 8.54); the binding `pref` matches (~9 / ~8); **only the
  non-binding frequency diverges.** Non-binding `pref` → `system_util=1` for all → speed penalty off → selector
  ignores speed → sim picks pool-average (A2c bias **+0.32**) while real picks fast (**−2.54**). → S3/4, P3, throughput, terminal.
- Not the unrun-default-60 s (sim has FEWER unrun: 2.2 % vs 6.0 %); not the 99999 sentinel (1.5 %).

**DONE this session (committed? NO — uncommitted working tree):**
1. **Sort fix** — added `sorted_round_duration.sort()` before the percentile index in BOTH
   `flame/selector/oort.py:~476` and `flame/selector/async_oort.py:~631` (felix). refl inherits oort's method (→ gets the fix; reconfirm in refl review).
2. **Direct logging** — `OortSelector._system_util_summary()` emits per-round `sys_util_mean` / `frac_penalized`
   / `pref_binds` over selected on the selection event (so binding is trackable without reconstruction).
   `round_preferred_duration_s` + `alpha` already emitted (Jun-14c).
3. **Unit tests** — `tests/selector/test_oort_selector.py::TestRoundPreferredDuration` (3 tests: pref==sorted
   percentile & ≠ unsorted; monotone in threshold; thr=100→99999). **8 passed.** Added `flame.selector.async_oort`
   to the `tests/conftest.py` framework-patch loop (for a parallel async guard test, NOT yet written).

**NEXT STEPS (updated Jun 16 — items 1–3 DONE; resume at 4):**
1. ✅ **async_oort guard test** — `TestAsyncRoundPreferredDuration` in `tests/selector/test_oort_selector.py`
   (3 tests; `AsyncOortSelector` needs kwargs `c/aggGoal/evalGoalFactor/roundNudgeType/selectType`, NOT
   `aggr_num`; conftest patch already in). **11 selector tests pass.**
2. ✅ **`preferred_duration_parity`** (check id `preferred_duration`, "Sd" in the report) added to
   `scripts/parity/checks.py` (Stage 3, MECHANISM/DIST, dep `eligible_speed`), registered in `run()` +
   `CHECK_META`, formatter in `report.py`, `test_ladder.py::test_preferred_duration_detects_binding_frequency_gap`.
   **Verified on the buggy oort pair: real binds 80.2 %/round vs sim 45.5 % (diff .346, FAIL) — it catches the bug.**
3. ✅ **PARITY.md** updated (this block + D-ledger correction below).
4. ✅ **D2 bundled (user decision Jun 16).** Reward normalization+clipping implemented in `scoring.py`
   (`oort_norm_stats` + `oort_normalize_reward`, ref `get_norm`) and wired into BOTH `oort.py` and
   `async_oort.py` `calculate_total_utility` (refl inherits oort's). Config-gated `normalize_reward` (default
   True) + `clip_bound` (default 0.95). `believed_I` audit now logs the normalized value. Guarded by
   `TestRewardNormalization` (4 tests). **243 selector/mode/sim + 22 parity tests PASS.** NOTE: because D1+D2
   ship together, the next oort run validates them jointly — `preferred_duration` binding (D1) and the
   exploration/temporal effect (D2) are separable in the telemetry (Sx `temporal` term should now be
   non-negligible relative to `believed_I`∈[0,1]).
5. **Commit + push** (branch `dg/fix_sim_fidelity`) — HELD per user (Jun 16); user will commit.
6. **Re-run oort real + sim** (4 h, agg-goal 10) WITH D1+D2. Validate: emitted `pref` median 22→~7;
   `preferred_duration` binding 46 %→~real 80 %; A2c sim bias +0.32→~−2.5; Sx `system_util` KS .12→pass +
   `believed_I` now ∈~[0,1]; S3/4, P3, K2/terminal close. (Real also shifts under D2 — it's the new reference.)
7. **Re-run refl real + sim as a PAIR** with D1+D2 + §4.5 (`simInflightResidence` now persisted in the SIM
   refl block of `OVERNIGHT_node2.yaml` + `SIMULATED_node2.yaml`). The §4.5-validated sim `152648` was paired
   against the *old* real `034934`; re-run both for a clean verdict. Validate emergent stays green; watch P3.

**DISCREPANCY LEDGER — flame-vs-reference Oort differences. RECONFIRMED Jun 16 against BOTH references
(`third_party/Oort/oort/oort.py` = the paper's standalone Oort; `third_party/REFL/thirdparty/oort/oort.py`
= REFL's own Oort fork, which is what flame's `refl_oort` should match). The two references DIFFER on
defaults (see D3/D4) — so "upstream" is per-baseline: standalone for the `oort` baseline, REFL fork for
`refl`. flame has ONE shared `OortSelector` inherited by both, so it cannot match both at once.**

| # | discrepancy | flame | standalone Oort | REFL fork | verdict |
|---|---|---|---|---|---|
| D1 | **`pref` not sorted** | FIXED (sort added, both `oort.py` + `async_oort.py`) | `sorted(...)` `oort/oort.py:272` | same (`thirdparty/oort/oort.py`) | **BUG (port error), fixed — UNRUN** |
| D2 | **stat-utility not normalized/clipped** | **FIXED Jun 16** (`scoring.oort_norm_stats`/`oort_normalize_reward`; wired into `oort.py` + `async_oort.py` `calculate_total_utility`; config `normalize_reward`/`clip_bound`) — was raw `PROP_STAT_UTILITY` (~73.7) → temporal term inert | `creward=min(reward,clip_value); sc=(creward−min)/range + temporal` then `if dur>pref: sc*=(pref/dur)^penalty` (`get_norm` @ `oort.py:286-308,394`) | same structure (REFL uses `abs(sc)`) | **BUG (logic omission), FIXED — UNRUN.** Parity-NEUTRAL → pure FIDELITY. Guard `TestRewardNormalization`. |
| D3 | `round_threshold` | **FIXED**: config-driven, default **10** (paper); refl override **30** | **10** (`argParser.py:52`) | **30** (`argParser.py:99`) | now per-baseline correct (oort/felix=10, refl=30) |
| D4 | `cut_off_util` | **FIXED**: config-driven, default **0.7** (paper); refl **0.05**; AND `cutoff_util` now thresholds the exploit-boundary score (was inert) | **0.7** (`argParser.py:105`) | **0.05** (`argParser.py:127`) | structural + value; refl exploitation also rewritten to the fork's weighted augmentation |
| D5 | temporal time-base | `last_selected_round` — **DEFERRED** (needs aggregator-stamped last-UPDATED prop; subtle, high blast radius) | `time_stamp` = round util UPDATED on completion (`oort.py:215,296`) | `training_round` (same) | the one remaining gap; flagged in code |
| D6 | `clip_bound` | **FIXED**: config-driven, default **0.98** (paper); refl **0.9** | **0.98** | **0.9** | per-baseline correct |
| S1 | `cutoff_util` index | **FIXED**: exploitLen-th HIGHEST score (ref) | `scores[sortedClientUtil[exploitLen]]` desc | same | flame had indexed near the bottom → factor inert |
| S2 | refl exploitation | **FIXED**: cut_off_util-augmented utility-weighted `np.random.choice` | n/a (oort uses `sample_by_util`) | `thirdparty/oort/oort.py:316-355` | was deterministic top-k (ignored cut_off_util) |
> `round_penalty=2.0` (orig exponent) **== flame `alpha=2`** for `system_util` — MATCHES, not a bug.
>
> **The shared-selector problem (the crux of "are we respecting the base implementations?").** flame has ONE
> `OortSelector` inherited by both the `oort` baseline (should match standalone Oort) and `refl` (should match
> the REFL fork). The two references DIFFER on defaults (round_threshold 10 vs 30; cut_off_util 0.7 vs 0.05;
> clip_bound 0.98 vs 0.9), so flame **cannot match both at once with code defaults.** Today the code defaults
> follow REFL (round_threshold=30) — so the **`oort` baseline does NOT match the oort paper's defaults.** All
> these knobs came from `oort.py.__init__` CODE defaults, NOT config. **RESOLVED Jun 16b:** all knobs are now
> `selector.kwargs`-driven, defaulting to the Oort paper (`scoring.OORT_PAPER_DEFAULTS`); refl overrides to the
> REFL fork in `OVERNIGHT_node2.yaml`. So each baseline matches its own reference.
>
> **Status of D1–D6 + structural after Jun 16b:** D1, D2, D3, D4, D6, and the two structural fixes (cutoff index;
> refl weighted exploitation) are **FIXED — UNRUN**. **Only D5 (temporal time-base) deferred** (documented). REFL's
> `abs(sc)` is subsumed (post-D2 scores are ≥0, so abs is a no-op); the explore/exploit split + augmentation now
> match the fork. All guarded (`TestAlgorithmHyperparams`, `TestRewardNormalization`, cutoff test). 246 + 22 PASS.

### Status: **Jun 14 — FULL 14400 s (4 h) runs (n=300, α=0.1)** — the run-length caveat RESOLVED

> **Read first.** These are the full **14400 s** runs the original tables were built for; they
> supersede the Jun-13 2700 s section. The 2700 s caveat predicted that several oort/refl/feddance
> selection/CONTROL checks *passed only because the horizon was too short for the selection-mix to
> accumulate* — **the 4 h runs bear this out**: `--batch` now roots **feddance, oort, and refl all at
> `trainer_speed` (P3)**, the commit-weighted footprint of the same selection-mix / in-flight-residence
> divergence. feddance's "40/40 ALL GREEN" at 2700 s was the horizon artifact — at 4 h it fails
> P3·K3·K2·terminal·commits. Pairs (`--batch` auto): felix sim 010018 / real 033735, refl sim 010048 /
> real 034934, oort sim 075449 / real 081648, feddance sim 081900 / real 083925. `--budget-s 14400`.
> `pytest tests/mode/ + scripts/parity/` = **58 PASS** (no code changed; checker run only).

**`--batch` summary (4 h, 14400 s, agg-goal 10):**

| baseline | rounds r/s | K2 | K3 | P3 | overall | ROOT (banner) | reduced to |
|---|---|---|---|---|---|---|---|
| **felix**    | 3421 / 3454 | ✓ | ✓ | ✓ | FAIL | **staleness** | **ONE miss — staleness** (gap landed) |
| **oort**     |  747 /  748 | ✓ | ✓ | ✗ | FAIL | trainer_speed | in-flight **residence** (quantified) |
| **refl**     | 10158 / 9574 | ✗ | ✓ | ✗ | FAIL | trainer_speed | eligible-**pool** comp + genuine loss |
| **feddance** |  464 /  513 | ✗ | ✗ | ✗ | FAIL | trainer_speed | selection-mix (masked at 2700 s) |

**felix — `simRedispatchGapSeconds 1.0→0.6` LANDED; whole throughput family GREEN, one miss left.**
At 4 h: K3b residual −0.05 (rel .011 ✓), K3 advance .075 ✓ (sim 4.17 ≈ real 4.12 — the 2700 s
over-advance 4.35 is gone), K2 rounds 3454/3421 (rel .013 ✓), terminal .013 ✓, commits .012 ✓, P3 .084 ✓,
participation .03 ✓, loss .016 ✓. **The single FAIL is staleness: sim 3.94 vs real 2.80** (KS shape .086
— close; a mean/tail shift, not a shape divergence). The post-commit gap is **over-loaded** — it sets
*both* advance (K3b) and the past-dated bunching that drives the staleness tail, and they want opposite
values: gap **1.0** → advance 4.35 (K3b ✗) / staleness **2.96 ✓**; gap **0.6** → advance 4.17 (K3b ✓) /
staleness **3.94 ✗**. Lowering the gap to fix advance re-bunched completions and regrew the §3j
past-dated tail (steady-state 12.54/4.17 ≈ 3.0; measured 3.94 = +0.9 tail). **They must be decoupled.**

**oort — in-flight RESIDENCE QUANTIFIED (the `inflight_residence` event populated):**

| metric (oort, per round) | sim | real | read |
|---|---|---|---|
| **`in_flight_after`** (carried stragglers) | **0.09** (median 0) | **2.40** (median 2, max 10) | sim drains to ~0; real holds ~2.4 |
| `residence_rounds` (rounds in-flight) | mean 0.01, max 2 | mean 0.17, max 8 | real stragglers reside longer |
| `carried_over_ages` count | 66 | **1792** | real carries ~27× more across rounds |
| selection_detail `mean_chosen`/`in_flight` | 13.09 | **15.39** | the 2.3 gap **== the 2.40 carried** |

Mid-run sim every round `before=13 → after=0`; real `before=13–16 → after=0–3`. **⚠ SUPERSEDED (see
§4.5):** this `in_flight_after` gap turned out to be a **false signal** — the round-300 log trace showed
sim's slow stragglers live in the persistent `_sim_buffer` (committing late as stale), not in
`selected_ends`, so 0.09-vs-2.40 compares different structures. The selected-vs-pool speed check then
showed oort's **pool already ≈ real** (12.12 vs 11.35) and the divergence is the **selector** picking
~average (12.21) vs real fast (8.81). **oort is a selector-scoring case, not a residence/carryover one**
— it gets §4.8-style localization, not the §4.5 slot fix. (The §4.5 fix applies to **refl**, below.)

**refl — eligible-POOL composition, worse at full horizon; loss now genuinely diverges.** A2b
`eligible_speed` sim pool **12.32 s** vs real **6.43 s** (KS .429, up from .363 at 2700 s — accumulates
as predicted). num_chosen/in_flight and staleness still match → it's the pool's speed mix (sim frees
busy clients back to the pool too early → slow re-enter → sim pool skews slow). participation
matched-count-KS **.316 ✗** (genuine), **convergence_loss now FAILS .1735** (>.15; the 2700 s .092 was
the ~2-eval weak pass — 192 evals at 4 h show the real divergence). This is the genuine `sct`-gated
pool-exclusion case (§4.5) — confirmed by selected-vs-pool speed: refl's *selected* matches real
(6.90/6.30) but its *pool* is full-population (12.41) while real's pool ≈ selected (6.56).

**feddance — the 2700 s "all-green" was the horizon artifact.** At 4 h: P3 .147 ✗ (root), K3 advance
**KS .522** (mean_rel .077 — shape diverges hard while means stay close), K2 rounds 513/464 (sim
**+10.6%**, 28.11 s/round vs real 30.4), terminal/commits +7.4%. **But feddance is NOT the residence
mechanism:** it selects exactly k=10 with **no overcommitment** (in_flight 10/10 both modes, no
`inflight_residence` telemetry). Its root is the **selection mix** — *which* 10 are picked — which drives
both the max-of-K advance **shape** (K3 KS .522) and the commit-weighted speed (P3 .147). See §4.8.

> **The roots are different per baseline (do not over-unify):** **refl** = `sct`-gated **pool composition**
> (selected matches 6.90/6.30, pool 12.41 vs 6.56) → §4.5 pool-exclusion fix (IMPLEMENTED, refl only).
> **oort** = **selector scoring** (pool ≈ real 12.12/11.35, but sim selects ~average 12.21 vs real fast
> 8.81) → §4.8-style utility-score localization, **not** §4.5. **feddance** = no overcommit → **selection
> mix** in the `feddance_I` (loss-utility) term, likely path-dependent (§4.8). **felix** = solved on
> throughput; residual gap↔staleness coupling held for a dedicated pass (§4.6). Only **refl** has a coded
> fix + re-run this round; oort/feddance need score localization first; felix is held.


### Distilled history — what worked / what didn't (Jun 12–14)

**Worked (kept, validated):**
- **felix §3j drain-by-readiness** — admit any in-flight end with a physically-ready msg into the
  reorder buffer so slow trainers commit in `sct` order (staleness 7.19→3.45). `_sim_recv_min`.
- **felix §3g probe the LIVE in-flight set** (recv_fifo pops min-`sct`) — reorders the past-dated tail.
- **felix §3m `realDistributeSettleSeconds=0`** — removes a real-only 2×`sleep(0.1)`/commit so real
  holds ~c computing (advance 4.14, staleness 2.80) instead of slowing sim to a wrong reference.
- **felix §4.1 redispatch gap** — post-commit re-dispatch leg as `simRedispatchGapSeconds`, slot-held
  by §3L cooling (freed slot isn't refilled from idle). `1.0` fixed K3/K2/terminal; **`1.0→0.6` (Jun 14)
  zeroed the over-advance** → throughput family GREEN. Residual: staleness (see Status).
- **Checker corrections (principled, not relaxation):** throughput family → one 5% bar (K2/K8/U2 are the
  same rounds/V quantity); S2 share-KS → **matched-round count-KS** (share folds in the throughput
  offset); A2b `eligible_speed` pool-composition check; P1/S1 gated WARN for stochastic selectors;
  F1-3 enforce **pooled** utility KS (per-trainer KS=1.0 was n<=2 mechanical); `phase_mqtt_fetch` → DIAG
  (in-mem cache wall-time, off the virtual clock). `validate_real` + the `task_send` event prove real is
  admissible (concurrency 28.8/c30, **double-dispatch 0**).

**Didn't work / dead ends (do NOT re-try):**
- overhead>0 on the clock (§3a/b — masks & drifts; clock must `=max(vclock,sct)`, overhead 0).
- prediction-only gates with no real blocking (§3d/e — never fire).
- `version_at(sct)` relabel (§3f — fedbuff consumes the *real* staleness number; relabel is inert).
- adding `mqtt_fetch` (57 s) to `sct` (§3h — not version-relevant, inflates staleness ~6×).
- §3m's theory that settle=0 lets sim match at **gap=0** — refuted (gap=0 → sim over-overlaps:
  advance 3.68, staleness 4.75). The gap is a real mechanism, not a fudge.
- `sendStaggerSeconds` (always 0; MQTT drops ruled out) — REMOVED.
- the bogus "redispatch 4549/26988" breach — a `validate_real` bookkeeping artifact (cleared in-flight
  on `contributing_trainers`, which omits overcommit-discarded completions), NOT a real violation.

**Ruled out (do not re-chase):** GPU contention (T3 overrun=0); SEND_TIMEOUT (fires 0×); `commit_gap`
as a staleness proxy (track staleness, not commit_gap); MQTT drops (0 both directions); the felix
post-compute leg as a "bug" (serial-aggregator scheduling, not network).


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

---

## §3  Felix sim parity — root-cause log

**Real pipeline (n=300, ~50-min):** trainer computes ~11.7s, then a ~1.0s
post-compute leg before its next dispatch → real **cycle ~12.96s**, advance 4.32s,
all_selected ~30 (of which ~27.6 actively computing), **staleness 2.81, commits in true
completion order**. The whole job is to reproduce that in sim where the trainer does
*not* sleep its budget (computes in real GPU ms, stamps a modeled completion `sct`).

**Why all_selected=30 but only ~27.6 compute — verified from LAG_DECOMP (Jun12).** The
post-compute leg is **serial-aggregator scheduling, NOT network.** `wall_lag` (send→commit) =
compute 11.92 + push/mqtt/post ~0.1 = **12.01s** (MQTT round trip <0.1s); + `queue_wait` 0.55s
(serial recv-queue backlog, pre-commit, counts toward staleness: 12.01/4.32≈2.78≈real 2.81) +
re-selection/send/`sleep(0.1)` ~0.4s (post-commit, doesn't count). Cycle 12.96 → computing
fraction 11.92/12.96 ≈ 0.92 → ~27.6 of 30. Drops ruled out (receptions ≥ dispatches,
SEND_TIMEOUT=0). The leg is not a bug — §3m speeds real up by removing the artificial part.

### §3m  Speed real up to ~c computing instead of slowing sim (Jun12, preferred)

The dominant slice of the post-compute leg is removable: a real-only `time.sleep(0.1)` before
selection, hit twice per commit (~0.2s ≈ 46% of the 0.43s/commit budget), which inflates
`queue_wait` and the re-dispatch gap. New knob `realDistributeSettleSeconds` (default 0.1 =
legacy), set to **0** for felix → aggregator compute-bound, real holds ~30, advance ~3.97,
staleness ~3.0. **Validated (Jun13):** real came down to advance 4.14, staleness 2.80 ✓.
**But the sim-side prediction was wrong** — with `simRedispatchGapSeconds=0` sim *over-overlaps*
(advance 3.68, staleness 4.75; K3b rel 0.112), so the §3k/§3L gap was load-bearing after all.
See §4.1: restore a small gap.
**CPU pinning:** aggregator already pinned to a reserved set (`runner.py`); do NOT pin to 1 core
(its MQTT thread shares CPU with the FL loop). The loop is latency-bound, not CPU-bound, so
pinning is second-order — the sleep removal is the win.

### Changelog (done / tried-and-failed — do not repeat)

| # | change | result |
|---|---|---|
| §3a | overhead 0.50→0.16 | 0.50 over-charged (clock = pure overhead ramp, staleness 9.3, rounds −40%); 0.16 then under-charged |
| §3b | overhead 0.16→0.315 | advance 2.98→4.95 matched real 4.52; staleness still 7.19 |
| §3c | **overhead → 0** (`simCommitOverheadSeconds=0`) | clock now TRACKS completions (`vclock=max(vclock,sct)`); overhead had been 98.8% of the clock. **KEPT.** |
| §3d | prediction gate (wait for early in-flight) | **inert** — predictor learned contention-inflated `SIM_ROUND_DURATION`; never fired |
| §3e | gate predictor → `TRAINING_BUDGET_S` | gate fires (`gate_holds`>0) but staleness still drifts |
| §3f | `version_at(sct)` relabel band-aid | **REVERTED** — felix `fedbuff` consumes staleness for `alpha` (40% of rate); relabel only rewrites telemetry. Proved root cause = commit ORDER (run 150131: advance+throughput matched, staleness still 7.19). |
| §3g | **probe the LIVE in-flight set** (not the stale `recv_ends` snapshot) | **WORKS by reordering** — `recv_fifo` pops min-`sct` buffered msg first → staleness 7.19→3.57 PASS, acc/loss PASS. **KEPT.** |
| §3i | **`simCompletionLegSeconds=1.6s`** added to `sct` only (not compute/budget) | advance 3.68→4.00 (mean PASS, throughput/terminal PASS) but **staleness 3.57→4.09 FAIL**. Leg is a **tuned scalar** (calibrated to real W−compute) → models real latency but as a fudge; raised the staleness baseline. UNDER REVIEW — replace with a principled latency model or back out (see §4.1). |
| §3k | **split the 1.6s leg into pre-commit holding (0.6) + post-commit re-dispatch gap (1.0)** | **PARTIAL on run 175309: staleness 3.45→3.16 (KS 0.078→0.042) BUT advance 4.24→4.03, KS 0.172→0.208 FAIL.** The gap was implemented as pure availability-exclusion → with a 300-trainer pool the selector **refills the freed slot from idle trainers**, so the gap is INERT on throughput (computing concurrency pinned at c=30, `inflight_tracked`≈30). Effective cycle collapses to holding → advance low, staleness ≈ F/g. Corrected by §3L. |
| §3L | **cooling trainers hold a concurrency slot (no idle-pool refill)** | **implemented; inert in the Jun13 run because §3m set the gap to 0.** `extra = max(0, c − in_flight − cooling_count)` in `async_oort._handle_send_state`; aggregator stamps `channel.properties["sim_cooling_count"] = len(_cooling)` in `_distribute_weights`. Predicted advance ≈4.4 / staleness ≈2.77 **only with a non-zero gap** — the Jun13 run ran gap=0 (§3m) and got advance 3.68 / staleness 4.75, so §3L never engaged. §4.1: re-test §3L with the gap restored. Guarded by `TestCoolingHoldsConcurrency`. |
| §3j | **drain by physical READINESS, not predicted completion** (`_sim_recv_min`) | **VALIDATED on run 161419: staleness 4.09→3.451 PASS** (KS 0.078, tail collapsed), advance KS 0.218→0.172 PASS, terminal/commits/acc/loss PASS. §3g drained in-flight ends only within `buffered_min + slack`, so a SLOW trainer (budget 38-56s, far-future `exp`) was excluded even though its message had physically arrived (wall_lag ~0.1s) → it sat undrained until the clock passed its `sct` → committed past-dated, staleness 33 vs real ~9 (same budget). Tell: tail commits have **residence≈0** (NOT buffer-resident) but **gap≈45**, corr(res,gap)=−0.27. Fix: also admit any in-flight end with a non-empty rxq (`_sim_end_has_ready_msg`), so slow trainers buffer as FUTURES and commit in `sct` order. recv_fifo on a ready end returns immediately → no added blocking. Guarded by `test_drains_ready_inflight_above_ceiling`. |

### §3k  Residual staleness gap → post-commit re-dispatch gap (Jun12)

After §3j, staleness PASSES but sim still sits 0.6 high at the mean (3.45 vs 2.81) and
wider at the tail (p99 19 vs 9). Root-caused from the real LAG_DECOMP, not tuned away.

**The model (verified on real):** in steady state `staleness = holding / advance`, where
`holding` = send→commit time. Real check: holding 11.79s / advance 4.32 = 2.73 ≈ measured
2.81 ✓. Equivalently `staleness = (L/C)·(F/g)` with cycle `C`, in-flight `F`, agg_goal `g`.
`F/g` is matched (in_flight 30.8 both → 3.08).

**The bug:** the §3i leg put the WHOLE real post-compute cycle (1.6s) *before* the commit
(in `sct`). But the real cycle splits into a **pre-commit holding** part and a **post-commit
re-dispatch** part, and only the former counts toward staleness. Measured real decomposition
(LAG_DECOMP, full run): holding = compute 11.71 + queue_wait 0.61 + delivery 0.06 + post/mqtt
0.03 = **11.79s**, then a **~1.0–1.15s post-commit** re-dispatch leg (re-selection + model
push). So real L/C = 11.79/12.94 = **0.91** → staleness 0.91·3.08 = **2.81** ✓. Sim folded
everything into `sct` (instant re-dispatch) → L/C ≈ 1.0 → 3.08 + tail = **3.45**.

**The fix (principled, cycle-preserving):** split the 1.6 leg:
- `simCompletionLegSeconds` 1.6 → **0.6** (pre-commit holding = real queue_wait+delivery; in `sct`).
- `simRedispatchGapSeconds` 0 → **1.0** (post-commit; aggregator-side). A just-committed end is
  held out of selection until `vclock ≥ sct + gap` (`_sim_cooldown_until`, consumed in
  `_distribute_weights` via `set_curr_unavailable_trainers`), so it returns with a FRESHER
  model_version. The selector refills to keep c=30 in-flight, so the gap spaces completions
  (raises advance, de-bunches overlap) WITHOUT counting toward the committed update's staleness.

Total cycle compute+0.6+1.0 = compute+1.6 is **unchanged**, so advance/throughput/terminal stay
matched (they currently PASS); predicted staleness → (compute+0.6)/advance ≈ **2.84** vs real
2.81. Not a relabel (§3f dead-end): the update genuinely commits earlier (smaller `sct`), so the
real number `fedbuff` consumes drops. Guarded by `TestRedispatchGap` in
`tests/mode/test_async_sim_ordering.py` (3 tests: cooldown recorded at sct+gap; off-by-default
inert; commit order/clock unchanged). **Status (Jun13):** not exercised — §3m turned the gap off
(`simRedispatchGapSeconds=0`), and that run confirms the gap is needed (sim over-overlaps without
it: advance 3.68 vs 4.14, staleness 4.75 vs 2.80). §4.1: turn the gap back on and re-run.

### §5  Checker category-errors (Jun12) — the real "final disparity"

Once §3j made the sim **dynamics** match, the residual FAILs were all the checker
enforcing *exact identity* on quantities a stochastic / in-memory simulator cannot
reproduce. Diagnostic tell: each was byte-identical across sim 120543→161419 despite a
large staleness change → not downstream of the dynamics. Fixes (in `scripts/parity/`):

| # | check | was | fix | why it's correct |
|---|---|---|---|---|
| §5a | **P1 `aggregation_sequence`** | hard DIST FAIL (exact_set_match 0.0) | **gate to WARN for stochastic selectors** (mirror S1's `DETERMINISTIC_SELECTORS` idiom) | The ladder spec (§2 Stage 6) already says P1 is "gated for stochastic"; the impl never did. AsyncOort picks different rounds across modes → exact per-round set identity is unattainable. S2 participation (PASS) is the enforced invariant. |
| §5b | **F1-3 `utility`** | hard FAIL (max_KS **1.0**, per-trainer) | **enforce the POOLED utility KS** (0.037 ✓); per-trainer max_KS becomes a *gated* diagnostic over trainers with ≥10 commits both modes | The 43 KS=1.0 trainers were all 1–6-sample (KS=1.0 is mechanical for n≤2); their means were identical (~2.2–2.4). Pooled distribution is the path-independent fidelity measure; per-trainer identity is path-dependent for a streaming selector. |
| §5c | **`phase_mqtt_fetch`** | enforced MECHANISM/DIST FAIL (real 57.5s vs sim 24.8s) | **reclassify to DIAG** (WARN) with a note | Pure network-I/O wall time; sim serves weights from an in-mem cache and folds the trainer cycle into budget+leg, so this phase is deliberately *not* part of the virtual clock (§3h). Comparing it is apples-to-oranges. gpu_compute + other modeled phases stay enforced DIST. |

Result: felix **ALL CHECKS PASSED**. Guard: `test_ladder.py` clean-pair + phase-split
tests still pass (per-phase `ok` unchanged; only tiers/gating moved). These are
append-only checker corrections, not tolerance loosening — a future *deterministic*
selector still gets exact-set + per-trainer enforcement via `DETERMINISTIC_SELECTORS`.

**Dead ends — do not retry:** overhead > 0 on the clock (§3a/b — masks & drifts);
prediction-only gates with no real blocking (§3d/e — never wait); `version_at(sct)`
relabel (§3f — fedbuff uses the real number); adding `mqtt_fetch` (57s) to `sct`
(§3h — not version-relevant, would inflate staleness ~6×). `phase_mqtt_fetch` FAIL is
a benign wall-time artifact (in-mem cache is faster), orthogonal to the virtual clock
— NOT chased. A2 eligibility point-mass handled (means match + CV<0.01 rescue).


## §4  Plan — code changes for the next 4 h run (Jun 14)

**Overriding principle (§4.0):** parity ≠ goal; a *correct* simulator is. Real is the reference only
after `validate_real` shows it admissible (done: concurrency 28.8/c30, double-dispatch 0). Never tune
sim to a wrong real; never add a scalar fudge where a mechanism is called for.

**Already landed (config/checker/instrumentation):** §4.0 `validate_real` + `task_send`; §4.1 felix
`simRedispatchGapSeconds 1.0→0.6` (throughput family green); §4.3 participation matched-round count-KS;
A2b `eligible_speed`; oort `inflight_residence` event (now populated — gave the calibration target).

**Rule (from this session): one change per baseline per run; the change is either a strong-candidate FIX
or instrumentation to GET MORE DATA; only re-run baselines we actually change.** The roots are
*different per baseline* (verified from this run's telemetry), so the changes are independent code paths
and can ship together without cross-contamination:

| baseline | stack | root (from data) | one change | type | re-run? |
|---|---|---|---|---|---|
| **felix** | async (async_oort/fedbuff) | gap↔staleness coupling; staleness shifted at every quantile (med 3 vs 2, **p99 28 vs 9**), `pastdated=0` so it's **buffer-depth aging** (buf_depth 26.6), not past-dating | **HOLD** (no confident single knob; see §4.6) | — | NO |
| **oort** | sync (oort/fedavg) | pool already ≈ real (12.12 vs 11.35); **selector picks ~average (12.21) vs real fast (8.81)** → selector-scoring/path issue, NOT pool | localize oort utility scores (like §4.8) | DATA | NO (no §4.5) |
| **refl** | sync (refl_oort/refl) | selected matches (6.90/6.30); busy client re-enters the pool before `sct` → **pool skews slow (A2b 12.41 vs 6.56)** | §4.5 sct-gated pool exclusion | FIX | YES (sim) |
| **feddance** | syncfl (feddance/fedavg) | **no overcommit** (in_flight 10/10) → not residence; selection-mix drives advance **shape** (K3 KS .522) + commit-speed (P3 .147) | §4.8 selector-score localization | DATA | YES |

### §4.5  refl — `sct`-gated pool exclusion (IMPLEMENTED)  [FIX]
**Principle (from the user, Jun 14): an update must become *available to the aggregator* at the time it
would in real — at the trainer's modeled completion `sct`, not at its (instant) physical arrival in sim.**
A trainer that has physically sent but is modeled as still computing (`vclock < sct`) must NOT re-enter
the eligible pool — in real it is busy that whole time. **The data scoped this to refl, not oort** (oort
note below): refl's *selected* speeds already match (sim 6.90 vs real 6.30) but its eligible **pool**
skews slow — sim **12.41 s** (≈ full population) vs real **6.56 s** (≈ selected), because in real the slow
clients are busy/out-of-pool while in sim they re-enter the pool before `sct`. Its in-flight *count* even
matches (53.55/53.65); it is purely *pool composition*.

**Commit *order* is already `sct`-gated** (`SimReorderBuffer.pop_min` commits the K-smallest-`sct` =
real's K-fastest). What is not is the **pool re-entry** of a still-computing trainer.

**Change (IMPLEMENTED, config-gated `simInflightResidence`, default off):** in
`oort/top_aggregator._distribute_weights`, in sim, compute the **still-computing set** =
`self._sim_buffer.pending_after(vclock)` (buffered ends with `sct > vclock`) and add them to the
**`trainer_unavail_list`** passed to `set_curr_unavailable_trainers` — the selector (`refl_oort`:145,
`oort`:443) excludes the unavailable list from candidates, so they stay out of the pool until `vclock ≥
sct`, then commit normally via the buffer. **Crucially via the *unavailable* path, NOT `selected_ends`** —
adding to `selected_ends` would put them in `select()`'s return and **re-dispatch** them (the send loop
weights every returned id), resetting their `sct`. Bounded: released at `vclock ≥ sct` (budget ≤ ~56 s ≈
≤3 rounds), so no "177 accumulate". Files: `config.py` (`sim_inflight_residence`),
`sim/virtual_clock.py` (`SimReorderBuffer.pending_after`), `oort/top_aggregator.py`, `baselines.yaml`
(refl only). **Target:** refl A2b pool 12.41→~6.5 → closes P3/participation/loss. **Guard:**
`test_virtual_clock.py::test_pending_after_*` + `test_sync_sim_ordering.py::TestSimInflightResidence`
(held set = `sct>vclock`, released after; merge is additive; off-by-default inert).

**oort note (NOT §4.5 — selector scoring; LOCALIZED to `system_util`).** oort's eligible **pool already ≈
real** (sim 12.12 vs real 11.35), so pool exclusion is the wrong lever and is left **off** for oort. The
new Stage-3 checks (below) localize it from the *existing* run: **A2c selection-bias** — real picks 2.54 s
faster than its pool (`bias −2.54`) while sim picks at pool-average (`bias +0.09`), pools matched →
selector, not composition; **Sx selector-score** — `believed_I` matches (KS .03) and `temporal` matches
(.01), but **`system_util` diverges (KS .132, sim 0.921 vs real 0.865)**. The Oort score is
`(believed_I + temporal)·system_util`; `system_util` is the speed/staleness penalty. Sim's is **higher**
(less penalty) → sim under-penalizes slow trainers → picks ~average instead of fast. **oort's root is the
`system_util` computation in sim** (`oort_system_utility(round_duration, round_preferred_duration,
alpha)`). Both inputs are duration-derived and differ across modes: **`PROP_ROUND_DURATION` is the MODELED
`sct`-leg in sim but WALL time in real**, and **`round_preferred_duration` is recomputed DYNAMICALLY each
round** (a percentile of candidate durations), so the target itself differs. **Instrumented (Jun14c):**
`oort.py` now emits `round_preferred_duration_s` + `alpha` on the selection event so the next oort sim run
pinpoints whether the `system_util` gap is the *target* (pref) or the *round_duration input*.

### Jun 14c — selection localizers (A2c, Sx) + oort instrumentation [DONE for oort/refl; EXTEND to others]
Turned this session's ad-hoc diagnostics into **standing parity checks** (the growth rule, §1) so any run
auto-localizes a selection divergence — and instrumented oort's missing score input. **All landed,
guarded (`test_ladder.py` +5, 79 parity/sim/mode tests PASS); no run needed (work on existing telemetry):**
- **A2c `selection_speed_bias_parity`** (`checks.py`, MECHANISM/DIST, dep A2b): selected-vs-pool speed,
  per mode, with `bias=selected−pool`. Distinguishes **pool composition** (A2b fails → refl) from
  **selector scoring** (pool matches but bias diverges → oort). Validated on the existing runs: oort bias
  **real −2.54 / sim +0.09** (pools 11.35/12.12 matched) → selector; refl selected matched (KS .098), pool
  **6.43/12.32** → composition.
- **Sx `selector_score_parity`** (DIAG, dep A2b): per-score-term KS over selected candidates (oort
  believed_I/temporal/system_util; feddance V/I/A/U). Localizes the diverging term: **oort → `system_util`
  KS .132**; refl → all terms match. `participation` now also deps on A2c.
- **oort instrumentation:** `round_preferred_duration_s` + `alpha` on the selection event (above).

**Status across baselines (these localizers are general; only oort/refl exercised so far):**
| baseline | A2c / Sx read | done? |
|---|---|---|
| **refl** | pool composition (A2b) → §4.5 fix IMPLEMENTED | ✅ fix in; run validates |
| **oort** | selector scoring → `system_util`; pref/alpha now instrumented | ✅ localized; **fix pending** the round_duration/pref input |
| **feddance** | Sx should localize `feddance_I` (matches the §4.8 manual finding); A2c not yet run on it | ⬜ wire feddance through A2c/Sx; confirm `feddance_*` keys land |
| **felix** | async stack — A2c/Sx untested; its open issue is staleness (§4.6), not selection | ⬜ run A2c/Sx on felix to confirm selection is clean |

> **What the oort+refl sim run is FOR (and isn't):**
> - **refl — VALIDATE the §4.5 fix** (the only coded change). Pass criteria: `[SIM_RESIDENCE]` log fires;
>   A2b pool **12.41 → ~6.5**; A2c selected still matches; P3/participation/convergence_loss flip to PASS.
> - **oort — DIAGNOSTIC only, not a fix run.** No oort behavior changed; the run just *populates the new
>   `round_preferred_duration_s`/`alpha` instrumentation* so we can localize the `system_util` gap to the
>   target vs the round_duration input. (If you'd rather not spend the run, the *existing* oort pair already
>   localizes to `system_util` via Sx — the re-run only adds the pref/alpha breakdown.)
>
> **Is everything instrumented?** **refl: yes** — fully covered (A2b + A2c + Sx + the fix), run validates.
> **oort: now yes** — the one gap (`round_preferred_duration`/`alpha`) is added; after the run, the
> remaining question is purely *which* duration input to align (modeled `sct`-leg vs wall), which the
> breakdown answers. **feddance/felix: not yet** — extend A2c/Sx to them next (table above).

### §4.6  felix — HOLD (no confident single change this run)
Throughput family is GREEN; the only miss is staleness (3.94 vs 2.80). It is a **gap↔staleness coupling**:
larger gap ↑advance AND ↓staleness, so the one knob can't hit advance 4.12 (wants gap≈0.6) and staleness
2.80 (wants gap>1.0) together. `pastdated=0` rules out the §3j tail; the excess is **buffer-depth aging**
(buf_depth 26.6, p99 28 vs 9) — i.e. the cooling/slot-hold dynamics, not a scalar. The decoupling knob
(leg, in `sct`, shifts holding not advance) is too weak (0.6→0 only buys ~0.14 of 1.14). **No one-line
fix; do not re-run felix** until a dedicated investigation of the buffer-aging vs gap is done. Dead ends
that re-break the green family: raise the gap; add a leg scalar; revert to gap 1.0 (re-fails throughput).

### §4.7  (merged into §4.5)
The refl pool fix is §4.5 above; oort is reclassified there as a selector-scoring case (§4.8-style),
not a slot/pool change. Nothing separate here.

### §4.8  feddance — selection-mix localized: same pool, real picks slower  [DATA → analysis done]
feddance has no overcommit (in_flight 10/10), so residence does **not** apply. **Localized from existing
`selection.per_trainer` telemetry (matched first 400 rounds):** the candidate **pool speed matches**
(sim 12.13 s vs real 12.20 s), but **real selects slower trainers (mean 13.32 s) than sim (12.42 s) from
that same pool** — the FedDance selector *ranks the identical pool differently*. That slower-pick mix is
exactly what lengthens real's max-of-K advance (30.4 vs 28.1 s/round, K3 KS .522) and raises commit-speed
(P3 .147). **Term localized (matched 400 rounds, V/I/A/U):** `V` is identical (0.967 both); A/U are tiny
(~0–2). The **`feddance_I` term dominates** (magnitude 82–90) and is where the modes split — real selects
**I=90.0** vs sim **82.6** (pool-I real 30.3 vs sim 28.1). `I` is the path-dependent age term
(`last_engaged_round`): real picks higher-I = staler = slower trainers. **Resolved — `I` is the trainer's last-round training LOSS (Oort
stat_utility), not an age term** (`feddance.py` docstring `I_m`; set via `on_update_received` → `PROP_I`).
That event fires **on commit in both modes** (`syncfl/top_aggregator` calls it identically), and sim does
**real** GPU compute so the loss is drawn from the same process — so the I update point is **not** biased;
the divergence is **path-dependent stochastic accumulation** (real picks higher-loss trainers; over 4 h
the two selection histories drift). **Implication: feddance is likely NOT a sim correctness bug** — it is
the same stochastic-selector class as §5 (P1/utility): per-round / commit-weighted *mix identity* is
unattainable for a chaotic selector, so P3/K3 (commit-weighted EMERGENT outcomes) should be judged
**distributionally**, like participation (matched-count-KS, which feddance PASSES). **One thing left to
rule out:** the `mean_I`/`prev_round_mean_I` normalization buffer accumulates over a *different round
count* at matched time (sim 513 vs real 464) — check it doesn't horizon-bias the I ranking. If clean,
the feddance fix is a **checker reclassification (P3/K3 → DIST for stochastic selectors)**, not a sim
change, and feddance needs **no re-run**.

### Sequencing
1. **felix:** no change, no re-run.
2. **refl (IMPLEMENTED):** §4.5 `sct`-gated pool exclusion, config-gated `simInflightResidence=True` (refl
   only), guarded. `pytest tests/sim/ tests/mode/ scripts/parity/` green under `dg_flame` (76 pass).
3. **oort:** NOT §4.5 (pool already matches). Localize the oort selector utility scores (§4.5 oort note,
   §4.8-style) before any oort change — **not re-run** this round.
4. **feddance:** no code/re-run — first rule out the `mean_I` horizon bias (§4.8); likely a checker
   reclassification, not a sim change.
5. Launch **refl sim only** (real is unchanged — pair against the existing 4 h real `034934`). 2 h reads
   the pool engagement (A2b per-round); 4 h reads the downstream verdicts (P3/participation/loss).
   `parity_check.py --real experiments/<refl_real_034934> --sim experiments/<new_refl_sim> --budget-s
   <runtime>`; compare A2b pool 12.41→~6.5 and watch `[SIM_RESIDENCE]` log lines confirm the hold fires.

**Telemetry to watch:** `inflight_residence` (`in_flight_after`, `residence_rounds`, `carried_over_ages`),
`agg_round.staleness/vclock_now`, `selection.per_trainer.{speed_s,selected}`, A2b pool speed,
`plots/aggregation/*`. Run per-baseline plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Done (one-liners, kept for reference):** lazy weight-deserialize via `materialize_weights`
(`test_weights_bytes_roundtrip.py`); feddance `trainer_speed_s` from `wall_lag_s`; hot-path logs → DEBUG
(parity reads JSONL only); plotting overhaul (`analyze_run.py`, `PLOTTING.md`, `plots/aggregation/*`).
**Logging headroom (§4.4, after correctness):** aggregator `.log` ~220k lines/run (oort), ~105k
`[RECV_FIFO]`; none read by the checker — demote to DEBUG in a separate phase, measure wall delta +
`pct_of_floor`, then re-tune felix's gap to real's de-contaminated leg.
