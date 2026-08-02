# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 ported fedbuff/felix-lineage
baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3).
Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf, sim barrier redesign, delay-factor
calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/
gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — how to use this doc
>
> **Fresh session? Read in this order:** §F (locked invariants — don't re-derive) → §E (dead ends — don't
> retry) → §A (current scoreboard) → §B's per-baseline table + "Next session" block → §D (only the lessons
> relevant to the rung you're chasing) → §C if still stuck on ladder-walk mechanics.
>
> | section | contents | update rule |
> |---|---|---|
> | §A | scoreboard: pass/fail per baseline, ≤2-line caption | rewrite in place on every >3600s run |
> | §B | per-baseline open issues + ONE "Next session" block + the RUN PLAN | current-state only; an issue lives here XOR §G, never both |
> | §C | ladder/decomposition method, run-length budget | timeless; edit only if the method itself changes |
> | §D | durable lessons — positive, transferable invariants | ≤30 words each; update in place, never append near-dupes |
> | §E | dead ends — falsified hypotheses, do not retry | append-only, one line each |
> | §F | locked invariants — always-true / always-do | append/amend with OPERATOR APPROVAL + evidence; amend in place, never renumber (§F header) |
> | §G | closed items, one line each | move here the instant a §A/§B issue resolves, delete the source in the same edit |
>
> **Living doc, not a log — no dated annotations.** Every claim here must read as true right now (or believed
> true until evidence says otherwise), never as "as of <date>". §G is the one intentional exception: a
> recent-fixes ledger ordered newest-first by position, not by date stamp — full history is `git log` on
> this file.
>
> **Non-negotiables:**
> - Correctness per mode first; parity is the consequence, never the goal — for a MATCH or a DIVERGENCE
>   alike. A rung passing because both sides are equally wrong is a regression dressed as a green rung
>   (§D-5). A DIVERGENT rung names two disagreeing sides, never which is at fault — verify each side's own
>   absolute signal before deciding which to change (§D-9).
> - Parity findings/fixes only here — design decisions, roadmap items, calibration derivations belong in
>   FWDLLM_DESIGN.md.
> - Ground every claim in telemetry already on disk before instrumenting or running; fix root causes, not
>   symptoms; always use the `dg_flame` conda env for python/pytest/analyze_run.py; ship new telemetry with
>   its plot + pytest in the same change.
> - Runs happen on a separate operator-controlled node: never launch or babysit one yourself — print the
>   command instead. Assume no run is in flight unless told otherwise, and that each baseline's real/sim
>   pair runs ONE AT A TIME per node (different baselines may run in parallel on different nodes, so
>   cross-baseline directory timestamps can legitimately interleave — only same-baseline real→sim ordering
>   is meaningful).
> - Run artifacts live under `lib/python/examples/fwdllm/experiments/run_<timestamp>_<name>_<syn>_<real|sim>/`.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) (methodology + rung catalog §F),
[UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) (availability substrate),
[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md) (build plan/roadmap/calibration),
[fluxtune_contributions.md](fluxtune_contributions.md) §8 (training-stability/convergence ledger — check
before opening a new stability investigation here).

**Comparator — discovers the latest real/sim pair per baseline and runs the shared parity battery.
Pairs grade in parallel (`--jobs`, default one worker per pair):**
```bash
cd lib/python/examples/fwdllm/expt_scripts
python run_parity.py                        # DEFAULT IS ONLY fwdllm/fwdllm_plus/fluxtune, not all 9
python run_parity.py --baselines fluxtune fwdllm felix_it felix_round fedbuff_round \
    fwdllm_it_unaware fwdllm_it_oracular fedbuff_it_unaware fedbuff_it_oracular   # the full scoreboard
python run_parity.py --yes                  # skip the confirm prompt
python run_parity.py --validate             # + live-run checks (staleness/vclock_now)
python replicate_floor.py --mode real       # same-seed replicate spread -> DIST tolerance floor (§D-24)
python profile_sim_charges.py --real-run <real_dir> \
    --out ../sim_charge_profiles/<baseline>.yaml --only-observed   # re-profile ONE baseline's charges (§D-36)
```
Rung catalog: PARITY.md §F. **Not redefined there:** per-stage wall-budget instrumentation
(`drain_wall_budget`, `trainer_phase_wall_budget`, `step_timing_breakdown`, `aggregation_compute_wall`) is
ONE-SIDED (`sim<=real`) where sim should collapse a real-transport phase to ~0, DISTRIBUTIONAL where it's
genuine shared compute. Implementation-level reference (tiers, the `pctl_band_ok` band-escape primitive
and its `min_abs` calibration rule, full wall-budget/timing rung table):
`async_cifar10/scripts/parity/PARITY_CHECKER_README.md`.

---

## §A  Score

**Latest run per baseline** (`run_parity.py`; ✓/✗/– = pass/fail/skip; PARITY.md §F). **Mixed durations — the
rows do NOT compare to each other.** Four baselines have a post-fix **7200s** pair; the other five have only
the **1200s** Phase-0 smoke, whose matched budgets (N=5-57) are far below scoreboard strength. `fwdllm_plus`
has no run dirs left on disk. Open fails and root-cause: §B.

| baseline | run pair | dur | pass/fail/skip | N | cohort | vclock | K4 | slots | sbias | thru | commits | terminal | V1c | V1 | V2 | U3 | S2 | conv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260801_030932`/`_150819` | 7200s | **77/0/16** | 94 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm/syn_0 | `run_20260801_005640`/`_145728` | 7200s | **68/0/24** | 39 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_round/syn_0 | `run_20260801_055314`/`_155005` | 7200s | 72/2/18 | 176 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| felix_round/syn_0 | `run_20260801_084040`/`_163513` | 7200s | 71/3/18 | 214 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_it/syn_0 | `run_20260731_195405`/`_201604` | 1200s | 70/5/16 | 57 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| fedbuff_it_unaware/syn_0 | `run_20260731_180442`/`_182654` | 1200s | 72/3/18 | 38 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fedbuff_it_oracular/syn_0 | `run_20260731_184247`/`_190502` | 1200s | 73/2/18 | 38 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fwdllm_it_unaware/syn_0 | `run_20260731_163249`/`_165420` | 1200s | 66/1/25 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260731_170129`/`_172308` | 1200s | 66/1/25 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. Open fails: §B.

**The per-baseline charges closed 9 of 14 open fails, and `felix_round`'s divergence with them.** Re-running
only the four sim legs against the same reals (Phase 0') cut the charge by 0.61–1.90 s/round and took
`fwdllm` 67/1 → **68/0** (clean), `felix_round` 65/9 → **71/3**, `fedbuff_round` 71/4 → 72/2; `fluxtune` holds
77/0. `felix_round`'s `v1c` flipped **`diverging` → `flat`** (λ 0.190→0.031, t 5.69→1.47; per-bin ratio was
1.03→1.50 monotone, now 1.08→1.11) and its whole level family closed with it: `v1` 16.3%→7.3%, `v2` matched
4.98%→0.25%, `thru` 12.0%→4.9%, `commits` 15.8%→5.9%, `overhead_residual` 13.6%→5.2%. **This kills H7** (§E):
the residual was a per-cycle overcharge, not a trajectory effect. Pipelining stays green on all nine.

**`v1c_iter_drift_rate` now reads `flat` on all nine** (§G) — the per-bin slope of `ln(sim/real
iterations-per-bin)` t-tested against zero. No baseline is a cadence divergence any more:

| baseline | λ/100 units | t | verdict |     | baseline | λ/100 units | t | verdict |
|---|---|---|---|---|---|---|---|---|
| felix_round | +0.031 | 1.47 | flat | | fluxtune | +0.060 | 1.14 | flat |
| fedbuff_it_unaware | +0.420 | 1.68 | flat (N=38) | | fedbuff_round | −0.002 | −0.05 | flat |
| felix_it | +0.046 | 0.21 | flat | | fwdllm | 0.000 | 0.00 | flat |
| fedbuff_it_oracular | −0.023 | −0.06 | flat | | fwdllm_it_* | – | – | SKIP (N=5) |

**Every remaining fail is now a tolerance-vs-floor question or a localized defect — none is a model
divergence.** `felix_round`'s `v1b` 7.34% and `cohort count` 7.3% are ONE number (sim runs 7.9% more
iterations/bin) held to 5% tolerances against a floor estimated 5–13%; `fedbuff_round`'s `v2` 2.37% and
`convergence` 5.79% are hairline against 2%/5%. The exception is `selection_detail`, which is a real
sim-side defect (below). Phase A decides the rest.

**Budget coverage.** The four 7200s pairs are healthy (min 91.2-100%). The five 1200s rows are not: `felix_it`
73.1% and `fedbuff_it_oracular` 76.0% trip the low-coverage flag, and `fwdllm_it_*` grade N=**5** units. Read
those five for INV/un-windowed rungs only; their cadence and convergence cells are not evidence.

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

| baseline | open fails | next step |
|---|---|---|
| `felix_round` (71/3/18) | `selection_detail` · `cohort_sequence` count 7.3% · `v1b` cum 7.34% | **`selection_detail` is a real sim defect (H11), not a tolerance item:** confirmed OVER-dispatch — one end sent 5×, another 2×, with no commit in between, so 35 in-flight against c=30. NO RUN FIXES THIS; root-cause AsyncOort's boundary path in code. The other two are ONE number (+7.9% iters/bin) vs 5% tols — Phase A |
| `fedbuff_round` (72/2/18) | `v2` 2.37% matched (tol 2%) · `convergence` 5.79% (tol 5%) | Both hairline against tolerances at/below the floor — `replicate_floor.py`'s docstring already names `v2`'s 2% as shipped against a ~2% floor. `v1c` flat (λ=−0.002), and `v1b`/`cohort count`/`drain_wall_budget` all closed. Blocked on Phase A, not on code |
| `fwdllm` (68/0/24) | none | Clean at 7200s. `drain_wall_budget` closed once the charge came from its own real (0.278→0.101) |
| `fwdllm_it_unaware` / `fwdllm_it_oracular` (66/1/25) | `drain_wall_budget` (charge 2.31x / 2.24x its own real) | Same as `fwdllm`. Cadence UNGRADED at 1200s — N=5 |
| `fluxtune` (77/0/16) | none | Clean at 7200s. `v2` closed on the matched budget (6.52% pooled → 1.79% matched, §G) |
| `felix_it` (70/5/16) | `preferred_duration` · `drain_wall_budget` · `terminal`/`commits` · `convergence` — all on a 1200s pair at 73.1% coverage | Do NOT read these as findings. Needs a 7200s pair before any of it counts; `preferred_duration` is the only one that also failed at 3600s |
| `fedbuff_it_unaware` / `fedbuff_it_oracular` (72/3, 73/2) | `v2` · `convergence` (+ `per_round_advance` on unaware) — 1200s pairs, N=38 | `v1c` is flat on both, but at t=1.68/−0.06 on N=38 the slope is simply unresolved. **H6 cannot be answered at 1200s** — it needs the 7200s pair |

### Next session

> **Update this block in place on every run — overwrite Part 1/Part 2, never stack a new dated block below.**

**Part 1 — what changed, and what it settled.** Phase 0' re-ran ONLY the four sim legs against the existing
7200s reals, isolating the per-baseline charge profiles. It cut the charge 0.61–1.90 s/round and closed **9 of
14** open fails: `fwdllm` 67/1 → **68/0**, `felix_round` 65/9 → **71/3**, `fedbuff_round` 71/4 → 72/2,
`fluxtune` 77/0 unchanged. `felix_round`'s `v1c` went `diverging` → `flat` and took `v1`/`v2`/`thru`/
`commits`/`overhead_residual`/`terminal` with it — **H7 is dead** (§E), the residual was a per-cycle
overcharge all along, not a trajectory effect with no mechanism.

**The `var_bad` flip was never actually tested** — `44fd5e39` reverted it to `charge: false` *after*
`fca293d9` turned it on and before these runs; the ledger confirms `charge_source=none, charged_s=0.0`. Phase
0' validated ONE change, not two. Two of its four checks are also retired as un-answerable: the **pass-through
coefficient cannot be measured this way** (`sim_s_per_round` = vclock ÷ INTEGER rounds, so three of four
baselines moved less than one round of quantization; only `felix_round` resolved, at 148% — above 100%,
because the charge also shifted cadence, §D-21), and `sim_rate` **fell** on the two baselines that gained the
most rounds, since it measures wall efficiency and more work per vclock second costs more wall.

**Then the mechanism walk finished on disk** (§D-37): `felix_round`'s remaining +7.92% iters/bin (t=5.19) is
**not** `var@it0` (+2.16%, t=0.41 — not significant) but the per-iteration decay ratio (+1.55%, t=2.79),
which sits in an exponent — `n = ln(thr/var₀)/ln(decay)` reproduces the gap to 1.0800 vs 1.0792 observed, and
holding decay fixed collapses it to **1.0089**. §D-34's identity holds on REAL to −0.09% and breaks on sim at
+5.54%. Every mechanical input is identical (`grad_pool_size` 1,2,3…; contributors always 10; `cached_v` to
2dp; staleness histograms 82.3%/83.6%), so the residual lives in gradient VALUES set by which trainers were
drawn — already gated stochastic (§D-2). `fedbuff_round` has the OPPOSITE sign and a different root (iters
−4.19%, driven by `var@it0` −5.20%), so the two do not share a cause (§D-16).

**Part 2 — the live hypotheses, each with the observation that would falsify it.** State the prediction
BEFORE the run; a hypothesis that can only be confirmed is not one (§D-9).

**H10 — the four remaining cadence/convergence fails sit INSIDE the replicate floor, so no code change can
close them and the tolerances are what is wrong.** All four hold a 2–5% tolerance against a floor estimated
5–13% from mismatched-duration legs: `felix_round` `v1b` 7.34% / `cohort count` 7.3%, `fedbuff_round` `v2`
2.37% / `convergence` 5.79%. The within-run paired SEM says the gaps are not bin-scatter (t=5.19 / −3.00), but
that error bar is blind to seed-level trajectory variance — exactly what a replicate measures. Predicts
`replicate_floor.py` prints `iters_per_bin` ≥ ~7.9% and `mean_var` ≥ ~2%. **FALSIFIED IF** the floor comes back
under ~4% on `iters_per_bin`, which promotes §D-37's decay-rate finding from "inside the noise" to a live
mechanism and sends the walk to why sim's cohorts decay variance more slowly.

**H11 — sim OVER-DISPATCHES at the round boundary: the `all_selected` in-flight guard does not hold across
successive top-up `select()` calls that land at the same vclock instant.** Confirmed over-dispatch, not
churn: at the round-1→2 re-draw `felix_round`'s sim dispatches end `…0395` **5 times** and `…0409` **2
times**, and NO `agg_round` commit from those ends occurs between the repeats — so 35 in-flight assignments
against c=30, all at a frozen vclock of 4448.539 with `in_flight` reporting 30 the whole way. Real issues 30
unique; `fedbuff_round`'s `FedBuffSelector` matches real exactly (11/11). Mechanism NOT yet established —
the frozen vclock fits §D-30 (sim compresses the inter-call gap real spreads over wall time, so two calls can
read `selected_ends` before either writes back), but a plain logic error in AsyncOort's boundary path is not
excluded. **Root-cause in code before touching it; a selector change owes its own A/B (§F).**

> **This is NOT testable by the `felix_it` run.** `felix_it`'s selection events are ALL `round: 0` (7471 of
> them — reselection is event-driven, so `round` never advances), which is exactly the case
> `sim_repicks_in_round` guards to N/A. Confirming H11 on a second AsyncOort baseline needs the finer
> predicate — a re-pick while the end is still in flight — not another run. `felix_it` tonight is Phase B
> work (its own 7200s row, `convergence`, `drain_wall_budget`), nothing more.

**H8 — `fedbuff_it_unaware`'s H6 signature is unresolvable below 7200s, not absent.** Its 1200s pair reads
λ=+0.42/100 units — the largest magnitude of all nine — at t=1.68 against a 3.355 critical value, i.e. a
slope the run is too short to resolve. Predicts a 7200s pair either resolves the same sign into `diverging`
(H6 confirmed, and it joins `felix_round` as a shared root worth one investigation) or collapses toward zero.
**FALSIFIED IF** the 7200s λ flips sign, which would make the 1200s reading noise rather than an underpowered
estimate of a real effect.

### Run plan — the replicate floor, then the five missing 7200s pairs

> **Update in place. Delete a phase the moment its exit criteria are met and its findings are in §A/§G.**

**NEXT BATCH — one node, ~7.4h unattended.** Phase A's two reals, the floor, then `felix_it`'s pair, then the
scoreboard. The floor runs BEFORE `felix_it` so a late failure cannot cost the one measurement that matters.
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_round,fedbuff_round --yes && \
python replicate_floor.py --mode real && \
bash run_sequential.sh --mode both --max-runtime-s 7200 --only felix_it --yes && \
python run_parity.py --yes --baselines felix_round fedbuff_round felix_it
```

**Phase A — the replicate floor. ONE new real per baseline, 7200s** (not two: `felix_round`
`run_20260801_084040` and `fedbuff_round` `run_20260801_055314` already supply the first leg — both commit
`969c1d4e`, seed 1234, and `969c1d4e..HEAD` touches only `*_sim.yaml`, `profile_sim_charges.py` and
`run_sequential.sh`, so a run at HEAD is config-identical). `replicate_floor.py` groups on
`hyperparameters.max_runtime_s`, which reads **7200** on both — so keep `--max-runtime-s 7200`.
⚠️ **Do NOT bump `max_experiment_runtime_s` on these two yamls.** The existing reals ran under a 7200 watchdog;
raising it makes the new leg non-comparable and destroys the pairing. That blocker belongs to Phase B only.

Second payoff: the new real did not seed the charge profiles, so grading sim against it makes
`charge_coverage` and `drain_wall_budget` informative for the first time — they currently read `null` / `0.0%`
**by construction** (the charge was set EQUAL to the mean of the same real they are graded against). Until
then, do not read either as evidence.

**Phase B — 7200s pairs for the five baselines that have only 1200s.** `felix_it` (launched tonight),
`fedbuff_it_unaware`, `fedbuff_it_oracular`, `fwdllm_it_unaware`, `fwdllm_it_oracular`. Their §A rows are
1200s smokes at N=5-57 and 73-83% coverage; nothing cadence- or convergence-shaped in them is evidence, and
H8 is unanswerable without this. **`felix_it` goes first** because it is the other AsyncOort baseline and so
is the discriminator for H11. Real 2h + sim wall (extrapolated from the 3600s legs): `felix_it` 1.4h,
`fedbuff_it_*` 1.0h each, `fwdllm_it_*` 0.3h each. **BLOCKER:** every real yaml sets
`max_experiment_runtime_s: 7200` against a 7200s target — bump to ≥10800 **on the Phase-B baselines only**
(done for `felix_it`; the other four still need it).

**What tonight's runs are FOR — they fix nothing, and the two scoreboard rows will not improve.** Phase A is
a CALIBRATION run: it changes no code, so `felix_round` 71/3 and `fedbuff_round` 72/2 should read the same in
the morning. Its output is not a better score, it is a decision — whether those four fails are gradeable at
all. Expecting parity to improve from it is a category error. The one confirmed defect (H11 over-dispatch) is
NOT addressed by any run in flight and is root-cause work on the selector, needing no run at all.

**Read the floor like this — the decision rule, fixed BEFORE the run (§D-9).** `replicate_floor.py` runs the
SAME mode twice and reports how far apart two runs land with nothing changed but wall-clock luck. That number
is the smallest gap any real↔sim tolerance can honestly resolve: below it, a rung is reporting the pipeline's
own irreproducibility and no code change can ever make it pass (§D-24). Each printed metric maps to exactly
one tolerance:

| floor metric | gates | tol | today's gap | floor ≥ gap ⇒ | floor ≪ gap ⇒ |
|---|---|---|---|---|---|
| `iters_per_bin` | `v1b`, `cohort count` | 5% | `felix_round` **7.9%** | ungradeable: raise tol to the floor, close both, no code fix exists | the +7.9% is real ⇒ §D-37's decay-rate finding goes live; walk to why sim's cohorts decay variance slower |
| `mean_var` | `v2_var_trajectory` | 2% | `fedbuff_round` **2.37%** | tol was never gradeable (the script's own docstring already suspects this) | a true per-cycle var divergence to localize |
| `cycles` | `cohort_sequence.count` | 5% | `felix_round` 7.3% | same call as `v1b` — it is rolled-up V1, never chase it separately | — |
| `committed_bins` | `throughput` | 8% | all pass | confirms the 8% tol is sound | — |
| — | `v1c.lambda_floor_per_100` | 0.05 | — | recalibrate from the new pair either way (currently 2 pairs, §D-35) | — |

**What this can and cannot settle.** CAN: H10 (all four floor-blocked fails at once), H11 via `felix_it`, H8,
and — because the new real did not seed the profiles — whether the per-baseline charges are priced right.
CANNOT: `felix_round`'s `selection_detail`, which is a code defect and needs a fix, not a tolerance.

### Other open items

- **FUTURE TASK — enforce §F-18 mechanically: a knob-CONTRACT per baseline, checked at implementation time
  and again pre-run. Never post-hoc again.** Two knobs went missing from yamls and were only caught by
  reading run telemetry days later: `sim_charge_profile_path` (absent from 6 of 9 sim yamls, so those runs
  folded sim's own contended span onto the vclock) and `eval_every_n_commits` (net-new). Both are
  *correctness-path* values, so the affected runs were **not fairly comparable** — that is the real cost, not
  the failing rung. §F-18 already states the rule; nothing enforces it.
  - **The mechanism mostly EXISTS — this is a coverage gap, not an architecture gap.** `run_sequential.sh`'s
    preflight gate already builds `checks[]` with `level: error|warn`, rendered by
    `expt_runner.render_and_gate`, which exits 2 and blocks the launch. It already ships the two exact
    patterns needed: `enable_training_delays matched across real/sim pair` (per-pair knob match) and
    `agg_goal matches across baselines` (cross-baseline fairness, warn). And `condition_fp` is already an
    8-char sha of the shared condition, explicitly designed as a two-node cross-check. **The gap: `_cond`
    hashes only N/K/C/partition/trace/delays/caps — the CLI-patched knobs. Anything that lives only in the
    yaml is invisible to it**, which is exactly the class both misses fell into.
  - **Where each layer should catch what** (all three already have a home; none needs a new file):
    | layer | home that already exists | catches |
    |---|---|---|
    | implementation | `tests/mode/test_baseline_readiness.py` — already parametrizes over `examples/_metadata/baselines.yaml` | a knob added in code but never added to the yamls; a baseline missing a knob its family declares |
    | pre-run | `run_sequential.sh` `checks[]` + `condition_fp` | a knob present but MISMATCHED across a real/sim pair, or across baselines in one comparison. **First tenant landed** (§G): the sim charge profile's provenance is gated here, blocking a launch whose charges came from another baseline's real |
    | post-run | parity `--validate` / `snapshot.yaml` | last-resort drift; should be redundant once the first two hold |
  - **The hard part, and the reason this needs design before code: "missing" vs "legitimately N/A".** A flat
    "every yaml has every key" diff is wrong and would be the verbose outcome to avoid —
    `sim_charge_profile_path` is sim-only (a real yaml must NOT have it), `reselect_cadence: round` is
    round-baselines-only, `trackTrainerAvail` is oracular-only. So applicability has to be **declared**, not
    inferred from a diff. Sketch worth evaluating: a `knob_contract` block in `_metadata/baselines.yaml`
    (which already carries per-baseline selector/optimizer/hyperparameter defaults) marking each
    correctness-path knob `required` / `sim_only` / `real_only` / `n/a` per baseline-family, with the reason
    inline. Then all three layers read ONE declaration instead of three hand-maintained lists.
  - Open questions to settle first: (a) does the contract live in `_metadata/baselines.yaml` or beside the
    fwdllm yamls? (b) is a knob missing an `error` or a `warn` — probably error for correctness-path,
    warn for tuning; (c) should `condition_fp` absorb the contract'd knobs so a mismatched pair is caught by
    the fingerprint alone; (d) who owns adding a new knob to the contract — ideally the same change that
    introduces it, enforced by the implementation-time test failing until it is declared.
  - Proposed §F-18 amendment once the mechanism exists, for operator approval (do NOT apply unilaterally —
    §F header): append *"A correctness-path knob must be DECLARED in the baseline knob-contract; an
    undeclared or unmatched knob blocks the launch, and is never a post-hoc telemetry finding."*

- KS-only rungs still unguarded against a level shift (same class as the `selection_bias` repair, §G;
  all currently clean on live data): `dk1_agg_goal_trajectory`, `dk2_dynamic_c`, `dk3_eligible_ends_metric`,
  `eligible_speed`, `v3_cached_v_pool`. `selector_score` is DIAG.
- Thin ABSOLUTE budgets: at 7200s `fwdllm` reaches N=39, but `fwdllm_it_*` are still on 1200s pairs at
  N=**5** and `v1c` SKIPs there. Coverage percentage cannot catch this; an absolute-N floor is proposed,
  threshold not chosen. `v1c` already declines to grade below 4 usable bins, which is the same idea narrowly.
- Real's over-`c` slot read is MOSTLY closed (§G) — drain lag, not a half-fixed §D-27 (§E). `fluxtune` still
  reads a max of **41** against c=30 on **4.3%** of dispatches (was 45 / 8.1%): a decaying tail 31..41, all in
  round 1, i.e. the window between a message being DEQUEUED and its inflight entry being cleared. Telemetry
  only — `contributor_intervals` reads peak 30 with `over_cap_frac` 0.0, and real's dispatch path does not
  read `_slot_holders()`. Low priority; the authoritative measure disagrees with the tripwire, not with `c`.
- **`convergence` — root CONFIRMED, fix validated** (§G). It closed on both 7200s baselines that failed it
  (`fluxtune` 5.16%→3.97%, `felix_round` 5.12%→4.86%), but the margins are hairline against a 5% tol and
  `avg_loss_diff` grew 3-9x (0.0025→0.026, 0.0091→0.0802 against a 0.15 tol). The three baselines still
  failing it are 1200s pairs and not evidence. Re-read after Phase B.
- **Real publishes no `_agg_slot_holders_ref`**, so `_cap_dispatch_to_concurrency` takes its fallback and
  reads the IDENTITY set on real. Harmless today (the round cadences that reach it dispatch after the
  boundary clear, where the two sets coincide) but it is the same conflation `slot_holders()` just fixed for
  the tripwire. Publishing one would let real dispatch into slots it currently withholds — a behaviour
  change owed its own A/B, deliberately NOT bundled with the measurement fix (§G).
- **Per-baseline charge profiles: VALIDATED on the four 7200s baselines** (§G) — but `charge_coverage` and
  `drain_wall_budget` are CIRCULAR until a real that did not seed them exists (Phase A). `fwdllm_it_*` still
  fail `drain_wall_budget` because their sim legs were not re-run; untested, not failing.
  **FUTURE TASK:** the five 1200s-sourced profiles (`felix_it`, `fedbuff_it_*`, `fwdllm_it_*`) should be
  regenerated from their own 7200s reals once Phase B produces them — the charge is not materially
  duration-sensitive (1200s↔7200s ratios 0.80-1.21), but `fwdllm_it_*` are thin at n=52 samples.
- `felix_round`'s lap-boundary fails (`preferred_duration`, `terminal_state`, hairline `convergence`) all pass
  at 3600s except `convergence` — underpowered measurement artifacts. Expect them at any duration where the
  graded window straddles a lap boundary. **`selection_detail` is NO LONGER in this class** (§E): it is a real
  duplicate-pick defect, and `fedbuff_round` passing the same rung on the same window proves the window is fine.
- `step_timing_breakdown` / `agg_step_timing_breakdown` are now REPORT-ONLY whenever the sim clock discards
  the span (§G, §D-31), which is the designed steady state on every baseline. Their numbers stay published:
  `_make_model_functional` runs real 0.0339-0.0362 vs sim 0.0292-0.0319 (KS 0.233-0.458 across four baselines
  — one physical effect straddling a 0.25 tolerance), and the agg-side group
  (`_compute_var`/`_prepare_round_state`/`_replay_buffered_cohort_contribs`) runs sim 1.7-5.8x slower. Both
  are §D-1 co-location contention and neither reaches the vclock. Watch `charge_coverage` instead.
- Do not re-tune `redispatch_turnaround` — its totals check already ruled it out (§D-14); no live residual
  is a charge-magnitude problem.
- `cohort_sequence` `count` is rolled-up V1 (it counts cohorts per matched bin), so it reports V1's number,
  not an independent one. Never chase it separately — it fails on `fedbuff_round`/`fedbuff_it_unaware`
  precisely because their `v1` does.
- Flag promotion: `sim_model_agg_compute_time` defaults ON. `sim_sct_ordered_drain` +
  `sim_model_dispatch_queue` are fluxtune-yaml-only but model general async-transport artifacts — smoke
  fwdllm/fwdllm_plus with both ON, confirm inert-or-better, then promote to code-level default-on.
- Checker invariants I1-I6 were drafted in a prior session and never committed anywhere (checked git log +
  repo-wide grep — genuinely unrecoverable). No longer blocked, but needs operator input on intended
  semantics before drafting fresh ones.
- Confirm the `async_oort`→`AsyncSelectorBase` re-basing (§G) at the INTEGRATION level — unit tests confirm
  the mechanism, not a live comparison:
  ```bash
  cd lib/python/examples/fwdllm/expt_scripts
  bash run_sequential.sh --mode both --only felix_round,felix_it
  ```
- felix (async_cifar10) may share fluxtune's round-1 cold-start gap (`_sim_recv_min`, no fallback for unseen
  ends) — unverified, out of scope here (`async_cifar10/PARITY.md` owns felix).
- felix 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Momentum (S1-S3) / server-optimizer retry — roadmap, not parity; resume after Phase-1 closes.
- Accuracy drop after 81% — known, deferred by operator (`fluxtune_contributions.md` §8).
- Sim's in-flight bookkeeping is still split across `_sim_pending_commit`/`_sim_inflight_expected`/
  `_sim_buffer`/`_sim_committed`/`selected_ends`/`all_selected` — should be one authoritative per-end state
  machine. The slot⇄guard split (§G) funnelled every CAPACITY read through `_slot_holders()`, which is the
  first half; the IDENTITY half is still ad-hoc. Scope behind `test_fwdllm_sim_grad_loop.py`'s
  commit/residence tests, never bundled with a correctness fix.
- Base `asyncfl/top_aggregator._sim_hold_busy_slots` deliberately left alone — its `held` set has no
  `_sim_committed` term, so it never had the conflation. Re-check if async_cifar10 shows the same under-fill.
- P3/infra: no automatic GPU skip-and-remap on a broken ordinal (manual `execution.gpu_ids` exclude works).

---

## §C  How we debug here — the ladder, the fwdllm decomposition tree, run-length budget

**Ladder walk** (full method: PARITY.md §1 + rung catalog §F). An FL run is a pipeline —
`clock → availability → selection → dispatch/train → return/order → aggregation → variance-cadence →
utility → emergent`. Parity must hold at every stage; break at stage N and every stage above diverges as a
*consequence, not a bug*. The checker labels the **lowest broken rung with sound (matched) inputs** the
ROOT and demotes higher fails to DOWNSTREAM. Tag every rung a role — CONTROL (input identical), MECHANISM
(one transform modeled — the prize), EMERGENT (aggregate; never fix directly, walk *down*) — and a tier —
INV/EXACT (hard fail), DIST (fail unless `--lenient`), DIAG (informational).

**fwdllm decomposition tree** (which rung fails → where to walk):
- `K2`✗ (throughput) but `K3a`✓ (per-pass advance) → clock is fine, commit COUNT diverged → walk to
  `V1`/`V5` (variance cadence), not the clock.
- `V1`✗ (iterations-per-data_id) but `V2`✓ given matched inputs → the variance *inputs* differ → walk to
  `U5`/`S2` (ordering/selection), not the variance gate.
- `V2`✗ with `V1` inputs matched → a true grad-pool accumulation-order bug.
- `drain_wall_budget`✗ but input byte-sizes identical → co-location contention, not over-compute → §D-1
  (don't tune sim compute).
- `cohort_sequence.composition`✗ but every marginal (S2/utility/count/v1/v2/speed) matches → boundary-race
  cascade = stochastic identity → §D-2 (gate index-identity, keep marginals).
- **Never touch `var_threshold` / `max_iterations_per_data_id`** (§F-3): baseline-defining config, not
  parity levers. A cadence gap is ALWAYS an upstream set/order/clock divergence.

**Run-length budget (fwdllm) — SHORT BY DEFAULT.** Verification runs are **900s (15 min) or 1800s (30 min)**
via `--max-runtime-s`; 3600s+ is reserved for scoreboard re-grades and the duration-gated rungs below. Every
run is operator-launched, so pick the shortest length that exhibits the issue.

**The split is not a judgement call — it follows the residual's SHAPE.** A *per-cycle* divergence is fully
present in the first cycle and a short run grades it at full strength; an *accumulating* one is under-reported
by construction, and a short run will falsely PASS it. Measured on the 1800s pairs
(`run_20260728_151831`/`_155008`, `_151722`/`_154939`): `overlap_factor` 22.8%/19.0%, `throughput`
18.8%/17.8%, `total_commits` 23.2%/19.8% — the whole clock family fires hard at 30 min. The cadence rungs do
not: the same legs read `v1` 3.1% and `v2` 1.4% where `fedbuff_round`'s 3600s pair reads 8.4% and 7.3%
(matched window). They are quiet at 30 min not because they are fixed but because the divergence hasn't
accumulated yet. The 900s batch made the same mistake in the other direction — it read `fedbuff_round`'s
whole cadence family green, and 3600s put seven of those rungs back (§A).

| validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | 5-10 min | a few hundred commits populate any per-commit field |
| **regression smoke after a shared-path change** | **900s** | INV tripwires + occupancy rungs are un-windowed, so they grade at any duration (below) |
| clock/pipelining family (`overlap_factor` K4, `throughput`, `per_round_advance`, `overhead_residual`) | **900-1800s** | per-cycle mechanisms — fire at full magnitude immediately (evidence above) |
| one MECHANISM rung (`drain_wall_budget`, `selection_detail`, `eligibility`) | **1800s** | the mechanism fires; per-commit dists stabilize |
| variance-cadence LEVELS (`V1`/`V2`/`V2b`/`V5`) | 3600s+ | the divergence ACCUMULATES; a short run reads a false PASS |
| variance-cadence RATE (`V1c`) | 1800s+ / N≥40 bins | duration-INVARIANT by construction (§D-35), but its t-test needs bins: at N=38 `fedbuff_it_unaware`'s λ=+0.42 is unresolved (t=1.68), and it SKIPs below 4 usable bins |
| stochastic identity / participation (`cohort_sequence`, `S2`) | 3600s+ | index overlap must reach its independent-draw floor to read as identity-not-bias (§D-2) |
| convergence sign-off (`terminal_state`, `conv`, `conv_loss`) | full 2h+ | terminal-state + curve parity only |

**WINDOWED vs UN-WINDOWED is what makes a short run readable — check before choosing a duration.** A rung
carrying `matched_logical_budget_n` grades only the work both sides did, so at 900s its N is ~1/4 of the
3600s N and a thin-N verdict is weak (`fwdllm` is unreadable at ANY short duration — only 20 committed bins
in 3600s). A rung with no `matched_logical_budget_n` pools the whole run and grades
at full strength immediately: every INV tripwire (`concurrency_cap`, `retask_before_close`,
`r1_inflight_overlap`, `sim_rate`), `slot_utilization`, `throughput`. Read the field in the pair's JSON
rather than assuming.

**Is a given duration long enough? Measure, don't guess.** The reproducibility floor grows as runs shorten,
so launch the SAME short config twice on one side and run
`python expt_scripts/replicate_floor.py --mode real`; any rung whose tolerance sits at or below the floor it
prints cannot be graded at that duration. Do this once per baseline when adopting a new run length.

Smoke (5-10 min) before any long run. One mechanism per run when a fix could perturb another baseline.

**pytest:** `setup.cfg` sets `addopts = -n auto`, which needs `pytest-xdist` — now installed in BOTH
`dg_flame` and `test_fwdllm` (it was missing from both, and every invocation died on
`unrecognized arguments: -n`). Full suite 6m43s → **1m58s**.
```bash
conda run -n dg_flame python -m pytest lib/python/tests lib/python/examples/fwdllm/expt_scripts -q
```
If a future env lacks xdist, `-o addopts=""` runs it serially rather than failing — but install xdist instead.

---

## §D  Durable lessons — fwdllm diagnostic patterns

> Positive, transferable invariants — things learned that must be followed. ≤30 words each; update in
> place, never append near-duplicates. A falsified hypothesis belongs in §E, not here. Shared (non-fwdllm)
> patterns live in [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) "Durable lessons."

**D-1.** A shared-compute wall rung failing with byte-identical inputs is co-location contention, not sim
over-compute. Fix by charging the vclock, never by tuning sim's compute.

**D-2.** A boundary-race cascade on a stochastic-async selector is core-IDENTITY, not skew. Gate
index-identity to diagnostic; keep count/marginals enforced.

**D-3.** Porting a baseline's selector class does not port its real↔sim timing parity — aggregator/trainer
classes are separate. Diff the destination against the shared base first.

**D-4.** Grade parity on the logical work budget N, never a matched virtual-time window — normalizing along
the axis under test is circular.

**D-5.** A green parity rung is not a correctness claim — common-mode bugs pass differential tests. Always
pair with an absolute, mode-independent sanity check.

**D-6.** A selector's parity record belongs to the SELECTOR+AGGREGATOR pair, not the selector alone. Verify
which side owns each guard before citing it as a reference.

**D-7.** A statistic computed over rate-scaled samples measures the rate, not the samples. Check both
sides' input to a shared formula before suspecting the formula itself.

**D-8.** A cache/reuse fast path can silently skip a guard the slow path enforces. Diff its return against
what the bypassed call would have filtered.

**D-9.** A parity DIVERGENCE names two disagreeing sides, never which is wrong. Find each side's own
absolute, same-side self-consistency signal before choosing which to change.

**D-10.** `reselect_cadence` changes which candidate POOL fills a freed dispatch slot, not whether a
just-committed trainer personally waits — the version_key re-pick guard is cadence-agnostic.

**D-11.** A per-round wall residual can be a per-cycle cost compounded many times over. Count actual
cycles-per-progress-unit before assuming a small measured gap can't explain a bigger one.

**D-12.** A telemetry span measured from a shared batch-start timestamp is cumulative across the whole
batch. Pool by first-difference between sorted events, never a flat mean.

**D-13.** A run truncated by a wall-clock deadline can emit one event for progress it never finished.
Require verified-completion evidence per progress-axis key, not event presence.

**D-14.** After fixing a proven overcharge, re-verify the residual's sign and mechanism, not just that the
old bug is gone — a flipped sign means re-decompose, not re-tune.

**D-15.** When sim runs faster than real with no charge category to explain it, suspect a
concurrency/scheduling policy divergence. Measure per-trainer idle time, not server wall.

**D-15a.** Anchor "what was sent when" on `processing_wall_ts` (arrival), not `dispatch_version_key` (read
at close) or a cycle's own close timestamp — both mis-date a mid-cycle send.

**D-16.** Factor a per-round residual into `(s/cycle) × (cycles/round)` before naming a mechanism — a
shared symptom across baselines can still have disjoint roots.

**D-17.** An event-count rung is only meaningful over matched WORK, not the whole run. Compare a divergent
event's progress key against the matched budget, not wall time.

**D-18.** A charge ledger prices the CHARGE, not the residual it removes. Never size a throughput
prediction off the charge delta — confirm the clock is actually charge-limited first.

**D-19.** Never normalize a per-bin statistic by distinct `data_id` — it wraps. Key on
`(round, cycle_data_id)`, count only completed visits, truncate to the matched budget.

**D-20.** Grade an invariant on its own quantity and axis, never a proxy sampled at the wrong instant. Check
for an existing per-entity span before instrumenting a new one.

**D-21.** A vclock charge is not a neutral accounting term — if sim selects work against the clock, a
charge change can shift sim's selections and cadence too.

**D-22.** A family of rungs reporting one quantity five ways cannot localize it. Split the residual against
an independent per-cycle measurement (barrier vs clock advance) before naming a mechanism.

**D-23.** A divergence that GROWS with progress is a trajectory divergence, not a per-cycle bug. Bin the
metric by progress and check the trend before hunting a mechanism.

**D-24.** A tolerance is only meaningful above the pipeline's own same-seed replicate spread. Measure the
floor from runs already on disk; below it, no code change can ever pass.

**D-25.** Verify a per-cycle mechanism on a short run, an accumulating one on a long run. Run length is set
by the residual's shape, never by habit.

**D-26.** A progress key must be monotone in TIME before anything sorts, maxes or windows on it. Order by
event `ts`; a composite key can wrap out of order.

**D-27.** One set serving two roles hides a bug until a rule changes one of them. When an invariant reads
"X but not Y", check whether the code has two sets or one — in BOTH modes. Real had the same conflation sim
did; only its dispatch ORDER hid it on 8 of 9 baselines.

**D-28.** A shared-path fix validated on the baselines it targeted must be re-graded on ALL of them at
scoreboard length. Sibling baselines can regress silently on a rung the validation set never exercised.

**D-29.** A short validation run can read a rung green that a long run fails. A green rung is evidence only
at a duration where that rung's residual has had room to accumulate (§C table) — never bank one below it.

**D-30.** Anything gated on "is the background worker free" is a wall-clock race, and sim loses it — sim
compresses the gap between events while the work costs the same wall. Gate on a progress INDEX.

**D-31.** Before grading a sim quantity, check the clock actually consumed it. Where sim folds a profiled
constant, its own span is a discarded contention artifact and comparing it is a guaranteed false fail.

**D-32.** A "worst offender" field that ranks exempted entries will be read as the cause. Rank only what
gates; report the raw winner under its own name.

**D-33.** Real clears its bookkeeping when it PROCESSES a message, not when the message lands. Any capacity
read off that state charges the aggregator's own drain lag to the trainers.

**D-34.** On an uncapped round baseline, iterations-per-bin = `var@it0` ÷ `var_threshold`. The identity is a
REAL-side property (residual −0.09%); where a side breaks it, that residual is the divergence, not the level.

**D-35.** A quantity inside a feedback loop has a residual that grows with run length, so no fixed tolerance
on its LEVEL is right at two durations. Gate the per-unit RATE against zero; the level is a readout.

**D-36.** A profiled constant shared across baselines is a hand-typed constant wearing a script's clothes.
Profile per baseline, from the paired real leg, and gate provenance at launch.

**D-37.** Decompose a decaying quantity into its START and its per-step RATE. A rate sits in an exponent, so a
1.5% rate gap buys an 8% count gap — test both, and read the rate's t-stat, not the level's.

**D-38.** Pair per-bin samples at equal bin index before differencing: it cancels the shared training curve
and gives an honest within-run error bar. It cannot see seed-level variance — only a replicate can.

**D-39.** A statistic that averages a bimodal burst reports how many small events fired, not what they did.
Report the raw gating quantity under its own name before trusting the mean (§D-32).

**D-40.** Before widening or re-windowing a failing rung, check whether a SIBLING baseline passes it. If one
passes and one fails on the same code path, the rung is working and the fix would mask a defect.

---

## §E  Dead ends — do NOT retry

> Falsified hypotheses, one line each, append-only. A dead end never un-dies; re-listing one wastes a
> session. Landed-but-inert cleanups belong in §G, not here.

- **`redispatch_turnaround` as a large per-baseline cost sim omits** — FALSIFIED by direct measurement, and
  the profiler's own number was the artifact. The first-difference marginal (§D-12) assumes a batch is ONE
  serial dispatch BURST. That holds on the round baselines (cohort pinned, dispatched together after the
  boundary clear) but NOT under event-driven dispatch, where a batch's timestamps span most of a cycle and
  consecutive readings differ by the gap between trainers RETURNING. Measured: `fluxtune`'s marginal 0.318s
  against a `cycle ÷ dispatches` of 0.400s and a batch ts-spread of 3.03s in a 3.99s cycle. The DIRECT cost
  (`_distribute_weights_async` step_timing) is real **0.016-0.032s** against sim **0.018-0.029s** — sim
  performs the same call and pays the same wall, so there is almost nothing to charge (real−sim −0.002 to
  +0.008s). Charging the marginal would have added **+62% (fluxtune) / +39% (felix_it) / +20-23%
  (fedbuff_it_*)** to those vclocks. `var_bad` is OFF and `weights` is pinned to the round-baseline value;
  the profiler now cross-checks against the direct measurement and refuses to overwrite (§G). Don't re-derive
  a redispatch charge from timestamp differences on an event-driven baseline.
- **H7: `felix_round`'s divergence is a trajectory effect carried by `var@it0` with no per-cycle mechanism** —
  DEAD. A per-cycle charge correction (its own real's `drain_tail`/`weights`, −1.85 s/round) closed the whole
  family: `v1c` 0.190→0.031 `diverging`→`flat`, `v1` 16.3%→7.3%, `thru` 12.0%→4.9%. The residual did NOT
  "survive any change". On the post-fix pair `var@it0` is statistically indistinguishable (+2.16%, t=0.41)
  and what remains is the per-iteration decay rate (§D-37). Don't re-open var@it0 as felix_round's root.
- **H9: the removed overcharge passes through to throughput at a fixed ~29% (or ~100%)** — FALSIFIED, and the
  question is malformed. `sim_s_per_round` = vclock ÷ INTEGER rounds, so on 3 of 4 baselines the whole charge
  delta is under one round of quantization (fluxtune Δ0.04s vs a 0.73s quantum; fwdllm Δ4.74 vs 4.28;
  fedbuff_round Δ0.19 vs 0.21) and the coefficient is unmeasurable. Only `felix_round` resolved (17 quanta) —
  at **148%**, above 100%, because the charge also moved cadence (§D-21). There is no single coefficient.
- **`selection_detail`'s felix_round fail as a windowing/thin-N artifact** — FALSIFIED, and re-windowing it
  would have MASKED a defect. It is already windowed (`_time_to_progress`). Every selection event in a 7200s
  run lives in 3 bins (0, 149, 150) because `round` only reaches 2, so "sim ran further so it topped up more"
  is wrong. Bins 0 and 149 are identical on all four legs; at bin 150 sim issues 35 picks for 30 slots with 5
  duplicates while real issues 30 unique and `fedbuff_round` matches real exactly (§D-40).
- **per-cycle committed-set overlap as the discriminator for trajectory divergence** — FALSIFIED. It looked
  like the source term feeding the compounding, but it does not separate the baselines: `fluxtune` has the
  LEAST set agreement of the three async baselines (0.262 against a 0.243 independent-draw floor) and the
  BEST trajectory agreement (var +2.1%, no trend), while `felix_round` sits at 0.357/0.202 and diverges.
  Don't gate on cohort composition; the discriminator is the drift RATE (§D-35).
- **`fedbuff_round`'s cadence residual as a mechanism to chase** — it is a LEVEL offset with a flat rate
  (`v1c` λ=−0.056, t=−1.70) sitting inside the real↔real replicate floor. What fails is its `v1b`/`cohort
  count` 5% tolerances against a floor of 5-13%. Calibrate the tolerance, don't hunt the mechanism (§D-24).
- **real's wall-vs-vclock clock anchor as the residual's cause** — REFUTED by direct measurement: anchoring
  real on cumulative `intrinsic_span_s` (the emitted per-cycle algorithmic clock) instead of raw wall moves
  every residual by ≤1pp — fluxtune 0.119→0.123, fedbuff 0.164→0.155, felix 0.047→0.035. The residuals
  survive the change of coordinate, so they are in the mechanism, not the measurement. Don't re-open
  `_real_intrinsic_clock`'s async bail as a parity fix (a max-fold envelope would still be a cleaner
  construction than the cumulative sum — but it will not close a gap).
- **real's over-`c` slot read as a half-fixed §D-27 conflation** — WRONG diagnosis, corrected by direct
  measurement. None of the three early-return paths that skip the inflight clear ever fires (duplicate /
  stale-reject / invalid all count 0 across four 3600s reals). The cause is drain lag: the clear runs when
  the drain loop PROCESSES a message, not when it arrives, and `felix_it`'s true peak concurrent training —
  swept from the trainers' own `train_with_data_id` spans — is exactly 30 while the tripwire read 55 on 49.1%
  of dispatches. Don't re-open the `_PendingCommitUnion` halves; they were already correct (§D-33).
- **`step_timing_breakdown`'s gating func "moving" to `_emulate_training_delay`** — it never gated. That func
  is a modeled sleep sim skips by design, exempted for exactly that reason, and KS 1.0 by construction; the
  rung's `worst_func` simply ranked every func including exempted ones. The gating set never changed from
  `_make_model_functional` (+ the fwdllm agg-side D-1 group). Fixed at the source (§G, §D-32).
- **eval-thread GPU contention reaching the commit order through `sct`** — FALSIFIED: sim orders commits on
  a deterministic modeled-delay grid (24 distinct `trainer_speed_s` values across 300 cycles) while real's
  measured wall has 532, so contention cannot move sim's ordering. The 5-9/10 per-cycle cohort overlap
  between modes is real's intra-speed-class tie noise (§D-2/§F-20), not a bug and not eval-driven.
- **`fedbuff_round`'s cadence as a progressive TRAJECTORY divergence (H3 as stated)** — FALSIFIED at 3600s.
  The residual survives (`v1` 8.4%, `v2` 7.3% matched, `terminal`/`commits` 11.4%), but `v2b_var_drift` no
  longer reads progressive: it flipped ρ=−0.61 → **ρ=+0.55**, verdict `level_offset`, ratio 0.88→0.947 —
  converging, not separating. The SIGN flipped with it, sim going from slower to FASTER than real (N=106 in
  2944.8s vclock vs 3322.9s). Don't go hunting separating `grad_norm`/loss curves; §D-14 says re-decompose.
- **Root C (a shared charge-coupled cadence level on both round baselines, via `SimReorderBuffer.pop_min()`)**
  — DEAD as stated: it required one root across `fedbuff_round`+`felix_round`, but the repaired
  `overlap_factor` clears BOTH (4.97/5.04 and 4.91/4.90) and `v2b_var_drift` splits them — fedbuff is a
  progressive drift (ρ=−0.61), felix a level offset (ρ=−0.12). The reorder buffer may still matter for felix;
  it is not a shared root, and `felix_round`'s throughput family is now green on its own terms, not by
  coincidence.
- **`selection_bias` on the round baselines as TWO independent selector faults (H2 as stated)** — FALSIFIED:
  the opposite signs (fedbuff slow-biased, felix fast-biased against an identical 12.51s pool) looked like
  two bugs, but BOTH flipped green on the slot⇄guard split with zero selector change. It was one upstream
  cause — who the slot bookkeeping left available to select — reading out with opposite signs through two
  different selectors. Don't open a per-selector walk. General form: opposite signs against a shared input
  do not imply separate roots when both sides draw from the same availability bookkeeping.
- **sct-order-membership lever** (admit the lowest-sct 10 instead of first-arrived) — REJECTED: the
  aggregator would BLOCK on future arrivals / hold slots for possibly-offline trainers → FIFO-violating,
  DEADLOCKS under Phase-2 unavailability; and the divergence is a stochastic tie-break, not a chargeable
  mechanism (§D-2).
- **recv_fifo→drain_ready as the async fluxtune D-skew fix** — LANDED but INERT: dropped 181k "already has
  active task" log lines but the D-skew (49.5/56.8) and per-cohort wall (4.02 vs 3.70) were unchanged. Kept
  as cleanup (flag `real_drain_ready_ingest` ON), NOT a parity fix; the recv_fifo streamer was not the
  mechanism.
- **`real_distribute_settle_s = 0.0` as the fluxtune parity cause** — NOT the cause: real ran clean at 0.0
  (droppable dead weight) but the skew is a stochastic tie-break; don't expect dropping it to move fluxtune.
- **`_compute_var` stop-the-world GC pause** — REFUTED: `gc_pause_s` telemetry shows ~0ms GC both sides.
- **`_flat_grad_norm` as the drain-wall contention amplifier** — landed (bit-identical, correct
  optimization) but INSUFFICIENT: p90 `drain_tail_s` rel unchanged (0.94/0.92); not the dominant contention
  source (§D-1).
- **GPU/resource contention blamed at n=10** — REFUTED once; held below ~100 trainers. (At n=100 contention
  IS a root — §D-1; this dead end is scale-bounded.)
- **`matched_virtual_budget` (V = min(vclock, wall))** — DELETED, don't reintroduce: conflates the two
  clocks (the axis `sim_rate` tests), masks throughput + fails to grade (§D-4).
- **"surplus idle" as the post-§D-15 residual (felix/fluxtune)** — REFUTED: the `c/(busy+idle)` identity is
  MATCHED on all three post-fix legs (felix 2.93 vs 2.87 s/cycle, fluxtune 9.59 vs 9.73). The residual is
  `cycles/bin` (felix) and `s/cycle` charges (fluxtune) — §D-16. Don't re-open the idle term.
- **`reselect_cadence` pool-size theory as the round-cadence throughput driver** — SUPERSEDED: the per-cycle
  charge-compounding root (§D-11) explains the gap. Pool-size may still matter for `slot_starvation`-style
  idling, just not this gap.
- **`felix_round`/`fluxtune` sharing one "overshoot root"** — REFUTED: disjoint factors, §D-16.
- **the round-cadence cohort pin as a defect ("never rotates / 70 of 100 never train")** — NOT a defect,
  operator ruling: `reselect_cadence: round` pins for a whole round by design and `rounds: 50` would rotate
  it 50×; a 5400s run just covers ~1 round. Don't re-key it on `_model_version` (§D-17).
- **cohort COMPOSITION as `felix_round`'s `v2_var_trajectory` driver** — REFUTED: both modes commit from the
  same frozen 30 for 89% of the run with the gap fully present; widening real's set moves `var` the wrong
  way (§D-2). The trajectory diverges, the contributor set doesn't.
- **"67% of `fluxtune`'s throughput residual is charged contention" (§D-18 sizing)** — REFUTED by the
  validation leg: 0.320 s/cycle of charge removed bought 0.092 s/cycle, 13.2%→11.9% not →7%. The charge fix
  is correct and stays; its predicted magnitude was wrong. Don't re-price a charge without checking the
  clock is charge-limited.
- **Root A′ as a `felix_round`-specific cadence defect ("chase what changes at bin ~30")** — REFUTED: the
  entire +14.7% vanished on a charge change that never touched felix's code path. It was sim's
  charge-coupled cadence level (Root C), not a felix mechanism. Don't reopen the bin-30 onset.
- **`fedbuff_round`'s 553 same-end dispatches / the round-boundary backfill as a throughput term** — FIXED
  and confirmed inert on throughput: cap green, self-overlap 0, dispatches/cycle unchanged at 10.02. It was
  a real INV breach worth fixing; it moved no timing rung.
- **cohort composition as a real↔sim divergence on the round baselines** — the round-1 pinned 30 is
  BIT-IDENTICAL real↔sim on both (30/30), and participation entropy matches to 4 digits across all six legs.
  Both modes run the same trainers on the same shards; stop looking for a set difference (§D-2).
- **`aggregate_grad_pool()` summation order as the `v2_var_trajectory` driver** — REFUTED: it's an
  element-wise SUM over ≤`max_iterations_per_data_id` items (float-reorder effect ~1e-6 relative, not the
  observed 3%) feeding the outgoing `GRAD_POOL` payload, not the variance gate. The gate is
  `calculate_var()`'s split-half over `grad_for_var_check_list`.

---

## §F  Locked invariants (from async_cifar10, carried over)

> Always-true / always-do rules. Numbers are cited across this doc — keep them stable, don't renumber.
> Diagnostic *patterns* (see X → means Y) live in §D, not here.
>
> **This list is not frozen — it is APPEND-and-AMEND with operator approval.** An invariant is the current
> best statement of a rule, not scripture; when work proves one incomplete, wrong, or worth adding, say so
> and propose the exact wording. Rules:
> - **Never silently.** Ask the operator before adding or changing one, and state the EVIDENCE (code, live
>   telemetry, or a test that fails without it) — not a plausible-sounding argument.
> - **Amend in place, keep the number.** §F-23 gained its two-set clause without renumbering. Other sections
>   cite these numbers; a renumber silently rewrites every citation.
> - **Deleting needs more evidence than adding.** An invariant usually exists because something broke once.
>   Prefer narrowing its scope to removing it.
> - **A new invariant must be always-true, both modes.** One-baseline or one-rung findings are §B/§G; a
>   see-X-means-Y pattern is §D. If it needs a caveat, it is probably not an invariant.
> - When a fix contradicts an invariant, that is a STOP — the invariant may be wrong, but resolve it
>   explicitly with the operator before landing (the fix that split §F-23 was gated exactly this way).

1. **Sim does real forward-grad compute, charges modeled time.** Agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Never put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`; identity/caching axis is `model_version`.** `data_id` wraps every
   `total_data_bins` lap — never key a cache/identity on it, use monotone `model_version` (§F-21). `_round`
   (the lap counter) equals `model_version` only in regular FL; don't carry a round-keyed construct over
   without re-deriving the axis.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold`/`max_iterations_per_data_id`
   are baseline-defining knobs, not parity levers.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct
   reorder buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** Check whether real is the divergent side before
   tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as real-transport artifact (`and not
   self.simulated`) vs algorithmic property; scope-check shared code first — `top_aggregator.py`/
   `_sim_recv_min` can silently break async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only → `pytest tests/mode -k fwdllm`; shared parity
   engine → add async_cifar10 tests too; shared stack → full `pytest tests/`.
8. **Telemetry-first, then instrument, then (rarely) run.** Validate/refute from telemetry already on disk
   before running anything. Ship telemetry + plot + pytest together with any new mechanism.
9. **Consult PARITY.md's vclock rules before any sim-clock change.** Clock is a monotone `max`; sim skips
   real waits and reconstructs order from sct (`SimReorderBuffer`).
10. **Sim MUST produce speedup: `sim_rate = vclock/wall ≥ 1`.** `< 1` means sim is stalling on a wait it
    should skip, or its commit throughput can't keep pace with arrivals.
11. **Correctness before speed; shared roots before per-baseline.** A bug failing rungs across ≥2
    baselines outranks a single-baseline one.
12. **Logical determinism is the parity definition.** Same trainers selected, same receipt order, same
    aggregations/rollbacks — differing ONLY in wall-clock. Prove it on the first data bin first.
13. **Do the right thing — no hacks.** A hack that moves a number without a correct mechanism is a
    regression in disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. No bare-scalar shortcut.
15. **Verify claims against code, not comments/docstrings.** A docstring claiming two functions are
    equivalent is a statement of intent, not a guarantee — diff them.
16. **Contention at scale → §D-1.** Refuted below ~100 trainers (§E); genuine root at n=100.
17. **A rotating cohort settling at `c − agg_goal` surplus is the correct steady state** for `c ≫ agg_goal`
    fedbuff — don't drive `carried_surplus_commits` toward 0.
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** A correctness-path value
    (seed, delay floor, agg_goal, c, trace, flag) must match across yaml, snapshot, and both roles'
    telemetry — divergent logging wastes sessions chasing phantoms.
19. **No compute on the critical path for a log the run doesn't need.** Gate any log with non-trivial
    args (`.item()`, hashing, `.norm()`) behind `logger.isEnabledFor(logging.DEBUG)` — an f-string
    evaluates its args even when the level would drop the line.
20. **Real/sim timing disagreement → fix real toward determinism, never inject noise into sim.** Sim's
    per-speed-class duration must stay clean (what makes `cohort_sequence` checkable). Fix real's
    measured completion time at the source.

### §F.1 Version & commit invariants (confirmed in code, both modes)

21. **`model_version` bumps once per COMPLETED data-bin** (variance PASS, `+= 1` at the data_id advance) —
    constant across one data-bin's iterations. `iteration_per_data_id` bumps on variance-FAIL retry, resets
    on data-bin advance. `version_key = (model_version, iteration_per_data_id)` changes every iteration —
    the sole step identity (§F-14).
22. **Commit == the update used for aggregation, at that instant — no lag.** Real: on ordered arrival. Sim:
    when vclock reaches the update's `sct` (buffer-unlock IS the commit). Never commit on a later event.
23. **Commit frees the compute slot immediately, but a version_key re-pick guard keeps the trainer
    un-pickable for the SAME `(model_version, iteration)`** until the version_key advances. TWO sets, never
    one (§D-27): CAPACITY is `_slot_holders()` (the only thing any cap may read), IDENTITY is
    `_sim_pending_commit`/`_real_pending_commit`. The slot must free at commit and never re-add after, or
    re-dispatch starves across variance-retry iterations; the guard holds to the agg-goal boundary (§D-15).
24. **Within a data-bin the global weights are constant; a re-picked trainer gets a RETRY, not a re-send.**
    Full WEIGHTS go out only for a `model_version` not yet received this data-bin
    (`_weights_sent_this_cycle`, cleared on the bump); a same-`model_version` re-pick gets VAR=bad, never a
    redundant weight re-send.
25. **One instruction per version_key: never dispatch to a trainer with an unresolved outstanding message
    for the CURRENT `version_key`.** Busy = silence, not a second message, until it returns or the
    version_key advances. Enforced by `_already_served_current_instruction`/`_mark_instruction_served`
    (end_id → last-served version_key). SYNC distribute always had this guard; ASYNC didn't (`fedbuff_round`'s
    `r1_inflight_overlap` was ~90% `VAR=bad` re-sends to an already-busy trainer — §G). Any new distribute
    call site must call both.
26. **Reuse the existing construct; don't duplicate per baseline.** New per-trainer/version state almost
    certainly needs an EXISTING mechanism (`_end_served_version_key`, `_keyed_topk`/`_keyed_draw`,
    `AsyncSelectorBase`), not a new one parallel to it — duplicate logic is duplicate bug surface. Tests
    too: extend a contract suite (`test_async_selector_base.py`, `test_selector_contract.py`,
    `test_selection_determinism.py`) before writing a bespoke one. Tell: about to add a variable/method/
    test whose name rhymes with an existing one (`_keyed_weighted_topk` next to `_keyed_topk`) — check
    whether the existing one should just take a parameter instead (already done: both now share one
    `_keyed_draw` primitive, §G).

27. **One event, one instant: never mix a pre-mutation snapshot with a live read.** An event describing a
    cycle must snapshot EVERY identity field (`round`, `data_id`, `iteration`, `model_version`) at the same
    point, before the branch that mutates them. `agg_round` mixed a live `self._round` with a snapshotted
    `cycle_data_id` and emitted a non-monotone key for a year of runs (§G). Corollary: `self.data_id` must
    never be observable outside `[0, total_data_bins-1]` — keep the lap wrap adjacent to the `+= 1`, and
    don't insert a read between them.

### §F.2 Porting a SELECTOR ≠ porting TIMING parity → §D-3

Moved to §D-3 (it's a diagnostic pattern, not an invariant). Kept here as a stub because prior sessions cite
"§F.2" — the class-hierarchy detail and the "diff destination aggregator/trainer against the shared base"
rule now live in §D-3.

---

## §G  Landed fixes — recent, load-bearing for current work only. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, immediately.** The instant a rung flips or a hypothesis resolves, write ONE line
> (mechanism + outcome) and delete it from §A/§B in the same edit. Newest first.

- **Per-baseline charges landed and closed 9 of 14 open fails.** Phase 0' re-ran only the four sim legs against
  the same reals, so the charge is the only variable. Cut 0.61-1.90 s/round; `fwdllm` 67/1→**68/0**,
  `felix_round` 65/9→**71/3** (incl. `v1c` `diverging`→`flat`, killing H7), `fedbuff_round` 71/4→72/2,
  `fluxtune` 77/0 held. `drain_wall_budget` closed on `fwdllm`/`fedbuff_round` — but CIRCULARLY, since the
  charge came from the same real it is graded against; Phase A's new real is the first honest test.
- **`selection_detail` reports re-picks inside one round-boundary re-draw (`sim_repicks_in_round`, §D-39).**
  `mean_chosen` averages a bimodal burst (one draw of c, then single-trainer top-ups) so it reported how many
  top-ups fired — `felix_round` read 2.41 vs 2.73 while the actual defect was **35 picks filling 30 slots, 5
  duplicates**. DIAGNOSTIC, does not gate: where reselection is event-driven `round` never advances, the run
  collapses to one bucket and every legitimate re-pick counts as a duplicate (fluxtune reads 18k on BOTH
  sides), so it is guarded on ≥2 distinct selection rounds. Gating wants the finer predicate — a re-pick while
  the end is still in flight — which needs in-flight state this rung lacks. Verdicts unmoved on all nine. 4 tests.
- **`inter_arrival_order` publishes `bucket_sizes` + `underpowered` (<4 buckets).** ρ was being read as a
  well-powered fail when `round` is coarse on the fwdllm family: a 7200s run reaches round 2, so `felix_round`'s
  0.662 is a mean over TWO buckets of ~15k/10k arrivals. `participation_parity` re-keys on event position for
  this; deliberately NOT copied — per-cycle buckets would correlate two ~10-trainer sets already at their
  independent-draw floor, manufacturing a fail (§D-2). Settle the intended unit first. 2 tests.
- **Cadence now gates on the drift RATE, not the level — `v1c_iter_drift_rate` (§D-35).** The variance gate is
  a feedback loop (var → iterations → updates → model → var), so a divergence COMPOUNDS and the pooled level
  is a function of run length: `felix_round` read +1.1% at 3600s and **+19.4% at 7200s on unchanged code**, so
  a tolerance calibrated at one duration is wrong at the next. The rung fits `ln(sim/real iterations-per-bin)`
  against the progress ordinal and t-tests the slope against zero — intrinsic to the run, and a longer run
  estimates it more precisely instead of mis-scaling. Calibrated on six real↔real pairs: |λ| ranges to 0.38
  per 100 units but NONE is significant (max |t| 1.61 vs a 3.355 critical value), and the two well-powered
  pairs sit at |λ| 0.003/0.033 — so `lambda_floor_per_100` is 0.05 and both it and the t-test must trip.
  It caught `felix_round` as the only `diverging` verdict (λ=+0.19, t=5.69) — since closed by the per-baseline
  charges, so all nine now read `flat`. `v1`/`v1b`/`v2` depend on it, so a flat rate makes a failing level
  DOWNSTREAM, not a root. 6 tests.
- **`v2_var_trajectory` grades the matched logical budget on async too (§D-4).** It applied the truncation only
  when `_real_intrinsic_clock` returned a coordinate — which is None for every async baseline by construction —
  so async graded the pooled run: `fluxtune` failed on **6.52%** while its matched window read **1.79%**, i.e.
  real's 94 bins against sim's 99. `real_coord` was a clock coordinate standing in for "is this sync"; the
  budget itself comes from `prog_fn`, which is mode-agnostic. Closed `fluxtune`; also ended a FALSE PASS on
  `fedbuff_round` (1.38% pooled → 4.7% matched). 2 tests.
- **`sim_clock_basis` — one primitive for "did the clock consume this?" (§D-31).** Sim's vclock advances from
  exactly two sources: `sct` on the trainer side and `charge_sim_vclock_overhead` on the aggregator side. A
  span outside both never reaches the clock, so grading sim's own wall for it fails as a function of host
  contention. `aggregation_compute_wall` was fixed for this last session; its three siblings were not.
  Now: `drain_wall_budget` grades `charged_s` when the label is profiled; `agg_step_timing_breakdown` gates
  only when a label charges `live`; `step_timing_breakdown` gates only when sim's compute BINDS
  `max(real_gpu_s, D)` (it never does at D~7-11s vs ~0.4s of JVP, so `_make_model_functional`'s KS
  0.233/0.237/0.282/0.458 across four baselines was one physical effect straddling a tolerance). An absent
  ledger reads UNKNOWN and keeps grading, never silently demotes. 6 tests.
- **`charge_coverage` [DIAG] — the standing audit.** Per label: sim wall vs what reached the clock vs REAL's
  own span for the same label. Exists so a mispriced or uncharged span announces itself on the run that
  introduces it. Immediately reproduced the Root-C mispricing unaided: `drain_tail` charged at 2.75x
  (`fwdllm`), 2.31x/2.24x (`fwdllm_it_*`), 1.47x (`fedbuff_round`), 1.30x (`felix_round`). 4 tests.
- **Per-baseline charge profiles + a launch-time provenance gate (§D-36).** One family-wide constant
  (`drain_tail` 0.2783 s/cycle, profiled 07-29 from two other baselines' reals) was **1.08-2.75x** each
  baseline's own real cost — 0.6-3.4% of sim's vclock, always making sim look slower. Nine profiles now live
  in `sim_charge_profiles/<baseline>.yaml`, generated by `profile_sim_charges.py` from that baseline's own
  post-fix real (7200s for the four that have one, the 1200s Phase-0 smoke for the other five; the charge is
  not materially duration-sensitive — 1200s↔7200s ratios are 0.80-1.21 where both exist). `sim_charge_profile_path`
  was already per-baseline in every sim yaml, so this is a repoint, not new machinery. `--only-observed` stops
  a refresh carrying an op the baseline never ran (sync fwdllm was inheriting `redispatch_turnaround`), and
  `run_sequential.sh`'s preflight now BLOCKS a launch whose charged entries were not profiled from a real run
  of the same baseline — matching on `_<baseline>_n` so `fwdllm` cannot accept `fwdllm_it_unaware`'s profile.
  4 tests + verified to block both the foreign-profile and sibling-prefix cases.
  - **`redispatch_turnaround` is NOT per-baseline** (§E). Its marginal is only valid under burst dispatch;
    on the event-driven baselines it prices inter-arrival waiting at 3.4-9.9x the real per-dispatch cost.
    `var_bad` stays OFF (sim performs the same dispatch and pays the same wall — nothing is skipped) and
    `weights` is pinned to the round-baseline 0.0598 except on `fedbuff_round`/`felix_round`, which measure
    their own (0.043/0.035) under a burst the guard accepts. Only `drain_tail`/`fedavg` are genuinely
    per-baseline — those are direct per-cycle span means, not timestamp differences, which is why the 2.75x
    mispricing they carried was real.
- **Eval cadence made DETERMINISTIC — the `convergence` root, and it was never duration.** `_eval_snapshot_model`
  returned None whenever the background eval thread was still busy, making WHICH commits evaluate a wall-clock
  race between the test-set pass and the inter-commit gap. Sim loses that race structurally: it compresses the
  gap (skipping real transport waits) while the eval costs the same or more wall. Measured at 3600s — real kept
  99-100% of its evals on all nine, sim kept **49-60% on seven of nine** (`fedbuff_it_unaware`: real 11.6s eval
  / 28.4s gap → 114/115; sim 17.0s / 12.7s → **63/111**). `fluxtune` was the lone escape at 98%, only because
  its cycles are 3-6x slower — which is why this hid for so long. So the modes sampled the accuracy trajectory
  at different, host-speed-dependent progress points (§F-12) and `_check_target_stop` saw a subsampled series
  in sim alone. Now gated on the commit INDEX (`eval_every_n_commits`, code default **2**, phased on commit 1);
  a still-busy thread is waited out and warned about, never silently dropped. N=2 fits every measured sim gap
  fleet-wide (tightest `felix_round`, 16.7s vs 14.7s). Set in all 18 yamls (§F-18). 15 tests, verified against
  the pre-fix behaviour. §D-30.
- **Six of nine sim yamls were missing `sim_charge_profile_path`** — so `fwdllm`, `fwdllm_it_*`, `fedbuff_it_*`
  and `felix_it` ran with `charge_source: live` and folded sim's OWN contention-inflated span onto the vclock:
  exactly the §D-18/§F-1/§F-20 defect, at `fedavg` 0.097s sim vs 0.051s real. The same omission §G already
  records once for `fluxtune`; it was never propagated to the rest. All nine now carry it. NOT retroactive —
  the runs on disk still charged live, so the six only clear on a fresh run.
- **`aggregation_compute_wall` now grades what reached the CLOCK, not sim's discarded span.** Where sim folds
  a real-PROFILED constant, its own wall is a §D-1 contention artifact the vclock deliberately throws away, so
  comparing it failed 8 of 9 purely as a function of how fast that baseline's real was: sim sat at a flat
  0.088-0.097s floor on every baseline while real tracked 0.051-0.098s. The rung now grades `charged_s` when
  `charge_source == "profiled"` (KS is dropped there — a constant has no shape) and reports
  `sim_wall_inflation_x` so §D-1 stays visible. `fluxtune`/`fedbuff_round`/`felix_round` flip green (1.23-1.36x
  inflation); the other six correctly still fail on `sim_wall`, being the six whose yaml was broken above.
  `vclock_charge` is now loaded by the checker. 8 tests. §D-31.
- **Real's over-`c` slot read CLOSED — drain lag, not the §D-27 conflation the tracker assumed** (§E).
  `_trainer_inflight_dispatch_version` clears when the drain loop PROCESSES a message, not when it lands, so
  the count charged the aggregator's own lag to trainer concurrency: `felix_it` real read up to **55** against
  c=30 on **49.1%** of dispatches (`fluxtune` 45, 8.1% — down from 90.4% but never closed) while the trainers'
  own `train_with_data_id` spans peaked at exactly **30**. Added `Channel.ends_with_pending_rx()` — queue depth
  only, never a deserialize, so it is safe on the dispatch path (§F-19) — and real's capacity read subtracts
  it. Sim is untouched (it ingests straight off the End queue, no lag). Identity half untouched (§D-27).
  15 tests. §D-33.
- **`step_timing_breakdown`'s `worst_func` now names a func that GATES.** It ranked over every func including
  the exempted ones, so `_emulate_training_delay` — a modeled sleep sim skips by design, KS 1.0 by
  construction — won every ranking and got read as a moving root cause (§E). The all-func winner is kept as
  `worst_func_incl_exempt`, and `failing_gating_funcs` now lists the real set. 5 tests. §D-32.
- **H4 CONFIRMED at 3600s — `fluxtune`'s clock family CLOSED, no code change.** `throughput` 9.7%→**1.4%**,
  `per_round_advance` KS 0.30→**0.157** (mean 2.1%), `overhead_residual` 2.1%, `overlap_factor` 5.151/5.161
  (0.2%), `slot_utilization` 29.43/29.78 (1.2%). The 900s residual was the duration exactly as predicted;
  `fluxtune` 62/11/16 → **71/3/16** and only `v2` (hairline), `convergence` (hairline) and D-1 remain.
- **`retask_before_close` CLOSED** — grades on real now, 0.0% both modes on every 3600s pair.
- **`felix_round`'s `v2_var_trajectory` and `fedbuff_it_oracular`'s CLOSED on their own at 3600s** —
  2.62%→**0.31%** and 4.05%→**0.87%**, no code change. `v2b_var_drift` reads `flat` on both. The Root-C level
  offset that survived the 900s batch is gone; `fedbuff_round` is the only cadence baseline still failing V2.
- **H1 + H2 both CONFIRMED on a 900s pair; 22 rungs flipped green across three baselines.** `fluxtune`
  62/11/16→**71/3/17**, `fedbuff_round` 58/10/21→**68/1/21**, `felix_round` 67/6/16→**68/1/21**, `fwdllm`
  unchanged. H1: sim mean in-flight 24.66→**29.64** vs real 29.22 (16.4%→1.4%), `overlap_factor`
  3.95→**5.836** vs real 5.79 (22.5%→0.8%), sim's own matched-window s/round 80.63→73.57; guard rails held
  (cap 30/30, `retask_before_close` 0.0%, `slot_starvation` 0/2153). H2 resolved to its SUB-hypothesis:
  `selection_bias` flipped on both round baselines with zero selector change, so the bias was downstream of
  the slot bookkeeping, never in the selectors — and it HOLDS at 3600s (≤1.2% on all nine). Its
  `fedbuff_round` cadence-family green did NOT hold: seven of those rungs fail again at 3600s (§D-29).
- **Real's compute SLOT split from its re-pick GUARD — the same §D-27 conflation, other mode.** Real's
  `_PendingCommitUnion` holds a returned end in `_per_agg_trainer_list` until the agg-goal boundary, so
  `_slot_holders()` counted trainers that had already stopped computing: `fluxtune` real read 30→39
  outstanding per cycle (max **52**) against c=30 and tripped the `outstanding_at_dispatch` INV on **90.4%**
  of dispatches (1259/1392), while its true peak concurrent training — swept from the trainers' own
  `train_with_data_id` spans — was **30** (mean 28.70, matching `slot_utilization`'s 29.22). So it was a
  measurement defect, not an over-dispatch. `_PendingCommitUnion.slot_holders()` is now real's capacity half
  (dispatched-not-yet-RETURNED; return IS real's commit instant, §F-22) and the union stays the identity set.
  The round baselines never showed it because they dispatch AFTER the boundary clear. Flag
  `real_commit_frees_slot`, default ON. Telemetry-only today — real's dispatch path does not read
  `_slot_holders()` — so it perturbs no run; the behaviour-changing half is scoped in §B. 5 tests, 3 verified
  to fail against the pre-fix read. The residual it left (`fluxtune` 52→45, `felix_it` 55) was a SECOND,
  unrelated defect — drain lag, closed separately above; this split itself was correct and complete.
- **Sim's compute SLOT split from its re-pick GUARD (§F-23) — `fluxtune`'s root** — §F-23 has two clauses
  (commit frees the slot; a version_key guard keeps the trainer un-pickable), and sim served both from one
  set, so §D-15's fix — hold a committed trainer's guard to the agg-goal boundary — also held its SLOT,
  denying it to everyone else. Sim ran at 24.66/30 mean in-flight vs real's 29.50/30 with `slot_starvation`
  0 in both modes: never short of candidates, just believing it had no free slots. Now `_slot_holders()` is
  the ONE capacity answer for both modes — `(inflight ∪ buffered) ∪ (pending − committed)`, published as
  `_agg_slot_holders_ref`; `_agg_pending_commit_ref` stays the identity set. Flag `sim_commit_frees_slot`,
  default ON. 16 tests. VALIDATED (H1, above). The "real is byte-identical, its roles already coincide"
  claim shipped with this fix was WRONG on live telemetry — see the real-side entry.
- **`slot_utilization` rung added (Stage-4 MECHANISM/EXACT)** — time-weighted MEAN/median slots busy, where
  `concurrency_cap` graded only PEAK: peak was 30/30 on every mode while the means were 29.50 vs 24.66
  (§D-20). Caught fluxtune's under-fill; fedbuff/felix pass at 1.3%/1.9%. `concurrency_cap` now also reports
  mean+median.
- **`selection_speed_bias` now GRADES the bias it was already computing** — it gated on KS alone, which is
  blind to a level shift when both sides have the same shape. Live finding: `fedbuff_round` 19.4% and
  `felix_round` 15.4% fail — sim's selector is speed-biased against its own pool in OPPOSITE directions on
  the two baselines (§B Part 2). `grad_norm` (G1) gained the same mean guard.
- **Lap-boundary identity snapshot fixed in `fwdllm_aggregator`** — `agg_round` read `round_num=self._round`
  live against a pre-mutation `cycle_data_id`, and `data_id` was transiently `== total_data_bins` when
  `version_bump_census` read it (§F-27). Training state was always correct — staleness keys on
  `_model_version`, never `_round`. 7 tests, each verified to fail without its fix.
- **`pytest-xdist` installed in BOTH `dg_flame` and `test_fwdllm`** — missing from both, so `setup.cfg`'s
  `addopts = -n auto` killed every invocation on `unrecognized arguments: -n`. Full suite 6m43s → 1m58s.
- **Budget COVERAGE graded + stamped on all 8 windowed rungs** — 8 of the 88 rungs share
  `_matched_logical_budget`; their numbers are only as good as the fraction of the run it covers, so
  `matched_budget_coverage` (Stage-0 CONTROL) grades it once and every windowed result now carries
  `budget_coverage` + a `low_budget_coverage` flag below 80%. Hard-fails below 50%, or on a
  `sequence_divergence` (the two sides committed DIFFERENT units — a real defect, not a windowing artifact).
  Current sweep: all `overrun`, min coverage 84.2-100%. 8 tests.
- **`run_parity.py` grades pairs in PARALLEL** — pairs share no state and read/write their own files; a pair
  is 77% file-load wall (13.3s of 17.3s). 9-baseline sweep 3m28s → 1m24s. `--jobs N`, default one worker per
  pair capped by cores and RAM (~4.5 GB/pair). 10 tests, incl. parallel≡serial output equality.
- **Non-monotone `(round, cycle_data_id)` key FIXED AT THE SOURCE (§D-26, §F-27)** — two separate
  `_process_aggregation_goal_met` defects. (a) `agg_round` read `round_num=self._round` LIVE while
  `cycle_data_id`/`cycle_iteration`/`cycle_model_version` were snapshotted pre-mutation, so the last bin of a
  lap emitted a post-bump round with a pre-mutation bin: `(1,148)→(2,149)→(2,0)`. Now snapshots `_cycle_round`
  alongside them. (b) Advancing off the last bin left `self.data_id == total_data_bins` (out of range) until
  the wrap several statements later, and `version_bump_census` read it there — emitting `data_id=150` once per
  lap. The wrap is now adjacent to the `+= 1`. Training state was always correct (staleness keys on
  `_model_version`, never `_round` — §F-2); these were record defects on a load-bearing key. 7 tests, both
  verified to fail without their fix.
- **Matched-budget primitive FIXED on the `data_id` axis (§D-26)** — independently of the source fix above,
  which only helps runs recorded after it: the primitive orders by event `ts` so it grades legacy telemetry
  correctly too. `max()` on the old key returned the
  FIRST bin of the last lap and `<= N` admitted the whole lap; and a max-key ceiling never required both
  sides to have committed the same bins (`felix_round`: 9 bins sim did and real didn't, inside a "matched"
  budget). N is now the position-wise common prefix of both chronological commit sequences, `prog_fn` its
  ordinal. Closed `terminal_state`/`total_commits`/`cohort_sequence` on FOUR baselines (§A) and turned
  felix's degenerate 4.7% into a real 0.9%. 5 tests.
- **`overlap_factor` (K4) REPAIRED and promoted DIAG→MECHANISM/EXACT** — its numerator `_per_round_max_speed`
  keys on FL `round`, which fwdllm holds static, so it returned the run-global max trainer, identical on both
  sides; the rung restated its own denominator. Now per-cycle barrier (`intrinsic_span_s`) ÷ per-cycle clock
  advance over the matched budget, and it is the only rung that localizes a throughput residual (§D-22).
  Separates fluxtune (5.09/3.95, FAIL) from fedbuff/felix (at parity). 4 tests.
- **`v2b_var_drift` rung added (§D-23)** — bins the matched budget by progress and reports the sim/real `var`
  ratio trend, distinguishing a per-cycle mechanism (level offset) from diverging trajectories (progressive).
  fedbuff_round = progressive_drift, felix_round = level_offset, fluxtune = flat. DIAG, never gates. 4 tests.
- **`replicate_floor.py` + throughput-family tolerance calibrated to it (§D-24)** — measures same-seed
  real↔real spread from runs already on disk: committed bins 3.4-4.5%, cycles 0.6-1.4%, iters/bin 2.8-3.9%,
  mean var 0.6-1.2% at n=100/3600s. `_THROUGHPUT_FAMILY_TOL_REL` 0.05→**0.08** (5% sat inside the pipeline's
  own noise). `v2`'s 2% survives the floor — felix's 3.45% is small but real. 13 tests.
- **Run-length policy switched to 900-1800s by default (§C, §D-25)** — grounded on the 1800s pairs: the clock
  family fires at 17-23% at 30 min, the cadence rungs read 3.1%/1.4% against the long run's 14.8%/5.9%.
- **Round-boundary BACKFILL validated** — `fedbuff_round` sim peak 35→30/30, self-overlap 553→0,
  `[ConcurrencyBackfill]` 29×, 0 `slot_starvation`; felix/fluxtune stay green. `concurrency_cap` CLOSED on
  all three. Inert on throughput (dispatches/cycle 10.02 unchanged).
- **Profiled `drain_tail`/`fedavg` charge validated as a charge, falsified as a sizing** — charge landed
  exactly (0.663→0.343 s/cycle, `charge_source: profiled`) but fluxtune throughput only 13.2%→11.9% (29%
  pass-through, §D-18), and it moved variance cadence 12-18pp on both round baselines — Root C, §B/§D-21.
- **`drain_tail`/`fedavg` moved from LIVE-span to REAL-PROFILED vclock charge (§D-18)** — sim's own span is
  contention-inflated 1.77x at n=100; charging it violated §F-1/§F-20. 2 tests.
- **Round-boundary BACKFILL landed** — `_cap_dispatch_to_concurrency()` + `_exclude_pending_commit` now on
  the cohort accumulate branch, not just the reuse branch. 5 tests.
- **Root A′ (`felix_round` +14.7% cycles/bin) CLOSED as a misattribution** — it was never felix-specific;
  the whole residual moved on a charge change that never touched felix's path. Absorbed into Root C (§B).
  felix throughput family now green, but for the wrong reason — do not bank it.
- **`concurrency_cap` re-based onto peak DISTINCT in-flight ends from `contributor_intervals` (§D-20)** —
  grades REAL on every baseline with no fresh run; `fluxtune`'s 31/30 was a phantom (1218 false positives).
  Same-end concurrent dispatches now graded separately (§F-25). 8 tests.
- **`_outstanding_dispatch_count()` cap arithmetic subtracts `_sim_committed`** — superseded by the
  `_slot_holders()` split above, which subtracts it only when no LIVE re-dispatch re-occupies the slot.
- **`selection_detail` windowed to matched WORK + event COUNT now graded** — was pooling each side's whole
  run, so it graded which side lapped. Green on all three; full-run counts still reported. 5 tests.
- **`v1_iter_per_data_id`/`v1b` normalizer FIXED (§D-19)** — keyed per data-bin VISIT `(round,
  cycle_data_id)`, completed visits only, truncated to the matched budget. Ends felix_round's false 4.4%
  PASS (true 14.7%); fedbuff `v1b` green. 4 tests. 1092 `pytest lib/python/tests` pass.
- **§D-15 fully validated, "overshoot" reclassified** — `busy+idle` matched on all three; the felix/fluxtune
  residual is two unrelated roots (§D-16/17/18), not an over-corrected barrier.
- **`selection_detail`'s "inverted granularity" closed as a non-finding** — lap-boundary crossing, not two
  selection policies. felix = checker windowing (21/22 events out-of-budget); fedbuff = duration-gated coin
  flip. Cohort pinning is CORRECT (operator ruling, §D-17).
- **`felix_round`'s "why does composition diverge when selection is deterministic" ANSWERED** — selection is
  deterministic; one side crossed the round boundary inside the graded window, the other didn't (§D-17).
- **`fedbuff_round`'s throughput family CLOSED by the §D-15 boundary-release fix** — validation pair
  `_034223`/`_220025`: `throughput`/`total_commits`/`terminal_state`/`per_round_advance`/`overhead_residual`/
  `cohort_sequence` all flip green, residual 14.0%→2.2%.
- **Mid-cycle redispatch ELIMINATED in sim, all three round-cadence baselines** (§D-15) —
  `retask_before_close` 78-90%→0.0%. Root confirmed; magnitude correct on all three.
- **`redispatch_turnaround.weights` cumulative-batch overcharge FIXED** (§D-12, 0.488→0.0598s, 7.3-8.3x) and
  RULED OUT as the throughput residual's source (§D-14 totals check). Value is now settled — do not re-tune
  (§F-13).
- **`fluxtune` sim yaml was missing `sim_charge_profile_path`** — every `redispatch_turnaround` row read
  `charge_source: "none"`, ~180s uncharged. Added; `convergence`/`selector_score`/`utility` green.
- **fwdllm's `throughput`/`terminal_state`/`total_commits` 5400s fails ROOT-CAUSED as a CHECKER bug, FIXED +
  VALIDATED** (§D-13). `_matched_logical_budget`/`_per_progress_last_event` treated an uncommitted trailing
  `data_id` (real hit `max_runtime_s` one variance-check attempt into it) as reached progress, comparing
  real's ~22s partial cycle against sim's ~205s completed one. Fixed to require `var_good_enough == True`
  evidence per `data_id`-axis key. Re-run: 56/6/22 → 59/3/22, only the pre-existing D-1 contention group
  remains. 1392 `pytest lib/python/tests` pass. CLOSED.
- **`redispatch_decomp`/`vclock_charge` dark-data gap closed** — `analyze_run.py` gained
  `redispatch_decomp_plots`/`vclock_charge_plots` (CDFs by mode/payload_kind + a `post_close_overhead` mean
  bar + a "real span vs sim charged" uncharged-gap bar), wired into `_PLOT_GROUPS`. Verified against live
  `fedbuff_round` real+sim telemetry (1.9M records) + 5 new pytest cases
  (`tests/analysis/test_redispatch_vclock_charge_plots.py`). CLOSED.
- **`redispatch_turnaround.weights` charge fix VALIDATED live on `fedbuff_round`/`felix_round`, 3600s.**
  `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/`terminal_state` all flip fail→pass,
  residual 1.5-4.8% (well under 10% tol) — the round-cadence dispatch-path root closed with no meaningful
  overshoot. CLOSED.
- **`async_oort.py` re-based onto `AsyncSelectorBase`** (2193→897 lines) — was the last selector carrying
  its own full copy of the send/recv concurrency mechanism, drifted from fixes landed only on the base.
  Utility-scoring POLICY (pacer, exploration split, `select_type` strategies) kept unchanged, routed through
  shared `_choose`/`_pre_choose` hooks; eval-task state and the `_cleanup_removed_ends` ghost-sweep override
  kept as genuine Oort-only behavior, not assumed away. `slot_starvation` telemetry promoted to the shared
  base; `_keyed_draw` extracted as a shared primitive. Full suite 1387/1392 pass (net +5 tests), 0
  regressions. Integration-level real+sim confirmation still open (§B).
- **General profiled-charge mechanism landed (FWDLLM_DESIGN.md §P).** `sim_charge_registry.yaml` +
  `get_profiled_charge_s()` loader + `charge_sim_vclock_overhead(..., profiled_s=...)` — a registry entry's
  `charge: true` flips a category on, independent of `sim_model_agg_compute_time`. `redispatch_turnaround.weights`
  seeded `charge: true, mean_s: 0.4365` (pooled n=3264) from a `run_20260728_151722`/`_151831` real pair;
  `.var_bad` left `charge: false` (measured negligible, ~2x not ~13x). `vclock_charge` ledger gained a
  `charge_source` field (`live`/`profiled`/`none`). 8 new tests.
- **`redispatch_decomp` widened to cover `VAR=bad` dispatches (not just fresh `weights` sends) + new
  `vclock_charge` ledger event added.** `_pk in ("weights", "var_bad")` gate + `payload_kind` field
  (`fwdllm_aggregator.py:4135`); `charge_sim_vclock_overhead` (the ONE shared fold-onto-vclock function, used
  by every fwdllm-family baseline) now emits `vclock_charge` on every call in both modes, plus a
  `charge: bool` param so a candidate category can be measured before being charged. Root-cause
  reconciliation (§D-11) found the round-cadence throughput gap is a per-cycle cost compounding ~10x — the
  old telemetry only measured the 1-of-10 terminal cycle. 4 new tests, 573 `-k "fwdllm or telemetry or
  parity"` pass.
- **`felix_round` sim + `fluxtune` (both sides) missing-run episode resolved by relaunch, not a code fix.**
  Both pairs re-ran clean (`run_20260728_102750`/`_113005`, `_102831`/`_113046`); the `DIRTY_ABORT`
  hypothesis was never confirmed on the run node and is now moot for this batch.
- **`selection_detail` CONFIRMED at production scale** — `_exclude_pending_commit` passes on both
  `fedbuff_round`/`felix_round` at n=100/c=30/3600s, not just n15. CLOSED.
- **`fedbuff_it_unaware` 6→1 fails confirmed duration-gated, not a bug** — running the same config 3600s
  instead of 1800s alone resolved 5 of 6 fails; no code change needed.
- **`run_sequential.sh`'s `fedbuff_round`/`felix_round` yaml mapping briefly mis-set to the n15 debug-scale
  files, then reverted.** The "n10"-named files are actually production n=100 scale (misleading filename
  only); n15 is a deliberate reduced-scale repro, not a substitute. `ALL_RUNS` now correct.
- **`fedbuff_it_oracular` 12→4 fails confirmed at scale** — `get_curr_unavail_trainers`/
  `get_curr_task_ineligible_trainers` DEBUG-gating fix (below) held at n=100/3600s; remaining 4 fails tracked
  in §B (profile-own-charge group).
- **`slot_starvation` telemetry landed** — surfaces a freed dispatch slot with no eligible candidate
  (`feasible_extra < extra`), shared by every async baseline.
- **All 9 baselines' smoke yamls bumped 1800s→3600s** + real/sim wall-ceiling watchdogs, matching the §C
  run-length bar (enabled the full-scale batch, §A).
- **`[LAG_DECOMP]` + `redispatch_decomp` telemetry landed for fwdllm** — splits redispatch wall gap into
  peer-wait vs post-close overhead, ported from asyncfl's shared version.
- **2 more stale "sim uses in-mem cache" doc claims found + fixed** (`checks.py` docstring, PARITY.md,
  PARITY_CHECKER_README.md) — no cache exists in either mode; real reason is dispatch-cadence /
  aggregator-side overhead.
- **`staleness` (U3) root-caused, FIXED, VALIDATED — CLOSED** (§D-9). Real's `_agg_pending_commit_ref` only
  covered return-time state, missing a still-training end and flooding it with re-dispatches (one end: 526
  sent vs 284 processed). Fixed via `_PendingCommitUnion` (dispatch-time + return-time, mirroring sim's
  `_sim_pending_commit` span). Confirmed on the post-fix pair: real 0.056/0.064 vs sim 0.095/0.12 (was
  real≫sim 7.62/9.10 pre-fix). Unmasked a separate `throughput`/`per_round_advance` gap, not a regression
  from this fix.
- **`[RecvBootstrap]` deadlock FIXED + VALIDATED.** A bootstrap added for a trainer-side crash also raced
  the aggregator's own first SEND tick, phantom-filling every dispatch slot and permanently freezing sim's
  vclock at 0.0 (reproduced at n=15 with zero GPU sharing possible — also falsified the GPU-contention
  hypothesis for the same eviction). Fixed by gating the bootstrap behind `allow_recv_bootstrap`, set only by
  single-parent callers (`channel.one_end()`); `channel.ends()` dispatchers default to no-bootstrap. Re-run
  confirms no repeat of the freeze.
- **`_exclude_pending_commit` FIXED + VALIDATED** (§D-8). Round-cadence's cohort-reuse cache-hit dispatch
  path bypassed the busy-exclusion guard entirely (called once per run, not per tick). Fixed by filtering
  the cache-hit list every tick. Flips `r1_inflight_overlap`/`selection_detail`/`participation`/
  `training_budget`/`overhead_residual`/`per_round_advance` fail→pass. NOT also a `staleness` fix — that was
  a separate, later-fixed bug (above).
- **Variance-check pool stopped rate-scaling stale contributions** (§D-7) — was scaling round-cadence's
  genuinely-stale carried-surplus entries toward zero before the variance gate, faking convergence (real
  committed data_ids in ~1.7 JVP samples vs fluxtune's ~16). Fixed: stop scaling the var-check pool by
  `rate`; model-update merge untouched. Validated for cadence effects; also exposed a previously-masked
  `fluxtune` compounding effect (§B).
- **R-D landed + VALIDATED** — async distribute never had the sync path's one-instruction-per-`version_key`
  guard (§F-25); `fedbuff_round`'s `r1_inflight_overlap` was ~90% `VAR=bad` re-sends to already-busy
  trainers. Fixed via `_already_served_current_instruction`. Confirmed: `r1_inflight_overlap` real 0.0%/sim
  0.0% both baselines (was ~90-91%); the flood's `phase_gpu_compute` contention side-effect also resolved
  (13.5s→4.8s fedbuff, 7.9s→3.6s felix).
- **`sample_by_util` reproducibility fix landed + VALIDATED** — `np.random.choice(p=...)` was pool/order-
  dependent; replaced with `_keyed_weighted_topk` (Efraimidis-Spirakis keys). Confirmed on `felix_it`:
  `cohort_sequence`/`v1b_iters_moving_avg`/`utility` all flip fail→pass.
- **`get_curr_unavail_trainers`/`get_curr_task_ineligible_trainers` INFO-logged every call, ungated** — a
  300-entry trace scan + log fired every iteration-cadence tick in ORACULAR mode (4735×/1243s run); gated
  behind DEBUG (§F-19). Confirmed effective at scale (`fedbuff_it_oracular` 12→4 fails, above).
- **R-C landed** — round-cache stuck-timeout now clocked on vclock in sim / wall in real
  (`_round_cache_clock_now`); sim evicted 0 stuck ends vs real's 6.
- **R-A landed** — `FedBuffSelector` re-based onto shared `AsyncSelectorBase` (832→54 lines), inheriting the
  version_key re-pick guard, R1 guard, vclock timeout, avl filter, full drain, `_keyed_topk` sampling. Real
  had been re-picking the same-version trainer on 34.4% of commits vs sim's 0.9% pre-fix.
- **R-B landed** — pinned round cohort was sized/checked against `agg_goal` instead of `c` (under-fill:
  `fedbuff_round` froze at 10/100 with 20 idle slots; over-fill: `felix_round` 30 real vs 40 sim). Fixed:
  `_round_cohort_target`/`_trim_round_cohort` target exactly `c`.
- **`reselect_cadence` knob added** (round/data_bin/iteration); `AsyncRandomSelector` collapsed 850→31 lines
  (zero methods of its own); selector stats de-duplicated ×4 into `AbstractSelector`; dead heartbeat
  mechanism deleted (no sender ever existed).
- **Async-baseline decisions settled** — extraction reference is `async_oort.py` (not `async_random.py`,
  collapsed to a stub); cohort target is `c`, trimmed exactly (`agg_goal` is only the aggregation trigger);
  `reselect_cadence` is a first-class knob.
- **fluxtune 3→0 fails (69/0/16)** — the 3 remaining fails were ONE boundary-race cascade on a
  stochastic-async selector, not a sim bug (§D-2); checker now gates index-identity for stochastic-async
  selectors, keeps marginals enforced.
- **fwdllm timing family root-caused** — co-location contention (byte-identical inputs, sim's per-op floor
  matches real's typical every decile), not sim over-compute (§D-1). `_flat_grad_norm` fix landed but
  insufficient alone (§E).
- **fwdllm/fwdllm_plus `throughput` CLOSED at 7200s** — `recv_fifo`→`drain_ready` + var_bad dedup held
  (3.2%/4.8%, both PASS).
- **Checker overhaul** — `matched_virtual_budget` deleted, graded on logical budget N instead (§D-4);
  `pctl_band_ok` DIST-band escape added; `_step_timing_compare`'s `band_min_abs_s` floor fixed (was masking
  5x regressions at ms-scale, copied from a 1s-scale metric).
- **Early foundation** (compressed — full detail via `git log` on this file): P0-1 deferred-merge buffering
  landed; fluxtune's sim-side vclock/pacer/EOT-stamping bugs fixed and real's `_agg_pending_commit_ref`
  bound, taking fluxtune 19→5 fails at 7200s; startup GPU-health crashes fixed; `cohort_sequence` grading
  made distributional; `[SELECT_TRACE]` debug logging added then removed once the divergence was localized.
