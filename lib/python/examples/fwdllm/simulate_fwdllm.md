# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 ported fedbuff/felix-lineage
baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3).
Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf, sim barrier redesign, delay-factor
calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/
gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — how to use this doc
>
> **Fresh session? Read in this order:** §F (locked invariants) → §E (dead ends) → §A.1 (readiness ledger:
> where every baseline is) → §B.0/§B.2 (exit criteria + what to run next) → §D (only the lessons for the rung
> you're chasing) → §C for ladder mechanics.
>
> | section | contents | update rule |
> |---|---|---|
> | §A.1 | readiness ledger — the five stages per baseline | tick a stage the moment its artifact exists |
> | §A.2-4 | parity rows, floors/accuracy bands, legacy OFF board | rewrite in place on every >3600s run |
> | §B | exit criteria · the pipeline · next-up queue · open questions · backlog | current state only; an item lives here XOR §G |
> | §C | ladder/decomposition method, run-length budget | edit only if the method itself changes |
> | §D | durable lessons — transferable invariants | ≤30 words each; update in place, never append near-dupes |
> | §E | dead ends — falsified hypotheses | one line each; never re-open |
> | §F | locked invariants — always-true / always-do | operator approval + evidence; amend in place, never renumber |
> | §G | closed items | move here the instant a §A/§B issue resolves; delete the source in the same edit |
>
> **No hypothesis numbering.** A question lives in §B.3 with its falsifier until it resolves, then it becomes
> a §D lesson (true and worth keeping), a §E dead end (false and worth not re-opening), or nothing at all.
> Carrying an H-number past its answer is how the doc grew a hypothesis zoo.
>
> **Living doc, not a log — no dated annotations.** Every claim must read as true right now. §G is the one
> exception: newest-first by position, not by date. Full history is `git log` on this file.
>
> **Non-negotiables:**
> - Correctness per mode first; parity is the consequence, never the goal. A rung green because both sides
>   are equally wrong is a regression (§D-5). A divergence names two disagreeing sides, never which is at
>   fault — check each side's own absolute signal before choosing which to change (§D-9).
> - **Grade against §B.0's EXIT CRITERIA, not the pass count.** Parity is done when sim cannot change the
>   conclusion — the claims are comparative, so a common-mode residual costs nothing and driving it to zero
>   is over-optimization. Read that list before opening any investigation into a red rung.
> - Parity findings/fixes only here; design decisions, roadmap and calibration derivations belong in
>   FWDLLM_DESIGN.md.
> - Ground every claim in telemetry already on disk before instrumenting or running; fix root causes, not
>   symptoms; use the `dg_flame` conda env; ship new telemetry with its plot + pytest in the same change.
> - **Reproduce on the bench before spending a run.** Once an issue is isolated, write a throwaway script
>   that drives the real code path (import it; a reimplementation proves nothing). Then: (1) confirm the
>   CONTROL reproduces — a clean control means the SCRIPT is wrong, and this is the step that gets skipped;
>   (2) A/B the candidate fix against it; (3) only then implement behind a flag and spend ONE run. Match the
>   bench conditions to the mechanism, including the object's MODE (§D-47), or the control will not fire.
>   A bench arm costs minutes and isolates one mechanism; an FL pair costs hours and carries every confound,
>   so its null rarely says *which* thing was wrong. Extends §F-8 with the missing rung: telemetry → bench
>   → run.
> - Runs happen on a separate operator-controlled node: print the command, never launch or babysit one.
>   Assume no run is in flight unless told otherwise, and that a baseline's real/sim pair runs ONE AT A TIME
>   per node — different baselines run in parallel elsewhere, so only same-baseline real→sim ordering is
>   meaningful.
> - Run artifacts live under `experiments/run_<timestamp>_<name>_<syn>_<real|sim>/`.

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

## §A  Score — the ON campaign

Everything here is **`jvp_eval_mode` ON**. **An OFF row is a different training config and NEVER compares to
an ON row** — §A.4's board, all nine tolerance sets and every charge profile older than the flag were measured
dropout-live. §A.4 survives only because `fwdllm_it_oracular` has no ON evidence at all.

### §A.1  Readiness ledger — what each baseline still needs

Five stages, in order, each feeding the next. **A parity row is readable only when all five are ✓.**

| | stage | artifact |
|---|---|---|
| **R2** | two 7200s ON real legs | `experiments/run_*_<b>_*_real` |
| **FL** | replicate floor measured from them | `parity_floors/<b>.yaml` |
| **CH** | charge profile derived from THOSE reals | `sim_charge_profiles/<b>.yaml` |
| **SIM** | ON sim leg launched AFTER CH | `experiments/run_*_<b>_*_sim` |
| **GR** | graded row | `experiments/_parity_reports/` |

| baseline | R2 | FL | CH | SIM | GR | next stage |
|---|---|---|---|---|---|---|
| `fluxtune` | ✓ | ✓ | ✓ | ✓ | ✓ **75/0/17** | — |
| `felix_round` | ✓ | ✓ | ✓ | ✓ | ✓ **73/1/18** | — |
| `fwdllm_it_unaware` | ✓ | ✓ | ✓ | ✓ | ✓ **69/0/23** | — |
| `fwdllm` | ✓ | ✓ | ✓ | ✓ | ✓ **68/0/24** | — |
| `fedbuff_round` | ✓ | ✓ | ✓ | ✓ | ✓ 72/3/18 | — |
| `felix_it` | ✓ | ✓ | ✓ | ✓ | ✓ 72/4/17 | — |
| `fedbuff_it_unaware` | ✓ | ✓ | ✓ | ✓ | ✓ 67/5/18 | — |
| `fedbuff_it_oracular` | ✓ | ✓ | ✓ | ✓ | ✓ 64/9/18 | — |
| `fwdllm_it_oracular` | ✗ | ✗ | ✗ | — | — | **2 ON reals → FL → CH → SIM** |

**Eight of nine baselines now have all five stages.** Every CH came from that baseline's own two ON reals,
every FL from a same-node replicate pair, every sim leg from a flag-matched real. The two mechanical guards
(preflight blocks a stale profile; comparator pairs on matching flag) are what make that true without anyone
remembering to do it — `felix_it` graded against its ON real and printed the two newer OFF controls it
skipped (§G).

### §A.2  Parity rows, ON

`run_parity.py`; ✓/✗/– = pass/fail/skip; rung catalog PARITY.md §F. Per-pair numbers:
`experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`.

| baseline | real / sim | N | pass/fail/skip | cohort | vclock | K4 | slots | sbias | thru | commits | terminal | V1c | V1 | V2 | U3 | S2 | conv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fluxtune` | `20260803_042016` / `_170102` | 96 | **75/0/17** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ |
| `felix_round` | `20260803_021757` / `_115430` | 196 | **73/1/18** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `fwdllm_it_unaware` | `20260803_175853` / `_200051` | 40 | **69/0/23** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fwdllm` | `20260803_062233` / `_160414` | 41 | **68/0/24** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fedbuff_round` | `20260803_021819` / `_161527` | 185 | 72/3/18 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `felix_it` | `20260803_062254` / `_174309` | 254 | 72/4/17 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | – | ✓ | ✓ | ✓ |
| `fedbuff_it_unaware` | `20260803_190011` / `_210301` | 190 | 67/5/18 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| `fedbuff_it_oracular` | `20260803_175959` / `_200248` | 183 | 64/9/18 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |

Budget coverage: 100.0/97.0 · 100.0/98.0 · 100.0/95.2 (N=40, thin) · 100.0/93.2 (N=41, thin) · 98.9/100.0 ·
100.0/100.0 · 97.4/100.0 · 95.8/100.0. All eight passed the validity gate — every trainer that trained logged
the knob, zero `False`, zero tracebacks, achieved span 6928-7194s. The only runtime source change inside the
window is the OFF-by-default variance-pool audit, and both the clean and the dirty group straddle it, so the
4/4 split below is not a code-version artifact.

**The whole board is ONE residual — sim over-iterates per data-bin — and on seven of eight rows that residual
is the pipeline's own replicate spread, not a sim defect.**

Measure `v1` on the `cycle_model_version` axis — never `data_id`, which wraps every lap (§F-2/§F-21) and
pools laps into one bin. Then grade every row against its own real↔real control before reading it as a sim
defect (§D-45, §D-55): both real legs are on disk for all seven, and the control is the same per-bin
comparison run realA↔realB.

| baseline | c vs agg_goal | grad bit-match real↔sim | `v1` real↔**sim** | `v1` real↔**real** | verdict |
|---|---|---|---|---|---|
| `fwdllm` | 10 = 10 | **420/420 all run** | +0.00% | 0.32% | **INSIDE floor** |
| `fwdllm_it_unaware` | 10 = 10 | **410/410 all run** | +0.00% | — | **INSIDE floor** |
| `fluxtune` | 30 > 10, capped | 26/71, then **0** | +0.00% | 0.83% | **INSIDE floor** |
| `felix_round` | 30 > 10 | 14/143, then **0** | +0.04% | 1.47% | **INSIDE floor** |
| `fedbuff_round` | 30 > 10 | 8/161, then **0** | +6.9% | **6.44%** | 1.1x floor |
| `fedbuff_it_unaware` | 30 > 10 | 18/70, then **0** | +9.3% | **7.11%** | 1.3x floor |
| `felix_it` | 30 > 10 | 4/116, then **0** | +8.2% | **4.80%** | 1.7x floor |
| `fedbuff_it_oracular` | 30 > 10 | 27/67, then **0** | **+13.8%** | 2.70% | **5.1x — GENUINE** |

Every red cell in the score table reduces to `v1`. `cohort_sequence` fails only on its `count` gate, which is
`v1` rolled up — its composition half is at the independent-draw floor and gated stochastic (§D-2).
`selection_detail`'s failing term is `rel_diff_n_selections`, the same 13.8% on `fedbuff_it_oracular` (24600
vs 28540 selections) with `chosen`/`inflight`/`effective_c` all exactly 0.0%. `v1b` is `v1` cumulative.
`fedbuff_it_oracular`'s three EXACT fails are the extra iterations reaching the clock: `throughput` 7.6%,
`terminal_state` and `total_commits` 11.9%. `convergence` follows the shifted curve.

**Seven of eight rows are at or near their own replicate floor. `fedbuff_it_oracular` is the one genuine
sim-specific residual on the board.** The control is not a formality here: it carries the SAME late-run
concentration the sim rows do (`felix_it` real↔real runs +15.3% in its last quintile, `fedbuff_it_unaware`
+7.7%), so the accumulation is a property of the variance→iteration feedback loop, not of sim. §B.0's first
over-optimization tell is "chasing a rung whose residual is inside the replicate band" — four rows are
squarely there and three more are within 1.7x.

**Where the loop is removed, parity is EXACT, real↔real and real↔sim alike.** `fwdllm` matches 420/420
gradient norms across modes and reads `v2` +0.00% between its two real legs. So nothing in the stack is
irreducibly noisy: model init, data partition, JVP direction and training are bit-reproducible, and the whole
spread is the async admission loop amplifying wall-clock luck. `c == agg_goal` removes it because all
in-flight ends ARE the cohort; every `c > agg_goal` baseline loses bit-identity inside the first quintile and
reads exactly 0 thereafter (§D-54).

**Three mechanisms were proposed for the residual and all three are REFUTED on telemetry** — the boundary
tie-break, the redispatch stagger, and cohort composition bias (§E). Sim already canonicalizes exact `sct`
ties by end id, and cohort statistics are indistinguishable: mean modeled duration within 0.2%, staleness
within 0.02, participation green on all eight.

**The sign reversal is gone.** Sim is ≥ real on all eight, never below; re-profiling took `fwdllm` from
+18.0% to +0.00% and `fedbuff_round` from +12.8% to +7.4%. That confirms the stale-charge diagnosis and
retires it as an explanation for what is left (§G, §D-50).

### §A.3  Replicate floors and accuracy bands, OFF → ON

`replicate_floor.py --mode real`, 7200s, seed 1234, config-identical legs, grouped by achieved span (§D-44)
and flag state. **The floors are inputs to the checker, not a table**: `--profile-out` writes
`parity_floors/<b>.yaml`, and DIST tolerances tighten toward `3x floor`, never past 2% absolute, never looser
than nominal, and SKIP once the floor swallows the tolerance. No profile ⇒ nominal gates (§D-36).

| baseline | agg rate | bins | cycles | iters/bin | mean_var (tol 2%) |
|---|---|---|---|---|---|
| `fwdllm` | old | 0.0 → **0.0%** | 0.0 → 0.3% | 0.0 → 0.3% | 6.8 → **0.2%** |
| `fluxtune` | old, capped | 1.1 → **0.0%** | 1.0 → 0.8% | 2.0 → **0.8%** | 21.3 → 6.6% |
| `fedbuff_round` | old | 1.1 → 4.3% | 2.8 → **0.2%** | 3.9 → 4.1% | 4.7 → **1.5%** |
| `felix_round` | new | 12.9 → **1.5%** | 0.1 → 0.5% | **13.3 → 0.6%** | 0.5 → 0.4% |
| `felix_it` | new | 2.3 → 2.7% | 0.1 → 0.4% | 2.2 → 2.3% | 2.4 → 2.4% |
| `fwdllm_it_unaware` | old | ON only **0.0%** | ON only 0.3% | ON only **0.3%** | ON only **0.1%** |
| `fedbuff_it_oracular` | old | ON only 1.5% | ON only **0.0%** | ON only 1.5% | ON only 1.9% (tight) |
| `fedbuff_it_unaware` | old | ON only 4.9% | ON only 0.1% | ON only 4.9% | ON only **0.6%** |
| `fwdllm_it_oracular` | — | — | — | — | **UNMEASURED — no replicate exists** |

The three ON-only rows have no OFF counterpart at this duration (their OFF legs are 1200s or single), so they
carry no flag delta — they are floors, not a before/after.

⚠ **These floors are measured on a window the rung does not grade, and they UNDERSTATE it.** `metrics()`
takes each leg's run-level mean over its FULL run; `v1` grades the matched prefix the two legs share. Run the
same two legs both ways (control = the shipped `metrics()`/`_spread()`, imported) and the matched-window floor
is 1.0-2.6x larger — and it diverges *exactly* where the variance-cadence loop is free to run:

| baseline | shipped floor | matched-window floor | ratio |
|---|---|---|---|
| `fwdllm` · `fluxtune` | 0.32% · 0.83% | 0.32% · 0.83% | **1.0x** — loop pinned, estimators agree |
| `felix_round` | 0.58% | 1.47% | 2.6x |
| `fedbuff_round` | 4.07% | 6.44% | 1.6x |
| `felix_it` | 2.26% | 4.80% | 2.1x |
| `fedbuff_it_unaware` | 4.92% | 7.11% | 1.4x |
| `fedbuff_it_oracular` | 1.54% | 2.70% | 1.8x |

The run-level mean averages the per-bin drift away against each leg's differing tail, so it reports a floor
the rung's own window never sees. Fix `metrics()` to grade the matched prefix (§B.4); until then every DIST
tolerance derived from these is too tight on exactly the baselines that need it loosest.

**Eval mode collapsed the cadence floor only where one existed.** `felix_round` went 13.3% → 0.6%, a 22x
drop, and it is the loss-derived-weight baseline; the integer-staleness ones had no floor to lose and did not
move. `felix_it` is also loss-derived and also did not move (2.2 → 2.3%) — so the rule is "where a floor
existed", not "every new-rate baseline". `fedbuff_round`'s bins floor moved the WRONG way (1.1 → 4.3%) — one
metric against nine, re-check before trusting its `throughput`.

`v2_var_trajectory` is gradeable on six of eight: `fwdllm_it_unaware` 0.1%, `fwdllm` 0.2%, `felix_round` 0.4%,
`fedbuff_it_unaware` 0.6%, `fedbuff_round` 1.5%, `fedbuff_it_oracular` 1.9% (tight). `felix_it` 2.4% and
`fluxtune` 6.6% are above its 2% tolerance and SKIP.

**Peak accuracy, ON vs the OFF band** (§D-44: both legs cut at the matched achieved span):

| baseline | OFF legs | ON legs | band OFF → ON | verdict |
|---|---|---|---|---|
| `fluxtune` | 83.38 / 83.61 | 84.33 / 83.18 | 0.23 → 1.15 | straddles OFF — no change, on the tightest control |
| `fedbuff_round` | 75.11 / 73.32 | **77.49 / 81.20** | 1.79 → 3.71 | ON entirely ABOVE OFF, +2.4 to +6.1 pts |
| `felix_round` | 77.17 / 66.01 | **75.87 / 81.12** | 11.16 → 5.25 | band halved, top end +3.9 pts |
| `fwdllm` | 28.74 / 41.41 | 38.17 / 38.17 | 12.67 → **0.00** | inside the band; the band vanished |
| `felix_it` | **84.72 / 83.28** | 81.30 / 82.29 | 1.44 → 0.99 | ON entirely BELOW OFF, −1.0 to −3.4 pts |
| `fwdllm_it_unaware` | 39.70 (1 leg) | **47.43 / 47.43** | — → **0.00** | +7.7 pts over the one OFF leg; band vanished |
| `fedbuff_it_oracular` | none at 7200s | 80.01 / 82.87 | — → 2.86 | ON only |
| `fedbuff_it_unaware` | none at 7200s | 81.49 / 81.14 | — → 0.34 | ON only |

Five of six comparable bands match or beat their OFF band; `felix_it` is the flag's one measured accuracy
cost, and it does not generalize by aggregation rate (`felix_round` is the same rate and improved). Open
question in §B.3. Both `fwdllm`-family baselines land at a band of exactly 0.00 ON — the same collapse, and
they are the two whose cadence is barrier-pinned (§A.2).

### §A.4  Legacy OFF board — reference only

**Never compare a row here to §A.2.** One baseline still has no ON evidence at all; every other OFF row is
superseded and lives in `git log`.

| baseline | run pair | dur | pass/fail/skip | N | notes |
|---|---|---|---|---|---|
| `fwdllm_it_oracular` | `20260802_193900`/`_214059` | 7200s | **69/0/23** | 39 | clean, unreplicated, OFF — the only row this baseline has |

⚠ **The cadence/convergence cells on an uncapped OFF row are not evidence about sim.** Swapping only which
real replicate a row was graded against — no code change — moved `felix_round` 71/3 → 65/9 and `fedbuff_round`
72/2 → 72/3 with *different* fails (§E). Read an OFF row for INV/un-windowed rungs only.

---

## §B  Next steps

### §B.0  Exit criteria — when parity is DONE

> Parity's job is that **sim does not change the CONCLUSION**, not that every rung is green. Every claim in
> this work is comparative, so a residual identical on every baseline cancels out of a ranking. Grade against
> this list, never against the pass count.

| # | criterion | status |
|---|---|---|
| 1 | Every INV/EXACT rung green on all nine | **FAILS on 4 of 8** — `cohort_sequence` on the four unpinned baselines, plus `throughput`/`terminal`/`commits` on `fedbuff_it_oracular`. All are roll-ups of the one `v1` residual |
| 2 | Convergence + terminal state inside each baseline's own replicate band | **PARTIAL** — `conv` green on 5 of 8, `terminal`/`commits` on 7 of 8. The three `fedbuff_*` rows fail `conv`; `fedbuff_it_oracular` fails `terminal`/`commits` too |
| 3 | Every remaining DIST residual is COMMON-MODE (same sign, spread smaller than the claimed effect) | **MET on 7 of 8 once graded against each row's own control** — same sign everywhere, and every residual but one is at or within 1.7x its real↔real floor |
| 4 | No residual correlates with a baseline-DISTINGUISHING knob (agg rate, selector, iteration cap) | **NOT SETTLED** — the raw residual tracks whether a barrier/cap pins the cadence, but so does the replicate FLOOR, by the same factor. The knob predicts the noise, not a bias |

One genuine residual is left — `fedbuff_it_oracular` at 5.1x its own control (§B.2). Criterion 4 cannot be
called until the floors are re-measured on the rung's own window (§A.3), because today the correlation it
tests for is visible in the control as well as the test. If all four hold → ship, record the residual as a
known bias, go run the real experiments.

**Tells that this has tipped into over-optimization:** chasing a rung whose residual is inside the replicate
band · the board getting worse from measurement changes rather than better from fixes · adding instrumentation
faster than closing bugs · a session ending with more red rungs and no code fix. All four at once means stop.

**Coverage is now nearly done — only `fwdllm_it_oracular` lacks an ON row.** Past that point polish is the
work, and criterion 4 is what it is for.

### §B.1  The pipeline — one procedure, run it per baseline

Stages are §A.1's. The `&&`-chain exists because the profile feeds the run feeds the grade; do not split it.

```bash
cd lib/python/examples/fwdllm/expt_scripts

# R2 — two ON real legs
bash run_sequential.sh --mode real --max-runtime-s 7200 --only <b> --yes

# FL — from runs already on disk; no sim leg needed
python replicate_floor.py --mode real --baselines <b> --profile-out ../parity_floors

# CH + SIM + GR — one chain, never split
cp ../sim_charge_profiles/<b>.yaml ../sim_charge_profiles/<b>.yaml.off-bak && \
python profile_sim_charges.py $(ls -d ../experiments/*_<b>_n100_*_real | sort | tail -2 | sed 's/^/--real-run /') \
    --out ../sim_charge_profiles/<b>.yaml --only-observed && \
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only <b> --yes && \
python run_parity.py --yes --baselines <b>
```

**Guards — both mechanical, neither needs remembering:**
- The launch preflight **BLOCKS** a sim leg whose charge profile predates any real of that baseline trained
  under the SAME `jvp_eval_mode`. A deliberate flag-OFF control landing later does NOT stale the profile.
  `--force` overrides; if you reach for it, write down why.
- `run_parity.py` pairs the sim leg with the latest real **whose flag matches**, and prints any newer real it
  skipped. Grading an ON sim against an OFF real reports the flag, not the code.

**Still on you:**
- ⚠ **Glob precision.** `*fwdllm*_real` also matches `fwdllm_it_unaware`/`_oracular`;
  `*_<baseline>_n100_*_real` is the safe form. Check what the `ls` resolves to before launching.
- ⚠ **`--dry-run` first, always.** It generates the configs and runs the full preflight without launching;
  a 30-second check against a 2h leg.
- ⚠ **Keep a baseline's replicate PAIR on ONE node** unless the nodes are known identical — a cross-node pair
  puts a hardware term in the floor that §D-44's span grouping only partly catches.

**Cost:** a 7200s REAL leg = ~2h05 wall (6940s achieved + ~400s startup/teardown); a 7200s-VCLOCK SIM leg =
~35-50 min, except `fwdllm`-family legs at ~10-15 min (few aggregations). A 6h node fits two real legs plus a
sim chain, or four-to-five sim legs.

### §B.2  In flight tonight, and where to resume

> **Update in place after every batch.** Delete a leg the instant its artifact exists and §A reflects it.

Three mechanisms are REFUTED (§E) and the real↔real control puts seven of eight rows at their own floor
(§A.2). Every verdict on this board now rests on a floor measured from ONE pair, which is n=1 for the
quantity it estimates — that is what tonight buys down.

**IN FLIGHT — 3 nodes, launched with the commands in `git log` on this file. Logs: `~/node{A,B,C}.log`.**

| node | legs | wall | buys |
|---|---|---|---|
| **A** | `fluxtune` real ×2 @ **14400s**, `--server-update-audit` | ~8h15 | is the per-bin spread flat 2h→4h (the transfer claim, §A.3) · `fluxtune`'s floor at the paper's 3.9h peak (EXPTS_CHARTER E1 error bar) · I-1 evidence through the round boundary |
| **B** | 3rd real leg, audit OFF: `fedbuff_it_oracular`, `fedbuff_round`, `felix_it`, `fedbuff_it_unaware` | ~8h20 | n=3 spread on the four baselines whose verdict hinges on it |
| **C** | `fwdllm_it_oracular` real ×2 → FL → CH → sim → parity, then `felix_round` 3rd real leg | ~6h40 | the 9th baseline's ON row · a 3rd leg on a row that reads NULL today, as a negative control for the n=3 estimator |

Node B deliberately runs NO `replicate_floor.py`/`profile_sim_charges.py`: re-profiling charges would
invalidate the pairing of every sim leg already on disk (§D-50). Floors get recomputed by hand in step 1.

**RESUME HERE, in order.**

1. **Recompute floors from n=3** — `replicate_floor.py --mode real --baselines <b> --profile-out
   ../parity_floors` for B's and C's baselines. `_spread` is max-pairwise, so a 3rd leg can only widen it.
   Then re-grade: `run_parity.py --yes --baselines <the 8>`. Rows are expected to flip green with no runtime
   change; **any row that does not flip is the real finding.**
2. **Fix `replicate_floor.metrics()` to grade the rung's window** — code, no run. It measures each leg's
   FULL run while `v1` grades the matched prefix, so every floor is 1.0-2.6x too small (§A.3). Compute over
   the matched prefix exactly as `v1` does. `fwdllm`/`fluxtune` must stay identical between the two
   estimators — that is the built-in negative control. Ship with a test pinning both on a barrier baseline.
3. **Read node A** — `plot_server_update.py <run_dir>` for I-1, and the per-quintile spread at 4h against
   the 2h numbers in §A.3. Flat ⇒ transfer to the 10h+ experiment runs is justified; growing ⇒ read the
   shape and extrapolate rather than assume.
4. **`fedbuff_it_oracular`, the one genuine residual** — +13.8% `v1` against a 2.70% control, and the only
   row where `throughput`/`terminal_state`/`total_commits` fail. It is also the only oracular baseline with
   an ON row, so `trackTrainerAvail` is untested against a sibling. **No run needed**: grade its two real
   legs against each other on every rung. **FALSIFIED IF** the real↔real grade fails the same rungs — then
   the baseline is noisier, not the mode.
5. **The sign-off re-grade** with every tolerance recalibrated against the ON floors (§B.4).

Then the experiment runs (`paper_expts_fluxtune/EXPERIMENTS.md`), sim-only, on fresh floors.

⚠ **Do not re-run the refuted three (§E).** The tie-break, the redispatch stagger and cohort-composition
bias each cost one telemetry pass to kill; none needs a run, none re-opens without a NEW signal.

⚠ **`run_parity.py` exits 1 whenever any rung fails** — the normal outcome on every current row (§D-51).
Never put it mid-`&&`-chain without `|| true`. The `$(ls ...)` inside a chain expands when its own command
runs, not at paste time, so it correctly sees the reals the earlier legs produced.

⚠ **Moving run dirs off a node does NOT bring its `parity_floors/` and `sim_charge_profiles/` with them.**
Both are re-derivable from the reals — `replicate_floor.py --profile-out`, then `profile_sim_charges.py`
in place (an existing entry keeps its `charge:` flag; a fresh `--out` path writes `charge: false` and would
silently disable the charge). Do that before grading, or every DIST rung grades at nominal tolerance.

### §B.3  Open questions — each with its falsifier

State the prediction BEFORE the run; a hypothesis that can only be confirmed is not one (§D-9).

- **`fedbuff_it_oracular`'s residual is oracular availability tracking, not the async loop.** It is the only
  row outside its own control (5.1x) and the only one carrying `trackTrainerAvail`. **FALSIFIED IF** its
  real↔real grade fails the same rungs — then the baseline is just noisier (§B.2, no run needed).
- **`felix_it`'s ON accuracy drop is baseline-specific.** Its ON band sits 1.0-3.4 pts below its OFF band
  while `felix_round` — same aggregation rate — improved. **FALSIFIED IF** a second loss-derived baseline
  degrades ON. Low priority: n=2 per side, seven other bands support the flag. Do not re-open the shared-mask
  JVP on one baseline.

### §B.4  Tolerances and rung gaps

- **Recalibrate every DIST tolerance against the ON floors (§A.3).** `3x floor` is still too generous where
  the floor is itself large: `fedbuff_it_unaware`'s `v1` floor of 4.9% buys a 14.8% gate, so its +10.3%
  residual — the second worst on the board — PASSES while `felix_it` fails at +9.0% against 6.8%. The gate is
  ranking floors, not residuals. `v2`'s 2% is still under two baselines' floors. Sign-off re-grade.
- **`cohort_sequence.count` and `v1` grade ONE residual against two gates, and count's is the right one.**
  `count_tol` is floor-gated on the `cycles` floor (0.0-0.8% everywhere, so it clamps to the 2% `min_abs`);
  `v1`'s is gated on `iters_per_bin` (0.3-4.9%, so up to 14.8%). Same +6.9% on `fedbuff_round` therefore
  fails count and passes `v1` by 5x. `cycles` is a whole-run count with a sub-1% replicate spread and `v1` is
  a ratio of two noisy quantities, so the sensitive gate is the trustworthy one — `v1`'s nominal 15% is the
  number to re-derive, not count's 2%.
- **`v1c` is a bin short of the power to see the drift it exists to catch.** It fits 10 bins;
  `fedbuff_it_oracular` reaches t = 3.11 against t_crit 3.36 and reads "flat" while its last quintile runs
  +31%. Raise `n_bins` or drop α — but re-derive against a real↔real pair first (§E: its real↔real
  calibration is already falsified on uncapped baselines).
- **`convergence`'s 5% `acc_tol` has no floor behind it.** It was ungradeable OFF (below the real↔real
  accuracy gap) and now fails four rows (`fedbuff_round` 9.3%, `fedbuff_it_unaware` 7.2%,
  `fedbuff_it_oracular` 6.8%). ON accuracy bands are 0.00-5.25 pts — re-derive it against those.
- KS-only rungs unguarded against a level shift (same class as the `selection_bias` repair, §G; all clean on
  live data): `dk1_agg_goal_trajectory`, `dk2_dynamic_c`, `dk3_eligible_ends_metric`, `eligible_speed`,
  `v3_cached_v_pool`. `selector_score` is DIAG.
- Thin ABSOLUTE budgets: the `fwdllm` family grades N=40-41 against 183-254 elsewhere, at 93-95% coverage.
  Both rows are 0-fail, so nothing is masked today, but coverage percentage cannot catch thinness; an
  absolute-N floor is proposed, threshold not chosen.
- `cohort_sequence.count` is rolled-up V1 — never chase it separately. It does NOT fail only when `v1` does:
  the two carry different floor-gated tolerances (above), so count fires first.
- `step_timing_breakdown` / `agg_step_timing_breakdown` are REPORT-ONLY wherever the sim clock discards the
  span (§D-31) — the designed steady state. Residuals there are §D-1 contention that never reaches the vclock;
  watch `charge_coverage` instead.
- Do not re-tune `redispatch_turnaround` (§D-14, §E).
- `felix_round`'s lap-boundary fails (`preferred_duration`, `terminal_state`) are underpowered artifacts at
  any duration whose graded window straddles a lap boundary. Both clear on the ON row.

### §B.5  Known-and-deliberate

- Real's over-`c` slot read is drain lag, not a §D-27 conflation (§E, §G). Telemetry only. Low priority.
- Real publishes no `_agg_slot_holders_ref`, so `_cap_dispatch_to_concurrency` falls back to the IDENTITY set.
  Harmless today; publishing one would let real dispatch into slots it now withholds — owed its own A/B.
- Sim's in-flight bookkeeping is split across six sets and should be one per-end state machine. The slot⇄guard
  split did the CAPACITY half; IDENTITY is still ad-hoc. Never bundle with a correctness fix.
- Base `asyncfl/top_aggregator._sim_hold_busy_slots` deliberately untouched — no `_sim_committed` term, so it
  never had the conflation. Re-check if async_cifar10 shows the same under-fill.

### §B.6  Backlog

- **A run dir cannot say which training config produced it.** `jvp_eval_mode` lives in the trainer's
  `config_overrides`, which the runner never dumps — the only record is 100 lines in the trainer log.
  `replicate_floor.py` greps that log, which works but is the wrong layer. Dump the resolved TRAINER config
  into the run dir and key off it. Same class as §F-18, and it is what makes the CH stage easy to skip.
  Same root, second symptom: a run's floor and charge profile live outside the run dir, so moving run dirs
  between nodes strands them (§B.2). Both should be run artifacts.
- **Enforce §F-18 mechanically: a per-baseline knob CONTRACT.** Two correctness-path knobs went missing from
  yamls and were caught only days later by reading telemetry. The mechanism mostly EXISTS —
  `run_sequential.sh`'s preflight `checks[]` blocks a launch and ships both patterns; the gap is that
  `condition_fp` hashes only CLI-patched knobs, so yaml-only knobs are invisible. Three homes, none new:
  `test_baseline_readiness.py` → preflight `checks[]` → parity `--validate`. **The hard part is "missing" vs
  "legitimately N/A"** (`sim_charge_profile_path` is sim-only, `reselect_cadence: round` round-only,
  `trackTrainerAvail` oracular-only), so applicability must be DECLARED, not diffed — sketch: a
  `knob_contract` block in `_metadata/baselines.yaml` read by all three layers. Open: where it lives;
  error-vs-warn; whether `condition_fp` absorbs it; who declares a new knob.
- **The preflight's python lives in a `run_sequential.sh` heredoc, so its checks cannot be unit-tested.**
  The CH gate is negative-controlled on live data, which is weaker than a test that survives a refactor.
  Extracting the heredoc into an importable module is the fix; `lib/python/tests/mode/test_baseline_readiness.py`
  is the natural home.
- Flag promotion: `sim_sct_ordered_drain` + `sim_model_dispatch_queue` are fluxtune-yaml-only but model
  general async-transport artifacts — smoke fwdllm/fwdllm_plus with both ON, confirm inert-or-better, promote.
- Checker invariants I1-I6 were drafted in a prior session and never committed anywhere (unrecoverable).
  Needs operator input on intended semantics before drafting fresh ones.
- felix (async_cifar10) may share fluxtune's round-1 cold-start gap — unverified, out of scope
  (`async_cifar10/PARITY.md` owns felix). felix 46/46 reconfirmation gates Phase 2.
- Momentum (S1-S3) / server-optimizer — roadmap, not parity. NOTE: S1's damping should also shrink the
  replicate floor (EXPTS_CHARTER I-1) — re-measure after it lands.
- P3/infra: no automatic GPU skip-and-remap on a broken ordinal (manual `execution.gpu_ids` exclude works).

**Standing rules.** A real↔real floor needs NO sim leg — halve the cost. One mechanism per run when a fix
could perturb another baseline. Never spend a run on a question a bench repro can answer (preamble).


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

**Run-length budget — SHORT BY DEFAULT** (`--max-runtime-s`): 900-1800s for verification, 3600s+ only for
scoreboard re-grades and the duration-gated rungs below. **Duration follows the residual's SHAPE, never
habit** (§D-25): a per-cycle divergence is fully present in cycle 1 and grades at full strength on a short
run; an accumulating one is under-reported by construction and a short run reads a FALSE PASS. The clock
family fires hard at 30 min; the cadence family does not, and reading it green there has burned a batch.

| validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | 5-10 min | a few hundred commits populate any per-commit field |
| **regression smoke after a shared-path change** | **900s** | INV tripwires + occupancy rungs are un-windowed, so they grade at any duration (below) |
| clock/pipelining family (`overlap_factor` K4, `throughput`, `per_round_advance`, `overhead_residual`) | **900-1800s** | per-cycle mechanisms — fire at full magnitude immediately (evidence above) |
| one MECHANISM rung (`drain_wall_budget`, `selection_detail`, `eligibility`) | **1800s** | the mechanism fires; per-commit dists stabilize. ⚠ a BOUNDARY-gated defect overrides this — the run must be long enough to REACH the boundary (`felix_round`'s round-1→2 is at wall ~4300-4800s) |
| variance-cadence LEVELS (`V1`/`V2`/`V2b`/`V5`) | 3600s+ | the divergence ACCUMULATES; a short run reads a false PASS |
| variance-cadence RATE (`V1c`) | 1800s+ / N≥40 bins | duration-invariant by construction (§D-35), but its t-test needs bins; SKIPs below 4 |
| stochastic identity / participation (`cohort_sequence`, `S2`) | 3600s+ | index overlap must reach its independent-draw floor to read as identity-not-bias (§D-2) |
| convergence sign-off (`terminal_state`, `conv`, `conv_loss`) | full 2h+ | terminal-state + curve parity only |

**WINDOWED vs UN-WINDOWED decides whether a short run is readable — read the field, don't assume.** A rung
carrying `matched_logical_budget_n` grades only the work both sides did, so a short run gives it a thin N and
a weak verdict (`fwdllm` is unreadable at any short duration). A rung without it pools the whole run and
grades at full strength immediately: every INV tripwire, `slot_utilization`, `throughput`.

**The floor grows as runs shorten, so re-measure it per baseline when adopting a new run length** (§D-24) —
two same-config legs + `replicate_floor.py --mode real`. Smoke (5-10 min) before any long run; one mechanism
per run when a fix could perturb another baseline.

**pytest** (`setup.cfg` sets `addopts = -n auto`, needs `pytest-xdist`; `-o addopts=""` runs serially if an
env lacks it):
```bash
conda run -n dg_flame python -m pytest lib/python/tests lib/python/examples/fwdllm/expt_scripts -q
```

---

## §D  Durable lessons — fwdllm diagnostic patterns

> Transferable invariants — patterns that must be followed. ≤30 words each; update in place, never append
> near-duplicates. A falsified hypothesis belongs in §E, not here. Gaps in the numbering are lessons merged
> into a neighbour; numbers are cited elsewhere, so they are never reused. Shared (non-fwdllm) patterns live
> in [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) "Durable lessons."

**D-1.** A shared-compute wall rung failing with byte-identical inputs is co-location contention, not sim
over-compute. Charge the vclock; never tune sim's compute.

**D-2.** A boundary-race cascade on a stochastic-async selector is core-IDENTITY, not skew. Gate
index-identity to diagnostic; keep counts and marginals enforced.

**D-3.** Porting a selector does not port its real↔sim timing parity — aggregator and trainer classes are
separate. Diff the destination against the shared base first.

**D-4.** Grade parity on the logical work budget, never a matched virtual-time window — normalizing along
the axis under test is circular.

**D-5.** A green parity rung is not a correctness claim; common-mode bugs pass differential tests. Pair it
with an absolute, mode-independent sanity check.

**D-6.** A selector's parity record belongs to the selector+aggregator PAIR. Check which side owns a guard
before citing it as a reference.

**D-7.** A statistic over rate-scaled samples measures the rate, not the samples. Check both sides' input
to a shared formula before suspecting the formula.

**D-8.** A cache/reuse fast path can silently skip a guard the slow path enforces. Diff its return against
what the bypassed call would have filtered.

**D-9.** A divergence names two disagreeing sides, never which is wrong. Find each side's own
self-consistency signal before choosing which to change.

**D-11.** Factor a per-round residual into per-cycle cost × cycles-per-progress-unit before naming a
mechanism: a small per-cycle gap explains a large per-round one, and a shared symptom can have disjoint roots.

**D-12.** A span measured from a shared batch-start timestamp is cumulative across the batch. Pool by
first-difference between sorted events, never a flat mean.

**D-13.** A run truncated by a deadline can emit an event for progress it never finished. Require
verified-completion evidence per progress key, not event presence.

**D-14.** After fixing a proven overcharge, re-verify the residual's sign and mechanism. A flipped sign
means re-decompose, not re-tune.

**D-15.** Sim faster than real with no charge category to explain it means a concurrency/scheduling policy
divergence. Measure per-trainer idle time, not server wall.

**D-15a.** Anchor "what was sent when" on the arrival timestamp; any field read at cycle close mis-dates a
mid-cycle send.

**D-17.** An event-count rung is meaningful only over matched WORK. Compare a divergent event's progress
key against the matched budget, not wall time.

**D-18.** A charge ledger prices the CHARGE, not the residual it removes. Confirm the clock is actually
charge-limited before predicting a throughput move from a charge delta.

**D-20.** Grade an invariant on its own quantity and axis, never a proxy sampled at the wrong instant. Look
for an existing per-entity span before instrumenting a new one.

**D-21.** A vclock charge is not neutral accounting: if sim selects work against the clock, changing a
charge shifts sim's selections and cadence too.

**D-22.** A family of rungs reporting one quantity five ways cannot localize it. Split the residual against
an independent per-cycle measurement before naming a mechanism.

**D-24.** A tolerance is meaningful only above the pipeline's own same-seed replicate spread. Measure that
floor from runs already on disk; below it, no code change can pass.

**D-25.** Run length follows the residual's SHAPE, never habit: a per-cycle mechanism grades at full
strength on a short run, an accumulating one reads a false PASS there (§C table).

**D-26.** A progress key must be monotone in TIME before anything sorts, maxes or windows on it. Order by
event timestamp; a composite key can wrap out of order.

**D-27.** One set serving two roles hides a bug until a rule changes one of them. When an invariant reads
"X but not Y", check whether the code has two sets or one — in BOTH modes, since dispatch order can hide it
on one side.

**D-28.** A shared-path fix validated on the baselines it targeted must be re-graded on ALL of them at
scoreboard length; siblings regress silently on rungs the validation set never exercised.

**D-30.** Anything gated on "is the background worker free" is a wall-clock race sim loses — sim compresses
the gap between events while the work costs the same wall. Gate on a progress INDEX.

**D-31.** Before grading a sim quantity, check the clock actually consumed it. Where sim folds a profiled
constant, its own span is a discarded contention artifact and comparing it is a guaranteed false fail.

**D-32.** A statistic that ranks exempted entries or averages a bimodal burst will be read as the cause,
while reporting how many small events fired. Rank only what gates; report the raw quantity under its own name.

**D-33.** Real clears its bookkeeping when it PROCESSES a message, not when the message lands. Capacity read
off that state charges the aggregator's own drain lag to the trainers.

**D-34.** On an uncapped baseline, iterations-per-bin ≈ the bin's FIRST variance ÷ the gate threshold. So a
cadence divergence is not a cadence bug — reduce it to that one number and grade that.

**D-35.** A quantity inside a feedback loop has a residual that grows with run length, so no fixed tolerance
on its LEVEL holds at two durations. Gate the per-unit RATE against zero and read its t-stat — a rate sits in
an exponent, so a small rate gap buys a large count gap.

**D-36.** A profiled constant shared across baselines is a hand-typed constant wearing a script's clothes.
Profile per baseline, from the paired real leg, and gate provenance at launch.

**D-38.** Pair per-bin samples at equal bin index before differencing: it cancels the shared training curve
and gives an honest within-run error bar. Only a replicate sees seed-level variance.

**D-40.** Before widening or re-windowing a failing rung, check whether a SIBLING baseline passes it. If one
passes and one fails on the same code path, the rung works and the fix would mask a defect.

**D-41.** Seeding fixes the stream you seeded, not arithmetic and not a stream some component draws from
privately. If the stream POSITION matches across runs, the divergence is below that layer.

**D-42.** A discrete choice over near-tied continuous values (argmax, sort, a threshold gate) turns
round-off into an O(1) difference. Look for one before accepting a divergence as irreducible.

**D-43.** Grade a finite-difference estimator by its condition number, not its formula — reduced precision
amplifies round-off by a factor you must measure, not assume small.

**D-44.** Group or truncate legs on the ACHIEVED span, never the configured duration or a fixed cutoff. A
truncated run reports the duration it asked for; a fixed cutoff silently hands the longer leg more time.

**D-45.** A replicate floor must come from legs that differ ONLY in wall-clock luck. Swapping which replicate
a fixed comparison is graded against is the cleanest test of whether a rung measures anything.

**D-46.** Validate a proposed fix on a bench repro that imports the real code path, never on an FL run first.
Confirm the CONTROL reproduces before crediting any fix (preamble).

**D-47.** A bench repro must build the object in the MODE production runs it. An inference-mode, fixed-batch
or regularizer-free "clean measurement" is a different system and returns bit-exact nulls. A container's mode
flag is not proof — count the live leaf modules, and record the mode alongside the result.

**D-49.** Before an overnight, spend 15 minutes proving the knob reached the TRAINER: preflight dry-run, then
one short leg grepped for the knob's own log line, one per trainer. A config-file-only knob is invisible to
the launch fingerprint (§F-18), so the run looks perfect and grades the old config.

**D-50.** A profiled constant outlives the config it was measured under. When a training knob changes,
re-derive the profile from the new paired real BEFORE reading any residual — a stale charge moves sim's own
cadence (§D-21), so the run grades the profile, not the code, and the residual can reverse sign per baseline.

**D-51.** A tool that exits nonzero to report FINDINGS, not failure, silently truncates any `&&` chain it sits
in. Check a grader's exit-code contract before scripting it; `|| true` the ones that report.

**D-52.** A pin (cap, barrier) suppresses the residual AND the replicate floor together. Before crediting the
green or calling it masking, read the pinned baseline's own real↔real spread — it decides which it is.

**D-53.** Measure a floor on the SAME window and axis the rung grades. A run-level mean averages away the
per-bin drift the rung sees, and understates the floor most where the residual is largest.

**D-54.** One divergent cohort is permanent: the weight trajectories separate and no later step is comparable
again. Grade cross-mode bit-identity of a per-end quantity to find WHERE parity is lost, not whether.

**D-55.** Before naming a mechanism for a residual, run the same comparison real↔real. A control carrying the
residual's own shape and magnitude means there is nothing to fix, whatever the mechanism story predicts.

---

## §E  Dead ends — do NOT retry

> One line each, append-only. A dead end never un-dies; re-listing one wastes a session. Numbers only where
> the number IS the lesson. Landed-but-inert cleanups belong in §G.

**Replicate floor / gradient noise**
- **Reduced-precision or GPU-kernel nondeterminism as the floor's SOURCE** — FALSIFIED on the bench: the real
  model stack, co-located processes on one GPU, every loss and jvp bit-identical within and across processes.
  The AMPLIFIER half of that hypothesis stands (§G); the source is live dropout inside the finite difference.
- **Every input-side audit of the spread** — trainer data hashes, dispatch order and rank, first-task
  iteration/model_version, seeding, client index, data partition, cohort choice — REFUTED by direct
  measurement, and all clean for one reason: the divergence enters BELOW them (§D-41).
- **Forcing inference mode in a probe** — what made the first probe runs read bit-exact on every arm. Not a
  clean measurement, a different model (§D-47).
- **"monotone drift shape means sim"** — FALSIFIED: a real↔real pair climbs harder than any real↔sim slope on
  that baseline. A diverging rate verdict on an uncapped baseline is not evidence about sim.
- **Any cadence/convergence verdict read off a SINGLE real leg on an uncapped baseline** — dead; the same sim
  leg's score swings on which real it is graded against. A verdict needs a replicate PAIR (§D-45).

**Charges**
- **Redispatch turnaround as a large cost sim omits** — FALSIFIED. Its first-difference marginal (§D-12)
  assumes ONE serial dispatch burst; under event-driven dispatch it prices inter-arrival WAITING and the
  direct cost is real≈sim. Don't re-derive it from timestamp differences.
- **A charge delta passing through to throughput at a fixed coefficient** — malformed: per-round sim time
  divides by an INTEGER round count, so small deltas sit under one quantum, and a charge also moves cadence
  (§D-21). Sizing a throughput prediction off a charge delta was refuted on its own validation leg (§D-18).
- **GC pause in the variance path** · **grad-norm as the drain-wall amplifier** · **a real-side distribute
  settle term** — all REFUTED as parity causes.

**Cadence / trajectory**
- **The agg-goal admission TIE-BREAK as the cohort-divergence root** — FALSIFIED on telemetry. Only 0.1% of
  2628 divergent `fedbuff_round` cycles swap ends of equal modeled D; the first divergence swaps three ends at
  D=4.294 for three at D=9.202. Sim already breaks exact `sct` ties by end id
  (`SimReorderBuffer.pop_min`), so that half was never broken. Speed-class ties are real (51 ends, 18 classes,
  largest holds 17) but they are not what moves membership.
- **Redispatch STAGGER as the divergence root** — FALSIFIED: the per-cycle spread of `task_recv` after a
  cohort closes is p50 8.5 ms in real and 63 ms in SIM, i.e. sim is the more staggered side.
- **Cohort COMPOSITION bias (sim favouring fast ends)** — FALSIFIED: mean modeled D matches within 0.2%, mean
  staleness within 0.02, distinct-ends-per-bin within 7%, and `participation` (S2) is green on all eight.
- **A grad-pool accumulation bug behind `felix_round`'s cadence family** — FALSIFIED. Re-profiling the charges
  from the baseline's own ON reals closed `v1`/`v1b`/`cohort_sequence` with ZERO code change, and the residual
  reverses sign on baselines still carrying stale charges. Every summary statistic matching while the output
  moved was the tell that the divergent input was the CLOCK (§D-50). Do not re-open `calculate_var` or the
  pool-assembly path on cadence evidence alone.
- **`felix_it`'s 84.72% OFF leg as a lucky draw** — FALSIFIED; the replicate landed at 83.28, so the ON band
  really is below the OFF band on that one baseline (§B.3).
- **The round-baseline divergence as a variance-at-iteration-0 trajectory effect** — DEAD; a per-cycle charge
  correction closed the whole family.
- **Cadence as a progressive trajectory divergence** — FALSIFIED; the drift rung flipped to a level offset and
  the sign reversed (§D-14). The residual is a level offset with a flat rate inside the replicate floor:
  calibrate the tolerance, don't hunt a mechanism (§D-24).
- **One charge-coupled cadence level shared by both round baselines** — DEAD as stated; the repaired overlap
  rung clears both and the drift rung splits them.
- **Per-cycle committed-set overlap as the trajectory discriminator** — FALSIFIED: it does not separate the
  baselines; the least set agreement came with the best trajectory agreement. The discriminator is the drift
  RATE (§D-35).
- **Cohort COMPOSITION as the variance-trajectory driver** — REFUTED; both modes commit from the same frozen
  cohort with the gap fully present.
- **Two baselines sharing one overshoot root** — REFUTED, disjoint factors (§D-11).
- **Reselect-cadence pool size as the round-cadence throughput driver** — SUPERSEDED by per-cycle charge
  compounding (§D-11).
- **"Surplus idle" as the residual after the scheduling fix** — REFUTED; the occupancy identity matches.
- **Eval-thread GPU contention reaching commit ORDER** — FALSIFIED; sim orders on a deterministic
  modeled-delay grid, so contention cannot move it.

**Slots / selection**
- **The selection fail as a windowing/thin-N artifact** — FALSIFIED, and re-windowing would have MASKED a real
  over-dispatch defect (§D-40).
- **Real's over-cap slot read as a half-fixed §D-27 conflation** — WRONG diagnosis; it is drain lag (§D-33).
  The two halves were correct; don't re-open them.
- **Selection bias on the round baselines as two separate selector faults** — FALSIFIED; both flipped green on
  the slot⇄guard split with zero selector change. Opposite signs against a shared input do not imply separate
  roots when both read the same availability bookkeeping.
- **Admitting by lowest modeled completion time instead of first-arrived** — REJECTED: FIFO-violating,
  deadlocks under unavailability, and the divergence is a stochastic tie-break (§D-2).
- **The round-cadence cohort pin as a defect** — operator ruling: it pins by design. Don't re-key it (§D-17).
- **A timing rung's gating function "moving"** — it never gated; the rung ranked exempted entries. Fixed at
  the source (§D-32).

**Measurement constructs**
- **A matched budget defined as min(vclock, wall)** — DELETED, don't reintroduce: it conflates the two clocks
  the comparison is testing and fails to grade (§D-4).
- **Real's wall-vs-vclock anchor as the residual's cause** — REFUTED; residuals survive the change of
  coordinate, so they are in the mechanism, not the measurement.
- **The receive-ordering rework as the fluxtune skew fix** — LANDED but INERT; cleanup, not a parity fix.
- **GPU contention blamed at n=10** — REFUTED below ~100 trainers; at n=100 it IS a root (§D-1). Scale-bounded.

## §F  Locked invariants (from async_cifar10, carried over)

> Always-true / always-do rules. Numbers are cited across this doc — keep them stable, don't renumber.
> Diagnostic *patterns* (see X → means Y) live in §D, not here.
>
> **APPEND-and-AMEND with OPERATOR APPROVAL, never silently** — state the evidence (code, telemetry, or a
> failing test), not an argument. **Amend in place, keep the number** (other sections cite them). Deleting
> needs more evidence than adding; prefer narrowing scope. A new invariant must be always-true in BOTH modes
> — one-baseline findings are §B/§G, see-X-means-Y patterns are §D, and if it needs a caveat it isn't one.
> A fix that contradicts an invariant is a STOP: resolve it with the operator before landing.

1. **Sim does real forward-grad compute, charges modeled time.** Agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`; the vclock is a monotone `max` over sct. Never put
   overhead on the vclock.
2. **Progress axis is `data_id`; identity/caching axis is `model_version`.** `data_id` wraps every lap —
   never key a cache or identity on it (§F-21). The lap counter equals `model_version` only in regular FL;
   re-derive the axis before carrying a round-keyed construct over.
3. **Variance is an emergent gate; localize, never tune it.** The variance threshold and the per-bin
   iteration cap are baseline-defining knobs, not parity levers.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the reorder
   buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** Check whether real is the divergent side before
   tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as real-transport artifact vs algorithmic
   property, and scope-check shared code first — the shared async aggregator can silently break
   async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only → its own mode tests; shared parity engine → add
   async_cifar10; shared stack → the full suite.
8. **Telemetry-first, then instrument, then (rarely) run.** Validate/refute from telemetry already on disk
   before running anything. Ship telemetry + plot + pytest together with any new mechanism.
9. **Consult PARITY.md's vclock rules before any sim-clock change.** Clock is a monotone `max`; sim skips
   real waits and reconstructs order from sct.
10. **Sim MUST produce speedup: `vclock/wall ≥ 1`.** Below 1, sim is stalling on a wait it should skip, or
    its commit throughput can't keep pace with arrivals.
11. **Correctness before speed; shared roots before per-baseline.** A bug failing rungs across ≥2
    baselines outranks a single-baseline one.
12. **Logical determinism is the parity definition.** Same trainers selected, same receipt order, same
    aggregations/rollbacks — differing ONLY in wall-clock. Prove it on the first data bin first.
13. **Do the right thing — no hacks.** A number moved without a correct mechanism is a regression in
    disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. No bare-scalar shortcut.
15. **Verify claims against code, not comments/docstrings.** A docstring claiming two functions are
    equivalent states intent, not a guarantee — diff them.
16. **Contention at scale → §D-1.** Refuted below ~100 trainers (§E); genuine root at n=100.
17. **A rotating cohort settling at `c − agg_goal` surplus is the correct steady state** for `c ≫ agg_goal`
    fedbuff — don't drive the carried surplus toward 0.
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** A correctness-path value
    (seed, delay floor, agg_goal, c, a flag) must match across config, snapshot and both roles' telemetry —
    divergent logging wastes sessions chasing phantoms.
19. **No compute on the critical path for a log the run doesn't need.** Gate any log with non-trivial args
    behind a level check — an f-string evaluates its args even when the level would drop the line.
20. **Real/sim timing disagreement → fix real toward determinism, never inject noise into sim.** Sim's
    per-speed-class duration must stay clean; fix real's measured completion time at the source.

### §F.1 Version & commit invariants (confirmed in code, both modes)

21. **`model_version` bumps once per COMPLETED data-bin** (variance PASS, at the bin advance) — constant
    across that bin's iterations. `iteration_per_data_id` bumps on a variance-FAIL retry and resets on bin
    advance, so `version_key` changes every iteration — the sole step identity (§F-14).
22. **Commit == the update used for aggregation, at that instant — no lag.** Real: on ordered arrival. Sim:
    when the vclock reaches the update's `sct` (buffer-unlock IS the commit). Never commit on a later event.
23. **Commit frees the compute slot immediately, but a re-pick guard keeps the trainer un-pickable for the
    SAME `version_key`** until it advances. TWO sets, never one (§D-27): CAPACITY is the slot-holder set —
    the only thing any cap may read — and IDENTITY is the pending-commit set. The slot must free at commit
    and never be re-added, or re-dispatch starves across variance-retry iterations; the guard holds to the
    agg-goal boundary (§D-15).
24. **Within a data-bin the global weights are constant; a re-picked trainer gets a RETRY, not a re-send.**
    Full WEIGHTS go out only for a `model_version` not yet received this bin; a same-version re-pick gets
    the retry signal, never a redundant weight re-send.
25. **One instruction per version_key: never dispatch to a trainer with an unresolved outstanding message
    for the CURRENT `version_key`.** Busy = silence, not a second message, until it returns or the key
    advances. Sync distribute always had this guard; async didn't, and the gap showed up as near-total
    in-flight overlap in round 1 (§G). Any new distribute call site must mark AND check.
26. **Reuse the existing construct; don't duplicate per baseline.** New per-trainer/version state almost
    certainly needs an EXISTING mechanism, not a parallel one — duplicate logic is duplicate bug surface.
    Tests too: extend a contract suite before writing a bespoke one. Tell: a new variable/method/test whose
    name rhymes with an existing one — check whether the existing one should take a parameter instead.
27. **One event, one instant: never mix a pre-mutation snapshot with a live read.** An event describing a
    cycle must snapshot EVERY identity field at the same point, before the branch that mutates them — one
    live field against a snapshotted one emitted a non-monotone key for a year of runs (§G). Corollary: the
    progress index must never be observable outside its valid range — keep the lap wrap adjacent to the
    increment, with no read between them.

### §F.2 Porting a SELECTOR ≠ porting TIMING parity

Moved to §D-3; stub kept because prior sessions cite "§F.2".

---

## §G  Landed fixes — recent + load-bearing ONLY. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, immediately.** The instant a rung flips or a hypothesis resolves, write ONE line
> (mechanism + outcome) and delete it from §A/§B in the same edit. Newest first. Delete an entry once nothing
> current depends on it — git log keeps it.

**This batch**
- **The real↔real CONTROL was run on all seven baselines and it collapsed the board.** Same per-bin
  comparison, realA↔realB, from legs already on disk: seven of eight rows sit at or within 1.7x their own
  floor, and the control carries the SAME late-run concentration (`felix_it` +15.3% in its last quintile
  real↔real). `fedbuff_it_oracular` alone is outside, at 5.1x. Three mechanism hypotheses died on telemetry
  first (§E) — the control is what should have run before any of them (§D-55).
- **The floor estimator grades the wrong window, and it understates by 1.0-2.6x.**
  `replicate_floor.metrics()` takes each leg's run-level mean; `v1` grades the matched prefix. A/B on the
  same legs with the shipped code imported as the control: identical on `fwdllm`/`fluxtune` (loop pinned,
  built-in negative control) and 1.4-2.6x larger everywhere the variance-cadence loop is free (§A.3, §D-53).
- **Cross-mode gradient bit-identity is the localization tool that made this readable.** The two barrier
  baselines are bit-identical for their whole run (`fwdllm` 420/420, `fwdllm_it_unaware` 410/410) and
  `fwdllm` reads `v2` +0.00% real↔real too — the stack is bit-reproducible, so the whole spread is the async
  admission loop amplifying wall-clock luck. Every `c > agg_goal` baseline matches for part of the first
  quintile, then reads exactly 0 forever (§D-54).
- **The 3-node batch landed: 8 of 9 baselines have a valid ON row, and the board collapsed to ONE residual.**
  Sim over-iterates per data-bin by +7-16% on the four baselines where nothing pins the cadence, and by
  0.00-0.04% on the four where a full-cohort barrier or an iteration cap does. Every remaining red rung —
  `cohort_sequence.count`, `v1b`, `selection_detail`'s selection count, `fedbuff_it_oracular`'s
  `throughput`/`terminal`/`commits`, `convergence` — is that number rolled up (§A.2). That answers exit
  criterion 4 in the negative and named the one investigation worth running.
- **The stale-charge hypothesis is CONFIRMED and closed.** Re-profiling took `fwdllm` from +18.0% to +0.00%
  and `fedbuff_round` from +12.8% to +7.4%, and the sign reversal that made criteria 3-4 unanswerable is gone
  — sim is now ≥ real on all eight rows. A stale charge was the whole residual on one baseline and half of it
  on the other (§D-50, §E).
- **`fedbuff_it_unaware`'s open λ question is retired — replaced by a better measurement.** With the
  replicate pair on disk its `v1c` λ is +0.21/100 units at t=2.81, verdict flat; but the per-quintile split
  shows +37.3% in the last fifth, so the rung's flat verdict is a power limit, not an answer. Grade the
  quintiles, not λ.
- **Two mechanical guards for the CH stage, both negative-controlled on live data.** (1) The launch preflight
  BLOCKS a sim leg whose charge profile predates any real of that baseline trained under the same
  `jvp_eval_mode` — it fired on all four stale profiles, then went 21/21 green after re-profiling, and does
  NOT fire on a deliberate flag-OFF control. `--force` overrides. (2) `run_parity.py` pairs the sim leg with
  the latest real whose FLAG matches, not the latest outright, and prints any newer real it skipped — the
  `felix_it` OFF-control trap is now structural rather than a note in this doc. 3 tests; the preflight half
  lives in a shell heredoc and has no importable test (§B.6).
- **The charge profile is a first-class parity stage, not a detail — it alone took `felix_round` 70/4 → 73/1.**
  Re-profiling from its own two ON reals, with no code change and the same real leg, took `v1` 5.4% → 0.0% and
  flipped `cohort_sequence`/`v1b` green. It killed the grad-pool hypothesis the cadence family had been
  charged to (§E) and became §A.1's CH stage and §D-50. `var_calc_audit` and `diff_var_pool.py` stay in the
  tree, OFF and now unused.
- **`felix_it` has an OFF band at last: 83.28 / 84.72.** Floors 2.3/0.1/2.2/2.4%, and the band sits 1.0-3.4
  pts ABOVE its ON band — the flag's only measured accuracy cost, and it does not follow the aggregation rate
  (§A.3, open in §B.3).
- **DIST tolerances are now sized from each baseline's measured replicate floor, not hand-typed (§D-24).**
  `replicate_floor.py --profile-out` writes `parity_floors/<baseline>.yaml`; the checker tightens toward
  `3x floor`, never past 2% absolute, never looser than nominal, and SKIPs a rung whose floor has swallowed
  its tolerance instead of returning a coin flip. Each gated result carries its floor, nominal and effective
  tolerance. **This caught a real defect immediately**: `v1` had been passing a 5.4% gap against a 15% gate
  calibrated when the floor was 13.3%; at the ON floor of 0.6% it fails. No floor file ⇒ nominal gates, so
  async_cifar10 is inert. 8 tests.
- **`jvp_eval_mode` PROMOTED to default ON, declared in all 22 baseline yamls, and confirmed as the replicate
  floor's DOMINANT term.** Live dropout inside the finite difference made the two JVP passes draw different
  masks. ON reals on five baselines: the worst cadence floor fell 13.3% → 0.6% (22x) and every measured
  variance floor fell with it (21.3% → 6.6% worst), at no accuracy cost against four of five OFF bands.
  **Verified rather than assumed** — a per-trainer census from inside the eval block reads 19 of 20 live
  dropout leaves before, 0 inside, while `model.training` reads False throughout, which is why the flag alone
  was never evidence (§D-47). Eval mode is applied before the model is made functional and held for the whole
  loop (the functional transform deep-copies the module), then every module's own flag is restored. A yaml
  contract test fails if any baseline drops the knob; the preflight warns on an explicit OFF. 24 tests.
  The proposed "forward-gradient training is not reproducible, widen the tolerances" invariant is RETIRED.
- **Only the loss-derived-weight baselines could lose a cadence floor, and only where one existed.** The
  integer-staleness baselines had none and did not move (0.0 → 0.3%, 3.9 → 4.1%); `felix_it` is loss-derived
  and also did not move, because its floor was already 2.2%. Cadence tracks the aggregation rate, accuracy
  tracks gradient noise — different floors, different causes.
- **`replicate_floor.py` splits groups by `jvp_eval_mode`, read back from the trainer log.** It pooled every
  same-duration leg of a baseline, so ON legs would have been averaged with OFF ones and the exit criterion
  would have measured the flag, not the floor (§D-45). The knob is in no config file in the run dir — a
  provenance gap tracked in §B.6. 4 tests.
- **Boundary over-dispatch FIXED, and validated live.** The release path cleared both re-pick guards before
  rebuilding the outstanding set from them, so top-up selections re-picked still-training ends. Sim now also
  folds in the dispatched-not-yet-returned record — the half real's equivalent always carried, which is why
  real never had the bug. The ON sim leg's round-2 boundary reads 30 unique picks / 0 re-picks against c=30.
  3 tests (2 fail without).
- **Watchdog now SCALES with the budget instead of being a constant.** The yamls' fixed 10800s was fine at
  7200s and silently truncated anything longer — a 4h leg would stop at 3h while reporting it asked for 4h,
  which is how a truncated leg poisons a floor (§D-44). `patch()` raises it to `budget + 1800s`, real mode
  only (sim's wall cap is `sim_wall_ceiling_s`); 2h legs are unchanged, so replicate pairs stay strict.
  Caught by `--dry-run`, not by a run. 6 tests, including that watchdog == budget is NOT safe.
- **`server_update` telemetry for I-1** — per-commit `||Δ||`, `||w||` and their ratio, accumulated at the
  CPU-sync site that already runs, behind `--server-update-audit` (default OFF: its wall cost perturbs async
  arrival order, §D-45). `plot_server_update.py` overlays the ratio on test accuracy with round boundaries
  marked. If I-1's undamped-optimizer root holds, the ratio climbs BEFORE accuracy falls; a flat ratio under
  a collapse falsifies it. 9 tests.
- **Two determinism knobs (fp32-outside-autocast, strict-determinism) exist, default OFF, byte-identical
  off.** Bench A/B across concurrent processes confirmed reduced precision AMPLIFIES the floor but is not its
  source; neither knob touches it. 23 tests. Don't re-run that bench (§E).
- **`replicate_floor.py` groups on ACHIEVED span, not the configured duration (§D-44).** A leg short of the
  group's longest by more than the span tolerance is DROPPED and named, even when that leaves <2 legs. One
  truncated leg had inflated a baseline's floor by ~3-6 points. 5 tests.
- **`felix_it` charge re-profiled from its own 7200s real** — built from a 1200s leg, it over-charged the
  drain tail 1.35× and the aggregation 1.29×, which WAS its wall-budget fail.

**Load-bearing machinery (older, still relied on)**
- **Per-baseline charge profiles + launch-time provenance gate (§D-36).** One family-wide drain constant was
  1.08-2.75× each baseline's own real cost. Nine profiles under `sim_charge_profiles/`, generated by
  `profile_sim_charges.py`; `--only-observed` stops a refresh carrying an op the baseline never ran, and the
  launcher BLOCKS a run whose charges came from another baseline's real. Redispatch turnaround is deliberately
  NOT per-baseline (§E); only the drain-tail and aggregation terms are genuine per-cycle span means. 4 tests.
- **`charge_coverage` [DIAG]** — per label: sim wall vs what reached the clock vs real's span. The standing
  audit that makes a mispriced span announce itself. 4 tests.
- **One primitive for "did the clock consume this?" (§D-31).** Sim's vclock advances only from `sct` and
  explicit overhead charges; a span outside both never reaches it, so grading sim's wall for it fails on host
  contention. Wall-budget rungs grade the CHARGED seconds when profiled; timing breakdowns gate only where sim's
  compute actually binds. An absent ledger reads UNKNOWN, never a silent demote. 6 tests.
- **Eval cadence made DETERMINISTIC — the `convergence` root, never duration.** The snapshot returned nothing
  while the background eval thread was busy, making WHICH commits get evaluated a wall-clock race sim loses
  structurally (real kept 99-100% of evals; sim 49-60% on seven of nine). Now gated on commit INDEX; a busy
  thread is waited out and warned, never dropped. Set in all 18 yamls (§F-18). 15 tests. §D-30.
- **Compute SLOT split from re-pick GUARD, BOTH modes (§F-23, §D-27).** One set served both roles, so the
  guard-hold also held the SLOT. There is now a single CAPACITY answer, published for the cap to read, with
  the pending-commit set left as IDENTITY only; both halves are flagged and default ON. Real's residual after
  this was a SECOND defect — drain lag, closed by reading queue depth instead (§D-33). 36 tests.
- **Matched-budget primitive on the progress axis (§D-26).** N is the position-wise common prefix of both
  chronological commit sequences, ordered by event timestamp so legacy telemetry grades correctly. A max-key
  ceiling never required both sides to commit the same bins. 5 tests.
- **Budget COVERAGE graded + stamped on all 8 windowed rungs** — a Stage-0 control, plus a per-result coverage
  figure and a low-coverage flag below 80%. Hard-fails below 50% or on a sequence divergence. 8 tests.
- **Iteration drift graded as a RATE (§D-35)** — fits log iteration ratio against progress and t-tests the
  slope, because the level is a function of run length (+1.1% @3600s → +19.4% @7200s on unchanged code). The
  cadence levels depend on it. ⚠ **Its real↔real calibration is FALSIFIED (§E)** — on an uncapped baseline a
  diverging verdict does not distinguish sim from noise. 6 tests.
- **`overlap_factor` repaired, DIAG→MECHANISM/EXACT** — per-cycle barrier span ÷ per-cycle clock advance over
  the matched budget; the only rung that localizes a throughput residual (§D-22). 4 tests.
- **`v2_var_trajectory` grades the matched budget on async too (§D-4)** — its truncation was gated on a clock
  coordinate that is None for every async baseline, so async graded the pooled run. 2 tests.
- **`selection_detail` reports the re-pick count directly (§D-32), DIAGNOSTIC.** Its mean averaged a bimodal
  burst — one full draw, then top-ups — so it reported how many top-ups fired, not the defect. Guarded on ≥2
  distinct selection rounds, since under event-driven reselection the round never advances and every legitimate
  re-pick would count. The finer in-flight predicate lives in the boundary trace script. 4 tests.
- **KS-only rungs gained mean guards** — selection-speed bias and grad norm gated on shape alone, blind to a
  level shift when both sides share that shape.
- **`slot_utilization` rung (Stage-4 MECHANISM/EXACT)** — time-weighted mean/median slots busy, where the cap
  rung graded only PEAK (identical peaks on both modes while the means were ~5 slots apart). §D-20.
- **`concurrency_cap` re-based onto peak DISTINCT in-flight ends** (§D-20) — the over-cap read was a phantom;
  same-end concurrent dispatches are graded separately (§F-25). 8 tests.
- **Lap-boundary identity snapshot fixed (§F-27, §D-26)** — a live round counter read against a pre-mutation
  bin index emitted a non-monotone key, and the bin index was transiently out of range when a census read it.
  Training state was always correct (staleness keys on `model_version`). 7 tests.
- **`async_oort.py` re-based onto `AsyncSelectorBase`** (2193→897 lines) — utility-scoring POLICY unchanged,
  routed through the shared choose path, with weighted sampling moved onto reproducible keys (the old
  pool/order-dependent draw was not reproducible). Live real↔sim confirmation landed with `felix_round`'s ON
  row — `selection`, `selection_detail`, `selection_bias` and `utility` all green; `felix_it` still owes one.
