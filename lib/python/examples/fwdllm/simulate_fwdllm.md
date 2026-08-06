# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 ported fedbuff/felix-lineage
baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3).
Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf, sim barrier redesign, delay-factor
calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/
gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — how to use this doc
>
> **Fresh session? Read in this order:** §B.0 (where this stands — the critical path is no longer parity) →
> §F (locked invariants) → §E (dead ends) → §A.2/§A.3 (the board and the control that reads it) → §B.3 (what
> to do next) → §D (only the lessons for the rung you're chasing) → §C for ladder mechanics.
>
> | section | contents | update rule |
> |---|---|---|
> | §A.1 | readiness ledger — the six stages per baseline | tick a stage the moment its artifact exists |
> | §A.2-5 | the board + its real↔real control, floors, duration transfer | rewrite in place on every >3600s run |
> | §B | SELF-CONTAINED RESUME POINT: status · runbook · the mechanism + what is ruled out · ordered queue · open questions · backlog | current state only; an item lives here XOR §G |
> | §C | ladder/decomposition method, run-length budget | edit only if the method itself changes |
> | §D | durable lessons — transferable invariants | ≤30 words each; update in place, never append near-dupes |
> | §E | dead ends — falsified hypotheses | one line each; never re-open |
> | §F | locked invariants — always-true / always-do | operator approval + evidence; amend in place, never renumber |
> | §G | closed items | move here the instant a §A/§B issue resolves; delete the source in the same edit |
>
> **No hypothesis numbering.** A question lives in §B.4 with its falsifier until it resolves, then it becomes
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
> - **Run the real↔real CONTROL before naming any mechanism** (§D-55): `run_parity.py --control`. It needs no
>   sim leg and no new run, and §A.3 still shows rungs that fail on config-identical real legs — a red cell is
>   not evidence until the control says it is. Where a rung fails both, calibrate it; do not hunt it.
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
> - **Analysis must stay INTERACTIVE — treat >2 min as a bug in the tool, not a cost of the data.** A leg's
>   aggregator log is ~1.5 GB, so anything that walks all nine baselines is IO-bound by construction: it must
>   parallelise across independent units (`--jobs`, one worker per baseline group) and reuse parsed telemetry
>   rather than re-parsing it. A serial nine-baseline floor sweep was >10 min per side and is now ~2.5 min.
>   New analysis scripts inherit this: parallel across groups, cached parse, stream output (never `| tail`,
>   which buffers a whole run and makes a working job look hung).
- Runs happen on a separate operator-controlled node: print the command, never launch or babysit one.
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
python run_parity.py --control --duration 7200 --yes    # the real↔real CONTROL over every pair of every
                                            # config's legs, + the per-rung fail rate (§A.3). No sim leg.
                                            # --control-mode sim|both · --jvp-eval-mode on|off|any
python replicate_floor.py --mode real --duration 7200   # replicate spread -> DIST floor (§D-24); --duration
                                            # is REQUIRED once a baseline has ON groups at two run lengths
python replicate_floor.py --mode real --duration 7200 --profile-out ../parity_floors   # write the floors
python replicate_floor.py --mode sim  --duration 7200 --profile-out ../parity_floors   # ...and the sim side;
                                            # the gate takes the max of the two (§D-61). Run BOTH or it is
                                            # sized on real's spread alone, i.e. assumes sim is deterministic
python profile_sim_charges.py --real-run <real_dir> \
    --out ../sim_charge_profiles/<baseline>.yaml --only-observed   # re-profile ONE baseline's charges (§D-36)
```
⚠ **`replicate_floor.py` now MEASURES most floors by calling the rung** (§D-53), so the full nine-baseline
sweep is ~15 min, not instant. `--run-level` is the old fast estimator, kept only for comparison.
Rung catalog: PARITY.md §F. **Not redefined there:** per-stage wall-budget instrumentation
(`drain_wall_budget`, `trainer_phase_wall_budget`, `step_timing_breakdown`, `aggregation_compute_wall`) is
ONE-SIDED (`sim<=real`) where sim should collapse a real-transport phase to ~0, DISTRIBUTIONAL where it's
genuine shared compute. Implementation-level reference (tiers, the `pctl_band_ok` band-escape primitive
and its `min_abs` calibration rule, full wall-budget/timing rung table):
`async_cifar10/scripts/parity/PARITY_CHECKER_README.md`.

---

## §A  Score — the ON campaign

Everything here is **`jvp_eval_mode` ON**; an OFF row is a different training config and never compares to an
ON one. No OFF row is load-bearing any more — the legacy board is in `git log`.

**Every DIST verdict is graded against that baseline's own real↔real CONTROL, never the tolerance alone**
(§D-55). A tolerance says what we hoped; the control says what the pipeline can actually resolve.

### §A.1  Readiness ledger

| | stage | artifact |
|---|---|---|
| **R** | ON real legs at 7200s — n=3 where the cadence is unpinned, n=2 where a barrier/cap pins it (§D-52) | `experiments/run_*_<b>_*_real` |
| **FL** | replicate floor from them, BOTH sides, at the graded duration (§D-78) | `parity_floors/<b>.yaml` |
| **CH** | charge profile from THOSE reals | `sim_charge_profiles/<b>.yaml` |
| **SIM** | ON sim legs launched AFTER CH, n to match R | `experiments/run_*_<b>_*_sim` |
| **CTL** | control, `--control-mode both` — sim↔sim reads what real↔real cannot (§A.3) | §A.3 |
| **GR** | graded row | `experiments/_parity_reports/` |

**ALL NINE baselines now carry all six stages, two-sided, on one commit
(`ceb119c6c` for block 2, `bdbde72b7` for block 1).** No baseline is graded against a one-sided floor and
none is ungraded. Blocker 1 is closed.

⚠ **`fwdllm_it` is NO LONGER POOLED** (§A.4). At n=3 per name per side the two names do not produce the same
numbers — real 38 bins / `iters/bin` 9.64-9.67 under `_unaware` against 40 / 9.20-9.22 under `_oracular`,
each name exact to 4 s.f. — so pooling reported a SYSTEMATIC offset as replicate noise and took a pinned
baseline's floor from 0.0% to 5.1%. Un-pooling took both rows from 64/0/28 to **69/0/23**: five rungs had
been SKIPping on a floor that was not noise. `fedbuff_it` still pools, and harmlessly — its pooled 9.2%
equals `_oracular`'s own 9.2%, and its legs interleave rather than split.

⚠ **`fwdllm_it` is also n=2 on `_unaware`'s sim side** — one leg died on an MQTT disconnect 8 minutes in and
is quarantined under `experiments/_aborted/`. It graded as 23 pass / 67 skip before that: a dead leg does not
announce itself (§D-83).

### §A.2  The board — and the control that reads it

`run_parity.py`, two-sided floors, duration-matched pairs. Per-pair JSON:
`experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`.

**Every row is n≥2 both sides, one commit per block, two-sided floors:**

**Every rung's verdict is the MEDIAN over all same-code real legs** (§D-90), not one drawn leg. `real_draws`
on every rung records each cell, and the CLI prints a `draw-split` line for any rung that failed a minority.

| baseline | pass/fail/skip | failing rungs | floor real / **sim**, `iters/bin` | verdict |
|---|---|---|---|---|
| `fluxtune` | **74/0/19** | — | 2.3% / **1.4%** | cap-pinned both sides; the tightest unpinned-adjacent row |
| `fwdllm_it_unaware` | **69/0/23** | — | 0.0% / **0.0%** | pinned; +5 rungs re-enforced by un-pooling |
| `fwdllm_it_oracular` | **69/0/23** | — | 0.0% / **0.0%** | pinned; its block completed this batch |
| `fwdllm` | **68/0/24** | — | 0.0% / **0.0%** | pinned negative control; every leg identical to 4 s.f. |
| `felix_it` | **66/0/27** | — | **18.9%** / 2.4% | clean, on the widest REAL floor on the board |
| `fedbuff_it_unaware` | **66/0/26** | — | 9.2% / **8.2%** | its pre-batch `v1` fail was a one-sided gate |
| `fedbuff_it_oracular` | 64/**1**/26 | `utility` | 9.2% / **8.2%** | fails 2 of 3 reals — a MAJORITY, below |
| `felix_round` | **62/0/30** | — | 15.9% / **8.2%** | its three fails were all checker defects (§G) |
| `fedbuff_round` | **62/0/30** | — | 15.4% / **21.2%** | its `utility` fail was the DRAW (§D-90) |

**`fedbuff_round`'s standing fail is CLOSED and was never about sim.** Over all 12 real×sim cells four rungs
fail 1-2 cells each and none systematically; `utility`'s 0.220 was the worst cell of twelve (median 0.096),
pairing the highest-loss sim leg with the lowest-loss real. Against the pooled real distribution the sim legs
read 0.059 / 0.066 / 0.137 where real's own leave-one-out reads 0.016-0.114, and pooled-vs-pooled is **0.040**.

⚠ **The same rule surfaced a fail it was hiding: `fedbuff_it_oracular`'s `utility` fails 2 of 3 reals** —
0.220 against a 0.185 floor and a 0.2 gate, the same knife-edge shape but a MAJORITY, so not the draw. One
more sim leg (§B.3 #2), not a widened gate. ⚠ Its floor is the POOLED `fedbuff_it` row, premise under review
in §B.3 #1.

**The board improved this batch by fixing MEASUREMENT, not the simulator.** Nine rungs' verdicts changed and
**no simulator code was touched**: three rungs compared one leg's wall against another's vclock, three graded
the full run because a matched window was gated on a clock it never reads, and one had its floor signal
discarded. That is the good version of this outcome and the one to be most suspicious of — §B.0.

### §A.3  Which rungs the control can and cannot read

`run_parity.py --control --control-mode both --duration 7200`. This table is computed, not hand-kept:
re-run it, don't edit it. **Readability is a property of the PAIR, not the rung** — block 1's sim replicates
made a whole family readable for the first time.

Three rungs stay UNREADABLE and should be: `field_coverage`, `vclock_telemetry` and `sim_send_ts` are
sim-only INVARIANTS, not comparisons, so they bail rather than measure (§D-56). The CLI counts a bail apart
from a fail.

**On REAL pairs every readable rung is 0-fail** — 27 pairs across both blocks, 52-61 rungs clean per pair,
and 66 rungs are 0-fail over EVERY pair in either mode. The real side of this pipeline has no rung that
fails on its own noise.

**The family that used to fire at 90% was a measurement bug, and it is gone.** Re-measured over 15 pairs on
`felix_round` + `fedbuff_round` after each side was made to read its own clock (§D-73, §G):

| rung | sim↔sim, before | after | reads |
|---|---|---|---|
| `overlap_factor` | **9/10** | **0/15** | was reporting sim's speedup ratio; now floor-gated too |
| `per_round_advance` | **9/10** | **0/15** | ditto; floor-gated, SKIPs where it cannot resolve |
| `overhead_residual` | **9/10** | 4/15 | the genuine residue — now floor-gated, floors 0.0-23.4% |
| `drain_wall_budget` | 7/25 | 6/15 | sim's own wall span — settled, do not investigate (below) |
| `utility` | 1/6 | **0/15** | now grades the matched window |
| `g2_grad_pool_size` | 1/6 | 1/15 | — |

⚠ **This table predates the last two floor-gatings** (`overhead_residual`, `overlap_factor`) and the
`fwdllm_it` un-pooling. Re-run `--control --control-mode both` before citing it; §A.2's board is current,
this is not. The shape of the finding stands: the family's control was a unit mismatch, not noise.

**`drain_wall_budget` is settled by `fwdllm`'s negative control.** Its three sim legs are logically identical
— bins 41, cycles 323, iters/bin 7.71, var 0.8099, vclock 7182, every digit — and differ only in wall span
(339/341/343s). The rung fails all three pairs anyway, so it is grading host contention, not work (§D-31).
Do not open an investigation on it.

### §A.4  Floors — two-sided, one commit per block

`replicate_floor.py --mode real|sim --duration 7200 --profile-out ../parity_floors`. **Run BOTH passes**: the
gate takes the per-metric MAX of the two sides (§D-78), so running only `--mode real` sizes it on real's
spread alone, i.e. assumes sim is deterministic. Floors are inputs to the checker, not a table: DIST
tolerances tighten toward `3x floor`, never past each field's own `min_abs`, never looser than nominal, and
SKIP once the floor swallows the tolerance (§D-24, §D-36).

**Every floor is measured BY THE RUNG** — `replicate_floor` calls `iters_per_data_id_parity(legA, legB)` and
reads its `mean_rel_diff`, so the floor and the tolerance it sizes grade the same window by construction
(§D-53). The floor tool and `--control` agree to 3 decimals; if they ever disagree, one is on the wrong window.

All nine, real / **sim**. `fedbuff_it` shares a row (still pooled); `fwdllm_it` no longer does.

| baseline | n r/s | `iters_per_bin` | `time_to_n` | `throughput` | `mean_var` |
|---|---|---|---|---|---|
| `fedbuff_round` | 4/3 | 15.4 / **21.2** | 14.5 / **19.9** | 14.6 / **20.0** | 6.3 / **9.8** |
| `felix_it` | 3/3 | **18.9** / 2.4 | **18.9** / 3.0 | **19.0** / 3.0 | **6.7** / 2.8 |
| `felix_round` | 3/3 | **15.9** / 8.2 | **15.6** / 7.9 | **15.6** / 7.9 | **7.6** / 6.4 |
| `fedbuff_it` (pooled) | 6/3 | **9.2** / 8.2 | **9.0** / 8.2 | **9.1** / 8.2 | **7.3** / 1.4 |
| `fluxtune` (capped) | 3/3 | **2.3** / 1.4 | **2.1** / 1.1 | **2.1** / 1.2 | 4.9 / **7.7** |
| `fwdllm` · `fwdllm_it_*` (pinned) | 3/2-3 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| nominal gate | | 15% | 8% | 8% | 2% |

**Four metrics joined the two-sided set this batch** — `round_advance_rel`, `round_advance_ks`,
`mean_chosen`, `utility_ks`, `overhead_rel`, `overlap_rel` — and every one of them exposed a gate that was
never a calibration: `overhead_rel` measures 0.0-23.4% against a flat 10%, `utility_ks` 0.0-19.8% against a
flat 0.2, `round_advance_ks` 0.0-32.4% against a flat 0.2. On the PINNED baselines all of them read 0.000, so
those rungs stay fully enforced; on the unpinned ones the gate SKIPs and says the number in its `reason`.

**Neither side is reliably the noisier one, measured on seven baselines.** Sim is wider on `fedbuff_round`
and on `fluxtune`'s `mean_var`; real is wider on `felix_it` (6-8x), `felix_round` and `fedbuff_it`; both are
exactly 0 on the pinned ones. The old "sim's spread is ~2.2x real's" was measured across three commits and is
dead (§E, §D-70) — which is why the gate takes the max rather than assuming a side.

**Every floor file now records its own provenance** (§D-87), so a future re-calibration can tell from the
file alone whether its legs are comparable: `code_commit`, `nodes`, `source_runs`, `n_replicates`,
`measured_at`, `span_axis`, `trace`, `max_runtime_s`, `floor_tool_version` — separately for each side
(`sim_*`). Bump `_FLOOR_TOOL_VERSION` when a floor's MEANING changes, so a stale file is visible rather than
silently mixed with a fresh one.

**The hitting-time model fits BOTH sides of `fedbuff_round`.** `var_threshold` 0.30 against an achieved mean
variance of ~0.97 and a 10th percentile of 0.31: a bin commits only when the noisy estimate dips into its
bottom ~7% tail, so the floor is a hitting time, not a convergence gap (§D-62, F7). Predicted 3-leg spread
~12%; the 4th real leg landed at `iters/bin` 13.01 over 198 bins against a 14.17-14.23 / 182-185 cluster,
taking the real floor **0.7% → 15.4%**. §B.4's falsifier fired on the first extra draw.

**The floor estimator is monotone in n, and `fedbuff_it` is the check on that** (§D-57, §D-68): pooling block
1's three legs with block 2's three took it from 6.3% at n=3 to **9.2% at n=6 — exactly block 1's number**.
A floor that does not move when you double the legs is one you can spend.

### §A.5  Duration: the 2h floors transfer, on the pinned baseline only

`fluxtune` ×2 at 14400s (node A), paired per-bin at equal bin index (§D-38):

| | Q1 | Q2 | Q3 | Q4 | Q5 | whole window |
|---|---|---|---|---|---|---|
| 2h segment-level spread | 3.6% | 1.4% | 0.3% | 0.9% | 0.3% | 0.83% |
| 4h segment-level spread | 2.6% | 1.6% | 1.3% | 3.2% | 1.0% | **0.38%** |

Flat across quintiles at both durations, and the whole-window spread *shrinks* with length — more bins average
out more noise. No accumulating divergence. ⚠ **This licenses transfer for the PINNED baselines only.** A pin
suppresses residual and floor together (§D-52), and node A bought 4h evidence on the pinned one. The unpinned
`fedbuff_*` have no 4h replicate; do not assume their 2h floors carry.

---

## §B  Next steps

> **Everything needed to resume from a cold start is in this section.** §B.0 = where we are. §B.1 = how to run
> anything. §B.2 = what the residual IS and what is ruled out. §B.3 = the ordered queue. §B.4 = open questions
> with falsifiers. §B.5-8 = threshold provenance, tolerances, deliberate gaps, backlog.

### §B.0  Where this stands

> Parity's job is that **sim does not change the CONCLUSION**, not that every rung is green. Every claim here
> is comparative, so a residual identical on every baseline cancels out of a ranking.

| # | criterion | status |
|---|---|---|
| 1 | Every INV/EXACT rung green on all nine | **MET.** Every INV/EXACT rung is green on all NINE. The one remaining fail is DIST — `fedbuff_it_oracular`'s `utility` at 1.19x its own replicate floor, on a MAJORITY of real legs (§A.2) |
| 2 | Convergence + terminal state inside each baseline's own replicate band | **MET, and enforced rather than argued** — `convergence`'s `acc_tol` is floor-gated, so "inside its own band" is what the rung tests |
| 3 | Every remaining DIST residual is COMMON-MODE | **MET on 9 of 9** |
| 4 | No residual correlates with a baseline-DISTINGUISHING knob | **MET** — the correlation §A.2 used to report was the replicate floor tracking the pin, not a sim bias |

**No parity criterion has a known failure left, and blocker 1 is DISCHARGED.** Both sides are replicated on
one commit for eight of nine baselines, and the two-sided floor turned out to matter in both directions: sim
is the noisier side on `fedbuff_round`, real is the noisier side on `felix_it` by 6-8x. What is left is one
unfinished node block and calibration, not a defect.

⚠ **The board improved this batch by fixing MEASUREMENT, not the simulator — 3 fails → 1, with no simulator
change at all.** Two gates were grading below their own measured noise, one statistic was pooling unequal
prefixes, and a whole family's control was reading wall against vclock. That is the good version of this
outcome and also the one to be suspicious of: see the over-optimization tells below, and note that
`overhead_residual` came out of the wash with a REAL 4/15 signal that was previously hidden under the bug.

**ONE blocker, and it is not a parity bug:**

⚠ **The critical path for the PAPER is a training bug.** Node A's two 14400s `fluxtune` legs both collapse to
chance — peak 84.7/85.3 in round 1, **25.0/25.2** by the end of round 2 — and I-1's `||Δ||/||w||` sits flat at
~0.006 through round 1, **steps ~2.5x at the round-2 boundary** (to 0.014/0.018 while accuracy is still
82/76), then climbs to ~0.03 as accuracy falls. The ratio rises BEFORE accuracy does; the flat-ratio
falsifier did not fire. **I-1 CONFIRMED, reproducibly, on both legs**, and sharpened: the trigger is the
**lap boundary**, a step change, not gradual drift. Owned by
[fluxtune_contributions.md](fluxtune_contributions.md) §8 / S1-S2, not here — but EXPTS_CHARTER's E1 error
bar sits at a ~3.9h peak (~234 min) and the collapse completes by ~232 min, so **no long `fluxtune`
experiment run is meaningful until it is damped.** Plots:
`experiments/run_20260804_0{03042,43301}_fluxtune_*/plots/server_update.png`.

**Parity coverage is DONE.** Nine of nine baselines, both sides, one commit per block, two-sided floors, and
every INV/EXACT rung green. What remains is calibration debt (§B.5) and one marginal DIST residual (§A.2) —
neither gates the experiment runs.

**Tells that this has tipped into over-optimization:** chasing a rung whose residual is inside its own control
· the board getting worse from measurement changes rather than better from fixes · adding instrumentation
faster than closing bugs. All live now that every row is at its floor — §A.3 is the check to run before
opening anything.

### §B.1  The pipeline — how to run anything  *(KEEP: this is the runbook)*

```bash
cd lib/python/examples/fwdllm/expt_scripts

# R — ON real legs (n=3 where the cadence is unpinned)
bash run_sequential.sh --mode real --max-runtime-s 7200 --only <b> --yes

# S — ON sim legs. n=3 TOO from now on (§B.2): 35-50 min each, so 3 legs is one short node block
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only <b> --yes

# S+ — ONE MORE sim leg on a baseline that already has a block. NEVER run CH first: a re-profile makes the
#      new leg a non-replicate of the others (§D-91). Preflight blocks on any newer real; --force is right
#      here, and ONLY here. Same node as the existing legs.
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only <b> --yes --force

# CTL — the control. No sim leg, no new run; the primary reader of any DIST verdict
python run_parity.py --control --duration 7200 --yes --baselines <b>
python run_parity.py --control --control-mode sim --duration 7200 --yes  # reads the 6 vclock rungs too

# FL — from runs already on disk; no sim leg needed
python replicate_floor.py --mode real --duration 7200 --baselines <b> --profile-out ../parity_floors
python replicate_floor.py --mode sim  --duration 7200 --baselines <b> --profile-out ../parity_floors
                                            # the sim side of the two-sided floor. --profile-out is
                                            # REQUIRED: without it the gate silently stays one-sided

# CH + SIM + GR — one chain, never split
cp ../sim_charge_profiles/<b>.yaml ../sim_charge_profiles/<b>.yaml.bak && \
python profile_sim_charges.py $(ls -d ../experiments/*_<b>_n100_*_real | sort | tail -2 | sed 's/^/--real-run /') \
    --out ../sim_charge_profiles/<b>.yaml --only-observed && \
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only <b> --yes && \
python run_parity.py --yes --baselines <b>
```

**Guards — all mechanical:**
- Launch preflight **BLOCKS** a sim leg whose charge profile predates any same-flag real. `--force` overrides;
  write down why.
- `run_parity.py` pairs on **BOTH `jvp_eval_mode` and `max_runtime_s`**, and names which term differed on any
  newer real it skipped.
- `replicate_floor.py --profile-out` warns when a baseline has ON groups at two durations; `--duration`
  selects. A floor must come from the run length the rung grades (§D-53).

**Still on you:**
- ⚠ **Glob precision.** `*fwdllm*_real` also matches `fwdllm_it_unaware`/`_oracular`; use
  `*_<baseline>_n100_*_real`.
- ⚠ **`--dry-run` first, always** — full preflight without launching; 30s against a 2h leg.
- ⚠ **Keep a config's replicate legs on ONE node** unless the nodes are known identical.
- ⚠ **Never wrap a long analysis in `conda run`** — it buffers all output to exit, so a working job is
  indistinguishable from a hung one. Call `.../envs/dg_flame/bin/python -u` directly.

**Cost:** 7200s REAL leg ≈ 2h05 wall; 7200s-vclock SIM leg ≈ 35-50 min (`fwdllm` family 10-15 min). A 6h node
fits two real legs plus a sim chain, **or three sim replicates plus a re-grade**.

### §B.2  What the spread IS — the mechanism, and what is ruled out

**The mechanism, named and measured.** `var_threshold` is **0.30** on every baseline, but achieved mean
variance is **0.77-1.89 — 2.6x to 6.3x the threshold — on all nine ON runs.** The gate never reaches a
plateau, so **every commit fires on a random noise dip.** `iterations_per_data_id` is therefore not a
converged quantity, it is the **hitting time of a noisy process**, and its run-to-run spread is that hitting
time's variance. This is `fluxtune_contributions.md` §8 **F7** measured on the parity board — one phenomenon,
not two.

Block 1 sharpened it from a ratio into a distribution. On `fedbuff_round` the variance p10 is 0.31 against
the 0.30 gate and every commit fires at ~0.286, so the gate samples the bottom ~7% tail; total cycles hold
constant across legs while the bin count wanders (§A.4). **Where the hitting time is truncated the spread is
exactly 0; where nothing truncates it, 8-21%.**

| baseline | mean var | var/thr | iters/bin | hitting time truncated by |
|---|---|---|---|---|
| `fwdllm` · `fwdllm_it_*` | 0.77-0.85 | 2.6-2.8x | 7.4-9.7 | sync full-cohort barrier |
| `fluxtune` | 1.89-2.23 | 6.3-7.4x | 18.6-19.0 | `max_iterations_per_data_id: 20` + plateau policy |
| `felix_round` · `felix_it` · `fedbuff_*` | 0.94-1.03 | 3.1-3.4x | 12.5-16.5 | — |

**What this RULES OUT — all on telemetry already on disk, no run needed:**

| candidate | verdict | evidence |
|---|---|---|
| **Dynamic KC policy** | **NOT the culprit — it never ran** | `dynamic_kc.enabled: False` in the only baseline that has the knob. All three DK rungs SKIP: *"DynamicKC disabled — constant K (real=10, sim=10)"* |
| **Max-iterations cap** | **NOT a culprit — but it IS the board's biggest confound** | `max_iterations_per_data_id` exists on `fluxtune` ONLY. It does not cause the spread, it **hides** it: capping truncates the hitting time, which is why `fluxtune` has a ~0 floor (§D-52) |
| **Aggregation method** | **NOT the culprit — it does not predict the spread** | `agg_rate_conf.type: old` brackets `new`'s range. Within-method spread exceeds between-method spread |
| **Oracular availability tracking** | **REFUTED — provably inert at syn_0** | `eligibility` pool **100.0/100.0**, ks 0.0 · `eligible_pool_reduction` **0.0/0.0** · `avail_timebase` **0.0** · `duty_cycle_duration` mean_err **0.0**. The oracle removes nobody |

**And the consequence that reframes two rows.** The resolved configs of `fedbuff_it_oracular` and
`fedbuff_it_unaware` differ in **exactly one key** — `trackTrainerAvail` — which the row above proves inert.
**At syn_0 they are the SAME experiment**, and the same holds for `fwdllm_it_oracular` ≡ `fwdllm_it_unaware`.
Both pairs pool their floors under a declarative syn_0-only alias that Phase 2 deletes (§D-63); the board
rows stay separate.

⚠ **"Inert" is doing more work in that sentence than the evidence supports, and it is now the top of the
queue (§B.3 #1).** What is measured is that the oracle removes NOBODY from the eligible pool — pool 100/100,
`eligible_pool_reduction` 0.0. What is NOT measured is that it changes nothing: at n=3 per name per side the
two `fwdllm_it` names differ deterministically (real 38 bins / 9.64 vs 40 / 9.20), on identical hardware,
with `trackTrainerAvail` the only difference in their resolved configs. Only the PINNED family can see this —
elsewhere a ~5% effect sits inside a 9-19% replicate floor. The availability rungs above compare real↔sim
WITHIN each name; none of them ever compared the two names (§D-88).

⚠ **Every cross-code number this section used to carry is deleted, not corrected.** The "sim is the noisier
side" ladder, the 8.1% / 25.7% sim↔sim spreads and the cross-graded residuals were all measured across three
commits (§E, §D-70). Block 1 replaced them with same-code n=3 on both sides (§A.4).

### §B.3  RESUME HERE — the ordered queue

1. **Settle whether `trackTrainerAvail` is really inert at syn_0 — ONE real leg.** The two `fwdllm_it` names
   differ deterministically (real 38 bins / 9.64 under `_unaware` vs 40 / 9.20 under `_oracular`, 3/3 each,
   both modes) on identical hardware, and their resolved configs differ in **exactly that knob**. §B.2 and §E
   both say the oracle is provably inert at syn_0; that claim rests on the eligible pool reading 100/100,
   which shows the oracle removes NOBODY — not that it changes nothing.
   ```bash
   bash run_sequential.sh --mode real --max-runtime-s 7200 --only fwdllm_it_unaware --yes   # on wash
   ```
   **CONFIG IF** an `_unaware` leg on `_oracular`'s node still reads 38 / 9.64 — then the knob is not inert
   and §B.2/§E need correcting, which also puts the `fedbuff_it` pooling premise in doubt (its ~5% effect
   would sit inside its own 9.2% noise, invisible). **HOST IF** it reads 40 / 9.20.
   ⚠ Only the PINNED family can resolve this: it is the one with a 0.0% floor, so a 5% systematic effect is
   visible instead of buried (§D-52 inverted).

2. **One more `fedbuff_it_oracular` SIM leg** — the board's last fail, and the only one the median says is
   not a draw (2 of 3 reals). `utility` reads 0.220 against a 0.185 pooled floor; n=3 → n=4 on the noisier
   side either takes the floor past the 0.2 nominal, making the rung an honest SKIP, or leaves it gradeable
   and the residual becomes a real question. 35-50 min. ⚠ Do NOT re-profile its charges first — that makes
   the new leg a non-replicate of the other three (§D-91). Preflight will block on a newer real; `--force`
   is correct here and only here.
   ```bash
   bash run_sequential.sh --mode sim --max-runtime-s 7200 --only fedbuff_it_oracular --yes --force
   ```

3. ~~Grade a sim leg against the real DISTRIBUTION~~ — **DONE** (§G, §D-90). Every rung's verdict is now the
   median over all same-code real legs, and it moved the board in both directions.

4. **Sign-off re-grade**, then the experiment runs (`paper_expts_fluxtune/EXPERIMENTS.md`), sim-only.
   ⚠ **Gated on I-1 damping (§B.0), not on parity — parity coverage is done.**

**Hand back to `fluxtune_contributions.md` §8:** F7 is CONFIRMED at N=100 α=1 on all nine ON baselines
(var/threshold 2.6-6.3x), and **S2 outranks S1 on impact** — the un-plateaued gate is what makes `iters/bin`
a hitting time, i.e. the source of the replicate floor. Predicted: a gate that commits on the plateau
collapses the spread toward `fluxtune`'s ~0, a falsifiable read-out that needs no accuracy win to interpret.

⚠ **Do not re-run the refuted (§E).** The tie-break, redispatch stagger, cohort-composition bias, "sim
over-iterates" and "sim is the noisier side" each cost one telemetry pass to kill.
⚠ **The oracular-tracking entry is the exception — it is UNDER REVIEW, not dead** (§B.3 #1).

⚠ **`run_parity.py` exits 1 whenever any rung fails** — the normal outcome (§D-51). Never mid-`&&`.

⚠ **Moving run dirs off a node does NOT bring `parity_floors/` or `sim_charge_profiles/`**, and copying those
directories wholesale between nodes CLOBBERS them (each node holds all nine files but freshly wrote only its
own). Re-derive from the run dirs instead — both are pure functions of them (§D-80).

### §B.4  Open questions — each with its falsifier  *(KEEP: not yet tasks)*

State the prediction BEFORE the run; a hypothesis that can only be confirmed is not one (§D-9).

- **Is `trackTrainerAvail` inert at syn_0, or only pool-neutral?** Promoted to a TASK (§B.3 #1) — the two
  `fwdllm_it` names differ deterministically on identical hardware with exactly that knob between them. The
  standing claim (§B.2, §E) rests on the oracle removing nobody from the eligible pool, which is a weaker
  statement than "changes nothing". ⚠ If it is the knob, `fedbuff_it`'s pooled floor is also unsound.
- **Do the oracular rows belong on the syn_0 board at all?** Sharpened by the above rather than answered: if
  the knob is not inert they are two configs even at syn_0, and the rows must stay. Operator call.
- **`felix_it`'s ON accuracy drop is baseline-specific.** Its ON band sits 1.0-3.4 pts below its OFF band
  while `felix_round` — same aggregation rate — improved. **FALSIFIED IF** a second loss-derived baseline
  degrades ON. Low priority; seven other bands support the flag. Do not re-open the shared-mask JVP.
### §B.5  Threshold provenance — where every gate's number is allowed to come from

One question classifies every rung: **does this quantity differ between two runs that should be identical?**

| class | criterion | gate may be | count |
|---|---|---|---|
| **INVARIANT** | structural; two identical runs agree exactly, floor is ZERO by construction | exact/boolean — floor-gating one would license drift | 14 |
| **CALIBRATED** | paired real↔sim comparison of a quantity with run-to-run variance | **must** come from a measured floor (§D-24) | 54 |
| **POLICY** | one-sided bound on ONE run; no paired comparison, so no floor is definable | a documented engineering choice | 13 |
| *unclassified* | needs its own rung read; guessing is worse than saying so | — | 11 |

There is deliberately **no "hand-typed" class — that is the failure state**, and it produced every defect this
batch fixed: a 15% gate over a 0.6% floor (§D-24), a 5% trainers gate under a 5.5% floor (§D-72), a 0.2 KS
gate that passed a 27% tail gap (§D-76).

⚠ **43 of the 54 CALIBRATED rungs still have NO measured floor** (`checks.calibration_debt()`). They are
0-fail on the control, but **0-fail does not mean calibrated — it can mean blind**: a gate 10-20x above its
noise never fires and never catches anything either. Shrinking this list is what each replicate batch buys,
and it costs no node time — re-grading reads run dirs already on disk. The whole clock family —
`per_round_advance`, `utility`, `overhead_residual`, `overlap_factor` — left it this batch, and a pytest
ratchet now holds the count so it can only fall.

**Most of the gates closed this batch were not merely uncalibrated — they were wrong in a way a floor alone
would not have caught**, which is why the order matters: fix the measurement, THEN floor-gate. `utility` was
comparing unequal prefixes and the clock family's control was on the wrong clock. A measured floor over a
broken measurement is still broken (§D-85).

`test_threshold_provenance` is a RATCHET: a new rung cannot land unclassified, an INVARIANT can never be
floor-gated, and the unclassified list may only shrink.

### §B.6  Tolerances and rung gaps

- **Every "recalibrate against the floor" item here is LANDED (§G).** `conv`/`v1b`/`v1c`/`cohort` are
  floor-gated on the rung's own window and 0-fail on every real control pair; `v1` was left alone.
  `utility`, `per_round_advance`, `overhead_residual`, `overlap_factor` and `selection_detail`'s
  `tol_chosen` all joined them this batch. **No clock rung is hand-typed any more.**
- **`v1c`'s power problem is now the opposite one.** Its 10 bins were thought too few to see a real drift; the
  19-pair control says its t-test rejects on the pipeline's own noise instead (|t| to 5.72, |λ| to 0.198/100
  against a 0.05 floor). It is gated on `lambda_floor_per_100` AND significance, and the floor is now measured
  — so it SKIPs where it cannot resolve rather than firing. Do not raise `n_bins` (§E).
- KS-only rungs unguarded against a level shift (all clean on live data): `dk1_agg_goal_trajectory`,
  `dk2_dynamic_c`, `dk3_eligible_ends_metric`, `eligible_speed`, `v3_cached_v_pool`.
- Thin ABSOLUTE budgets: the `fwdllm` family grades N=40-41 against 173-254 elsewhere, at 93-95% coverage.
  All four rows are 0-fail so nothing is masked, but coverage percentage cannot catch thinness; an absolute-N
  floor is proposed, threshold not chosen.
- `step_timing_breakdown` / `agg_step_timing_breakdown` are REPORT-ONLY wherever the sim clock discards the
  span (§D-31) — the designed steady state. Watch `charge_coverage` instead.
- Do not re-tune `redispatch_turnaround` (§D-14, §E).

### §B.7  Known-and-deliberate

- Real's over-`c` slot read is drain lag, not a §D-27 conflation (§E, §G). Telemetry only. Low priority.
- Real publishes no `_agg_slot_holders_ref`, so `_cap_dispatch_to_concurrency` falls back to the IDENTITY set.
  Harmless today; publishing one would let real dispatch into slots it now withholds — owed its own A/B.
- Sim's in-flight bookkeeping is split across six sets and should be one per-end state machine. The slot⇄guard
  split did the CAPACITY half; IDENTITY is still ad-hoc. Never bundle with a correctness fix.
- Base `asyncfl/top_aggregator._sim_hold_busy_slots` deliberately untouched — no `_sim_committed` term, so it
  never had the conflation. Re-check if async_cifar10 shows the same under-fill.

### §B.8  Backlog

- **A run dir cannot say which training config produced it.** `jvp_eval_mode` lives in the trainer's
  `config_overrides`, which the runner never dumps — the only record is 100 lines in the trainer log, which
  both `replicate_floor.py` and `run_parity.py` now grep. Dump the resolved TRAINER config into the run dir
  and key off it. Same root, second symptom: a run's floor and charge profile live outside the run dir, so
  moving run dirs strands them. Both should be run artifacts.
- **Enforce §F-18 mechanically: a per-baseline knob CONTRACT.** Two correctness-path knobs went missing from
  yamls and were caught days later by reading telemetry. The mechanism mostly EXISTS — `run_sequential.sh`'s
  preflight `checks[]` blocks a launch; the gap is that `condition_fp` hashes only CLI-patched knobs, so
  yaml-only knobs are invisible. **The hard part is "missing" vs "legitimately N/A"** (`trackTrainerAvail` is
  oracular-only, etc.), so applicability must be DECLARED, not diffed — sketch: a `knob_contract` block in
  `_metadata/baselines.yaml` read by all three layers (`test_baseline_readiness.py` → preflight → parity
  `--validate`). Open: where it lives; error-vs-warn; who declares a new knob.
- **Leg COMPLETENESS is checked by one tool, not by the shared discovery helpers.** `replicate_floor` drops a
  dead leg; `--control` pooled one and reported artifact fails from it (§D-83, §G). `drop_truncated` already
  grades the achieved span — it needs a companion that rejects a leg whose telemetry never reached the first
  commit, applied wherever legs are discovered. Quarantining to `experiments/_aborted/` is the manual
  stopgap and depends on someone noticing.
- **The preflight's python lives in a `run_sequential.sh` heredoc, so its checks cannot be unit-tested.**
  Extract into an importable module; `lib/python/tests/mode/test_baseline_readiness.py` is the natural home.
- Flag promotion: `sim_sct_ordered_drain` + `sim_model_dispatch_queue` are fluxtune-yaml-only but model
  general async-transport artifacts — smoke fwdllm/fwdllm_plus with both ON, confirm inert-or-better, promote.
- Checker invariants I1-I6 were drafted in a prior session and never committed (unrecoverable). Needs operator
  input on intended semantics before drafting fresh ones.
- felix (async_cifar10) may share fluxtune's round-1 cold-start gap — unverified, out of scope. felix 46/46
  reconfirmation gates Phase 2.
- Momentum (S1-S3) / server-optimizer — roadmap, not parity, but see §B.0: I-1 is now CONFIRMED and gates the
  experiment runs. S1's damping should also shrink the replicate floor — re-measure after it lands.
- P3/infra: no automatic GPU skip-and-remap on a broken ordinal (manual `execution.gpu_ids` exclude works).
- ~~Five RED tests~~ — **ALL FIVE CLOSED** (§G). ⚠ **Two suites, and `lib/python/tests` alone does not run
  both**: `test_ladder.py` and the rest of the checker's tests live under `async_cifar10/scripts/parity/`, so
  the pytest line in §C is the one to use. **588 green across all four suites.** Four were tests trailing a
  landed change; the fifth was a real coverage hole and is now closed in the CHECKER, not the test (§D-92).

**Standing rules.** A real↔real control needs NO sim leg — it is the cheapest evidence in this whole
pipeline, and it should run BEFORE any mechanism hypothesis (§D-55). One mechanism per run. Never spend a run
on a question a bench repro can answer (preamble).

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
| **anything past the first lap boundary** | **4h+** | `fluxtune` trains cleanly through round 1 and collapses to chance in round 2 (§B.0). A 2h leg cannot see it — every accuracy number in §A.4 is a round-1 number |

**WINDOWED vs UN-WINDOWED decides whether a short run is readable — read the field, don't assume.** A rung
carrying `matched_logical_budget_n` grades only the work both sides did, so a short run gives it a thin N and
a weak verdict (`fwdllm` is unreadable at any short duration). A rung without it pools the whole run and
grades at full strength immediately: every INV tripwire, `slot_utilization`, `throughput`.

**The floor grows as runs shorten, so re-measure it per baseline when adopting a new run length** (§D-24) —
`replicate_floor.py --mode real --duration <s>`, and prefer THREE legs on an unpinned baseline: two give a
floor with zero degrees of freedom (§D-57). Smoke (5-10 min) before any long run; one mechanism per run when
a fix could perturb another baseline.

**pytest** (`setup.cfg` sets `addopts = -n auto`, needs `pytest-xdist`; `-o addopts=""` runs serially if an
env lacks it):
```bash
# ⚠ THREE paths. The checker's own suite lives beside it, NOT under lib/python/tests — omitting the third
# hides `test_ladder.py` and `test_floor_gated_tol.py`, i.e. every test that guards the gates (§B.8).
conda run -n dg_flame python -m pytest lib/python/tests lib/python/examples/fwdllm/expt_scripts \
    lib/python/examples/async_cifar10/scripts/parity -q      # 1948 pass, 5 pre-existing REDs (§B.8)
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

**D-53.** Measure a floor on the SAME window and axis the rung grades — by CALLING that rung over two
replicate legs. Reimplementing its quantity understates the floor most where the residual is largest.

**D-54.** One divergent cohort is permanent: the weight trajectories separate and no later step is comparable
again. Grade cross-mode bit-identity of a per-end quantity to find WHERE parity is lost, not whether.

**D-55.** Before naming a mechanism for a residual, run the same comparison real↔real. A control carrying the
residual's own shape and magnitude means there is nothing to fix, whatever the mechanism story predicts.

**D-56.** A rung that reads a sim-only field cannot be controlled real↔real — it bails, and the bail looks
like a fail. List which rungs the control can actually read before crediting any of its verdicts.

**D-57.** A max-pairwise floor from TWO legs is a sample of size one: no spread, no degrees of freedom, biased
low. Use three on anything unpinned. Monotone in n, so a residual still outside on n=3 is real.

**D-58.** A residual that changes SIGN with which replicate it is graded against is noise, not a mechanism.
Grade against every replicate before believing a direction.

**D-59.** Two legs comparable means same flag AND same run length AND same achieved span. Any pairing rule
that matches on a subset silently grades the unmatched term.

**D-60.** A metric that only rises after a boundary needs a run that CROSSES the boundary. Peak accuracy from
a run that ends inside round 1 says nothing about round 2.

**D-61.** Replicate BOTH sides. A floor measured on one side only assumes the other is deterministic — assert
that, don't assume it. Sim legs are cheap; an n=1 sim makes every residual unreadable.

**D-62.** When a gate's threshold sits far below the achieved signal, the gate fires on noise dips and the
count it produces is a HITTING TIME, not a converged quantity. Its spread is the process's, not the code's.

**D-63.** Two configs differing only in a knob you can prove inert are ONE config. Pool their legs before
grading either — otherwise each is graded against half its own replicate evidence.

**D-64.** One quantity, one tolerance. Before believing a rung, check whether a sibling grades the SAME number
under a different hand-typed gate — one will fail while the other passes, and neither is the defect.

**D-65.** A replicate floor and a real↔real control are the same measurement. If the two disagree, one is
computed on the wrong window; make the floor tool call the rung and they cannot.

**D-66.** A tolerance in absolute units cannot share a relative floor's `min_abs`. Carry the tightest
meaningful value per FIELD, in that field's own units, or the gate is nonsense at 2%.

**D-67.** Two names for one config give each half the draws, so their floors differ by which extremes landed
where — not by behaviour. Sort every leg of the pooled config on the graded quantity before believing either.

**D-68.** A max-pairwise floor is an extreme-value statistic: it grows with leg count by construction, so a
"rising" floor can be pure estimator bias. Past ~5 legs quote a dispersion (sd) with its df instead.

**D-69.** Pairing one sim leg against "the latest real" makes the verdict an accident of run order. With the
replicate legs on disk, grade against the real DISTRIBUTION and report where in it the sim lands.

**D-70.** Two legs are replicates only if they ran the same CODE — pair on the run's own recorded commit, not
on recency. Compare by DIFFING the trees for run-affecting paths: raw SHA inequality splits on a docs commit.

**D-71.** A rung gating on several bounds can be calibrated on one and still fail on another. Read WHICH
sub-gate fired — and whether the rung reports the field it actually decides on.

**D-72.** A rung that bails for want of a sim-only field can still have a measurable floor: grade the REAL
side's own quantity across replicate legs. "Uncontrollable rung" does not mean "uncalibratable gate".

**D-73.** Every leg reads its OWN clock. Comparing one leg's process wall time to another's virtual clock
manufactures a residual out of a unit mismatch — 51% between two legs of one config. Fix it for the FAMILY
in one shared helper, with a sweep test: fixed rung-by-rung it came back on six more.

**D-74.** A floor measured from the SAME legs the control then grades cannot fail: the gate is 3x the worst
pair it is graded against. A floor-gated control is a consistency check, never independent evidence.

**D-75.** Compare the same WORK, not each side's own full run. Where per-unit cost trends (2.1-2.7x across
quintiles here), unequal unit counts are unequal windows and the longer side's tail biases its mean.

**D-76.** One summary cannot state a skewed, trending distribution: sim matched real's MEDIAN to 1.0% while
running a 27% heavier p90. Report the quantiles; gate only once a replicate floor exists for the one you gate.

**D-77.** Judge a leg's completeness on the clock that measures its WORK. A sim leg's wall span measures the
host, so a complete leg reads truncated for running on a quieter node — and gets silently dropped.

**D-78.** A residual drawn from two populations needs a gate sized on BOTH. Take the max of the two
replicate floors, never one side's; pooling both sides into one floor instead hides a genuine bias in it.

**D-79.** "Same code" is per-SIDE: an input only one mode reads cannot split the other's replicate group.
Diff the trees against what THAT leg actually consumed.

**D-80.** Derived artifacts belong to the node that computed them. Copying a shared directory between nodes
is last-writer-wins over every file; re-derive from the run dirs, which are the only real inputs.

**D-81.** A driver that only SEQUENCES the launcher is not run-affecting. Splitting replicate groups on it
discards the very leg it was edited to add — deny-list the orchestrator, keep the launcher.

**D-83.** A dead leg does not read as dead. It reports a run dir, a snapshot and a telemetry directory, and
grades as a near-total SKIP; quarantine it, or every tool that does not check completeness will pool it.

**D-84.** A guard gating a computed statistic on an input that statistic never reads is dead code with a
verdict attached. Check what the window is built FROM, not what the function happens to have in scope.

**D-85.** "Fails its own control" is only evidence once the control is known to measure the rung's quantity.
A broken control and an uncalibrated gate look identical on the board; check the control first.

**D-86.** A replicate floor is a SAME-EVERYTHING spread. Pool two groups only if they interleave on the
graded quantity — a systematic offset reported as noise inflated a pinned baseline's floor 0.0% → 5.1%.

**D-87.** A derived artifact must record what it was derived FROM — runs, code, hosts, duration, tool
version. Without it a file states a number but not what the number is OF, and re-calibration mixes eras.

**D-88.** "Inert" needs the comparison that would show it moving. A rung comparing real↔sim WITHIN each
config can never see a between-config effect; only a PINNED baseline resolves one below its own floor.

**D-89.** Analysis that takes >2 min is a tool bug, not a data cost. Parallelise across independent units,
cache the parse, stream the output — a buffered pipe makes a working job indistinguishable from a hung one.

**D-90.** One leg per side is a DRAW, and the draw can be the whole verdict — `fedbuff_round` failed
`utility` on 1 of 12 cells and that cell was the board. Grade against ALL same-code legs on the reference
side and take the median: fails a minority = measuring the draw, fails a majority = measuring the code.
**Symmetry is the proof it is not gate-tuning** — the same rule cleared one fail and surfaced another. Never
absorb the minority silently; report every cell, or you have deleted §D-69's evidence.

**D-91.** Provenance must compare what a leg CONSUMED, not what was committed beside it. A leg launched dirty
charges values its SHA does not name: three `fedbuff_round` sim legs recorded `bdbde72b7` yet charged the
profile committed one commit later, so a SHA check split three true replicates and would have silently
discarded a fourth. Where the input leaves a trace in telemetry, that trace outranks the commit.

**D-92.** "One measurement fails once" is right, but "same number on two rungs" must be CHECKED, not
assumed — defer only the part the owner explains. `selection_detail`'s re-draw count was handed to `v1` as
being v1's number; it is on 8 of 9 baselines, and on `felix_round` it reads 18.5% against v1's 1.6%, so 5
extra cohort draws over identical iteration volume were graded nowhere. Subtract what the owner explains and
vote on the remainder: coverage restored at ZERO change to all nine boards. **A red test is a hypothesis —
confirm the code is right before you fix the test**; four of these five were stale, the fifth was not.

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
- **A 2-leg replicate floor as a usable reference** — dead. `fedbuff_it_unaware`'s doubled (4.9 → 9.9%) on a
  3rd leg. Two legs give one pairwise difference and no way to know it was the close pair (§D-57).
- **"SIM is the noisier side"** — REFUTED. The 8.1% / 25.7% sim↔sim spreads and the 2.2x per-leg sd behind it
  pooled legs across THREE commits. Same-code, both sides, seven baselines: real is wider on `felix_it`
  (18.9 vs 2.4%), `felix_round`, `fedbuff_it` and `fwdllm_it`; sim is wider on `fedbuff_round` (21.2 vs
  15.4%); both are 0.0 on the pinned baseline. Neither side is reliably noisier (§D-78, §D-70).
- **"Real is structurally TIGHTER than sim on `fedbuff_round`"** — FALSIFIED by a 4th real leg (0.7% →
  15.4%). Do not go looking for a mechanism behind a floor measured on three legs of a hitting-time process.
- **Any floor, control or residual measured before the same-code filter** — dead wholesale, not case by case.
  Cross-code pooling produced the sim-noise story above and a phantom 18% `fwdllm` sim floor on a baseline
  whose real floor is 0.0%.

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
- **"Sim over-iterates per data-bin on the unpinned baselines"** — REFUTED as a general claim. It was a 2-leg
  artifact: on n=3 `fedbuff_round` flips SIGN (+6.9% → −4.7%) and `fedbuff_it_unaware` goes +9.3% → +0.4%,
  both inside real↔real spreads of 3.0-9.9% and 7.4-13.8% (§D-58). One baseline survives — see §B.2.
- **The pin/no-pin split as a sim BIAS** — REFUTED. The pin suppresses residual and floor by the same factor,
  so the knob predicted the noise, not a direction (§D-52). Exit criterion 4 is met, not failed.
- **`convergence`, `v1b`, `v1c`, `cohort_sequence` and `v2` AT THEIR OLD GATES as evidence about sim** — all
  five fired between config-identical REAL legs (10, 6, 6, 10 and 3 of 19 pairs). Four have since been
  re-derived against that control and read 0 of 19 (§G); `v2` has not. Do not resurrect the old thresholds,
  and do not open an investigation off a rung that fails its own control.
- **The agg-goal admission TIE-BREAK as the cohort-divergence root** — FALSIFIED on telemetry. Only 0.1% of
  2628 divergent `fedbuff_round` cycles swap ends of equal modeled D; the first divergence swaps three ends at
  D=4.294 for three at D=9.202. Sim already breaks exact `sct` ties by end id
  (`SimReorderBuffer.pop_min`), so that half was never broken. Speed-class ties are real (51 ends, 18 classes,
  largest holds 17) but they are not what moves membership.
- **Redispatch STAGGER as the divergence root** — FALSIFIED: the per-cycle spread of `task_recv` after a
  cohort closes is p50 8.5 ms in real and 63 ms in SIM, i.e. sim is the more staggered side.
- **Cohort COMPOSITION bias (sim favouring fast ends)** — FALSIFIED: mean modeled D matches within 0.2%, mean
  staleness within 0.02, distinct-ends-per-bin within 7%, and `participation` (S2) is green on all eight.
- **`fedbuff_it_oracular`'s residual as ORACULAR AVAILABILITY TRACKING** — REFUTED on telemetry. The oracle is
  inert at syn_0: eligible pool 100/100 both modes, `eligible_pool_reduction` 0.0, `avail_timebase` 0.0,
  `duty_cycle_duration` mean_err 0.0. Its resolved config differs from `fedbuff_it_unaware` in that ONE key,
  so the two are the same experiment (§B.2, §D-63).
- **DYNAMIC KC and the MAX-ITERATIONS CAP as culprits** — neither can be: `dynamic_kc.enabled: False`
  everywhere (dk1/dk2/dk3 all SKIP, "constant K real=10 sim=10"), and `max_iterations_per_data_id` exists on
  `fluxtune` alone. The cap is a CONFOUND (it hides the residual by truncating the hitting time), never a cause.
- **The AGGREGATION METHOD as the residual's driver** — REFUTED: `agg_rate_conf.type: old` spans −4.7% to
  +11.6%, bracketing `new`'s 0.0% to +7.5%. Within-method spread exceeds between-method spread.
- **A grad-pool accumulation bug behind `felix_round`'s cadence family** — FALSIFIED. Re-profiling the charges
  from the baseline's own ON reals closed `v1`/`v1b`/`cohort_sequence` with ZERO code change, and the residual
  reverses sign on baselines still carrying stale charges. Every summary statistic matching while the output
  moved was the tell that the divergent input was the CLOCK (§D-50). Do not re-open `calculate_var` or the
  pool-assembly path on cadence evidence alone.
- **`felix_it`'s 84.72% OFF leg as a lucky draw** — FALSIFIED; the replicate landed at 83.28, so the ON band
  really is below the OFF band on that one baseline (§B.4).
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
- **`fedbuff_round`'s standing `utility` fail is CLOSED, and it was the DRAW, not the simulator (§D-90).**
  Every rung's verdict is now the median over all same-code real legs, each cell kept in `real_draws` with a
  `draw-split` CLI line. Symmetric, and it proved it: `fedbuff_it_oracular`'s `utility` fails 2 of 3 reals
  and is now ON the board. Evidence in §A.2.
- **Two `_discover` defects fixed with it.** It matched reals on RAW SHA, excluding `fedbuff_round`'s 4th over
  a docs/floors/sim-charge diff a real leg never opens; now `code_differs(mode="real")`, as
  `largest_same_code` does. And a bailed rung carries no `ok`, which the first median ranked FAIL (§D-56).
- **All five standing RED tests closed — and the fifth was a real hole, not a stale test (§D-92).**
  `selection_detail`'s re-draw COUNT was deferred to `v1` as "the same number". It is, on 8 of 9 baselines;
  on `felix_round` it reads **18.5% against v1's 1.6%** — 5 extra cohort draws over identical iteration
  volume, graded by nothing. Fixed in the CHECKER: defer only the part `v1` explains, vote on the remainder.
  **Zero change to all nine boards**, re-graded to confirm; `felix_round` now records the 16.9% unexplained
  and SKIPs on its own measured floor instead of silently deferring. §D-64's case still defers.
- **`test_ladder.py`'s two RED tests were trailing a landed change, not defects.** The clock family's floor
  gating made `overlap_factor` co-fail, and the ladder declares `overhead_residual` DEPENDS on it, so the
  root is `overlap_factor` by construction and overhead demotes. Detection never regressed — both read 0.7
  against a 0.1 gate. The tests now pin BOTH verdicts plus the demotion, and are negative-controlled.
- **A sim charge re-profile no longer splits legs that charged the same table (§D-91).** `largest_same_code`
  settles it from each leg's `vclock_charge` telemetry — the SHA check was not baseline-scoped, and the three
  `fedbuff_round` sim legs ran dirty on values their SHA does not name. Negative-controlled: a re-profiled
  leg still splits, and identical charges never rescue a real code change.
- **ALL NINE baselines graded, two-sided, and eight of nine are 0-fail.** `fwdllm_it_oracular`'s block
  completed (3 real + 3 sim on `ceb119c6c`), and the leg that read "truncated at 5113s" was mid-copy, not
  truncated — `drop_truncated` was right to refuse it and right to accept it once complete.
- **The whole matched-window family is fixed, not just `utility`.** A source sweep found the same dead
  `real_coord` guard on `per_round_advance` and `overhead_residual`: all three computed a matched-budget
  window and then declined to use it on async, grading two unequal prefixes instead (§D-84). Removing it made
  `overhead_residual` fail on two baselines — with a 10% gate nobody had derived — so both it and
  `overlap_factor` are now floor-gated too. Their measured floors are **0.0-23.4%** and **0.0-2.8%** against
  flat 10% nominals: never calibrations. `test_no_rung_gates_a_matched_window_on_real_coord` sweeps the
  module so the pattern cannot return.
- **Calibration debt 45 → 41, and the whole clock family is out of it.** `per_round_advance`, `utility`,
  `overhead_residual` and `overlap_factor` all have measured floors now; the ratchet holds the count.
- **Analysis got ~4x faster, which is why any of this was tractable.** A nine-baseline floor sweep was
  >10 min per side, serial. `replicate_floor` now grades groups in a process pool (`--jobs`, default one per
  CPU) — group-level because each leg parses to ~0.7 GB and shipping that between processes costs more than
  the parse — and `load_agg_jsonl` / `load_trainer_jsonl_dir` disk-cache on file identity (size + mtime +
  parser version, LRU-capped, `FLAME_PARITY_CACHE_DIR=off` to bypass). Sweep is now **~2.5 min per side**.
  §D-89, now a preamble non-negotiable.
- **Floor profiles record their own provenance** (§D-87): `code_commit`, `nodes`, `source_runs`,
  `n_replicates`, `measured_at`, `span_axis`, `trace`, `floor_tool_version`, per side. A stale `pooled_from`
  is now dropped rather than merged forward, which is how an un-pooled baseline kept advertising a pooling it
  no longer had.
- **`felix_round`'s three fails are CLOSED — 62/3/27 → 63/0/29 — and not one was a sim defect.** All three
  were measurement defects in the checker, found by benching the rungs against config-identical legs before
  touching a gate:
  - **`per_round_advance`, `overhead_residual` and `overlap_factor` compared one leg's process WALL against
    the other's VCLOCK** — §D-73, on the three rungs the original fix never reached. Sim's wall runs ~3.4x
    its vclock, so every config-identical sim pair read a 64-74% "advance divergence" (ratio median 2.8-4.0,
    A 10.5-12.2s vs B 34-42s) and the family fired **9/10**. That number is what §A.3 cited as proof they
    were uncalibrated, so the evidence and the defect were the same bug (§D-85). Reading each side on its
    own clock takes the same pairs to 2-20%, and the control to **0/15, 0/15, 4/15**. It also makes them
    readable REAL↔real for the first time, which is where their floors now come from (§D-72).
  - **`utility` graded the FULL RUN on async**, because its matched-budget truncation was gated on
    `_real_intrinsic_clock` — a sync-only wall coordinate the matched window never reads (§D-84). It pooled
    24420 real against 20710 sim samples for KS 0.232 where the matched window reads **0.137**. Unequal
    prefixes are exactly what the truncation exists to prevent (§D-4, §D-75).
  - **`selection_detail`'s `tol_chosen` floor signal was being discarded.** `mean_chosen` averages a bimodal
    burst (§D-32) and was left on a hand-typed 5%; floor-gating it measured **11.7%**, but a line clearing
    the rung's ungradeable flag — correct when the COUNT was the only floor-gated field — threw the reason
    away and graded the 5% anyway. Now tracked per FIELD.
- **Fixed for the FAMILY, not rung by rung.** All six paired-clock rungs route through one `pair_clocks`
  helper, and `test_every_paired_clock_rung_reads_each_side_on_its_own_clock` runs two IDENTICAL sim legs
  through the whole ladder and fails any rung that reports a divergence. **Negative-controlled**: re-introduce
  the bug in one rung and the sweep fails; restore it and it passes. The audit that found the last two
  (`modeled_compute_advance`, `wall_disparity` — both DIAG, both reporting a leg's speedup ratio as a
  residual) is now that test's job, not a person's. §D-73 amended, 9 tests.
- **`per_round_advance` and `utility` left the calibration debt (45 → 43), and the ratchet holds it.**
  Their floors are measured by their own rungs across replicates: `round_advance_ks` reads 0.0-28.1% and
  `utility_ks` 0.0-19.8% against flat 0.2 gates that were never calibrations. Where the floor swallows the
  gate the rung now SKIPs with the number in its `reason` — `felix_round`'s `per_round_advance` at
  "floor 15.6% >= tolerance 15.0%" instead of a fail nobody could act on.
- **BLOCK 2 LANDED on all five, and blocker 1 is discharged.** `felix_it`, `fedbuff_it_unaware`,
  `fluxtune`, `fwdllm_it_unaware`, `fwdllm_it_oracular` — all 0-fail (current counts in §A.2),
  n=3 per side on one commit, two-sided floors. **No residual moved; only the gates did**, exactly as in
  block 1: every row gained skips, and `fedbuff_it_unaware`'s standing `v1` fail turned out to be a residual
  inside sim's own noise.
  **`felix_it` priced what a one-sided n=2 floor was worth: its `iters/bin` gate was sized on 4.8% and the
  measured two-sided floor is 18.9% — 4x, in the direction that manufactures fails** (§D-57, §D-61).
- **`fedbuff_round`'s 0.7% REAL floor WAS a lucky draw — §B.4's falsifier fired on the first extra draw.**
  A 4th real leg landed at `iters/bin` 13.01 over 198 bins against the 14.17-14.23 / 182-185 cluster, taking
  the real floor to **15.4%** against sim's 21.2%. There is no "real is structurally tighter than sim"
  mechanism to find; both sides now fit the hitting-time model's ~12% prediction, and the candidate
  explanation (sim's modeled-delay grid correlating pool composition) is dead without ever being tested.
- **`felix_it` is the mirror image, and it settles the question in general: REAL is the noisier side by 6-8x
  there** (18.9-19.0% real against 2.4-3.0% sim). Sim wider on `fedbuff_round`, real wider on four others,
  both zero on the pinned one. "Which side is noisier" is not a property of the simulator (§D-78, §E).
- **An orchestration script was splitting replicate groups, and it silently discarded the `--add-real` leg
  it exists to add.** `run_block.sh` only sequences `run_sequential.sh` and the analysis tools — every
  parameter it can vary is already in the grouping key — but `code_differs` saw a `.sh` under `expt_scripts`
  and called it run-affecting, so the 4th `fedbuff_round` leg was dropped as "other code" and `fedbuff_it`'s
  two blocks would not pool. Deny-listed; `run_sequential.sh`, which launches, stays run-affecting. §D-81,
  1 test. Consequences: `fedbuff_round` n=4 (the finding above) and `fedbuff_it` **n=6 real, floor unmoved at
  9.2%** — the first direct check that the max-pairwise estimator has converged (§D-57, §D-68).
- **`fwdllm_it` UN-POOLED, and the pair's "metric-for-metric identical" claim is dead.** With all three
  `_oracular` legs on disk, each name reproduces itself EXACTLY in both modes and the two names differ
  systematically: real 38 bins / `iters/bin` 9.64-9.67 (`_unaware`) against 40 / 9.20-9.22 (`_oracular`),
  sim 39 / 9.88 against 42 / 9.21. Pooling reported that offset as replicate noise and took a PINNED
  baseline's floor from **0.0% to 5.1%**; un-pooling took both rows 64/0/28 → **69/0/23**, because five rungs
  had been SKIPping on a floor that was not noise. `fedbuff_it` still pools — its legs interleave and the
  pooled 9.2% equals `_oracular`'s own. §D-86, 1 test.
  ⚠ **Cause not established, and the leading candidate is the KNOB, not the host** — the nodes are identical
  hardware (operator ruling) and the resolved configs differ in exactly `trackTrainerAvail`. Perfect 3/3
  reproducibility also rules out background load. The sim side cannot corroborate: each name carries its own
  charge profile and they differ materially (drain_tail 0.147s vs 0.125s), which moves sim cadence by itself
  (§D-50). §B.3 #1 settles it with one leg.
- **A dead sim leg graded as 23 pass / 67 skip rather than as dead.** An MQTT disconnect killed a
  `fwdllm_it_unaware` sim leg 8 minutes in; it still wrote a run dir, a snapshot and a telemetry directory.
  `replicate_floor` dropped it, `--control` pooled it, and it contributed 2 of 15 `terminal_state` and 2 of
  12 `field_coverage` control fails that were pure artifact. Quarantined under `experiments/_aborted/`.
  §D-83 — the completeness check belongs in the shared discovery helpers, not in one tool (§B.8).
- **BLOCK 1 LANDED: n=3 per side on one commit for four baselines, and it discharged blocker 1.** The sim
  floor exists for the first time. `fedbuff_it_oracular` — the row that had blocked criterion 1 with a
  +11.6% `v1` — cleared it (current counts in §A.2). `fedbuff_round` went 3 fails → 0 once the gate saw sim's own 21.2%
  spread. `fwdllm` is the negative control it was chosen to be: three real legs and three sim legs identical
  to 4 s.f., both floors 0.000. **No residual moved; only the gates did.**
- **Truncation was judged on the wrong clock, and it cost a replicate.** `achieved_span_s` read aggregator
  wall time, which for a SIM leg measures host load — sim skips real waits. A complete `fedbuff_round` sim
  leg (vclock 7199s, identical to its siblings) was dropped for finishing 6% faster in wall time, and it was
  the extreme draw, so the floor it left behind was biased low: 6.8% against the true 21.2%. Now judged on
  `vclock_now` where a leg has one, with a group-level axis so the two are never mixed, and the drop message
  names the axis. Shared with `--control`, which had lost two of three sim pairs the same way. §D-77.
- **Floor gating is TWO-SIDED.** Profiles carry `sim_metrics` alongside `metrics`, merged rather than
  clobbered so the two modes' passes compose, and the gate takes the per-metric max. Sizing on real alone had
  put `fedbuff_round`'s `v1` gate at 2.1% — 3x a 0.7% real floor — while the sim side reproduced itself only
  to 21.2%, failing a 6.9% residual no code change could ever have closed. `--mode sim` now needs
  `--profile-out` or the sim side is measured and discarded. §D-78, 5 tests.
- **A charge re-profile no longer splits REAL replicate groups.** `sim_charge_profiles/` reaches a sim run
  and nothing else (§F-1), but `code_differs` was path-based and mode-blind, so committing a profile split
  every real group across it — legs byte-identical in every input they consumed. `code_differs` and
  `largest_same_code` now take the mode. §D-79, 3 tests.
- ⚠ **`parity_floors/` and `sim_charge_profiles/` were CLOBBERED by an scp between nodes** and had to be
  re-derived. Each node holds all nine files but freshly writes only its own baseline's, so copying the
  directory wholesale is last-writer-wins: three of four fresh floors and three of four fresh charge profiles
  were lost, including the ones this node had computed itself. Both are pure functions of the run dirs —
  re-deriving reproduced sheph's `felix_round` profile byte-for-byte. §D-80.
- **The vclock/time family is calibrated, and `terminal_state`/`throughput` are now READABLE real↔real for
  the first time.** §D-56 called them uncontrollable; that was true of the RUNG, not the quantity (§D-72).
  A `same_mode` switch lets a clock rung grade two legs of one mode, each on its OWN clock. Measured
  same-code, real's time-to-N spreads **0.0% pinned to 6.7% unpinned** and trainers-at-N to 1.9%, against
  hand-typed 8%/5% gates nothing had ever sized. `throughput` was floor-gated on `committed_bins` — work
  VOLUME, while it grades a time RATIO — and now carries its own rung-measured floor; it also states
  `decided_on`, because it switches between `rel_diff` and `matched_window_rel_diff` by baseline and a floor
  read off the other one grades a window it never decided on (§D-71). The unreadable list drops 6 rungs → 3,
  and the 3 left are genuine sim-only invariants. 12 tests.
- **A sim↔sim pair compared one leg's PROCESS WALL CLOCK against the other's vclock — a 51.3% "residual"
  between two legs of one config, all of it unit mismatch (§D-73).** Every leg now reads its own vclock when
  it has one; only the B side of a real↔sim pair is REQUIRED to carry one, so a broken sim run still bails
  loudly. This sat directly on the measurement §B.3 #1 exists to produce: without it every sim-side number in
  the run batch would have been garbage.
- **One cadence number stopped voting three times.** `v1`, `cohort_sequence.count` and `selection_detail`'s
  count are ONE measurement under ONE floor-sized gate since §D-64 — but all three still voted, so a single
  number produced three fails on each `fedbuff_it` row (6 of the board's 12). The count now reports with a
  sub-verdict and `..._owned_by: v1_iter_per_data_id`; the two rungs keep gating what they uniquely own
  (set/order/composition, chosen/inflight). On an identity-gated async baseline the count was
  `cohort_sequence`'s ONLY enforced bound — which is exactly how one number became three fails. 4 tests.
- **The CONTROL now enforces same-CODE, as the floor tool already did (§D-65, §D-70).** Without it the two
  disagreed **4.5x on the same quantity** — 13.6% real↔real time-to-N against a 3.0% same-code floor — and
  the control was reporting code drift as pipeline noise. `--any-code` opts back out. 2 tests.
- **`fedbuff_it_oracular` and `_unaware` now POOL their floor (§D-63), rows still separate.** One config at
  syn_0 (`trackTrainerAvail` proven inert), so each name was graded against half its own evidence and
  `_unaware` had n=1 — no floor at all. Declarative syn_0-only alias that Phase 2 deletes; naming either
  member selects the whole group, and the pooled floor is written to BOTH members' files because the checker
  looks a floor up by the baseline it is grading. 9 tests.
- **Legs are now paired on the CODE THEY RAN, not on recency (§D-70) — and this is the finding that
  reframes every floor in this doc.** Every run dir already recorded its commit in `snapshot.yaml` (30/30
  legs); nothing read it. Grouping on it shows **`fedbuff_it_unaware`'s three real legs sit on three
  different code versions** — so its 13.8% floor, the largest on the board, is code drift, not replicate
  noise. It also killed a false finding of my own: `fwdllm`'s two sim legs straddle a charge re-profile and
  pooled to an 18% "sim floor" on a baseline whose real floor is 0.0% — the stale-profile leg reads +18.4%
  against real, the current one −0.7%. That is §D-50 quantified on the one baseline where nothing else can
  move. Comparability is decided by DIFFING the two trees for run-affecting paths, not by SHA equality, so a
  docs-only commit between two legs does not split them; the deny-list is re-derived from
  `run_sequential.sh` by a test so it cannot drift. `run_parity` now prefers a same-code real and prints
  ⚠ CODE DIFFERS when none exists. 13 tests.
- **`v2` may now LOOSEN to at most 2x nominal, and `selection_detail`'s count shares `v1`'s gate.** v2's
  floor (1.8-1.9%) had caught up with a 2.0% nominal that was never calibrated; a bounded widen to 4% beats a
  SKIP because a loose gate still catches a large regression. `selection_detail` was failing ONLY on
  `rel_diff_n_selections` while `rel_diff_chosen` and `rel_diff_inflight` read **exactly 0.0**, so the count
  is `v1`'s number a fourth time (§D-64). ⚠ Block 1 found `rel_diff_chosen` is NOT always 0: `felix_round`
  reads 0.132 (real 2.77 / sim 2.41) over only 22 vs 27 selections, and that bound now gates the rung. The
  "chosen/cycle is 10.01 on every leg of every baseline" claim held for the round-robin baselines only.
  9 tests.
- **The real↔real CONTROL is now `run_parity.py --control`, and §A.3 is computed rather than hand-kept.** It
  grades every pair of every config's legs in either mode, prints the per-rung fail rate, and **separates a
  BAIL from a FAIL** (§D-56) by reading the result rather than a static list — so the same rung correctly
  reads unreadable real↔real and readable sim↔sim. Groups and drops truncated legs through
  `replicate_floor`'s own helpers, so a control pair and a floor always cover the same legs. ON-only by
  default (§D-45); exits 0 whatever it finds (§D-51). 11 tests.
- **The floors were measured on the wrong window, and fixing it took the board from 21 fails to 9 with no
  simulator change.** `replicate_floor` computed each leg's FULL-run mean while the rungs grade the matched
  prefix. It now measures every windowed floor **by calling the rung itself** — nothing left to reimplement
  (§D-53, §D-65). Unpinned floors rose 1.4-2.0x (`fedbuff_it_unaware` 9.9 → 13.8%), the pinned ones did not
  (`fwdllm` 0.3 → 0.0%, the negative control). That alone took `fedbuff_it_oracular`'s `v1` gate 7.56% →
  14.7% and its +11.6% green — the row that had been blocking criterion 1, closed without pooling. 6 tests.
- **`cohort_sequence.count` and `v1b`'s cumulative mean ARE `v1`'s number** — identical to 3 decimals on all
  nine baselines over 19 control pairs — but were graded at their own hand-typed 5% while `v1` was
  floor-gated, failing the same measurement one rung passed. They now share `v1`'s tolerance, the way the
  throughput family shares one (§D-64, §D-22). `cohort_sequence` went 10 of 19 real↔real fails to **0**.
- **`v1b`, `v1c` and `convergence` had no floor behind their gates; all three now do.** The floor mechanism
  grew per-FIELD `min_abs` so it can size absolute tolerances — MA deviation in iterations, slope per 100
  units, accuracy points (§D-66). Measured real↔real, the pipeline's own noise is 2-5x each nominal gate
  (v1b max dev 7.65 vs 0.75; |λ| 0.198 vs 0.05; acc 9.85 pts vs 5). All three went to **0 of 19**, and
  `v1c`'s docstring calibration ("max |t| 1.61 over six pairs") is falsified at n=3: |t| reaches 5.72. On the
  pinned baselines every one of these floors is 0.000, so the rungs stay fully enforced where they can
  measure. 8 tests.
- **`fedbuff_it_oracular` and `fedbuff_it_unaware` are ONE config at syn_0** — their resolved configs differ
  in `trackTrainerAvail` alone, which every availability rung reads inert (pool 100/100, reduction 0.0). They
  pool their floor (§D-63). ⚠ The 13.8% / 8.1% spreads this entry used to quote were cross-code (§E).
- **The mechanism behind the whole unpinned-cadence spread is named: an un-plateaued variance gate.**
  `var_threshold` 0.30 against an achieved mean variance of 0.77-1.89 — **2.6-6.3x** — on all nine ON runs, so
  every commit fires on a noise dip and `iters/bin` is a HITTING TIME, not a converged quantity (§D-62). Where
  a barrier or `max_iterations_per_data_id` truncates it the spread is 0; where nothing does, 3-14%. This is
  `fluxtune_contributions.md` §8 F7 measured on the parity board — one phenomenon, not two, and it makes S2
  outrank S1 on impact (§B.3).
- **Dynamic KC, the iteration cap and the aggregation method are all REFUTED as culprits** (§E). Dynamic KC
  never ran; the cap is fluxtune-only and is a confound rather than a cause; the aggregation rate's
  within-method spread exceeds its between-method spread.
- **A 3rd real leg refuted the board's old headline, "sim over-iterates per data-bin"** — residuals flipped
  sign with which replicate they were graded against (§D-58, §E). Exit criteria 2-4 are MET.
- **`fwdllm_it_oracular` closed the ledger: 69/0/23 first time, all nine baselines complete.** It is also the
  sibling `fedbuff_it_oracular` never had — and at syn_0 it is metric-for-metric IDENTICAL to
  `fwdllm_it_unaware` (bins 40, cycles 376, iters/bin 9.20, var 0.8512), i.e. oracular tracking is inert at
  100% availability, as it must be. That makes "why is it NOT inert on `fedbuff_it_oracular`" a testable
  question for the first time — and §B.2 then answered it: the oracle is inert at syn_0.
- **I-1 CONFIRMED on node A, and it is now the critical path — not parity.** Both 14400s `fluxtune` legs
  collapse to chance (peak 84.7/85.3 → **25.0/25.2**), and `||Δ||/||w||` steps ~2.5x AT the round-2 boundary
  while accuracy is still 82/76, then climbs to ~0.03 as accuracy falls. Ratio rises before accuracy does;
  the flat-ratio falsifier did not fire. The trigger is the LAP BOUNDARY, not gradual drift. Owned by
  fluxtune_contributions.md §8 / S1 damping, but it gates the experiment runs (§B.0, §D-60).
- **`run_parity.py` paired on flag but not on RUN LENGTH.** Node A's 4h reals became the "latest real" for a
  2h sim leg, dropping `fluxtune` to 73/3 at 53.5% budget coverage — the same class as the OFF/ON trap, one
  term over. Now matches on `(jvp_eval_mode, max_runtime_s)` and names which term differed on a skipped real.
  4 tests (§D-59).
- **`replicate_floor.py --profile-out` would have written `fluxtune`'s floor from the 4h group** while the
  rung grades a 2h pair — "longest ON group wins" broke the moment one baseline had two ON durations. Added
  `--duration` plus a warning naming the ambiguity. 4 tests (§D-53).
- **The 2h floors transfer to 4h — on the PINNED baseline only.** `fluxtune`'s per-bin segment-level spread is
  flat across quintiles at both durations and the whole-window spread SHRINKS with length (0.83% → 0.38%). No
  accumulating divergence. The unpinned baselines have no 4h replicate; §D-52 says do not assume (§A.5).
- **Cross-mode gradient bit-identity is the localization tool that made this readable.** The two barrier
  baselines are bit-identical for their whole run (`fwdllm` 420/420, `fwdllm_it_unaware` 410/410) and
  `fwdllm` reads `v2` +0.00% real↔real too — the stack is bit-reproducible, so the whole spread is the async
  admission loop amplifying wall-clock luck. Every `c > agg_goal` baseline matches for part of the first
  quintile, then reads exactly 0 forever (§D-54).
- **The stale-charge hypothesis is CONFIRMED and closed.** Re-profiling took `fwdllm` from +18.0% to +0.00%
  and `fedbuff_round` from +12.8% to +7.4%. A stale charge was the whole residual on one baseline and half of
  it on the other (§D-50, §E).
- **Two mechanical guards for the CH stage, both negative-controlled on live data.** (1) The launch preflight
  BLOCKS a sim leg whose charge profile predates any real of that baseline trained under the same
  `jvp_eval_mode` — it fired on all four stale profiles, then went 21/21 green after re-profiling, and does
  NOT fire on a deliberate flag-OFF control. `--force` overrides. (2) `run_parity.py` pairs the sim leg with
  the latest COMPARABLE real, not the latest outright — the `felix_it` OFF-control trap is now structural
  rather than a note in this doc. 3 tests; the preflight half lives in a shell heredoc and has no importable
  test (§B.8).
- **The charge profile is a first-class parity stage, not a detail — it alone took `felix_round` 70/4 → 73/1.**
  Re-profiling from its own two ON reals, with no code change and the same real leg, took `v1` 5.4% → 0.0% and
  flipped `cohort_sequence`/`v1b` green. It killed the grad-pool hypothesis the cadence family had been
  charged to (§E) and became §A.1's CH stage and §D-50. `var_calc_audit` and `diff_var_pool.py` stay in the
  tree, OFF and now unused.
- **`felix_it` has an OFF band at last: 83.28 / 84.72**, sitting 1.0-3.4 pts ABOVE its ON band — the flag's
  only measured accuracy cost, and it does not follow the aggregation rate (§A.4, open in §B.4).
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
  provenance gap tracked in §B.8. 4 tests.
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
