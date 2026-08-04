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
> - **Run the real↔real CONTROL before naming any mechanism** (§D-55). It needs no sim leg and no new run,
>   and §A.3 shows five rungs that fail on config-identical real legs — a red cell is not evidence until the
>   control says it is.
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
python replicate_floor.py --mode real --duration 7200   # replicate spread -> DIST floor (§D-24); --duration
                                            # is REQUIRED once a baseline has ON groups at two run lengths
python replicate_floor.py --mode real --duration 7200 --profile-out ../parity_floors   # write the floors
python profile_sim_charges.py --real-run <real_dir> \
    --out ../sim_charge_profiles/<baseline>.yaml --only-observed   # re-profile ONE baseline's charges (§D-36)
```
**The real↔real CONTROL has no CLI yet** — it is `parity.checks.run_all_parity(realA, realB, ...)` over every
pair of a baseline's real legs, and it is the primary way a DIST verdict is read (§A.3, §D-55). Promoting it
to a `run_parity.py --control` flag is §B.3 #4.
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

### §A.1  Readiness ledger — COMPLETE, all nine

| | stage | artifact |
|---|---|---|
| **R** | ON real legs at 7200s — n=3 where the cadence is unpinned, n=2 where a barrier/cap pins it (§D-52) | `experiments/run_*_<b>_*_real` |
| **FL** | replicate floor from them, at the graded duration | `parity_floors/<b>.yaml` |
| **CH** | charge profile from THOSE reals | `sim_charge_profiles/<b>.yaml` |
| **SIM** | ON sim leg launched AFTER CH | `experiments/run_*_<b>_*_sim` |
| **CTL** | real↔real control on every rung | §A.3 |
| **GR** | graded row | `experiments/_parity_reports/` |

All nine baselines carry all six on the REAL side. `fwdllm_it_oracular` closed on node C and graded
**69/0/23** first time. ⚠ **The SIM side is n=1 everywhere** — no baseline has a sim replicate, which is what
§B.3 #1 buys. The pipeline is §B.1.

### §A.2  The board — and the control that reads it

`run_parity.py`, n=3 floors, duration-matched pairs. Per-pair JSON:
`experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`.

| baseline | N | pass/fail/skip | failing rungs | `v1` real↔**sim** | `v1` real↔**real** (all pairs) | verdict |
|---|---|---|---|---|---|---|
| `fluxtune` | 96 | **75/0/17** | — | +0.0% | 0.8% (n=2) | clean |
| `fwdllm` | 41 | **68/0/24** | — | +0.0% | 0.3% (n=2) | clean |
| `fwdllm_it_unaware` | 40 | **69/0/23** | — | +0.0% | — (n=2) | clean |
| `fwdllm_it_oracular` | 40 | **69/0/23** | — | +0.0% | — (n=2) | clean |
| `felix_round` | 195 | 73/1/18 | `v2` | +0.0% | 0.0 · 1.4 · 1.6% | INSIDE control |
| `fedbuff_it_unaware` | 185 | 71/1/20 | `conv` | +0.4% | 7.4 · 8.4 · 13.8% | INSIDE control |
| `felix_it` | 254 | 72/4/17 | `sel_detail` `cohort` `v1b` `conv` | +7.5% | 0.8 · 4.8 · 5.6% | INSIDE control |
| `fedbuff_round` | 173 | 70/4/18 | `v1c` `v2` `terminal` `conv` | **−4.7%** | 3.0 · 7.0 · 9.9% | INSIDE control |
| `fedbuff_it_oracular` | 183 | 65/7/19 | `thru` `sel_detail` `cohort` `v1` `v1b` `terminal` `commits` | **+11.6%** | 2.3 · 4.9% (own name)<br>0.7 … **13.8%** (pooled, §B.2) | INSIDE pooled control |

Budget coverage 93-100% on both sides everywhere; all nine passed the validity gate.

**Every row on the board is inside its own replicate spread.** On eight of nine that is visible from the
baseline's own three legs. The ninth, `fedbuff_it_oracular`, needed §B.2's finding that it is the SAME config
as `fedbuff_it_unaware` at syn_0 — pooled, its control spans 13.8% and the +11.6% falls inside.

⚠ **This is a "no known defect" verdict, not a clean bill of health.** Every floor here is real↔real, and the
one sim↔sim comparison that exists puts two legs of one config **8.1%** apart. Criterion 1 is not signable
until the sim floor is measured (§B.0 blocker 1, §B.3 #1).

**"Sim over-iterates on the unpinned baselines" is REFUTED as a general claim (§E).** `fedbuff_round`'s
residual **flips sign** with which real leg it is graded against (+6.9% on the n=2 pair, −4.7% on the n=3
one); `fedbuff_it_unaware` went +9.3% → +0.4%. Those were one-replicate artifacts, and only n≥3 could show it.

`fedbuff_it_oracular` looked like the exception — +11.6/13.8/16.2% against its three reals, which sit 2.3-4.9%
apart. **§B.2 dissolves it:** graded against its own name it has only three legs, but it is config-identical to
`fedbuff_it_unaware` at syn_0, and the pooled six-leg control spans 13.8%. Its apparently tight 2.5% floor was
an artifact of splitting one config's evidence across two names (§D-63).

**Where the loop is pinned, parity is EXACT.** The four barrier/cap-pinned baselines read `v1` +0.00%, and
`fwdllm` matches 420/420 gradient norms across modes. Nothing in the stack is irreducibly noisy: init,
partition, JVP direction and training are bit-reproducible; the whole spread is the async admission loop
amplifying wall-clock luck (§D-54).

### §A.3  Which rungs the control can and cannot read

Three of `fedbuff_it_oracular`'s seven fails — `throughput`, `terminal_state`, `total_commits` — **cannot be
controlled real↔real at all**: they compare sim's `vclock_now` against real's wall clock, and a real leg emits
no vclock, so the rung bails with `ok:false` rather than measuring anything (§D-56). Same for
`field_coverage`, `vclock_telemetry`, `sim_send_ts`, `inter_arrival_order`. Their real↔real "failure" is a
bail-out, not evidence — do not read it either way.

For the rungs the control *can* read, it fires on config-identical real legs this often:

| rung | fails real↔real | reads |
|---|---|---|
| `convergence` | **12 of 15 pairs** | acc diff 3.4-9.9%; every real↔sim residual (3.2-6.1%) is inside its own baseline's band |
| `cohort_sequence` | 9 of 15 | clean only on `felix_round` |
| `v1b_iters_moving_avg` | 6 of 15 | its 0.25 abs-dev tolerance has no floor behind it |
| `v1c_iter_drift_rate` | 5 of 15 | **sign-flips** real↔real on `fedbuff_round` (λ −0.116 vs +0.153) |
| `selection_detail` | 5 of 15 | `v1` rolled up |
| `v2_var_trajectory` | 3 of 15 | matched-window real↔real spans 0.06-4.5%; the 2% gate sits inside it |
| `v1_iter_per_data_id` | **0 of 15** | the one rung whose gate clears its own control everywhere |

**So `v1` is the only cadence rung that measures anything** — and once its control is pooled correctly (§B.2),
no row fails outside it. Every red cell on the board is a rung firing inside its own noise.

### §A.4  Floors, n=3 — and what the 3rd leg bought

`replicate_floor.py --mode real --duration 7200 --profile-out ../parity_floors`. Floors are inputs to the
checker, not a table: DIST tolerances tighten toward `3x floor`, never past 2% absolute, never looser than
nominal, and SKIP once the floor swallows the tolerance (§D-24, §D-36).

| baseline | `iters_per_bin` n=2 → n=3 | `mean_var` n=2 → n=3 |
|---|---|---|
| `fedbuff_it_unaware` | 4.9 → **9.9%** | 0.6 → 2.3% |
| `fedbuff_round` | 4.1 → 5.0% | 1.5 → 1.9% |
| `felix_it` | 2.3 → 3.0% | 2.4 → 2.4% |
| `fedbuff_it_oracular` | 1.5 → 2.5% | 1.9 → 3.0% |
| `felix_round` | 0.6 → 1.1% | 0.4 → **1.8%** |
| pinned four (`fluxtune` `fwdllm` `fwdllm_it_*`) | 0.3-0.8%, n=2 | 0.1-6.6%, n=2 |

**A 2-leg floor is one pairwise difference — a sample of size one, zero degrees of freedom, and biased low by
construction** since `_spread` is max-pairwise. `fedbuff_it_unaware`'s doubled; nothing inside n=2 could have
shown that. The estimator is monotone in n, so the inference is one-sided and sound: **a row still red on n=3
is a real finding, because the gate only got more generous** (§D-57).

**Peak accuracy, ON, matched span (§D-44).** Bands are 0.0-3.7 pts except `felix_round` 5.25; `felix_it` sits
1.0-3.4 pts below its OFF band and is the flag's one measured accuracy cost (open, §B.4). ⚠ These are 2h
numbers — see §B.0 on what happens at 4h.

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
> with falsifiers. §B.5-7 = tolerances, deliberate gaps, backlog.

### §B.0  Where this stands

> Parity's job is that **sim does not change the CONCLUSION**, not that every rung is green. Every claim here
> is comparative, so a residual identical on every baseline cancels out of a ranking.

| # | criterion | status |
|---|---|---|
| 1 | Every INV/EXACT rung green on all nine | **1 row short**, `fedbuff_it_oracular` — but §B.2 shows that row is **not established as a sim defect**: its +11.6% sits inside the 13.8% real↔real spread of its own config |
| 2 | Convergence + terminal state inside each baseline's own replicate band | **MET** — every real↔sim `conv` residual is inside its own real↔real band |
| 3 | Every remaining DIST residual is COMMON-MODE | **MET on 9 of 9** once the last row is graded against its pooled control (§B.2) |
| 4 | No residual correlates with a baseline-DISTINGUISHING knob | **MET** — the correlation §A.2 used to report was the replicate floor tracking the pin, not a sim bias |

**No parity criterion has a known failure left.** What blocks sign-off is not a red rung, it is that one
number was never measured — see blocker 1.

**Two blockers, neither of them a parity bug:**

1. ⚠ **The sim side has NEVER been replicated.** Every floor in this doc is real↔real; each baseline has
   exactly ONE sim leg. The first sim↔sim comparison ever run (§B.2) puts two legs of the *same* config
   **8.1%** apart — the same order as the residual we were calling genuine. Criterion 1 cannot be signed off
   against an n=1 sim. **Cheapest missing evidence in the project — a sim leg is 35-50 min** (§B.3 #1).
2. ⚠ **The critical path for the PAPER is a training bug, not parity.** Node A's two 14400s `fluxtune` legs
   both collapse to chance — peak 84.7/85.3 in round 1, **25.0/25.2** by the end of round 2 — and I-1's
   `||Δ||/||w||` sits flat at ~0.006 through round 1, **steps ~2.5x at the round-2 boundary** (to 0.014/0.018
   while accuracy is still 82/76), then climbs to ~0.03 as accuracy falls. The ratio rises BEFORE accuracy
   does; the flat-ratio falsifier did not fire. **I-1 CONFIRMED, reproducibly, on both legs**, and sharpened:
   the trigger is the **lap boundary**, a step change, not gradual drift. Owned by
   [fluxtune_contributions.md](fluxtune_contributions.md) §8 / S1-S2, not here — but EXPTS_CHARTER's E1 error
   bar sits at a ~3.9h peak (~234 min) and the collapse completes by ~232 min, so **no long `fluxtune`
   experiment run is meaningful until it is damped.** Plots:
   `experiments/run_20260804_0{03042,43301}_fluxtune_*/plots/server_update.png`.

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

# FL — from runs already on disk; no sim leg needed
python replicate_floor.py --mode real --duration 7200 --baselines <b> --profile-out ../parity_floors
python replicate_floor.py --mode sim  --duration 7200 --baselines <b>    # the sim-side floor, §B.3 #1

# CH + SIM + GR — one chain, never split
cp ../sim_charge_profiles/<b>.yaml ../sim_charge_profiles/<b>.yaml.bak && \
python profile_sim_charges.py $(ls -d ../experiments/*_<b>_n100_*_real | sort | tail -2 | sed 's/^/--real-run /') \
    --out ../sim_charge_profiles/<b>.yaml --only-observed && \
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only <b> --yes && \
python run_parity.py --yes --baselines <b>
```

**The CONTROL has no CLI yet** — it is `parity.checks.run_all_parity(legA, legB, ...)` over any two legs of
one config, in either mode, and it is now the primary way a DIST verdict is read (§A.3, §D-55). Promoting it
is §B.3 #4.

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

### §B.2  What the residual IS — the mechanism, and what is ruled out

**The mechanism, named and measured.** `var_threshold` is **0.30** on every baseline, but achieved mean
variance is **0.77-1.89 — 2.6x to 6.3x the threshold — on all nine ON runs.** The gate never reaches a
plateau, so **every commit fires on a random noise dip.** `iterations_per_data_id` is therefore not a
converged quantity, it is the **hitting time of a noisy process**, and its run-to-run spread is that hitting
time's variance. This is `fluxtune_contributions.md` §8 **F7** ("variance floor > threshold ⇒ bins commit only
on a noise dip") measured on the parity board — **one phenomenon, not two.**

| baseline | mean var | var/thr | iters/bin | `v1` real↔sim | hitting time truncated by |
|---|---|---|---|---|---|
| `fwdllm` | 0.77 | 2.6x | 7.4 | **+0.0%** | sync full-cohort barrier |
| `fwdllm_it_unaware` | 0.85 | 2.8x | 9.3 | **+0.0%** | sync full-cohort barrier |
| `fwdllm_it_oracular` | 0.85 | 2.8x | 9.3 | **+0.0%** | sync full-cohort barrier |
| `fluxtune` | 1.89 | 6.3x | 18.8 | **+0.0%** | `max_iterations_per_data_id: 20` + plateau policy |
| `felix_round` | 0.94 | 3.1x | 12.5 | +0.0% | — |
| `fedbuff_it_unaware` | 1.00 | 3.3x | 14.5 | +0.4% | — |
| `fedbuff_round` | 0.99 | 3.3x | 14.7 | −4.7% | — |
| `felix_it` | 1.00 | 3.3x | 15.0 | +7.5% | — |
| `fedbuff_it_oracular` | 0.95 | 3.2x | 13.8 | +11.6% | — |

**Where the hitting time is truncated the spread is exactly 0; where nothing truncates it, 3-14%.** That is
the whole board, in one line.

**What this RULES OUT — all four on telemetry already on disk, no run needed:**

| candidate | verdict | evidence |
|---|---|---|
| **Dynamic KC policy** | **NOT the culprit — it never ran** | `dynamic_kc.enabled: False` in the only baseline that has the knob (`fluxtune`). All three DK rungs SKIP: `dk1` reports *"DynamicKC disabled — constant K (real=10, sim=10)"*; `dk2`/`dk3` find no `dynamic_c` in cadence events |
| **Max-iterations cap** | **NOT a culprit — but it IS the board's biggest confound** | `max_iterations_per_data_id` exists on `fluxtune` ONLY. It does not cause the residual, it **hides** it: capping truncates the hitting time, which is why `fluxtune` reads +0.0% AND has a ~0 floor (§D-52). Every "fluxtune is clean" reading is confounded by it |
| **Aggregation method** | **NOT the culprit — it does not predict the residual** | `agg_rate_conf.type: old` spans **−4.7% to +11.6%**, which BRACKETS `new`'s **0.0% to +7.5%**. Within-method spread exceeds between-method spread. `grad_aware` reads +0.0% but is confounded by the cap above |
| **Oracular availability tracking** | **REFUTED — provably inert at syn_0** | `eligibility` pool **100.0/100.0**, ks 0.0 · `eligible_pool_reduction` **0.0/0.0** · `avail_timebase` max_rel_diff **0.0** · `duty_cycle_duration` mean_err **0.0**, frac_within_tol 1.0. The oracle removes nobody |

**And the consequence that reframes the board.** The resolved configs of `fedbuff_it_oracular` and
`fedbuff_it_unaware` differ in **exactly one key** — `trackTrainerAvail` — which the row above proves inert.
**At syn_0 they are the SAME experiment**, exactly as `fwdllm_it_oracular` ≡ `fwdllm_it_unaware` (which are
metric-for-metric identical: bins 40, cycles 376, iters/bin 9.20, var 0.8512). So their eight legs pool into
one replicate group:

| pooled group (ONE config) | `v1` spread over all pairs |
|---|---|
| **6 REAL legs**, 15 pairs | 0.7% … **13.8%** — 6 of 15 FAIL the 7.56% gate, including two same-name pairs |
| **2 SIM legs**, 1 pair | **8.1%** — the first sim↔sim replicate ever measured, and it fails too |
| real↔sim, all 12 pairs | 0.4% … 20.0% |

Cross-grading each sim against the *other* name's reals gives **4.7 / 6.9 / 9.3%** (sim_unaware) and
**7.6 / 15.4 / 20.0%** (sim_oracular): **the residual tracks WHICH SIM LEG, not which baseline.**

> **`fedbuff_it_oracular`'s +11.6% is therefore NOT established as a sim defect.** It is inside the 13.8%
> real↔real spread of its own configuration, and it is one draw from a sim distribution whose own spread is
> ≥8.1%. The real↔sim range (0.4-20.0%) and the real↔real range (0.7-13.8%) overlap almost completely. This is
> §D-57 again, on the side we never replicated. §B.3 #1 settles it for good.

### §B.3  RESUME HERE — the ordered queue

> Only item 1 needs runs, and they are sim legs: 35-50 min each, no real leg, no re-profiling.

1. **Measure the SIM-side replicate floor — the one thing that decides criterion 1.** Three sim legs of
   `fedbuff_it_oracular` (unpinned, the row in question) and three of `fwdllm` (pinned, the negative control).
   Same config, same charge profile — do NOT re-profile, it would invalidate every sim leg on disk (§D-50).
   **~2.5h on one node, the cheapest decisive evidence left.**
   ```bash
   bash run_sequential.sh --mode sim --max-runtime-s 7200 --only fedbuff_it_oracular --yes   # x3
   bash run_sequential.sh --mode sim --max-runtime-s 7200 --only fwdllm --yes                # x3
   python replicate_floor.py --mode sim --duration 7200 --baselines fedbuff_it_oracular fwdllm
   ```
   **PREDICTED:** sim↔sim ≈ 8-13% on `fedbuff_it_oracular` (matching its real↔real), ~0% on `fwdllm`.
   **IF CONFIRMED** → the residual is inside pooled noise, criterion 1 is met by re-grading against a two-sided
   floor, and **parity is DONE**. **FALSIFIED IF** sim↔sim ≈ 0 while real↔real is 13.4% — then sim is
   deterministic, the offset is real, and the mechanism hunt restarts at §B.4's first entry. Either outcome
   closes it; today's n=1 cannot.
2. **Pool the two fedbuff_it baselines and re-grade as one config.** They are the same experiment (§B.2), so
   `parity_floors/fedbuff_it_oracular.yaml` should come from all six legs, not three — its `iters_per_bin`
   floor goes 2.5% → 13.4%, which alone moves the row inside its gate. Needs a `--pool`/alias in
   `replicate_floor.py`. **Operator call:** is pooling two *named* baselines acceptable book-keeping, or
   should `fedbuff_it_oracular` be dropped from the syn_0 board as a duplicate? It is genuinely distinct only
   under a scarcity trace (Phase 2).
3. **Re-derive `convergence`, `v1b`, `v1c`, `cohort_sequence`, `v2` against the control (§A.3).** All five fire
   between config-identical REAL legs — `convergence` on 12 of 15 pairs, `v1c` sign-flips. They are noise
   generators, not gates. **Keep `v1`'s gate as-is: 0 of 15.** Code + tests, no runs.
4. **Promote the control to `run_parity.py --control`.** It is now the primary reader of every DIST verdict
   and exists only as a scratch script. Should grade every pair of a config's legs in EITHER mode and print
   the per-rung fail count §A.3 tabulates by hand today.
5. **Fix `replicate_floor.metrics()` to grade the rung's window** — it measures each leg's FULL run while `v1`
   grades the matched prefix, so floors are 1.0-2.6x too small (§D-53). `fwdllm`/`fluxtune` must stay identical
   between the two estimators — the built-in negative control.
6. **Sign-off re-grade**, then the experiment runs (`paper_expts_fluxtune/EXPERIMENTS.md`), sim-only.
   ⚠ **Gated on I-1 damping (§B.0 blocker 2), not on parity.**

**Hand back to `fluxtune_contributions.md` §8 — two things this batch earned it:**
- **F7 is CONFIRMED at N=100 α=1 on all nine ON baselines**, not just fluxtune: var/threshold **2.6-6.3x**.
  Promote its evidence from the charter reference to a nine-baseline measurement.
- **S2 (variance-gate recalibration) now has a SECOND, independent payoff, and it outranks S1 on impact.** It
  was scoped as a stability fix; §B.2 shows the same un-plateaued gate is what makes `iters/bin` a hitting
  time — i.e. it is also the source of the replicate floor blocking parity criterion 1. **Predicted:** a gate
  that commits on the plateau collapses the real↔real spread from 13.4% toward `fluxtune`'s ~0. That is a
  cheap falsifiable read-out for S2 that does **not** require an accuracy win to interpret.

⚠ **Do not re-run the refuted (§E).** The tie-break, redispatch stagger, cohort-composition bias, the general
"sim over-iterates" claim, and now the oracular-tracking story each cost one telemetry pass to kill.

⚠ **`run_parity.py` exits 1 whenever any rung fails** — the normal outcome (§D-51). Never mid-`&&` without
`|| true`.

⚠ **Moving run dirs off a node does NOT bring `parity_floors/` or `sim_charge_profiles/`.** Both are
re-derivable from the reals; do it before grading or every DIST rung grades at nominal tolerance.

### §B.4  Open questions — each with its falsifier  *(KEEP: not yet tasks)*

State the prediction BEFORE the run; a hypothesis that can only be confirmed is not one (§D-9).

- **If §B.3 #1 falsifies (sim deterministic, real not), the asymmetry IS the finding.** Sim replays a modeled
  delay grid while real draws fresh wall-clock luck every leg, so a deterministic sim against a 13.4%-spread
  real is the *expected* shape — and "parity" would then mean sim landing inside real's distribution, not on
  its mean. **That reframes the comparison and needs an operator ruling before any fix.** Do not start a
  mechanism hunt without it.
- **`felix_it`'s ON accuracy drop is baseline-specific.** Its ON band sits 1.0-3.4 pts below its OFF band while
  `felix_round` — same aggregation rate — improved. **FALSIFIED IF** a second loss-derived baseline degrades
  ON. Low priority; seven other bands support the flag. Do not re-open the shared-mask JVP.
- **`fedbuff_it_oracular` may not belong on the syn_0 board at all** — provably identical to
  `fedbuff_it_unaware` under full availability (§B.2). It earns a row only under a scarcity trace (Phase 2).
  Operator call, tracked in §B.3 #2.

### §B.5  Tolerances and rung gaps

- **§B.3 #3 supersedes every "recalibrate against the floor" item here.** The floor was the wrong reference:
  it is measured on the wrong window (§D-53) *and* it is a 2-leg point estimate on the pinned baselines. The
  real↔real control on the rung's own window is the reference. Re-derive `conv`/`v1b`/`v1c`/`cohort`/`v2`
  against it; leave `v1` alone.
- **`v1c` is a bin short of the power to see the drift it exists to catch** — 10 bins, `fedbuff_it_oracular`
  reaches t=3.29 against t_crit 3.36 and reads "flat" while its residual is real. Raising `n_bins` is the fix,
  but its real↔real calibration is falsified (§E), so re-derive against a control pair first.
- KS-only rungs unguarded against a level shift (all clean on live data): `dk1_agg_goal_trajectory`,
  `dk2_dynamic_c`, `dk3_eligible_ends_metric`, `eligible_speed`, `v3_cached_v_pool`.
- Thin ABSOLUTE budgets: the `fwdllm` family grades N=40-41 against 173-254 elsewhere, at 93-95% coverage.
  All four rows are 0-fail so nothing is masked, but coverage percentage cannot catch thinness; an absolute-N
  floor is proposed, threshold not chosen.
- `step_timing_breakdown` / `agg_step_timing_breakdown` are REPORT-ONLY wherever the sim clock discards the
  span (§D-31) — the designed steady state. Watch `charge_coverage` instead.
- Do not re-tune `redispatch_turnaround` (§D-14, §E).

### §B.6  Known-and-deliberate

- Real's over-`c` slot read is drain lag, not a §D-27 conflation (§E, §G). Telemetry only. Low priority.
- Real publishes no `_agg_slot_holders_ref`, so `_cap_dispatch_to_concurrency` falls back to the IDENTITY set.
  Harmless today; publishing one would let real dispatch into slots it now withholds — owed its own A/B.
- Sim's in-flight bookkeeping is split across six sets and should be one per-end state machine. The slot⇄guard
  split did the CAPACITY half; IDENTITY is still ad-hoc. Never bundle with a correctness fix.
- Base `asyncfl/top_aggregator._sim_hold_busy_slots` deliberately untouched — no `_sim_committed` term, so it
  never had the conflation. Re-check if async_cifar10 shows the same under-fill.

### §B.7  Backlog

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
- **`convergence`, `v1b`, `v1c`, `cohort_sequence` and `v2` as evidence about sim on an unpinned baseline** —
  all five fire between config-identical REAL legs (`convergence` on 12 of 15 pairs; `v1c` sign-flips). Only
  `v1` clears its own control everywhere (§A.3). Do not open an investigation off the other five.
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
- **The last "genuine residual" did not survive its own control either — and the SIM side turns out to be
  unreplicated.** `fedbuff_it_oracular` and `fedbuff_it_unaware` resolved configs differ in ONE key,
  `trackTrainerAvail`, which every availability rung reads inert at syn_0 (pool 100/100, reduction 0.0). They
  are one config, so their 8 legs pool: **6 reals span 13.8%** (6 of 15 pairs fail) and the **2 sims — the
  first sim↔sim replicate ever run — span 8.1%.** The +11.6% sits inside both. Cross-grading shows the
  residual tracks WHICH SIM LEG, not which baseline (§B.2, §D-61, §D-63).
- **The mechanism behind the whole unpinned-cadence spread is named: an un-plateaued variance gate.**
  `var_threshold` 0.30 against an achieved mean variance of 0.77-1.89 — **2.6-6.3x** — on all nine ON runs, so
  every commit fires on a noise dip and `iters/bin` is a HITTING TIME, not a converged quantity (§D-62). Where
  a barrier or `max_iterations_per_data_id` truncates it the spread is 0; where nothing does, 3-14%. This is
  `fluxtune_contributions.md` §8 F7 measured on the parity board — one phenomenon, not two, and it makes S2
  outrank S1 on impact (§B.3).
- **Dynamic KC, the iteration cap and the aggregation method are all REFUTED as culprits** (§E). Dynamic KC
  never ran; the cap is fluxtune-only and is a confound rather than a cause; the aggregation rate's
  within-method spread exceeds its between-method spread.
- **The n=3 batch REFUTED the board's headline claim.** With a 3rd real leg on the five unpinned baselines,
  "sim over-iterates per data-bin" collapsed: `fedbuff_round` flips SIGN (+6.9% → −4.7%), `fedbuff_it_unaware`
  goes +9.3% → +0.4%, and on eight of nine rows the real↔real spread ENVELOPS the real↔sim residual (§A.2,
  §D-58). The ninth fell to the pooling finding above. Exit criteria 2-4 are MET.
- **The real↔real control was extended to EVERY rung, and it disqualified five of them.** `convergence` fails
  on 12 of 15 config-identical real pairs, `cohort_sequence` 9, `v1b` 6, `v1c` 5 (sign-flipping), `v2` 3 —
  while `v1` fails **0 of 15**. Only `v1` measures anything on an unpinned baseline (§A.3, §E).
- **Three rungs CANNOT be controlled real↔real at all** — `throughput`, `terminal_state`, `total_commits`
  compare sim's `vclock_now` to real's wall clock, so with a real leg in the sim slot they return `ok:false`
  without measuring. Their real↔real "failure" is a bail-out. Same for `field_coverage`, `vclock_telemetry`,
  `sim_send_ts`, `inter_arrival_order` (§D-56).
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
  test (§B.7).
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
  provenance gap tracked in §B.7. 4 tests.
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
