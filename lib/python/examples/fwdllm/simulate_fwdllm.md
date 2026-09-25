# FwdLLM — Real↔Sim Parity

> **DEPRECATED — reference only.** The single source of truth is [ROBUST_FL_READINESS.md](../_metadata/ROBUST_FL_READINESS.md) → [FLUXTUNE_READINESS.md](../_metadata/FLUXTUNE_READINESS.md). This file only gets trimmed from here on as its content moves there; don't add to it.

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 ported fedbuff/felix-lineage
baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3).
Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf, sim barrier redesign, delay-factor
calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/
gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

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
| `fedbuff_round` | **63/0/30** | — | 15.4% / **21.2%** | n=4 both sides; `utility` ENFORCED and green at 0.111 |
| `fedbuff_it_unaware` | **63/0/29** | — | 9.2% / **14.7%** | pooled n=6/7; 3 rungs went enforced → SKIP |
| `fedbuff_it_oracular` | **62/0/29** | — | 9.2% / **14.7%** | same; its `utility` fail became a SKIP, not a pass |
| `felix_round` | **62/0/30** | — | 15.9% / **8.2%** | its three fails were all checker defects (§G) |

**ZERO FAILS ON ALL NINE.** `fedbuff_round`'s standing fail was the DRAW and was never about sim: over all 12
real×sim cells four rungs fail 1-2 cells each and none systematically, and pooled-sim vs pooled-real reads
**0.040**. With a 4th sim leg its `utility` now reads **0.111 against a 0.2 gate, ENFORCED and unanimous over
all four real legs** — the strongest form of green on the board.

⚠ **`fedbuff_it_*`'s `utility` is a SKIP, not a pass — do not read it as one.** The 4th `_oracular` sim leg
took the pooled sim floor 0.185 → **0.314**, past the 0.2 nominal, so the rung is now UNGRADEABLE: sim's own
replicate spread on that statistic exceeds any residual it could report. The residual did also collapse
(0.220 → 0.048), so nothing is hiding under the SKIP — but the rung no longer certifies anything.
**The 0.314 is a WITHIN-name pair** (`_oracular`×`_oracular`, max cross-name 0.260), so it is genuine sim
noise and does NOT rest on the pooling premise FLUXTUNE FT-N2 has under review.

⚠ **Adding legs cost enforcement on that family**: `overhead_residual`, `per_round_advance` and `utility` all
went enforced-PASS → SKIP on BOTH `fedbuff_it` names as the n=7 floors widened. That is the honest direction
(§D-24) but it is six fewer enforced verdicts, and it is the reason those two rows lost rungs.

**The board improved this batch by fixing MEASUREMENT, not the simulator.** Nine rungs' verdicts changed and
**no simulator code was touched**: three rungs compared one leg's wall against another's vclock, three graded
the full run because a matched window was gated on a clock it never reads, and one had its floor signal
discarded. That is the good version of this outcome and the one to be most suspicious of — FLUXTUNE FT-N4.

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
| `fedbuff_round` | 4/4 | 15.4 / **21.2** | 14.5 / **19.9** | 14.6 / **20.0** | 6.3 / **9.8** |
| `felix_it` | 3/3 | **18.9** / 2.4 | **18.9** / 3.0 | **19.0** / 3.0 | **6.7** / 2.8 |
| `felix_round` | 3/3 | **15.9** / 8.2 | **15.6** / 7.9 | **15.6** / 7.9 | **7.6** / 6.4 |
| `fedbuff_it` (pooled) | 6/7 | 9.2 / **14.7** | 9.0 / **14.8** | 9.1 / **14.8** | 7.3 / 7.3 |
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
taking the real floor **0.7% → 15.4%**. FLUXTUNE open questions's falsifier fired on the first extra draw.

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

## §B  Runbook and calibration notes

> Status, queue, open questions and backlog moved to `FLUXTUNE_READINESS.md`. What remains: §B.1 runbook,
> §B.2 mechanism, §B.5-7 calibration notes.

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
queue (FLUXTUNE FT-N2).** What is measured is that the oracle removes NOBODY from the eligible pool — pool 100/100,
`eligible_pool_reduction` 0.0. What is NOT measured is that it changes nothing: at n=3 per name per side the
two `fwdllm_it` names differ deterministically (real 38 bins / 9.64 vs 40 / 9.20), on identical hardware,
with `trackTrainerAvail` the only difference in their resolved configs. Only the PINNED family can see this —
elsewhere a ~5% effect sits inside a 9-19% replicate floor. The availability rungs above compare real↔sim
WITHIN each name; none of them ever compared the two names (§D-88).

⚠ **Every cross-code number this section used to carry is deleted, not corrected.** The "sim is the noisier
side" ladder, the 8.1% / 25.7% sim↔sim spreads and the cross-graded residuals were all measured across three
commits (§E, §D-70). Block 1 replaced them with same-code n=3 on both sides (§A.4).

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
| **anything past the first lap boundary** | **4h+** | `fluxtune` trains cleanly through round 1 and collapses to chance in round 2 (FLUXTUNE FT-N4). A 2h leg cannot see it — every accuracy number in §A.4 is a round-1 number |

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
# hides `test_ladder.py` and `test_floor_gated_tol.py`, i.e. every test that guards the gates .
conda run -n dg_flame python -m pytest lib/python/tests lib/python/examples/fwdllm/expt_scripts \
    lib/python/examples/async_cifar10/scripts/parity -q      
```

---

## §D  Durable lessons — fwdllm diagnostic patterns

> Transferable invariants — patterns that must be followed. ≤30 words each; update in place, never append
> near-duplicates. A falsified hypothesis belongs in §E, not here. Gaps in the numbering are lessons merged
> into a neighbour; numbers are cited elsewhere, so they are never reused. Shared (non-fwdllm) patterns live
> in [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) "Durable lessons."

**D-1.** → moved to `ROBUST_FL_READINESS` T11.

**D-2.** → moved to `ROBUST_FL_READINESS` L14.

**D-3.** Porting a selector does not port its real↔sim timing parity — aggregator and trainer classes are
separate. Diff the destination against the shared base first.

**D-4.** → moved to `ROBUST_FL_READINESS` L10.

**D-5.** → moved to `ROBUST_FL_READINESS` R1.

**D-6.** A selector's parity record belongs to the selector+aggregator PAIR. Check which side owns a guard
before citing it as a reference.

**D-7.** A statistic over rate-scaled samples measures the rate, not the samples. Check both sides' input
to a shared formula before suspecting the formula.

**D-8.** A cache/reuse fast path can silently skip a guard the slow path enforces. Diff its return against
what the bypassed call would have filtered.

**D-9.** → moved to `ROBUST_FL_READINESS` R2.

**D-11.** → moved to `ROBUST_FL_READINESS` L15.

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

**D-24.** → moved to `ROBUST_FL_READINESS` L12.

**D-25.** → moved to `ROBUST_FL_READINESS` R6.

**D-26.** A progress key must be monotone in TIME before anything sorts, maxes or windows on it. Order by
event timestamp; a composite key can wrap out of order.

**D-27.** → moved to `ROBUST_FL_READINESS` L5.

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

**D-34.** → moved to `FLUXTUNE_READINESS` FT-L5.

**D-35.** A quantity inside a feedback loop has a residual that grows with run length, so no fixed tolerance
on its LEVEL holds at two durations. Gate the per-unit RATE against zero and read its t-stat — a rate sits in
an exponent, so a small rate gap buys a large count gap.

**D-36.** → moved to `FLUXTUNE_READINESS` FT-L6.

**D-38.** Pair per-bin samples at equal bin index before differencing: it cancels the shared training curve
and gives an honest within-run error bar. Only a replicate sees seed-level variance.

**D-40.** → moved to `ROBUST_FL_READINESS` T7.

**D-41.** Seeding fixes the stream you seeded, not arithmetic and not a stream some component draws from
privately. If the stream POSITION matches across runs, the divergence is below that layer.

**D-42.** A discrete choice over near-tied continuous values (argmax, sort, a threshold gate) turns
round-off into an O(1) difference. Look for one before accepting a divergence as irreducible.

**D-43.** Grade a finite-difference estimator by its condition number, not its formula — reduced precision
amplifies round-off by a factor you must measure, not assume small.

**D-44.** → moved to `ROBUST_FL_READINESS` L18.

**D-45.** A replicate floor must come from legs that differ ONLY in wall-clock luck. Swapping which replicate
a fixed comparison is graded against is the cleanest test of whether a rung measures anything.

**D-46.** → moved to `ROBUST_FL_READINESS` R3.

**D-47.** → moved to `FLUXTUNE_READINESS` FT-L9.

**D-49.** → moved to `ROBUST_FL_READINESS` L9.

**D-50.** → moved to `FLUXTUNE_READINESS` FT-L6.

**D-51.** → moved to `ROBUST_FL_READINESS` T10.

**D-52.** → moved to `FLUXTUNE_READINESS` FT-L4.

**D-53.** → moved to `ROBUST_FL_READINESS` L12.

**D-54.** One divergent cohort is permanent: the weight trajectories separate and no later step is comparable
again. Grade cross-mode bit-identity of a per-end quantity to find WHERE parity is lost, not whether.

**D-55.** → moved to `ROBUST_FL_READINESS` R7.

**D-56.** A rung that reads a sim-only field cannot be controlled real↔real — it bails, and the bail looks
like a fail. List which rungs the control can actually read before crediting any of its verdicts.

**D-57.** → moved to `ROBUST_FL_READINESS` L12.

**D-58.** A residual that changes SIGN with which replicate it is graded against is noise, not a mechanism.
Grade against every replicate before believing a direction.

**D-59.** → moved to `ROBUST_FL_READINESS` L18.

**D-60.** → moved to `FLUXTUNE_READINESS` FT-L8.

**D-61.** → moved to `ROBUST_FL_READINESS` L12.

**D-62.** → moved to `FLUXTUNE_READINESS` FT-L4.

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

**D-69.** → moved to `ROBUST_FL_READINESS` L13.

**D-70.** → moved to `ROBUST_FL_READINESS` L18.

**D-71.** A rung gating on several bounds can be calibrated on one and still fail on another. Read WHICH
sub-gate fired — and whether the rung reports the field it actually decides on.

**D-72.** A rung that bails for want of a sim-only field can still have a measurable floor: grade the REAL
side's own quantity across replicate legs. "Uncontrollable rung" does not mean "uncalibratable gate".

**D-73.** → moved to `ROBUST_FL_READINESS` L10.

**D-74.** A floor measured from the SAME legs the control then grades cannot fail: the gate is 3x the worst
pair it is graded against. A floor-gated control is a consistency check, never independent evidence.

**D-75.** Compare the same WORK, not each side's own full run. Where per-unit cost trends (2.1-2.7x across
quintiles here), unequal unit counts are unequal windows and the longer side's tail biases its mean.

**D-76.** One summary cannot state a skewed, trending distribution: sim matched real's MEDIAN to 1.0% while
running a 27% heavier p90. Report the quantiles; gate only once a replicate floor exists for the one you gate.

**D-77.** Judge a leg's completeness on the clock that measures its WORK. A sim leg's wall span measures the
host, so a complete leg reads truncated for running on a quieter node — and gets silently dropped.

**D-78.** → moved to `ROBUST_FL_READINESS` L12.

**D-79.** "Same code" is per-SIDE: an input only one mode reads cannot split the other's replicate group.
Diff the trees against what THAT leg actually consumed.

**D-80.** → moved to `ROBUST_FL_READINESS` T9.

**D-81.** A driver that only SEQUENCES the launcher is not run-affecting. Splitting replicate groups on it
discards the very leg it was edited to add — deny-list the orchestrator, keep the launcher.

**D-83.** → moved to `ROBUST_FL_READINESS` L18.

**D-84.** A guard gating a computed statistic on an input that statistic never reads is dead code with a
verdict attached. Check what the window is built FROM, not what the function happens to have in scope.

**D-85.** "Fails its own control" is only evidence once the control is known to measure the rung's quantity.
A broken control and an uncalibrated gate look identical on the board; check the control first.

**D-86.** → moved to `ROBUST_FL_READINESS` T8.

**D-87.** → moved to `ROBUST_FL_READINESS` L17.

**D-88.** "Inert" needs the comparison that would show it moving. A rung comparing real↔sim WITHIN each
config can never see a between-config effect; only a PINNED baseline resolves one below its own floor.

**D-89.** → moved to `ROBUST_FL_READINESS` L19.

**D-90.** → moved to `ROBUST_FL_READINESS` L13.

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
- **Every input-side audit of the spread** — trainer data hashes, dispatch order and rank, first-task
  iteration/model_version, seeding, client index, data partition, cohort choice — REFUTED by direct
  measurement, and all clean for one reason: the divergence enters BELOW them (§D-41).
- **Forcing inference mode in a probe** — what made the first probe runs read bit-exact on every arm. Not a
  clean measurement, a different model (§D-47).
- **"monotone drift shape means sim"** — FALSIFIED: a real↔real pair climbs harder than any real↔sim slope on
  that baseline. A diverging rate verdict on an uncapped baseline is not evidence about sim.
- **"SIM is the noisier side"** — REFUTED. The 8.1% / 25.7% sim↔sim spreads and the 2.2x per-leg sd behind it
  pooled legs across THREE commits. Same-code, both sides, seven baselines: real is wider on `felix_it`
  (18.9 vs 2.4%), `felix_round`, `fedbuff_it` and `fwdllm_it`; sim is wider on `fedbuff_round` (21.2 vs
  15.4%); both are 0.0 on the pinned baseline. Neither side is reliably noisier (§D-78, §D-70).
- **"Real is structurally TIGHTER than sim on `fedbuff_round`"** — FALSIFIED by a 4th real leg (0.7% →
  15.4%). Do not go looking for a mechanism behind a floor measured on three legs of a hitting-time process.

**Charges**
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
- **The AGGREGATION METHOD as the residual's driver** — REFUTED: `agg_rate_conf.type: old` spans −4.7% to
  +11.6%, bracketing `new`'s 0.0% to +7.5%. Within-method spread exceeds between-method spread.
- **`felix_it`'s 84.72% OFF leg as a lucky draw** — FALSIFIED; the replicate landed at 83.28, so the ON band
  really is below the OFF band on that one baseline (FLUXTUNE open questions).
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
- **A timing rung's gating function "moving"** — it never gated; the rung ranked exempted entries. Fixed at
  the source (§D-32).

**Measurement constructs**
- **Real's wall-vs-vclock anchor as the residual's cause** — REFUTED; residuals survive the change of
  coordinate, so they are in the mechanism, not the measurement.
- **The receive-ordering rework as the fluxtune skew fix** — LANDED but INERT; cleanup, not a parity fix.

## §F  Locked invariants (from async_cifar10, carried over)

> Always-true / always-do rules. Numbers are cited across this doc — keep them stable, don't renumber.
> Diagnostic *patterns* (see X → means Y) live in §D, not here.
>
> **APPEND-and-AMEND with OPERATOR APPROVAL, never silently** — state the evidence (code, telemetry, or a
> failing test), not an argument. **Amend in place, keep the number** (other sections cite them). Deleting
> needs more evidence than adding; prefer narrowing scope. A new invariant must be always-true in BOTH modes
> — one-baseline findings are §B/§G, see-X-means-Y patterns are §D, and if it needs a caveat it isn't one.
> A fix that contradicts an invariant is a STOP: resolve it with the operator before landing.

1. → moved to `ROBUST_FL_READINESS` L1.
2. → moved to `FLUXTUNE_READINESS` FT-L1.
3. → moved to `ROBUST_FL_READINESS` T5.
4. → moved to `FLUXTUNE_READINESS` FT-L3.
5. → moved to `ROBUST_FL_READINESS` R2.
6. → moved to `ROBUST_FL_READINESS` R11.
7. → moved to `ROBUST_FL_READINESS` R10.
8. → moved to `ROBUST_FL_READINESS` R8.
9. → moved to `ROBUST_FL_READINESS` L1.
10. → moved to `FLUXTUNE_READINESS` FT-L10.
11. → moved to `ROBUST_FL_READINESS` R5.
12. **Logical determinism is the parity definition.** Same trainers selected, same receipt order, same
    aggregations/rollbacks — differing ONLY in wall-clock. Prove it on the first data bin first.
13. → moved to `ROBUST_FL_READINESS` R11.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. No bare-scalar shortcut.
15. **Verify claims against code, not comments/docstrings.** A docstring claiming two functions are
    equivalent states intent, not a guarantee — diff them.
16. → moved to `ROBUST_FL_READINESS` T11.
17. **A rotating cohort settling at `c − agg_goal` surplus is the correct steady state** for `c ≫ agg_goal`
    fedbuff — don't drive the carried surplus toward 0.
18. → moved to `ROBUST_FL_READINESS` L9.
19. **No compute on the critical path for a log the run doesn't need.** Gate any log with non-trivial args
    behind a level check — an f-string evaluates its args even when the level would drop the line.
20. **Real/sim timing disagreement → fix real toward determinism, never inject noise into sim.** Sim's
    per-speed-class duration must stay clean; fix real's measured completion time at the source.

### §F.1 Version & commit invariants (confirmed in code, both modes)

21. → moved to `FLUXTUNE_READINESS` FT-L2.
22. **Commit == the update used for aggregation, at that instant — no lag.** Real: on ordered arrival. Sim:
    when the vclock reaches the update's `sct` (buffer-unlock IS the commit). Never commit on a later event.
23. → moved to `ROBUST_FL_READINESS` L5.
24. → moved to `FLUXTUNE_READINESS` FT-L2.
25. → moved to `FLUXTUNE_READINESS` FT-L2.
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

**Parity is fully landed.** All nine baselines are 0-fail on two-sided floors, one commit per block (§A.2,
FLUXTUNE FT-N4). Every transferable lesson the campaign produced lives in §D; falsified hypotheses live in §E. Full
entry-by-entry history of how each fix landed: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.
