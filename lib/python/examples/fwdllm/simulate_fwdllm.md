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
> - **Once an issue is isolated and a fix proposed, REPRODUCE IT ON THE BENCH — never validate a hypothesis
>   with an FL run first.** Write a throwaway script that faithfully reproduces the issue, in the real code
>   path (import the actual function; do not reimplement it — a reimplementation proves nothing about the
>   system). Then, in order:
>     1. **Confirm the control reproduces.** If the unfixed arm comes back clean, the SCRIPT is wrong, not the
>        code — fix the repro before touching anything else. A fix credited against a control that never
>        showed the bug is worthless, and this is the step that gets skipped.
>     2. Apply the candidate fix in the script and A/B it against that control.
>     3. Only if it works: implement in the real code behind a flag, then spend ONE run to validate.
>   Extends §F-8's telemetry→instrument→run ladder with the missing rung: telemetry on disk → **bench repro**
>   → run. A bench arm costs minutes and isolates one mechanism; an FL pair costs 2-4h and carries every
>   confound in the stack, so a null result from a run rarely tells you *which* thing was wrong. Worked
>   example: `expt_scripts/probe_jvp_determinism.py` (H12) — 4 arms × 8 concurrent processes, no aggregator,
>   no trainers, and it refuses to credit any arm when the control comes back clean.
>   **Match the bench conditions to the mechanism**, or the control will not fire: H12 needed *concurrent
>   processes* because co-location is what varies GPU kernel selection — repeats inside one process reuse a
>   single kernel plan and read bit-exact.
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

**Latest run per baseline** (`run_parity.py`; ✓/✗/– = pass/fail/skip; PARITY.md §F). **Mixed durations — rows
do NOT compare to each other.** Five baselines have a 7200s pair; four have only the 1200s Phase-0 smoke
(N=5-38), far below scoreboard strength. `fwdllm_plus` has no run dirs on disk.

| baseline | run pair | dur | pass/fail/skip | N | cohort | vclock | K4 | slots | sbias | thru | commits | terminal | V1c | V1 | V2 | U3 | S2 | conv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260801_030932`/`_150819` | 7200s | **77/0/16** | 94 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm/syn_0 | `run_20260801_005640`/`_145728` | 7200s | **68/0/24** | 39 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_round/syn_0 | `run_20260802_104607`/`_155005` | 7200s | 72/3/18 | 178 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| felix_round/syn_0 | `run_20260802_104547`/`_163513` | 7200s | 65/9/18 | 189 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| felix_it/syn_0 | `run_20260802_105134`/`_130206` | 7200s | 67/10/16 | 259 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| fedbuff_it_unaware/syn_0 | `run_20260731_180442`/`_182654` | 1200s | 72/3/18 | 38 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fedbuff_it_oracular/syn_0 | `run_20260731_184247`/`_190502` | 1200s | 73/2/18 | 38 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fwdllm_it_unaware/syn_0 | `run_20260731_163249`/`_165420` | 1200s | 66/1/25 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260731_170129`/`_172308` | 1200s | 66/1/25 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. Open fails: §B.

**⚠ The cadence/convergence cells on the three uncapped rows are NOT currently evidence about sim.** Swapping
only which real replicate they were graded against — no code change — moved `felix_round` 71/3 → 65/9 and
`fedbuff_round` 72/2 → 72/3 *with different fails*. Read only INV/un-windowed rungs until H12 (§B) resolves.

**Replicate floor** — `replicate_floor.py --mode real`, 7200s, seed 1234, config-identical legs. The number
every DIST tolerance must clear (§D-24); running the actual rung functions on two REAL legs is worse still:

| baseline | bins | cycles | iters/bin | mean_var | rung functions, real↔real |
|---|---|---|---|---|---|
| `fedbuff_round` | 1.1% | 2.8% | 3.9% | **4.7%** (tol 2%) | `v2` 4.68%, `conv` 5.79 pts; `v1c` flat |
| `felix_round` | **12.9%** (tol 5%) | 0.1% | **13.3%** | 0.5% | `v1`/`v1b` **18.9%**, `conv` **7.30 pts**, `v1c` λ=+0.279 t=+8.23 `diverging` |
| all seven others | — | — | — | — | **UNMEASURED — no replicate exists** |

Peak accuracy across those same replicates: `felix_round` 77.17% vs 66.01% (**11.16 pts**), `fedbuff_round`
75.11% vs 73.32% (1.79 pts). Mechanism, and why the floor differs per baseline: §B-H12.

**Budget coverage.** The five 7200s pairs are healthy (min 88.3-100%). The four 1200s rows are not:
`fedbuff_it_oracular` 76.0% trips the low-coverage flag and `fwdllm_it_*` grade N=**5**. Read those four for
INV/un-windowed rungs only.

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

| baseline | open fails | next step |
|---|---|---|
| `felix_round` (65/9/18) | whole cadence family + `thru`/`commits`/`terminal`/`conv` · `selection_detail` | **All but `selection_detail` are inside the real↔real floor** — the same sim leg reads 71/3 against real A. `v1` 13.0% vs a 18.9% real↔real gap. Blocked on H12, NOT on code. `selection_detail` was H11 and is FIXED (§G) — needs a validation run |
| `fedbuff_round` (72/3/18) | `selection_bias` 10.4% (tol 10%) · `utility` · `convergence` 6.80% | Fails CHANGED with the real replicate (`v2` now passes, these three appeared) — same floor problem, smaller. `v1c` flat both real↔sim and real↔real. Blocked on H12 |
| `fwdllm` (68/0/24) | none | Clean at 7200s. `drain_wall_budget` closed once the charge came from its own real (0.278→0.101). No replicate yet — floor UNMEASURED |
| `fwdllm_it_unaware` / `fwdllm_it_oracular` (66/1/25) | `drain_wall_budget` (charge 2.31x / 2.24x its own real) | Same as `fwdllm`. Cadence UNGRADED at 1200s — N=5 |
| `fluxtune` (77/0/16) | none | Clean at 7200s. The ONLY baseline with the iteration cap (`plateau`/`max_iter=20`), which is why (§A). No replicate yet — floor UNMEASURED |
| `felix_it` (67/10/16) | cadence family · `terminal`/`commits` 19.5% · `conv` 8.26% · `selection_detail` | First 7200s pair. `preferred_duration` **PASSES** now (0.124 vs tol 0.2) — closed. `drain_wall_budget` was the stale 1.35× charge, re-profiled (§G). Uncapped + `new` agg-rate ⇒ expect `felix_round`'s floor; no replicate yet |
| `fedbuff_it_unaware` / `fedbuff_it_oracular` (72/3, 73/2) | `v2` · `convergence` (+ `per_round_advance` on unaware) — 1200s pairs, N=38 | `v1c` is flat on both, but at t=1.68/−0.06 on N=38 the slope is simply unresolved. **H6 cannot be answered at 1200s** — it needs the 7200s pair |

### Next session

> **Update in place on every run — overwrite, never stack a new dated block below.**

**What the last batch settled.** Three new 7200s reals (`felix_round`, `fedbuff_round`, `felix_it`) + `felix_it`'s
first 7200s sim leg. **H10 CONFIRMED far more broadly than stated** — not four hairline rungs but
`felix_round`'s entire cadence family failing against its own replicate, including `v1c`, the rung built to be
the duration-invariant ROOT (§A). **H11 CONFIRMED and FIXED** (§G); `felix_it` is a clean negative, so the
defect is the round-boundary batch path, not AsyncOort's `select()`. **Charge circularity closed** (§G).

**Live hypotheses — each with the observation that would falsify it.** State the prediction BEFORE the run;
a hypothesis that can only be confirmed is not one (§D-9).

**H12 — SPLIT. Amplifier CONFIRMED (189x on the real stack); source FALSIFIED (§E) and now OPEN.**
The JVP's central difference is ill-conditioned by construction and that stands on its own. But the noise it
amplifies does NOT come from fp16/GPU nondeterminism — the real stack reproduces bit-exactly. **Something
still has to supply production's 1.6e-3, and with data, dispatch order, iteration, model_version, RNG
position and arithmetic all verified identical, no candidate is currently standing.** Next step is
`--hetero` + `--compare` (below) to rule out heterogeneous co-tenancy, then a weights/logits hash on the
trainer's first task — one short run, not a scoreboard run.
`IN PROGRESS — probe RUN ONCE, SPLIT RESULT.` Evidence on disk: §A. Mechanism:
`jvp = (L(p+hv) − L(p−hv))/(2h)` at h=0.01 under `autocast()`, so its condition number is ~`|L|/(2h·|ΔL|)`.
Two candidate fixes are landed behind env flags, default OFF, byte-identical off — `FWDLLM_JVP_FP32`
(JVP passes outside autocast) and `FWDLLM_STRICT_DETERMINISM` (`use_deterministic_algorithms` +
`CUBLAS_WORKSPACE_CONFIG` + TF32 off; `cudnn.deterministic` alone covers neither autocast kernel choice nor
cuBLAS reduction order).
**Decision rule, fixed BEFORE the run** (`probe_jvp_determinism.py --sweep`, also printed by the tool):
- some arm goes bit-exact ⇒ **BUG**. Promote the flag, re-measure. Don't widen tolerances, don't buy seeds.
- spread unchanged on every arm ⇒ **IRREDUCIBLE**. Write the §F invariant, widen each DIST tolerance to its
  floor, size the multi-seed budget (EXPTS_CHARTER D1).
- fp32 alone closes it ⇒ the amplifier is the whole story; strict determinism stays off the critical path.
- **FALSIFIED IF** the `base` arm is itself bit-exact across concurrent processes — the probe then proved
  nothing about any arm, and the PROBE is what needs fixing (preamble).

**Probe runs 1-2 (A40, torch 2.12, 8 concurrent processes) — the two halves split, and the SOURCE half is
now FALSIFIED (§E):**
- **AMPLIFIER CONFIRMED and it is WORSE on the real stack.** fp16 vs fp32 on the SAME input:
  **70x** on the proxy, **189x** on real DistilBERT+adapter (1.04M trainable of 67.4M) — deeper accumulation,
  worse conditioning. Production telemetry independently measured a 72x median. This needs no nondeterminism
  to hold and is the standing, run-independent case for `FWDLLM_JVP_FP32`.
- **NONDETERMINISM NOT REPRODUCED, on either model.** Every value bit-identical across 8 co-located processes
  on the REAL stack, so `base` was already exact and no arm can be credited.
  `FWDLLM_STRICT_DETERMINISM` is *inert* — there was nothing to pin, which is not evidence it is broken.

**H12a — the per-baseline floor DISPARITY is the aggregation rate, not the trainer.** The same fp16 noise
enters every trainer, so H12 alone cannot explain 3.9% vs 13-19%. `felix_round`/`felix_it` use
`agg_rate_type: new`, whose weight carries `β(stat_utility)` — loss-derived, so the noise perturbs the
aggregation WEIGHTS as well as the gradient values and compounds instead of cancelling. `fedbuff_round`'s
`old` rate is a function of integer staleness only. `fluxtune` additionally caps iterations. Selector is ruled
OUT (cohorts bit-identical between replicates). **Predicts** floors rank
`fluxtune` < `fedbuff_round` < `felix_round` ≈ `felix_it`, and that `felix_round` re-run with
`agg_rate_type: old` lands near `fedbuff_round`'s. **FALSIFIED IF** `fwdllm` (uncapped, `old` rate) comes back
with a `felix_round`-sized floor — that would put the cause upstream of the rate entirely.

**H8 — `fedbuff_it_unaware`'s H6 signature is unresolvable below 7200s, not absent.** Unchanged, still open.
**FALSIFIED IF** the 7200s λ flips sign. Now needs a replicate PAIR: a lone `diverging` verdict is
uninterpretable until H12 resolves.

### Roadmap to parity on all nine — 3 nodes

> **Update in place. Delete a step the moment its exit criteria are met and its findings are in §A/§G.**
> Goal is parity on all nine, fast. Order is by *information per node-hour*, not by baseline.
> H11/H12 validate-invalidate state + the pending H11 launch: [HANDOFF_H11_H12.md](HANDOFF_H11_H12.md)
> (temporary — delete when both close).

**Where the nine stand.** 2 clean at 7200s (`fluxtune`, `fwdllm`) · 3 blocked on H12 (`felix_round`,
`felix_it`, `fedbuff_round` — tolerances below their measured floor, **no code change can close them**) ·
4 on 1200s smokes at N=5-38, which are not evidence (`fwdllm_it_*`, `fedbuff_it_*`). **Seven of nine have no
replicate**, so their verdicts are single-leg readings — and we have direct proof single-leg readings flip.

**IMMEDIATE — tonight. Nothing here waits on anything else; run all three nodes in parallel.**

| | what | why | cost |
|---|---|---|---|
| **bench** | two `--sweep --model real --hetero` runs, then `--compare` | runs 1-2 confirmed the amplifier (189x) but did NOT reproduce the nondeterminism; this is the last arithmetic hypothesis before the walk moves to inputs | minutes |
| **node 1** | `felix_round` real+sim **7200s**, then `run_parity.py` | validates the landed H11 fix — `selection_detail` green and `trace_boundary_repicks.py` reads OVER-DISPATCH=0. **NOT 3600s**: the round-1→2 boundary the defect needs first fires at wall 4270-4823s (real) / vclock 4441s (sim), so a 3600s pair grades a run in which the defect cannot occur | ~4h |
| **node 2** | `fluxtune` real, then `fwdllm` real, then `run_parity.py` | the two CLEAN rows have unmeasured floors; a 77/0 graded against one real is a single-leg reading. Re-grading against the new real is FREE and is the same A/B that flipped `felix_round` | ~4h |
| **node 3** | `fwdllm_it_unaware` 7200s pair, then `fwdllm_it_oracular` 7200s pair | gets two rows off N=5 smokes onto real evidence; their `drain_wall_budget` fail also closes because the sim leg finally picks up the 08-01 profile | ~5h |

⚠️ **`fwdllm_it_*` need a full 7200s PAIR, not a sim-leg re-run.** Their reals are 1200s, so a 7200s sim leg
would grade against a 1200s real. Watchdogs on all four Phase-B real yamls are bumped to 10800 (§G).

```bash
cd lib/python/examples/fwdllm/expt_scripts

# bench — run this FIRST, anywhere with a GPU
# co-tenants doing DIFFERENT work, then the same-work-across-two-launches diff
python probe_jvp_determinism.py --sweep --model real --hetero --replicas 8 --out-dir probe_A
python probe_jvp_determinism.py --sweep --model real --hetero --replicas 8 --out-dir probe_B
python probe_jvp_determinism.py --compare probe_A probe_B

# node 1 — validate the H11 fix (7200s; the boundary arrives after 3600s)
bash run_sequential.sh --mode both --max-runtime-s 7200 --only felix_round --yes
python trace_boundary_repicks.py ../experiments/<new felix_round sim dir>   # expect OVER-DISPATCH=0
python run_parity.py --yes --baselines felix_round

# node 2 — floors for the two clean baselines (real only; a floor needs no sim leg)
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fluxtune --yes
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fwdllm --yes
python replicate_floor.py --mode real --baselines fluxtune fwdllm
python run_parity.py --yes --baselines fluxtune fwdllm      # free re-grade vs the NEW real

# node 3 — get the two thinnest rows onto real evidence
bash run_sequential.sh --mode both --max-runtime-s 7200 --only fwdllm_it_unaware,fwdllm_it_oracular --yes
python run_parity.py --yes --baselines fwdllm_it_unaware fwdllm_it_oracular
```

**THEN — branch on the probe. This is the whole critical path for the three blocked baselines.**

- **REDUCIBLE** (a flag collapses the spread) ⇒ it is a BUG. Promote the flag to default-on, re-run the three
  blocked pairs, re-grade. Their fails likely vanish; no tolerance moves. **~1 night, 3 nodes.**
- **IRREDUCIBLE** ⇒ no run can close them. Widen every DIST tolerance to its measured floor, re-grade, write
  the §F invariant. Needs one extra real per baseline to have a floor to widen TO: 7 baselines × 2h ≈
  **2 nights on 3 nodes**, and the reals parallelize perfectly.
- **PROBE DIDN'T REPRODUCE** ⇒ fix the probe (`--model real`, more load), not the code. The issue is measured
  independently of it; only the FIX is unvalidated (preamble).

**SHORT TERM — the remaining gaps, in priority order.**
1. **`fedbuff_it_*` off 1200s** (7200s pairs). N=38 smokes; nothing cadence-shaped there is evidence and H8 is
   unanswerable without it. Run as replicate PAIRS if H12 came back irreducible.
2. **Re-profile the four 1200s-sourced charge profiles** from their new 7200s reals — no run of its own, just
   `profile_sim_charges.py`, then one sim leg to pick it up.
3. **Floors for the last baselines without one**, so every row on the scoreboard is interpretable.
4. **Full 9-baseline re-grade** at 7200s with every tolerance calibrated. This is the parity sign-off.
5. **`async_oort`→`AsyncSelectorBase` integration confirmation** (`--only felix_round,felix_it`) — unit tests
   cover the mechanism, not a live comparison.

**Standing rules for this campaign.** A real↔real floor needs **no sim leg** — halve the cost. Compare
flag-OFF vs flag-ON **at one duration** (§C). One mechanism per run when a fix could perturb another baseline.
Never spend a run on a question a bench repro can answer (preamble).

### Other open items

**Gated on H12**
- **PROPOSED §F-28 — do NOT apply without operator approval + the probe result.** If IRREDUCIBLE: *"Forward-
  gradient training is not bit-reproducible across runs. Seeding fixes the RNG stream, not the floating-point
  path. Therefore (a) no DIST tolerance may sit below its measured replicate floor, (b) every reported result
  on an uncapped baseline needs a replicate-derived error bar, (c) a single-leg real↔sim verdict is not
  evidence."* If a flag closes it, this is a BUG not an invariant — promote the flag and delete this item.
- **`convergence` is UNGRADEABLE at its 5% `acc_tol`** — below the real↔real gap on both baselines that have
  a replicate. The optimizer-side root fix did land and did close it (§G), but against a single leg.

**Rung/tolerance gaps**
- KS-only rungs unguarded against a level shift (same class as the `selection_bias` repair, §G; all clean on
  live data): `dk1_agg_goal_trajectory`, `dk2_dynamic_c`, `dk3_eligible_ends_metric`, `eligible_speed`,
  `v3_cached_v_pool`. `selector_score` is DIAG.
- Thin ABSOLUTE budgets: `fwdllm_it_*` grade N=5 and `v1c` SKIPs. Coverage percentage cannot catch this; an
  absolute-N floor is proposed, threshold not chosen.
- `cohort_sequence.count` is rolled-up V1 — never chase it separately; it fails exactly when `v1` does.
- `step_timing_breakdown` / `agg_step_timing_breakdown` are REPORT-ONLY wherever the sim clock discards the
  span (§G, §D-31) — the designed steady state everywhere. Residuals there are §D-1 contention that never
  reaches the vclock; watch `charge_coverage` instead.
- Do not re-tune `redispatch_turnaround` (§D-14, §E).
- `felix_round`'s lap-boundary fails (`preferred_duration`, `terminal_state`) are underpowered artifacts —
  expect them at any duration whose graded window straddles a lap boundary.

**Known-and-deliberate**
- Real's over-`c` slot read is drain lag, not a §D-27 conflation (§E, §G). Telemetry only; the tripwire
  disagrees with the authoritative measure, not with `c`. Low priority.
- Real publishes no `_agg_slot_holders_ref`, so `_cap_dispatch_to_concurrency` falls back to the IDENTITY
  set. Harmless today; publishing one would let real dispatch into slots it now withholds — owed its own A/B.
- Sim's in-flight bookkeeping is still split across six sets and should be one per-end state machine. The
  slot⇄guard split did the CAPACITY half; IDENTITY is still ad-hoc. Never bundle with a correctness fix.
- Base `asyncfl/top_aggregator._sim_hold_busy_slots` deliberately untouched — no `_sim_committed` term, so it
  never had the conflation. Re-check if async_cifar10 shows the same under-fill.

**Future tasks**
- **Enforce §F-18 mechanically: a per-baseline knob CONTRACT.** Two correctness-path knobs went missing from
  yamls and were caught only by reading telemetry days later, so those runs were not fairly comparable. The
  mechanism mostly EXISTS — `run_sequential.sh`'s preflight `checks[]` already blocks a launch and ships both
  patterns; the gap is that `condition_fp` hashes only CLI-patched knobs, so yaml-only knobs are invisible.
  Three homes, none new: `test_baseline_readiness.py` → preflight `checks[]` (first tenant landed, §G) →
  parity `--validate`. **The hard part is "missing" vs "legitimately N/A"** (`sim_charge_profile_path` is
  sim-only, `reselect_cadence: round` round-only, `trackTrainerAvail` oracular-only), so applicability must be
  DECLARED, not diffed — sketch: a `knob_contract` block in `_metadata/baselines.yaml` read by all three
  layers. Open: where it lives; error-vs-warn; whether `condition_fp` absorbs it; who declares a new knob.
- Regenerate the four 1200s-sourced charge profiles (`fedbuff_it_*`, `fwdllm_it_*`) once their 7200s reals
  exist. Not urgent — the charge is not materially duration-sensitive.
- Flag promotion: `sim_sct_ordered_drain` + `sim_model_dispatch_queue` are fluxtune-yaml-only but model
  general async-transport artifacts — smoke fwdllm/fwdllm_plus with both ON, confirm inert-or-better, promote.
- Checker invariants I1-I6 were drafted in a prior session and never committed anywhere (unrecoverable).
  Needs operator input on intended semantics before drafting fresh ones.
- felix (async_cifar10) may share fluxtune's round-1 cold-start gap — unverified, out of scope
  (`async_cifar10/PARITY.md` owns felix). felix 46/46 reconfirmation gates Phase 2.
- Momentum (S1-S3) / server-optimizer — roadmap, not parity. NOTE: S1's damping should also shrink the
  replicate floor (EXPTS_CHARTER I-1) — re-measure after it lands.
- P3/infra: no automatic GPU skip-and-remap on a broken ordinal (manual `execution.gpu_ids` exclude works).

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

**D-41.** Seeding fixes RNG, not arithmetic. Before blaming a seed, check whether the stream POSITION matches
across runs; if it does, the divergence is in the floating-point path.

**D-42.** A discrete choice over near-tied continuous values (argmax, sort, a threshold gate) turns
round-off into an O(1) difference. Look for one before accepting a divergence as irreducible.

**D-43.** Grade a finite-difference estimator by its condition number, not its formula. `(f(x+h)−f(x−h))/2h`
in reduced precision amplifies round-off by ~`|f|/(2h·|Δf|)` — measure it, don't assume it's small.

**D-44.** Group replicates on the ACHIEVED span, never the configured duration. A truncated run reports the
duration it was asked for and reads as irreproducibility.

**D-45.** A replicate floor must come from legs that differ ONLY in wall-clock luck. Swapping which replicate
a fixed comparison is graded against is the cleanest A/B for whether a rung measures anything.

**D-46.** Validate a proposed fix on a bench repro that imports the real code path, never on an FL run first.
Confirm the CONTROL reproduces before crediting any fix (preamble).

---

## §E  Dead ends — do NOT retry

> One line each, append-only. A dead end never un-dies; re-listing one wastes a session. Numbers only where
> the number IS the lesson. Landed-but-inert cleanups belong in §G.

**Replicate floor / H12**
- **H12's SOURCE half — fp16/GPU kernel nondeterminism as the replicate floor's cause** — **FALSIFIED on the
  bench.** The REAL DistilBERT+adapter stack, 8 co-located processes on one A40, same node: every loss and
  jvp bit-identical, within and across processes. Also falsified for the production divergence, on disk:
  the trainer data is bit-identical (all 100 `CLIENT n DATA HASH` lines match across runs), dispatch order
  matches, and every first task is `iteration 0` / `model_version 0` — yet 22 of 30 trainers still differ.
  Don't spend more probe load on this; the arithmetic reproduces.
- **"first task at iteration k>0 sees mid-bin updated weights"** — FALSIFIED: all 30 first tasks are at
  iteration 0 in both runs.
- **arrival/dispatch order as the discriminator for which trainers reproduce** — FALSIFIED: the two runs'
  dispatch ranks are identical (two adjacent swaps in 30), and the 8 reproducing trainers are scattered
  across ranks 1-29.
- **`v1c`'s "monotone shape is the discriminator" calibration** — FALSIFIED: a real↔real pair climbs
  monotonically at t=8.23, bigger than any real↔sim λ on that baseline. Don't cite shape or
  `lambda_floor_per_100` as evidence a `diverging` verdict means sim.
- **Any cadence/convergence verdict read off a SINGLE real leg on an uncapped baseline** — dead. The same sim
  leg scores 71/3 or 65/9 depending on the real. A verdict needs a replicate PAIR (§D-45).
- **Seeding / `client_idx` / data partition / cohort choice as the replicate spread's cause** — all four
  REFUTED by direct measurement; the divergence is floating-point, not RNG (§D-41).

**Charges**
- **`redispatch_turnaround` as a large cost sim omits** — FALSIFIED. Its first-difference marginal (§D-12)
  assumes ONE serial dispatch burst; under event-driven dispatch it prices inter-arrival WAITING, and the
  direct cost is real≈sim so there is nothing to charge. The profiler now refuses to overwrite (§G). Don't
  re-derive a redispatch charge from timestamp differences on an event-driven baseline.
- **A charge delta passing through to throughput at a fixed coefficient (H9)** — malformed question:
  `sim_s_per_round` is vclock ÷ INTEGER rounds, so small deltas sit under one quantum, and a charge also
  moves cadence (§D-21). There is no single coefficient.
- **Sizing a throughput prediction off a charge delta** — REFUTED on the validation leg; the charge fix was
  correct, its predicted magnitude wrong. Confirm the clock is charge-limited first (§D-18).
- **`_compute_var` GC pause** · **`_flat_grad_norm` as the drain-wall amplifier** (landed, bit-identical, but
  not the dominant contention source) · **`real_distribute_settle_s`** — all REFUTED as parity causes.

**Cadence / trajectory**
- **H7: `felix_round`'s divergence as a `var@it0` trajectory effect** — DEAD; a per-cycle charge correction
  closed the whole family. Don't re-open `var@it0` as its root.
- **`fedbuff_round`'s cadence as a progressive trajectory divergence (H3)** — FALSIFIED; `v2b_var_drift`
  flipped to `level_offset` and the sign reversed. §D-14 says re-decompose, don't hunt separating curves.
- **Root C (one charge-coupled cadence level shared by both round baselines)** — DEAD as stated; the repaired
  `overlap_factor` clears both and `v2b_var_drift` splits them.
- **`fedbuff_round`'s cadence residual as a mechanism** — it is a level offset with a flat rate inside the
  replicate floor. Calibrate the tolerance, don't hunt the mechanism (§D-24).
- **per-cycle committed-set overlap as the trajectory discriminator** — FALSIFIED: it does not separate the
  baselines (`fluxtune` has the least set agreement and the best trajectory agreement). Discriminator is the
  drift RATE (§D-35).
- **cohort COMPOSITION as `felix_round`'s `v2` driver** — REFUTED; both modes commit from the same frozen 30
  with the gap fully present. The trajectory diverges, the contributor set doesn't.
- **`felix_round`/`fluxtune` sharing one overshoot root** — REFUTED, disjoint factors (§D-16).
- **`reselect_cadence` pool size as the round-cadence throughput driver** — SUPERSEDED by per-cycle charge
  compounding (§D-11).
- **"surplus idle" as the post-§D-15 residual** — REFUTED; the `c/(busy+idle)` identity matches. Don't
  re-open the idle term.
- **eval-thread GPU contention reaching commit order via `sct`** — FALSIFIED; sim orders on a deterministic
  modeled-delay grid, so contention cannot move it.

**Slots / selection**
- **`selection_detail`'s `felix_round` fail as a windowing/thin-N artifact** — FALSIFIED, and re-windowing
  would have MASKED a real over-dispatch defect (§D-40).
- **real's over-`c` slot read as a half-fixed §D-27 conflation** — WRONG diagnosis; it is drain lag (§D-33).
  Don't re-open the `_PendingCommitUnion` halves, they were correct.
- **`selection_bias` on the round baselines as TWO selector faults (H2)** — FALSIFIED; both flipped green on
  the slot⇄guard split with zero selector change. Opposite signs against a shared input do not imply separate
  roots when both draw from the same availability bookkeeping.
- **sct-order membership (admit lowest-sct instead of first-arrived)** — REJECTED: FIFO-violating, deadlocks
  under Phase-2 unavailability, and the divergence is a stochastic tie-break (§D-2).
- **the round-cadence cohort pin as a defect** — NOT a defect, operator ruling; it pins by design. Don't
  re-key it on `_model_version` (§D-17).
- **`step_timing_breakdown`'s gating func "moving" to `_emulate_training_delay`** — it never gated; the rung
  ranked exempted funcs. Fixed at the source (§G, §D-32).

**Measurement constructs**
- **`matched_virtual_budget` (V = min(vclock, wall))** — DELETED, don't reintroduce: conflates the two clocks
  `sim_rate` tests and fails to grade (§D-4).
- **real's wall-vs-vclock clock anchor as the residual's cause** — REFUTED; residuals survive the change of
  coordinate, so they are in the mechanism, not the measurement.
- **recv_fifo→drain_ready as the fluxtune D-skew fix** — LANDED but INERT; kept as cleanup, not a parity fix.
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

## §G  Landed fixes — recent + load-bearing ONLY. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, immediately.** The instant a rung flips or a hypothesis resolves, write ONE line
> (mechanism + outcome) and delete it from §A/§B in the same edit. Newest first. Delete an entry once nothing
> current depends on it — git log keeps it.

**This batch**
- **Phase-B watchdog blocker cleared** — `max_experiment_runtime_s` 7200→10800 on the four real yamls that
  still had it equal to the 7200s target (`fwdllm_it_*`, `fedbuff_it_*`); `felix_it` was already done.
- **H11 over-dispatch FIXED.** `_release_sim_slots_at_agg_goal`'s legacy path clears `_sim_inflight_expected`
  AND `_sim_pending_commit`, then calls `_sim_hold_busy_slots`, which rebuilds `outstanding` from those emptied
  sets and strips `all_selected` — both guards zero at one instant, so boundary top-ups re-picked still-training
  ends (35 picks vs c=30). Sim now folds in `_trainer_inflight_dispatch_version`, the half real's
  `_PendingCommitUnion` already carries. 3 tests (2 fail without). **Not validated live — §B roadmap, node 2.**
- **H12 probe + two candidate fixes, default OFF, byte-identical off.**
  `probe_jvp_determinism.py --sweep` (bench A/B across concurrent processes, no FL run); `FWDLLM_JVP_FP32`
  (JVP passes outside autocast); `FWDLLM_STRICT_DETERMINISM` (`use_deterministic_algorithms` +
  `CUBLAS_WORKSPACE_CONFIG` + TF32 off). 23 tests. **Experiment NOT run** — decision rule in §B-H12.
- **`replicate_floor.py` groups on ACHIEVED span, not configured `max_runtime_s` (§D-44).** A leg >5%
  (`--span-tol`) short of the group's longest is DROPPED and named, even when that leaves <2 legs.
  `run_20260801_232459_felix_round` ran 5563s of 7200s and had inflated `felix_round`'s floor
  13.3%→16.0% / 12.9%→18.9%. 5 tests.
- **`felix_it` charge re-profiled from its own 7200s real** — was built from the 1200s leg and over-charged
  `drain_tail` 1.35× (0.3726 vs 0.2768) / `fedavg` 1.29×, which WAS its `drain_wall_budget` fail. Now
  0.2768/0.0689 at n=3827 (was 549).

**Load-bearing machinery (older, still relied on)**
- **Per-baseline charge profiles + launch-time provenance gate (§D-36).** One family-wide `drain_tail`
  constant was 1.08-2.75× each baseline's own real cost. Nine profiles in `sim_charge_profiles/<baseline>.yaml`
  from `profile_sim_charges.py`; `--only-observed` stops a refresh carrying an op the baseline never ran;
  `run_sequential.sh` BLOCKS a launch whose charges came from another baseline's real (matches `_<baseline>_n`).
  **`redispatch_turnaround` is NOT per-baseline** (§E) — its marginal is only valid under burst dispatch;
  `var_bad` stays OFF, `weights` pinned to 0.0598 except the two round baselines (0.043/0.035). Only
  `drain_tail`/`fedavg` are genuine per-cycle span means. 4 tests.
- **`charge_coverage` [DIAG]** — per label: sim wall vs what reached the clock vs REAL's span. The standing
  audit that makes a mispriced span announce itself. 4 tests.
- **`sim_clock_basis` — one primitive for "did the clock consume this?" (§D-31).** Sim's vclock advances only
  from `sct` and `charge_sim_vclock_overhead`; a span outside both never reaches it, so grading sim's wall for
  it fails on host contention. `drain_wall_budget` grades `charged_s` when profiled; `agg_step_timing_breakdown`
  gates only on `live` labels; `step_timing_breakdown` gates only when sim's compute BINDS `max(real_gpu_s, D)`
  (it never does). Absent ledger ⇒ UNKNOWN, never a silent demote. 6 tests.
- **Eval cadence made DETERMINISTIC — the `convergence` root, never duration.** `_eval_snapshot_model` returned
  None while the background eval thread was busy, making WHICH commits evaluate a wall-clock race sim loses
  structurally (real kept 99-100% of evals; sim 49-60% on seven of nine). Now gated on commit INDEX
  (`eval_every_n_commits`, code default **2**); a busy thread is waited out and warned, never dropped. Set in
  all 18 yamls (§F-18). 15 tests. §D-30.
- **Compute SLOT split from re-pick GUARD, BOTH modes (§F-23, §D-27).** One set served both roles, so §D-15's
  guard-hold also held the SLOT. `_slot_holders()` is now the single CAPACITY answer
  (`(inflight ∪ buffered) ∪ (pending − committed)`, published as `_agg_slot_holders_ref`);
  `_agg_pending_commit_ref` stays IDENTITY. Real's half is `_PendingCommitUnion.slot_holders()`
  (dispatched-not-yet-RETURNED). Flags `sim_commit_frees_slot` / `real_commit_frees_slot`, both default ON.
  Real's residual after this was a SECOND defect — drain lag, closed via `Channel.ends_with_pending_rx()`
  (queue depth only, safe on the dispatch path §F-19). 16 + 15 + 5 tests. §D-33.
- **Matched-budget primitive on the `data_id` axis (§D-26).** N is the position-wise common prefix of both
  chronological commit sequences, ordered by event `ts` (so legacy telemetry grades correctly too); `prog_fn`
  is its ordinal. A max-key ceiling never required both sides to commit the same bins. 5 tests.
- **Budget COVERAGE graded + stamped on all 8 windowed rungs** — `matched_budget_coverage` (Stage-0 CONTROL);
  every windowed result carries `budget_coverage` + `low_budget_coverage` below 80%. Hard-fails below 50% or
  on a `sequence_divergence`. 8 tests.
- **`v1c_iter_drift_rate` (§D-35)** — fits `ln(sim/real iters-per-bin)` against progress and t-tests the slope,
  because the level is a function of run length (`felix_round` +1.1% @3600s → +19.4% @7200s, unchanged code).
  `v1`/`v1b`/`v2` depend on it. ⚠ **Its real↔real calibration is FALSIFIED (§E)** — a `diverging` verdict on an
  uncapped baseline does not currently distinguish sim from noise. 6 tests.
- **`overlap_factor` (K4) repaired, DIAG→MECHANISM/EXACT** — per-cycle barrier (`intrinsic_span_s`) ÷ per-cycle
  clock advance over the matched budget; the only rung that localizes a throughput residual (§D-22). 4 tests.
- **`v2_var_trajectory` grades the matched budget on async too (§D-4)** — the truncation was gated on a clock
  coordinate that is None for every async baseline, so async graded the pooled run. Also ended a false PASS on
  `fedbuff_round`. 2 tests.
- **`selection_detail` reports `sim_repicks_in_round` (§D-39), DIAGNOSTIC.** `mean_chosen` averages a bimodal
  burst (one draw of c, then top-ups) so it reported how many top-ups fired, not the defect. Guarded on ≥2
  distinct selection rounds — under event-driven reselection `round` never advances and every legitimate
  re-pick would count. The finer in-flight predicate lives in `trace_boundary_repicks.py`. 4 tests.
- **KS-only rungs gained mean guards** — `selection_speed_bias` and `grad_norm` gated on KS alone, blind to a
  level shift when both sides share a shape.
- **`slot_utilization` rung (Stage-4 MECHANISM/EXACT)** — time-weighted mean/median slots busy, where
  `concurrency_cap` graded only PEAK (30/30 on both modes while means were 29.50 vs 24.66). §D-20.
- **`concurrency_cap` re-based onto peak DISTINCT in-flight ends from `contributor_intervals`** (§D-20) —
  `fluxtune`'s 31/30 was a phantom. Same-end concurrent dispatches graded separately (§F-25). 8 tests.
- **Lap-boundary identity snapshot fixed (§F-27, §D-26)** — `agg_round` read `round_num=self._round` LIVE
  against a pre-mutation `cycle_data_id`, emitting a non-monotone key; and `data_id` was transiently
  `== total_data_bins` when `version_bump_census` read it. Training state was always correct (staleness keys on
  `_model_version`, never `_round`). 7 tests.
- **`async_oort.py` re-based onto `AsyncSelectorBase`** (2193→897 lines) — utility-scoring POLICY unchanged,
  routed through shared `_choose`/`_pre_choose`. Integration-level real+sim confirmation still open (§B).
- **`run_parity.py` grades pairs in PARALLEL** — `--jobs N`, default one worker per pair capped by cores/RAM
  (~4.5 GB each). 9-baseline sweep 3m28s → 1m24s. 10 tests.
- **`sample_by_util` reproducibility** — `np.random.choice(p=...)` was pool/order-dependent; replaced with
  `_keyed_weighted_topk` (Efraimidis-Spirakis keys).
- **`reselect_cadence` knob** (round/data_bin/iteration); round cohort targets `c`, trimmed exactly
  (`agg_goal` is only the aggregation trigger).
