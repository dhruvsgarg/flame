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
>   **Match the bench conditions to the mechanism**, or the control will not fire — and that includes the
>   object's MODE, not just the launch shape (§D-47). The H12 probe ran *concurrent processes* because
>   co-location varies GPU kernel selection, but called `.eval()`, which silenced the live dropout that was
>   the actual defect: two runs of bit-exact nulls on a model production never runs.
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
| fluxtune/syn_0 | `run_20260802_172341`/`_150819` | 7200s | 73/3/16 | 95 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fwdllm/syn_0 | `run_20260802_192556`/`_145728` | 7200s | 64/3/24 | 39 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| fedbuff_round/syn_0 | `run_20260802_104607`/`_155005` | 7200s | 72/3/18 | 178 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| felix_round/syn_0 | `run_20260802_104547`/`_163513` | 7200s | 65/9/18 | 189 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| felix_it/syn_0 | `run_20260802_105134`/`_130206` | 7200s | 67/10/16 | 259 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| fedbuff_it_unaware/syn_0 | `run_20260731_180442`/`_182654` | 1200s | 72/3/18 | 38 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fedbuff_it_oracular/syn_0 | `run_20260731_184247`/`_190502` | 1200s | 73/2/18 | 38 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| fwdllm_it_unaware/syn_0 | `run_20260802_172249`/`_192444` | 7200s | **69/0/23** | 39 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260802_193900`/`_214059` | 7200s | **69/0/23** | 39 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. Open fails: §B.
**Four 08-02 runs are on disk and NOT in this table yet** — `fluxtune`/`fwdllm` second reals and both
`fwdllm_it_*` 7200s pairs. Grading them is §B T0.1; until then the `fwdllm_it_*` rows below are the old 1200s
smokes.

**⚠ The cadence/convergence cells on the three uncapped rows are NOT currently evidence about sim.** Swapping
only which real replicate they were graded against — no code change — moved `felix_round` 71/3 → 65/9 and
`fedbuff_round` 72/2 → 72/3 *with different fails*. Read only INV/un-windowed rungs until H13 (§B) resolves.

**Replicate floor** — `replicate_floor.py --mode real`, 7200s, seed 1234, config-identical legs. The number
every DIST tolerance must clear (§D-24); running the actual rung functions on two REAL legs is worse still:

| baseline | bins | cycles | iters/bin | mean_var | rung functions, real↔real |
|---|---|---|---|---|---|
| `fwdllm` | **0.0%** | **0.0%** | **0.0%** | 6.8% (tol 2%) | cadence reproduces EXACTLY; only `v2` is above its tolerance |
| `fluxtune` | 1.1% | 1.0% | 2.0% | **21.3%** (tol 2%) | `v2` is 10x its tolerance — the worst `mean_var` floor measured |
| `fedbuff_round` | 1.1% | 2.8% | 3.9% | **4.7%** (tol 2%) | `v2` 4.68%, `conv` 5.79 pts; `v1c` flat |
| `felix_round` | **12.9%** (tol 5%) | 0.1% | **13.3%** | 0.5% | `v1`/`v1b` **18.9%**, `conv` **7.30 pts**, `v1c` λ=+0.279 t=+8.23 `diverging` |
| `felix_it` + the four `*_it_*` | — | — | — | — | **UNMEASURED — no replicate exists** |

**`v2_var_trajectory` is ungradeable on all four measured baselines** — its 2% tolerance sits under every
`mean_var` floor (4.7% / 6.8% / 21.3%), so a `v2` verdict is a coin flip, pass or fail. **`fwdllm`'s cadence
floor is 0.0%**, which makes its `cohort_sequence`/`v1b` fails REAL signal, not noise — see §B.

Peak accuracy across those same replicates: `felix_round` 77.17% vs 66.01% (**11.16 pts**), `fedbuff_round`
75.11% vs 73.32% (1.79 pts). Mechanism, and why the floor differs per baseline: §B-H13. **All of it measured
dropout-live** — the OFF-mode control for every `jvp_eval_mode` run (§B).

**Budget coverage.** The seven 7200s pairs are healthy (min 88.3-100%). Only `fedbuff_it_*` are still 1200s;
`fedbuff_it_oracular` 76.0% trips the low-coverage flag. Read those two for INV/un-windowed rungs only.

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

| baseline | open fails | next step |
|---|---|---|
| `felix_round` (65/9/18) | whole cadence family + `thru`/`commits`/`terminal`/`conv` · `selection_detail` | **All but `selection_detail` are inside the real↔real floor** — the same sim leg reads 71/3 against real A. `v1` 13.0% vs a 18.9% real↔real gap. Blocked on H13, NOT on code. `selection_detail` was H11 and is FIXED (§G) — needs a validation run |
| `fedbuff_round` (72/3/18) | `selection_bias` 10.4% (tol 10%) · `utility` · `convergence` 6.80% | Fails CHANGED with the real replicate (`v2` now passes, these three appeared) — same floor problem, smaller. `v1c` flat both real↔sim and real↔real. Blocked on H13 |
| `fwdllm` (64/3/24) | `cohort_sequence` · `v1b_iters_moving_avg` · `v2` | **The only baseline whose cadence floor is 0.0%, so `cohort_sequence`/`v1b` are REAL sim↔real gaps, not noise — the strongest parity lead on the board.** Was 68/0 against its old real; the new replicate flipped it. `v2` is under its 6.8% floor and ungradeable |
| `fwdllm_it_unaware` / `fwdllm_it_oracular` (69/0/23) | none | **Both CLEAN at 7200s, N=39** — `drain_wall_budget` closed once the sim leg picked up the 08-02 profile. No replicate yet — floor UNMEASURED |
| `fluxtune` (73/3/16) | `drain_wall_budget` · `v2` · `convergence` | Was 77/0 against its old real; the new replicate flipped it, same single-leg effect as `felix_round`. `v2` sits under a **21.3%** floor and `convergence` under the usual acc gap — both ungradeable. `drain_wall_budget` is the one to chase |
| `felix_it` (67/10/16) | cadence family · `terminal`/`commits` 19.5% · `conv` 8.26% · `selection_detail` | First 7200s pair. `preferred_duration` **PASSES** now (0.124 vs tol 0.2) — closed. `drain_wall_budget` was the stale 1.35× charge, re-profiled (§G). Uncapped + `new` agg-rate ⇒ expect `felix_round`'s floor; no replicate yet |
| `fedbuff_it_unaware` / `fedbuff_it_oracular` (72/3, 73/2) | `v2` · `convergence` (+ `per_round_advance` on unaware) — 1200s pairs, N=38 | `v1c` is flat on both, but at t=1.68/−0.06 on N=38 the slope is simply unresolved. **H6 cannot be answered at 1200s** — it needs the 7200s pair |

### Next session

> **Update in place on every run — overwrite, never stack a new dated block below.**

**What the last batch settled.** Three new 7200s reals (`felix_round`, `fedbuff_round`, `felix_it`) + `felix_it`'s
first 7200s sim leg. **H10 CONFIRMED far more broadly than stated** — not four hairline rungs but
`felix_round`'s entire cadence family failing against its own replicate, including `v1c`, the rung built to be
the duration-invariant ROOT (§A). **H11 CONFIRMED and FIXED** (§G); `felix_it` is a clean negative, so the
defect is the round-boundary batch path, not AsyncOort's `select()`. **Charge circularity closed** (§G).
**H13 found the replicate floor's source and closed it on the bench: live dropout inside `calculate_jvp`.**
Probe C confirmed it on the real stack and `jvp_eval_mode` drives the spread to exactly 0, so the three
blocked baselines are no longer heading for a tolerance-widening ending. What is unmeasured is what eval mode
does to ACCURACY — that is the whole point of the next runs.

**Live hypotheses — each with the observation that would falsify it.** State the prediction BEFORE the run;
a hypothesis that can only be confirmed is not one (§D-9).

**H13 — the noise source is DROPOUT inside the finite difference. CONFIRMED on the real stack, and
`jvp_eval_mode` (§G) removes it completely. Only the ACCURACY question is open.**
**Probe C (A40, real DistilBERT+adapter, 8 concurrent processes, 20 repeats/arm) — within one process, same
input and same perturbation repeated:**

| arm | live dropout | loss exact | loss spread | jvp spread |
|---|---|---|---|---|
| `base` | 13 | 10% | 8.38e-03 | **3.44** |
| **`evalmode`** | **0** | **100%** | **0.00e+00** | **0.00e+00** |
| `fp32` | 13 | 5% | 9.87e-03 | 5.44 |
| `determ` | 13 | 10% | 8.38e-03 | 3.44 |
| `fp32determ` | 13 | 5% | 9.87e-03 | 5.44 |

A jvp *relative* spread of 3.44 means the estimate's range is 3.4x its own largest value — the gradient is
noise-dominated, not noisy. `determ` is bit-for-bit `base` and `fp32` is WORSE, both as predicted: neither
flag touches dropout, and with live masks each arm draws its own stream, so only `evalmode` reaching exactly
0 is a signal. **Decision rule (§B, fixed before the run) says BUG, not IRREDUCIBLE: fix it, do not widen
tolerances.** Cross-process still reads exact on every arm because each replica restarts the same global RNG
stream; production advances it by a task count timing decides. `create_model` → `train_adapter` leaves **13 of DistilBERT's 20 `nn.Dropout` modules training at
p=0.1** while `model.training` reads **False**, and nothing ever calls `.eval()` on the training path. So
`calculate_jvp` evaluates `L(p−hv)` and `L(p+hv)` under **two different dropout masks**, drawn from the
process-global RNG that no per-task seed pins. First seen on CPU with no autocast (`jvp` ∈ [−0.31, −1.50]
for one fixed input), so it is neither a GPU nor an fp16 effect.
- Explains every prior null at once: the per-client `torch_rng` is a *different* generator from the global one
  dropout draws on, so the RNG-position audit passes; data, dispatch order, iteration and model_version all
  match because the divergence enters *below* them (§E).
- Explains the 8-of-30 split: a trainer reproduces exactly when its global stream sits at the same offset,
  and that offset is advanced by prior forward passes — a count that async timing decides.
- **Correctness, not just reproducibility.** Two masks means the estimator is not a directional derivative of
  any one function. The 189x amplifier (H12, still CONFIRMED) then multiplies mask noise, not fp16 noise.
- **Dropout is live for 100% of training.** `eval_model()` is the only `.eval()` on the trainer path and it
  NEVER FIRES: `evaluate_during_training` is hardcoded False (`trainer/main.py:125`) and `test_on_the_server`
  is the aggregator's model. Verified on disk — 0 `len(test_dl)` lines in a 100-trainer `felix_round` real log.
  So **the 75-77% peaks on record were trained dropout-live; eval-mode accuracy is UNMEASURED.**
**STILL FALSIFIABLE as the *dominant* term** — if `jvp_eval_mode` runs land and the 13.3% iters/bin /
11.16-point floor does not move, something else supplies most of it and §F-28 comes back.
**What is NOT yet known: accuracy.** Every peak on record was trained dropout-live, so eval-mode accuracy is
unmeasured — that, not the noise, is what the next runs buy.

**Probe runs 1-2 were INCONCLUSIVE, not negative.** `_build_real` called `.to(device).eval()`, which silences
exactly the mechanism above — so "every value bit-identical across 8 co-located processes" measured a model
production never runs (§D-47; probe fixed, `evalmode` is now an arm and `--hetero`'s meaningless cross-process
table is suppressed). Two results survive: the **AMPLIFIER**, CONFIRMED and worse on the real stack — fp16 vs
fp32 on the same input is **70x** on the proxy, **189x** on real DistilBERT+adapter (1.04M trainable of
67.4M), against a 72x median in production telemetry — and `FWDLLM_STRICT_DETERMINISM` reading *inert*, which
was never evidence it is broken. Arithmetic as the SOURCE stays falsified (§E): it reproduces bit-exactly.

**H12a — CONFIRMED by its own falsifier. The per-baseline floor DISPARITY is the aggregation rate.**
It said *"FALSIFIED IF `fwdllm` (uncapped, `old` rate) comes back with a `felix_round`-sized floor."* `fwdllm`
came back at **0.0% on bins, cycles AND iters/bin** — its two reals are cadence-identical. Measured ranking is
`fwdllm` 0.0% < `fluxtune` 2.0% < `fedbuff_round` 3.9% << `felix_round` 13.3%, exactly the predicted order.
Mechanism: `felix_round`/`felix_it` use `agg_rate_type: new`, whose weight carries `β(stat_utility)` —
loss-derived, and `_compute_batch_stat_utility` runs the same dropout-live model, so the noise perturbs the
aggregation WEIGHTS as well as the gradient values and compounds instead of cancelling. `old` is a function of
integer staleness only, and `fluxtune` additionally caps iterations. Selector is ruled OUT (cohorts
bit-identical between replicates).
**Consequence, and it is the one that matters:** an `old`-rate baseline is nearly immune to H13's noise, so
`jvp_eval_mode` should barely move `fwdllm`/`fedbuff_round` cadence and should move `felix_round`/`felix_it` a
lot. **That asymmetry is a prediction the ON runs will test.** Still untested: `felix_round` re-run with
`agg_rate_type: old` landing near `fedbuff_round`'s floor.

**H8 — `fedbuff_it_unaware`'s H6 signature is unresolvable below 7200s, not absent.** Unchanged, still open.
**FALSIFIED IF** the 7200s λ flips sign. Now needs a replicate PAIR: a lone `diverging` verdict is
uninterpretable until H13 resolves.

### Roadmap to parity on all nine — 3 nodes

> **Update in place. Delete a step the moment its exit criteria are met and its findings are in §A/§G.**
> Goal is parity on all nine, fast. Order is by *information per node-hour*, not by baseline.
> H11's validate-invalidate state + its still-owed launch: [HANDOFF_H11.md](HANDOFF_H11.md)
> (temporary — delete when H11 closes).

**Where the nine stand.** 2 clean at 7200s (`fluxtune`, `fwdllm`) · 3 blocked on H13 (`felix_round`,
`felix_it`, `fedbuff_round` — tolerances below their measured floor, and H13 is a CODE bug, so a code change
is what closes them) · 4 thin (`fwdllm_it_*` now have 7200s pairs on disk and are UNGRADED; `fedbuff_it_*`
are still 1200s smokes at N=38).

**Landed on disk, not yet read (T0 below).** `fluxtune` + `fwdllm` second reals (`run_20260802_172341`,
`run_20260802_192556`) and `fwdllm_it_unaware` / `fwdllm_it_oracular` 7200s pairs (`run_20260802_172249` +
`_192444`, `run_20260802_193900` + `_214059`). All four reals achieved their 7200s. **These are OFF-mode
runs** — they are the last word on the current config and the control for every ON run below.

---

#### T0 — DONE (08-02). What it established.

| | task | outcome |
|---|---|---|
| **T0.1** | grade the four runs on disk | §A updated. `fwdllm_it_*` **69/0/23 CLEAN**; `fluxtune` 77/0→**73/3**, `fwdllm` 68/0→**64/3** on the new reals; floors measured for `fluxtune`/`fwdllm`. **H12a CONFIRMED** by its own falsifier |
| **T0.2** | `jvp_eval_mode: true` in the trainer block of 10 yamls (real+sim × `felix_round`, `felix_it`, `fedbuff_round`, `fluxtune`, `fwdllm`) | all five 7200s baselines can now run ON. Flip one line per file to go back |
| **T0.3** | preflight check in `run_sequential.sh` | blocks the launch on a mismatched OR one-sided knob; both cases negative-controlled. Warns when absent (code default = dropout LIVE) |
| **T0.4** | OFF-mode peak accuracy — **the control every ON leg is graded against** | below |

**OFF-mode peak accuracy, 7200s reals (dropout LIVE).** Two legs where a replicate exists:

| baseline | legs | band | cadence floor |
|---|---|---|---|
| `fluxtune` | 83.38% · 83.61% | **0.23 pts** | 2.0% |
| `felix_it` | 84.72% (one leg) | — | unmeasured |
| `fedbuff_round` | 75.11% · 73.32% | 1.79 pts | 3.9% |
| `felix_round` | 77.17% · 66.01% | **11.16 pts** | 13.3% |
| `fwdllm` | 28.74% · 41.41% | **12.67 pts** | **0.0%** |

**`fwdllm` is the finding.** Its schedule reproduces EXACTLY (0.0% on bins, cycles and iters/bin) and its
learning still lands 12.67 points apart. Cadence and accuracy have DIFFERENT floors and therefore different
causes: cadence tracks the aggregation rate (H12a), accuracy tracks gradient noise (H13). It also sets the
grading rule — **an ON leg must be compared to a BAND, not a number.** For `fluxtune` (0.23 pts) a 1-point
move is signal; for `fwdllm` anything inside 28-42% says nothing.

#### Step 1 — DONE (08-02 23:27, 903s per node). **CLEARED for step 2.**

All four gates green on all three nodes: **100/100 trainers logged `jvp_eval_mode=True`, zero `False`**, zero
tracebacks, full 903s span. Learning, in the ON run's OWN eval window applied identically to each OFF leg
(matching the window matters — an unmatched cutoff silently handed OFF more time):

| baseline | window | ON | OFF leg 1 | OFF leg 2 | evals ON/OFF |
|---|---|---|---|---|---|
| `felix_round` | 597s | **34.22%** | 32.55% | 32.00% | 12 / 12 / 12 |
| `fedbuff_round` | 559s | 40.47% | 36.21% | 41.43% | 12 / 11 / 11 |
| `fluxtune` | 445s | 40.72% | 31.00% | 40.96% | 4 / 3 / 4 |

**No baseline degrades outside its OFF spread, and `felix_round` beats both its OFF legs.** Matched eval
counts say throughput did not regress. **Read this as a go/no-go, NOT as the accuracy answer** — dropout is a
regularizer, so its absence flatters an early curve and can still cost peak accuracy at 7200s. That verdict
is Step 3's.

**JVP cost (T1.3).** `tb_forward_jvp` median 23.75ms ON vs 24.48ms OFF, `tb_stat_utility` 8.32 vs 8.42ms —
~3% cheaper, direction as predicted, no slowdown. Small enough that the OFF-derived charge profile is not
badly wrong, large enough that node A's re-profile stays in the plan.

#### Step 1 (for reference) — 15-MINUTE VERIFICATION on all three nodes, BEFORE the overnight.

Same yamls, same command, `--max-runtime-s 1800`. Costs 35 minutes and is the only thing standing between a
mistyped knob and nine wasted node-hours.

**15 minutes is enough, and the second baseline on each node needs no run at all.** The knob check is a
preflight, so `--dry-run` settles it in seconds; the live leg only has to prove the trainer still learns, and
at 900s `felix_round`/`fedbuff_round` produce ~10 evals and `fluxtune` ~5 — a curve, not a peak.

```bash
cd lib/python/examples/fwdllm/expt_scripts

# node A  (--mode both: also exercises the real/sim knob-match check A needs tonight)
bash run_sequential.sh --mode both --max-runtime-s 7200 --only felix_round --dry-run && \
bash run_sequential.sh --mode real --max-runtime-s 900  --only felix_round --yes

# node B
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fedbuff_round,felix_it --dry-run && \
bash run_sequential.sh --mode real --max-runtime-s 900  --only fedbuff_round --yes

# node C
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fluxtune,fwdllm --dry-run && \
bash run_sequential.sh --mode real --max-runtime-s 900  --only fluxtune --yes
```

Check all four, per node, before launching anything long:

| | check | how |
|---|---|---|
| 1 | preflight is green on the knob | `jvp_eval_mode (…) true on every leg` in the PRE-FLIGHT block |
| 2 | the knob reached the TRAINER | `grep -m1 JVP_EVAL_MODE ../experiments/<run>/*trainers.log` reads `jvp_eval_mode=True` — **100 lines, one per trainer** |
| 3 | it still learns | `plot_run.py --run-dir …` max acc well above 25% (4-class chance). A 30-min leg will not reach the 7200s peak; you are looking for a curve, not a number |
| 4 | nothing threw | no `Traceback` in the trainer or aggregator log |

#### Step 2 — the 9h overnight, 3 nodes. Every 7200s baseline gets an ON replicate PAIR.

Reals first by design: a real↔real floor needs no sim leg, accuracy is a reals-only question, and a sim leg
launched before T1.3 would price the JVP off an OFF-derived charge profile.

| node | jobs, in order | buys | cost |
|---|---|---|---|
| **A** | `felix_round` real ON ×2 → `replicate_floor` → `profile_sim_charges` → **sim ON** → `trace_boundary_repicks` + `run_parity` | the whole `felix_round` story — ON floor, ON accuracy, ON parity — **and the H11 live validation still owed** ([HANDOFF_H11.md](HANDOFF_H11.md)). Slack is deliberate: this is the only chain that can fail mid-way | ~5h + slack |
| **B** | `fedbuff_round` real ON ×2, then `felix_it` real ON ×2 | ON floor + accuracy for the other two blocked baselines | ~8h |
| **C** | `fluxtune` real ON ×2, then `fwdllm` real ON ×2 | the two tightest controls: `fluxtune`'s 0.23-pt band makes any accuracy change visible, and `fwdllm`'s 0.0% cadence floor makes its 12.67-pt accuracy band a clean H13 test | ~8h |

One command per node. Node A is `&&`-chained because every step feeds the next; B and C use `;` so a bad
baseline cannot cost the other one its night. The `$(ls -dt …)` substitutions resolve when that step runs, not
at paste time, so they pick up the runs the earlier steps just produced.

```bash
cd lib/python/examples/fwdllm/expt_scripts

# ---- node A: felix_round end to end, and the H11 validation ----
cp ../sim_charge_profiles/felix_round.yaml ../sim_charge_profiles/felix_round.yaml.off-bak && \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_round --yes && \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_round --yes && \
python replicate_floor.py --mode real --baselines felix_round && \
python profile_sim_charges.py $(ls -dt ../experiments/*felix_round*_real | head -2 | sed 's/^/--real-run /') \
    --out ../sim_charge_profiles/felix_round.yaml --only-observed && \
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only felix_round --yes && \
python trace_boundary_repicks.py $(ls -dt ../experiments/*felix_round*_sim | head -1) && \
python run_parity.py --yes --baselines felix_round

# ---- node B ----
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fedbuff_round --yes ; \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fedbuff_round --yes ; \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_it --yes ; \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_it --yes ; \
python replicate_floor.py --mode real --baselines fedbuff_round felix_it

# ---- node C ----
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fluxtune --yes ; \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fluxtune --yes ; \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fwdllm --yes ; \
bash run_sequential.sh --mode real --max-runtime-s 7200 --only fwdllm --yes ; \
python replicate_floor.py --mode real --baselines fluxtune fwdllm
```

**The `.off-bak` copy is the revert path.** Node A regenerates `felix_round`'s charge profile from ON reals,
which is correct for an ON sim leg and WRONG the moment the flag goes back off — restoring is one `cp`.

**⚠ MUST be 7200s, not 3600s** — node A doubles as the H11 validation and that defect only fires at the
round-1→2 boundary, which arrives at wall 4270-4823s (real) / vclock 4441s (sim).

#### Step 3 — TOMORROW MORNING. Read in this order, stop at the first failure.

| | check | if it fails |
|---|---|---|
| **1** | every ON run logs `jvp_eval_mode=True` | the run is void — relaunch, do not analyse it |
| **2** | peak accuracy ON vs the T0.4 **band** | **this is not a parity question.** A material drop means the regularizer was earning its keep: abandon eval mode, build the shared-mask fix (one mask reused across the ± passes — still a correct derivative of that masked loss) |
| **3** | `replicate_floor` ON vs OFF, per baseline | H12a predicts the ASYMMETRY: `felix_round`/`felix_it` should move a lot, `fwdllm`/`fedbuff_round` cadence barely at all, while `fwdllm`'s ACCURACY band should collapse. A uniform result on all five falsifies that split |
| **4** | no floor movement anywhere | **H13 falsified as the dominant term** — §F-28's tolerance recalibration comes back, and the ON runs stand as the control |
| **5** | `run_parity.py` on `felix_round` | `selection_detail` green + OVER-DISPATCH=0 closes H11 |

#### T1 — TBD tomorrow morning, once Step 3 has a verdict

- **T1.1 — promote or revert the flag deliberately.** If ON wins on both accuracy and floor, `jvp_eval_mode`
  stops being an A/B flag: it is a correctness fix and ships default-ON in code, not just in yamls. Operator
  call, not an automatic one.
- **T1.2 — every OFF result becomes a different config.** The §A scoreboard, all nine baselines' tolerances
  and the four charge profiles were measured dropout-live. Re-grading is a full re-run of the board, ~2
  nights on 3 nodes; do not pretend an ON row and an OFF row compare.
- **T1.3 — charge profiles must come from an ON real.** Dropout is a mask RNG plus an elementwise multiply, so
  the JVP should get marginally cheaper; `tb_forward_jvp` / `tb_stat_utility` in the ON legs give the actual
  number. Regenerate before any ON sim leg — a sim leg on OFF-derived charges mis-prices the trainer and the
  parity verdict is then about the profile, not the code.
- **T1.4 — `fedbuff_it_*` stay OFF and stay at 1200s until the verdict.** They are the only two baselines
  not going ON tonight; getting them onto 7200s pairs is the next night's work either way.

**SHORT TERM — the remaining gaps, in priority order.**
1. **`fedbuff_it_*` off 1200s** (7200s pairs). N=38 smokes; nothing cadence-shaped there is evidence and H8 is
   unanswerable without it. Run as replicate PAIRS if H13 came back irreducible.
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

**Gated on H13**
- **PROPOSED §F-28 — H13 says this is a BUG, so this item is now the FALLBACK only.** Apply nothing until
  the H13 fix has been A/B'd and the floor has *not* moved. If IRREDUCIBLE after that: *"Forward-
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

**D-47.** A bench repro must build the object in the MODE production runs it. `probe_jvp_determinism.py`
called `.eval()` "for a clean measurement" and thereby switched off the dropout that turned out to be the
whole defect — two runs of bit-exact nulls. Same class: `dropout=0`, `torch.no_grad()`, a fixed batch, a
warmup. Assert the mode in the record (`live drop`), don't assume it.

**D-48.** `model.training` is not the answer to "is dropout on". `train_adapter` left 13 of 20 `nn.Dropout`
modules training while the root module read False. Count the live submodules.

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
  Don't spend more probe load on this; the arithmetic reproduces. **The source is now identified — live
  dropout inside the finite difference (H13, §B) — which is why every input-side audit above came back clean.**
- **Forcing `.eval()` in the probe** — that is what made runs 1-2 read bit-exact on every arm. It is not a
  clean measurement, it is a different model (§D-47).
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
- **H13 CONFIRMED on the real stack and `jvp_eval_mode` closes it (probe C, §B).** `base` 10% of repeats
  bit-exact / jvp spread 3.44; `evalmode` 100% / 0.00e+00. `determ` bit-for-bit `base`, `fp32` worse — neither
  touches dropout. Probe now names the direction of a spread change and refuses cross-arm comparisons while
  dropout is live.
- **H13 fix landed behind `jvp_eval_mode`, default False = today's behavior.** `_eval_mode()` puts the model
  in eval for `_make_model_functional` + the whole training loop (functorch deep-copies the module, so a later
  toggle never reaches the fmodel the JVP evaluates), then restores EVERY module's own flag — a blanket
  `.train()` would invent a state the model never had. Covers `_compute_batch_stat_utility` too. Wired in
  `trainer/main.py`; 6 tests. **NOT bit-identical when ON, NOT parity-only — A/B first (§B).**
- **Phase-B watchdog blocker cleared** — `max_experiment_runtime_s` 7200→10800 on the four real yamls that
  still had it equal to the 7200s target (`fwdllm_it_*`, `fedbuff_it_*`); `felix_it` was already done.
- **H11 over-dispatch FIXED.** `_release_sim_slots_at_agg_goal`'s legacy path clears `_sim_inflight_expected`
  AND `_sim_pending_commit`, then calls `_sim_hold_busy_slots`, which rebuilds `outstanding` from those emptied
  sets and strips `all_selected` — both guards zero at one instant, so boundary top-ups re-picked still-training
  ends (35 picks vs c=30). Sim now folds in `_trainer_inflight_dispatch_version`, the half real's
  `_PendingCommitUnion` already carries. 3 tests (2 fail without). **Not validated live — §B roadmap, node 2.**
- **Probe fixed to run the production model (§D-47).** It forced `.eval()`, silencing the dropout H13 turns
  on; now builds as-created, adds an `evalmode` arm, records a `live drop` census, reads the verdict off the
  WITHIN-process column, and suppresses `--hetero`'s cross-process table (different seeds = different work,
  not nondeterminism). 7 tests. Runs 1-2's nulls are void; the 189x amplifier stands.
- **H12 probe + two candidate fixes, default OFF, byte-identical off.**
  `probe_jvp_determinism.py --sweep` (bench A/B across concurrent processes, no FL run); `FWDLLM_JVP_FP32`
  (JVP passes outside autocast); `FWDLLM_STRICT_DETERMINISM` (`use_deterministic_algorithms` +
  `CUBLAS_WORKSPACE_CONFIG` + TF32 off). 23 tests. Amplifier CONFIRMED at 189x; the two flags do NOT
  address H13's source — decision rule in §B-H13.
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
