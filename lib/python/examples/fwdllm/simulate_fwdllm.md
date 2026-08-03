# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 ported fedbuff/felix-lineage
baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3).
Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf, sim barrier redesign, delay-factor
calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/
gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — how to use this doc
>
> **Fresh session? Read in this order:** §F (locked invariants) → §E (dead ends) → §A (scoreboard) → §B
> (open issues + "Next session" + the run plan) → §D (only the lessons for the rung you're chasing) → §C
> for ladder mechanics.
>
> | section | contents | update rule |
> |---|---|---|
> | §A | scoreboard: pass/fail per baseline, ≤2-line caption | rewrite in place on every >3600s run |
> | §B | open issues + ONE "Next session" block + the run plan | current state only; an issue lives here XOR §G |
> | §C | ladder/decomposition method, run-length budget | edit only if the method itself changes |
> | §D | durable lessons — transferable invariants | ≤30 words each; update in place, never append near-dupes |
> | §E | dead ends — falsified hypotheses | one line each; never re-open |
> | §F | locked invariants — always-true / always-do | operator approval + evidence; amend in place, never renumber |
> | §G | closed items | move here the instant a §A/§B issue resolves; delete the source in the same edit |
>
> **Living doc, not a log — no dated annotations.** Every claim must read as true right now. §G is the one
> exception: newest-first by position, not by date. Full history is `git log` on this file.
>
> **Non-negotiables:**
> - Correctness per mode first; parity is the consequence, never the goal. A rung green because both sides
>   are equally wrong is a regression (§D-5). A divergence names two disagreeing sides, never which is at
>   fault — check each side's own absolute signal before choosing which to change (§D-9).
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

## §A  Score

**Two boards, one per `jvp_eval_mode` setting. A row NEVER compares across boards** — ON and OFF are
different training configs, and every OFF tolerance and charge profile was measured dropout-live (T1.2).

### §A.1  eval mode OFF (dropout live)

**Latest OFF run per baseline** (`run_parity.py`; ✓/✗/– = pass/fail/skip; PARITY.md §F). **Mixed durations —
rows do NOT compare to each other.** Five baselines have a 7200s pair; four have only the 1200s Phase-0 smoke
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

**⚠ The cadence/convergence cells on the three uncapped OFF rows are not evidence about sim.** Swapping only
which real replicate they were graded against — no code change — moved `felix_round` 71/3 → 65/9 and
`fedbuff_round` 72/2 → 72/3 *with different fails*. Read OFF rows for INV/un-windowed rungs only; the ON
board is where a cadence verdict is now readable.

**Budget coverage.** The seven 7200s OFF pairs are healthy (min 88.3-100%). Only `fedbuff_it_*` are still
1200s; `fedbuff_it_oracular` 76.0% trips the low-coverage flag. Read those two for INV rungs only.

### §A.2  eval mode ON (`jvp_eval_mode`, dropout off inside the JVP)

Only `felix_round` has an ON real↔sim PAIR. Four baselines have ON reals only (floors and accuracy, no
parity verdict); the four `*_it_*` have no ON run at all. NA = not run, not a skip.

| baseline | run pair | dur | pass/fail/skip | N | cohort | vclock | K4 | slots | sbias | thru | commits | terminal | V1c | V1 | V2 | U3 | S2 | conv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| felix_round/syn_0 | `run_20260803_021757`/`_042024` | 7200s | **70/4/18** | 196 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| fluxtune/syn_0 | reals ×2, no sim leg | 7200s | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fwdllm/syn_0 | reals ×2, no sim leg | 7200s | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fedbuff_round/syn_0 | reals ×2, no sim leg | 7200s | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| felix_it/syn_0 | reals ×2, no sim leg | 7200s | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| the four `*_it_*` | no ON run | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

`felix_round` ON: `selection_detail`, `commits`, `terminal_state` and `convergence` flipped green; `v1c` reads
`flat`; budget 196 matched units, real 100.0% / sim 94.2%. **`v1` fails only after the tolerance was sized
against the ON floor** (§G) — at its old 15% gate it passed a 5.4% gap over a 0.6% floor. Every ON run passed
the validity gate: 100/100 trainers logged the knob, zero `False`, zero tracebacks, achieved span 6928-7192s
on all eleven.

**All four fails are sim-SPECIFIC, confirmed by running the rung functions real↔real** (§D-45). The control
pair fails none of them, and fails only rungs that are structurally meaningless without a sim leg
(`sim_send_ts`, `vclock_telemetry`, `field_coverage`) or that compare one real's wall clock to another's
(`throughput`, `total_commits`, `terminal_state`). So the cadence family is not stochastic-selector noise.

### §A.3  Replicate floor, OFF → ON

`replicate_floor.py --mode real`, 7200s, seed 1234, config-identical legs, grouped by achieved span (§D-44)
and by flag state (§G). The number every DIST tolerance must clear (§D-24):

| baseline | agg rate | bins | cycles | iters/bin | mean_var (tol 2%) |
|---|---|---|---|---|---|
| `fwdllm` | old | 0.0 → **0.0%** | 0.0 → 0.3% | 0.0 → 0.3% | 6.8 → **0.2%** |
| `fluxtune` | old, capped | 1.1 → **0.0%** | 1.0 → 0.8% | 2.0 → **0.8%** | 21.3 → 6.6% |
| `fedbuff_round` | old | 1.1 → 4.3% | 2.8 → **0.2%** | 3.9 → 4.1% | 4.7 → **1.5%** |
| `felix_round` | new | 12.9 → **1.5%** | 0.1 → 0.5% | **13.3 → 0.6%** | 0.5 → 0.4% |
| `felix_it` | new | — → 2.7% | — → 0.4% | — → 2.3% | — → 2.4% |
| the four `*_it_*` | — | — | — | — | **UNMEASURED — no replicate exists** |

**The cadence floor collapsed exactly where H12a predicted and nowhere else.** `felix_round` (loss-derived
aggregation weight) went 13.3% → 0.6%, a 22x drop; the `old`-rate baselines had no cadence floor to lose and
did not move (`fwdllm` 0.0 → 0.3%, `fedbuff_round` 3.9 → 4.1%). The VARIANCE floor fell on every baseline that
had one. `fedbuff_round`'s bins floor moved the wrong way (1.1 → 4.3%) — one metric against nine, watch it.

**`v2_var_trajectory` is now gradeable on three of five** — `fwdllm` 0.2%, `felix_round` 0.4%, `fedbuff_round`
1.5% (tight) are under its 2% tolerance; `felix_it` 2.4% and `fluxtune` 6.6% still are not, and the checker now
SKIPs those rather than returning a coin flip (§G).

**These floors are inputs to the checker, not just a table.** `replicate_floor.py --profile-out` writes
`parity_floors/<baseline>.yaml`, and the DIST tolerances are sized from it: tighten toward the floor, never
past 2% absolute, never looser than nominal, and refuse to grade once the floor exceeds the tolerance (§G).
A baseline with no profile keeps its nominal gates, so nothing is ever graded against another baseline's
floor (§D-36).

**Peak accuracy, ON vs the OFF band** (§D-44: both legs cut at the matched achieved span, 6947s / 7189s):

| baseline | OFF legs | ON legs | band OFF → ON | verdict |
|---|---|---|---|---|
| `fluxtune` | 83.38 / 83.61 | 84.33 / 83.18 | 0.23 → 1.15 pts | straddles OFF — no change, on the tightest control |
| `fedbuff_round` | 75.11 / 73.32 | **77.49 / 81.20** | 1.79 → 3.71 pts | ON entirely ABOVE the OFF band, +2.4 to +6.1 pts |
| `felix_round` | 77.17 / 66.01 | **75.87 / 81.12** | 11.16 → 5.25 pts | band halved, top end +3.9 pts |
| `fwdllm` | 28.74 / 41.41 | 38.17 / 38.17 | 12.67 → **0.00 pts** | inside the band; the band vanished |
| `felix_it` | 84.72 (ONE leg) | 81.30 / 82.29 | — → 0.99 pts | −2.4 to −3.4 pts, but against no OFF band at all |

**No baseline degrades against a measured OFF band; two beat theirs.** The only apparent drop is `felix_it`,
the one baseline whose OFF side is a single unreplicated leg — and `felix_round`'s OFF band spanned 11 points,
so one leg is not a band. That comparison needs a second OFF leg before it means anything (§B).

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

| baseline | open fails | next step |
|---|---|---|
| `felix_round` **ON** (70/4/18) | `cohort_sequence` · `v1` · `v1b` · `v2` | **All four are sim-specific and all four are ONE quantity** — var@it0 7.3% low (H14). Clean real↔real control. Chase the grad pool, not the four rungs |
| `fwdllm` (OFF 64/3/24) | `cohort_sequence` · `v1b_iters_moving_avg` · `v2` | Its fails were already real (0.0% cadence floor) and `v2` is gradeable ON (0.2%). ON reals only — **needs an ON sim leg**, which is also H14's test |
| `fluxtune` (OFF 73/3/16) | `drain_wall_budget` · `v2` · `convergence` | ON reals only. `v2` stays ungradeable ON (6.6% floor, tol 2%). `drain_wall_budget` is the one to chase once an ON sim leg exists |
| `fedbuff_round` (OFF 72/3/18) | `selection_bias` 10.4% (tol 10%) · `utility` · `convergence` | ON reals only; ON var floor 1.5% makes `v2` gradeable. ⚠ its bins floor moved the WRONG way OFF→ON (1.1 → 4.3%) — re-check on the next replicate before trusting `throughput` |
| `felix_it` (OFF 67/10/16) | cadence family · `terminal`/`commits` · `conv` · `selection_detail` | ON reals only; first floor now exists (2.3% cadence / 2.4% var). **Owed a second OFF real** — its ON accuracy sits 2.4-3.4 pts under a single unreplicated OFF leg, which is not yet a comparison (§A.3) |
| `fwdllm_it_unaware` / `fwdllm_it_oracular` (69/0/23) | none | **Both CLEAN at 7200s OFF, N=39.** No replicate and no ON run — floor UNMEASURED |
| `fedbuff_it_unaware` / `fedbuff_it_oracular` (72/3, 73/2) | `v2` · `convergence` (+ `per_round_advance` on unaware) | Still 1200s OFF smokes, N=38. `v1c` flat on both, but at t=1.68/−0.06 the slope is unresolved. **H8 cannot be answered at 1200s** |

### Next session

> **Update in place on every run — overwrite, never stack a new dated block below.**

**What the last batch settled — three hypotheses closed at once (§G).** **H13 is CONFIRMED as the DOMINANT
term of the replicate floor, and closed**: `jvp_eval_mode` collapsed `felix_round`'s cadence floor 13.3% →
0.6% and every measured variance floor, at no accuracy cost against any measured OFF band (§A.3). **H12a is
CONFIRMED by its own asymmetry** — only the loss-derived-weight baseline moved; the integer-staleness ones had
no floor to lose and did not. **H11 is validated live**: the boundary trace reads 30 unique picks / 0
re-picks and `selection_detail` is green, so the H11 handoff file is deleted. The proposed
"forward-gradient training is not reproducible, widen the tolerances" invariant is dead — this was a bug, and
a flag closed it.

**Live hypotheses — each with the observation that would falsify it.** State the prediction BEFORE the run;
a hypothesis that can only be confirmed is not one (§D-9).

**H14 — ONE quantity drives the whole cadence family: sim's variance at the FIRST aggregation of each data
bin runs 7.3% below real's.** Everything else follows arithmetically through §D-34's identity
(iterations-per-bin ≈ var@it0 ÷ threshold): 7.3% less variance ⇒ the gate trips a bin early ⇒ 5.4% fewer
iterations per bin (11.86 vs 12.53) ⇒ 5.4% fewer cycles over matched work, which is what `v1`, `v1b`,
`v2` and `cohort_sequence`'s count gate each report in their own units.

**Ruled OUT on disk, all of it at matched (bin, iteration):** pool SIZE (exactly 10 contributors both sides,
every cycle) · staleness (0.98 vs 0.98 at it0, KS 0.002) · contributor CONCENTRATION (gini 0.31 vs 0.30, 58
vs 60 distinct) · contributed trainer SPEED (mean 6.85 vs 6.72) · utility (+0.9%) · `g2_grad_pool_size` (0.9%)
· `v3_cached_v_pool` (KS 0.0) · `v4`/`v5`.

**And contributor IDENTITY is ruled out by the strongest control available.** Two REAL legs disagree about
who contributes exactly as much as real disagrees with sim — it0 overlap 0.362 vs 0.385, both sitting on the
independent-draw floor of 0.333 — yet the two reals land within **0.25%** on var@it0 while sim is **6.7%** off
(paired per-bin median, §D-38). Different contributor sets demonstrably do NOT move this quantity.

So every observable INPUT to the variance matches within the real↔real spread while the OUTPUT does not,
which is §C's decomposition tree pointing at a grad-pool accumulation bug in sim. The gap also DRIFTS rather
than sitting flat (+5.6% → −15.5% → −9.2% across progress quartiles), so it is not a fixed computational
offset. **FALSIFIED IF** an ON sim leg on `fwdllm` shows the cadence family failing with var@it0 matched, or
if the residual reverses sign under a different aggregation rate.

**Next step is NOT a run** (preamble): the gradient values themselves are not in telemetry, so this needs
either a bench repro driving the real `_compute_var` over a controlled pool, or per-cycle telemetry capturing
the pool's identity AT THE INSTANT var is computed (§D-20 — check for an existing span first).

**H15 — `felix_it`'s single 84.72% OFF leg is a lucky draw, not a real ON regression.** Its OFF side has never
been replicated, and the comparable uncapped baseline had an 11-point OFF band. **FALSIFIED IF** a second OFF
leg lands within ~1 pt of 84.72%, which would make the ON pair a genuine 2.4-3.4 pt drop and put a shared-mask
JVP back on the table for the `new`-rate baselines.

**H8 — `fedbuff_it_unaware`'s H6 signature is unresolvable below 7200s, not absent.** Unchanged, still open.
**FALSIFIED IF** the 7200s λ flips sign. Needs a replicate PAIR, now readable since the floor is no longer
dominated by gradient noise.

### Roadmap to parity on all nine — 3 nodes

> **Update in place. Delete a step the moment its exit criteria are met and its findings are in §A/§G.**
> Goal is parity on all nine, fast. Order is by *information per node-hour*, not by baseline.

**Where the nine stand.** 1 ON parity pair with three REAL fails (`felix_round` 71/3) · 4 with ON reals only,
so no parity verdict until they get a sim leg (`fwdllm`, `fluxtune`, `fedbuff_round`, `felix_it`) · 2 clean at
7200s but OFF and unreplicated (`fwdllm_it_*` 69/0) · 2 still on 1200s OFF smokes (`fedbuff_it_*`).

---

#### Step 4a — H14's grad pool. **THE ACTIVE TASK, and it needs no node.**

One quantity explains four fails. **The summary statistics are exhausted — every one of them matches**, so
the next evidence has to be the pool's individual entries.

**Also ruled out, on disk:** the AGGREGATION RATE (computed from telemetry through the real `weight_factor`:
rate@it0 0.4643 / 0.4641 real vs 0.4649 sim, i.e. sim marginally HIGHER, which would push variance up) and
the raw GRAD MAGNITUDE (`g1_grad_norm` 2278.8 vs 2276.7, **0.09%**, KS 0.006).

**So the pool has the same size, the same magnitude and the same weights, but less DISPERSION** — variance
reacts to spread, not scale. That is the one thing no existing summary reports.

The instrument already existed and had never been wired to a run: `build_var_calc` emits each pool entry's
norm beside the reduction's output, but only under global DEBUG, which would also flood the log and perturb
timing. It is now behind `var_calc_audit` (default OFF, ~one GPU→CPU sync per pool entry), set on both legs
of the `felix_round` pair. `diff_var_pool.py` pairs cycles at identical (bin, iteration) and separates the two
possible roots: **inputs match but output differs ⇒ the reduction; inputs less dispersed in sim ⇒ pool
assembly.** 5 tests.

⚠ **This needs the FULL 7200s, not a smoke.** The gap DRIFTS (+5.6% → −15.5% → −9.2% by progress quartile),
so a short run can show it with the opposite sign — §D-25's accumulating case, where a short run reads a
false PASS.

Do NOT widen a tolerance to make these four pass — they now sit 9x above a measured floor. Remove
`var_calc_audit` from both yamls once H14 closes.

```bash
cd lib/python/examples/fwdllm/expt_scripts

# ---- node A: H14's instrumented pair (var_calc_audit already set on both legs) ----
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_round --yes && \
bash run_sequential.sh --mode sim  --max-runtime-s 7200 --only felix_round --yes && \
python diff_var_pool.py $(ls -d ../experiments/*_felix_round_n100_*_real | sort | tail -1) \
                        $(ls -d ../experiments/*_felix_round_n100_*_sim  | sort | tail -1)

# ---- node B: H15's control — felix_it with the flag OFF, the ONLY leg that should be ----
#   the code default is now ON, so this leg needs an explicit false, and the
#   preflight will (correctly) warn about it. Restore the yaml straight after.
sed -i 's/^\( *\)jvp_eval_mode: true/\1jvp_eval_mode: false/' felix_it_n10_smoke.yaml
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_it --yes
git checkout -- felix_it_n10_smoke.yaml            # put the default back immediately
grep -h 'JVP_EVAL_MODE' $(ls -d ../experiments/*_felix_it_n100_*_real | sort | tail -1)/*trainers.log | sort | uniq -c
python replicate_floor.py --mode real --baselines felix_it
```

The `grep` is the point of §D-49: it must report **100 trainers at `jvp_eval_mode=False`** and
`live_dropout=19/20`. An ON control measures nothing, and this is the leg that decides whether the promotion
holds.

#### Step 4b — ON sim legs, once 4a has a verdict.

Every baseline with ON reals needs its charge profile re-derived from THOSE reals before its sim leg — a sim
leg graded against OFF-derived charges reports on the profile, not the code. One `&&`-chain per node, because
the profile feeds the run feeds the grade.

| node | jobs, in order | buys |
|---|---|---|
| **A** | `fwdllm` re-profile → sim ON → parity | **H14's test** — same three fails as `felix_round`, or a different set. Its 0.0-0.3% floor makes any verdict readable |
| **B** | `fluxtune`, then `fedbuff_round` (each: re-profile → sim ON → parity) | `drain_wall_budget` under ON charges; and whether `fedbuff_round`'s bins-floor regression is real |
| **C** | `felix_it` real **OFF** ×1 → `felix_it` re-profile → sim ON → parity | H15's control first (the missing OFF replicate), then the parity leg |

```bash
cd lib/python/examples/fwdllm/expt_scripts

# ---- node A: fwdllm ON sim = H14's test ----
cp ../sim_charge_profiles/fwdllm.yaml ../sim_charge_profiles/fwdllm.yaml.off-bak && \
python profile_sim_charges.py $(ls -d ../experiments/*_fwdllm_n100_*_real | sort | tail -2 | sed 's/^/--real-run /') \
    --out ../sim_charge_profiles/fwdllm.yaml --only-observed && \
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only fwdllm --yes && \
python run_parity.py --yes --baselines fwdllm

# ---- node B: fluxtune, then fedbuff_round ----
for b in fluxtune fedbuff_round ; do \
  cp ../sim_charge_profiles/$b.yaml ../sim_charge_profiles/$b.yaml.off-bak && \
  python profile_sim_charges.py $(ls -d ../experiments/*_${b}_n100_*_real | sort | tail -2 | sed 's/^/--real-run /') \
      --out ../sim_charge_profiles/$b.yaml --only-observed && \
  bash run_sequential.sh --mode sim --max-runtime-s 7200 --only $b --yes && \
  python run_parity.py --yes --baselines $b ; \
done

# ---- node C: the felix_it OFF control (H15) first, then its ON sim ----
#   flip jvp_eval_mode back to False in felix_it's REAL yaml for this one leg, then restore
bash run_sequential.sh --mode real --max-runtime-s 7200 --only felix_it --yes ; \
python replicate_floor.py --mode real --baselines felix_it ; \
cp ../sim_charge_profiles/felix_it.yaml ../sim_charge_profiles/felix_it.yaml.off-bak && \
python profile_sim_charges.py $(ls -d ../experiments/*_felix_it_n100_*_real | sort | tail -2 | sed 's/^/--real-run /') \
    --out ../sim_charge_profiles/felix_it.yaml --only-observed && \
bash run_sequential.sh --mode sim --max-runtime-s 7200 --only felix_it --yes && \
python run_parity.py --yes --baselines felix_it
```

⚠ **Glob precision.** `*fwdllm*_real` also matches `fwdllm_it_unaware`/`_oracular`, which would profile one
baseline's charges from another's real; `*_<baseline>_n100_*_real` is the safe form. Node C's re-profile must
run AFTER its OFF control leg but must pick the two ON reals — check what the `ls` resolves to before
launching, or run the control on a different node.

⚠ **Node C's control leg must actually be OFF.** Flip the knob in `felix_it`'s real yaml for that leg only and
grep the trainer log for it afterwards (§D-49) — an ON control measures nothing.

**Exit criteria.** Each ON sim leg produces a §A.2 row. H14 resolves on node A. H15 resolves on node C's first
leg. `.off-bak` is the revert path if the flag ever goes back off.

#### T1 — the flag decision, now answerable

- **T1.1 — DONE (§G).** Promoted to default ON, declared in every baseline yaml, and verified on the real
  model rather than inferred from the config. H15 remains the one open check against it.
- **T1.2 — every OFF result is a different config.** §A.1's board, all nine tolerance sets and the charge
  profiles were measured dropout-live. A full ON re-grade is ~2 nights on 3 nodes; never read an ON row
  against an OFF row.
- **T1.3 — DONE.** ON charge profiles are ~3% cheaper on the JVP, direction as predicted; the re-profile step
  is now standing procedure before any ON sim leg (Step 4).
- **T1.4 — `fedbuff_it_*` stay OFF at 1200s** until the promotion decision; getting them onto 7200s pairs is
  the next night's work either way.

**SHORT TERM — the remaining gaps, in priority order.**
1. **`fedbuff_it_*` off 1200s** (7200s pairs) — the LAST two with no 7200s evidence; H8 is unanswerable
   without it, and their charge profiles need regenerating once those reals exist.
2. **Floors for the four `*_it_*`** — the only baselines with no replicate at all.
3. **Full 9-baseline re-grade** at 7200s, ON, with every tolerance recalibrated against the ON floors (§A.3).
   This is the parity sign-off.
4. **`async_oort`→`AsyncSelectorBase` integration confirmation** (`--only felix_round,felix_it`) — unit tests
   cover the mechanism, not a live comparison.

**Standing rules for this campaign.** A real↔real floor needs **no sim leg** — halve the cost. Compare
flag-OFF vs flag-ON **at one duration** (§C). One mechanism per run when a fix could perturb another baseline.
Never spend a run on a question a bench repro can answer (preamble).

### Other open items

**Tolerances, post-ON**
- **Recalibrate every DIST tolerance against the ON floors (§A.3), not the OFF ones.** Some are now far too
  loose to catch anything — `v1`'s 15% sits 25x above `felix_round`'s 0.6% ON floor — and `v2`'s 2% is still
  under two baselines' floors. Neither direction is safe to leave; this is part of the sign-off re-grade.
- **`convergence` was ungradeable at its 5% `acc_tol` OFF** — below the real↔real accuracy gap on both
  baselines that had a replicate. It PASSES on the ON pair, and the ON accuracy bands are 0.99-5.25 pts, so
  it is worth re-deriving the tolerance rather than assuming the old one.

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
  expect them at any duration whose graded window straddles a lap boundary. Both are clear on the ON pair
  (`terminal_state` passes, `preferred_duration` SKIPs).

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
- **A run dir cannot say which training config produced it.** `jvp_eval_mode` lives in the trainer's
  `config_overrides`, which the runner never dumps — not in `aggregator_config.json`, not in `snapshot.yaml`;
  the only record is 100 lines in the trainer log. `replicate_floor.py` now greps that log, which works but is
  the wrong layer. Dump the resolved TRAINER config into the run dir and key off it. Same class as §F-18.
- **Enforce §F-18 mechanically: a per-baseline knob CONTRACT.** Two correctness-path knobs went missing from
  yamls and were caught only by reading telemetry days later, so those runs were not fairly comparable. The
  mechanism mostly EXISTS — `run_sequential.sh`'s preflight `checks[]` already blocks a launch and ships both
  patterns; the gap is that `condition_fp` hashes only CLI-patched knobs, so yaml-only knobs are invisible.
  Three homes, none new: `test_baseline_readiness.py` → preflight `checks[]` (first tenant landed, §G) →
  parity `--validate`. **The hard part is "missing" vs "legitimately N/A"** (`sim_charge_profile_path` is
  sim-only, `reselect_cadence: round` round-only, `trackTrainerAvail` oracular-only), so applicability must be
  DECLARED, not diffed — sketch: a `knob_contract` block in `_metadata/baselines.yaml` read by all three
  layers. Open: where it lives; error-vs-warn; whether `condition_fp` absorbs it; who declares a new knob.
- Regenerate `fedbuff_it_*`'s charge profiles once their 7200s reals exist (`fwdllm_it_*` done, 08-02). Not
  urgent — the charge is not materially duration-sensitive.
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
- **DIST tolerances are now sized from each baseline's measured replicate floor, not hand-typed (§D-24).**
  `replicate_floor.py --profile-out` writes `parity_floors/<baseline>.yaml`; the checker tightens toward
  `3x floor`, never past 2% absolute, never looser than nominal, and SKIPs a rung whose floor has swallowed
  its tolerance instead of returning a coin flip. Each gated result carries its floor, nominal and effective
  tolerance. **This caught a real defect immediately**: `v1` had been passing a 5.4% gap against a 15% gate
  calibrated when the floor was 13.3%; at the ON floor of 0.6% it fails. No floor file ⇒ nominal gates, so
  async_cifar10 is inert. 8 tests.
- **`jvp_eval_mode` PROMOTED to default ON (T1.1), and declared in all 22 baseline yamls.** Both read sites
  default True; the launch preflight now warns on an explicit OFF (the H13 defect) instead of on absence.
  **Verified rather than assumed**: a per-trainer census logged from inside the eval block counts the model's
  live dropout leaves, and on the real DistilBERT+adapter stack it reads **19 of 20 live before, 0 inside** —
  `model.training` reads False throughout, which is why the flag alone was never evidence (§D-47). A yaml
  contract test now fails if any baseline drops the knob. 11 tests.
- **H13 CLOSED — eval mode is confirmed as the floor's DOMINANT term, at no accuracy cost.** ON reals on five
  baselines: the worst cadence floor fell 13.3% → 0.6% (22x) and every measured variance floor fell with it
  (worst 21.3% → 6.6%), while peak accuracy landed inside or above every OFF band that has a replicate behind
  it (§A.3). The proposed tolerance-widening invariant is RETIRED — this was a bug, and a flag closed it.
- **H12a CONFIRMED by the asymmetry it predicted.** Only the baseline whose aggregation weight is
  loss-derived lost its cadence floor; the integer-staleness ones had none to lose and did not move
  (0.0 → 0.3%, 3.9 → 4.1%). Cadence tracks the aggregation rate, accuracy tracks gradient noise — different
  floors, different causes, and the fix separates them cleanly.
- **H11 CLOSED — validated live.** The ON sim leg's round-2 boundary reads **30 unique picks, 0 re-picks**
  against c=30, `selection_detail` is green and the occupancy rungs did not regress. `HANDOFF_H11.md` deleted.
- **`replicate_floor.py` splits groups by `jvp_eval_mode`, read back from the trainer log.** It pooled every
  same-duration leg of a baseline, so ON legs would have been averaged with OFF ones and the exit criterion
  would have measured the flag, not the floor (§D-45). The knob is in no config file in the run dir — a
  provenance gap worth closing separately. 4 tests.
- **`jvp_eval_mode` wired end to end for the A/B.** Trainer knob in the **trainer** override block of the 10
  yamls (real+sim × the five 7200s baselines); the launch preflight BLOCKS a mismatched or one-sided pair and
  warns when absent (= code default, dropout live). Both failure modes negative-controlled. One line per yaml
  to revert.
- **H13 fix landed behind `jvp_eval_mode`, default False = today's behavior.** Eval mode is applied before the
  model is made functional and held for the whole training loop (the functional transform deep-copies the
  module, so a later toggle never reaches what the JVP evaluates), then every module's own flag is restored —
  a blanket re-enable would invent a state the model never had. Covers the utility path too. 6 tests.
  **NOT bit-identical when ON, NOT parity-only — A/B first (§B).**
- **H13 CONFIRMED on the real stack once the probe was rebuilt to run the PRODUCTION model (§D-47).** The
  probe had forced inference mode, silencing the very dropout at issue; it now builds as-created, carries an
  eval-mode arm, records a live-dropout census, reads its verdict off the within-process column and suppresses
  the cross-process table (different seeds = different work, not nondeterminism). Numbers in §B. 7 tests.
- **H11 over-dispatch FIXED.** The boundary's release path cleared both re-pick guards before rebuilding the
  outstanding set from them, so top-up selections re-picked still-training ends. Sim now also folds in the
  dispatched-not-yet-returned record — the half real's equivalent always carried, which is why real never had
  the bug. 3 tests (2 fail without). Validated live on the ON sim leg — 0 over-dispatch.
- **Watchdog runtime raised above the run budget on all eight real yamls** — a watchdog equal to the target
  duration can kill the run it is meant to outlive.
- **H12 probe + two candidate flags, default OFF, byte-identical off** — a bench A/B across concurrent
  processes with no FL run, plus fp32-outside-autocast and strict-determinism knobs. 23 tests. The amplifier
  is CONFIRMED; neither flag touches H13's source (§B).
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
  pool/order-dependent draw was not reproducible). Integration-level real+sim confirmation still open (§B).
