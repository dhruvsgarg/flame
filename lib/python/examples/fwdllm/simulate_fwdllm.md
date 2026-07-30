# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 newly-ported
fedbuff/felix-lineage baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2),
then beyond syn_0 (Phase 3). Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf,
sim barrier redesign, delay-factor calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared
parity methodology (ladder, roles/tiers/gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — maintaining this doc
> **Read §F before any fresh debugging session.** Re-deriving a rule already locked in §F wastes a
> session and risks landing a fix that violates one; check §F first, then start from telemetry (below).
> **Short runs (< 3600s) are for early-onset issues, not full parity sign-off.** A run below the §C
> min-duration bar for a rung can still surface a mechanism bug (wrong dispatch pattern, missing guard,
> RNG desync) that would persist at any length — chase those now. Don't wait for a multi-hour run to
> reveal something a 30-minute run already showed; conversely don't grade duration-gated rungs
> (`terminal_state`, `conv`, cadence distributions) off a short run — they need the run-length §C calls for.
> **Correctness per mode first; parity is the consequence, never the goal — for a MATCH or a DIVERGENCE
> alike.** Real and sim must each be independently correct against the configured intent (e.g. `c`
> trainers training at any instant). Aligning the two by making either wrong — or by preserving a defect
> symmetrically — is a regression even when every rung is green. Matched-but-wrong is the hardest failure
> to find: parity reports it as a pass. The mirror case is just as easy to get wrong: a DIVERGENT rung
> names two sides that disagree, never which one is at fault — don't default to "make the outlier match
> the other side." Verify each side against its OWN independent absolute signal (an existing invariant,
> a self-consistency tripwire, a duck-typed guard already in the code) before deciding which side to
> change (→ §D-9). A fix that's really just parity-imitation — forcing the correct side to reproduce the
> broken one's number — is a regression dressed as a green rung.
> **Parity findings/fixes only** — design decisions, roadmap items, calibration derivations belong in
> FWDLLM_DESIGN.md.
> **Living doc, not a changelog** — §A/§B describe the state *right now*, rewritten in place, never
> stacked as dated "UPDATE" blocks. Full history is `git log` on this file + the parity JSONs.
> **§A** scoreboard only (pass/fail/skip + key-rung table + ≤2-line caption) — no prose essays; analysis
> goes to §B. Refresh whenever `run_parity.py` runs a >3600s pair, per baseline (carry stale numbers
> forward, tagged STALE).
> **§B** open issues, per baseline, current-state only, updated in place. An issue lives in exactly one
> place: open (§B) xor closed (§G, one line) — never both, never a stale copy left behind.
> **§C** method + run-length budget — timeless; edit only when the debugging method itself changes.
> **§D** durable lessons — transferable diagnostic patterns (see X → it means Y → discriminate by Z);
> update in place, never append near-duplicates. Shared (non-fwdllm) patterns live in PARITY.md.
> **§E** dead ends — falsified hypotheses, one line each, append-only (a dead end never un-dies).
> **§F** locked invariants — always-true / always-do rules; numbers are cited elsewhere, keep them stable.
> **§G** closed items, one line each (problem → fix, ≤30 words). The moment a rung flips or a hypothesis
> resolves, write the line and delete the §A/§B entry in the same edit.
> **Every fix**: ground claims in telemetry already on disk before instrumenting or running; fix root
> causes, not symptoms; never launch an experiment directly (print the command for the operator to run);
> always use the `dg_flame` conda env for python/pytest/analyze_run.py; ship new telemetry with its plot +
> pytest in the same change (a field with no reader in `analyze_run.py` is dark data).
> **Runs happen on a separate node the operator controls** — never check GPU/hardware specs or launch/babysit
> a run yourself; ground claims in on-disk telemetry or the command you hand back. Each run costs the
> operator real time, so before handing one back, sweep for other open non-blocking, non-conflicting fixes
> and land them in the same pass. **Trust the operator on run scheduling**: assume no run is currently
> pending/in-flight unless told otherwise (don't check for live processes on this node), and that each
> baseline's real/sim pair ran ONE AT A TIME per node (never two runs concurrently on the same node) — one
> or more nodes may run different baselines in parallel, so cross-baseline directory timestamps can
> legitimately interleave; only same-baseline real→sim ordering is meaningful for sequencing.
> Run artifacts live under `lib/python/examples/fwdllm/experiments/run_<timestamp>_<name>_<syn>_<real|sim>/`.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) (methodology + rung catalog §F),
[UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) (availability substrate),
[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md) (build plan/roadmap/calibration),
[fluxtune_contributions.md](fluxtune_contributions.md) §8 (training-stability/convergence ledger — check
before opening a new stability investigation here).

**Comparator — discovers the latest real/sim pair per baseline and runs the shared parity battery:**
```bash
cd lib/python/examples/fwdllm/expt_scripts
python run_parity.py                        # all baselines, latest pairs, confirm
python run_parity.py --baselines fluxtune   # one baseline
python run_parity.py --yes                  # skip the confirm prompt
python run_parity.py --validate             # + live-run checks (staleness/vclock_now)
```
Rung catalog: PARITY.md §F. **Not redefined there:** per-stage wall-budget instrumentation
(`drain_wall_budget`, `trainer_phase_wall_budget`, `step_timing_breakdown`, `aggregation_compute_wall`) is
ONE-SIDED (`sim<=real`) where sim should collapse a real-transport phase to ~0, DISTRIBUTIONAL where it's
genuine shared compute. Implementation-level reference (tiers, the `pctl_band_ok` band-escape primitive
and its `min_abs` calibration rule, full wall-budget/timing rung table):
`async_cifar10/scripts/parity/PARITY_CHECKER_README.md`.

---

## §A  Score — refreshed 2026-07-29 (post §D-15 mid-cycle-redispatch validation pairs)

**Latest run per baseline** (`run_parity.py`; ✓/✗/– = pass/fail/skip; PARITY.md §F). `fluxtune`,
`fwdllm`, `fedbuff_round`, `felix_round` now have a 5400s pair (up from 3600s); the other 6 remain
3600s+ n=100/c=30 except `fwdllm_plus` (STALE, 07-23). Open fails: §B.

**`fedbuff_round`/`felix_round`/`fluxtune` re-run sim-only against the SAME real logs — this IS the §D-15
validation run.** `retask_before_close` = 0% on all three (was 78-90%) and the §D-15 `busy+idle` identity is
now MATCHED on all three (§D-16): the root fix holds and did NOT overshoot. `fedbuff_round` lands on target
(6 rungs green). The `felix_round`/`fluxtune` 13-16% residual is TWO UNRELATED roots, neither of them idle
— felix is variance-cadence (Root A′), fluxtune is charged contention (§D-18). `concurrency_cap`
still red on `fedbuff_round`/`fluxtune`. `fwdllm_plus` still predates the 07-27/28 batch. `fwdllm` remains
CURRENT (59/3/22, §G).

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260729_080111`/`_230952` | 5400s | 60/10/16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm/syn_0 | `run_20260729_020102`/`_033249` | 5400s | 59/3/22 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus/syn_0 STALE | `run_20260723_161647`/`_171829` | ~3600s | 61/2/21 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_it/syn_0 | `run_20260728_000201`/`_010416` | 3600s | 63/6/16 | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_unaware/syn_0 | `run_20260728_011134`/`_021316` | 3600s | 59/4/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_unaware/syn_0 | `run_20260728_015339`/`_025553` | 3600s | 67/1/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260728_022355`/`_032547` | 3600s | 58/5/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_oracular/syn_0 | `run_20260728_032938`/`_043152` | 3600s | 64/4/18 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fedbuff_round/syn_0 | `run_20260729_034223`/`_220025` | 5400s | 62/4/21 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_round/syn_0 | `run_20260729_055151`/`_223623` | 5400s | 56/10/21 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. Open fails: §B.
**Counts refreshed 07-30 against the SAME pairs after two checker fixes** (§G) — no re-run. `felix_round`'s
V1 was a FALSE PASS and is now honestly red (10.69 vs 12.54, +14.7%); `selection_detail` was a false FAIL on
all round baselines and `concurrency_cap` a false FAIL on `fluxtune`, both now green; `concurrency_cap` also
grades REAL now, on every baseline, with no fresh run (§D-20). Net: fedbuff 60/6→**62/4**, felix
57/9→**56/10**, fluxtune 59/11→**60/10**.

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.
> **Starting a session with fresh logs? Go straight to "▶ NEXT SESSION" below this table.**

| baseline | open fails | next step |
|---|---|---|
| `fedbuff_round` (62/4/21) | `concurrency_cap` (INV, sim 35 distinct ends vs c=30 **and** 553 same-end concurrent dispatches, ALL at the lap boundary — §D-20) · `v1_iter_per_data_id` (KS 0.201 vs 0.200, mean 1.9% — marginal) · `g2_grad_pool_size` (KS 0.211) · `step_timing_breakdown` (D-1) | `concurrency_cap` is the only real item; V1/g2 are hairline. `selection_detail`/`v1b` went green on the §G checker fixes |
| `felix_round` (56/10/21) | `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/`terminal_state`/`v1_iter_per_data_id`/`v1b`/`g2_grad_pool_size`/`v2_var_trajectory` — ALL one root: +14.7% cycles-per-bin (Root A′) · `cohort_sequence` = §D-2 | ONE real defect (A′), now honestly measured on 4 rungs instead of hidden. Next probe in the A′ block |
| `fluxtune` (60/10/16) | `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/`terminal_state` — 67% attributed to charged `drain_tail` contention (§D-18) · `drain_wall_budget` · `preferred_duration`/`phase_gpu_compute` · `v2_var_trajectory` | NOT felix's root — cadence matches (0.994), cost/cycle doesn't (1.154). Decide charge-the-floor vs relax (§D-18). V-family all green |
| `fwdllm`/`fwdllm_plus` | `drain_wall_budget` (GATING)/`step_timing_breakdown`/`agg_step_timing_breakdown` (DIAG) — pre-existing, D-1 co-location contention, unresolved (`sim_model_agg_compute_time` charges sim's own contention-inflated live span onto the vclock — the "charge-the-floor vs relax" decision is still open — now priced, §D-18) | Decide it here and on fluxtune together (§D-18): same charge, one call. `throughput`/`terminal_state`/`total_commits` CLOSED 07-29 (§D-13, §G) |
| `fwdllm_it_unaware`/`fwdllm_it_oracular` | `terminal_state`/`total_commits`, new at 3600s | likely same §D-1 contention family (shares fwdllm's sync dispatch path) — not independently diagnosed |
| `felix_it` | `total_commits`/`terminal_state`/`cohort_sequence`/`convergence` | run §D-16's two-factor split on its own pair FIRST; profile a charge only if `s/cycle` is the failing factor (§D-3, §F-13) |
| `fedbuff_it_oracular` (64/4/18) | `total_commits`/`terminal_state`/`v2_var_trajectory`/`convergence` | same: split before charging. Iteration cadence, so §D-17 does not apply |
| `fedbuff_it_unaware` (67/1/18) | `convergence` only | duration-gated (§C bar 2h+), not a bug |

**07-30 — the "overshoot" was never one root, and never idle. §D-15 is fully validated; §D-16 splits the
residual into two independent roots.** Re-derived from the three 07-29 pairs on disk, no new run. §D-15's
own `c/(busy+idle)` identity per contribution is now MATCHED on all three (felix sim 2.93 vs real 2.87
s/cycle, fluxtune 9.59 vs 9.73, fedbuff level) — **"decompose the surplus idle" was chasing a term that had
already closed.** Decomposing `s/round = (s/cycle) × (cycles/bin)` instead separates the baselines cleanly:

| baseline | s/cycle sim/real | cycles/bin sim/real | s/bin sim/real | root |
|---|---|---|---|---|
| `fedbuff_round` | 0.985 | 1.037 | 1.022 ✓ | — (passes) |
| `felix_round` | 1.023 | **1.141** | 1.167 ✗ | variance cadence → §D-17 |
| `fluxtune` | **1.154** | 0.994 | 1.147 ✗ | charged contention → §D-18 |

Felix's cost per cycle is at parity and its cadence is not; fluxtune's is the exact mirror. Sizing either
one off `implied_per_commit_overhead_s` (−0.50s / −0.99s) was reading an EMERGENT aggregate as if it named a
mechanism. `retask_before_close` = 0.0% on all three (was 78-90%). `concurrency_cap` now breaches on
`fedbuff_round` ONLY (35 distinct ends vs c=30, plus 553 same-end concurrent dispatches, all at the lap
boundary) — the one remaining INV, unrelated to either root; `fluxtune`'s was a measurement phantom (§D-20).

**ROOT A (`selection_detail`) is NOT a code defect — the cohort pin is correct; the RUNG is mis-windowed
and the RUN is too short to grade it. §D-17.** `_cohort_cache_key()` (`fwdllm_aggregator.py:3367`) returns
`self._round`, which increments only at `data_id == total_data_bins` (`:2386`) — one full ~150-bin lap.
**Operator ruling 07-30: this is correct and intended.** `reselect_cadence: "round"` means once per ROUND
(all data bins), not per aggregation or per data bin; the fedbuff/felix ports change only the trainer's work
(forward perturbations vs backward pass), everything else stays as the source baselines. `rounds: 50` is the
configured experiment — the cohort would rotate 50×; a 5400s run simply covers ~1 round. Two real
consequences follow, neither a sim bug:

**CHECKER BUG — the whole of it, both baselines. FIXED 07-30, `selection_detail` green on all three, no
run needed.** The rung pooled each side's WHOLE run instead of the matched logical budget, so it graded
which side crossed a lap boundary rather than how either selects. Every divergent selection event on both
baselines is out-of-budget:

| baseline | selections in budget | full run | max progress key vs budget N |
|---|---|---|---|
| `felix_round` | real 1 / sim 1, 30 chosen both | real 22 / sim 1 | real (2,13) vs N=(1,146) |
| `fedbuff_round` | real 1 / sim 1, 30 chosen both | real 1 / sim 32 | sim (2,149) vs N=(1,148) |
| `fluxtune` | real 12150 / sim 12090 | 13385 / 12175 | — (iteration cadence) |

Inside matched work all three agree exactly. **Grading rule (operator, 07-30): a 1.5h run IS how this
baseline operates — no SKIP, no "underpowered". PASS iff the number of cohort re-draws AND the cohort sizes
both match; FAIL if either does. Trivializing some comparisons is acceptable where it is true to the
system.** So the event COUNT is now graded too (`rel_diff_n_selections`), at matched work; full-run counts
are reported alongside so the window hides nothing. `g2_grad_pool_size` was never part of this group — it is
the honest per-completed-bin iteration count, i.e. Root A′ below (§D-19). RETRACTED from earlier drafts of
this entry: "70 of 100 trainers never train" as a correctness defect (within one round, `c`=30 of `n`=100
pinned is the specified behavior), and "fedbuff's 32 re-draws are INSIDE the budget" — they are not; that
read compared vclock SECONDS against a PROGRESS key.

**ROOT A′ (`felix_round`'s throughput family) — the baseline's ONE genuine code-side defect, cause open.**
With Root A ruled correct-by-design, this stands alone: nothing upstream explains it. Felix needs
**+17.3% more variance-check cycles per completed bin** (real 10.69 vs sim 12.54 over the matched first 147
bins), and that alone IS the whole 13-16% `throughput`/`per_round_advance`/`overhead_residual`/
`total_commits`/`terminal_state` residual — its cost per cycle is at parity. Its `var` curve sits above
real's at EVERY iteration index (it=1: 3.59 vs 3.07; `var_threshold` 0.3 in both), so it crosses later.
**The sharpest remaining question in this doc:** the ratio GROWS with bin index (0.79 → 1.07 → 1.19 → 1.11
→ 1.48 → 1.42 across 25-bin blocks) — sim starts CHEAPER and compounds past real, i.e. a diverging
optimization trajectory, not a static statistical offset. Cohort composition is refuted as the driver (§D-2
retraction): both modes commit from the SAME pinned 30 for the whole graded window with the gap fully
present, and when real's set rotates at the round boundary its var goes UP (2.95→3.83) and cycles/bin UP
(10.62→13.50) — the opposite direction. Next probe (no run needed): walk the early bins where the ratio is
still <1 and find where the two trajectories first separate, on the `agg_round` `var`/`grad_pool_size`
series — sim is CHEAPER for the first ~25 bins, so the divergence has an onset, not a constant offset.
**Probe run 07-30, 10-bin blocks — the onset is around bin 30, and cycles/bin is the cleaner signal:**

| bins | cycles/bin real→sim | var@it=1 real→sim |
|---|---|---|
| 0-10 | 9.50 → 9.60 | 2.791 → 2.146 |
| 10-20 | 9.20 → 9.20 | 2.318 → 1.887 |
| 20-30 | 9.30 → 9.50 | 3.251 → 3.061 |
| 30-40 | 9.30 → **11.20** | 2.480 → 3.048 |
| 40-50 | 10.00 → 10.90 | 3.391 → 3.221 |
| 50-60 | 9.70 → **11.60** | 2.133 → 2.545 |

Cycles/bin tracks to within 2% for the first 30 bins and then steps to +17-20% and stays there; the `var`
signal is far noisier block-to-block and does NOT move monotonically with it. So chase what changes at bin
~30, not the variance statistic itself — and do not assume `var` is the mediator just because it is the gate.
NOT root-caused; this is a lead, not a mechanism.

**ROOT B (`fluxtune`): `drain_tail` is charged onto the vclock at sim's own contention-inflated live wall —
§D-18.** From the `vclock_charge` ledger, per cycle: sim charges `drain_tail` 0.585s (`charge_source:
"live"`) against real's measured 0.330s — **1.77×** — plus `fedavg` 0.078 vs 0.059 and profiled
`redispatch_turnaround` 0.124. Sim charge total 0.787 s/cycle vs real's actual 0.389 → excess 0.398 against
an observed s/cycle gap of 0.590, so **67% of fluxtune's residual is charged contention**. The same
over-charge exists on fedbuff (0.370 vs 0.264) and felix (0.361 vs 0.294) but is smaller and non-binding
there. This is exactly the parked "charge-the-floor vs relax" decision, and it is not DIAG-only: it is
fluxtune's throughput root, and `drain_wall_budget` (already failing on fluxtune) is its honest signal.

**FIXED 07-30 — `drain_tail`/`fedavg` charged from the REAL profile, not sim's live span.** The registry
already held both (`sim_charge_profile.yaml`, profiled 07-29) at `charge: false` on the rationale "sim runs
the identical op, so its own span IS the cost" — falsified above. Flipped ON and the call site
(`fwdllm_aggregator.py:2186`) now resolves `profiled_s` via `get_profiled_charge_s`, the same path
`redispatch_turnaround` uses (§F-26); the live path stays as the fallback when no registry is configured, so
this is inert for any baseline that hasn't opted in. **Predicted, not yet validated:** fluxtune sim s/cycle
4.422 → ~4.10 (charge 0.663 → 0.343 s/cycle), i.e. residual +15.4% → **~+7%**. Necessary but NOT sufficient —
still outside the 5% tol, and the remaining ~33% of the gap is unattributed (§B next step).
**Accuracy caveat, worth a follow-up:** the registry is family-wide and was profiled from `fedbuff_round` +
`felix_round` reals only, but real's drain_tail differs 25% across baselines (fluxtune 0.330, felix 0.294,
fedbuff 0.264 s/cycle), so fluxtune is now slightly UNDER-charged. `sim_charge_profile_path` is already a
per-baseline yaml field, so per-baseline profiles need no code change — just a `profile_sim_charges.py` pass
per real leg, all of which are on disk.

**`concurrency_cap` RE-BASED and RESOLVED into one real defect + one phantom — §D-20.** Graded now as peak
DISTINCT ends whose `[dispatch_ts, commit_ts]` interval is open, per mode on its own clock, from
`contributor_intervals` — telemetry both legs already emit, so **real is graded on every baseline with no
fresh run** (it was `UNGRADED` before). Results:

| | real peak | sim peak | same-end concurrent dispatches (§F-25) | verdict |
|---|---|---|---|---|
| `fedbuff_round` | 30/30 | **35**/30 (0.31% of contribs) | real 0 / **sim 553** | FAIL, sim |
| `felix_round` | 30/30 | 30/30 | 0 / 0 | pass |
| `fluxtune` | 30/30 | 30/30 | 0 / 0 | pass (was a phantom) |

`fluxtune`'s old "31/30 at 10% of dispatches" was 1218 warnings, one per cycle, **every one a false
positive**: `_outstanding_dispatch_count()` counted ends that had committed in virtual time but whose slot
the agg-goal boundary had not yet released (deliberate, §D-15). Fixed by subtracting `_sim_committed` in the
cap arithmetic only — the pending set itself is untouched. `fedbuff_round`'s is real, and localizes exactly:
**all 1218 over-cap samples fall in vclock 5319-5393 and the lap wrap (round 1→2) starts at 5312.** At the
round boundary sim dispatches a fresh cohort while the previous one is still in flight, and re-dispatches
ends that already have a live dispatch (553 times; the `[SIM_R1_DISPATCH]` tripwire fired 47×). `felix_round`
sim never lapped → clean; `felix_round` REAL did lap (wall 5010s) and stayed at 30/30 with mean 21.8, so
**real handles the boundary correctly and sim does not.**

**FIXED 07-30 — BACKFILL at the round boundary (operator call).** Root cause was two guards missing from the
cohort ACCUMULATE branch that the sibling REUSE branch already had: `_exclude_pending_commit` (added there
07-27, §D-8, never mirrored) and any awareness of `c` at all. So a boundary re-draw handed back a full `c`
roster including ends with a live dispatch. Fix: new `_cap_dispatch_to_concurrency()` trims the dispatch list
to `c − len(_agg_pending_commit_ref)` — same set `_exclude_pending_commit` reads, so the two can't disagree
(§F-26) — and BOTH guards now run on BOTH branches. The cohort ROSTER stays `c`; only what may go out *right
now* is capped, with the remainder dispatched as stragglers commit. Mode-symmetric and a no-op when nothing
is in flight. 5 tests; 1550 `pytest lib/python/tests` pass. **NEEDS a fresh sim leg to validate** — it
changes dispatch behavior, so unlike the checker fixes above it cannot be re-graded off the existing logs.

**Do NOT re-tune `redispatch_turnaround`** (§F-13, §D-14): its totals check ruled it out, and neither root
above is a charge magnitude.

**`preferred_duration`/`phase_gpu_compute` (fluxtune) stay filed as presumed-downstream of Root B**, and
`fedbuff_round`'s `v1`/`v1b`/`g2` as downstream of Root A — hypotheses from direction and timing, not
proven by telemetry. Re-grade after each root closes.

**`aggregation_compute_wall` (DIAG) fails on all three** — `aggregate_fedavg_s` KS 0.354-0.380 vs tol
0.300, sim mean 0.078-0.084s vs real 0.059-0.068s. Ungated and pre-existing; logged so it stops being
re-discovered as new.

### ▶ NEXT SESSION — the run to launch, and exactly what to check when it lands

Two code fixes landed 07-30 that change sim behavior and **cannot be graded off the existing logs**
(the three checker fixes already were, §A). Reals are untouched by both and already grade clean on
`concurrency_cap`, so **sim-only is sufficient** — half the machine time of a `--mode both` pair:
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode sim --only fedbuff_round,felix_round,fluxtune
```
Then `python run_parity.py --baselines <b> --yes` per baseline, and walk this list in order. **Grade each
prediction pass/fail explicitly before moving on** — a fix that moves a number the wrong way is §D-14's
sign-flip pattern and must be re-decomposed (§D-16), never re-tuned.

| # | fix under test | expected | fails if | then |
|---|---|---|---|---|
| 1 | round-boundary backfill | `fedbuff_round` `concurrency_cap` GREEN: sim peak 30/30, `sim_n_self_overlap_dispatches` 0 (was 35 and 553) | peak still >30, or self-overlap >0 | the guard is on the wrong path — check `[ConcurrencyBackfill]` fires at the lap wrap at all |
| 2 | ditto, no regression | `felix_round`/`fluxtune` `concurrency_cap` stay green; all three keep `retask_before_close` 0.0% | any goes red | the cap is starving dispatch — check for idle slots (`slot_starvation`) |
| 3 | profiled `drain_tail`/`fedavg` | `fluxtune` `throughput` residual **+13.2% → ~+7%**; `vclock_charge` shows `charge_source: profiled` for both labels | still ~+13% | the yaml has no `sim_charge_profile_path` (§G, 07-29 precedent) — check before blaming the charge |
| 4 | ditto | `fedbuff_round` stays ≤5%, `felix_round` unchanged (~14%) — its root is cadence, not charge | `fedbuff_round` goes sim-SLOW | over-corrected; §D-14, re-decompose per §D-16 |

**Known-remaining after this run — do not expect these to close:**
- `fluxtune` `throughput` at ~7% is still outside the 5% tol. Only ~67% of its gap was ever attributed
  (§D-18); the rest is unexplained. Next probe: re-run §D-16's two-factor split on the NEW sim leg and see
  whether the residue sits in `s/cycle` still, or has moved.
- `felix_round`'s Root A′ (+14.7% cycles/bin) is untouched by both fixes — it is the one open mechanism.
  Lead, not a mechanism: cycles/bin tracks real to within 2% for ~30 bins then steps to +17-20% and stays.
  Chase what changes at bin ~30. `var` is noisier than cycles/bin and is NOT a reliable mediator.
- Accuracy follow-up (no run needed): the charge registry is family-wide but real's `drain_tail` varies 25%
  across baselines (fluxtune 0.330, felix 0.294, fedbuff 0.264 s/cycle), so fluxtune is now slightly
  UNDER-charged. `sim_charge_profile_path` is already a per-baseline yaml field — generate one profile per
  baseline with `profile_sim_charges.py` against each real leg (all on disk), no code change.
- `retask_before_close` is still real-UNGRADED (it reads the dispatch tripwire, which the reals predate).
  Single-side check (§D-9), sim reads 0.0%, so not blocking — it closes on the next `--mode both` pair.

**What did NOT need a run, audited 07-30.** Every §B fail was diagnosable from telemetry already on disk;
none was blocked on missing instrumentation. Both apparent telemetry gaps were false: `concurrency_cap`'s
"needs a fresh real leg" was answered by `contributor_intervals`, which every leg already emits (§D-20).
Before ordering a run to instrument an invariant, check whether an existing per-entity span already implies it.

**07-30: felix_round's `v2_var_trajectory` — cohort composition REFUTED as the driver (supersedes the
07-28 evening reading).** The prior entry traced the 4.4% `var` mean shift to a real/sim cohort-composition
difference at matched `(data_id, iteration)` coordinates, and asked why composition diverges when selection
is meant to be deterministic. Both halves are now answered, and the causal claim does not survive:
composition diverges for the mundane §D-17 reason (the lap-wrap race, not nondeterministic selection), and
composition is NOT what moves `var` — for the first 4800s BOTH modes commit from the SAME frozen 30-trainer
set (sim's committers ⊆ real's, overlap 30, sim-only 0, identical speed mix), yet sim's `var` already runs
10-17% high at every iteration index. When real's set does widen at the wrap, its `var` rises (2.95→3.83)
and its cycles/bin rise (10.62→13.50) — opposite to what the composition theory predicts. `calculate_var()`
(`fwdgrad_utils.py:171`) remains the right function; what feeds it differently is the model trajectory, not
the contributor set (the ratio compounds with bin index). Stays OPEN and stays enforced — do not gate it to
diagnostic.

**Flag-promotion (operator call).** `sim_model_agg_compute_time` default ON (all three). `sim_sct_ordered_drain`
+ `sim_model_dispatch_queue` are fluxtune-yaml-only but model general async-transport artifacts — next: smoke
fwdllm/fwdllm_plus with both ON, confirm inert-or-better, promote to code-level default-on.

### Known gaps

- Checker invariants I1-I6 — drafted in a prior chat session, never committed to any file (checked git log
  + repo-wide grep: genuinely unrecoverable, not just unwritten). No longer blocked (root-cause landed §G),
  but re-deriving them means drafting NEW invariants from scratch, not recovering the old ones — needs
  operator input on what they were meant to check before landing as this doc's "I1-I6".

### 07-28 evening: `async_oort` re-based onto `AsyncSelectorBase` (was: known gap, blocked)

`flame/selector/async_oort.py` 2193→897 lines. Was the last selector still carrying its own full copy of
the send/recv concurrency mechanism (reclaim/cooling/eligibility/version_key guard/cleanup family) —
`async_base.py`'s docstring already flagged this file as the pre-extraction original that had since drifted
from fixes landed only on the base (R1 pending-commit guard, recv-bootstrap gate). `felix_round` (61/3/21,
clean) served as the unblocking control per this doc's prior note.

- **Kept as Oort POLICY** (unchanged): utility scoring (`fetch_statistical_utility`/`calculate_total_utility`/
  `cutoff_util`/`sample_by_util`/`sample_by_speed`), the pacer, exploration/exploitation split, the 5
  `select_type` strategies, `_keyed_weighted_topk`. Routed through new `_choose`/`_pre_choose`/
  `_concurrency_for_task`/`_task_extra_eligible`/`_selection_extra`/`_per_trainer_selection_extra` hooks.
- **Eval-task support is genuine extra state** base doesn't have (`curr_round_eval_slots_left`,
  `trainer_eval_recv_ends`) — threaded through the hooks + a `_cleanup_recvd_ends` override, not assumed away.
- **`_cleanup_removed_ends` kept as a full override**, not delegated to base: Oort's version additionally
  sweeps a departed end as a ghost from EVERY requester's `selected_ends` (base only touches
  `self.requester`'s) — a real behavior difference, not drift, so composing with `super()` would have
  silently dropped it.
- **`slot_starvation` telemetry promoted from Oort-only to `AsyncSelectorBase._handle_send_state`** (was
  already claimed "shared by every async baseline" in a prior §G entry — checked and found FALSE before
  this session; now true). `_keyed_draw` extracted as a shared primitive (`_keyed_topk` now calls it;
  Oort's `_keyed_weighted_topk` reuses it instead of re-deriving the seed-material format, §F-26).
- **Verified, not asserted**: read all 2193 lines before touching anything; ruled out composing
  `_cleanup_recvd_ends`/`_cleanup_removed_ends` naively by diffing against base line-by-line first. Full
  suite before/after: 1387/1392 passed (net +5 new test methods), 0 regressions — every touched test was
  updated to the new `_handle_send_state(ends, concurrency, ctx: SelectContext)` signature (was discrete
  kwargs), not weakened. `AsyncOortSelector` folded into `test_async_selector_base.py`'s `BUILDERS` (§F-26)
  — passes the full shared contract generically (falls back to its own first-round `select_random` path
  when no `PROP_STAT_UTILITY` is set, same as fedbuff/async_random's `_keyed_topk`).
- `OracleSelector(AsyncOortSelector)` (`flame/selector/oracle.py`) only overrides `calculate_total_utility`
  + a few `__init__` attrs — untouched by the port, confirmed via its own method signature staying identical.
- **Not yet done, deliberately out of scope**: a fresh `felix_round`/`felix_it` real+sim run to confirm the
  port is parity-neutral at the INTEGRATION level (unit tests confirm the MECHANISM is preserved, not a live
  real/sim comparison). Hand back to operator:
  ```bash
  cd lib/python/examples/fwdllm/expt_scripts
  bash run_sequential.sh --mode both --only felix_round,felix_it
  ```

### Cross-baseline / shared

- felix (async_cifar10) may share fluxtune's round-1 cold-start gap (`_sim_recv_min`, no fallback for
  unseen ends) — unverified, out of scope here (`async_cifar10/PARITY.md` owns felix).
- felix 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Momentum (S1-S3) / server-optimizer retry — roadmap, not parity; resume after Phase-1 closes.
- Accuracy drop after 81% — known, deferred by operator (`fluxtune_contributions.md` §8).

**Tech debt.** Sim's in-flight bookkeeping is split across `_sim_pending_commit`, `_sim_inflight_expected`,
`_sim_buffer`, `_sim_committed`, `selected_ends`, `all_selected` — should be one authoritative per-end state
machine with slot/guard sets derived — §D-15 was exactly a lifetime bug inside that split, and the
residual `concurrency_cap` breach is confirmed as another (`fedbuff_round`'s lap-boundary double-cohort +
553 same-end dispatches, §D-20). Simplify once both roots close; scope behind `test_fwdllm_sim_grad_loop.py`'s
`TestCommitThenProcessFreesTheSlot` + `TestResidenceHoldsCommitterToCycleClose`, never bundled with a
correctness fix.

**P3, infra:** no automatic GPU skip-and-remap on a broken ordinal (manual `execution.gpu_ids` exclude works,
§G). `fedbuff_round`'s 07-27 sim run SIGABRT'd at exit after a clean, fully-flushed run (native atexit crash,
not a training-loop bug) — didn't recur same-session on felix_round; flag if seen again.

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

**Run-length budget (fwdllm) — state the min duration up front; never default to 3-4h.** Every run is
operator-launched, so pick the shortest length that exhibits the issue.

| validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | 5-10 min | a few hundred commits populate any per-commit field |
| one MECHANISM rung (`drain_wall_budget`, `selection_detail`, `eligibility`) | 45 min | the mechanism fires; per-commit dists stabilize |
| variance-cadence rungs (`V1`/`V2`/`V5`, iterations-per-data_id) | ~90 min | enough committed data_ids for the cadence dist to stabilize |
| throughput / `per_round_advance` (`K2`/`K3`) compounding residual | 3600s (~1h) | round-count-compounding drift needs the data_ids (fwdllm/fwdllm_plus validated here) |
| stochastic identity / participation (`cohort_sequence`, `S2`) | 3600s+ | index overlap must reach its independent-draw floor to read as identity-not-bias (§D-2) |
| convergence sign-off (`terminal_state`, `conv`, `conv_loss`) | full 2h+ | terminal-state + curve parity only |

Smoke (5-10 min) before any multi-hour run. One mechanism per run when a fix could perturb another baseline.

---

## §D  Durable lessons — fwdllm diagnostic patterns

> Transferable "see X → it means Y → discriminate by Z" patterns from fwdllm's own roots. Shared
> (non-fwdllm) patterns live in [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) "Durable lessons."
> Update in place; don't append near-duplicates.

**D-1. A shared-compute wall rung that fails with BYTE-IDENTICAL inputs is co-location contention, not sim
over-compute.** When `drain_wall_budget`/`agg_step_timing_breakdown` fail but the inputs are byte-identical
(agg_goal 10, grad_pool 2.07, cached_v 25.92) and sim's per-op floor equals real's typical in EVERY decile
(drain p10 35ms = real 37ms) with a fixed fraction (~19%) real-matched throughout, the extra wall is bursty
contention from ~100 co-located trainer threads, not more work. **Tell:** thread-local `cpu_duration_s`
tracks wall (on-CPU burn, not deschedule); a warm-up leak would concentrate early, this is uniform. Don't
tune sim compute — the real question is whether to charge the contention-inflated wall onto the vclock
(§F-20: never inject sim-host noise into the clock). **Scale caveat:** the old "never blame contention at
n=10" (refuted once, §E) held BELOW ~100 trainers; at n=100 contention IS a genuine root. Diagnose from
stored telemetry (input byte-sizes + per-decile drain floor) before instrumenting.

**D-2. A boundary-race cascade on a stochastic-async selector is core-IDENTITY, not a sim skew — gate
index-identity, enforce marginals.** When `cohort_sequence.composition` (index-paired overlap) fails but
EVERY marginal matches (participation S2 tvd 0.023, utility-dist KS 0.036, count 4.5%, v1/v2, speed_s
0/100), and index overlap decays to and PLATEAUS at the independent-draw floor (observed 0.239 = computed
0.237), real and sim are two independent samples of the SAME process. Mechanism: the marginal K-th cohort
slot is a sub-100ms arrival tie among interchangeable same-speed trainers — physical-FIFO (real) vs
lowest-sct (sim) — a coin-flip that cascades (the excluded trainer fronts the next cohort) and Oort
path-dependence amplifies. Index-paired identity is unattainable (0.8 target vs 0.237 ceiling). **Fix:**
gate `composition`/first-bin/`utility`-identity/`v1b`-MA to diagnostic for stochastic-async selectors; keep
`count`/`cum_mean_rel`/`speed_s` enforced; `S2`-by-speed-class owns the mix-bias catch. Confirm
identity-not-bias with A2c/K8 pass + speed-matched mode-specific cores BEFORE reclassifying — don't suppress
a real mix bias. (Mirrors PARITY.md's refl S2 core-identity lesson.) **Extension (07-28), RETRACTED 07-30:**
this entry claimed `felix_round`'s `v2_var_trajectory` mean shift traced to `calculate_var()`'s split-half
reading a different real/sim contributor SET. Refuted — both modes commit from the SAME frozen 30 for 89% of
the run and the gap is fully present there, and widening real's set moves `var` the WRONG way (§B). The
general rule the extension states is still sound (a set-membership-sensitive statistic can fail while its
count/marginal inputs pass — check that sensitivity before walking further down); felix is simply not an
instance of it. **Tell that would have caught it a day earlier:** before attributing a stat to a set
difference, verify the two sets actually differ over the WINDOW the stat is measured on, not just at one
sampled coordinate.

**D-3. Porting a baseline's SELECTOR does not port its real↔sim TIMING parity.** The selector class is
shared and travels with a straight import (`felix`'s `AsyncOortSelector`, `flame/selector/async_oort.py`,
→ `felix_round`/`felix_it` for free). But selection policy and sim-clock/timing modeling live in DIFFERENT
classes and only the first is shared: fwdllm's `flame/mode/horizontal/syncfl/fwdllm_aggregator.py::
TopAggregator` is a separate 4200-line subclass of the shared base — it **redefines `_sim_hold_busy_slots`
with different semantics** (same name, "fwdllm-class override only"), adds an fwdllm-only
`_sim_recv_min_grad` alongside (not replacing) the base's `_sim_recv_min`, and owns timing knobs with no
async_cifar10 consumer (`sim_model_agg_compute_time`, `sim_model_dispatch_queue`,
`charge_sim_vclock_overhead`, `_release_sim_slots_at_agg_goal`, `_flat_grad_norm`); the trainer side is a
fully separate `fwdllm_trainer.py::Trainer` (async_cifar10 uses the generic `trainer.py::Trainer`, no shared
trainer class). **Consequence:** a fix against the shared base auto-propagates to fwdllm ONLY for methods
fwdllm still inherits unmodified (`sim_sct_ordered_drain`, `VirtualClock`/`SimReorderBuffer`) — anything
overridden (even same-named, different body) or newly added is unverified from scratch, orthogonal to the
selector's parity record. **Before trusting a source example's parity result for a ported baseline, diff the
destination's aggregator/trainer against the shared base.** This is why `felix_round`/`felix_it` needed a
fresh pair despite `felix` being clean in async_cifar10, and why the round-cadence fedbuff fails (§B) are
NEW code paths, not a re-verify.

**D-4. Grade parity on the LOGICAL budget N, never a matched virtual-time window.** `matched_virtual_budget`
(V = min(sim_vclock, real_wall)) both MASKS and MIS-GRADES: it read fwdllm/fwdllm_plus at-parity under V
while HIDING a real 1.57× throughput gap (found only via raw databins/wall), AND let fluxtune run 1795 vs
1648 cycles inside the matched V. Normalizing along the axis under test (whether `vclock ≈ wall`, the
`sim_rate` question) is circular. **Fix the WORK (progress ≤ N data_ids), let TIME be the measured output**
— real algorithmic-time-to-N vs sim vclock-to-N. Design: PARITY.md §1.5; `matched_virtual_budget` is
DELETED, don't reintroduce (§E).

**D-5. Parity PASSES when both sides are equally wrong — a green rung is not a correctness claim.** A
bug whose effect is deterministic and mode-independent produces identical real and sim output, so every
comparison rung passes. `reselect_each_iteration: false` froze `fedbuff_round`'s participant set to the
first 10 of 100 joiners — in BOTH modes, the same ids 370-379 — and `cohort_sequence` passed;
`felix_round` failed 17 rungs only because its freeze happened to be asymmetric (30 vs 40, a wall-clock
race). **Tell:** a rung passes while an absolute invariant is violated — here, selection telemetry
stopping at t+19s of a 3738s run, and `distinct committers ≪ n`. **Discriminate by:** checking absolute
sanity (participation coverage, event-stream liveness, does the progress counter actually advance)
independently of the real-vs-sim diff, on the CLEAN baselines too. Parity is a differential test; it is
blind to common-mode faults by construction. Corollary: a clean §A row is evidence of *agreement*, not
of correctness — never treat one as a regression baseline without an absolute check alongside.

**D-6. A selector's parity record is a property of the SELECTOR+AGGREGATOR pair, not the selector.** The
version_key re-pick guard (§F-23) can be supplied by either side, and the two dispatch paths choose
differently: fwdllm's SYNC gate caches selection per `version_key` in the aggregator, so `RandomSelector`
needs no guard and reads clean; the ASYNC gate passes `trainer_version_keys` down and expects the
SELECTOR to filter, which `async_oort` does and `fedbuff` does not (34.4% vs 3.5% same-version re-picks).
**Consequence:** "selector X is verified" is unsound as a porting reference — verify the pair. Before
citing a clean baseline as the model to copy, check which side of ITS pair owns each guard, then confirm
the destination pair still has an owner for that guard. (Extends §D-3 from timing to selection policy.)

| baseline | dispatch path | cadence | version_key guard from |
|---|---|---|---|
| `fwdllm`, `fwdllm_it_*` | sync gate | mixed | aggregator (`_reselect_true_cache_key`) |
| `felix_it` | async gate | iteration | selector (`async_oort`) |
| `fedbuff_it_unaware/oracular` | async gate | iteration | selector (shared `AsyncSelectorBase`, post-R-A) |
| `fedbuff_round`, `felix_round` | async gate | round | selector, pinned-cohort trimmed to `c` (post-R-B/R-C) |

**D-7. A statistic computed over rate-scaled samples measures the rate, not the samples.** FedBuff's
staleness-decayed `rate` correctly down-weights a stale contribution INTO the model update, but scaling the
same sample toward zero before a variance/noise gate just makes the pool look less noisy — a spurious
claim, not a real one. Neither FedBuff's nor Felix's papers define a variance gate at all (pure FwdLLM
overlay), so there's no baseline-fidelity reason for rate to touch it; fluxtune's own `inverse_var` already
computes its reliability score from the unscaled sample. **Tell:** real and sim run the identical formula
but diverge 10x on an EXACT/DIST rung — check whether the two sides' INPUT to that shared formula differs
before suspecting the formula. Here real's genuinely-larger staleness (round-cadence's carried surplus,
§F-17) drove the same rate formula harder than sim's, whose own convergence artifact (this same bug) kept
`model_version` — and its own staleness — from growing within a short run. One bug, two symptoms.

**D-8. A cohort-reuse fast path that returns a cached selection directly bypasses whatever guard lives
inside the selector call it skips.** Round-cadence's `_select_ends_for_async_respecting_reselect_gate`
returns `self._round_selected_ends` once the cohort is full, without calling `channel.ends()`/`select()` —
the only place `_agg_pending_commit_ref` (busy-exclusion) is enforced. Real never notices because a
physically-busy trainer process can't act on a redundant dispatch regardless; sim's trainers run genuinely
fast (§F-1: real GPU compute, not slept-out to the modeled duration), so they're free again before the
guard would have mattered — collapsing intended per-trainer speed heterogeneity into lockstep. **Tell:**
every entity in a cache/reuse fast path shows IDENTICAL behavior (same update count, same rate) where the
un-cached path shows heterogeneity — the fast path dropped a per-entity guard the slow path had.
**Discriminate by:** diffing the fast path's return against what the bypassed call would have filtered, not
by re-deriving the guard's logic from scratch.

**D-9. A parity DIVERGENCE names two sides that disagree, not which one is wrong — check each side's own
absolute signal before picking which to fix.** Real `staleness` unbounded vs sim capped at 1 (U3) invites
the framing "why doesn't sim show what real shows" — but that presumes real is the reference. Before
touching either side, look for an independent, mode-specific self-consistency check on the mechanism in
question (an existing invariant, a diagnostic warning log, a duck-typed guard) that already tells you
something true-or-false on its OWN terms, with no cross-mode diff involved. Here: sim's `_sim_pending_commit`
busy-set (bound to `_agg_pending_commit_ref`) already excludes a dispatched-but-not-yet-returned end from
redispatch, and a paired tripwire log (`[SIM_R1_DISPATCH]`) fires zero times across the run — proof sim
never floods a busy end, established before any real/sim comparison. Real's equivalent binding
(`_per_agg_trainer_list`) only covered returned-but-uncommitted ends — no construct existed for
still-training-never-returned — so real, not sim, had the gap. **Tell:** one side's bound looks
suspiciously clean next to the other's growth. **Discriminate by:** finding or building the SAME-side
absolute check (never a real/sim diff) before deciding which side's behavior to change (→ preamble).

**D-10. `reselect_cadence` narrows or widens the CANDIDATE POOL for a freed dispatch slot, not whether a
committed trainer personally waits.** Easy to over-state (corrected mid-session, 2026-07-27): a just-
committed trainer is excluded from re-selection under EITHER cadence until its own `agg_goal`-sized batch's
`version_key` advances — `async_oort.py`'s `select()` filters any candidate whose `trainer_version_keys[end]`
already equals the current `agg_version_key` (the R-A/D-6 re-pick guard), and that gate is cadence-agnostic.
What `reselect_cadence` actually changes is which POOL fills a slot that just freed up
(`extra = concurrency - len(selected_ends) - cooling_count`, async_oort.py): **round** cadence
(`fedbuff_round`/`felix_round`) restricts it to the pinned cohort (up to `c`, e.g. 10) — as more of that
small set contributes to the current version_key, fewer remain eligible to fill a newly-freed slot, so a
slot can idle even though other candidates exist system-wide. **iteration** cadence (`fluxtune` — never
overrides `reselect_each_iteration` in `baselines.yaml`, so `_resolve_reselect_cadence` defaults it) re-runs
`channel.ends()` against the FULL trainer pool every tick, so a freed slot has a much larger not-yet-
contributed candidate set to draw from. **Superseded 07-28:** read against fresh `fedbuff_round`/
`felix_round` pairs, `post_close_overhead_wall_s` is NOT near-zero (real 0.46-0.52s vs sim 0.03-0.04s,
~13-15x) — the pool-size theory doesn't explain the observed gap on its own; §G's `redispatch_turnaround`
per-cycle-charge root-cause supersedes this as the throughput driver. Kept here as the pool-size mechanism
may still matter for `slot_starvation`-style idling, just not this gap.

**D-11. A per-ROUND wall-time residual can be a per-CYCLE cost compounded, not a one-time round cost —
check the compounding count before sizing the mechanism.** `fedbuff_round`/`felix_round`'s
`per_round_advance` gap (~15-21%) looked, from means alone, too large for `redispatch_decomp`'s measured
`post_close_overhead_wall_s` gap (~0.46-0.52s) to explain — a naive one-per-round accounting left ~90% of
the residual unattributed. Reconciled by checking how many `agg_round` (variance-check) CYCLES occur per
progress unit (`_per_progress_last_event`'s round key, `data_id` here): ~9.8-9.9 cycles/data_id (event
count ÷ data_id count, not assumed from `var_threshold`). Every cycle — not just the terminal
model-version-bumping one — showed the same ~25-30% real-vs-sim wall gap (2.96s/2.79s vs 2.29s/2.20s
vclock-basis, both baselines), and compounding that per-cycle gap ~10x reproduces the full per-round
residual almost exactly (felix_round: 114-span mean 29.13s vs the checker's own 29.35s). **Tell:** a
telemetry field's measured gap (small) doesn't obviously scale to a checker rung's reported gap (larger) by
any simple per-round multiplication. **Discriminate by:** count actual cycles-per-progress-unit from raw
event counts, then check whether the SAME per-cycle telemetry (if it existed) would apply to the
uninstrumented cycles too — this is what led to widening `redispatch_decomp`'s gate to `weights`-payload
sends and landing the `redispatch_turnaround` charge fix (§G).

**D-12. A telemetry span measured `now - shared_batch_start` is CUMULATIVE across every entity in the
batch, not one entity's marginal cost — pooling it flat overcharges by the batch size.**
`redispatch_decomp`'s `post_close_overhead_wall_s` = `now - round-close wall ts`, sampled once per trainer
in a serial `for end in ends:` loop — the k-th trainer's reading already includes trainers 1..k-1's wall
time ahead of it. `profile_sim_charges.py` pooled these raw readings into one flat mean and charged that
mean to EVERY trainer in the batch — a batch of K got charged ≈K× its true cost (confirmed 7.3-8.3x on
fedbuff_round/felix_round's 5400s pair, ≈1900-2100s of pure excess vclock advance, 39-42% of the run).
**Tell:** a per-event field measured inside a loop over a shared pre-loop reference point — its mean isn't
any one entity's marginal cost. **Fix:** group by batch key, sort by ts, pool the FIRST DIFFERENCE between
consecutive positions. Standalone per-round spans measured once (`drain_tail`/`fedavg`) don't have this bias.

**D-13. A run stopped by a wall-clock deadline can emit ONE event for a progress unit it never finished —
treating that event's presence as "reached" manufactures a spurious terminal divergence, a CHECKER bug
that reads as a sim rate bug.** fwdllm's `throughput`/`terminal_state`/`total_commits` failed at 5400s
(5.5-6.2% vs 5% tol) with no code-side mechanism to explain it. `wall_disparity` (DIAG) showed a smooth
~3.5s/data_id trend, then a 3x jump exactly at the checker's chosen terminal N — real's last `agg_round`
event there had `var_good_enough: False` (one failed variance check, ~22s, before `max_runtime_s` hit; a
normal bin needs ~9-10), while sim had fully committed the same data_id (~205s vclock). Excluding that
point (N-1) dropped the residual 5.5%→2.15%, a clean pass. **Tell:** a DIAG per-unit series is smooth
except one outlier at the exact point a gating rung calls its terminal N. **Fix:** `_per_progress_last_event`
(`checks.py`) drops a `data_id`-axis key unless verified (`_DATA_ID_COMMIT_FIELD`, `var_good_enough`),
falling back to unfiltered if a side has zero verified keys. `_matched_logical_budget` reuses this same
helper instead of its own max (§F-26). Not fwdllm-specific: any `data_id`-axis baseline hitting its
wall-clock deadline mid-cycle is exposed the same way (`fedbuff_round`/`felix_round`/`fluxtune` included).
`round`-axis baselines (async_cifar10) are untouched — a `round` only increments after a genuine close, so
presence there already proves completion; no analogous truncation risk, no filter applied.

**D-14. Fixing a proven overcharge can overcorrect past zero — re-verify the SIGN of the residual, not just
that the old bug is gone.** `redispatch_turnaround.weights`'s cumulative-batch-pooling overcharge (D-12, 7.3-
8.3x) was real and the marginal-diff fix is mechanically correct — but applying the smaller value to a fresh
sim run flipped `fedbuff_round`/`felix_round`'s `throughput` from 8.2-8.6% sim-SLOWER to 10.0-14.0% sim-
FASTER, a worse relative gap in the other direction, cascading into new `v1_iter_per_data_id`/`cohort_sequence`
fails (§D-11 pattern: sim's cheaper rounds buy it more iterations per data_id within the same budget). **Tell:**
a "before: X% one direction, after: Y% the OTHER direction" pattern where |Y| > |X| means the fix's magnitude
was probably right in kind but the specific value moved past the true target, OR (as verified here) the fixed
category was never the dominant term — check both. **Discriminate by:** sum the charged category's real
`vclock_charge` total against an independent real-side reconstruction of its true total (not the per-event
mean, the actual aggregate) at the SAME progress granularity (here: per `data_id`, matching `redispatch_
turnaround`'s real total of 276-296s against sim's charged 325-336s — sim charges *more*, not less, ruling
the category out as the residual's source). Don't re-tune a charge a second time on the strength of a
downstream rung alone (§F-13) — confirm via the category's own totals first. **Resolved by §D-15: the
residual was not a charge at all.** The elimination this entry performed was correct and load-bearing —
what it could not do was name the mechanism, because no charge could. **Second instance, 07-29/30 — and the
limit of this entry:** the §D-15 policy fix flipped the sign AGAIN on `felix_round`/`fluxtune` (10.0/7.9%
fast → 15.8/13.8% slow) while landing `fedbuff_round` at 2.2%, so the tell fires for non-charge fixes too
and a fix validated on one baseline is not validated on its siblings — grade the sign per baseline, never on
the batch. But the tell's OWN reading ("the magnitude moved past the true target") was wrong here: the fix
landed exactly right and UNMASKED two pre-existing roots of similar size (§D-16). A sign flip means
re-decompose, not re-tune — a residual that merely resembles the old one in magnitude is not the old one.

**D-15. When sim is FASTER than real per unit of progress and no charge category explains it, the gap is a
CONCURRENCY/scheduling policy divergence, not an unpriced cost — measure per-trainer IDLE, not the server's
wall.** After D-14 ruled `redispatch_turnaround` out, the round-cadence residual (sim 15.7-28.3% faster)
resolved to the identity `commits/s = c / (busy + idle)` per contribution, verified to 3 decimals in both
modes: `busy` matched (6.3-11.3s), `idle` did not — real 1.6-17.7s vs sim 0.2-14.7s. Substituting real's
idle into sim's cycle collapsed the residual to 1.1-4.8%. The mechanism: sim released a trainer's slot at
its own COMMIT (`_sim_pending_commit.discard` in `_sim_recv_min_grad`) instead of at the agg-goal boundary,
so it re-dispatched mid-cycle under the SAME `version_key` the trainer had just contributed to — 78-90% of
sim dispatches, 0% of real's — and peak concurrency ran to 38-46 against `c`=30. Both configs declare
`inflight_residence: true`, so this was sim violating its own contract: a single-side failure decidable
with no real/sim diff (§D-9). **Tell:** matched per-contribution busy time + a large idle deficit, with the
gap's own decomposition (`redispatch_decomp`) showing real's gap is 76-98% `peer_wait_wall_s` (the barrier,
already modeled) and only 2-24% `post_close_overhead_wall_s` (what the charge prices). A charge cannot buy
back time a scheduling policy gave away.

**NOT excess dispatch, and nothing was wasted — the sole defect is WHEN the dispatch is stamped.** Measured
on the same pair: dispatches/cycle **10.02 real vs 10.03 sim**, contributions/cycle 10.00 both,
`trainer_round` forward passes per contribution **1.002 vs 1.003**. Every sim dispatch produced exactly one
forward pass and one accepted grad, same as real; sim ran 2502 cycles where real ran 1857, at an identical
per-cycle message cost. The trainer's own same-key self-guard
(`fwdllm_trainer.py:309`, `abort_training`) never fires on these sends because it compares the incoming key
against what the trainer last SENT — and with a 6.7s round trip against a 2.8s cycle its last-sent key is
~2 cycles stale, so the labels differ and it computes. The illegitimacy is the AGGREGATOR asserting a
verdict for a `version_key` it hasn't evaluated, not the trainer receiving duplicate work. **Mechanism of
the throughput gain:** the mid-cycle dispatch is stamped at a vclock that hasn't yet advanced past the
cycle's close, so `sct = sim_send_ts + max(gpu, D) + leg` lands ~1.0s early (82.6% of sim dispatches
past-date the max committed sct of the cycle they land in; amortized 0.83s each). Per contribution
`c/(busy+idle)`: sim 30/(6.27+0.20) = 4.64/s → 2.16 s/cycle → 2502 cycles in 5400s; real 30/(6.67+1.63) =
3.61/s → 2.77 s/cycle. Restoring the barrier restores the idle, and with it the cycle cost. **Fix:** defer
the release to
`_release_sim_slots_at_agg_goal`, matching real's `inflight_residence` → `cleanup_recvd_ends()` path, plus
two per-dispatch INV tripwires (`concurrency_cap`, `retask_before_close`) so the next run grades itself.
**Validated 07-29/30:** `retask_before_close` 78-90%→0.0% on all three, sim s/round +20-29%, and the
`busy+idle` identity itself now MATCHES on all three (felix 2.93 vs 2.87 s/cycle, fluxtune 9.59 vs 9.73) —
the mechanism and its magnitude are both confirmed. The felix/fluxtune residual that remained is NOT this
term: it factors out to `cycles/bin` and `s/cycle` respectively (§D-16/17/18). `concurrency_cap` still
peaks 35/30 and 31/30 on two of three — an independent INV, open in §B.

**D-15a. Three measurement traps, all hit before the raw trace settled it.** (1)
`contributor_intervals.dispatch_version_key` is read at cycle-CLOSE time, so it holds the end's NEXT
dispatch, not the one that produced that contribution — never use it for "what key was this sent under".
(2) Anchoring a was-it-already-served test on the cycle's `agg_round.ts` misses a mid-cycle send entirely
(the send PRECEDES the close) — anchor on `contributor_intervals.processing_wall_ts`, the wall instant the
grad was accepted. (3) Contribution-level and dispatch-level invariant 1 are different questions:
contribution level is clean in BOTH modes (0 duplicates) *because the round-trip outruns the cycle*, so a
green `r1_inflight_overlap` does not clear the dispatch path. Ground truth for "what was sent, to whom,
under what key" is the `comm` event (`direction=agg_to_trainer`, `model_version`,
`iteration_per_data_id`, `payload_kind`) interleaved with `processing_wall_ts`; everything else is derived.

**D-16. Factor a per-round time residual into `(s/cycle) × (cycles/round)` BEFORE naming a mechanism — the
two factors have disjoint root sets, and a shared symptom is not a shared root.** `felix_round` and
`fluxtune` both landed at a 13-16% sim-SLOW `per_round_advance` after the §D-15 fix and were tracked as one
"overshoot" for a day. They share nothing: felix is `s/cycle` 1.023 / `cycles/bin` **1.141** (a variance-
cadence root), fluxtune is `s/cycle` **1.154** / `cycles/bin` 0.994 (a charge root), `fedbuff_round` is
0.985/1.037 (fine). Compute both factors from raw `agg_round` counts on each side's OWN clock
(`wall_elapsed_s` real, `vclock_now` sim), normalizing by COMPLETED bins (`var_good_enough == True`), never
by distinct `data_id` (§D-19). **Tell:** two baselines fail the same rung by a similar percentage and the
per-cycle telemetry that explains one doesn't move on the other. **Corollary:** `overhead_residual`'s
`implied_per_commit_overhead_s` is an EMERGENT aggregate — it divides the whole residual by `agg_goal` and
therefore names no mechanism at all. Sizing a fix off it (as "decompose the surplus idle" was about to)
attributes the entire gap to whichever term you happened to be looking at.

**D-17. When a rung grades EVENTS but its window is the whole run, it measures which side got further —
convert the window to matched WORK before believing any divergence.** `reselect_cadence: "round"` pins the
cohort for a whole round = all ~150 data bins (`_cohort_cache_key`, `fwdllm_aggregator.py:3367`;
`self._round` bumps at `data_id == total_data_bins`, `:2386`) — correct and intended, and a 5400s run covers
~1 rotation, so each side emits ~1 cohort draw. `selection_detail` nonetheless reported "real 2.73 vs sim
30.0" (felix) and "real 30.0 vs sim 2.0" (fedbuff): opposite-looking failures, ONE cause. Whichever side
happened to lap emitted a burst of backfill re-draws AFTER passing the matched budget, and the rung counted
them. Inside matched work both sides are identical on both baselines (1 draw of 30 each). **Tell:** a rung's
two sides differ CATEGORICALLY (30 vs 2) while every continuous measure of the same subsystem agrees —
categorical output from a near-tied input; and the direction flips between sibling baselines with no policy
difference. **Discriminate by:** compare the divergent events' PROGRESS KEY against the rung's own
`matched_logical_budget_n` — not their timestamp against a vclock/wall figure. Those are different units and
mixing them is how this was mis-read once already: fedbuff's 31 extra draws sit at progress (2,149) against
N=(1,148), i.e. out-of-budget, even though they occur at wall t+1618s of a 1673s leg. **General rule:** an
event-count rung is only meaningful over matched work, because event counts scale with progress and progress
divergence is `throughput`'s job to grade, not the selector's.

**D-18. Charging sim's own contention-inflated wall onto the vclock is a THROUGHPUT root, not a DIAG
curiosity — price it from the ledger before filing it as §D-1 noise.** `fluxtune` sim charges `drain_tail`
at `charge_source: "live"`: 0.585 s/cycle against real's measured 0.330 (1.77×), plus `fedavg` 0.078 vs
0.059. Total charged 0.787 s/cycle vs real's actual 0.389 — an excess of 0.398 against an observed s/cycle
gap of 0.590, i.e. **67% of the baseline's whole throughput residual**, from a term filed as informational.
§D-1 correctly says don't tune sim's compute down; it does NOT license folding the inflated wall onto the
clock (§F-1, §F-20). **Tell:** a `drain_wall_budget`-family rung fails DIST while a gating throughput rung
fails by a similar magnitude on the same baseline. **Discriminate by:** sum the `vclock_charge` ledger by
`label` × `charge_source`, divide by cycle count, and compare per-cycle charged-sim against per-cycle
measured-real — the ratio is the answer, the KS stat is not.

**D-20. An INV measured as a SET SIZE at event instants is not the invariant — measure the invariant's own
quantity, on its own axis, and it usually turns out both legs already have the telemetry.** `concurrency_cap`
("keep `c` trainers training at any instant") was graded off `len(_sim_pending_commit)` sampled at dispatch
instants on the WALL clock. That reading was wrong in both directions at once: it counted ends that had
committed in virtual time but whose slot the agg-goal boundary deliberately still held (§D-15), inventing
`fluxtune`'s 31/30 — 1218 warnings, one per cycle, every one false; and by sampling only at dispatch instants
it under-reported `fedbuff_round`, whose true peak is 35 distinct ends with 553 same-end concurrent
dispatches. **Fix:** sweep each contribution's `[dispatch_ts, commit_ts]` interval from `contributor_intervals`
and take peak DISTINCT ends — vclock in sim, wall in real, each mode on the axis `c` is defined on.
**Two dividends worth generalizing.** (1) REAL became gradeable on every baseline with no fresh run: the
"needs a fresh real leg" gap was never a telemetry gap, the same fact was already recorded in a different
event. Before ordering a run to instrument an invariant, check whether an existing per-entity span already
implies it. (2) Counting intervals rather than distinct entities inflated the peak to 61; an entity holding
two concurrent dispatches is ONE busy trainer occupying one slot — and that second dispatch is a *different*
violation (§F-25), so count it separately instead of letting it corrupt the cap number. **Tell:** an INV
fires at a suspiciously regular rate (exactly once per cycle) or with a magnitude that doesn't match any
plausible mechanism — suspect the measurement before the mechanism.

**D-19. Normalizing a per-bin statistic by distinct `data_id` under-counts any side that wrapped, and can
turn a 14% gap into a 4% PASS.** `v1_iter_per_data_id` divides cycles by distinct `data_id` — 150 real /
147 sim on `felix_round`'s 5400s pair — but real completed **163** bins, having wrapped past
`total_data_bins` and re-run 13 of them. Real's iters/bin reads 8.7% high, the rung passes at 4.4% against
a 15% tol, and the true per-completed-bin gap (11.02 vs 12.54, **+13.8%**) — which exactly equals the
failing `throughput` residual — is hidden. `g2_grad_pool_size` measures the same quantity correctly (11.006
vs 12.537, FAIL) and was being filed as "marginal, downstream" while it was the root signal. Same family as
§D-13, same violated rule (§F-2: never key on `data_id`, it wraps). **Tell:** a V-family rung passes while
`throughput`/`per_round_advance` fail by a wholly unexplained margin on the same pair. **Discriminate by:**
count `var_good_enough == True` events per side and compare against distinct `data_id`; if they differ, the
normalizer is wrong. Real's repeat bins here were *more* expensive (13.06 vs 10.69 cycles), so the wrap gave
real no artificial speed advantage — only the checker's normalizer did. **Fixed 07-30** by keying
`_iters_per_data_id` on `(round, cycle_data_id)` + counting only COMPLETED visits + truncating to the matched
budget. Attribution, so the next reader doesn't over-credit the window: raw-key/no-window 11.980 (the false
4.4% pass) → per-visit key 10.926 (12.8%, already fails) → +budget 10.694 (14.7%). **The key was the bug;
the window is a 2% refinement.**

---

## §E  Dead ends — do NOT retry

> Falsified hypotheses, one line each, append-only. A dead end never un-dies; re-listing one wastes a
> session. Landed-but-inert cleanups belong in §G, not here.

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
  clocks (the axis `sim_rate` tests), masks throughput + fails to grade (§D-4 / PARITY.md §1.5).
- **"surplus idle" as the post-§D-15 residual (felix/fluxtune)** — REFUTED 07-30: the `c/(busy+idle)`
  identity is MATCHED on all three post-fix legs (felix 2.93 vs 2.87 s/cycle, fluxtune 9.59 vs 9.73). The
  residual is `cycles/bin` (felix) and `s/cycle` charges (fluxtune) — §D-16. Don't re-open the idle term.
- **`felix_round`/`fluxtune` sharing one "overshoot root"** — REFUTED 07-30: disjoint factors, §D-16.
- **the round-cadence cohort pin as a defect ("never rotates / 70 of 100 never train")** — NOT a defect,
  operator ruling 07-30: `reselect_cadence: round` pins for a whole round by design and `rounds: 50` would
  rotate it 50×; a 5400s run just covers ~1 round. Don't re-key it on `_model_version` (§D-17).
- **cohort COMPOSITION as `felix_round`'s `v2_var_trajectory` driver** — REFUTED 07-30: both modes commit
  from the same frozen 30 for 89% of the run with the gap fully present; widening real's set moves `var`
  the wrong way (§D-2 retraction). The trajectory diverges, the contributor set doesn't.
- **`aggregate_grad_pool()` summation order as the `v2_var_trajectory` driver** — REFUTED: it's an
  element-wise SUM over ≤`max_iterations_per_data_id` items (float-reorder effect ~1e-6 relative, not the
  observed 3%) feeding the outgoing `GRAD_POOL` payload, not the variance gate. The gate is
  `calculate_var()`'s split-half over `grad_for_var_check_list` — see §D-2 extension.

---

## §F  Locked invariants (from async_cifar10, carried over)

> Always-true / always-do rules. Numbers are cited across this doc — keep them stable, don't renumber.
> Diagnostic *patterns* (see X → means Y) live in §D, not here.

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

### §F.1 Version & commit invariants (confirmed real+sim in code + real logs, 2026-07-21)

21. **`model_version` bumps once per COMPLETED data-bin** (variance PASS, `+= 1` at the data_id advance) —
    constant across one data-bin's iterations. `iteration_per_data_id` bumps on variance-FAIL retry, resets
    on data-bin advance. `version_key = (model_version, iteration_per_data_id)` changes every iteration —
    the sole step identity (§F-14).
22. **Commit == the update used for aggregation, at that instant — no lag.** Real: on ordered arrival. Sim:
    when vclock reaches the update's `sct` (buffer-unlock IS the commit). Never commit on a later event.
23. **Commit frees the compute slot immediately, but a version_key re-pick guard keeps the trainer
    un-pickable for the SAME `(model_version, iteration)`** until the version_key advances. Sim's slot-hold
    (`_sim_pending_commit`) must clear at commit and never re-add after, or re-dispatch starves across
    variance-retry iterations.
24. **Within a data-bin the global weights are constant; a re-picked trainer gets a RETRY, not a re-send.**
    Full WEIGHTS go out only for a `model_version` not yet received this data-bin
    (`_weights_sent_this_cycle`, cleared on the bump); a same-`model_version` re-pick gets VAR=bad, never a
    redundant weight re-send.
25. **One instruction per version_key: never dispatch to a trainer with an unresolved outstanding message
    for the CURRENT `version_key`.** Busy = silence, not a second message, until it returns or the
    version_key advances. Enforced by `_already_served_current_instruction`/`_mark_instruction_served`
    (end_id → last-served version_key). SYNC distribute always had this guard; ASYNC didn't (found
    2026-07-26: `fedbuff_round`'s `r1_inflight_overlap` was ~90% `VAR=bad` re-sends to an already-busy
    trainer — §G). Any new distribute call site must call both.
26. **Reuse the existing construct; don't duplicate per baseline.** New per-trainer/version state almost
    certainly needs an EXISTING mechanism (`_end_served_version_key`, `_keyed_topk`/`_keyed_draw`,
    `AsyncSelectorBase`), not a new one parallel to it — duplicate logic is duplicate bug surface. Tests
    too: extend a contract suite (`test_async_selector_base.py`, `test_selector_contract.py`,
    `test_selection_determinism.py`) before writing a bespoke one. Tell: about to add a variable/method/
    test whose name rhymes with an existing one (`_keyed_weighted_topk` next to `_keyed_topk`) — check
    whether the existing one should just take a parameter instead (done same session: both now share one
    `_keyed_draw` primitive, §G).

### §F.2 Porting a SELECTOR ≠ porting TIMING parity → §D-3

Moved to §D-3 (it's a diagnostic pattern, not an invariant). Kept here as a stub because prior sessions cite
"§F.2" — the class-hierarchy detail and the "diff destination aggregator/trainer against the shared base"
rule now live in §D-3.

---

## §G  Landed fixes — recent, load-bearing for current work only. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, ≤30 words, immediately.** The instant a rung flips or a hypothesis resolves, write
> ONE line (mechanism + outcome) and delete it from §A/§B in the same edit.

- **07-30: `drain_tail`/`fedavg` moved from LIVE-span to REAL-PROFILED vclock charge (§D-18)** — sim's own
  span is contention-inflated 1.77x at n=100; charging it violated §F-1/§F-20. 2 tests. Awaits a sim leg.
- **07-30: round-boundary BACKFILL landed** — `_cap_dispatch_to_concurrency()` + `_exclude_pending_commit`
  now on the cohort accumulate branch, not just the reuse branch. Fixes `fedbuff_round` sim's 35-vs-c=30 and
  553 same-end concurrent dispatches. 5 tests. Awaits a sim leg to validate.
- **07-30: `concurrency_cap` re-based onto peak DISTINCT in-flight ends from `contributor_intervals`
  (§D-20)** — grades REAL on every baseline with no fresh run; `fluxtune`'s 31/30 was a phantom (1218 false
  positives). Same-end concurrent dispatches now graded separately (§F-25). 8 tests.
- **07-30: `_outstanding_dispatch_count()` cap arithmetic subtracts `_sim_committed`** — a committed end
  whose slot the boundary still holds is not in flight; the pending set is unchanged.
- **07-30: `selection_detail` windowed to matched WORK + event COUNT now graded** — was pooling each side's
  whole run, so it graded which side lapped. Green on all three; full-run counts still reported. 5 tests.
- **07-30: `v1_iter_per_data_id`/`v1b` normalizer FIXED (§D-19)** — keyed per data-bin VISIT
  `(round, cycle_data_id)`, completed visits only, truncated to the matched budget. Ends felix_round's false
  4.4% PASS (true 14.7%); fedbuff `v1b` green. 4 tests. 1092 `pytest lib/python/tests` pass.
- **07-30: §D-15 fully validated, "overshoot" reclassified** — `busy+idle` matched on all three; the
  felix/fluxtune residual is two unrelated roots (§D-16/17/18), not an over-corrected barrier.
- **07-30: `selection_detail`'s "inverted granularity" closed as a non-finding** — lap-boundary crossing,
  not two selection policies. felix = checker windowing (21/22 events out-of-budget); fedbuff = duration-
  gated coin flip. Cohort pinning is CORRECT (operator ruling, §D-17).
- **07-30: `felix_round`'s "why does composition diverge when selection is deterministic" ANSWERED** —
  selection is deterministic; one side crossed the round boundary inside the graded window, the other
  didn't (§D-17).
- **07-29 pm: `fedbuff_round`'s throughput family CLOSED by the §D-15 boundary-release fix** — validation
  pair `_034223`/`_220025`: `throughput`/`total_commits`/`terminal_state`/`per_round_advance`/
  `overhead_residual`/`cohort_sequence` all flip green, residual 14.0%→2.2%.
- **07-29 pm: mid-cycle redispatch ELIMINATED in sim, all three round-cadence baselines** (§D-15) —
  `retask_before_close` 78-90%→0.0%. Root confirmed; magnitude correct on all three (07-30 line above).
- **07-29: `redispatch_turnaround.weights` cumulative-batch overcharge FIXED** (§D-12, 0.488→0.0598s,
  7.3-8.3x) and RULED OUT as the throughput residual's source (§D-14 totals check). Value is now settled —
  do not re-tune (§F-13).
- **07-29 pm: `fluxtune` sim yaml was missing `sim_charge_profile_path`** — every `redispatch_turnaround`
  row read `charge_source: "none"`, ~180s uncharged. Added; `convergence`/`selector_score`/`utility` green.
- **07-29: fwdllm's `throughput`/`terminal_state`/`total_commits` 5400s fails ROOT-CAUSED as a CHECKER bug,
  FIXED + VALIDATED** (§D-13). `_matched_logical_budget`/`_per_progress_last_event` treated an uncommitted
  trailing `data_id` (real hit `max_runtime_s` one variance-check attempt into it) as reached progress,
  comparing real's ~22s partial cycle against sim's ~205s completed one. Fixed to require `var_good_enough
  == True` evidence per `data_id`-axis key. Re-run: 56/6/22 → 59/3/22, only the pre-existing D-1 contention
  group remains. 1392 `pytest lib/python/tests` pass. CLOSED.
- **07-28 evening: `redispatch_decomp`/`vclock_charge` dark-data gap closed** — `analyze_run.py` gained
  `redispatch_decomp_plots`/`vclock_charge_plots` (CDFs by mode/payload_kind + a `post_close_overhead` mean
  bar + a "real span vs sim charged" uncharged-gap bar), wired into `_PLOT_GROUPS`. Verified against live
  `fedbuff_round` real+sim telemetry (1.9M records) + 5 new pytest cases
  (`tests/analysis/test_redispatch_vclock_charge_plots.py`). CLOSED.
- **07-28 evening: `redispatch_turnaround.weights` charge fix VALIDATED live on `fedbuff_round`/
  `felix_round`, 3600s.** `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/
  `terminal_state` all flip fail→pass, residual 1.5-4.8% (well under 10% tol) — the round-cadence
  dispatch-path root closed with no meaningful overshoot. CLOSED.
- **07-28: general profiled-charge mechanism landed (FWDLLM_DESIGN.md §P).** `sim_charge_registry.yaml` +
  `get_profiled_charge_s()` loader + `charge_sim_vclock_overhead(..., profiled_s=...)` — a registry entry's
  `charge: true` flips a category on, independent of `sim_model_agg_compute_time`. `redispatch_turnaround.weights`
  seeded `charge: true, mean_s: 0.4365` (pooled n=3264) from a `run_20260728_151722`/`_151831` real pair;
  `.var_bad` left `charge: false` (measured negligible, ~2x not ~13x). `vclock_charge` ledger gained a
  `charge_source` field (`live`/`profiled`/`none`). 8 new tests.
- **07-28: `redispatch_decomp` widened to cover `VAR=bad` dispatches (not just fresh `weights` sends) +
  new `vclock_charge` ledger event added.** `_pk in ("weights", "var_bad")` gate + `payload_kind` field
  (`fwdllm_aggregator.py:4135`); `charge_sim_vclock_overhead` (the ONE shared fold-onto-vclock function, used
  by every fwdllm-family baseline) now emits `vclock_charge` on every call in both modes, plus a
  `charge: bool` param so a candidate category can be measured before being charged. Root-cause
  reconciliation (§D-11) found the round-cadence throughput gap is a per-cycle cost compounding ~10x — the
  old telemetry only measured the 1-of-10 terminal cycle. 4 new tests, 573 `-k "fwdllm or telemetry or
  parity"` pass.
- **07-28: `felix_round` sim + `fluxtune` (both sides) missing-run episode resolved by relaunch, not a code
  fix.** Both pairs re-ran clean same-day (`run_20260728_102750`/`_113005`, `_102831`/`_113046`); the
  `DIRTY_ABORT` hypothesis was never confirmed on the run node and is now moot for this batch.
- **07-28: `selection_detail` CONFIRMED at production scale** — `_exclude_pending_commit` (07-27) passes on
  both `fedbuff_round`/`felix_round` at n=100/c=30/3600s, not just n15. CLOSED.
- **07-28: `fedbuff_it_unaware` 6→1 fails confirmed duration-gated, not a bug** — running the same config
  3600s instead of 1800s alone resolved 5 of 6 fails; no code change needed.
- **07-28: `run_sequential.sh`'s `fedbuff_round`/`felix_round` yaml mapping briefly mis-set to the n15
  debug-scale files, then reverted.** The "n10"-named files are actually production n=100 scale (misleading
  filename only); n15 is a deliberate reduced-scale repro, not a substitute. `ALL_RUNS` now correct.
- **07-28: `fedbuff_it_oracular` 12→4 fails confirmed at scale** — `get_curr_unavail_trainers`/
  `get_curr_task_ineligible_trainers` DEBUG-gating fix (07-26, below) held at n=100/3600s; remaining 4
  fails tracked in §B (profile-own-charge group).
- **07-27: `slot_starvation` telemetry landed** — surfaces a freed dispatch slot with no eligible candidate
  (`feasible_extra < extra`), shared by every async baseline.
- **07-27: all 9 baselines' smoke yamls bumped 1800s→3600s** + real/sim wall-ceiling watchdogs, matching the
  §C run-length bar (enabled the 07-28 full-scale batch, §A).
- **07-27: `[LAG_DECOMP]` + `redispatch_decomp` telemetry landed for fwdllm** — splits redispatch wall gap
  into peer-wait vs post-close overhead, ported from asyncfl's shared version.
- **07-27: 2 more stale "sim uses in-mem cache" doc claims found + fixed** (`checks.py` docstring,
  PARITY.md, PARITY_CHECKER_README.md) — no cache exists in either mode; real reason is dispatch-cadence /
  aggregator-side overhead.
- **07-27: `staleness` (U3) root-caused, FIXED, VALIDATED — CLOSED** (§D-9). Real's `_agg_pending_commit_ref`
  only covered return-time state, missing a still-training end and flooding it with re-dispatches (one end:
  526 sent vs 284 processed). Fixed via `_PendingCommitUnion` (dispatch-time + return-time, mirroring sim's
  `_sim_pending_commit` span). Confirmed on the post-fix pair: real 0.056/0.064 vs sim 0.095/0.12 (was real≫sim
  7.62/9.10 pre-fix). Unmasked a separate `throughput`/`per_round_advance` gap, not a regression from this fix.
- **07-27: `[RecvBootstrap]` deadlock FIXED + VALIDATED.** A bootstrap added 07-26 (for a trainer-side crash)
  also raced the aggregator's own first SEND tick, phantom-filling every dispatch slot and permanently
  freezing sim's vclock at 0.0 (reproduced at n=15 with zero GPU sharing possible — also falsified the
  GPU-contention hypothesis for the same eviction). Fixed by gating the bootstrap behind
  `allow_recv_bootstrap`, set only by single-parent callers (`channel.one_end()`); `channel.ends()`
  dispatchers default to no-bootstrap. Re-run confirms no repeat of the freeze.
- **07-27: `_exclude_pending_commit` FIXED + VALIDATED** (§D-8). Round-cadence's cohort-reuse cache-hit
  dispatch path bypassed the busy-exclusion guard entirely (called once per run, not per tick). Fixed by
  filtering the cache-hit list every tick. Flips `r1_inflight_overlap`/`selection_detail`/`participation`/
  `training_budget`/`overhead_residual`/`per_round_advance` fail→pass. NOT also a `staleness` fix — that was
  a separate, later-fixed bug (above).
- **07-27: variance-check pool stopped rate-scaling stale contributions** (§D-7) — was scaling
  round-cadence's genuinely-stale carried-surplus entries toward zero before the variance gate, faking
  convergence (real committed data_ids in ~1.7 JVP samples vs fluxtune's ~16). Fixed: stop scaling the
  var-check pool by `rate`; model-update merge untouched. Validated for cadence effects; also exposed a
  previously-masked `fluxtune` compounding effect (§B).
- **07-26: R-D landed + VALIDATED** — async distribute never had the sync path's one-instruction-per-
  `version_key` guard (§F-25); `fedbuff_round`'s `r1_inflight_overlap` was ~90% `VAR=bad` re-sends to
  already-busy trainers. Fixed via `_already_served_current_instruction`. Confirmed: `r1_inflight_overlap`
  real 0.0%/sim 0.0% both baselines (was ~90-91%); the flood's `phase_gpu_compute` contention side-effect
  also resolved (13.5s→4.8s fedbuff, 7.9s→3.6s felix).
- **07-26: `sample_by_util` reproducibility fix landed + VALIDATED** — `np.random.choice(p=...)` was
  pool/order-dependent; replaced with `_keyed_weighted_topk` (Efraimidis-Spirakis keys). Confirmed on
  `felix_it`: `cohort_sequence`/`v1b_iters_moving_avg`/`utility` all flip fail→pass.
- **07-26: `get_curr_unavail_trainers`/`get_curr_task_ineligible_trainers` INFO-logged every call, ungated**
  — a 300-entry trace scan + log fired every iteration-cadence tick in ORACULAR mode (4735×/1243s run);
  gated behind DEBUG (§F-19). Confirmed effective at scale 07-28 (`fedbuff_it_oracular` 12→4 fails, above).
- **07-26: R-C landed** — round-cache stuck-timeout now clocked on vclock in sim / wall in real
  (`_round_cache_clock_now`); sim evicted 0 stuck ends vs real's 6.
- **07-25: R-A landed** — `FedBuffSelector` re-based onto shared `AsyncSelectorBase` (832→54 lines),
  inheriting the version_key re-pick guard, R1 guard, vclock timeout, avl filter, full drain, `_keyed_topk`
  sampling. Real had been re-picking the same-version trainer on 34.4% of commits vs sim's 0.9% pre-fix.
- **07-25: R-B landed** — pinned round cohort was sized/checked against `agg_goal` instead of `c`
  (under-fill: `fedbuff_round` froze at 10/100 with 20 idle slots; over-fill: `felix_round` 30 real vs
  40 sim). Fixed: `_round_cohort_target`/`_trim_round_cohort` target exactly `c`.
- **07-25: `reselect_cadence` knob added** (round/data_bin/iteration); `AsyncRandomSelector` collapsed
  850→31 lines (zero methods of its own); selector stats de-duplicated ×4 into `AbstractSelector`; dead
  heartbeat mechanism deleted (no sender ever existed).
- **07-25: async-baseline decisions settled** — extraction reference is `async_oort.py` (not
  `async_random.py`, collapsed to a stub); cohort target is `c`, trimmed exactly (`agg_goal` is only the
  aggregation trigger); `reselect_cadence` is a first-class knob.
- **07-23: fluxtune 3→0 fails (69/0/16)** — the 3 remaining fails were ONE boundary-race cascade on a
  stochastic-async selector, not a sim bug (§D-2); checker now gates index-identity for stochastic-async
  selectors, keeps marginals enforced.
- **07-23: fwdllm timing family root-caused** — co-location contention (byte-identical inputs, sim's
  per-op floor matches real's typical every decile), not sim over-compute (§D-1). `_flat_grad_norm` fix
  landed but insufficient alone (§E).
- **07-23: fwdllm/fwdllm_plus `throughput` CLOSED at 7200s** — `recv_fifo`→`drain_ready` + var_bad dedup
  held (3.2%/4.8%, both PASS).
- **07-23: checker overhaul** — `matched_virtual_budget` deleted, graded on logical budget N instead
  (§D-4); `pctl_band_ok` DIST-band escape added; `_step_timing_compare`'s `band_min_abs_s` floor fixed (was
  masking 5x regressions at ms-scale, copied from a 1s-scale metric).
- **07-18→07-22 foundation** (compressed — full detail via `git log` on this file): P0-1 deferred-merge
  buffering landed (07-18); fluxtune's sim-side vclock/pacer/EOT-stamping bugs fixed and real's
  `_agg_pending_commit_ref` bound (07-20/21), taking fluxtune 19→5 fails at 7200s (07-22); startup
  GPU-health crashes fixed; `cohort_sequence` grading made distributional; `[SELECT_TRACE]` debug logging
  added then removed once the divergence was localized.
