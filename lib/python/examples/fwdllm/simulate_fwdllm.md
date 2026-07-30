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

## §A  Score — refreshed 2026-07-29 (post redispatch-charge-fix validation pairs)

**Latest run per baseline** (`run_parity.py`; ✓/✗/– = pass/fail/skip; PARITY.md §F). `fluxtune`,
`fwdllm`, `fedbuff_round`, `felix_round` now have a 5400s pair (up from 3600s); the other 6 remain
3600s+ n=100/c=30 except `fwdllm_plus` (STALE, 07-23). Open fails: §B.

**`fedbuff_round`/`felix_round`/`fluxtune` re-run with fresh sim legs against the SAME real logs**
(real unaffected, only the `redispatch_turnaround.weights` registry value + checker logical-budget fix
changed) — no longer STALE, this IS the validation run §B was awaiting. Result: **not a clean win** — see
§B, `throughput`/`per_round_advance` got WORSE, not better (§D-14). These three rows now PREDATE the §D-15
mid-cycle-redispatch fix (landed 07-29 pm, sim-only) — their sim legs need a re-run before the numbers mean
anything. `fwdllm_plus` still predates the 07-27/28 batch generally. `fwdllm` remains CURRENT (59/3/22, §G).

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260729_080111`/`_142902` | 5400s | 57/10/16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fwdllm/syn_0 | `run_20260729_020102`/`_033249` | 5400s | 59/3/22 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus/syn_0 STALE | `run_20260723_161647`/`_171829` | ~3600s | 61/2/21 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_it/syn_0 | `run_20260728_000201`/`_010416` | 3600s | 63/6/16 | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_unaware/syn_0 | `run_20260728_011134`/`_021316` | 3600s | 59/4/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_unaware/syn_0 | `run_20260728_015339`/`_025553` | 3600s | 67/1/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260728_022355`/`_032547` | 3600s | 58/5/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_oracular/syn_0 | `run_20260728_032938`/`_043152` | 3600s | 64/4/18 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fedbuff_round/syn_0 | `run_20260729_034223`/`_125616` | 5400s | 53/11/21 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_round/syn_0 | `run_20260729_055151`/`_134315` | 5400s | 59/8/18 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. Open fails: §B.

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

| baseline | open fails | next step |
|---|---|---|
| `fedbuff_round` (53/11/21) | `throughput`/`per_round_advance` (sim 14.0% FASTER) · `total_commits`/`terminal_state` (24%) · `v1_iter_per_data_id`/`v1b_iters_moving_avg` (sim 16.7 vs real 12.4) · `cohort_sequence` (2502 vs 1855 cycles) · `overhead_residual`/`agg_step_timing_breakdown` | ALL one root: §D-15 mid-cycle redispatch, FIXED, awaiting sim-only validation. V1/cohort are downstream (§D-11) |
| `felix_round` (59/8/18) | `throughput`/`per_round_advance` (sim 10.0% FASTER) · `v1_iter_per_data_id`/`v1b_iters_moving_avg` (sim 15.7 vs real 12.0) · `cohort_sequence`/`v2_var_trajectory`/`selection_detail`/`g2_grad_pool_size` (D-2 family, downstream) | Same §D-15 root + validation. Its 19 in-flight redispatches / 2 duplicate steps (0 in real) should also hit 0; else separate defect |
| `fluxtune` (57/10/16) | `throughput`/`per_round_advance`/`total_commits`/`terminal_state` (~7.9-8.5%, sim FASTER) · `v2_var_trajectory` (12.4%) · `preferred_duration` (21.5%, barely over) · `convergence` (5.7%, marginal) · `utility` (raw KS 0.213 fails, matched-window 0.112 passes — post-N tail artifact) · `drain_wall_budget`/`step_timing_breakdown` (D-1 family) | Same §D-15 root (78.3% retask) PLUS missing `sim_charge_profile_path`; both fixed, same validation run. Re-check `convergence`/`utility` there |
| `fwdllm`/`fwdllm_plus` | `drain_wall_budget` (GATING)/`step_timing_breakdown`/`agg_step_timing_breakdown` (DIAG) — pre-existing, D-1 co-location contention, unresolved (`sim_model_agg_compute_time` charges sim's own contention-inflated live span onto the vclock — the "charge-the-floor vs relax" decision is still open) | `throughput`/`terminal_state`/`total_commits` ROOT-CAUSED + FIXED 07-29 (§D-13, checker bug not a sim bug) — CLOSED, see §G |
| `fwdllm_it_unaware`/`fwdllm_it_oracular` | `terminal_state`/`total_commits`, new at 3600s | likely same §D-1 contention family (shares fwdllm's sync dispatch path) — not independently diagnosed |
| `felix_it` | `total_commits`/`terminal_state`/`cohort_sequence`/`convergence` | profile `redispatch_turnaround` from felix_it's own real log (§D-3 — timing parity isn't inherited from a sibling that shares only the selector) |
| `fedbuff_it_oracular` (64/4/18) | `total_commits`/`terminal_state`/`v2_var_trajectory`/`convergence` | same: profile own real log, don't reuse round-cadence's charge |
| `fedbuff_it_unaware` (67/1/18) | `convergence` only | duration-gated (§C bar 2h+), not a bug |

**07-29 pm — §D-14: the `redispatch_turnaround` charge fix landed correctly but did NOT fix `throughput`;
it overcorrected the DIRECTION.** Fresh sim-only reruns (`_125616` fedbuff_round, `_134315` felix_round,
same real logs) confirm: `sim_s_per_round` dropped enough that sim flipped from 8.2-8.6% SLOWER than real
to 10.0-14.0% FASTER — a worse relative gap than pre-fix, not better, and it now cascades into new
`v1_iter_per_data_id`/`v1b_iters_moving_avg`/`cohort_sequence.count` fails (sim races through more
iterations per data_id because its rounds are cheaper — §D-11 downstream-compounding pattern, not
independent bugs).

**Checked whether the reprofiled value (0.0598s) is itself under-scaled — it is NOT.** Two lines of
evidence, both from the SAME real telemetry the registry was built from:
1. The distribution is heavily right-tailed (`redispatch_decomp`'s `weights`-kind marginal: median
   ~0.011-0.013s, mean ~0.059-0.066s, max 4.7-5.2s, ~1.5% of events >1s) — a flat scalar mean charge is
   the right statistic for reproducing an AGGREGATE total (law of large numbers, n>4400 per baseline), not
   an under-estimate from averaging away the tail.
2. Directly summed real's TRUE per-batch total (`post_close_overhead_wall_s`'s last/cumulative position per
   `(data_id, iteration)` batch — the actual batch-total wall cost, not the double-counted raw pool D-12
   already fixed) against sim's own `vclock_charge` ledger sum for this label, per data_id:
   **fedbuff_round: real 296s vs sim-charged 325s (sim ~10% OVER, not under); felix_round: real 276s vs
   sim-charged 336s (sim ~22% OVER).** `redispatch_turnaround` is ruled out as the source of the residual
   sim-too-fast gap — raising it further would make the flip worse, not better.

**07-29 pm — §D-15: residual ROOT-CAUSED + FIXED. It was never a charge.** Sim released a trainer's slot at
its own COMMIT instead of at the agg-goal boundary, so it re-dispatched mid-cycle under the SAME
`version_key` that trainer had just contributed to — before that cycle's variance check had run, making
both possible payloads illegitimate (`var_bad` asserts a verdict with no evidence; `weights` would be
unchanged). Measured on the 5400s pairs: 78-90% of sim dispatches vs **0%** of real's; peak concurrency
38-46 against `c`=30; per-contribution idle 0.2s (sim) vs 1.6s (real) on identical busy time. **Not excess
messages or wasted compute** — dispatches/cycle 10.03 vs 10.02, forward passes per contribution 1.003 vs
1.002; the defect is purely the `sim_send_ts` stamp landing pre-close (§D-15). Substituting
real's idle collapses the residual **28.3%→4.8% / 24.9%→1.1% / 15.7%→3.1%**. Both configs declare
`inflight_residence: true`, so sim was violating the run's own contract — single-side, no diff needed
(§D-9). Fixed by deferring the release to `_release_sim_slots_at_agg_goal` (real's
`cleanup_recvd_ends()` twin); `redispatch_turnaround` stays at 0.0598s — do NOT re-tune it (§F-13, §D-14
verified it). **Needs a fresh sim run per baseline to validate** (real unaffected — sim-only policy + the
`fluxtune` charge-path fix below):
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode sim --only fedbuff_round,felix_round,fluxtune
```
Two INV tripwires shipped WITH the fix so that run grades itself instead of being argued about:
`concurrency_cap` (outstanding dispatched-not-committed vs `c`) and `retask_before_close` (dispatch to an
open cycle's contributor), both per-mode on `redispatch_decomp`, both wired into the checker + README.
Expected post-fix: `retask_before_close`→0, `concurrency_cap`→under 30, sim idle→real's ~1.6s, sim
`pastdated_commits`→0, throughput residual ~1-5%.

**Also fixed alongside: `fluxtune` never applied the charge registry at all** — its sim yaml had no
`sim_charge_profile_path`, so every `redispatch_turnaround` ledger row read `charge_source: "none"` and 0s
was charged (~180s, same direction as the main bug). §B's earlier claim that it shares the round-cadence
entry was false in effect. Added.

**Not the mechanism (checked, keep as negative results).** Past-dated commits are small and mostly the
legitimate carried-surplus case (sim 2.0-9.0% of cycles, 17-96s of 5400s; real `pastdated_commits = 0`) —
the vclock stall is dispatches ISSUED mid-cycle at a still-low vclock, not out-of-order commits. The
distinct-contributor gap (fedbuff real 30 vs sim 49) is a round-lap artifact: real never crossed the lap
boundary so exactly `c`=30 is correct; sim crossed because it ran further. Both are DOWNSTREAM of the
throughput gap (§D-11) — re-grade `cohort_sequence` only after the fix.

**07-28 evening: `redispatch_turnaround.weights` reprofiled — registry was stale, not cross-baseline-pooled.**
Split-by-baseline test (§B's prior next-step) on the ORIGINAL smoke-scale source runs
(`run_20260728_151831`/`_151722`) gave fedbuff-only mean_s=0.4143 vs felix-only 0.4585 (+10.7%
felix-over-fedbuff). Re-running the SAME split on the latest full 3600s real logs
(`run_20260728_175756`/`_175740`) gave fedbuff-only 0.4957 vs felix-only 0.4812 — **direction reversed**
(fedbuff now higher) and the gap shrank to 3%. A per-baseline difference that flips sign between two
independent sampling sessions is sampling noise, not a structural split — pooling wasn't the bug. The real
finding: BOTH latest full-run means (0.496, 0.481) sit ~11-14% above the live registry value (0.4365,
seeded from the smaller/older smoke runs) — the registry was stale. Refreshed via the sanctioned path
(`expt_scripts/profile_sim_charges.py`, pooled across both baselines' latest real logs, n 3264→6648):
`redispatch_turnaround.weights` mean_s 0.4365→0.488 (+11.8%). Landed in `sim_charge_profile.yaml`; 649
`-k "fwdllm or telemetry or parity"` pass. **Validated at 3600s (§G, CLOSED), REOPENED at 5400s.** A fresh
full real+sim pair per baseline at 5400s (`run_20260729_034223`/`_051440` fedbuff_round,
`run_20260729_055151`/`_072408` felix_round) flips `throughput`/`per_round_advance`/`total_commits`/
`terminal_state` back to failing — but in the OPPOSITE direction from the pre-fix gap: sim is now 8.2-8.6%
SLOWER than real (matched-window ratio median 1.13, max 1.72 for fedbuff_round), not faster/undercharged.
The registry value is unchanged (0.488, confirmed in `sim_charge_profile.yaml`) between the two runs, so
this isn't a config drift.

**07-29: ROOT-CAUSED + FIXED — mechanism in §D-12** (profiler pooled a cumulative per-batch reading as a
flat per-trainer charge, overcharging 7.35x/8.34x on fedbuff_round/felix_round's 5400s pair, ≈1900-2100s
of pure excess vclock, ~40% of the run). `profile_sim_charges.py` gained `_load_real_redispatch_marginal`
(first-difference per batch, `redispatch_turnaround` only). Re-run on the same 5400s real logs:
`redispatch_turnaround.weights` mean_s 0.488→**0.0598** (n=9310); projected against sim's own event counts,
lands within 1-13% of real's true cost instead of 7-8x over. 663 `-k "fwdllm or telemetry or parity or
charge"` pass. **Needs a fresh sim run to validate** (real unaffected, §D-12 — charge is sim-only; also
picks up §D-13's checker fix, and `fluxtune` shares the same registry entry — real doesn't need a rerun):
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode sim --only fedbuff_round,felix_round,fluxtune
```

**07-28 evening: felix_round's `v2_var_trajectory` candidate corrected — `aggregate_grad_pool()` ruled
out, redirected to `calculate_var()`'s split-half.** §B previously pointed at `aggregate_grad_pool()`
(`fwdllm_aggregator.py:957`) as a grad-list-order bug candidate. Reading it: it's a plain element-wise SUM
over `self.grad_pool` (≤`max_iterations_per_data_id` items) — float-addition reordering there is ~1e-6
relative, not a 3% mean shift, AND it feeds the outgoing `GRAD_POOL` payload field, not the variance gate.
The function that actually computes `var` is `calculate_var()` (`fwdgrad_utils.py:171`) — a SPLIT-HALF
stat (`var(mean(first_half), mean(second_half))`) over `grad_for_var_check_list`, whose membership order
comes from `_per_agg_trainer_list` post-`_canonicalize_cohort_commit_order()`. Checked live telemetry
(`run_20260728_175740`/`_165628`, matched data_id=0/iteration=6 `agg_round` events): `contributing_trainers`
is naturally speed-ascending in BOTH modes (canonicalization/arrival order isn't the gap —
`inter_arrival_order` mean_spearman_rho=1.0 agrees), but the COHORT COMPOSITION itself differs at this
exact (data_id, iteration) coordinate — real: 6 trainers @4.297s + 3 @5.524s + 1 @9.819s; sim: 8 @4.294s +
2 @8.6-9.2s — different speed-mix, different `grad_norm`s, feeding split-half `var` differently. This
traces to the SAME cohort-composition divergence `cohort_sequence.composition` shows (mean_overlap 0.387).
**OPEN, NOT a reclassification candidate (operator override 07-28): selection + aggregation are meant to be
deterministic given matched inputs — composition SHOULD match exactly.** `fedbuff_round` doesn't show this
composition gap at the same severity, so treating `felix_round`'s as accepted stochastic identity (à la D-2)
understates rather than explains it — do not gate `v2_var_trajectory` to diagnostic. Unresolved question to
chase next: WHY does `felix_round`'s cohort composition diverge from sim (and from `fedbuff_round`'s) when
selection should be deterministic — root-cause that, don't grade around it. Kept open in §B.

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
machine with slot/guard sets derived — §D-15 was exactly a lifetime bug inside that split. Simplify after
the timing fix validates; scope behind `test_fwdllm_sim_grad_loop.py`'s
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
a real mix bias. (Mirrors PARITY.md's refl S2 core-identity lesson.) **Extension (07-28):** the same
composition divergence propagates into ANY per-cycle stat computed over the specific contributor SET, not
just `cohort_sequence`'s own set-membership rungs — `felix_round`'s `v2_var_trajectory` mean shift traces to
`calculate_var()`'s split-half stat reading a different real/sim speed-mix at matched (data_id, iteration)
coordinates (telemetry-confirmed, §B), despite `v1_iter_per_data_id`/participation marginals passing. A
downstream rung failing while its INPUT rung (V1, participation) passes does not by itself prove a mechanism
bug (contra the naive read of the decomposition tree below) — check whether the failing rung's statistic is
sensitive to SET MEMBERSHIP (not just count/marginal) before walking further down.

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
what it could not do was name the mechanism, because no charge could.

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

- **07-29 pm: round-cadence + fluxtune throughput residual ROOT-CAUSED as sim's mid-cycle redispatch
  (§D-15); release deferred to the agg-goal boundary + 2 INV tripwires shipped. NOT yet closed — awaiting
  the sim-only validation run (§B).**
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
