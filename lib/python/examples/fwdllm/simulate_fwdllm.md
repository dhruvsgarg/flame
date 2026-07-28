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

## §A  Score — refreshed 2026-07-28

**Progression checklist, round-cadence/oracular family (07-25 → 07-28) — commit → rung(s) flipped.** The
big pass-count jumps this window are concentrated here; `fwdllm`/`fwdllm_it_*` (timing family, §D-1) and
`felix_it` stayed flat — no landed fix targets them yet.

| # | Commit | Fix | Rung(s) flipped | Baseline Δ (pass/fail) |
|---|---|---|---|---|
| 1 | `945b815f` | R-A: `FedBuffSelector` re-based onto shared `AsyncSelectorBase` (832→54 lines) | `r1_inflight_overlap`, `avail_composition`, staleness cluster | fedbuff family: went from broken/undefined selection to a working baseline |
| 2 | `945b815f` | R-B: round cohort sized/trimmed to `c`, not `agg_goal` | `cohort_sequence` (matched-but-wrong → real), `participation` | `fedbuff_round`/`felix_round` unfroze from 10-of-30/30-of-30 lockstep |
| 3 | `908280b6` | R-C: round-cache stuck-timeout clock is vclock in sim, wall in real | latent stuck-end evictions | sim 0 stuck vs real 6, pre-existing gap closed |
| 4 | `842e7600` | R-D: one-instruction-per-`version_key` guard ported to async distribute + `sample_by_util` keyed-weighted-topk reproducibility | `r1_inflight_overlap` 90%→0%; `felix_it` `cohort_sequence`/`v1b_iters_moving_avg`/`utility` | `fedbuff_round`/`felix_round` `r1_inflight_overlap` fixed; `felix_it` 3 rungs fail→pass |
| 5 | `161ac725` | `[RecvBootstrap]` gated behind `allow_recv_bootstrap` (was phantom-filling dispatch slots on the aggregator's own channel) | fixed a sim permanent deadlock (`vclock_now` stuck at 0.0) | `fedbuff_round_n15_smoke_sim` terminates instead of hanging forever |
| 6 | `2d181d8f` | `_exclude_pending_commit`: round-cohort cache-hit dispatch now re-checks the busy-exclusion guard every tick | `selection_detail`, `participation`, `training_budget`, `overhead_residual`, `per_round_advance` | `fedbuff_round` 42/18/21→56/7/21; `felix_round` 43/17/21→54/9/21 |
| 7 | `5f990fc0` | Variance-check pool stopped rate-scaling stale contributions (was faking convergence) | `v1_iter_per_data_id`, `v5_variance_pass_ratio`, `g2_grad_pool_size` | both round-cadence baselines, +3 rungs |
| 8 | `918ef533` | Real's `_agg_pending_commit_ref` now unions dispatch-time + return-time state, matching sim's `_sim_pending_commit` span | `staleness` (U3) | real 7.62/9.10 → 0.056/0.064 (now *below* sim); CLOSED |
| 9 | `d859de05`+`d21d0402`+`0a76365a` | Fixed `run_sequential.sh`'s stale n10→n100 baseline mapping + bumped runtime to 3600s | unblocked — enabled the first full-scale (n=100/c=30) 3600s+ run for `fedbuff_round`/`felix_round` at all | full 07-28 batch (below) became possible |

**Net, full-scale 07-28 vs the pre-window 07-26 WIP baseline** (pass/fail, same rung set): `fedbuff_round`
42/18 → **57/7**; `fedbuff_it_unaware` 62/6 → **67/1** (5 of 6 remaining fails were duration-gated per §C,
confirmed by simply running longer — not a new fix); `fedbuff_it_oracular` 55/12 → **64/4**. `fwdllm`/
`fwdllm_it_*`/`felix_it` didn't move (still gated on the open §D-1 timing-family decision, §B) — so the
improvement is real but scoped to the round-cadence/`fedbuff_it_*` family, not baseline-wide.

**fluxtune/syn_0 — CLEAN** (69/0/16): the three former fails were ONE boundary-race cascade, resolved by
gating index-identity for stochastic-async selectors (→ §D-2, §G). Only the deferred 81% accuracy drop
remains (§B).
**fwdllm / fwdllm_plus/syn_0 — timing family OPEN:** `drain_wall_budget` (GATING) +
`step_timing_breakdown`/`agg_step_timing_breakdown` (DIAG) are ONE co-location-contention family
(root-caused → §D-1); fix-1 `_flat_grad_norm` landed but insufficient, charge-floor-vs-relax decision
open (§B).
**Flag inventory:** `sim_model_agg_compute_time` ON all three; `sim_sct_ordered_drain` +
`sim_model_dispatch_queue` fluxtune-yaml-only (promotion call §B).
**Round-cadence family (`fedbuff_round`/`felix_round`) — R-D guard VALIDATED 2026-07-26; `_exclude_pending_commit`
staleness-cadence fix LANDED + VALIDATED 2026-07-27 (commit `2d181d8f`); `staleness` (U3) itself
root-caused, FIXED, and now VALIDATED on a fresh post-fix run (commit `918ef533`, §G).** R-D (§F-25 guard, §G) and the var-check-pool rate
fix (§D-7, §G) together close 12 rungs at n15/1800s vs the 2026-07-26 POST-FIX pair: `overhead_residual`,
`per_round_advance`, `selection_detail`, `participation`, `training_budget`, `phase_weights_to_gpu`,
`v1_iter_per_data_id`, `v5_variance_pass_ratio`, `g2_grad_pool_size` on both; `agg_step_timing_breakdown`/
`cohort_sequence`/`v1b_iters_moving_avg` on `fedbuff_round` only (still open on `felix_round`, §B).
`fedbuff_round` 42/18/21→56/7/21→**53/10/21**, `felix_round` 43/17/21→54/9/21→**54/9/21**
(`run_20260727_150357`/`_153538` fedbuff, `run_20260727_150412`/`_153552` felix, both 1800s, post-`918ef533`).
`overhead_residual`/`per_round_advance` flip back to fail on this pair (both baselines) — see §B, next rung
to chase. Prior re-run also confirmed the `phase_gpu_compute` flood-contention hypothesis from last session: real
mean compute-phase wall dropped 13.5s→4.8s (`fedbuff_round`), 7.9s→3.6s (`felix_round`) once the flood
stopped — CLOSED, §G (`felix_round`'s `phase_gpu_compute` fails again on this pair too, §B).
Oort's utility-weighted draw (`sample_by_util`) fix is also validated: `felix_it`'s
`cohort_sequence`/`v1b_iters_moving_avg`/`utility` all flip fail→pass.

**`staleness` (U3) — CLOSED, VALIDATED 2026-07-27 post-`918ef533` (§G).** `fedbuff_round` real mean 0.056 vs
sim 0.095 (KS 0.039); `felix_round` real mean 0.064 vs sim 0.12 (KS 0.056) — both well inside DIST tol, real
now *lower* than sim (was real≫sim pre-fix: 7.62/9.10 vs sim ~0.1). Confirmed on both round-cadence
baselines, same shared dispatch/pending-commit code path.

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fedbuff_round/syn_0 (post-`918ef533`) | `run_20260727_150357`/`_153538` | 1800s | 53/10/21 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| felix_round/syn_0 (post-`918ef533`) | `run_20260727_150412`/`_153552` | 1800s | 54/9/21 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |

fluxtune/syn_0 (n15) row dropped from this table — its 61/7/16 pair (`run_20260727_021855`/`_032028`, 3600s)
predates today's fix and is unrelated (fluxtune isn't round-cadence); cited in prose above by telemetry only
(staleness max, not the full rung set — not re-run this session, don't infer other columns from it).
STALE 3600s n15 rows (`run_20260727_005613`/`_015747` fedbuff, `_005705`/`_015909` felix) superseded by the
1800s pair above — those predate today's `2d181d8f` fix. `fedbuff_round`'s new fail vs the old 3600s pair:
`utility` (DIST, pooled_ks 0.311 vs tol 0.2) — downstream of `staleness` (utility deps on it, PARITY.md
rung table), not an independent regression; `felix_round`'s utility passes (ks 0.114).

**Latest run per baseline** (`run_parity.py`; ✓ pass · ✗ fail · – skip; rung catalog PARITY.md §F):

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260723_044359`/`_064615` (agg_goal=10) | ~7200s | 69/0/16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm/syn_0 | `run_20260723_161459`/`_171648` (agg_goal=10) | ~3600s | 59/3/22 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus/syn_0 | `run_20260723_161647`/`_171829` (agg_goal=10) | ~3600s | 61/2/21 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

STALE — the three run-dir pairs above were cleaned from disk; numbers carried forward per this doc's rule,
not re-verified this session. Re-run via `run_parity.py --baselines <name>` when either is touched.
**fluxtune row doubly stale as of 07-27**: the var-check-pool rate fix (§G) touches the same shared method
its V1/V2/V5/`cohort_sequence` rungs depend on — expected inert, not yet confirmed. Don't cite as current
until the next long fluxtune pair re-runs.

**9-baseline batch, full scale (`run_parity.py`, 2026-07-28, n=100/c=30/agg_goal=10, 3600s target — clears
the §C sign-off bar for all rungs, including duration-gated ones).** All 9 baselines now have a usable
real/sim pair at this scale; the two that initially dropped (`felix_round` real killed mid-run/no sim,
`fluxtune` neither side ran) were relaunched clean same-day (`run_20260728_102750`/`_113005` felix_round,
`run_20260728_102831`/`_113046` fluxtune) — see update note below the table. This table supersedes the
2026-07-26 ~1800s WIP batch below for the baselines it covers; the 2026-07-27 `fedbuff_round`/`felix_round`
1800s rows and the 2026-07-26 `fluxtune` WIP row are now superseded too and kept only as prior-state history.

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| felix_it/syn_0 | `run_20260728_000201`/`_010416` | 3600s | 63/6/16 | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm/syn_0 | `run_20260728_000209`/`_010330` | 3600s | 59/3/22 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_round/syn_0 | `run_20260728_000744`/`_010958` | 3600s | 57/7/21 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_it_unaware/syn_0 | `run_20260728_011134`/`_021316` | 3600s | 59/4/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_unaware/syn_0 | `run_20260728_015339`/`_025553` | 3600s | 67/1/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260728_022355`/`_032547` | 3600s | 58/5/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_oracular/syn_0 | `run_20260728_032938`/`_043152` | 3600s | 64/4/18 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| felix_round/syn_0 | `run_20260728_102750`/`_113005` | 3600s | 55/9/21 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fluxtune/syn_0 | `run_20260728_102831`/`_113046` | 3600s | 58/10/16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |

**Update 07-28 (later same day): both missing pairs relaunched clean, batch now 9/9** — `felix_round`
(`run_20260728_102750`/`_113005`) and `fluxtune` (`run_20260728_102831`/`_113046`), both 3600s. Neither
repeated the earlier early-exit/no-launch symptom (§G); the DIRTY_ABORT hypothesis was never confirmed and
is now moot for this pair — flag again only if a future batch drops runs the same way.

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. At full scale
+ duration, `fedbuff_it_unaware` and `fedbuff_it_oracular` are now the cleanest of the round-cadence/oracular
family (1 and 4 fails); `fedbuff_round` keeps the `throughput`/`v2_var_trajectory`/`terminal_state`/
`total_commits` cluster seen at 1800s (§B, unchanged root). `felix_it` picked up a new `cohort_sequence` fail
not present in the 07-26 1800s pair — check against §D-2's independent-draw-floor gate before treating as a
regression (not yet done this session). `fwdllm`/`fwdllm_it_*` stay on the §D-1 shared-compute timing family
(`step_timing_breakdown`/`drain_wall_budget`/`agg_step_timing_breakdown`), now joined by `terminal_state`/
`total_commits` on `fwdllm_it_unaware`/`fwdllm_it_oracular` (new at this duration — not seen at 1800s, likely
downstream of the same timing family, not independently diagnosed this session).

**`felix_round` reproduces `fedbuff_round`'s exact fail cluster** — `throughput`/`per_round_advance`/
`overhead_residual`/`v2_var_trajectory`/`terminal_state`/`total_commits` all fail on both round-cadence
baselines, same shared dispatch path (§B dispatch-path table); `felix_round` additionally fails
`cohort_sequence` (§D-2 stochastic-identity candidate, not yet gated) where `fedbuff_round` passes it. This
is the strongest cross-baseline signal so far for a SHARED root over the round-cadence dispatch path, not two
independent bugs — see §B decision note.
**`fluxtune` now also fails `throughput`/`per_round_advance`/`total_commits`/`terminal_state`** — a new
result at this scale/duration (the STALE 7200s-clean row above predates the round-cadence fixes and wasn't
this rung set). Fluxtune is iteration-cadence, not round-cadence (§D-10), so this is either (a) the same
underlying mechanism reaching further than the dispatch-path label suggests, or (b) a coincidental
scale-driven regression independent of round-cadence. Not yet discriminated — §B.

`fluxtune`'s only remaining WIP-scale (not yet re-run full-scale post-fix) pair: `run_20260726_130810`/
`_134012`, 1800s, 60/7/16 (fails: `per_round_advance`, `preferred_duration`, `step_timing_breakdown`,
`drain_wall_budget`, `cohort_sequence`, `v1b_iters_moving_avg`, `v2_var_trajectory` — all §D-1 timing-family
or duration-gated, consistent with the STALE 7200s-clean row above, not a regression).

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

⭐ **`throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/`terminal_state` ROOT-CAUSED
(round-cadence family, 07-28 telemetry-only session, not yet fix-landed).** All five rungs are ONE shared
root, not five bugs — `total_commits`/`terminal_state` are literally `sim_vclock_to_n_s` vs
`real_time_to_n_s` (cumulative round-advance to matched N), and `throughput` is rounds/final_vclock, so all
three are arithmetic functions of `per_round_advance`/`overhead_residual`'s single per-round-advance gap
(§C ladder: walk to the lowest broken rung with sound inputs). `v2_var_trajectory`'s marginal DIST fail
(mean_rel_diff 0.02-0.039 vs 0.02 tol — barely over) is very likely the same downstream cascade (differing
cycle counts from a shifted timeline), not independently diagnosed.

**Localized below `per_round_advance` itself, using `agg_round` telemetry (both `fedbuff_round`/
`felix_round`, 07-28 full-scale pairs) — the gap is NOT one per-round cost, it's a per-VARIANCE-ITERATION
cost that compounds ~10x:**
- Per data_id (`_per_progress_last_event`'s round key), real needs a fresh WEIGHTS/`VAR=bad` cycle
  ~9.8-9.9× before variance passes (1130 `agg_round` events / 114 data_ids, felix_round; 1203/110,
  fedbuff_round) — confirmed via literal event counts, not assumed from `var_threshold`.
- Per-CYCLE wall advance (consecutive `agg_round` events, same or new data_id — no distinction, both equally
  slow): real mean 2.96s (felix)/2.79s (fedbuff) vs sim's own vclock-basis mean 2.29s/2.20s — a genuine
  ~25-30% per-cycle gap, present on EVERY cycle (intra-data_id retry and inter-data_id alike), not just the
  final one. Compounding ~10× reproduces the full observed ~15-21% per-round(data_id) gap (reconciled
  directly: felix_round's 114-span mean-per-data_id wall = 29.13s, matching the checker's own
  `real_mean_advance_s=29.35` almost exactly).
- **Ruled OUT with hard telemetry, do not re-chase (candidates for §E):** (a) aggregation-compute
  contention (§D-1's fwdllm/fwdllm_plus mechanism) — `drain_tail_s`/`aggregate_fedavg_s` are NOT larger in
  real (felix: real 0.31s/0.069s vs sim 0.40s/0.094s — sim's own is larger if anything), and both are
  already explicitly charged onto the vclock symmetrically via `charge_sim_vclock_overhead`
  (`fwdllm_aggregator.py:2140-2144`) — not an unmodeled asymmetry. (b) per-trainer modeled compute duration
  — matched real vs sim `agg_observed_s` for the SAME trainer ID: ratio 1.0001-1.0008 (30/30 common
  trainers, felix_round) — essentially bit-identical, the "productive" compute time is faithfully modeled.
  (c) per-message MQTT fetch/receive wall (`phase_mqtt_fetch`/`_fetch_weights` DIAG) — percentile bands show
  sim ≥ real at p50/p90, not real-slower.
- **The one component that DOES show a clean, reproducible, cross-baseline gap:** `redispatch_decomp`'s
  `post_close_overhead_wall_s` (round-close → this end's actual next dispatch) — real mean 0.499s
  (felix)/0.524s (fedbuff) vs sim 0.034s/0.038s, a consistent ~13-15x gap on BOTH baselines (n=3387/3290
  real events). `peer_wait_wall_s`'s much larger real-vs-sim gap (1.8-1.9s vs ~0.08s) is NOT independent
  evidence — it's structurally circular (a longer real round mechanically inflates average per-trainer
  wait; the event's own docstring warns of this, `events.py:349`). **But `post_close_overhead` is only
  instrumented for `_pk == "weights"` (fresh-dispatch) sends (`fwdllm_aggregator.py:4103`) — the ~9-of-10
  cycles per data_id that are `VAR=bad` "keep training" pings share the IDENTICAL dispatch loop
  (`_distribute_weights_async`, same `channel` send call, `fwdllm_aggregator.py:4005-4039`) but carry ZERO
  telemetry today.** Given the magnitude match (post_close's measured ~0.46-0.52s gap vs the ~0.6-0.7s
  per-cycle gap implied by the compounding reconciliation above), the working hypothesis is that the SAME
  uninstrumented, unmodeled-onto-vclock turnaround recurs on every cycle, not just the terminal one.

**Telemetry widened + LANDED 07-28 (not yet re-run live) — two changes, both routed through the ONE shared
aggregator so every fwdllm-family baseline gets them for free, no per-baseline repeat:**
1. `redispatch_decomp`'s emission gate now covers `_pk in ("weights", "var_bad")` instead of
   `_pk == "weights"` only (`fwdllm_aggregator.py:4135`), with a new `payload_kind` field so the two can be
   told apart post-hoc (`build_redispatch_decomp`, `events.py`).
2. **New `vclock_charge` ledger event** (`build_vclock_charge`, `events.py`) — `charge_sim_vclock_overhead`
   (the ONE function that folds any measured wall span onto the vclock, currently called for `drain_tail`/
   `fedavg` only) now emits this event on EVERY call, both modes: `label`, `span_s` (measured), `charged_s`
   (what actually landed on the vclock — 0.0 in real always), `time_mode`, `vclock_now`, `payload_kind`. This
   directly answers "is a real-only cost reflected in the vclock at all" for ANY category, not just the one
   under investigation — a real-vs-sim `span_s` gap with `charged_s` staying 0 IS the §F-1 unmodeled-cost
   signature, no inference needed. New optional `charge: bool = True` param: `charge=False` still emits the
   ledger (so real/sim stay comparable) but never calls `vclock.advance()`, regardless of mode/flag — for a
   candidate category not yet decided on. **`redispatch_turnaround` (the post-close span, both `weights` and
   `var_bad`) is wired through with `charge=False`** (`fwdllm_aggregator.py:~4175`) — instrumented for
   visibility only; nothing about any baseline's vclock behavior changes from this alone.

4 new tests total (2 `vclock_charge` ledger tests in `test_fwdllm_sim_grad_loop.py`, 1
`payload_kind`/`redispatch_turnaround` test in `test_fwdllm_redispatch_decomp.py`, 1 `payload_kind` assertion
on the existing weights test); 573 `tests/mode -k "fwdllm or telemetry or parity"` pass, 0 regressions.
**Needs one fresh pair to actually read the numbers** — this session only landed instrumentation, it did not
re-run live (no broker in this environment, per this doc's own rule). Once a pair lands: group
`redispatch_decomp`/`vclock_charge` events by `payload_kind`, compare `post_close_overhead_wall_s` (and
`vclock_charge`'s `span_s` for `redispatch_turnaround`) real-vs-sim for `var_bad` the same way this session
did for `weights` (§D-11) — if elevated similarly (~13-15x), that CONFIRMS the per-cycle-turnaround
hypothesis across all ~10 cycles/data_id, not just the terminal one, and the fix becomes flipping that ONE
call site's `charge` to `True` (mirroring `drain_tail`/`fedavg`, §F-20 — never inject noise into sim, so this
is charging a real, measured cost, not tuning sim down). Scoped to the shared `_distribute_weights_async`
path — fixing it should flip `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/
`terminal_state` together on BOTH `fedbuff_round` and `felix_round` (§F-11).

**`fluxtune` — same-direction signal already visible in TODAY's telemetry, not yet proven same mechanism.**
`fluxtune` is async/iteration-cadence (§D-10), not round-cadence, but shares the identical
`_distribute_weights_async`/`charge_sim_vclock_overhead` code, so it already has `redispatch_decomp` events
(no widening needed to see this much): `post_close_overhead_wall_s` real mean 1.695s vs sim 0.178s (~9.5x,
n=2068 real events, 07-28 pair) — same DIRECTION and comparable order of magnitude to round-cadence's
13-15x. But `fluxtune`'s `per_round_advance` fails on a DIFFERENT signature: KS 0.5-0.52 (vs 0.2 tol, badly
over) with only 6.5-8.5% mean_rel_diff (vs 15% tol, actually PASSING the mean check) — round-cadence fails
both mean AND KS together. A large-KS/small-mean failure reads as a distributional SHAPE mismatch (e.g. a
heavy right tail on some data_ids, `matched_window_ratio_max` 3.46 vs round-cadence's 2.3-9.9 — comparable,
not clearly distinguishing), not a uniform per-cycle additive shift — so don't assume this is byte-identical
to round-cadence's mechanism until the widened telemetry is read on a fresh fluxtune pair. `fluxtune` also
needs far more cycles/data_id than round-cadence (~18.6 vs ~9.8, today's pair) — MORE opportunities for the
same per-cycle gap to compound, consistent with (but not proof of) the same root.

**Why did `fluxtune` pass clean (69/0/16) on 07-23 but fail 10 rungs today, with no `fluxtune`-targeted code
change in between?** Leading hypothesis, not yet confirmed: the 07-27 variance-check-pool rate-scaling fix
(§G, §D-7) — which correctly stopped scaling stale/carried contributions toward zero before the variance
gate — makes variance readings HONEST (higher) instead of artificially low, so MORE variance-retry cycles
are now needed before a data_id's gate passes. This doc ITSELF already flagged the exposure in writing
BEFORE today's run: "the var-check-pool rate fix touches the same shared method [fluxtune's] V1/V2/V5/
`cohort_sequence` rungs depend on — expected inert, not yet confirmed" (§A, 07-27 note). Today's pair is the
first re-verification since that fix landed, and `v2_var_trajectory` DOES now fail for `fluxtune` too — direct
confirmation the variance computation shifted, exactly the flagged risk. The mechanism connecting that to
`throughput`/`per_round_advance`: the SAME uninstrumented per-cycle `post_close_overhead` gap shown above was
presumably ALWAYS present, but mattered less when fewer cycles were needed per data_id (pre-fix, artificially
-low variance passed the gate sooner) — MORE cycles now needed means the SAME per-cycle gap compounds MORE,
crossing tolerance for the first time. This reframes the finding: **not a new bug introduced today, and not
purely a scale effect — a correct fix (07-27) removing a convergence-faking shortcut increased how many
times a pre-existing, still-uncharged per-cycle cost gets to compound.** Can't fully confirm without the
07-23 run's raw telemetry (cleaned from disk per this doc's own STALE-data policy, so the pre-fix cycles/
data_id count isn't recoverable) — treat as the leading, well-corroborated hypothesis, not a closed root
cause. `fluxtune`'s newly-failing `throughput`/`per_round_advance` (§A) needs its OWN read of the widened
telemetry once a fresh pair lands, not an assumption that fixing round-cadence's `redispatch_turnaround`
charge automatically covers it.

**Known gap, not newly introduced:** `redispatch_decomp` (landed 07-27) and the new `vclock_charge` event
have no reader in `scripts/analysis/analyze_run.py` — checked, zero hits for either event name in any
analysis/plotting script. Per §F-8 ("ship telemetry + plot + pytest together... a field with no reader is
dark data") this is already a standing violation predating this session's changes, not something introduced
here. Flagged, not fixed — building the plot is a separate, larger scoped task; this session's python-script-
direct-read approach (as used throughout §B this session) is a working substitute for now.

**`selection_detail` CLOSED — CONFIRMED at production scale.** `_exclude_pending_commit` fix (§D-8, §G)
passes on both `fedbuff_round`/`felix_round` at n=100/c=30/3600s (07-28, §A) — re-confirmation from n15 is
done; unblocks decision 4 below.

⭐ **Open — `throughput`/`per_round_advance`/`overhead_residual` still FAIL on `fedbuff_round`
(07-28, full scale): real 15-27% SLOWER per round than sim** (fedbuff 73.15s vs 62.12s pre-fix numbers,
direction confirmed unchanged at full scale). Symptom: `step_timing_breakdown`'s `phase_mqtt_fetch_s` — real
spends 2.0-2.3s mean waiting on `channel.recv()` for its next dispatch, sim reads ~0s. Ruled out: sim
in-memory shortcut (both modes run the identical MQTT `_fetch_weights` path — no cache exists in either
mode, a prior doc note claiming otherwise was fabricated and is now fixed), `sim_model_agg_compute_time`
(off on this pair), aggregator SEND-loop being busy (only 15% duty cycle, weak overlap with trainer wait).
Most of the gap is plausibly genuine peer-wait (waiting on round/cohort-mates before the pinned cohort's
`version_key` advances, §D-8/F-25) already modeled via the vclock's `max()` — not necessarily an unmodeled
cost. **`[LAG_DECOMP]` + `redispatch_decomp` telemetry landed (§G) to settle this on the next run** — a prior
offline reconstruction attempt gave inconsistent numbers depending on method and should not be trusted or
reused. Re-run `fedbuff_round`/`felix_round` and read the new fields before touching `sim_redispatch_gap_s`
(unused by fwdllm today; calibrating it off unreliable data would violate §F-13).

**`fedbuff_it_oracular` — CONFIRMED at scale: 12→4 fails (07-26→07-28, §A).** Root for the 8 that closed:
`get_curr_unavail_trainers`/`get_curr_task_ineligible_trainers` did a 300-entry trace scan + unguarded INFO
log every iteration-cadence tick in ORACULAR mode only (4735×/1243s run) — gated behind DEBUG (§G, §F-19).
Remaining 4 fails (`v2_var_trajectory`, `terminal_state`, `total_commits`, `convergence`) match `fedbuff_round`'s
still-open throughput gap above, not a separate oracular-specific bug.

**`fedbuff_it_unaware` — CONFIRMED duration-gated, not a bug: 6→1 fails (07-26 1800s→07-28 3600s, §A).**
All 5 that closed (`cohort_sequence`, `v1b_iters_moving_avg`, `v2_var_trajectory`, `terminal_state`,
`total_commits`) were rungs §C says need 3600s+ to grade — simply running longer resolved them, no code
change needed. Only `convergence` remains (needs the full 2h+ §C bar).

**Decisions settled 2026-07-25 (implemented, §G):**
1. `reselect_cadence: round | data_bin | iteration` is a first-class knob (`data_bin` unused so far, wired
   for a future cadence ablation).
2. Cohort target is `c`, trimmed to exactly `c` — `agg_goal` is only the aggregation trigger.
3. Extraction reference for new async baselines is `async_oort.py`, not `async_random.py` (the latter had
   zero methods of its own, now collapsed to a stub kept only for parked `fluxtune_dynkc`).
4. **`async_oort` is not yet re-based onto `AsyncSelectorBase`** — its Oort scoring/eval branches need
   careful `_choose`/`_pre_choose` mapping. Still BLOCKED: `felix_round` was the intended clean control but
   isn't clean yet (open throughput gap above, plus the 07-28 early-exit needing operator follow-up, top of
   this section). `felix_it` (no round-cache) is closer to clean and may serve once its own duration-gated
   fails (§A) are confirmed non-regressions. Once unblocked: fold `async_oort` into
   `test_async_selector_base.py`'s `BUILDERS` (§F-26).

**Dispatch-path reference** (which baseline owns its version_key guard where — R-A/R-B/R-C/R-D all FIXED,
§G; kept here as a quick-lookup, not an open item):

| baseline | dispatch path | cadence | version_key guard from |
|---|---|---|---|
| `fwdllm`, `fwdllm_it_*` | sync gate | mixed | aggregator (`_reselect_true_cache_key`) |
| `felix_it` | async gate | iteration | selector (`async_oort`) |
| `fedbuff_it_unaware/oracular` | async gate | iteration | selector (shared `AsyncSelectorBase`, post-R-A) |
| `fedbuff_round`, `felix_round` | async gate | round | selector, pinned-cohort trimmed to `c` (post-R-B/R-C) |

### fwdllm / fwdllm_plus — shared-compute timing family (co-location contention, root-caused → §D-1)

Only `drain_wall_budget` GATES (MECHANISM); `step_timing_breakdown` + `agg_step_timing_breakdown` are DIAG.
fix-1 `_flat_grad_norm` LANDED + re-measured (3600s, 07-23): `drain_tail_s` p90 rel UNCHANGED (fwdllm
0.935→0.942, fwdllm_plus 0.912→0.919) — fix-1 alone insufficient; `sim_model_agg_compute_time: true` still
charges the contention-inflated drain wall onto the vclock (`per_round_advance` central-escape still needed,
KS 0.371/0.108).

| Baseline | Rung | State | Next |
|---|---|---|---|
| FW, FW+ | `drain_wall_budget` (GATING) | fix-1 landed + re-run; p90 rel unchanged (0.94/0.92) | decide charge-floor vs relax |
| FW, FW+ | `agg_step_timing_breakdown` (DIAG) | same contention; non-gating | informational |
| FW | `step_timing_breakdown` (DIAG) | same; non-gating | informational |

**Decision open:** charge-the-floor vs relax — do NOT inject sim-host noise into the clock (§F-20); or keep
digging for the actual contention amplifier (fix-1 wasn't it — §E).

**Flag-promotion (operator call, [[flag-gate-ab-lifecycle]]).** `sim_model_agg_compute_time` effectively
default (ON all three). `sim_sct_ordered_drain` + `sim_model_dispatch_queue` are fluxtune-yaml-only but both
model GENERAL async-transport artifacts, not fluxtune-specific. Next: run fwdllm/fwdllm_plus sim smoke with
both ON to confirm inert-or-better (they already pass cohort/throughput/per_round), then promote all three
to code-level default-on and delete the gates.

### Cross-baseline / shared

- **felix (async_cifar10) may share fluxtune's round-1 cold-start gap** — `_sim_recv_min` uses the same
  reactive gate shape, no fallback for unseen ends; felix's own "empirically inert" comment is UNVERIFIED.
  Out of this session's scope (`async_cifar10/PARITY.md` owns felix).
- felix 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Momentum (S1-S3) / fluxtune server-optimizer retry — roadmap item, not parity
  (`fluxtune_contributions.md` §8.2 / FWDLLM_DESIGN.md); resume only after Phase-1 parity closes.
- Accuracy drop after reaching 81% — known, deferred by operator (`fluxtune_contributions.md` §8).
- **Checker invariants I1-I6** — drafted in chat, not written up; re-derive AFTER the vclock/throughput
  root-cause lands (they hinge on it).

**Tech debt — sim/real in-flight bookkeeping is over-complex; simplify AFTER the timing fix validates.**
The §F.1-23 deadlock was a "too many sources of truth" bug: sim tracks the same virtual in-flight set across
`_sim_pending_commit`, `_sim_inflight_expected`, `_sim_buffer`, `_sim_committed`, `selected_ends`,
`all_selected`, reconciled by `_sim_hold_busy_slots` — one add at the wrong seam desyncs them. Two smells:
(a) those should be ONE authoritative per-end state (`dispatched → returned/buffered → committed`) with the
slot/guard sets DERIVED; (b) `_process_single_trainer_message` means RECEIPT in real but COMMIT in sim — the
exact ambiguity that bit here — so split receipt vs commit responsibilities. Do it as its own scoped step
behind the loop-level characterization tests
(`test_fwdllm_sim_grad_loop.py::TestCommitThenProcessFreesTheSlot`), never bundled with a correctness fix.

**P3 — infra, not parity-blocking:** `_check_gpu_health()` aborts pre-spawn on a broken ordinal (§G) and
`execution.gpu_ids` lets the operator exclude one manually. Still no *automatic* skip-and-remap of a broken
card. Lower priority.

**P3 — `fedbuff_round`'s 07-27 sim run (`run_20260727_105616`) SIGABRTs at process exit, after a clean run.**
`terminate called without an active exception` / `Fatal Python error: Aborted`, immediately after the last
`channel leave done` log line — teardown happens AFTER `max_runtime_s` reached, all telemetry flushed, and
the parity checker read the run cleanly (no truncated data). Looks like a native (CUDA/torch) atexit
teardown crash, not a training-loop bug; `felix_round`'s same-session sim run did NOT crash. Not investigated
further this session — flag if it recurs.

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
a real mix bias. (Mirrors PARITY.md's refl S2 core-identity lesson.)

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
contributed candidate set to draw from. **Tell:** a throughput/`overhead_residual` gap on a round-cadence
baseline but not a same-selector-family iteration-cadence one is consistent with this, but NOT yet confirmed
by telemetry — the pool-size story is a code trace, not a measurement. **Superseded 07-28:** read against
fresh `fedbuff_round`/`felix_round` pairs, `post_close_overhead_wall_s` is NOT near-zero (real 0.46-0.52s vs
sim 0.03-0.04s, ~13-15x) — the pool-size theory doesn't explain the observed gap on its own; §B's 07-28
root-cause entry (per-variance-cycle turnaround, not pool size) supersedes this as the throughput driver.
Kept here as the pool-size mechanism may still matter for `slot_starvation`-style idling, just not this gap.

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
uninstrumented cycles too — here, `post_close_overhead` is gated to `_pk == "weights"` sends only
(`fwdllm_aggregator.py:4103`), leaving the ~90% `VAR=bad` retry cycles — which share the IDENTICAL
`_distribute_weights_async` dispatch loop — with zero coverage; the magnitude match between the two
independently-derived numbers (~0.5s measured vs ~0.6-0.7s implied) is the corroborating signal, not proof
by itself (§B's next step: widen the telemetry gate to confirm directly).

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

- **07-28: `redispatch_decomp` widened to cover `VAR=bad` dispatches (not just fresh `weights` sends) +
  new `vclock_charge` ledger event added.** `_pk in ("weights", "var_bad")` gate + `payload_kind` field
  (`fwdllm_aggregator.py:4135`); `charge_sim_vclock_overhead` (the ONE shared fold-onto-vclock function, used
  by every fwdllm-family baseline) now emits `vclock_charge` on every call in both modes, plus a
  `charge: bool` param so a candidate category (`redispatch_turnaround`, wired in `charge=False`
  measurement-only) can be measured before being charged. Root-cause reconciliation this session (§B/§D-11)
  found the round-cadence throughput gap is a per-cycle cost compounding ~10x, but the OLD telemetry only
  measured the 1-of-10 terminal cycle and had no way to see whether ANY cost was reaching the vclock — this
  closes both gaps at once, generically, not just for this one investigation. 4 new tests, 573 `-k "fwdllm or
  telemetry or parity"` pass. **Not yet read against a live run** (§B, still the open item there).
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
- **07-27: `slot_starvation` telemetry landed** — surfaces a freed dispatch slot with no eligible candidate
  (`feasible_extra < extra`), shared by every async baseline.
- **07-27: all 9 baselines' smoke yamls bumped 1800s→3600s** + real/sim wall-ceiling watchdogs, matching the
  §C run-length bar (enabled the 07-28 full-scale batch, §A).
- **07-27: `[LAG_DECOMP]` + `redispatch_decomp` telemetry landed for fwdllm** — splits redispatch wall gap
  into peer-wait vs post-close overhead, ported from asyncfl's shared version; not yet read against a fresh
  run (§B, still the open item there).
- **07-27: 2 more stale "sim uses in-mem cache" doc claims found + fixed** (`checks.py` docstring,
  PARITY.md, PARITY_CHECKER_README.md) — no cache exists in either mode; real reason is dispatch-cadence /
  aggregator-side overhead.
- **07-27: `staleness` (U3) root-caused, FIXED, VALIDATED — CLOSED** (§D-9). Real's `_agg_pending_commit_ref`
  only covered return-time state, missing a still-training end and flooding it with re-dispatches (one end:
  526 sent vs 284 processed). Fixed via `_PendingCommitUnion` (dispatch-time + return-time, mirroring sim's
  `_sim_pending_commit` span). Confirmed on the post-fix pair: real 0.056/0.064 vs sim 0.095/0.12 (was real≫sim
  7.62/9.10 pre-fix). Unmasked a separate `throughput`/`per_round_advance` gap (§B), not a regression from
  this fix.
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
  var-check pool by `rate`; model-update merge untouched. Validated for cadence effects.
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
  gated behind DEBUG (§F-19). Confirmed effective at scale 07-28 (`fedbuff_it_oracular` 12→4 fails, §B).
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
