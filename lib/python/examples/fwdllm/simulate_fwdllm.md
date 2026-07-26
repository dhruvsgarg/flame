# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 newly-ported
fedbuff/felix-lineage baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2),
then beyond syn_0 (Phase 3). Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf,
sim barrier redesign, delay-factor calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared
parity methodology (ladder, roles/tiers/gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — maintaining this doc
> **Correctness per mode first; parity is the consequence, never the goal.** Real and sim must each be
> independently correct against the configured intent (e.g. `c` trainers training at any instant).
> Aligning the two by making either wrong — or by preserving a defect symmetrically — is a regression
> even when every rung is green. Matched-but-wrong is the hardest failure to find: parity reports it as
> a pass.
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

## §A  Score — refreshed 2026-07-25

**fluxtune/syn_0 — CLEAN** (69/0/16): the three former fails were ONE boundary-race cascade, resolved by
gating index-identity for stochastic-async selectors (→ §D-2, §G). Only the deferred 81% accuracy drop
remains (§B).
**fwdllm / fwdllm_plus/syn_0 — timing family OPEN:** `drain_wall_budget` (GATING) +
`step_timing_breakdown`/`agg_step_timing_breakdown` (DIAG) are ONE co-location-contention family
(root-caused → §D-1); fix-1 `_flat_grad_norm` landed but insufficient, charge-floor-vs-relax decision
open (§B).
**Flag inventory:** `sim_model_agg_compute_time` ON all three; `sim_sct_ordered_drain` +
`sim_model_dispatch_queue` fluxtune-yaml-only (promotion call §B).

**Latest run per baseline** (`run_parity.py`; ✓ pass · ✗ fail · – skip; rung catalog PARITY.md §F):

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260723_044359`/`_064615` (agg_goal=10) | ~7200s | 69/0/16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm/syn_0 | `run_20260723_161459`/`_171648` (agg_goal=10) | ~3600s | 59/3/22 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus/syn_0 | `run_20260723_161647`/`_171829` (agg_goal=10) | ~3600s | 61/2/21 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

STALE — the three run-dir pairs above were cleaned from disk; numbers carried forward per this doc's rule,
not re-verified this session. Re-run via `run_parity.py --baselines <name>` when either is touched.

**FIRST parity pass — 6 newly-ported baselines** (fedbuff/felix lineage, rebuild landed 2026-07-23,
`_metadata/BASELINES.md`), one ~2h real/sim pair each launched post-17:30 the prior day. These had **never**
been parity-checked before — this is the first read, not a re-verify. **ROOT-CAUSED 2026-07-25 from
stored telemetry, no fix landed, implementation blocked on one decision — §B.**

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwdllm_it_unaware/syn_0 | `run_20260724_211542`/`_231751` | ~2h | 57/6/21 | ✓ | ✓ | ✗ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_it/syn_0 | `run_20260724_173702`/`_193925` | ~2h | 67/3/16 | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_round/syn_0 | `run_20260724_201348`/`_212714` | ~2h | 45/17/21 | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| fedbuff_round/syn_0 | `run_20260724_173554`/`_183913` | ~2h | 54/12/18 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_unaware/syn_0 | `run_20260724_185838`/`_194932` | ~2h | 47/17/18 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fedbuff_it_oracular/syn_0 | `run_20260724_173639`/`_182737` | ~2h | 43/21/18 | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json` — NOT on disk
for these six; re-derive from each run dir's `telemetry/*.jsonl` (run dirs are present).

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

**⭐ RESUME HERE — 6 newly-ported baselines: R-A + R-B + R-C ALL FIXED IN CODE (07-25/07-26), AWAITING one
batch of re-runs.** No root is open. All 5 round/+IT baselines plus `fwdllm` need fresh pairs before §A
means anything; operator will run them as a single batch. `async_oort` re-base (decision 4) comes AFTER
the batch — it keeps `felix_it`/`felix_round` a clean control for the R-A/R-B/R-C cohort fixes.

R-B's original framing was WRONG and is corrected below — the round cohort is *supposed* to be pinned per
round; the defect was its SIZE. R-A stands as diagnosed.

| baseline | dispatch path | cadence | version_key guard from | root |
|---|---|---|---|---|
| `fwdllm`, `fwdllm_it_*` | sync gate | mixed | aggregator (`_reselect_true_cache_key`) | R-B (size only) |
| `felix_it` | async gate | iteration | selector (`async_oort` honors it) | timing fails only |
| `fedbuff_it_unaware/oracular` | async gate | iteration | selector — `fedbuff` DROPPED it | R-A |
| `fedbuff_round`, `felix_round` | async gate | round | — | R-B, R-C |

**R-A — `FedBuffSelector` never received the real→sim port. FIXED** by re-basing it onto the new
`AsyncSelectorBase` (§G). `channel.ends()` threads `agg_version_key`/`trainer_version_keys`;
`async_oort` filtered on them, `fedbuff.py::select()` swallowed them in `**kwargs`. Measured
repeat-commits on the same `(data_id, iteration_per_data_id)`:

| baseline | selector | real re-pick | sim re-pick | real staleness tail |
|---|---|---|---|---|
| `felix_it` | async_oort (ported) | 3.5% | 4.0% | {0,1,2,3} |
| `fedbuff_it_unaware` | fedbuff | **34.4%** | **0.9%** | out to **312** |

Real is the divergent side (§F-5): 34% of its 10742 commits are same-version re-picks. Sim's
`_sim_pending_commit` slot-hold masks the gap on one side only. Owns `fedbuff_it_*`'s
staleness/`v1`/`v1b`/`v2`/`v5`/`g2_grad_pool_size`/`convergence` cluster. Every row below is now
inherited from the shared base, not re-implemented:

| construct | async_oort | fedbuff.py (pre-fix) | rung |
|---|---|---|---|
| version_key re-pick filter | yes | absent | R-A cluster |
| `enforce_min_start` | called | never called (`minInitialTrainers:100` inert) | cohort/eligibility |
| `_abandon_clock_now()` (vclock in sim) | yes | bare `time.time()` | latent |
| `send_timeout_wait_s` from config | yes | hardcoded 90 (config 300) | latent |
| `_agg_pending_commit_ref` R1 guard | yes | absent | `r1_inflight_overlap` |
| `task_eligible_states`/`PROP_AVL_STATE` | yes | kwarg set, never read | `avail_composition`, `eligibility` |
| `_handle_recv_state` re-samples | removed (raced dispatch) | still present | latent |
| `_cleanup_recvd_ends` drain | all | capped at `agg_goal` | latent deadlock |
| sampling | `_keyed_topk` (pool-size-independent) | reservoir `randrange(idx)` | RNG desync |

**R-B — the pinned cohort was sized by `agg_goal`, not `c`. FIXED** (§G). NOT a cache-invalidation bug:
`_round` advances once per `total_data_bins` lap by design (`fwdllm_aggregator.py:2325`), and a 2h run
never completes one, so a round-level baseline pinning for the whole run is the SPEC. The real defects,
both in `_select_ends_*_respecting_reselect_gate` and both wrong in real and sim independently:

* **under-fill** — target was `_agg_goal`, so `fedbuff_round` (c=30, agg_goal=10) froze at 10/100
  committers with 20 dispatch slots idle all run.
* **over-fill** — the `>= target` check ran BEFORE the merge and nothing trimmed, so one batch could
  overshoot; final size fell out of arrival timing (`felix_round`: 30 real vs 40 sim).

| baseline | agg_round events | distinct `round` | distinct `data_id` | committers |
|---|---|---|---|---|
| `fedbuff_round` real | 398 | {1} | 46 | 10/100 |
| `fedbuff_round` sim | 753 | {1} | 78 | 10/100 |
| `felix_round` real | 348 | {1} | 64 | 30/100 |
| `felix_round` sim | 1679 | {1,2} | 150 | 40/100 |

`fedbuff_round` froze the identical trainers in both modes, so `cohort_sequence` PASSED on a 10-of-30
run — the matched-but-wrong case the preamble invariant now names. `felix_round` runs the fully-ported
`async_oort` and still failed 17 rungs, which is what proves R-B is not a selector bug.

**Decisions settled 2026-07-25 (all now implemented — see §G):**
1. **Selection cadence is a first-class knob**, `reselect_cadence: round | data_bin | iteration`, boolean
   `reselect_each_iteration` kept as a deprecated alias. `round` = pin for a full lap (fwdllm's intent);
   `data_bin` = re-pin per completed databin, keyed on monotone `_model_version` (§F-2), no baseline uses
   it yet — wired for the cadence ablation; `iteration` = today's default.
2. **Cohort target is `c`, trimmed to exactly `c`** — `c` is the promise ("keep c trainers training"),
   `agg_goal` is only the aggregation trigger.
3. **Extraction reference is `async_oort.py`, NOT `async_random.py`.** `fwdllm_it` uses
   `selector: random` → `RandomSelector` (`random.py`). `AsyncRandomSelector` had **zero** methods of its
   own — all drifted copies — so it collapsed to a `_choose` + docstring; kept registered because
   `fluxtune_dynkc` is parked-not-deleted (`baselines.yaml:862`). `RandomSelector` (sync) is clean only
   because the SYNC aggregator guards for it (§D-6) — it too lacks the version_key filter, the R1 guard,
   and vclock timeouts, and is NOT yet on the shared base.
4. **`async_oort` is not yet re-based** onto `AsyncSelectorBase` — the base was extracted *from* it, but
   its Oort scoring/eval branches need the `_choose`/`_pre_choose`/`_task_extra_eligible` mapping done
   carefully, and `felix_it` currently passes 67/3. Next step; until then `async_oort.py` keeps its own
   ~600-line copy of the mechanism.

Full per-baseline fail list (2026-07-25 `run_parity.py`; also in each pair's JSON):

| Baseline | Fails |
|---|---|
| `fedbuff_round` | overhead_residual, per_round_advance, throughput, avail_composition, eligibility, avail_timebase, selection_detail, drain_wall_budget, agg_step_timing_breakdown, r1_inflight_overlap, terminal_state, total_commits |
| `fedbuff_it_unaware` | trainer_speed, overhead_residual, per_round_advance, throughput, selection_detail, staleness, agg_step_timing_breakdown, cohort_sequence, v1_iter_per_data_id, v1b_iters_moving_avg, v2_var_trajectory, v5_variance_pass_ratio, g2_grad_pool_size, utility, terminal_state, total_commits, convergence |
| `fedbuff_it_oracular` | trainer_speed, trainer_speed_identity, overhead_residual, per_round_advance, throughput, eligibility, selection_detail, training_budget, staleness, drain_wall_budget, agg_step_timing_breakdown, cohort_sequence, v1_iter_per_data_id, v1b_iters_moving_avg, v2_var_trajectory, v5_variance_pass_ratio, g2_grad_pool_size, r1_inflight_overlap, terminal_state, total_commits, convergence |
| `felix_round` | overhead_residual, per_round_advance, throughput, selection_detail, participation, training_budget, gpu_budget_real, staleness, cohort_sequence, v1_iter_per_data_id, v1b_iters_moving_avg, v2_var_trajectory, v5_variance_pass_ratio, g2_grad_pool_size, r1_inflight_overlap, terminal_state, total_commits |
| `felix_it` | throughput, cohort_sequence, v1b_iters_moving_avg |
| `fwdllm_it_unaware` | throughput, step_timing_breakdown, drain_wall_budget, agg_step_timing_breakdown, terminal_state, total_commits |

Root ownership: `felix_it`/`fwdllm_it_unaware` = timing family only (below, no new work). `fedbuff_round`
+ `felix_round` = R-B + R-C. `fedbuff_it_unaware`/`fedbuff_it_oracular` = R-A. Re-run one baseline after
a fix: `cd expt_scripts && python run_parity.py --baselines <name> --yes`. Run dirs are on disk
(`experiments/run_20260724_*`); `_parity_reports/` was not written, so re-derive numbers from the
`telemetry/*.jsonl` as above.

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

1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Never put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`; identity/caching axis is `model_version`.** Updates-per-data_id is the
   dynamic-K output to match, not an input to assume. But `data_id` wraps to 0 every `total_data_bins` lap,
   so it must never key a cache or an identity — use `model_version`, which bumps in lockstep and is
   monotone (§F-21). `_round` is the lap counter, equal to `model_version` only in regular FL
   (async_cifar10); carrying a round-keyed construct over from there without re-deriving the axis is the trap.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold`/`max_iterations_per_data_id` are
   baseline-defining knobs, not parity levers.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** Check whether the real input is the divergent side
   before tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as real-transport artifact (`and not
   self.simulated`) vs algorithmic property. Scope-check before editing shared code — `top_aggregator.py` /
   `_sim_recv_min` can silently break async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only → `pytest tests/mode -k fwdllm`; shared parity engine →
   add async_cifar10 parity tests too; shared stack → full `pytest tests/`.
8. **Telemetry-FIRST, then instrument, then (rarely) run.** Validate/refute from telemetry already on disk
   first. Ship telemetry + plot + pytest in the same change as any new mechanism.
9. **Consult PARITY.md vclock rules before any sim-clock change.** Clock is a monotone `max`; sim skips real
   waits and reconstructs order from sct (`SimReorderBuffer`).
10. **The vclock is virtual wall-time; sim MUST produce speedup (`sim_rate = vclock/wall ≥ 1`).** `sim_rate < 1`
    means sim is stalling on a wait it should skip, or its commit throughput can't keep pace with arrivals.
11. **Correctness before speed; shared roots before per-baseline.** A bug failing rungs across ≥2 baselines
    outranks a single-baseline one.
12. **Logical determinism is the parity definition.** Same trainers selected, same order of update receipt,
    same aggregations/rollbacks — differing ONLY in wall-clock. Prove it on the first data bin before extending.
13. **Do the right thing — no hacks.** A hack that moves a number without a correct mechanism is a regression
    in disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. No bare-scalar shortcut.
15. **Verify claims against code, not documentation/comments.** A docstring claiming two functions are
    equivalent is a statement of intent, not a guarantee — diff them.
16. **Contention at scale → §D-1.** (Was "never blame GPU/resource contention at n=10" — refuted once (§E),
    held below ~100 trainers; now a genuine root at n=100. Full pattern + tells in §D-1.)
17. **A bounded rotating in-flight cohort settling at `c − agg_goal` surplus is the correct steady state** for a
    `c ≫ agg_goal` fedbuff pool, not a backlog to eliminate — don't drive `carried_surplus_commits` toward 0.
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** A configurable/correctness-path
    value (seed, delay floor, agg_goal, c, trace, flag state) must surface identically across yaml, snapshot,
    and both roles' telemetry — divergent/missing logging wastes sessions chasing phantoms.
19. **No compute on the critical path for a log the run doesn't need.** Any log with non-trivial arguments
    (`.item()`, hashing, `torch.allclose`/`.norm()`) MUST be gated behind `logger.isEnabledFor(logging.DEBUG)` —
    an f-string evaluates its args even when the level would drop the line.
20. **When real and sim disagree on timing, optimize REAL toward determinism — never inject noise into sim.**
    Sim's per-speed-class modeled duration must stay clean/reproducible (that's what makes `cohort_sequence`
    checkable at all). Fix real's measured completion time at its source instead.

### §F.1 Version & commit invariants (confirmed real+sim in code + real logs, 2026-07-21)

21. **`model_version` bumps once per COMPLETED data-bin** (variance PASS → global model update, `+= 1` at the
    data_id advance) — constant across all iterations of one data-bin. `iteration_per_data_id` bumps on every
    variance-FAIL retry, resets to 0 on data-bin advance. `version_key = (model_version, iteration_per_data_id)`
    therefore changes EVERY iteration and is the sole step identity (§F-14).
22. **Commit == the update being used for aggregation, and it happens at that instant — no lag.** Real: on
    ordered arrival (trainer already waited). Sim: when the vclock reaches the update's `sct` (buffer-unlock IS
    the commit). An update drained for aggregation must be committed in the same step, never on a later
    event/aggregation.
23. **Commit partially unlocks the trainer: it frees the compute slot immediately, but a version_key re-pick
    guard (`_trainer_state_dict`) keeps it un-pickable for the SAME `(model_version, iteration)`.** It re-enters
    the pool once the version_key advances (next iteration or next data-bin). Sim's slot-hold
    (`_sim_pending_commit`) must be cleared at commit — never re-added after — or the slot never frees and
    re-dispatch across variance-retry iterations starves.
24. **Within a data-bin the global weights are constant; a re-picked trainer gets a RETRY, not a re-send.** The
    aggregator sends full WEIGHTS to a trainer only for a `model_version` it has not yet received this data-bin
    (`_weights_sent_this_cycle`, cleared on the `model_version` bump). A re-pick at the same `model_version`
    (still on this data-bin) is told VAR=bad — recompute new perturbations — never a redundant weight re-send.

### §F.2 Porting a SELECTOR ≠ porting TIMING parity → §D-3

Moved to §D-3 (it's a diagnostic pattern, not an invariant). Kept here as a stub because prior sessions cite
"§F.2" — the class-hierarchy detail and the "diff destination aggregator/trainer against the shared base"
rule now live in §D-3.

---

## §G  Landed fixes — recent, load-bearing for current work only. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, ≤30 words, immediately.** The instant a rung flips or a hypothesis resolves, write
> ONE line (mechanism + outcome) and delete it from §A/§B in the same edit.

- **R-C round-cache stuck timeout stamped/checked on wall in sim** (07-26) — new
  `_round_cache_clock_now()` (vclock in sim, wall in real, #1c); sim evicted 0 stuck ends vs real's 6.
- **R-B cohort sized by `agg_goal` not `c`, and never trimmed** (07-25) — `_round_cohort_target`/
  `_trim_round_cohort` on both gates; fixes 10-of-30 under-fill and the 30-vs-40 overshoot race.
- **`reselect_cadence` knob added** (07-25) — round/data_bin/iteration; boolean kept as alias, every
  shipped baseline byte-identical; `data_bin` keyed on monotone `_model_version`, unused so far.
- **R-A: `FedBuffSelector` re-based onto new `AsyncSelectorBase`** (07-25) — 832→54 lines; inherits
  version_key guard, R1 guard, vclock timeout, avl filter, full drain, `_keyed_topk`.
- **`AsyncRandomSelector` collapsed 850→31 lines** (07-25) — had zero methods of its own; stays
  registered for parked `fluxtune_dynkc`.
- **Selector stats block de-duplicated ×4 into `AbstractSelector`** (07-25) — `record_selection_stats`/
  `maybe_log_stat_summary`; found `RandomSelector` never recorded at all.
- **Heartbeat mechanism deleted** (07-25) — no sender existed, tags never in `func_tag_map`, no baseline
  used `type: HEARTBEAT`; removed across 10 files incl. `VAL_CH_STATE_HTBT_*`, `MessageType.HEARTBEAT`.
- **fluxtune 3→0 (69/0/16): all fails were ONE boundary-race cascade, not a sim bug** (07-23) — marginal cohort
  slot is a physical-FIFO vs modeled-sct near-tie; index overlap 0.239 = independent-draw floor 0.237; every
  marginal criterion matches (S2/utility-dist/count/v1/v2/speed). Checker: gate index-identity for stochastic-async
  (`cohort_sequence.composition`+first-bin, `trainer_speed_identity.utility`, `v1b` MA-shadow) → diagnostic; count/
  cum_mean_rel/speed_s enforced; added `independent_draw_floor` diagnostic. 204 tests. sct-order lever rejected (§E).
- **fwdllm timing family: co-location contention, NOT over-compute** (07-23) — input sizes byte-identical, sim
  drain floor p10 35ms = real 37ms every decile, thread-local cpu tracks wall. Only `drain_wall_budget` gates;
  step-timing checks are DIAG. `_flat_grad_norm` per-parameter GPU sync → single on-device reduce (bit-identical,
  55 tests). Re-measured: fix-1 insufficient alone, vclock-charge gap persists (§B / §D-1).
- **fwdllm/fwdllm_plus throughput CLOSED, validated at 7200s** (07-23) — recv_fifo→`drain_ready` + var_bad
  dedup/`pause_execution` removal held: fwdllm 3.2%, fwdllm_plus mw 4.8%, both PASS. Was the sync gap.
- **fluxtune `v2_var_trajectory` + `drain_wall_budget` PASS under logical-N** (07-23) — v2 real 0.921/sim 0.919
  (0.3%); fluxtune drain_tail real 0.434/sim 0.69 in-band. fluxtune 5→3 fails; both dropped from the cascade.
- **`cohort_sequence.count` PASSES on the matched logical budget** (07-23) — real 1750/sim 1833 rel 4.5% <5%.
  (`composition` since resolved as a boundary-race cascade → gated, top of §G.)
- **fwdllm `per_round_advance` PASSES via central-tendency escape** (07-23) — mean_rel 3.2% (tol 15%), KS 0.345
  tolerated by the `pctl_band_ok` central escape. Round-1 tail no longer trips it.
- **Checker: `matched_virtual_budget` deleted → grade on the LOGICAL budget N** (07-23) — `V=min(sim
  vclock, real wall)` conflated the two clocks (the axis `sim_rate` tests). New `_matched_logical_budget`
  (progress ≤ N); U2/K8 reshaped count→**time-to-N**, v2/utility/`cohort_sequence.count` swapped to
  N-truncation, `cohort_sequence` deps V1. 666 tests pass. Design: PARITY.md §1.5. (§D-4)
- **Checker: `pctl_band_ok` DIST-band escape landed + tested** (07-23) — `_step_timing_compare`,
  `drain_tail_s`, `per_round_advance` central-tendency escape. 24 new tests; README (`parity/PARITY_CHECKER_README.md`).
- **Checker: `_step_timing_compare`'s `band_min_abs_s` (0.5s) silently passed 5x step-timing regressions**
  (07-23) — 10-100ms-scale functions vs a 500ms floor copied from drain's ~1s scale; anchored to the
  metric's own noise constant (`_STEP_TIMING_NEAR_ZERO_ABS_DIFF_S`=3e-4s) instead.
- **fluxtune 19→5 validated at 7200s** (07-22) — selector rebind + `_keyed_topk` + sim-deadlock fix (§F.1-23)
  + dispatch-queue/commit-fold vclock charging all held; `preferred_duration`/`terminal_state`/`total_commits`
  now PASS. Remaining fails all downstream of a residual ~9% vclock under-charge.
- **Sim deadlock: `_process_single_trainer_message`'s `else: _sim_pending_commit.add` re-pinned committed
  trainers** (07-21 pm) — dropped the commit-time add; dispatch-time add + version_key guard already cover
  re-pick. Cleared `sim_rate` 0.07 stall.
- **Startup crashes (unhealthy GPU 0 + unconditional CUDA RNG init)** (07-21) — device-gated `torch_cuda_rng`,
  `_check_gpu_health()` preflight allocation, `execution.gpu_ids` ordinal allowlist; 7200s pairs ran clean.
- **`cohort_sequence` grading made distributional (set-overlap ≥0.8)** (07-21) — absorbs boundary-race
  cascades; fluxtune now fails it only via the upstream `data_id` drift, not the grade.
- **`[SELECT_TRACE]` debug logging removed** (07-22) — divergence localized, verbose tracing deleted.
- **GPU kernel pre-warm landed & ran** (07-21) — `_warmup_gpu_kernels` at trainer startup (trainer only, not
  agg eval model); round-1 tail still visible in fwdllm `per_round_advance`, tracked in §B.
- **Real's `_agg_pending_commit_ref` was unbound, letting a buffered-but-uncommitted trainer get re-dispatched**
  (07-21) — bound to `_per_agg_trainer_list` (real-only), mirroring sim's own `_sim_pending_commit` binding;
  validated at 7200s (19→5, line above).
- **`_release_end_on_return` held real's slot to full-cohort commit, causing sawtooth (not flat) concurrency**
  (07-21 am) — `buffered=True` releases as soon as P0-1 has safely buffered the contribution. Exposed the
  asymmetry above; superseded by it, not reverted.
- **fwdllm trainer's remainder-wait sleep only compensated `gpu_time_s`, not real's total elapsed overhead**
  (07-20 pm-10) — now sleeps against elapsed-since-dispatch (`_wall_recv_ts`), closing an avoidable real-side
  noise source (§F-20).
- **`_compute_var`'s stop-the-world GC pause hypothesis REFUTED** (07-20 pm-11) — new `gc_pause_s` telemetry
  shows ~0ms GC time both sides. (§E)
- **`_distribute_weights_sync` missing from the real-only timing exemption set** (07-20 pm-11) — added,
  matching its already-exempted async twin.
- **`v2_var_trajectory`/`utility`/`throughput`/`per_round_advance` false-failed on population-length, not a
  real gap** (07-20 pm-5/pm-7) — gated on matched VIRTUAL BUDGET (each event's own commit timestamp filtered to
  `<= V`), not index-count or raw population. Superseded by the logical-N regrade (§D-4).
- **fluxtune's `pacer()` fired once per `select()` call instead of once per round**, ratcheting
  `round_threshold` to max and disabling the speed penalty (07-20 am) — fixed with an explicit
  `_last_pacer_round` guard, closing `preferred_duration`'s gap.
- **fluxtune `sim_send_ts` was the EOT/shutdown broadcast skipping the stamp by design, not a mid-run gap**
  (07-20 pm-2) — now stamped unconditionally.
- **fluxtune's `selection_train.vclock_now` was never stamped in sim**, blinding sim-side selection
  diagnostics without affecting the real side (07-20 pm-6) — fixed.
- **P0-1: buffer each trainer's contribution on receipt, merge into `self.grad` in canonical (D, trainer_id)
  order at commit** (07-18) — the deferred-merge foundation this session's fixes build on.
