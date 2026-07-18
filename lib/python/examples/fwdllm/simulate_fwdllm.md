# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity ONLY**, for the **fluxtune / fwdllm / fwdllm_plus** baselines — at **100%
availability (syn_0)** first (Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3). Everything
else about the fwdllm build — how it differs structurally from async_cifar10, the baseline matrix, the phased
roadmap, the JVP compute/perf profile, the sim receive/barrier redesign, the NPU delay-factor calibration, and
open (non-parity) design decisions — lives in **[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md)**. This doc mirrors
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md)'s focus and discipline, scoped to fwdllm; that doc owns
the shared parity methodology (ladder, roles/tiers/gating, run-length budget, landed sim mechanisms) and
fwdllm's rung catalog (§F) — read it first if you're new to this track.

> ## PREAMBLE — how to maintain this doc (READ BEFORE EDITING)
> **Parity only.** If what you're adding is a design decision, build-plan step, roadmap item, performance
> optimization, or calibration derivation rather than a real↔sim parity finding or fix, it belongs in
> [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md), not here — don't let non-parity content creep back in.
>
> This is a **living status doc**, not a changelog. §A/§B describe the state **right now** — rewrite in place,
> never stack dated "UPDATE" blocks. Per-run history lives in git + the parity JSONs; the code is the source of
> truth for *what* a mechanism is.
>
> **Score-tracking trigger.** Whenever `run_parity.py` is run over a real/sim pair with wall duration >3600s
> (1h) for one or more baselines, refresh §A's scoreboard in the SAME edit — for EVERY baseline, not just the
> one(s) freshly run (carry the others' last-known numbers forward, tagged STALE, rather than leaving them
> silently outdated).
>
> **§A is a scoreboard, never prose:** per-baseline pass/fail/skip + key-rung ✓/✗, nothing else. **§B is
> next-steps/open-issues, per baseline** — short (1-3 line) entries only, no essays. **§G is CLOSED items,
> ONE LINE each, under ~20 words** (problem → fix, terse) — no paragraphs, no trace dumps, no multi-sentence
> justification; that reasoning belongs in the code comment/commit that landed the fix, not here. **An issue
> lives in EXACTLY ONE place: open (§B) xor closed (§G, one line).** Never both, never neither, never repeated
> across sections in different states (open in one place, closed in another, "deferred" in a third) — when you
> close something, DELETE its §B entry as part of the same edit and add the §G one-liner; don't leave a stale
> copy anywhere. When a chain of hypotheses gets superseded, keep only the FINAL correct one — no wrong turns,
> no "superseded" sections.
>
> **Working checklist for every fix:** (a) ground every claim in a metric actually captured and diffable —
> telemetry/banked logs first, logical-determinism traces over aggregate curve-matching; (b) isolate the true
> bottleneck, not its symptom — verify claims against code, not against what a docstring/comment says it does;
> (c) design fixes from first principles at the root, no hack that moves a number without a correct mechanism;
> (d) **never launch an experiment run directly** — print the exact command and let the operator run it. Code
> edits, telemetry reads of already-banked logs, and pytest are fine unattended; (e) **always use conda env
> `dg_flame`** for any python/pytest/analyze_run.py invocation in this repo — running in the wrong env (e.g.
> `base`) silently skips deps (`sortedcontainers`, etc.) and produces misleading collection errors, not a real
> signal; (f) **new debugging telemetry ships with its plot in the same change** — a `build_*`/`emit()` field
> added without a reader in `scripts/analysis/analyze_run.py` is dark data (2026-07-13 audit found several
> rounds' worth of already-emitted phase/residence/comm telemetry with zero plots). Reuse the existing plot
> style for that data's shape (binned_line over progress for a per-round series, cdf_multi for a distribution,
> bar_plot for a per-category summary — see `scripts/analysis/plot_helpers.py`); only introduce a new plot
> shape if the telemetry is a genuinely new kind of quantity nothing existing already renders.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — parity methodology (ladder,
roles/tiers/gating, run-length budget, landed sim mechanisms); fwdllm's rung catalog is PARITY.md §F.
[async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) — the availability substrate
fwdllm inherits via its aggregator class chain. [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md) — build plan, structural
deltas, roadmap, performance work, calibration. [fluxtune_contributions.md](fluxtune_contributions.md) §8 — the
LIVE ledger for fluxtune training-stability/convergence issues (oscillation, collapse, accuracy degradation);
check it BEFORE opening a new stability investigation here (§G, 07-14 session 7).

**Comparator — discovers the latest real/sim pair per baseline and runs the shared parity battery:**
```bash
cd lib/python/examples/fwdllm/expt_scripts
python run_parity.py                       # all 3 baselines, latest pairs, confirm
python run_parity.py --baselines fluxtune   # one baseline
python run_parity.py --yes                  # skip the confirm prompt
python run_parity.py --validate             # + live-run checks (staleness/vclock_now)
```
Rung catalog: PARITY.md §F. **Not redefined there:** per-stage wall-budget instrumentation
(`drain_wall_budget`, `trainer_phase_wall_budget`, `step_timing_breakdown`, `aggregation_compute_wall`) is
ONE-SIDED (`sim<=real`) where sim should collapse a real-transport phase to ~0, DISTRIBUTIONAL where it's
genuine shared compute.

---

## §A  Score — refreshed 2026-07-17 (see PREAMBLE's score-tracking trigger)

> **STALE re: `cohort_sequence`/fluxtune (07-17d):** the `var_match_frac`/`set_match_frac` numbers below for
> fluxtune predate the checker fix (§G 07-17d) — fluxtune's part of "the shared `var_match_frac` gate gap"
> callout is now CLOSED (confirmed checker blind spot + legitimate stochastic noise, not a shared root with
> fwdllm/fwdllm_plus). fwdllm/fwdllm_plus's `var_match_frac` gap is UNCHANGED and still open, but is a
> DIFFERENT mechanism (their SET/ORDER/CADENCE are already EXACT, unlike fluxtune's) — see cross-baseline §B.
> This table itself isn't rewritten (no >3600s run landed this session, only a ~900s validation pair) — the
> next >3600s run over any baseline should refresh it per the trigger below.
>
> **Fresh 5400s (90-min) pairs for ALL THREE baselines** — longest parity pairs run to date (up from ~3600s),
> same config as 07-16 (fluxtune divisor 0.48/floor 4.0; fwdllm/fwdllm_plus divisor 1.63/floor 11.0; all three
> now min-init=**N=100**, agg_goal=10). This session's deep-dive is fluxtune-only (operator focus); fwdllm/
> fwdllm_plus fails are listed but untriaged. **Headline cross-baseline finding**: the `minInitialTrainers=
> N=100` join-race/deadlock fix (§G 07-17/07-17b) now VALIDATES for fwdllm/fwdllm_plus too, not just fluxtune —
> `cohort_sequence` SET+ORDER+CADENCE are EXACT for both sync baselines over the full run (fwdllm 132/132
> cycles, fwdllm_plus 259/259).

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260717_121517`/`_134732` (delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10) | ~5400s | 61 | 6 | 18 |
| fwdllm/syn_0 | `run_20260717_121511`/`_134658` (delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~5400s | 49 | 13 | 22 |
| fwdllm_plus/syn_0 | `run_20260717_121456`/`_134702` (delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~5400s | 55 | 8 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fwdllm | ✗ | ✓ | ✗ | ✗ | ✗ | – | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✗ | ✓ | ✗ | ✓ | ✓ | – | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |

**All failing rungs, this run:**
- **fluxtune** (6, up from 5 at 1h): `cohort_sequence` — SET still diverges at cycle_index 2 of 733 compared
  (cycles 0-1 exact, same onset as last session) but now measured over the full 90-min run: overlap decays to
  `set_match_frac=0.003`, deeper than the 1h pair's partial view. `v2_var_trajectory` (mean_rel_diff 0.0536 vs
  0.02 tol, up from 0.0229 at 1h) and `v1b_iters_moving_avg` (ma_max_abs_dev 2.25 vs 0.75 tol, up from 0.85) —
  both got WORSE roughly in proportion to run length, confirming last session's hypothesis that they're
  downstream of the cohort cascade, not independent bugs. `convergence` (acc_diff 0.0509 vs 0.05, essentially
  unchanged hairline miss). `agg_step_timing_breakdown` unchanged root (`_process_aggregation_goal_met` KS
  0.341 sim 20% slower, `aggregate` KS 0.376 sim 28% slower — same queue-bound gap as §G, gap narrowing
  slightly vs the 1h pair's 28%/32%). **NEW**: `step_timing_breakdown` — `tb_prepare_perturbation` KS 0.251
  (tol 0.25, hairline) / mean_rel 36%, but real 1.9ms vs sim 3.0ms (both near the 1ms degenerate-noise floor,
  just above the p99 skip threshold at this sample count n=7362/8340). Likely quantization noise surfaced by
  more samples, not a new mechanism — not yet triaged, don't over-index on it.
- **fwdllm** (13, up from 7 at 1h): cohort SET/ORDER/CADENCE now EXACT (validates §G join-race+deadlock fix,
  see above) — `cohort_sequence` itself still fails, on `var_match_frac=0.25` (cross-baseline, see §B).
  `utility` now PASSES (was failing at 1h). New fails vs the 1h pair: `v1_iter_per_data_id`,
  `v1b_iters_moving_avg`, `v2_var_trajectory`, `v5_variance_pass_ratio`, `g2_grad_pool_size`, `terminal_state`,
  `total_commits` (n_sim=51 vs n_real=58 commits, 12% over tol) — a growing commits/rounds gap
  (`throughput` rel_diff 0.199) layered on the pre-existing, already root-caused `overhead_residual`/
  `per_round_advance` (real-only `num_min_req=1` clamp, §B item 2) that likely compounds over a longer run.
  `step_timing_breakdown`, `agg_step_timing_breakdown` still fail, UNEXAMINED. Untriaged this session (fluxtune
  was focus).
- **fwdllm_plus** (8, up from 7 at 1h): cohort SET/ORDER/CADENCE now EXACT (same validation as fwdllm) —
  `cohort_sequence` fails on `var_match_frac=0.5` (cross-baseline, see §B). `eligibility` and
  `v1b_iters_moving_avg` no longer fail (were failing at 1h). New fails: `trainer_speed`, `training_budget`.
  `convergence` (acc_diff 0.0598 vs 0.05, similar to before). `step_timing_breakdown`,
  `agg_step_timing_breakdown` still fail, UNEXAMINED. Untriaged this session.

See §B for what's actively being worked per baseline; see §G for what's already closed.

---

## §B  Next steps / open issues — per baseline, as of the §A runs above

### fluxtune (~5400s, delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10)
1. **`agg_step_timing_breakdown`: aggregator gap — LOCATED, not yet ROOT-CAUSED.** `aggregate()` is now
   fully sub-decomposed (7 timed sub-blocks, ~91% of its wall time directly attributed, n=40/max-data-id=3
   pair `run_20260718_003750`/`_004148`); GPU-sharing and pool-size/workload-scaling are refuted as the
   mechanism, but **every block that DOES carry the gap is plain CPU/memory-bound tensor work** (deepcopy,
   a weighted-sum loop, tensor clone/detach, `torch.stack`+var) with no mode-dependent branching of its
   own — which reopens ambient-contention as the explanation, just not the form already tested. Gap
   attribution (sum over 12 calls, sim 1686.6ms vs real 1204.7ms, total gap 481.9ms / 40.2ms per call):
   | sub-block | what it is | % of total gap |
   |---|---|---|
   | `_snapshot_retry_cache` (own cost) | `copy.deepcopy(model_dict)` + `copy.deepcopy(get_global_model_params())` | 36.7% |
   | `_apply_weighted_update` | FedAvg weighted-sum double-loop + per-param server update | 25.0% |
   | `_compute_var` | `calculate_var` (`torch.stack`+mean+var) | 18.3% (single-call outlier in both legs — see below) |
   | residual (unattributed) | dispatch logic + un-gated `logger.info` lines | 9.2%, concentrated in data_id=0's warmup calls, ~EQUAL real vs sim (not part of the real/sim gap) |
   | `_snapshot_last_round_update` | `[p.clone().detach() for p in weighted_gradient_sum]` | 7.5% |
   | `get_global_model_params` (both call sites) | `.cpu().state_dict()` | 3.2%, and BOTH calls measure only ~2ms each — NOT the earlier-suspected big driver, ruled out by direct measurement |
   | `_prepare_round_state`/`_accumulate_retry_cache`/`_cache_grad_for_retry` | preamble, retry-cache merge, rollback cache-append | ~0% |
   `_compute_var`'s 18.3% is mostly ONE anomalous call in EACH leg (real 50ms / sim 120ms, both on the
   same logical position, data_id boundary) against an otherwise flat 4-9ms baseline both sides — an
   outlier at this sample size (n=12), not a systematic mechanism; re-check at higher n before trusting it.
   **Self-correction on the CPU-contention hypothesis (refuted 07-17f, reopened 07-18d):** the earlier
   refutation showed the gap is a pool-size-INDEPENDENT fixed intercept, not a workload-scaling slope, and
   concluded from that "not contention." That conclusion doesn't follow — ambient contention (other
   processes competing for CPU/memory bandwidth/GIL) is BY DEFINITION independent of this call's own
   workload size, so a pool-size-independent intercept is equally consistent with contention as with any
   other fixed-cost explanation; the regression only ruled out *workload-scaling* contention, not ambient
   contention generally. However, this run's n=40 doesn't even oversubscribe the host's 96 cores (41
   processes total), so simple core-count contention (the mechanism assumed for the original 100-trainer/
   96-core numbers) doesn't apply either — if it's contention at all, it's a subtler form: memory-bandwidth/
   cache/NUMA pressure from 40 concurrently-running trainer processes (deepcopy of a ~267MB model is
   memory-bandwidth-bound, not core-bound).
   **Two candidate mechanisms RULED OUT by direct code inspection 2026-07-18e (not indirect regression):**
   (1) OS core-scheduling contention — `flame/launch/runner.py:172-190` + `aggregator_spawner.py:88-109`
   already reserve dedicated cores for the aggregator (8/96) with matching `OMP_NUM_THREADS` etc., excluded
   from trainer pinning, identically in both real and sim launches — the aggregator is never time-sliced
   with trainers at the OS level in either mode, so this can't be the differential. (2) A sim-only
   background thread stealing GIL time inside the aggregator's own process — `flame/sim/virtual_clock.py`
   (vclock + reorder buffer) is plain synchronous dict/heap code, no `threading.Thread`; the only
   background thread anywhere (`eval_model`) exists in both modes and is already exempted from this rung.
   **CONFIRMED 2026-07-18f**: n=15-vs-n=40 A/B (`run_20260718_005859`/`_010529` vs the n=40 pair above) —
   cutting trainers 40→15 shrank `_apply_weighted_update`'s relative gap +38%→+12.5% (>3× less, 40.1ms→
   11.9ms per call) and `_snapshot_retry_cache`'s +31%→+23% (14.8ms→6.3ms per call), roughly proportional
   to the trainer-count cut — real, causal evidence for ambient (memory-bandwidth/cache) contention from
   sim's continuously-active trainers on these two specific memory-heavy blocks. This also explains why
   n=40 wasn't smaller than the original n=100 numbers: bandwidth contention saturates past some trainer
   count, so 100→40 stayed above threshold while 40→15 crossed below it. `_compute_var`'s gap did NOT
   track trainer count (two isolated sim-only outlier calls, unrelated mechanism, likely GC/scheduler
   jitter at this sample size — re-check at higher n).
   **FIXED 2026-07-18g, two of the three mitigations discussed in §B, landed together:**
   (1) *NUMA isolation* (`flame/launch/runner.py` + `spawner.py`): CPU-partition logic was reserving
   "first 8 core IDs" for the aggregator with no topology awareness, so trainer memory traffic could
   still share its NUMA node. Added `_read_numa_nodes()` (parses `/sys/devices/system/node`); on a
   ≥2-node host (confirmed: operator's box has 2, 64 cores/node), trainers now PREFER the
   aggregator-free node(s) and only spill onto the aggregator's node's remaining cores as overflow
   once that preference is exhausted — **not** a blanket node exclusion, which was tried first and
   reverted (2026-07-18h) after the operator flagged it would force >1 trainer/core at their real
   target scale (n=100 > 64-core node ⇒ 36 trainers doubling up, reintroducing the exact contention
   this exists to prevent). Verified via simulation: n≤64 trainers get full isolation, n=100 still
   gets exactly 1 dedicated core/trainer (64 isolated + 36 overflow onto the agg node's spare 56
   cores), core-sharing only starts past n=120 (same ceiling the pre-NUMA-aware code always had).
   Shared launcher code (affects async_cifar10 too); 124/124 `tests/launch` pass. (2) *Reduce the
   aggregator's own footprint* (`FedSgdAggregator.py`): `_snapshot_retry_cache` deepcopied TWO things
   every call — `self.model_dict` (used) and `get_global_model_params()` (dead: its only reader was a
   commented-out `set_global_model_params` call). Removed the dead deepcopy — halves this call's
   memory-copy volume, unconditionally, both modes; 275/275 fwdllm tests pass. (3) *Widen the rung's
   tolerance* — NOT implemented, kept as fallback per operator if 1+2 don't close the gap enough.
   Next: rerun the same n=15 or n=40 pair with both fixes live and re-measure
   `_snapshot_retry_cache`/`_apply_weighted_update`'s gap.
2. `_distribute_weights_async` still exempted (`gates_ok=False`, real-only sleep) — unrelated, unaffected by
   the above.
3. `v2_var_trajectory` (mean_rel_diff 0.0536 vs 0.02 tol, up from 0.0229 at 1h), `v1b_iters_moving_avg`
   (ma_max_abs_dev 2.25 vs 0.75 tol, up from 0.85), `convergence` (acc_diff 0.0509 vs 0.05, ~flat) — this
   5400s pair predates the `cohort_sequence` checker fix (§G 07-17d); re-measure on a fresh pair before
   treating any of these as independent bugs (they were downstream of the now-explained SET cascade/tie noise).
4. **NEW, marginal — `step_timing_breakdown` / `tb_prepare_perturbation`**: KS 0.251 (tol 0.25) / mean_rel 36%,
   but real 1.9ms vs sim 3.0ms — both near the 1ms degenerate-noise floor, likely surfaced only by the larger
   n=7362/8340 sample count at 5400s. Not yet triaged; check whether it's noise before spending time on it.
5. `sim_sct_ordered_drain` A/B — unblocked. Run `fluxtune_n10_smoke_sim_no_sct_drain.yaml` against next pair.
6. **Accuracy drop after reaching 81%** — known, deferred by operator (07-15). Not yet triaged.
7. **Real↔real admissibility (§F-5)** — rungs now finalized (tie-window + tiered dep graph, `CHECK_META`
   `deps`); unblocked, ready to resume (the `cohort_sequence` admission investigation that deferred it is
   closed, §G 07-17d).

### fwdllm (~5400s, delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10)
> 13 fails (up from 7 at the 1h pair — longer run surfaces more downstream drift). None individually triaged
> this session (fluxtune was the operator's stated focus) — next session's queue:
1. **`cohort_sequence` — SET/ORDER/CADENCE now VALIDATED exact (set_match_frac 1.0 over all 132 cycles,
   confirms §G 07-17/07-17b's join-race+deadlock fix transfers to the SYNC full-cohort barrier too).** The
   rung itself still fails, now purely on `var_match_frac=0.25` — same cross-baseline gate gap as fluxtune #1;
   don't re-investigate the join-race here, track the fix in the cross-baseline item.
2. `overhead_residual`/`per_round_advance` — previously root-caused (real-only `num_min_req=1` clamp calls the
   sync collect path once per LAP, not per cycle) but unfixed; needs a compose-loop refactor, risks stranding
   messages if done blind. On this fresh pair the gap has grown with run length: `total_commits` now also
   fails (51 sim vs 58 real, 12% over tol) and `throughput` rel_diff is 0.199 — re-check the root cause still
   holds before refactoring.
3. `throughput`, `step_timing_breakdown`, `agg_step_timing_breakdown`, `terminal_state` — failing, UNEXAMINED,
   likely downstream of #2's compounding gap given the pattern.
4. **NEW this run**: `v1_iter_per_data_id`, `v1b_iters_moving_avg`, `v2_var_trajectory`,
   `v5_variance_pass_ratio`, `g2_grad_pool_size` — all UNEXAMINED; check whether they're downstream of #1's
   `var_match_frac` gap before treating as independent. `utility` now PASSES (was failing at 1h).

### fwdllm_plus (~5400s, delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10)
> 8 fails (up from 7 at the 1h pair). None individually triaged this session — next session's queue:
1. **`cohort_sequence` — SET/ORDER/CADENCE now VALIDATED exact (set_match_frac 1.0 over all 259 cycles),
   same validation as fwdllm #1.** Rung still fails, now purely on `var_match_frac=0.5` — cross-baseline gate
   gap, track the fix in the cross-baseline item, not here.
2. `eligibility` and `v1b_iters_moving_avg` no longer fail (were failing at 1h) — no action needed.
3. **NEW this run**: `trainer_speed`, `training_budget` — UNEXAMINED.
4. `step_timing_breakdown`, `agg_step_timing_breakdown`, `convergence` (acc_diff 0.0598 vs 0.05) — still
   failing, UNEXAMINED.

### Cross-baseline / shared
- **fwdllm/fwdllm_plus `cohort_sequence` fails on `var_match_frac` (0.25 / 0.5) with SET/ORDER/CADENCE EXACT —
  now the highest-priority open item, and a DIFFERENT mechanism from fluxtune's (closed, §G 07-17d).** The
  5400s pairs (07-17, `_134658`/`_134702`) show SET+ORDER+CADENCE EXACT for both sync baselines over their
  full runs (fwdllm 132/132 cycles, fwdllm_plus 259/259) — this validates the `minInitialTrainers=N=100`
  join-race+deadlock fix (§G 07-17/07-17b) transfers cleanly beyond fluxtune (next bullet). Because SET is
  EXACT here (not tie-resolved), fluxtune's checker fix doesn't apply — a same-cohort/same-order real-vs-sim
  var mismatch with IDENTICAL membership needs its own root cause (real per-trainer JVP/perturbation-sampling
  numerics vs sim, or the GPU fp16-jitter floor already documented for CADENCE/VAR beyond ~bin 6 — check which
  before assuming a bug). Not yet investigated this session (fluxtune was focus).
- **`minInitialTrainers=c` (not N) reopened the post-barrier join-order race for ALL THREE baselines —
  ROOT-CAUSED+FIXED, §G 07-17.** Sim's cycle cadence legitimately outruns real's (transport-collapse), so
  identical wall-clock-bound trainer spawn timing lands in different cycles per mode — no algorithmic bug.
  Fixed in the yamls `run_sequential.sh`'s `BASE_YAML_MAP` actually generates from: `fwdllm_n100_smoke[_sim]`,
  `fwdllm_plus_n100_smoke[_sim]`, and (despite the name) `fluxtune_n10_smoke[_sim].yaml` — NOT
  `fluxtune_n100_smoke_4h.yaml`, which the first pass mistakenly targeted and which this pipeline never reads.
  VALIDATED on all three at 5400s scale (see above) — no longer suspect for any remaining gap.
- **felix (async_cifar10) likely has the same round-1 cold-start gap fluxtune had** — `asyncfl/top_aggregator.py`
  `_sim_recv_min` uses the identical `_sim_inflight_expected`/reactive-`_sim_known_delay_s` gate shape (no
  fallback for unseen ends), same theoretical blind spot on first contact. Felix's own code comment claims the
  gate is empirically "inert" there (real GPU compute ~0.4s wall, felix's `_SIM_RECV_MARGIN_S=0.5s` fallback
  probe already wide enough to catch it) — plausible but UNVERIFIED, not data-checked this session. See
  `simulate_fwdllm.md` §G 07-16 / `fwdllm_aggregator.py`'s `unknown_stuck` gate for the fix pattern if felix's
  own data later shows it's not actually inert. Not implemented for felix — flagged only, per operator (out of
  this session's blast radius: `async_cifar10/PARITY.md` owns felix).
- felix (async_cifar10) 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Operator-run seeded real↔real pairs (`*_seeded.yaml`) — GPU-nondeterminism floor; the seed fix (§G) makes
  the default yamls seeded, so these now measure only the GPU-jitter floor.
- Momentum (S1-S3) / fluxtune server-optimizer retry — roadmap item, not parity; see
  `fluxtune_contributions.md` §8.2 / FWDLLM_DESIGN.md. Resume only after Phase-1 parity closes.

---

## §F  Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. **Never** put overhead on the vclock (`vclock = max(vclock, sct)`).
   **RESOLVED 07-14** — an earlier session questioned whether the "never" is too broad for
   genuinely non-trivial aggregator work. Settled: fwdllm's ~10s/cycle aggregator overhead is compose-loop-
   coupling + (now-trimmed) logging volume — real-harness implementation waste, not genuine FL work — so
   engineering it away (already done for the logging half) is the right direction, not folding it onto the
   vclock. "Never put overhead on the vclock" stands as originally written.
2. **Progress axis is `data_id`.** Updates-per-data_id is the dynamic-K random variable — an output to match,
   not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold` / `max_iterations_per_data_id`
   are baseline-defining config knobs.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** A real↔sim gap has two fix directions — check whether
   the **real** input is the divergent side before tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as **real-transport artifact** (no sim analog,
   gate `and not self.simulated`) vs **algorithmic property**. Scope-check before editing shared code:
   `fwdllm_aggregator.py` = fwdllm blast radius; `top_aggregator.py` / shared parity engine / `_sim_recv_min`
   can silently break async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only edit → `pytest tests/mode -k fwdllm`; shared parity
   engine → add `examples/async_cifar10/scripts/parity` + `tests/mode -k parity`; shared stack → full `pytest tests/`.
8. **Telemetry-FIRST, then instrument, then (rarely) run.** Validate/refute from telemetry ALREADY ON DISK
   first — name the exact field/line. Ship telemetry + plot + pytest IN THE SAME CHANGE as any new mechanism.
   A run is justified only to observe an EMERGENT quantity no stored telemetry can yield.
9. **Consult PARITY.md vclock rules BEFORE any sim-clock change.** Clock is a monotone `max`; NEVER put
   overhead on it; the sim SKIPS real waits and reconstructs order from sct (`SimReorderBuffer`).
10. **The vclock is virtual wall-time; the sim MUST produce SPEEDUP (`sim_rate = vclock/wall ≥ 1`).** The
    forward-grad "train" pass is the only irreducible real wall (parallel across trainers); transport/
    inter-round/delay waits are vclock jumps, never process sleeps. `sim_rate < 1` means the sim is stalling
    on a real wait it should skip, OR (§B, fluxtune) its per-commit processing throughput can't keep pace with
    arrivals — check BOTH before assuming it's a wait-modeling gap.
11. **Correctness before speed; SHARED roots before per-baseline.** Fix major logical-correctness divergences
    before any throughput/wall tuning. A bug that fails rungs across ≥2 baselines outranks a single-baseline one.
12. **Logical determinism is the parity definition.** For a matched scope the sim must take the SAME sequence
    of steps in the SAME order as real — same trainers selected, same order of update receipt, same
    aggregations and rollbacks — differing ONLY in wall-clock. Prove this on the first data bin before
    extending length.
13. **Do the right thing — no hacks.** A hack that moves a number without a correct mechanism is a regression
    in disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. Any new code comparing/stamping a version goes through it — no
    bare-scalar shortcut "for now."
15. **Verify claims against code, not documentation/comments.** A docstring saying two functions are
    "the analog of" each other is a claim about intent, not a guarantee of behavioral equivalence — diff them
    (fluxtune vs felix, §B/§G).
16. **Don't blame GPU/resource contention at n=10** — checked and refuted once already; won't apply until
    ≥100-trainer scale. Any unexplained real-wall gap should be assumed closeable by measurement (a wall-clock
    + vclock phase timer around the suspect stage), not guessing.
17. **A bounded rotating in-flight cohort settling at `c − agg_goal` surplus is the correct steady state for
    a `c ≫ agg_goal` fedbuff pool, not a backlog to eliminate.** Total concurrency is held constant by
    construction: a boundary that closes on `agg_goal` commits frees exactly `agg_goal` slots and dispatches
    exactly `agg_goal` replacements, so surplus fixed-points around `c − agg_goal` (matches fluxtune's own
    measured `buf_depth` sitting at 7-8/10 for `c=10, agg_goal=3`, exactly). `carried_surplus_commits` will be
    the MAJORITY commit-source bucket in steady state (~70% at fluxtune's ratio) — don't drive it toward 0;
    only `pastdated_commits` (genuine scheduling anomalies, distinct bucket) should read ~0.
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** If a value is configurable OR an
    always-on correctness path (seed, delay floor, agg_goal, c, availability trace, flag-gate state), it MUST
    surface identically across the yaml, the run snapshot, AND both roles' telemetry — the same resolved value,
    not None on one side. Divergent/missing logging silently breaks run reproducibility and wastes a session
    chasing a phantom (07-16: trainer `seed` logged `None` while the run was genuinely seeded, sparking a false
    "seed regression" hunt; snapshot recorded no seed at all). When you add or wire a knob, add its log on every
    surface in the SAME change and diff a real run to confirm it reads the resolved value, not a default.
19. **No compute on the critical path for a log the run doesn't need.** Logging is for monitoring; the real
    experiment must spend wall time on compute, not on building log strings. Any log whose ARGUMENTS are
    non-trivial (`_calculate_hash`/GPU→CPU `.cpu()`/`.item()`/`.tolist()`, `torch.allclose`/`.norm()`/`stack`,
    a comprehension or repr over params/grads/state_dict) MUST be gated behind `logger.isEnabledFor(logging.DEBUG)`
    (or a purpose flag like `_perturb_audit`) so it computes ONLY when explicitly enabled — an f-string evaluates its
    args even when the level would drop the line, so an ungated `logger.debug(f"...{hash(x)}")` still pays the
    cost. Determinism/correctness audits belong here: verify once with DEBUG on, then run with it off at zero cost.
    The high-perf run keeps at INFO only what plotting/sanity scripts parse (`extract_sanity_checks.py`'s regexes:
    selector `select()`/`_select_candidates`, `_distribute_weights…data_id`, `eval_model(): results after eval`,
    trainer PID, client data hash/samples) + telemetry `emit()`; everything else is DEBUG or deleted. This holds
    for ALL baselines and BOTH roles — grep `logger.(info|debug).*_calculate_hash|format_hash|\.item\(\)` before
    a perf run. Never change a value inside a gated log (reads are inert; gating must stay correctness-neutral).

---

## §G  Landed fixes — one line each (problem → fix). Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

- **`cohort_sequence`'s SET tie-window was blind to dispatch timing (async, fluxtune)** (07-17d) — compared bare
  per-unit registry delay (`max(raw,floor)/divisor`), valid only when every candidate dispatches at the same
  reference time (round-1); once `c≪N` causes mid-run slot refills, a late-dispatched fast trainer and an
  early-dispatched slow one can tie in EFFECTIVE completion time despite a large bare-delay gap. Fixed
  `_cohort_set_tie_ok`/`checks.py`: per differing trainer, check (single-mode) whether its own commit landed
  within `tie_window_s` of a cohort boundary in the mode that includes it, and that it exists nearby in the
  mode that excluded it — also transparently resolves the cascade (a granted tie echoes into cycle N+1).
  VALIDATED against banked telemetry (n40 pair, no new run): cycle 0's `...383`/`...407` swap correctly graded
  a tie (real gap 0.69s, sim 0.0s).
- **`cohort_sequence`'s SET rung uncapped over the entire run** (07-17d) — a legitimate admission tie cascades
  unboundedly once one occurs (the displaced member becomes the front of the next cohort, displacing whoever
  the other mode picked there, ad infinitum); chasing exact SET match past the achievable-determinism window
  chased an artifact of the cascade, not a bug. Capped to the same `max_bin` (default bin 1) window
  CADENCE/VAR/ORDER already used; a cycle resolved only via a tie also exempts VAR/var-derived
  CADENCE/ORDER for that cycle (differing tie-explained trainers legitimately produce differing gradients).
- **`participation_parity` (S2) windowed on `round`, coarse/near-constant for fwdllm** (07-17d) — the
  matched-window degenerated to `n=1` (full-run aggregate counts, unmatched, silently) for every fwdllm run to
  date. Now windows on cycle position for fwdllm-shaped telemetry (n40 pair `n_rounds_matched`: 1 → 13).
  VALIDATED: `speed_class_tvd=0.031` (tol 0.15) — population-level participation genuinely matches.
- **fluxtune `cohort_sequence` admission-divergence investigation — CLOSED** (07-17d) — with the above landed,
  the n40 pair's remaining divergence (`...395`/`...384`, cycle_index=2/iteration=3) is confirmed legitimate
  stochastic noise, not a bug: `async_oort.py`'s `_handle_send_state` uses `select_random` (uniform seeded
  draw) for the ENTIRE first data_id (fluxtune's `model_version` only bumps on data_id completion), and the
  eligible-candidate SET fed to that draw differs real-vs-sim due to timing (sim's post-cohort-close
  carried-surplus commits drain in a wall-clock burst that drags the shared vclock forward at different rates
  than real's continuously-flowing arrival order) — same category as the already-documented `refl`
  stochastic-core-identity pattern, verified at the population level (S2) instead of per-cycle identity. No
  `async_oort.py` change; 23 new/updated tests, 436 `tests/mode -k "parity or fwdllm"` pass.
- **`minInitialTrainers=c` reopened the post-barrier join-order race, all 3 baselines** (07-17) — sim's cycle
  cadence outruns real's wall-clock (transport-collapse), so identical wall-clock-bound trainer spawn timing
  landed in different cycles per mode (fluxtune 6-min pair: sim cycle1 t=106s/33 joined vs real t=125s/42).
  Fixed: `minInitialTrainers=N=100` in all 5 n100 parity yamls. VALIDATED 07-17 — see next entry.
- **`_handle_recv_state`'s recv-side resample fallback deadlocked ALL dispatch once minInitialTrainers=N
  released 100 trainers at once** (07-17b) — the fallback resampled fresh candidates itself whenever
  `selected_ends` was empty, contradicting its own docstring ("get() will proceed and wait on
  distribute_weights") and racing `_handle_send_state`'s real dispatch; a `curr_end_state != VAL_END_STATE_NONE`
  bug (Python `None` vs the string `"none"`) made it claim never-touched ends into `all_selected` before
  send-state ever sent them anything — 0 agg_rounds in 6 min (rare at c=30's staggered arrival, guaranteed once
  minInitialTrainers=N released all 100 at the same instant). Fixed by REMOVING the fallback entirely (not
  patching the comparison) in `async_oort.py`/`async_random.py` — `_handle_recv_state` is now strictly
  read-only over `selected_ends`, matching its docstring; 3 new tests. **VALIDATED 07-17**: fresh 6-min pair,
  sim ran to completion (72 agg_rounds, 725 dispatches, 0 stalls); `cohort_sequence` SET divergence pushed from
  cycle 1 → cycle 2 (cycles 0+1 now exact SET+ORDER matches). Surfaced a separate `var_ok` gap — see §B.
  **07-17c**: also VALIDATED for fwdllm/fwdllm_plus at 5400s (SET+ORDER+CADENCE exact over the full run, 132/
  259 cycles) — join-race+deadlock fix confirmed baseline-agnostic; only the shared `var_match_frac` gap
  remains (§B cross-baseline).
- **Round-1 cold-start cohort scramble (async only)** (07-16) — `_sim_recv_min_grad`'s gate was blind on a
  trainer's first contact (`_sim_known_delay_s` reactive, no fallback); added `unknown_stuck`, a wall-clock-cap
  hold, instead of oracle-seeding the delay. `sim_gate_compute_cap_s` re-derived 16.0→11.0 (stale). VALIDATED
  on a 6-min pair — cycle 0 now 10/10 (was 8/10). `cohort_sequence` still fails from a new cycle-1+ divergence
  (§B item 1, separate mechanism).
- **`cohort_sequence` tie-window for arrival-race divergence** (07-16) — SET/ORDER divergence now granted a TIE, not a fail, when every differing trainer's registry-derived expected delay is within `tie_window_s=1.0` of the others'; ungrantable (no delay model/unknown trainer) stays strict. 6 new tests.
- **`agg_goal` 3→10 for fluxtune's `c=30` pool** (07-16) — K=3 was too tight for a `c=30` near-degenerate fast class, admitting only a coin-flip subset per cycle; `fluxtune_n10_smoke[_sim].yaml`. VALIDATED on the `_202633`/`_212850` 1h pair: `throughput`/`total_commits`/`terminal_state` all PASS (was ~6% over tol) and `convergence_loss` 0.172→0.007. `cohort_sequence` itself still fails — re-diagnosed as a cycle-0 seed cascade, not the `agg_goal=3` contention (§B fluxtune #1).
- **`step_timing_breakdown` degenerate-noise skip was max-gated, not p99-gated** (07-16) — a single GC/cache-miss outlier in a multi-thousand-sample near-zero function (`tb_batch_to_device`) defeated the skip and scored KS=0.3+ on quantization noise; gated on p99 instead. `_emulate_training_delay`/`_fetch_weights` were already correctly exempted since 07-10 — the doc previously misattributed the fail to them.
- **fwdllm/fwdllm_plus yamls renamed `_n10_smoke*` → `_n100_smoke*`** (07-16) — filename only; `num_trainers` was already 100. `run_sequential.sh`'s baseline map + `test_config_generator.py` updated.
- **`minInitialTrainers` defaults to N** (07-16) — `run_sequential.sh` now waits for ALL trainers before the first selection (removes the `>=threshold` pool-size race: real fired at 98, sim at 99 → divergent seeded draw). Set-exact initial cohort; `--min-initial-frac <1` opts back into straggler tolerance. Caveat: blocks forever if a trainer never joins.
- **AVL_TRAIN stamped at registration** (07-16) — `Channel.add` now sets `PROP_AVL_STATE=AVL_TRAIN` so a just-joined end is never read UNKNOWN before the first selection stamp; trace/tracker overwrites per selection (aware) or it stays (unaware). Kills the startup UNKNOWN transient in avail_composition; 4 tests (init→aware-override→unaware-persist).
- **seed added to snapshot.yaml** (07-16) — serializer dropped `hyperparameters.seed` while aggregator_config.json carried it; snapshot's `aggregator` block now records the resolved seed (§F-18 completeness).
- **`eval_model` exempted from `agg_step_timing_breakdown`** (07-16) — daemon-backgrounded, off-vclock, off-critical-path; real↔sim wall gap is pure GPU contention (sim trainers never sleep → GPUs 3.5× denser). Excluded from gating; the still-enforced trainer-side `phase_gpu_compute`/`step_timing_breakdown` catch any bleed into trainer compute.
- **All selectors leaked trainer JOIN order into the seeded draw** (07-16) — async_oort built its candidate list from raw `ends.keys()` (join order) before `_rng.choice`; canonicalized to `sorted(ends.keys())` in oort/async_oort/fedbuff/refl_oort/feddance; selectors now default seed 1234. Root of fluxtune cohort cycle-0 divergence (→ v2/utility/pref). 199 selector + 156 parity tests pass; validate on next fluxtune rerun.
- **`v1b_iters_moving_avg` rung added** (07-16) — moving-avg iters-per-data_id over the whole run within a tight bound; catches trajectory DRIFT that v1's pooled KS+mean cancels. FAILs the current pre-fix fluxtune pair (sim 14.5 vs real 13.6).
- **Trainer compute re-measured post overhead-removal** (07-16) — genuine JVP is mean 0.47s / max 6.1s (was 3.63s; ~87% was harness hashing/gc/logging), matches real↔sim ~1%; drove the §O floor re-derivation 7.0→4.0 (divisor unchanged). Detail in FWDLLM_DESIGN §O.
- **Trainer wall-time attribution — READ, no anomaly** (07-16) — n100 `_train_one_batch` 354ms real ≈ 349ms sim, `tb_*` account ~99%; prior 3952ms didn't reproduce.
- **Eval is NOT the residual (corrects a same-day wrong call)** (07-16) — eval is daemon-backgrounded both modes; `_process` not slowed during eval (0.4×), cadence +11%; residual is dispatch-order + raw-wall/vclock, not eval.
- **Aggregator is queue-bound, not 350ms** (07-16) — send→ingest 6.8s real/26s sim = queue wait (70% busy, serial commits, in-flight cap c=30), not MQTT transit; gates `sim_rate`.
- **Perturbations validated deterministic across modes** (07-16) — utility matches to 5 d.p. when aligned; divergence is dispatch ORDER (which data_id per trainer), onset median 1 update.
- **Trainer `seed` telemetry logged `None`** (07-16) — seed lived only in aggregator config; added `seed: 1234` to trainer `config_overrides` in all 6 base yamls (inert: trainer self-selects), confirms next run.
- **Aggregator per-commit waste removed** (07-16) — `FedSgdAggregator.aggregate` did the SAME full-model `deepcopy` block ×3/commit (copy-paste; only last used) → collapsed to 1; gated ~8 eager `_calculate_hash` GPU→CPU sha256 (debug strings built even at INFO) across `aggregate`+`fwdllm_aggregator`. 267 fwdllm tests pass; next run measures the `aggregate`/`_process` wall drop.
- **`--min-initial-frac` startup-barrier lever added** (07-16) — `run_sequential.sh` computes `minInitialTrainers=floor(frac·N)` into `selector.kwargs` (all selectors); opt-in A/B for the dispatch-order root (§B fluxtune #4), unchanged when unset.
- **fluxtune `preferred_duration` + `avail_composition` PASS post-fix** (07-16) — seed/pacer fix cleared pref (real 0.273→match); AVL-at-registration drove avail UNKNOWN→0. Confirmed on the `_161235`/`_161255` rerun.
- **Sim ran UNSEEDED while real had `seed=1234`** (07-15) — sim yamls omitted the key; added to all 3 + `config.py` default `None`→`1234`; drove `cohort_sequence` set-match to 0.0 at cycle 0.
- **`_handle_recv_state` leaked dispatch order via PYTHONHASHSEED** (07-15) — `select_random`'s `dict.fromkeys` fix never reached it; ported to async_oort/fedbuff/async_random.
- **Trainer batch interior emitted zero telemetry** (07-15) — `timer_decorator` keys off `args[0].fwd_llm_stage`; nested helpers pass `device`; added `_stage_timer` + 10 `tb_*` phases + phase/unaccounted CDFs.
- **`agg_step_timing_breakdown` false positives** (07-15) — KS on all-zero + tight distributions; degenerate-skip, 5% mean escape, exempt `_distribute_weights_async` (real-only `sleep(0.1)`).
- **`aggregation_plots` dead on NameError** (07-15) — `pc_x`/`pc_y`/`pgm_y` collection loop missing; restored, `pastdated_commits_over_rounds.pdf` renders again.
- **TIMING_OVERRUN** (07-15) — §O's margin used fast-class MEAN not FLOOR; `training_delay_floor_s` fix; VALIDATED 0 overruns (5400s run).
- **fluxtune accuracy floor** (07-14/15) — cross-refs `fluxtune_contributions.md` §8's tracked H0/F1-F15 collapse; not a parity bug, both legs match.
- **`r1_inflight_overlap`** (07-15) — checker flagged FedBuff's legit stale-accept redispatch as a violation; rescoped per `version_key`; FIXED, 0.0%/0.0%.
- **`_sim_gate_compute_cap_s`** (07-15) — blind `10.0` thinner than observed max; derived `16.0` for fluxtune, `10.0` fallback documented as non-universal.
- **`select_random` order leaked via PYTHONHASHSEED** (07-14) — `set()`→`dict.fromkeys()`; VALIDATED, fwdllm `cohort_sequence` 100% match.
- **fluxtune `preferred_duration`** (07-14) — oort pacer one-branch port bug; faithful both-branch port; 50.7pp→9.3pp gap (see §B for a 07-15 borderline re-open).
- **Parity-CLI progress-axis bugs** (07-14) — axis picked per-side independently, glob collided fwdllm/fwdllm_plus; prefer `data_id`, anchor glob on `_{tag}_n<N>_`.
- **Aggregator `step_timing` unparsed** (07-14) — zero checks read it; added loader capture + `agg_step_timing_breakdown` rung.
- **fwdllm had no seeded yaml** (07-14) — seed plumbing was already correct, just unexercised; added 6 seeded yaml pairs.
- **`recv_fifo` hot path logged at INFO** (07-14) — 425k lines/run, unread; downgraded 9 mechanical lines to DEBUG.
- **Server-momentum (S1)** (07-14) — landed flag-gated (`server_momentum`, default 0.0 no-op) + A/B yamls; NOT run, resume deferred to `fluxtune_contributions.md` §8.2.
- **Reactive gate blocked real wall** (Bug A, 07-13) — re-checked stale state after a blocking call; `_sim_gate_is_safe` checks first; `sim_rate` 0.97→1.82×.
- **Carried-surplus commits misclassified "round1"** (Bug B, 07-13) — classifier wasn't re-keyed to `data_id`; ingest-time carry-over stamp, separate bucket.
- **`eval_model()` blocked dispatch** (07-13) — sync eval stalled the loop; backgrounded on a daemon thread, dead assignment removed.
- **fwdllm_plus livelock** (07-13) — `RandomSelector` freed only `k=5` of `c=10`; removed the stale `k` knob entirely.
- **fluxtune commit-path stall** (07-13) — phantom `_sim_inflight_expected` entry (stamped, not computing); `sim_compute_truthful_gate` skips stale dispatches.
- **fluxtune cohort-SET divergence** (07-13) — real released a busy trainer's guard on RETURN not commit; hold-to-commit + `send_timeout_wait_s=300`.
- **`version_key` unification** (07-13) — version identity was bare-int in some places, 3-tuple in others; one shared 2-tuple property everywhere.
- **Remainder-wait delay model** (07-13) — additive `send+gpu+D` gave nondeterministic arrival order; real sleeps `max(0,D-gpu)`, sim never sleeps D.
- **Slot residence hold-to-COMMIT** (07-13) — release-on-RETURN undercounted in-flight state 3×; hold slot until commit, sync+async.
- **Async surplus-grad handling** (07-13) — dropping at agg-goal boundary wasted ~7 grads/cycle at `c≫agg_goal`; carry surplus + hold busy trainers.
- **Async `total_commits`/`throughput` overlap bug** — summed overlapping cycles as sequential (76-86% spurious diff); fall back to raw wall for async.
- **Clock-rate rungs anchored on transport artifact** — used full wall (localhost-only latency) instead of `intrinsic_span_s`.
- **`cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER** — one cap tripped on real GPU fp16 jitter; SET hard/uncapped, rest cap to bin 1.
- **"GPU under-provisioned at n=10"** — refuted; spawn table is balanced round-robin, 8 GPUs, 1 core/trainer.
