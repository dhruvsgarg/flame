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

## §A  Score — refreshed 2026-07-18 (see PREAMBLE's score-tracking trigger)

> **fwdllm/syn_0 row is STALE** (07-17 5400s numbers, carried forward) — 07-18 relaunch crashed pre-training,
> root-caused + fixed same session (§G 07-18k), not yet rerun.
>
> **Post-fix spot-check** (sub-1h triple, HEAD `a1adcea8`, not a score-trigger run): `cohort_sequence` now EXACT
> 1.0 on every sub-metric for fwdllm and fwdllm_plus — P0-1 confirmed (§G 07-18l/n); re-run at ≥3600s to make it
> official below. fluxtune's leg in that triple logged a transient `git_info.dirty` at launch — its throughput/
> round-count deltas there are untrusted noise, not a regression.
>
> **fluxtune/fwdllm_plus rows are fresh 7200s (2h) pairs** — longest parity pairs to date. fluxtune down to 5
> fails (07-18j NUMA-isolation fix validated `agg_step_timing_breakdown`+`convergence` at 2h scale, §G).
> fwdllm_plus got WORSE (8→13) — the real-only `num_min_req=1` clamp gap (previously fwdllm-only) is now
> confirmed shared across both sync baselines, only surfacing once a run is long enough to compound (invisible
> at 5400s, clear by 7200s) — cross-baseline P0-2, top open item after fluxtune's cohort cascade.

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260718_015337`/`_035554` (delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10) | ~7200s | 62 | 5 | 18 |
| fwdllm/syn_0 | `run_20260717_121511`/`_134658` (delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10) — **STALE, 07-18 relaunch crashed pre-training (fixed, not rerun)** | ~5400s | 49 | 13 | 22 |
| fwdllm_plus/syn_0 | `run_20260718_015405`/`_035617` (delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~7200s | 50 | 13 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm (STALE) | ✗ | ✓ | ✗ | ✗ | ✗ | – | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✗ | ✓ | ✗ | ✗ | ✗ | – | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |

**All failing rungs, this run:**
- **fluxtune** (5, down from 6 at 1.5h): `cohort_sequence` — SET still diverges at cycle_index 2 (same onset as
  every prior session), capped-window `set_match_frac=0.2` over the bin-1 comparison window (§G 07-17d cap) —
  unchanged mechanism, legitimate stochastic tie cascade, not re-triaged. `v2_var_trajectory` (mean_rel_diff
  0.0334 vs 0.02 tol) and `v1b_iters_moving_avg` (ma_max_abs_dev 2.15 vs 0.75 tol) — both roughly FLAT vs the
  1.5h pair (were 0.0536/2.25), not still growing with run length as hypothesized last session; supports
  "downstream of the fixed-onset cascade" over "unbounded drift." **`convergence` and `agg_step_timing_breakdown`
  now PASS** (were failing/hairline at 1.5h) — validates the 07-18j NUMA-isolation + tolerance fix at 2h scale,
  closed (§G 07-18g/j). **NEW**: `trainer_speed_identity` — `speed_s` sub-check passes (max rel dev 0.8%, pure
  registry-assigned speed matches), but `utility` sub-check fails: 4/100 trainers outside 10% tol (worst 14.1%,
  real 8.499s vs sim 7.3s) — plausibly the same 4-ish trainers whose cohort membership diverged in the SET
  cascade accumulating different grad/utility history, not independently triaged. **ESCALATED**: `step_timing_
  breakdown` / `tb_prepare_perturbation` — was a hairline KS 0.251 "maybe noise" call at 1.5h, now confirmed KS
  0.267 (tol 0.25) / mean_rel 45.5%, real 1.8ms vs sim 3.4ms — grew with run length rather than staying flat,
  which argues against pure quantization noise; needs actual triage next session (§B item 3).
- **fwdllm** (13, STALE from the 1.5h pair — crashed on relaunch this session before producing new data, see
  §B fwdllm and §G 07-18k). Numbers unchanged from last write-up, not reproduced here to avoid a false diff.
- **fwdllm_plus** (13, up from 8 at 1.5h — got WORSE): cohort SET/ORDER/CADENCE still EXACT (1.0/1.0/1.0 over
  the compared window) — `cohort_sequence` still fails only on `var_match_frac=0.5` (cross-baseline, see §B).
  **NEW, major**: `overhead_residual`/`per_round_advance` now FAIL (sim 69.06s/round vs real 58.15s/round,
  +15.3%) — same shape as fwdllm's previously root-caused real-only `num_min_req=1` clamp gap, now confirmed
  present here too and only visible once the run is long enough to compound (absent at 5400s). Everything else
  new this run is a downstream symptom of the same rounds-completed gap, not an independent bug: `throughput`
  (105 sim vs 108 real rounds), `total_commits` (94 sim vs 108 real, -13%), `terminal_state`,
  `g2_grad_pool_size`, `v1_iter_per_data_id`, `v1b_iters_moving_avg`, `v2_var_trajectory`, `convergence`
  (acc_diff 0.0506, hairline, tracks the fewer-rounds-by-end-of-run gap). `step_timing_breakdown` — new
  `_send_grads` fail (KS 0.9, real 0.9ms vs sim 8.5ms, both sub-ms) — likely the same degenerate-noise category
  as other tiny funcs but not yet gated/exempted, UNEXAMINED. `agg_step_timing_breakdown` — `_apply_weighted_
  update` (mean_rel 50%) and `_compute_var` (mean_rel 82%) fail even past the widened 50% tolerance — same two
  functions flagged in fluxtune's aggregator contention investigation (§B fluxtune, now closed) — plausibly the
  same NUMA/memory-bandwidth mechanism, not yet checked for fwdllm_plus specifically. `trainer_speed`/
  `training_budget` no longer fail (were new fails at 1.5h) — no action needed.

See §B for what's actively being worked per baseline; see §G for what's already closed.

---

## §B  Next steps / open issues — per baseline, as of the §A runs above

### Priority plan — 2026-07-18 deep-dive (analysis only; implement next session, in this order)

This session root-caused two of the table items below to code-level bugs (not accepted noise) and
downgraded/confirmed three others via telemetry already on disk — no new runs needed for any of it. Ordered by
(confirmed-bug > cheap-verification > new-instrumentation > infra):

~~**P0 — sim sync barrier can't do incremental (`num_min_req=1`) collection**~~ — **LANDED (code+tests), see
§G. NOT YET VALIDATED against a live pair** (P0-1, the grad-merge-order bug, closed separately — §G 07-18l/n).

**P1 — cheap verifications against telemetry already on disk, no new runs, do alongside/before P0:**
3. Cross-reference `trainer_speed_identity`'s 4 utility outliers against the FULL-run cohort-cascade displaced
   set (item 2 below) — this session only checked the first 5 rounds (partial: 2/4 matched).
4. Correlate fwdllm_plus's `_apply_weighted_update`/`_compute_var` timing gap with round-boundary contention
   burstiness (item 4 below) — structural hypothesis identified, not yet measured. **New data point this
   session**: fwdllm (not just fwdllm_plus) now shows the identical `agg_step_timing_breakdown` residual shape
   on the fresh short pair (`_aggregate_grads_sync`/`_apply_weighted_update`/`_compute_var`/`_prepare_round_
   state`/`_process_aggregation_goal_met`/`aggregate` all fail at 44-87% mean_rel past the 50% tol) — since
   fwdllm shares fwdllm_plus's `c=agg_goal=10` zero-slack config, this strengthens (doesn't yet prove) the
   contention-spike hypothesis over a fwdllm_plus-specific cause. **New hypothesis (untested)**: P0-1's fix
   itself may add a second contributing factor — `_process_single_trainer_message` now only buffers per
   message (`_pending_cohort_contribs.append`, cheap), deferring ALL of the cohort's `aggregate_grads_from_
   trainers` tensor-arithmetic to one back-to-back burst in `_process_aggregation_goal_met`, right before
   `aggregate()`'s own memory-heavy blocks — vs the pre-fix behavior of spreading that work across the round,
   interleaved between successive `recv_fifo` waits. Total compute is unchanged, but the burst now lands at the
   same instant as `aggregate()`'s contention-sensitive work, which could compound the gap under investigation
   here. Check by timing the buffered-replay loop itself (add a timer around the loop in
   `_process_aggregation_goal_met`, `fwdllm_aggregator.py:1931-1944`) and correlating its wall cost against
   real vs sim GPU/CPU density at that instant — no new run needed, existing pair has the data once instrumented.
5. Extend the candidate-set divergence measurement (item 1 below) across the full run and resolve the
   data_id=0 iteration-count mismatch (real 4 iters vs sim 5) it surfaced.
7. **New, this session**: `v2_var_trajectory` still fails post-P0-1 (fwdllm 11.0% mean_rel vs 2% tol; fwdllm_plus
   2.51% vs 2%, hairline) — since `var_match_frac` (cohort_sequence's per-cycle variance-decision match) is now
   exact 1.0, this is NOT simply downstream of the fixed grad-merge-order gap; check whether it's a genuinely
   separate variance-calc discrepancy or a short-run artifact (re-check at the next ≥3600s pair before treating
   as independent).

**P2 — new instrumentation + A/B (single baseline, fluxtune):**
6. ~~`tb_prepare_perturbation` branch-taken + concurrent-trainer-density logging, then A/B~~ — **LANDED, NOT
   YET RUN.** `_stage_timer` (`tc_transformer_trainer_distribute.py`) gained an `extra: dict` param; the
   `tb_prepare_perturbation` call site now passes `extra={"branch": "cached"|"fresh"}` (same condition the
   function itself gates on). Concurrent-trainer density needs no new field — derived post-hoc from the
   EXISTING `trainer_round` `gpu_pass_start_wall`/`gpu_pass_end_wall` windows, same overlap method as
   `measure_agg_overlap.py` (§G 07-18f). New standalone script `analyze_tb_prepare_perturbation.py`
   (`expt_scripts/`) prints duration-by-branch and duration-by-concurrency-bucket for one run dir; smoke-tested
   against the existing 07-18 fluxtune real+sim pair (pre-dates the `branch` field, so that column reads empty,
   but concurrency already shows a clean gradient: real 1.82ms low-density → 11.0ms high-density, n=15 top
   bucket; sim similar direction, noisier). Not unit-testable in isolation — `tc_transformer_trainer_distribute.
   py` requires `transformers`, not installed in `dg_flame`; same constraint applies to all trainer-side code in
   that file, no existing test imports it either. Validate the `branch` field lands on the next real fluxtune
   run: `python analyze_tb_prepare_perturbation.py --run <new_run_dir>` should show `branch field present on`
   count > 0.

**P3 — infra robustness, not parity-blocking (this session's GPU-crash tangent):**
7. Dynamic GPU health filtering (see below) — `CUDA_DEVICE_ORDER=PCI_BUS_ID` (landed today, `runner.py`) only
   fixes *which* physical card a given ordinal maps to; it does not detect or skip a genuinely broken card.

---

**Q: will the system dynamically filter out non-working GPUs and keep going on the healthy ones?**
No, not yet — today's fix only makes CUDA's ordinal numbering match `nvidia-smi`'s (see the crash writeup,
previous turn), so assignment is *deterministic and reasoned-about-able*, but there is still no health check
anywhere in `flame/launch/`. If GPU 0 (nvidia-smi) is still broken next launch, whichever role's ordinal maps
to it (a trainer, under round-robin, or the aggregator, under the fixed "spare Nth GPU" rule) will still hit it
and crash the same way. Proposed design (P3, not yet built): a preflight pass in `runner.py` before spawning —
for each candidate CUDA index 0..`num_gpus`, attempt a cheap op (`torch.zeros(1, device=f'cuda:{i}')` or check
`nvidia-smi --query-gpu=index,memory.used --format=csv` for a card reporting `[Insufficient Permissions]`/`ERR!`);
build a `healthy_indices` list; remap both the trainer round-robin (`spawner.py:315`,
`gpu_id = (trainer_id-1) % self.num_gpus`) and the aggregator's spare-GPU pick (`runner.py:277-282`) to index
into `healthy_indices` instead of raw `range(num_gpus)`. Reduces `num_gpus` effective capacity by 1 per bad
card found (log it loudly) rather than crashing. Not attempted this session — infra work, lower priority than
the actual parity bugs above.

---

### fluxtune (~7200s, delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10) — TOP PRIORITY
1. **`cohort_sequence` SET cascade — quantified this session, user hypothesis about its historical cause
   PARTIALLY REFUTED.** Pulled the raw `agg_round` telemetry (both legs, `run_20260718_015337`/`_035554`):
   cycle_index 2 (real `data_id=0,iter=3`) diverges by exactly **4/10 members (60% overlap)** — real cohort
   `{458,461,420,467,387,372,414,456,371,448}` vs sim `{405,434,463,457,387,372,414,456,371,448}`. Operator
   hypothesis was that this traces to a historical period when aggregation was slow + eval was on the critical
   path (both since fixed, §G 07-13/07-16), and that with those fixed, real/sim should now release-and-select
   at the same rate, tightening the candidate set. **Checked directly: doesn't hold as stated.** `aggregate()`'s
   own wall cost is now ~70-140ms/call (§G 07-18g/i) — two orders of magnitude below the observed ~10s
   inter-round cadence (dominated by trainer JVP compute, 8-15s/trainer per the same telemetry), so
   aggregator speed was never the pacing bottleneck at this `agg_round` cadence and the eval/aggregate fixes
   couldn't have meaningfully closed this gap. The real driver is structural: fluxtune's `c=30 ≫ agg_goal=10`
   fedbuff pool has a designed-in surplus/carry-over (§F-17 — NOT a bug, `carried_surplus_commits` is supposed
   to be the majority bucket), and real's round-4 inter-round gap is anomalously fast (4.46s vs a ~10s
   baseline) exactly where the divergence appears — consistent with a surplus-drain burst, but one driven by
   the `c≫agg_goal` pool's inherent queueing dynamics, not by aggregator/eval slowness. **Also newly surfaced**:
   real and sim don't even take the same NUMBER of iterations to clear data_id=0 — real advances to data_id=1
   after iteration 4, sim is still on data_id=0 at iteration 5 — so cadence isn't matched even in aggregate,
   independent of the cycle-2 SET tie. **Next step (P1-5):** extend this same telemetry-diff to the full run
   (not just the first 5 rounds) to get a run-wide divergence-magnitude trend, and separately investigate the
   iteration-count mismatch at data_id=0 (4 vs 5) — that's a distinct, possibly more tractable, target than the
   SET tie itself for "tightening the case," since it's an aggregate cadence question, not a per-cycle stochastic
   one.
2. **`trainer_speed_identity` `utility` sub-check — PARTIALLY explained this session, not fully closed.**
   Cross-referenced the 4 outlier trainers (`0411,0420,0429,0430`, real 8.499s vs sim 7.3s worst-case) against
   item 1's SET-cascade-displaced trainers over data_id=0/iterations 1-5 only (`0412,0434,0373,0449,0402,0405,
   0461,0424,0420,0409,0430,0457,0436,0390,0439,0383,0467,0451,0458,0463`): **2 of 4 overlap** (`0420`,`0430`);
   `0411`/`0429` don't appear in this partial window. Answering "are these DIST (per-trainer mean, tol_rel=0.10,
   `min_samples=3`) or exact matches": DIST — `trainer_speed_identity_parity` (`checks.py:1715`) averages each
   field per trainer over the whole run and compares means, it's not a distributional KS test and not an exact
   match. Given the cascade continues past iteration 5 (cycles 2-9 in the bin-≤1 window), the 2 non-matching
   outliers may still be explained by later-run cascade displacement — this session only checked the first 5
   rounds. **If item 1 is resolved (cascade tightened/closed), re-run this cross-reference over the FULL run
   before concluding independence for `0411`/`0429`** (P1-3) — don't assume yet either way.
3. **`step_timing_breakdown` / `tb_prepare_perturbation` — instrumentation LANDED (07-18o), A/B not yet run**:
   KS 0.267 (tol 0.25), mean_rel 45.5%, real 1.8ms vs sim 3.4ms, grew with run length (was KS 0.251 at 1.5h).
   The call (`tc_transformer_trainer_distribute.py:499`) is mode-invariant code with two branches — index a
   cached `v_buffer` vs `torch.randn_like` a fresh one — so a real/sim gap here must be either (a) the same
   GPU-density artifact already accepted for `eval_model` (§G 07-16: sim's continuously-active trainers keep
   GPUs ~3.5× denser than real's paced ones) or (b) sim taking the two branches at a different RATE than real
   (itself likely downstream of item 1's cadence mismatch). Per-call `branch` telemetry now on the event
   (`_stage_timer`'s new `extra` param); concurrency derived post-hoc from existing `trainer_round` GPU-pass
   windows, no new field needed. `analyze_tb_prepare_perturbation.py` (`expt_scripts/`) prints both breakdowns
   — smoke-tested against the pre-fix 07-18 pair (no `branch` data yet, but concurrency already trends: real
   1.82ms low-density → 11.0ms high-density). Run it against the next fluxtune pair to get the actual verdict:
   density correlates independent of branch → exempt like `eval_model`; branch-rate explains it → rides item 1.
4. `_distribute_weights_async` still exempted (`gates_ok=False`, real-only sleep) — unrelated, unaffected.
5. `sim_sct_ordered_drain` A/B — unblocked. Run `fluxtune_n10_smoke_sim_no_sct_drain.yaml` against next pair.
6. **Accuracy drop after reaching 81%** — known, deferred by operator (07-15). Not yet triaged.
7. **Real↔real admissibility (§F-5)** — rungs now finalized (tie-window + tiered dep graph, `CHECK_META`
   `deps`); unblocked, ready to resume (the `cohort_sequence` admission investigation that deferred it is
   closed, §G 07-17d).

### fwdllm (fresh short pair `run_20260718_154808`/`_161939`, 886s — sub-1h, NOT a score-trigger run)
> The 07-17 5400s numbers below are gone — a fresh (if short) post-P0-1 pair landed this session and fully
> supersedes them for the items it touches. 57 pass / 5 fail / 22 skip. Re-run at ≥3600s to make this official
> in §A and re-check every item below at scale before treating it as closed.
> `cohort_sequence` `var_match_frac` gap: RESOLVED at this run's scale (P0-1, §G 07-18l/n) — see cross-baseline §B.
1. `overhead_residual`/`per_round_advance` — **absent at this scale, as expected** (doc already knew this only
   compounds past ~5400s); root cause unchanged (P0 item 2 above), not re-triaged here.
2. **`step_timing_breakdown`/`agg_step_timing_breakdown` — EXAMINED this session** (were UNEXAMINED). Trainer
   side: every fail except `_make_model_functional` is an already-exempted real-only func
   (`_emulate_training_delay`/`pause_execution`/`_fetch_weights`/`recv_wrapper`/`train_with_data_id`, all
   `gates_ok=False`); `_make_model_functional` itself is a small, likely GPU-density-contention gap (real
   35.9ms vs sim 30.3ms, 15.6% rel) — same shape/class as fluxtune's still-open `tb_prepare_perturbation`
   (§B fluxtune item 3), not independently triaged. Agg side: 6 of 7 functions fail at 44-87% mean_rel past the
   widened 50% tol — same functions and same magnitude as fwdllm_plus's residual post-fix gap (§B fwdllm_plus
   item 3); fwdllm sharing `c=agg_goal=10` (zero slack) with fwdllm_plus strengthens that item's
   contention-spike hypothesis (P1-4). `terminal_state`/`total_commits` fail only by 1 round (15 vs 16, 6.25%
   vs 5% tol) — almost certainly a small-N artifact of this run's short length (16 real rounds total), not a
   bug; expect this to clear on its own at ≥3600s.
3. `v1_iter_per_data_id`, `v1b_iters_moving_avg`, `v5_variance_pass_ratio`, `g2_grad_pool_size` — now PASS on
   this pair (were failing/UNEXAMINED at 5400s pre-fix) — consistent with P0-1 fixing their upstream driver.
   `v2_var_trajectory` still fails (11.0% mean_rel vs 2% tol) despite `var_match_frac` now exact — see P1-7,
   this is NOT simply downstream of the grad-merge-order gap and needs its own look.

### fwdllm_plus (fresh short pair `run_20260718_162558`/`_165734`, 1415s — sub-1h, NOT a score-trigger run)
> The 7200s numbers below are gone — a fresh (if short) post-P0-1 pair landed this session and fully supersedes
> them for the items it touches. 59 pass / 4 fail / 21 skip (down from 13 fails at 2h — but not apples-to-apples,
> this run is 5x shorter, see item 2). Re-run at ≥3600s to make this official in §A.
> `cohort_sequence` `var_match_frac` gap: RESOLVED at this run's scale (P0-1, §G 07-18l/n) — see cross-baseline §B.
1. `overhead_residual`/`per_round_advance` and every item-2-downstream symptom from the 2h pair (`throughput`,
   `total_commits`, `terminal_state`, `g2_grad_pool_size`, `v1_iter_per_data_id`, `v1b_iters_moving_avg`,
   `convergence`) — **absent at this scale, as expected** (root mechanism only compounds past ~5400s, per the
   2h finding); unchanged, still open, see cross-baseline §B / P0 item 2.
2. **`step_timing_breakdown`'s `_send_grads` fail — EXAMINED this session, no longer "likely degenerate
   noise."** Sim's mean (8.9-9.0ms) is an order of magnitude above the 1ms degenerate-noise floor used
   elsewhere, so this isn't quantization dither. New hypothesis (replaces the old one): downstream of the SAME
   P0-2 dispatch-cadence gap as `overhead_residual` — real's continuous per-commit refill dispatches (and
   sends) far more candidate trainers per round than sim's one-shot batch dispatch, so real racks up many more
   (cheaper, ~1ms) `_send_grads` calls while sim's fewer calls each do more work relatively. Not independently
   fixable; rides P0-2 (confirmed via call-count telemetry, see P0-2 item above).
3. **`agg_step_timing_breakdown` — still a genuine RESIDUAL gap post-fix** (both this run and fluxtune's now-
   passing pair share the NUMA-isolation + 50%-tolerance fix, §G 07-18g/j, so this survives that fix). Same
   functions failing as before (`_apply_weighted_update`/`_compute_var`/etc, 51-89% mean_rel). Structural
   hypothesis (untested, P1-4): `c=agg_goal=10` zero-slack barrier creates a contention SPIKE at `aggregate()`-
   call time (all 10 dispatched trainers finish in a tight cluster), vs fluxtune's `c=30≫agg_goal=10` soft pool
   that churns continuously. **Strengthened this session**: fwdllm (also `c=agg_goal=10`) now shows the
   identical residual shape on its own fresh pair (§B fwdllm item 2) — same config class, same symptom, raises
   confidence this is config-driven and not fwdllm_plus-specific. Next step unchanged: correlate
   `_apply_weighted_update`/`_compute_var` latency against how tightly clustered the 10 contributing trainers'
   finish timestamps are (existing telemetry, no new run needed).

### Cross-baseline / shared

- **P0-1 — `cohort_sequence` `var_match_frac` gap despite EXACT SET/ORDER/CADENCE — LANDED + VALIDATED, see
  §G 07-18l/n.** Confirmed on a fresh post-fix triple for fwdllm/fwdllm_plus (the two baselines the gap
  affected); fluxtune's `cohort_sequence` fail is a different, pre-existing mechanism (SET cascade, §G 07-17d)
  and was never a `var_match_frac` case. Re-confirm at ≥3600s scale next long run (this validation pair was
  short, see §A note) before treating as fully closed for the record.

- **P0-2 — `overhead_residual`/`per_round_advance` real-only `num_min_req=1` clamp: root-caused + IMPLEMENTED
  this session, NOT YET VALIDATED against a live pair.** Mechanism, CORRECTED after operator pushback exposed an
  imprecise earlier framing (`fwdllm_aggregator.py`): `ends_not_selected_yet` is set `True` whenever a dispatch
  pass selects `>= agg_goal` candidates; under it real collects via `channel.recv_fifo(ends, num_min_req=1,
  ...)`, returning control to the composer after EVERY commit. **This is NOT re-selection** — selection is
  cached per `version_key` (`_select_ends_respecting_reselect_gate:3121-3162`, `_reselect_each_iteration=True`
  default), so the SAME fixed `agg_goal`-sized cohort persists all cycle; no new/different trainers are ever
  picked mid-cohort. What actually happens: `_distribute_weights_sync` re-runs against that SAME cached `ends`
  list, and since `_trainer_last_model_version` is written only on grad-RETURN (not dispatch, `:1753` +
  comment `:383/392`), a still-computing dispatched-but-not-yet-returned trainer reads as "stale" every re-run
  and gets the FULL model payload RE-SENT (`_should_send_full_weights:3241-3267`; the redundancy is even
  self-documented there, and there's an opt-in flag, `_suppress_redundant_weights`, to downgrade it — OFF by
  default, not set in fwdllm's parity yamls). **Empirically confirmed, not just inferred from code**: on the
  banked `run_20260718_154808` real pair, 3851/4359 (88%) of trainer `_fetch_weights` calls were aborted
  duplicates (`grep -c "Fetch weights aborted"`), matching EXACTLY the 3851 near-zero-duration (<1ms) `_send_
  grads` telemetry samples out of 4259 total (90.4%) — confirming the trainer's own `abort_training` dedup
  (`fwdllm_trainer.py:303-339`) catches the redundant SEND, but the wasted FETCH (full-model MQTT round trip)
  still happens first. Once that noise is excluded, real's GENUINE send rate (408/16 rounds ≈ 25.5/round)
  closely matches sim's OLD one-shot-batch rate (810/33 rounds ≈ 24.5/round) — the two sides' actual training
  cadence was never far apart; the previously-reported "~10x call-count gap" was ~90% redundant-fetch noise
  with no sim analog, not ~10x more genuine work. **Fluxtune is NOT exposed**: 0/9986 real `_fetch_weights`
  calls aborted on its banked pair (`run_20260718_015337`) — its async `_distribute_weights_async` passes
  `trainer_version_keys=self._trainer_state_dict` into the channel query (`:3571-3577`), and its `c=30≫
  agg_goal=10` pool rotates through genuinely different candidates as slots free, rather than re-visiting the
  same still-computing one. Confirmed empirically, not just by selection-code reading.
  **Fix landed** (mechanism unaffected by the correction above — the fix targets COLLECT granularity, not
  selection): new fwdllm-scoped method `_sim_sync_recv_incremental` (`fwdllm_aggregator.py`, right before
  `sync_collect_and_accumulate_grads`) gives sim's collect path a PERSISTENT, instance-level sct-ordered pending
  buffer (`self._sim_sync_pending`, a `SimReorderBuffer`) — populated once per dispatched end, only ever popped
  (never wholesale recomputed/discarded), cleared at the agg-goal boundary by `_release_sim_slots_at_agg_goal`
  alongside the existing sim-state clears. The `and not self.simulated` gate on the `num_min_req` clamp is
  removed — sim now uses the identical `num_min_req=1` call site and clamp condition as real. Deliberately
  fwdllm-scoped, NOT a change to shared `top_aggregator.py::_sync_sim_recv_first_k` (§F-6: that method stays
  byte-identical for its other callers — scaffold/eager_syncfl/coord_syncfl/lifl_coord_syncfl/hybrid, and felix
  which inherits but never calls it — avoiding the larger, riskier shared-barrier-plumbing edit the design
  originally considered). 7 new tests (`test_fwdllm_sim_sync_incremental.py`: never-strand, never-double-commit,
  late-arrival pickup, vclock monotonicity, backward-compat with the old single-call shape, duration stamping);
  444 `tests/mode -k "fwdllm or parity"` + full 592 `tests/mode` pass (felix/asyncfl untouched, confirmed).
  **PREDICTED (unverified) side effect, flagged for the next run, not code-fixed pre-emptively (no evidence yet
  to act on):** since sim's sync selection path lacks fluxtune's `trainer_version_keys` re-pick guard, removing
  the `num_min_req` gate means sim's `_distribute_weights_sync` will now ALSO re-run against its cached,
  unchanged cohort between commits — plausibly triggering the SAME redundant-fetch → trainer-`abort_training`
  pattern in sim for the first time (it never exercised this path before, since sim previously collected its
  whole cohort in one call with no mid-cohort re-dispatch). Expected effects, all worth checking against the
  next pair rather than assuming: (a) `step_timing_breakdown`'s `_send_grads` KS-fail may IMPROVE (both sides'
  distributions become bimodal degenerate-noise + real-work, closer in shape) or may not, depending on whether
  the proportions match; (b) sim's own wall-clock run duration may increase somewhat (extra channel round trips
  don't touch the vclock, so shouldn't affect correctness/rung outcomes, but could affect `sim_rate` /
  absolute wall time) — real already pays this cost, so it's not a new asymmetry, just newly-shared; (c) no
  deadlock/stall risk expected (the abort path is a fast no-op, doesn't block dispatch or collection) but not
  proven absent a run.
  **Not yet run against a live real/sim pair** — do that next, checking `overhead_residual`/`per_round_advance`/
  `_send_grads`/`agg_step_timing_breakdown` (fwdllm §B item 2-3, fwdllm_plus §B item 2-3) clear, AND the three
  predicted effects above.

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

- **P2-6: `tb_prepare_perturbation` had no way to correlate its real/sim ms gap against branch or GPU density**
  (07-18o) — `_stage_timer` gained an `extra` param (branch now on the event); concurrency derived post-hoc
  from existing `trainer_round` GPU-pass windows, no new field. `analyze_tb_prepare_perturbation.py`
  (`expt_scripts/`), same overlap method as `measure_agg_overlap.py`. NOT YET RUN against fresh (post-fix)
  telemetry.
- **P0-2: sim sync barrier's one-shot drain-and-discard couldn't do real's incremental `num_min_req=1`
  collect** (07-18o) — new fwdllm-scoped `_sim_sync_recv_incremental` (persistent per-instance
  `SimReorderBuffer`, populated once per dispatched end, only ever popped not discarded) replaces
  `_sync_sim_recv_first_k` at fwdllm's one call site; `and not self.simulated` gate on the `num_min_req` clamp
  removed. Shared `top_aggregator.py::_sync_sim_recv_first_k` untouched (§F-6: other sync baselines + felix
  unaffected). 7 new tests + 444 `tests/mode -k "fwdllm or parity"` + full 592 `tests/mode` pass. **NOT YET
  VALIDATED against a live real/sim pair** — next run should confirm `overhead_residual`/`per_round_advance`/
  `_send_grads`/`agg_step_timing_breakdown` clear (§B cross-baseline).
- **Mechanism correction: real's `num_min_req=1` clamp is NOT re-selection, and is real-only-exposed to
  redundant re-fetch** (07-18o) — selection is cached per version_key (`_select_ends_respecting_reselect_gate`),
  so the same cohort persists all cycle; the clamp only makes `_distribute_weights_sync` re-run against that
  SAME cached set, redundantly re-sending to still-computing trainers (`_trainer_last_model_version` updates on
  RETURN only). Empirically confirmed 3851/4359 (88%) of fwdllm real `_fetch_weights` calls were aborted
  duplicates on the banked `run_20260718_154808` pair (trainer's own `abort_training` dedup catches the
  resend, wasted fetch still happens); fluxtune 0/9986 on its own real pair — its async dispatch passes
  `trainer_version_keys` into the channel query, no re-visit of the same in-flight trainer.
- **P0-1 canonical grad-merge fix (07-18l) VALIDATED post-fix** (07-18n) — fresh sub-1h fwdllm/fwdllm_plus pair
  at HEAD `a1adcea8`: `cohort_sequence` `var_match_frac` 0.25/0.5 → exact 1.0 (all sub-metrics) both baselines;
  fluxtune's separate SET-cascade fail unaffected, as expected. Needs a ≥3600s rerun to officially refresh §A.
- **A 07-18 fluxtune relaunch missing `--delays` silently ran D=0, collapsing sim throughput
  ~22x (buf_depth pinned near c, sim_rate≈0.045)** (07-18m) — not a code bug (P0-1 unaffected);
  `run_sequential.sh`'s `--delays`/`--delay-divisor`/`--delay-floor` defaulted to global off/base.
  Added per-baseline `BASELINE_DELAY_DEFAULTS` (fluxtune 0.48/4.0, fwdllm(+) 1.63/11.0, all on);
  explicit flags still override. Pre-flight table/fingerprint now show resolved per-baseline values.
- **`self.grad`'s per-cycle FedAvg merge summed in raw arrival order, not canonical order** (07-18l) — real
  physical-arrival vs sim modeled-sct order diverged (non-associative float add; fluxtune's `grad_aware` rate
  also reads the running `self.grad`, so it's order-*dependent* not just noisy). `_process_single_trainer_
  message` now buffers (`_pending_cohort_contribs`) instead of merging eagerly; `_process_aggregation_goal_met`
  replays the buffer in the same canonical (D, trainer_id) order `_canonicalize_cohort_commit_order` already
  computes. `fwdllm_aggregator.py`; 3 new tests prove canonical-order replay converges and un-canonicalized
  replay doesn't (negative control); 437 `tests/mode -k "fwdllm or parity"` pass. VALIDATE next parity run.
- **fwdllm aggregator crashed hard when CUDA is unavailable** (07-18k) — `ForwardTextClassificationTrainer.
  __init__` (used by both trainers and the aggregator) unconditionally called CUDA-only diagnostics
  (`torch.cuda.current_device()`/`mem_get_info`/`get_device_name`) after already resolving `self.device` to
  `"cpu"` when unavailable — an uncaught `RuntimeError` killed the aggregator instantly on relaunch (3
  concurrent baseline launches raced for the same reserved GPU 7). Gated the whole diagnostic block on
  `self.device.type == "cuda"`; CPU path now just logs and continues, matching the aggregator's own intended
  CPU-capable design (recent NUMA CPU-isolation work). `trainer/forward_training/tc_transformer_trainer_
  distribute.py`; 275 `tests/mode -k fwdllm` pass.
- **fluxtune `agg_step_timing_breakdown` gap — NUMA isolation + dead-deepcopy removal + tolerance widen**
  (07-18g/h/i/j) — CPU/memory-bandwidth contention from sim's densely concurrent trainers on `aggregate()`'s
  memory-heavy blocks (`_snapshot_retry_cache` deepcopy, `_apply_weighted_update`); fixed via NUMA-node-aware
  trainer placement (prefer non-aggregator node, spill not exclude) + removing a dead second deepcopy in
  `_snapshot_retry_cache`, plus widening this rung's `mean_tol_rel` 5%→50% for the CPU-bound residual (trainer-
  side `step_timing_breakdown` untouched, still 5%). VALIDATED PASS at 7200s/2h scale (07-18, this session).
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
