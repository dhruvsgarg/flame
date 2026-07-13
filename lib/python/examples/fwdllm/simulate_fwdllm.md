# High-Fidelity Simulator for FwdLLM — Real↔Sim Parity

**Active build (branch `dg/sim_parity_fwdllm`).** A simulated-clock runner for the `fwdllm` example
(FedFwd / forward-gradient FL) reaching **real↔sim parity** across the **fluxtune / fwdllm / fwdllm++**
baselines — at **100% availability (syn_0)** first (Phase 1), then **unavailability** (syn_20/50/mobiperf,
Phase 2), then **beyond syn_0** (Phase 3). fwdllm has no native sim clock; the build wires flame-core's virtual
clock + availability substrate into fwdllm's variance-gated gradient loop. It reuses async_cifar10's virtual
clock, sct reorder buffer, in-flight gate, availability substrate, and parity ladder where they transfer, and
deviates where the workload demands (fwdllm aggregates **gradients** not weights; **variance-gated dynamic-K**
commit cadence; **`data_id`** progress axis; one-message-per-call grad loop; rollback across agg-goal cycles).

> ## PREAMBLE — how to maintain this doc (READ BEFORE EDITING)
> This is a **living status doc**, not a changelog. It must stay **rich but crisp**: enough to reconstruct *why*
> a decision was made, never a running history. Per-run history lives in git + the parity JSONs; the code is the
> source of truth for *what* the mechanism is. Every edit obeys:
> - **Current-truth only.** §A, the scoreboard, and the open-issues table describe the state **right now**.
>   Rewrite them **in place** — never stack dated "UPDATE" blocks. When something closes, **delete it from the
>   open list** and leave at most a one-line trace in §G (fixes) or §H (dead-ends).
> - **One line per landed item.** A fix that worked = one line in §G (**≤20 words problem + ≤20 words fix**). A
>   belief that was refuted = one line in §H. A deviation = one line in §K (anchor + rationale). No paragraphs.
> - **Keep only what teaches.** Retain a big root-cause or a conceptual correction that would otherwise be
>   re-litigated; **drop nitpicks** and mechanical scaffolding once landed (leave a pointer only if still
>   referenced). If an entry no longer changes a future decision, delete it.
> - **No contradictions.** An issue is in exactly one place: OPEN (open table) **xor** CLOSED (§G/§H one-liner).
>   Never both. Cross-link with §-anchors instead of restating.
> - **Decide forks explicitly.** When fwdllm diverges from async_cifar10, log the choice in §K and surface it in
>   the §B.1 delta table — don't silently copy or silently invent.
> - **Date hygiene, every revisit.** Any session that edits this doc sweeps the WHOLE file for prior-dated
>   narrative (e.g. "as of 2026-07-05", "(2026-07-09)"): fold what's still load-bearing into crisp, date-free
>   prose (or a one-liner in §G/§H/§K), delete what's superseded, and tag anything left unchanged-but-still-valid
>   `[checked YYYY-MM-DD]` so the next sweep re-examines it fresh rather than trusting an old tag forever. Run
>   identifiers (`run_YYYYMMDD_HHMMSS_...`) and JSON report filenames are evidence anchors, not narrative dates —
>   always keep those.
> - **Conceptual survives, mechanical gets purged.** A design choice, root cause, refuted hypothesis, or
>   real↔sim nuance is worth a permanent line — it changes future judgment. Instrumentation/plotting/test/code
>   scaffolding descriptions do NOT survive past landing, even crisply worded — git + the code ARE that record.
>   Concretely: drop pytest/test-green counts, field-by-field telemetry additions, and "which files changed"
>   narration once a fix is landed; keep the one-line problem+fix (§G) or before→after numbers that prove the fix
>   actually worked. Only OPEN/pending items earn a date stamp and expanded detail — closed ones get the terse
>   §G/§H/§K treatment or nothing at all.

> **Working checklist for every fix in this doc (already the §F ruleset — indexed here, not restated):**
> (a) Ground every claim in a metric that is actually captured and diffable — telemetry/banked-logs FIRST (§F#11),
>   logical-determinism traces over aggregate curve-matching (§F#15).
> (b) Isolate the true bottleneck, not its symptom — classify real-transport artifact vs algorithmic property
>   (§F#8), rank by blast radius / SHARED-before-per-baseline (§F#14), confirm which side (real or sim) is
>   actually divergent before tuning either (§F#6).
> (c) Design fixes from first principles at the root — no hack that moves a number without a correct mechanism
>   (§F#16); baseline-defining knobs are not parity levers (§F#3).
> (d) **Never launch an experiment run (GPU/cluster job) directly.** Print the exact command (cwd + flags) and
>   let the operator run it. Code edits, telemetry-only reads of already-banked logs, and pytest are fine
>   unattended; a new `run_sequential.sh`/`run_parity.py` invocation is not.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — parity methodology (ladder,
roles/tiers/gating, run-length budget, landed sim mechanisms); **fwdllm's rung catalog is PARITY.md §F**.
[async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) — the availability substrate
fwdllm inherits via its aggregator class chain (ClientAvailability mixin, trace-read effect path, two-ledger
discipline, starvation self-termination, A6/A7/A8/K11 ground-truth rungs).

---

## §A  Current status

**Last landed:** §M's code (event-driven sim recv/barrier redesign, see §M) — all 9 subtasks, 723 tests green.
**Live validation READ 2026-07-12** (15-min real+sim pairs, all 3 baselines, `run_20260712_175052` onward):
§M fixed fluxtune's numeric `sim_rate` but not its root cause (still open, see below), left fwdllm untouched,
and **exposed a full livelock in fwdllm_plus that is worse than the slowdown it was meant to fix** (R2
materialized, see below). Do not re-baseline at 1200s until both are fixed.

| baseline | `sim_rate` | verdict |
|---|---|---|
| **fwdllm** (sync) | **5.77** ✓ (unchanged) | healthy, no regression |
| **fwdllm_plus** (sync) | **uncomputable** ("insufficient ts data") | **LIVELOCKED, not slow** — 1 commit total, then zero progress; sim never self-stops, external watchdog kills it past the 900s budget. Root below. |
| **fluxtune** (async) | **1.0002** (was 0.97) | number improved but the mechanism didn't — `_fetch_weights` still 3.2× real, per-step vclock/wall ratio never exceeds ~1×. Root below. |

**fwdllm_plus root: `k`≠`c` in `RandomSelector` + §M's reselect-gate cache turns a dormant bug fatal.**
`RandomSelector._cleanup_recvd_ends` (`flame/selector/random.py:338`) frees at most
`min(len(ordered_updates_recv_ends), self.k)` completed trainers per cleanup call — a batch-size knob from the
selector's original n≈150/`k=5`/`c=15` design (≈33% pool turnover per round). The n=10 smoke YAMLs
(`expt_scripts/fwdllm_n10_smoke.yaml`, `fwdllm_plus_n10_smoke.yaml`) correctly rescaled `c: 10` to match
`agg_goal=10` (full-cohort barrier, "every selected trainer is required") but left `k: 5` at its old absolute
value. With a full-cohort design `k` must equal `c` — `async_oort`'s `_cleanup_recvd_ends` already drains ALL
received ends with the code comment *"min(N, agg_goal) deadlocks when K changes dynamically"* (`async_oort.py`
~line 1002), i.e. this exact bug class was found and fixed there but never ported to `RandomSelector`.
Confirmed live in **both** real and sim fwdllm_plus runs: round 1 commits 10/10, then only 5 trainers get freed
(the other 5 sit "in use" though their grad already committed), so round 2 can only select 5 → commits 5, frees
another 5 (FIFO), etc. — a permanent partial-cohort trickle. In real mode this is silently absorbed (wall-clock
ticks regardless of progress, so the run "completes" at 900s having made **exactly 1 commit in 15 minutes**).
In sim mode §M subtask 7's reselect-gate cache then **freezes** onto whichever trickle-shrunk cohort got
selected for a given `version_key=(round, iteration_per_data_id)` and never re-polls; once the cached 5 have
all sent for `iteration_per_data_id=0` (confirmed only value ever seen — it never advances) they correctly
abort re-sends, nothing new ever arrives, the barrier polls `buf_depth=0` forever, vclock stays frozen at
T_v=68s, and the app can never reach its own `max_runtime_s` self-stop — so the external wall-clock watchdog
kills it (~973s, over the 900s budget). **This is R2 (§M, below) materializing** — worse than the "suppressed
real-mode top-up" framing predicted: with `k`≠`c` the cached cohort is fully exhausted, not just partial, so
it's a full livelock, not a shrink. **Fix landed 2026-07-12, live validation pending:** `k` turned out to be
entirely redundant — grep confirms `oort`/`async_oort`/`fedbuff`/`async_random` have no `self.k` concept at
all, they run on `c` alone. Removed `k` from `RandomSelector` (constructor requirement + a vestigial
`select()` early-exit that only ever duplicated the `required_trainers` check right after it) instead of
just retuning its value — `RandomSelector` now matches the other selectors' shape exactly. `_cleanup_recvd_ends`
drains ALL received ends unconditionally (mirrors `async_oort`'s already-fixed pattern). Dropped the now-dead
`k:` line from the 4 smoke YAMLs. This stops the cohort from shrinking at all, which starves R2's cache of
anything to freeze onto. R2's general risk (cache has no exhaustion escape) is NOT yet fixed — only
neutralized for this specific full-cohort config; a future partial-cohort (`c` > `agg_goal`) baseline could
still hit it. 723/723 selector+mode tests green; live 15-min pair not yet re-run.

**fluxtune root: sim_rate≈1.0 is a wash, not a fix — `_fetch_weights` still 3.2× real.**
Per-step `vclock` vs wall-clock (all 28 `VCLOCK_PROGRESS` commits, this run) never sustains a multiplier: ratio
oscillates **0.65–1.40 around a mean of 0.96** for the entire run, with no upward trend — contrast fwdllm's
healthy sim, which ramps from 1.08× (step 1) to **4.28× cumulative by step 5** as trainers warm up. `_fetch_weights`
(`step_timing_breakdown`) is **15.29s sim vs 4.83s real (3.2×)** — essentially unchanged from pre-§M's 3.6×
(16.7s vs 4.6s); `mqtt_fetch_s` component of `trainer_phase_wall_budget` fails its own budget outright
(15.88s vs a 5.8s budget). Traced to the mechanism: `time_mode=simulated` trainers do **not** sleep their
modeled per-trainer delay (`FedSgdTrainer.py`: *"modeled delay ... = 22.000s (not slept; charged to vclock;
gpu=3.440s)"*), so a trainer sends its grad and immediately re-enters `_fetch_weights`/`channel.recv()`,
blocking in REAL wall-clock on the aggregator's own serial eval+aggregate cycle (`aggregation_compute_wall`:
`eval_s`≈8.8s + `aggregate_fedavg_s`≈1.1s real GPU work, both modes) before the next dispatch arrives — that
real dead time lands in `mqtt_fetch_s`, uncredited to vclock. **Single-trainer trace confirms it's not GPU-bound:**
one trainer's consecutive fetch→train→send→fetch cycles show **14–31s real gaps while its own GPU pass is only
~3.4s** and it does not sleep — it's genuinely blocked waiting on the aggregator to service it, 4–8× longer than
its own compute. `model_version` in the same trace stays flat for 2–4 consecutive dispatches to the same
trainer (fedbuff/agg_goal=3 batching: a freed trainer isn't redispatched until the aggregator's current batch-of-3
resolves), so the wait is plausibly "for 2 batch-mates," not aggregator compute per se — **not yet distinguished
from genuine GPU-sharing queueing (10 trainers, 8 GPUs) without added phase timers around the aggregator's
async select→aggregate→distribute cycle; needs live instrumentation, not just log archaeology.**
`vclock_fold_diagnostic` confirms a residual gap
even after K-D41's fold: sim uncredited fraction **18.2%** vs real's **14.2%** (was 21%/26% pre-K-D41) — narrower
but not closed. The overall `sim_rate`≈1.0 is these effects roughly cancelling (fewer real-transport artifacts
elsewhere), not the barrier-grace root actually closing. `cohort_sequence` SET divergence still onsets at
iter 6 of `data_id=0` (bit-identical iters 1–5, reconfirms #S1). `preferred_duration` gap **narrowed but not
re-checked this pass** (was 0.346).

### §M landed 2026-07-12 — outcome vs prediction (see roots above for detail)
- fwdllm_plus's zero-progress barrier calls: **the specific pre-§M symptom (EMA-grace empty polls) is gone**,
  but the underlying selector cohort-shrink (`k`≠`c`, pre-existing) is not — §M's cache turned it from a
  self-healing trickle into a permanent livelock. Net: WORSE than pre-§M, not fixed.
- fluxtune's `_fetch_weights`/`recv_wrapper` gap: predicted to close toward ≤1× — **did not close** (3.6×→3.2×).
- fwdllm's minor `barrier_wait_s` overrun: not re-checked this pass (fwdllm otherwise healthy/unchanged).
- **R1 (felix/async_cifar10 46/46 parity)** — still deferred, not re-tested this pass. Do this before trusting
  felix numbers again.
- **R2 (version_key SEND-reselect cache applies in both real and sim, no `self.simulated` gate)** —
  **CONFIRMED MATERIALIZED**, see fwdllm_plus root above. Worse than predicted: full livelock, not a shrink.

### Parity scoreboard
1200s pairs (PRE-§M reference, superseded by the roots above for fwdllm_plus/fluxtune — kept for fwdllm):
| baseline | pass / fail / skip |
|---|---|
| **fwdllm/syn_0** | 51 / 9 / 20 |
| **fwdllm_plus/syn_0** | 50 / 9 / 20 |
| **fluxtune/syn_0** | 50 / 11 / 18 |

15-min smoke pairs, post-§M (`run_20260712_175052` onward — NOT directly comparable to the 1200s numbers above
due to run length; fwdllm_plus's low fail count is an artifact of the livelock truncating the run to ~1 round,
not health — most rungs SKIP for lack of data, not pass):
| baseline | pass / fail / skip |
|---|---|
| **fwdllm/syn_0** | 52 / 8 / 20 |
| **fwdllm_plus/syn_0** | 47 / 5 / 28 |
| **fluxtune/syn_0** | 51 / 8 / 18 |

**Fails by blast radius (principle #14).** SHARED — all 3: `cohort_sequence` (var VALUE past bin 1, pure #N),
`v2_var_trajectory` (#N accumulation), `step_timing_breakdown` (K-D37-exempted real-only funcs +
`_force_cuda_memory_cleanup` small-N noise), `total_commits`/`terminal_state` (length-confound from each
baseline's own `sim_rate`, though fluxtune's now PASSES post-K-D41). fluxtune-only: `preferred_duration`,
`g2_grad_pool_size`. fwdllm_plus's normal fail set is currently masked by the livelock (see scoreboard note).

### STRATEGY — nail first-data-bin logical parity before any longer run
Prove parity by **logical determinism, not aggregate curve-matching**: for a matched scope the sim must take
**the same sequence of steps in the same order** as real — same trainers selected, same update-receipt order,
same aggregations/rollbacks — differing ONLY in wall-clock. **Scope to the first 1 data bin** (`--max-data-id 1`
/ `--max-bin 1`) before chasing the time dimension; isolates length-confound from genuine logic bugs.

### Logical-parity check (TIME-STRIPPED, all available bins) — CURRENT REFERENCE
Tool: `expt_scripts/logical_parity.py [--max-bin N]` — diffs the `agg_round` event stream (data_id, iteration,
receive-ordered contributors, variance decision) real vs sim with every timestamp removed. Real receive-order is
DETERMINISTIC in both real and sim by design (operator-confirmed) → exact match is the correct target. `--max-bin`
already generalizes to a sweep (e.g. `--max-bin 15`) to see WHERE a divergence onsets/grows/plateaus, not just
whether bin ≤1 passes — no code change needed for that, just a wider invocation on a longer run.

**Cohort-SIZE axis (K-D42)** answers what receive-SET/cadence can't: per-selection-call cohort *size*
(num_eligible/num_chosen). A selector that always exhausts its eligible pool (fwdllm's `random`) can pass
receive-SET/cadence at bin ≤1 while diverging on how MANY trainers were eligible per call — exactly what
happened to fwdllm_plus (§A). SKIPs for baselines with no per-iteration reselect concept (fluxtune).

2026-07-12 1200s pairs, bin ≤1 (STRATEGY's target):

| baseline | receive-SET/cadence | cohort-SIZE | verdict |
|---|---|---|---|
| **fwdllm** | 19/19 identical | 1/1 identical | LOGICAL + COHORT-SIZE PARITY |
| **fwdllm_plus** | 12/12 identical | 1/22 identical | LOGICAL PARITY, but cohort-SIZE DIVERGES — real polls the reselect gate 11–13×/iteration (mostly empty), sim 1–2× (grabs the full cohort at once). **Reframed 2026-07-12: IS a selector bug** — `RandomSelector`'s `k=5`≠`c=10` cleanup mismatch (§A), not the barrier-grace root; the trickle-in polling here is real mode absorbing the same pool-shrink that livelocks sim. |
| **fluxtune** | 8/157 identical | n/a | cohorts bit-identical iters 1–5 of `data_id=0`, diverge at iter 6 (matches #S1); iters-to-clear-bin-0 = 7/7 |

**Root (#N):** grad non-reproducibility given matched order — ~1e-3 GPU fp16 jitter, amplified by the split-half
variance ratio, flips the `var<0.3` gate at a sensitive bin (receive-ORDER itself is 41/41 identical, K-D31 — not
the cause). **Parity target:** cohort SET = HARD; `var_good`/cadence = HARD to bin 1, DISTRIBUTIONAL beyond; `var`
VALUE = SOFT. **#N is SYNC-only** — fluxtune's async cohort-SET divergence past bin ≤1 was #S1 (fixed, §G), not
#N; the residual past iter 6 is consistent with #N. See §H for the refuted "async #N" framing.

### Open issues (OPEN only — closed items live in §G/§H)
| # | issue | baseline(s) | next step |
|---|---|---|---|
| **fwdllm_plus sim livelock — `k`≠`c` selector bug + reselect-gate cache** | ROOT-CAUSED + FIX LANDED 2026-07-12 (§A): `RandomSelector._cleanup_recvd_ends` drained only `k=5` of a `c=10` full cohort per call; §M's version_key reselect cache then froze onto the shrunk/exhausted cohort. Fix: `k` was redundant (no other selector has it) — removed entirely from `RandomSelector` + the 4 smoke YAMLs; `_cleanup_recvd_ends` now drains all (matches `async_oort`). | fwdllm_plus (was fatal in sim, silent in real) | Re-run the 15-min real+sim pair to confirm the livelock is gone and real `total_commits` recovers from 1/15min. Reselect-gate cache still has no general exhaustion escape (R2 residual risk for any future partial-cohort baseline). |
| **fluxtune `_fetch_weights`/`mqtt_fetch_s` still 3.2× real** | CONFIRMED root (§A): trainers don't sleep their modeled delay in sim, so they block in real wall-clock on the aggregator's serial eval+aggregate cycle instead. `sim_rate`≈1.0 is a wash of compensating effects, not this closing. | fluxtune | Root the aggregator-side serial eval+aggregate gate — either overlap eval with the next dispatch, or credit trainers' idle wait to vclock like `_sim_model_agg_compute_time` does for the aggregator's own compute. |
| **fluxtune `vclock_fold_diagnostic` uncredited fraction (18.2% sim vs 14.2% real)** | Narrower than pre-K-D41 (was 21%/26%) but not closed — real aggregator-side wall-clock work still isn't fully reflected in vclock progress. | fluxtune | Likely the same root as `_fetch_weights` above (aggregator-gates-trainer serial wait); re-check after that fix. |
| **fluxtune `preferred_duration` gap** | Was 0.346 pre-this-pass; not re-checked against the new runs. | fluxtune | Re-check against the current runs once the `_fetch_weights` root above is fixed (may be entangled). |
| **fluxtune `total_commits`/`terminal_state`** | **RESOLVED this pass** — both now PASS (0% rel_diff, was ~19%). Kept here only as a re-regression tripwire. | fluxtune | Watch for regression when the `_fetch_weights` fix lands; no action otherwise. |
| **fluxtune `step_timing_breakdown` residual `_force_cuda_memory_cleanup`** (minor) | KS fails but means differ by only 0.01s — looks like small-N distribution-shape noise (K-D37 class). | fluxtune | Low priority; re-check if it starts moving means, not just KS. |
| **#N (var-VALUE nondeterminism wall)** | Float-nondeterminism flips the `var<0.3` gate → cohort SET (async) / var VALUE (sync) diverge past a sensitive bin. Not a sim bug. | fwdllm, fluxtune (fwdllm_plus latent) | DISTRIBUTIONAL target beyond bin 1 already covers it. |
| **#11** | Real-mode critical-path waste (`sleep(0.1)` MQTT-settle; one-grad-per-poll drain tail) — real-only, zero parity impact. | fwdllm, fwdllm_plus (real) | Deferred — needs a real run to touch (principle #8/#11c). |
| **fwdllm `barrier_wait_s` overrun** (minor) | sim > real, fwdllm-only; drain-tail/spread PASS; not re-checked this pass. New `phase_vclock_bottlenecks` (§G) independently flags `mqtt_fetch_s` here too (real 3.9s/sim 6.1s, 0% vclock-credited) — smaller instance of fluxtune's root; not yet confirmed same cause. | fwdllm | Re-check once fwdllm_plus's fix lands (shares `fwdllm_aggregator.py`); `phase_vclock_bottlenecks`'s `by_phase.mqtt_fetch_s` is now the first thing to check. |
| **felix/async_cifar10 46/46 re-confirmation (R1, §A)** | §M subtask 4 changed asyncfl's budget-fallback behavior; live parity battery not yet re-run. | felix | Deferred by request — re-run before trusting felix numbers again. |

### Next roots — ranked (correctness before time; SHARED before per-baseline — principle #14)
1. **Re-run fwdllm/fwdllm_plus 15-min pairs to confirm the livelock fix (landed, unvalidated live).** If cohort
   trickle is gone, decide whether R2's cache still needs a general exhaustion escape (residual risk only for a
   future partial-cohort config, §A) or can wait.
2. **Root fluxtune's `_fetch_weights` aggregator-dispatch gate.** Single-trainer trace rules out GPU compute
   (3.4s) as the cause of its 14-31s real wait; leading hypothesis is fedbuff batch-of-`agg_goal`=3 redispatch
   waiting on batch-mates, not distinguished yet from GPU-sharing queueing (10 trainers/8 GPUs). Needs phase
   timers around the async select→aggregate→distribute cycle (live instrumentation, not log archaeology) to
   close.
3. **Re-run felix (R1)** before trusting its 46/46 parity claim again — deferred twice now.
4. **LLM-mobile runtime trace swap (principled).** The papaya/fedbuff 4–18s trace is a modeling choice; a real
   mobile-LLM forward-grad trace would give honest headroom (`sim_rate`>1) AND set the delay regime the
   staleness fix must hold under. Divisor tuning (0.25/0.1) is NOT this — it's a diagnostic knob (principle #3).
5. Then C1/C2 convergence (distributional target) at matched `data_id` per baseline → gate to Phase 2.

### SKIP audit (19–21 skips; ~17 legit)
Legit at Phase-1 syn_0 + `random` selector: 7 availability ground-truth rungs + 4 delivery/withheld (Phase-2
effect path, not built) + 3 DynamicKC (disabled by design) + 2 oort-only (`random` baselines) + `residence`
(async telemetry). `timing_overrun` now POPULATES (no longer a skip). The 12 former rigor-gap skips (4 advance, 8
phase-timing) are un-skipped (K-D21).

---

## §B  How fwdllm differs structurally

fwdllm aggregates **gradients** (JVPs) not weights; commit cadence is **endogenous** (variance-gated dynamic-K);
progress axis is committed **`data_id`** (variance passes), not update count. Gradient values are mode-invariant
given identical input+perturbation seed, so parity reduces to **clock + selection + ordering parity plus a
variance-cadence layer**. Anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150`; force-commit cap
`max_iterations_per_data_id`; `reselect_each_iteration` (fwdllm++ per-iteration reselection); sync path
`_aggregate_grads_sync`. Full detail: PARITY.md §F.1.

### §B.1  Real↔sim design deltas vs async_cifar10 — **CURATED, KEEP CURRENT** `[checked 2026-07-09]`
Separates an **intentional fwdllm choice** from an accidental discrepancy. The §K column points at the one-line
rationale.

| # | Axis | async_cifar10 | fwdllm | Why | §K |
|---|---|---|---|---|---|
| 1 | Aggregated object | model **weights** | **gradients** (JVPs) | grad values mode-invariant → parity = clock+order+selection + variance-cadence | §F.1 |
| 2 | Progress axis | update/round count | committed **`data_id`** | cadence (updates-per-data_id) is an **output to match**, not an input | principle #2 |
| 3 | Commit cadence | fixed `agg_goal` | endogenous **variance-gated dynamic-K** | the emergent layer cifar doesn't model; V/DK/G rungs verify it | §F.1 |
| 4 | sct delay model | `send + max(gpu, D)` | `send + max(gpu, D)` (**remainder-wait, was additive**) | K-D29: real now sleeps `max(0,D−gpu)` (device wall = D, GPU hidden), so update order = per-trainer D order = deterministic & real↔sim identical. Reverses K-D2 | K-D2/**K-D29** |
| 5 | Per-eval sct | distinct, ~20× faster | **collapses to train sct** | eval lives on the aggregator; forward-grad "train" IS a forward pass (no 20× factor) | K-D3 |
| 6 | Slot residence | per-commit release | **hold slot to COMMIT** (felix port, K-D17b) — a trainer is freed only when its update commits (guards: no same-version dispatch, none while computing, none while returned-but-uncommitted) | Correct for BOTH sync & async — it is a correctness check, not a lever. fluxtune's `sim_rate<1` is a COMMIT-PATH STALL (K-D34/#15), NOT this rule. Commit RATE = throughput | K-D17b/**K-D34**, principle #4 |
| 7 | Surplus grad on rollback | carried | **carried** for async (fluxtune, c≫agg_goal); **drop** stays correct for sync (c≈agg_goal) | drop was benign only for sync; fluxtune dropped ~7/cycle → 2× passes | K-D12 |
| 8 | Async drain primitive | `_sim_recv_min` verbatim | purpose-built `_sim_recv_min_grad` / sync `_sync_sim_recv_first_k` | cifar's per-commit release + withheld paths key on WEIGHTS semantics | K-D4 |
| 9 | `time_mode` default | `"simulated"` | `"real"` (getattr fallback) | fwdllm's whole config corpus is `real`; a `simulated` default risks half-activating an unbuilt path | K-D1 |
| 10 | Cadence telemetry | n/a | **pre-mutation** cycle snapshot (`cycle_data_id`/`cycle_iteration`/`grad_pool_size`/`cached_v_size`) | post-mutation `data_id` advances before emit → off-by-one; snapshot makes V1 exact | K-D9 |
| 11 | Availability tracking (v1) | all `trace_read` | **mixed**: fwdllm unaware, fwdllm_plus `oracular`, fluxtune `client_notify`→approx `trace_read` | baselines carry different models; first-class `client_notify` deferred to Phase 2 | D1 |
| 12 | Launch tooling | single parity template (`debug_run.sh`) | per-baseline yamls + `_sim` files (`run_sequential.sh`) | different config models; both source the shared harness `examples/scripts/expt_runner.{sh,py}` | this work |

---

## §C  Baseline matrix (resolved from the launcher configs) `[checked 2026-07-09]`
*Canonical cross-cutting catalog + the planned 5-baseline restructure: [`../_metadata/BASELINES.md`](../_metadata/BASELINES.md).*
| baseline | sync/async | selector | agg | tracking / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify`→`trace_read` v1 (aware-at-select, reactive-90s) | — | 3 | disabled |
| **fwdllm** | sync | `random` | fedavg | `default` unaware (reactive-90s) | per-round | 10 (=c) | — |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` (aware-at-select via trace read) | per-iteration | 10 (yaml is source of truth) | — |

Phase order (locked): Phase-1 syn_0 → Phase-2 unavailability → Phase-3 beyond syn_0. One baseline at a time.
Stage 3 oort rungs run **only** for fluxtune; sync-barrier rungs run for **fwdllm/fwdllm_plus**.

---

## §D  Parity ladder for fwdllm (rung defs live in PARITY.md §F)
- **Reused verbatim:** Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1–A3 (+A6/A7/A8 once avail telemetry lands);
  Stage 3 S3/4, A2c + the oort stack (fluxtune only); Stage 4 T2/K6/T_mqtt; Stage 8 C1; Stage 9 budget/stop.
- **Modified** for variance-gated dynamic-K: K3a/K3b/K2/U3/K8/U2 → re-keyed to the variance-pass boundary /
  committed `data_id` (PARITY.md §F.3).
- **New** variance-cadence layer: V1–V5, DK1–DK3, G1–G2 (PARITY.md §F.4). Localize down; never fix an EMERGENT
  rung directly. `var_threshold` / `max_iterations_per_data_id` are baseline knobs, not parity levers.
- **Availability** rungs (A1–A5, A6/A7/A8/K11, withheld/abandon/starvation) inherited; apply once Phase-2 wires
  the effect path + telemetry.
- **Per-stage wall-budget instrumentation (K-D36).** New ONE-SIDED (`sim<=real`) rungs where sim should collapse
  a real-transport phase to ~0, DISTRIBUTIONAL/EQUALITY where it's genuine shared compute: `drain_wall_budget`
  (Stage 6, + `processing_wall_ts`), `trainer_phase_wall_budget` + `step_timing_breakdown` (Stage 4),
  `aggregation_compute_wall` + `cycle_model_version` (Stage 6). First live-pair run surfaced a
  `step_timing_breakdown` checker bug (K-D37, fixed).

---

## §E  Roadmap — remaining phases

**Phase 1 (syn_0) — CLOSE-OUT (near done):** #14/#1c/#13/#12c/#7 fixed or explained; fwdllm's `sim_rate` is
healthy (§A). Remaining: `sim_rate<1` for fwdllm_plus/fluxtune — root isolated to the receive-barrier
grace-timeout, fix planned (§M, §A Next roots #1); then C1/C2 convergence at matched `data_id` → gate to Phase 2.

**Phase 2 — unavailability (syn_20/50/mobiperf).** Wire the ClientAvailability effect path into the grad loop:
send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); two ledgers; reactive-90s in-flight;
starvation vclock-advance; per-baseline `avail_select_filter`. Emit `EVENT_AVAIL_CHANGE` +
`agg_belief_change`/`send_gate_wait` (builders in `flame/telemetry/events.py`) → unlocks A6/A7/A8/K11. **D3 watch:**
does a withheld/late grad roll into `cached_v` on rollback, or a stale-version late grad inflate `var` (thus
dynamic-K)? Over-instrument first. Exit: A1–A5 + A6/A7/A8 PASS; self-stops; withheld grads delivered not dropped.

**Phase 3 — beyond syn_0.** Full ladder under scarcity; bin V1/V2 by run-fraction to separate a constant mix bias
from a compounding variance-feedback loop. Exit: curves within tolerance at matched `data_id`; K8/U2 within bar;
V1/V2 binned residual flat.

---

## §F  Locked principles (from async_cifar10, carried over) `[checked 2026-07-09]`
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. **Never** put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`.** Updates-per-data_id is the dynamic-K random variable V1 validates — an output
   to match, not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold` / `max_iterations_per_data_id` are
   baseline-defining config knobs.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **DynamicKC: validate the input before the policy** (DK3 CONTROL before DK1/DK2). The controller is shared;
   don't fork it per baseline.
6. **Real is the reference only after admissibility.** A real↔sim gap has two fix directions — check whether the
   **real** input is the divergent side before tuning sim.
7. **syn_0 byte-identical gate OFF; unavailability config-gated.** Reuse the async_cifar10 flag axes; don't delete
   eligibility plumbing.
8. **Fix the concept, not the symptom — never regress a working example.** Classify a mechanism as
   **real-transport artifact** (incremental drain, `recv_fifo` timeouts, queue persistence — no sim analog; gate
   `and not self.simulated`) vs **algorithmic property**. Scope-check before editing shared code:
   `fwdllm_aggregator.py` = fwdllm blast radius; `top_aggregator.py` / shared `parity` engine / `_sim_recv_min`
   can silently break async_cifar10.
9. **Match pytest scope to blast radius.** fwdllm-only edit → `pytest tests/mode -k fwdllm`; shared parity engine
   → add `examples/async_cifar10/scripts/parity` + `tests/mode -k parity`; shared stack → full `pytest tests/`.
10. **Comment the WHY, crisply.** A conceptual choice, a real↔sim divergence + rationale, or a failure mode + why
    the fix takes its shape — one or two tight sentences. Deeper rationale → §K.
11. **Telemetry-FIRST, then instrument, then (rarely) run.** (a) Validate/refute from telemetry ALREADY ON DISK
    first — name the exact field/line. (b) Ship telemetry + plot + pytest IN THE SAME CHANGE as any new mechanism.
    (c) A run is justified only to observe an EMERGENT quantity no stored telemetry can yield — then run the
    SHORTEST length that exhibits it, smoke first, one mechanism per round. (d) fluxtune surfaces roots first.
12. **Consult PARITY.md vclock rules BEFORE any sim-clock change.** Clock is a monotone `max`
    (`vclock = max(vclock, sct)`); NEVER put overhead on it; `sct = send + max(gpu,D) + leg` (leg NOT on
    `trainer_speed`/utility); sync charges MAX-of-K sct, async the K-th-fastest; the sim SKIPS real waits and
    reconstructs order from sct (`SimReorderBuffer`).
13. **The vclock is virtual wall-time; the sim MUST produce SPEEDUP (`sim_rate = vclock/wall ≥ 1`).** The vclock
    advances exactly as the real wall-clock WOULD after the trainer's simulated waits — but faster, because the sim
    **elapses through** the modeled time remaining after GPU compute instead of actually waiting it out. fwdllm
    nuance: the forward-grad "train" is a REAL GPU pass (~2s) that MUST run in sim for grad mode-invariance — that
    ONE GPU pass (parallel across trainers) is the only irreducible real wall; transport/inter-round/delay waits
    are vclock jumps, never process sleeps. `sim_rate < 1` means the sim under-charges the vclock OR is stalling on
    a real wait it should skip. Emit it every run (`[VCLOCK_PROGRESS]`).
14. **Correctness before speed; SHARED roots before per-baseline.** Fix major logical-correctness divergences
    (wrong selection order, wrong receive order, wrong cadence) before any throughput/wall tuning. Rank a root by
    blast radius: a bug that fails rungs across ≥2 baselines outranks a single-baseline one — fix the shared cause
    once, then narrow to per-baseline residuals. Never chase an aggregate-curve rung while a logic rung is red.
15. **Logical determinism is the parity definition.** For a matched scope the sim must take the SAME sequence of
    steps in the SAME order as real — same trainers selected, same order of update receipt, same aggregations and
    rollbacks — differing ONLY in wall-clock. Prove this on the **first data bin** (`--max-data-id 1`: multiple
    iterations + variance passes + aggregations, short + diffable) before extending length. A matched, ordered
    trace is the proof; a matched summary statistic is not.
16. **Do the right thing — no hacks.** Ask as many conceptual questions as it takes to understand the real↔sim
    divergence, but solve it at the root. A hack that moves a number without a correct mechanism is a regression in
    disguise — it will not close parity and it will mask the real bug. When unsure, stop and ask.
17. **`version_key` is the ONLY version-identity vocabulary — enforce it everywhere, no legacy scalar path.**
    Pre-K-D39, version/staleness/no-repeat identity was a bare `model_version` int in some places and an inconsistent
    3-tuple in others; K-D39 unified it to one shared `version_key` property (2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`) across trainer + aggregator + selector, both async_cifar10 and fwdllm.
    Any NEW code that compares/stamps a version must go through `version_key` (or explicitly document why it reduces
    to one component, e.g. staleness's `model_version`-only diff — iteration isn't a "how many global updates
    behind" axis, so dropping it there is a deliberate reduction, not a parallel legacy path). Do not add a
    bare-scalar shortcut "for now" — it silently reintroduces the pre-K-D39 inconsistency and is easy to miss since
    it type-checks fine. No backward-compatibility shim for the old scalar/3-tuple forms is needed or wanted.
    #S1's diagnostic telemetry (`dispatch_version_key`/`agg_version_key_at_commit`/`version_bump_census`) audits
    this directly by carrying the full tuple, not a reduction, end-to-end.
18. **Don't blame GPU/resource contention at n=10 — it was checked and refuted once already (§H, "GPU
    under-provisioned" dead-end) and won't apply until experiments run at ≥100-trainer scale, and even then
    only if trainer/GPU load-balancing turns out uneven.** At current scale, ANY unexplained real-wall-clock gap
    (a wait, a slow phase, a sim/real mismatch) should be assumed closeable by **measurement, not guessing**:
    add a wall-clock + vclock phase timer around the suspect stage (see the phase-timing telemetry framework,
    once built) and read what it says before proposing a mechanism. A hypothesis that isn't backed by a named
    field in already-captured telemetry is not a root cause yet (principle #11a).

**Open design decisions:** D2 (avail telemetry port — Phase 2); D4 (eval-delay factor — confirmed ~1× train
cost, K-D3). D1/D3/D6 resolved (§K) — D3 ("sim must reproduce real's grad staleness") closed 2026-07-11:
`#S1` root-caused + validated it as a REAL-side dispatch bug (§G, K-D40), not a sim-fidelity gap.

---

## §G  Fixes landed (what worked — ≤20-word problem + ≤20-word fix; do not redo)
- **No vclock/wall phase telemetry outside hand-scraped `VCLOCK_PROGRESS` log lines, fwdllm-only.** Fix:
  `vclock_now` property (aggregator + trainer base classes, real=`None`/sim=float, one `if self.simulated`
  each) wired through `_phase()`/`timer_decorator`/`build_step_timing`, both examples, both modes; 15 scattered
  `x if self.simulated else None` sites collapsed to the property; the 5 previously-hand-timed-only fields
  (`gpu_compute_s`/`pre_train_s`/`post_train_s`/`aggregate_fedavg_s`/`eval_s`) + `mqtt_fetch_s` (direct-dict-write,
  bypassed `_phase()`) now carry it too. New `analyze_run.py::phase_vclock_plots` (per-run) + `checks.py::
  phase_vclock_bottlenecks` (real-vs-sim, one flag = sim costs wall beyond real AND its own vclock doesn't
  credit the gap) — on the banked fluxtune/fwdllm pairs, immediately and correctly re-found `mqtt_fetch_s` as
  the sole bottleneck (fluxtune: real 4.6s/sim 15.9s/0% credited; fwdllm: real 3.9s/sim 6.1s/0% credited, a
  smaller instance of the same residual, previously only "likely resolves as a side effect of §M"). 849 tests
  green.
- **K-D43 G1 grad-norm rung permanently SKIPping** — assumed trainer-side changes needed; aggregator already
  receives raw per-contributor gradients. Fix: compute L2 norm aggregator-side, emit `agg_round.grad_norm`.
  Not yet run-validated.
- **K-D41 fluxtune `sim_rate<1` — `aggregate()`'s real per-cycle wall never credited to the vclock.** Fix:
  `sim_model_agg_compute_time` flag folds it in every cycle, config-gated OFF=byte-identical. **VALIDATED
  2026-07-12: `sim_rate` 0.89→0.97**, uncredited fraction narrowed real 14.8%/sim 18.3% (was 21%/26%).
- **K-D42 fwdllm_plus reselect-telemetry gap** — SEND-reselect never threaded `data_id` through
  `channel.ends()`, and `random.py` sniffed a dead pre-K-D39 3-tuple shape (silently never fired). Fix:
  thread `data_id`, fix the 2-tuple parse. **VALIDATED: `selection_detail.rel_diff_chosen` 0.218→0.053**
  (near-pass), `avail_timebase` now PASSES — but exposed a distinct root, see §A/§M.
- **`_real_intrinsic_clock` async-cycle-overlap checker artifact** inflated fluxtune's `total_commits`/
  `throughput`/etc. to 76–86% rel_diff (cumulative-summed overlapping async cycles as if sequential). Fix:
  falls back to raw wall for async (same fallback async_cifar10 already uses). Validated: `total_commits`
  rel_diff 0.857→0.167.
- **fluxtune sim Oort speed-penalty never binds** — `calculate_round_preferred_duration` scored a transient
  singleton dispatch batch (`filtered_ends`, size 1 — async dispatches one freed trainer at a time), not the
  full registry; reference Oort scores `client_list=self.totalArms.keys()` — an unfaithful port, same class
  as K-D40. Fix: widen to the already-threaded `connected_ends`. **Real gap persists post-fix** (§A,
  `preferred_duration.frac_diff` 0.346 on a clean pair) — the fix is correct but doesn't fully close the
  rung; not yet re-root-caused.
- **`convergence`/`convergence_loss` checker bug — round-keyed on fwdllm's static `round`.** Collapsed every
  eval to one entry, comparing real's LAST checkpoint vs sim's LAST at mismatched `data_id`. Fix: re-key by
  `data_id` when present (`_eval_progress_axis`), `round` fallback for async_cifar10. Validated: fluxtune
  `avg_accuracy_diff` 0.1509→0.0739.
- **`step_timing_breakdown` `train_with_data_id` wrapper gated on a nested real-only sleep** — wraps the
  already-exempted `_emulate_training_delay`, so its own KS=1.0 was structurally guaranteed. Fix: added to
  `_STEP_TIMING_REAL_ONLY_FUNCS` (K-D37 pattern).
- **`#S1` fluxtune staleness — busy-trainer residence violation (K-D40).** Real released a busy trainer's
  re-pick guard on RETURN not commit; `async_oort`'s 90s abandon had no liveness check → real dispatched
  fresh work to still-busy trainers. Fix: unconditional hold-to-commit + configurable `send_timeout_wait_s`
  (300 for fluxtune). Validated: `staleness` rung PASSES, holds at the same iter-6 cohort-SET onset across
  every independent re-run since (§A).
- **fluxtune selection-mix collapse (RC1+RC3).** Selector was blind to modeled delay D, and a same-tuple
  double-pick starved the pool via a stale prune. Fix: stamp the speed signal from modeled duration; re-key
  the no-repeat guard by `version_key`, pruned only on advance (K-D39).
- **§M `version_key` unification (K-D39).** version/staleness/no-repeat used 3 inconsistent shapes across
  trainer/aggregator/selector. Fix: one shared `version_key` property + vocabulary.
- **#7 fwdllm_plus eligible-count gap** (real 5.3 vs sim 10.0 mean eligible) — root-caused, NOT a sim bug: real's
  first reselect after a full-cohort dispatch always sees eligible=1 (drain-then-refill), sim's barrier drain
  always sees eligible=10. Real-transport artifact, no code fix needed.
- **`step_timing_breakdown` real-only-func exemption (K-D37).** Rung DIST-gated real-only sleeps as if shared
  compute. Fix: `_STEP_TIMING_REAL_ONLY_FUNCS` reported but `gates_ok=False`.
- **Stale SHARED-fail list retired.** `gpu_budget_*`/`utility`/`v5_variance_pass_ratio`/`overhead_residual`(sync)
  no longer reproduce — superseded by since-landed fixes; don't re-investigate unless they reappear.
- **Per-stage wall-budget instrumentation (K-D36).** No rung caught "sim a little slower at one stage." Fix:
  `drain_wall_budget`/`trainer_phase_wall_budget`/`step_timing_breakdown`/`aggregation_compute_wall`.
- **#N bin-7 checker fix (K-D35).** `cohort_sequence` conflated 4 targets under one bin cap. Fix: SET
  uncapped/HARD, CADENCE/VAR/ORDER capped to bin 1.
- **#12c sync `sim_rate` (delay-factor).** No delay-headroom starved the vclock. Fix: `--delay-factor 1` → sync
  `sim_rate` 0.94→2.9-3.0.
- **K-D31 validated.** Bin-1 cohort order is bit-exact once benign delay-ties are canonicalized.
- **Aggregator GPU pin (K-D33).** Aggregator eval contended with trainers on GPU 0. Fix: pin to a dedicated GPU.
- **#13 drain stall (K-D28/b/c).** A stuck straggler re-fired the full 30s `RECV_TIMEOUT` every cycle. Fix: felix
  stuck-end eviction + recv-grace + ready-gating → `sim_rate` 0.06→0.30, 30s stall gone.
- **#1c R1 two-ledger bridge (K-D27/b).** Selector eligibility never consulted the aggregator's virtual in-flight
  set. Fix: agg maintains `_sim_pending_commit`, bound live into the selector's filter.
- **#14 MQTT join-notify startup race.** `join()`'s fire-and-forget notify could drop before `on_connect`. Fix:
  `_wait_for_connect()` before subscribe+notify.
- **#6 clock-rate anchor (K-D25).** Checker compared sim-vclock against real's FULL wall (carries a localhost
  transport artifact). Fix: anchor real on its own `intrinsic_span_s`.
- **Phase-4 stopping rule (K-D24).** `sim_wall_ceiling_s` truncated a real-compute sim. Fix: decouple to
  `max_runtime_s × 20`.
- **Residence: commit-then-carry + felix realign (K-D12/14/17b).** Boundary-drop stranded async grads; slot-on-
  return undercounted in-flight. Fix: carry the surplus, hold the compute slot to COMMIT.
- **Clock family re-keyed to `data_id` (#2).** Rungs keyed on `round` mismeasured fwdllm's variance-pass progress.
- **Foundational sim mechanisms (K-D2-5/9/11).** `sct = send + gpu + D`, no-sleep sim path, sct reorder buffer +
  in-flight gate + agg-goal rollback cleanup, V1-V5/DK1-3/G1-2 rungs.
- **Scaffolding (landed, pointer only).** Pre-run instrumentation (K-D21); availability params end-to-end (K-D22);
  Phase-2 skips (K-D23); `staleness_policy` wired from config (K-D13/15).

---

## §H  Dead-ends & corrections — do NOT retry
- **"fluxtune sim Oort speed-penalty never binds because `filtered_ends` is diluted with never-yet-returned
  trainers, which default to the `calculate_round_preferred_duration` 60s HACK placeholder and inflate `pref`
  above every real duration."** REFUTED by direct computation (`round_threshold=10`, N=10 → percentile index=1,
  the 2nd-smallest sorted entry): since real durations (8–36s) always sort below the 60s placeholder, `pref`
  only lands on a placeholder when **fewer than 2 of 10 candidates have EVER returned a grad** — a cold-start-only
  edge case, not something that persists across a full run. Doesn't explain `system_util`≡1.0 across all 2845 sim
  samples. *Lesson:* the 60s-default HACK is real and worth knowing about, but check the actual `filtered_ends`
  population size/composition (or just read the new `round_preferred_duration_s` telemetry) before assuming
  dilution explains a SUSTAINED non-binding pattern — the arithmetic doesn't support it past the first ~2 commits.
- **"fluxtune's async cohort-SET divergence is an #N nondeterminism wall / a GPU-vs-D headroom collision, closed by
  more delay headroom."** REFUTED by the `run_20260710_2242` `--delay-divisor 0.25` diagnostic. With D≫gpu (sct
  104–144s vs gpu 3.77s) the SET still diverges at the same iter (~8), so headroom is not the cause; and the
  divergence is systematic + delay-coupled (`v2` gap 5%→11% as delay 0.5→0.25), not random fp16 jitter. Cohorts are
  bit-identical iteration-for-iteration — the real root is **#S1** (real dispatching to busy trainers → staleness
  bias → cadence desync → SET misalignment; root-caused + fixed + validated, §G/K-D40). *Lesson:* when async cohorts diverge but
  cadence/first-cohorts match, suspect the variance INPUT (staleness/grad values), not the selector; amplify with
  a diagnostic knob to expose it.
- **fluxtune commit-path stall — three superseded framings (same investigation; final root = COMMIT-PATH
  STALL, K-D34).** (1) "GPU-PIPELINING loss; keep gate, decouple dispatch" — WRONG, felix's arrival gate is
  INERT (gate_holds=0). (2) "re-dispatch on physical RETURN to keep GPUs busy" — WRONG, fedbuff never re-hands a
  returner the same version, and real's 3.37 concurrency is a duty cycle `gpu/max(gpu,D)`, not under-use. (3)
  "hold-to-commit is a SYNC barrier / over-restrictive; replace with model-advance re-dispatch" — WRONG,
  hold-to-commit is a CORRECTNESS check (freed only on commit; guards no-same-version / not-while-computing /
  not-while-returned-uncommitted; §B.1 row 6). *Final root:* the commit path STALLS (drain gate blocks real wall
  on PHANTOM `_sim_inflight_expected` entries) → correctly-held trainers idle. Fix = fast/non-stalling commit
  path; hold-to-commit untouched. *Lesson:* commit RATE is the throughput lever, not the residence rule.
- **"The sync cadence break is a sim ORDER bug (order → split-half var → RNG desync) — exact cadence parity is
  achievable once order matches."** CORRECT for bin ≤1, REFUTED for the full run. K-D31 made
  receive-ORDER 41/41 identical, yet fwdllm cadence STILL breaks at bin 7. Root is grad NON-reproducibility given
  matched order: ~1e-3 GPU fp16 jitter, amplified by the split-half variance ratio, flips the `var<0.3` gate at a
  sensitive bin. *Lesson:* exact `var` is only a valid target ≤bin 1; beyond it the target is DISTRIBUTIONAL. Don't
  chase exact cadence past the nondeterminism wall (principle #16) — it reds on float noise, not a bug.
- **"The fluxtune runs are GPU under-provisioned (2–4 of 8 GPUs, 5 trainers/GPU) → contention is the root."** WRONG —
  a misread of `gpu=4.9s` (GPU compute SECONDS in the delay log) as device IDs. The spawn table is authoritative:
  **8 GPUs, balanced round-robin** (`spawner.py:305` `(tid-1)%num_gpus`), CPU `sched_setaffinity` 1 core/trainer,
  threads capped. Only structural imbalance is 10 trainers > 8 GPUs (GPU 0,1 carry 2 each). Compute IS mode-invariant
  at the floor (min 2.4s both modes). *Lesson:* verify a "device id" is a device id; confirm pinning from the spawn
  table, not a grep of timing logs.
- **"#13 step 4 (freed-slot staggered re-dispatch) closes the residual 11-12s drain holds."** NEUTRAL/REFUTED
  (K-D28d): `sim_rate` 0.289→0.274, holds persisted. The holds are the drain correctly waiting (strict sct order)
  for the earliest-sct in-flight straggler while higher-sct grads buffer — INHERENT to real-GPU + strict-order +
  trickle dispatch, not a dispatch-bunching artifact. Feeding an OLD freed-slot vclock as `send_ts` makes the gate
  expect trainers even earlier → more holds. *Lesson:* the residual after steps 1-3 is **#12c** (no delay-headroom),
  NOT the drain. Code kept flag-gated OFF; a reworked attempt would stamp the ACTUAL dispatch vclock.
- **"K-D19 fixed fluxtune R1 / the NONE reset is a red herring."** REFUTED (K-D26). R1 stayed 60.4%: K-D19 fixed
  the wrong release path and dismissed the NONE hypothesis by checking the *selection filter*, but `all_selected`
  MEMBERSHIP was deleted upstream (90s wall-timeout + async_oort reading `KEY_END_STATE=NONE` as "left").
  *Lessons:* (i) an R1 fix isn't done until the smoke banks R1≤2%; (ii) when a guard "should exclude" but doesn't,
  check whether the member is *deleted* upstream, not just whether the filter reads it.
- **"Free the compute slot on physical RETURN" (K-D16 Option-A).** WRONG for virtual time — a returned-but-
  uncommitted trainer is still in flight until the vclock reaches its sct; freeing undercounted `in_flight` 3×.
  Reverted to hold-to-COMMIT (K-D17b). (The K-D16 re-run also DEADLOCKED, K-D17: drain gated on channel RECV not
  the buffer; triplet stamped at DISPATCH froze the pool pre-commit.)
- **"Drop stranded grads at the agg-goal boundary" (K-D6).** REVERSED for async (K-D12) — the "|selected|≈agg_goal"
  premise holds only for sync; fluxtune (c=10≫agg_goal=3) dropped ~7/cycle → 2× passes. Drop stays correct for sync.
- **Adding a scalar overhead to the vclock to close #6.** Forbidden (principle #1/#12) — the vclock is
  `max(vclock, sct)`. Fold only genuine unmodeled compute (eval_s, straggler spread); artifacts stay off. #6 was a
  checker-anchor bug (K-D25), not a missing fold. (Including FedAvg in `intrinsic_span_s` over-charged it → excluded.)
- **Tuning `var_threshold` / `max_iterations_per_data_id` to close a cadence gap.** Rejected — baseline-defining
  knobs, not parity levers. A cadence gap is an upstream set/order/clock divergence.
- **"sim wall ≈ real, comparable" / "D=0 smoke ⇒ clock fails are artifacts".** WRONG — always check avg in-flight
  concurrency (not just pass counts); the runs are D>0 and the fails traced to `round`-vs-`data_id` keying (#2).

---

## §K  Deviation log — one line per decision (anchor + rationale; referenced from §B.1/§G)
- **K-D1** — `time_mode` default `"real"` (not cifar's `"simulated"`): fwdllm's whole corpus is `real`; a
  `simulated` default risks half-activating an unbuilt path.
- **K-D2** — additive `sct = send + gpu + D` (not `max`): fwdllm's real mode slept D on top of GPU time.
  Reversed by K-D29 (remainder-wait) once determinism required it.
- **K-D3** — per-eval sct collapses to the train sct: eval lives on the aggregator; forward-grad "train" IS a
  forward pass (no 20× factor).
- **K-D4** — purpose-built `_sim_recv_min_grad` (not `_sim_recv_min` verbatim): cifar's per-commit release +
  withheld paths key on WEIGHTS semantics.
- **K-D5** — slot release + buffer clear on the AGG-GOAL boundary (not per-commit): a `data_id` spans many cycles
  with rollbacks; per-commit release would strand a re-contributing trainer.
- **K-D9** — cadence telemetry is a pre-mutation cycle snapshot: post-mutation `data_id` advances before emit,
  which would off-by-one V1.
- **K-D11** — `ends_not_selected_yet` "commit-1-per-pass" clamp gated real-only: a real-transport draining
  discipline; the sim barrier is single-pass.
- **K-D12** — fluxtune async: commit-then-CARRY the surplus + hold residence (reverses K-D6); drop stays correct
  for sync (c≈agg_goal).
- **K-D13/K-D15** — fluxtune `staleness_policy = fedbuff` staleness-weighted accept (was silently `none`); set
  identically real+sim.
- **K-D14** — R1/W1 sourced from an ECHOED per-contribution interval, not the agg's per-end dispatch stamp
  (overwritten on re-dispatch, exactly when residence breaks).
- **K-D17b** — hold the compute slot to COMMIT (felix port): `len(selected_ends)` = virtual-time in-flight.
  Confirmed correct for BOTH sync and async (K-D34) — a correctness check, not a throughput lever.
- **K-D21** — pre-run instrumentation A–E landed; un-skipped the 12 rigor-gap rungs.
- **K-D22** — availability params respected end-to-end; Phase-1 syn_0 default; print==run.
- **K-D24** — Phase-4 ceiling decouple (×20) + B1/B2 sct folds; fixed root S1.
- **K-D25** — #6 was a CHECKER-ANCHOR bug, not a sim under-charge: real's clock-rate rungs carried a localhost
  transport artifact. Fix: agg emits `intrinsic_span_s`, rungs anchor real on it.
- **K-D26** — #1c R1 root-caused to a physical-wall vs vclock desync in `async_oort`'s re-pick guard
  (`all_selected`). Fix: `_abandon_clock_now()` runs the 90s timeout on the vclock in sim. Corrected K-D19;
  sustained NONE-delete path → deeper root K-D27.
- **K-D27/b** — the two-ledger split: async_oort's selection eligibility never consulted the aggregator's virtual
  in-flight truth. Fix: agg maintains `_sim_pending_commit` felix-style, bound live to the selector's filter;
  `outstanding = inflight ∪ buffer` (K-D27b — NOT `− _sim_committed`, which dropped re-dispatched-after-commit
  trainers).
- **K-D28/b/c** — #13 drain-stall felix port: stuck-end eviction + recv-grace floor + probe-ceiling/ready-gating +
  direct `drain_ready` ingest. `sim_rate` 0.06→0.30, 30s stall gone.
- **K-D28d** — #13 step-4 freed-slot staggered re-dispatch: IMPLEMENTED but NEUTRAL → flag DISABLED. Residual
  holds are inherent strict-sct-order straggler waits, not a dispatch artifact.
- **K-D29** — REMAINDER-WAIT delay model (reverses K-D2): deterministic commit order needs a deterministic
  per-trainer arrival order, which flat-additive `gpu+D` didn't give. Fix: real sleeps `max(0, D−gpu)` so device
  wall = D (GPU hidden); sct = `send + max(gpu, D)`; order = D-order = real↔sim identical. Overrun (gpu>D) flagged
  as `training_overran`.
- **K-D30** — full-cohort determinism gate + `timing_overrun` signal: selection set/sequence rungs gated to a
  trivial pass for every stochastic selector, so fwdllm's syn_0 selection was never checked. Fix: enforce EXACT
  whenever `num_chosen==num_candidates`.
- **K-D31** — canonical `(D, trainer_id)` cohort commit order: two trainers sharing a registry delay could swap
  receive order (benign, same split-half) but flagged the exact-order rung. Fix: canonicalize by `(D, trainer_id)`.
- **K-D32** — fluxtune JVP perf optimizations (`jvp_perf_opt`, §L; config-gated, bit-identical): trainable-only
  finite difference + drop 3 diagnostic-only forward passes + reuse cached JVP → fluxtune −37% compute.
- **K-D33** — aggregator GPU pin: eval defaulted to GPU 0, contending with trainers. Fix: pin to a dedicated
  (idle/least-loaded) GPU.
- **K-D34** — fluxtune commit-path stall (supersedes 3 mis-framings, §H): the drain gate blocked real wall on a
  PHANTOM `_sim_inflight_expected` entry (stamped-expected-at-dispatch but idle-in-recv). Fix
  (`sim_compute_truthful_gate`, default off = byte-identical): skip any expected entry whose last dispatch is
  older than `sim_gate_compute_cap_s`. Validated: `STUCK_EVICT=0`, 30s failsafes gone. Residual (vclock omits
  `aggregate()` compute wall; GPU≈D no-headroom) re-diagnosed by K-D38 as headroom, not a vclock fold.
- **K-D35** — bin-7 checker fix: `cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER under one bin cap,
  demanding exact cadence past the nondeterminism wall. Fix: SET stays HARD/uncapped, CADENCE/VAR/ORDER cap to
  bin 1. n=10 pair shows the var-VALUE wall is n-scale-sensitive, not fixed at ~bin 7.
- **K-D36** — per-stage wall-budget instrumentation: no rung caught "sim a little slower at one stage" (only
  whole-run `sim_rate`/two-sided KS). Fix: `drain_wall_budget`/`trainer_phase_wall_budget`/`step_timing_breakdown`/
  `aggregation_compute_wall`, one-sided or distributional per phase.
- **K-D37** — `step_timing_breakdown` real-only-func exemption: the rung DIST-gated real-only sleeps
  (`_emulate_training_delay`/`pause_execution`) as genuine shared compute. Fix: report but don't gate on them.
- **K-D38** — fluxtune `sim_rate<1` root RE-DIAGNOSED = GPU-vs-D headroom (supersedes K-D34's "vclock fold"
  framing): min registry delay (4s) ≈ JVP GPU (3.86s) → 11% overrun → async fastest-3 cohort is a GPU coin-flip,
  not deterministic D. Fix = config headroom (`--delay-divisor 0.5 --num-gpus 10`); principled fix = a real
  LLM-mobile runtime trace (§A Next roots #1). Also: `training_delay_factor` clarified as a DIVISOR (<1
  lengthens, was documented backwards). **PARTLY SUPERSEDED (see #S1/§H):** the `run_20260710_2242` 0.25 diagnostic
  showed that at 0.5 headroom is already adequate and the cohort-SET divergence is NOT a D-collision coin-flip but
  the #S1 staleness bug; the compute-floor + divisor-clarification parts of K-D38 still stand.
- **K-D39** — §M `version_key` unification: version/staleness/no-repeat were named + compared inconsistently
  across trainer/aggregator/selector/async_cifar10. Fix: one shared `version_key` property
  (`syncfl/top_aggregator.py` base → `(round, 0)`; fwdllm overrides → `(model_version, iteration_per_data_id)`;
  asyncfl inherits it unoverridden) — folds RC3's 3-tuple guard into `(model_version, iteration)` once
  `model_version` bumps unconditionally per data-bin (`inc_model_version_per_data_id` purged). Also fixed a
  latent cross-round dedup false-abort in the trainer, and added the same optional no-repeat plumbing to sync
  `oort.py` for symmetry (left unwired — no driving bug there). Landed + tested.
- **K-D40** — `#S1` busy-trainer residence fix: `_release_end_on_return` collapsed to ONE unconditional
  check (hold to commit whenever `inflight_residence` is set — sync/async, real/sim alike; renamed from
  `sim_inflight_residence`, the "sim_" prefix was misleading once the flag governs all four); `async_oort`'s
  abandon timeout became the configurable `send_timeout_wait_s` (default 90 = byte-identical, fluxtune sets
  300 — its rounds run up to ~108s incl. connect warmup, past the old constant tuned for felix's shorter
  CNN/speech rounds). Validated 2026-07-11 (§G, §A) — `staleness` rung passes, iters-to-clear-bin-0 exact.
- **K-D41** — fluxtune `sim_rate<1` REVISED root (supersedes K-D38's compute-floor framing at the official
  0.5-divisor basis): `aggregate()`'s genuine per-cycle real wall was never folded into the vclock (only
  `eval_s` was). Fix: `sim_model_agg_compute_time` flag, same pattern as `sim_model_eval_time`. Not yet
  run-validated.
- **K-D42** — fwdllm_plus SEND-reselect call omitted `agg_version_key`/`data_id` from `channel.ends()`,
  and `random.py`'s extraction sniffed a 3-tuple shape dead since K-D39's 2-tuple `version_key`
  unification (silently never fired for any real caller). Fix: thread `data_id` as its own kwarg through
  `channel.ends()`; `random.py` reads the correct 2-tuple. Unblocks localizing the fwdllm_plus `sim_rate`
  regression via `logical_parity.py`'s new cohort-size axis. Not yet run-validated.
- **K-D43** — G1 grad-norm rung wired (§G above); no trainer-side change needed, contrary to the initial
  read. Not yet run-validated.

*Retired/superseded anchors (kept only as pointers): K-D6 (→K-D12), K-D7/K-D8/K-D10/K-D16/K-D18/K-D19/K-D20/K-D23
— landed scaffolding or corrections, folded into §G/§H; see git history for detail.*

---

## §L  Forward-grad JVP compute profile & retained fluxtune optimizations
*(tool: `scripts/profile_jvp_opt.py` — reuses real `create_model` + `calculate_jvp`; distilbert-base
+ AdapterHub adapters, batch 8, seq 192, A40, fp16. Absolute ms are a CLEAN single-trainer profile; the real run
is ~10× from GPU contention across the 10 concurrent trainers, but pass-counts/ratios/memory transfer.)*
`[checked 2026-07-09]`

**Mechanism.** Forward-grad trains via a **central finite-difference JVP** (`fwdgrad_utils.calculate_jvp`): each
perturbation = **2 forward passes** `f(θ±hv)`, h=0.01, autocast+no_grad → `jvp=(f(θ+hv)−f(θ−hv))/2h`. **fluxtune**
selects the best of `perturbation_count`(=10) perturbations by |jvp| (2P=**20 passes**); **fwdllm/sync** selects
by cos-sim (**0 forward passes**) + 1 final JVP. Only **~1.5% of params trainable** (bottleneck adapters in all 6
layers + head, 1.04M/67.4M); backbone frozen.

| path | fwd passes | ms/batch (clean) |
|---|---|---|
| sync fwdllm (current) | 5 | 50 |
| **sync fwdllm (opt)** | 2 | **16 (−68%)** |
| fluxtune P=1 (opt) | 2 | 16 (== sync) |
| fluxtune P=5 (opt) | 10 | 80 |
| fluxtune P=10 (current) | 25 | 251 |
| **fluxtune P=10 (opt)** | 20 | **159 (−37%)** |
| backprop ref (1 fwd+1 bwd) | — | 17 |

- **Compute vs sync:** fluxtune = `2P × per-pass` → **10× sync at P=10**, linear in P, **equals sync at P=1**.
  JVP-selection is the entire fluxtune surcharge; sync's cos-sim selection is free.
- **Memory:** forward-grad peak is **FLAT in P** (~3.2–3.4 GB = model + one held forward; **no autograd graph**).
  Backprop stores activations (3.71 GB). fluxtune's extra JVP inferences cost **TIME, not memory** — same
  footprint as sync (forward-grad's design tradeoff: many cheap forward passes, no backward, low memory).
- **Per-pass:** full-param FD 10.0 ms; trainable-only FD 7.95 ms.

**RETAINED — bit-identical (fidelity-preserving; real↔sim parity untouched):**
1. **Trainable-only FD** — skip `p−h·0=p` on the 98.5% frozen params inside `calculate_jvp`: **1.26× + −251 MB**,
   `max|Δjvp|=0`.
2. **Drop the 3 diagnostic-only forward passes** (`_train_one_batch:646-648`, loss before/after-update logging —
   never feed grads/telemetry) **+ reuse the winner's cached JVP** (`:645`, fluxtune): fluxtune 25→20, sync 5→2.

Combined: **sync −68%, fluxtune −37%**, all bit-identical → should clear the `delay_factor=1` overrun (4.2s →
~2.6s < min cohort D 4.0s) WITHOUT touching fidelity. **LANDED (K-D32), fluxtune-only & config-gated**
(`jvp_perf_opt`, default false = byte-identical; true in both fluxtune yamls; sync untouched).

**NOT retained — change fidelity:**
- **vmap-batching the perturbations** — **2.0×** (biggest single win), mathematically exact (fp64 seq==vmap
  BIT-IDENTICAL, deterministic → *would* be real↔sim safe) BUT differs ~5% from sequential in fp16/fp32: the FD
  subtracts two O(1) losses (catastrophic cancellation floors precision), so any reduction-order change
  re-baselines the trajectory. Excluded per the fidelity bar; available if a re-baseline is accepted.
- **Forward-mode AD** (exact JVP) — slower (0.5×, needs eager attention; not impl for SDPA) + different math.
- **`perturbation_count`↓** — the direct lever, but changes the baseline algorithm.

---

## §M  Sim receive/barrier redesign — event-driven, zero-hardcoded-wait `[CODE LANDED 2026-07-12, live-run VALIDATION pending]`

**Status: all 9 subtasks landed.** One shared cache (`syncfl.TopAggregator._sim_known_delay_s` +
`_note_sim_known_delay`/`_sim_recv_timeout_s`) replaces the deleted `_sim_recv_grace_s`/`_note_sim_fill`/
`_sim_fill_ema`/`SIM_RECV_GRACE_FLOOR_S`/`SIM_RECV_GRACE_FACTOR`/`_sim_trainer_budget`/`_sim_budget_min`/
`_sim_budget_running_mean`/`_sim_budget_n`/`MessageType.TRAINING_BUDGET_S` (zero remaining call sites).
`tests/mode tests/selector` green (723 passed). **Not yet done:** subtask 9's live real+sim smoke
validation — see §A for expected-fixed items and regression risks now that runs are launching.

**Motivation.** The 2026-07-12 1200s re-baseline (§A) measured **829.6s of fwdllm_plus's 1237s sim wall (67%)**
burned on barrier calls that returned **zero new grads** (97/131), and fluxtune's `_fetch_weights`/`recv_wrapper`
showing sim **3.6× slower** than real (16.7s vs 4.6s mean) — both trace to the same root: sim's receive/barrier
wait is bounded by an arbitrary, reactive, easily-undershooting ceiling instead of genuine per-trainer knowledge.
Operator directive: eliminate wall-clock waits that aren't waiting on a real event — the simulator should either
know exactly how long to wait (and wait exactly that long, event-driven) or not bound the wait at all. This is a
correctness redesign, not a throughput tweak — land as a straight replacement (delete the old mechanism, its
tests, comments, and doc references), not a config-gated toggle.

### Current-state architecture (why this drifted — read before touching code)
One base mechanism, three increasingly-diverged per-subclass patches on top — the same "unfaithful port" failure
class as K-D40/§H, now found a third time:
- **`syncfl.TopAggregator`** (`top_aggregator.py:358-408`) — the ONLY common ancestor of all three families below.
  Defines the primitive: `_sim_recv_grace_s()` = `max(2.0, 4.0 × _sim_fill_ema)`, `_sim_fill_ema` updated ONLY on
  a fully-successful drain (`_note_sim_fill`, `:362`). Used directly by `_sync_sim_recv_first_k` (fwdllm/
  fwdllm_plus's sync barrier) and by **oort** (`oort/top_aggregator.py:91,104` — oort inherits `syncfl.TopAggregator`
  directly, NOT through asyncfl, so it has none of the sophistication below).
- **`asyncfl.TopAggregator(SyncTopAgg)`** (`asyncfl/top_aggregator.py`) — felix's async path. Overrides the
  primitive with its OWN per-trainer state: `_sim_trainer_budget: dict` (`:133`), `_sim_budget_min = 12.0`
  hardcoded seed (`:137`), `_sim_budget_running_mean`/`_sim_budget_n`, populated from `MessageType.TRAINING_BUDGET_S`
  on each commit (`:530-535`), consumed via `_sim_inflight_expected[end] = sst + budget` (`:1674`) — a genuine
  per-end deterministic gate, MUCH better than the primitive, but still has a hardcoded fallback for
  never-yet-observed trainers.
- **`fwdllm.TopAggregator(AsyncTopAgg)`** (`fwdllm_aggregator.py`) — fluxtune's async grad loop. Extends
  `asyncfl.TopAggregator`, so it inherits `_sim_trainer_budget`/`_sim_budget_min`'s *shape* — but reimplements the
  update logic a THIRD time (`:1055-1056`, its own copy, missing the running-mean tracking) and its per-pass drain
  timeout (`_sim_recv_min_grad`, `:919`) still calls the **primitive** `_sim_recv_grace_s()`, not the smarter
  per-end budget its own `_sim_inflight_expected` (`:3189`) otherwise uses. Half-migrated.
- **fwdllm/fwdllm_plus's sync barrier** (`_sync_sim_recv_first_k`, `top_aggregator.py:368`) never received ANY of
  this — it's still on the raw primitive, and its EMA-lock (one early small drain permanently caps the ceiling,
  §A) is what produced the measured 829.6s waste. It also has no per-end `_sim_inflight_expected`-style structure
  at all — it's a single whole-cohort timeout, not per-trainer.

### Target design (decisions locked in this session — do not re-litigate without new evidence)
1. **One canonical delay-report field: `MessageType.MODELED_DELAY_S`.** Retire `TRAINING_BUDGET_S` — switch
   async_cifar10/felix's trainer to stamp `MODELED_DELAY_S` too (same semantic value: the trainer's own configured
   `training_delay_s`, deterministic from the registry, mode-invariant, already stamped in both real and sim by
   fwdllm's trainer). One field, no drift between baselines going forward.
2. **One shared per-trainer delay cache, living in `syncfl.TopAggregator`** (the actual common ancestor of
   sync/async/oort/fwdllm-async) — replaces `_sim_fill_ema`/`_sim_recv_grace_s`/`SIM_RECV_GRACE_FLOOR_S`/
   `SIM_RECV_GRACE_FACTOR` (syncfl) AND `_sim_trainer_budget`/`_sim_budget_min`/`_sim_budget_running_mean`/
   `_sim_budget_n` (asyncfl + fwdllm_aggregator's duplicate) with ONE `dict[end_id -> float]`, updated whenever any
   received message carries `MODELED_DELAY_S` — **regardless of sync/async/oort subclass**.
3. **No hardcoded seed, no cross-trainer fallback (no running mean, no global min).** A trainer never yet observed
   THIS run gets **no bound** — the barrier blocks genuinely (the underlying `recv_fifo`/`drain_ready` primitives
   are already real `asyncio` event waits, confirmed not CPU-polling) until that trainer's first message arrives,
   at which point its exact delay is known for the rest of the run (delays are deterministic per `trainer_id`, so
   one observation suffices — no decay/EMA needed for a value that never changes). A trainer WITH a known delay
   gets an exact deterministic wait bound (`dispatch_vclock + known_delay + small margin`), not a guess.
4. **Dead-end/non-responding-trainer handling is explicitly OUT of scope for this pass** — deferred to Phase 2
   (unavailability isn't wired up yet; syn_0 is 100% availability so every dispatched trainer WILL eventually
   respond). Do not add a "give up" ceiling now; when Phase 2 lands, model it on `ROUND_CACHE_STUCK_TIMEOUT_S`'s
   pattern (`fwdllm_aggregator.py:95` — a long, generous, non-adaptive constant used only as a safety net), not
   another reactive EMA.
5. **felix is in scope, re-validated as part of this effort**, not carved out — the new mechanism is a strict
   improvement over its current hardcoded-seed/running-mean fallback, so its previously-banked 46/46 parity number
   must be reconfirmed (or shown to improve) before this lands, per the effort's own bar.
6. **Bundle the `version_key`-gated SEND-reselect fix** (same investigation, same call sites) — cache
   `_select_ends_respecting_reselect_gate`'s `reselect_each_iteration=True` branch (`fwdllm_aggregator.py:2644`)
   result keyed on `self.version_key`, mirroring the existing `reselect_each_iteration=False` branch's per-`round`
   cache (`:2651-2657`), instead of calling `channel.ends()` fresh on every loop tick. Fixes the fwdllm_plus
   11-13-calls-per-iteration real/sim mismatch at the root (§A).

### Subtasks — **ALL LANDED 2026-07-12**
1. **Field consolidation.** Async_cifar10's trainer now stamps `MODELED_DELAY_S` (`syncfl/trainer.py`, shared
   `_send` path; fwdllm already did). `TRAINING_BUDGET_S` fully retired — enum deleted (`message.py`), both
   stamp sites removed (the fwdllm one was 100% redundant with `SIM_CLIENT_TASK_TRAIN_DURATION_S`).
2. **Shared cache primitive.** `syncfl.TopAggregator._sim_known_delay_s` + `_note_sim_known_delay` (write) /
   `_sim_recv_timeout_s` (read: max known delay + margin, or `None` if any end unknown).
3. **`_sync_sim_recv_first_k`** uses `_sim_recv_timeout_s`; `None` flows into `recv_fifo`'s genuinely-blocking
   `timeout=None` path.
4. **asyncfl (felix risk step).** Budget-fallback fields deleted; `_sim_inflight_expected` entries are now
   conditional (no fallback). `_sim_recv_min`'s `drain_ready` branch can't block on `timeout=None` (confirmed
   against `channel.py`), so it polls at `_SIM_GATE_POLL_TICK_S` and relies on its own outer retry loop; the
   `recv_fifo` branch blocks genuinely. Learning moved to ingest time. `tests/mode`+`tests/selector` green —
   felix's live 46/46 re-confirmation is part of the still-pending subtask 9 run. **See risk R1 in §A.**
5. **fwdllm_aggregator's duplicate budget logic** migrated the same way (both `_sim_recv_min_grad` branches +
   dispatch-side write); the third budget-tracking copy deleted outright.
6. **oort + primitive deletion.** `_oort_sim_recv` migrated (its persistent cross-round buffer untouched, only
   the timeout source changed). Confirmed zero remaining callers, then deleted `_sim_recv_grace_s` and kin.
   Also mechanically renamed 3 more `TRAINING_BUDGET_S` telemetry-only readers the original scan missed.
7. **`version_key`-gated SEND-reselect.** Caches `reselect_each_iteration=True`'s `channel.ends()` result per
   `self.version_key`; empty results not cached. **See risk R2 in §A — this one's semantics are the least
   validated of the 9.**
8. **Purge dead references.** 11 test files updated; one test that pinned the deleted mechanism itself
   (`test_grace_is_adaptive_floor_not_half_second`) deleted. `checks.py`'s `drain_wall_budget_parity` docstring
   flags (doesn't guess) the `barrier_wait_s` tolerance re-derivation. PARITY.md's §3.drain/§3.resid/§3.evt left
   as-is (historical logs, still accurate).
9. **Validation.** `pytest lib/python/tests/mode lib/python/tests/selector`: 723 passed, 0 failed. Live real+sim
   smoke — **now launching, see §A.**

### Remaining smaller items — resolved
- `recv_fifo(timeout=None)` genuinely blocks; `drain_ready(timeout=None)` returns immediately-empty (confirmed
  against `channel.py`, pinned by `test_timeout_none_returns_immediately_does_not_block`).
- `MODELED_DELAY_S` is `None` cleanly when `training_delay_enabled=False` (both fwdllm and the new async_cifar10
  stamp) — `_note_sim_known_delay` skips caching `None`, so "not configured" and "not yet observed" both read
  as "no bound."
