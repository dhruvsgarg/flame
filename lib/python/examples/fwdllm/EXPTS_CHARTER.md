# FLUXTUNE evaluation charter — paper ⇄ code reconciliation (living doc)

**Purpose.** Single source of truth for reconciling the paper draft
([`05-evaluation.tex`](05-evaluation.tex), conceptual/narrative) with the code-side
design doc ([`EXPERIMENTS.md`](EXPERIMENTS.md), implementation/telemetry/plots). This file
records **conflicts resolved** (the charter) and **tracks tasks** as they are done. The run
ledger (which log file feeds which result) lives in [`EXPERIMENTS.md`](EXPERIMENTS.md) §10.

**Owner:** dgarg39 · **Branch:** `dg/fwdllm_sim_unavail` · **Opened:** 2026-07-07

**STATUS (2026-07-08).** Paper reconciliation ✅ delivered (§1–§3). Bottleneck optimizations (§5) built —
all **flag-gated, byte-identical off, fluxtune-only**, unit-tested. Measurements are at the paper's
primary **α=1** operating point (experiments never go below α=1; ablations go UP to α∈{10,100} —
[[fwdllm-alpha-convention]]; the `...185045...alpha0p1...` dir name was a mislabel, it loaded α=1).

| Opt | Feature | Flag(s) | Status |
|---|---|---|---|
| **1** | Suppress redundant intra-databin weight re-sends | `suppress_redundant_weights` (ON all baselines) | ✅ validated: 0% redundant, −79% down-bytes, no deadlock (`5503100b`) |
| **2** | Variance-plateau force-commit (plateau early + max-iter cap late) | `var_stopping_policy=plateau`, `var_plateau_patience`, `var_plateau_rel_delta`, `max_iterations_per_data_id` | ✅ **default ON fluxtune** (N=3, ε=0.15, cap=20); smoke vs var≤0.3 baseline: −25% iters, plateau fired on the grindy early bin |
| **3** | Gradient-aware aggregation / C3 (align gate + inverse-var) | `agg_rate_conf.type=grad_aware` | ✅ **default ON fluxtune** (`type=new` = FeLiX ablation only); align_gate on, inverse_var opt-in |
| 4 | Dynamic C | `dynamic_kc.enabled` | ⬚ wired, not enabled (§5c Opt-4) — remaining |

**Default fluxtune = full stack** (C1 JVP + Opt-1 comm + Opt-2 plateau + Opt-3 grad_aware). Remaining
opts (§2f): **Opt-4 dynamic C** (wired, not enabled), **Opt-5 finer staleness clock** (bundle w/ Opt-3),
and Opt-1's **cross-databin delta-cache** (the larger comm piece); X1 dynamic-K / X2 staleness-C parked.

**Four planned full runs (operator, N=100, target_acc=84%, parallel — a 2×2 over the two
learning-affecting opts, C1+Opt-1 constant):**

| Run | C1 JVP | Opt-1 comm | Opt-2 var-plateau stop | Opt-3 grad-aware agg | Isolates |
|---|---|---|---|---|---|
| **R1** FeLiX baseline | ✓ | ✓ | ✗ (var≤0.3 only) | ✗ (`new`) | reference (≈ last night) |
| **R2** + var-stop | ✓ | ✓ | ✓ plateau | ✗ (`new`) | Opt-2 alone |
| **R3** + grad-aware agg | ✓ | ✓ | ✗ | ✓ grad_aware | Opt-3 alone |
| **R4** all contributions (= default) | ✓ | ✓ | ✓ | ✓ | full fluxtune |

R2−R1 & R4−R3 = Opt-2 main effect; R3−R1 & R4−R2 = Opt-3 main effect; the 2×2 also gives the
interaction. Compare all vs last night's var≤0.3-only run. **Next:** launch, read `commit_reason` +
`grad_aware_gated_total`, tune ε/cap + align_floor/inverse_var, then re-characterize at α∈{10,100}.

**Directional source-of-truth (P4).** Implementation facts (model, trace, denominators,
session defs, surcharge numbers) flow **code → paper**. Narrative/positioning (the "why", SPRY
framing, threat-model preempts) flow **paper → code**. `experiments.yaml` stays SoT for *what ran*;
this charter is SoT for *what we decided and why*.

---

## Next steps — START HERE (code/impl, excluding C2 dynamic-K/C & C3 aggregation)

Pick up in a fresh context from this ordered list. **All of these are pure reducer/figure/telemetry
work over the *existing* run-set** (the `run_20260706/07…` α=0.1 smoke runs) — none needs C2 or C3, so
each can be built and validated now, before the α=1 re-runs. Full detail per item in §2b / §2b′.
Entry points: `expt_scripts/plotlib/reducers.py` (metrics), `plotlib/figures.py` (`FIG_BUILDERS`),
`compare_baselines.py` (CSV table), `make_paper_figs.py` (fig driver), `plot_run.py` (per-run).

**Tier 1 — correctness (do first; these affect whether a plotted number is valid):**
1. **N6 — sync one-round-span sessions (E5).** E5 currently derives session durations from
   `contributor_intervals` (dispatch→commit) for *all* baselines; the sync baselines (fwdllm,
   fwdllm_plus) need the paper's one-round-span definition, or the E5 "money plot" compares
   inconsistent semantics. Also wire `agg_round.contributing_trainers` (emitted, unused) if needed.
2. **E1 iterations-to-target metric.** `expt1` reports wall/vclock/rounds/data_bins but not iterations
   (the paper's 4th unit). Define (aggregator commits, or Σ trainer iters, up to `target_event`) and add
   the reducer field + CSV column.
3. **N3 — E1 convergence provenance.** The reducer *reconstructs* the convergence event from `agg_eval`
   instead of reading the watcher's `converge.json`; cross-check (or read converge.json) so the plotted
   time-to-τ can't silently diverge from the run-time verdict.

**Tier 2 — new figures the paper now promises (numbers already in the CSV, just no builder):**
4. **E2 aggregator stacked-bar** (compute vs barrier-wait vs drain) — `FIG_BUILDERS` add.
5. **E5 participation-count bars** (rounds / data-bins / iterations) — `FIG_BUILDERS` add.
6. **E3 Δloss-vs-cumulative-compute trajectory** (both denominators, 3 lines) — keep the existing
   summary bars; add the trajectory (needs per-eval compute accounting aligned to eval points).
7. **E4 message-size-distribution inset/panel** — from `comm` sizes (`up_sizes`/`down_sizes`/`comm_by_kind`).

**Tier 3 — minor/cleanup:**
8. N5 (`mqtt_fetch_s` idle decomposition — implement or drop from doc), the
   `real_gpu_time_s`/`gpu_compute_s` schema-name footgun, and the `FwdLLM++` vs `FwdLLM_Plus` label unify.

**After Tier 1–3:** re-run `make_paper_figs.py` on the smoke runs to sanity-check every figure renders,
then the **α=1 re-runs** (§2c) produce the final numbers into the ledger. The two big mechanisms
(§2c full-system, §2d C3) are the operator's separate track.

---

## 0. Governing principles

- **P1 — Substrate honesty.** The paper describes the substrate that *actually ran*, or we commit
  to running the claimed substrate before submission. No claiming a model/task/hardware we never ran.
- **P2 — Isolation vs full-system are two conditions.** The `main` run-set matches `agg_goal=10`
  and fixes fluxtune `C=30` *static*, which **disables dynamic K/C**. `main` isolates
  **contribution 1 (guided perturbations)** only. Efficiency claims (E2–E4) need a **separate
  full-system run** with dynamic K/C (+ gradient-aware aggregation) ON.
- **P3 — No takeaway ships ahead of its evidence.** Any takeaway currently contradicted by data
  (E3, E4) is either re-scoped or explicitly gated behind a not-yet-run experiment and marked.
- **P4 — Directional source-of-truth** (above).
- **P5 — One name per concept.** FwdLLM_Plus, the three contributions, and the real-world trace
  each mean the *same thing* in both docs.

---

## 1. Resolved decisions (the charter)

| # | Conflict | Decision | Applies to |
|---|----------|----------|-----------|
| **A1** Model | Paper claimed LLaMA2-7B + Mistral-7B | **Describe what ran: DistilBERT-base-uncased (66.4M, frozen) + AdapterHub bottleneck adapters (PEFT, ~1.5% trainable).** Drop 7B claims. | tex Setup |
| **A2** Task | Paper claimed Alpaca/FLAN instruction-following | **AG News 4-class topic classification.** | tex Setup |
| **A3** Hardware | Paper claimed Pixel 7 Pro NPU on-device | **NVIDIA A40, 8-GPU box, with a *modeled* mobile-latency delay (`delay_factor=2` = base/2, forward-only cheaper than backprop). Mobile figures are argued from structural properties, not measured.** | tex Setup |
| **A4** Real-world trace | Paper claimed REFL 136K-device trace | **`mobiperf_*` is the condition of record** (wired in `FedSgdTrainer.py`). **REFL** (`third_party/REFL/`) is a scoped future item, not a claim. | tex Setup |
| **A5** Non-IID α | Paper {0.1, 0.5}; `experiments.yaml` main=1; early runs ran α=0.1 | **RESOLVED (operator 2026-07-08): primary α=1; experiments NEVER go below α=1.** α=0.1/0.5 excluded (learning too slow; kept as a one-line "excluded, too slow" note). **Non-IID ablation moves toward MORE IID → α ∈ {10, 100}** (higher α = clients agree more = lower variance floor). All early α=0.1 smoke runs are **deprecated for tuning** and re-run at α=1 for final numbers. See [[fwdllm-alpha-convention]]. | tex + code |
| **B1** FwdLLM_Plus def | Paper: "cosine filtering / async" (`\tbd`) | **FwdLLM + three mechanisms: (1) per-iteration reselection [defining], (2) oracular availability, (3) relaxed `round_data_id` staleness. Sync, random selector, agg_goal=10.** Oracular is **inert under syn_0** (100% avail) — scope its benefit to unavailability (mobiperf, deferred); do not attribute current behavior to it. | tex Baselines |
| **B2** Contribution taxonomy | Paper used fine C1/C3/C6; code used coarse 1/2/3 | **Align on THREE contributions, code labels:** **C1** guided (JVP-magnitude) perturbation selection · **C2** dynamic K/C · **C3** intelligent (gradient-aware) aggregation. Async/iteration-level is the *structural substrate*, not a numbered contribution. | both |
| **C1/C2** E3/E4 vs data | Observed data contradicts E3 (compute) & E4 (comm) takeaways | **Retain E3/E4 as hypotheses** (they are `\plannedexp`); they assume C2 + C3 (being implemented) deliver. Current contribution-1-only data is the *expected intermediate* — documented in `EXPERIMENTS.md` §4 as ⚠. E1 stands on C1 alone. | both |
| **N1** fwdllm_plus oracular inert | Oracular excludes nobody at syn_0 | Scope oracular to unavailability; frame honestly (see B1). | tex |
| **N2** C3 status | `EXPERIMENTS.md` said "intelligent aggregation OFF"; code has fedbuff "new" staleness×utility rate **active** | **Current aggregation is a *borrowed placeholder* (fedbuff "new" scalar rate, scale 0.4 / a_exp 0.25 / b_exp 0.1; async_cifar10/REFL lineage), NOT fluxtune's intended C3.** Hypothesis: scalar multiplication of an *update* is wrong for *gradients*. **Design fluxtune's gradient-aware aggregation** → document dimensions in `EXPERIMENTS.md` now (§4.C3-investigation), implement later, then move feature doc to `fluxtune_contributions.md` and remove from `EXPERIMENTS.md`. | code now, both later |
| **N4** E1 clock | Paper claimed 4 units incl virtual-clock; real mode emits none; sim vclock unvalidated | **Report wall-clock + rounds + data-bins + iterations.** Drop virtual-clock for now; note as future. | both |
| **N7** Memory story | Systems thesis in `fluxtune_contributions.md`, absent from eval | **Motivation/design claim, argued from structure** (peak memory bounded by inference, independent of P & depth; inference-only-NPU fit). **No memory eval experiment.** | tex motivation/design |
| **D1** Seeds | Paper "median over ≥3 seeds"; runs single-seed | **Single seed now; leave a paper TODO for ≥3-seed medians on headline E1.** | both |
| **D2** GPU-second confound | Paper used GPU-seconds as co-equal E3 denominator | **Forward-pass count is the primary/clean denominator; GPU-seconds is secondary with an explicit 8-GPU-contention caveat** (wall-time GPU seconds inflate ~20× under 100-trainers/8-GPU contention; forward-pass count is hardware-independent). | both |
| **D3** Fidelity | Paper `\tbd` matched metric/value | **Accuracy parity (not time-to-accuracy): our Flame port reaches the accuracy `xu2024fwdllm` reports.** We explicitly do NOT compare wall-clock (faithful distribution — separate processes, real MQTT, per-trainer state — is *harder* on the baseline). Source from **old run data** (ledger row, §10). | tex Setup |
| **D4** SPRY / excluded baselines | Paper discusses SPRY as upper bound, not run | **Exclude FL methods that split/personalize the model per device** (incompatible with a homogeneous, churn-tolerant cross-device fleet where any device runs the same model). One-line pointer in Baselines; substantive *why* in Background/Related Work. Citations to add to bib: `panchal2024spry` (have); suggest split-learning (Vepakomma 2018), model-heterogeneous FL (HeteroFL, Diao 2021), personalized-FL survey (Tan 2022). | tex Setup + Background |
| **D6** Surcharge numbers | Paper `[10]×/[5]×` `\tbd` | **Fill from real:** P=10 default ⇒ 2P=20 forward passes/iter ⇒ ~10× sync compute at P=10, collapsing to parity at P=1. Fidelity-preserving opt cuts fluxtune −37% GPU time (sync −68%), zero gradient change. | tex |

---

## 2. Task tracker

Legend: ☐ todo · ◐ in progress · ☑ done. Update inline as work lands.

### 2a. Paper writing — ✅ `05-evaluation.tex` DELIVERED (operator integrating into paper repo)
- ☑ Rewrite Setup substrate (A1–A4): DistilBERT+adapters / AG News / A40+modeled-delay / mobiperf.
- ☑ Rewrite FwdLLM_Plus definition (B1) + oracular-scoped-to-unavailability (N1).
- ☑ Relabel contributions to C1/C2/C3 (B2).
- ☑ E1 units → wall/rounds/data-bins/iterations; vclock noted future (N4).
- ☑ E3 denominators → forward-pass primary, GPU-seconds secondary+caveat (D2).
- ☑ Fill surcharge numbers (D6).
- ☑ Fidelity → accuracy-parity framing (D3).
- ☑ Multi-seed TODO note (D1).
- ☑ α resolved → primary α=1 everywhere; α=0.1 excluded (too slow), kept as a one-line note; ablation → {0.5, 1.0} (A5).
- ☑ Reconcile "Expected plot" lines to code+builds: E1 loss companion; E2/E5 → CDF; E3 bars+trajectory.
- ☐ **(operator-owned, in paper repo)** SPRY/excluded-baseline justification in Background/Related Work +
  fill `\tbd` bib keys (D4: split-learning/HeteroFL/personalized-FL), Papaya cite, fidelity accuracy value,
  `sec:arch` ref for C3, and confirm the `~1.5%`/`delay_factor=2`/`~3.6s` numerics.

### 2b. Code / telemetry fixes (found in the reducer audit)
- ☐ **N3** — E1 time-to-τ reducer *reconstructs* convergence from `agg_eval` (over the loss-truncated series), does **not** read `converge.json`; can silently diverge from the watcher's verdict. Fix: read `converge.json` as source, or cross-check + document the derivation. (`plotlib/reducers.py:target_event`)
- ☐ **N5** — E2 idle decomposition partly unimplemented: `mqtt_fetch_s` emitted but unused (idle is just `1−busy_frac`); `barrier_wait_s`/`drain_tail_s` ≈0 in sim. Either implement the documented idle formula or simplify the doc claim.
- ☐ **N6** — E5 sync-session uses `contributor_intervals` (dispatch→commit) for *all* baselines (doc claims one-round-span for sync); `agg_round.contributing_trainers` emitted but never consumed. **Correctness item** — implement the one-round-span for sync baselines (fwdllm/fwdllm_plus) so the E5 comparison is valid.
- ☑ Fix stale note: reducers live in `plotlib/reducers.py`, not `compare_baselines.py` (`EXPERIMENTS.md` §4).
- ☐ Minor: `build_trainer_round` names its GPU field `real_gpu_time_s`; reducer reads `gpu_compute_s` from `extra`. Reconcile the schema footgun.

### 2b′. Plots/metrics to build so figures match the paper (decisions 2026-07-07)
The `compare_baselines.py` CSV already has every metric's *numbers*; these are the missing/changed *figures* + one missing metric. Builders live in `plotlib/figures.py` (`FIG_BUILDERS`), reducer fields in `plotlib/reducers.py`.
- ☐ **E1 iterations-to-target metric** — `expt1` reports wall/vclock/rounds/data_bins but NOT iterations. Add a cumulative-iterations-to-convergence metric (define: aggregator commits, or Σ iters, up to `target_event`) + column.
- ☑ **E1 loss-vs-time figure** — already built (`fig_e1_loss_vs_time`) and in `FIG_BUILDERS`; paper now cites it.
- ☐ **E2 aggregator-breakdown figure** — NEW builder: stacked bar of aggregator compute vs barrier-wait vs drain fraction (fields `agg_compute_s`/`agg_barrier_s`/`agg_drain_s`/`agg_wall_s` exist). Not in `FIG_BUILDERS`.
- ☑ **E2 trainer-busy wording** — code draws a CDF (`fig_e2_trainer_busy_cdf`); paper now says CDF (no code change).
- ☐ **E3 trajectory figure** — NEW builder: Δloss-accumulated vs compute-accumulated (both denominators), three lines each — needs per-eval compute accounting aligned to eval points. Keep the existing summary bars (`fig_e3_dloss_per_mfwd`/`_per_gpu_hour`). Decision: **both**.
- ☐ **E4 message-size-distribution inset** — NEW: inset (or panel) of per-message size distribution from `comm` sizes (`up_sizes`/`down_sizes`/`comm_by_kind`). Bytes bar (`fig_e4_network_bytes`) already matches.
- ☐ **E5 participation-count figure** — NEW builder: bars at three granularities (part_rounds/part_bins/part_iters — reducer has them, CSV reports p50/90/total). Not in `FIG_BUILDERS`.
- ☑ **E5 session wording** — code draws a CDF (`fig_e5_session_cdf`); paper now says CDF (no code change).
- ☐ **Cosmetic:** unify baseline display label — code plots `FwdLLM++` (`plotlib/baselines.py` SYSTEM label), paper prose/table says `FwdLLM_Plus`. Pick one.

**None of §2b′ depends on dynamic K/C or intelligent aggregation** — they are pure reducer/figure work over existing telemetry, runnable now on the current run-set.

### 2c. New runs required by the charter
- ☐ **Full-system fluxtune** condition (C2 dynamic K/C ON + C3 gradient-aware aggregation ON) — breaks the `agg_goal=10` match, so a *separate* run-set. Required to (re)make E2/E3/E4 efficiency claims (P2/P3).
- ☐ **mobiperf E1** real-world run — E1's headline ("near-failure → tractable") is currently only demonstrated on `syn_0` (100%). Needs the fwdllm_plus-under-scarcity barrier policy decision first (sync `agg_goal ≥ available` can't assemble; gate blocks it).
- ☐ **Fidelity accuracy-parity** — locate old run data (accuracy vs round/time) matching `xu2024fwdllm`; record value (D3, ledger §10).
- ☐ **≥3-seed** medians for headline E1 (D1) — optional, if budget allows.

### 2d. Contribution C3 — gradient-aware aggregation (design → implement → document) — *operator's separate track*
- ☑ Write up the C3 investigation in `EXPERIMENTS.md` §4 (hypothesis + 4 dimensions).
- ☑ **Design starter doc:** [`docs/aggregation_design.md`](docs/aggregation_design.md) — FedBuff→FeLiX→FluxTune regime, the weights≠gradients + slow-staleness hypothesis, 5-axis design space, candidate schemes S0–S5, and how to evaluate the aggregation itself.
- ☐ Implement fluxtune's gradient-aware aggregation; disable the borrowed fedbuff "new" placeholder.
- ☐ On landing: move feature doc → `fluxtune_contributions.md`; remove from `EXPERIMENTS.md`.

**C3 dimensions to evaluate (seed for the investigation):**
1. **Staleness definition under iteration-based progression.** Staleness = `agg_model_version − trainer_version`; `_model_version` advances **per data-bin completion**, and a data-bin only completes when the **variance threshold is met** → staleness accrues at the variance-gated data-bin rate, *not* wall-clock. Hypothesis: staleness grows **slower** in fluxtune than round-based FL. Quantify the effective staleness distribution vs a round-based baseline.
2. **Scalar rate vs gradient-aware combination.** The current `weight_factor` scalar-multiplies the whole update — borrowed from weight-averaging async FL. For *gradient* updates (forward-mode JVP estimates), a scalar down-weight may be the wrong operator. Explore direction-/variance-aware combination (e.g. weight by JVP magnitude / SNR / agreement with the running aggregate), not just staleness×loss.
3. **Interaction with C1 (guided perturbations) & the variance gate.** Updates already passed a `var ≤ var_threshold` gate; does re-weighting by loss (`stat_utility`) double-count what the gate filtered?
4. **Interaction with C2 (dynamic K/C).** Concurrency (C) sets how many stale/in-flight updates coexist; the aggregation rule and the concurrency controller co-determine wasted work (the E3/E4 root cause).

### 2f. Bottleneck-driven optimizations (measured — see §5; all flag-gated, byte-identical off)
Ordered by the §5b optimization ledger. Each is an independent knob we can turn on/off.
- ☑ **M0 — measure at α=1: DONE.** Tool `expt_scripts/characterize_variance_curve.py` (per-databin
  variance-decay curve + evolution + counterfactual fixed_cap/plateau sweeps, §5e). The §5a/§5e source
  run turned out to be **α=1** (loaded `niid_label_clients=100_alpha=1`, verified from `*trainers.log`;
  the `alpha0p1` dir name is a mislabel). So the grind IS confirmed at the primary operating point
  (var@commit ≈0.29 noise-dip, floor ~0.45, bins 15→34 iters) → Opt-2 unblocked, tunable off §5e.
  Follow-up: re-characterize at α∈{10,100} to see if the grind (hence Opt-2's win) survives more IID.
- ◐ **2f-1 — redundant weight-send elimination** (`suppress_redundant_weights`, 🟢).
  - ☑ **Opt-1 intra-databin suppression IMPLEMENTED + VALIDATED** (§5d): shared sync+async
    `_should_send_full_weights` + `_weights_sent_this_cycle` (≤1 payload/trainer/model_version), flag
    default off, enabled on all baselines; 9-case unit test + `audit_weight_redundancy.py` (47 tests
    green). Pre-fix: fwdllm 90%/22 GB, fluxtune 71%/68 GB. Validation `run_20260708_001641`: **0%
    redundant, −79% down-bytes, no deadlock** (committed `5503100b`).
  - ☐ Cross-databin delta/version-cache (compress the model *change* when a trainer genuinely needs a new
    version) — the remaining `fluxtune.comm.delta_weights` piece; larger, do after Opt-1 validates.
- ☑ **2f-2 — variance-gate force-commit (Opt-2): IMPLEMENTED + DEFAULT ON (fluxtune)** (`var_stopping_policy`,
  `max_iterations_per_data_id`, `var_plateau_patience`, `var_plateau_rel_delta`, 🟡). Policy ∈
  {off, fixed_cap, plateau}. **Two triggers on top of var≤thr, per the operator's regime model:**
  **plateau = PRIMARY/early** (commit when the per-bin var curve flattens — rel var-drop over last N < ε
  while var>thr — scale-invariant → commits the denoised estimate at ~iter 12–17), **cap = LATE backstop**
  (`max_iterations_per_data_id`; late-stage plateau onset drifts to ~33 and the model is already stable, so
  the cap force-commits first). Whichever fires first wins; `commit_reason` (natural/cap/plateau) on
  `agg_round` records which. Defaults `_metadata/baselines.yaml` fluxtune: **policy=plateau, N=3, ε=0.15,
  cap=20** (cap ABOVE early plateau fire, BELOW late grind). Pure decision
  `TopAggregator._should_force_commit_on_plateau` (8-case unit test, green). **byte-identical when
  `off`.** No A/B — compared directly against the earlier var≤0.3-only fluxtune run. Target: iters ~18→~12
  (E3), faster model-version (revives staleness). **Next: 10-trainer smoke vs the var≤0.3 baseline, read
  commit_reason split + learning; then tune ε/cap; re-characterize at α∈{10,100}.** (The M0 threshold-raise
  idea is subsumed — the plateau commits ≈ the denoised estimate ~0.45, so raising the 0.30 bar is moot.)
- ☑ **2f-3 — gradient-aware aggregation (C3): IMPLEMENTED, default OFF** (`agg_rate_conf.type=grad_aware`).
  New rate branch in `aggregate_grads_from_trainers` replacing the scalar staleness×utility weight with a
  DIRECTION/reliability-aware one, **bounded ≤ base so the effective LR never inflates** (no re-tune needed):
  **align gate (S2, primary)** down-weights an update whose cos vs the running aggregate `self.grad` <
  `align_floor` (→0 at cos=−1; fixes H1/M-12 — averaging anti-aligned JVP estimates), **inverse-variance
  (S1, optional `inverse_var`)** ×min(1, var_ref/var_i) from the trainer's own perturbation grads. Pure
  `TopAggregator._grad_aware_rate` + `_cosine_flat` (13-case unit test). Telemetry `agg_rate_type` +
  `grad_aware_gated_total` on `agg_round`. **DEFAULT ON for fluxtune** (align_gate on, base=new,
  inverse_var off); `type=new` = FeLiX, kept only for the R1 ablation arm. Validated by the operator's
  target_acc runs (the run IS the weight↔Δloss test). **Next: run the 2×2 (R1–R4), tune align_floor +
  inverse_var on/off.**
- ☐ **2f-4 — dynamic C** (`dynamic_kc.enabled` + `EligibleEndsBasedPolicy`, 🟡) — sequence **after** 2f-1;
  drive by eligible-pool/wasted-work, not staleness. Re-measure the C↔wall-clock tradeoff post-delta-encode.
- ☐ **2f-5 — finer staleness clock** (`agg_rate_conf.staleness_clock`) — Δcommits/wall-age; **bundle with
  2f-3 only** (re-check sim parity — reorders commits).
- 🔴 **Parked (measured dead-ends):** dynamic-K-to-lower-variance-floor (floor is structural non-IID) and
  StalenessBasedPolicy-for-C (staleness too low at syn_0 to trigger). Revisit X2 only under mobiperf scarcity.

### 2e. Ablations (paper has them planned; no tooling yet — D5)
- ☐ JVP guidance sensitivity (C1): threshold, refresh frequency.
- ☐ K/C policy sensitivity (C2): window N, growth/shrink factors, C_max.
- ☐ Non-IID robustness: Dirichlet α ∈ {10, 100} (more-IID direction; α<1 excluded — A5 / [[fwdllm-alpha-convention]]).
- ☐ Build ablation tooling (reuse the reducer backbone; new run-sets).

---

## 3. Post-reconciliation audit — ✅ DONE (2026-07-07)

Full audit of the paper's metrics/plots/baselines against the code completed across three layers:
paper claims (`05-evaluation.tex`), the numeric reducers (`compare_baselines.py` CSV — covers all 5
experiments' metrics), and the paper-figure builders (`plotlib/figures.py` → 7 figures). Findings:
- **Numbers:** all 5 experiments' metrics are computed in the CSV; baselines consistent (FwdLLM /
  FwdLLM++ / FluxTune + ablation slot).
- **Figures:** 4 of the paper's plots are built (E1 acc+loss, E2 trainer-busy CDF, E3 two summary bars,
  E4 bytes bar, E5 session CDF); **missing builders** = E2 aggregator breakdown, E3 trajectory, E4
  msg-size inset, E5 participation bars (→ §2b′ / §Next-steps).
- **Metric gap:** E1 iterations-to-target not computed.
- **Correctness:** N3 (convergence reconstructed, not from converge.json), N6 (sync sessions use
  async semantics), N5 (mqtt idle unused).
The actionable output is the **§Next steps** list (top of file) + §2b/§2b′. Re-run once the full-system
and α=1/mobiperf runs land to fill the ledger.

---

## 4. Open items still needing a decision

- _(none open)_ — **A5 (α) resolved** (2026-07-07, refined 2026-07-08): primary **α=1**, experiments
  **never below 1**; non-IID ablation moves to MORE IID → **α∈{10,100}** ([[fwdllm-alpha-convention]]).
  **Config↔run-name mismatch RESOLVED (2026-07-08):** the mismatch is only in the run *directory names*
  (three n100 runs are dir-named `alpha0p1` but their `*trainers.log` + config loaded
  `niid_label_clients=100_alpha=1`); the *data* matched `experiments.yaml`'s α=1. Only
  `run_20260706_161938_fwdllm` is genuinely α=0.1. **Verify α from the loaded-partition log line, never
  the dir name.** So §5's ledger is already an α=1 measurement — no α=1 re-run of §5 needed.

---

## 5. FluxTune bottleneck analysis & optimization ledger (measured)

**Source.** All numbers below are a full streaming reducer pass over
`run_20260706_185045_fluxtune_n100_smoke_syn_0_real` (**α=1**, N=100, C=30, agg_goal=10, C1-only,
C2 off, C3=borrowed placeholder). **✅ α CORRECTED (2026-07-08):** the directory name says
`alpha0p1` but the run's config AND its `*trainers.log` both show it loaded
`niid_label_clients=100_alpha=1` — verified by grepping the loaded-partition log line. **These are
α=1 measurements at the paper's primary operating point**, not α=0.1 as originally labeled (that came
from the stale dir name + the known config↔run-name mismatch). Consequence: **task M0 is effectively
satisfied** — the variance grind is real at α=1, so Opt-2 tuning may proceed off these numbers. (Of the
n100 runs only `run_20260706_161938_fwdllm` is genuinely α=0.1. See [[fwdllm-alpha-convention]].)
Ablations move to MORE IID (α∈{10,100}) where the floor should drop further — re-measure there.

### 5a. Measured diagnostics (what the run actually did)

| # | Metric | Measured | What it means |
|---|--------|----------|---------------|
| M-1 | agg iterations that **commit** a model update | **4.3%** (139 / 3212) | 95.7% of aggregation work advances nothing |
| M-2 | median `var` vs `var_threshold` | **0.55 vs 0.30** | the *median* iteration is 1.8× over the commit bar |
| M-3 | achievable **variance floor** (asymptote by itr≈10) | **~0.45 median / ~0.30 min** | denoising plateaus **above** the threshold → gate crossed only by noise dips |
| M-4 | marginal var drop, itr 11→19 | **0.506 → 0.453** (≈0.007/itr) | iterations past ~10 buy almost nothing |
| M-5 | median iterations per data-bin (max) | **18 (61)** | bins grind far past the point of diminishing returns |
| M-6 | `force_commit` (anti-grind escape valve) firings | **0.0%** | the safety valve that should cap the tail is **inert** |
| M-7 | buffers with **zero** staleness spread | **61.7%** | staleness axis is degenerate → aggregation rate can't differentiate (H2 ✓) |
| M-8 | staleness distribution | **median 1, p90 3, mean 1.23** | model-version crawls (caused by M-1) → updates read identical versions |
| M-9 | model versions shipped vs weight **downloads** | **139 vs ~26k (≈189×)** | each committed model is re-sent ~189 times |
| M-10 | redundant weight bytes | **~88 GB weights = 65% of 146 GB total; ≈99% re-sends of an unchanged model** | async re-enlist re-pulls the *same* weights during non-commit stretches |
| M-11 | trainer busy fraction (p50) | **8.3%** | trainers idle 92% (expected for async; not the bottleneck) |
| M-12 | accuracy 50%→75%→100% wall | **0.825 → 0.743 → 0.806** | ~10-pt mid-run regression while loss keeps falling → aggregation instability (H1) |

**The through-line: the variance gate is the hub.** M-1 (wasted compute) is *caused by* the M-3
threshold-vs-floor mismatch + M-6 inert escape valve; M-7/M-8 (dead staleness) are *caused by* M-1
(slow model-version advance); and half of M-9/M-10 (redundant comm) is *caused by* M-1's long
non-commit stretches. So one ML knob (the gate) sits upstream of compute, staleness, and part of comm;
one systems knob (delta/version-cache) owns the rest of comm independently.

**Key inference — the floor looks structural, not sample-limited.** The grad pool grows across
iterations yet `var` asymptotes (M-3/M-4). That means more samples *from the same α=0.1-skewed clients*
hit an irreducible cross-client-disagreement floor → **"collect more" (dynamic K) will not lower it**;
"weight smarter" (C3) or "accept the floor" (threshold) will. This directly de-prioritizes dynamic-K.

### 5b. Optimization ledger — ordered by maximum gain

Legend — **Risk/Reward**: 🟢 high-reward/low-risk · 🟡 high-reward/medium-risk · 🔴 high-risk-low-reward (avoid/park).
Flags: every change lands **byte-identical-off** behind the named flag.

| Rank | Lever (flag) | Current | Issue | Prevalence / cost | Upside across metrics | Probable downside | How to fix | Priority / risk |
|---|---|---|---|---|---|---|---|---|
| **1** | **Delta / version-cache weight download** (`fluxtune.comm.delta_weights`) | full model re-sent on every (re-)dispatch | model changes 139× but ships ~26k× | **65% of all bytes (~88 GB), ≈99% redundant** (M-9/M-10) | **E4 huge** (−~85 GB down); frees the C-tradeoff for lever 4; **zero learning change** | cache-invalidation bug could serve stale weights → correctness | version-tag each dispatch; client at current version → no-op/ack, else send delta (or gzip). Pure systems. | 🟢 **do first** |
| **2** | **Variance-gate threshold + real force-commit** (`varGate.threshold`, `varGate.maxItersPerBin`, `varGate.plateauRelDelta`) | `var_threshold=0.3`; force-commit never fires | threshold **below** achievable floor (~0.45) → grind to 18–61 iters | **95.7% of forward-pass compute doesn't commit** (M-1); iters past ~10 wasted (M-4/M-5) | **E3 big** (iters 18→~7 ≈ 2–3× less wasted compute); **revives staleness** (faster model-version → fixes M-7/M-8); trims non-commit re-pulls (helps E4) | committing at the plateau = noisier updates → could worsen M-12 instability or final acc | (a) raise threshold to ~0.45–0.5; (b) wire diminishing-returns force-commit (commit if var drop < `plateauRelDelta` over N iters, or at `maxItersPerBin`). A/B on E1 acc. | 🟡 **do second** (cheap, isolating) |
| **3** | **Gradient-aware aggregation — C3** (`agg_rate_conf.type=grad_aware`: inverse-variance S1 + alignment-gate S2) | FeLiX scalar rate (staleness×utility), grad-stats discarded | scalar rate can't refuse anti-aligned gradients (H1); best signals (var/SNR) computed then thrown away (H3) | drives M-12 instability; underlies E3 per-unit inefficiency | **E3 quality** (usable direction from fewer samples); **E1 stability** (stop averaging opposing grads); makes each commit count | re-tunes effective LR (Axis E); double-counts the C1 gate if naive; more agg cost | inverse-variance weight `w∝1/var_i` (reuse computed stats) + drop/down-weight cos<0 vs running aggregate; re-normalize + re-tune server LR. Instrument weight↔Δloss corr first. | 🟡 **do third** (real contribution; higher effort) |
| **4** | **Dynamic C** (`dynamic_kc.enabled`, `EligibleEndsBasedPolicy`) | static C=30 | high concurrency → speculative stale work + redundant re-pulls | contributes to M-9/M-10 and speculative compute | **E4/E3** (fewer in-flight → fewer re-pulls, less stale work) | too-low C slows wall-clock (E1) — the actual win | drive C by eligible-pool / wasted-work signal (**not** staleness — M-8 too low to trigger). Sequence **after** lever 1 (delta-encode removes most of C's comm penalty → re-measure the tradeoff). | 🟡 medium |
| **5** | **Finer staleness clock** (`agg_rate_conf.staleness_clock`) | `Δmodel_version` (coarse, gated) | 61.7% degenerate (M-7); axis contributes ~0 differentiation | only matters *inside* C3's rate | small alone; multiplies C3 | may reorder commits → sim-parity re-check | Δcommits-since-dispatch or wall/vclock age. **Do only bundled with lever 3**; lever 2 already partly revives staleness for free. | 🟡 low-standalone |
| **X1** | **Dynamic K to lower the variance floor** (`VarianceBasedPolicy`) | — | floor is **structural non-IID** (5a inference), not sample-count | — | ~none: more samples of the same skewed clients won't cross 0.3 | more compute for no floor movement | — | 🔴 **park** — high-risk-low-reward given the structural floor |
| **X2** | **StalenessBasedPolicy for C** | — | assumes high staleness to shed | staleness median 1 (M-8) → never triggers at syn_0 | ~none under current availability | — | (revisit only under mobiperf scarcity) | 🔴 **park** for syn_0 |

**Sequencing rationale.** 1 is a free, isolated systems win that also changes the economics of 4 →
do it first and independently. 2 is the cheapest test of the *entire* wasted-compute thesis (one
threshold + one escape valve) and its side effect revives the staleness signal 5 depends on → do it
second and read E1 before committing to 3. 3 is the genuine C3 contribution and the fix for the M-12
instability, but it's the highest-effort and needs LR re-tuning → third. 4 after 1. 5 only with 3.
X1/X2 are parked as measured dead-ends.

### 5c. Fleshed-out implementation designs (verified against code 2026-07-07)

Each lever below is **verified against the live code** with anchors, states the exact change + flag
(byte-identical off, fluxtune-only), the logging/telemetry to prove the fix, and flags open **design
decisions** (→ ask operator). Confidence tag: **SURE** = implement + validate on a 5-databin run ·
**MEASURE-FIRST** = add telemetry, short run, then design the fix from the observed curve.

#### Opt-1 — suppress redundant intra-databin weight re-sends (`fluxtune.comm.suppress_redundant_weights`) — **SURE**
**Verified.** `_distribute_weights_async` ([`fwdllm_aggregator.py:2731`](../../flame/mode/horizontal/syncfl/fwdllm_aggregator.py#L2731)).
With `inc_model_version_per_data_id=true` the model version is **constant within a databin**, and the
full payload is **byte-identical across that databin's iterations**: `WEIGHTS =
get_trainable_param_state_dict()` (params don't change with no commit) and `GRAD_POOL =
cached_shared_grad_pool_trainable`, which is only recomputed `if self._is_model_updated` (`:2359`,
false mid-databin). A guard already downgrades a *current* trainer to a tiny VAR=bad "keep training"
message (`:2843`), but the currency map `_trainer_last_model_version[end]` is written **only on grad
RETURN** (`:1446`), never at send — so every trainer pulled in to refill concurrency mid-databin reads
as "stale" and gets the identical full payload again.
**Measured cost.** 26,274 weight-sends vs 10,049 VAR=bad; **24,570 (93.5%) of weight-sends are at
iter>0** inside a databin where the payload is identical; ~189 weight-sends/databin against only 100
trainers ⇒ provable repeat-sends. This is ~88 GB, the E4 headline. **Async path only ⇒ fluxtune-only.**
**Trainer side is safe.** The trainer caches `self.weights`/`self._model_version` and on VAR=bad "does
not update weights" ([`fwdllm_trainer.py:304`](../../flame/mode/horizontal/syncfl/fwdllm_trainer.py#L304)),
i.e. trains on its cached copy — so a trainer that received version M once can be sent VAR=bad thereafter.
**Change.** When the flag is on and we send full WEIGHTS at the current model_version, mark the end as
having it (set `_trainer_last_model_version[end]=self._model_version` at send, OR a dedicated
`_sent_current_version` set cleared on commit). Re-dispatch within the same databin then hits the
existing VAR=bad path automatically. Off ⇒ current behavior byte-identical.
**Logging to prove it.** Extend the `[Distribute] Done` line + a counter: `redundant_weights_suppressed`
(sends converted weights→VAR=bad); expect within-databin weight-sends to collapse toward "≤1 per trainer
per databin." Re-run the §5a reducer: weight bytes ↓, `agg_eval` acc/loss + var-vs-iter curve **unchanged**.
**Design decision (D-1):** mark-at-send is self-correcting if a first-send is dropped (trainer returns a
grad tagged M−1 → normal staleness down-weight), but not belt-and-suspenders. Accept mark-at-send
(simplest), or add an ack/confirm before downgrading? *Recommend mark-at-send + the suppressed-counter telemetry.*

#### Opt-2 — variance-plateau force-commit (`var_stopping_policy`) — **IMPLEMENTED + DEFAULT ON (2026-07-08)**
**Status.** Measure-first done (§5e, α=1 curve) → framework landed and enabled as the fluxtune default.
`var_stopping_policy ∈ {off, fixed_cap, plateau}` (config, `Extra.allow` hyperparameters). `plateau` =
commit when the per-bin var curve flattens: pure `TopAggregator._should_force_commit_on_plateau`
(rel var-drop over last `var_plateau_patience` N < `var_plateau_rel_delta` ε, while var>thr) called from
`FedSGDAggregator.aggregate()` right after the cycle's var is appended to `var_prev_iter_list`; it ORs
into the existing `_force_commit_this_cycle` cap path. **Operator regime model (drives the values):**
plateau is the PRIMARY early trigger (curve flattens fast when the model is still moving → commit the
denoised estimate ~iter 12–17); the max-iter CAP is the LATE-stage backstop (near convergence the plateau
onset drifts to ~33 but the model is already stable, so the cap force-commits first). So the cap sits
ABOVE the early plateau fire and BELOW the late grind. `_metadata/baselines.yaml` fluxtune defaults:
**policy=plateau, N=3, ε=0.15, cap=20**; disable with `var_stopping_policy: off` (byte-identical).
`commit_reason` (natural/cap/plateau) + `stopping_policy` on `agg_round` measure the actual split.
**Byte-identical when off** (unit-tested, 8 cases). The "adaptive" tier is deferred (D-2) — the plateau
rule already commits ≈ the denoised estimate, covering the fixed-cap+plateau intent. Original
MEASURE-FIRST notes retained below for provenance.

**Verified.** The force-commit path **already exists** — `_force_commit_this_cycle` fires when
`iteration_per_data_id+1 >= self._max_iter_per_data_id` (`:1687`), read from config
`max_iterations_per_data_id` (`:307`), **default `None` ⇒ never armed** (0.0% in the run; bins grind to 61).
**Measured.** var decays but asymptotes ~0.45 (median) by itr≈10 vs threshold 0.30 (M-3/M-4); a fixed
cap would help but you want the stopping rule to **adapt to the variance curve** (more patience early
where variance reduction is real; commit sooner near convergence where gradients are consistent).
**Two-step plan.** (a) **Measure-first:** we already emit `var` + `iteration_per_data_id` per `agg_round`,
so extract the full **per-databin variance-decay curve** and characterize it (initial value, decay rate,
plateau level, plateau onset) across the run and over training-time — does the plateau level fall as
accuracy rises? does onset move earlier? (b) **Then implement** `varGate.stopping_policy ∈
{off, fixed_cap, plateau, adaptive}`: `plateau` = commit if the relative var-drop over the last `N` iters
< `ε` (patience `N`, tolerance `ε`); `adaptive` = tune `N`/`ε` (or the effective threshold) from the
observed curve regime (e.g. loosen early, tighten as best-loss improves). Reuses the existing
`_force_commit_this_cycle` plumbing; the policy just supplies the trigger.
**Trackable numbers:** iterations-per-databin, var-at-commit, var-drop-rate (last N), Δloss-per-commit,
plus accuracy — the levers the adaptive rule trades off.
**Design decisions (D-2):** (i) ship a plain `fixed_cap` (e.g. 12) as a safety floor **now** while the
adaptive rule is designed, or wait? (ii) should the rule adapt on **wall/compute budget** too (commit
sooner when behind), or purely on the variance curve? *Recommend: land `fixed_cap=~12` behind the flag
immediately as a floor, run the curve-characterization in parallel, then design `adaptive`.*

#### Opt-3 — gradient-aware aggregation, C3 (`agg_rate_conf.type=grad_aware`) — **IMPLEMENTED (default OFF)**
**Status (2026-07-08).** Landed as a `grad_aware` branch in `aggregate_grads_from_trainers` (the same
per-update rate hook the `new`/`old` types use). It replaces the scalar staleness×utility weight with a
direction/reliability-aware one, **bounded ≤ base (never inflates the effective LR → no re-tune)**:
**align gate (S2)** = down-weight `cos(update, running self.grad) < align_floor`, linearly → 0 at cos=−1
(fixes H1: averaging anti-aligned JVP estimates, the M-12 instability); **inverse-variance (S1, opt-in)** =
×min(1, var_ref/var_i) from the trainer's own perturbation grads. Pure `TopAggregator._grad_aware_rate` +
`_cosine_flat`, 13-case unit test; `agg_rate_type` + `grad_aware_gated_total` on `agg_round`. Config in
`_metadata/baselines.yaml` (documented; default `new` → byte-identical). Deviates from the original
MEASURE-FIRST plan (instrument-then-prototype) at the operator's request so run B/C are launchable — the
target_acc run is the validation. The design notes below are retained for provenance.
**Verified.** `FedSgdAggregator.aggregate()` ([`FedSgdAggregator.py:192`](aggregator/FedSgdAggregator.py#L192))
computes `var`, `real_var` (JVP), `snr`, `grads_snr`, `cv` — then uses them **only to gate the commit**;
the per-update weight is the scalar fedbuff `weight_factor` (staleness×utility,
[`fedbuff.py:110`](../../flame/optimizer/fedbuff.py#L110)). Confirms H3 (best signals discarded) + H2
(staleness degenerate, M-7). **The formulation chain to make crystal-clear before coding:**
1. **Issue:** a scalar rate rescales a gradient's *magnitude*, never its *direction*; anti-aligned JVP
   estimates are still averaged (H1) — the plausible cause of the M-12 mid-run accuracy regression.
2. **Numbers it manifests in:** weight-spread entropy (near-uniform ⇒ not differentiating), staleness
   spread (M-7 ≈ 0), and — the real test — **weight↔realized-Δloss correlation** (does a higher weight
   predict a larger loss drop?) + **wasted-work fraction** (share of committed forward-pass compute sitting
   in low-weight / anti-aligned updates).
3. **Formulation:** S1 inverse-variance `w∝1/var_i` (or SNR) — min-variance combine of noisy estimates;
   S2 alignment-gate — drop/down-weight `cos(update, running_aggregate)<0`; S3 `|JVP|·SNR`. Re-normalize +
   re-tune server LR (Axis E — a weight-scale change silently rescales the effective LR).
4. **Measure after:** the four instrumented quantities above **besides** loss/var/iters/accuracy.
**Open granularity question:** the computed stats are **per-batch over the pool**, but S1/S2 want
**per-update** var/alignment — needs per-contribution stats (extra compute) or a per-update proxy.
**Design decisions (D-3):** (i) prototype **S1 (inverse-variance)** or **S2 (alignment-gate)** first, or
both composed? (ii) per-update signals (accurate, costs compute) vs per-batch proxy (cheap)? (iii)
interaction with the C1 variance gate — loosen the gate and move discrimination into the weight, or keep
the gate and weight on an orthogonal signal (alignment/magnitude) to avoid double-counting? *Recommend:
instrument weight↔Δloss + weight-entropy on the current rule first (one short run), then prototype S1 as
the principled baseline.*

#### Opt-4 — dynamic C (`dynamic_kc.enabled=true` + policy) — **MEASURE-FIRST / configure**
**Verified.** Infra is **fully wired, just disabled**: controller + policies exist
([`dynamic_kc_controller.py`](../../flame/selector/dynamic_kc_controller.py),
[`dynamic_kc_policy.py`](../../flame/selector/dynamic_kc_policy.py)); the aggregator builds the controller
when `dynamic_kc.enabled` (`:417`); `async_oort` consumes pushed `dynamic_c` (`:295`,
`effective_c=channel_props.get("dynamic_c", self.c)`). So opt-4 = **enable + choose a policy + validate in
a run**, not new plumbing.
**Design note.** StalenessBasedPolicy won't trigger (staleness median 1, M-8), so drive C by
**eligible-pool / wasted-work**, not staleness. **Sequence after opt-1**: delta/suppress removes most of
C's *communication* penalty, so the C↔wall-clock tradeoff must be **re-measured post-opt-1** before tuning
C down for byte reasons.
**Design decisions (D-4):** (i) which policy signal — eligible-pool right-size, or a wasted-work
(stale-grad fraction) controller? (ii) bounds `c_min/c_max` + `update_every_n_aggs`? (iii) confirm the
strict "after opt-1" sequencing. *Recommend: enable with `EligibleEndsBasedPolicy`, wide bounds, and
treat the first run as a measurement of the post-opt-1 tradeoff.*

**Net.** Opt-1 is SURE (implement + validate). Opt-2/3/4 are MEASURE-FIRST: each needs one short
instrumented run (or reducer pass) to fix the design, per the operator's guidance. Decisions D-1…D-4 gate
the code.

**Decisions RESOLVED (2026-07-07, operator):**
- **D-1 → mark-at-send** (self-correcting; add `redundant_weights_suppressed` telemetry). Opt-1 → implement now.
- **D-2 → fixed cap ~12 now** as a safety floor (flag-gated, fluxtune-only) **+** characterize the variance
  curve in parallel to design the adaptive plateau rule.
- **D-3 → instrument the current scalar rule first** (weight-entropy, weight↔Δloss corr, wasted-work), then
  prototype **S1 inverse-variance**.
- **D-4 → `EligibleEndsBasedPolicy`, sequenced strictly after Opt-1** (first run measures the post-suppression
  C↔wall-clock tradeoff).
- **Build order:** Opt-1 (impl+validate) → Opt-2 fixed-cap + curve reducer → Opt-3 instrumentation → Opt-4 enable.

### 5d. Opt-1 redundancy audit + root cause + fix (all baselines, verified 2026-07-07)

**Per-databin audit (ground truth, `expt_scripts/audit_weight_redundancy.py`).** For each data-bin
(reconstructed via commit count, robust to `data_id` cycling) we counted, per trainer, how many full
WEIGHTS payloads it received. A trainer should get the byte-identical model **at most once per
data-bin** (re-dispatches get the tiny VAR=bad "keep training" message).

| baseline | data-bins | unique trainers/bin | weight-sends/bin | bins w/ a trainer sent weights 2+× | redundant weight-sends | redundant bytes |
|---|---|---|---|---|---|---|
| **fwdllm** | 69 | 10 | **100** | **100%** | **90.0%** (6210) | **22.3 GB** |
| **fwdllm_plus** | 166 | 72 | 19 | 38% | 3.5% (93) | 0.34 GB |
| **fluxtune** | 140 | 46 | **136** (max 573) | **97%** | **71.1%** (18,673) | **67.6 GB** |

fwdllm's per-peer histogram is literally `{10 sends: 10 trainers}` — the same K=10 selected trainers
each got the identical model **10×** (once per within-bin iteration). fluxtune's tail reaches **21× to
one trainer in a single bin**.

**Where the bug was & why we thought it was fixed (corrected 2026-07-07).** The WEIGHTS-vs-VAR=bad guard
*exists* in both distribute paths: `if var_good_enough: WEIGHTS; elif is_stale: WEIGHTS; else: VAR=bad`.
An earlier draft of this section blamed the `is_stale` path — **that was wrong.** The telemetry shows
`staleness == 0` for the active trainers every iteration (they return grads tagged the current version,
so `_trainer_last_model_version` *is* updated and `is_stale` is False). The real cause: the sync run loop
is `loop(distribute >> aggregate)`, so **distribute is called once per iteration (~10–12×/data-bin)** and
each call takes the **unconditional `var_good_enough → WEIGHTS` branch**, re-shipping the byte-identical
model to the same selected trainers. The `is_stale` check the guard added never even runs for these — it
sits *after* the `var_good_enough` short-circuit. Verified from the log keyed on the printed
`model_version`: `{10 sends: 10 trainers}` at model_version 4/5/6, all WEIGHTS-labeled/1.8 MB. So the
guard that was added was real but **guarded the wrong branch**, which is why it looked done yet never
suppressed anything on fwdllm/fluxtune. (fwdllm_plus's per-iteration reselect spreads dispatches across
different trainers, so it rarely re-hits the same one → ~3.5%, incidentally clean.)
**⚠ Residual unknown:** *why* `var_good_enough` reads True across a data-bin's distributes (agg_round
reports it False for iters 1–11) is not fully explained by static analysis — the distribute-time value is
decoupled from the post-aggregate agg_round value. The fix does not depend on resolving this (the sent-set
guarantees ≤1 payload/trainer/model_version regardless), but it makes the **short validation run
essential** (confirm redundancy→0, trajectory unchanged, no deadlock).

**The fix (implemented, all baselines).** A send-time set `_weights_sent_this_cycle` (independent of the
return-driven staleness map), cleared on every `_model_version` advance. A single shared decision
`_should_send_full_weights(end, is_stale)` drives **both** the sync and async loops (parity = regression
guard) and **checks the set FIRST, before the `var_good_enough` branch** — so it gates the commit-branch
re-sends that are the actual redundancy, not just the stale path. When suppression is on, the VAR=bad
variant is prepared even at `var_good_enough=True` (a commit-branch re-send to an already-served trainer
downgrades to VAR=bad; the trainer keeps training on its cached current weights → no deadlock).
Invariant: **≤1 full payload per trainer per model_version**. Flag `suppress_redundant_weights` (default
**off** = byte-on-wire identical), enabled on **all three** baselines (learning-neutral → keeps E4 fair).
Counter `redundant_weights_suppressed_total` on each `[Distribute] Done`. **Also fixed a sync telemetry
mislabel** (size/label keyed off `var_good_enough`, so any `is_stale` WEIGHTS send would be logged as
VAR=bad → now labeled by the actual payload).

**Tests/checks (so it can't silently regress).**
- **Unit:** [`tests/mode/test_fwdllm_suppress_redundant_weights.py`](../../tests/mode/test_fwdllm_suppress_redundant_weights.py)
  (9 cases) — flag-off == legacy; **commit-branch first-send weights / within-cycle repeat VAR=bad** (the
  fwdllm pattern); stale-branch same; **exactly one weights send per trainer per data-bin** (both the
  is_stale and var_good_enough repeat patterns); weights resume after the cycle clears on a commit.
- **Run-level regression:** [`expt_scripts/audit_weight_redundancy.py`](expt_scripts/audit_weight_redundancy.py)
  streams any run's telemetry → per-databin redundant fraction; `--max-redundant-frac F` exits non-zero
  over a smoke run (CI-able). Pre-fix baseline numbers above are the reference; post-fix expect ≈0%.

**Validation ✅ (2026-07-08, `run_20260708_001641_fluxtune_n10_smoke`, flag ON).** Audit:
8 databins, exactly **10 weight-sends/bin = 1 per unique trainer**, **0/8 bins with a repeat, 0.0%
redundant** (was 71–90%). `redundant_weights_suppressed_total` climbed to **299** (those re-sends
downgraded to VAR=bad). Down-bytes **264 MB vs 1278 MB counterfactual → −79.3%** (~1 GB saved on an
8-databin N=10 run; scales to the ~68 GB→single-digit-GB projection at N=100). **No deadlock:** 8
commits + 7 evals progressed normally (the VAR=bad keep-training path did not stall). Learning-neutral
by construction (trainers train on identical cached weights); a strict same-length A/B trajectory diff is
optional and not required for a byte-level change.

### 5e. Opt-2 measure-first tooling + α=0.1 tooling-validation (2026-07-08)

**Tool.** [`expt_scripts/characterize_variance_curve.py`](expt_scripts/characterize_variance_curve.py)
streams `agg_round` telemetry, reconstructs each data-bin by **commit counting** (robust to the
`data_id` cycling footgun), extracts each bin's (iteration → `var`) curve keyed on `cycle_iteration`,
and reports: per-bin distribution (initial var, commit var, min, plateau onset), **evolution over
training** (databins split in thirds), and two counterfactual sweeps that directly design the policy —
**fixed_cap** (iters/forward-passes saved + the var we'd commit at) and **plateau** (patience N, tol ε →
fire iteration + var). α-agnostic; runs over multi-GB logs. This IS the M0 reducer.

**✅ These ARE α=1 numbers (the M0 measurement).** The source run's dir name says `alpha0p1` but it
loaded `niid_label_clients=100_alpha=1` (verified from `*trainers.log`, 2026-07-08), so this is the
paper's primary operating point — a valid tuning basis, NOT α=0.1 as first assumed. Per
[[fwdllm-alpha-convention]] experiments are α=1 primary (never lower); ablations go to α∈{10,100} where
the floor should drop further (re-measure there before claiming Opt-2 helps at α≥10).

Run `run_20260706_185045_fluxtune_n100_smoke_syn_0_real` (**α=1**, 139 committed data-bins):
- **var @ commit ≈ 0.29 always** (p10 .282 / p50 .293 / p90 .298) — the gate commits only when a noise
  dip crosses thr=0.30; `var_min == var_commit` (commit = first sub-0.30 dip). **var @ iter 0 ≈ 5.4**
  (p90 10.6); iters/bin p50 **19** (p90 37) — matches §5a M-5.
- **Evolution:** bins grind LONGER as training progresses (median iters early **15** → late **34**);
  plateau (rel-drop<5% over 3 iters) fires late at iter ~33, var ~0.35 ≈ the structural floor.
- **fixed_cap sweep:** cap 12 → cuts 88.5% of bins, saves **47.9%** of iters, commits at median var
  **0.50** (p90 0.89); cap 15 → 71.9% of bins, saves 37%, commits at var 0.47 (p90 0.72); cap 10 →
  56.3% saved but commits at var 0.56 (p90 1.05, riskier). Committing at the plateau (~0.45–0.50) is the
  *denoised* estimate, arguably better signal than a lucky 0.29 noise dip.
- **plateau sweep (N=3):** ε=0.10 fires 33.8% of bins at iter ~22 (var .38); ε=0.15 fires 56.8% at
  iter 17 (var .43); ε=0.25 fires 94.2% at iter 11 (var .52). A plateau rule commits the denoised
  estimate; pair it with a fixed_cap ceiling for the bins that never flatten.

**Design consequence.** The floor sits above the threshold **at α=1** (var@commit ≈0.29 by noise dip,
plateau ~0.45), so the grind is real at the paper's operating point and **M0 is satisfied** — Opt-2 can
proceed with these numbers. Starting design point: **fixed_cap ≈ 12** (safety ceiling; −48% iters,
commit-var ~0.50) as a floor now, plus a **plateau rule (N=3, ε≈0.10–0.15)** that commits the denoised
estimate earlier on the bins that flatten, deferring to the cap for bins that never do. Re-characterize
at α∈{10,100} before claiming the win there.

## 6. Changelog
- **2026-07-08 (n) — Opt-3 grad_aware made the fluxtune DEFAULT + 4-run 2×2 ablation (STATUS, §2f-3).**
  Per operator: `type=new` is FeLiX (not a fluxtune contribution), so fluxtune now defaults to
  `agg_rate_conf.type=grad_aware` (align_gate on, base=new, inverse_var off); `new` kept only for the R1
  ablation arm. Defined the four full N=100 target_acc=84% runs as a 2×2 over {Opt-2 var-stop, Opt-3
  grad-aware}, C1+Opt-1 constant: R1 FeLiX baseline, R2 +var-stop, R3 +grad-aware, R4 all (=default).
- **2026-07-08 (m) — Opt-3 gradient-aware aggregation IMPLEMENTED (§5c, §2f-3, baselines.yaml).** Added
  `agg_rate_conf.type=grad_aware` to `aggregate_grads_from_trainers`: bounded direction/reliability-aware
  per-update weight (align gate S2 primary + optional inverse-variance S1), ≤ base so the LR never
  inflates. Pure `_grad_aware_rate`/`_cosine_flat` (13-case unit test, green; 30 total across the 3 opts).
  `agg_rate_type`/`grad_aware_gated_total` telemetry. Default `new` (byte-identical); enable per-run for
  the "new-aggregation" and "all-features" runs. Crisp STATUS + feature/run matrix added at the top.
- **2026-07-08 (l) — Opt-2 smoke compared vs var≤0.3 baseline; reducer `--max-databins`/`commit_reason`.**
  New run `run_20260708_014305` (plateau on) vs `run_20260708_001641` (var≤0.3), first 5 databins: **45 vs
  60 iters (−25%)**; plateau fired on the grindy early bin (databin 0: 17→10 iters, committed var 0.55),
  4/5 committed natural, cap unused (bins < 20, expected this early). Learning inconclusive (both ~chance
  at 5 bins). Watch: plateau at var 0.55 = a temporary-flat "false plateau" on the first high-var bin.
- **2026-07-08 (k) — Opt-2 ENABLED as the fluxtune default (operator decision; §5c, §2f-2, baselines.yaml).**
  Per the operator's regime model (plateau primary/early, cap = late-stage backstop) set
  `_metadata/baselines.yaml` fluxtune: **var_stopping_policy=plateau, N=3, ε=0.15, cap(max_iter)=20**
  (cap above the early plateau fire ~12–17, below the late grind ~33). fluxtune-only (fwdllm/fwdllm_plus
  unset). No A/B — to be compared directly against the earlier var≤0.3-only fluxtune run via a 10-trainer
  smoke. Disable/tune documented inline. Byte-identical when `off`.
- **2026-07-08 (j) — Opt-2 variance-plateau force-commit FRAMEWORK implemented (§5c, §2f-2).** Added
  `var_stopping_policy ∈ {off, fixed_cap, plateau}` (+ `var_plateau_patience` N, `var_plateau_rel_delta`
  ε; `max_iterations_per_data_id` = cap ceiling), read in `internal_init`. Pure decision
  `TopAggregator._should_force_commit_on_plateau` (rel var-drop over last N < ε while var>thr), called
  from `FedSGDAggregator.aggregate()` after the cycle var is appended to `var_prev_iter_list`; ORs into
  the existing cap path. `stopping_policy`/`commit_reason` (natural/cap/plateau) added to `agg_round`
  telemetry. **8-case unit test** `test_fwdllm_var_stopping_policy.py` (all green; Opt-1's 9 still green).
  **Byte-identical OFF.**
- **2026-07-08 (i) — Opt-2 measure-first (M0) DONE + α-label bug found (§5e, A5, §5a, M0).** Wrote
  `characterize_variance_curve.py` (per-databin variance-decay curve + evolution + fixed_cap/plateau
  counterfactual sweeps) — the M0 reducer. **Discovered the §5a/§5e source run
  `run_20260706_185045_fluxtune…alpha0p1…` actually loaded α=1** (verified from its `*trainers.log`; the
  `alpha0p1` in the dir name is a mislabel — only `…161938_fwdllm` is genuinely α=0.1). So §5's whole
  bottleneck ledger IS an **α=1** measurement, not α=0.1 — resolving the operator's "where are you finding
  α=0.1". **M0 satisfied:** the grind is real at the primary operating point (var@commit≈0.29 noise-dip,
  floor~0.45, bins 15→34 iters, cap-12 ≈ −48% iters @ commit-var ~0.50) → Opt-2 unblocked, tunable off
  §5e. **α convention (operator):** α=1 primary, never below 1; ablations go UP to α∈{10,100} (was
  {0.5,1.0}) — re-characterize there before claiming Opt-2 survives more IID. New memory
  [[fwdllm-alpha-convention]]. No aggregator/config code changed yet; next is the flag-gated
  `stopping_policy` framework.
- **2026-07-08 (h) — Opt-1 VALIDATED end-to-end.** Run `run_20260708_001641_fluxtune_n10_smoke` (flag ON):
  audit shows **0.0% redundant** (exactly 1 weight-send/trainer/databin over 8 databins), 299 re-sends
  suppressed, **−79.3% agg→trainer down-bytes** (264 MB vs 1278 MB counterfactual), and **no deadlock**
  (8 commits + 7 evals). Confirms the residual-unknown (var_good_enough lifecycle) does not stall the
  VAR=bad keep-training path. §5d + STATUS + §2f-1 updated. Opt-1 closed; next is Opt-2. (commit `5503100b`)
- **2026-07-07 (g) — Opt-1 root cause CORRECTED + fix completed (§5d).** The (f) root cause was wrong:
  staleness is **0** for active trainers (the return-map IS updated, `is_stale` is False), so the stale
  path was never the cause. The real cause: the sync loop is `distribute >> aggregate`, so distribute runs
  ~10–12×/data-bin and every call takes the **unconditional `var_good_enough → WEIGHTS` branch**, which
  my (f) fix did **not** gate → it would not have fixed fwdllm at all. Corrected: the shared
  `_should_send_full_weights` now checks the send-set **first**, gating both branches; the VAR=bad payload
  is prepared even at `var_good_enough` (commit-branch downgrade, no deadlock — trainer keeps training on
  cached current weights). Tests expanded to 9 (added the var_good_enough repeat pattern; fixed the
  now-invalid "commit always weights" case). Residual unknown (why var_good_enough reads True across a
  bin's distributes) documented; **the short validation run is required** to confirm redundancy→0 +
  unchanged trajectory + no deadlock before this is trusted.
- **2026-07-07 (f) — Opt-1 redundancy audited across baselines (§5d).** Per-databin audit: **fwdllm 90%
  redundant (22.3 GB, same 10 trainers ×10/bin), fluxtune 71% (67.6 GB, up to 21×), fwdllm_plus 3.5%**.
  Added shared `_should_send_full_weights` + `_weights_sent_this_cycle` across sync+async, a sync
  telemetry-mislabel fix, a unit test, and reusable `audit_weight_redundancy.py`. (Root cause corrected in
  (g).) Flag `suppress_redundant_weights` default off, enabled on all baselines (learning-neutral).
- **2026-07-07 (e) — code-verified implementation designs for opts 1–4 (§5c).** Verified each lever
  against the live code with anchors. **Opt-1 CONFIRMED:** intra-databin weight payload is byte-identical
  (WEIGHTS unchanged; GRAD_POOL recomputed only `if _is_model_updated`), the VAR=bad guard exists but
  `_trainer_last_model_version` is written only on grad-return, so 93.5% (24,570) of weight-sends re-ship
  the identical model mid-databin; trainer caches weights so suppression is safe → **SURE, implement +
  validate**. **Opt-2:** the `_max_iter_per_data_id` force-commit already exists but was `None` (never
  armed) — fixed cap trivial, adaptive plateau rule is the design → **MEASURE-FIRST** (characterize the
  variance-decay curve). **Opt-3:** all trust signals (var/real_var/snr/grads_snr/cv) computed then
  discarded for weighting; laid out the intuition→numbers→formulation→measurement chain (weight-entropy,
  weight↔Δloss corr, wasted-work fraction) → design. **Opt-4:** controller/policy fully wired, just
  disabled → enable+policy+run. Open decisions D-1…D-4 logged for operator. Subtasks in §2f updated.
- **2026-07-07 (d) — measured bottleneck analysis + optimization ledger (§5).** Ran a full streaming
  reducer over `run_20260706_185045_fluxtune…` and captured 12 diagnostics (§5a) + a gain-ordered
  optimization ledger (§5b) with flags. Headline findings: the **variance gate is the hub** — only
  **4.3%** of aggregation iterations commit, because the achievable variance floor (~0.45) sits **above**
  the 0.30 threshold and the force-commit escape valve **never fires** (0%); the slow model-version this
  causes makes staleness **degenerate** (61.7% zero-spread, confirming H2); and **65% of all bytes
  (~88 GB) re-send an unchanged model** (139 versions, ~26k downloads). Inferred the variance floor is
  **structural non-IID, not sample-limited** → **parked dynamic-K** (X1) and staleness-based-C (X2) as
  high-risk-low-reward. Priority order: (1) delta/version-cache comm, (2) threshold + real force-commit,
  (3) gradient-aware C3, (4) dynamic C, (5) finer staleness clock. Subtasks logged in §2f. **Caveat:**
  numbers are α=0.1; task M0 re-measures at α=1 before any tuning.
- **2026-07-07 (c) — reconciliation complete + plots/telemetry audit; `05-evaluation.tex` delivered.**
  Audited paper metrics/plots/baselines vs code (three layers) — §3. Reconciled the paper's "Expected
  plot"/"Metrics reported" lines to code+planned builds (E1 loss companion; E2/E5 → CDF; E3 bars +
  trajectory). Logged the self-contained plot/metric build tasks (§2b′) and a prioritized **§Next steps**
  (Tier 1 correctness: N6 sync sessions, E1 iterations-to-target, N3 provenance; Tier 2 figures: E2 agg
  breakdown, E5 participation, E3 trajectory, E4 msg-size; Tier 3 minor). Marked statuses: §2a paper
  DONE/delivered (SPRY+bib+numerics operator-owned in paper repo), C3 write-up (§2d) done. `05-evaluation.tex`
  handed off to the operator's paper repo.
- **2026-07-07 (b) — α resolved.** Primary **α=1** everywhere; α=0.1 excluded (too slow across all
  baselines), kept only as a one-line note; non-IID ablation → {0.5, 1.0}. Propagated to
  `05-evaluation.tex`, `EXPERIMENTS.md` §10, and this charter. Early α=0.1 smoke runs to be re-run at α=1.
  Also fixed lingering `REFL`→"real-world availability trace" and ≥3-seed→single-seed in the tex E1 block.
- **2026-07-07 — charter opened.** Reconciled `05-evaluation.tex` ⇄ `EXPERIMENTS.md`. Decisions
  A1–A5, B1–B2, C1/C2, N1/N2/N4/N7, D1–D6 recorded (§1). Deep code audit surfaced N1–N9; reducer
  gaps N3/N5/N6 logged as tasks (§2b). C3 recast as a design investigation (§2d). Open item: α (§4).
