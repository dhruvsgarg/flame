# FLUXTUNE evaluation charter — paper ⇄ code reconciliation (living doc)

**Purpose.** Single source of truth for reconciling the paper draft
([`05-evaluation.tex`](05-evaluation.tex), conceptual/narrative) with the code-side
design doc ([`EXPERIMENTS.md`](EXPERIMENTS.md), implementation/telemetry/plots). This file
records **conflicts resolved** (the charter) and **tracks tasks** as they are done. The run
ledger (which log file feeds which result) lives in [`EXPERIMENTS.md`](EXPERIMENTS.md) §10.

**Owner:** dgarg39 · **Branch:** `dg/fwdllm_sim_unavail` · **Opened:** 2026-07-07

**STATUS (2026-07-07):** ✅ **Reconciliation complete; `05-evaluation.tex` delivered** and being moved
back into the paper repo by the operator. Both docs are in sync; the run ledger (EXPERIMENTS.md §10) and
this charter are current. **Remaining work is all code/implementation** — see **§Next steps** below.
Two large items (C2 dynamic-K/C, C3 gradient-aware aggregation) are tracked in §2c/§2d but are
**out of scope for the immediate next steps** (operator is designing them separately).

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
| **A5** Non-IID α | Paper {0.1, 0.5}; `experiments.yaml` main=1; early runs ran α=0.1 | **RESOLVED: primary α=1** (matches `experiments.yaml main`). **α=0.1 excluded** — learning too slow across all baselines to finish convergence runs; kept only as a one-line "excluded, too slow" note. Non-IID ablation → {0.5, 1.0}. The α=0.1 smoke runs will be **re-run at α=1** for final numbers. | tex + code |
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
- ☐ Implement fluxtune's gradient-aware aggregation; disable the borrowed fedbuff "new" placeholder.
- ☐ On landing: move feature doc → `fluxtune_contributions.md`; remove from `EXPERIMENTS.md`.

**C3 dimensions to evaluate (seed for the investigation):**
1. **Staleness definition under iteration-based progression.** Staleness = `agg_model_version − trainer_version`; `_model_version` advances **per data-bin completion**, and a data-bin only completes when the **variance threshold is met** → staleness accrues at the variance-gated data-bin rate, *not* wall-clock. Hypothesis: staleness grows **slower** in fluxtune than round-based FL. Quantify the effective staleness distribution vs a round-based baseline.
2. **Scalar rate vs gradient-aware combination.** The current `weight_factor` scalar-multiplies the whole update — borrowed from weight-averaging async FL. For *gradient* updates (forward-mode JVP estimates), a scalar down-weight may be the wrong operator. Explore direction-/variance-aware combination (e.g. weight by JVP magnitude / SNR / agreement with the running aggregate), not just staleness×loss.
3. **Interaction with C1 (guided perturbations) & the variance gate.** Updates already passed a `var ≤ var_threshold` gate; does re-weighting by loss (`stat_utility`) double-count what the gate filtered?
4. **Interaction with C2 (dynamic K/C).** Concurrency (C) sets how many stale/in-flight updates coexist; the aggregation rule and the concurrency controller co-determine wasted work (the E3/E4 root cause).

### 2e. Ablations (paper has them planned; no tooling yet — D5)
- ☐ JVP guidance sensitivity (C1): threshold, refresh frequency.
- ☐ K/C policy sensitivity (C2): window N, growth/shrink factors, C_max.
- ☐ Non-IID robustness: Dirichlet α ∈ {0.5, 1.0} (α=0.1 excluded, too slow — A5).
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

- _(none open)_ — **A5 (α) resolved** (2026-07-07): primary **α=1**; α=0.1 excluded (too slow across all
  baselines), retained only as a one-line note; non-IID ablation → {0.5, 1.0}. The α=0.1 smoke runs
  (ledger, EXPERIMENTS.md §10) are to be **re-run at α=1** for final numbers. Follow-up (not blocking):
  confirm the earlier config↔run mismatch (runs ran α=0.1 while `experiments.yaml` recorded α=1) doesn't
  recur when the α=1 runs launch.

---

## 5. Changelog
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
