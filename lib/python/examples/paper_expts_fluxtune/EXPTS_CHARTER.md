# FLUXTUNE evaluation charter (living doc)

**Purpose.** Current state + resolved decisions for the fluxtune eval, reconciling the paper
(`05-evaluation.tex`) with the code (`EXPERIMENTS.md`, telemetry, plots). History lives in git; this doc
holds only what's *current* plus rationale a diff wouldn't explain. Run ledger: `EXPERIMENTS.md` §10.
**Owner:** dgarg39 · **Branch:** `dg/fwdllm_sim_unavail`

## Status

Paper reconciliation ✅ delivered. Bottleneck optimizations built — all **flag-gated, byte-identical
off, fluxtune-only, unit-tested**. Measurements are at the paper's primary **α=1** point (experiments
never go below α=1; ablations go UP to α∈{10,100} — [[fwdllm-alpha-convention]]; the `…185045…alpha0p1…`
dir name mislabels an α=1 run — **verify α from the loaded-partition log line, not the dir name**).

| Opt | Feature | Flag(s) | Status |
|---|---|---|---|
| 1 | Suppress redundant intra-databin weight re-sends | `suppress_redundant_weights` (all baselines) | ✅ validated: 0% redundant, −79% down-bytes |
| 2 | Variance-plateau force-commit (plateau early + max-iter cap late) | `var_stopping_policy=plateau`, `var_plateau_patience`, `var_plateau_rel_delta`, `max_iterations_per_data_id` | ✅ default ON fluxtune (N=3, ε=0.15, cap=20) |
| 3 | Gradient-aware aggregation / C3 (align gate + inverse-var) | `agg_rate_conf.type=grad_aware` | ✅ default ON fluxtune (`type=new` = FeLiX ablation only) |
| 4 | Dynamic C | `dynamic_kc.enabled` | ⬚ wired, not enabled — remaining |

**Default fluxtune = full stack** (C1 JVP + Opt-1 + Opt-2 + Opt-3). Code:
`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`, `examples/fwdllm/aggregator/FedSgdAggregator.py`;
config `examples/_metadata/baselines.yaml`; tools `expt_scripts/{characterize_variance_curve,
audit_weight_redundancy}.py`; tests `tests/mode/test_fwdllm_*`.

**Remaining opts:** Opt-4 dynamic C (wired, unenabled) · Opt-5 finer staleness clock (bundle w/ Opt-3) ·
Opt-1 cross-databin delta-cache (larger comm piece). **Parked** (measured dead-ends): dynamic-K to lower
the floor (floor is structural non-IID) and staleness-based-C (staleness too low at syn_0).

## Four planned full runs

N=100, `target_acc=84%`, parallel — a **2×2 over the two learning-affecting opts** (C1+Opt-1 constant).
Toggled via `run_sequential.sh` CLI flags (they patch per-run config_overrides, which WIN over the
baselines.yaml catalog at launch — dry-run validated); no per-run config files needed.

Common: `--only fluxtune --mode real --num-trainers 100 --partition-method niid_label_clients=100_alpha=1
--agg-goal 10 --c 30 --target-acc 0.84 --yes` (agg_goal/C match the n100 reference run).

**Naming:** R1 is the **FluxTune-base** (C1 guided JVP perturbations + Opt-1, both ON in all four runs),
NOT FeLiX — it does forward-mode LLM perturbation fine-tuning and merely *borrows* FeLiX's scalar
aggregation rate (`agg_rate_conf.type=new`, decision N2). FeLiX targets the CNN/speech family with no
perturbation-based fine-tuning (`baselines.yaml` fluxtune spec). `select_perturbation_using_jvp=true` is a
**trainer-side** flag (the aggregator config's copy reads false and is unused).

| Run | Opt-2 var-stop | Opt-3 grad-aware | Extra CLI flags | Isolates |
|---|:---:|:---:|---|---|
| **R1** FluxTune-base | ✗ | ✗ | `--var-stopping-policy off --agg-rate-type new` | C1-only reference (≈ last night) |
| **R2** + var-stop | ✓ | ✗ | `--agg-rate-type new` | Opt-2 alone |
| **R3** + grad-aware | ✗ | ✓ | `--var-stopping-policy off` | Opt-3 alone |
| **R4** all (= default) | ✓ | ✓ | *(none)* | full fluxtune |

R2−R1 & R4−R3 = Opt-2 effect; R3−R1 & R4−R2 = Opt-3 effect; the 2×2 also gives the interaction. After:
read `commit_reason` (natural/cap/plateau) + `grad_aware_gated_total` telemetry; tune ε/cap +
align_floor/inverse_var; re-characterize at α∈{10,100}.

**LANDED 2026-07-08** (N=100, α=1 — dir names say `alpha0p1`, MISLABELED; loaded-partition log reads
`niid_label_clients=100_alpha=1`). Runs identified by `aggregator_config.json` (`var_stopping_policy` ×
`agg_rate_conf.type`); figures via `expt_scripts/figs_ablation.yaml` + blue-ramp styles in
`plotlib/baselines.py`. R4 = full default, also feeds the baseline comparison (`figs.yaml fluxtune`).

**Results — peak test accuracy (the paper number; every run diverges after → see Issue I-1).**
Peak is at/near target and rises with the opt ladder (R4 full ≈ target); no run *sustains* it.

| Run | var_stopping_policy | agg-rate type | Run dir | **peak acc @ round-1** | gap to 84% |
|---|---|---|---|---|---|
| **R1** FluxTune-base | off | new | `run_20260708_025543_fluxtune_n100_smoke_syn_0_real` | 83.00% @ 2.91h | −1.0 |
| **R2** +var-stop | plateau | new | `run_20260708_025616_fluxtune_n100_smoke_syn_0_real` | 82.25% @ 4.07h | −1.8 |
| **R3** +grad-aware | off | grad_aware | `run_20260708_025636_fluxtune_n100_smoke_syn_0_real` | 83.91% @ 2.44h | −0.1 |
| **R4** full (=default) | plateau | grad_aware | `run_20260708_025716_fluxtune_n100_smoke_syn_0_real` | **84.08% @ 3.90h** | **+0.1** |

**Baseline comparison (`figs.yaml`, R4 = fluxtune):** FluxTune **84.1% @ 3.9h** · FwdLLM++ 80.9% @ 7.4h
(−3.1 from target, ~2× slower) · FwdLLM 30.3% (never learned, shown full). E1 headline (speed + reaching
target) holds; the peak-accuracy ★ + legend value on every E1 acc plot shows each baseline's gap to target.

## Bottleneck (measured, α=1 run `run_20260706_185045_fluxtune_n100…`)

**The variance gate is the hub:** the fixed absolute `var≤0.3` gate sits *below* the achievable variance
floor (~0.45), so a data-bin commits only on a noise dip and grinds 15→34 iters. The resulting slow
model-version makes staleness degenerate and drives long non-commit comm stretches.

| # | Metric | Measured | Meaning |
|---|--------|----------|---------|
| M-1 | agg iters that commit | 4.3% | 95.7% of compute advances nothing |
| M-3 | variance floor vs threshold | ~0.45 med / 0.30 min vs thr 0.30 | gate crossed only by noise dips |
| M-5 | iters/data-bin (max) | 18 (61) | bins grind past diminishing returns |
| M-6 | force-commit firings | 0.0% | escape valve inert → Opt-2 |
| M-8 | staleness (median/p90) | 1 / 3 | model-version crawls → rate can't differentiate |
| M-10 | redundant weight bytes | ~88 GB, ≈99% re-sends | unchanged model re-pulled → Opt-1 |
| M-12 | acc 50/75/100% wall | 0.825 / 0.743 / 0.806 | mid-run regression → aggregation instability → Opt-3 |

Floor is **structural** (grad pool grows but `var` asymptotes) → more samples won't lower it; "weight
smarter" (C3) or "accept the floor" (plateau) will → dynamic-K de-prioritized.

### ⚠ Issue I-1 — runs don't hold the minimum (ROOT-CAUSED; fix = `fluxtune_contributions.md` §8)

**Status: root-caused, fix in progress (§8 track, next = S1).** All 4 ablation runs learn in round 1
(peak 82-84%, min loss 0.55-0.66) then oscillate and collapse to ~25% (mcc 0) with loss blow-up (R4
loss→4.9); force-stopped 2026-07-08. Paper uses each run's round-1 peak; E1 plots clipped at peak. Not a
blocker for the E1 headline.

**Where the variance COMES FROM (new, see Reproducibility above):** not the seed — fp16 `autocast` round-off
in the two JVP forward passes (~1.6e-3), amplified a median **72×** by the central difference at h=0.01.
Same mechanism, upstream end; it also means seeding cannot damp I-1, and that **S1's momentum/EMA should
shrink the replicate spread too — re-measure the floor after S1 lands.**

**Root cause (H0 diagnostic — `fluxtune_contributions.md` §8, F1-F15):** an **undamped, high-variance
forward-gradient optimizer** — each noisy JVP commit applied raw (`FedSgdAggregator.py:322`, no momentum/EMA)
→ the model random-walks. Severity tracks aggregation aggressiveness (R4 grad-aware worst; R1 least → Opt-3
as tuned *amplifies*, `align_floor=0`/`inverse_var` off). **Refuted:** the epoch-boundary reset bug (F10 —
`_model_version` monotonic across the boundary) and data class-bias (F11 — K=10 cohort near class-balanced
at α=1). The frozen data schedule only *freezes* the noise → position-locked collapse (F13).

**Fix:** server-side optimizer (momentum/weight-EMA) = descent, not random walk (§8 **S1** / EXPERIMENTS.md M2);
interim tempering = cross-round LR decay + grad-aware `align_floor>0`/`inverse_var` (§8 S3). Also make the stall
guard **divergence-aware** (fire on rising loss) — today it can't: W=20 needs consecutive bins ≥0.84 (accuracy
only grazed it) and the `either` guard stays alive while round-1 loss still improves, so it can't fire before
round 2 destroys the model.

**Opt tuning (kept, `characterize_variance_curve.py`):** var@commit≈0.29 (noise dip), plateau ~0.45; cap-12 ≈
−48% iters → chose plateau **ε=0.15** + **cap=20**. Opt-1 audit: fwdllm 90%/22 GB, fluxtune 71%/68 GB → 0% post-fix.

## Latest figures (for paper embedding)

Four PDF sets, same 7 basenames, from `make_paper_figs.py` (cutoff `--cutoff-mode peak_acc` default →
every run clipped at its peak, Issue I-1 tail excluded; E1 acc plots carry a ★ + legend "peak X%" per
run). Rebuild: `cd expt_scripts && python make_paper_figs.py --manifest <m> [--out-root <r>]`. Output
is flat and overwritten each render (stable basenames, no timestamped subdirs or `latest` symlink,
decided 2026-07-24) — copy PDFs into Overleaf by basename directly.

| Set | Manifest | Dir |
|---|---|---|
| **Baseline comparison** (FwdLLM / FwdLLM++ / FluxTune=R4) | `expt_scripts/figs.yaml` | `expt_scripts/paper_figs/` |
| **FluxTune opt-ablation** (R1–R4, blue ramp) | `expt_scripts/figs_ablation.yaml` | `expt_scripts/paper_figs_ablation/` |
| **`sec:eval:sota` (top-row anchors)** | `expt_scripts/figs_main_v2.yaml` | `expt_scripts/paper_figs_main_v2/` |
| **`sec:eval:attribution` (+IT staircase)** | `expt_scripts/figs_attribution.yaml` | `expt_scripts/paper_figs_attribution/` |

Figure basenames (all sets): `e1_acc_vs_time.pdf` (time-to-acc, peak ★) · `e1_loss_vs_time.pdf` ·
`e2_trainer_busy_cdf.pdf` · `e3_dloss_per_gpu_hour.pdf` · `e3_dloss_per_mfwd.pdf` ·
`e4_network_bytes.pdf` · `e5_session_cdf.pdf`. Each dir also has `manifest.json` (run dirs + cutoff h).
See [`PLOT_TRACKER.md`](PLOT_TRACKER.md) for the last two sets' per-baseline readiness.

**Palette preview** (dummy data, no telemetry — eyeball a `plotlib/baselines.py` color/marker change in
seconds instead of waiting on a real render): `cd expt_scripts && python preview_palette.py
[--baselines k1,k2,...|--manifest <m>] [--charts line,scatter,bar]` → `expt_scripts/palette_preview/`
(same flat/overwrite convention).

## ⚠ Reproducibility — how many runs a reported number needs (OPEN, gated on H12)

**Status: measured, root-caused, fix under test. Do not finalize an error bar or a seed count until the H12
probe returns** (`fwdllm/simulate_fwdllm.md` §B-H12; probe is
`fwdllm/expt_scripts/probe_jvp_determinism.py --sweep`, minutes on one GPU, no FL run).

**What was measured.** Two 7200s real runs per baseline, same seed (1234), config-identical, different nodes:

| baseline | peak acc A | peak acc B | gap | iters/bin floor |
|---|---|---|---|---|
| `felix_round` | 77.17% | 66.01% | **11.16 pts** | 13.3% |
| `fedbuff_round` | 75.11% | 73.32% | 1.79 pts | 3.9% |

**It is not a seeding bug.** Both runs agree on every RNG-stream coordinate — same `client_idx` (fixed by the
registry), same partition, same `(data_id, iteration, model_version)`, `perturbations_total` and
`forward_passes_total` on every trainer's first task. The divergence is floating-point: 8 of 30 trainers
reproduce their first-task loss bit-exactly and the other 22 differ at a median 1.6e-3 (fp16 `autocast`),
which `calculate_jvp`'s central difference at h=0.01 amplifies by a **median 72× (max 1311×)** into the
gradient. 7 of the 8 bit-identical-loss trainers also have bit-identical gradients, so nothing else injects
randomness.

**This is Issue I-1 seen from the input end.** I-1 is already root-caused as an undamped high-variance
forward-gradient optimizer; this supplies where the variance originates and shows seeding cannot remove it.
The planned I-1 fix (server-side momentum/EMA, §8 S1) is also the damping that should shrink this spread —
**re-measure the floor after S1 lands**, it may be the cheaper lever than buying runs.

**Why it differs per baseline** — how much loss-derived `stat_utility` reaches the model update, and whether
iterations are capped. Cohorts are bit-identical between replicates (Jaccard 1.000), so it is not selection:

| baseline | `agg_rate_type` | weight depends on | iter cap |
|---|---|---|---|
| `fedbuff_round` | `old` | staleness only (integer) | none |
| `felix_round`, `felix_it` | `new` | `+ β(stat_utility)` — loss-derived | none |
| **`fluxtune`** | `grad_aware` | align gate + inverse-var | **plateau + `max_iterations_per_data_id=20`** |

`fluxtune` — the baseline the headline numbers come from — is the only one with the cap, which bounds the
compounding. Its own floor is **UNMEASURED** (no replicate exists); measuring it is the single most important
missing number for the paper, because every E1 accuracy claim rests on it.

**Decision, once H12 returns.** If a flag (`FWDLLM_JVP_FP32` / `FWDLLM_STRICT_DETERMINISM`) collapses the
spread, this is a bug: fix it, keep single-seed, delete this section. If it does not, then per-baseline
replicate error bars are mandatory on every reported accuracy, `n` sized from that baseline's measured floor
(not a conventional 3), and the paper must say the pipeline is not bit-reproducible and why.

**Immediate action regardless of H12: measure `fluxtune`'s floor.** One extra `fluxtune` real replicate at
the run length the paper uses. Until it exists, the ±  on the headline 84.08% is unknown.

## Resolved decisions

| # | Decision |
|---|---|
| **A1** Model | DistilBERT-base-uncased (66.4M, frozen) + AdapterHub adapters (~1.5% trainable). Drop 7B claims. |
| **A2** Task | AG News 4-class topic classification. |
| **A3** Hardware | A40, 8-GPU box + a *modeled* mobile delay (`delay_factor=2`); mobile figures argued from structure, not measured. |
| **A4** Trace | `mobiperf_*` is the condition of record; REFL (`third_party/REFL/`) is a scoped future item. |
| **A5** α | Primary **α=1**, never below 1; non-IID ablation → **α∈{10,100}** (more IID). [[fwdllm-alpha-convention]] |
| **B1** FwdLLM_Plus | FwdLLM + per-iteration reselection [defining] + oracular availability (inert at syn_0 → scope to unavailability) + relaxed staleness. Sync, random, agg_goal=10. |
| **B2** Contributions | **C1** guided (JVP-magnitude) perturbation · **C2** dynamic K/C · **C3** gradient-aware aggregation. Async/iteration-level is the *substrate*, not numbered. |
| **C1/C2** E3/E4 | Retain E3/E4 as hypotheses (assume C2+C3 deliver); E1 stands on C1 alone. |
| **N2** C3 | `type=new` fedbuff rate is a borrowed **FeLiX** placeholder (scalar staleness×utility), NOT fluxtune's C3 → replaced by `grad_aware` (Opt-3, default on). |
| **N4** Clock | Report wall + rounds + data-bins + iterations; drop virtual-clock (note as future). |
| **N7** Memory | Motivation/design claim argued from structure (peak mem bounded by inference); no memory experiment. |
| **D1** Seeds | ⚠️ **REOPENED — single seed is not defensible on the uncapped baselines.** Two same-seed, config-identical *real* replicates of `felix_round` differ by **11.16 accuracy points at peak** (77.17% vs 66.01%); `fedbuff_round` differs by 1.79 pts. Seeding is not the cause and cannot fix it (§ Reproducibility below). "≥3-seed medians" was chosen by convention, not from a measured spread — **size n from the floor, per baseline, once H12 resolves.** |
| **D2** GPU-sec | Forward-pass count is the primary denominator; GPU-seconds secondary w/ an 8-GPU-contention caveat. |
| **D3** Fidelity | Accuracy parity (not time-to-accuracy) vs `xu2024fwdllm`; do **not** compare wall-clock. |
| **D4** SPRY | Exclude split/personalized-per-device FL; 1-line Baselines pointer + Background *why* (bib: spry, split-learning, HeteroFL, personalized-FL survey). |
| **D6** Surcharge | P=10 ⇒ 2P=20 fwd/iter ⇒ ~10× sync compute at P=10 → parity at P=1; fidelity opt −37% GPU (sync −68%), zero gradient change. |

## Remaining work

- **Paper repo (operator-owned):** D4 SPRY/excluded-baseline justification + `\tbd` bib keys; fidelity
  accuracy value; confirm the `~1.5%` / `delay_factor=2` / `~3.6s` numerics.
- **Telemetry/reducers:** N3 (E1 time-to-τ reconstructs convergence instead of reading `converge.json`) ·
  N5 (E2 idle `mqtt_fetch_s` emitted-but-unused) · N6 (E5 sync-session should use one-round-span, not
  `contributor_intervals`) · `real_gpu_time_s`↔`gpu_compute_s` schema footgun.
- **Figures (`plotlib/figures.py`):** E1 iterations-to-target metric · E2 aggregator-breakdown bar ·
  E3 Δloss-vs-cumulative-compute trajectory · E4 msg-size inset · E5 participation-count bars ·
  unify `FwdLLM++`/`FwdLLM_Plus` label.
- **Runs:** **`fluxtune` replicate for the floor (BLOCKING the E1 error bar — see Reproducibility)** ·
  full-system E2/E3/E4 (C2+C3 on) · mobiperf E1 (needs the fwdllm_plus-under-scarcity barrier decision) ·
  fidelity accuracy-parity (locate old run) · multi-seed E1 medians, `n` sized from the measured floor once
  H12 resolves (no longer "optional ≥3", D1).
- **Ablations:** C1 JVP sensitivity (threshold, refresh) · C2 K/C sensitivity · α∈{10,100}.

## Principles

**P1** substrate honesty — describe what ran, or run the claimed substrate before submission. **P2**
isolation vs full-system are two conditions — efficiency claims (E2–E4) need C2+C3 on. **P3** no takeaway
ships ahead of its evidence. **P4** directional source-of-truth — implementation facts flow code→paper,
narrative paper→code. **P5** one name per concept.
