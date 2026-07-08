# FLUXTUNE evaluation charter (living doc)

**Purpose.** Current state + resolved decisions for the fluxtune eval, reconciling the paper
(`05-evaluation.tex`) with the code (`EXPERIMENTS.md`, telemetry, plots). Change history lives in git;
this doc holds only what's *current* and the decisions/rationale a diff wouldn't explain. The run ledger
(which log feeds which result) is `EXPERIMENTS.md` §10.
**Owner:** dgarg39 · **Branch:** `dg/fwdllm_sim_unavail`

## Status

Paper reconciliation ✅ delivered. Bottleneck optimizations built — all **flag-gated, byte-identical
off, fluxtune-only, unit-tested**. Measurements are at the paper's primary **α=1** operating point
(experiments never go below α=1; ablations go UP to α∈{10,100} — [[fwdllm-alpha-convention]]; the
`…185045…alpha0p1…` dir name mislabels an α=1 run — **verify α from the loaded-partition log line, not
the dir name**).

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

N=100, `target_acc=84%`, parallel — a **2×2 over the two learning-affecting opts** (C1+Opt-1 constant):

| Run | Opt-2 var-stop | Opt-3 grad-aware | Overrides vs default | Isolates |
|---|:---:|:---:|---|---|
| **R1** FeLiX baseline | ✗ | ✗ | `var_stopping_policy: off` **+** `agg_rate_conf.type: new` | reference (≈ last night) |
| **R2** + var-stop | ✓ | ✗ | `agg_rate_conf.type: new` | Opt-2 alone |
| **R3** + grad-aware | ✗ | ✓ | `var_stopping_policy: off` | Opt-3 alone |
| **R4** all (= default) | ✓ | ✓ | none | full fluxtune |

R2−R1 & R4−R3 = Opt-2 effect; R3−R1 & R4−R2 = Opt-3 effect; the 2×2 also gives the interaction. After:
read `commit_reason` (natural/cap/plateau) + `grad_aware_gated_total` telemetry; tune ε/cap +
align_floor/inverse_var; re-characterize at α∈{10,100}.

## Bottleneck (measured, α=1 run `run_20260706_185045_fluxtune_n100…`)

**The variance gate is the hub:** the fixed absolute `var≤0.3` gate sits *below* the achievable variance
floor (~0.45), so a data-bin commits only on a noise dip and grinds 15→34 iters. The slow model-version
this causes makes staleness degenerate and drives long non-commit comm stretches.

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

**α=1 tuning (`characterize_variance_curve.py`):** var@commit≈0.29 (noise dip), plateau ~0.45; cap-12 ≈
−48% iters. Chose plateau **ε=0.15** (fires ~iter 17, denoised var ~0.43) + **cap=20** (late backstop).
Opt-1 audit: pre-fix fwdllm 90% / 22 GB, fluxtune 71% / 68 GB → 0% post-fix.

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
| **D1** Seeds | Single seed now; paper TODO for ≥3-seed medians on headline E1. |
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
- **Runs:** full-system E2/E3/E4 (C2+C3 on) · mobiperf E1 (needs the fwdllm_plus-under-scarcity barrier
  decision) · fidelity accuracy-parity (locate old run) · optional ≥3-seed medians for E1.
- **Ablations:** C1 JVP sensitivity (threshold, refresh) · C2 K/C sensitivity · α∈{10,100}.

## Principles

**P1** substrate honesty — describe what ran, or run the claimed substrate before submission. **P2**
isolation vs full-system are two conditions — efficiency claims (E2–E4) need C2+C3 on. **P3** no takeaway
ships ahead of its evidence. **P4** directional source-of-truth — implementation facts flow code→paper,
narrative/positioning paper→code. **P5** one name per concept.
