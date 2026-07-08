# Fluxtune: Systems & ML Contributions for On-Device LLM Fine-Tuning

**Thesis.** Fluxtune makes federated LLM fine-tuning practical on *memory- and hardware-constrained*
devices by training with **forward-mode (backprop-free) gradients** over **parameter-efficient adapters**,
aggregated **asynchronously**, with an **informed (JVP-magnitude) perturbation selection**. The result is a
training regime whose **peak memory is independent of both model depth and the number of perturbations**, whose
compute is **pure forward inference** (no autograd graph, no backward GEMMs), and which therefore maps onto the
**inference-only accelerators** (mobile NPUs/DSPs) that cannot run backpropagation at all. This document states the
contributions precisely, quantifies them against baselines and alternatives, and separates what is *measured* from
what is *argued*.

> Numbers below are from `scripts/profile_jvp_opt.py` (reuses the production `create_model` + `calculate_jvp`):
> DistilBERT-base + AdapterHub bottleneck adapters, batch 8, seq 192, fp16/autocast, NVIDIA A40. They are a *clean
> single-trainer* profile; a shared-GPU run multiplies wall-time by the contention factor, but **pass-counts,
> ratios, and memory transfer directly**. Mobile figures are *argued from structural properties*, not measured.

---

## 1. The problem & why the baseline choice matters

On-device fine-tuning of a transformer via **backpropagation** requires (i) a full **autograd graph** (stored
activations for the backward pass), (ii) **training-mode** kernels (backward GEMMs, transposes), and (iii) an
optimizer state. Mobile SoCs expose **inference-optimized** NPUs/DSPs (forward GEMMs, quantized, no autograd) and
tight memory budgets — so backprop fine-tuning is often infeasible on the device that owns the data.

**Forward-gradient** methods (FedFwd / FwdLLM family) replace the backward pass with **directional derivatives**
estimated from *forward passes only*. Fluxtune is a system built on this idea. Its baselines in this repo are the
**sync** forward-grad variants (`fwdllm`: cosine-similarity perturbation selection; `fwdllm++`: per-iteration
reselection). Fluxtune's distinguishing choices — **async aggregation + JVP-magnitude selection + adapter PEFT +
server-side LR** — are what this document evaluates.

---

## 2. The computation, precisely

Each perturbation's gradient estimate is a **central finite-difference JVP** (`fwdgrad_utils.calculate_jvp`):

```
jvp(v) = ( f(θ + h·v) − f(θ − h·v) ) / (2h),     h = 0.01,     under autocast + no_grad
```

- `f` = one full forward pass of the (frozen-backbone) model + cross-entropy loss. **1 perturbation = 2 forward
  passes.** No backward pass, no stored graph.
- **Fluxtune** evaluates `P` candidate perturbations (`perturbation_count`, default 10) and **selects the one with
  the largest |jvp|** (the steepest measured descent direction) → `2P` forward passes. The **sync `fwdllm`
  baseline** selects by cosine similarity to a cached gradient → **0 forward passes** for selection, 1 final JVP.
- **PEFT:** the DistilBERT backbone (66.4M params) is **frozen**; only **bottleneck adapters in every layer + the
  head** are trainable — **1.04M / 67.4M = 1.5%**. The perturbation `v` is non-zero only on trainable params.

---

## 3. Systems contributions

### 3.1 Memory is flat — in perturbation count *and* in depth
Because there is **no autograd graph**, peak memory ≈ *model weights + one in-flight forward*. It does **not grow
with `P`** (perturbations are evaluated one at a time and discarded) and does **not accumulate activations for a
backward**.

| regime | peak memory (measured) |
|---|---|
| forward-grad, any P (trainable-only, retained) | **3.19 GB** |
| forward-grad, any P (current) | 3.44 GB |
| backprop reference (1 fwd + 1 bwd) | 3.71 GB |

At this favorable-to-backprop config (small batch, 98.5% frozen) the gap is ~14%; it **widens with batch size and
sequence length** (backprop's stored activations scale with both; forward-grad's do not) and with the **trainable
fraction**. The durable claim is structural: **forward-grad removes the autograd graph entirely**, so memory is
bounded by inference, not training.

### 3.2 Compute is pure forward inference (hardware fit)
Every FLOP fluxtune spends is a **forward-pass GEMM** — the exact operator set an inference NPU/DSP is built for.
There are **no backward GEMMs, no transposed weight matmuls, no autograd bookkeeping**. This is the crux of the
**mobile-generality argument**: a device that can *run* the model can *train* it under fluxtune, with no autograd
runtime. (Argued from operator structure; not yet measured on-device.)

### 3.3 Compute/latency cost — characterized honestly
Forward-grad trades memory for **time**: many forward passes instead of one forward+backward. The selection is the
surcharge, and it is linear in `P`:

| path | fwd passes | ms/batch (clean A40) |
|---|---|---|
| sync `fwdllm` (opt) | 2 | 16 |
| **fluxtune P=1** | 2 | 16 — *equals sync* |
| fluxtune P=5 | 10 | 80 |
| **fluxtune P=10** | 20 | 159 |
| backprop reference | 1f+1b | 17 |

Fluxtune's cost is **`2P × per-pass`** → **10× the sync compute at P=10**, collapsing to parity at `P=1`. The
JVP-selection is thus a **tunable accuracy/compute knob**, not a fixed tax. Per-pass ≈ 8–10 ms here; a shared-GPU
deployment multiplies wall-time by the contention factor (the real 10-trainer run saw ~0.21 s/pass).

### 3.4 Communication
Only the **1.5% adapter** parameters are exchanged per round (PEFT), and async aggregation (`agg_goal=3 < K=10`)
commits updates **as stragglers arrive** with no synchronization barrier — directly targeting the intermittent
connectivity and device heterogeneity of the mobile setting.

---

## 4. Machine-learning contributions

### 4.1 Informed perturbation selection (JVP vs cosine / random)
Random-direction forward-grad (MeZO-style) and the cosine-similarity baseline pick a perturbation *without*
measuring its effect on the loss. Fluxtune **measures** each candidate's directional derivative and keeps the
steepest — a better single-sample gradient estimate per communication round, improving sample/round efficiency at
the cost of the `2P` forward passes (§3.3). The value of the knob `P` is an accuracy/compute trade the operator
controls.

### 4.2 Finite-difference numerics & precision (a measured caution + an exactness result)
- The FD estimate subtracts two **O(1)** losses that differ by **O(h)** ≈ 1e-3. Under **fp16/fp32** this is
  **catastrophic cancellation**: the JVP retains only ~1–2 significant figures, so the perturbation *ranking* is
  mildly precision-limited. This is a genuine finding about forward-grad-by-finite-difference in mixed precision,
  relevant to any deployment that lowers precision for the mobile NPU.
- **Batching is mathematically exact.** Vectorizing all `P` perturbations (`torch.func.vmap`) yields JVPs that are
  **bit-identical to the sequential loop in fp64** and deterministic run-to-run — the fp32 divergence is *only* the
  cancellation above, not a batching error. This bounds when the 2× batching speedup is safe to adopt.

### 4.3 Tensor-operation profile
The finite-difference perturbation touches **only trainable tensors** (`p ± h·v` with `v=0` on frozen params, so
`p−0=p` exactly). The forward is otherwise identical inference. There is **no backward transpose-GEMM, no
grad-accumulation kernel**. This minimal op set is what makes the trainable-only optimization (§5) and the
inference-hardware mapping (§3.2) possible.

---

## 5. Fidelity-preserving optimizations (retained)

Validated **bit-identical** (`max|Δjvp| = 0`) by the profiler — they change *cost*, never the computed gradient,
so training fidelity and real↔sim simulator parity are untouched:

1. **Trainable-only finite difference** — skip the `p − h·0 = p` arithmetic on the 98.5% frozen backbone:
   **1.26× faster, −251 MB.**
2. **Remove redundant/diagnostic forward passes** — 3 passes that only fed a log line, plus (fluxtune) reusing the
   selected perturbation's already-computed JVP: **fluxtune 25→20 passes, sync 5→2.**

**Combined: sync −68%, fluxtune −37% GPU time, zero fidelity change.** Measured under the real 10-trainer / 8-GPU
run this brings fluxtune's **mean** per-batch compute (3.61s) under the 4.0s modeled mobile-delay budget, but the
**tail** (4.1–5.4s) still overruns on the two GPUs that carry 2 trainers each (10>8) plus the aggregator's eval
GPU — a contention effect, not the JVP cost. Clearing the tail needs the sim's GPU pipelining fix (keeps compute
near the ~2.4s uncontended floor) and/or 1-trainer-per-GPU.

**Deliberately *not* adopted** (they change fidelity / are inferior here): **vmap batching** (2× but re-baselines
the fp32 trajectory via §4.2 cancellation — exact only in fp64), **exact forward-mode AD** (0.5×, needs eager
attention, different math), **lowering `P`** (changes the algorithm — an accuracy knob, not a free optimization).

---

## 6. Comparison summary

| dimension | backprop FL | sync forward-grad (`fwdllm`/`++`) | **fluxtune** |
|---|---|---|---|
| training memory | activation graph (grows w/ batch·seq·depth) | flat (inference) | **flat (inference)** |
| hardware needed | autograd/training runtime | inference-only | **inference-only** |
| perturbation selection | n/a | cosine to cached grad (0 fwd) | **JVP magnitude (2P fwd, informed)** |
| aggregation | sync/async | **sync (barrier)** | **async (straggler-tolerant)** |
| compute/round | 1 fwd + 1 bwd | 1 JVP (2 fwd) | 2P fwd (tunable) |
| communication | full or PEFT | PEFT adapters | **PEFT adapters** |
| best when | server-class HW | homogeneous, fast clients | **memory/HW-constrained, heterogeneous, intermittent clients** |

**Where fluxtune wins:** the on-device regime — constrained memory, inference-only accelerators, stragglers, and
intermittent availability — where backprop is infeasible and a synchronization barrier stalls on the slowest phone.
It buys a better per-round gradient (JVP selection) and straggler tolerance (async) for a forward-pass compute cost
that the §5 optimizations cut ~40% without any fidelity loss, and that `P` tunes directly.

**Honest limits:** fluxtune is **compute-heavier** than the sync baselines (≈10× at P=10) and than backprop
per-round; its advantage is memory/hardware feasibility and robustness, not raw FLOPs. The FD JVP is
precision-sensitive in fp16 (§4.2). The memory gap over backprop is modest at small scale and grows with
batch/seq/trainable-fraction.

---

## 7. Reproducibility

All figures: `scripts/profile_jvp_opt.py` (env `test_fwdllm`), which reuses the production model builder and JVP
math so measured gains transfer directly to the trainer. It reports, per stage, forward-pass count, latency
(mean±std, warmup + `cuda.synchronize`), peak memory, speedup, and a BIT-IDENTICAL / WITHIN-TOL / DIVERGED verdict
against the ground-truth sequential path, plus an fp64 cancellation diagnosis (`--fp64-check`). The high-fidelity
real↔sim simulator that validates fluxtune's *training dynamics* under a virtual clock is documented separately in
`simulate_fwdllm.md` (compute profile persisted there in §L).

---

## 8. Training-stability track — root cause & resolution (LIVE)

> **DOC DISCIPLINE — STRICT. This section is *current truth*, not a log.**
> 1. **Edit in place. Do NOT append.** When a finding or fix changes state, **rewrite its existing row** —
>    never add a dated "update:" note, a changelog entry, or a sibling row. Git holds the history; this
>    section holds only what is true *now*. (This is the opposite of `EXPERIMENTS.md`, which keeps a changelog.)
> 2. **One row per finding, one row per fix.** If a finding is superseded or refuted, overwrite it. No duplicates.
> 3. **Every claim carries a status tag:** `VERIFIED` (evidence cited) · `SUSPECTED` (hypothesis, unchecked) ·
>    `REFUTED` (checked false — keep the row, it stops us re-chasing it) · `TODO` (fix not started) ·
>    `WIP` · `DONE` (fix landed **and** sanity-checked).
> 4. **No fix moves to `DONE` without its sanity check** named in the same row (telemetry signal / unit test /
>    metric that proved it). Sanity checks are first-class here, not afterthoughts.
> 5. **Crisp only:** claim · evidence · status. No narration, no prose paragraphs in the ledgers.
> 6. **Flag-gate every change for A/B; the lifecycle decision is the operator's, not mine.** Each fix lands
>    **behind a named flag, default = old behavior (byte-identical off)**, so old vs. new run A/B. Name the flag
>    in the row's "Code — how". A flagged change moves through lifecycle states: **`A/B`** (both variants live,
>    testing) → one of **`PERMANENT`** (new is default, old fully deprecated/removed) · **`FLAGGED`** (new default
>    but flag retained) · **`REVERTED`** (new dropped, old reinstated behind the flag). **Never pick the terminal
>    state unilaterally — ASK the operator** with the A/B evidence; record the chosen state (and the deciding
>    metric) in the row, in place.

**The issue (one line).** On the N=100 α=1 runs the global model never converges — it **oscillates** with
recurring **single-class collapses** (acc 0.250, mcc 0.000 on balanced 4-class); peaks are transient and the
*same* positions collapse every epoch. Root cause (H0): an **undamped, high-variance forward-gradient
optimizer** (F6-F9) — each noisy JVP commit is applied raw, so the model random-walks. Data/heterogeneity is
**not** the driver (F11 refuted class-bias; F13: the fixed data schedule only *freezes* the noise → the
position-lock). Supersedes the charter's I-1 "epoch-boundary bug" framing (F10 refuted).

### 8.1 Verified-findings ledger (sanity checks / things found)

| # | Finding | Evidence | Status |
|---|---|---|---|
| F1 | `data_id` selects the *actual training data*: at `data_id=k` a trainer trains on ONE fixed 8-sample batch `train_local_list[0][k]` | `FedSgdTrainer.py:500-505`, `:148-150`; `train_batch_size=8` (`configs/aggregator_base.json:37`); 1200 samples/client → 150 bins | VERIFIED |
| F2 | Per-client **train** loader is `shuffle=False` (SequentialSampler) → raw partition index order; the global **test** loader is `shuffle=True` | `base_data_manager.py:490` vs `:287` | VERIFIED |
| F3 | Bins materialized once, **never reshuffled across epochs**; round boundary resets `data_id=0` only → round 2 replays identical batch order | `FedSgdTrainer.py:148-150`; `fwdllm_aggregator.py:1940-1946` | VERIFIED |
| F4 | Down-swings are **single-class collapse**, not generic noise: acc = 0.250 **and** mcc = 0.000 exactly (32 of 150 R1 bins) | `agg_eval` telemetry, run `…025543…` | VERIFIED |
| F5 | Collapses are **position-locked and replay**: R1 and R2 collapse at the same `data_id` ranges (≈4-7, ≈63-79) | `agg_eval` telemetry | VERIFIED |
| F6 | Early variance-gate commits yield worse models: commit at it<15 → mean acc **0.289**; it≥20 → **0.595** | `agg_eval` telemetry | VERIFIED |
| F7 | Variance **floor (~0.45) > threshold (0.30)** → bins commit only on a *noise dip*, i.e. the noisiest estimates | charter M-3; `characterize_variance_curve.py` | VERIFIED |
| F8 | Global update is a **direct in-place SGD step** — `param.sub_(lr·Σ rate·gᵢ / N)` — with **no momentum, no EMA of global weights, no server optimizer state** across commits | `FedSgdAggregator.py:322-324`; §5 (aggregator trace) | VERIFIED |
| F9 | fedbuff "new" rate can exceed 1 (`beta_polynomial_upshift` adds +0.5) → some updates **amplified**, not damped | `fedbuff.py:100-101,139` | VERIFIED |
| F10 | Suspected round-boundary **staleness-reset bug is REFUTED**: `_model_version` is monotonic across the boundary, staleness stays ≥ 0. Only real discontinuity is the LR `ratio` warmup→decay step | `fwdllm_aggregator.py:1926-1929,1940-1946`; `FedSgdAggregator.py:234-242` | REFUTED |
| F11 | **REFUTED as the primary driver:** the per-`data_id` gradient the aggregator sees is **not** class-biased. Pooled over a realistic K=10 commit cohort at α=1 the class mix is near-balanced (dominant frac med **0.36**, entropy **0.95**, only **2.8%** of commits majority-one-class); pooled composition has **zero correlation** with which data_ids collapse (Spearman +0.002, p=0.98). A 0.36-dominant gradient cannot drive an exact single-class (25%/mcc=0) collapse. | H0 `diagnose_partition_binning.py` Layer C′ + overlay, `_diag_partition/` | REFUTED |
| F12 | Per-client class skew by α (100-client dist, exact): **α=0.1** dom-frac med **1.00** (½ of clients single-class) · **α=1** med **0.72**, 3 classes, entropy 0.50 · **α=100** med **0.30**, 4 classes, entropy 0.99 | H0 Layer A | VERIFIED |
| F13 | Collapses are **frozen-noise events, not class-structure events**: data (bins/cohort) is identical every epoch → the same high-variance JVP estimates recur at the same data_ids → position-locked collapse (F5). The *magnitude/variance* of the update, not its class direction, is the driver → points at F6-F9 (optimizer), not H1 (shuffle). | H0 (F11) + F6-F9 | VERIFIED |
| F14 | Partition is **quantity-balanced label-skew**: exactly **1200 samples/trainer at every α** (per-class totals 30k each) — α changes only the class *mix*, never the amount. Heterogeneity ladder (per-bin dom-frac mean / % bins ≥50%-one-class / #single-class trainers): α0.1 **0.94/96%/53** · α0.5 0.83/90%/0 · α1 **0.75/81%/1** · α5 0.57/46% · α10 0.52/32% · α100 **0.45/14%/0**. | H0 `--dist`, `trainer_class_heatmap.pdf` | VERIFIED |
| F15 | **Small-batch lumpiness is α-independent:** even at α=100 (near-IID clients) 14% of 8-sample bins are ≥50% one class — an artifact of the bin *size*, not heterogeneity. Within a trainer, bins are ≈ iid draws of that trainer's own class mix (the `executor.map` load already scrambles intra-trainer order — no class-sorted sequence). So **bin *composition/size* matters; intra-bin *order* is a no-op** (JVP/loss is a mean over the bin → permutation-invariant). | H0 `--dist`; F1 | VERIFIED |

### 8.2 Resolution plan — by scope

Two independent levers. Each item is tagged **cross-baseline hygiene** (applied identically to `fwdllm` /
`fwdllm++` / `fluxtune` — a fair-comparison correctness fix, **not** claimed as a fluxtune contribution) or a
**fluxtune-specific contribution** (claimed improvement over baselines). Cross-references to the charter's Opt
ladder are noted where they overlap.

**Cross-baseline hygiene (H).**

| ID | Fix | Why (finding) | Code — how | Sanity check to pass | Status |
|---|---|---|---|---|---|
| H0 | Diagnostic: per-client × per-`data_id` × per-K-cohort class distribution from `agnews_partition.h5` + frozen caches; overlay vs. observed collapses | Confirm/refute F1-F5 before any change | `expt_scripts/diagnose_partition_binning.py` (faithful cache order; α∈{0.1,1,100}) | ran; overlay Spearman +0.002 → **refuted** the data-bias story (F11) | **DONE** |
| H1 | ~~Shuffle each client's local train data before binning~~ **DE-PRIORITIZED** | F11: bins/cohort already near class-balanced at α=1 → little bias to remove. Only value is de-correlating F13 frozen noise across epochs (marginal) | `base_data_manager.py:490` `shuffle=True`+seed / per-round permute | would break F5 position-lock but not the collapse magnitude | PARKED (revisit if α=0.1) |
| H2 | **Larger data bins / higher agg_goal** (variance reduction) — ⚠ *not unambiguously good:* a large class-mixed bin averages conflicting per-sample gradients → the scalar JVP signal can sink below its noise floor and learning stalls. α-dependent optimum → sweep first (EXPERIMENTS.md **M1**). | F13,F6,F15: 8-sample bins × K=10 → high-variance, lumpy JVP is the driver, but too-large kills the signal | `train_batch_size` / bin regroup; `agg_goal` | M1 sweep finds the acc-maximizing bin size per α | DEFERRED |
| H3 | **Per-epoch, seeded, aggregator-orchestrated bin-order permutation** (visit bins in a fresh random order each round, once each) | F13,F15: visit order is irrelevant *in expectation* but the frozen 0→149 order replays the same noisy-estimate sequence every epoch → position-locked collapse. Randomizing de-correlates it across epochs. Intra-bin shuffle is NOT needed (F15). | aggregator drives `data_id` (`fwdllm_aggregator.py:1922`); broadcast a per-round seeded permutation — **must stay deterministic for real↔sim cohort_sequence parity** | F5 position-lock disappears across epochs; collapses stop recurring at fixed data_ids | TODO |

**Fluxtune-specific contributions (S — stability).** Claimed over baselines; these are where we take credit.

| ID | Contribution | Why (finding) | Code — how | Sanity check to pass | Status |
|---|---|---|---|---|---|
| S1 | **Server-side optimizer with momentum / EMA of the global model** (a damping / restoring force) | F8: undamped direct SGD → random walk on the loss surface | `FedSgdAggregator.py:322-324` add a server momentum buffer or global-weight EMA | loss envelope becomes monotone; peak is *sustained*, not transient | TODO |
| S2 | **Variance-gate recalibration** — commit on the *plateau*, not on a noise dip; align threshold to the achievable floor | F6,F7: gate commits the noisiest updates | `var_threshold` + plateau policy (extends charter **Opt-2**) | mean it-at-commit rises; early-commit collapse (F6) gone | TODO (Opt-2 partial) |
| S3 | **Aggregation-rate tempering** — cap rate ≤ 1; retune grad-aware to damp, not amplify | F9 + charter: R4 (full grad-aware) diverged *worst* | `fedbuff.py` beta upshift; grad_aware `align_floor`/`inverse_var` (retunes charter **Opt-3**) | R4 no longer the worst diverger; per-commit step magnitude bounded | TODO |

**Order of attack — pick by IMPACT, not table order.** Re-rank each turn to whatever best fixes the problem:
- **NEXT → S1** (server optimizer: descent, not random walk) — the root cause (EXPERIMENTS.md M2).
- then **S2** (don't commit on noise dips), **S3** (rate cap).
- **DEFERRED:** H2 (bin size — needs the M1 sweep), H3 (bin-order permutation). **PARKED:** H1.

### 8.3 Claim vs. correctness
- **Claimable (S1-S3):** fluxtune's async forward-grad aggregation is uniquely exposed to high-variance,
  amplifiable, undamped updates (F8,F9) → a *server optimizer* (S1) and *signal-aware commit gate* (S2) that
  stabilize forward-mode async FL are genuine contributions the sync baselines don't need.
- **Not claimable (H1-H3):** shuffle / bin-size / bin-order are correctness fixes any FL should have; applied to
  all three baselines so E1 stays fair (P1). Reported as fixed, not as wins.

### 8.4 Design Q&A
- **Bin vs. classical-FL round?** FedAvg updates from the *whole* local set (batch washed out before aggregation);
  FwdLLM commits per **8-sample bin** ⇒ bin/cohort size *is* the per-update variance. A knob that matters here,
  not in FedAvg (M1).
- **Shuffle *within* a bin?** No — JVP/loss is a mean over the bin ⇒ permutation-invariant (F15). No-op.
- **Why sequential bin order?** Order is irrelevant *in expectation*; the harm is the **frozen** 0→149 replaying
  the same noisy sequence every epoch → position-locked collapse (F13). Fix = per-epoch **seeded, aggregator-driven**
  permutation (H3), deterministic for sim parity. Reduces *repetition*, not *amplitude*.
- **Why does order matter if the model ignores sequence?** Only because the optimizer random-walks today (F8). At a
  real minimum order won't matter ⇒ fix the optimizer (S1), don't lean on order.

**S1 scoping (next task).** Flag `server_optimizer` (default off = raw SGD, byte-identical). Add momentum /
weight-EMA at `FedSgdAggregator.py:322-324` (`param.sub_(lr·Σg/N)`). Validate before A/B: (i) optimizer state
**deterministic** under frozen update order (sim parity); (ii) no **double-damp** with the fedbuff rate + variance
gate; (iii) A/B vs off at α=1 N=100 → sustained peak + W=20 window fills. Composable with C3/Opt-3, not a replacement.
