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

**Combined: sync −68%, fluxtune −37% GPU time, zero fidelity change.** This alone is expected to bring fluxtune's
per-batch compute under its modeled mobile-delay budget.

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
