# Trainer memory and speed, explained

Companion to [fl_fwd_ft_smollm2_smoke_plan.md](fl_fwd_ft_smollm2_smoke_plan.md) §0, which holds the optimisation list
(O1–O5), measured results and regression checks (RC1–RC10). This doc explains the *why* in plain terms.

## 1. What a trainer is training

Only LoRA (or adapters) and the classifier head train; the backbone is frozen.

| | DistilBERT (adapters rf=16) | SmolLM2-360M (LoRA r=8) | SmolLM2-1.7B (LoRA r=4) |
|---|---|---|---|
| Total parameters | 66,813,220 | 362,644,160 | 1,712,171,008 |
| Trainable | 450,340 (0.67%) | 823,040 (0.23%) | 794,624 (0.05%) |
| Frozen | 66,362,880 | 361,821,120 | 1,711,376,384 |
| – frozen Linear weights | 42,508,800 | 314,572,800 | — |
| – frozen embeddings + norms | 23,854,080 | 47,248,320 | — |
| Model size, all fp32 | 255 MB | 1,383 MB | ~6,530 MB |
| Model size, frozen Linear in bf16 (O2) | (O2 off) | 783 MB | 3,459 MB (measured) |

**Linear weights** are the big matrices in each transformer layer (output = input × W). SmolLM2-360M has seven per
layer (attention q/k/v/o, MLP gate/up/down), 9,830,400 parameters per layer × 32 layers = 314,572,800. LoRA attaches small
trainable matrices beside q and v; the big matrices stay frozen.

## 2. Where the memory went (before) and what each fix removed

FluxTune estimates the gradient from losses at θ + h·v and θ − h·v (10 probes per update, 2 forwards each). The code
built every per-parameter list over *all* parameters, frozen ones included.

| Copy, per trainer (SmolLM2-360M) | Before | After O1–O5 | Fix |
|---|---|---|---|
| The model (`self.model`) | 1,383 MB | 783 MB | O2 |
| Functional copy (functorch deep copy) | 1,383 MB | 0 | O3 |
| Grad buffer | 1,383 MB | 3 MB | O1 |
| Perturbation `v` (one probe at a time) | 1,383 MB | 3 MB | O1 |
| CPU copy of the grad after each update | 1,383 MB host | 3 MB | O1 |
| `full_grad` padding for the carried grad (`var_control`) | 1,383 MB host | 0 | O1 |

- **O1, zero-size frozen slots.** A frozen parameter's perturbation is always 0, so its grad is always 0. The slot now
  holds a stride-0 view: correct shape, one element of storage. Accumulation skips those slots (it was adding `jvp × 0`).
- **O2, bf16 frozen Linear weights.** bf16 is a 16-bit float: 2 bytes instead of 4, the same range as fp32, ~3
  significant digits. Every forward already runs under bf16 autocast, which converted these weights to bf16 on each
  matmul. Storing them in bf16 does that conversion once, so the matmul sees identical values. Embeddings and norms are
  not converted by autocast, so they stay fp32; LoRA and the head stay fp32 for the small updates. Llama only by default:
  DistilBERT's autocast is fp16, and its saving (~80 MB) isn't worth parity reruns.
- **O3, one weight copy.** `fmodel(params, buffers, x)` runs the model with whatever tensor list it is given. θ ± h·v is
  a list whose frozen entries *point at* existing tensors and whose ~0.2% trainable entries are new temporary tensors
  (~3 MB). functorch built that list from a deep copy of the model; `torch.func.functional_call` uses the model's own
  tensors. Nothing writes to them during an update, and `load_state_dict` updates them in place when new weights arrive.

## 3. Why an idle trainer held its peak memory

- **GPU (O4).** PyTorch's caching allocator keeps freed GPU memory for reuse, because getting memory from the CUDA driver
  (`cudaMalloc`) is slow and freeing it (`cudaFree`) stalls the GPU. The cache grows to the process's peak and stays
  there until `empty_cache()`, an OOM retry inside the same process, or exit. There is no timer, and a neighbour's OOM
  doesn't trigger it, so a trainer picked once every 10–20 iterations blocked its peak the whole time.
  `empty_cache()` after each update returns it; re-allocating next update costs milliseconds.
- **Host RAM (O5).** glibc likewise keeps small freed blocks. The model loads on the CPU (fp32) before moving to the GPU;
  360M's ~10 MB matrices stayed in that pool after the move. `malloc_trim(0)` returns free pages. 1.7B's 64 MB matrices
  go straight back to the OS, so it never had this problem.

Host RSS of a standalone stand-in trainer (360M), step by step:

| Step | RSS |
|---|---|
| `import torch` | 473 MB |
| CUDA context + first matmul | 674 MB |
| transformers, adapters, trainer code | 1,004 MB |
| model loaded in fp32 on CPU | 2,431 MB |
| `model.to(cuda)` | 2,251 MB (freed but kept) |
| `malloc_trim(0)` | 1,035 MB |
| first forward (kernel code) | 1,477 MB |

A real trainer also loads flame, the data manager and telemetry; it settles at ~2.0 GB with O5 (2.45 GB without).

## 4. GPU copies vs moves: what costs time

| Operation | Speed | 783 MB takes |
|---|---|---|
| Copy within the GPU | ~700 GB/s (A40) | ~2–3 ms |
| GPU → CPU over PCIe | ~25 GB/s | ~30 ms |
| `cudaMalloc` / `cudaFree` | bookkeeping + device sync | ms, plus a pipeline stall |

The functional copy cost little time but 783 MB of VRAM per trainer, and VRAM sets how many trainers fit. The old
per-update GPU → CPU grad copy was the slow one (1.38 GB each update); O1 is why `_finalize_training` fell from 0.28 s to
~0.02 s.

## 5. Memory caps N; compute contention depends on C

- Every trainer holds its footprint whether selected or not, so memory limits **N**.
- Trainer processes on one GPU take turns on its compute (no MPS), so only *active* trainers slow each other: an update
  that takes 6 s alone takes ~24 s with four active. That depends on **C**.
- Sim charges each update max(measured GPU time, modeled delay). While GPU time stays below the delay, contention is
  invisible to the vclock. §O's delays are calibrated to DistilBERT on a phone NPU (+6% buffer); bigger models need
  them scaled with forward cost (smoke plan §13), or packing starts setting the vclock (`[TIMING_OVERRUN]`).
- Delays never add wall time in sim (it doesn't sleep them); they only stretch the vclock, so raise `--max-runtime-s`
  with them.

## 6. How the "identical" checks were run

Two runs of the same config, old code vs new, on the same GPU, compared update by update.

```
# from expt_scripts/, env test_fwdllm; GPU 4 free
FWDLLM_STRICT_DETERMINISM=1 EXPT_GPU_ALLOW_PEER=1 ./run_sequential.sh --yes --only fluxtune --mode sim --allow-stale-profile \
  --dataset agnews --num-trainers 4 --gpu-ids 4 --agg-goal 4 --c 4 --min-initial-frac 1.0 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-360M --peft-method lora \
  --adapter-reduction-factor 16 --fd-scale-invariant \
  --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --commit-gate fixed --pool-target 8 --rho-schedule pool --agg-rate-type uniform \
  --var-stopping-policy off --max-iter-per-data-id 20 \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  --server-update-audit --no-cos-ground-truth-audit \
  --max-runtime-s 500 --sim-wall-ceiling-h 0.5
```

Compare, per run dir:
- **Updates:** every `[probe_combine=mean] … rms_jvp=X for trainer : T … data-id: D. iteration: I` line in
  `*trainers.log`, keyed by (T, model version, D, I). All must be equal.
- **Eval:** `agg_eval` events in `telemetry/aggregator_*.jsonl` (`test-accuracy`, `test-loss`).
- **Memory:** `[MEM:<tag>] GPU Allocated / GPU Reserved / CPU Memory` lines (logged at idle points; measure the in-update
  peak separately with `torch.cuda.max_memory_reserved`).
- **Time:** `step_timing` events in `telemetry/trainer_*.jsonl`, field `func` (`_train_one_batch`, `_finalize_training`,
  `_make_model_functional`), field `duration_s`.
- **Errors:** counts of `Traceback`, `out of memory`, `nan`, `[TIMING_OVERRUN]`.

Short runs prove identity, not learning: at 7 commits accuracy is still at chance. RC5 in the smoke plan extends this to
~150 commits and to the N = 50 config.
