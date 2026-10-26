# SmolLM2-360M + LoRA smoke test

2026-10-10. Ported, checked offline and smoked (results §6; simplified 6 h run §7–§8; DistilBERT parity run, N=50, §9;
trainer and aggregator performance optimisations and regression checks §0; further improvements §11; ~1B model §12;
1.7B port and first run §13; rank / probes / momentum smokes §14).
Why this model and the full-run estimate: session notes §9–§10 ([fl_fwd_ft_aishwwarya_session_2026-10-01.md](fl_fwd_ft_aishwwarya_session_2026-10-01.md)).
Long-form explanations (why each copy existed, bf16, caching allocator, host RAM): [fl_fwd_ft_trainer_memory_explained.md](fl_fwd_ft_trainer_memory_explained.md).

## 0. Trainer and aggregator performance optimisations (2026-10-13/14)

**Problem.** `self.params` held every parameter, frozen backbone included, so each trainer kept up to five full-size fp32
copies of the model (6.1 GB GPU, 6.6 GB host for 360M) although only 823k of 362.6M parameters train.

| # | Optimisation | Where | Scope | State |
|---|---|---|---|---|
| O1 | Frozen slots of `v_params`, `self.grad`, its CPU copy and `full_grad` are a stride-0 `frozen_zero(p)`; accumulation skips them | `fwdgrad_utils.py`, `tc_transformer_trainer_distribute.py`, `fwdllm_trainer.py` | all models | `8292a1b4` |
| O2 | Frozen `nn.Linear` weights stored in `amp_dtype()` (`store_frozen_linears_lowp`); `lowp_autocast` on warmup + `eval_model` | `fwdgrad_utils.py`, `trainer/main.py` | llama by default; `FWDLLM_FROZEN_LOWP=1\|0`; off without `fp16` or under `FWDLLM_JVP_FP32` | `00b8474d` |
| O3 | One weight copy: `make_functional_view` (`torch.func.functional_call` on `self.model`) replaces functorch's deep copy | same | all models | `2357c3d2` |
| O4 | `torch.cuda.empty_cache()` at the end of each update | `_finalize_training` | all models | `2357c3d2` |
| O5 | `trim_host_heap()` (glibc `malloc_trim`) after warmup and after each update | `fwdgrad_utils.py`, trainer | all models | `2357c3d2` |

Why each is exact: O1 drops only `jvp × 0` terms; O2 stores what autocast already casts to, and `adapters` LoRA runs the
frozen matmul apart from the delta; O3 feeds `fmodel` the same tensors (order, buffers and shifted forwards checked equal on
DistilBERT and SmolLM2); O4–O5 touch no values. O1–O5 are trainer-only (the aggregator's fp32 reference probes are
unchanged); the aggregator's own copies are A1–A4 below.

**Measured** (SmolLM2-360M, 4 trainers on GPU 4, `FWDLLM_STRICT_DETERMINISM=1`, 500 vclock-s, 7 commits; per trainer;
GPU excludes ~0.4 GB CUDA context):

| Run `experiments/run_20261013_…` | `005113` original | `010457` O1 | `011909` +O2 | `022559` +O3–O4 | `024423` +O5 |
|---|---|---|---|---|---|
| `rms_jvp` (60) and eval (3) vs original | — | identical | identical | identical | identical |
| GPU allocated / reserved | 2,951 / 6,084 MB | 2,951 / 4,425 | 1,714 / 2,609 | **876 / 912** | 876 / 912 |
| Host RAM | 6,562 MB | 2,526 | 2,467 | 2,447 | **2,006** |
| `_train_one_batch` median | 6.12 s | 5.33 | 4.67 | 4.83 | 4.80 |
| `_finalize_training` median | 0.28 s | — | — | 0.014 | 0.021 |
| Traceback / OOM / NaN / overrun | 0 | 0/0/0/1 | 0 | 0 | 0 |

Peak during one update with O1–O3 (offline): 900 MB reserved; the model is 797 MB, so idle ≈ peak. The single overrun
was a 4.3 s first `_make_model_functional`. `tests/mode/test_fwdllm_gpu_warmup.py` + `test_fwdllm_probe_report.py`: 51 pass.

**1.7B per trainer** (model 3,459 MiB = bf16 linears 3,072 + fp32 tied embedding 384 + LoRA/`score` 3):

| | O1–O2 (`run_20261013_014657`) | O1–O5 (`run_20261014_014720`, 25 trainers, 1 h) |
|---|---|---|
| GPU allocated / reserved, idle | 7,267 / 7,852 MiB | **3,642 / 3,863** (model + ~183 MiB not itemised) |
| Reserved peak (`after_fmodel_setup`; released by O4 each update) | — | 4,048 |
| nvidia-smi incl. context, idle / updating (100 ms sampling) | — | **4,004 / 4,184** |
| Per-update Δ allocated | — | +3.2 MiB (the 794,624-float grad), freed after upload |
| Host RSS (heap ~1.0 GB, libs ~0.6 GB) | 2,782 MB | **~1,975**, ±25 per update (O5), flat over 1 h |

Capacity at 4,184 MiB peak: 10 trainers per 46 GB GPU, 9 on the aggregator's GPU → **~79 on 8 GPUs** (O1–O2: ~39;
unoptimised: ~15). GPU memory binds, not host RAM (79 × 2 GB + helpers 14 GB + aggregator). Compute is already
92–100% at 3–4 trainers per GPU, so keep C ≈ 30.

### Aggregator optimisations (2026-10-14, `1de4bb79`)

**Problem.** O1–O5 never reached the aggregator. On 1.7B (`run_20261014_014720`) it held **108–110 GB host RSS**,
104 GB of it glibc `[heap]`, growing ~5 GB/h; trainers totalled 47 GB. Each gradient list carried full-size fp32 zeros
for all 315 parameter tensors (218 frozen, 6.5 GB a list) although trainers upload only the 97 trainable ones (3 MB).
The per-iteration retry snapshot deep-copied it (`_snapshot_retry_cache` 1.5–1.8 s), `cached_v` held up to 5 copies
before each commit, and a functorch deep copy of the model was rebuilt every iteration only for shapes.

| # | Optimisation | Where |
|---|---|---|
| A1 | Frozen slots of `self.grad` (hence `grad_pool`, `model_dict`, `cached_v`) are stride-0 zero views: `grad_slots(model)`; `aggregate_grad_pool` and `_apply_weighted_update` skip them (`is_frozen_slot`); the trust-ratio step and `_snapshot_last_round_update` keep them stride-0 | `fwdllm_aggregator.py`, `FedSgdAggregator.py` |
| A2 | Per-iteration `fc.make_functional_with_buffers(self.model)` removed; nothing read `self.fmodel/params/buffers` | same |
| A3 | `trim_host_heap()` at the end of each `aggregate()` | `FedSgdAggregator.py` |
| A4 | `[MEM:agg] rss=… MiB` logged per `aggregate()` | same |

Why exact: frozen slots are only ever zero (the merge writes trainable names only), so skipping them drops `+0` terms;
`deepcopy` keeps a stride-0 tensor at 4 bytes. Three test patches moved from `fc.make_functional_with_buffers` to
`grad_slots` (`test_fwdllm_rounds_termination.py`, `test_fwdllm_lap_boundary_identity.py`).

| Check | Result |
|---|---|
| `tests/mode` + `tests/telemetry` | 1,159 pass, 7 skip, 0 fail |
| Offline: real `_apply_weighted_update` / `aggregate_grad_pool` / `_snapshot_last_round_update`, full-zero vs stride-0 slots, raw_sgd and trust_ratio, momentum 0 and 0.9 | bit-identical params, audit norms, pools |
| Bit-identity vs `2357c3d2` (worktree), DistilBERT fluxtune n4, `--agg-goal 4 --pool-target 8`, `FWDLLM_STRICT_DETERMINISM=1`, 14 commits (`cached_v` 0 → 1 every commit) | 120/120 `rms_jvp`, 14/14 `[ServerStep]` and `server_update`, 6/6 `agg_eval` identical; 0 errors (`run_20261014_030737` vs worktree `run_20261014_030518`) |
| `_snapshot_retry_cache` median (DistilBERT) | 22 → 6 ms |
| Launcher dry runs (1.7B n25, DistilBERT n4) | preflight 9/9 and 10/10; 1.7B yaml unchanged |

**Expected on 1.7B (not yet measured):** aggregator RSS ~110 GB → under ~15 GB and flat; ~1.8 s less per iteration.

**Behaviour changes.** `eval_model` outside the training loop runs in `amp_dtype()` when weights are low precision. With
`jvp_eval_mode` off, the JVP now follows the live model's mode, and an in-loop `eval_model` leaves it in eval.

### Regression checks (next steps, in order)

Baseline = pre-optimisation code `a44e0896` in a git worktree (data and token caches linked), run back-to-back with the
new code on the same GPUs. "Identical" = every `rms_jvp`, `[ServerUpdate]` audit line and `agg_eval` equal under
`FWDLLM_STRICT_DETERMINISM=1`.

| ID | Check | Exit criterion | State |
|---|---|---|---|
| RC1 | Launcher dry runs: fluxtune DistilBERT, fwdllm, fedbuff_round, felix_round, SmolLM2-360M, 1.7B | preflight passes; rendered yaml equal to baseline's | todo |
| RC2 | Full `tests/mode` + `tests/telemetry` | same failures as baseline (5 known) | done: 0 fail with A1–A4 (1,159 pass) |
| RC3 | Bit-identity, DistilBERT fluxtune n4, ~5 min | identical; DistilBERT is O1/O3/O4/O5 only | todo |
| RC4 | Bit-identity, paths the n4 runs skip: fwdllm (winner-only probe, `var_control` + odd-data_id `old_grad`), perf-opt off, `jvp_eval_mode` off, eval during training, retention + cos audits, one real-mode leg | identical, 0 errors | todo |
| RC5 | **Accuracy per commit**: 360M n4 identity run to ~150 commits (~1.5 h); then the §9 config (N=50) for ~200 commits | n4: identical at every commit and acc clearly above chance; N=50: acc at commits 50/100/150/200 within ±0.02 of `run_20261012_032648` | todo |
| RC6 | **Iteration time**: from RC5's paired runs, `_train_one_batch`, `_finalize_training`, `_make_model_functional` median and p95, and aggregator wall per iteration | new ≤ baseline + 5% (median and p95); 0 `[TIMING_OVERRUN]` | todo |
| RC7 | **Commit time**: wall and vclock per commit at matched commits | wall ≤ baseline + 5% (§9: ~34 s/commit at N=50); vclock per commit equal in the identity runs | todo |
| RC8 | Memory stability over RC5's N=50 run: per-trainer allocated, reserved, host RSS every eval | flat (no growth); per-trainer GPU ≈ 1.3 GB incl. context | todo |
| RC9 | DistilBERT parity vs floors (O1/O3–O5 are shared code, R10) | `run_parity.py` passes | todo |
| RC10 | 1.7B smoke (§13 step 6) after RC1–RC9 | §13 pass condition; per-trainer GPU ≈ 4 GB | partly: 4.0 / 4.2 GB measured (n25, §13) |
| RC11 | Aggregator A1–A4 bit-identity vs `2357c3d2` | identical, 0 errors | done (DistilBERT n4, 14 commits) |
| RC12 | Aggregator RAM on 1.7B: `[MEM:agg]` in the N = 50 run (§13) | under ~15 GB and flat vs 108–110 GB before (n25, old aggregator) | wip |
| RC13 | A1–A4 on the `fwdllm` baseline path (`var_control`, odd-data_id `old_grad`; RC4's legs) | identical, 0 errors | todo |

**Goal:** show the model ports cleanly and read the controller's numbers (`p`, trips/commit, commits/h, `cos·Φ`).
Accuracy is out of scope: the roberta smoke `145932` sat at chance and still counted as a clean port. The porting
ladder is buildplan §5.8.

**Model:** `HuggingFaceTB/SmolLM2-360M` (Llama: hidden 960, 32 layers, GQA 15/5 heads) + LoRA r=8, α=16 on q,v.
Run it as `model_type=llama`, `peft_method=lora`.

## 1. Code changes

| File | Change |
|---|---|
| `expts/initializer.py` | `"llama"` entry; pad = cls = sep = EOS (the converter adds `[cls] text [sep]`, and SmolLM2 has none of the three); sets `FWDLLM_AMP_DTYPE=bf16` |
| `trainer/.../fwdgrad_utils.py` | `amp_dtype()`; all 3 JVP autocasts use it (default fp16, unchanged for other models) |
| `tc_transformer_trainer_distribute.py` | training-loop autocast uses `amp_dtype()`; `layer_id_for_check=16` for llama (a `lora_B`: `lora_A` grads are exactly 0 at init, so var would be 0) |
| `fwdllm_aggregator.py` | eval autocast honours `FWDLLM_AMP_DTYPE` |
| `expts/dataset_registry.py` | measured llama row (`p` 823,040 on agnews); `score` has no bias |
| `run_sequential.sh`, `run_node_p4.sh` | `--peft-method` / `P4_PEFT_METHOD` into both blocks; `P4_FD_SCALE_INVARIANT` override |
| `probe_port_init.py` | `--peft-method`; imports this checkout, not `/home/dgarg39/flame` |

`pre_classifier` needed no guard: its drop is gated on `model_type == "distilbert"`.

## 2. Offline checks (measured)

| Check | Result |
|---|---|
| `p` | **823,040** = LoRA 819,200 + `score` 3,840. Only `score` is trainable outside LoRA |
| Weights, pooling | load fp32; pooling = rightmost non-pad token (transformers 4.57.6), i.e. the last text token |
| Chord ON / OFF | **0.511** / 0.691. Keep the flag ON (default). LoRA's ‖θ_tr‖/√p is 0.0145, not 0.0197, because `lora_B` = 0 |
| fp16 autocast | **NaN loss, even unperturbed**: fp16 can't hold Llama's activations. Fixed with bf16 |
| FD JVP vs autograd (4 probes) | fp32 and bf16 both within ~2–20%, about equal; that is h truncation, not precision. bf16 is 63 ms/fwd, fp32 154 ms |
| Grad split at init | `lora_A` 0, `lora_B` 0.27, `score` 24.7. Early steps go almost entirely to the head |
| Launcher dry run | `model_type`, `model_name` and `peft_method` reach both the trainer and aggregator blocks |

## 3. How to run

Steps 1–3 are CPU-only and take a few minutes; step 4 uses the GPUs.

**0. Environment** (from the repo root)

```
conda activate test_fwdllm
cd lib/python/examples/fwdllm/expt_scripts
```

**1. Port probe** (~1 min). Builds the model exactly as a real run does.

```
python probe_port_init.py --model-type llama --model-name HuggingFaceTB/SmolLM2-360M --peft-method lora
```

Expect `p` 823,040, chord ON 0.511 and chord OFF 0.691. A `p` in the millions means the backbone isn't frozen.

**2. Pre-tokenize** (<1 min for 10 clients; already done once on this node). `--clients` must be ≥ the trainer
count; the cache is keyed by model.

```
python pretokenize_dataset.py --dataset agnews --clients 10 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-360M
```

**3. Dry run** (optional; launches nothing). The rendered yaml in `smoke_logs/<ts>/` should show
`model_type: llama` and `peft_method: lora` in both blocks.

```
./run_sequential.sh --dry-run --only fluxtune --mode sim --allow-stale-profile --dataset agnews \
  --num-trainers 10 --num-gpus 8 --agg-goal 10 --c 30 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-360M --peft-method lora \
  --probe-combine mean --commit-gate n_target --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --max-iter-per-data-id 20 --max-runtime-s 2500 --sim-wall-ceiling-h 2.5
```

**4. Launch**, detached (background shells are killed after 2 h).

```
cd nodes
SMOKE=1 CEIL_OVERRIDE=2.5 EXPT_GPU_ALLOW_PEER=1 P4_RETENTION_EVERY=10 \
  P4_MODEL_TYPE=llama P4_MODEL_NAME=HuggingFaceTB/SmolLM2-360M P4_PEFT_METHOD=lora \
  P4_NUM_TRAINERS=10 \
  setsid nohup ./run_node_p4.sh agnews controller > ../../smollm2_smoke_$(date +%Y%m%d_%H%M).log 2>&1 &
```

| Knob | Why |
|---|---|
| `SMOKE=1` | vclock 2,500: the production code path, cut short. The run ends on `max_runtime_s`, not `[BudgetStop]`, as expected |
| `CEIL_OVERRIDE=2.5` | the preflight projects law C's 898 commits × 5.87 trips = 8,021 s, over SMOKE's 2 h ceiling. Without this it blocks |
| `EXPT_GPU_ALLOW_PEER=1` | the GPU check reads the busiest GPU on the node, and other jobs share these GPUs |
| `P4_RETENTION_EVERY=10` | turns on `[Retention]`, the source of `cos·Φ`. Off by default |
| `P4_NUM_TRAINERS=10` | memory. Reads only shards 0–9 of 100: valid for a smoke, not for a scored run |
| *(not set)* `P4_FD_SCALE_INVARIANT` | the default 1 is right for this model (chord 0.511) |
| *(not set)* `FWDLLM_AMP_DTYPE` | `create_model` sets bf16 for llama. Setting `fp16` brings back the NaN loss |
| *(not set)* lr | `trust_ratio` doesn't read it, so no retune |

Expect more than 15 min of real wall: the sim profile is DistilBERT's, so each vclock-s costs more real time.

**5. Read the results.** The run directory is `experiments/run_<ts>_fluxtune_agnews_n10_smoke_syn_0_sim/`.

```
R=$(ls -td ../../experiments/run_*fluxtune_agnews_n10_smoke* | head -1)
grep -h '\[ProbeDim\]' $R/*trainers.log | head -1          # p=823040
grep -h '\[FD\] spacing' $R/*trainers.log | head -1        # h≈0.0074, scale_invariant=on
grep -h '\[Retention\]' $R/*aggregator.log | tail -3       # cos*Phi≈1.0000
grep -ich 'nan' $R/*.log                                   # 0
python ../check_arm_health.py $R                           # trips/commit per quintile, rho_star != 0
```

| Reading | Pass condition |
|---|---|
| `[ProbeDim] p` | 823,040 |
| `cos·Φ` | ≈ 1.0000 (roberta's "ported clean" signal) |
| Trips/commit | 10–20 |
| Commits/h | from `server_update` commit count over wall time; projected full run ≈ 400 ÷ commits/h |
| Deaths / OOMs / NaNs | 0 |

## 4. Decide

- `p` and `cos·Φ` clean and ≤ ~10 h projected (session §10 estimates ~5–10 h): launch the full run to its own stop.
  Φ* at a new `p` (N5c) is the open question, so don't cut it early.
- Chord drifting: flip `P4_FD_SCALE_INVARIANT` and rerun the smoke.
- Slower than expected: re-profile `sim_charge_profiles`, or sim timing won't match real.

## 5. If something breaks

- Loss `nan`: check that `FWDLLM_AMP_DTYPE` isn't set to fp16 in the shell.
- Dtype mismatch in a forward, or to rule out O2's bf16 frozen weights (§0): rerun with `FWDLLM_FROZEN_LOWP=0`.
- Trainer `KeyError`/`AttributeError` on `layer_id_for_check`, or preflight `KeyError: no adapter param counts`:
  `model_type` isn't `llama`.
- OOM: lower `P4_NUM_TRAINERS`. With §0's O1–O5 a 360M trainer holds ~1.3 GB of GPU (1.7B 4.2 GB peak, measured).

## 6. Smoke result (2026-10-10)

Run `experiments/run_20261010_225445_fluxtune_agnews_n10_smoke_syn_0_sim/` (full controller, 10 trainers,
`P4_GPU_IDS=0,1`). It ended on its own vclock budget after **716 s real** (2,500 vclock-s at a sim rate of about 3.9×):
65 iterations, 3 commits. The node's "committed 3 times -- systemic fault" line is the `<5 commits` guard firing on a
short run, not a failure.

| Check | Result |
|---|---|
| `p`, FD | 823,040; `scale_invariant=on`, h 0.0074 ✓ |
| NaN / OOM / crash | 0 ✓ |
| Trips/commit | 16–16.7 ✓ (`n_req` 169) |
| `cos·Φ` | not read: no `[Retention]` line in 3 commits |
| Accuracy | 0.23 → 0.26 (chance; out of scope) |

**Timing, against the DistilBERT simplified run `20261002_000757` (n100; same batch 8, seq 192, P=10):**

| | SmolLM2 smoke | DistilBERT |
|---|---|---|
| Wall / iteration (median) | 9.2 s | 1.1 s |
| vclock / iteration | 38.0 s | 6.5 s |
| Trainer GPU / update (median) | 7.8 s | 0.6 s |
| Aggregation step / iteration | 1.1 s (agg 0.36 s) | — |
| Iterations / commit | ~17 (`n_target` gate) | 5 (`fixed` gate, N=50) |
| Wall between commits | ~155 s | — |
| Model apply / eval | 2.2 s / 45 s (eval on a side thread) | — |

**Why it's slower; neither cause is the model:**
- **Wall:** an update is 20 forwards (10 probes × central difference). At 63 ms/fwd that is ~1.3 s on a free GPU.
  The measured 7.8 s came from packing: `P4_GPU_IDS=0,1` put 5 trainers on each of 2 GPUs that peer jobs also used.
  GPU time scales with trainers per GPU (n4 run, 4 on one GPU: 3.8 s). Only 3 updates overran the 8.33 s minimum
  delay (`[TIMING_OVERRUN]`), but a bit more contention would let GPU noise into the vclock.
- **vclock:** with N = K = 10, every iteration waits for all 10 trainers, so it costs the slowest modeled delay
  (37.5 s). DistilBERT's n100 run waits for the fastest 10 of 30 in flight. Smoke vclock numbers don't project a
  full run.

**How runs are timed:**
- **Sim mode never sleeps the training delay.** It is charged to the vclock as max(gpu, delay). Wall per iteration
  = GPU + aggregation. Larger delays end a vclock-budgeted run *sooner* in wall time.
- `--delays off` (`run_sequential.sh` only; `run_node_p4.sh` has no knob) makes the vclock equal GPU time. Update
  order then follows GPU noise and results aren't comparable to delay-on runs. Use it for timing only.
- Real mode does sleep the delay remainder.

**`run_node_p4.sh` vs `run_sequential.sh`:** the same experiment code. The node script wraps `run_sequential.sh` with
preset flags (`n_target` gate, rf=16, `FWDLLM_FD_SCALE_INVARIANT=1`, SMOKE/production budgets), a `watch_arm.py`
watchdog (20 min hang, trainer deaths, ceiling + 1 h), and end checks (run dir, enactment lines, `<5 commits` abort,
`check_arm_health.py`). Its flags are fixed to the full controller, so the simplified arm must use `run_sequential.sh`.

## 7. Simplified controller, 6 h wall

The agnews-validated simplified arm (session notes §5–§6): `fixed` gate at N=50, constant ρ = s·√(P·N/p) ≈ **0.037**
at this `p` (DistilBERT 0.050), ω = 1, no B_max probe, no budget stop, no GL decay; it stops on Φ ≥ 3 or the stall
rule. The dry run passes preflight 9/9.

- **6 h wall:** `--sim-wall-ceiling-h 6`.
- **vclock never ends it first:** `--max-runtime-s 200000` (production agnews is 48,000 ≈ 3.5 h at the smoke's pace).
- `P4_GPU_IDS` unset: `--num-gpus 8` spreads the trainers.
- `--adapter-reduction-factor 16 --fd-scale-invariant` copy what the node script sets; without them preflight blocks.

```
conda activate test_fwdllm
cd lib/python/examples/fwdllm/expt_scripts
EXPT_GPU_ALLOW_PEER=1 setsid nohup ./run_sequential.sh --yes --clean --only fluxtune --mode sim --allow-stale-profile \
  --dataset agnews --num-trainers 10 --num-gpus 8 --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-360M --peft-method lora \
  --adapter-reduction-factor 16 --fd-scale-invariant \
  --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --commit-gate fixed --pool-target 50 --rho-schedule pool --agg-rate-type uniform \
  --var-stopping-policy off --max-iter-per-data-id 20 \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  --saturation-stop --no-sat-decay --sat-stall-frac 0.003 \
  --server-update-audit --no-cos-ground-truth-audit --retention-probe-every 10 \
  --max-runtime-s 200000 --sim-wall-ceiling-h 6 \
  > ../smollm2_simple_$(date +%Y%m%d_%H%M).log 2>&1 &
sleep 3; PGID=$(ps -o pgid= -p "$(pgrep -of 'run_sequential.sh --yes')" | tr -d ' '); echo "PGID=$PGID"  # not $!: setsid forks
# watchdog (the node script's side-car): hang / trainer deaths / ceiling + 1 h
nohup python watch_arm.py --exp-dir ../experiments --since $(date +%s) --pgid $PGID --kill --max-hours 7 \
  > ../smollm2_simple_watch.log 2>&1 &
```

**Stop mid-run:** `kill -TERM -- -$PGID`, then `kill -KILL -- -$PGID` after ~20 s if anything is still alive.
The launcher leads its own group under `setsid`, so the trainers and the aggregator go too. Logs up to the kill stay in the run dir.
Afterwards: §3 step 5's greps and `python check_arm_health.py <run_dir>`.

**Before launching:** check `nvidia-smi` (peer jobs held GPUs at 75–100% on 2026-10-11). N = 50 chosen (§9), keeping the
DistilBERT A/B setting; the full controller asked for ~169 at this `p` (ρ ≈ 0.068, ~17 iterations/commit).

## 8. Simplified run on 2 GPUs (launched 2026-10-11 22:34)

The §7 command with one change: `--num-gpus 8` → `--gpu-ids 0,1` (5 trainers per GPU, the same packing as the §6 smoke).
`--gpu-ids` sets `execution.gpu_ids` for the trainers and the aggregator, and `num_gpus = 2`. Dry run: preflight 9/9. N = 50.

```
EXPT_GPU_ALLOW_PEER=1 setsid nohup ./run_sequential.sh --yes --clean --only fluxtune --mode sim --allow-stale-profile \
  --dataset agnews --num-trainers 10 --gpu-ids 0,1 --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  ...   # rest identical to §7
  > ../smollm2_simple_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
```

**N, C, K:** all 10 in effect. `--c 30` is capped at the trainer count (`min(num_ends, effective_c)` in
`flame/selector/async_oort.py`; the log reads `chosen concurrency: 10`), so every iteration waits for all 10 trainers.
`--pool-target 50` is the commit gate's N, not a trainer count.

**Delays:** sim mode charges them to the vclock (max(gpu, delay)); they add no wall time. Watch `[TIMING_OVERRUN]` in
the trainers log: at 5 trainers/GPU, GPU time can exceed the modeled delay and reorder updates.

**Watchdog:** §7's `PGID=$(... pgrep -of ...)` picked the wrong process (533051; the launcher's group was 533061) and
the side-car exited with code 2. Restarted by hand with absolute paths:

```
D=$PWD/..   # fwdllm/
setsid nohup python $D/expt_scripts/watch_arm.py --exp-dir $D/experiments --since <launch epoch - 120> \
  --pgid <pgid of run_sequential.sh, from: ps -o pid,pgid,cmd -C bash | grep run_sequential> --kill --max-hours 7 \
  > $D/smollm2_simple_watch.log 2>&1 < /dev/null &
```

| File | What |
|---|---|
| `smollm2_simple_20261011_2233.log` | launcher stdout |
| `smollm2_simple_watch.log` | watchdog (commits, trips/commit, dead trainers) |
| `experiments/run_20261011_223407_fluxtune_agnews_n10_smoke_syn_0_sim/` | run dir |
| `.../telemetry/aggregator_fluxtune_agnews_n10_smoke_syn_0_sim.jsonl` | **accuracy**: `agg_eval` events, field `test-accuracy` |
| `.../*_aggregator.log` | only `Eval Loss` per eval; no accuracy line |

**Read accuracy** (from `fwdllm/`):

```
R=experiments/run_20261011_223407_fluxtune_agnews_n10_smoke_syn_0_sim
grep '"agg_eval"' $R/telemetry/aggregator_*.jsonl | python -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    print(d['round'], d['data_id'], d['iteration_per_data_id'], round(d['test-accuracy'], 4), round(d['test-loss'], 4), round(d['top_class_share'], 3))"
```

First eval (commit 1, data_id 0, iteration 5): acc 0.249, loss 1.99, top_class_share 0.93 (chance). Eval takes ~43 s
wall on a side thread.

**Result (stopped by hand 2026-10-12 ~02:50, 4.37 h wall, 351 commits, round 3 data_id 48):**

| Wall | vclock | Point | Test acc |
|---|---|---|---|
| 45 min | 3.2 h | R1 d58 | 0.52 |
| 112 min | 7.9 h | R1 d148 (lap 1 ends) | 0.61 |
| 180 min | 12.7 h | R2 d88 | 0.70 (first) |
| 4.34 h | — | R3 d46 | **0.751** (best) |

- Not a scored run: N = C = K = 10 and only shards 0–9 of `clients=100` (10% of the data). Superseded by §9.
- Availability: `syn_0`, all 10 `AVL_TRAIN` on every selection. Lap = 150 data_ids ≈ 112 min wall.
- Each trainer held 6.1 GB, flat for 4 h; both GPUs at 100%. The aggregator landed on GPU 2 despite `--gpu-ids 0,1`.
- `lib/smollm2_simple_watch.log` is the failed first watchdog (`watch_arm.py` not found); delete it.

## 9. Parity run vs DistilBERT simplified, N=50 (launched 2026-10-12 03:26)

**Reference:** DistilBERT simplified arm `run_20261002_000757` (session notes §3, §5–§6). Its run dir and launcher
logs are gone from every checkout and scratch; the setup below is rebuilt from those notes.

| | DistilBERT simplified | This run |
|---|---|---|
| Model | DistilBERT + adapters rf=16, fp16, `p` 450,340 | SmolLM2-360M + LoRA r=8, bf16, `p` 823,040 (unavoidable) |
| Data | agnews `niid_label_clients=100_alpha=1` | agnews **`niid_label_clients=50_alpha=1`** (full data) |
| N / C / K | 100 / 30 / 10 | **50** / 30 / 10 (100 × 6.1 GB > 8 × 46 GB) |
| Gate | `fixed` N=50, `pool` ρ 0.050 | `fixed` N=50, `pool` ρ ≈ 0.037 (same N, ρ follows `p`) |
| P, combine, s, ω, step | 10, `mean`, 1.5, uniform, trust-ratio | same |
| Off | var-stop, B_max probe, budget stop, GL, cos/retention audits | same |
| Stop | Φ rail 3, stall 0.003, 3 h wall | same rules, **6 h wall** |
| GPUs | 0, 2, 4, 5, 6 | all 8 (7, 7, 6, 6, 6, 6, 6, 6 trainers) |
| Result | peak 0.8726, Φ rail at commit 881, 1.74 h | — |

**Partition `niid_label_clients=50_alpha=1`:** built 2026-08-24 by `build_niid_partitions.py` v1.0 (seed 42,
`order: shuffled_within_client`; group attrs). `check_partitions.py`: ok, 50 × 2,400 train / 152 test, full coverage.

| | `clients=50` | `clients=100` (DistilBERT) |
|---|---|---|
| Dominant-class share (Dir exp 0.522) | 0.508 | 0.717 |
| Single-class bins (batch 8) | 1.7% | 19.7% |
| \|class mean position − 0.5\|, median | 0.008 | 0.008 |
| First vs last 10% class-mix TV, median | 0.05 | 0.04 |

Both are shuffled within client (no class blocks). The 50-client group is milder non-IID, and a lap is 300 data_ids,
twice DistilBERT's, so the lap-boundary collapse (FT-L8) arrives later.

**Commands** (env `test_fwdllm`; from `expt_scripts/`):

```
python pretokenize_dataset.py --dataset agnews --clients 50 --jobs 16 --model-type llama \
  --model-name HuggingFaceTB/SmolLM2-360M --partition-method "niid_label_clients=50_alpha=1"   # cache is keyed by partition
EXPT_GPU_ALLOW_PEER=1 setsid nohup ./run_sequential.sh --yes --clean --only fluxtune --mode sim --allow-stale-profile \
  --dataset agnews --partition-method "niid_label_clients=50_alpha=1" --num-trainers 50 --num-gpus 8 \
  --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-360M --peft-method lora \
  --adapter-reduction-factor 16 --fd-scale-invariant \
  --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --commit-gate fixed --pool-target 50 --rho-schedule pool --agg-rate-type uniform \
  --var-stopping-policy off --max-iter-per-data-id 20 \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  --saturation-stop --no-sat-decay --sat-stall-frac 0.003 \
  --server-update-audit --no-cos-ground-truth-audit \
  --max-runtime-s 200000 --sim-wall-ceiling-h 6 \
  > ../smollm2_parity_n50p50_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
```

Watchdog: §8's form with `--pgid 692306`, log `smollm2_parity_n50p50_watch.log`. Stop: `kill -TERM -- -692306`.
`run_20261012_031719_*_n50_*` is void: the first launch read shards 0–49 of `clients=100`, interrupted at 5 min.

**Checks before and after launch:**
- Dry run `smoke_logs/20261012_032618`: preflight 9/9; rendered yaml differs from §8's only in N, partition (both
  roles), GPUs, retention off, `minInitialTrainers` 45. `client_idx_modulo: 100` is inert at 50 trainers (indices 0–49).
- Live: 50 trainers, `effective_c: 30`, `p=823040`, FD `scale_invariant=on` h 0.0074, `pool_size` 50 and 5 iterations
  per commit, 0 OOM / traceback / NaN / `[TIMING_OVERRUN]`. GPUs 0–1 at 43.0 / 46 GB (watch for OOM).
- Participation after 19 commits: all 50 contributed, 2–46 uploads each (fast trainers re-picked first; by design).

**Pace and stop:** ~34 s wall per commit, eval every 2 commits; vclock now runs slower than wall (first 10 of 30 per
iteration). Lap 1 ≈ 2.8 h. Φ = 3 at ρ 0.037 needs ~1,600 commits (~15 h), so the 6 h ceiling (~630 commits) ends the
run: compare at matched commit / Φ / wall, not stop-to-stop. A ~16 h ceiling would let it reach its own stop.

| Run dir | Logs | Plot |
|---|---|---|
| `experiments/run_20261012_032648_fluxtune_agnews_n50_smoke_syn_0_sim/` | `smollm2_parity_n50p50_20261012_0326.log`, `smollm2_parity_n50p50_watch.log` | `smollm2_acc_vs_wall.png` (accuracy vs wall, §8 overlaid) |

**Result:** ended on the 6 h wall ceiling at 651 commits (Φ stop not reached). Best acc **0.837**, last 0.836, over
325 evals; DistilBERT's simplified arm peaked at 0.8726. 0 dead trainers; trips/commit held at 5.0.

## 11. Improvements not yet made

| Improvement | Gain | Cost / risk |
|---|---|---|
| Staggered trainer start + `low_cpu_mem_usage=True` | avoids the startup host-RAM spike (each trainer loads fp32 on CPU first) | launcher change |
| Recalibrate `training_delay_factor` per model (FWDLLM_DESIGN §O method × forward cost vs DistilBERT) | realistic device model; GPU time stays under modeled delay, so contention never reaches the vclock | config; per model |
| Batch the 20 probe forwards (10 probes × ±h) into 1–2 calls | big for 360M (63 ms/fwd vs ~7 ms of arithmetic), ~1.5–2× at 1.7B | `calculate_jvp` + batched LoRA forward |
| CUDA MPS on the node | kernels from co-located trainers overlap | no code; node-wide daemon, shared with other users |
| Charge profiled solo GPU time in sim, not measured | packing can never distort the vclock | sim-mode design change |
| One process per GPU sharing a frozen backbone, swapping per-client LoRA (~3 MB) | clients no longer memory-bound | restructure trainer launch/channels; needed for 7B |
| 4-bit backbone (bitsandbytes) | ~2× less weight memory | slower forwards (dequant); `functorch` + `adapters` compatibility unknown. Hold for 7B |

**Contention.** Active trainers on one GPU time-slice it, so update GPU time grows with them; once it exceeds the modeled
delay (sim charges max), packing sets the vclock. §O's delays fit DistilBERT only; resize them per model (§13).

## 12. Handling a ~1B model

**Model.** SmolLM2 has no 1B (135M, 360M, 1.7B). Estimates below are extrapolated from 360M; none are measured.

| | SmolLM2-1.7B | TinyLlama-1.1B | Llama-3.2-1B |
|---|---|---|---|
| Architecture | Llama, 24 layers, hidden 2048, MHA 32/32 | Llama, 22 layers, GQA 32/4 | Llama, 16 layers, GQA 32/8 |
| Tokenizer / data | same as 360M / 11T | Llama-2 32k / 3T (weak) | 128k / gated licence |
| `p`, LoRA q,v | r=8 ~1.58M; **r=4 ~0.79M** | r=8 ~1.13M | r=8 ~0.86M |
| Forward cost vs 360M | **2.4× measured** (§13) | ~3× est. | ~3.4× est. |

**Pick: SmolLM2-1.7B, LoRA r=4.** Same family and tokenizer as 360M with `p` ≈ 823k, so only model size changes: the
cleanest test that commits scale with `p`, not size (N5c). TinyLlama r=8 if the paper must say "1B" or memory is short.
All three run on the existing `model_type=llama` path (pad = cls = sep = EOS, bf16). Rerun `probe_port_init.py`: `p`,
chord, and that `layer_id_for_check=16` still lands on a `lora_B`.

**Memory per trainer (1.7B).** ~22 GB unoptimised; 7.9 GB reserved with O1–O2; **4.2 GB peak with O1–O5** (§0), so
~10 per A40 and ~79 on the node. Host RAM ~2.0 GB per trainer, so 50 trainers ≈ 100 GB; the aggregator needs A1–A3
(§0) or it alone holds ~110 GB. Stagger launches: 50 simultaneous fp32 loads would need ~340 GB of host RAM.

**C and delays.** Memory caps N; compute contention depends on C (only active trainers share a GPU's compute). With
§13's recalibrated delays (fastest ~208 s vs a ~1.9 s solo update), even all trainers on a GPU active stay under the
delay, so any C ≤ N is overrun-free. Keep C = 30 for parity with DistilBERT.

**Wall time.** Delays add no wall time in sim. ~80 s per commit (360M: 34 s × 2.4× measured forward cost) and ~1,600
commits to Φ = 3 → **~35 h**; ~14 h to §9's 630-commit point. Probe batching: maybe ~20 h.

**Plan.**

| # | Step | Done when |
|---|---|---|
| 1 | Trainer optimisations O1–O5, aggregator A1–A4 (§0) | ✅ identity (O on 360M, A on DistilBERT); RC1, RC3–RC9, RC12–RC13 owed |
| 2 | Staggered trainer start + `low_cpu_mem_usage` | 50 trainers start with host RAM well under 266 GB |
| 3 | Port SmolLM2-1.7B, `lora_r=4`; pretokenize `clients=50` | ✅ port (§13); pretokenize owed |
| 4 | One trainer on a free GPU | ✅ forward-only and full updates (§13 n25 run: 4.2 GB peak, 2–10 s per update packed 3–4 per GPU) |
| 5 | Recalibrate `training_delay_factor`; raise `--max-runtime-s` and ceiling | ✅ values in §13; derivation next to FWDLLM_DESIGN §O owed |
| 6 | ~1 h smoke, N = 50, C = 30 | 0 OOM / NaN / dead trainers / `[TIMING_OVERRUN]`; commits/h → projection |
| 7 | Probe batching | only if step 6 projects > ~3 days |
| 8 | Full run to its own stop, ceiling ~80 h, watchdog | Φ rail or stall; compare with 360M and DistilBERT at matched commit / Φ (N5c) |

## 13. SmolLM2-1.7B port (2026-10-13)

`HuggingFaceTB/SmolLM2-1.7B` + LoRA r=4 on q,v, `model_type=llama`, no trainer-code change. Changes:
`LLAMA_BY_NAME` in `expts/dataset_registry.py` (hidden 2048; `probe_dim(..., model_name)`; 360M stays the default),
`run_sequential.sh` passes `model_name` to the preflight, `expt_scripts/test_llama_probe_dim.py`.

| Check | Result |
|---|---|
| `p` (`probe_port_init.py`, r=4 / r=8) | **794,624** / 1,581,056 = registry |
| `‖θ_tr‖/√p`; chord ON / OFF (r=4) | 0.0092; **0.819** / 1.088 (360M: 0.511 / 0.691). Flag stays ON; first steps are ~1.6× longer relative to θ |
| `layer_id_for_check=16` | `layers.1.q_proj` `lora_B` ✓ (315 params, 97 trainable) |
| Forward, bf16, batch 8 × seq 192, GPU 4, solo | 360M 38.9 ms · **1.7B 92.5 ms (2.4×)**; 20-forward update 0.78 s vs 1.85 s |
| Peak GPU alloc, forward only | 851 MB vs 3,588 MB (trainer adds LoRA grads, JVP buffers) |
| Dry run (N=50, C=30, 8 GPUs) | preflight 8 ✓ 1 ⚠ (feature cache cold); `model_name`, `lora_r: 4`, delays reach both roles |

**Delays.** `--delay-divisor 0.0192 --delay-floor 4.0` = §O's fluxtune 0.48 ÷ 25 (≈ DistilBERT→1.7B parameter ratio, 26×),
so the fastest modeled delay is ~208 s (DistilBERT 8.33 s). Not measured on a real device; revisit if `[TIMING_OVERRUN]` or
the vclock/wall ratio looks off. Vclock scales with the delays: `--max-runtime-s 2000000`, `--sim-wall-ceiling-h 80`.

**Commands** (env `test_fwdllm`; from `expt_scripts/`; the operator launches, R4):

```
python pretokenize_dataset.py --dataset agnews --clients 50 --jobs 16 --model-type llama \
  --model-name HuggingFaceTB/SmolLM2-1.7B --partition-method "niid_label_clients=50_alpha=1"
# smoke (step 6): ~1 h; same flags, 1 h ceiling, vclock capped
EXPT_GPU_ALLOW_PEER=1 setsid nohup ./run_sequential.sh --yes --clean --only fluxtune --mode sim --allow-stale-profile \
  --dataset agnews --partition-method "niid_label_clients=50_alpha=1" --num-trainers 50 --num-gpus 8 \
  --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 \
  --adapter-reduction-factor 16 --fd-scale-invariant --delay-divisor 0.0192 --delay-floor 4.0 \
  --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --commit-gate fixed --pool-target 50 --rho-schedule pool --agg-rate-type uniform \
  --var-stopping-policy off --max-iter-per-data-id 20 \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  --saturation-stop --no-sat-decay --sat-stall-frac 0.003 \
  --server-update-audit --no-cos-ground-truth-audit \
  --max-runtime-s 2000000 --sim-wall-ceiling-h 1 \
  > ../smollm2_1p7b_smoke_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
# full run: identical but --sim-wall-ceiling-h 80, plus the §8 watchdog with --max-hours 81
```

**Smoke pass condition (step 6):** 0 OOM / NaN / dead trainers / `[TIMING_OVERRUN]`; `[ProbeDim] p=794624`; 5.0 trips/commit;
commits/h → projection (~35 h to Φ = 3 predicted). GPUs 0–3 carry peer jobs (15–20 GB): watch memory there, or use
`--gpu-ids 4,5,6,7` with `--num-trainers` ≤ ~30.

**First run: n25, O1–O5, old aggregator** (`run_20261014_014720_fluxtune_agnews_n25_smoke_syn_0_sim`; launcher log
`smollm2_1p7b_n25_20261014_0147.log`). The step-6 command with `--num-trainers 25` (shards 0–24 of `clients=100`,
pretokenized), `--phi-stop log_only`, no saturation stop, caps 1e9 vclock-s / 1000 h; stopped by hand at 1 h 05 min.
Not a scored run.

| Reading | Result |
|---|---|
| `p`, FD | 794,624; `scale_invariant=on`, h 0.0075 ✓ |
| Commits, pace | 63 in 1.08 h (~50/h; 360M n10 80/h, N=50 109/h) |
| Accuracy | 0.271 → 0.368 (14 min) → 0.451 (28) → 0.498 (43) → **0.567** best (~1 h); 360M n10 had 0.553 at 1 h |
| Per commit vs 360M | on par or ahead: 0.317 at commit 10 (360M 0.285 n10 / 0.258 N=50) |
| Trips/commit, deaths, NaN / OOM / overrun | 4.95, 0, 0 / 0 / 0 |
| Memory | trainers per §0; aggregator 108–110 GB host (fixed by A1–A3, RC12) |

**Dispatch timeout too short for the scaled delays.** `send_timeout_wait_s: 300` (yaml templates; no launcher flag)
against modeled delays of 208–2,448 vclock-s: 1,885 `abandon_timeout` (logged as `abandon_90s_wall`) on 20 of 25
trainers, 2,856 `slot_starvation`. No update lost (3,165 computed, 3,140 aggregated, 25 in flight at stop; 62% of
contributions from "abandoned" trainers; staleness 0–1), because C = N here. With C < N a freed slot would dispatch past
C. For 1.7B pass `--send-timeout-s 3600` (above the 2,448 s slowest delay), as the N = 50 run below does.

**N = 50 run, 8 h wall (launched 2026-10-14 03:46).** O1–O5 trainers + A1–A4 aggregator; agnews
`niid_label_clients=50_alpha=1` (pretokenized for 1.7B, 51 shards in 1.3 min); N / C / K = 50 / 30 / 10; all 8 GPUs.
The step-6 command with these changes:

```
--partition-method "niid_label_clients=50_alpha=1" --num-trainers 50 --num-gpus 8 --c 30 --min-initial-frac 0.9
--send-timeout-s 3600            # new launcher flag -> send_timeout_wait_s (yaml default 300)
--phi-stop log_only              # no Φ halt, no --saturation-stop: only the wall ceiling ends it
--max-runtime-s 1000000000 --sim-wall-ceiling-h 8
```

| | |
|---|---|
| Run dir | `experiments/run_20261014_034646_fluxtune_agnews_n50_smoke_syn_0_sim/` |
| Aggregator / trainers log | `<run dir>/14_10_26_03_47_async_oort_n50_client_notify_alpha1_syn0_{aggregator,trainers}.log` |
| Accuracy, commits | `<run dir>/telemetry/aggregator_fluxtune_agnews_n50_smoke_syn_0_sim.jsonl` (`agg_eval`, `server_update`) |
| Aggregator RAM (RC12) | `[MEM:agg] rss=` lines in the aggregator log |
| Launcher stdout | `smollm2_1p7b_n50_20261014_0346.log` |
| Watchdog | `smollm2_1p7b_n50_watch.log` (`--pgid 1494946 --kill --max-hours 9`) |
| Rendered config | `expt_scripts/smoke_logs/20261014_034638/` (dry run: `…/20261014_034613/`, preflight 9/9) |
| Stop by hand | `kill -TERM -- -1494946`, then `kill -KILL -- -1494946` after ~20 s |

**Stop:** the aggregator's `sim_wall_ceiling_s` 28,800 s from its start (03:47:19) → ~11:47; launcher hard kill at
+20 min, watchdog at 9 h. **Spawn:** aggregator on GPU 7 (03:47:10); 50 trainers `(id − 1) % 8` round-robin, 2 s apart
(03:47:15–03:48:58), 7/7/6/6/6/6/6/6 per GPU, one pinned core each, each loading its own model. All 50 updating by
03:54; C = 30 honoured (`effective_c: 30`).

**Projection** (ρ = 1.5·√(10·50/794,624) ≈ 0.0376, as 360M N=50 §9; 50–75 commits/h): ~390–580 commits and
**~0.81–0.83** accuracy at 8 h (360M N=50 at matched commits; up to ~0.85 if 1.7B keeps its early per-commit lead).
Φ = 3 needs ~1,560 commits (~21–31 h), so 8 h ends before the run's own stop.

**Owed:** RC12 (aggregator RAM, read from this run); step 6 pass condition from this run (= RC10).

## 14. SmolLM2-1.7B on Yahoo: rank / probes / momentum smokes (2026-10-25)

Follows §13 (p = 794,624 there; the harness telemetry reads 806,912 at r=4 with the head). Snapshot taken 2026-10-26 00:30; the four sim legs below are still running, so the tables are a point in time (regenerate from the run dirs).

### Summary
- **Backprop ceiling (r=4, centralised, pooled test sample): ~0.77** (plateau from step 750; single seed). Best FluxTune SmolLM so far: 0.598 (shepherd, n=60). The 0.65 target was inherited from DistilBERT's FL target (0.660 vs its 0.734 ceiling, a 7-pt gap); for SmolLM a 5-8 pt drop from 0.77 suggests **0.69-0.72**.
- **Best so far: LoRA r=2 + server momentum 0.9 at the same s (2.81).** At matched commits it is ahead of plain r=2 (c12: 0.249 / 2.119 vs 0.212 / 2.237) and of base (0.182 / 2.302). Promising, not proven: one seed, n=12 smoke, and only ~4 evals past the point where the momentum buffer filled.
- **r=2 beats r=4 at the same s** (loss ~0.05-0.09 lower, acc +3-6 pts by c30), once rho is scaled with the cosine.
- **K=40 probes is compute-neutral at best** (4x cost per commit for ~4x progress per commit), so it is not worth running.
- **Decay (`t_res`) is irrelevant in the first ~30 commits**; only the combined r=2+momentum leg uses the slower t_res 1119, so that confound is untested.

### What we learned (order of discovery)
1. `--cos-ground-truth-audit` stalls every leg at commit 1 (aggregator at 100% CPU, ~60-70 GB RAM, GPUs idle), even with `--cos-probe-batch-size 64 --cos-probe-every 5`. Keep it `--no-cos-ground-truth-audit` for SmolLM.
2. In this harness `s = rho0 / cos_theory`, `cos_theory = sqrt(G*N/p)` (G probes, N pool, p trainable). A fixed `rho0` across arms lets s vary (first r=2 / K=40 legs had s = 2.01 / 1.40 against base 2.81) and handicaps them. Hold **s fixed**: rho0 = 0.0698 (r=4,k=10), 0.0978 (r=2,k=10), 0.1387 (r=4,k=40).
3. Under `trust_ratio` the step is `rho * ||theta||`; ||theta|| scales ~sqrt(p), so with a fixed rho0 the cosine gain from a smaller p is cancelled (step x cos was 0.0148 base vs 0.0155 r=2).
4. LoRA rank barely changes per-task cost (probe forwards dominate): ~21 s r=4 vs ~23 s r=2 per task; commit gap 5.1 vs 5.4 min.
5. Momentum is heavy-ball applied before the trust-ratio normalisation, so the step norm stays `rho*||theta||`; its benefit appears after the buffer fills (~10 commits). It started behind plain r=2 (c3, c6) and crossed at c9.

### Shared arm settings (`expt_scripts/run_sequential.sh`, sim mode)
`--only fluxtune --dataset yahoo --partition-method niid_label_clients=100_alpha=1 --num-trainers 12 --c 6 --c-async 6 --agg-goal 5 --min-initial-frac 0.9 --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --adapter-reduction-factor 16 --fd-scale-invariant --delay-divisor 0.0047 --delay-floor 4.0 --send-timeout-s 3600 --train-batch-size 32 --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 2 --commit-gate fixed --pool-target 50 --rho-schedule exp --agg-rate-type uniform --var-stopping-policy off --phi-stop-threshold 4 --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt --server-update-audit --no-cos-ground-truth-audit --eval-max-samples 2000 --eval-resample --eval-every-n-commits 3 --max-commits 150 --max-runtime-s 1000000000 --sim-wall-ceiling-h 24 --force --yes`
Launched through `scripts/bench_pool.py --gpus-per-leg 2 --per-gpu 1` (jobs files in `bench_sm_20261025/`), 2 GPUs per leg, `workers=6`.

### Arms (per-arm values)
| arm | k | LoRA r | p | rho0 | s | t_res | momentum | GPUs | config yaml | run dir |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 10 | 4 | 806,912 | 0.07 | 2.81 | 560 | 0 | 0,1 | expt_scripts/smoke_logs/20261025_204018/ | bench_sm_20261025/POOL/sm_base/runs/syn_0_fluxtune_s1/run_20261025_204054_* |
| r2_s | 10 | 2 | 413,696 | 0.0978 | 2.81 | 560 | 0 | 4,5 | expt_scripts/smoke_logs/20261025_211919/ | bench_sm_20261025/POOL_s/sm_r2s/runs/syn_0_fluxtune_s1/run_20261025_211930_* |
| k40_s | 40 | 4 | 806,912 | 0.14 | 2.81 | 560 | 0 | 2,3 | expt_scripts/smoke_logs/20261025_211907/ | bench_sm_20261025/POOL_s/sm_k40s/runs/syn_0_fluxtune_s1/run_20261025_211930_* |
| r2_mom | 10 | 2 | 413,696 | 0.0978 | 2.81 | 1119 | 0.9 | 6,7 | expt_scripts/smoke_logs/20261025_224827/ | bench_sm_20261025/POOL_m/sm_r2m/runs/syn_0_fluxtune_s1/run_20261025_224838_* |

Logs inside each run dir: `bench.log` (aggregator + trainers), `telemetry/aggregator_*.jsonl` (`server_update`, `agg_eval`, `agg_round`), `telemetry/trainer_*.jsonl`; the bench driver log is `bench_sm_20261025/bench{3,4,5}.log` and `leg.out` under each `POOL*/sm_*/runs/*/`.

### Per-arm controller values (first commit; mean commit gap)
| arm | commits | rho (c1) | cos_theory | s=rho/cos | step norm (c1) | weight norm now | min / commit |
|---|---|---|---|---|---|---|---|
| base | 44 | 0.0698 | 0.0249 | 2.81 | 0.595 | 9.42 | 5.1 |
| r2_s | 36 | 0.0973 | 0.0348 | 2.80 | 0.620 | 7.50 | 5.3 |
| k40_s | 9 | 0.1387 | 0.0498 | 2.79 | 1.190 | 9.27 | 19.6 |
| r2_mom | 17 | 0.0973 | 0.0348 | 2.80 | 0.620 | 11.76 | 5.7 |

### Matched evals: accuracy / test loss (wall minutes since leg start)
| eval | base | r=2 same s | K=40 same s | r=2 + momentum |
|---|---|---|---|---|
| c0 | 0.131 / 2.765 (0 m) | 0.135 / 2.698 (0 m) | 0.144 / 2.506 (0 m) | 0.132 / 2.699 (0 m) |
| c3 | 0.158 / 2.464 (16 m) | 0.153 / 2.407 (16 m) | 0.180 / 2.281 (58 m) | 0.142 / 2.480 (18 m) |
| c6 | 0.143 / 2.408 (31 m) | 0.155 / 2.367 (32 m) | 0.196 / 2.189 (117 m) | 0.170 / 2.510 (34 m) |
| c9 | 0.149 / 2.336 (47 m) | 0.171 / 2.288 (48 m) | – | 0.202 / 2.207 (51 m) |
| c12 | 0.182 / 2.302 (63 m) | 0.212 / 2.237 (65 m) | – | 0.249 / 2.119 (68 m) |
| c15 | 0.203 / 2.246 (77 m) | 0.214 / 2.209 (81 m) | – | 0.283 / 2.065 (85 m) |
| c18 | 0.210 / 2.217 (92 m) | 0.226 / 2.170 (97 m) | – | – |
| c21 | 0.228 / 2.168 (107 m) | 0.266 / 2.115 (112 m) | – | – |
| c24 | 0.233 / 2.128 (122 m) | 0.263 / 2.065 (128 m) | – | – |
| c27 | 0.241 / 2.088 (137 m) | 0.285 / 1.999 (143 m) | – | – |
| c30 | 0.243 / 2.118 (152 m) | 0.300 / 2.029 (158 m) | – | – |
| c33 | 0.266 / 2.071 (167 m) | 0.319 / 1.983 (174 m) | – | – |
| c36 | 0.290 / 2.031 (183 m) | – | – | – |
| c39 | 0.277 / 2.035 (198 m) | – | – | – |
| c42 | 0.328 / 1.963 (212 m) | – | – | – |

Eval noise: 2000 resampled test rows, accuracy ~±0.8 pt, loss ~±0.02.

### Backprop ceiling (`expt_scripts/backprop_ceiling.py`, AdamW lr 2e-4, bs 32, bf16 autocast, seed 1233, 3000-row test sample)
- r=4: step 250 0.751, 500 0.757, 750 0.770, 1000 0.761, 1250 0.775, 1500 0.767, 1750 0.769 (stopped). Trainable p = 806,912, same as the harness. Curve `bp_ceiling_20261025/r4/curve.jsonl`, log `bp_ceiling_20261025/r4.log`.
- r=2: only step 250 logged, 0.746 / 0.816 (stopped by request). p = 413,696. `bp_ceiling_20261025/r2/curve.jsonl`, `r2.log`.

### Earlier / discarded runs (for reference)
| run | what | result | location |
|---|---|---|---|
| n=60 baseline (stopped at c398) | rho0 0.0497 (s=2), t_res 1119, k 10, r 4, pool 50, agg-goal 10, c 30, eval 6000 | acc ~0.56 (best 0.576 by c398); loss still falling ~0.09 per 100 commits when stopped, heading to the 10-23 run's ~0.59 plateau | experiments/run_20261025_075720_fluxtune_yahoo_n60_smoke_syn_0_sim/ |
| yesterday n=60 (shepherd, 14.4 h, 531 commits) | rho0 0.07, t_res 283 | best 0.598, plateau 0.57-0.595 from c350 | shepherd:experiments/run_20261023_220053_fluxtune_yahoo_n60_smoke_syn_0_sim/ |
| audit stalls | cos audit batch 1024 / 64 | stuck at commit 1 | bench_sm_20261025/POOL_audit1024_stalled/, POOL_audit64_stalled/ |
| first k40 / r2 legs (fixed rho0 0.07, s 1.40 / 2.01) | killed ~21:19 | r2 c3 2.455 vs base 2.464 (equal) | bench_sm_20261025/POOL/sm_k40, POOL/sm_r2 |

### Caveats
- n=12 trainers, 150-commit cap, single seed; do not compare with the n=60 runs (different n, agg-goal, s, eval size).
- r=2+momentum also carries `t_res 1119` (the others 560); both changes were requested together.
- Momentum's trainable weight norm grows faster (11.15 at c16 vs 7.43 for plain r=2 at c34, from the same 6.4 start); watch for instability.

### Next
1. Read r2_mom against r2_s at c15-c30; if it holds, launch a full-size (n=60) r=2+momentum run with `--target-stop 0.70` and no commit cap.
2. Try r=1 at the same s, and r=2+momentum with `t_res 560` to separate the decay confound.
3. Port the same-s rule and the audit-off setting to the RoBERTa-L recipe (see FLUXTUNE_READINESS.md).
