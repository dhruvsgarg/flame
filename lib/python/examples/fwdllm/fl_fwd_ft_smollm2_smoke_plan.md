# SmolLM2-360M + LoRA smoke test

2026-10-10. Ported and checked offline; the smoke itself has not been launched. Why this model and the full-run
estimate: session notes §9–§10 ([fl_fwd_ft_aishwwarya_session_2026-10-01.md](fl_fwd_ft_aishwwarya_session_2026-10-01.md)).

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
- Trainer `KeyError`/`AttributeError` on `layer_id_for_check`, or preflight `KeyError: no adapter param counts`:
  `model_type` isn't `llama`.
- OOM: lower `P4_NUM_TRAINERS`. Each trainer holds fp32 weights plus a full-size grad buffer, about 3 GB before
  activations.
