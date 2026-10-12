# SmolLM2-360M + LoRA smoke test

2026-10-10. Ported, checked offline and smoked (results §6; simplified 6 h run §7). Why this model and the full-run
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

**Before launching:** check `nvidia-smi` (peer jobs held GPUs at 75–100% on 2026-10-11). The open choice is N: 50 keeps
the DistilBERT A/B setting; the full controller asked for ~169 at this `p` (ρ ≈ 0.068, ~17 iterations/commit).
