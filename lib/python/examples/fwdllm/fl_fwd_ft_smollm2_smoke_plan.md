# SmolLM2-360M + LoRA smoke test

2026-10-10. Ported, checked offline and smoked (results §6; simplified 6 h run §7–§8; DistilBERT parity run, N=50, §9;
trainer memory fixes §10; further improvements §11; ~1B model §12). Why this model and the full-run
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
- Dtype mismatch in a forward, or to rule out §10's bf16 frozen weights: rerun with `FWDLLM_FROZEN_LOWP=0`.
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

## 10. Trainer memory fixes (2026-10-13, uncommitted)

**Diagnosis.** `self.params` (from `make_functional_with_buffers`) holds every parameter, frozen backbone included.
Each trainer therefore carried full-size fp32 zeros for frozen slots in three more places, which explains the 6.1 GB:

| Where | What |
|---|---|
| `v_params` (`_prepare_perturbation_tensors`) | `zeros_like` per frozen param, GPU |
| `self.grad` (`_setup_training_state`) | `zeros_like` per param, GPU; copied to CPU every update (`_finalize_training`) |
| `full_grad` (`fwdllm_trainer.py`, `GRAD_POOL`) | zero padding per frozen param, CPU |
| `make_functional_with_buffers` | a second full copy of the model, kept alongside `self.model` (**not fixed**) |

**Fix 1: zero-size frozen slots (all models).** `frozen_zero(p)` in `fwdgrad_utils.py` is a stride-0 view: same shape,
one element of storage. It replaces the zeros above; the two accumulation loops skip frozen slots (they only added
`jvp × 0`). The var-check slice still reads zeros if `layer_id_for_check` is frozen. The upload is unchanged
(`_send_grads` sends trainable `p.grad` only).

**Fix 2: frozen Linear weights in `amp_dtype()` (llama only by default).** `store_frozen_linears_lowp` in
`fwdgrad_utils.py`, called from `trainer/main.py` after `create_model`. Exact because autocast already casts these
weights to bf16 on every call, and `adapters` LoRA runs the frozen matmul separately from the LoRA delta. LoRA, `score`,
embeddings and norms stay fp32.
- Gate `frozen_lowp_enabled`: `FWDLLM_FROZEN_LOWP=1|0`, unset = llama only; forced off without `fp16` (autocast) or
  under `FWDLLM_JVP_FP32=1`. DistilBERT/roberta (fp16 autocast) are untouched.
- Trainer only: the aggregator's retention and `_cos_probe_gradient` probes run fp32 outside autocast.
- `lowp_autocast(model)` wraps GPU warmup and `eval_model`, which run outside the training loop's autocast. Behaviour
  change: `eval_model` called from `fed_trainer_transformer.py:40` runs bf16 when the model holds bf16 weights.
  Reported accuracy comes from the aggregator and is unaffected.

**Verification.** Same 4-trainer sim on GPU 4, `FWDLLM_STRICT_DETERMINISM=1`, 500 vclock-s, 7 commits (data_id 0–7):

| | Original | Fix 1 | Fixes 1 + 2 |
|---|---|---|---|
| Run dir (`experiments/run_20261013_…`) | `005113` | `010457` | `011909` |
| `rms_jvp`, 60 updates | — | identical | identical |
| Eval acc / loss, 3 evals | — | identical | identical |
| GPU allocated / reserved per trainer | 2,951 / 6,084 MB | 2,951 / 4,425 MB | **1,714 / 2,609 MB** |
| Host RAM per trainer | 6,562 MB | 2,526 MB | 2,467 MB |
| `_train_one_batch` median | 6.12 s | 5.33 s | **4.67 s** |
| Traceback / OOM / NaN / `[TIMING_OVERRUN]` | 0 | 0 / 0 / 0 / 1 | 0 |

The one overrun was a 4.3 s first `_make_model_functional`, not a slowdown. Offline (`FWDLLM_STRICT_DETERMINISM=1`,
SmolLM2-360M): loss, JVP and eval output identical with fp32 vs bf16 frozen storage; params 1,451 → 821 MB.
`tests/mode/test_fwdllm_gpu_warmup.py` + `test_fwdllm_probe_report.py`: 51 pass.

**Not yet shown.** 7 commits prove identity, not learning (acc at chance) and cover few paths. Sanity runs still owed:

| # | Run | Exit criterion |
|---|---|---|
| S1 | 360M, 4-trainer deterministic before/after, ~150 commits (~1.5 h) | every JVP identical; acc clearly above chance |
| S2 | §9 config with the fixes, ~2 h (~200 commits) | acc-vs-commit tracks §9 within noise |
| S3 | DistilBERT deterministic before/after, then parity vs floors (fix 1 is shared code, R10) | identical; parity passes |
| S4 | short identity runs: `var_control` (odd data_id `old_grad`), winner-only probe, perf-opt off, retention + cos audits | identical, no errors |

## 11. Improvements not yet made

| Improvement | Gain | Cost / risk |
|---|---|---|
| Drop the `make_functional` copy: call the model with its own params (`torch.func.functional_call`) | removes one full model copy; largest remaining cost | trainer refactor; re-verify identity |
| Staggered trainer start + `low_cpu_mem_usage=True` | avoids the startup host-RAM spike (each trainer loads fp32 on CPU first) | launcher change |
| Recalibrate `training_delay_factor` per model (FWDLLM_DESIGN §O method × forward cost vs DistilBERT) | realistic device model; GPU time stays under modeled delay, so contention never reaches the vclock | config; per model |
| Batch the 20 probe forwards (10 probes × ±h) into 1–2 calls | big for 360M (63 ms/fwd vs ~7 ms of arithmetic), ~1.5–2× at 1.7B | `calculate_jvp` + batched LoRA forward |
| CUDA MPS on the node | kernels from co-located trainers overlap | no code; node-wide daemon, shared with other users |
| Charge profiled solo GPU time in sim, not measured | packing can never distort the vclock | sim-mode design change |
| One process per GPU sharing a frozen backbone, swapping per-client LoRA (~3 MB) | clients no longer memory-bound | restructure trainer launch/channels; needed for 7B |
| 4-bit backbone (bitsandbytes) | ~2× less weight memory | slower forwards (dequant); `functorch` + `adapters` compatibility unknown. Hold for 7B |

**Why contention matters.** Trainers on one GPU time-slice its compute (no MPS), so an update's GPU time grows with the
number of *active* trainers on that GPU, on every forward pass. Sim charges max(gpu, modeled delay); once GPU time
exceeds the delay, packing sets the vclock and update order. DistilBERT never hit this: §O's delays are calibrated to
DistilBERT's NPU cost with a +1.5 s buffer. SmolLM2 runs reused those delays (`--allow-stale-profile`), so the margin
shrank (§6: 7.8 s against 8.33 s).

## 12. Handling a ~1B model

**Model.** SmolLM2 has no 1B (135M, 360M, 1.7B). Estimates below are extrapolated from 360M; none are measured.

| | SmolLM2-1.7B | TinyLlama-1.1B | Llama-3.2-1B |
|---|---|---|---|
| Architecture | Llama, 24 layers, hidden 2048, MHA 32/32 | Llama, 22 layers, GQA 32/4 | Llama, 16 layers, GQA 32/8 |
| Tokenizer / data | same as 360M / 11T | Llama-2 32k / 3T (weak) | 128k / gated licence |
| `p`, LoRA q,v | r=8 ~1.58M; **r=4 ~0.79M** | r=8 ~1.13M | r=8 ~0.86M |
| Forward cost vs 360M | ~4.7× | ~3× | ~3.4× |

**Pick: SmolLM2-1.7B, LoRA r=4.** Same family and tokenizer as 360M with `p` ≈ 823k, so only model size changes: the
cleanest test that commits scale with `p`, not size (N5c). TinyLlama r=8 if the paper must say "1B" or memory is short.
All three run on the existing `model_type=llama` path (pad = cls = sep = EOS, bf16). Rerun `probe_port_init.py`: `p`,
chord, and that `layer_id_for_check=16` still lands on a `lora_B`.

**Memory per trainer (1.7B).** ~22 GB originally; ~9 GB with fixes 1 + 2 (4–5 per A40, ~35–40 clients on today's node:
GPUs 0, 1, 3 carry ~20 GB of peer jobs); ~5 GB once the `make_functional` copy is dropped (~8 per A40, ~50 today, ~63
with all 8 free). Host RAM (266 GB free, ~2.5 GB per trainer) caps near 80–100 trainers. Load trainers with a delay:
50 simultaneous fp32 loads would need ~340 GB of host RAM.

**C and delays.** Recalibrate delays first: DistilBERT's fastest modeled delay (~4 s) is below one 1.7B update (~6 s
solo), so no C is overrun-free today. Scaled ~25× for 1.7B, the fastest delay is ~100 s; even 8 active trainers on one
GPU (~48 s) fit, so any C ≤ N works. Keep C = 30 for parity with DistilBERT. Raise `--max-runtime-s` (vclock) with the
delays, or the run ends early; the preflight ceiling projection needs the same.

**Wall time.** Delays add no wall time in sim. ~120–160 s per commit (360M: 34 s × ~4.7× forward cost) and ~1,600
commits to Φ = 3 → **~55–70 h**; ~20–27 h to §9's 630-commit point. Probe batching: maybe ~30–45 h.

**Plan.**

| # | Step | Done when |
|---|---|---|
| 1 | Memory fixes 1 + 2 (§10) | ✅ identity on 360M; sanity runs S1–S4 still owed |
| 2 | Drop the `make_functional` copy; staggered start + `low_cpu_mem_usage` | 360M identical; ~1 GB less per trainer |
| 3 | Port SmolLM2-1.7B, `lora_r=4`; pretokenize `clients=50` | `p` ≈ 794k, chord checked |
| 4 | One trainer on a free GPU | measured solo update time and peak memory |
| 5 | Recalibrate `training_delay_factor`; raise `--max-runtime-s` and ceiling | value derived next to FWDLLM_DESIGN §O |
| 6 | ~1 h smoke, N = 50, C = 30 | 0 OOM / NaN / dead trainers / `[TIMING_OVERRUN]`; commits/h → projection |
| 7 | Probe batching | only if step 6 projects > ~3 days |
| 8 | Full run to its own stop, ceiling ~80 h, watchdog | Φ rail or stall; compare with 360M and DistilBERT at matched commit / Φ (N5c) |
