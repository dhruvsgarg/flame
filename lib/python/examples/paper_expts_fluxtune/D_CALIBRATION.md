# Task cost D: calibration per model

How the modeled device delay D is set for each model and arm, the values in use, and how to re-derive them for a new
model. Used by [RUN_PLAN.md §2](RUN_PLAN.md#2-cells-pool-size-and-eval) (launch flags) and
[EXPERIMENTS.md §2](EXPERIMENTS.md#2-setup) (setup). Doc rules: [ROBUST_FL_READINESS.md](../_metadata/ROBUST_FL_READINESS.md).

## 1. What D is

Each trainer models a phone. A training task takes the phone **D = `training_delay_s` / divisor** seconds
(`FedSgdTrainer._emulate_training_delay`):

- `training_delay_s` is the trainer's device-class delay from the 100-trainer registry, floored at 4.0 s
  (`--delay-floor 4.0`). It sets the spread between devices; the divisor sets the absolute scale.
- **Real** mode sleeps the remainder of D after the GPU work. **Sim** mode charges D to the vclock and does not sleep.
  Both finish the task at max(GPU time, D) after dispatch.
- The divisor is `--delay-divisor` (config key `training_delay_factor`), one value per model and arm.

| Registry (`training_delay_s`, 100 trainers) | fast (16) | medium (22) | slow (19) | very slow (43) | all |
|---|---|---|---|---|---|
| Mean, s | 3.0 → 4.0 after floor | 6.32 | 10.53 | 20.09 | 12.51 (median 11, range 2–47) |

## 2. Values in use

| Model · cell | bs | `fluxtune` divisor | `fwdllm` divisor | Basis |
|---|---|---|---|---|
| DistilBERT · AG News | 8 | **0.478** | **4.78** | phone-NPU anchor (§3) |
| RoBERTa-L · Yelp-P | 32 | **0.0169** | **0.169** | measured 0.0171 (§4); 0.0169 kept, within 1% |
| SmolLM2-1.7B · Yahoo | 32 | **0.0047** | **0.047** | measured (§4); replaces 0.0192 (see §6) |

Resulting D per task over the 100-trainer registry:

| Model · arm | min (fast) | medium | median | slow | mean | very slow | max |
|---|---|---|---|---|---|---|---|
| DistilBERT `fluxtune` | 8.4 s | 13.2 s | 23 s | 22 s | 26.5 s | 42 s | 1.6 min |
| DistilBERT `fwdllm` | 0.8 s | 1.3 s | 2.3 s | 2.2 s | 2.7 s | 4.2 s | 9.8 s |
| RoBERTa-L `fluxtune` | 3.9 min | 6.2 min | 10.8 min | 10.4 min | 12.5 min | 19.8 min | 46 min |
| RoBERTa-L `fwdllm` | 24 s | 37 s | 65 s | 62 s | 75 s | 2.0 min | 4.6 min |
| SmolLM2-1.7B `fluxtune` | 14.2 min | 22.4 min | 39 min | 37 min | 45 min | 71 min | 2.8 h |
| SmolLM2-1.7B `fwdllm` | 85 s | 2.2 min | 3.9 min | 3.7 min | 4.5 min | 7.1 min | 16.7 min |

Class columns are class means; min and max are the registry extremes after the floor.

## 3. Anchor: DistilBERT

DistilBERT's D comes from a phone measurement, not an A40 one (FWDLLM_DESIGN.md §O; EXPERIMENTS §9, 10-22):

```
cost per task on the device = forward passes × s/sample × bs / 2 × 1.06
  s/sample   0.3086  FwdLLM paper reference NPU, DistilBERT (FwdLLM AE bundle, train_time_dict_dict["distilbert"]["ours"])
  passes     20 fluxtune (P 10 probes × 2), 2 fwdllm (jvp_perf_opt); from trainer telemetry forward_passes_iter
  bs         8;  /2: 1 JVP = 2 forward-pass units;  × 1.06: overrun buffer, same on both arms
fluxtune: 20 × 0.3086 × 8 / 2 × 1.06 = 26.17 s     fwdllm: 2 × 0.3086 × 8 / 2 × 1.06 = 2.62 s

divisor = registry mean / cost per task:  fluxtune 12.51 / 26.17 = 0.478    fwdllm 12.51 / 2.62 = 4.78
```

## 4. Other models: measured A40 cost ratio

No phone measurement exists for RoBERTa-L or SmolLM2-1.7B (the FwdLLM public repo has none; the AE bundle, gitignored
in `4c469c71c`, is not on jayne or shepherd). Each model's cost is scaled from DistilBERT by its measured A40 cost per
task, at the **same bs**:

```
r        = model A40 s/task  /  DistilBERT A40 s/task        (both at the model's bs)
divisor  = 0.478 / (r × bs / 8)          fwdllm divisor = 10 × fluxtune divisor (arm ratio kept)
```

**Measured 10-24** (shepherd, `fwdllm/ft_profile_20261024/`; one trainer alone per A40, sim mode, eval off, first 3
tasks dropped, tasks overlapping the aggregator's eval dropped):

| Lane | Tasks | A40 s / task | Spread (CV) | r | `fluxtune` divisor |
|---|---|---|---|---|---|
| DistilBERT bs 8 | 1,943 | 0.238 | 9.8% | — | 0.478 (anchor) |
| DistilBERT bs 32 | 1,355 | 0.375 | 13.5% | 1 | — |
| RoBERTa-L bs 32 | 129 | 2.617 | 1.5% | 6.98 | 0.478 / (6.98 × 4) = 0.0171 |
| SmolLM2-1.7B bs 32 | 8 | 9.518 | 0.4% | 25.4 | 0.478 / (25.4 × 4) = 0.0047 |

**What to read:** the trainer log line `modeled delay for trainer … (not slept; charged to vclock; gpu=X.XXs)`
(also telemetry `trainer_round.real_gpu_time_s`). The per-task cost is the mean `gpu=` per lane.

## 5. Re-deriving D for a new model

1. Run [`fwdllm/expt_scripts/profile_task_cost.sh`](../fwdllm/expt_scripts/profile_task_cost.sh) (4 GPUs, ~12 min). It
   runs one trainer alone per GPU with the recipe's trainer settings, caps eval at 100 rows, and writes
   `summary.tsv` (s / task, CV, r, both divisors). Add a lane for the new model at its recipe bs.
2. Check the measurement is clean: CV ≤ ~5% and ≥ ~25 tasks after warm-up. A wide or two-level spread means the
   GPU was shared (another trainer, or the aggregator's eval: a full-test eval slowed tasks 2.7–3.5× for 4–8 min).
3. Set `--delay-divisor` per arm in the cell's launcher and in RUN_PLAN §2; update §2 here.
4. Confirm on a real leg at the paper's packing: 0 `[TIMING_OVERRUN]` (PARITY PL19 allows ≤ 2% per leg).

## 6. Rules and caveats

- **GPU time must stay well below D.** On an overrun (GPU > D) real finishes at its GPU time and sim at its own
  measured GPU time; the two differ (sim packs more concurrent work), so update order follows the hardware and real↔sim
  parity breaks. Logged as `[TIMING_OVERRUN]` in both modes. Fix: smaller divisor (longer D) or fewer trainers per GPU.
  Margins at §2 values (min D / contended GPU time): DistilBERT `fluxtune` ~35×, RoBERTa-L ~25×, SmolLM2-1.7B ~34×;
  thinnest is DistilBERT `fwdllm` (min D 0.8 s; 0% overruns measured at floor 4.0, 10-22).
- **D scales with bs.** Size-only scaling misses it: SmolLM2-1.7B's 0.0192 (0.48 / 25, the parameter ratio, from the
  bs 8 smoke plan) was 4.1× too large at bs 32. Its sim runs before 10-24 under-charge D 4.1×; accuracy per commit stands.
- **Transfer assumption.** r is measured on an A40 and assumed to hold on the phone NPU. Plausible for RoBERTa-L;
  optimistic for SmolLM2-1.7B (likely memory-bound on a phone), so its D is a lower bound.
- **D sets real wall time.** Real legs sleep D: a RoBERTa-L iteration takes 4–46 min, SmolLM2-1.7B 14 min–2.8 h.
- **Sim vclock caps.** At these D a sim leg runs ~30–100× faster than real, so pass `--max-runtime-s 1000000000` and let
  `--sim-wall-ceiling-*` end the run; the wall-budget preflight then needs `--force` (it expects the vclock cap to bind).
