# Uncommitted changes: `aish-flux-new-strategy-validation` (as of 2026-10-22)

Base: `7ad6130b5`. Paths are relative to `lib/python/examples/`. Nothing below has been committed.

## Code

1. **`--rho-schedule exp`** (`fwdllm/aggregator/FedSgdAggregator.py`)
   - Requires `pool_target`. ρ0 = `gate_safety_s * sqrt(G_rule * N / p)` (same as `pool`).
   - `_rho_star_now()` returns `ρ0 * exp(-(t-1) / (2*T_res))`.
   - `fwdllm/expt_scripts/run_sequential.sh`: only the comment and help text changed to list `exp`. Args already pass through.
   - **Untested.** RUN_PLAN §6 requires an `exp` smoke run before figure runs.
2. **Trace bug fix** (`fwdllm/trainer/forward_training/FedSgdTrainer.py`)
   - `syn_20` / `syn_50` now read `avl_events_syn_{20,50}_3st`. The 2-state keys were deleted in `b1c53e940`, so the old reads were broken.
   - The trace names `syn_20_3st` / `syn_50_3st` are also accepted.
3. **RoBERTa-large head** (`fwdllm/trainer/forward_training/tc_transformer_trainer_distribute.py`, `fwdllm/expt_scripts/probe_port_init.py`)
   - `classifier.dense` is replaced with an empty `nn.Sequential()`, mirroring DistilBERT's `pre_classifier` drop. This cuts trainable p by 1,049,600.
   - Under LoRA, `layer_id_for_check` changes from 12 to 28. Layer 12 is a frozen bias under LoRA; 28 is layer.1 query `lora_B`.
   - **Effect:** RoBERTa results from before this change used a different p, so they don't compare directly.
4. **p registry** (`fwdllm/expts/dataset_registry.py`)
   - `ADAPTER_P_BY_MODEL["roberta-large"]` = `{16: 3171840, 6: 589824}`. The rf=16 value was 4,221,440.
   - Key 6 is LoRA r=6, measured 2026-10-18 (Yelp-P total p 591,874).

## Docs (`paper_expts_fluxtune/`)

- **EXPERIMENTS.md**
  - The 1.7B · Yahoo smoke check is marked passed.
  - Still open: the 1.7B `fwdllm` arm, RoBERTa · Yelp-P (both arms), and real charge profiles.
  - Estimate for P4 changed from ~320 to ~720 commits in 12 h.
  - Decision logged 10-20: 1.7B eval = 6,000 fixed rows, every 7 commits, and every 1.7B run passes `--no-cos-ground-truth-audit`.
- **RUN_PLAN.md**
  - The §2 eval-cost row is filled with measured values: eval ~251 s, ~43 s per commit at N 50.
  - New §3.3 run log covers the voided 10-19 run, the 15-min screen, and `run_20261020_015616`. That run stopped at 1.23 h after 99 commits, with accuracy rising from 0.141 to 0.266.
  - §4: 1.7B at N 70 ≈ 60 s per commit.
  - §6: 1.7B timings are now measured. RoBERTa timings and both charge profiles are still open.

## New, untracked files (`fwdllm/expt_scripts/`)

- `launch_smollm17b_yahoo_n50.sh`
  - Launches the 1.7B · Yahoo N=50 FluxTune run (`pool`, not `exp`).
  - Set the wall-clock hours with `HOURS=<n>`; the default is 24.
  - Also starts `watch_arm.py` (kills the run past `HOURS` + 0.5) and `track_acc.py`.
- `track_acc.py`
  - Read-only side-car that tails `telemetry/aggregator_*.jsonl` for `agg_eval` events.
  - Logs one line per eval.
  - Emits `ALERT` on: an accuracy drop of more than 0.05 below the best so far, a non-finite loss, or no eval for 20 min.
  - Exits when the run's process group exits.
- `fluxtune_llama_agnews_smoke.yaml`: **empty (0 bytes).** Delete it or leave it out of commits.

## Known issues not fixed here

- The cos ground-truth audit hangs on 1.7B (fp32 backward on the CPU model). It is only avoided with the flag.
- `run_sequential.sh` printed `COMPLETED crash=0` after the aggregator was SIGKILLed, so it does not detect crashes.
- The 1.7B runs borrowed DistilBERT's charge profile (`fluxtune_yahoo.yaml`, via `--allow-stale-profile`).

## Commit guidance

Follow the CLAUDE.md commit rules. Suggested split: (a) code items 1–4; (b) docs plus the two new scripts.
