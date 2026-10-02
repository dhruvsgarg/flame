# Aishwwarya — cheatsheet audit, telemetry and agnews run (2026-10-01)

Session notes for [fl_fwd_ft_pipeline_cheatsheet.md](fl_fwd_ft_pipeline_cheatsheet.md). They record what
was changed, why, and what is still open. The four-doc corpus remains authoritative.

## 1. Cheatsheet changes

All edits are to the concepts table (§2) unless noted.

| Change | What it does |
|---|---|
| **"not in code" marks** | Each table variable was grepped against the code. 13 are marked `*(not in code → code_name)*`; `D`, `Λ` and `A` have no code variable at all |
| **Flowchart variables added** | 16 symbols from the §1 flowchart that were missing from the table (`I`, `pool`, `var`, `m`, `N_req`, `G`, `θ_tr`, `Δθ_tr`, `φ`, `φ_knee`, `ρ*_t`, …), placed in their rows. New row 20, *Commit indexing* (`b`, `t`, `commit_count`) |
| **Impact on training** column | 🔴 accuracy/convergence · 🟠 cost only · 🟢 marginal · ⚪ none, each with the measured evidence from the corpus |
| **Simplify? (proposed)** column | A simplified maths model per row. **These are proposals, not corpus results** |
| **§3 Telemetry per concept** (new section) | Maps each row to the `run_meta` and per-commit fields that track it |

### Parity findings, not fixed in the doc

- **Defaults differ from shipped values.** Code defaults are `gate_safety_s`=0.4, `rho_star`=0.01, `rho_exp`=0.55,
  `probe_combine`=`select`, `commit_gate`=`var` and `gate_rho_ref`=`annealed`. `b_max_probe_every`=150 comes only from
  the p4 launcher. Launch scripts disagree on `s` (1.5 vs 2.9).
- **Budget bookkeeping uses `_last_rho`.** The flowchart says `B += ½·ln(1+ρ*_t²)`, but
  `FedSgdAggregator.py` uses `_last_rho`, the step actually taken.

### Main proposed simplification

The shipped `rho_star` = 0.06 is the commit gate solved backwards at `I`=2:
`ρ = s·√(P·K·I/p)` = 1.5·√(200/118,348) ≈ 0.062. Fixing `I` and deriving `ρ` this way removes `rho_star`,
`rho_exp`, `T_res`, law C and `ρ_max`. With constant `ρ`, the Φ rail becomes a commit count:
`T = 2·ln 3/ln(1+ρ²)` ≈ 611. **Untested:** constant `ρ` gives up the anneal and relies on the rail or stall stop.

## 2. Telemetry added (code)

| Record | New content | Rows |
|---|---|---|
| `run_meta`, once (`scope` = `aggregator`, `trainer_probe`, `trainer_fd`) | Every controller knob, `p_trainable` and `‖θ_tr‖` at init. The trainer's own `probe_combine`/`P` (to catch the mismatch gotcha) and FD `h` | all |
| `server_update`, per commit | `commit_count`, `g_norm`, `step_skipped`, `rho_max`, `pool_mean_sq`, `var_dim`, `pool_size`. Calculated: `cos_theory`, `aim_d` (`D`), `progress_lambda` (`Λ`) | 4, 8, 11, 13, 15, 20 |
| `bmax_probe` (new) | Each `B_max` re-sense: accuracy at each tested `Φ`, `phi_knee`, `B_max` before and after | 14 |
| `sat_state` (new), per eval | Smoothed accuracy, best, `gl`, `progress`, stall/decay streaks | 16, 17 |
| `agg_round` | Per-upload `agg_weight` (`ω_k`) and `align_cos`, in the same order as `grad_norm` | 9 |

**Files changed:**
- `flame/telemetry/events.py`
- `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`
- `examples/fwdllm/aggregator/FedSgdAggregator.py`
- `examples/fwdllm/expts/saturation_stop.py`
- `examples/fwdllm/trainer/forward_training/{fwdgrad_utils,tc_transformer_trainer_distribute}.py`

New tests are in `tests/telemetry/test_controller_events.py`. All telemetry is guarded so it can never fault training.

**Gating:**
- `server_update` needs `--server-update-audit`.
- `sat_state` needs `--saturation-stop`.
- `aim_d` needs `--cos-ground-truth-audit`.

**Tests:** 7 new tests pass. The `tests/telemetry` + `tests/mode` suite has 5 failures, and the same 5 fail on the
code without these changes (`eval_background` ×3, `probe_report`, `norms_are_inside_the_gate`).

**Not yet done:** `analyze_run.py` / `telemetry_manifest.yaml` don't plot `bmax_probe` or `sat_state`.

## 3. Running

**Conda env: `test_fwdllm`.** It links `flame` to `/home/dgarg39/flame`, but the launcher pins `PYTHONPATH` to this
checkout. That was verified: the new telemetry code is what loads.

### Full agnews run (3 h hard stop, full logging)

```bash
conda activate test_fwdllm
cd /home/dgarg39/aish_test/flame/lib/python/examples/fwdllm
EXPT_GPU_ALLOW_PEER=1 FLAME_CONDA_ENV=test_fwdllm FWDLLM_FD_SCALE_INVARIANT=1 \
./expt_scripts/run_sequential.sh \
  --only fluxtune --mode sim --yes --clean --allow-stale-profile --dataset agnews \
  --num-trainers 100 --num-gpus 5 --gpu-ids 0,2,4,5,6 --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --probe-combine mean --commit-gate n_target --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --gate-rho-ref annealed --adapter-reduction-factor 16 --max-iter-per-data-id 20 \
  --rho-schedule landing --t-res 300 --budget-stop-frac 0.95 \
  --b-max-policy anchor --b-max-probe-every 150 --b-max-probe-n 512 \
  --saturation-stop --phi-stop halt \
  --server-update-audit --cos-ground-truth-audit --cos-probe-every 50 --retention-probe-every 25 \
  --max-runtime-s 48000 --sim-wall-ceiling-h 3 \
  2>&1 | tee agnews_full_$(date +%Y%m%d_%H%M).log
```

**Gotchas hit while launching:**
- **Runtime flags:** without `--max-runtime-s 48000 --sim-wall-ceiling-h 3`, the defaults (600 s, no ceiling) make
  the preflight block.
- **GPU check:** it reads the busiest GPU on the node and ignores `--gpu-ids`. Another user's jobs on GPUs 1, 3 and 7
  trigger it, so `EXPT_GPU_ALLOW_PEER=1` is needed, set inline: an `export` in another shell doesn't carry over.
  This also skips the memory check on our own GPUs, so watch for `CUDA out of memory`.
- **Backward-pass audit cost:** about 85 s per fire, which is why `--cos-probe-every 50`. The preflight estimated
  7,848 s against the 10,800 s ceiling.

**Run status:** started 12:07:14 as `experiments/run_20261001_120715_fluxtune_agnews_n100_smoke_syn_0_sim`
("smoke" is only the launcher's label). Trainers loaded on GPUs 0, 2, 4, 5 and 6 with no errors. The aggregator
`run_meta` was emitted (`p_trainable`=450,340, `‖θ_tr‖`=13.34). First commit at 12:14:46, about 7.5 min after start.

**Log files.** Paths are relative to `/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/`.

| What | Path |
|---|---|
| Run directory | `experiments/run_20261001_120715_fluxtune_agnews_n100_smoke_syn_0_sim/` |
| Aggregator log (commits, `[ServerStep]`, `[CommitGate]`, `[BmaxProbe]`, `[SatStop]`) | `<run dir>/01_10_26_12_07_async_oort_n100_client_notify_alpha1_syn0_aggregator.log` |
| Trainers log (`[FD] spacing`, `[JVP_EVAL_MODE]`, OOMs) | `<run dir>/01_10_26_12_07_async_oort_n100_client_notify_alpha1_syn0_trainers.log` |
| Aggregator telemetry (`run_meta`, `server_update`, `bmax_probe`, `sat_state`, `agg_round`, `agg_eval`) | `<run dir>/telemetry/aggregator_fluxtune_agnews_n100_smoke_syn_0_sim.jsonl` |
| Trainer telemetry, one file per trainer (`run_meta` `trainer_probe` / `trainer_fd`, `trainer_round`) | `<run dir>/telemetry/trainer_*.jsonl` (100 files) |
| Resolved config as launched | `<run dir>/aggregator_config.json`, `execution_config.yaml`, `snapshot.yaml` |
| Launcher logs (progress, stdout) | `expt_scripts/smoke_logs/20261001_120712/` (`expt_runner.log`, `fluxtune_agnews_n100_smoke_syn_0_sim.out`) |
| Console output (`tee`) | `agnews_full_20261001_1206.log` |

`agnews_full_20261001_1156.log` and `agnews_full_20261001_1203.log` are the two aborted launch attempts
(preflight block and GPU-peer refusal).

### Checking a run

```bash
R=$(ls -1dt experiments/run_* | head -1)
grep -c "\[ServerStep\] trust_ratio commit=" $R/*aggregator.log
grep -iE "Traceback|out of memory" $R/*.log | head
```

`check_smoke.py` (session scratchpad, not in the repo) checks that every new field is present and that training
behaves: realised `ρ` ≈ `ρ*`, `B` never decreases, no NaN weights, accuracy rises.

## 4. Next step: does each concept follow its expected trend?

Once the agnews run finishes, check each concept row against the trend the corpus predicts. A **pass**
confirms the corpus on this run. A **fail** is a finding: either the code doesn't do what the corpus says,
or the corpus claim doesn't hold. Fields are from §3 of the cheatsheet.

Expected values are worked out for this run: `p`=450,340 (`rf`=16), `s`=1.5, `P`=`G_rule`=10, `K`=10,
`max_iter`=20, `ρ*_0`≈0.068 (preflight), `‖θ_tr‖`=13.34 at init.

| # | Concept | Expected trend | Check (fields) | Pass if |
|---|---|---|---|---|
| 1 | Forward gradient | ‖u_k‖ stays the same order of magnitude; no blow-up | `agg_round.grad_norm` over commits | median ‖u_k‖ per commit stays within one order of magnitude |
| 2 | FD spacing | Relative nudge `ε = fd_displacement/‖θ_tr‖` **shrinks** as `‖θ_tr‖` grows (13.35 → ~62 on agnews ⇒ `ε` 0.50 → ~0.11) | `run_meta.fd_displacement`, `trainable_weight_norm` | `ε` falls in step with `1/‖θ_tr‖` |
| 3 | Probe combination | Trainer and aggregator agree | `run_meta.probe_combine` (all scopes), `g_rule` | all `mean`, `g_rule`=10 |
| 4 | Pooling / aim | `n_eff` is an identity, so it can't see aim. Measured aim is far from theory and drifts | `n_eff/pool_size`, `aim_d` = `cos_ground_truth/cos_theory` | `n_eff/pool_size` = 1.00 ± 0.01; `aim_d` ≠ 1 and drifts 2–3× (corpus: 20× off theory) |
| 5 | Cohort | Async pipeline stays full; throughput steady | `agg_round.in_flight`, commit timestamps, `staleness` | `in_flight` ≈ `C`=30; commits/s has no downward trend |
| 6 | Safety criterion | Gate rule `ρ ≤ s·cos`, the "don't outstep your aim" rule | `rho / cos_ground_truth` | report the ratio vs `s`=1.5. Ratio ≫ `s` ⇒ the rule holds only in theory, consistent with row 4 |
| 7 | Commit gate | `n_req = p·(ρ*/s)²/P` exactly. At `ρ*_0` ≈ 0.068: `n_req` ≈ 93 ⇒ `I` ≈ 10. Under `annealed`, `I` falls as `ρ*` falls | `n_req`, `rho_star`, `iteration_per_data_id`, `commit_reason` | `n_req` matches the formula to float precision; `I` ≈ `ceil(n_req/K)`; mostly `natural` commits, few `cap` |
| 8 | Reachability cap | `ρ_max = s·√(max_iter·K·P/p)` ≈ 0.0999; `ρ*` never exceeds it | `rho_max`, `rho_star` | `rho_star ≤ rho_max` on every commit |
| 9 | Aggregation weights | Near-inert: narrow band 0.70–0.87, median ≈ 0.82 | `agg_round.agg_weight`, `align_cos`, `grad_aware_gated_total` | `ω_k` inside the band, median 0.80–0.84; gated count ≈ 0 |
| 10 | Legacy var gate | Not used under `n_target` | `agg_round.commit_reason` | no commits caused by `var_threshold` |
| 11 | Trust-ratio step | Realised step equals the requested step | `rho` vs `rho_star`, `step_skipped` | `|rho − rho_star|/rho_star` < 1% on every commit; no skipped steps |
| 12 | Budget law | `Φ = e^B` predicts weight growth (corpus: median miss 0.09%) | `phi` vs `trainable_weight_norm / theta_tr_norm_init` | median relative miss < 1% |
| 13 | Progress law | Accuracy is a function of `Λ = 2B/s`, not of the schedule | `agg_eval.test-accuracy` vs `progress_lambda` | accuracy rises monotonically (smoothed) with `Λ` until the plateau |
| 14 | `B_max` | Probe falsified: `Φ_knee` ≈ 1.25 and `B_max` recedes with `B`, so headroom stays ≈ constant | `bmax_probe.phi_knee`, `b_max_after − budget_b` | `phi_knee` ≈ 1.2–1.3; headroom flat at ≈ 0.22–0.25 (`ln 1.25` ≈ 0.22). Confirms the falsification |
| 15 | Law C anneal | Degenerates to constant `ρ` once the first probe lands (flat headroom ⇒ flat `ρ*`) | `rho_star` over `commit_count` | `rho_star` flat after commit 150 rather than annealing |
| 16 | Stop: stall | `progress` falls toward ≤ 0.003 on the plateau. The stall trigger is **off** in this run, so `stalls` stays 0 | `sat_state.progress`, `stalls` | `progress` decays to ~0.003 at the plateau; note the commit where it would have fired |
| 17 | Stop: decay (GL) | Blind to a plateau at the best: `gl` ≈ 0 while flat | `sat_state.gl`, `breaches` | `gl` ≈ 0 on the plateau; if it fires, only after a real decline |
| 18 | Stop: Φ rail | Peak accuracy lands at `Φ` ≈ 2.8–3.0 | `phi` at max smoothed accuracy; `stop_reason` | peak `Φ` in 2.8–3.0; rail fires at 3.0 if nothing earlier does |
| 19 | Stop: budget | Unreachable under a receding `B_max` | `budget_frac` | stays < 0.95 throughout |
| 20 | Commit indexing | One data bin per commit | `data_id`, `commit_count` | `data_id` advances by one per commit, wrapping at 150 |

**Run-level target:** agnews FL target accuracy 0.88; the best forward-gradient result so far is 0.876.

**How:** write `expt_scripts/evaluate_trends.py`. It reads the run's telemetry, computes each row's check,
and prints pass/fail with the numbers. Extend `check_smoke.py`, which already loads and groups the events.
Rows 4 and 6 need `cos_ground_truth`, so only commits where the backward-pass audit fired (every 50th) count.

## 5. Open items

- Run §4 on the agnews run; then repeat on yahoo / yelp-p to separate dataset effects from code effects.
- Decide whether to fix the `_last_rho` vs `ρ*_t` wording and the default-vs-shipped gaps.
- Test the row 7/15/18 simplification (constant `ρ` from fixed `I`) as an A/B arm.
- Extend `analyze_run.py` for `bmax_probe` / `sat_state`.
- Move `check_smoke.py` into `expt_scripts/` if it is worth keeping.
- Nothing is committed yet. Per CLAUDE.md: tighten new comments, keep the diff minimal, short commit message.
