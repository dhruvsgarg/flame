# FluxTune pipeline — AI context

Dense reference for an AI assistant. Human-readable version with diagrams:
[fl_fwd_ft_pipeline_cheatsheet.md](fl_fwd_ft_pipeline_cheatsheet.md).

**Authority:** the four-doc corpus (`fl_fwd_ft_solution.md`, `practice`, `buildplan`, `writeup`) wins over this
file. Scope: the live loop only. Evidence base: one model (DistilBERT + adapters), three datasets; `rf`=64 broke
law C. SmolLM2 port: §9 (not yet run). Session notes: [fl_fwd_ft_aishwwarya_session_2026-10-01.md](fl_fwd_ft_aishwwarya_session_2026-10-01.md).

## 1. Loop (baseline, code order)

Files: `A` = `aggregator/FedSgdAggregator.py`, `F` = `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`,
`U` = `trainer/forward_training/fwdgrad_utils.py`.

```
setup:   p from rf (450,340 @ rf=16; 118,348 @ rf=64); ‖θ_tr‖0 ≈ 13.35; B=0; commit_count=0; data_id=0
per round (data_id = b):
  trainer: v_i ~ N(0,I), i=1..P; d_i = [L(θ+h·v_i) − L(θ−h·v_i)]/2h; u_k = (1/P)·Σ d_i·v_i   # probe_combine=mean
  server:  keep C in flight; on K uploads: grad += ω_k·u_k (F.aggregate_grads_from_trainers);
           grad_for_var_check_list.append(check slice, unscaled by ω); iteration_per_data_id += 1
           var = U.calculate_var(pool)        # split-half proxy, arrival order, mean over m coords
           n_eff = 2·mean_k‖u_k‖²/(m·var)     # A._compute_n_eff; pool = all uploads since last commit
  gate:    A._gate_satisfied: fixed → len(pool) ≥ pool_target; n_target → n_eff ≥ N_req = p·(ρ/s)²/G_rule
           (A._n_required, ρ from A._gate_rho); else commit forced when I ≥ max_iterations_per_data_id
commit (A._apply_weighted_update):
  5 build: G = Σ pool / training_num; momentum (off) applied BEFORE normalising (A._server_update_step)
           pass 1 over whole trainable slice: ‖G‖, ‖θ_tr‖ (= _tn, L2 of all requires_grad tensors)
           ρ*_t = A._rho_star_now() with t = commit_count+1; scale = ρ*_t·‖θ_tr‖/‖G‖ (0 if ‖G‖=0)
  6 apply: pass 2 per tensor: θ −= scale·G; frozen tensors have G=0; weight decay (off) after; step_skipped logged
  7 bank:  commit_count += 1; _last_rho = ρ*_t (trust_ratio) else measured ‖Δθ_tr‖/‖θ_tr‖;
           B += ½·log1p(_last_rho²); Φ = e^B; A._log_retention
  8 stop:  if commit_count % b_max_probe_every == 0: A._resense_b_max (BEFORE stop check, same-commit stop)
           A._check_budget_stop: latched; order saturation (stall|decay latch from eval thread) → phi_fixed
           (e^B ≥ phi_stop_threshold) → budget (landing only, B ≥ budget_stop_frac·B_max); halt → _work_done
  reset:   F._update_state_after_payload_prepared clears pool lists; data_id → next bin
```

`_rho_star_now`: `const` → ρ*; `rm` → ρ*·t^−rho_exp; `landing` → `min(ρ_max, √(2(B_max−B)/T_res))`
(`expts/landing_law.rho_star_now`, B_rem clamped at 0). `B_max` re-sense: for φ ∈ 1.5…4.0 add noise so ‖θ_tr‖
grows ×φ, read held-out acc, φ_knee = where acc falls to 0.5 of normal, `B_max = B + ln φ_knee` (`anchor`), restore θ.

## 2. Simplified variant

Flags: `--commit-gate fixed --pool-target 50 --rho-schedule pool --agg-rate-type uniform --no-sat-decay
--budget-stop-frac 0 --var-stopping-policy off` (the last is required, or the plateau rule forces early commits).
Also available: `--fd-scale-invariant`, `--dynamic-kc on|off`.

`ρ = s·√(P·N/p)` = 1.5·√(500/450,340) ≈ 0.050 constant; commit at N=50 (I=5); `G = Σ u_k` (ω=1); no probe; stop
on stall (0.003, patience 20) or Φ ≥ 3 (t = 2·ln 3/ln(1+ρ²) ≈ 880). `n_eff` still logged under audit, unused.

## 3. Terms

| Term | Meaning | Code |
|---|---|---|
| `ρ` | relative step `‖Δθ_tr‖/‖θ_tr‖` | — |
| `rho_star` | config setpoint (0.06); `rm`/`const` base; `N_req` under `gate_rho_ref=setpoint` | `_rho_star` |
| `ρ*_t` | step allowed at commit t; sets step length and `B` | `_rho_star_now()`, local `_rho_t` |
| `_last_rho` | step taken; feeds `B` and gate under `raw_sgd` | `_last_rho` |
| `ρ_max` | largest ρ whose `N_req` fits in `max_iter`: `s·√(max_iter·K·G_rule/p)`; caps `landing` | `_rho_max` |
| `G_rule` | cos² gain of one upload over one probe: `P` (mean), 2.988 (select) | `_g_rule` |
| `G` | pooled gradient vector, not `G_rule` | `weighted_gradient_sum` |
| `p`, `s` | trainable param count; safety factor in `ρ ≤ s·cos` | `_p_trainable`, `_gate_safety_s` |
| `n_eff` | independent-upload equivalent of pool; < pool if uploads disagree | `_compute_n_eff` |
| `N`, `I`, pool | pool size `K·I`; rounds this bin; uploads since commit | `len(grad_for_var_check_list)`, `iteration_per_data_id`, `grad_for_var_check_list` |
| `C`, `K` | in-flight trainers; uploads per round | `concurrency`, `aggGoal` |
| `b`, `t` | data bin; commit index | `data_id`, `commit_count` |
| `ω_k` | per-upload aggregation weight | `_grad_aware_rate` (F) |

**Proxy vs real var:** `n_eff` uses `calculate_var` (split-half, ∝ 1/N, so `n_eff` is a count and sees systematic
disagreement). `calculate_real_var` (per-sample spread over `jvp_for_snr_check_list`) is DEBUG-log only;
`real_var/N` would return ≈N by construction. Caveats: two means per coord (noisy); arrival-order split biases
it. Pool is unscaled by ω because scaling stale uploads shrank `var` → premature commits.

**Not in code:** `D`, `Λ`, `A` (audit-only concepts).

## 4. Parity gaps and gotchas

- Code defaults ≠ shipped: `s`=0.4, `rho_star`=0.01, `probe_combine`=`select`, `commit_gate`=`var`,
  `rho_schedule`=`const`. Set `probe_combine=mean` on trainer **and** aggregator (aggregator derives `G_rule`
  from its own copy, unchecked).
- `B` uses `_last_rho` (step taken), not `ρ*_t`; identical under `trust_ratio`.
- Stall progress in code is `(m_t−m_{t−h})/m_t` (no chance level); corpus says `/(m_t−chance)`.
- GL breach also requires no gain over the 150-commit horizon.
- Gate ≡ `cos ≥ ρ/s` (since `cos ≈ √(G_rule·N/p)`). With `gate_rho_ref=setpoint`, `N_req` is constant ⇒ `I`=2.
- Pool grows in steps of `K`: pool = `N_req` rounded up to a multiple of `K`.
- `B_max` probe is falsified (measures the trajectory, not the model): pins `Φ_knee`≈1.25, so `B_max` recedes
  as fast as `B` grows and law C degenerates to constant ρ.
- `landing` refuses `server_momentum` ≠ 0 (Φ ≠ e^B under momentum).
- Telemetry offsets: `rho_star`, `n_req` in `server_update` belong to the next commit; `iteration_per_data_id`
  counts from 0; `rho` is measured after the step (~0.2% low).

## 5. Concepts

Generalises: ✅ scale-free · 🟡 form carries, one empirical constant · ❌ moves with model/data/optimizer.
Impact: 🔴 accuracy · 🟠 cost only · 🟢 marginal · ⚪ none live. ✂️ = proposed removal/merge (not corpus).

| # | Concept | Variables (shipped) | Status | Impact | Gen. | Proposal |
|---|---|---|---|---|---|---|
| 1 | Forward gradient | `P`=10, `v_i`, `d_i`, `u_k`, `jvp_eval_mode` | Current | 🔴 `cos ∝ √P` | ✅ | keep |
| 2 | FD spacing | `h`=0.01, `FWDLLM_FD_SCALE_INVARIANT` | Current | 🟢 fp16 round-off | 🟡 | ✂️ `h = ε·‖θ_tr‖/√p`, ε≈0.5 |
| 3 | Probe combination | `probe_combine`=`mean`, `G_rule`=10 | Current | 🔴 mean 3.35× faster | ✅ | ✂️ delete `select`; `G_rule ≡ P` |
| 4 | Pooling / aim | `N`, `I`, pool, `var`, `m`, `n_eff`, `D`, `p` | Current | 🔴 `cos ≈ √(N·P/p)` ≈ 0.07 | ❌ `D` 20× off theory | ✂️ `cos = c·√(P·N/p)`; drop `n_eff` |
| 5 | Cohort | `C`=30, `K`=10, `train_batch_size`=8, `dynamic_kc` off | Current | 🟠 throughput ∝ `C` | 🟡 | ✂️ fix `N`, hill-climb `C` |
| 6 | Safety criterion | `s`=1.5 | Current | 🟠 efficiency knob | 🟡 | ✂️ only `ρ/s` matters; keep one |
| 7 | Commit gate | `commit_gate`=`n_target`, `max_iter`=20, `gate_rho_ref`, `N_req` | Current | 🔴 `N ∝ ρ²`; cap binds 100% in G-1 | 🟡 | ✂️ fix `I`, `ρ = s·√(P·K·I/p)` |
| 8 | Reachability cap | `ρ_max` 0.068–0.10 | `landing` only | 🟢 | 🟡 | ✂️ gone once `I` fixed |
| 9 | Aggregation weights | `ω` 0.70–0.87, `scale`=0.4, `a_exp`=0.25, `b_exp`=0.1, `align_gate` | Landed, inert | 🟢 median 0.82 | ❌ | ✂️ `ω=1`, `G = Σ u_k` |
| 10 | Legacy var gate | `var_threshold`=0.3 | v1 only | ⚪ | ❌ `‖g‖²` units | ✂️ delete |
| 11 | Trust-ratio step | `server_step_rule`=`trust_ratio` | Current | 🔴 killed runaway | ✅ | keep: `Δθ_tr = −ρ·‖θ_tr‖·G/‖G‖` |
| 12 | Budget law | `B`, `Φ` | Exact | ⚪ accounting | ✅ | ✂️ const ρ: `Φ_t = (1+ρ²)^{t/2}` |
| 13 | Progress law | `Λ = 2B/s`, `A` | Current | ⚪ audit | 🟡 | ✂️ drop, track `B` |
| 14 | `B_max` | prior `ln 2`, every 150, `anchor`, φ 1.5…4.0, knee @ 0.5 | Falsified | 🟢 | ❌ | ✂️ `B_max = ln Φ_cap` |
| 15 | Law C anneal | `rho_star`=0.06, `rho_exp`=0.25, `T_res`=300 | Degenerates | 🔴 | ❌ `T_res` tied to `p` | ✂️ constant ρ from row 7 |
| 16 | Stop: stall | `sat_stall_frac`=0.003 (env `P4_SAT_STALL_FRAC`), patience 20 | Flagged, off | 🟠 | 🟡 | ✂️ sole accuracy stop |
| 17 | Stop: GL decay | `SAT_GL_THRESHOLD`=0.005, patience 20 | Shipped | 🟢 misses plateaus | 🟡 | ✂️ delete (stall catches decays) |
| 18 | Stop: Φ rail | `phi_stop_threshold`=3.0 | Shipped | 🔴 sets endpoint | ❌ biggest risk | ✂️ commit cap `T = 2·ln 3/ln(1+ρ²)` |
| 19 | Stop: budget | `budget_stop_frac`=0.95 | Unreachable | ⚪ | ❌ | ✂️ delete |
| 20 | Commit indexing | `data_id`, `commit_count` | Current | ⚪ | ✅ | ✂️ `b = t mod total_data_bins` |

**Constants:** `total_data_bins` 150 / 1,750 / 650 (agnews / yahoo / yelp-p); `server_momentum`=0 (refuted);
`server_weight_decay` off. **New model:** retune `c` (≡ `s`) and `Φ_cap`; check stall horizon; first test G-1
(roberta-large, where `Φ_knee` already shifted).

## 6. Telemetry per concept

`server_update` needs `server_update_audit: true`; `sat_state` needs `saturation_stop: true`; knobs once in
`run_meta` (`scope` = `aggregator`, `trainer_probe`, `trainer_fd`). `*` needs `cos_ground_truth_audit`.

| # | `run_meta` | Per commit / round | Read as |
|---|---|---|---|
| 1 | `perturbation_count`, `jvp_eval_mode` | `agg_round.grad_norm` | ‖u_k‖ spread |
| 2 | `fd_h`, `fd_displacement`, `p` | `server_update.trainable_weight_norm` | ε drift as ‖θ_tr‖ grows |
| 3 | `probe_combine`, `g_rule` (both scopes) | — | values must match |
| 4 | `p_trainable` | `pool_size`, `iteration_per_data_id`, `var_at_commit`, `var_dim`, `pool_mean_sq`, `n_eff`, `cos_theory`, `aim_d`* | `aim_d` = measured/theory cos |
| 5 | `agg_goal`, `concurrency`, `dynamic_kc`, `train_batch_size` | `agg_round.in_flight`, `staleness`, `ts` | commits/s vs `C` |
| 6 | `gate_safety_s` | `rho`, `cos_ground_truth`* | `ρ/cos` vs `s` |
| 7 | `commit_gate`, `gate_rho_ref`, `max_iterations_per_data_id` | `n_req`, `n_eff`, `agg_round.commit_reason` | `cap` vs `natural` share |
| 8 | `rho_max` | `rho_max`, `rho_star` | equal ⇒ capped |
| 9 | `agg_rate_conf` | `agg_round.agg_weight`, `align_cos`, `grad_aware_gated_total` | ω spread; ≈const ⇒ inert |
| 10 | `var_threshold` | `agg_round.var`, `var_good_enough` | v1 only |
| 11 | `server_step_rule`, `theta_tr_norm_init` | `g_norm`, `step_skipped`, `rho`, `trainable_delta_norm` | `rho` vs `rho_star` |
| 12 | — | `budget_b`, `phi` | `phi` vs norm ratio |
| 13 | — | `progress_lambda`, `agg_eval.test-accuracy` | accuracy vs Λ slope |
| 14 | `b_max_prior`, `b_max_probe_every`, `b_max_probe_phis`, `b_max_policy` | `bmax_probe` (`phis`, `accs`, `phi_knee`, `b_max_before/after`), `budget_b_max` | headroom shrinking ⇒ receding |
| 15 | `rho_star`, `rho_schedule`, `rho_exp`, `t_res` | `rho_star`, `commit_count` | ρ*_t curve |
| 16 | `sat_stall_frac` | `sat_state.progress`, `stalls` | streak |
| 17 | `saturation_stop` | `sat_state.gl`, `breaches`, `smoothed`, `best` | `gl > 0.005` streak |
| 18 | `phi_stop`, `phi_stop_threshold` | `phi`, `stop_reason` | `phi` at peak |
| 19 | `budget_stop_frac` | `budget_frac` | never reaches f |
| 20 | `total_data_bins` | `data_id`, `commit_count` | — |

## 7. Agnews A/B evidence

Setup: rf=16, `p`=450,340, `s`=1.5, P=10 (mean), C=30, K=10, 100 trainers.

- `n_eff/pool` 0.98–1.03 in both arms: uploads independent; `n_eff` acted as a counter.
- Baseline (`n_target` + `landing`): `N_req` 92.2 (c1, ρ*=0.068) → 20.3; pool sawtooths 100 → 30, reset by each
  `B_max` probe. ρ sawtooths 0.032–0.068 (mean 0.046): law C decays ~8%/50 commits, probes reset it
  (c151 ↓0.039, c301 ↑0.053, c451 —, c601 ↑0.058, c751 ↓0.037).
- Simplified: `N_req` = 50.0, `n_eff` 49–51 (unused); ρ = 0.0499 flat. `‖Δθ_tr‖_t = ρ·‖θ_0‖·(1+ρ²)^{(t−1)/2}`
  ≈ 0.667 → 2.00, telemetry within 0.16% on 881 commits; angular step ≈ ρ rad, so constant ρ = no anneal.
- Returns: +15–19 pts per 0.1 Φ at Φ 1.1–1.2; < 0.5 past Φ 1.8; ≈ 0 past Φ 2.2.
- Overshoot (simplified): smoothed peak 0.8726 at c791 (Φ 2.68) → 0.866 at the Φ=3 rail (eval noise ±0.005).
  Stall streak hit 9/20 at c741, reset by one noisy eval (c761, 0.8737), reached only 7 more before the rail.

## 8. Proposal: ρ_t = ρ0/Φ_t (untested)

Rationale, settings and alternatives: cheatsheet §6; evidence: session §8. Code-level only here:

- Schedule: `ρ_t = ρ0·θ0_norm/_tn` (constant `‖Δθ_tr‖`), valid because |cos(θ_tr, Δθ_tr)| < 0.0006 on every
  commit. Differs from `rm` (ρ*·t^−0.5), which decays from t=1.
- N fixed ≈ 98 = `p·(ρ0/s)²/P` at ρ0 0.07. Don't couple N ∝ ρ²: the gate's `s` is ~10× off (ρ/cos 14–20 vs 1.5).
- Code: new branch in `A._rho_star_now` (or a new `rho_schedule` value); `_tn` is available from pass 1;
  `theta_tr_norm_init` is already in `run_meta`. Stop on stall (streak decay, not reset) + commit cap.

## 9. Second model: SmolLM2-360M + LoRA (ported 2026-10-10, smoke not run)

Measurements and runbook: [fl_fwd_ft_smollm2_smoke_plan.md](fl_fwd_ft_smollm2_smoke_plan.md). `model_type=llama`,
`model_name=HuggingFaceTB/SmolLM2-360M`, `peft_method=lora` (r=8, α=16, q,v; `rf` ignored). Code-level facts:

- `p`=823,040 (LoRA 819,200 + `score` 960·4, no bias). ‖θ_tr‖0=13.13; ‖θ‖/√p 0.0145, not 0.0197, because
  `lora_B`=0. Chord ON 0.511 ⇒ keep `FWDLLM_FD_SCALE_INVARIANT=1` (h=0.0074).
- `lora_A` grad ≡ 0 at init (B=0) ⇒ about half of `p` carries zero signal early, lowering cos below √(n/p).
- fp16 autocast ⇒ NaN loss. `create_model` sets `FWDLLM_AMP_DTYPE=bf16`; `U.amp_dtype()` feeds the 3 JVP
  autocasts and the trainer loop; `F` eval reads the env directly. `FWDLLM_JVP_FP32=1` still overrides the JVP.
- Tokens: pad = cls = sep = EOS (id 0), so the converter emits `[EOS] text [EOS] pad…`. The model pools the
  rightmost non-pad token (the last text token). The trainer passes `input_ids` only, with no attention mask;
  padding is on the right, so the causal mask keeps real tokens from seeing it.
- Trainer `layer_id_for_check`=16 (the layers.1 q `lora_B`; it indexes all params, frozen included).
  `dsreg.ADAPTER_P_BY_MODEL["llama"]={16: 819200}`, `_NO_HEAD_BIAS`. `HIDDEN_SIZE` 960 is SmolLM2-specific, so a
  1B Llama needs its own row.
- `learning_rate` is inert under `trust_ratio`. The sim profile is still DistilBERT's.
