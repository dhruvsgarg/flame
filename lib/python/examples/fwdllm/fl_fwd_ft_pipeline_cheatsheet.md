# FluxTune pipeline — quick reference

> Derived summary of the four-doc corpus (`fl_fwd_ft_solution.md` / `practice` / `buildplan` / `writeup`).
> On any disagreement the corpus wins — fix this file, not the corpus. Covers only the **live loop**;
> audit-only instruments (`cos(G,g)` probe, `D`, `Λ`/`A` tracking) are omitted.
> Session notes on this sheet's code-parity marks, added columns and telemetry:
> [fl_fwd_ft_aishwwarya_session_2026-10-01.md](fl_fwd_ft_aishwwarya_session_2026-10-01.md).

**Operator supplies only:** model + PEFT scheme, rank `rf` (⇒ `p`), compute budget. Everything else is
derived, sensed, or still-being-validated (`s`, stop thresholds).

---

## 1. One commit

```
                        ┌──────────────────────────────────┐
                        │  START COMMIT t on data bin b    │
                        │  pool = ∅,  I = 0                │
                        └────────────────┬─────────────────┘
                                         ▼
 ┌─────────────────────────── POOLING LOOP (one round per pass) ───────────────────────────┐
 │                                                                                          │
 │   ┌────────────────────────────────────────────────┐                                     │
 │   │ SERVER: keep C=30 trainers in flight on bin b  │                                     │
 │   └──────────────────────┬─────────────────────────┘                                     │
 │                          ▼                                                               │
 │   ┌────────────────────────────────────────────────┐                                     │
 │   │ EACH TRAINER (forward passes only):            │                                     │
 │   │  draw P=10 Gaussian probes v_i                 │                                     │
 │   │  d_i = [L(θ+h·v_i) − L(θ−h·v_i)] / 2h          │                                     │
 │   │  u_k = (1/P)·Σ d_i·v_i      (mean, G_rule=10)  │                                     │
 │   │  upload u_k                                    │                                     │
 │   └──────────────────────┬─────────────────────────┘                                     │
 │                          ▼                                                               │
 │   ┌────────────────────────────────────────────────┐                                     │
 │   │ SERVER: wait for K=10 uploads (async;          │                                     │
 │   │ stragglers roll into later rounds)             │                                     │
 │   │ add them to pool,  I += 1                      │                                     │
 │   └──────────────────────┬─────────────────────────┘                                     │
 │                          ▼                                                               │
 │   ┌────────────────────────────────────────────────┐                                     │
 │   │ var of pool → n_eff = 2·mean‖u_k‖² / (m·var)   │                                     │
 │   │ (disagreeing uploads lower n_eff)              │                                     │
 │   └──────────────────────┬─────────────────────────┘                                     │
 │                          ▼                                                               │
 │   ┌────────────────────────────────────────────────┐                                     │
 │   │ N_req = p·(rho_star/s)² / G_rule               │                                     │
 │   │       = 118,348·(0.06/1.5)²/10 ≈ 19            │                                     │
 │   └──────────────────────┬─────────────────────────┘                                     │
 │                          ▼                                                               │
 │                ◇ n_eff ≥ N_req ? ◇ ──yes──────────────────────────────┐                  │
 │                          │ no                                         │                  │
 │                          ▼                                            │                  │
 │         ◇ I ≥ max_iterations_per_data_id (20) ? ◇ ──yes (forced)──────┤                  │
 │                          │ no                                         │                  │
 │                          └──────── back to top of loop ◄──            │                  │
 └───────────────────────────────────────────────────────────────────────┼──────────────────┘
                                                                         ▼
                        ┌──────────────────────────────────────────────────┐
                        │ AGGREGATE:  G = Σ ω_k·u_k / Σ ω_k                │
                        │ (ω_k: grad_aware weights, 0.70–0.87)             │
                        └────────────────────────┬─────────────────────────┘
                                                 ▼
                        ┌──────────────────────────────────────────────────┐
                        │ STEP SIZE:  ρ*_t = _rho_star_now()               │
                        │   rm:      0.06 · t^(−0.25)                      │
                        │   [landing] min(ρ_max, √(2(B_max−B)/T_res))      │
                        └────────────────────────┬─────────────────────────┘
                                                 ▼
                        ┌──────────────────────────────────────────────────┐
                        │ TRUST-RATIO STEP (trainable slice only):         │
                        │   Δθ_tr = −ρ*_t · ‖θ_tr‖ · G/‖G‖                 │
                        │   (skip if ‖G‖ = 0)                              │
                        └────────────────────────┬─────────────────────────┘
                                                 ▼
                        ┌──────────────────────────────────────────────────┐
                        │ BOOKKEEPING:                                     │
                        │   commit_count += 1                              │
                        │   _last_rho = ρ*_t                               │
                        │   B += ½·ln(1 + ρ*_t²)      (Φ = e^B)            │
                        └────────────────────────┬─────────────────────────┘
                                                 ▼
                          ◇ [landing] commit_count % 150 == 0 ? ◇
                             │ yes                           │ no
                             ▼                               │
          ┌─────────────────────────────────────────┐        │
          │ B_max RE-SENSE (forward only):          │        │
          │  for φ in {1.5 … 4.0}: add noise so     │        │
          │  ‖θ_tr‖ grows ×φ, read held-out acc     │        │
          │  φ_knee = where acc falls to 0.5 norm   │        │
          │  B_max = B + ln φ_knee   (anchor)       │        │
          │  restore θ                              │        │
          └────────────────────┬────────────────────┘        │
                               └──────────────┬──────────────┘
                                              ▼
                   ◇ STOP?  Φ-rail ≈ 3.0 │ accuracy saturated │ [landing] B ≥ 0.95·B_max ◇
                             │ yes                                   │ no
                             ▼                                       ▼
                     ┌───────────────┐             ┌──────────────────────────────────┐
                     │ END TRAINING  │             │ move to bin b+1, reset pool      │
                     └───────────────┘             │ → START COMMIT t+1               │
                                                   └──────────────────────────────────┘
```

**Gotchas not visible in the chart**
- Gate ≡ `cos ≥ ρ/s` (since `cos ≈ √(G_rule·N/p)`): a longer step needs a better-aimed direction. With
  `gate_rho_ref=setpoint`, `N_req` is constant ⇒ `I`=2 every bin.
- `probe_combine` code default is `select` — set `mean` on trainer **and** aggregator; the aggregator
  derives `G_rule` from its own copy, unchecked.
- `rho_schedule`: `const` (code default) · `rm` (shipped) · `landing` (law C, not yet default).
- The `B_max` probe is falsified (it measures the trajectory, not the model): it pins `Φ_knee`≈1.25, so
  `B_max` recedes as fast as `B` grows and law C degenerates to constant ρ.

| ρ term | meaning | used for |
|---|---|---|
| `rho_star` | config setpoint (0.06) | `rm`/`const` base; `N_req` under `setpoint` |
| `ρ*_t` | step allowed at commit t | step length, `B` |
| `_last_rho` | step taken, `‖Δθ_tr‖/‖θ_tr‖` | gate under `raw_sgd` only |
| `ρ_max` | largest ρ whose `N_req` fits in `max_iter` | caps `ρ*_t` under `landing` |

---

## 2. Concepts: values, status, generality

Evidence is **one model** (DistilBERT + adapters) on three datasets (G-1); the only real `p` change
(`rf`=64) broke law C. ✅ scale-free by construction · 🟡 form carries, one empirical constant ·
❌ moves with model/dataset/optimizer or has absolute units.
Impact on training: 🔴 changes accuracy or convergence · 🟠 changes cost/wall clock only · 🟢 marginal ·
⚪ none in the live loop.
Simplify?: proposed here, not yet in the corpus · ✂️ = term can be removed or merged. Net: rows 7 + 15 + 18
collapse to fixed `I`, `ρ = s·√(P·K·I/p)`, stop at `T = 2·ln Φ_cap/ln(1+ρ²)` or stall.

| # | Concept | Variables (shipped) | Intuition | Current → proposed | Status | Impact on training | Generalises? | Simplify? (proposed) |
|---|---|---|---|---|---|---|---|---|
| 1 | Forward-gradient estimate | `P`=10, `v_i`, `d_i`, `u_k`, `L`, `jvp_eval_mode` | `E[d·v]=g`; forward-only ⇒ flat memory, inference HW | central FD, mean of P → unchanged | Current | 🔴 High — it *is* the gradient; `cos ∝ √P` | ✅ `P` trades compute for noise | No — already minimal: `u = (1/P)·Σ dᵢ·vᵢ` |
| 2 | FD spacing | `h`=0.01, `FWDLLM_FD_SCALE_INVARIANT` | nudge ≪ weights | `h·√p` const → `h = c_h·‖θ‖/√p` | Current | 🟢 Low — only fp16 round-off at bad `h` | 🟡 fp16 round-off depends on precision | ✂️ One relative constant: `h = ε·‖θ_tr‖/√p` (so `‖h·v‖ = ε·‖θ_tr‖`, `ε`≈0.5 at init); drops the env flag |
| 3 | Probe combination | `probe_combine`=`mean`, `G_rule`=10 | select: `b²/a`=1 (no gain); mean: `1/P`, 3.35× faster | select (v1, `G_rule`≈2.99) → mean | Current | 🔴 High — `mean` = 3.35× less time-to-accuracy; stability-neutral | ✅ rule property, not data | ✂️ Delete `select`; `G_rule ≡ P` — removes the trainer/aggregator mismatch |
| 4 | Pooling / aim | `N=K·I` *(not in code → `len(grad_for_var_check_list)`)*, `I` *(not in code → `iteration_per_data_id`)*, `pool` *(not in code → `grad_for_var_check_list`)*, `var`, `m`, `n_eff`, `D` *(not in code)*, `p` | signal adds linearly, noise in quadrature | `cos = D·√(G_rule·N/p)` → `c·√(n/p)` | Current | 🔴 High — `cos ≈ √(N·P/p)` ≈ 0.07: sets each step's signal | ❌ `D` 20× off theory, drifts 2–3× | ✂️ Fold `D`, `G_rule` into one fitted `c`: `cos = c·√(P·N/p)`; drop `n_eff` (reads 1.00·N, blind to direction) |
| 5 | Cohort | `C`=30 *(not in code → `concurrency`)*, `K`=10 *(not in code → `aggGoal`)*, `train_batch_size`=8, `dynamic_kc` (off) | async, no barrier | hand-set → fix K, hill-climb C | Current | 🟠 Medium — wall clock only; throughput ∝ `C`, flat in `K` | 🟡 deployment property | ✂️ One knob: fix `N = K·I`, hill-climb `C` on commits/s |
| 6 | Safety criterion | `s`=1.5 *(not in code → `gate_safety_s`)* | don't outstep your aim | `ρ ≤ s·cos` → `ρ ≤ c·√(n/p)` | Current | 🟠 Medium — efficiency knob, not safety; sole lever on `Λ` per budget | 🟡 `s` empirical | ✂️ `s` and `ρ` only appear as `ρ/s` — keep one (row 7 derives `ρ` from `s`) |
| 7 | Commit gate | `commit_gate`=`n_target`, `max_iter`=20 *(not in code → `max_iterations_per_data_id`)*, `gate_rho_ref`, `N_req`, `_last_rho` | bigger stake ⇒ more evidence (`N ∝ ρ²`) | `N_req` → fix `I`, derive ρ | Current | 🔴 High — sets pool per commit (`N ∝ ρ²`); `max_iter` binds on 100% of commits in G-1 | 🟡 `1/√p` adapts; `c` unchecked | ✂️ Invert it: fix `I`, `ρ = s·√(P·K·I/p)` = 1.5·√(200/118,348) ≈ 0.062 (= shipped 0.06); drops pool loop, `var`, `max_iter` |
| 8 | Reachability cap | `ρ_max` ≈0.068–0.10 | largest affordable step | `s·√(max_iter·K·G_rule/p)` → fold into 7 | `landing` only | 🟢 Low — `landing` only | 🟡 | ✂️ Row 7 at `I = max_iter`; gone once `I` is fixed |
| 9 | Aggregation weights | `ω`=0.70–0.87 *(not in code → `_grad_aware_rate`)*, `scale`=0.4, `a_exp`=0.25, `b_exp`=0.1, `align_gate` | fresh/useful/agreeing; measured near-inert | FeLiX×align → `ω=1` or `max(0,cos)` | Landed, inert | 🟢 Low — measured near-inert (median `ω` 0.82) | ❌ hand-set; proposal ✅ | ✂️ `ω = 1`, and trust ratio keeps only `G/‖G‖`, so `G = Σ u_k` (no normaliser) |
| 10 | Legacy var gate | `var_threshold`=0.3 | commit when variance low | → remove | v1 only | ⚪ None live — v1's accidental stabiliser | ❌ carries `‖g‖²` units | ✂️ Delete — superseded by row 7 |
| 11 | Trust-ratio step | `server_step_rule`=`trust_ratio`, `G` *(not in code → `weighted_gradient_sum`)*, `θ_tr` *(not in code → trainable params; `‖θ_tr‖` = `_tn`)*, `Δθ_tr` *(not in code → `_update`)* | direction from G, length from θ; kills runaway and α | unchanged | Current | 🔴 High — removed the runaway collapse; makes `ρ` a set knob | ✅ | No — already minimal: `Δθ_tr = −ρ·‖θ_tr‖·G/‖G‖` |
| 12 | Budget law | `B`, `Φ` | steps ⟂ θ ⇒ inflation; `1/Φ` retention | `½Σln(1+ρ²)` → `½Σρ²` | Exact | ⚪ None — accounting identity; read by stops/anneal | ✅ | ✂️ At constant `ρ`, closed form: `Φ_t = (1+ρ²)^{t/2}` — no accumulator |
| 13 | Progress law | `Λ`, `A` *(not in code)* | accuracy tracks Λ; `Λ = 2B/s` | → track `Σρ²` only | Current | ⚪ None — audit only | 🟡 `A` inherits row 4 | ✂️ `Λ = 2B/s` is `B` renamed — drop, track `B` (or `t`) |
| 14 | `B_max` | prior `ln 2`, every 150, `anchor`, `φ` ∈ 1.5…4.0, `φ_knee` @ 0.5 | tolerated inflation | probe → const `ln Φ_cap` | Sensor falsified | 🟢 Low — sensor falsified; stop moved off it | ❌ least general | ✂️ Delete the probe: `B_max = ln Φ_cap` (= ln 3, the rail) |
| 15 | Law C anneal | `rho_star`=0.06, `rho_exp`=0.25, `T_res`=300, `ρ*_t` = `_rho_star_now()` | spend remaining budget at a rate | law C → `ρ_0·e^{−t/τ}` or const | Degenerates; row A open | 🔴 High — `ρ` is the step length; hand-set 0.06 didn't transfer | ❌ `T_res` tied to one `p` | ✂️ Constant `ρ` from row 7 — drops `rho_star`, `rho_exp`, `T_res`, law C |
| 16 | Stop: stall | `P4_SAT_STALL_FRAC`=0.003 *(launcher env only → `sat_stall_frac`)*, patience 20 | stop when progress stops | `(m_t−m_{t−h})/(m_t−chance)` → sole rule | Flagged, default off | 🟠 Medium — compute, not accuracy: ends plateaus | 🟡 cadence-dependent | ✂️ Sole accuracy stop: `(m_t−m_{t−h})/(m_t−chance) ≤ 0.003` for 20 evals |
| 17 | Stop: decay (GL) | `thr`=0.005 *(not in code → `SAT_GL_THRESHOLD`)*, patience 20 | stop when below best | → delete | Shipped; misses plateaus | 🟢 Low — misses plateaus; rarely fires | 🟡 | ✂️ Delete — a decay has progress `< 0`, so row 16 fires on it too |
| 18 | Stop: Φ rail | `phi_rail`=3.0 *(not in code → `phi_stop_threshold`)* | ~70° off pretrained | `e^B ≥ 3` | Shipped, fired on yelp-p | 🔴 High — sets the endpoint; every run peaks at `Φ`≈2.9 | ❌ biggest new-model risk | ✂️ At constant `ρ`, a commit cap: `T = 2·ln 3/ln(1+ρ²)` ≈ 611 at `ρ`=0.06 |
| 19 | Stop: budget | `f`=0.95 *(not in code → `budget_stop_frac`)* | stop at 95% of budget | → delete | Unreachable | ⚪ None — unreachable | ❌ | ✂️ Delete — duplicate of row 18 once `B_max = ln 3` |
| 20 | Commit indexing | `b` *(not in code → `data_id`)*, `t`, `commit_count` | one bin per commit; `t` drives the anneal and probe cadence | unchanged | Current | ⚪ None — bookkeeping | ✅ | ✂️ `b = t mod total_data_bins` — one counter |

**Other constants:** `p` = 450,340 (`rf`=16) / 118,348 (`rf`=64, shipped); `‖θ_tr‖`=13.35 at init;
`total_data_bins` 150 / 1,750 / 650 (agnews / yahoo / yelp-p); `server_momentum`=0 (refuted);
`server_weight_decay` off.

**On a new model:** retune `c` (≡ `s`) and `Φ_cap`; check the stall horizon in evals; carry the rest.
First test is G-1: a second model (roberta-large), where `Φ_knee` already shifted.

---

## 3. Telemetry per concept

Per-commit fields live in `server_update` (needs `server_update_audit: true`); `sat_state` needs
`saturation_stop: true`. Knobs are snapshotted once in `run_meta` (`scope` = `aggregator`,
`trainer_probe`, `trainer_fd`).

| # | Concept | Once (`run_meta`) | Throughout the run | Read impact as |
|---|---|---|---|---|
| 1 | Forward gradient | `perturbation_count`, `jvp_eval_mode` | `agg_round.grad_norm` (‖u_k‖) | ‖u_k‖ spread across trainers |
| 2 | FD spacing | `fd_h`, `fd_displacement`, `p` (`trainer_fd`) | `server_update.trainable_weight_norm` | `ε = fd_displacement/‖θ_tr‖` drift as ‖θ_tr‖ grows |
| 3 | Probe combination | `probe_combine`, `g_rule` (aggregator **and** `trainer_probe`) | — | the two `probe_combine` values must match |
| 4 | Pooling / aim | `p_trainable` | `pool_size`, `iteration_per_data_id`, `var_at_commit`, `var_dim`, `pool_mean_sq`, `n_eff`, `cos_theory`, `aim_d`* | `aim_d` = measured/theory cos |
| 5 | Cohort | `agg_goal`, `concurrency`, `dynamic_kc`, `train_batch_size` | `agg_round.in_flight`, `staleness`, `ts` | commits/s vs `C` |
| 6 | Safety criterion | `gate_safety_s` | `rho` vs `cos_ground_truth`* | `ρ/cos` vs `s` |
| 7 | Commit gate | `commit_gate`, `gate_rho_ref`, `max_iterations_per_data_id` | `n_req`, `n_eff`, `agg_round.commit_reason` | share of `cap` vs `natural` commits |
| 8 | Reachability cap | `rho_max` | `rho_max`, `rho_star` | `rho_star == rho_max` ⇒ capped |
| 9 | Aggregation weights | `agg_rate_conf` | `agg_round.agg_weight`, `align_cos`, `grad_aware_gated_total` | spread of `ω_k`; ≈const ⇒ inert |
| 10 | Legacy var gate | `var_threshold` | `agg_round.var`, `var_good_enough` | v1 only |
| 11 | Trust-ratio step | `server_step_rule`, `theta_tr_norm_init` | `g_norm`, `step_skipped`, `rho`, `trainable_delta_norm` | `rho` vs `rho_star`; skipped steps |
| 12 | Budget law | — | `budget_b`, `phi` | `phi` vs `trainable_weight_norm/theta_tr_norm_init` |
| 13 | Progress law | — | `progress_lambda` vs `agg_eval.test-accuracy` | slope of accuracy vs `Λ` |
| 14 | `B_max` | `b_max_prior`, `b_max_probe_every`, `b_max_probe_phis`, `b_max_policy` | `bmax_probe` (`phis`, `accs`, `phi_knee`, `b_max_before/after`), `budget_b_max` | `b_max_after − budget_b` shrinking ⇒ receding |
| 15 | Law C anneal | `rho_star`, `rho_schedule`, `rho_exp`, `t_res` | `rho_star`, `commit_count` | `ρ*_t` curve over `t` |
| 16 | Stop: stall | `sat_stall_frac` | `sat_state.progress`, `stalls` | `progress ≤ sat_stall_frac` streak |
| 17 | Stop: decay (GL) | `saturation_stop` | `sat_state.gl`, `breaches`, `smoothed`, `best` | `gl > 0.005` streak |
| 18 | Stop: Φ rail | `phi_stop`, `phi_stop_threshold` | `phi`, `stop_reason` | `phi` at peak accuracy |
| 19 | Stop: budget | `budget_stop_frac` | `budget_frac` | never reaches `f` ⇒ unreachable |
| 20 | Commit indexing | `total_data_bins` | `data_id`, `commit_count` | — |

\* `cos_ground_truth` (and so `aim_d`) needs the backward-pass audit `cos_ground_truth_audit`.
