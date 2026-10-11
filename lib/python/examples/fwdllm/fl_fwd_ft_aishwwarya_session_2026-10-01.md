# Aishwwarya — cheatsheet audit, agnews trend check and simplified controller (2026-10-01/02)

Session notes for [fl_fwd_ft_pipeline_cheatsheet.md](fl_fwd_ft_pipeline_cheatsheet.md). The four-doc corpus remains
authoritative. Nothing here is committed yet.

## 1. Cheatsheet changes

- **"not in code" marks.** 13 variables are marked `*(not in code → code_name)*`. `D`, `Λ` and `A` have no code
  variable at all.
- **Flowchart variables added.** 16 symbols that were missing (`I`, `pool`, `N_req`, `G`, `ρ*_t`, …) are now in the
  table, plus a new row 20, *Commit indexing*.
- **New columns.** *Impact on training* (🔴 accuracy · 🟠 cost only · 🟢 marginal · ⚪ none) and *Simplify?*, which
  holds proposals, not corpus results.
- **New §3, Telemetry per concept.** Maps each row to the fields that track it.
- **Parity gaps found** (not yet fixed): code defaults ≠ shipped values, `B` fed from `_last_rho`, stall progress
  without a chance level, an extra GL condition. Listed in [AI context §4](fl_fwd_ft_pipeline_context.md).

## 2. Code changes

**Telemetry.** Every record is guarded so it can never fault training.

| Record | Content |
|---|---|
| `run_meta` | every knob, `p_trainable`, `‖θ_tr‖` at init. The trainers also log `probe_combine`, `P` and FD `h` |
| `server_update` | adds `commit_count`, `g_norm`, `step_skipped`, `rho_max`, `pool_size`, `cos_theory`, `aim_d`, `progress_lambda`. Needs `--server-update-audit` |
| `bmax_probe` | accuracy per tested Φ, `phi_knee`, `B_max` before and after |
| `sat_state` | smoothed, best, `gl`, `progress`, streak counts. Needs `--saturation-stop` |
| `agg_round` | `agg_weight` (ω_k) and `align_cos` per upload |

**Flags for the simplified model** (§5):

| Flag | Effect |
|---|---|
| `--commit-gate fixed --pool-target N` | commit at exactly N uploads |
| `--rho-schedule pool` | constant ρ = s·√(P·N/p) |
| `--agg-rate-type uniform` | ω = 1 |
| `--no-sat-decay` | GL decay trigger off |
| `--budget-stop-frac 0` | budget stop off. Before this change, 0 fell back to 0.95 |
| `--fd-scale-invariant` | exports `FWDLLM_FD_SCALE_INVARIANT=1` |
| `--dynamic-kc on\|off` | the selector's K/C controller |

With `--commit-gate fixed`, also pass `--var-stopping-policy off`, or the plateau rule can force a commit early.

**Files:** `run_sequential.sh`, `FedSgdAggregator.py`, `fwdllm_aggregator.py`, `saturation_stop.py`, `events.py`,
`fwdgrad_utils.py`, `tc_transformer_trainer_distribute.py`.

**Tests:**
- `tests/telemetry/test_controller_events.py` and `tests/mode/test_simplified_controller_flags.py` pass.
- `tests/mode` + `tests/telemetry`: 1,027 pass. 5 fail, and the same 5 failed before this session.

## 3. Runs

Both runs: conda env `test_fwdllm`, run from `examples/fwdllm/`, agnews, 100 trainers, C=30, K=10, P=10 (`mean`),
`s`=1.5, rf=16 (`p`=450,340), GPUs 0, 2, 4, 5 and 6, a 3 h wall ceiling, and `EXPT_GPU_ALLOW_PEER=1`.

| | Baseline (full controller) | Simplified |
|---|---|---|
| Started | 2026-10-01 12:07 | 2026-10-02 00:07 (detached with `setsid nohup`) |
| Controller | `n_target` gate, `landing` (`T_res` 300), `B_max` probe every 150 (`anchor`), `grad_aware` ω, budget stop 0.95, GL decay | `fixed` gate (N=50), `pool` (ρ=0.050), ω=1, no probe, no budget stop, stall rule 0.003 + Φ rail, no GL |
| Audits | cos ground truth every 50 commits, retention every 25 | none |
| Run directory | `experiments/run_20261001_120715_fluxtune_agnews_n100_smoke_syn_0_sim/` | `experiments/run_20261002_000757_fluxtune_agnews_n100_smoke_syn_0_sim/` |
| Logs (in the run directory) | `01_10_26_12_07_…_{aggregator,trainers}.log` | `02_10_26_00_08_…_{aggregator,trainers}.log` |
| Telemetry | `<run dir>/telemetry/aggregator_*.jsonl`, `trainer_*.jsonl` | same |
| Launcher logs | `expt_scripts/smoke_logs/20261001_120712/` | `expt_scripts/smoke_logs/20261002_000754/` |
| Console output | `agnews_full_20261001_1206.log` | `agnews_simple_20261002_0007.log` |

**Launch gotchas:**
- Pass `--max-runtime-s 48000 --sim-wall-ceiling-h 3`, or the preflight blocks the launch.
- The GPU check reads the busiest GPU on the node, so set `EXPT_GPU_ALLOW_PEER=1` inline on the command.
- A backward-pass cos audit costs about 85 s per fire.
- Background shells are killed after 2 h, so detach runs that last longer.

**Comparing the runs:** `compare_runs.py <baseline> <simplified>` (session scratchpad). It compares accuracy at the same
Φ, the same commit and the same wall-clock hour, plus time to reach a given accuracy. The baseline spent about 50 min
on probes and audits, so the wall-clock comparison favours the simplified arm. The Φ and commit comparisons isolate
the controllers.

## 4. Agnews baseline: observed against expected

855 commits, stopped by the wall ceiling at Φ 2.58. Peak smoothed accuracy **0.873** (best single eval 0.8746);
target 0.88.

Read with the telemetry offsets in AI context §4 corrected (pool = `10·(I+1)`); once corrected, the gate, the step
and Φ are exact.

**Verdict key:** ✅ matches the corpus · ⚠️ partly · ❌ contradicts · ❓ this run can't tell.

| # | Concept | Expected | Observed | Verdict | Impact (expected → observed) |
|---|---|---|---|---|---|
| 1 | Forward gradient | ‖u_k‖ stable | 688 → 1030; 3.6× spread within a round | ✅ | 🔴 → 🔴 |
| 2 | FD spacing | ε ∝ 1/‖θ‖ | ε 0.50 → 0.195 | ✅ | 🟢 → 🟢 |
| 3 | Probe combination | `mean` on both sides | `mean`, P=10 everywhere | ✅ | 🔴 → 🔴 |
| 4 | Pooling / aim | `n_eff`=pool; aim off theory | `n_eff/pool` 0.98–1.03; cos 12× below theory, ~40% lower late | ✅ | 🔴 → 🔴 |
| 5 | Cohort | pipeline full | `in_flight`=30; 22–26 rounds/min | ✅ | 🟠 → 🟠 |
| 6 | Safety rule | ρ/cos ≫ s | 14–20 vs 1.5 | ✅ | 🟠 → 🟠 |
| 7 | Commit gate | `n_req` exact, mostly `natural` | exact; pool 100 → 30; 0 `cap` (corpus: 100% `cap`) | ✅ | 🔴 → 🟠 |
| 8 | ρ_max | never exceeded | max ρ* 0.068 < 0.0999 | ✅ | 🟢 → ⚪ |
| 9 | ω weights | 0.70–0.87, median 0.82 | median 0.767; 1.11× within a round | ⚠️ | 🟢 → ⚪ |
| 10 | Legacy var gate | no effect | no effect | ✅ | ⚪ → ⚪ |
| 11 | Trust-ratio step | ρ = ρ* | exact; 0 skipped | ✅ | 🔴 → 🔴 |
| 12 | Budget law | Φ predicts ‖θ‖ | 0.04% median miss | ✅ | ⚪ → ⚪ |
| 13 | Progress law | accuracy rises with Λ | 0.83 by Φ 1.45, then a slow creep | ✅ / ❓ | ⚪ → ⚪ |
| 14 | `B_max` probe | knee ≈ 1.25, flat headroom | knee 1.25–1.73, set by one noisy point at φ=1.5 | ⚠️ | 🟢 → 🔴 noise sets ρ* |
| 15 | Law C anneal | flat ρ* | sawtooth 0.032–0.068 | ❌ | 🔴 → 🔴 |
| 16 | Stall stop | progress → 0.003 | reached at commit 833; streak 7/20 | ✅ | 🟠 → ⚪ |
| 17 | GL decay | silent on a plateau | 0 breaches | ✅ | 🟢 → ⚪ |
| 18 | Φ rail | peak at Φ 2.8–3.0 | not reached (Φ 2.58) | ❓ | 🔴 → ⚪ |
| 19 | Budget stop | < 0.95 | max 0.86 | ⚠️ | ⚪ → ⚪ |
| 20 | Commit indexing | +1 per commit | +1 every commit | ✅ | ⚪ → ⚪ |

**Totals:** 13 ✅, 4 ⚠️, 1 ❌, 2 ❓. The only lever that changed during the run was ρ*_t (rows 14 + 15), and probe
noise drove it.

## 5. Simplified model (proposal)

Constant ρ = s·√(P·N/p), commit at N uploads, ω = 1, stop on Φ ≥ 3 or the stall rule. Flowchart: cheatsheet §2;
flags: §2 above.

N=50 matches the baseline's mean pool (49.5) and gives ρ=0.050, inside the observed range of 0.032–0.068.

**Kept:** `P`, `mean`, FD spacing, `p`, trust-ratio, `N`, `s`, Φ rail, stall rule, `C`, `K`.

**Risk:** constant ρ has no late anneal.

### What the removed variables did on agnews

| Group | Variables | On agnews |
|---|---|---|
| **Never fired (6)** | `max_iter` cap, `ρ_max`, var plateau force-commit, legacy var gate, budget stop, GL decay | none triggered |
| **Never read (3)** | `rho_star`, `rho_exp` (not used by `landing`), `learning_rate` (not used by trust-ratio) | inert |
| **Switched off (3)** | `dynamic_kc`, momentum, weight decay | never exercised |
| **Fired, no effect (3)** | align gate (×0.998), ω staleness weights (1.11×), cos audit | inert or audit-only |
| **Fired and mattered (3)** | `B_max` probe, law C (`T_res`, `b_max_prior`), `n_req`/`n_eff` gate | set ρ* and the pool, mostly from noise |

**"Never fired on agnews" does not mean "never fires".** Validate every removal on yahoo, yelp-p and a second model
before treating it as safe. The corpus already shows the `max_iter` cap binding on 100% of commits in G-1 (`rf`=64),
and the Φ rail firing on yelp-p.

### Why GL never fired

GL is a decay detector: it fires after 20 straight evals that are all past warm-up (commit 450), more than 0.5% below
the best, and no higher than 150 commits earlier. Agnews never declined. The 7 evals with `gl > 0.005` all came
before commit 105, and the largest `gl` after warm-up was 0.0044. GL worked; there was nothing to catch.

- **Value:** a free safety net against collapse (runaway, training past the useful Φ, overfitting datasets).
- **Weakness:** the "no higher than 150 commits ago" test ignores a quick peak-then-drop (for example 0.80 → 0.87 →
  0.85), so GL is late on a sharp collapse.
- **Recommendation:** the stall rule also catches decays, but keep GL **on** for yahoo and yelp-p until it is validated.

### What the budget stop is

`B = ½·Σ ln(1+ρ²)` is the total weight growth, with Φ = e^B. `B_max` is the growth the model is assumed to tolerate:
a prior of `ln 2`, re-measured by the probe. Under `landing`, training stops at `B ≥ 0.95·B_max`. It never fired on
agnews (max 0.86) for two reasons:
- law C approaches `B_max` only gradually, at a rate set by `T_res`=300;
- the probe moved `B_max` up 3 times out of 5.

It watches the same `B` as the Φ rail; only the threshold differs (measured instead of fixed). With `B_max = ln 3` it
just repeats the Φ rail.

## 6. A/B result (agnews): **pass**

The simplified run stopped on the **Φ rail at commit 881** (Φ=3.001, 01:52), after 1.74 h of training. The baseline
ran out the 3 h wall ceiling.

| | Baseline | Simplified |
|---|---|---|
| Peak smoothed accuracy | 0.8728 (commit 797, Φ 2.49) | **0.8726** (commit 791, Φ 2.68) |
| Best single eval | 0.8746 | 0.8755 |
| Training wall time | 3.0 h (wall ceiling) | **1.74 h** (Φ rail) |
| Time to 0.80 / 0.84 / 0.86 / 0.87 | 1.19 / 1.55 / 2.06 / 2.81 h | **0.59 / 0.73 / 1.09 / 1.50 h** |
| Commits per hour | 285 | 508 |
| Probes / audits | 5 probes (1,481 s), 18 cos audits | none |

- **Accuracy:** the same peak (−0.0002, inside the ±0.005 pass band).
- **At the same commit:** behind early (0.558 vs 0.670 at commit 100, since its ρ starts at 0.050 against 0.068).
  Level from commit 450 onward.
- **At the same Φ:** 0.002–0.006 lower, so constant ρ needs a little more Φ for the same accuracy.
- **Wall time:** reached 0.87 about 47% sooner. The baseline's ~50 min of probes and audits explain only part of the
  gap; the rest is not yet explained.
- **End of run:** a plateau from commit 761 to 851, then a small dip in the last 20 commits (smoothed 0.8726 → 0.866
  at Φ 2.93–3.0). `gl` reached 0.0067. GL was off in this arm and the stall streak only got to 7, so the Φ rail is
  what stopped the run. With no anneal, Φ=3 overshoots the peak (Φ 2.68) slightly.

**Not yet tested:** other datasets, and whether a rail at about 2.7, or GL turned on, would stop at the peak instead
of after the dip.

## 7. Open items

- Repeat the §6 A/B on yahoo / yelp-p (with GL on), so the §5 removals are tested on other datasets.
- Explain the extra wall-time gap beyond the audits, by comparing the two runs' `step_timing`.
- Constant ρ overshoots the peak by Φ ≈ 0.3. Try a rail at about 2.7, or one step-down (ρ/√2 at Φ=2).
  Preferred: anneal with ρ_t = ρ0/Φ_t (§8).
- Fix the telemetry offsets and the misleading log lines (`[Variance=GOOD]`, `[CommitGate] n_target` under `fixed`).
- Decide on the parity gaps in §1.
- Extend `analyze_run.py` for `bmax_probe` / `sat_state`. Move `compare_runs.py` into `expt_scripts/`.
- Commit following CLAUDE.md: tighten new comments, keep the diff minimal, short message.

## 8. Follow-up (2026-10-07): is constant ρ a good idea?

Re-read of both runs' `server_update` and `agg_eval` telemetry.

**Findings**
- **‖θ_tr‖ growth is set by ρ alone.** |cos(θ_tr, Δθ_tr)| < 0.0006 on every commit in both arms, so each step is
  sideways and ‖θ_t‖² = ‖θ_{t−1}‖²·(1+ρ²). With a ground-truth gradient cos of ≈ 0.003, that growth is almost all
  noise.
- **Early, a larger ρ is faster per commit; late, it isn't:**

| Accuracy band | Baseline commits (mean ρ) | Simplified commits (ρ 0.050) |
|---|---|---|
| 0.70 → 0.80 | 97 (0.046) | 70 |
| 0.80 → 0.84 | 129 (0.040) | 76 |
| 0.84 → 0.86 | 176 (0.046) | 194 |
| 0.86 → 0.87 | 234 (0.047) | 214 |

- So a large late ρ mostly adds Φ (noise), which fits the dip before the Φ = 3 rail.
- **Caveat:** one run per arm and ±0.005 eval noise. The baseline's ρ jumps at its probes are too noisy to read
  either way.

**Proposal** (ρ_t = ρ0/Φ_t, constant absolute step), its settings and the alternatives: cheatsheet §6. Code hook:
AI context §8.

## 9. Follow-up (2026-10-09): which model next, and does FluxTune scale to ~1B?

Extrapolated from buildplan §5.11 (roberta-large smoke `145932`). Nothing here is measured on a 1B model.

**Next model: SmolLM2-360M + LoRA r=8 (q,v).**
- Llama architecture, so it exercises the decoder path that LLaMA2-7B / Mistral-7B (G1) will use: pad token,
  last-token pooling, causal mask, LoRA placement. `adapters` 1.3.0 supports `llama` and `mistral`, not Qwen.
- Same size as roberta-large (355M), so its memory and trainers/GPU numbers carry over. `p` ≈ 0.8M sits between
  DistilBERT (450k) and roberta-large (4.23M): a third point for N5c.

**Scaling rule: commits grow with `p` (trainable), not model size.**
- `ρ_max ∝ 1/√p`, so per-commit progress `½ln(1+ρ²) ∝ 1/p` and commits to Φ* = ln(2.9)/(½ln(1+ρ_max²)).
  Reproduces DistilBERT ≈ 214 and roberta ≈ 2,000.
- Model size sets the time per trip: each perturbation is 2 forward passes, 10 perturbations per batch.

| Setup | `p` | Commits to Φ* | Fwd cost vs roberta | Wall time |
|---|---|---|---|---|
| DistilBERT 66M + adapters | 450k | 214 | ~0.2× | ~3 h (measured) |
| roberta-large 355M + adapters | 4.23M | ~2,000 | 1× | ~50 h (measured rate) |
| **1B Llama + LoRA r=8** | ~0.85–1.1M | ~450–550 | ~3× | **~25–55 h, central ~40 h** |
| 1B full fine-tune | 1.1B | ~500,000 | ~3× | infeasible |
| 7B + LoRA r=8 | ~4.2M | ~2,000 | ~20× | ~1,000 h, infeasible as-is |

1B estimate: 50 h ÷ 3.7 (fewer commits) × 3 (costlier passes) ≈ 40 h. The range is trips/commit, 10 to 20
(roberta stayed pinned at `ρ_max`, so 20).

**Does it scale?**
- ✅ Memory stays flat in depth and in P (no backward pass). The best-of-10 JVP gain doesn't depend on `p`.
- ❌ Gradient cos ∝ 1/√p. Keep `p` ≈ 1M; raising LoRA rank or adding target matrices costs time in proportion.
- ✅ *(resolved for 360M, §11)* bf16 precision in the JVP: as accurate as fp32. fp16 is what fails (NaN).
- ⚠ `FWDLLM_FD_SCALE_INVARIANT` ON makes the chord drift as 1/√p (buildplan row F). At 360M + LoRA it doesn't
  (0.511, §11); re-check on a 1B model.
- ❓ Whether Φ* holds at a new `p` (N5c) is still open. A 1B run is that test, so let it reach its own stop.

**Practical limits**
- **Memory:** ~4.5 GB per trainer (fp16 weights + tangent) before activations, so ~6–8 trainers per A40 and
  ~13–16 GPUs for 100 trainers. Fix: co-located trainers share one frozen backbone and keep separate LoRA weights
  (new code).
- **Stragglers:** ~3× roberta's per-batch compute. Re-profile `sim_charge_profiles` or sim timing won't match real.

**Next:** superseded by §10, which smoke-tests the 360M model first. For a 1B run: wall ≈ 500 ÷ commits/h; `s` = 2.9
cuts it ~3.7× (~11 h) but under-trains by about half (§5.12), so use it to check the controller, not accuracy.

## 10. Follow-up (2026-10-10): trying a different model: pick and time estimate

Extrapolation from §9 and buildplan §5.11, not measured. Current baselines: DistilBERT 66M total / 450k trainable;
roberta-large 355M / 4.23M.

**Pick: SmolLM2-360M + LoRA r=8 (q,v) first, ~1B Llama + LoRA r=8 as stretch.** Avoid 7B (~1,000 h) and any full
fine-tune at ≥1B (~500k commits). Keep `p` ≈ 1M. The 1B checkpoint is unchosen; Llama-3.2-1B and TinyLlama-1.1B
both fit, and neither is named in the docs.

**360M time estimate: ~5–10 h, central ~9 h.**
- Anchor: roberta-large ≈ 2,000 commits ≈ 50 h (rate from smoke `145932`, never run to convergence).
- Commits ∝ `p`: 4.23M / 0.8M ≈ 5.3× fewer, ≈ 380 commits. Forward cost ≈ 1× (355M vs 360M).
- 50 h ÷ 5.3 ≈ 9.5 h at 20 trips/commit; ~5 h at 10.
- An earlier ~15 h figure was wrong: it reused the 1B row's ÷3.7, which assumes `p` ≈ 1.1M.
- Floor: per-commit fixed overhead (aggregation, probe, sim charging) doesn't shrink with `p`, so expect above the low end.

**Smoke recipe:** superseded by §11.

## 11. Follow-up (2026-10-10): SmolLM2-360M ported; smoke ready, not yet run

Everything about this port (code changes, measurements, runbook, pass table) is in
[fl_fwd_ft_smollm2_smoke_plan.md](fl_fwd_ft_smollm2_smoke_plan.md). Findings that change earlier sections:

- `p` = **823,040**: the third `p` point for N5c.
- Chord with the flag ON = **0.511**, not §10's estimated 0.38. LoRA's ‖θ_tr‖/√p is 0.0145, not the adapters'
  0.0197, because `lora_B` = 0. Keep `FWDLLM_FD_SCALE_INVARIANT=1`.
- **fp16 autocast gives a NaN loss**, even unperturbed. Llama now runs in bf16, which is as accurate as fp32 here
  and 2.4× faster.
- `tests/mode` + `tests/telemetry`: 1,027 pass, 5 fail. Whether these are the same 5 as §2 was not re-checked.
