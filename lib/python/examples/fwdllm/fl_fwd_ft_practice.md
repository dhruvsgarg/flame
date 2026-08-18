# Forward-gradient fine-tuning — implementation, evidence, and how to run it

> **This document owns every number a run produced**, plus the flags, arms, dead ends, instruments and
> launch procedure. **[P3](#p3--knob-ledger) (knobs) and [P4](#p4--arm-ledger) (arms) are the two
> ledgers, and they are the source of truth for evidence.**
>
> **Status and next steps are NOT here — they live in
> [fl_fwd_ft_buildplan.md §-1](fl_fwd_ft_buildplan.md), the one status board, updated in place.**
> [fl_fwd_ft_solution.md](fl_fwd_ft_solution.md) (`§0…§8`) owns the *why*; cited here as "model §x".

| you want… | go to |
|---|---|
| **status · what to do next · how to build it** | **[fl_fwd_ft_buildplan.md](fl_fwd_ft_buildplan.md)** — status board, queue, specs |
| **the generality claim, as a checklist** | **[buildplan §-0](fl_fwd_ft_buildplan.md)** |
| what to launch an arm with · what defaults to change | [P2](#p2--the-shipped-stack) · [P2.1](#p21-ship-checklist) |
| what a lever did, and its flag | [P3](#p3--knob-ledger) |
| what an arm scored | [P4](#p4--arm-ledger) |
| **before proposing anything** | [P6](#p6--dead-ends--do-not-retry) — append-only |
| before launching | [P9.1](#p91-preflight) · [P9.2](#p92-launcher-gotchas) |
| to reproduce a number | [P8](#p8--reproducing-any-number-from-logs) |

---

## P1 — The system under test

**MEASURED**, from `aggregator_config.json` + `expt_scripts/fluxtune_n10_smoke.yaml`:

```
population N_clients=100, alpha=1 (Dirichlet), agnews, DistilBERT-base + AdapterHub adapters
trainable p = 450,340  (0.67% of 67.4M; backbone frozen AND pre_classifier dropped)
train_batch_size = 8          -> one "data bin" = 8 samples, 150 bins per round
perturbation_count P = 10, h = 0.01, central finite difference, fp16/autocast, no_grad
probe selection: rank by |JVP|, coin-flip between top-2  (tc_transformer_trainer_distribute.py:481)
client selection: async_oort, c = 30, agg_goal K = 10, dynamic_kc.enabled = FALSE
commit gate: var_threshold = 0.3, var_stopping_policy = plateau (patience 3, rel_delta 0.15),
             max_iterations_per_data_id = 20
server step: theta <- theta - eta * (1/N_acc) * sum_k omega_k * d_k * v_k   (raw in-place SGD)
             eta = 0.01, constant to 4 digits for the whole run; no optimizer state of any kind
omega: grad_aware / base=new / align_gate=true / scale=0.4 [measured 0.702..0.865, median 0.818]
```

**Where the probe dimensions live.** `create_model` builds 1,040,932 trainable params, but the trainer's
`__init__` then runs `self.model.add_module("pre_classifier", nn.Sequential())`
(`tc_transformer_trainer_distribute.py:217`) — **replacing that layer before any probe is drawn** — and
`self.params` comes from `make_functional_with_buffers` *after* that:

```
                         create_model     PRODUCTION (post-:217)
pre_classifier              590,592          0        <- dropped, not frozen
adapters                    447,264    447,264
classifier                    3,076      3,076
p                         1,040,932    450,340       <- the number every cos uses
||theta_tr|| at init          20.33        13.33
h*||v|| = h*sqrt(p)          10.203        6.711
```

---

## P2 — The shipped stack

*What an **arm** is launched with. Evidence for every line is its [P3](#p3--knob-ledger) row.*

```
probe_combine        = mean
server_step_rule     = trust_ratio
rho_schedule / exp   = rm / 0.25                  # mechanism settled; the LANDING law is C-1, unbuilt
rho*                 = min(rho_max, sqrt(2*(B_max-B)/T_res))   # law C; T_res=300 FIXED, never
                                                   # decremented. ~0.068 at the ln2 prior (T5)
commit_gate          = n_target
gate_rho_ref         = annealed                   # if the objective is time-to-accuracy; G-2 confirmed
                                                    # 2026-08-15 (P4.5) -- but died on wall-ceiling getting
                                                    # there, so safe only once 3.3's backstop lands
gate_safety_s        = 1.5                        # an EFFICIENCY value, not a safety one
adapter_reduction_factor = 64  + FWDLLM_FD_SCALE_INVARIANT=1    # memory/compute only
server_momentum      = 0.0
cos_ground_truth_audit = on, cos_probe_batch_size >= 1024, cos_probe_every = 25
monitors             = ||theta_tr||, rho, top_class_share;  score B and A
```

**`K`/`C` is not a setting but a standing instruction: hill-climb the pair against availability.** 10/30
is the tested point; `D(·)` is unmeasured against either, and **which of the two carries the wall clock is
itself open** (model K-C, §5.5e). The least-explored corner of the stack.

### P2.1 Ship checklist

*What the **defaults** should become. A flag earns a default change only when its A/B is scored and the
operator signs off.*

| # | change | from → to | gated on | state |
|---|---|---|---|---|
| **1** | `probe_combine` | `select` → **`mean`** | 5/5 arms, prediction to 1.5% | ready — ask the operator |
| **2** | `server_step_rule` | `raw_sgd` → **`trust_ratio`** | enacts to 8.7e-5, α-portable to 6 s.f. | ready — ask |
| **3** | `rho_schedule`/`rho_exp` | none → **`rm` / 0.25** | 3 arms, 313–318 commits | ready, but item 7 may supersede the schedule |
| **4** | `commit_gate`/`gate_rho_ref` | `var` → **`n_target`/`setpoint`** | G-1: 2.13× per vclock-hour | ready — ask |
| **5** | `gate_safety_s` | 0.4 → **1.5** | G-1b: 2× efficiency at matched `B` | ready — ask |
| **5b** | `gate_rho_ref` | `setpoint` → **`annealed`** | model §4.6; **G-2 confirmed 2026-08-15** (P4.5) | **decided, ask — but contingent on item 7/3.3 landing first.** `annealed` won decisively (peak 0.847 @ 65% vclock vs `setpoint`'s 0.833 @ 100%) but its own arm died on `[SIM_WALL_CEILING]` getting there — unsafe as a default without 3.3's backstop |
| **6** | `cos_probe_every` | 1 → **25** with the audit on | B19; emit-only, so a cost default | ready, with item 9 |
| **7** | **a controller** replacing the one-shot setpoint | fixed schedule → two-phase `B_max`, budget-landing anneal, adaptive `K`/`C`+`P`, saturation stop | model §1.4 item 5 | **not started — tracked as [P5.2](#p52-execution-plan--to-a-zero-input-run) phase 3** |
| **8** | `adapter_reduction_factor` | 16 → **64** + `FWDLLM_FD_SCALE_INVARIANT=1` | 3/3; cost knob only | ready |
| **9** | `replay_scoring.py` `BLOCK` = 50 | → block by **probe fires** | at stride 25 a 50-commit block holds 2 fires; the `len(blk) < 5` guard drops every row | **landed (2026-08-12)**, `--cos-block-fires` |

**Items 1–6, 8 are decisions waiting on you; 9 is a tool fix. Item 7 is the only genuine engineering** —
G-1b promoted it from "needed" to "the only remaining route", since `s` cannot bound `Φ`.

---

## P3 — Knob ledger

**One row per knob: flag, what happened, what to do with it.** Every feature is flag-gated, default =
old / byte-identical off; terminal flag state (`PERMANENT`/`FLAGGED`/`REVERTED`) is the operator's call.
Nothing reaches a verdict without predicted-vs-observed numbers and a run id.

| knob · flag | **measured** | verdict |
|---|---|---|
| **combination rule** · `probe_combine: {select\|mean}` | at matched `N`=200: `ρ` **5.386×** down, `‖Δθ‖` **5.430×**, against a predicted 5.466× (1.5%). `‖θ_tr‖` end 60.7 → 14.5; `‖θ‖²` growth rate **106×** lower (predicted 30×) | **WORKS · 5/5 — 10× in `ρ/cos` for zero extra compute, bytes or `η`/`N`/`p`.** Site `_accumulate_mean_over_probes`; also fixes the `P=1` crash (`sorted_indices[-2]`) and an RNG-stream mismatch |
| **step rule** · `server_step_rule` | `ρ = ρ*` to **8.7e-5**; `(1+ρ*²)^{T/2}` predicts `‖θ_tr‖` to **0.013%**; α moves `ρ` by **0 to 6 s.f.** | **WORKS.** Alone it does not bound `‖θ‖` — constant `ρ*` is still geometric. Ships with the anneal. Site `_apply_weighted_update` (pool-then-apply). Subsumes S3 |
| **anneal** · `rho_schedule`/`rho_exp` | enacts to **4.5e-4** over 318 commits. `exp`=0.55 drives `ρ` **17× below setpoint** by c186 → arm flat at 0.34; **`exp`=0.25 adequate** though formally outside the RM window | **WORKS; `exp`=0.25**, sized to the horizon. Superseded in principle by the budget-landing law (model §4.6a), unbuilt |
| **`ρ*` setpoint** | 0.03/0.06/0.09 → peak **0.695 / 0.801 / 0.846**, all still climbing at cutoff, all stable (`Φ` ≤ 1.145) | **band not closed.** The anneal spends most of the setpoint. Prefer the derivation (model §4.6a); read P4 by `Λ`, not by `ρ` |
| **commit gate** · `commit_gate: {var\|n_target}` | `N_req` closed form **exact** (28.1 `mean` / 1,013.3 at ρ*=.06; 94.2 `select` at ρ*=.01). At `K` = 30/50 it left the cap for the first time: `I` = **4 / 2**, `N` = 120 / 100, `commit_reason=natural` 100% vs `cap` 100% at `K`=10 | **MECHANISM SETTLED** — an `N`-controller, and it enacts |
| **gate `s`, throughput** (G-1) · `gate_safety_s` | `s` = 2.9/1.5 ⇒ `n_req` = 19.3/72.1, `I` = **2/8** off the cap 100%. Commits/vclock-h **11.3×/2.6×** the control; progress/commit **0.316/0.632** vs `√(N/200)` (**exact**); net `A`/vclock-h **2.13×/1.33×** vs a registered 2.2×/1.4× | **WORKS — the gate converts pool into commit rate at a net gain.** But `A`/vclock-h is **not** accuracy/vclock-h (P4.3) |
| **gate `s`, as safety** (G-1b) | at `ρ` = 0.06 `const`, `s` = 2.9/1.5 give `B`/commit **0.001789 on both**. Both turn: sustained −0.015 at `Φ` = **3.43/3.79**, peak **0.849/0.876**, `Λ`/`B` **0.706/1.411** (= `√(N₂/N₁)` = 2.000 exactly) | **REFUTED as safety, CONFIRMED as efficiency.** `B` has no `N`, `cos` or `s` in it, so no `s` can bound `Φ`. **`s` = 1.5 settled as a value** |
| **gate `ρ` reference** · `gate_rho_ref` | composition bug: `N_req ∝ ρ_t²`, so annealing `ρ` demands *less* pooling. `N_req` → 0 by c20, `I` floored at 1 on 1,271/1,273 commits, peak 0.394 decaying to 0.274; identical at α=0.1 | **fixed by `gate_rho_ref=setpoint`** (sizes from `ρ*₀`). No-op under `const` + `raw_sgd` |
| **`annealed` vs `setpoint`** (G-2, 2026-08-15) | matched config, matched **vclock budget** (40,000): `setpoint` (`084554`) spent its **entire** budget and reached peak 0.833; `annealed` (`003648`) was killed by `[SIM_WALL_CEILING]` at only 65% of the same budget (vclock 26,026) and was *already* at peak 0.847 — ahead on 65% of the vclock `setpoint` needed 100% of. `annealed` landed 4.2× the commits (3,353 vs 799) at 4× the real-wall cost per vclock-second. **Death mechanism corrected by T5 (2026-08-15): the gate's `I` was floored at 1 on 98% of `003648`'s commits** — 3,429 round trips for 3,353 commits, i.e. **8.40 s/round-trip vs `145729`'s 1.75** | **G-2 decided: `annealed` confirmed, sinking condition refuted.** Full readout in [P4.5](#p45-g-2--annealed-confirmed-but-real-wall-bound). Ships **contingent on 3.3** — this is the second `annealed` arm to die on the wall ceiling, and T5 names why: `N_req ∝ ρ_t²`, so annealing `ρ` demands less pooling until each commit is one round trip and the per-commit server path has nothing amortising it |
| **`Λ = 2B/s`** (T5, 2026-08-15, replay) | `N = p(ρ/s)²/G_rule` ⇒ `cos = ρ/s` ⇒ `Λ = Σρ²/s = 2B/s`. Out of sample: **−0.3%** on both `s`-pinned arms (`145729` 2.385 vs 2.379; `112201` 1.722 vs 1.717), **+21.5 to +23.3%** on the three whose `s` drifts (`035045`/`084554`/`003648`) — the same arms in the same order as §4.6's time-law miss table | **CONFIRMED as an identity wherever the gate holds `s`.** Consequence: **the `ρ` schedule is `Λ`-neutral at fixed `B`** — two schedules differ only in commits spent, never in learning banked. Any "this schedule learns more" claim is a comparison at unequal `B` |
| **real-wall cost model** (T5, 2026-08-15, replay) | `wall = 7.81·commits + 0.77·round_trips` (cos audit on, stride 25), least squares over 5 arms, **every one predicted to ≤1%**. The audit is charged per *commit* (85 s / 25), so audit-off is **4.41 s/commit + 0.77 s/trip** | **WORKS — the first cost model that separates the two terms**, possible only because the portfolio spans 1.02–20 round trips/commit. Feeds task 0.7's preflight; supersedes projecting from `s/commit` alone, which mis-prices any arm whose `I` floors |
| **`p`** · `adapter_reduction_factor` + `FWDLLM_FD_SCALE_INVARIANT` | `cos ∝ 1/√p` confirmed directly. But `‖θ_tr‖` 13.37/9.65/6.86 against `√p` 671/479/344 — ratio constant to **±1.3%** — so `A` is `p`-free. At matched `ρ*`,`N`,`T`: `A` = 1.64 (`rf`=16) vs **1.63** (`rf`=64), acc 0.592 vs 0.605 | **WORKS as built; INERT at a pinned `ρ*`.** The old ladder's win was raw SGD lowering `ρ` (0.2008→0.1863). Keep `rf`=64 for memory only |
| **`K` cohort width** | `ρ·√N` invariant to 4% over `K` 10→50; naive `ρ ∝ 1/√K` **falsified** (the gate returns the gain as fewer `I`). `K`≥20 holds **0.860** where `K`=10 collapses | **works**, but `K` is not a knob on `N` — set `K` and the gate together |
| **`K` at fixed `n_req`** | `K` 30→50: `I` 4→2 but vclock/round trip 8.9→12.3 s ⇒ commits/vclock-h **101.6 → 146.4 (1.44×)**, progress/vclock-h ×1.32. Staleness ≥1 on **0.143/0.458/0.408** of uploads at `K` = 10/30/50, max **1/2/4**; `pastdated_commits` max 1/0/**34** | **WORKS at ~70% efficiency**, and wide cohorts **do** produce genuine staleness (H-E answered). **`C` moved with `K` on all three arms** (30/60/100, `C/K` = 3/2/2), so neither the 1.44× nor the staleness rise is attributable to `K`: model K-C, and staleness tracks `C`, not `C/K` |
| **`K` at fixed `C` (K-1, 2026-08-13)** | First arm to move `K` without `C`: `K`=10/20/30, `C`=30 fixed (`145729`/`003601`/`054321`, `mean`/`trust_ratio`/`n_target`/`s`=1.5/`ρ*`=0.06/stride 25). Commits/vclock-h **72.4 / 64.9 / 65.6** (≤10% spread, inside the ±15% gate); stale≥1 frac **0.359 / 0.359 / 0.321** (also flat). `n_req`=72.1 and `C`=30 exact on all three — kill clean | **DECIDES K-C: throughput is flat in `K`, refuting the `K^0.37` pooling model (predicted ×1.5 over 10→30) — `C` is the wall-clock lever, `K` is not.** Only raw commit throughput was tested; sync↔async trainer-utilization shape and per-gradient staleness tolerance were not — see H-T (P5.3) for why `K`/`C` may still need to be dynamic under variable availability |
| **`I` iterations/bin** | at `K`=50 the gate cut `I` 18.5 → 5.9 — 3× fewer round trips at slightly larger `N`, best peak of the 08-07 portfolio (0.861), held | **works — buy `N` with `K`, not `I`** |
| **`η` server LR** | `ρ` 0.2004 → 0.0404 → 0.0101 over `η` .01/.002/.0005, **to 1%**. `η`=0.002 reaches 0.601 @ c120 / 0.815 @ c327 vs `K`≥20's 0.860 | **works as physics, FAILED as a fix** — 1:1 cost, and `Σρ²` still diverges |
| **`P` under selection** | `E` 2.988 (P=10) → 4.744 (P=30) over 37k events; predicted `ρ` ratio 1.260, **measured 1.236 (2%)**. `P`=30 learns faster per commit and **collapses sooner** (doubling 57 vs 68) | **prediction confirmed, including its harmful direction.** Never sweep `P` under selection again |
| **`P` under `mean`** (P-1, 2026-08-15) | matched `mean`/`const`/`setpoint`/`ρ*`=.06/`s`=1.5/`K`=10, same 10,800 vclock budget: round trips fell exactly as predicted (median `I` 3→1, floored), `T` 288→585, `Λ` 0.727→1.566, peak 0.858→0.873. But median trainer `τ` (real GPU time/round) **0.525s → 1.342s, ratio 2.56** — over the registered 2.5× kill line — so `P`=30 cost 54% more real wall (6,694s vs 4,342s) to spend the same vclock | **P-1 decided: `τ(30)/τ(10)` ≥ 2.5 ⇒ compute-bound, report and stop.** `bytes ÷3` is still real if bandwidth ever binds, but raising `P` is not a free throughput lever — do not build adaptive `P` (3.4) on that premise. Full readout [P4.6](#p46-p-1--p-under-mean-is-compute-bound-report-and-stop) |
| **top-k averaging** | offline over 34,447 events: random 1.00× / coin-top-2 1.73× / top-1 1.95× / top-3 2.72× / **all-10 3.16×** in `cos` gain, at `ρ/cos` 1×/1×/1×/3×/**10×**. All cost the same 20 passes | **settled offline — use all `P`.** The shipped rule is k=1: **zero** stability gain |
| **probe distribution** | `d_i` indistinguishable from iid `N(0,‖g‖²)`: top-1 of 10 → 3.811 observed / 3.798 synthetic; coin top-2 → 2.991 / 2.987 | **No "good probe" exists** — closes off all `\|d\|`-based selection; also validates FD linearity at the large chord |
| **server momentum** · `server_momentum` | β 0/0.5/0.75 at matched `ρ*`,`N`,`T`=66: `S` 0.479/0.747/2.059, `cos` ×1.00/1.47/2.61 (heavy-ball predicts 1.73/2.65), acc **0.524/0.594/0.626**. Orthogonality ratio **1.007/2.981/6.615** vs `(1+β)/(1−β)` = 1/3/7 | **WORKS mechanically, REFUTED as a lever** — `√x` progress for `x` budget, identical to raising `ρ`. **Leave at 0.0** |
| **weight decay** · `server_weight_decay: auto\|FLOAT` | **BUILT, ARM NOT RUN.** Decoupled decay on the trainable slice after the step; `auto` = `ρ²/2` from the realised `ρ`. Preflight: `Φ` = 1.029 with vs 2.258 without, at `ρ`=0.09 | **DEMOTED** — model §7.2 answered Q2 on the rig; optional confirmation only |
| **α heterogeneity** · `--partition-method ...alpha=` | 4/4 arms over 1000× in α: `var` floor 1.33/0.64/0.28, invariant 0.53 ± 0.01, both `K` predictions falsified | **understood and neutralised** by the step rule (model §2.6) |
| **`n_eff` sensor** · rides on `server_update_audit` | synthetic: recovers true `n`, scale-invariant over 100× in `‖g‖`, but **FAILS on directional disagreement** (4/20/100 distinct directions all → 1.00). 17 real arms: `n_eff/N` = **1.00 ± 0.01** | **FAILED as a sensor, KEPT as an audit.** An identity — in a gate it is a counter. Do not wire it to a controller |
| **split-half commit gate** · `commit_gate: cos` | per-commit SNR 0.07 | **DEAD** at `p`=450k (P6). **adaptive `P`** (`probe_budget: adaptive`) is PARKED behind the same wall |
| **`B_max` combiner** · `b_max_policy: {mean\|ratchet\|anchor}` (new 2026-08-16) | consequence of the origin fix: every anchored sense lands *above* the spend, so `anchor` (latest-only) walks `B_max` up with `B` and may never stop — on `125619`'s six fires `B/B_max` = 0.52 0.48 0.33 0.42 0.47 0.64, no crossing. The anchored quantity `B+R` is far more stable than raw `R` (CV ≈25% vs a 2.4× spread on agnews; ±6% on yahoo), i.e. it behaves like one constant measured six times | **`mean`, and the reasoning is the point:** P4.1's fixed-`Φ` stop holding across 10 arms with different schedules says the collapse boundary is a property of *total* inflation, so the fires estimate **one lifetime budget** and the estimator for a constant is the mean. It terminates (`B_max` settles, law C drives `B` to it) where `anchor` need not, and it does not compound the probe's documented conservative bias the way `ratchet` does. `ratchet`/`anchor` kept for the free after-the-fact A/B — every policy emits `sensed=` per fire |
| **eval subsample** · `eval_max_samples` (new 2026-08-16) | the eval is backgrounded but blocks the main thread when the prior one is unfinished: **359/359** evals on the yahoo control, 0/400 on the agnews control. 60,000 rows at seq 256 = ~30 GPU-s idle, **89.2 s under contention**, vs a ~91 s gap at stride 2. Batch 8 → 128 buys only **1.23×** (compute-bound, measured) | **fewer rows, not bigger batches.** Default 0 = full set, byte-identical. Fixed, fixed-seed *shuffled* subsample so error is a constant offset, not per-eval noise — and never `[:n]`, which is one client's shard (B17). **10,000 on yahoo** |
| **data bins** · `total_data_bins` (new 2026-08-16) | was hardcoded 150 = agnews' `1,200/8`, and the trainer indexes its own batch list by `data_id`, so **yahoo trained on 8.6% of its data and yelp-p on 23%** — silently, since the list is merely longer than the index | **CORRECTNESS FIX, ships enabled.** Derived from `dataset_registry.total_data_bins` (150 / 1,750 / 650 at `C`=100, batch 8); agnews byte-identical. The hyperparameter is the manual override. Echoed as `[DataBins]` |

### P3.1 Instruments and audits

| instrument · flag | state |
|---|---|
| **`cos(G,g)` probe** · `cos_ground_truth_audit`, `cos_probe_batch_size` | **RE-RUN, VALID, 6 arms.** Everything under the old 64-sample reference is void, **norm ratio included** (P8.1). `D` = **0.050**, invariant across a 3.57× swing in the prediction: 0.0506 (`select`,`p`=450k) vs 0.0501 (`mean`,`p`=118k). Splits into `L` = 8.5–9.5 and `S` = 0.48 — **scaling confirmed, constant 20× wrong, both halves data-side** (model §6) |
| **fixed reference (B17)** · `cos_probe_batch_size` default **1024** (was 64) | **WORKS · 8/8, and it costs.** Fixed-seed permutation over the whole test set; dominant-class share **0.27–0.29** vs balanced 0.25 on every arm. `test_cos_probe.py` fails the launch on skew. **Open defect: it costs ~50× the rest of the commit path, linearly in the reference** — so the fix is a stride, not chunking ([P9.2](#p92-launcher-gotchas) has the breakdown) |
| **stride (B19)** · `cos_probe_every` (default **1** = byte-identical) | **WORKS · 2/2.** Gates `_cos_probe_gradient()` on `_commit_count % k` at `FedSgdAggregator.py:500`. At `k`=25 both G-1b arms ran 945–1,364 commits at 9–14 s/commit against G-1's 88.9 — **16× the horizon, and the first arms since B17 not killed by `[SIM_WALL_CEILING]`**. Breaks `replay_scoring.py --cos` (P2.1 item 9) |
| **staleness histogram (B8)** · log replay | **DONE · H-E answered** — see P3's `K` row |
| **`Λ` out of sample (B16)** | **DONE · split verdict** (P4.3, Q3): the rule half predicts, the `p` half does not |
| **two-batch `cos` control (B13)** | **CANCELLED** — answered offline by `scripts/probe_reference_quality.py`; superseded by B17 |
| **long `K`=50 arm (B6)** | **DEMOTED** — `B` extrapolates the answer exactly |

---

## P4 — Arm ledger

Sorted by `Λ`, the ordering that predicts peak accuracy. `Φ` pred is `e^B`; `Φ` obs is `‖θ_T‖/‖θ_0‖` —
their agreement *is* the norm law. `‖θ_0‖` = 13.35 (`rf`=16), 9.6 (`rf`=32), 6.86 (`rf`=64). All arms
α=1, `mean`/`rf`=16 unless the row says otherwise.

| `Λ` | arm | run | `T` | `ρ` c1 | `B` | `Φ` pred → obs | peak | final |
|---|---|---|---|---|---|---|---|---|
| 0.008 | `n_target` `select` ρ*=.01, **`K`=50 `c`=100** | `171739` | 74 | 0.0100 | 0.0008 | 1.001 → 1.001 | 0.378 | 0.377 |
| 0.009 | `n_target` `select` ρ*=.01, **`K`=30 `c`=60** | `151316` | 72 | 0.0100 | 0.0008 | 1.001 → 1.001 | 0.382 | 0.382 |
| 0.008 | `n_target` rm ρ*=.01 e=.55 **@ α=0.1** | `001008` | 1263 | 0.0100 | 0.0002 | 1.000 → 1.000 | 0.388 | 0.264 |
| 0.008 | `n_target` s=0.4 rm ρ*=.01 | `220627` | 1273 | 0.0100 | 0.0002 | 1.000 → 1.000 | 0.394 | 0.274 |
| 0.009 | `var` gate, rm ρ*=.01 | `200209` | 1249 | 0.0100 | 0.0002 | 1.000 → 1.000 | 0.377 | 0.282 |
| 0.016 | `rm` ρ*=.02 e=.55 | `223510` | 186 | 0.0200 | 0.0007 | 1.001 → 1.001 | 0.379 | 0.329 |
| 0.044 | **`n_target` `s`=2.9 ρ*=.06 (G-1; `I`=2, `N`=20)** | `002208` | 79 | 0.0599 | 0.0292 | 1.030 → 1.027 | 0.448 | 0.432 |
| 0.066 | **`select` `rf`=16 ρ*=.06 (Q3)** | `151530` | 66 | 0.0599 | 0.0265 | 1.027 → 1.025 | 0.524 | 0.516 |
| 0.066 | `select` ρ*=.06 **+ `β`=0.5** (S1) | `160547` | 66 | 0.0599 | 0.0264 | 1.027 → **1.077** | 0.594 | 0.572 |
| 0.066 | `select` ρ*=.06 **+ `β`=0.75** (S1) | `160614` | 66 | 0.0599 | 0.0263 | 1.027 → **1.176** | 0.626 | 0.626 |
| 0.068 | `const` ρ*=.01 | `211736` | 186 | 0.0100 | 0.0092 | 1.009 → 1.009 | 0.488 | 0.429 |
| 0.083 | **`n_target` `s`=1.5 ρ*=.06 (G-1; `I`=8, `N`=80)** | `022448` | 73 | 0.0599 | 0.0280 | 1.028 → 1.026 | 0.508 | 0.433 |
| 0.238 | **`mean` `rf`=64 ρ*=.06 (Q3)** | `171950` | 67 | 0.0599 | 0.0267 | 1.027 → 1.025 | 0.605 | 0.595 |
| 0.199 | `rm` ρ*=.03 e=.25 | `013843` | 318 | 0.0300 | 0.0150 | 1.015 → 1.015 | 0.695 | 0.662 |
| 0.369 | `mean` raw, N=200 | `211800` | 179 | 0.0346 | 0.0850 | 1.089 → 1.088 | 0.804 | 0.793 |
| 0.393 | gate `setpoint` ρ*=.06 | `035557` | 312 | 0.0599 | 0.0592 | 1.061 → 1.061 | 0.804 | 0.804 |
| 0.398 | `rm` ρ*=.06 e=.25 **(Q3 anchor)** | `035045` | 317 | 0.0599 | 0.0597 | 1.061 → 1.062 | 0.801 | 0.799 |
| 0.398 | gate `setpoint` ρ*=.06 **@ α=0.1** | `062213` | 317 | 0.0599 | 0.0597 | 1.062 → 1.062 | 0.775 | 0.770 |
| 0.460 | gate `annealed` ρ*=.06 (`N` 200→40) | `013917` | 707 | 0.0599 | 0.0913 | 1.096 → 1.096 | 0.821 | 0.819 |
| 0.505 | gate `setpoint` ρ*=.06 `rm`, matched vclock **(G-2, completed 40,000/40,000)** | `084554` | 799 | 0.0599 | 0.0990 | 1.10 → 1.10 | 0.833 | 0.833 |
| 0.538 | gate `annealed` ρ*=.06 `rm`, matched vclock **(G-2, killed `[SIM_WALL_CEILING]` @ 26,026/40,000)** | `003648` | 3353 | 0.0599 | 0.2057 | 1.23 → 1.23 | 0.847 | 0.846 |
| 0.591 | `rm` ρ*=.09 e=.25 | `060834` | 313 | 0.0896 | 0.1332 | 1.142 → 1.145 | 0.846 | 0.843 |
| 0.631 | `mean` raw, N=200 **+cos** | `042027` | 312 | 0.0344 | 0.1434 | 1.154 → 1.158 | 0.849 | 0.849 |
| 0.727 | `mean` `const` ρ*=.06, **`P`=10 (P-1)** | `003628` | 288 | 0.0599 | 0.5138 | 1.67 → 1.67 | 0.858 | 0.854 |
| 0.819 | `raw_sgd` control (`select`) | `200325` | 177 | 0.1837 | 1.4172 | 4.12 → 4.23 | 0.855 | **0.672** |
| 0.837 | `select` raw, N=200 | `200242` | 179 | 0.1861 | 1.4687 | 4.34 → 4.47 | 0.853 | **0.761** |
| 0.891 | `select` `rf`=16 (`p` ladder) | `200358` | 195 | 0.2008 | 1.7146 | 5.55 → 5.74 | 0.852 | **0.353** |
| 0.937 | **`mean` raw, free gate +cos** | `065837` | 715 | 0.1202 | 1.2810 | 3.60 → 3.63 | **0.865** | **0.862** |
| 0.953 | `mean` raw, free gate (replicate) | `223446` | 693 | 0.1202 | 1.2776 | 3.59 → 3.62 | 0.864 | 0.850 |
| 1.014 | `select` `rf`=32 `p`=229012 | `212009` | 187 | 0.1941 | 1.1647 | 3.20 → 3.28 | 0.857 | 0.851 |
| 1.353 | `select` raw, N=200 **+cos** | `013806` | 328 | 0.1861 | 2.2086 | 9.10 → 9.47 | 0.855 | **0.251** |
| 1.566 | `mean` `const` ρ*=.06, **`P`=30 (P-1)** | `015455` | 585 | 0.0599 | 1.0455 | 2.84 → 2.86 | 0.873 | 0.869 |
| 1.722 | **`n_target` `s`=2.9 `const` ρ*=.06 (G-1b arm 1)** | `112201` | 1364 | 0.0599 | 2.4402 | 11.48 → 11.61 | **0.849** | **0.250** |
| 1.368 | `select` `rf`=64 `p`=118348 | `222817` | 188 | 0.1863 | 1.0939 | 2.99 → 3.04 | **0.859** | 0.852 |
| 2.385 | **`n_target` `s`=1.5 `const` ρ*=.06 (G-1b arm 2)** | `145729` | 945 | 0.0599 | 1.6901 | 5.42 → 5.46 | **0.876** | 0.793 |
| 0.709 | **law C, sensed `B_max`, `halt` (P-4 agnews controller)** | `125619` | 975 | 0.0678 | 0.4911 | 1.63 → 1.63 | **0.857** | 0.854 |
| 0.506 | **`rm`/.25 `setpoint` `log_only` (P-4 agnews control)** | `130614` | 801 | 0.0599 | 0.0992 | 1.10 → 1.10 | 0.835 | 0.834 |
| 0.404 | law C, sensed `B_max`, `halt` (P-4 **yahoo**, `p`=454,954) | `125713` | 580 | 0.0678 | 0.2838 | 1.33 → 1.33 | 0.299 | 0.268 |
| 0.464 | `rm`/.25 `setpoint` (P-4 **yahoo** control, killed at 34,441/40,000) | `125753` | 719 | 0.0599 | 0.0938 | 1.10 → 1.10 | 0.302 | 0.296 |

*The 4 surviving α-sweep arms, replayed as an **out-of-sample test of both laws**: `225718` Λ=1.021 peak
0.828 (pred 0.865), `012201` Λ=0.685 peak 0.865 (pred 0.854), `001241` Λ=0.634 peak 0.848 (pred 0.851),
`023623` Λ=0.605 peak 0.868 (pred 0.850) — **mean |peak error| 0.017**, the one large miss being α=0.1
with its known 3-point penalty. All four sit at Λ ≥ 0.6, so this confirms the law where it was already
strong and does not touch the steep region (Q3). The 08-07 K/η/P portfolio was deleted from disk.*

**The six 08-10 arms and the two G-1 arms were cut at 66–79 commits and were still climbing** — their
`peak` is accuracy-at-cutoff, comparable only against each other and against the anchor cut at the same
commit. **The cause was never the wall cap** but the 1024-sample audit's per-commit tax
([P9.2](#p92-launcher-gotchas)): every cos-carrying arm since B17 burned its real-wall ceiling at 4–15%
of its vclock budget.
`002208` and `022448` died on `[SIM_WALL_CEILING]` at vclock 633 s and 2,218 s against 14,400 s.

**Read four things off this table:**

1. **Peak accuracy is monotone in `Λ` at fixed `p`, and the plateau is ≈0.876, not 0.865** — `145729`
   banks `Λ` = 1.300 by its peak. Read the curve as rising to ≈0.865 by `Λ` ≈ 0.95 and creeping to
   ≈0.876 by `Λ` ≈ 1.3. `ρ*` is no longer sized from it (model §4.6a); its jobs are to forecast and to
   cross-check.
2. **`Λ` does not transfer across `p`** — `171950` banks 0.238 for 0.605 where the calibration says 0.72.
   Use `A`: out of sample it predicts the `rf`=32/64 arms to **0.0246** where `Λ` gives 0.0623.
3. **Whether an arm holds its peak is decided by `Φ` alone** — unless `β` > 0, where `Φ` follows
   `exp(((1+β)/(1−β))·B)`.
4. **Efficiency `Λ/B = 2cos/ρ` spans 20×.** `mean` at `N`=200, `ρ`=0.03 banks `Λ` = 0.63 for `B` = 0.14
   (11% of budget); `mean` under a free gate at `ρ`=0.086 spends **9× the budget for +0.016 accuracy**.
   **The best arm in the portfolio is the least efficient one.**

### P4.1 The Φ-stop counterfactual

**C-1's core rule, validated on replay. MEASURED, 10 arms, no GPU.** For every arm that learned: what
would a controller have banked had it stopped the first time `Φ` crossed a threshold? Smoothed curve,
11 evals.

| stopping rule | mean accuracy given up vs peak | worst arm |
|---|---|---|
| stop at `Φ` = 2.3 | 0.0154 | 0.0488 |
| stop at `Φ` = 2.5 | 0.0072 | 0.0199 |
| **stop at `Φ` = 2.7** | **0.0054** | **0.0139** |
| **stop at `Φ` = 3.0** | **0.0049** | **0.0122** |
| **no stop — run to the end** | **0.1408** | **0.5951** |

The 10 arms span both combination rules, both step rules, `rf` = 16/32/64, α = 0.1 and 1, and `T` from
177 to 1,364. **The rule needs no eval, no accuracy history and no task constant** — `Φ` is exact from
`ρ`. `013917` and `035045` never reach `Φ` = 2.3 and the rule correctly never fires. What this does
*not* settle is the threshold's **value** on an unseen task: B-1.

### P4.2 G-1b — `s` is efficiency, not safety

Two arms at `ρ*` = 0.06 **`const`**, `mean`, `n_target`+`setpoint`, `s` = 2.9 / 1.5 ⇒ `N` = 20 / 80.
Enactment exact on both: `n_req` = 19.3 / 72.1, `I` = 2 / 8 off the cap 100%, `ρ` = 0.0599 first and
last commit, no `[SIM_WALL_CEILING]`.

| | arm 1 `112201` `s`=2.9 | arm 2 `145729` `s`=1.5 |
|---|---|---|
| `T`, `N` | 1,364, `N` = 20 | 945, `N` = 80 |
| peak (smoothed / raw) | 0.845 / 0.849 | **0.874 / 0.876** |
| commit, `Φ` at peak | 578, `Φ` = 2.82 | 514, `Φ` = 2.51 |
| sustained −0.015 | commit 688, `Φ` = **3.43** | commit 744, `Φ` = **3.79** |
| drop at `Φ` = 4.0 / 5.0 | −0.045 / −0.100 | −0.015 / −0.044 |
| `Λ`/`B` (efficiency) | 0.706 | **1.411** |
| `B` per commit | 0.001789 | **0.001789** |

**Both arms spend budget identically and both turn.** What `s` buys is efficiency, doubled exactly
(`√(80/20)` = 2.000): a higher peak, the same peak sooner (arm 1's 0.849 at commit 251 / **0.97
vclock-h** against 570 / 1.45 h), and 2–3× less accuracy lost at every `Φ` past the turn. Score
retention on the **smoothed** curve — raw first-crossings put both turns at `Φ` ≈ 2.95, a false
agreement caused by a single 0.049 dip. The `Φ` ≤ 3.63 harbour is marginal at both `s` (0.019 / 0.015
below peak).

**`D` is elevated on both `const` arms, but tracks `Φ`, not training:**

| arm / bin | `D` |
|---|---|
| `002208` / `022448` (`rm`, ~75 commits, never left init) | 0.0506 ± 0.0106 / 0.0454 ± 0.0048 |
| `112201` / `145729` (`const`, trained then turned) | 0.1485 ± 0.0200 / 0.1035 ± 0.0172 |
| pooled: trained **and pre-turn** (acc > 0.78, `Φ` < 3.4) | **0.1025 ± 0.0213** (n=36) |
| pooled: **post-turn** (`Φ` ≥ 3.4) | **0.1627 ± 0.0246** (n=37) |
| pooled by accuracy: 0.60–0.84 / **> 0.84** | 0.185 ± 0.018 / **0.092 ± 0.022** |

`D` binned by accuracy is **non-monotone, and its highest-accuracy bin reads the lowest `D`** — so
"having trained" is not the driver. The `c` < 100 windows of the two `const` arms read 0.079 / 0.091
against the `rm` arms' 0.045–0.051, leaving **`const` vs `rm` as the leading confound**. Per-fire `sd` ≈
the mean, with **10/55 and 6/38 fires returning a negative `cos`** — SNR ≈ 1, so only arm-level averages
mean anything. This is D-2.

*Launch:* `expt_scripts/nodes/run_node_g1b_gate_s.sh`. **`run_node_g1_gate_s.sh` is superseded.**

### P4.3 G-1, Q3, S1, H-P — verdicts

**G-1** — throughput numbers in P3's "gate `s`, throughput" row. Both arms died on the wall ceiling at
4.4% / 15.4% of vclock budget, so their peaks are not a comparison. **Its extrapolation ordered the two
arms backwards**: projected `s`=2.9 → ~0.858 and `s`=1.5 → ~0.844 at 8 vclock-hours; the full-horizon
arms read **0.849 and 0.876**. `A` accumulates *through* the turn while accuracy falls (arm 1: `A` = 16.8
at peak, 99.6 at acc 0.250).

**Q3 — `A`, not `Λ`, is the progress coordinate.** Three arms matched at `ρ*`=0.06, `N`=200, `T`=66.

| arm | `G_rule` | `p` | `Λ` | `A` | predicted → **observed** |
|---|---|---|---|---|---|
| `mean` `rf`=16 (anchor) | 10 | 450,340 | 0.121 | 1.64 | 0.60 → **0.592** ✓ |
| `select` `rf`=16 `151530` | 2.988 | 450,340 | 0.066 | 0.90 | 0.485 → **0.524** ✓ |
| `mean` `rf`=64 `171950` | 10 | 118,348 | 0.238 | **1.63** | 0.72 → **0.605** ✗ |

**S1 — momentum refuted;** numbers in P3's `server_momentum` row. The registered "zero cost in `B`" is
false.

**H-P — the closed form is 20× high.** Registered `cos` ≈ 0.067 / 0.036 with a clean reference; observed
**0.0066 / 0.0019**.

### P4.4 Scoring rules for any A/B

**Replicate spread** (`200242`/`200325` are byte-identical configs; `223446` was repeated as `065837`):

| | `ρ` c1 | `ρ` c40-80 | `‖θ_tr‖` end | peak acc | final acc |
|---|---|---|---|---|---|
| `200242` vs `200325` | 1.3% | 5.0% | 5.7% | **±0.0009** | **±0.045** |
| `223446` vs `065837` | 0.0% | 0.0% | 0.5% | **±0.0007** | ±0.012 |

**Score on peak accuracy and the stability columns, never on final accuracy of a diverging arm** — the
spread is 50× wider past the turn.

**The interaction that shapes every A/B.** `var ∝ b²‖g‖²/n`, so averaging's 30× cut in `b²` puts `var`
under 0.3 at iteration 1: the gate commits at `I ≈ 1–2` instead of 18.5 and `N` collapses 185 → ~10–20.
Since `cos = √(G_rule·N/p)`, the 10× gain in `G_rule` and the ~10× loss in `N` **cancel** — under the
shipped gate, averaging buys **wall clock (~6.5× fewer round trips), not aim.** Therefore:

1. **"`ρ` drops 5.45×" holds only at matched `N`.** Score **`ρ·√N`**, which is `N`-free.
2. **Pin the pool** (`--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off` ⇒ `I` = 20,
   `N` = 200) or the arms are unmatched by ~10× in `N`.
3. `_pool_split_half_stats` returns `None` for a pool of one, so at `I` = 1 the record carries **no
   `pool_size` and no split-half components**. Reconstruct `N = K·(iteration_per_data_id + 1)`.

### P4.5 G-2 — annealed confirmed, but real-wall-bound

**Registered node, landed 2026-08-15.** `gate_rho_ref` `annealed` (`003648`) vs `setpoint` (`084554`),
otherwise identical: `ρ*`=0.06, `rm`/0.25, `K`=10, `s`=1.5, `max_runtime_s`=40,000 (the shared vclock
target), `sim_wall_ceiling_s`=28,800. Both start `ρ` c1 = 0.0599; neither arm turns — `trainable_weight_norm`
is still rising at cutoff on both, so peak ≈ final on both and `Φ` (1.10–1.23) never nears a stop threshold.

| | `annealed` `003648` | `setpoint` `084554` |
|---|---|---|
| vclock reached | 26,026 / 40,000 (65%) — killed by `[SIM_WALL_CEILING]` | **39,998 / 40,000 (100%) — completed** |
| real wall used | 28,803 s (100% of ceiling) | 11,255 s (39% of ceiling) |
| `T` (commits) | 3,353 | 799 |
| peak / final acc | **0.847** / 0.846 | 0.833 / 0.833 |
| `Λ` | 0.538 | 0.505 |
| `D` | 0.1145 ± 0.0125 (n=135 fires) | 0.0765 ± 0.0108 (n=32 fires) |

**Sinking condition refuted — G-2 confirms the shipped default.** `setpoint` spent its *entire* vclock
budget and still landed below where `annealed` sat when it was killed at only 65% of the same budget. This
is a tighter result than the earlier `013917`/`035045` pair it replaces as the registered evidence: same
config modulo the one flag, same target vclock, and `setpoint` still lost despite a 54%-larger effective
allowance. **P2's `gate_rho_ref=annealed` stands.**

**Real wall, not vclock, is the actual constraint.** `annealed` banked 4.2× the commits (3,353 vs 799) in
less vclock — that's what burned the ceiling: 4.2 vclock-s per real-wall-s vs `setpoint`'s 3.55. **This is
the second `annealed` arm to die this way** (buildplan §5 already flagged this same `003648` run when it
first hit the ceiling). Confirms 3.3's backstop is not optional before `annealed` ships as a default.

**Why it happened — corrected 2026-08-15 by T5's replay, and the first version was wrong.** This section
originally attributed the death to the cos-audit stride tax, and buildplan §4 to "a schedule that spends
its budget unevenly". Both were plausible; neither was measured. **`003648` ran with the commit gate's `I`
floored at 1 on 98% of its commits** — 3,429 round trips for 3,353 commits, against `145729`'s 7,560 for
945. The audit tax is real but it is a *per-commit* charge, so the mechanism is the commit count, and the
commit count blew up because `N_req ∝ ρ_t²` under `gate_rho_ref=annealed`: annealing `ρ` demands
monotonically less pooling until every commit is a single round trip with nothing amortising the
server-side path. Per round trip that is **8.40 s against `145729`'s 1.75 — 4.8×**.

| arm | commits | round trips | trips/commit | s/round-trip | fate |
|---|---|---|---|---|---|
| `003648` `annealed` | 3,353 | 3,429 | **1.02** | **8.40** | `[SIM_WALL_CEILING]` |
| `084554` `setpoint` | 799 | 6,392 | 8.00 | 1.76 | completed |
| `145729` `const` | 945 | 7,560 | 8.00 | 1.75 | completed |

**Round trips per commit is therefore the metric a preflight has to carry**, not commits and not the
floored fraction — flooring is *safe* (`N > n_req` ⇒ `ρ/cos < s`, conservative), it is merely the point at
which per-commit cost stops being amortised. T5 pre-registered **≥3** as the gate and used it to refute
`T_res` = 500 for the landing law (buildplan §5).

### P4.6 P-1 — `P` under `mean` is compute-bound, report and stop

**Registered node, landed 2026-08-15.** `P` ∈ {10, 30} under `probe_combine=mean`, otherwise matched:
`rho_schedule=const`, `gate_rho_ref=setpoint`, `ρ*`=0.06, `s`=1.5, `K`=10, same 10,800 vclock budget (both
completed cleanly, no wall-ceiling kill, `sim_wall_ceiling_s`=14,400 with 46–70% headroom used).

| | `P`=10 `003628` | `P`=30 `015455` |
|---|---|---|
| median trainer `τ` (real GPU s/round) | 0.525 s | **1.342 s — ratio 2.56** |
| median round trips/commit `I` | 3 | 1 (floor) |
| real wall used (same 10,800 vclock) | 4,342 s | 6,694 s (**+54%**) |
| `T` (commits) | 288 | 585 |
| `Λ` | 0.727 | 1.566 |
| peak / final acc | 0.858 / 0.854 | 0.873 / 0.869 |

**Round-trip mechanism predicted exactly** — median `I` fell 3→1 (floored), matching `÷3`. But
`τ(30)/τ(10)` = 2.56 (mean 2.69) sits over the registered **≥2.5 kill line**: batching does not make the
extra probes near-free, so `P`=30 cost 54% more real wall to spend the identical vclock budget. `Λ`/commit
only rose 1.06× (not the naive 1.73×) — `I` floors at 1 well before `N_req` would ask it to, so part of the
predicted per-commit gain is capped by the pool-size floor rather than realised.

**P-1 decided: `τ(30)/τ(10)` ≥ 2.5 ⇒ compute-bound — report and stop.** The accuracy numbers are not a
wash (`P`=30 does land more commits and a higher peak in the same vclock window), but per the
pre-registered kill test, raising `P` is not a free throughput lever. `bytes ÷3` is still real if bandwidth
ever binds, but **do not build 3.4's adaptive `P` on the premise that `τ(P)` is nearly flat — it isn't.**

### P4.7 P-4 — the law beats the control; the implementation had four defects

**First controller arms, run 2026-08-16, all four scored 2026-08-16. The direction is confirmed and the
arms are VOID as an acceptance test** — every controller arm ended on `max_runtime_s`, which
[P5.2](#p52-execution-plan--to-a-zero-input-run) already defines as void. Four defects, all now fixed
(regression checks in `test_landing_law.py` / `test_dataset_launcher.py`).

**What the law did, defects and all.** Law C beat the `rm`/`setpoint` control at the same 40,000 vclock on
both datasets, and beat it hardest on time-to-accuracy — the comparison the controller exists to win:

| | agnews law C `125619` | agnews control `130614` | yahoo law C `125713` | yahoo control `125753` |
|---|---|---|---|---|
| peak / final | **0.857** / 0.854 | 0.835 / 0.834 | 0.299 / 0.268 | 0.302 / 0.296 |
| `Λ` · `B` | 0.709 · 0.491 | 0.506 · 0.099 | 0.404 · 0.284 | 0.464 · 0.094 |
| commits per 40k vclock | **975** | 801 | 580 | 719 @ 34,441 |
| reached 0.83 / 0.28 at | **62%** of budget | 92% | **23%** of budget | 76% |
| trips/commit, by quintile | 10.7 11.1 **2.9 2.1** 8.4 | 8.0 flat | 8.3 16.1 12.0 **3.5** 20.0 | 8.0 flat |

Both law-C arms did that while **23% (agnews) / 48% (yahoo) of their commits took a step of length zero**
— so the margin is a floor on the effect, not a ceiling.

**(1) `halt` did not halt.** `[BudgetStop] reason=budget action=halt commit=150` fired on both law-C arms;
both ran on to the vclock ceiling. `_check_budget_stop` sets `_work_done`, and the data-bin lap in
`fwdllm_aggregator.py` *assigned* it `self._round > rounds` — `7 > 50` is False, so the lap silently
un-set the stop on the very next event (the stop fired at `data_id` 149 of 150). `_stop_fired` stays
latched, which is why it is logged once and never again while `stop_reason: budget` is stamped on all 826
following records. **Fixed: the lap ORs, never assigns.**

**(2) `B_max` was sensed in the wrong origin.** The probe inflates the *current* `θ`, so `ln Φ_knee` is the
budget remaining **from here**; `_B` accumulates **from `θ_0`**. Law C subtracted one from the other, so
the first re-sense returned `B_max` < `B` on every arm and `B_rem` clamped to 0:

| commit | sensed `B_max` | `B` | `B_rem` | `ρ*` |
|---|---|---|---|---|
| 149 | 0.248 | 0.273 | **0** | **0** |
| 299 | 0.298 | 0.273 | 0.025 | 0.013 |
| 449 | 0.586 | 0.283 | 0.303 | 0.045 |
| 899 | 0.281 | 0.492 | **0** | **0** |

**Fixed: `B_max = B + ln Φ_knee`**, which is monotone above the spend by construction, so a re-sense can
never zero `ρ*` on its own. Under it, commit 149 would have set `ρ*` = √(2·0.248/300) = **0.041**.

**(3) `ρ*` = 0 cost the *most* round trips, not the fewest.** `_n_required` folded `ρ`=0 into its "no `ρ`
yet" None branch, so `_gate_satisfied` was False forever and the commit landed only through the
`max_iterations_per_data_id` bypass: **20 round trips to move `‖θ_tr‖` not at all** (frozen at 17.847 for
`125713`'s last 131 commits). **Fixed: `ρ`=0 ⇒ `n_req`=0 ⇒ commit in one trip.** Note the mid-run
quintiles at **2.1 / 2.9 / 3.5** trips/commit — under T5's ≥3 gate, which the preflight passed because it
priced law C off the `ln 2` prior rather than the sensed value.

**(4) `total_data_bins` was hardcoded at agnews' 150** (`fwdllm_aggregator.py`), and the trainer indexes
its own batch list with `data_id`. agnews' shard is exactly 150 batches, so agnews was right by
coincidence; **yahoo trained on 1,200 of each client's 14,000 samples — 8.6% of the dataset, the same
1,200 every lap**, and yelp-p would have used 23%. Nothing raises: the list is merely longer than the
index. **Fixed: derived from `dataset_registry.total_data_bins` (150 / 1,750 / 650 at `C`=100, batch 8 —
the numbers buildplan §1 already states); agnews byte-identical.**

**Fixes verified on GPU, `225224`** (agnews, `--b-max 0.05 --t-res 20` to force the stop early, probe off,
~13 min): `[DataBins] total_data_bins=150 source=registry` · `[BudgetStop] reason=budget action=halt
commit=59` and the run **ended there** — 59 commits, `stop_reason` on **1** record against `125619`'s 826,
`inform_end_of_training` reached instead of `max_runtime_s` · **0 zero-`ρ` commits** · 218 trips / 59
commits = **3.69**. Defect 4's derivation is confirmed live.

**And on yahoo, `234931`** (2026-08-16 23:49, law C + `annealed` + `halt`, `--eval-max-samples 10000`,
audit off; force-killed by the runner 120 s past a 2,500 s **wall** budget with 31 commits): `[DataBins]
total_data_bins=1750 source=registry dataset=yahoo`, coverage **1,400,000/1,400,000 = 100%**, and the
trainer-side cross-check now confirms live — **`[DataBins] confirmed by trainer …: 1750 batches`**, which
`225224` could only compile-check. Also `n_req` = 93.4 at `iteration_per_data_id` ≈ 9 (**~10 trips/commit**,
against the ≥3 gate), `ρ` 0.0678 → 0.0645 monotone, **0/31 zero-`ρ` commits**. **Defects 2, 3 and 4 are
therefore verified on the dataset they were found on.** Defect 1's `[BudgetStop]` ending is verified on
agnews only — it needs a full-length yahoo arm.

**One new behaviour this run exposed, not a defect:** `[BmaxProbe] commit=25 base_acc=0.105 too close to
chance 0.100; keeping B_max=0.693147`. See [P4.8](#p48-yahoo-is-under-trained-not-broken).

**Cost note for any short yahoo arm:** `234931` spent **32 of its 44 minutes before commit 1** — test set
plus 100 client shards tokenized cold — and the runner charges its budget from launch, not from the first
commit. Warm caches remove most of it; budget wall-from-launch regardless.

**Two standing caveats on the yahoo pair, neither a defect.** (a) Both yahoo arms ran
`sim_charge_profiles/fluxtune.yaml`, profiled on agnews — 0.255 real-s/vclock-s on agnews against 0.658 on
yahoo, so **yahoo-vs-agnews per-vclock comparisons are invalid** until `profile_sim_charges.py` runs for
yahoo. Yahoo-vs-yahoo is fine.

**(b) The eval is backgrounded, and it still serialises into the critical path.** `_eval_snapshot_model`
blocks the main thread when the previous eval is still running, and it was — `eval still running at commit`
fired on **359 of 359** evals on the yahoo control, 286/289 on the yahoo controller, 287/488 on the agnews
controller, and **0/400** on the agnews control. Yahoo's 60,000 rows at seq 256 cost ~30 GPU-s on an idle
A40 and **89.2 s measured under trainer contention**, against a ~91 s inter-eval gap at stride 2. **Batch
size is not the lever** — measured 1.23× from batch 8 → 128, because the pass is compute-bound rather than
launch-bound. Fewer rows is: `eval_max_samples` (new, default 0 = full set = byte-identical) takes a
**fixed, fixed-seed shuffled** subsample, so the sampling error is a constant offset rather than per-eval
noise and peak-vs-final stays as precise as the full set. Shuffled because `test_index_list` is per-client
shards in client order — a head slice is one client's Dirichlet shard (B17). **10,000 for yahoo.**

**Why yahoo sits at 0.30 — see [P4.8](#p48-yahoo-is-under-trained-not-broken).**

### P4.8 Yahoo is under-trained, not broken

B-1's backprop reference reaches **0.73** on yahoo (model §7.1, matching FwdLLM's own ~0.76); these arms
reach **0.30**. **Confirmed 2026-08-17 (§9 rung 1): 0.734 on the FL rig's own data path** — same client
shards, same `test_global`, exact gradient instead of pooled forward differences — so the pipeline is
cleared and the gap is budget. Every collapse fingerprint already said so:

| | agnews control | yahoo control | reads as |
|---|---|---|---|
| prediction entropy (final) | 1.342 vs `ln 4`=1.386 | 2.297 vs `ln 10`=**2.303** | yahoo's softmax is still ~uniform |
| logit norm | 0.573 | 0.337 | the head has barely left its init |
| MCC | 0.780 | **0.227** | but it is real signal, not chance |
| accuracy trajectory | monotone to 0.834 | **monotone to 0.296, still rising when killed** | no turn, no collapse |
| `top_class_share` | 0.279 (balanced 0.25) | 0.304 (balanced **0.10**) | early-training skew, not collapse |

A *collapsed* model has low entropy and high `top_class_share`; yahoo has the opposite. **Λ is the
quantitative version of the same statement:** T5 needs `Λ` ≥ 0.95 for yahoo and these arms banked
**0.40–0.46** — under half. Defect (2) is most of that gap on the law-C arm (it spent `B`=0.284 of a
`ln 2`=0.693 prior before `ρ*` was zeroed; law C run to its asymptote banks `Λ` = 2`B`/`s` ≈ 0.92).

**Checked and cleared as explanations:** `p` reads 454,954 from `[ProbeDim]` (10-way head, correct);
`max_seq_length`=256 and `num_labels`=10 are both plumbed on both sides; the run evaluates on the full
60,000-row official test set, which is label-balanced to ±2%. **One caveat on the reference number
itself:** `probe_inflation_damage.py` reads accuracy off `test_global[:2000]`, and `test_index_list` is
per-client shards concatenated in client order — so B-1's 0.73 is measured on a mildly Dirichlet-skewed
slice (top class 20.7% against a balanced 10%), not the balanced 60,000 the FL arms use. Not enough to
explain 0.73 → 0.30, but the two numbers are not measured on the same distribution and the comparison
should not be quoted as if they were.

**Under-training also disables the `B_max` sensor on yahoo, which is new (`234931`, 2026-08-16).** 3.1's
probe refuses to read a knee off a model that is at chance — `[BmaxProbe] commit=25 base_acc=0.105 too
close to chance 0.100; keeping B_max=0.693147`, the guard at `FedSgdAggregator.py:978`, with the whole
`Φ` grid (1.5…4.0) returning 0.078–0.145 — noise around chance, no knee to find. **Consequence: early
yahoo commits run law C off the `ln 2` prior rather than a sensed budget.** The controller is unharmed —
a prior is exactly what law C carries for this case, and `ρ*` annealed cleanly all 31 commits — but the
**pre-registered agnews-vs-yahoo `ρ*` divergence cannot be measured until yahoo clears chance**, so the
sensor's first firing commit is itself a result to read off the re-run. This is the *same* under-training
statement as the table above, now visible in the controller's own instrument.

**The discriminating test now HAS been run** (buildplan §9 rung 1, 2026-08-17): centralized AdamW on the
**FL rig's own data path** — one client's `TextClassificationDataManager` loader, not the B-1 rig's
test-global half — evaluated on the same 60,000. **yahoo 0.734** (0.7333 / 0.7263 / 0.7339 over three
epochs, untrained 0.1018), **yelp-p 0.874**, against a pre-registered ≈0.70-clears / ≈0.30-indicts. The
path is clean and the gap is budget. Both curves are flat from epoch 1, so 0.73 is the ceiling itself.

### P4.9 The 2026-08-17 smoke — four defects, two of them silent

The smoke that was meant only to size the long runs found four, none of which fail loudly:

**(1) `eval_max_samples` destroyed every evaluation it touched.** `compute_metrics_with_logging` walked
the *full* test loader while indexing `preds[i*8+j]`, but under subsampling `preds` holds 10,000 rows —
`index 10000 is out of bounds`. It is caught as non-fatal, so training continued and the arm looked
healthy while emitting **zero `agg_eval` records**. Perfect correlation: every run with `eval_max_samples`
set produced no accuracy data at all (yahoo controller 20/20 evals failed, control 23/23, yelp-p 11/11).
**Fixed:** the dump is debug-only, so it now returns early unless DEBUG is on *and* the row counts match,
and counts rows with a running index (the old `i*8+j` also assumed every batch was full).

**(2) The sim-profile provenance gate rejected every per-dataset profile.** It required `_fluxtune_n` in
the source-run name; dataset-suffixed run dirs read `_fluxtune_yelp-p_n100_`. The yelp-p profile was
correct and the gate simply could not recognise it, so **the entire yelp-p pair never launched**.
**Fixed:** `_{baseline}_{dataset}_n` is accepted too, enumerated from the registry — never a bare
wildcard, which would re-admit the sibling baselines (`fwdllm` ⊂ `fwdllm_it_unaware`) the gate exists to
reject.

**(3) The commit counter double-counted zero and disabled its own guard.** `grep -c … || echo 0` emits
`0` *and* runs the fallback, so `commits` became `"0\n0"`, making `[ "$commits" -lt 5 ]` a syntax error —
the "under 5 commits ⇒ systemic fault, abort" check silently never fired. Compounding it, `server_update`
only exists under `server_update_audit`, which the real profiling arm does not set. **Fixed:** counts
`[ServerStep] … commit=` as a mode-independent fallback; the two sources agree exactly on all six sim runs.

**(4) A 600 s real arm cannot price a vclock.** yelp-p's profile charged `fedavg` **2.61 s** off a single
114.44 s warmup stall — 91% of the pooled total — against a 0.0476 s median and agnews' healthy 0.0563 s
(n=3601): a ~50× over-charge on a number that converts vclock into real work. **Fixed:** the profiler now
refuses any category whose largest sample carries >25% of the pooled total. Note the criterion is *not*
mean > p90 — that fires on the good agnews entry too (0.0563 > 0.0525). `drain_tail` is a second warning:
its spans ramp 0.13 s → 1.9 s across the smoke and never reach steady state. **Profile off a full-budget
real arm, never a smoke.** `SMOKE=1` now writes to the gitignored
`sim_charge_profiles/smoke/`, so a plumbing check exercises the profiler without ever pricing a scored
arm — which is what `d5b10a2` asked for and nothing enforced.

---

## P5 — The queue

**The estimator is finished; what is left is the controller.**

> **The queue itself moved.** What to do next, in order, with costs and done-when, is
> **[buildplan §-1](fl_fwd_ft_buildplan.md)** — one status board, updated in place. What remains here is
> the *record*: P5.1's registered arms, P5.2's phase outcomes, P5.3's open hypotheses.

### P5.1 Registered nodes

*Four registered 2026-08-11; all four now landed. B-1 landed 2026-08-13 (erratic — result in model
§7.1/§5.5b, R3); K-1 landed 2026-08-13 (flat in `K` — result in P3's "`K` at fixed `C`" row); **P-1 landed
2026-08-15 (compute-bound, report and stop — result in P3's "`P` under `mean`" row and
[P4.6](#p46-p-1--p-under-mean-is-compute-bound-report-and-stop))**; **G-2 landed 2026-08-15 (`annealed`
confirmed, ships contingent on 3.3 — result in P3's "`annealed` vs `setpoint`" row and
[P4.5](#p45-g-2--annealed-confirmed-but-real-wall-bound))**. Table intentionally empty — no node currently
registered (R3).*

### P5.2 Execution plan — to a zero-input run

**The target.** Full FluxTune, DistilBERT + adapters, α = 1, 100 trainers — **agnews first, then yahoo on
the same binary and the same flags**, no operator input beyond model / PEFT / `p`. **If the second run
needs one edit, the method is not general.**

**How this section is kept.** One block per phase, recording its *outcome*; a phase's rows are deleted
once the result has landed in [P3](#p3--knob-ledger), [P4](#p4--arm-ledger) or the model doc (R3).

> **Live task state, ordering and specs are all in [fl_fwd_ft_buildplan.md](fl_fwd_ft_buildplan.md)** —
> §-1 the status board, §-0 the generality checklist, §0 the conventions every spec inherits, §1 the
> dataset substrate table (`p`, split, shards, bins/round, sequence length per dataset) that phases 1–4
> read.

#### Phase 0 — unblock the instruments · **all 10 done (2026-08-12/15)**

Specs and gates: buildplan §2. Two have known gaps, both live: **0.7**'s projection prices law C off the
`ln 2` prior, so an arm can pass the ≥3 trips/commit check at launch and breach it in flight
([P4.7](#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects) defect 3); **0.10**'s
grep never covered `lib/python/flame/`, which is where defect 4 lived.


#### Phase 1 — B-1 · **done (2026-08-13)** — see model §7.1/§5.5b

Validation gate reproduced model §7.1 on agnews unchanged (knee at `Φ`≈3.0, three-mode separation exact).
Yahoo/yelp-p wired through the registry, `p` confirmed against `[ProbeDim]`. Outcome: **erratic** — knees
neither invariant nor monotone in `num_labels` (agnews ~3.0–3.3; yahoo and yelp-p both ~2.0–2.3 despite
being at opposite ends of the class-count range). Yahoo's low base accuracy checked against an
undertraining confound (9-epoch rerun) and ruled out — more training overfits, doesn't help, and the
early knee holds regardless. Per this task's own decision table, **3.1 (the online injection probe) is
now mandatory infrastructure**, not a fallback. **Replicated same day** — 2 independent runs per dataset,
all agreeing tightly (agnews knee 3.0–3.5 both times; yelp-p's normalized knee at Φ=2.0–2.5 both times).

#### Phase 2 — K-1, P-1, G-2 · **done (2026-08-15)**

All three registered nodes landed: K-1 (2026-08-13, `C` not `K` carries the wall clock — P3), P-1
(2026-08-15, compute-bound at `τ(30)/τ(10)`=2.56 — report and stop, P3/[P4.6](#p46-p-1--p-under-mean-is-compute-bound-report-and-stop)),
G-2 (2026-08-15, `annealed` confirmed but real-wall-bound — P3/[P4.5](#p45-g-2--annealed-confirmed-but-real-wall-bound)).
**K-1 was the one that changed a phase-3 design decision** (which knob 3.4 climbs); P-1 and G-2 each
resolved a ship-checklist item without changing 3.x's design.

> **Preflight, do not skip, still applies to any future node.** `--dry-run`, then `--only --yes --clean
> --force`; `--num-trainers 100`; assert `⌈n_req/K⌉ ≤ max_iter`. **The wall-clock budget check (task 0.7)
> now runs inside `--dry-run` itself** and refuses on breach — no longer by hand. Note G-2's `annealed` leg
> still died on `[SIM_WALL_CEILING]` despite this preflight: the preflight catches a *gross* breach, it
> does not replace 3.3's runtime backstop for a schedule that spends its budget unevenly.

#### Phase 3 — the controller · **built 2026-08-15, corrected 2026-08-16 by its first arms**

All five shipped components are live in `FedSgdAggregator`; the two that are not built are 3.5 and 3.4's
mid-run `P` change. Decisions, constants and the open questions a re-run must answer: **buildplan §5**.

| # | component | state |
|---|---|---|
| **T5** | replay the landing law against the gate | **done (2026-08-15)** — `replay_landing_law.py`. Settled `T_res`=300, `f`=0.95, `ρ_max`; refuted `T_res`=500 and the `ρ*≤ρ*₀` clamp; confirmed `Λ`=2B/s; found `003648`'s real death mechanism ([P4.5](#p45-g-2--annealed-confirmed-but-real-wall-bound)) |
| **3.3** | law-C anneal + `B ≥ f·B_max` stop, `phi_stop: off\|log_only\|halt` | **done (2026-08-15); `halt` fixed 2026-08-16** (P4.7 defect 1) and verified on GPU (`225224`) |
| **3.1** | `B_max` re-sensed on a stride by injection probe on a copy of `θ_tr` | **done (2026-08-15); origin + combiner fixed 2026-08-16** (defect 2). `B_max = B + ln Φ_knee`, combined across fires by `b_max_policy=mean`. **Open, new 2026-08-16:** the chance guard declines to fire on an at-chance model, so on yahoo the controller runs the `ln 2` prior until the model clears chance ([P4.8](#p48-yahoo-is-under-trained-not-broken)) |
| **3.2** | `ρ*_t` from *remaining* budget, every commit | **done (2026-08-15)**; the `ρ`=0 gate stall fixed 2026-08-16 (defect 3) |
| **3.5** | Prechelt GL/patience stop on smoothed held-out accuracy | **todo, not urgent.** Window, threshold and patience all unsized — size by replay before it ships. Buildplan §5 |
| **3.4** | adaptive `K`/`C` and `P` | **K-1: `C` is the knob, not `K`** (P3). **P-1: compute-bound** (P3/[P4.6](#p46-p-1--p-under-mean-is-compute-bound-report-and-stop)). Only the mid-run-`P`-change engineering and a bandwidth-driven case remain live |

**The three design decisions, all now settled.** (1) A re-sense moves `B_max` and nothing else — `T_res` is
a chosen rate, not run state, so there is nothing to reset (**law C**). (2) The stop **halts**, via
`_work_done`; the counterfactual P4.1 needs is preserved by `log_only`, not by declining to halt. (3) The
`B_max` fires are noisy estimates of **one lifetime budget**, so they combine by **mean** — P4.1's fixed-`Φ`
stop working across 10 arms with different schedules is the evidence, and `anchor` (latest-only) may never
terminate. `ratchet` and `anchor` remain available for the after-the-fact A/B.


#### Phase 4 — the two zero-input runs · *state: **run 2026-08-16, VOID as an acceptance test; re-run after the four fixes** — full readout [P4.7](#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)*

> **Where this stands (2026-08-17).** Four arms ran 2026-08-16 and are void: law C beat the control on
> both datasets and by 1.5–3× on time-to-accuracy, but both controller arms ended on `max_runtime_s`
> rather than `[BudgetStop]`, which the rule below defines as void. All four defects are fixed
> (`5441db34a`) and **verified live on both datasets** — agnews `225224`, yahoo `234931`
> ([P4.7](#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)). **No scored arm
> exists yet under the fixed code.**
>
> **Re-run gate — all four must hold before the next launch is scored:** (a) `[DataBins]` reads 1,750 on
> yahoo and 100% coverage; (b) no commit reports `rho_star` = 0; (c) the controller arm ends on
> `[BudgetStop]`; (d) trips/commit ≥ 3 at every quintile. **(a), (b), (d) already hold on yahoo**
> (`234931`, 31 commits); (c) is verified on agnews only. Yahoo also needs `profile_sim_charges.py`, a
> bigger vclock, and `--eval-max-samples 10000`.
>
> **The pre-registered acceptance criterion — `ρ*` diverging by dataset at the commit-150 re-sense — may
> not be measurable on yahoo at all.** With the origin fixed the two knees at commit 150 were 0.248 vs
> 0.237 (`ρ*` 0.0407 vs 0.0397), nearly identical; and `234931` showed the probe *declining to fire* on
> yahoo while the model sits at chance ([P4.8](#p48-yahoo-is-under-trained-not-broken)). Either way that
> is a finding about the sensor, not a failure of the law, and it moves T5's first open row.
>
> **Ordered task list for the next session: [buildplan §-1](fl_fwd_ft_buildplan.md).** Task A (the
> centralized-backprop control on the FL data path) gates C and D — both yahoo arms are worthless if it
> comes back at 0.30.

**Launch:** `expt_scripts/nodes/run_node_p4.sh <agnews|yahoo> <controller|control>`, four invocations,
one per machine, all four at the same 40,000-vclock target with the **cos audit off** (Phase 4 scores `B`,
`Φ`, `Λ`, `A`, every one exact from `ρ`; only `D` needs the audit, and the audit is 3.40 s of a 7.81 s
commit — the tax that killed G-2's annealed leg). `control` = fluxtune_v2 at `--phi-stop log_only`, which
keeps measuring the past-the-stop counterfactual P4.1 was built from instead of destroying it.

**Projected, audit-off, from T5's cost fit:**

| arm | commits | trips | trips/commit | wall |
|---|---|---|---|---|
| agnews controller (law C) | 967 | 4,848 | 5.01 | 2.2 h |
| yahoo controller (law C) | 916 | 3,382 | 3.69 | 1.9 h |
| control, `gate_rho_ref=setpoint` | 799 | 6,390 | 8.00 | 2.4 h |
| ~~control, shipped `annealed`~~ | 5,196 | 5,271 | **1.01** | **7.5 h** |

**The control runs `setpoint`, not the shipped `annealed`, and the last row is why:** `rm`/0.25 +
`annealed` is bit-for-bit `003648`'s config, and it reproduces that arm's gate starvation exactly
(1.01 trips/commit). `setpoint` is also the only G-2 leg that completed (`084554`), and a control has to
finish to be a control. **Task 0.7's `const`/`rm` branch does not catch this** — it has no model of `rm`
decaying `n_req`, so it projects ~0.6 h for that 7.5 h arm. Extending it is the standing follow-up before
any `rm`+`annealed` arm runs unattended again.

**All four arms are pinned to `rf=16` (2026-08-16), and `rf=64` will not carry `annealed` at any `T_res`.**
T5 settled law C's constants against `p=450,340`; the v1/v2 split (2026-08-15) repointed the `fluxtune`
alias to `rf=64`, `p=118,348`, so every projection above silently described a model 3.8× the one that
would have launched. Re-derived at `rf=64`, the `Λ ≥ 0.95` and `trips/commit ≥ 3` floors close against
each other on the prior phase — a 9-unit window at `T_res` 82–90 where `Λ` clears by 0.001–0.007 and 28%
of commits still floor to `I=1` — and the two-phase trajectory reads **1.71** trips/commit on agnews,
**1.43** on yahoo. `setpoint` composes at every `T_res`, but `annealed` is what P4 exists to test (item 5b
ships it *contingent on 3.3*, and 3.3's backstop is this controller), so `rf` moves instead, on both arms.
P4 therefore compares law C against v2's iteration-control policy **at `rf=16`**, not against v2 entire.
**That `rf=64` cannot carry `annealed` is a standing blocker on item 5b**, independent of P4.

**Two launcher defects found the same day, both fixed.** (a) The Task 0.7 preflight read its knobs from
the `config_overrides` layer alone, so every catalog-set value — `rf` above all — silently took a code
default; it now resolves catalog → overrides. (b) `cos_ground_truth_audit` is *on* in v2's catalog and
`--cos-ground-truth-audit` was opt-in only, so the audit-off design above was not expressible;
`--no-cos-ground-truth-audit` now exists. `replay_landing_law.py` took `p` from a hardcoded `rf=16` table
and now takes `--rf` from the registry.

**Infrastructure (measured 2026-08-16, `jayne`).** Datasets *and* the `test_fwdllm` conda env are on
`/coc/scratch` (NFS, shared by every machine); **only the repo is machine-local** (`/home` is `/dev/md1`,
ext4, not NFS — and it is a different directory from the NFS `/nethome`). So a new node needs a `git
clone` and nothing else. Run output must stay on local `/home`: telemetry is ~700 MB/run.

**Read `[BmaxProbe]` before anything else — that line is the result.** T5 pre-registered the divergence at
the commit-150 re-sense: agnews `ρ*` 0.0530 → **0.0764** (`I` 6 → 12) against yahoo's 0.0530 → **0.0573**
(`I` 6 → 7). **A controller arm that ends on `max_runtime_s` rather than
`[BudgetStop] reason=budget` is void** — the law did not land.

**4.1** agnews · **4.2** yahoo, *same binary, same flags*.

> **Acceptance:** both reach their plateau and **end within 0.015 of peak**, with sensed `B_max`, `ρ*`,
> `K`, `P` logged per run and **differing between datasets without anyone having supplied them**.

**Every arm reports `B` as a fraction of `B_max` and `A` against P4's calibration** — both exact at any
horizon, so both failure modes are diagnosable ~20 commits in, before the accuracy curve resolves.

### P5.3 Open hypotheses

*Resolved and deleted: H-A, H-B, H-C, H-D, H-E (confirmed), H-F, H-G, H-I, H-K, H-L, H-N, H-O, H-P
(refuted), H-Q (refuted), H-R (confirmed) — all folded into the model doc.*

| ID | hypothesis | test | rung |
|---|---|---|---|
| **H-S** | The residual 3.5× between the rig's `S` (1.68) and the arms' (0.48) is the **FD chord** — at `h‖v‖` = 6.71 vs `‖θ_tr‖` = 6.75 each probe steps a full parameter-norm, so `d` is a chord-averaged slope | Rig: same `v`, true `⟨g,v⟩` by backprop vs the shipped central FD, correlated over many draws. `cos(d_FD, d_true)` ≈ 0.3 if this is the cause, ≈ 1 if not. A chord-averaged slope is **still Gaussian**, which is why P3's distribution check passed it | 2 |
| **H-H** | The FD's discarded curvature term `vᵀHv` carries usable signal | The central difference uses only the *difference* of `L(θ±hv)`; their **sum** ≈ `h²vᵀHv`, **already computed and thrown away**. Log it, correlate with realised loss decrease at the step scale taken | 2 |
| **H-J** | `K ≥ 20` **defers** collapse rather than preventing it (commit 1,200–1,700) | `B` extrapolates it exactly — **demoted to a confirmation** unless a cheap node is free | 1, then 4 |
| **K-C** | **`C`, not `K`, carries both the wall clock and the staleness.** The dispatcher refills to `C` and never reads `agg_goal` (`async_base.py:405`), and a contributor is held from re-dispatch until a release boundary (`_agg_pending_commit_ref`), so commits/hour ≈ `C/(n_req·τ)` and staleness ≈ `C/n_req` — both `K`-free | **Rung 1 first, free:** replay the three existing `K`/`C` arms for commits/vclock-h against `C/n_req` vs `K/τ(K)` — they cannot discriminate (`C/K` = 3/2/2) but they *can* falsify the arithmetic. Then **K-1** at fixed `C`; predictions and kill in [P5.1](#p51-registered-nodes) | **1, then 4 — done, confirmed. See P3's "`K` at fixed `C`" row** |
| **H-T** | Dynamic `K`/`C` matters more as device availability drops (async's constant-work-rate benefit vs sync's peak-then-drain-to-zero pattern near a databin boundary), and/or matters more for forward-perturbation gradients than backprop — the noisy scalar-JVP estimate needs a larger aggregation pool for SNR to begin with, and is plausibly more staleness-sensitive than an exact backprop gradient computed at a single point. K-1 only measured raw commit throughput at 100% availability, so it cannot speak to either claim | (a) Replay K-1's own three arms for *trainer busy-fraction over wall time within a round* (not commits/vclock-h) — tests whether the burst-then-drain shape actually differs by `K`/`C` ratio. (b) An availability-trace sweep (reduced-availability `syn_*` traces) at fixed `K`,`C` vs a `dynamic_kc`-enabled arm, under FluxTune and, if a backprop baseline exists, FeLiX's own selector — compare which shows the bigger delta. **If real, it's a selector-level (FeLiX) mechanism FluxTune inherits for free, not a FluxTune-owned contribution** — the async_oort selector's `dynamic_kc` hook already lives there; a forward-gradient-specific *trigger* (tied to measured SNR/`cos`) tied into that hook could still be FluxTune's own layered policy | 1, untested |
| | **(a) done, partial, 2026-08-13**: real GPU-busy occupancy (`gpu_pass_start_wall`→`gpu_pass_end_wall`, all 100 trainer files) is bursty and **flat across `K`**: median ≈1/8 GPUs busy, `frac(<2 busy)` = 0.551/0.593/0.576, `cv` = 1.05/1.21/1.20 at `K`=10/20/30 — a third confirmation (after commit-rate and staleness) that `K` doesn't move this system. But it's confounded: the delay model (`training_delay_floor_s`=4.0 ≫ `compute_s`≈0.6–1.6) dominates round wall time regardless of `K`, so this metric can't see a sync-vs-async *shape* difference even if one exists — that needs a true round-barrier baseline (`felix_round`/`fedbuff_round`), none of which exists at this scale on disk. **(b) not run.** No backprop trainer exists in this repo at all (`trainer/` has only `forward_training/`; `felix_round`/`felix_it` are `fl_algorithm=FedFwd` too — same forward-mode trainer, different aggregation-rate rule) — the forward-vs-backprop half of H-T cannot be answered empirically here without building one | |

**If a probe-selection stage is retained**, select on something other than `|d|` — three candidates,
all already paid for: **curvature `vᵀHv`** (≈free, one extra `L(θ)` amortised over `P`); **split-half
SNR within the bin** (free, orthogonal to `|d|`); **loss decrease at the step scale** (under trust-ratio
the step size is known ahead, so pick the `v` minimising `L(θ − ρ*‖θ‖v̂)`; 1 pass per candidate).

**Predicted inert, cheap to settle on paper:** block-coordinate probing over `L` blocks. Progress per
commit falls as `1/√L` while budget per commit falls as `1/L`, so progress per unit `B` is **unchanged**
and it needs `L`× the commits. It escapes `√(n/p)` only if the gradient is *unevenly* spread.

---

## P6 — Dead ends — do not retry

**Append-only (R5). Read before proposing anything.** Superseded *numbers*: [P8.1](#p81-superseded-numbers--quote-check).
Process lessons: [P9.3](#p93-process-lessons).

| do not — or, we believed | why it is dead |
|---|---|
| **Lower `η` alone** | Pays 1:1 in progress; a constant `ρ` of any size has `Σρ² = ∞` |
| **Treat it as an aggregation bug — pool harder** | `K` ≥ 20 reaches 0.860 where `η` = 0.002 reaches 0.601 at the same commit: same stability, one free |
| **Retune `var_threshold`** | Its units make the setpoint a per-model constant, *and* it absorbs a heterogeneity floor moving 1.33 → 0.28 across α. Any retune fits one corner of an (α,`K`) grid |
| **Believe the gate is dead, or live only at `K` ≥ 20** | `var ∝ b²‖g‖²/n` — live wherever it can reach its threshold |
| **Sweep `P` under the shipped selection rule** | Measures `E` and nothing else, and raising `P` under selection is *harmful* |
| **Keep selecting probes by `\|d\|`** | `b²/a` = 1 by construction; candidates carry no other structure in `\|d\|`. Top-k with k < P is strictly worse. **C1 refuted as written** |
| **Split-half cosine as a gate or `cos` estimator** | Per-commit sd 1.5e-3 against a 1e-4 signal (SNR 0.07); at matched `N`=200 over 491 commits it returns **−3e-5 ± 1.1e-4**. Even `mean`+`rf`=64 gives SNR ≈ 2 |
| **Wire `n_eff` to a controller** | An identity (1.00 ± 0.01 in 17 arms), blind to directional disagreement |
| **Spend server momentum as a pooling lever** | Correlating steps inflates the norm by `(1+β)/(1−β)`: `√x` progress for `x` budget. **Only pooling that leaves steps independent is free** |
| **Shrink `p` expecting more learning** ("`p` first") | `‖θ_tr‖ ∝ √p` cancels the `cos` gain and `B` never saw `p`. At matched `ρ*`,`N`,`T`: `A` = 1.63 (`rf`=64) vs 1.64 (`rf`=16) |
| **`trainable_scope: adapters_only` as a `p` lever** | No-op — `pre_classifier` is already dropped at `:217`. The remaining lever is adapter width |
| **Size `N` from the closed-form `cos` without `D`** | `N_req ∝ 1/D²` would be 400× what it says, which no cohort can pool |
| **Evaluate `N_req` at `s` = 0.4** | Derived wrong; it was absorbing a shortfall of unknown size *and direction*. Two laws and 17 arms give **`s` ≈ 2.9**, 50× less pool. Mixing `s`=0.4 with a closed-form `cos` made every gate A/B unrunnable |
| **The `K` ≥ 30 / ≥ 51 cohort requirement** | An artifact of `gate_safety_s` = 0.4 left in the yaml. Cost three inconclusive runs and a wasted node |
| **Expect a lower `gate_safety_s` to hold the peak** | `B` contains no `N`, `cos` or `s`, so both `const` arms spend identically and both turn ([P4.2](#p42-g-1b--s-is-efficiency-not-safety)). Cost one arm |
| **Walk `ρ*` up to 0.12/0.15/0.20**, or hunt "a critical `ρ` ≈ 0.09" | **Withdrawn — this program's own recommendation one revision ago.** A horizon artifact; every `ρ > 0` inflates geometrically. Pick `ρ = Λ_req/(T·cos)` |
| **Test a setpoint through a schedule that spends it** | `setpoint` sizes from `ρ*₀` while `rm` anneals the step: realised `ρ/cos` fell 2.84 → 0.95 in 79 commits. Cost G-1's setpoint half |
| **Run `annealed` at `ρ*` ≤ 0.01** | `220627`'s dead zone: `N_req` → 0 by c20, `I` floored at 1, peak 0.394 decaying to 0.274 |
| **Extrapolate `A` per vclock-hour into an accuracy ranking** | `A` accumulates *through* the turn while accuracy falls — G-1's projection ordered its arms backwards |
| **Score stability by the `‖θ‖²` log-log slope** | Bounded above by 1 under trust-ratio **by construction**. Score `B` and `Λ` |
| **Reset `T_res` when `B_max` is re-sensed**, or count it down at all | A receding horizon that never lands (`ρ*`→0 forever); counting down makes `T` an operator input, which §4.6a forbids. `T_res` is a **rate**, not a deadline — buildplan §5, T5 |
| **Clamp `ρ*_t ≤ ρ*₀` under the landing law** | `ρ*₀` is computed from the `ln 2` **prior**, so the clamp pins `ρ*` there and blocks 3.1's re-sense from spending the budget it just measured — it defeats the probe. The correct cap is gate reachability, `s·√(max_iter·K·G_rule/p)` |
| **`T_res` = 500 as the landing-law constant** | Refuted on replay (T5): 2.46 round trips/commit on yahoo, 2.26 on the `ln 2` prior, against a ≥3 gate. **300** passes both. 500 was only ever chosen to reproduce the portfolio's empirical `ρ*`=0.06 (§4.6a), never derived |
| **Judge a landing schedule by how much `Λ` it banks** | `Λ = 2B/s` wherever the gate holds `s` (P3, T5), so **every** schedule banks the same `Λ` at the same `B`. Schedules differ in commits spent, nothing else. Compare at matched `B` or the comparison is empty |
| **Use `top_class_share` as a turnover alarm** | 4/4 degrading arms with median lead +61 evals, but **15 false positives across 17 holding arms**. `loss > ln(num_classes)` has none but fires ~27 evals *after* the drop — a post-mortem, not a monitor |
| **Believe `Φ` decides whether *any* arm holds its peak** | True only for arms that learned; the missing "peak ≥ 0.80" qualifier made G-1's sinking condition unreadable |
| **Slice `test_global.dataset.tensors[:n]` for a reference** | Returns *one client's* Dirichlet shard — `test_index_list` is per-client blocks in client order, never shuffled (`base_data_manager.py:204-216`). Anti-correlated (−0.46) with held-out truth; broke B1, and three failed consistency checks were written up as findings about the optimizer. **Draw with a fixed-seed shuffle over the full set, ≥1024** |
| **Believe the `cos` shortfall is instrumental** | The clean reference reproduces it exactly — `D` = 0.050, invariant to rule, `p`, `N`. Client disagreement and reference noise both quantitatively excluded |
| **Assume probe noise dominates data noise (~40×)** | Never measured, and false: `L` is flat over **64×** in bin size where `1/√B` predicts an 8× fall |
| **H2 bin-size sweep** — bigger bins to recover `D` | Same measurement: `L` flat 9.5–10.9, `S` flat in bin count. An ordinary `√compute` stage |
| **Assume an audit flag is free because it is emit-only** | The `cos` probe backprops 1024 samples per commit: **85 s** vs 1.64 s for the rest of the path. Cost eight arms |
| **Least-squares / min-norm gradient solve** over `{(v_i,d_i)}` | At `P,N ≪ p` it equals the average up to scale |
| **Orthogonalise the `P` probes, or coordinate across trainers** | Random probes are already orthogonal to `1/√p`, and `K·P` = 500 ≪ `p` |
| **Normalise `v` expecting a variance win** | `‖v‖` concentrates to 0.07%. *Do* rescale `h` when `p` changes — hygiene, not a fix |
| **Chase momentum in the probe distribution** | The accumulated trajectory has `cos ≈ 0.1–0.23` after 100 commits: as a control variate, ~5% of the variance |
| **Invest further in ω / inverse-variance weighting** | Two orders of magnitude below the problem; trust-ratio removes ω from magnitude. ω-*freshness* is different and now has something to act on |
| **Shrink `h`** | Pinned between truncation error and fp16 catastrophic cancellation |
| **H1 shuffle / H3 bin-order permutation** | Do not address the mechanism. Parked |

**Two patterns account for most of this table.** The `p` census, the `ρ`/orthogonality reconstruction
and the `cos` reference were each *an instrument whose arithmetic was right and whose input was not*.
`gate_safety_s` = 0.4 and the annealed-schedule row are each *a value that stayed in a config after this
document superseded it*. **Every one was caught by a ratio that came out wrong, never by an accuracy
curve** — the argument for the standing preflights.

**Not costed, not dead — the two ideas that beat the `√(n/p)` barrier.** **Block-coordinate probing**
improves `ρ/cos` by ~`L` while each commit updates `1/L` of the params (analysis first, P5.3).
**Low-rank / subspace probing** needs a good subspace *and* a way to broadcast it.

---

## P7 — The instrument ladder

**R7 in practice.** Each rung is ~3 orders of magnitude cheaper than the next. Every hypothesis starts
at the highest rung that can falsify it; a sim run **confirms**, it never explores.

| rung | what it is | cost | what it answers |
|---|---|---|---|
| **1 · log replay** | Python over `aggregator_*.jsonl` and trainer logs on disk. No GPU | minutes | anything in `ρ`, `‖θ_tr‖`, `N`, `var`, `B`, `Λ`, the JVP distribution, staleness, commit reasons. **The k-sweep, `E[v∥²]`, `n_eff`, both laws and the Φ-stop rule were settled here** |
| **2 · offline rig** | one GPU, real model + real JVP math, **no FL stack** | minutes–1 h | anything needing backprop ground truth or a scaled-`‖θ‖` sweep. **Model §7 and the `cos`-reference defect were found here** |
| **3 · single-process replica** | the whole loop in one process — probes, JVPs, pool `K`, real gate, real server arithmetic. No MQTT, selector or sim clock | hours per trajectory | **interactions**: the feedback loop, the gate drifting out from under itself, A/B ranking, and the `K`/`P` ablations impossible in production because `ρ` and `cos` move at once. **Validation gate non-negotiable:** reproduce the shipped arm's `ρ`, orthogonality ratio, doubling time and rise-peak-collapse first |
| **4 · sim run** | the real stack on the virtual clock. `sim_rate` 3.33 (fluxtune) / 12.6 (fwdllm) ⇒ 4 h vclock ≈ 1.2 h wall | 1–3 h wall/arm | end-to-end confirmation, genuine staleness, real concurrency and timing |
| **5 · real run** | the 8-GPU emulation harness | ~4 h wall/arm | confirming a *winner* only |

**Two standing rules for rungs 2–3.** (1) **Probes import the production code**, so a validated result
transfers as a **config flag, not a rewrite**. (2) **Nothing on rungs 1–3 modifies `trainer/`,
`aggregator/`, or any yaml on the critical path**, and no probe result ships until two independent
instruments agree.

> **Rung 3 has never existed**, which is why every A/B in P4 cost a sim run — build it before the next
> fix. Rungs 1–2 have standing tooling: `expt_scripts/replay_scoring.py` (`B`/`Λ`/`Φ`, plus `--cos`),
> `scripts/probe_reference_quality.py`, `scripts/probe_inflation_damage.py`, `scripts/probe_data_noise.py`.

**Reuse handles** — why rungs 2 and 3 are affordable at all:

| handle | where | what it buys |
|---|---|---|
| `create_model` | `expts/initializer.py:61` | the exact production DistilBERT + AdapterHub model, backbone frozen |
| `build_model(num_labels, seq)` | `scripts/profile_jvp_opt.py:64` | that call with the real `ClassificationArgs` (adapter PEFT, fp16, seq 192, batch 8) |
| `calculate_jvp`, `functional_get_loss` | `trainer/forward_training/fwdgrad_utils.py:105,66` | the real central-FD JVP math, importable standalone |
| `stage1_vmap_fd` | `scripts/profile_jvp_opt.py:143` | all `P` probes in one batched pass — **verified bit-identical** to production |
| `calculate_var`, `calculate_snr`, `calculate_cos_sim` | `fwdgrad_utils.py:186,243,349` | the real commit-gate statistics |
| `TextClassificationDataManager.load_federated_data` | `data_manager/text_classification_data_manager.py` | the real agnews H5 partitions at `niid_label_clients=100_alpha=1` |
| `_server_update_step` / `_apply_weighted_update` | `aggregator/FedSgdAggregator.py:238,311` | the exact server arithmetic (~15 lines) to mirror |
| `_emit_server_update` | `FedSgdAggregator.py:338` | `‖Δθ‖`, `‖W‖`, `η` per commit under `server_update_audit` |
| `characterize_variance_curve.py`, `audit_weight_redundancy.py`, `plot_run.py` | `expt_scripts/` | the streaming-telemetry idiom — **including the `data_id`-cycling workaround**: `data_id` cycles per round and must never be a reducer key |

**Rung 3 cost.** Per upload the trainer does `2P` = 20 forward passes (batch 8, seq 192); one commit
pools `N ≈ 185` ⇒ **≈ 3,700 passes/commit, ≈ 700k for a 189-commit trajectory.** `stage1_vmap_fd`
batches all `P` probes into ~2 effective passes and there is no orchestration overhead. If still too
slow, shorten `seq` and/or reduce `N`: **the replica's constants will shift; its mechanism and its
ranking of fixes will not.**

**Sim caveats.** Trainers do real forward-grad compute; what sim removes is the *waiting*. (a) **Compare
sim to sim** — several reference constants were measured on real runs. (b) Ordering differs between
modes, so pool composition differs commit-to-commit: harmless for `ρ`, `cos`, `B`, `Λ`, **not** safe for
a claimed A/B win of a few percent.

---

## P8 — Reproducing any number from logs

All from `lib/python/examples/fwdllm/experiments/`; no GPU. **Everything in the ledgers comes from two
telemetry events:**

```
server_update : trainable_weight_norm, trainable_delta_norm, rho, pool_size, var_at_commit,
                n_eff_ratio, split_half_dot, split_half_norm_a, split_half_norm_b,
                cos_ground_truth, pooled_norm, probe_grad_norm
agg_eval      : acc, mcc, loss, logit_norm, pred_entropy, top_class_share
```

Emit flags, off by default, emit-only, wrapped so they cannot fault training: **`--server-update-audit`**
· **`--pool-split-half-audit`** (its own flag: it adds a pass over params × uploads) ·
**`--cos-ground-truth-audit`**. `logit_norm`, `pred_entropy`, `top_class_share` and the direct `‖θ_tr‖`
are correctness-or-free and **on by default since 2026-08-07** — older runs lack them.

```bash
RUN=run_20260810_042027_fluxtune_n100_smoke_syn_0_sim   # the cos-probe mean/N=200 arm
# telemetry is ~700 MB per run; slice first (~1 min):
grep -hE '"event": "(server_update|agg_eval)"' $RUN/telemetry/aggregator_*.jsonl > /tmp/$RUN.jsonl

grep 'All JVPs sorted by magnitude' $RUN/*trainers.log     # all P candidates, per selection event
#   "model version" in that line is actually the ROUND; count `tensor(` to read P (a trainer override,
#   never in aggregator_config.json). Normalise each event by its own rms, then:
#   E[v_par^2 | coin top-2] = mean( (d1^2+d2^2)/2 / mean(d^2) )   -> 2.988 (P=10) / 4.744 (P=30)
#   top-k average objective = mean( mean(top-k d^2)/mean(d^2) )*k -> monotone, 3.81 .. 10.00

grep -o '\[IterProgress\] data_id=.* force_commit_planned=[A-Za-z]*' $RUN/*aggregator.log
#   split on data_id change; last row of each bin is the commit.
#   reason = CAP if iter >= max_iter-1 else natural if var < 0.3 else plateau
```

```python
ortho  = (tw[b]**2 - tw[a]**2) / sum(dn[a+1:b+1]**2)   # per 25-commit block; expect 1.000 +- 0.005
rho_t  = dn[t] / tw[t]                                 # == the logged `rho`
step   = dn[t]                                         # flat => ||theta||^2 linear
N_t    = K * pool_size[t]                              # pool_size == iterations, exactly, every record
GG     = pooled_norm / probe_grad_norm    # vs b*sqrt(p/N); b = sqrt(E) select, 1/sqrt(P) mean

# NEVER average per-commit split-half cosines: one commit's cosine sits under the 1/sqrt(p) ~ 1e-3
# sampling floor while the signal is ~3e-4. The record carries the RAW components for this reason
# (split_half_dot, split_half_norm_a, split_half_norm_b, pool_size) -- pool them per ARM:
cos_sh = sqrt( 2 * sum(split_half_dot) / sum(split_half_norm_a * split_half_norm_b) )
# Simulation at the real p, N confirms this identity and cos = sqrt(N/p) to 0.1%. It is also why the
# split-half GATE is dead (P6) -- the same arithmetic, one commit at a time, has SNR 0.07.

# THE SCORING NUMBERS. All exact at any horizon; G_rule = 2.988 select / P mean.
B   = 0.5*sum(log1p(rho[t]**2) for t in range(0,T-1))     # budget spent -- SUM OVER t = 0 .. T-2,
Phi = exp(B)                                              # the steps that lie between tw[0] and tw[-1]
Lam = sum(rho[t]*sqrt(G_rule*N[t]/p) for t in range(T))   # progress banked, in units of ||theta_tr||
A   = sum(rho[t]*sqrt(G_rule*N[t]/p)*tw[t] for t in range(T))   # ABSOLUTE progress -- compare across p
# N[t] = K*(pool_size or iteration_per_data_id+1) -- pool_size is absent when I == 1 (P4.4)
# Under server_momentum beta > 0, Phi = exp( (1+beta)/(1-beta) * B ) -- steps are no longer orthogonal.

# COS AUDIT. D = cos_ground_truth / sqrt(G_rule*N/p) is the data-side shortfall, ~0.050 and invariant
# to rule and p. Split it: L = (pooled_norm/probe_grad_norm) / (b*sqrt(p/N)) is the length excess and
# equals rms|d|/||g_probe||; S = cos*pooled_norm/probe_grad_norm/a is the shadow. D = S/L.
```

`expt_scripts/replay_scoring.py` does all of the above for a run dir, including `--cos`.

**The `p` census** (needs the `test_fwdllm` env, ~1 min) — for `p` and the layer split only; in-run,
`[ProbeDim]` logs the same `p`, so this is usually unnecessary:

```bash
cd /home/dgarg39/flame/lib/python
/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python -c "
import sys, torch, math; sys.path.insert(0,'.')
from examples.fwdllm.scripts.profile_jvp_opt import build_model
m = build_model(4, 192)
m.add_module('pre_classifier', torch.nn.Sequential())   # the line the trainer runs at :217
tr = [x for _,x in m.named_parameters() if x.requires_grad]
p = sum(x.numel() for x in tr)
print(p, torch.sqrt(sum((x.detach().float()**2).sum() for x in tr)).item(), 0.01*math.sqrt(p))"
# -> 450340  13.325  6.711   <- production. Without the add_module line: 1040932 / 20.356 / 10.203
```

**Telemetry gotcha.** Before 2026-08-07, `tc_transformer_trainer_distribute.py:485` logged the argmax
under the label `chosen jvp` while the actual pick is the coin-flip result. It now logs the coin-flip
winner as `chosen jvp`, the argmax as `max jvp`, the index as `chosen idx`. **Runs before that date
carry the old, mislabelled field.**

### P8.1 Superseded numbers — quote check

If you see one of these in `fluxtune_contributions.md`, `FLUXTUNE_CODE_QA.md`, or a paper draft, **it is
wrong.** A lookup table for *numbers*; the refuted *ideas* are [P6](#p6--dead-ends--do-not-retry).

| superseded | replaced by |
|---|---|
| `p = 1,040,932`, `‖θ_tr‖` init 20.356 | **450,340 / 13.35** — the trainer drops `pre_classifier` before probing (P1) |
| "freezing `pre_classifier` buys 2.31×" | **already banked** — the layer is not in `p` |
| `ρ` = 0.115 flat all run; orthogonality ratio 1.032 | **`ρ` = 0.16 at commit 1**, falling iff `N` grows; ratio **1.000 ± 0.005**. Both were artifacts of using total `‖W‖` |
| "the variance gate is 100% dead" / "live at `K` ≥ 20" | **live wherever `2b²‖g‖²/n` can reach 0.3**: live at `K`=10/α=100, cap-bound at `K`=20/α=0.1 |
| `ρ·√N` = 1.68–1.81 is the pooling invariant | true **only at fixed α**; the invariant is `ρ√N‖θ_tr‖/(η·rms\|d\|·√p)` = 0.53 ± 0.01 |
| `n_eff` is the scale-free replacement for `var_threshold` | **an identity** — the step rule carries the portability claim |
| `cos` = 0.0351 predicted; `cos ≤ 0.015` from split-half | both retired, and so is their replacement — next row |
| `cos` measured 0.0004–0.003; shortfall **25–80×**; `a` = 0.007–0.09; "`cos` rises 11× at fixed `N`,`p`,rule"; "`ρ/cos` anti-orders the arms" | **ALL VOID — instrument defect.** Measured against one client's non-IID shard. Replaced by `D` = 0.050 (P3.1) |
| "the norm ratio `‖G‖/‖g‖` survives the B1 defect, matching `b√(p/N)` to 5–10%" | **Also void.** It matched only because a 64-sample reference gradient (‖g‖ = 3.1–8.0) has roughly the norm of an 8-sample *client* gradient. Against a clean 1024-sample reference (‖g‖ = 0.27–0.34) the ratio is **8.5–9.5× high**: `L` = 0.95 / 1.11 / 1.01 on `042027`/`013806`/`065837` |
| "`Λ` orders every arm" | true at fixed `p` only; across `p` use `A` |
| "`p` is the best cost/benefit lever in the stack" | **inert at a pinned `ρ*`** — a memory/compute knob |
| `ρ/cos ∝ p` | **∝ √p** — `ρ` is `p`-invariant |
| `s = 0.3–0.5` | **2.6–4.3** derived; **2.9** measured |
| "there is a critical `ρ ≈ 0.09`" | a horizon artifact |
| `exp = 0.55` | **0.25** — sized to the horizon; 0.55 spends the budget in the dead zone |
| "4 h minimum or the result is uninformative" | score `B` and `Λ` — exact at any horizon, readable in ~20 commits |
| collapse = logit saturation | **directional degeneracy**; `logit_norm` does not discriminate |
| "never converges — it *oscillates*" (`fluxtune_contributions.md` §8) | at 4 h it is a **monotone rise then monotone divergence** |
| QA §D2 "the k sweep cannot run today" | answered offline from logged JVPs — monotone, optimum `k = P` |
| QA §E1 "measuring `cos(G,g)` needs `v_k` uploaded" | a server-side backprop gradient on a probe batch suffices |
| the accuracy plateau is ≈0.865 | **≈0.876** (P4 read 1) |

---

## P9 — Preflight, launch, and process lessons

### P9.1 Preflight

**Four preflights run in `run_sequential.sh`** (CPU-only, ~4 s, refuses to launch on failure);
`_node_lib.sh` aborts a node on any arm producing <5 commits:

- `test_model_args_parity.py` — every arg the trainer reads unguarded is supplied by **both**
  `trainer/main.py` and `aggregator/main_fedfwd_agg.py`.
- `test_commit_gate.py` — `N_req`'s closed form and its annealed-vs-setpoint divergence.
- `test_cos_probe.py` — cos = ±1 on a pool built from `g` itself, **and** the reference batch is not
  class-skewed (the B17 guard).
- `test_weight_decay.py` — `auto` = `ρ²/2`, pins `Φ` = 1, leaves frozen params and the disabled path
  untouched.
- `test_landing_law.py` — C-1's law C and its stop: `const`/`rm` byte-identical, `B(t)` tracks
  `B_max(1−e^{−t/T_res})` to ≤1%, the `ρ_max` cap holds `⌈n_req/K⌉ ≤ max_iter`, a `B_max` re-sensed below
  the spend gives `ρ*`=0, the stop latches, and only `halt` sets `_work_done`. **Its last check is the
  cross-instrument one:** the live aggregator reproduces `replay_landing_law.py`'s pre-registered
  two-phase enactment on both datasets to 4 d.p.

> **Two environment requirements, both of which have cost a debugging session.** These scripts shell out
> to `run_sequential.sh`, which needs **`FLAME_CONDA_ENV=test_fwdllm`** (base lacks `h5py`) and
> **`FWDLLM_FD_SCALE_INVARIANT=1`** (fluxtune is `rf`=64, so the FD-rescale preflight refuses without it).
> Without either, every case exits 2 for a reason that has nothing to do with what is being tested.

**The wall-clock budget check that cost eight arms now exists** — `expts/wall_clock_preflight.py`, task
0.7, run from inside `--dry-run`, refusing the launch when projected real wall exceeds
`sim_wall_ceiling_s`. **It is a prior, not a guarantee:** it prices law C off the `ln 2` prior, so an arm
can clear the ≥3 trips/commit gate at launch and breach it in flight once `B_max` is sensed (P4.7 defect
3). Read trips/commit **per quintile at run time**.

**Read the enactment checks before the science:**

```bash
grep -m1 '\[ServerStep\]'    $RUN/*aggregator.log   # rho* must equal the logged rho
grep -m1 '\[CommitGate\]'    $RUN/*aggregator.log   # n_have vs n_req per iteration
grep -m1 '\[CosProbe\]'      $RUN/*aggregator.log   # batch size, dominant-class share, then cos
grep -m1 '\[probe_combine'   $RUN/*trainers.log     # trainer-side; NOT in aggregator_config.json
grep -m1 '\[ProbeDim\]'      $RUN/*trainers.log     # THE p (post pre_classifier drop)
grep -m1 '\[FD\] spacing'    $RUN/*trainers.log     # THE h -- read h*sqrt(p) HERE, not off ProbeDim
```

> **`[ProbeDim]`'s `h*sqrt(p)` is not the FD spacing** — it prints the *nominal* `h`, so the `p` ladder
> read 6.71 / 4.79 / 3.44 and looked like the FD had shrunk with `p`. It had not: `[FD]` showed `h`
> rescaled to 0.01 / 0.014023 / 0.019507, holding `h√p` at 6.7107 on all three arms.

### P9.2 Launcher gotchas

Each costs a wasted run.

- **`--only`**, not `--baselines` (unknown args abort).
- **`--yes`** — otherwise each invocation stops at an interactive `[y/N]` prompt.
- **`--clean`** — back-to-back runs otherwise `DIRTY_ABORT` on a prior run's stray workers.
- **`--force`** — the sim-charge-profile preflight blocks when audit-on reals are newer than the profile.
  Confirm with **`--dry-run`** that it is the only `✗` first; `--force` overrides every check.
- **`--num-trainers 100`** pins `minInitialTrainers`, so varying `--c` does not move the warmup threshold
  underneath a sweep.
- The baseline → yaml map is **hardcoded** in `ALL_RUNS`; no custom-yaml flag, and `fwdllm_plus` has no
  entry. Use config flags, never a hand-edited yaml.
- **Pin the pool for any A/B** — `--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off`
  ⇒ `I` = 20, `N` = 200 exactly.
- **Budget the cos audit into the wall clock, or stride it first.** Per commit, measured on
  `_apply_weighted_update`: `035045` **1.64 s** (audit off) · `042027`/`065837` **6.5 s** (batch 64) ·
  `151530`/`171950`/`002208`/`022448` **85 s** (batch 1024). The cost is **linear** in the reference
  (77–81 ms/sample at both sizes), so chunking buys peak memory and nothing else — the lever is a
  **stride**. On `002208` the audit is **96% of the run's real wall** and kills the run through
  `[SIM_WALL_CEILING]` while the vclock still reads healthy. **A low `I` makes it worse linearly**, since
  the cost is per *commit*.
- **Never redirect a node script's output under `/nethome/dgarg39/`** — that home quota is
  space-restricted and node logs are large (tens–hundreds of MB per arm). `run_sequential.sh` already
  writes each run's config/telemetry/`*.log` under `experiments/run_*/` by construction; let it, and read
  results from there, not from a hand-redirected `~/nodeN.log`.

### P9.3 Process lessons

*Method only. Results are P3–P4; refuted beliefs are [P6](#p6--dead-ends--do-not-retry).*

**Worked**

- **Sweep, never sample** — every surviving law was confirmed by a slope, and `ρ ∝ 1/√K` died because a curve exposed feedback a point would have hidden.
- **Pin `N` for any A/B** — unpinned, you measure the gate re-spending the fix, not the fix.
- **Vary the axis every earlier sweep held fixed** — the α-sweep cost four arms, no code, and moved three things.
- **Measure an `O(1)` quantity, not a cosine** — at `p` = 4.5e5, norms answer where cosines cannot.
- **Pre-register a sinking condition, not just a point prediction.**
- **Inject the endpoint instead of waiting for it** — now also how `B_max` gets sensed.

**Did not work**

- **Scale-free-by-measurement as a strategy** — it produced an identity (`n_eff`); portability has to come from the step rule.
- **Deriving a setpoint from theory, twice** — `s` = 0.3–0.5 was absorbing a data-side shortfall (`D`).
- **Scoring features without scoring compositions** — gate and anneal are each correct and multiply into a stall (`N_req ∝ ρ_t²`).
- **Letting a superseded constant live on in a config** — when a doc supersedes a constant, grep the configs in the same edit.
- **Launching without a napkin check on the operating point, three times** — assert `⌈n_req/K⌉ ≤ max_iter` and echo the `s` used.
- **A wall cap chosen without reference to the question** — six arms at 66–74 commits answered every matched-commit question and no peak question.
- **An emit-only flag never re-costed after being made correct** — B17 multiplied the audit's per-commit wall by 16×, and the vclock looks healthy right up to the moment the runaway safety fires.
- **A sinking condition without its precondition** — G-1's fired on arms at `Φ` = 1.03 that never learned.
- **Subtracting two accumulated quantities without checking they share an origin** — `B_max` was measured from `θ_t` and `B` from `θ_0`; the expression type-checks, runs, and is meaningless (P4.7 defect 2). Write the origin next to the formula.
- **Adding a writer to shared run state without grepping the existing ones** — one of `_work_done`'s two writers *assigned* where it should have OR'd, and silently un-set a stop that had already fired (P4.7 defect 1).
- **Treating a special value as a missing value** — `ρ` = 0 meant "no `ρ` yet" to the gate, so the cheapest possible commit became the most expensive one, 20 round trips for a zero-length step (P4.7 defect 3).
- **Trusting a dataset sweep that greps for the dataset's name** — `total_data_bins` = 150 is agnews' `1,200/8`, correct by coincidence and invisible to `grep agnews`, and it fed yahoo 8.6% of its data for a whole node (P4.7 defect 4). Grep the *derived* constants, and grep `lib/python/flame/` too.
- **Reading a preflight as a guarantee rather than a prior** — trips/commit passed the launch check off the `ln 2` prior and then breached the same ≥3 gate mid-run once `B_max` was sensed. A gate that matters is a run-time metric, not only a launch-time one.
- **A sinking condition without a smoothing rule** — raw crossings put both G-1b turns at `Φ` ≈ 2.95; smoothed, 3.43 / 3.79.
- **Testing a setpoint through a schedule that spends it** — a node testing a boundary must hold the run at it.
- **Extrapolating a progress rate as an accuracy rate** — extrapolate `A` only alongside `Φ`.
- **Final accuracy as an A/B statistic** — ±0.045 between byte-identical replicates against ±0.0009 at peak.
- **Reaching a rig's operating point by the wrong road** — at acc 0.57 a backprop model reads `‖g_test‖` = 1.95 against every arm's 0.25–0.32; match on the quantity the measurement depends on.
- **Trusting a new instrument because it was unit-tested** — a probe needs a test that its *input* is what you think it is.
- **A missing attribute costing a full night**, then a quieter repeat that emitted a wrong number instead of crashing — both now preflighted.
- **Attributing a run's death to a plausible mechanism instead of a measured one.** `003648`'s
  `[SIM_WALL_CEILING]` was written up twice — as the cos-audit tax (P4.5) and as "a schedule that spends
  its budget unevenly" (buildplan §4) — before anyone counted its round trips. Both were wrong: `I` was
  floored at 1 on 98% of commits, and one `grep` of `pool_size` on telemetry already on disk would have
  said so on the day. **A death gets a ratio, like every other claim in this document, before it gets a
  sentence.**
- **Sizing a controller constant to reproduce an empirical value.** `T_res` = 500 exists only because it
  returns `ρ*` = 0.062 against the 0.06 the portfolio found by search (§4.6a) — a back-fit, and it failed
  the first composition test it ever faced. A constant that has never been checked against anything but
  the number it was chosen to match is not evidence.
- **Scoping a task from its pseudocode summary tag alone.** §5.1's `[NOT BUILT]` on 3.1's `PHASE B` line
  reads as "no method exists" if you stop there — it doesn't, until you also read §5.5b two sections down,
  which says the method is fully specified *and already coded* (`probe_inflation_damage.py`, used for the
  whole B-1 sweep). What's actually unbuilt is the live-loop wiring, a materially smaller task. Almost
  scoped Phase 3 as blocked-on-undesigned-research on this alone (2026-08-15) — read the full section a
  pseudocode line points at before sizing the work behind it.

---

## P10 — Coverage and transfer: what has never been tried

[P3](#p3--knob-ledger) records what each *measured* knob did and [P6](#p6--dead-ends--do-not-retry) what
died. This answers what has **never been tried**, and which findings are federated versus general.
`GEN` = any perturbation-based trainer · `FL` = federated-specific.

| knob | stage | scope | what is known | next |
|---|---|---|---|---|
| **`P` under `mean` beyond 10** | trainer | GEN | **P-1 landed (2026-08-15): compute-bound**, not invariant — `τ(30)/τ(10)`=2.56 crosses the registered kill line. Round trips did fall `÷3` as predicted (`I` 3→1, floored); the "3× less wall clock" half of the old prediction does not hold | report and stop — do not build adaptive `P` (3.4) on a free-compute premise; a bandwidth case for `bytes÷3` remains open |
| **adaptive `P`** | trainer | GEN | same hill-climb as `K`, but `P` also cuts bytes. Needs a **mid-run `P` change**, which no code path supports, and a per-commit `G_rule` | model §5.5e · P5.2 phase 3.4 |
| **adaptive `K` / `C`** | selection | **FL** | **K-C landed (K-1, 2026-08-13, see P3): `C` buys the wall clock, `K` doesn't** at fixed availability. `dynamic_kc`'s `k_max` = 15 is backwards regardless. Whether `K` matters under *variable* availability, or more for forward- vs backprop-trained gradients, is untested — H-T (P5.3) | hill-climb `C` (3.4); H-T for the availability/forward-vs-backprop split |
| **staleness / freshness weighting** | aggregation | **FL** | genuine at `C` ≥ 60. The **only** cost of running the cohort wide, so its price — `D(·)` — is what caps it | **C3's freshness half**, the one untried paper claim |
| **`B_max` across models** | — | GEN | transfers across model capacity (`rf`=16 vs 64, §7.1), does NOT transfer across task (B-1, 2026-08-13: agnews/yahoo/yelp-p erratic, not monotone in `num_labels`) | **3.1**, now mandatory |
| **probe selection on anything but `\|d\|`** — curvature `vᵀHv` · split-half SNR · trust-region | trainer | GEN | the three candidates that are *not* stability-neutral, all already paid for; specified in [P5.3](#p53-open-hypotheses) | **H-H**, then future work |
| **block-coordinate probing** | trainer | GEN | costed on paper and predicted inert (P5.3), needs `L`× the commits | model §8 — free in MoE |
| **low-rank / subspace probing** | trainer | GEN | needs a good subspace *and* a way to broadcast it | the other `√(n/p)` escape |
| **precision (bf16 / fp32)** | trainer | GEN | sets the usable `h` window, so it gates H-S | future work |
| **adapter placement / PEFT family** | model | GEN | `‖θ_tr‖ ∝ √p` is what makes `p` inert, and it holds for *adapter-style* init | **the assumption most likely to break elsewhere** |
| **model / task** | — | GEN | every OTHER constant is fitted on DistilBERT/agnews — `B_max`, the one constant on the operating path, is now measured erratic across task (B-1) | **3.1** covers `B_max`; the rest is future work |
| **ω / freshness magnitude** | aggregation | **FL** | ω spans 0.70–0.87 against a ≥10× gap; trust-ratio removes it from magnitude | **C3's magnitude half — parked** |
| **`C` concurrency** | selection | **FL** | caps `K`; **never varied independently, and every `K` result is confounded with it** | the axis K-1 must **hold fixed** |
| **sim compute-vs-delay crossover (SCT parity)** | trainer | **FL, sim-only** | `FedSgdTrainer.py:683-685` charges vclock as `max(real_gpu_time_s, delay_s)` per round — the availability-delay model (floor `training_delay_floor_s`=4.0s) currently dominates and hides real GPU compute (`compute_s`≈0.6–1.6s at `P`=10/30, measured on `003628`/`015455`) inside it, so `sct` (the simulated-completion timestamp the aggregator orders updates by) tracks the delay model, not raw compute. **Untested: once real compute exceeds the delay floor** (higher `P`, a bigger model, or a tighter/lower-floor availability trace), `sct` starts tracking real GPU time directly instead — whether aggregator ordering, `commit_gate`, and `wall_clock_preflight.py`'s flat `τ(K)` model still hold in that regime is unknown. Not a bug today; flagged for whenever `P` is raised past the P-1 default | revisit before raising `P` in production, or when moving to a tighter availability trace. Not urgent — parked |

**Two reads.** *(1)* **The trainer-side knobs are the generic ones, and the most interesting are
untried** — curvature selection and `P` under averaging; both rung 1–2, neither needs the FL stack.
*(2)* **Everything calibrated rather than derived is calibrated on one model and one task** — but D4 and
the time law shrank that debt to **one constant, `B_max`**, the cheapest to measure.
