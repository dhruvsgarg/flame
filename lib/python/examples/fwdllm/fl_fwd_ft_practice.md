# Forward-gradient fine-tuning — implementation, evidence, and how to run it

> **Companion to [fl_fwd_ft_solution.md](fl_fwd_ft_solution.md)**, which owns the *why*: the failure,
> its mechanism, the symbols, the model and what is unknown. Sections there are `§0…§8` and are cited
> here as "model §x".
>
> **This document owns everything a run produced**: the shipped configuration, every feature built and
> its flag, every knob measured, every arm run, the dead ends, the instruments, and the launch and
> replay procedure. Per R2, the model doc cites these numbers rather than restating them.
>
> Three ledgers are the single source of truth: **P1** (what is built), **P3** (what each lever did),
> **P4** (what each arm scored). No status tags anywhere else.

---

## P1 — What is built

### P1.1 The system under test

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

**Where the probe dimensions live** (**MEASURED**), and it matters everywhere. `create_model` builds
1,040,932 trainable params, but for distilbert the trainer's `__init__` then runs
`self.model.add_module("pre_classifier", nn.Sequential())`
(`tc_transformer_trainer_distribute.py:217`) — **replacing that layer with an empty module before any
probe is drawn** — and `self.params` comes from `make_functional_with_buffers` *after* that:

```
                         create_model     PRODUCTION (post-:217)
pre_classifier              590,592          0        <- dropped, not frozen
adapters                    447,264    447,264
classifier                    3,076      3,076
p                         1,040,932    450,340       <- the number every cos uses
||theta_tr|| at init          20.33        13.33
h*||v|| = h*sqrt(p)          10.203        6.711
```

### P1.2 Feature ledger

**Single source of truth for implementation status.** Every feature is flag-gated, default = old /
byte-identical off. Terminal state (`PERMANENT` / `FLAGGED` / `REVERTED`) is the **operator's** call —
ask, with the A/B evidence. Nothing reaches `WORKS` without predicted-vs-observed numbers and a run id.

| # | feature | flag (default = old) | status |
|---|---|---|---|
| **B2** | **average all `P` probes** | `probe_combine: {select\|mean}` | **WORKS · 5/5 arms** (P3). Site: `_accumulate_mean_over_probes`. Also fixes the `P=1` crash (`sorted_indices[-2]` on a 1-element list) and the RNG-stream mismatch that blocked the C1 ablation |
| **B3** | **trust-ratio step + `t^-exp` anneal** | `server_step_rule` + `rho_star` / `rho_schedule` / `rho_exp` | **WORKS mechanically and portably; setpoint open** (P3). Site: `FedSgdAggregator._apply_weighted_update` (pool-then-apply). Subsumes S3: ω can no longer influence step magnitude |
| **B5** | **shrink `p` via adapter bottleneck** | `adapter_reduction_factor` (16) + `FWDLLM_FD_SCALE_INVARIANT` | **WORKS · 3/3** (P3). Read by `expts.initializer.create_model` on **both** sides; `trainable_scope` stays, inert, so old configs parse |
| **B4** | scale-free `n_target` commit gate | `commit_gate: {var\|n_target}` + `gate_safety_s` + `gate_rho_ref` | **INCONCLUSIVE, twice** (P3). Closed form exact; composition bug found and fixed; both A/Bs cap-bound. **Re-run at `K` = 30** |
| **B1** | `cos(G,g)` ground-truth probe | `cos_ground_truth_audit` + `cos_probe_batch_size` | **BROKEN AS SHIPPED · must be re-run** (model §6). The norm ratio `‖G‖/‖g‖` survives (reference-direction-free); **every `cos_ground_truth` value is void** |
| **B17** | **fix the `cos` probe reference** | `cos_probe_batch_size` (**default now 1024**, was 64) | **BUILT** (model §6). Fixed-seed permutation over the whole test set; logs dominant-class share and warns above 0.5. `test_cos_probe.py` fails the launch on a class-skewed reference. Emit-only. **Unblocks H-P; nothing re-measured yet** |
| **B18** | S1 · heavy-ball on the pooled **direction** | `server_momentum: BETA` (default 0.0 = byte-identical) | **BUILT, ARM IN FLIGHT.** Was applied to the already-scaled update, inflating `ρ` by `1/(1−β)` — the arithmetic behind the original S1 NaN. Now applied *before* the trust-ratio scale, so `ρ = ρ*` to 1e-7 at β up to 0.9 |
| **B7/B15** | weight decay `λ ≈ ρ²/2` | `server_weight_decay: auto\|FLOAT` (default off) | **BUILT, ARM NOT RUN — demoted.** Decoupled decay on the trainable slice after the step; `auto` = `ρ²/2` from the realised `ρ`. Preflight: `Φ` = 1.029 with decay vs 2.258 without at `ρ`=0.09. **Model §7.2 answered Q2 on the rig**, so this is an optional confirmation |
| **B11** | `n_eff` sensor | rides on `server_update_audit` (emit-only) | **FAILED as a sensor, KEPT as an audit** (P3). Do not wire it to a controller |
| **B12** | α-sweep | `--partition-method ...alpha={0.1,1,100}` | **WORKS · 4/4** (model §2.6) |
| **B8** | staleness histogram from the K-sweep | none (log replay) | **BLOCKED — data deleted.** Surviving runs emit `staleness`/`inflight_staleness` but only at `K`=10/`c`=30. Needs the K/C re-run |
| **B6** | long `K`=50 arm, ≥32 h vclock | none (runtime only) | **DEMOTED** — `B` now extrapolates the answer exactly (model §4.1) |
| **B13** | two-batch `cos` control | — | **CANCELLED — answered offline** by `scripts/probe_reference_quality.py`. Superseded by B17 |
| **B9** | split-half commit gate | `commit_gate: {var\|cos}` | **DEAD** at `p`=450k (P6) |
| **B10** | adaptive `P` per client | `probe_budget: adaptive` | **PARKED** — same wall as B9, worse (P6) |
| **B16** | `Λ` out of sample: rule × `p` at pinned `ρ*`,`N` | `probe_combine` × `adapter_reduction_factor` | **TODO** — this is Q3, pre-registered (P5) |

---

## P2 — The recommended stack

| element | setting | rests on | confidence |
|---|---|---|---|
| combination rule | `probe_combine=mean` | P3, 5/5 arms, prediction hit to 1.5% | **settled** |
| step rule | `server_step_rule=trust_ratio` | P3, enacts to 8.7e-5; α-portable to 6 s.f. | **settled** |
| anneal | `rho_schedule=rm`, `rho_exp=0.25` | P3, three arms, 313–318 commits | **settled** |
| `p` | `adapter_reduction_factor=64` **+ `FWDLLM_FD_SCALE_INVARIANT=1`** | P3, 3/3, 0.859 at a quarter of `p` | **settled** |
| `ρ*₀` | read off P4 by target `Λ`, then `ρ = Λ_req/(T·cos)` | model §4.4 design rule | **empirical** — the formula route is blocked on model §6 |
| commit gate | `commit_gate=n_target`, `gate_rho_ref=setpoint`, `s`=0.4 | closed form exact; A/B inconclusive twice | **open** — needs `K`=30 |
| `K`, `C` | `K` ≥ 30, `C` ≥ `K` | `N_req` at ρ*=0.06 is 1,013, five times what `K`=10 can pool | **open**, and a *requirement*, not a preference |
| monitors | `‖θ_tr‖`, `ρ`, `top_class_share`; score `B` and `Λ` | model §2.8, §4 | **settled** |
| `cos` audit | `cos_ground_truth_audit` on; `cos_probe_batch_size` ≥ 1024 (default) | model §6 sizing | **fixed, unvalidated** — H-P |

---

## P3 — Knob ledger — what each lever actually did

One row per concept. "predicted" is the model §5.2 lever table; **"measured" is what happened.**

| knob | predicted effect | **measured** | verdict |
|---|---|---|---|
| **`η` server LR** | `ρ ∝ η`, pays 1:1 in progress | `ρ` 0.2004 → 0.0404 → 0.0101 over `η` .01/.002/.0005, **to 1%**. `η`=0.002 reaches 0.601 @ c120 / 0.815 @ c327 vs `K`≥20's 0.860 | **works as physics, FAILED as a fix** — full 1:1 cost, and `Σρ²` still diverges |
| **`K` cohort width** | `ρ/cos ∝ 1/K`, progress invariant | `ρ·√N` invariant to 4% over `K` 10→50; the *naive* `ρ ∝ 1/√K` **falsified** (gate returns the gain as fewer `I`). `K`≥20 holds **0.860** where `K`=10 collapses | **works**, but `K` is not a knob on `N` — the gate absorbs it. Set `K` and the gate together |
| **`I` iterations/bin** | `ρ/cos ∝ 1/I`, but serial | at `K`=50 the gate cut `I` 18.5 → 5.9 — 3× fewer round trips at slightly larger `N`, best peak of the 08-07 portfolio (0.861), held | **works — buy `N` with `K`, not `I`** |
| **`P` probes, under selection** | stability-**neutral** (`b²/a = 1`); more `E` ⇒ *faster* divergence | `E` 2.988 (P=10) → 4.744 (P=30) over 37k events; predicted `ρ` ratio 1.260, **measured 1.236 (2%)**. `P`=30 learns faster per commit and **collapses sooner** (doubling 57 vs 68) | **prediction confirmed, including its harmful direction.** Never sweep `P` under selection again |
| **combination rule** (`select`→`mean`) | `ρ/cos ∝ 1/P`; `ρ` down `√(E·P)` = 5.466× | at matched `N`=200: `ρ` **5.386×**, `‖Δθ‖` **5.430×** (1.5%). `‖θ_tr‖` end 60.7 → 14.5; `‖θ‖²` growth rate **106×** lower (prediction was 30×; the effect compounds) | **WORKS — 10× in `ρ/cos` for zero extra compute, bytes, or `η`/`N`/`p` change** |
| **top-k averaging** | `b²/a = 1/k`; monotone, optimum `k = P` | offline over 34,447 events: random 1.00× / coin-top-2 1.73× / top-1 1.95× / top-3 2.72× / **all-10 3.16×** in `cos` gain, at `ρ/cos` 1× / 1× / 1× / 3× / **10×**. All cost the same 20 passes | **settled offline — use all `P`.** The shipped rule is k=1, i.e. **zero** stability gain |
| **probe distribution** | is there a "good probe" to find? | `d_i` indistinguishable from iid `N(0,‖g‖²)`: top-1 of 10 → 3.811 observed / 3.798 synthetic; coin top-2 → 2.991 / 2.987 | **NO — closes off all cleverer `\|d\|`-based selection**, and double-duty validates FD linearity at the large chord |
| **`p` (adapter `rf`)** | `cos ∝ 1/√p`; `ρ/cos ∝ √p` | `rf` 16/32/64: peak **0.852 / 0.857 / 0.859**, `‖θ_tr‖` end 78.1 / 31.6 / **20.8**, `Φ` 5.74 / 3.28 / **3.04**. `ρ` ratio **1.00 / 0.967 / 0.928** (the `0.71/0.51` prediction is **falsified** — hence `√p`) | **WORKS — best cost/benefit in the stack.** A quarter of `p` learns agnews at least as well. Must run with `FWDLLM_FD_SCALE_INVARIANT=1` |
| **step rule** (`raw_sgd`→`trust_ratio`) | makes `ρ` an operator constant; removes `\|JVP\|` scale and the α multiplier | `ρ = ρ*` to **8.7e-5**; `(1+ρ*²)^{T/2}` predicts `‖θ_tr‖` to **0.013%**; **α moves `ρ` by 0 to 6 s.f.** | **WORKS.** But **alone it does not bound `‖θ‖`** — constant `ρ*` is still geometric. Ships with the anneal |
| **anneal** (`rho_schedule=rm`, `t^-exp`) | Robbins–Monro: `Σρ=∞`, `Σρ²<∞` | enacts to **4.5e-4** over 318 commits. `exp`=0.55 takes `ρ` **17× below setpoint** by c186 → arm flat at 0.34. **`exp`=0.25 is measured adequate** despite being formally outside the RM window | **WORKS; `exp`=0.25.** The RM bound is conservative at `T`≈300. Size the exponent to the **horizon** |
| **`ρ*` setpoint** | peak monotone in `ρ*` | 0.03/0.06/0.09 → peak **0.695 / 0.801 / 0.846**, all still climbing at cutoff, all stable (`Φ` ≤ 1.145). **The anneal spends most of the setpoint**: realised mid-run `ρ` is an order of magnitude below the best free-gate arm's | **band not closed.** Read `ρ*` off P4 by `Λ`, not by `ρ` — and note model §4.3(a): "walk `ρ*` up" is withdrawn |
| **commit gate** (`var`→`n_target`) | scale-free `N` target replaces a unit-carrying threshold | `N_req` closed form **exact** (28.1 at ρ*=.01; **1,013.3** at ρ*=.06). Both `setpoint` arms sat at the `I` cap for all 312 commits ⇒ byte-equivalent to a hard pin | **INCONCLUSIVE twice.** `s` and `K` are **coupled**: at ρ*=.06 the criterion demands 5× what `K`=10 can pool. A **cohort-width requirement**, not a gate parameter. Re-run at `K`=30 |
| **gate `ρ` reference** | — | **composition bug**: `N_req ∝ ρ_t²`, so annealing `ρ` makes the gate demand *less* pooling and progress decays as `ρ²`. Observed: `N_req` → 0 by c20, `I` floored at 1 on 1,271/1,273 commits, peak 0.394 *decaying* to 0.274 — reproduced identically at α=0.1 | **fixed by `gate_rho_ref=setpoint`** (sizes `N_req` from `ρ*₀`). No-op under `const` and `raw_sgd` |
| **`annealed` vs `setpoint` gate** | — | at **matched commits** `setpoint` wins every column; over the same **vclock** `annealed` gets 2.3× more commits and ends higher (0.821 vs 0.804) | **not right vs wrong** — `annealed` is progress-per-wall-clock, `setpoint` is progress-per-commit. Belongs to the controller, not the gate |
| **α heterogeneity** | enters via `‖g‖` only | model §2.6 — 1000× in α, `var` floor 1.33/0.64/0.28, invariant 0.53 ± 0.01, both `K` predictions falsified | **understood and neutralised** by the step rule |
| **`n_eff` sensor** | dimensionless pooling-adequacy sensor | `n_eff = 2·mean(d²)/var`. Synthetic: recovers true `n`, detects redundancy, scale-invariant over 100× in `‖g‖`, **FAILS on directional disagreement** (4/20/100 distinct directions all → 1.00). In 17 real arms: **`n_eff/N` = 1.00 ± 0.01** | **an identity, not a measurement.** In a gate it reduces to a counter |

---

## P4 — Arm ledger — every arm ever run

**Single source of truth for run results.** Sorted by `Λ` (progress banked), which is the ordering that
predicts peak accuracy. `Φ` pred is `e^B`; `Φ` obs is `‖θ_T‖/‖θ_0‖` — their agreement *is* the norm law.
`‖θ_0‖` = 13.35 (`rf`=16), 9.6 (`rf`=32), 6.86 (`rf`=64). All arms are α=1, `mean`/`rf`=16 unless the
row says otherwise.

| `Λ` | arm | run | `T` | `ρ` c1 | `B` | `Φ` pred → obs | peak | final |
|---|---|---|---|---|---|---|---|---|
| 0.008 | `n_target` rm ρ*=.01 e=.55 **@ α=0.1** | `001008` | 1263 | 0.0100 | 0.0002 | 1.000 → 1.000 | 0.388 | 0.264 |
| 0.008 | `n_target` s=0.4 rm ρ*=.01 | `220627` | 1273 | 0.0100 | 0.0002 | 1.000 → 1.000 | 0.394 | 0.274 |
| 0.009 | `var` gate, rm ρ*=.01 | `200209` | 1249 | 0.0100 | 0.0002 | 1.000 → 1.000 | 0.377 | 0.282 |
| 0.016 | `rm` ρ*=.02 e=.55 | `223510` | 186 | 0.0200 | 0.0007 | 1.001 → 1.001 | 0.379 | 0.329 |
| 0.068 | `const` ρ*=.01 | `211736` | 186 | 0.0100 | 0.0092 | 1.009 → 1.009 | 0.488 | 0.429 |
| 0.199 | `rm` ρ*=.03 e=.25 | `013843` | 318 | 0.0300 | 0.0150 | 1.015 → 1.015 | 0.695 | 0.662 |
| 0.369 | `mean` raw, N=200 | `211800` | 179 | 0.0346 | 0.0850 | 1.089 → 1.088 | 0.804 | 0.793 |
| 0.393 | gate `setpoint` ρ*=.06 | `035557` | 312 | 0.0599 | 0.0592 | 1.061 → 1.061 | 0.804 | 0.804 |
| 0.398 | `rm` ρ*=.06 e=.25 **(Q3 anchor)** | `035045` | 317 | 0.0599 | 0.0597 | 1.061 → 1.062 | 0.801 | 0.799 |
| 0.398 | gate `setpoint` ρ*=.06 **@ α=0.1** | `062213` | 317 | 0.0599 | 0.0597 | 1.062 → 1.062 | 0.775 | 0.770 |
| 0.460 | gate `annealed` ρ*=.06 (`N` 200→40) | `013917` | 707 | 0.0599 | 0.0913 | 1.096 → 1.096 | 0.821 | 0.819 |
| 0.591 | `rm` ρ*=.09 e=.25 | `060834` | 313 | 0.0896 | 0.1332 | 1.142 → 1.145 | 0.846 | 0.843 |
| 0.631 | `mean` raw, N=200 **+cos** | `042027` | 312 | 0.0344 | 0.1434 | 1.154 → 1.158 | 0.849 | 0.849 |
| 0.819 | `raw_sgd` control (`select`) | `200325` | 177 | 0.1837 | 1.4172 | 4.12 → 4.23 | 0.855 | **0.672** |
| 0.837 | `select` raw, N=200 | `200242` | 179 | 0.1861 | 1.4687 | 4.34 → 4.47 | 0.853 | **0.761** |
| 0.891 | `select` `rf`=16 (`p` ladder) | `200358` | 195 | 0.2008 | 1.7146 | 5.55 → 5.74 | 0.852 | **0.353** |
| 0.937 | **`mean` raw, free gate +cos** | `065837` | 715 | 0.1202 | 1.2810 | 3.60 → 3.63 | **0.865** | **0.862** |
| 0.953 | `mean` raw, free gate (replicate) | `223446` | 693 | 0.1202 | 1.2776 | 3.59 → 3.62 | 0.864 | 0.850 |
| 1.014 | `select` `rf`=32 `p`=229012 | `212009` | 187 | 0.1941 | 1.1647 | 3.20 → 3.28 | 0.857 | 0.851 |
| 1.353 | `select` raw, N=200 **+cos** | `013806` | 328 | 0.1861 | 2.2086 | 9.10 → 9.47 | 0.855 | **0.251** |
| 1.368 | `select` `rf`=64 `p`=118348 | `222817` | 188 | 0.1863 | 1.0939 | 2.99 → 3.04 | **0.859** | 0.852 |

*The 4 surviving α-sweep arms were replayed in `Λ`/`B` as an **out-of-sample test of both laws**:
`225718` Λ=1.021 peak 0.828 (pred 0.865), `012201` Λ=0.685 peak 0.865 (pred 0.854), `001241` Λ=0.634
peak 0.848 (pred 0.851), `023623` Λ=0.605 peak 0.868 (pred 0.850) — **mean |peak error| 0.017**, the one
large miss being α=0.1, which carries the known 3-point α penalty. All four sit at Λ ≥ 0.6, so this
confirms the law where it was already strong and does **not** touch the steep region (that is Q3). The
whole 08-07 K/η/P portfolio was deleted from disk and cannot be replayed.*

**Read three things off this table:**

1. **Peak accuracy is monotone in `Λ`**, saturating at ≈0.865 by `Λ ≈ 0.95`, with no exception outside
   the ±0.005 replicate band. This is the dose-response curve you size `ρ*` from.
2. **Whether an arm holds its peak is decided by `Φ`, and by nothing else** (model §4.2).
3. **Efficiency `Λ/B = 2cos/ρ` spans 20× across the portfolio.** `mean` at `N`=200, `ρ`=0.03 banks
   `Λ` = 0.63 for `B` = 0.14 (11% of budget); `mean` under a free gate at `ρ`=0.086 spends **9× the
   budget for +0.016 accuracy**. **The best arm in the portfolio is the least efficient one** — only
   visible in these coordinates.

### P4.1 Scoring rules for any A/B

**Replicate spread** (`200242` vs `200325` are byte-identical configs; `223446` was repeated as
`065837`):

| | `ρ` c1 | `ρ` c40-80 | `‖θ_tr‖` end | peak acc | final acc |
|---|---|---|---|---|---|
| `200242` vs `200325` | 1.3% | 5.0% | 5.7% | **±0.0009** | **±0.045** |
| `223446` vs `065837` | 0.0% | 0.0% | 0.5% | **±0.0007** | ±0.012 |

**Score on peak accuracy and the stability columns, never on final accuracy of a diverging arm** —
post-turnover trajectories are chaotic and the spread is 50× wider there than at the peak.

### P4.2 The interaction that shapes every A/B

`var ∝ b²‖g‖²/n`, so averaging's 30× cut in `b²` puts `var` under `var_threshold` = 0.3 at iteration 1:
the gate commits at `I ≈ 1–2` instead of 18.5, and `N` collapses from 185 to ~10–20. Since
`cos = √(G_rule·N/p)`, the 10× gain in `G_rule` and the ~10× loss in `N` **cancel** — what averaging
buys under the shipped gate is **wall clock (~6.5× fewer serial round trips), not aim.**

1. **"`ρ` drops 5.45×" holds only at matched `N`.** Score **`ρ·√N`**, which is `N`-free.
2. **`mean` and the var gate cannot both be free.** Either pin the pool (`--var-threshold 0
   --max-iter-per-data-id 20 --var-stopping-policy off` ⇒ `I` = 20, `N` = 200 exactly) or the arms are
   unmatched by ~10× in `N`.

**Scoring gotcha:** `_pool_split_half_stats` returns `None` for a pool of one, so at `I = 1` the
`server_update` record carries **no `pool_size` and no split-half components**. Reconstruct
`N = K·(iteration_per_data_id + 1)`, which is always present.

---

## P5 — In flight, and the run queue

### P5.1 In flight

*Delete a row the moment its run lands: the result moves to P4 and its question moves to Settled or Dead
in the model doc's §0.*

| node | arm | settles | prediction, registered before launch | early kill check |
|---|---|---|---|---|
| **3** | `select`/`rf`=16, then `mean`/`rf`=64 — both at `ρ*`=0.06, `exp`=0.25, `N` pinned 200 | **Q3** + **H-P** (first valid `cos`) | peaks **≈0.70** and **≈0.857** against the 0.801 anchor (`035045`). `cos` should come back ≈0.036 / 0.067 and stop drifting within an arm | `[CosProbe]` dominant-class share ≈0.25, not 0.75 |
| **4** | `K`=30/`c`=60, then `K`=50/`c`=100, `n_target`+`setpoint` | **C2** (cohort width) + **H-E** (staleness) | `N_req` = 1,013 is now reachable, so the gate should finally produce a contrast instead of sitting at the `I` cap | `[CommitGate]` `n_have` must rise above the cap |
| **2** | `mean`, `ρ*`=0.06, `exp`=0.25, `N` pinned 200, **`β` = 0.5** | **S1** (momentum as temporal pooling) | `B` **unchanged at 0.060**; `cos` **×1.41**; peak **0.801 → ≈0.84**. Control is `035045`, byte-identical at `β`=0 | `B` at commit ~20: if it moved, `ρ` is not pinned — kill the arm |

**S1's sinking condition:** peak ≤ 0.81 at unchanged `B` ⇒ no gradient component persists across
commits, and temporal pooling is dead for this method. Note `Λ` as defined is computed from config and
**will not move** with `β`; if accuracy rises at unchanged formula-`Λ`, the progress law needs a
`√(1/(1−β))` term.

### P5.2 Q3's pre-registered predictions

Written before launch; `Λ` from `Σρ_t·√(G_rule·N/p)`, peak from P4's calibration:

| arm | `G_rule` | `p` | `Λ` | **predicted peak** | status |
|---|---|---|---|---|---|
| `mean`, `rf`=16 | 10 | 450,340 | 0.398 | 0.801 | **anchor — already measured at 0.801** (`035045`) |
| **`select`, `rf`=16** | 2.988 | 450,340 | **0.218** | **≈ 0.70** | a −0.10 swing from the *rule alone*, at pinned `ρ` and `N` |
| **`mean`, `rf`=64** | 10 | 118,348 | **0.776** | **≈ 0.857** | a +0.056 swing from `p` alone, at pinned `ρ` and `N` |

**Sinking condition: if `select` lands near 0.80 rather than 0.70, `G_rule` does not enter progress**,
`Λ` is `Σρ_t·√(N/p)` at best, the `2P` probe budget buys nothing measurable, and averaging loses its
quantitative case.

### P5.3 Order of operations, cheapest-decisive first

0. ~~B17 — fix the `cos` probe reference.~~ **Landed**; every arm below now carries a valid reference,
   so **H-P comes free with them.**
1. **Q3 (rule × `p` at pinned `ρ`, `N`).** Three arms, pre-registered above. Converts `Λ` from fitted to
   predictive — the last thing between the model and a usable sizing rule.
2. **The K/C sweep**, which carries **both** the gate re-run at `K` = 30 (C2, a throughput question)
   **and** the staleness histogram (H-E) lost when the 08-07 logs were deleted.
3. **M1 on the rig** (split-half over probes vs over data). Minutes, no node, and it is the only
   evidence behind the `P`-before-`K` priority order.
4. **Optional:** the weight-decay arm as a real-trajectory confirmation of model §7.2; the curvature
   probe (H-H); `fwdllm_plus`, which has never launched.

**Every arm reports `B` and `Λ`**, `B` as a fraction of `B_max` and `Λ` against P4's calibration. Both
are exact at any horizon, so the two failure modes are diagnosable before the run ends: **too little `Λ`
= never learned; too much `B` = learned and then lost it.**

### P5.4 Open hypotheses

*Resolved ones are folded into the model doc and deleted from here: H-A (root cause), H-B (`|d|` growth
is genuine gradient growth), H-C (the FD is not the driver), H-D (collapse is directional degeneracy),
H-F (the defect is shared with sync fwdllm), H-G (`p` can be cut without costing accuracy), H-I (the
orthogonality ratio is stationary), H-K (α acts through `‖g‖`), H-L (`n_eff` is an identity), H-N
(the budget is relative — model §7.1), H-O (`‖θ_tr‖` is a symptom — model §7.2).*

| ID | hypothesis | how it gets tested | rung (P7) |
|---|---|---|---|
| **H-P** | With a shuffled, ≥1024-sample reference, the measured `cos` matches `(a/b)√(N/p)` — i.e. the closed form was right all along and the shortfall was entirely instrumental | **B17.** One `mean` and one `select` arm at pinned `N`=200. Predicted: `cos` ≈ 0.067 / 0.036, `ρ/cos` orders the arms, and the 11× within-arm drift disappears. If `cos` is still ≪ closed form with a clean reference, **the shadow model needs rebuilding** | 4 (rides on any arm) |
| **H-E** | Staleness stays ≤ 1 **only because `N` is small**; raising `K`/`C` produces genuine staleness | The 08-07 K-sweep logs were deleted, so the free replay is gone. Surviving runs are all `K`=10/`c`=30 and can only establish the baseline. **The K/C sweep has to be re-run** | 1 baseline, 4 sweep |
| **H-J** | `K ≥ 20` **defers** collapse rather than preventing it; collapse at commit 1,200–1,700 | Largely *predicted* now — `B` extrapolates exactly. **Demoted to a confirmation** unless a cheap node is free | 1, then 4 |
| **H-H** | The FD's discarded curvature term `vᵀHv` carries usable signal | The central difference uses only the *difference* of `L(θ±hv)`; their **sum** is `≈ h²vᵀHv` — **already computed and thrown away.** Log it, correlate with realised loss decrease at the step scale actually taken. Unlocks the only probe-selection metric that is free and *not* stability-neutral | 2 |

**If a probe-selection stage is retained at all**, select on something other than `|d|` — three
candidates, all already paid for: **curvature `vᵀHv`** (≈free, one extra `L(θ)` amortised over `P`);
**split-half SNR within the bin** (compute `d` on each half of the 8-sample bin, select on agreement —
free); **loss decrease at the step scale** (under trust-ratio the step size is known in advance, so pick
the `v` minimising `L(θ − ρ*‖θ‖v̂)` — trust-region rather than derivative selection; 1 pass per
candidate).

---

## P6 — Dead ends — do not retry

**Append-only (R5). Read before proposing anything.**

| do not | why it is dead |
|---|---|
| **Lower `η` alone** | Pays 1:1 in progress and does not restore square-summability — a constant `ρ` of any size has `Σρ² = ∞` |
| **Retune `var_threshold`** | Not because it is inert (it controls `N` whenever reachable) but because its units make the setpoint a per-model constant *and* it simultaneously absorbs a heterogeneity floor that moves 1.33 → 0.28 across α. Any retune fits one corner of an (α,`K`) grid |
| **Sweep `P` under the shipped selection rule** | Measures `E` and nothing else; and raising `P` under selection is *harmful* (P3) |
| **Keep selecting probes by `\|d\|`** | Stability-neutral by construction (`b²/a = 1`, measured); candidates provably carry no other structure in `\|d\|`. Top-k average with k < P is strictly worse |
| **Split-half cosine as a gate or a `cos` estimator** | **Measurability wall, measured.** Per-commit sd 1.5e-3 against a 1e-4 signal (SNR 0.07); at matched `N`=200 over 491 commits it returns **−3e-5 ± 1.1e-4**. Even `mean` + `rf`=64 at the measured `cos` gives SNR ≈ 2. The wall is specific to the *cosine*, not the split-half pair. Revisit only if `p` falls by orders of magnitude |
| **Wire `n_eff` to a controller** | An identity (`n_eff/N` = 1.00 ± 0.01 in 17 arms), blind to directional disagreement (P3) |
| **Retry momentum before `ρ` is bounded** | `ρ_eff = ρ/(1−β)`; at β=0.9 that is 1.6 and the norm doubles *every step* — the historical NaN is arithmetic, not a bad idea. **Now unblocked** by the trust-ratio step: at `ρ*`=0.06, β ≤ 0.5 is the testable range (B18) |
| **Build the least-squares / min-norm gradient solve** over `{(v_i,d_i)}` | At `P,N ≪ p` it equals the average up to scale — no gain in `cos` |
| **Orthogonalise the `P` probes, or coordinate probes across trainers** | No-ops: random probes are already orthogonal to `1/√p`, and `K·P` = 500 ≪ `p` |
| **Normalise `v` expecting a variance win** | `‖v‖` concentrates to 0.07%. *Do* rescale `h` when `p` changes — hygiene, not a fix |
| **Chase momentum in the probe distribution** | Computed negative: the accumulated trajectory has `cos ≈ 0.1–0.23` after 100 commits, so as a control variate it removes ~5% of the variance |
| **Invest further in ω-direction / inverse-variance weighting** | Two orders of magnitude below the problem, and the trust-ratio step removes ω from magnitude entirely. ω-*freshness* is different and still open (H-E) |
| **Shrink `h`** | Pinned between truncation error and fp16 catastrophic cancellation |
| **Size `N` from the closed-form table** (model §4.5) | Unverified — the only direct `cos` measurement was instrumental. Size `ρ*` from P4 until H-P closes |
| **Slice `test_global.dataset.tensors[:n]` for a reference batch** | Bypasses the DataLoader sampler and returns *one client's* Dirichlet-skewed shard: `test_index_list` is per-client blocks concatenated in client order, never shuffled (`base_data_manager.py:204-216`). This is what broke B1. Draw any reference with a fixed-seed shuffle over the full set, sized ≥1024 |
| **Use `top_class_share` as a turnover alarm** | Fires on 4/4 degrading arms with median lead +61 evals but **15 false positives across 17 holding arms** — an untrained model is already single-class and the statistic re-crosses mid-training. `loss > ln(num_classes)` has zero false positives but fires ~27 evals *after* the 5-point drop: a post-mortem fingerprint, not a monitor |
| **Score stability by the `‖θ‖²` log-log slope** | Bounded above by 1 under trust-ratio **by construction** — every "sub-linear ⇒ safe" reading on such an arm is vacuous. Score `B` and `Λ` |
| **Walk `ρ*` up to 0.12 / 0.15 / 0.20** | This program's own recommendation one revision ago, **withdrawn**: budget cost for fixed `Λ` is ∝ `ρ`, so raising `ρ*` buys wall clock and spends safety. Pick `ρ = Λ_req/(T·cos)` from the commit budget |
| **`trainable_scope: adapters_only` as a `p` lever** | No-op — `pre_classifier` is already dropped at `:217` (P1.1). The remaining `p` lever is adapter width |
| **H1 shuffle / H3 bin-order permutation** | Do not address the mechanism. Parked |
| **H2 bin size sweep** | Spend compute on probes, not bigger bins — probe noise dominates data noise by ~40× (**ANALYSIS**, and M1 is exactly the arm that would measure it) |

**Not costed, not dead — the only two ideas that beat the `√(n/p)` barrier.** **Block-coordinate
probing** (one adapter layer at a time, `p → p/L` per probe) improves `ρ/cos` by ~`L` while each commit
updates `1/L` of the params; known in ZO optimisation, needs analysis first. **Low-rank / subspace
probing** needs a good subspace *and* a way to broadcast it.

---

## P7 — The instrument ladder

**R7 in practice.** Each rung is ~3 orders of magnitude cheaper than the next. *Every* new hypothesis
starts at the highest rung that can falsify it; a sim run is what **confirms** a hypothesis that already
survived a cheaper instrument.

| rung | what it is | cost | what it can answer |
|---|---|---|---|
| **1 · log replay** | Python over `aggregator_*.jsonl` and trainer logs already on disk. No GPU | minutes | anything expressible in `ρ`, `‖θ_tr‖`, `N`, `var`, `B`, `Λ`, the JVP distribution, staleness, commit reasons. **The k-sweep, `E[v∥²]`, `n_eff`, the norm law and the progress law were all settled here** |
| **2 · offline rig** | one GPU, real model + real JVP math, **no FL stack** | minutes–1 h | anything needing a backprop ground truth or a scaled-`‖θ‖` sweep: `cos(G,g)` vs `N`, `‖G‖/‖g‖`, the collapse endpoint, curvature `vᵀHv`. **The inflation-damage results (model §7) and the `cos`-reference defect were both found here** |
| **3 · single-process replica** | the whole optimization loop in one process — draw probes, compute JVPs, pool `K` uploads, run the real gate, commit with the real server arithmetic. No MQTT, no selector, no sim clock | hours per trajectory | **interactions**: the feedback loop, the gate drifting out from under itself, A/B ranking of fixes, and the `K`/`P` ablations that are structurally impossible in production because `ρ` and `cos` move at once. **Validation gate is non-negotiable:** reproduce the shipped arm's `ρ`, orthogonality ratio, doubling time and rise-peak-collapse first, or nothing downstream counts |
| **4 · sim run** | the real stack with the virtual clock. `sim_rate` 3.33 (fluxtune) / 12.6 (fwdllm) virtual-s per wall-s ⇒ 4 h vclock ≈ 1.2 h wall | ~1–3 h wall per arm | end-to-end confirmation, genuine staleness, anything needing real concurrency or real timing |
| **5 · real run** | the 8-GPU emulation harness | ~4 h wall per arm | confirming a *winner* only. Discover in sim, confirm on real |

**Two standing rules for rungs 2–3.** (1) **Probes import the production code** — never reimplement the
math, so a validated result transfers into the trainer as a **config flag, not a rewrite**. (2) **Nothing
on rungs 1–3 modifies `trainer/`, `aggregator/`, or any yaml on the critical path**, and no probe result
lands as a production change until two independent instruments agree on the number it produces.

> **Rungs 1 and 2 have standing tooling; rung 3 still does not.** `expt_scripts/replay_scoring.py`
> scores any run dir in `B`/`Λ`/`Φ` and audits the cos probe (`--cos`);
> `scripts/probe_reference_quality.py` and `scripts/probe_inflation_damage.py` are the offline rig.
> **Rung 3 has never existed**, which is why every A/B in P4 cost a sim run — build it before the next
> fix, not after.

**Reuse handles that already exist** — this is why rungs 2 and 3 are affordable at all:

| handle | where | what it buys |
|---|---|---|
| `create_model` | `expts/initializer.py:61` | the exact production DistilBERT + AdapterHub model, backbone frozen |
| `build_model(num_labels, seq)` | `scripts/profile_jvp_opt.py:64` | that call wrapped with the real `ClassificationArgs` (adapter PEFT, fp16, seq 192, batch 8) |
| `calculate_jvp`, `functional_get_loss` | `trainer/forward_training/fwdgrad_utils.py:105,66` | the *real* central-FD JVP math, importable standalone |
| `stage1_vmap_fd` | `scripts/profile_jvp_opt.py:143` | all `P` probes in one batched pass — **verified bit-identical** to the production loop, and the reason a replica is affordable |
| `calculate_var`, `calculate_snr`, `calculate_cos_sim` | `fwdgrad_utils.py:186,243,349` | the real commit-gate statistics |
| `TextClassificationDataManager.load_federated_data` | `data_manager/text_classification_data_manager.py` | the real agnews H5 partitions at `niid_label_clients=100_alpha=1` |
| `_server_update_step` / `_apply_weighted_update` | `aggregator/FedSgdAggregator.py:238,311` | the exact server arithmetic (~15 lines) to mirror in a replica |
| `_emit_server_update` | `FedSgdAggregator.py:338` | `‖Δθ‖`, `‖W‖`, `η` per commit under `server_update_audit` |
| `characterize_variance_curve.py`, `audit_weight_redundancy.py`, `plot_run.py` | `expt_scripts/` | the streaming-telemetry idiom to copy — **including the `data_id`-cycling workaround**: `data_id` cycles per round, so it must never be used as a reducer key |

**Rung 3 cost, stated honestly.** Per upload the trainer does `2P` = 20 forward passes (batch 8, seq
192); one commit pools `N ≈ 185` ⇒ **≈ 3,700 passes/commit, ≈ 700k for a 189-commit trajectory.** Two
things make that tractable: `stage1_vmap_fd` batches all `P` probes into ~2 effective passes, and there
is no orchestration overhead. If it is still too slow, shorten `seq` and/or reduce `N` and accept the
trade explicitly: **the replica's constants will shift; its mechanism and its ranking of fixes will
not** — which is exactly what the validation gate licenses.

**Sim caveats that survive.** Trainers do real forward-grad compute (gradient values are mode-invariant);
what sim removes is the *waiting*. (a) **Compare sim to sim** — include a sim baseline replicate, since
several reference constants were measured on real runs. (b) Ordering differs between modes, so pool
composition differs commit-to-commit; harmless for `ρ`, `cos`, `B` and `Λ` (aggregates over 100+
commits), **not** safe for a claimed A/B win of a few percent.

---

## P8 — Reproducing any number from logs

All from `lib/python/examples/fwdllm/experiments/`; nothing needs a GPU. **Everything in the evidence
ledgers comes from two telemetry events**, `server_update` and `agg_eval`:

```
server_update : trainable_weight_norm, trainable_delta_norm, rho, pool_size, var_at_commit,
                n_eff_ratio, split_half_dot, split_half_norm_a, split_half_norm_b,
                cos_ground_truth, pooled_norm, probe_grad_norm
agg_eval      : acc, mcc, loss, logit_norm, pred_entropy, top_class_share
```

Emit flags, all off by default, emit-only, wrapped so they cannot fault training:
**`--server-update-audit`** (the base record) · **`--pool-split-half-audit`** (the split-half
components; its own flag because it adds a pass over params × uploads and must not change the base
record's cost profile) · **`--cos-ground-truth-audit`** (B1's backward pass). `logit_norm`,
`pred_entropy`, `top_class_share` and the direct `‖θ_tr‖` are correctness-or-free and **on by default
since 2026-08-07** — runs older than that lack them.

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

# THE TWO SCORING NUMBERS. Both exact at any horizon; G_rule = 2.988 select / P mean.
B   = 0.5*sum(log1p(rho[t]**2) for t in range(0,T-1))     # budget spent -- SUM OVER t = 0 .. T-2,
Phi = exp(B)                                              # the steps that lie between tw[0] and tw[-1]
Lam = sum(rho[t]*sqrt(G_rule*N[t]/p) for t in range(T))   # progress banked
# N[t] = K*(pool_size or iteration_per_data_id+1) -- pool_size is absent when I == 1 (P4.2)
```

`expt_scripts/replay_scoring.py` does all of the above for a run dir, including `--cos`.

**The `p` census** (needs the `test_fwdllm` env, ~1 min) — for `p` and the layer split only. In-run,
`[ProbeDim]` logs the same `p` from `self.params`, so this is usually unnecessary:

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
winner as `chosen jvp`, the argmax as `max jvp`, and the index as `chosen idx`. **Runs before that date
carry the old, mislabelled field** — any parser must handle both.

### P8.1 Superseded numbers — quote check

Only numbers that were quoted in other documents or drafts before being corrected. If you see one of
these in `fluxtune_contributions.md`, `FLUXTUNE_CODE_QA.md`, or a paper draft, **it is wrong**.

| superseded | replaced by |
|---|---|
| `p = 1,040,932`, `‖θ_tr‖` init 20.356 | **`p` = 450,340, `‖θ_tr‖` = 13.35** — the trainer drops `pre_classifier` before probing (P1.1) |
| "freezing `pre_classifier` buys 2.31×" | **already banked** — the layer is not in `p` |
| `ρ` = 0.115 flat all run; orthogonality ratio 1.032 | **`ρ` = 0.16 at commit 1**, falling iff `N` grows; ratio **1.000 ± 0.005**. Both were reconstruction artifacts of using total `‖W‖` |
| "the variance gate is 100% dead" / "live at `K` ≥ 20" | **live wherever `2b²‖g‖²/n` can reach 0.3**: live at `K`=10/α=100, cap-bound at `K`=20/α=0.1 |
| `ρ·√N` = 1.68–1.81 is the pooling invariant | true **only at fixed α**; the invariant is `ρ√N‖θ_tr‖/(η·rms\|d\|·√p)` = 0.53 ± 0.01 |
| `n_eff` is the scale-free replacement for `var_threshold` | **an identity** — the step rule carries the portability claim |
| `cos` = 0.0351 predicted; `cos ≤ 0.015` from split-half | both retired; so is their replacement — next row |
| `cos` measured 0.0004–0.003; shortfall **25–80×**; `a` = 0.007–0.09; "`cos` rises 11× at fixed `N`,`p`,rule"; "`ρ/cos` anti-orders the arms" | **ALL VOID — instrument defect.** Measured against one client's non-IID shard, anti-correlated (−0.46) with held-out truth and sign-swinging with model state. **No trustworthy `cos` measurement exists yet** |
| `ρ/cos ≥ 10×` over budget | came from the broken probe. Against the **closed-form** `cos`, `ρ/cos` is the efficiency `2cos/ρ` and it *does* order the arms |
| `ρ/cos ∝ p` | **∝ √p** — `ρ` is `p`-invariant |
| `s = 0.3–0.5` | **`s` = 2.6–4.3**, derived from the two conserved laws |
| "there is a critical `ρ ≈ 0.09`" | **debunked by arithmetic** — a horizon artifact. Every `ρ > 0` inflates geometrically |
| `exp = 0.55` (strictly inside Robbins–Monro) | **0.25** — sized to the horizon; 0.55 spends the whole budget in the dead zone |
| "4 h minimum or the result is uninformative" | score `B` and `Λ` — exact at any horizon, readable in ~20 commits |
| collapse = logit saturation | **directional degeneracy**; `logit_norm` does not discriminate |
| "never converges — it *oscillates*" (`fluxtune_contributions.md` §8) | at 4 h it is a **monotone rise then monotone divergence** |
| QA §D2 "the k sweep cannot run today" | answered offline from logged JVPs — monotone, optimum `k = P` |
| QA §E1 "measuring `cos(G,g)` needs `v_k` uploaded" | not for this quantity — a server-side backprop gradient on a probe batch suffices |

---

## P9 — Preflight, launch, and process lessons

### P9.1 Preflight

**Four preflights run in `run_sequential.sh`** (CPU-only, ~4 s total, refuses to launch on failure);
`_node_lib.sh` aborts a node on any arm producing <5 commits:

- `test_model_args_parity.py` — every arg the trainer reads unguarded is supplied by **both**
  `trainer/main.py` and `aggregator/main_fedfwd_agg.py`.
- `test_commit_gate.py` — `N_req`'s closed form and its annealed-vs-setpoint divergence.
- `test_cos_probe.py` — cos = ±1 on a pool built from `g` itself, **and** the reference batch is not
  class-skewed (the B17 regression guard).
- `test_weight_decay.py` — `auto` = `ρ²/2`, pins `Φ` = 1, leaves frozen params and the disabled path
  untouched.

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
> read 6.71 / 4.79 / 3.44 and looked like the FD had silently shrunk with `p`. It had not: `[FD]` showed
> `h` rescaled to 0.01 / 0.014023 / 0.019507, holding `h√p` at 6.7107 on all three arms.

### P9.2 Launcher gotchas, each of which costs a wasted run

- The baseline flag is **`--only`**, not `--baselines` (unknown args abort).
- **`--yes`** — otherwise each invocation stops at an interactive `[y/N]` prompt.
- **`--clean`** — back-to-back runs otherwise `DIRTY_ABORT` on a prior run's stray workers.
- **`--force`** — the sim-charge-profile pre-flight blocks when audit-on reals are newer than the
  profile; re-profiling from them would bake diagnostic overhead into the vclock model. Confirm with
  **`--dry-run`** that it is the only `✗` first, since `--force` overrides every check.
- **`--num-trainers 100`** pins `minInitialTrainers`, so varying `--c` does not move the warmup
  threshold underneath a sweep.
- The baseline → yaml map is **hardcoded** in `ALL_RUNS`; there is no custom-yaml flag, and
  `fwdllm_plus` has no entry. Use config flags, never a hand-edited yaml.
- **Pin the pool for any A/B** — `--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off`
  ⇒ `I` = 20, `N` = 200 exactly (P4.2).

### P9.3 Process lessons

*Only entries about **method**. Results live in P3–P4.*

**Worked:**

- **Sweeps over single points.** Every surviving law was confirmed by a *slope*; the one claim that died
  (`ρ ∝ 1/√K`) died because a curve exposed feedback a single point would have hidden. A slope cannot be
  rescued by a fudge factor.
- **Pinning `N`.** Every fix moves a quantity the var gate immediately re-spends (P4.2), so an unpinned
  A/B measures the gate, not the fix.
- **Varying the axis every earlier sweep held fixed.** The α-sweep cost four arms and no code, and moved
  three things — two of them falsifications of our own predictions.
- **Measuring norms rather than cosines.** `‖G‖/‖g‖` and `‖g‖`-vs-`‖θ_tr‖` are `O(1)` ratios and gave
  clean answers from the same probe whose headline cosine is still contested. **At `p` = 4.5e5, build
  the instrument that reads an `O(1)` quantity.**
- **Pre-registering a sinking condition, not just a point prediction.** The `ρ*` node missed both point
  predictions and the read was still unambiguous, because the sinking condition was written down first.
- **Injecting the endpoint instead of waiting for it.** Both inflation blockers (model §7) were settled
  in an afternoon on one GPU by adding noise to a trained model, after months of 4 h trajectories.

**Did not work:**

- **Scale-free-by-measurement as a strategy.** The plan was to make each unit-carrying constant portable
  by measuring the units away; for the gate that produced an identity (`n_eff`), and for `cos` a number
  that failed its own consistency checks. **At this `p` the informative content of pool statistics is
  `O(1/√p)` below their noise, so the portable quantity has to come from the *step rule*, where `ρ` is
  exact.**
- **Deriving a setpoint from theory, twice.** `s` = 0.3–0.5 was never calibrated and was silently
  absorbing a shortfall of unknown size *and direction*. **A third re-derivation would repeat the
  error** — the operating point comes from P4 until H-P closes.
- **Scoring features without scoring compositions.** The gate and the anneal are each correct and
  multiply into a stall (`N_req ∝ ρ_t²`, P3). Nothing in either feature's own prediction could have
  caught it.
- **Launching a portability test without a napkin check on its own operating point.** `N_req` = 1,013
  against a cohort ceiling of 200 was computable in one line, and the risk was named before launch.
- **Final accuracy as an A/B statistic.** ±0.045 between byte-identical replicates vs ±0.0009 at the
  peak (P4.1).
- **Trusting a new instrument because it was unit-tested.** B1's index alignment was unit-tested and
  correct; its *reference batch* was one client's non-IID shard, and nothing tested that. Three
  consistency checks failed and were written up as findings about the optimizer for a full revision
  before anyone asked what the reference actually contained. **A probe needs a test that its input is
  what you think it is, not only that its arithmetic is right.**
- **A missing attribute costing a full night on three nodes** (08-08): twelve of sixteen arms died ~30 s
  in with an `AttributeError` while the chain marched on. **08-09 was the same class, quieter:** a
  feature that emits a *wrong number* rather than crashing. Both are now preflighted (P9.1).
