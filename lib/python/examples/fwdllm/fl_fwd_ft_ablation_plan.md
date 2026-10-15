# FluxTune vs FwdLLM — concept ablation plan (2026-10-15)

## Summary

FluxTune's gain over FwdLLM most likely comes from four concepts: averaging all P probes (E1), the trust-ratio server
step (E2), a fixed commit pool (E3), and async execution (S1, for wall-clock). Grad-aware weighting, dynamic K/C and the
extra controller parts (ρ schedule, budget probe) look inert. This ranking comes from code reading and the measurements
below; no clean ablation has been run yet.

Recommended first step, after the ρ anneal lands (next section): Phase A on DistilBERT + AG News in sim mode. That is FwdLLM plus one FluxTune concept at a time,
7 arms × 3 seeds ≈ 21 runs, scored on time-to-target accuracy. Its results decide which Phase C alternatives are worth
running.

**Open question:** A6 uses the simplified recipe as the full-FluxTune reference. Should it be the full controller instead?

## Before any experiment: anneal ρ (action item)

**How ρ is set today.** `--rho-schedule pool` fixes ρ = s·√(P·N/p) for the whole run (`FedSgdAggregator.py:278`), with
s = 1.5, P = 10, N = 50. Only `p` differs between models. Nothing was tuned per model:

| | `p` | ρ |
|---|---|---|
| DistilBERT (adapters rf=16) | 450,340 | 0.050 |
| SmolLM2-360M (LoRA r=8) | 823,040 | 0.0370 |
| SmolLM2-1.7B (LoRA r=4) | 794,624 | 0.0376 |

- N = 50 matched the full controller's mean pool (49.5) on DistilBERT, giving ρ = 0.050 inside its observed 0.032–0.068
  (session notes §5). s = 1.5 was measured against 1.0 and 2.9 and acts as an efficiency knob, not a safety bound
  (`fl_fwd_ft_practice.md` G-1/G-1b).
- On Llama, keeping N = 50 let ρ fall as 1/√p (~26% smaller steps than DistilBERT). Holding ρ = 0.050 instead would
  need N ≈ 90. Neither choice has been compared.

**Why anneal.** Each step is sideways (|cos(θ_tr, Δθ_tr)| < 0.0006), so ‖θ_t‖² = ‖θ_{t−1}‖²·(1+ρ²) and Φ grows from ρ
alone. A constant ρ is faster per commit early but late commits mostly add Φ (noise): the simplified arm peaked at
Φ 2.68 and dipped before the Φ = 3 rail (session notes §8). The existing `rm` schedule (ρ0·t^−exp) and `landing` don't
fit: `rm` ignores Φ, and `landing`'s probe-driven ρ was mostly noise (session notes rows 14–15).

**Proposal** (cheatsheet §6): keep the absolute step constant, ρ_t = ρ0/Φ_t, so Φ_t = √(1 + t·ρ0²). ρ stays roughly flat
for ~1/ρ0² commits, then decays as 1/√t. No new knob. Suggested start: ρ0 ≈ 0.07, N ≈ 98 on DistilBERT. Φ = 3 then
takes ~1,600 commits, so stop on the stall rule (fix its streak reset first) plus a commit cap.

**Action item (blocks Phase A–D):**

- [ ] Implement `--rho-schedule phi` (ρ_t = ρ0/Φ_t, ρ0 from the `pool` formula or `--rho-star`) in `_rho_star_now`.
- [ ] Fix the stall-rule streak reset (one noisy eval zeroes it) and add a commit cap.
- [ ] A/B on DistilBERT + AG News: simplified vs simplified + `phi` (ρ0 = 0.07, N = 98), cap ~1,000 commits, ≥ 2 seeds.
  Compare time to 0.86 and 0.87, peak accuracy and Φ at the peak.
- [ ] If `phi` wins, it becomes the E2 setting in A2, A6 and the Phase B ladder; the `pool` constant-ρ recipe becomes a
  Phase C alternative.
- [ ] Decide how ρ0 scales to a new `p` (hold N and let ρ0 ∝ 1/√p, or hold ρ0 and set N ∝ p) with one SmolLM2-360M
  pair before the 1.7B arm in Phase D.

## Runs this is based on

Three FluxTune runs, all AG News, sim mode, 100% availability (`syn_0`), training delays on. The two DistilBERT runs were
an A/B; their run folders are deleted, so their values come from `fl_fwd_ft_aishwwarya_session_2026-10-01.md` §3 and
`fl_fwd_ft_smollm2_smoke_plan.md` §9.

| | DistilBERT – full controller | DistilBERT – simplified | SmolLM2-1.7B |
|---|---|---|---|
| Run | `run_20261001_120715` | `run_20261002_000757` | `run_20261014_034646` |
| Model, trainable params | DistilBERT + adapters rf=16, 450,340 | same | LoRA r=4, 794,624 |
| Clients (N / C / K) | 100 / 30 / 10 | 100 / 30 / 10 | 50 / 30 / 10 |
| Partition | 100 clients, Dirichlet α=1 | same | 50 clients, α=1 (milder non-IID) |
| Best accuracy | 0.873 | 0.8726 | 0.832 |
| Commits, time | 855, hit 3 h wall limit | 881, stopped by Φ ≥ 3 at 1.74 h | 399, hit 8 h wall limit |

FluxTune with DistilBERT has also been tested on Yahoo Answers (≈ 0.73, matching the FwdLLM paper) and Yelp Polarity.
Those run folders are not in this checkout.

## FluxTune vs FwdLLM, concept by concept

FluxTune differs from FwdLLM on six concepts (E1–E3, S1–S3). S4 and S5 are the same in both, and the grad-aware weights
(S2′) were only on in the DistilBERT full-controller run.

| # | Concept | FwdLLM | DistilBERT – full controller | DistilBERT – simplified | SmolLM2-1.7B | Standard ML equivalent |
|---|---|---|---|---|---|---|
| E1 | Gradient estimate per client | 1 probe, chosen by alignment with the last gradient | average of P=10 probes | average of P=10 | average of P=10 | SPSA/MeZO (1 probe); guided evolution strategies; multi-sample zeroth-order estimate |
| E2 | Server step | SGD, η = 0.01 | trust ratio, ρ shrinks toward a sensed budget (re-sensed every 150 commits) | trust ratio, constant ρ = 0.050 | trust ratio, constant ρ ≈ 0.038 | LARS/LAMB trust ratio; normalised GD; learning-rate decay |
| E3 | When to commit | variance check ≤ 0.3 | pool size computed from ρ (`n_target`) | every 50 uploads | every 50 uploads | adaptive batch size vs fixed batch size |
| S1 | Execution | sync rounds | async (C=30, K=10) | async | async | FedSGD vs FedBuff |
| S2 | Stale updates | rejected | accepted | accepted | accepted | drop stale vs async SGD |
| S2′ | Staleness / utility weights | none | `grad_aware` | none (uniform) | none (uniform) | FedBuff 1/√(1+s); FedAsync decay |
| S3 | Client selection | random, once per round | Oort | Oort | Oort | uniform vs utility-based sampling |
| S4 | Upload weights | uniform | `grad_aware` | uniform | uniform | FedAvg sample weighting |
| S5 | Dynamic K/C | off | off | off | off | — |
| — | Stop rule | data or time cap | budget stop 0.95 + accuracy-decay stop + 3 h wall | stall rule 0.003 + Φ ≥ 3 + 3 h wall | wall limit only | early stopping |

FwdLLM's probe rule (`tc_transformer_trainer_distribute.py:451-462`): draw 10 random candidates per parameter tensor,
keep the one most aligned with the last committed gradient, run one forward pass with it. With no previous gradient
(alternate data bins) it uses one random probe.

## What is high impact and what is not

E1, E2 and E3 are expected to carry the per-commit gain; S1 carries the wall-clock gain. These are predictions to be
tested by the ablation, not measured results.

| Concept | Expected impact | Evidence |
|---|---|---|
| E1 average of P probes | High | Earlier estimator work measured about 10× better step direction and step-size accuracy vs picking one probe, at the same compute and bytes |
| E2 trust-ratio step | High | Introduced as the fix for FluxTune's earlier peak-then-collapse; bounds how fast the trainable weights grow |
| E3 fixed commit pool | High | Pool size sets each step's noise level; the pool is about 99% effective (`n_eff` 49.5 of 50 in the 1.7B run) |
| S1 async execution | High on wall-clock, about zero per commit | Removes the sync barrier; 30 clients in flight |
| S2 accept stale updates | Medium, only with S1 | Staleness is small: of about 11,000 stale uploads in the 1.7B run, 95% were 1 commit old and none older than 4 |
| S3 Oort selection | Low at `syn_0` | Everyone is always available, so selection has little to choose between |
| E2/E3 controller extras (ρ schedule, budget probe, `n_target`) | Low | Full controller 0.873 vs simplified 0.8726 (single seed); simplified got there in 1.74 h |
| S2′/S4 grad-aware weights | None | Alignment check changes weights by ≤ 0.6%; staleness term by ≤ 8% on 15.6% of uploads (next section) |
| S5 dynamic K/C | None | Off in every run |

All single-seed differences above sit inside the 2.3-point same-seed replicate noise (`parity_floors/fluxtune.yaml`).

## The grad-aware update policy

`grad_aware` (`fwdllm_aggregator.py:957-1000`, `:3964`) gives each arriving upload a weight ω and adds ω × gradient into
the running sum. It was on only in the DistilBERT full-controller run, and measured as essentially inert.

```
ω_k    = base_k × align_k × invvar_k
base_k = 0.4 · 1/(1+s_k)^0.25  +  0.6 · (1.5 − 1/(1+U_k)^0.1)
```

- **base:** staleness × utility (the FeLiX rate, `optimizer/fedbuff.py:101`). s = commits since the client got its
  model; U = batch size × √(mean loss).
- **align:** 1 if the cosine with the running sum of this batch is ≥ 0; otherwise falls linearly to 0 at cosine −1. On
  by default.
- **invvar:** min(1, 0.3 / upload variance). Off by default.

Worked example: a fresh upload (s = 0, U ≈ 8) gets ω ≈ 0.82; the same upload 2 commits stale gets ≈ 0.72.

Why it does little (measured, `FLUXTUNE_CODE_QA.md:203-217`):

- The alignment check fired on 44.8% of 34,410 merges, but every cosine was in [−0.006, 0], so weights moved by ≤ 0.6%.
  Random-probe gradients in ~1M dimensions have cosines near 1/√p ≈ 0.001.
- The staleness term moved weights by ≤ 8%, on the 15.6% of uploads that were stale.
- Under the trust-ratio step, G is normalised, so the overall ~0.82 scale cancels; only small relative differences
  remain.
- ω does not affect commits: the variance check uses unweighted uploads.
- The code comment says ω ≤ 1, but the utility term can reach 1.5, so it can exceed 1.

## Ablation design

Start from FwdLLM and add one FluxTune concept at a time, using the simplified recipe (the one 1.7B runs) as the
full-FluxTune reference.

**Ground rules**

- DistilBERT + AG News, sim mode, for all sweeps. A 1.7B arm costs 8 h.
- At least 3 seeds per arm; same-seed replicate noise is 2.3 points.
- Score on simulated time to reach 0.85 and 0.88, plus accuracy vs commit count. This separates better steps (E1–E3)
  from more steps per hour (S1–S3).
- Use the same stop rule in every arm; the three existing runs stopped for three different reasons.
- When the step rule changes, tune η for plain SGD on a grid, so the comparison is not just a learning-rate mismatch.

**Phase A: FwdLLM + one concept**

| Arm | Added to FwdLLM | How |
|---|---|---|
| A0 | nothing | `--only fwdllm` |
| A1 | E1: average of 10 probes | `--probe-combine mean --perturbation-count 10` |
| A2 | E2: trust-ratio step | `--server-step-rule trust_ratio --rho-schedule const --rho-star {grid}` |
| A3 | E3: commit every 50 uploads | `--commit-gate fixed --pool-target 50 --var-stopping-policy off` |
| A4 | S1 + S2: async, stale accepted | `fedbuff_it_unaware` with FwdLLM's estimator settings |
| A5 | S3: Oort selection | on top of A4 if the sync path won't take Oort |
| A6 | everything = FluxTune simplified | `--probe-combine mean --server-step-rule trust_ratio --rho-schedule pool --commit-gate fixed --pool-target 50 --agg-rate-type uniform --var-stopping-policy off` |

If single arms gain little but A6 jumps, the concepts depend on each other: add the pairwise arms A1+A2, A1+A3 and
A2+A3.

**Phase B: cumulative ladder (headline figure)**

1. FwdLLM
2. \+ E1 (average probes)
3. \+ E2 (trust-ratio step)
4. \+ E3 (fixed pool), still sync
5. \+ S1/S2 (async)
6. \+ S3 (Oort) = FluxTune

Reorder steps 2–4 if Phase A shows a dependency.

**Phase D: does it generalise**

- Rerun only the 2–3 winning concepts' arms on Yahoo and Yelp-P (DistilBERT).
- One FwdLLM-vs-FluxTune pair on SmolLM2-1.7B for the single biggest concept.

## Phase C: alternatives to compare against

For each concept, these are the standard ML and FL alternatives to run once Phase A shows it matters. Items marked
*needs code* are not runnable today.

| Concept | Alternative | Standard name | How |
|---|---|---|---|
| E1 probes | 1 random probe | SPSA / MeZO | `--perturbation-count 1 --probe-combine mean` |
| E1 probes | pick by alignment with last gradient | FwdLLM rule; guided ES | A0 |
| E1 probes | average of P ∈ {2, 5, 10, 20}, equal compute | multi-sample estimate | `--perturbation-count P --probe-combine mean` |
| E1 probes | antithetic (±v) or orthogonal probes | variance reduction | *needs code* |
| E2 step | SGD with tuned η grid | plain SGD | `--server-step-rule raw_sgd --learning-rate {grid}` |
| E2 step | normalised SGD, η·G/‖G‖ | normalised GD | *needs code* |
| E2 step | gradient clipping on ‖G‖ | clipping | *needs code* |
| E2 step | constant vs decaying ρ | learning-rate decay | `--rho-schedule const\|rm\|phi` (`phi` *needs code*, see action item) |
| E2 step | server momentum 0.5 | heavy-ball | `--server-momentum 0.5` (0.9 diverged before) |
| E2 step | server Adam / Yogi | FedOpt | *needs code* (Yogi exists for backprop, not wired into fwdllm) |
| E3 commit | pool N ∈ {10, 20, 50, 100} | batch-size sweep | `--commit-gate fixed --pool-target N` |
| E3 commit | variance check at {0.1, 0.3, 1.0} | adaptive batch size | `--commit-gate var --var-threshold T` |
| E3 commit | pool sweep with ρ scaled to N | linear LR scaling | `--rho-schedule pool` (ρ ∝ √N) |
| S1 async | buffer K ∈ {5, 10, 20} at fixed C | FedBuff | `--k K` |
| S2 staleness | reject / accept / 1/√(1+s) / polynomial | FedBuff, FedAsync | `stalenessPolicy` yaml override; `--agg-rate-type old\|new` |
| S3 selection | random, Oort, Felix, REFL under a real trace | uniform vs utility sampling | `--avail-trace <mobiperf>`; `fedbuff_it_unaware`, `felix_round`, `refl_oort` |
| S2′/S4 weights | uniform vs `grad_aware` vs 1/√(1+s) | FedAvg / FedBuff weighting | `--agg-rate-type uniform\|grad_aware\|old` |

Run S3 only under a non-trivial availability trace with mixed client speeds; at `syn_0` it should tie.

## Predictions, gaps and caveats

**Predictions to check results against**

- E1 and E3 are both variance reduction (more probes vs more uploads per step), so their combined gain should be less
  than the sum.
- Plain SGD with a tuned η may match the trust ratio early, then drift as the trainable weights' norm grows.
- S1 should improve time-to-target, not accuracy per commit.
- S3 should tie at `syn_0`.

**Gaps that need code before some arms can run**

- [ ] ρ0/Φ anneal (`--rho-schedule phi`) and the stall-rule fix: blocks every arm (see "Before any experiment").

- [ ] Seed-matched probe arms: changing P changes how many random draws are consumed, so trajectories diverge for
  reasons unrelated to the combine rule.
- [ ] Launcher flag for `stalenessPolicy` (today a yaml override).
- [ ] Normalised SGD, gradient clipping, antithetic/orthogonal probes.
- [ ] Server Yogi/Adam wired into the fwdllm aggregator.

**Caveats**

- The impact ranking is from code reading and single-seed runs, not a controlled ablation.
- The 1.7B run used 50 clients (milder non-IID) and DistilBERT used 100, so 1.7B vs DistilBERT accuracy is not a
  like-for-like comparison.
- The 1.7B run's config is not the documented `fluxtune_v2` default (`grad_aware`, plateau stop, `n_target`, decaying
  ρ); launch flags switched those to the simplified recipe.
- In fixed-pool mode the log line `[Variance=GOOD] var=0.79 <= thr=0.3` is misleading: the commit was triggered by the
  pool count, not the variance.
