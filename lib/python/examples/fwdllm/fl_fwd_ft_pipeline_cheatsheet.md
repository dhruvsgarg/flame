# FluxTune pipeline — quick reference

FluxTune fine-tunes a model federatedly using forward passes only. Trainers estimate gradients from random
nudges; the server pools them and takes one trust-ratio step per **commit**.

> Derived from the four-doc corpus (`fl_fwd_ft_solution.md` / `practice` / `buildplan` / `writeup`); the corpus
> wins on any disagreement. Code names, parity gaps and telemetry fields:
> [fl_fwd_ft_pipeline_context.md](fl_fwd_ft_pipeline_context.md) (AI context). Session notes:
> [fl_fwd_ft_aishwwarya_session_2026-10-01.md](fl_fwd_ft_aishwwarya_session_2026-10-01.md).

**Operator supplies only:** model + PEFT scheme, rank `rf` (⇒ `p`), compute budget.

---

## 1. The algorithm in 8 steps

Steps 2–4 repeat every round; steps 5–8 are one commit.

| Step | (a) | (b) | (c) |
|---|---|---|---|
| 1. Setup (once) | model + PEFT + `rf` ⇒ `p` | `θ_tr` (‖θ_tr‖≈13.35), `B`=0, `t`=0, bin `b`=0 | knobs `P`, `C`, `K`, `s`, Φ rail |
| 2. Trainers | draw `P` random probes `v_i` | slope `d_i` from 2 forward passes | `u_k = (1/P)·Σ d_i·v_i`, upload |
| 3. Pool | keep `C` trainers busy on bin `b` | `K` uploads → pool, `I += 1` | measure agreement: `n_eff` |
| 4. Gate | evidence needed: `N_req = p·(ρ/s)²/P` | `n_eff ≥ N_req` ⇒ commit | `I ≥ max_iter` ⇒ forced commit; else → 3 |
| 5. Build | direction `G` = weighted pool sum | step size `ρ*_t` from the schedule | `scale = ρ*_t·‖θ_tr‖/‖G‖` |
| 6. Apply | skip if `‖G‖=0` | `θ_tr ← θ_tr − scale·G` | frozen weights untouched |
| 7. Bank | `t += 1` | `B += ½·ln(1+ρ²)`, Φ = e^B | ‖θ_tr‖ grows ≈√(1+ρ²) |
| 8. Stop? | every 150: re-probe `B_max` | stop on stall, decay, Φ ≥ 3 or budget | else next bin, empty pool |

Only `θ_tr`, `B`/Φ and the counters carry over between commits; the pool and `G` are thrown away.
At ρ = 0.05 each commit moves `θ_tr` by 5% of its length, so Φ reaches 3 after ≈880 commits.

## 2. Flowcharts

**Baseline** (full controller):

```
                 ┌──────────────────────────────────────────────┐
                 │ 1. SETUP   p, θ_tr, B = 0, t = 0, bin b = 0  │
                 └──────────────────────┬───────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────┐
     ┌─────────▶ │ 2. TRAINERS  C=30 in flight on bin b         │
     │           │    P=10 probes → central FD → u_k → upload   │
     │           └──────────────────────┬───────────────────────┘
     │                                  ▼
     │           ┌──────────────────────────────────────────────┐
     │           │ 3. POOL   +K=10 uploads,  I += 1             │ ◄─────────┐
     │           │    var, n_eff over every upload since commit │           │
     │           └──────────────────────┬───────────────────────┘           │
     │                                  ▼                                   │
     │                ◇ 4. n_eff ≥ N_req = p·(ρ*_t/s)²/P ? ◇                │
     │                   │ yes (natural)          │ no                      │
     │                   │                        ▼                         │
     │                   │          ◇ I ≥ max_iter (20) ? ◇ ──no────────────┘
     │                   │                        │ yes (cap)
     │                   ▼                        ▼
     │           ┌──────────────────────────────────────────────┐
     │           │ 5. BUILD  G = Σ ω_k·u_k   (grad_aware ω)     │
     │           │    ρ*_t = law C, capped at ρ_max (landing)   │
     │           │    scale = ρ*_t·‖θ_tr‖/‖G‖                   │
     │           ├──────────────────────────────────────────────┤
     │           │ 6. APPLY  θ_tr −= scale·G   (skip if ‖G‖=0)  │
     │           ├──────────────────────────────────────────────┤
     │           │ 7. BANK   t += 1,  B += ½·ln(1+ρ²),  Φ = e^B │
     │           └──────────────────────┬───────────────────────┘
     │                                  ▼
     │                       ◇ t % 150 == 0 ? ◇ ──yes──▶ B_max probe:
     │                                  │ no             B_max = B + ln φ_knee
     │                                  │◄──────────────────────┘
     │                                  ▼
     │   ◇ 8. STOP?  stall │ GL │ Φ ≥ 3 │ B ≥ 0.95·B_max   (latched, this order) ◇
     │                                  │ no                           │ yes
     └──── bin b+1, pool = ∅, I = 0 ◄───┘                              ▼
                                                                      END
```

**Simplified** (proposal; passed the agnews A/B):

```
                 ┌──────────────────────────────────────────────┐
                 │ 1. SETUP   p, θ_tr, t = 0, bin b = 0         │
                 │    ρ = s·√(P·N/p) = 1.5·√(500/450,340)       │
                 │      ≈ 0.050, constant for the whole run     │
                 └──────────────────────┬───────────────────────┘
                                        ▼
                 ┌──────────────────────────────────────────────┐
     ┌─────────▶ │ 2. TRAINERS  unchanged                       │
     │           └──────────────────────┬───────────────────────┘
     │                                  ▼
     │           ┌──────────────────────────────────────────────┐
     │           │ 3. POOL   +K=10 uploads,  I += 1             │ ◄─────────┐
     │           └──────────────────────┬───────────────────────┘           │
     │                                  ▼                                   │
     │                   ◇ 4. pool ≥ N = 50 ?  (I = 5) ◇ ──no───────────────┘
     │                                  │ yes
     │                                  ▼
     │           ┌──────────────────────────────────────────────┐
     │           │ 5. BUILD  G = Σ u_k   (ω = 1)                │
     │           │    scale = ρ·‖θ_tr‖/‖G‖                      │
     │           ├──────────────────────────────────────────────┤
     │           │ 6. APPLY  θ_tr −= scale·G   (skip if ‖G‖=0)  │
     │           ├──────────────────────────────────────────────┤
     │           │ 7. BANK   t += 1,  Φ = (1+ρ²)^(t/2)          │
     │           └──────────────────────┬───────────────────────┘
     │                                  ▼
     │      ◇ 8. STOP?  stall (0.003, 20 evals) │ Φ ≥ 3 (t ≈ 880) ◇
     │                                  │ no                  │ yes
     └──── bin b+1, pool = ∅, I = 0 ◄───┘                     ▼
                                                             END
```

Removed: the `n_eff` gate and `max_iter`, ω weights, law C / `ρ_max`, the `B_max` probe, GL and the budget stop.

## 3. Key terms

| Term | Meaning |
|---|---|
| `ρ` | relative step: `‖Δθ_tr‖/‖θ_tr‖` (0.05 = move 5% of the weight norm) |
| `rho_star` | the configured step (0.06); base of the `const`/`rm` schedules |
| `ρ*_t` | the step allowed at commit `t`, after the schedule |
| `_last_rho` | the step actually taken (= `ρ*_t` under trust-ratio) |
| `ρ_max` | largest step the pool can afford within `max_iter` rounds (`landing` only) |
| `‖θ_tr‖` | L2 norm of all trainable (adapter) weights as one vector, recomputed each commit |
| `B`, Φ | budget spent, `½·Σ ln(1+ρ²)`; Φ = e^B = how much ‖θ_tr‖ has grown |
| `p`, `P`, `s` | trainable params; probes per upload; `s = ρ/cos`, step over alignment (derived, logged; §8) |
| `C`, `K`, `I` | trainers in flight; uploads per round; rounds in this commit |
| `G` | pooled gradient direction |
| `n_eff` | how many independent uploads the pool is worth; below pool size if uploads disagree |
| `N_req` | uploads needed for step ρ: a longer step needs a better-aimed direction (`N ∝ ρ²`) |

**How `n_eff` is measured.** Over every upload since the last commit (≈`K·I`), recomputed each round. It uses
a *proxy* variance: split the pool in two halves, compare their averages. That variance shrinks as 1/N, so
`n_eff` comes out as a count and drops if the halves disagree. The per-upload ("real") variance doesn't
shrink with N, so it can't tell when the pool is big enough. On agnews `n_eff` ≈ pool size: a plain counter.

## 4. Concepts and proposed simplifications

Impact: 🔴 accuracy · 🟠 cost only · 🟢 marginal · ⚪ none. Proposals are not yet in the corpus.

| # | Concept | What it does | Impact | Proposal |
|---|---|---|---|---|
| 1 | Forward gradient | gradient from forward passes only | 🔴 | keep |
| 2 | FD spacing | nudge size `h` | 🟢 | make it relative to ‖θ_tr‖ |
| 3 | Probe combination | average the `P` probes (`mean`) | 🔴 | delete the `select` option |
| 4 | Pooling | signal adds, noise cancels | 🔴 | one fitted constant; drop `n_eff` |
| 5 | Cohort | `C` in flight, `K` per round | 🟠 | fix `N`, tune `C` |
| 6 | Safety factor `s` | don't step further than you can aim | 🟠 | merge with ρ |
| 7 | Commit gate | more evidence for bigger steps | 🔴 | fix the pool, derive ρ from it |
| 8 | `ρ_max` | cap on the step | 🟢 | gone with row 7 |
| 9 | ω weights | weight fresh/agreeing uploads | 🟢 | ω = 1 (measured inert) |
| 10 | Legacy var gate | v1 commit rule | ⚪ | delete |
| 11 | Trust-ratio step | direction from `G`, length from θ | 🔴 | keep |
| 12 | Budget law | Φ tracks weight growth exactly | ⚪ | closed form at constant ρ |
| 13 | Progress law | accuracy tracks spent budget | ⚪ | drop (it is `B` renamed) |
| 14 | `B_max` probe | measures tolerated growth | 🟢 | delete (sensor falsified) |
| 15 | Law C anneal | shrinks ρ over time | 🔴 | constant ρ |
| 16 | Stall stop | stop when accuracy stops rising | 🟠 | sole accuracy stop |
| 17 | GL decay stop | stop when accuracy falls | 🟢 | delete (stall also catches it) |
| 18 | Φ rail | stop at Φ = 3 | 🔴 | becomes a commit cap |
| 19 | Budget stop | stop at 95% of `B_max` | ⚪ | delete (unreachable) |
| 20 | Commit indexing | one data bin per commit | ⚪ | one counter |

## 5. Agnews A/B: what we learned

One run per arm, ±0.005 eval noise: read these as directions, not results. Full numbers: session notes §6 and §8.

- **Same accuracy, faster:** the simplified arm matched the baseline peak (0.8726 vs 0.8728) in 1.74 h vs 3.0 h.
- **`n_eff` never mattered:** it tracked the pool size (0.98–1.03×) in both arms.
- **Baseline ρ was noise-driven:** it sawtoothed 0.032–0.068 as each `B_max` probe reset it.
- **Φ measures steps, not learning:** every step is sideways to θ, so ‖θ_tr‖ grows by exactly √(1+ρ²) per
  commit. With a true gradient cos of ≈ 0.003, that growth is almost all noise. A constant ρ is a constant
  *relative* step: the absolute step grows with ‖θ_tr‖ (0.67 → 2.0), so there is no built-in anneal.
- **ρ matters early, not late:** 0.80 → 0.84 took 129 commits at ρ 0.040 (baseline) vs 76 at 0.050
  (simplified). 0.84 → 0.87 took 410 vs 408. Gains are large up to Φ ≈ 1.8 and near zero past Φ 2.2.
- **Overshoot:** the simplified arm peaked at Φ 2.68, then dipped before the Φ = 3 rail stopped it; one noisy
  eval had reset the stall streak. Idea (untested): decay the streak instead of zeroing it, which would stop
  ≈120 commits earlier.

## 6. Proposal: anneal ρ as ‖θ_tr‖ grows (untested)

Keep the *absolute* step constant, so late commits stop adding noise to Φ:

```
ρ_t = ρ0 · ‖θ_0‖ / ‖θ_t‖ = ρ0 / Φ_t        ⇒   Φ_t = √(1 + t·ρ0²)   (exact while steps stay sideways)
```

- ρ stays about flat for ~1/ρ0² commits, then decays like 1/√t. No new knob: no `T_res`, no `B_max`, no probe.
- Start higher: ρ0 ≈ 0.07 with a fixed N ≈ 98. That gives ρ ≈ 0.057 at commit 100, 0.045 at 300 and 0.030 at
  880, where Φ ≈ 2.3 instead of 3.0.
- **Side effect:** Φ = 3 now takes 8/ρ0² ≈ 1,600 commits, so the Φ rail stops being the stop. Stop on the
  stall rule (fix the streak reset first) plus a commit cap.

| Alternative | Verdict |
|---|---|
| One step-down (ρ/√2 at Φ = 2) | Works, but a cruder version of ρ0/Φ with an extra threshold |
| Constant ρ, grow N ∝ Φ² | Better direction per commit; slower commits and Φ still grows. Backup arm |
| Hold ‖θ_tr‖ fixed (projection or weight decay) | No: adapters are not scale-invariant, and the step never anneals |

**Next test (agnews):** simplified vs simplified + `ρ0/Φ` (ρ0 = 0.07, N = 98), stopping on the fixed stall
rule with a cap of ~1,000 commits. Compare time to 0.86 and 0.87, the peak, and Φ at the peak.

## 7. Second model: SmolLM2-360M (ported, smoke not yet run)

Every scored run so far is DistilBERT. SmolLM2-360M (a small Llama) with LoRA is ported for a smoke test that
checks plumbing, not accuracy. What changes:

- `p` = 823,040 trainable (DistilBERT: 450,340). Gradient quality falls as `p` grows, so expect more commits per
  unit of progress.
- It runs in **bf16**, set automatically for `model_type=llama`: fp16 overflows and the loss is NaN.
- The FD nudge size works out the same as DistilBERT's (chord 0.51), so no FD flag changes.

How to run it, what to read and what to do next: [fl_fwd_ft_smollm2_smoke_plan.md](fl_fwd_ft_smollm2_smoke_plan.md).

## 8. Choosing N, ρ0 and T_res

Three knobs, none tuned per setting. A **cell** is one model + dataset + batch size.

| Knob | Value | Scope | Set by |
|---|---|---|---|
| `N` (uploads per commit) | **50** | per dataset · bs; same for every model | `N ≥ N* ≈ 29·Λ_need²·(ρ0/0.07)²`, rounded up to a multiple of `K` |
| ρ0 (starting relative step) | **0.07** | global | largest step that has run cleanly; ρ_max test R1/R2 pending |
| Φ\* (weight-growth budget) | **2** | global | P0 `exp` ended at Φ 1.93 at its 0.878 peak |
| `T_res` | **283** | derived | `2·lnΦ*/ρ0²` |
| `C` | clients in flight | system | daily availability; keep `C ≤ N` |

Derived, not knobs: `s = ρ0/cos` (pass it as `--gate-safety-s` until ρ0 can be passed directly: 2.10 / 2.41 / 2.81 for
DistilBERT / RoBERTa-L / SmolLM2-1.7B at N 50), and `n_eff = N·(1 + O(P/p))` ≡ N (a correlated-probe alarm, not an input).

```
cos   = √(P·N/p)                      alignment of the pooled update (closed form)
ρ_t   = ρ0·e^{−(t−1)/(2·T_res)}       Φ_∞ = e^{ρ0²·T_res/2} = Φ*
Λ(t)  = Λ_∞·(1 − e^{−t/(2·T_res)})    Λ_∞ = 4·lnΦ*·cos/ρ0     (Λ = Σ ρ_t·cos_t, progress)
N*    = p·(ρ0·Λ_need/(4·lnΦ*))²/P     minimum N so that Λ_∞ ≥ Λ_need
```

The Λ law matches measured Λ to 0.5–0.6% (P0 `exp` c1071, P0b bs 32 c717).

**Why none of the three is tuned:**
- **Accuracy depends on Λ alone.** Three schedules reach 0.86 on AG News at Λ 0.81–0.85, and total uploads to a given
  Λ are ρ-free (time law). Inside the safe range ρ0 changes speed, not accuracy, so it needs a one-sided validity check,
  not a sweep. ρ is unitless, so one value can transfer.
- **N is model-free.** A larger p gives noisier uploads (cos ∝ 1/√p) but a longer absolute step (ρ·‖θ_tr‖, ‖θ_tr‖ ∝ √p),
  so useful movement per commit `ρ·‖θ_tr‖·cos` is p-free and Λ_need ∝ 1/√p. Hence `N* ∝ p·Λ_need²` is constant.
  N above N\* only costs probes; client availability moves C, not N.
- **The late step anneals.** ρ_t decays and Φ stops at Φ\*: the absolute step falls from 0.93 to 0.31 by c1000
  (DistilBERT). Constant ρ grew it 0.67 → 2.0 and overshot.

**Measured** (DistilBERT; `expt_scripts/lambda_need.py <run_dir> --targets …`, 5-eval mean, Λ_need at peak − 0.5 pt;
the logged `progress_lambda` is 2B/s, right only at constant s):

| Dataset · bs | Run | Λ at target | Λ_need | N\* |
|---|---|---|---|---|
| AG News · 8 | P0 `exp` / G1 / P0 `pool` | 0.839 / 0.808 / 0.852 (0.86) | 1.216 / 1.089 / — | 42 / 34 |
| Yahoo · 32 | P0b | 0.830 (0.65) | 1.148 | 38 |
| Yahoo · 8 | P0b | 1.059 (0.65) | 1.363 | 53 |

N = 50 covers all but Yahoo at bs 8 (not run). bs moves Λ_need, so N is per dataset · bs. Λ_need is needed again only
for a new, possibly much harder dataset (N\* ∝ Λ_need²): one DistilBERT run to plateau (early slopes do not predict it).

**Rests on, untested:** (1) ‖θ_tr‖ ∝ √p for the LoRA models: check the first `trainable_weight_norm` against 15.3
(RoBERTa-L) and 17.9 (SmolLM2-1.7B); (2) the same accuracy needs the same absolute movement across models: one point,
SmolLM2-1.7B reached 0.832 on AG News at 0.70× DistilBERT's Λ vs 0.75 predicted; (3) ρ0 0.07 on the large models.
If (1)–(2) fail, N picks up a model factor; if a run reaches Φ\* still climbing, N was too small.

**First-hours abort rule (new model):** Φ off `(1+ρ²)^{t/2}` by > 1%, no accuracy above chance by ~c50, or GL decay
before Φ ≈ 1.5 → halve ρ0 (0.035), quadruple T_res (1,132); N stays.

**Compute check, before a long run.** Λ reachable = Λ_∞·(1 − e^{−T_wall/(2·T_res)}), T_wall = cap ÷ measured s/commit.
Below Λ_need·√(450,340/p) at target ⇒ no N or ρ fixes it: change bs, rank (p) or the cap. Estimate: SmolLM2-1.7B ·
Yahoo at N 50 ~255 s/commit → ~170 commits in 12 h → Λ ≈ 0.25 vs ≈ 0.62 needed. A capped run ends mid-decay
(ρ ≈ 0.05): report the best checkpoint or shorten T_res.
