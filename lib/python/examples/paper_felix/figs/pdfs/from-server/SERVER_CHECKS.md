# Server checks, 2026-10-30

Run root `R=/coc/scratch/dgarg/mlsys27_felix_runs_20261026`. Cells: `s50u` = Syn-50%, `smu` = MobiPerf. FLuX legs = `FB2E2E_<cell>_fx` (cifar), `gs_FB2E2E_<cell>_fx` (speech); FedBuff = `*_ctl_srv`. No results changed. Code vs paper mismatches are flagged **MISMATCH**.

## A. Figures (this folder)
- A1 `fig_architecture.pdf`: 3.25 x 1.95 in, 7 pt text, steps 1-8 as specified. Weight written `(1+τ)^-0.25 u(ℓ)` (see B2: the code is a sum, not a product). Dropped: τ≈c/K, "≤49 s", "Timeout 6-21 min", Mild/Steep sketch, example ledger rows, dashed/solid legend (put it in the caption: dashed = control, bytes; solid = model/updates, MB). Source `paper_felix/sec4_overview.py:fig_architecture`. Two badges sit on the plane border (`figlib.overlaps` warns "touches bar"; cosmetic).
- A2 `fig2_trace.pdf`: "Persistence", "vs persistence" (annotation now reads "Prophet vs persistence: -0.16 to -0.30"), (c) axis "Correlation".
- A3 `make_tables.py` now emits the exact caption; `table_headline.tex` regenerated (caption only).
- `fig5_4_agg.pdf` regenerated from the code's formula (B2).

## B. Design vs code
**B1 FLuX-Pace** (`flame/selector/dynamic_kc_policy.py:FelixCPolicy`, run config `selector.kwargs.dynamic_kc`: policy `felix_c`, `pool: 0.5`, `ramp: true`, `c_min: 10`; `avail`/`phase` off).
- Formula: `bound = min(c_cfg, 0.5 * n_eligible_train)`; `c = φ·c_cfg + (1-φ)·bound`; `c = max(1, min(c_cfg, round(c)))`; then controller clamps to `[c_min=10, c_max=c_cfg]` (`dynamic_kc_controller.py:106`). Matches Eq. 1.
- c_max = `selector.kwargs.c`: 60 cifar (n=300), 20 speech (n=100). K = 10 both.
- Recompute: after every commit (`top_aggregator._dynamic_kc_step`, update_every_n_aggs = 1).
- Pool = `n_eligible_train`: joined clients whose notify state is AVL_TRAIN at the commit (includes clients busy on a task).
- φ (`flame/selector/loss_phase.py`): mean of the last W per-sample rms losses (`stat_utility/count` of each arriving update) ÷ the highest such window mean so far; 1.0 until the first full window. Window, not EMA. W = `phase_window` or else the trainer count (300 cifar, 100 speech).
- **Late in training: cifar follows the pool, speech follows a blend.** cifar smu: φ 0.12 at 12 h, 0.03-0.11 after 65 h, so c ≈ pool/2 (second half: mean |c - clip(pool/2)| = 2.9). speech smu: φ stays 0.24-0.43 (mean 0.33 in second half), c = 0.33·20 + 0.67·min(20, pool/2), mean |c - clip(pool/2)| = 2.2; c 10-15 against pool/2 of 5.5-12.
- Constants: c_max, K = fixed by the deployment/baseline's paper ratio (c/n = 20%, FedBuff K 10); ½ and c_min = K = fixed by design (ablation-chosen, not derived); pool = sensed (Notify ledger); φ = sensed from update losses, W set by population size; no EMA rate.

**B2 FLuX-Agg** (`flame/optimizer/fedbuff.py`, `weight_factor`, `agg_rate_conf.type: new`; run config scale 0.4, a_exp 0.25, **b_exp 0.2** (not 0.1), beta type `polynomial_upshift`).
- **MISMATCH: the weight is a sum, not a product:** `w = 0.4·(1+τ)^-0.25 + 0.6·(1 - (1+u)^-0.2 + 0.5)`. `weight_factor` returns `scale·α + (1-scale)·β`; the "upshift" adds 0.5 to β, so the range is [0.2, 1.5], not [0,1].
- u = `tres.stat_utility` = sqrt(sample-weighted mean of squared first-pass losses, pre-adaptation prefix) × trained samples (`syncfl/trainer.py:normalize_stat_utility`, `statUtility: fedscale`). Not per-sample, no clipping. Measured u (agg_round logs): cifar smu median 5.1 (p10 0.5, p90 49); speech smu median 336 (p10 128, p90 753).
- Server step: `lr = 1.0` (cifar and speech configs), `base += lr · Σ_i w_i Δ_i / K`, K = 10. Not normalized by mean weight (that is only for rates `norm`/`value`). `norm_clip` off.
- Per-sample normalization: **client side**, `lr_batch_normalize: true`: each local step uses `lr · min(1, len(batch)/batch_size)` (`async_cifar10/trainer/pytorch/main.py:1192`). So only short batches (last batch of a shard) get a smaller step. No server-side normalization.
- "One multiply per update": accurate (scalar w per update, one `v * rate`; `_aggregate_pytorch`).
- Measured mean w over all aggregated updates (actual τ and u): cifar 0.79-0.83 (s0u..smu), speech 1.02-1.06. **MISMATCH with "0.61-0.78".** The outline's formula (no +0.5, b 0.1) is not what runs.
- τ=10 ratio to 1/√(1+τ): with real u, 1.9x (u 0.5), 2.3x (u 5, cifar median), 2.8x (u 50), 3.1x (u 340, speech median); event-weighted means 2.3-2.6x cifar, 3.1x speech. The old plot used `0.4(1+τ)^-0.25 + 0.6(1-(1+u)^-0.1)`, u=1000: 0.70 at τ=0, 0.52 at τ=10, ratio 1.72 = the "1.7x" in the text. I could not reproduce 1.8x from any variant; 1.7x is what the old plot gives, but that plot is not the code. **Neither number is right for the code.**
- Regenerated `fig5_4_agg.pdf` from the code: u = 340 (speech median) and u = 5 (cifar MobiPerf median); annotation "2.3x to 3.1x FedBuff's weight at τ = 10". Authors decide the text (FedBuff's weight 0.30 at τ=10; FLuX-Agg 0.70 (u 5) to 0.93 (u 340)).

## C. Appendix
**C1 Time-of-day vote** (`scripts/eligibility_forecast.py:seasonal`): for query window [a,b], score = mean over earlier days k = 1..floor(t/86400) of the fraction of [a-k·day, b-k·day] the client was eligible (`_avl_train_share`, 5-min grid). No time bin beyond the window itself, all earlier days count equally (day 0 is history), no threshold: the score is the continuous share and is scored by AUC/Brier directly. Ties in AUC get average ranks (`scipy.rankdata`). It is a share average, not a majority vote; "REFL-fc's majority-vote input" is the docstring's wording.

**C2 Brier** (MobiPerf, question B, h = 5 min / 15 min / 1 h): Prophet 0.526 / 0.483 / 0.367; FedDance 0.116 / 0.158 / 0.287; **0.05-0.24 = base-rate predictor** (0.050 / 0.117 / 0.235; in B "now" is replaced by the base rate, `eligibility_forecast.py:run`). Persistence is degenerate in B (always 1): Brier = 1 - pos_rate = 0.053 / 0.135 / 0.377. In A (every client), persistence: 0.021 / 0.050 / 0.109.

**C3 Check-in wait, speech MobiPerf, sync deployable stacks** (`rounds.rows`; wait = next round's open minus previous commit, per round):

| stack | rounds (120 h) | median / mean / p90 wait (s) | wait share of round+wait | median round open→commit (s) |
|---|---|---|---|---|
| FedAvg | 2156 | 22 / 42 / 107 | 21% | 111 |
| FedDance | 2052 | 22 / 41 / 110 | 19% | 121 |
| REFL | 1985 | 22 / 57 / 173 | 26% | 109 |
| Oort | 592 | 63 / 125 / 281 | 17% | 486 |

Outline "41-109 s": 41-57 s is the mean wait for FedAvg/FedDance/REFL; 107-110 s is their p90. "20-50% of a 110-220 s round" is not supported: wait is 19-26% of the cycle (median round 109-121 s; Oort 486 s). Oracle (oracle waits are 0 s by construction) rounds vs deployable: FedAvg* 3035 vs 2156 = 1.41x; FedDance* 4660 vs 2052 = 2.27x; REFL* 3110 vs 1985 = 1.57x; Oort* 935 vs 592 = 1.58x. Range 1.4-2.3x (outline 1.2-2.3). For cifar MobiPerf the oracle round counts differ by leg end (REFL/FedDance legs ended at target at 74-80 h), so use speech.

**C4 Speech selection** (FLuX legs, picks digest; speech 100 clients vs cifar 300, `|D|` = per-client dataset size from `picks.per_client`):
- (a) size skew: speech |D| mean 840, median 800, max 2559, **CV 0.52, top-10% share 20-21%**; cifar |D| mean 164, median 114, max 1193, **CV 1.03-1.04, top-10% share 33%**. Speech shards are about half as skewed. Selection-size correlation (picks vs |D|): speech s50u 0.39, smu -0.15; cifar s50u 0.40, smu -0.01.
- (b) timeouts/evictions (share of tasks): `abandon_timeout`: speech s50u 3569/30350 = **11.8%**, smu 596/26383 = 2.3%; cifar s50u 1.7%, smu 0.4%. `fates`: speech s50u late 3552, evicted 16; smu late 534, evicted 62 (counts of tasks; 0 `timeout` fates). Tasks lost per departure: `left_mid_task` speech s50u 13.5%, smu 3.2%; cifar 1.7% / 0.5%. Samples per update: speech 336 (s50u) / 362 (smu) vs mean |D| 840; cifar 78 / 75 vs 164. Both causes hold in part: smaller skew (a) for all cells, timeouts (b) mostly on s50u speech. I did not run a FedBuff-side comparison here (the 0.8-1.0x figure comes from the existing `picks_speech` plot).

## D. Evaluation numbers
**D1 FedBuff peak vs FLuX** (5-eval trailing mean, MobiPerf, seed 1):
- cifar: FedBuff peak 74.13% reached at 1,323,889 updates (106.0 h); FLuX reaches 74.13% at 476,015 updates (76.8 h, 2049 GB vs 5697 GB): **2.78x fewer updates**. FLuX peak 75.96%.
- speech: FedBuff peak 83.45% at 24,331 updates (99.95 h); FLuX reaches it at 23,633 updates (97.1 h): **1.03x fewer updates**. FLuX peak 84.22%.

**D2 Check-in arm** (`FB2SWA_<cell>_fx_ck` vs `FB2E2E_<cell>_fx`, cifar; whole leg, ends at target or 120 h):

| cell | arm | updates/h | mean tasks in flight (busy slot-time / leg time) | mean c allocated | end h | GB |
|---|---|---|---|---|---|---|
| s50u | FLuX | 10,487 | 59.9 | 60.0 | 72.0 | 3249 |
| s50u | check-in | 2,142 | 12.3 | 22.4 | 73.5 | 678 |
| smu | FLuX | 4,883 | 27.2 | 27.2 | 120 | 2522 |
| smu | check-in | 1,643 | 11.0 | 14.7 | 120 | 849 |

Check-in keeps 0.2-0.4x the tasks in flight and 0.2-0.3x the updates/h, consistent with "fewer tasks in flight". Leg ends: s50u both hit 80% target; smu accuracy 75.1% vs 75.3% at 120 h.

**D3** `baseline_deviations.md` (from `scripts/baseline_deviations.py --md`), this folder.

**D4 Knobs varied per baseline (scope for the ¶4 claim).** In the FB2 run set (root above) every baseline ran at one setting per dataset: FedBuff K 10, c 60 (cifar) / 20 (speech); sync stacks K = ceil(max(paper K/N, 5%)·n), Oort 5% (FELIX_READINESS "Cohort and participation"). Runs: `FB2E2E_*_ctl_srv`, `*_fbstar_srv`, `*_sync_srv`, `*_rstar_srv`, `*_fdstar_srv`, `*_favstar_srv` (cifar and `gs_` speech). **No FedBuff c or K sweep and no Oort cohort-size sweep exists in the FB2 set.** What was varied (pre-FB2 screens, `experiments/lrcheck_*`, `$R/n163_every`):
- FedBuff server lr (FX-D73; 0.0002 paper, 0.04 and 1.0 retuned; `lrcheck_20261008`); client/server lr and yogi eta/tau for REFL, Oort, FedDance (`lrcheck_20261008_fxn80`, `lrcheck_20261009_fxd106`, `lrcheck_20261010_*`); Oort K ≥ 9 screen (FX-L63); lr-decay `every` 20-400 for sync stacks (`$R/n163_every/every_{20,50,100,200,400}.log`, appendix sensitivity, FX-N166).
- c 30 and K 20 were tried on FLuX's own predecessor stack (F46/F47, FELIX_READINESS line 259: c 30 final at 2.2-2.4x GB, K 20 -0.4 pp at 2.3x GB) and as FLuX swaps `fx_cfix` (c 30), `fx_ca`, not on FedBuff. So the claim "no knob of the existing systems removes the problem" should be scoped to: server/client lr, decay schedule, optimizer for baselines; FedBuff c/K and Oort cohort were fixed at the papers' ratios.
