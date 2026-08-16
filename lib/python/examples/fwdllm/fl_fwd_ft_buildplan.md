# Build plan — implementation specs for [P5](fl_fwd_ft_practice.md#p5--the-queue)

> **Companion to [fl_fwd_ft_practice.md](fl_fwd_ft_practice.md).** That doc owns *what is true* (P3/P4
> ledgers) and *what state each task is in* ([P5.2](fl_fwd_ft_practice.md#p52-execution-plan--to-a-zero-input-run)).
> **This doc owns *how to build it*** — one spec per P5 task: files, algorithm, edge cases, sanity gate.
>
> **No status tags here.** A task's state lives in P5.2 and nowhere else. When a task lands, its result
> goes to P3/P4 and its spec here is deleted (R3).
>
> **Read [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) before proposing any change to a spec.**

---

## §0 — Conventions every spec below inherits

**These are not restated per task. Violating one is a rejected change, however good the result.**

1. **Flag-gated, default = old, byte-identical off.** Every behaviour change lands behind a named
   hyperparameter whose default reproduces today's bytes. Parity/correctness fixes are the exception —
   they ship enabled, including the code-level default. Terminal flag state (`PERMANENT` / `FLAGGED` /
   `REVERTED`) is the operator's call, never the implementer's.
2. **A flag is read in exactly one place and echoed once.** Aggregator-side flags come off
   `self.config.hyperparameters`; trainer-side flags off the trainer's own config. A flag both sides read
   (`perturbation_count`, `probe_combine`) must be written to **both** override blocks and must agree —
   `test_model_args_parity.py` enforces this and a new dual-read flag must be added to it.
3. **Emit-only is not free.** Anything added to the commit path gets timed before it ships
   (`_apply_weighted_update` breakdown, [P9.2](fl_fwd_ft_practice.md#p92-launcher-gotchas)). The cos audit
   cost eight arms by being "just logging".
4. **Nothing on rungs 1–3 modifies `trainer/`, `aggregator/`, or any yaml on the critical path**
   ([P7](fl_fwd_ft_practice.md#p7--the-instrument-ladder)). Probes import production code; a validated
   result transfers as a config flag, not a rewrite.
5. **Every new number needs predicted-vs-observed and a run id** before it reaches a ledger.
6. **Sanity gate is part of the task.** A task is not done when the code runs; it is done when its stated
   gate reproduces a number already in P3/P4, or produces one that was pre-registered.
7. **Dataset-specific constants come from `expts/dataset_registry.py`.** Never re-hardcode `num_labels`,
   `p`, a class-balance threshold, or an h5 path. Adding a dataset must stay a row in
   `configs/datasets.yaml`.

**Scoring vocabulary** (definitions in [P8](fl_fwd_ft_practice.md#p8--reproducing-any-number-from-logs);
all exact at any horizon): `ρ` step/norm ratio · `B = ½Σlog(1+ρ²)` budget spent · `Φ = e^B` norm inflation
· `Λ = Σρ√(G_rule·N/p)` progress · `A = Σρ√(G_rule·N/p)·‖θ_tr‖` absolute progress, the only one comparable
across `p`.

---

## §1 — The dataset substrate (landed 2026-08-12; specs kept only where work remains)

**What exists now**, so no spec below has to re-derive it:

| | agnews | yahoo | yelp-p |
|---|---|---|---|
| classes | 4 | 10 | 2 |
| `p` at `rf`=16 / 64 | 450,340 / 118,348 | 454,954 / 122,962 | 448,802 / 116,810 |
| official split | 120,000 / 7,600 | 1,400,000 / 60,000 | 520,000 / 40,000 |
| shard at `C`=100 | 1,200 / 76 | 14,000 / 600 | 5,200 / 400 |
| bins/round at `C`=100 | 150 | 1,750 | 650 |
| token length p50/p95 | 41 / 70 | 84 / 367 | 137 / 493 |
| `max_seq_length` | 192 | 256 | 256 |
| niid groups on disk | α = 0.1…100, `C`=100 | **α = 1, 100 at `C` = 100 and 1000** | **α = 1, 100 at `C` = 100 and 1000** |

`configs/datasets.yaml` + `expts/dataset_registry.py` hold every one of these. New groups were built by
`expt_scripts/build_niid_partitions.py` (equal shards, full coverage, disjoint, Dirichlet label mix,
**shuffled within client**) and pass `expt_scripts/check_partitions.py` on all six checks.

**Three facts that change how a yahoo/yelp-p arm is read:**

- **Bins per round differ 11.7×** (150 / 1,750 / 650). Anything expressed *per round* — `max_data_id_progress`,
  a `data_id` sweep, an epoch — is not comparable across datasets. Score per **commit** or per **vclock-hour**.
- **`max_seq_length` 256 is a cost choice**, covering ~p89 of yahoo and ~p82 of yelp-p. Trainer wall clock
  is ~linear in it, so a yahoo arm costs ~1.33× an agnews arm per forward pass **before** the 11.7× in bins.
  Budget the wall-clock preflight (task 0.7) against that, not against agnews' constants.
- **`Λ` does not transfer across `p`** (P4 read 2) and `p` now differs by dataset. **Use `A`.**

**`--dataset NAME` (task 0.8, landed 2026-08-12).** `run_sequential.sh` writes the registry's 4 keys
(`dataset`, `data_file_path`, `partition_file_path`, `max_seq_length`) into **both** the aggregator and
trainer override blocks, plus `trainer.dataset.name`, exactly like `--partition-method`. Unset ⇒
byte-identical to today (agnews, no `max_seq_length` line — inherits the 192 code default). Also landed
with it: the partition-group-exists check now verifies against the **resolved** dataset's own partition
h5 (not just whichever group name was typed) even when `--dataset` alone changed which file is in play;
and a sim-mode dry-run on any non-agnews dataset is **blocked** unless `--force` — every
`sim_charge_profiles/*.yaml` today was profiled on agnews only, and per-pass cost scales with
`max_seq_length`, so reusing one for yahoo/yelp-p silently mis-prices the vclock. **Sanity gate
(`expt_scripts/test_dataset_launcher.py`, CPU-only, ~30-60s): 17/17 checks pass** — yahoo/yelp-p paths in
both roles, unset is byte-identical, an unknown name fails fast (exit 3, not the `--force`-able exit 2), a
group absent from the resolved dataset's h5 is caught, and the sim-profile mismatch is caught.

**Task 0.10 (dataset-switch checklist), landed 2026-08-12.** `diagnose_partition_binning.py` was the one
real bug the sweep found: `N_CLASSES = 4`, agnews-only default paths, and an agnews-only cache-filename
template were all hardcoded, and — the actual correctness bug — the single-class-collapse detector
compared accuracy to a hardcoded `0.25` (agnews' `1/K`), which would have silently missed collapses on
yahoo (`1/10`) and yelp-p (`1/2`). Now takes `--dataset` (default agnews, byte-identical) and derives all
four from the registry. Verified inert (registry lookups / comments / arm-name references, not bugs):
`test_commit_gate.py`'s `P_TRAIN=450340` (a synthetic stub-aggregator fixture, not a live config read);
`replay_scoring.py`'s `P_BY_RF` table (the documented fallback tier, task 0.2); `fwdgrad_utils.py`'s
`_FD_REF_P`/`probe_fd_chord.py`'s `450340` (the FD scale-invariance reference anchor — deliberately fixed
regardless of the running dataset, this *is* the h-rescaling machinery P9.1(b) asks for); the two
`nodes/run_node*.sh` comments (historical arm predictions). `scripts/calculate_seq_len_cdf.py` (named in
the original suspect list) does not exist in the repo — stale reference, nothing to fix; the token-length
numbers it once produced are already the source-of-truth values in `configs/datasets.yaml`.
**Sanity gate reproduced:** `grep -rn "agnews\|450340\|N_CLASSES = 4" expt_scripts scripts aggregator
trainer` (excluding `*.yaml` and `smoke_logs/`, which are the 20 intentionally-agnews-default yaml pairs
and generated run logs) returns only the inert set above.

**Not yet done — next steps:** no GPU-side validation has run yet. Phase 1's `probe_inflation_damage.py`
reproduction (buildplan §3) and the Phase 4.2 zero-input yahoo run are still ahead, and both should use
`--dataset yahoo` now that it exists. A non-agnews **sim** smoke run additionally needs
`profile_sim_charges.py` run once for that dataset (or `--force` for a real-only / one-off check that
doesn't need the vclock to be accurate).

---

## §2 — Phase 0 · unblock the instruments · no GPU

*Order is by impact. 0.1, 0.4, 0.8 unblock everything downstream; 0.7 is the one that has been costing arms.*

### 0.1 — `replay_scoring.py --cos` block sizing *(landed 2026-08-12)*

Blocks by *probe fires*, not commits (`--cos-block-fires`, default 10) — at `cos_probe_every = 25` the
old `BLOCK = 50`-commit / `len(blk) < 5` guard dropped every row. `cos_audit()` now reports per-block
rows (commit range, for joining `Φ`/accuracy) plus a run-level summary `D` = mean ± SEM over every
individual fire's ratio — SEM, not the block-averaged spread, because per-fire sd ≈ the mean (SNR ≈ 1,
P4.2/D-2), so only the arm-level mean is meaningful. Zero fires prints "no cos fires" and returns; a
trailing partial block below half `--cos-block-fires` is dropped and the drop is printed; median fire
spacing is printed alongside `D`.

**Sanity gate reproduced:** `112201` → `D` = 0.1485 ± 0.0200 (n=55), `145729` → `D` = 0.1035 ± 0.0172
(n=38), against [P4.2](fl_fwd_ft_practice.md#p42-g-1b--s-is-efficiency-not-safety).

### 0.2 — dataset-derived `p` *(landed 2026-08-12)*

`resolve_p(cfg, rf, run_dir)` in `replay_scoring.py`, three-tier (`[ProbeDim]` → registry → `P_BY_RF`).
The `[ProbeDim]` regex (`\bp[= ]+(\d+)`) is confirmed against real trainer logs — `[ProbeDim] p=450340
h*sqrt(p)=6.7107` (rf=16) and the rf=32/64 variants (`p=229012`/`118348`) all match on the first `p=`,
unambiguous even with the later `sqrt(p)=` on the same line; no silent miss. `p_source` prints in every
`--cos` header. The tier-3 `P_BY_RF(agnews)` fallback now **warns to stderr** (`resolve_p`) rather than
passing silently — correct for every P4 arm (all agnews), silently wrong for any future dataset whose
config predates the `dataset` field.

**Sanity gate reproduced:** agnews arms still read `p` = 450,340 (source `ProbeDim`) with no warning; a
synthetic run with no `[ProbeDim]` line and an unregistered dataset prints the fallback warning.

### 0.3 — class-skew preflight *(landed)*

Threshold is now `dataset_registry.max_dominant_share(K) = min(0.999, 1/K + 0.25)`, used by **both**
`test_cos_probe.py` and the production warning in `FedSgdAggregator._cos_probe_gradient`. The test
exercises K = 2 / 4 / 10 and still fails the single-class fixture at every K.

### 0.4 — per-commit `G_rule` in the scorer *(landed 2026-08-12)*

`enrich()` computes `G_rule_t` per commit via `g_rule_of(rule, P_t)`: `P_t` under `mean`, measured
`E_select(P_t)` under `select` (`E_SELECT_BY_P = {10: 2.988, 30: 4.744}`) — an unmeasured `P` under
`select` raises (`SystemExit`), it is never interpolated. `Λ` accumulates
`ρ_t·√(G_rule_t·N_t/p)` unchanged. `P_t` reads `r.get("p_probes")` first (unset on every record today) and
falls back to the run's constant `hyperparameters.perturbation_count` — so adding `p_probes` to the
`server_update` record when 3.4 lands is a one-line switch, no scorer change needed.

**Sanity gate reproduced:** constant-`P`=10 arms `112201`/`145729` score bit-identical `Λ` (1.722 / 2.385,
matching P4) before and after.

### 0.5 — commit the Φ-stop replay *(landed 2026-08-12)*

`expt_scripts/replay_phi_stop.py`: smooths each arm's accuracy over an 11-eval **trailing** window (raw
crossings gave the false agreement at `Φ`≈2.95 that P4.2 already flagged), finds the first commit where
`Φ` crosses each threshold, reports smoothed-peak minus smoothed-accuracy-at-stop. (a) A non-crossing arm
scores given-up=0, counted not skipped. (b) The learned/not-learned gate uses the **raw** peak (matching
P4's own peak column) so a smoothing-window boundary effect can't drop a real arm — `035045`'s raw peak
is 0.801 but its smoothed peak is 0.798, just under 0.80; gating on the smoothed value would have wrongly
excluded it. (c) momentum-corrected `Φ` when `server_momentum` > 0.

**Sanity gate: close but not exact, and here is why.** The natural on-disk candidate set — every arm with
raw peak ≥ 0.80 and `T` ≥ 177 that actually crosses a threshold — is exactly 10 arms (`200325 200242
200358 065837 223446 212009 013806 112201 222817 145729`), independently matching P4.1's stated `T` range
177–1,364 and both combination rules/all three `rf`. Separately, `013917`/`035045` (peak just at 0.80,
never crossing) reproduce **exactly** as described. Scored together, the **no-stop worst arm reproduces
exactly (0.5951, `112201`)**; `Φ`=2.5/3.0 worst reproduce to ≤5% (0.0202/0.0199, 0.0117/0.0122); `Φ`=2.7
mean is 5% high (0.0057 vs 0.0054). `Φ`=2.3's mean/no-stop mean run 20–35% high (0.0207/0.1565 vs
0.0154/0.1408) — P4's own note that **the 08-07 K/η/P portfolio was deleted from disk** means an
α=0.1 arm P4.1 states its 10 spanned is no longer available to include, which would raise these two
mean-only cells without moving the exact-matching worst-arm cells. Not re-derivable from current disk
state; recorded here rather than silently forced to match.

### 0.6 — K-C rung 1 (replay-only) *(landed 2026-08-12)*

`expt_scripts/replay_kc_rung1.py` reads `agg_round` telemetry directly (`staleness[]`,
`pastdated_commits`, `vclock_now`) plus the `server_update` commit count — no GPU. Only `151316` (K=30)
and `171739` (K=50) are on disk; the `K`=10 leg is not (same disk-cleanup as 0.5's missing α=0.1 arm, so
its `stale_frac`≈0.143/`pastdated_max`≈1 are not independently re-checked here).

**Sanity gate reproduced, closely:** commits/vclock-h = 100.7 / 144.4 (doc: 101.6 / 146.4, ≤1.4% off, ✓
the ≤15% gate) with observed ratio **1.434×** (doc: 1.44×). Staleness ≥1 fraction **0.458 / 0.408** and
`pastdated_commits` max **0 / 34** both reproduce **exactly**. The measured per-round vclock time (8.934 s
/ 12.461 s) lands within 1.5% of the doc's REAL-wall 8.9 s / 12.3 s (a different clock; sim's per-round
vclock tracking real wall time this closely is itself informative, and is what task 0.7's `τ(K)`
estimate leans on).

**Model check** (the arithmetic-falsification the task asks for, not a decision): the `C`-model
(`C/(n_req·τ)`) predicts **1.195×**, the `K`-model (`K/τ(K)`, task 0.7's pooling estimate) predicts
**1.208×** — both under-predict the observed 1.434× by about the same ~18%, and by nearly the same
amount as each other. Confirms the doc's framing exactly: `C` moved with `K` on both arms (`C/K` = 2/2
here), so this rung **cannot separate the two models** — it only shows neither one's arithmetic is
complete on its own, which is what sizes K-1.

### 0.7 — wall-clock budget preflight *(landed 2026-08-12)* ⟵ *the one that has cost eight arms*

**`expts/wall_clock_preflight.py`** (`g_rule`, `tau_round_s`, `project`) holds the arithmetic, importable
the same way as `dataset_registry`; `run_sequential.sh` calls it once per `(run_key, variant)` right after
`per_baseline` is resolved, next to the other feasibility checks (§ "wall-clock budget preflight"),
CPU-only, refuses on breach exactly like every other `level: error` entry (`--force`-able, exit 2).

`n_req = p·(ρ*/s)²/G_rule` matches `aggregator/FedSgdAggregator.py:418`'s closed form exactly (only
defined under `commit_gate=n_target`; `var` has no such form and is skipped). `commits_projected =
(vclock_budget/τ(K))·K/n_req`, where `τ(K) = 8.9·(K/30)^0.63` s — the **same unvalidated pooling model
K-1 is the registered A/B for** (P5.1), calibrated at the one measured point (`K`=30 → 8.9 s, P3 "K at
fixed n_req"). This is a preflight *estimate* meant to catch a gross breach, not a claim K-1 is settled.
`per_commit_cost = base + audit_cost/stride`, `base` = 1.64 s, `audit_cost` = 85·(batch/1024)/stride s
(0 if the audit is off), `stride` = `cos_probe_every` or 1 (edge case b). Refuses when
`projected_real_wall` exceeds `sim_wall_ceiling_s` (sim, default `max_runtime_s × 20`, edge case c) or
`max_experiment_runtime_s` (real, default `max_runtime_s + 1800`, edge case d) — same defaults the code
itself falls back to. `per_commit_cost` has no `K`/`I` term (edge case a). The refusal detail string
carries `commits_projected`, `per_commit_cost` and the ceiling (edge case e).

**Sanity gate reproduced**, both as a pure-function check and end to end through
`run_sequential.sh --dry-run` (`expt_scripts/test_wall_clock_preflight.py`): `002208`/`022448` (stride 1)
project a breach (145,288 s / 38,870 s vs their 7,200 s ceiling) and the live launcher exits 2;
`112201`/`145729` (stride 25) project under their ceiling (16,903 s / 7,380 s vs 21,600 s) and the
launcher exits 0 — all four against their real historical fate (P4/P9.2).

### 0.7a — the preflight, extended for law C *(landed 2026-08-15, with 3.2/3.3)*

Task 0.7's projection prices a **constant-`ρ`** arm: `commits = (vclock/τ(K))·K/n_req` with `n_req` from
`rho_star`. Under `rho_schedule=landing` that is meaningless — `rho_star` is unset, so it falls back to the
0.01 code default and projects a phantom **45,000-commit** run, refusing every launch. Law C's commit count
comes from `(B_max, T_res, f)` instead, so `project()` now takes `rho_schedule`/`b_max`/`t_res`/
`budget_stop_frac`/`max_iter`/`gate_rho_ref` and, under `landing`, walks the real trajectory via
`expts/landing_law.simulate` and costs it with **T5's two-term fit**
(`4.41 s/commit + audit/stride + 0.77 s/trip`). **The const/rm branch is untouched** — it stays the model
already validated against four historical arms.

**It also carries the new gate.** `Projection.gate_starved` refuses when round trips/commit falls below 3 —
the metric `003648` died on while looking vclock-healthy, which no wall projection could have caught
(that arm's *total* wall was in budget right up until it wasn't).

**Sanity gate reproduced:** the preflight's own numbers now equal `replay_landing_law.py`'s exactly —
agnews 899 commits / 4,817 trips (5.36 per commit), yahoo 898 / 3,374 (3.76) at `T_res`=300, both passing;
yahoo at `T_res`=500 refused with `gate starved ... 2.46 per commit`. All four historical arms
(`002208`/`022448`/`112201`/`145729`) still score exactly as before, and `test_wall_clock_preflight.py`
passes end to end.

### 0.8 — `--dataset` in the launcher *(landed — see §1)*

### 0.9 — partitions for yahoo / yelp-p *(landed — see §1)*

### 0.10 — the dataset-switch checklist *(landed — see §1)*

---

## §3 — Phase 1 · B-1 · 1 GPU, no FL stack · the decision gate *(landed 2026-08-13)*

Validation gate reproduced model §7.1 unchanged. Yahoo/yelp-p wired through `dataset_registry`, `p`
confirmed against `[ProbeDim]` (454,954 / 448,802). Swept `Φ` ∈ {1.5,2,2.5,3,3.5,4} × 3 datasets,
reps≥3, mode=noise.

**Pre-registered prediction refuted.** Predicted monotone in classes (yelp-p > agnews > yahoo); landed
**erratic** instead — agnews knee ~3.0–3.3, yahoo and yelp-p both ~2.0–2.3 despite opposite ends of the
class-count range. `expts/prep_b1_configs.py` (new) generates the per-dataset configs; a
`--phis`/tmp-config-race fix landed in `probe_inflation_damage.py` for the multi-dataset launch. Full
result and its "3.1 now mandatory" implication: model §7.1/§5.5b, P10, `fl_fwd_ft_practice.md` P5.2
Phase 1.

**Replicated 2026-08-13** — 2 independent runs per dataset now, all agreeing tightly (agnews knee 3.0–3.5
both times; yelp-p crosses its normalized-0.5 knee at Φ=2.0–2.5 both times; yahoo agrees across a 3- and
9-epoch run, the latter ruling out undertraining as the driver). Below this doc's 10-arm bar (P4.1), but
no longer single-run.

---

## §4 — Phase 2 · registered-node launch shape

**All three registered nodes (K-1, P-1, G-2) landed 2026-08-13/15 — their specs are deleted per §0 rule 6;
results are in `fl_fwd_ft_practice.md` P3/P4.** Kept below: the launch shape and standing preflight, reusable
for the next registered node. Common launch shape: `expt_scripts/nodes/run_node_*.sh` built on
`_node_lib.sh` (which aborts a node on any arm producing < 5 commits); `run_node_g1b_gate_s.sh` is the
current best template.

**Standing preflight for every arm here** (do not skip; this list is what the eight dead arms were missing):

```
--dry-run first, then --only --yes --clean --force ; --num-trainers 100
assert ceil(n_req/K) <= max_iter        # existing
assert projected_real_wall < sim_wall_ceiling_s   # task 0.7; now --dry-run's own preflight, no longer by hand
pin the pool for any A/B: --var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off
echo the enactment lines: [ServerStep] [CommitGate] [CosProbe] [probe_combine] [ProbeDim] [FD] spacing
```

**Note from G-2, corrected by T5 (2026-08-15):** the wall-clock preflight above did not stop `annealed`'s
`[SIM_WALL_CEILING]` death, and the reason is **not** "a schedule that spends its budget unevenly" — that
was a plausible mechanism, never a measured one. Replay says `003648` ran with the gate's `I` **floored at
1 on 98% of its commits** (3,353 commits against 3,429 round trips), so every commit paid the full
server-side path with a single round trip's worth of trainer work amortising it: **8.40 s/round-trip
against `145729`'s 1.75**. The preflight's `τ(K)` model prices round trips, so a run that converts its
budget into commits instead is invisible to it. **Add round-trips-per-commit to the preflight** (task 0.7)
and do not treat a clean `--dry-run` as a guarantee for `gate_rho_ref=annealed` until 3.3 ships.

---

## §5 — Phase 3 · the controller · multi-day

**None of this needs new science.** Design in model §5.5f. Each component is a flag, default-off,
byte-identical until its A/B scores. **Build 3.3 first**: the whole P4.1 gain (0.141 → 0.005) is there and
it depends on nothing else.

**3.1 + 3.2 + 3.3 are one closed-loop controller, not three independent features — read all three specs
below together before building any of them** (2026-08-13). **Both open design decisions were resolved
2026-08-15 by T5's replay** (law C for the anneal; `halt` for the stop) — the specs below carry the
decisions and the numbers behind them, so implementation is unblocked. 3.2 is
the continuous per-commit throttle, 3.1 is the strided re-sense feeding it, and 3.3's stop is the discrete
backstop for when the throttle's `T_res` estimate turns out wrong (edge case (f) below) — it is not
redundant with a working anneal, it is what the anneal fails safe into. Live proof the backstop is needed:
G-2's annealed leg (`003648`) hit `[SIM_WALL_CEILING]` at vclock 26,026 of its intended 40,000 (real wall
4.6× over the preflight's estimate) on 2026-08-13 — cut short before it could land, exactly the case the
backstop exists for. **G-2 landed 2026-08-15** against a completed `setpoint` counterpart at the same
40,000-vclock target (`084554`): `annealed` still won on accuracy-per-vclock despite dying early, confirming
P2's default, but the death itself stands — full readout `fl_fwd_ft_practice.md` P4.5.

### T5 — the landing law replayed against the gate *(landed 2026-08-15, no GPU)*

**`expt_scripts/replay_landing_law.py`.** Run before any Phase-3 code, and it moved three defaults and
found one new mechanism. Parts A–E: arm validation, the two candidate laws, an `f` sweep, D1's two-phase
trajectory, and a wall projection.

**`Λ = 2B/s` is an identity wherever the gate holds `s`** — because `N = p(ρ/s)²/G_rule` makes
`cos = ρ/s`, so `Λ = Σρ·cos = Σρ²/s = 2B/s`. Confirmed out of sample to **−0.3%** on both `s`-pinned arms
(`145729` 2.385 vs 2.379; `112201` 1.722 vs 1.717) and missing **+21.5 to +23.3%** on the three whose `s`
drifts (`035045`/`003648`/`084554`) — the same arms, in the same order, as §4.6's time-law miss table.
**Consequence: the `ρ` schedule is `Λ`-neutral at fixed `B`.** Law A and law C bank the same `Λ`; they
differ only in how many commits they take to spend the budget. Any claim that one schedule "learns more"
is an artifact of comparing them at unequal `B`.

**Cost model, fitted over 5 arms to ±1%:** `real_wall = 7.81·commits + 0.77·round_trips` (cos audit on,
stride 25). Separable only because the portfolio spans 1.02–20 round trips/commit. The audit is charged
per *commit* (85 s / 25), so an audit-off arm costs **4.41 s/commit + 0.77 s/trip**.

**Sanity gate reproduced:** the fit predicts all five arms' measured real wall to ≤1%
(`035045` 2.03→2.05 h · `145729` 3.67→3.67 · `112201` 3.57→3.55 · `003648` 8.00→8.01 · `084554` 3.13→3.11).

**What T5 could not settle — verify these on the Phase-4 arms, not by argument.** Every one is a number
the overnight run produces for free; none blocks the build.

| open | why replay can't close it | what closes it | if it comes out wrong |
|---|---|---|---|
| **`Λ` → accuracy on yahoo.** T5 clears yahoo at `Λ` = 1.04 against a **≥0.95 floor read off P4's agnews curve** | P4 read 2 already says `Λ` doesn't transfer across `p`; across *task* it has never been tested at all. `p` differs only 1% here, but the task differs entirely | node 3/4's own accuracy-vs-`Λ` curve — the first ever measured off agnews | the floor moves, `f` rises, and yahoo needs more budget than 0.95·`B_max`. Cheap: wall headroom is 4× |
| **Does `B_max` drift within one run?** (3.1 question 2, still open) | needs the probe to fire repeatedly inside a live loop, which is the thing being built | `B_max_old → B_max_new` per fire on nodes 1/3 | if it drifts a lot, the stride shortens; law C already self-corrects, so control is unaffected |
| **trips/commit ≥ 3** is calibrated on **one death and two survivals** | three arms is not a curve; the true knee could be anywhere in 1–8 | every Phase-4 arm reports it per commit; the gate gets re-sized once there are ten | a config passes preflight and still burns wall — the same failure the preflight exists to stop |
| **4.41 s/commit audit-off** is `7.81 − 85/25`, a **subtraction, not a measurement** | no audit-off arm at this operating point exists on disk | node 1's own `wall / commits`, read ~50 commits in | projections shift; the 4× headroom absorbs a 2× miss |
| **`f` = 0.95** rests on the `Λ` floor, with **no accuracy evidence** | `f` only ever moves the last few percent of budget, which no arm on disk isolates | replay nodes 1/3 at every `f` after the fact — the curve is free once the arm exists | a smaller `f` ends runs sooner at the same peak, which is a win, not a loss |

### 3.3 — budget-landing anneal + `Φ` stop

**Sensed from `B`**, which is exact and free (`B = ½Σlog(1+ρ_t²)`, already computable per commit).

- **Anneal:** the landing law, **law C — `T_res` is a rate, not a deadline** (decided 2026-08-15, below).
  `ρ*_t = min(ρ_max, √(2·(B_max_t − B_t)/T_res))` with `T_res` a **constant that is never decremented**.
- **Stop:** `B ≥ f·B_max`, `f` = 0.95. **`Φ`-stop and budget-exhausted are the same trigger, not two** —
  `Φ = e^B` and `B` is monotone, so `Φ ≥ Φ_thresh ⟺ B ≥ ln Φ_thresh`, and 3.1's `B_max = ln Φ_peak`. The
  fixed `Φ` = 2.7 constant (P4.1: 0.0054 given up, worst arm 0.0139) survives only as the fallback for an
  arm with no sensed `B_max`. **The rule needs no eval, no accuracy history and no task constant.**

**Decided 2026-08-15 — `T_res` is a rate (law C), and the re-sense question dissolves.** `B_max` is
*measured* and drifts; `T_res` is *chosen* and no probe returns evidence about it. Three candidate
denominators, and only one is well-posed:

| law | denominator | `B(t)` | terminates | on re-sense |
|---|---|---|---|---|
| **A** fixed-horizon | `T_res − t` | linear, lands at `t = T_res` | at `T_res` | ill-posed at the edges |
| **B** receding horizon | resets to `T_res` per re-sense | piecewise | **never** — each reset defers the landing | pathological |
| **C** fixed-rate ✅ | `T_res`, never decremented | `B_max(1 − e^{−t/T_res})` | on the stop rule | **nothing to reset** |

**Law C, for five reasons, the third decisive.** (1) It dissolves the re-sense question — `B_max` moves,
`ρ*` re-derives in one commit, no horizon bookkeeping exists to be inconsistent. (2) Under law A, perfect
tracking makes `ρ*` *exactly constant* and `B_rem` deplete linearly — it is a constant-`ρ` policy with a
deadline, not an anneal at all. (3) **Law A smuggles `T` back in as an input**, and §4.6a is titled *"why
`T` is not an input"*; D4 admits only model/PEFT/`p`. A run that ends when a commit counter hits an
operator-set number has a horizon input wearing a new name. (4) Every edge case below dissolves rather
than needing a guard. (5) `B → B_max` monotonically **from below**, so the controller is safe by
construction and the stop is a genuine backstop — versus `rm`, whose `Σρ²` diverges logarithmically so it
*always* eventually needs the stop.

**Price, measured not estimated (T5):** law C takes **3.1× the commits** of law A for the *same* `Λ`
(1,498 vs 477 at `T_res`=500, `Λ` 1.565 vs 1.523). Commits cost 4.41 s each with the audit off, so on the
Phase-4 arms this is **~0.6 h against a 10 h slot** — the right thing to buy.

**Constants, all settled on replay (T5), none fitted to an accuracy curve:**

| constant | value | why, and what refuted the alternative |
|---|---|---|
| `T_res` | **300** | **`T_res` = 500 is refuted** — 2.46 round trips/commit on yahoo and 2.26 on the `ln 2` prior, against the ≥3 gate. 300 passes on both datasets and in the two-phase trajectory |
| `f` | **0.95** | yahoo needs `f` ≥ 0.90 to clear `Λ` ≥ 0.95; agnews clears at 0.70. 0.95 is free given the wall headroom |
| `ρ_max` | `s·√(max_iter·K·G_rule/p)` ≈ **0.0999** | gate reachability, `⌈n_req/K⌉ ≤ max_iter` solved for `ρ` — mechanical, no operator input. **This replaces a `ρ* ≤ ρ*₀` clamp, which is wrong**: `ρ*₀` comes from the `ln 2` prior, so that clamp would block 3.1's re-sense from ever spending the budget it just found. Law C is monotone non-increasing at fixed `B_max` anyway, so only a re-sense can raise `ρ*` — exactly when it should |
| `B_max` prior | `ln 2` | D1, unchanged |

**Decided 2026-08-15 — edge case (e), what the stop does: `halt`, through the existing exit path.**
`FedSGDAggregator` subclasses `TopAggregator`, so `self._work_done = True` routes the stop through the
identical shutdown `max_runtime_s` and `[SIM_WALL_CEILING]` already use
(`fwdllm_aggregator.py:_check_early_stop_conditions`). **Halting is one line into a tested path; freezing
`θ` is new lifecycle state in a component that has none.** And freeze-and-keep-evaluating is *dominated*,
not a trade-off: a frozen `θ` has a fixed accuracy, so re-evaluating it measures eval noise and returns
nothing, at real GPU cost. On fire: latch (a re-sensed `B_max` can move the threshold, so the predicate
can flip even though `Φ` cannot) · skip this commit's update · discard in-flight uploads · checkpoint `θ`
· one final eval · `_work_done = True` with `stop_reason` in telemetry.

> **This makes Phase 4's "end within 0.015 of peak" mechanically satisfiable** — final *is* the stop,
> which is what P4.4's peak-vs-final rule has been working around all along.

**But halting alone destroys the evidence that validated the rule** — P4.1's 0.0054-vs-0.1408 came from
arms that ran *past* their stop. So ship three states: **`phi_stop: off`** (default, byte-identical) ·
**`log_only`** (emit the crossing and the would-be stop commit, keep training — zero cost, and how the
rule gets validated on an unseen task) · **`halt`**.

**State to carry:** cumulative `B`, commit count, latest sensed `B_max`. **Emit per commit:** `B`,
`B_max`, `B/B_max`, `Φ`, `rho_star_t`, `n_req`, `I`, and the stop reason when it fires.

**Edge cases.** (a) **dissolved by law C** — the denominator is a constant, so nothing divides by zero.
(b) Under `β > 0` the `Φ` law changes: **refuse to launch** (`β` = 0 on every shipped arm). (c) A run that
reaches `f·B_max` before its eval cadence fires must still stop — the trigger is `B`, not accuracy.
(d) `Φ` from `ρ` is exact; **never** re-derive it from `‖θ‖` ratios, which carry the audit's own noise.
(f) **dissolved by law C** — `T_res` never runs out. `B_max` re-sensed *below* `B_spent` clamps `B_rem`
to 0 ⇒ `ρ*` = 0, which is itself the correct stop, not a `sqrt` of a negative.

**Three stop reasons, one predicate, one code path:** `budget` (`B ≥ f·B_max`) · `phi_fixed` (the
pre-3.1 constant, for an arm with no sensed `B_max`) · `saturation` (3.5, later).

### 3.1 — two-phase `B_max`, continuously re-sensed

**Corrected 2026-08-15 (see P9.3 process note below) — the METHOD is not new.** It's fully specified
(model §5.5b) and **already coded and validated**: isotropic Gaussian noise on a **copy** of `θ_tr`,
scaled so `‖θ_tr‖` grows by `Φ`, read accuracy at `Φ` ∈ {1.5…4}, knee = `Φ_peak`, `B_max = ln(Φ_peak)`.
This is exactly `expt_scripts/probe_inflation_damage.py`, already run standalone for the whole B-1 sweep
(agnews/yahoo/yelp-p, 2026-08-13). **What's actually unbuilt is wiring that same probe to fire repeatedly,
on a stride, from inside `FedSgdAggregator`'s live commit loop** — not just once, offline, before a run —
and feeding its `B_max` into 3.2/3.3 live. Phase 1 (B-1) already decided this is needed at all: `Φ_knee` is
neither invariant nor derivable from `num_labels` (agnews ≈3.0–3.5 vs yahoo/yelp-p ≈2.0–2.3).

**Question 1 answered by 3.3's law-C decision (2026-08-15): a re-sense moves `B_max` and nothing else.**
`T_res` is not run state — it never counts down, so there is nothing to reset. Reset-on-re-sense (law B)
is a *receding horizon*: each reset defers the landing, `ρ*` asymptotes to zero and the run burns wall
clock at vanishing step size, reaching `220627`'s dead zone by a different road.

**Question 2 (does `B_max` drift materially within one run?) stays open and is what this task measures** —
emit `B_max_old → B_max_new` per fire rather than assume either answer. **T5's two-phase simulation says
the answer barely matters for control**: re-sensing at commit 50 / 150 / 300 gives 920 / 967 / 1,054
commits and `Λ` 1.528 / 1.531 / 1.538 on agnews. Law C self-corrects, so a late or noisy first sense costs
almost nothing — which is the property that made law C the choice.

**Predicted enactment (T5, pre-registered).** Prior `ln 2` → sensed, re-sense at commit 150, `T_res`=300:
agnews `ρ*` 0.0530 → **0.0764**, `I` 6 → 12; yahoo `ρ*` 0.0530 → **0.0573**, `I` 6 → 7. **The divergence
between the two datasets, with nothing supplied by anyone, *is* Phase 4's acceptance criterion** — so this
is the line to grep for first on nodes 1 and 3.

**Edge cases (mechanical).** The probe must be strided and budgeted (0.7) — the extra `~6 evals` per fire
is real cost, report it as a fraction of the commit path before this defaults on; it runs on a copy, never
the live model. Reuse the cos probe's fixed-seed reference batch **and its class-skew guard** — with the
cos audit off on the Phase-4 arms, that guard is otherwise not running at all (P6: slicing
`test_global.dataset.tensors[:n]` returns one client's Dirichlet shard).

### 3.2 — `ρ*_t` = `min(ρ_max, √(2·(B_max_t − B_t)/T_res))`

Trivial once 3.1 lands. Replaces P4's dose-response lookup. Recomputed every commit from the *remaining*
budget `B_max_t − B_t`, never the original constant — that's what makes it a landing law rather than a
horizon-sized `rm` schedule wearing a new formula. `T_res` is fixed at **300** and is *not* part of the
remainder (3.3's law-C decision); `ρ_max` is the gate-reachability cap, **not** `ρ*₀` (that clamp is
refuted — 3.3's constants table).

**Composition with the commit gate — checked on replay (T5), and it is the thing that nearly broke this.**
`N_req ∝ ρ_t²` under `gate_rho_ref=annealed`, so an annealing `ρ` demands monotonically *less* pooling
until `I` floors at 1 (§8 failure mode 5). **Flooring is safe but expensive**: `N > n_req` means
`ρ/cos < s`, conservative — what it costs is that the server-side per-commit path loses the trainer work
that was amortising it. **Score round trips per commit, not the floored fraction.** At `T_res`=300 the
two-phase trajectory holds 5.01 (agnews) / 3.69 (yahoo) against a ≥3 gate; at `T_res`=500 yahoo falls to
2.44 and the config is refused.

### 3.5 — saturation stop, replacing the raw `dAcc/dΛ` slope test (revised 2026-08-13, not yet implemented)

**What it's for, and why it is not a duplicate of 3.3.** 3.3's `Φ` stop catches noise-driven collapse:
random-walk displacement in log-norm space (§6's `D` = 0.050 — most of every commit's step is noise, not
signal) accumulating past the point where the local structure the forward-difference estimate relies on
still holds. That failure is FwdLLM-specific and can be catastrophic (0.874→0.296 on `003601`, K-1's K=20
leg, 2026-08-13), and needs no eval history at all. 3.5 catches something structurally different and far
more ordinary: **the model has extracted the generalizable signal the task and capacity allow, full
stop.** Training loss can keep falling past that point (classic overfitting), while held-out accuracy has
flattened or started drifting down. This is the standard ML early-stopping problem, not a FwdLLM one — and
critically, **training loss cannot detect it by construction**, since training loss keeps improving under
overfitting. That's why the field never uses it for this and always uses a held-out metric instead; this
system already computes one (`test-accuracy` via `agg_eval`), so the right ingredient exists, only the
criterion applied to it needs fixing.

**Original spec, superseded by this section:** `dAcc/dΛ` flattens over a ≥100-commit window, resampled
onto `Λ`. Problem: a raw derivative of a noisy signal is itself noisy, and held-out accuracy here *is*
noisy near a turn — ±0.045 between byte-identical replicates vs ±0.0009 at peak (P4.4). A slope test over
noisy points can both false-trigger on a dip and miss a real plateau masked by sampling noise on either
side of the window — the same failure mode P4.2/task 0.5 already found and fixed for raw `Φ` crossings.

**Revised design: a Prechelt-style generalization-loss / patience criterion** (Prechelt, *"Early Stopping —
But When?"* — the standard reference for exactly this problem), applied to smoothed held-out
`test-accuracy`, smoothed the same way `Φ` already is (11-eval trailing window, task 0.5) rather than a raw
per-eval slope.

- Track `Acc_best`, the running max of smoothed held-out accuracy so far — monotone by construction, so a
  single noisy good point can only raise it, never get fooled into resetting the clock the way a two-point
  slope can.
- Generalization loss `GL_t = (Acc_best − Acc_t) / Acc_best` (sign/normalisation TBD at implementation —
  keep it reported alongside `Φ` in the same units family).
- Stop when `GL_t` exceeds a threshold **for a patience window** — both the threshold and the patience are
  parameters to size empirically by replay against the arms already on disk (P4's portfolio plus
  2026-08-13's K-1/P-1/G-2 arms), the same way task 0.5 sized `Φ`'s smoothing window and P4.1 sized its
  threshold.
- **Also the citable, standard formulation** for a paper's methodology section, rather than a bespoke slope
  test that needs its own justification from scratch.

**Edge cases (carried over, still apply).** Eval cadence and commit cadence are different clocks —
resample onto `Λ`, never onto `comm_round`. Combine with 3.3 as `stop = Φ-cross OR saturation`, whichever
fires first (model §5.5f D2) — the two are answering different questions and neither supersedes the other.

**Not yet sized: the smoothing window, the `GL` threshold, and the patience count. Size all three by replay
before this ships — in the implementing session, not now.**

### 3.4 — adaptive `K`/`C` and `P`

**K-1 landed (2026-08-13): hill-climb `C`, not `K`** — commit throughput flat in `K` at fixed `C` (P3),
confirming model §5.2's own prediction that `K`'s effect on wall clock is "open — `∝1/K` only if `C=K`".
**P-1 landed (2026-08-15): compute-bound**, `τ(30)/τ(10)`=2.56 — report and stop (P3/P4.6). Adaptive `P` is
no longer motivated as a throughput lever; a **mid-run `P` change**, which no code path supports today, is
the real remaining engineering here if a bandwidth case ever revives it, and it also forces 0.4's per-commit
`G_rule`.
`dynamic_kc`'s `k_max` = 15 is backwards and must not be reused as a starting point. Whether `K` needs to
be dynamic anyway under *variable* device availability, or matters more for forward- than backprop-trained
gradients, is untested — H-T (`fl_fwd_ft_practice.md` P5.3), not blocking this task.

**Do not wire `n_eff` to any of this** (P6: it is an identity, 1.00 ± 0.01 over 17 arms).

---

## §6 — Phase 4 · the two zero-input runs

**4.1** agnews · **4.2** yahoo — *same binary, same flags*, no operator input beyond model / PEFT / `p`.

**Acceptance:** both reach their plateau and **end within 0.015 of peak**, with sensed `B_max`, `ρ*`, `K`,
`P` logged per run and **differing between datasets without anyone having supplied them**.

**Read the run while it is alive:** every arm reports `B` as a fraction of `B_max` and `A` against P4's
calibration. Both are exact at any horizon, so both failure modes are diagnosable ~20 commits in — do not
wait for the accuracy curve.

**Scoring, from [P4.4](fl_fwd_ft_practice.md#p44-scoring-rules-for-any-ab):** score **peak** accuracy and
the stability columns, never final accuracy of a diverging arm (±0.045 between byte-identical replicates
past the turn, against ±0.0009 at peak). Compare yahoo to agnews on **`A` and per-vclock-hour**, never on
`Λ` (different `p`) and never per round (11.7× different bins/round, §1).

---

## §7 — P5.3 · the open hypotheses, as buildable specs

| ID | rung | what to build | the discriminating number |
|---|---|---|---|
| **H-S** | 2 (offline rig) | Same `v`; true `⟨g,v⟩` by backprop vs the shipped central FD, correlated over many draws. `scripts/probe_fd_chord.py` is the site | `cos(d_FD, d_true)` ≈ **0.3** if the FD chord is the residual 3.5× in `S`, ≈ **1** if not. A chord-averaged slope is still Gaussian, which is why P3's distribution check passed it |
| **H-H** | 2 | The central difference already computes `L(θ+hv)` and `L(θ−hv)` and throws their **sum** away; it is `≈ h²vᵀHv`. Log it, correlate with realised loss decrease **at the step scale actually taken** | a usable correlation makes curvature the first probe-selection criterion that is not stability-neutral |
| **K-C** | 1, then 4 | Rung 1 = task 0.6. Rung 4 = **K-1** | commit rate flat in `K` within ±15% ⇒ `K` is not a time lever and the controller hill-climbs `C` |
| **H-J** | 1, then 4 | `B` extrapolates it exactly — **demoted to a confirmation**, run only if a cheap node is free | — |

**If probe selection is ever retained**, select on something other than `|d|` (P6 closed that): curvature
`vᵀHv` (≈free), split-half SNR within the bin (free, orthogonal to `|d|`), or loss decrease at the step
scale (under trust-ratio the step size is known ahead, 1 pass per candidate).

**Predicted inert, settle on paper only:** block-coordinate probing — progress/commit falls as `1/√L`
while budget/commit falls as `1/L`, so progress per unit `B` is unchanged and it needs `L`× the commits.
It escapes `√(n/p)` only if the gradient is *unevenly* spread.

---

## §8 — Failure modes this plan is written against

*Every one of these has already happened once ([P9.3](fl_fwd_ft_practice.md#p93-process-lessons)).*

1. **A superseded constant left in a config.** When a doc supersedes a value, grep the configs in the same
   edit. `gate_safety_s` = 0.4 cost three runs and a wasted node.
2. **An emit-only flag never re-costed after being made correct.** B17 multiplied the audit's per-commit
   wall by 16×, and the vclock looks healthy right up to the moment the runaway safety fires.
3. **A sinking condition without its precondition or its smoothing rule.** State both, before launch.
4. **An instrument whose arithmetic is right and whose input is not.** A probe needs a test that its
   *input* is what you think it is — that is what `check_partitions.py` is for on the data side.
5. **Scoring a feature without scoring the composition.** Gate and anneal are each correct and multiply
   into a stall (`N_req ∝ ρ_t²`). Any two new flags need a composition test before both default on.
6. **Extrapolating a progress rate as an accuracy rate.** `A` accumulates *through* the turn while accuracy
   falls. Extrapolate `A` only alongside `Φ`.
