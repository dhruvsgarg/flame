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

### 0.8 — `--dataset` in the launcher *(landed — see §1)*

### 0.9 — partitions for yahoo / yelp-p *(landed — see §1)*

### 0.10 — the dataset-switch checklist *(landed — see §1)*

---

## §3 — Phase 1 · B-1 · 1 GPU, no FL stack · the decision gate

**This is the highest-value GPU work in the queue and it needs no FL stack.** It decides whether the
controller can ship a fixed `Φ` target or needs an online probe (3.1).

1. **Validation gate first, non-negotiable.** Run `scripts/probe_inflation_damage.py` **unchanged** on
   agnews and reproduce model §7.1: knee at `Φ` ≈ 3.0 plus the three-mode separation. **Record how the
   base model is obtained** — matching a rig on *accuracy* is not matching it (P9.3: at acc 0.57 a
   backprop model reads `‖g_test‖` = 1.95 against every arm's 0.25–0.32; match on the quantity the
   measurement depends on). **If it does not reproduce, stop and fix the rig.**
2. Wire yahoo and yelp-p through `dataset_registry`. `num_labels` from `label_vocab`; assert `p` =
   454,954 / 448,802 against `[ProbeDim]`.
3. Sweep `Φ` ∈ {1.5, 2, 2.5, 3, 3.5, 4} × 3 datasets, **reps ≥ 3**, mode = `noise`.
4. Score `Φ_knee` vs `num_labels`; write into model §5.5b, §7.1 and P10.

**Pre-registered prediction (do not edit after seeing data):** `Φ_peak` falls as class count rises —
yelp-p > agnews (2.7) > yahoo.

| outcome | what changes |
|---|---|
| invariant ±0.3 | `B_max` is universal — ship a fixed `Φ` target, 3.1 becomes unnecessary |
| monotone in classes | `B_max` derivable from `num_labels` — still zero-profiling |
| erratic | the injection probe becomes **mandatory online infrastructure** (3.1 is on the critical path) |

**Edge cases.** (a) The injection is on a **copy** of `θ_tr` — the probe must not perturb the model under
test. (b) `h` must be rescaled when `p` changes (P9.1: `[ProbeDim]` prints nominal `h`, `[FD]` prints the
real one — read `h√p` off `[FD]`). (c) yelp-p's 2 classes make `loss > ln(num_classes)` a much tighter
bound; do not reuse agnews' absolute loss thresholds. (d) Report `Φ_knee` with its rep spread — a
single-rep knee is not a knee.

---

## §4 — Phase 2 · the three registered nodes

**Predictions and kills are in [P5.1](fl_fwd_ft_practice.md#p51-registered-nodes) and are pre-registered
(R6) — implement against them, do not restate or soften them.** Common launch shape:
`expt_scripts/nodes/run_node_*.sh` built on `_node_lib.sh` (which aborts a node on any arm producing
< 5 commits); `run_node_g1b_gate_s.sh` is the current best template.

**Standing preflight for every arm here** (do not skip; this list is what the eight dead arms were missing):

```
--dry-run first, then --only --yes --clean --force ; --num-trainers 100
assert ceil(n_req/K) <= max_iter        # existing
assert projected_real_wall < sim_wall_ceiling_s   # task 0.7; now --dry-run's own preflight, no longer by hand
pin the pool for any A/B: --var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off
echo the enactment lines: [ServerStep] [CommitGate] [CosProbe] [probe_combine] [ProbeDim] [FD] spacing
```

**K-1 leads** — it is the only node that changes a phase-3 design decision (which knob 3.4 climbs), and
task 0.6 sizes it. `K` ∈ {10,20,30} at fixed `n_req`, **`C` = 30 on all three** — the first arm ever to
move `K` without `C`. Needs **≥1000 commits/arm** for ~40 cos fires, which at stride 25 is affordable.
Kill immediately if `n_req` ≠ 72, `ρ` ≠ 0.06, or `C` ≠ 30 on any arm.

**P-1** is a build prerequisite, not an ablation — 3.4 needs `τ(P)`. `P` ∈ {10,30} under
**`probe_combine=mean`**, `N` pinned. `P` is a **trainer-side** override: verify via
`[probe_combine=mean] P=…` in the trainer log, not in `aggregator_config.json`. Report and stop if
`τ(30)/τ(10)` ≥ 2.5 — compute-bound, the gain cancels.

**G-2** re-tests a P2 default that was changed on a re-read of two old arms rather than an A/B:
`gate_rho_ref` `annealed` vs `setpoint` at `ρ*`=0.06, `rm`, **matched vclock** (not matched commits — the
two rules order differently under the two clocks, which is the entire question). **Never run `annealed`
at `ρ*` ≤ 0.01** — `220627`'s dead zone.

---

## §5 — Phase 3 · the controller · multi-day

**None of this needs new science.** Design in model §5.5f. Each component is a flag, default-off,
byte-identical until its A/B scores. **Build 3.3 first**: the whole P4.1 gain (0.141 → 0.005) is there and
it depends on nothing else.

### 3.3 — budget-landing anneal + `Φ` stop

**Sensed from `B`**, which is exact and free (`B = ½Σlog(1+ρ_t²)`, already computable per commit).

- **Anneal:** replace the horizon-sized `rm` exponent with the landing law `ρ* = √(2·B_max/T_res)`
  (model §4.6a), recomputed each commit from remaining budget and remaining control resolution.
- **Stop:** halt when smoothed `Φ` crosses the threshold. **The rule needs no eval, no accuracy history
  and no task constant** — that is its whole value. Threshold default 2.7 (P4.1: 0.0054 given up, worst
  arm 0.0139); its value on an unseen task is what B-1 settles.

**State to carry:** cumulative `B`, commit count, `Φ` smoothed over the same 11-eval window the replay
uses. **Emit per commit:** `B`, `B_max`, `B/B_max`, `Φ`, `rho_star_t`, and the stop reason when it fires.

**Edge cases.** (a) `T_res` (remaining control resolution) must be finite and positive — floor it, or the
first commit divides by zero. (b) Under `β > 0` the `Φ` law changes; either refuse to combine the stop
with momentum or use the momentum-corrected form. (c) A run that reaches `B_max` before its eval cadence
fires must still stop — the trigger is `B`, not accuracy. (d) `Φ` from `ρ` is exact; **never** re-derive
it from `‖θ‖` ratios, which carry the audit's own noise. (e) Stopping is not the same as ending the run:
decide and document whether the aggregator halts, freezes `θ`, or keeps evaluating.

### 3.1 — two-phase `B_max`

Prior `ln 2` → injection probe on a **copy** of `θ_tr`, ~6 evals, on a stride. **Phase 1 decides whether
this is needed at all** — if `Φ_knee` is invariant or derivable from `num_labels`, ship the constant.

**Edge cases.** The probe must be strided and budgeted (0.7); on a copy, never the live model; and its
cost must be reported as a fraction of the commit path before it defaults on.

### 3.2 — `ρ*` = `√(2·B_max/T_res)`

Trivial once 3.1 lands. Replaces P4's dose-response lookup. **Do not walk `ρ*` up** — P6.

### 3.5 — saturation stop on `dAcc/dΛ`

The second stop; 3.3's `Φ` stop covers the destructive case alone, so this one only buys wall clock.
Window ≥ 100 commits. **Edge case:** eval cadence and commit cadence are different clocks — resample onto
`Λ` before differencing, and never onto `comm_round`.

### 3.4 — adaptive `K`/`C` and `P`

**Blocked on K-1 (which knob) and P-1 (`τ(P)`).** Needs a **mid-run `P` change**, which no code path
supports today — that is the real engineering here, and it also forces 0.4's per-commit `G_rule`.
`dynamic_kc`'s `k_max` = 15 is backwards and must not be reused as a starting point.

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
