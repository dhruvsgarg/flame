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

## §-1 — Next session, in order *(written 2026-08-16 night, nothing running)*

**Where things stand.** Phase 4's first four arms ran and are void as an acceptance test; they found five
defects, all fixed and gated
([P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)).
The law itself beat its control on both datasets, so the controller is not in question — its plumbing was.
**Nothing is running. No arm is in flight.**

| # | task | cost | done when |
|---|---|---|---|
| **A** | **Rung 1 of §9 — `expt_scripts/probe_backprop_ceiling.py`** (written and calibrated on agnews 2026-08-16; yahoo launched that night, result not yet read) | 1 GPU; minutes once the tokenizer cache is warm, ~1 h cold | a number. **≈0.70 ⇒ the data path is clean and yahoo is purely under-trained; ≈0.30 ⇒ the path is at fault and B–D are moot.** Run this first: it is the only cheap test that can *falsify* the under-training story |
| **B** | `profile_sim_charges.py --dataset yahoo` | CPU + 1 GPU, ~20 min | `sim_charge_profiles/fluxtune_yahoo.yaml` exists and the sim-profile preflight passes **without** `--force` |
| **C** | **Re-run P-4 agnews** (`run_node_p4.sh agnews controller` + `control`) | 2 × ~2.5 h | the four in-flight checks in §6 all hold, and the controller arm ends on `[BudgetStop]`. This is the arm that scores the controller |
| **D** | **Re-run P-4 yahoo**, `--eval-max-samples 10000`, vclock ≫ 40,000, after B | 2 × ~6 h | same four checks; then score accuracy vs `Λ` against §5's first open row |
| **E** | 3.5's saturation stop — size window/threshold/patience by replay first | CPU | replay reproduces a sensible stop commit on the arms already on disk |

**Do A before C/D if only one slot is free.** C and D are the expensive arms and both are worthless if A
comes back at 0.30 — that would mean the yahoo *data path*, not the controller, is what needs fixing, and
every yahoo arm to date would be measuring the wrong thing.

**Two decisions already made, so nobody re-opens them:** `b_max_policy` defaults to **`mean`** (§5's
constants table has the argument), and databin **size stays 8 while the bin count moves per dataset** (§1).

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

**The sweep missed one, found 2026-08-16 (P4.7 defect 4).** `total_data_bins = 150` was hardcoded in
`flame/mode/horizontal/syncfl/fwdllm_aggregator.py` — outside the `expt_scripts scripts aggregator
trainer` paths task 0.10 grepped, and not matched by `agnews|450340|N_CLASSES = 4` even if it had been.
It is agnews' `1,200/8` exactly, and the trainer indexes its **own** batch list with `data_id`
(`FedSgdTrainer.py:520`), so yahoo trained on the first 1,200 of each client's 14,000 samples — **8.6% of
the dataset, the same 1,200 every lap** — and yelp-p would have used 23%. Nothing raises: the list is
merely longer than the index. Now derived from `dataset_registry.total_data_bins(name, C, batch)`
(150 / 1,750 / 650 — the table above), echoed as `[DataBins]`, agnews byte-identical.
**Lesson for the next dataset switch: grep `lib/python/flame/` too, and grep for the derived agnews
constants (150, 1200, 7600, 192), not just the dataset's name.**

**Still not done:** `profile_sim_charges.py` has never run for yahoo or yelp-p, so every non-agnews sim
arm to date mis-prices its vclock and needed `--force` to launch (both 2026-08-16 yahoo arms did).

---

## §2 — Phase 0 · unblock the instruments · no GPU *(all 10 landed 2026-08-12/15)*

**Specs deleted per §0 rule 6.** What exists, and where the result lives:

| task | what shipped | result |
|---|---|---|
| 0.1 | `replay_scoring.py --cos`, blocks by probe *fires* (`--cos-block-fires`) | [P4.2](fl_fwd_ft_practice.md#p42-g-1b--s-is-efficiency-not-safety) |
| 0.2 | `resolve_p()`, three-tier (`[ProbeDim]` → registry → `P_BY_RF`, tier 3 warns) | P3.1 |
| 0.3 | class-skew threshold from `dataset_registry.max_dominant_share(K)` | P3.1 |
| 0.4 | per-commit `G_rule_t` in `enrich()`; `p_probes` is a one-line switch when 3.4 lands | P3.1 |
| 0.5 | `replay_phi_stop.py` — 11-eval trailing smoothing, raw-peak learned gate | [P4.1](fl_fwd_ft_practice.md#p41-the-φ-stop-counterfactual) |
| 0.6 | `replay_kc_rung1.py` — reads `agg_round` telemetry directly | P3 "`K` at fixed `n_req`" |
| 0.7 / 0.7a | `expts/wall_clock_preflight.py`, called from `run_sequential.sh`; law-C branch walks `landing_law.simulate` and costs it with T5's two-term fit; `gate_starved` refuses below 3 trips/commit | P3 "real-wall cost model" |
| 0.8–0.10 | `--dataset NAME`, yahoo/yelp-p partitions, the dataset-switch sweep | §1 |

**Two of these have known gaps, both live:** 0.7's projection prices law C off the `ln 2` prior, so a
run can pass the ≥3 trips/commit check at launch and breach it in flight once `B_max` is sensed (P4.7
defect 3) — read trips/commit **per quintile at run time**, not once. And 0.10's sweep did not cover
`lib/python/flame/`, which is where the `total_data_bins` bug lived (§1).

---

## §3 — Phase 1 · B-1 *(landed 2026-08-13, replicated same day)*

**Spec deleted.** Swept `Φ` ∈ {1.5…4} × 3 datasets, reps ≥ 3, mode=noise, 2 independent runs each.
Pre-registered monotone-in-classes prediction **refuted**; knees are erratic (agnews ~3.0–3.5, yahoo and
yelp-p both ~2.0–2.3). Consequence: **3.1's online probe is mandatory infrastructure, not a fallback.**
Full result: model §7.1/§5.5b, P10, P5.2 Phase 1. `expts/prep_b1_configs.py` generates the per-dataset
configs.

> **`probe_inflation_damage.py` reads accuracy off `test_global[:2000]`**, and `test_index_list` is
> per-client shards concatenated in client order — so B-1's base accuracies (agnews ~0.88, **yahoo 0.73**)
> are measured on a mildly Dirichlet-skewed slice, not the balanced full test set the FL arms use. It does
> not move B-1's verdict (the knee is read on chance-normalized accuracy within one curve), but **the 0.73
> is not measured on the same distribution as an FL arm's 0.30** — see §9.

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
assert projected_real_wall < sim_wall_ceiling_s   # task 0.7, now --dry-run's own preflight
pin the pool for any A/B: --var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off
echo the enactment lines: [ServerStep] [CommitGate] [ProbeDim] [DataBins] [Landing] [BmaxProbe]
```

**Then read these four IN FLIGHT, ~200 commits in — a clean `--dry-run` is a prior, not a guarantee**
(each of the four cost a node): `[DataBins]` coverage is 100% · no `rho_star` = 0 · trips/commit ≥ 3 per
quintile · a controller arm ends on `[BudgetStop]`, never `max_runtime_s`. G-2's `003648` died with the
gate's `I` floored at 1 on **98%** of its commits (8.40 s/round-trip against `145729`'s 1.75) and the
preflight could not see it, because `τ(K)` prices round trips and that arm converted its budget into
commits instead.

---

## §5 — Phase 3 · the controller

**3.1 + 3.2 + 3.3 shipped 2026-08-15 as one closed-loop controller and were corrected 2026-08-16 by their
first four arms** ([P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)).
Specs deleted per §0 rule 6; **what survives here is every decision that still binds a re-run, plus the two
components that are not built** (3.5, and 3.4's mid-run `P`).

### What is settled, and what refuted the alternative

| | value | why |
|---|---|---|
| anneal law | **C** — `ρ*_t = min(ρ_max, √(2·(B_max_t − B_t)/T_res))`, `T_res` a rate **never decremented** | A makes `ρ*` exactly constant under perfect tracking and smuggles `T` back as an input (§4.6a); B is a receding horizon that never terminates. C dissolves every edge case and approaches `B_max` monotonically **from below**, so the stop is a genuine backstop |
| `T_res` | **300** | 500 refuted on replay: 2.46 trips/commit on yahoo, 2.26 on the `ln 2` prior, against the ≥3 gate |
| `f` (stop at `B ≥ f·B_max`) | **0.95** | yahoo needs ≥0.90 to clear `Λ` ≥ 0.95; agnews clears at 0.70. Free given the wall headroom. **Re-derive it — T5 sized it against the pre-fix `B_max` semantics** |
| `ρ_max` | `s·√(max_iter·K·G_rule/p)` ≈ **0.0999** | gate reachability, mechanical. **Not** a `ρ* ≤ ρ*₀` clamp — that would block a re-sense from ever spending the budget it just found |
| `B_max` prior | `ln 2` | D1. Replaced outright by the first sense, never averaged into it |
| `B_max` origin | **`B + ln Φ_knee`** | the probe measures headroom from `θ_t`; `B` accumulates from `θ_0` (P4.7 defect 2) |
| `B_max` combiner | **`mean`** over fires (`b_max_policy`) | P4.1's fixed-`Φ` stop working across 10 arms with different schedules is the evidence that the boundary is a property of *total* inflation — i.e. that the fires are noisy estimates of **one constant**, so the estimator for a constant is the right combiner. The anchored quantity behaves like one (`B+R` CV ≈ 25% on agnews, ±6% on yahoo, against 2.4× spread in raw `R`). It also terminates, which `anchor` does not: `B_max` settles while law C drives `B` up to it |
| what the stop does | **`halt`** via `_work_done` | one line into a tested path; freezing `θ` is new lifecycle state, and re-evaluating a frozen model measures eval noise at GPU cost. Ship three states: `off` (default) · `log_only` (emit the crossing, keep training — how the rule gets validated on an unseen task) · `halt` |
| stop reasons | `budget` · `phi_fixed` (no sensed `B_max`) · `saturation` (3.5) | one predicate, one code path |

**Λ = 2B/s is an identity wherever the gate holds `s`** (T5, −0.3% out of sample on both `s`-pinned arms,
+21.5–23.3% on the three whose `s` drifts). **Consequence: the `ρ` schedule is `Λ`-neutral at fixed `B`** —
law A and law C bank the same `Λ` and differ only in commits spent (C takes 3.1×, ~0.6 h against a 10 h
slot). Any "this schedule learns more" claim is a comparison at unequal `B`.

**Real-wall cost model (T5, ±1% over 5 arms):** `wall = 7.81·commits + 0.77·trips` with the audit on at
stride 25; **audit-off is 4.41 s/commit + 0.77 s/trip** — a subtraction, not a measurement, still unverified.

**Still-binding cautions.** (a) `N_req ∝ ρ_t²` under `gate_rho_ref=annealed`, so an annealing `ρ` demands
monotonically less pooling until `I` floors at 1. Flooring is *safe* (`N > n_req` ⇒ conservative) but
strips the trainer work amortising the server path — **score trips/commit per quintile, at run time**.
(b) `ρ` = 0 is a requirement of *zero*, not an absent one (P4.7 defect 3). (c) Under `β > 0` the `Φ` law
changes — refuse to launch. (d) `Φ` from `ρ` is exact; never re-derive it from `‖θ‖` ratios.
(e) `Φ`-stop and budget-exhausted are the **same** trigger: `Φ = e^B`, `B` monotone, `B_max = ln Φ_peak`.
(f) The 3.1 probe must reuse the cos probe's fixed-seed reference batch **and its class-skew guard** —
with the audit off on Phase-4 arms that guard is otherwise not running at all.

### Open: what the first correct arm has to answer

| open | closes on | if it comes out wrong |
|---|---|---|
| **`Λ` → accuracy on yahoo.** T5 cleared yahoo at `Λ` = 1.04 against a **≥0.95 floor read off P4's agnews curve**; `Λ` has never been tested across *task* | node 3/4's own accuracy-vs-`Λ` curve, the first measured off agnews | the floor moves and yahoo needs more than 0.95·`B_max`. See §9 — the 2026-08-16 arms banked 0.40–0.46 |
| **Does `B_max` drift within one run?** (3.1 question 2) | `sensed=` per fire on nodes 1/3 — still emitted under every `b_max_policy` | if it drifts a lot, `mean` lags and an EWMA is the fallback; law C self-corrects, so control is unaffected |
| **Is `mean` the right combiner?** (question 3, new) | replay nodes 1/3 under all three policies after the fact — free once the arm exists | `ratchet` stops sooner, `anchor` may not stop. Terminal-flag call, operator's |
| **trips/commit ≥ 3** is calibrated on **one death and two survivals** | every arm reports it per commit; re-size once there are ten | a config passes preflight and still burns wall — now also breached *in flight*, see P4.7 |
| **4.41 s/commit audit-off** is `7.81 − 85/25`, a subtraction | node 1's own `wall / commits`, read ~50 commits in | projections shift; 4× headroom absorbs a 2× miss |
| **`f` = 0.95** rests on the `Λ` floor, with **no accuracy evidence**, and was sized pre-fix | replay nodes 1/3 at every `f` after the fact | a smaller `f` ends runs sooner at the same peak, which is a win |

**Predicted enactment (T5, pre-registered, and NOT met by the 2026-08-16 arms):** prior `ln 2` → sensed,
re-sense at commit 150, `T_res`=300: agnews `ρ*` 0.0530 → **0.0764** (`I` 6 → 12); yahoo 0.0530 →
**0.0573** (`I` 6 → 7). Both went to **0** instead, from defect 2. **Registered before the re-run:** with
the origin fixed the two knees at commit 150 were 0.248 / 0.237 ⇒ `ρ*` 0.0407 / 0.0397 — nearly identical,
so **the divergence may not reproduce**. The live knee tracks how far the model sits above chance (agnews
base 0.809 vs yahoo 0.311), which is a property of the *sensor*, not the task. If it does not diverge,
that is a finding about 3.1 and it moves the first open row above.


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

**First four arms ran 2026-08-16 and are VOID as an acceptance test** — result and the four defects they
found are in [P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects);
why yahoo sits at 0.30 is [P4.8](fl_fwd_ft_practice.md#p48-yahoo-is-under-trained-not-broken). The law
itself won on both datasets (agnews peak 0.857 vs the control's 0.835, and it reached 0.83 at 62% of the
vclock budget against 92%) while **23–48% of its commits took a step of length zero**, so that margin is a
floor. All four defects are fixed; the specs they invalidated are corrected in place below.

**Re-launch gate — check all four in the first 200 commits, before letting an arm run unattended:**

```
[DataBins] total_data_bins=1750 source=registry dataset=yahoo   # not 150
no server_update record with rho_star == 0                      # defect 2
controller ends on [BudgetStop] reason=budget, not max_runtime_s # defect 1
trips/commit >= 3 at every quintile                             # defect 3 + T5's gate
```

Plus three that are not defects but invalidate the scoring:

- **Run `profile_sim_charges.py` for yahoo.** Every `sim_charge_profiles/*.yaml` is agnews-profiled; yahoo
  burns **0.658 real-s per vclock-s against agnews' 0.255**, so it needs `--force` to launch and no
  yahoo-vs-agnews per-vclock comparison is valid until it is profiled. Yahoo-vs-yahoo is fine.
- **Budget yahoo far above 40,000 vclock.** Its control was still climbing monotonically at 0.296 when
  killed at 86% of that budget, against a 0.73 backprop reference (§9).
- **Set `--eval-max-samples 10000` on yahoo.** The eval *is* backgrounded, but `_eval_snapshot_model`
  blocks the main thread when the previous eval is still running — and it was, on **359 of 359** evals on
  the yahoo control and 286 of 289 on the yahoo controller (agnews: 287/488 and 0/400). 60,000 rows at seq
  256 costs ~30 GPU-s idle and **89.2 s measured under trainer contention**, against a ~91 s inter-eval gap
  at stride 2. Batch size is not the lever — measured 1.23× from batch 8 → 128 on an idle A40, because the
  pass is compute-bound, not launch-bound. The subsample is **fixed and shuffled** (seed 20260810), so its
  sampling error is a constant offset rather than per-eval noise and peak-vs-final stays as precise as the
  full set — which is what P4.4's 0.015 bar compares.

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
7. **Two quantities with different origins, subtracted.** `B_max` (measured from `θ_t`) minus `B` (from
   `θ_0`) type-checks, runs, and is meaningless. State the origin of every accumulated quantity next to
   its formula (P4.7 defect 2).
8. **A flag whose writer is not its only writer.** `_work_done` had two, one of which assigned rather than
   OR'd, so a fired stop was silently un-set (P4.7 defect 1). Grep every assignment to shared run state
   before adding a writer.
9. **A dataset constant that is right on agnews by arithmetic coincidence.** `total_data_bins` = 150 *is*
   agnews' `1,200/8`, so it looked correct forever and silently gave yahoo 8.6% of its data. When a switch
   lands, grep for the *derived* agnews numbers (150 / 1,200 / 7,600 / 192), not just `"agnews"`, and grep
   `lib/python/flame/` as well as the example tree.

---

## §9 — The yahoo gap · a sanity ladder, cheapest first

**The question.** B-1's backprop reference reaches **0.73** on yahoo; the P-4 arms reach **0.30**
([P4.8](fl_fwd_ft_practice.md#p48-yahoo-is-under-trained-not-broken)). Every fingerprint on disk says
under-training rather than a broken pipeline — entropy pinned at `ln 10`, logit norm 0.34, MCC 0.23 (real
signal), accuracy monotone and still rising when killed, `Λ` = 0.40–0.46 against T5's ≥0.95 floor. **But
"consistent with" is not "shown", and the ladder below is what would show it.** Order is by
cost-per-bit-of-information; stop as soon as a rung indicts the implementation.

**Rung 0 — data plumbing · CPU · minutes.** *Now largely automated; run it, don't re-derive it.*

| check | how | what a failure means |
|---|---|---|
| every sample reachable | `dataset_registry.data_coverage(ds,C,batch)["exact"]` — 6 cases in `test_dataset_launcher.py`, and the aggregator logs it as `[DataBins] coverage` at init | the §1 bin-cap bug, or a new one |
| shards equal, disjoint, full coverage | `check_partitions.py` on yahoo α=1 `C`=100 (6/6) | the partition, not the trainer |
| the aggregator and the trainers agree | `[DataBins] confirmed by trainer` — the trainer's own `len(train_local[0])` arrives on the wire and is now cross-checked | the derivation disagrees with the loader |
| the run visits every bin | `data_id` max over a long arm == `total_data_bins − 1` | `max_data_id_progress` or a lap bug |
| the visited union covers all 10 classes | label histogram over the bins actually trained on | a skew the α=1 draw does not explain |

**Verified 2026-08-16: `bins × 8 × C == n_train` exactly, for all three datasets at `C` = 100 *and*
1,000** (agnews 150/15, yahoo 1,750/175, yelp-p 650/65 — shards are equal to the row on every client).
Exactness is not automatic: it needs equal shards **and** `shard % batch == 0`, so `data_coverage()`
returns both remainders and the aggregator warns rather than assumes. **Batch stays 8 and the bin count
moves per dataset** — the batch is the unit each JVP is estimated on, so changing it changes the
estimator, while the bin count is pure addressing.

**Rung 1 — the discriminating test · `expt_scripts/probe_backprop_ceiling.py` (new 2026-08-16) · 1 GPU.
Run this first if only one rung fits.** Centralized AdamW on the **FL rig's own data path** — client
shards through `TextClassificationDataManager`, the exact call `trainer/main.py` makes, capped at
`total_data_bins` so it sees exactly the rows an FL arm can reach — evaluated on the **same**
`test_global` `agg_eval` uses, plus the fixed 10k subsample so its offset is measured rather than assumed.
It instantiates `ForwardTextClassificationTrainer` purely to drop `pre_classifier`, which is what makes
`p` **450,340 rather than 1,040,932** — without it the rig trains a bigger model than any arm runs. Not
`probe_inflation_damage.py`'s rig, which trains on half the *test-global* tensor and reads accuracy off
`test_global[:2000]` (per-client shards in client order, so a skewed slice — top class 20.7% against a
balanced 10%).

```
./probe_backprop_ceiling.py --config <a yahoo run's aggregator_config.json> --clients 10 --epochs 3
```

**Pre-register: ≈0.70 clears the entire data path — tokenization, label vocab, `max_seq_length`=256, the
h5 ranges, the partition, the loader — and makes the gap purely optimization. ≈0.30 indicts the path and
every rung below is moot.**

**Calibrated on agnews first (2026-08-16):** 3 client shards, 3,600 rows, **one** epoch → **0.850** on the
full 7,600-row test set, against the FL control's 0.835 after 801 commits and 2.3 h. Same data path, same
`p`, same test set; the only change is an exact gradient instead of pooled forward differences. That is
the scale of the optimization-budget gap yahoo's number has to be read against. The subsample offset
measured **−0.0089** at 2,000 rows — small, and constant by construction.

> **Cost note:** tokenization dominates, not training. Yahoo's client shards are 14,000 rows each at
> ~40 rows/s uncached, so `--clients 10` is ~1 h of tokenizing before the first epoch. The cache
> (`cache_dir/..._256_..._yahoo_..._<client_idx>`) is shared with the FL runs, so **run this after a yahoo
> FL arm has warmed it** and it costs minutes instead.

**Rung 2 — is the estimator itself weaker on yahoo? · 1 GPU · ~1 h.** Only if rung 1 clears.
- **cos audit on yahoo**, `cos_ground_truth_audit` on for ~100 commits, `replay_scoring.py --cos`. Compare
  `D` against agnews' 0.10–0.15 ([P4.2](fl_fwd_ft_practice.md#p42-g-1b--s-is-efficiency-not-safety)). A
  materially lower `D` means the forward estimate degrades with 10 classes / seq 256, which is a real
  FwdLLM-layer finding and not a controller one.
- **H-S on yahoo** (`probe_fd_chord.py`): is the FD chord still faithful at `max_seq_length` 256?

**Rung 3 — the budget answer · FL · the actual re-run.** Score accuracy against `Λ`, not against commits.
**Pre-register both branches:** yahoo reaching ~0.6–0.7 by `Λ` ≈ 1.0 ⇒ it was a budget problem and the
agnews `Λ`-curve transfers; yahoo plateauing near 0.35 with `Λ` > 1.0 ⇒ **`Λ` does not transfer across
task**, which closes T5's first open row and is a more interesting result than the arm itself.

**Knobs to suspect, in order, and why the obvious ones are already cleared.** `p` = 454,954 confirmed from
`[ProbeDim]` (10-way head); `max_seq_length` 256 and `num_labels` 10 plumbed on both sides; `learning_rate`
is inert under `trust_ratio` (the scale is `ρ‖θ_tr‖/‖G‖`); `G_rule`, `s`, `P` and `probe_combine` are
dataset-free. What is left is genuinely dataset-shaped: **(a)** the `ρ*` band was sized on agnews (§4.6a)
and a 10-class head may need a larger relative step to leave its init — the sensed `B_max` is supposed to
discover this, and defect 2 is exactly why it could not; **(b)** `train_batch_size` = 8 means each JVP is
estimated on a batch missing most of the 10 classes, where 8 samples cover most of agnews' 4; **(c)** seq
256 truncates ~11% of yahoo documents. None of the three is worth an arm until rung 1 has cleared the
data path.
