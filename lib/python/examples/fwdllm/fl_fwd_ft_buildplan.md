# Build plan — **the status doc**: where things stand, what is next, and how to build it

> **This is the one file to read for status and next steps, and the one to update in place.**
> [§-1](#-1--status-board) is the status board: what is running, what each dataset's state is, and the
> ordered queue. [§-0](#-0--the-zero-input-claim-as-a-checklist) is the generality claim as a checklist.
> Everything from §0 down is one spec per queued task — files, algorithm, edge cases, sanity gate — and
> [§11](#11--runbook--the-exact-invocation-for-every-queued-task) is the runbook: the exact command for
> each, copy-pasteable.
>
> **The other two docs are evidence, not status.**
> [fl_fwd_ft_practice.md](fl_fwd_ft_practice.md) owns *what is true*: the P3 knob ledger, the P4 arm
> ledger, P6's dead ends, P8's reproduction recipes. Every number cited here lives there.
> [fl_fwd_ft_solution.md](fl_fwd_ft_solution.md) owns *why* — the model. Cited as "model §x".
>
> **When a task lands:** its result goes to P3/P4, its row here moves to `done` and its spec is deleted
> (R3). **Read [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) before proposing any change.**

---

## §-1 — Status board

*Updated in place. State read **2026-08-17 afternoon**: nothing running, 8 GPUs idle. **Task F landed
this session** (F1–F3 + the pre-tokenizer + task B's profile plumbing + the in-flight gate reader);
the queue below is what is left.*

**Where the program is.** Phases 0–2 landed. **The Phase-3 controller is built and, as of 2026-08-16,
correct** — its first four arms found four defects, all fixed and now verified live on both datasets
(agnews `225224`, yahoo `234931`,
[P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)).
The law beat its control on both datasets, so the controller was never in question — its plumbing was.
**Phase 4 is the whole remaining question and it has no scored arm under the fixed code.**

### Per-dataset state

| | agnews | yahoo | yelp-p |
|---|---|---|---|
| registry row, partitions, `--dataset` | **done** | **done** | **done** |
| `check_partitions.py`, exact bin coverage | **6/6, exact** | **6/6, exact** | **6/6, exact** |
| tokenizer cache — **one** shared `/coc/scratch/…/cache_dir`, runs *and* probes (§10 F1, landed) | **101/101 @ 192** | **101/101 @ 256** | **101/101 @ 256** |
| has ever run the FL stack | many arms | 3 arms | **never launched** |
| sim charge profile | `fluxtune.yaml` | **missing** (task B) | **missing** |
| scored under the fixed controller | **no — task C** | **no — task D** | **no — task H** |
| backprop ceiling (§9 rung 1) | **0.850** | **launched, result lost** (task A) | not run |

### Ordered queue — and the wave it runs in

| # | wave · slot | task | cost | done when |
|---|---|---|---|---|
| **A** | 1 · node 1 | **§9 rung 1, `probe_backprop_ceiling.py --dataset yahoo`.** The 2026-08-16 attempt tokenized all ten shards and **its stdout was never captured** — `tee` it | 1 GPU, **~10 min**, all 101 shards warm | a number. **≈0.70 ⇒ the data path is clean and yahoo is purely under-trained; ≈0.30 ⇒ the path is at fault and D/H are moot** |
| **B** | 1 · nodes 1, 2 | **A real-mode run per dataset, then `profile_sim_charges.py`.** It pools `vclock_charge` events with `time_mode == "real"`, so a sim run cannot feed it — the cost is the real run, not the script. The launcher plumbing is done; the artifact is the whole remaining step and `--force` lifts by itself once it exists | 1 GPU real run + minutes, per dataset | `sim_charge_profiles/fluxtune_{yahoo,yelp-p}.yaml` exist and the sim-profile preflight passes **without** `--force` |
| **C** | 1 · nodes 3, 4 | **Re-run P-4 agnews** | 2 × ~2.5 h | §6's four gates hold and the controller ends on `[BudgetStop]`. **This is the arm that decides whether agnews learns effectively in the new regime — nothing to date does** |
| **D** | 2 · nodes 1, 2 | **Re-run P-4 yahoo** at 80,000 vclock | 2 × ~6 h | same four gates; then score accuracy vs `Λ` against §5's first open row |
| **H** | 2 · nodes 3, 4 | **yelp-p bring-up** — first FL arm on the third dataset. Registry, partitions and cache are all done; what is untested is the *stack*, and 2 classes is the opposite corner from yahoo's 10 | 2 × ~4 h | same four gates; `A` and per-vclock-hour comparable against the other two |
| **E** | any time, CPU | 3.5's saturation stop — **not built.** Size window/threshold/patience by replay against the arms on disk before it ships | CPU | replay reproduces a sensible stop commit |
| **G** | **after** the P-4 arms, CPU | **`read_instance_from_h5` returns rows in thread-completion order**, so a shard's row order — and therefore its bin composition — is not reproducible across tokenizations, and `guid` names the wrong row (§10). No ledger number is affected. It waits because it re-orders every future shard against the caches those arms run on | CPU, minutes | two tokenizations of one client agree byte-for-byte, and `guid` round-trips |
| **F** | — | **done 2026-08-17** — shared absolute `cache_dir`, `pretokenize_dataset.py`, cold-cache preflight (§10) | — | 101/101 shards per dataset on `/coc/scratch` |

### How to run it, and what to expect

**One command per node. No waves, no barrier, no cross-node dependency.** Each node owns one dataset
end to end:

| node | chain | needs from other nodes |
|---|---|---|
| **1** | agnews controller → agnews control | nothing |
| **2** | real yahoo → `fluxtune_yahoo.yaml` → controller → control | nothing |
| **3** | real yelp-p → `fluxtune_yelp-p.yaml` → controller → control | nothing |
| **4** | backprop ceilings, yahoo then yelp-p (§9 rung 1, ~10 min each) | nothing |

**Why a pair stays on one node.** `/home/dgarg39/flame` is local disk per node; only `/coc/scratch` is
shared. Splitting a pair across nodes forces either a cross-node handoff of the sim charge profile or
two independently-profiled runs — and the profile is what converts vclock into real work, so a
controller and control charged from *different* profiles are no longer compared at equal vclock. This
project already rejected a shared profile over a 0.6–3.4% clock effect. One node per dataset makes the
pair identically priced **by construction**.

**Step 1 — smoke all four, ~20–30 min each.** Nothing below has run live: `_node_lib.sh` now backgrounds
the launcher under `set -m` (that changed how *every* arm launches), `watch_arm.py` has only been tested
against finished logs, **real mode has never run on yahoo or yelp-p**, and **yelp-p has never launched
the FL stack at all**. A clean `--dry-run` is a prior, not a guarantee — each of the four P4.7 defects
passed one and cost a node.

```bash
N=<1|2|3|4>
cd $REPO && git pull
tmux new -s smoke "SMOKE=1 $FW/expt_scripts/nodes/run_node.sh $N 2>&1 | tee ~/smoke_node$N.log"
```

**Smoke pass condition:** exit 0 · every arm ≥ 5 commits · no `arm_stall.json` · `[DataBins]` right per
dataset (150 / 1,750 / 650, `source=registry`, trainer-confirmed) · **0 zero-steps**. Gate 4 reads WARN
on a smoke controller — its budget is deliberately too small to reach `[BudgetStop]`, and that is the one
gate only the long run settles. **The smoke also prints the `rate` and `budget sizing` lines that size
the real run** (§11.1a-bis) — read them before step 2.

**Step 2 — the real run**, same command without `SMOKE=1`:

```bash
tmux new -s p4 "$FW/expt_scripts/nodes/run_node.sh $N 2>&1 | tee ~/p4_node$N.log"
```

Node 1 is ~5.5 h. Nodes 2 and 3 are the long poles and their length is **unknown until the smoke
measures it** — at the pre-eval-fix yahoo rate a pair would be ~23 h, and `--eval-max-samples 10000` is
aimed squarely at the term that caused it. Node 4 finishes in ~20 min and is then free for task E.

**Expected outcome, stated up front so it is not re-litigated later.** These runs give the first scored
arms under the fixed code on three datasets, closing **hole 1 of §-0's four**. They do **not** complete
the zero-input claim. Hole 2 (the `B_max` sensor is unexercised below chance on yahoo) closes only if
yahoo clears chance. Hole 3 (`rf`=64 cannot carry `annealed`, pinning `T_res`=300 and `f`=0.95 to
`p`=450,340) is **untouched** — every arm is pinned rf=16. Hole 4 (`f`=0.95 has no accuracy evidence) is
partly addressed by node 2's accuracy-vs-`Λ` curve. And §5 already pre-registers that the agnews-vs-yahoo
`ρ*` divergence **may not reproduce** — the two knees at commit 150 were 0.248 and 0.237. A clean run
showing no divergence is a finding about the *sensor*, not a failure.

**Two decisions already made, so nobody re-opens them:** `b_max_policy` defaults to **`mean`** (§5's
constants table has the argument), and databin **size stays 8 while the bin count moves per dataset** (§1).

---

## §-0 — The zero-input claim, as a checklist

**The claim.** FluxTune reaches a plateau and holds it on a new dataset **without anyone tuning a
learning knob** — because every constant on the operating path is either universal machinery, mechanically
derived from the data, or **sensed online**. B-1 is why the third category exists: `B_max` was measured
**erratic across task** (agnews knee ≈3.0–3.5, yahoo and yelp-p both ≈2.0–2.3, non-monotone in class
count), so it cannot be a shipped constant and 3.1 senses it instead.

**What a new dataset actually costs the operator.** Three columns, and only the first is *input*:

| supplied by hand | mechanically derived | universal constant |
|---|---|---|
| a `configs/datasets.yaml` row (h5 paths, `num_labels`, `max_seq_length`, split sizes) — a **description of the data**, not a tuning choice | `dataset` / `data_file_path` / `partition_file_path` / `max_seq_length` into both override blocks (`--dataset`) | `probe_combine=mean` · `server_step_rule=trust_ratio` · `commit_gate=n_target` · `gate_rho_ref=annealed` |
| a partition build (`build_niid_partitions.py`) + `check_partitions.py` | `num_labels` from the h5 label vocab | `s` = 1.5 · `T_res` = 300 · `f` = 0.95 |
| **a compute budget** (`max_runtime_s`, `sim_wall_ceiling_s`) | `p` from `[ProbeDim]`; `total_data_bins` from the registry (150 / 1,750 / 650) | `P` = 10 · `K`/`C` = 10/30 · `b_max_policy=mean` |
| `eval_max_samples` — **a cost knob, not a learning knob** (10,000 on yahoo) | `ρ*_t` from law C · `ρ_max` from gate reachability · `n_req` from its closed form | `B_max` **prior** `ln 2`, replaced outright by the first sense |
| a sim charge profile (**sim-only artifact**, needs a real run) | `B_max` itself — **sensed** by the 3.1 probe | — |

**So the answer to "what knobs do we give it": for learning, none.** What moves per dataset is a
description of the data, a compute budget, and one sim-harness artifact. That is the claim Phase 4 exists
to test, and **it has not been tested yet.**

**Four honest holes in the claim, all live:**

1. **No scored arm exists under the fixed code, on any dataset** — including agnews. The best evidence is
   `125619`, which won on time-to-accuracy (0.83 at **62%** of budget against the control's 92%, peak
   0.857 vs 0.835) while **23% of its commits took a step of length zero** and it ended on
   `max_runtime_s` instead of `[BudgetStop]`. It is a floor on the effect and void as acceptance. Task C.
2. **The sensor does not fire below chance.** On yahoo it declined at commit 25 (`base_acc=0.105` against
   chance 0.100), so the controller ran the `ln 2` prior. Universality of the *machinery* is intact; the
   *sensing* half is unexercised on the one dataset that most needs it (P4.8).
3. **The universal constants are not `p`-portable.** `rf`=64 cannot carry `annealed` at any `T_res` (§6),
   so `T_res`=300 and `f`=0.95 are pinned to `p`=450,340. That is a generality gap **within one dataset**,
   and it is a standing blocker on ship-checklist item 5b.
4. **`f` = 0.95 was sized against the pre-fix `B_max` semantics** and rests on a `Λ` ≥ 0.95 floor read off
   the agnews curve. `Λ` has never been tested across task. Re-derive both (§5's open table).

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

**Landed here, one line each — read the code, not a re-derivation.**

- **0.8 `--dataset NAME`** (2026-08-12): `run_sequential.sh` fans `dataset_registry.hyperparameter_overrides()`
  into **both** override blocks plus `trainer.dataset.name`; unset ⇒ byte-identical. The
  partition-group check verifies against the **resolved** dataset's own h5, and a sim leg is blocked
  against a profile taken on another dataset.
- **0.10 dataset-switch sweep** (2026-08-12): `diagnose_partition_binning.py` was the one real bug —
  its single-class-collapse detector compared accuracy to a hardcoded `0.25`, agnews' `1/K`, and would
  have silently missed collapses on yahoo and yelp-p. Everything else the sweep flagged is inert
  (registry lookups, the deliberately-fixed `_FD_REF_P` anchor, comments).
- **F, the shared tokenized cache** (2026-08-17): §10.
- **Sanity gate for all of the above: `expt_scripts/test_dataset_launcher.py`, CPU-only, ~60 s,
  23/23.**

**The one the sweep missed, and the lesson (P4.7 defect 4).** `total_data_bins = 150` was hardcoded in
`flame/mode/horizontal/syncfl/fwdllm_aggregator.py` — outside the paths 0.10 grepped, and not matched by
`agnews|450340|N_CLASSES = 4` even if it had been. It is agnews' `1,200/8` exactly, and the trainer
indexes its **own** batch list with `data_id`, so yahoo trained on the first 1,200 of each client's
14,000 rows — **8.6% of the dataset, the same 1,200 every lap.** Nothing raises; the list is merely
longer than the index. **Next dataset switch: grep `lib/python/flame/` too, and grep for the *derived*
agnews constants (150 / 1,200 / 7,600 / 192), not just the dataset's name.**

**Still not done:** `profile_sim_charges.py` has never run for yahoo or yelp-p, so every non-agnews sim
arm mis-prices its vclock and needs `--force`. It profiles from a **real-mode** run's `vclock_charge`
telemetry, so the blocker is a real run per dataset (§-1 B). The plumbing around it landed 2026-08-17 —
the launcher resolves `fluxtune_<dataset>.yaml` when it exists and gates on the profile's own
`_meta.datasets` tag — so only the artifact is missing.

**The onboarding sequence for the next dataset, in order:** a `configs/datasets.yaml` row →
`build_niid_partitions.py` → `check_partitions.py` → **`pretokenize_dataset.py`** (§10 F2) → a real-mode
run → `profile_sim_charges.py` → the arm.

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
defect 3) — read trips/commit **per quintile at run time**, which is what `check_arm_health.py` now
does; the projection itself is still priced off the prior. And 0.10's sweep did not cover
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

**And on yahoo the sensor may not fire at all early.** `234931`'s first probe returned
`[BmaxProbe] commit=25 base_acc=0.105 too close to chance 0.100; keeping B_max=0.693147` — the guard in
`FedSgdAggregator.py:978` refuses to read a knee off a model sitting at chance, so early yahoo commits run
on the `ln 2` **prior**, not on a sensed budget. Law C is unharmed (the prior is exactly what it is for),
but **the pre-registered agnews-vs-yahoo `ρ*` divergence is not measurable until yahoo clears chance** —
which is the same under-training question §9 exists to answer. Read the *first firing* commit off
`[BmaxProbe]` on the re-run; that commit index is itself the result.


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

**Re-launch gate — check all four in the first 200 commits, before letting an arm run unattended.**
`expt_scripts/check_arm_health.py <run_dir> --expect-controller` is these four, and exits 1 on a breach:

```
[DataBins] total_data_bins=1750 source=registry dataset=yahoo   # not 150
no server_update record with rho_star == 0                      # defect 2
controller ends on [BudgetStop] reason=budget, not max_runtime_s # defect 1
trips/commit >= 3 at every quintile                             # defect 3 + T5's gate
```

**Three of the four already hold on yahoo under the fixed code** (`234931`, 31 commits before its wall
budget expired): `[DataBins] total_data_bins=1750 source=registry`, 100% coverage, **trainer-confirmed at
1750**; 0/31 commits with `rho_star` = 0; `n_req` = 93.4 at ~10 trips/commit. Only the `[BudgetStop]`
ending is untested on yahoo — that one needs a full-length arm.

**Three more that are not defects but invalidate the scoring — all now enacted by `run_node_p4.sh`, so
they need no operator action; kept because they are the *reasons* its constants are what they are:**

- **A per-dataset sim charge profile.** yahoo burns **0.658 real-s per vclock-s against agnews' 0.255**,
  so no cross-dataset per-vclock comparison is valid until each is profiled. Yahoo-vs-yahoo is fine.
- **A vclock budget far above 40,000 on yahoo** (now 80,000). Its control was still climbing
  monotonically at 0.296 when killed at 86% of 40,000, against a 0.73 backprop reference (§9).
- **`--eval-max-samples 10000` on both seq-256 datasets.** The eval *is* backgrounded, but
  `_eval_snapshot_model` blocks the main thread when the previous eval is still running — and it was, on
  **359 of 359** evals on the yahoo control (agnews: 287/488). 60,000 rows at seq 256 costs **89.2 s
  measured under trainer contention** against a ~91 s inter-eval gap. Batch size is not the lever (1.23×
  from batch 8 → 128 — the pass is compute-bound). The subsample is **fixed and shuffled** (seed
  20260810), so its sampling error is a constant offset rather than per-eval noise and peak-vs-final
  stays as precise as the full set, which is what P4.4's 0.015 bar compares.

**4.1** agnews · **4.2** yahoo · **4.3** yelp-p — *same binary, same flags*, no operator input beyond
model / PEFT / `p`. The only per-dataset arguments are a compute budget and a sim-harness artifact (§-0).

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

**Rung 1 — the discriminating test · `expt_scripts/probe_backprop_ceiling.py` · 1 GPU, ~10 min.
Run this first if only one rung fits.** Centralized AdamW on the **FL rig's own data path** — client
shards through `TextClassificationDataManager`, capped at `total_data_bins` so it sees exactly the rows
an FL arm can reach, evaluated on the **same** `test_global` `agg_eval` uses. It instantiates
`ForwardTextClassificationTrainer` purely to drop `pre_classifier`, which is what makes `p` **450,340
rather than 1,040,932**. The one change from an arm is an exact gradient instead of pooled forward
differences. (Not `probe_inflation_damage.py`'s rig, which reads accuracy off a Dirichlet-skewed slice.)

**Pre-register: ≈0.70 clears the entire data path** — tokenization, label vocab, `max_seq_length`, the
h5 ranges, the partition, the loader — **and makes the gap purely optimization; ≈0.30 indicts the path**
and every rung below is moot. **Calibrated on agnews (2026-08-16): 3,600 rows, one epoch → 0.850**
against that dataset's FL control at 0.835 after 801 commits and 2.3 h. Command: [§11.3](#113-individual-invocations-if-a-wave-has-to-be-taken-apart).
**`tee` it** — the script only prints, and the 2026-08-16 number was lost to a closed terminal.

> **No attention mask anywhere in the stack** (seen 2026-08-17 on node 4's own log:
> *"We strongly recommend passing in an `attention_mask` since your input_ids may be padded"*).
> `tc_transformer_trainer_distribute.py:713` and `:950` both do `x = batch[1]` then `self.model(x)`,
> dropping `batch[2]`, and `probe_backprop_ceiling.py` does the same — so the probe is **faithful to
> production** and rung 1 stays a valid test. But the model attends to PAD tokens on every arm, which
> depresses absolute accuracy everywhere and plausibly hurts yahoo most: its length *variance* is far
> higher (p50 84 / p95 367 truncated at 256, against agnews' 41 / 70 at 192), so how much of each
> sequence is real content swings much more. **If rung 1 comes back ≈0.30, this is the first thing to
> test** — it is the cheapest candidate for a data-path fault and it has never been examined.

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

---

## §10 — The tokenized-feature cache *(task F, landed 2026-08-17)*

**One shared cache, `/coc/scratch/dgarg/fl_datasets/fwdllm/cache_dir`, 101/101 shards for all three
datasets.** Absolute, so the launch directory no longer decides which cache a process gets, and on
`/coc/scratch` so every node and every probe reads the same bytes.

### How it is addressed — the part that matters when you add a dataset or an α

A cache file is one client's tokenized shard, keyed by everything that changes its tensors:

```
{model_type}_{model_name}_cached_{max_seq_length}_{model_class}_{dataset}_{partition_method}_{client_id}
    distilbert_distilbert-base-uncased_cached_256_ClassificationModel_yahoo_niid_label_clients=100_alpha=1_37
```

- **`partition_method` carries both `C` and α**, so **α=1 and α=100 are different files** and a
  `--partition-method` switch is a MISS, never a stale hit. Tokenized today: agnews `alpha=1` + `uniform`
  (101 each) and `alpha=0.1` (90, partial); yahoo and yelp-p `alpha=1` (101 each). **An α ablation must
  be pre-tokenized first** or it pays the ~30 min inside its own wall budget.
- **`client_id` is the trainer's `client_idx`, not its trainer id.** `runner.py:389` sets
  `client_idx = (trainer_id - 1) % client_idx_modulo` (100 in every fwdllm yaml), so trainer 1 → shard 0
  … trainer 100 → shard 99, and 200 trainers would wrap onto the same 100 shards.
- `client_id = -1` is the aggregator's global test set, which `agg_eval` needs.

**So yes: at startup a trainer resolves `(dataset, seq, partition_method, client_idx)` to one file and
`pickle.load`s it** — no h5 read, no tokenization, no `alpha` lookup beyond the group name already in
its config.

### Using it

```bash
$PY expt_scripts/pretokenize_dataset.py --dataset NAME --clients 100 --dry-run   # missing + size
$PY expt_scripts/pretokenize_dataset.py --dataset NAME --clients 100 --jobs 16   # ~9 s/client @ seq 256
$PY expt_scripts/pretokenize_dataset.py --dataset agnews --partition-method niid_label_clients=100_alpha=100 ...
```

Idempotent, drives the production loader so the bytes are the trainer's own, writes the `-1` global.
Every `--dry-run` now carries a `feature cache warm (<baseline>)` row — `ok` at 101/101, or a `warn`
naming this command. Override the location with `FWDLLM_CACHE_ROOT`.

### The three findings this task produced, none of which needed a run

1. **F0 answered off the code.** `_load_federated_data_local` ran its load-or-tokenize block **twice**
   (`base_data_manager.py:324` and `:438`) and the *second* result reached the loader — which is exactly
   why one run's log showed 100 cache hits, 100 tokenization bars and 101 cache writes in the same
   window. Duplicate removed; a warm cache does skip tokenization entirely.
2. **The cache does not change a number.** Cold vs warm tensors are byte-identical within a process, and
   the shard `(input_ids, label)` multiset matches the pre-dedup cache across three independent draws
   (`7ee980d0fccc63d2`).
3. **But a shard's row ORDER is not reproducible across tokenizations** — queue row **G**.
   `read_instance_from_h5` fans the h5 reads over a 20-thread pool appending to shared `X`/`y` **in
   completion order** (and the whole body is inside the lock, so the threads buy no parallelism), while
   `transform_examples` pairs `X[i]`/`y[i]` with `index_list[i]` as the **guid**. `X`/`y` stay paired
   under one lock and nothing reads `guid`, so **no ledger number is wrong** — but which 8 rows form a
   given `data_id` bin changes if you re-tokenize. That makes the frozen shared cache the thing that
   makes an A/B byte-comparable, and it is why row G waits until after the P-4 arms.

**Sanity gate, standing:** a `--num-trainers 100` yahoo arm reaches commit 1 in minutes, `[DataBins]
confirmed by trainer … 1750 batches` holds, and the first commit's `ρ` still reads `234931`'s **0.0678**.

---

## §11 — Runbook · one command per node, no cross-node dependency

**Every slot below is longer than a login survives — run each inside `tmux`.** The 2026-08-16 backprop
ceiling was lost to a closed terminal; that is the whole reason this section says so twice.

### §11.0 Environment — required by everything

```bash
export FLAME_CONDA_ENV=test_fwdllm        # base lacks h5py; every preflight exits 2 without it
export FWDLLM_FD_SCALE_INVARIANT=1        # the FD-rescale preflight refuses without it
PY=/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python
REPO=/home/dgarg39/flame
FW=$REPO/lib/python/examples/fwdllm
```

**`/home/dgarg39/flame` is LOCAL disk on each node; `/coc/scratch` is the shared one.** So every node
needs its own `git pull` for code.

**Launch directory no longer matters** (fixed 2026-08-17). Two things used to resolve against the
launching shell's cwd, and only one was obvious. `cache_dir` was the known one (§10). The other was
`sim_charge_profile_path`, emitted repo-root-relative and opened by the aggregator with a bare `open()`
against **its own** cwd — `spawner.py` starts every process with no `cwd=`, so it inherits the shell's.
A miss there is a `[SIM_CHARGE_PROFILE] failed to load` **warning** and an empty dict, so the vclock
silently loses every profiled charge and nothing fails. It is now emitted absolute. **Nothing else crosses nodes** — each node produces the sim charge
profile it consumes (§-1), and the tokenizer cache is already on `/coc/scratch` at 101/101 for all three
datasets (§10).

### §11.1 One command per node

```bash
N=<1|2|3|4>
cd $REPO && git pull
tmux new -s p4 "$FW/expt_scripts/nodes/run_node.sh $N 2>&1 | tee ~/p4_node$N.log"
```

Node 1 = the agnews pair · node 2 = real yahoo → profile → pair · node 3 = the same for yelp-p ·
node 4 = the two backprop ceilings, then free. The per-node table and the reasoning for keeping a pair
together are in [§-1](#-1--status-board).

**Why the two arms of a pair cost so differently.** The **controller stops itself** at `B ≥ f·B_max`, and
law C's length comes from `(B_max, T_res, f)` — not from the budget, so its ~898 commits cost what they
cost. That is the *whole point*: the arm ends on `[BudgetStop]`. The **control has no stop**
(`--phi-stop log_only`, deliberately, so P4.1's past-the-stop counterfactual keeps being measured), so it
runs its vclock budget out.

### §11.1a Smoke every node first — `SMOKE=1`, ~20-30 min

**Do this before the real run.** `SMOKE=1` runs the *identical* chain at a small vclock budget (agnews
2,500 / seq-256 1,500) against a 2.0 h ceiling — the ceiling must stay **above** law C's own ~1.76 h
projection or the preflight refuses the arm outright, so a small ceiling makes a blocked run, not a
short one.

```bash
SMOKE=1 $FW/expt_scripts/nodes/run_node.sh $N
```

Smoke profiles come from a 10-minute real run and are **not** fit to price an arm, so they are written
under `sim_charge_profiles/smoke/` and never overwrite the production ones. The watchdog tightens to a
7-minute stall window and a 25-commit grace so it is armed inside the horizon.

**What a smoke covers:** the `set -m` backgrounding and watcher teardown (new — it changed how every arm
launches) · the watchdog attaching to a live run without false-firing · real mode on yahoo and yelp-p
(**never run before**) · yelp-p's first-ever FL arm · `check_arm_health` after each arm · and the `rate`
line that sizes the real run.

**What it cannot cover:** a smoke controller ends on `max_runtime_s`, not `[BudgetStop]` — its budget is
deliberately too small — so gate 4 reads WARN and only the long run settles it. The smoke also uses
`--force`, since its profile is in `smoke/`; the force-drop path was verified separately against a
tagged stub (`--force` count 0, profile row `ok`, zero non-ok checks).

### §11.1a-ter The profile-staleness check is node-dependent — `--allow-stale-profile`

The `sim charge profile is CURRENT` preflight globs the **local** `experiments/run_*_<baseline>_n*_real`
directory to find reals newer than the profile's own source runs. That directory is **machine-local
disk**, so the same profile, config and code pass on a node with no old reals and **block** on one that
has them — `kaylee` blocked on two 2026-08-04 agnews reals where `jayne` (which has none) passed.
Profile validity is a property of the profile and the config, not of which box holds which run dirs.

**`run_node_p4.sh` passes `--allow-stale-profile`, which downgrades that one check to a warn and leaves
the other nine armed.** Not `--force`, which would also disable `sim charge profile matches dataset` —
the config-derived check that actually protects the vclock. And **not** a re-profile: `fluxtune.yaml` is
what every historical agnews arm and P4's own calibration were priced against, so re-profiling now would
re-price the comparison these arms exist to make.

### §11.1a-quater What the first smoke found (2026-08-17)

Three defects, none visible to any dry-run, all in code added the same day:

1. **The watchdog killed a legitimate real-mode arm at commit 0.** The stall check was not gated on
   having *reached* commit 1, so the 7-minute smoke window applied to spin-up — and a real yelp-p arm at
   seq 256 with 100 trainers takes far longer than that to its first commit. Fixed: a separate
   `--first-commit-grace-s` (default 45 min) applies until commit 1 lands.
2. **A killed run was still profiled.** `profile_dataset` wrote `fluxtune_yelp-p.yaml` from **n=7**
   `vclock_charge` samples against a healthy **n=3601** — and a bad profile is worse than none, because
   it then *looks present* and skips the real run on every retry. Fixed: refuse to profile when
   `arm_stall.json` exists or when fewer than `MIN_CHARGE_SAMPLES` (200) samples were collected.
3. **A from-scratch per-dataset profile charges NOTHING.** `profile_sim_charges.py` writes a brand-new
   entry with `charge: false` ("review before enabling"), so `fluxtune_yelp-p.yaml` came out inert —
   silently *un*-pricing the very arms task B exists to price, which is the opposite of the intent.
   Fixed: seed the new file from `fluxtune.yaml` first so the `charge:` flags and rationales carry over,
   then assert at least one entry is charging. Verified: `NONE -- inert` becomes
   `drain_tail._default, fedavg._default` with the dataset tag intact.

### §11.1a-bis The rate constant, and why the smoke sizes the real run

**Measured, 2026-08-16 arms** (first commit to last, so the pre-commit-1 tokenization stall is excluded
by construction):

| arm | commits/h | vclock/h | 898 commits would need |
|---|---|---|---|
| agnews `130614` | 351 | 17,523 | 44,839 vclock · 2.6 h |
| agnews `125619` | 345 | 14,138 | — |
| **yahoo `125713`** | **79** | **5,470** | **61,932 vclock · 11.3 h** |

**The wall-clock preflight cannot see this.** It prices every commit at a dataset-independent 4.41 s;
yahoo measured **45.4 s**, so the projection under-books the seq-256 datasets ~10×. Budgets are now sized
off measured rate instead, and `check_arm_health.py` prints a `budget sizing` line that converts any
short arm's rate into the vclock 898 commits will cost.

**Yahoo's 4.4× is not the tokenization stall** — checked: the rate is flat across all four commit
quartiles (84/71/91/73 against agnews' 347/354/348/352), and a startup stall would show as a slow first
quartile. It is sustained, and two things make it so: seq 256 costs ~1.33× per pass, and
`_eval_snapshot_model` **blocked the commit loop on 359 of 359 yahoo evals** (agnews 287/488) at 89.2 s
against a ~91 s inter-eval gap.

> **Pre-registered:** `--eval-max-samples 10000` attacks that second term directly — 10,000 rows instead
> of 60,000 — and **has never been measured on a full arm.** If the eval tax is the dominant term, the
> smoke's yahoo rate should land well above 79 commits/h, and node 2's budget comes down with it.
> If it stays near 79, the cost is intrinsic to seq 256 and the yahoo arms are ~11 h each. **Read the
> smoke's `rate` and `budget sizing` lines before launching the real run — that is what they are for.**

### §11.1b Can this be shorter?

**The controller arms cannot, and that is not a scheduling problem — it is the measurement.** Their
length *is* "how long law C takes to reach 0.95·`B_max`", and `T_res`=300 / `f`=0.95 are the constants
under test (§5). Shorten them by shrinking the budget and the arm dies on `max_runtime_s`, which voids
it by the project's own rule — that is exactly how all four 2026-08-16 arms were lost. At ~2 h each they
are already the cheap half.

**The control arms can, and the yahoo one should be re-cut.** Its 60,000 vclock is a judgement call: its
predecessor was still climbing at 86% of 40,000 (P4.8), so the budget went up 1.5×, and at yahoo's
**mis-priced** 0.658 real-s/vclock-s that is ~10 h. Two things follow. (1) 80,000 would have breached the
ceiling outright, which is why it is 60,000. (2) That 0.658 comes from charging yahoo against an
**agnews** profile — the exact error task B fixes — so **once `fluxtune_yahoo.yaml` exists the number is
worth re-reading, and the budget re-cut against it.** The `--sim-wall-ceiling-h 12` backstop is set so the
ceiling does not clip before the budget does.

**Cheaper still, if a slot is scarce:** the controller arms alone (nodes 2 and 3) answer the
acceptance question — the controls are the comparison, and agnews already has one on disk from
2026-08-16 to compare against.

### §11.1c Stalling and early termination

`_node_lib.sh` runs `expt_scripts/watch_arm.py` as a side-car for every arm and **kills the run** on:

| predicate | default | why this and not accuracy |
|---|---|---|
| no new commit | 20 min | a genuine hang; the only unambiguous one |
| trips/commit over the last 200 commits < 3 | after 200 commits | `N_req ∝ ρ_t²` under `annealed`, so an annealing `ρ` demands less pooling every commit until `I` floors at 1. **This is how G-2's `003648` died**, and the launch projection cannot see it because it prices law C off the `ln 2` prior (P4.7 defect 3) |
| any `rho_star == 0` | after 200 commits | a requirement of *zero*, not an absent one (defect 2) |

**Deliberately NOT the harness's own stall guard.** `converge_watch.py` arms on held-out accuracy and
requires `--target-acc`, which would end the arm on convergence — and an arm that does not end on
`[BudgetStop]` is void. Worse, its signal is backwards here: **reaching a plateau and holding it is what
the controller is supposed to do**, so an accuracy stall guard would kill the success case.

`NODE_WATCH=0` disables it; `NODE_WATCH_ARGS="--stall-window-s 600"` retunes it. A killed arm leaves
`arm_stall.json` in its run dir and the node prints it.

### §11.2 Reading an arm — the only command needed

```bash
RUN=$(ls -dt $FW/experiments/run_* | head -1)
$PY $FW/expt_scripts/check_arm_health.py $RUN --expect-controller   # exit 1 = a gate is breached
$PY $FW/expt_scripts/replay_scoring.py $RUN                         # B, Lambda, Phi, A
```

`check_arm_health.py` is §6's four gates as one command — `[DataBins]` (value vs the registry, `source=`,
coverage, trainer confirmation) · `rho_star == 0` count · **trips/commit per quintile** against the ≥3
floor · the `[BudgetStop]` reason — plus the `[BmaxProbe]` trajectory including its **first firing
commit**, which on yahoo is itself a result. `_node_lib.sh` already runs it after every arm, so a chained
node reports its own gates; run it by hand to check a live arm at ~200 commits. **A controller arm that
ends on `max_runtime_s` instead of `[BudgetStop] reason=budget` is void** — that rule voided all four
2026-08-16 arms.

Scoring, from [P4.4](fl_fwd_ft_practice.md#p44-scoring-rules-for-any-ab): score **peak**, never final of
a diverging arm. Compare across datasets on **`A` and per-vclock-hour**, never on `Λ` (different `p`)
and never per round (11.7× different bins/round).

### §11.3 Individual invocations, if a wave has to be taken apart

```bash
# task A alone
cd $FW && $PY expt_scripts/probe_backprop_ceiling.py \
  --config $(ls -1dt $FW/experiments/run_*yahoo*/aggregator_config.json | head -1) \
  --dataset yahoo --clients 10 --epochs 3 2>&1 | tee $FW/experiments/_probe_logs/bpc_yahoo.log

# task B / a yelp-p profile alone: real mode is the ONLY source -- profile_sim_charges.py
# pools vclock_charge events with time_mode == "real" and finds nothing in a sim run
cd $REPO && $FW/expt_scripts/run_sequential.sh --only fluxtune --mode real --dataset yahoo \
  --yes --clean --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30 \
  --adapter-reduction-factor 16 --max-runtime-s 3000
cd $FW/expt_scripts && $PY profile_sim_charges.py \
  --real-run $(ls -1dt $FW/experiments/run_*yahoo*real* | head -1) \
  --out ../sim_charge_profiles/fluxtune_yahoo.yaml --only-observed

# any P-4 pair
$FW/expt_scripts/nodes/run_node_p4.sh <agnews|yahoo|yelp-p> <controller|control>
```

`run_node_p4.sh` pins everything: `rf`=16, cos audit **off**, `--num-trainers 100 --c 30 --agg-goal 10`,
per-dataset vclock (agnews 40,000 · yahoo 80,000 · yelp-p 60,000) and `--eval-max-samples 10000` on both
seq-256 datasets; controller = law C at `T_res`=300 with **no `--rho-star` and no `--b-max`** (that is
what makes it zero-input), control = `rm`/0.25 at `ρ*`=0.06 with `gate_rho_ref=setpoint` and
`--phi-stop log_only`. **`NODE_DRY_RUN=1` runs every arm's preflight in seconds without burning a node**
— do that after any `git pull` and confirm the only `✗` is one you understand (P9.2).

### §11.4 Task E — the saturation stop *(§5's 3.5; CPU only, no node needed)*

No launch. Size the three unsized constants — smoothing window, `GL` threshold, patience — by replay
against the arms already on disk (P4's portfolio plus 2026-08-13's K-1/P-1/G-2 arms), the same way task
0.5 sized `Φ`'s window and P4.1 sized its threshold. `replay_phi_stop.py` is the model to copy.

### §11.5 Task G — the shard row-order fix *(§10; CPU, minutes, AFTER the P-4 arms)*

Preserve `index_list` order in `read_instance_from_h5` so `guid` names its own row and two tokenizations
of one client agree byte-for-byte. It re-orders every future shard against the caches the P-4 arms
run on, which is the only reason it is not done already.
