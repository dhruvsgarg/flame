# Build plan — **the status doc**: where things stand, what is next, and how to build it

> **This is the one file to read for status and next steps, and the one to update in place.**
> [§-1](#-1--status-board) is the status board. [§-0](#-0--the-zero-input-claim-as-a-checklist) is the
> generality claim as a checklist. Everything from §0 down is one spec per queued task, and
> [§11](#11--runbook--one-command-per-node-no-cross-node-dependency) is the runbook.
>
> **The other two docs are evidence, not status.**
> [fl_fwd_ft_practice.md](fl_fwd_ft_practice.md) owns *what is true*: the P3 knob ledger, the P4 arm
> ledger, P6's dead ends, P8's reproduction recipes. Every number cited here lives there.
> [fl_fwd_ft_solution.md](fl_fwd_ft_solution.md) owns *why* — the model. Cited as "model §x".
> **Read [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) before proposing any change.**

## Standing facts — read before re-deriving any of these

*These are the ones that get measured twice because the first answer was in someone's terminal.
**Probe logs and run dirs are node-local disk** (`/home/dgarg39/flame`); only `/coc/scratch` is shared,
so "I did not find it on this node" is never evidence that it was not run. This table is the record.*

| | agnews | yahoo | yelp-p |
|---|---|---|---|
| **backprop ceiling** (§9 rung 1) | **0.850** | **0.734** | **0.874** — all 2026-08-17, all clear the ≈0.70 bar |
| production sim charge profile | `fluxtune.yaml` (agnews-priced) | **missing** | **missing** |
| tokenizer cache @ `alpha=1` (§10) | 101/101 @ 192 | 101/101 @ 256 | 101/101 @ 256 |
| registry row · partitions · `check_partitions.py` | done · 6/6 exact | done · 6/6 exact | done · 6/6 exact |
| bins/round at `C`=100 | 150 | 1,750 | 650 |
| arms ever run on the FL stack | many | 6 | 3 |
| **scored pair under the fixed code** | control **valid**; controller live | controller **void** x2, both watchdog | **controller VALID** — first `[BudgetStop]` ever; control live |

**Known-broken, currently unfixed:** the watchdog's `I`-floor kill voids healthy landing controllers
(queue row **W**); launch with `--i-floor-frac 1.01` until it lands. **Recently fixed, do not
re-diagnose:** see the dated rows in [§-1](#-1--status-board).

## How to update this doc — UPDATE IN PLACE, never append

**The failure this policy exists to stop:** a fact is established, the session ends, the next session
re-derives it or contradicts it. An append-only log makes that worse, not better — it buries the current
answer under its own history.

1. **One fact, one home.** Every fact has exactly one cell or line in this doc. Change *that* cell.
   Never add a second statement of the same fact anywhere, including a "latest findings" section.
2. **Replace, do not accumulate.** A measurement carries its value and the date it was taken — the old
   value is deleted, not struck through. If the old value matters, it is a finding and belongs in P4.
3. **No changelog, no session log, no dated append sections in this file.** Chronology lives in git and
   in P4. This doc answers "what is true now and what is next", nothing else.
4. **When a task lands:** its number goes to P3/P4, its row moves to done, **its spec is deleted** (§0
   rule 6). A landed section shrinks to its result and its command.
5. **When a defect is found and fixed:** update the row it invalidates and the runbook line it changes.
   The postmortem goes to P4. Do not leave the wrong number visible next to the right one.
6. **Budget: Standing facts through the end of [§-1](#-1--status-board) stays under ~120 lines** — two
   screens. If an edit pushes past that, something in it has stopped being status: push the detail down
   to its spec section or out to P4 and leave a link, which is rule 1 applied to prose.
7. **End of every working session, before anything else:** reconcile this doc against what you actually
   ran. Land the numbers, delete what they replace, re-cut the queue. Cheaper than re-running.

---

## §-1 — Status board

*State read **2026-08-20 16:10**. **The first valid controller arm exists**: node 3's yelp-p ended on
`[BudgetStop] reason=budget action=halt` at commit 1,348, `B`=0.9721/1.0231 (95.0%), every bin visited,
no zero steps. It is also the arm the `I`-floor kill would have destroyed at ~900 commits.*

**The blocker — the watchdog's `I`-floor kill voids healthy landing controllers.** Same defect class as
the `trips/commit` floor it replaced: `n_req ~ ρ²`, so a landing `ρ` drives `I` to 1 **by design**. It
killed node 2's yahoo controller at commit 964 / **86.0% of `B_max`** with pool demand met on *every*
commit, and node 3 reproduced the climb live (`I==1` 12% → 44% in 9 min while `B` rose). Fix is row
**W**, landed. **Live arms still hold the old module in memory** — a running watcher is not fixed by
the patch, so an armed arm launched before it needs `kill $(pgrep -f watch_arm.py)` or a relaunch.

| node | arm | state |
|---|---|---|
| 1 | agnews **controller** | live 15:22, `condition_fp c2ef1528` matches the valid control. **Watchdog armed — exposed to W** |
| 2 | yahoo **control** | live 15:16, ends ~22:15. `setpoint`, trips/commit ~7.9, so **immune to W**. `condition_fp 7174b984` |
| 2 | yahoo **controller** | **void** — killed 15:15 by the `I` floor. Never budget-starved: ~20,600 of 60,000 vclock used. Re-run is row **D** |
| 3 | yelp-p **controller** | **VALID** — `[BudgetStop]` at commit 1,348, 95.0% of `B_max`, 8 `BmaxProbe` fires (0.693 → **1.023**). Gate 3 FAILs at Q5=1.34 with `I==1` on 100% and demand met on every commit: annealing on plan, read §11.7 |
| 3 | yelp-p **control** | **not started as of 16:12** — 16:08 was the controller's `[BudgetStop]`, and its post-arm gate reader is still running. `run_node.sh 3` starts the control behind it (row **H**), on the **old** watcher: the launcher was up before W landed |
| 4 | — | **idle and usable** — take row **D** there rather than queue it behind node 2's control |

**The one number to take from 2026-08-20's night arms:** the controller reached **80.7% of `B_max` on
29% of its vclock** against the control's **15.5% on 100%**. That is the effect, visible in the first
pair — and the arm showing it is void on a technicality, which is why C is re-cut below rather than scored.

**Where the program is.** Phases 0–2 landed. The Phase-3 controller is built and correct as of
2026-08-16 — the law beat its control on both datasets, so the controller was never in question, its
plumbing was ([P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)).
**Phase 4 is the whole remaining question and still has no scored arm under the fixed code.**

### Ordered queue

**W first, then length.** Every controller row below is voided in its endgame by the bug W fixes, so
W is not optional caution — it is the difference between an arm and a void arm. The S1-S3 sanity rungs
passed 2026-08-20 and are retired.

| # | node | task | cost | done when |
|---|---|---|---|---|
| **W** | — | **LANDED.** `I`-floor kill now needs a progress conjunct: fire only when `I` is floored **and** `B/B_max` gained ≤ `--b-advance-min` (0.005) across the window ([§11.6](#116-stalling-and-early-termination)). Replayed silent on node 3's yelp-p (**`I==1` on 100%** of its last 200 commits, `ΔB`=0.0339, would have been killed by the old rule) and on the agnews smoke. **Still open:** the G-2 side is unverified — `003648`'s run dir is node-local and not on jayne | CPU, minutes | replay `003648` and confirm it still fires |
| **C** | 1 | **P-4 agnews — controller only.** Live from 15:22 at `condition_fp c2ef1528`; the 2026-08-20 control is valid on disk, do not re-run it | ~2.3 h | §6's gates hold **and** it ends on `[BudgetStop] reason=budget` — the rule that voids it otherwise |
| **D** | **4** | **P-4 yahoo controller re-run** at 60,000 vclock. Node 2 is busy with the valid half of the pair, so run this on the idle node — copy node 2's `sim_charge_profiles/fluxtune_yahoo.yaml` across **first**, since one profile on both arms is the only reason `run_node.sh` keeps a pair on one node | ~3.2 h | ends on `[BudgetStop]`; pairs with node 2's control at `condition_fp 7174b984`; then score accuracy vs `Λ` against §5's first open row |
| **H** | 3 | **P-4 yelp-p control** at **50,000** vclock — auto-starts behind the halted controller; not up as of 16:12. 2 classes is the opposite corner from yahoo's 10 | ~4.9 h | same gates; `A` and per-vclock-hour comparable against the other two |
| **E** | 4, or any CPU | 3.5's saturation stop — **not built.** Size window/threshold/patience by replay against the arms on disk before it ships | CPU | replay reproduces a sensible stop commit |
| **B4** | any CPU | **`budget_stop_frac` needs a margin against a moving `B_max`.** `mean` is damping well — raw senses 0.5272 / 0.8216 / 0.7428 (1.56× spread) combine to 0.5272 → 0.6744 → 0.6972, i.e. **+3.4% by n=3** — and re-sensing is *correct*, `base_acc` rose 0.805 → 0.846 across the same probes. The exposure is the **first** sense: it replaces the `ln 2` prior outright at n=1, maximum variance, and it came in **below** the prior (0.5272 vs 0.6931). Had `B` been past 0.95·0.5272 at commit 150 the arm would have stopped on the spot. Candidate: do not arm the stop until n ≥ 2 senses. **yelp-p 2026-08-20 confirms the exposure**: n=1 sensed 0.5223, then 0.7440 / 0.8961 / 0.9425 / 0.9598 by n=5 — the first sense was 46% low | CPU, replay | a rule with a stated margin, replayed against `021735` and 2026-08-16's arms; `ratchet` is **not** it — it is `min`, which stops even sooner |
| **G** | **after** the P-4 arms, CPU | **`read_instance_from_h5` returns rows in thread-completion order**, so a shard's row order — and its bin composition — is not reproducible across tokenizations, and `guid` names the wrong row (§10). No ledger number is affected. It waits because it re-orders every future shard against the caches those arms run on | CPU, minutes | two tokenizations of one client agree byte-for-byte, and `guid` round-trips |

**The profile is still the one hard gate on D and H:** each is scored against its own dataset's `vclock`
pricing, so `sim_charge_profiles/fluxtune_{yahoo,yelp-p}.yaml` must exist and its preflight pass
**without** `--force` ([§11.8](#118-individual-invocations-if-a-wave-has-to-be-taken-apart)).

### How to run it

**One command per node, no waves, no barrier, no cross-node dependency.** A pair stays on one node so it
is priced by one profile — `run_node.sh`'s header has the argument.

```bash
N=<1|2|3|4>
cd $REPO && git pull
NODE_DRY_RUN=1 $FW/expt_scripts/nodes/run_node.sh $N     # S3, seconds
tmux new -s p4 "$FW/expt_scripts/nodes/run_node.sh $N 2>&1 | tee ~/p4_node$N.log"
```

**Every slot is longer than a login survives — always `tmux`.** Commands for the individual pieces
(ceiling probe, profile alone, one arm of a pair) are in [§11.8](#118-individual-invocations-if-a-wave-has-to-be-taken-apart);
smoking a node is [§11.2](#112-smoke-a-node-first--smoke1-2030-min); what the watchdog kills on and how
to retune it is [§11.6](#116-stalling-and-early-termination).

**Expected outcome, so it is not re-litigated.** These close **hole 1 of §-0's four** and no more. Hole 2
closes only if yahoo clears chance; hole 3 (`rf`=64) is untouched, every arm pinned rf=16; hole 4 is
partly addressed by node 2's accuracy-vs-`Λ` curve and now also depends on **B4**. §5 pre-registers that
the agnews-vs-yahoo `ρ*` divergence **may not reproduce** (knees 0.248 vs 0.237) — no divergence is a
finding about the *sensor*, not a failure. **Settled, do not re-open:** `b_max_policy` = `mean`, and
databin size stays 8 while the bin count moves per dataset (§1).

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

### Scoreboard — what is still missing before the sentence is earned

> *FluxTune's law-C controller reaches and holds a plateau on a new dataset with no learning knob tuned
> by hand — same DistilBERT + adapters, three datasets, controller vs control at equal vclock.*

A pair counts only if **both** arms are valid at the **same** `condition_fp` and the controller ended on
`[BudgetStop] reason=budget` ([§11.7](#117-reading-an-arm--the-only-command-needed-and-the-scoring-rules)).

| what the sentence needs | agnews | yahoo | yelp-p |
|---|---|---|---|
| backprop ceiling clears ≈0.70 (§9) | **yes** 0.850 | **yes** 0.734 | **yes** 0.874 |
| its own sim charge profile | `fluxtune.yaml` | built, **node 2 local** | built, node 3 local |
| **control** arm valid | **yes**, 938 commits | live, ends ~22:15 | live, ends ~21:00 (row **H**) |
| **controller** ends on `[BudgetStop]` | live (row **C**) | **void x2** — row **D** | **yes** — commit 1,348, 95.0% of `B_max` |
| ends within 0.015 of peak | no valid arm yet | no | **score it** — `replay_scoring.py`, first arm that can be |
| `B_max` / `ρ*` **differ across datasets, unsupplied** | sensed **down**, 0.693 → 0.510 | — | sensed **up**, 0.693 → **1.023** over 8 probes |
| no learning knob supplied by hand | **yes**, by construction (table above) | **yes** | **yes** |

**Where that leaves it: not yet, and the gap is arms, not code — except W.** The divergence row is the
one already paying: agnews and yelp-p sensed `B_max` in **opposite directions** with nobody supplying it,
which is what §5 pre-registered as possibly failing to reproduce. Every other empty cell is one
completed arm away, and three of the four are blocked behind the same watchdog bug.

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

**Still not done: no production sim charge profile exists for yahoo or yelp-p**, so every
non-agnews sim arm mis-prices its vclock and needs `--force`. The path itself is proven — both nodes ran
`profile_sim_charges.py` on 2026-08-18 and wrote `smoke/` copies, which price nothing by design — and the
gate has been exercised without `--force` (§11.2). What is left is a full-budget real run per dataset,
which each node does for itself before its pair.

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

**Spec deleted.** Swept `Φ` ∈ {1.5…4} × 3 datasets, reps ≥ 3, mode=noise, 2 independent runs each. The
pre-registered monotone-in-classes prediction was **refuted**; knees are erratic (agnews ~3.0–3.5, yahoo
and yelp-p both ~2.0–2.3). Consequence: **3.1's online probe is mandatory infrastructure, not a fallback.**
Full result: model §7.1/§5.5b. `expts/prep_b1_configs.py` generates the per-dataset configs.

> **One caveat that outlives the sweep.** `probe_inflation_damage.py` reads accuracy off
> `test_global[:2000]`, and `test_index_list` is per-client shards concatenated in client order — so B-1's
> base accuracies (agnews ~0.88, **yahoo 0.73**) are measured on a mildly Dirichlet-skewed slice, not the
> balanced full test set the FL arms use. It does not move B-1's verdict (the knee is read on
> chance-normalized accuracy *within* one curve), but **the 0.73 is not measured on the same distribution
> as an FL arm's 0.30** — see §9.

---

## §4 — Phase 2 · registered-node launch shape

**All three registered nodes (K-1, P-1, G-2) landed 2026-08-13/15 — their specs are deleted per §0 rule 6;
results are in `fl_fwd_ft_practice.md` P3/P4.** Kept below: the launch shape and standing preflight, reusable
for the next registered node. Common launch shape: `expt_scripts/nodes/run_node_*.sh` built on
`_node_lib.sh` (which aborts a node on any arm producing < 5 commits); `run_node_g1b_gate_s.sh` is the
current best template.

**Standing preflight for every arm here** (do not skip; this list is what the eight dead arms were
missing): `--dry-run` first, then `--only --yes --clean --force`; `--num-trainers 100`; assert
`⌈n_req/K⌉ ≤ max_iter` and `projected_real_wall < sim_wall_ceiling_s` (task 0.7, now inside `--dry-run`);
pin the pool for any A/B with `--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off`; and
echo the enactment lines `[ServerStep] [CommitGate] [ProbeDim] [DataBins] [Landing] [BmaxProbe]`.

**A clean `--dry-run` is a prior, not a guarantee** — each of §6's four gates cost a node, and each passed
one first. Read them in flight (§6). G-2's `003648` died with the gate's `I` floored at 1 on **98%** of its
commits (8.40 s/round-trip against `145729`'s 1.75) and the preflight could not see it, because `τ(K)`
prices round trips and that arm converted its budget into commits instead.

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

**And on yahoo the sensor may not fire at all early.** `234931`'s first probe declined —
`base_acc=0.105` against chance 0.100 — because the guard at `FedSgdAggregator.py:978` refuses to read a
knee off a model sitting at chance, so early yahoo commits run the `ln 2` **prior**. Law C is unharmed
(the prior is what it is for), but the divergence is not measurable until yahoo clears chance
([P4.8](fl_fwd_ft_practice.md#p48-yahoo-is-under-trained-not-broken)). **Read the first firing commit off
`[BmaxProbe]`; that index is itself the result.**

### 3.5 — saturation stop *(revised 2026-08-13, not yet implemented)*

**Not a duplicate of 3.3.** 3.3's `Φ` stop catches noise-driven collapse — random-walk displacement in
log-norm space accumulating past the point where the local structure the forward difference relies on
still holds. That failure is FwdLLM-specific, can be catastrophic (0.874→0.296 on `003601`) and needs no
eval history. 3.5 catches the ordinary thing: **the model has extracted the signal the task and capacity
allow.** Training loss keeps falling there by construction, so it cannot detect it; held-out accuracy can,
and the system already computes one (`test-accuracy` via `agg_eval`).

**Design: a Prechelt-style generalization-loss / patience criterion** (Prechelt, *"Early Stopping — But
When?"*) on smoothed held-out accuracy — smoothed the way `Φ` already is (11-eval trailing window, task
0.5), **not** a raw `dAcc/dΛ` slope: a raw derivative of a signal this noisy (±0.045 between byte-identical
replicates near a turn, against ±0.0009 at peak) both false-triggers on a dip and misses a plateau masked
by sampling noise — the failure P4.2/task 0.5 already fixed for raw `Φ` crossings.

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

**All four gates now hold live, the ending included** — yelp-p's `125010` ended on
`[BudgetStop] reason=budget action=halt` at commit 1,348 (2026-08-20), which is the first full-length
arm to reach its own stop. Gate 3 FAILs there at Q5=1.34 **by design**: `I==1` on 100% of the last 200
commits with pool demand met on every one of them. Read the `G-2 signature` line, not gate 3 alone
(§11.7); the arm is valid.

**Three more that are not defects but invalidate the scoring — all enacted by `run_node_p4.sh`, so they
need no operator action; kept because they are the *reasons* its constants are what they are:**

- **A per-dataset sim charge profile.** yahoo burns **0.658 real-s per vclock-s against agnews' 0.255**,
  so no cross-dataset per-vclock comparison is valid until each is profiled. Yahoo-vs-yahoo is fine.
- **A vclock budget well above 40,000** on the seq-256 datasets (now 60,000 yahoo / 50,000 yelp-p). Its
  predecessor was still climbing monotonically at 0.296 when killed at 86% of 40,000, against a 0.73
  backprop reference (§9).
- **`--eval-max-samples 10000` on both seq-256 datasets** — the single largest cost term in the stack, and
  a correctness fix besides (P3's `eval_max_samples` row, [P4.10](fl_fwd_ft_practice.md#p410-the-2026-08-18-smoke--clean-and-what-it-settled)).
  The subsample is fixed and shuffled, so its sampling error is a constant offset rather than per-eval
  noise and peak-vs-final stays as precise as the full set — which is what P4.4's 0.015 bar compares.

**4.1** agnews · **4.2** yahoo · **4.3** yelp-p — *same binary, same flags*, no operator input beyond
model / PEFT / `p`. The only per-dataset arguments are a compute budget and a sim-harness artifact (§-0).

**Acceptance:** each reaches its plateau and **ends within 0.015 of peak**, with sensed `B_max`, `ρ*`, `K`,
`P` logged per run and **differing between datasets without anyone having supplied them**. Read the arm
while it is alive and score it by §11.7's rules.

---

## §7 — The open hypotheses

**The specs, their rungs and their discriminating numbers are
[P5.3](fl_fwd_ft_practice.md#p53-open-hypotheses)** — H-S (is the FD chord the residual 3.5× in `S`?),
H-H (the discarded curvature term `vᵀHv`), H-T (does `K`/`C` matter under *variable* availability?), and
H-J, demoted to a confirmation. K-C is closed. None blocks the queue; all are rung 1–2, so none needs a
node.

**Two standing conclusions that are not hypotheses.** *If probe selection is ever retained*, select on
something other than `|d|` (P6 closed that): curvature `vᵀHv` (≈free), split-half SNR within the bin
(free, orthogonal to `|d|`), or loss decrease at the step scale (the step size is known ahead under
trust-ratio, 1 pass per candidate). *Block-coordinate probing is predicted inert and settles on paper* —
progress/commit falls as `1/√L` while budget/commit falls as `1/L`, so progress per unit `B` is unchanged
and it needs `L`× the commits. It escapes `√(n/p)` only if the gradient is *unevenly* spread.

---

## §8 — Failure modes this plan is written against

*Every one has already happened once; the full accounting is
[P9.3](fl_fwd_ft_practice.md#p93-process-lessons). Kept here as the pre-launch checklist.*

1. **A superseded constant left in a config.** When a doc supersedes a value, grep the configs in the same
   edit. `gate_safety_s` = 0.4 cost three runs and a node.
2. **An emit-only flag never re-costed after being made correct.** The vclock looks healthy right up to
   the moment the runaway safety fires.
3. **A sinking condition without its precondition or its smoothing rule.** State both, before launch.
4. **An instrument whose arithmetic is right and whose input is not.** A probe needs a test that its
   *input* is what you think it is — that is what `check_partitions.py` is on the data side.
5. **Scoring a feature without scoring the composition.** Gate and anneal are each correct and multiply
   into a stall (`N_req ∝ ρ_t²`). Any two new flags need a composition test before both default on.
6. **Extrapolating a progress rate as an accuracy rate.** `A` accumulates *through* the turn while
   accuracy falls; extrapolate `A` only alongside `Φ`.
7. **Two quantities with different origins, subtracted.** State the origin of every accumulated quantity
   next to its formula (P4.7 defect 2).
8. **A flag whose writer is not its only writer.** Grep every assignment to shared run state before adding
   a writer (defect 1).
9. **A dataset constant that is right on agnews by arithmetic coincidence.** Grep the *derived* agnews
   numbers (150 / 1,200 / 7,600 / 192), not just `"agnews"`, and grep `lib/python/flame/` as well as the
   example tree (defect 4).

---

## §9 — The yahoo gap · a sanity ladder, cheapest first

**The question.** B-1's backprop reference reaches **0.73** on yahoo; the P-4 arms reach **0.30**
([P4.8](fl_fwd_ft_practice.md#p48-yahoo-is-under-trained-not-broken)). Every fingerprint on disk said
under-training rather than a broken pipeline. **Rungs 0 and 1 have since been run and both clear the data
path, so the gap is optimization budget** — rung 3, the arms themselves, is what is left.

**Rung 0 — data plumbing · CPU · minutes. VERIFIED 2026-08-16.** `bins × 8 × C == n_train` exactly on
all three datasets at `C` = 100 *and* 1,000. Exactness needs equal shards **and** `shard % batch == 0`,
so `data_coverage()` returns both remainders and the aggregator warns rather than assumes. Automated —
run it, don't re-derive it: `dataset_registry.data_coverage(ds,C,batch)["exact"]` (6 cases in
`test_dataset_launcher.py`, logged in-run as `[DataBins] coverage`) · `check_partitions.py` (6/6, all
three) · `[DataBins] confirmed by trainer` on the wire · `data_id` max over a long arm ==
`total_data_bins − 1`.

**Rung 1 — the discriminating test · `probe_backprop_ceiling.py` · 1 GPU, ~10 min per dataset.
ANSWERED 2026-08-17: the data path clears on both.** Centralized AdamW on the **FL rig's own data path**
— client shards through `TextClassificationDataManager`, capped at `total_data_bins`, evaluated on the
**same** `test_global` `agg_eval` uses; it instantiates `ForwardTextClassificationTrainer` purely to drop
`pre_classifier`, which is what makes `p` 450,340 rather than 1,040,932. The one change from an arm is an
exact gradient instead of pooled forward differences.

> Against a pre-registered ≈0.70-clears / ≈0.30-indicts: yahoo **0.7333 / 0.7263 / 0.7339** over three
> epochs (untrained 0.1018, chance 0.100), matching B-1's 0.73 reference; yelp-p **0.8603 / 0.8596 /
> 0.8736** (untrained 0.4917, chance 0.500). Both flat from epoch 1, so ~0.73 **is** yahoo's ceiling on
> this path, not a truncated curve. agnews calibrates at **0.850** against that dataset's FL control at
> 0.835. **These three numbers are in Standing facts; do not re-measure them.** Command:
> [§11.8](#118-individual-invocations-if-a-wave-has-to-be-taken-apart), and **`tee` it** — the script only
> prints, and the 2026-08-16 run was lost to a closed terminal.

> **No attention mask anywhere in the stack.** `tc_transformer_trainer_distribute.py:713` and `:950` both
> do `x = batch[1]` then `self.model(x)`, dropping `batch[2]`, and `probe_backprop_ceiling.py` does the
> same — so the probe is **faithful to production** and rung 1 stays valid. The model attends to PAD
> tokens on every arm, which depresses absolute accuracy everywhere and plausibly hurts yahoo most (p50
> 84 / p95 367 truncated at 256, against agnews' 41 / 70 at 192). **Rung 1 came back 0.734, so the mask is
> not the fault** — it depresses both sides of every comparison equally. A candidate for absolute
> accuracy, nothing more.

**Rung 2 — is the estimator itself weaker on yahoo? · 1 GPU · ~1 h. Unlocked but unrun.**
cos audit on yahoo (`cos_ground_truth_audit` for ~100 commits, `replay_scoring.py --cos`): a `D`
materially below agnews' 0.10–0.15 means the forward estimate degrades with 10 classes / seq 256 — an
FwdLLM-layer finding, not a controller one. Plus H-S on yahoo (`probe_fd_chord.py`): is the FD chord still
faithful at `max_seq_length` 256?

**Rung 3 — the budget answer · the P-4 arms themselves.** Score accuracy against `Λ`, not commits.
**Pre-registered both branches:** yahoo reaching ~0.6–0.7 by `Λ` ≈ 1.0 ⇒ it was a budget problem and the
agnews `Λ`-curve transfers; yahoo plateauing near 0.35 with `Λ` > 1.0 ⇒ **`Λ` does not transfer across
task**, which closes T5's first open row and is a more interesting result than the arm itself.

**Knobs still to suspect, in order — the obvious ones are cleared.** `p` = 454,954 confirmed from
`[ProbeDim]`; `max_seq_length` 256 and `num_labels` 10 plumbed on both sides; `learning_rate` inert under
`trust_ratio`; `G_rule`, `s`, `P`, `probe_combine` dataset-free. What is left is genuinely dataset-shaped:
**(a)** the `ρ*` band was sized on agnews and a 10-class head may need a larger relative step to leave its
init — the sensed `B_max` is supposed to discover this; **(b)** `train_batch_size` = 8 means each JVP is
estimated on a batch missing most of the 10 classes, where 8 samples cover most of agnews' 4; **(c)** seq
256 truncates ~11% of yahoo documents.

---

## §10 — The tokenized-feature cache *(task F, landed 2026-08-17)*

**One shared cache, `/coc/scratch/dgarg/fl_datasets/fwdllm/cache_dir`, 101/101 shards for all three
datasets.** Absolute, so the launch directory no longer decides which cache a process gets, and on
`/coc/scratch` so every node and every probe reads the same bytes.

### What you need to know when you add a dataset or an α

A cache file is one client's tokenized shard, keyed by everything that changes its tensors:

```
{model_type}_{model_name}_cached_{max_seq_length}_{model_class}_{dataset}_{partition_method}_{client_id}
    distilbert_distilbert-base-uncased_cached_256_ClassificationModel_yahoo_niid_label_clients=100_alpha=1_37
```

- **`partition_method` carries both `C` and α**, so **α=1 and α=100 are different files** and a
  `--partition-method` switch is a MISS, never a stale hit. **An α ablation must be pre-tokenized first**
  or it pays ~30 min inside its own wall budget. Tokenized today: agnews `alpha=1` + `uniform` (101 each)
  and `alpha=0.1` (90, partial); yahoo and yelp-p `alpha=1` (101 each).
- **`client_id` is the trainer's `client_idx`, not its trainer id** — `runner.py:389` sets
  `client_idx = (trainer_id - 1) % client_idx_modulo` (100 in every fwdllm yaml), so 200 trainers would
  wrap onto the same 100 shards. `client_id = -1` is the aggregator's global test set, which `agg_eval`
  needs.

```bash
$PY expt_scripts/pretokenize_dataset.py --dataset NAME --clients 100 --dry-run   # missing + size
$PY expt_scripts/pretokenize_dataset.py --dataset NAME --clients 100 --jobs 16   # ~9 s/client @ seq 256
```

Idempotent, drives the production loader so the bytes are the trainer's own, writes the `-1` global.
Every `--dry-run` carries a `feature cache warm (<baseline>)` row — `ok` at 101/101, or a `warn` naming
this command. Override the location with `FWDLLM_CACHE_ROOT`.

**Two findings this task produced, neither needing a run.** (1) The cache does not change a number: cold
vs warm tensors are byte-identical within a process, and the shard `(input_ids, label)` multiset matches
across three independent draws. (2) **But a shard's row ORDER is not reproducible across tokenizations** —
queue row **G**. `read_instance_from_h5` fans the h5 reads over a 20-thread pool appending to shared
`X`/`y` **in completion order** (and the whole body is inside the lock, so the threads buy no
parallelism), while `transform_examples` pairs `X[i]`/`y[i]` with `index_list[i]` as the **guid**. `X`/`y`
stay paired under one lock and nothing reads `guid`, so **no ledger number is wrong** — but which 8 rows
form a given `data_id` bin changes if you re-tokenize. That is what makes the frozen shared cache the
thing that makes an A/B byte-comparable, and why G waits until after the P-4 arms.

**Sanity gate, standing:** a `--num-trainers 100` yahoo arm reaches commit 1 in minutes, `[DataBins]
confirmed by trainer … 1750 batches` holds, and the first commit's `ρ` still reads `234931`'s **0.0678**.

---

## §11 — Runbook · one command per node, no cross-node dependency

**Every slot below is longer than a login survives — run each inside `tmux`.** The 2026-08-16 backprop
ceiling was lost to a closed terminal.

### §11.0 Environment — required by everything

```bash
export FLAME_CONDA_ENV=test_fwdllm        # base lacks h5py; every preflight exits 2 without it
export FWDLLM_FD_SCALE_INVARIANT=1        # the FD-rescale preflight refuses without it
PY=/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python
REPO=/home/dgarg39/flame
FW=$REPO/lib/python/examples/fwdllm
```

**`/home/dgarg39/flame` is LOCAL disk on each node; `/coc/scratch` is the shared one**, so every node
needs its own `git pull`. **Launch directory no longer matters** (fixed 2026-08-17: `cache_dir` and
`sim_charge_profile_path` are both emitted absolute). **Nothing else crosses nodes** — each node produces
the sim charge profile it consumes (§-1), and the tokenizer cache is on `/coc/scratch` at 101/101 for all
three datasets (§10).

### §11.1 One command per node

```bash
N=<1|2|3|4>
cd $REPO && git pull
tmux new -s p4 "$FW/expt_scripts/nodes/run_node.sh $N 2>&1 | tee ~/p4_node$N.log"
```

Node 1 = the agnews pair · node 2 = real yahoo → profile → pair · node 3 = the same for yelp-p ·
node 4 = the two backprop ceilings, then free. The per-node table and why a pair stays on one node are in
[§-1](#-1--status-board).

**Why the two arms of a pair cost so differently.** The **controller stops itself** at `B ≥ f·B_max`, and
law C's length comes from `(B_max, T_res, f)` — not from the budget, so its ~898 commits cost what they
cost. That is the point: the arm ends on `[BudgetStop]`. The **control has no stop**
(`--phi-stop log_only`, deliberately, so P4.1's past-the-stop counterfactual keeps being measured), so it
runs its vclock budget out.

### §11.2 Smoke a node first — `SMOKE=1`, ~20–30 min

```bash
SMOKE=1 $FW/expt_scripts/nodes/run_node.sh $N
```

The *identical* chain at a small vclock budget (agnews 2,500 / seq-256 1,500) against a 2.0 h ceiling —
the ceiling must stay **above** law C's own ~1.76 h projection or the preflight refuses the arm, so a
small ceiling makes a blocked run, not a short one. Smoke profiles come from a 10-minute real run and
cannot price an arm, so they are written under `sim_charge_profiles/smoke/` and never overwrite the
production ones. The watchdog tightens to a 7-minute stall window and a 25-commit grace.

**It covers** the `set -m` backgrounding and watcher teardown · the watchdog attaching without
false-firing · real mode on every dataset · the profiler, writing into `smoke/` · `check_arm_health`
after each arm · and the `rate` line that sizes the real run (§11.3).

**It cannot cover** the `[BudgetStop]` ending (gate 4 reads WARN by construction: the budget is
deliberately too small) or a `[BmaxProbe]` firing (cadence 150 commits, a smoke controller reaches ~42).
A smoke also runs under `--force`, since its profile is in `smoke/`.

**All four nodes are green** — node 1 and node 4 2026-08-17, nodes 2 and 3 2026-08-18
([P4.10](fl_fwd_ft_practice.md#p410-the-2026-08-18-smoke--clean-and-what-it-settled) has the numbers).
Re-smoke a node only after code lands on its path. **Pass condition:** exit 0 · every arm ≥ 5 commits ·
no `arm_stall.json` · `[DataBins]` right per dataset (150 / 1,750 / 650, `source=registry`,
trainer-confirmed) · **0 zero-steps**.

**The force-drop path is checked by hand, on the node itself** — copy `smoke/fluxtune_<ds>.yaml` to the
production path, `run_sequential.sh --dry-run` with the p4 flag set and **without** `--force`, read the
preflight, then delete the file. Both hosts returned 10 ✓ / 0 ⚠ / 0 ✗. Do not leave it there: a
production profile that merely *looks* present skips the real run on every retry.

**What only the long run can settle, in the order it will show it:**

1. **`[BudgetStop]` on a controller** — gate 4.
2. **The `B_max` injection probe.** It is §-0's hole-2 sensor, and every in-run firing on record was taken
   under the pre-fix anchoring, so the long arm is its **first honest exercise**. Check `[BmaxProbe]`
   moves `B_max` **up** from the `ln 2` prior inside the first 200 commits.
3. **The production profile**, written off `REAL_BUDGET`=3000 before each pair launches. `drain_tail`
   (~5× agnews, the term that matters) will price; `fedavg` will almost certainly be refused again and
   stay agnews-priced. That is the guard working, and it is deterministic (P4.10).

### §11.3 The rate constant — why the smoke sizes the real run

**Measured** (first commit to last, so the pre-commit-1 tokenization stall is excluded by construction):

| arm | commits/h | vclock/h | 898 commits would need |
|---|---|---|---|
| agnews `130614` | 351 | 17,523 | 44,839 vclock · 2.6 h |
| agnews `125619` | 345 | 14,138 | — |
| **yahoo `125713`** (no `--eval-max-samples`) | **79** | **5,470** | **61,932 vclock · 11.3 h** |
| **yahoo smoke `162439`** (controller, 2026-08-18, `--eval-max-samples 10000`) | **282** | **10,088** | **32,082 vclock · 3.2 h** |
| **yelp-p smoke `162510`** (controller, 2026-08-18, same) | **296** | **10,556** | **32,044 vclock · 3.0 h** |

*Size a budget off the **controller** row: it is the arm law C has to carry to 898 commits. The controls
of the same smokes run slower per vclock (9,412 / 9,415 vclock·h⁻¹) and that is the rate their
budget-out wall should be read against, not their commit count.*

**The wall-clock preflight cannot see this.** It prices every commit at a dataset-independent 4.41 s;
pre-fix yahoo measured **45.4 s**. Budgets are therefore sized off measured rate, and
`check_arm_health.py` prints a `budget sizing` line converting any short arm's rate into the vclock 898
commits will cost. **Read a smoke's `rate` and `budget sizing` lines before launching a real run — that
is what they are for.**

**yahoo's old 4.4× was the eval tax, not seq 256** — `--eval-max-samples 10000` took it from 79 to 282
commits/h and yelp-p lands at the same 296 (P4.10). Eval remains the largest single tax at ~30% of arm
wall; that is now a cadence choice, not a defect.

### §11.4 The profile-staleness check is node-dependent — `--allow-stale-profile`

The `sim charge profile is CURRENT` preflight globs the **local** `experiments/run_*_<baseline>_n*_real`
directory for reals newer than the profile's own source runs. That directory is machine-local disk, so
the same profile, config and code pass on a node with no old reals and **block** on one that has them —
`kaylee` blocked on two 2026-08-04 agnews reals where `jayne` passed. Profile validity is a property of
the profile and the config, not of which box holds which run dirs.

**`run_node_p4.sh` passes `--allow-stale-profile`, which downgrades that one check to a warn and leaves
the other nine armed.** Not `--force`, which would also disable `sim charge profile matches dataset` —
the config-derived check that actually protects the vclock. And **not** a re-profile: `fluxtune.yaml` is
what every historical agnews arm and P4's own calibration were priced against.

### §11.5 Can this be shorter?

**The controller arms cannot, and that is the measurement, not a scheduling problem.** Their length *is*
"how long law C takes to reach 0.95·`B_max`", and `T_res`=300 / `f`=0.95 are the constants under test
(§5). Shrink the budget and the arm dies on `max_runtime_s`, which voids it — how all four 2026-08-16
arms were lost. At ~2 h each they are already the cheap half.

**The control arms can, and the yahoo one should be re-cut once its profile exists.** Its 60,000 vclock
was set against a **mis-priced** 0.658 real-s/vclock-s that comes from charging yahoo against an
**agnews** profile — the exact error its own profile fixes. `--sim-wall-ceiling-h 12` is set so the ceiling does
not clip before the budget does.

**Cheaper still, if a slot is scarce:** the controller arms alone answer the acceptance question — the
controls are the comparison, and agnews already has one on disk from 2026-08-16.

### §11.6 Stalling and early termination

`_node_lib.sh` runs `expt_scripts/watch_arm.py` as a side-car for every arm and **kills the run** on:

| predicate | default | why this and not accuracy |
|---|---|---|
| no new commit | 20 min steady-state, 45 min pre-first-commit | a genuine hang; the only unambiguous one |
| `I` floored at 1 over the last 200 commits **and `B` not advancing** (`--b-advance-min`, default 0.005) | ≥ 90% | **G-2's `003648` died at 98%**, but the `I` share alone is not that death — it voided a healthy yahoo controller at 86% of `B_max`. `B` is the exact progress measure and rides on the same record. Needs `--server-update-audit`, so it reads *unavailable* — never *passing* — on an arm without it |
| trips/commit < 3 **and** pool demand unmet > 50% | after 200 commits | the conjunction is the predicate. `trips/commit` alone is `n_req/K`, and law C drives `n_req` down **by design** |
| any `rho_star == 0` | after 200 commits | a requirement of *zero*, not an absent one (P4.7 defect 2) |

**Commits are counted from `version_bump_census`, not `server_update`.** The latter exists only under
`--server-update-audit`, which the scored arms set and the real profiling arm deliberately does not — so
reading it alone saw `commits=0` on two healthy 61- and 63-commit arms and killed both at the
pre-first-commit grace. `version_bump_census` fires once per commit unconditionally. The scan is
incremental (per-file byte offsets): re-reading the whole file each poll is O(run²), and at ~4.5 MB/min a
14 h arm would re-read ~1.6 TB.

**Why `trips/commit` alone cannot be a kill.** It is `n_req / agg_goal`, and the n_target gate sizes
`n_req` from the step it is about to take, so a landing `ρ` shrinks it on purpose. Fitting the
2026-08-20 agnews controller's last 150 commits gives `n_req ≈ 89.4 − 88.4·B_frac`, i.e. **`n_req` ≈ 5 at
its own 0.95 stop** — the floor voids every controller arm at any setting above ~0.5. That arm was killed
at `n_req`=18 of 100 trainers with the demand met on **every one of its 577 commits**. Starving means the
gate is *not being met*; asking for less is the controller working.

**The same argument voids `I` as a solo kill — row W, landed.** `I` is the iteration count
that same gate drives, so it floors at 1 for the same reason `n_req` shrinks — not because the arm is
dying. On 2026-08-20 it killed node 2's yahoo controller at commit 964 / **86.0% of `B_max`** with the
demand met on every commit, and node 3's yelp-p arm climbed `I==1` 12% → 44% in nine minutes while `B`
rose toward its stop; by 1,080 commits that arm read `I==1` on **100%** of its window while still
gaining 0.034 of `B_max` per 200 commits. **Kill on `B` not advancing, never on the shape of a healthy
landing.**

> **Open, and a real question rather than a bug (queue row B4):** a commit pooled from ~5 of 100 trainers
> at the landing point is what the n_target gate says is correct for a tiny step, but it is also the
> regime where the server path has little trainer work amortising it — the sim-fidelity worry the old
> floor was reaching for. Decide it on the arms, not in the watchdog.

**Deliberately NOT the harness's own stall guard.** `converge_watch.py` arms on held-out accuracy and
requires `--target-acc`, which would end the arm on convergence — and an arm that does not end on
`[BudgetStop]` is void. Worse, its signal is backwards here: **reaching a plateau and holding it is what
the controller is supposed to do.**

`NODE_WATCH=0` disables it; `NODE_WATCH_ARGS="--stall-window-s 600"` retunes it. A killed arm leaves
`arm_stall.json` in its run dir and the node prints it. **An arm launched before W landed is not fixed
by it** — the watcher holds the old module in memory. Either relaunch, or drop that predicate alone with
`NODE_WATCH_ARGS="--max-hours <CEIL+1> --i-floor-frac 1.01"` (`1.01` is unreachable; setting
`NODE_WATCH_ARGS` replaces `run_node_p4.sh`'s `--max-hours` default, so pass both), or as a last resort
`kill $(pgrep -f watch_arm.py)`, which drops the hang guard too.

### §11.7 Reading an arm — the only command needed, and the scoring rules

```bash
RUN=$(ls -dt $FW/experiments/run_* | head -1)
$PY $FW/expt_scripts/check_arm_health.py $RUN --expect-controller   # exit 1 = a gate is breached
$PY $FW/expt_scripts/replay_scoring.py $RUN                         # B, Lambda, Phi, A
```

`check_arm_health.py` is §6's four gates as one command — `[DataBins]` (value vs the registry, `source=`,
coverage, trainer confirmation) · `rho_star == 0` count · **trips/commit per quintile** against the ≥3
floor · the `[BudgetStop]` reason — plus the `[BmaxProbe]` trajectory including its **first firing
commit**, which on yahoo is itself a result. Gate 3 still FAILs at the ≥3 floor, deliberately: it now
prints a `G-2 signature` line beneath it (`I==1` share, pool demand unmet, `n_req`) so a controller
annealing on plan can be told from a starving gate — read both before voiding an arm. On an arm without
`--server-update-audit`, gate 2 reads `UNREADABLE`, not `ok`. `_node_lib.sh` runs it after every arm; run it by hand at
~200 commits on a live one. **A controller arm that ends on `max_runtime_s` instead of
`[BudgetStop] reason=budget` is void** — that rule voided all four 2026-08-16 arms.

**Scoring rules, stated once and inherited everywhere** (derivation:
[P4.4](fl_fwd_ft_practice.md#p44-scoring-rules-for-any-ab)):

- Score **peak** accuracy and the stability columns, **never final** accuracy of a diverging arm — ±0.045
  between byte-identical replicates past the turn, against ±0.0009 at peak.
- Compare across datasets on **`A` and per-vclock-hour** — never on `Λ` (different `p`), never per round
  (11.7× different bins/round, §1).
- Read `B` as a fraction of `B_max` and `A` against P4's calibration **while the arm is alive**. Both are
  exact at any horizon, so both failure modes are diagnosable ~20 commits in — do not wait for the
  accuracy curve.

### §11.8 Individual invocations, if a wave has to be taken apart

```bash
# the backprop ceiling (§9 rung 1), one dataset. `tee` it -- the script only prints.
# `nodes/run_node.sh 4` does yahoo then yelp-p with this exact call.
cd $FW && mkdir -p experiments/_probe_logs && $PY expt_scripts/probe_backprop_ceiling.py \
  --config $(ls -1dt $FW/experiments/run_*yahoo*/aggregator_config.json | head -1) \
  --dataset yahoo --clients 10 --epochs 3 2>&1 \
  | tee $FW/experiments/_probe_logs/bpc_yahoo_$(date +%Y%m%d).log

# a per-dataset profile alone. TWO steps, and the first needs a GPU: profile_sim_charges.py
# is offline but pools `vclock_charge` events with time_mode == "real", so only a REAL-mode
# run produces its input -- a sim run contains none. Both flags below MATCH THE SCORED ARMS
# and are not optional: the cos probe runs a backward pass inside the `fedavg` span being
# priced, and an uncapped eval measures every span under contention the priced arms never see.
cd $REPO && $FW/expt_scripts/run_sequential.sh --only fluxtune --mode real --dataset yahoo \
  --yes --clean --no-cos-ground-truth-audit --eval-max-samples 10000 \
  --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30 \
  --adapter-reduction-factor 16 --max-runtime-s 3000
cd $FW/expt_scripts && $PY profile_sim_charges.py \
  --real-run $(ls -1dt $FW/experiments/run_*yahoo*real* | head -1) \
  --out ../sim_charge_profiles/fluxtune_yahoo.yaml --only-observed

# any P-4 pair, or one arm of one
$FW/expt_scripts/nodes/run_node_p4.sh <agnews|yahoo|yelp-p> <controller|control>

# a SHORT arm on the PRODUCTION path -- what the buildplan's S1/S2 rungs use.
# Unset, all three are byte-identical to the built-in budgets. Prefer these over
# SMOKE=1 for a sanity arm: SMOKE also swaps the watch config and routes profiles
# to smoke/, which price nothing by design.
REAL_BUDGET=1200 $FW/expt_scripts/nodes/run_node.sh 2               # real-mode SECONDS
VCLOCK_OVERRIDE=6000 CEIL_OVERRIDE=2.0 \
  $FW/expt_scripts/nodes/run_node_p4.sh agnews controller           # vclock SECONDS, wall HOURS
```

**Read `profile_sim_charges.py`'s `WARN` lines, never `--force` past them.** A refused entry keeps its
prior (agnews) value, which is a *known* mis-pricing rather than a plausible wrong one. The guard scores
the mass carried by the **top 1%** of samples, not the single largest — at n=47 a top-1 test caught the
cos probe and at n=489 it did not, because three stalls of 21% each each sat under the 25% threshold.

`run_node_p4.sh` pins everything: `rf`=16, cos audit **off**, `--num-trainers 100 --c 30 --agg-goal 10`,
per-dataset vclock and real-wall ceiling (agnews 48,000 · 10 h; yahoo 60,000 · 14 h; yelp-p 50,000 · 14 h) and `--eval-max-samples 10000` on both
seq-256 datasets; controller = law C at `T_res`=300 with **no `--rho-star` and no `--b-max`** (that is
what makes it zero-input), control = `rm`/0.25 at `ρ*`=0.06 with `gate_rho_ref=setpoint` and
`--phi-stop log_only`. **`NODE_DRY_RUN=1` runs every arm's preflight in seconds without burning a node**
— do that after any `git pull` and confirm the only `✗` is one you understand (P9.2).

### §11.9 The two CPU tasks, E and G

**E — the saturation stop** (§5's 3.5). No launch. Size the three unsized constants — smoothing window,
`GL` threshold, patience — by replay against the arms on disk (P4's portfolio plus 2026-08-13's
K-1/P-1/G-2 arms), the same way task 0.5 sized `Φ`'s window and P4.1 sized its threshold.
`replay_phi_stop.py` is the model to copy.

**G — the shard row-order fix** (§10), **after** the P-4 arms. Preserve `index_list` order in
`read_instance_from_h5` so `guid` names its own row and two tokenizations of one client agree
byte-for-byte. It re-orders every future shard against the caches the P-4 arms run on, which is the only
reason it waits.

