# Baseline catalog — cross-cutting view + finalized paper matrix

**What this is.** Two self-contained sections — **Felix** (CNN/speech, this project's async substrate
paper) and **FluxTune** (LLM forward-mode, the fwdllm paper) — each with its baseline catalog and
comparison-fit justification in one table, since the two papers never share a results table (§Felix ↔
FluxTune disambiguation). `baselines.yaml` is the source of truth for exact per-baseline knobs; this doc
is the human index of identity + status + comparison fit.

> **Keep in sync.** Every edit to `baselines.yaml` updates the tables below in the same change (§ Keep
> in sync proposes a generator). On disagreement, `baselines.yaml` wins.

**Related:** parity → [`../fwdllm/simulate_fwdllm.md`](../fwdllm/simulate_fwdllm.md);
experiment design → [`../fwdllm/EXPERIMENTS.md`](../fwdllm/EXPERIMENTS.md); paper reconciliation →
[`../fwdllm/EXPTS_CHARTER.md`](../fwdllm/EXPTS_CHARTER.md) (what exactly is ported per baseline lives
here, not in this doc); contributions doc →
[`../fwdllm/fluxtune_contributions.md`](../fwdllm/fluxtune_contributions.md); async substrate
learnings → [`../async_cifar10/PARITY.md`](../async_cifar10/PARITY.md).

---

## Felix — CNN / speech family (async_cifar10, google-speech)

Felix's own paper's baseline set, on the generic asyncfl/syncfl/oort stack. Untouched by the FluxTune
reconciliation below — Felix's evaluation lives entirely on this substrate (see the disambiguation
note at the end of the FluxTune section).

| baseline | sync/async | selector | optimizer | avail tracking | native agg_goal / overcommit | agg_main |
|---|---|---|---|---|---|---|
| **felix** | async | `async_oort` | fedbuff (oort LR, rate `new`) | `client_notify` | c=10 | `main_asyncfl_agg` |
| **oracle** | async | `oracle` | fedbuff | unaware (c=N, all eval'd) | perf-ceiling | `main_asyncfl_agg` |
| **refl** | sync | `refl_oort` | refl (staleness-aware) | ORACULAR | aggr_num=10, ×1.3 | `main_oort_sync_agg` |
| **feddance** | sync | `feddance` | fedavg | own check-in predictor | aggr_num=10, ×1.0 | `main_fedavg_agg` |
| **oort** | sync | `oort` | fedavg | ORACULAR | =agg_goal | `main_oort_sync_agg` |
| **oort_star** | sync | `oort` | fedavg | ORACULAR + `avail_select_filter` | =agg_goal | `main_oort_sync_agg` |
| **fedbuff** | async | `fedbuff` (uniform) | fedbuff (rate `old`) | unaware | — | `main_asyncfl_agg` |
| **fedavg** | sync | `random` | fedavg | unaware | — | `main_fedavg_agg` |

**Comparison fit for Felix** (contribution = handling streaming/temporal misprioritization):
`oracle` (ceiling, same stack), `oort`/`refl`/`feddance` (classic selection/staleness/availability
baselines, same backprop CNN substrate), `fedbuff` (async no-selection floor), `fedavg` (sync
lower-bound reference) — all **✅ compare**. A possible small-on-device-LM backprop extension (still
Felix's own selection/agg, still backprop) would be a Felix-internal generalization result, not a
cross-paper comparison. `fluxtune` / `Felix(P)` / `Felix(P)+IT` (below) never appear in Felix's results
— different substrate/training method (forward-mode) and different paper; see the disambiguation note
(including why C2/`dynamic_kc` is the one FluxTune mechanism plausible as a *Felix-internal* ablation).

---

## FluxTune — LLM forward-mode family (fwdllm: AG News / DistilBERT)

All entries aggregate **gradients** (JVPs), commit on a variance gate, progress on `data_id`, and share
one aggregator entrypoint (`aggregator/main_fedfwd_agg.py`).

**Naming grammar** (compositional: `base` < `base+IT` < `base+IT+O`): unmarked = round-level ·
**`+IT`** = iteration-level reselect/commit cadence · **`+O`** = oracular availability-tracking twin
(unmarked = unaware) · **`(P)`** = **ported** — selection/aggregation/availability-tracking strategy
borrowed from a backprop-trained published system (`FedBuff(P)`, `Felix(P)`), substituted onto
forward-mode/perturbation (JVP) client training here. `FwdLLM`/`FluxTune` carry no `(P)` — FwdLLM is
natively forward-mode, FluxTune is our own system, neither substitutes anything. `(P)` is
display-name-only (yaml/code keys stay plain snake_case) and never itself says *what* was ported —
that's in `EXPTS_CHARTER.md`, not restated per occurrence. **Status:** ✅ EVAL = headline comparison
(published baseline or FluxTune itself) · ⚠ ABLATION = ours, isolates one axis, never headline.

| display name | yaml key | status | substrate | isolates | note |
|---|---|---|---|---|---|
| **FwdLLM** | `fwdllm` | ✅ EVAL (core) | sync, random select | — (the floor) | anchor: same forward-grad substrate + task everything else is measured against |
| FwdLLM+IT | `fwdllm_it_unaware` | ⚠ ABLATION | sync, random select | L1 alone | round→iteration on the sync substrate; pairs with the oracular twin |
| FwdLLM+IT+O | `fwdllm_it_oracular` | ⚠ ABLATION | sync, random select | L1 alone, aware twin | identical to `+IT` at `syn_0` (oracular tracking inert under full availability), diverges Phase 2 |
| **FedBuff(P)** | `fedbuff_round` | ✅ EVAL | async, random select | L0 alone | published anchor (Nguyen et al. FedBuff): does naive async already help before any of our contributions? |
| FedBuff(P)+IT | `fedbuff_it_unaware` | ⚠ ABLATION (key) | async, random select | L1 alone / L2 floor | **the direct L2 attribution floor** = FluxTune − C1/C2ᵖ/C3, L0+L1 held fixed |
| FedBuff(P)+IT+O | `fedbuff_it_oracular` | ⚠ ABLATION | async, random select | L1 alone, aware twin | L2 floor's aware twin |
| **Felix(P)** | `felix_round` | ✅ EVAL | async, oort-smart (ported) | L1, smart-selection substrate | published anchor (Felix, this project's own async substrate paper — preprint public): does availability-aware smart selection alone already close most of the gap before any of our L2 contributions? |
| Felix(P)+IT | `felix_it` | ⚠ ABLATION | async, oort-smart (ported) | L1 (smart) + C1+C3 | **not Felix's native behavior** — grafts FluxTune's own iteration-level reselect cadence onto Felix's selector, done only to isolate L1's contribution and show iteration-level control *alone* isn't enough (motivating C1+C3); never headlined, since crediting a published baseline with our own mechanism in the EVAL row would be an unearned advantage to that baseline. `→ FluxTune` isolates **exactly C1+C3** (not C2 — see ᵖ) — FluxTune's own delta over Felix's contributions, no raw Felix needed |
| **FluxTune** | `fluxtune` ⁿ | ✅ EVAL (ours) | async, oort-smart (ours) | L0+L1+C1+C2ᵖ+C3 | the headline result |

**ᵖ C2 (dynamic K/C) is pending, not landed in the headline config.** `Opt-4 dynamic_kc.enabled` is
wired but **off by default** — `fluxtune`'s current run does *not* include C2 (`EXPTS_CHARTER.md`'s
"Default fluxtune = full stack" line names only C1+Opt-1+Opt-2+Opt-3, i.e. C1+C3, confirming this).
C1/C2/C3 remain the intended three-contribution L2 story; every `C2`/`C1+C2+C3` mention in this doc is
the *target*, not yet the *shipped* headline number, until C2 lands. `fluxtune_dynkc` (below) is the
flag that will promote it from research to headline when that happens — **parked, not deleted** (see
Remaining work #3).

*(`fluxtune_dynkc` — dynamic_kc enabled, ⛔ RESEARCH, not part of the paper matrix yet — this *is* the
flag path that lands C2 into the headline `fluxtune` config once it's validated; parked pending that
decision, not slated for removal — see Remaining work.)*

### FluxTune estimator versions (ⁿ) — `fluxtune_v1` / `fluxtune_v2`, landed 2026-08-15

**Orthogonal to the C1/C2/C3 async-execution story above.** The table's `fluxtune` row (L0+L1+C1+C2ᵖ+C3)
describes *which async-execution/selection/aggregation contributions are on* — it says nothing about the
forward-gradient **estimator** itself (how the JVP probes become an update, and how the server paces the
step). That estimator layer is a **separate versioning axis**, tracked in `fl_fwd_ft_practice.md`
(P2/P3/P4) and reconciled here so it stays runnable from the catalog:

| version | yaml key | estimator layer | fate |
|---|---|---|---|
| v1 (original) | `fluxtune_v1` | coin-flip top-2 probe select, raw in-place SGD, no anneal, sample-variance commit gate (all code defaults, unset in the catalog) | peaks near target then oscillates and collapses (`EXPTS_CHARTER.md` Issue I-1) |
| **v2 (default)** | `fluxtune_v2`, aliased as plain **`fluxtune`** | + `probe_combine=mean`, `server_step_rule=trust_ratio`, `rho_schedule=rm`/`0.25`, `commit_gate=n_target`/`gate_rho_ref=setpoint`, `gate_safety_s=1.5`, cos ground-truth audit (stride 25), `adapter_reduction_factor=64` — `fl_fwd_ft_practice.md` P2.1 items 1,2,3,4,5,6,8, validated 2026-08-09…2026-08-15 | current default for everything not explicitly ablating the estimator |

**The plain, unversioned `fluxtune` name now means v2**, everywhere in this repo and in
`EXPERIMENTS.md`/`EXPTS_CHARTER.md` going forward — use `fluxtune_v1` explicitly only when the ablation
target *is* the pre-estimator-fix behavior. `EXPERIMENTS.md`'s existing run ledger (R1–R4, the
`run_2026070…` entries) predates this split by five weeks and ran what is now called `fluxtune_v1` — those
rows are unchanged/correct as historical fact, just read "fluxtune" there as "fluxtune_v1" mentally, or see
the dated note added at the top of that doc's run ledger.

`fluxtune` is a YAML merge-key alias for `fluxtune_v2` (`<<: *fluxtune_v2_def`) — not a copy-pasted
duplicate — so the two can never drift apart; `flame/launch/baselines.py`'s `deep_merge` deep-copies before
any mutation, so the shared reference is safe across concurrent/sequential experiment launches (verified).
A future **`fluxtune_v3`** follows the same pattern once its knobs validate: copy `fluxtune_v2`'s block,
layer in the new fixes, then re-point the `fluxtune` alias.

**Two known gaps, not yet resolved (see Remaining work #7, #8):**
- `fluxtune_v2` requires `export FWDLLM_FD_SCALE_INVARIANT=1` before launch — this schema has no way to
  express an environment variable, so it's a manual step, documented in the yaml/smoke-yaml comments but
  not enforced.
- `sim_charge_profiles/fluxtune.yaml` was profiled before the estimator-layer changes (i.e. against v1's
  behavior, notably *without* the cos ground-truth audit's real-wall cost). The launcher's sim-mode
  provenance preflight correctly blocks `--only fluxtune_v1`/`--only fluxtune_v2` over this (profile name
  doesn't match baseline name) — but **cannot catch it for plain `--only fluxtune`**, since the name still
  literally says "fluxtune" even though what it resolves to changed underneath it. Real-mode runs are
  unaffected (the sim charge model isn't in that path); a **sim** run of `fluxtune`/`fluxtune_v2` should be
  treated as unvalidated at the vclock level until `sim_charge_profiles/fluxtune.yaml` is re-profiled
  against a v2 real run.

Exact per-baseline selector/optimizer/avail-tracking knobs live in `baselines.yaml`, not repeated here.
**Opt flags (fluxtune, flag-gated, byte-identical off):** Opt-1 `suppress_redundant_weights` (all
baselines) · Opt-2 `var_stopping_policy=plateau` · Opt-3 `agg_rate_conf.type=grad_aware` · Opt-4
`dynamic_kc.enabled` = **C2, wired, off** (pending — see ᵖ above).

**Excluded — related work only, not run on this substrate:** `oort` (raw CNN utility-guided selection)
— subsumed, its utility idea already lives inside `async_oort`/Felix(P)/FluxTune's own selector, cite as
ancestor · `refl` (CNN deadline/staleness-aware agg) — deadline/overcommit model doesn't map to a
variance-gated forward-grad commit · `feddance` (CNN availability-aware selection) — Phase-1 excluded,
revisit Phase-2.

**Reads as three staircases, not one:**
- **sync → async** (FwdLLM → FedBuff(P), both EVAL, round-level): does simply removing the sync barrier
  already help, before any of our contributions? — isolates **L0**.
- **round → iteration**, measured *three times* independently on three selection substrates
  (FwdLLM → FwdLLM+IT; FedBuff(P) → FedBuff(P)+IT; Felix(P) → Felix(P)+IT) — isolates **L1**, with a
  robustness check that it isn't an artifact of one particular selector.
- **random/Felix-selection → FluxTune's own selection+agg** (FedBuff(P)+IT → FluxTune is the
  L0+L1-held, full L2 floor; Felix(P)+IT → FluxTune isolates **exactly C1+C3**) — isolates **L2**.

### Layered contribution framing

Three layers: **L0 async execution** (adapts FedBuff, *not claimed*) · **L1 iteration-level control** —
the reframe that collapses the *control* grain onto FwdLLM's already-fine *execution* grain
(perturbation=one scalar, databin, iteration); *enabling but insufficient alone* · **L2** the three
contributions **C1/C2/C3** (the policies the reframe makes expressible; **C2 pending**, not yet in the
headline config — see ᵖ note under the FluxTune paper matrix). FwdLLM owns the execution primitives;
FluxTune owns L1+L2. **SPRY**: discussed gradient-quality upper bound, not run (weight-splitting ⟂
homogeneous deployment).

### Why round→iteration matters *more* here than in classical FL

1. **The round barrier can outright fail to assemble, not just run slow.** `FwdLLM+IT+O`'s sync barrier
   (`agg_goal ≥ available trainers`) **cannot assemble under unavailability** — round-based selection
   under scarcity is a categorical stall, not a slowdown. The iterative variant can swap out the
   blocking client on the next iteration instead of deadlocking the round.
2. **Forward-mode gradient estimation is inherently multi-iteration per unit of work** — unlike
   backprop-FedAvg, where a round ≈ one fixed-cost local step. A JVP-based data-bin accumulates
   perturbation samples until running gradient-variance drops below `var_threshold` (empirically
   15→34 iterations/bin early→late in training). Round-granularity selection means a slow or departed
   client gates the entire multi-iteration grind underneath it, not just one round; iteration-level
   reselection (FwdLLM+IT, FedBuff(P)+IT, FluxTune) swaps that client out immediately.
3. **Net effect:** classical FL's round boundary is a cheap, bounded sync cost; in forward-mode
   fine-tuning it's multiplied by however many perturbation iterations a bin needs to converge — why
   collapsing the *control* grain onto FwdLLM's already-fine *execution* grain is the enabling move
   (L1), not an implementation detail.

**Quantifiable eval numbers this motivates:** (a) iterations/wall-clock-to-target under `syn_0`, round
vs iteration variants; (b) fraction of aggregator time blocked on the round barrier vs. productively
committing; (c) the categorical does-it-even-complete result for round-based rungs under a Phase-2
`mobiperf_*` trace vs. iteration-based ones.

### Felix ↔ FluxTune disambiguation (both are the operator's own work — Felix's preprint is already
public, FluxTune has not yet published — each paper's results must still stand fully independent)

Felix's paper evaluates image/speech classification via backprop on `async_cifar10`/`google-speech`;
FluxTune only exists on AG News/DistilBERT, forward-mode — no shared task/metric axis for one to appear
in the other's results table. `fluxtune`, `felix_round`, `felix_it` never run, cite, or report inside
Felix's own evaluation.

**Structural rules:**
- **Substrate firewall.** FluxTune-family entries live only in this section; Felix's own baselines live
  only in the Felix section above. Neither is ever merged into a shared cross-paper results table.
- **Naming firewall.** Anything derived from Felix but run on the forward-grad substrate is always
  labeled `Felix(P)` or "NOT FeLiX" in text, captions, and yaml descriptions — never bare "Felix"/"FeLiX"
  — so a FluxTune-paper `Felix(P)`/`Felix(P)+IT` number (EVAL or ABLATION alike) can never be cited as
  "Felix's result." This firewall is what makes headlining `Felix(P)` in FluxTune's own results table
  safe — the `(P)` marks it as FluxTune's re-run, not a claim about Felix's paper.
- **Directionality.** FluxTune may say "we build on Felix's client-tier pools, async execution, and
  staleness/utility-aware selection" (`Felix(P)`/`Felix(P)+IT` are the receipts). Felix's paper may
  never say "FluxTune's approach, evaluated on our task, does worse than ours" — FluxTune's approach is
  never run on Felix's task.
- **Why the asymmetry is principled, not just a house rule.** What ports *into* FluxTune from Felix
  (oort-smart selection, fedbuff aggregation, availability-tiered tracking) are substrate-agnostic
  scheduling/aggregation policies — they decide *when/which clients* to select or commit, never *how* a
  client computes its update — so they transplant cleanly onto forward-mode JVP training
  (`Felix(P)`/`Felix(P)+IT` are that transplant). What does **not** port back — **C1** (JVP-magnitude
  perturbation selection: pick the steepest of `P` measured candidate directions) and **C3**
  (gradient-aware aggregation: align-gate + inverse-variance weighting, built specifically to tame the
  single-scalar JVP estimate's high variance, `fluxtune_contributions.md` §8.3) — are contributions
  *about the forward-gradient computation itself*. Backprop already yields one exact minibatch gradient
  per step: there's no candidate-direction to select among (C1 has no backprop analog) and no JVP-scale
  single-sample noise for C3's variance-gate to damp. So FluxTune's headline contributions are
  *structurally* inapplicable to Felix's training method — not merely "different paper, different task"
  — which is why Felix's paper can argue non-comparison qualitatively rather than needing to run
  FluxTune on its task.
- **The one exception: C2 (dynamic K/C).** Cohort-sizing is a scheduling policy, not a
  gradient-computation one — exactly as substrate-agnostic as the mechanisms that already port *into*
  FluxTune from Felix. It plausibly benefits Felix's own async substrate too. Treat any Felix-side use
  of it as a **Felix-paper-internal ablation candidate** (`dynamic_kc` applied to `felix`/`fedbuff`,
  argued and evaluated entirely within Felix's own contribution set) — not a FluxTune comparison, and a
  separate concern from `fluxtune_dynkc`'s removal from *FluxTune's* paper matrix (Remaining work #3),
  which is scoped to this paper only.
- If Felix's paper later adds a small-on-device-LM backprop experiment for generalization, it stays
  entirely Felix's own method (still backprop) and is never compared against FluxTune in either
  direction — it's a **cited motivating result** in FluxTune's introduction, not a baseline entry.

---

## Keep this table in sync (proposed generator)

Rather than hand-maintaining the tables above, add `expt_scripts/gen_baselines_table.py` (a pure reducer
over `baselines.yaml`) that regenerates them between markers, run in the same change as any
`baselines.yaml` edit. Until then: **edit by hand whenever `baselines.yaml` changes**.

---

## Remaining work (pick up here next session)

Naming is finalized (this doc); none of it is propagated to code/tests/yaml files yet.

1. **Wire the 5 new baselines** (`fedbuff_round`, `fedbuff_it_unaware`, `fedbuff_it_oracular`,
   `felix_round`, `felix_it` — new keys, not renames) into `run_sequential.sh` (baseline/smoke-yaml pair
   list), a smoke YAML each, `test_baselines.py`, `test_config_generator.py`.
2. **Propagate the `fwdllm_plus` → `fwdllm_it_unaware`/`fwdllm_it_oracular` rename into code**
   (currently yaml/doc-only): `test_baselines.py` (rename dict keys +
   `test_fwdllm_plus_is_sync_random_fedavg_oracular`), `test_config_generator.py` (parametrized case),
   `run_sequential.sh`, rename the 7 `fwdllm_plus_n10_smoke*.yaml` files
   (base/momentum/seeded/sim/sim_seeded/sim_short/sim_short_momentum/short/short_momentum);
   cosmetic-only comment fixes in `flame/config.py`, `flame/launch/runner.py`,
   `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`, `flame/selector/random.py`.
3. **Park `fluxtune_dynkc` (do not delete)**: C2/dynamic K/C is a pending, not descoped, third L2
   contribution (see ᵖ note under the FluxTune paper matrix) — `fluxtune_dynkc` is the flag path that
   promotes it from `⛔ RESEARCH` to part of the headline `fluxtune` config once validated. Keep
   `test_fluxtune_dynkc_preserves_legacy_production_default` and its `test_config_generator.py` param
   case as-is; leave the `configs/trainer_base.yaml` comment. Revisit only once C2 either lands (rename
   off the `_dynkc` research name into the finalized headline config) or is explicitly descoped (ask the
   operator first — this doc's ✅/⚠/pending framing depends on the answer). A *Felix*-side reuse of
   dynamic K/C as a Felix-internal ablation (disambiguation note, C2 exception) is a separate, later
   concern either way.
4. **Sweep remaining referencing docs/scripts** for old-key mentions and apply this doc's display names:
   `EXPERIMENTS.md` (§1 table + §10 run ledger), `EXPTS_CHARTER.md`, `simulate_fwdllm.md`,
   `fluxtune_contributions.md`, `MIGRATION_TO_LAUNCHER_FWDLLM.md`, `expt_scripts/compare_baselines.py`,
   `expt_scripts/plotlib/baselines.py` (this feeds figure-legend text — apply the `(P)`/`+IT`/`+O`
   display names here so legends actually render them), `expt_scripts/logical_parity.py`,
   `expt_scripts/run_parity.py`, `experiments.yaml`, `telemetry_manifest.yaml`, `figs.yaml`.
5. **Retune** `felix_round`/`felix_it`'s placeholder `learning_rate: 0.075` (ported from fluxtune,
   marked `TODO(verify)` in `baselines.yaml`) against a smoke run.
6. **Verification pass**: repo-wide grep for every old key name (zero non-historical hits expected),
   `test_baselines.py` + `test_config_generator.py` green.
7. ~~`fluxtune_v2`'s `FWDLLM_FD_SCALE_INVARIANT=1` requirement is a manual env-var export, not enforced by
   config.~~ **Done (2026-08-15).** Still a manual export (the schema has no env-var mechanism to auto-set
   it), but `run_sequential.sh` now has a preflight check: any resolved `adapter_reduction_factor != 16`
   with the var unset is `level: error`, `--force`-able, same treatment as the wall-clock/sim-charge-profile
   checks. Verified: blocks `--only fluxtune --mode real --dry-run` without the export, passes with it,
   and is silent for `fluxtune_v1` (rf=16, doesn't need the var).
8. **Re-profile `sim_charge_profiles/fluxtune.yaml` against a `fluxtune_v2` real run.** It predates the
   estimator-layer changes (2026-08-15) and in particular has no charge for the cos ground-truth audit's
   real-wall cost — a **sim**-mode `fluxtune`/`fluxtune_v2` run's vclock should be treated as unvalidated
   until this lands (real-mode is unaffected). The launcher's provenance preflight blocks `--only
   fluxtune_v1`/`--only fluxtune_v2` on this already (correctly); it cannot block plain `--only fluxtune`
   since the profile's recorded name still string-matches. Once re-profiled, decide whether `fluxtune_v1`
   needs its own profile too or can keep sharing `fluxtune.yaml` (it's byte-identical to what that profile
   was originally profiled against, so sharing is currently correct for v1, not v2).
