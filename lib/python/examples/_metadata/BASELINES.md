# Baseline catalog — cross-cutting view + restructure plan (living doc)

**What this is.** One organized view of every baseline in [`baselines.yaml`](baselines.yaml)
(§1 — selector / optimizer / tracking / knobs, cutting across the CNN-speech and LLM-forward-mode
families), the plan to restructure the fwdllm family into a clean 5-baseline ablation matrix (§2),
and **which baselines are legitimate comparisons for our contributions** (§3 — kept as separate
tables from §1 by design). `baselines.yaml` is the source of truth for *what runs*; this doc is the
human index.

> **Keep in sync.** Every edit to `baselines.yaml` updates the table below in the same change
> (§3 proposes a generator so this never drifts). On disagreement, `baselines.yaml` wins.

**Related:** parity → [`../fwdllm/simulate_fwdllm.md`](../fwdllm/simulate_fwdllm.md);
experiment design → [`../fwdllm/EXPERIMENTS.md`](../fwdllm/EXPERIMENTS.md); paper reconciliation →
[`../fwdllm/EXPTS_CHARTER.md`](../fwdllm/EXPTS_CHARTER.md); async substrate learnings →
[`../async_cifar10/PARITY.md`](../async_cifar10/PARITY.md).

---

## 1. Current baselines (as of `baselines.yaml`)

Per-experiment values (`c`, `agg_goal`, `rounds`, …) live in the experiment YAML and deep-merge on
top; the columns below are the baseline-defining knobs only. `agg_main` abbreviates the
`aggregator_main` entrypoint.

### 1a. CNN / speech family (async_cifar10, google-speech) — generic asyncfl/syncfl/oort stack

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

### 1b. LLM forward-mode family (fwdllm: AG News / DistilBERT) — single variance-gated dynamic-K aggregator (`main_fedfwd_agg`)

All four aggregate **gradients** (JVPs), commit on a variance gate, progress on `data_id`.

| baseline | sync/async | selector | optimizer | avail tracking | reselect | staleness | native agg_goal | defining knobs |
|---|---|---|---|---|---|---|---|---|
| **fwdllm** | sync | `random` | fedavg | unaware | per-round | `exact` | 10 (=c) | var_thr 0.3, unbounded iter cap, suppress_redundant on |
| **fwdllm_plus** | sync | `random` | fedavg | **ORACULAR** (mobiperf_2st) | **per-iteration** | `round_data_id` | 10 | same, `reselect_each_iteration=true` |
| **fluxtune** | **async** | `async_oort` | fedbuff (lr 0.075, C3 `grad_aware`) | `client_notify` 3-tier (mobiperf_3st_50) | continuous | `none` (down-weight) | 3 | JVP-select on; Opt-1/2/3 on; dynamic_kc off |
| **fluxtune_dynkc** | async | `async_random` | fedbuff (rate `old`; `google-speech` name artifact) | unaware | continuous | reject off | 5 | **dynamic_kc ENABLED** (adaptive-K), iter cap 15 — research variant |

**Opt flags (fluxtune, flag-gated, byte-identical off — charter EXPTS_CHARTER §Status):** Opt-1
`suppress_redundant_weights` (all baselines) · Opt-2 `var_stopping_policy=plateau` · Opt-3
`agg_rate_conf.type=grad_aware` · Opt-4 `dynamic_kc.enabled` (wired, off).

---

## 2. PLAN — restructure the fwdllm family to the 5-baseline matrix

**Goal (operator ask, 2026-07-09).** Reframe the fwdllm family as a clean 2×2 of *selection+aggregation
granularity* (round vs iteration) × *substrate* (sync vs async-random), with **FluxTune** the
async-smart full system on top. **Don't add comparison points we won't measure against** (§3 + the
EXPERIMENTS.md §1 note): this set is exactly the related-work axes our contributions innovate on —
**for now we run the set we have.**

**Naming (crisp — decided).** Round-level baselines reuse the canonical published names; the
iteration-level reframings take an **`-It`** suffix. So: no suffix = round-level, `-It` = per-iteration
selection+aggregation.

|                   | **round-level** | **iteration-level** |
|-------------------|-----------------|---------------------|
| **sync**          | **FwdLLM** *(keep; yaml `fwdllm`)* | **FwdLLM-It** *(new; ≈ today's `fwdllm_plus`)* |
| **async, random** | **FedBuff** *(new)* | **FedBuff-It** *(new)* |
| **async, smart**  | — | **FluxTune** *(keep; yaml `fluxtune` = FedBuff-It + C1 guided-JVP + C3 grad-aware + Opt-2)* |

Isolates three axes: sync→async (FwdLLM→FedBuff), round→iteration (FwdLLM→FwdLLM-It,
FedBuff→FedBuff-It), random→smart+opts (FedBuff-It→FluxTune). The story reads cleanly: **FluxTune is
FedBuff-It plus our three contributions**, so FedBuff-It is the direct ablation floor.

### 2a. The new baselines (proposed `baselines.yaml` entries)

| new baseline | yaml key | ≈ existing | sync/async | selector | optimizer | reselect / agg granularity |
|---|---|---|---|---|---|---|
| **FwdLLM-It** | `fwdllm_it` | rename of `fwdllm_plus` | sync | `random` | fedavg | per-iteration (`reselect_each_iteration=true`) |
| **FedBuff** | `fedbuff_round` | (new) | async | `async_random` | fedbuff | select+aggregate at **round** cadence |
| **FedBuff-It** | `fedbuff_it` | (new) | async | `async_random` | fedbuff | select+aggregate at **iteration** cadence |

### 2b. Decisions (resolved 2026-07-09 unless noted)

- **D-B1 — `FwdLLM-It` = rename of `fwdllm_plus`** (operator: shift to new names). Move ORACULAR
  tracking to a Phase-2 unavailability variant — it is inert at syn_0 (charter B1), so `FwdLLM-It` is
  the unaware sync per-iteration baseline. Run ledgers (EXPERIMENTS.md §10) get the name map.
- **D-B2 — "round vs iteration aggregation" on the async fedbuff substrate** = the reselect/commit
  cadence: `FedBuff` advances+reselects per **round** (like FwdLLM); `FedBuff-It` per **iteration**
  (`reselect_each_iteration=true`, like FluxTune). Confirm the exact knob mapping
  (`reselect_each_iteration`, agg-goal boundary) against `fwdllm_aggregator.py` before wiring.
- **D-B3 — `async_random` reuse.** `FedBuff`/`FedBuff-It` reuse `fluxtune_dynkc`'s `async_random` +
  fedbuff with `dynamic_kc` **off** (fixed K/C) and the **agnews** dataset name (not the
  `google-speech` artifact) — the static-K/C, random-selector, no-opts siblings of FluxTune.
- **D-B4 — additive catalog entries, not flags** → [[flag-gate-ab-lifecycle]] "default old" doesn't
  bind; the only lifecycle care is the `fwdllm_plus`→`fwdllm_it` rename (D-B1).

### 2c. Task sequence

1. Add `fedbuff_round`, `fedbuff_it` (+ rename `fwdllm_plus`→`fwdllm_it`) to `baselines.yaml`;
   regenerate §1 (§4).
2. Wire launcher recognition (`run_sequential.sh --only <name>`) + a smoke YAML per new baseline.
3. Add each to the parity + experiment tracks: parity rungs (simulate_fwdllm.md §J — the async pair
   inherits the felix/PARITY.md async substrate) and the EXPERIMENTS.md matrix.
4. Update the run ledger name map (EXPERIMENTS.md §10) for the `fwdllm_plus` rename.

---

## 3. Comparison fit — which baselines legitimately measure against our contributions

**Principle.** A baseline earns its place only if it *innovates on the same axis our contribution
claims*, on a substrate where the comparison isn't confounded. Related-work methods from a different
substrate (backprop CNN/speech selection schemes) are **related work to cite, not eval baselines**,
unless ported onto the shared substrate — and a port that changes the training method confounds
method with strategy. These tables are kept **separate** from the §1 config catalog on purpose
(separation of concerns): §1 is *what each baseline is*; §3 is *what it's a fair comparison for*.

> ⚠ **The Decision column is a reasoned recommendation, not a locked call** — it is the operator's
> scientific judgement. Confirm/override per row.

### 3a. Legitimate comparisons for **FluxTune** (async iteration-level forward-mode LLM fine-tuning; contributions C1 guided-JVP · C2 dynamic K/C · C3 grad-aware agg)

| candidate | primary contribution of that work | why it does NOT fit vs FluxTune | why it DOES fit vs FluxTune | **Decision** |
|---|---|---|---|---|
| **FwdLLM** | backprop-free (forward-mode) on-device LLM fine-tuning, sync round-based | — | same forward-grad substrate + AG News/DistilBERT task; isolates sync→async | **✅ compare** (core) |
| **FwdLLM-It** | *(ours)* sync + per-iteration selection | — | isolates round→iteration on the sync side | **✅ compare** (ablation) |
| **FedBuff** | uniform async buffered aggregation | — | isolates sync→async at round granularity on the shared substrate | **✅ compare** |
| **FedBuff-It** | *(ours)* async random + iteration-level agg | — | **the direct ablation floor** — FluxTune minus C1/C3/Opt-2 | **✅ compare** (key) |
| **felix / FeLiX** | async_oort selection + fedbuff staleness-scalar aggregation for streaming misprioritization (CNN/speech) | raw FeLiX is a backprop CNN/speech method → comparing raw confounds training method (backprop vs forward) and task (image/speech vs text) | FluxTune *borrows* FeLiX's async_oort axis + scalar rate → **FeLiX-on-forward-grad = the `agg_rate type=new` arm** (charter N2, R1 base) | **✅ compare — but only as the ported `type=new` substrate arm**, not raw CNN Felix |
| **oort** | utility-guided participant selection (statistical + system) | selection-only method for backprop FL; not a training-method comparison | its utility idea is already inside `async_oort` (felix/fluxtune) | **⚠ subsumed** — cite as ancestor, don't run standalone |
| **refl** | resource-efficient FL: deadline + staleness-aware agg + priority/pacer selection (CNN) | backprop CNN; deadline/overcommit model doesn't map to variance-gated forward-grad commit → confounded | staleness-aware agg is conceptually adjacent to C3 | **❌ related-work only** (port = separate project) |
| **feddance** | availability-aware selection (V·I·A + MAB, check-in prediction) (CNN) | backprop CNN selection; availability is Phase-2 territory | availability-aware selection relevant to FluxTune's Phase-2 unavailability story | **❌ Phase-1; revisit Phase-2** |

### 3b. Legitimate comparisons for **Felix** (async CNN/speech; contribution = handling streaming/temporal misprioritization)

*Felix's experiments live in `../async_cifar10/`; that example + `PARITY.md` are authoritative. Listed here for the shared cross-example view.*

| candidate | primary contribution | why it does NOT fit vs Felix | why it DOES fit vs Felix | **Decision** |
|---|---|---|---|---|
| **oracle** | fresh-utility performance ceiling (same async stack) | not a real method (upper bound) | shares the exact stack → clean ceiling | **✅ compare** (ceiling) |
| **oort** | utility-guided selection | — | classic selection baseline, same backprop CNN substrate | **✅ compare** |
| **refl** | resource-efficient (deadline/staleness) FL | — | deadline/staleness baseline on the shared substrate | **✅ compare** |
| **feddance** | availability-aware selection | — | availability baseline on the shared substrate | **✅ compare** |
| **fedbuff** | uniform async buffered agg | — | async no-selection floor | **✅ compare** (floor) |
| **fedavg** | sync random + FedAvg | different regime (sync) | classic sync lower-bound reference | **✅ compare** (reference) |
| **fluxtune** | *(ours, LLM forward-mode)* | different substrate (forward-grad LLM) → confounded | — | **❌** — FluxTune is the LLM contribution, not a Felix baseline |

---

## 4. Keep this table in sync (proposed generator)

Rather than hand-maintaining §1, add `expt_scripts/gen_baselines_table.py` (a pure reducer over
`baselines.yaml`) that regenerates §1 between markers and is run in the same change as any
`baselines.yaml` edit (optionally a pre-commit/CI check). Until then: **edit §1 by hand whenever
`baselines.yaml` changes** and note it in that PR.

---

## 5. Changelog
- **2026-07-09 (b)** — crisp naming (FwdLLM/FedBuff round, FwdLLM-It/FedBuff-It iteration, FluxTune)
  + §3 comparison-fit tables (FluxTune vs Felix; feddance/felix/refl/oort scored).
- _(init)_ Cross-cutting baseline table + 5-baseline restructure plan (FwdLLM+Iter / fedbuff+round /
  fedbuff+Iter) + generator proposal.
