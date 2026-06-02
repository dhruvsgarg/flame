# Selector Staleness Audit — design (review before implementation)

## Goal
For each selector, objectively measure **how far its view of the world is from
reality, across every input factor its algorithm uses, over time** as training
dynamics (esp. streaming data unlock) change — and how that staleness translates
into worse selections. Computed **within each baseline** (self-relative), so the
sync-vs-async round-count mismatch never pollutes it. Headline cross-baseline
comparison stays time-to-accuracy + communication; this is the explanatory "why."

Three layers:
1. **Per-factor staleness**: for every input factor a selector consumes, log the
   selector-VISIBLE (possibly stale) value and compute the TRUE current value per
   trainer over time; report believed−true divergence per factor.
2. **Counterfactual selection (replay)**: re-run the selector's *actual scoring
   formula* with the true factor values substituted; compare its pick to the real
   pick → self-relative mis-selection + utility-regret.
3. **Reporting**: per-baseline gap + streaming-vs-control delta; cross-baseline of
   the *normalized* gap (each vs its own perfect-info self) plotted vs wall-clock.

## Input-factor registry (per selector)

Legend — **Dim?**: is this a staleness dimension (believed can differ from true) or
exact bookkeeping the selector knows precisely?

### OORT (`selector/oort.py`, sync) — score = clip(I_m)·sysUtil + temporalUCB
| Factor | Role in score | Believed value (source) | Staleness source | True value (offline) | Dim? |
|---|---|---|---|---|---|
| I_m statistical utility | primary rank | `PROP_STAT_UTILITY` = last value reported when trainer last **trained** (oort_loss @ epoch1/batch0 on data unlocked *then*) | stale by (round − last_train_round); computed on *smaller past* visible data | `|D_vis(now)|·sqrt(mean(loss² ; current global model, currently-unlocked data))` (checkpoint recon) | ✓ primary |
| speed / round_duration | `sysUtil=(pref/dur)^α` | `PROP_ROUND_DURATION` = last measured train duration | under streaming, more data ⇒ longer true train time; believed under-estimates | model from current `visible_samples` (∝ visible fraction × full-data duration) | ✓ secondary |
| temporal UCB | `+sqrt(0.1·ln(R)/last_sel_round)` | round R, last_selected_round | — exact | n/a | ✗ bookkeeping |
| 95th-pct clip | outlier bound | derived from believed I_m list | follows I_m | recompute on true I_m | (derived) |

### REFL (`selector/refl_oort.py`, sync, extends OORT)
| Factor | Role | Believed | Staleness | True | Dim? |
|---|---|---|---|---|---|
| I_m, speed, temporal | as OORT | as OORT | as OORT | as OORT | ✓/✗ as OORT |
| availability priority | split candidates into priority/remaining | `avail_tracker.split_by_priority` (predicted, accuracy<1) | prediction error | actual availability from trace (**syn_0 ⇒ all available ⇒ moot**) | ✓ (moot @ syn_0) |
| blacklist | exclude over-selected | `PROP_SELECTED_COUNT` > threshold | — exact | n/a | ✗ bookkeeping |

### FedDance (`selector/feddance.py`, sync) — score = V_m·I_m·A_m·MAB
| Factor | Role | Believed | Staleness | True | Dim? |
|---|---|---|---|---|---|
| I_m importance | rank | `last_loss` (≈stat_utility) from last engaged round | stale by engagement gap; past visible data | true current loss-utility (checkpoint recon) | ✓ primary |
| A_m accuracy increment | rank | slope of local-acc over last β engagements (`accuracy_history`) | stale; reflects past model/data | current local-acc slope between consecutive checkpoints on current visible data | ✓ feddance-specific |
| V_m availability | rank | Poisson `1−e^{−λK}`, λ from observed check-ins | prediction; check-in history | actual availability over next K (**syn_0 ⇒ ≈1 ⇒ moot**) | ✓ (moot @ syn_0) |
| J_m / MAB | `·(1+log10(R+1)/(10(1+J_m)))` | last_engaged_round | — exact | n/a | ✗ bookkeeping |
| cold-start mean I,A | substitute unseen | prev-round participant means | — exact | n/a | ✗ |

### Felix = AsyncOort (`selector/async_oort.py`, async) — train: OORT-style; eval: staleness
| Factor | Role | Believed | Staleness | True | Dim? |
|---|---|---|---|---|---|
| I_m statistical utility | train rank | `PROP_STAT_UTILITY` = **freshest of last train OR last eval** report (eval refreshes it) | much **smaller** staleness than OORT — the eval-selector refreshes it between trains | same true as OORT | ✓ primary (but fresher) |
| speed / round_duration | sysUtil | `PROP_ROUND_DURATION` | as OORT | as OORT | ✓ secondary |
| eval scheduling | picks eval clients | `PROP_LAST_EVAL_ROUND` staleness gate (≥35 mv) | — exact | n/a | ✗ (this is the *mechanism* that keeps I_m fresh) |
| temporal UCB | train rank | round, last_sel_round | — exact | n/a | ✗ |

**Key contrast the audit should surface:** I_m is a staleness dimension for *all four*,
but Felix's *believed* I_m tracks true I_m much more closely because eval refreshes it
between train rounds — so Felix's per-factor staleness (and resulting counterfactual gap)
should be small and roughly flat as data unlocks, while OORT/REFL/FedDance staleness on
I_m grows with the unlock.

## True-value computation (offline, from checkpoints + deterministic data)
- **I_m(true)**: already implemented in `oracle_misselection.py` (`_oort_utility` on the
  visible prefix at the checkpoint's sim/wall time, under the checkpoint model).
- **speed(true)**: `round_duration_full × visible_fraction(t)` (or measured per-trainer
  full-data duration from the registry × fraction). Cheap, no model needed.
- **A_m(true)** (feddance): local top-1 accuracy on current visible data under checkpoint r
  minus under checkpoint r−Δ, per the β-window definition. Reuses the checkpoint forward pass.
- **V_m(true)/availability**: from the availability trace (syn_0 ⇒ 1). Moot for this experiment.

## Telemetry to add (believed factor values at selection time)
Extend `emit_selection`'s `per_trainer` via each selector's `per_trainer_extra` so every
candidate logs the believed value of each factor it scored on:
- OORT/Felix: `believed_util` (PROP_STAT_UTILITY), `speed_s` (have), `last_train_round`,
  `last_eval_round` (Felix), `temporal_uncertainty`.
- FedDance: `V`, `I`, `A`, `U`, `last_engaged_round` (FedDance already sets PROP_V/I/A/U on
  ends — just surface them).
- REFL: `believed_util`, `speed_s`, `priority_flag`, `predicted_available`.
Also log the per-round explore/exploit split sizes so the replay reproduces the exact k-split.

## Counterfactual replay + metrics (offline, no-drift)
Factor each selector's "score candidates → split explore/exploit → pick top-k" into a
**pure ranking function** that both the live selector and the offline replay call (so no
re-implementation drift). Offline: feed the logged believed factors but substitute the
chosen staleness dimensions with their true values; get `oracle_selected`; compare to
`actual_selected`:
- **mis-selection rate** = 1 − |actual ∩ oracle_selected| / k
- **utility regret** = mean(true I_m of oracle_selected) − mean(true I_m of actual)
- **per-factor staleness** = believed−true divergence (MAE / rank-corr) per factor, over time
All per-baseline; plus streaming−control delta per baseline; cross-baseline only on the
normalized gap, x-axis = wall-clock/elapsed (never raw round).

## Open questions to confirm before building
1. **Which staleness dimensions to substitute in the counterfactual?** Propose: I_m for
   all; + A_m for FedDance; + speed for all (it's a real streaming effect). Leave
   availability out (moot at syn_0). Track *all* listed factors for the per-factor
   staleness report regardless of whether they're substituted.
2. **Refactor selectors to expose a pure ranking fn** (no-drift, ~moderate per selector) vs
   approximate re-implementation in the oracle (faster, slight drift risk). I recommend the
   refactor since you want rigor.
3. **A_m(true) and speed(true)** computations above — acceptable definitions?
