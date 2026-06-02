# FedDance on async_cifar10 — RESOLVED

FedDance now runs as an async_cifar10 baseline (option **A** from the original
plan: wired into async_cifar10), alongside felix / oort / refl / fedbuff /
fedavg. Validated end-to-end on an n10 smoke (selects, emits per-trainer utility
U_m, streaming util_disparity, checkpoints, no crashes).

## What was done

1. **Stack.** FedDance uses `aggregator/pytorch/main_fedavg_agg.py` (base syncfl
   `TopAggregator`), which DOES drive the selector hooks
   `on_update_received` / `on_round_completed` (`syncfl/top_aggregator.py:305,353`).
   The Oort sync stack was *not* usable: it overrides `_aggregate_weights` and
   never calls those hooks, so FedDance's loss/accuracy/engagement state would
   never update.

2. **Required kwarg.** `aggr_num` is now set in the `feddance` baseline
   (`_metadata/baselines.yaml`, default 10) and overridden per experiment to
   match `agg_goal`.

3. **Tracking.** Switched the baseline from `trackTrainerAvail: ORACULAR` to
   `enabled: "False"` — FedDance derives availability from its own check-in
   predictor (`FedDancePredictor.record_checkin`) inside `select()`, not an
   oracular trace. At syn_0 everyone is available anyway.

4. **Trainer signals.** FedDance needs I_m (`STAT_UTILITY`) and A_m
   (`LOCAL_ACCURACY`). The async_cifar10 trainer already sends both via the
   syncfl `_send_weights`, and now also *accumulates* per-round local training
   accuracy (`reset_local_accuracy` at round start + `update_local_accuracy`
   per batch in `trainer/pytorch/main.py`), so A_m is real instead of 0.

5. **Base-stack bug fixes** (these had made the base syncfl stack unusable for
   any non-FwdLLM / non-Oort aggregator, i.e. fedavg + feddance):
   - `_distribute_weights`: `data_id` / `iteration_per_data_id` are FwdLLM-only;
     defaulted via `getattr` so they don't `AttributeError`.
   - `_aggregate_weights`: guarded `PROP_ROUND_DURATION.total_seconds()` against
     `None` (only the Oort stack populates it).
   - `_aggregate_weights`: `self.cache` is a diskcache `Cache` (no `.keys()`);
     use `list(self.cache)`.

6. **Eval cadence.** `main_fedavg_agg.py:evaluate()` now honors
   `evalEveryNRounds` (was every round) so n300 runs are tractable.

## How to run

`feddance` is now a normal baseline. See
`expt_scripts_2026/felix_oort_refl_feddance_quickpilot_n32.yaml` for a 4-way
(felix/oort/refl/feddance) pilot, and add a `feddance` arm to the full
experiment the same way (set `selector: feddance`, `selector.kwargs.aggr_num`
= agg_goal).
