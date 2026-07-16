# FwdLLM Simulator — Design, Build Plan & Roadmap

Non-parity companion to [simulate_fwdllm.md](simulate_fwdllm.md) (which owns real↔sim parity status,
open parity questions, and landed parity fixes for the fluxtune/fwdllm/fwdllm_plus baselines) and
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) (shared parity methodology + fwdllm's rung
catalog, §F). This doc covers everything else about the fwdllm build: how it differs structurally
from async_cifar10 (§B), the baseline matrix (§C), the phased roadmap (§E), the forward-gradient JVP
compute profile (§L), the sim receive/barrier redesign (§M), and the NPU-calibrated delay-factor
derivation (§O) — plus open design decisions not yet made.

**Section letters are kept as they were when this content lived in `simulate_fwdllm.md`**, so existing
code-comment references (`simulate_fwdllm.md §B`/`§C`/`§E`/`§L`/`§M`/`§O`) resolve to the sections
below by letter even though the file changed. `simulate_fwdllm.md` kept §A (status) and §G (landed
fixes) — those stay live there.

---

## §B  How fwdllm differs structurally

fwdllm aggregates **gradients** (JVPs) not weights; commit cadence is **endogenous** (variance-gated dynamic-K);
progress axis is committed **`data_id`** (variance passes), not update count. Gradient values are mode-invariant
given identical input+perturbation seed, so parity reduces to **clock + selection + ordering parity plus a
variance-cadence layer**. Anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150`; force-commit cap
`max_iterations_per_data_id`; `reselect_each_iteration` (fwdllm_plus per-iteration reselection); sync path
`_aggregate_grads_sync`. Full detail: PARITY.md §F.1.

### §B.1  Real↔sim design deltas vs async_cifar10

| # | Axis | async_cifar10 | fwdllm | Why |
|---|---|---|---|---|
| 1 | Aggregated object | model **weights** | **gradients** (JVPs) | grad values mode-invariant → parity = clock+order+selection + variance-cadence |
| 2 | Progress axis | update/round count | committed **`data_id`** | cadence (updates-per-data_id) is an **output to match**, not an input |
| 3 | Commit cadence | fixed `agg_goal` | endogenous **variance-gated dynamic-K** | the emergent layer cifar doesn't model |
| 4 | sct delay model | `send + max(gpu, D)` | `send + max(gpu, D)` (remainder-wait) | real sleeps `max(0,D−gpu)` (device wall = D) so update order = per-trainer D order = deterministic, real↔sim identical |
| 5 | Per-eval sct | distinct, ~20× faster | collapses to train sct | eval lives on the aggregator; forward-grad "train" IS a forward pass |
| 6 | Slot residence | per-commit release | **hold slot to COMMIT** (felix port) — freed only when its update commits | correctness check, not a lever, for BOTH sync & async |
| 7 | Surplus grad on rollback | carried | **carried** for async (c≫agg_goal); **drop** stays correct for sync (c≈agg_goal) | drop was benign only for sync |
| 8 | Async drain primitive | `_sim_recv_min` (+`recv_fifo` default) | `_sim_recv_min_grad` (+opt-in `drain_ready` via `sim_sct_ordered_drain`) | fluxtune's higher per-grad probe frequency vs felix's per-round — necessity of the opt-in is an open parity question, see `simulate_fwdllm.md` §A |
| 9 | `time_mode` default | `"simulated"` | `"real"` (getattr fallback) | fwdllm's whole config corpus is `real`; a `simulated` default risks half-activating an unbuilt path |
| 10 | Availability tracking (v1) | all `trace_read` | mixed: fwdllm unaware, fwdllm_plus `oracular`, fluxtune `client_notify` | baselines carry different models; first-class `client_notify` deferred to Phase 2 |
| 11 | Aggregator-side eval | backgrounded (daemon thread, off critical path, `eval_every_n_rounds`) since inception | now ALSO backgrounded (was synchronous, needed `sim_model_eval_time`'s vclock fold; `simulate_fwdllm.md` §G) | the axis that matters is synchronous/blocking vs. backgrounded, not centralized vs. decentralized — cifar was only ever exempt from a fold by implementation choice, not a structural guarantee (`simulate_fwdllm.md` §F #1/#10) |

---

## §C  Baseline matrix

| baseline | sync/async | selector | agg | tracking / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify`→`trace_read` v1 | — | 3 | disabled |
| **fwdllm** | sync | `random` | fedavg | `default` unaware | per-round | 10 (=c) | — |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` | per-iteration | 10 | — |

Phase order (locked): Phase-1 syn_0 → Phase-2 unavailability → Phase-3 beyond syn_0. One baseline at a time.

---

## §E  Roadmap

**Phase 1 (syn_0) — in close-out, blocked on a new regression.** The parity-CLI checker bugs are fixed
(`simulate_fwdllm.md` §A/§G) and the selector RNG-order fix (`simulate_fwdllm.md` §G) is validated for
fwdllm/fwdllm_plus's dispatch-order identity, but the 07-15 re-run is NOT a trustworthy scoreboard for
fluxtune — its `sim_rate` collapsed to 0.60× (SLOWDOWN) and both real/sim accuracy collapsed to the
random-guess floor, correlated with `[TIMING_OVERRUN]` firing for the first time since §O's calibration
(`simulate_fwdllm.md` §A, top priority). `overhead_residual` for fwdllm/fwdllm_plus remains root-caused
to the real-only `num_min_req=1` compose-loop clamp (`simulate_fwdllm.md` §A) — the RNG-order fix closed
it for fwdllm_plus but not fwdllm, proving selection-order identity and this clamp are separate
mechanisms. `preferred_duration` (fluxtune, oort) is now CLOSED (`simulate_fwdllm.md` §G). Open: the
TIMING_OVERRUN regression (`simulate_fwdllm.md` §A #1), the `num_min_req=1` compose-loop refactor
(`simulate_fwdllm.md` §A #2), `sim_sct_ordered_drain` A/B (`simulate_fwdllm.md` §A #3, now blocked on #1).

Exit: all 3 baselines' `sim_rate`/throughput/terminal_state pass, then C1/C2 convergence at matched `data_id`.

**Phase 2 — unavailability (syn_20/50/mobiperf).** Wire the ClientAvailability effect path into the grad loop:
send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); two ledgers; starvation
vclock-advance; per-baseline `avail_select_filter`. Exit: A1–A5 + A6/A7/A8 PASS; self-stops; withheld grads
delivered not dropped.

**Phase 3 — beyond syn_0.** Full ladder under scarcity. Exit: curves within tolerance at matched `data_id`;
K8/U2 within bar; V1/V2 binned residual flat.

---

## §L  Forward-grad JVP compute profile & retained fluxtune optimizations
*(tool: `scripts/profile_jvp_opt.py` — reuses real `create_model` + `calculate_jvp`; distilbert-base
+ AdapterHub adapters, batch 8, seq 192, A40, fp16. Absolute ms are a CLEAN single-trainer profile.)*

> **The "real run is ~10× from GPU contention" note that used to sit here was wrong — measured 2026-07-15.**
> Replaying the real code (real `create_model` adapter model, 1 pinned core + `OMP_NUM_THREADS=1` as the
> spawner sets, real batch) on one A40: **alone 144.9ms/batch; 4 concurrent 212ms; 12 concurrent 216ms** — GPU
> contention is **1.5×, not 10×**. The n100 real run nonetheless reports `_train_one_batch` at **3952ms**
> (n=4902), while a 10-trainer smoke reproduces **366ms**. So the 18× is a 100-trainer SCALE effect that is
> neither the code nor GPU sharing, and it is unexplained. `tb_*` phase telemetry + `train_batch_unaccounted_cdf`
> (`simulate_fwdllm.md` §B, cross-baseline) exist to settle it on the next n100 pair. **Do not optimize this
> loop further until that reads out** — the math is ~10ms of a 3952ms batch.

> **`select_perturbation_using_jvp` is FALSE in the shipped fluxtune config**, so the 2P=20-pass row below does
> NOT describe the runs: the trainer takes the cos-sim path (1 final JVP = **2 passes**), confirmed by
> `JVP of the perturbation` appearing exactly 4902× for 4902 batches. The 20-pass path is what fluxtune does
> *if that flag is turned on*.

**Mechanism.** Forward-grad trains via a **central finite-difference JVP** (`fwdgrad_utils.calculate_jvp`): each
perturbation = **2 forward passes** `f(θ±hv)`, h=0.01, autocast+no_grad → `jvp=(f(θ+hv)−f(θ−hv))/2h`. **fluxtune**
selects the best of `perturbation_count`(=10) perturbations by |jvp| (2P=**20 passes**); **fwdllm/sync** selects
by cos-sim (**0 forward passes**) + 1 final JVP. Only **~1.5% of params trainable** (bottleneck adapters in all 6
layers + head, 1.04M/67.4M); backbone frozen.

| path | fwd passes | ms/batch (clean) |
|---|---|---|
| sync fwdllm (current) | 5 | 50 |
| sync fwdllm (opt) | 2 | 16 (−68%) |
| fluxtune P=10 (current) | 25 | 251 |
| fluxtune P=10 (opt) | 20 | 159 (−37%) |
| backprop ref (1 fwd+1 bwd) | — | 17 |

- **Compute vs sync:** fluxtune = `2P × per-pass` → 10× sync at P=10, linear in P, equals sync at P=1.
  JVP-selection is the entire fluxtune surcharge; sync's cos-sim selection is free.
- **Memory:** forward-grad peak is FLAT in P (~3.2–3.4 GB = model + one held forward; no autograd graph).
  fluxtune's extra JVP inferences cost TIME, not memory.

**LANDED, fluxtune-only & config-gated** (`jvp_perf_opt`, default false = byte-identical; true in both fluxtune
yamls; sync untouched): trainable-only finite-difference (skip the 98.5% frozen params inside `calculate_jvp`)
+ drop 3 diagnostic-only forward passes + reuse the winner's cached JVP. Bit-identical, real↔sim parity
untouched.

**NOT retained (changes fidelity, excluded per the fidelity bar):** vmap-batching (2.0× win, but ~5% different
in fp16/fp32 from catastrophic-cancellation reduction-order sensitivity); forward-mode AD (slower, different
math); `perturbation_count`↓ (changes the baseline algorithm).

**LANDED 07-15 — pure-overhead removals (no numerics touched; RNG stream verified byte-identical to raw
`torch.randn`).** All were invisible to their own `@timer_decorator` or ran under a disabled log level:
- **Determinism-audit hashes gated** behind `FWDLLM_PERTURB_AUDIT=1`/DEBUG. `_calculate_hash` pulls a tensor
  GPU→CPU and sha256s it; `params hashes` was **unfiltered over all 67.4M params** and ran **twice per batch**
  (`_train_one_batch` + `_prepare_perturbation_tensors`), ~489ms/copy — i.e. re-hashing **253MB of frozen
  weights that never change**, ~59× the ~10ms of actual math, for a log line DEBUG-off discards (f-string args
  evaluate before `logging.debug` checks the level). Same trap in `_randn_wrapper` (12.3µs hashing vs 5.7µs of
  RNG work, per param per pass).
- **`_force_cuda_memory_cleanup` deleted** (trainer + aggregator) and the post-send `gc.collect()`/
  `empty_cache()` in `fwdllm_trainer`. `empty_cache()` issues `cudaFree` (device-wide sync) and returns every
  cached block, so the next round re-`cudaMalloc`s it. Measured: **+14% wall AND peak allocated 817MB→1090MB**
  — it made pressure *worse*; without it a 300-cycle soak drifts **0.00MB** (46GB A40, ~32% used, no leak).
  The aggregator's copy also ran on the eval daemon thread, stalling the main `aggregate()`. On OOM, tune
  `PYTORCH_CUDA_ALLOC_CONF`.
- **127k lines/run of INFO** dropped (`len of candidate_v` = a constant, once per trainable tensor per batch;
  `cos sim values`) — the bulk of the 184–219MB trainer logs.
- **157 lines of dead code** removed: two unreachable stale forks of `_select_optimal_perturbations` /
  `_setup_training_state` (the live ones are nested in `_train_one_batch`) and `_randn_like_wrapper`, whose
  per-perturbation `torch.cuda.synchronize()` existed only to make a debug hash accurate — and was never called.

> **Cross-baseline coverage (checked 2026-07-16, NOT re-implemented) — the overhead removals above are already
> shared by all three baselines.** fluxtune/fwdllm/fwdllm_plus route through the SAME files
> (`aggregator/FedSgdAggregator.py`, `trainer/forward_training/{FedSgdTrainer,tc_transformer_trainer_distribute}.py`);
> they differ by config (selector / agg mode / reselection), not by class. So the hash-gating, `_force_cuda_
> memory_cleanup` deletion, INFO-log drops, and the aggregator per-commit dedup (`simulate_fwdllm.md` §G) all
> apply to fwdllm/fwdllm_plus automatically — nothing to port. Two caveats: (1) **`jvp_perf_opt` is fluxtune-only
> by design** (trainable-only FD + dropped diagnostic passes + cached-JVP reuse) and the SYNC cos-sim path is
> intentionally untouched — that is a different algorithm, not a missing overhead removal, so there is nothing to
> port there either. (2) **the sync baselines' compute has NOT been re-measured post-removal** — §O's
> fwdllm/fwdllm_plus figures (mean 1.215s / max 1.712s) are pre-removal 07-12/13 numbers, so when Phase-1 parity
> switches to those baselines, re-read their `gpu_compute_s` at n100 (as fluxtune's 3.63→0.47s here) and
> re-derive their floor (11.0) the same way — it is almost certainly over-provisioned now too.

---

## §M  Sim receive/barrier redesign — event-driven, zero-hardcoded-wait

**Status: code landed 2026-07-12 (all 9 subtasks), 723 tests green.** Live validation ran 07-13
(`simulate_fwdllm.md` §A) — partial: fixed fwdllm_plus's livelock (confirmed), fwdllm stayed healthy, but
fluxtune's `sim_rate<1` persists under a **different, more precise root** than what motivated this redesign
(`simulate_fwdllm.md` §A: one-grad-per-tick throughput backlog, not a hardcoded-wait/grace-ceiling problem —
the shared `_sim_known_delay_s` cache this redesign built is not obviously the bottleneck, see
`simulate_fwdllm.md` §A's A/B test).

**Design (still current):** one canonical delay-report field `MessageType.MODELED_DELAY_S`; one shared
per-trainer delay cache in `syncfl.TopAggregator` (`_sim_known_delay_s` / `_note_sim_known_delay` /
`_sim_recv_timeout_s`) replacing three previously-divergent per-subclass EMA/budget-fallback copies (syncfl,
asyncfl, fwdllm_aggregator). No hardcoded seed, no cross-trainer fallback — an unseen trainer gets no bound
(the barrier blocks genuinely via `recv_fifo(timeout=None)`, confirmed non-CPU-polling); a known trainer gets
an exact deterministic wait bound. `drain_ready(timeout=None)` returns immediately-empty (can't block like
`recv_fifo` can) — this is why fluxtune's opt-in `sim_sct_ordered_drain` path needs a poll-tick fallback that
felix's default `recv_fifo` path doesn't (`simulate_fwdllm.md` §A / §B.1#8).

**2026-07-13 addition:** `_sim_recv_min_grad` now tracks past-dated-commit telemetry
(`_sim_pastdated_commits`/`_sim_pastdated_gap_max`/`_sim_pastdated_by_source`, ported from felix's
`_sim_recv_min`/`_sim_pop_committable` path which fluxtune's grad loop never went through) — folded into the
`[SIM_GRAD_RECV]` log line. Purpose: make the pending `sim_sct_ordered_drain` A/B (`simulate_fwdllm.md` §A)
legible on the correctness dimension, not just `sim_rate`.

**2026-07-16 sim-sleep audit (fluxtune async = CLEAN; sync-baseline TODOs).** Swept every `time.sleep` on the
sim path. fluxtune's async loop has NO artificial waits: the `_distribute_weights_async` pads (0.1s×2) are
`if not self.simulated` gated, `_await_dispatchable_under_scarcity` is `if self.simulated: return`, the trainer
`pause_execution` (1s) is sim-gated, the aggregator `pause_execution` is dead (not in any `>>` chain), and the
drain uses a 0.01s fast-probe + a timeout *ceiling* (`recv_fifo` returns on arrival). The modeled device delay
(~14.6s) is a vclock jump, never slept — corroborated by `sim_rate` 2.79× (>1). **The 2.79×-vs-~32×-theoretical
gap is NOT sleeps** — it's the queue-bound aggregator's serial per-commit throughput (~225ms; `simulate_fwdllm.md`
§B fluxtune #3); pipelining `_process_aggregation_goal_met` is the real sim-speedup lever. **Sync-baseline TODOs
(fwdllm/fwdllm_plus, revisit when Phase-1 moves to them):** (1) `top_aggregator._aggregate_weights:621` and
`_distribute_weights:1124` each `time.sleep(0.5)` as a retry-backoff when `channel.ends()` is transiently empty —
621's own comment notes sim's back-to-back distribute/aggregate can null `ends`, so this can fire in the sync sim;
guarded (only when nothing to process) but worth confirming against telemetry and gating/shortening if it stalls.
(2) The aggregator `pause_execution` unconditional `time.sleep(1)` is dead now but should be `if not simulated`
gated for safety before any sync-loop rewire re-enables it.

---

## §O  NPU-calibrated `training_delay_factor` per baseline

**Problem.** `lib/python/examples/_metadata/trainer_registry.yaml`'s `training_delay_s` (4–19s, Papaya/FedBuff
mobile-CNN traces) is one shared constant scaled by one shared `training_delay_factor` (0.5, all three
baselines) — a CNN training-round budget, not calibrated to fwdllm's actual forward-grad JVP cost, and (§L)
fluxtune and fwdllm/fwdllm_plus don't cost the same: fluxtune's `perturbation_count`=10 selection is 20
fwd-pass-units/data-bin vs fwdllm/fwdllm_plus's 5 (1 JVP + 3 diagnostic passes) — **~4×, not the ~10× the raw
`perturbation_count` alone would suggest**. One shared divisor can't be right for both.

**Ground data.**
- Real per-sample forward-grad JVP cost for distilbert, measured on the FwdLLM paper's reference NPU device
  (`third_party/ae/fig15/b&c-energy&network.ipynb`, `train_time_dict_dict["distilbert"]["ours"] = 0.3085584`
  s/sample) — matches our config exactly (`use_adapter: false`, `fl_algorithm: FedFwd`,
  `configs/aggregator_base.json:38-40`).
- Per-baseline fwd-pass-unit counts: §L's clean single-trainer profile (`scripts/profile_jvp_opt.py`, A40) —
  fwdllm/fwdllm_plus 5 units, fluxtune(opt) 20 units — cross-validated against the banked 07-12/07-13 real runs'
  `forward_passes_iter`/`perturbations_iter` telemetry (`FedSgdTrainer.py:740-745`): both agree exactly
  (fwdllm/plus 5/1 constant, fluxtune 20/10 constant across all iterations in both runs). A static trace of the
  `select_perturbation_using_jvp=False` code path suggested 1 JVP for all three baselines — **this is wrong,
  discard it; the telemetry+profile agreement is ground truth.**
- 100-trainer registry stats (`trainer_id` 1–100, the pool `client_idx_modulo` draws from): mean=12.51s,
  median=11.0s, stdev=8.49s, range=[2,47]s. By `speed_class`: fast (n=16) mean 3.00s [2,4]; medium (n=22) mean
  6.32s [5,8]; slow (n=19) mean 10.53s [9,12]; very_slow (n=43) mean 20.09s [13,47].

**Reference-device target cost per data bin** (`= fwd_pass_units × per-sample-JVP-time × batch_size / 2`,
batch_size=8; `/2` because 1 JVP = 2 fwd-pass-units by `fwdgrad_utils`' own counting convention):
```
1 fwd-pass-unit (NPU) = 0.3085584 × 8 / 2 = 1.2342 s
fwdllm / fwdllm_plus:  5 units × 1.2342  = 6.171 s / data bin
fluxtune:              20 units × 1.2342 = 24.685 s / data bin
```

**`training_delay_factor` (÷ on `training_delay_s`, `FedSgdTrainer.py:546`, config-only, no code change).**
Anchor: `divisor = registry_mean / target_cost`, uniform across all 100 trainers (preserves the Papaya/FedBuff
relative fast:medium:slow:very_slow spread; only re-anchors the absolute magnitude). A flat **+1.5s buffer**
(midpoint of the 1–2s asked for) is added to each baseline's target cost before deriving the divisor, so the
gap between modeled delay and real GPU compute doesn't run to zero:

```
fwdllm / fwdllm_plus:  target 6.171+1.5=7.671s → divisor = 12.51/7.671 ≈ 1.63
fluxtune:               target 24.685+1.5=26.185s → divisor = 12.51/26.185 ≈ 0.48
```

| | old (shared) | new fwdllm/fwdllm_plus | new fluxtune |
|---|---|---|---|
| `training_delay_factor` | 0.5 | **1.63** | **0.48** |
| registry-mean delay | 25.02s | 7.67s | 26.06s |
| fast-class delay | 6.00s | 1.84s | 6.25s |

**Fast-class headroom (the binding constraint — smallest budget, so checked explicitly, not just the mean).**
Real observed GPU compute (07-12/13 banked runs, this dev GPU, not the NPU): fwdllm/plus mean 1.215s max
1.712s; fluxtune mean 3.630s max 5.618s.

> **RE-MEASURED 2026-07-16 (post overhead-removal) — the 3.630s "compute" was ~87% harness overhead, now
> gone.** The §L determinism-hash / gc / logging removals + the aggregator per-commit dedup (`simulate_fwdllm.md`
> §G) landed AFTER the 07-12/13 numbers above. On the 07-16 n100 pair (`run_20260716_112707`/`_112753`,
> `gpu_compute_s` over 4787 sim / 4198 real updates) fluxtune's genuine forward-grad JVP is **mean 0.47s,
> median 0.39s, p95 0.66s, max 6.1s (real) / 4.9s (sim)** — and it matches real↔sim to ~1% (mean 0.470 vs
> 0.476), confirming the update-duration identity. The MEAN collapsed 3.63→0.47s (the removed hashing was a
> flat per-update tax); the MAX barely moved (5.6→6.1s) because it is now genuine GPU-contention spikes at
> n=100, not overhead. **Zero `[TIMING_OVERRUN]`** in either leg. The p95 (0.66s) is the real budget floor to
> size against; the 6.1s max is a rare contention outlier.
>
> **Recomputed floor (÷0.48, ×1.3 over the new max compute).** The divisor stays **0.48** — it is NPU-fidelity
> (models the reference mobile device's per-databin cost, independent of our harness overhead), so removing our
> waste must NOT lower it. Only the overrun-safety FLOOR changes: `budget ≥ 1.3 × max_compute` → `floor ≥ 0.48
> × 1.3 × 6.1 = 3.8s`. So **fluxtune floor 7.0 → 4.0** (budget 14.58s → **8.33s**, still ×1.36 over the 6.1s max
> and ×12 over p95). The old 7.0 more-than-DOUBLED the fast-class device time (NPU-derived 6.25s → floored
> 14.58s) purely for a now-vanished overrun risk — that is the "don't slow fluxtune trainers unnecessarily"
> over-provisioning. Re-measure `training_overran` on the next run with `--delay-floor 4.0`; fwdllm/fwdllm_plus
> floors are NOT re-derived here (needs their own fresh n100 compute read).

```
fwdllm/plus fast-class: 3.00/1.63 = 1.840s vs observed max 1.712s → margin +0.13s (THIN — watch first)
fluxtune fast-class:    3.00/0.48 = 6.250s vs observed max 5.618s → margin +0.63s (comfortable)
```
fwdllm/fwdllm_plus's fast class is the one to watch for `[TIMING_OVERRUN]` (`FedSgdTrainer.py:549-556`) —
re-tighten (raise the divisor slightly) or accept per that warning's own guidance if it fires.

**Caveats (unchanged from the derivation discussion):** the NPU number is a single benchmark point from one
unnamed device, not a distribution; per-sample × batch_size is an upper-bound linear approximation (NPU
batching may parallelize part of this in reality); this recalibrates delay *magnitude* only — it does not
give LLM-specific heterogeneity *shape* (no data exists on whether cheap phones degrade disproportionately more
on transformer ops than CNN ops).

**Action — update configs to use these, not the old shared divisor.** `run_sequential.sh`'s `--delay-divisor`
is a single value per invocation (§ "Usage"), so the three baselines now need **separate invocations**, not one
shared `--delay-divisor 0.5 --delays on` run across all of them:
```
run_sequential.sh --only fluxtune               --delays on --delay-divisor 0.48
run_sequential.sh --only fwdllm,fwdllm_plus      --delays on --delay-divisor 1.63
```
Any future parity/smoke run that passes `--delay-divisor` must use the baseline-appropriate value above, not
the old 0.5 default.

**Live validation status — see `simulate_fwdllm.md` §A, not this section, for the current state.** Was clean
through 07-14, then blew on 07-15 (`[TIMING_OVERRUN]` 357×/169×/0× fluxtune/fwdllm_plus/fwdllm). **ROOT-CAUSED
07-15, not a GPU-contention or RNG-order-fix artifact**: this section's margin math checks the fast class's
MEAN (3.00s raw delay) as the reference trainer, but the actual binding constraint is the class FLOOR (2.00s,
5 of 100 trainers) — every single overrun traced to exactly those 5 trainers, zero involvement from same-GPU
concurrency or any other speed class (both checked and refuted directly). **Fix landed** (`training_delay_
floor_s`, `simulate_fwdllm.md` §G): floors the raw registry delay before dividing, so only the floor-adjacent
trainers get a wider budget rather than rescaling everyone via the divisor. Originally derived ×1.3 over the
THEN-observed max (fluxtune floor 7.0 / fwdllm+plus 11.0). **fluxtune's floor is now re-derived to 4.0** on the
post-overhead-removal compute (see the RE-MEASURED box above: 6.1s max → budget 8.33s); the 07-16 pair ran
`--delay-floor 7.0` with **0 overruns**, so 4.0 is the tighter, still-safe value to validate next. fwdllm+plus
floor 11.0 is unre-derived (needs their own fresh compute read).

---

## Open design decisions

### Real-mode sync visibility-lag anchor (not yet decided)

`update_visibility_lag_s` is now populated for fwdllm/fwdllm_plus's sync path in SIM mode (surfaces the
pre-existing `_barrier_anchored_lags` computation in `sync_collect_and_accumulate_grads`, previously computed
but never reaching structured telemetry — same gap `commit_gap_s` had for fluxtune, `simulate_fwdllm.md` §G).
**REAL mode has no equivalent wiring at all** — `sync_collect_and_accumulate_grads`'s real branch never
computed a `_barrier_durs` list the way its sim branch (or the base class's own `_aggregate_weights`) does.
Undecided: whether streaming per-message `_update_visibility_lag` or barrier-anchored `_barrier_anchored_lags`
(adapted to fwdllm's variance-gated dynamic-K cadence, not a fixed round) is the right anchor — needs deciding
by reading how `sync_collect_and_accumulate_grads`'s collection loop actually shapes arrival vs. commit for
fwdllm's dynamic-K. Left `None` deliberately rather than guessed at.
