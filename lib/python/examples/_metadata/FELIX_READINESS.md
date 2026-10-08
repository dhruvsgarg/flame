# Felix readiness — backprop FL (async_cifar10, google_speech)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first**; its rules, L and T apply here unrepeated.
- **CURRENT FOCUS:** Felix feature-complete (syn_0 + unavailability, both datasets), paper runs sim-only.
- **Correct, not just equal:** every baseline follows its reference and hits target (cifar 50%, speech 60%) on full data.
- Root and fix every red cell per run, so the matrix reaches parity in fewest runs.
- **Parity work follows [PARITY_READINESS.md](PARITY_READINESS.md)**: rules, method, scoreboard, run queue.
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ `*_oracle` arms, FX-N13). Out: `fedavg`, `oracle`.
- **Traces:** `syn_0`, `syn_20`, `syn_50`, `mobiperf_3st`. `syn_10` = the old "syn_20" (10.8%, EuroSys'26; FX-D18).
- Levels (n=300, TRAIN/EVAL/UN %) flat 1 min-3 h: syn_10 89/0/11 · syn_20 80/0/20 · syn_50 50/0/50 · mobiperf_2st
  10/0/90 · 3st_50 10/25/65 · 3st_75 10/10/80. Mobiperf trainable share rises to 18-22% by 6-24 h.
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built. Shared work is `S#` in the parent.
- **Deprecated reference:** [PARITY.md](../async_cifar10/PARITY.md) §2-5 · [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md)
  · [BASELINES.md](BASELINES.md).

---

## Status grid

Parity per run and per baseline: [PARITY_READINESS.md](PARITY_READINESS.md) → Progress dashboard, Felix scoreboard.
pytest: 2450 passed · 0 failed · 7 skipped (2026-10-08, FX-D77-D91).

**Accuracy table (FX-N74; `scripts/accuracy_table.py --runs <dirs>`).** Test acc % on the leg's own clock, syn_0, reference n.
G1A = full data (runs 16-18); G2 streams data 0 → 100% over 3 h (lower bound). `-` = no eval in window.

| dataset · baseline | leg | real @30 / @60 / @90 · max | sim same | note |
|---|---|---|---|---|
| cifar · felix | G1A | 36.2 / 51.8 / 56.9 · 57.0 | 40.6 / 50.6 / 53.7 · 56.3 | **target at 57 / 60 min** |
| cifar · refl | G1A | 24.9 / 32.2 / 44.5 · 44.5 | 29.1 / 35.7 / 44.1 · 44.1 | rising at 90 min |
| cifar · fedbuff | G1A (2024 lr) | 20.0 / 24.0 / 32.7 · 34.7 | 23.1 / 31.5 / 31.7 · 38.1 | superseded by FX-D73 row |
| cifar · fedbuff | G1AS SGD 0.04 × 1.0 (run 21) | — | 39.6 / 46.0 / 56.4 · 56.4 | **sim target at 63 min** |
| cifar · oort, feddance | G2 stream | oort 24.8 @60 · 37.7; feddance · 16.6 | oort · 31.4; feddance · 35.4 | |
| speech · felix | G1A SGD (run 18) | 41.5 / 58.8 / 67.5 · 67.5 | 46.0 / 56.0 / 68.1 · 68.1 | **target at 72 min both** |
| speech · fedbuff | G1A SGD (run 18) | 40.8 / 54.6 / 59.1 · 60.2 | 42.5 / 55.9 / 64.0 · 64.0 | **target at 82 / 88 min** |
| speech · refl | G1A Adam (run 16) | 3.9 / 29.3 / 38.7 · 38.7 | 3.9 / 31.0 / - · 39.7 | 82 rounds, 65 s/round |
| speech · refl | G1AS SGD (runs 19, 21) | — | lr 0.005: 43.7 max; lr 0.05: 53.0 max | rising, below 60% |
| speech · oort | G1A SGD (run 18) | 25.8 / 44.1 / 50.8 · 50.8 | 18.8 / 45.3 / 55.1 · 55.1 | 102 / 113 rounds |
| speech · feddance | G1A SGD (run 18) | 3.7 / 28.8 / 28.8 · 28.8 | 3.7 / 27.7 / 27.7 · 27.7 | round-bound: 38 rounds, 142 s/round |
| speech · oort, feddance | G2 stream | oort **60.7 at 80 min**; feddance 21.5 | oort 57.2; feddance 45.2 | |

**Cross-cutting capabilities**

| capability | status | item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ both datasets | run 6 L4 |
| fail fast; ladder runner + offline re-gate | ✅ | FX-D22, FX-N42 |
| GPU join/warm-up at n=100; sync wait-K | ✅ joins 44-99 s; EV7 green | FX-D19, FX-D20/21 |
| real async ingest < 2 s | ✅ G0U queue_wait max 0.37 s (run 25) | FX-D16, FX-N70 |
| profiled sim charges per dataset/platform | ✅ K3b green every syn_0 cell | FX-D23 |
| DIST tolerances from replicate floors | 🟡 n=2 floors gate regrade; none tightened | FX-D37, PARITY Q2 |
| streaming / oracle experiment | 🟡 cifar EV19 green; oracle advantage ungraded | FX-N13 |
| sim speedup vs real | 🟡 cifar 4.5-7.4×; 🔴 speech 0.6-1.1× (GPU-bound) | FX-N22 P8 |
| paper experiments sim-only | ⬚ | FX-N12 |

---

## Active build — fast parallel harness (FX-N22)

CPU harness catches logic bugs; distribution parity is decided on GPU.

**Isolation contract: a parallel leg behaves as if alone.** Each slot gets:
- disjoint cores via `taskset` (`FLAME_AGG_CORES`: 2 at n≤30, 8 on GPU legs);
- a private mosquitto (`FLAME_MQTT_BROKER`); fixed client ids collide on shared brokers;
- a run tag (`FLAME_RUN_TAG`) scoping kills, and exact run-dir handoff (`FLAME_RUN_DIR_FILE`);
- GPU legs: healthy GPUs (FX-T36), CPUs = 8 + 0.4·n (cifar 0.2·n, FX-D49).

**Shapes** (`harness_pool.SHAPES`; aggGoal via launcher `agg_goal` = sync K, FX-L36):

| shape | n | aggGoal | c | runtime | trace scale |
|---|---|---|---|---|---|
| syn_0 | 12 | 3 | 5 | 180s | – |
| syn_0b | 15 | 2 | 8 | 180s | – |
| syn_20 | 15 | 3 | 6 | 240s | 4 |
| syn_50 | 15 | 3 | 6 | 1200s | 4 (FX-L43) |
| mobiperf_3st | 45 | 2 | 4 | 240s | 4 (FX-L34) |

**Tiers** (`scripts/harness_pool.py --tier A[,B] --datasets cifar10|google_speech|all`). Every pool opens with a ~4-min gate
(pytest collect + felix smoke pair per dataset). Speech CPU sims get a 2× wall ceiling (aggregator-bound).

| tier | jobs | wall |
|---|---|---|
| T1 | sim-only, syn_0 + syn_50, 120 s, EV only | 6 min |
| T2 | sim-only × 4 shapes vs banked reals + P11a-c | 38 min |
| T3 / T3C | real+sim pairs × 4 shapes / a real replicate per cell (floor) | 82 / 22 min |
| T3S | oort/oort_star/refl at aggGoal 10 (stragglers cross rounds) | 24 min |
| T4 | campaign P1-P11 (`harness_campaign.sh`) | 70 min per dataset |
| TS / TSo | streaming CPU pairs × {linear, events} × {syn_0, syn_50}; +oracle | 7 min |
| PROF | n=10, aggGoal = c = 10, `FLAME_PYSPY` profiles (FX-D77) | 4 min |
| G0 / G0C | GPU 30-min screen, syn_0 + syn_20 / real replicate | ~6.9 h node |
| G0U / G0UC | unavailability screen syn_50 + mobiperf, n=50 / real replicate | cifar 15 min, speech 30 min |
| G0T / G0To | streaming GPU screen × {lin, events} × {syn_0, syn_50} / oracle | cifar ~2 h |
| G1 / G2 | 90 min reference config: felix+fedbuff / other four | ~2.5 h per pair |
| G1S | felix + fedbuff, reference n, 45 min, 4 GPUs (fix confirm) | speech ~100 min |
| G1A / G1AS | accuracy: syn_0, full data, eval every 20 rounds, 90 min / sims only | ~2.5 h / 25 min |
| G1U / G1L | felix+fedbuff syn_50 at reference n / felix syn_0 7500 s + replicate | ~5 h each |
| G2S / G2C | one G2 cell replicated sim-only / real-only (chaotic-selector floor) | 8 / 97 min |
| N62 / N64 | unaware oort syn_50 3 h CPU / refl speech health (EV18) | ~5 h / 45 min |

- `--changed <ref>` picks affected baselines; `--exclude-phases` drops phase ids; `--gpu-ids` pins GPUs.
- Pool skips busy GPUs/cores between legs, scaling with foreign load (`LOAD` in `pool.log`).
- All pools run on jayne; run dirs live in `async_cifar10/experiments/`.
- Tests: `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`, `tests/launch/test_debug_run_dataset_profile.py`.

## Active build — streaming time-to-accuracy (FX-N12, FX-N13)

- **Experiment:** six baselines × two datasets × four traces × two stream modes; metric time-to-target.
- **Claim:** Felix's eval selector finds clients whose data just grew; others don't.
- **linear:** each trainer grows 10% → 100% of its shard by the horizon.
- **events:** starts 10%; nine 10% chunks land at seeded uniform times, unsynchronized.
- Stream clock = trace time; horizon 6 days (paper) or leg span (screens); chunks ignore availability.
- **Built (FX-D47):** one `stream_schedule.py` for trainer, oracle, replay and EV19; tiers TS/TSo, G0T/G0To.
- **Left:** ST5 speech (after FX-N70, FX-N30) · ST6 sim-only sweep + oracle advantage (needs P8).

## Unavailability v1 (FX-N6 audit, 2026-10-04): kept; open rows below

Kept rules live in Built (Availability) and FX-L11-L15, L25, L33.

| item | verdict | note |
|---|---|---|
| AVL_EVAL + empty-pool cleanup (land-mines 8, 13) | open | ran on mobiperf G0U; counts unread |
| A2 two-tolerance shape (land-mine 3) | open | only refl syn_50 A2 red at n=50 (KS 0.24) |
| legacy `trackTrainerAvail` | change | delete (FX-N7) |

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10. Run queue + pre-launch checklist: PARITY_READINESS.
**Test levels:** pytest (~4 min, in-process) · CPU tiers T1-T4 (real processes, MQTT, n 12-45) · GPU G0 screens, G1/G2 production n.
**Unblock map:** FX-N9 + FX-N70 → FX-N11 → FX-N12; FX-N12 also needs FX-N13 ST6 + FX-N22 P8.

- **FX-N77 `[C][S]` · Profile-led real-cost fixes, then confirm · wip.** FX-D77-D91 landed from PROF profiles.
  - Speech commits per window up 20-90%; real queue_wait max now < 0.4 s.
  - Run 27 G0U syn_50 green: speech felix + fedbuff, cifar felix + oort.
  - Cifar fedbuff red vs own real, green vs run 26 real: n=2 floor can't split.
  - Eval stays fp32 (fp16 1.66× faster but changes the metric).
  - C7 gap: real asyncfl selector abandons emit no `abandon_timeout` event.
  - *Next:* PR21. *Exit:* cifar fedbuff green on a ≥ 3-leg floor.
- **FX-N76 `[C][S]` · C13 real-cost audit (absorbs FX-N73) · wip: confirm in PR21.**
  - Recv, cache release, eval, tail and ingest costs cut (FX-D74, D77-D83).
  - Open: run 22 sim recv gap contention-inflated (weights_to_ram 0.23 vs 0.006 s).
  - *Exit:* speed identity + phase DIST green on both datasets.
- **FX-N74 `[C]` · Every baseline reaches target on full data, both sides · wip.**
  - At target: cifar felix, speech felix + fedbuff (SGD 0.04 b16).
  - Cifar fedbuff: FX-D73 lr reaches 56% in sim; needs a real pair.
  - Cifar refl 44% and speech refl 53% (lr 0.05) still rising at 90 min.
  - Speech oort 51 / 55%; speech feddance 28%: round-bound by design (operator: longer window?).
  - *Next:* PR21 G1A pairs. *Exit:* target or named root per baseline; acc diff within floor.
- **FX-N9 · GPU unavailability syn_50 → mobiperf_3st, all six · wip.**
  - G0U syn_50 green: felix, oort, fedbuff (speech) after FX-D66-D76, D87-D91.
  - Open: oort_star G0T lin syn_50 K3b 21.8 vs 16.6 s; refl syn_50 A2 KS 0.24.
  - Open: G0U/G0UC mobiperf all six (lost in run 25); AVL_EVAL counts.
  - *Next:* PR21. *Exit:* A1-A8/K11 + syn_0 ladder green; G1U n=300 INV/EXACT green.
- **FX-N70 `[C][S]` · Real ingest spikes on 29 MB updates · wip: fix in tree (FX-D79), confirm.**
  - Run 18 G1A queue_wait max 11 s felix; 25/45 spikes followed an eval.
  - Felix +0.12 s/update over fedbuff unexplained (BN ring?). *Exit:* U6 green or floor-SKIP.
- **FX-N10 · google_speech on the launcher · wip.** *Next:* stop rule (20 evals ≥ 60%), then S4 cleanup.
  *Exit:* all six graded on speech, CPU + GPU.
- **FX-N13 · Streaming experiments, both datasets · wip: cifar ST5 done.** *Exit:* ST5 speech; ST6 oracle advantage.
- **FX-N71 · tiny_cpu cifar felix K4 4.1 vs 4.9 in flight · todo.** Sim per-commit clock 0.66 vs 0.55 s.
  *Next:* `logical_diff.py` on P6, then per-commit `agg_timing`. *Exit:* K4 green on P6/P7.
- **FX-N42 `[S]` · Parity ladder · wip (PARITY Q2-Q6).** *Exit:* a run graded per cell on both axes.
- **FX-N33 `[C][S]` · Aggregator aborts at interpreter exit · todo.** C++ thread destroyed joinable after clean leave.
  Data intact; FX-D22 allowlists the signature. *Next:* diff what sim alone starts at exit.
- **FX-N30 `[S]` · Speech tiny_cpu too heavy for CPU slots · todo.** P7/P7o killed early; oracle select blocks MQTT.
  *Exit:* speech P7/P7o EV green (shrink model, more cores, or GPU).
- **FX-N22 · Fast parallel harness · wip.** *Exit:* P6 EQUIVALENT at cpt ≤ 0.5; T2 both datasets < 25 min.
- **FX-N7 · Remove legacy `trackTrainerAvail` · todo: unblocked.** *Exit:* code gone, T12 retired, pytest green.
- **FX-N2 · Parent S2 pipeline for async_cifar10 · todo.** *Exit:* stored Jun pairs regrade within floor.
- **FX-N8 · Concurrent-run confound on fedbuff · likely closed by FX-N22.** *Exit:* P6 EQUIVALENT.
- **FX-N11 · google_speech GPU parity · blocked: FX-N9, FX-N10.**
- **FX-N12 · Felix paper experiments sim-only · blocked: FX-N11, FX-N13, P8.** = streaming ST6.

---

## Baseline hyperparameters (operator 2026-10-01)

Source of truth: `datasets.yaml` `by_baseline`, each field tagged [paper]/[repo]/[ours]. Baselines stay themselves (PARITY C14).
- Each baseline uses its paper's values; lr decay only where configured (REFL).
- Late updates follow the baseline's rule: REFL staleness ≤ 5; oort, oort_star, feddance none.
- REFL speech lr: paper 0.005 is default; repo 0.05 under screen.
- REFL `stale_update: 5` kept (operator), though the paper default is unbounded.
- FedScale lr decay 0.98/10 kept off except REFL.
- Cifar felix SGD 0.04 × server 0.3 (2024); cifar fedbuff SGD 0.04 b32 × 1.0 (FX-D73).
- All speech baselines: SGD 0.04 b16 (oort's client); server 0.5 felix, 1.0 fedbuff (FX-T27).
- Our CifarNet differs from REFL/FedDance's ResNet18; their lrs are starting points.

## Felix lessons (dos)

**Selectors (Oort family)**
- **FX-L1** Oort pacer fires on TRAIN only; flat trend raises, sharp drops `round_threshold`.
- **FX-L2** UCB temporal term keys on last receipt round, initialised at registration.
- **FX-L3** If a faithful controller still diverges, instrument its input by quartile.
- **FX-L4** Async fires per-round terms 2-3× more; felix uses `exploration_decay` 0.999.
- **FX-L5** Record stale-but-returned trainers' speed and utility, else Oort re-picks them.

**Aggregation, ordering, clock**
- **FX-L6** Async sim drains per-end queues in sct order; holds busy slots until commit.
- **FX-L7** Sync oort over-selects ×1.3; stragglers land stale next round, as real.
- **FX-L8** refl/oort: a still-computing trainer leaves eligibility via the unavailable path.
- **FX-L9** Strict-barrier baselines anchor visibility lag on the barrier (`max_dur − dur_i`).
- **FX-L10** Split eval from train in `agg_rounds` checks; each eval has its own sct.
- **FX-L30** Replay reads the stream clock: sim vclock, real wall since `AGG_START_TS`.
- **FX-L31** Committed ends leave RECV; a phantom buffered end blocks starvation wake-up.
- **FX-L32** A dispatch consumes the end's earlier receipt and RECVD state.
- **FX-L45** In sim, arrival ≠ receipt: unreceivable updates keep real's unanswered-dispatch state.
- **FX-L54** Reinject every withheld delivery due before the next buffered sct.
- **FX-L60** Keep real measurement off the ingest path; read queue_wait spikes against evals.

**Availability**
- **FX-L11** Two axes: select filter (felix, oort_star, refl, feddance); proactive evict (felix only).
- **FX-L12** Mid-flight dropout: compute completes, send gated or buffered, commits late and stale.
- **FX-L13** Sim jumps the vclock under scarcity; size `--runtime-s` for it.
- **FX-L14** Size n ≈ threshold / (1 − unavailable fraction); in-flight differs per baseline.
- **FX-L15** Ramp syn_0 → syn_20 → syn_50 → mobiperf; only mobiperf exercises AVL_EVAL.
- **FX-L25** Sim starvation wake-up = earliest future slot-freeing event, never a due one.
- **FX-L33** Under the substrate only withhold, evict or abandon free an in-flight slot.
- **FX-L34** mobiperf_3st is ~10% AVL_TRAIN: size n ≈ select / 0.1 × 1.5.
- **FX-L43** A leg must see ≥ 80% of the named unavailability and several cycles.
- **FX-L44** Trace time 0 = join barrier; trainers hold trace time 0 until first dispatch.
- **FX-L58** Instant decisions (send gate, select filter) read the trace at that instant.

**One task per version, rounds**
- **FX-L26** One task per (trainer, version); train at v blocks eval at v; retry policy `none`.
- **FX-L27** Update identity is (trainer, version): dedup commits, drop answered requests.
- **FX-L28** Version advances only on `agg_goal` aggregated updates; empty rounds still free slots.
- **FX-L40** Time out each trainer 90 s after its own dispatch; timeouts never close rounds.
- **FX-L41** Accumulate state in float, cast once; server steps touch float parameters only.
- **FX-L42** An empty sync round re-dispatches the same version, excluding ends already tasked.
- **FX-L48** BN stats = convex mean of absolute stats, never base + scaled stale deltas.
- **FX-L55** Real idle loops wake at the earliest future deadline, on its stamping clock.
- **FX-L59** Watch real selection and telemetry rates; busy-loops commit normally while writing GBs.

**Harness, compute, datasets**
- **FX-L29** Stub compute span was fit on CPU-fallback runs; refit from a GPU real leg.
- **FX-L36** Set shapes via `experiment.aggregator.agg_goal`; it sets aggGoal and sync K.
- **FX-L37** Datasets differ only in `datasets.yaml` + `fl_data.py`; machinery is shared.
- **FX-L38** Pay first CUDA touch in `initialize()` (`_warmup_device`), never in timed tasks.
- **FX-L39** Time sim-speed guards (wall ceiling) from the join barrier.
- **FX-L49** Run no-backward forwards under `no_grad`; free CUDA cache after idle GPU steps.
- **FX-L50** Near-D GPU time makes speed contention-driven; check overrun share first.
- **FX-L51** Profile the aggregator with fresh buffers; reused buffers hide page-fault cost.
- **FX-L57** Check lr pairs in-process (`fl_lr_check.py`) first; grade accuracy on full data.

**Reading the checker**
- **FX-L16** A2 red with S3/4 green is one in-flight gap; walk to residence.
- **FX-L17** U6 KS on a sub-ms point mass is noise; read `mean_diff`. P3 needs `grid_KS`.
- **FX-L18** `gate_holds = 0` all run means an inert gate; suspect upstream accounting.
- **FX-L19** Recheck `phase_gpu_compute` and refl K2 at ≥ 2.5 h before acting.
- **FX-L21** `Sdet`: eligible differs + aggregates match = PASS; same eligible, different picks = score values.
- **FX-L22** Past-dating has two streams (U6 train-only, SIM_CLOCK_DIAG); name the source.
- **FX-L23** `sim_committed_fresh == agg_goal` confirms oort block-for-K; later gaps are other roots.
- **FX-L24** Re-run real only when the real path changed.
- **FX-L46** Split sync clock residual into stall-free advance and stall count.
- **FX-L47** Audit real receive: arrivals vs processed, and growing `active_task_skips`.
- **FX-L52** A sync round holding an abandon or gated update is a stall at any length.
- **FX-L53** syn_* flips share slot boundaries; grade n=50 timing on G0UC floors.
- **FX-L56** Align real and sim on trace time or matched work, not round fraction.

## Felix tripwires (don'ts)

**Selectors (Oort family)**
- **FX-T3** Don't add a `system_util` recency guard or widen oort's slow-speed tail.
- **FX-T6** Don't key oort latency by task type; sync oort sends no evals.
- **FX-T8** Don't expect seeding to align per-round selections; judge S2 by speed class.
- **FX-T17** Oort carry-over decay is structural; felix min-budget seed alone won't fix past-dating.

**Aggregation, ordering, clock**
- **FX-T1** Don't enable `simStaggeredRedispatch` or retune `simRedispatchGapSeconds`.
- **FX-T2** Don't add `mqtt_fetch` to sct; it is pre-selection wait, not transfer.
- **FX-T4** Don't tune the `_sim_recv_min` gate predictor; `exp == sct` exactly.
- **FX-T7** Don't clamp felix's clock jump for past-dating; it was eval reusing stale sct.
- **FX-T14** Don't add prediction-only gates or a `version_at(sct)` relabel; both inert.
- **FX-T15** Don't read wall-clock commit density as sim speed; judge per-round vclock advance.
- **FX-T18** Don't add a scalar fudge for P3's ~1 s `mean_overhead` (wall-capture).
- **FX-T20** Don't gate sim ingest on "committed this cycle"; it livelocked P3.
- **FX-T21** Don't call `ends()` except for a real dispatch; use `all_ends()` to list.
- **FX-T22** Don't skip a trainer after the selector chose it.
- **FX-T33** Don't enable `simInflightCarryover`; exclude stragglers at eligibility instead.
- **FX-T34** Don't mark a stale gated straggler's landing round as a stall.

**Availability**
- **FX-T5** Don't hold all buffered ends out of refl's pool; it over-holds.
- **FX-T9** Don't express "busy" through the UN_AVL list.
- **FX-T10** Don't broadcast availability per tick over MQTT; read the trace.
- **FX-T11** Don't exclude AVL_TRAIN from eval on 2-state traces; the pool empties.
- **FX-T12** Don't zero legacy `trackTrainerAvail` before `simUnavailability` is static (FX-N7).
- **FX-T19** Don't fork withhold/abandon per stack; grade A4 via A4dur.
- **FX-T35** Don't let a wall failsafe drop a sim update before its task timeout.

**One task per version, rounds**
- **FX-T24** Don't add a staleness cutoff to fedbuff; its 1/√(1+s) discount is the baseline.
- **FX-T26** Don't set `aggr_num` or `aggGoal` directly; `agg_goal` overwrites them.
- **FX-T32** Don't fix negative `running_var` by clamping at 0; eval loss explodes.
- **FX-T37** Don't count stale updates toward REFL's K; reference closes on K fresh.

**Harness, compute, datasets**
- **FX-T13** Don't grade fedbuff/felix real while another n=300 real shares the broker.
- **FX-T16** Don't re-chase GPU contention, SEND_TIMEOUT or MQTT drops at cifar n=300.
- **FX-T23** Don't run more stub trainers than node cores.
- **FX-T25** Don't kill FL workers by process name; scope by `FLAME_RUN_TAG`.
- **FX-T27** Don't use 2024 speech lr pairs or Adam on speech; staleness makes it oscillate.
- **FX-T28** Don't read leg health from shared `experiments/`; read the leg's own dirs.
- **FX-T29** Don't write replay input in place or from a daemon thread.
- **FX-T30** Name traces by measured unavailable fraction, not intent.
- **FX-T31** Don't retune sim charge knobs to close a clock rung; use profiles.
- **FX-T36** Don't exceed ~75 cifar / ~25 speech trainers or 7 speech trainings per A40.
- **FX-T38** Don't key liveness on log growth past `run_end`; teardown is silent.
- **FX-T39** Don't stack two eval gates; evals land every 2N rounds.

---

## Built (one line each; details in code, tests, `git log`; IDs kept because code cites them)

**Simulator fidelity**
- **FX-D1** Core sim: sct-ordered drain, one-in-flight residence, faithful oort/refl/feddance mechanics.
- **FX-D4/D8** Cold-start gate (`simColdStartGate`) and uncapped busy hold, freed on abandon/evict.
- **FX-D6** asyncfl sim: no stranded same-cycle re-dispatch; refl starvation wake-up.
- **FX-D9** One task per version: dispatch ledger, no-repeat guard, commit dedup, EV15.
- **FX-D10** Sync rounds advance only on a committed aggregation; all-stale rounds free slots.
- **FX-D17** Sync stacks abandon 90 s after dispatch in real too; refl math on float only.
- **FX-D20** `syncWaitForK`: version advances only on `agg_goal` accepted updates; picks topped up.
- **FX-D21** Empty sync rounds re-dispatch from the ledger; `run_end` marks the stop point.
- **FX-D23** Sim non-compute charges profiled per (dataset, harness, stack) (`profile_felix_charges.py`).
- **FX-D26** Run-4 roots: warm-up optimizer, syn_50 ordering, trace origin at join barrier.
- **FX-D27** Oort-family selection follows REFL fork's `getTopK`.
- **FX-D28** asyncfl sim re-injects due withheld deliveries before starving (`sim_reinject_when_idle`).
- **FX-D34** Sim mirrors unanswered dispatches: withheld picks hold slots; evicted updates ingested.
- **FX-D39** `simInflightCarryover` off for oort/oort_star; stale receipts land one round late.
- **FX-D50** Async sim pops withheld deliveries before next sct (`sim_reinject_lookahead`); gate holds to timeout.
- **FX-D66** Oort sync never awaits an end already on the cleanup queue.
- **FX-D69** FX-D50 lookahead also counts earliest in-flight expected completion (`_sim_next_event_ts`).
- **FX-D70** Oort sync stack reinjects withheld deliveries with the FX-D50 lookahead.
- **FX-D71** Felix D.1 eviction skips a buffered update only once complete (`sct <= now`).
- **FX-D74** C13: sim adds measured pre + post-train time to duration (`sim_charge_trainer_overhead`).
- **FX-D75** D.1 eviction also frees a send-gated buffered update (real never receives it).
- **FX-D76** Late withheld deliveries of evicted/abandoned ends never retake freed slots.
- **FX-D88** Async sim withholds send-gated buffer heads first (`sim_withhold_before_gate`).
- **FX-D89** Async sim abandon deadline is a clock event (`sim_abandon_wakes`).
- **FX-D90** Send-gated held slot frees at delivery commit, not reinjection (`sim_hold_delivering_slot`).

**Availability**
- **FX-D2** Unavailability v1 for all six: send-gate/deliver-late, two ledgers, evict, starvation advance.
- **FX-D5** Real withheld updates are delivered; `mobiperf_3st` launchable.
- **FX-D12** Only withhold/evict/abandon free slots; dispatch consumes the end's earlier receipt.
- **FX-D18** Traces at named level from t=0, 149 h, C loader; `effective_unavailability()`.
- **FX-D55** Real send gate rereads the trace at send, polls 0.1 s (`FLAME_SEND_GATE_REFRESH`).

**Real path and wire**
- **FX-D16** Real asyncfl ingests on arrival via `drain_ready`; no settle sleep.
- **FX-D24** Real async frees an ingested slot at SEND (`release_recvd_at_send`).
- **FX-D35** Real `recv_fifo` releases reader slots when its streamer exits.
- **FX-D40** Aggregator math libs get a quarter of pinned cores (`FLAME_AGG_MATH_THREADS`).
- **FX-D42** asyncfl caches dispatch payload per (version, task, stamp).
- **FX-D44** Aggregators tune glibc malloc to keep freed buffers (`FLAME_MALLOC_TUNE`).
- **FX-D59** Trainers keep glibc malloc defaults; host RAM no longer fills.
- **FX-D60** Real asyncfl idle wait compares stamps on their own clock.
- **FX-D64** Aggregators decode the test set once at load (`FLAME_EVAL_PRELOAD`).
- **FX-D77** `FLAME_PYSPY` profiles every role; `profile_report.py` ranks; tier PROF.
- **FX-D78** Flat tensor weight codec, zero-copy out-of-band frames (`FLAME_WEIGHT_CODEC`).
- **FX-D79** Aggregation on CPU, GPU only for eval/oracle (`FLAME_AGG_MODEL_DEVICE`); eval off ingest.
- **FX-D80** MQTT sends await PUBCOMP asynchronously; whole messages FIFO; one chunk copy.
- **FX-D81** No log line formats a whole payload (`test_no_eager_payload_logs.py`).
- **FX-D82** GPU cache release runs inside real's D padding, outside sim overhead.
- **FX-D83** Waits wake on deliveries (`Channel.wait_arrival`); no polling sleeps.
- **FX-D84** Zero-copy MQTT client `paho_fast.FastClient` (`FLAME_MQTT_FAST`).
- **FX-D85** QoS-2 chunks never re-published; partial message dropped only on new seqno 0.
- **FX-D86** Teardown flushes sends; JOIN/LEAVE ordered after in-assembly chunks.
- **FX-D87** Trainer atexit stops and joins availability/heartbeat threads.

**Training correctness**
- **FX-D38** REFL BN stats average fresh updates only; `clamp_running_var` guard.
- **FX-D41** Trainer `evaluate()` resets utility and runs under `no_grad`.
- **FX-D43** Trainer frees GPU cache after the util_cf telemetry forward.
- **FX-D53** REFL fills K with fresh updates only; stale still aggregate (`refl_fresh_k`).
- **FX-D61** FedBuff/Felix BN = mean of absolute stats (`bn_absolute_mean`).
- **FX-D63** Aggregators evaluate every `evalEveryNRounds` round exactly.
- **FX-D72** `--trainer-hp` overrides by_baseline `trainer.hyperparameters` keys.
- **FX-D73** Cifar fedbuff SGD 0.04 b32 × server 1.0 (r1000 53% vs 32%).

**Harness, checkers, datasets**
- **FX-D7** Trainer availability thread stops at EOT or shutdown.
- **FX-D11** Streaming + oracle harness (P7/P7o) with offline replay.
- **FX-D13** Event checker EV0-EV19, injected bugs P11a-c, isolated parallel pool.
- **FX-D14** Dataset switch (`fl_data.py`, `datasets.yaml`); speech on the launcher.
- **FX-D15** No cold start in timed tasks: startup warm-up on CPU and GPU.
- **FX-D19** NVML polled only with `FLAME_STAT_THREADS=1`; GPU joins under 100 s.
- **FX-D22** Fail fast on fatal lines (`fail_fast.py`); pool aborts with `ABORT.txt`.
- **FX-D25** Checker times both sides from first train selection.
- **FX-D29** SUMMARY lists UN_AVL share at selection; real bank hashes the trace store.
- **FX-D30** Grader reads the run's own `max_experiment_runtime_s`.
- **FX-D31** P11b `order_by_sct` runs on fedbuff; CAUGHT on both datasets.
- **FX-D32** Run-6 confirms: unaware fedbuff picks UN_AVL; `[TRAINER_HP]` matches config.
- **FX-D33** A7 graded at its instant; K2/K3 stall-free rounds; K3s grades stall rate.
- **FX-D36** Timeout stall = an episode; K4 drops stall time.
- **FX-D37** `parity_check.py --control/--floors` feed `floor_gated_tol` (never tightens).
- **FX-D45** `StallWatch` S1/S2 rules kill only stalled legs (`STALLED.txt`).
- **FX-D46** Speech `device_time_scale: 5` drives delay factor and task timeout.
- **FX-D47** Streaming `stream_schedule.py`, EV19, `stream_growth` DIAG.
- **FX-D48** Checker stamps sync stalls by cause (abandon, gated update).
- **FX-D49** GPU slot CPUs sized per baseline from measured p95; `CPU_SAT` flag.
- **FX-D51** Checker fixes: EV16 sync close, A3 by trace time, S3/4 total picks.
- **FX-D52** `--regrade` reruns event checker; G0UC floors G0U/G0T; `G0U_RUNTIME_X`.
- **FX-D54** Accuracy tooling: `accuracy_table.py`, `fl_lr_check.py`, tier G1A.
- **FX-D56** `--scale-smoke`, DONE lines print GPU peak, `GPU_TIGHT` above 85%.
- **FX-D57** GC pause callback binds its clock at def; no exit traceback.
- **FX-D58** Per-GPU training cap via `flock` slots (`gpu_train_slots`, speech 7).
- **FX-D62** `StallWatch` disarms at `run_end`; pool legs skip plot analysis.
- **FX-D65** Stall-killed legs skip fail-fast; `merge()` skips unstarted phases.
- **FX-D67** StallWatch S1 counts `abandon_timeout` as progress.
- **FX-D68** EV1 needs two train rounds, not 2 × agg_goal.
- **FX-D91** Checker keeps zero per-round advances (same-vclock commits).

## Open questions (operator)

- Speech feddance is round-bound at 90 min: lengthen its window? (FX-N74)
