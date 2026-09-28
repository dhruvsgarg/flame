# Felix readiness — backprop FL (async_cifar10, google_speech)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** Its doc rules, operating rules (R1-R13),
  shared lessons (L) and tripwires (T) apply here and are not repeated. This doc holds only what is specific
  to Felix: weight-aggregating backprop FL, round/`agg_goal` progress axis, Oort-family selectors.
- **CURRENT FOCUS.** Goal: Felix feature-complete (syn_0 + unavailability, both datasets) and paper
  experiments running sim-only.
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ each one's `*_oracle` arm for the
  streaming experiment, FX-N13). Out of scope: `fedavg`, `oracle`.
- **Traces (both datasets, both papers):** `syn_0`, `syn_20`, `syn_50` (synthetic) and `mobiperf_3st` (the
  real-world 3-state trace). `syn_10` is the pre-2026-09-27 "syn_20" (10.8%, used by the EuroSys'26 runs; FX-D18).
  Levels (300 trainers, instantaneous TRAIN/EVAL/UN %) are flat from 1 min to 3 h: syn_10 89/0/11, syn_20 ~80/0/20,
  syn_50 ~50/0/50, mobiperf_2st 10/0/90, 3st_50 10/25/65, 3st_75 10/10/80; mobiperf's trainable share rises to 18-22%
  by 6-24 h (diurnal).
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built features. Shared work is `S#` in the parent.
- **Reference (read only for detail):** rung catalog and derivations → [PARITY.md](../async_cifar10/PARITY.md)
  §1-§5 · checker internals → [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · availability design → [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · streaming
  experiment → [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md) · identity →
  [BASELINES.md](BASELINES.md) (Felix section).

---

## Status grid

✅ parity at HEAD · 🟡 last good on old code (stale) · ⚠ open failures · ⬚ not started. Score = enforced
passing / total, with run length.

**async_cifar10** (n=300, α=0.1)

| baseline | real | sim syn_0 | sim syn_20 / syn_50 | sim mobiperf_3st |
|---|---|---|---|---|
| felix | ✅ | 🟡 46/46 (3h) — the §S.pacer fix landed after it | ⚠ 62/62 (syn_50) is INVALID: real dropped withheld updates (FX-D5) | ⬚ |
| oort | ✅ | 🟡⚠ 42/46 (1.5h) — Sd: pacer input signal | ⚠ A2 / K3b / P3 / throughput | ⬚ |
| oort_star | ✅ | ⬚ | ⬚ | ⬚ |
| refl | ✅ | 🟡 44/46 (3h) | ⬚ | ⬚ |
| feddance | ✅ | 🟡 43/44 (3h) — C2 loss only | ⚠ A2 / U5 | ⬚ |
| fedbuff | ✅ | ⬚ | ⚠ run spoiled by a concurrent run | ⬚ |

Every real leg under unavailability before FX-D5 dropped its withheld updates, so all syn_20/50 cells
are re-run items. Everything is 🟡 because PRs #72-#85 rewrote code these runs depended on: `async_oort.py` (+ new
`async_base.py`), `fedbuff.py`, both `top_aggregator.py`, `syncfl/trainer.py`, `channel.py`, the checker.
Stored grades: `async_cifar10/experiments/parity_{felix,oort,refl}_20260624_{5400,3h}.json`,
`parity_feddance_20260623_3h.json` (run dirs are named inside each JSON).

**Harness (CPU):** last T4 at a68fe7cc8: cifar `experiments/pool_20260926_223155_T4` (63 min) and speech
`pool_20260926_223125_T4` (79 min, wash). EV-fail non-control pairs: cifar 5/44 (was 11), speech 9/44 (was 30). All are
fedbuff EV14 (FX-N15), speech P7/P7o (FX-N30) or P3 feddance real/refl sim EV12 on both datasets (FX-N31). P11a/c CAUGHT on
both; P11b CAUGHT on cifar, MISSED on speech (FX-N32). One run dir per leg, all `run_<ts>_<phase>_<name>`.

**google_speech** (n=100, α=0.1; profile `_metadata/datasets.yaml`) — on the launcher (`--dataset
google_speech`), real + sim. Smokes only: felix/oort CPU pairs EV PASS (`pool_smoke_ds`), felix GPU pair EV
PASS, parity 0.918 at 60s (`pool_smoke_gsG1`). No graded run yet: all six ⬚.

---

## Active build — fast parallel harness (FX-N22)

**Why.** Campaign 3 ran ~9h serially at n=60-120, with the real leg re-run on every pair. The harness exists
to catch logic bugs (event invariants, injected bugs); distribution parity is decided on GPU.

**Isolation contract: a parallel leg must behave as if it ran alone.** Each slot gets:
- disjoint physical cores via `taskset`. The launcher pins the aggregator (`FLAME_AGG_CORES`: 2 at n≤30,
  8 for GPU legs) and the trainers inside that affinity;
- a private mosquitto (`FLAME_MQTT_BROKER`). MQTT client ids are fixed task ids, so two runs on one broker
  kick each other off;
- a run tag (`FLAME_RUN_TAG`) scoping every kill and clean-slate sweep, and an exact run-dir handoff
  (`FLAME_RUN_DIR_FILE`);
- GPU legs: every healthy GPU (volatile-ECC GPUs are skipped), CPUs = 8 + 0.4·n. cifar n=300 therefore
  takes the whole node; speech n=100 takes 48 CPUs and runs beside CPU slots.

**Test shapes** (`harness_pool.SHAPES`): small cohorts with varied c and aggGoal. aggGoal goes through the
launcher's `agg_goal`, which is also the sync K (FX-L36).

| shape | n | aggGoal | c | runtime | trace scale |
|---|---|---|---|---|---|
| syn_0 | 12 | 3 | 5 | 180s | – |
| syn_0b | 15 | 2 | 8 | 180s | – |
| syn_20 | 15 | 3 | 6 | 240s | 4 |
| syn_50 | 15 | 3 | 6 | 1200s | 4 (FX-L43) |
| mobiperf_3st | 45 | 2 | 4 | 240s | 4 (FX-L34) |

google_speech CPU sims get a 2× wall ceiling: with 29 MB updates on 2 aggregator cores they are
aggregator-bound (EV12 still grades the vclock budget).

**Tiers** (`lib/python/examples/scripts/harness_pool.py --tier A[,B] --datasets cifar10|google_speech|all`; runs
from any cwd; `EXAMPLE_OF` maps each dataset to the example that runs it; roots and per-node state in
`lib/python/examples/experiments/`). Every run starts with a
gate: pytest collection plus one felix smoke pair per dataset; it aborts in ~4 min.

| tier | jobs | est. wall, one node (cpt 1 / 0.5) |
|---|---|---|
| T1 | sim-only, affected × {syn_0, syn_50}, 120s, EV only | 6 / 3 min |
| T2 | sim-only × 4 shapes vs the banked real legs + P11a-c | 38 / 22 min (both datasets) |
| T3 | real+sim pairs × 4 shapes; refreshes the bank | 82 / 47 min (both datasets) |
| T4 | campaign P1-P11 (`harness_campaign.sh` = `--tier T4 --pytest`) | 70 / 40 min per dataset |
| G0 / G0C | GPU 30 min screen: all six × syn_0 + syn_20, n 100/50 (FX-N34) / a second real leg per syn_0 cell (R7) | T4+G0+G0C ~6.9h per 8-GPU node, `--shard i/2` |
| G1 / G2 | GPU 90 min: felix+fedbuff / the other four, reference config | ~2.5h per pair (cifar), in parallel with CPU (speech) |

`--changed <ref>` picks the affected baselines; `--shard i/N` splits across nodes; `--exclude-phases` drops exact phase
ids. Whole-node jobs run last. Between leg starts (never mid-leg) the pool skips GPUs and cores another process is
using and checks live free memory, so it scales down under foreign load and back up when it clears (`LOAD` lines in
`pool.log`); `--gpu-ids` pins the usable GPUs.

**Several nodes.** Each node runs its shard; then pull every node's pool and run dirs to the analysis node (jayne). Run
dirs live in `async_cifar10/experiments/` for both datasets, and the P7/P7o checkpoints are FX-N13 replay input. Per
node `N` (wash, kaylee):
```
R=/home/dgarg39/flame/lib/python/examples; N=wash
POOL=$(ssh $N "ls -dt $R/experiments/pool_*_T4* | head -1 | xargs basename")   # check the name
rsync -av $N:$R/experiments/$POOL $R/experiments/
ssh $N "cat $R/experiments/$POOL/*/*/runs/*/legs.txt" | sort -u > /tmp/legs_$POOL.txt
rsync -av --files-from=/tmp/legs_$POOL.txt -r $N:/ /
bash $R/scripts/harness_report.sh $R/experiments/$POOL   # re-runs the P7 replay here
```

**Tasks**
- P1-P5 · done. Smokes: `pool_smoke_T3/T2/G1` (cifar), `pool_smoke_ds` (both datasets, 8 legs in parallel),
  `pool_smoke_gsG1` (speech GPU), `pool_smoke_gate`; SIGINT tore down 4 slots in 11s. Their SUMMARY files remain;
  21 of their 30 run dirs were pruned on jayne (2026-09-27). Tests:
  `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`, `tests/launch/test_debug_run_dataset_profile.py`.
- P6 · isolation control · todo (operator, ~15 min): `$P --tier ISO --max-parallel 1 --output-dir
  $E/iso_solo`, then `$P --tier ISO_FILL --cpus-per-trainer 0.5 --output-dir $E/iso_packed`, then
  `examples/scripts/harness_iso_compare.py $E/iso_solo $E/iso_packed` ($P/$E: see FX-N20).
  EQUIVALENT → 0.5 becomes the default. Measured: a small slot averages 1-2 cores (p95 4-6) with its aggregator.
- P7 · done: T4 on the pool, gate + `--pytest`; `harness_campaign.sh` is now a shim.
- P8 · in-process fast sim: fake trainer replies, no MQTT and no processes (~100×). It is the
  paper-experiment engine (parent S6). After P6.

## Next steps (persistent queue — top item is next)

**Test levels.** pytest (P0/T0) = in-process unit/integration tests, no FL processes, ~4 min. CPU tests =
pool tiers T1-T4: real aggregator + trainer processes over MQTT on CPU (stub/tiny_cpu data, n 12-30), graded
by the event checker and the parity battery. GPU tests = G0 (30 min screen at n 100/50, + G0C real replicates) and
G1/G2 (the production path at the reference n, cifar 300 / speech 100, 90 min). "CPU+GPU" = one pool run with both.

**Unblock map.** FX-N20 T4 re-run (confirms FX-N31/N33) + FX-N34 G0 screen → FX-N4 (G1) + FX-N5 (G2) on the cells G0
passes, + FX-N7 → FX-N6 + FX-N9 (GPU unavailability)
→ FX-N11 (speech GPU parity) → FX-N12. FX-N10 grades on FX-N20 + speech G1. FX-N13 design can start any time.

- **FX-N20 `[C][S]` · 6h two-node run: run-1 fixes + FX-N36 + FX-N37 · todo (operator; local P2 check first).** Run 1
  (`pool_20260927_060727_T4_G0_G0C` kaylee, `pool_20260927_060911_T4_G0_G0C` wash) is not gradable: GPU real legs were
  watchdog-killed during 10-min joins (FX-N36), and wash lost 26 legs to external SIGKILL waves. Fixed in tree: felix
  syn_50 0 commits (`AsyncOortSelector.requester` unset when an UN_AVL report precedes the first select); feddance rx task
  died on trainer leave (no `_cleanup_removed_ends`); speech G0 CUDA OOM; oort_star chose UN_AVL ends (EV9); felix/fedbuff
  sim double dispatch after a late commit (EV10/EV11, FX-L32); EV10 counted withhold releases; FX-N36; FX-N37; real
  felix/fedbuff read every reply one dispatch late after a late withheld update (FX-L27; P2 real queue_wait max 17-90s).
  Local checks `experiments/pool_local_fxn37{,b,c}` (jayne) predate the last two FX-N37 fixes, so first:
  ```
  P="conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/harness_pool.py"
  $P --tier T4 --datasets cifar10 --phases 'P2' --deadline-h 1 --output-dir lib/python/examples/experiments/pool_local_fxn37d   # jayne, ~30 min
  ```
  *Local predictions:* P2 real queue_wait max < 2s on every baseline (fxn37c: feddance 149s, fedbuff 35s, felix 60s);
  feddance never logs `in-flight total` > 3 (was 6 in 136 rounds); `[LATE_UPDATE]` lines in felix/fedbuff real; EV real
  PASS on all six; sim unchanged (FX-N38 fails remain). Then both nodes check out `dg/pending_fl_baselines` and pull:
  ```
  $P --tier T4,G0 --datasets cifar10 --exclude-phases 'G0_syn_0' --pytest --deadline-h 6              # kaylee, ~270 min
  $P --tier T4,G0 --datasets google_speech --exclude-phases 'gs_G0_syn_0' --deadline-h 6          # shepherd, ~246 min
  ```
  Then "Several nodes" (Active build). *Predictions:* every G0 real leg passes its join barrier ≤ 3 min after its first
  trainer starts, max `[WARMUP]` ≤ 5s (FX-N36; refuted if any leg still takes ≥ 5 min); no `requester`,
  `_cleanup_removed_ends` or CUDA OOM traceback; every GPU real leg logs its budget stop (EV0/EV12); oort_star EV9 green;
  EV7 green on every sync leg (no commit < `agg_goal`, FX-N37); P3 oort real/sim commit counts within 2× (was 2 vs 84;
  EV1 may fail there: unaware oort at mobiperf really commits ~3 in 240s); real queue_wait max < 2s on every cifar
  leg; P11a/c CAUGHT; known: P2 felix/fedbuff sim EV10/EV11/EV16 (FX-N38), fedbuff EV14 (FX-N15), speech P7/P7o (FX-N30).
  *On return:* read both `SUMMARY.txt`; an item per unpredicted miss. *Exit:* that holds, or each new miss has an item.
- **FX-N36 `[C][S]` · n=100 GPU legs took 10 min to join · fix in tree, confirm in FX-N20.** Root: `Role.mc =
  MetricCollector()` (import time, every flame process) polled NVML on all 8 GPUs at 1 Hz (~6.4 ms a sweep); 200
  trainers ≈ 1.3 s/s of driver-lock time, so CUDA init starved. Run 1 signature: init GIL-held for ~3.5 s × spawn
  index node-wide, then all `.to(cuda)` released together; bare CUDA probes (no flame import) never stalled. Poll threads
  are now opt-in (`FLAME_STAT_THREADS=1`; nothing read those stats). Test: `tests/test_metric_collector_threads.py`.
  Local (`pool_local_fxn37` G0 syn_20 oort real, one leg): 92/92 joined 99s after the first trainer, max `[WARMUP]`
  0.56s. *Exit:* FX-N20 join prediction holds with two GPU legs per node.
- **FX-N37 `[C]` · Round progress vs timeouts, all sync baselines · fix in tree, confirm in FX-N20.** Rule (operator):
  a version advances only when `agg_goal` updates aggregate; each trainer's 90s timeout runs from its own dispatch and
  is applied at dispatch time; no mode frees a slot on knowledge real lacks. Was: sync real ended a round at the latest
  pick's timeout and committed partial (P3 oort 2 commits in 240s); sim freed withheld picks at `sct` and committed
  partial (21 of 84 rounds). Now (`syncWaitForK`, on for oort/oort_star/refl/feddance in `baselines.yaml`): a version
  collects K accepted updates across passes; real's recv stops at the first pending timeout; distribute abandons it and
  tops up only the missing slots at the same version (`num_to_select`); sim keeps a withheld pick's slot to dispatch+90
  and jumps the vclock to the next timeout/delivery; a replied pick is never abandoned, a re-dispatch re-arms its
  timeout, and a pick with an undelivered withheld update is not re-selectable (its reinject overwrote the fresh reply
  in the per-end reorder buffer); an end accepted at v (a late update counts) is not re-pickable at v; recv awaits every
  unaccepted in-flight pick, incl. an earlier version's leftover (feddance stranded both, `pool_local_fxn37c`). Local (`pool_local_fxn37`): EV7 under-K = 0 on every sync leg; G0 syn_20 oort 47 real vs 49 sim commits; P3
  oort 3 vs 3 (was 2 vs 84). Also fixed: oort's utility path re-sent every in-flight end; syncfl
  stamped `PROP_ROUND_START_TIME` with the builtin `round`. asyncfl already followed the rule except sim's send-gate
  freeing an unaware fedbuff pick at `sct` (open). Tests: `tests/availability/test_sync_wait_k.py`,
  `test_sync_repeat_round.py`, EV7 in `test_event_invariants.py`. *Exit:* FX-N20 EV7/P3 predictions hold.
- **FX-N38 `[C]` · P2 syn_50 async sims break one-in-flight and ordering · todo.** Predates FX-N37 (run 1 had more):
  `pool_local_fxn37` felix sim EV10 11 redispatch-while-outstanding, EV11 4.5% past-dated, EV16 2 delivery_ts ≠ trace
  (450 vs 149.5: the evict-time estimate, FX-L11); fedbuff sim EV11 3.3%. Real legs pass. Lead: felix proactive evict
  registers delivery before the update completes, and a later re-dispatch overlaps it; `withheld_held_ends` frees an
  end at `dts == now`, before its reinject, so a re-pick collides in the per-end reorder buffer (fixed for sync only). *Exit:* P2 felix/fedbuff sim
  EV10/EV11/EV16 green.
- **FX-N31 `[C][S]` · Sync round that does not advance livelocks · fix in tree, confirm in FX-N20.** P3 mobiperf, both datasets:
  refl sim (withheld round) stuck at vclock 74s; feddance real (both abandoned at 90s) stuck at round 7. When a round
  commits nothing (FX-D10), the refl_oort/feddance round cache re-returned the freed trainers, and they discarded the
  repeat (`task_discard`). Fix: a dispatch never takes the round cache; the FX-D9 ledger moves to the sync base and feeds
  the sync selectors and `num_eligible`; the oort pacer runs once per round. A trace-starved sync sim then jumps its vclock
  to budget with no selection, so a new `run_end` event records the stop vclock for EV12. Tests:
  `tests/selector/test_sync_repeat_round.py`, `test_run_exit_artifacts.py`, EV12 in `test_event_invariants.py`.
  *Exit:* FX-N20 predictions hold → Built.
- **FX-N33 `[C][S]` · Aggregator aborts at interpreter exit · todo (root open).** After a clean channel leave:
  `terminate called without an active exception` → `Fatal Python error: Aborted` (a C++ thread destroyed while
  joinable; no Python frame). 6 runs in the 09-26 pools, and again in `pool_smoke_final` G0 syn_20 refl sim, which wrote
  no checkpoint, so checkpointing is not the root. The checkpoint writer is now non-daemon with an atomic rename (a
  killed writer truncated one `.pt`; `tests/mode/test_run_exit_artifacts.py`), and crash_lines counts the abort. Data
  is intact (post-leave). Candidates: torch/CUDA or MQTT native threads at teardown. *Exit:* root named, abort gone.
- **FX-N32 `[S]` · EV16's injected bug is weakly exercised · todo.** Every sim leg withholds 0-1 updates (felix evicts, stub
  compute is short, and harness syn_50 was only ~10% unavailable before FX-D18), so P11b is caught only by 1-2 chance commits (cifar CAUGHT, speech MISSED). Needs a mid-flight-drop
  shape (parent S1 scenario library): compute long relative to AVL spans, on a withholding baseline. *Exit:* P11b sim
  shows ≥10 withheld commits and is CAUGHT on both datasets.
- **FX-N26 `[C][S]` · Stub real per-trainer speed runs above sim · todo.** `trainer_speed_identity` fails on 22 cifar and
  ~30 speech stub legs, always with real > sim. The gap is +1.0-1.5s on sync oort/oort_star/feddance (most trainers, e.g. 2.19s vs 1.0s),
  and +0.1-1.3s on some felix/fedbuff trainers. Cifar tiny_cpu legs are 0 out. Sim equals D on the 0.25s grid. Find which endpoint
  real adds (L6). *Exit:* root named. Then fix it, or show it is stub-only and not on GPU (FX-N4 prediction).
- **FX-N30 `[S]` · Speech tiny_cpu is too heavy for CPU slots · todo.** P7/P7o sims run at sim_rate 0.2-0.5 and are
  killed at 63-141s of 180 (EV0/EV12). In P7o real, the oracle's `select` blocks the MQTT thread for 20-84s on the 2 aggregator
  cores (felix/fedbuff queue_wait p99 111-147s), and real per-trainer speed is 2-4× sim. Options: shrink the speech tiny_cpu
  model/data, give oracle legs more aggregator cores, or run P7/P7o speech on GPU/P8. Speech replay arms get 1-9
  selections each (`gs_P7_figures`); P7o real has none. *Exit:* speech P7/P7o EV green; FX-N13 speech arms usable.
- **FX-N22 · Fast parallel harness (Active build) · wip: P6 isolation control next.** *Exit:* P6
  EQUIVALENT at cpt ≤ 0.5, and T2 for both datasets under 25 min on one node.
- **FX-N34 `[C][S]` · G0 GPU screen, all six × both datasets × syn_0/syn_20 · wip: re-run with FX-N20.** 30 min,
  cohorts scaled with the reference c/n and aggGoal/c (cifar n=100 c=10 aggGoal 3, 3 GPUs; speech n=50 c=15 aggGoal 5,
  4 GPUs), keeping the reference trainers per GPU. Speech GPU memory binds: idle trainers kept ~3 GB of cached
  activations and 13 per A40 OOM'd; `train()` now frees an idle cache above 1 GB.
  syn_20 is stationary, so 30 min sees 17-19% unavailable (GPU syn_50 is FX-N9). G0C adds a
  second real leg per syn_0 cell for the real↔real floor (R7). Smoke `pool_smoke_G0` (felix): EV all PASS, cifar
  parity 1.0. *Predictions:* EV green on every leg except cifar fedbuff EV14 (FX-N15); syn_20 legs withhold and deliver
  (EV16 withheld > 0); a cell's real↔real (G0C) gap bounds which real↔sim residuals are noise. *Exit:* per cell (baseline × dataset × trace) INV/EXACT green → that cell goes to
  G1/G2 at reference n; each red cell gets an item. Parity sign-off still needs the reference n, controls (R7) and 3h runs.
- **FX-N35 `[C][S]` · Report each leg's effective unavailability · todo.** Traces are fixed (FX-D18); the pool still doesn't
  print the fraction a leg actually saw. Also re-key or drop `_real_bank.tsv` rows banked under the old syn_20 (per node)
  before any T2. *Exit:* SUMMARY carries each leg's effective fraction; no bank row predates FX-D18 for syn_20.
- **FX-N4 · First GPU block: felix + fedbuff, syn_0, 90 min · blocked: FX-N20, FX-N34.** cifar:
  `$P --tier G1 --datasets cifar10` (whole node, both pairs in sequence, ~5h; or `--shard 1/2`, `2/2` across two
  nodes, ~2.5h). speech: `$P --tier G1 --datasets google_speech` (~2.5h, can share the node with its T4). jayne
  runs on 7 GPUs (GPU 1 ECC); each pair stays on one node and one layout (L18). *Predictions:* EV green both
  legs; felix real queue_wait p99 < 1s; `trainer_speed_identity` inside tolerance now the cold start is gone.
  *Exit:* INV/EXACT green, convergence inside the replicate band, DIST residuals common-mode.
- **FX-N10 · google_speech on the launcher · wip: code + profile + splits landed, graded runs next.** Reference
  = 2024 SoCC n=100 (the 2024 n=300 configs are malformed past trainer 100). The model, data and optimizer
  switch lives in `fl_data.py`. Splits come from `_metadata/scripts/import_speech_splits_2024.py` (α 0.1/1/10,
  n=100). The profile is `datasets.yaml`. Trainer Adam lr 0.000195: the 2024 "_oort" value 0.04 stays at
  chance in a centralized check. *Next:* speech T4 fix FX-N30; GPU lr check in the first speech G1 (0.000195 vs
  0.001 via `--trainer-hp learningRate=…`); target accuracy + stop rule (2024: 20 evals ≥ 60%); then S4 removes
  the 2024 JSON/scripts and the import script. Data: `<data_root>/google_speech/SpeechCommands/…` via
  `datasets.yaml` `data_roots` = `/coc/scratch/dgarg/fl_datasets` (verified complete by the pool gate). *Exit:* all six real+sim
  graded on speech (T4 CPU + G1/G2 GPU).
- **FX-N19 · asyncfl sim serializes more than real at small n · todo (after the FX-N20 re-run).** From cifar T4,
  felix/fedbuff: sim per-commit advance is 13-32% above real (K3b: P1 felix 2.15 vs 1.88s, P1b fedbuff 0.88 vs 0.67s).
  Overlap is lower in sim (K4: P1b felix 6.0 vs 7.2, P2 4.4 vs 5.1), and U6 visibility lag is 0.2-0.4s in sim vs 0.006s in real. P3 mobiperf
  goes the other way: real overlap 1.0-1.25 vs sim 2.9-3.8. Don't widen (T7, R11). *Exit:* root-caused, or
  graded against a real↔real floor (L12).
- **FX-N5 · syn_0 GPU block (G2): oort, oort_star, refl, feddance · blocked: FX-N4.** Same protocol.
  oort's open root: per-round `relative_change` of the exploited utility, binned by quartile, in both modes
  (don't touch the pacer). refl: confirm at 3h.
- **FX-N15 · fedbuff training diverges; now reproduces on the CPU harness · todo.** Stored Jul-2 sim (CPU fallback, FX-L29):
  NaN from round 600 at syn_50. Cifar T4 stub legs: sim test-loss rises (P1b syn_0 2.55→3.31, P2 syn_50 2.63→3.03);
  real is NaN at the first eval (round 50) on P1b, P7 and P8 (EV14; `pool_20260926_223155_T4`). Leads: the real/sim split
  on one shape, and the hardcoded cifar server lr 40.9 (`fedbuff.py:_scale_add_agg_weights_pytorch`) applied to a 2-3
  update mean; diff against the reference FedBuff first (L7, T5). Max staleness is 4, and felix on the same shapes stays flat
  at 2.30. The same leg at aggGoal 10 stayed at 2.33. So staleness is not the driver: suspect the fedbuff server step on a
  small buffer (aggGoal 2-3). Cifar only: speech fedbuff (Adam trainer) stays flat at chance (3.57→3.60) on the stub. Operator: no staleness cutoff (FX-T24). *Exit:* root in the fedbuff optimizer
  named, and fixed if it is a port bug (L7) or recorded as baseline behaviour.
- **FX-N13 · Streaming motivation experiment, both datasets · design after FX-N22 + FX-N10.** Show (a)
  per-trainer statistical utility changes as data streams in, (b) an unaware aggregator mis-selects, (c) one
  that tracks utility but mis-estimates it still mis-selects. Pipeline built: campaign P7/P7o, oracle replay
  `scripts/oracle_misselection.py` + `scripts/felix_streaming_figures.py`. Cifar T4 Spearman, real/sim: felix 0.62/0.47
  → +oracle 0.99/0.97 (regret 0.050/0.056 → 0.025/0.025); oort −0.31/−0.30 → 0.92/0.92. Sim and real agree in
  sign on every arm (`pool_20260926_223155_T4/P7_figures`). Speech (`gs_P7_figures`, tiny arms, FX-N30): Spearman ≈0
  without the oracle, 0.93-0.99 with it. Next: move the arms onto the launcher (S3), calibrate the horizon, operator's design, sweep.
- **FX-N2 · Parent S2 (parity pipeline) for async_cifar10 · todo.** *Exit:* the stored Jun 23-24 pairs
  re-grade through it to within the floor, or each difference is explained.
- **FX-N6 · Unavailability design re-audit · todo.** Keep v1 semantics; check against the fwdllm invariants,
  logical-budget grading and the drain primitives. *Exit:* one audit table (item · keep/change · evidence).
- **FX-N7 · Remove legacy `trackTrainerAvail` (oort, oort_star, refl) · blocked: FX-N20.** oort/oort_star
  EV are green on every cifar T4 leg; refl's fails were fixed by FX-D10/D17. *Exit:* refl EV green; then delete the dead
  check and legacy branch (S4).
- **FX-N8 · Concurrent-run confound on fedbuff · likely closed by FX-N22.** Two runs on one broker
  collide by construction: MQTT client ids are fixed task ids. *Exit:* P6 EQUIVALENT.
- **FX-N9 · GPU unavailability: syn_20 → syn_50 → mobiperf_3st, all six · blocked: FX-N5, FX-N6.**
  *Exit:* A1-A8/K11 + the syn_0 ladder green; clean self-stop; withheld updates delivered; AVL_EVAL and the
  empty-pool cleanup exercised live on mobiperf_3st.
- **FX-N11 · google_speech GPU parity · blocked: FX-N9, FX-N10.** Reuse the FX-N4/5/9 protocol.
- **FX-N12 · Felix paper experiments, sim-only · blocked: FX-N11, FX-N22 P8.** List: open question below.

---

## Felix lessons (dos)

**Selectors (Oort family)**
- **FX-L1** Oort's pacer has two branches: a flat trend (`|Δ| ≤ 0.1·last`) raises `round_threshold`, a
  sharp one (`≥ 5·last`) lowers it. It fires on TRAIN only. Log `round_threshold` every round.
- **FX-L2** The UCB temporal term keys on the round of the last RECEIPT (`PROP_LAST_RETURNED_ROUND`),
  initialised at registration — never None, never the dispatch round.
- **FX-L3** Once a faithful controller still diverges, instrument its INPUT by quartile; stop touching the
  controller.
- **FX-L4** Async and sync selector knobs differ: Oort paper defaults are sync values. Per-round terms
  fire 2-3× more often in async. Felix uses `exploration_decay` 0.999.
- **FX-L5** Real records a stale-but-returned trainer's speed and utility too; skipping it makes Oort treat
  slow trainers as unexplored and re-pick them forever.

**Aggregation, ordering, clock**
- **FX-L6** Async (felix): drain each in-flight end's queue directly (`simSctOrderedDrain`) and hold busy
  slots until commit (`_sim_hold_busy_slots`). This one root cleared K3b/K2/U3/U6/K8/U2 together.
- **FX-L7** Sync oort over-selects ×1.3: a prior-round straggler with `sct` past the pinned round start is
  held in `selected_ends` and commits later (`simInflightCarryover`).
- **FX-L8** refl/oort: a trainer still computing (`vclock < sct`) is kept out of the eligible pool via the
  unavailable path, not `selected_ends`.
- **FX-L9** For a strict-barrier baseline (feddance, fedavg), real visibility lag is anchored on the barrier
  (`max_dur − dur_i`); streaming oort/refl stay per-message.
- **FX-L31** A committed end is RECVD so it leaves RECV; a buffered one reset to NONE then committed stays a
  phantom that blocks the starvation wake-up (P3 felix/fedbuff 6-min livelock).
- **FX-L32** A dispatch consumes the end's earlier receipt (eval reply, commit) and its RECVD state; else cleanup or
  recv frees the in-flight trainer and it is re-dispatched (FX-D16 22s tail; late withheld commits, FX-N20).
- **FX-L10** Split eval from train in any check that reads `agg_rounds`. Each eval gets its own `sct`,
  never the last train `sct`.
- **FX-L30** Anything replaying trainer data (oracle, replay) reads the trainers' stream clock: vclock in sim,
  wall since the broadcast `AGG_START_TS` in real — never 0.

**Availability**
- **FX-L11** Two separate axes per baseline: knowledge at selection (`avail_select_filter`: felix, oort_star,
  refl, feddance) and in-flight slot release (`proactive_inflight_evict`: felix only; others use the 90s
  vclock abandon).
- **FX-L12** If a trainer drops mid-flight, its compute still completes; the send is gated (real), or
  buffered to `delivery_ts = max(sct, next_avail)` (sim), and it commits late as stale. Nothing is dropped.
- **FX-L13** Under scarcity, sim jumps the vclock to the next availability transition; size `--runtime-s`
  for it rather than subtracting the jumps.
- **FX-L14** In-flight count differs per baseline (oort over-selects, async is bound by concurrency, refl
  and feddance clear each round). Size n ≈ threshold / (1 − unavailable fraction).
- **FX-L33** With the substrate on, only withhold/evict/abandon release an in-flight slot; the channel's MQTT
  UN_AVL release re-dispatched trainers mid-update and a same-end eval overwrote the buffered train (P2 EV10/EV5).
- **FX-L34** Size a harness phase for its trace: mobiperf_3st has ~10% AVL_TRAIN from t=0, so n ≈ select / 0.1 × 1.5
  (P3: n=45 for sync select 3; select 13 needs n ≈ 200).
- **FX-L35** Parallel legs need a slot each: own physical cores (`taskset`), private broker, run tag, the
  aggregator's solo core share (`FLAME_AGG_CORES`). Prove density with the ISO control before raising it.
- **FX-L36** Set a test shape through the launcher's `experiment.aggregator.agg_goal`: it fans into
  `aggGoal` and sync `aggr_num` as the last merge layer, so K = aggGoal for sync baselines.
- **FX-L37** A dataset differs by `datasets.yaml` + `fl_data.py` only; baselines, selectors and the sim/real
  machinery stay shared (google_speech = n=100, ResNet34-1D, Adam, 29 MB updates).
- **FX-L38** The first CUDA touch in a process (incl. the first CPU backward: autograd queries the device
  count) initializes the driver, serialized across processes (0.26s alone, 9-35s in a busy pool). Pay it in
  `initialize()` (`_warmup_device`), never inside a timed task.
- **FX-L39** Time sim-speed guards (wall ceiling) from the join barrier, like real's trace origin.
- **FX-L25** A sim starvation wake-up is the earliest FUTURE event that frees a slot: availability
  transition, withheld delivery, residence-held `sct`. A due entry is no wake-up (it stops the run).
- **FX-L43** Size a leg for its trace: over runtime × trace scale it must see ≥ 80% of the named unavailability
  (`effective_unavailability`; pytest over T4/G0) and a few outage/up cycles per trainer for the mechanism under test.
- **FX-L44** Trace time 0 is the join barrier (real re-anchors `agg_start_time_ts`; sim's vclock starts at round 0). Spawn
  stagger and registration fall before it (an UN_AVL trainer still joins), so a trace needs no all-available head; a
  trainer holds trace time at 0 until its first dispatch brings the origin (FX-D18).
- **FX-L15** The ramp is syn_0 (byte-identical to availability off) → syn_20 → syn_50 → mobiperf. 2-state
  traces collapse AVL_EVAL (`_trace_has_avl_eval`); only mobiperf exercises it.

**One task per version, rounds**
- **FX-L26** One task per (trainer, model version): train at v blocks train and eval at v (a train already
  returned the utility); eval at v still allows train at v (operator). Default `taskRetryPolicy: none`: the 90s
  timeout frees the slot and the trainer waits for the next version. `fixed`/`exponential` are A/B only.
- **FX-L27** Identity of an update is (trainer, version): dedup commits on it; the trainer drops a request it
  already answered; a late update from an older dispatch never answers the end's newer one.
- **FX-L28** A round advances the model version only when `agg_goal` updates aggregate (FX-N37); a starved, empty or
  all-stale iteration keeps it and still frees every consumed slot. Test the cache, not the optimizer's result.
- **FX-L40** Time out each trainer 90s after its own dispatch, both modes, applied at dispatch time; a timeout never
  closes a round (FX-N37). A per-recv timeout waits (1+K)× and overruns the budget (FX-D17).
- **FX-L41** Accumulate model state in float and cast back to each tensor's dtype once; apply baseline-specific
  server steps to parameters (float) only. Integer buffers exist only in some models (BatchNorm) (FX-D17).
- **FX-L42** A sync round that commits nothing re-dispatches at the same version: select afresh, excluding ends
  already tasked at it. A selector's round cache serves only the same round's RECV lookups (FX-N31).

**Reading the checker**
- **FX-L29** Stub legs charge a seeded compute span (`flame.harness.stub_compute_s`, fit to run_20260702 real). Those
  runs fell back to CPU (CUDA failed to start before S0), so refit it from a G0/G1 real leg (L17).
- **FX-L16** A2 failing (KS) while S3/4 passes is one in-flight gap graded at two tolerances; walk to
  residence, not eligibility.
- **FX-L17** U6 KS on a sub-ms point mass is signal-free; read `mean_diff`. P3 sub-second opposite-sign
  offsets are wall-capture; trust P3 only when `grid_KS` also fails.
- **FX-L18** `gate_holds = 0` over a whole run means the gate is structurally inert, so suspect the
  accounting upstream. Measure one-in-flight from overlapping dispatch→commit intervals, not warning
  counters.
- **FX-L19** `phase_gpu_compute` and refl K2 are sensitive to run length; re-check at ≥2.5h before acting
  on a short-run FAIL.
- **FX-L20** Run lengths: telemetry sanity 5-10 min · one mechanism rung 45 min · compounding clock residual
  (K2/K3b) 90 min-2h · refl low-frequency K2 drift 3h · C1/C2 sign-off 3-4h. Smoke 5 min first.
- **FX-L21** `Sdet` triage: eligible set differs + aggregates match = stochastic, PASS; + clock diverges = fix
  the clock; eligible set matches but decisions differ = the score VALUES diverge.
- **FX-L22** Past-dating comes in two streams (train-only U6 vs all-commit SIM_CLOCK_DIAG); always say which
  one a number came from.
- **FX-L23** `sim_committed_fresh == agg_goal` confirms oort's block-for-K fix; a later fresh-count gap is a
  different root.
- **FX-L24** Only re-run real when the real path changed; sim-only changes grade against the stored real dir.

## Felix tripwires (don'ts)

**Selectors (Oort family)**
- **FX-T3** Don't add a `system_util` recency guard or widen the slow-speed tail for oort carry-over decay.
- **FX-T6** Don't key oort latency by task type (sync oort sends no eval tasks).
- **FX-T8** Don't expect seeding to align per-round selection sets; judge S2 by speed class.
- **FX-T17** Don't expect oort carry-over decay to be a run-length transient (structural), or the felix
  min-budget seed alone to fix past-dating.

**Aggregation, ordering, clock**
- **FX-T1** Don't enable `simStaggeredRedispatch` (falsified; kept off) or retune `simRedispatchGapSeconds`.
- **FX-T2** Don't add `mqtt_fetch` (~20-57s) to `sct`; it is the wait before re-selection, not transfer.
- **FX-T4** Don't tune the `_sim_recv_min` gate predictor; `exp == sct` exactly.
- **FX-T7** Don't clamp felix's clock jump or pace dispatch to fix "fresh" past-dating; that was eval
  reusing a stale `sct`.
- **FX-T14** Don't add prediction-only gates with no real blocking (they never fire), or a `version_at(sct)`
  staleness relabel (inert).
- **FX-T15** Don't read high wall-clock commit density as sim "running fast"; judge per-round vclock
  advance and `commit_gap`.
- **FX-T18** Don't add a scalar fudge for P3's ~1s `mean_overhead` offset (wall-capture).
- **FX-T20** Don't gate sim ingestion on "committed this agg cycle"; a re-dispatch starts a new outstanding
  update. Stranding it livelocked P3 felix/fedbuff (FX-D6).
- **FX-T21** Don't call `ends()` for anything but a real dispatch: RECV or state-less calls run the selector
  (feddance phantom picks, oracle crash). List the pool with `all_ends()`.
- **FX-T22** Don't skip a trainer after the selector chose it; the selector keeps a slot nothing was sent to.
  Exclude at eligibility (the removed `[SELECTION_CHECK] Skipping`, 46-143 per felix run).

**Availability**
- **FX-T5** Don't hold ALL buffered ends out of refl's pool (`pending_ends`); it over-holds.
- **FX-T9** Don't express "busy" through the UN_AVL list, including in any unavailability redesign.
- **FX-T10** Don't use per-tick MQTT availability broadcasts (comms storm); v1 reads the trace.
- **FX-T11** Don't exclude AVL_TRAIN from eval on a 2-state trace (it empties the pool and wipes
  `selected_ends`).
- **FX-T12** Don't zero the legacy `trackTrainerAvail` block before `simUnavailability` is set
  statically (FX-N7).
- **FX-T19** Don't fork withhold/abandon logic per stack (one shared `ClientAvailability`), or grade A4 on
  the bare transition fraction (use A4dur).

**One task per version, rounds, baselines**
- **FX-T24** Don't add a staleness cutoff to fedbuff: its 1/√(1+s) discount is the baseline (operator).
- **FX-T26** Don't set `selector.kwargs.aggr_num` or `aggGoal` directly in an experiment: `agg_goal`
  overwrites them (a small-n oort smoke kept K=10 > n and starved).

**Harness and runs**
- **FX-T13** Don't grade a fedbuff/felix real run while another n=300 real run shares the broker (FX-N8).
- **FX-T16** Don't re-chase GPU contention (overrun 0), SEND_TIMEOUT or MQTT drops at n=300 cifar; all
  measured 0.
- **FX-T23** Don't run more stub trainers than node cores (1 core each): n=200 on 128 cores crawled the sim to
  its wall ceiling at vclock 6.
- **FX-T25** Don't kill FL workers by process name alone; scope by `FLAME_RUN_TAG` (`_expt_pids`,
  `slot_pids`). The runner's pre-leg `pkill -9` would have killed every neighbouring slot.
- **FX-T28** Don't derive a leg's health from the shared `experiments/` dir; parallel neighbours' logs leak
  in (FX-D13). Read the leg's own run dirs.
- **FX-T29** Don't write a replay input in place or from a daemon thread: a writer killed at exit truncates it (FX-N33).

**Datasets**
- **FX-T27** Don't take trainer lr from the 2024 speech "_oort" configs (Adam 0.04 stays at chance).
- **FX-T30** Don't name a trace by intent; name it by its measured full-day unavailable fraction (the old "syn_20" was
  10.8%, FX-D18).

---

## Built (current capabilities; one line each — details live in code, tests and `git log`)

IDs are kept because code comments cite them.

**Simulator fidelity**
- **FX-D1** Core sim fidelity: sct-ordered drain, one-in-flight residence, oort carry-over, refl pool exclusion,
  intrinsic selector duration, UCB temporal term, faithful pacer, feddance barrier-anchored U6.
- **FX-D4/D8** Cold-start gate (`simColdStartGate`, default on) and uncapped busy hold, freed on abandon/evict;
  one-in-flight holds off syn_0 (felix/fedbuff sims EV10 = 0 on syn_20/50/mobiperf_3st, cifar T4; P4 control fails).
- **FX-D6** asyncfl sim: no stranded same-cycle re-dispatch; inner loop honours `_work_done`; refl residence
  starvation wake-up.
- **FX-D9** One task per version: dispatch ledger + no-repeat guard (a train at v also blocks eval at v),
  `taskRetryPolicy` (default none), (trainer, version) commit dedup, trainer-side discard, EV15.
- **FX-D10** Sync rounds advance only on a committed aggregation (non-empty cache); an all-stale round frees its slots.
- **FX-D17** Sync stacks (syncfl, oort) abandon a trainer 90s after its dispatch in real too (shared
  `_abandon_stalled`); a real round's recvs stop at the latest awaited trainer's timeout (`recv_fifo(deadline=)`;
  the first one under `syncWaitForK`, FX-N37);
  an abandoned trainer's late update is still received, stale-gated. refl sums integer buffers in float and applies
  YoGi/QFedAvg and its staleness norm to float tensors only (reference REFL: `model.parameters()`).

**Availability**
- **FX-D2** Unavailability v1 substrate for all six: send-gate / deliver-late, two ledgers, proactive evict,
  starvation advance.
- **FX-D5** Real withheld updates are delivered (bool send-gate fix); `mobiperf_3st` launchable.
- **FX-D12** Under the substrate only withhold/evict/abandon free an in-flight slot; committed ends leave RECV;
  a dispatch consumes the end's earlier receipt.
- **FX-D16** (code cites FX-N18) Real asyncfl ingests updates on arrival: no settle sleep for felix/fedbuff (`baselines.yaml`).
  T4 real queue_wait p99: cifar ≤ 0.06s on every felix/fedbuff leg (0.1s-settle control P10: 0.227s); speech ≤ 0.43s outside FX-N30.

**Harness, checkers, datasets**
- **FX-D7** Trainer availability thread stops at EOT or shutdown (SIGTERM/atexit); clean teardown.
- **FX-D11** Streaming + oracle harness (P7/P7o) with offline replay and figures.
- **FX-D13** Event checker EV0-EV16 + injected bugs (P11a-c); parallel isolated pool with tiers T1-T4/G1/G2,
  real bank, `--changed`, `--shard`, gate (FX-N22). Run dirs are `run_<ts>_<phase>_<name>` (`FLAME_RUN_LABEL`) and never reused; health reads
  `FLAME_RUN_DIR_FILE`; the leg watchdog budgets the sim wall ceiling; sub-0.2s phases grade on mean (50 ms).
- **FX-D15** No cold start in timed tasks: startup warm-up (CPU + GPU), CUDA-only sync in the weights phase,
  sim wall ceiling from the join barrier (first-task compute 9-14s → 0.1s; fedbuff EV11 fixed).
- **FX-D18** Traces sit at their named level from t=0: `syn_10` (11%, the old syn_20, origin +600s), `syn_20` (20%, new,
  stationary, `_metadata/scripts/gen_synthetic_trace.py` seed 20), `syn_50` (50%, origin +2400s), mobiperf (origin
  +300s: the injected 5-min head) via `_metadata/scripts/shift_trace_origin.py`; syn_* extended from 24 h (then all
  UN_AVL) to mobiperf's 149 h with the same chain (`gen_synthetic_trace.py --extend`; 13 MB, parsed once per process
  with the C loader by `trace.read_trace_file`, 9 s); `effective_unavailability()`; any `syn_*`
  reaches real trainers (spawner + trainer); a real trainer's trace clock holds at 0 until the aggregator's origin
  arrives (it used to run from process start and pop transitions up to ~5 min early). T4 syn_50 legs 1200s (~2 cycles per trainer), P3 n=45, G0 uses syn_20.
  Tests: `tests/availability/test_synthetic_trace_fractions.py`, `test_harness_pool.py`.
- **FX-D14** Dataset switch (`fl_data.py`, `datasets.yaml`, `data_roots` = /coc/scratch/dgarg/fl_datasets); google_speech on the
  launcher (FX-N10).

## Open questions (operator)
- Felix paper experiment list, and the exact streaming-experiment design (FX-N13), once FX-N22 and FX-N10 land.
- Make `real_drain_ready_ingest` the default (R9)? Cifar T4 P9 vs P1: RECV_FIFO skips 887→0 (fedbuff) and 908→0
  (felix); queue_wait p99 0.028 vs 0.035s (fedbuff) and 0.019 vs 0.025s (felix); fedbuff parity 0.984 vs 0.952; no new EV fail.
  Speech P9 is mixed: skips go to 0, but felix p99 is 0.286s vs 0.186s (P1), and speech P1 fedbuff is void (shared run dir, FX-D13).
- Sync wait-K counts a late (abandoned, older-version) update toward K at the current version
  (oort stack and syncfl alike). Keep that, or count only updates trained on the current version (FX-N37)?
