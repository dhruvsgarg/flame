# Plan: Fine-grained trainer timing + CPU pinning + wall/virtual-clock correctness

Scope: `async_cifar10` (`oort` on node1, `refl` on node2 via
[scripts/debug_run.sh](scripts/debug_run.sh)). Three independent tasks; shared
logic lands in the `flame` library so other examples inherit it.

Status legend: ☐ todo. Each task is self-contained — implement and verify in
isolation. **Do not implement from this file; this is the design.**

---

## Task 1 — Per-phase trainer timing (CPU vs GPU split), averaged per round

### Problem
Spikes in "trainer compute time" but we only emit three coarse phases today
(`pre_train_s`, `real_gpu_time_s`, `post_train_s`) in the `trainer_round` event
([trainer/pytorch/main.py:785-797](trainer/pytorch/main.py#L785-L797)). We
cannot tell whether a spike is MQTT, host→device copy, GPU compute, the modeled
sleep, or the upload leg. We want a finer breakdown, **split into CPU-bound and
GPU-bound phases**, plotted separately, averaged across trainers per round.

### Phase taxonomy (map to existing code)
The full round straddles the trainer's `get` → `train` → `send` tasklets, so
instrumentation spans **both** the base trainer and the example. Phases and
their nature:

| phase key            | where (code)                                                                                  | nature |
|----------------------|-----------------------------------------------------------------------------------------------|--------|
| `mqtt_fetch_s`       | `channel.recv(end)` in `_fetch_weights` ([syncfl/trainer.py:169](../../flame/mode/horizontal/syncfl/trainer.py#L169)) | CPU/net |
| `weights_to_ram_s`   | deserialize portion of `weights_to_model_device` ([syncfl/trainer.py:228](../../flame/mode/horizontal/syncfl/trainer.py#L228)) | CPU |
| `weights_to_gpu_s`   | host→device copy in `_update_model` / `load_state_dict` ([syncfl/trainer.py:557](../../flame/mode/horizontal/syncfl/trainer.py#L557)) | GPU xfer |
| `pre_train_s`        | already computed ([main.py:684](trainer/pytorch/main.py#L684))                                 | CPU |
| `gpu_compute_s`      | `_real_gpu_time_s` ([main.py:691](trainer/pytorch/main.py#L691))                               | GPU |
| `sleep_s`            | `_remaining_time` real-mode sleep ([main.py:804-805](trainer/pytorch/main.py#L804-L805))      | wall (neither) |
| `post_process_s`     | `_send_weights` build: `_update_weights`+`_delta_weights_fn`+DP+`weights_to_device(...CPU)` ([syncfl/trainer.py:361-372](../../flame/mode/horizontal/syncfl/trainer.py#L361-L372)) | CPU + device→host xfer |
| `mqtt_send_s`        | `channel.send(end, msg)` ([syncfl/trainer.py:417](../../flame/mode/horizontal/syncfl/trainer.py#L417))  | CPU/net |

Note `post_process_s` mixes CPU (delta math) and a device→host copy; split it
into `post_cpu_s` and `weights_from_gpu_s` if cheap, else classify whole as CPU
and footnote it.

CPU-plot phases: `mqtt_fetch_s, weights_to_ram_s, pre_train_s, post_cpu_s,
mqtt_send_s`. GPU-plot phases: `weights_to_gpu_s, gpu_compute_s,
weights_from_gpu_s`. `sleep_s` shown on neither (or a thin "wall" overlay).

### Mechanism
1. **Phase accumulator on the trainer.** Add `self._phase_times: dict[str,float]`
   reset at the start of each `get` (round boundary). A tiny context manager
   `with self._phase("mqtt_fetch_s"):` wraps each region. For GPU phases that are
   async, call `torch.cuda.synchronize()` before stopping the timer (already
   implicit for `gpu_compute_s` via `.item()` syncs; add an explicit sync around
   `weights_to_gpu_s`). Put the helper in the **base trainer**
   ([syncfl/trainer.py](../../flame/mode/horizontal/syncfl/trainer.py)) so async
   inherits it; the example reuses it for its train-loop phases.
2. **Emit.** Extend the `trainer_round` event `extra` dict
   ([main.py:785](trainer/pytorch/main.py#L785)) with all phase keys. Schema
   addition only — no new event type; keep `build_trainer_round`
   ([telemetry/events.py:134](../../flame/telemetry/events.py#L134)) backward
   compatible (all new fields optional). Telemetry already no-ops when disabled,
   so cost is gated.
3. **Cross-round stamps already exist** for the inter-tasklet legs:
   `WALL_RECV_TS`/`WALL_SEND_TS` ([syncfl/trainer.py:407-415](../../flame/mode/horizontal/syncfl/trainer.py#L407-L415)).
   Reuse them so `mqtt_fetch_s` and `mqtt_send_s` are measured from the same
   stamps the aggregator-side lag decomposition uses (consistency with
   `analyze_send_recv_lag.py`).

### Plots (new analyzer)
New `scripts/analyze_trainer_phases.py` reading `telemetry/trainer_*.jsonl`:
- Group `trainer_round` events by `round`, **average each phase across
  trainers** (also emit p50/p90/max bands to expose the spikes).
- **Plot A (CPU):** stacked bar/area of CPU phases over rounds.
- **Plot B (GPU):** stacked bar/area of GPU phases over rounds.
- Optional **Plot C:** per-trainer heatmap (round × trainer) of total round
  time to localize *which* trainers spike.
- Reuse matplotlib helpers from `scripts/plotters/stacked_bar_plot.py` /
  `stall_stacked_bar_plot.py` rather than rewriting.
- `--compare run_a run_b` to overlay oort vs refl.
- Wire into the existing post-run analysis path so PNGs land in
  `<run_dir>/plots/` automatically (same hook as `analyze_send_recv_lag.py`).

### Deliverables
- ☐ `_phase` helper + `_phase_times` in base trainer; GPU sync points.
- ☐ Phase stamps in `_fetch_weights`, `_send_weights`, and `train()`.
- ☐ Extended `trainer_round` telemetry fields (optional, documented).
- ☐ `scripts/analyze_trainer_phases.py` (CPU plot, GPU plot, spike heatmap).
- ☐ Verify: smoke run shows phase sums ≈ measured round wall time (±5%).

---

## Task 2 — Automatic CPU core pinning per trainer

### Problem
Workers are GPU-pinned (`CUDA_VISIBLE_DEVICES = (trainer_id-1) % num_gpus`,
[spawner.py:239-243](../../flame/launch/spawner.py#L239-L243)) but **not**
CPU-pinned. With 100–300 trainers the OS scheduler bounces threads across cores,
and each PyTorch process spawns many intra-op threads → oversubscription and the
GPU-scheduling contention behind the spikes in Task 1.

### Mechanism (in [spawner.py](../../flame/launch/spawner.py))
1. **Discover usable cores** (respects cgroup/Slurm affinity, not just
   `os.cpu_count()`): `cores = sorted(os.sched_getaffinity(0))`. Compute once in
   `TrainerSpawner.__init__`.
2. **Assign one core per trainer, round-robin, evenly:**
   `core = cores[(trainer_id - 1) % len(cores)]`. With 300 trainers on 64 cores
   this lands ~5 trainers/core, evenly. Optional refinement: keep the assigned
   core **NUMA-local to the trainer's GPU** (group cores by NUMA node, pick from
   the node hosting `gpu_id`) — document as a follow-up, default to flat
   round-robin.
3. **Apply affinity in the child** via `subprocess.Popen(preexec_fn=...)`
   (Linux-only, fine here): `preexec_fn=lambda c={core}: os.sched_setaffinity(0, c)`.
   Assign a small contiguous block (e.g. the single core, optionally `{core}`
   plus a shared "spillover" core) — start with exactly one core to force the
   "all work per trainer on one core" requirement.
4. **Prevent thread oversubscription** (critical — pinning to one core without
   this just serializes many threads on that core): set in the child `env`
   before `Popen`:
   `OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
   NUMEXPR_NUM_THREADS=1`, and have the trainer call `torch.set_num_threads(1)`
   at startup ([main.py](trainer/pytorch/main.py) `initialize`/`main`).
5. **Config toggle:** `--cpu_pinning {on,off}` (default `on`) plumbed like
   `--time_mode` (spawner field → CLI arg). Off = today's behavior.
6. **Log the map:** extend the existing spawn line
   ([spawner.py:275](../../flame/launch/spawner.py#L275)) to
   `"Spawned trainer N on GPU g, CPU core c"` and dump the full
   trainer→(gpu,core) table once.

### Edge cases
- `sched_getaffinity` unavailable (non-Linux) → skip pinning, warn.
- trainers > cores → round-robin shares evenly (intended).
- `preexec_fn` + threads: safe here (child immediately `exec`s the trainer).

### Deliverables
- ☐ Core discovery + round-robin assignment in `TrainerSpawner`.
- ☐ `preexec_fn` affinity + thread-limit env + `torch.set_num_threads(1)`.
- ☐ `--cpu_pinning` toggle plumbed through spawner/CLI.
- ☐ Logged trainer→(gpu,core) table.
- ☐ Verify: `taskset -cp <pid>` / `/proc/<pid>/status Cpus_allowed_list`
  confirms one core per trainer; spikes in Task 1 plots shrink.

---

## Task 3 — Wall-clock vs virtual-clock termination correctness

### Problem
`--runtime-s` in [debug_run.sh](scripts/debug_run.sh) sets
`hyperparameters.max_runtime_s` for both modes. Intended semantics:
- **real** run → terminate at **wall = T** seconds. ✅ works.
- **simulated** run → terminate when **virtual clock reaches T**, however long
  wall takes (should be ≪ T since sleeps are skipped). ❌ unreliable.

Root cause in `increment_round` (defined in
[syncfl/top_aggregator.py:704-734](../../flame/mode/horizontal/syncfl/top_aggregator.py#L704-L734),
inherited by async): the **wall-clock failsafe reuses the same `max_runtime_s`**
as the virtual cap. In sim mode, if real GPU work (300 trainers, contention)
pushes wall past `T` before `vclock.now` reaches `T`, the
`[WALL_CLOCK_FAILSAFE]` fires and kills the sim run early — so the sim never
covers `T` virtual seconds. The failsafe is meant only as a hang guard, but it
currently dictates the stop.

### Fix
1. **Decouple the failsafe budget.** New config
   `hyperparameters.max_wall_runtime_s` (the true real-time hang guard),
   independent of `max_runtime_s` (the virtual budget in sim mode). In
   `increment_round`:
   - sim mode primary stop: `vclock.now >= max_runtime_s`.
   - sim mode failsafe: `wall_elapsed >= max_wall_runtime_s` (default generous,
     e.g. `max(max_runtime_s, K * estimated_wall)` or a plain large constant like
     `4 * max_runtime_s`) — fires only on a genuine vclock stall, with the
     existing `[WALL_CLOCK_FAILSAFE]` warning.
   - real mode unchanged: `wall_elapsed >= max_runtime_s`.
2. **Periodic virtual-clock progress log.** Emit
   `[VCLOCK_PROGRESS] vclock=<v>s wall=<w>s speedup=<v/w>x round=<r>` every N
   seconds (or every K rounds) from the aggregator loop in sim mode — both async
   ([asyncfl/top_aggregator.py](../../flame/mode/horizontal/asyncfl/top_aggregator.py),
   near the `T_v` advance at line ~217) and sync (in `_sync_sim_recv_first_k`
   after `advance`, [syncfl/top_aggregator.py:326](../../flame/mode/horizontal/syncfl/top_aggregator.py#L326)).
   This makes "where is virtual time vs wall" visible live, and feeds the
   comparison below.
3. **Plumb through debug_run.sh.** Add `--wall-runtime-s` (default e.g.
   `4 * runtime_s`) and write `max_wall_runtime_s` into the patched YAML in
   `make_debug_yaml` ([scripts/debug_run.sh](scripts/debug_run.sh), alongside the
   existing `h["max_runtime_s"]` assignment).
4. **Apples-to-apples comparison tool.** New `scripts/compare_clock_parity.py`
   (or extend [scripts/compare_parity.py](scripts/compare_parity.py)): given a
   `real` run and a `simulated` run of the same config, plot **virtual-time
   trajectory** — for real, `wall_elapsed`; for sim, `vclock.now` — vs round
   (and vs model_version). Both should land on the **same curve per round**;
   report (a) per-round virtual-time deviation, (b) the sim wall speedup
   (`T / sim_wall`), (c) whether sim stopped at `vclock≈T` (correct) or was cut
   by the failsafe (bug). Reuse `VCLOCK_PROGRESS` and the `trainer_round`
   `sim_completion_ts` already emitted.

### Note on sync sim ordering
Sync sim-ordering **is** implemented (`_sync_sim_recv_first_k` advances `vclock`,
[syncfl/top_aggregator.py:285-336](../../flame/mode/horizontal/syncfl/top_aggregator.py#L285-L336)),
so refl's vclock does advance — the failsafe decoupling is the actual fix. Still,
verify on a refl sim run that `vclock` advances monotonically and the run stops
at `vclock≈T`, not at the wall failsafe.

### Deliverables
- ☐ `max_wall_runtime_s` config + decoupled failsafe in `increment_round`.
- ☐ `[VCLOCK_PROGRESS]` periodic log (async + sync).
- ☐ `--wall-runtime-s` in `debug_run.sh` → `max_wall_runtime_s` in YAML.
- ☐ `scripts/compare_clock_parity.py` (virtual-time trajectory + speedup +
  failsafe-triggered check).
- ☐ Verify: real run stops at wall≈T; sim run stops at vclock≈T with wall≪T;
  per-round virtual-time curves match within tolerance.

---

## Suggested order
1. **Task 2** (CPU pinning) first — cheapest, and likely reduces the Task 1
   spikes, changing what the timing plots show.
2. **Task 1** (phase timing + plots) — confirms pinning's effect and localizes
   residual spikes.
3. **Task 3** (clock correctness) — independent; needed for trustworthy
   real-vs-sim comparison once timing is clean.

All telemetry additions are optional/backward-compatible; all toggles default to
preserve current behavior except `--cpu_pinning` (default `on`).
