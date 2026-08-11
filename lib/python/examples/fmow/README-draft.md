## FMoW on Flame — Federated Learning over Satellite Imagery

This example ports **fMoW** (Functional Map of the World) onto Flame's federated
learning stack, using a satellite-constellation framing: each trainer is a LEO
satellite that only "sees" (and can train on) the subset of images it has
physically captured so far in simulated time, rather than a static per-client
data partition.

### What is FMoW?

fMoW is a satellite-image scene-classification dataset: RGB chips labeled with
one of 62 land-use/scene categories (`airport`, `race_track`,
`amusement_park`, ...), split into `train` / `val` / `test` / `seq`. This port
uses the WILDS-preprocessed release (fMoW-rgb v1.1), which is a flat directory
of already-resized 224x224 PNGs, `images/rgb_img_{i}.png`, where `i` is simply
the row's position in `rgb_metadata.csv` — **not** the descriptive
`train/<category>/<category>_<id>/...` path that the CSV's `img_path` column
happens to contain (that column is vestigial in this distribution; see
"Bugs found and fixed" below).

### The satellite framing

Instead of a Dirichlet or index-based split, data availability here is physical. `captures.py` uses the `geodetic.npz` LEO constellation to simulate every timestep and track when a satellite comes within a specified `radius` (km) of an image. Each trainer only trains on images captured up to the current simulated time, with its dataset growing over rounds rather than being fixed.

Captures are given as `events`, where an event contains the `timestep` the event occurred on and the `image` captured. 

### Layout

```
configs/
  fmow_config.yaml     FMoW specific configuration
  trainer_base.yaml    Flame trainer hyperparameters template
                       (adds fmow_config_path, reuses satellite_coordinates_path)

dependencies/
  model.py             DenseNet161 (ImageNet-pretrained), group normalized and
                       freeze_layers leaves only denseblock4/norm5/classifier trainable
  fmow_dataset.py      Transforms dataset into Tensor arrays and allows for selection
                       by split

setup/
  download_fmow_dataset.sh   downloads + extracts fMoW-rgb v1.1 (~54GB)
  config.py                  FMoW configuration schema
  captures.py                Builds captures.npz based on geodetic.npz (events + per-satellite offsets)
  setup_fmow.py              One-time setup for FMoW: download, then schedule captures

trainer/pytorch/main.py               PyTorchFMoWTrainer
aggregator/pytorch/main_fedavg_agg.py PyTorchFMoWAggregator

exports/fmow_fedavg_smoke_n10.yaml    FMoW smoketest:
                                      baseline=fedavg, 10 trainers, 50 rounds
exports/fmow_fedavg_n200.yaml         FMoW at scale:
                                      baseline=fedavg, 200 trainers, 500 rounds,
                                      capture-driven data (path_style: true)
```

`model.py`, `fmow_dataset.py`, and `config.py` are symlinked into both
`trainer/pytorch/` and `aggregator/pytorch/` (mirroring the existing
`metadata -> ../_metadata` convention), so both roles import the same code
with a plain top-level `import model` / `from fmow_dataset import ...` rather
than using `sys.path`, since Python auto-adds a directly-run script's own directory
to `sys.path`. The one thing symlinks can't solve is locating the plain data
file `fmow_config.yaml` itself, which is why `fmow_config_path` exists as a
hyperparameter (set once in `trainer_base.yaml` for the trainer, and again
under the experiment YAML's `aggregator.config_overrides.hyperparameters`,
since the aggregator's config comes from a completely separate merge chain
that never reads `trainer_base.yaml`).

### How a round works

**Trainer** (`PyTorchFMoWTrainer`, one per satellite): `initialize()` loads
`fmow_config.yaml` and builds the model; `load_data()` loads this satellite's
slice of `captures.npz`, admits whatever's been captured as of the current
simulated time into `self.image_buffer`, and builds the train loader from it.
Every `train()` call re-admits newly captured images and rebuilds the loader
before training, so the dataset a satellite trains on grows as simulated time advances.

**Aggregator** (`PyTorchFMoWAggregator`): standard FedAvg `TopAggregator` —
`random` selector, `fedavg` optimizer, evaluates against the fMoW `test` split.

### Scaling to 200 satellites

The initial port validated at smoke-test scale (`fmow_fedavg_smoke_n10.yaml`:
10 trainers, 50 rounds). Running `fmow_fedavg_n200.yaml` at real scale (200
trainers, 500 rounds) surfaced failures smoke scale never exercised — mostly
concurrency assumptions in shared framework code that no prior example on
this framework had ever stressed, not bugs unique to FMoW. `async_cifar10`'s
trainer is a small hand-written CNN over 32x32 images; FMoW's trainer runs a
full DenseNet-161 (~28.7M params) over 224x224 images, with per-satellite
datasets that grow into the thousands of images over a run. Nothing in the
chunk transport or selector bookkeeping had ever needed to move or track this
much per round before.

### Bugs found and fixed

**Framework-level** (`flame/selector/default.py`, affects every example using
`selector.sort: default`, not just FMoW):
- `DefaultSelector`'s initializer didn't accept `**kwargs`, so it broke the moment `channel_manager.py` started unconditionally injecting a `_seed` kwarg into every selector's constructor. Fixed to accept and forward `**kwargs` to `AbstractSelector`, matching every other selector class.
- `DefaultSelector` was also missing `_cleanup_recvd_ends` and `_cleanup_send_ends` methods which seemed to be required by `channel.py` and `trainer.py`. Also missing `_cleanup_removed_ends(end_id)`, which discards a removed end from `selected_ends`.

**FMoW-specific:**
- `_rebuild_train_loader()` built its `DataLoader` with `shuffle=True`, and shuffling crashed whenever a satellite's capture buffer is still empty. Trainers having an empty dataset is specific to the way FMoW captures images over time. Fixed by gating `shuffle=len(indices) > 0`, and by skipping the epoch loop entirely in `train()` when there's nothing captured yet that round.

**Framework-level, concurrency at real scale** (`flame/backend/chunk_manager.py`, `flame/backend/chunk_store.py`, affects any example moving large payloads under real concurrency, not just FMoW):
- `ChunkStore.assemble()` required chunks to arrive in exact sequential order (`if self.seqno + 1 != msg.seqno: return False`), and any out-of-order chunk discarded the entire in-progress transfer, forcing a full resend. This only mattered once many trainers were sending large payloads concurrently over the same transport — at smoke scale, chunks arrived in order often enough that it never triggered. Fixed by keying `recv_buf` on `seqno` (a dict instead of a position-dependent list), so arrival order no longer matters; a transfer is only complete once every chunk from `0` to `eom_seqno` has actually arrived (`len(recv_buf) == eom_seqno + 1`). A duplicate chunk is now silently ignored instead of erroring. A new `is_stale()` check paired with a 30-second `TRANSFER_TIMEOUT` resets a transfer that started but never finished. `DEFAULT_CHUNK_SIZE` also went from 1MB to 4MB, cutting the number of chunks (and therefore the surface area for reordering) per transfer.
- `RandomSelector` had no `on_round_completed()`, so `selected_ends` / `all_selected` were never cleared between rounds — once a trainer had been selected and responded, it stayed marked as such for the rest of the run. After the first round this made `agg_goal` unreachable and produced the same symptoms as `SIM_STARVATION`'s fast-forward mechanism, even though the aggregator wasn't actually out of eligible data, the selector's own bookkeeping had just never been cleared. Fixed by adding `RandomSelector.on_round_completed()`, called both when a round succeeds and when aggregation fails and returns no weights, so a failed round doesn't leave behind the same stale state a successful one used to. `channel._selector.on_update_received()` was also only being called from the branch handling real, weight-bearing responses, not the one for a trainer with nothing to contribute — moved so it fires regardless of which message branch a response came through. `SIM_STARVATION` logging was also extended with eligible count, threshold, total ends, and in-flight ends/IDs, since diagnosing this kind of state-tracking bug needs to see what the selector actually believed at the moment it gave up.

**FMoW-specific, reducing what has to move per round:**
- A trainer's `delta_weights` now drops every parameter whose `requires_grad` is `False` before sending. Since the model only trains its last dense block and classifier, the transmitted payload is a small fraction of the full DenseNet-161, meaning fewer chunks and less concurrent network pressure at any given moment.
- An empty-dataset trainer previously still entered the full training/send path and produced a payload regardless of whether it had anything meaningful to contribute. It now sends a minimal status-only message (model version, dataset size (zero), stat utility, local accuracy) with no weights key at all.
- A `random.uniform(0, 1)` second sleep was added on the simulated send path. Simulated trainers running the same modeled delay tend to finish and send in lockstep, producing synchronized bursts of concurrent traffic; the jitter spreads that out.

**FMoW-specific, evaluation:**
- The eval-skip condition (`if self._round != 1 and self._round % eval_every != 0: return`) could skip evaluation on the last round entirely if the round count didn't happen to land on the cadence. An explicit `is_last_round` check now forces evaluation on the final round regardless of cadence, and runs it synchronously rather than in a background thread, so the run doesn't exit before that last evaluation finishes.
- Intermediate evaluations cap at 5 batches instead of the full test set, deliberately, since a full-test-set evaluation on every cadence hit at real scale would dominate per-round time on its own. The final round still evaluates the complete test set, uncapped. Progress is now logged per batch (`[EVAL_START]` / `[EVAL_PROGRESS]`).

### Guards ahead of a dg-fork-main merge

`dg-fork-main`'s latest commit (`4a75cb7a`, landing FwdLLM real/sim parity)
doesn't touch FMoW's own code, but it changes two pieces of the shared
framework FMoW depends on. Neither is fixed by a merge conflict, both are
silent otherwise, so the guards are added here ahead of time:
- `eval_every_n_commits: 1` added to `fmow_fedavg_n200.yaml`'s hyperparameters. `dg-fork-main` adds a separate per-commit stride gate inside `_eval_snapshot_model()` that stacks on top of `eval_every_n_rounds` rather than replacing it; left unset, it silently drops roughly half of every evaluation FMoW schedules (default stride is 2). Verified directly against the real run's cadence (500 rounds, `eval_every_n_rounds: 10`): 25 of 51 scheduled evaluations dropped with the gate unset, zero dropped with `eval_every_n_commits: 1`.
- `main.py`'s heartbeat setup now reads `getattr(self.config.hyperparameters, "heartbeats", {})` instead of accessing `self.config.hyperparameters.heartbeats` directly. `dg-fork-main` removes the `heartbeats` field from the shared config schema entirely, with no default, so the direct access raises `AttributeError` on every trainer at startup, confirmed against the real run's actual config, which never sets `heartbeats` itself.
