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

### Bugs found and fixed

**Framework-level** (`flame/selector/default.py`, affects every example using
`selector.sort: default`, not just FMoW):
- `DefaultSelector`'s initializer didn't accept `**kwargs`, so it broke the moment `channel_manager.py` started unconditionally injecting a `_seed` kwarg into every selector's constructor. Fixed to accept and forward `**kwargs` to `AbstractSelector`, matching every other selector class.
- `DefaultSelector` was also missing `_cleanup_recvd_ends` and `_cleanup_send_ends` methods which seemed to be required by `channel.py` and `trainer.py`.

**FMoW-specific:**
- `_rebuild_train_loader()` built its `DataLoader` with `shuffle=True`, and shuffling crashed whenever a satellite's capture buffer is still empty. Trainers having an empty dataset is specific to the way FMoW captures images over time. Fixed by gating `shuffle=len(indices) > 0`, and by skipping the epoch loop entirely in `train()` when there's nothing captured yet that round.