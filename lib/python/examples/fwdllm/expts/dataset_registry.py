"""One place that knows what a dataset IS: paths, labels, split, sequence length.

Everything dataset-specific that used to be hardcoded -- the h5 paths repeated in
every expt yaml, `num_labels`, the 4-class assumptions in the preflights, the
classifier's contribution to `p` -- resolves through here, so adding a dataset is
a row in `configs/datasets.yaml` and switching one is `--dataset NAME`.

Facts live in the yaml; this module only loads, validates and derives. Stdlib +
pyyaml at import; h5py only when a call actually reads the data file.

    from examples.fwdllm.expts.dataset_registry import get, probe_dim
    ds = get("yahoo")
    ds.data_file_path, ds.num_labels, ds.max_seq_length
    probe_dim(ds.num_labels, reduction_factor=16)   # -> 454,954
"""
import os
from dataclasses import dataclass
from typing import Tuple

import yaml

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "configs", "datasets.yaml")

# Adapter params at reduction_factor rf, DistilBERT-base, measured off the
# production model (fl_fwd_ft_practice.md P1). The classifier is dataset-sized
# and the pre_classifier is dropped by the trainer at :217, so
#   p = ADAPTER_P[rf] + 768*num_labels + num_labels.
ADAPTER_P = {16: 447264, 32: 225936, 64: 115272}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_labels: int
    max_seq_length: int
    partition_method: str
    train_range: Tuple[int, int]
    test_range: Tuple[int, int]
    data_file_path: str
    partition_file_path: str

    @property
    def n_train(self) -> int:
        return self.train_range[1] - self.train_range[0]

    @property
    def n_test(self) -> int:
        return self.test_range[1] - self.test_range[0]

    def probe_dim(self, reduction_factor: int = 16) -> int:
        return probe_dim(self.num_labels, reduction_factor)

    def label_vocab(self) -> dict:
        """The h5's own label map -- what the run actually derives num_labels from."""
        import h5py
        import json
        with h5py.File(self.data_file_path, "r", swmr=True) as f:
            return json.loads(f["attributes"][()])["label_vocab"]


def probe_dim(num_labels: int, reduction_factor: int = 16) -> int:
    """Trainable `p` after the trainer drops pre_classifier. Dataset-dependent
    through the classifier alone: agnews 450,340 / yahoo 454,954 / yelp-p 448,802
    at rf=16."""
    if reduction_factor not in ADAPTER_P:
        raise KeyError(f"no adapter param count for reduction_factor={reduction_factor}; "
                       f"known: {sorted(ADAPTER_P)}")
    return ADAPTER_P[reduction_factor] + 768 * num_labels + num_labels


def total_data_bins(name: str, num_clients: int, train_batch_size: int) -> int:
    """Data bins one client holds: `ceil(shard / batch)` with equal shards.

    The aggregator's `data_id` range AND the trainer's own batch index
    (`FedSgdTrainer:520`), so too low silently drops every shard's tail -- the
    list is merely longer than the index. Hardcoded at agnews' 150, which gave
    yahoo 1,200 of each client's 14,000 rows (8.6%).

    **Batch stays 8 and the bin count moves**: the batch is the unit each JVP is
    estimated on, the bin count is pure addressing. agnews 150 / yahoo 1,750 /
    yelp-p 650 at C=100 -- buildplan §1's table.
    """
    if num_clients <= 0 or train_batch_size <= 0:
        raise ValueError(f"num_clients={num_clients} train_batch_size={train_batch_size}")
    shard = get(name).n_train // num_clients
    return -(-shard // train_batch_size)          # ceil; the loaders drop_last=False


def data_coverage(name: str, num_clients: int, train_batch_size: int) -> dict:
    """Does `bins x batch x C` equal the training set? Not automatic -- it needs
    equal shards AND `shard % batch == 0`; a short last batch undercounts by up
    to `C*(batch-1)` rows. True for all three datasets at C=100 and 1,000 today.
    Returns the pieces for the caller to report; raises nothing.
    """
    ds = get(name)
    shard = ds.n_train // num_clients
    bins = total_data_bins(name, num_clients, train_batch_size)
    reached = bins * train_batch_size * num_clients
    return {
        "dataset": ds.name, "n_train": ds.n_train, "clients": num_clients,
        "batch": train_batch_size, "shard": shard, "bins": bins,
        "reached": reached, "exact": reached == ds.n_train,
        "shard_remainder": ds.n_train - shard * num_clients,
        "batch_remainder": shard % train_batch_size,
    }


def max_dominant_share(num_labels: int) -> float:
    """Ceiling on a reference batch's dominant-class share before it counts as
    class-skewed. Balanced is 1/K, so the old fixed 0.5 encoded agnews' K=4: it
    waves through a 10-class reference that is 5x over-concentrated, and it fires
    on every *balanced* 2-class one (yelp-p), which refused to launch."""
    return min(0.999, 1.0 / max(1, int(num_labels)) + 0.25)


def _load():
    with open(_CONFIG) as fh:
        return yaml.safe_load(fh)


def root() -> str:
    return os.environ.get("FWDLLM_DATA_ROOT") or _load()["root"]


def cache_root() -> str:
    """Tokenized features. Absolute + shared: `model_args.cache_dir`'s own default
    is relative, so the launch directory used to pick the cache (§10)."""
    return os.environ.get("FWDLLM_CACHE_ROOT") or _load()["cache_root"]


def cache_file(name: str, client_id: int, partition_method: str = None,
               model_type: str = "distilbert",
               model_name: str = "distilbert-base-uncased",
               model_class: str = "ClassificationModel") -> str:
    """The path `_load_data_loader_from_cache` builds (`base_data_manager.py:583`);
    `client_id=-1` is the server's global test set. The key is everything that
    changes the tensors, so a seq-length or partition switch MISSES rather than
    silently reusing another group's shard."""
    ds = get(name)
    return os.path.join(cache_root(), "_".join([
        model_type, model_name.split("/")[-1], "cached", str(ds.max_seq_length),
        model_class, ds.name, partition_method or ds.partition_method,
        str(client_id),
    ]))


def sim_charge_profile(current: str, name: str = None) -> str:
    """The `<current>_<dataset>.yaml` sibling when it exists, else `current`.

    Per-pass cost scales with `max_seq_length`, so an agnews-profiled file
    mis-prices a yahoo vclock. Keyed off the yaml's own value, not the baseline
    name, because `fluxtune_v1`/`v2` share `fluxtune.yaml`. The fallback is
    agnews-profiled, which the launcher's preflight then refuses elsewhere."""
    if not current or not name:
        return current
    stem, ext = os.path.splitext(current)
    per_ds = f"{stem}_{name}{ext}"
    repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
    return per_ds if os.path.exists(os.path.join(repo, per_ds)) else current


def names():
    return sorted(_load()["datasets"])


def get(name: str, check_h5: bool = False) -> DatasetSpec:
    """Spec for `name`. `check_h5=True` also asserts the h5's label_vocab agrees
    with the registry -- cheap (attributes only), so any tool that is about to
    act on real data should pass it."""
    cfg = _load()
    if name not in cfg["datasets"]:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(cfg['datasets'])}")
    d = cfg["datasets"][name]
    r = root()
    spec = DatasetSpec(
        name=name,
        num_labels=int(d["num_labels"]),
        max_seq_length=int(d["max_seq_length"]),
        partition_method=str(d["partition_method"]),
        train_range=tuple(d["train_range"]),
        test_range=tuple(d["test_range"]),
        data_file_path=os.path.join(r, "data_files", f"{name}_data.h5"),
        partition_file_path=os.path.join(r, "partition_files", f"{name}_partition.h5"),
    )
    if check_h5:
        n = len(spec.label_vocab())
        if n != spec.num_labels:
            raise ValueError(f"{name}: registry says num_labels={spec.num_labels}, "
                             f"{spec.data_file_path} label_vocab says {n}")
    return spec


def hyperparameter_overrides(name: str) -> dict:
    """The hyperparameters block a run needs to be pointed at this dataset --
    the same keys on both the aggregator and the trainer side."""
    ds = get(name)
    return {
        "dataset": ds.name,
        "data_file_path": ds.data_file_path,
        "partition_file_path": ds.partition_file_path,
        "max_seq_length": ds.max_seq_length,
        # §10. Dual-read: both mains build model_args from this block.
        "cache_dir": cache_root(),
    }


if __name__ == "__main__":
    for n in names():
        d = get(n)
        print(f"{n:8s} labels={d.num_labels:3d} seq={d.max_seq_length:4d} "
              f"train={d.n_train:>9,} test={d.n_test:>7,} "
              f"p(rf=16)={d.probe_dim(16):,} p(rf=64)={d.probe_dim(64):,}")
