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
    }


if __name__ == "__main__":
    for n in names():
        d = get(n)
        print(f"{n:8s} labels={d.num_labels:3d} seq={d.max_seq_length:4d} "
              f"train={d.n_train:>9,} test={d.n_test:>7,} "
              f"p(rf=16)={d.probe_dim(16):,} p(rf=64)={d.probe_dim(64):,}")
