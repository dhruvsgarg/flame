#!/usr/bin/env python3
"""Task F2 (buildplan §10): fill the shared feature cache once, off the critical path.

A cold run pays tokenization inside the arm's own wall budget: `234931` spent 32 of
its 44 minutes before commit 1, with 100 trainers each reading and tokenizing a
14,000-row shard in parallel. That cost is one-time per (dataset, seq length,
partition group) and belongs here, on a CPU box, not in a GPU arm.

**This drives the production loader.** `TextClassificationDataManager.load_federated_data`
is the exact call `trainer/main.py:158` makes, and populating the cache is its side
effect -- so the bytes are the trainer's own (§0 rule 4). A hand-rolled tokenizer that
differs by one token is worse than no cache at all, because nothing would report it.

    ./pretokenize_dataset.py --dataset yelp-p --clients 100 --jobs 8
    ./pretokenize_dataset.py --dataset yahoo --dry-run        # what is missing, and how big

Onboarding order for a new dataset (§1 checklist):
    build_niid_partitions.py -> check_partitions.py -> pretokenize_dataset.py -> launch
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from examples.fwdllm.expts import dataset_registry as dsreg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(HERE, "..", "configs", "aggregator_base.json")

# One worker builds these once and reuses them for every client it is handed.
_W = {}


def _hyperparameters(config_path, dataset, partition_method):
    """The launcher's own merge order: the shipped base config, then the registry's
    per-dataset overrides on top (run_sequential.sh:671)."""
    import json
    import types
    hp = json.load(open(config_path))["hyperparameters"]
    hp.update(dsreg.hyperparameter_overrides(dataset))
    if partition_method:
        hp["partition_method"] = partition_method
    return types.SimpleNamespace(**hp)


def _init_worker(config_path, dataset, partition_method):
    # The loader's per-row tqdm bars and the preprocessor's `print(df)` are one
    # trainer's debug output; 16 of them interleaved bury this script's own report.
    os.environ["TQDM_DISABLE"] = "1"
    from examples.fwdllm.data_manager.base_data_manager import BaseDataManager
    from examples.fwdllm.data_preprocessing.text_classification_preprocessor import (
        TLMPreprocessor,
    )
    from examples.fwdllm.expts.initializer import create_model
    from examples.fwdllm.trainer.model_args_builder import build_model_args

    hp = _hyperparameters(config_path, dataset, partition_method)
    attrs = BaseDataManager.load_attributes(hp.data_file_path)
    margs = build_model_args(hp, len(attrs["label_vocab"]))
    # create_model also loads the classifier we never use; it is how the trainer
    # gets its tokenizer, and the tokenizer is what has to match.
    _, _model, tok = create_model(margs, formulation="classification")
    del _model
    _W.update(hp=hp, margs=margs,
              pre=TLMPreprocessor(args=margs, label_vocab=attrs["label_vocab"],
                                  tokenizer=tok))


def _tokenize(client_id):
    """Load one client's shard through the production path; the cache write is the
    point. `-1` is the server's global test set, which `agg_eval` needs."""
    from examples.fwdllm.data_manager.text_classification_data_manager import (
        TextClassificationDataManager,
    )
    import contextlib
    hp, margs, pre = _W["hp"], _W["margs"], _W["pre"]
    hp.client_idx = max(client_id, 0)
    t0 = time.time()
    with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
        dm = TextClassificationDataManager(hp, margs, pre, 1 if client_id >= 0 else 0,
                                           hp.data_loader_num_workers)
        if client_id >= 0:
            dm.load_federated_data(process_id=1, client_idx=client_id)
        else:
            dm.load_federated_data(process_id=0, client_idx=0)
    return client_id, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=dsreg.names())
    ap.add_argument("--clients", type=int, default=100,
                    help="client shards to cache; must be >= a run's --num-trainers")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel workers. CPU tokenization -- no GPU is used")
    ap.add_argument("--partition-method", default=None,
                    help="override the registry's group (the cache key includes it)")
    ap.add_argument("--config", default=BASE_CONFIG,
                    help="hyperparameters template; the registry overrides it")
    ap.add_argument("--force", action="store_true",
                    help="re-tokenize clients that are already cached")
    ap.add_argument("--no-global", action="store_true",
                    help="skip the -1 global test set")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what is missing and its projected size, write nothing")
    a = ap.parse_args()

    spec = dsreg.get(a.dataset, check_h5=True)
    part = a.partition_method or spec.partition_method
    root = dsreg.cache_root()
    ids = ([] if a.no_global else [-1]) + list(range(a.clients))
    paths = {c: dsreg.cache_file(a.dataset, c, part) for c in ids}

    have = {c: os.path.getsize(p) for c, p in paths.items() if os.path.exists(p)}
    todo = ids if a.force else [c for c in ids if c not in have]
    # -1 is the whole test set, not a shard -- averaging it in triples the estimate.
    shards = [s for c, s in have.items() if c >= 0]
    mean = (sum(shards) / len(shards)) if shards else None

    print(f"[cache] {root}")
    print(f"[key]   {os.path.basename(paths[ids[0]]).rsplit('_', 1)[0]}_<client>")
    print(f"[state] dataset={a.dataset} seq={spec.max_seq_length} partition={part}  "
          f"cached {len(have)}/{len(ids)}, {len(todo)} to write")
    if mean:
        print(f"[size]  {mean / 1e6:.0f} MB/client observed -> "
              f"~{mean * len(todo) / 1e9:.1f} GB more")
    elif todo:
        # ~123 MB per 14,000-row yahoo shard at seq 256, scaled by shard rows.
        est = 123e6 * (spec.n_train / a.clients) / 14000 * (spec.max_seq_length / 256)
        print(f"[size]  no sample yet; projected ~{est / 1e6:.0f} MB/client, "
              f"~{est * len(todo) / 1e9:.1f} GB total")
    if not todo:
        print("[done] nothing to do")
        return 0
    if a.dry_run:
        print(f"[dry-run] would write {len(todo)}: "
              f"{todo[:5]}{' ...' if len(todo) > 5 else ''}")
        return 0

    os.makedirs(root, exist_ok=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=max(1, a.jobs), initializer=_init_worker,
                             initargs=(a.config, a.dataset, part)) as pool:
        futs = {pool.submit(_tokenize, c): c for c in todo}
        for f in as_completed(futs):
            c, secs = f.result()
            done += 1
            sz = os.path.getsize(paths[c]) if os.path.exists(paths[c]) else 0
            rate = (time.time() - t0) / done
            print(f"  [{done}/{len(todo)}] client {c:>4}  {secs:6.1f}s  "
                  f"{sz / 1e6:6.1f} MB   eta {rate * (len(todo) - done) / 60:.0f} min",
                  flush=True)

    missing = [c for c in ids if not os.path.exists(paths[c])]
    print(f"\n[done] {len(ids) - len(missing)}/{len(ids)} cached in "
          f"{(time.time() - t0) / 60:.1f} min")
    if missing:
        print(f"[FAIL] still missing: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
