#!/usr/bin/env python3
"""Rung 1 of buildplan §9: what can this data path reach with an EXACT gradient?

The discriminating test for the yahoo gap. B-1's backprop reference reaches 0.73
on yahoo where the FL arms reach 0.30, but the two are not comparable: B-1's rig
(`scripts/probe_inflation_damage.py`) trains on half the *test-global* tensor and
reads accuracy off `test_global[:2000]`, which is per-client shards in client
order -- a Dirichlet-skewed slice. This script removes both differences.

  * trains on the SAME tensors the trainers consume -- client shards through
    `TextClassificationDataManager.load_federated_data`, the exact call
    `trainer/main.py` makes, with the run's own partition group;
  * respects `total_data_bins`, so it sees exactly the rows an FL arm can reach;
  * evaluates on the SAME `test_global` the aggregator's `agg_eval` uses, and
    additionally on the fixed 10k subsample `eval_max_samples` selects, so the
    subsample's offset is measured rather than assumed;
  * changes ONE thing: AdamW on the exact gradient instead of pooled forward
    differences.

**Pre-registered reading.** ~0.70 on yahoo clears the whole data path -- token-
ization, label vocab, `max_seq_length`, the h5 ranges, the partition, the loader
-- and makes the FL gap purely an optimization-budget question (P4.8). ~0.30
indicts the path, and every yahoo FL arm to date is measuring the wrong thing.

    ./probe_backprop_ceiling.py --config <aggregator_config.json> [--clients 20]
    ./probe_backprop_ceiling.py --config ... --dataset yahoo --epochs 3

Costs one GPU and ~30 min at `--clients 20` on yahoo. Rung 2 (P7): imports
production code, writes nothing, touches no yaml on the critical path.
"""
import argparse
import collections
import json
import math
import os
import sys
import time
import types

import torch
from torch.nn import CrossEntropyLoss

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from examples.fwdllm.data_manager.base_data_manager import (  # noqa: E402
    BaseDataManager,
)
from examples.fwdllm.data_manager.text_classification_data_manager import (  # noqa: E402
    TextClassificationDataManager,
)
from examples.fwdllm.data_preprocessing.text_classification_preprocessor import (  # noqa: E402
    TLMPreprocessor,
)
from examples.fwdllm.expts.dataset_registry import (  # noqa: E402
    get as ds_get, hyperparameter_overrides, total_data_bins,
)
from examples.fwdllm.trainer.model_args_builder import build_model_args  # noqa: E402
from examples.fwdllm.expts.initializer import create_model  # noqa: E402
from examples.fwdllm.trainer.forward_training.tc_transformer_trainer_distribute import (  # noqa: E402
    ForwardTextClassificationTrainer,
)

EVAL_CHUNK = 128


@torch.no_grad()
def evaluate(model, ids, labels, dev, num_labels):
    """Accuracy / loss / the same collapse fingerprints agg_eval emits."""
    model.eval()
    ok, loss_sum, n = 0, 0.0, ids.shape[0]
    preds = []
    lf = CrossEntropyLoss(reduction="sum")
    for s in range(0, n, EVAL_CHUNK):
        out = model(ids[s:s + EVAL_CHUNK].to(dev))
        lg = (out.logits if hasattr(out, "logits") else out[0]).float()
        y = labels[s:s + EVAL_CHUNK].view(-1).to(dev)
        loss_sum += float(lf(lg.view(-1, num_labels), y))
        p = lg.argmax(-1)
        ok += int((p == y).sum())
        preds += p.cpu().tolist()
    c = collections.Counter(preds)
    return dict(acc=ok / n, loss=loss_sum / n,
                top_class_share=max(c.values()) / n, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="an aggregator_config.json from a real run")
    ap.add_argument("--dataset", default=None,
                    help="override the config's dataset (registry name)")
    ap.add_argument("--clients", type=int, default=20,
                    help="how many client shards to pool (0 = all)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-subsample", type=int, default=10000,
                    help="also report accuracy on this fixed subsample (0 = skip)")
    ap.add_argument("--all-bins", action="store_true",
                    help="use every batch in a shard, not just the first "
                         "total_data_bins (which is all an FL arm can reach)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hp = json.load(open(a.config))["hyperparameters"]
    if a.dataset:
        hp.update(hyperparameter_overrides(a.dataset))
    name = hp["dataset"]
    spec = ds_get(name)
    hp = types.SimpleNamespace(**hp)

    attrs = BaseDataManager.load_attributes(hp.data_file_path)
    num_labels = len(attrs["label_vocab"])
    margs = build_model_args(hp, num_labels)
    _, model, tok = create_model(margs, formulation="classification")
    # The trainer's __init__ replaces pre_classifier with nn.Sequential()
    # (tc_transformer_trainer_distribute.py:217) -- that is what sets `p` to
    # 450,340 rather than 1,040,932, so the rig must do it too or it is training
    # a bigger model than any FL arm ever runs.
    ForwardTextClassificationTrainer(margs, 0, model, None, None, "rig")
    model.to(dev)
    p_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rig] dataset={name} num_labels={num_labels} p={p_tr} "
          f"seq={margs.max_seq_length} dev={dev}")
    if num_labels != spec.num_labels:
        print(f"  WARNING: h5 says {num_labels} labels, registry says "
              f"{spec.num_labels}")

    pre = TLMPreprocessor(args=margs, label_vocab=attrs["label_vocab"],
                          tokenizer=tok)

    # ---- the FL data path, one client at a time (trainer/main.py:158-176) ----
    n_clients_cfg = int(getattr(hp, "client_num_in_total", 0) or 100)
    bins = total_data_bins(name, n_clients_cfg, int(hp.train_batch_size))
    want = a.clients or n_clients_cfg
    print(f"[data] {want} of {n_clients_cfg} client shards, "
          f"{bins} bins x {hp.train_batch_size} each"
          + ("" if a.all_bins else "  (the FL-reachable prefix)"))

    # The GLOBAL test set comes from the aggregator's own role (process_id=0);
    # a trainer (process_id=1) is handed test_data_local_dict and gets None here.
    hp.client_idx = 0
    test_global = TextClassificationDataManager(
        hp, margs, pre, 0, hp.data_loader_num_workers
    ).load_federated_data(process_id=0, client_idx=0)[2]

    ids_parts, lab_parts = [], []
    t0 = time.time()
    for c in range(want):
        hp.client_idx = c
        dm = TextClassificationDataManager(hp, margs, pre, 1,
                                           hp.data_loader_num_workers)
        train_local = dm.load_federated_data(process_id=1, client_idx=c)[4]
        batches = [b for b in train_local[c]]
        if not a.all_bins:
            batches = batches[:bins]
        for b in batches:
            ids_parts.append(b[1])
            lab_parts.append(b[4])
        if (c + 1) % 5 == 0:
            print(f"  {c + 1}/{want} shards loaded ({time.time() - t0:.0f}s)")

    X = torch.cat(ids_parts)
    Y = torch.cat(lab_parts).view(-1)
    counts = torch.bincount(Y, minlength=num_labels).tolist()
    print(f"[data] pooled {X.shape[0]:,} train rows; class counts {counts}")

    tX, tY = test_global.dataset.tensors[1], test_global.dataset.tensors[4]
    print(f"[data] test_global {tX.shape[0]:,} rows (the same set agg_eval uses)")
    sub = None
    if a.eval_subsample and a.eval_subsample < tX.shape[0]:
        g = torch.Generator().manual_seed(20260810)      # same seed the run uses
        sub = torch.randperm(tX.shape[0], generator=g)[:a.eval_subsample]

    # ---- one thing changes: AdamW on the exact gradient ---------------------
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr)
    g2 = torch.Generator().manual_seed(1)
    lf = CrossEntropyLoss()
    base = evaluate(model, tX, tY, dev, num_labels)
    print(f"\n  epoch  0 (untrained): acc={base['acc']:.4f} "
          f"loss={base['loss']:.3f} top_class={base['top_class_share']:.3f}  "
          f"(chance {1.0 / num_labels:.3f})")

    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(X.shape[0], generator=g2)
        t1, seen = time.time(), 0
        for s in range(0, len(perm), a.batch):
            sel = perm[s:s + a.batch]
            out = model(X[sel].to(dev))
            lg = out.logits if hasattr(out, "logits") else out[0]
            lf(lg.view(-1, num_labels), Y[sel].to(dev)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            seen += len(sel)
        full = evaluate(model, tX, tY, dev, num_labels)
        line = (f"  epoch {ep + 1:>2}: acc={full['acc']:.4f} "
                f"loss={full['loss']:.3f} top_class={full['top_class_share']:.3f} "
                f"({seen:,} rows, {time.time() - t1:.0f}s)")
        if sub is not None:
            s10 = evaluate(model, tX[sub], tY[sub], dev, num_labels)
            line += (f"   | {a.eval_subsample} subsample acc={s10['acc']:.4f} "
                     f"(offset {s10['acc'] - full['acc']:+.4f})")
        print(line)

    print(f"\n[verdict] pre-registered (buildplan §9): ~0.70 on yahoo clears the "
          f"data path\n          and makes the FL gap an optimization-budget "
          f"question; ~0.30 indicts the path.")


if __name__ == "__main__":
    main()
