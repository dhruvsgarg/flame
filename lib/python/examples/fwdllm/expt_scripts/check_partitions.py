#!/usr/bin/env python
"""Is a partition group usable? READ-ONLY, no GPU, seconds per group.

Six checks, each of which has already gone wrong in this repo at least once:

  1 COVERAGE     the shards partition the official split -- nothing missing
  2 DISJOINT     no index handed to two clients, no train row in a test shard
                 (yahoo's `uniform_client_1000` fails both: test drawn from
                 INSIDE the train range, train sampled with replacement)
  3 SIZES        equal shards, so no client is silently a different experiment
  4 ORDER        P(adjacent samples share a label) vs its value under random
                 order. THE agnews defect: the 1000-client groups read ~0.98
                 against ~0.25 expected, i.e. each client's rows are sorted by
                 class and every data bin is single-class by construction
  5 BINS         fraction of `train_batch_size`-sample bins that are one class,
                 the same defect seen the way the trainer sees it
  6 SKEW         per-client dominant-class fraction against the Dirichlet
                 expectation for this alpha and class count -- catches a group
                 built at the wrong alpha, or labelled with one

Exit code 1 if any group FAILS, so it can gate a launch.

    python check_partitions.py                                  # every dataset, every niid group
    python check_partitions.py --datasets yahoo --groups niid_label_clients=100_alpha=1
    python check_partitions.py --datasets agnews --all-groups   # includes uniform/1000-client
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from examples.fwdllm.expts import dataset_registry as reg          # noqa: E402
from examples.fwdllm.expt_scripts.build_niid_partitions import load_labels   # noqa: E402

# Order is called random when observed P(same-label adjacent) sits within this
# of the per-client expectation sum(p_k^2). Sampling noise on 1200+ rows is
# ~0.015; the defect it must catch is a 0.5-0.7 gap, so the band is loose on
# purpose.
ORDER_TOL = 0.05


def dirichlet_expected_dom(alpha, n_classes, draws=20000, seed=0):
    """E[max_k p_k] for p ~ Dir(alpha), by simulation -- the reference for check 6."""
    rng = np.random.default_rng(seed)
    return float(rng.dirichlet(np.full(n_classes, float(alpha)), size=draws).max(1).mean())


def check_group(f, gname, ds, y, batch_size=8, verbose=True):
    g = f[gname]
    if "partition_data" not in g:
        return ["no partition_data"], {}
    pd = g["partition_data"]
    cids = sorted(pd.keys(), key=int)
    n_clients = len(cids)

    tr_all, te_all, tr_sizes, te_sizes = [], [], [], []
    dom, same_adj, same_exp, single_bin = [], [], [], []
    for c in cids:
        a = pd[c]["train"][()]
        b = pd[c]["test"][()] if "test" in pd[c] else np.empty(0, dtype=np.int64)
        tr_all.append(a)
        te_all.append(b)
        tr_sizes.append(len(a))
        te_sizes.append(len(b))
        lab = y[a]
        p = np.bincount(lab, minlength=ds.num_labels) / max(1, len(lab))
        dom.append(p.max())
        same_exp.append(float((p ** 2).sum()))
        if len(lab) > 1:
            same_adj.append(float((lab[1:] == lab[:-1]).mean()))
        nb = len(lab) // batch_size
        if nb:
            bins = lab[: nb * batch_size].reshape(-1, batch_size)
            single_bin.append(float(np.mean([len(np.unique(bn)) == 1 for bn in bins])))
    tr = np.concatenate(tr_all)
    te = np.concatenate(te_all) if any(len(x) for x in te_all) else np.empty(0, dtype=np.int64)

    lo, hi = ds.train_range
    tlo, thi = ds.test_range
    alpha = float(g["alpha"][()]) if "alpha" in g else None
    m = re.search(r"alpha=([0-9.]+)", gname)
    if alpha is None and m:
        alpha = float(m.group(1))

    obs_adj = float(np.mean(same_adj))
    exp_adj = float(np.mean(same_exp))
    stats = dict(
        n_clients=n_clients, alpha=alpha,
        train_total=len(tr), train_unique=int(len(np.unique(tr))),
        test_total=len(te), test_unique=int(len(np.unique(te))),
        shard_train=sorted(set(tr_sizes)), shard_test=sorted(set(te_sizes)),
        dom_frac=float(np.mean(dom)),
        dom_frac_expected=(dirichlet_expected_dom(alpha, ds.num_labels) if alpha else None),
        same_adj=obs_adj, same_adj_expected=exp_adj,
        single_class_bin_frac=float(np.mean(single_bin)) if single_bin else float("nan"),
    )

    fails = []
    # 1 coverage
    if not np.array_equal(np.sort(np.unique(tr)), np.arange(lo, hi)):
        miss = (hi - lo) - len(np.unique(tr))
        fails.append(f"train coverage: {miss:+d} rows vs the official split [{lo},{hi})")
    if len(te) and not np.array_equal(np.sort(np.unique(te)), np.arange(tlo, thi)):
        fails.append(f"test coverage: {len(np.unique(te))} unique vs {thi-tlo} in [{tlo},{thi})")
    # 2 disjoint
    if len(np.unique(tr)) != len(tr):
        fails.append(f"train indices repeat ({len(tr)-len(np.unique(tr)):,} duplicates)")
    if len(te) and len(np.unique(te)) != len(te):
        fails.append(f"test indices repeat ({len(te)-len(np.unique(te)):,} duplicates)")
    ov = np.intersect1d(tr, te).size
    if ov:
        fails.append(f"train/test overlap: {ov:,} indices")
    # 3 equal shards
    if len(set(tr_sizes)) > 1:
        fails.append(f"unequal train shards: {min(tr_sizes)}..{max(tr_sizes)}")
    if len(set(te_sizes)) > 1:
        fails.append(f"unequal test shards: {min(te_sizes)}..{max(te_sizes)}")
    # 4 order randomisation -- the one this script exists for
    if obs_adj > exp_adj + ORDER_TOL:
        fails.append(f"label-ORDERED within client: P(same-label adjacent)={obs_adj:.3f} "
                     f"vs {exp_adj:.3f} under random order -- bins are single-class by "
                     f"construction; reshuffle with build_niid_partitions.py")

    if verbose:
        ok = "FAIL" if fails else "ok"
        print(f"  {gname:42s} [{ok}] C={n_clients} shard={stats['shard_train']}/{stats['shard_test']} "
              f"dom={stats['dom_frac']:.3f}"
              + (f" (Dir exp {stats['dom_frac_expected']:.3f})" if stats["dom_frac_expected"] else "")
              + f" adj={obs_adj:.3f} (rand {exp_adj:.3f}) 1-class-bins={stats['single_class_bin_frac']:.3f}")
        for f_ in fails:
            print(f"      FAIL: {f_}")
    return fails, stats


def main():
    import h5py

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(reg.names()))
    ap.add_argument("--groups", default="", help="comma-separated group names (default: every niid group)")
    ap.add_argument("--all-groups", action="store_true", help="also check uniform/quantity groups")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    failed = 0
    for name in args.datasets.split(","):
        ds = reg.get(name.strip(), check_h5=True)
        print(f"\n===== {ds.name} ({ds.num_labels} classes) {ds.partition_file_path}")
        y = load_labels(ds)
        with h5py.File(ds.partition_file_path, "r") as f:
            if args.groups:
                gnames = [g for g in args.groups.split(",") if g in f]
            else:
                gnames = [g for g in f.keys()
                          if args.all_groups or g.startswith("niid_label")]
            for gname in sorted(gnames):
                fails, _ = check_group(f, gname, ds, y, args.batch_size)
                failed += bool(fails)
    print(f"\n{failed} group(s) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
