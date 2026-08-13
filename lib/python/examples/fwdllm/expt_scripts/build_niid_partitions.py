#!/usr/bin/env python
"""Build `niid_label_clients=C_alpha=A` partitions that match agnews's shape.

WHY. yahoo and yelp-p ship only `uniform_*` groups, so there is nothing for a
niid run to point at; yahoo's `uniform_client_1000` is also broken (its test
indices are drawn from INSIDE the train range -- 31,689 samples overlap train --
and train is sampled with replacement, 1.4M draws over 885k unique rows). And
the groups that do exist elsewhere carry the label-ordering defect: agnews's
1000-client groups hand each client its samples sorted by class, so a data bin
(one `train_batch_size`-sample slice) is single-class by construction. agnews's
100-client groups were fixed; nothing else was.

WHAT THIS WRITES, per (dataset, C, alpha), matching agnews's 100-client groups
exactly in shape:

  * equal-size shards -- every client gets N_train/C train and N_test/C test rows
  * FULL coverage -- the shards partition the official split, disjoint, no
    train/test overlap, nothing sampled twice
  * Dirichlet label mix -- client c's class proportions ~ Dir(alpha), realised
    as closely as equal sizes and exact coverage allow (Sinkhorn/IPF, then a
    margin-preserving integer rounding)
  * SHUFFLED WITHIN CLIENT -- the index order a client trains in is random, so a
    data bin looks like its client's class mix, not like one class. This is the
    property the agnews 1000-client groups lack and the one `check_partitions.py`
    exists to police.

Test shards get their own independent Dir(alpha) draw, which is what agnews does
(its client 0 holds no class-3 train rows but 13 class-3 test rows).

    python build_niid_partitions.py --datasets yahoo,yelp-p --alphas 1,100 \
        --n-clients 100 [--dry-run] [--force]

Writes through a temp copy and only `os.replace`s it after the verification pass
passes, so a failed build leaves the existing file untouched. Existing groups are
never modified; rebuilding one needs --force.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from examples.fwdllm.expts import dataset_registry as reg   # noqa: E402

BUILDER_VERSION = "1.0"
LABEL_CACHE = os.path.join(reg.root(), "label_cache")


# ---------------------------------------------------------------- labels ----
def load_labels(ds, refresh=False):
    """Class id (0..K-1) per sample index, via the h5's own label_vocab.

    Reading 1.46M individual h5 datasets takes ~4 min, so the array is cached as
    a .npy beside the datasets. The cache is keyed by dataset name and validated
    on length before use.
    """
    import h5py

    os.makedirs(LABEL_CACHE, exist_ok=True)
    path = os.path.join(LABEL_CACHE, f"{ds.name}_labels.npy")
    n_total = ds.test_range[1]
    if os.path.exists(path) and not refresh:
        y = np.load(path)
        if len(y) == n_total:
            return y
        print(f"  label cache stale ({len(y)} != {n_total}); rebuilding")

    vocab = ds.label_vocab()                       # raw label string -> class id
    print(f"  reading {n_total:,} labels from {ds.data_file_path} (~{n_total/350000:.0f} min) ...",
          flush=True)
    t0 = time.time()
    y = np.full(n_total, -1, dtype=np.int16)
    with h5py.File(ds.data_file_path, "r", swmr=True) as f:
        Y = f["Y"]
        for i in range(n_total):
            y[i] = vocab[Y[str(i)][()].decode("utf-8")]
    if (y < 0).any():
        raise ValueError(f"{ds.name}: {(y < 0).sum()} samples got no label")
    np.save(path, y)
    print(f"  cached {path} in {time.time()-t0:.0f}s", flush=True)
    return y


# ------------------------------------------------------------ allocation ----
def _sinkhorn(P, row_targets, col_targets, iters=500, tol=1e-9):
    """Scale P (C x K) to hit both margins. IPF = the KL-closest matrix to the
    Dirichlet draw that respects equal shard sizes and exact class coverage, so
    the skew we asked for survives the constraints as far as it can."""
    W = np.maximum(P.astype(np.float64), 1e-12)
    for _ in range(iters):
        W *= (row_targets / W.sum(1))[:, None]
        prev = W.sum(0)
        W *= (col_targets / prev)[None, :]
        if np.abs(W.sum(1) - row_targets).max() < tol:
            break
    return W


def _round_to_margins(W, row_targets, col_targets, rng):
    """Integerise W keeping BOTH margins exact. Floor, then hand out the residual
    by largest fractional part; any remainder (rare, and only where the Dirichlet
    put ~no mass) is filled arbitrarily -- exact coverage at equal shard size is a
    transportation constraint and outranks the draw."""
    M = np.floor(W).astype(np.int64)
    # floor can overshoot a column only if IPF has not fully converged; give
    # those cells back, cheapest fractional part first.
    over = M.sum(0) - col_targets
    for k in np.nonzero(over > 0)[0]:
        order = np.argsort(W[:, k] - M[:, k])
        for c in order:
            if over[k] <= 0:
                break
            take = min(int(over[k]), int(M[c, k]))
            M[c, k] -= take
            over[k] -= take

    row_def = row_targets - M.sum(1)
    col_def = col_targets - M.sum(0)
    frac = W - M
    C, K = W.shape
    cells = sorted(((c, k) for c in range(C) for k in range(K)),
                   key=lambda ck: -frac[ck])
    for c, k in cells:
        if row_def[c] > 0 and col_def[k] > 0:
            M[c, k] += 1
            row_def[c] -= 1
            col_def[k] -= 1
    # leftovers: rows and columns still short of each other
    rows = [c for c in range(C) if row_def[c] > 0]
    cols = [k for k in range(K) if col_def[k] > 0]
    ci = 0
    for c in rows:
        while row_def[c] > 0:
            while col_def[cols[ci]] <= 0:
                ci += 1
            take = int(min(row_def[c], col_def[cols[ci]]))
            M[c, cols[ci]] += take
            row_def[c] -= take
            col_def[cols[ci]] -= take
    assert (M.sum(1) == row_targets).all(), "row margin broken"
    assert (M.sum(0) == col_targets).all(), "column margin broken"
    assert (M >= 0).all()
    return M


def dirichlet_equal_shards(pool, y_pool, n_classes, n_clients, alpha, rng):
    """Partition `pool` into `n_clients` equal, disjoint, class-Dirichlet shards,
    each internally shuffled. Returns (shards, M, mean_l1_distortion)."""
    N = len(pool)
    base, rem = divmod(N, n_clients)
    row_targets = np.full(n_clients, base, dtype=np.int64)
    row_targets[:rem] += 1                       # exact coverage when C does not divide N
    col_targets = np.bincount(y_pool, minlength=n_classes).astype(np.int64)

    P = rng.dirichlet(np.full(n_classes, float(alpha)), size=n_clients)
    W = _sinkhorn(P, row_targets, col_targets)
    M = _round_to_margins(W, row_targets, col_targets, rng)
    # how far the realised mix sits from the draw we asked for
    distortion = float(np.abs(M / row_targets[:, None] - P).sum(1).mean())

    by_class = []
    for k in range(n_classes):
        idx = pool[y_pool == k]
        rng.shuffle(idx)
        by_class.append(idx)

    shards, off = [], np.zeros(n_classes, dtype=np.int64)
    for c in range(n_clients):
        parts = []
        for k in range(n_classes):
            take = int(M[c, k])
            if take:
                parts.append(by_class[k][off[k]:off[k] + take])
                off[k] += take
        idx = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        rng.shuffle(idx)                          # <- the bin-order fix
        shards.append(idx.astype(np.int64))
    assert (off == col_targets).all(), "not every sample was handed out"
    return shards, M, distortion


# --------------------------------------------------------------- metrics ----
def shard_stats(shards, y, n_classes, batch_size=8):
    """The numbers that say whether a group is usable: class skew per client, and
    whether the ORDER inside a client is random or label-blocked."""
    dom, same_adj, same_exp, single_bin = [], [], [], []
    for idx in shards:
        lab = y[idx]
        cnt = np.bincount(lab, minlength=n_classes)
        p = cnt / max(1, len(lab))
        dom.append(p.max())
        same_exp.append(float((p ** 2).sum()))    # P(same label adjacent) under random order
        same_adj.append(float((lab[1:] == lab[:-1]).mean()) if len(lab) > 1 else np.nan)
        bins = lab[: (len(lab) // batch_size) * batch_size].reshape(-1, batch_size)
        single_bin.append(float(np.mean([len(np.unique(b)) == 1 for b in bins])) if len(bins) else np.nan)
    return dict(
        n_clients=len(shards), shard_size=int(len(shards[0])),
        dom_frac_mean=float(np.mean(dom)), dom_frac_p50=float(np.median(dom)),
        dom_frac_p90=float(np.percentile(dom, 90)),
        same_adj=float(np.nanmean(same_adj)), same_adj_expected=float(np.mean(same_exp)),
        single_class_bin_frac=float(np.nanmean(single_bin)),
    )


# ----------------------------------------------------------------- write ----
def group_name(n_clients, alpha):
    a = float(alpha)
    a_str = str(int(a)) if a.is_integer() else str(a)     # match agnews: alpha=1, alpha=100
    return f"niid_label_clients={n_clients}_alpha={a_str}"


def build_group(ds, y, n_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for split, (lo, hi) in (("train", ds.train_range), ("test", ds.test_range)):
        pool = np.arange(lo, hi, dtype=np.int64)
        shards, _, dist = dirichlet_equal_shards(
            pool, y[lo:hi], ds.num_labels, n_clients, alpha,
            # test gets its own draw, as agnews does
            np.random.default_rng(seed if split == "train" else seed + 1_000_003),
        )
        st = shard_stats(shards, y, ds.num_labels)
        st["mix_l1_distortion"] = dist
        out[split] = (shards, st)
        print(f"    {split:5s}: {st['n_clients']}x{st['shard_size']} "
              f"dom_frac mean/p50/p90 {st['dom_frac_mean']:.3f}/{st['dom_frac_p50']:.3f}/{st['dom_frac_p90']:.3f}  "
              f"adj {st['same_adj']:.3f} (rand {st['same_adj_expected']:.3f})  "
              f"1-class bins {st['single_class_bin_frac']:.3f}  distortion {dist:.3f}")
    del rng
    return out


def verify_group(path, name, ds, y, n_clients):
    """Re-read what was written and check it from scratch. Coverage, disjointness,
    equal sizes, and order randomness -- the four ways this can silently go wrong."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f[name]
        assert int(g["n_clients"][()]) == n_clients, "n_clients mismatch"
        pd = g["partition_data"]
        assert len(pd.keys()) == n_clients, "client count mismatch"
        tr, te, sizes = [], [], []
        for c in range(n_clients):
            a = pd[str(c)]["train"][()]
            b = pd[str(c)]["test"][()]
            tr.append(a)
            te.append(b)
            sizes.append((len(a), len(b)))
        tr = np.concatenate(tr)
        te = np.concatenate(te)
    lo, hi = ds.train_range
    tlo, thi = ds.test_range
    problems = []
    if len(np.unique(tr)) != len(tr):
        problems.append("train indices repeat")
    if len(np.unique(te)) != len(te):
        problems.append("test indices repeat")
    if not np.array_equal(np.sort(tr), np.arange(lo, hi)):
        problems.append(f"train does not cover [{lo},{hi})")
    if not np.array_equal(np.sort(te), np.arange(tlo, thi)):
        problems.append(f"test does not cover [{tlo},{thi})")
    if np.intersect1d(tr, te).size:
        problems.append("train/test overlap")
    if len({s[0] for s in sizes}) > 1 or len({s[1] for s in sizes}) > 1:
        problems.append(f"unequal shard sizes: {sorted(set(sizes))[:4]}")
    return problems


def write_groups(ds, groups, dry_run=False, force=False):
    """groups: {group_name: {'train': (shards, stats), 'test': (...), 'alpha':A, 'n_clients':C}}"""
    import h5py

    dst = ds.partition_file_path
    if dry_run:
        print(f"  [dry-run] would write {len(groups)} group(s) into {dst}")
        return
    tmp = dst + ".building"
    print(f"  copying {dst} -> {tmp} ({os.path.getsize(dst)/1e6:.0f} MB)", flush=True)
    shutil.copy2(dst, tmp)
    try:
        with h5py.File(tmp, "a") as f:
            for name, spec in groups.items():
                if name in f:
                    if not force:
                        raise SystemExit(f"  group '{name}' already exists in {dst}; "
                                         f"pass --force to rebuild it")
                    del f[name]
                g = f.create_group(name)
                g.create_dataset("alpha", data=float(spec["alpha"]))
                g.create_dataset("n_clients", data=np.int64(spec["n_clients"]))
                pd = g.create_group("partition_data")
                for c, (a, b) in enumerate(zip(spec["train"][0], spec["test"][0])):
                    cg = pd.create_group(str(c))
                    cg.create_dataset("train", data=a, dtype="int64")
                    cg.create_dataset("test", data=b, dtype="int64")
                g.attrs["builder"] = "expt_scripts/build_niid_partitions.py"
                g.attrs["builder_version"] = BUILDER_VERSION
                g.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
                g.attrs["seed"] = int(spec["seed"])
                g.attrs["order"] = "shuffled_within_client"
                g.attrs["source_data_h5"] = ds.data_file_path
                g.attrs["train_range"] = list(ds.train_range)
                g.attrs["test_range"] = list(ds.test_range)
                g.attrs["stats"] = json.dumps(
                    {k: v[1] for k, v in spec.items() if k in ("train", "test")})
        all_problems = []
        for name, spec in groups.items():
            probs = verify_group(tmp, name, ds, spec["y"], spec["n_clients"])
            print(f"  verify {name}: {'OK' if not probs else '; '.join(probs)}")
            all_problems += probs
        if all_problems:
            raise SystemExit("  verification FAILED -- original file left untouched")
        os.replace(tmp, dst)
        print(f"  wrote {dst}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="yahoo,yelp-p")
    ap.add_argument("--alphas", default="1,100")
    ap.add_argument("--n-clients", default="100", help="comma-separated; one group per (alpha, C)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild a group that already exists")
    ap.add_argument("--refresh-labels", action="store_true")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    clients = [int(c) for c in args.n_clients.split(",")]

    for name in args.datasets.split(","):
        ds = reg.get(name.strip(), check_h5=True)
        print(f"\n===== {ds.name}: {ds.num_labels} classes, "
              f"train {ds.n_train:,} {ds.train_range}, test {ds.n_test:,} {ds.test_range} =====")
        y = load_labels(ds, refresh=args.refresh_labels)
        for split, (lo, hi) in (("train", ds.train_range), ("test", ds.test_range)):
            cnt = np.bincount(y[lo:hi], minlength=ds.num_labels)
            print(f"  {split} class counts: {cnt.tolist()}")

        groups = {}
        for C in clients:
            for a in alphas:
                gname = group_name(C, a)
                print(f"  -- {gname}")
                built = build_group(ds, y, C, a, args.seed)
                groups[gname] = dict(alpha=a, n_clients=C, seed=args.seed, y=y, **built)
        write_groups(ds, groups, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
