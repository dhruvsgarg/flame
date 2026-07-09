#!/usr/bin/env python
"""H0 — partition & data-bin diagnostic (fluxtune stability track).

READ-ONLY. Checks whether the data a trainer sees at each `data_id` focuses on one
class, and whether that predicts the single-class collapses seen in run telemetry.

Measures, per Dirichlet alpha (100-client distribution):

  Layer A  per-CLIENT class skew (from the partition h5 + labels): dominant-class
           fraction, #classes present, normalized entropy -- "how non-IID is each
           client" across alpha=0.1 / 1 / 100.

  Layer B  per-BIN narrowness (a data_id = one `train_batch_size`-sample bin):
           per-bin dominant frac / entropy; per-client single-class-bin fraction.
           Order source is FAITHFUL (the frozen per-client pickle cache the run
           trained on) when present, else raw partition index order (flagged); for
           alpha=1 both are reported to expose the read_instance_from_h5 shuffle.

  Layer C  per-DATA_ID pooled bias (proxy for what the aggregator sees at commit):
           pool bin #k across clients -> pooled dominant frac / entropy. Overlaid
           (alpha=1) against the OBSERVED collapse data_ids from agg_eval telemetry
           to test whether low-entropy data_ids predict collapse.

Usage:
  python diagnose_partition_binning.py                      # all defaults
  python diagnose_partition_binning.py --alphas 0.1,1,100 --plots
"""
import argparse, json, math, os, pickle, sys
from collections import Counter

import numpy as np

# ---- defaults resolved from the run's aggregator_config.json (the /coc/scratch h5s) ----
DEF_PART = "/coc/scratch/dgarg/fl_datasets/fwdllm/fednlp_data/partition_files/agnews_partition.h5"
DEF_DATA = "/coc/scratch/dgarg/fl_datasets/fwdllm/fednlp_data/data_files/agnews_data.h5"
# cache lives with the datasets on scratch; override with --cache-dir if elsewhere.
DEF_CACHE = "/coc/scratch/dgarg/fl_datasets/fwdllm/fednlp_data/cache_dir"
DEF_RUN = ("/home/dgarg39/flame/lib/python/examples/fwdllm/experiments/"
           "run_20260708_025543_fluxtune_n100_smoke_syn_0_real")
CACHE_TMPL = ("distilbert_distilbert-base-uncased_cached_192_ClassificationModel_"
              "agnews_niid_label_clients=100_alpha={alpha}_{cid}")
N_CLASSES = 4  # AG News


def norm_entropy(counts):
    """Shannon entropy normalized to [0,1] over N_CLASSES (1 = uniform, 0 = single class)."""
    tot = sum(counts.values())
    if tot == 0:
        return float("nan")
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / tot
            h -= p * math.log(p)
    return h / math.log(N_CLASSES)


def dom_frac(counts):
    tot = sum(counts.values())
    return (max(counts.values()) / tot) if tot else float("nan")


def load_label_map(data_h5):
    """Full {index:int label} from the data h5 Y group. Built once, reused for index-order alphas."""
    import h5py
    print(f"  reading labels from {data_h5} ...", flush=True)
    with h5py.File(data_h5, "r") as f:
        Y = f["Y"]
        m = {}
        for k in Y.keys():
            m[int(k)] = int(Y[k][()].decode("utf-8"))
    print(f"  loaded {len(m)} labels; classes present: {sorted(set(m.values()))}", flush=True)
    return m


def client_labels_faithful(cache_dir, alpha, cid):
    """Labels in the frozen order the run trained on (from the pickle cache), or None."""
    path = os.path.join(cache_dir, CACHE_TMPL.format(alpha=alpha, cid=cid))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as h:
        tup = pickle.load(h)
    train_examples = tup[0]
    return [int(e.label) for e in train_examples]


def client_labels_index(partition_h5_group, cid, label_map):
    """Labels in raw partition-index order (fallback when no cache)."""
    idx = partition_h5_group["partition_data"][str(cid)]["train"][()]
    return [label_map[int(i)] for i in idx]


def analyze_alpha(alpha, part_h5, label_map, cache_dir, batch_size, n_clients):
    import h5py
    group_name = f"niid_label_clients={n_clients}_alpha={alpha}"
    per_client = []            # dicts of client-level metrics
    per_client_bins = []       # list of list-of-bin-label-Counters (for Layer C pooling)
    order_src = Counter()
    with h5py.File(part_h5, "r") as f:
        if group_name not in f:
            raise SystemExit(f"partition group '{group_name}' not in {part_h5}")
        g = f[group_name]
        cids = sorted((int(c) for c in g["partition_data"].keys()))
        for cid in cids:
            labels = client_labels_faithful(cache_dir, alpha, cid)
            if labels is not None:
                order_src["faithful(cache)"] += 1
            else:
                labels = client_labels_index(g, cid, label_map)
                order_src["index(partition)"] += 1
            cc = Counter(labels)
            # bins
            bins = [Counter(labels[i:i + batch_size]) for i in range(0, len(labels), batch_size)]
            single = sum(1 for b in bins if len([1 for v in b.values() if v > 0]) == 1)
            per_client.append(dict(
                cid=cid, n=len(labels), n_classes=len([1 for v in cc.values() if v > 0]),
                dom_frac=dom_frac(cc), entropy=norm_entropy(cc),
                n_bins=len(bins), single_class_bin_frac=single / len(bins) if bins else float("nan"),
                mean_bin_entropy=float(np.nanmean([norm_entropy(b) for b in bins])) if bins else float("nan"),
                class_counts={int(k): int(v) for k, v in cc.items()},
            ))
            per_client_bins.append(bins)
    return group_name, per_client, per_client_bins, order_src


def pooled_per_dataid(per_client_bins, max_k=None):
    """Layer C: pool bin #k across all clients -> pooled Counter per data_id k."""
    if max_k is None:
        max_k = max(len(b) for b in per_client_bins)
    pooled = []
    for k in range(max_k):
        agg = Counter()
        n_contrib = 0
        for bins in per_client_bins:
            if k < len(bins):
                agg += bins[k]
                n_contrib += 1
        pooled.append(dict(data_id=k, n_contrib=n_contrib,
                           dom_frac=dom_frac(agg), entropy=norm_entropy(agg)))
    return pooled


def cohort_sampled(per_client_bins, K, R, seed=0):
    """Realistic aggregation batch: for each data_id, draw R random K-client cohorts and
    pool their bin #k. Returns arrays of pooled entropy and dominant-fraction over all
    (data_id, cohort) draws — the class bias an agg_goal=K commit actually experiences."""
    rng = np.random.default_rng(seed)
    n_clients = len(per_client_bins)
    max_k = max(len(b) for b in per_client_bins)
    ent, dfr = [], []
    for k in range(max_k):
        avail = [ci for ci in range(n_clients) if k < len(per_client_bins[ci])]
        if len(avail) < K:
            continue
        for _ in range(R):
            coh = rng.choice(avail, size=K, replace=False)
            agg = Counter()
            for ci in coh:
                agg += per_client_bins[ci][k]
            ent.append(norm_entropy(agg))
            dfr.append(dom_frac(agg))
    return np.array(ent), np.array(dfr)


def observed_collapse(run_dir):
    """Round-1 agg_eval: (data_id -> accuracy) and the set of single-class-collapse data_ids."""
    tele = os.path.join(run_dir, "telemetry")
    agg = [f for f in os.listdir(tele) if f.startswith("aggregator")]
    if not agg:
        return None, None
    acc_by_did, collapse = {}, set()
    with open(os.path.join(tele, agg[0])) as fh:
        for line in fh:
            if '"agg_eval"' not in line:
                continue
            d = json.loads(line)
            if d.get("round") != 1:
                continue
            did, a, m = d.get("data_id"), d["test-accuracy"], d.get("mcc", 0.0)
            acc_by_did[did] = a
            if abs(a - 0.25) < 0.006 and abs(m) < 0.02:
                collapse.add(did)
    return acc_by_did, collapse


def pct(vals, p):
    v = [x for x in vals if not math.isnan(x)]
    return float(np.percentile(v, p)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default=DEF_PART)
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--cache-dir", default=DEF_CACHE)
    ap.add_argument("--run-dir", default=DEF_RUN, help="run for the alpha=1 collapse overlay (Layer C)")
    ap.add_argument("--alphas", default="0.1,1,100")
    ap.add_argument("--n-clients", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--agg-goal", type=int, default=10, help="clients pooled per commit (Layer C')")
    ap.add_argument("--cohort-draws", type=int, default=200, help="random cohorts sampled per data_id")
    ap.add_argument("--overlay-alpha", default="1", help="which alpha to overlay against run telemetry")
    ap.add_argument("--out", default=None, help="dir for csv/json/plots (default: ./_diag_partition)")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--dist", action="store_true",
                    help="sample→trainer→bin distribution report + heatmaps (heterogeneity study)")
    args = ap.parse_args()

    alphas = [a.strip() for a in args.alphas.split(",")]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_partition")
    os.makedirs(out, exist_ok=True)

    # labels only needed if some alpha lacks a cache (index-order fallback)
    need_labels = any(
        not os.path.exists(os.path.join(args.cache_dir, CACHE_TMPL.format(alpha=a, cid=0)))
        for a in alphas)
    label_map = load_label_map(args.data) if need_labels else {}

    summary = {}
    pooled_all = {}
    bins_all = {}
    for a in alphas:
        print(f"\n===== alpha = {a} (group niid_label_clients={args.n_clients}_alpha={a}) =====")
        gname, pc, pcb, osrc = analyze_alpha(a, args.partition, label_map, args.cache_dir,
                                             args.batch_size, args.n_clients)
        print(f"  order source: {dict(osrc)}")
        df = [c["dom_frac"] for c in pc]
        ncl = [c["n_classes"] for c in pc]
        ent = [c["entropy"] for c in pc]
        scb = [c["single_class_bin_frac"] for c in pc]
        mbe = [c["mean_bin_entropy"] for c in pc]
        nsz = [c["n"] for c in pc]
        print(f"  Layer A  per-client (n={len(pc)} clients, samples/client p10/50/90="
              f"{pct(nsz,10):.0f}/{pct(nsz,50):.0f}/{pct(nsz,90):.0f}):")
        print(f"    dominant-class fraction  p10/50/90 = {pct(df,10):.3f} / {pct(df,50):.3f} / {pct(df,90):.3f}"
              f"   (mean {np.mean(df):.3f})")
        print(f"    #classes present         p10/50/90 = {pct(ncl,10):.1f} / {pct(ncl,50):.1f} / {pct(ncl,90):.1f}")
        print(f"    normalized entropy       p10/50/90 = {pct(ent,10):.3f} / {pct(ent,50):.3f} / {pct(ent,90):.3f}")
        print(f"  Layer B  per-bin (batch_size={args.batch_size}):")
        print(f"    single-class-bin frac    p10/50/90 = {pct(scb,10):.3f} / {pct(scb,50):.3f} / {pct(scb,90):.3f}"
              f"   (mean {np.mean(scb):.3f})")
        print(f"    mean bin entropy         p10/50/90 = {pct(mbe,10):.3f} / {pct(mbe,50):.3f} / {pct(mbe,90):.3f}")
        pooled = pooled_per_dataid(pcb)
        pooled_all[a] = pooled
        bins_all[a] = pcb
        pe = [p["entropy"] for p in pooled]
        pdf = [p["dom_frac"] for p in pooled]
        print(f"  Layer C  per-data_id pooled over cohort ({len(pooled)} data_ids):")
        print(f"    pooled entropy           p10/50/90 = {pct(pe,10):.3f} / {pct(pe,50):.3f} / {pct(pe,90):.3f}")
        print(f"    pooled dominant frac     p10/50/90 = {pct(pdf,10):.3f} / {pct(pdf,50):.3f} / {pct(pdf,90):.3f}")
        # realistic aggregation batch: agg_goal=K random cohorts
        ce, cd = cohort_sampled(pcb, args.agg_goal, args.cohort_draws)
        frac_biased = float(np.mean(cd > 0.5))
        print(f"  Layer C' realistic cohort (K={args.agg_goal} clients/commit, {args.cohort_draws} draws/data_id):")
        print(f"    cohort pooled entropy    p10/50/90 = {pct(list(ce),10):.3f} / {pct(list(ce),50):.3f} / {pct(list(ce),90):.3f}")
        print(f"    cohort dominant frac     p10/50/90 = {pct(list(cd),10):.3f} / {pct(list(cd),50):.3f} / {pct(list(cd),90):.3f}")
        print(f"    P(cohort dominant frac > 0.5) = {frac_biased:.3f}  (fraction of commits that are majority-one-class)")
        summary[a] = dict(group=gname, order=dict(osrc), per_client=pc, pooled=pooled,
                          cohort=dict(K=args.agg_goal, entropy_p50=pct(list(ce),50),
                                      dom_p50=pct(list(cd),50), frac_biased=frac_biased))

    # ---- Layer C overlay: predicted-biased data_ids vs observed collapses (overlay alpha) ----
    oa = args.overlay_alpha
    if oa in pooled_all and os.path.isdir(args.run_dir):
        print(f"\n===== Layer C overlay: alpha={oa} predicted bias vs OBSERVED collapse ({os.path.basename(args.run_dir)}) =====")
        acc_by_did, collapse = observed_collapse(args.run_dir)
        if acc_by_did:
            pooled = pooled_all[oa]
            common = [p["data_id"] for p in pooled if p["data_id"] in acc_by_did]
            pe = np.array([pooled[d]["entropy"] for d in common])
            ac = np.array([acc_by_did[d] for d in common])
            # rank correlation (low pooled entropy should co-occur with low accuracy)
            from scipy.stats import spearmanr
            rho, pval = spearmanr(pe, ac)
            print(f"  observed round-1 collapse data_ids (acc~0.25 & mcc~0): {sorted(collapse)}")
            # predicted-biased = lowest-entropy quartile
            thr = np.percentile(pe, 25)
            predicted = set(d for d in common if pooled[d]["entropy"] <= thr)
            hit = len(predicted & collapse)
            print(f"  predicted-biased data_ids (pooled entropy <= p25={thr:.3f}): n={len(predicted)}")
            print(f"  overlap predicted ∩ collapsed = {hit} / {len(collapse)} collapses "
                  f"({100*hit/max(1,len(collapse)):.0f}% of collapses are in the biased quartile)")
            print(f"  Spearman(pooled_entropy, accuracy) over {len(common)} data_ids = {rho:+.3f} (p={pval:.1e})")
            print("  -> positive rho ⇒ higher-entropy (class-mixed) data_ids have higher accuracy = data-bias hypothesis supported")
            summary["_overlay"] = dict(alpha=oa, collapse=sorted(collapse), spearman=rho, p=pval,
                                       predicted_biased=sorted(predicted), overlap=hit)
        else:
            print("  no agg_eval telemetry found in run-dir; overlay skipped")

    with open(os.path.join(out, "diag_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(out,'diag_summary.json')}")

    if args.plots:
        make_plots(summary, pooled_all, args, out)
    if args.dist:
        dist_report(summary, bins_all, alphas, out)


def _canon_matrix(per_client):
    """100×4 counts matrix with class ids canonicalized to 0..3 by rank over the α's label domain."""
    domain = sorted({k for c in per_client for k in c["class_counts"].keys()})
    cmap = {lbl: i for i, lbl in enumerate(domain)}
    K = len(domain)
    M = np.zeros((len(per_client), K), dtype=int)
    for r, c in enumerate(per_client):
        for lbl, n in c["class_counts"].items():
            M[r, cmap[lbl]] = n
    return M, domain


def dist_report(summary, bins_all, alphas, out):
    """How samples distribute across trainers and their data-bins, as heterogeneity varies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n===== DISTRIBUTION STUDY (sample → trainer → bin; N=100) =====")
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 4.6), squeeze=False)
    for j, a in enumerate(alphas):
        pc = summary[a]["per_client"]
        M, domain = _canon_matrix(pc)                     # 100×4 counts
        tot = M.sum()
        per_class_tot = M.sum(0)
        # global structural facts
        sizes = M.sum(1)
        print(f"\n  α={a}:  total samples {tot}  |  samples/trainer min/med/max = "
              f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}  |  per-class totals {per_class_tot.tolist()}")
        # concentration: for each class, share held by its top-10 trainers (of 100)
        top10 = []
        for c in range(M.shape[1]):
            col = np.sort(M[:, c])[::-1]
            top10.append(col[:10].sum() / max(1, col.sum()))
        print(f"        class spread — top-10-trainer share per class: "
              f"{[round(x,2) for x in top10]}  (0.10 = perfectly even; 1.0 = all in 10 trainers)")
        # #classes present per trainer histogram
        ncl = (M > 0).sum(1)
        hist = {k: int((ncl == k).sum()) for k in range(1, M.shape[1] + 1)}
        print(f"        #classes/trainer histogram (1..4 classes): {hist}")
        # per-bin composition across ALL bins of all trainers
        bindom = [dom_frac(b) for bins in bins_all[a] for b in bins]
        print(f"        per-bin dominant-class frac: mean {np.mean(bindom):.3f}  "
              f"(fraction of bins that are ≥50% one class: {np.mean(np.array(bindom)>0.5):.3f})")
        # heatmap: trainers (rows sorted by dominant class then dom frac) × class, row-normalized
        rown = M / M.sum(1, keepdims=True)
        order = sorted(range(len(rown)), key=lambda r: (int(np.argmax(rown[r])), -rown[r].max()))
        ax = axes[0][j]
        im = ax.imshow(rown[order], aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"α={a}\n(mean dom-frac {np.mean([c['dom_frac'] for c in pc]):.2f})")
        ax.set_xlabel("class"); ax.set_xticks(range(len(domain)))
        if j == 0:
            ax.set_ylabel("trainer (sorted by dominant class)")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, label="within-trainer class proportion")
    fig.suptitle("Class distribution across N=100 trainers as heterogeneity decreases (α↑)")
    p = os.path.join(out, "trainer_class_heatmap.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"\n  wrote {p}")


def make_plots(summary, pooled_all, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alphas = list(pooled_all.keys())
    # (1) CDF of per-client dominant-class fraction
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for a in alphas:
        df = np.sort([c["dom_frac"] for c in summary[a]["per_client"]])
        ax.plot(df, np.linspace(0, 1, len(df)), label=f"α={a}")
    ax.set_xlabel("per-client dominant-class fraction"); ax.set_ylabel("CDF")
    ax.set_title("Client class skew by α (100 clients)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "clientskew_cdf.pdf")); plt.close(fig)
    # (2) alpha=1 pooled entropy vs observed accuracy per data_id
    oa = args.overlay_alpha
    ov = summary.get("_overlay")
    if oa in pooled_all and ov:
        acc, collapse = observed_collapse(args.run_dir)
        pooled = pooled_all[oa]
        ks = [p["data_id"] for p in pooled if p["data_id"] in acc]
        fig, ax1 = plt.subplots(figsize=(7, 3.2))
        ax1.plot(ks, [pooled[k]["entropy"] for k in ks], color="tab:blue", label="pooled bin entropy")
        ax1.set_xlabel("data_id"); ax1.set_ylabel("pooled entropy", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ks, [acc[k] for k in ks], color="tab:red", alpha=.6, label="observed accuracy")
        ax2.axhline(0.25, ls="--", color="grey", lw=.8)
        for k in collapse:
            ax2.axvline(k, color="tab:red", alpha=.08)
        ax2.set_ylabel("round-1 accuracy", color="tab:red")
        ax1.set_title(f"α={oa}: pooled data-bin entropy vs observed accuracy (collapses shaded)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "dataid_entropy_vs_acc.pdf")); plt.close(fig)
    print(f"wrote plots to {out}")


if __name__ == "__main__":
    main()
