"""Paired per-client held-out loss/acc change between consecutive checkpoints (~20-70 min): mean, sd, z with every client reporting once."""
import sys, math, torch, torch.nn.functional as F, statistics as st
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
pr = op.Probe(sys.argv[1], sys.argv[2]); m = pr.b.model
def hold_idx(c, t):
    n_vis = int(round(len(pr.train_idx(c, t)) / 0.9)); return [p for j, p in enumerate(c.order[:n_vis]) if j % 10 == 9]
prev = None
for ts, path in pr.ckpts[::max(1, len(pr.ckpts) // 30)]:
    m.load_state_dict(torch.load(path, map_location=pr.dev, weights_only=False)["state_dict"]); m.eval(); cur = {}
    with torch.no_grad():
        for c in pr.b.cl:
            h = hold_idx(c, ts)
            if h:
                out = pr.b.fwd(m, pr.b.X[h]); cur[c.tid] = (h, float(F.nll_loss(out, pr.b.Y[h])), float((out.argmax(1) == pr.b.Y[h]).float().mean()))
    if prev:
        pts, pcur = prev
        # paired on the same held-out samples (the earlier set) so new arrivals don't add noise
        dl, da = [], []
        with torch.no_grad():
            for tid, (h, l0, a0) in pcur.items():
                out = pr.b.fwd(m, pr.b.X[h]); dl.append(float(F.nll_loss(out, pr.b.Y[h])) - l0); da.append(float((out.argmax(1) == pr.b.Y[h]).float().mean()) - a0)
        n = len(dl); zl = st.mean(dl) / (st.stdev(dl) / math.sqrt(n)); za = st.mean(da) / (st.stdev(da) / math.sqrt(n))
        print(f"{pts/3600:5.1f} -> {ts/3600:5.1f} h ({(ts-pts)/60:3.0f} min): d holdout loss {st.mean(dl):+.3f} (sd {st.stdev(dl):.2f}, z {zl:+5.1f}) | d holdout acc {100*st.mean(da):+5.1f} pp (z {za:+5.1f}) | n {n}", flush=True)
    prev = (ts, cur)
