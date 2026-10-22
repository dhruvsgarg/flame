"""A: does the current global model's loss on a client's NEW samples (arrived in (t, t+dt]) exceed its loss on samples it already trained on?"""
import sys, torch, torch.nn.functional as F, statistics as st
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
pr = op.Probe(sys.argv[1], sys.argv[2])
m = pr.b.model
@torch.no_grad()
def nll(idx):
    m.eval(); return float(F.nll_loss(pr.b.fwd(m, pr.b.X[idx]), pr.b.Y[idx], reduction="mean"))
for h in (2, 4, 8):
    ts, p = min(pr.ckpts, key=lambda x: abs(x[0] - h * 3600))
    m.load_state_dict(torch.load(p, map_location=pr.dev, weights_only=False)["state_dict"])
    LB, AB = pr.loss_acc(m, pr.B)
    for dt in (600, 3600):
        old_l, new_l, ratios, nnew = [], [], [], 0
        for c in pr.b.cl:
            a, b = pr.train_idx(c, ts), pr.train_idx(c, ts + dt)
            new = b[len(a):]
            if not a or not new: continue
            lo, ln = nll(a), nll(new); old_l.append(lo); new_l.append(ln); ratios.append(ln / max(lo, 1e-6)); nnew += len(new)
        if ratios:
            print(f"t {ts/3600:.1f} h dt {dt/60:.0f} min: clients with new data {len(ratios)}/{len(pr.b.cl)} ({nnew} samples) | loss trained {st.median(old_l):.3f} new {st.median(new_l):.3f} test {LB:.3f} | median new/trained {st.median(ratios):.1f}x, share > 2x {sum(r > 2 for r in ratios)/len(ratios):.2f}", flush=True)
        else:
            print(f"t {ts/3600:.1f} h dt {dt/60:.0f} min: no client got new data", flush=True)
