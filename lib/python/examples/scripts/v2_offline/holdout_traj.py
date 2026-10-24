"""Population held-out loss (clients' own 10% holdout, visible at t) at every checkpoint vs test loss/acc; per-sample sd for detection power."""
import sys, json, torch, torch.nn.functional as F, statistics as st
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
pr = op.Probe(sys.argv[1], sys.argv[2]); m = pr.b.model; rows = []
def hold_idx(c, t):
    n_tr = len(pr.train_idx(c, t)); n_vis = int(round(n_tr / 0.9))
    return [p for j, p in enumerate(c.order[:n_vis]) if j % 10 == 9]
for ts, path in pr.ckpts[::max(1, len(pr.ckpts) // 24)]:
    m.load_state_dict(torch.load(path, map_location=pr.dev, weights_only=False)["state_dict"]); m.eval()
    per, nh = [], 0
    with torch.no_grad():
        for c in pr.b.cl:
            h = hold_idx(c, ts)
            if h:
                l = F.nll_loss(pr.b.fwd(m, pr.b.X[h]), pr.b.Y[h], reduction="none").float().tolist(); per += l; nh += 1
    tl, ta = pr.loss_acc(m, pr.B)
    rows.append((ts / 3600, st.mean(per), st.pstdev(per), len(per) / max(nh, 1), tl, ta))
    print(f"{ts/3600:5.1f} h  holdout loss {st.mean(per):.3f} (sd/sample {st.pstdev(per):.2f}, {len(per)/max(nh,1):.1f} samples/client)  test loss {tl:.3f} acc {100*ta:.1f}", flush=True)
json.dump(rows, open(sys.argv[3], "w"))
