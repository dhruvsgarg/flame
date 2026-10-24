"""FX-N127 offline: paired per-client loss change from train-task returns as the pacer's signal (stored F42b telemetry)."""
import sys, json, glob, os, math, statistics as st
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "scripts")
from accuracy_table import curve
BIN = 600
def one(d):
    tasks = []
    for f in glob.glob(d + "/telemetry/trainer_*.jsonl"):
        for line in open(f, "rb"):
            if b'"trainer_round"' in line:
                e = json.loads(line)
                if e.get("task_to_perform") == "train" and e.get("final_loss") is not None:
                    tasks.append((float(e["sim_completion_ts"]), e["end_id"], float(e["final_loss"]), int(e.get("dataset_size") or 0)))
    tasks.sort(); last = {}; ds = []
    for t, i, q, n in tasks:
        if i in last:
            t0, q0 = last[i]
            if t - t0 > 0: ds.append((t, (q - q0) / ((t - t0) / 3600)))  # loss change per hour, same client
        last[i] = (t, q)
    c, _ = curve(d)
    bins = {}
    for t, x in ds: bins.setdefault(int(t // BIN), []).append(x)
    rows = []
    for b in sorted(bins):
        xs = bins[b]
        if len(xs) < 3: continue
        m = st.mean(xs); se = st.stdev(xs) / math.sqrt(len(xs))
        acc = max([a for tt, _, a in c if tt <= (b + 1) * BIN] or [0])
        rows.append(((b + 1) * BIN / 3600, len(xs), m, se, acc))
    return os.path.basename(d).split("_dbg_")[0].split("_", 3)[-1] + ":" + os.path.basename(d).split("_dbg_")[1].split("_n")[0].replace("google_speech_", ""), rows
E = "async_cifar10/experiments/"
pats = ["F42_s0u_ctl_dbg_felix", "F42_s0u_l5_dbg_felix", "F42_s50u_ctl_dbg_felix", "F42_s50u_ctl_dbg_fedbuff", "gs_F42_s0u_ctl_dbg_google_speech_felix",
        "gs_F42_s0u_ctl_dbg_google_speech_fedbuff", "gs_F42_s50u_ctl_dbg_google_speech_felix"]
legs = [sorted(glob.glob(E + f"run_20261022_1*_{p}*_sim"))[0] for p in pats]
out = {}
with ProcessPoolExecutor(8) as ex:
    for n, rows in ex.map(one, legs):
        out[n] = rows
        print(f"== {n}: hour | tasks/10min | paired loss change per hour (mean ± se) | z | best acc so far")
        for h, k, m, se, acc in rows:
            if abs(h * 6 - round(h * 6)) < 1e-6 and (h <= 3 or round(h * 6) % 3 == 0):
                print(f"   {h:5.2f} h  {k:4d}  {m:+7.3f} ± {se:.3f}  z {m/se if se else 0:+6.1f}  acc {100*acc:5.1f}")
json.dump(out, open(sys.argv[1], "w"))
