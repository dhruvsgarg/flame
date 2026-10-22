"""FX-N127 offline rule test: relative paired loss change, trimmed per 10-min bin, EWMA; cut when |rate| < r x |peak rate|."""
import sys, json, glob, os, math, statistics as st
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "scripts")
from accuracy_table import curve
from felix_claims import traffic
BIN = 600
def tmean(xs, p=0.1):
    xs = sorted(xs); k = int(len(xs) * p); xs = xs[k:len(xs) - k] or xs; return sum(xs) / len(xs)
def one(d):
    tasks = []
    for f in glob.glob(d + "/telemetry/trainer_*.jsonl"):
        for line in open(f, "rb"):
            if b'"trainer_round"' in line:
                e = json.loads(line)
                if e.get("task_to_perform") == "train" and e.get("final_loss"):
                    tasks.append((float(e["sim_completion_ts"]), e["end_id"], float(e["final_loss"])))
    tasks.sort(); last = {}; bins = {}
    for t, i, q in tasks:
        if i in last and q > 0 and last[i][1] > 0 and t > last[i][0]:
            bins.setdefault(int(t // BIN), []).append(math.log(q / last[i][1]) / ((t - last[i][0]) / 3600))
        last[i] = (t, q)
    ew, ser = None, []
    for b in sorted(bins):
        if len(bins[b]) < 5: continue
        x = tmean(bins[b]); ew = x if ew is None else 0.5 * x + 0.5 * ew
        ser.append(((b + 1) * BIN, ew, len(bins[b])))
    c, _ = curve(d); tr = traffic(d); total = sum(b for _, b in tr)
    accs = [a for _, _, a in c]; sm = [st.mean(accs[max(0, i - 2):i + 1]) for i in range(len(accs))]
    final = max(sm); plateau = next(t for (t, _, _), a in zip(c, sm) if a >= final - 0.01)
    res = {}
    for r in (0.05, 0.1, 0.2):
        peak, cut, below = 0.0, None, 0
        for t, ew, n in ser:
            peak = min(peak, ew)
            below = below + 1 if peak < 0 and abs(ew) < r * abs(peak) else 0
            if below >= 2: cut = t; break
        if cut is None: res[r] = None; continue
        acc_cut = max([a for (tt, _, _), a in zip(c, sm) if tt <= cut] or [0])
        res[r] = dict(cut_h=cut / 3600, acc_at_cut=100 * acc_cut, after_bytes=sum(b for tt, b in tr if tt > cut) / total)
    name = os.path.basename(d).split("_dbg_")[0].split("_", 3)[-1] + ":" + os.path.basename(d).split("_dbg_")[1].split("_n")[0].replace("google_speech_", "")
    return name, plateau / 3600, 100 * final, res
E = "async_cifar10/experiments/"
pats = ["F42_s0u_ctl_dbg_felix", "F42_s0u_l5_dbg_felix", "F42_s0u_l3b_dbg_felix", "F42_s50u_ctl_dbg_felix", "F42_s50u_l5_dbg_felix", "F42_s50u_ctl_dbg_fedbuff",
        "F42_s0u_ctl_dbg_refl", "F42_s50u_rstar_dbg_refl",
        "gs_F42_s0u_ctl_dbg_google_speech_felix", "gs_F42_s0u_l5_dbg", "gs_F42_s0u_ctl_dbg_google_speech_fedbuff", "gs_F42_s50u_ctl_dbg_google_speech_felix", "gs_F42_s50u_l5_dbg", "gs_F42_s50u_fbstar"]
legs = [sorted(glob.glob(E + f"run_20261022_1*_{p}*_sim"))[0] for p in pats]
with ProcessPoolExecutor(14) as ex:
    for n, pl, fin, res in ex.map(one, legs):
        s = " | ".join(f"r {r}: " + ("never" if v is None else f"cut {v['cut_h']:.1f} h at {v['acc_at_cut']:.1f}% ({v['after_bytes']:.0%} of bytes after)") for r, v in res.items())
        print(f"{n:24} plateau (within 1 pp of {fin:.1f}%) {pl:4.1f} h || {s}")
