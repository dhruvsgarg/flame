import sys, os, json, glob, csv, collections, statistics as st
from concurrent.futures import ProcessPoolExecutor
def one(d):
    tr = collections.Counter(); ev = 0; samp = []; dur = []; evdur = 0.0; vis = []
    for f in glob.glob(d + "/telemetry/trainer_*.jsonl"):
        for line in open(f, "rb"):
            if b'"trainer_round"' not in line: continue
            e = json.loads(line)
            if e.get("task_to_perform") == "eval":
                ev += 1; evdur += e.get("sim_round_duration_s") or 0; continue
            tr[e["end_id"]] += 1; samp.append(e.get("model_samples") or e.get("dataset_size") or 0)
            dur.append(e.get("sim_round_duration_s") or 0); vis.append(e.get("visible_samples") or 0)
    stl = []; T = 0; v = 0
    f = glob.glob(d + "/telemetry/aggregator_*.jsonl")[0]
    for line in open(f, "rb"):
        if b'"agg_round"' in line:
            e = json.loads(line); stl += e.get("staleness") or []; T = max(T, float(e.get("vclock_now") or 0)); v = max(v, e["round"])
    n = sum(tr.values()); h = T / 3600; top = sorted(tr.values(), reverse=True)
    return dict(leg=os.path.basename(d).split("_dbg_")[0].split("_", 3)[-1] + ":" + os.path.basename(d).split("_dbg_")[1].split("_n")[0].replace("google_speech_", ""),
                trace="s50" if "syn_50" in d else "s0", ds="speech" if "google_speech" in d else "cifar",
                h=h, train_per_h=n / h, eval_per_h=ev / h, eval_slot_share=evdur / (evdur + sum(dur)) if dur else 0,
                samp=st.mean(samp), samp_per_h=sum(samp) / h, task_s=st.mean(dur), stale=st.mean(stl) if stl else 0,
                versions=v, uniq=len(tr), top20=sum(top[:max(1, len(top) // 5)]) / n, vis_pick=st.mean(vis))
legs = []
for p in glob.glob("experiments/pool_20261022_1123_F42B_*"):
    for f in glob.glob(p + "/*/summary.tsv"):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["sim_dir"] and r["baseline"] in ("felix", "fedbuff") and "_r2" not in r["sim_dir"] and "fck" not in r["sim_dir"]: legs.append(r["sim_dir"])
legs += glob.glob("async_cifar10/experiments/run_20261022_1*F42_s0u_ctl_dbg_fedbuff*_sim")
with ProcessPoolExecutor(24) as ex:
    rs = list(ex.map(one, sorted(set(legs))))
print("leg trace h train/h eval/h evalslot samp/upd samp/h task_s stale versions uniq top20 vis")
for r in sorted(rs, key=lambda r: (r["ds"], r["trace"], r["leg"])):
    print(f"{r['ds']:6} {r['trace']:3} {r['leg']:22} {r['h']:4.1f} {r['train_per_h']:6.0f} {r['eval_per_h']:5.0f} {r['eval_slot_share']:5.2f} {r['samp']:6.0f} {r['samp_per_h']:8.0f} {r['task_s']:6.0f} {r['stale']:5.2f} {r['versions']:5d} {r['uniq']:3d} {r['top20']:5.2f} {r['vis_pick']:5.0f}")
