import sys, json, glob, os, statistics as st
from concurrent.futures import ProcessPoolExecutor
def one(d):
    per = {}
    for f in glob.glob(d + "/telemetry/trainer_*.jsonl"):
        for line in open(f, "rb"):
            if b'"trainer_round"' in line:
                e = json.loads(line)
                if e.get("task_to_perform") == "train" and e.get("stream_clock_s") is not None:
                    per.setdefault(e["end_id"], []).append((float(e["stream_clock_s"]), int(e["dataset_size"])))
    fresh, tot, gaps, repeat = 0, 0, [], 0
    for tid, xs in per.items():
        xs.sort(); last = 0; lt = None
        for t, n in xs:
            fresh += max(0, n - last); tot += n
            if lt is not None: gaps.append(t - lt)
            if n <= last: repeat += 1
            last, lt = max(last, n), t
    h = max(t for xs in per.values() for t, _ in xs) / 3600
    # backlog: sample at hourly marks the share of visible-but-never-trained data (vis from each client's next task)
    ntasks = sum(len(x) for x in per.values())
    gaps.sort()
    return os.path.basename(d).split("_dbg_")[0].split("_", 3)[-1] + ":" + os.path.basename(d).split("_dbg_")[1].split("_n")[0].replace("google_speech_", ""), \
        dict(fresh_per_h=fresh / h, fresh_share=fresh / tot, retrain_gap_med_min=gaps[len(gaps)//2] / 60, gap_p90=gaps[int(.9*len(gaps))]/60, no_new_data_tasks=repeat / ntasks)
E = "async_cifar10/experiments/"
legs = [g for p in ("gs_F42_s0u_ctl_dbg_google_speech_felix", "gs_F42_s0u_ctl_dbg_google_speech_fedbuff", "gs_F42_s0u_l5p_dbg", "gs_F42_s0u_l3b_dbg",
                    "gs_F42_s50u_ctl_dbg_google_speech_felix", "gs_F42_s50u_ctl_dbg_google_speech_fedbuff", "gs_F42_s50u_l5p_dbg",
                    "F42_s0u_ctl_dbg_felix", "F42_s0u_l5p_dbg", "F42_s50u_ctl_dbg_felix", "F42_s50u_ctl_dbg_fedbuff", "F42_s50u_l5p_dbg", "F42_s50u_fbstar")
        for g in sorted(glob.glob(E + f"run_20261022_1*_{p}*_sim"))[:1]]
with ProcessPoolExecutor(16) as ex:
    for n, r in ex.map(one, legs):
        print(f"{n:28} fresh samples/h {r['fresh_per_h']:7.0f}  fresh share of trained {r['fresh_share']:.2f}  re-train gap med {r['retrain_gap_med_min']:5.1f} min (p90 {r['gap_p90']:5.1f})  tasks with no new data {r['no_new_data_tasks']:.2f}")
