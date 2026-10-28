#!/usr/bin/env python3
"""FX-N153: one leg's digest = every series a paper plot reads (PAPER_TEMPLATE Part E), then its raw telemetry gzipped.

  leg_digest.py RUN_DIR... [--no-gzip] [--check]

Writes RUN_DIR/digest.json.gz (schema SCHEMA, source event of each series in SOURCES) and gzips telemetry/*.jsonl; a missing or
empty required series exits 1 and leaves the raw logs as they are. `--check` validates existing digests only.
Tasks come from trainer `trainer_round` (one per dispatch: t0 = sim_send_ts, done = sim_completion_ts); a task's slot frees at
its commit (agg_round contributing_trainers), eviction / timeout (abandon_timeout) or the next dispatch to that client.
Departures are read from the leg's own trace at the run's offset (L29), not from trainer events.
"""
import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..")]
SCHEMA = 3
SOURCES = {  # series -> telemetry it is built from
    "acc": "agg_eval / agg_eval_raw (test accuracy), t = last agg_round vclock_now, gb = cumulative wire bytes, updates = cumulative",
    "hit": "acc 5-eval mean (targetWindow) >= report_target (config target, lowered by trace_targets.json)", "end": "run_end / last agg_round, acc",
    "updates": "agg_round (len staleness = updates), agg_timing (versions)",
    "staleness": "agg_round.staleness (all), withheld_delivery.staleness (held across a departure)",
    "slots": "trainer_round tasks, release by agg_round / abandon_timeout, departures from the trace, capacity selection.effective_c",
    "rounds": "sync only: trainer_round per model version (model_compute_s, model_comm_s), agg_round vclock_now",
    "picks": "trainer_round per client (total_samples, model_samples)", "c": "dynamic_kc (Felix) else selection",
    "fates": "task release kind x median wire send bytes", "departures": "trace flips inside a task, release time",
    "overhead": "checkin, wire, trace flips, agg_timing",
}
REQUIRED = ("acc", "end", "updates", "staleness", "slots", "picks", "c", "fates", "departures", "overhead")
EV = re.compile(rb'"event": "([a-z_]+)"')


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _files(run_dir, pat):
    return sorted(glob.glob(os.path.join(run_dir, "telemetry", pat + ".jsonl")) + glob.glob(os.path.join(run_dir, "telemetry", pat + ".jsonl.gz")))


def _hist(xs):
    return {str(k): v for k, v in sorted(collections.Counter(xs).items())}


def trace_events(run_dir, cfg):
    """{end: [(t, state)]} on the run's clock (offset and scale from the run's config, L29); {} on syn_0 / no trace."""
    from flame.availability.trace import offset_events, raw_events
    import yaml
    hp = cfg.get("hyperparameters", {})
    trace = hp.get("availability_trace") or ""
    if not trace or trace.startswith("syn_0"):
        return {}
    off, scale = float(hp.get("trace_offset_s") or 0.0), float(hp.get("trace_time_scale") or 1.0)
    reg = yaml.safe_load(open(os.path.join(HERE, "..", "_metadata", "trainer_registry.yaml")))["trainers"]
    out = {}
    for key, meta in reg.items():
        try:
            out[meta["task_id"]] = [(float(t) / scale, s) for t, s in offset_events(raw_events(trace, key), off)]
        except KeyError:
            continue
    return out


def train_after(ev, t):
    """First time >= t the client is AVL_TRAIN (t itself when it is; inf = never)."""
    if not ev:
        return t
    ts = [x for x, _ in ev]
    i = bisect.bisect_right(ts, t) - 1
    if i < 0 or ev[i][1] == "AVL_TRAIN":
        return t
    return next((x for x, s in ev[i + 1:] if s == "AVL_TRAIN"), float("inf"))


def first_departure(ev, t0, t1):
    """First time in (t0, t1] the client is not AVL_TRAIN, or None."""
    if not ev:
        return None
    ts = [t for t, _ in ev]
    i = bisect.bisect_right(ts, t0) - 1
    if i >= 0 and ev[i][1] != "AVL_TRAIN":
        return t0
    for t, s in ev[i + 1:]:
        if t > t1:
            return None
        if s != "AVL_TRAIN":
            return t
    return None


def aggregator_pass(run_dir):
    """One streamed pass over the aggregator log: every aggregator-side series."""
    acc, raw, rounds, evals = [], [], {}, []
    t, nbytes, send_sizes, msgs, n_upd = 0.0, 0, [], 0, 0
    upd_t, ver_t, stale, wstale, commits, aband = [], [], [], [], collections.defaultdict(list), collections.defaultdict(list)
    csel, cdkc, checkins, timing, run_end = [], [], 0, collections.defaultdict(list), None
    nonfinite = 0
    wire_b, vclk, accre = re.compile(rb'"bytes": (\d+)'), re.compile(rb'"vclock_now": ([-0-9.e]+)'), re.compile(rb'"test-accuracy": ([-0-9.e]+)')
    for path in _files(run_dir, "aggregator_*"):
        with _open(path) as fh:
            for line in fh:
                m = EV.search(line, 0, 160)
                k = m.group(1) if m else b""
                if k == b"wire":
                    b = int(wire_b.search(line).group(1))
                    nbytes, msgs = nbytes + b, msgs + 1
                    if b'"dir": "send"' in line and len(send_sizes) < 20000:
                        send_sizes.append(b)
                elif k == b"checkin":
                    checkins += 1
                elif k == b"update_rejected":
                    nonfinite += b"nonfinite" in line
                elif k in (b"agg_eval", b"agg_eval_raw"):
                    a = float(accre.search(line).group(1))
                    (acc if k == b"agg_eval" else raw).append((t, a, nbytes, n_upd))
                elif k == b"agg_round":
                    e = json.loads(line)
                    t = float(e.get("vclock_now") or t)
                    n = len(e.get("staleness") or [])
                    n_upd += n
                    upd_t.append((t, n))
                    stale += [int(s) for s in e.get("staleness") or []]
                    for end in e.get("contributing_trainers") or []:
                        commits[end].append(t)
                    rounds[int(e["round"])] = (t, e.get("trainer_speed_s") or [], e.get("contributing_trainers") or [])
                elif k == b"agg_timing":
                    e = json.loads(line)
                    ver_t.append(float(e.get("vclock_now") or t))
                    for f in ("cycle_s", "ingest_s", "commit_s"):
                        timing[f].append(float(e.get(f) or 0.0))
                elif k == b"withheld_delivery":
                    e = json.loads(line)
                    if e.get("accepted", True):
                        wstale.append(int(e.get("staleness") or 0))
                elif k == b"abandon_timeout":
                    e = json.loads(line)
                    aband[e["end_id"]].append((float(e["sim_send_ts"]), float(e["vclock_now"]), e.get("reason") or "?"))
                elif k == b"dynamic_kc":
                    e = json.loads(line)
                    cdkc.append((float(e.get("now") or t), e.get("c"), e.get("n_eligible_train"), e.get("loss_phase")))
                elif k == b"selection":
                    if b'"task": "train"' not in line[:400]:
                        continue
                    e = json.loads(line)
                    vt = e.get("vclock_now")
                    if vt is not None:
                        csel.append((float(vt), e.get("effective_c") or e.get("num_chosen"), e.get("num_eligible")))
                elif k == b"run_end":
                    e = json.loads(line)
                    run_end = float(e.get("vclock_now") or t)
    return dict(acc=acc, raw=raw, end=run_end if run_end is not None else t, gb=nbytes / 1e9, msgs=msgs, model_b=st.median(send_sizes) if send_sizes else 0,
                upd_t=upd_t, ver_t=ver_t, stale=stale, wstale=wstale, commits=commits, aband=aband, csel=csel, cdkc=cdkc,
                checkins=checkins, timing=timing, rounds=rounds, nonfinite=nonfinite)


def trainer_tasks(run_dir):
    """[(end, t0, done, dur, compute, comm, round, total, samples)] from every trainer's trainer_round."""
    out = []
    for path in _files(run_dir, "trainer_*"):
        end = re.sub(r"\.jsonl(\.gz)?$", "", os.path.basename(path))[len("trainer_"):]
        with _open(path) as fh:
            for line in fh:
                if b'"trainer_round"' not in line[:160]:
                    continue
                e = json.loads(line)
                if e.get("task_to_perform", "train") != "train":
                    continue
                t0 = float(e.get("sim_send_ts") or 0.0)
                out.append((end, t0, float(e.get("sim_completion_ts") or t0), float(e.get("sim_round_duration_s") or 0.0),
                            float(e.get("model_compute_s") or 0.0), float(e.get("model_comm_s") or 0.0), int(e.get("round") or 0),
                            int(e.get("total_samples") or 0), float(e.get("model_samples") or 0.0)))
    return out


def binned(pts, bin_s=600.0):
    """Last value per bin_s bin: [(t, ...)] -> same shape, ≤ 1 point per bin."""
    out = {}
    for p in pts:
        out[int(p[0] // bin_s)] = p
    return [out[k] for k in sorted(out)]


def report_target(cfg):
    """The leg's reporting target: its config target, lowered to the trace's `_metadata/trace_targets.json` override."""
    hp = cfg.get("hyperparameters", {})
    tgt = hp.get("targetAccuracy")
    ds = hp.get("dataset_name") or {"cifar-10": "cifar10"}.get(str(cfg.get("job", {}).get("name")))  # speech's job name is "cifar-10"
    ov = json.load(open(os.path.join(HERE, "..", "_metadata", "trace_targets.json"))).get(ds, {}).get(hp.get("availability_trace"))
    return tgt if tgt is None or ov is None else min(float(tgt), ov)


def digest(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "aggregator_config.json")))
    hp = cfg.get("hyperparameters", {})
    A = aggregator_pass(run_dir)
    tasks = trainer_tasks(run_dir)
    end_t = A["end"]
    sync = str(cfg.get("selector", {}).get("sort", "")) not in ("fedbuff", "async_oort") and not A["cdkc"] and \
        any(len(r[2]) > 1 for r in A["rounds"].values())
    d = dict(schema=SCHEMA, run_dir=os.path.abspath(run_dir), sources=SOURCES, sync=sync,
             config=dict(job=cfg.get("job", {}).get("name"), selector=cfg.get("selector", {}).get("sort"),
                         target=hp.get("targetAccuracy"), report_target=report_target(cfg), window=hp.get("targetWindow"), K=hp.get("aggGoal"),
                         c=(cfg.get("selector", {}).get("kwargs") or {}).get("c"), runtime_s=hp.get("max_experiment_runtime_s"),
                         trace_offset_s=hp.get("trace_offset_s"), knowledge=hp.get("avail_knowledge")))
    # accuracy, hit, end
    acc = A["acc"]
    d["acc"] = dict(t=[p[0] for p in acc], acc=[p[1] for p in acc], gb=[p[2] / 1e9 for p in acc], updates=[p[3] for p in acc],
                    raw=[p[1] for p in A["raw"]] if len(A["raw"]) == len(acc) else [])
    w, tgt = int(hp.get("targetWindow") or 5), report_target(cfg)
    sm = [st.mean(d["acc"]["acc"][max(0, i - w + 1):i + 1]) for i in range(len(acc))]
    i = next((i for i, a in enumerate(sm) if tgt is not None and i + 1 >= w and a >= float(tgt)), None)
    d["hit"] = None if i is None else dict(target=float(tgt), h=acc[i][0] / 3600, gb=acc[i][2] / 1e9, acc=sm[i], updates=acc[i][3])
    d["end"] = dict(h=end_t / 3600, gb=A["gb"], acc_last=sm[-1] if sm else None, hit=i is not None,
                    diverged=A["nonfinite"] >= 100)  # a diverged global model rejects every update as non-finite (FX-N166b)
    # updates / versions
    hours = max(1, int(end_t // 3600) + 1)
    per_h, vper_h = [0] * hours, [0] * hours
    for t, n in A["upd_t"]:
        per_h[min(hours - 1, int(t // 3600))] += n
    for t in A["ver_t"]:
        vper_h[min(hours - 1, int(t // 3600))] += 1
    d["updates"] = dict(total=sum(per_h), per_h=per_h, versions=len(A["ver_t"]), versions_per_h=vper_h,
                        rate_h=sum(per_h) / max(end_t / 3600, 1e-9))
    d["staleness"] = dict(all=_hist(A["stale"]), withheld=_hist(A["wstale"]), mean_all=st.mean(A["stale"]) if A["stale"] else None,
                          mean_withheld=st.mean(A["wstale"]) if A["wstale"] else None, sum_all=sum(A["stale"]), sum_withheld=sum(A["wstale"]))
    # tasks: release, fate, departure
    trace = trace_events(run_dir, cfg)
    by_end = collections.defaultdict(list)
    for tk in tasks:
        by_end[tk[0]].append(tk)
    fates, lags, left, held_s, busy_s, wait_s, comp_s = collections.Counter(), [], 0, 0.0, 0.0, 0.0, 0.0
    notify = hp.get("avail_knowledge") == "notify"
    by_fate = collections.defaultdict(lambda: [0.0, 0.0, 0.0])  # fate -> [compute, wait, held] slot-seconds
    detected = collections.Counter()
    hb, hh = [0.0] * hours, [0.0] * hours

    def add(arr, a, b):
        while a < b:
            h = min(hours - 1, int(a // 3600))
            nxt = min(b, (h + 1) * 3600.0) if h < hours - 1 else b
            arr[h] += nxt - a
            a = nxt

    for end, ts in by_end.items():
        ts.sort(key=lambda x: x[1])
        cm = sorted(A["commits"].get(end, []))
        abl = sorted(A["aband"].get(end, []))
        abs0 = [x[0] for x in abl]
        for j, (_, t0, done, dur, comp, comm, rnd, total, samples) in enumerate(ts):
            nxt = ts[j + 1][1] if j + 1 < len(ts) else end_t
            i_ab = bisect.bisect_left(abs0, t0 - 0.5)  # the abandon stamped with this task's send time
            ab = abl[i_ab][1:] if i_ab < len(abl) and abs(abs0[i_ab] - t0) <= 0.5 else None
            k = bisect.bisect_left(cm, done - 1e-6)
            c_t = cm[k] if k < len(cm) and cm[k] <= nxt + 1e-6 else None
            arrive = train_after(trace.get(end), done)  # an offline client's upload waits for its return (withheld)
            if ab is not None:
                rel, fate = ab[0], "evicted" if "evict" in ab[1] else "timeout"
                if c_t is not None and c_t > ab[0]:
                    fate = "late"
            elif c_t is not None:
                rel, fate = c_t, "aggregated"
            else:  # arrived and not aggregated (sync straggler, stale-rejected), or still out at the end
                rel, fate = min(arrive, nxt, end_t), "open" if arrive > end_t else "discarded"
            if ab is None and notify and arrive > done:  # FX-N167: notify's send gate frees the slot at completion (no event)
                rel = min(rel, done)
            fates[fate] += 1
            rel = max(rel, t0)
            dep = first_departure(trace.get(end), t0, rel)
            if dep is not None and dep < done:  # left mid-task
                left += 1
                if ab is not None:
                    lags.append(max(0.0, ab[0] - dep))
                    detected[fate] += 1
                else:
                    detected["returned"] += 1
            on = rel if dep is None else max(t0, dep)  # [t0, on] online, [on, rel] held by a departed client
            comp_s += min(done, on) - t0
            wait_s += max(0.0, on - done)  # finished, online, waiting for the commit (sync barrier)
            bf = by_fate[fate]
            bf[0], bf[1], bf[2] = bf[0] + min(done, on) - t0, bf[1] + max(0.0, on - done), bf[2] + rel - on
            busy_s += on - t0
            held_s += rel - on
            add(hb, t0, on)
            add(hh, on, rel)
    cap = None  # async slot capacity: c(t) from dynamic_kc (Felix's c), else the configured c
    if not sync:
        if A["cdkc"]:
            cs = sorted((t, c) for t, c, _, _ in A["cdkc"])
            cs = [(0.0, cs[0][1])] + cs + [(end_t, None)]
            cap = sum(float(a[1] or 0) * (b[0] - a[0]) for a, b in zip(cs, cs[1:]) if b[0] > a[0])
        elif d["config"]["c"]:
            cap = float(d["config"]["c"]) * end_t
    occ = busy_s + held_s
    d["slots"] = dict(busy_s=busy_s, held_departed_s=held_s, compute_s=comp_s, wait_s=wait_s, capacity_s=cap, by_fate=dict(by_fate),
                      idle_s=None if cap is None else max(0.0, cap - occ), busy_per_h=hb, held_per_h=hh)
    m = A["model_b"]
    deliv = ("aggregated", "late", "discarded")
    d["fates"] = dict(counts=dict(fates), model_mb=m / 1e6, gb={f: n * m * (2 if f in deliv else 1) / 1e9 for f, n in fates.items()})
    q = sorted(lags)
    d["departures"] = dict(tasks=len(tasks), left_mid_task=left, share=left / len(tasks) if tasks else None, detected=dict(detected),
                           lag_s=dict(n=len(q), p50=q[len(q) // 2] if q else None, p90=q[int(0.9 * (len(q) - 1))] if q else None,
                                      mean=st.mean(q) if q else None))
    # picks
    picks, size = collections.Counter(), {}
    for tk in tasks:
        picks[tk[0]] += 1
        size[tk[0]] = max(size.get(tk[0], 0), tk[7])
    n_clients = len(_files(run_dir, "trainer_*")) or len(picks)
    top = sorted(picks.values(), reverse=True)
    k20 = max(1, round(0.2 * n_clients))
    xs = [(size[e], picks[e]) for e in picks]
    corr = None
    if len(xs) > 2 and len({x for x, _ in xs}) > 1 and len({y for _, y in xs}) > 1:
        corr = st.correlation([x for x, _ in xs], [y for _, y in xs])
    d["picks"] = dict(n=sum(top), coverage=len(picks) / n_clients if n_clients else None, top20_share=sum(top[:k20]) / sum(top) if top else None,
                      corr_size=corr, samples_per_update=st.mean(tk[8] for tk in tasks) if tasks else None,
                      per_client={e: [picks[e], size[e]] for e in picks})
    # concurrency / eligible / loss phase
    src = A["cdkc"] or [(t, c, e, None) for t, c, e in A["csel"]]
    b = binned(src)
    d["c"] = dict(source="dynamic_kc" if A["cdkc"] else "selection", t=[p[0] for p in b], c=[p[1] for p in b],
                  eligible=[p[2] for p in b], phi=[p[3] for p in b])
    # sync rounds: open (first dispatch of the version), commit, slowest committed pick's compute / transfer, median pick
    if sync:
        per_v = collections.defaultdict(list)
        for tk in tasks:
            per_v[tk[6]].append(tk)
        rows = []
        for r, (ct, speeds, contrib) in sorted(A["rounds"].items()):
            ts = per_v.get(r, [])
            if not ts:
                continue
            slow = max(ts, key=lambda x: x[3])
            rows.append([min(x[1] for x in ts), ct, slow[4], slow[5], st.median(x[3] for x in ts), len(ts)])
        d["rounds"] = dict(cols=["open", "commit", "slow_compute_s", "slow_comm_s", "median_pick_s", "picks"], rows=rows)
    else:
        d["rounds"] = None
    flips = sum(sum(1 for t, _ in ev if 0 < t <= end_t) for ev in trace.values())
    d["overhead"] = dict(checkins=A["checkins"], wire_msgs=A["msgs"], trace_flips=flips, client_h=n_clients * end_t / 3600,
                         flips_per_client_h=flips / max(n_clients * end_t / 3600, 1e-9),
                         **{f"{f}_mean": st.mean(v) for f, v in A["timing"].items() if v})
    try:
        import config_audit as ca
        d["audit"] = ca.audit_run(run_dir)
    except Exception as e:  # recorded, never fatal: status treats a missing audit as stale
        d["audit"] = dict(ok=False, miss=[f"audit error: {e}"])
    return d


def check(d):
    """Missing / empty required series, [] when valid."""
    bad = [k for k in REQUIRED if d.get(k) is None]
    if d.get("schema") != SCHEMA:
        bad.append(f"schema {d.get('schema')} != {SCHEMA}")
    if not (d.get("acc") or {}).get("t"):
        bad.append("acc empty")
    if not (d.get("updates") or {}).get("total"):
        bad.append("updates 0")
    if not (d.get("departures") or {}).get("tasks"):
        bad.append("no tasks")
    if d.get("sync") and not (d.get("rounds") or {}).get("rows"):
        bad.append("sync rounds empty")
    return bad


def load(run_dir):
    p = os.path.join(run_dir, "digest.json.gz")
    return json.load(gzip.open(p, "rt")) if os.path.exists(p) else None


def gzip_raw(run_dir):
    files = glob.glob(os.path.join(run_dir, "telemetry", "*.jsonl"))
    if not files:
        return
    if shutil.which("pigz"):
        subprocess.run(["pigz", "-p", "4", *files], check=True)
    else:
        subprocess.run(["gzip", *files], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--no-gzip", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    for r in a.runs:
        d = load(r) if a.check else digest(r)
        bad = ["no digest"] if d is None else check(d)
        if bad:
            print(f"[leg_digest] FAIL {r}: {', '.join(bad)}", flush=True)
            rc = 1
            continue
        if not a.check:
            tmp = os.path.join(r, "digest.json.gz.tmp")
            with gzip.open(tmp, "wt") as f:
                json.dump(d, f)
            os.replace(tmp, os.path.join(r, "digest.json.gz"))
            if not a.no_gzip:
                gzip_raw(r)
        print(f"[leg_digest] ok {r}: {d['end']['h']:.1f} h, {d['updates']['total']} updates, hit={d['hit'] is not None}, "
              f"{os.path.getsize(os.path.join(r, 'digest.json.gz')) / 1e6:.2f} MB", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
