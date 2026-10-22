#!/usr/bin/env python3
"""F42 (R27): how availability knowledge turns into stalls, per sim leg (telemetry + the leg's own trace; no runs).

  predictor_audit.py POOL... [--runs RUN ...]

Per leg: train picks; survival = share of picks that stay AVL_TRAIN for the leg's median task time after dispatch (ground truth);
REFL: survival of the picks its predictor flagged as leaving soon (`priority_ids`) vs the rest; FedDance: Spearman of its believed
availability V_m with survival; abandons / h (slots held to the timeout) and wasted slot-hours (dead picks x timeout); knowledge
mode (R27). Felix (notify) is the reference: it evicts at the flip, so a dead pick costs it one flip, not a timeout.
REFL: precision / recall of "leaves soon" (flagged picks mostly off AVL_TRAIN in [t + μ, t + 2μ], its forecast horizon, μ = median
round / gone picks it flagged). Check-in (F42-T1): check-ins, rejects
and missed opportunities per client-hour. Notify (F42-T6, A7): believed ≠ trace seconds per client-hour (missed = AVL_TRAIN believed
UN_AVL, false = the reverse); sim notify is the trace delayed by `notify_latency_s`, real reads the aggregator's notify receipts.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "async_cifar10", "scripts"))
from accuracy_table import rows_from_pools  # noqa: E402
from parity import event_invariants as ev  # noqa: E402
from selection_audit import spearman  # noqa: E402


def _share_train(trace, a, b):
    from flame.availability.trace import state_at
    if not trace:
        return 1.0
    ts = [a] + [t for t in trace.irange(a, b, inclusive=(False, False))] + [b]
    on = sum((t1 - t0) for t0, t1 in zip(ts, ts[1:]) if str(state_at(trace, t0)).endswith("AVL_TRAIN"))
    return on / (b - a) if b > a else 1.0


def audit(run_dir):
    run = ev.load_run(run_dir)
    gt = ev._ground_truth(run)
    durs = [e.get("sim_round_duration_s") for evs in run["trainers"].values() for e in evs
            if e.get("event") == "trainer_round" and e.get("task_to_perform", "train") == "train" and e.get("sim_round_duration_s")]
    d = st.median(durs) if durs else 60.0
    hp = run["cfg"].get("hyperparameters", {}) or {}
    know = hp.get("avail_knowledge") or ("notify" if str(hp.get("proactive_inflight_evict")) == "True"
                                         else "oracle" if str(hp.get("avail_select_filter")) == "True" else "none")
    rt = sorted(e["vclock_now"] for e in run["agg"] if e.get("event") == "agg_round" and e.get("vclock_now") is not None)
    mu = st.median([b - a for a, b in zip(rt, rt[1:]) if b > a] or [d])  # REFL's horizon unit: the round
    picks, prio, vm, surv = 0, [], [], []
    for e in ev._events(run, "selection", task="train"):
        t = e.get("vclock_now")
        new = ev._newly_chosen(e) or e.get("chosen") or []  # REFL's random mode leaves explore / exploit empty
        if t is None or not new:
            continue
        pri = set(e.get("priority_ids") or [])
        for c in new:
            s = _share_train((gt or {}).get(c), t, t + d) >= 1.0
            picks += 1
            surv.append(s)
            if "priority_ids" in e:
                prio.append((c in pri, s, _share_train((gt or {}).get(c), t + mu, t + 2 * mu) < 0.5))
            v = ((e.get("per_trainer") or {}).get(c) or {}).get("feddance_V")
            if v is not None:
                vm.append((v, s))
    ab = [x for x in ev._events(run, "abandon_timeout") if x.get("reason") != "aware_boundary_eviction"]
    to = float(hp.get("send_timeout_wait_s") or 0)
    span = max([e.get("vclock_now") or 0 for e in run["agg"] if e.get("event") in ("agg_round", "selection")] or [0])
    hours = span / 3600 or 1
    out = dict(know=know, picks=picks, survive=sum(surv) / max(1, len(surv)), abandons_h=len(ab) / hours,
               wasted_slot_h=len(ab) * to / 3600 / hours, task_s=d)
    if prio:
        p = [s for f, s, _ in prio if f]
        q = [s for f, s, _ in prio if not f]
        gone_f = [g for f, _, g in prio if f]
        gone = [f for f, _, g in prio if g]
        out.update(prio_share=len(p) / len(prio), survive_prio=sum(p) / max(1, len(p)), survive_rest=sum(q) / max(1, len(q)),
                   precision=sum(gone_f) / max(1, len(gone_f)), recall=sum(gone) / max(1, len(gone)))
    gt = {k: v for k, v in (gt or {}).items() if k in run["trainers"]} or gt  # the leg's own clients
    ck = [e for e in run["agg"] if e.get("event") == "checkin"]
    if ck:
        n = max(1, len(gt or run["trainers"])) * hours  # syn_0 has no trace: count the leg's trainers
        out.update({f"{k}_ch": sum(e.get("kind") == k for e in ck) / n for k in ("pooled", "rejected", "missed")})
    if know == "notify" and gt:
        recv = [e for e in run["agg"] if e.get("event") == "notify_recv"]
        org = next((e["origin_ts"] for e in run["agg"] if e.get("event") == "trace_origin"), None)
        if ev._simulated(run):
            out.update(notify_fidelity(gt, float(hp.get("notify_latency_s") or 0), span))
        elif recv and org is not None:
            out.update(real_notify_fidelity(gt, [(e["recv_ts"] - org, e["end_id"], e["state"]) for e in recv], span))
    if vm:
        out["rho_V_survive"] = spearman([v for v, _ in vm], [float(s) for _, s in vm])
    return out


def notify_fidelity(gt, latency, span):
    """Sim notify: belief = trace(t - latency); seconds where belief ≠ truth, per client-hour, split missed / false."""
    from flame.availability.trace import state_at
    miss = false = 0.0
    for trace in gt.values() if latency > 0 else ():
        for f in trace.irange(0, span):
            a, b = f, min(span, f + latency)
            now, was = str(state_at(trace, f)), str(state_at(trace, f - 1e-9))
            if now.endswith("AVL_TRAIN") and not was.endswith("AVL_TRAIN"):
                miss += b - a
            elif was.endswith("AVL_TRAIN") and not now.endswith("AVL_TRAIN"):
                false += b - a
    ch = max(1, len(gt)) * max(span, 1) / 3600
    return dict(notify_missed_s_ch=miss / ch, notify_false_s_ch=false / ch)


def real_notify_fidelity(gt, recv, span, step=0.5):
    """F42-T7 real notify: belief = the client's last STATE_UPDATE received (AVL_TRAIN before any); missed / false s per client-hour,
    and flip → receipt latency (median, p95) over AVL_TRAIN flips that got a matching receipt within 60 s."""
    import bisect
    from flame.availability.trace import state_at
    by = {}
    for t, end, s in sorted(recv):
        by.setdefault(end, []).append((t, s.endswith("AVL_TRAIN")))
    miss = false = 0.0
    lat = []
    for end, trace in gt.items():
        rs = by.get(end, [])
        ts = [t for t, _ in rs]
        on = lambda t: str(state_at(trace, t)).endswith("AVL_TRAIN")  # noqa: E731
        for k in range(int(span / step)):
            t = k * step
            i = bisect.bisect_right(ts, t)
            bel, tru = (rs[i - 1][1] if i else True), on(t)
            miss += step * (tru and not bel)
            false += step * (bel and not tru)
        for f in trace.irange(0, span):
            if on(f) == on(f - 1e-9):
                continue
            j = bisect.bisect_left(ts, f)
            nxt = next((t for t, b in rs[j:] if b == on(f)), None)
            if nxt is not None and nxt - f < 60:
                lat.append(nxt - f)
    ch = max(1, len(gt)) * max(span, 1) / 3600
    lat.sort()
    return dict(notify_missed_s_ch=miss / ch, notify_false_s_ch=false / ch, notify_lat_p50=lat[len(lat) // 2] if lat else None,
                notify_lat_p95=lat[int(0.95 * (len(lat) - 1))] if lat else None, notify_flips=len(lat))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*")
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    legs = [(p, b, s) for p, b, _, _, s in rows_from_pools(a.pools) if s] + [(os.path.basename(r), "", r) for r in a.runs]
    print("| leg | baseline | knowledge | picks | survive | REFL flagged share · survive flagged / rest · precision / recall "
          "| FedDance ρ(V, survive) | abandons / h | wasted slot-h / h | check-in pooled · rejected · missed / client-h "
          "| notify missed · false s / client-h |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for phase, b, d in legs:
        if not os.path.isdir(d):
            continue
        m = re.search(r"_dbg_(?:google_speech_)?(\w+?)_n\d+", d)
        b = b or (m.group(1) if m else "?")
        r = audit(d)
        pr = (f"{r['prio_share']:.2f} · {r['survive_prio']:.2f} / {r['survive_rest']:.2f} · {r['precision']:.2f} / {r['recall']:.2f}"
              if "prio_share" in r else "–")
        fd = f"{r['rho_V_survive']:+.2f}" if "rho_V_survive" in r else "–"
        ck = f"{r['pooled_ch']:.2f} · {r['rejected_ch']:.2f} · {r['missed_ch']:.2f}" if "pooled_ch" in r else "–"
        nf = f"{r['notify_missed_s_ch']:.1f} · {r['notify_false_s_ch']:.1f}" if "notify_missed_s_ch" in r else "–"
        if r.get("notify_lat_p50") is not None:
            nf += f" (lat p50 {r['notify_lat_p50']:.2f} / p95 {r['notify_lat_p95']:.2f} s, {r['notify_flips']} flips)"
        print(f"| {phase} | {b} | {r['know']} | {r['picks']} | {r['survive']:.2f} | {pr} | {fd} | {r['abandons_h']:.1f} "
              f"| {r['wasted_slot_h']:.2f} | {ck} | {nf} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
