# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Single-run ground-truth event invariants (ROBUST_FL_READINESS S1).

Parity compares two runs; this grades ONE run against what the FL paradigm says
must happen, from its own telemetry + the registry. A run that passes every
check dispatched, trained, returned, committed and advanced its clock the way the
protocol requires, independent of any other run. Each check returns
``{"status": PASS|FAIL|SKIP|ERROR, "detail": str, ...}``; a check that raises is
ERROR (never aborts the battery).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import yaml

_EPS_S = 0.05            # float slack on modeled durations
_PASTDATE_SLACK_S = 2.0  # = _SIM_ORDER_SLACK_S
_PASTDATE_MAX_FRAC = 0.01
_BUDGET_FRAC = 0.85      # a run must reach this fraction of its budget
_REGISTRY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "_metadata",
                         "trainer_registry.yaml")


# ── loading ─────────────────────────────────────────────────────────────────
def _jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_run(run_dir: str) -> dict:
    tel = os.path.join(run_dir, "telemetry")
    agg_files = glob.glob(os.path.join(tel, "aggregator_*.jsonl"))
    agg = _jsonl(agg_files[0]) if agg_files else []
    trainers = {}
    for p in glob.glob(os.path.join(tel, "trainer_*.jsonl")):
        evs = _jsonl(p)
        if evs:
            trainers[evs[0].get("end_id")] = sorted(evs, key=lambda e: e.get("ts", 0))
    cfg = {}
    cfg_path = os.path.join(run_dir, "aggregator_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    logs = glob.glob(os.path.join(run_dir, "*_aggregator.log"))
    return {"dir": run_dir, "agg": sorted(agg, key=lambda e: e.get("ts", 0)),
            "trainers": trainers, "cfg": cfg, "agg_log": logs[0] if logs else None,
            "logs": glob.glob(os.path.join(run_dir, "*.log"))}


def _hp(run, *keys, default=None):
    hp = run["cfg"].get("hyperparameters", {}) or {}
    for k in keys:
        if hp.get(k) is not None:
            return hp[k]
    return default


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def _simulated(run) -> bool:
    tm = str(_hp(run, "time_mode", default="")).lower()
    if tm:
        return tm.startswith("sim")
    return any(e.get("time_mode") == "sim" for e in run["agg"] if e.get("event") == "dispatch")


def _is_async(run) -> bool:
    return str(run["cfg"].get("optimizer", {}).get("sort", "")).lower() == "fedbuff"


def _agg_goal(run):
    g = _hp(run, "aggGoal", "agg_goal")
    return int(g) if g else None


def _concurrency(run):
    c = (run["cfg"].get("selector", {}).get("kwargs", {}) or {}).get("c")
    return int(c) if c else None


def _events(run, name, **match):
    return [e for e in run["agg"] if e.get("event") == name
            and all(e.get(k) == v for k, v in match.items())]


def _train_commits(run):
    return [e for e in run["agg"] if e.get("event") == "agg_round"
            and e.get("task_to_perform", "train") == "train"]


def _registry_delays() -> dict:
    """end_id (task_id) -> raw registry D."""
    try:
        with open(_REGISTRY) as f:
            reg = yaml.safe_load(f)
    except OSError:
        return {}
    return {t["task_id"]: float(t["training_delay_s"]) for t in reg.get("trainers", {}).values()}


def _effective_d(run, raw: float) -> float:
    floor_s = float(_hp(run, "trainingDelayFloorSeconds", "training_delay_floor_s", default=0.0) or 0.0)
    factor = _hp(run, "trainingDelayFactor", "training_delay_factor")
    d = max(raw, floor_s)
    return d / float(factor) if factor else d


def _trace(run) -> str:
    """Availability trace name: asyncfl stacks use client_notify, oort/refl/feddance trackTrainerAvail."""
    for key in ("client_notify", "trackTrainerAvail"):
        block = _hp(run, key, default={})
        if isinstance(block, dict) and block.get("trace"):
            return str(block["trace"])
    return "syn_0"


def _res(status, detail="", **kw):
    return {"status": status, "detail": detail, **kw}


# ── checks ──────────────────────────────────────────────────────────────────
def ev0_clean_exit(run):
    """No tracebacks in any log; the aggregator logged its budget stop."""
    tb = 0
    for p in run["logs"]:
        with open(p, errors="replace") as f:
            tb += sum(1 for line in f if "Traceback (most recent call last)" in line)
    stopped = False
    if run["agg_log"]:
        with open(run["agg_log"], errors="replace") as f:
            stopped = any("stopping run" in line.lower() for line in f)  # incl. [SIM_WALL_CEILING]
    status = "FAIL" if (tb or not stopped) else "PASS"
    return _res(status, f"tracebacks={tb} stopping_run={stopped}", tracebacks=tb, stopped=stopped)


def ev1_progress(run):
    """The run committed at least two aggregation goals' worth of train updates."""
    n = len(_train_commits(run))
    k = _agg_goal(run) or 1
    return _res("PASS" if n >= 2 * k else "FAIL", f"train_commits={n} need>={2 * k}", n=n)


def ev2_task_alternation(run):
    """Trainer side: recv and send strictly alternate (one instruction in flight),
    and each send answers the round it received."""
    bad_double_recv = bad_orphan_send = bad_round = 0
    examples = []
    for tid, evs in run["trainers"].items():
        pending = None
        for e in evs:
            k = e.get("event")
            if k == "task_recv":
                if pending is not None:
                    bad_double_recv += 1
                    examples.append((tid[-4:], "recv_while_busy", e.get("round")))
                pending = e
            elif k == "task_send":
                if pending is None:
                    bad_orphan_send += 1
                    examples.append((tid[-4:], "send_without_recv", e.get("round")))
                elif pending.get("round") != e.get("round"):
                    bad_round += 1
                    examples.append((tid[-4:], "round_mismatch", pending.get("round"), e.get("round")))
                pending = None
    n_bad = bad_double_recv + bad_orphan_send + bad_round
    if not run["trainers"]:
        return _res("SKIP", "no trainer telemetry")
    return _res("PASS" if n_bad == 0 else "FAIL",
                f"recv_while_busy={bad_double_recv} send_without_recv={bad_orphan_send} "
                f"round_mismatch={bad_round}", examples=examples[:10])


def ev3_duration_model(run):
    """Trainer's modeled duration = max(gpu, D) with D = registry D (floor, factor);
    sim: sct = send + duration + leg."""
    reg = _registry_delays()
    sim = _simulated(run)
    n = bad_max = bad_budget = bad_sct = bad_stamp = 0
    examples = []
    legs = []
    # S1: the trainer's clock is the stamp it was dispatched with (catches a frozen trainer clock).
    stamps = defaultdict(set)
    for e in _events(run, "dispatch"):
        if e.get("sim_send_ts") is not None:
            stamps[(e.get("end_id"), e.get("round"), e.get("task", "train"))].add(round(e["sim_send_ts"], 6))
    for tid, evs in run["trainers"].items():
        d_true = _effective_d(run, reg[tid]) if tid in reg else None
        for e in evs:
            if e.get("event") != "trainer_round" or e.get("task_to_perform", "train") != "train":
                continue
            n += 1
            gpu, dur, budget = e.get("real_gpu_time_s"), e.get("sim_round_duration_s"), e.get("training_budget_s")
            if None not in (gpu, dur, budget) and abs(dur - max(gpu, budget)) > _EPS_S:
                bad_max += 1
                examples.append((tid[-4:], "dur!=max(gpu,D)", dur, gpu, budget))
            if d_true is not None and budget is not None and budget > 0 and abs(budget - d_true) > 1e-3:
                bad_budget += 1
                examples.append((tid[-4:], "D!=registry", budget, d_true))
            if sim and None not in (e.get("sim_send_ts"), e.get("sim_completion_ts"), dur):
                legs.append((tid, e["sim_completion_ts"] - e["sim_send_ts"] - dur))
                sent = stamps.get((tid, e.get("round"), "train"))
                if sent and round(e["sim_send_ts"], 6) not in sent:
                    bad_stamp += 1
                    examples.append((tid[-4:], "send_ts!=dispatch", e["sim_send_ts"], sorted(sent)))
    if n == 0:
        return _res("SKIP", "no train trainer_round events")
    # sct = send + dur + leg, where the (trainer-config) leg is one non-negative constant per run.
    leg = None
    if legs:
        leg = sorted(x for _, x in legs)[len(legs) // 2]
        for tid, x in legs:
            if abs(x - leg) > _EPS_S or x < -_EPS_S:
                bad_sct += 1
                examples.append((tid[-4:], "sct!=send+dur+leg", round(x, 3), round(leg, 3)))
    bad = bad_max + bad_budget + bad_sct + bad_stamp
    return _res("PASS" if bad == 0 else "FAIL",
                f"rounds={n} dur!=max={bad_max} D!=registry={bad_budget} sct!=send+dur+leg={bad_sct}"
                + (f" send_ts!=dispatch={bad_stamp}" if sim else "")
                + (f" leg={leg:.3f}s" if leg is not None else ""), examples=examples[:10])


def ev4_real_sleep(run):
    """Real mode: a train task's wall span covers its modeled D (the device really waited)."""
    if _simulated(run):
        return _res("SKIP", "sim mode")
    budgets = {}
    for tid, evs in run["trainers"].items():
        for e in evs:
            if e.get("event") == "trainer_round":
                budgets[(tid, e.get("round"))] = e.get("training_budget_s") or 0.0
    n = bad = 0
    examples = []
    for tid, evs in run["trainers"].items():
        for e in evs:
            if e.get("event") != "task_send" or e.get("task_to_perform") != "train":
                continue
            b = budgets.get((tid, e.get("round")))
            if b is None or e.get("wall_recv_ts") is None:
                continue
            n += 1
            span = e["wall_send_ts"] - e["wall_recv_ts"]
            if span < b - 0.2:
                bad += 1
                examples.append((tid[-4:], round(span, 3), b))
    if n == 0:
        return _res("SKIP", "no matched real train sends")
    return _res("PASS" if bad == 0 else "FAIL", f"sends={n} short_of_D={bad}", examples=examples[:10])


def ev5_commit_accounting(run):
    """Every committed train update was sent by that trainer (commits <= sends), and
    (async) none is lost: a sent version never committed while a LATER version of the
    same trainer did (trailing ones were cut off by the stop). Abandons/withholds excuse one each."""
    sends, sent_v = Counter(), defaultdict(set)
    for tid, evs in run["trainers"].items():
        for e in evs:
            if e.get("event") == "task_send" and e.get("task_to_perform") == "train":
                sends[tid] += 1
                sent_v[tid].add(e.get("round"))
    if not sends:
        return _res("SKIP", "no trainer telemetry")
    commits, commit_v = Counter(), defaultdict(set)
    for e in _train_commits(run):
        for t, st in zip(e.get("contributing_trainers") or [], e.get("staleness") or [None] * 99):
            commits[t] += 1
            if st is not None and e.get("round") is not None:
                commit_v[t].add(e["round"] - st)
    excused = Counter(e.get("end_id") for e in run["agg"]
                      if e.get("event") in ("abandon_timeout", "withheld_delivery"))
    over = {t: (commits[t], sends[t]) for t in commits if commits[t] > sends.get(t, 0)}
    lost = {}
    if _is_async(run):
        for t, vs in sent_v.items():
            top = max(commit_v[t], default=None)
            gone = sorted(v for v in vs if v is not None and v not in commit_v[t]
                          and top is not None and v < top)
            if len(gone) > excused[t]:
                lost[t] = gone[:5]
    ok = not over and not lost
    return _res("PASS" if ok else "FAIL",
                f"trainers={len(sends)} commits>sends={len(over)} lost(async)={len(lost)}"
                + ("" if _is_async(run) else " (sync: loss not graded, stale rejects allowed)"),
                over=dict(list(over.items())[:5]), lost=dict(list(lost.items())[:5]))


def ev6_staleness(run):
    bad = [e.get("staleness") for e in _train_commits(run)
           if any(s is not None and s < 0 for s in (e.get("staleness") or []))]
    n = len(_train_commits(run))
    if n == 0:
        return _res("SKIP", "no commits")
    return _res("PASS" if not bad else "FAIL", f"commits={n} negative_staleness={len(bad)}")


def ev7_agg_goal_cadence(run):
    """Async: every closed round aggregated exactly agg_goal train updates; both: round
    indices advance by one with no gaps (sync: no empty rounds, none over agg_goal)."""
    k = _agg_goal(run)
    commits = _train_commits(run)
    if not k or not commits:
        return _res("SKIP", "no agg_goal or commits")
    per_round = Counter(e.get("round") for e in commits)
    rounds = sorted(r for r in per_round if r is not None)
    if not _is_async(run):
        # A sync round commits >=1 update and indices advance by one (FX-D10: no empty rounds).
        over = {r: c for r, c in per_round.items() if c > k}
        empty = [e.get("round") for e in commits if not (e.get("contributing_trainers") or [])]
        gaps = [(a, b) for a, b in zip(rounds, rounds[1:]) if b - a != 1]
        return _res("PASS" if not (over or empty or gaps) else "FAIL",
                    f"sync rounds={len(rounds)} rounds_over_agg_goal={len(over)} "
                    f"empty={len(empty)} gaps={len(gaps)}",
                    over=dict(list(over.items())[:5]), empty=empty[:5], gaps=gaps[:5])
    closed = rounds[:-1]
    wrong = {r: per_round[r] for r in closed if per_round[r] != k}
    gaps = [(a, b) for a, b in zip(rounds, rounds[1:]) if b - a != 1]
    ok = not wrong and not gaps
    return _res("PASS" if ok else "FAIL",
                f"rounds={len(rounds)} wrong_count={len(wrong)} gaps={len(gaps)}",
                wrong=dict(list(wrong.items())[:5]), gaps=gaps[:5])


def ev8_concurrency_cap(run):
    """Async: a selection that picks trainers never leaves more than c in flight. (Held
    slots -- buffered updates, eval tasks -- may exceed c; the selector must then pick 0.)"""
    c = _concurrency(run)
    sels = _events(run, "selection", task="train")
    if not _is_async(run) or not c or not sels:
        return _res("SKIP", "sync or no concurrency/selection")
    choosing = [e for e in sels if (e.get("num_chosen") or 0) > 0 and e.get("in_flight") is not None]
    over = [(e.get("round"), e["in_flight"], e["num_chosen"]) for e in choosing if e["in_flight"] > c]
    held_over = sum(1 for e in sels if (e.get("in_flight") or 0) > c)
    return _res("PASS" if not over else "FAIL",
                f"choosing_selections={len(choosing)} c={c} over_cap_while_choosing={len(over)} "
                f"(held>c, chose 0: {held_over})", examples=over[:10])


def _newly_chosen(sel):
    """Oort-family `chosen` also lists carried in-flight stragglers; new picks are explore+exploit."""
    if "exploit_ids" in sel or "explore_ids" in sel:
        return (sel.get("explore_ids") or []) + (sel.get("exploit_ids") or [])
    return sel.get("chosen") or []


def ev9_selector_state(run):
    """State snapshot at selection: a newly chosen trainer was not holding an uncommitted
    update, and (availability-aware) was not believed UN_AVL."""
    aware = _truthy(_hp(run, "avail_select_filter", default=False))
    busy = unavail = n = 0
    examples = []
    for e in _events(run, "selection", task="train"):
        pt = e.get("per_trainer")
        if not isinstance(pt, dict):
            continue
        for t in _newly_chosen(e):
            s = pt.get(t) or {}
            n += 1
            if s.get("in_pending_commit") is True:
                busy += 1
                examples.append((t[-4:], "chosen_while_pending_commit", e.get("round")))
            if aware and s.get("avl_state") == "UN_AVL":
                unavail += 1
                examples.append((t[-4:], "chosen_while_UN_AVL", e.get("round")))
    if n == 0:
        return _res("SKIP", "no per_trainer snapshots")
    return _res("PASS" if busy + unavail == 0 else "FAIL",
                f"choices={n} pending_commit={busy} un_avl(aware={aware})={unavail}",
                examples=examples[:10])


def ev10_dispatch_one_in_flight(run):
    """Sim aggregator: never dispatch a train task to an end whose previous train task
    has neither committed nor been abandoned."""
    if not _simulated(run):
        return _res("SKIP", "dispatch events are sim-only")
    timeline = []
    for e in run["agg"]:
        k = e.get("event")
        if k == "dispatch" and e.get("task") == "train":
            timeline.append((e["ts"], 0, "d", e["end_id"]))
        elif k == "agg_round" and e.get("task_to_perform", "train") == "train":
            for t in e.get("contributing_trainers") or []:
                timeline.append((e["ts"], 1, "c", t))
        elif k in ("abandon_timeout",):
            timeline.append((e["ts"], 1, "c", e["end_id"]))
    if not timeline:
        return _res("SKIP", "no dispatch events")
    outstanding = set()
    viol = []
    for ts, _, kind, end in sorted(timeline):
        if kind == "d":
            if end in outstanding:
                viol.append(end[-4:])
            outstanding.add(end)
        else:
            outstanding.discard(end)
    n = sum(1 for x in timeline if x[2] == "d")
    return _res("PASS" if not viol else "FAIL", f"dispatches={n} redispatch_while_outstanding={len(viol)}",
                examples=viol[:10])


def ev11_vclock(run):
    """Sim: the clock never moves backwards over train commits, and past-dated commits
    (clock already beyond sct by > slack, excluding withheld deliveries) stay rare."""
    if not _simulated(run):
        return _res("SKIP", "real mode")
    commits = _train_commits(run)
    vc = [e.get("vclock_now") for e in commits if e.get("vclock_now") is not None]
    if not vc:
        return _res("SKIP", "no vclock_now")
    back = sum(1 for a, b in zip(vc, vc[1:]) if b < a - 1e-6)
    withheld = {(e.get("end_id"), round(e.get("actual_commit_ts") or -1, 3))
                for e in _events(run, "withheld_delivery")}
    past = 0
    for e in commits:
        gap = e.get("commit_gap_s")
        who = (e.get("contributing_trainers") or [None])[0]
        if gap is not None and gap > _PASTDATE_SLACK_S and (who, round(e.get("vclock_now") or -1, 3)) not in withheld:
            past += 1
    frac = past / len(commits)
    ok = back == 0 and frac <= _PASTDATE_MAX_FRAC
    return _res("PASS" if ok else "FAIL",
                f"commits={len(commits)} backwards={back} pastdated={past} ({frac:.2%}, max {_PASTDATE_MAX_FRAC:.0%})")


def ev12_reached_budget(run):
    """The run consumed its budget (vclock in sim, wall span in real) instead of
    stalling or dying early."""
    budget = _hp(run, "max_experiment_runtime_s", "maxExperimentRuntimeS")
    commits = _train_commits(run)
    if not budget or not commits:
        return _res("SKIP", "no budget or commits")
    budget = float(budget)
    # Span of commits AND selections: a trace-starved run still consumes its budget (FX-D12).
    sels = [e for e in run["agg"] if e.get("event") == "selection"]
    if _simulated(run):
        reached = max(e.get("vclock_now") or 0.0 for e in commits + sels)
    else:
        ts = [e["ts"] for e in commits + sels]
        reached = max(ts) - (min(e["ts"] for e in sels) if sels else commits[0]["ts"])
    ok = reached >= _BUDGET_FRAC * budget
    return _res("PASS" if ok else "FAIL", f"reached={reached:.0f}s budget={budget:.0f}s (need {_BUDGET_FRAC:.0%})")


def ev13_no_stall(run):
    """No wall-clock hang between consecutive train commits (real: > 20x median gap,
    floor 120s; sim: floor 60s). Scarcity waits make this informational off syn_0."""
    ts = [e["ts"] for e in _train_commits(run)]
    if len(ts) < 3:
        return _res("SKIP", "too few commits")
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    med = gaps[len(gaps) // 2]
    limit = max(60.0 if _simulated(run) else 120.0, 20 * med)
    worst = gaps[-1]
    trace = _trace(run)
    status = "PASS" if worst <= limit else ("WARN" if trace != "syn_0" else "FAIL")
    return _res(status, f"max_gap={worst:.1f}s median={med:.2f}s limit={limit:.0f}s trace={trace}")


def ev14_eval_sane(run):
    evs = _events(run, "agg_eval")
    if not evs:
        return _res("SKIP", "no agg_eval")
    def _ok(x, lo=-math.inf, hi=math.inf):  # None check, not `or`: 0.0 is a valid value
        return x is not None and math.isfinite(x) and lo <= x <= hi
    bad = [e.get("round") for e in evs
           if not _ok(e.get("test-accuracy"), 0.0, 1.0) or not _ok(e.get("test-loss"))]
    return _res("PASS" if not bad else "FAIL", f"evals={len(evs)} bad={len(bad)}", examples=bad[:10])


def ev15_one_task_per_version(run):
    """FX-D9: each (trainer, task, model version) is answered once and committed once, and no eval
    follows a train at the same version (it would repeat the utility); the sim aggregator re-dispatches
    the same one only after a timeout under a retry policy."""
    dup_send = 0
    for tid, evs in run["trainers"].items():
        sends = [(e.get("task_to_perform"), e.get("round")) for e in evs if e.get("event") == "task_send"]
        dup_send += sum(c - 1 for c in Counter(sends).values() if c > 1)
        trained = set()  # eval@v before train@v is allowed (FX-L26); only eval after train counts
        for t, v in sends:
            if t == "train":
                trained.add(v)
            elif t == "eval" and v in trained:
                dup_send += 1
    discards = sum(1 for evs in run["trainers"].values() for e in evs if e.get("event") == "task_discard")
    commits = Counter()
    for e in _train_commits(run):
        for t, st in zip(e.get("contributing_trainers") or [], e.get("staleness") or []):
            if st is not None and e.get("round") is not None:
                commits[(t, e["round"] - st)] += 1
    dup_commit = sum(c - 1 for c in commits.values() if c > 1)
    retry_ok = str(_hp(run, "task_retry_policy", "taskRetryPolicy", default="none")).lower() != "none"
    last, timed_out, dup_dispatch = {}, set(), 0
    for e in run["agg"]:
        k = e.get("event")
        if k == "abandon_timeout" and str(e.get("reason", "")).startswith("abandon"):
            timed_out.add(e.get("end_id"))
        elif k == "dispatch":
            key = (e.get("end_id"), e.get("task"))
            if last.get(key) == e.get("round") and not (retry_ok and key[0] in timed_out):
                dup_dispatch += 1
            elif key[1] == "eval" and last.get((key[0], "train")) == e.get("round"):
                dup_dispatch += 1  # eval after train at the same version
            last[key] = e.get("round")
            timed_out.discard(key[0])
    if not run["trainers"] and not run["agg"]:
        return _res("SKIP", "no telemetry")
    bad = dup_send + dup_commit + dup_dispatch
    return _res("PASS" if bad == 0 else "FAIL",
                f"dup_send={dup_send} dup_commit={dup_commit} dup_dispatch={dup_dispatch} "
                f"trainer_discards={discards}")


def _ground_truth(run):
    """task_id -> SortedDict trace at the run's scale (FLAME_TRACE_TIME_SCALE); None off-trace."""
    if _trace(run) == "syn_0":
        return None
    from flame.availability.trace import read_trainer_unavailability
    return read_trainer_unavailability(_trace(run))


def ev16_withheld_delivery(run):
    """Sim: a train update whose trainer is UN_AVL at its sct commits only as a withheld
    delivery, at delivery_ts = next AVL after sct (FX-L12); catches committing at sct."""
    if not _simulated(run):
        return _res("SKIP", "sim-only (real send-gate is trainer-side, A8)")
    gt = _ground_truth(run)
    if not gt:
        return _res("SKIP", "no availability trace")
    from flame.availability.trace import next_avail_after, state_at
    from flame.config import TrainerAvailState
    held = {}
    early = bad_dts = 0
    examples = []
    for e in _events(run, "withheld_delivery"):
        held[(e.get("end_id"), round(float(e["sct"]), 3))] = e
        tr = gt.get(e.get("end_id"))
        if tr is not None:
            sct = float(e["sct"])  # = ClientAvailability.compute_delivery_ts
            want = sct if state_at(tr, sct) != TrainerAvailState.UN_AVL else next_avail_after(tr, sct)
            if abs(float(e["delivery_ts"]) - want) > _EPS_S:
                bad_dts += 1
                examples.append((e["end_id"][-4:], "delivery_ts!=next_avl", e["delivery_ts"], want))
        if e.get("actual_commit_ts") is not None and float(e["actual_commit_ts"]) < float(e["delivery_ts"]) - _EPS_S:
            early += 1
            examples.append((e["end_id"][-4:], "commit<delivery_ts", e["actual_commit_ts"], e["delivery_ts"]))
    committed = {(t, e["round"] - st) for e in _train_commits(run) if e.get("round") is not None
                 for t, st in zip(e.get("contributing_trainers") or [], e.get("staleness") or [])
                 if st is not None}
    unheld = n = 0
    for tid, evs in run["trainers"].items():
        tr = gt.get(tid)
        if not tr:
            continue
        for e in evs:
            if (e.get("event") != "trainer_round" or e.get("task_to_perform", "train") != "train"
                    or e.get("sim_completion_ts") is None or (tid, e.get("round")) not in committed):
                continue
            n += 1
            sct = float(e["sim_completion_ts"])
            if state_at(tr, sct) == TrainerAvailState.UN_AVL and (tid, round(sct, 3)) not in held:
                unheld += 1
                examples.append((tid[-4:], "UN_AVL@sct committed unheld", e.get("round"), round(sct, 2)))
    bad = unheld + early + bad_dts
    return _res("PASS" if bad == 0 else "FAIL",
                f"commits={n} withheld={len(held)} unavail_committed_unheld={unheld} "
                f"commit_before_delivery={early} delivery_ts_mismatch={bad_dts}", examples=examples[:10])


CHECKS = [ev0_clean_exit, ev1_progress, ev2_task_alternation, ev3_duration_model, ev4_real_sleep,
          ev5_commit_accounting, ev6_staleness, ev7_agg_goal_cadence, ev8_concurrency_cap,
          ev9_selector_state, ev10_dispatch_one_in_flight, ev11_vclock, ev12_reached_budget,
          ev13_no_stall, ev14_eval_sane, ev15_one_task_per_version, ev16_withheld_delivery]


def check_run(run_dir: str) -> dict:
    run = load_run(run_dir)
    results = {}
    for fn in CHECKS:
        name = re.sub(r"^ev(\d+)_", lambda m: f"EV{m.group(1)}_", fn.__name__)
        try:
            results[name] = fn(run)
        except Exception as ex:  # noqa: BLE001 -- a broken check must not hide the others
            results[name] = _res("ERROR", f"{type(ex).__name__}: {ex}")
    counts = Counter(r["status"] for r in results.values())
    return {"run_dir": run_dir, "mode": "sim" if _simulated(run) else "real",
            "passed": counts["FAIL"] == 0 and counts["ERROR"] == 0,
            "counts": dict(counts), "checks": results}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)
    out, ok = [], True
    for d in args.run_dirs:
        r = check_run(d)
        out.append(r)
        ok &= r["passed"]
        print(f"\n[{'PASS' if r['passed'] else 'FAIL'}] {os.path.basename(d.rstrip('/'))} "
              f"({r['mode']}) {r['counts']}")
        for name, c in r["checks"].items():
            print(f"  {c['status']:5s} {name:32s} {c['detail']}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
