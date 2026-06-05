"""Canonical real/sim parity checks — importable single source of truth.

This module holds the *pure* loaders, helpers, and invariant/parity functions
used both by the `compare_parity.py` CLI and by the pytest suite
(`tests/mode/test_parity_checks.py` unit tests + the opt-in
`tests/mode/test_real_sim_e2e_parity.py` end-to-end check). Keeping the logic
here means the CLI report and the automated tests assert the *same* thing.

Two families of functions:
  * loaders/helpers: parse aggregator/trainer telemetry JSONL into typed lists.
  * checks: each takes parsed events and returns a small result dict with an
    ``ok`` bool plus metrics. Thresholds follow the determinism analysis in
    TELEMETRY_AND_SPEEDUP_PLAN.md (Task 3): exact-matchable quantities are
    asserted tightly; distributional ones use tolerances.

All functions are dependency-free (stdlib only) so they run in the default
pytest without MQTT/GPU/scipy.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def short(end_id: str) -> str:
    return end_id[-4:] if end_id else "None"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)


def ks_stat(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no scipy needed)."""
    if not a or not b:
        return float("nan")
    combined = sorted(set(a + b))
    na, nb = len(a), len(b)
    sa, sb = sorted(a), sorted(b)
    ia = ib = 0
    d = 0.0
    for v in combined:
        while ia < na and sa[ia] <= v:
            ia += 1
        while ib < nb and sb[ib] <= v:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    return d


# ── loaders ────────────────────────────────────────────────────────────────

def load_agg_jsonl(path: str) -> dict:
    """Parse an aggregator telemetry JSONL into typed, sorted lists."""
    selection_train: list[dict] = []
    agg_rounds: list[dict] = []
    agg_evals: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ev = e.get("event")
            if ev == "selection" and e.get("task") == "train":
                selection_train.append(e)
            elif ev == "agg_round":
                agg_rounds.append(e)
            elif ev == "agg_eval":
                agg_evals.append(e)
    selection_train.sort(key=lambda x: (x["round"], x["ts"]))
    agg_rounds.sort(key=lambda x: (x["round"], x.get("agg_goal_count", 0), x["ts"]))
    agg_evals.sort(key=lambda x: x["round"])
    return {
        "selection_train": selection_train,
        "agg_rounds": agg_rounds,
        "agg_evals": agg_evals,
    }


def load_trainer_jsonl_dir(telemetry_dir: Optional[str]) -> dict:
    """Load all trainer_*.jsonl from a telemetry dir.

    Returns {short_id: {"task_recv": [...], "trainer_round": [...]}}.
    """
    if not telemetry_dir:
        return {}
    d = Path(telemetry_dir)
    result: dict = {}
    for f in sorted(d.glob("trainer_*.jsonl")):
        short_id = f.stem[-4:]
        task_recv_evs, trainer_round_evs = [], []
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = e.get("event")
                if ev == "task_recv":
                    task_recv_evs.append(e)
                elif ev == "trainer_round":
                    trainer_round_evs.append(e)
        result[short_id] = {
            "task_recv": task_recv_evs,
            "trainer_round": trainer_round_evs,
        }
    return result


# ── cross-mode parity checks (real vs sim) ───────────────────────────────────

def _by_round_selection(selection_train: list[dict]) -> dict:
    out: dict[int, set] = {}
    for e in selection_train:
        out.setdefault(e["round"], set()).update(e.get("chosen", []))
    return out


# Selectors whose per-round SET selection is a deterministic function of
# (candidate set, seed) — so real and sim must select identically and the Jaccard
# check is ENFORCED. Currently EMPTY, on purpose. Every shipped selector samples
# its picks with the seeded RNG from a candidate LIST ordered by channel-join
# order (channel._ends insertion order). Trainers are real processes in both
# modes (simulation only virtualizes their training *sleeps*), and their join
# order varies run-to-run with physical spawn/connect timing — independent of the
# time mode — so the same seed draws a different subset in real vs sim even from
# an identical candidate *set*, and that first difference then cascades through
# the shared RNG stream. (The much larger early-round divergence, where sim
# outran trainer joins and selected from a half-filled pool, is removed
# separately by the `min_trainers_to_start` join barrier.) For these stochastic
# selectors the meaningful, achievable invariant is participation-FREQUENCY
# parity (`participation_parity`, asserted separately) — not exact per-round set
# identity. To enforce exact selection parity for a selector, make it sort its
# candidate list before sampling (so selection is a pure function of set+seed),
# then add its telemetry class name here.
DETERMINISTIC_SELECTORS: set[str] = set()


def _selector_name(*loaded: dict) -> str:
    """Selector class name from selection telemetry (e.g. 'OortSelector'); '' if
    unknown (the random selector emits no selector field)."""
    for d in loaded:
        for e in d.get("selection_train", []):
            name = e.get("selector")
            if name:
                return name
    return ""


def selection_parity(real: dict, sim: dict, max_rounds: Optional[int] = None,
                     warn_jaccard: float = 0.7) -> dict:
    """Per-round selection overlap (Jaccard).

    Enforced only for DETERMINISTIC_SELECTORS (see its docstring). For stochastic
    selectors the result is reported but not enforced (``gated=True``): exact
    per-round set identity is unattainable across real/sim (join-order-dependent
    candidate ordering + exploration RNG), so ``participation_parity`` is the
    enforced selection invariant instead. Jaccard is still surfaced as a signal."""
    r = _by_round_selection(real["selection_train"])
    s = _by_round_selection(sim["selection_train"])
    rounds = sorted(set(r) & set(s))
    if max_rounds is not None:
        rounds = [x for x in rounds if x <= max_rounds]
    js, exact = [], 0
    for rd in rounds:
        j = jaccard(r[rd], s[rd])
        js.append(j)
        if j == 1.0:
            exact += 1
    mean_j = sum(js) / len(js) if js else float("nan")
    selector = _selector_name(real, sim)
    gated = bool(selector) and selector not in DETERMINISTIC_SELECTORS
    enforced_ok = (not js) or mean_j >= warn_jaccard
    return {
        "ok": True if gated else enforced_ok,
        "gated": gated,
        "selector": selector or None,
        "rounds_compared": len(rounds),
        "mean_jaccard": round(mean_j, 3) if js else None,
        "exact_match_frac": round(exact / len(js), 3) if js else None,
    }


def aggregation_sequence_parity(real: dict, sim: dict,
                                max_rounds: Optional[int] = None) -> dict:
    """Per-round set of contributing trainers matches across modes."""
    def by_round(agg_rounds):
        out: dict[int, set] = {}
        for e in agg_rounds:
            out.setdefault(e["round"], set()).update(e.get("contributing_trainers", []))
        return out

    r, s = by_round(real["agg_rounds"]), by_round(sim["agg_rounds"])
    rounds = sorted(set(r) & set(s))
    if max_rounds is not None:
        rounds = [x for x in rounds if x <= max_rounds]
    matches = sum(1 for rd in rounds if r[rd] == s[rd])
    return {
        "ok": (not rounds) or matches == len(rounds),
        "rounds_compared": len(rounds),
        "exact_set_match_frac": round(matches / len(rounds), 3) if rounds else None,
    }


def staleness_parity(real: dict, sim: dict, warn_ks: float = 0.2,
                     warn_mean_diff: float = 1.0) -> dict:
    """Staleness distributions should match within tolerance."""
    def vals(agg_rounds):
        out = []
        for e in agg_rounds:
            out.extend(e.get("staleness", []))
        return out

    rv, sv = vals(real["agg_rounds"]), vals(sim["agg_rounds"])
    rm, _ = mean_std(rv)
    sm, _ = mean_std(sv)
    ks = ks_stat(rv, sv)
    mean_diff = abs(rm - sm) if not (math.isnan(rm) or math.isnan(sm)) else float("nan")
    ok = True
    if not math.isnan(ks):
        ok = ks <= warn_ks and (math.isnan(mean_diff) or mean_diff <= warn_mean_diff)
    # staleness must never be negative in either mode
    nonneg = all(v >= 0 for v in rv + sv)
    return {
        "ok": ok and nonneg,
        "real_mean": round(rm, 3) if not math.isnan(rm) else None,
        "sim_mean": round(sm, 3) if not math.isnan(sm) else None,
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        "all_nonnegative": nonneg,
    }


def participation_parity(real: dict, sim: dict, warn_avg_diff: float = 10.0) -> dict:
    """Per-trainer participation counts match within tolerance."""
    def counts(agg_rounds):
        c = collections.Counter()
        for e in agg_rounds:
            for t in e.get("contributing_trainers", []):
                c[t] += 1
        return c

    rc, sc = counts(real["agg_rounds"]), counts(sim["agg_rounds"])
    trainers = set(rc) | set(sc)
    diffs = [abs(rc.get(t, 0) - sc.get(t, 0)) for t in trainers]
    avg = sum(diffs) / len(diffs) if diffs else 0.0
    return {
        "ok": avg <= warn_avg_diff,
        "avg_diff": round(avg, 2),
        "max_diff": max(diffs) if diffs else 0,
    }


# ── sim-mode invariants (the sanity-check list) ──────────────────────────────

def sim_send_ts_ok(real_trainers: dict, sim_trainers: dict) -> dict:
    """Real: task_recv.sim_send_ts null. Sim: non-null and increasing (>0 after r1)."""
    issues = []
    for tid, data in real_trainers.items():
        bad = [e["sim_send_ts"] for e in data.get("task_recv", [])
               if e.get("sim_send_ts") is not None]
        if bad:
            issues.append(f"real/{tid}: unexpected non-null sim_send_ts {bad[:3]}")
    for tid, data in sim_trainers.items():
        evs = [e for e in data.get("task_recv", []) if e.get("round", 0) > 1]
        if not evs:
            continue
        vals = [e.get("sim_send_ts") for e in evs]
        if any(v is None for v in vals):
            issues.append(f"sim/{tid}: null sim_send_ts (vclock stamp missing)")
        elif not any((v or 0) > 0 for v in vals):
            issues.append(f"sim/{tid}: all sim_send_ts==0 (vclock not advancing)")
    return {"ok": not issues, "issues": issues}


def gpu_budget_ok(trainers: dict, warn_overrun_frac: float = 0.25) -> dict:
    """Fraction of rounds where real_gpu_time_s exceeded the modeled budget."""
    fracs = []
    for _tid, d in trainers.items():
        evs = [e for e in d.get("trainer_round", [])
               if "real_gpu_time_s" in e and e.get("training_budget_s", 0) > 0]
        if not evs:
            continue
        over = sum(1 for e in evs if e["real_gpu_time_s"] > e["training_budget_s"])
        fracs.append(over / len(evs))
    if not fracs:
        return {"ok": True, "note": "no training_budget_s telemetry", "mean_overrun_frac": None}
    mean_frac = sum(fracs) / len(fracs)
    return {"ok": mean_frac <= warn_overrun_frac,
            "mean_overrun_frac": round(mean_frac, 3),
            "trainers_with_any_overrun": int(sum(1 for f in fracs if f > 0))}


def sim_commit_order_monotone(sim: dict) -> dict:
    """In sim mode the per-update vclock_now stamped on agg_round events must be
    non-decreasing (the virtual clock only advances forward)."""
    seq = [e.get("vclock_now") for e in sim["agg_rounds"] if e.get("vclock_now") is not None]
    monotone = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
    return {"ok": monotone, "n_stamped": len(seq), "monotone": monotone}


def agg_goal_cycles_ok(agg: dict, agg_goal: int) -> dict:
    """agg_goal_count within each round should cycle 1..agg_goal (no update lost
    or double-counted before a round closes)."""
    if agg_goal <= 0:
        return {"ok": True, "note": "agg_goal unknown"}
    bad_rounds = []
    by_round: dict[int, list] = {}
    for e in agg["agg_rounds"]:
        by_round.setdefault(e["round"], []).append(e.get("agg_goal_count"))
    for rd, counts in by_round.items():
        present = [c for c in counts if c is not None]
        if present and max(present) > agg_goal:
            bad_rounds.append(rd)
    return {"ok": not bad_rounds, "rounds_over_goal": bad_rounds}


def commit_sequence(agg: dict) -> list:
    """Mode-agnostic *logical* sequence of committed updates: one entry per
    aggregated update in (round, agg_goal_count) order — no wall-clock. If sim
    faithfully mimics real ordering, the two sequences match; the first mismatch
    localizes a control/ordering bug independent of timing."""
    evs = sorted(agg["agg_rounds"], key=lambda e: (e["round"], e.get("agg_goal_count", 0)))
    seq = []
    for e in evs:
        ends = e.get("contributing_trainers", [])
        stales = e.get("staleness", [])
        for i, end in enumerate(ends):
            seq.append({"round": e["round"], "end": short(end),
                        "staleness": stales[i] if i < len(stales) else None})
    return seq


def first_divergence(real_agg: dict, sim_agg: dict, ctx: int = 2) -> dict:
    """First index where the real vs sim commit sequences differ (by end+round),
    with a small context window around it. ``index=None`` => sequences agree on
    the shared prefix (lengths may still differ)."""
    rs, ss = commit_sequence(real_agg), commit_sequence(sim_agg)
    for i in range(min(len(rs), len(ss))):
        if (rs[i]["end"], rs[i]["round"]) != (ss[i]["end"], ss[i]["round"]):
            lo = max(0, i - ctx)
            return {"index": i, "real": rs[lo:i + ctx + 1], "sim": ss[lo:i + ctx + 1],
                    "real_len": len(rs), "sim_len": len(ss)}
    return {"index": None, "real_len": len(rs), "sim_len": len(ss)}


def run_all_parity(real_agg: dict, sim_agg: dict,
                   real_trainers: dict, sim_trainers: dict,
                   agg_goal: int = 0, max_rounds: Optional[int] = None) -> dict:
    """Run the full parity + invariant battery; returns {name: result}."""
    results = {
        "selection": selection_parity(real_agg, sim_agg, max_rounds),
        "aggregation_sequence": aggregation_sequence_parity(real_agg, sim_agg, max_rounds),
        "staleness": staleness_parity(real_agg, sim_agg),
        "participation": participation_parity(real_agg, sim_agg),
        "sim_send_ts": sim_send_ts_ok(real_trainers, sim_trainers),
        "gpu_budget_real": gpu_budget_ok(real_trainers),
        "gpu_budget_sim": gpu_budget_ok(sim_trainers),
        "sim_commit_monotone": sim_commit_order_monotone(sim_agg),
    }
    if agg_goal:
        results["agg_goal_cycles_real"] = agg_goal_cycles_ok(real_agg, agg_goal)
        results["agg_goal_cycles_sim"] = agg_goal_cycles_ok(sim_agg, agg_goal)
    return results
