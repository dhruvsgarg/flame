# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared per-trainer availability-state series builder (Stage C.6.2).

Single resolver, imported by both checks.py (Aa / A4dur / observation_lag rungs)
and analyze_run.py (the four plots re-pointed in C.6.4), so the forward-fill
semantics never diverge between the checker and the plotter (Challenge 12
discipline — one function, not a duplicated copy in each consumer).

Reads `per_trainer[end_id]["avl_state"]` on each `selection` event (C.6.1).
Time-base: sim uses `vclock_now` (stamped on the event by C.6.1); real uses
`ts - t0` where t0 = the run's first selection event ts — the same wall-elapsed
approximation A3/K8 already use elsewhere in this checker.
"""

from __future__ import annotations

from typing import Optional


def _event_time(e: dict, mode: str, t0: float) -> Optional[float]:
    if mode == "sim":
        return e.get("vclock_now")
    ts = e.get("ts")
    return None if ts is None else ts - t0


def build_trainer_state_series(
    selection_events: list, mode: str
) -> dict[str, list]:
    """{end_id: [(t, avl_state), ...]} forward-fill series, sorted by t.

    mode: "sim" (t = vclock_now) or "real" (t = ts - t0). One sample per
    end_id per selection event it appears as a candidate in (whether or not
    selected) — `avail_composition`/`per_trainer` already cover every
    candidate in the pool, not just the chosen subset. Consecutive samples at
    the same t collapse to the last write (same-instant events, e.g. carried
    pacer state).
    """
    real_ts = [e.get("ts") for e in selection_events if e.get("ts") is not None]
    t0 = min(real_ts) if (mode == "real" and real_ts) else 0.0

    series: dict[str, list] = {}
    for e in sorted(
        selection_events, key=lambda x: (x.get("round", 0), x.get("ts", 0.0))
    ):
        t = _event_time(e, mode, t0)
        if t is None:
            continue
        for end_id, cand in (e.get("per_trainer") or {}).items():
            state = cand.get("avl_state")
            if state is None or state == "UNKNOWN":
                continue
            pts = series.setdefault(end_id, [])
            if pts and pts[-1][0] == t:
                pts[-1] = (t, state)
            else:
                pts.append((t, state))
    return series


def run_span(series: dict) -> float:
    """Max observed t across all trainers — the run's own time horizon."""
    return max((pts[-1][0] for pts in series.values() if pts), default=0.0)


def state_fractions(series: dict, t_end: Optional[float] = None) -> dict:
    """Per-trainer {state: fraction_of_span} from a forward-filled series.

    Dwell-time integration: each sample's state holds until the next sample
    (or `t_end` for the trailing segment, default = the trainer's own last
    sample — i.e. no tail credited beyond its last observation). Trainers
    with < 2 samples are omitted (no observed dwell to integrate). Per-trainer
    vectors sum to 1.
    """
    out: dict = {}
    for end_id, pts in series.items():
        if len(pts) < 2:
            continue
        durations: dict = {}
        for (t_a, s_a), (t_b, _) in zip(pts, pts[1:]):
            durations[s_a] = durations.get(s_a, 0.0) + max(0.0, t_b - t_a)
        last_t, last_s = pts[-1]
        end_t = t_end if t_end is not None else last_t
        if end_t > last_t:
            durations[last_s] = durations.get(last_s, 0.0) + (end_t - last_t)
        total = sum(durations.values())
        if total <= 0:
            continue
        out[end_id] = {s: d / total for s, d in durations.items()}
    return out


def total_variation_distance(a: dict, b: dict) -> float:
    """0.5 * Σ_s |a_s - b_s| over the union of states; 0 if identical."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
