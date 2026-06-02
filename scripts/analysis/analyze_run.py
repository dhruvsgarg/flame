#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Post-run telemetry analyzer for FLAME experiments.

Reads the JSONL event streams written by :mod:`flame.telemetry` (one file per
process in a run's telemetry directory) and emits a bundle of PNG plots plus a
short text summary. Schema-driven, so it works identically across selectors /
aggregators -- enabling apples-to-apples comparison.

Usage
-----
    python analyze_run.py <telemetry_dir> [--out <plots_dir>]
    python analyze_run.py --compare <dir1> <dir2> ... [--labels a b ...]

``<telemetry_dir>`` contains ``aggregator_*.jsonl`` and ``trainer_*.jsonl``.
Default output is ``<telemetry_dir>/../plots``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_helpers as ph  # noqa: E402

# Event-type names (kept in sync with flame/telemetry/events.py). Imported from
# the package when available, else hardcoded so the analyzer also runs stand-alone.
try:
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "python"
        ),
    )
    from flame.telemetry.events import (  # noqa: E402
        EVENT_AGG_EVAL,
        EVENT_AGG_ROUND,
        EVENT_AVAIL_CHANGE,
        EVENT_SELECTION,
        EVENT_TRAINER_ROUND,
        EVENT_UTIL_DISPARITY,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    EVENT_SELECTION = "selection"
    EVENT_AGG_EVAL = "agg_eval"
    EVENT_AGG_ROUND = "agg_round"
    EVENT_TRAINER_ROUND = "trainer_round"
    EVENT_UTIL_DISPARITY = "util_disparity"
    EVENT_AVAIL_CHANGE = "avail_change"


def load_events(telemetry_dir: str) -> list[dict]:
    """Load all JSONL records from a telemetry directory."""
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(telemetry_dir, "*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a torn final line from a killed process
    return records


def by_event(records: list[dict], event: str) -> list[dict]:
    return [r for r in records if r.get("event") == event]


# --- individual plots -------------------------------------------------------


def plot_accuracy_loss(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_AGG_EVAL)
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("round", 0))
    rounds = [r.get("round") for r in rows]
    saved = []
    # accept either "test-accuracy"/"test-loss" or generic keys
    acc = [r.get("test-accuracy") for r in rows]
    loss = [r.get("test-loss") for r in rows]
    if any(a is not None for a in acc):
        rs = [r for r, a in zip(rounds, acc) if a is not None]
        av = [a for a in acc if a is not None]
        p = ph.line_plot(
            {"test-accuracy": (rs, av)}, "round", "accuracy",
            "Aggregator test accuracy over rounds", out_dir, "accuracy_over_rounds.png",
        )
        if p:
            saved.append(p)
    if any(x is not None for x in loss):
        rs = [r for r, x in zip(rounds, loss) if x is not None]
        lv = [x for x in loss if x is not None]
        p = ph.line_plot(
            {"test-loss": (rs, lv)}, "round", "loss",
            "Aggregator test loss over rounds", out_dir, "loss_over_rounds.png",
        )
        if p:
            saved.append(p)
    return saved


def plot_staleness(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_AGG_ROUND)
    if not rows:
        return []
    saved = []
    stale_vals = []
    for r in rows:
        for s in (r.get("staleness") or []):
            stale_vals.append(s)
    if stale_vals:
        p = ph.cdf_plot(
            stale_vals, "update staleness (rounds)", "Update staleness CDF",
            out_dir, "staleness_cdf.png",
        )
        if p:
            saved.append(p)
    # in-flight / queue timeline (by event arrival order)
    inflight = [(i, r.get("updates_in_queue")) for i, r in enumerate(rows)
                if r.get("updates_in_queue") is not None]
    if inflight:
        xs = [i for i, _ in inflight]
        ys = [v for _, v in inflight]
        p = ph.line_plot(
            {"updates_in_queue": (xs, ys)}, "aggregation step", "updates in queue",
            "Async queue depth over aggregation steps", out_dir, "queue_depth.png",
        )
        if p:
            saved.append(p)
    return saved


def plot_avail_composition(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_SELECTION)
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("round", 0))
    # one composition per round (last selection in that round wins)
    per_round: dict[int, dict] = {}
    for r in rows:
        comp = r.get("avail_composition")
        if comp:
            per_round[r.get("round", 0)] = comp
    if not per_round:
        return []
    rounds = sorted(per_round.keys())
    state_names = sorted({s for c in per_round.values() for s in c.keys()})
    series = {s: [per_round[rd].get(s, 0) for rd in rounds] for s in state_names}
    p = ph.stacked_area(
        rounds, series, "round", "trainer count",
        "Availability composition over rounds (selector view)",
        out_dir, "availability_composition.png",
    )
    return [p] if p else []


def plot_utility_speed_scatter(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_SELECTION)
    xs, ys, sel = [], [], []
    for r in rows:
        per = r.get("per_trainer") or {}
        for _, info in per.items():
            u = info.get("utility")
            s = info.get("speed_s")
            if u is None or s is None:
                continue
            xs.append(s)
            ys.append(u)
            sel.append(bool(info.get("selected")))
    if not xs:
        return []
    p = ph.scatter_plot(
        xs, ys, sel, "round duration / speed (s)", "statistical utility",
        "Utility vs speed: selected vs eligible", out_dir, "utility_vs_speed.png",
    )
    return [p] if p else []


def plot_selection_frequency(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_SELECTION)
    counter: Counter = Counter()
    for r in rows:
        for eid in (r.get("chosen") or []):
            counter[str(eid)] += 1
    if not counter:
        return []
    items = counter.most_common()
    cats = [k for k, _ in items]
    vals = [v for _, v in items]
    p = ph.bar_plot(
        cats, vals, "times selected", "Selection frequency per trainer (fairness)",
        out_dir, "selection_frequency.png",
    )
    return [p] if p else []


def plot_trainer_time_breakdown(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_TRAINER_ROUND)
    if not rows:
        return []
    agg: dict[str, dict[str, list]] = defaultdict(
        lambda: {"gpu": [], "sim": [], "wait": []}
    )
    for r in rows:
        tid = str(r.get("end_id", "?"))
        agg[tid]["gpu"].append(r.get("real_gpu_time_s") or 0.0)
        agg[tid]["sim"].append(r.get("sim_round_duration_s") or 0.0)
        agg[tid]["wait"].append(r.get("wait_time_s") or 0.0)
    cats = sorted(agg.keys())

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    segments = {
        "real_gpu_time_s": [mean(agg[c]["gpu"]) for c in cats],
        "sim_round_duration_s": [mean(agg[c]["sim"]) for c in cats],
        "wait_time_s": [mean(agg[c]["wait"]) for c in cats],
    }
    p = ph.stacked_bar(
        cats, segments, "mean seconds per round",
        "Trainer time breakdown: real GPU vs simulated delay vs wait",
        out_dir, "trainer_time_breakdown.png",
    )
    return [p] if p else []


def plot_util_disparity(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_UTIL_DISPARITY)
    if not rows:
        return []
    saved = []
    # ratio over elapsed time, one series per trainer
    by_trainer: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_trainer[str(r.get("end_id", "?"))].append(r)
    ratio_series = {}
    streamed_series = {}
    full_series = {}
    for tid, rs in by_trainer.items():
        rs.sort(key=lambda r: r.get("elapsed_s", 0))
        xs = [r.get("elapsed_s") for r in rs]
        ratio_series[tid] = (xs, [r.get("utility_ratio") for r in rs])
        streamed_series["%s streamed" % tid] = (xs, [r.get("utility_streamed") for r in rs])
        full_series["%s full" % tid] = (xs, [r.get("utility_full") for r in rs])
    p = ph.line_plot(
        ratio_series, "elapsed sim time (s)", "streamed / full utility ratio",
        "Streamed-vs-full utility ratio over time (1.0 = no disparity)",
        out_dir, "util_disparity_ratio.png",
    )
    if p:
        saved.append(p)
    # absolute streamed vs full (combine; can be busy with many trainers)
    combined = {}
    combined.update(streamed_series)
    combined.update(full_series)
    p = ph.line_plot(
        combined, "elapsed sim time (s)", "statistical utility",
        "Streamed-prefix vs full-dataset utility over time",
        out_dir, "util_disparity_absolute.png",
    )
    if p:
        saved.append(p)
    return saved


# --- streaming / mis-selection / comm-cost helpers --------------------------


def _visible_fraction(r: dict) -> Optional[float]:
    vs, ts = r.get("visible_samples"), r.get("total_samples")
    if vs is None or not ts:
        return None
    return vs / ts


def accuracy_by_round(records) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in by_event(records, EVENT_AGG_EVAL):
        a = r.get("test-accuracy")
        if a is not None:
            out[int(r.get("round", 0))] = a
    return out


def sim_time_by_round(records) -> dict[int, float]:
    """round -> max sim_completion_ts seen at/under that round (sim wall time)."""
    best: dict[int, float] = {}
    for r in by_event(records, EVENT_TRAINER_ROUND):
        rd = int(r.get("round", 0))
        sc = r.get("sim_completion_ts")
        if sc is None:
            sc = r.get("sim_round_duration_s")
        if sc is None:
            continue
        best[rd] = max(best.get(rd, 0.0), float(sc))
    # make monotonic cumulative
    out: dict[int, float] = {}
    run = 0.0
    for rd in sorted(best):
        run = max(run, best[rd])
        out[rd] = run
    return out


def cumulative_comm_by_round(records) -> tuple[list[int], list[float]]:
    """Cumulative communication in model-equivalents vs round.

    Per selected trainer: train = 2 (download model + upload delta),
    eval = 1 (download model; scalar utility upload ~= 0). Felix's eval-selector
    rows are charged automatically; OORT/REFL (no eval-distribute) have none.
    """
    per_round: dict[int, float] = defaultdict(float)
    for r in by_event(records, EVENT_SELECTION):
        rd = int(r.get("round", 0))
        n = len(r.get("chosen") or [])
        cost = 2.0 * n if r.get("task", "train") == "train" else 1.0 * n
        per_round[rd] += cost
    rounds = sorted(per_round)
    cum, run = [], 0.0
    for rd in rounds:
        run += per_round[rd]
        cum.append(run)
    return rounds, cum


def comm_vs_accuracy_series(records) -> tuple[list[float], list[float]]:
    """(cumulative model-equivalents, accuracy) aligned by round."""
    rounds, cum = cumulative_comm_by_round(records)
    cum_at = {}
    for rd, c in zip(rounds, cum):
        cum_at[rd] = c
    acc = accuracy_by_round(records)
    xs, ys = [], []
    running = 0.0
    for rd in sorted(acc):
        # comm accumulated up to this round
        ups = [c for r, c in cum_at.items() if r <= rd]
        running = max(ups) if ups else running
        xs.append(running)
        ys.append(acc[rd])
    return xs, ys


def time_to_target(records, target: float) -> dict:
    """First round/wall/sim time the test accuracy reaches ``target``."""
    evs = sorted(by_event(records, EVENT_AGG_EVAL), key=lambda r: r.get("round", 0))
    start_ts = min((r.get("ts") for r in records if r.get("ts")), default=None)
    sim_map = sim_time_by_round(records)
    for r in evs:
        a = r.get("test-accuracy")
        if a is not None and a >= target:
            rd = int(r.get("round", 0))
            wall = (r.get("ts") - start_ts) if (r.get("ts") and start_ts) else None
            return {"round": rd, "wall_s": wall, "sim_s": sim_map.get(rd),
                    "accuracy": a, "reached": True}
    return {"round": None, "wall_s": None, "sim_s": None,
            "accuracy": None, "reached": False}


def load_oracle_misselection(telemetry_dir: str) -> list[dict]:
    """Read <run>/analysis/oracle_misselection.csv (run = parent of telemetry)."""
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    path = os.path.join(run_dir, "analysis", "oracle_misselection.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def load_oracle_utility(telemetry_dir: str) -> list[dict]:
    """Read <run>/analysis/oracle_utility.csv (per round x candidate believed/true)."""
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    path = os.path.join(run_dir, "analysis", "oracle_utility.csv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _spearman(a, b):
    """Rank correlation (ties broken arbitrarily); None if degenerate."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def im_staleness_by_round(telemetry_dir: str):
    """Per-round I_m believed-vs-true staleness over the *candidate* pool.

    For each train-selection round, over candidates whose believed utility is
    known: rank-correlation(believed, true) (1.0 = believed ranks clients exactly
    like reality; lower = more mis-ranking) and normalized mean|believed-true|.
    This is the I_m staleness dimension; a selector that refreshes utility (Felix)
    should keep correlation high/flat as data unlocks.
    """
    rows = [r for r in load_oracle_utility(telemetry_dir) if r.get("task") == "train"]
    by_round = defaultdict(lambda: {"bel": [], "tru": []})
    for r in rows:
        b, t = r.get("believed"), r.get("true")
        try:
            b = float(b); t = float(t)
        except (TypeError, ValueError):
            continue
        by_round[int(float(r["round"]))]["bel"].append(b)
        by_round[int(float(r["round"]))]["tru"].append(t)
    rounds, corr, ngap = [], [], []
    for rd in sorted(by_round):
        bel = by_round[rd]["bel"]; tru = by_round[rd]["tru"]
        if len(bel) < 3:
            continue
        sc = _spearman(bel, tru)
        mt = sum(tru) / len(tru)
        g = sum(abs(b - t) for b, t in zip(bel, tru)) / len(bel)
        rounds.append(rd)
        corr.append(sc)
        ngap.append(g / mt if mt else None)
    return rounds, corr, ngap


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


# --- new single-run plots ---------------------------------------------------


def plot_data_unlock_effects(records, out_dir) -> list[str]:
    """delta_weight_l2 / final_loss / stat_utility vs visible_fraction."""
    rows = by_event(records, EVENT_TRAINER_ROUND)
    saved = []
    specs = [
        ("delta_weight_l2", "update L2 norm", "update_norm_vs_visible.png",
         "Update magnitude vs unlocked-data fraction"),
        ("final_loss", "final training loss", "loss_vs_visible.png",
         "Training loss vs unlocked-data fraction"),
        ("stat_utility", "statistical utility", "utility_vs_visible.png",
         "Statistical utility vs unlocked-data fraction"),
    ]
    for key, ylab, fname, title in specs:
        xs, ys = [], []
        for r in rows:
            vf = _visible_fraction(r)
            y = r.get(key)
            if vf is None or y is None:
                continue
            xs.append(vf)
            ys.append(y)
        if not xs:
            continue
        p = ph.scatter_plot(xs, ys, None, "visible fraction (unlocked data)",
                            ylab, title, out_dir, fname)
        if p:
            saved.append(p)
    return saved


def plot_comm_vs_accuracy(records, out_dir) -> list[str]:
    xs, ys = comm_vs_accuracy_series(records)
    if not xs:
        return []
    p = ph.line_plot(
        {"accuracy": (xs, ys)}, "cumulative comm (model-equivalents)", "accuracy",
        "Communication cost vs accuracy", out_dir, "comm_vs_accuracy.png",
    )
    return [p] if p else []


def plot_misselection(records, out_dir, telemetry_dir) -> list[str]:
    rows = [r for r in load_oracle_misselection(telemetry_dir)
            if r.get("task") == "train"]
    if not rows:
        return []
    rows.sort(key=lambda r: float(r.get("round", 0)))
    xs = _floats(rows, "round")
    series = {
        "misselection_rate (1 - top-k overlap)": (xs, _floats(rows, "misselection_rate")),
        "utility_regret (normalized units)": (xs, _floats(rows, "utility_regret")),
    }
    saved = []
    p = ph.line_plot(series, "round", "value",
                     "Mis-selection over time (from oracle)",
                     out_dir, "misselection_over_time.png")
    if p:
        saved.append(p)
    gap_series = {
        "believed - true (selected)": (xs, _floats(rows, "believed_minus_true_gap")),
    }
    p = ph.line_plot(gap_series, "round", "utility gap",
                     "Believed-minus-true utility gap (selected set)",
                     out_dir, "utility_belief_gap.png")
    if p:
        saved.append(p)
    return saved


def write_summary(records, out_dir, telemetry_dir) -> str:
    counts = Counter(r.get("event") for r in records)
    n_trainers = len({r.get("end_id") for r in records if r.get("role") == "trainer"})
    lines = [
        "Telemetry summary for: %s" % telemetry_dir,
        "total events: %d" % len(records),
        "trainers seen: %d" % n_trainers,
        "event counts:",
    ]
    for ev, c in counts.most_common():
        lines.append("  %-16s %d" % (ev, c))
    text = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as fh:
        fh.write(text)
    return path


def analyze(telemetry_dir: str, out_dir: Optional[str] = None) -> list[str]:
    records = load_events(telemetry_dir)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(telemetry_dir)), "plots")
    if not records:
        print("no telemetry events found in %s" % telemetry_dir)
        return []
    saved: list[str] = []
    plotters = [
        plot_accuracy_loss,
        plot_staleness,
        plot_avail_composition,
        plot_utility_speed_scatter,
        plot_selection_frequency,
        plot_trainer_time_breakdown,
        plot_util_disparity,
        plot_data_unlock_effects,
        plot_comm_vs_accuracy,
    ]
    for fn in plotters:
        try:
            saved.extend(fn(records, out_dir))
        except Exception as e:  # one bad plot must not stop the rest
            print("  (plot %s failed: %s)" % (fn.__name__, e))
    # mis-selection plots need the oracle CSV next to the telemetry dir
    try:
        saved.extend(plot_misselection(records, out_dir, telemetry_dir))
    except Exception as e:
        print("  (plot plot_misselection failed: %s)" % e)
    saved.append(write_summary(records, out_dir, telemetry_dir))
    print("wrote %d artifact(s) to %s" % (len(saved), out_dir))
    for p in saved:
        print("  %s" % p)
    return saved


def compare(dirs: list[str], labels: Optional[list[str]], out_dir: str) -> list[str]:
    """Overlay accuracy / staleness across runs (one per selector)."""
    if labels is None or len(labels) != len(dirs):
        labels = [os.path.basename(os.path.dirname(os.path.abspath(d))) or d for d in dirs]
    acc_series = {}
    stale_series_vals = {}
    for label, d in zip(labels, dirs):
        recs = load_events(d)
        ev = by_event(recs, EVENT_AGG_EVAL)
        ev.sort(key=lambda r: r.get("round", 0))
        rs = [r.get("round") for r in ev if r.get("test-accuracy") is not None]
        av = [r.get("test-accuracy") for r in ev if r.get("test-accuracy") is not None]
        if rs:
            acc_series[label] = (rs, av)
        stale = [s for r in by_event(recs, EVENT_AGG_ROUND) for s in (r.get("staleness") or [])]
        if stale:
            stale_series_vals[label] = stale
    saved = []
    p = ph.line_plot(
        acc_series, "round", "accuracy",
        "Accuracy comparison across runs", out_dir, "compare_accuracy.png",
    )
    if p:
        saved.append(p)
    # overlay staleness CDFs by plotting each as its own line
    if stale_series_vals:
        import numpy as np

        series = {}
        for label, vals in stale_series_vals.items():
            arr = np.sort(np.asarray(vals, dtype=float))
            y = np.arange(1, len(arr) + 1) / len(arr)
            series[label] = (arr, y)
        p = ph.line_plot(
            series, "staleness (rounds)", "CDF",
            "Staleness CDF comparison across runs", out_dir, "compare_staleness_cdf.png",
        )
        if p:
            saved.append(p)
    print("wrote %d comparison artifact(s) to %s" % (len(saved), out_dir))
    return saved


def compare_streaming(
    dirs: list[str], labels: Optional[list[str]], out_dir: str, target: float = 0.6
) -> list[str]:
    """Cross-baseline streaming comparison (felix vs oort vs refl, etc.).

    Produces: accuracy overlay, time-to-target table+bars, comm-vs-accuracy,
    mean true-utility of selected set, mis-selection over time, streamed/full
    utility ratio, and update-norm vs visible-fraction -- all overlaid by run.
    """
    if labels is None or len(labels) != len(dirs):
        labels = [
            os.path.basename(os.path.dirname(os.path.abspath(d))) or d for d in dirs
        ]
    os.makedirs(out_dir, exist_ok=True)
    saved: list[str] = []

    acc_series: dict = {}
    comm_acc_series: dict = {}
    true_util_series: dict = {}
    missel_series: dict = {}
    regret_series: dict = {}
    util_ratio_series: dict = {}
    delta_vis_series: dict = {}
    ttt_rows: list[dict] = []

    for label, d in zip(labels, dirs):
        recs = load_events(d)

        acc = accuracy_by_round(recs)
        if acc:
            rs = sorted(acc)
            acc_series[label] = (rs, [acc[r] for r in rs])

        xs, ys = comm_vs_accuracy_series(recs)
        if xs:
            comm_acc_series[label] = (xs, ys)

        ttt = time_to_target(recs, target)
        ttt_rows.append({"label": label, **ttt})

        # util disparity: mean streamed/full ratio per round
        ud = by_event(recs, EVENT_UTIL_DISPARITY)
        if ud:
            by_round: dict[int, list[float]] = defaultdict(list)
            for r in ud:
                v = r.get("utility_ratio")
                if v is not None:
                    by_round[int(r.get("round", 0))].append(v)
            rs = sorted(by_round)
            util_ratio_series[label] = (
                rs, [sum(by_round[r]) / len(by_round[r]) for r in rs]
            )

        # update-norm vs visible-fraction (binned mean over 20 bins)
        tr = by_event(recs, EVENT_TRAINER_ROUND)
        pts = [(_visible_fraction(r), r.get("delta_weight_l2")) for r in tr]
        pts = [(a, b) for a, b in pts if a is not None and b is not None]
        if pts:
            bins: dict[int, list[float]] = defaultdict(list)
            for vf, dn in pts:
                bins[min(19, int(vf * 20))].append(dn)
            bxs = sorted(bins)
            delta_vis_series[label] = (
                [(b + 0.5) / 20 for b in bxs],
                [sum(bins[b]) / len(bins[b]) for b in bxs],
            )

        # oracle-derived mis-selection metrics (train task)
        orows = [
            r for r in load_oracle_misselection(d) if r.get("task") == "train"
        ]
        if orows:
            orows.sort(key=lambda r: float(r.get("round", 0)))
            oxs = _floats(orows, "round")
            true_util_series[label] = (oxs, _floats(orows, "mean_true_selected"))
            missel_series[label] = (oxs, _floats(orows, "misselection_rate"))
            regret_series[label] = (oxs, _floats(orows, "utility_regret"))

    def _line(series, xl, yl, title, fname):
        p = ph.line_plot(series, xl, yl, title, out_dir, fname)
        if p:
            saved.append(p)

    _line(acc_series, "round", "accuracy",
          "Accuracy vs round (target=%.0f%%)" % (target * 100),
          "compare_accuracy.png")
    _line(comm_acc_series, "cumulative comm (model-equivalents)", "accuracy",
          "Communication cost vs accuracy", "compare_comm_vs_accuracy.png")
    _line(true_util_series, "round", "mean true utility of selected set",
          "Selected-set true utility (oracle)", "compare_true_utility_selected.png")
    _line(missel_series, "round", "mis-selection rate (1 - top-k overlap)",
          "Mis-selection rate over time (oracle)", "compare_misselection.png")
    _line(regret_series, "round", "utility regret",
          "Selection utility regret over time (oracle)", "compare_utility_regret.png")
    _line(util_ratio_series, "round", "streamed / full utility ratio",
          "Streamed-vs-full utility ratio (1.0 = no disparity)",
          "compare_util_disparity.png")
    _line(delta_vis_series, "visible fraction (unlocked data)", "mean update L2 norm",
          "Update magnitude vs unlocked data", "compare_delta_norm_vs_visible.png")

    # I_m staleness: how well each selector's BELIEVED utility tracks TRUE utility
    # over the candidate pool, as data unlocks (per-baseline, self-relative).
    corr_series, gap_series = {}, {}
    for label, d in zip(labels, dirs):
        rounds, corr, ngap = im_staleness_by_round(d)
        if rounds:
            corr_series[label] = (rounds, corr)
            gap_series[label] = (rounds, ngap)
    _line(corr_series, "round", "rank-corr(believed I_m, true I_m)",
          "I_m staleness: belief-vs-truth ranking over time (1.0=perfect)",
          "compare_Im_rankcorr.png")
    _line(gap_series, "round", "normalized |believed - true| I_m",
          "I_m staleness: belief-vs-truth magnitude over time",
          "compare_Im_gap.png")

    # time-to-target table + bars
    ttt_path = os.path.join(out_dir, "compare_time_to_target.csv")
    with open(ttt_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["label", "reached", "round", "wall_s", "sim_s", "accuracy"]
        )
        w.writeheader()
        for r in ttt_rows:
            w.writerow({k: r.get(k) for k in
                        ["label", "reached", "round", "wall_s", "sim_s", "accuracy"]})
    saved.append(ttt_path)
    cats = [r["label"] for r in ttt_rows if r.get("round") is not None]
    if cats:
        vals = [r["round"] for r in ttt_rows if r.get("round") is not None]
        p = ph.bar_plot(cats, vals, "rounds to target",
                        "Rounds to reach %.0f%% accuracy" % (target * 100),
                        out_dir, "compare_time_to_target_rounds.png")
        if p:
            saved.append(p)

    print("wrote %d streaming-comparison artifact(s) to %s" % (len(saved), out_dir))
    for p in saved:
        print("  %s" % p)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_dir", nargs="?", help="run telemetry directory")
    parser.add_argument("--out", help="output plots directory")
    parser.add_argument("--compare", nargs="+", help="telemetry dirs to compare")
    parser.add_argument("--compare-streaming", nargs="+",
                        help="telemetry dirs for the streaming mis-selection comparison")
    parser.add_argument("--labels", nargs="+", help="labels for --compare* dirs")
    parser.add_argument("--target", type=float, default=0.6,
                        help="target accuracy for time-to-target (default 0.6)")
    args = parser.parse_args()

    if args.compare_streaming:
        out = args.out or "compare_plots"
        compare_streaming(args.compare_streaming, args.labels, out, args.target)
        return
    if args.compare:
        out = args.out or "compare_plots"
        compare(args.compare, args.labels, out)
        return
    if not args.telemetry_dir:
        parser.error("provide a telemetry_dir or --compare dirs...")
    analyze(args.telemetry_dir, args.out)


if __name__ == "__main__":
    main()
