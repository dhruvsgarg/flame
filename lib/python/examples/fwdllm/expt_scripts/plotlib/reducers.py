# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unified telemetry reducer — the single source of the five experiments' metrics,
shared by plot_run / compare_baselines / make_paper_figs via load_run().

Streams the aggregator JSONL in one pass (can be multi-GB) and keeps evals as an
ORDERED series, not a data_id-keyed dict: data_id cycles per round (round 2 reuses
data_id 0), so keying by it silently overwrites earlier rounds and corrupts Δloss.
Field names per EXPERIMENTS.md §5.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# streaming JSONL iterator (bounded memory)
# --------------------------------------------------------------------------- #
def _iter_jsonl(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line
    except OSError:
        return


# --------------------------------------------------------------------------- #
# the per-run result — scalars + BOUNDED plot series (never the raw JSONL)
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    key: str
    run_dir: str
    # is this a sim-mode run? (aggregator_config.json hyperparameters.time_mode,
    # dirname-suffix fallback — see load_run). Drives which clock learning_curve()
    # plots on: sim runs report simulated vclock time, real runs wall-clock time,
    # since a real run's wall-clock IS its time-to-accuracy but a sim run's isn't
    # (it's inflated by however fast/slow the simulator itself executed).
    is_sim: bool = False
    # E1/E3 — ordered evals, emission order preserved. Each: dict(ts,round,data_id,
    # iter,acc,loss). Bounded by #bins * #rounds (~hundreds), safe to keep.
    evals: list = field(default_factory=list)
    t0: float | None = None
    # E2 — aggregator wall decomposition
    agg_compute_s: float = 0.0
    agg_barrier_s: float = 0.0
    agg_drain_s: float = 0.0
    agg_wall_s: float = 0.0
    # E2/E3 — trainer aggregates
    busy_frac: list = field(default_factory=list)
    trainer_gpu_s: float = 0.0
    fwd_total: int = 0
    pert_total: int = 0
    have_fwd: bool = False
    part_rounds: list = field(default_factory=list)
    part_bins: list = field(default_factory=list)
    part_iters: list = field(default_factory=list)
    n_trainers: int = 0
    # E4 — communication
    down_sizes: list = field(default_factory=list)
    up_sizes: list = field(default_factory=list)
    comm_by_kind: dict = field(default_factory=dict)
    have_comm: bool = False
    # E5 — sessions (async: dispatch->commit per contribution, validated).
    session_durs: list = field(default_factory=list)
    # E5 — sync "one round span" (BRIDGE_DESIGN.md N6): per-data_id engagement
    # duration (first agg_round cycle's ts -> commit ts for that data_id), the
    # sync-reselect-cadence analog of session_durs. Populated for BOTH sync and
    # async runs (harmless for async — compare_baselines.py's is_async switch
    # decides which list a baseline actually reports).
    round_span_durs: list = field(default_factory=list)
    # E2 — per-trainer fraction of its span spent waiting on mqtt_fetch_s (N5):
    # emitted but previously unused, so idle was only ever 1-busy_frac (one
    # undifferentiated bucket). Same denominator as busy_frac (ts_hi-ts_lo).
    net_wait_frac: list = field(default_factory=list)
    # E1 (sim) — (ts, vclock_now) from agg_round, for vclock-at-convergence
    vclock_track: list = field(default_factory=list)
    # E1 (N3) — converge_watch.py's own verdict for this run, matched via its
    # agg_telemetry field (converge.json lives in expt_scripts/smoke_logs/,
    # not co-located with run_dir). None if no --target-acc watcher ran.
    converge_json: dict | None = None
    # productive-learning cutoff (see load_run): telemetry beyond `cutoff_ts` is
    # IGNORED so a non-productive stalled tail can't skew the accumulated metrics.
    cutoff_ts: float | None = None      # None = no cutoff applied (full telemetry)
    plateau_ts: float | None = None     # ts of the last significant loss improvement

    # ---- derived views (pure over the fields above) ----------------------- #
    def learning_curve(self):
        """Ordered (hours, acc%, loss, round) arrays over the eval series.

        Hours are on each run's NATIVE clock: wall-clock for real runs (its
        wall-clock IS the time-to-accuracy claim); simulated vclock for sim runs
        (vclock_now, already zeroed near run start -- wall-clock on a sim run only
        measures how fast the simulator executed, not simulated time-to-accuracy).
        Falls back to wall-clock for a sim eval with no vclock_track entry yet
        (only possible before the first agg_round cycle)."""
        if not self.evals or self.t0 is None:
            return [], [], [], []
        if self.is_sim:
            hrs = []
            for e in self.evals:
                v = self.vclock_at(e["ts"])
                hrs.append((v if v is not None else (e["ts"] - self.t0)) / 3600)
        else:
            hrs = [(e["ts"] - self.t0) / 3600 for e in self.evals]
        acc = [e["acc"] * 100 if e["acc"] is not None else None for e in self.evals]
        loss = [e["loss"] for e in self.evals]
        rnd = [e["round"] for e in self.evals]
        return hrs, acc, loss, rnd

    def round_transition_indices(self):
        """Indices into the eval series where a new round begins (round increments
        or data_id wraps to a lower value) — the epoch boundaries to mark on E1."""
        idx = []
        for i in range(1, len(self.evals)):
            prev, cur = self.evals[i - 1], self.evals[i]
            r0, r1 = prev.get("round"), cur.get("round")
            if r0 is not None and r1 is not None and r1 > r0:
                idx.append(i)
            elif (cur.get("data_id") is not None and prev.get("data_id") is not None
                  and cur["data_id"] < prev["data_id"]):
                idx.append(i)
        return idx

    def max_accuracy(self):
        accs = [e["acc"] for e in self.evals if e["acc"] is not None]
        return max(accs) if accs else None

    def final_accuracy(self):
        accs = [e["acc"] for e in self.evals if e["acc"] is not None]
        return accs[-1] if accs else None

    def delta_loss(self):
        """first eval loss − last eval loss, in TIME order (not data_id order)."""
        losses = [e["loss"] for e in self.evals if e["loss"] is not None]
        return (losses[0] - losses[-1]) if len(losses) >= 1 else None

    def target_event(self, target: float, window: int):
        """The eval dict at which the trailing `window` consecutive evals (time
        order) first stay all >= target. None if never reached."""
        streak = 0
        for e in self.evals:
            if e["acc"] is None:
                continue
            streak = streak + 1 if e["acc"] >= target else 0
            if streak >= window:
                return e
        return None

    def time_to_target(self, target: float, window: int):
        """Wall-seconds to the convergence event (see target_event). None if never."""
        e = self.target_event(target, window)
        if e is None or self.t0 is None:
            return None
        return e["ts"] - self.t0

    def vclock_at(self, ts):
        """Last agg_round vclock_now at or before `ts` (sim runs). None if absent."""
        v = None
        for t, vc in self.vclock_track:
            if ts is not None and t is not None and t <= ts:
                v = vc
        return v

    def gpu_s_total(self):
        return self.trainer_gpu_s + self.agg_compute_s


# --------------------------------------------------------------------------- #
# streaming loaders (cutoff-aware — see load_run)
# --------------------------------------------------------------------------- #
def _read_aggregator(path: str) -> dict:
    """One streaming pass → the aggregator's BOUNDED raw structures (each carries a
    ts so a post-peak cutoff can be applied AFTER the peak time is known)."""
    evals, agg_rounds, comm_down, sessions, vclock = [], [], [], [], []
    t0 = None
    for e in _iter_jsonl(path):
        ev = e.get("event")
        ts = e.get("ts")
        if ts is not None:
            t0 = ts if t0 is None else min(t0, ts)
        if ev == "agg_eval":
            if e.get("data_id") is None:
                continue
            evals.append({
                "ts": ts, "round": e.get("round"),
                "data_id": int(e["data_id"]),
                "iter": e.get("iteration_per_data_id"),
                "acc": (float(e["test-accuracy"]) if e.get("test-accuracy") is not None else None),
                "loss": (float(e["test-loss"]) if e.get("test-loss") is not None else None),
            })
        elif ev == "agg_round":
            agg_rounds.append({
                "ts": ts,
                "compute": (e.get("aggregate_fedavg_s") or 0.0) + (e.get("eval_s") or 0.0),
                "barrier": e.get("barrier_wait_s") or 0.0,
                "drain": e.get("drain_tail_s") or 0.0,
                "wall": e.get("wall_elapsed_s"),
                # N6: data_id + is_async let the round-span reducer group
                # consecutive same-data_id cycles into one sync "session".
                "data_id": e.get("data_id"),
                "is_async": e.get("is_async"),
            })
            if e.get("vclock_now") is not None:
                vclock.append((ts, float(e["vclock_now"])))
            for iv in (e.get("contributor_intervals") or []):
                d, c = iv.get("dispatch_ts"), iv.get("commit_ts")
                if d is not None and c is not None and c >= d:
                    sessions.append((c, c - d))       # (commit_ts, duration)
        elif ev == "comm" and e.get("direction") == "agg_to_trainer":
            sz = e.get("size_bytes")
            if sz is not None:
                comm_down.append((ts, sz, e.get("payload_kind")))
    return {"evals": evals, "agg_rounds": agg_rounds, "comm_down": comm_down,
            "sessions": sessions, "vclock": vclock, "t0": t0}


def _compute_cutoff(evals, grace_s, loss_rel, mode="peak_acc"):
    """(cutoff_ts, anchor_ts): where the productive window ends.

    Two anchors, same "did it actually learn?" gate:
      * ``loss_plateau`` — the last eval where running-best test-loss dropped
        cumulatively >= ``loss_rel`` since its previous milestone. Loss is the
        grounded, un-quantized learning signal.
      * ``peak_acc`` (default) — the first eval attaining the run's global-max
        accuracy, so a post-peak accuracy drift or a round-boundary divergence is
        clipped off (stop each baseline near its highest accuracy).

    The loss scan is ALSO the learned-at-all detector: a run whose loss never
    dropped ``loss_rel`` (flat at chance) returns inf (no cutoff → shown in full
    so the flat non-learning line stays visible), regardless of mode.
    Deterministic, so grace_s defaults to 0; grace_s=None disables cutoff entirely.
    """
    if grace_s is None or not evals:
        return float("inf"), None
    # loss-improvement scan — doubles as the "did it learn?" gate
    rmin = milestone = plateau_ts = None
    advanced = False
    for e in evals:
        L, ts = e["loss"], e["ts"]
        if L is None or ts is None:
            continue
        if rmin is None:
            rmin = milestone = L
            plateau_ts = ts
            continue
        rmin = min(rmin, L)
        if rmin <= milestone * (1 - loss_rel):     # cumulative >= loss_rel drop
            milestone = rmin
            plateau_ts = ts
            advanced = True
    if plateau_ts is None or not advanced:
        return float("inf"), None                  # never learned → show full run
    if mode == "loss_plateau":
        return plateau_ts + grace_s, plateau_ts
    # peak_acc: first occurrence of the global-max accuracy (strict > keeps the
    # earliest peak, so a later noisy re-touch of the same max doesn't extend the tail)
    best = peak_ts = None
    for e in evals:
        a, ts = e["acc"], e["ts"]
        if a is None or ts is None:
            continue
        if best is None or a > best:
            best, peak_ts = a, ts
    if peak_ts is None:
        return plateau_ts + grace_s, plateau_ts    # no accuracy series → fall back
    return peak_ts + grace_s, peak_ts


def _compute_round_spans(agg_rounds):
    """N6: per-data_id engagement span (contiguous run of same-data_id agg_round
    cycles) — the SYNC "one round span" session definition, distinct from the
    async dispatch_to_commit one (see RunResult.round_span_durs). Only computed
    over is_async=False cycles (the concept doesn't apply to async's own
    per-contribution reselect cadence, already covered by session_durs)."""
    spans = []
    run_ts_lo = run_ts_hi = run_data_id = None
    for r in agg_rounds:
        if r.get("is_async"):
            continue
        ts, did = r["ts"], r.get("data_id")
        if ts is None:
            continue
        if run_data_id is None or did != run_data_id:
            if run_ts_lo is not None:
                spans.append(run_ts_hi - run_ts_lo)
            run_data_id, run_ts_lo, run_ts_hi = did, ts, ts
        else:
            run_ts_hi = ts
    if run_ts_lo is not None:
        spans.append(run_ts_hi - run_ts_lo)
    return spans


def _detect_is_sim(run_dir: str) -> bool:
    """sim vs. real, read from the run's own self-describing config (not the
    dirname, which is an operator-typed convention and per EXPERIMENTS.md §7.0
    has a known history of mislabeling -- e.g. alpha0p1 dirs that were actually
    alpha=1). aggregator_config.json's hyperparameters.time_mode is the ground
    truth ('simulated'/'real'); dirname's trailing _sim/_real is only a fallback
    for run dirs saved without that file."""
    cfg_path = os.path.join(run_dir, "aggregator_config.json")
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        tm = (cfg.get("hyperparameters") or {}).get("time_mode")
        if tm:
            return str(tm).lower().startswith("sim")
    except (OSError, json.JSONDecodeError):
        pass
    base = os.path.basename(run_dir.rstrip("/"))
    return base.endswith("_sim")


def _find_converge_json(run_dir: str, smoke_logs_dir: str | None = None) -> dict | None:
    """Locate this run's converge.json (written by converge_watch.py to
    expt_scripts/smoke_logs/<ts>/converge_<label>.json — NOT co-located with
    run_dir) by matching its `agg_telemetry` field against this run's own
    aggregator file, the one value converge.json emits that names the run
    unambiguously. None if no --target-acc watcher ran for this run (the
    common case — most launches are governed by --max-runtime-s/--max-data-id).
    `smoke_logs_dir` override is for tests; defaults to the real expt_scripts/
    smoke_logs next to this file."""
    agg_files = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not agg_files:
        return None
    agg_real = os.path.realpath(agg_files[0])
    smoke_logs = smoke_logs_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "smoke_logs")
    for cj_path in glob.glob(os.path.join(smoke_logs, "*", "converge_*.json")):
        try:
            with open(cj_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        at = payload.get("agg_telemetry")
        if at and os.path.realpath(at) == agg_real:
            return payload
    return None


def _read_trainers(tdir: str, cutoff: float, rr: RunResult):
    """Per-trainer aggregates, counting only trainer telemetry at ts <= cutoff."""
    for f in glob.glob(os.path.join(tdir, "trainer_*.jsonl")):
        rr.n_trainers += 1
        rounds_seen, bins_seen, n_iters = set(), set(), 0
        gpu = mqtt_wait = 0.0
        ts_lo = ts_hi = None
        fp_max = pt_max = 0
        saw_fp = False
        for e in _iter_jsonl(f):
            ev = e.get("event")
            ts = e.get("ts")
            if ts is not None and ts > cutoff:
                continue                                   # past the cutoff — ignore
            if ev == "trainer_round":
                gpu += e.get("gpu_compute_s") or 0.0
                mqtt_wait += e.get("mqtt_fetch_s") or 0.0
                n_iters += 1
                if e.get("round") is not None:
                    rounds_seen.add(e["round"])
                if e.get("data_id") is not None:
                    bins_seen.add(e["data_id"])
                if e.get("forward_passes_total") is not None:
                    fp_max = max(fp_max, e["forward_passes_total"]); saw_fp = True
                if e.get("perturbations_total") is not None:
                    pt_max = max(pt_max, e["perturbations_total"])
                if ts is not None:
                    ts_lo = ts if ts_lo is None else min(ts_lo, ts)
                    ts_hi = ts if ts_hi is None else max(ts_hi, ts)
            elif ev == "comm" and e.get("direction") == "trainer_to_agg":
                # trainer_to_agg (upload) is emitted TRAINER-side only.
                sz = e.get("size_bytes")
                if sz is not None:
                    rr.up_sizes.append(sz)
                    rr.have_comm = True
                    key = ("trainer_to_agg", e.get("payload_kind"))
                    rr.comm_by_kind[key] = rr.comm_by_kind.get(key, 0) + sz
        rr.trainer_gpu_s += gpu
        if saw_fp:
            rr.fwd_total += fp_max
            rr.pert_total += pt_max
            rr.have_fwd = True
        if n_iters:
            rr.part_rounds.append(len(rounds_seen))
            rr.part_bins.append(len(bins_seen))
            rr.part_iters.append(n_iters)
        if ts_lo is not None and ts_hi is not None and ts_hi > ts_lo:
            span = ts_hi - ts_lo
            rr.busy_frac.append(min(1.0, gpu / span))
            rr.net_wait_frac.append(min(1.0, mqtt_wait / span))


def load_run(run_dir: str, key: str | None = None,
             post_peak_grace_s: float | None = 0.0,
             loss_plateau_rel: float = 0.01,
             cutoff_mode: str = "peak_acc") -> RunResult | None:
    """Stream one run dir's telemetry into a RunResult (None if no aggregator file).
    Truncates the productive-learning tail (`cutoff_mode`: 'peak_acc' cuts at the
    highest-accuracy eval, 'loss_plateau' at the last significant loss improvement)
    so a post-peak drift/divergence can't skew the metrics; `post_peak_grace_s`=None
    disables it. The learned-at-all gate is loss-based either way. See _compute_cutoff."""
    run_dir = os.path.abspath(run_dir)
    tdir = os.path.join(run_dir, "telemetry")
    agg_files = sorted(glob.glob(os.path.join(tdir, "aggregator_*.jsonl")))
    if not agg_files:
        return None
    raw = _read_aggregator(agg_files[0])
    cutoff, plateau_ts = _compute_cutoff(raw["evals"], post_peak_grace_s,
                                         loss_plateau_rel, mode=cutoff_mode)

    rr = RunResult(key=key or os.path.basename(run_dir), run_dir=run_dir)
    rr.is_sim = _detect_is_sim(run_dir)
    rr.t0 = raw["t0"]
    rr.cutoff_ts = None if cutoff == float("inf") else cutoff
    rr.plateau_ts = plateau_ts

    within = lambda t: t is not None and t <= cutoff
    rr.evals = [e for e in raw["evals"] if within(e["ts"])]
    used_rounds = [r for r in raw["agg_rounds"] if within(r["ts"])]
    rr.agg_compute_s = sum(r["compute"] for r in used_rounds)
    rr.agg_barrier_s = sum(r["barrier"] for r in used_rounds)
    rr.agg_drain_s = sum(r["drain"] for r in used_rounds)
    walls = [r["wall"] for r in used_rounds if r["wall"] is not None]
    if walls:
        rr.agg_wall_s = max(walls)
    elif rr.evals and rr.t0 is not None:
        rr.agg_wall_s = max(0.0, rr.evals[-1]["ts"] - rr.t0)
    rr.vclock_track = [(t, v) for (t, v) in raw["vclock"] if within(t)]
    rr.round_span_durs = _compute_round_spans(used_rounds)
    rr.converge_json = _find_converge_json(run_dir)
    for (t, sz, kind) in raw["comm_down"]:
        if within(t):
            rr.down_sizes.append(sz)
            key_k = ("agg_to_trainer", kind)
            rr.comm_by_kind[key_k] = rr.comm_by_kind.get(key_k, 0) + sz
    rr.have_comm = bool(rr.down_sizes)
    rr.session_durs = [d for (c, d) in raw["sessions"] if within(c)]

    _read_trainers(tdir, cutoff, rr)
    return rr
