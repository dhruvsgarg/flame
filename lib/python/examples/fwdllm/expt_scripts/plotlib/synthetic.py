# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Fake RunResults for layout previews and tests (`make_paper_figs.py --synthetic`).

Shapes only: a faster arm reaches the plateau sooner, pools fewer uploads per commit
and engages clients for shorter spans. Never a source of paper numbers.
"""

from __future__ import annotations

import numpy as np

from .reducers import RunResult


def fake_run(key, speed=1.0, seed=0, n_clients=100, hours=8.0, peak=0.88,
             is_async=True, uploads_per_commit=50, session_s=120.0, with_audit=True):
    rng = np.random.default_rng(abs(hash((key, seed))) % (2 ** 32))
    rr = RunResult(key=key, run_dir=f"<synthetic:{key}:{seed}>")
    rr.t0, rr.is_sim, rr.is_async, rr.n_trainers = 0.0, False, is_async, n_clients
    n_ev = 160
    ts = np.linspace(60.0, hours * 3600.0, n_ev)
    acc = peak - (peak - 0.25) * np.exp(-speed * ts / 3600.0 / 1.6)
    acc = np.clip(acc + rng.normal(0, 0.006, n_ev), 0, 1)
    loss = 1.4 * (1 - acc) + 0.2
    rr.evals = [{"ts": float(t), "round": 0, "data_id": i, "iter": 0,
                 "acc": float(a), "loss": float(l)}
                for i, (t, a, l) in enumerate(zip(ts, acc, loss))]
    n_commits = int(120 * speed * hours / 8.0) + 20
    rr.commit_ts = list(np.linspace(30.0, hours * 3600.0, n_commits))
    rr.commit_uploads = [max(1, int(rng.normal(uploads_per_commit, uploads_per_commit * 0.2)))
                         for _ in rr.commit_ts]
    size = 1.8e6
    for t, n in zip(rr.commit_ts, rr.commit_uploads):
        ends = rng.choice(n_clients, size=n, replace=True,
                          p=_skew(n_clients, 0.3 if is_async else 1.2))
        rr.contribs.extend((float(t), f"t{e}") for e in ends)
        rr.up_events.extend((float(t), size) for _ in range(n))
        rr.down_events.extend((float(t), size) for _ in range(n if is_async else max(1, n // 5)))
    rr.up_sizes = [s for _, s in rr.up_events]
    rr.down_sizes = [s for _, s in rr.down_events]
    rr.have_comm = True
    durs = rng.lognormal(np.log(session_s), 0.4, 2000)
    rr.session_durs = list(durs)
    rr.round_span_durs = list(durs)
    if with_audit:
        rho0 = 0.07
        phi = np.sqrt(1 + np.arange(1, n_commits + 1) * rho0 ** 2)
        rr.server_updates = [{"ts": float(t), "rho": float(rho0 / p), "phi": float(p)}
                             for t, p in zip(rr.commit_ts, phi)]
    return rr


def _skew(n, strength):
    w = np.exp(-strength * np.linspace(0, 3, n))
    return w / w.sum()


def fake_arm_runs(key, idx, n_arms, n_seeds=3, n_clients=100, champion="fluxtune"):
    """Seeds for one arm; later arms (and the champion) are faster."""
    fast = key == champion
    speed = 2.5 if fast else 0.6 + 1.6 * idx / max(1, n_arms - 1)
    sync = key.startswith("fwdllm")
    return [fake_run(key, speed=speed * (1 + 0.08 * (s - 1)), seed=s, n_clients=n_clients,
                     is_async=not sync, uploads_per_commit=50 if fast else 150,
                     session_s=90.0 if fast else 900.0, with_audit=not sync)
            for s in range(n_seeds)]
