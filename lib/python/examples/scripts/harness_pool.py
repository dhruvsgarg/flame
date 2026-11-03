#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N22 harness pool: run harness legs in parallel, isolated slots on one node (any example, any dataset).

A slot = disjoint physical cores (taskset) + a private mosquitto + a run tag (harness_suite.sh --isolate)
[+ exclusive GPUs]. A real+sim pair runs as two parallel legs plus a grade job. Jobs are packed
longest-first with backfill; the degree of parallelism follows from free cores, memory, GPUs and
--max-parallel. Durations are learned per job key (experiments/_harness_durations.json).

  harness_pool.py --tier T1 --changed origin/main     # sim-only delta smoke (EV only), affected baselines
  harness_pool.py --tier T2 --baselines 'felix fedbuff'  # sim-only matrix, parity vs banked real legs
  harness_pool.py --tier T3 --changed HEAD            # real+sim pairs; refreshes the real bank
  harness_pool.py --tier T4 [--phases 'P1 P2']        # the campaign phases
  ... --shard 1/2    this node's half (by estimated time)     ... --dry-run   print the plan only
Output: experiments/pool_<ts>/{SUMMARY.txt, pool.log, <phase>/summary.tsv, <phase>/<job>/...}.
Fail fast (FX-N40): a fatal line in any leg's logs stops the pool within ~30s -> ABORT.txt (--no-fail-fast).
Stall rules (FX-D45): a leg with no new committed round for --stall-min (15) min, or no log growth for 10 min, is killed alone -> STALLED.txt.
Early exit: every --doom-min (5) min each live leg runs the prefix-safe EV checks (event_invariants.EARLY); a FAIL kills the
leg and its pair partner -> DOOMED.txt (the cell is red whatever the rest of the run does).
"""

import argparse
import fcntl
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "async_cifar10" / "scripts" / "parity"))
from fail_fast import EXIT_FATAL, Scanner, StallWatch, leg_run_dirs, write_abort, write_stalled  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES = SCRIPT_DIR.parent                 # lib/python/examples
REPO = EXAMPLES.parents[2]                   # the git top level
OUT_DIR = EXAMPLES / "experiments"           # pool roots + per-node state (bank, durations)
DURATIONS = OUT_DIR / "_harness_durations.json"
# dataset -> the launcher example that runs it (one shared backprop-FL example today)
EXAMPLE_OF = {"cifar10": "async_cifar10", "google_speech": "async_cifar10"}


def example_dir(dataset: str) -> Path:
    return EXAMPLES / EXAMPLE_OF[dataset]

B6 = ("felix", "fedbuff", "oort", "oort_star", "refl", "feddance")
DATASETS = ("cifar10", "google_speech")
DS_TAG = {"cifar10": "", "google_speech": "gs_"}  # phase-id prefix; cifar keeps the campaign names
# CPU test shapes (FX-N22): small cohorts with varied c / aggGoal (= sync K via the launcher's agg_goal). A shape is one bank key, so T2 sim
# legs grade against the T3/T4 real legs of the same shape.
SHAPES = {
    "syn_0": dict(trace="syn_0", n=12, agg_goal=3, c=5, runtime_s=180, trace_scale=""),
    "syn_0b": dict(trace="syn_0", n=15, agg_goal=2, c=8, runtime_s=180, trace_scale=""),
    # FX-N67: aggGoal 10 over-selects 13, so stragglers cross rounds as at n=300
    "syn_0s": dict(trace="syn_0", n=30, agg_goal=10, c=13, runtime_s=300, trace_scale=""),
    "syn_20_3sts": dict(trace="syn_20_3st", n=40, agg_goal=10, c=13, runtime_s=300, trace_scale="4"),
    "syn_20_3st": dict(trace="syn_20_3st", n=15, agg_goal=3, c=6, runtime_s=240, trace_scale="4"),
    # 1200s x scale 4 = 4800s of trace: ~2 outage/up cycles per trainer (means ~1200s of trace each; FX-D18)
    "syn_50_3st": dict(trace="syn_50_3st", n=15, agg_goal=3, c=6, runtime_s=1200, trace_scale="4"),
    # FX-L34: ~10% AVL_TRAIN from t=0 (FX-D18), so n=45 keeps ~4-5 trainable for aggGoal 2 / sync select 3
    "mobiperf_3st": dict(trace="mobiperf_3st", n=45, agg_goal=2, c=4, runtime_s=240, trace_scale="4"),
    # FX-D106 Oort-family K=10 shapes; n by FX-L34 (select 11 / available share x 1.5)
    "syn_50_3sts": dict(trace="syn_50_3st", n=40, agg_goal=10, c=13, runtime_s=1200, trace_scale="4"),
    "mobiperf_3sts": dict(trace="mobiperf_3st", n=165, agg_goal=10, c=13, runtime_s=240, trace_scale="4", cpt=0.4),
}
# FX-D106: upstream exploitLen = int(K x (1 - 0.9)) is 0 below K=9, so Oort screens run at K=10.
OORT_FAMILY = ("oort", "oort_star")
OORT_SHAPE = {"syn_0": "syn_0s", "syn_0b": None, "syn_20_3st": "syn_20_3sts", "syn_50_3st": "syn_50_3sts", "mobiperf_3st": "mobiperf_3sts"}
# CPU legs that need longer for EV1's >= 4 commits (unaware oort waits out 90s timeouts; run 4: 0 in 240s).
MIN_RUNTIME_S = {("oort", "mobiperf_3st"): 3000,  # FX-D165: AVL_EVAL is UN_AVL; PR36 960 s gave 1 commit, G0U sim's 4th at 2508 s
                 ("oort", "syn_20_3st", "google_speech"): 600}  # FX-N85 d: 300 s gave 10 matched rounds (PR31 C2)
# Longer under --scale-smoke (value: harness, None = all): at 240 s unaware oort got 1 round, GPU fedbuff 6 (PR36-short).
SHORT_KEEP = {("oort", "syn_50_3st"): None, ("fedbuff", "syn_50_3st"): "none"}
SHORT_KEEP_S = 3600  # their smoke length (FX-N125)
# FX-D52: G0U legs whose unaware picks hold slots a full timeout run longer, to reach a gradeable round count.
G0U_RUNTIME_X = {("oort", "mobiperf_3st"): 4, ("fedbuff", "mobiperf_3st", "google_speech"): 3}
GPU_N = {"cifar10": 300, "google_speech": 100}  # the datasets' reference cohorts (datasets.yaml)
EUROSYS_TARGET = {"cifar10": 0.7, "google_speech": 0.7}  # EuroSys'26 runs: 70% on both (operator 10-12)
EUROSYS_HORIZON_S = 72000  # 20 h virtual: the paper's slowest MobiPerf TTA is 16.3 h
EUROSYS_K = 10  # May-2025 runs: aggr_num 10 (sync), aggGoal 10 (async)
# May-2025 shared trainer (trainer/config_dir0.1_num300_* @ 966797558): 1 epoch, bs 32, lr 0.04, plain SGD, no decay.
EUROSYS_TRAINER_HP = ("localSteps=null epochs=1 batchSize=32 learningRate=0.04 trainerOptimizer=sgd trainerMomentum=0 "
                      "trainerWeightDecay=0 lrDecayEnabled=false")
# FedDance paper Table 1 + FedScale defaults (baseline_reference.yaml feddance), applied to every F3 arm.
FEDDANCE_PAPER_HP = {
    ds: f"localSteps=1 epochs=1 batchSize={bs} learningRate={lr} trainerOptimizer=sgd trainerMomentum=0.9 "
        "trainerWeightDecay=0.0005 lrDecayEnabled=true lrDecayFactor=0.98 lrDecayEpoch=10 minLearningRate=0.00005"
    for ds, bs, lr in (("cifar10", 24, 0.1), ("google_speech", 32, 0.075))}
# G0 screen: reference c/n and aggGoal/c ratios at a smaller n; GPUs per leg keep the reference trainers per GPU.
G0_SHAPE = {"cifar10": dict(n=100, agg_goal=3, c=10, gpus=3),
            "google_speech": dict(n=50, agg_goal=5, c=15, gpus=4)}
# FX-N9 screen (C11): n=50 keeps cifar on 1 GPU, speech at 25/GPU.
G0U_SHAPE = {"cifar10": dict(n=50, agg_goal=3, c=5, gpus=1),
             "google_speech": dict(n=50, agg_goal=5, c=15, gpus=2)}
G0U_MOBIPERF = dict(agg_goal=2, c=4)  # FX-L34: ~10% AVL_TRAIN of n=50 leaves ~5 trainable
# FX-D106 Oort-family G0U; speech mobiperf at n=165 needs 7 GPUs a leg, so T3 mobiperf_3sts covers it.
G0U_OORT = {("cifar10", "syn_50_3st"): dict(n=50, agg_goal=10, c=13, gpus=1),
            ("cifar10", "mobiperf_3st"): dict(n=165, agg_goal=10, c=13, gpus=3),
            ("google_speech", "syn_50_3st"): dict(n=50, agg_goal=10, c=13, gpus=2)}
# PR37 (operator 10-14): one n:c:K per dataset for all six, cohort = the reference split's first n trainers (same shard, compute,
# D). K=10 keeps Oort's exploitLen >= 1 (FX-L63), sync K/n >= 5% (L34); c=15 > K keeps async overlap; <= 75 / 25 per A40.
G3_SHAPE = {"cifar10": dict(n=75, agg_goal=10, c=15, gpus=1), "google_speech": dict(n=50, agg_goal=10, c=15, gpus=2, sim_gpus=4)}
G3_RUNTIME_S = 7200  # 2 h legs (operator 10-15; R6: convergence sign-off 2-4 h)
G3_EVAL = "evalEverySeconds=auto"  # FX-N112: test eval every leg / EVAL_POINTS of the leg's clock, same for all six
EVAL_POINTS = 24
G3S_REPS = 3  # FX-N103: sim replicates per G3 cell, graded against its real-path-identical reals (parity_ladder)
F42_SEEDS = (1,)  # --seeds on F42: replicates for arms still in contention (R25)
_STATS = "sel.dynamic_kc.stats=true"  # I5 telemetry only: B and stale-vs-fresh cos per cycle
_PRE = "utilityPrefixSamples=64"


def usage_hp(trace: str) -> str:
    """Q-STREAM: data accrues while the trainer is off AVL_TRAIN on the leg's own trace (syn_0: linear)."""
    return ('data_streaming={"enabled":"True","mode":"usage","initial_frac":0.0,"full_data_available_after_s":1,'
            '"trace":"%s"}' % trace)


_C3 = "sel.utilityPerSecond=true"
_BAL, _E0 = "sel.participationDecay=1.0", "sel.evalGoalFactor=0"
# New Felix runs without eval tasks (C6: no gain); ctl = old Felix (eval on) is their control, e0 = ctl without eval is each arm's base.
# F42 (operator 10-21): R27 baselines + their oracle `*` variants, Felix old and the l3b ladder (C4, C5, pacing dropped: FELIX).
F42_RUNTIME_S = {"cifar10": 172800, "google_speech": 129600}
F42_CELLS = {"s0u": "syn_0", "s50u": "syn_50_3st"}
F42_EVAL_S = 600  # test eval every 10 min of the leg's clock: TTA resolution (30 min quantized F42S01)
_L3, _L3B = f"{_E0} {_C3}", f"{_E0} {_C3} {_BAL}"
F42_ARMS = {"ctl": dict(bls=("felix", "fedbuff", "refl", "feddance", "fedavg", "oort", "oort_star")),  # Felix old; baselines (R27)
            "e0": dict(hp=f"{_E0} {_STATS}"),
            "l3": dict(hp=f"{_L3} {_STATS}", thp=_PRE), "l3b": dict(hp=f"{_L3B} {_STATS}", thp=_PRE),
            "rstar": dict(bls=("refl",), hp="sel.avail_predictor=oracle sel.avail_priority=1 sel.holdoff_rounds=0 avail_knowledge=oracle"),
            "fdstar": dict(bls=("feddance",), hp="sel.checkin_source=selection avail_knowledge=oracle"),
            "fbstar": dict(bls=("fedbuff",), hp="avail_knowledge=oracle"),  # `*` = the trace's online set at selection
            # K1 ablation: Felix new learning availability by check-in (no notify, no eviction)
            "fck": dict(hp=f"{_L3B} {_STATS} serveEma=sweep avail_knowledge=checkin proactive_inflight_evict=False", thp=_PRE),
            # FX-N128 serve-EMA: the served (and graded) model averages ~one population sweep of versions (b = 1 - K / n)
            "ctl_srv": dict(bls=("felix", "fedbuff", "refl", "feddance", "fedavg", "oort", "oort_star"), hp="serveEma=sweep"),
            "l3b_srv": dict(hp=f"{_L3B} {_STATS} serveEma=sweep", thp=_PRE)}
F42_ARMS.update({f"{a}_srv": dict(F42_ARMS[a], hp=F42_ARMS[a]["hp"] + " serveEma=sweep") for a in ("rstar", "fdstar", "fbstar")})  # FX-N133
# FX-N132 (c) screen: does pick concentration cap cifar's late plateau? l3b with a 3x stronger balance exponent
F42_ARMS["l3b_b3_srv"] = dict(hp=f"{_E0} {_C3} sel.participationDecay=3.0 {_STATS} serveEma=sweep", thp=_PRE)
F42_ARMS["l3b_pb_srv"] = dict(hp=f"{_L3B} {_STATS} serveEma=sweep", thp="utilityPrefixSamples=-1")  # FX-N132 (f): prefix = one batch
F42_ARMS["rnd_srv"] = dict(hp=f"{_E0} sel.selectType=random serveEma=sweep")  # FX-N132 (c1): Felix with uniform picks
F42_ARMS["rnd_ro_srv"] = dict(hp=f"{_E0} sel.selectType=random opt.agg_rate_conf.type=old serveEma=sweep")
F42_ARMS["rnd_ron_srv"] = dict(hp=f"{_E0} sel.selectType=random opt.agg_rate_conf.type=old serveEma=sweep",
                               thp="lr_batch_normalize=true")  # + FedBuff's last-batch lr normalization
F42_ARMS["l3bn_srv"] = dict(hp=f"{_L3B} {_STATS} serveEma=sweep", thp=f"{_PRE} lr_batch_normalize=true")  # FX-N132 (c1): l3b + lr norm
F42_ARMS.update({f"l3b_e{e}_srv": dict(hp=f"{_L3B} {_STATS} serveEma=sweep", thp=f"{_PRE} epochs={e}") for e in (2, 3, 4)})  # FX-N135
F43_ARMS = ("ctl_srv", "rstar_srv", "fdstar_srv", "fbstar_srv", "l3b_srv", "l3b_e2_srv", "l3b_e3_srv")  # F43: baselines + Felix contenders
F43_SHAPE = {"cifar10": dict(n=300, c=60, agg_goal=15), "google_speech": dict(n=100, c=30, agg_goal=10, gpus=4, sim_gpus=4)}  # 3 GPUs OOM (33 trainers x 1.2 GB + agg)
F43_RUNTIME_S = {"cifar10": 432000, "google_speech": 86400}  # to the trace end (cifar ceiling 75% at 120 h); speech ceiling 73-79% at 24 h
F43_STOP = "targetAccuracy=0.7 targetWindow=5"  # operator 10-23: stop at the 70% goal on the mean of the last 5 evals
# F44 = loop FL1 (FX-N141): Felix built up from FedBuff, 12 h screens; static full-data cifar races to 70%, streams track the ceiling.
F44_CELLS = {"cifar10": {"s0": ("syn_0", False), "s0u": ("syn_0", True), "s50u": ("syn_50_3st", True)},
             "google_speech": {"s0u": ("syn_0", True)}}
F44_RUNTIME_S = 43200
F44_SHAPE = {"cifar10": F43_SHAPE["cifar10"], "google_speech": {}}  # speech screens at G3's n 50 (n 100 confirms)
_B1 = f"{_E0} sel.selectType=random opt.agg_rate_conf.type=old serveEma=sweep"  # S1: FedBuff on Felix's stack (notify / evict)
_B1T = "lr_batch_normalize=true"  # FedBuff's §5 sub-batch lr normalization
_GUARD, _WD = "opt.norm_clip=3", "trainerWeightDecay=0.0005"  # S2: FX-D232 guard; FedScale clients' wd (baseline_reference)
F44_ARMS = {"ctl_srv": dict(bls=("felix", "fedbuff")), "l3b_srv": {},  # no FedBuff* (F43: within 0.4 pp of FedBuff)
            "b1": dict(hp=_B1, thp=_B1T),  # S2 decided as a group: no single-knob b1w / b1g unless b1gw is ambiguous
            "b1gw": dict(hp=f"{_B1} {_GUARD}", thp=f"{_B1T} {_WD}"),
            **{f"b1gw_e{e}": dict(hp=f"{_B1} {_GUARD}", thp=f"{_B1T} {_WD} epochs={e}") for e in (2, 3)}}
F44_SPEECH_ARMS = ("ctl_srv", "l3b_srv", "b1", "b1gw", "b1gw_e3")
# F45 = FL1 S3 on b1: `prop` selection, score terms one at a time, both utility q's, Oort sampling; FedBuff anchor (Felix old / new
# from F44). `b1_c33` closes S1: b1 at FedBuff's effective s50u concurrency (60 × 116k / 213k updates).
_S3 = _B1.replace("sel.selectType=random", "sel.selectType=prop")
_S3U, _PB = f"{_S3} {_C3}", f"{_B1T} utilityPrefixSamples=-1"
F45_ARMS = {"ctl_srv": dict(bls=("fedbuff",)),
            "s3v": dict(hp=f"{_S3U} sel.scoreUtility=false", thp=_B1T),  # speed per second
            "s3u": dict(hp=_S3, thp=_PB),  # utility |D| q alone (q = pre-step loss, one batch: Felix new)
            "s3uo": dict(hp=_S3, thp=_B1T),  # utility alone, q = EMA loss over the pass (Oort / Felix old)
            "s3vu": dict(hp=_S3U, thp=_PB),  # |D| q per second
            "s3vuo": dict(hp=_S3U, thp=_B1T),  # |D| q per second, Oort's q
            "s3vub": dict(hp=f"{_S3U} {_BAL}", thp=_PB),  # + balance
            "s3vubx": dict(hp=f"{_S3U} {_BAL} sel.scoreUcb=true", thp=_PB),  # + UCB
            "s3vubx_oc": dict(hp=f"{_B1.replace('sel.selectType=random', 'sel.selectType=default')} {_C3} {_BAL}", thp=_PB),
            "b1_c33": dict(hp=_B1, thp=_B1T, c=33, cells=("s50u",))}
# F46 = FL1 S4 on both S3 carries (uo = s3uo, vub = s3vub) at FedBuff's own K 10 (paper §6.1) and c 60 (= its cifar c / n 20%):
# rate at fixed K / c; c at fixed rate (availability / loss phase, in [K, c]); both; a K x c probe (c / K sets staleness, K the
# averaging). `o` = the carry at rate old (F45 ran K 15). Telemetry `dynamic_kc` per commit (c, phi, step_scale).
F46_SHAPE = dict(F43_SHAPE["cifar10"], agg_goal=10, gpus=None)  # FX-D52: n=300 takes every GPU; None overrides G3's gpus=1
_RNEW = "opt.agg_rate_conf.type=new"
_DKCF = "sel.dynamic_kc.enabled=true sel.dynamic_kc.policy=felix_c sel.dynamic_kc.c_min=10"
_S4V = {"o": {}, "n": dict(hp=_RNEW), "nl": dict(hp=_RNEW, lrn=False),  # Felix's rate, with / without lr batch-normalize
        "p1": dict(hp=f"{_RNEW} opt.agg_rate_conf.scale=1.0 opt.agg_rate_conf.a_exp=1.0"),  # rate 1/(1+τ)
        "ps": dict(hp="opt.phase_step=true"),  # server lr x loss phase
        "cp": dict(hp=f"{_DKCF} sel.dynamic_kc.policy_kwargs.phase=true"),  # c x loss phase
        "pscp": dict(hp=f"opt.phase_step=true {_DKCF} sel.dynamic_kc.policy_kwargs.phase=true"),
        "ca": dict(hp=f"{_DKCF} sel.dynamic_kc.policy_kwargs.avail=true", cells=("s50u",)),  # c x trainable share (syn_0: = c)
        "cap": dict(hp=f"{_DKCF} sel.dynamic_kc.policy_kwargs.avail=true sel.dynamic_kc.policy_kwargs.phase=true", cells=("s50u",)),
        "psca": dict(hp=f"opt.phase_step=true {_DKCF} sel.dynamic_kc.policy_kwargs.avail=true", cells=("s50u",)),
        "c30": dict(c=30, cells=("s0u", "s50u")), "k20": dict(agg_goal=20, cells=("s0u", "s50u"))}  # K x c probe: c / K 3
F46_ARMS = {"ctl_srv": dict(bls=("fedbuff",))}
for _b, (_bhp, _bthp) in {"uo": (_S3, ""), "vub": (f"{_S3U} {_BAL}", "utilityPrefixSamples=-1")}.items():
    for _v, _kw in _S4V.items():
        _vhp = _kw.get("hp", "")
        _hp = f"{_bhp.replace('opt.agg_rate_conf.type=old', '') if _RNEW in _vhp else _bhp} {_vhp} {_STATS}"
        F46_ARMS[f"{_b}_{_v}"] = dict(hp=" ".join(_hp.split()), thp=" ".join(f"{_B1T if _kw.get('lrn', True) else ''} {_bthp}".split()),
                                      **{k: _kw[k] for k in ("cells", "c", "agg_goal") if k in _kw})
# F47 = FL1 S4b + S5 on Felix new (`fn` = F46 `uo_n`); streams 48 h (two trace days, ceiling 68.8%), static 24 h (stops at 70%):
# c by phase / availability, guarded local epochs, fresh augmentation per batch (FX-D233; FedBuff too); `smu` = MobiPerf stream.
F47_CELLS = {**F44_CELLS["cifar10"], "smu": ("mobiperf_3st", True)}
F47_RUNTIME_S = {"s0": 86400, "s0u": 172800, "s50u": 172800, "smu": 172800}
_FN = " ".join(f"{_S3.replace('opt.agg_rate_conf.type=old', '')} {_RNEW} {_STATS}".split())
_FCP, _AUG = f"{_FN} {_DKCF} sel.dynamic_kc.policy_kwargs.phase=true", "trainerAugmentEachPass=true"
F47_ARMS = {"ctl_srv": dict(bls=("fedbuff",)), "ctl_aug": dict(bls=("fedbuff",), hp="serveEma=sweep", thp=_AUG),
            "fn": dict(hp=_FN, thp=_B1T), "fn_cp": dict(hp=_FCP, thp=_B1T),
            "fn_c30": dict(hp=_FN, thp=_B1T, c=30, cells=("s0u", "s50u")),
            "fn_ca": dict(hp=f"{_FN} {_DKCF} sel.dynamic_kc.policy_kwargs.avail=true", thp=_B1T, cells=("smu",)),
            "fn_cap": dict(hp=f"{_FCP} sel.dynamic_kc.policy_kwargs.avail=true", thp=_B1T, cells=("smu",)),
            **{f"fn_e{e}": dict(hp=f"{_FN} {_GUARD}", thp=f"{_B1T} epochs={e}", cells=("s0", "s0u", "s50u")) for e in (2, 3)},
            "fn_aug": dict(hp=_FN, thp=f"{_B1T} {_AUG}"), "fn_cp_aug": dict(hp=_FCP, thp=f"{_B1T} {_AUG}", cells=("s0u", "s50u"))}
# F48R = rate `new` sweep on `fn_chr` (constants from 2024, before every setup fix): w = s (1+τ)^-a + (1-s)(1-(1+u)^-b), one knob
# at a time around s 0.4 / a 0.25 / b 0.1; `s0u` + `smu`, 12 h, FedBuff anchor.
F48R_CELLS = {"s0u": ("syn_0", True), "smu": ("mobiperf_3st", True)}
# F48 = FL1 S6 (augmentation now default, FX-N143); F48L = its 24 h confirm (FedBuff, `fn`, `fn_chr`, `fn_ev`). cifar: 3 h MobiPerf screen of c bounds (FX-N144: `pool` keeps c <= pool x
# trainable, binds only on a small pool); speech n 100: Felix new vs old vs FedBuff at c 20 (FedBuff's c / n, Q-SPEECHC).
F48_CELLS = {"cifar10": {"smu": ("mobiperf_3st", True)},
             "google_speech": {"s0u": ("syn_0", True), "s50u": ("syn_50_3st", True), "smu": ("mobiperf_3st", True)}}
F48_RUNTIME_S = {"cifar10": 10800, "google_speech": 43200}
F48_SHAPE = {"cifar10": F46_SHAPE, "google_speech": dict(F43_SHAPE["google_speech"], c=20)}
_FPOOL = f"{_FN} {_DKCF} sel.dynamic_kc.policy_kwargs.pool="
_FCHR = f"{_FPOOL}0.5 sel.dynamic_kc.policy_kwargs.ramp=true"
# Felix new, frozen 10-25: `fn_chr` at rate b 0.2 (baseline_reference.yaml); arm `fx`
FELIX_FINAL = _FCHR
F48_ARMS = {"cifar10": {"ctl_srv": dict(bls=("fedbuff",)), "fn": dict(hp=_FN, thp=_B1T), "fn_ca": F47_ARMS["fn_ca"],
                        "fn_ch2": dict(hp=f"{_FPOOL}0.5", thp=_B1T), "fn_ch3": dict(hp=f"{_FPOOL}0.33", thp=_B1T),
                        "fn_chp": dict(hp=f"{_FPOOL}0.5 sel.dynamic_kc.policy_kwargs.phase=true", thp=_B1T),
                        "fn_chr": dict(hp=f"{_FPOOL}0.5 sel.dynamic_kc.policy_kwargs.ramp=true", thp=_B1T),
                        "fn_ev": dict(hp=f"{_FPOOL}0.5 sel.dynamic_kc.policy_kwargs.ramp=true".replace(_E0, ""), thp=_B1T)},  # + Felix old's eval tasks
            "google_speech": {"ctl_srv": dict(bls=("felix", "fedbuff")), "fx": dict(hp=FELIX_FINAL, thp=_B1T)}}
F48R_ARMS = {"ctl_srv": dict(bls=("fedbuff",)), "fn_chr": dict(hp=_FCHR, thp=_B1T),
             **{f"r_{k}": dict(hp=f"{_FCHR} opt.agg_rate_conf.{v}", thp=_B1T) for k, v in dict(
                 s02="scale=0.2", s07="scale=0.7", a0="a_exp=0.0", a05="a_exp=0.5", b005="b_exp=0.05", b02="b_exp=0.2").items()}}
# F49 = decider (operator 10-24): Felix new (`fn_chr`) vs every baseline + `*` arms, both datasets x 3 stream cells, 12 h, bench.
# Async stacks at FedBuff's K 10 / c (F48_SHAPE); sync stacks keep the launcher's K rule (agg_goal None); `fn` (c 60) on MobiPerf.
F49_CELLS = {"s0u": ("syn_0", True), "s50u": ("syn_50_3st", True), "smu": ("mobiperf_3st", True)}
F49_ARMS = {"fn_chr": dict(hp=_FCHR, thp=_B1T), "ctl_srv": dict(bls=("fedbuff",), hp="serveEma=sweep"),
            "fn": dict(hp=_FN, thp=_B1T, cells=("smu",)),
            "sync_srv": dict(bls=("refl", "feddance", "fedavg", "oort", "oort_star"), hp="serveEma=sweep", agg_goal=None),
            "rstar_srv": dict(F42_ARMS["rstar_srv"], agg_goal=None), "fdstar_srv": dict(F42_ARMS["fdstar_srv"], agg_goal=None)}
# FB1 = paper runs (FX-N150 / N131): F49's stacks + `*` arms with Felix = `fx`, plus the ladder (FedBuff, b1, uo_o, fn, fx); 4 cells,
# streams from +24 h of the 6-day trace (<= 120 h); each leg stops when its 5-eval mean hits targets.json. FB1S = 2 h smoke, low target.
FB1_CELLS = {"s0": ("syn_0", False), **F49_CELLS}
FB1_ARMS = {"fx": dict(hp=FELIX_FINAL, thp=_B1T), **{a: F49_ARMS[a] for a in ("ctl_srv", "sync_srv", "rstar_srv", "fdstar_srv")},
            "b1": F44_ARMS["b1"], "uo_o": F46_ARMS["uo_o"], "fn": dict(hp=_FN, thp=_B1T)}
FB1_RUNTIME_S = {"FB1": 432000, "FB1S": 7200}
FB1S_TARGETS = {"cifar10": 0.35, "google_speech": 0.15}  # reachable in 2 h by the async stacks (F49; FB1S speech: Felix 22%, FedBuff 16%)
# FB2 = the paper's run sets (FX-N154; felix_paper_outline Table 1, manifest scripts/paper_runs.py): tier FB2<set>, smoke FB2<set>S
# (2 h, FB1S_TARGETS). Cells as FB1 + `s20u`; swaps are `fx` with one layer replaced; every leg stops at its target or 120 h.
FB2_CELLS = {**FB1_CELLS, "s20u": ("syn_20_3st", True)}
FB2_ABL = ("s50u", "smu")  # ablation cells
_fxs = lambda old, new, **kw: dict(hp=" ".join(FELIX_FINAL.replace(old, new).split()), thp=_B1T, **kw)  # `fx`, one knob swapped
_fxa = lambda add, thp="", **kw: dict(hp=f"{FELIX_FINAL} {add}", thp=f"{_B1T} {thp}".strip(), **kw)  # `fx` + knobs
_DEPLOY = ("refl", "feddance", "fedavg", "oort")
FB2_SETS = {
    "E2E": dict(cells=("s0u", "s20u", "s50u", "smu"), prio=1, arms={
        **{a: FB1_ARMS[a] for a in ("fx", "ctl_srv", "sync_srv", "rstar_srv")},
        # FX-N163 target anchor: FedDance* has no decay-off crossing -> its family's every (baseline_reference.yaml)
        "fdstar_srv": FB1_ARMS["fdstar_srv"],
        "fbstar_srv": F42_ARMS["fbstar_srv"], "favstar_srv": dict(bls=("fedavg",), hp="avail_knowledge=oracle serveEma=sweep", agg_goal=None)}),
    "LAD": dict(cells=FB2_ABL, prio=2, arms={a: FB1_ARMS[a] for a in ("b1", "uo_o", "fn")}),
    "SWC": dict(cells=FB2_ABL, prio=2, arms={"fx_cfix": dict(hp=_FN, thp=_B1T, c={"cifar10": 30, "google_speech": 10}),  # ½ FedBuff's c
                                             "fx_ca": dict(hp=f"{_FN} {_DKCF} sel.dynamic_kc.policy_kwargs.avail=true", thp=_B1T)}),
    "SWA": dict(cells=FB2_ABL, prio=2, arms={"fx_ck": _fxa("avail_knowledge=checkin proactive_inflight_evict=False"),
                                             "fx_or": _fxa("avail_knowledge=oracle proactive_inflight_evict=False")}),
    "SWG": dict(cells=FB2_ABL, prio=3, arms={"fx_rold": _fxs(_RNEW, "opt.agg_rate_conf.type=old"),  # 1/√(1+τ)
                                             "fx_r1": _fxa("opt.agg_rate_conf.scale=1.0 opt.agg_rate_conf.a_exp=1.0")}),  # 1/(1+τ)
    "SWS": dict(cells=FB2_ABL, prio=3, arms={"fx_rnd": _fxs("sel.selectType=prop", "sel.selectType=random"),
                                             "fx_oort": _fxs("sel.selectType=prop", "sel.selectType=default")}),  # speed + UCB + cutoff
    "REJ": dict(cells=FB2_ABL, prio=3, arms={"fx_ucb": _fxa("sel.scoreUcb=true"), "fx_e2": _fxa(_GUARD, thp="epochs=2")}),
    "STA": dict(cells=("s0",), prio=4, arms={"fx": FB1_ARMS["fx"], "ctl_srv": FB1_ARMS["ctl_srv"],
                                             "sync_srv": dict(FB1_ARMS["sync_srv"], bls=_DEPLOY)}),
}
FB2_TIERS = {f"FB2{k}{s}": k for k in FB2_SETS for s in ("", "S")}


def final_targets() -> dict:
    """{dataset: final target} from `_metadata/targets.json` (FX-N146)."""
    return {k: v for k, v in json.load(open(SCRIPT_DIR.parent / "_metadata" / "targets.json")).items() if k != "src"}


def trace_target(ds: str, trace: str) -> float:
    """Reporting target of a dataset on a trace: `_metadata/trace_targets.json` override, else the final target."""
    ov = json.load(open(SCRIPT_DIR.parent / "_metadata" / "trace_targets.json")).get(ds, {})
    return min(final_targets()[ds], ov.get(trace, 1.0))


F42_ARMS["l3b_ro_srv"] = dict(hp=f"{_L3B} {_STATS} serveEma=sweep opt.agg_rate_conf.type=old", thp=_PRE)  # FX-N132: FedBuff's rate
DETERMINISTIC = "trainerDeterministic=true"  # FX-D158: G0U/G0T pairs are equality legs (PL15); off, GPU numerics fork near-ties
G0U_RUNTIME_S = {"cifar10": 900, "google_speech": 1800}  # speech: ~4 cycles of its 450s timeout (FX-D46)
STREAM_T = 'data_streaming={"enabled":"True","full_data_available_after_s":240}'
STREAM_A = STREAM_T + ' checkpoint={"enabled":"True","every_n_rounds":10}'
STREAM_MODES = ("linear", "events")


def stream_hp(mode: str, runtime_s: int, trace_scale: str) -> str:
    """FX-N13 screen: 10% -> 100% over the leg's own trace span (horizon = runtime x trace scale, ST3)."""
    horizon = int(runtime_s * float(trace_scale or 1))
    return ('data_streaming={"enabled":"True","mode":"%s","initial_frac":0.1,"full_data_available_after_s":%d}'
            % (mode, horizon))


def stream_phases(pid, ds, traces, oracle, mk) -> "List[Phase]":
    """One phase per (stream mode, trace); mk(pid, trace) builds the Phase at its tier's shape."""
    out = []
    for mode in STREAM_MODES:
        for t in traces:
            ph = mk(f"{pid}_{mode[:3]}_{t}", t)
            ph.trainer_hp = stream_hp(mode, ph.runtime_s, ph.trace_scale)
            ph.agg_hp = ph.trainer_hp + (" " + oracle_a(ds) if oracle else "")
            out.append(ph)
    return out


def oracle_a(dataset: str) -> str:
    return ('oracle_utility_injection={"enabled":"True","alpha":0.1,"num_trainers":%d,"sample_size":64}'
            % GPU_N[dataset])


# Changed path -> affected baselines (first match wins); real=True means the real path changed (T3 needed).
DELTA_RULES = (
    ("*.md", (), False),
    ("lib/python/flame/mode/horizontal/syncfl/trainer.py", B6, True),
    ("lib/python/flame/mode/horizontal/asyncfl/trainer.py", ("felix", "fedbuff"), True),
    ("lib/python/flame/mode/horizontal/asyncfl/*", ("felix", "fedbuff"), False),
    ("lib/python/flame/mode/horizontal/syncfl/top_aggregator.py", ("oort", "oort_star", "refl", "feddance"), False),
    ("lib/python/flame/mode/horizontal/syncfl/fwdllm_*", (), False),
    ("lib/python/flame/selector/async_oort.py", ("felix",), False),
    ("lib/python/flame/selector/fedbuff.py", ("fedbuff",), False),
    ("lib/python/flame/selector/async_base.py", ("felix", "fedbuff"), False),
    ("lib/python/flame/selector/scoring.py", ("felix", "oort", "oort_star", "refl"), False),
    ("lib/python/flame/selector/oort.py", ("oort", "oort_star", "refl"), False),
    ("lib/python/flame/selector/refl_oort.py", ("refl",), False),
    ("lib/python/flame/selector/feddance.py", ("feddance",), False),
    ("lib/python/flame/optimizer/fedbuff.py", ("felix", "fedbuff"), False),
    ("lib/python/flame/optimizer/fedavg.py", ("oort", "oort_star", "feddance"), False),
    ("lib/python/flame/optimizer/refl.py", ("refl",), False),
    ("lib/python/flame/optimizer/fedscale_yogi.py", ("oort", "oort_star", "refl"), False),
    ("lib/python/flame/availability/feddance_predictor.py", ("feddance",), False),
    ("lib/python/flame/availability/refl_tracker.py", ("refl",), False),
    ("lib/python/flame/channel*.py", B6, True),
    ("lib/python/flame/backend/*", B6, True),
    ("lib/python/flame/*", B6, False),
    ("lib/python/examples/async_cifar10/trainer/*", B6, True),
    ("lib/python/examples/async_cifar10/aggregator/pytorch/main_asyncfl_agg.py", ("felix", "fedbuff"), False),
    ("lib/python/examples/async_cifar10/aggregator/pytorch/main_oort_sync_agg.py", ("oort", "oort_star", "refl"), False),
    ("lib/python/examples/async_cifar10/aggregator/pytorch/main_fedavg_agg.py", ("feddance",), False),
    ("lib/python/examples/async_cifar10/aggregator/*", B6, False),
    ("lib/python/examples/async_cifar10/scripts/*", B6, False),
    ("lib/python/examples/scripts/expt_runner.sh", B6, False),
    ("lib/python/examples/async_cifar10/expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml", B6, False),
    ("lib/python/examples/_metadata/baselines.yaml", B6, False),
    ("lib/python/examples/_metadata/datasets.yaml", B6, False),
    ("lib/python/examples/_metadata/dataset_splits/*", B6, False),
    ("lib/python/examples/async_cifar10/fl_data.py", B6, True),
)


def affected(paths) -> tuple:
    """(baselines in B6 order, real_path_changed) for a list of repo-relative changed paths."""
    hit, real = set(), False
    for p in paths:
        for pat, bls, is_real in DELTA_RULES:
            if fnmatch.fnmatch(p, pat):
                hit |= set(bls)
                real |= is_real and bool(bls)
                break
    return tuple(b for b in B6 if b in hit), real


def changed_paths(ref: str) -> List[str]:
    """Tracked diff vs `ref` (committed + working tree) plus untracked files."""
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.split()
    return sorted(set(run("diff", "--name-only", ref)) | set(run("ls-files", "--others", "--exclude-standard")))


# ---------------------------------------------------------------- phases and jobs
@dataclass
class Phase:
    pid: str
    baselines: tuple
    trace: str
    runtime_s: int
    n: int
    dataset: str = "cifar10"
    harness: str = "stub"
    trace_scale: str = ""
    trace_offset_s: str = ""  # FX-N123: leg starts this far into every trace (availability + usage stream)
    agg_goal: Optional[int] = None  # None = the config's (GPU legs keep the reference knobs)
    c: Optional[int] = None
    sim_ceiling_x: int = 1  # sim wall ceiling = x * runtime (speech stub sims are aggregator-bound on CPU)
    agg_hp: str = ""
    trainer_hp: str = ""
    inject_bug: str = ""
    kind: str = "pair"  # pair: real+sim legs + grade | sim: sim leg vs banked real | sim_ev: sim leg, EV only | real: real leg, EV only
    cpt: Optional[float] = None  # CPUs per trainer for this phase; None = --cpus-per-trainer
    gpus: Optional[int] = None  # GPU legs: GPUs each; None = --gpus-per-job
    sim_gpus: Optional[int] = None  # a pair's sim leg: GPUs each (FX-L81: one train slot per GPU paces it); None = gpus
    delay_factor: str = ""  # divides every trainer's D (debug_run --delay-factor)
    full_runtime_s: int = 0  # runtime before a --scale-smoke cap (0 = uncapped)


def shaped(pid, baselines, shape, kind, dataset, **kw) -> Phase:
    sh = dict(SHAPES[shape])
    if dataset == "google_speech":
        sh["sim_ceiling_x"] = 2
    sh.update(kw)
    stub = sh.get("harness", "stub") == "stub"
    # FX-D110, FX-N85 b: 20-step stub passes overran D (speech; cifar Oort family 11/177)
    if stub and (dataset == "google_speech" or (baselines and all(b in OORT_FAMILY for b in baselines))):
        sh["trainer_hp"] = (sh.get("trainer_hp", "") + " harness_stub_max_steps=1").strip()
    return Phase(DS_TAG[dataset] + pid, tuple(baselines), kind=kind, dataset=dataset, **sh)


def oort_split(pid, baselines, shape, kind, dataset, **kw) -> List[Phase]:
    """shaped(), with the Oort family moved to its K=10 shape (FX-D106) under `<pid>s`."""
    rest = tuple(b for b in baselines if b not in OORT_FAMILY)
    fam = tuple(b for b in baselines if b in OORT_FAMILY)
    out = [shaped(pid, rest, shape, kind, dataset, **kw)] if rest else []
    if dataset == "google_speech" and shape == "mobiperf_3st":
        fam = ()  # mobiperf_3sts n=165 > speech's 100 partitions (FX-T36: needs a new split)
    if fam and OORT_SHAPE.get(shape, shape):
        out.append(shaped(pid + "s" if shape in OORT_SHAPE else pid, fam, OORT_SHAPE.get(shape, shape), kind, dataset, **kw))
    return out


def campaign_phases(ds: str) -> List[Phase]:
    ph = lambda pid, bls, shape, **kw: shaped(pid, bls, shape, "pair", ds, **kw)
    sp = lambda pid, shape: oort_split(pid, B6, shape, "pair", ds)
    return [
        *sp("P1", "syn_0"), *sp("P1b", "syn_0b"), *sp("P2", "syn_50_3st"), *sp("P3", "mobiperf_3st"),
        ph("P4", ("felix", "fedbuff"), "syn_0", agg_hp="simColdStartGate=false"),
        ph("P5", ("felix", "refl"), "syn_20_3st"),
        ph("P6", ("felix", "oort"), "syn_0", harness="tiny_cpu"),
        *([ph("P7", B6, "syn_0", harness="tiny_cpu", trainer_hp=STREAM_T, agg_hp=STREAM_A),
           ph("P7o", B6, "syn_0", harness="tiny_cpu", trainer_hp=STREAM_T, agg_hp=STREAM_A + " " + oracle_a(ds))]
          if ds == "cifar10" else []),  # FX-N86 (c): speech tiny_cpu compute > D; speech streaming is G0T only
        ph("P8", ("fedbuff",), "syn_50_3st", agg_hp="taskRetryPolicy=exponential taskRetryBackoffSeconds=10"),
        ph("P9", ("felix", "fedbuff"), "syn_0", agg_hp="real_drain_ready_ingest=false"),  # legacy recv_fifo control
        # FX-N18 control: fedbuff with the old 0.1s real-only settle sleep (default is now 0)
        ph("P10", ("fedbuff",), "syn_0", agg_hp="realDistributeSettleSeconds=0.1"),
    ] + injected_phases("pair", ds)


def injected_phases(kind, ds) -> List[Phase]:
    """S1: each injected bug must FAIL its sim rung (EV10 / EV16 / EV3)."""
    # FX-N32: P11b on fedbuff, which withholds (felix evicts, so order_by_sct had 0-1 withheld commits to catch).
    return [shaped(pid, (bl,), "syn_50_3st", kind, ds, inject_bug=bug)
            for pid, bl, bug in (("P11a", "felix", "no_busy_hold"), ("P11b", "fedbuff", "order_by_sct"),
                                 ("P11c", "felix", "freeze_trainer_clock"))]


def tier_phases(tier: str, baselines: tuple, ds: str = "cifar10") -> List[Phase]:
    matrix = ("syn_0", "syn_0b", "syn_20_3st", "syn_50_3st", "mobiperf_3st")  # FX-N78: syn_20_3st in T3
    if tier == "T1":
        # syn_50_3st 360 s: unaware oort first commits at vclock 150 / 300 (FX-L43).
        return [shaped("T1", baselines, sh, "sim_ev", ds, runtime_s={"syn_0": 120, "syn_50_3st": 360}[sh])
                for sh in ("syn_0", "syn_50_3st")]
    if tier == "T2":
        return [p for sh in matrix for p in oort_split(f"T2_{sh}", baselines, sh, "sim", ds)] + injected_phases("sim_ev", ds)
    if tier == "T3":
        return [p for sh in matrix for p in oort_split(f"T3_{sh}", baselines, sh, "pair", ds)]
    if tier == "T4":
        return campaign_phases(ds)
    if tier == "T3S":  # FX-N67: straggler carry-over pair for the over-selecting sync stacks
        bls = tuple(x for x in baselines if x in ("oort", "oort_star", "refl")) if baselines != B6 else ("oort", "oort_star", "refl")
        return [shaped(f"T3S_{sh}", bls, sh, "pair", ds) for sh in ("syn_0s", "syn_20_3sts")]
    if tier == "G0":  # GPU screen: all six, syn_0 + syn_20_3st (stationary, ~20% within 30 min), smaller cohort (FX-N34)
        return [Phase(f"{DS_TAG[ds]}G0_{t}", baselines, t, runtime_s=1800, dataset=ds, harness="none", **G0_SHAPE[ds])
                for t in ("syn_0", "syn_20_3st")]
    if tier == "GS":  # FX-N42 L5: GPU short pairs, G0 cohort, 10 min (GPU-only faults and the clock rungs, cheaply)
        return [Phase(f"{DS_TAG[ds]}GS_{t}", baselines, t, runtime_s=600, dataset=ds, harness="none", **G0_SHAPE[ds])
                for t in ("syn_0", "syn_20_3st")]
    if tier == "G0C":  # R7 control: a second real leg per G0 syn_0 cell (real<->real floor for G0's pairs)
        return [Phase(f"{DS_TAG[ds]}G0C_syn_0", baselines, "syn_0", runtime_s=1800, dataset=ds, harness="none",
                      kind="real", **G0_SHAPE[ds])]
    if tier == "N64":  # FX-N64: REFL speech model health, 25 min pair + a real replicate (model_health telemetry, EV18)
        return [Phase(f"{DS_TAG[ds]}N64", ("refl",), "syn_0", runtime_s=1500, dataset=ds, harness="none", **G0_SHAPE[ds]),
                Phase(f"{DS_TAG[ds]}N64C", ("refl",), "syn_0", runtime_s=1500, dataset=ds, harness="none",
                      kind="real", **G0_SHAPE[ds])]
    if tier == "N64S":  # FX-N64 verify: sim-only refl past round 650 (where running_var went negative), EV18
        return [Phase(f"{DS_TAG[ds]}N64S", ("refl",), "syn_0", runtime_s=2400, dataset=ds, harness="none",
                      kind="sim_ev", **G0_SHAPE[ds])]
    if tier == "G1":  # GPU block on the production path at the dataset's reference config (L7)
        bls = tuple(b for b in baselines if b in ("felix", "fedbuff")) if baselines != B6 else ("felix", "fedbuff")
        return [Phase(DS_TAG[ds] + "G1", bls, "syn_0", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none")]
    if tier == "G2":  # the other four (FX-N5)
        bls = tuple(b for b in baselines if b not in ("felix", "fedbuff")) or B6[2:]
        return [Phase(DS_TAG[ds] + "G2", bls, "syn_0", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none")]
    if tier in ("G2S", "G2C"):  # FX-N68: sim-only / real-only G2 replicates (floor, R7)
        bls = tuple(b for b in baselines if b not in ("felix", "fedbuff")) or B6[2:]
        return [Phase(DS_TAG[ds] + tier, bls, "syn_0", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none",
                      kind="sim_ev" if tier == "G2S" else "real")]
    if tier == "G1S":  # FX-D42/D43/D44 confirm: felix + fedbuff at the reference n, 45 min, 3 GPUs (7 healthy on jayne)
        bls = tuple(b for b in baselines if b in ("felix", "fedbuff")) if baselines != B6 else ("felix", "fedbuff")
        return [Phase(DS_TAG[ds] + "G1S", bls, "syn_0", runtime_s=2700, n=GPU_N[ds], dataset=ds, harness="none", gpus=4)]
    if tier == "G0U":  # FX-N9 screen: syn_50_3st + mobiperf_3st at trace scale 4, EV + logical parity
        rest = tuple(b for b in baselines if b not in OORT_FAMILY)
        fam = tuple(b for b in baselines if b in OORT_FAMILY)
        g0u = lambda pid, bls, t, shape: Phase(pid, bls, t, runtime_s=G0U_RUNTIME_S[ds], dataset=ds, harness="none",
                                               trace_scale="4", trainer_hp=DETERMINISTIC, **shape)
        return ([g0u(f"{DS_TAG[ds]}G0U_{t}", rest, t, {**G0U_SHAPE[ds], **(G0U_MOBIPERF if t == "mobiperf_3st" else {})})
                 for t in ("syn_50_3st", "mobiperf_3st") if rest]
                + [g0u(f"{DS_TAG[ds]}G0U_{t}s", fam, t, G0U_OORT[(ds, t)])
                   for t in ("syn_50_3st", "mobiperf_3st") if fam and (ds, t) in G0U_OORT])
    if tier in ("TS", "TSo"):  # FX-N13 ST5 CPU: tiny_cpu pairs x {linear, events}; TSo = the *_oracle arms
        if ds == "google_speech":
            return []  # FX-N86 (c)
        return stream_phases(DS_TAG[ds] + tier, ds, ("syn_0", "syn_50_3st"), tier == "TSo",
                             lambda pid, t: shaped(pid, baselines, t, "pair", ds, harness="tiny_cpu"))
    if tier in ("G0T", "G0To"):  # FX-N13 ST5 GPU screen (C11): G0U cohort x {linear, events} x {syn_0, syn_50_3st}
        rest = tuple(b for b in baselines if b not in OORT_FAMILY)
        fam = tuple(b for b in baselines if b in OORT_FAMILY)
        mk = lambda bls, shape: lambda pid, t: Phase(pid, bls, t, runtime_s=G0U_RUNTIME_S[ds], dataset=ds, harness="none",
                                                     trace_scale="" if t == "syn_0" else "4", **shape)
        out = stream_phases(DS_TAG[ds] + tier, ds, ("syn_0", "syn_50_3st"), tier == "G0To", mk(rest, G0U_SHAPE[ds])) if rest else []
        if fam:  # FX-L63: Oort family at its K=10 G0U shape; `<pid>s` so G0U_syn_50_3sts floors map onto it
            fam_ph = stream_phases(DS_TAG[ds] + tier, ds, ("syn_0", "syn_50_3st"), tier == "G0To",
                                   mk(fam, G0U_OORT[(ds, "syn_50_3st")]))
            out += [replace(ph, pid=ph.pid + "s") for ph in fam_ph]
        return [replace(ph, trainer_hp=f"{ph.trainer_hp} {DETERMINISTIC}") for ph in out]
    if tier == "G0TC":  # R7 control: a real replicate per G0T cell (same-code real<->real streaming floor)
        return [replace(ph, pid=ph.pid.replace("G0T_", "G0TC_"), kind="real") for ph in tier_phases("G0T", baselines, ds)]
    if tier == "T3C":  # R7 control: a real replicate per T3 cell (floors flip-boundary forks, FX-L53)
        return [replace(ph, pid=ph.pid.replace("T3_", "T3C_"), kind="real") for ph in tier_phases("T3", baselines, ds)]
    if tier == "G0UC":  # R7 control: a real replicate per G0U cell (same-code real<->real floor, Q2)
        return [replace(ph, pid=ph.pid.replace("G0U_", "G0UC_"), kind="real") for ph in tier_phases("G0U", baselines, ds)]
    if tier == "G0US":  # FX-D194: sim-only replicates of every G0U cell, graded as G3S
        return [replace(ph, pid=ph.pid.replace("G0U_", f"G0US{i}_"), kind="sim")
                for i in range(1, G3S_REPS + 1) for ph in tier_phases("G0U", baselines, ds)]
    if tier == "CHG":  # L17: real syn_0 legs at the G0U cohort, input to profile_felix_charges.py after a transport change
        return [Phase(f"{DS_TAG[ds]}CHG", baselines, "syn_0", runtime_s=600, dataset=ds, harness="none", kind="real",
                      **G0U_SHAPE[ds])]
    if tier == "G1U":  # FX-N9 long confirm: felix + fedbuff at the reference n on syn_50_3st, production trace timeline
        bls = tuple(b for b in baselines if b in ("felix", "fedbuff")) if baselines != B6 else ("felix", "fedbuff")
        # FX-D52: cifar n=300 takes every GPU (3 GPUs OOMed at init).
        return [Phase(DS_TAG[ds] + "G1U_syn_50_3st", bls, "syn_50_3st", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none",
                      gpus=4 if ds == "google_speech" else None)]
    if tier in ("G1A", "G1AS"):  # FX-N74: accuracy at the reference config on full data; G1AS = sim legs only
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        return [Phase(DS_TAG[ds] + tier, baselines, "syn_0", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none",
                      trainer_hp=full, agg_hp=full + " evalEveryNRounds=20", gpus=4,
                      **({"kind": "sim_ev"} if tier == "G1AS" else {}))]
    if tier in ("G3", "G3C"):  # PR37: convergence pairs at one n:c:K for all six; G3C = real replicate
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        kw = dict(dataset=ds, harness="none", runtime_s=G3_RUNTIME_S, **G3_SHAPE[ds])
        out = [Phase(f"{DS_TAG[ds]}G3_{t}", baselines, t, trainer_hp=f"{full} {DETERMINISTIC}",
                     agg_hp=f"{full} {G3_EVAL}", **kw) for t in ("syn_0", "syn_50_3st")]
        lin = stream_hp("linear", G3_RUNTIME_S, "")
        out.append(Phase(f"{DS_TAG[ds]}G3_lin_syn_0", baselines, "syn_0", trainer_hp=f"{lin} {DETERMINISTIC}",
                         agg_hp=f"{lin} {G3_EVAL}", **kw))
        if tier == "G3C":
            out = [replace(ph, pid=ph.pid.replace("G3_", "G3C_"), kind="real") for ph in out]
        return out
    if tier == "F42":
        g3 = next(ph for ph in tier_phases("G3", ("felix",), ds) if ph.pid.endswith("G3_syn_0"))
        out = []
        for cell, trace in F42_CELLS.items():
            dsh = usage_hp(trace)
            base = replace(g3, runtime_s=F42_RUNTIME_S[ds], trace=trace, trace_offset_s="86400", kind="sim_ev",
                           trainer_hp=f"{dsh} {DETERMINISTIC}", agg_hp=f"{dsh} evalEverySeconds={F42_EVAL_S}")
            for arm, kw in F42_ARMS.items():
                bls = tuple(b for b in kw.get("bls", ("felix",)) if b in baselines)
                for i in F42_SEEDS if bls else ():  # seed 1 keeps the bare id (F42S01); --seeds 2 adds _r2 (seed 1235)
                    out.append(replace(base, pid=f"{DS_TAG[ds]}F42_{cell}_{arm}" + (f"_r{i}" if i > 1 else ""), baselines=bls,
                                       agg_hp=f"{base.agg_hp} {kw.get('hp', '')}{f' seed={1233 + i}' if i > 1 else ''}".strip(),
                                       trainer_hp=f"{base.trainer_hp} {kw.get('thp', '')}".strip()))
        return out
    if tier == "F43":  # FX-N135 / N139: F43_ARMS at the reference cohort (c the same share of n, sync K/N >= 5%, L34)
        out = []
        for ph in tier_phases("F42", baselines, ds):
            arm = re.sub(r"^(?:gs_)?F42_s(?:0|50)u_", "", ph.pid)
            if arm in F43_ARMS:
                out.append(replace(ph, pid=ph.pid.replace("F42_", "F43_"), runtime_s=F43_RUNTIME_S[ds], agg_hp=f"{ph.agg_hp} {F43_STOP}", **F43_SHAPE[ds]))
        return out
    if tier in ("F44", "F45", "F46", "F47", "F48", "F48L", "F48R", "F49"):  # FX-N141 loop FL1: F42 arms reused for anchors, the step's arms on top
        g3 = next(ph for ph in tier_phases("G3", ("felix",), ds) if ph.pid.endswith("G3_syn_0"))
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        out = []
        f48 = tier in ("F48", "F48L", "F48R", "F49")
        cells = F49_CELLS if tier == "F49" else F48R_CELLS if tier == "F48R" else F48_CELLS[ds] if f48 else F47_CELLS if tier == "F47" and ds == "cifar10" else F44_CELLS[ds]
        for cell, (trace, stream) in cells.items():
            dsh = usage_hp(trace) if stream else full
            rt = 86400 if tier == "F48L" else 43200 if tier in ("F48R", "F49") else F48_RUNTIME_S[ds] if f48 else F47_RUNTIME_S[cell] if tier == "F47" else F44_RUNTIME_S
            shape = F48_SHAPE[ds] if f48 else F46_SHAPE if tier in ("F46", "F47") and ds == "cifar10" else F44_SHAPE[ds]
            base = replace(g3, runtime_s=rt, trace=trace, trace_offset_s="86400" if stream else "", kind="sim_ev",
                           trainer_hp=f"{dsh} {DETERMINISTIC}", agg_hp=f"{dsh} evalEverySeconds={F42_EVAL_S} {'' if f48 else F43_STOP}".strip(),
                           **shape)  # F48 runs to the horizon (FX-T90)
            arms = {a: F48_ARMS[ds][a] for a in ("ctl_srv", "fn", "fn_chr", "fn_ev")} if tier == "F48L" else F48R_ARMS if tier == "F48R" \
                else F49_ARMS if tier == "F49" else F48_ARMS[ds] if f48 else {"F44": F44_ARMS, "F45": F45_ARMS, "F46": F46_ARMS, "F47": F47_ARMS}[tier]
            for arm, kw in arms.items():
                if ds == "google_speech" and tier not in ("F44", "F48", "F48L", "F49") or tier == "F44" and ds == "google_speech" and arm not in F44_SPEECH_ARMS \
                        or cell not in kw.get("cells", (cell,)):
                    continue
                ref = F42_ARMS.get(arm, {})
                hp, thp = kw.get("hp", ref.get("hp", "")), kw.get("thp", ref.get("thp", ""))
                bls = tuple(b for b in kw.get("bls", ref.get("bls", ("felix",))) if b in baselines)
                if bls:
                    out.append(replace(base, pid=f"{DS_TAG[ds]}{tier}_{cell}_{arm}", baselines=bls,
                                       agg_hp=f"{base.agg_hp} {hp}".strip(), trainer_hp=f"{base.trainer_hp} {thp}".strip(),
                                       **{k: kw[k] for k in ("c", "agg_goal") if k in kw}))
        return out
    if tier in FB2_TIERS:
        rs, smoke = FB2_SETS[FB2_TIERS[tier]], tier[3:] != FB2_TIERS[tier]
        g3 = next(ph for ph in tier_phases("G3", ("felix",), ds) if ph.pid.endswith("G3_syn_0"))
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        out = []
        for cell in rs["cells"]:
            trace, stream = FB2_CELLS[cell]
            dsh = usage_hp(trace) if stream else full
            stop = f"targetAccuracy={FB1S_TARGETS[ds] if smoke else trace_target(ds, trace)} targetWindow=5"
            base = replace(g3, runtime_s=FB1_RUNTIME_S["FB1S" if smoke else "FB1"], trace=trace, trace_offset_s="86400" if stream else "",
                           kind="sim_ev", trainer_hp=f"{dsh} {DETERMINISTIC}", agg_hp=f"{dsh} evalEverySeconds={F42_EVAL_S} {stop}",
                           **F48_SHAPE[ds])
            for arm, kw in rs["arms"].items():
                bls = tuple(b for b in kw.get("bls", ("felix",)) if b in baselines)
                if bls:
                    out.append(replace(base, pid=f"{DS_TAG[ds]}{tier}_{cell}_{arm}", baselines=bls,
                                       agg_hp=f"{base.agg_hp} {kw.get('hp', '')}".strip(),
                                       **{k: kw[k][ds] if isinstance(kw[k], dict) else kw[k] for k in ("c", "agg_goal") if k in kw},
                                       trainer_hp=f"{base.trainer_hp} {kw['thp'].get(ds, '') if isinstance(kw.get('thp'), dict) else kw.get('thp', '')}".strip()))
        return out
    if tier in FB1_RUNTIME_S:
        g3 = next(ph for ph in tier_phases("G3", ("felix",), ds) if ph.pid.endswith("G3_syn_0"))
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        out = []
        for cell, (trace, stream) in FB1_CELLS.items():
            dsh = usage_hp(trace) if stream else full
            stop = f"targetAccuracy={FB1S_TARGETS[ds] if tier == 'FB1S' else trace_target(ds, trace)} targetWindow=5"
            base = replace(g3, runtime_s=FB1_RUNTIME_S[tier], trace=trace, trace_offset_s="86400" if stream else "", kind="sim_ev",
                           trainer_hp=f"{dsh} {DETERMINISTIC}", agg_hp=f"{dsh} evalEverySeconds={F42_EVAL_S} {stop}", **F48_SHAPE[ds])
            for arm, kw in FB1_ARMS.items():
                bls = tuple(b for b in kw.get("bls", ("felix",)) if b in baselines)
                if bls:
                    out.append(replace(base, pid=f"{DS_TAG[ds]}{tier}_{cell}_{arm}", baselines=bls,
                                       agg_hp=f"{base.agg_hp} {kw.get('hp', '')}".strip(),
                                       trainer_hp=f"{base.trainer_hp} {kw.get('thp', '')}".strip(),
                                       **{k: kw[k] for k in ("c", "agg_goal") if k in kw}))
        return out
    if tier == "F42R":  # F42-T7 / T9 on GPU: G3 real legs, 8 min wall: Felix notify + the check-in baselines
        g3 = next(ph for ph in tier_phases("G3", ("felix",), ds) if ph.pid.endswith("G3_syn_0"))
        dsh = usage_hp("syn_50_3st")
        return [replace(g3, pid=f"{DS_TAG[ds]}F42R_s50u_ctl", baselines=tuple(b for b in F42_ARMS["ctl"]["bls"] if b in baselines),
                        runtime_s=480, trace="syn_50_3st", trace_offset_s="86400", kind="real",
                        trainer_hp=f"{dsh} {DETERMINISTIC}", agg_hp=f"{dsh} evalEverySeconds={F42_EVAL_S}")]
    if tier in ("F42C", "F42CR"):  # F42-T1 local gate on the CPU stub harness: F42C every arm in sim (30 min vclock);
        real = tier == "F42CR"      # F42CR the deployed arms in real mode (10 min wall, trace x4): check-in client, Felix notify (T7)
        out = []
        for arm, kw in F42_ARMS.items():
            bls = tuple(b for b in kw.get("bls", ("felix",)) if b in baselines)
            if bls and (not real or arm == "ctl"):
                out.append(Phase(f"{tier}_s50u_{arm}", bls, "syn_50_3st", runtime_s=600 if real else 1800, n=40, dataset=ds,
                                 harness="stub", trace_offset_s="86400", kind="real" if real else "sim_ev",
                                 trace_scale="4" if real else "", agg_hp=f"evalEverySeconds={F42_EVAL_S} {kw.get('hp', '')}".strip(),
                                 trainer_hp=kw.get("thp", "")))
        return out
    if tier == "G3S":  # sim-only replicates of every G3 cell: G3S<i>_<cell>, graded on banked + equivalent reals
        return [replace(ph, pid=ph.pid.replace("G3_", f"G3S{i}_"), kind="sim")
                for i in range(1, G3S_REPS + 1) for ph in tier_phases("G3", baselines, ds)]
    if tier in ("G1X", "G1XP"):  # FedDance sprint: EuroSys'26 setup, every baseline on one config; G1XP = real+sim convergence pair
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        kw = dict(n=GPU_N[ds], dataset=ds, harness="none", trainer_hp=f"{full} {EUROSYS_TRAINER_HP}", gpus=4, agg_goal=EUROSYS_K)
        if tier == "G1XP":  # FX-D149
            kw["trainer_hp"] += " trainerDeterministic=true"
            return [Phase(f"{DS_TAG[ds]}G1XP_syn_0", baselines, "syn_0", runtime_s=3600,
                          agg_hp=full + " evalEveryNRounds=20", **kw)]
        return [Phase(f"{DS_TAG[ds]}G1X_{t}", baselines, t, runtime_s=EUROSYS_HORIZON_S, kind="sim_ev",
                      agg_hp=f"{full} evalEveryNRounds=20 targetAccuracy={EUROSYS_TARGET[ds]}", **kw)
                for t in ("mobiperf_3st", "syn_50_3st", "syn_10_3st", "syn_0")]
    if tier in ("G1XS", "G1XQ"):  # FedDance sprint sanity: shared config at n=50 real+sim (15 min) / n=300 sim speed probe (2 h vclock)
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        kw = dict(dataset=ds, harness="none", trainer_hp=f"{full} {EUROSYS_TRAINER_HP}", agg_goal=EUROSYS_K)
        if tier == "G1XS":  # n=50 x K=10 can't fill mobiperf (G0U covers it)
            kw["trainer_hp"] += " trainerDeterministic=true"  # FX-D149
            return [Phase(f"{DS_TAG[ds]}G1XS_{t}", baselines, t, runtime_s=900, n=50, gpus=1 if ds == "cifar10" else 2,
                          agg_hp=full + " evalEveryNRounds=5", **kw) for t in ("syn_50_3st", "syn_10_3st")]
        # Speech sims run ~1x: 30 min vclock suffices to measure speed.
        return [Phase(f"{DS_TAG[ds]}G1XQ_mobiperf_3st", baselines, "mobiperf_3st", runtime_s=7200 if ds == "cifar10" else 1800,
                      n=GPU_N[ds], gpus=4, agg_hp=full + " evalEveryNRounds=20", kind="sim_ev", **kw)]
    if tier in ("G1XF", "G1XFS"):  # FedDance sprint F3: FedDance vs random (fedavg) on the FedDance paper's own training config
        # Reference n, K=10 as the paper: at n=50 both arms picked every eligible end (FX-L67).
        full = 'data_streaming={"enabled":"False","full_data_available_after_s":0}'
        return [Phase(f"{DS_TAG[ds]}{tier}_{t}", baselines, t, runtime_s=12000 if tier == "G1XF" else 3000, n=GPU_N[ds], dataset=ds,
                      harness="none", trainer_hp=f"{full} {FEDDANCE_PAPER_HP[ds]}", agg_hp=full + " evalEveryNRounds=20", gpus=4,
                      agg_goal=EUROSYS_K, kind="sim_ev")
                for t in (("mobiperf_3st", "syn_0") if tier == "G1XF" else ("mobiperf_3st",))]
    if tier == "G1L":  # FX-N65: felix syn_0 past the 7200s convergence floor, pair + a real replicate
        return [Phase(f"{DS_TAG[ds]}G1L", ("felix",), "syn_0", runtime_s=7500, n=GPU_N[ds], dataset=ds, harness="none"),
                Phase(f"{DS_TAG[ds]}G1LC", ("felix",), "syn_0", runtime_s=7500, n=GPU_N[ds], dataset=ds,
                      harness="none", kind="real")]
    if tier == "N62":  # FX-N62: unaware oort syn_50_3st long enough to grade its stall-free timing
        return [shaped("N62_syn_50_3st", ("oort",), "syn_50_3st", "pair", ds, runtime_s=10800)]
    if tier == "PROF":  # FX-N77: n=10, aggGoal = c = 10, D/10 real legs; profile with FLAME_PYSPY
        bls = baselines if baselines != B6 else ("fedbuff", "oort", "feddance")  # one per aggregator stack
        return [Phase(f"{DS_TAG[ds]}PROF", bls, "syn_0", runtime_s=240, n=10, dataset=ds, harness="none", agg_goal=10, c=10,
                      kind="real", gpus=1, delay_factor="10")]
    if tier in ("ISO", "ISO_FILL"):  # P6: felix/fedbuff pairs solo (--max-parallel 1), then packed
        bls = baselines if baselines != B6 else ("felix", "fedbuff")  # --baselines picks the stacks to control
        iso = oort_split("ISO", bls, "syn_0", "pair", ds)
        # packed stage: neighbours load the node (and double as FX-N20 syn_50_3st checks)
        return iso + ([shaped("FILL", B6[2:], "syn_50_3st", "sim_ev", ds)] if tier == "ISO_FILL" else [])
    raise SystemExit(f"unknown tier {tier}")


def agg_cores(n: int) -> int:
    """Aggregator cores for a slot: a solo n>=60 run gets 8 (runner CPU partition); small cohorts less."""
    return 2 if n <= 30 else 4 if n <= 60 else 8


def slot_cpus(n: int, per_trainer: float = 1.0, agg: Optional[int] = None) -> int:
    """Even logical-CPU count: the aggregator's cores + `per_trainer` CPUs per trainer
    (< 1 packs trainers onto shared CPUs; the spawner wraps them round-robin)."""
    s = (agg_cores(n) if agg is None else agg) + math.ceil(n * per_trainer)
    return s + (s % 2)


@dataclass
class Job:
    jid: str
    phase: str
    trace: str
    baseline: str
    mode: str  # real | sim | grade
    args: List[str]
    n: int
    runtime_s: int
    harness: str
    cpus: int
    mem_gb: float
    gpus: int = 0
    deps: tuple = ()
    est_s: float = 0.0
    unit: str = ""  # jobs that must land on the same shard (a pair and its grade)
    whole: bool = False  # takes the whole node: no taskset, no FLAME_AGG_CORES (= a solo run's layout)
    dataset: str = "cifar10"

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.harness}|{self.mode}|{self.trace}|{self.baseline}|n{self.n}|rt{self.runtime_s}"


def stall_limit_s(job: Job, stall_min: float) -> float:
    """FX-N111: S1 waits at least 1.25 x the leg's task timeout (FX-N115), so a round held by one abandon is no stall."""
    base = stall_min * 60
    if job.harness != "none":
        return base  # harness legs compress their timeout with the delay factor
    try:
        import yaml
        from flame.launch import task_timeout
        prof = yaml.safe_load(open(SCRIPT_DIR.parent / "_metadata" / "datasets.yaml", encoding="utf-8"))["datasets"][job.dataset] or {}
        return max(base, 1.25 * task_timeout.task_timeout_s(job.baseline, job.dataset, prof))
    except (KeyError, TypeError, OSError):
        return base


def estimate_s(job: Job, history: dict) -> float:
    past = history.get(job.key, [])
    if past:
        return float(statistics.median(past[-3:]))
    if job.mode == "grade":
        return 90.0
    join = job.n / 4 + 60 + 30  # join + startup/teardown + events
    return (job.runtime_s if job.mode == "real" else job.runtime_s / 1.6) + join


def cap_runtime(phases: List[Phase], s: int) -> None:
    """--scale-smoke S: every phase at most S seconds (SHORT_KEEP legs keep full length in build_jobs)."""
    for p in phases:
        p.full_runtime_s, p.runtime_s = p.runtime_s, min(p.runtime_s, s)


def build_jobs(phases: List[Phase], per_trainer: float, gpus_per_job: int, history: dict,
               gpu_per_trainer: float = 0.4) -> List[Job]:
    """gpu_per_trainer: CPUs per GPU-leg trainer (0.4 = the historical n=300 on 128 CPUs -> whole node)."""
    jobs: List[Job] = []
    for ph in phases:
        t = ph.trace
        for b in ph.baselines:
            at_shape = ph.harness != "none" and ph.runtime_s in {v["runtime_s"] for v in SHAPES.values() if v["trace"] == t}  # not smoke/T1
            rt = max(ph.runtime_s, MIN_RUNTIME_S.get((b, t, ph.dataset), MIN_RUNTIME_S.get((b, t), 0))) if at_shape else ph.runtime_s
            if ph.full_runtime_s and (b, t) in SHORT_KEEP and SHORT_KEEP[(b, t)] in (None, ph.harness):
                rt = max(rt, min(ph.full_runtime_s, SHORT_KEEP_S))
            if "G0U" in ph.pid:
                rt *= G0U_RUNTIME_X.get((b, t, ph.dataset), G0U_RUNTIME_X.get((b, t), 1))
            base = ["--harness", ph.harness, "--baselines", b, "--traces", t, "--runtime-s", str(rt),
                    "--num-trainers", str(ph.n), "--dataset", ph.dataset]
            agg_hp = ph.agg_hp.replace(G3_EVAL, f"evalEverySeconds={rt / EVAL_POINTS:g}")
            for flag, v in (("--trace-scale", ph.trace_scale), ("--trace-offset-s", ph.trace_offset_s), ("--agg-goal", ph.agg_goal),
                            ("--concurrency", ph.c), ("--agg-hp", agg_hp),
                            ("--trainer-hp", ph.trainer_hp), ("--inject-bug", ph.inject_bug),
                            ("--delay-factor", ph.delay_factor)):
                if v:
                    base += [flag, str(v)]
            if ph.sim_ceiling_x != 1:
                base += ["--sim-ceiling-x", str(ph.sim_ceiling_x)]
            gpu = ph.harness == "none"
            cpus = slot_cpus(ph.n, gpu_per_trainer, 8) if gpu else slot_cpus(ph.n, ph.cpt or per_trainer)
            mem = leg_ram_formula_gb(ph.dataset, gpu, ph.n)
            g = (ph.gpus or gpus_per_job) if gpu else 0
            stem = f"{ph.pid}_{b}" if ph.pid.endswith(t) else f"{ph.pid}_{t}_{b}"
            mk = lambda mode, args, deps=(), c=cpus, m=mem, gg=g: Job(
                f"{stem}_{mode}", ph.pid, t, b, mode, args, ph.n, rt, ph.harness, c, m, gg, deps,
                unit=stem, dataset=ph.dataset)
            if ph.kind == "pair":
                real, sim = mk("real", base + ["--mode", "real"]), mk("sim", base + ["--mode", "sim"], gg=ph.sim_gpus or g)
                jobs += [real, sim, mk("grade", base, (real.jid, sim.jid), 2, 1.0, 0)]
            elif ph.kind == "real":  # replicate for a real<->real control (R7)
                jobs.append(mk("real", base + ["--mode", "real"]))
            elif ph.kind == "sim":
                jobs.append(mk("sim", base + ["--mode", "sim", "--real-from", "auto"], gg=ph.sim_gpus or g))
            else:
                jobs.append(mk("sim", base + ["--mode", "sim"]))
    size_gpu_slots(jobs, history, gpu_per_trainer)
    if MEASURED:
        measured_lease(jobs, history)
    for j in jobs:
        j.est_s = estimate_s(j, history)
        j.mem_gb = mem_estimate_gb(j, history)
    return jobs


# FX-D59/FX-D141: host RAM per trainer and per-leg base (agg + broker), by (dataset, gpu); ~1.05 x ledger max (PR29b/PR30),
# less FX-N84's measured 0.13-0.15 GB/trainer (lazy mlflow, own-satellite coords)
TRAINER_RAM_GB = {("cifar10", True): 1.16, ("google_speech", True): 1.4, ("cifar10", False): 0.52,
                  ("google_speech", False): 0.92}
LEG_BASE_RAM_GB = {("cifar10", True): 8, ("google_speech", True): 8, ("cifar10", False): 3, ("google_speech", False): 6}


def leg_ram_formula_gb(dataset: str, gpu: bool, n: int) -> float:
    k = (dataset if dataset in DATASETS else "cifar10", gpu)
    return TRAINER_RAM_GB[k] * n + LEG_BASE_RAM_GB[k]


GPU_TIGHT = 0.85  # FX-D56: a leg peaking above this share of a GPU's memory is one burst from OOM


def node_gpu_peak_gb(run_dirs, gpus) -> float:
    """Peak used GB on any of `gpus` in the legs' resources.log (fine-grained; all residents of those GPUs)."""
    peak = 0.0
    for d in run_dirs:
        for f in Path(d).glob("*_resources.log"):
            for m in re.finditer(r"GPU(\d+): ([\d.]+)/([\d.]+)GB", f.read_text(errors="replace")):
                if int(m.group(1)) in gpus:
                    peak = max(peak, float(m.group(2)))
    return peak


def gpu_peak_note(run_dirs, gpus) -> str:
    """FX-D56/D59: ' gpu peak U/T GB' over the leg's GPUs and the node's ' ram peak' (resources.log); *_TIGHT above GPU_TIGHT."""
    peak = ram = None
    for d in run_dirs:
        for f in Path(d).glob("*_resources.log"):
            text = f.read_text(errors="replace")
            for m in re.finditer(r"GPU(\d+): ([\d.]+)/([\d.]+)GB", text):
                if int(m.group(1)) in gpus and (peak is None or float(m.group(2)) > peak[0]):
                    peak = (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"RAM: ([\d.]+)GB / ([\d.]+)GB", text):
                if ram is None or float(m.group(1)) > ram[0]:
                    ram = (float(m.group(1)), float(m.group(2)))
    note = ""
    if peak:
        note += f" gpu peak {peak[0]:.1f}/{peak[1]:.0f} GB" + (" GPU_TIGHT" if peak[0] > GPU_TIGHT * peak[1] else "")
    if ram:
        note += f" ram peak {ram[0]:.0f}/{ram[1]:.0f} GB" + (" RAM_TIGHT" if ram[0] > GPU_TIGHT * ram[1] else "")
    return note


CPU_HEADROOM = 1.5  # a GPU slot gets >= this x its baseline's measured cores p95
CPU_SAT_BUSY = 0.9  # a sample with the slot this busy is throttled


def cores_key(j: "Job") -> str:
    return "cores|" + j.key


def size_gpu_slots(jobs: List["Job"], history: dict, gpu_per_trainer: float) -> None:
    """A reduced CPU share stays per-baseline accurate: each GPU pair gets max(formula, 1.5 x its own measured cores
    p95), capped at the 0.4/trainer default; real and sim of a pair always get the same slot."""
    units: Dict[str, List[Job]] = {}
    for j in jobs:
        if j.gpus and j.mode != "grade":
            units.setdefault(j.unit, []).append(j)
    for legs in units.values():
        seen = [p for j in legs for p in history.get(cores_key(j), [])[-3:]]
        if not seen or gpu_per_trainer >= 0.4:
            continue
        cap = slot_cpus(legs[0].n, 0.4, 8)
        want = math.ceil(CPU_HEADROOM * max(seen))
        cpus = min(cap, max(legs[0].cpus, want + want % 2))
        for j in legs:
            j.cpus = cpus


# FX-N99 (`--lease-from-telemetry`, default off): measured leases, live admission, back-pressure, GPU priority across pools.
MEASURED = False
LEASE_X = 1.25  # a real CPU leg leases this x its measured cores p95 ...
LEASE_MIN_CORES = 2  # ... in whole physical cores, at least this many
ADMIT_FRAC = 0.7  # a leg starts only while smoothed node busy + fresh legs + its predicted p95 stay under this share of CPUs
SAT_PAUSE = 0.05  # a leg saturating its slot in more samples than this pauses starts and is timing-suspect
RUNQ_PAUSE = 1.0  # runnable threads per CPU that pause starts
MARGIN_X, MARGIN_MAX = 1.5, 4.0  # a saturated key's next lease margin


def margin_key(j: "Job") -> str:
    return "lease_x|" + j.key


def lease_margin(j: "Job", history: dict) -> float:
    return (history.get(margin_key(j)) or [1.0])[-1]


def cores_seen(j: "Job", history: dict) -> list:
    """The key's last 3 measured cores p95; else the same leg at any runtime (cores don't scale with run length)."""
    seen = history.get(cores_key(j), [])[-3:]
    if not seen:
        stem = cores_key(j).rsplit("|rt", 1)[0] + "|rt"
        seen = [p for k, v in history.items() if k.startswith(stem) for p in v[-3:]]
    return seen


def predicted_cores(j: "Job", history: dict) -> float:
    """A leg's expected busy cores: its key's measured p95 x margin, else its formula lease."""
    seen = cores_seen(j, history)
    return max(seen) * lease_margin(j, history) if seen else float(j.cpus)


SIM_LEASE_MIN_CORES = 4  # sims' wall time is their throughput: 2 cores slowed T1 sims 1.5-2.2x (FX-N99 A/B)


def measured_lease(jobs: List["Job"], history: dict) -> None:
    """FX-N99 1: a CPU leg leases ceil(LEASE_X x margin x measured p95) whole cores, at least LEASE_MIN_CORES (real) or
    SIM_LEASE_MIN_CORES (sim), never above the formula."""
    for j in jobs:
        seen = cores_seen(j, history)
        if j.mode in ("real", "sim") and not j.gpus and seen:
            floor = LEASE_MIN_CORES if j.mode == "real" else SIM_LEASE_MIN_CORES
            j.cpus = min(j.cpus, 2 * max(floor, math.ceil(LEASE_X * lease_margin(j, history) * max(seen) / 2)))


def shard(jobs: List[Job], i: int, n: int) -> List[Job]:
    """Units (a pair + its grade stay together) dealt longest-first to the least-loaded shard."""
    units: Dict[str, List[Job]] = {}
    for j in jobs:
        units.setdefault(j.unit, []).append(j)
    load, pick = [0.0] * n, []
    for u in sorted(units.values(), key=lambda js: (-sum(j.est_s for j in js), js[0].jid)):
        k = load.index(min(load))
        load[k] += sum(j.est_s for j in u)
        if k == i - 1:
            pick += u
    return [j for j in jobs if j in pick]


# ---------------------------------------------------------------- node resources
def core_groups() -> Dict[int, List[List[int]]]:
    """{numa node: [[cpu, sibling], ...]} over this process's affinity (whole physical cores)."""
    aff = set(os.sched_getaffinity(0))
    sysd = Path("/sys/devices/system")
    node_of = {}
    for nd in sorted(sysd.glob("node/node[0-9]*")):
        for c in _cpulist((nd / "cpulist").read_text()):
            node_of[c] = int(nd.name[4:])
    seen, out = set(), {}
    for c in sorted(aff):
        if c in seen:
            continue
        sib_f = sysd / f"cpu/cpu{c}/topology/thread_siblings_list"
        sib = [s for s in (_cpulist(sib_f.read_text()) if sib_f.exists() else [c]) if s in aff]
        seen |= set(sib)
        out.setdefault(node_of.get(c, 0), []).append(sorted(sib))
    return out


def _cpulist(text: str) -> List[int]:
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


class CoreMap:
    """Whole-physical-core allocator; a slot stays on one NUMA node when it fits (best fit)."""

    def __init__(self, groups: Dict[int, List[List[int]]], reserve_cores: int):
        self.free = {nd: list(gs) for nd, gs in groups.items()}
        for _ in range(reserve_cores):  # keep the lowest cores for the pool, checkers and the OS
            nd = max(self.free, key=lambda k: len(self.free[k]))
            self.free[nd].pop(0)

    def total(self) -> int:
        return sum(len(g) for gs in self.free.values() for g in gs)

    def alloc(self, cpus: int, avoid: frozenset = frozenset()) -> Optional[List[int]]:
        """`avoid`: logical CPUs another process keeps busy right now; their physical cores are skipped."""
        ok = {nd: [g for g in gs if not avoid.intersection(g)] for nd, gs in self.free.items()}
        fits = [nd for nd, gs in ok.items() if sum(len(g) for g in gs) >= cpus]
        order = sorted(fits, key=lambda nd: sum(len(g) for g in ok[nd])) or \
            sorted(ok, key=lambda nd: -sum(len(g) for g in ok[nd]))
        if sum(len(g) for gs in ok.values() for g in gs) < cpus:
            return None
        got: List[int] = []
        for nd in order:
            for g in list(ok[nd]):
                if len(got) >= cpus:
                    break
                self.free[nd].remove(g)
                got += g
            if len(got) >= cpus:
                break
        return sorted(got)

    def release(self, cpus: List[int], groups: Dict[int, List[List[int]]]) -> None:
        cs = set(cpus)
        for nd, gs in groups.items():
            for g in gs:
                if set(g) <= cs:
                    self.free[nd].append(g)
            self.free[nd].sort()


def mem_available_gb(key: str = "MemAvailable") -> float:
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1e6
    return 0.0


def ram_blocks(mem_gb: float, available_gb: float, headroom_gb: float, busy: bool) -> bool:
    """FX-D131: a leg waits for free RAM even in an idle pool; only one larger than the node starts unchecked."""
    too_big = mem_gb > mem_available_gb("MemTotal") - headroom_gb
    return (busy or not too_big) and mem_gb > available_gb - headroom_gb


GPU_ALLOW: Optional[set] = None  # --gpu-ids; None = every healthy GPU


def _gpu_query() -> List[tuple]:
    """(index, uncorrected volatile ECC count, MiB used) per GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        idx, ecc, used = (x.strip() for x in line.split(","))
        rows.append((int(idx), int(ecc) if ecc.isdigit() else 0, float(used) if used.replace(".", "").isdigit() else 0.0))
    return rows


def gpu_ids(healthy_only: bool = True) -> List[int]:
    """Node GPU ordinals; by default without a volatile uncorrected ECC error (the runner's health
    check refuses those, e.g. jayne GPU 1 on 2026-09-26) and within GPU_ALLOW."""
    return [i for i, ecc, _ in _gpu_query()
            if not healthy_only or (ecc == 0 and (GPU_ALLOW is None or i in GPU_ALLOW))]


def set_gpu_allow(explicit: str = "") -> None:
    global GPU_ALLOW
    GPU_ALLOW = {int(x) for x in explicit.split(",") if x.strip()} if explicit else None


def busy_gpus(busy_mib: float = 1024) -> set:
    """GPUs holding >= busy_mib now; the pool only asks about GPUs none of its legs hold, so this is foreign load."""
    return {i for i, _, used in _gpu_query() if used >= busy_mib}


def _cpu_times() -> Dict[int, tuple]:
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:]))
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])  # (total, idle + iowait)
    return out


def busy_cpus(threshold: float = 0.5, window_s: float = 1.0) -> frozenset:
    """Logical CPUs above `threshold` utilisation over a short window (the caller drops its own slots' CPUs)."""
    a = _cpu_times()
    time.sleep(window_s)
    b = _cpu_times()
    return frozenset(c for c in b if c in a and (b[c][0] - a[c][0]) > 0
                     and 1 - (b[c][1] - a[c][1]) / (b[c][0] - a[c][0]) > threshold)


class NodeLoad:
    """FX-N99 2: node-wide busy cores (/proc/stat, smoothed over pool ticks) and runnable threads per CPU."""

    def __init__(self, alpha: float = 0.3):
        self.alpha, self.ema, self.last, self.runq, self.ncpu = alpha, None, None, 0.0, os.cpu_count() or 1

    def tick(self) -> None:
        a = _cpu_times()
        tot, idle = sum(v[0] for v in a.values()), sum(v[1] for v in a.values())
        if self.last and tot > self.last[0]:
            busy = len(a) * (1 - (idle - self.last[1]) / (tot - self.last[0]))
            self.ema = busy if self.ema is None else self.alpha * busy + (1 - self.alpha) * self.ema
        self.last, self.ncpu = (tot, idle), len(a) or self.ncpu
        for line in open("/proc/stat"):
            if line.startswith("procs_running"):
                self.runq = (int(line.split()[1]) - 1) / self.ncpu  # less this pool's own sampler


def foreign_owners(own_pgids: set, top: int = 6) -> List[str]:
    """PL-LOG: per foreign owner (user + run label or script), the GPUs and CPUs (1 s sample) it holds, busiest first."""
    import psutil
    try:
        uuid_idx = dict(reversed(l.split(", ")) for l in subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout.splitlines() if ", " in l)
        apps = [l.split(", ") for l in subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout.splitlines() if ", " in l]
    except (OSError, subprocess.SubprocessError):
        uuid_idx, apps = {}, []
    gpu_of = {int(pid): int(uuid_idx[u]) for pid, u in apps if u in uuid_idx}
    me, procs = os.getpid(), []
    for pr in psutil.process_iter(["pid", "username", "cmdline"]):
        try:
            if os.getpgid(pr.pid) in own_pgids or pr.pid == me:
                continue
            pr.cpu_percent(None)
            procs.append(pr)
        except (psutil.Error, OSError):
            pass
    time.sleep(1.0)
    owners: Dict[str, list] = {}
    for pr in procs:
        try:
            cpu = pr.cpu_percent(None) / 100
            if cpu < 0.05 and pr.pid not in gpu_of:
                continue
            cmd = pr.info["cmdline"] or []
            script = next((os.path.basename(c) for c in cmd if c.endswith((".py", ".sh"))), os.path.basename(cmd[0]) if cmd else "?")
            label = pr.environ().get("FLAME_RUN_LABEL", script) if pr.info["username"] == os.environ.get("USER") else script
            o = owners.setdefault(f"{pr.info['username']} {label}", [set(), 0.0])
            if pr.pid in gpu_of:
                o[0].add(gpu_of[pr.pid])
            o[1] += cpu
        except (psutil.Error, OSError):
            pass
    rows = sorted(((len(g), c, k, g) for k, (g, c) in owners.items() if g or c >= 1), reverse=True)[:top]
    return [f"{k}: {len(g)} GPUs {sorted(g)}, {c:.0f} CPUs" if g else f"{k}: {c:.0f} CPUs" for _, c, k, g in rows]


class Leases:
    """L28: node-wide flock leases on ports, cores and GPUs across pools; the kernel drops a dead pool's leases."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or Path(f"/tmp/flame_pool_leases_{os.getuid()}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.held: Dict[str, int] = {}

    def take(self, name: str) -> bool:
        if name in self.held:
            return False
        fd = os.open(self.root / name, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.held[name] = fd
        return True

    def take_all(self, names: List[str]) -> bool:
        got = []
        for n in names:
            if not self.take(n):
                self.drop(got)
                return False
            got.append(n)
        return True

    def drop(self, names) -> None:
        for n in names:
            fd = self.held.pop(n, None)
            if fd is not None:
                os.close(fd)

    def foreign(self, names) -> set:
        """Names another process holds now."""
        out = set()
        for n in names:
            if n in self.held:
                continue
            if self.take(n):
                self.drop([n])
            else:
                out.add(n)
        return out


    def reserve_ram(self, name: str, gb: float) -> None:
        """FX-D136: a running leg's RAM budget, visible to every pool (a leg's footprint ramps, so free RAM at its start lies)."""
        if self.take(name):
            os.ftruncate(self.held[name], 0)
            os.write(self.held[name], f"{gb:.1f}".encode())

    def put(self, name: str, value) -> None:
        """A live lease carrying `value` (a number or text; taken on first use, rewritten after)."""
        if name in self.held or self.take(name):
            os.ftruncate(self.held[name], 0)
            os.pwrite(self.held[name], (value if isinstance(value, str) else f"{value:g}").encode(), 0)

    def reserved_gb(self) -> float:
        """Sum of live RAM reservations across pools (a lockable file is a dead pool's)."""
        tot = 0.0
        for f in self.root.glob("ram_*"):
            fd = os.open(f, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                try:
                    tot += float(os.pread(fd, 32, 0) or 0)
                except ValueError:
                    pass
            os.close(fd)
        return tot


def _lease_texts(root: Path, pattern: str, held: Optional[Dict[str, int]] = None) -> Dict[str, str]:
    """{name: content} of live (locked) lease files matching `pattern`; `held` are this process's own (live)."""
    out = {}
    for f in root.glob(pattern):
        if held is not None and f.name in held:
            out[f.name] = os.pread(held[f.name], 64, 0).decode(errors="replace")
            continue
        fd = os.open(f, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            out[f.name] = os.pread(fd, 64, 0).decode(errors="replace")
        os.close(fd)
    return out


def _lease_values(root: Path, pattern: str) -> Dict[str, float]:
    """{name: value} of live (locked) lease files matching `pattern`."""
    out = {}
    for k, v in _lease_texts(root, pattern).items():
        try:
            out[k] = float(v or 0)
        except ValueError:
            pass
    return out


# FX-N100 2 (`--gpu-share K`, default 1 = exclusive GPUs, byte-identical): co-locate legs on a GPU by measured footprint.
GPU_SHARE = 1
SHORT_LEGS = False  # --scale-smoke legs record no GPU footprint: they read well below the full-length peak
GPU_GB = 44.4  # A40 usable GiB (torch OOM message, E2)
GPU_MEM_CAP, GPU_SM_CAP = 0.9, 90.0  # committed peak memory share and SM % per GPU
GPU_LIVE_CAP = 0.9  # live used memory + the newcomer's peak (operator 10-15: no resource past 90%)


def gpu_footprint(j: "Job", history: dict) -> tuple:
    """(peak GB, mean SM %) per held GPU: max over the key's last 6 legs, real and sim alike (same trainers; a short sim's
    30 s samples missed a 28 GB peak, E2 OOM); unmeasured legs claim a whole GPU."""
    h = history.get(gpu_key(j), [])[-6:]
    if not h:
        return GPU_MEM_CAP * GPU_GB, GPU_SM_CAP
    return min(GPU_MEM_CAP * GPU_GB, max(x[0] for x in h)), min(GPU_SM_CAP, max(x[1] for x in h))


AGG_GPU_GB = 3.0  # the aggregator's eval model + context, on the leg's last GPU (runner `_agg_gpu`)
MIN_TRAINERS_PER_GPU = 5  # smaller shards fragment GPUs for little gain
TRAINERS_PER_GPU = {"cifar10": 75, "google_speech": 25}  # FX-T36: an A40's trainer cap; a trainer holds 1/cap of a GPU
TRAINER_GPU_GB_MIN = {"cifar10": 0.4, "google_speech": 0.8}  # exclusive-leg minima (0.31 GB is bare CUDA context)


def _gpu_commits(gpus: List[int], leases: "Leases") -> Dict[int, list]:
    """{gpu: [committed GB, SM %, {slot indices}, trainer share]} over every pool's live `gslot_<g>_<i>` leases."""
    load = {g: [0.0, 0.0, set(), 0.0] for g in gpus}
    for name, txt in _lease_texts(leases.root, "gslot_*", leases.held).items():
        _, g, i = name.split("_")
        if int(g) in load:
            try:
                gb, sm, share = (list(map(float, txt.split(","))) + [0.0])[:3]
            except ValueError:
                gb, sm, share = GPU_MEM_CAP * GPU_GB, GPU_SM_CAP, 1.0  # half-written: assume full
            load[int(g)][0] += gb
            load[int(g)][1] += sm
            load[int(g)][2].add(int(i))
            load[int(g)][3] += share
    return load


def place_trainers(n: int, per_trainer: tuple, gpus: List[int], leases: "Leases",
                   used_gb: Dict[int, float], cap: int = 75) -> Optional[List[tuple]]:
    """FX-N100 2: spread a leg's n trainers over GPUs with room, [(gpu, trainers, slot lease, GB, SM, share)], aggregator GPU
    last; at most `cap` trainers per GPU across legs (share k/cap, FX-T36). Fills partly used GPUs first; all or nothing."""
    gb_t, sm_t = per_trainer
    load = _gpu_commits(gpus, leases)
    excl = leases.foreign([f"gpu{g}" for g in gpus])
    room = {}
    for g in gpus:
        if f"gpu{g}" in excl or len(load[g][2]) >= GPU_SHARE:
            continue
        gb = min(GPU_MEM_CAP * GPU_GB - load[g][0], GPU_LIVE_CAP * GPU_GB - used_gb.get(g, 0.0))
        fit = min(int(gb // max(gb_t, 1e-6)), int((GPU_SM_CAP - load[g][1]) // max(sm_t, 1e-6)) if sm_t > 0 else n,
                  int((1.0 - load[g][3]) * cap + 1e-6))
        if fit >= min(n, MIN_TRAINERS_PER_GPU):
            room[g] = fit
    picked, left = [], n
    for g in sorted(room, key=lambda g: (-load[g][0], g)):  # fill partly used GPUs first
        take = min(room[g], left)
        if left and take >= min(left, MIN_TRAINERS_PER_GPU):
            picked.append((g, take))
            left -= take
    if left:
        return None
    need, spare = math.ceil(AGG_GPU_GB / max(gb_t, 1e-6)), {g: room[g] - k for g, k in picked}
    agg = max(spare, key=lambda g: (spare[g], -g))
    if spare[agg] < need:  # no picked GPU has room for the aggregator: one more GPU hosts it alone
        agg = next((g for g in sorted(room, key=lambda g: (-load[g][0], g)) if g not in spare and room[g] >= need), None)
        if agg is None:
            return None
        picked.append((agg, 0))
    picked.sort(key=lambda gk: gk[0] == agg)  # aggregator GPU last (runner `_agg_gpu = gpu_ids[-1]`)
    out = []
    for g, k in picked:
        slot = f"gslot_{g}_{min(set(range(GPU_SHARE)) - load[g][2])}"
        out.append((g, k, slot, round(k * gb_t + (AGG_GPU_GB if g == agg else 0.0), 1), round(k * sm_t, 1), round(k / cap, 3)))
    return out


def per_trainer_footprint(j: "Job", history: dict) -> Optional[tuple]:
    """(GB, SM %) per trainer: the measured per-GPU peak over its even split ceil(n/G), at least TRAINER_GPU_GB_MIN;
    None if unmeasured."""
    if not history.get(gpu_key(j)):
        return None
    fp, per = gpu_footprint(j, history), -(-j.n // max(1, j.gpus))
    return max(fp[0] / per, TRAINER_GPU_GB_MIN.get(j.dataset, 0.4)), fp[1] / per


def place_gpus(m: int, fp: tuple, gpus: List[int], leases: "Leases", used_gb: Dict[int, float],
               share: float = 0.0) -> Optional[List[tuple]]:
    """FX-N100 2: [(gpu, slot lease)] on m GPUs whose live slot leases leave room for fp and `share` of the trainer cap;
    least committed SM first."""
    load = _gpu_commits(gpus, leases)
    excl = leases.foreign([f"gpu{g}" for g in gpus])  # an exclusive pool's GPU
    ok = [g for g in gpus if f"gpu{g}" not in excl and len(load[g][2]) < GPU_SHARE
          and load[g][0] + fp[0] <= GPU_MEM_CAP * GPU_GB + 1e-6 and load[g][1] + fp[1] <= GPU_SM_CAP + 1e-6
          and load[g][3] + share <= 1.0 + 1e-6 and used_gb.get(g, 0.0) + fp[0] <= GPU_LIVE_CAP * GPU_GB]
    if len(ok) < m:
        return None
    ok.sort(key=lambda g: (load[g][1], load[g][0], g))
    return [(g, f"gslot_{g}_{min(set(range(GPU_SHARE)) - load[g][2])}") for g in ok[:m]]


def gpus_wanted_above(leases: "Leases", priority: int) -> int:
    """FX-N99 4: most GPUs a higher-priority pool's blocked leg waits for (`want_p<N>_<pid>`, N < priority)."""
    vals = [v for k, v in _lease_values(leases.root, "want_p*").items() if int(k.split("_")[1][1:]) < priority]
    return int(max(vals, default=0))


def exclusive_gpus(m: int, usable: List[int], leases: "Leases") -> Optional[List[int]]:
    """FX-D192: m whole GPUs from `usable` carrying no live `gslot_*` lease of any pool (call under gpu_place.lock)."""
    held = {g for g, v in _gpu_commits(usable, leases).items() if v[2]}
    free = [g for g in usable if g not in held]
    return free[:m] if len(free) >= m else None


def slot_leases(cpus: List[int], gpus: List[int], port: Optional[int] = None) -> List[str]:
    return [f"cpu{c}" for c in cpus] + [f"gpu{g}" for g in gpus] + ([f"port{port}"] if port else [])


def free_port(start: int, used: set, leases: Optional[Leases] = None) -> int:
    p = start
    while True:
        if p not in used and (leases is None or leases.take(f"port{p}")):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    if leases is not None:
                        leases.drop([f"port{p}"])
        p += 1


def tag_pids(tag: str) -> List[int]:
    """Every process of this user carrying FLAME_RUN_TAG=<tag>."""
    out, uid, needle = [], os.getuid(), f"FLAME_RUN_TAG={tag}".encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            if needle in (d / "environ").read_bytes().split(b"\0"):
                out.append(int(d.name))
        except OSError:
            continue
    return out


def tag_cpu_s(tag: str) -> Dict[int, float]:
    """{pid: user+sys CPU seconds} for this tag's live processes."""
    tick = os.sysconf("SC_CLK_TCK")
    out = {}
    for pid in tag_pids(tag):
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            out[pid] = (int(f[11]) + int(f[12])) / tick
        except (OSError, IndexError, ValueError):
            continue
    return out


def kill_tag(tag: str, sig=signal.SIGKILL) -> None:
    for pid in tag_pids(tag):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


# ---------------------------------------------------------------- resource ledger (FX-D140)
BLOCKED_LOG_S = 600  # FX-D141
RES_SAMPLE_TICKS = 15  # loop ticks (2 s) between ledger samples: 30 s
RAM_PROFILE_X = 1.15   # a leg's next RAM estimate >= this x its measured peak


def overrun_share(run_dirs) -> tuple:
    """FX-N100 / PL19: (overran, train tasks) over a leg's trainers: compute past D, the GPU-sharing interference signal."""
    n = o = 0
    for d in run_dirs:
        for f in Path(d).glob("telemetry/trainer_*.jsonl"):
            for line in open(f, errors="replace"):
                if '"trainer_round"' in line and '"overran"' in line:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("task_to_perform", "train") == "train":
                        n += 1
                        o += bool(e.get("overran"))
    return o, n


def gpu_key(j: "Job") -> str:
    """FX-N100: history of [peak GB, mean SM %] per held GPU; memory depends on the trainers, not mode, trace or runtime."""
    return f"gpu|{j.dataset}|{j.harness}|{j.baseline}|n{j.n}|g{j.gpus}"


GRADE_RAM_GB = 8.0  # parity_check + event checker on two legs' telemetry


def ram_key(j: "Job") -> str:
    return "ram|" + j.key


def mem_estimate_gb(j: "Job", history: dict) -> float:
    """FX-D140: the formula, raised to the measured peak of this leg kind's last runs (never lowered)."""
    if j.mode == "grade":  # FX-D187: a checker, not a leg; MEASURED peers below gave it the leg's 90 GB
        return GRADE_RAM_GB
    past = history.get(ram_key(j), [])
    if MEASURED:  # FX-N100: the measured peak sizes the reservation both ways; a short sim's 30 s samples miss peaks
        peers = [max(v[-3:]) for k, v in history.items() if k.startswith(f"ram|{j.dataset}|{j.harness}|")
                 and k.split("|")[5] == j.baseline and k.split("|")[6] == f"n{j.n}" and v]
        if peers:
            return RAM_PROFILE_X * max(peers)
    return max(j.mem_gb, RAM_PROFILE_X * max(past[-3:])) if past else j.mem_gb


def pids_by_tag(tags) -> Dict[str, List[int]]:
    """One /proc scan: {tag: pids carrying FLAME_RUN_TAG=<tag>}."""
    want = {f"FLAME_RUN_TAG={t}".encode(): t for t in tags}
    out: Dict[str, List[int]] = {t: [] for t in tags}
    uid = os.getuid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            for kv in (d / "environ").read_bytes().split(b"\0"):
                if kv in want:
                    out[want[kv]].append(int(d.name))
                    break
        except OSError:
            continue
    return out


def proc_role(pid: int) -> str:
    try:
        head = b" ".join(Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:3])  # program + script, not the inline config
    except OSError:
        return "other"
    return next((r for k, r in ((b"mosquitto", "broker"), (b"/trainer/", "trainer"), (b"/aggregator/", "agg")) if k in head), "other")


def proc_mem_gb(pid: int) -> float:
    """PSS (shared pages split, so a sum over processes is honest); RSS if smaps_rollup is unreadable."""
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith("Pss:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        return int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, IndexError, ValueError):
        return 0.0


def gpu_mib_by_pid() -> Dict[int, float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    res = {}
    for line in out.splitlines():
        try:
            pid, mib = (x.strip() for x in line.split(","))
            res[int(pid)] = res.get(int(pid), 0.0) + float(mib)
        except ValueError:
            continue
    return res


def gpu_sm_pct() -> Dict[int, float]:
    """FX-N100: SM utilisation % per GPU (nvidia-smi's last sample period)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    res = {}
    for line in out.splitlines():
        try:
            i, u = (x.strip() for x in line.split(","))
            res[int(i)] = float(u)
        except ValueError:
            continue
    return res


def gpu_sm_by_pid() -> Dict[int, float]:
    """FX-N100: SM % per process, summed over GPUs (`nvidia-smi pmon`, one sample)."""
    try:
        out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    res = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) > 3 and not line.startswith("#") and f[1].isdigit() and f[3].replace(".", "").isdigit():
            res[int(f[1])] = res.get(int(f[1]), 0.0) + float(f[3])
    return res


def ram_growth_gb(series: List[tuple]) -> float:
    """Least-squares RAM growth over a leg's second half (GB); a steady leg reads ~0, a leak keeps climbing.
    (t, GB, procs) samples: the window starts once procs peaks, so a short sim's spawn ramp is not a leak."""
    peak = max((s[2] for s in series if len(s) > 2), default=None)
    start = next((i for i, s in enumerate(series) if len(s) > 2 and s[2] >= peak), 0) if peak else 0
    half = series[max(len(series) // 2, start):]
    if len(half) < 4:
        return 0.0
    ts, ys = [s[0] for s in half], [s[1] for s in half]
    mt, my = sum(ts) / len(ts), sum(ys) / len(ys)
    var = sum((t - mt) ** 2 for t in ts)
    return 0.0 if var <= 0 else sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / var * (ts[-1] - ts[0])


def leak_suspect(peak_gb: float, growth_gb: float) -> bool:
    return growth_gb > max(1.0, 0.1 * peak_gb)


class ResourceLedger:
    """FX-D140: `<pool>/resources.jsonl` — leg start/end events and 30 s samples of node RAM/CPU/GPU and each running
    leg's PSS by role and GPU memory; `resource_report.py` reads it (profiles, leak suspects, packing)."""

    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)
        self.load = NodeLoad(alpha=1.0)  # FX-N100: busy cores over each sample interval

    def event(self, ev: str, **kw) -> None:
        self.f.write(json.dumps({"t": round(time.time(), 1), "ev": ev, **kw}) + "\n")

    def sample(self, running: Dict[str, "Running"], leases: "Leases") -> None:
        by_tag = pids_by_tag([r.tag for r in running.values()])
        gpu, sm = gpu_mib_by_pid(), gpu_sm_by_pid()
        self.load.tick()
        legs = {}
        for jid, r in running.items():
            roles: Dict[str, float] = {}
            pids = by_tag.get(r.tag, [])
            for pid in pids:
                role = proc_role(pid)
                roles[role] = roles.get(role, 0.0) + proc_mem_gb(pid)
            r.ram.append((time.time(), sum(roles.values()), len(pids)))
            gpu_gb = sum(gpu.get(pid, 0.0) for pid in pids) / 1024
            r.gpu_peak_gb = max(r.gpu_peak_gb, gpu_gb)
            leg_sm = sum(sm.get(pid, 0.0) for pid in pids)
            if r.gpus:
                r.gpu_sm.append(leg_sm / len(r.gpus))
            legs[jid] = {"ram_gb": {k: round(v, 2) for k, v in roles.items()}, "gpu_gb": round(gpu_gb, 2),
                         "gpu_sm": round(leg_sm, 1), "gpus": r.gpus,
                         "cores": round(r.busy[-1], 2) if r.busy else 0.0, "procs": len(pids)}
        node = {"mem_avail_gb": round(mem_available_gb(), 1), "reserved_gb": round(leases.reserved_gb(), 1),
                "load1": float(open("/proc/loadavg").read().split()[0]),
                "gpu_used_gb": {i: round(m / 1024, 1) for i, _, m in _gpu_query()},
                "gpu_sm": gpu_sm_pct(), "cpu_busy": round(self.load.ema or 0.0, 1), "ncpu": self.load.ncpu,
                "ram_used_gb": round(mem_available_gb("MemTotal") - mem_available_gb(), 1)}
        self.event("sample", node=node, legs=legs)


# ---------------------------------------------------------------- run
@dataclass
class Running:
    job: Job
    proc: subprocess.Popen
    cpus: List[int]
    gpus: List[int]
    port: int
    tag: str
    t0: float
    out: Path
    cpu: Dict[int, float] = field(default_factory=dict)  # pid -> last seen CPU seconds
    busy: List[float] = field(default_factory=list)      # cores busy per sample interval
    last: tuple = (0.0, 0.0)                              # (time, total CPU s) at the last sample
    watch: Optional[StallWatch] = None                    # FX-D45; None for grade jobs or --stall-min 0
    stalled: str = ""
    stalled_at: float = 0.0
    doom_at: float = 0.0  # last early-exit check
    ram: List[tuple] = field(default_factory=list)  # (time, GB PSS) per ledger sample (FX-D140)
    gpu_peak_gb: float = 0.0
    saturated: bool = False  # FX-N99 3: its slot saturated (back-pressure logged)
    gpu_sm: List[float] = field(default_factory=list)  # FX-N100: SM % per held GPU, per ledger sample
    gslots: List[str] = field(default_factory=list)  # FX-N100: shared-GPU slot leases

    def sample(self) -> None:
        self.cpu.update(tag_cpu_s(self.tag))
        now, tot = time.time(), sum(self.cpu.values())
        if self.last[0]:
            self.busy.append((tot - self.last[1]) / max(1e-6, now - self.last[0]))
        self.last = (now, tot)


@dataclass
class Pool:
    root: Path
    jobs: List[Job]
    max_parallel: int
    reserve_cores: int
    mem_headroom_gb: float
    deadline_s: float
    dry: bool
    log: object = None
    done: Dict[str, Path] = field(default_factory=dict)
    bad_gpus: List[int] = field(default_factory=list)
    fail_fast: bool = True
    aborted: bool = False
    scanner: Scanner = field(default_factory=Scanner)
    label: str = ""  # progress prefix, e.g. the ladder rung
    leases: Optional["Leases"] = None
    stall_min: float = 15.0  # FX-D45 S1 limit; 0 disables the stall rules
    doom_min: float = 5.0  # early-exit check period; 0 disables it
    priority: int = -1  # FX-N99 4: GPU priority among co-running pools (lower first); -1 = off
    queue: Optional[Path] = None  # FX-D186: dispatch.py releases units here (one per line; END / ABORT)
    unassigned: set = field(default_factory=set)

    def _stalled(self, r: "Running") -> bool:
        """FX-D45: a leg that stopped progressing is killed alone (STALLED.txt); its neighbours keep running."""
        if r.watch is None or r.stalled:
            return False
        dirs = leg_run_dirs(r.out)
        rule = r.watch.check(dirs, time.time())
        if not rule:
            return False
        r.stalled, r.stalled_at = rule, time.time()
        write_stalled(r.out / "STALLED.txt", r.job.jid, rule, dirs)
        self.say(f"STALLED {r.job.jid}: {rule} -- killing this leg only ({r.out}/STALLED.txt)")
        try:
            os.killpg(r.proc.pid, signal.SIGTERM)
        except OSError:
            pass
        return True

    def _doomed(self, r: "Running", running: Dict[str, "Running"]) -> None:
        """Kill a leg that already FAILs a prefix-safe EV check, and its pair partner (same unit); exempt: expected-FAIL phases."""
        now = time.time()
        if (r.watch is None or r.stalled or self.doom_min <= 0 or now - r.doom_at < self.doom_min * 60
                or re.match(r"(gs_)?(P11|P4$)", r.job.phase)):
            return
        r.doom_at = now
        import event_invariants
        dirs = leg_run_dirs(r.out)
        fails = [f for d in dirs for f in event_invariants.doomed(d)]
        if not fails:
            return
        for x in [r] + [o for o in running.values() if o is not r and o.job.unit == r.job.unit and o.job.mode != "grade"]:
            why = f"DOOMED {'; '.join(fails)[:300]}" if x is r else f"DOOMED partner {r.job.jid}"
            x.stalled, x.stalled_at = why, now
            write_stalled(x.out / "DOOMED.txt", x.job.jid, why, leg_run_dirs(x.out))
            with open(self.root / "DOOMED.txt", "a") as f:
                f.write(f"{x.job.jid}: {why}\n")
            self.say(f"{why} -- killing {x.job.jid} ({x.out}/DOOMED.txt)")
            try:
                os.killpg(x.proc.pid, signal.SIGTERM)
            except OSError:
                pass

    def _fatal(self, r: "Running") -> bool:
        """FX-N40: a fatal line in this leg's logs aborts the pool (ABORT.txt names it)."""
        if r.stalled:  # FX-N75: post-kill output is our own teardown
            return False
        found = (self.scanner.scan(leg_run_dirs(r.out)) + self.scanner.scan_broker(r.out / "mosquitto.log")
                 + self.scanner.scan_launch(r.out))
        if not found or not self.fail_fast:
            return False
        write_abort(self.root / "ABORT.txt", r.job.jid, found)
        self.say(f"FAIL-FAST {r.job.jid}: {found[0].path.name}:{found[0].lineno}: {found[0].line.strip()[:120]} "
                 f"-- {self.root}/ABORT.txt")
        self.aborted = True
        return True

    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")
            self.log.flush()

    def progress(self, pending, running, cpus: int) -> None:
        """Terminal-only progress: done/total, running legs vs estimate, ETA, deadline cut (nothing persisted)."""
        now, total = time.time(), len(self.jobs)
        done = total - len(pending) - len(running)
        left = [replace(j, deps=tuple(d for d in j.deps if d not in self.done)) for j in pending]
        left += [replace(r.job, deps=(), est_s=max(30.0, r.job.est_s - (now - r.t0))) for r in running.values()]
        eta = simulate_makespan(left, cpus, self.max_parallel, len(gpu_ids())) if left else 0.0
        bar = "#" * round(20 * done / max(1, total))
        runs = ", ".join(f"{r.job.jid} {(now - r.t0) / 60:.0f}/{r.job.est_s / 60:.0f}m" for r in running.values())
        hm = lambda t: time.strftime("%H:%M", time.localtime(t))
        if math.isinf(eta):
            print(f"[{time.strftime('%F %T')}] PROGRESS {done}/{total} done | running: {runs or '-'} | ETA ? (pending legs "
                  f"cannot be placed: {len(gpu_ids())} healthy GPUs)", flush=True)
            return
        msg = (f"PROGRESS {self.label + ' ' if self.label else ''}[{bar:<20}] {done}/{total} done | running: {runs or '-'} "
               f"| elapsed {(now - self.t0) / 60:.0f}m, left ~{eta / 60:.0f}m, ETA {hm(now + eta)}")
        if now + eta > self.t0 + self.deadline_s:
            msg += f" | past deadline {hm(self.t0 + self.deadline_s)}: later starts SKIPPED"
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    def plan(self, groups) -> None:
        cm = CoreMap(groups, self.reserve_cores)
        self.say(f"node: {cm.total()} cpus for slots (reserve {self.reserve_cores} cores), "
                 f"{mem_available_gb():.0f} GB available, gpus {gpu_ids() or '-'}")
        tot = sum(j.est_s for j in self.jobs)
        self.say(f"{len(self.jobs)} jobs, {tot / 60:.0f} slot-min total")
        for j in sorted(self.jobs, key=lambda j: -j.est_s):
            self.say(f"  {j.jid:<44} cpus={j.cpus:<3} mem={j.mem_gb:<5.0f} est={j.est_s / 60:5.1f}m deps={list(j.deps)}")
        span = simulate_makespan(self.jobs, cm.total(), self.max_parallel, len(gpu_ids()))
        self.say(f"estimated makespan ~{span / 60:.0f} min")
        if span > self.deadline_s:
            self.say(f"WARNING: makespan ~{span / 3600:.1f}h > --deadline-h {self.deadline_s / 3600:g}: jobs not started by "
                     f"{time.strftime('%H:%M', time.localtime(time.time() + self.deadline_s))} are SKIPPED")

    def run(self, groups) -> int:
        cm = CoreMap(groups, self.reserve_cores)
        cap = cm.total()
        leases = self.leases or Leases()
        node = slot_leases([c for gs in groups.values() for g in gs for c in g], gpu_ids())
        node_gpus = gpu_ids()
        gpus_free = list(node_gpus)
        gpu_seen_t = time.time()
        blocked: Dict[str, str] = {}  # FX-D141: why each ready leg is waiting; logged every BLOCKED_LOG_S
        blocked_shown = time.time()
        pending = sorted(self.jobs, key=_order_key(self.jobs))
        running: Dict[str, Running] = {}
        ports: set = set()
        history = _load_history()
        history_base = {k: list(v) for k, v in history.items()}
        t0 = self.t0 = time.time()
        stop = {"flag": False}
        probe, last_note, owners_shown = None, None, 0.0
        tick, shown = 0, 0.0
        load, paused_until = NodeLoad(), 0.0  # FX-N99 2-3
        want_lease = f"want_p{self.priority:03d}_{os.getpid()}"
        jobs_tsv = open(self.root / "jobs.tsv", "a")
        jobs_tsv.write("jid\trc\tdur_s\test_s\tcpus\tcores_avg\tcores_p95\tcpu_sat\n")
        ledger = ResourceLedger(self.root / "resources.jsonl")

        def _on_signal(signum, _frm):
            stop["flag"] = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        try:
            released, queue_t = set(), 0.0
            while (pending or running) and not stop["flag"] and not self.aborted:
                if self.queue is not None and time.time() - queue_t > 10:  # FX-D186: units the dispatcher assigned here
                    queue_t = time.time()
                    lines = set(self.queue.read_text().split()) if self.queue.exists() else set()
                    released = lines - {"END", "ABORT"}
                    if "ABORT" in lines:
                        self.say("ABORT from the dispatcher queue")
                        stop["flag"] = True
                        break
                    if "END" in lines:
                        for j in [j for j in pending if (j.unit or j.jid) not in released]:
                            pending.remove(j)
                            self.unassigned.add(j.jid)
                    self._write_status(pending, running, released, blocked)
                if time.time() - gpu_seen_t > 30 and any(j.gpus for j in pending):
                    # FX-D141: refresh; a failed nvidia-smi returns [] and a start-time snapshot starved PR29b B1
                    node_gpus = gpu_ids() or node_gpus
                    ours = {g for r in running.values() for g in r.gpus}
                    gpus_free = [g for g in node_gpus if g not in ours]
                    gpu_seen_t = time.time()
                if blocked and time.time() - blocked_shown > BLOCKED_LOG_S:
                    for jid, why in list(blocked.items())[:6]:
                        self.say(f"BLOCKED {jid}: {why}")
                    blocked_shown = time.time()
                if last_note and (last_note[0] or last_note[1]) and time.time() - owners_shown > BLOCKED_LOG_S:
                    own = {os.getpgid(os.getpid())} | {r.proc.pid for r in running.values()}
                    rows = foreign_owners(own)  # PL-LOG: whom to contact, every 10 min while foreign load exists
                    if rows:
                        self.say("LOAD foreign owners: " + "; ".join(rows))
                    owners_shown = time.time()
                blocked.clear()
                gpu_wait = 0
                if MEASURED:
                    load.tick()
                    if load.runq > RUNQ_PAUSE:
                        paused_until = max(paused_until, time.time() + 30)
                for j in list(pending):
                    if self.queue is not None and (j.unit or j.jid) not in released:
                        continue
                    if any(d not in self.done for d in j.deps):
                        continue
                    if time.time() - t0 > self.deadline_s:
                        self.say(f"SKIP {j.jid}: past the deadline")
                        pending.remove(j)
                        self.done[j.jid] = None
                        continue
                    if len(running) >= self.max_parallel:
                        break
                    if MEASURED and time.time() < paused_until:
                        blocked[j.jid] = f"back-pressure (slot saturated or run queue {load.runq:.2f}/CPU)"
                        break
                    share = GPU_SHARE > 1 and j.gpus > 0
                    if not share and j.gpus > len(gpus_free):
                        blocked[j.jid] = f"needs {j.gpus} GPUs, {len(gpus_free)} free of {node_gpus}"
                        gpu_wait = max(gpu_wait, j.gpus)
                        continue
                    if probe is None or time.time() - probe["t"] > 30:  # foreign load, between starts only
                        ours = {c for r in running.values() for c in r.cpus}
                        leased = leases.foreign(node)  # another pool's slots, busy or not yet
                        probe = {"t": time.time(),
                                 "mem": min(mem_available_gb(), mem_available_gb("MemTotal") - leases.reserved_gb()),
                                 "gpus": busy_gpus() | {int(n[3:]) for n in leased if n.startswith("gpu")},
                                 "cpus": frozenset((busy_cpus() - ours) | {int(n[3:]) for n in leased if n.startswith("cpu")})}
                        note = (sorted(probe["gpus"] & set(gpus_free)), len(probe["cpus"]) // 8 * 8)  # log on change
                        if note != last_note:
                            self.say(f"LOAD foreign: busy GPUs {note[0] or '-'}, busy CPUs ~{note[1]}, "
                                     f"{probe['mem']:.0f} GB available")
                            last_note = note
                    usable = [g for g in gpus_free if g not in probe["gpus"]]
                    if share:
                        usable = list(node_gpus)  # placed under the node lock after CPU/RAM pass
                    if j.gpus and self.priority >= 0 and not share:
                        if j.gpus > len(usable):
                            gpu_wait = max(gpu_wait, j.gpus)
                        elif len(usable) - j.gpus < gpus_wanted_above(leases, self.priority):
                            blocked[j.jid] = "GPU priority: a higher-priority stage waits for GPUs"
                            continue
                    if MEASURED and load.ema is not None:
                        fresh = sum(predicted_cores(r.job, history) for r in running.values() if time.time() - r.t0 < 60)
                        need = predicted_cores(j, history)
                        if load.ema + fresh + need > ADMIT_FRAC * load.ncpu:
                            blocked[j.jid] = (f"node busy {load.ema:.0f} + fresh {fresh:.0f} + {need:.1f} cores > "
                                              f"{ADMIT_FRAC:.0%} of {load.ncpu}")
                            continue
                    avail = min(probe["mem"], mem_available_gb("MemTotal") - leases.reserved_gb())
                    if j.gpus > len(usable) or ram_blocks(j.mem_gb, avail, self.mem_headroom_gb, bool(running)):
                        blocked[j.jid] = (f"GPUs usable {usable} (foreign busy {sorted(probe['gpus'])}), "
                                          f"RAM {j.mem_gb:.0f} GB vs {avail:.0f} available - {self.mem_headroom_gb:.0f} headroom")
                        continue
                    cpus = cm.alloc(j.cpus, probe["cpus"])
                    if cpus is None:
                        own = sum(len(r.cpus) for r in running.values())
                        blocked[j.jid] = (f"needs {j.cpus} CPUs, resources-busy ({own} held by this pool's legs, "
                                          f"{len(probe['cpus'])} foreign-busy)" if own >= len(probe["cpus"]) else
                                          f"needs {j.cpus} CPUs, {len(probe['cpus'])} foreign-busy ({own} held by this pool)")
                        continue
                    gp, gslots, caps, lock = usable[:j.gpus], [], "", None
                    if j.gpus and not share:  # FX-D192: an exclusive leg skips GPUs a sharing pool holds slots on
                        lock = open(leases.root / "gpu_place.lock", "w")
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        gp = exclusive_gpus(j.gpus, usable, leases)
                        if gp is None:
                            lock.close()
                            cm.release(cpus, groups)
                            blocked[j.jid] = f"needs {j.gpus} GPUs without another pool's shared slots"
                            continue
                    if share:  # FX-N100: read commitments, choose, commit, as one step per node
                        lock = open(leases.root / "gpu_place.lock", "w")
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        per = per_trainer_footprint(j, history)
                        used = {i: m / 1024 for i, _, m in _gpu_query()}
                        cap = TRAINERS_PER_GPU.get(j.dataset, 75)
                        m = max(j.gpus, -(-j.n // cap))  # unmeasured legs still respect the trainer cap (share <= 1)
                        k_even = -(-j.n // m)
                        if per is None:  # unmeasured: whole GPUs, even split
                            placed = place_gpus(m, gpu_footprint(j, history), node_gpus, leases, used, k_even / cap)
                            plan = ([(g, k_even, sl, GPU_MEM_CAP * GPU_GB, GPU_SM_CAP, round(k_even / cap, 3)) for g, sl in placed]
                                    if placed else None)
                        else:
                            plan = place_trainers(j.n, per, node_gpus, leases, used, cap)
                        if plan is None:
                            lock.close()
                            cm.release(cpus, groups)
                            blocked[j.jid] = "no GPU room for its trainers" + (f" ({per[0]:.2f} GB each)" if per else " (unmeasured: whole GPUs)")
                            continue
                        gp, gslots = [g for g, *_ in plan], [sl for _, _, sl, *_ in plan]
                        caps = ",".join(f"{g}:{k}" for g, k, *_ in plan) if per else ""
                    if not leases.take_all(slot_leases(cpus, [] if share else gp) + gslots):  # a neighbour took one
                        blocked[j.jid] = "lease race with a neighbour pool"
                        cm.release(cpus, groups)
                        probe = None
                        if lock:
                            lock.close()
                        continue
                    for g, k, sl, gb, sm, tshare in (plan if share else []):
                        leases.put(sl, f"{gb:.1f},{sm:.1f},{tshare:.3f}")
                    if lock:
                        lock.close()
                    for g in ([] if share else gp):
                        gpus_free.remove(g)
                    probe["mem"] -= j.mem_gb
                    leases.reserve_ram(f"ram_{os.getpid()}_{j.jid}", j.mem_gb)
                    port = free_port(18830, ports, leases)
                    ports.add(port)
                    running[j.jid] = self._launch(j, cpus, gp, port, caps)
                    running[j.jid].gslots = gslots
                    ledger.event("start", jid=j.jid, mode=j.mode, baseline=j.baseline, trace=j.trace, dataset=j.dataset,
                                 harness=j.harness, n=j.n, mem_est_gb=round(j.mem_gb, 1), cpus=len(cpus), gpus=gp,
                                 runtime_s=j.runtime_s, est_s=round(j.est_s))
                    pending.remove(j)
                if self.priority >= 0:
                    if gpu_wait:
                        leases.put(want_lease, gpu_wait)
                    else:
                        leases.drop([want_lease])
                time.sleep(2)
                tick += 1
                if tick % RES_SAMPLE_TICKS == 0 and running:
                    ledger.sample(running, leases)
                for jid, r in list(running.items()):
                    rc = r.proc.poll()
                    if rc is None:
                        if tick % 3 == 0:
                            r.sample()
                            recent = r.busy[-10:]
                            if MEASURED and r.job.mode != "grade" and len(recent) == 10 and \
                                    sum(b >= CPU_SAT_BUSY * len(r.cpus) for b in recent) / 10 > SAT_PAUSE:
                                paused_until = max(paused_until, time.time() + 30)
                                if not r.saturated:
                                    r.saturated = True
                                    self.say(f"BACKPRESSURE {jid}: slot of {len(r.cpus)} CPUs saturated; starts paused, timing-suspect")
                        if tick % 15 == 0 and self._fatal(r):
                            break
                        if tick % 30 == 0:
                            self._stalled(r)
                            self._doomed(r, running)
                        elif r.stalled and time.time() - r.stalled_at > 60:
                            kill_tag(r.tag)  # SIGTERM ignored: force it
                        continue
                    kill_tag(r.tag)  # leftovers of a leg that died hard
                    dur = time.time() - r.t0
                    avg = sum(r.cpu.values()) / max(1.0, dur)
                    p95 = sorted(r.busy)[int(0.95 * (len(r.busy) - 1))] if r.busy else 0.0
                    sat = sum(b >= CPU_SAT_BUSY * len(r.cpus) for b in r.busy) / max(1, len(r.busy))
                    ram_peak, ram_grow = max((s[1] for s in r.ram), default=0.0), ram_growth_gb(r.ram)
                    over = overrun_share(leg_run_dirs(r.out)) if r.gpus else (0, 0)
                    ledger.event("end", jid=jid, rc=rc, dur_s=round(dur), mem_est_gb=round(r.job.mem_gb, 1),
                                 ram_peak_gb=round(ram_peak, 2), ram_growth_gb=round(ram_grow, 2),
                                 gpu_peak_gb=round(r.gpu_peak_gb, 2), cores_p95=round(p95, 2),
                                 gpu_sm_mean=round(statistics.mean(r.gpu_sm), 1) if r.gpu_sm else None,
                                 overran=over[0], train_tasks=over[1])
                    self.say(f"DONE  {jid} rc={rc}{(' DOOMED' if r.stalled.startswith('DOOMED') else ' STALLED') if r.stalled else ''} {dur / 60:.1f}m (est {r.job.est_s / 60:.1f}m) "
                             f"cores avg {avg:.1f} p95 {p95:.1f} of {len(r.cpus)}{f' CPU_SAT {sat:.0%}' if sat > 0.05 else ''}"
                             f"{gpu_peak_note(leg_run_dirs(r.out), r.gpus)}"
                             f"{f' leg RAM {ram_peak:.0f}/{r.job.mem_gb:.0f} GB' if ram_peak else ''}"
                             f"{' RAM_OVER' if ram_peak > r.job.mem_gb else ''}"
                             f"{f' LEAK? +{ram_grow:.1f} GB 2nd half' if leak_suspect(ram_peak, ram_grow) else ''}"
                             f"{' TIMING_SUSPECT' if MEASURED and r.job.mode != 'grade' and sat > SAT_PAUSE else ''}"
                             f"{f' gpu SM {statistics.mean(r.gpu_sm):.0f}%' if r.gpu_sm else ''}"
                             f"{f' overran {over[0]}/{over[1]}' if over[1] else ''}"
                             f"{' OVERRUN' if over[1] and over[0] > 0.02 * over[1] else ''}")
                    jobs_tsv.write(f"{jid}\t{rc}\t{dur:.0f}\t{r.job.est_s:.0f}\t{len(r.cpus)}\t{avg:.2f}\t{p95:.2f}\t{sat:.3f}\n")
                    jobs_tsv.flush()
                    row = _summary_row(r.out)
                    if rc == 0 and (r.job.mode == "grade" or row.get(f"{r.job.mode}_dir")):
                        history.setdefault(r.job.key, []).append(round(dur))
                        if r.busy:  # CPU legs too: the ISO control sizes their slots from it (FX-N22)
                            history.setdefault(cores_key(r.job), []).append(round(p95, 2))
                        if ram_peak and r.job.mode != "grade":  # FX-D140: next estimate >= measured
                            history.setdefault(ram_key(r.job), []).append(round(ram_peak, 1))
                        if r.gpus and r.gpu_sm and not SHORT_LEGS:  # FX-N100: per-GPU footprint for GPU packing
                            per_gpu = r.gpu_peak_gb / r.job.gpus  # per nominal GPU (the key's split), not per placed GPU
                            if not r.gslots:  # exclusive GPUs: the node log is this leg alone
                                per_gpu = max(per_gpu, node_gpu_peak_gb(leg_run_dirs(r.out), r.gpus))
                            history.setdefault(gpu_key(r.job), []).append([round(per_gpu, 1), round(statistics.mean(r.gpu_sm), 1)])
                    if MEASURED and r.job.mode != "grade" and sat > SAT_PAUSE:  # FX-N99 3: the next lease of this key grows
                        history.setdefault(margin_key(r.job), []).append(
                            round(min(MARGIN_MAX, lease_margin(r.job, history) * MARGIN_X), 2))
                    cm.release(r.cpus, groups)
                    leases.drop(slot_leases(r.cpus, r.gpus, r.port) + [f"ram_{os.getpid()}_{jid}"] + r.gslots)
                    gpus_free += [g for g in r.gpus if g not in gpus_free and not r.gslots]
                    ports.discard(r.port)
                    self.done[jid] = r.out
                    del running[jid]
                    shown = 0.0  # a leg finished: show progress now
                    if self._fatal(r):
                        break
                if time.time() - shown > 300 and (pending or running):
                    self.progress(pending, running, cap)
                    shown = time.time()
        finally:
            if running:
                self.say(f"INTERRUPT — tearing down {len(running)} slot(s)")
                for r in running.values():
                    try:
                        os.killpg(r.proc.pid, signal.SIGTERM)
                    except OSError:
                        pass
                deadline = time.time() + 45
                while time.time() < deadline and any(r.proc.poll() is None for r in running.values()):
                    time.sleep(1)
                for r in running.values():
                    try:
                        os.killpg(r.proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    kill_tag(r.tag)
                    leases.drop(slot_leases(r.cpus, r.gpus, r.port) + [f"ram_{os.getpid()}_{r.job.jid}"] + r.gslots)
            _save_history(history, history_base)
        if stop["flag"]:
            return 130
        self.merge()
        return EXIT_FATAL if self.aborted else 0

    def _write_status(self, pending, running, released, blocked) -> None:
        """FX-D186: status.json for dispatch.py: queued / running (elapsed vs estimate) / done legs, node RAM, blocks."""
        now = time.time()
        st = {"t": now, "host": socket.gethostname(), "aborted": self.aborted,
              "held": sorted({j.unit or j.jid for j in pending if (j.unit or j.jid) not in released}),
              "queued": sorted(j.jid for j in pending if (j.unit or j.jid) in released),
              "running": {jid: {"elapsed_s": round(now - r.t0), "est_s": round(r.job.est_s), "gpus": r.gpus,
                                "stalled": r.stalled} for jid, r in running.items()},
              "done": {jid: (str(p) if p else None) for jid, p in self.done.items()},
              "blocked": dict(list(blocked.items())[:20]),
              "mem_available_gb": round(mem_available_gb(), 1), "mem_total_gb": round(mem_available_gb("MemTotal"), 1)}
        tmp = self.root / "status.json.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(self.root / "status.json")

    def _launch(self, j: Job, cpus: List[int], gp: List[int], port: int, caps: str = "") -> Running:
        out = self.root / j.phase / j.jid
        out.mkdir(parents=True, exist_ok=True)
        # Unique per pool path: two ladders' gate pools are both named P00 (one killed the other's legs).
        tag = f"pool_{self.root.name}_{hashlib.sha1(str(self.root).encode()).hexdigest()[:8]}_{j.jid}"
        args = list(j.args)
        if j.mode == "grade":
            legs = [_summary_row(self.done.get(d)) for d in j.deps]
            real = next((r.get("real_dir", "") for r in legs if r.get("real_dir")), "")
            sim = next((r.get("sim_dir", "") for r in legs if r.get("sim_dir")), "")
            args += ["--grade-only", "--real-dir", real, "--sim-dir", sim]
        else:
            args += ["--isolate", "--broker-port", str(port), "--run-tag", tag]
        if gp and (not j.whole or self.bad_gpus):
            args += ["--gpu-ids", ",".join(map(str, gp))]
        pin = [] if j.whole else ["taskset", "-c", ",".join(map(str, cpus))]
        suite = example_dir(j.dataset) / "scripts" / "harness_suite.sh"
        cmd = [*pin, "bash", str(suite), *args, "--output-dir", str(out)]
        (out / "cmd.txt").write_text(shlex.join(cmd) + "\n")
        env = {**os.environ, "EXPT_AUTOCLEAN": "1", "FLAME_RUN_TAG": tag, "FLAME_RUN_LABEL": j.phase,
               "FLAME_POST_ANALYSIS": "0"}  # FX-D62: nothing reads a leg's plots/
        if caps:
            env["FLAME_GPU_TRAINER_CAPS"] = caps  # FX-N100: trainers per GPU from the placer
        if not j.whole:
            agg = 8 if j.gpus else agg_cores(j.n)
            env["FLAME_AGG_CORES"] = str(min(agg, max(1, len(cpus) // 2)) if MEASURED else agg)  # FX-N99: trainers keep half
        proc = subprocess.Popen(cmd, stdout=open(out / "suite.log", "w"), stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, cwd=str(example_dir(j.dataset)))
        self.say(f"START {j.jid} cpus={_ranges(cpus)} port={port}{' gpus=' + str(gp) if gp else ''}"
                 f"{' trainers/gpu=' + caps if caps else ''} est={j.est_s / 60:.1f}m")
        t = time.time()
        watch = StallWatch(t, stall_limit_s(j, self.stall_min)) if self.stall_min > 0 and j.mode != "grade" else None
        return Running(j, proc, cpus, gp, port, tag, t, out, watch=watch)

    def merge(self) -> None:
        """<phase>/summary.tsv: the grade row of a pair, else the sim row (harness_suite columns)."""
        by_phase: Dict[str, List[Job]] = {}
        for j in self.jobs:
            by_phase.setdefault(j.phase, []).append(j)
        for ph, js in by_phase.items():
            js = [j for j in js if j.jid not in self.unassigned]  # FX-D186: ran on another node
            if not js or not (self.root / ph).is_dir():  # FX-N75: phase never started
                continue
            rows, header = [], None
            graded = {d for j in js if j.mode == "grade" for d in j.deps}
            for j in sorted(js, key=lambda j: j.jid):
                if j.jid in graded:
                    continue
                f = (self.done.get(j.jid) or Path("/nonexistent")) / "summary.tsv"
                if not f.exists():
                    rows.append("\t".join([j.trace, j.baseline] + ["MISSING"] * 3 + [""] * 8))
                    continue
                lines = f.read_text().splitlines()
                header = header or lines[0]
                rows += lines[1:]
            header = header or "\t".join(["trace", "baseline", "ev_real", "ev_sim", "parity", "score", "n_fail",
                                          "roots", "crash_lines", "timeout", "real_dir", "sim_dir", "real_src"])
            (self.root / ph / "summary.tsv").write_text("\n".join([header] + rows) + "\n")


def _summary_row(d: Optional[Path]) -> dict:
    f = (d or Path("/nonexistent")) / "summary.tsv"
    if not f.exists():
        return {}
    lines = f.read_text().splitlines()
    if len(lines) < 2:
        return {}
    return dict(zip(lines[0].split("\t"), lines[1].split("\t")))


def _ranges(cpus: List[int]) -> str:
    out, start, prev = [], None, None
    for c in sorted(cpus) + [None]:
        if start is None:
            start = prev = c
        elif c is not None and c == prev + 1:
            prev = c
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = c
    return ",".join(out)


def _order_key(jobs: List[Job]):
    """Longest unit (pair + grade) first, a unit's legs together; whole-node jobs last (they block every slot).
    Unit-major so a --deadline cut drops whole pairs, never every pair's sim."""
    unit_s: Dict[str, float] = {}
    for j in jobs:
        unit_s[j.unit or j.jid] = unit_s.get(j.unit or j.jid, 0.0) + j.est_s
    return lambda j: (j.whole, -unit_s[j.unit or j.jid], j.unit or j.jid, -j.est_s, j.jid)


def simulate_makespan(jobs: List[Job], cpus: int, max_parallel: int, gpus: int = 0) -> float:
    """List-scheduling estimate of wall time (cores and GPUs; deps respected)."""
    pending = sorted(jobs, key=_order_key(jobs))
    done_at: Dict[str, float] = {}
    running: List[tuple] = []  # (end, cpus, gpus, jid)
    now, free, gfree = 0.0, cpus, gpus
    while pending or running:
        for j in list(pending):
            if any(done_at.get(d, math.inf) > now for d in j.deps) or len(running) >= max_parallel:
                continue
            if j.cpus <= free and j.gpus <= gfree:
                running.append((now + j.est_s, j.cpus, j.gpus, j.jid))
                free, gfree = free - j.cpus, gfree - j.gpus
                pending.remove(j)
        if not running:  # FX-D131: nothing can start or free up
            return math.inf
        running.sort()
        end, c, g, jid = running.pop(0)
        now, free, gfree = end, free + c, gfree + g
        done_at[jid] = end
    return now


def _load_history() -> dict:
    try:
        return json.loads(DURATIONS.read_text())
    except Exception:
        return {}


def _save_history(h: dict, base: Optional[dict] = None) -> None:
    """FX-D141: merge this pool's appends onto the file under a lock; a plain write lost every concurrent pool's."""
    DURATIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(DURATIONS.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        disk = _load_history()
        for k, v in h.items():
            disk[k] = disk.get(k, []) + v[len((base or {}).get(k, [])):]
        tmp = DURATIONS.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v[-10:] for k, v in disk.items()}, indent=1, sort_keys=True))
        os.replace(tmp, DURATIONS)


def run_gate(root: Path, datasets, pool: "Pool") -> int:
    """R19: collect every test, check each dataset is complete, run one felix smoke pair per dataset;
    nonzero = abort (4)."""
    lib = REPO / "lib" / "python"
    # One collect at a time per node (FX-D101).
    with open(Leases().root / "gate_collect", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", "tests",
                                "examples/async_cifar10/scripts/parity", "examples/async_cifar10/trainer/pytorch",
                                "examples/fwdllm/expt_scripts"], cwd=str(lib), capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as e:
            r = subprocess.CompletedProcess(e.cmd, 1, str(e.stdout or ""), f"collect timed out after {e.timeout}s")
    (root / "P00_collect.txt").write_text(r.stdout + r.stderr)
    if r.returncode:
        pool.say(f"ABORT gate: pytest collection failed -- {root}/P00_collect.txt")
        return 4
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import fl_data; "
                        "[print(fl_data.verify(d)) for d in sys.argv[2:]]", str(example_dir(datasets[0])), *datasets],
                       capture_output=True, text=True, timeout=600)
    (root / "P00_data.txt").write_text(r.stdout + r.stderr)
    if r.returncode:
        pool.say(f"ABORT gate: dataset missing or incomplete -- {root}/P00_data.txt")
        return 4
    pool.say("data ok: " + " | ".join(r.stdout.split("\n")[:-1]))
    # One smoke per code state, dataset and node (FX-D101, FX-D154).
    stamps = {d: Leases().root / f"gate_ok_{code_key()}_{d}" for d in datasets}
    todo = [d for d in datasets if not stamps[d].exists()]
    if not todo:
        pool.say(f"gate ok (collect; smoke passed earlier on this code: {stamps[datasets[0]].read_text().strip()})")
        return 0
    g = root / "P00"
    cmd = [sys.executable, __file__, "--tier", "T3", "--baselines", "felix", "--phases", "T3_syn_0",
           "--smoke", "--no-gate", "--datasets", ",".join(todo), "--output-dir", str(g)]
    if GPU_ALLOW is not None:
        cmd += ["--gpu-ids", ",".join(map(str, sorted(GPU_ALLOW)))]
    rc = run_smoke(cmd, g / "pool.log")
    if rc is None:
        pool.say(f"ABORT gate: smoke pair timed out -- {g}")
        return 4
    leg_bad = lambda v: v == "MISSING" or "EV1" in v.replace("FAIL:", "").split(",")
    bad = [row for f in g.glob("*/summary.tsv") for row in f.read_text().splitlines()[1:]
           if leg_bad(row.split("\t")[2]) or leg_bad(row.split("\t")[3])
           or row.split("\t")[4] in ("MISSING", "CHECKER_ERROR")]
    if rc or bad or not list(g.glob("*/summary.tsv")):
        pool.say(f"ABORT gate: smoke pair made no progress -- {g}")
        return 4
    for d in todo:
        stamps[d].write_text(str(g))
    pool.say("gate ok (collect + smoke pair per dataset)")
    return 0


def run_smoke(cmd, log: Path, run_s: float = 900, wait_s: float = 3600) -> Optional[int]:
    """FX-D154: gate smoke timed from its first leg START, not while queued for leases; None = timed out."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # same group: Ctrl+C reaches it (R20)
    t0, started = time.time(), None
    while p.poll() is None:
        if started is None and log.exists() and " START " in log.read_text():
            started = time.time()
        if time.time() - (started or t0) > (run_s if started else wait_s):
            p.send_signal(signal.SIGINT)  # the pool tears its legs down
            try:
                p.wait(120)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            return None
        time.sleep(min(5.0, run_s / 2))
    return p.returncode


# L18: files no leg executes (docs, tests, graders, batch drivers); editing them keeps the code state.
NOT_RUN_AFFECTING = ("**/*.md", "**/*.ipynb", "lib/python/tests/**", "lib/python/examples/batches/**", "lib/python/examples/paper_felix/**", "lib/python/examples/async_cifar10/scripts/parity/**",
                     "lib/python/examples/async_cifar10/scripts/parity_check.py", "lib/python/examples/scripts/run_*.py",
                     *(f"lib/python/examples/fwdllm/expt_scripts/{p}" for p in (  # plots: PR36 split its key mid-batch
                         "plotlib/**", "make_paper_figs.py", "preview_palette.py", "figs*.yaml")),
                     *(f"lib/python/examples/scripts/{s}.py" for s in (
                         "parity_ladder", "readiness_score", "logical_diff", "insights", "accuracy_table", "replicate_spread",
                         "resource_report", "profile_report", "fd_sanity", "eurosys_export", "baseline_deviations",
                         "cleanup_experiments", "test_inventory", "harness_iso_compare", "felix_claims", "fl_lr_check", "dispatch",
                         "paper_runs", "leg_digest", "paper_plots", "paper_json")))


def code_key() -> str:
    """Blob ids of the tracked run-affecting files as on disk: two pools share a gate and floors only on identical code; a
    screen on uncommitted code keys as its commit does (FX-D194)."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout
    spec = ["--", ".", *(f":(exclude,glob){p}" for p in NOT_RUN_AFFECTING)]
    blobs = {f: meta.split()[1] for meta, f in (ln.split("\t", 1) for ln in git("ls-files", "-s", *spec).splitlines())}
    changed = [f for f in git("diff", "--name-only", "HEAD", *spec).splitlines() if f in blobs]
    for f in changed:
        blobs[f] = git("hash-object", f).strip() if (REPO / f).exists() else ""
    return hashlib.sha1("".join(f"{b}\t{f}\n" for f, b in sorted(blobs.items()) if b).encode()).hexdigest()[:12]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True,
                    help="comma list of T1 T2 T3 T4 GS G0 G0C N64 N64S G1 G2 G1A G1L N62 ISO ISO_FILL F42 F42C F42CR F42R F43 F44 F45 F46 F47 F48 F48L F48R F49 FB1 FB1S FB2<set>[S] (FB2_SETS), e.g. 'T2,G1' = CPU matrix + GPU block")
    ap.add_argument("--datasets", default="cifar10", help="comma list of cifar10, google_speech, or 'all'")
    ap.add_argument("--baselines", default="", help="space- or comma-separated; default = --changed set, else all six")
    ap.add_argument("--changed", default="", help="git ref: run only baselines affected by the diff vs it")
    ap.add_argument("--phases", default="", help="subset of the tier's phase ids (e.g. 'P1 P2', 'P11a')")
    ap.add_argument("--exclude-phases", default="", help="exact phase ids to drop (e.g. 'G0_syn_0 gs_G0_syn_50_3st')")
    ap.add_argument("--shard", default="1/1", help="i/N: this node's share of the job list")
    ap.add_argument("--max-parallel", type=int, default=32)
    ap.add_argument("--reserve-cores", type=int, default=4, help="physical cores kept out of every slot")
    ap.add_argument("--cpus-per-trainer", type=float, default=1.0,
                    help="logical CPUs per trainer; < 1 only after the ISO control proves it (FX-N22 P6)")
    ap.add_argument("--traces", default="", help="restrict the tier to these traces")
    ap.add_argument("--agg-hp", default="", help="'k=v ...' appended to every phase's aggregator hp (A/B)")
    ap.add_argument("--trainer-hp", default="", help="'k=v ...' appended to every phase's trainer hp")
    ap.add_argument("--inject-bug", default="", help="S1 bug for every leg (trainer_crash: FX-N40 fail-fast smoke)")
    ap.add_argument("--gpu-ids", default="", help="GPUs this pool may use (default: all healthy; busy ones skipped per start)")
    ap.add_argument("--gpus-per-job", type=int, default=8, help="GPU legs: GPUs each (capped at the healthy count)")
    ap.add_argument("--gpu-cpus-per-trainer", type=float, default=0.4,
                    help="GPU legs: CPUs per trainer (0.4 = n=300 on 128 CPUs -> whole node; n=100 -> 48 CPUs)")
    ap.add_argument("--mem-headroom-gb", type=float, default=None,
                    help="host RAM kept free (default: 10%% of MemTotal; operator 10-15: no resource past 90%%)")
    ap.add_argument("--deadline-h", type=float, default=12)
    ap.add_argument("--progress-label", default="", help="prefix of the PROGRESS lines (the ladder passes its rung)")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--smoke", action="store_true", help="every leg 60s, n<=12 (the pool's own gate)")
    ap.add_argument("--seeds", default="", help="F42 tiers: replicate ids to run, e.g. 1 or 2,3 (default 1)")
    ap.add_argument("--scale-smoke", type=int, default=0, metavar="S",
                    help="FX-D56: every leg S seconds at its production n, c and GPUs (density/OOM check before a block)")
    ap.add_argument("--gate", action=argparse.BooleanOptionalAction, default=None,
                    help="R19 gate first: pytest --collect-only + a felix smoke pair per dataset; abort on "
                         "failure (default on unless --smoke/--dry-run)")
    ap.add_argument("--pytest", action="store_true", help="run the full pytest (P0) before the jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fail-fast", dest="fail_fast", action="store_false",
                    help="FX-N40: keep going after a fatal line (Traceback, CUDA OOM, ...) in a leg's logs")
    ap.add_argument("--stall-min", type=float, default=15.0,
                    help="FX-D45: kill a leg with no new committed round for this many minutes (0 = off)")
    ap.add_argument("--doom-min", type=float, default=5.0,
                    help="early exit: prefix-safe EV checks on live legs every this many minutes (0 = off)")
    ap.add_argument("--lease-from-telemetry", action="store_true",
                    help="FX-N99: lease measured cores, admit by node load, back-pressure on saturated slots")
    ap.add_argument("--priority", type=int, default=-1, help="FX-N99: GPU priority among co-running pools (lower first)")
    ap.add_argument("--gpu-share", type=int, default=1, help="FX-N100: max legs per GPU, placed by measured footprint")
    ap.add_argument("--jobs-json", default="", help="FX-D186: write this node's jobs, estimates and capacity here and exit")
    ap.add_argument("--queue", default="", help="FX-D186: run only units listed in this file (dispatch.py); END / ABORT")
    return ap


def plan(a, ap):
    """Phases and jobs for parsed args (main, and bench_pool --tier: the same legs, FX-D213)."""
    global MEASURED, GPU_SHARE, SHORT_LEGS, F42_SEEDS
    if a.seeds:
        F42_SEEDS = tuple(int(x) for x in a.seeds.split(","))
    MEASURED, GPU_SHARE, SHORT_LEGS = a.lease_from_telemetry, a.gpu_share, bool(a.scale_smoke)
    set_gpu_allow(a.gpu_ids)

    real_changed = False
    if a.baselines:
        baselines = tuple(a.baselines.replace(",", " ").split())
        unknown = [b for b in baselines if b not in B6 + ("fedavg",)]  # fedavg: feddance's random arm (G1XF/G1XFS)
        if unknown:  # an unknown name used to run nothing and exit 0
            ap.error(f"--baselines: unknown {unknown}; expected any of {B6 + ('fedavg',)}")
    elif a.changed:
        baselines, real_changed = affected(changed_paths(a.changed))
    else:
        baselines = B6
    datasets = DATASETS if a.datasets == "all" else tuple(a.datasets.replace(",", " ").split())
    tiers = a.tier.replace(",", " ").split()
    phases = [p for ds in datasets for t in tiers for p in tier_phases(t, baselines, ds)]
    if a.phases:
        want = set(a.phases.split())
        phases = [p for p in phases if p.pid in want or p.pid[len(DS_TAG[p.dataset]):] in want]
    if a.exclude_phases:
        phases = [p for p in phases if p.pid not in set(a.exclude_phases.split())]
    if a.traces:
        phases = [p for p in phases if p.trace in a.traces.split()]
    phases = [p for p in phases if p.baselines]
    for p in phases:
        p.agg_hp = " ".join(x for x in (p.agg_hp, a.agg_hp) if x)
        p.trainer_hp = " ".join(x for x in (p.trainer_hp, a.trainer_hp) if x)
        p.inject_bug = p.inject_bug or a.inject_bug
    if a.smoke:  # R19 gate: every leg 60s, small cohorts, 1 GPU per GPU leg (plumbing, not timing)
        for p in phases:
            p.runtime_s, p.n = 60, min(p.n, 12)
            p.c = min(p.c, p.n) if p.c else p.c
            p.gpus = 1 if p.harness == "none" else p.gpus
    if a.scale_smoke:  # FX-D56: production density, short: the first dispatch wave is the memory peak
        cap_runtime(phases, a.scale_smoke)
    history = _load_history()
    jobs = build_jobs(phases, a.cpus_per_trainer, a.gpus_per_job, history, a.gpu_cpus_per_trainer)
    i, n = map(int, a.shard.split("/"))
    jobs = shard(jobs, i, n)
    return jobs, tiers, datasets, baselines, real_changed


def main(argv=None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    jobs, tiers, datasets, baselines, real_changed = plan(a, ap)

    root = Path(a.output_dir or OUT_DIR / f"pool_{time.strftime('%Y%m%d_%H%M%S')}_{'_'.join(tiers)}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "code_key.txt").write_text(code_key() + "\n")  # parity_ladder pairs floors only within one code state
    headroom = a.mem_headroom_gb if a.mem_headroom_gb is not None else 0.1 * mem_available_gb("MemTotal")
    pool = Pool(root, jobs, a.max_parallel, a.reserve_cores, headroom, a.deadline_h * 3600, a.dry_run,
                log=open(root / "pool.log", "a"), fail_fast=a.fail_fast, label=a.progress_label,
                stall_min=a.stall_min, doom_min=a.doom_min, priority=a.priority,
                queue=Path(a.queue).resolve() if a.queue else None)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    pool.say(f"pool {root} host={socket.gethostname()} tier={a.tier} datasets={','.join(datasets)} baselines={' '.join(baselines) or '-'} "
             f"shard={a.shard} commit={commit}")
    if a.changed:
        pool.say(f"changed vs {a.changed}: real path {'CHANGED -> run T3 too' if real_changed else 'unchanged'}")
    groups = core_groups()
    cap = CoreMap(groups, a.reserve_cores).total()
    for j in jobs:
        if j.cpus > cap:  # runs alone on every slot CPU, as a solo run does today
            pool.say(f"note: {j.jid} wants {j.cpus} cpus > {cap}: takes the whole node")
            j.cpus, j.whole = cap, True
        if j.gpus:
            j.gpus = min(j.gpus, len(gpu_ids()))
    bad_gpus = sorted(set(gpu_ids(healthy_only=False)) - set(gpu_ids()))
    if bad_gpus:
        pool.say(f"GPU(s) {bad_gpus} excluded (ECC error or not in --gpu-ids); GPU legs use {gpu_ids()}, "
                 f"skipping any another process holds when a leg starts")
    pool.bad_gpus = bad_gpus
    if a.jobs_json:  # FX-D186: dispatch.py plans on each node's own history and capacity
        Path(a.jobs_json).write_text(json.dumps({
            "host": socket.gethostname(), "code_key": code_key(), "cpus": cap, "gpus": gpu_ids(),
            "mem_budget_gb": round(mem_available_gb("MemTotal") - headroom, 1),
            "jobs": [{"jid": j.jid, "unit": j.unit or j.jid, "mode": j.mode, "phase": j.phase, "dataset": j.dataset,
                      "est_s": round(j.est_s), "cpus": j.cpus, "mem_gb": round(j.mem_gb, 1), "gpus": j.gpus,
                      "deps": list(j.deps), "runtime_s": j.runtime_s} for j in jobs]}))
        return 0
    pool.plan(groups)
    if a.dry_run or not jobs:
        return 0
    if (a.gate is None and not a.smoke and not a.scale_smoke) or a.gate:
        rc = run_gate(root, datasets, pool)
        if rc:
            return rc
    if a.pytest:
        pool.say("P0 full pytest ...")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-n", "auto", "-p", "no:cacheprovider", "tests",
                            "examples/fwdllm/expt_scripts", "examples/async_cifar10/scripts/parity",
                            "examples/async_cifar10/trainer/pytorch"], cwd=str(REPO / "lib" / "python"),
                           capture_output=True, text=True, timeout=2400)
        (root / "P0_pytest.txt").write_text(r.stdout + r.stderr)
        pool.say(f"P0 rc={r.returncode} :: {(r.stdout.strip().splitlines() or [''])[-1]}")
    rc = pool.run(groups)
    if rc in (0, EXIT_FATAL):
        subprocess.run(["bash", str(SCRIPT_DIR / "harness_report.sh"), str(root), str(int(pool.t0))])
        pool.say(f"done: {root}/SUMMARY.txt")
        print((root / "SUMMARY.txt").read_text() if (root / "SUMMARY.txt").exists() else "")
    return rc


if __name__ == "__main__":
    sys.exit(main())
