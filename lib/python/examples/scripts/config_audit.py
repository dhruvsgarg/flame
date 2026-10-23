#!/usr/bin/env python3
"""F42: the knobs every leg actually ran with (aggregator_config.json + the generated trainer overrides) vs the confirmed table
(FELIX_READINESS "F42 knob table"), plus V3 P3 / P4 from trainer telemetry: every train task trained on exactly the visible samples
minus the holdout (none held out at holdout 0). Exit 1 on any mismatch, so a smoke catches a leg that fell back to a wrong config.

  config_audit.py POOL...
"""
import glob
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "async_cifar10", "experiments")

# (baseline, arm) -> expected knobs; arm "*" = every arm of that baseline. "-" = unset / off.
_OLD = dict(know="notify", evict="true", eval="1.0", ups="-", bal="-", dkc="-", rate="new", probe="-", vprobe="-", holdout="-")
_E0 = dict(_OLD, eval="0")
_L3 = dict(_E0, ups="true", prefix="64")
_L3B = dict(_L3, bal="1.0")
_L4 = dict(_L3B, dkc="aimd_k")
_L5 = dict(_L4, rate="value")
_VAL = dict(probe="60", vprobe="true", holdout="0.1")  # FX-N130: arms reading held-out data hold out 10%
_V2 = dict(_L3B, dkc="pace_gain", kfc="false", vrep="true", **_VAL)  # FX-N127: l3b + pacer v2, K fixed
EXPECT = {
    ("felix", "ctl"): _OLD, ("felix", "e0"): _E0, ("felix", "l3"): _L3, ("felix", "l3b"): _L3B, ("felix", "l4"): _L4,
    ("felix", "l5"): _L5, ("felix", "l5p"): dict(_L5, dkc="composite[aimd_k,pace_c]", **_VAL),
    ("felix", "e0p"): dict(_E0, dkc="pace_c", **_VAL),
    ("felix", "fck"): dict(_L5, dkc="composite[aimd_k,pace_c]", know="checkin", evict="false", **_VAL),
    ("felix", "v2p"): _V2, ("felix", "v2pr"): _V2, ("felix", "v2p_s025"): dict(_V2, lr="0.25"),
    ("felix", "ctl_s05"): dict(_OLD, lr="0.5"), ("felix", "ctl_s025"): dict(_OLD, lr="0.25"),
    ("felix", "l3b_s05"): dict(_L3B, lr="0.5"), ("felix", "l3b_s025"): dict(_L3B, lr="0.25"),
    ("felix", "ctl_ema"): _OLD, ("felix", "l3b_ema"): _L3B, ("fedbuff", "ctl_ema"): dict(know="checkin", evict="false", rate="old"),
    ("felix", "ctl_srv"): dict(_OLD, serve="sweep"), ("felix", "l3b_srv"): dict(_L3B, serve="sweep"),
    ("felix", "l3b_b3_srv"): dict(_L3B, bal="3.0", serve="sweep"),
    ("felix", "l3b_pb_srv"): dict(_L3B, prefix="-1", serve="sweep"),
    ("felix", "l3b_ro_srv"): dict(_L3B, rate="old", serve="sweep"),
    ("felix", "rnd_srv"): dict(_E0, serve="sweep"),
    ("felix", "l3bn_srv"): dict(_L3B, serve="sweep"),
    ("felix", "rnd_ro_srv"): dict(_E0, rate="old", serve="sweep"), ("felix", "rnd_ron_srv"): dict(_E0, rate="old", serve="sweep"),
    ("fedbuff", "ctl_srv"): dict(know="checkin", evict="false", rate="old", serve="sweep"),
    ("fedbuff", "ctl_s05"): dict(know="checkin", evict="false", rate="old", lr="0.5"),
    ("fedbuff", "ctl_s025"): dict(know="checkin", evict="false", rate="old", lr="0.25"),
    ("fedbuff", "*"): dict(know="checkin", evict="false", rate="old"), ("fedbuff", "fbstar"): dict(know="oracle", evict="false", rate="old"),
    ("refl", "ctl"): dict(know="checkin", evict="false", pred="prophet", holdoff="0"),
    ("refl", "rstar"): dict(know="oracle", evict="false", pred="oracle", prio="1", prob="0.9", holdoff="0"),
    ("feddance", "ctl"): dict(know="checkin", evict="false", checkin="selection"),
    ("feddance", "fdstar"): dict(know="oracle", evict="false", checkin="selection"),
    ("fedavg", "*"): dict(know="checkin", evict="false"),
    ("oort", "*"): dict(know="checkin", evict="false"), ("oort_star", "*"): dict(know="oracle", evict="false"),
}
COMMON = dict(holdout="-", serve="-", p34="ok")


def _s(v):
    return "-" if v is None or v == "" else str(v).lower()


def knobs(run_dir, trainer_hp):
    cfg = json.load(open(os.path.join(run_dir, "aggregator_config.json")))
    hp, sel, opt = cfg.get("hyperparameters", {}), cfg.get("selector", {}).get("kwargs", {}) or {}, \
        cfg.get("optimizer", {}).get("kwargs", {}) or {}
    dk = sel.get("dynamic_kc") or {}
    pol = dk.get("policy") if str(dk.get("enabled")).lower() == "true" else None
    if pol == "composite":
        pol = "composite[" + ",".join(p["name"] for p in (dk.get("policy_kwargs") or {}).get("sub_policies", [])) + "]"
    filt, evict = str(hp.get("avail_select_filter")), str(hp.get("proactive_inflight_evict"))
    return dict(know=hp.get("avail_knowledge") or ("notify" if evict == "True" else "oracle" if filt == "True" else "none"),
                evict=_s(evict), eval=_s(sel.get("evalGoalFactor")), ups=_s(sel.get("utilityPerSecond")),
                bal=_s(sel.get("participationDecay")), dkc=_s(pol), rate=_s((opt.get("agg_rate_conf") or {}).get("type")),
                probe=_s(sel.get("probePerWindow")), vprobe=_s(trainer_hp.get("validation_probe")),
                prefix=_s(trainer_hp.get("utilityPrefixSamples")), holdout=_s(trainer_hp.get("validation_holdout")),
                pred=_s(sel.get("avail_predictor")), prio=_s(sel.get("avail_priority")), prob=_s(sel.get("avail_probability")),
                checkin=_s(sel.get("checkin_source")), holdoff=_s(sel.get("holdoff_rounds")), K=_s(hp.get("aggGoal")), c=_s(sel.get("c")),
                timeout=_s(hp.get("send_timeout_wait_s")), lr=_s(opt.get("learning_rate")), kfc=_s(dk.get("k_follows_c")),
                vrep=_s(trainer_hp.get("validation_report")), serve=_s(hp.get("serveEma")))


def holdout_check(run_dir, holdout):
    """P3 / P4: each train task's trained count = visible − visible // k (k = 1 / holdout; 0 held out when off)."""
    k = round(1 / float(holdout)) if holdout not in ("-", "0", "0.0") else 0
    bad = n = 0
    for f in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        for line in open(f):
            if '"trainer_round"' not in line:
                continue
            r = json.loads(line)
            if r.get("task_to_perform", "train") != "train" or "visible_samples" not in r or "dataset_size" not in r:
                continue
            v = int(r["visible_samples"])
            n, bad = n + 1, bad + (int(r["dataset_size"]) != v - (v // k if k else 0))
    return "ok" if n and not bad else f"{bad}/{n} bad"


def legs(pool):
    for y in sorted(glob.glob(os.path.join(pool, "*", "*_sim", "runs", "*", "debug_run.yaml"))):
        if f"{os.sep}_yaml{os.sep}" in y:  # bench_pool's dry-run inputs, not legs
            continue
        leg = y.split(os.sep)[-4]
        m = re.match(r"(?:gs_)?F\d+\w*?_(s\d+u?)_(\w+?)_(syn_\w+|mobiperf\w*)_(\w+)_sim$", leg)
        if not m:
            continue
        cell, arm, _, b = m.groups()
        arm = re.sub(r"_r\d+$", "", arm)  # F42-T12 (e): seed replicates audit against their arm's row
        exp = (yaml.safe_load(open(y)).get("experiments") or [{}])[0]
        thp = ((exp.get("trainer") or {}).get("config_overrides") or {}).get("hyperparameters") or {}
        phase = re.match(rf".*?_{cell}_{arm}(?:_r\d+)?(?=_)", leg).group(0)
        runs = sorted(glob.glob(os.path.join(RUNS, f"run_*_{phase}_dbg_*{b}_n*_sim")), key=os.path.getmtime)
        runs = [r for r in runs if re.fullmatch(rf"run_\d+_\d+_{phase}_dbg_(?:google_speech_)?{b}_n\d+\S*_sim", os.path.basename(r))]
        if runs:
            k = knobs(runs[-1], thp)
            yield leg, b, arm, dict(k, p34=holdout_check(runs[-1], k["holdout"]))
    for run in sorted(glob.glob(os.path.join(pool, "*", "runs", "*_s[0-9]*", "run_*"))):  # bench_pool layout (FX-D213)
        phase, slot = run.split(os.sep)[-4], run.split(os.sep)[-2]
        m = re.match(r"(?:gs_)?F\d+\w*?_s\d+u?_(\w+?)$", phase)
        b = re.match(r"(?:syn_\w+?|mobiperf\w*?)_([a-z_]+)_s\d+$", slot)
        if not (m and b):
            continue
        exp = (yaml.safe_load(open(os.path.join(run, "debug_run.yaml"))).get("experiments") or [{}])[0]
        thp = ((exp.get("trainer") or {}).get("config_overrides") or {}).get("hyperparameters") or {}
        k = knobs(run, thp)
        yield f"{phase}/{slot}", b.group(1), re.sub(r"_r\d+$", "", m.group(1)), dict(k, p34=holdout_check(run, k["holdout"]))


def main(argv=None):
    pools = (argv if argv is not None else sys.argv[1:])
    bad = 0
    cols = ("know", "evict", "eval", "ups", "bal", "dkc", "rate", "probe", "vprobe", "prefix", "holdout", "pred", "prio", "prob",
            "checkin", "holdoff", "K", "c", "timeout", "lr", "kfc", "vrep", "serve", "p34")
    print("| leg | " + " | ".join(cols) + " | ok |\n|" + "---|" * (len(cols) + 2))
    for pool in pools:
        for leg, b, arm, k in legs(pool):
            base = re.sub(r"_srv$", "", arm)  # FX-N133: an `_srv` arm = its base arm + the averaged model
            row = EXPECT.get((b, arm)) or EXPECT.get((b, base)) or EXPECT.get((b, "*"))
            row = dict(row, serve="sweep") if row is not None and arm != base else row
            want = dict(COMMON, **(row or {}))
            miss = [f"{x}={k[x]}≠{v}" for x, v in want.items() if k.get(x) != v] + ([] if row else ["no expected row"])
            bad += bool(miss)
            print(f"| {leg} | " + " | ".join(k[x] for x in cols) + f" | {'✅' if not miss else '❌ ' + ', '.join(miss)} |")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
