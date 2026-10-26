#!/usr/bin/env python3
"""FX-N163 screen: are the sync cifar stacks lr-bound? FedAvg + REFL on `syn_0` (FB2E2E_s0u_sync_srv), bench, one trainer-hp
change per arm vs the source-faithful FB2E2E legs (control, read from their digests at the same horizon).

  python scripts/run_lr_screen.py --out experiments/N163_<mmdd> [--runtime-s 86400] [--gpus 1,7] [--arms nodecay ...]
  python scripts/run_lr_screen.py --out experiments/N163_<mmdd> --read [--control <FB2 root>]   # accuracy table vs control

Arms: nodecay (lr decay off, as FedBuff / Felix) · horizon (decay every 100 rounds: the source's 10 x our ~10x rounds) · epoch
(1 local epoch, the REFL paper's Table 1, instead of the code's 1 step). One bench pool per arm (phase ids collide); one Ctrl+C
stops all (R20). Predict: nodecay and horizon pass 55% by 24 h and still rise = lr-bound; epoch highest.
"""
import argparse
import glob
import gzip
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL = "/coc/scratch/dgarg/mlsys27_felix_runs_20261026/cifar"
HOURS = (2, 6, 12, 24)
ARMS = {"nodecay": "lrDecayEnabled=false", "horizon": "lrDecayEpoch=100", "epoch": "localWork=cap localSteps=100000"}


def read(out, control):
    """Accuracy at HOURS, best, rounds per leg: control (source config) then each arm."""
    legs = [("control", p) for p in glob.glob(f"{control}/FB2E2E_s0u_sync_srv/runs/syn_0_*_s1/run_*/digest.json.gz")]
    legs += [(p.split(os.sep)[-6], p) for p in glob.glob(f"{out}/*/FB2E2E_s0u_sync_srv/runs/syn_0_*_s1/run_*/digest.json.gz")]
    print(f"| arm | baseline | {' | '.join(f'{h} h' for h in HOURS)} | best | rounds | end h |\n|---|---|{'---|' * len(HOURS)}---|---|---|")
    for arm, p in sorted(legs):
        d = json.load(gzip.open(p))
        b = p.split(os.sep)[-3].split("_")[2]
        if b not in ("fedavg", "refl"):
            continue
        a, t = d["acc"]["acc"], d["acc"]["t"]
        at = lambda h: next((f"{100 * x:.1f}" for x, tt in reversed(list(zip(a, t))) if tt <= h * 3600), "-")
        h_end = d["end"]["h"]
        print(f"| {arm} | {b} | {' | '.join(at(h) if h <= h_end + 0.1 else '-' for h in HOURS)} | "
              f"{100 * max(x for x, tt in zip(a, t) if tt <= 24.1 * 3600):.1f} | {d['updates']['versions']} | {h_end:.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-s", type=float, default=86400)
    ap.add_argument("--gpus", default="1,7")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--baselines", default="fedavg,refl")
    ap.add_argument("--timeout-s", type=float, default=6 * 3600)
    ap.add_argument("--read", action="store_true", help="print the result table and exit")
    ap.add_argument("--control", default=CONTROL)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.read:
        return read(a.out, a.control)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "code_state.txt"), "w") as f:
        f.write(subprocess.run(["git", "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout)
    py = ["conda", "run", "--no-capture-output", "-n", "dg_flame", "python"]
    procs = []
    for arm in a.arms:
        hargs = (f"--tier FB2E2E --datasets cifar10 --baselines {a.baselines} --phases FB2E2E_s0u_sync_srv "
                 f"--trainer-hp '{ARMS[arm]}'")
        cmd = py + ["scripts/bench_pool.py", "--harness-args", hargs, "--out", os.path.join(a.out, arm), "--gpus", a.gpus,
                    "--per-gpu", "4", "--seeds", "1", "--runtime-s", str(int(a.runtime_s)), "--timeout-s", str(int(a.timeout_s)),
                    "--digest"]
        log = open(os.path.join(a.out, f"{arm}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        print(f"[N163] {arm}: {ARMS[arm]} -> {a.out}/{arm}.log", flush=True)

    def stop(*_):
        print("[N163] stopping", flush=True)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(10)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        sys.exit(130)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)
    rc = max(p.wait() for p in procs)
    print(f"[N163] done rc={rc}; read: python3 scripts/run_lr_screen.py --out {a.out} --read", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
