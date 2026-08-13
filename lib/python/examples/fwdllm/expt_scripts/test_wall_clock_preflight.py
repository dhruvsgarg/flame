#!/usr/bin/env python
"""Task 0.7 sanity gate.

Part 1 (CPU-only, instant): `wall_clock_preflight.project()` against the four
historical arms it must reproduce the fate of (fl_fwd_ft_practice.md P4/P4.2):
002208/022448 (stride 1, killed by SIM_WALL_CEILING) must project a breach;
112201/145729 (stride 25) must not.

Part 2 (needs an active conda env with the fluxtune deps, ~30-60s each): the
same two cases end to end through run_sequential.sh --dry-run, reading the
"wall-clock budget preflight" entry out of manifest.tsv.spec.json -- refuses
002208's config, passes a stride-25 arm, and prints the three factors on
refusal (edge case e).

    python test_wall_clock_preflight.py
"""
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", ".."))
from examples.fwdllm.expts import wall_clock_preflight as wcp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SEQ = os.path.join(SCRIPT_DIR, "run_sequential.sh")
SMOKE_LOGS = os.path.join(SCRIPT_DIR, "smoke_logs")

failures = []


def check(cond, msg):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        failures.append(msg)


# ---------------- part 1: the pure function against the four known arms -------
print("part 1: wall_clock_preflight.project() vs the four historical arms")
CASES = [
    # rid, s, vclock_budget, ceiling, stride, expect_breach
    ("002208", 2.9, 14400, 7200.0, None, True),
    ("022448", 1.5, 14400, 7200.0, None, True),
    ("112201", 2.9, 28800, 21600.0, 25, False),
    ("145729", 1.5, 47000, 21600.0, 25, False),
]
for rid, s, vb, ceiling, stride, expect in CASES:
    proj = wcp.project(p=450340, rho_star=0.06, gate_safety_s=s, rule="mean",
                        perturbation_count=10, K=10, vclock_budget_s=vb,
                        real_wall_ceiling_s=ceiling, cos_audit_on=True,
                        cos_probe_every=stride, cos_probe_batch_size=None)
    check(proj.breach == expect,
          f"{rid}: breach={proj.breach} (want {expect}) -- {proj.explain()}")

# edge case (a): zero cos fires / audit off -> no audit tax, per_commit_cost = base only
p_off = wcp.project(p=450340, rho_star=0.06, gate_safety_s=2.9, rule="mean",
                     perturbation_count=10, K=10, vclock_budget_s=14400,
                     real_wall_ceiling_s=7200.0, cos_audit_on=False,
                     cos_probe_every=None, cos_probe_batch_size=None)
check(p_off.per_commit_cost == wcp.AUDIT_BASE_S,
      f"audit off -> per_commit_cost is base only ({p_off.per_commit_cost})")

# select rule with an unmeasured P must refuse, not interpolate
try:
    wcp.project(p=450340, rho_star=0.06, gate_safety_s=2.9, rule="select",
                perturbation_count=20, K=10, vclock_budget_s=14400,
                real_wall_ceiling_s=7200.0, cos_audit_on=True,
                cos_probe_every=None, cos_probe_batch_size=None)
    check(False, "select at unmeasured P=20 should have raised")
except ValueError:
    check(True, "select at unmeasured P=20 refuses (no interpolation)")


# ---------------- part 2: end to end through the real launcher ----------------
def run_launcher(args, timeout=180):
    before = set(glob.glob(os.path.join(SMOKE_LOGS, "*")))
    p = subprocess.run(["bash", RUN_SEQ] + args, capture_output=True, text=True, timeout=timeout)
    after = set(glob.glob(os.path.join(SMOKE_LOGS, "*")))
    new_dirs = sorted(after - before)
    logdir = new_dirs[-1] if new_dirs else None
    return p.returncode, p.stdout + p.stderr, logdir


def load_spec(logdir):
    path = os.path.join(logdir, "manifest.tsv.spec.json") if logdir else None
    if not path or not os.path.exists(path):
        return None
    return json.load(open(path))


def find_check(spec, name_substr):
    for c in (spec or {}).get("checks", []):
        if name_substr in c.get("name", ""):
            return c
    return None


COMMON = ["--only", "fluxtune", "--mode", "sim", "--dry-run",
          "--cos-ground-truth-audit", "--num-trainers", "100", "--num-gpus", "8",
          "--agg-goal", "10", "--c", "30", "--probe-combine", "mean",
          "--commit-gate", "n_target", "--server-step-rule", "trust_ratio",
          "--rho-star", "0.06"]

if os.environ.get("SKIP_LAUNCHER_CASES"):
    print("part 2: skipped (SKIP_LAUNCHER_CASES set)")
else:
    print("part 2: end to end through run_sequential.sh --dry-run")
    rc, out, logdir = run_launcher(COMMON + [
        "--max-runtime-s", "14400", "--sim-wall-ceiling-h", "2.0",
        "--rho-schedule", "rm", "--rho-exp", "0.25", "--gate-rho-ref", "annealed",
        "--gate-safety-s", "2.9"])
    spec = load_spec(logdir)
    c = find_check(spec, "wall-clock budget preflight")
    check(rc == 2, f"002208-equivalent (stride 1): launcher exit 2 (got {rc})")
    check(c is not None and c["level"] == "error",
          f"002208-equivalent: check is level=error (got {c})")
    check(c is not None and "commits_projected" in c.get("detail", "")
          and "per_commit_cost" in c.get("detail", ""),
          "002208-equivalent: refusal prints the three factors (edge case e)")

    rc, out, logdir = run_launcher(COMMON + [
        "--max-runtime-s", "28800", "--sim-wall-ceiling-h", "6.0",
        "--rho-schedule", "const", "--gate-rho-ref", "setpoint",
        "--cos-probe-every", "25", "--gate-safety-s", "2.9"])
    spec = load_spec(logdir)
    c = find_check(spec, "wall-clock budget preflight")
    check(rc == 0, f"112201-equivalent (stride 25): launcher exit 0 (got {rc})")
    check(c is not None and c["level"] == "ok",
          f"112201-equivalent: check is level=ok (got {c})")

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    sys.exit(1)
print("all checks passed")
