#!/usr/bin/env python
"""Task 0.8 sanity gate — CPU-only, no GPU, ~30-60s (each case shells out to
run_sequential.sh --dry-run, which pays conda-activation + h5py-open cost).

Exercises `--dataset NAME` end to end through the real launcher and inspects
the machine-readable `manifest.tsv.spec.json` it writes, rather than scraping
terminal output. Five cases, each the sanity gate for one buildplan §2 task-0.8
edge case:

  1. unset --dataset            -> byte-identical to today (agnews paths, seq 192)
  2. --dataset yahoo            -> yahoo paths in BOTH override blocks, seq 256,
                                   trainer.dataset.name=yahoo, run name tagged
  3. --dataset yelp-p           -> same, yelp-p paths, seq 256
  4. unknown --dataset name     -> fails fast (exit 3), no spec.json written
  5. --dataset + a group that only exists in a DIFFERENT dataset's h5
                                   -> "partition group exists" check is level=error
                                      (edge case a: don't silently read the wrong file)
  6. --dataset yahoo, --mode sim, no --force
                                   -> "sim charge profile matches dataset" is
                                      level=error (edge case d: profile is agnews-only)

Needs an active conda env with the fluxtune deps (same requirement as
run_sequential.sh itself) -- set FLAME_CONDA_ENV if none is active.

    python test_dataset_launcher.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SEQ = os.path.join(SCRIPT_DIR, "run_sequential.sh")
SMOKE_LOGS = os.path.join(SCRIPT_DIR, "smoke_logs")

failures = []


def check(cond, msg):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        failures.append(msg)


def run_launcher(args, timeout=180):
    """bash run_sequential.sh <args>, returns (returncode, stdout+stderr, new_logdir_or_None)."""
    before = set(glob.glob(os.path.join(SMOKE_LOGS, "*")))
    p = subprocess.run(["bash", RUN_SEQ] + args, capture_output=True, text=True, timeout=timeout)
    after = set(glob.glob(os.path.join(SMOKE_LOGS, "*")))
    new_dirs = sorted(after - before)
    logdir = new_dirs[-1] if new_dirs else None
    return p.returncode, p.stdout + p.stderr, logdir


def load_spec(logdir):
    path = os.path.join(logdir, "manifest.tsv.spec.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def check_level(spec, name_substr):
    for c in spec["checks"]:
        if name_substr in c["name"]:
            return c["level"]
    return None


def yaml_text(logdir, pattern):
    matches = glob.glob(os.path.join(logdir, pattern))
    if not matches:
        return None
    with open(matches[0]) as fh:
        return fh.read()


_cleanup_dirs = []


def main():
    if not os.environ.get("FLAME_CONDA_ENV") and not os.environ.get("CONDA_DEFAULT_ENV"):
        print("ERROR: no conda env active and FLAME_CONDA_ENV unset (same requirement as "
              "run_sequential.sh itself).", file=sys.stderr)
        sys.exit(2)

    # ---- 1. unset --dataset: byte-identical (agnews paths, seq 192) ----
    print("case 1: --dataset unset")
    rc, out, logdir = run_launcher(["--dry-run", "--only", "fluxtune"])
    _cleanup_dirs.append(logdir)
    txt = yaml_text(logdir, "fluxtune_n100_smoke_*_real.yaml") if logdir else None
    check(rc == 0, f"unset --dataset dry-run exits 0 (got {rc})")
    check(bool(txt) and "agnews_data.h5" in txt and "agnews_partition.h5" in txt,
          "unset --dataset keeps agnews data/partition paths")
    # the source yaml never sets max_seq_length explicitly (inherits the code/
    # trainer_base.yaml default of 192) -- unset --dataset must not add it either.
    check(bool(txt) and "max_seq_length" not in txt,
          "unset --dataset does not inject max_seq_length (inherits the 192 code default)")
    check(bool(txt) and "name: agnews" in txt, "unset --dataset keeps trainer.dataset.name: agnews")

    # ---- 2/3. --dataset yahoo / yelp-p: both override blocks patched ----
    for ds, seq in (("yahoo", 256), ("yelp-p", 256)):
        print(f"case: --dataset {ds}")
        rc, out, logdir = run_launcher(["--dry-run", "--only", "fluxtune", "--dataset", ds, "--force"])
        _cleanup_dirs.append(logdir)
        txt = yaml_text(logdir, f"fluxtune_{ds}_n100_smoke_*_real.yaml") if logdir else None
        check(bool(txt), f"--dataset {ds} produced a run named with the {ds} tag")
        if txt:
            n_data = txt.count(f"{ds}_data.h5")
            n_part = txt.count(f"{ds}_partition.h5")
            n_seq = txt.count(f"max_seq_length: {seq}")
            check(n_data == 2, f"--dataset {ds}: data_file_path in both override blocks (found {n_data})")
            check(n_part == 2, f"--dataset {ds}: partition_file_path in both override blocks (found {n_part})")
            check(n_seq == 2, f"--dataset {ds}: max_seq_length={seq} in both override blocks (found {n_seq})")
            check(f"name: {ds}" in txt, f"--dataset {ds}: trainer.dataset.name set")
        spec = load_spec(logdir) if logdir else None
        check(spec is not None and check_level(spec, "partition group exists") == "ok",
              f"--dataset {ds}: partition group verified against {ds}'s own partition h5")

    # ---- 4. unknown dataset name: fails fast, no spec.json ----
    print("case: unknown --dataset name")
    rc, out, logdir = run_launcher(["--dry-run", "--only", "fluxtune", "--dataset", "not_a_real_dataset"])
    _cleanup_dirs.append(logdir)
    check(rc == 3, f"unknown --dataset exits 3, not 2 (--force can't fix a typo'd name) (got {rc})")
    check("unknown" in out.lower(), "unknown --dataset prints a clear error")
    check(logdir is None or load_spec(logdir) is None,
          "unknown --dataset writes no spec.json (fails before any cfg is generated)")

    # ---- 5. --dataset + a group absent from that dataset's h5: caught, not silently read ----
    print("case: --dataset yahoo + a group only agnews has")
    rc, out, logdir = run_launcher([
        "--dry-run", "--only", "fluxtune", "--dataset", "yahoo", "--force",
        "--partition-method", "niid_label_clients=100_alpha=100000",  # not a real group anywhere
    ])
    _cleanup_dirs.append(logdir)
    spec = load_spec(logdir) if logdir else None
    check(spec is not None and check_level(spec, "partition group exists") == "error",
          "a partition group absent from the resolved dataset's h5 is caught, not silently launched")

    # ---- 6. --dataset yahoo, sim mode, no --force: sim charge profile mismatch blocks ----
    print("case: --dataset yahoo, sim mode, no --force")
    rc, out, logdir = run_launcher(["--dry-run", "--only", "fluxtune", "--dataset", "yahoo", "--mode", "sim"])
    _cleanup_dirs.append(logdir)
    spec = load_spec(logdir) if logdir else None
    check(rc == 2, f"non-agnews sim dry-run without --force is BLOCKED (got rc={rc})")
    check(spec is not None and check_level(spec, "sim charge profile matches dataset") == "error",
          "sim charge profile / dataset mismatch is flagged, not silently reused from agnews")

    for d in _cleanup_dirs:
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
