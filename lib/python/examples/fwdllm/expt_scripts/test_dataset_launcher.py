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
                                   -> launches (rc=0): fluxtune_yahoo.yaml landed
                                      2026-08-22, so there is no agnews fallback
                                      left to refuse (edge case d)
  6c. sim_charge_profile contract  -> a profiled name gets its sibling; an
                                      unprofiled one falls back to the agnews
                                      baseline, the state the preflight refuses
  6b. cold vs warm feature cache -> "feature cache warm" preflight fires (§10 F3)

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
    # fluxtune runs at rf=64, so the FD-rescale preflight (which landed after
    # this test) refuses without it, turning every case into an unrelated exit 2.
    env = dict(os.environ, FWDLLM_FD_SCALE_INVARIANT="1")
    p = subprocess.run(["bash", RUN_SEQ] + args, capture_output=True, text=True,
                       timeout=timeout, env=env)
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


def cache_root():
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")))
    from examples.fwdllm.expts.dataset_registry import cache_root as _cr
    return _cr()


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
    # §10 F1: cache_dir joins the registry's override block, so unset must not
    # inject it either -- the relative "cache_dir/" default is what byte-identical
    # means here, cwd-dependent and all.
    check(bool(txt) and "cache_dir" not in txt,
          "unset --dataset does not inject cache_dir (keeps the relative default)")

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
            n_cache = txt.count(f"cache_dir: {cache_root()}")
            check(n_cache == 2,
                  f"--dataset {ds}: absolute cache_dir in both override blocks "
                  f"(found {n_cache}) -- §10 F1, so the launch directory stops "
                  f"deciding which cache a process gets")
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

    # ---- 6. --dataset yahoo, sim mode, no --force: the profile RESOLVES ----------
    # Was exit 2, when yahoo fell back to agnews-profiled fluxtune.yaml. fluxtune_yahoo.yaml
    # landed 2026-08-22, so rc=0 is now correct; 6c still covers the guard's fallback.
    print("case: --dataset yahoo, sim mode, no --force")
    rc, out, logdir = run_launcher(["--dry-run", "--only", "fluxtune", "--dataset", "yahoo", "--mode", "sim"])
    _cleanup_dirs.append(logdir)
    spec = load_spec(logdir) if logdir else None
    check(rc == 0, f"yahoo sim dry-run launches on its OWN profile, no --force (got rc={rc})")
    check(spec is not None and check_level(spec, "sim charge profile matches dataset") != "error",
          "sim charge profile resolves to yahoo's own, so nothing is flagged")

    # ---- 6c. the fallback the guard exists for, at the resolver ------------------
    # Every registry dataset is profiled now, so no real name reaches the fallback.
    # Assert the resolver's contract directly instead.
    print("case: sim_charge_profile fallback contract")
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
    from examples.fwdllm.expts import dataset_registry as dsreg
    base = "lib/python/examples/fwdllm/sim_charge_profiles/fluxtune.yaml"
    for name, want in (("yahoo", "fluxtune_yahoo.yaml"), ("yelp-p", "fluxtune_yelp-p.yaml")):
        got = dsreg.sim_charge_profile(base, name)
        check(got.endswith(want), f"--dataset {name} resolves to {want} (got {os.path.basename(got)})")
    check(dsreg.sim_charge_profile(base, "not_a_dataset") == base,
          "an unprofiled dataset falls back to the agnews baseline -- what the preflight refuses")

    # ---- 6b. the cold-cache preflight (§10 F3) reads the resolved key ----------
    # Cold is a WARN, not an error: the run tokenizes its way out. But it does so
    # inside its OWN wall budget (32 of 234931's 44 minutes), which nothing else
    # at launch time can see.
    print("case: feature cache warm/cold preflight")
    for ds, want_level in (("yahoo", "ok"), ("yelp-p", "warn")):
        rc, out, logdir = run_launcher(["--dry-run", "--only", "fluxtune",
                                        "--dataset", ds, "--force"])
        _cleanup_dirs.append(logdir)
        spec = load_spec(logdir) if logdir else None
        lvl = check_level(spec, "feature cache warm") if spec else None
        check(lvl is not None, f"--dataset {ds}: the cache preflight ran")
        if lvl is not None and lvl != want_level:
            print(f"  [note] {ds} cache preflight is '{lvl}', expected "
                  f"'{want_level}' -- run pretokenize_dataset.py --dataset {ds}")

    # ---- 7. data bins are the dataset's own, not agnews' hardcoded 150 -------
    # The aggregator's data_id range IS the trainer's batch index, so a value
    # below the true batch count silently trains on a prefix of every shard:
    # yahoo ran on 1,200 of each client's 14,000 samples (8.6%) on 125713/125753.
    print("case: total_data_bins per dataset")
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")))
    from examples.fwdllm.expts.dataset_registry import data_coverage, total_data_bins
    for ds, want in (("agnews", 150), ("yahoo", 1750), ("yelp-p", 650)):
        got = total_data_bins(ds, 100, 8)
        check(got == want, f"{ds} at C=100/batch=8 -> {want} bins (got {got})")
    check(total_data_bins("agnews", 100, 8) == 150,
          "agnews still resolves to today's hardcoded 150 (byte-identical)")
    # The property the bin count exists to give: bins x batch x C == n_train,
    # at BOTH client counts on disk. Batch is pinned at 8 (it is the unit each
    # JVP is estimated on); the bin count is what moves per dataset.
    for ds in ("agnews", "yahoo", "yelp-p"):
        for C in (100, 1000):
            cov = data_coverage(ds, C, 8)
            check(cov["exact"],
                  f"{ds} C={C}: {cov['bins']}x8x{C} = {cov['reached']:,} "
                  f"== n_train {cov['n_train']:,}")

    for d in _cleanup_dirs:
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
