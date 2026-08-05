#!/usr/bin/env bash
# One baseline's whole frozen-commit replicate block, on ONE node, unattended:
#
#   R (x3 real) -> CH (re-profile charges FROM those reals) -> S (x3 sim)
#     -> FL (two-sided floors) -> GR (board) -> CTL (control, both modes)
#
# NEVER reorder. The charge re-profile sits BETWEEN the sides because a stale
# profile moves sim's own cadence ~19 points, so the run would grade the profile
# rather than the code (simulate_fwdllm.md D-50). Legs run strictly one at a
# time: `run_sequential.sh` blocks until each finishes, and refuses to launch on
# top of a prior run's stray workers.
#
# Usage:  bash run_block.sh <baseline> [--dry-run]
# Detached, one node, one baseline:
#   setsid nohup bash run_block.sh fwdllm > ~/block1_fwdllm.log 2>&1 &
set -u
B="${1:-}"
DRY="${2:-}"
[ -n "$B" ] || { echo "usage: $0 <baseline> [--dry-run]" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
PY=/coc/scratch/dgarg/miniconda3/envs/dg_flame/bin/python
DUR=7200
say() { echo "=== [$(date '+%F %T')] $B: $*"; }

say "START  commit $(git rev-parse --short HEAD 2>/dev/null)  clean=$([ -z "$(git status --porcelain 2>/dev/null)" ] && echo true || echo FALSE)"
if [ -z "$(git status --porcelain 2>/dev/null)" ]; then :; else
  say "ABORT — tree is dirty; the batch must record a real commit (D-70)"; exit 1
fi

# ---- R: three real legs, one at a time ----
for i in 1 2 3; do
  say "REAL leg $i/3"
  bash run_sequential.sh --mode real --max-runtime-s "$DUR" --only "$B" --yes $DRY \
    || { say "ABORT — real leg $i failed"; exit 1; }
done

# ---- CH: re-profile charges from THOSE three reals ----
say "CH re-profile from the 3 newest reals"
REALS=$(ls -d ../experiments/*_"${B}"_n100_*_real 2>/dev/null | sort | tail -3)
[ "$(echo "$REALS" | grep -c .)" -eq 3 ] || { say "ABORT — expected 3 real legs, got $(echo "$REALS" | grep -c .)"; exit 1; }
if [ -z "$DRY" ]; then
  cp "../sim_charge_profiles/$B.yaml" "../sim_charge_profiles/$B.yaml.bak" 2>/dev/null
  # shellcheck disable=SC2046
  "$PY" -u profile_sim_charges.py $(echo "$REALS" | sed 's|^|--real-run |') \
      --out "../sim_charge_profiles/$B.yaml" --only-observed \
    || { say "ABORT — charge re-profile failed; sim would grade a stale profile (D-50)"; exit 1; }
fi

# ---- S: three sim legs, AFTER the re-profile ----
for i in 1 2 3; do
  say "SIM leg $i/3"
  bash run_sequential.sh --mode sim --max-runtime-s "$DUR" --only "$B" --yes $DRY \
    || { say "ABORT — sim leg $i failed"; exit 1; }
done
[ -z "$DRY" ] || { say "DRY-RUN OK — chain is wired, nothing launched"; exit 0; }

# ---- FL + GR + CTL: analysis. Never `&&` a grader that exits 1 on findings (D-51) ----
say "FL real-side floor"
"$PY" -u replicate_floor.py --mode real --duration "$DUR" --baselines "$B" --profile-out ../parity_floors
say "FL sim-side floor — the two-sided number this block exists to buy"
"$PY" -u replicate_floor.py --mode sim  --duration "$DUR" --baselines "$B"
say "GR board"
"$PY" -u run_parity.py --yes --baselines "$B" || true
say "CTL control, both modes — sim<->sim reads the three vclock rungs (D-72)"
"$PY" -u run_parity.py --control --control-mode both --duration "$DUR" --yes --baselines "$B" || true
say "DONE"
