#!/usr/bin/env bash
# harness_campaign.sh — the nightly campaign (T4) = harness_pool.py --tier T4 --pytest (FX-N22).
# Kept as a name only: gate, P0 pytest, the P1-P11 phases in parallel slots, and one report.
#   bash harness_campaign.sh [--datasets cifar10|google_speech|all] [--deadline-h 12] [--phases 'P1 P2'] [...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(conda run -n "${FLAME_CONDA_ENV:-dg_flame}" which python 2>/dev/null | tail -1)"
exec "$PY" "$SCRIPT_DIR/harness_pool.py" --tier T4 --pytest "$@"
