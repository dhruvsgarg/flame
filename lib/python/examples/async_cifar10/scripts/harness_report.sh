#!/usr/bin/env bash
# harness_report.sh ROOT [T0_epoch] [pytest_line] -- one SUMMARY.txt across <ROOT>/P*/summary.tsv (FX-N22):
# phase table, real queue_wait, FX-N13 oracle replay (P7/P7o), S1 injected-bug verdicts. Spawns no FL process.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$1"; T0="${2:-$(date +%s)}"; P0_LINE="${3:-}"
PY="$(conda run -n "${FLAME_CONDA_ENV:-dg_flame}" which python 2>/dev/null | tail -1)"
_elapsed() { echo $(( $(date +%s) - T0 )); }

# FX-N13: oracle replay + figures on the P7/P7o legs (no processes spawned).
if { [ -f "$ROOT/P7/summary.tsv" ] || [ -f "$ROOT/P7o/summary.tsv" ]; }; then
  echo "=== [$(date '+%F %T')] P7 analysis: oracle_misselection + felix_streaming_figures"
  _p7_dirs="$(cat "$ROOT"/P7/summary.tsv "$ROOT"/P7o/summary.tsv 2>/dev/null | awk -F'\t' 'NR>1 && $1!="trace" {print $11; print $12}' | grep -v '^$')"
  ( timeout --foreground 1800 "$PY" "$SCRIPT_DIR/oracle_misselection.py" $_p7_dirs \
    && timeout --foreground 600 "$PY" "$SCRIPT_DIR/felix_streaming_figures.py" --campaign "$ROOT" ) \
    > "$ROOT/P7_analysis.log" 2>&1
  echo "  P7 analysis rc=$? :: $ROOT/P7_figures/summary.txt"
fi

# One table across phases.
{
  echo "$ROOT  total=$(( $(_elapsed) / 60 ))m"
  [ -n "$P0_LINE" ] && echo "P0 pytest: $P0_LINE"
  echo
  printf 'phase\t'; head -1 "$(ls "$ROOT"/*/summary.tsv 2>/dev/null | head -1)" 2>/dev/null | cut -f1-10
  for f in "$ROOT"/*/summary.tsv; do
    [ -f "$f" ] || continue
    p="$(basename "$(dirname "$f")")"
    tail -n +2 "$f" | cut -f1-10 | sed "s/^/$p\t/"
  done
} | column -t -s $'\t' > "$ROOT/SUMMARY.txt"
# FX-N18: real-leg queue wait per row (P9 vs its P1 control).
{
  echo; echo "real queue_wait (arrival -> processing):"
  for f in "$ROOT"/*/summary.tsv; do
    [ -f "$f" ] || continue
    p="$(basename "$(dirname "$f")")"
    tail -n +2 "$f" | while IFS=$'\t' read -r tr bl _ _ _ _ _ _ _ _ real_dir _; do
      [ -d "$real_dir" ] || continue
      echo "$p $tr $bl :: $(timeout --foreground 120 "$PY" "$SCRIPT_DIR/analyze_send_recv_lag.py" "$real_dir" --queue-wait 2>&1 | tail -1)"
    done
  done
  [ -f "$ROOT/P7_figures/summary.txt" ] && { echo; echo "FX-N13 oracle replay (P7 vs P7o):"; cat "$ROOT/P7_figures/summary.txt"; }
  for pre in "" gs_; do
    ls -d "$ROOT/${pre}P11"* >/dev/null 2>&1 || continue
    echo; echo "S1 injected bugs ${pre:+($pre) }(sim must FAIL the named rung):"
    for pr in "P11a EV10" "P11b EV16" "P11c EV3"; do
      set -- $pr; f="$ROOT/$pre$1/summary.tsv"
      [ -f "$f" ] || { echo "$pre$1 $2 :: MISSING"; continue; }
      sim_ev="$(tail -n +2 "$f" | cut -f4 | head -1)"
      if [[ "$sim_ev" =~ (^|[:,])$2(,|$) ]]; then echo "$pre$1 $2 :: CAUGHT ($sim_ev)"; else echo "$pre$1 $2 :: MISSED ($sim_ev)"; fi
    done
  done
} >> "$ROOT/SUMMARY.txt"
