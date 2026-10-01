#!/usr/bin/env bash
# harness_report.sh ROOT [T0_epoch] [pytest_line] -- one SUMMARY.txt across <ROOT>/P*/summary.tsv (FX-N22):
# phase table, real queue_wait, FX-N13 oracle replay (P7/P7o), S1 injected-bug verdicts. Spawns no FL process.
set -uo pipefail
# The analysis tools live with the example that produced the legs (async_cifar10 for both datasets).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../async_cifar10/scripts" && pwd)"
ROOT="$1"; T0="${2:-$(date +%s)}"; P0_LINE="${3:-}"
PY="$(conda run -n "${FLAME_CONDA_ENV:-dg_flame}" which python 2>/dev/null | tail -1)"
_elapsed() { echo $(( $(date +%s) - T0 )); }

# FX-N13: oracle replay + figures on the P7/P7o legs, per dataset prefix (no processes spawned).
for pre in "" gs_; do
  [ -f "$ROOT/${pre}P7/summary.tsv" ] || [ -f "$ROOT/${pre}P7o/summary.tsv" ] || continue
  echo "=== [$(date '+%F %T')] ${pre}P7 analysis: oracle_misselection + felix_streaming_figures"
  _p7_dirs="$(cat "$ROOT/${pre}P7/summary.tsv" "$ROOT/${pre}P7o/summary.tsv" 2>/dev/null | awk -F'\t' 'NR>1 && $1!="trace" {print $11; print $12}' | grep -v '^$')"
  ( timeout --foreground 1800 "$PY" "$SCRIPT_DIR/oracle_misselection.py" $_p7_dirs \
    && timeout --foreground 600 "$PY" "$SCRIPT_DIR/felix_streaming_figures.py" --campaign "$ROOT" --prefix "$pre" ) \
    > "$ROOT/${pre}P7_analysis.log" 2>&1
  echo "  ${pre}P7 analysis rc=$? :: $ROOT/${pre}P7_figures/summary.txt"
done

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
  # FX-N35: the UN_AVL share each leg saw at selection (real / sim).
  echo; echo "effective unavailability (UN_AVL share at selection, real / sim):"
  for f in "$ROOT"/*/summary.tsv; do
    [ -f "$f" ] || continue
    p="$(basename "$(dirname "$f")")"
    # awk, not `read`: a tab IFS merges empty columns (sim-only rows)
    tail -n +2 "$f" | awk -F'\t' '{print $1, $2, ($11 == "" ? "-" : $11), ($12 == "" ? "-" : $12)}' |
    while read -r tr bl real_dir sim_dir; do
      u_r="$([ -d "$real_dir" ] && "$PY" "$SCRIPT_DIR/leg_unavailability.py" "$real_dir" | cut -d' ' -f1 || echo -)"
      u_s="$([ -d "$sim_dir" ] && "$PY" "$SCRIPT_DIR/leg_unavailability.py" "$sim_dir" | cut -d' ' -f1 || echo -)"
      echo "$p $tr $bl :: $u_r / $u_s"
    done
  done
  for pre in "" gs_; do
    [ -f "$ROOT/${pre}P7_figures/summary.txt" ] || continue
    echo; echo "FX-N13 oracle replay ${pre:+($pre) }(P7 vs P7o):"; cat "$ROOT/${pre}P7_figures/summary.txt"
  done
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
