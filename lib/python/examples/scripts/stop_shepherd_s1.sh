#!/usr/bin/env bash
# Offload watcher: stop shepherd's PR37 pool once every leg NOT moved to jayne (S1J) is DONE. Report-only unless --kill.
# usage: stop_shepherd_s1.sh [--kill]   (poll every 60 s; --kill sends SIGINT to the run_pr37 parent = one Ctrl+C, R20)
POOL=${POOL:-/home/dgarg39/flame/lib/python/examples/experiments/pool_20261016_0542_PR37_SHEPHERD_S1}
MOVED='^(G3_syn_50_3st_(felix|fedbuff|feddance|refl)|G3_syn_0_felix)_(real|sim|grade)$'
LOG=$POOL/pool.log
[ "$(mine_n=$(grep -cE "^\[.*\]   G3_" "$LOG" 2>/dev/null); echo ${mine_n:-0})" -ge 54 ] || { echo "ABORT: $LOG missing or lists <54 jobs"; exit 2; }
mine() { grep -E '^\[.*\]   G3_' "$LOG" | awk '{print $3}' | grep -Ev "$MOVED"; }
while :; do
  left=$(comm -23 <(mine | sort -u) <(grep ' DONE ' "$LOG" | awk '{print $4}' | sort -u))
  n=$(echo -n "$left" | grep -c .)
  echo "$(date +%H:%M) shepherd-owned legs not DONE: $n $(echo $left | cut -c1-200)"
  if [ "$n" = 0 ]; then
    echo "ALL shepherd-owned legs DONE; running (to be killed): $(pgrep -f 'harness_pool.py.*SHEPHERD_S1' | head -1)"
    if [ "$1" = --kill ]; then
      pkill -INT -f 'run_pr37.py --node shepherd'; sleep 60; pgrep -f 'harness_pool.py.*SHEPHERD_S1' >/dev/null && pkill -TERM -f 'harness_pool.py.*SHEPHERD_S1'
      echo "stopped"
    fi
    break
  fi
  sleep 60
done
