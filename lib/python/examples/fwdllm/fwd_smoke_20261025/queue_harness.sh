#!/usr/bin/env bash
# Harness legs from a jobs file (name<TAB>yaml), one per free GPU pair: queue_harness.sh <jobs> <pair> [<pair> ...]
# A pair is free when no leg of this queue holds it and both GPUs use < 1 GB (other lanes may hold them).
set -u
B="$(cd "$(dirname "$0")" && pwd)"
jobs=$1; shift
pairs=("$@")
declare -A held
mapfile -t lines < <(grep -v '^#' "$jobs")
cpus_of () { local a=${1%,*}; echo "$((a * 8))-$((a * 8 + 15)),$((64 + a * 8))-$((64 + a * 8 + 15))"; }
free () {
  local p=$1 g
  [ -n "${held[$p]:-}" ] && kill -0 "${held[$p]}" 2>/dev/null && return 1
  for g in ${p//,/ }; do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -lt 1000 ] || return 1
  done
}
for l in "${lines[@]}"; do
  IFS=$'\t' read -r name y <<< "$l"
  while :; do
    for p in "${pairs[@]}"; do
      if free "$p"; then
        bash "$B/harness_leg.sh" "$y" "$name" "$p" "$(cpus_of "$p")" &
        held[$p]=$!
        echo "[queue] $name on GPUs $p $(date -Is)"
        sleep 90  # let the leg allocate before the next free check
        continue 3
      fi
    done
    sleep 30
  done
done
wait
echo "[queue] all done $(date -Is)"
