#!/usr/bin/env bash
# Phase 4 -- the zero-input runs (fl_fwd_ft_practice.md P5.2 phase 4).
#
#   run_node_p4.sh <agnews|yahoo|yelp-p> <controller|control>
#
# Six invocations. Within a dataset the pair is at the SAME vclock budget, which
# is what a controller-vs-control comparison needs; ACROSS datasets the budget
# moves, because bins/round differ 11.7x (150/1750/650) and yahoo's control was
# still climbing monotonically when killed at 86% of 40,000 (§6, P4.8):
#   node 1: run_node_p4.sh agnews controller     node 2: run_node_p4.sh agnews control
#   node 3: run_node_p4.sh yahoo  controller     node 4: run_node_p4.sh yahoo  control
#   then:   run_node_p4.sh yelp-p controller  /  run_node_p4.sh yelp-p control
#
# ACCEPTANCE (P5.2): both controller arms reach their plateau and END WITHIN
# 0.015 OF PEAK -- which a `halt` stop makes mechanically true, so the real test
# is that the sensed B_max, rho*, K and P DIFFER BETWEEN DATASETS with nobody
# having supplied them. T5 pre-registered the divergence: at the commit-150
# re-sense, agnews rho* 0.0530 -> 0.0764 (I 6 -> 12) against yahoo's 0.0530 ->
# 0.0573 (I 6 -> 7). Grep `[BmaxProbe]` first; that line IS the result.
#
# The `control` arms are today's shipped fluxtune_v2 (rm/0.25 at rho*=0.06) at
# the same vclock. They carry `--phi-stop log_only`, which emits the commit the
# stop WOULD have fired at without ending the run -- so the past-the-stop
# counterfactual that P4.1 was built from keeps being measured instead of being
# destroyed by the controller shipping.
#
# THE CONTROL USES gate_rho_ref=setpoint, NOT the shipped `annealed`. Not a
# preference -- `rm`/0.25 + `annealed` is bit-for-bit 003648's config, and T5
# says why that arm died: N_req ~ rho_t^2, so an annealing rho demands less
# pooling every commit until I floors at 1 and the per-commit server path has no
# trainer work amortising it. Projected here at 5,196 commits / 5,271 trips =
# 1.01 per commit and 7.5 h, versus `setpoint`'s 799 / 6,390 = 8.00 and 2.4 h
# for the SAME vclock. `setpoint` is also the only leg of G-2 that completed
# (084554), so it is the last configuration known to finish -- which is what a
# control has to do to be a control. P2.1 item 5b already makes `annealed`
# contingent on 3.3, and 3.3 is what the treatment arm is testing.
#
# THE COS AUDIT IS OFF ON ALL FOUR ARMS, deliberately. Phase 4 scores B, Phi,
# Lambda and A, every one of which is exact from rho; only `D` needs the audit,
# and the audit is 3.40s of a 7.81s commit (T5) -- the tax that killed G-2's
# annealed leg. Same setting on treatment and control, so the pair stays matched.
# Takes an EXPLICIT `--no-cos-ground-truth-audit`: v2's catalog sets it on.
#
# ALL FOUR ARMS ARE PINNED TO rf=16, not v2's catalog rf=64. T5 settled law C's
# constants against p=450,340; the v1/v2 split (2026-08-15) moved the `fluxtune`
# alias to p=118,348. Law C + `annealed` does not compose with the gate there
# under ANY T_res -- the Lambda>=0.95 and trips/commit>=3 floors leave a 9-unit
# window (T_res 82-90), and the two-phase trajectory reads 1.71 trips/commit on
# agnews against rf=16's 5.01. `annealed` is not negotiable (item 5b ships it
# CONTINGENT on 3.3, and 3.3 is this controller), so rf moves instead, on both
# arms: P4 scores law C against v2's iteration-control policy at rf=16, not
# against v2 entire. rf=64 refusing `annealed` is a standing blocker on 5b.
#
# Budget: T5 projects agnews 899 commits / 4,817 trips (~2.1 h) and yahoo 898 /
# 3,374 (~1.8 h) audit-off, against a 10 h slot. The vclock target is sized off
# G-2's measured 6.26 vclock-s per round trip, so the BUDGET stop fires well
# before max_runtime_s does -- if a controller arm dies on max_runtime_s instead
# of `[BudgetStop] reason=budget`, the law did not land and the arm is void.
#
# THE FIRST FOUR ARMS (2026-08-16) WERE VOID BY EXACTLY THAT RULE. Four defects,
# all fixed; readout in fl_fwd_ft_practice.md P4.7. Check all four inside the
# first 200 commits before leaving an arm unattended:
#
#   grep '\[DataBins\]'   -> 1750 on yahoo, 150 on agnews, and 'confirmed by trainer'
#   grep '\[BmaxProbe\]'  -> B_max moves UP from the ln 2 prior, never below B
#   no server_update with rho_star == 0 (was 23% of commits on agnews, 48% yahoo)
#   round trips / commit >= 3 in EVERY quintile, not just the launch projection
#
# Yahoo and yelp-p each still need `profile_sim_charges.py` off a real run of
# their own (yahoo burns 0.658 real-s/vclock-s against agnews' 0.255, hence the
# --force below, which lifts automatically once the profile exists). See P4.8 and
# buildplan §9 before reading a yahoo arm as a controller result -- the dataset is
# under-trained, not broken.
#
# BEFORE THE FIRST ARM ON A DATASET, pre-tokenize it: 100 trainers each tokenizing
# their own shard cost `234931` 32 of its 44 wall minutes, inside the arm's own
# budget (§10). `run_sequential.sh` warns when the cache is cold; the fix is
# `pretokenize_dataset.py --dataset $DATASET --clients 100`.
set -u
. "$(dirname "$0")/_node_lib.sh"

DATASET="${1:?usage: run_node_p4.sh <agnews|yahoo|yelp-p> <controller|control>}"
ARM="${2:?usage: run_node_p4.sh <agnews|yahoo|yelp-p> <controller|control>}"

# Identity at the pinned rf=16 (_FD_REF_P is 450,340); set so the pin is safe to lift.
export FWDLLM_FD_SCALE_INVARIANT=1

FW="$(cd "$(dirname "$0")/../.." && pwd)"

# Per-dataset budget, eval cost and real-wall ceiling. VCLOCK differs across a
# pair's datasets, never within a pair -- a controller-vs-control comparison is at
# equal vclock.
#
# THE CONTROL ARM IS THE LONG POLE, and the vclock budget is what sets it. The
# CONTROLLER stops at f*B_max -- T5 projects ~898 commits / ~2 h on all three,
# independent of the budget, because law C's length comes from (B_max, T_res, f).
# The control has no stop (`--phi-stop log_only`), so it runs the budget out:
# agnews measured 0.255 real-s per vclock-s (2.3 h at 40,000), yahoo 0.658 -- and
# that 0.658 is priced off an AGNEWS profile, so it is exactly what task B fixes.
#   yahoo   60,000: 1.5x the budget its control was still climbing through at 86%
#           (P4.8), and ~10 h at the mis-priced rate -- 80,000 would have blown
#           the ceiling. Re-cut it once fluxtune_yahoo.yaml prices it properly.
#   yelp-p  50,000: 650 bins/round against agnews' 150, at seq-256 cost per pass.
#   CEIL is the real-wall backstop; 12 h so it does NOT clip before the budget,
#   which would end the arm on the ceiling instead of on its own terms.
#   EVAL is a COST knob, not a learning one -- the seq-256 datasets' full test sets
#   blocked the commit loop on 359 of 359 fires (§6).
#
# SIZED FROM MEASURED RATE, not from the wall-clock preflight -- that prices every
# commit at a dataset-independent 4.41 s, and yahoo measured 45.4 (79 commits/h
# against agnews' 351), so it under-books the seq-256 datasets ~10x. Law C needs
# ~898 commits, and `check_arm_health.py`'s `budget sizing` line converts a short
# arm's rate into the vclock that buys them:
#   agnews  48,000: 130614 measured 17,523 vclock/h => 898 commits want 44,839.
#           40,000 was UNDER that -- the controller would have ended on
#           max_runtime_s, which is void.
#   yahoo   60,000: 125713 measured 5,470 vclock/h => 61,932 for 898 commits, at
#           11.3 h. But that arm ran WITHOUT --eval-max-samples and blocked on eval
#           359 of 359 times, so its rate is a floor. Re-read the smoke's own
#           `budget sizing` line before committing wave 2.
#   yelp-p  50,000: no arm has ever run -- this is a placeholder the smoke replaces.
case "$DATASET" in
  agnews) VCLOCK=48000; CEIL=10.0; EVAL=() ;;
  yahoo)  VCLOCK=60000; CEIL=14.0; EVAL=(--eval-max-samples 10000) ;;
  yelp-p) VCLOCK=50000; CEIL=14.0; EVAL=(--eval-max-samples 10000) ;;
  *) echo "unknown dataset '$DATASET' (want agnews|yahoo|yelp-p)" >&2; exit 2 ;;
esac

# SMOKE=1: the SAME code path at ~15 min, to prove four nodes survive unattended
# before committing hours to them. A smoke controller arm ends on max_runtime_s,
# not [BudgetStop] -- that is expected here and is the one gate a smoke cannot
# check. Everything else (DataBins, rho_star, trips/commit, the watchdog, the
# profile barrier) reads exactly as it will on the long run.
#
# The VCLOCK budget is what cuts a smoke short; the CEILING must stay ABOVE law C's
# own projection or the preflight refuses the arm outright. That projection is
# ~898 commits / 1.76 h of REAL wall and is independent of the vclock budget --
# law C's length comes from (B_max, T_res, f) -- so a small ceiling does not make
# a small run, it makes a blocked one. 2 h clears it honestly, with no --force.
if [ "${SMOKE:-0}" = "1" ]; then
  CEIL=2.0
  case "$DATASET" in
    agnews) VCLOCK=2500 ;;               # ~0.255 real-s/vclock-s => ~11 min
    *)      VCLOCK=1500 ;;               # seq 256 costs ~0.66 => ~16 min
  esac
  echo "### SMOKE: vclock=$VCLOCK ceiling=${CEIL}h -- plumbing check, NOT a result"
fi

# Last word, so the sanity ladder can cut a SHORT arm on the PRODUCTION path --
# SMOKE is a different ceiling and a different watch config, which is not what
# buildplan S2 is testing. Unset => the table above, byte-identical.
VCLOCK="${VCLOCK_OVERRIDE:-$VCLOCK}"
CEIL="${CEIL_OVERRIDE:-$CEIL}"
[ -n "${VCLOCK_OVERRIDE:-}${CEIL_OVERRIDE:-}" ] && \
  echo "### OVERRIDE: vclock=$VCLOCK ceiling=${CEIL}h"

# Runaway cap: the watchdog kills the arm an hour past its own ceiling, so a
# mis-priced vclock cannot silently eat a node for a day. Computed here, AFTER
# every branch that can move CEIL.
export NODE_WATCH_ARGS="${NODE_WATCH_ARGS:---max-hours $(awk "BEGIN{print $CEIL+1}")}"

# A missing profile REFUSES; it does not fall back. `--force` also disables
# `matches dataset`, so it would price yahoo on agnews' 0.255 real-s/vclock-s
# against its own 0.658 -- and `condition_fp` does not cover the profile, so a
# split pair reads one fp either way. `/home` is node-local: bring the file here.
# The override still overrides EVERY check, so per P9.2 --dry-run without it first.
FORCE=()
if [ ! -f "$FW/sim_charge_profiles/fluxtune_$DATASET.yaml" ] && [ "$DATASET" != "agnews" ]; then
  if [ "${P4_ALLOW_AGNEWS_PRICING:-0}" = "1" ]; then
    echo "!!! [p4-$DATASET] NO fluxtune_$DATASET.yaml -- pricing this arm on AGNEWS." >&2
    echo "!!! [p4-$DATASET] Its vclock is NOT comparable to an arm priced on $DATASET." >&2
    FORCE=(--force)
  else
    echo "!!! [p4-$DATASET] $FW/sim_charge_profiles/fluxtune_$DATASET.yaml is MISSING on this node." >&2
    echo "!!! [p4-$DATASET] Copy it here (md5-check it) or commit it; \$P4_ALLOW_AGNEWS_PRICING=1 overrides." >&2
    exit 2
  fi
fi

# --allow-stale-profile, NOT --force: the staleness check globs the LOCAL
# experiments dir, so the same profile passes on a node with no old reals and
# fails on one that has them -- kaylee blocked where jayne did not. Keeping
# fluxtune.yaml is also what makes these arms comparable to every historical
# agnews arm and to P4's calibration; re-profiling now would re-price them all.
# --force would additionally disable the dataset-match check, which is
# config-derived and is the one that actually protects the vclock.
COMMON=(--only fluxtune --mode sim --yes --clean --allow-stale-profile "${FORCE[@]}"
        --dataset "$DATASET" "${EVAL[@]}"
        --server-update-audit --no-cos-ground-truth-audit
        --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30
        --probe-combine mean --commit-gate n_target
        --server-step-rule trust_ratio --gate-safety-s "${P4_GATE_S:-1.5}"
        --adapter-reduction-factor 16 --max-iter-per-data-id 20
        --max-runtime-s "$VCLOCK" --sim-wall-ceiling-h "$CEIL")

# Tells _node_lib.sh's post-arm gate reader whether to require a [BudgetStop]
# ending -- gate 4 of §6, which applies to the controller only.
export NODE_ARM_KIND
case "$ARM" in
  controller)
    NODE_ARM_KIND=controller
    # Zero input: no --rho-star and no --b-max. rho* is derived from the ln 2
    # prior until 3.1's first probe lands, then from the sensed B_max.
    # Two overrides, both defaulting to the shipped behaviour. `anchor` uses the
    # LATEST sense instead of the mean: across 19 fires the probe reports a flat
    # B_rem (~0.25) while the mean turns it into headroom that vanishes, which
    # anneals rho* to 1.7x below what the current measurement supports.
    # `log_only` makes BOTH stops emit and keep training -- the only way to see
    # past the Phi=2.7 rail. P4_BMAX_PHIS re-ranges the probe grid; P4_GATE_S
    # moves `s`, the only lever the model allows on progress per unit budget.
    node_run "p4-$DATASET" "controller (law C, T_res=300, sensed B_max, ${P4_BMAX_POLICY:-mean}/${P4_PHI_STOP:-halt})" \
      "${COMMON[@]}" --gate-rho-ref annealed \
      --rho-schedule landing --t-res 300 --budget-stop-frac 0.95 \
      --b-max-policy "${P4_BMAX_POLICY:-mean}" \
      --phi-stop "${P4_PHI_STOP:-halt}" --b-max-probe-every 150 --b-max-probe-n 512 \
      ${P4_BMAX_PHIS:+--b-max-probe-phis "$P4_BMAX_PHIS"}
    ;;
  control)
    node_run "p4-$DATASET" "control (fluxtune_v2: rm/0.25, rho*=0.06, setpoint)" \
      "${COMMON[@]}" --gate-rho-ref setpoint \
      --rho-schedule rm --rho-exp 0.25 --rho-star 0.06 \
      --phi-stop log_only
    ;;
  *) echo "unknown arm '$ARM' (want controller|control)" >&2; exit 2 ;;
esac

echo "=== [p4-$DATASET] $ARM done $(date -Is) ==="
