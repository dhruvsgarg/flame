#!/usr/bin/env bash
# Phase 4 -- the zero-input runs (fl_fwd_ft_practice.md P5.2 phase 4).
#
#   run_node_p4.sh <agnews|yahoo> <controller|control>
#
# Four invocations, one per node, all four at the SAME vclock budget:
#   node 1: run_node_p4.sh agnews controller     node 2: run_node_p4.sh agnews control
#   node 3: run_node_p4.sh yahoo  controller     node 4: run_node_p4.sh yahoo  control
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
# THE COS AUDIT IS OFF ON ALL FOUR ARMS, deliberately. Phase 4 scores B, Phi,
# Lambda and A, every one of which is exact from rho; only `D` needs the audit,
# and the audit is 3.40s of a 7.81s commit (T5) -- the tax that killed G-2's
# annealed leg. Same setting on treatment and control, so the pair stays matched.
#
# Budget: T5 projects agnews 899 commits / 4,817 trips (~2.1 h) and yahoo 898 /
# 3,374 (~1.8 h) audit-off, against a 10 h slot. The vclock target is sized off
# G-2's measured 6.26 vclock-s per round trip, so the BUDGET stop fires well
# before max_runtime_s does -- if a controller arm dies on max_runtime_s instead
# of `[BudgetStop] reason=budget`, the law did not land and the arm is void.
set -u
. "$(dirname "$0")/_node_lib.sh"

DATASET="${1:?usage: run_node_p4.sh <agnews|yahoo> <controller|control>}"
ARM="${2:?usage: run_node_p4.sh <agnews|yahoo> <controller|control>}"

# fluxtune_v2 is rf=64; without this the FD step is not rescaled for it and the
# launcher refuses (baselines.yaml cannot express an env var).
export FWDLLM_FD_SCALE_INVARIANT=1

# --force is required for a non-agnews SIM arm (every sim_charge_profiles/*.yaml
# was profiled on agnews, and per-pass cost scales with max_seq_length). It
# overrides EVERY check, so per P9.2 run --dry-run WITHOUT it first and confirm
# the sim-charge-profile mismatch is the ONLY x -- in particular that the
# wall-clock/gate-starved check reads ok.
FORCE=()
[ "$DATASET" = "agnews" ] || FORCE=(--force)

COMMON=(--only fluxtune --mode sim --yes --clean "${FORCE[@]}"
        --dataset "$DATASET"
        --server-update-audit
        --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30
        --probe-combine mean --commit-gate n_target
        --server-step-rule trust_ratio --gate-safety-s 1.5
        --gate-rho-ref annealed --max-iter-per-data-id 20
        --max-runtime-s 40000 --sim-wall-ceiling-h 10.0)

case "$ARM" in
  controller)
    # Zero input: no --rho-star and no --b-max. rho* is derived from the ln 2
    # prior until 3.1's first probe lands, then from the sensed B_max.
    node_run "p4-$DATASET" "controller (law C, T_res=300, sensed B_max)" \
      "${COMMON[@]}" \
      --rho-schedule landing --t-res 300 --budget-stop-frac 0.95 \
      --phi-stop halt --b-max-probe-every 150 --b-max-probe-n 512
    ;;
  control)
    node_run "p4-$DATASET" "control (shipped fluxtune_v2: rm/0.25, rho*=0.06)" \
      "${COMMON[@]}" \
      --rho-schedule rm --rho-exp 0.25 --rho-star 0.06 \
      --phi-stop log_only
    ;;
  *) echo "unknown arm '$ARM' (want controller|control)" >&2; exit 2 ;;
esac

echo "=== [p4-$DATASET] $ARM done $(date -Is) ==="
