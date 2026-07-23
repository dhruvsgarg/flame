"""Canonical real/sim parity checks — re-export shim.

All logic has moved to ``scripts/parity/checks.py``.  This file re-exports
every public name so that existing callers (pytest suite, compare_parity.py,
etc.) continue to work without modification.

See ``parity/checks.py`` for the full §3 battery including the new §3.H
clock/throughput checks (K1–K10) that catch the 410-vs-673 rounds regression.
"""

from __future__ import annotations

import sys
import os

# Make the parity sub-package importable when this file's directory is on sys.path
# (standard when running scripts directly or when tests add scripts/ to sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parity.checks import (  # noqa: F401, E402
    # helpers
    short,
    jaccard,
    mean_std,
    ks_stat,
    spearman_rho,
    percentile,
    pctl_band_ok,
    _matched_logical_budget,
    # loaders
    load_agg_jsonl,
    load_trainer_jsonl_dir,
    load_run_dir,
    # selection constants
    DETERMINISTIC_SELECTORS,
    _selector_name,
    _has_cohort_counts,
    _full_cohort_selection,
    _selection_is_deterministic,
    # §3.B selection
    selection_parity,
    # §3.D updates
    aggregation_sequence_parity,
    staleness_parity,
    commit_sequence,
    first_divergence,
    agg_goal_cycles_ok,
    inter_arrival_order_parity,
    # §3.E processing
    participation_parity,
    trainer_speed_parity,
    # §3.F utility
    utility_parity,
    # §3.G convergence
    convergence_parity,
    # §3.C sim invariants
    sim_send_ts_ok,
    gpu_budget_ok,
    timing_overrun,
    trainer_phase_parity,
    # Stage 0 / 1 / 2 / 4 / 8 additions (causal ladder)
    field_coverage,
    modeled_compute_advance,
    overhead_residual,
    avail_timebase_parity,
    duty_cycle_parity,
    training_budget_parity,
    trainer_phase_split,
    convergence_loss_parity,
    avail_composition_parity,
    eligibility_parity,
    eligible_speed_composition_parity,
    selection_detail_parity,
    # registry / verdict helpers
    CHECK_META,
    check_stage,
    check_role,
    # §3.H clock
    vclock_telemetry_present,
    sim_commit_order_monotone,
    sim_rate_ok,
    failsafe_ok,
    throughput_parity,
    per_round_advance_parity,
    overlap_factor,
    wall_disparity,
    sim_speedup,
    total_commits_parity,
    terminal_state_parity,
    budget_not_cap,
    # §F fwdllm variance-cadence layer (V/DK/G rungs)
    _iters_per_data_id,
    _moving_avg,
    cohort_sequence_parity,
    iters_per_data_id_parity,
    iters_per_data_id_moving_avg_parity,
    var_trajectory_parity,
    cached_v_pool_parity,
    force_commit_rate_parity,
    variance_pass_ratio_parity,
    agg_goal_trajectory_parity,
    dynamic_c_trajectory_parity,
    eligible_ends_metric_parity,
    grad_norm_parity,
    grad_pool_size_parity,
    # fwdllm async residence (R1/W1, simulate_fwdllm.md §L.3)
    _overlap_fraction,
    _forward_passes,
    _committed_grads,
    inflight_overlap_parity,
    compute_conservation_parity,
    drain_wall_budget_parity,
    trainer_phase_wall_budget_ok,
    _STEP_TIMING_REAL_ONLY_FUNCS,
    _AGG_STEP_TIMING_OFF_CRITICAL_PATH_FUNCS,
    step_timing_breakdown_parity,
    agg_step_timing_breakdown_parity,
    aggregation_compute_wall_parity,
    # overall
    run_all_parity,
    overall_verdict,
)
