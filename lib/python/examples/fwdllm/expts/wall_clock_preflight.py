"""Wall-clock budget preflight (fl_fwd_ft_buildplan.md 0.7): projects the REAL
wall-clock cost of an arm's commit path -- dominated by the cos-ground-truth
audit -- and refuses launch when it would blow the REAL wall ceiling. This is
the failure mode that killed eight arms (fl_fwd_ft_practice.md P4/P9.2): the
audit runs per COMMIT, not per round, and vclock time (what `max_runtime_s`
budgets in sim mode) and REAL wall time (what `sim_wall_ceiling_s` caps)
diverge once the audit is on -- a run can look vclock-healthy while its real
wall clock is already exhausted.

    from examples.fwdllm.expts.wall_clock_preflight import project
    p = project(p=450340, rho_star=0.06, gate_safety_s=2.9, rule="mean",
                perturbation_count=10, K=10, vclock_budget_s=14400,
                real_wall_ceiling_s=7200, cos_audit_on=True,
                cos_probe_every=None, cos_probe_batch_size=None)
    p.breach   # True
"""
import math
from dataclasses import dataclass
from typing import Optional

from examples.fwdllm.expts.landing_law import (
    B_MAX_PRIOR, BUDGET_STOP_FRAC_DEFAULT, T_RES_DEFAULT, TRIPS_PER_COMMIT_MIN,
    simulate,
)

# E[v_par^2] under `select`, measured per P -- never derived, never interpolated
# (same table as expt_scripts/replay_scoring.py's G_rule_t, buildplan 0.4).
E_SELECT_BY_P = {10: 2.988, 30: 4.744}

AUDIT_BASE_S = 1.64                     # rest of the commit path, per commit
AUDIT_S_PER_SAMPLE = 85.0 / 1024        # linear in the reference (measured 77-81 ms/sample)
COS_PROBE_BATCH_DEFAULT = 1024          # B17's shipped default

# `tau(K)`: real seconds per round trip. Calibrated at ONE measured point (K=30
# -> 8.9s; fl_fwd_ft_practice.md P3 "K at fixed n_req") and extrapolated with
# the POOLING MODEL K-1 is the registered A/B for (fl_fwd_ft_practice.md P5.1).
# This is a preflight ESTIMATE, not a validated law -- it exists to catch a
# gross breach before launch, not to predict commit rate precisely.
TAU_REF_S = 8.9
TAU_REF_K = 30
TAU_EXPONENT = 0.63

# T5 (2026-08-15): least squares over 5 arms, every one predicted to <=1%.
# `wall = COST_PER_COMMIT*commits + COST_PER_TRIP*trips` with the cos audit ON
# at stride 25; the audit is charged per COMMIT (85s/25), so subtracting it
# leaves the audit-off per-commit path. Used only by the `landing` branch --
# the const/rm branch keeps the model already validated against four arms.
COST_PER_COMMIT_AUDIT_OFF_S = 4.41
COST_PER_TRIP_S = 0.77

GATE_SAFETY_S_DEFAULT = 0.4             # aggregator/FedSgdAggregator.py code default
RHO_STAR_DEFAULT = 0.01                 # aggregator/FedSgdAggregator.py code default
PERTURBATION_COUNT_DEFAULT = 10
PROBE_COMBINE_DEFAULT = "select"


def g_rule(rule: str, perturbation_count: int) -> float:
    """G_rule: P under `mean`, measured E_select(P) under `select`. Refuses
    rather than interpolates for an unmeasured P (buildplan 0.4)."""
    if rule == "mean":
        return float(perturbation_count)
    e = E_SELECT_BY_P.get(perturbation_count)
    if e is None:
        raise ValueError(
            f"E_select unmeasured for P={perturbation_count} under `select` -- "
            f"only {sorted(E_SELECT_BY_P)} are measured; refusing to interpolate")
    return e


def tau_round_s(K: int) -> float:
    return TAU_REF_S * (K / TAU_REF_K) ** TAU_EXPONENT


@dataclass
class Projection:
    n_req: float
    I: int
    tau: float
    commits_projected: float
    per_commit_cost: float
    audit_cost: float
    stride: int
    projected_real_wall: float
    ceiling: float
    trips: Optional[int] = None
    trips_per_commit: Optional[float] = None
    schedule: str = "const"

    @property
    def gate_starved(self) -> bool:
        """T5: `I` floored at 1 is safe but strands the per-commit cost with no
        trainer work amortising it -- how 003648 died (1.02 trips/commit,
        8.40 s/trip) while looking vclock-healthy. Only law C can drift here,
        since only it anneals rho without bound."""
        return (self.trips_per_commit is not None
                and self.trips_per_commit < TRIPS_PER_COMMIT_MIN)

    @property
    def breach(self) -> bool:
        return self.projected_real_wall > self.ceiling or self.gate_starved

    def explain(self) -> str:
        if self.schedule == "landing":
            why = ("gate starved: " if self.gate_starved else "")
            return (
                f"{why}law C over {self.commits_projected:,.0f} commits / "
                f"{self.trips:,} round trips "
                f"({self.trips_per_commit:.2f} per commit, floor "
                f"{TRIPS_PER_COMMIT_MIN:g}; rho*_0={self.n_req:.4f} is rho, not n_req) x "
                f"({COST_PER_COMMIT_AUDIT_OFF_S}s/commit + audit "
                f"{self.audit_cost:.2f}s/stride-{self.stride} + "
                f"{COST_PER_TRIP_S}s/trip) = {self.projected_real_wall:,.0f}s real "
                f"wall, vs ceiling {self.ceiling:,.0f}s"
            )
        return (
            f"commits_projected={self.commits_projected:,.0f} "
            f"(n_req={self.n_req:.1f}, I={self.I}, tau(K)={self.tau:.2f}s) x "
            f"per_commit_cost={self.per_commit_cost:.2f}s "
            f"(base={AUDIT_BASE_S}s + audit={self.audit_cost:.2f}s/stride-{self.stride}) "
            f"= {self.projected_real_wall:,.0f}s real wall, vs ceiling {self.ceiling:,.0f}s"
        )


def project(*, p: int, rho_star: Optional[float], gate_safety_s: Optional[float],
            rule: str, perturbation_count: Optional[int], K: int,
            vclock_budget_s: float, real_wall_ceiling_s: float,
            cos_audit_on: bool, cos_probe_every: Optional[int],
            cos_probe_batch_size: Optional[int],
            rho_schedule: Optional[str] = None, b_max: Optional[float] = None,
            t_res: Optional[float] = None,
            budget_stop_frac: Optional[float] = None,
            max_iter: Optional[int] = None,
            gate_rho_ref: Optional[str] = None) -> Projection:
    rho_star = rho_star if rho_star else RHO_STAR_DEFAULT
    gate_safety_s = gate_safety_s if gate_safety_s else GATE_SAFETY_S_DEFAULT
    perturbation_count = perturbation_count or PERTURBATION_COUNT_DEFAULT
    rule = rule or PROBE_COMBINE_DEFAULT

    G = g_rule(rule, perturbation_count)
    stride = cos_probe_every or 1
    batch = cos_probe_batch_size or COS_PROBE_BATCH_DEFAULT
    audit_cost = (AUDIT_S_PER_SAMPLE * batch / stride) if cos_audit_on else 0.0

    if (rho_schedule or "").lower() == "landing":
        # Law C's commit count is set by (B_max, T_res, f) -- NOT by the vclock
        # budget divided by a constant-rho round-trip rate. Pricing it off
        # `rho_star` (unset => the 0.01 code default) projects a ~45,000-commit
        # phantom run and refuses every launch.
        traj = simulate(
            b_max=b_max or B_MAX_PRIOR, t_res=t_res or T_RES_DEFAULT,
            gate_safety_s=gate_safety_s, p=p, g_rule=G, K=K,
            max_iter=max_iter or 20,
            stop_frac=budget_stop_frac or BUDGET_STOP_FRAC_DEFAULT,
            gate_rho_ref=(gate_rho_ref or "annealed"),
        )
        wall = (traj.commits * (COST_PER_COMMIT_AUDIT_OFF_S + audit_cost)
                + traj.trips * COST_PER_TRIP_S)
        return Projection(
            n_req=traj.rho_0, I=traj.I_0, tau=tau_round_s(K),
            commits_projected=traj.commits, per_commit_cost=(
                COST_PER_COMMIT_AUDIT_OFF_S + audit_cost),
            audit_cost=audit_cost, stride=stride, projected_real_wall=wall,
            ceiling=real_wall_ceiling_s, trips=traj.trips,
            trips_per_commit=traj.trips_per_commit, schedule="landing")

    # matches aggregator/FedSgdAggregator.py:418's N_req closed form exactly
    n_req = p * (rho_star / gate_safety_s) ** 2 / G
    I = max(1, math.ceil(n_req / K))
    tau = tau_round_s(K)
    # (a) a low I is not a discount: the audit taxes each COMMIT, not each round.
    commits_projected = (vclock_budget_s / tau) * (K / n_req)

    # (b) unset cos_probe_every means stride 1, the expensive default.
    per_commit_cost = AUDIT_BASE_S + audit_cost
    projected_real_wall = commits_projected * per_commit_cost

    return Projection(n_req=n_req, I=I, tau=tau, commits_projected=commits_projected,
                       per_commit_cost=per_commit_cost, audit_cost=audit_cost,
                       stride=stride, projected_real_wall=projected_real_wall,
                       ceiling=real_wall_ceiling_s)
