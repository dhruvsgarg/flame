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

    @property
    def breach(self) -> bool:
        return self.projected_real_wall > self.ceiling

    def explain(self) -> str:
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
            cos_probe_batch_size: Optional[int]) -> Projection:
    rho_star = rho_star if rho_star else RHO_STAR_DEFAULT
    gate_safety_s = gate_safety_s if gate_safety_s else GATE_SAFETY_S_DEFAULT
    perturbation_count = perturbation_count or PERTURBATION_COUNT_DEFAULT
    rule = rule or PROBE_COMBINE_DEFAULT

    G = g_rule(rule, perturbation_count)
    # matches aggregator/FedSgdAggregator.py:418's N_req closed form exactly
    n_req = p * (rho_star / gate_safety_s) ** 2 / G
    I = max(1, math.ceil(n_req / K))
    tau = tau_round_s(K)
    # (a) a low I is not a discount: the audit taxes each COMMIT, not each round.
    commits_projected = (vclock_budget_s / tau) * (K / n_req)

    # (b) unset cos_probe_every means stride 1, the expensive default.
    stride = cos_probe_every or 1
    batch = cos_probe_batch_size or COS_PROBE_BATCH_DEFAULT
    audit_cost = (AUDIT_S_PER_SAMPLE * batch / stride) if cos_audit_on else 0.0
    per_commit_cost = AUDIT_BASE_S + audit_cost
    projected_real_wall = commits_projected * per_commit_cost

    return Projection(n_req=n_req, I=I, tau=tau, commits_projected=commits_projected,
                       per_commit_cost=per_commit_cost, audit_cost=audit_cost,
                       stride=stride, projected_real_wall=projected_real_wall,
                       ceiling=real_wall_ceiling_s)
