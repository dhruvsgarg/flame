"""C-1's budget-landing anneal, "law C" -- ONE definition of the law.

Imported by the aggregator that RUNS it (`aggregator/FedSgdAggregator.py`), the
preflight that BUDGETS it (`expts/wall_clock_preflight.py`) and the replay that
SCORED it (`expt_scripts/replay_landing_law.py`). Kept here for the same reason
`dataset_registry` exists: the two failure modes this program keeps hitting are
"a superseded constant left in a config" and "an instrument whose arithmetic is
right and whose input is not" (fl_fwd_ft_buildplan.md §8). A law living in three
files is one edit away from both.

    rho*_t = min( rho_max, sqrt( 2 * (B_max - B_t) / T_res ) )

**`T_res` is a RATE, never decremented** -- so `B` approaches `B_max` as
`B_max*(1 - e^{-t/T_res})`, always from below, and a mid-run `B_max` re-sense
(3.1) has no horizon to reset. Decrementing it would reinstate `T` as an
operator input, which the model doc's §4.6a is named for forbidding, and would
buy nothing: `Lambda = 2B/s` is schedule-free (P3, measured to -0.3%).

Constants settled by T5 (2026-08-15) on replay against arms on disk, not fitted
to an accuracy curve. See fl_fwd_ft_buildplan.md §5 for what refuted each
alternative.
"""
import math
from dataclasses import dataclass

B_MAX_PRIOR = math.log(2.0)          # D1's Phase-A prior: "the weights may double"
T_RES_DEFAULT = 300.0                # T5: 500 floors the gate on yahoo and the prior
BUDGET_STOP_FRAC_DEFAULT = 0.95      # stop at B >= f*B_max; law C never reaches B_max
PHI_RAIL_DEFAULT = 3.0               # the retention floor; peaks land at 2.82-3.00 (§5.5)
TRIPS_PER_COMMIT_MIN = 3.0           # 003648 died at 1.02; 145729 survived at 8.0


def rho_gate_cap(gate_safety_s, p, g_rule, K, max_iter):
    """Largest `rho*` the commit gate can reach: `ceil(n_req/K) <= max_iter`
    solved for rho. Mechanical -- no operator input, nothing fitted.

    Deliberately NOT a `rho* <= rho*_0` clamp: `rho*_0` is derived from the ln 2
    PRIOR, so that clamp pins the step there and blocks 3.1's re-sense from ever
    spending the budget it just measured. Law C is monotone non-increasing at
    fixed `B_max` anyway, so only a re-sense can raise `rho*` -- exactly when it
    should.
    """
    if not (p and g_rule and max_iter):
        return None
    return gate_safety_s * math.sqrt(float(max_iter) * K * g_rule / p)


def rho_star_now(b_max, b_spent, t_res, rho_max=None):
    """One commit of law C. `B_rem` clamps at 0, so a `B_max` re-sensed below
    what is already spent returns 0 -- itself the correct stop, not a sqrt of a
    negative."""
    rho = math.sqrt(2.0 * max(0.0, b_max - b_spent) / t_res)
    return min(rho, rho_max) if rho_max else rho


def n_required(rho, gate_safety_s, p, g_rule):
    """S-C's closed form, the same one `FedSGDAggregator._n_required` applies."""
    return p * (rho / gate_safety_s) ** 2 / g_rule


def iterations(n_req, K, max_iter):
    """Round trips this commit's pool costs. Floors at 1 (a round trip is
    indivisible) and is capped by `max_iter`.

    The `-1e-9` is a tolerance, not cosmetics: `rho_gate_cap` solves
    `n_req == max_iter*K` exactly, and a bare `ceil` on the sqrt/square round
    trip returns `max_iter + 1` for the very rho the cap was built to permit.
    """
    want = math.ceil(n_req / K - 1e-9) if n_req > 0 else 1
    return max(1, min(want, max_iter)), want


@dataclass
class Trajectory:
    commits: int
    trips: int
    B: float
    Lam: float
    rho_0: float
    rho_T: float
    I_0: int
    floored: float
    capped: float

    @property
    def trips_per_commit(self):
        return self.trips / self.commits if self.commits else 0.0

    @property
    def gate_starved(self):
        """The metric that discriminates, per T5. Flooring `I` is SAFE
        (`N > n_req` means `rho/cos < s`, conservative) -- what it costs is the
        trainer work that was amortising the server-side per-commit path."""
        return self.trips_per_commit < TRIPS_PER_COMMIT_MIN


def simulate(*, b_max, t_res, gate_safety_s, p, g_rule, K, max_iter,
             stop_frac=BUDGET_STOP_FRAC_DEFAULT, gate_rho_ref="annealed",
             b_max_prior=None, resense_at=0, t_cap=20000):
    """The whole trajectory, closed form and arm-independent -- under
    trust_ratio the realised rho IS the setpoint (P3, to 8.7e-5).

    `b_max_prior`/`resense_at` model D1's two-phase start: run at the prior until
    `resense_at`, then 3.1's probe lands and `B_max` jumps. `T_res` never moves.
    """
    cap = rho_gate_cap(gate_safety_s, p, g_rule, K, max_iter)
    b = lam = 0.0
    trips = floored = capped = 0
    b_now = b_max_prior if b_max_prior is not None else b_max
    rho_0 = rho = rho_star_now(b_now, 0.0, t_res, cap)
    i_0 = None
    t = 0
    for t in range(t_cap):
        if b_max_prior is not None and t == resense_at:
            b_now = b_max
        rho = rho_star_now(b_now, b, t_res, cap)
        if rho <= 0:
            break
        ref = rho if gate_rho_ref == "annealed" else rho_star_now(b_now, 0.0, t_res, cap)
        n_req = n_required(ref, gate_safety_s, p, g_rule)
        i, want = iterations(n_req, K, max_iter)
        if i_0 is None:
            i_0 = i
        floored += (want <= 1)
        capped += (want > i)
        trips += i
        lam += rho * math.sqrt(g_rule * K * i / p)
        b += 0.5 * math.log1p(rho ** 2)
        if b >= stop_frac * b_now:
            break
    n = t + 1
    return Trajectory(commits=n, trips=trips, B=b, Lam=lam, rho_0=rho_0,
                      rho_T=rho, I_0=i_0 or 1, floored=floored / n,
                      capped=capped / n)
