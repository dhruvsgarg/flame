"""Row E's saturation stop -- ONE definition of the detector.

Imported by the aggregator that RUNS it (`aggregator/FedSgdAggregator.py`) and
by the replay that SIZED it (`expt_scripts/replay_saturation_stop.py`), for the
same reason `landing_law.py` exists: a rule living in two files is one edit away
from disagreeing with its own evidence.

    GL_t = (Acc_best - Acc_t) / Acc_best      on an 11-eval trailing mean

Fire when `GL_t` exceeds `threshold` on `patience` consecutive evals **while the
trailing mean is no higher than it was one probe-cadence ago**, and only once the
run is past `warmup_commits`. Prechelt's generalization-loss criterion, read on
held-out accuracy instead of validation loss, with his progress term.

**The progress term is not optional and it is why this rule survives out of
sample.** GL with a running max is scale-free but not slope-aware: it cannot tell
"plateaued at the top" from "still grinding upward with noise", because a slow
climb dips below its own running max for 20 straight evals without having
saturated at all. Replayed on the six cached curves it was NOT sized on, GL alone
fires on `yahoo_control` at commit 597 and **0.153 below that run's eventual
peak** -- a run still crawling at 0.27 that went on to 0.42. Requiring the
trailing mean to have stopped RISING over `slope_horizon` commits removes that
fire and every other out-of-sample one, and costs the three sized curves
0.000 / 0.000 / 0.002 of accuracy.

**Why the SHAPE of the curve and never its level** (buildplan §6.7): a rule that
takes a target accuracy makes the target a knob and voids the zero-input claim.
This one only ever asks whether accuracy is still rising.

**The warm-up is a backstop, not what makes this work -- and that changed when
the progress term landed.** Before it the warm-up was load-bearing: at
`warmup`=300 the detector fired on yahoo at commit 346 and 0.24 accuracy, because
a running-max GL has a real maximum to compare against long before a run has
learned anything. The slope test now catches that case, and the warm-up is
**inert on all nine cached curves at every setting from 0 to 600**. It stays
because it costs nothing and a pathological curve could still need it, and it is
expressed as a MULTIPLE OF THE PROBE CADENCE (`3 x b_max_probe_every`) so that it
is not a constant read off N1-N3 (row E2). Do not cite it as the reason the rule
does not false-fire.

Sized by replay on N1-N3 and re-checked against six curves it was not sized on
(2026-08-22), reproduced by `expt_scripts/replay_saturation_stop.py`: fires at
commit 1,194 / 1,084 / 1,179 on agnews / yahoo / yelp-p, within 0.008 of each
run's peak, saving 38 / 55 / 37% of the run -- and on none of the six out-of-sample
curves does it fire before that run's own peak.
"""

SAT_GL_THRESHOLD = 0.005        # Prechelt GL, on chance-uncorrected accuracy
SAT_PATIENCE = 20               # consecutive breaching evals before firing
SAT_SMOOTH_WINDOW = 11          # trailing-mean width, the same one §5.5 scores on
SAT_WARMUP_CADENCE_MULT = 3     # warm-up = this x b_max_probe_every (row E2)
SAT_SLOPE_CADENCE_MULT = 1      # progress horizon = this x the same cadence
SAT_WARMUP_FALLBACK = 450       # when the probe is off there is no cadence to tie to
SAT_SLOPE_FALLBACK = 150        # ditto for the progress horizon


def warmup_commits(probe_every, mult=SAT_WARMUP_CADENCE_MULT):
    """The arming point, as a multiple of the probe cadence. Falls back to the
    same number the shipped cadence produces when the probe is disabled."""
    return int(mult * probe_every) if probe_every else SAT_WARMUP_FALLBACK


def slope_horizon_commits(probe_every, mult=SAT_SLOPE_CADENCE_MULT):
    """How far back "is it still rising?" looks. Same cadence, different
    multiple -- 1x, 1.5x and 2x all give the same out-of-sample verdict, so the
    multiple sits on a plateau; 0.5x does not and still fires early."""
    return int(mult * probe_every) if probe_every else SAT_SLOPE_FALLBACK


class SaturationDetector:
    """Streaming form of the replay. One `update` per held-out eval.

    Keeps only the trailing window and two scalars, so it costs nothing on the
    commit path (§6.3) and its state is exactly what the replay carries.
    """

    def __init__(self, warmup, threshold=SAT_GL_THRESHOLD,
                 patience=SAT_PATIENCE, window=SAT_SMOOTH_WINDOW,
                 slope_horizon=None):
        self.warmup = int(warmup)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.window = int(window)
        # Independent of `warmup` on purpose: deriving it from there couples
        # them, so a warm-up sweep silently sweeps this too (E2's gate caught it).
        self.slope_horizon = (int(slope_horizon) if slope_horizon is not None
                              else SAT_SLOPE_FALLBACK)
        self._recent = []
        self._hist = []              # (commit, smoothed) back over the horizon
        self._best = None
        self._breaches = 0
        self.fired_at = None

    def update(self, commit, acc):
        """Feed one eval; True the moment the run has saturated.

        The running best tracks from the FIRST eval, warm-up or not -- what the
        warm-up gates is firing, not measuring, so a run that peaks early is
        still compared against its own peak.
        """
        if acc is None or self.fired_at is not None:
            return self.fired_at is not None
        self._recent.append(float(acc))
        if len(self._recent) > self.window:
            self._recent.pop(0)
        smoothed = sum(self._recent) / len(self._recent)
        if self._best is None or smoothed > self._best:
            self._best = smoothed
        gl = (self._best - smoothed) / self._best if self._best > 0 else 0.0
        self._hist.append((commit, smoothed))
        # Keep one entry PAST the horizon: that is the one the comparison reads.
        while len(self._hist) > 2 and commit - self._hist[1][0] >= self.slope_horizon:
            self._hist.pop(0)
        if commit < self.warmup:
            self._breaches = 0
            return False
        breach = gl > self.threshold
        if breach and self.slope_horizon:
            # Prechelt's progress term: a run still climbing over the horizon has
            # not saturated, however far it has dipped below its own running max.
            breach = smoothed <= self._hist[0][1]
        self._breaches = self._breaches + 1 if breach else 0
        if self._breaches >= self.patience:
            self.fired_at = int(commit)
            return True
        return False

    @property
    def best(self):
        return self._best
