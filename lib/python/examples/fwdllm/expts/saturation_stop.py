"""Row E's saturation stop -- ONE definition of the detector.

Imported by the aggregator that RUNS it (`aggregator/FedSgdAggregator.py`) and
by the replay that SIZED it (`expt_scripts/replay_saturation_stop.py`), for the
same reason `landing_law.py` exists: a rule living in two files is one edit away
from disagreeing with its own evidence.

    GL_t = (Acc_best - Acc_t) / Acc_best      on an 11-eval trailing mean

Fire when `GL_t` exceeds `threshold` on `patience` consecutive evals, and only
once the run is past `warmup_commits`. Prechelt's generalization-loss criterion,
read on held-out accuracy instead of validation loss.

**Why the SHAPE of the curve and never its level** (buildplan §6.7): a rule that
takes a target accuracy makes the target a knob and voids the zero-input claim.
This one only ever asks whether accuracy is still rising.

**The warm-up is load-bearing and is NOT fitted to three curves.** A running-max
GL has a real maximum to compare against long before a run has learned anything:
yahoo sits near chance for ~350 commits, so at `warmup`=300 the detector fires at
commit 346 and 0.24 accuracy. It is expressed as a MULTIPLE OF THE PROBE CADENCE
(`3 x b_max_probe_every` = 450 at the shipped 150), not as a constant read off
N1-N3 -- 400, 450 and 600 all give the same three fire commits, so the setting
sits on a plateau and the multiple is what is real (row E2).

Sized by replay on N1-N3 (2026-08-21), reproduced by
`expt_scripts/replay_saturation_stop.py`: fires at commit 1,184 / 1,084 / 1,126
on agnews / yahoo / yelp-p, within 0.008 of each run's peak, saving 38 / 55 / 40%
of the run.
"""

SAT_GL_THRESHOLD = 0.005        # Prechelt GL, on chance-uncorrected accuracy
SAT_PATIENCE = 20               # consecutive breaching evals before firing
SAT_SMOOTH_WINDOW = 11          # trailing-mean width, the same one §5.5 scores on
SAT_WARMUP_CADENCE_MULT = 3     # warm-up = this x b_max_probe_every (row E2)
SAT_WARMUP_FALLBACK = 450       # when the probe is off there is no cadence to tie to


def warmup_commits(probe_every, mult=SAT_WARMUP_CADENCE_MULT):
    """The arming point, as a multiple of the probe cadence. Falls back to the
    same number the shipped cadence produces when the probe is disabled."""
    return int(mult * probe_every) if probe_every else SAT_WARMUP_FALLBACK


class SaturationDetector:
    """Streaming form of the replay. One `update` per held-out eval.

    Keeps only the trailing window and two scalars, so it costs nothing on the
    commit path (§6.3) and its state is exactly what the replay carries.
    """

    def __init__(self, warmup, threshold=SAT_GL_THRESHOLD,
                 patience=SAT_PATIENCE, window=SAT_SMOOTH_WINDOW):
        self.warmup = int(warmup)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.window = int(window)
        self._recent = []
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
        if commit < self.warmup:
            self._breaches = 0
            return False
        self._breaches = self._breaches + 1 if gl > self.threshold else 0
        if self._breaches >= self.patience:
            self.fired_at = int(commit)
            return True
        return False

    @property
    def best(self):
        return self._best
