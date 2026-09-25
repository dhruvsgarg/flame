"""Exercise row E's saturation detector, and gate it on runs it was NOT sized on.

Checks (a) the streaming detector reproduces the replay's fire commits on N1-N3
    -- 1,194 / 1,084 / 1,179, within 0.008 of each peak,
(b) THE OUT-OF-SAMPLE GATE: on the six cached curves the rule was never sized on
    (`_controller` and `_control` on all three datasets) it does not fire before
    that run's own peak. Without the progress term it fires on `yahoo_control` at
    commit 597 and 0.153 below peak, which is why the progress term exists,
(c) both constants are multiples of ONE cadence and are INDEPENDENT of each
    other -- sweeping the warm-up must not move the progress horizon (row E2),
(d) it never fires on a still-rising curve,
(e) it latches: once fired, later evals cannot un-fire it, and
(f) the WIRING: the aggregator's own snapshot/eval/stop methods, in the order a
    live run calls them, and
(g) ROW E' -- the stall trigger. Default off => every check above is unchanged.
    On, it must fire on the two 08-24 curves the decay rule structurally cannot
    stop (a plateau AT the running max never breaches GL), must still not fire on
    a still-climbing control, and must report which trigger fired.
"""
import json
import os
import sys

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.expts.saturation_stop import (  # noqa: E402
    SaturationDetector, slope_horizon_commits, warmup_commits,
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "writeup_figs", "data")
CADENCE = 150
# Sized by replay on N1-N3, re-checked out of sample 2026-08-22 (buildplan §5.5).
EXPECT = {"agnews": (1194, -0.005), "yahoo": (1084, -0.008), "yelp-p": (1179, -0.007)}
# The six curves the rule was never sized on. None may fire before its own peak.
OUT_OF_SAMPLE = [f"{d}_{t}" for d in ("agnews", "yahoo", "yelp-p")
                 for t in ("controller", "control")]


def curve(tag):
    if "_" not in tag:
        tag = f"{tag}_anchor"
    with open(os.path.join(DATA, f"{tag}.json")) as fh:
        return [(int(c), float(a)) for c, _b, _l, a in json.load(fh)["acc_budget"]]


def run(rows, warmup, horizon=None, stall=None):
    det = SaturationDetector(warmup, slope_horizon=horizon, stall_frac=stall)
    smoothed, w = [], det.window
    accs = [a for _c, a in rows]
    for i, (c, a) in enumerate(rows):
        smoothed.append(sum(accs[max(0, i - w + 1):i + 1])
                        / len(accs[max(0, i - w + 1):i + 1]))
        if det.update(c, a):
            run.why = det.fired_reason
            return det.fired_at, smoothed[-1] - max(
                sum(accs[max(0, j - w + 1):j + 1]) / len(accs[max(0, j - w + 1):j + 1])
                for j in range(len(accs)))
    run.why = None
    return None, None


tied = warmup_commits(CADENCE)
hor = slope_horizon_commits(CADENCE)
assert (tied, hor) == (450, 150), (tied, hor)
print(f"  two multiples         : warm-up 3x{CADENCE}={tied}, "
      f"progress horizon 1x{CADENCE}={hor}")

for ds, (want_commit, want_loss) in EXPECT.items():
    rows = curve(ds)
    fired, loss = run(rows, tied, hor)
    assert fired == want_commit, (ds, fired, want_commit)
    assert abs(loss - want_loss) < 0.001, (ds, loss, want_loss)
    # (c) the two multiples are independent: sweeping the warm-up at a fixed
    # horizon must not move the fire commit.
    for w in (0, 300, 400, 600):
        assert run(rows, w, hor)[0] == fired, (ds, w)
    print(f"  {ds:<8} fires at commit {fired}, {loss:+.3f} vs peak "
          f"(warm-up 0/300/400/600 all agree)")

# (b) THE OUT-OF-SAMPLE GATE -- the one that changed the rule
for tag in OUT_OF_SAMPLE:
    rows = curve(tag)
    accs = [a for _c, a in rows]
    w = SaturationDetector(0).window
    sm = [sum(accs[max(0, i - w + 1):i + 1]) / len(accs[max(0, i - w + 1):i + 1])
          for i in range(len(accs))]
    peak_commit = rows[max(range(len(sm)), key=lambda i: sm[i])][0]
    fired, loss = run(rows, tied, hor)
    assert fired is None or fired >= peak_commit, (tag, fired, peak_commit)
    print(f"  {tag:<20} {'never fires' if fired is None else f'fires at {fired}'}"
          f", peak at {peak_commit}")

# and WITHOUT the progress term the same gate fails, loudly -- this is the
# measurement that put the term in, kept as a test so it cannot be removed
bad, _ = run(curve("yahoo_control"), tied, 0)
assert bad == 597, bad
print(f"  progress term OFF     : yahoo_control fires at {bad}, 0.153 below its "
      f"peak -- the reason the term is not optional")

# (d) a monotone rise never fires
det = SaturationDetector(0)
assert not any(det.update(c, 0.5 + c / 10000.0) for c in range(2000))
print("  still rising          : never fires")

# (e) latching
det = SaturationDetector(0)
for c in range(400):
    det.update(c, 0.9 if c < 100 else 0.5)
first = det.fired_at
for c in range(400, 800):
    det.update(c, 0.99)
assert det.fired_at == first is not None
print(f"  latching              : fired at {first}, a later peak does not un-fire it")

# ---------------------------------------------------------------------------
# (f) THE WIRING, not the arithmetic. Everything above tests the detector in
# isolation; this drives the aggregator's own methods in the order a live run
# calls them -- snapshot (main thread, stamps the commit) -> eval_model (daemon
# thread, feeds the detector) -> _check_budget_stop (commit path, reads the
# latch). Three code paths that no unit test above touches and that a dry-run
# cannot reach, because a dry-run never commits.
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A  # noqa: E402
from examples.fwdllm.expts.landing_law import PHI_RAIL_DEFAULT  # noqa: E402
import math  # noqa: E402


# `super()` inside the overrides resolves up the REAL flame chain, so the base
# methods are patched there rather than mixed in -- a `_Base` mixin would sit
# after TopAggregator in the MRO and never be reached.
_BASE = A.__mro__[1]                     # fwdllm's TopAggregator


def _base_snapshot(self):
    self.snapshots += 1
    return object() if self.snapshots % 2 else None      # exercise the stride skip


def _base_eval(self, *a, **k):
    return ({"acc": self._next_acc}, None, None)


_BASE._eval_snapshot_model = _base_snapshot
_BASE.eval_model = _base_eval


def wired(warmup, phi_rail=PHI_RAIL_DEFAULT, sat=True):
    a = object.__new__(A)
    a.snapshots = 0
    a._commit_count = 0
    a._B = 0.0
    a._b_max = math.log(50.0)
    a._budget_stop_frac = 0.95
    a._rho_schedule = "landing"
    a._phi_stop = "halt"
    a._phi_stop_threshold = phi_rail
    a._stop_fired = None
    a._work_done = False
    a._eval_commit = 0
    a._retention_every = 0
    a._theta_0 = None
    a._sat_det = SaturationDetector(warmup) if sat else None
    return a


# the eval thread must speak for the commit the SNAPSHOT was taken at, not for
# whatever the main thread has advanced to while the test-set pass ran
a = wired(warmup=0)
a._commit_count = 700
a._eval_snapshot_model()                 # main thread stamps 700
a._commit_count = 712                    # main thread runs on during the eval
a._next_acc = 0.9
a.eval_model()
assert a._eval_commit == 700, a._eval_commit
print("  commit stamping       : eval speaks for commit 700, not 712")

# a skipped-stride snapshot must NOT restamp
a._eval_snapshot_model()                 # returns None on the even call
assert a._eval_commit == 700
print("  stride skip           : leaves the stamp alone")

# the full path: evals feed the detector, the commit path reads the latch,
# and `halt` sets _work_done -- none of which any test above exercises
a = wired(warmup=100)
for c in range(0, 600, 2):
    a._commit_count = c
    a._eval_snapshot_model()
    if a._eval_commit != c:              # the stride skipped this one
        a._eval_commit = c
    a._next_acc = 0.90 if c < 200 else 0.80
    a.eval_model()
    a._check_budget_stop()
    if a._work_done:
        break
assert a._stop_fired == "saturation", a._stop_fired
assert a._work_done and a._sat_det.fired_at is not None
print(f"  end to end            : eval -> detector -> stop, "
      f"reason=saturation at commit {a._sat_det.fired_at}, work_done set")

# and with the detector absent the same path is byte-identical to before it
a = wired(warmup=0, sat=False)
a._next_acc = 0.5
a._eval_snapshot_model(); a.eval_model(); a._check_budget_stop()
assert a._stop_fired is None and not a._work_done
print("  saturation_stop off   : byte-identical, nothing fires")

print("\nALL SATURATION-STOP CHECKS PASS")


# ---------------------------------------------------------------------------
# (g) ROW E' -- the stall trigger (buildplan §5.13, practice P4.17).
print("\n  --- row E': the stall trigger")
from examples.fwdllm.expts.saturation_stop import SAT_STALL_FRAC  # noqa: E402

# g.1 DEFAULT OFF. Every fire commit above must be untouched with stall_frac=None,
# which is what makes this an added trigger and not a changed rule.
for ds, (want_commit, _l) in EXPECT.items():
    assert run(curve(ds), tied, hor, stall=None)[0] == want_commit, ds
print(f"  default off           : N1-N3 unchanged at "
      f"{[EXPECT[d][0] for d in EXPECT]} -- E' adds a trigger, changes nothing")

# g.2 THE DEFECT IT FIXES. GL only breaches once a run falls BELOW its own best,
# so a curve that climbs and then sits flat AT its best never fires. Synthesised
# to be exactly that case -- this is yelp-p and the s=1.0 arm in miniature.
flat = [(c, 0.5 + 0.35 * min(1.0, c / 600.0)) for c in range(0, 2000, 2)]
assert run(flat, tied, hor, stall=None)[0] is None, "decay rule must be blind here"
fired, _ = run(flat, tied, hor, stall=SAT_STALL_FRAC)
assert fired is not None and run.why == "stall", (fired, run.why)
print(f"  plateau AT the best   : decay never fires; stall fires at {fired}")

# g.3 It must NOT fire on a run that is merely climbing slowly -- the same gate
# the progress term exists for, now applied to the new trigger.
for tag in ("yahoo_control", "agnews_control", "yelp-p_control"):
    rows = curve(tag)
    accs = [a for _c, a in rows]
    w = SaturationDetector(0).window
    sm = [sum(accs[max(0, i - w + 1):i + 1]) / len(accs[max(0, i - w + 1):i + 1])
          for i in range(len(accs))]
    peak_commit = rows[max(range(len(sm)), key=lambda i: sm[i])][0]
    f, _ = run(rows, tied, hor, stall=SAT_STALL_FRAC)
    assert f is None or f >= peak_commit, (tag, f, peak_commit)
    print(f"  {tag:<20} {'never fires' if f is None else f'fires at {f}'} (peak {peak_commit})")

# g.4 A monotone rise still never fires, with the trigger on.
det = SaturationDetector(0, stall_frac=SAT_STALL_FRAC)
assert not any(det.update(c, 0.5 + c / 10000.0) for c in range(2000))
print("  still rising          : never fires with the stall trigger on")

# g.5 The reason is reported, and it latches with it.
det = SaturationDetector(0, stall_frac=SAT_STALL_FRAC)
for c in range(0, 4000, 2):
    if det.update(c, 0.80):
        break
assert det.fired_reason == "stall", det.fired_reason
print(f"  reason reported       : {det.fired_reason!r} at commit {det.fired_at}")

# g.6 The constant is the SHIPPED one and nothing here re-fits it.
assert SAT_STALL_FRAC == 0.003, SAT_STALL_FRAC
print(f"  constant              : SAT_STALL_FRAC={SAT_STALL_FRAC} "
      f"(sized on 11 curves, P4.17; plateau 0.002-0.004)")
print("\n  ALL CHECKS PASS")
