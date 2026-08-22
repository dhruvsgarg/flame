"""Exercise row E's saturation detector, and gate it on the runs that sized it.

Checks (a) the streaming detector reproduces the replay's fire commits on N1-N3
    -- 1,184 / 1,084 / 1,126, within 0.008 of each peak,
(b) the warm-up is a MULTIPLE of the probe cadence, not a constant fitted to
    those three curves: 3 x 150 = 450 gives the same three commits as 400 and
    600 (row E2),
(c) the warm-up is load-bearing -- at 300 the detector false-fires on yahoo's
    early near-chance plateau at commit 346,
(d) it never fires on a still-rising curve, and
(e) it latches: once fired, later evals cannot un-fire it.
"""
import json
import os
import sys

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.expts.saturation_stop import (  # noqa: E402
    SaturationDetector, warmup_commits,
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "writeup_figs", "data")
CADENCE = 150
# Sized by replay on N1-N3, 2026-08-21 (buildplan §5.5). fire commit, loss vs peak.
EXPECT = {"agnews": (1184, -0.005), "yahoo": (1084, -0.008), "yelp-p": (1126, -0.005)}


def curve(ds):
    with open(os.path.join(DATA, f"{ds}_anchor.json")) as fh:
        return [(int(c), float(a)) for c, _b, _l, a in json.load(fh)["acc_budget"]]


def run(rows, warmup):
    det = SaturationDetector(warmup)
    smoothed, w = [], det.window
    accs = [a for _c, a in rows]
    for i, (c, a) in enumerate(rows):
        smoothed.append(sum(accs[max(0, i - w + 1):i + 1])
                        / len(accs[max(0, i - w + 1):i + 1]))
        if det.update(c, a):
            return det.fired_at, smoothed[-1] - max(
                sum(accs[max(0, j - w + 1):j + 1]) / len(accs[max(0, j - w + 1):j + 1])
                for j in range(len(accs)))
    return None, None


tied = warmup_commits(CADENCE)
assert tied == 450, tied
print(f"  warm-up               : 3 x {CADENCE} = {tied}, tied to the probe cadence")

for ds, (want_commit, want_loss) in EXPECT.items():
    rows = curve(ds)
    fired, loss = run(rows, tied)
    assert fired == want_commit, (ds, fired, want_commit)
    assert abs(loss - want_loss) < 0.001, (ds, loss, want_loss)
    # (b) the plateau: 400 and 600 must give the same commit, or the warm-up
    # is fitted to these curves rather than derived from the cadence.
    assert run(rows, 400)[0] == fired and run(rows, 600)[0] == fired, ds
    print(f"  {ds:<8} fires at commit {fired}, {loss:+.3f} vs peak "
          f"(warm-up 400/450/600 agree)")

# (c) the warm-up is load-bearing, and yahoo is the case that proves it
assert run(curve("yahoo"), 300)[0] == 346
print("  warm-up 300           : false-fires on yahoo at commit 346 (0.24 acc)")

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

print("\nALL SATURATION-STOP CHECKS PASS")
