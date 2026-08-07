"""Regression tests for the PLOT_TRACKER.md open-work #3/#4 figure additions
(auto broken y-axis, CDF P50/P90 annotations).

Run:  python -m pytest plotlib/test_figures.py -q
  or:  python plotlib/test_figures.py
"""

from __future__ import annotations

import os
import sys

_EXPT_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPT_SCRIPTS not in sys.path:
    sys.path.insert(0, _EXPT_SCRIPTS)

import numpy as np

from plotlib import figures as F  # noqa: E402


# --------------------------------------------------------------------------- #
# #3 — _detect_axis_break
# --------------------------------------------------------------------------- #
def test_no_break_on_a_continuous_climb():
    """A normal FL accuracy curve (random-guess baseline climbing steadily to
    target) has no real empty band -- must NOT trigger a break."""
    values = list(np.linspace(25, 85, 60))
    assert F._detect_axis_break(values) is None


def test_break_on_two_tight_clusters_far_apart():
    """The motivating case: one baseline stuck near-zero, another near the
    target -- two tight clusters with a huge empty gap between them. This is
    exactly the shape a *range*-based side-guard would wrongly reject (each
    cluster spans almost none of the total range) -- must trigger."""
    low = list(np.linspace(2.0, 4.5, 40))
    high = list(np.linspace(78.0, 84.0, 40))
    brk = F._detect_axis_break(low + high)
    assert brk is not None
    lo, hi = brk
    assert 4.5 <= lo < hi <= 78.0


def test_no_break_when_gap_too_small_relative_to_span():
    """A modest gap that's a small fraction of the total span shouldn't
    trigger -- only a genuinely wasteful empty band should."""
    values = list(np.linspace(0, 100, 50))  # uniform, largest gap is tiny
    assert F._detect_axis_break(values) is None


def test_break_ignores_a_single_outlier_point():
    """One stray point far from a dense cluster must not create a break that
    strands 39/40 points in a sliver panel -- the point-count side-guard
    should reject it even though the gap itself is huge."""
    cluster = list(np.linspace(50.0, 55.0, 39))
    outlier = [99.0]
    assert F._detect_axis_break(cluster + outlier) is None


def test_break_picks_the_largest_qualifying_gap_with_three_clusters():
    low = list(np.linspace(2.0, 4.0, 20))
    mid = list(np.linspace(40.0, 42.0, 20))
    high = list(np.linspace(78.0, 80.0, 20))
    lo, hi = F._detect_axis_break(low + mid + high)
    # the two gaps are ~36 (4->40) and ~36 (42->78); either qualifies -- just
    # confirm it picked A real inter-cluster gap, not a fake intra-cluster one
    assert (lo, hi) in [(4.0, 40.0), (42.0, 78.0)]


# --------------------------------------------------------------------------- #
# fig_e1_acc_vs_time integration — no-break path unchanged, break path renders
# --------------------------------------------------------------------------- #
class _FakeRun:
    def __init__(self, key, hrs, acc):
        self.key = key
        self._hrs, self._acc = hrs, acc
        self.evals = [1] * len(hrs)
        self.is_sim = True

    def learning_curve(self):
        return self._hrs, self._acc, [None] * len(self._acc), list(range(len(self._acc)))

    def round_transition_indices(self):
        return []


def _use_style_once():
    from plotlib.style import use_paper_style
    use_paper_style()


def test_fig_e1_single_axes_when_no_gap():
    _use_style_once()
    hrs = list(np.linspace(0, 5, 30))
    acc = list(60 + 20 * (1 - np.exp(-np.array(hrs))))
    runs = [_FakeRun("fwdllm", hrs, acc), _FakeRun("fluxtune", hrs, [a + 3 for a in acc])]
    fig = F.fig_e1_acc_vs_time(runs, target=0.84, smooth=0.5)
    assert len(fig.axes) == 1


def test_fig_e1_two_axes_when_gap_detected():
    _use_style_once()
    hrs = list(np.linspace(0, 5, 40))
    low = [2.0 + 0.5 * h for h in hrs]
    high = [78 + 6 * (1 - np.exp(-h)) for h in hrs]
    runs = [_FakeRun("fwdllm", hrs, low), _FakeRun("fluxtune", hrs, high)]
    fig = F.fig_e1_acc_vs_time(runs, target=0.84, smooth=0.3)
    assert len(fig.axes) == 2
    # exactly one legend on the whole figure (no duplicate from the 2nd panel)
    legends = [a.get_legend() for a in fig.axes if a.get_legend() is not None]
    assert len(legends) == 1


def test_fig_e1_two_axes_survives_no_target():
    """Broken-axis path must not assume `target` is set (the `_e1_target_and_
    speedup` no-op branch still has to run on the right panel)."""
    _use_style_once()
    hrs = list(np.linspace(0, 5, 40))
    low = [2.0 + 0.5 * h for h in hrs]
    high = [78 + 6 * (1 - np.exp(-h)) for h in hrs]
    runs = [_FakeRun("fwdllm", hrs, low), _FakeRun("fluxtune", hrs, high)]
    fig = F.fig_e1_acc_vs_time(runs, target=None, smooth=0.0)
    assert len(fig.axes) == 2


# --------------------------------------------------------------------------- #
# #4 — CDF P50/P90 annotation math
# --------------------------------------------------------------------------- #
class _RecordingAx:
    def __init__(self):
        self.annotations = []

    def plot(self, *_a, **_k):
        pass

    def annotate(self, text, **k):
        self.annotations.append((text, k["xy"]))


def test_cdf_percentiles_match_known_distribution():
    rng = np.random.default_rng(0)
    vals = rng.normal(50, 10, 2000).tolist()
    xs, ys = F._cdf_xy(vals)
    ax = _RecordingAx()
    F._annotate_cdf_percentiles(ax, xs, ys, "#000000")
    labels = {text: xy for text, xy in ax.annotations}
    assert set(labels) == {"P50", "P90"}
    assert abs(labels["P50"][0] - 50) < 1.5
    assert abs(labels["P90"][0] - 62.8) < 1.5


def test_cdf_percentile_skipped_if_curve_never_reaches_it():
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([20.0, 45.0, 70.0])       # never reaches 90
    ax = _RecordingAx()
    F._annotate_cdf_percentiles(ax, xs, ys, "#000000")
    labels = {text for text, _xy in ax.annotations}
    assert labels == {"P50"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
