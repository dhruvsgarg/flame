import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aggregator" / "pytorch"))
from eval_subset import select_eval_indices  # noqa: E402


def test_disabled_returns_all():
    assert select_eval_indices(range(10), None, 1) == list(range(10))
    assert select_eval_indices(range(10), 0, 1) == list(range(10))
    assert select_eval_indices(range(10), 50, 1) == list(range(10))


def test_subset_is_fixed_sorted_and_unique():
    a = select_eval_indices(range(1000), 100, 1234)
    assert a == select_eval_indices(range(1000), 100, 1234)
    assert a == sorted(set(a)) and len(a) == 100
    assert a != select_eval_indices(range(1000), 100, 7)
