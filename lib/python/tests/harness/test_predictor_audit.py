"""F42-T7: real notify fidelity from the aggregator's STATE_UPDATE receipts (predictor_audit.real_notify_fidelity)."""
import sys
from pathlib import Path

from sortedcontainers import SortedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import predictor_audit as pa  # noqa: E402


def test_receipt_lag_counts_as_missed_and_false_time():
    gt = {"a": SortedDict({0.0: "AVL_TRAIN", 100.0: "UN_AVL", 200.0: "AVL_TRAIN"})}
    recv = [(101.0, "a", "UN_AVL"), (202.0, "a", "AVL_TRAIN")]  # 1 s late leaving, 2 s late returning
    r = pa.real_notify_fidelity(gt, recv, span=300.0)
    ch = 300.0 / 3600
    assert abs(r["notify_false_s_ch"] - 1.0 / ch) < 1e-6 and abs(r["notify_missed_s_ch"] - 2.0 / ch) < 1e-6
    assert r["notify_flips"] == 2 and r["notify_lat_p50"] == 2.0


def test_a_lost_notify_has_no_latency_and_stays_wrong():
    gt = {"a": SortedDict({0.0: "AVL_TRAIN", 100.0: "UN_AVL"})}
    r = pa.real_notify_fidelity(gt, [], span=200.0)
    assert r["notify_flips"] == 0 and r["notify_lat_p50"] is None
    assert abs(r["notify_false_s_ch"] - 100.0 / (200.0 / 3600)) < 1e-6
