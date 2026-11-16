import math

import numpy as np
import pandas as pd
import pytest

from recs.eval import metrics as m


def test_recall_precision_hit():
    recs, rel = ["a", "b", "c", "d"], {"b", "x"}
    assert m.recall_at_k(recs, rel, 2) == 0.5
    assert m.precision_at_k(recs, rel, 2) == 0.5
    assert m.hit_at_k(recs, rel, 1) == 0.0
    assert m.hit_at_k(recs, rel, 2) == 1.0


def test_ndcg_perfect_and_known_value():
    assert m.ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    # Single relevant item at rank 2: 1/log2(3) over ideal 1.
    assert m.ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))
    assert m.ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_ap_at_k():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / min(k, 2)
    assert m.ap_at_k(["a", "x", "b"], {"a", "b"}, 10) == pytest.approx((1 + 2 / 3) / 2)


def test_ranking_metrics_skip_users_without_targets():
    recs = {"u1": ["a"], "u2": ["b"]}
    out = m.ranking_metrics(recs, {"u1": {"a"}, "u2": set()}, k=1)
    assert out["recall@1"] == 1.0


def test_coverage():
    recs = {"u1": ["a", "b"], "u2": ["a", "c"]}
    assert m.catalog_coverage(recs, ["a", "b", "c", "d"], k=2) == 0.75
    assert m.catalog_coverage(recs, ["a", "b", "c", "d"], k=1) == 0.25


def test_popularity_bias():
    pop = pd.Series({"a": 100, "b": 10, "c": 1})
    hi = m.popularity_bias({"u": ["a"]}, pop, 1)
    lo = m.popularity_bias({"u": ["c"]}, pop, 1)
    assert hi["arp"] > lo["arp"]
    assert hi["gini"] == pytest.approx(lo["gini"])


def test_gini_extremes():
    assert m.gini_index(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert m.gini_index(np.array([0.0, 0.0, 9.0])) == pytest.approx(2 / 3)


def test_profitability_equal_weight_portfolio():
    fwd = pd.Series({"a": 0.10, "b": -0.10, "c": 0.21})
    out = m.profitability_at_k({"u1": ["a", "b"], "u2": ["c"]}, fwd, k=2, months=2)
    assert out["roi@2"] == pytest.approx((0.0 + 0.21) / 2)
    assert out["monthly_roi@2"] == pytest.approx((0.0 + 0.1) / 2)
    assert out["pct_positive@2"] == pytest.approx(2 / 3)


def test_monthly_roi_inverts_compounding():
    assert m.monthly_roi((1.01) ** 6 - 1, 6) == pytest.approx(0.01)
