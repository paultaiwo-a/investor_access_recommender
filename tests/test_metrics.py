import math

import pandas as pd
import pytest

from recs.metrics import average_return, ndcg_at_k, ranking_metrics, recall_at_k


def test_recall():
    assert recall_at_k(["a", "b", "c"], {"b", "x"}, 2) == 0.5
    assert recall_at_k(["x"], set(), 1) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_ranking_metrics_skip_customers_with_nothing_to_hit():
    recs = {"u1": ["a"], "u2": ["b"]}
    out = ranking_metrics(recs, {"u1": {"a"}, "u2": set()}, k=1)
    assert out["recall@1"] == 1.0


def test_average_return_is_the_mean_of_the_lists():
    fwd = pd.Series({"a": 0.10, "b": -0.10, "c": 0.20})
    got = average_return({"u1": ["a", "b"], "u2": ["c"]}, fwd, k=2)
    assert got == pytest.approx((0.0 + 0.20) / 2)
