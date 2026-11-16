from recs.candidates.base import recommend
from recs.candidates.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    RecentPopularityRecommender,
)
from recs.data import make_snapshot
from recs.eval.runner import evaluate_model
from tests.conftest import CUTOFF


def test_popularity_orders_by_distinct_buyers(ds):
    snap = make_snapshot(ds, CUTOFF)
    model = PopularityRecommender().fit(snap)
    assert model.item_scores.idxmax() == "A"  # bought by u1, u2, u3


def test_recommend_excludes_history_and_ineligible(ds):
    snap = make_snapshot(ds, CUTOFF)
    recs = recommend(PopularityRecommender().fit(snap), snap, k=10)
    for u, items in recs.items():
        assert not set(items) & snap.history_items[u]
        assert set(items) <= set(snap.eligible_items)


def test_recent_popularity_prefers_recent_buys(ds):
    snap = make_snapshot(ds, CUTOFF)
    model = RecentPopularityRecommender(window_days=30).fit(snap)
    # Only u3's B buy (2020-12-15) is inside the 30-day window.
    assert model.item_scores.idxmax() == "B"


def test_item_knn_scores_co_purchased_items(ds):
    snap = make_snapshot(ds, CUTOFF)
    model = ItemKNNRecommender(k_neighbors=None).fit(snap)
    # u2 holds A and C. B co-occurs with A (u1, u3), D never co-occurs with anything u2 holds.
    s = model.score(["u2"], ["B", "D"])[0]
    assert s[0] > 0 and s[1] == 0
    assert model.explain("u2", "B")[0][0] == "A"


def test_random_is_seeded(ds):
    snap = make_snapshot(ds, CUTOFF)
    a = recommend(RandomRecommender(seed=1).fit(snap), snap, k=3)
    b = recommend(RandomRecommender(seed=1).fit(snap), snap, k=3)
    assert a == b


def test_evaluate_model_returns_all_metrics(ds):
    snap = make_snapshot(ds, CUTOFF)
    res = evaluate_model(PopularityRecommender(), snap, k=3)
    for key in ("ndcg@3", "recall@3", "map@3", "coverage@3", "arp@3", "monthly_roi@3"):
        assert key in res
    assert 0 <= res["ndcg@3"] <= 1
