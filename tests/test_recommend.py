from recs.data import make_snapshot
from recs.recommend import item_item, popularity
from tests.conftest import CUTOFF


def test_popularity_ranks_by_how_many_people_bought_it(ds):
    snap = make_snapshot(ds, CUTOFF)
    counts = snap.history.groupby("isin")["customer_id"].nunique()
    assert counts.idxmax() == "A"
    # u2 already holds A. The next one is B, bought by u1 and u3.
    assert popularity(snap, k=10)["u2"][0] == "B"


def test_lists_leave_out_assets_the_customer_already_bought(ds):
    snap = make_snapshot(ds, CUTOFF)
    for model in (popularity, item_item):
        recs = model(snap, k=10)
        for user, items in recs.items():
            assert not set(items) & snap.held[user]
            assert set(items) <= set(snap.eligible_items)


def test_item_item_prefers_something_bought_with_what_you_hold(ds):
    snap = make_snapshot(ds, CUTOFF)
    # u2 holds A and C. B was bought with A. D was not bought with either.
    recs = item_item(snap, k=10)
    assert recs["u2"][0] == "B"
    assert "A" not in recs["u2"]
    assert "C" not in recs["u2"]
