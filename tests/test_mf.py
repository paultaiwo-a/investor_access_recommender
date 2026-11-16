import numpy as np

from recs.candidates.base import recommend
from recs.candidates.mf import ALSRecommender, BPRRecommender
from recs.data import make_snapshot
from tests.conftest import CUTOFF


def test_als_scores_shape_and_determinism(ds):
    snap = make_snapshot(ds, CUTOFF)
    a = ALSRecommender(factors=4, iterations=5).fit(snap)
    b = ALSRecommender(factors=4, iterations=5).fit(snap)
    s = a.score(snap.users, snap.eligible_items)
    assert s.shape == (len(snap.users), len(snap.eligible_items))
    np.testing.assert_allclose(s, b.score(snap.users, snap.eligible_items), rtol=1e-4)


def test_als_fits_training_signal(ds):
    snap = make_snapshot(ds, CUTOFF)
    model = ALSRecommender(factors=8, regularization=0.01, iterations=20).fit(snap)
    s = model.score(["u4"], ["D", "A"])[0]
    assert s[0] > s[1]  # u4 only ever bought D


def test_bpr_recommends_unseen_items(ds):
    snap = make_snapshot(ds, CUTOFF)
    recs = recommend(BPRRecommender(factors=4, iterations=10).fit(snap), snap, k=3)
    for u, items in recs.items():
        assert len(items) == 3
        assert not set(items) & snap.history_items[u]
