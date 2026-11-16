"""Non-personalised and neighbourhood baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recs.candidates.base import Recommender
from recs.config import SEED
from recs.data import Snapshot


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, snap: Snapshot) -> RandomRecommender:
        self._rng = np.random.default_rng(self.seed)
        return self

    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        return self._rng.random((len(users), len(items)))


class _ItemScoreRecommender(Recommender):
    """A recommender that gives every user the same per-item score."""

    item_scores: pd.Series

    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        s = self.item_scores.reindex(items).fillna(0.0).to_numpy(dtype=np.float64)
        return np.broadcast_to(s, (len(users), len(items)))


class PopularityRecommender(_ItemScoreRecommender):
    """Number of distinct customers who bought the asset before the cutoff."""

    name = "popularity"

    def fit(self, snap: Snapshot) -> PopularityRecommender:
        self.item_scores = snap.history.groupby("isin")["customer_id"].nunique().astype(float)
        return self


class RecentPopularityRecommender(_ItemScoreRecommender):
    """Distinct buyers in the last ``window_days`` before the cutoff.

    Ties (and assets with no recent buyers) are broken by all-time popularity, scaled so it
    never outranks a single recent buyer.
    """

    name = "recent_popularity"

    def __init__(self, window_days: int = 90):
        self.window_days = window_days

    def fit(self, snap: Snapshot) -> RecentPopularityRecommender:
        buys = snap.ds.buys
        start = snap.cutoff - pd.Timedelta(days=self.window_days)
        recent = buys[
            (buys["transaction_date"] >= start) & (buys["transaction_date"] < snap.cutoff)
        ]
        recent_pop = recent.groupby("isin")["customer_id"].nunique().astype(float)
        all_pop = snap.history.groupby("isin")["customer_id"].nunique().astype(float)
        tie = all_pop / (all_pop.max() + 1.0) if len(all_pop) else all_pop
        self.item_scores = recent_pop.add(tie, fill_value=0.0)
        return self


class ItemKNNRecommender(Recommender):
    """Item-item cosine similarity on the binary purchase matrix.

    score(u, i) = sum over held items j of sim(i, j), keeping the ``k_neighbors`` most similar
    items per item (``None`` keeps all).
    """

    name = "item_knn"

    def __init__(self, k_neighbors: int | None = 50, shrink: float = 0.0):
        self.k_neighbors = k_neighbors
        self.shrink = shrink

    def fit(self, snap: Snapshot) -> ItemKNNRecommender:
        ds = snap.ds
        x = snap.interaction_matrix("binary").tocsc()
        self._user_items = x.tocsr()
        co = (x.T @ x).toarray().astype(np.float64)
        norms = np.sqrt(np.diag(co))
        denom = np.outer(norms, norms) + self.shrink
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, co / denom, 0.0)
        np.fill_diagonal(sim, 0.0)
        if self.k_neighbors is not None and self.k_neighbors < sim.shape[0]:
            kth = np.partition(sim, -self.k_neighbors, axis=0)[-self.k_neighbors, :]
            sim = np.where(sim >= kth[None, :], sim, 0.0)
        self.sim = sim
        self._ds = ds
        return self

    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        ui = [self._ds.user_index[u] for u in users]
        ii = [self._ds.item_index[i] for i in items]
        x = self._user_items[ui]
        return np.asarray(sp.csr_matrix(x) @ self.sim[:, ii])

    def explain(self, user: str, item: str, top: int = 1) -> list[tuple[str, float]]:
        """Held items that contribute most to ``score(user, item)``."""
        u = self._ds.user_index[user]
        held = self._user_items[u].indices
        j = self._ds.item_index[item]
        contrib = sorted(((self.sim[h, j], h) for h in held), reverse=True)[:top]
        return [(str(self._ds.item_ids[h]), float(s)) for s, h in contrib if s > 0]
