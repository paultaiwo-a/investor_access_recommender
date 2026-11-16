"""Stage one: fused candidate generation.

ALS on its own reaches ~0.80 recall@50 on validation. Fusing ALS, item-item kNN and recent
popularity with reciprocal rank fusion (RRF) lifts that to ~0.87, so the ranker sees more of
the assets customers actually go on to buy. Each source's score and rank are kept as ranker
features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recs.candidates.base import Recommender
from recs.candidates.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RecentPopularityRecommender,
)
from recs.candidates.mf import ALSRecommender
from recs.config import N_CANDIDATES
from recs.data import Snapshot

RRF_K = 60
DEFAULT_ALS = {"factors": 64, "regularization": 200.0, "alpha": 20.0, "iterations": 15}


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of each column within its row (0 = best)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


class HybridCandidates(Recommender):
    name = "hybrid_rrf"

    def __init__(self, als_params: dict | None = None, sources: tuple[str, ...] | None = None):
        self.als_params = {**DEFAULT_ALS, **(als_params or {})}
        self.sources = sources or ("als", "item_knn", "recent_popularity")

    def fit(self, snap: Snapshot) -> HybridCandidates:
        self.models: dict[str, Recommender] = {
            "als": ALSRecommender(**self.als_params).fit(snap),
            "item_knn": ItemKNNRecommender().fit(snap),
            "recent_popularity": RecentPopularityRecommender().fit(snap),
            "popularity": PopularityRecommender().fit(snap),
        }
        self._snap = snap
        return self

    def _masked_scores(self, users: list[str], items: list[str]) -> dict[str, np.ndarray]:
        col = {it: j for j, it in enumerate(items)}
        mask = np.zeros((len(users), len(items)), dtype=bool)
        hist = self._snap.history_items
        for r, u in enumerate(users):
            for it in hist.get(u, ()):
                j = col.get(it)
                if j is not None:
                    mask[r, j] = True
        out = {}
        for name, model in self.models.items():
            s = np.asarray(model.score(users, items), dtype=np.float64).copy()
            s[mask] = -np.inf
            out[name] = s
        out["_mask"] = mask
        return out

    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        s = self._masked_scores(users, items)
        fused = sum(1.0 / (RRF_K + _ranks(s[src])) for src in self.sources)
        fused[s["_mask"]] = -np.inf
        return fused

    def candidates(
        self, users: list[str], items: list[str], n: int = N_CANDIDATES, batch: int = 2048
    ) -> pd.DataFrame:
        """Top-n fused candidates per user, with per-source scores and ranks as columns."""
        frames = []
        if not users or not items:
            cols = ["customer_id", "isin", "rrf_score"]
            for src in self.models:
                cols += [f"{src}_score", f"{src}_rank"]
            return pd.DataFrame(columns=[*cols, "rrf_rank"])
        items_arr = np.asarray(items)
        n = min(n, len(items))
        for start in range(0, len(users), batch):
            ub = users[start : start + batch]
            s = self._masked_scores(ub, items)
            ranks = {src: _ranks(s[src]) for src in self.models}
            fused = sum(1.0 / (RRF_K + ranks[src]) for src in self.sources)
            fused[s["_mask"]] = -np.inf
            top = np.argpartition(-fused, n - 1, axis=1)[:, :n]
            rows = np.repeat(np.arange(len(ub)), n)
            cols = top.ravel()
            keep = np.isfinite(fused[rows, cols])
            rows, cols = rows[keep], cols[keep]
            df = pd.DataFrame(
                {
                    "customer_id": np.asarray(ub)[rows],
                    "isin": items_arr[cols],
                    "rrf_score": fused[rows, cols],
                }
            )
            for src in self.models:
                df[f"{src}_score"] = s[src][rows, cols]
                df[f"{src}_rank"] = ranks[src][rows, cols]
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out["rrf_rank"] = out.groupby("customer_id")["rrf_score"].rank(
            ascending=False, method="first"
        )
        return out
