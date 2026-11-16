"""Matrix factorisation candidate generators (implicit ALS and BPR)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from threadpoolctl import threadpool_limits

from recs.candidates.base import Recommender
from recs.config import SEED
from recs.data import Snapshot


def _to_numpy(x) -> np.ndarray:
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


class _MFRecommender(Recommender):
    weight = "log_count"

    def _matrix(self, snap: Snapshot) -> sp.csr_matrix:
        return snap.interaction_matrix(self.weight)

    def _store(self, model, snap: Snapshot) -> None:
        self.user_factors = _to_numpy(model.user_factors)
        self.item_factors = _to_numpy(model.item_factors)
        self._ds = snap.ds

    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        ui = [self._ds.user_index[u] for u in users]
        ii = [self._ds.item_index[i] for i in items]
        return self.user_factors[ui] @ self.item_factors[ii].T

    def item_embedding(self, items: list[str]) -> np.ndarray:
        return self.item_factors[[self._ds.item_index[i] for i in items]]


class ALSRecommender(_MFRecommender):
    """Implicit-feedback ALS (Hu, Koren, Volinsky 2008).

    Confidence is ``1 + alpha * log(1 + n_buys)``.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.1,
        alpha: float = 10.0,
        iterations: int = 15,
        seed: int = SEED,
    ):
        self.params = dict(
            factors=factors, regularization=regularization, alpha=alpha, iterations=iterations
        )
        self.seed = seed

    def fit(self, snap: Snapshot) -> ALSRecommender:
        p = self.params
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(
                factors=p["factors"],
                regularization=p["regularization"],
                alpha=p["alpha"],
                iterations=p["iterations"],
                random_state=self.seed,
                calculate_training_loss=False,
            )
            model.fit(self._matrix(snap), show_progress=False)
        self._store(model, snap)
        return self


class BPRRecommender(_MFRecommender):
    """Bayesian Personalised Ranking (Rendle et al. 2009)."""

    name = "bpr"
    weight = "binary"

    def __init__(
        self,
        factors: int = 32,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 100,
        seed: int = SEED,
    ):
        self.params = dict(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
        )
        self.seed = seed

    def fit(self, snap: Snapshot) -> BPRRecommender:
        model = BayesianPersonalizedRanking(**self.params, random_state=self.seed)
        with threadpool_limits(1, "blas"):
            model.fit(self._matrix(snap), show_progress=False)
        self._store(model, snap)
        return self
