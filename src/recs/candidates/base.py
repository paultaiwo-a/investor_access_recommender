"""Common recommender interface.

Every model scores a (users x items) block. :func:`recommend` turns scores into top-k lists,
restricted to the snapshot's eligible assets and excluding assets the customer already bought.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from recs.data import Snapshot


class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, snap: Snapshot) -> Recommender: ...

    @abstractmethod
    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        """Return a float array of shape (len(users), len(items)); higher is better."""


def recommend(
    model: Recommender,
    snap: Snapshot,
    k: int = 10,
    users: list[str] | None = None,
    items: list[str] | None = None,
    exclude_history: bool = True,
) -> dict[str, list[str]]:
    users = snap.users if users is None else users
    items = snap.eligible_items if items is None else items
    if not users or not items:
        return {u: [] for u in users}
    item_arr = np.asarray(items)
    col = {it: j for j, it in enumerate(items)}
    out: dict[str, list[str]] = {}
    batch = 4096
    hist = snap.history_items
    for start in range(0, len(users), batch):
        ub = users[start : start + batch]
        scores = np.asarray(model.score(ub, items), dtype=np.float64).copy()
        if exclude_history:
            for r, u in enumerate(ub):
                for it in hist.get(u, ()):
                    j = col.get(it)
                    if j is not None:
                        scores[r, j] = -np.inf
        kk = min(k, len(items))
        top = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        for r, u in enumerate(ub):
            idx = top[r]
            idx = idx[np.argsort(-scores[r, idx], kind="stable")]
            idx = idx[np.isfinite(scores[r, idx])]
            out[u] = item_arr[idx].tolist()
    return out
