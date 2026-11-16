"""Two models. Popularity, and item-item cosine on what people bought."""

from __future__ import annotations

import numpy as np

from recs.data import Snapshot


def popularity(snap: Snapshot, k: int = 10) -> dict[str, list[str]]:
    """Assets bought by the most customers before the cutoff."""
    counts = snap.history.groupby("isin")["customer_id"].nunique().astype(float)
    scores = counts.reindex(snap.eligible_items).fillna(0.0)
    grid = np.tile(scores.to_numpy(), (len(snap.users), 1))
    return _top_k(snap, grid, k)


def item_item(snap: Snapshot, k: int = 10) -> dict[str, list[str]]:
    """Suggest assets that were bought by the same people as the ones you hold."""
    items = snap.eligible_items
    if not snap.users or not items:
        return {user: [] for user in snap.users}

    pairs = snap.history[["customer_id", "isin"]].drop_duplicates()
    columns = sorted(set(pairs["isin"]) | set(items))
    mat = (
        pairs.assign(v=1.0)
        .pivot_table(index="customer_id", columns="isin", values="v", aggfunc="max")
        .reindex(index=snap.users, columns=columns)
        .fillna(0.0)
    )
    user_item = mat.to_numpy(dtype=float)
    norms = np.linalg.norm(user_item, axis=0)
    denom = np.outer(norms, norms)
    sim = np.zeros_like(denom)
    np.divide(user_item.T @ user_item, denom, out=sim, where=denom > 0)
    np.fill_diagonal(sim, 0.0)

    col = {isin: i for i, isin in enumerate(columns)}
    wanted = [col[isin] for isin in items]
    return _top_k(snap, user_item @ sim[:, wanted], k)


def _top_k(snap: Snapshot, scores: np.ndarray, k: int) -> dict[str, list[str]]:
    scores = np.array(scores, dtype=float, copy=True)
    items = snap.eligible_items
    col = {isin: i for i, isin in enumerate(items)}
    for row, user in enumerate(snap.users):
        for held in snap.held.get(user, ()):
            j = col.get(held)
            if j is not None:
                scores[row, j] = -np.inf

    if scores.size == 0:
        return {user: [] for user in snap.users}

    order = np.argsort(-scores, axis=1, kind="stable")
    names = np.asarray(items)
    out: dict[str, list[str]] = {}
    for row, user in enumerate(snap.users):
        keep = order[row][np.isfinite(scores[row, order[row]])][:k]
        out[user] = names[keep].tolist()
    return out
