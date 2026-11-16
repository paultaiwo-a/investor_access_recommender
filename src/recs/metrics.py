"""Recall and nDCG, averaged over customers who bought something new."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

Recs = Mapping[str, Sequence[str]]
Targets = Mapping[str, set[str]]


def recall_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recs[:k]) & relevant) / len(relevant)


def ndcg_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(recs[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal


def ranking_metrics(recs: Recs, targets: Targets, k: int) -> dict[str, float]:
    users = [user for user in targets if targets[user]]
    if not users:
        return {f"recall@{k}": float("nan"), f"ndcg@{k}": float("nan")}
    return {
        f"recall@{k}": float(
            np.mean([recall_at_k(list(recs.get(user, [])), targets[user], k) for user in users])
        ),
        f"ndcg@{k}": float(
            np.mean([ndcg_at_k(list(recs.get(user, [])), targets[user], k) for user in users])
        ),
    }


def average_return(recs: Recs, forward_returns: pd.Series, k: int) -> float:
    """Mean price change of the suggested assets, then averaged across customers."""
    portfolios = []
    for picked in recs.values():
        rets = forward_returns.reindex(list(picked)[:k]).dropna()
        if len(rets):
            portfolios.append(float(rets.mean()))
    if not portfolios:
        return float("nan")
    return float(np.mean(portfolios))
