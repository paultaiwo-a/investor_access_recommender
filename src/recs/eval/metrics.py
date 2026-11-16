"""Offline metrics: ranking accuracy, catalog behaviour and investor profitability.

All per-user metrics are averaged over the users passed in (users with no targets are
skipped by the ranking metrics, matching the usual offline protocol).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

Recs = Mapping[str, Sequence[str]]
Targets = Mapping[str, set[str]]


# ---------------------------------------------------------------- per-list metrics
def recall_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recs[:k]) & relevant) / len(relevant)


def precision_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def hit_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    return float(bool(set(recs[:k]) & relevant))


def dcg_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1.0 / np.log2(r + 2) for r, it in enumerate(recs[:k]) if it in relevant)


def ndcg_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(relevant))))
    return dcg_at_k(recs, relevant, k) / ideal


def ap_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision at k, normalised by min(k, |relevant|)."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for r, it in enumerate(recs[:k]):
        if it in relevant:
            hits += 1
            total += hits / (r + 1)
    return total / min(k, len(relevant))


def monthly_roi(total_return: float, months: float) -> float:
    """Convert a total return over ``months`` months into a compound monthly rate."""
    return (1.0 + total_return) ** (1.0 / months) - 1.0


# ---------------------------------------------------------------- aggregate metrics
def ranking_metrics(recs: Recs, targets: Targets, k: int) -> dict[str, float]:
    users = [u for u in targets if targets[u]]
    if not users:
        return {f"{m}@{k}": float("nan") for m in ("recall", "ndcg", "map", "precision", "hit")}
    fns = {
        "recall": recall_at_k,
        "ndcg": ndcg_at_k,
        "map": ap_at_k,
        "precision": precision_at_k,
        "hit": hit_at_k,
    }
    return {
        f"{name}@{k}": float(np.mean([fn(list(recs.get(u, [])), targets[u], k) for u in users]))
        for name, fn in fns.items()
    }


def catalog_coverage(recs: Recs, catalog: Sequence[str], k: int) -> float:
    """Share of the recommendable catalog that appears in at least one top-k list."""
    if not catalog:
        return 0.0
    shown = {it for r in recs.values() for it in list(r)[:k]}
    return len(shown & set(catalog)) / len(set(catalog))


def popularity_bias(recs: Recs, item_popularity: pd.Series, k: int) -> dict[str, float]:
    """How much the lists lean on already-popular assets.

    * ``arp``: average popularity percentile (0 = least, 1 = most popular) of recommended items.
    * ``gini``: Gini index of exposure across the catalog (1 = all exposure on one item).
    """
    pct = item_popularity.rank(pct=True)
    shown = [it for r in recs.values() for it in list(r)[:k]]
    if not shown:
        return {"arp": float("nan"), "gini": float("nan")}
    arp = float(pct.reindex(shown).fillna(0.0).mean())
    counts = pd.Series(shown).value_counts().reindex(item_popularity.index, fill_value=0)
    gini = gini_index(counts.to_numpy(dtype=float))
    return {"arp": arp, "gini": gini}


def gini_index(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def profitability_at_k(
    recs: Recs, forward_returns: pd.Series, k: int, months: float = 6.0
) -> dict[str, float]:
    """Return of an equally weighted portfolio of each user's top-k, held for the horizon.

    * ``roi@k``: mean total return over the horizon, averaged across users.
    * ``monthly_roi@k``: the same as a compound monthly rate (the FAR-Trans ROI@k).
    * ``pct_positive@k``: share of recommended assets whose price went up.
    """
    port, positive = [], []
    for r in recs.values():
        rets = forward_returns.reindex(list(r)[:k]).dropna()
        if len(rets) == 0:
            continue
        port.append(rets.mean())
        positive.extend((rets > 0).tolist())
    if not port:
        return {f"roi@{k}": float("nan"), f"monthly_roi@{k}": float("nan")}
    mean_ret = float(np.mean(port))
    return {
        f"roi@{k}": mean_ret,
        f"monthly_roi@{k}": float(np.mean([monthly_roi(p, months) for p in port])),
        f"pct_positive@{k}": float(np.mean(positive)),
    }


def evaluate(
    recs: Recs,
    targets: Targets,
    catalog: Sequence[str],
    item_popularity: pd.Series,
    forward_returns: pd.Series,
    k: int = 10,
    months: float = 6.0,
) -> dict[str, float]:
    out = ranking_metrics(recs, targets, k)
    out[f"coverage@{k}"] = catalog_coverage(recs, catalog, k)
    bias = popularity_bias(recs, item_popularity, k)
    out[f"arp@{k}"] = bias["arp"]
    out[f"gini@{k}"] = bias["gini"]
    out.update(profitability_at_k(recs, forward_returns, k, months))
    return out
