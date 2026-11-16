"""Glue between snapshots, recommenders and metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from recs.candidates.base import Recommender, recommend
from recs.data import Dataset, Snapshot, make_snapshot
from recs.eval.metrics import evaluate

DAYS_PER_MONTH = 30.4375

# The FAR-Trans paper's benchmark: 61 cutoffs, two weeks apart, from 2019-08-01.
PAPER_FIRST_CUTOFF = "2019-08-01"
PAPER_N_CUTOFFS = 61
PAPER_STEP_DAYS = 14


def paper_cutoffs(step: int = 1) -> list[pd.Timestamp]:
    start = pd.Timestamp(PAPER_FIRST_CUTOFF)
    every = [start + pd.Timedelta(days=PAPER_STEP_DAYS * i) for i in range(PAPER_N_CUTOFFS)]
    return every[::step]


def item_popularity(snap: Snapshot) -> pd.Series:
    pop = snap.history.groupby("isin")["customer_id"].nunique()
    return pop.reindex(snap.eligible_items, fill_value=0)


def evaluate_recs(snap: Snapshot, recs: dict[str, list[str]], k: int = 10) -> dict[str, float]:
    months = (snap.end - snap.cutoff).days / DAYS_PER_MONTH
    res = evaluate(
        recs,
        snap.targets,
        snap.eligible_items,
        item_popularity(snap),
        snap.forward_returns,
        k=k,
        months=months,
    )
    res["n_users"] = len(snap.users)
    res["n_items"] = len(snap.eligible_items)
    return res


def evaluate_model(model: Recommender, snap: Snapshot, k: int = 10) -> dict[str, float]:
    model.fit(snap)
    recs = recommend(model, snap, k=k)
    return evaluate_recs(snap, recs, k)


def market_average(snap: Snapshot) -> float:
    """Monthly ROI of an equal-weight portfolio of every eligible asset."""
    months = (snap.end - snap.cutoff).days / DAYS_PER_MONTH
    r = float(snap.forward_returns.mean())
    return (1 + r) ** (1 / months) - 1


def rolling_evaluation(
    ds: Dataset,
    factories: dict[str, Callable[[], Recommender]],
    cutoffs: Iterable[pd.Timestamp],
    k: int = 10,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate fresh models at every cutoff. Returns one row per (cutoff, model)."""
    rows = []
    for t in cutoffs:
        snap = make_snapshot(ds, t)
        if not snap.users:
            continue
        for name, factory in factories.items():
            res = evaluate_model(factory(), snap, k)
            rows.append({"cutoff": t, "model": name, **res})
        rows.append(
            {"cutoff": t, "model": "market_average", f"monthly_roi@{k}": market_average(snap)}
        )
        if progress:
            print(f"  {t.date()}: {len(snap.users)} users, {len(snap.eligible_items)} assets")
    return pd.DataFrame(rows)
