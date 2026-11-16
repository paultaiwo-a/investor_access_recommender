"""Load the warehouse and build point-in-time evaluation snapshots.

A :class:`Snapshot` is everything a recommender may see at a cutoff date ``t`` (buys strictly
before ``t``) plus what the evaluator needs for the window ``[t, t + horizon)``: the first-time
buys (targets) and each asset's forward return. This mirrors the FAR-Trans protocol:

* relevance is a first-time buy in the window; pairs already bought before ``t`` are dropped;
* only customers with at least one buy before ``t`` and one target in the window are evaluated;
* only assets priced at ``t`` and through ``t + horizon`` are recommendable or relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from recs.config import DB_PATH, PROFIT_HORIZON_DAYS

# Slack when deciding whether an asset is priced "at" a date (weekends, holidays, sparse bonds).
PRICE_SLACK_DAYS = 10


@dataclass
class Dataset:
    """In-memory copy of the warehouse tables the models need."""

    transactions: pd.DataFrame
    assets: pd.DataFrame
    prices: pd.DataFrame
    customers: pd.DataFrame

    @classmethod
    def load(cls, db_path: Path | str = DB_PATH) -> Dataset:
        with duckdb.connect(str(db_path), read_only=True) as con:
            tx = con.sql(
                "select customer_id, isin, transaction_type, transaction_date, total_value, units "
                "from stg_transactions"
            ).df()
            assets = con.sql("select * from stg_assets").df()
            prices = con.sql("select isin, price_date, close_price from stg_prices").df()
            customers = con.sql("select * from stg_customers").df()
        return cls.from_frames(tx, assets, prices, customers)

    @classmethod
    def from_frames(
        cls,
        transactions: pd.DataFrame,
        assets: pd.DataFrame,
        prices: pd.DataFrame,
        customers: pd.DataFrame,
    ) -> Dataset:
        tx = transactions.copy()
        tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
        pr = prices.copy()
        pr["price_date"] = pd.to_datetime(pr["price_date"])
        pr = pr.sort_values(["isin", "price_date"]).reset_index(drop=True)
        cu = customers.copy()
        for col in ("valid_from", "valid_to"):
            if col in cu:
                cu[col] = pd.to_datetime(cu[col])
        return cls(tx, assets.reset_index(drop=True), pr, cu)

    @cached_property
    def buys(self) -> pd.DataFrame:
        return self.transactions[self.transactions["transaction_type"] == "buy"]

    @cached_property
    def user_ids(self) -> np.ndarray:
        return np.sort(self.transactions["customer_id"].unique())

    @cached_property
    def item_ids(self) -> np.ndarray:
        ids = set(self.assets["isin"]) | set(self.transactions["isin"])
        return np.array(sorted(ids))

    @cached_property
    def user_index(self) -> dict[str, int]:
        return {u: i for i, u in enumerate(self.user_ids)}

    @cached_property
    def item_index(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.item_ids)}

    @cached_property
    def price_wide(self) -> pd.DataFrame:
        """Calendar-day close prices, forward-filled over short gaps, one column per asset."""
        wide = self.prices.pivot(index="price_date", columns="isin", values="close_price")
        idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
        return wide.reindex(idx).ffill(limit=PRICE_SLACK_DAYS)

    def price_on(self, date: pd.Timestamp) -> pd.Series:
        """Last close on or before ``date`` (within the slack window)."""
        wide = self.price_wide
        pos = wide.index.searchsorted(pd.Timestamp(date), side="right") - 1
        if pos < 0:
            return pd.Series(np.nan, index=wide.columns)
        return wide.iloc[pos]

    def forward_returns(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Simple return from ``start`` to ``end`` for every asset priced at both dates."""
        p0, p1 = self.price_on(start), self.price_on(end)
        return (p1 / p0 - 1.0).dropna()

    def customer_profile(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Latest known profile per customer as of ``cutoff`` (no future updates)."""
        cu = self.customers[self.customers["valid_from"] < pd.Timestamp(cutoff)]
        cu = cu.sort_values("valid_from").groupby("customer_id").tail(1)
        return cu.set_index("customer_id")


@dataclass
class Snapshot:
    cutoff: pd.Timestamp
    end: pd.Timestamp
    history: pd.DataFrame  # per (customer_id, isin): n_buys, total_value, first_date, last_date
    targets: dict[str, set[str]]
    users: list[str]
    eligible_items: list[str]
    forward_returns: pd.Series  # isin -> return over [cutoff, end)
    _ds: Dataset = field(repr=False)

    @property
    def ds(self) -> Dataset:
        return self._ds

    @cached_property
    def history_items(self) -> dict[str, set[str]]:
        return self.history.groupby("customer_id")["isin"].agg(set).to_dict()

    def interaction_matrix(self, weight: str = "binary") -> sp.csr_matrix:
        """User x item matrix over the dataset's global index, built from history only."""
        ds = self._ds
        rows = self.history["customer_id"].map(ds.user_index).to_numpy()
        cols = self.history["isin"].map(ds.item_index).to_numpy()
        if weight == "binary":
            vals = np.ones(len(rows), dtype=np.float32)
        elif weight == "count":
            vals = self.history["n_buys"].to_numpy(dtype=np.float32)
        elif weight == "log_count":
            vals = np.log1p(self.history["n_buys"].to_numpy(dtype=np.float32))
        else:
            raise ValueError(f"unknown weight {weight!r}")
        shape = (len(ds.user_ids), len(ds.item_ids))
        return sp.csr_matrix((vals, (rows, cols)), shape=shape)


def history_before(buys: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    h = buys[buys["transaction_date"] < cutoff]
    return (
        h.groupby(["customer_id", "isin"], as_index=False)
        .agg(
            n_buys=("transaction_date", "size"),
            total_value=("total_value", "sum"),
            first_date=("transaction_date", "min"),
            last_date=("transaction_date", "max"),
        )
        .reset_index(drop=True)
    )


def eligible_assets(ds: Dataset, cutoff: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Assets with a price at the cutoff and at the end of the horizon."""
    return sorted(ds.forward_returns(cutoff, end).index)


def serving_snapshot(ds: Dataset, as_of: str | pd.Timestamp) -> Snapshot:
    """Snapshot for live scoring: every customer with history, every asset priced at ``as_of``.

    There is no future yet, so targets and forward returns are empty.
    """
    as_of = pd.Timestamp(as_of)
    history = history_before(ds.buys, as_of)
    eligible = sorted(ds.price_on(as_of).dropna().index)
    users = sorted(set(history["customer_id"]))
    empty = pd.Series(np.nan, index=eligible)
    return Snapshot(as_of, as_of, history, {}, users, eligible, empty, ds)


def make_snapshot(
    ds: Dataset,
    cutoff: str | pd.Timestamp,
    horizon_days: int = PROFIT_HORIZON_DAYS,
    end: str | pd.Timestamp | None = None,
    require_targets: bool = True,
) -> Snapshot:
    cutoff = pd.Timestamp(cutoff)
    end = pd.Timestamp(end) if end is not None else cutoff + pd.Timedelta(days=horizon_days)
    buys = ds.buys
    history = history_before(buys, cutoff)
    eligible = eligible_assets(ds, cutoff, end)
    eligible_set = set(eligible)

    window = buys[(buys["transaction_date"] >= cutoff) & (buys["transaction_date"] < end)]
    window = window[window["isin"].isin(eligible_set)]
    seen = set(zip(history["customer_id"], history["isin"], strict=True))
    pairs = {
        (u, i)
        for u, i in zip(window["customer_id"], window["isin"], strict=True)
        if (u, i) not in seen
    }
    targets: dict[str, set[str]] = {}
    for u, i in pairs:
        targets.setdefault(u, set()).add(i)

    known = set(history["customer_id"])
    if require_targets:
        users = sorted(u for u in targets if u in known)
    else:
        users = sorted(known)
    targets = {u: targets[u] for u in users if u in targets}
    fwd = ds.forward_returns(cutoff, end).reindex(eligible)
    return Snapshot(cutoff, end, history, targets, users, eligible, fwd, ds)
