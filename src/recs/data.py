"""Load the warehouse and build a point-in-time snapshot.

A snapshot is what a model may use at a cutoff, plus the first-time buys that
happen after it. Buys on or after the cutoff are not in the history. A repeat
buy of something the customer already held is not a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from recs.config import DB_PATH, PROFIT_HORIZON_DAYS

# Weekends and holidays. A price a few days earlier still counts.
PRICE_SLACK_DAYS = 10


@dataclass
class Dataset:
    transactions: pd.DataFrame
    assets: pd.DataFrame
    prices: pd.DataFrame

    @classmethod
    def load(cls, db_path: Path | str = DB_PATH) -> Dataset:
        with duckdb.connect(str(db_path), read_only=True) as con:
            tx = con.sql(
                """
                select
                    customer_id,
                    isin,
                    'buy' as transaction_type,
                    transaction_date,
                    total_value
                from buys
                """
            ).df()
            assets = con.sql("select * from assets").df()
            prices = con.sql("select isin, price_date, close_price from prices").df()
        return cls.from_frames(tx, assets, prices)

    @classmethod
    def from_frames(
        cls,
        transactions: pd.DataFrame,
        assets: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dataset:
        tx = transactions.copy()
        tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
        pr = prices.copy()
        pr["price_date"] = pd.to_datetime(pr["price_date"])
        pr = pr.drop_duplicates(["isin", "price_date"], keep="last")
        pr = pr.sort_values(["isin", "price_date"]).reset_index(drop=True)
        return cls(tx, assets.reset_index(drop=True), pr)

    @property
    def buys(self) -> pd.DataFrame:
        tx = self.transactions
        if "transaction_type" not in tx.columns:
            return tx
        return tx[tx["transaction_type"] == "buy"]

    @cached_property
    def price_wide(self) -> pd.DataFrame:
        wide = self.prices.pivot(index="price_date", columns="isin", values="close_price")
        if wide.empty:
            return wide
        idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
        return wide.reindex(idx).ffill(limit=PRICE_SLACK_DAYS)

    def price_on(self, date: pd.Timestamp) -> pd.Series:
        wide = self.price_wide
        if wide.empty:
            return pd.Series(dtype=float)
        pos = wide.index.searchsorted(pd.Timestamp(date), side="right") - 1
        if pos < 0:
            return pd.Series(np.nan, index=wide.columns)
        return wide.iloc[pos]

    def forward_returns(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        p0, p1 = self.price_on(start), self.price_on(end)
        return (p1 / p0 - 1.0).dropna()


@dataclass
class Snapshot:
    cutoff: pd.Timestamp
    end: pd.Timestamp
    history: pd.DataFrame
    targets: dict[str, set[str]]
    users: list[str]
    eligible_items: list[str]
    forward_returns: pd.Series
    held: dict[str, set[str]]


def make_snapshot(
    ds: Dataset,
    cutoff: str | pd.Timestamp,
    horizon_days: int = PROFIT_HORIZON_DAYS,
    end: str | pd.Timestamp | None = None,
) -> Snapshot:
    cutoff = pd.Timestamp(cutoff)
    end = pd.Timestamp(end) if end is not None else cutoff + pd.Timedelta(days=horizon_days)
    buys = ds.buys
    history = (
        buys[buys["transaction_date"] < cutoff]
        .groupby(["customer_id", "isin"], as_index=False)
        .agg(n_buys=("transaction_date", "size"), last_date=("transaction_date", "max"))
    )
    held = history.groupby("customer_id")["isin"].agg(set).to_dict()

    fwd = ds.forward_returns(cutoff, end)
    eligible = sorted(fwd.index)
    window = buys[(buys["transaction_date"] >= cutoff) & (buys["transaction_date"] < end)]
    window = window[window["isin"].isin(eligible)]
    seen = set(zip(history["customer_id"], history["isin"], strict=True))

    targets: dict[str, set[str]] = {}
    for user, isin in zip(window["customer_id"], window["isin"], strict=True):
        if (user, isin) in seen:
            continue
        targets.setdefault(user, set()).add(isin)

    known = set(history["customer_id"])
    users = sorted(user for user in targets if user in known)
    targets = {user: targets[user] for user in users}
    return Snapshot(cutoff, end, history, targets, users, eligible, fwd.reindex(eligible), held)
