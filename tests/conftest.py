"""Small made-up dataset. Tests do not need the real download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recs.data import Dataset

ASSETS = pd.DataFrame(
    {
        "isin": ["A", "B", "C", "D", "E", "F"],
        "asset_name": ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Phi"],
        "asset_category": ["Stock", "Stock", "Bond", "MTF", "MTF", "Stock"],
    }
)


def _prices() -> pd.DataFrame:
    days = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    rng = np.random.default_rng(0)
    rows = []
    drift = {"A": 0.001, "B": -0.001, "C": 0.0001, "D": 0.00005, "E": 0.0, "F": 0.0008}
    vol = {"A": 0.01, "B": 0.01, "C": 0.002, "D": 0.0002, "E": 0.03, "F": 0.015}
    for isin in drift:
        price = 10.0
        for day in days:
            price *= 1 + drift[isin] + vol[isin] * rng.standard_normal()
            rows.append((isin, day, price))
    return pd.DataFrame(rows, columns=["isin", "price_date", "close_price"])


def _transactions() -> pd.DataFrame:
    t = pd.Timestamp
    rows = [
        ("u1", "A", "buy", t("2020-03-01")),
        ("u1", "B", "buy", t("2020-04-01")),
        ("u2", "A", "buy", t("2020-05-01")),
        ("u2", "C", "buy", t("2020-06-01")),
        ("u3", "A", "buy", t("2020-07-01")),
        ("u3", "B", "buy", t("2020-12-15")),
        ("u4", "D", "buy", t("2020-08-01")),
        ("u1", "A", "sell", t("2020-09-01")),
        ("u1", "C", "buy", t("2021-02-01")),
        ("u1", "A", "buy", t("2021-02-15")),
        ("u2", "B", "buy", t("2021-03-01")),
        ("u3", "E", "buy", t("2021-04-01")),
        ("u5", "A", "buy", t("2021-04-01")),
        ("u4", "F", "buy", t("2021-09-01")),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "isin", "transaction_type", "transaction_date"])
    df["total_value"] = 1000.0
    return df


@pytest.fixture(scope="session")
def ds() -> Dataset:
    return Dataset.from_frames(_transactions(), ASSETS, _prices())


CUTOFF = "2021-01-01"
