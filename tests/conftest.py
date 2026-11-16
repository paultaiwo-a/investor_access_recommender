"""Tiny synthetic FAR-Trans-shaped dataset so tests never need the real download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recs.data import Dataset

ASSETS = pd.DataFrame(
    {
        "isin": ["A", "B", "C", "D", "E", "F"],
        "asset_name": ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Phi"],
        "asset_short_name": ["ALP", "BET", "GAM", "DEL", "EPS", "PHI"],
        "asset_category": ["Stock", "Stock", "Bond", "MTF", "MTF", "Stock"],
        "asset_subcategory": [
            "Unspecified",
            "Unspecified",
            "Corporate",
            "Money Market",
            "Equity",
            "Unspecified",
        ],
        "market_id": ["XATH"] * 6,
        "sector": ["Energy", "Energy", "Corporate", None, None, "Technology"],
        "industry": [None] * 6,
        "asset_group": [
            "Energy",
            "Energy",
            "Corporate",
            "MTF: Money Market",
            "MTF: Equity",
            "Technology",
        ],
    }
)


def _prices() -> pd.DataFrame:
    days = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    rng = np.random.default_rng(0)
    rows = []
    # Daily drift per asset: A rises, B falls, C flat-ish bond, D money market, E volatile, F up.
    drift = {"A": 0.001, "B": -0.001, "C": 0.0001, "D": 0.00005, "E": 0.0, "F": 0.0008}
    vol = {"A": 0.01, "B": 0.01, "C": 0.002, "D": 0.0002, "E": 0.03, "F": 0.015}
    for isin in drift:
        p = 10.0
        for d in days:
            p *= 1 + drift[isin] + vol[isin] * rng.standard_normal()
            rows.append((isin, d, p))
    return pd.DataFrame(rows, columns=["isin", "price_date", "close_price"])


def _transactions() -> pd.DataFrame:
    t = pd.Timestamp
    rows = [
        # before the 2021-01-01 cutoff
        ("u1", "A", "buy", t("2020-03-01")),
        ("u1", "B", "buy", t("2020-04-01")),
        ("u2", "A", "buy", t("2020-05-01")),
        ("u2", "C", "buy", t("2020-06-01")),
        ("u3", "A", "buy", t("2020-07-01")),
        ("u3", "B", "buy", t("2020-12-15")),
        ("u4", "D", "buy", t("2020-08-01")),
        ("u1", "A", "sell", t("2020-09-01")),
        # after the cutoff
        ("u1", "C", "buy", t("2021-02-01")),
        ("u1", "A", "buy", t("2021-02-15")),  # repeat buy, must not be a target
        ("u2", "B", "buy", t("2021-03-01")),
        ("u3", "E", "buy", t("2021-04-01")),
        ("u5", "A", "buy", t("2021-04-01")),  # cold-start customer, not evaluated
        ("u4", "F", "buy", t("2021-09-01")),  # outside a 180-day horizon
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "isin", "transaction_type", "transaction_date"])
    df["total_value"] = 1000.0
    df["units"] = 10.0
    return df


CUSTOMERS = pd.DataFrame(
    {
        "customer_id": ["u1", "u2", "u3", "u4", "u5", "u1"],
        "customer_type": ["Mass"] * 6,
        "risk_level": [
            "Conservative",
            "Balanced",
            "Aggressive",
            "Income",
            "Not_Available",
            "Aggressive",
        ],
        "risk_level_is_predicted": [False] * 6,
        "investment_capacity": ["CAP_LT_30K"] * 6,
        "last_questionnaire_date": [pd.NaT] * 6,
        "valid_from": pd.to_datetime(
            ["2019-01-01", "2019-01-01", "2019-01-01", "2019-01-01", "2019-01-01", "2021-06-01"]
        ),
        "valid_to": pd.to_datetime([pd.NaT] * 6),
    }
)


@pytest.fixture(scope="session")
def ds() -> Dataset:
    return Dataset.from_frames(_transactions(), ASSETS, _prices(), CUSTOMERS)


CUTOFF = "2021-01-01"
