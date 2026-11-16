"""Leakage tests: features as of a cutoff must not change when the future changes."""

import numpy as np
import pandas as pd

from recs.candidates.hybrid import HybridCandidates
from recs.data import Dataset, make_snapshot
from recs.ranking.features import FEATURES, attach_labels, build_features, item_features
from tests.conftest import CUTOFF


def _with_future_scrambled(ds: Dataset) -> Dataset:
    cut = pd.Timestamp(CUTOFF)
    prices = ds.prices.copy()
    fut = prices["price_date"] >= cut
    prices.loc[fut, "close_price"] = prices.loc[fut, "close_price"] * 7.0
    tx = ds.transactions.copy()
    extra = pd.DataFrame(
        {
            "customer_id": ["u9"] * 50,
            "isin": ["F"] * 50,
            "transaction_type": ["buy"] * 50,
            "transaction_date": [cut + pd.Timedelta(days=1)] * 50,
            "total_value": 1.0,
            "units": 1.0,
        }
    )
    tx = pd.concat([tx, extra], ignore_index=True)
    return Dataset.from_frames(tx, ds.assets, prices, ds.customers)


def test_item_features_ignore_future(ds):
    a = item_features(ds, pd.Timestamp(CUTOFF))
    b = item_features(_with_future_scrambled(ds), pd.Timestamp(CUTOFF))
    pd.testing.assert_frame_equal(a, b)


def test_item_features_react_to_past(ds):
    early = item_features(ds, pd.Timestamp("2020-06-01"))
    late = item_features(ds, pd.Timestamp(CUTOFF))
    assert early.loc["A", "buyers_all"] < late.loc["A", "buyers_all"]
    # A drifts up, B drifts down in the synthetic prices.
    assert late.loc["A", "ret_6m"] > late.loc["B", "ret_6m"]


def test_srri_orders_by_volatility(ds):
    f = item_features(ds, pd.Timestamp(CUTOFF))
    assert f.loc["D", "srri"] < f.loc["C", "srri"] < f.loc["E", "srri"]


def test_built_frame_has_all_features_and_labels(ds):
    snap = make_snapshot(ds, CUTOFF)
    gen = HybridCandidates({"factors": 4, "iterations": 3}).fit(snap)
    cands = gen.candidates(snap.users, snap.eligible_items, n=5)
    df = attach_labels(build_features(snap, cands), snap)
    assert set(FEATURES) <= set(df.columns)
    assert df["label"].sum() == sum(
        len(snap.targets[u] & set(df.loc[df.customer_id == u, "isin"])) for u in snap.users
    )
    # never a candidate the customer already holds
    for u, items in df.groupby("customer_id")["isin"]:
        assert not set(items) & snap.history_items[u]


def test_user_features_point_in_time(ds):
    snap = make_snapshot(ds, CUTOFF)
    gen = HybridCandidates({"factors": 4, "iterations": 3}).fit(snap)
    df = build_features(snap, gen.candidates(["u1"], snap.eligible_items, n=3))
    row = df.iloc[0]
    assert row["u_n_buys"] == 2  # buys of A and B before the cutoff, not the 2021 ones
    assert row["u_n_sells"] == 1
    assert row["risk_level"] == "Conservative"  # the Aggressive update comes later
    assert np.isclose(row["u_max_srri_allowed"], 3)
