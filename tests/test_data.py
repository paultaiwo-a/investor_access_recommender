import pandas as pd

from recs.data import make_snapshot
from tests.conftest import CUTOFF


def test_history_only_uses_buys_before_cutoff(ds):
    snap = make_snapshot(ds, CUTOFF)
    assert snap.history["last_date"].max() < pd.Timestamp(CUTOFF)
    assert set(snap.history["customer_id"]) == {"u1", "u2", "u3", "u4"}


def test_targets_are_first_time_buys_in_window(ds):
    snap = make_snapshot(ds, CUTOFF)
    # u1 re-buys A (held) -> excluded; u4's F buy falls outside the 180-day horizon.
    assert snap.targets == {"u1": {"C"}, "u2": {"B"}, "u3": {"E"}}
    assert snap.users == ["u1", "u2", "u3"]


def test_cold_start_users_not_evaluated(ds):
    snap = make_snapshot(ds, CUTOFF)
    assert "u5" not in snap.users


def test_forward_returns_match_prices(ds):
    snap = make_snapshot(ds, CUTOFF)
    p0 = ds.price_on(snap.cutoff)["A"]
    p1 = ds.price_on(snap.end)["A"]
    assert abs(snap.forward_returns["A"] - (p1 / p0 - 1)) < 1e-12


def test_customer_profile_is_point_in_time(ds):
    before = ds.customer_profile(pd.Timestamp("2021-01-01"))
    after = ds.customer_profile(pd.Timestamp("2021-07-01"))
    assert before.loc["u1", "risk_level"] == "Conservative"
    assert after.loc["u1", "risk_level"] == "Aggressive"


def test_interaction_matrix_shape(ds):
    snap = make_snapshot(ds, CUTOFF)
    x = snap.interaction_matrix()
    assert x.shape == (len(ds.user_ids), len(ds.item_ids))
    assert x.sum() == len(snap.history)
