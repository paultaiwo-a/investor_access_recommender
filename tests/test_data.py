import pandas as pd

from recs.data import make_snapshot
from tests.conftest import CUTOFF


def test_history_only_uses_buys_before_cutoff(ds):
    snap = make_snapshot(ds, CUTOFF)
    assert snap.history["last_date"].max() < pd.Timestamp(CUTOFF)
    assert set(snap.history["customer_id"]) == {"u1", "u2", "u3", "u4"}


def test_targets_are_first_time_buys_in_window(ds):
    snap = make_snapshot(ds, CUTOFF)
    # u1 buys A again, so that one is not a target. u4 buys F after the window.
    assert snap.targets == {"u1": {"C"}, "u2": {"B"}, "u3": {"E"}}
    assert snap.users == ["u1", "u2", "u3"]


def test_cold_start_customer_is_left_out(ds):
    snap = make_snapshot(ds, CUTOFF)
    assert "u5" not in snap.users


def test_forward_returns_match_prices(ds):
    snap = make_snapshot(ds, CUTOFF)
    p0 = ds.price_on(snap.cutoff)["A"]
    p1 = ds.price_on(snap.end)["A"]
    assert abs(snap.forward_returns["A"] - (p1 / p0 - 1)) < 1e-12
