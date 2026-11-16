"""Point-in-time ranker features.

Everything here is computed "as of" a cutoff date ``t`` using only prices and transactions
strictly before ``t``. Labels come from the window after ``t`` and are attached separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recs.data import Dataset, Snapshot
from recs.rules.risk import RISK_ORDINAL, asset_srri, max_srri

TRADING_DAYS = 252
CATEGORICAL = ["asset_category", "asset_subcategory", "customer_type"]
CAPACITY_ORDINAL = {"CAP_LT_30K": 0, "CAP_30K_80K": 1, "CAP_80K_300K": 2, "CAP_GT300K": 3}


# ---------------------------------------------------------------------- item features
def price_features(ds: Dataset, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Trailing returns, volatility and drawdown per asset, from prices before ``cutoff``."""
    wide = ds.price_wide
    hist = wide[wide.index < cutoff]
    last = hist.iloc[-1] if len(hist) else pd.Series(np.nan, index=wide.columns)

    def ret(days: int) -> pd.Series:
        past = hist[hist.index < cutoff - pd.Timedelta(days=days)]
        if not len(past):
            return pd.Series(np.nan, index=wide.columns)
        return last / past.iloc[-1] - 1.0

    # Daily log returns on business days only (calendar index is forward filled).
    bd = hist[hist.index.dayofweek < 5]
    logret = np.log(bd).diff()

    def vol(days: int) -> pd.Series:
        w = logret[logret.index >= cutoff - pd.Timedelta(days=days)]
        v = w.std() * np.sqrt(TRADING_DAYS)
        return v.where(w.count() >= 20)

    y = bd[bd.index >= cutoff - pd.Timedelta(days=365)]
    drawdown = (y / y.cummax() - 1.0).min()
    first_price = ds.prices.groupby("isin")["price_date"].min()
    out = pd.DataFrame(
        {
            "ret_1m": ret(30),
            "ret_3m": ret(91),
            "ret_6m": ret(182),
            "ret_12m": ret(365),
            "vol_3m": vol(91),
            "vol_12m": vol(365),
            "max_drawdown_12m": drawdown,
            "days_priced": (cutoff - first_price).dt.days.reindex(wide.columns),
        }
    )
    out["sharpe_12m"] = out["ret_12m"] / out["vol_12m"].replace(0, np.nan)
    out.index.name = "isin"
    return out


def popularity_features(ds: Dataset, cutoff: pd.Timestamp) -> pd.DataFrame:
    tx = ds.transactions[ds.transactions["transaction_date"] < cutoff]
    buys = tx[tx["transaction_type"] == "buy"]
    sells = tx[tx["transaction_type"] == "sell"]

    def buyers(frame: pd.DataFrame, start_days: int, end_days: int = 0) -> pd.Series:
        lo = cutoff - pd.Timedelta(days=start_days)
        hi = cutoff - pd.Timedelta(days=end_days)
        w = frame[(frame["transaction_date"] >= lo) & (frame["transaction_date"] < hi)]
        return w.groupby("isin")["customer_id"].nunique()

    out = pd.DataFrame(
        {
            "buyers_all": buys.groupby("isin")["customer_id"].nunique(),
            "buyers_7d": buyers(buys, 7),
            "buyers_30d": buyers(buys, 30),
            "buyers_90d": buyers(buys, 90),
            "buyers_prev_90d": buyers(buys, 180, 90),
            "sellers_90d": buyers(sells, 90),
            "days_since_last_buy_item": (
                cutoff - buys.groupby("isin")["transaction_date"].max()
            ).dt.days,
        }
    ).fillna(
        {
            "buyers_all": 0,
            "buyers_7d": 0,
            "buyers_30d": 0,
            "buyers_90d": 0,
            "buyers_prev_90d": 0,
            "sellers_90d": 0,
        }
    )
    out["pop_trend"] = (out["buyers_90d"] + 1) / (out["buyers_prev_90d"] + 1)
    # Shares of all active buyers, so the scale is comparable across cutoffs.
    for w in (7, 30, 90):
        active = buys[buys["transaction_date"] >= cutoff - pd.Timedelta(days=w)]
        n = max(active["customer_id"].nunique(), 1)
        out[f"buyer_share_{w}d"] = out[f"buyers_{w}d"] / n
    out["buyers_30d_rank"] = out["buyers_30d"].rank(ascending=False, method="min")
    out["net_flow_90d"] = (out["buyers_90d"] - out["sellers_90d"]) / (
        out["buyers_90d"] + out["sellers_90d"] + 1
    )
    out.index.name = "isin"
    return out


def item_features(ds: Dataset, cutoff: pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    pf = price_features(ds, cutoff)
    pop = popularity_features(ds, cutoff)
    meta = ds.assets.set_index("isin")[["asset_category", "asset_subcategory", "asset_group"]]
    out = meta.join(pf, how="left").join(pop, how="left")
    out["srri"] = asset_srri(out["vol_12m"], ds.assets)
    return out


# ---------------------------------------------------------------------- user features
def user_features(snap: Snapshot, items: pd.DataFrame, users: list[str]) -> pd.DataFrame:
    ds, cutoff = snap.ds, snap.cutoff
    tx = ds.transactions[ds.transactions["transaction_date"] < cutoff]
    tx = tx[tx["customer_id"].isin(set(users))]
    buys = tx[tx["transaction_type"] == "buy"]
    g = buys.groupby("customer_id")
    out = pd.DataFrame(
        {
            "u_n_buys": g.size(),
            "u_n_assets": g["isin"].nunique(),
            "u_n_buys_90d": buys[buys["transaction_date"] >= cutoff - pd.Timedelta(days=90)]
            .groupby("customer_id")
            .size(),
            "u_days_since_last_buy": (cutoff - g["transaction_date"].max()).dt.days,
            "u_days_since_first_buy": (cutoff - g["transaction_date"].min()).dt.days,
            "u_log_total_value": np.log1p(g["total_value"].sum().clip(lower=0)),
            "u_log_avg_ticket": np.log1p(g["total_value"].mean().clip(lower=0)),
            "u_n_sells": tx[tx["transaction_type"] == "sell"].groupby("customer_id").size(),
        }
    ).reindex(users)
    out[["u_n_buys_90d", "u_n_sells"]] = out[["u_n_buys_90d", "u_n_sells"]].fillna(0)

    held = snap.history[snap.history["customer_id"].isin(set(users))].merge(
        items[["asset_category", "srri", "vol_12m"]],
        left_on="isin",
        right_index=True,
        how="left",
    )
    shares = pd.crosstab(held["customer_id"], held["asset_category"], normalize="index")
    for cat in ("Stock", "Bond", "MTF"):
        out[f"u_share_{cat.lower()}"] = shares.get(cat, pd.Series(dtype=float)).reindex(users)
    hg = held.groupby("customer_id")
    out["u_avg_srri"] = hg["srri"].mean()
    out["u_max_srri_held"] = hg["srri"].max()
    out["u_avg_vol"] = hg["vol_12m"].mean()

    prof = ds.customer_profile(cutoff).reindex(users)
    out["risk_level"] = prof["risk_level"].fillna("Not_Available")
    out["u_risk_ordinal"] = out["risk_level"].map(RISK_ORDINAL).astype(float)
    out["u_max_srri_allowed"] = out["risk_level"].map(max_srri).astype(float)
    out["u_capacity"] = prof["investment_capacity"].map(CAPACITY_ORDINAL).astype(float)
    out["customer_type"] = prof["customer_type"].fillna("Unknown")
    out.index.name = "customer_id"
    return out


# ---------------------------------------------------------------------- assemble
def build_features(
    snap: Snapshot, cands: pd.DataFrame, items: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join candidate scores with item, user and user x item features."""
    items = item_features(snap.ds, snap.cutoff) if items is None else items
    users = cands["customer_id"].unique().tolist()
    uf = user_features(snap, items, users)
    df = cands.merge(items, left_on="isin", right_index=True, how="left")
    df = df.merge(uf, left_on="customer_id", right_index=True, how="left")

    # user x item: overlap with what the customer already holds
    held = snap.history[snap.history["customer_id"].isin(set(users))].merge(
        items[["asset_category", "asset_group"]], left_on="isin", right_index=True, how="left"
    )
    cat_n = held.groupby(["customer_id", "asset_category"]).size().rename("ui_n_same_category")
    grp_n = held.groupby(["customer_id", "asset_group"]).size().rename("ui_n_same_group")
    df = df.merge(cat_n, left_on=["customer_id", "asset_category"], right_index=True, how="left")
    df = df.merge(grp_n, left_on=["customer_id", "asset_group"], right_index=True, how="left")
    df[["ui_n_same_category", "ui_n_same_group"]] = df[
        ["ui_n_same_category", "ui_n_same_group"]
    ].fillna(0)
    df["ui_share_same_category"] = df["ui_n_same_category"] / df["u_n_assets"].clip(lower=1)
    df["ui_srri_gap"] = df["srri"] - df["u_avg_srri"]
    df["ui_srri_over_allowed"] = df["srri"] - df["u_max_srri_allowed"]
    df["ui_vol_gap"] = df["vol_12m"] - df["u_avg_vol"]

    return set_categories(df, snap.ds)


def set_categories(df: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Use the same category levels at train and predict time (LightGBM needs this)."""
    levels = {
        "asset_category": ds.assets["asset_category"],
        "asset_subcategory": ds.assets["asset_subcategory"],
        "customer_type": pd.concat([ds.customers["customer_type"], pd.Series(["Unknown"])]),
    }
    for c in CATEGORICAL:
        cats = sorted(levels[c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype("string"), categories=cats)
    return df


def attach_labels(df: pd.DataFrame, snap: Snapshot) -> pd.DataFrame:
    pairs = {(u, i) for u, its in snap.targets.items() for i in its}
    keys = zip(df["customer_id"], df["isin"], strict=True)
    df = df.copy()
    df["label"] = np.fromiter((k in pairs for k in keys), dtype=np.int8, count=len(df))
    return df


# Absolute counts (all-time buyers, days listed, raw popularity scores) drift in scale from one
# cutoff to the next and made the ranker overfit to the training period, so the model only
# sees ranks, shares and ratios for popularity. The raw columns stay in the frame for reasons.
FEATURES = [
    # candidate scores and ranks
    "rrf_score", "rrf_rank",
    "als_score", "als_rank", "item_knn_score", "item_knn_rank",
    "recent_popularity_rank", "popularity_rank",
    # item: price behaviour
    "asset_category", "asset_subcategory",
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_3m", "vol_12m", "max_drawdown_12m",
    "sharpe_12m", "srri",
    # item: popularity trend
    "buyer_share_7d", "buyer_share_30d", "buyer_share_90d", "buyers_30d_rank",
    "days_since_last_buy_item", "pop_trend", "net_flow_90d",
    # user
    "u_n_buys", "u_n_assets", "u_n_buys_90d", "u_days_since_last_buy",
    "u_log_total_value", "u_log_avg_ticket", "u_n_sells", "u_share_stock", "u_share_bond",
    "u_share_mtf", "u_avg_srri", "u_max_srri_held", "u_avg_vol", "u_risk_ordinal",
    "u_max_srri_allowed", "u_capacity", "customer_type",
    # user x item
    "ui_n_same_category", "ui_n_same_group", "ui_share_same_category", "ui_srri_gap",
    "ui_srri_over_allowed", "ui_vol_gap",
]  # fmt: skip
