"""Batch scoring for serving.

``build`` trains the full pipeline for deployment on the day after the last transaction,
scores every customer with history, applies the suitability layer and writes:

* ``artifacts/serving/recommendations.parquet``: top-k per customer with score and reasons
* ``artifacts/serving/assets.parquet``: asset metadata and risk class as of the serving date
* ``artifacts/serving/customers.parquet``: profile and holdings summary per customer
* ``artifacts/serving/fallback.parquet``: suitable trending assets per risk level (cold start)
* ``artifacts/serving/meta.json``: serving date and model settings

This is the daily batch pattern many fintech apps use: the API only does lookups, so it is
fast and cheap, and the heavy work runs once a day.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from recs.candidates.hybrid import HybridCandidates
from recs.config import ARTIFACTS_DIR, K_DEFAULT, N_CANDIDATES
from recs.data import Dataset, serving_snapshot
from recs.ranking.features import attach_labels, build_features, item_features
from recs.ranking.ranker import FrameBuilder, LambdaRanker
from recs.rules.risk import MAX_SRRI, RISK_LEVELS, max_srri
from recs.rules.suitability import SuitabilityRules

SERVING_DIR = ARTIFACTS_DIR / "serving"
CHUNK = 4000
CATEGORY_LABEL = {"Stock": "stock", "Bond": "bond", "MTF": "mutual fund"}
# Stocks whose price has not moved for this many days are treated as suspended and not served.
STALE_STOCK_DAYS = 30


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def reasons_for(row: pd.Series, names: dict[str, str], similar_to: str | None) -> list[str]:
    """Plain-language reasons, most specific first. The suitability line is always last."""
    out: list[str] = []
    if similar_to:
        name = names.get(similar_to, similar_to)
        out.append(f"Investors who bought {name}, as you did, often bought this too")
    if row.get("als_rank", 99) < 10 and not similar_to:
        out.append("Popular with investors whose portfolios look like yours")
    if row.get("buyers_30d_rank", 999) <= 20 and row.get("buyers_30d", 0) > 0:
        out.append(f"Trending: {int(row['buyers_30d'])} customers bought it in the last 30 days")
    ret = row.get("ret_6m")
    if ret is not None and not pd.isna(ret) and ret > 0.05:
        out.append(
            f"Price {_pct(ret)} over the past 6 months (past returns do not guarantee "
            "future results)"
        )
    if row.get("ui_n_same_category", 1) == 0:
        label = CATEGORY_LABEL.get(str(row["asset_category"]), "new")
        out.append(f"Adds a {label} to a portfolio that has none yet")
    out = out[:3]
    out.append(
        f"Fits your {row['risk_level']} risk profile (risk class {int(row['srri'])} of 7, "
        f"your limit is {int(row['u_max_srri_allowed'])})"
    )
    return out


def _chunks(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def build(
    ds: Dataset,
    als_params: dict | None = None,
    num_boost_round: int = 400,
    out_dir: Path = SERVING_DIR,
    k: int = K_DEFAULT,
    as_of: pd.Timestamp | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = as_of or ds.transactions["transaction_date"].max() + pd.Timedelta(days=1)

    print(f"Training ranker for deployment on {as_of.date()} ...")
    builder = FrameBuilder(ds, als_params=als_params)
    ranker = LambdaRanker(num_boost_round=num_boost_round).fit(builder.training_frame(as_of))
    builder._cache.clear()

    snap = serving_snapshot(ds, as_of)
    gen = HybridCandidates(als_params).fit(snap)
    knn = gen.models["item_knn"]
    items = item_features(ds, as_of)
    names = ds.assets.set_index("isin")["asset_name"].to_dict()
    profile = ds.customer_profile(as_of)
    rules = SuitabilityRules(k=k)
    stale = (items["asset_category"] == "Stock") & (items["stale_days"] > STALE_STOCK_DAYS)
    tradable = [i for i in snap.eligible_items if not stale.get(i, False)]
    srri = items["srri"].reindex(tradable)

    rows = []
    for i, users in enumerate(_chunks(snap.users, CHUNK)):
        limits = pd.Series([max_srri(profile["risk_level"].get(u)) for u in users], index=users)
        parts = []
        for limit, group in limits.groupby(limits):
            pool = srri.index[srri <= limit].tolist()
            parts.append(gen.candidates(group.index.tolist(), pool, n=N_CANDIDATES))
        cands = pd.concat(parts, ignore_index=True)
        df = attach_labels(build_features(snap, cands, items), snap)
        df["score"] = ranker.predict(df)
        df = df.sort_values(["customer_id", "score"], ascending=[True, False])
        meta = items[["srri", "asset_group"]]
        for u, g in df.groupby("customer_id", sort=False):
            kept = rules.apply(
                g["isin"].tolist(),
                g["risk_level"].iloc[0],
                meta["srri"].to_dict(),
                meta["asset_group"].to_dict(),
            ).items
            g = g.set_index("isin").loc[kept]
            for rank, (isin, r) in enumerate(g.iterrows(), start=1):
                similar = knn.explain(u, isin)
                similar_to = similar[0][0] if similar and r["item_knn_rank"] < 15 else None
                rows.append(
                    {
                        "customer_id": u,
                        "rank": rank,
                        "isin": isin,
                        "score": float(r["score"]),
                        "reasons": json.dumps(reasons_for(r, names, similar_to)),
                    }
                )
        print(f"  scored chunk {i + 1}: {len(users)} customers")
    recs = pd.DataFrame(rows)
    recs.to_parquet(out_dir / "recommendations.parquet", index=False)

    # Asset metadata for display
    assets = ds.assets.set_index("isin")[
        [
            "asset_name",
            "asset_short_name",
            "asset_category",
            "asset_subcategory",
            "sector",
            "asset_group",
        ]
    ]
    assets = assets.join(items[["srri", "ret_1m", "ret_6m", "ret_12m", "vol_12m", "buyers_30d"]])
    assets.reset_index().to_parquet(out_dir / "assets.parquet", index=False)

    # Customer profiles + holdings (net units > 0 as of serving date)
    tx = ds.transactions
    signed = np.where(tx["transaction_type"] == "buy", tx["units"], -tx["units"])
    pos = tx.assign(net=signed).groupby(["customer_id", "isin"])["net"].sum()
    holdings = pos[pos > 1e-9].reset_index().groupby("customer_id")["isin"].agg(list)
    cust = profile[["customer_type", "risk_level", "investment_capacity"]].copy()
    cust["risk_level"] = cust["risk_level"].fillna("Not_Available")
    cust["max_srri"] = cust["risk_level"].map(max_srri)
    cust["holdings"] = holdings.reindex(cust.index).apply(
        lambda v: json.dumps(v if isinstance(v, list) else [])
    )
    cust["n_past_assets"] = snap.history.groupby("customer_id").size().reindex(cust.index)
    cust["n_past_assets"] = cust["n_past_assets"].fillna(0).astype(int)
    cust.reset_index().to_parquet(out_dir / "customers.parquet", index=False)

    # Cold start: trending suitable assets per risk level, sector-capped
    trending = items.loc[tradable].sort_values(["buyers_30d", "buyers_90d"], ascending=False)
    fb = []
    n_buyers = items["buyers_30d"].fillna(0).astype(int)
    srri_int = items["srri"].astype(int)
    for lvl in [*RISK_LEVELS, "Not_Available"]:
        kept = rules.apply(
            trending.index.tolist(), lvl, items["srri"].to_dict(), items["asset_group"].to_dict()
        ).items
        for rank, isin in enumerate(kept, start=1):
            fb.append(
                {
                    "risk_level": lvl,
                    "rank": rank,
                    "isin": isin,
                    "reasons": json.dumps(
                        [
                            f"Trending: {n_buyers[isin]} customers bought it in the last 30 days",
                            f"Fits a {lvl} risk profile (risk class {srri_int[isin]} of 7, "
                            f"limit {MAX_SRRI[lvl]})",
                        ]
                    ),
                }
            )
    pd.DataFrame(fb).to_parquet(out_dir / "fallback.parquet", index=False)

    meta = {
        "as_of": str(as_of.date()),
        "k": k,
        "n_customers_scored": int(recs["customer_id"].nunique()),
        "n_candidates": N_CANDIDATES,
        "n_tradable_assets": len(tradable),
        "n_stale_stocks_excluded": len(snap.eligible_items) - len(tradable),
        "num_boost_round": num_boost_round,
        "als_params": als_params,
        "sector_cap": rules.sector_cap,
        "max_srri_by_profile": MAX_SRRI,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main() -> None:
    from recs.pipelines.train_ranker import load_mf_params

    cfg_path = ARTIFACTS_DIR / "ranker_config.json"
    rounds = json.loads(cfg_path.read_text())["num_boost_round"] if cfg_path.exists() else 400
    ds = Dataset.load()
    als, _ = load_mf_params()
    meta = build(ds, als_params=als, num_boost_round=rounds)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
