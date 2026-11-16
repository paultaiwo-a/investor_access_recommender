"""FastAPI service: GET /recommend/{customer_id} returns the top 10 with reasons.

Recommendations are precomputed by the daily batch (``python -m recs.serving``); the service
loads the parquet files at startup and serves lookups.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "serving"

DISCLAIMER = (
    "Suggestions are for information only and are not personal investment advice. "
    "Past performance does not guarantee future results."
)


class Recommendation(BaseModel):
    rank: int
    isin: str
    name: str
    category: str
    subcategory: str | None
    sector: str | None
    risk_class: int
    return_6m: float | None
    score: float | None
    reasons: list[str]


class RecommendationResponse(BaseModel):
    customer_id: str
    risk_level: str
    max_risk_class: int
    as_of: str
    strategy: str  # "personalised" or "cold_start"
    recommendations: list[Recommendation]
    disclaimer: str = DISCLAIMER


class Store:
    """In-memory view of the serving artifacts."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.meta = json.loads((directory / "meta.json").read_text())
        self.assets = pd.read_parquet(directory / "assets.parquet").set_index("isin")
        self.customers = pd.read_parquet(directory / "customers.parquet").set_index("customer_id")
        recs = pd.read_parquet(directory / "recommendations.parquet")
        self.recs = {cid: g.sort_values("rank") for cid, g in recs.groupby("customer_id")}
        fb = pd.read_parquet(directory / "fallback.parquet")
        self.fallback = {lvl: g.sort_values("rank") for lvl, g in fb.groupby("risk_level")}

    def _item(self, row: pd.Series) -> Recommendation:
        a = self.assets.loc[row["isin"]]
        ret = a.get("ret_6m")
        return Recommendation(
            rank=int(row["rank"]),
            isin=row["isin"],
            name=str(a["asset_name"]),
            category=str(a["asset_category"]),
            subcategory=None if pd.isna(a["asset_subcategory"]) else str(a["asset_subcategory"]),
            sector=None if pd.isna(a["sector"]) else str(a["sector"]),
            risk_class=int(a["srri"]),
            return_6m=None if ret is None or pd.isna(ret) else round(float(ret), 4),
            score=None if "score" not in row or pd.isna(row["score"]) else float(row["score"]),
            reasons=json.loads(row["reasons"]),
        )

    def recommend(self, customer_id: str, k: int) -> RecommendationResponse:
        if customer_id not in self.customers.index:
            raise KeyError(customer_id)
        c = self.customers.loc[customer_id]
        risk = str(c["risk_level"])
        if customer_id in self.recs:
            rows, strategy = self.recs[customer_id], "personalised"
        else:
            rows, strategy = self.fallback[risk], "cold_start"
        items = [self._item(r) for _, r in rows.head(k).iterrows()]
        return RecommendationResponse(
            customer_id=customer_id,
            risk_level=risk,
            max_risk_class=int(c["max_srri"]),
            as_of=self.meta["as_of"],
            strategy=strategy,
            recommendations=items,
        )


def create_app(directory: Path | None = None) -> FastAPI:
    directory = Path(directory or os.environ.get("RECS_SERVING_DIR", DEFAULT_DIR))
    state: dict[str, Store] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["store"] = Store(directory)
        yield
        state.clear()

    app = FastAPI(
        title="investor-asset-recs",
        description="Stocks and funds you might like, filtered for suitability.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        s = state["store"]
        return {
            "status": "ok",
            "as_of": s.meta["as_of"],
            "customers": len(s.customers),
            "scored": len(s.recs),
        }

    @app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
    def recommend(customer_id: str, k: int = Query(10, ge=1, le=10)) -> RecommendationResponse:
        try:
            return state["store"].recommend(customer_id, k)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown customer") from exc

    @app.get("/customers/{customer_id}")
    def customer(customer_id: str) -> dict:
        s = state["store"]
        if customer_id not in s.customers.index:
            raise HTTPException(status_code=404, detail="Unknown customer")
        c = s.customers.loc[customer_id]
        holdings = json.loads(c["holdings"])
        return {
            "customer_id": customer_id,
            "customer_type": c["customer_type"],
            "risk_level": c["risk_level"],
            "max_risk_class": int(c["max_srri"]),
            "investment_capacity": c["investment_capacity"],
            "holdings": [
                {
                    "isin": h,
                    "name": str(s.assets.loc[h, "asset_name"]) if h in s.assets.index else h,
                }
                for h in holdings
            ],
        }

    return app


app = create_app()
