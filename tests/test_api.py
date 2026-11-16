import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture()
def client(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"as_of": "2022-12-01"}))
    pd.DataFrame(
        {
            "isin": ["A", "B"],
            "asset_name": ["Alpha", "Bond Fund"],
            "asset_short_name": ["A", "B"],
            "asset_category": ["Stock", "MTF"],
            "asset_subcategory": [None, "Bond"],
            "sector": ["Energy", None],
            "asset_group": ["Energy", "MTF: Bond"],
            "srri": [6, 3],
            "ret_1m": [0.01, 0.0],
            "ret_6m": [0.12, 0.01],
            "ret_12m": [0.2, 0.02],
            "vol_12m": [0.3, 0.04],
            "buyers_30d": [10, 3],
        }
    ).to_parquet(tmp_path / "assets.parquet")
    pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "customer_type": ["Mass", "Mass"],
            "risk_level": ["Balanced", "Conservative"],
            "investment_capacity": ["CAP_LT30K"] * 2,
            "max_srri": [6, 3],
            "holdings": [json.dumps(["B"]), json.dumps([])],
            "n_past_assets": [1, 0],
        }
    ).to_parquet(tmp_path / "customers.parquet")
    pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "rank": [1, 2],
            "isin": ["A", "B"],
            "score": [0.9, 0.1],
            "reasons": [json.dumps(["r1", "fits"]), json.dumps(["r2"])],
        }
    ).to_parquet(tmp_path / "recommendations.parquet")
    pd.DataFrame(
        {
            "risk_level": ["Conservative"],
            "rank": [1],
            "isin": ["B"],
            "reasons": [json.dumps(["trending"])],
        }
    ).to_parquet(tmp_path / "fallback.parquet")
    with TestClient(create_app(tmp_path)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["scored"] == 1


def test_personalised_recommendations(client):
    body = client.get("/recommend/u1").json()
    assert body["strategy"] == "personalised"
    assert [x["isin"] for x in body["recommendations"]] == ["A", "B"]
    assert body["recommendations"][0]["reasons"] == ["r1", "fits"]
    assert body["recommendations"][0]["risk_class"] <= body["max_risk_class"]
    assert "not personal investment advice" in body["disclaimer"]


def test_k_limits_results(client):
    assert len(client.get("/recommend/u1?k=1").json()["recommendations"]) == 1
    assert client.get("/recommend/u1?k=50").status_code == 422


def test_cold_start_uses_profile_fallback(client):
    body = client.get("/recommend/u2").json()
    assert body["strategy"] == "cold_start"
    assert body["recommendations"][0]["isin"] == "B"


def test_unknown_customer_404(client):
    assert client.get("/recommend/nope").status_code == 404


def test_customer_profile(client):
    body = client.get("/customers/u1").json()
    assert body["holdings"] == [{"isin": "B", "name": "Bond Fund"}]
