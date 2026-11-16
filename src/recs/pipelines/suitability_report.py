"""Measure what the suitability layer costs in accuracy on the test window.

Variants for the two-stage ranker:

* ``no rules``: the unconstrained top-10.
* ``post-filter``: risk filter and sector cap applied to the ranker's 50 candidates. Lists
  can come back short when most candidates are too risky.
* ``suitable candidates``: candidate generation only draws from assets the profile allows,
  then the ranker orders them and the sector cap is applied. Lists stay full.

Accuracy is reported against every first-time purchase (``ndcg@10``) and against only the
purchases that were suitable for the buyer (``ndcg_suitable@10``). In FAR-Trans most
customers buy single Greek stocks whatever their profile says, so the first number mostly
measures customer behaviour; the second measures how well a compliant recommender does on
the part of demand it is allowed to serve.
"""

from __future__ import annotations

import pickle

import pandas as pd

from recs.candidates.base import recommend
from recs.candidates.baselines import PopularityRecommender
from recs.config import ARTIFACTS_DIR, K_DEFAULT, N_CANDIDATES, TEST_END, TEST_START
from recs.data import Dataset
from recs.eval.metrics import ranking_metrics
from recs.eval.runner import evaluate_recs
from recs.pipelines.common import save_table
from recs.pipelines.train_ranker import load_mf_params
from recs.ranking.features import item_features
from recs.ranking.ranker import FrameBuilder, rank_frame
from recs.rules.risk import max_srri
from recs.rules.suitability import SuitabilityRules, violations

NO_RULES = SuitabilityRules(enforce_risk=False, enforce_sector_cap=False)
RULES = SuitabilityRules()
COLS = [
    "ndcg@10",
    "ndcg_suitable@10",
    "recall@10",
    "coverage@10",
    "monthly_roi@10",
    "pct_positive@10",
    "avg_list_length",
    "ndcg_cost_pct",
    "ndcg_suitable_cost_pct",
]


def main() -> None:
    ds = Dataset.load()
    als, _ = load_mf_params()
    with open(ARTIFACTS_DIR / "ranker_test.pkl", "rb") as f:
        ranker = pickle.load(f)
    builder = FrameBuilder(ds, als_params=als)
    end = pd.Timestamp(TEST_END)
    df, snap = builder.frame(TEST_START, end=end)
    sdf, _ = builder.frame(TEST_START, end=end, suitable_only=True)

    meta = item_features(ds, snap.cutoff)[["srri", "asset_group"]]
    risk = ds.customer_profile(snap.cutoff)["risk_level"].reindex(snap.users)
    risk = risk.fillna("Not_Available").to_dict()
    srri = meta["srri"].to_dict()
    suitable_targets = {
        u: {i for i in t if srri.get(i, 7) <= max_srri(risk[u])} for u, t in snap.targets.items()
    }
    suitable_targets = {u: t for u, t in suitable_targets.items() if t}

    deep = rank_frame(df, ranker.predict(df), N_CANDIDATES)
    deep_suitable = rank_frame(sdf, ranker.predict(sdf), N_CANDIDATES)
    pop_deep = recommend(PopularityRecommender().fit(snap), snap, k=len(snap.eligible_items))

    variants = [
        ("two_stage_ranker", "no rules", deep, NO_RULES),
        ("two_stage_ranker", "post-filter (risk + sector cap)", deep, RULES),
        ("two_stage_ranker", "suitable candidates + sector cap", deep_suitable, RULES),
        ("popularity", "no rules", pop_deep, NO_RULES),
        ("popularity", "risk + sector cap", pop_deep, RULES),
    ]
    rows, final = [], {}
    for model, name, lists, rules in variants:
        recs = {u: o.items for u, o in rules.apply_all(lists, risk, meta).items()}
        res = evaluate_recs(snap, recs)
        res["ndcg_suitable@10"] = ranking_metrics(recs, suitable_targets, K_DEFAULT)["ndcg@10"]
        res["avg_list_length"] = sum(map(len, recs.values())) / max(len(recs), 1)
        if name == "no rules":
            res.update({f"breaks_{k}": v for k, v in violations(recs, risk, meta).items()})
        rows.append({"model": model, "rules": name, **res})
        final[(model, name)] = recs

    table = pd.DataFrame(rows)
    for col in ("ndcg@10", "ndcg_suitable@10"):
        base = table.groupby("model")[col].transform("first")
        table[col.replace("@10", "_cost_pct")] = 100 * (table[col] - base) / base
    save_table(table[["model", "rules", *COLS]], "suitability_cost")
    print(table[["model", "rules", *COLS]].round(4).to_string(index=False))
    breaks = table.filter(like="breaks_").assign(model=table["model"]).dropna()
    save_table(breaks, "suitability_violations_without_rules")
    print(breaks.round(3).to_string(index=False))

    # Breakdown by profile for the ranker (unconstrained vs suitable candidates)
    a_recs = final[("two_stage_ranker", "no rules")]
    b_recs = final[("two_stage_ranker", "suitable candidates + sector cap")]
    by = []
    for lvl in sorted(set(risk.values())):
        users = [u for u in snap.users if risk[u] == lvl and snap.targets.get(u)]
        if not users:
            continue
        tg = {u: snap.targets[u] for u in users}
        a = ranking_metrics(a_recs, tg, K_DEFAULT)["ndcg@10"]
        b = ranking_metrics(b_recs, tg, K_DEFAULT)["ndcg@10"]
        n_t = sum(len(t) for t in tg.values())
        n_ok = sum(len(suitable_targets.get(u, ())) for u in users)
        by.append(
            {
                "risk_level": lvl,
                "max_srri": max_srri(lvl),
                "n_users": len(users),
                "ndcg_no_rules": a,
                "ndcg_with_rules": b,
                "ndcg_cost_pct": 100 * (b - a) / a if a else float("nan"),
                "own_buys_above_profile_pct": 100 * (1 - n_ok / n_t),
            }
        )
    by_df = pd.DataFrame(by)
    save_table(by_df, "suitability_by_profile")
    print(by_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
