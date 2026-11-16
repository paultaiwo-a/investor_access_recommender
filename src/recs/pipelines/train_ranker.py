"""Train the LambdaRank ranker and compare it with every baseline.

1. Tune the number of boosting rounds on the validation window (ranker trained on cutoffs
   before VAL_START).
2. Retrain with the chosen rounds on cutoffs before TEST_START and evaluate on test, next to
   popularity, recent popularity, item-kNN, ALS, BPR and the fused candidate list.
3. Optionally run the FAR-Trans rolling benchmark (``--rolling-step``).
"""

from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd

from recs.candidates.base import recommend
from recs.candidates.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    RecentPopularityRecommender,
)
from recs.candidates.hybrid import DEFAULT_ALS
from recs.candidates.mf import ALSRecommender, BPRRecommender
from recs.config import ARTIFACTS_DIR, K_DEFAULT, TEST_END, TEST_START, VAL_START
from recs.data import Dataset
from recs.eval.runner import evaluate_recs, market_average, paper_cutoffs
from recs.pipelines.common import log_metrics, mlflow_run, save_table
from recs.ranking.ranker import TRAIN_OFFSETS_DAYS, FrameBuilder, LambdaRanker, rank_frame

ROUNDS_GRID = [100, 200, 400, 800]
COLS = [
    "ndcg@10",
    "recall@10",
    "map@10",
    "coverage@10",
    "arp@10",
    "monthly_roi@10",
    "pct_positive@10",
]


def load_mf_params() -> tuple[dict, dict]:
    path = ARTIFACTS_DIR / "best_mf_params.json"
    if path.exists():
        best = json.loads(path.read_text())
        return best["als"], best["bpr"]
    return DEFAULT_ALS, {}


def baseline_factories(als: dict, bpr: dict) -> dict:
    return {
        "random": RandomRecommender,
        "popularity": PopularityRecommender,
        "recent_popularity": RecentPopularityRecommender,
        "item_knn": ItemKNNRecommender,
        "als": lambda: ALSRecommender(**als),
        "bpr": lambda: BPRRecommender(**bpr),
    }


def evaluate_at(builder: FrameBuilder, ranker: LambdaRanker, cutoff, end, factories) -> list[dict]:
    """Evaluate every model on the same snapshot. The fused list and the ranker share frames."""
    df, snap = builder.frame(cutoff, end=end)
    rows = []
    for name, factory in factories.items():
        recs = recommend(factory().fit(snap), snap, k=K_DEFAULT)
        rows.append({"model": name, **evaluate_recs(snap, recs)})
    rows.append(
        {
            "model": "hybrid_rrf",
            **evaluate_recs(snap, rank_frame(df, df["rrf_score"].to_numpy(), K_DEFAULT)),
        }
    )
    rows.append(
        {
            "model": "two_stage_ranker",
            **evaluate_recs(snap, rank_frame(df, ranker.predict(df), K_DEFAULT)),
        }
    )
    rows.append({"model": "market_average", f"monthly_roi@{K_DEFAULT}": market_average(snap)})
    return rows


def tune_rounds(builder: FrameBuilder) -> tuple[int, pd.DataFrame]:
    train = builder.training_frame(pd.Timestamp(VAL_START))
    val_df, val_snap = builder.frame(VAL_START, end=TEST_START)
    rows = []
    for n in ROUNDS_GRID:
        ranker = LambdaRanker(num_boost_round=n).fit(train)
        res = evaluate_recs(val_snap, rank_frame(val_df, ranker.predict(val_df), K_DEFAULT))
        rows.append({"num_boost_round": n, **res})
        with mlflow_run(
            "ranker",
            f"ranker-val-{n}",
            {**ranker.params, "num_boost_round": n, "train_cutoffs": len(TRAIN_OFFSETS_DAYS)},
        ):
            log_metrics(res, prefix="val_")
        print(f"rounds={n}: val ndcg@10={res['ndcg@10']:.4f}")
    tuning = pd.DataFrame(rows)
    best = int(tuning.sort_values("ndcg@10", ascending=False).iloc[0]["num_boost_round"])
    return best, tuning


def rolling(builder: FrameBuilder, rounds: int, factories: dict, step: int) -> pd.DataFrame:
    rows = []
    for t in paper_cutoffs(step):
        ranker = LambdaRanker(num_boost_round=rounds).fit(builder.training_frame(t))
        for r in evaluate_at(builder, ranker, t, None, factories):
            rows.append({"cutoff": t, **r})
        # Drop frames no later cutoff will need.
        oldest = t - pd.Timedelta(days=max(TRAIN_OFFSETS_DAYS))
        for key in [k for k in builder._cache if k[0] < oldest]:
            del builder._cache[key]
        ndcg = rows[-2]["ndcg@10"]
        print(f"  {t.date()}: ranker ndcg@10={ndcg:.4f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rolling-step",
        type=int,
        default=0,
        help="Also run the paper's rolling benchmark on every n-th cutoff (0 = skip)",
    )
    args = p.parse_args()

    ds = Dataset.load()
    als, bpr = load_mf_params()
    builder = FrameBuilder(ds, als_params=als)
    factories = baseline_factories(als, bpr)

    rounds, tuning = tune_rounds(builder)
    save_table(tuning[["num_boost_round", *COLS]], "ranker_tuning")
    print(f"chosen rounds: {rounds}")

    # Validation comparison (ranker trained before VAL_START).
    val_ranker = LambdaRanker(num_boost_round=rounds).fit(
        builder.training_frame(pd.Timestamp(VAL_START))
    )
    val_rows = evaluate_at(builder, val_ranker, VAL_START, pd.Timestamp(TEST_START), factories)

    # Test: retrain on cutoffs before TEST_START, evaluate once.
    test_ranker = LambdaRanker(num_boost_round=rounds).fit(
        builder.training_frame(pd.Timestamp(TEST_START))
    )
    test_rows = evaluate_at(builder, test_ranker, TEST_START, pd.Timestamp(TEST_END), factories)
    with mlflow_run("ranker", "ranker-test", {**test_ranker.params, "num_boost_round": rounds}):
        log_metrics(next(r for r in test_rows if r["model"] == "two_stage_ranker"), prefix="test_")

    table = pd.DataFrame(
        [{"split": "val", **r} for r in val_rows] + [{"split": "test", **r} for r in test_rows]
    )
    save_table(table[["split", "model", *COLS]], "ranker_fixed_split")
    print(table[["split", "model", *COLS]].round(4).to_string(index=False))

    imp = test_ranker.feature_importance()
    save_table(
        imp.rename("gain").reset_index().rename(columns={"index": "feature"}),
        "ranker_feature_importance",
        float_fmt="{:.1f}",
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "ranker_test.pkl", "wb") as f:
        pickle.dump(test_ranker, f)
    (ARTIFACTS_DIR / "ranker_config.json").write_text(json.dumps({"num_boost_round": rounds}))

    if args.rolling_step:
        print("Rolling benchmark (FAR-Trans protocol):")
        roll = rolling(builder, rounds, factories, args.rolling_step)
        roll.to_csv("results/rolling_all_models_raw.csv", index=False)
        summary = (
            roll.groupby("model", sort=False)[
                ["ndcg@10", "recall@10", "coverage@10", "monthly_roi@10"]
            ]
            .mean()
            .reset_index()
        )
        summary.insert(1, "n_cutoffs", roll["cutoff"].nunique())
        save_table(summary, "rolling_all_models")
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
