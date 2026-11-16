"""Evaluate the baselines on the fixed split and on the FAR-Trans rolling benchmark."""

from __future__ import annotations

import argparse

import pandas as pd

from recs.candidates.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    RecentPopularityRecommender,
)
from recs.config import K_DEFAULT
from recs.data import Dataset
from recs.eval.runner import evaluate_model, market_average, paper_cutoffs, rolling_evaluation
from recs.pipelines.common import (
    log_metrics,
    mlflow_run,
    save_table,
    test_snapshot,
    val_snapshot,
)

FACTORIES = {
    "random": RandomRecommender,
    "popularity": PopularityRecommender,
    "recent_popularity": RecentPopularityRecommender,
    "item_knn": ItemKNNRecommender,
}

COLS = ["ndcg@10", "recall@10", "map@10", "coverage@10", "arp@10", "monthly_roi@10"]


def fixed_split(ds: Dataset, k: int = K_DEFAULT) -> pd.DataFrame:
    rows = []
    for split, snap in (("val", val_snapshot(ds)), ("test", test_snapshot(ds))):
        for name, factory in FACTORIES.items():
            res = evaluate_model(factory(), snap, k)
            rows.append({"split": split, "model": name, **res})
            with mlflow_run("baselines", f"{name}-{split}", {"model": name, "split": split}):
                log_metrics(res)
        rows.append(
            {"split": split, "model": "market_average", f"monthly_roi@{k}": market_average(snap)}
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rolling-step",
        type=int,
        default=1,
        help="Use every n-th of the paper's 61 bi-weekly cutoffs",
    )
    p.add_argument("--skip-rolling", action="store_true")
    args = p.parse_args()

    ds = Dataset.load()
    fixed = fixed_split(ds)
    save_table(
        fixed[["split", "model", *COLS, "pct_positive@10", "n_users", "n_items"]],
        "baselines_fixed_split",
    )
    print(fixed[["split", "model", *COLS]].round(4).to_string(index=False))

    if not args.skip_rolling:
        print("Rolling benchmark (FAR-Trans protocol):")
        roll = rolling_evaluation(ds, FACTORIES, paper_cutoffs(args.rolling_step))
        roll.to_csv("results/baselines_rolling_raw.csv", index=False)
        summary = roll.groupby("model", sort=False)[["ndcg@10", "monthly_roi@10"]].mean()
        save_table(summary.reset_index(), "baselines_rolling")
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
