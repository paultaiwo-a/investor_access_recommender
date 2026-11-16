"""Tune ALS and BPR on the validation window, then score the best configs on test.

Every configuration is logged to the ``mf`` MLflow experiment. The selection metric is
nDCG@10 on validation; test is touched once per model family, with the chosen config refit on
train + val.
"""

from __future__ import annotations

import itertools
import json

import pandas as pd

from recs.candidates.base import recommend
from recs.candidates.mf import ALSRecommender, BPRRecommender
from recs.config import ARTIFACTS_DIR, K_DEFAULT, N_CANDIDATES
from recs.data import Dataset
from recs.eval.metrics import recall_at_k
from recs.eval.runner import evaluate_model
from recs.pipelines.common import log_metrics, mlflow_run, save_table, test_snapshot, val_snapshot

ALS_GRID = {
    "factors": [8, 16, 32, 64, 128],
    "regularization": [1.0, 10.0, 50.0, 200.0, 500.0],
    "alpha": [5.0, 20.0, 50.0],
    "iterations": [15],
}
BPR_GRID = {
    "factors": [32, 64],
    "learning_rate": [0.01, 0.05, 0.1],
    "regularization": [0.001, 0.01],
    "iterations": [100, 300],
}
FAMILIES = {"als": (ALSRecommender, ALS_GRID), "bpr": (BPRRecommender, BPR_GRID)}
SELECT = f"ndcg@{K_DEFAULT}"


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*space.values())]


def candidate_recall(model, snap, n: int = N_CANDIDATES) -> float:
    recs = recommend(model, snap, k=n)
    users = [u for u in snap.users if snap.targets.get(u)]
    return sum(recall_at_k(recs[u], snap.targets[u], n) for u in users) / len(users)


def main() -> None:
    ds = Dataset.load()
    val, test = val_snapshot(ds), test_snapshot(ds)
    rows, best = [], {}
    for family, (cls, space) in FAMILIES.items():
        for params in grid(space):
            res = evaluate_model(cls(**params), val, K_DEFAULT)
            rows.append({"model": family, **params, **res})
            with mlflow_run("mf", f"{family}-val", {"model": family, **params}):
                log_metrics(res, prefix="val_")
            print(f"{family} {params} -> {SELECT}={res[SELECT]:.4f}")
        fam = pd.DataFrame([r for r in rows if r["model"] == family])
        top = fam.sort_values(SELECT, ascending=False).iloc[0]
        best[family] = {k: top[k] for k in space}

    tuning = pd.DataFrame(rows)
    save_table(tuning.sort_values(["model", SELECT], ascending=[True, False]), "mf_tuning")

    final = []
    for family, params in best.items():
        cls = FAMILIES[family][0]
        params = {
            k: (int(v) if float(v).is_integer() and k != "alpha" else float(v))
            for k, v in params.items()
        }
        best[family] = params
        for split, snap in (("val", val), ("test", test)):
            model = cls(**params).fit(snap)
            res = evaluate_model(model, snap, K_DEFAULT)
            res[f"candidate_recall@{N_CANDIDATES}"] = candidate_recall(model, snap)
            final.append({"split": split, "model": family, **res})
            with mlflow_run("mf", f"{family}-best-{split}", {"model": family, **params}):
                log_metrics(res, prefix=f"{split}_")
    out = pd.DataFrame(final)
    cols = [
        "split",
        "model",
        "ndcg@10",
        "recall@10",
        "map@10",
        "coverage@10",
        "arp@10",
        "monthly_roi@10",
        f"candidate_recall@{N_CANDIDATES}",
    ]
    save_table(out[cols], "mf_fixed_split")
    print(out[cols].round(4).to_string(index=False))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "best_mf_params.json").write_text(json.dumps(best, indent=2))
    print("best params:", best)


if __name__ == "__main__":
    main()
