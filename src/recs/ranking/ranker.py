"""Stage two: LightGBM LambdaRank over fused candidates.

Training protocol for a model deployed at cutoff ``T`` (no leakage):

* pick earlier cutoffs ``c = T - 182 - 28 i`` days for ``i = 0..11``;
* at each ``c``, fit the candidate models on buys before ``c``, generate 50 candidates per
  customer and compute features as of ``c``;
* label a candidate 1 if the customer bought it for the first time in ``[c, c + 180d)``.

Every label window closes before ``T``, so nothing after ``T`` is used to train the model
that is evaluated at ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from recs.candidates.base import Recommender
from recs.candidates.hybrid import HybridCandidates
from recs.config import N_CANDIDATES, PROFIT_HORIZON_DAYS, SEED
from recs.data import Dataset, Snapshot, make_snapshot
from recs.ranking.features import FEATURES, attach_labels, build_features, item_features

# 12 training cutoffs, four weeks apart, the latest 26 weeks before deployment.
TRAIN_OFFSETS_DAYS = tuple(182 + 28 * i for i in range(12))

DEFAULT_LGB = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_child_samples": 1000,
    "feature_fraction": 0.3,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 30.0,
    "lambdarank_truncation_level": 20,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 4,
}


@dataclass
class FrameBuilder:
    """Builds (and caches) labelled candidate frames per cutoff."""

    ds: Dataset
    als_params: dict | None = None
    n_candidates: int = N_CANDIDATES
    horizon_days: int = PROFIT_HORIZON_DAYS
    _cache: dict = field(default_factory=dict, repr=False)

    def frame(
        self, cutoff: pd.Timestamp, end: pd.Timestamp | None = None, users: list[str] | None = None
    ) -> tuple[pd.DataFrame, Snapshot]:
        cutoff = pd.Timestamp(cutoff)
        end = None if end is None else pd.Timestamp(end)
        key = (cutoff, end, None if users is None else tuple(users))
        if key not in self._cache:
            snap = make_snapshot(self.ds, cutoff, self.horizon_days, end=end)
            if not snap.users:
                self._cache[key] = (pd.DataFrame(columns=["customer_id", "isin", "label"]), snap)
                return self._cache[key]
            gen = HybridCandidates(self.als_params).fit(snap)
            us = snap.users if users is None else users
            cands = gen.candidates(us, snap.eligible_items, n=self.n_candidates)
            items = item_features(self.ds, cutoff)
            df = attach_labels(build_features(snap, cands, items), snap)
            self._cache[key] = (df, snap)
        return self._cache[key]

    def training_frame(
        self, deploy_cutoff: pd.Timestamp, offsets: tuple[int, ...] = TRAIN_OFFSETS_DAYS
    ) -> pd.DataFrame:
        frames = []
        for off in offsets:
            c = pd.Timestamp(deploy_cutoff) - pd.Timedelta(days=off)
            df, _ = self.frame(c)
            if len(df):
                frames.append(df.assign(train_cutoff=c))
        return pd.concat(frames, ignore_index=True)


def _groups(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort by query (cutoff, customer) and keep only queries with at least one positive."""
    qcols = [c for c in ("train_cutoff", "customer_id") if c in df.columns]
    df = df.sort_values(qcols, kind="stable")
    pos = df.groupby(qcols, sort=False)["label"].transform("max")
    df = df[pos > 0]
    sizes = df.groupby(qcols, sort=False).size().to_numpy()
    return df, sizes


class LambdaRanker:
    def __init__(self, params: dict | None = None, num_boost_round: int = 400):
        self.params = {**DEFAULT_LGB, **(params or {})}
        self.num_boost_round = num_boost_round
        self.best_iteration: int | None = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> LambdaRanker:
        tr, g_tr = _groups(train)
        dtrain = lgb.Dataset(tr[FEATURES], tr["label"], group=g_tr, free_raw_data=False)
        sets, callbacks = [dtrain], []
        if valid is not None:
            va, g_va = _groups(valid)
            dvalid = lgb.Dataset(va[FEATURES], va["label"], group=g_va, reference=dtrain)
            sets.append(dvalid)
            callbacks.append(lgb.early_stopping(30, verbose=False))
        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=sets,
            callbacks=callbacks,
        )
        self.best_iteration = self.booster.best_iteration or self.num_boost_round
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df[FEATURES], num_iteration=self.best_iteration)

    def feature_importance(self) -> pd.Series:
        imp = self.booster.feature_importance(importance_type="gain")
        return pd.Series(imp, index=FEATURES).sort_values(ascending=False)


def rank_frame(df: pd.DataFrame, scores: np.ndarray, k: int) -> dict[str, list[str]]:
    d = df[["customer_id", "isin"]].assign(score=scores)
    d = d.sort_values(["customer_id", "score"], ascending=[True, False], kind="stable")
    return (
        d.groupby("customer_id", sort=False)["isin"].apply(lambda s: s.head(k).tolist()).to_dict()
    )


class TwoStageRecommender(Recommender):
    """Wraps a trained ranker so it can be evaluated like any other recommender.

    ``score`` is only defined on candidates: non-candidates get ``-inf``.
    """

    name = "two_stage_ranker"

    def __init__(self, ranker: LambdaRanker, builder: FrameBuilder):
        self.ranker = ranker
        self.builder = builder

    def fit(self, snap: Snapshot) -> TwoStageRecommender:
        self._df, self._snap = self.builder.frame(snap.cutoff, end=snap.end)
        self._df = self._df.assign(score=self.ranker.predict(self._df))
        return self

    def score(self, users: list[str], items: list[str]) -> np.ndarray:
        piv = self._df.pivot_table(index="customer_id", columns="isin", values="score")
        piv = piv.reindex(index=users, columns=items)
        return piv.fillna(-np.inf).to_numpy()

    def ranked(self, k: int = 10) -> dict[str, list[str]]:
        return rank_frame(self._df, self._df["score"].to_numpy(), k)
