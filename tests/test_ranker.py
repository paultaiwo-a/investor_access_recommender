import pandas as pd

from recs.ranking.ranker import FrameBuilder, LambdaRanker, _groups, rank_frame


def test_training_label_windows_close_before_deployment(ds):
    deploy = pd.Timestamp("2021-07-01")
    fb = FrameBuilder(ds, als_params={"factors": 4, "iterations": 3}, n_candidates=5)
    train = fb.training_frame(deploy, offsets=(182,))
    assert len(train)
    for c in train["train_cutoff"].unique():
        _, snap = fb.frame(c)
        assert snap.end <= deploy


def test_groups_drop_queries_without_positives():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "a", "b", "b", "c"],
            "isin": ["x", "y", "x", "y", "x"],
            "label": [1, 0, 0, 0, 1],
        }
    )
    kept, sizes = _groups(df)
    assert set(kept["customer_id"]) == {"a", "c"}
    assert sizes.tolist() == [2, 1]


def test_ranker_trains_and_ranks(ds):
    fb = FrameBuilder(ds, als_params={"factors": 4, "iterations": 3}, n_candidates=5)
    df, _ = fb.frame("2021-01-01")
    ranker = LambdaRanker({"min_child_samples": 1, "min_data_in_bin": 1}, num_boost_round=5)
    ranker.fit(df)
    recs = rank_frame(df, ranker.predict(df), k=3)
    assert set(recs) == set(df["customer_id"])
    assert all(len(v) <= 3 for v in recs.values())
