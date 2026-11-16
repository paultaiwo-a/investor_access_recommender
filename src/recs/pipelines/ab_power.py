"""Inputs and power calculations for the online test plan (docs/ab_test_plan.md).

Needs the optional ``experiment`` extra (``uv sync --extra experiment``), which installs
``abkit`` from the deposit-nudge-experiment repo plus statsmodels.

Offline proxy for the primary metric: among customers active in the 180 days before the
cutoff, the share who make a first-time buy of at least one of their 10 suggested assets in
the next 30 days. Both arms get the same suitability layer. Without exposure this is an
organic rate, so it is a floor for the online rate, but it gives the variance and the
relative gap between arms.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from abkit import srm_check, two_prop_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from recs.candidates.base import recommend
from recs.candidates.baselines import PopularityRecommender
from recs.config import ARTIFACTS_DIR, N_CANDIDATES, TEST_START
from recs.data import Dataset, make_snapshot
from recs.pipelines.common import RESULTS_DIR
from recs.pipelines.train_ranker import load_mf_params
from recs.ranking.features import item_features
from recs.ranking.ranker import FrameBuilder, rank_frame
from recs.rules.suitability import SuitabilityRules

ALPHA = 0.05
POWER = 0.8
WINDOW_DAYS = 30
ACTIVE_DAYS = 180


def hit_rate(recs: dict[str, list[str]], targets: dict[str, set[str]], users: list[str]) -> float:
    return float(np.mean([bool(set(recs.get(u, [])) & targets.get(u, set())) for u in users]))


def monthly_active(ds: Dataset, start: str, end: str) -> float:
    buys = ds.buys
    months = pd.date_range(start, end, freq="MS")
    counts = [
        buys[
            (buys["transaction_date"] >= m)
            & (buys["transaction_date"] < m + pd.DateOffset(months=1))
        ]["customer_id"].nunique()
        for m in months
    ]
    return float(np.mean(counts))


def main() -> None:
    ds = Dataset.load()
    cutoff = pd.Timestamp(TEST_START)
    end = cutoff + pd.Timedelta(days=WINDOW_DAYS)
    snap = make_snapshot(ds, cutoff, end=end, require_targets=False)
    recent = snap.history[snap.history["last_date"] >= cutoff - pd.Timedelta(days=ACTIVE_DAYS)]
    active = sorted(set(recent["customer_id"]))

    rules = SuitabilityRules()
    meta = item_features(ds, cutoff)[["srri", "asset_group"]]
    risk = ds.customer_profile(cutoff)["risk_level"].reindex(active).fillna("Not_Available")
    risk = risk.to_dict()

    pop = recommend(
        PopularityRecommender().fit(snap), snap, k=len(snap.eligible_items), users=active
    )
    pop = {u: o.items for u, o in rules.apply_all(pop, risk, meta).items()}

    als, _ = load_mf_params()
    with open(ARTIFACTS_DIR / "ranker_test.pkl", "rb") as f:
        ranker = pickle.load(f)
    df, _ = FrameBuilder(ds, als_params=als).frame(
        cutoff, end=end, users=active, suitable_only=True, all_users=True
    )
    deep = rank_frame(df, ranker.predict(df), N_CANDIDATES)
    rk = {u: o.items for u, o in rules.apply_all(deep, risk, meta).items()}

    p0 = hit_rate(pop, snap.targets, active)
    p1 = hit_rate(rk, snap.targets, active)
    n_arm = len(active) // 2

    # Offline read of the two arms on the same users, using abkit's z-test (paired in reality,
    # so this CI is conservative).
    offline = two_prop_ztest(
        round(p1 * len(active)), len(active), round(p0 * len(active)), len(active)
    )

    solver = NormalIndPower()
    out = {
        "cutoff": str(cutoff.date()),
        "window_days": WINDOW_DAYS,
        "active_customers": len(active),
        "monthly_active_buyers_2021_2022": monthly_active(ds, "2021-01-01", "2022-10-01"),
        "control_rate_popularity": p0,
        "treatment_rate_ranker_offline": p1,
        "offline_relative_lift": p1 / p0 - 1,
        "offline_ztest": offline._asdict(),
        "mde": {},
        "n_per_arm_for_relative_lift": {},
    }
    for n in (n_arm, 5_000, 10_000, 25_000, 50_000):
        h = solver.solve_power(alpha=ALPHA, power=POWER, nobs1=n, ratio=1)
        mde = np.sin(np.arcsin(np.sqrt(p0)) + h / 2) ** 2 - p0
        out["mde"][str(n)] = {"abs_pp": 100 * mde, "relative": mde / p0}
    for lift in (0.05, 0.10, 0.15, 0.20, 0.30):
        es = proportion_effectsize(p0 * (1 + lift), p0)
        out["n_per_arm_for_relative_lift"][f"{int(lift * 100)}%"] = int(
            np.ceil(solver.solve_power(effect_size=es, alpha=ALPHA, power=POWER, ratio=1))
        )
    # Sanity: SRM check on a 50/50 split of the active base, as the readout would run it.
    out["srm_p_value_example"] = srm_check([n_arm, len(active) - n_arm])

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ab_power.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
