import pandas as pd
import pytest

from recs.rules.risk import srri_from_vol
from recs.rules.suitability import SuitabilityRules, violations

META = pd.DataFrame(
    {
        "srri": [2, 3, 6, 7, 6, 6, 6, 6, 2, 2, 2, 2],
        "asset_group": [
            "Bonds",
            "Bonds",
            "Energy",
            "Energy",
            "Energy",
            "Energy",
            "Energy",
            "Energy",
            "MM",
            "MM",
            "MM",
            "MM",
        ],
    },
    index=[f"i{n}" for n in range(12)],
)
SRRI, GROUP = META["srri"].to_dict(), META["asset_group"].to_dict()


def test_srri_bands():
    assert srri_from_vol([0.001, 0.01, 0.03, 0.07, 0.12, 0.2, 0.4]).tolist() == list(range(1, 8))


def test_risk_filter_removes_unsuitable_assets():
    rules = SuitabilityRules(k=3, enforce_sector_cap=False)
    out = rules.apply(["i2", "i0", "i3", "i1", "i8"], "Conservative", SRRI, GROUP)
    assert out.items == ["i0", "i1", "i8"]
    assert ("i2", "risk") in out.removed and ("i3", "risk") in out.removed


def test_aggressive_profile_keeps_high_risk():
    rules = SuitabilityRules(k=2, enforce_sector_cap=False)
    assert rules.apply(["i3", "i2"], "Aggressive", SRRI, GROUP).items == ["i3", "i2"]


def test_unknown_profile_treated_as_conservative():
    rules = SuitabilityRules(k=2, enforce_sector_cap=False)
    assert rules.apply(["i2", "i0"], None, SRRI, GROUP).items == ["i0"]


def test_sector_cap_limits_to_40_percent_and_backfills():
    rules = SuitabilityRules(k=10, enforce_risk=False)
    ranked = [f"i{n}" for n in (2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10, 11)]
    out = rules.apply(ranked, "Aggressive", SRRI, GROUP)
    groups = [GROUP[i] for i in out.items]
    assert groups.count("Energy") == 4
    assert len(out.items) == 10
    assert sum(r == "sector_cap" for _, r in out.removed) == 2


@pytest.mark.parametrize("k,cap,expected", [(10, 0.4, 4), (5, 0.4, 2), (3, 0.2, 1)])
def test_max_per_group(k, cap, expected):
    assert SuitabilityRules(k=k, sector_cap=cap).max_per_group == expected


def test_violations_report():
    recs = {"u1": ["i2", "i3", "i4", "i5", "i6"], "u2": ["i0", "i1"]}
    v = violations(recs, {"u1": "Conservative", "u2": "Conservative"}, META, SuitabilityRules(k=5))
    assert v["lists_with_unsuitable_risk"] == 0.5
    assert v["lists_breaking_sector_cap"] == 0.5
    assert v["unsuitable_item_share"] == pytest.approx(5 / 7)


def test_flat_priced_stock_is_not_low_risk():
    from recs.rules.risk import asset_srri

    assets = pd.DataFrame(
        {
            "isin": ["S", "M"],
            "asset_category": ["Stock", "MTF"],
            "asset_subcategory": ["Unspecified", "Money Market"],
        }
    )
    out = asset_srri(pd.Series({"S": 0.0, "M": 0.001}), assets)
    assert out["S"] >= 5
    assert out["M"] == 1
