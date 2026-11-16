"""Asset and customer risk scales.

Assets get an SRRI-style class from 1 (lowest) to 7 (highest) based on annualised volatility,
following the bands of the EU UCITS synthetic risk and reward indicator. Customers get the
highest class their MiFID risk profile allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Upper volatility bound of SRRI classes 1..6; class 7 is anything above 25%.
SRRI_BANDS = [0.005, 0.02, 0.05, 0.10, 0.15, 0.25]

# Fallback class when an asset has too little price history to measure volatility.
CATEGORY_DEFAULT_SRRI = {
    ("MTF", "Money Market"): 1,
    ("MTF", "Bond"): 3,
    ("MTF", "Bonds"): 3,
    ("MTF", "Structured"): 3,
    ("MTF", "Balanced"): 4,
    ("MTF", "Other"): 4,
    ("MTF", "Equity"): 6,
    ("Bond", "Government"): 3,
    ("Bond", "Corporate"): 4,
    ("Stock", None): 6,
}

RISK_LEVELS = ["Conservative", "Income", "Balanced", "Aggressive"]
RISK_ORDINAL = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}

# Highest SRRI class a profile may be offered. Profiles without a known level are treated as
# Conservative, the safe default for an unassessed retail client.
MAX_SRRI = {
    "Conservative": 3,
    "Income": 4,
    "Balanced": 6,
    "Aggressive": 7,
    "Not_Available": 3,
}


def srri_from_vol(vol: float | np.ndarray) -> np.ndarray:
    """Map annualised volatility to SRRI class 1..7 (NaN stays NaN)."""
    v = np.asarray(vol, dtype=float)
    cls = np.digitize(v, SRRI_BANDS, right=True) + 1.0
    return np.where(np.isnan(v), np.nan, cls)


def default_srri(category: str, subcategory: str | None) -> int:
    if category == "Stock":
        return CATEGORY_DEFAULT_SRRI[("Stock", None)]
    return CATEGORY_DEFAULT_SRRI.get((category, subcategory), 5)


# Lowest class an asset type can get. A single stock with a near-flat price is usually
# suspended, not safe, so measured volatility alone must not make it look low risk.
CATEGORY_FLOOR_SRRI = {"Stock": 5}


def asset_srri(vol: pd.Series, assets: pd.DataFrame) -> pd.Series:
    """SRRI per asset: measured from volatility, falling back to the category default and
    never below the category floor."""
    a = assets.set_index("isin")
    measured = pd.Series(srri_from_vol(vol.reindex(a.index).to_numpy()), index=a.index)
    fallback = pd.Series(
        [
            default_srri(c, s)
            for c, s in zip(a["asset_category"], a["asset_subcategory"], strict=True)
        ],
        index=a.index,
        dtype=float,
    )
    floor = a["asset_category"].map(CATEGORY_FLOOR_SRRI).fillna(1)
    return measured.fillna(fallback).clip(lower=floor).astype(int)


def max_srri(risk_level: str | None) -> int:
    if risk_level is None or (isinstance(risk_level, float) and np.isnan(risk_level)):
        return MAX_SRRI["Not_Available"]
    return MAX_SRRI.get(str(risk_level), MAX_SRRI["Not_Available"])
