"""Project-wide paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("RECS_DB_PATH", DATA_DIR / "warehouse.duckdb"))
ARTIFACTS_DIR = Path(os.environ.get("RECS_ARTIFACTS_DIR", ROOT / "artifacts"))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

# Temporal split. FAR-Trans transactions run from 2018-01 to 2022-11.
# Train on everything before VAL_START, tune on [VAL_START, TEST_START),
# report on [TEST_START, TEST_END). Each window is six months.
VAL_START = "2021-11-30"
TEST_START = "2022-05-30"
TEST_END = "2022-11-30"

# Horizon (trading days of calendar time) for profitability@k.
PROFIT_HORIZON_DAYS = 180

K_DEFAULT = 10
N_CANDIDATES = 50
SEED = 42
