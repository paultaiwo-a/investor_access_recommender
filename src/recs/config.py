"""Paths and the date split."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("RECS_DB_PATH", DATA_DIR / "warehouse.duckdb"))

# Same dates as sql/prepare.sql.
VAL_START = "2021-11-30"
TEST_START = "2022-05-30"
TEST_END = "2022-11-30"

# How far ahead a price has to exist for an asset to be suggestable.
PROFIT_HORIZON_DAYS = 180
K_DEFAULT = 10
