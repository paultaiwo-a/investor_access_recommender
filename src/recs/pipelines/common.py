"""Shared helpers for pipelines: snapshots for the fixed split, result tables, MLflow."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from recs.config import MLFLOW_URI, ROOT, TEST_END, TEST_START, VAL_START
from recs.data import Dataset, Snapshot, make_snapshot

RESULTS_DIR = ROOT / "results"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def val_snapshot(ds: Dataset) -> Snapshot:
    """Fit on buys before VAL_START, evaluate on the validation window."""
    return make_snapshot(ds, VAL_START, end=TEST_START)


def test_snapshot(ds: Dataset) -> Snapshot:
    """Fit on buys before TEST_START (train + val), evaluate on the test window."""
    return make_snapshot(ds, TEST_START, end=TEST_END)


def save_table(df: pd.DataFrame, name: str, float_fmt: str = "{:.4f}") -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv = RESULTS_DIR / f"{name}.csv"
    df.to_csv(csv, index=False)
    md = RESULTS_DIR / f"{name}.md"
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    md.write_text(fmt.to_markdown(index=False) + "\n")
    return csv


@contextlib.contextmanager
def mlflow_run(experiment: str, run_name: str, params: dict | None = None) -> Iterator[object]:
    """Start an MLflow run, or a no-op if MLflow is disabled (RECS_NO_MLFLOW=1)."""
    if os.environ.get("RECS_NO_MLFLOW") == "1":
        yield None
        return
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    if os.environ.get("RECS_NO_MLFLOW") == "1":
        return
    import math

    import mlflow

    clean = {
        f"{prefix}{k}".replace("@", "_at_"): float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not math.isnan(float(v))
    }
    mlflow.log_metrics(clean)
