"""Fit the two models on the test window and print the scores."""

from __future__ import annotations

import pandas as pd

from recs.config import K_DEFAULT, RAW_DIR, ROOT, TEST_END, TEST_START
from recs.data import Dataset, make_snapshot
from recs.metrics import average_return, ranking_metrics
from recs.recommend import item_item, popularity
from recs.warehouse import build


def main() -> int:
    if not (RAW_DIR / "transactions.csv").exists():
        print(f"No csvs in {RAW_DIR}. Run make data first.")
        return 1

    build()
    snap = make_snapshot(Dataset.load(), TEST_START, end=TEST_END)
    rows = []
    for name, model in (("popularity", popularity), ("item_item", item_item)):
        recs = model(snap, k=K_DEFAULT)
        row = {"model": name, "n_users": len(snap.users)}
        row.update(ranking_metrics(recs, snap.targets, K_DEFAULT))
        row["avg_return"] = average_return(recs, snap.forward_returns, K_DEFAULT)
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out = ROOT / "results" / "scores.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
