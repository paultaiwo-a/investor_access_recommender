from pathlib import Path

import duckdb

from recs.warehouse import build


def _write(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text.strip() + "\n")


def test_prepare_sql_cleans_and_splits(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(
        raw,
        "transactions.csv",
        """
        transactionID,customerID,ISIN,transactionType,timestamp,totalValue,units
        1,u1,A,Buy,2021-01-01,10,1
        1,u1,A,Buy,2021-01-02,99,9
        2,u1,B,Sell,2021-06-01,5,1
        3,u2,A,buy,2022-06-01,8,1
        """,
    )
    _write(
        raw,
        "asset_information.csv",
        """
        ISIN,assetName,assetCategory,sector,timestamp
        A,Alpha,Stock,Energy,2020-01-01
        A,Alpha later,Stock,Energy,2021-01-01
        """,
    )
    _write(
        raw,
        "close_prices.csv",
        """
        ISIN,timestamp,closePrice
        A,2021-01-01,10
        A,2021-01-01,12
        A,2021-01-02,0
        """,
    )

    db = tmp_path / "warehouse.duckdb"
    build(raw, db)

    with duckdb.connect(str(db)) as con:
        tx = con.sql("select * from transactions order by transaction_id").df()
        buys = con.sql("select * from buys order by transaction_date").df()
        assets = con.sql("select * from assets").df()
        prices = con.sql("select * from prices").df()

    assert list(tx["transaction_id"]) == [1, 2, 3]
    assert tx.loc[tx["transaction_id"] == 1, "total_value"].item() == 10
    assert list(buys["split"]) == ["train", "test"]
    assert assets.loc[0, "asset_name"] == "Alpha later"
    assert len(prices) == 1
    assert prices.loc[0, "close_price"] == 10
