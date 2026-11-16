"""Run sql/prepare.sql against the raw csvs."""

from __future__ import annotations

from pathlib import Path

import duckdb

from recs.config import DB_PATH, RAW_DIR, ROOT

SQL_PATH = ROOT / "sql" / "prepare.sql"


def build(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> Path:
    sql = SQL_PATH.read_text().replace("{{raw}}", raw_dir.as_posix())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(sql)
    return db_path


if __name__ == "__main__":
    print(build())
