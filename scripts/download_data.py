"""Download and unpack the FAR-Trans dataset into data/raw.

FAR-Trans (Sanz-Cruzado, Droukas, McCreadie, 2024) is released under CC-BY 4.0 by the
University of Glasgow: https://doi.org/10.5525/gla.researchdata.1658
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = [
    "asset_information.csv",
    "close_prices.csv",
    "customer_information.csv",
    "limit_prices.csv",
    "markets.csv",
    "transactions.csv",
]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / f).exists() for f in EXPECTED):
        print(f"FAR-Trans already present in {RAW_DIR}")
        return 0
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if not name or member.is_dir():
                continue
            (RAW_DIR / name).write_bytes(zf.read(member))
    missing = [f for f in EXPECTED if not (RAW_DIR / f).exists()]
    if missing:
        print(f"Missing files after extraction: {missing}", file=sys.stderr)
        return 1
    print(f"Extracted FAR-Trans to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
