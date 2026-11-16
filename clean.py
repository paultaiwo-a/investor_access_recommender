# Investor access recommender
# Paul Taiwo-Adeyemo, November 2026
#
# Load the transactions, drop duplicate ids, keep the buys, and save a few charts.

import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("tableau-colorblind10")

# download the csvs if they are not already here
raw = Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)

if not (raw / "transactions.csv").exists():
    url = "https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if name.endswith(".csv"):
                (raw / name).write_bytes(zf.read(member))

transactions = pd.read_csv(raw / "transactions.csv")
print(transactions.head())
print("rows, columns:", transactions.shape)
print("missing values, percent:")
print(transactions.isna().sum() / len(transactions) * 100)

# 28,920 transaction ids show up more than once. Keep the earlier date.
duplicates = transactions["transactionID"].duplicated().sum()
print("duplicate transaction ids:", duplicates)

transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
transactions = transactions.sort_values("timestamp")
transactions = transactions.drop_duplicates("transactionID", keep="first")
print("duplicate transaction ids after:", transactions["transactionID"].duplicated().sum())

# sells are not used
print(transactions["transactionType"].value_counts())

figures = Path("figures")
figures.mkdir(exist_ok=True)

type_counts = transactions["transactionType"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(type_counts.index, type_counts.values, color="#58508d")
plt.title("Transaction type")
plt.ylabel("Number of rows")
plt.tight_layout()
plt.savefig(figures / "transaction_type.png")
plt.close()

buys = transactions[transactions["transactionType"] == "Buy"].copy()
print("buys:", buys.shape[0])
print("customers:", buys["customerID"].nunique())
print("assets:", buys["ISIN"].nunique())

buys_by_year = buys["timestamp"].dt.year.value_counts().sort_index()
print(buys_by_year)

plt.figure(figsize=(8, 4))
plt.bar(buys_by_year.index.astype(str), buys_by_year.values, color="#58508d")
plt.title("Buys by year")
plt.xlabel("Year")
plt.ylabel("Number of buys")
plt.tight_layout()
plt.savefig(figures / "buys_by_year.png")
plt.close()

buyers = buys.groupby("ISIN")["customerID"].nunique().sort_values(ascending=False)
print(buyers.head(15))

top = buyers.head(15).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.values, color="#58508d")
plt.title("Assets bought by the most customers")
plt.xlabel("Number of customers")
plt.tight_layout()
plt.savefig(figures / "top_assets.png")
plt.close()

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
buys[["customerID", "ISIN", "timestamp"]].to_csv(out / "buys.csv", index=False)
print("wrote", out / "buys.csv")
