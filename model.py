# Investor access recommender
# Paul Taiwo-Adeyemo, November 2026
#
# Suggest assets a customer has not bought yet.
# The models only see buys from before 30 May 2022.
# The test window is 30 May 2022 up to 30 Nov 2022.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("tableau-colorblind10")

buys_path = Path("data/processed/buys.csv")
if not buys_path.exists():
    raise SystemExit("No cleaned buys. Run clean.py first.")

buys = pd.read_csv(buys_path)
buys["timestamp"] = pd.to_datetime(buys["timestamp"])

cutoff = pd.Timestamp("2022-05-30")
end = pd.Timestamp("2022-11-30")

history = buys[buys["timestamp"] < cutoff]
window = buys[(buys["timestamp"] >= cutoff) & (buys["timestamp"] < end)]
print("history rows, test rows:", history.shape[0], window.shape[0])

# a hit is the first time a customer buys an asset in the test window
seen = set(zip(history["customerID"], history["ISIN"]))
targets = {}
for customer, isin in zip(window["customerID"], window["ISIN"]):
    if (customer, isin) not in seen:
        targets.setdefault(customer, set()).add(isin)

known = set(history["customerID"])
customers = sorted(customer for customer in targets if customer in known)
targets = {customer: targets[customer] for customer in customers}
held = history.groupby("customerID")["ISIN"].agg(set).to_dict()
print("customers scored:", len(customers))


def top10(scores, customer):
    already = held.get(customer, set())
    kept = scores.drop(labels=[isin for isin in already if isin in scores.index])
    return list(kept.sort_values(ascending=False).head(10).index)


# popularity: assets the most customers bought before the cutoff
popularity_scores = history.groupby("ISIN")["customerID"].nunique().astype(float)
print(popularity_scores.sort_values(ascending=False).head(10))
popularity_lists = {customer: top10(popularity_scores, customer) for customer in customers}

# item-item: assets bought by the same customers
pairs = history[["customerID", "ISIN"]].drop_duplicates()
pairs["bought"] = 1.0
user_item = pairs.pivot_table(index="customerID", columns="ISIN", values="bought", aggfunc="max")
user_item = user_item.fillna(0.0)
print("customer-asset matrix:", user_item.shape)

matrix = user_item.to_numpy()
norms = np.linalg.norm(matrix, axis=0)
norms[norms == 0] = 1
with np.errstate(divide="ignore", invalid="ignore"):
    similarity = (matrix.T @ matrix) / np.outer(norms, norms)
similarity = np.nan_to_num(similarity)
np.fill_diagonal(similarity, 0)

asset_names = list(user_item.columns)
held_matrix = user_item.reindex(customers).fillna(0.0)
item_scores = pd.DataFrame(
    held_matrix.to_numpy() @ similarity,
    index=customers,
    columns=asset_names,
)
item_lists = {customer: top10(item_scores.loc[customer], customer) for customer in customers}


def recall_at_10(recommended, relevant):
    return len(set(recommended[:10]) & relevant) / len(relevant)


def ndcg_at_10(recommended, relevant):
    gained = 0
    for rank, isin in enumerate(recommended[:10]):
        if isin in relevant:
            gained += 1 / np.log2(rank + 2)
    ideal = sum(1 / np.log2(rank + 2) for rank in range(min(10, len(relevant))))
    return gained / ideal


def average_scores(lists):
    recall = np.mean([recall_at_10(lists[customer], targets[customer]) for customer in customers])
    ndcg = np.mean([ndcg_at_10(lists[customer], targets[customer]) for customer in customers])
    return recall, ndcg


popularity_recall, popularity_ndcg = average_scores(popularity_lists)
item_recall, item_ndcg = average_scores(item_lists)

print(f"Popularity  recall@10 {popularity_recall:.3f}   nDCG@10 {popularity_ndcg:.3f}")
print(f"Item-item   recall@10 {item_recall:.3f}   nDCG@10 {item_ndcg:.3f}")

figures = Path("figures")
figures.mkdir(exist_ok=True)
labels = ["popularity", "item-item"]
x = np.arange(len(labels))

plt.figure(figsize=(7, 4))
plt.bar(x - 0.15, [popularity_recall, item_recall], width=0.3, label="recall@10", color="#58508d")
plt.bar(x + 0.15, [popularity_ndcg, item_ndcg], width=0.3, label="nDCG@10", color="#bc5090")
plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Test window, 30 May 2022 to 30 Nov 2022")
plt.legend()
plt.tight_layout()
plt.savefig(figures / "scores.png")
plt.close()

example = customers[0]
print(example)
print("popularity", popularity_lists[example])
print("item-item", item_lists[example])
print("bought", sorted(targets[example]))
