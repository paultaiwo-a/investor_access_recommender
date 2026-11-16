# Investor access recommender

Project walkthrough

This suggests assets a customer has not bought yet. The data is FAR-Trans, retail investment transactions from a Greek bank, 2018 to 2022.

## Cleaning

`sql/prepare.sql` loads the csv files into DuckDB.

It keeps one row per transaction, drops prices that are missing or zero, and keeps the latest description of each asset. Buys are labeled from the date:

- train: before 30 Nov 2021
- validation: 30 Nov 2021 up to 30 May 2022
- test: 30 May 2022 up to 30 Nov 2022

## Models

Both models only see buys from before 30 May 2022.

Popularity. Suggest the assets that the most customers bought.

Item-item. If you bought an asset, suggest other assets that the same customers tended to buy.

Anything the customer already bought is left off the list. An asset is only suggested if it has a price at the cutoff and at the end of the test window.

## Scores

Recall@10 and nDCG@10. These are averaged over customers who bought something new in the test window. A repeat buy of an asset they already held does not count.

`avg_return` is the average price change of the suggested assets over the test window. The models are not fit to maximize that. It is just a check.

## How to run

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
make setup
make data
make evaluate
```

`make test` uses a small made-up dataset. It does not need the download.

## Data

FAR-Trans is from the University of Glasgow under CC-BY 4.0.
<https://doi.org/10.5525/gla.researchdata.1658>

Javier Sanz-Cruzado, Nikolaos Droukas, Richard McCreadie. FAR-Trans: An Investment Dataset for Financial Asset Recommendation. IJCAI 2024 workshop on recommender systems in finance.
